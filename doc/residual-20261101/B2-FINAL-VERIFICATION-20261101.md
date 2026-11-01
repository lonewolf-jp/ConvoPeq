# B-2 最終確認資料 — deferredDestroy_ の型・leak detach の安全性・ownerId 一意性・実行経路（2026-11-01）

- 状態: **調査・設計資料のみ。製品コード・テスト・CI 定義は未変更。build / test / debugger / replay 未実行**
- 基準: `main` = `origin/main` = `6bb7c8d5`（ソースは `fd92c6e7` と同一）
- 目的: Owner 指示 5 項目（deferredDestroy_ の型消去、leak detach の安全性、ownerId 一意性、
  prepare の実行スレッドとアクセス制御、診断バッファの参照経路）に回答する

## 0. 結論

| # | 確認項目 | 結果 |
|---|---|---|
| 1 | `deferredDestroy_` の型消去 | **DSPCore 専用**（唯一の書き手が DSPCore 破棄関数）。型付き vector に変更して `void*` キャストを廃止できる |
| 2 | leak コアの detach 安全性 | **安全**。`activeReaderCount` は RCU 読み取り中のスレッド数であり、診断生産者（RT）とは別。RT は join 済み |
| 3 | ownerId の一意性 | プロセス全体の atomic カウンタで採番。0 は予約、再利用なし、オーバーフローは fail-closed |
| 4 | prepare の実行スレッドと排他 | rebuild スレッド（1 箇所）と message スレッド（1 箇所）。teardown は rebuild join 後 |
| 5 | 診断バッファの参照経路 | すべて**同一エンジンの同一 sink** を指す（エンジン直下は `diagSink_`、DSPCore は bind 済み sink） |

## 1. `deferredDestroy_` の型消去（Owner 指示 1）

### 1.1 実測: この vector の唯一の書き手

```cpp
// AudioEngine.Threading.cpp:70-86
void AudioEngine::deferDestroyRetry(void* p) noexcept {
    ...
    for (void* existing : deferredDestroy_) if (existing == p) return;   // 重複登録を防ぐ
    deferredDestroy_.push_back(p);
    ...
}
```

`deferDestroyRetry` の呼び出し元は **1 箇所のみ**:

```cpp
// AudioEngine.Threading.cpp:42-51（destroyDSPCoreNodeRetryable 内）
if (!canFree) {                       // pin 残留
    owner->deferDestroyRetry(p);      // ← ここだけ
    return DestroyOutcome::Deferred;
}
```

そして `destroyDSPCoreNodeRetryable` は DSPCore 専用の破棄関数である（`:56-58` で
`static_cast<DSPCore*>(p); core->~DSPCore(); aligned_free(core);`）。

**結論: `deferredDestroy_` に入るのは DSPCore ノードだけ**（型は呼び出し経路で一意）。

### 1.2 混同しやすい別機構（型が混在するのはこちら）

`enqueueDeferredDeleteNonRt(void* ptr, void (*deleter)(void*), ...)` は**別のストア**（retire router）を使い、
`CacheMap`（`AudioEngine.Cache.cpp:16/40`）や `RuntimePublishWorld`（`AudioEngine.h:3901/3917`）など
**複数型**を格納する。**`deferredDestroy_` とは無関係**。

| 機構 | 格納型 | 用途 |
|---|---|---|
| `deferredDestroy_`（`AudioEngine.h:5254`） | **DSPCore のみ** | pin 残留コアの再試行リスト |
| `enqueueDeferredDeleteNonRt` のストア | 複数型（deleter 付き） | 汎用の遅延破棄 |

### 1.3 重複・既破棄・teardown 中の変異

| 懸念 | 実測 |
|---|---|
| 重複登録 | 防止済み（`Threading.cpp:75-79` で同一ポインタをスキップ） |
| 既に破棄された要素 | 破棄成功時にリストから外れる（`retryDeferredDestroys` が所有権を swap して処理） |
| teardown 中の変異 | `retryDeferredDestroys` は **mutex 下で vector を swap**（`:102-109`）。以降の走査対象は呼び出し元が独占する |

### 1.4 型安全化（実装時の変更）

```cpp
// 変更前
std::vector<void*> deferredDestroy_;
// 変更後（型付き。キャスト不要）
std::vector<DSPCore*> deferredDestroy_;
```

`deferDestroyRetry(void* p)` の引数は `DSPCore*` に変えられない場合、
`destroyDSPCoreNodeRetryable` 内で `static_cast<DSPCore*>(p)` を**既に実施済み**なので、
その型付きポインタを渡す形にすればよい（**未検証の `void*` をキャストしない**）。

**teardown の detach は次の形になる**（`void*` の総なめをしない）:

```cpp
// ~AudioEngine（member teardown 前）
(void)retryDeferredDestroys();                 // 回収できるものは回収
{
    std::lock_guard<std::mutex> lock(deferredDestroyMutex_);
    for (DSPCore* core : deferredDestroy_)     // ★ 型付き
        core->detachDiagSink();                // sink を nullptr に
}
```

## 2. leak コアの detach 安全性（Owner 指示 2）

### 2.1 `activeReaderCount()` の意味（実測）

`src/core/EpochDomain.h:305-315`:

```cpp
uint32_t activeReaderCount() const noexcept override {
    uint32_t count = 0;
    for (const auto& slot : readers) {
        const uint32_t depth = convo::consumeAtomic(slot.depth, acquire);
        if (depth != 0) ++count;                 // ← critical section 滞在中のみ
    }
    return count;
}
```

**「登録済みスレッド数」ではなく「RCU 読み取り区間（depth > 0）に滞在中のスレッド数」**である。

### 2.2 reader 残留と RT 生産者残留の区別

| 観点 | reader（`activeReaderCount > 0`） | 診断生産者 |
|---|---|---|
| 何をするか | retire 済み **world/core** の参照（RCU 保護） | `DiagEvent` の push（RT のみ、14 箇所） |
| 診断 sink を触るか | **触らない**（`process()` を呼ばない） | 触る |
| 停止の保証 | 不要（reader は sink を使わない） | **join / lock で停止済み** |

**したがって「コアを leak する理由（reader の寿命を保証できない）」と
「診断 sink を detach する操作」は独立であり、detach が reader の安全性を損なわない。**

### 2.3 detach 時点で「sink を取得済みの生産者」がいないことの根拠

| # | 根拠 |
|---|---|
| 1 | 生産者は **RT（audio callback）のみ**（`B2-LIFETIME-PROOF` §4.1 の 14 箇所を関数単位で確認） |
| 2 | その RT は teardown の `StopAudio` で **join / `audioCallbackLock`** により停止済み（`ReleaseResources.cpp:153`、`AudioEngineHarness.cpp:59-66`、`juce_AudioDeviceManager.cpp:1065-1076`） |
| 3 | `activeReaderCount > 0` の reader は `process()` を呼ばない（§2.2） |
| 4 | したがって「load 済みの sink ポインタを持つスレッド」は存在しない |

### 2.4 実際の順序（teardown）

```
① StopAcceptingWork        (ReleaseResources.cpp:95)
② StopAudio                (:153)
③ StopWorkers              (:236-238)  ← Coordinator join → Builder join
④ DrainRetire              (:267) → drainAllQuarantineStore (:444) / :584
⑤ retryDeferredDestroys    (CtorDtor.cpp:346)
⑥ 【新規】leak コアの detach（deferredDestroy_ を型付きで走査）
⑦ member teardown（diagSink_ 破棄）
```

**⑥ は ③ の後**に置くため、RT・rebuild のどちらも実行していない。

## 3. ownerId の一意性（Owner 指示 3）

| 項目 | 設計 |
|---|---|
| 生成 | `AudioEngine` 内の **プロセス全体 `std::atomic<uint64_t>`** から `fetch_add(1, relaxed)` |
| 0 の予約 | **0 = 未束縛**。カウンタは **1 から**開始 |
| エンジン再生成時の再利用 | **なし**（単調増加。破棄済みエンジンの ID は再発行されない） |
| 同一 ID の重複 | 構造的に不可（`fetch_add` は一意な値を返す） |
| オーバーフロー | 実質到達不能（2^64）。到達時は **`assert` + 異常終了（fail-closed）**。0 に戻して再利用しない |
| プロセス外 | 対象外（プロセス内識別。永続化しない） |

**「ID はプロセス内で一意・単調・再利用なし」**を不変条件とする。

## 4. prepare の実行スレッドとアクセス制御（Owner 指示 4）

### 4.1 `DSPCore::prepare` の全呼び出し元（実測・2 箇所）

| # | 呼び出し元 | スレッド | owner 引数 |
|---|---|---|---|
| 1 | `RuntimeBuilder.cpp:474`（rebuild） | **rebuild スレッド** | `&engine` |
| 2 | `PrepareToPlay.cpp:272`（placeholder） | **message スレッド** | `this` |

テスト（`V11EngineRetireProtectionTests.cpp` 等）は `prepare` を**呼ばない**。

### 4.2 bind / detach の単一 NonRT 制御経路

| 操作 | 実行点 | 排他 |
|---|---|---|
| bind | `DSPCore::prepare`（上記 2 箇所） | rebuild は teardown の ③ で join 済み。message は teardown スレッド自身 |
| detach | ① コアのデストラクタ（drain 中）、② teardown の ⑥ | どちらも ③ の後 |

**「単に private」ではなく、③（join）と ⑦（member teardown）の間に置くことで排他を保証する。**

### 4.3 private な診断 sink へのアクセス方法

- `DSPCore` は `AudioEngine` の**入れ子クラス**であり、**外側の private にアクセスできる**
  （[class.access.nest]/1）。
- したがってロジックは `DSPCore::prepare` に置き、`owner->diagnosticsEngineId()` /
  `owner->diagnosticsSink()`（いずれも private）を呼ぶ。**`friend` 不要**。
- 外側（`AudioEngine`）から入れ子（`DSPCore`）の private へは**アクセスしない**設計にする
  （規格上不可のため）。

### 4.4 ownerId 設定後・sink 設定前の状態で RT が走らないこと

| 段階 | 事実 |
|---|---|
| `prepare` の実行 | **publish より前**（`RuntimeBuilder.cpp:474` は publish 前） |
| RT がコアを参照できる条件 | world への publish 完了後 |
| したがって | 中間状態（ownerId 設定・sink 未設定）は **RT から観測されない** |
| 仮に観測されても | `sink_ == nullptr` → push なし（T9） |

## 5. 診断バッファの参照経路（Owner 指示 5）

### 5.1 sink の受け渡し

| 生産者 | sink の取得元 | 変更 |
|---|---|---|
| `AudioBlock.cpp:262/295/707/750`（4 箇所） | **エンジンの `diagSink_`**（メンバ） | グローバルを使っていない。参照先を `diagSink_` に統一 |
| `BlockDouble.cpp:249/285/659/702`（4 箇所） | 同上 | 同上 |
| `DSPCoreFloat.cpp:86`（`logEqTime`） | **`DSPCore` が渡す sink** | 引数で受け取る（グローバル廃止） |
| `DSPCoreDouble.cpp:46`（`logEqTime`） | 同上 | 同上 |
| `DSPCoreDouble.cpp:650`（`processOutputDouble`） | 同上 | 同上 |
| `DSPCoreIO.cpp:438`（`processOutput`） | 同上 | 同上 |
| `ConvolverProcessor.Runtime.cpp:774/863` | **`process(block, sink)` の引数** | 引数で受け取る（グローバル廃止） |

### 5.2 「すべての push が同一の owner に対応する sink を使う」ことの確認

- エンジン直下（`AudioBlock` / `BlockDouble`）は `&engine.diagSink_`。
- DSPCore の bind は `owner->diagnosticsSink()` = **同じ `&engine.diagSink_`**。
- したがって **両者は同一オブジェクト**を指す（別エンジンの sink は構造的に渡らない）。
- 旧グローバル（`convDiagBuffer` / `eqDiagBuffer`）は**削除**する（残さない）。

### 5.3 変更対象の総数

| 種別 | 箇所数 |
|---|---|
| グローバル定義の削除 | 2 群（`ConvolverProcessor.Runtime.cpp:91-94`、`DSPCoreFloat.cpp:31-35`） |
| push サイトの書き換え | **14** |
| 生産者への sink 受け渡し（シグネチャ） | `process`（1）＋ `logEqTime`（2）＋ `processOutput`/`processOutputDouble`（2） |

## 6. 残る不確定（実装時に確定してよい）

1. `deferDestroyRetry` / `destroyDSPCoreNodeRetryable` の引数型を `DSPCore*` へ変える範囲
   （`void` deleter 経由の呼び出しとの整合）。
2. `DSPCore` の `sink_` を `std::atomic` にするか素のポインタにするか
   （publish の HB で足りるが、防御的に atomic(relaxed) を推奨）。
3. `sizeof` の実測値（`static_assert` は補助。ODR は単一定義で担保）。

## 7. この資料で言えること

- `deferredDestroy_` は **DSPCore 専用**であり、型付き vector にすれば `void*` キャストは不要。
- leak コアの detach は、**reader の安全性と独立**（reader は sink を使わない）かつ
  **RT 静止後**（join/lock）であるため安全。
- ownerId は**プロセス内で一意・再利用なし**。
- bind は rebuild/message の 2 箇所のみで、teardown は join 後に走る。
- すべての診断 push が**同一エンジンの同一 sink** を指す。

**上記により、Owner 指示の 5 項目は設計として閉じた。実装は HOLD を維持する。**
