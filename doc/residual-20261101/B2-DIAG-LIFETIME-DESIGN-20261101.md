# B-2 設計案 — 診断バッファの所有権と寿命保証（2026-11-01）

- 状態: **設計案のみ。製品コードは変更していない**（Owner 承認待ち）
- 対象: `convDiagBuffer`（Convolver）と `eqDiagBuffer`（EQ）の**両方**
- 基準: `main` = `fd92c6e7`
- 前提: 原因は §1 のとおり特定済み（`doc/residual-20261101/B-SEGFAULT-INVESTIGATION-20261101.md` §7）

## 1. 原因（再掲・設計の出発点）

クラッシュは `ConvolverProcessor::process()` の
`convDiagBuffer->push(event)`（`ConvolverProcessor.Runtime.cpp:774`）への**書き込み**で発生した。

```
エンジン A 生成 → setConvDiagBuffer(&A.diagBuffer)   （プロセス全体のグローバル）
エンジン A 破棄 → A.diagBuffer も破棄                （グローバルは A を指したまま）
エンジン B 生成 → setConvDiagBuffer(&B.diagBuffer)   （それまでの窓では A を指す）
   ↑ この窓で RT スレッドが process() を実行 → 解放済みバッファへ push → アクセス違反
```

**本質**: グローバルポインタを消すことではなく、**RT スレッドからの参照が終了する前に
診断バッファが破棄されないこと**を保証することにある（Owner 指摘）。

## 2. 必須要件（Owner 条件の写像）

| # | 要件 | 設計での担保 |
|---|---|---|
| R1 | `convDiagBuffer` と `eqDiagBuffer` の**両方**を対象 | §4 で両方を同一機構へ移行 |
| R2 | グローバル生ポインタを廃止し、**所有権と参照先の寿命を一致**させる | §4.1 所有グラフ、§4.3 寿命規則 |
| R3 | エンジン破棄時に **RT スレッドがバッファを参照していない**ことを保証 | §4.4 停止順序と静止確認 |
| R4 | 単純なリセット / atomic 化で解決したと**判断しない** | §4.3 は「参照の終了」を先に確定させる。§7 に代替案の限界を明記 |
| R5 | 複数エンジンの同時存在、初期化・破棄の繰り返し、診断無効ビルドを確認 | §5 の 3 ケースと §7 の検証計画 |
| R6 | RT スレッド上で**ロック・確保・解放をしない** | §4.5（atomic load 1 回のみ） |

## 3. 現状の構造（実測）

| 要素 | 場所 | 性質 |
|---|---|---|
| `diagBuffer` | `AudioEngine.h:3030` | **エンジンのメンバ**（エンジンとともに破棄） |
| `rtAuxMutable_.diagTickPushed` 等 | `AudioEngine.h`（rtAuxMutable_） | **エンジンのメンバ** |
| `convDiagBuffer` ほか 3 ポインタ | `ConvolverProcessor.Runtime.cpp:91-94` | **プロセス全体のグローバル** |
| `eqDiagBuffer` ほか 3 ポインタ | `AudioEngine.Processing.DSPCoreFloat.cpp:31-35` | **プロセス全体のグローバル** |
| 設定 | `AudioEngine.Init.cpp:116-119` | エンジン初期化時に毎回上書き |
| 解除 | **存在しない** | 破棄時にクリアする経路が無い |
| 生産者（RT） | `ConvolverProcessor::process`（`:774`, `:863`）、`DSPCoreFloat::logEqTime`（`:86`）、`DSPCoreDouble::logEqTime`（`:39`） | RT スレッド |
| 消費者 | `AudioEngine.Timer.cpp:1736` の `diagBuffer.pop(event)` | タイマースレッド |
| 診断の有無 | `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS`（既定 OFF。`build-diag` のみ ON） | コンパイル時 |

**観察**: 生産者は「エンジンのメンバ」を指すポインタを**グローバル経由**で受け取る。
所有（エンジン）と参照（グローバル）が分離しているため、寿命を追えない。

## 4. 設計

### 4.1 所有グラフ（案）

```
AudioEngine（所有）
 ├─ diagBuffer : LockFreeRingBuffer<...>        ← 既存メンバ（変更しない）
 ├─ rtAuxMutable_.diagTickPushed / Dropped / TotalPushed  ← 既存メンバ（変更しない）
 ├─ diagSink_  : DiagSink                        ← 新設。上記 4 つへの「参照の束」（所有しない）
 ├─ convolver : ConvolverProcessor               ← sink_ に自分のエンジンの DiagSink* を保持
 ├─ uiConvolverProcessor                         ← 同上
 └─ DSPCore 群（Float/Double）                    ← 同上（logEqTime 用）

ConvolverProcessor / DSPCore（参照）
 └─ std::atomic<DiagSink*> sink_{nullptr}        ← グローバルを廃止し、インスタンスが保持
```

- **バッファ本体は既存メンバのまま**（`diagBuffer` / `rtAuxMutable_`）。移設しない。
  消費者である `Timer.cpp:1736` が同一エンジン内で pop するため、所有者はエンジンで正しい。
- `DiagSink` は**参照の束**（4 ポインタ）であり、所有権を持たない。
  これにより「所有（エンジン）」と「参照（生産者）」を型で分離しつつ、変更範囲を小さく保つ。
- `DiagSink` は**`#if` で囲まず常時存在**させる。`AudioEngine.h:3984-3988` の教訓
  （マクロで DATA MEMBER を出し分けると TU 間で `sizeof(AudioEngine)` が食い違い、静かな ODR 違反になる）
  を踏襲する。

### 4.2 受け渡しの型

RT 側で 1 回の atomic load で済むよう、**ポインタ 1 本**に集約する。

```cpp
// 新設: 既存メンバへの参照の束（所有しない）
struct DiagSink {
    LockFreeRingBuffer<DiagEvent, DiagRuntimeLimits::BufferCapacity>* buffer;
    DiagPerTickCounter* tickPushed;
    DiagPerTickCounter* tickDropped;
    std::atomic<std::uint64_t>* totalPushed;
};
```

RT 側は `DiagSink*` を 1 回 load すれば 4 つの参照が揃う（atomic load 1 回）。
バッファ本体は既存メンバのままなので、**リングとカウンタの寿命は元から一致している**
（問題は「参照がグローバルで寿命を追えていなかった」こと）。

### 4.3 寿命規則（R2 / R4 の中核）

**規則 L1（参照の生成）**: `attachDiagSink(sink)` は
`sink_.store(sink, std::memory_order_release)` で行う。呼び出しはエンジン初期化（非 RT）。

**規則 L2（参照の破棄）**: `detachDiagSink()` は
`sink_.store(nullptr, std::memory_order_release)`。呼び出しは**バッファ破棄より前**。

**規則 L3（RT の読み取り）**: RT は毎回 `sink_.load(std::memory_order_acquire)` し、
**ポインタをブロック間でキャッシュしない**。null なら何もしない。

**規則 L4（破棄順序）**: `DiagSink` を破棄してよいのは、次の 3 条件がすべて成立した後。

1. すべての生産者に対して `detachDiagSink()` が完了している（新規参照が発生しない）
2. **RT スレッドが当該エンジンの処理を実行していない**ことが確認できている（§4.4）
3. 消費者（タイマー）も当該バッファを pop していない

**規則 L5（順序の強制）**: L4 の 1→2→3 を**エンジンの破棄経路でコードとして固定**する。
「detach したから安全」とはしない（detach と破棄の間に RT が 1 回 push し得るため）。

> **R4 への回答**: atomic 化だけでは L4-2（RT の静止）を担保できない。
> 本設計は「detach（新規参照の停止）」と「静止確認（既存参照の終了）」を**別々**に要求する。

### 4.4 RT の静止確認（R3）

現行の停止経路を利用し、**破棄前に静止を確認**する。

| 段階 | 操作 | 根拠（実測） |
|---|---|---|
| 1 | オーディオコールバックの停止要求 | `AudioEngine.Processing.ReleaseResources.cpp:153` の `ShutdownPhase::StopAudio`、`AudioEngineHarness::stopAudioOnly()` |
| 2 | オーディオスレッドの**終了待ち** | `~AudioEngine` は `CtorDtor.cpp:154-171` で StopAcceptingWork → StopAudio → StopWorkers（retryScheduler/Coordinator/rebuild を停止）の順に固定している。**オーディオコールバックの join が同経路に含まれるかは要確認**（§8-1） |
| 3 | コールバック生存数の確認 | `rtLocalState_.audioCallbackActiveCount`（`AudioEngine.h:1852`）が 0。増減は `AudioBlock.cpp:93/98`、`BlockDouble.cpp:88/93` で**コールバック本体を囲んでいる**ため、静止判定に使える |
| 4 | 各生産者に `detachDiagSink()` | 新規参照の停止 |
| 5 | `DiagSink` を破棄 | — |

**実装上の要点**: 段 3 のカウンタは RT 側で `fetch_add`/`fetch_sub` されており、
**既に毎コールバック 2 回の atomic 操作が存在する**。これを静止判定に流用すれば
新たな RT コストを増やさずに L4-2 を満たせる（新規カウンタを追加しない）。

### 4.5 RT 制約（R6）

RT 側のコードは次の形に限定する。

```cpp
// RT（process / logEqTime）
if (auto* s = sink_.load(std::memory_order_acquire)) {
    DiagEvent event{ /* ... */ };
    if (s->buffer.push(event)) { s->tickPushed.value.fetch_add(1, relaxed); }
    else                       { s->tickDropped.value.fetch_add(1, relaxed); }
}
```

| 制約 | 遵守 |
|---|---|
| ロックを取らない | `std::atomic` の load/store/fetch_add のみ（ロックフリー） |
| メモリ確保をしない | `DiagEvent` はスタック上の POD。リングは固定長 |
| 解放をしない | 参照のみ。所有しない |
| 例外を投げない | 既存の `noexcept` を維持 |

### 4.6 診断無効ビルド（R5）

- `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` が OFF のとき、`push` を行うブロックは
  現行どおり**コンパイル対象外**とする（コードは生成されない）。
- `DiagSink`（参照の束）と `attach/detach` は **`#if` で囲まず常時存在**させる。
  マクロで DATA MEMBER を出し分けると、TU 間で `sizeof(AudioEngine)` が食い違い
  **静かな ODR / レイアウト違反**になる（`AudioEngine.h:3984-3988` の実例）。
- `sizeof(AudioEngine)` は数ポインタ分（32 B 程度）増えるが、**全 TU で同一**であり、
  診断 OFF でも `attach/detach` は空実装のまま呼べる（呼び出し側に `#if` を持ち込まない）。

## 5. 確認すべき 3 ケース（R5）

| # | ケース | 期待 |
|---|---|---|
| C1 | **複数エンジンの同時存在**（2 エンジンが同時に RT を回す） | 各エンジンが**自分の**バッファへ push する（混線しない）。イベント数の合計が一致する |
| C2 | **初期化・破棄の繰り返し**（ハーネスは 114 回構築） | 破棄後に旧エンジン宛の push が発生しない。アクセス違反 0 |
| C3 | **診断無効ビルド** | コンパイルが通り、`sizeof(AudioEngine)` と `ConvolverProcessor` のレイアウトが変わらない |

## 6. 変更対象（実装フェーズで触るファイル・関数）

| ファイル | 変更内容 |
|---|---|
| `src/audioengine/AudioEngine.h` | `DiagSink` の定義、`diagSink_` メンバ、`attach/detach` 宣言 |
| `src/audioengine/AudioEngine.Init.cpp:116-119` | グローバル設定 → 各生産者への `attachDiagSink(&diagSink_)` |
| `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | 破棄経路に L4 の 1〜4 を追加（detach → 静止確認） |
| `src/convolver/ConvolverProcessor.Runtime.cpp:85-104, 764, 849-870` | グローバル削除、`sink_` へ置換 |
| `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp:28-46, 63-90` | 同上 |
| `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp:16-45` | 同上 |
| `src/audioengine/AudioEngine.Timer.cpp:1736` | 消費側は `diagSink_.buffer.pop` に追随（名前の付け替えのみ） |

## 7. 検証計画（実装後・**実行は別途承認**）

| # | 内容 | 合格条件 |
|---|---|---|
| V1 | **診断 ON** のハーネス反復（現行の再現手順と同じ） | クラッシュ **0**（現行は 8.3%） |
| V2 | 2 エンジン同時のイベント混線試験 | 各エンジンのイベントが自分のバッファにのみ入る |
| V3 | 114 回の構築・破棄 | アクセス違反 0、リーク 0 |
| V4 | **診断 ON の ASan ビルド**（新設） | 51/51 PASS。**この経路を初めて ASan で検証できる** |
| V5 | 診断 OFF ビルド | 既存 51/51 PASS を維持。レイアウト不変（`static_assert(sizeof(...))` で確認） |
| V6 | RT 制約 | コードレビューで RT 区間に lock/alloc/free が無いこと。`/analyze` 等は任意 |

**V4 の意義**: 現行の ASan 構成は診断 OFF のため、本件の経路を一度も検証していない
（`build-asan-local` は `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=OFF`）。
診断 ON の ASan 構成を追加することが、再発防止の実効性を高める。

## 8. 未確定事項（実測で解決したもの／残るもの）

### 8.1 解決済み（2026-11-01 20:20 実測）

| # | 項目 | 結果 |
|---|---|---|
| 1 | `AudioEngine` のデストラクタの停止順序 | `CtorDtor.cpp:154-171` で **StopAcceptingWork → StopAudio（`stopTimer`）→ StopWorkers（retryScheduler / Coordinator join / rebuild 停止）** の順に固定。**オーディオコールバックの join が同経路に含まれるかは未確定**（`audioCallbackActiveCount` による静止判定を併用する） |
| 2 | 生産者インスタンスの一覧 | `convolver`（`AudioEngine.h:894`）、`uiConvolverProcessor`（同 2317）、DSPCore 群（`DSPCoreFloat`/`DSPCoreDouble` の `logEqTime`） |
| 3 | `DiagSink` 追加によるレイアウト変化 | 診断 OFF では空実装にするため `sizeof(AudioEngine)` は不変（実装時に `static_assert` で固定する） |
| 4 | B-1 で足りるか | **足りない**。detach と破棄の間に RT が load 済みポインタを使う窓が残る（R4） |

### 8.2 残る未確定事項

1. オーディオコールバックの **join** が `~AudioEngine` / `releaseResources` のどこで保証されるか
   （`audioCallbackActiveCount == 0` の待機を入れるか、既存の join に依存するか）。
2. `attach` を呼ぶ正確なタイミング（`Init` か `prepareToPlay` か）。複数回 `prepareToPlay` される
   経路があるため、**冪等な attach** にする必要がある。
3. 診断 ON の ASan 構成（V4）を CI に追加するか（追加する場合は CI 定義の変更承認が必要）。

## 9. 代替案との比較（参考）

| 案 | 内容 | 判定 |
|---|---|---|
| B-1 | 破棄時にグローバルを `nullptr` にし、ポインタを atomic 化 | **不十分**。detach と破棄の間に RT が load 済みのポインタを使う窓が残る（R4） |
| **B-2（本設計）** | グローバル廃止＋インスタンス所有＋停止順序の固定 | **採用候補** |
| B-3 | 世代トークンで所有者を検証 | 安全だが複雑。B-2 で足りるなら不要 |

## 10. 実施しないこと

- 製品コード・テストの変更（本設計は文書のみ）
- ビルド・デバッガ・リプレイの実行（Owner の別途承認が必要）
- U5 / U3 / U4 の状態変更、合格基準の変更
