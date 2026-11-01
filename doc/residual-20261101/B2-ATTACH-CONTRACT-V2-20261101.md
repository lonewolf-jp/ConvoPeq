# B-2 attach 契約 第 2 版 — owner/sink の整合・Release 強制・T10 廃止（2026-11-01）

- 状態: **設計資料のみ。製品コード・テスト・CI 定義は未変更。build / test / debugger / replay 未実行**
- 基準: `main` = `origin/main` = `6bb7c8d5`（ソースは `fd92c6e7` と同一）
- 目的: Owner 指示 4 点の修正（owner/sink 整合、owner 一致の Release 強制、T10 廃止、メンバ配置）と、
  `drainAllQuarantineStore()` の対象範囲の確認

## 0. 前版からの変更点（要約）

| # | 前版 | 第 2 版 |
|---|---|---|
| 1 | `owner_` を `AudioEngine*`（実体ポインタ） | **`uint64_t ownerId_`（識別情報。参照しない）** |
| 2 | `assert` で owner 一致 | **Release でも比較し、不一致なら bind しない**（戻り値で通知） |
| 3 | T10「leak したコアは dangling を許容」 | **T10 を廃止**。teardown で leak 対象コアも detach する |
| 4 | `DSPCore` に 16 B 追加（owner_ + sink_） | **`DSPCore` に 16 B（u64 + atomic ptr）のみ**。`ConvolverProcessor` / EQ は **0 B**（sink を引数で受け取る） |
| 5 | 外側クラスが入れ子の private に触れる前提 | **規格どおり不可**。ロジックを入れ子側（`DSPCore`）へ寄せ、外側は触らない |

## 1. `drainAllQuarantineStore()` の対象範囲（実測）

### 1.1 実装

`ISRRetireRouter.cpp:628-635`:

```cpp
void ISRRetireRouter::drainAllQuarantineStore() noexcept {
    m_retireQuarantine.drainAllUnsafe();      // Q
    m_emergencyQuarantine.drainAllUnsafe();   // E
    m_terminalReclaim.drainAll();             // T
}
```

**D（deferred/epoch ストア）は含まない。** `drainAll()`（`:821` で `provider_->drainAll()` を呼ぶ）が D を含む全ストアを対象とする。

### 1.2 `~AudioEngine` の実際の分岐（`CtorDtor.cpp:329-336`）

```cpp
if (m_retireRouter->activeReaderCount() == 0)
    m_retireRouter->drainAll();                       // D + Q + E + T
else {
    m_epochDomain.drainAll();                         // D のみ
    m_retireRouter->drainAllQuarantineStore();        // Q + E + T
}
```

### 1.3 **コアがエンジンより長生きする経路が実在する**

`CtorDtor.cpp:338-359`:

> `retryDeferredDestroys()` 後も `deferredDestroyResidentCount() > 0` の場合、
> **「強制解放すると UAF になるため、リークを選ぶ」**（コメント: `uncollected: pins still held — leaked, not UAF`）。

**したがって「エンジン破棄時にすべてのコアが破棄される」とは言えない。**
一方、**leak されるコアは `deferredDestroy_` に保持されており、列挙できる**。

| 項目 | 実測 |
|---|---|
| 保持構造 | `std::vector<void*> deferredDestroy_`（`AudioEngine.h:5254`、`deferredDestroyMutex_` で保護） |
| 件数 | `deferredDestroyResidentAtomic_`（`:5256`）、`deferredDestroyResidentCount()`（`:5235`） |
| 列挙 | **可能**（vector。要素は DSPCore ノードのポインタ） |

→ **teardown 時にこの vector を走査して各コアの sink を detach できる。** これが T10 廃止の根拠である。

## 2. 修正 1: owner は識別情報にする（Owner 指示 2 の一部）

| 項目 | 決定 |
|---|---|
| `owner_` の型 | **`std::uint64_t ownerId_`**（0 = 未束縛） |
| 生成 | `AudioEngine` が生成時に一意 ID を採番（`std::atomic<uint64_t>` のカウンタ） |
| 用途 | **比較のみ**（bind 時の一致検査）。**逆参照しない** |
| エンジン破棄後 | ID はただの数値。dangling ポインタにならない |

**「owner の識別情報と owner オブジェクトの寿命を混同しない」**という Owner 指摘に、型で答える形である。
実体への参照が必要な場面（`bind` 時の `&engine.diagSink_` 取得）は、**その場で渡される `AudioEngine&` から取る**（保存しない）。

## 3. 修正 2: owner 一致を Release でも強制する（Owner 指示 2）

```cpp
// DSPCore（入れ子クラス）の prepare 内 — 外側 AudioEngine の private にアクセスできる
bool bound = false;
if (owner != nullptr)
{
    const std::uint64_t id = owner->diagnosticsEngineId();     // private accessor（入れ子→外側は可）
    std::uint64_t expected = 0;
    if (ownerId_.compare_exchange_strong(expected, id, std::memory_order_acq_rel))
        bound = true;                                          // 初回束縛
    else if (expected == id)
        bound = true;                                          // 同一 owner（再 prepare）
    else
        return;                                                // ★ 別 owner: 束縛しない（Release でも）

    if (bound)
        sink_.store(&owner->diagnosticsSink(), std::memory_order_release);
}
```

| 要件 | 担保 |
|---|---|
| assert に依存しない | **`if (expected != id) return;` は全ビルドで実行**。assert は補助のみ |
| Release で誤 bind を成立させない | 上記の比較が Release でも走る。失敗時は `sink_` を**書かない** |
| 外側が入れ子の private に触れない | ロジックを **`DSPCore::prepare`（入れ子側）** に置く。外側は `owner->diagnosticsSink()` を返すだけ |
| 入れ子→外側のアクセス | 規格どおり**可**（[class.access.nest]/1: 「入れ子クラスはメンバであり、他のメンバと同じアクセス権を持つ」） |
| 外側→入れ子のアクセス | **不可**（同項: 「外側クラスのメンバは入れ子クラスのメンバに特別なアクセス権を持たない」）。**本設計は外側から触らないので `friend` 不要** |

## 4. 修正 3: owner/sink の整合性（Owner 指示 1）

### 4.1 不変条件

| ID | 不変条件 |
|---|---|
| **I1** | `sink_ != nullptr ⟹ ownerId_ != 0`（sink を持つ個体は必ず owner を持つ） |
| **I2** | `ownerId_` は 0 → E の一方向のみ（E → 0 には戻さない。detach は sink のみ） |
| **I3** | 書き込みは **単一の NonRT 制御経路**（rebuild スレッドの `prepare`、または `initialize`）からのみ |

### 4.2 「2 つの atomic を独立更新しても不整合にならない」理由

| 懸念（Owner 例） | 本設計での扱い |
|---|---|
| Thread A: owner=E、Thread B: owner=F が並行 | **I3 により並行しない**（bind は rebuild スレッド 1 本、または初期化 1 回）。加えて `ownerId_` は CAS なので、仮に並行しても**片方だけが成功**する |
| owner 設定後に sink 設定が失敗（部分設定） | 状態 **U(ownerId=E)** は**合法**（T4 と同じ）。I1 は「sink 非 null なら owner 非 0」であり、逆は要求しない |
| sink だけ設定される | コード上あり得ない（sink の store は owner 取得成功後にのみ実行） |
| 再試行 | 同一 owner の再 prepare は `expected == id` で成功し、sink を再 store（同一値） |

**「単に private だから排他」ではなく、I3（単一経路）＋ CAS（単一成功）＋ 合法な中間状態（U(ownerId)）で閉じる。**

### 4.3 排他の根拠（実測・前版から維持）

`ReleaseResources.cpp`: `:236-238` で Coordinator/Builder を **join** → `:444` で drain（コア破棄）。`~AudioEngine` も `stopRebuildThread()` 後に drain。**prepare と detach は並行しない**。

## 5. 修正 4: T10 の廃止（Owner 指示 3）

### 5.1 変更後の状態遷移

| # | 現在 | 事象 | 遷移 | 許否 |
|---|---|---|---|---|
| T1 | U(id=0) | `prepare(E)` | B(E) | 許容 |
| T2 | B(E) | `prepare(E)` | B(E)（不変） | 許容 |
| T3 | B(E) | `prepare(F)` | B(E)（不変） | **拒否**（Release でも） |
| T4 | B(E) | `detach()` | U(id=E) | 許容 |
| T5 | U(id=E) | `prepare(E)`（再 attach） | B(E) | 許容（条件 §6） |
| T6 | U(id=E) | `prepare(F)` | U(id=E)（不変） | **拒否** |
| T7 | B(E) | コア破棄（dtor） | 終了 | 許容 |
| T8 | B(E) | エンジン teardown（静止後） | **detach → 破棄** | 許容 |
| **T10（改）** | B(E) | **エンジン破棄・コア leak** | **detach → U(id=E) → leak** | **許容（sink は nullptr）** |
| T9 | U | `process()` | 不変（push なし） | 許容 |

**T10 の実現方法**: `~AudioEngine` の member teardown 前に、`deferredDestroy_`（列挙可能）の各要素に対して
`detachDiagSink()` を呼ぶ。**leak されるコアにも解放済み sink へのポインタが残らない。**

```
B(E)
  ↓ RT停止・全生産者静止
detach（drained されたコアは dtor で、leak されるコアは teardown の走査で）
  ↓
U(id=E)   ← sink は nullptr。owner ID は数値のみ
  ↓
エンジン破棄 → コア leak（sink=nullptr）
```

**残余**: ID（数値）は残るが、参照も逆参照もされないため無害。

## 6. detach 後の再 attach 契約（Owner 指示の追加確認）

再 attach（T5）を許す条件を**明示**する。

| # | 条件 | 担保 |
|---|---|---|
| R1 | 旧 sink を参照する生産者が存在しない | 全生産者は RT（§7）。RT は join 済み（T4 は teardown 経路でのみ発生） |
| R2 | detach が完了している | `store(nullptr)` の完了。単一経路 |
| R3 | 新しい sink が有効 | 同一 owner の diagSink は**エンジンの生存中**有効（owner は生涯固定） |
| R4 | 別 owner からの attach は拒否 | `ownerId_` の CAS 比較（Release でも） |
| R5 | detach と再 attach が並行しない | どちらも単一 NonRT 制御経路（I3） |

**「sink が nullptr であること」だけでは R1〜R5 を導かない**ため、上表を契約として明記する。

## 7. 生産者の範囲（前版の補強）

全 14 push サイトは RT 処理関数内（前版 §4.1 の実測）。
**間接呼び出し・関数ポインタ・共通ヘルパー経由の有無**を確認した結果:

| 確認 | 結果 |
|---|---|
| `ConvolverProcessor::process` の呼び出し元 | **`DSPCoreDouble.cpp:390/454`** と **`DSPCoreFloat.cpp:274/339`** の 4 箇所のみ（いずれも `convolverRt().process(processBlock)`） |
| `logEqTime` の呼び出し元 | `DSPCoreDouble.cpp:408/432`、`DSPCoreFloat.cpp:292`（いずれも `DSPCore::process` 系） |
| `getNextAudioBlock` の呼び出し元 | `AudioEngineProcessor.cpp:88`（JUCE の audio callback）のみ |
| 関数ポインタ・`std::function`・`std::bind` 経由 | **0 件**（`ConvolverProcessor::process` の束縛なし） |

→ **生産者は audio callback に限定できる**（B-2 適用後）。

## 8. メンバ配置（Owner 指示 4）

| クラス | 追加メンバ | 増加 | 理由 |
|---|---|---|---|
| **`DSPCore`** | `std::uint64_t ownerId_` ＋ `std::atomic<DiagSink*> sink_` | **16 B** | 束縛の唯一の保持者。RT はこの 1 ポインタを読む |
| `ConvolverProcessor` | **なし** | **0 B** | sink を **引数**で受け取る（`process(block, sink)`） |
| `EQProcessor` / `DSPCore::eq` | **なし** | **0 B** | `logEqTime(..., sink)` で受け取る |
| `AudioEngine` | なし（`engineId_` のみ） | 8 B | ID 採番用 |

**「DSPCore に 16 B」が最小構成**である。`ConvolverProcessor` に sink を持たせると、
非所有インスタンスの attach 状態管理が再び必要になる（§0 の変更点 5）。

### 8.1 ODR について（`static_assert` を完全証明にしない）

- ODR の担保は **「単一のクラス定義・メンバを `#if` で出し分けない」**ことによる。
  `AudioEngine.h:3984-3988` の教訓（マクロでメンバを出し分けると `sizeof` が食い違う）を踏襲する。
- `static_assert(sizeof(DSPCore) == N)` は**サイズの一致を検査する補助**であり、
  レイアウト・ODR の完全証明ではない（Owner 指摘のとおり）。
- 追加する 2 メンバは**診断 ON/OFF で同じ**（`#if` を付けない）。

## 9. 検証への追加（ASan 計画の更新）

| 段 | 検証 | 判定 |
|---|---|---|
| S11 | **leak 経路**（pin を残したままエンジン破棄） | leak されたコアの `sink_` が **nullptr**（teardown の走査で detach） |
| S12 | **Release ビルドでの誤 bind**（別エンジンの prepare を強制） | bind されない（`sink_` が設定されない）。assert 無効でも成立 |
| S13 | 再 attach（detach → 同一 owner で prepare） | R1〜R5 を満たす |
| S8（再掲） | 異なる sink への attach 拒否 | 拒否 |
| S9（再掲） | `sink_ == nullptr` での書き込み | 0 回 |

**ASan 未検出を UAF 不存在の証明としない**（前版どおり）。TSan は別軸。

## 10. 残る確認（実装時）

1. `deferredDestroy_` の各要素を `DSPCore*` として安全に扱えるか（型消去 `void*` のため、
   格納時に「sink detach 用のコールバック」も併せて保持する案が安全）。
2. `DSPCore::prepare` の呼び出しが**同一スレッド**（rebuild）であることの再確認
   （`PrepareToPlay.cpp:272` は message スレッド。エンジンごとに 1 回）。
3. `AudioEngine::diagnosticsEngineId()` / `diagnosticsSink()` を private にしたうえで、
   `DSPCore`（入れ子）から到達できることの実装確認。

## 11. 結論

- owner は**識別情報**、sink は**単一経路 + CAS + 合法な中間状態**で整合を保つ。
- owner 一致は **Release でも強制**（assert 依存をやめた）。
- **T10 を廃止**し、leak されるコアも teardown で detach → **解放済み sink を指す個体を残さない**。
- **メンバ追加は `DSPCore` の 16 B のみ**。プロセッサは sink を引数で受ける（0 B）。
- 入れ子クラスのアクセス規格に合わせ、ロジックを**入れ子側**に置く（`friend` 不要）。

**上記で「エンジン破棄後に、そのエンジンが所有していた診断バッファを参照できるオブジェクトを残さない」
という原則を、状態機械（T10 改）と API（Release 強制・単一経路）の両方で強制できる。**
