# D101-1.5 — Finite-Bound Source Audit

| 項目 | 内容 |
| --- | --- |
| **日付** | 2026-08-20 |
| **対象** | `src/core/RuntimeStore.h`, `src/audioengine/RuntimeWorldAuthority.h`, `src/audioengine/ISRRetireRouter.h/.cpp`, `src/DeferredDeletionQueue.h`, `src/audioengine/RetireQuarantineStore.h`, `src/audioengine/RuntimeBuilder.h`, `src/audioengine/AudioEngine.h/.Commit.cpp/.Timer.cpp`, `src/audioengine/ISRWorldRetirementTelemetry.h` |
| **目的** | D101-1 で不足と判定した 4 つの有限上界（1 区間 acquire 数 / 同時 outstanding world 数 / burst・producer activity / sampler 最大 gap G）について、既存コード・設計契約に**暗黙の有限上限**が存在しないかを完全列挙し、`N`（pool/capacity）が実際の outstanding world 数の上限を意味する不変条件かを確認する。 |
| **判定** | **NO_FINITE_BOUND** — producer activity / outstanding lifetime に有限上限が存在しない |

---

## 1. 監査対象と結論の要約

| # | 監査対象 | 発見 | 有限上界? |
| --- | --- | --- | --- |
| 1 | `RuntimeStore`（world 保持数） | **単一スロット** `std::atomic<T*> current`。`publishAndSwap` で交換。 | 1（current のみ） |
| 2 | `DeferredDeletionQueue`（retire キュー） | Vyukov MPMC、`kQueueSize = 4096`。満杯で `enqueue` は `false`。 | 4096（境界付き） |
| 3 | `RetireQuarantineStore` | `kMaxQuarantinedEntries = 512`（**8192 ではない**）。 | 512（境界付き） |
| 4 | `EmergencyQuarantineStore` | `RetireQuarantineStore` の別インスタンス → 容量 512。 | 512（境界付き） |
| 5 | **`TerminalReclaimAuthority`** | **`std::vector<Entry>`（GROWABLE・無制限）**。`store()` は常に true。 | **∞（無制限）** |
| 6 | `RuntimeBuilder` | `buildRuntimePublishWorld()`。同時 build 数の明示 cap なし。 | なし |
| 7 | `publicationSequenceCounter_` | 単調増加 `uint64`（`AudioEngine.h:2225`）。採番 ID であり個数上限ではない。 | なし（ID 空間） |
| 8 | `worldReclaimCount_` | **world 物理破棄数**（type==World の terminal deleter 実行時）。3 ストア + TerminalReclaim 合算。 | — |
| 9 | acquire/release の semantic event | **1 world = 1 acquire（publish）+ 1 release（破棄）**。1:1 対応。 | — |
| 10 | producer レート | publish は Timer/Transition/PrepareToPlay/Executor/Orchestrator から多重呼び出し。**レート制限なし**。 | なし |
| 11 | `maxSamplingGapUs` | 観測された最大 gap を追跡するのみ。**hard cap なし**。 | なし |

---

## 2. 詳細調査結果

### 2.1 RuntimeStore — 単一スロット（world 保持数 = 1）

`src/core/RuntimeStore.h`:
```cpp
std::atomic<T*> current { nullptr };
```
- `publishAndSwap(next)` は `exchangeAtomic(current, next, acq_rel)` でポインタを交換し、旧 world を返す。
- ストアが同時に保持する公開 world は **最大 1 個**（current）。
- 旧 world は caller（PublishExecutor / Bootstrap）が retire 対象として受け取り、`ISRRetireRouter` へ移送される。

### 2.2 DeferredDeletionQueue — 4096（境界付き）

`src/DeferredDeletionQueue.h:262-266`:
```cpp
static constexpr uint32_t kQueueSize = 4096;
static constexpr uint32_t kMask = kQueueSize - 1;
alignas(64) std::array<DeletionEntry, kQueueSize> ringBuffer;
alignas(64) std::array<std::atomic<uint32_t>, kQueueSize> sequences;
```
- Vyukov bounded MPMC。`enqueue` は満杯（`diff < 0`）で `false` を返す。
- `reclaim()` は epoch 安全到達後に deleter 実行。type==World で `worldReclaimCount_` をインクリメント。

### 2.3 RetireQuarantineStore — 512（境界付き・8192 ではない）

`src/audioengine/RetireQuarantineStore.h`:
```cpp
static constexpr std::size_t kMaxQuarantinedEntries = 512;
```
- **ユーザーが想定した 8192 ではなく 512**。`std::array<QuarantinedEntry, 512>`。
- 満杯時 `quarantine()` は `false`（deleter は実行しない = UAF 構造的排除）。`overflowCount_` を記録。

### 2.4 EmergencyQuarantineStore — 512（境界付き）

`src/audioengine/ISRRetireRouter.h:356`:
```cpp
RetireQuarantineStore m_emergencyQuarantine;  // 同一タイプ・別インスタンス
```
- `RetireQuarantineStore` の別インスタンス → 容量 512。

### 2.5 ★ TerminalReclaimAuthority — GROWABLE（無制限）＝有限上界の否定

`src/audioengine/ISRRetireRouter.h:113-115`:
```cpp
// ★ P-4: Growable store (std::vector) — Non-RT only, heap allocation acceptable.
std::vector<Entry> entries_;
```
`src/audioengine/ISRRetireRouter.cpp:27-36`:
```cpp
bool TerminalReclaimAuthority::store(...) noexcept {
    ...
    entries_.push_back(Entry{ptr, deleter, epoch, type, reason});
    residentAtomic_.fetch_add(1, std::memory_order_release);
    return true;  // ★ P-4: growable store — ALWAYS accepts
}
```
- **`std::vector` により無制限に成長**。`store()` は常に true（"store full" 経路なし）。
- 設計意図: 「enqueueWithRetry() が ptr を unowned のまま返さない」所有権不変条件を保証（UAF 排除）。
- `drain()` は epoch 安全到達後に deleter 実行し、type==World で `reclaimCount_` をインクリメント + `referenceObserver_->onRelease()`。
- **結果**: DDQ(4096) + RetireQuarantine(512) + EmergencyQuarantine(512) が満杯でも、world は TerminalReclaim に**無制限に蓄積**し得る。

### 2.6 RuntimeBuilder — 同時 build 上限なし

`src/audioengine/RuntimeBuilder.h:209`:
```cpp
class RuntimeBuilder {
    [[nodiscard]] convo::aligned_unique_ptr<const RuntimePublishWorld>
    buildRuntimePublishWorld(...) noexcept;
    ...
};
```
- world 構築の唯一の入口。**同時 build 数を制限する cap は存在しない**。

### 2.7 publicationSequenceCounter_ — 採番 ID（個数上限ではない）

`src/audioengine/AudioEngine.h:2225`:
```cpp
std::atomic<convo::isr::PublicationSequenceId> publicationSequenceCounter_ { 0 };
```
`AudioEngine.h:3451`:
```cpp
identity.publicationSequence = convo::fetchAddAtomic(publicationSequenceCounter_, ...);
```
- 単調増加の採番カウンタ（uint64）。**world の個数上限を意味しない**（ID 空間の非重複保証のみ）。

### 2.8 worldReclaimCount_ — world 物理破棄数（release イベント総数）

`src/audioengine/ISRRetireRouter.cpp:400-406`:
```cpp
uint64_t ISRRetireRouter::worldReclaimCount() const noexcept {
    return provider_->worldReclaimCount()
        + m_retireQuarantine.worldReclaimCount()
        + m_emergencyQuarantine.worldReclaimCount()
        + m_terminalReclaim.reclaimCount();   // ★ TerminalReclaim 合算
}
```
- **意味**: type==World の terminal deleter 実行数 = **world 物理破棄数**（release イベント総数）。
- 4 経路（DDQ / RetireQuarantine / EmergencyQuarantine / TerminalReclaim）を合算。

### 2.9 acquire/release の semantic event — 1 world につき 1:1

- acquire: `onAcquireObserved()` は `AudioEngine.Commit.cpp:406` の **1 箇所のみ**（publish 成功時）。
- release: `worldReclaimCount_` は world 物理破棄時（type==World deleter 実行）にインクリメント。
- **1 world = 1 acquire + 1 release**。したがって outstanding = (publish 数) − (破棄数) = 生存 world 数。

### 2.10 producer レート — 制限なし

`commitRuntimePublication` の呼び出し元（多重）:
- `AudioEngine.Processing.PrepareToPlay.cpp:155, 277`
- `AudioEngine.Processing.ReleaseResources.cpp:175`
- `AudioEngine.Timer.cpp:964`
- `AudioEngine.Transition.cpp:25`
- `PublicationExecutor.cpp:53`（async facade）
- `RuntimePublicationOrchestrator.cpp:269`

- **publish レートを制限する throttle / minInterval / cooldown は存在しない**（`AudioEngine.Commit.cpp:669` の `minIntervalTicks` は診断ログ用の 1 秒間隔であり publish 制限ではない）。

### 2.11 maxSamplingGapUs — hard cap なし

`src/audioengine/ISRWorldRetirementTelemetry.h:297-299`:
```cpp
auto current = convo::consumeAtomic(maxSamplingGapUs_, ...);
... compareExchangeAtomic(maxSamplingGapUs_, current, gapUs, ...);  // 最大 gap を追跡
```
- 観測された最大 gap を記録するのみ。**missed tick 発生時の hard cap は存在しない**。

---

## 3. 合成: なぜ NO_FINITE_BOUND か

### 3.1 outstanding world 数の上界

生存 world 数 = current(1) + DDQ(≤4096) + RetireQuarantine(≤512) + EmergencyQuarantine(≤512) + **TerminalReclaim(≤∞)**。

- 境界付きストア（4096 + 512 + 512 + 1 = 5121）は**通常経路の容量**に過ぎない。
- `TerminalReclaimAuthority` は `std::vector` で**無制限に成長**し、`store()` は常に true。
- したがって **outstanding world 数には有限上界が存在しない**。

### 3.2 producer activity の上界

- publish は複数スレッド（Timer / Transition / PrepareToPlay / Executor / Orchestrator）から駆動。
- **レート制限なし**。publish が破棄（epoch 安全化）を上回ると、world は境界付きストアを溢れ、TerminalReclaim に無制限に蓄積する。

### 3.3 1 区間 acquire 数 / sampler gap

- 1 サンプリング区間内の acquire 数は producer レートに依存し、**上界なし**。
- `maxSamplingGapUs` は追跡のみで hard cap なし。100ms は名目値（D101-1 で確認済み）。

### 3.4 結論

既存コード・設計契約には、**outstanding world 数 / producer activity に有限上限を保証する不変条件が存在しない**。
`N`（queue capacity）は「world オブジェクトの通常経路容量」であり、**outstanding world 数の上限を意味する不変条件ではない**（TerminalReclaim の growable ストアが有限性を否定）。

---

## 4. 判定: NO_FINITE_BOUND

| 判定 | 適用条件 | 本件 |
| --- | --- | --- |
| BOUND_FOUND | 既存 invariant から有限 bound を証明できる | ✗ 該当せず |
| BOUND_EXISTS_BUT_UNCONNECTED | 数値 capacity はあるが、B_max^true と結び付く invariant がない | △ 境界付きストアはあるが、TerminalReclaim（growable）が有限性を否定 |
| **NO_FINITE_BOUND** | **producer activity / outstanding lifetime に有限上限が存在しない** | **◯ 該当** |

**理由**:
- `TerminalReclaimAuthority` は `std::vector`（growable・無制限）で、DDQ + 両 quarantine が満杯でも world を無制限に受け入れる。
- producer にレート制限がなく、publish が破棄を上回ると outstanding world 数は無制限に成長し得る。
- したがって $B_{\max}^{\text{true}}$ を有限の $M$ で上界することは**コード・契約上不可能**。

---

## 5. 次のゲート（ユーザー規定）

- **NO_FINITE_BOUND** のため、**D101 の M 証明は UNPROVABLE として閉じる**。
- **I-T2 / R の数値導出を停止**する。
- Phase I GO 判定は停止（D101-0 / D101-1 の INCOMPLETE から UNPROVABLE へ確定）。

---

## 6. ソースリンク

- `evidence/phase-d101-0-m-bound-mathematical-audit.md` — D101-0（verdict: INCOMPLETE）
- `evidence/phase-d101-1-m-bound-step2-counter-observation-error.md` — D101-1（verdict: INCOMPLETE）
- `doc/work88/I4_DESIGN_CONTRACT.md` — D101.4 末尾に D101-0 / D101-1 監査リンク済み