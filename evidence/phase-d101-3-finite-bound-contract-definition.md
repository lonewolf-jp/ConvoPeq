# D101-3 — Finite-Bound Contract Definition

| 項目 | 内容 |
| --- | --- |
| **ID** | D101-3 |
| **日付** | 2026-08-20 |
| **対象ブランチ** | `main` |
| **前提** | D101-2 verdict: **FINITE_BOUND_REQUIRES_ARCHITECTURAL_CHANGE** — 現行は growable TerminalReclaim (`std::vector`) により `M < ∞` を証明不能。`M = 1+4096+512+512` は通常経路容量であり `M_world` の証明ではない |
| **目的** | D101-2 の `ARCHITECTURAL_CHANGE_REQUIRED` を、実装可能な契約へ変換する。「何を有限にするのか」「その有限性をどの invariant で保証するのか」を数学的に固定する |
| **制約** | **コード変更なし・契約定義のみ**。`M_world / M_terminal / A_max / P_max / H_max / G_max` を数値ではなく invariant として定義。queue capacity を `M_world` と同一視しない。`G_max = observed maxSamplingGapUs` は禁止。I4 D14/D18.4 の reservation-first / backpressure と整合 |
| **判定** | **CONTRACT_REQUIRES_NEW_INVARIANT** — 契約の骨格（bound 分離・I4 接続・TerminalReclaim 修復モデル・M_world 分解）は本監査で定義可能だが、有限性を証明するには下記「Required architectural invariants」の新不変条件の導入が必須であり、現行コード・現行 I4 契約だけでは充足しない |

---

## 1. Scope

- 本監査は D101-2 の architectural change 要求を、**コード変更なし**で契約定義に落とすゲートである。
- 単に TerminalReclaim を bounded にするだけでは ownership conservation を壊す。`ConvoPeq.md` / `src/audioengine/ISRRetireRouter.h` では TerminalReclaim は D+Q+E 枯渇時の最終所有権 authority として `std::vector` に蓄積し `store()` が常に成功する。epoch-safe なら Non-RT で即時破棄、unsafe なら保持する構造である。
- したがって D101-3 では、**数値を置くのではなく**、有限性の対象・保証手段・I4 との接続・修復モデル・M_world の数学的分解を契約として固定し、不足する invariant を明示する。
- 構成は指示どおり 15 章（Scope / D101-2引継ぎ / Bound definitions / Existing invariant / Missing invariant / TerminalReclaim A/B/C/D / Ownership conservation / Reader hold / Producer activity / Sampling gap / M_world composition / I4 D14/D18 compatibility / Required invariants / Verdict / Next gate）。

---

## 2. D101-2 引継ぎ

| D101-2 の確定事実 | 本監査への含意 |
| --- | --- |
| `RuntimeStore 1 + DDQ 4096 + Q 512 + EQ 512` は通常経路容量。`TerminalReclaim = std::vector` により `outstandingWorlds <= 5121` は不成立 | `M_world` は **`M_current + M_reader + M_retire + M_quarantine + M_terminal`** の分解で定義し、各項の有限性を個別に証明する必要がある。単なる capacity 合算は `M_world` の証明ではない |
| 4案（A: bounded TerminalReclaim / B: outstanding cap / C: producer bound / D: 現状維持）は単独では M 証明不可。合成 + reclaim latency bound + backpressure が同時に必要 | 本監査の bound 分離（`M_world / M_terminal / A_max / P_max / H_max / G_max`）はその合成を契約化する |
| I4 D14 の `reservation-first + backpressure (terminal-failure 撤廃)` に整合する必要がある | `logical-obligation budget ≠ RuntimeWorld lifetime budget` を混同しない。reservation semantics は D18.4 の「新規 logical obligation 生成に対して exactly once」に従う |
| `maxSamplingGapUs` は observed maximum であり `G_max` の根拠にできない | 本監査で `G_max` を `G_contract` として再定義し、telemetry からの導出を禁止 |

---

## 3. Bound definitions

### 3.1 定義一覧

| Bound | 意味 | 型 | 備考 |
| --- | --- | --- | --- |
| `M_world` | 同時 outstanding `RuntimeWorld`（≒ `RuntimeState`）数 | `ℕ, M_world < ∞` を主張する invariant | 現行は `∞`（TerminalReclaim growable のため）。本監査で分解定義する |
| `M_terminal` | `TerminalReclaimAuthority` に保持可能な world 数 | `ℕ, M_terminal < ∞` を主張する invariant | 現行は `∞`（`std::vector`）。bounded 修復で `K` に置換 |
| `M_current` | `RuntimeStore::current` に保持される world 数 | `= 1`（単一スロット） | 既存 invariant。`RuntimeStore<T,Owner>` は単一 `std::atomic<T*> current` |
| `M_reader` | Reader（Audio Thread の `processBlock` 等）が同時に保持する world 参照数 | `ℕ` | `M_current` とは別。reader hold の並行度に依存 |
| `M_retire` | retire 経路（`DeferredDeletionQueue`）に滞留する world 数 | `≤ 4096`（`kQueueSize`） | 既存 bounded。ただし `M_world` の証明には単独で不十分 |
| `M_quarantine` | `RetireQuarantineStore` + `EmergencyQuarantineStore` に滞留する world 数 | `≤ 512 + 512 = 1024` | 既存 bounded |
| `A_max` | 1 sampling interval に許される `acquire` 数（= publish 成功数） | `ℕ` | `A_max` がなければ `M_world` の時間的 burst を上界できない |
| `P_max` | Producer が生成できる publish activity の上限（時間窓あたり） | `ℕ / 時間` | `P_max × H_max` が `M_world` の流入側 |
| `H_max` | Reader hold / reclaim latency の契約上限（world が解放されるまでの最大時間） | `時間` | `H_max` が `∞` なら `P_max` が有限でも `M_world` は `∞` |
| `G_max` | Sampler gap の契約上限（`G_contract`） | `時間` | `observed maxSamplingGapUs` から導出しない。新 contract として定義 |

### 3.2 禁止事項

- `M_world = 1 + 4096 + 512 + 512` と置くことは禁止（D101-1.5 で否定済み）。
- `G_max = observed maxSamplingGapUs` と置くことは禁止（D101-2 で禁止済み）。
- `M_world` に数値を直接代入することは禁止。本監査では **invariant の形**で定義し、充足条件を明示する。

---

## 4. Existing invariant

| Invariant | 現行の保証 | 有限性への寄与 | 不足 |
| --- | --- | --- | --- |
| `M_current = 1` | `RuntimeStore` は単一 `std::atomic<T*> current`。`WriteAccess::publishAndSwap` で旧 world を retire へ | `M_world` の 1 項を確定 | 単独では不十分 |
| `M_retire ≤ 4096` | `DeferredDeletionQueue kQueueSize = 4096`。Vyukov MPMC、満杯で `enqueue` は `false` | `M_retire` を bounded に | DDQ 満杯時は `RetireQuarantineStore` へ退避するが、その先の有限性は別途必要 |
| `M_quarantine ≤ 1024` | `RetireQuarantineStore kMaxQuarantinedEntries = 512` ×2（本体 + EmergencyQ）。`std::array` 固定容量、満杯で `store()` は `false`（deleter 実行せず UAF 排除） | `M_quarantine` を bounded に | さらに溢れた場合の最終所有権が growable TerminalReclaim に流れるため `M_terminal` が `∞` |
| `PendingPublishRegistry ≤ 64` | `RuntimeWorldAuthority::PendingPublishRegistry kPendingPublishCapacity = 64`。enqueue→commit の async gap を保持 | publish 前の lifetime gap を bounded に | publish 後の outstanding には含まない |
| `ISRRetireRouter::enqueueWithRetry` の ownership chain | D → Q → EQ → TerminalReclaim の順に `enqueueWithRetry()` が ownership transfer。いずれかが成功すれば ownership は失われない | 現行の所有権保全（bounded 前提では崩れる） | TerminalReclaim が growable であることが前提。bounded 化で再設計が必要 |
| `TerminalReclaimAuthority::store() ALWAYS true` | `std::vector` growable。`isOlder(entry.epoch, minReaderEpoch)` で epoch-safe なら即時破棄、unsafe なら pending 保持。`drain()` / `drainAll()` で epoch 到達後に破棄 | 現行の所有権保全を保証するが、有限性を犠牲にする | 有限性と両立させるには修復モデル A/B/C/D のいずれかが必要 |
| `I4 D14 reservation-first + backpressure` | 1 logical obligation = exactly 1 reservation (D18.4 で新規生成に対して exactly once に修正)。budget 枯渇 → backpressure（terminal-failure 撤廃）。COALESCE は新規 reservation を取得しない | Recovery obligation budget の有限性 | `RuntimeWorld lifetime budget` とは別 budget であることを明示する必要がある |
| `I4 D18.1 coalesce/supersede 境界` | `CoalesceIdentity = {handle, RecoveryEpisodeId, SemanticRecoveryTarget}`。同一 identity → COALESCE（obligation 不増）。Phase I で SUPERSEDE は不活性 | obligation 増加の抑制 | `M_world` の流入側（`A_max` / `P_max`）とは別レイヤー |

---

## 5. Missing invariant

| 不足 invariant | なぜ必要か | 現行の状態 |
| --- | --- | --- |
| `M_terminal < ∞` | `M_world` の最終項。growable `std::vector` のままでは `M_world = ∞` | 不存在。修復モデルで導入が必要 |
| `M_reader < ∞` | `M_world` の reader 項。reader が無制限時間 world を保持できるなら `M_world` は `∞` | 不存在。chapter 8 で検証 |
| `A_max < ∞` | 1 interval の acquire 上限。なければ burst で `M_world` が一時的に `∞` に発散し得る | 不存在。producer bound とセットで必要 |
| `P_max < ∞` | 時間窓あたりの publish activity 上限 | 不存在。chapter 9 で検証 |
| `H_max < ∞` | Reader hold / reclaim latency の契約上限。`H_max = ∞` なら `P_max` が有限でも `M_world = ∞` | 不存在。chapter 8 で検証 |
| `G_max = G_contract < ∞` | Sampler gap の契約上限。`observed maxSamplingGapUs` からの導出は禁止 | 不存在。chapter 10 で検証 |
| `RuntimeWorld lifetime budget` の明示 | `logical-obligation budget (kMaxLogicalRecoveryObligations)` と混同しない | I4 は前者のみ定義。後者は本監査で新設が必要 |
| `TerminalReclaim exhaustion` 時の ownership-preserving invariant | Bounded 化で `store(full)` が発生した際の所有権保全 | 不存在。chapter 6 で A/B/C/D 比較 |

---

## 6. TerminalReclaim repair alternatives A/B/C/D

### 前提（コード照合結果）

- `ISRRetireRouter.h`: `class TerminalReclaimAuthority { std::vector<Entry> entries_; std::mutex mtx_; std::atomic<uint32_t> residentAtomic_; }`。P-4 コメント「GROWABLE — ALWAYS accepts — NO store full failure path」。
- `ISRRetireRouter.cpp:TerminalReclaimAuthority::store()`: `std::lock_guard<std::mutex> lock(mtx_); entries_.push_back(...); residentAtomic_.fetch_add(1); return true;`。
- `RetireQuarantineStore.h`: `std::array<QuarantinedEntry, kMaxQuarantinedEntries=512>`。満杯で `store()` は `false`（deleter 実行せず）。
- `ISRRetireRouter.cpp:enqueueWithRetry()`: D（`DeferredDeletionQueue::enqueue`）→ Q（`RetireQuarantineStore::store`）→ EQ（EmergencyQ `store`）→ `TerminalReclaimAuthority::store()` の順。いずれか成功で `RetireEnqueueResult::Success` 相当。TerminalReclaim は常に成功。
- `ConvoPeq.md` の `ISRRetireRouter` 記述: epoch-safe なら即時 `deleter(ptr)`、unsafe なら pending 保持、`drain(minReaderEpoch)` で safe 到達後に破棄、`drainAll()` で shutdown 時に全強制解放。

### 比較

#### A — bounded TerminalReclaim + drop

```text
store(full) → reject (ptr を捨てる / deleter を呼ばずに放置)
```text

- **判定: NO-GO**（指示どおり）。
- 理由:

```text
ptr ownership transferred (caller は所有権を手放した)
        ↓
store(full) → false (bounded array 満杯)
        ↓
誰も ptr を所有しない → leak または UAF（誰かが後で ptr を触れば UAF、触らなければ leak）
```text

- I4 D14 の「capacity 枯渇を terminal-failure にして obligation を失わせない」に反し、「bounded にするため overflow 時に捨てる」禁止にも反する。
- `RetireQuarantineStore` は満杯時に `false` を返して deleter を実行しないことで UAF を構造的に排除しているが、それは **caller が `false` をハンドリングして再退避する**ことが前提。TerminalReclaim が最終 authority である場合、再退避先がないため drop は許されない。

#### B — bounded TerminalReclaim + caller rollback

```text
store(full)
    ↓
caller retains ownership (store() は所有権を移転せず false を返す)
    ↓
caller が安全に破棄（drain 到達まで保持 or 同期破棄）
```text

- **判定: 条件付き GO だが、証明責任が高い。第一候補ではない。**
- 必要な証明:

| 検証項目 | 現行コードとの照合 | 判定 |
| --- | --- | --- |
| RT/Non-RT 境界 | `TerminalReclaimAuthority::store()` の全 callers は Non-RT（`enqueueWithRetry` は `DSPLifetimeManager` / `SnapshotCoordinator` / `DeferredFreeThread` の Non-RT path）。`ConvoPeq.md` の P-4 前提「all callers are Non-RT」は現行で成立 | ✅ 成立 |
| Epoch safety | `store(full)` 時に epoch が safe なら caller が即時 `deleter(ptr)` できる。unsafe なら caller が保持し `drain()` 到達まで待つ必要があるが、caller の保持場所（stack / local pending list）が bounded か | ⚠️ 追加 invariant が必要。caller の pending が growable なら `M_terminal` を caller に移動しただけになる |
| `enqueueWithRetry()` の ownership disposition | 現行は `store()` が常に成功するため disposition は単純。bounded 化で `store(full) → caller retains` に変更すると、`enqueueWithRetry()` の戻り値・caller の分岐・再試行ループを再設計する必要がある。`RetireEnqueueResult` の enum 拡張と全 caller の対応が必要 | ⚠️ 設計変更が必要 |
| 既存 quarantine の `false` ハンドリングとの一貫性 | `RetireQuarantineStore::store()` が `false` → `enqueueWithRetry()` が EQ → TerminalReclaim へフォールバックする現行パターンと同様のフォールバックチェーンを caller rollback でも維持できるか | ⚠️ TerminalReclaim が最終段であるためフォールバック先がない。caller 自身が最終保持者になる |

- **結論**: B は原理的に可能だが、caller の保持場所を bounded にし、`enqueueWithRetry()` の disposition を再設計するコストが高い。C/D の方がクリーン。

#### C — reservation-first

```text
capacity reservation
        ↓ success (slot を確保)
ownership transfer (caller → TerminalReclaim)
        ↓
TerminalReclaim store (予約済みスロットへ配置、失敗しない)
```text

- **判定: 有力候補（指示どおり）。**
- 構造:

```text
reserve(M_terminal slot)  ──→  success: slot index を取得
        │ fail: backpressure（publish admission を BLOCK、既存 world の retire を待つ）
        ↓ success
build / publish world
        ↓
ownership transfer → reserved slot へ配置（store は失敗しない、確保済みのため）
        ↓
epoch-safe なら即時破棄 / unsafe なら pending 保持 → drain() で safe 到達後に破棄
```text

- 証明すべきこと: `reservation acquired → ownership transfer → store` の間に **失敗可能な操作が存在しない**こと。

| 区間 | 失敗可能性 | 対策 |
| --- | --- | --- |
| `reservation → ownership transfer` | `buildWorld` が失敗する可能性（OOM / validation 失敗） | 失敗時は reservation を `release` し world は生成しない。所有権は発生していないため UAF なし |
| `ownership transfer → store` | `store` 自体は予約済みスロットへの配置であり、allocation なし（`std::array` 固定容量 + index 配置）。失敗しない | `RetireQuarantineStore` と同様の `std::array + mutex + index` で allocation-free にすれば証明可能 |

- I4 D14/D18.4 との整合:

| I4 の reservation | 本監査の reservation |
| --- | --- |
| `kMaxLogicalRecoveryObligations` budget に対する `reservation-first`（新規 logical obligation 生成に対して exactly once。COALESCE は新規 reservation を取得しない） | `M_terminal` budget に対する `reservation-first`（新規 `RuntimeWorld` の publish に対して slot を予約） |
| budget 枯渇 → backpressure（新規生成を BLOCK） | slot 枯渇 → backpressure（新規 publish を BLOCK、coalesce 相当の重複 publish は既存 world の再利用で回避可能か検討） |
| `reservation acquired → ownership transfer → store` の間に失敗可能な操作がないことが前提 | 同じ構造。build 失敗時は reservation を release する |

- **C の利点**: `store(full) → caller retains` の複雑な rollback を避け、**失敗を reservation 取得時点に前倒し**する。所有権が発生する前に拒否するため UAF の余地がない。I4 の backpressure モデルと同一パターンであり、既存の設計思想と整合する。

#### D — unified lifetime budget（第一候補として検証）

```text
RuntimeWorld budget  (M_world)
    ↓
publish admission (A_max / P_max で流入制御)
    ↓
reservation (M_terminal slot)
    ↓
ownership transfer (RuntimeStore::publishAndSwap で current 交換、旧 world を retire へ)
    ↓
retire (DeferredDeletionQueue / Quarantine / TerminalReclaim のいずれか)
    ↓
release (epoch safe 到達 → deleter 実行 → world 破棄)
```text

- **判定: 第一候補として推奨（指示どおり）。C を包含し、さらに `M_world` 全体の有限性を一つの invariant として扱う。**
- 構造:

```text
Invariant: outstandingWorlds ≤ M_world

M_world = M_current (=1)
        + M_reader   (≤ H_max から導出)
        + M_retire   (≤ 4096)
        + M_quarantine (≤ 1024)
        + M_terminal (≤ K, reservation-first で保証)

流入:  publish admission は A_max / P_max で制御
       reservation-first で M_terminal slot を確保してから publish
       失敗 → backpressure（BLOCK）

流出:  epoch 進行 → drain() → deleter → release
       流出速度は H_max により下界が保証される
       （流出が停止すれば backpressure で流入も停止するため、M_world は発散しない）
```text

- **D が C を包含する理由**: D の `reservation` ステップは C と同一。D はそれを `M_world` 全体の budget の一部として位置づけ、流入（`A_max / P_max`）と流出（`H_max`）の両面から `M_world` の有限性を証明する枠組みを提供する。
- **I4 との接続**: D14 の `logical-obligation budget` は `M_world` とは別 budget であるが、**同一の reservation-first + backpressure パターン**を共有する。両 budget を統一的に扱うことで、設計の一貫性を保ちつつ、混同を防ぐために **budget の分離**を契約で明示する（chapter 12）。
- **D の証明責任**: `M_reader ≤ f(H_max)` / `H_max < ∞` / `A_max < ∞` / `P_max < ∞` / `G_max = G_contract` の各 invariant が別途必要。これらは chapter 8/9/10 で検証する。

### 比較まとめ

| 案 | 判定 | 所有権保全 | I4 整合 | 実装コスト | 推奨 |
| --- | --- | --- | --- | --- | --- |
| A (drop) | **NO-GO** | ❌ leak/UAF | ❌ terminal-failure 撤廃に反する | — | 採用不可 |
| B (caller rollback) | 条件付き GO | △ caller の保持場所が bounded か要証明 | △ disposition 再設計が必要 | 高 | 第二候補 |
| C (reservation-first) | **有力候補** | ✅ 所有権発生前に拒否 | ✅ D14/D18.4 と同型 | 中 | 採用可 |
| D (unified lifetime budget) | **第一候補** | ✅ C を包含 + `M_world` 全体を一つの invariant に | ✅ D と同型 + budget 分離を明示 | 中（C + 分解定義） | **推奨** |

**本監査の結論**: **D を第一候補、C を D の構成要素として採用**。A は禁止、B はフォールバックとして記録するが第一候補としない。

---

## 7. Ownership conservation

### 現行の conservation

- `ISRRetireRouter::enqueueWithRetry()` は D → Q → EQ → TerminalReclaim の順に ownership transfer を試み、いずれかが成功すれば退避成功。TerminalReclaim が growable であるため、現行は必ず成功する。
- `RetireQuarantineStore::store()` が `false` の場合、caller は deleter を実行せず、Router が次の退避先へフォールバックする。これにより UAF を構造的に排除。
- I4 D15/D18.3 の ownership conservation 式:

```text
admittedLogicalObligationCount = liveOwnershipCount + terminalDispositionCount
liveOwnershipCount = transportCount + durableCount + buildingCount + stalledCount
terminalDispositionCount = successCount + supersededCount + shutdownDiscardCount
admissionEventCount ≥ admittedLogicalObligationCount  (coalesce は event を増やすが logical obligation を増やさない)
```text

- Phase I では `supersededCount == 0`（D18.1 の不活性帰結）だが、式としては将来含めて正しい。

### Bounded 化後の conservation（D 案での再定義）

- **RuntimeWorld lifetime の conservation**は I4 の logical obligation conservation とは **別式**であることを明示する（chapter 12 で詳述）。

```text
publishedWorlds = liveWorlds + reclaimedWorlds
liveWorlds = currentWorlds (= M_current = 1)
           + readerHeldWorlds (≤ M_reader)
           + retireQueueWorlds (≤ M_retire)
           + quarantineWorlds (≤ M_quarantine)
           + terminalWorlds (≤ M_terminal)

reclaimedWorlds = drainSuccessCount (deleter 実行済み)

Invariant: liveWorlds ≤ M_world < ∞
```text

- **Bounded TerminalReclaim での conservation 条件**:

```text
1. reservation-first: 新規 publish は M_terminal slot の reservation 取得後にのみ許可
2. reservation 取得失敗 → backpressure（publish を BLOCK、coalesce 相当の重複は既存 world 再利用で回避可能か検討）
3. reservation → ownership transfer → store の間に失敗可能な操作がない（C 案の証明）
4. store は予約済みスロットへの配置であり失敗しない（allocation-free: std::array + mutex + index）
5. epoch-safe なら即時 deleter、unsafe なら pending 保持 → drain(minReaderEpoch) で safe 到達後に deleter
6. drainAll() は shutdown 時に audio thread 停止後に全 pending を強制解放（現行と同様、bounded でも成立）
```text

- **所有権が失われないことの証明スケッチ**:

```text
- 新規 world の publish 前に reservation を取得。失敗なら publish 自体を行わない → 所有権は発生しない。
- 成功なら world を build し、publishAndSwap で current 交換。旧 world の所有権は retire 経路へ移転。
- retire 経路は enqueueWithRetry() で D/Q/EQ/Terminal のいずれかに配置。Terminal は reservation 済み slot への配置であり失敗しない。
- したがって、publish された全ての world は必ずいずれかの authority に所有され、epoch safe 到達後に必ず deleter が実行される。
- 例外: build 失敗時は reservation を release し world は生成しない。所有権は発生していない。
```text

---

## 8. Reader hold bound

### 8.1 `M_reader` の問題

- `M_reader` は reader（Audio Thread の `processBlock` / `getNextAudioBlock` 等）が同時に保持する world 参照数である。
- Reader が `RuntimeStore::observe()` で `const RuntimeState*` を borrow し、block 処理中に保持する。この参照は **非所有（borrow）**であり、world の lifetime は retire 側の epoch 管理に委ねられる。
- Reader が無制限時間 world を保持できるなら、`isOlder(entry.epoch, minReaderEpoch)` が永遠に `false` のままとなり、retire 経路の drain が進行せず、`M_world` は発散する。

### 8.2 現行コードの reader hold

| 観測事実 | ソース | 判定 |
| --- | --- | --- |
| `AudioEngine` の `processBlock` は `RuntimeStore::observe()` 相当で current world を borrow し、block 終了後に参照を離す | `src/audioengine/AudioEngine.Processing.*.cpp` / `src/core/RuntimeStore.h:observe()` | block 単位の hold は **block 時間**（sampleRate / blockSize に依存）に bounded |
| `DeferredDeletionQueue::enqueue` の epoch は publish 時の `PublicationSequenceId` / `generation` | `src/DeferredDeletionQueue.h` | epoch 進行は reader の `minReaderEpoch` 進行に依存 |
| `RetireQuarantineStore::drain(minReaderEpoch, isOlderFn)` は `entry.epoch < minReaderEpoch` のみ破棄 | `src/audioengine/RetireQuarantineStore.h` | reader が epoch を進めなければ drain は停止 |
| `TerminalReclaimAuthority::drain(minReaderEpoch, isOlderFn)` も同様 | `src/audioengine/ISRRetireRouter.h` | 同上 |

### 8.3 `reader hold duration ≤ H_max` は導入可能か

- **Block 単位の hold**: 通常の `processBlock` は block 時間（例: 512 samples / 48kHz ≈ 10.7ms）に bounded。ただし、**異常系**（audio thread stall / device 停止 / suspend / デバッガ停止）では hold が indefinite に延長し得る。
- **現行で `H_max` を保証する invariant は存在しない**。D101-2 で指摘した「Reader が無制限に critical section を保持できるなら、producer rate に上限を設けても outstanding lifetime の有限性は自動的には証明できない」は依然として成立する。
- **導入可能性**: `H_max` を契約として導入するには、以下のいずれかが必要:

| 方式 | 内容 | 現行との差分 | 証明可能性 |
| --- | --- | --- | --- |
| **a. Block 時間由来の `H_max`** | `H_max = maxBlockDuration + ε`（blockSize / sampleRate から導出）。異常系は `H_max` 超過として HealthEvent / fail-safe に | 異常系の検出と fail-safe の追加が必要 | 可能（`H_max` 超過時の drain 停止を検出し、producer を backpressure で停止すれば `M_world` は発散しない） |
| **b. Epoch 進行の watchdog** | `minReaderEpoch` が `H_max` 時間進行しなければ、reader を stuck とみなし、shutdown 相当の `drainAll()` または HealthEvent に | Watchdog timer の追加が必要 | 可能（stuck reader 時の `TerminalReclaimAuthority` 滞留は `drainAll()` で強制解放可能。ただし audio thread 停止が前提） |
| **c. Reader hold の明示的 timeout** | Reader が `H_max` 超過で world を保持し続けた場合、その参照を invalid とみなす（reader 側の参照を切る） | Reader 側の参照管理の変更が必要。borrow 参照の lifetime を epoch で再検証する機構 | 可能だが、audio thread のリアルタイム性に影響するため慎重な設計が必要 |

- **本監査の結論**: `H_max` は **architectural invariant として導入可能**だが、現行コードには存在せず、**新 invariant として追加が必要**。方式 a（block 時間由来 + 異常系 fail-safe）が最も現実的であり、b/c はフォールバックとして検討可能。いずれの場合も `H_max` 超過時の producer backpressure（流入停止）が同時に必要である。

### 8.4 `H_max` と `M_reader` の関係

```text
M_reader ≤ ceil(H_max / minPublishInterval)  … ただし minPublishInterval は P_max から導出
または
M_reader ≤ maxConcurrentReaders  … reader の並行数が bounded なら
```text

- Audio Thread は単一 reader（1 block 1 world 参照）であるため、通常は `M_reader = 1`。しかし、retire 経路の drain が `H_max` 時間停止すれば、その間に publish された全ての旧 world が滞留し、`M_reader` の寄与は `H_max` に比例して増大する。
- したがって `M_reader` の有限性は `H_max < ∞` に依存する。

---

## 9. Producer activity bound

### 9.1 `A_max` / `P_max` の必要性

- `A_max` は 1 sampling interval に許される `acquire` 数（≒ publish 成功数）。`P_max` は時間窓あたりの publish activity 上限。
- Producer は複数（`AudioEngine.Commit` / `RuntimeBuilder` / `OwnerChannel` / 各種 `AudioEngine.*.cpp` の publish 経路）から駆動され、現行は **publish 頻度を制限する hard invariant が存在しない**。
- `P_max = ∞` なら、`H_max` が有限でも `M_world = P_max × H_max = ∞` となるため、producer bound は `M_world` の有限性に必須である。

### 9.2 現行の producer 制御

| 観測事実 | ソース | 判定 |
| --- | --- | --- |
| `RuntimeBuilder::buildWorld()` は `OwnerChannel` / `PendingPublishRegistry` 経由で world を生成 | `src/audioengine/RuntimeBuilder.h` / `src/audioengine/RuntimeWorldAuthority.h` | build 自体は bounded（`PendingPublishCapacity = 64`）だが、build 呼び出し頻度は unbounded |
| `RuntimeWorldAuthority::commitRuntimePublication()` 相当の publish 経路は複数 entry point から駆動 | `src/audioengine/AudioEngine.Commit.cpp` 等 | publish 呼び出し頻度は unbounded |
| I4 D14/D18.4 の coalesce は同一 `CoalesceIdentity` への重複を吸収するが、**異なる target への publish は吸収しない** | `doc/work88/I4_DESIGN_CONTRACT.md` D18.1 | 異なる target への publish は `M_world` を増加させる |

### 9.3 `A_max` / `P_max` は導入可能か

- **導入可能性**: `A_max` / `P_max` は **admission control** として導入可能である。

| 方式 | 内容 | 現行との差分 |
| --- | --- | --- |
| **a. Publish admission queue** | `M_world` の reservation-first と連動。`M_terminal` slot が枯渇すれば publish を backpressure で BLOCK | D 案の reservation-first と同一。`M_world` の budget と統合 |
| **b. Rate limiter** | 時間窓あたりの publish 回数を `P_max` に制限。超過分は coalesce または queue に滞留 | 新機構。`P_max` 超過時の coalesce 可否は `CoalesceIdentity` の一致に依存 |
| **c. Epoch-based throttle** | `minReaderEpoch` の進行速度に応じて publish 頻度を調整 | 新機構。`H_max` と連動 |

- **本監査の結論**: `A_max` / `P_max` は **architectural invariant として導入可能**だが、現行コードには存在せず、**新 invariant として追加が必要**。方式 a（reservation-first と連動した admission control）が D 案と最も整合し、第一候補である。

---

## 10. Sampling gap contract

### 10.1 現行の `maxSamplingGapUs`

- `ISRWorldRetirementTelemetry.h` の `maxSamplingGapUs` は **観測値（observed maximum）**であり、telemetry の `sample()` 間隔の最大値を記録する。
- D101-2 で禁止した `G_max = observed maxSamplingGapUs` は、観測値を契約上限として誤用するものであり、将来の観測値が契約を破る可能性があるため不成立である。

### 10.2 `G_max = G_contract` の定義

- `G_max` は **新 architectural contract** として定義する:

```text
G_contract: sampler が gap を検出した際の bounded recovery を保証する契約

Invariant: samplingGap ≤ G_max  または  gap > G_max なら boundedRecovery が発動し、
           producer admission が制御され、M_world の有限性が維持される
```text

- 具体的な契約:

| 項目 | 契約内容 |
| --- | --- |
| `G_max` の値 | `G_contract` として設計時に固定する値（例: block 時間の数倍）。観測値から導出しない |
| `missed sampling → bounded recovery` | gap が `G_max` を超過した場合、sampler は `H_max` 超過と同様に HealthEvent / fail-safe を発動し、producer を backpressure で停止する |
| `producer admission control` | gap 超過時は新規 publish を BLOCK し、既存 world の drain 完了を待つ |
| Telemetry との分離 | `maxSamplingGapUs` は引き続き telemetry として観測・記録するが、`G_contract` の根拠にはしない。両者は独立した値である |

### 10.3 証明可能性

- `G_max = G_contract` は **architectural contract として導入可能**だが、現行コードには存在せず、**新 invariant として追加が必要**。
- Gap 超過時の bounded recovery と producer admission control が同時に実装されれば、`G_max` の有限性と `M_world` の有限性を同時に保証できる。

---

## 11. `M_world` mathematical composition

### 11.1 分解式

```text
M_world = M_current
        + M_reader
        + M_retire
        + M_quarantine
        + M_terminal

where:

  M_current    = 1                          (RuntimeStore single slot, existing invariant)
  M_retire     ≤ 4096                       (DeferredDeletionQueue kQueueSize, existing invariant)
  M_quarantine ≤ 1024                       (RetireQuarantineStore 512 + EmergencyQ 512, existing invariant)
  M_terminal   ≤ K                          (TerminalReclaim bounded capacity K, NEW invariant, reservation-first)
  M_reader     ≤ f(H_max, P_max)            (reader hold bound × producer rate, NEW invariant)

Therefore:

  M_world ≤ 1 + 4096 + 1024 + K + f(H_max, P_max)

Sufficient condition for M_world < ∞:

  K < ∞  ∧  H_max < ∞  ∧  P_max < ∞  ∧  A_max < ∞  ∧  G_max = G_contract < ∞
  ∧  reservation-first holds
  ∧  ownership conservation holds (chapter 7)
  ∧  drain() eventually reclaims all epoch-safe worlds (epoch progress guaranteed by H_max)
```text

### 11.2 各項の有限性の根拠

| 項 | 有限性の根拠 | 既存/新規 | コード・契約の対応 |
| --- | --- | --- | --- |
| `M_current` | `RuntimeStore<T,Owner>` は単一 `std::atomic<T*> current`。`publishAndSwap` で旧 world は retire へ移転し current は常に 1 つのみ | 既存 | `src/core/RuntimeStore.h:WriteAccess::publishAndSwap` |
| `M_retire` | `DeferredDeletionQueue kQueueSize = 4096`。Vyukov MPMC bounded queue、満杯で `enqueue` は `false` → Quarantine へ退避 | 既存 | `src/DeferredDeletionQueue.h` |
| `M_quarantine` | `RetireQuarantineStore kMaxQuarantinedEntries = 512` ×2。`std::array` 固定容量、満杯で `false` → TerminalReclaim へ退避 | 既存 | `src/audioengine/RetireQuarantineStore.h` |
| `M_terminal` | `TerminalReclaimAuthority` を `std::vector` から `std::array<Entry, K>` + `std::mutex` + `residentAtomic_` に変更。`reservation-first` で slot 枯渇時に publish を backpressure で BLOCK。allocation-free により store 自体は失敗しない | **新規** | `src/audioengine/ISRRetireRouter.h` の P-4 growable を bounded に置換（実装は D101-4 以降） |
| `M_reader` | `M_reader ≤ f(H_max, P_max)`。`H_max` は block 時間由来 + 異常系 fail-safe で導入。`H_max` が有限なら `M_reader` も有限 | **新規** | chapter 8 で `H_max` の導入可能性を検証 |
| — | `G_max = G_contract` は `M_world` の直接の項ではないが、`P_max` / `H_max` の前提となる sampler gap の契約 | **新規** | chapter 10 で `G_max` の契約定義を検証 |

### 11.3 流入・流出の観点からの再表現

```text
流入:  publish rate ≤ P_max  ∧  1 interval acquire ≤ A_max  ∧  reservation-first で M_terminal slot を確保
流出:  epoch 進行速度 ≥ 1/H_max  →  drain(minReaderEpoch) で epoch-safe worlds を reclaim
       流出速度の下界は H_max により保証

M_world(t) の時間発展:

  d/dt M_world(t) ≤流入(t) - 流出(t)

  流入が P_max で上界され、流出が H_max で下界されるため、
  定常状態で M_world(t) ≤ 1 + 4096 + 1024 + K + f(H_max, P_max) < ∞

  異常系（H_max 超過 / G_max 超過）では流入を backpressure で停止するため、
  M_world は発散しない（流入停止中に流出が追いつく）。
```text

---

## 12. I4 D14/D18 compatibility

### 12.1 Budget の分離（最重要）

| Budget | 対象 | I4 での定義 | D101-3 での定義 | 混同の禁止 |
| --- | --- | --- | --- | --- |
| **Recovery obligation budget** | Logical recovery obligation（`RecoveryEpisodeId` + `SemanticRecoveryTarget` 単位） | `kMaxLogicalRecoveryObligations`（候補 32）。D14 の reservation-first: 新規 logical obligation 生成に対して exactly once（D18.4）。COALESCE は新規 reservation を取得しない。Budget 枯渇 → backpressure | 本監査の対象ではない。I4 の定義をそのまま維持 | `Recovery obligation budget ≠ RuntimeWorld lifetime budget` を契約で明示する |
| **RuntimeWorld lifetime budget** | `RuntimeWorld`（≒ `RuntimeState`）の outstanding 数 | I4 では未定義 | `M_world` として本監査で新設。`M_world = M_current + M_reader + M_retire + M_quarantine + M_terminal`。流入は `A_max` / `P_max`、流出は `H_max` で制御。TerminalReclaim は `M_terminal ≤ K` で reservation-first | 両 budget は **別 invariant** として契約化する。数値も機構も共有しない |

### 12.2 D14 reservation-first の適用

| I4 D14 の reservation-first | D101-3 の reservation-first |
| --- | --- |
| 新規 logical obligation の生成時に `kMaxLogicalRecoveryObligations` budget から reservation を取得。COALESCE は新規 reservation を取得しない（D18.4）。budget 枯渇 → backpressure（新規生成を BLOCK）。terminal-failure 撤廃 | 新規 `RuntimeWorld` の publish 時に `M_terminal` budget（capacity `K`）から reservation（slot）を予約。reservation 取得失敗 → backpressure（新規 publish を BLOCK）。`store()` 自体は予約済み slot への配置であり失敗しない |

- **パターンは同一**: `reservation → ownership transfer → store` の間に失敗可能な操作がないことを証明し、失敗は reservation 取得時点に前倒しする。
- **Budget は分離**: 数値（32 vs `K`）も、対象（logical obligation vs `RuntimeWorld`）も、機構（admission 時の reservation 取得）も別である。混同しない。

### 12.3 D18.4 の「新規 logical obligation 生成に対して exactly once」の適用

- D18.4 では `COALESCE: 新しい reservation を取得しない` と修正された。これは recovery obligation budget に関する修正である。
- RuntimeWorld lifetime budget でも同様の考慮が必要: **同一 `CoalesceIdentity` への重複 publish が `RuntimeWorld` の新規生成を伴わない**場合、その publish は `M_world` を増加させない。逆に、異なる target への publish は新規 `RuntimeWorld` を生成し `M_world` を増加させる。
- したがって `A_max` / `P_max` の counting は **新規 `RuntimeWorld` 生成数** で行い、coalesce 相当の重複は `M_world` の増加に含めない。ただし `RuntimeWorld` の生成と publish は I4 の logical obligation とは **別レイヤー**であるため、両者の coalesce 条件は独立に定義する。

### 12.4 その他 D18 との整合

| D18 項目 | D101-3 との関係 |
| --- | --- |
| D18.1 Coalesce/Supersede 境界 | Phase I で SUPERSEDE は不活性。`M_world` の流入抑制は coalesce による吸収に依存するが、異なる target への publish は吸収されないため `P_max` による流入制御が別途必要 |
| D18.2 Phase I supersession | 全値等価のみが supersession。`M_world` の流入側では異なる target は別 world として counting する |
| D18.3 Ownership conservation | `admittedLogicalObligationCount = liveOwnershipCount + terminalDispositionCount` は logical obligation の conservation。D101-3 の `publishedWorlds = liveWorlds + reclaimedWorlds` は別式として定義する（chapter 7） |
| D18.5 D13〜D17 改訂 | Reservation-first は D18.4 の「新規生成に対して exactly once」に限定。D101-3 でも同様に `M_terminal` の reservation は新規 `RuntimeWorld` publish に対してのみ取得する |

---

## 13. Required architectural invariants

D101-3 の契約を充足し `M_world < ∞` を証明するには、以下の新 invariant が必須である。いずれも現行コード・現行 I4 契約には存在せず、**architectural change として導入が必要**である。

| # | Invariant | 内容 | 現行 | 導入方法 |
| --- | --- | --- | --- | --- |
| 1 | `M_terminal ≤ K < ∞` | `TerminalReclaimAuthority` を `std::vector` から `std::array<Entry, K>` + `std::mutex` + index 配置に変更。`store()` は失敗しないが、reservation が前提 | 不存在（`std::vector` growable） | `ISRRetireRouter.h` の P-4 を bounded に置換。`K` の値は `M_world` の分解から導出（例: `K = M_world - (1+4096+1024) - M_reader`） |
| 2 | `M_terminal reservation-first` | 新規 `RuntimeWorld` publish 前に `M_terminal` slot を予約。失敗 → backpressure（BLOCK）。`reservation → ownership transfer → store` の間に失敗可能な操作がない | 不存在 | D 案の reservation-first。`std::array` + index で allocation-free に |
| 3 | `M_reader ≤ f(H_max, P_max) < ∞` | Reader hold の並行度を `H_max` と `P_max` から上界する | 不存在 | `H_max` の導入後に導出 |
| 4 | `H_max < ∞` | Reader hold / reclaim latency の契約上限。`H_max = maxBlockDuration + ε` + 異常系 fail-safe（watchdog / HealthEvent） | 不存在 | chapter 8 の方式 a（block 時間由来 + fail-safe）を第一候補として実装 |
| 5 | `A_max < ∞` | 1 sampling interval の acquire 上限 | 不存在 | Reservation-first と連動した admission control |
| 6 | `P_max < ∞` | 時間窓あたりの publish activity 上限 | 不存在 | Reservation-first と連動した admission control / rate limiter |
| 7 | `G_max = G_contract < ∞` | Sampler gap の契約上限。`observed maxSamplingGapUs` から導出しない。新 contract として固定し、gap 超過時は bounded recovery + producer admission control | 不存在 | chapter 10 の `G_contract` 定義。telemetry と分離 |
| 8 | `RuntimeWorld lifetime budget` の明示 | `logical-obligation budget ≠ RuntimeWorld lifetime budget` を契約で分離。両者は別 invariant、別数値、別機構 | 不存在（I4 は前者のみ） | 本監査の契約定義を I4 に追記 |
| 9 | `Ownership conservation (RuntimeWorld)` | `publishedWorlds = liveWorlds + reclaimedWorlds`。Bounded 化後も全 publish された world は必ずいずれかの authority に所有され、epoch safe 到達後に必ず deleter が実行される | 現行は growable 前提で成立。bounded 化で再証明が必要 | D 案の証明スケッチ（chapter 7）を実装時に検証 |
| 10 | `Shutdown Drain → Reclaim → Verify Empty` の bounded 下での成立 | Bounded `M_terminal` でも shutdown 時の `drainAll()` が全 pending を解放し Verify Empty が成立すること | 現行は growable で成立。bounded 化で `drainAll()` の capacity が shutdown 時の残留数を上回ることの保証が必要 | `M_world` の上界と同一の `K` で保証される（循環依存に注意。`K` は shutdown 時の worst-case 残留数を上回る必要がある） |

---

## 14. Verdict

### 判定: `CONTRACT_REQUIRES_NEW_INVARIANT`

| 判定 | 定義 | 本監査の該当性 |
| --- | --- | --- |
| `CONTRACT_SUFFICIENT` | 既存 architecture + 既存 I4 契約だけで `M_world < ∞` を証明できる | **該当せず** — 上記 10 の新 invariant がいずれも現行には存在しない |
| `CONTRACT_REQUIRES_NEW_INVARIANT` | 契約の骨格は本監査で定義可能だが、有限性を証明するには新 invariant の導入が必須 | **◯ 該当（本監査の結論）** |
| `STRUCTURALLY_UNPROVABLE` | Practical Stable ISR の「overflow しても所有権を失わない」要求と finite hard cap が両立しないため、`M_world < ∞` をこの architecture では要求できない | **該当せず** — D 案（unified lifetime budget + reservation-first + backpressure）により finite cap と ownership preservation は両立可能。RT 非関与の publish 停止 + bounded reader hold により原理的に両立する |

### なぜ `CONTRACT_REQUIRES_NEW_INVARIANT` か

- **骨格は定義できた**: `M_world` の分解（`M_current + M_reader + M_retire + M_quarantine + M_terminal`）、各 bound の分離（`M_world / M_terminal / A_max / P_max / H_max / G_max`）、I4 との budget 分離、TerminalReclaim の修復モデル（D 案を第一候補）、ownership conservation の再定義は、本監査で契約として固定できた。
- **しかし新 invariant がなければ証明不能**: `M_terminal` の bounded 化、`H_max` / `A_max` / `P_max` / `G_contract` の導入、`RuntimeWorld lifetime budget` の明示、bounded 下での ownership conservation の再証明は、いずれも現行コード・現行 I4 契約には存在しない。これらを architectural change として導入しなければ `M_world < ∞` は証明できない。
- **`STRUCTURALLY_UNPROVABLE` ではない理由**: D101-2 と同様、finite hard cap と ownership preservation は backpressure で両立可能。growable sink を維持しなければ所有権を失うのは drop 方式（A 案）を前提にした場合にのみ真であり、reservation-first（C/D 案）では両立する。

---

## 15. Next gate

```text
D101-2  FINITE_BOUND_REQUIRES_ARCHITECTURAL_CHANGE
   │
   ▼
D101-3  CONTRACT_REQUIRES_NEW_INVARIANT  ◀ 本監査
   │
   ├── M_world 分解定義
   ├── I4 D14/D18 接続（budget 分離）
   ├── TerminalReclaim D 案（unified lifetime budget）を第一候補に固定
   ├── H_max / A_max / P_max / G_contract の新 invariant を列挙
   └── 10 の Required architectural invariants を明示
   │
   ▼
I4 Contract 更新
   │
   ├── logical-obligation budget ≠ RuntimeWorld lifetime budget の明示
   ├── M_world 分解と各 bound の契約定義
   ├── TerminalReclaim bounded + reservation-first の設計
   ├── H_max / A_max / P_max / G_contract の invariant 追加
   └── ownership conservation (RuntimeWorld) の再定義
   │
   ▼
D101-4 — Capacity derivation（数値導出）
   │
   ├── M_world / K / H_max / A_max / P_max / G_max の具体的数値導出
   ├── M_world の証明（M_world < ∞ の形式的証明）
   └── backpressure progress の証明
   │
   ▼
Phase I GO/NO-GO 再判定
   │
   ├── D101-4 で capacity derivation と backpressure progress が証明されれば GO
   └── そうでなければ NO-GO 継続
```text

- **本監査でも production code は変更しない**（指示どおり）。
- I4 Contract 更新では、本監査の 10 の Required architectural invariants を I4 の D14/D18 と整合させて追記する。
- D101-4 では、本監査で定義した契約骨格に基づき、具体的数値（`M_world / K / H_max / A_max / P_max / G_max`）を導出し、`M_world < ∞` の形式的証明と backpressure progress の証明を行う。D101-4 でも `G_max = observed maxSamplingGapUs` は禁止である。

---

## 付録: D101-3 監査チェックリスト

- [x] Finite-bound の対象を分離（`M_world / M_terminal / A_max / P_max / H_max / G_max` を別々に定義）
- [x] Queue capacity（4096/512/512）を `M_world` と同一視しない
- [x] I4 D14 の reservation/backpressure と接続（`logical-obligation budget ≠ RuntimeWorld lifetime budget` を混同しない）
- [x] I4 D18.4 の「新規 logical obligation 生成に対して exactly once」（COALESCE は新規 reservation を取得しない）を D101-3 に反映
- [x] TerminalReclaim 修復モデル A/B/C/D を比較（A=NO-GO / B=条件付き / C=有力候補 / D=第一候補）
- [x] Ownership conservation の検証（I4 の式と RuntimeWorld の式を分離、bounded 化後の conservation 条件を定義）
- [x] `M_world` の分解（`M_current + M_reader + M_retire + M_quarantine + M_terminal`）と各項の有限性根拠を提示
- [x] `M_reader` の有限性が `H_max` に依存することを明示し、`reader hold duration ≤ H_max` の導入可能性を調査
- [x] Producer activity bound（`A_max / P_max`）の必要性と導入可能性を検証
- [x] `G_max` を telemetry（`maxSamplingGapUs`）から作らない（`G_contract` として新定義、missed sampling → bounded recovery → admission control）
- [x] 最終 verdict を 3択（`CONTRACT_SUFFICIENT / CONTRACT_REQUIRES_NEW_INVARIANT / STRUCTURALLY_UNPROVABLE`）で確定
- [x] Production code 変更なし（契約定義のみ）
- [x] 15章構成（Scope / D101-2引継ぎ / Bound definitions / Existing invariant / Missing invariant / TerminalReclaim A/B/C/D / Ownership conservation / Reader hold / Producer / Sampling gap / M_world composition / I4 compatibility / Required invariants / Verdict / Next gate）
