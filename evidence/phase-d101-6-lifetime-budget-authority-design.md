# D101-6 — Lifetime Budget Authority Design

| 項目 | 内容 |
| --- | --- |
| **ID** | D101-6 |
| **日付** | 2026-08-20 |
| **対象ブランチ** | `main` |
| **一次資料** | `ConvoPeq.md`（2026-08-19 全ソース連結）、`src/audioengine/RuntimeWorldAuthority.h`、`src/audioengine/ISRRuntimePublicationCoordinator.h`、`src/audioengine/ISRRetireRouter.h/.cpp`、`src/audioengine/RetireQuarantineStore.h`、`src/DeferredDeletionQueue.h`、`src/core/RuntimeStore.h`、`src/core/EpochDomain.h`、`src/audioengine/RuntimeBuilder.h`、`src/audioengine/AudioEngine.Commit.cpp`、`src/audioengine/AudioEngine.Retire.cpp`、`src/audioengine/AudioEngine.Timer.cpp`、`src/audioengine/ISRWorldRetirementTelemetry.h`、`doc/work88/I4_DESIGN_CONTRACT.md`、D101-3/D101-4/D101-5 evidence |
| **前提** | D101-5 verdict: **ARCHITECTURAL_CHANGE_REQUIRED** — 実装順序は案B（`I08 lifetime budget authority → I02 reservation → I09 ownership conservation → ... → I10 M_world formal bound`）。本監査は案Bの Phase 1 を具体設計する |
| **目的** | I08 の Lifetime Budget Authority（`A_max / P_max` を含む）を、既存コードの責務境界を壊さずに定義する。既存の `reservation → push → rollback` パターンを置き換えるのではなく、一本の conservation chain として再構成できるかを検証する |
| **制約** | **コード変更なし・設計監査のみ**（17章構成）。`ConvoPeq.md` を一次資料とする。Budget unit / Capacity / Admission / Rate / Reservation / Transfer / Rollback / Release / Conservation の各要素を確定し、既存 reservation counter を Lifetime Budget Authority とみなせるかを厳密に判定する |
| **判定** | **DESIGN_REQUIRED** — Lifetime Budget Authority の骨格（Budget unit / Authority 所在 / Conservation chain / Enforcement point）は本監査で定義可能だが、具体的 state machine / contract（D101-7 の対象）を新設計として導入する必要がある。既存 reservation counter はそのまま Lifetime Budget Authority とはみなせない |

---

## 1. Scope

- D101-5 で確定した実装順序（案B）の Phase 1 として、Lifetime Budget Authority を設計する。
- 既存の Publish Intent の `reservation → push → rollback` パターンを、別系統の新機構に置き換えるのではなく、一本の conservation chain として再構成できるかを検証する。
- 本監査はコード変更なしで、既存機構の調査 → Budget 定義 → Conservation 設計 → Authority mapping → Failure/Shutdown 検証の順で進める。
- 判定は `SATISFIED / PARTIAL / MISSING / CONFLICT / DESIGN_REQUIRED` の 5 段階。

---

## 2. D101-5 からの入力条件

| 入力 | 内容 | 本監査での扱い |
| --- | --- | --- |
| Implementation Order | 案B: `I08 → I02 → I09 → I10` の順序が正しい | I08（budget separation）を最初に設計する |
| D101-3 の 10 invariants | I01〜I10 の Required architectural invariants | I08 を本監査で具体設計、残りは D101-7 以降で順次設計 |
| D101-4 の適合性 | I08 は MISSING（I4 への追記が未実施） | I4 追記 + Lifetime Budget Authority の新設を設計する |
| D 案（unified lifetime budget） | `publish admission → reservation → ownership transfer → retire → release` を一つの chain として扱う | 既存 `reservation → push → rollback` を chain として再構成できるかを検証 |
| Authority mapping | D101-5 で I01〜I10 の Authority/State/Enforcement boundary を確定 | I08 の Authority を本監査で具体化する |

---

## 3. Existing reservation mechanisms

### 3.1 Publish Intent の `reservation → push → rollback`

現行 `ConvoPeq.md` / `src/audioengine/ISRRuntimePublicationCoordinator.h` に存在する既存パターンを整理する。

| 要素 | 現行の実装 | 所在 |
| --- | --- | --- |
| **Reservation** | `enqueuePublicationIntent()` が `push` 前に `pendingIntentCount_.fetchAdd(1)`（reservation-before-push）。`work88 P2-1 §1.1.3` で導入 | `ISRRuntimePublicationCoordinator.h` / `ConvoPeq.md:55921` |
| **Push** | `intentQueue_.push()` / `deferred ring push` の全層 push。成功すれば reservation を維持 | `ConvoPeq.md:55939` |
| **Rollback** | 全層 push 失敗（`intentQueue_ + deferred ring` 溢れ）→ `pendingIntentCount_.fetchSub(1)` で rollback + drop カウンタ | `ConvoPeq.md:56143` |
| **Counter** | `pendingIntentCount_` は `std::atomic<int>`。reservation-before-push で `fetchAdd` → push 成功で維持 → push 失敗で `fetchSub` rollback | `ConvoPeq.md:40350`「`pendingIntentCount_` は reservation ベース（push 成功 fetchAdd / pop 成功 fetchSub）」 |
| **Pop 側** | `popRecoveryRequest` で reservation 消費（pop 成功数 == push 成功数の不変条件） | `ConvoPeq.md:94462` |
| **Publication residency** | `publicationIntentResidencyCount_` は `enqueuePublicationIntent` の `reservation → push → rollback` で管理。`intentQueue_` 内の Publish Intent 数 + producer 側の状態を追う | `ConvoPeq.md:56674` |
| **Shutdown** | Shutdown 確定後は Publish Intent を `Path B admission gate` で拒否（`ConvoPeq.md:56686`） | `ConvoPeq.md:56686` |

### 3.2 既存 reservation の特性

| 特性 | 内容 | Lifetime Budget との関係 |
| --- | --- | --- |
| **対象** | `pendingIntentCount_` は Intent（Publish/Recovery/Quarantine）の reservation であり、RuntimeWorld の lifetime budget ではない | 対象が異なる。Intent の reservation ≠ RuntimeWorld の reservation |
| **粒度** | 1 Intent = 1 reservation（`pendingIntentCount_` の単位は Intent） | RuntimeWorld の単位は world（`RuntimeState`）であり、粒度が異なる |
| **Enforcement** | Queue full 時に rollback するが、backpressure（BLOCK）ではない。Push 失敗 → rollback → drop | Lifetime Budget は backpressure（BLOCK）を要求する。rollback の意味が異なる |
| **Authority** | `ISRRuntimePublicationCoordinator` が `pendingIntentCount_` を所有 | Lifetime Budget Authority は `RuntimeWorldAuthority` または新 Authority が所有すべき |
| **Scope** | `pendingIntentCount_` は Intent queue の capacity 管理に限定される | Lifetime Budget は `M_world` 全体の budget であり、Scope が広い |

### 3.3 既存 reservation を Lifetime Budget Authority とみなせるか

**判定**: みなせない（CONFLICT ではないが、PARTIAL でもない — 対象・粒度・Scope が異なるため、流用ではなく新設が必要）

| 検証観点 | 既存 reservation | Lifetime Budget の要求 | みなせるか |
| --- | --- | --- | --- |
| 対象（Budget unit） | Intent（Publish/Recovery/Quarantine） | RuntimeWorld（`RuntimeState`） | ❌ 対象が異なる |
| 粒度 | 1 Intent = 1 count | 1 RuntimeWorld = 1 budget unit | ❌ 粒度が異なる |
| Scope | Intent queue の capacity | `M_world` 全体の lifetime budget | ❌ Scope が異なる |
| Mechanism | `fetchAdd → push → fetchSub rollback`（queue 溢れ対応） | `reserve → transfer → release`（lifetime 全体 + backpressure） | △ パターンは類似するが、backpressure の意味が異なる |
| Authority | `ISRRuntimePublicationCoordinator` | `RuntimeWorldAuthority` または新 Authority | ❌ Authority が異なる |

**結論**: 既存の `reservation → push → rollback` パターンは **概念的に参考になる**（reservation-before-push の ordering は流用可能）が、**そのまま Lifetime Budget Authority とはみなせない**。Budget unit / Authority / Scope が全て異なるため、新設が必要である。ただし、パターン自体（reservation-before-push + push 失敗時の rollback）は D101-7 での新設計に流用可能である。

---

## 4. Existing ownership states

### 4.1 現行の ownership 状態

| 状態 | 所在 | Owner | 内容 |
| --- | --- | --- | --- |
| **Build** | `RuntimeBuilder::buildWorld()` | `RuntimeBuilder`（一時所有） | `RuntimeState` を構築し、seal 後に OwnerChannel / PendingPublishRegistry へ移転 |
| **OwnerChannel** | `RuntimeWorldAuthority::OwnerChannelType` | `RuntimeWorldAuthority`（value 所有） | `aligned_unique_ptr<const RuntimeState>` を `OwnerChannel` に保持。`take()` が唯一の Owner-consumption point |
| **PendingPublishRegistry** | `RuntimeWorldAuthority::PendingPublishRegistry` | `RuntimeWorldAuthority`（value 所有） | `kPendingPublishCapacity=64`。enqueue→commit の async gap を `registerPublish(seqId, world)` / `lookup(seqId)` / `unregister(seqId)` で管理。Lock-free |
| **RuntimeStore::current** | `RuntimeWorldAuthority::Store runtimeStore_` | `RuntimeWorldAuthority`（value 所有、CRTP Owner） | `std::atomic<RuntimeState*> current`（単一スロット）。`WriteAccess::publishAndSwap()` で旧 world を retire へ |
| **Retire** | `ISRRetireRouter`（`RetireQuarantineStore` / `DeferredDeletionQueue` / `TerminalReclaimAuthority`） | `ISRRetireRouter` | `enqueueWithRetry()` で D→Q→EQ→TerminalReclaim の順に ownership transfer |
| **Reader** | `AudioEngine.Processing`（Audio Thread） | `AudioEngine`（borrow） | `RuntimeStore::observe()` で borrow（非所有）参照。block 終了後に参照を離す |
| **Shutdown** | `AudioEngine.ReleaseResources` / `ISRRetireRouter::drainAll()` | `AudioEngine` | drainAll() で all pending を強制解放、Verify Empty |

### 4.2 現行の lifetime 状態遷移

```text
Build (RuntimeBuilder)
  → OwnerChannel (RuntimeWorldAuthority, aligned_unique_ptr)
  → PendingPublishRegistry (seqId keyed, 64 slots)
  → PublishExecutor::executePublish → authority.ownerChannel().take(seqId)
  → authority.commit() → RuntimeStore::publishAndSwap()
  → RuntimeStore::current (single slot, current world)
       │
       └─→ old world → ISRRetireRouter::enqueueWithRetry()
                        → DeferredDeletionQueue(4096) → RetireQuarantineStore(512)
                        → EmergencyQ(512) → TerminalReclaimAuthority(std::vector, ∞)
       │
       └─→ reader borrow (Audio Thread, non-owning)
             → epoch 進行 → drain(minReaderEpoch, isOlder) → deleter → reclaim
```

### 4.3 現行の問題点（Lifetime Budget 観点）

| 問題 | 現行の状態 | Lifetime Budget との関係 |
| --- | --- | --- |
| Publish 前の reservation なし | Publish 前に `M_terminal` slot の reservation を取得しない | `M_world` の流入側が unbounded |
| PendingPublishRegistry 64 は gap 容量のみ | Gap の bounded 性を示すが、publish 頻度の上限（P_max）ではない | A_max / P_max が MISSING |
| TerminalReclaim growable | `std::vector` により `M_terminal = ∞` | `M_world = ∞` |
| Reader hold 上限なし | Epoch 進行の保証がない | H_max が MISSING |

---

## 5. Lifetime state machine

### 5.1 新 Lifetime Budget の状態機械（D101-7 で詳細設計する骨格）

```text
                    ┌──────────────┐
                    │  Budget Pool  │  K_world slots (finite)
                    │  (free slots) │
                    └──────┬───────┘
                           │ reserve(K_world slot)
                           ▼
                    ┌──────────────┐
                    │   Reserved    │  1 slot reserved for this publish
                    │  (admission)  │
                    └──────┬───────┘
                           │ build + seal
                           ▼
                    ┌──────────────┐
                    │  OwnerChannel │  aligned_unique_ptr<RuntimeState>
                    │  (pending)    │
                    └──────┬───────┘
                           │ PendingPublishRegistry (seqId)
                           ▼
                    ┌──────────────┐
                    │  Commit gap   │  async gap (64 slots, existing)
                    │  (registered) │
                    └──────┬───────┘
                           │ PublishExecutor → authority.commit() → publishAndSwap
                           ▼
                    ┌──────────────┐
                    │   Current     │  RuntimeStore::current (1 slot)
                    │  (published)  │
                    └──────┬───────┘
                           │ publishAndSwap evicts old world
                           ▼
                    ┌──────────────┐
                    │    Retire     │  DeferredDeletionQueue(4096)
                    │  (queued)     │
                    └──────┬───────┘
                           │ enqueueWithRetry overflow
                           ▼
                    ┌──────────────┐
                    │  Quarantine   │  RetireQuarantineStore(512) + EmergencyQ(512)
                    │  (quarantined)│
                    └──────┬───────┘
                           │ overflow (bounded TerminalReclaim with reservation)
                           ▼
                    ┌──────────────┐
                    │   Terminal    │  TerminalReclaimAuthority<K>(K slots, bounded)
                    │  (terminal)   │
                    └──────┬───────┘
                           │ drain(minReaderEpoch, isOlder) — H_max 保証
                           ▼
                    ┌──────────────┐
                    │   Reclaimed   │  deleter executed → Budget release
                    │  (released)   │
                    └──────┬───────┘
                           │ free slot returns to Budget Pool
                           ▼
                    ┌──────────────┐
                    │  Budget Pool  │  slot returns
                    └──────────────┘
```

### 5.2 失敗パス（Rollback / Shutdown）

```text
Reserved ──build fail──→ Budget release (reservation cancel, no ownership)
OwnerChannel ──publish fail──→ Budget release (owner discarded, slot freed)
Pending gap ──stale/shutdown──→ Budget release (registry entry dropped)
Retire/Quarantine/Terminal ──shutdown──→ drainAll() → Budget release (all pending reclaimed)
```

### 5.3 既存 `reservation → push → rollback` との対応

| 既存パターン | 新 Lifetime Budget での対応 | 差異 |
| --- | --- | --- |
| `reservation` (fetchAdd) | `reserve(K_world slot)` (Budget Pool から slot 取得) | Budget Pool は `K_world` 全体の budget。既存は Intent queue の reservation |
| `push` (intentQueue push) | `publish` (publishAndSwap + retire enqueue) | Push 対象が Intent → RuntimeWorld に変わる |
| `rollback` (fetchSub) | `release` (Budget Pool へ slot 返却) | Backpressure 時の rollback ではなく、lifetime 終了時の release。失敗時の cancel も含む |
| `pop` (reservation 消費) | `drain` (epoch-safe 到達後の deleter) | Pop は queue 消費、drain は epoch 進行待ち |

**結論**: パターン（reservation-before-push + 失敗時 rollback）は流用可能だが、対象・Authority・Scope が異なるため、新 Lifetime Budget としての再設計が必要である。

---

## 6. Budget unit definition

### 6.1 Budget unit の定義

| 候補 | 内容 | 採用 |
| --- | --- | --- |
| `RuntimeWorld` (≒ `RuntimeState`) | 1 publish で生成される `RuntimeState` 1 個を 1 unit とする | **✅ 採用** |
| `publication` | 1 publish 操作（`commitRuntimePublication` 呼び出し）を 1 unit とする | ❌ publication 失敗時は world が生成されないため、unit と world が 1:1 でない |
| `logical obligation` | I4 の `kMaxLogicalRecoveryObligations` の単位 | ❌ I4 は logical obligation の budget であり、RuntimeWorld の budget とは混同しない（D101-3 chapter 12） |

### 6.2 なぜ `RuntimeWorld` を 1 unit とするか

- `RuntimeWorld`（`RuntimeState`）は `RuntimeStore::current` に 1 個、retire 経路に複数個滞留する物理的なオブジェクトであり、`M_world` の counting 対象そのものである。
- I4 の `logical obligation` は `RecoveryEpisodeId + SemanticRecoveryTarget` 単位であり、`RuntimeWorld` とは異なる。I4 では `COALESCE` で同一 `CoalesceIdentity` への重複が吸収されるが、`RuntimeWorld` は publish ごとに新規生成される可能性がある。
- D101-3 でも `RuntimeWorld lifetime budget` は `M_world` として定義し、`logical-obligation budget` とは別 invariant として分離した。

### 6.3 Budget capacity `K_world`

```text
K_world = M_world の有限値（D101-3 chapter 11 の M_world 分解式の上界）

M_world ≤ 1 + 4096 + 1024 + K + f(H_max, P_max)

ここで K = M_terminal（TerminalReclaim の bounded capacity）
K_world の値は D101-7 で具体的数値を導出するが、本監査では K_world < ∞ の invariant として定義する
```

- `K_world` は `M_world` の上界そのものであり、`Budget Pool` の capacity である。
- `K`（`M_terminal`）は `K_world` の一部であり、`K_world = M_terminal + (1 + 4096 + 1024 + M_reader)` の関係にある。

---

## 7. `A_max` definition

### 7.1 `A_max` の定義

```text
A_max: 1 sampling interval に許される acquire 数（= publish 成功数）の上限

Invariant: acquire(interval) ≤ A_max < ∞

where:
  interval = ISRWorldRetirementTelemetry の sampling interval（既存の sampler 間隔）
  acquire = RuntimeWorldAuthority::publish 成功（RuntimeStore::publishAndSwap 成功）
```

### 7.2 `A_max` の意味と enforcement point

| 項目 | 内容 |
| --- | --- |
| **意味** | 1 interval に publish 成功できる RuntimeWorld 数の上限。Burst 時の `M_world` 一時的発散を上界する |
| **Enforcement point** | `RuntimeWorldAuthority::commitRuntimePublication()` の publish 境界。1 interval 内の publish 成功数をカウントし、`A_max` 超過で backpressure（BLOCK） |
| **Existing code** | `ISRWorldRetirementTelemetry` は観測専用（D76.4「T1 telemetry is observational, not reservation authority」）。`A_max` の制御は存在しない |
| **I4 との関係** | I4 の `kMaxLogicalRecoveryObligations` は異なる budget。`A_max` は RuntimeWorld の interval 単位の上限であり、I4 の budget とは混同しない |
| **D101-3 との関係** | D101-3 chapter 9 で `A_max` を producer bound として定義。本監査で Authority mapping を確定 |

### 7.3 なぜ `A_max` が必要か

- `A_max` がなければ、1 interval 内に無制限の publish が成功し、`M_world` が一時的に `∞` に発散し得る。
- I4 の coalesce は同一 `CoalesceIdentity` への重複を吸収するが、異なる target への publish は吸収しない。

### 7.4 `A_max` の値の導出（D101-7 で詳細化）

- `A_max` は `M_world` の分解式の `M_reader = f(H_max, P_max)` とは異なる粒度（interval 単位 vs 時間窓単位）。
- D101-7 で具体的数値を導出するが、本監査では `A_max < ∞` の invariant として定義する。

---

## 8. `P_max` definition

### 8.1 `P_max` の定義

```text
P_max: 時間窓（例: 1秒）あたりの publish activity 上限（publish 成功数の上限）

Invariant: publish(window) ≤ P_max < ∞

where:
  window = 固定時間窓（例: 1秒、H_max の数倍）
  publish = RuntimeWorldAuthority::publish 成功
```

### 8.2 `P_max` と `PendingPublishRegistry 64` の分離

| 項目 | 内容 | 関係 |
| --- | --- | --- |
| `PendingPublishRegistry kPendingPublishCapacity=64` | enqueue→commit の async gap を bounded にする。Lock-free、Non-RT → ISR/audio thread 間 | Gap の容量。`P_max` の代わりにはならない |
| `P_max` | 時間窓あたりの publish 成功数の上限 | Publish 頻度の上限。Gap 容量とは異なる |

- `PendingPublishRegistry 64` は **「同時に gap に存在できる world 数」**の上限であり、**「時間窓あたりの publish 回数」**の上限ではない。
- 例えば、gap が 64 でも、gap の滞留時間が短ければ、1秒間に 64 を超える publish が可能である（gap を通過して retire へ流れるため）。
- したがって `P_max` は `PendingPublishRegistry 64` とは別に定義する必要がある。

### 8.3 `P_max` の意味と enforcement point

| 項目 | 内容 |
| --- | --- |
| **意味** | 時間窓あたりの publish 成功数の上限。`M_world` の流入側を `P_max × H_max` で上界する |
| **Enforcement point** | `RuntimeBuilder::buildWorld()` / `RuntimeWorldAuthority::publish` の呼び出し境界。時間窓内の publish 成功数をカウントし、`P_max` 超過で backpressure（BLOCK） |
| **Existing code** | `RuntimeBuilder::buildWorld()` の呼び出し頻度に hard limit は存在しない。`PendingPublishRegistry 64` は P_max の代わりにならない |
| **I4 との関係** | I4 の budget は異なる対象。P_max は RuntimeWorld の時間窓単位の上限 |
| **H_max との関係** | `M_world(t) ≤ ceil(P_max × H_max) + pipeline_depth`。H_max がなければ P_max 単独では M_world を上界できない |

### 8.4 なぜ `P_max` が必要か

- `P_max = ∞` なら、`H_max` が有限でも `M_world = ∞`（無制限 publish が H_max 時間滞留すれば発散）。
- `H_max` と `P_max` の積が `M_world` の上界を決定する。

---

## 9. Reservation / transfer / rollback / release

### 9.1 Reservation

```text
Reservation: 新規 RuntimeWorld publish 前に Budget Pool から K_world slot を 1 個予約する

Invariant: reservation は ownership transfer の前に取得する

Mechanism:
  1. producer が RuntimeWorld publish を要求
  2. Lifetime Budget Authority が Budget Pool から空き slot を 1 個予約（atomic fetchAdd 相当）
  3. 成功 → publish を許可（build → OwnerChannel → PendingPublishRegistry → commit）
  4. 失敗（K_world slots 全占有）→ backpressure（BLOCK、publish を拒否）

Enforcement point: RuntimeWorldAuthority / RuntimeBuilder の publish 境界（admission gate）
```

- I4 D14/D18.4 と同型: `reservation-first` は「新規 logical obligation 生成に対して exactly once」→ RuntimeWorld でも「新規 RuntimeWorld publish に対して exactly once」。
- `COALESCE` 相当の重複 publish は新規 reservation を取得しない（I4 D18.4 と同様）。

### 9.2 Transfer（publish → owner channel への所有権移転）

```text
Transfer: 予約済み Budget slot に対応する RuntimeWorld の所有権を、
          RuntimeBuilder → OwnerChannel → PendingPublishRegistry → RuntimeStore::current へ移転する

Invariant: reservation → transfer → retire の間に失敗可能な操作がない

Mechanism:
  1. RuntimeBuilder::buildWorld() で RuntimeState を構築・seal
  2. RuntimeWorldAuthority::OwnerChannel に aligned_unique_ptr<RuntimeState> として配置
  3. PendingPublishRegistry::registerPublish(seqId, world) で gap に登録
  4. PublishExecutor::executePublish で authority.ownerChannel().take(seqId) → authority.commit() → publishAndSwap()
  5. 旧 world は ISRRetireRouter::enqueueWithRetry() で retire 経路へ（D→Q→EQ→TerminalReclaim[reserved slot]）

Enforcement point: RuntimeWorldAuthority::publishAndSwap()（CRTP WriteAccess の唯一の publish gateway）
```

### 9.3 Rollback（queue full / shutdown / stale 等での予約返却）

```text
Rollback: 予約済み slot を Budget Pool に返却する（reservation cancel）

Cases:
  1. Build 失敗（OOM / validation 失敗）→ reservation を release（所有権は発生していないため UAF なし）
  2. Pending gap の stale（seqId が無効化された場合）→ registry entry を drop し reservation を release
  3. Queue full（Intent queue 全層溢れ）→ push 失敗 → reservation rollback（既存パターンと同様）
  4. Shutdown 時の admission close → 予約済みだが未 publish の slot を全て release

Invariant: rollback 時に所有権が失われない（所有権は reservation 時には発生していないため、返却は safe）

Mechanism: 既存の reservation→push→rollback パターン（ISRRuntimePublicationCoordinator の fetchAdd→push→fetchSub rollback）
           と同型の機構を RuntimeWorld に対しても適用する
```

### 9.4 Release（retire/reclaim 完了時に budget を返す）

```text
Release: epoch-safe 到達後の deleter 実行完了時に Budget Pool へ slot を返却する

Invariant: 1 publish → 1 release（1:1 対応、conservation の前提）

Mechanism:
  1. retire 経路（DeferredDeletionQueue / RetireQuarantineStore / TerminalReclaim）の
     drain(minReaderEpoch, isOlder) で epoch-safe 到達を判定
  2. isOlder(entry.epoch, minReaderEpoch) == true なら deleter(entry.ptr) を実行
  3. deleter 成功 → Budget Pool へ slot を 1 個返却（atomic fetchSub 相当）
  4. Shutdown 時は drainAll() で全 pending を強制解放し、全 slot を返却

Enforcement point: ISRRetireRouter / RetireQuarantineStore / DeferredDeletionQueue の drain 境界
                   + TerminalReclaimAuthority::drain() / drainAll()
```

### 9.5 既存パターンとの対応表

| 既存 `reservation → push → rollback` | 新 `reserve → transfer → rollback/release` | 差異 |
| --- | --- | --- |
| `fetchAdd` (reservation) | `reserve(K_world slot)` | 対象: Intent → RuntimeWorld |
| `push` (intentQueue) | `transfer` (OwnerChannel → RuntimeStore) | 対象: Intent → RuntimeWorld |
| `fetchSub rollback` (queue full) | `rollback` (build/push 失敗時) | 失敗時の cancel。パターンは同型 |
| `pop` (reservation 消費) | `drain → release` (epoch-safe 後に release) | Pop は queue 消費、drain は epoch 待ち。epoch 依存が追加される |

---

## 10. Conservation invariant

### 10.1 Lifetime Budget の conservation

```text
Invariant: reserved + owned + retired + reclaimed_total = K_world  ... (C1)

where:
  reserved       = Budget Pool から予約済みだが未 publish の slots（admission 段階）
  owned          = RuntimeStore::current(1) + readerHeld + retire queue + quarantine + terminal
                   = liveWorlds（各 Authority に所有されている worlds）
  retired        = drain() で reclaim 待ちの worlds（epoch-unsafe pending）
  reclaimed_total = 累積 reclaim 数（monotone、Budget Pool への返却済み）

  K_world = Budget Pool の有限 capacity（M_world の上界）

At any time:
  reserved + owned ≤ K_world  ... (C2) — live invariant（reclaimed_total を除いた現時点の占有）
  reserved + owned + retired ≤ K_world  is NOT required（retired は owned に含まれる場合があるため、
  正確には上記 C1 の分類に依存する）

Simplified live invariant:
  outstandingWorlds = owned = liveWorlds ≤ K_world < ∞  ... (C3)
```

### 10.2 I4 の conservation との分離

| Conservation | 対象 | 式 | Invariant |
| --- | --- | --- | --- |
| I4 (logical obligation) | Logical recovery obligation | `admittedLogicalObligationCount = liveOwnershipCount + terminalDispositionCount` | I4 D15/D18.3。`successCount` を含む |
| D101-6 (RuntimeWorld) | RuntimeWorld lifetime | `publishedWorlds = liveWorlds + reclaimedWorlds` または `outstandingWorlds ≤ K_world` | 本監査で定義。I4 とは別式 |

- 両者は **別 invariant** として契約で分離する（I08）。混同しない。
- I4 の conservation は logical obligation の lifecycle（transport→durable→building→stalled→success/superseded/shutdownDiscard）を扱う。
- D101-6 の conservation は RuntimeWorld の lifecycle（reserved→owned→retired→reclaimed）を扱う。

### 10.3 Conservation chain としての統合

D101-5 で判定した「`publish admission → reservation → ownership transfer → retire → release` を一つの lifetime-budget conservation chain として扱う」は、本章の `reserve → transfer → rollback/release` と統合される:

```text
Budget Authority (K_world pool)
      │
      ├─ Admission (A_max / P_max gate)
      │     │
      │     ├─ reserve(K_world slot) ──fail──→ backpressure (BLOCK)
      │     │
      │     ▼ success
      │  RuntimeWorld ownership (OwnerChannel → PendingPublishRegistry → RuntimeStore::current)
      │     │
      │     ├─ publishAndSwap → old world → Retire ownership
      │     │     │
      │     │     └─ DeferredDeletionQueue(4096) → Quarantine(512+512) → TerminalReclaim(K slots)
      │     │           │
      │     │           └─ drain(minReaderEpoch, H_max 保証) → Budget release
      │     │
      │     └─ reader borrow (EpochDomain, H_max bounded)
      │           │
      │           └─ epoch 進行 → drain 進行 → Budget release
      │
      └─ Shutdown ──→ drainAll() → Budget release (all pending)
```

- 本 chain は **一つの budget（K_world）の下で conservation が成立する** ことを invariant として保証する。
- I01/I02/I08 を別々に設計しすぎず、D 案として一つの chain として扱うことが D101-5 で判定済みであり、本章でその骨格を具体化した。

---

## 11. Authority mapping

| Invariant | Authority | State | 判定 |
| --- | --- | --- | --- |
| `K_world` (Budget Pool) | **Lifetime Budget Authority**（新設）— RuntimeWorldAuthority の拡張または新コンポーネント | `std::atomic<int> budgetCount_` + `K_world` 定数（I4 の pendingIntentCount_ と同型だが別 budget） | **DESIGN_REQUIRED** — 新 Authority として設計が必要 |
| `A_max` | Lifetime Budget Authority（admission gate） | `A_max` 定数 + interval 内の acquire count | **DESIGN_REQUIRED** — admission 制御として設計が必要 |
| `P_max` | Lifetime Budget Authority（rate gate） | `P_max` 定数 + 時間窓内の publish count | **DESIGN_REQUIRED** — rate 制御として設計が必要 |
| `M_terminal ≤ K` | `ISRRetireRouter::TerminalReclaimAuthority`（I01 と同一） | `std::array<Entry, K>` + `residentAtomic_` | **DESIGN_REQUIRED** — std::vector → std::array&lt;K&gt; 置換 |
| `H_max` | `EpochDomain` + `AudioEngine.Processing` | `H_max` 定数 + watchdog / HealthEvent | **DESIGN_REQUIRED** — fail-safe として設計が必要 |
| `G_contract` | `ISRWorldRetirementTelemetry`（観測）+ Lifetime Budget Authority（強制） | `G_contract` 定数 | **DESIGN_REQUIRED** — contract として設計が必要 |
| Budget separation | I4 Contract（契約レベル） | 契約記述 | **DESIGN_REQUIRED** — I4 追記として設計が必要 |

### Authority 所在の原則

- **Lifetime Budget Authority** は `RuntimeWorldAuthority` の拡張として配置するのが自然である。理由: `RuntimeWorldAuthority` は既に `RuntimeStore`（current の唯一の write gateway）、`OwnerChannel`（所有権移転の唯一の point）、`PendingPublishRegistry`（gap 管理）を所有しており、Budget Pool の管理はこれらと同一の責務境界に属する。
- 代替案として **新コンポーネント**（例: `LifetimeBudgetAuthority`）を独立させる案もあるが、D101-5 の「I01/I02/I08 を別々に設計しすぎない」原則に反し、Authority 間の協調コストが増大する。したがって **RuntimeWorldAuthority の拡張**を第一候補とする。
- I4 の `ISRRuntimePublicationCoordinator` の `pendingIntentCount_` は **流用しない**（対象・粒度・Scope が異なるため）。Lifetime Budget の `budgetCount_` は新設する。

---

## 12. Enforcement points

| Invariant | Enforcement point | 既存 code | 新 enforcement |
| --- | --- | --- | --- |
| `K_world` budget | `RuntimeWorldAuthority::publish` の admission 境界 | 存在しない | Budget Pool から reserve、失敗→backpressure |
| `A_max` | `RuntimeWorldAuthority::publish` の interval 境界 | 存在しない（telemetry 観測のみ） | Interval 内の acquire count で gate |
| `P_max` | `RuntimeBuilder::buildWorld()` / `RuntimeWorldAuthority::publish` の時間窓境界 | 存在しない | 時間窓内の publish count で gate |
| `M_terminal ≤ K` | `ISRRetireRouter::TerminalReclaimAuthority::store()` の呼び出し境界 | `std::vector` growable（enforcement なし） | `std::array<K>` + reservation 前提で store は失敗しない |
| `H_max` | `EpochDomain::isOlder()` / `AudioEngine.Processing` の block 境界 | 存在しない | `H_max` 超過時に HealthEvent + producer backpressure |
| `G_contract` | `ISRWorldRetirementTelemetry` の gap 検出境界 → `RuntimeWorldAuthority` の publish 境界 | 存在しない（observed maximum のみ） | Gap 超過時に bounded recovery + producer admission control |
| Budget separation | I4 Contract の契約境界 | `kMaxLogicalRecoveryObligations` のみ | I4 追記で分離を明示 |

---

## 13. Failure paths

### 13.1 各失敗パスでの予約返却（Rollback）

| 失敗パス | 予約の状態 | 返却方法 | 所有権の安全性 |
| --- | --- | --- | --- |
| Build 失敗（OOM / validation） | Reserved（未 transfer） | Budget Pool へ reservation を release（cancel） | 所有権は発生していないため UAF なし |
| OwnerChannel push 失敗 | Reserved（未 commit） | Budget Pool へ release + OwnerChannel の owner を破棄 | 所有権は OwnerChannel にあるため、破棄は safe |
| PendingPublishRegistry gap 溢れ | Reserved（gap に登録済み） | Registry entry を drop + reservation を release | Stale 検出後に release |
| Intent queue 全層溢れ | Reserved（push 前） | `fetchSub rollback` と同様に Budget Pool へ release | 既存パターンと同型 |
| Shutdown 時の admission close | Reserved だが未 publish の全 slots | 全 reservation を release（shutdown path で一括） | Shutdown は publish を拒否するため、新規 reservation は発生しない |

### 13.2 Queue full 時の扱い

| Queue | 現行の扱い | Lifetime Budget 下での扱い |
| --- | --- | --- |
| Intent queue（`intentQueue_ + deferred ring`） | 全層 push 失敗 → `fetchSub` rollback + drop カウンタ | 同様に Budget reservation を rollback。ただし Lifetime Budget では backpressure（BLOCK）が原則であり、drop は発生しない |
| DeferredDeletionQueue(4096) | Enqueue 失敗 → RetireQuarantineStore へ退避 | 同様（既存の enqueueWithRetry chain を維持） |
| RetireQuarantineStore(512) | Store 失敗 → EmergencyQ → TerminalReclaim へ退避 | 同様（bounded TerminalReclaim でも reservation 済みのため store は失敗しない） |
| TerminalReclaim(K) | 現行は growable で失敗しない | Bounded 化後は reservation 済み slot への配置であり失敗しない |

---

## 14. Shutdown path

### 14.1 現行の shutdown

| ステップ | 現行の実装 | 所在 |
| --- | --- | --- |
| Publish admission close | Shutdown 確定後に Publish Intent を Path B admission gate で拒否 | `ConvoPeq.md:56686` |
| Pending publication | PendingPublishRegistry の残留を `drainAll()` 相当で処理 | `ConvoPeq.md` / `AudioEngine.ReleaseResources.cpp` |
| Retire/Quarantine | `ISRRetireRouter::drainAll()` で all pending を強制解放 | `ConvoPeq.md` / `ISRRetireRouter.cpp` |
| Final drain | `drainAll()` は audio thread 停止後に全 pending を `drainAll()` で解放し Verify Empty | `ConvoPeq.md` |
| Budget release | 現行は Budget Pool が存在しないため、release の概念なし | — |

### 14.2 Lifetime Budget 下での shutdown

```text
Shutdown
  │
  ├─ 1. Admission close: 新規 publish の reservation を拒否（Path B gate と同様）
  │     新規 RuntimeWorld の生成を停止 → K_world の新規消費を停止
  │
  ├─ 2. Pending gap drain: PendingPublishRegistry の残留を全て処理
  │     登録済みだが未 commit の worlds を commit または drop（reservation を release）
  │
  ├─ 3. Retire drain: DeferredDeletionQueue / RetireQuarantineStore の残留を drain
  │     minReaderEpoch の進行を待ち、epoch-safe 到達後に deleter → Budget release
  │
  ├─ 4. Terminal drain: TerminalReclaimAuthority::drainAll() で全 pending を強制解放
  │     Bounded K でも drainAll() は全 entry を走査して解放可能（capacity 保証は K が worst-case 残留数を上回ることで成立）
  │
  └─ 5. Budget release: 全 drain 完了後、Budget Pool の全 slots が返却され Verify Empty
        K_world slots 全てが free → shutdown 完了
```

### 14.3 有限完了性の条件

- `K` が shutdown 時の worst-case 残留数を上回ること（I10 の条件）。
- `K` は `M_world` の上界と同一の値から導出されるため、循環依存に注意。`K` の値は D101-7 で導出するが、`K ≥ worst-case shutdown 残留数` の条件を満たす必要がある。
- 現行は growable で有限完了性が自明に成立するが、bounded 化後は `K` の値が条件を満たすことの証明が必要である。

---

## 15. I01-I10 dependency impact

### 15.1 本監査（Lifetime Budget Authority）が依存する invariant

| 依存先 | 本監査での扱い | 状態 |
| --- | --- | --- |
| I08 (budget separation) | **本監査で定義** — Lifetime Budget Authority の新設として具体化 | DESIGN_REQUIRED |
| I02 (reservation-first) | **本監査で定義** — Budget Pool の reserve として具体化 | DESIGN_REQUIRED |
| I09 (ownership conservation) | **本監査で骨格を定義** — `publishedWorlds = liveWorlds + reclaimedWorlds` の chain として | DESIGN_REQUIRED |

### 15.2 本監査が影響する invariant（D101-7 以降で設計）

| 影響先 | 影響内容 | 状態 |
| --- | --- | --- |
| I01 (M_terminal) | Budget Pool の `K` が `M_terminal` の capacity でもある。I01 の実装は本監査の Budget Pool 設計に依存 | DESIGN_REQUIRED（D101-7） |
| I03 (M_reader) | `M_reader = f(H_max, P_max)` は Budget Pool とは独立だが、`M_world` の分解で統合される | DESIGN_REQUIRED（D101-7） |
| I04 (H_max) | Budget release の epoch 進行保証に H_max が必要 | DESIGN_REQUIRED（D101-7） |
| I05 (A_max) | Budget の admission gate として A_max が Budget Authority で強制される | DESIGN_REQUIRED（D101-7） |
| I06 (P_max) | Budget の rate gate として P_max が Budget Authority で強制される | DESIGN_REQUIRED（D101-7） |
| I07 (G_contract) | Sampler gap の契約は Budget の admission 制御と連動 | DESIGN_REQUIRED（D101-7） |
| I10 (M_world) | `M_world ≤ K_world` は全 invariant の統合として定義される | DESIGN_REQUIRED（D101-7） |

### 15.3 Cross-invariant dependency（D101-5 からの継承）

```text
I08 (budget separation) ──→ I02 (reservation-first) ──→ I09 (ownership conservation)
     │                            │                          │
     └────────────→ I01 (M_terminal) ────────────────────────┘
                                                    │
I04 (H_max) ──→ I03 (M_reader) ───────────────────→ I10 (M_world)
     │                    ↑
     └────→ I07 (G_contract) ─┘

I05 (A_max) ──→ I10 (M_world)（流入側）
I06 (P_max) ──→ I10 (M_world)（流入側）
```

- 本監査（I08 + I02 の Budget Authority）は dependency グラフの根元に位置し、以降の全 invariant の前提となる。
- D101-5 の案B（`lifetime budget authority → reservation → ownership conservation → M_terminal/M_reader/... → M_world`）に整合する。

---

## 16. Open questions

| # | 質問 | 現時点の見解 | 解決フェーズ |
| --- | --- | --- | --- |
| Q1 | `K_world` の具体的数値は？ | `M_world` の分解式 `K_world ≤ 1+4096+1024+K+f(H_max,P_max)` から導出するが、具体的値は D101-7 で決定 | D101-7 |
| Q2 | `A_max` / `P_max` の具体的数値は？ | Interval / 時間窓の定義と `H_max` に依存する。D101-7 で導出 | D101-7 |
| Q3 | `H_max` の具体的数値は？ | `maxBlockDuration + ε` + fail-safe の設計に依存。D101-7 で導出 | D101-7 |
| Q4 | `G_contract` の具体的数値は？ | `block 時間の数倍` として設計時に固定。D101-7 で導出 | D101-7 |
| Q5 | Lifetime Budget Authority を `RuntimeWorldAuthority` の拡張とするか、新コンポーネントとするか | **RuntimeWorldAuthority の拡張を第一候補**（既に Store/OwnerChannel/Registry を所有する同一責務境界）。新コンポーネントは Authority 間協調コストが増大するため第二候補 | D101-7 |
| Q6 | `COALESCE` 相当の重複 publish は新規 reservation を取得しないか | **取得しない**（I4 D18.4 と同様）。ただし RuntimeWorld の coalesce 条件は I4 の CoalesceIdentity とは独立に定義する必要がある | D101-7 |
| Q7 | `K` が shutdown 時の worst-case 残留数を上回ることの保証（循環依存） | `K` の値は D101-7 で導出するが、`K ≥ worst-case shutdown 残留数` の条件を満たす必要がある | D101-7 |
| Q8 | 既存 `PendingPublishRegistry 64` と `P_max` の関係 | 64 は gap 容量であり、P_max の代わりにならないことを本監査で確定。P_max は別途定義 | 解決済み（本監査） |
| Q9 | 既存 `reservation → push → rollback` の流用範囲 | パターン（reservation-before-push + 失敗時 rollback）は流用可能だが、対象・Authority・Scope が異なるため新設が必要であることを本監査で確定 | 解決済み（本監査） |

---

## 17. Verdict

### 判定: `DESIGN_REQUIRED`

| 判定 | 定義 | 本監査の該当性 |
| --- | --- | --- |
| `SATISFIED` | 既存コードで Lifetime Budget Authority が充足している | **該当せず** — 既存 reservation counter は対象・粒度・Scope が異なるため流用不可 |
| `PARTIAL` | 部分的に充足し、拡張で充足可能 | **該当せず** — パターン（reservation-before-push）は参考になるが、Budget unit / Authority / Scope が全て異なるため PARTIAL ではない |
| `MISSING` | 既存コードに Lifetime Budget Authority が存在しない | **部分的該当** — Lifetime Budget Authority 自体は MISSING だが、本監査で骨格を定義できたため、単なる MISSING ではない |
| `CONFLICT` | 現行設計と Lifetime Budget Authority が衝突する | **該当せず** — D101-5 で衝突 0 件を確認済み。既存 Authority の拡張で実現可能 |
| `DESIGN_REQUIRED` | 骨格は本監査で定義可能だが、具体的 state machine / contract を新設計として導入する必要がある | **◯ 該当（本監査の結論）** |

### なぜ `DESIGN_REQUIRED` か

- **骨格は定義できた**: Budget unit（RuntimeWorld）、Budget capacity（K_world）、A_max / P_max の定義、reservation / transfer / rollback / release の各ステップ、conservation invariant（`reserved + owned + retired = K_world`）、Authority mapping（Lifetime Budget Authority は RuntimeWorldAuthority の拡張）、Enforcement points、Failure paths、Shutdown path は、本監査で設計監査として確定できた。
- **しかし具体的 state machine / contract は新設計が必要**: `K_world` / `A_max` / `P_max` / `H_max` / `G_contract` の具体的数値、`Budget Pool` の具体的 State（`std::atomic<int> budgetCount_` 等）、`reservation → transfer → release` の具体的 ordering と失敗時の cancel 条件、conservation の形式的証明は、D101-7 での詳細設計を要する。
- **既存 reservation counter は流用不可だが、パターンは流用可能**: `ISRRuntimePublicationCoordinator::pendingIntentCount_` の `reservation → push → rollback` パターンは、Lifetime Budget の新設計に概念的に流用可能であるが、対象・粒度・Scope が異なるため、そのまま Lifetime Budget Authority とはみなせない。新設が必要である。
- **`CONFLICT` ではない理由**: 現行の `RuntimeWorldAuthority`（Store/OwnerChannel/Registry を既に所有）の拡張として Lifetime Budget Authority を実現可能であり、既存の責務境界を壊さずに一本の conservation chain として再構成できる。D101-5 で衝突 0 件を確認済みである。

### 全体判定

```text
D101-3  CONTRACT_REQUIRES_NEW_INVARIANT
   │
   ▼
D101-4  ARCHITECTURAL_CHANGE_REQUIRED（10 invariants × 4 段階照合）
   │
   ▼
D101-5  ARCHITECTURAL_CHANGE_REQUIRED（mapping 具体化完了、案B 確定）
   │
   ▼
D101-6  DESIGN_REQUIRED  ◀ 本監査
   │  Lifetime Budget Authority の骨格を定義
   │  Budget unit / A_max / P_max / reservation-transfer-rollback-release / conservation を確定
   │  既存 reservation counter は流用不可だが、パターンは流用可能
   │
   ▼
D101-7 — Lifetime Budget Authority の具体的 state machine / contract
   │
   ├── K_world / A_max / P_max / H_max / G_contract の具体的数値と State
   ├── reservation → transfer → rollback → release の具体的 ordering
   ├── conservation の形式的証明
   └── I08 の I4 Contract 追記
   │
   ▼
D101-8 — reservation-first 設計 + TerminalReclaim bounded 化
   │
   ▼
Phase I GO/NO-GO 再判定
```

- **本監査でも production code は変更しない**（指示どおり）。
- D101-7 では、本監査で定義した骨格に基づき、Lifetime Budget Authority の具体的 state machine / contract（`K_world` / `A_max` / `P_max` / `H_max` / `G_contract` の数値と State、reservation ordering、conservation 証明）を設計する。
- D101-6 が `DESIGN_REQUIRED` になったため、そこで初めて D101-7 に進むのが安全である（指示どおり）。

---

## 付録: D101-6 監査チェックリスト

- [x] D101-5 の案B（`lifetime budget authority → reservation → ownership conservation → ...`）を前提として設計
- [x] RuntimeWorldAuthority の現行機構（PendingPublishRegistry / OwnerChannel / shutdown）を監査
- [x] Publish Intent 経路の既存 reservation（reservation→push→rollback）の authority / counter / push failure rollback / reservation と実 ownership の対応を調査
- [x] Retire 経路（ISRRetireRouter / RetireQuarantineStore / TerminalReclaimAuthority）の ownership transfer を調査
- [x] Epoch / Reader（EpochDomain / reader slot / pending retire / reclaim completion）を調査
- [x] Shutdown 経路（publish admission close / pending publication / retire/quarantine / final drain / budget release）を調査
- [x] Budget authority（誰が lifetime budget を所有するか）を確定（RuntimeWorldAuthority の拡張を第一候補）
- [x] Budget unit（RuntimeWorld / publication / logical obligation のどれを 1 単位とするか）を確定（RuntimeWorld）
- [x] Capacity（K_world の有限値）を invariant として定義
- [x] Admission（A_max の意味と enforcement point）を確定
- [x] Rate（P_max の意味と enforcement point）を確定（PendingPublishRegistry 64 との分離を明示）
- [x] Reservation（admission 時点で何を予約するか）を確定（K_world slot）
- [x] Transfer（publish → owner channel への所有権移転）を確定
- [x] Rollback（queue full / shutdown / stale 等での予約返却）を確定
- [x] Release（retire/reclaim 完了時に何を返すか）を確定（Budget Pool への slot 返却）
- [x] Conservation（reserved + owned + retired = budget）の不変条件を確定
- [x] Authority mapping（各 invariant の Authority/State）を確定
- [x] Enforcement points を確定
- [x] Failure paths（各失敗パスでの予約返却）を検証
- [x] Shutdown path（有限完了性）を検証
- [x] I01-I10 dependency impact を検証
- [x] 既存 reservation counter を Lifetime Budget Authority とみなせるかを厳密に判定（みなせない、パターンのみ流用可能）
- [x] Production code 変更なし（設計監査のみ）
