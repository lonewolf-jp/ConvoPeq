# D101-7 — Lifetime Budget State Machine / Contract Definition

| 項目 | 内容 |
| --- | --- |
| **ID** | D101-7 |
| **日付** | 2026-08-20 |
| **対象ブランチ** | `main` |
| **一次資料** | `ConvoPeq.md`（2026-08-19 全ソース連結）、`src/audioengine/RuntimeWorldAuthority.h`、`src/audioengine/ISRRuntimePublicationCoordinator.h`、`src/audioengine/ISRRetireRouter.h/.cpp`、`src/audioengine/RetireQuarantineStore.h`、`src/DeferredDeletionQueue.h`、`src/core/RuntimeStore.h`、`src/core/EpochDomain.h`、`src/audioengine/ISRRetire.h`、`src/audioengine/RuntimeBuilder.h`、`src/audioengine/AudioEngine.Commit.cpp`、`src/audioengine/AudioEngine.Retire.cpp`、`src/audioengine/ISRWorldRetirementTelemetry.h`、`doc/work88/I4_DESIGN_CONTRACT.md`、D101-3/D101-4/D101-5/D101-6 evidence |
| **前提** | D101-6 verdict: **DESIGN_REQUIRED** — Budget unit=`RuntimeWorld`, `K_world`=`M_world` の有限上界、`A_max`/`P_max` の定義、`reserve→transfer→rollback→release` の lifecycle、`RuntimeWorldAuthority` を Lifetime Budget Authority の第一候補とすることは確定。具体的 state / ordering / 数値 / enforcement / 証明は未確定 |
| **目的** | **実装に入らず、契約を閉じる**。Lifetime Budget の具体的 State Machine / Contract（`K_world`/`A_max`/`P_max`/reservation token の invariant、ordering、enforcement point、形式的条件）を定義する。数値を決めるのではなく「その数値がなぜ十分であり、どこで強制され、どの状態を含み、どの遷移で減少するか」を閉じる。既存 `pendingIntentCount_` を昇格させる設計は避け、`RuntimeWorld` lifetime 用の独立した budget state として設計する |
| **制約** | **コード変更なし・contract 定義のみ**。`K_world = 4096` のような数値は決めない。既存 `pendingIntentCount_` を Lifetime Budget Authority に昇格させない |
| **判定** | **CONTRACT_DEFINED** — Lifetime Budget State Machine（8状態 + 各遷移の唯一の Authority）、`K_world`/`A_max`/`P_max`/reservation token の invariant と ordering、Failure matrix、Shutdown contract、他 invariant との接続を契約として閉じた。D101-8（reservation-first 詳細設計）へ進む準備が整った |

---

## 1. Scope

- D101-6 で Lifetime Budget Authority の骨格（Budget unit = `RuntimeWorld`、`K_world` = `M_world` の有限上界、`A_max`/`P_max` の定義、`reserve→transfer→rollback→release` の lifecycle）が確定した。
- D101-7 では、その骨格を **具体的 State Machine / Contract** として閉じる。State を8つ（Available → Reserved → Transferred → Published → Retiring → Quarantined → Terminal → Released）に分解し、各遷移の唯一の Authority を確定する。
- `K_world` の数値は決めない。「その数値がなぜ十分であり、どこで強制され、どの状態を含み、どの遷移で減少するか」を先に閉じる。
- 既存 `pendingIntentCount_` を Lifetime Budget Authority に昇格させる設計は避ける（D101-6 で判定済み）。
- 本監査はコード変更なしで、D101-8（reservation-first 詳細設計）へ進む準備を整える。

---

## 2. D101-6 からの入力条件

| 入力 | D101-6 での確定 | 本監査での扱い |
| --- | --- | --- |
| Budget unit | `RuntimeWorld`（≒ `RuntimeState` 1個 = 1 unit） | State Machine の counting 対象として確定 |
| `K_world` | `M_world` の有限上界（`M_world ≤ K_world < ∞`） | 具体的 invariant と ordering を本監査で定義。数値は決めない |
| `A_max` | admission interval あたりの acquire 上限 | 具体的 invariant と enforcement point を本監査で定義 |
| `P_max` | 時間窓あたりの publish 上限（`PendingPublishRegistry 64` とは非同一） | 具体的 invariant と enforcement point を本監査で定義 |
| `reserve→transfer→rollback→release` | Lifecycle の骨格 | State Machine として具体化する |
| `RuntimeWorldAuthority` を Lifetime Budget Authority の第一候補 | Authority mapping として確定 | 各遷移の唯一の Authority を本監査で確定 |
| 既存 `pendingIntentCount_` は流用不可 | 対象・粒度・Scope が異なるため新設が必要 | 新独立 budget state として設計する |

---

## 3. Existing reservation mechanisms（再確認）

D101-6 で整理した既存機構を、本監査の State Machine 設計の参考として再掲する。

| 機構 | 所在 | パターン | Lifetime Budget との関係 |
| --- | --- | --- | --- |
| `ISRRuntimePublicationCoordinator::pendingIntentCount_` | `ISRRuntimePublicationCoordinator.h` | `fetchAdd(reservation) → push → fetchSub(rollback)`（queue full 時） | Intent の reservation であり、RuntimeWorld の Lifetime Budget とは別。パターンは参考になるが流用不可 |
| `publicationIntentResidencyCount_` | 同上 | `reservation → push → rollback`（Publish Intent 専用） | 同上 |
| `PendingPublishRegistry(64)` | `RuntimeWorldAuthority.h` | `registerPublish(seqId, world) → lookup → unregister` | Gap の bounded 性（64）は `P_max` の代わりにならない。gap 容量 vs 頻度上限の分離が必要 |
| `ISRRetireRouter::enqueueWithRetry()` | `ISRRetireRouter.h/.cpp` | `D(4096) → Q(512) → EQ(512) → TerminalReclaim(∞)` の chain | 現行は growable で必ず成功。Bounded 化で reservation 前提に置換する |

**結論（D101-6 継承）**: 既存 `reservation → push → rollback` パターンは **概念的に流用可能**（reservation-before-push の ordering）が、Budget unit / Authority / Scope が異なるため **そのまま Lifetime Budget Authority とはみなせない**。新独立 budget state として設計する。

---

## 4. Existing ownership states（再確認）

| 状態 | 所在 | Owner | 本監査での State Machine 対応 |
| --- | --- | --- | --- |
| Build | `RuntimeBuilder::buildWorld()` | `RuntimeBuilder`（一時） | `Reserved → Transferred` の間の一時状態。Build 失敗時は Reserved → Released（rollback） |
| OwnerChannel | `RuntimeWorldAuthority::OwnerChannelType` | `RuntimeWorldAuthority`（value） | `Transferred` 状態の Owner 保有 |
| PendingPublishRegistry | `RuntimeWorldAuthority::PendingPublishRegistry`（64 slots） | `RuntimeWorldAuthority`（value） | `Transferred` → `Published` の gap。seqId で keying |
| RuntimeStore::current | `RuntimeWorldAuthority::Store runtimeStore_` | `RuntimeWorldAuthority`（CRTP Owner） | `Published` 状態（current 1 slot） |
| Retire | `ISRRetireRouter` | `ISRRetireRouter` | `Retiring` 状態（DeferredDeletionQueue → Quarantine → Terminal の chain） |
| Reader borrow | `AudioEngine.Processing` | `AudioEngine`（borrow、非所有） | `Published` の reader 側。`H_max` で hold 時間を上界 |
| Shutdown | `AudioEngine.ReleaseResources` | `AudioEngine` | `Released` への強制遷移（drainAll） |

---

## 5. Lifetime Budget State Machine

### 5.1 状態定義（8状態）

| # | 状態 | 意味 | Budget counting | 所在 |
| --- | --- | --- | --- | --- |
| S0 | `Available` | Budget Pool の free slot。未予約 | `K_world` の空きとして counting。`available = K_world - (Reserved+...+Terminal)` | Lifetime Budget Authority（新設、RuntimeWorldAuthority 拡張） |
| S1 | `Reserved` | Budget Pool から 1 slot を予約済み。未 build | `reserved` として counting。`M_world` の live には含むが、まだ world は存在しない | Lifetime Budget Authority（admission gate） |
| S2 | `Transferred` | `Reserved` slot に対応する `RuntimeState` が OwnerChannel / PendingPublishRegistry に存在 | `transferred` として counting。world が物理的に存在する | `RuntimeWorldAuthority::OwnerChannel` + `PendingPublishRegistry` |
| S3 | `Published` | `RuntimeStore::current` に world が publish された。旧 world は Retiring へ | `published` として counting。`M_current = 1` の寄与 | `RuntimeWorldAuthority::Store runtimeStore_`（current） |
| S4 | `Retiring` | 旧 world が `ISRRetireRouter::enqueueWithRetry()` で DeferredDeletionQueue(4096) に退避 | `retiring` として counting。`M_retire` の寄与 | `ISRRetireRouter` → `DeferredDeletionQueue` |
| S5 | `Quarantined` | Retire queue 溢れで `RetireQuarantineStore(512)` / `EmergencyQ(512)` に退避 | `quarantined` として counting。`M_quarantine` の寄与 | `RetireQuarantineStore` ×2 |
| S6 | `Terminal` | Quarantine 溢れで `TerminalReclaimAuthority<K>`（bounded, K slots）に退避 | `terminal` として counting。`M_terminal` の寄与 | `TerminalReclaimAuthority`（bounded 化後） |
| S7 | `Released` | `drain(minReaderEpoch, isOlder)` で epoch-safe 到達後に deleter 実行完了。Budget Pool へ返却済み | counting しない（reclaimed_total として累積のみ） | Budget release（各 drain 境界） |

### 5.2 状態遷移と唯一の Authority

```text
S0 Available ──reserve(K_world slot)──→ S1 Reserved
                    │  Authority: Lifetime Budget Authority (RuntimeWorldAuthority 拡張)
                    │  Enforcement: admission gate（A_max/P_max と連動）
                    │  Fail: K_world slots 全占有 → backpressure（BLOCK、publish 拒否）
                    ▼
S1 Reserved ──build+seal──→ S2 Transferred
                    │  Authority: RuntimeBuilder（build）+ RuntimeWorldAuthority（seal + OwnerChannel）
                    │  Fail: build 失敗 → S1 → S0（reservation cancel、rollback）
                    ▼
S2 Transferred ──register(seqId)──→ S2 (gap registered)
                    │  Authority: RuntimeWorldAuthority::PendingPublishRegistry
                    │  Fail: registry 64 溢れ → S1 → S0（stale 検出後に rollback）
                    ▼
S2 Transferred ──commit(publishAndSwap)──→ S3 Published
                    │  Authority: RuntimeWorldAuthority::Store (WriteAccess::publishAndSwap)
                    │  Enforcement: 唯一の publish gateway（X4-B）
                    │  Note: 旧 world S3 → S4 へ同時に遷移（eviction）
                    ▼
S3 Published ──publishAndSwap evicts old──→ S4 Retiring
                    │  Authority: ISRRetireRouter::enqueueWithRetry()
                    │  Chain: D(4096) → Q(512) → EQ(512) → Terminal(K)
                    │  Note: Bounded TerminalReclaim は予約済み slot への配置であり失敗しない
                    ▼
S4 Retiring ──enqueue fail──→ S5 Quarantined
                    │  Authority: ISRRetireRouter → RetireQuarantineStore
                    ▼
S5 Quarantined ──store fail──→ S6 Terminal
                    │  Authority: ISRRetireRouter → TerminalReclaimAuthority<K>
                    │  Note: S6 は予約済み slot のため、store は失敗しない（allocation-free）
                    ▼
S4/S5/S6 ──drain(minReaderEpoch, isOlder)──→ S7 Released
                    │  Authority: ISRRetireRouter / RetireQuarantineStore / DeferredDeletionQueue
                    │  Condition: isOlder(entry.epoch, minReaderEpoch) == true
                    │  Guarantee: H_max < ∞ により epoch は有限時間で進行する（I04）
                    ▼
S7 Released ──Budget release──→ S0 Available
                    │  Authority: Lifetime Budget Authority（slot 返却）
                    │  Mechanism: Budget Pool の空き slot 数が 1 増加（atomic 相当）

Failure / Shutdown の遷移（Rollback）:
  S1 ──build fail──→ S0（reservation cancel）
  S2 ──stale / shutdown──→ S0（registry entry drop + reservation release）
  S4/S5/S6 ──shutdown──→ S7 ──→ S0（drainAll() で全 pending を強制解放）
```

### 5.3 各遷移の唯一の Authority（Singularization）

| 遷移 | Authority | 根拠 |
| --- | --- | --- |
| S0→S1 (reserve) | **Lifetime Budget Authority**（RuntimeWorldAuthority 拡張） | Budget Pool の唯一の owner。Admission の唯一の gate |
| S1→S2 (build+seal) | **RuntimeBuilder**（build）+ **RuntimeWorldAuthority**（seal+OwnerChannel） | Build は RuntimeBuilder、seal は RuntimeState 完全型を持つ側（AudioEngine.h 経由） |
| S2→S3 (commit) | **RuntimeWorldAuthority::Store**（WriteAccess::publishAndSwap） | X4-B: 唯一の publish gateway。Store を value 所有する CRTP Owner |
| S3→S4 (retire) | **ISRRetireRouter::enqueueWithRetry()** | Retire の唯一の entry point。D→Q→EQ→Terminal の chain を内部で管理 |
| S4→S5 (quarantine) | **RetireQuarantineStore**（ISRRetireRouter 内部） | QuarantinedEntry の唯一の owner。store() が失敗時は EmergencyQ → Terminal へ |
| S5→S6 (terminal) | **TerminalReclaimAuthority&lt;K&gt;**（bounded） | 予約済み slot への配置。失敗しない |
| S4/S5/S6→S7 (drain) | **ISRRetireRouter / RetireQuarantineStore / DeferredDeletionQueue** | 各 store の drain() が epoch-safe 到達後に deleter 実行 |
| S7→S0 (release) | **Lifetime Budget Authority**（Budget Pool 返却） | Budget release の唯一の Authority |
| Rollback (S1→S0, S2→S0) | **Lifetime Budget Authority** | Reservation cancel の唯一の Authority |
| Shutdown (S4/S5/S6→S7) | **AudioEngine.ReleaseResources** → **ISRRetireRouter::drainAll()** | drainAll() は audio thread 停止後に全 pending を強制解放 |

---

## 6. Budget unit definition

### 6.1 Budget unit

| 候補 | 内容 | 採用 |
| --- | --- | --- |
| `RuntimeWorld` (≒ `RuntimeState`) | 1 publish で生成される `RuntimeState` 1 個を 1 unit とする | **✅ 採用** |
| `publication` | 1 publish 操作（`commitRuntimePublication` 呼び出し）を 1 unit | ❌ 失敗時は world が生成されないため 1:1 でない |
| `logical obligation` | I4 の `kMaxLogicalRecoveryObligations` の単位 | ❌ I4 と混同しない（D101-3 で分離済み） |

**確定**: Budget unit = `RuntimeWorld`（≒ `RuntimeState`）。1 publish 成功で生成される `RuntimeState` 1 個が 1 unit。`pendingIntentCount_` の単位（Intent）とは異なる独立した budget として設計する。

### 6.2 Budget capacity `K_world`

```text
Invariant: K_world = M_world の有限上界

K_world < ∞ かつ outstandingWorlds ≤ K_world を保証する

K_world の分解（D101-3 chapter 11）:
  K_world ≤ 1 (M_current) + 4096 (M_retire) + 1024 (M_quarantine) + K (M_terminal) + f(H_max, P_max) (M_reader)

ここで K = M_terminal（TerminalReclaim の bounded capacity）
K_world は D101-8 で具体的数値を導出するが、本監査では K_world < ∞ の invariant と
「どの状態を含み、どの遷移で減少するか」を閉じる

Counting 対象:
  - S1 Reserved: 予約済みだが未 publish の slots（admission 段階）→ K_world に含む
  - S2 Transferred: OwnerChannel / PendingPublishRegistry の worlds → K_world に含む
  - S3 Published: RuntimeStore::current (1) → K_world に含む
  - S4 Retiring: DeferredDeletionQueue の worlds → K_world に含む
  - S5 Quarantined: RetireQuarantineStore ×2 の worlds → K_world に含む
  - S6 Terminal: TerminalReclaimAuthority<K> の worlds → K_world に含む
  - S7 Released: reclaimed（Budget Pool へ返却済み）→ K_world に含まない（累積のみ）
  - S0 Available: free slots → K_world の空きとして管理

Enforcement: S0→S1 の reserve 時に K_world の空きを消費。失敗→backpressure（BLOCK）
Release: S7→S0 の Budget release 時に K_world の空きが 1 増加
```

**確定**: `K_world` は `RuntimeWorld` の lifetime 全体（S1〜S6）を counting する budget の有限上界。数値は D101-8 で導出するが、counting 対象（S1〜S6）と enforcement point（S0→S1 reserve）を本監査で閉じる。

---

## 7. `A_max` definition

### 7.1 定義

```text
Invariant: A_max = 1 sampling interval に許される acquire 数の上限

A_max < ∞ かつ acquire(interval) ≤ A_max

where:
  interval = ISRWorldRetirementTelemetry の sampling interval（既存の sampler 間隔）
  acquire = RuntimeWorldAuthority::publish 成功（RuntimeStore::publishAndSwap 成功）
```

### 7.2 意味と enforcement point

| 項目 | 内容 |
| --- | --- |
| **意味** | 1 interval に publish 成功できる RuntimeWorld 数の上限。Burst 時の `M_world` 一時的発散を上界する |
| **Enforcement point** | `RuntimeWorldAuthority::publish` の admission 境界（S0→S1 reserve の前または同時）。1 interval 内の publish 成功数をカウントし、`A_max` 超過で backpressure（BLOCK）。Lifetime Budget Authority の admission gate と統合 |
| **Existing code** | `ISRWorldRetirementTelemetry` は観測専用（D76.4）。`A_max` の制御は存在しない |
| **Reject / Rollback 条件** | Interval 内の acquire 数が `A_max` に達すれば、残りの publish 要求は BLOCK（queue に滞留せず、caller に backpressure を伝播）。次の interval で再試行可能 |
| **`A_max` と `K_world` の関係** | `A_max` は `K_world` の流入側を制御する。`A_max` が小さければ burst 時の `M_world` 増加が抑制される。`A_max` と `P_max` は異なる粒度（interval 単位 vs 時間窓単位）で協調する |

### 7.3 なぜ数値を決めないか

- `A_max` の値は interval の長さと `P_max` / `H_max` に依存する。具体的数値は D101-8 で導出するが、本監査では `A_max < ∞` の invariant と「どこで強制され、どの状態を含み、どの遷移で減少するか」を閉じる。
- Interval の定義（telemetry の sampling interval）は既存の `ISRWorldRetirementTelemetry` に存在するが、`A_max` の enforcement は publish 側（`RuntimeWorldAuthority`）で行う。

---

## 8. `P_max` definition

### 8.1 定義

```text
Invariant: P_max = 時間窓（例: 1秒、H_max の数倍）あたりの publish 上限

P_max < ∞ かつ publish(window) ≤ P_max

where:
  window = 固定時間窓（例: 1秒）。H_max の数倍として定義することも可能
  publish = RuntimeWorldAuthority::publish 成功
  burst = window 内の一時的 burst（A_max で制御）
  sustained rate = 長期的な publish 頻度（P_max で制御）

  Burst と sustained rate の分離:
    A_max は burst（interval 単位）を制御
    P_max は sustained rate（window 単位）を制御
    両者は異なる粒度で流入側を上界する
```

### 8.2 `PendingPublishRegistry 64` との非同一性

| 項目 | 内容 | `P_max` との関係 |
| --- | --- | --- |
| `PendingPublishRegistry kPendingPublishCapacity=64` | enqueue→commit の async gap を bounded にする（64 slots）。Lock-free、Non-RT → ISR/audio thread 間 | Gap の容量。`P_max` の代わりにはならない |
| `P_max` | 時間窓あたりの publish 成功数の上限 | Publish 頻度の上限。Gap 容量とは異なる |

- 64 は **「同時に gap に存在できる world 数」**の上限。`P_max` は **「時間窓あたりの publish 回数」**の上限。
- Gap が 64 でも、gap 滞留時間が短ければ、1秒間に 64 を超える publish が可能（gap を通過して retire へ流れるため）。
- したがって `P_max` は 64 とは別に定義する必要がある。

### 8.3 意味と enforcement point

| 項目 | 内容 |
| --- | --- |
| **意味** | 時間窓あたりの publish 成功数の上限。`M_world(t) ≤ ceil(P_max × H_max) + pipeline_depth` の流入側を制御 |
| **Enforcement point** | `RuntimeBuilder::buildWorld()` / `RuntimeWorldAuthority::publish` の時間窓境界。時間窓内の publish 成功数をカウントし、`P_max` 超過で backpressure（BLOCK）。Lifetime Budget Authority の rate gate と統合 |
| **Existing code** | `RuntimeBuilder::buildWorld()` の呼び出し頻度に hard limit は存在しない。PendingPublishRegistry 64 は P_max の代わりにならない |
| **Publish admission の唯一の Authority** | **Lifetime Budget Authority**（RuntimeWorldAuthority 拡張）。`A_max` と同様に admission gate で強制する |
| **Burst と sustained の分離** | Burst: `A_max`（interval 単位）で制御。Sustained: `P_max`（window 単位）で制御。両者は異なる粒度で協調する |

---

## 9. Reservation token / ownership

### 9.1 Reservation が何を所有するのか

| 項目 | 内容 |
| --- | --- |
| **Reservation の対象** | `K_world` Budget Pool の 1 slot。Slot は `RuntimeWorld` 1 個の lifetime に対応する |
| **Token の表現** | Reservation token は Budget Pool の空き slot 数を 1 消費することで表現する。具体的 state は `std::atomic<int> budgetAvailable_`（または同等の counter）で管理する。Token 自体は slot index または counter の decrement として表現可能 |
| **State の表現** | `S0 Available → S1 Reserved` の遷移で `budgetAvailable_.fetchSub(1)`。`S1 Reserved` にある間は `reservedCount_` として counting する |

### 9.2 Double-release 防止

| 項目 | 内容 |
| --- | --- |
| **問題** | 同一 reservation を二重に release すれば、Budget Pool の空き数が不正に増加し、`K_world` の invariant が破れる |
| **対策** | 各 reservation は S1〜S6 のいずれかに exactly once で存在する。`S1→S0`（rollback）と `S7→S0`（release）は排他的であり、同一 reservation が両方で release されることはない。State Machine の遷移が唯一の release point であることを保証する |
| **Mechanism** | Reservation の state を `S1 Reserved` で追跡し、`rollback`（S1→S0）と `release`（S7→S0）を別遷移として定義する。Build 失敗時は S1→S0 のみ、drain 完了時は S7→S0 のみが発火する |

### 9.3 Reservation leak 防止

| 項目 | 内容 |
| --- | --- |
| **問題** | Reservation を取得したまま release/rollback されなければ、Budget Pool の空きが減少し続け、最終的に全 publish が BLOCK される（leak） |
| **対策** | 全ての reservation は必ず `S1→S0`（rollback）または `S7→S0`（release）のいずれかで返却される。Failure matrix（chapter 12）で全 failure path の rollback/release を網羅する。Shutdown 時は `drainAll()` で全 pending を強制解放する |
| **Mechanism** | `S1 Reserved` にある reservation は build 失敗 → S1→S0、成功 → S2 へ進行する。`S2〜S6` にある reservation は drain → S7→S0 で必ず返却される。Shutdown 時は全 S1〜S6 を drainAll() で S7→S0 に強制遷移させる |

### 9.4 Transfer 前後の invariant

| 遷移 | invariant | 保証 |
| --- | --- | --- |
| S0→S1 (reserve) | `reserved + owned + ... ≤ K_world` | Reserve 時に空きを消費。失敗→backpressure で invariant を維持 |
| S1→S2 (transfer) | Reservation は world の所有権に変換される（1 slot = 1 world） | Build 成功 → OwnerChannel に world が存在。失敗 → S1→S0 で slot 返却 |
| S2→S3 (publish) | 旧 world は S3→S4 へ eviction。`M_world` は増加するが `K_world` の範囲内 | Publish 成功で新 world が S3 に、旧 world が S4 へ。`K_world` の counting は S1〜S6 全てを含むため、eviction 自体は `K_world` を増やさない（S3 の旧 world が S4 へ移動するだけ） |
| S4/S5/S6→S7 (drain) | epoch-safe 到達後に deleter → Budget release | `H_max` により epoch は有限時間で進行する（I04）。drain は必ず完了する |
| S7→S0 (release) | Budget Pool の空きが 1 増加。`K_world` の invariant を回復 | Release は S7 からのみ発火し、double-release は State Machine の遷移制約で防止する |

### 9.5 既存 `pendingIntentCount_` との分離

| 項目 | `pendingIntentCount_`（既存） | Lifetime Budget reservation（新設） |
| --- | --- | --- |
| Budget unit | Intent（Publish/Recovery/Quarantine） | RuntimeWorld（≒ RuntimeState） |
| Authority | `ISRRuntimePublicationCoordinator` | Lifetime Budget Authority（RuntimeWorldAuthority 拡張） |
| State | `std::atomic<int> pendingIntentCount_` | 新設の `std::atomic<int> budgetCount_`（または `budgetAvailable_`） |
| Scope | Intent queue の capacity | `K_world` 全体の lifetime budget |
| 流用 | パターン（reservation-before-push）のみ参考 | 新独立 budget state として設計する |

**確定**: 既存 `pendingIntentCount_` を Lifetime Budget Authority に昇格させる設計は避ける。新独立 budget state として設計する（D101-6 で判定済み、本監査で再確認）。

---

## 10. Conservation invariant

### 10.1 Lifetime Budget の conservation

```text
Invariant C1: reserved + owned + retired_pending = K_world_occupied ≤ K_world < ∞

where:
  reserved       = S1 Reserved の slots（予約済みだが未 publish）
  owned          = S2 Transferred + S3 Published + S4 Retiring + S5 Quarantined + S6 Terminal
                   = liveWorlds（各 Authority に所有されている worlds）
  retired_pending = S4/S5/S6 のうち drain 待ちの worlds（owned に含まれるため、C1 では owned に包含）
  K_world_occupied = reserved + owned（reclaimed_total を除いた現時点の占有）
  K_world        = Budget Pool の有限 capacity（M_world の上界）
  available      = K_world - K_world_occupied（S0 の free slots）

Simplified:
  C2: K_world_occupied = reserved + owned ≤ K_world < ∞  ... (live invariant)
  C3: outstandingWorlds = owned ≤ K_world < ∞           ... (M_world の上界)
  C4: available = K_world - K_world_occupied ≥ 0         ... (Budget Pool の空きは非負)

At any time, C2 ∧ C3 ∧ C4 が成立する。
```

### 10.2 I4 の conservation との分離

| Conservation | 対象 | 式 | 本監査での扱い |
| --- | --- | --- | --- |
| I4 (logical obligation) | Logical recovery obligation | `admittedLogicalObligationCount = liveOwnershipCount + terminalDispositionCount` | I4 D15/D18.3。`successCount` を含む。Lifetime Budget とは別 |
| D101-7 (RuntimeWorld) | RuntimeWorld lifetime | `K_world_occupied = reserved + owned ≤ K_world < ∞` | 本監査で定義。I4 とは別式。混同しない |

- 両者は **別 invariant** として契約で分離する（I08 / D101-3 chapter 12）。混同しない。
- I4 の conservation は logical obligation の lifecycle（transport→durable→building→stalled→success/superseded/shutdownDiscard）を扱う。
- D101-7 の conservation は RuntimeWorld の lifecycle（S0→S1→...→S6→S7→S0）を扱う。

### 10.3 Conservation chain としての統合（D 案の再確認）

D101-5 で判定した「`publish admission → reservation → ownership transfer → retire → release` を一つの lifetime-budget conservation chain として扱う」は、本章の State Machine と統合される:

```text
Budget Authority (K_world pool, S0 Available)
      │
      ├─ Admission (A_max / P_max gate, S0→S1)
      │     │
      │     ├─ reserve(K_world slot) ──fail──→ backpressure (BLOCK)
      │     │
      │     ▼ success (S1 Reserved)
      │  RuntimeWorld ownership (S1→S2→S3)
      │     │
      │     ├─ build → OwnerChannel → PendingPublishRegistry (S2 Transferred)
      │     │
      │     └─ PublishExecutor → publishAndSwap → RuntimeStore::current (S3 Published)
      │           │
      │           └─ old world → Retire ownership (S3→S4)
      │                 │
      │                 └─ DeferredDeletionQueue(4096, S4) → Quarantine(512+512, S5)
      │                       → TerminalReclaim(K slots, S6)
      │                             │
      │                             └─ drain(minReaderEpoch, H_max 保証) → S7 Released
      │                                   │
      │                                   └─ Budget release → S0 Available
      │
      └─ Shutdown ──→ drainAll() → S7 → S0（all pending）
```

- 本 chain は **一つの budget（K_world）の下で conservation が成立する** ことを invariant として保証する。
- D101-6 の `Budget Authority → Admission → RuntimeWorld ownership → Retire → Quarantine → Terminal → Budget release` の chain を、State Machine として具体化したものである。

---

## 11. Authority mapping

| Invariant / State | Authority | State 所在 | 判定 |
| --- | --- | --- | --- |
| `K_world` (Budget Pool) | **Lifetime Budget Authority**（RuntimeWorldAuthority 拡張を第一候補） | `std::atomic<int> budgetAvailable_` + `K_world` 定数（I4 の `pendingIntentCount_` と同型だが別 budget） | **DESIGN** — 新設 |
| `A_max` | Lifetime Budget Authority（admission gate、S0→S1 の前） | `A_max` 定数 + interval 内の acquire count | **DESIGN** — 新設 |
| `P_max` | Lifetime Budget Authority（rate gate、S0→S1 の前） | `P_max` 定数 + 時間窓内の publish count | **DESIGN** — 新設 |
| `A_max`/`P_max` と `K_world` の関係 | Lifetime Budget Authority が統合して管理 | `A_max` は burst 制御、`P_max` は sustained 制御、両者は `K_world` の流入側を上界 | **DESIGN** — 統合 |
| `S1 Reserved` | Lifetime Budget Authority | `reservedCount_` | **DESIGN** — 新設 |
| `S2 Transferred` | `RuntimeWorldAuthority::OwnerChannel` + `PendingPublishRegistry` | `aligned_unique_ptr<RuntimeState>` + seqId 64 slots | **EXISTING** — 既存を流用 |
| `S3 Published` | `RuntimeWorldAuthority::Store`（WriteAccess::publishAndSwap） | `std::atomic<RuntimeState*> current`（単一スロット） | **EXISTING** — 既存を流用 |
| `S4 Retiring` | `ISRRetireRouter` → `DeferredDeletionQueue` | `DeletionEntry` ring buffer(4096) | **EXISTING** — 既存を流用 |
| `S5 Quarantined` | `RetireQuarantineStore` ×2 | `QuarantinedEntry` array(512) ×2 | **EXISTING** — 既存を流用 |
| `S6 Terminal` | `TerminalReclaimAuthority<K>`（bounded 化） | `std::array<Entry, K>` + `residentAtomic_` | **DESIGN** — `std::vector` → `std::array<K>` 置換 |
| `S7 Released` | Lifetime Budget Authority（Budget release） | Budget Pool への返却 | **DESIGN** — 新設 |
| `H_max` | `EpochDomain` + `AudioEngine.Processing` | `H_max` 定数 + watchdog | **DESIGN** — D101-8 以降 |
| `G_contract` | `ISRWorldRetirementTelemetry`（観測）+ Lifetime Budget Authority（強制） | `G_contract` 定数 | **DESIGN** — D101-8 以降 |
| Budget separation | I4 Contract（契約レベル） | 契約記述 | **DESIGN** — I4 追記 |

### Authority 所在の原則（D101-6 継承）

- **Lifetime Budget Authority** は `RuntimeWorldAuthority` の拡張として配置するのが自然である。`RuntimeWorldAuthority` は既に `RuntimeStore`（current の唯一の write gateway）、`OwnerChannel`（所有権移転の唯一の point）、`PendingPublishRegistry`（gap 管理）を所有しており、Budget Pool の管理はこれらと同一の責務境界に属する。
- `ISRRuntimePublicationCoordinator::pendingIntentCount_` は **流用しない**。Lifetime Budget の `budgetCount_` は新設する。

---

## 12. Enforcement points

| Invariant | Enforcement point | 既存 code | 新 enforcement |
| --- | --- | --- | --- |
| `K_world` budget | `RuntimeWorldAuthority::publish` の admission 境界（S0→S1 reserve） | 存在しない | Budget Pool から reserve、失敗→backpressure（BLOCK） |
| `A_max` | 同上（S0→S1 reserve の前の interval gate） | 存在しない（telemetry 観測のみ） | Interval 内の acquire count で gate |
| `P_max` | 同上（S0→S1 reserve の前の window gate） | 存在しない | 時間窓内の publish count で gate |
| `M_terminal ≤ K` | `TerminalReclaimAuthority::store()` の呼び出し境界（S5→S6） | `std::vector` growable（enforcement なし） | `std::array<K>` + 予約済み slot への配置であり失敗しない |
| `H_max` | `EpochDomain::isOlder()` / `AudioEngine.Processing` の block 境界（S4/S5/S6→S7 drain） | 存在しない | `H_max` 超過時に HealthEvent + producer backpressure |
| `G_contract` | `ISRWorldRetirementTelemetry` の gap 検出境界 → `RuntimeWorldAuthority` の publish 境界 | 存在しない（observed maximum のみ） | Gap 超過時に bounded recovery + producer admission control |
| Budget separation | I4 Contract の契約境界 | `kMaxLogicalRecoveryObligations` のみ | I4 追記で分離を明示 |
| State transition ordering | 各遷移の Authority 境界（chapter 5.3） | 部分的に存在（S2→S3 の publishAndSwap 等） | S0→S1 reserve を新設し、全遷移の唯一の Authority を確定する |

---

## 13. Failure matrix

| # | Failure | 状態 | 予約の状態 | 対処（rollback/release） | 唯一の Authority | invariant |
| --- | --- | --- | --- | --- | --- | --- |
| F1 | Build failure（OOM / validation） | S1 Reserved（未 transfer） | Reserved（未 ownership） | **Rollback**: S1→S0（reservation cancel）。所有権は発生していないため UAF なし | Lifetime Budget Authority | `K_world_occupied` は増加しない |
| F2 | Admission reject（K_world 満杯 / A_max / P_max 超過） | S0 Available（reserve 失敗） | 未予約 | **Reject**: backpressure（BLOCK）。予約自体が発生しないため rollback 不要 | Lifetime Budget Authority | `K_world_occupied` は変化しない |
| F3 | Push failure（Intent queue 全層溢れ） | S1 Reserved（push 前） | Reserved | **Rollback**: S1→S0（`fetchSub rollback` と同型）。既存パターンと同様に Budget Pool へ release | Lifetime Budget Authority | `K_world_occupied` は増加しない |
| F4 | Publication failure（publishAndSwap 失敗） | S2 Transferred（commit 前） | Transferred | **Rollback**: S2→S0（OwnerChannel の owner を破棄 + Budget Pool へ release）。Stale 検出後に release | RuntimeWorldAuthority + Lifetime Budget Authority | `K_world_occupied` は S2→S0 で減少する |
| F5 | Shutdown race（shutdown 時の admission close） | S1 Reserved だが未 publish の全 slots | Reserved | **Rollback**: 全 S1→S0（一括 cancel）。Shutdown は新規 reservation を拒否 | Lifetime Budget Authority | Shutdown は新規 reservation を停止するため `K_world_occupied` は増加しない |
| F6 | Retire failure（DeferredDeletionQueue enqueue 失敗） | S3/S4（retire 経路） | Owned（S4 相当） | **No rollback**: enqueueWithRetry の chain で Q→EQ→Terminal へ退避。Budget release は drain 後（S7→S0）まで遅延する。S4→S5→S6 の遷移で owned は維持される | ISRRetireRouter | `K_world_occupied` は変化しない（S4→S5→S6 は owned 内の移動） |
| F7 | Quarantine overflow（RetireQuarantineStore 満杯） | S5 Quarantined（溢れ時） | Owned（S5→S6） | **No rollback**: EmergencyQ → TerminalReclaim へ退避。Bounded TerminalReclaim は予約済み slot のため失敗しない | RetireQuarantineStore → TerminalReclaimAuthority | `K_world_occupied` は変化しない（S5→S6 は owned 内の移動） |
| F8 | Terminal reclaim failure（bounded K 満杯） | S6 Terminal（理論上発生しない） | Owned（S6） | **No failure**: 予約済み slot への配置であり失敗しない（allocation-free）。D101-7 の reservation-first により、S0→S1 reserve 時に K が確保済みのため、S5→S6 で失敗することはない | TerminalReclaimAuthority | `K_world_occupied` は変化しない。reservation-first により S5→S6 の失敗は設計上発生しない |

### 重要な invariant

```text
各 failure において rollback / release が必ず一度だけ起きること:

  F1: S1→S0 rollback（exactly once、S1 にある reservation を cancel）
  F2: rollback なし（予約自体が発生していない）
  F3: S1→S0 rollback（exactly once）
  F4: S2→S0 rollback（exactly once、OwnerChannel の owner 破棄 + Budget release）
  F5: 全 S1→S0 rollback（exactly once per reserved slot）
  F6/F7: rollback なし（owned 内の移動、release は drain 後）
  F8: failure 自体が発生しない（reservation-first により設計上排除）

Double-release 防止: 各 reservation は S1〜S6 のいずれかに exactly once で存在し、
S1→S0（rollback）と S7→S0（release）は排他的である。
State Machine の遷移制約により、同一 reservation が両方で release されることはない。
```

---

## 14. Shutdown contract

### 14.1 Shutdown の定義

```text
Shutdown contract: 新規 reservation/admission の停止 + outstanding budget の有限時間収束

Invariant: shutdown 開始後、K_world_occupied → 0 を有限時間で証明できる条件

Steps:
  1. Admission close: 新規 publish の reservation を拒否（Path B gate と同様）
     新規 RuntimeWorld の生成を停止 → K_world の新規消費を停止
     Authority: Lifetime Budget Authority（S0→S1 reserve の拒否）
     Existing: ConvoPeq.md:56686「shutdown 確定後は Publish Intent を Path B admission gate で拒否」

  2. Pending gap drain: PendingPublishRegistry の残留を全て処理
     登録済みだが未 commit の worlds を commit または drop（S2→S3 または S2→S0）
     Authority: RuntimeWorldAuthority::PendingPublishRegistry
     Existing: ConvoPeq.md / AudioEngine.ReleaseResources.cpp の gap 処理

  3. Retire drain: DeferredDeletionQueue / RetireQuarantineStore の残留を drain
     minReaderEpoch の進行を待ち、epoch-safe 到達後に deleter → S7→S0
     Authority: ISRRetireRouter::drain() / RetireQuarantineStore::drain()
     Guarantee: H_max < ∞ により epoch は有限時間で進行する（I04）

  4. Terminal drain: TerminalReclaimAuthority::drainAll() で全 pending を強制解放
     Bounded K でも drainAll() は全 entry を走査して解放可能
     Authority: TerminalReclaimAuthority::drainAll()
     Condition: K が shutdown 時の worst-case 残留数を上回ること（I10 の条件）

  5. Budget release: 全 drain 完了後、Budget Pool の全 slots が返却され Verify Empty
     K_world slots 全てが free（S0 Available = K_world）→ shutdown 完了
     Authority: Lifetime Budget Authority（S7→S0 の全 release）
```

### 14.2 `K_world → 0` を有限時間で証明できる条件

| 条件 | 内容 | 本監査での位置づけ |
| --- | --- | --- |
| Admission close | Shutdown 開始後に新規 reservation を停止する | S0→S1 reserve の拒否。既存の Path B gate と同様 |
| `drainAll()` の有限完了性 | Bounded K でも `drainAll()` が全 pending を有限回で解放する | `K` が worst-case 残留数を上回ること（D101-7 で K の値確定後に証明） |
| `drainAll()` との関係 | `drainAll()` は audio thread 停止後に全 pending を強制解放する。epoch に依存しないため、H_max の保証がなくても完了する | Shutdown の最終段階では `H_max` に依存しない。`drainAll()` 自体が有限完了性の保証である |
| `K_world → 0` の証明 | `K_world_occupied = reserved + owned` が shutdown 開始後に単調減少し、有限時間で 0 に収束すること | Admission close（新規増加停止）+ drainAll()（既存の強制解放）の組合せで証明可能 |

### 14.3 Shutdown と他 invariant の接続

| invariant | Shutdown との接続 |
| --- | --- |
| I01 `M_terminal` | Shutdown 時の Terminal 残留は `K` の範囲内。`K` が worst-case を上回れば drainAll() は成功する |
| I04 `H_max` | Shutdown 前の drain は `H_max` に依存するが、Shutdown の drainAll() は `H_max` に依存しない（audio thread 停止後の強制解放） |
| I07 `G_contract` | Shutdown 時の gap は admission close により新規 publish が停止するため、gap 自体が収束する |
| I10 `M_world` | Shutdown 完了時に `M_world → 0`（全 worlds が reclaim された）。`K_world → 0` と同義 |

---

## 15. 他 invariant との接続

### 15.1 D101-5 の I01〜I10 との接続

| invariant | 本監査（D101-7）での接続 | 状態 |
| --- | --- | --- |
| I01 `M_terminal` | S6 Terminal の bounded capacity `K`。State Machine の S5→S6 遷移で `K` の範囲内で配置される | **DEFINED**（本監査で State Machine として定義） |
| I02 reservation-first | S0→S1 reserve の ordering。S1→S2 transfer の前に reservation を取得する invariant | **DEFINED**（本監査で ordering として定義） |
| I03 `H_max` | S4/S5/S6→S7 drain の epoch 進行保証。`H_max` がなければ drain が停止する | **DEFINED**（骨格のみ、具体的数値は D101-8） |
| I04 `A_max` | S0→S1 reserve の前の interval gate。`A_max` 超過で backpressure（BLOCK） | **DEFINED**（本監査で invariant として定義） |
| I05 `P_max` | S0→S1 reserve の前の window gate。`P_max` 超過で backpressure（BLOCK） | **DEFINED**（本監査で invariant として定義） |
| I06 `G_contract` | Sampler gap の契約。`G_contract` 超過時に bounded recovery + admission control | **DEFINED**（骨格のみ、具体的数値は D101-8） |
| I07 budget separation | I4 の `kMaxLogicalRecoveryObligations` と Lifetime Budget `K_world` の分離 | **DEFINED**（D101-6 で分離を確定、本監査で再確認） |
| I08 ownership conservation | State Machine の conservation chain（`reserved + owned ≤ K_world`）として統合 | **DEFINED**（本監査で C1〜C4 として定義） |
| I09 shutdown | S4/S5/S6→S7→S0 の shutdown contract（drainAll() 有限完了性） | **DEFINED**（本監査で shutdown contract として定義） |
| I10 `M_world` decomposition | `M_world ≤ 1 + 4096 + 1024 + K + f(H_max, P_max) ≤ K_world < ∞`。全 invariant の統合として定義 | **DEFINED**（本監査で分解式として定義） |

### 15.2 Cross-invariant dependency への影響

D101-5 で確定した依存グラフに対する本監査の影響:

```text
I08 (budget separation) ──→ I02 (reservation-first) ──→ I09 (ownership conservation)
     │                            │                          │
     └────────────→ I01 (M_terminal) ────────────────────────┘
                                                    │
I04 (H_max) ──→ I03 (M_reader) ───────────────────→ I10 (M_world)
     │                    ↑
     └────→ I07 (G_contract) ─┘

I05 (A_max) ──→ I10 (M_world)（流入側、S0→S1 gate）
I06 (P_max) ──→ I10 (M_world)（流入側、S0→S1 gate）
```

- 本監査は I08 + I02 の Budget Authority（S0→S1 reserve）を State Machine として具体化し、依存グラフの根元を確定した。
- I04/I05/I06/I07 の具体的数値は D101-8 で導出するが、State Machine 上の位置（S0→S1 gate / S4/S5/S6→S7 drain）は本監査で確定した。

---

## 16. Open questions

| # | 質問 | 現時点の見解 | 解決フェーズ |
| --- | --- | --- | --- |
| Q1 | `K_world` の具体的数値は？ | `M_world` の分解式から導出するが、具体的値は D101-8 で決定。本監査では `K_world < ∞` の invariant と counting 対象を閉じた | D101-8 |
| Q2 | `A_max` の具体的数値は？ | Interval の定義と `K_world` / `H_max` に依存する。D101-8 で導出 | D101-8 |
| Q3 | `P_max` の具体的数値は？ | Window の定義と `H_max` に依存する。D101-8 で導出 | D101-8 |
| Q4 | `H_max` / `G_contract` の具体的数値は？ | `maxBlockDuration + ε` / `block 時間の数倍` として設計時に固定。D101-8 で導出 | D101-8 |
| Q5 | Lifetime Budget Authority の具体的 State（`std::atomic<int> budgetAvailable_` 等）の型と配置 | `RuntimeWorldAuthority` の拡張として `std::atomic<int> budgetCount_` を新設するのが第一候補。PendingPublishRegistry 64 とは別 State | D101-8 |
| Q6 | `COALESCE` 相当の重複 publish は新規 reservation を取得しないか | **取得しない**（I4 D18.4 と同様）。ただし RuntimeWorld の coalesce 条件は I4 の CoalesceIdentity とは独立に定義する必要がある | D101-8 |
| Q7 | `K` が shutdown 時の worst-case 残留数を上回ることの保証（循環依存） | `K` の値は D101-8 で導出するが、`K ≥ worst-case shutdown 残留数` の条件を満たす必要がある | D101-8 |
| Q8 | `pendingIntentCount_` との具体的分離（naming / State / Authority） | 対象・粒度・Scope が異なるため、新独立 budget state として設計する。Naming は `budgetAvailable_` / `budgetCount_` 等で明示的に分離する | D101-8 |
| Q9 | Conservation の形式的証明（reserved + owned ≤ K_world の帰納的証明） | State Machine の全遷移で C2 が維持されることの帰納的証明。D101-8 で形式的証明を行う | D101-8 |

---

## 17. Verdict

### 判定: `CONTRACT_DEFINED`

| 判定 | 定義 | 本監査の該当性 |
| --- | --- | --- |
| `SATISFIED` | 既存コードで Lifetime Budget State Machine が充足している | **該当せず** — 既存 `pendingIntentCount_` は対象・粒度・Scope が異なるため流用不可 |
| `PARTIAL` | 部分的に充足し、拡張で充足可能 | **該当せず** — パターンのみ参考になるが、State Machine 及び全状態の定義は新設が必要 |
| `MISSING` | 既存コードに Lifetime Budget State Machine が存在しない | **部分的該当** — State Machine 自体は MISSING だが、本監査で契約として定義できたため、単なる MISSING ではない |
| `CONFLICT` | 現行設計と Lifetime Budget State Machine が衝突する | **該当せず** — D101-5 で衝突 0 件を確認済み。既存 Authority の拡張で実現可能 |
| `CONTRACT_DEFINED` | 骨格は本監査で契約として定義できたが、具体的 state / ordering / 数値 / 証明は D101-8 での詳細設計を要する | **◯ 該当（本監査の結論）** |

### なぜ `CONTRACT_DEFINED` か

- **State Machine は定義できた**: 8状態（Available → Reserved → Transferred → Published → Retiring → Quarantined → Terminal → Released）と各遷移の唯一の Authority を確定し、conservation invariant（`K_world_occupied ≤ K_world`）を定義できた。
- **`K_world`/`A_max`/`P_max`/reservation token の invariant と ordering は定義できた**: 各 bound の意味、enforcement point、reject/rollback 条件、`K_world` との関係を閉じた。数値は D101-8 で導出するが、「なぜ十分であり、どこで強制され、どの状態を含み、どの遷移で減少するか」を本監査で閉じた。
- **Failure matrix / Shutdown contract / 他 invariant との接続も定義できた**: 全 failure path の rollback/release を網羅し、shutdown の有限完了性（`K_world → 0`）を証明できる条件を閉じた。
- **しかし具体的 state / 数値 / 証明は D101-8 を要する**: `K_world` / `A_max` / `P_max` / `H_max` / `G_contract` の具体的数値、`Budget Pool` の具体的 State 型、`reservation → transfer → release` の具体的 ordering の形式的証明は、D101-8 での詳細設計を要する。

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
D101-6  DESIGN_REQUIRED（Lifetime Budget Authority の骨格定義）
   │
   ▼
D101-7  CONTRACT_DEFINED  ◀ 本監査
   │  Lifetime Budget State Machine（8状態 + 各遷移の唯一の Authority）
   │  K_world / A_max / P_max / reservation token の invariant と ordering
   │  Failure matrix（全 failure path の rollback/release を網羅）
   │  Shutdown contract（K_world → 0 を有限時間で証明できる条件）
   │  他 invariant との接続（I01〜I10 の統合）
   │
   ▼
D101-8 — Lifetime Budget Authority の具体的設計 + reservation-first 詳細設計
   │
   ├── K_world / A_max / P_max / H_max / G_contract の具体的数値と State 型
   ├── reservation → transfer → rollback → release の具体的 ordering と形式的証明
   ├── I4 Contract 追記（budget separation）
   └── conservation の形式的証明
   │
   ▼
D101-9 — TerminalReclaim bounded 化 + drainAll() 有限完了性の設計
   │
   ▼
Phase I GO/NO-GO 再判定
```

- **本監査でも production code は変更しない**（指示どおり）。
- D101-8 では、本監査で定義した State Machine / Contract に基づき、Lifetime Budget Authority の具体的設計（`K_world` / `A_max` / `P_max` の数値と State 型、ordering の形式的証明）と reservation-first 詳細設計を行う。

---

## 付録: D101-7 監査チェックリスト

- [x] Lifetime Budget State Machine を定義（Available / Reserved / Transferred / Published / Retiring / Quarantined / Terminal / Released の8状態）
- [x] 各遷移の唯一の Authority を確定（Singularization）
- [x] `K_world` の定義（Budget unit = RuntimeWorld、K_world が何を数えるか、RuntimeStore::current をどう数えるか、Pending/Retire/Quarantine/Terminal をどう conservation に含めるか）
- [x] `K_world` の counting 対象（S1〜S6）と enforcement point（S0→S1 reserve）を確定。数値は決めない
- [x] `A_max` の定義（acquire の厳密化、interval の定義、唯一の enforcement point、reject/rollback 条件、K_world との関係）
- [x] `P_max` の定義（publish の定義、時間窓の定義、burst と sustained の分離、PendingPublishRegistry 64 との非同一性、唯一の Authority）
- [x] `P_max` と `PendingPublishRegistry 64` の非同一性を明示（gap 容量 ≠ 頻度上限）
- [x] Reservation token / ownership の定義（何を所有するか、token/state の表現、double-release 防止、leak 防止、transfer 前後の invariant）
- [x] 既存 `pendingIntentCount_` を昇格させる設計を避け、RuntimeWorld lifetime 用の独立した budget state として設計
- [x] Failure matrix（build/admission/push/publication/shutdown/retire/quarantine/terminal の各 failure で rollback/release が必ず一度だけ起きること）
- [x] Shutdown contract（新規 reservation/admission の停止点、outstanding budget の収束、drainAll() との関係、K_world → 0 を有限時間で証明できる条件）
- [x] 他 invariant との接続（I01 M_terminal / I02 reservation-first / I03 H_max / I04 A_max / I05 P_max / I06 G_contract / I07 budget separation / I08 ownership conservation / I09 shutdown / I10 M_world decomposition）
- [x] 「その数値がなぜ十分であり、どこで強制され、どの状態を含み、どの遷移で減少するか」を閉じる（数値自体は決めない）
- [x] Production code 変更なし（contract 定義のみ）
