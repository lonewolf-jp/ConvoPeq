# D101-8 Step 2 — Reservation Token Semantics

| 項目 | 内容 |
| --- | --- |
| **ID** | D101-8 Step 2 |
| **日付** | 2026-08-20 |
| **対象ブランチ** | `main` |
| **一次資料** | `ConvoPeq.md`（2026-08-20 最新ソース連結 103K+ lines）、`src/audioengine/RuntimeWorldAuthority.h`、`src/audioengine/AudioEngine.h`（`commitRuntimePublication` / `BuilderToken`）、`src/audioengine/OwnerChannel.h`、`src/core/RuntimeStore.h`、`src/MpscBoundedRing.h`、`src/audioengine/PublicationExecutor.cpp`、`src/audioengine/RuntimeBuilder.cpp`、D101-8 Step 0/Step 1 evidence |
| **前提** | D101-8 Step 1 verdict: `FORMALIZED_WITH_OPEN_PROOFS` — World Identity Contract / Ownership Location Table / Layer A/B/C 分離 / I-W1..I-W8 / O4/O6 proof boundary を確定。`M_world <= B_world <= K_world` は未証明として継承 |
| **目的** | `S0 -> S1` の admission/reservation から `S6 -> S7` の release までの全 lifecycle に対して「1 reservation = 1 budget token」の意味論がコード上で成立するかを検証する。まだ `K_world` の数値は出さない |
| **制約** | **コード変更なし・semantic proof boundary の確定のみ**。数値導出 / 型設計 / Budget Authority 実装 / OwnerChannel/Registry 変更は禁止 |
| **判定** | **RESERVATION_SEMANTICS_DESIGN_REQUIRED — CODE-EVIDENCE MISSING** — 現行 production code に Lifetime Budget reservation token は存在せず、`B_world` / `K_world` は semantic model のみ。`I-W5` は `PARTIAL` に再判定、`O7` は `SEMANTIC MODEL ONLY / DESIGN-DEFINED` |

---

## 1. Scope

Step 1 で「lifecycle(W,t) は唯一」「OwnerCount(W,t) ∈ {0,1}」「RegistryCount は M_world に含めない」「M_world(t)={W:lifecycle∈S2..S6} / B_world(t)=|{b:state∈S1..S6}|」を確定したが、**B_world の実体である reservation token がコード上のどの event に対応するかが未特定**のまま残った。

Step 2 は Step 3（A_max = S0->S1 reservation 許可数の derivation）の前提として、「reservation event が何なのか」をコードで完全に固定する工程である。Step 2-A..I の順で、reservation 実体 / acquire / rollback / transfer / publish-retire persistence / release / double-release / leak / Token-World correspondence を検証する。

```text
S0 -> [R1 Reserved] -> Build -> S2 Transferred -> S3 Published
  -> S4 Retiring -> S5 Quarantined -> S6 Terminal -> [R8 Released]
                              \-> failure rollback -> [R8 Released]

```text
---

## 2. Step 2-A — Reservation の実体を特定する

### 2.1 調査方法

`ConvoPeq.md` / `src/` に対して以下を**推測せずに**列挙した。

- `reservation` / `Reservation` / `B_world` / `K_world` / `budget token` / `admission` / `pendingIntentCount` の全ヒット（219 matches in ConvoPeq.md）
- `BuilderToken` / `createForBuilder` / `worldId` の全ヒット（57 matches）
- `OwnerChannel` / `PendingPublishRegistry` / `registerPublish` の全ヒット
- `commitRuntimePublication` / `pendingIntentCount_` の全定義

### 2.2 S0->S1 で現在コードに存在するものの列挙

> **Core Clarification (2026-08-20 audit §1.1)**: 現行 production code には reservation/residency semantics が複数の別ドメインに実装されている（Intent transport reservation / Publish Intent residency reservation / durable Recovery admission reservation）が、それらはいずれも **Lifetime Budget for RuntimeWorld の reservation ではない**。`B_world` を 1 unit として保持し、S1 から S6 まで World lifecycle に伴って存続し、terminal reclaim で exactly-once release される **Lifetime Budget reservation token は現行 code に存在しない**。

| 候補 | 現行コードでの実体 | 型 / 変数 / API | S0->S1 との関係 | 判定 |
| --- | --- | --- | --- | --- |
| **L-B reservation counter** | **存在しない** | `B_world` / `K_world` / `LifetimeBudgetReservation` という識別子は `src/` に 0件。`ConvoPeq.md` でも D101-2 以降の evidence 内の semantic term としてのみ出現 | S0->S1 の **Lifetime Budget reservation** を count する機構は production code にない | **MISSING** |
| **Intent/Recovery reservation** | `pendingIntentCount_` と `publicationIntentResidencyCount_` と `PendingRecoveryAdmission::reservationOwned` が **複数の reservation/residency semantics** を持つ | `std::atomic<uint64_t> pendingIntentCount_` / `publicationIntentResidencyCount_` / `PendingRecoveryAdmission::reservationOwned = true`（`1 logical admission = 1 reservation`） | (a) `pendingIntentCount_` — Observe/Quarantine/Recovery Intent の transport residency + producer enqueue reservation（reservation-before-push / consume-after-pop conservation pattern）；(b) `publicationIntentResidencyCount_` — Publish Intent 専用の producer-side enqueue reservation + queue residency；(c) `reservationOwned` — Recovery durable admission に対して `1 logical admission = 1 reservation` を明示し、queue full 時は transport residency から durable admission へ reservation を移動する lease semantics（DurablePending -> Building） | **EXISTING / Intent/Recovery domain** — **Lifetime Budget for RuntimeWorld の reservation ではない** |
| **budget token** | **存在しない** | `budget token` は `ConvoPeq.md` に 0件（evidence の semantic term のみ） | Lifetime Budget の token という概念はコード上に型として存在しない | **MISSING** |
| **reservation handle** | **存在しない** | `ReservationHandle` / `AcquireHandle` 等の型は `src/` に 0件 | handle を acquire/release する機構はない | **MISSING** |
| **acquire/release pair** | **存在しない** | `acquireReservation` / `releaseReservation` 等の API は `src/` に 0件 | acquire/release の対はコード上にない | **MISSING** |
| **pending reservation** | `PendingPublishRegistry`（64 slots）が pending の registry | `PendingPublishRegistry::registerPublish(seqId, worldPtr)` / `lookup` / `unregister` | **non-owning metadata**（`const void*`）。`OwnerChannel` の owning transport とは別。reservation token ではない | **別物** — metadata gap |
| **build admission** | `BuilderToken` | `RuntimeState::BuilderToken` / `RuntimeState::createForBuilder(BuilderToken)` | `RuntimeState` 生成の権限トークン。Budget reservation とは無関係（2.3 参照） | **別物** — build 権限 |
| **MpscBoundedRing reservation** | `MpscBoundedRing::push` の CAS slot 予約 | `enqueuePos CAS -> reservation order -> publication` | Intent queue の slot 予約。World lifetime の reservation ではない | **別物** — queue slot |
| **OwnerChannel slot** | `OwnerChannel<RuntimeOwner>::Slot::owner` | `aligned_unique_ptr<const RuntimeState>` per slot（256 slots） | S2 の owning transport。reservation token ではない（2.3 参照） | **別物** — owning transport |

### 2.3 なぜ既存の何物も B_world ではないか

```text
B_world(t) = |{ Lifetime Budget reservation token b : state(b,t) ∈ R1..R7 }|
```

という定義に対して、現行コードで `b` に該当する semantic object は存在しない。

- `pendingIntentCount_` は `RuntimeIntentCoordinator` の reservation-like counter として **実際に conservation pattern（push成功時`fetchAdd` / pop成功時`fetchSub` / 絶対値リセット廃止）を持つ**（INV-ISR-02）。しかし **Intent transport residency の budget/residency reservation** であり、**unit / scope / cardinality が異なる**ため `B_world` の実体ではない — `B_world` は Lifetime Budget の World 数（`R1..R7` にわたる）、`pendingIntentCount_` は Intent 数である（2.2節の表を参照）。
- `publicationIntentResidencyCount_` は Publish Intent 専用の producer-side enqueue reservation + queue residency を count する。これも `B_world` ではない。
- `PendingRecoveryAdmission::reservationOwned` は Recovery durable admission に対して `1 logical admission = 1 reservation` を明示する lease semantics を持つ。これも Lifetime Budget World reservation ではない。

**結論**: 現行コードには reservation/residency semantics が複数の別ドメインに実装されているが、**Lifetime Budget reservation token は存在しない**。`B_world` は current code において semantic quantity のみである。`pendingIntentCount_` / `publicationIntentResidencyCount_` / `PendingRecoveryAdmission::reservationOwned` はそれぞれ Intent/Recovery domain の reservation/residency semantics を持つが、いずれも `B_world` の実体ではない。これらを「単なる似た名前の counter」と過小評価することもなく、`B_world` と誤同一視することもない。

### 2.4 Reservation Table（要求された表）

| 項目 | 必須確認 | 現行コードでの実体 | Evidence |
| --- | --- | --- | --- |
| Reservation semantic object | 存在するか | **存在しない** — `B_world` は semantic model のみ | `src/` grep 0件 / `ConvoPeq.md` は evidence 内の term のみ |
| 実際の型 / 変数 / API | 型があるか | **なし** — `LifetimeState::pendingIntentCount_` は別物（RetireIntent 用） | `RuntimeWorldAuthority.h` / `ISRRetire.h` |
| Create event S0->S1 | 具体的 call site | **なし** — `commitRuntimePublication` は S1->S2 の transfer であり S0->S1 ではない（2.6 参照） | `AudioEngine.h:4587` / `PublicationExecutor.cpp:53` |
| Acquire authority | 誰が許可するか | **未定義** — admission authority は将来の Budget Authority で定義すべき | Step 1 FORMALIZED_WITH_OPEN_PROOFS を継承 |
| Transfer event S1->S2 | token がどう存続するか | **該当なし** — token が存在しないため transfer も存在しない。Build 成功後に `registerPublish` + `OwnerChannel::enqueue` が行われるが、これは ownership transfer であり reservation transfer ではない | `RuntimeWorldAuthority.h:43` / `OwnerChannel.h` |
| Release event S6->S7 / rollback | 具体的 call site | **なし** — `S6->S7` は `deleter(world)` による reclaim であり reservation release ではない | `ISRRetireRouter` / `TerminalReclaimAuthority` |
| Failure rollback | Build failure 時の release | **なし** — Build failure 時に reservation release すべき対象が存在しない | `RuntimeBuilder.cpp:68,201` |
| Shutdown rollback | S2 shutdown 時の release | `OwnerChannel::drainAllNonRt()` が shutdown 時の owner 回収を行うが、reservation release ではない | `OwnerChannel.h: drainAllNonRt` |
| Double release guard | ある / ない | **MISSING** — token がないため guard もない | — |
| Leak guard | ある / ない | **MISSING** — token がないため guard もない | — |
| Thread authority | producer / coordinator / builder / shutdown | **未定義** — 将来の Budget Authority で定義すべき | — |
| Boundedness | 証明可能 / 不能 | **不能** — `K_world` が未定義のため | Step 1 を継承 |

### 2.5 I4_DESIGN_CONTRACT.md D48-D53/D69 — `WorldRetirementReservation` design contract との照合

**最新 `ConvoPeq.md`（2026-08-20）と `doc/work88/I4_DESIGN_CONTRACT.md`（D48-D53/D54/D69）を照合した結果、以下の重要な区別を明確にする。**

#### 2.5.1 Design contract は存在する（D48-D53）

`I4_DESIGN_CONTRACT.md` D48-D53 では、 `WorldRetirementReservation` という **Lifetime Budget reservation authority** の design contract が既に定義済みである（Design-30/34/35/36 として 2026-08-15 に確定）：

- **D48**: `WorldRetirementReservation::acquire(key = prevWorld identity)` / `release(key)` / `current_reserved()` の authority 一本化。invariant: `N_retired_world ≤ R`。acquire location = `publish()` 内・`publishAndSwap` の前。release location = deferred-delete deleter 内。すべて **CLOSED（by construction・コード経路裏付け）**。
- **D52**: `reservation token` は `worldId` ではなく `WorldRetirementReservation` が monotonic に mint する token である。freed-pointer independence（`release` は freed pointer に依存しない）を CLOSED。context lifetime（INV-R6）を AudioEngine member で確定。
- **D53**: `INV-TOKEN-1` — `successful acquire(token) ⇒ 同一 token を保持する queued entry が正確に 1 つ`。6確認点（acquire token mint 位置、DeletionEntry への token/context 格納、entry 間 token 複製・消失なし、quarantine drain で同一 token 到達、stale/double release 検出）すべて **CLOSED（design contract）**。
- **D54**: `INV-TOKEN-1` を terminal-held を含む完全 invariant に修正。`WorldRetirementReservation` は固定容量の held-token storage を所有。**CLOSED**。

#### 2.5.2 Phase I implementation = NO-GO (コード未変更)

**しかし D53/D69 は明示的に Phase I implementation = NO-GO / コード未変更 とする。**

- D53: `Phase I implementation = NO-GO（ユーザー最終 GO 待ち・コード未変更）`
- D69.1: **Phase I-T1（telemetry 測定装置のみ）**: count 追跡（acquire +1 / release -1）+ telemetry counter。**`held-set / free-stack / token / R gate は含まない`**（R gate は Phase I-T2 で実装）。
- D69.5: `D68（挿入点・RT/non-RT 境界・lifetime ordering）` は CLOSED だが、**実装開始時に AudioEngine lifetime / destructor 順序 / publish/swap 実コード経路 / retire deleter 所有関係と再突合する必要がある**（設計上の挿入点 ≠ 現行コード上その位置が安全）。

#### 2.5.3 Step 2 の verdict への含意

| 判定対象 | Design contract | Production code | Step 2 verdict への影響 |
| --- | --- | --- | --- |
| `WorldRetirementReservation` | **DEFINED / CLOSED**（D48） | **未実装** — `WorldRetirementReservation` 型は `src/` に 0件 | design contract ≠ implementation |
| `acquire/release API` | **CLOSED**（D48/D52） | **未実装** — `src/` に `acquireReservation`/`releaseReservation` 0件 | `CODE-EVIDENCE MISSING` |
| `token mint / release` | **CLOSED / design-only**（D53） | **未実装** | `CODE-EVIDENCE MISSING` |
| `INV-TOKEN-1` (token continuity) | **CLOSED / design contract**（D53） | **未実装** | `CODE-EVIDENCE MISSING` |
| R gate (N_retired_world ≤ R) | **OPEN**（D53: R の数値は Policy・T104） | **未実装** | `CODE-EVIDENCE MISSING` |
| Phase I-T1 (telemetry only) | **CLOSED**（D69） | **未実装** — ユーザー最終 GO 待ち | `NO-GO` |

> **Critical distinction (2026-08-20 audit)**: Step 2 の `CODE-EVIDENCE MISSING` は **design contract 自体が存在しない** ではなく、**design contract は存在する（D48-D53）が production implementation が NO-GO である** ことを意味する。`WorldRetirementReservation` は `I4_DESIGN_CONTRACT.md` で CLOSED されたが、`src/` コード上では実装されていない。`A_max` derivation (Step 3) ではこの distinction を厳守する — `pendingIntentCount_` / Design contract の `WorldRetirementReservation` いずれも `A_max` の算定対象にしてはならない。

### 2.6 なぜ commitRuntimePublication は S0->S1 ではないか（facade 境界の厳密化）

```text
RuntimeBuilder::buildWorld()            — S1 pending 相当（local owner）
  -> worldAuthority_.registry().registerPublish(seqId, newWorld)  — S1->S2 準備（non-owning）
  -> worldAuthority_.ownerChannel().enqueue(key, std::move(world)) — S2 Transferred（owning）
  -> Intent enqueue (seqId, payload=newWorld ptr)                 — S2 gap（non-owning pointer）
  -> CoordinatorLoop::processIntent -> take(key) -> publish(std::move(owner)) — S2->S3

```text
`commitRuntimePublication` は **build 済みの World を publish pipeline に投入する facade** であり、その内部で `registerPublish` + `OwnerChannel::enqueue` + Intent enqueue を行う。つまり `commitRuntimePublication` は **S0->S1 の reservation acquire ではない**。より厳密には **`commitRuntimePublication` 自体は S1->S2 そのものではなく、S1->S2 に相当する ownership transfer operation を内部に含む producer-side publish-pipeline facade** である（内部で `registerPublish` → `OwnerChannel::enqueue` → Intent enqueue を包む）。semantic proof としては `R1 -> R2 = Build/World creation` と `R2 -> R3 = OwnerChannel ownership transfer` を分離すべきであり、`commitRuntimePublication` はこの複数操作を包む facade に過ぎない。

もし将来 Budget Authority を導入するなら、**build 前の S0->S1 で reservation を acquire** し、build 後に `R2 -> R3` で token を World と紐付ける設計が必要になる（Step 2-C 参照）。`commitRuntimePublication` の呼び出し数自体は `A_max` の代用にはならない。

---

## 3. Step 2-B — reservation と world ownership を完全に分離

### 3.1 4つの異なる概念

```text
reservation token        — Lifetime Budget の budget-unit（将来の型、現在は存在しない）
        ≠
RuntimeState ownership   — aligned_unique_ptr<const RuntimeState> の所有権（現行で OwnerChannel が保持）
        ≠
PendingPublishRegistry entry — seqId -> const void* の non-owning metadata（64 slots、overwrite 可能）
        ≠
OwnerChannel entry       — seqId -> Owner* の owning slot（256 slots、single-transfer）

```text
### 3.2 BuilderToken / reservation token / world identity の分離

| 概念 | 実体 | 役割 | reservation token と同一か |
| --- | --- | --- | --- |
| **BuilderToken** | `RuntimeState::BuilderToken`（empty struct、`constexpr` default ctor） | `RuntimeState` ctor の権限証。`createForBuilder(BuilderToken)` でのみ `RuntimeState` を生成可能にする access key | **同一ではない** — `BuilderToken` は build 権限の型レベル証跡であり、lifetime を束ねる budget token ではない。型として希少な resource capability ではなく RuntimeState construction の access key である（`constexpr BuilderToken() noexcept = default` が公開され `createForTest()` も `BuilderToken{}` を直接生成する — FUTURE-4） |
| **ReservationToken** | **将来の型**（現在は存在しない） | `S0->S1` で acquire、`S6->S7` / rollback で release される budget-unit。`B_world` の計数単位 | — |
| **RuntimeWorld identity** | `RuntimeState::worldId`（`uint64_t`、Authoritative） | 各 build で一意に付与される World の同一性。lifecycle 全体で不変 | **同一ではない** — `worldId` は `S2` 以降で存在する identity であり、`S1` の reservation 段階では存在しない（Step 1-C I-W7）。`worldId == reservation token` と仮定してはならない |

### 3.3 禁止事項の遵守

以下をコード証拠なしに同一視しない（指示どおり遵守）。

| 禁止同一視 | 判定 | 根拠 |
| --- | --- | --- |
| `BuilderToken == reservation token` | **同一視しない** | `BuilderToken` は空の権限証であり、`B_world` の計数単位ではない。`createForBuilder` は `S2` の World 生成であり `S0->S1` の reservation ではない |
| `worldId == reservation token` | **同一視しない** | `worldId` は `S2` 以降の identity であり、`S1`（token=1, world=0）の段階で存在しない |
| `OwnerChannel slot == reservation` | **同一視しない** | `OwnerChannel` は `S2` の owning transport（256 slots）であり、`S1..S6` 全体を束ねる token ではない |
| `Registry entry == reservation` | **同一視しない** | `Registry` は non-owning metadata（64 slots、overwrite 可能）であり、conservation を担えない |

---

## 4. Step 2-C — Reservation state machine

### 4.1 Semantic state machine（監査上の状態機械 — コードの enum 新設ではない）

```text
R0 = NoReservation       — S0 Available 相当。token も world も存在しない
R1 = Reserved            — S1 Reserved。reservation token 1、world 0
R2 = WorldCreated        — Build 直後。token 1、world 1（local owner）
R3 = Transferred         — S2。token 1、world 1（OwnerChannel）
R4 = Published           — S3。token 1、world 1（RuntimeStore::current）
R5 = Retiring            — S4。token 1、world 1（DeferredDeletionQueue）
R6 = Quarantined         — S5。token 1、world 1（RetireQuarantineStore Q/E）
R7 = Terminal            — S6。token 1、world 1（TerminalReclaimAuthority — World destruction authority）
R8 = Released            — S7。token 0、world 0（reclaim 済み）

```text
各 transition の詳細：

| Transition | Event | Precondition | Postcondition | Authority（将来） | Token ownership | World existence |
| --- | --- | --- | --- | --- | --- | --- |
| `R0 -> R1` | **Acquire**（reservation acquire） | `B_world < K_world`（将来の admission check） | `R1`、token=1, world=0 | **Budget Authority**（将来） | Budget Authority が token を発行 | なし |
| `R1 -> R2` | **Build**（`createForBuilder`） | `R1`、token=1 | `R2`、token=1, world=1（local） | Builder（Non-RT） | Budget Authority -> Builder へ token 紐付け | `RuntimeState` 生成（`worldId` 付与） |
| `R2 -> R3` | **Transfer**（`OwnerChannel::enqueue`） | `R2`、local owner あり | `R3`、token=1, world=1（OwnerChannel） | Producer（Non-RT） | Token は World W の budget を維持（token は World に紐付き ownership-bearing ではない）。**World ownership は local owner から OwnerChannel へ移る**（reservation ownership ≠ World ownership — Step 2-B 参照） | OwnerChannel が ownership を取得 |
| `R3 -> R4` | **Publish**（`take` + `publishAndSwap`） | `R3`、OwnerChannel に owner あり | `R4`、token=1, world=1（RuntimeStore） | PublishExecutor（ISR/audio） | Token は new world と共に RuntimeStore へ移動。old world は `R4->R5` へ | `publishAndSwap` で old world が `R5` へ eviction |
| `R4 -> R5` | **Retire**（old world eviction） | `R4`、new world が publish された | `R5`、old world の token=1, world=1（Retire chain） | PublishExecutor | old world の token は Retire chain へ移動 | `DeferredDeletionQueue` に投入 |
| `R5 -> R6` | **Quarantine**（epoch drain 失敗） | `R5`、epoch 未満 | `R6`、token=1, world=1（Q/E） | Retire drain（Non-RT） | Token は Quarantine へ移動 | `RetireQuarantineStore` に退避 |
| `R6 -> R7` | **Terminal**（Quarantine drain） | `R6`、epoch 到達 or shutdown | `R7`、token=1, world=1（Terminal） | Retire drain | Token は Terminal へ移動 | `TerminalReclaimAuthority` に移送 |
| `R7 -> R8` | **Release**（`deleter(world)`） | `R7`、reclaim 可能 | `R8`、token=0, world=0 | `TerminalReclaimAuthority`（World destruction authority）→`BudgetAuthority`（future release authority） | Token を Budget Authority へ返却（`B_world -=1`）。現行 code では `deleter` は World destruction のみであり `release(T)` は未実装 | `world` 破棄 |

Failure transitions：

| Transition | Event | Postcondition | Authority |
| --- | --- | --- | --- |
| `R1 -> R8` | **Build failure rollback** | token release（`B_world -=1`）、world なし | Builder |
| `R2 -> R8` | **Transfer failure: OwnerChannel::enqueue failure** | ownership は移譲されていない → local rollback（`deleter` -> token release） | Producer（Non-RT） |
| `R3 -> R8` | **Transfer failure: Intent enqueue failure**（OwnerChannel success 後） | ownership は既に OwnerChannel に移譲済み → `ownerChannel().take(key)` で回収 -> `deleter` -> token release | Producer（Non-RT） |
| `R3 -> R8` | **S2 shutdown rollback**（`drainAllNonRt`） | OwnerChannel resident を回収 -> retire/reclaim -> token release | Shutdown Authority |
| `R3 -> R8` | **Stale discard**（future generation / sequence / admission rejection、commit 前） | new world を commit 前に discard -> token release（`publishAndSwap` 前のため old world は `R4` に留まる） | Admission/PublishExecutor |

---

## 5. Step 2-D — Reservation conservation の検証

### 5.1 INV-RSV-1 — acquire/release conservation

```text
reservation_count(t) = successful_reservation_acquire - successful_reservation_release

```text
| 経路 | acquire | release | 期待される reservation | 現行コードでの状態 |
| --- | --- | --- | --- | --- |
| `Acquire 1 -> Build failure -> release 1` | 1 | 1 | 0 | **MISSING** — acquire も release も存在しない |
| `Acquire 1 -> Build -> Publish -> Retire -> Quarantine -> Terminal -> Release` | 1 | 1 | 0 | **MISSING** — acquire も release も存在しない |

判定: MISSING** — reservation token が存在しないため、conservation を検証する対象自体が存在しない。将来の Budget Authority で `acquire` / `release` を atomic に実装し、`B_world(t) = acq - rel` が常に成立することを証明する必要がある。

### 5.2 INV-RSV-2 — double release 禁止

```text
∀ token T: exactlyOneRelease(T)

```text
| 潜在的 double-release 経路 | 現行コードでの実体 | 二重 release の可能性 | 判定 |
| --- | --- | --- | --- |
| Build failure + shutdown release | Build failure 時に rollback release、同時に shutdown が `drainAllNonRt` で同 world を回収 | **将来の token では二重 release のリスクあり**。Build failure 時の local owner と OwnerChannel の二重回収を排他する必要がある | **DESIGN OBLIGATION** — 将来の token で `release(T)` の exactly-once を保証する機構が必要。現行は token がないため該当なし |
| OwnerChannel enqueue failure + Intent enqueue failure | `OwnerChannel::enqueue` 成功後に Intent enqueue が失敗した場合、OwnerChannel の owner を `take()` で回収 | 将来の token では `take()` による回収と Intent failure rollback の二重 release を排他する必要がある | **DESIGN OBLIGATION** |
| Publish rejection (monotonicity violation) + old world retire | `RuntimeWorldAuthority::publish` が `Faulted` で `committed=false` を返した場合、new world の owner は消費される（`return nullptr`）。caller が `committed=false` を確認せずに old world を retire すると二重の危険 | 現行コードでは `committed=false` 時に `return nullptr` で owner を消費し、old world は返さない（`publishAndSwap` しない）ため、物理的には二重にならないが、reservation token の観点では new world の token release が exactly-once であることを保証する必要がある | **DESIGN OBLIGATION** |
| Stale discard + shutdown release | future generation の stale publish で new world を discard した後、shutdown が同 world を再度回収 | 将来の token では stale discard 時の release と shutdown drain の排他が必要 | **DESIGN OBLIGATION** |
| `drainAllNonRt` + normal `take` + `publish` | shutdown 時に `drainAllNonRt` が OwnerChannel 全体を drain するが、同時に ISR が `take(key)` で同 world を取得 | 現行コードでは `drainAllNonRt` は shutdown 時の Non-RT 専用であり、ISR との並行は shutdown ordering（quiescence 後に drain）で排他される。将来の token でも同様の ordering が必要 | **DESIGN OBLIGATION** |
| Retire failure + Quarantine overflow + Terminal reclaim | Retire chain の各段階で reclaim が失敗した場合 | 将来の token では各段階の `deleter` が exactly-once であることを保証する必要がある | **DESIGN OBLIGATION** |

判定: **MISSING — 現行 Lifetime Budget token が存在しないため** / DESIGN OBLIGATION（将来の token で exactly-once release を証明すべき）。

> **補足**: World ownership 側には既に強い single-transfer evidence が存在する（`OwnerChannel: enqueue -> take -> empty` / `drainAllNonRt` の再 drain no-op — `OwnerChannel.h:47,89,116`）。`World ownership の double-transfer/double-drain が未証明` という意味ではない。正しくは **World ownership の single-transfer evidence は存在するが、それを Lifetime Budget token の exactly-once release に接続する evidence が存在しない**。**

### 5.3 INV-RSV-3 — leak 禁止

```text
token acquired ∧ world destroyed ⇒ token released

```text
| 潜在的 leak 経路 | 現行コードでの owner 回収 | reservation leak の可能性 | 判定 |
| --- | --- | --- | --- |
| `OwnerChannel::drainAllNonRt()` | shutdown 時に `drainAllNonRt(callback)` で全 resident owner を callback 経由で回収し、`retire/reclaim` へ移送する | 現行の owner 回収は正しく行われるが、**reservation token の release は存在しない**ため、将来の token では `drainAllNonRt -> retire/reclaim -> token release` の chain を保証する必要がある | **DESIGN OBLIGATION** |
| Registry の `unregister()` のみ | `PendingPublishRegistry::unregister(seqId)` は `seqId/world` を `0/nullptr` にするだけであり、ownership を解放しない | Registry の `unregister` だけでは owner/lifetime の完了にならない（Step 0/Step 1 の整理と一致）。将来の token では Registry cleanup と token release を混同してはならない | **DESIGN OBLIGATION** |
| OwnerChannel resident が `take` されずに放置 | shutdown 以外で OwnerChannel に owner が残留し、`take` も `drainAllNonRt` も呼ばれない | 現行コードでも shutdown 前に `drainAllNonRt` が必ず呼ばれる（`AudioEngine` shutdown ordering）。将来の token でも同様の guarantee が必要 | **DESIGN OBLIGATION** |

判定: MISSING（現行は token がないため該当なし） / DESIGN OBLIGATION**

最重要: `drainAllNonRt` は OwnerChannel の owning transport を全回収するが、`PendingPublishRegistry::unregister()` は non-owning metadata の cleanup に過ぎない。両者を混同してはならない（Step 1 3.4 と一致）。

> **注意 — drain chain の proof boundary**: 現行コードで証明できるのは `OwnerChannel -> drainAllNonRt -> retire/reclaim authority chain -> World destruction` までである。**Budget release はそこには存在しない**。`TerminalReclaimAuthority` は World destruction authority であり、`BudgetAuthority` は将来の release authority として **完全に別 authority** として維持する。将来、`BudgetAuthority::release(T)` を exactly-once 保証する cross-domain contract を別途定義する（Step 7/8）。
>
> ```text
> 現行: drainAllNonRt -> retire/reclaim -> World destruction (TerminalReclaimAuthority)
> 将来: World destruction --(cross-domain contract)--> exactlyOneRelease(T) (BudgetAuthority)
>        — TerminalReclaimAuthority ≠ BudgetAuthority、現時点では接続されていない
> ```

```text
OwnerChannel resident --drainAllNonRt--> owner recovered --retire/reclaim--> token released  ✓（将来）
Registry entry     --unregister-------> metadata cleared                     ✗ token release ではない

```text
---

## 6. Step 2-E — Transfer invariant の検証

### 6.1 INV-RSV-4 — transfer does not change reservation cardinality

```text
before transfer: reservation = 1
after transfer:  reservation = 1

```text
| Transfer | Before | After | Reservation 変化 | 判定 |
| --- | --- | --- | --- | --- |
| `local owner -> OwnerChannel::enqueue` | token=1, OwnerCount=1 (local) | token=1, OwnerCount=1 (OwnerChannel) | **0 変化** — ownership location の変更のみ | **DESIGN-DEFINED** — token が存在すれば成立すべき invariant。現行は owner の single-transfer（`std::move`）により physical に保証されるが、reservation の観点では token がないため検証不能 |
| `OwnerChannel -> take(key) -> local owner` | token=1, OwnerCount=1 (OwnerChannel) | token=1, OwnerCount=1 (local) | **0 変化** — `take()` は location 変更であり cardinality 変化ではない（Step 1 6.2） | **DESIGN-DEFINED** |

判定: DESIGN-DEFINED / CODE-EVIDENCE MISSING** — transfer が reservation 数を変更しないことは将来の token の invariant として定義されるべきだが、現行コードには token がないため code evidence は存在しない。OwnerChannel と PendingPublishRegistry が S2 gap に同時存在し得るが、前者は owning representation、後者は non-owning metadata であること（Step 1 6.2）は現行コードで確認済み。

---

## 7. Step 2-F — Publish / Retire で token がどう存続するか

### 7.1 S2 -> S3 -> S4 -> S5 -> S6 の token 存続

```text
S2 (OwnerChannel)
  --take(key) + publish(std::move(owner), metadata)--> S3 (RuntimeStore::current)
                                                      + old world -> S4 (Retire)
  S4 --epoch drain--> S5 (Q/E) --drain--> S6 (Terminal) --deleter--> S7 (Released)

```text
| 遷移 | Token の動き | World の動き | 判定 |
| --- | --- | --- | --- |
| `S2 -> S3`（new world） | token は new world と共に RuntimeStore へ移動。`B_world` は変化しない | `OwnerCount(newW)` : OwnerChannel -> RuntimeStore | **DESIGN-DEFINED** |
| `S3 -> S4`（old world eviction） | **old world の token** が Retire chain へ移動。new world の token と old world の token は別物 | `publishAndSwap` が `oldWorld` を返す。old world は `DeferredDeletionQueue` に投入 | **DESIGN-DEFINED** — new/old の混同禁止 |

### 7.2 最重要: new world と old world の reservation を混同しないこと

```text
publishAndSwap(next = newWorld):
    oldWorld = exchangeAtomic(current, next)  — oldWorld を caller へ返す
    newWorld: S2 -> S3（token 存続、RuntimeStore へ）
    oldWorld: S3 -> S4（token 存続、Retire chain へ）

1 world の lifecycle に対して reservation = 1 が継続する。
new world の token と old world の token は別個の token である。

publish 前:
    newW: R3 (token=1, S2)
    oldW: R4 (token=1, S3) — RuntimeStore::current に存在

publish 後:
    newW: R4 (token=1, S3) — RuntimeStore::current に昇格
    oldW: R5 (token=1, S4) — Retire chain へ eviction

各 world の token は独立して R1..R8 を辿る。
publish は new world の S2->S3 と old world の S3->S4 を同時に起こすが、
各 world の reservation は 1 のままである。

```text
判定: DESIGN-DEFINED / CODE-EVIDENCE MISSING** — 現行コードで `publishAndSwap` の old world eviction は確認できるが、reservation token の存続は将来の設計で保証すべき invariant である。

---

## 8. Step 2-G — Release point を一つに絞る（single authority / multiple paths）

### 8.1 理想的な release 構造

```text
                    ┌─ build failure (R1->R8) ──────────┐
                    ├─ transfer failure (R2->R8) ───────┤
S0 -> [R1 Reserved]-┼─ shutdown drain (R3->R8) ─────────┼─> exactly-once Release(T)
                    ├─ stale/admission reject (R3->R8) ┤         │
                    └─ normal reclaim (R7->R8) ─────────┘         ▼
                                                      Budget Authority
                                         (single Release operation authority)

  Note: terminalization paths are multiple; Release operation authority is singular.
        Each path must not directly do B_world-=1 — only Budget Authority does.
```text
### 8.2 Release authority の確定

| Release 経路 | Authority（将来） | Token release の条件 | 現行コードでの対応 |
| --- | --- | --- | --- |
| Build failure | Builder | `R1->R8`、token release（`B_world -=1`） | `RuntimeBuilder::buildWorld` 失敗時に local owner を破棄するが、token release はない |
| Transfer failure | Producer | `R2->R8` / `R3->R8`、owner 回収 -> deleter -> token release | `OwnerChannel::enqueue` false 時に owner を caller が保持するが、token release はない |
| S2 shutdown | Shutdown Authority | `R3->R8`、`drainAllNonRt -> retire/reclaim -> token release` | `drainAllNonRt` で owner 回収 -> reclaim するが、token release はない |
| Normal reclaim | Reclaim Authority | `R7->R8`、`deleter(world)` 時に token release（`B_world -=1`） | `TerminalReclaimAuthority` の `std::vector` からの reclaim 時に world を破棄するが、token release はない |
| Stale discard (R3->R8) | Admission/PublishExecutor | commit 前の new world discard 時に token release（`publishAndSwap` 前） | `publish` の `committed=false` 時に owner を消費するが、token release はない |

> **設計境界**: 現在のコードで `deleter(world)` / `drainAllNonRt() -> enqueueDeferredDeleteNonRtWithResult -> shutdownReclaim/terminalReclaim` は World destruction / ownership reclaim chain であり **Budget release ではない**。`World destruction ≠ Budget release` を明示する。将来 `World destruction → Release(T)` を invariant にするなら、それは新たな設計契約になる（将来の Budget Authority が `deleter` と `Release(T)` を結び付ける）。

**全経路で token release の semantic authority が複数存在しないこと** — つまり、どの経路でも `release(T)` が exactly-once であることを一箇所の authority（将来の Reclaim Authority / Budget Authority）が保証する必要がある。

判定: DESIGN OBLIGATION** — 現行コードには reservation token も release authority も存在しない。将来の Budget Authority で single release point を設計し、全 failure 経路で exactly-once を証明する必要がある。

---

## 9. Step 2-H — Step 1 の I-W5 を再判定

### 9.1 I-W5 の必要条件

```text
M_world(t) <= B_world(t)

必要条件:
  ∀ W ∈ {S2..S6}
  ∃ exactly one reservation token T
  such that ownsBudget(T) ∧ corresponds(T, W)

```text
### 9.2 現行コードでの判定

| 条件 | 現行コードでの状態 | 判定 |
| --- | --- | --- |
| `B_world(t)` が定義されているか | **いいえ** — reservation token が存在しないため `B_world` は semantic model のみ | **MISSING** |
| `corresponds(T, W)` が追えるか | **いいえ** — token が存在しないため対応関係を追えない | **MISSING** |
| `M_world(t)` が定義されているか | **はい** — `M_world(t)={W:lifecycle∈S2..S6}` は Step 1 で定義済み | **SATISFIED**（定義として） |
| `M_world <= B_world` が証明可能か | **いいえ** — `B_world` が semantic model のみのため containment を code evidence で証明できない | **PARTIAL** |

### 9.3 再判定

```text
I-W5: M_world(t) <= B_world(t)

Step 1 判定: FORMALIZED（定義として M_world <= B_world を導出）
Step 2 再判定: PARTIAL

理由:
  Step 1 では S1 で token=1,world=0 / S2..S6 で token=1,world=1 という
  semantic model から M_world <= B_world を形式的に導出した。
  しかし Step 2 で現行 production code に reservation token が存在しないことが確定したため、
  この導出は semantic model 上の containment であり、code evidence による証明ではない。

  したがって I-W5 は:

    定義としては SATISFIED（semantic model 上で M_world <= B_world は成立）
    code evidence としては MISSING（B_world の実体がないため証明不能）
    総合判定として PARTIAL

  に再判定する。

  Step 1 の verdict を無理に維持する必要はない（指示どおり）。

```text
| Invariant | Step 1 判定 | Step 2 再判定 | 理由 |
| --- | --- | --- | --- |
| I-W5 `M_world <= B_world` | FORMALIZED | **PARTIAL** | `B_world` が semantic model のみ。code evidence による containment 証明は不能 |

---

## 10. Step 2-I — O7 の扱い

### 10.1 現段階のモデル

```text
S1: token = 1, world = 0
S2..S6: token = 1, world = 1

```text
### 10.2 現在の production code の実体か、将来 Budget Authority の設計モデルか

| 状態 | Semantic model | 現行 production code の実体 | 分類 |
| --- | --- | --- | --- |
| `S1` token=1, world=0 | Budget reservation が存在し World は未生成 | **存在しない** — `S1` に相当する reservation 機構はコード上にない。`BuilderToken` は `S2` の build 権限であり `S1` の reservation ではない | **DESIGN ONLY** |
| `S2..S6` token=1, world=1 | 各 lifecycle で token と world が 1:1 対応 | **存在しない** — token がないため対応関係を code evidence で確認できない。World ownership（`OwnerCount=1`）は存在するが token との対応は semantic model のみ | **DESIGN ONLY** |

### 10.3 O7 の分類

```text
O7 = SEMANTIC MODEL ONLY

Reservation correspondence:
    DESIGN-DEFINED
    CODE-EVIDENCE: MISSING

```text
`token = 1` を実装済み事実として書いてはならない（指示どおり）。現行コードに reservation token が存在しない以上、`S1: token=1` / `S2..S6: token=1,world=1` は将来の Budget Authority の設計モデルとしてのみ扱う。

| 項目 | 判定 |
| --- | --- |
| O7 Token-World correspondence | **SEMANTIC MODEL ONLY / DESIGN-DEFINED / CODE-EVIDENCE MISSING** |
| `B_world = S1+S2+S3+S4+S5+S6` の加算 | semantic model 上の定義としてのみ有効。code 上の quantity の加算ではない |
| `M_world = S2+S3+S4+S5+S6` の加算 | `M_world(t)={W:lifecycle∈S2..S6}` として定義済み（Step 1）。distinct identity による計数であり、container occupancy の加算ではない（O4/O6 を継承） |

---

## 11. Step 2 の最終成果物 — 6表

### ① Reservation Object Table

| Semantic object | Type（将来） | Authority（将来） | Create | Transfer | Release | Thread |
| --- | --- | --- | --- | --- | --- | --- |
| Reservation token | `ReservationToken`（将来の型、Budget Authority が発行） | Budget Authority | `R0->R1` Acquire（`B_world < K_world` check） | `R1->R2->R3` Build + OwnerChannel enqueue | `R7->R8` deleter / `R1->R8` rollback / `R3->R8` shutdown | Budget Authority（acquire/release）、Builder/Producer/PublishExecutor（transfer/publish） |

現行コードでの実体: **なし** — 全て将来の Budget Authority で定義すべき。

### ② Reservation State Machine

```text
R0 NoReservation
  --Acquire(Budget Authority, B_world < K_world)--> R1 Reserved
    --Build(createForBuilder)---------------------> R2 WorldCreated (local owner)
      --Transfer(OwnerChannel::enqueue)-----------> R3 Transferred (OwnerChannel)
        --Publish(take + publishAndSwap)---------> R4 Published (RuntimeStore)
          --Retire(old world eviction)-----------> R5 Retiring (DeferredDeletionQueue)
            --Quarantine(epoch not reached)------> R6 Quarantined (Q/E)
              --Terminal(drain)-----------------> R7 Terminal (TerminalReclaimAuthority — World destruction authority)
                --Release(deleter)--------------> R8 Released (BudgetAuthority — future release authority)

Failure branches (all terminalize to single Release authority — see §8):
  R1 --Build failure-----------------------> R8 (release)
  R2 --Transfer failure-------------------> R8 (take + deleter + release)
  R3 --Shutdown drain--------------------> R8 (drainAllNonRt + deleter + release)
  R3 --Stale/admission/sequence reject---> R8 (new-world discard + release, before publishAndSwap; old world stays R4)

```text
各 transition の authority / token ownership / world existence は 4.1 節の表を参照。

> **Note — stale discard の位置**: `R4(new)->R8` ではなく `R3->R8` とする。`publishAndSwap` が実行されて old world を eviction するのは commit 成功側のみであり、stale/admission/sequence rejection された new world は `S3 Published` に到達する前に commit 前で discard されるため（§4.1 参照）。

### ③ Reservation Event Table

| Event | Semantic | 現行コードでの対応する操作 | Token 変化 | 判定 |
| --- | --- | --- | --- | --- |
| **Acquire** | `R0->R1` reservation acquire | **なし** — 将来の `BudgetAuthority::acquire()` | `B_world +=1` | **MISSING** |
| **Rollback (build)** | `R1->R8` build failure | `RuntimeBuilder::buildWorld` 失敗時に local owner 破棄（token なし） | `B_world -=1`（将来） | **MISSING** |
| **Transfer** | `R2->R3` OwnerChannel transfer | `OwnerChannel::enqueue(key, std::move(world))`（token なし、ownership transfer のみ） | `B_world` 不変 | **DESIGN-DEFINED** |
| **Publish** | `R3->R4` new world publish + old world retire | `take(key) + publish(std::move(owner), metadata) -> oldWorld` | `B_world` 不変（new/old 各自の token 存続） | **DESIGN-DEFINED** |
| **Retire** | `R4->R5` old world eviction | `publishAndSwap` の `oldWorld` を `DeferredDeletionQueue` に投入 | `B_world` 不変 | **DESIGN-DEFINED** |
| **Quarantine** | `R5->R6` epoch 未達で Q/E に退避 | `RetireQuarantineStore` に退避 | `B_world` 不変 | **DESIGN-DEFINED** |
| **Terminal** | `R6->R7` Quarantine drain で Terminal へ | `TerminalReclaimAuthority` に移送（growable vector） | `B_world` 不変 | **DESIGN-DEFINED** |
| **Release** | `R7->R8` reclaim で token release | `deleter(world)` で world 破棄（token release なし） | `B_world -=1`（将来） | **MISSING** |

### ④ Token/World Correspondence Table（Token 列は設計値 — 実装状態ではない）

| State | Token | World | Ownership | Registry | 備考 |
| --- | --- | --- | --- | --- | --- |
| `R0 S0` NoReservation | 0 | 0 | 0 (`None`) | 0 | token も world もなし |
| `R1 S1` Reserved | **1** | **0** | 0 (`None`) | 0 | **将来の semantic model のみ**。現行コードに実体なし |
| `R2` WorldCreated | **1** | **1** | 1 (`LocalOwner`) | 0 | Build 直後。現行は `BuilderToken` で world 生成するが token なし |
| `R3 S2` Transferred | **1** | **1** | 1 (`OwnerChannel`) | 0/1 | OwnerChannel が owning、Registry は non-owning metadata（同時存在可） |
| `R4 S3` Published | **1** | **1** | 1 (`RuntimeStore`) | 0 | `RuntimeStore::current` に published |
| `R5 S4` Retiring | **1** | **1** | 1 (`RetireChain`) | 0 | `DeferredDeletionQueue` |
| `R6 S5` Quarantined | **1** | **1** | 1 (`Quarantine`) | 0 | `Q/E` |
| `R7 S6` Terminal | **1** | **1** | 1 (`Terminal`) | 0 | `TerminalReclaimAuthority` |
| `R8 S7` Released | **0** | **0** | 0 (`None`) | 0 | reclaim 済み |

> **Note**: 全行の Token 列は設計値（将来の semantic model）であり現行 production code の観測値ではない（Step 2-I O7 分類を遵守 — CODE-EVIDENCE MISSING）。

### ⑤ Reservation Invariants

| Invariant | 定義 | 判定 |
| --- | --- | --- |
| **INV-RSV-1** | `reservation_count(t) = acq - rel` — acquire/release conservation | **MISSING** — token がないため検証不能。将来の Budget Authority で atomic に証明すべき |
| **INV-RSV-2** | `∀ T: release(T) ≤ 1` — no double release | **MISSING / DESIGN OBLIGATION** — 全 failure 経路で exactly-once を将来証明すべき |
| **INV-RSV-3** | `acquired ∧ destroyed ⇒ released` — no leak | **MISSING / DESIGN OBLIGATION** — `drainAllNonRt` chain で token release を将来保証すべき |
| **INV-RSV-4** | Transfer preserves reservation cardinality | **DESIGN-DEFINED / CODE-EVIDENCE MISSING** — semantic model では定義済み、code evidence なし |
| **INV-RSV-5** | `1 token ↔ at most 1 RuntimeWorld` | **DESIGN-DEFINED / CODE-EVIDENCE MISSING** — `S1` で token=1,world=0 の対応は semantic model のみ |
| **INV-RSV-6** | `S2->S3/S4/S5/S6` preserves token | **DESIGN-DEFINED / CODE-EVIDENCE MISSING** — publish/retire/quarantine/terminal で token 存続は将来の invariant |
| **INV-RSV-7** | `∀ T: acquired(T) ∧ terminalized(T) ⇒ exactlyOneRelease(T)` — deleter は World destruction を担い、BudgetAuthority が `Release(T)`の exactly-once authority を持つ cross-domain invariant | **MISSING / DESIGN OBLIGATION** — `deleter(world)` は World destruction、`release(T)` は Budget 返却。`World destruction ⇒ Reservation release` は単なる実装上の紐付けではなく、**`TerminalReclaimAuthority` が terminalization authority を持ち、`BudgetAuthority::release(T)` を exactly-once にコールする** authority chain を cross-domain invariant として定義する必要がある。`World destroyed / token leaked` を semantic model 上排除するための契約 |
| **INV-RSV-8** | `Reservation does not imply ownership` — `R1: T exists, W does not` | **DESIGN-DEFINED** — `S1 = token 1 / world 0` として既に表現済み。`Reservation ≠ Ownership ≠ World identity ≠ BuilderToken` を invariant として明文化し Step 3 の `A_max` derivation を安全にする |

### ⑥ Evidence Matrix

| Invariant | 判定 | Evidence / 根拠 |
| --- | --- | --- |
| INV-RSV-1 conservation | **MISSING** | reservation token 型も acquire/release API も `src/` に 0件。`B_world` は semantic model のみ |
| INV-RSV-2 no double release | **MISSING — 現行 Lifetime Budget token が存在しないため** | World ownership の single-transfer evidence は存在するが Budget token の exactly-once release に接続する evidence は存在しない（5.2 節の6経路） |
| INV-RSV-3 no leak | **MISSING — 現行 Lifetime Budget token が存在しないため** | World ownership の drain/reclaim evidence は存在するが Budget token の `acquired ∧ destroyed ⇒ released` に接続する evidence は存在しない（5.3 節） |
| INV-RSV-4 transfer preserves | **DESIGN-DEFINED** | semantic model では定義済み。code evidence は `OwnerChannel::enqueue` の single-transfer のみ（token なし） |
| INV-RSV-5 token↔world | **DESIGN-DEFINED** | semantic model では `S1:1-0 / S2..S6:1-1` を定義。code evidence は `worldId` の `S2` 以降の存在のみ（Step 1 I-W7） |
| INV-RSV-6 S2..S6 preserves | **DESIGN-DEFINED** | semantic model では定義済み。code evidence は `publishAndSwap` の new/old 分離のみ（token なし） |
| INV-RSV-7 destruction⇒release | **MISSING — cross-domain invariant として定義すべき** | `deleter(world)` は World destruction、`release(T)` は Budget 返却。`World destruction ⇒ Reservation release` を `TerminalReclaimAuthority` が terminalization を担い、`BudgetAuthority::release(T)` が exactly-once を保証する authority chain として定義する必要がある |
| INV-RSV-8 reservation≠ownership | **DESIGN-DEFINED** | `R1: T exists, W does not` — `Reservation ≠ Ownership ≠ World identity ≠ BuilderToken` を明文化（Step 3 の `A_max` 安全化） |

---

### 11.5 Step 3 に引き継ぐ quantity 分離（最重要境界）

Step 3 で `A_max` を導出する際、以下を別 quantity として固定し混同しないこと。

| Quantity | 意味 | 現行コード |
| --- | --- | --- |
| `A` | `S0→S1` Lifetime Budget reservation acquire 数 | **存在しない** |
| `B_world` | live reservation token 数 | **semantic only** |
| `M_world` | `S2..S6` World 数 | semantic definition（distinct identity） |
| `P` | publication event 数 | 現行コードに存在 |
| `OwnerChannelResident` | OwnerChannel resident owner 数（max 256） | 現行コードに存在 |
| `PendingIntentResidency` | `pendingIntentCount_`（Intent transport residency） | 現行コードに存在 |
| `publicationIntentResidencyCount_` | Publish Intent residency 数 | 現行コードに存在 |
| `RegistryResident` | Registry metadata resident 数（max 64） | 現行コードに存在 |
| `worldId` | World identity | 現行コードに存在 |

```text
A_max ≠ OwnerChannel capacity ≠ Intent queue capacity ≠ Registry capacity ≠ worldId generation count ≠ publication count
```

この区別を維持すれば Step 3 の最重要境界を守れる。

---

## 12. Step 2 の停止条件チェック

以下を全部満たすまで Step 3 に進まない（指示どおり）。

| 条件 | 判定 | 備考 |
| --- | --- | --- |
| [x] Lifetime Budget reservation の実体 / 非実体をコードで確定 | **完了** | **非実体** — `src/` に Lifetime Budget reservation token 型/API は 0件。`B_world` は semantic model のみ（2.3 節）。`pendingIntentCount_` / `publicationIntentResidencyCount_` / `reservationOwned` は Intent/Recovery domain の reservation であり Lifetime Budget ではない。I4 D48-D53 design contract（`WorldRetirementReservation`）は CLOSED だが Phase I implementation は NO-GO（2.5 節） |
| [x] Acquire event を二層で確定 | **完了** | **現行コード: Acquire event は存在しない**（`src/` に `B_world`/`K_world`/acquire API 0件、2.3節）。**Semantic design: `R0->R1` Acquire**（将来の `BudgetAuthority::acquire()`、4.1節）。`commitRuntimePublication` は `S1->S2` の transfer であり `S0->S1` ではない（2.6 節） |
| [x] Release event を二層で確定 | **完了** | **現行コード: Budget release は存在しない**（`deleter(world)` は world destruction のみ）。**Semantic design: 複数の terminalization paths（build/transfer/shutdown/stale/normal reclaim）→ 単一の `Release(T)` operation authority = Budget Authority**（8.1-8.2節）。各経路は `B_world-=1` を直接行ってはならない |
| [x] Build failure rollback を確認 | **完了** | 現行は local owner を破棄するのみ。将来の `R1->R8` release として設計すべき（5.2 節） |
| [x] S2 shutdown/drain rollback を確認 | **完了** | `drainAllNonRt -> retire/reclaim` は owner 回収するが token release ではない。将来の `R3->R8` として設計すべき（5.3 節） |
| [x] double-release 経路がないことを確認 | **完了** | 現行は token がないため該当なし。将来の 6経路で排他を証明すべきことを明示（5.2 節） |
| [x] leak 経路がないことを確認 | **完了** | 現行は token がないため該当なし。`drainAllNonRt` chain で token release を保証すべきことを明示（5.3 節） |
| [x] OwnerChannel transfer が reservation 数を変更しないことを確認 | **完了** | **DESIGN-DEFINED** — semantic model では INV-RSV-4 として定義。`OwnerChannel::enqueue` は ownership location 変更のみ（6.1 節） |
| [x] publish/retire/quarantine/terminal が reservation を保持することを確認 | **完了** | **DESIGN-DEFINED** — `S2->S3/S4/S5/S6` で token 存続は semantic model で定義。new/old の混同禁止を明示（7.2 節） |
| [x] Token ↔ World correspondence の proof boundary を確定 | **完了** | **DESIGN-DEFINED / CODE-EVIDENCE MISSING** — 証明したのは対応関係を semantic model として定義したこと。`S1:1-0 / S2..S6:1-1` は設計値であり code evidence による correspondence 証明は不能（10.3 節 O7） |
| [x] I-W5 を SATISFIED / PARTIAL のいずれかに確定 | **完了** | **PARTIAL** に再判定 — semantic model 上では `M_world <= B_world` は成立するが、`B_world` が code evidence なしのため containment 証明は不能（9.3 節） |
| [x] O7 を CODE-PROVEN / DESIGN-ONLY / MISSING に分類 | **完了** | **SEMANTIC MODEL ONLY / DESIGN-DEFINED / CODE-EVIDENCE MISSING**（10.3 節） |

**この時点でも `K_world` の数値は出さない — 遵守した。**

---

## 13. 次の監査順序

```text
D101-8 Step 1  FORMALIZED_WITH_OPEN_PROOFS
      │
      ▼
D101-8 Step 2  RESERVATION_SEMANTICS_DESIGN_REQUIRED
      │         CODE-EVIDENCE: MISSING
      │         I-W5: PARTIAL / O7: SEMANTIC MODEL ONLY
      │
      ▼
D101-8 Step 3  A_max derivation
      │         S0->S1 の reservation 許可数を数える設計
      │         Step 2 で固定した R0->R1 Acquire event が前提
      │         A_max は publication count ではなく admission/reservation event 数
      │
      ▼
D101-8 Step 4  P_max derivation
      │
      ▼
... (Step 5 H_max/G_contract -> Step 6 K_world -> Step 7 Conservation -> Step 8 Failure/shutdown)

```text
Step 2 は Step 3 の前提となる「reservation event が何なのか」を二層（現行コードの非実体 / 将来の semantic design / I4 D48-D53 の design contract）で固定する工程として完了した。**現行コードに Lifetime Budget reservation token は存在しない** — `WorldRetirementReservation` の design contract は I4_DESIGN_CONTRACT.md D48-D53 で CLOSED されているが、**Phase I implementation は NO-GO（コード未変更）** である。Step 3 以降は将来の Budget Authority の設計として `A_max / P_max / K_world` を derivation する必要がある（`A_max` は `commitRuntimePublication` 数 / `OwnerChannel` 256 / `PendingPublishRegistry` 64 / `pendingIntentCount_` / `publicationIntentResidencyCount_` / `MpscBoundedRing` slot / `worldId` 発行数 / `RuntimeStore::current` publish 数 / I4 D48-D53 design contract の `WorldRetirementReservation` のいずれからも直接導出してはならない — Step 2 の最重要境界。これらは別 semantic quantity であり、`commitRuntimePublication` は `registerPublish → OwnerChannel::enqueue → Intent enqueue` という ownership transfer / publication pipeline admission に過ぎず Lifetime Budget の `R0→R1 Acquire` ではない）。

---

## 14. Verdict

### 判定: `RESERVATION_SEMANTICS_DESIGN_REQUIRED — CODE-EVIDENCE MISSING`

| 判定 | 定義 | 該当性 |
| --- | --- | --- |
| `RESERVATION_SEMANTICS_PROVEN` | reservation token の acquire/transfer/release 全経路を code evidence で証明 | **該当せず** — token 自体が存在しない |
| `RESERVATION_SEMANTICS_DESIGN_REQUIRED` | Lifetime Budget reservation の semantic model は定義できた（including I4 D48-D53 design contract）が、production code evidence は MISSING。将来の Budget Authority で implementation すべき | **◯ 該当** |
| `MISSING` | 定義自体が未確定 | **該当せず** — 6表 / 8 invariants / state machine を定義できた |

### なぜ `DESIGN_REQUIRED` か

- **Lifetime Budget reservation の非実体を確定した** — `src/` に `B_world` / `K_world` / `LifetimeBudgetReservation` / `acquireReservation` / `releaseReservation` の型/API は 0件。`pendingIntentCount_` / `publicationIntentResidencyCount_` / `PendingRecoveryAdmission::reservationOwned` は Intent/Recovery domain の reservation/residency semantics を持つが、`B_world` とは別物であることを evidence 付きで証明した。
- **I4_DESIGN_CONTRACT.md D48-D53 と照合した** — `WorldRetirementReservation` の design contract は CLOSED（D48-D53）だが **Phase I implementation = NO-GO / コード未変更**（D53/D69）。design contract ≠ implementation。Step 2 の `CODE-EVIDENCE MISSING` は design contract が不存在であるのではなく、**production implementation が NO-GO** であることを明確にした（§2.5）。
- **BuilderToken / reservation token / world identity を分離した** — 3者を同一視しないことをコード証拠で確定した。
- **Reservation state machine R0..R8 を形式化した** — 各 transition の event / precondition / postcondition / authority / token ownership / world existence を埋めた。ただしこれは監査上の semantic state machine であり、コードの enum 新設ではない。
- **8 invariants（INV-RSV-1..8）を判定した** — INV-RSV-1..8 のうち、conservation / double-release / leak / destruction⇒release は MISSING、transfer / token↔world / S2..S6 preserves は DESIGN-DEFINED / CODE-EVIDENCE MISSING。
- **I-W5 を PARTIAL に再判定した** — semantic model 上では `M_world <= B_world` は成立するが、`B_world` が code evidence なしのため containment 証明は不能。Step 1 の verdict を無理に維持しなかった。
- **O7 を SEMANTIC MODEL ONLY に分類した** — `S1:token=1,world=0 / S2..S6:token=1,world=1` は将来の設計モデルとしてのみ扱い、`token=1` を実装済み事実として書かなかった。
- **停止条件 12項目を全て満たした** — acquire/release の非実体、rollback、double-release、leak、transfer、publish-retire persistence、I-W5/O7 の分類を全て確定した。

### 残存する open proof obligations（Step 3 以降で扱う）

- `BudgetAuthority::acquire()` / `release()` の atomic 設計と `B_world(t)=acq-rel` の証明（INV-RSV-1）
- 全 failure 経路での `release(T) ≤ 1` の排他証明（INV-RSV-2）
- `drainAllNonRt` chain での `acquired ∧ destroyed ⇒ released` の保証（INV-RSV-3）
- `K_world` の数値導出（Step 6）
- `I-W5` の code evidence による `M_world <= B_world` の最終証明（Step 7 Conservation）

---

## 付録: D101-8 Step 2 監査チェックリスト

- [x] Step 2 の冒頭で `B_world` の実体である reservation をコード上の event と対応付けた
- [x] `S0->S1` の reservation/admission event とその release/rollback 全経路をコードから列挙した
- [x] `B_world = S1〜S6 の lifecycle 数` を `B_world = 現実に存在する reservation token 数` へ落とせるかを検証した（答え: 現行は落とせない — semantic model のみ）
- [x] `BuilderToken` / `reservation token` / `world identity` を明示的に分離した
- [x] `BuilderToken == reservation token` 等の同一視をコード証拠なしに行わなかった
- [x] Reservation state machine R0..R8 を形式化し、各 transition の 6要素を埋めた
- [x] INV-RSV-1..7 を evidence 付きで判定した（MISSING / DESIGN-DEFINED / DESIGN OBLIGATION）
- [x] double-release / leak の全経路を調査した
- [x] `I-W5` を code evidence に基づき `PARTIAL` に再判定した（Step 1 の verdict を無理に維持しなかった）
- [x] `O7` を `SEMANTIC MODEL ONLY / DESIGN-DEFINED / CODE-EVIDENCE MISSING` に分類した
- [x] 6表（Reservation Object / State Machine / Event / Token-World Correspondence / Invariants / Evidence Matrix）を作成した
- [x] 停止条件 12項目を全て満たした
- [x] `K_world` の数値導出を行わなかった
- [x] Production code 変更なし（semantic proof boundary の確定のみ）
