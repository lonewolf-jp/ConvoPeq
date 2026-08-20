# D101-8 Step 6 — K_world Derivation

> **Status**: Step 3 `A_max` 未実装（design-defined）/ Step 4 `P_queue_max=4096 PROVEN` `P_max<=4098 CONDITIONAL` / Step 5 `H_hold<∞ liveness` `G_contract NOT PROVEN` `E_max_audio<=1 topology-dependent` `E_max_message 未確定` を入力として `K_world<∞` を production code から閉じる試み。**コード変更なし**。

## Step 4/5 — Frozen Inputs

```text
Step 3: A_max < ∞              design-defined / code MISSING（I4 D48-D53 Phase I 未実装）
Step 4: P_queue_max = 4096     PROVEN（MpscBoundedRing kQueueSize）
        P_max <= 4098          CONDITIONAL（R-PROD-1..4 current-code topology）
Step 5: H_hold < ∞             liveness成立（全21 reader経路 RAII）/ 数値boundなし
        G_contract < ∞         NOT PROVEN（samplerは観測のみ・watchdogなし）
        E_max_audio <= 1       production topology依存（1 callback = 1 hold, publishes_per_callback<=1）
        E_max_message          未確定 — 本Stepで再監査
Step 6: K_world < ∞            本証明対象
```

`G_contract` は Step 5 結論どおり NOT PROVEN のまま残す。K_world の主題は「観測できるか」ではなく「production state machineが同時に何個のRuntimeWorldを保持できるか」である。

---

## 6-A — K_world Counting Semantics（凍結定義）

### Budget Unit

```text
1 budget unit = 1 RuntimeWorld = 1 RuntimeState（aligned_unique_ptr<const RuntimeState>）
```

`RuntimeWorld` と `RuntimeState` は本コードでは同一実体（`RuntimeWorldAuthority::RuntimeOwner = aligned_unique_ptr<const RuntimeState>`）。`RuntimePublishWorld` は `RuntimeState` の build-time variant ではなく同一型の別名として扱う。

### State Machine（S0..S7）

```text
S0 Available   — free capacity（world非存在・budget未消費）
S1 Reserved    — reservation acquired（Lifetime Budget reservation — design contract）
S2 Transferred — OwnerChannel/PendingPublishRegistry に ownership 移転済み
S3 Published   — RuntimeStore::current として公開中（reader可視）
S4 Retiring    — DeferredDeletionQueue に retire enqueue 済み（epoch保護下）
S5 Quarantined — RetireQuarantineStore / EmergencyQuarantineStore に退避
S6 Terminal    — TerminalReclaimAuthority に移送（growable pending or synchronous reclaim）
S7 Released    — deleter実行済み・budget返却（非counting）
```

### Counting Rule

```text
outstanding(t) = |S1(t)| + |S2(t)| + |S3(t)| + |S4(t)| + |S5(t)| + |S6(t)|
S7(t) は counting対象外
S0(t) は free capacity（budget - outstanding）
```

**不変条件**: 同一 `RuntimeWorld` は任意の時刻で高々1つのS状態にのみ属する。遷移は `S1->S2->S3->S4->S5->S6->S7` の順序で進行し、逆行・分岐・複製はしない（rollbackは当該worldをS7へ直接移行して破棄）。

### Intent vs World の非同一視

```text
P = publication intent residency（MpscBoundedRing占有）
K = RuntimeWorld lifetime residency（S1..S6占有）
```

`P` と `K` は異なるbudgetである。`1 intent -> 0 or 1 world` の変換は別途証明が必要であり、P_maxをK_worldへ直接加算してはならない（6-Eで検証）。

---

## 6-B — 各S状態の Production Owner 完全列挙

|State|Owner / Container|Creation|Transfer|Release|備考|
|---|---|---|---|---|---|
|S1 Reserved|（design）`WorldRetirementReservation` — `src/`未実装|admission reservation acquire|reservation -> world ownership transfer|rollback or publish commit|Step 3結論どおり code MISSING。D101-9 assumptionとして扱う|
|S2 Transferred|`OwnerChannel<aligned_unique_ptr<const RuntimeState>>`（RuntimeWorldAuthority所有） + `PendingPublishRegistry`（k=64）|`RuntimeBuilder::buildRuntimePublishWorld()` が `aligned_make_unique<RuntimeState>` で生成|`commitRuntimePublication` -> `ownerChannel().store(owner)` -> `registry.registerPublish(seqId, sealedWorld)`|`PendingPublishRegistry::unregister(seqId)` after `publishAndSwap` success / failure時 `owner` 破棄|OwnerChannelは単一take、key isolation、pending registryはlock-free|
|S3 Published|`RuntimeStore<RuntimeState, RuntimeWorldAuthority>::current`（atomic<RuntimeState*>）|`RuntimeWorldAuthority::publish(owner, metadata)` 内 `writeAccess_.publishAndSwap(next)` で公開|旧currentが `oldWorld` として callerへ返却（retire対象）|次のpublishで追い出される or shutdown clear|capacity=1（単一currentポインタ）。`K_current<=1`|
|S4 Retiring|`DeferredDeletionQueue`（Vyukov bounded MPMC kQueueSize=4096）|`ISRRetireRouter::enqueueRetire` / `enqueueWithRetry` が oldWorldを `currentEpoch()` 付きでenqueue|reclaimは `tryReclaim()` で `isOlder(entry.epoch, minReaderEpoch)` が真の先頭エントリのみ回収|reclaim成功でdeleter実行|S4はFIFO先頭ブロック性あり — 先頭がunsafeなら後続も待機|
|S5 Quarantined|`RetireQuarantineStore m_retireQuarantine`（kMax=512, mutex） + `m_emergencyQuarantine`（同型 512）|`enqueueWithRetry` Stage 3/4: D full retry後 Qへ `quarantine()` / Eへ `emergencyQuarantine()`|drainは `tryReclaim` 直後に `drainQuarantineStore()` / `drainEmergencyAndTerminal()` でepoch-gated回収|epoch safeでdeleter実行、unsafeは保持|Q/Eはfixed capacity・allocation-free（array+size）。overflowは `overflowCount_` 記録、deleter実行禁止（UAF排除）|
|S6 Terminal|`TerminalReclaimAuthority`（std::vector growable, mutex, Non-RT only）|`enqueueWithRetry` Stage 5: D+Q+E全滿で `terminalReclaim()` へ移送|drainは `drainTerminalReclaim()`（epoch-gated） + `drainAll()`（shutdown時無条件）|epoch safeなら即時deleter（Non-RT synchronous）、unsafeならpending保持|growableのため `store()` は常にtrue（no store-full failure path）。K_terminalはheap growthに依存|
|S7 Released|（none — heap free）|—|—|deleter実行（`AlignedObjectDeleter` -> `aligned_free`）|counting対象外|

### 補足: Shutdown専用経路

`ShutdownReclaimAuthority`（drainAll / shutdown path）はS6の変種として扱う。`AudioEngine.CtorDtor.cpp` / `ReleaseResources.cpp` の destructor / `finalizeShutdown` で `retireCurrentAndTarget()` + `tryReclaim()` + `drainAllQuarantineStore()` + `drainAll()` が実行される。shutdownの有限完了性はStep 8責務のため本Stepではconservationの遷移存在のみ確認する。

---

## 6-C — B_world と M_world の分離

設計契約（phase-d101-6 lifetime-budget-authority-design）で明示された分離を本証明で採用する。

```text
B_world(t) = logical budget / capacity bound（設計上の上限）
M_world(t) = 実際にproduction code上で存在するworld数（outstanding実測）
```

```text
M_world(t) <= B_world(t) <= K_world < ∞
```

- `B_world(t)` は各S状態のcapacity上限の和として定義される（`K_reserved+K_transferred+K_current+K_retire+K_quarantine+K_terminal+K_reader`）。
- `M_world(t)` は同時刻に outstanding（S1..S6）に存在するworldの実数。
- `K_world` は `B_world(t)` の時間非依存なarchitectural upper bound（`max_t B_world(t)`）。

本分離により、telemetryのobserved maximum（`M_observed`）とarchitectural bound（`K_world`）を混同しない。

---

## 6-D — K_world 候補上限の構成（二重計上排除）

### Symbolic Candidate

```text
K_world <= K_reserved + K_transferred + K_current + K_retire + K_quarantine + K_terminal + K_reader
```

各項は outstanding world の互いに排他な同時滞留上限を表す。

### 二重計上排除の原則

S遷移 `S2->S3->S4->S5->S6->S7` は同一worldの状態遷移であり、同時刻に複数状態として数えてはならない。したがって上式は「各段階のcontainerが同時に保持しうるworld数の最大値の和」として解釈し、同一worldが複数項に跨って数えられることはない。

正しくは

```text
K_world <= max simultaneous occupancy across S1..S6
         <= Σ (capacity of container for Si)
```

として整理する。Σは worst-case で全containerが同時に満杯になる同時滞留の上限であり、二重計上ではない。同一worldがS2とS3に同時に存在することは構造的に不可能（publishAndSwapでownershipが移動する）。

### 各項の意味

|項|対象|Container|
|---|---|---|
|K_reserved|S1|Lifetime Budget reservation（design）|
|K_transferred|S2|OwnerChannel + PendingPublishRegistry|
|K_current|S3|RuntimeStore::current|
|K_retire|S4|DeferredDeletionQueue|
|K_quarantine|S5|RetireQuarantineStore + EmergencyQuarantineStore|
|K_terminal|S6|TerminalReclaimAuthority|
|K_reader|S3/S4のstranding|reader epoch保護による未回収world数（S4/S5内のうち minReaderEpoch < entry.epoch の部分）|

注意: `K_reader` は独立したcontainerではなく、S4/S5内に滞留するworldのうち readerにより回収がblockされている部分の追加boundである。したがって `K_retire+K_quarantine` と `K_reader` の関係は「capacity + stranding」であり、strandingがcapacityを超過することはない（reclaimはFIFO先頭blockのため、readerが1つでも古いepochを保持すると全後続がblockされる）。本証明では保守的に加算するが、tight boundでは `K_reader <= K_retire+K_quarantine` が成り立つ。

---

## 6-E — P_max を K_world に安易に足さない

### 非加算性の根拠

```text
P = publication intent residency（MpscBoundedRing占有、intent単位）
K = RuntimeWorld lifetime residency（S1..S6占有、world単位）
```

両者は異なる生成元・異なるlifecycle・異なるcontainerを持つ。したがって `K_world <= ... + 4098` と直結させることは誤りである。

### 変換関係の検証

```text
1 intent -> 0 or 1 world ?
```

production code上で検証する。

- Producer（Non-RT）: `RuntimeBuilder::buildRuntimePublishWorld()` が `RuntimePublishSpecification` から `aligned_make_unique<const RuntimeState>` でworldを生成する。生成失敗（BuildError）は `nullptr` を返し intentは破棄（rollback）される。
- Enqueue: `commitRuntimePublication` は `ownerChannel().store(owner)` で OwnerChannelへ移送し、同時に `registry.registerPublish(seqId, sealedWorld)` で pending登録する。
- Consumer（CoordinatorLoop / RebuildThread）: `ISRRuntimePublicationCoordinator::processIntent` -> `RuntimePublishExecutor::executePublish` が `ownerChannel.take(key)` で単一takeし、`authority.publish(owner, metadata)` で `publishAndSwap` する。
- 1 intentの処理は高々1回の `publishAndSwap` を試みる。commit失敗（monotonicity violation等）なら `nullptr` を返し worldは破棄される。

したがって production invariant として

```text
1 intentの正常処理 -> 高々1 worldがS3へ到達（publish成功）
1 intentの失敗処理 -> 0 world（owner破棄、S2で消滅）
```

が成り立つ。ただし「intentがworldを生成する」関係は `BuildResult` の分岐に依存し、intent residency（P）と world生成（K）は 1:1 ではない。

### 結論

```text
K_world <= ... + P_max   は直接成立しない。
K_worldへの寄与は、P経由ではなくS2（OwnerChannel/PendingRegistry）のcapacity（64）で評価するのが正しい。
```

P_max（4098）は intent queueのboundであり、world lifetime budgetの直接項ではない。Pが大きいことは「多数のintentが滞留しうる」ことを意味するが、それらintentが全てworldへ変換される保証はなく、また変換されても PendingPublishRegistry（64）と RuntimeStore（1）の段階で直列化される。

**本証明では `P_max` を `K_world` の加算項として使用しない。** 代わりに `K_transferred <= 64`（PendingPublishRegistry capacity）を world側のboundとして採用する。

---

## 6-F — Reader Stranding K_reader の導出

Step 5 の `H_max` をここで消費する。ただし `K_reader = H_max` とはしない。

### 定義

```text
K_reader = maximum number of worlds that can remain unreclaimable
           while one or more readers retain an older epoch
```

これは時間ではなくworld数のboundである。

### 分解式

```text
K_reader = reader_count × epoch_advancement_during_hold × worlds_retired_per_epoch
```

各因子をproduction codeから導出する。

#### reader_count

`EpochDomain::kMaxReaders = 64` が登録上限だが、実際に reader slot を確保しているのは固定メンバのみ。

|Reader Domain|Fixed Members|Count|
|---|---|---|
|AudioEngine EpochDomain|audioThreadRcuReader + messageThreadRcuReader|2|
|ConvolverProcessor EpochDomain|runtimeRcuReader等|1|
|EQProcessor EpochDomain|rcuReader|1|

productionで同時にactiveになりうるreader数は `activeReaderCount()` で観測可能だが、architectural boundは `K_reader` の算出では「同時に古いepochを保持するreader数」として評価する。最悪でも `reader_count <= 2`（AudioEngine domain）が支配的。Convolver/EQ domainは別EpochDomainのため AudioEngineのworld reclaimをblockしない。

**本証明では `reader_count <= 2` を採用する**（AudioEngine domainの2固定reader）。

#### worlds_retired_per_epoch

1回の `publishEpoch()` -> `publishAndSwap` で旧current 1個が retire対象になる。したがって `worlds_retired_per_epoch <= 1`（正常 publish時）。

例外: `switchImmediate` / `finalizeShutdown` で `target` も retireされる場合は 1 epochで2 worldが retireされうるが、これはshutdown/例外pathであり steady-stateでは1が上限。

**本証明では `worlds_retired_per_epoch <= 1`（steady-state）を採用する。**

#### epoch_advancement_during_hold

これが核心であり、Step 5で未確定だった `E_max` に相当する。

```text
E_max = max epoch gap a reader can span = max (currentEpoch - readerEpoch) while reader active
```

`K_reader = Σ_over_readers E_max(reader) × 1` として導出する。

詳細は 6-G で再監査する。

---

## 6-G — Message Thread E_max 再監査（最優先）

Step 5 の記述「Message Thread readerはfunction scopeだが、RebuildThreadがその間に何publishできるかは不明」をそのまま持ち込み、production topologyから導出する。

### 調査対象

|対象|役割|publishEpochとの関係|
|---|---|---|
|RuntimePublishExecutor::publishEpoch|—|存在しない（executorはpublishEpochを直接呼ばない）|
|ISRRetireRouter::publishEpoch|publishEpochの唯一の公開API|SnapshotCoordinator / RuntimeWorldAuthority経由で呼ばれる|
|EpochDomain::publishEpoch|epoch前進の実体|`fetchAdd(globalEpoch,1)` + `fetchAdd(epochGeneration,1)`|
|RebuildThread|非同期 world buildを担うワーカ（存在すれば）|intent enqueueのproducer|
|messageThreadRcuReader|Message Threadの固定RCUReader|timerCallback等のreader hold|
|timerCallback|Message Thread上の100ms周期コールバック|reader holdの主体|
|runCoordinatorPhase / CoordinatorLoop|dedicated juce::Thread（1ms fallback）|intent dispatch + publish実行の主体|
|makeRuntimeReadHandle|reader guard生成|enter/exitの境界|

### epoch前進の呼び出し元列挙

`publishEpoch()` を呼ぶ production path（grep結果より）:

|#|File|Line|Caller|Context|
|---|---|---|---|
|1|AudioEngine.CtorDtor.cpp|194|AudioEngine destructor|`m_retireRouter->publishEpoch()`（shutdown）|
|2|AudioEngine.CtorDtor.cpp|215|AudioEngine destructor|同上（2回目、isolation）|
|3|AudioEngine.Processing.ReleaseResources.cpp|237|releaseResources|`m_retireRouter->publishEpoch()`（shutdown）|
|4|AudioEngine.Processing.ReleaseResources.cpp|255|releaseResources|同上（2回目）|
|5|AudioEngine.Publication.cpp|18|markRetireEpoch / advanceRetireEpoch|public API wrapper -> `m_retireRouter->publishEpoch()`|
|6|core/SnapshotCoordinator.cpp|91|SnapshotCoordinator::publishNew|`m_epochProvider->publishEpoch()`（publishNew時）|
|7|core/SnapshotCoordinator.cpp|109|SnapshotCoordinator::switchImmediate|`m_epochProvider->publishEpoch()`（switch時）|
|8|core/SnapshotCoordinator.h|105|SnapshotCoordinator::switchImmediate（inline）|同上|
|9|core/SnapshotCoordinator.h|173|SnapshotCoordinator::retireCurrentAndTarget|`m_epochProvider->publishEpoch()`（shutdown）|
|10|eqprocessor/EQProcessor.Core.cpp|75|EQProcessor::retireEQStateDeferred|自domainの `m_epochDomain.publishEpoch()`（EQ専用domain）|

**重要**: 上記のうち steady-state publishで実際に呼ばれるのは **#6 SnapshotCoordinator::publishNew のみ** である。#6は `ISRRuntimePublicationCoordinator::processIntent` -> `RuntimePublishExecutor::executePublish` -> `RuntimeWorldAuthority::publish` -> `SnapshotCoordinator::publishNew` の chainで CoordinatorLoop thread上でのみ実行される。

EQProcessor (#10) は別EpochDomainのため AudioEngine worldの epochを前進させない。

Shutdown path (#1-4, #9) は S7移行のため K_world steady-stateとは無関係。

### Reader Hold 中に何回 publishEpoch() が発生可能か

```text
reader enters at epoch N  (Message Thread: timerCallback内で makeRuntimeReadHandle)
        ↓
how many successful publishEpoch() on CoordinatorLoop before reader exits?
        ↓
before reader exits (function scope end)?
```

#### Audio Threadの場合

`audioThreadRcuReader` は `getNextAudioBlock` の function scope（~10ms）で holdされる。CoordinatorLoopは別threadで並行実行されるが、1 callback中に CoordinatorLoopが処理できるpublish数は throughputに依存する。

production topology上、**1 audio callback中に CoordinatorLoopが publishEpochを呼ぶ回数は高々1** と推定される。根拠:

- `RuntimeBuilder::buildRuntimePublishWorld()` は数msのbuild時間を要する
- `PublicationAdmission::evaluate` で admission gateがあり、連続publishはthrottleされる
- `PendingPublishRegistry` は64だが、steady-stateでは1 publish per build cycle

ただし厳密な `E_max_audio <=1` は「topology-dependent」であり、固定不変条件ではない。Step 5の結論どおり `E_max_audio <=1` は current-code topology factとして扱う。

**本証明では `E_max_audio <=1` を conditional factとして採用する。**

#### Message Threadの場合（核心）

`messageThreadRcuReader` は以下で holdされる:

|Call Site|Scope Duration|Publish Concurrency|
|---|---|---|
|timerCallback (AudioEngine.Timer.cpp:373)|100ms周期、関数内数ms|CoordinatorLoopと並行|
|prepareToPlay|初期化時のみ、数ms|同上|
|releaseResources|shutdown時のみ|同上|
|Snapshot processing|数ms|同上|

Message Thread reader hold中に CoordinatorLoopが `publishEpoch()` を何回呼べるかは、**CoordinatorLoopのintent処理throughput × hold duration** で決まる。

CoordinatorLoopは `runCoordinatorPhase` を dedicated threadで実行し、MpscBoundedRing（4096）から intentを dequeueして `processIntent` する。各intentの処理は `RuntimeBuilder` + `PublicationAdmission` + `publishAndSwap` を含み、1 intentあたり数ms〜数十msを要する。

したがって hold durationが数msなら `E_max_message <=1..2` 程度、holdが100msなら `E_max_message <= 数回` 程度と推定されるが、**production code上に固定上限を保証する定数は存在しない**。

固定上限の候補として検討したもの:

|候補|値|bounded?|判定|
|---|---|---|---|
|kMaxReaders|64|fixed|reader数であり epoch advancementではない|
|kPendingPublishCapacity|64|fixed|PendingPublishRegistry capacityだが、CoordinatorLoopは逐次処理するため同時刻のepoch advancement数ではない|
|kQueueSize (MpscBoundedRing)|4096|fixed|intent queue capacityだが、hold中に全て処理される保証はない|
|MpscBoundedRing drain rate|—|unbounded|throughputはbuild時間に依存、固定boundなし|
|CoordinatorLoop 1ms fallback|—|—|polling intervalであり publish rateではない|

**結論**:

```text
E_max_message <= C  (fixed C)  は current production codeから導出できない。
```

`H_hold_message < ∞`（function scopeによるliveness）は成立するが、`H_hold_message <= K`（固定Kでbounded）は不成立であり、かつ `publish rate` も固定boundを持たないため、`E_max_message` は unboundedとして扱わなければならない。

言い換えると:

```text
H_hold_message < ∞  (liveness)  ✅
H_hold_message <= K (bounded)   ❌ — host buffer / OS scheduling依存
publish rate <= R (bounded)     ❌ — build complexity依存
∴ E_max_message < ∞             ❌ cannot be proven from current production invariants
```

### 6-G の判定

```text
E_max_message = unbounded under current production invariants
```

これは Step 6 の `K_reader` に直接影響する。

---

## 6-H — Retire / Quarantine Capacity の実数確定

### Production Constants（最新ConvoPeq.md / src/ 照合）

|Container|Constant|Value|Type|Overflow Path|Fallback Path|
|---|---|---|---|---|---|
|DeferredDeletionQueue|kQueueSize|4096|fixed (Vyukov bounded MPMC, array<DeletionEntry,4096>)|enqueue false -> retry -> quarantine|RetireQuarantineStore|
|RetireQuarantineStore (Q)|kMaxQuarantinedEntries|512|fixed (array<QuarantinedEntry,512> + mutex)|quarantine false -> EmergencyQ|EmergencyQuarantineStore|
|EmergencyQuarantineStore (E)|kMaxQuarantinedEntries|512|fixed (同型)|quarantine false -> Terminal|TerminalReclaimAuthority|
|TerminalReclaimAuthority|—|growable (std::vector growable)|dynamic growth (heap)|store() always true (no failure path)|synchronous reclaim or pending|
|PendingPublishRegistry|kPendingPublishCapacity|64|fixed (array<Entry,64> + atomic cursor)|overwrite oldest (cursor % 64)|—|
|RuntimeStore::current|—|1|fixed (atomic<RuntimeState*> single pointer)|—|—|
|OwnerChannel|—|1 per channel|fixed (single Owner slot, SPSC)|store false -> caller retains|caller must retry or drop|

### M_retire / M_quarantine / K_terminal の分類

|Quantity|Value|Classification|Evidence|
|---|---|---|---|
|M_retire (DeferredDeletionQueue)|4096|fixed capacity|`DeferredDeletionQueue.h:262 kQueueSize=4096`|
|M_quarantine (Q)|512|fixed capacity|`RetireQuarantineStore.h:65 kMaxQuarantinedEntries=512`|
|M_quarantine (E)|512|fixed capacity|同上（別インスタンス）|
|M_quarantine total|1024|fixed (512+512)|Q+E合計|
|M_terminal (K_terminal)|growable|dynamic growth|`ISRRetireRouter.h TerminalReclaimAuthority: std::vector<Entry> entries_` + `store() ALWAYS true`|
|M_transferred (PendingPublishRegistry)|64|fixed capacity|`RuntimeWorldAuthority.h:34 kPendingPublishCapacity=64`|
|M_current|1|fixed|`RuntimeStore<RuntimeState, RuntimeWorldAuthority>` single current pointer|

### 以前候補との照合

以前の設計候補:

```text
4096   M_retire       -> ✅ 一致（kQueueSize=4096）
1024   M_quarantine   -> ✅ 一致（512+512=1024）
K      M_terminal     -> ⚠️ growable（固定Kなし・heap依存）
```

`K_terminal` については Step 0 で決めたとおり `K_terminal < ∞` を形式的パラメータとして仮定する。Terminalのbounded implementationは D101-9の責務であり、本Stepでは `K_terminal < ∞` を assumptionとして扱う。

### 補足: RetireQuarantineStore の設計意図

`RetireQuarantineStore.h:224` の `std::array<QuarantinedEntry, 512>` と `RetireQuarantineStore.h:65` の capacityは、通常100msオーダーで解放されるRT参照中オブジェクトの退避を想定した固定値である。過剰なbacklogは異常系（HealthEvent対象）として扱われる。capacity exhaustion時は `overflowCount_++` し `false` を返すが、呼出し元（Router）は `deleter` を実行せず health escalationする（UAF構造的排除）。

---

## 6-I — Release / Reclaim の単調性証明

### S6->S7->S0 Budget Return

```text
S6 (Terminal) -> S7 (Released) -> S0 (Available)
```

がbudgetを返すことの確認。

#### 不変条件

```text
world is counted exactly once while S1..S6
world is not counted after S7
reclaim success => outstanding world count decreases by 1
```

production code上の確認:

|Transition|Code Path|Counting Effect|
|---|---|---|
|S6->S7 (epoch safe)|`TerminalReclaimAuthority::drain(minReaderEpoch, isOlderFn)` が `isOlder(entry.epoch, minReaderEpoch)` で真の entryのみ `deleter` 実行|1 world released|
|S6->S7 (overflow safe)|`store()` 内で `isOlder(entry.epoch, minReaderEpoch)` が真なら即時 `deleter` 実行（synchronous destruction）|1 world released（store直後に解放）|
|S6->S7 (shutdown)|`drainAll()` が全entryを無条件で `deleter` 実行|全world released|
|S4->S7|`DeferredDeletionQueue::reclaim(minReaderEpoch)` が先頭から epoch safeな entryのみ deleter実行|1..N worlds released|
|S5->S7|`RetireQuarantineStore::drain(minReaderEpoch, isOlderFn)` が epoch safeな entryのみ deleter実行|1..N worlds released|

`worldReclaimCount` / `reclaimCount_` は `recordWorldReclaim()` で incrementされ、`WorldRetirementReferenceObserver::onRelease()` に通知される（telemetry）。

### Failure Path での World Leak 検証

|Failure Path|World Ownership|Leak?|Evidence|
|---|---|---|---|
|reserve failure (admission reject)|world未生成（BuildError）|No leak|Builderがnullptr返却、intent破棄|
|build failure (BuildError)|world生成失敗 or 部分生成|No leak|`aligned_unique_ptr` RAIIで自動解放|
|publish reject (commit Faulted)|world生成済みだが publishAndSwapせず|No leak|`RuntimeWorldAuthority::publish()` が `commit Faulted` で `nullptr` 返却、ownerは破棄|
|publish reject (monotonicity)|同上|No leak|同上|
|retire failure (queue full)|oldWorldがD enqueue失敗|No leak|Q -> E -> Terminalへ退避 chain（全て ownership transfer）|
|quarantine (Q full)|oldWorldがQ enqueue失敗|No leak|E -> Terminalへ退避|
|terminal (growable)|oldWorldがTerminal pending|No leak|growable storeが常にownership受領、drainで回収|
|shutdown (destructor)|全containerの残留world|No leak|`finalizeShutdown` -> `retireCurrentAndTarget` -> `drainAll` / `drainAllQuarantineStore`|
|stuck reader|reclaim block|No leak（but residency prolonged）|`detectStuckReaders` -> `quarantineReader` -> `getMinReaderEpoch` から除外 -> reclaim unblock|

全failure/shutdown pathで worldが宙に浮く（unowned）ことはない。ownership chain `D->Q->E->Terminal` は各段階で `store() ALWAYS true`（Terminal）または `false時に次段階へ移送` するため、ptrがunownedになる経路は存在しない。

ただし「有限時間で回収完了するか」（shutdownの有限完了性）は Step 8の責務であり、本Stepでは conservation（countingの正確性）のみを証明する。

### Conservation Invariant

```text
∀t: M_world(t) = |S1(t)| + |S2(t)| + |S3(t)| + |S4(t)| + |S5(t)| + |S6(t)|
∀ transition S_i -> S_j: |S_i| decreases by 1, |S_j| increases by 1（or S7で消滅）
∴ Σ|S1..S6| は publish成功で+1、reclaim成功で-1 の単調なbudget accounting
```

double counting / missing stateは存在しない。各worldは生成から解放まで高々1つのS状態にのみ属し、遷移は単一owner transferで完結する。

---

## 最終成果 — K_world 導出の試み

### Symbolic Bound（再掲）

```text
K_world <= K_reserved + K_transferred + K_current + K_retire + K_quarantine + K_terminal + K_reader
```

### 各項の Finiteness 判定

|項|Bound|値 / 根拠|判定|
|---|---|---|---|
|K_reserved|S1|Step 3 design-defined `A_max < ∞`を assumption|CONDITIONAL（code MISSINGだが design contractで有限）|
|K_transferred|S2|OwnerChannel(1) + PendingPublishRegistry(64) = 65|`<∞` PROVEN（fixed capacity）|
|K_current|S3|RuntimeStore::current = 1|`<=1` PROVEN|
|K_retire|S4|DeferredDeletionQueue kQueueSize=4096|`<∞` PROVEN（fixed）|
|K_quarantine|S5|RetireQuarantineStore 512 + Emergency 512 = 1024|`<∞` PROVEN（fixed）|
|K_terminal|S6|TerminalReclaimAuthority growable|`<∞` ASSUMPTION（D101-9で bounded impl、現行はheap growthで有限だが固定boundなし）|
|K_reader|stranding|`reader_count(2) × E_max × 1`|OPEN — E_max_message unbounded|

### K_reader の詳細

```text
K_reader = reader_count × E_max × worlds_per_epoch
         <= 2 × E_max_message × 1   （Message Threadが支配的）
```

6-Gの結論より `E_max_message` は current production invariantsから固定boundを導出できない。

- `E_max_audio <=1` は conditional fact（topology-dependent）として成立
- `E_max_message` は unbounded（publish throughputに固定上限なし）

したがって

```text
K_reader < ∞   は current production codeから証明できない
```

ただし `K_reader <= K_retire + K_quarantine`（strandingは retire/quarantine内のworld数を超えない）は成り立つため、保守的には `K_reader` を独立加算せず `K_retire+K_quarantine` に含める解釈も可能。この解釈では `K_reader` の unbounded性は `K_world` 全体の unbounded性には直結しない（reclaim block中も world数は capacity内に収まる）。

### K_world 全体の判定

#### Tight Bound（K_readerを独立加算しない解釈）

```text
K_world <= K_reserved + K_transferred + K_current + K_retire + K_quarantine + K_terminal
        <= A_max + 65 + 1 + 4096 + 1024 + K_terminal
        < ∞    under assumptions: A_max<∞, K_terminal<∞
```

この解釈では `K_world < ∞` は **CONDITIONAL GO**（`K_terminal<∞` と `A_max<∞` を explicit assumptionとして）で閉じる。

#### Conservative Bound（K_readerを独立加算する解釈）

```text
K_world <= ... + K_reader
K_reader = 2 × E_max_message × 1
E_max_message = unbounded
∴ K_world = unbounded  →  NO-GO
```

### 本Stepの最終判定

```text
GO:            K_readerを独立項としない tight boundでは CONDITIONAL GO
CONDITIONAL:   K_terminal<∞（D101-9 assumption）+ A_max<∞（design assumption）が明記されれば Step 7へ進める
NO-GO:         K_readerを独立加算する conservative boundでは E_max_message unboundedのため NO-GO
```

### 本証跡文書の判定: CONDITIONAL GO（tight bound解釈を採用）

理由:

- `K_reader` は S4/S5内のworldの回収遅延を表すものであり、独立したcontainerではない
- `K_reader <= K_retire + K_quarantine` が構造的に成り立つため、capacity内に包含される
- `G_contract NOT PROVEN` は K_worldの主題ではない（Step 5結論を維持）
- 残る assumptionは `K_terminal<∞` と `A_max<∞` の2つのみで、いずれも D101-9 / Phase I で bounded implementationが予定されている

### Step 7 への Assumption 引き継ぎ

|Assumption|内容|解消予定|
|---|---|---|
|A1: A_max<∞|Lifetime Budget reservationのacquire上限が有限|Phase I: WorldRetirementReservation実装|
|A2: K_terminal<∞|TerminalReclaimAuthorityのpending数が有限|D101-9: Terminal bounded implementation|
|A3: E_max_message bounded（optional）|Message Thread hold中のepoch前進回数が有限|将来: CoordinatorLoop throughput throttle or reader hold time boundの固定化|

A3は tight bound解釈では不要だが、conservative boundで K_readerを独立評価する場合に必要となる。

---

## Task 完了チェックリスト

|Task|内容|Status|
|---|---|---|
|Task 1|ConvoPeq.md最新定義から RuntimeWorld creation/ownership/release topology完全列挙|✅ 完了（6-B）|
|Task 2|S1〜S6の production owner/container表|✅ 完了（6-B）|
|Task 3|各stateの capacityを fixed/bounded-by-X/unbounded/unknown 4分類|✅ 完了（6-H）|
|Task 4|P_max<=4098とK_worldの直接加算関係検証|✅ 完了（6-E: 非加算、S2 capacity 64で評価）|
|Task 5|Message Thread hold中の publishEpoch()回数導出|✅ 完了（6-G: unbounded）|
|Task 6|K_reader symbolic導出|✅ 完了（6-F: reader_count×E_max×1）|
|Task 7|retire/quarantine/terminal capacityを production constantsと照合|✅ 完了（6-H）|
|Task 8|全S1->S7 transitionの conservation invariant|✅ 完了（6-I）|
|Task 9|M_world<=B_world<=K_world<∞ 証明|✅ 完了（CONDITIONAL GO — tight bound）|

---

## Chain Completion Table（Step 6反映）

|Step|Quantity|Bound|Status|
|---|---|---|---|
|Step 0|Contract reconciliation|—|DONE|
|Step 1|World identity / ownership formalization|—|DONE|
|Step 2|Reservation token semantics|—|DONE|
|Step 3|A_max < ∞|design-defined / code MISSING|CONDITIONAL|
|Step 4|P_queue_max = 4096|PROVEN||
|Step 4|P_max <= 4098|CONDITIONAL (R-PROD current-code)||
|Step 5|H_hold < ∞ (liveness)|PROVEN (RAII)||
|Step 5|H_max <= K (bounded)|NOT PROVEN (no fixed K)||
|Step 5|G_contract < ∞|NOT PROVEN||
|Step 5|E_max_audio <=1|CONDITIONAL (topology-dependent)||
|Step 5|E_max_message|UNBOUNDED||
|Step 6|K_reserved < ∞|CONDITIONAL (A_max assumption)||
|Step 6|K_transferred < ∞|PROVEN (65)||
|Step 6|K_current <=1|PROVEN||
|Step 6|K_retire < ∞|PROVEN (4096)||
|Step 6|K_quarantine < ∞|PROVEN (1024)||
|Step 6|K_terminal < ∞|ASSUMPTION (D101-9)||
|Step 6|K_reader < ∞|CONDITIONAL (tight bound: contained in K_retire+K_quarantine) / OPEN (conservative: E_max_message unbounded)||
|Step 6|K_world < ∞|CONDITIONAL GO (tight) / NO-GO (conservative)||

---

## References

- `src/audioengine/RuntimeWorldAuthority.h` — PendingPublishRegistry k=64, OwnerChannel, RuntimeStore::current
- `src/DeferredDeletionQueue.h:262` — kQueueSize=4096
- `src/audioengine/RetireQuarantineStore.h:65,224` — kMaxQuarantinedEntries=512
- `src/audioengine/ISRRetireRouter.h` — TerminalReclaimAuthority growable, enqueueWithRetry D->Q->E->Terminal chain
- `src/core/EpochDomain.h:22,190` — kMaxReaders=64, publishEpoch fetchAdd
- `src/core/SnapshotCoordinator.cpp:91` — publishNew publishEpoch call site
- `src/audioengine/RuntimeBuilder.h` — buildRuntimePublishWorld aligned_make_unique
- `evidence/phase-d101-8-step3-a-max-derivation.md` — A_max design-defined
- `evidence/phase-d101-8-step4-p-max-derivation.md` — P_max<=4098 CONDITIONAL
- `evidence/phase-d101-8-step5-hmax-gcontract-derivation.md` — H_hold liveness, G_contract NOT PROVEN, E_max analysis
- `evidence/phase-d101-6-lifetime-budget-authority-design.md` — B_world/M_world separation
- `evidence/phase-d101-7-lifetime-budget-state-machine-contract.md` — S0..S7 state machine

---

## Appendix — Production Topology Diagram

```text
Non-RT Producer                          CoordinatorLoop (dedicated thread)
─────────────                            ──────────────────────────────────
RuntimeBuilder                           ISRRuntimePublicationCoordinator
  │ buildRuntimePublishWorld               │ processIntent
  │ aligned_make_unique<RuntimeState>      │  ├─ PublicationAdmission::evaluate
  ▼                                      │  ├─ RuntimeBuilder (if needed)
OwnerChannel ──store(owner)──► PendingPublishRegistry
  │  (single slot)              (k=64, cursor%64)
  │                              │
  │              ┌───────────────┘
  │              ▼
  │         RuntimePublishExecutor::executePublish
  │              │ take(key) — sole consumption
  │              ▼
  │         RuntimeWorldAuthority::publish(owner, metadata)
  │              │ commit(PublishAuthority::Granted, ...)
  │              │ publishAndSwap(next) — single WriteAccess
  │              ▼
  │         RuntimeStore::current (K_current=1)
  │              │ oldWorld returned
  │              ▼
  │         ISRRetireRouter::enqueueWithRetry(oldWorld, epoch)
  │              │
  │    ┌─────────┼─────────┬──────────┐
  │    ▼         ▼         ▼          ▼
  │    D(4096)   Q(512)    E(512)     Terminal(growable)
  │    │         │         │          │
  │    └────┬────┴────┬────┴────┬─────┘
  │         │         │         │
  │    tryReclaim() → drain chain (epoch-gated: isOlder(entry.epoch, minReaderEpoch))
  │         │         │         │
  │         ▼         ▼         ▼
  │      deleter → S7 Released (budget returned)
  │
  ▼
MpscBoundedRing (intent queue, 4096) — P budget (intent residency, NOT K_world term)
```

```text
Reader Protection (EBR)
───────────────────────
AudioEngine EpochDomain (kMaxReaders=64, active=2)
  ├─ audioThreadRcuReader ──► ObservedRuntime(guard) in getNextAudioBlock (~10ms hold)
  └─ messageThreadRcuReader ──► ObservedRuntime(guard) in timerCallback (~ms hold)

getMinReaderEpoch() = min(active reader epochs) — gates reclaim
  └─ quarantined readers excluded (detectStuckReaders -> quarantineReader)
  └─ K_reader = worlds blocked by minReaderEpoch < entry.epoch
  └─ K_reader <= K_retire + K_quarantine (tight bound)
```
