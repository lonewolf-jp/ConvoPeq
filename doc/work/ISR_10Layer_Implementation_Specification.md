# ConvoPeq ISR 10層実装仕様（完成案）

## 位置づけ

本書は **safe runtime + evidence export / CI validation architecture** を実装するための「10層完成案の詳細仕様」である。

レビュアーフィードバック（2026-05-20）に基づき、specification-driven な現状から
**safe runtime first architecture** へ進化させるための
実装班向け具体的ガイドを提供する。

### REV3.1運用優先注記（参照設計 vs 実装運用）

- 本書の10層モデルは **参照設計（reference architecture）** として扱う。
- ConvoPeq の実装運用は `plan5.md` REV3.1 Few-Authority 方針を優先し、
    `PublicationWorld / ExecutionWorld` の 2-world 簡略プロファイルと
    capability-first authority の解釈を採用する。
- Phase 0 からの読み替えは `ISR_Minimal_Phase0_Recommended.md` の
    REV3.1 Operational Stability Readiness を優先ゲートとする。

### REV3.2実運用ハードニング注記（上書き規則）

- 本書内の `self-proving / runtime-resident verification / autonomous emit` 記述は
    **設計参照表現**として扱い、実装運用は `plan5.md` REV3.2 を優先する。
- 実運用は `runtime exposes evidence` / `CI validates evidence` を固定方針とする。
- Build別方針: Release=minimal barriers + sampled trace / Debug=partial trace + assert-trap / CI=full verify-simulation + strict schema validation
- stale DSP handle build方針: Release=quarantine + silence（必要時 fail-safe bypass）/ Debug=assert（必要時 trap）/ CI=abort

用語正規化（齟齬回避）:

- 本書では `RuntimePublication` を正規記法として扱う。

---

## 根本課題の認識

### 現状の不足（REV2 更新）

```text
runtime が invariant を"守る設計"をしているのではなく、
runtime が invariant を"保持・証明する実装"が不足している

REV2 追加: runtime object architecture は成立しているが、
execution authority architecture が未閉塞
= JUCE lifecycle / DSP ownership / RT execution / federated runtime の4系統
```

### 完成条件

```text
runtime が invariant を安全側で保持し、必要な evidence を export し、
CI が strict validation を担う状態

= 各層は build profile に応じて evidence を export し、
    verification は CI / Debug pipeline を中心に運用する
+ JUCE callback が ISR-safe event stream へ完全変換済み
+ DSP ownership source-of-truth が DSPHandleRuntime に統合済み
+ RT thread への authority leakage がゼロ
+ helper-level world/epoch 整理が few-authority 実装へ読み替え可能
```

---

## 10層の実装構造（+ Layer 0）

### 層0: JUCE Lifecycle Isolation Runtime（JUCE変換層）

**位置づけ**: ConvoPeq は JUCE host driven architecture であり、ISR runtime はその上に載る。
JUCE callback が ISR invariant を直接破壊できる状態を解消する。
詳細仕様: `doc/work/ISR_JUCE_Lifecycle_Isolation.md`

**必須interface**:

```cpp
class LifecycleIsolationRuntime
{
public:
    LifecycleToken enterPrepare(int sampleRate, int blockSize);

    void leavePrepare(LifecycleToken);

    LifecycleToken enterAudioCallback();

    void leaveAudioCallback(LifecycleToken);

    LifecycleToken enterRelease();

    void leaveRelease(LifecycleToken);

    LifecyclePhase current() const noexcept;
};
```

**LifecyclePhase**:

```cpp
enum class LifecyclePhase
{
    Uninitialized,
    Preparing,
    Prepared,
    AudioRunning,
    Releasing,
    Released,
    Shutdown
};
```

**必須 invariants**:

- `LIF-1`: prepareToPlay overlap 禁止
- `LIF-2`: releaseResources during audio callback 禁止
- `LIF-3`: runtime publish during Releasing 禁止
- `LIF-4`: crossfade start before Prepared 禁止
- `LIF-5`: callback 中 runtimeVersion 変化不可
- `LIF-6`: callback 中 DSP generation 変化不可

**必須 artifacts**:

- `lifecycle_phase_trace.json`: phase transitions + barrier timestamps

**Closed criteria**:

- [ ] LifecycleIsolationRuntime が prepareToPlay/releaseResources/audioCallback の入口に配置済み
- [ ] LIF-1 ～ LIF-6 の違反が runtime で abort される
- [ ] lifecycle_phase_trace.json が emit される
- [ ] JUCE callback から直接 publish graph を変更するパスが存在しない

---

### 層1: ClosureBuilder（publish-time helper）

**必須interface**:

```cpp
class ClosureBuilder
{
public:
    ClosureNodeId createNode(ClosureNodeDescriptor desc);

    void connect(
        ClosureNodeId from,
        ClosureNodeId to,
        ClosureEdgeKind kind);

    void seal(ClosureNodeId root);

    ClosureValidationResult validateAcyclic(ClosureNodeId root) const;

private:
    ClosureArena arena_;
    ClosureGraph graph_;
};
```

**RuntimePublication 最終形**:

```cpp
struct RuntimePublication
{
    ClosureNodeId root;
};
```

**node metadata**:

```cpp
struct ClosureNodeDescriptor
{
    PayloadTier tier;
    CapabilitySet capabilities;
    bool immutable;
    bool reclaimable;
    AuthorityId authority;
};
```

**必須invariants**:

| invariant | 内容 | enforcement |
| --- | --- | --- |
| CR-1 | publish graph は DAG（cycle禁止） | static analysis + runtime check |
| CR-2 | sealed root から mutable node reachable禁止 | runtime audit on publish |
| CR-3 | reclaimable node は authority必須 | compile-time + runtime |
| CR-4 | RTLocalOnly tier leak禁止 | compile-time contract |

**必須validators**:

```cpp
validateReachability();        // CR-1: DAG verification
validateMutability();          // CR-2: sealed mutable check
validateAuthority();           // CR-3: authority ownership
validateTierPropagation();     // CR-4: tier leak detection
validateReclaimability();      // CR-3: reclaim precondition
```

**mandatory artifact**:

- **closure_graph.json**: ClosureBuilder の graph export hook から自動emit
  - schema: node/edge/validation result含む
    - timing: publish時 + shutdown時（Debug/CI では on-demand dump を許可）
  - CI rule: missing = merge reject

**重要**: ClosureBuilder は validator subsystem ではなく **publish-time builder utility** として扱う。
introspection / governance / proof orchestration を Runtime Core 常駐責務へ含めない。

---

### 層2: Payload Contract System（compile-time enforcement）

**必須trait definition**:

```cpp
template<typename T>
struct PayloadTraits;
```

**trait example**:

```cpp
template<>
struct PayloadTraits<RuntimeGraphNode>
{
    static constexpr bool publishable = true;

    static constexpr PayloadTier tier =
        PayloadTier::InlineImmutable;

    static constexpr CapabilitySet capabilities =
        Capability::ImmutableRead |
        Capability::EpochTracked;
};
```

**publish API**:

```cpp
template<typename T>
requires(PayloadTraits<T>::publishable)
PublishToken publish(T&& payload);
```

**compile-time reject table**:

| 違反 | rejection |
| --- | --- |
| mutable payload | yes |
| forbidden tier | yes |
| RTLocal leak | yes |
| authority missing | yes |
| reclaim undefined | yes |

**capability propagation**:

Closure traversal で **effective capability** を合成。

- root capability は全reachable node に適用
- tier downgrade チェック（immutable tier のみ reachable）

**重要**: validator reliance を減らし **type-enforced ISR** へ移行。

---

### 層3: SealedRuntime（immutability enforcement）

**必須base class**:

```cpp
class SealedObject
{
public:
    void seal();

    bool sealed() const noexcept;

protected:
    void assertMutable() const;
};
```

**mandatory policy**:

- 全mutator: `assertMutable()` mandatory
- publish phase: `sealRecursively(root)` mandatory
- deep seal: Closure traversal による reachable graph全seal必須
- Release build: silent mutation ignore禁止（abort/quarantine mandatory）

**sealing semantics**:

```cpp
// publish phase
sealRecursively(root);  // reachable graph 全seal

// runtime
// どこかで mutation を試みると:
assertMutable();        // sealed→exception→crash/quarantine
```

**必須invariants**:

| invariant | 内容 |
| --- | --- |
| SR-1 | sealed object mutation = fatal violation |
| SR-2 | reachable mutable object under sealed root禁止 |

**mandatory artifacts**:

- **mutation_fault_trace.json**: mutation attempt時に自動emit
  - call stack, timestamp, object id等を含む
  - schema: violation record array
  - timing: each violation
  - CI rule: invalid schema = merge reject

---

### 層4: Executable HB Runtime（runtime proof graph / 3分割）

**runtime events**:

```cpp
enum class HBEventType
{
    Publish,
    Observe,
    Retire,
    Reclaim,
    ShutdownBarrier,
    EpochAdvance
};
```

**HB graph structure**:

```cpp
struct HBNode
{
    HBEventId id;
    HBEventType type;
    Timestamp ts;
};

struct HBEdge
{
    HBNodeId from;
    HBNodeId to;
    MemoryOrder order;  // seq_cst / acq_rel / release / acquire
};
```

**runtime verifier（REV2 3分割）**:

```cpp
class HBRuntimeCore
{
public:
    void publishBarrier(...);
    void observeBarrier(...);
    void retireBarrier(...);
};

class HBTraceRuntime
{
public:
    void emit(HBEventType type, ...);
    HBTrace snapshot() const;
};

class HBVerifierRuntime
{
public:
    VerifyResult verify(const HBTrace& trace) const;
};

// 互換注記: 旧記法の "HBRuntime" は上記3 runtime の総称として扱う
class HBRuntime
{
public:
    HBTraceRuntime& trace();
    HBVerifierRuntime& verifier();
    HBRuntimeCore& core();
};

```

**mandatory instrumentation（GI-7準拠）**:

- publish, observe, retire, reclaim, shutdown barrier, epoch settle
- RT path は bounded fixed-size telemetry のみ許可し、full trace は Debug/CI 側を主とする

**mandatory checks**:

| rule | 内容 | action |
| --- | --- | --- |
| HB-1 | reclaim before grace禁止 | reject |
| HB-2 | observe after reclaim禁止 | reject |
| HB-3 | stale publish visibility禁止 | reject |
| HB-4 | shutdown reorder禁止 | reject |

**mandatory artifacts**:

- **hb_graph_trace.json**: runtime trace から自動emit
- **hb_violation_report.json**: HB check failure時に自動emit
  - schema: violation + trace excerpt
  - CI rule: violation detected = merge reject

**重要**: HB を **runtime-generated proof object** へ昇格。

---

### 層5: Retire Runtime（lane-separated reclaim）

**lane separation**:

| lane | 用途 | authority |
| --- | --- | --- |
| RTIntentLane | RT emit only | RT thread |
| CoordinationLane | authority enqueue | dedicated coordinator |
| EpochLane | grace tracking | epoch manager |
| ReclaimLane | actual destroy | reclaim worker |
| QuarantineLane | delayed reclaim | quarantine manager |

**runtime interface**:

```cpp
class RetireRuntime
{
public:
    void emitIntent(...);       // RTIntentLane
    void enqueueRetire(...);    // CoordinationLane
    void settleEpoch(...);      // EpochLane
    void reclaim(...);          // ReclaimLane
    void quarantine(...);       // QuarantineLane
};
```

**mandatory invariants**:

| invariant | 内容 | enforcement |
| --- | --- | --- |
| RR-1 | RT thread delete/free/reclaim禁止 | static lint (Atomic Dot-Call) |
| RR-2 | reclaim requires settled epoch | runtime gate |
| RR-3 | all reclaim emits bounded trace（RT非摂動） | mandatory instrumentation |

**mandatory artifacts**:

- **retire_timeline.json**: retire events across 5 lanes
- **retire_quarantine.json**: quarantine entries + reason
  - schema: timeline array + quarantine record array
  - CI rule: RR-2 violation = merge reject

---

### 層6: Shutdown Runtime（barrier-backed FSM）

**state enum**:

```cpp
enum class ShutdownState
{
    Running,
    AudioStopped,
    ObserverDrained,
    RetireClosed,
    EpochSettled,
    ReclaimComplete,
    ShutdownComplete
};
```

**runtime interface**:

```cpp
class ShutdownRuntime
{
public:
    void transition(ShutdownState next);

    void barrier();

    ShutdownState current() const;
};
```

**mandatory HB chain**:

```text
AudioStopped
    HB ↓
ObserverDrained
    HB ↓
RetireClosed
    HB ↓
EpochSettled
    HB ↓
ReclaimComplete
    HB ↓
ShutdownComplete
```

**mandatory verification**:

```cpp
verifyShutdownTrace();  // HB chain + state transition integrity
```

**mandatory artifacts**:

- **shutdown_trace.json**: state transitions + HB events
- **shutdown_violation_report.json**: state machine violation detail
  - schema: transition record array + barrier failure detail
  - CI rule: state mismatch / HB missing = merge reject

---

### 層7: Evidence Export Hooks（artifact generation）

**architecture**:

| runtime | artifact | generator |
| --- | --- | --- |
| ClosureBuilder | closure_graph.json | ClosureBuilder graph export hook |
| HBRuntime | hb_graph_trace.json | HBRuntime::snapshot() |
| RetireRuntime | retire_timeline.json | RetireRuntime::snapshot() |
| ShutdownRuntime | shutdown_trace.json | ShutdownRuntime::snapshot() |
| SealedRuntime | mutation_fault_trace.json | SealedRuntime::dumpViolations() |

**CI role shrinkage**:

CI は：

```text
artifact evaluator only へ縮退

= CI が validator を実装するのではなく、
    runtime が export した artifact を evaluate するのみ
```

**mandatory invariants**:

| invariant | 内容 |
| --- | --- |
| PR-1 | evidence missing = CI merge reject（Releaseはoptional, Debugはrecommended） |
| PR-2 | evidence invalid = CI merge reject（CI only mandatory） |

**timing**:

- on publish（runtime phase）
- on shutdown（finalization）
- periodic snapshot は Debug/CI or on-demand のみ

---

### 層8: Budget / Trace Governance（complexity control）

**mandatory budgets**:

| budget | limit | measure |
| --- | --- | --- |
| closure traversal | O(N) bounded | node count / iteration depth |
| validation latency | < deadline | wall-time ms |
| retire latency | < epoch | grace period pct |
| RT instrumentation | zero alloc | static analysis pass |
| metadata growth | < memory quota | artifact size + resident size |
| artifact size | per-type limit | JSON size bytes |

**mandatory policies**:

| policy | 内容 |
| --- | --- |
| RB-1 | validator proliferation禁止（count frozen） |
| RB-2 | RT trace sampling mandatory（100%→probabilistic） |
| RB-3 | metadata compaction mandatory（periodic defrag） |

**mandatory artifact**:

- **runtime_budget_report.json**: resource usage vs limits
  - schema: budget record array + policy compliance status
  - CI rule: budget exceeded = merge reject

---

### 層9: Safe Failure Handling（failure containment）

**action enum**:

```cpp
enum class RecoveryAction
{
    RejectPublish,      // publish abort
    Rollback,           // state rollback
    Quarantine,         // delayed reclaim
    SafeMode,           // degraded continue
    Abort               // fatal shutdown
};
```

**mandatory mapping**:

| failure | action | reason |
| --- | --- | --- |
| closure invalid | RejectPublish | root invariant broken |
| HB violation | Quarantine | causality broken |
| retire timeout | DelayedReclaim | safety margin |
| shutdown violation | SafeMode | graceful degrade |
| seal violation | Abort | immutability broken |

**mandatory invariant**:

```text
RC-1: unsafe continuation 禁止

= failure detect → safe action を deterministic に選択
    （log only 継続は禁止）
```

**mandatory artifact**:

- **recovery_trace.json**: failure → action decision + outcome
  - schema: recovery event array
  - CI rule: unsafe continuation detected = merge reject

---

### 層10: Introspection（operational debugging）

**observable state**:

```cpp
class ISRIntrospectionConsole
{
public:
    RuntimeSnapshot snapshot() const;
    RetireLaneState retireLanes() const;
    ShutdownState shutdownFSM() const;
};
```

**export artifact**:

```text
runtime_snapshot.json

= minimal operational state dump（snapshot / retire / shutdown）
（Debug/CI または on-demand）
```

**schema**:

- current snapshot summary
- retire lane states（summary）
- shutdown FSM（current + recent transitions）

注記（REV3.2整合）:

- HB full graph / proof archive / schema linkage は Debug/CI layer 側の責務とする。

---

## 実装完成状態の定義

### Pre-完成（Current: specification-driven）

```text
✓ Specification formalized (P1-P8)
✓ Invariants named and documented
✗ few-authority 実装へ責務収束
✗ evidence export / CI validation 境界固定
✗ Release 最小責務 / Debug-CI 拡張の build 分離
```

### 完成（Target: safe runtime first）

```text
✓ Specification formalized (P1-P8)
✓ Invariants named and documented
✓ few-authority 実装へ責務収束（7 subsystem）
✓ Evidence export（CI mandatory / Release minimal）
✓ Verification は CI / Debug pipeline で運用
✓ Safe failure handling（mute / bypass / quarantine / abort）
✓ Budget governance（RT非攪乱）
✓ Introspection は Debug/CI or on-demand
```

---

## 層実装順序（ボトルネック優先）

**1-6層（P9 Runtime Object Model）**:

1. ClosureBuilder（publish-time helper）
2. Payload Contract System（type enforcement）
3. SealedRuntime（immutability）
4. Executable HB Runtime（concurrency proof）
5. Retire Runtime（lifecycle）
6. Shutdown Runtime（synchronization）

**7-10層（P10 Evidence & Safe Failure Handling）**:

1. Evidence Export Hooks（artifact generation）
2. Budget / Trace Governance（complexity control）
3. Safe Failure Handling（failure mapping）
4. Introspection（operational visibility）

---

## 層完成の判定基準（各層Closed requirement）

### ClosureBuilder Closed条件

- [ ] createNode/connect/seal/validateAcyclic implemented
- [ ] CR-1～CR-4 invariants verify in runtime
- [ ] closure_graph.json export on publish（CI mandatory / Release minimal）
- [ ] artifact schema validation passing
- [ ] static analysis (ownership/tier/authority) all green
- [ ] CI artifact validator integrated

### Payload Contract System Closed条件

- [ ] PayloadTraits template defined for all publish types
- [ ] compile-time publish() requires() all enforced
- [ ] tier-forbidden payload compile-time reject
- [ ] RTLocal leak detection at compile-time
- [ ] authority-missing payload compile-time reject
- [ ] Capability propagation runtime audit
- [ ] CI compile-error catch enabled

### SealedRuntime Closed条件

- [ ] SealedObject.seal() / sealed() / assertMutable() implemented
- [ ] sealRecursively() deep seal all reachable nodes
- [ ] SR-1/SR-2 invariants enforced at mutation attempt
- [ ] mutation_fault_trace.json export（build profile 準拠）
- [ ] Release build silent-ignore disabled (abort/quarantine)
- [ ] mutation violation test case passing

### Executable HB Runtime Closed条件

- [ ] HBRuntime emitEvent() all event types instrumented
- [ ] HB graph construction は Debug/CI layer で実行（Release runtime core は最小 barrier を維持）
- [ ] HB-1～HB-4 verification all implemented
- [ ] hb_graph_trace.json export（Debug/CI中心）
- [ ] hb_violation_report.json on violation
- [ ] verify() callable post-shutdown

### Retire Runtime Closed条件

- [ ] 5 lanes implemented (RTIntent/Coordination/Epoch/Reclaim/Quarantine)
- [ ] RR-1 static lint (Atomic Dot-Call Scan) all pass
- [ ] RR-2 runtime gate (epoch settle required for reclaim)
- [ ] RR-3 all reclaim instrumented
- [ ] retire_timeline.json export（Debug/CI中心）
- [ ] retire_quarantine.json on quarantine
- [ ] lane-separated audit test passing

### Shutdown Runtime Closed条件

- [ ] 7-state FSM implemented with barrier() calls
- [ ] HB chain (AudioStopped→...→ShutdownComplete) verified
- [ ] state transition guards enforced
- [ ] verifyShutdownTrace() passing
- [ ] shutdown_trace.json export（shutdown/CI中心）
- [ ] shutdown_violation_report.json on error
- [ ] state machine test case (all transitions) passing

### Evidence Export Hooks Closed条件

- [ ] 5 core artifacts emit from respective layers
- [ ] artifact naming canonical (schema + filename match)
- [ ] CI artifact validator all pass
- [ ] artifact missing detection enabled
- [ ] artifact invalid detection enabled
- [ ] timing (on-publish + on-shutdown + periodic) all working

### Budget / Trace Governance Closed条件

- [ ] 6 budgets monitored runtime
- [ ] closure traversal complexity bounded
- [ ] validation latency within SLA
- [ ] retire latency < epoch
- [ ] RT instrumentation zero-alloc verified
- [ ] metadata compaction working (defrag policy)
- [ ] runtime_budget_report.json emit on SLA violation

### Safe Failure Handling Closed条件

- [ ] 5 recovery actions implemented and mapped
- [ ] failure→action decision automatic
- [ ] RC-1 (unsafe continuation forbidden) enforced
- [ ] recovery_trace.json export（Debug/CI中心）
- [ ] recovery test case (each failure type) passing

### Introspection Closed条件

- [ ] ISRIntrospectionConsole interface complete
- [ ] snapshot() dumps minimal state（snapshot / retire / shutdown）
- [ ] runtime_snapshot.json emit on request
- [ ] snapshot schema validation passing
- [ ] diagnostic console functional test passing

---

## 最終統合形

```text
ClosureBuilder (publish-time helper)
         ↓
Payload Contract System (type enforcement)
         ↓
SealedRuntime (immutability)
         ↓
Executable HB Runtime (concurrency proof)
         ↓
Retire Runtime (lifecycle)
         ↓
Shutdown Runtime (synchronization)
         ↓
Evidence Export Hooks (artifact generation)
         ↓
Budget / Trace Governance (complexity governance)
         ↓
Safe Failure Handling (failure mapping)
         ↓
Introspection (operational observability)
         ↓
[COMPLETE] safe runtime + evidence/CI governance
```

---

## 最重要ポイント

**specification excellence** is necessary but not sufficient.

Completion requires:

```text
runtime が invariant を安全側で保持し、
必要な evidence を export して CI が検証可能な状態
```

This is achieved by:

1. **Runtime minimality**: Release は最小 barrier / fail-safe を維持
2. **Evidence export**: runtime は必要証跡を export
3. **CI strict validation**: CI が schema/整合を厳格検証
4. **Safe failure handling**: unsafe continuation を禁止し安全側へ降格
5. **Budget governance**: RT deadline 非攪乱を最優先

---

## 参照

- **Plan hub**: `doc/work/plan5.md` (10層ロードマップ)
- **Phase0 baseline**: `doc/work/ISR_Minimal_Phase0_Recommended.md`
- **P9 specification**: `doc/work/ISR_Runtime_Object_Model_Integration.md`
- **P10 specification**: `doc/work/ISR_Runtime_Proof_and_Recovery_Integration.md`
- **Governance baseline**: `doc/work/ISR_Completeness_Risk_Backlog.md` (R11-R18)
