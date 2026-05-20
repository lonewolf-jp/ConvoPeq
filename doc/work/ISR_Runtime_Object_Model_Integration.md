# ConvoPeq ISR Runtime Object Model Integration

## 目的

本書は **P9: Runtime Object Model Integration** の authoritative specification である。

ISR の中核を「設計仕様」から **few-authority runtime subsystem** へ統合し、
specification-driven から **safe runtime + evidence export / CI validation** へ移行するための実装指針を定義する。

### REV3.2運用優先注記

- 本書の `evidence-export / autonomous` 記述は参照設計表現として扱う。
- 実装運用は `plan5.md` REV3.2 を優先し、
    `runtime exposes evidence / CI validates evidence` を固定方針とする。
- 10-layer は conceptual reference、実装authorityは few-authority 収束を優先する。
- Runtime Core は 5 subsystem（RuntimePublication/DSPHandleRuntime/RTExecutionRuntime/RetireRuntime/ShutdownRuntime）を優先し、Lifecycle は host adapter として扱う。
- HB trace / artifact export / budget audit / introspection / verifier は Debug/CI layer を主責務とする。

### 実装班向けガイド

詳細な実装形（interface / invariants / artifacts / Closed criteria 全て）は
**`ISR_10Layer_Implementation_Specification.md`** を参照。
本書はその specification foundation を提供する。

---

## 基本原則

- specification が命ずることを few-authority subsystem が最小責務で保持する
- validator は observer（事後確認）であり、Release 常駐 enforcer を増やしすぎない
- 各 subsystem は必要な evidence を公開し、CI が検証責務を担う

---

## P9 実装統合層（runtime core 6 + Debug/CI facets）

注: 既存の 5 core（Closure/Sealed/HB/Retire/Shutdown）に加えて、
REV2 で `DSPHandleRuntime` を P9 実装対象へ含める。

運用解釈（最新版レビュー反映）:

- 実働の small runtime core は RuntimePublication / DSPHandleRuntime / RTExecutionRuntime / RetireRuntime / ShutdownRuntime を中心とする。
- Lifecycle は host adapter として扱い、core相互依存を増やさない。
- Closure は persistent runtime ではなく publish-time helper（builder utility）として扱う。

### 1) ClosureBuilder（publish-time helper）

**目的**: publish graph の所有権を publish-time helper として保証する

**実装要件**:

```cpp
class ClosureBuilder {
public:
    ClosureNodeId createNode(ClosureNodeDescriptor desc);
    void connect(ClosureNodeId from, ClosureNodeId to, ClosureEdgeKind kind);
    void seal(ClosureNodeId root);
    ClosureValidationResult validateAcyclic(ClosureNodeId root) const;
};
```

責務制限（REV3.2整合）:

- ClosureBuilder は publish closure correctness（create/connect/seal/validateAcyclic/freeze）に限定する。
- introspection / governance / proof orchestration は Debug/CI layer 側へ委譲する。
- persistent runtime lifecycle を持たせない（builder utility 扱い）。

**必須 invariants**:

- CR-1: publishable graph は DAG（cycle 禁止）
- CR-2: sealed root から mutable node reachable 禁止
- CR-3: all reclaimable nodes require authority owner
- CR-4: RTLocal tier の closure leak 禁止

**artifact**: `closure_graph.json`（CI mandatory / Debug recommended / Release optional）

### 2) SealedRuntime（immutability kernel）

**目的**: publish 後の mutation を runtime で防止する

**実装要件**:

```cpp
class SealedObject {
public:
    void seal();
    bool sealed() const noexcept;
protected:
    void assertMutable() const;
};
```

**必須 policy**:

- publish 時 `sealRecursively(root)` mandatory
- 全 mutator に `assertMutable()` mandatory
- Release build: silent ignore 禁止（abort / quarantine 必須）

**artifact**: `mutation_fault_trace.json`

### 3) HBRuntime（concurrency correctness kernel / REV2 3分割）

**目的**: happens-before order を minimal barrier（Runtime Core）+ Debug/CI instrumentation として扱う

**実装要件**:

```cpp
struct HBNode {
    HBEventId id;
    HBEventType type;      // Publish/Observe/Retire/Reclaim/ShutdownBarrier
    Timestamp ts;
};

struct HBEdge {
    HBNodeId from;
    HBNodeId to;
    MemoryOrder order;
};

class HBRuntimeCore {
public:
    void publishBarrier(...);
    void observeBarrier(...);
    void retireBarrier(...);
};

class HBTraceRuntime {
public:
    void emit(HBEventType type);
    HBTrace snapshot() const;
};

class HBVerifierRuntime {
public:
    VerifyResult verify(const HBTrace& trace) const;
};
```

**必須 instrumentation**:

- publish / observe / retire / reclaim / shutdown barrier / epoch settle の観測点を定義
- reorder 検出は Debug/CI verifier で実装する

**artifact**: `hb_graph_trace.json`, `hb_violation_report.json`（Debug/CI / shutdown 時中心）

### 3.5) DSPHandleRuntime（DSP ownership kernel / REV2）

**目的**: DSP lifetime source-of-truth を単一窓口に統合する

**実装要件**: `ISR_DSPHandle_Runtime.md` を正本とする

**必須 invariants**:

- GI-3: all DSP lifetime routed through DSPHandleRuntime

**artifact**: `dsp_ownership_trace.json`

### 4) RetireRuntime（lifecycle ownership kernel）

**目的**: retire flow を lane-separated で管理し observability を確保する

**実装要件**:

```cpp
enum class RetireLane {
    RTIntent,        // RT emits only
    Coordination,    // authority enqueue
    Epoch,          // grace tracking
    Reclaim,        // actual destroy
    Quarantine      // delayed reclaim
};

class RetireRuntime {
public:
    void emitIntent(...);          // RT lane
    void enqueueRetire(...);       // Coordination lane
    void settleEpoch(...);         // Epoch lane
    void reclaim(...);             // Reclaim lane
    void quarantine(...);          // Quarantine lane
};
```

**必須 invariants**:

- RR-1: RT thread delete/free/reclaim 禁止（static analysis mandatory）
- RR-2: reclaim requires settled epoch
- RR-3: all reclaim emits trace

**artifact**: `retire_timeline.json`, `retire_quarantine.json`

### 5) ShutdownRuntime（synchronization FSM kernel）

**目的**: shutdown state を barrier-backed FSM で enforce する

**実装要件**:

```cpp
enum class ShutdownState {
    Running,
    AudioStopped,
    ObserverDrained,
    RetireClosed,
    EpochSettled,
    ReclaimComplete,
    ShutdownComplete
};

class ShutdownRuntime {
public:
    void transition(ShutdownState);
    void barrier();
    ShutdownState current() const;
};
```

**必須 HB chain**:

```text
AudioStopped ---HB---> ObserverDrained ---HB---> RetireClosed
    ---HB---> EpochSettled ---HB---> ReclaimComplete
```

**artifact**: `shutdown_trace.json`, `shutdown_violation_report.json`

---

## 統合条件（P9 Closed 判定基準）

以下をすべて満たすこと:

- [ ] RuntimePublication が publish graph の ownership / immutability を few-authority で保持する
- [ ] Sealed/Closure 系が post-publish mutation を fail-fast / quarantine で閉塞する
- [ ] HBRuntimeCore は minimal barrier とし、trace/verify は Debug/CI 中心で運用される
- [ ] Runtime Core は publish/retire/shutdown barrier + quarantine を最小責務で維持し、verification 常駐化を行わない
- [ ] RetireRuntime が lane-separated lifecycle を autonomous に管理する
- [ ] ShutdownRuntime が barrier-backed state machine を runtime で enforce する
- [ ] 5 つの artifact（closure/mutation/hb/retire/shutdown）が merge blocker として archive される

---

## ステータス

- Spec-Fixed: 2026-05-20
- Closed: 未完（runtime object 実装と artifact emit が未完）
