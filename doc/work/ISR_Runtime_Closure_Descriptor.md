# ConvoPeq ISR Runtime Closure Descriptor

## 目的

本書は **R11: Closure Descriptor System** の authoritative specification である。

RuntimePublication に publish される全オブジェクトの transitive ownership を
`ClosureNode` ツリーとして形式化し、`validateClosureGraph()` により
publish 前に closure の完全性を検証可能にする。

用語正規化（齟齬回避）:

- 本書では `RuntimePublication` を正規記法として扱う。

---

## 背景・動機

ISR の真の危険は top-level immutability ではなく **transitive dependency leak** にある。

```text
RuntimeGraph
→ DSPHandle
→ FFT cache
→ background mutable state   ← ここで ISR 崩壊
```

このような hidden mutable dependency は、closure を形式化しない限り
静的・動的検証のいずれでも発見できない。

---

## Core 構造体定義

```cpp
struct ClosureNodeId
{
    uint64_t value;
};

enum class ClosureEdgeKind
{
    Owns,           // exclusive ownership / lifetime managed by parent
    References,     // shared ownership, epoch-managed
    Borrows,        // temporary read-only borrow, HB-guarded
    ExternalPinned  // externally allocated, lifetime pinned by caller
};

struct ClosureEdge
{
    ClosureNodeId from;
    ClosureNodeId to;
    ClosureEdgeKind kind;
};

enum class ClosureObjectKind
{
    RuntimeGraphNode,
    DSPHandle,
    CoeffBuffer,
    FFTPlan,
    IRBlob,
    SmoothingState,
    TelemetryCounter,
    ExternalDevicePtr,
    Unknown
};

enum class ClosureMutability
{
    Immutable,       // never mutated after publish
    RTLocalOnly,     // mutated only on Audio Thread, not in publish closure
    Mutable          // forbidden in publish closure
};

enum class ClosureLifetime
{
    EpochManaged,    // retired via EpochDomain
    Quarantine,      // held in quarantine until grace period
    ExternalPinned   // lifetime managed externally
};

enum class ClosureAllocatorFamily
{
    Pool,            // pool allocator (ISR-managed)
    Quarantine,      // quarantine allocator
    System           // system allocator (forbidden in publish closure)
};

enum class ClosureHBDomain
{
    Publish,         // Domain A: publish/observe
    Retire,          // Domain B: retire/reclaim
    Telemetry,       // Domain C: telemetry observation
    External         // outside ISR HB graph
};

struct ClosureNode
{
    ClosureNodeId       id;
    ClosureObjectKind   objectKind;
    ClosureMutability   mutability;
    ClosureLifetime     lifetime;
    ClosureAllocatorFamily allocator;
    ClosureHBDomain     hbDomain;
    ClosureNodeId       reclaimAuthority;  // who retires this node (0 = unmanaged = invalid)
};
```

---

## Closure Graph

```cpp
struct ClosureGraph
{
    ClosureNode              root;           // RuntimePublication root
    std::vector<ClosureNode> nodes;          // all reachable nodes
    std::vector<ClosureEdge> edges;          // ownership/reference edges
};
```

- `RuntimePublication` は **Closure Root** として publish される
- graph は **DAG** でなければならない（cycle = 不正）

---

## 不変条件

### C1: mutable dependency 禁止

```text
publish closure から到達可能な node で
mutability == Mutable である node が存在してはならない
```

### C2: closure 外 raw pointer 禁止

```text
publish closure 内のすべての pointer が
ClosureGraph に登録されていなければならない
closure 外 raw pointer は存在禁止
```

### C3: closure graph は DAG

```text
ClosureGraph に cycle が存在してはならない
（cycle は ownership ambiguity を意味する）
```

### C4: reclaimable node は retire authority 必須

```text
lifetime == EpochManaged または Quarantine の node は
reclaimAuthority が有効な ClosureNodeId でなければならない
```

---

## validateClosureGraph()

publish 前に必ず呼び出すこと。

```cpp
enum class ClosureValidationResult
{
    Valid,
    MutableNodeReachable,       // C1 violation
    UnregisteredRawPointer,     // C2 violation
    CycleDetected,              // C3 violation
    MissingReclaimAuthority,    // C4 violation
    ForbiddenObjectKind,        // ExternalDevicePtr など reject 対象
    AllocatorMismatch           // allocator family が lifetime と矛盾
};

ClosureValidationResult validateClosureGraph(const ClosureGraph& graph);
```

### reject 対象

以下は closure 内存在が自動 reject される：

| 種別 | 理由 |
| --- | --- |
| JUCE mutable device ptr | ClosureObjectKind::ExternalDevicePtr |
| async callback state | mutability == Mutable |
| hidden singleton | C2 violation (unregistered) |
| background mutable cache | C1 violation |
| unmanaged external ptr | C4 violation (no retire auth) |

---

## CI Graph Dump

CI build で以下を自動生成すること：

```text
closure_graph.json
```

移行期の互換 alias（`runtime_publishworld_graph.json`）は
`ISR_Proof_Artifact_Schema_Registry.md` の規定に従って扱う。

フォーマット（最小）：

```json
{
  "root": { "id": "...", "kind": "RuntimeGraphNode", "mutability": "Immutable" },
  "nodes": [...],
  "edges": [...],
  "validationResult": "Valid"
}
```

CI fail 条件：

- `validationResult != "Valid"`
- graph dump が存在しない（生成スクリプト未実行）

---

## 関連正本

- `ISR_Payload_Tier_Model.md` — ClosureNode の tier 分類詳細
- `ISR_Formal_Guarantee_Package.md` P1 — 統合保証パッケージ参照
- `ISR_Immutability_Enforcement_Spec.md` — immutability enforcement 方針

## Backlog 参照

- `ISR_Completeness_Risk_Backlog.md` R11 — Closed 最小検証項目

## ステータス

- Spec-Fixed: 2026-05-20
- Closed: 未完（実装・CI検証未実施）
