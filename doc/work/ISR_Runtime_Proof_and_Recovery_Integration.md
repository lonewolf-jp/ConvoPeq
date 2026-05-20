# ConvoPeq ISR Runtime Evidence and Safe Failure Integration

## 目的

本書は **P10: Evidence Export, Safe Failure Handling, Introspection** の authoritative specification である。

ISR の完成を「runtime object model の evidence export」と「safe failure containment」により達成し、
最終的に **safe runtime + evidence/CI governance** を実現する。

### REV3.2運用優先注記（上書き規則）

- 本書の `self-proving / autonomous emit / runtime self-verification` 記述は
    **設計参照表現**として扱い、実装運用は `plan5.md` の REV3.2 を優先する。
- 固定方針は `runtime exposes evidence / CI validates evidence` とする。
- Build別運用: Release=minimal crash-only evidence（常時full artifact必須としない）/ Debug=optional-partial trace / CI=full artifact + strict schema validation（mandatory）

---

## P10 完成層（3 subsystems）

### 1) Evidence Export Hooks（self-verification reference）

**目的**: runtime object が build profile に応じて evidence artifact を export する

**設計原則**:

- runtime が exports → CI が validates（責務分離）
- evidence missing = CI merge reject（Releaseはoptional, Debugはrecommended）
- evidence invalid = CI merge reject（CI only mandatory）

**mandatory artifacts**:

| Artifact | Producer | Purpose |
| --- | --- | --- |
| `closure_graph.json` | ClosureBuilder | ownership closure 証跡 |
| `mutation_fault_trace.json` | SealedRuntime | immutability 違反 trace |
| `hb_graph_trace.json` | HBRuntime | concurrency order 証跡 |
| `retire_timeline.json` | RetireRuntime | lifecycle phase trace |
| `shutdown_trace.json` | ShutdownRuntime | barrier-backed state trace |

**CI role**: artifact evaluator のみ

**merge blocker**:

- artifact missing
- JSON parse error
- schema mismatch
- schemaVersion 不一致
- artifactType とファイル名不一致

### 2) Budget / Trace Governance（complexity control）

**目的**: ISR 自体の複雑性暴走を防止する

**mandatory limits**:

| 項目 | 制限 |
| --- | --- |
| closure traversal | O(N) bounded |
| publish validation latency | bounded |
| RT instrumentation | zero alloc / lock-free |
| artifact size | bounded（storage budget） |
| retire latency overhead | bounded（jitter budget） |
| metadata growth | bounded（memory budget） |

**mandatory policies**:

- RB-1: validator 増殖禁止（同一カテゴリ重複 validator 禁止）
- RB-2: RT trace sampling mandatory（全量 trace 禁止）
- RB-3: metadata compaction mandatory（canonical fields に限定）

**monitoring artifact**: `runtime_budget_report.json`

### 3) Safe Failure Handling & Introspection（lifecycle completion）

#### Safe Failure Handling（failure containment）

**目的**: failure を deterministic に contained state へ移行させる

**recovery actions**:

```cpp
enum class RecoveryAction {
    RejectPublish,    // publish validation fail
    Quarantine,       // HB/retire violation
    DelayedReclaim,   // retire timeout
    SafeMode,         // shutdown violation
    Abort             // seal violation
};
```

**mandatory mapping**:

| Failure | Recovery Action |
| --- | --- |
| closure invalid | RejectPublish |
| tier violation | RejectPublish |
| HB violation | Quarantine |
| retire timeout | DelayedReclaim |
| seal violation | Abort |
| shutdown violation | SafeMode |

**invariant**: RC-1: unsafe continuation 禁止

**artifact**: `recovery_trace.json`

#### Introspection（operational visibility）

**目的**: runtime 内部状態を operational debugging 向けに expose する

**observable state**:

- current snapshot（summary）
- retire queue / retire lanes（phase summary）
- shutdown FSM（current + recent transitions）

注記（REV3.2運用）:

- HB full graph / proof archive / schema linkage は Debug/CI layer の責務とし、
    Runtime Core 常駐責務にしない。

**export interface**:

```cpp
class ISRIntrospectionConsole {
public:
    RuntimeSnapshot snapshot() const;
    RetireView retireLanes() const;
    ShutdownView shutdownFSM() const;
};
```

**export artifact**: `runtime_snapshot.json`（minimal state dump / Debug/CI or on-demand）

---

## P9 ＋ P10 統合完成形

```text
ClosureBuilder
    ↓
    SealedRuntime
    ↓
    Payload Contract System (typed publish)
    ↓
    RetireRuntime
    ↓
    HBRuntime
    ↓
    ShutdownRuntime
    ↓
    Evidence Export Hooks (runtime artifacts)
    ↓
    Budget / Trace Governance (complexity control)
    ↓
    Safe Failure Handling (failure containment)
    ↓
    Introspection Console (operational debugging)
```

---

## ISR 完成判定基準（exit gate）

以下をすべて満たすこと:

- [ ] P9: 6 つの runtime object（5 core + DSPHandleRuntime）が few-authority で invariant を保証する
- [ ] P10: runtime が 5 つの mandatory artifact を build profile 準拠で stable に export する
- [ ] CI: 全 artifact を evaluator として merge blocker で検証する
- [ ] Budget: runtime complexity が制御予算内に収まる
- [ ] Recovery: failure を safe に contained state へ移行させる
- [ ] Introspection: runtime snapshot（minimal）が operational debugging に use できる

---

## 最終成果物

- `ISR_Formal_Guarantee_Package.md` P1-P10（package）
- `ISR_Runtime_Object_Model_Integration.md`（P9 detail）
- `ISR_Runtime_Proof_and_Recovery_Integration.md`（P10 detail）
- evidence artifacts: `closure_graph.json`, `mutation_fault_trace.json`, `hb_graph_trace.json`, `retire_timeline.json`, `shutdown_trace.json`
- monitoring artifacts: `runtime_budget_report.json`, `recovery_trace.json`, `runtime_snapshot.json`

---

## ステータス

- Spec-Fixed: 2026-05-20
- Closed: 未完（P9 runtime object + P10 evidence export の実装が未完）
