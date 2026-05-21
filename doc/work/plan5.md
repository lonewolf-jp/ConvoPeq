# ConvoPeq ISR移行計画（軽量ハブ・正本参照専用）

## 位置づけ

本書は **ISR implementation governance baseline** を維持するためのハブ文書です。
詳細仕様は個別正本に保持し、本書は「要約・優先順位・ゲート管理」のみを扱います。

## 正本ドキュメント（1 topic = 1 authoritative spec）

- 全体方針: `doc/work/ISR改修計画書_修正版_現状認識.md`
- Phase A ガイド: `doc/work/ISR_Phase_A_詳細実装ガイド.md`
- Runtime分類: `doc/work/ISR_Runtime_State_Matrix.md`
- DSP分解: `doc/work/ISR_DSPCore_Decomposition_Analysis.md`
- ownership可視化: `doc/work/ISR_Runtime_Ownership_Graph_完全可視化.md`
- retire authority: `doc/work/ISR_Retire_Authority_Graph.md`
- HB仕様: `doc/work/ISR_HB_Graph_Specification.md`
- immutability enforcement: `doc/work/ISR_Immutability_Enforcement_Spec.md`
- DSPHandle allocator policy: `doc/work/ISR_DSPHandle_Allocator_Policy.md`
- shared EpochDomain scalability検証: `doc/work/ISR_Shared_EpochDomain_Scalability_Validation_Plan.md`
- 未完リスク統制バックログ: `doc/work/ISR_Completeness_Risk_Backlog.md`
- 形式保証パッケージ: `doc/work/ISR_Formal_Guarantee_Package.md`
- 最小フェーズ0（推奨）: `doc/work/ISR_Minimal_Phase0_Recommended.md`
- [★NEW] Closure Descriptor: `doc/work/ISR_Runtime_Closure_Descriptor.md`
- [★NEW] Payload Tier: `doc/work/ISR_Payload_Tier_Model.md`
- [★NEW] Retire Intent Bridge: `doc/work/ISR_Deferred_Retire_Intent_Bridge.md`
- [★NEW] Shutdown FSM: `doc/work/ISR_Shutdown_State_Machine.md`
- [★NEW] HB Failure Model: `doc/work/ISR_Minimal_HB_Failure_Model.md`
- [★NEW] Verification Pipeline: `doc/work/ISR_Verification_Pipeline.md`
- [★NEW] Runtime Reduction Strategy: `doc/work/ISR_Runtime_Reduction_Strategy.md`
- [★NEW] Proof Artifact Schema Registry: `doc/work/ISR_Proof_Artifact_Schema_Registry.md`
- [★NEW] Runtime Object Model Integration（P9）: `doc/work/ISR_Runtime_Object_Model_Integration.md`
- [★NEW] Runtime Evidence and Safe Failure Integration（P10 / legacy filename）: `doc/work/ISR_Runtime_Proof_and_Recovery_Integration.md`
- [★NEW] 10層実装仕様（完成案）: `doc/work/ISR_10Layer_Implementation_Specification.md`
- [★REV2] JUCE Lifecycle Isolation Runtime: `doc/work/ISR_JUCE_Lifecycle_Isolation.md`
- [★REV2] DSPHandle Runtime（resolution + quarantine）: `doc/work/ISR_DSPHandle_Runtime.md`
- [★REV2] RT Execution Frame Separation: `doc/work/ISR_RT_Execution_Frame.md`
- [★REV2] World Bridge Runtime（world/epoch reference semantics）: `doc/work/ISR_World_Bridge_Runtime.md`
- [★REV3] Execution Authority Convergence: `doc/work/ISR_Execution_Authority_Convergence.md`

## 総合評価（2026-05-20 Rev.2）

| 項目 | 評価 |
| --- | --- |
| governance | 非常に良好 |
| spec separation | 良好 |
| HB formalization | 良好 |
| retire authority | 良好 |
| allocator policy | 良好 |
| backlog governance | 良好 |
| enforcement direction | 良好 |
| recursive closure | 未完（spec新規作成済み） |
| payload boundary | 未完（spec新規作成済み） |
| shutdown convergence | 未完（spec新規作成済み） |
| HB failure model | 未完（spec新規作成済み） |
| formal enforcement | まだ弱い（pipeline spec新規作成済み） |
| **JUCE lifecycle isolation** | **未閉塞（REV2 spec新規作成済み）** |
| **DSP ownership ambiguity** | **未閉塞（REV2 spec新規作成済み）** |
| **RT execution contamination** | **未閉塞（REV2 spec新規作成済み）** |
| **world/epoch ambiguity（reference）** | **未閉塞（REV2 spec新規作成済み）** |
| **execution-time invariant closure** | **未閉塞（REV3 spec新規作成済み）** |

## 設計判断（固定）

- `Spec-Fixed` と `Closed` を厳密分離する
- `1 topic = 1 authoritative spec` を維持し、ハブと正本の責務混在を禁止する
- `single authority != single mega-manager` を維持し、authority identity と lane を分離する
- RuntimePublication は runtime-only publish world とし、GlobalSnapshot / RTLocalState と分離する
- `const化` ではなく sealed-at-publish + CI enforceable で post-publish mutation を禁止する
- 既存の snapshot 系（`GlobalSnapshot` / SnapshotCoordinator / SnapshotRetireManager / epoch / immutable snapshot）を
  破棄せず few-authority へ収束させる（完全新規ISRの再構築を行わない）

## 実装開始可否の境界（固定）

### Phase A/B 開始: 可（条件付き）

- Runtime Matrix / HB / authority / allocator / enforcement / backlog が正本化済み
- R1〜R25 の最小検証項目が定義済み（R1〜R18=必須コア、R19〜R25=ガード付き拡張）

### RuntimePublication 最終固定・ISR完成宣言: 不可（現時点）

以下がすべて `Closed` になるまで不可:

- recursive ownership closure
- shutdown HB completeness
- bug2 minimal HB model
- RuntimePublication の payload boundary
- RT detect -> NonRT retire bridge 実装検証
- shared/split epoch migration の実測比較

### 未決事項の確定（R番号固定）

未決事項は以下 **7件に固定**する（追加・名称変更は `ISR_Completeness_Risk_Backlog.md` のR採番更新を必須とする）。

| 未決事項（固定名） | R | 状態 | 正本 |
| --- | --- | --- | --- |
| recursive ownership closure | R11 | Spec-Fixed | `doc/work/ISR_Runtime_Closure_Descriptor.md` |
| RuntimePublication の payload boundary | R12 | Spec-Fixed | `doc/work/ISR_Payload_Tier_Model.md` |
| RT detect -> NonRT retire bridge 実装検証 | R14 | Spec-Fixed | `doc/work/ISR_Deferred_Retire_Intent_Bridge.md` |
| shutdown HB completeness | R15 | Spec-Fixed | `doc/work/ISR_Shutdown_State_Machine.md` |
| bug2 minimal HB model | R16 | Spec-Fixed | `doc/work/ISR_Minimal_HB_Failure_Model.md` |
| shared/split epoch migration 実測比較 | R17 | Spec-Fixed | `doc/work/ISR_Shared_EpochDomain_Scalability_Validation_Plan.md` |
| formal enforcement（merge blocker 化） | R18 | Spec-Fixed | `doc/work/ISR_Verification_Pipeline.md` |

`Closed` 判定は本ハブで再定義しない。各 R の **Closed最小検証項目**（`ISR_Completeness_Risk_Backlog.md`）を唯一の判定基準とする。

---

## 形式保証と実装順の正本参照

- 形式保証パッケージ（P1〜P10）: `doc/work/ISR_Formal_Guarantee_Package.md`
  - P1〜P8: Specification formalization（governance baseline）
  - P9: Runtime Object Model Integration（implementation）
  - P10: Evidence Export, Safe Failure Handling, Introspection（completion / Debug-CI centered）
- 未完リスクとClosed最小検証項目（R1〜R25）: `doc/work/ISR_Completeness_Risk_Backlog.md`
- 当面の実行順（最小フェーズ0）: `doc/work/ISR_Minimal_Phase0_Recommended.md`

本ハブでは P1〜P10 の仕様本文・優先順テーブルを再掲しない。

## 優先実装シーケンス（P1〜P10 統合）

本系列の完成目標は **safe audio runtime + evidence export / CI validation architecture** である。
以下は実装班向けの推奨シーケンスであり、詳細ルール・受入条件は各正本にのみ保持する。

---

## ⭐ 10層実装ロードマップ（完成案）

### 根本課題の認識

current specification は「well-governed ISR specification」として優秀だが、未完なのは：

```text
few-authority 実装へ安全に着地するための
evidence export / debug verification 境界
```

である。完成条件は：

```text
runtime が memory-safety / authority invariants を最小責務で保持し、
必要な evidence を build profile に応じて export し、
CI が strict validation を担う状態
```

- 10-layer は **conceptual reference only** とする。
- 実装authorityは `ISR_Execution_Authority_Convergence.md` の few-authority 7 subsystem を正本とする。

### 10層構造（実装優先順）

| 層 | 名称 | 出口 artifact | 関連R |
| --- | --- | --- | --- |
| 1 | ClosureBuilder（publish-time helper） | closure_graph.json | R11 |
| 2 | Payload Contract System | (compile-time enforcement) | R12 |
| 3 | SealedRuntime | mutation_fault_trace.json | R13 |
| 4 | Executable HB Runtime | hb_graph_trace.json | R16 |
| 5 | Retire Runtime | retire_timeline.json | R14 |
| 6 | Shutdown Runtime | shutdown_trace.json | R15 |
| 7 | Evidence Export Hooks | 5 core artifacts（CI mandatory / Release minimal） | R18 |
| 8 | Budget / Trace Governance | runtime_budget_report.json（Debug/CI） | R18 |
| 9 | Safe Failure Handling | recovery_trace.json（Debug/CI） | (P10) |
| 10 | Introspection | runtime_snapshot.json（Debug/CI or on-demand） | (P10) |

### 各層の mandatory completeness criteria

詳細は **`ISR_10Layer_Implementation_Specification.md`** を参照（実装班向け正本）。

本ドキュメント提供内容：

- 各層の必須interface（C++ code例）
- 各層の必須invariants（名付き法則）
- 各層の必須artifacts（生成物 + schema）
- 各層の完成判定基準（全checklist）

### 完成形の定義

**Pre-完成（current: specification-driven）**:

```text
✓ Specification formalized (P1-P8)
✓ Invariants named and documented
✗ few-authority 実装へ責務収束
✗ evidence export / CI validation 境界固定
✗ Release 最小責務 / Debug-CI 拡張の build 分離
```

**完成（target: safe runtime first）**:

```text
✓ Specification formalized (P1-P8)
✓ Invariants named and documented
✓ few-authority 7 subsystem へ実装収束
✓ Release: minimal barriers + fail-safe only
✓ Debug: partial trace / assert-trap
✓ CI: full evidence validation + schema strict
✓ Safe failure handling（mute / bypass / quarantine / abort）
✓ Introspection は Debug/CI または明示要求時のみ
```

---

### Phase 0 (Specification Formalization)：P1〜P8（既存）

優先順は `ISR_Minimal_Phase0_Recommended.md` を参照。

### Phase 1-2 (Runtime Object Model)：P9（新規）

実装優先順（ボトルネック順）:

1. ClosureBuilder（R11: ownership root invariant）
2. SealedRuntime（R13: immutability enforcement）
3. Payload Contract System（R12: typed publish）
4. RetireRuntime（R14: lifecycle management）
5. HBRuntimeCore + HBTraceRuntime + HBVerifierRuntime（R16: concurrency proof）
6. ShutdownRuntime（R15: synchronization FSM）

詳細: `doc/work/ISR_Runtime_Object_Model_Integration.md`

補足（REV2優先）:

- 本節の「実装優先順」は P9 実装観点の要約であり、execution authority を含む最終順序は
  下記「REV2 未閉塞4系統」の **修正版実装順序（13ステップ）** を正本とする。

### Phase 3 (Evidence & Safe Failure Handling)：P10（新規）

Completion シーケンス（Phase 1-2 後続）:

1. Evidence Export Hooks（R18: artifact export contract）
2. Budget / Trace Governance（R18: complexity control）
3. Safe Failure Handling（failure containment / fail-safe downgrade）
4. Introspection（operational debugging, Debug/CI中心）

詳細: `doc/work/ISR_Runtime_Proof_and_Recovery_Integration.md`

### 最終成果物

- Layer 0: LifecycleIsolationRuntime（JUCE isolation）
- P9: 5 core runtime（RuntimePublication / DSPHandleRuntime / RTExecutionRuntime / RetireRuntime / ShutdownRuntime）
  - host adapter（Lifecycle）
- P9+: 補完 runtime/helper（RTExecutionFrame / CrossfadeAuthorityRuntime / helper-level Epoch arbitration / World bridge）
- P10: completion subsystem（Evidence / Budget / Safe Failure / Introspection）
- core evidence artifacts（closure/seal/HB/retire/shutdown）は CI mandatory、Release は minimal / on-demand を優先

---

## ⭐ REV2 未閉塞4系統と最適閉塞案（2026-05-20）

レビュアー第2回フィードバック（2026-05-20）に基づく。
**root issue**: `execution authority architecture が未閉塞`。

### 未閉塞4系統

| 系統 | 本質 | 正本 |
| --- | --- | --- |
| A | JUCE lifecycle 非決定性 | `ISR_JUCE_Lifecycle_Isolation.md` |
| B | DSP ownership ambiguity | `ISR_DSPHandle_Runtime.md` |
| C | RT execution contamination | `ISR_RT_Execution_Frame.md` |
| D | World/Epoch ambiguity（reference） | `ISR_World_Bridge_Runtime.md` |

### 追加検討 runtime/helper（参照設計）

| runtime | 役割 | 系統 |
| --- | --- | --- |
| LifecycleIsolationRuntime | JUCE callback → ISR-safe event stream | A |
| LifecycleBarrierRuntime | phase ordering HB 化 | A |
| DSPHandleRuntime | ownership source-of-truth | B |
| CrossfadeAuthorityRuntime | crossfade lifecycle isolation | B |
| DSPQuarantineRuntime | retire/reclaim safety | B |
| RTExecutionFrame | RT state 完全局所化 | C |
| RTCapabilityFirewall | RT authority leakage 防止 | C |
| EpochArbitrationHelper | helper-level epoch arbitration reference | D |
| WorldBridgeUtility | helper-level world/epoch routing reference | D |
| ShutdownConvergenceRuntime | shutdown executable convergence proof | E |

### 修正版実装順序

| 順 | 実装 | 系統 |
| -- | --- | --- |
| 0 | LifecycleIsolationRuntime | A |
| 1 | RTLocalState separation（RTExecutionFrame） | C |
| 2 | DSPHandleRuntime | B |
| 3 | ClosureBuilder | Layer 1 |
| 4 | SealedRuntime | Layer 3 |
| 5 | RetireRuntime | Layer 5 |
| 6 | HBRuntimeCore + HBTraceRuntime + HBVerifierRuntime | Layer 4 |
| 7 | ShutdownConvergenceRuntime | Layer 6 |
| 8 | Evidence hooks + Budget gate | Layer 7-8 |
| 9 | SafeFailureHandling（recoverable/non-recoverable 分離） | Layer 9 |
| 10 | Introspection（Debug/CI中心） | Layer 10 |
| 11 | Budget gate（bounded ring model） | Layer 8 |
| 12 | WorldBridgeUtility + EpochArbitrationHelper | D |

**理由**: DSP lifetime ambiguity を先行閉塞しないと "unsafe runtime を綺麗に観測するだけ" になる。

### Global Invariants（GI-1 ～ GI-7）

詳細は各正本参照。7つの cross-system invariants:

| 番号 | 内容 | 正本 |
| --- | --- | --- |
| GI-1 | publish graph immutable after seal | `ISR_Immutability_Enforcement_Spec.md` |
| GI-2 | RT thread owns no reclaim authority | `ISR_RT_Execution_Frame.md` |
| GI-3 | all DSP lifetime routed through DSPHandleRuntime | `ISR_DSPHandle_Runtime.md` |
| GI-4 | shutdown convergence provable | `ISR_Shutdown_State_Machine.md` |
| GI-5 | cross-world epoch globally arbitrated | `ISR_World_Bridge_Runtime.md` |
| GI-6 | HB edges executable（Debug/CI may observe） | `ISR_HB_Graph_Specification.md` |
| GI-7 | verification must never perturb RT deadline（RT path は bounded fixed-size telemetry のみ） | `ISR_Runtime_Reduction_Strategy.md` |

### HBRuntime 3分割（必須）

現行の `emitEvent() / verify()` 混在を解消:

- `HBRuntimeCore`: publishBarrier / observeBarrier / retireBarrier（latency-critical）
- `HBTraceRuntime`: emit / snapshot（non-RT side）
- `HBVerifierRuntime`: verify（CI / shutdown time）

詳細: `doc/work/ISR_HB_Graph_Specification.md`

### Safe Failure Handling 分類（必須）

| violation | action | 理由 |
| --- | --- | --- |
| seal violation | Abort | memory safety violation |
| stale DSP handle | CI: Abort / Release: Quarantine+Silence | build profile で安全側へ降格 |
| reclaim-before-grace | Abort | memory safety violation |
| telemetry mismatch | Continue | non-safety |
| introspection failure | Ignore | observability only |

詳細: `doc/work/ISR_Runtime_Proof_and_Recovery_Integration.md`

### Runtime Budget System（bounded ring model 必須）

`hb_graph_trace.json` はイベント数指数増加リスクあり。bounded ring model 必須:

```text
Release: sampled / Debug: full / CI: deterministic bounded
RB-4: evidence generation must not perturb RT deadline
```

運用固定（最新レビュー反映）:

- Runtime budget の governance/監査は Debug/CI layer を主責務とし、runtime core 常駐責務にしない。
- Release runtime は bounded ring buffer + fixed telemetry cap を最小実装とする。

詳細: `doc/work/ISR_10Layer_Implementation_Specification.md`（Layer 8）

---

## ⭐ REV3 実行時 invariant closure（2026-05-20 follow-up）

REV2 で runtime object architecture は大幅に前進したが、
follow-up review により以下の差分が未閉塞と判定された。

```text
runtime object exists
!=
runtime execution cannot violate invariants
```

本ハブでは要約のみを保持し、詳細は正本へ委譲する。

- 正本: `doc/work/ISR_Execution_Authority_Convergence.md`
- 対象領域（7）:
  - Authority convergence
  - Temporal determinism
  - Crossfade visibility
  - World/Epoch convergence（reference）
  - RT contamination closure
  - Shutdown convergence under failure
  - Evidence/Debug-system containment

関連 R 採番（R19-R25）は `ISR_Completeness_Risk_Backlog.md` を正本とする。

---

## ⭐ 安定性優先プロファイル（実運用推奨）

follow-up review に基づき、ConvoPeq の現実運用では
「理論的完全性」より「実装者が invariant を壊しにくいこと」を優先する。

```text
理論的に閉じている
<
実装時に破壊しにくい
```

本プロファイルは runtime object の増殖を抑制し、memory safety 直結領域へ収束させる。

### 必須（Release 常時）

- LifecycleIsolationRuntime
- RTExecutionFrame
- DSPHandleRuntime
- RetireRuntime
- ShutdownConvergenceRuntime

### 次点（必要時）

- HBRuntimeCore
- CrossfadeAuthorityRuntime

### Debug/CI 限定

- HBTraceRuntime
- HBVerifierRuntime
- Evidence/Artifact 系 runtime

### Build プロファイル

- Release: barrier 最小セット（publish/retire/shutdown）
- Debug: trace 部分有効
- CI: verify/simulation フル有効

### 収束先（実運用ゲート）

以下 6 条件を満たすことを優先完了条件とする:

- A. publish graph immutable
- B. callback-local frozen view
- C. DSP ownership singular
- D. reclaim centralized
- E. RT isolation
- F. bounded shutdown

---

## ⭐ REV3.1 Few-Authority 収束（実装者保全優先）

追加レビューに基づき、ConvoPeq では以下を強制方針とする。

```text
many-runtime architecture
->
few-authority architecture
```

### 実コード上の推奨 subsystem（7）

- LifecycleHostAdapter（host adapter helper）
- RuntimePublication
- DSPHandleRuntime
- RTExecutionRuntime
- RetireRuntime
- ShutdownRuntime
- DebugRuntime（Debug/CI 限定）

運用解釈（最新版レビュー反映）:

- Runtime Core は **5 subsystem**（RuntimePublication/DSPHandleRuntime/RTExecutionRuntime/RetireRuntime/ShutdownRuntime）を基本とする。
- Lifecycle は host adapter helper として扱う。
- `DebugRuntime` は **Optional Debug Layer** として扱い、runtime core とは分離する。
- 原則: **debug system ≠ runtime core**。

### 統合方針（過剰分裂の抑制）

- `ClosureBuilder` + `SealedRuntime` -> `RuntimePublication`
- `CrossfadeAuthorityRuntime` -> `DSPHandleRuntime`
- `DSPQuarantineRuntime` -> `RetireRuntime`
- `HBTraceRuntime` + `HBVerifierRuntime` -> `DebugRuntime`
- `Validation/Budget/Proof` -> `DebugRuntime`

Closure責務境界（最新レビュー反映）:

- 実装面では `ClosureBuilder` を巨大化させず、publish ownership closure（create/connect/seal/validateAcyclic/freeze 相当）に限定する。
- governance/proof orchestration/introspection は Runtime Core へ常駐させない。

### world model の簡略化

- `PublicationWorld`（immutable runtime）
- `ExecutionWorld`（audio callback）

full federation semantics は採用しない（single plugin process 前提）。

### Authority model（capability-first）

- authority 制約は runtime object 単体より **type-level capability** を優先する。
- 推奨 capability tag:
  - `PublishAuthority`
  - `RetireAuthority`
  - `ShutdownAuthority`
- authority coordinator の runtime lifecycle は導入しない。
- 移行期互換が必要な場合でも、coordinator は build-time/helper（非RT・非core）へ限定する。

### ISR-1..ISR-6（実装者が壊しにくい invariant）

- ISR-1: publish後runtime immutable
- ISR-2: callback中runtime snapshot stable
- ISR-3: RetireRuntime以外reclaim禁止
- ISR-4: crossfade完了前retire禁止
- ISR-5: RT thread authority mutation禁止
- ISR-6: shutdown bounded completion mandatory

---

## ⭐ REV3.2 実運用ハードニング（混在解消ルール）

最新レビューに基づき、ConvoPeq の運用方針を以下で固定する。

```text
safe runtime first
> self-verification framework
```

### 実装権威の固定

- `10-layer` は **conceptual reference only**。
- 実装authorityは `few-authority 7 subsystem` を正本とする。
- authority 制約は coordinator 中心ではなく capability-first で運用する。
- authority coordinator の runtime object 化を禁止し、
  authority は compile-time capability（Publish/Retire/Shutdown）で固定する。

### 証跡責務の固定

- 方針は `runtime verifies runtime` ではなく
  **`runtime exposes evidence / CI validates evidence`** とする。
- artifact policy は build-dependent に固定する（Release=optional / Debug=recommended / CI=mandatory）。

### Build別の固定挙動

- Release: minimal barriers（publish/retire/shutdown）+ sampled trace のみ
- Debug: partial trace + assert/trap
- CI: full verify/simulation + artifact schema strict

補足（最新版レビュー反映）:

- Release は crash-only minimal evidence を優先し、artifact 常時full emit を要求しない。
- Debug/CI の trace/verification は Optional Debug Layer で運用する。
- Evidence Export Hooks / Budget Governance / HBTrace / Introspection / schema linkage は
  Debug/CI layer を主責務とし、Runtime Core 常駐責務にしない。
- Runtime Core は publish/retire/shutdown の最小 barrier と fail-safe quarantine を優先する。

### 現実運用レビュー追補（verification platform化抑制）

- Runtime Core に残す常駐責務は以下のみとする:
  - `publishBarrier`
  - `retireBarrier`
  - `shutdownBarrier`
  - `quarantine`
- Runtime Core は **small hard runtime core** を優先し、
  RuntimePublication / DSPHandleRuntime / RTExecutionRuntime / RetireRuntime / ShutdownRuntime を実働中心とする。
  （Lifecycle は host adapter として扱い、core相互依存を増やさない）
- 以下は Runtime Core へ常駐させず、Debug/CI layer へ隔離する:
  - HB graph construction
  - artifact schema validation
  - budget governance
  - introspection graph
  - proof archive
- `ClosureBuilder`（統合先: RuntimePublication）は巨大化を禁止し、
  publish closure correctness（`createNode()`, `connect()`, `seal()`, `validateAcyclic()`, `freeze()` 相当）に限定する。
- plugin 規模では `WorldBridgeUtility` / `EpochArbitrationHelper` の新規常駐導入を行わない
  （2-world: Publication/Execution を運用正本とする）。
- `WorldBridgeUtility` は Debug helper 程度へ縮退し、
  `EpochArbitrationHelper` は helper-level utility を超えて昇格させない。
- artifact は postmortem evidence として扱い、runtime correctness は runtime invariants で成立させる。
  （運用: Release=minimal, Debug=optional, CI=mandatory）
- Introspection は最小状態（current snapshot / retire queue state / shutdown state）を上限とし、
  HB full graph export は CI 限定とする。
- contributor survivability を優先し、新規 runtime object 追加は
  「authority 減少」または「interaction complexity 純減」の定量根拠がある場合のみ許可する。

### stale DSP handle の運用

- CI: abort
- Debug: assert（必要時 trap）
- Release: quarantine + silence（必要時 fail-safe bypass）

### World / Epoch の整理

- `WorldBridgeUtility` は utility helper（authority root ではない）。
- epoch arbitration は RetireRuntime 内部責務への統合を優先する。
- `WorldBridgeUtility` / `EpochArbitrationHelper` は plugin 規模では reference 扱いとし、
  Runtime Core への新規常駐導入を行わない。

---

## ⭐ 文書優先順位（解釈衝突時）

解釈が衝突する場合の優先順位を以下に固定する。

1. `plan5.md` の REV3.1 Few-Authority 方針（実装運用の最上位）
2. `ISR_Completeness_Risk_Backlog.md`（R1〜R25 の判定基準）
3. 各個別正本（詳細設計）

補足:

- 3-world/federation の記述は **参考モデル** とし、ConvoPeq 実装運用は 2-world（Publication/Execution）を優先する。
- authority 制約は capability-first を優先し、runtime coordinator lifecycle を導入しない。

運用判定ラベルは `doc/work/R11-R25_Closed判定監査表_2026-05-21.md` と同期し、現行CIゲート（seed evidence + verify + scan + build）で安定運用可能な状態を **Closed（運用重視）**、seed非依存の実行時挙動検証まで満たす状態を **Closed（厳密）**、実装はあるが運用ゲート未結線の状態を **部分適合** とする。

### 実装優先順位（現実コード整合）

- **Phase A**: DSP ownership singularity
- **Phase B**: RTExecutionFrame 完全局所化
- **Phase C**: RetireRuntime centralized reclaim
- **Phase D**: Shutdown bounded completion

上記 A-D 完了前に Debug/CI 拡張を先行しない。

---

## 重複防止規約

- `plan5.md` は要約とリンクのみ保持
- 正本にある仕様本文を `plan5.md` へ再掲しない
- 結合ファイル化（他文書全文貼り戻し）を禁止

## 監査結果リンク

- R11〜R25 Closed判定監査表（証拠ファイル/コード行付き）:
  - `doc/work/R11-R25_Closed判定監査表_2026-05-21.md`
