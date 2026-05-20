# ConvoPeq ISR Formal Guarantee Package

## 目的

ISR を「設計規律」から **formal enforceable system** へ移行し、最終的に
**runtime-enforced safe architecture** へ到達するための統合仕様。
本書は package 正本であり、各 P の詳細仕様は専用正本に委譲する。

### REV3.2 運用優先注記

- 本書に残る `self-proving` 系記述は参照設計表現として扱う。
- 実装運用は `plan5.md` REV3.2 を優先し、
  `runtime exposes evidence / CI validates evidence` を固定方針とする。
- 本書では `RuntimePublication` を正規記法として扱う。

## 階層構成

- **P1〜P8**: Specification formalization（governance）
- **P9**: Runtime object model（implementation）
- **P10**: Evidence export & safe failure lifecycle management（completion）

---

## P1. Closure Descriptor System（R11）

詳細仕様は `doc/work/ISR_Runtime_Closure_Descriptor.md` を参照。

---

## P2. Payload Tier System（R12）

詳細仕様は `doc/work/ISR_Payload_Tier_Model.md` を参照。

---

## P3. Immutable Facade + Mutable Core（R13）

### P3仕様

- publish graph は read-only projection のみ公開
- mutable cache / lazy-init / mutex は RuntimePublication から隔離

### P3必須ルール

- Rule-I1: publish graph 内 mutable atomic 禁止
- Rule-I2: publish graph 内 mutex 禁止
- Rule-I3: publish graph 内 lazy-init 禁止
- Rule-I4: publish 時 seal recursion と mutation write barrier を mandatory とする

参照: `doc/work/ISR_Immutability_Enforcement_Spec.md`

---

## P4. Deferred Retire Intent Queue（R14）

詳細仕様は `doc/work/ISR_Deferred_Retire_Intent_Bridge.md` を参照。

---

## P5. Shutdown HB FSM（R15）

詳細仕様は `doc/work/ISR_Shutdown_State_Machine.md` を参照。

---

## P6. HB Failure Spec + Reorder Simulation（R16）

詳細仕様は `doc/work/ISR_Minimal_HB_Failure_Model.md` を参照。

---

## P7. Epoch Abstraction Layer（R17）

### P7仕様

- epoch 実装へ直接依存せず、coordinator interface を経由する
- `IEpochStrategy` 相当の抽象境界を定義し、concrete 実装への直接依存を禁止する
- shared/split/hybrid を runtime policy として切替可能にする

### P7目的

- shared epoch を architecture invariant にしない

参照: `doc/work/ISR_Shared_EpochDomain_Scalability_Validation_Plan.md`

---

## P8. CI Verification Pipeline（R18）

詳細仕様は `doc/work/ISR_Verification_Pipeline.md` を参照。

---

## P9. Runtime Object Model Integration（実装統合層）

### P9仕様

本書は specification-driven から **runtime-enforced invariant architecture** へ移行するための中核層を定義する。

コア実装体：

- **ClosureRuntime**: publish graph の runtime ownership kernel
- **SealedRuntime**: runtime-enforced immutability
- **HBRuntime**: executable HB graph（trace・verify・emit）
- **RetireRuntime**: lane-separated lifecycle management
- **ShutdownRuntime**: barrier-backed FSM

詳細仕様は `doc/work/ISR_Runtime_Object_Model_Integration.md` を参照。

---

## P10. Evidence Export, Safe Failure Handling, Introspection（完成層）

### P10仕様

ISR 完成条件：

- **Evidence Export Hooks**: runtime が invariant 証跡を build profile 準拠で export
- **Budget / Trace Governance**: complexity 制御（validator proliferation 禁止、metadata compaction、RT sampling）
- **Safe Failure Handling**: failure containment と安全な降格
- **Introspection**: 運用 debugging（runtime snapshot export）

詳細仕様は `doc/work/ISR_Runtime_Proof_and_Recovery_Integration.md` を参照。

---

## 参照

- `doc/work/plan5.md`
- `doc/work/ISR_Completeness_Risk_Backlog.md`
