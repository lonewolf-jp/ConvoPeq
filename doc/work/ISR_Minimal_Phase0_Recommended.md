# ConvoPeq ISR 最小フェーズ0（推奨）

## 目的

`Spec-Fixed` 済みだが実装未充足の形式保証コアを先行導入し、
**Phase 1-2（P9 Runtime Object Model）：10層参照設計を few-authority 実装へ安全に読み替える基盤**を整えた状態で開始できるようにする。

本フェーズは「ISR完成」を目的としない。目的は以下の3点:

- RuntimePublication 周辺の重大乖離（R11/R12/R13/R14/R15/R16/R18）を最小コストで封じる
- **Phase 1-2（10層参照設計：ClosureBuilder/PayloadContract/SealedRuntime/HBRuntime/RetireRuntime/ShutdownRuntime）** の基盤を整備し、few-authority 実装へ読み替え可能な **Closed criteria** を準備する
- **REV2 execution authority architecture（Layer 0 + 系統A/B/C）の骨格を整備し、JUCE lifecycle isolation / DSP ownership / RT execution authority の基盤を Phase 1-2 実装前に確立する**

運用優先注記（REV3.1）:

- 10層モデルは参照設計として維持するが、ConvoPeq 実装運用は `plan5.md` REV3.1 に従い
  **few-authority / 7 subsystem 収束**を優先する。
- 解釈衝突時は `plan5.md` の「文書優先順位（解釈衝突時）」を優先する。

### REV3.2運用優先注記

- 本書の理論/参照設計記述は設計参照表現として扱う。
- 実装運用は `plan5.md` REV3.2 を優先し、
  `runtime exposes evidence / CI validates evidence` を固定方針とする。
- 解釈衝突時は few-authority（7 subsystem）/ 2-world（Publication/Execution）/
  capability-first（runtime coordinator lifecycle 非導入）を優先する。
- stale handle は CI=Abort / Debug=Assert / Release=Quarantine+Silence を優先する。
- Build別責務は Release=minimal evidence / Debug=optional-partial trace / CI=full validate を優先する。
- Runtime Core は 5 subsystem（RuntimePublication / DSPHandleRuntime / RTExecutionRuntime / RetireRuntime / ShutdownRuntime）を基本とし、DebugRuntime は Optional Debug Layer として分離する。
- 実働運用では small runtime core（RuntimePublication / DSPHandleRuntime / RTExecutionRuntime / RetireRuntime / ShutdownRuntime）を中心にし、
  Lifecycle は host adapter として扱って core 相互依存を増やさない。
- 原則: debug system ≠ runtime core。
- Runtime Core 常駐責務は publish/retire/shutdown の最小 barrier と fail-safe quarantine に限定し、
  HB trace / validator / artifact schema linkage は Debug/CI layer（Optional Debug Layer）へ委譲する。
- authority 制約は coordinator 中心ではなく type-level capability（Publish/Retire/Shutdown）を優先する。
- 既存 snapshot 系（GlobalSnapshot / SnapshotCoordinator / SnapshotRetireManager / epoch / immutable snapshot）を
  few-authority 実装へ収束させる方針を優先する。

## フェーズ全体像と 10層対応

```text
Phase 0 (Specification Formalization)  P1-P8 governance + REV2 authority 準備 + 10層準備
  P0-A0 (Layer 0) → LifecycleIsolationRuntime 骨格（REV2 系統A）
  P0-A1 (系統C)   → RTExecutionFrame / RTCapabilityFirewall 骨格（REV2 系統C）
  P0-A2 (系統B)   → DSPHandleRuntime 骨格（REV2 系統B）
  P0-1  (R11)     → Layer 1: ClosureBuilder
  P0-2  (R12)     → Layer 2: Payload Contract System
  P0-3  (R13)     → Layer 3: SealedRuntime
  P0-4  (R16)     → Layer 4: HBRuntimeCore + HBTraceRuntime + HBVerifierRuntime（3分割）
  P0-5  (R14)     → Layer 5: Retire Runtime
  P0-6  (R15)     → Layer 6: Shutdown Runtime
  P0-7  (R18)     → Layer 7-10: Evidence/Budget/Safe Failure Handling/Introspection
  ↓
Phase 1-2 (Runtime Object Model)     P9: 10層参照設計を few-authority 実装へ読み替え（修正版 13ステップ）
  ↓
Phase 3 (Evidence & Safe Failure Handling)  P10: Evidence export + Budget + Safe Failure Handling
```

Phase 0 では上記フェーズ1-2の実装基盤を整備する。

---

## スコープ（最小）

Phase 0は **REV2 execution authority architecture の骨格** と **10層参照設計の foundation** を以下10項目で整備する:

### REV2 execution authority 準備（steps 0-2 先行封止）

A0. Layer 0: LifecycleIsolationRuntime 骨格（JUCE callback → ISR lifecycle event stream 変換、系統A）
A1. 系統C: RTExecutionFrame 分離（RT callback 内状態のスタックローカル局所化）
A2. 系統B: DSPHandleRuntime 骨格（DSP lifetime 単一窓口確立、GI-3 基盤）

### 10層 runtime object model 準備（Layer 1-10）

1. R11: Closure Descriptor System （層1 ClosureBuilder 準備）
2. R12: Payload Tier System （層2 Payload Contract System 準備）
3. R13: Sealed Object Mutation Detection （層3 SealedRuntime 準備）
4. R14: Deferred Retire Intent Queue （層5 Retire Runtime 準備）
5. R15: Shutdown State Machine + HB FSM （層6 Shutdown Runtime 準備）
6. R16: HB Graph Instrumentation（3分割: HBRuntimeCore / HBTraceRuntime / HBVerifierRuntime）（層4 準備）
7. R18: CI Verification Pipeline （層7-10 Evidence export 準備）

### REV3.1 安定性優先ガード（実装運用）

- world model は `PublicationWorld / ExecutionWorld` の 2-world 簡略運用を優先
- authority source-of-truth は capability-first を優先し、runtime coordinator lifecycle を導入しない
- runtime core は RuntimePublication/DSPHandleRuntime/RTExecutionRuntime/RetireRuntime/ShutdownRuntime の 5 subsystem を優先
- Lifecycle は host adapter として扱い、core相互依存を増やさない
- 以下 ISR-1..ISR-6 を Release 完了条件の先行ゲートとして扱う:
  - ISR-1: publish後runtime immutable
  - ISR-2: callback中runtime snapshot stable
  - ISR-3: RetireRuntime以外reclaim禁止
  - ISR-4: crossfade完了前retire禁止
  - ISR-5: RT thread authority mutation禁止
  - ISR-6: shutdown bounded completion mandatory
- GI-7（運用追加）: verification は RT deadline を摂動してはならない
  （RT path は bounded fixed-size telemetry のみ）

非スコープ（Phase 0ではやらない）:

- WorldBridgeUtility / EpochArbitrationHelper の helper-level 整理を超える完全実装（Phase 1-2 ステップ12、系統D）
- DSPHandleRuntime の CrossfadeAuthorityRuntime / DSPQuarantineRuntime 完全実装（Phase 1-2 以降）
- R16 reorder simulator の「完全」実装（Phase 1-2）
- R17 epoch abstraction の全面切替
- 層8-10 Runtime Budget/Safe Failure Handling/Introspection 完全実装（Phase 3）
- Runtime matrix 全再編
- R19-R25 の full展開（Authority/2-world整理/Evidence 拡張の全面導入）
  - ただし R20（Host chaos normalization）/R21（DSP ownership単純化）は
    Release 安定性直結のため、Phase 1-2 早期導入候補として扱う

---

## 入口条件（Start Gate）

- `plan5.md` がハブ専用（要約+リンク）を維持している
- `ISR_10Layer_Implementation_Specification.md` が実装班向けガイドとして存在
- `ISR_Completeness_Risk_Backlog.md` の R11-R18 が `Spec-Fixed`
- `ISR_Completeness_Risk_Backlog.md` の R11-R25 が `Spec-Fixed`
  （R11-R18=必須コア、R19-R25=ガード付き拡張）
- 既存 CI（`audioengine-lint.yml`, `list-compliance.yml`）が green
- 10層実装仕様の「各層 Closed criteria」が定義済み

### REV2 追加入口条件（2026-05-20 reviewer 2nd feedback）

Phase 0 開始前に、以下 4 系統の正本仕様が存在することを確認すること。
これらは Layer 0 および execution authority architecture の基盤であり、
本 Phase 0 の各パッケージが依存する前提システムである。

| 系統 | 正本仕様 | 概要 |
| --- | --- | --- |
| A: JUCE Lifecycle 非決定性 | `ISR_JUCE_Lifecycle_Isolation.md` | LifecycleIsolationRuntime / Layer 0 |
| B: DSP Ownership Ambiguity | `ISR_DSPHandle_Runtime.md` | DSPHandleRuntime / DSP lifetime 単一窓口 |
| C: RT Execution Contamination | `ISR_RT_Execution_Frame.md` | RTExecutionFrame / RTCapabilityFirewall |
| D: World/Epoch Ambiguity（reference） | `ISR_World_Bridge_Runtime.md` | WorldBridgeUtility / EpochArbitrationHelper |

- 上記 4 ファイルが `doc/work/` に存在し lint-clean であること
- `ISR_10Layer_Implementation_Specification.md` に Layer 0 セクションが追加済みであること
- `plan5.md` に REV2 未閉塞4系統セクションが追加済みであること

### REV3.1 追加入口条件（安定性優先）

- `plan5.md` に REV3.1 Few-Authority セクションと文書優先順位セクションが存在すること
- authority は capability-first 優先で解釈し、
  runtime coordinator lifecycle を導入しない注記が
  `ISR_Retire_Authority_Graph.md` / `ISR_Runtime_State_Matrix.md` に反映済みであること
- 3-world 記述は参照モデル、2-world 運用優先の注記が `ISR_World_Bridge_Runtime.md` に反映済みであること

---

## 実装パッケージ

### P0-A0: LifecycleIsolationRuntime 骨格（Layer 0）→ REV2 系統A 準備

詳細仕様: `doc/work/ISR_JUCE_Lifecycle_Isolation.md`

目的:

- JUCE callback（prepareToPlay / getNextAudioBlock / releaseResources）を ISR-safe lifecycle event stream に変換する入口を確立する
- **Layer 0 として、Layer 1 以降の全 runtime object の lifecycle 依存関係を宣言的に管理する基盤を整備する**

最小実装:

- `LifecyclePhase` enum 導入（7値: Uninitialized / Preparing / Prepared / AudioRunning / Releasing / Released / Shutdown）
- `LifecycleToken` 構造体（epochId + entryTimestamp）
- `LifecycleIsolationRuntime::enterAudioCallback() / leaveAudioCallback()` stub
- `LifecycleBarrierRuntime::publishPreparedBarrier()` stub
- `CallbackExecutionEpoch` 構造体（lifecycleEpoch + sampleCursor）

受入条件:

- `getNextAudioBlock` が LifecycleToken を生成・消費する
- LIF-1: LifecyclePhase の逆行遷移が Abort で検出される
- LIF-3: `assertAudioRunning()` が AudioRunning 以外で Abort する
- LifecycleBarrier が shutdown/verification で観測可能な順序入口として宣言されている

**→次フェーズへの遷移**:

- Phase 1-2（ステップ0）で、本骨格を LifecycleIsolationRuntime として完全統合
- LifecycleBarrierRuntime → HBRuntimeCore / DebugRuntime 観測系の接続を実装

---

### P0-A1: RTExecutionFrame 分離（系統C）→ REV2 RT execution authority 準備

詳細仕様: `doc/work/ISR_RT_Execution_Frame.md`

目的:

- RT callback 内の全状態をスタックローカルな RTExecutionFrame に局所化する
- **RT thread からの authority leakage（publish/reclaim/alloc）を封止する基盤を整備する（GI-2 基盤）**

最小実装:

- `RTExecutionFrame` struct（activeDSP / fadingDSP / fade / scratch / sampleCursor / epoch）
- `RTAllocatorFirewall::markRTContext()` + Debug/CI build での RT context 内 heap allocation 検出
- `getNextAudioBlock` 入口への RTExecutionFrame stack-local 生成パターン適用
- `ScratchArena` の `prepareToPlay` 時 preallocate 化

受入条件:

- RT-1: RTExecutionFrame が全 RT callback でスタックローカルのみ生成される
- RT-7: Debug/CI build で RT callback 内 heap allocation を検出する
- publishAtomic の呼び出しパスから RT callback scope が除外されている
- GI-2 準拠: RT thread が reclaim authority を持たないことが構造的に確認できる

**→次フェーズへの遷移**:

- Phase 1-2（ステップ1）で、本分離を RTCapabilityFirewall として完全施行
- RTAllocatorFirewall を全 RT 経路に展開（Debug/CI 優先）

---

### P0-A2: DSPHandleRuntime 骨格（系統B）→ REV2 DSP ownership 単一窓口準備

詳細仕様: `doc/work/ISR_DSPHandle_Runtime.md`

目的:

- DSP instance lifetime の管理を DSPHandleRuntime に集約し、ownership ambiguity を閉塞する
- **GI-3「all DSP lifetime routed through DSPHandleRuntime」の基盤を整備する**

最小実装:

- `DSPHandle` 構造体（slot: uint32_t + generation: uint32_t）
- `DSPState` enum（7値: Constructing / Active / CrossfadingIn / CrossfadingOut / Retired / Quarantined / Reclaimed）
- `DSPHandleRuntime::create() / resolve()` + generation table stub
- stale handle（generation mismatch）の build別処理検出
  （CI=Abort / Debug=Assert / Release=Quarantine+Silence）

受入条件:

- DSP-1: 全 DSP 生成が `DSPHandleRuntime::create()` を経由する
- DSP-4: stale handle resolve が build別ポリシーで検出される（generation mismatch）
- GI-3 準拠: 直接 new/delete が DSP 生成経路から除外されている
- DSPState の不正遷移（DSP-2）が Abort で検出される

**→次フェーズへの遷移**:

- Phase 1-2（ステップ2）で、本骨格に CrossfadeAuthorityRuntime / DSPQuarantineRuntime を統合し、few-authority では DSPHandleRuntime / RetireRuntime へ読み替える
- 系統B の完全 DSP lifetime authority を確立

---

### P0-1: Closure Descriptor 骨格（R11）→ P9-1 ClosureBuilder 準備

詳細仕様: `doc/work/ISR_Runtime_Closure_Descriptor.md`

目的:

- publish直前に closure 検証ポイントを必須化する
- **Phase 1-2で ClosureBuilder（publish-time helper）へ読み替えるための基盤を整備**

最小実装:

- `PayloadClosureDescriptor`（最小ノード）を導入
- ノード属性: `kind`, `ownership`, `mutability`, `lifetime`, `hbDomain`, `authority`, `allocatorFamily`
- `validateClosureGraph(...)` の呼び出しを publish 経路に mandatory で挿入

受入条件:

- descriptor未登録 payload を publish しようとすると reject される
- external mutable dependency を検出して失敗させる

**→次フェーズへの遷移**:

- Phase 1-2 で、本 descriptor をベースに ClosureBuilder 参照設計へ昇格
- few-authority 実装では RuntimePublication の validation 側へ統合し、
  evidence export は Debug/CI layer 主責務として運用する

### P0-2: Payload Tier 骨格（R12）→ Layer 2: Payload Contract System 準備

詳細仕様: `doc/work/ISR_Payload_Tier_Model.md`

目的:

- payload boundary を静的に識別可能にする
- **Phase 1-2で PayloadTraits template compile-time contract へ昇格するための基盤を整備**

最小実装:

- `PayloadTier` を定義
  - `InlineImmutable`
  - `ImmutableShared`
  - `ExternalPinned`
  - `RTLocalOnly`
  - `Forbidden`
- publish 対象 family に tier 割当表を作成
- 検証ルール:
  - `Forbidden` は publish payload 禁止
  - `RTLocalOnly` は RuntimePublication の publish surface への混入を禁止

受入条件:

- tier未割当 object family が検出される
- Forbidden/RTLocalOnly混入で CI が fail する

**→次フェーズへの遷移**:

- Phase 1-2 で、本 tier model を PayloadTraits template + requires() へ統合
- compile-time rejection（mutable/RTLocal/forbidden tier の publish禁止）を強制
- 層2「Payload Contract System」の Closed criteria（`PayloadTraits template defined`, `publish() requires() enforced`等）を準備し、Release 常時の追加traceは要求しない

### P0-3: Sealed Object Mutation Detection（R13）→ Layer 3: SealedRuntime 準備

詳細仕様: `doc/work/ISR_Immutability_Enforcement_Spec.md`

目的:

- publish 後の mutation を早期検出する
- **Phase 1-2で SealedRuntime として runtime object化するための基盤を整備**

最小実装:

- `SealedObject` base class を導入（最小: seal() / sealed() / assertMutable()）
- publish 時に sealRecursively() を mandatory で呼び出す
- 全 mutator に assertMutable() を mandatory で埋め込む
- Release build: mutation attempt は exception ではなく abort/quarantine 必須

受入条件:

- sealed object への mutation attempt が detection される
- Release build でも silent ignore されない（abort/quarantine される）

**→次フェーズへの遷移**:

- Phase 1-2 で、本検出機構を SealedRuntime 参照設計へ統合し、few-authority 実装では RuntimePublication 側へ収束させる
- 層3「SealedRuntime」の Closed criteria（`seal()`, `sealed()`, `assertMutable()`, `mutation_fault_trace.json` export等）を準備

### P0-4: HB Graph Instrumentation + 3分割設計（R16）→ Layer 4: HBRuntimeCore / HBTraceRuntime / HBVerifierRuntime 準備

詳細仕様: `doc/work/ISR_HB_Graph_Specification.md`, `doc/work/ISR_Minimal_HB_Failure_Model.md`

目的:

- happens-before order を runtime で最小 barrier + trace/verify 分離として扱う基盤を整備
- **Phase 1-2 で HBRuntime 3分割（HBRuntimeCore / HBTraceRuntime / HBVerifierRuntime）を Release最小 / Debug部分 / CI検証 の責務分離で扱う基盤を整備**
- `emitEvent() / verify()` 混在を解消し、latency-critical / non-RT / CI 用途を分離する設計を準備する

最小実装:

- `HBNode`, `HBEdge`, `HBTrace` 構造体を定義
- publish/observe/retire/reclaim/shutdown barrier/epoch settle に trace 埋め込み
- runtime trace を sampled/event-bounded で evidence 化する枠組み
- `verifyHBGraph()` を用いた post-shutdown verification
- Release 経路では HBRuntimeCore の最小 barrier を優先し、trace/verify は Debug/CI 側で運用する

受入条件:

- HB-1～HB-4 checks（reclaim-before-grace禁止等）が Debug/CI または shutdown verifier で検証可能である
- `hb_graph_trace.json` / `hb_violation_report.json` の schema/命名が CI評価可能である
- HBRuntimeCore（publishBarrier / observeBarrier / retireBarrier）・HBTraceRuntime（sampled emit / snapshot）・HBVerifierRuntime（verify / CI）への分割設計が仕様として確認済み

**→次フェーズへの遷移**:

- Phase 1-2 で、本 trace 機構を HBRuntime 3分割へ統合
- `HBRuntimeCore` を RT/NonRT 共通 latency-critical barrier として実装
- `HBTraceRuntime` を Debug/CI 優先の sampled emit + snapshot として実装
- `HBVerifierRuntime` を CI / shutdown time verification として実装
- 層4「Executable HB Runtime」の Closed criteria（minimal barriers + sampled trace + CI verifier）を準備し、Release 常時full traceは要求しない

### P0-5: RetireIntent Bridge（R14）→ Layer 5: Retire Runtime 準備

詳細仕様: `doc/work/ISR_Deferred_Retire_Intent_Bridge.md`

目的:

- RT detect と retire authority 実行を明示分離する
- **Phase 1-2で 5-lane RetireRuntime（RTIntent/Coordination/Epoch/Reclaim/Quarantine）へ昇格するための基盤を整備**

最小実装:

- `RetireIntent` を導入（RTは emission のみ）
- NonRT coordinator が dequeue 後に authority enqueue 実行
- RT で retire/reclaim/delete を行わない監視を追加

受入条件:

- RT経路から direct enqueue が禁止される
- RT detect -> NonRT dequeue -> authority enqueue を再現試験で確認

**→次フェーズへの遷移**:

- Phase 1-2 で、本 bridge を RetireRuntime へ統合
- 5 lanes による lifecycle separation（RTIntentLane→CoordinationLane→EpochLane→ReclaimLane→QuarantineLane）を実装
- 層5「Retire Runtime」の Closed criteria（`emitIntent()`, `enqueueRetire()`, `RR-1/RR-2/RR-3` invariants, `retire_timeline.json` export等）を準備し、evidence export は NonRT/CI 優先とする

### P0-6: Shutdown FSM 最小整流（R15）→ Layer 6: Shutdown Runtime 準備

詳細仕様: `doc/work/ISR_Shutdown_State_Machine.md`

目的:

- 現行 shutdown phase と spec phase の乖離を最小で縮める
- **Phase 1-2で barrier-backed FSM + mandatory HB chain へ昇格するための基盤を整備**
- shutdown では proof completeness より finite deterministic completion を優先する
  （callback count / active crossfade / pending retire / observer count のゼロ化を完了条件とする）

最小実装:

- phase enum を以下に合わせる（互換マッピング可）
  - `Running`
  - `AudioStopped`
  - `ObserverDrained`
  - `RetireClosed`
  - `EpochSettled`
  - `ReclaimComplete`
  - `ShutdownComplete`
- 禁止遷移（逆行・飛び越し）を reject
- 各遷移に barrier() 呼び出しを埋め込む
- late callback / post-stop enqueue の検知フックを追加

受入条件:

- state逆行/飛び越し遷移が fail する
- post-stop enqueue を shutdown verifier が検出する
- HB chain（各遷移間の happens-before）が Debug/CI または shutdown 時に観測可能な状態

**→次フェーズへの遷移**:

- Phase 1-2 で、本 FSM を ShutdownRuntime へ統合
- barrier-backed state transition + mandatory HB chain（AudioStopped→ObserverDrained→...→ReclaimComplete）を強制
- 層6「Shutdown Runtime」の Closed criteria（`transition()`, `barrier()`, `verifyShutdownTrace()`, `shutdown_trace.json` export等）を準備し、shutdown trace は Release常時full採取を要求しない

### P0-7: CI 先行パイプライン（R18）→ Layer 7-10: Evidence Export Infrastructure 準備

詳細仕様: `doc/work/ISR_Verification_Pipeline.md`, `doc/work/ISR_Proof_Artifact_Schema_Registry.md`, `doc/work/ISR_Runtime_Reduction_Strategy.md`

目的:

- フェーズ0成果を merge blocker 化する
- **Phase 1-2/3（P9/P10 Evidence Export / CI Validation）のための artifact naming/schema 基盤を整備**

最小実装（先行3段 + evidence contract）:

1. Atomic Dot-Call Scan（既存活用）
2. Ownership Closure Validator（新規）→ Layer 1 ClosureBuilder 検証
3. Shutdown Sequencing Verifier（新規）→ Layer 6 Shutdown Runtime 検証
4. Evidence artifact naming/schema contract 固定
  canonical artifact 名（closure_graph.json, mutation_fault_trace.json, hb_graph_trace.json, retire_timeline.json, shutdown_trace.json）を固定する。
  artifact missing / parse error / schema mismatch は CI fail（non-negotiable）とする。
  注記: runtime correctness は artifact existence に依存させず、runtime invariants で成立させる。

Phase 1-2 追加分:

- SealedObject Mutation Detector（Layer 3 SealedRuntime対応）
- HB Runtime Trace Validator（Layer 4 HBRuntime対応）
- Retire Timeline Validator（Layer 5 Retire Runtime対応）

Phase 3 追加分（P10 実装時）:

- Evidence Export Hooks（runtime exposes evidence）
- Runtime Budget Monitor
- Safe Failure Action Validator
- Introspection Console Output Validator

注記（層8-10運用）:

- Budget/validator/introspection は Debug/CI layer を主責務とし、runtime core 常駐責務にしない。

受入条件:

- 先行3段が CI 統合され、失敗時に PR merge を停止する
- artifact policy（Release=optional / Debug=recommended / CI=mandatory）が明文化され、命名が canonical で統一される
- schema version が固定される
- runtime が evidence を公開し、CI が validation を担う責務分離が成立している

**→次フェーズへの遷移**:

- Phase 1-2 で、各層（1-6）が export する 5 core artifacts を CI evaluator として統合
- Phase 3 で、各層（7-10）が export する completion artifacts を統合
- 層7-10「Evidence/Budget/Safe Failure Handling/Introspection」の Closed criteria（evidence export / CI validator 等）を準備

---

## 実行順（推奨）

1. P0-A0 LifecycleIsolationRuntime 骨格 （Layer 0 準備 / 系統A）
2. P0-A1 RTExecutionFrame 分離 （系統C 準備）
3. P0-A2 DSPHandleRuntime 骨格 （系統B 準備）
4. P0-1 Closure Descriptor （Layer 1 準備）
5. P0-2 Payload Tier （Layer 2 準備）
6. P0-3 Sealed Object Mutation Detection （Layer 3 準備）
7. P0-4 HB Graph Instrumentation + 3分割 （Layer 4 準備）
8. P0-5 RetireIntent Bridge （Layer 5 準備）
9. P0-6 Shutdown FSM （Layer 6 準備）
10. P0-7 CI Pipeline （Layer 7-10 準備）

理由:

- 先に「lifecycle と execution authority の基盤」を確立（P0-A0/A1/A2 → Layer 0 + REV2 系統A/B/C）
  - P0-A0: JUCE callback の lifecycle isolation なしに Layer 1-6 の lifetime guarantee は成立しない
  - P0-A1: RT execution frame の分離なしに RT contamination を検出できない
  - P0-A2: DSP ownership 単一窓口なしに closure / retire が不完全なままになる
- 次に「何を publish してよいか」を定義（P0-1/P0-2 → Layer 1/2）
- 次に「publish後は mutations から守る」（P0-3 → Layer 3）
- 次に「concurrent orders を Debug/CI 中心の trace / 3分割設計で準備」（P0-4 → Layer 4）
- 次に「いつ/どこで retire するか」を分離（P0-5 → Layer 5）
- 最後に「終了順序を固定」（P0-6 → Layer 6）し、「CIで封止」（P0-7 → Layer 7-10）

修正版 13ステップ実装順序（plan5.md REV2 セクション）との対応:

| plan5.md ステップ | 本 Phase 0 パッケージ |
| --- | --- |
| 0: LifecycleIsolationRuntime | P0-A0 |
| 1: RTLocalState separation | P0-A1 |
| 2: DSPHandleRuntime | P0-A2 |
| 3: ClosureBuilder | P0-1 |
| 4: SealedRuntime | P0-3 |
| 5: RetireRuntime | P0-5 |
| 6: HBRuntimeCore + 3分割 | P0-4 |
| 7: ShutdownConvergenceRuntime | P0-6 |
| 8-11: Evidence/Budget/Safe Failure Handling/Introspection | P0-7 |
| 12: WorldBridgeUtility + EpochArbitrationHelper | 非スコープ（Phase 1-2+） |

---

## 完了判定（Exit Gate）→ Phase 1-2 開始条件

以下をすべて満たしたら最小フェーズ0完了・Phase 1-2（P9 Runtime Object Model：10層参照設計の few-authority 読み替え）の実装を開始可能。

### REV2 Execution Authority Readiness

REV2 execution authority architecture（steps 0-2）が Phase 1-2 で安全に実装できる状態であること:

- **Layer 0 (LifecycleIsolationRuntime)**: `enterAudioCallback() / leaveAudioCallback()` が shutdown/verification で観測可能な順序入口として機能し、LifecyclePhase の逆行遷移 Abort が動作する
- **系統C (RTExecutionFrame)**: RTExecutionFrame が全 RT callback でスタックローカル生成されており、Debug/CI build で RT context 内 heap allocation を検出できる
- **系統B (DSPHandleRuntime)**: generation table による stale handle 検出が build別ポリシー（CI=Abort / Debug=Assert / Release=Quarantine+Silence）で動作し、全 DSP 生成が `DSPHandleRuntime::create()` を経由している
- **GI-2 / GI-3 基盤**: RT thread が reclaim authority を持たない構造と、全 DSP lifetime 単一窓口の構造が Phase 0 レベルで準備済み
- 4 REV2 正本仕様ファイル（`ISR_JUCE_Lifecycle_Isolation.md` / `ISR_RT_Execution_Frame.md` / `ISR_DSPHandle_Runtime.md` / `ISR_World_Bridge_Runtime.md`）が lint-clean で存在する

### REV3.1 Operational Stability Readiness

few-authority 運用での Phase 1-2 進行条件として、以下が満たされること:

- ISR-1..ISR-6 が設計上の優先ゲートとして明示されている
- Release/Debug/CI の責務境界が固定されている
  - Release: barrier 最小セット（publish/retire/shutdown）
  - Debug: trace 部分有効
  - CI: verify/simulation フル有効
- `PublicationWorld / ExecutionWorld` 2-world 運用を前提に、
  full federation semantics を必須前提にしないことが明文化されている
- authority 制約が capability-first で解釈され、runtime coordinator lifecycle を導入しないこと
- runtime core（5 subsystem）と Lifecycle host adapter、Optional Debug Layer（DebugRuntime）の境界が明文化されていること

### REV3.2 Operational Hardening Readiness

- 10-layer は conceptual reference、実装は few-authority 読み替えを優先する
- Release で full artifact / full verify を常時要求しない
- `WorldBridgeUtility` は helper、epoch arbitration は RetireRuntime 内部責務統合優先で解釈される
- stale handle の build別ポリシーが関連正本と一致している
- Debug/verification 機能（HB trace / ownership audit / artifact emitter / verifier）が runtime core から分離されている
- Runtime Core への proof/validator/budget/introspection の新規常駐導入を行わない
- Runtime Core 常駐責務が `publishBarrier` / `retireBarrier` / `shutdownBarrier` / `quarantine` に限定されている
- `ClosureBuilder`（統合先: RuntimePublication）の責務が
  `createNode() / connect() / seal() / validateAcyclic() / freeze()` 相当に制限されている
- Introspection は最小状態表示（snapshot / retire queue / shutdown state）を上限とし、
  HB full graph export を CI 限定で運用する
- contributor survivability を優先し、新規 runtime object 追加は
  authority 減少または interaction complexity 純減の定量根拠がある場合のみ許可する
- `WorldBridgeUtility` は Debug helper 程度、`EpochArbitrationHelper` は helper-level utility を超えて昇格させない

### Specification & Implementation readiness

- R11-R18 が `Spec-Fixed` から「実装+検証証跡あり」へ進捗
- `ISR_10Layer_Implementation_Specification.md` が 10層参照設計を few-authority 実装へ読み替える実装班向けガイドとして確立
- 各層（Layer 1-6）の Closed criteria が定義され、Phase 0 がそれらの準備状態を達成

### CI & Artifact governance

- 先行3段 CI（Atomic Dot-Call + Closure Validator + Shutdown Verifier）が main で安定稼働
- 5 core artifacts（closure_graph/mutation_fault/hb_graph/retire_timeline/shutdown_trace）の naming が canonical で統一
- artifact missing / parse error / schema mismatch が CI fail（merge blocker）として機能

### Failure injection & verification

- publish/retire/shutdown/mutation/HB の違反注入ケースで fail-fast が確認済み
- runtime が artifact/evidence を export し、CI が validation/evaluation を担う責務分離が成立

### Phase 1-2 実装基盤の準備完了

各層の Closed criteria に向けた準備が以下で完了:

- **Layer 0（LifecycleIsolationRuntime）**: JUCE callback → ISR lifecycle event stream 変換が統合可能な状態
- **REV2 系統C（RTExecutionFrame）**: RTCapabilityFirewall が全 RT callback に配置可能な状態
- **REV2 系統B（DSPHandleRuntime）**: CrossfadeAuthorityRuntime が few-authority 実装側へ統合可能な状態
- **Layer 1（ClosureBuilder）**: descriptor + validator が RuntimePublication / ClosureBuilder 参照設計へ統合可能な状態
- **Layer 2（Payload Contract System）**: tier model が compile-time contract へ統合可能な状態
- **Layer 3（SealedRuntime）**: mutation detection が seal enforcement / RuntimePublication 側へ統合可能な状態
- **Layer 4（Executable HB Runtime）**: HB trace が Debug/CI 中心の executable verifier へ統合可能な状態
- **Layer 5（Retire Runtime）**: intent bridge が 5-lane lifecycle へ統合可能な状態
- **Layer 6（Shutdown Runtime）**: FSM が barrier-backed state machine へ統合可能な状態
- **Layer 7-10（Evidence/Budget/Safe Failure Handling/Introspection）**: evidence export / CI validation infrastructure へ統合可能な状態

### Documentation & Governance

- `plan5.md` が 10層参照ロードマップ + governance hub として確立（REV2 未閉塞4系統セクション含む）
- 本書（Phase 0）が「Phase 1-2 実装班向けの few-authority 準備ガイド」として position されている
- REV2 4正本ファイルが存在し lint-clean:
  - `doc/work/ISR_JUCE_Lifecycle_Isolation.md`（系統A / Layer 0）
  - `doc/work/ISR_RT_Execution_Frame.md`（系統C）
  - `doc/work/ISR_DSPHandle_Runtime.md`（系統B）
  - `doc/work/ISR_World_Bridge_Runtime.md`（系統D）
- `ISR_10Layer_Implementation_Specification.md` に Layer 0 セクションが追加済み
- REV3.1/REV3.2 関連注記（few-authority / 2-world / capability-first / evidence-CI 分離）が関連正本に反映済み

---

## リスクと回避

- リスク: 既存 phase 名変更で運用ログが読みにくくなる
  - 回避: 互換表示名（旧->新）をログに併記
- リスク: ルール導入直後に false positive が増える
  - 回避: 1週間は fail閾値を段階適用（ただし merge blocker化は維持）
- リスク: 実装先行で文書が遅れる
  - 回避: PR テンプレに「R番号・検証証跡」を必須欄として追加

---

## 成果物

- 本書: `doc/work/ISR_Minimal_Phase0_Recommended.md`
- 参照更新: `doc/work/plan5.md`（ハブリンク追加）
- トラッキング: `doc/work/ISR_Completeness_Risk_Backlog.md`（進捗更新）
