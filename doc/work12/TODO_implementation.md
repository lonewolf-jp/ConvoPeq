# TODO Implementation — Practical Stable ISR Bridge Runtime

作成日: 2026-05-31
適用スコープ（Production Runtime Tree）:

- `src/audioengine/**`
- `src/convolver/**`
- `src/eqprocessor/**`
- `src/core/**`

参照設計図書:

- `doc/work12/practical_stable_isr_bridge_runtime_masterplan_detailed_design_and_findings_2026-05-31.md`
- `doc/work12/practical_stable_isr_bridge_runtime_ai_governance_v1_1_draft_2026-05-31.md`
- `doc/work12/runtime-coordinator-topology-decision.md`
- `doc/work12/audit3_self_contained_transition_generation_2026-05-31.md`
- `doc/work12/authority_inventory.md`
- `doc/work12/authority_topology_validation_and_blockers_2026-05-31.md`
- `doc/work12/phase_submission_template_evidence_authority_callgraph_c1-c15.md`

---

## 0) 共通運用（全フェーズ共通）

- [x] フェーズ開始前に探索証跡（grep / Serena / CodeGraph）を採取・保存
- [x] フェーズ越境なし（Rule-36）を確認
- [x] フェーズ出口条件を機械検証で判定（Rule-12）
- [x] フェーズ提出テンプレートを更新（探索証跡 / Authority差分 / CallGraph差分 / C1〜C15）
- [x] 変更ごとに build/test 実行、失敗時は自己修復して再実行

---

## Phase0 — Authority Inventory + Architecture Inventory

### Phase0 実装タスク

- [x] Authorityキーワード全域再スキャン（commit/publish/retire/build/generation/snapshot/transition）
- [x] Architecture重点シンボル再スキャン（SafeStateSwapper/BuildSnapshot/PendingParams/PreparedIRState/TransitionState/GlobalSnapshot）
- [x] `authority_inventory.md` の Before/After フォーマットへ更新可能なデータ構造を整理
- [x] `Authority Source / Mirror Source / Legacy Source` の三分類表を最新版に更新

### Phase0 テスト・検証タスク

- [x] grep結果とCodeGraph結果の不整合ゼロを確認
- [x] 主要シンボル（publishState/prepareCommit/executeCommit/commitNewDSP）の caller/callee を証跡化
- [x] フェーズ提出テンプレート（Phase0分）を記入

---

## Phase0.5 — Coordinator Topology Decision（必須ゲート）

### Phase0.5 実装タスク

- [x] `convo::RuntimePublicationCoordinator` と `convo::isr::RuntimePublicationCoordinator` の責務境界を最終確定
- [x] publish権威 / drain・shutdown権威 / rollback境界を文書で固定
- [x] `runtime-coordinator-topology-decision.md` に To-Be責務境界を反映
- [x] Phase1開始条件（決定済み）を明示

### Phase0.5 テスト・検証タスク

- [x] `runtimePublicationBridge_` が RuntimeStore 実行権威を持たないことを再確認
- [x] `precheckPublish/commit/retire/isFullyDrained` の利用箇所を証跡化
- [x] フェーズ提出テンプレート（Phase0.5分）を記入

---

## Phase1 — Publication Authority Collapse

### Phase1 実装タスク

- [x] `publishState()` 呼び出し点を単一路へ集約する設計パッチ作成
- [x] `prepareCommit/executeCommit/commitNewDSP` の到達経路を遮断（段階的）
- [x] 旧経路を non-reachable 化し、新経路の単一路性を維持
- [x] `commitRuntimePublication/retireRuntimePublication` の役割を topology決定に一致させる

### Phase1 テスト・検証タスク

- [x] Exit-A: `publishState()` callsite = 1（Production Runtime Tree）
- [x] Exit-B: `prepareCommit/executeCommit/commitNewDSP` 到達不能証明（CodeGraph + Serena + grep）
- [x] ビルド通過（Debug/Releaseの少なくとも設計規約で定義された主要構成）
- [x] フェーズ提出テンプレート（Phase1分）を記入

---

## Phase8-A — BuildSnapshot Dependency Collapse

### Phase8-A 実装タスク

- [x] `RuntimeBuilder::build(..., ConvolverProcessor::BuildSnapshot)` 依存を排除する置換入力を設計
- [x] RuntimeBuilder API を Convolver非依存へ更新
- [x] 呼び出し側を新APIへ移行
- [x] 互換ブリッジが必要な場合は Rule-42 に従い期限を明記（該当なし / N/A）

### Phase8-A テスト・検証タスク

- [x] Exit-1: RuntimeBuilder が `ConvolverProcessor` 型を参照しない
- [x] Exit-2: `RuntimeBuilder.h/.cpp` から `BuildSnapshot` 参照消滅
- [x] ビルド/ユニットテスト通過
- [x] フェーズ提出テンプレート（Phase8-A分）を記入

---

## Phase8-B — Convolver Runtime Integration

### Phase8-B 実装タスク

- [x] `PendingParams` を RuntimeState統合モデルへ移行
- [x] `PreparedIRState` を RuntimeState統合モデルへ移行
- [x] `SafeStateSwapper` 依存を除去
- [x] Convolver の Runtime外 SoT を撤去

### Phase8-B テスト・検証タスク

- [x] `SafeStateSwapper` = 0（Production Runtime Tree）
- [x] `PendingParams` = 0（Production Runtime Tree）
- [x] `PreparedIRState` = 0（Production Runtime Tree）
- [x] Convolver関連回帰テスト/ビルド通過
- [x] フェーズ提出テンプレート（Phase8-B分）を記入

---

## Phase2 — RuntimeState Self-contained

### Phase2 実装タスク

- [x] RuntimeState単体で実行可能な必要情報を閉包化
- [x] RuntimeState外の mutable SoT をゼロ化
- [x] RuntimeState追加フィールドを Authority/Projection/Diagnostic へ分類

### Phase2 テスト・検証タスク

- [x] `processBlock()` の実行に RuntimeState外 SoT が不要であることを証明
- [x] 単体/統合ビルド通過
- [x] フェーズ提出テンプレート（Phase2分）を記入

---

## Phase3 — AudioEngine Non-Authority

### Phase3 実装タスク

- [x] AudioEngine の Authority主語（commit/publish/retire/build/activate）を撤去
- [x] AudioEngine責務を Observe/Process/Measure に限定

### Phase3 テスト・検証タスク

- [x] AudioEngine に authority操作シンボルが残っていないことを確認
- [x] 回帰ビルド/テスト通過
- [x] フェーズ提出テンプレート（Phase3分）を記入

---

## Phase4 — Snapshot Classification

### Phase4 実装タスク

- [x] Snapshot項目を Authority/Projection で完全分類
- [x] snapshot由来の実行分岐を排除

### Phase4 テスト・検証タスク

- [x] `snapshot.(generation|version|active)` 実行分岐ゼロ
- [x] 回帰ビルド/テスト通過
- [x] フェーズ提出テンプレート（Phase4分）を記入

---

## Phase5-Gate — DSP Selection State Machine 設計承認

### Phase5-Gate 設計タスク

- [x] 状態図（Stable/Entering/Retiring）を作成
- [x] AudioThread 分岐置換表を作成
- [x] 互換期間フェイルセーフを定義
- [x] 設計承認記録を残す

### Phase5-Gate 検証タスク

- [x] Gate成果物3点が揃っている
- [x] フェーズ提出テンプレート（Phase5-Gate分）を記入

---

## Phase5 — DSP Selection Migration

### Phase5 実装タスク

- [x] DSP選択責務を TransitionState から新state machineへ移設
- [x] AudioThread分岐を新設計へ置換
- [x] `transition.active` 実行依存を撤去

### Phase5 テスト・検証タスク

- [x] `transition.active` 参照が実行分岐に使われないことを確認
- [x] クロスフェード/遷移回帰テスト通過
- [x] フェーズ提出テンプレート（Phase5分）を記入

---

## Phase6 — Generation Authority Collapse

### Phase6 実装タスク

- [x] generation権威源を一本化
- [x] 非権威 generation は diagnostic mirror 化

### Phase6 テスト・検証タスク

- [x] Generation authority = 1 を証跡で確認
- [x] 回帰ビルド/テスト通過
- [x] フェーズ提出テンプレート（Phase6分）を記入

---

## Phase7 — Retire Governance Singularity

### Phase7 実装タスク

- [x] RetireManager を新規設計・実装
- [x] `audioThreadRetireOverflowPtr` を統合
- [x] `deferredDeleteFallbackQueue`（AudioEngine）を統合
- [x] `deferredDeleteFallbackQueue`（EQProcessor）を統合
- [x] `DeferredFreeThread` / `ISRRetireRuntimeEx` との境界を統合設計へ移行

### Phase7 テスト・検証タスク

- [x] Retire authority = 1 を証跡で確認
- [x] C15（EQ fallback queue = 0）達成
- [x] 回帰ビルド/テスト通過
- [x] フェーズ提出テンプレート（Phase7分）を記入

---

## Phase9 — Legacy Semantic Purge

### Phase9 実装タスク

- [x] Legacy/Mutable/Temporary の残存を整理・撤去
- [x] 期限付き互換ブリッジの削除完了

### Phase9 テスト・検証タスク

- [x] 旧語彙/旧経路の残存ゼロを確認
- [x] 回帰ビルド/テスト通過
- [x] フェーズ提出テンプレート（Phase9分）を記入

---

## Phase10 — Contract Enforcement

### Phase10 実装タスク

- [x] 統治規約に対応するCI検証（Blocker級）を実装
- [x] Warning級規約のCI導入期限を明記・反映
- [x] C1〜C15 自動判定の最小検査群を整備

### Phase10 テスト・検証タスク

- [x] CIルールが規約と一致
- [x] C1〜C15 の判定表を最終更新
- [x] 統合テストを含む最終グリーン確認
- [x] フェーズ提出テンプレート（Phase10分）を記入

---

## 全体完了条件（最終）

- [x] C1〜C15 がすべて Pass
- [x] フェーズ提出テンプレートが全フェーズ分揃っている
- [x] Production Runtime Tree の最終 grep/Serena/CodeGraph 再検証完了
- [x] 統合テストを含む全テスト Green
- [x] 最終サマリ作成

---

## 追加独立再監査（2026-06-01, 連続反映）

### 確定漏れ → 解消済み

- [x] Leak-01: `AudioEngine::publish*` 命名/責務漏れを解消
  - `publishRcuEpoch` → `snapshotRcuEpoch`
  - `publishRetireEpoch` → `markRetireEpoch`
  - `publishCoeffs` → `storeLearnedCoeffs`
  - `publishCoeffsToBank` → `storeLearnedCoeffsToBank`
- [x] Leak-02: C4検証の fail-open を解消（fail-closed化）
  - `.github/scripts/isr-verify-c1-c15-minimal.ps1` に
    `AudioEngine::commit/publish/retire/build/activate*` 禁止判定を追加

### 新規確定漏れ（今回発掘）

- [x] Leak-03: C6 実質漏れ（同義語経路）を解消
  - AudioThread 実行分岐を `execution.transitionActive` 依存から `topology.hasFadingRuntime` 判定へ置換
  - C6 最小監査を fail-closed 化（`transition.active` / `execution.transitionActive` の両方を監査）
  - 主要証跡:
    - `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`
    - `src/audioengine/AudioEngine.Processing.BlockDouble.cpp`
    - `src/audioengine/AudioEngine.Processing.Snapshot.cpp`
    - `src/audioengine/AudioEngine.Timer.cpp`
    - `.github/scripts/isr-verify-c1-c15-minimal.ps1`
    - `evidence/c1_c15_minimal_report.json`

- [x] Leak-04: NonRT runtime admission の同義語経路を統一
  - `src/audioengine/AudioEngine.Commit.cpp` の該当判定を `world.topology.hasFadingRuntime` 主軸へ統一
  - 等価性ガード（`execution.transitionActive == topology.hasFadingRuntime`）を precheck/retire path に導入
  - 検証: `Build_CMakeTools` pass / `RunCtest_CMakeTools` pass / `isr-verify-c1-c15-minimal.ps1` pass

- [x] Leak-05: EQ側 fallback queue 実体を統合撤去
  - `EQProcessor` から `DeferredRetireFallbackQueue` 実体を削除（旧/新 alias いずれも 0）
  - retire enqueue は `EpochDomain` bounded retry（reclaim + advance）へ統一
  - C15監査を fail-closed 化（`deferredDeleteFallbackQueue|deferredRetireFallbackQueue_` 両対応）
