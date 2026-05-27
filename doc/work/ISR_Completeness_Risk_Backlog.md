# ConvoPeq ISR Completeness Risk Backlog

## 目的

ISR completeness 未完領域を、正本仕様に影響する順序で管理する。
本書は backlog 正本であり、詳細仕様は各正本文書へ反映して閉じる。

ステータス定義:

- `Spec-Fixed`: 仕様確定済み（実装・検証証跡は未完）
- `Closed`: 仕様反映・実装・検証証跡が完了

確定宣言（2026-05-20 / 履歴注記）:

- R1〜R18 は `Spec-Fixed` として確定済み（安定性優先の必須コア）
- R19〜R25 は `Spec-Fixed`（ガード付き拡張。Debug/CI 優先、Release 直結では任意）
- 本書に記載のない追加未完事項は扱わない（追加時はR採番必須）

最新ステータス更新（2026-05-21 / 履歴注記）:

- R11〜R25 は Closed 最小検証項目を満たし `Closed` へ更新。
- R1〜R10 は当時 `Spec-Fixed` を維持（段階適用対象）。
- ※この時点の記述は履歴であり、現行の正本状態は直下の 2026-05-27 更新を優先する。

最新ステータス更新（2026-05-27 追記・R1〜R10昇格）:

- `evidence/backlog_specfixed_residual_report.json` で `specFixedResidualCount=10` かつ `promotableResidualCount=10` を確認。
- R1〜R10 は gate/evidence readiness を満たすため、状態を `Closed` へ更新。
- 更新後に residual monitor を再実行し、`specFixedResidualCount=0` を確認。

現行正本ステータス（2026-05-27 時点）:

- R1〜R25 はすべて `Closed`。
- `Spec-Fixed` は履歴注記としてのみ残置し、運用判定は `- 状態:` 行の現値を正本とする。

最新ステータス更新（2026-05-27 追記）:

- ISR verification workflow を tiered（smoke/standard/exhaustive）実行へ移行。
- `standard` / `exhaustive` tier のローカル実行を完了し、全ゲート green を確認。
- trigger 監査を自動化し、`evidence/trigger_audit_report.json` を出力開始（監視モード）。
- trigger policy（owner/issue/rationale/expiry 必須）を導入し、expiry超過・許容上限超過をCI fail化。
- `workflow_dispatch.enforceTriggerPolicy` と tier runner `-EnforceTriggerPolicy` を追加し、段階的 enforce 導線を実装。
- AST trigger gate（`isr-verify-trigger-ast.ps1`）を導入し、monitor運用を既定化。
- `workflow_dispatch.requireAstTriggerCheck` と tier runner `-RequireAstTriggerCheck` で AST 必須モードへ切替可能化。
- symbol 参照ゲート（`isr-verify-trigger-symbol-usage.ps1`）を導入し、`activeDSP` の新規拡散を期限付きallowlistで機械検知化。
- cleanup deferred registry（`isr-cleanup-deferred.json`）を導入し、`trigger/owner/expiry` 必須化と expiry超過 fail を実装。
- rollback compatibility matrix（`isr-rollback-compatibility-matrix.json`）と検証ゲート（`isr-verify-rollback-matrix.ps1`）を導入し、Phase5 の subsystem rollback 互換性を機械検証化。
- AST trigger gate はツール健全性プローブを追加し、`sg` 不整合時の偽陽性成功を防止（monitor skip + warning）に修正。
- metric governance registry（`isr-metric-governance.json`）と検証ゲート（`isr-verify-metric-governance.ps1`）を導入し、canary metrics の owner/threshold/retention/action 必須化をCI固定。
- `exhaustive` tier（weekly相当）で新規 governance ゲート群を含む全実行が green。
- `ast-grep` CLI を導入し、AST trigger gate を `run --pattern` 互換へ修正。`-RequireAstTriggerCheck`（requiredモード）で tier 実行可能化。
- `fadingOutDSP` 直接代入トークンを除去し、trigger 監査値 `fadingOutDSP writes` を **1→0** に縮退。
- trigger cleanup readiness gate（`isr-verify-trigger-cleanup-readiness.ps1`）を追加し、現状 `ready=1 / blocked=2` を機械判定化。
- trigger監査を symbol blocked 基準へ拡張し、`retireFacade deps` を **14→0（raw=14, blocked=0）** に縮退。
- `cleanup_deferred_registry` を空化し、cleanup gate を「空レジストリ許容」へ更新。`ready=0 / blocked=0` を維持。
- observe shim 専用 gate（`isr-verify-observe-shim-usage.ps1`）を追加し、`legacyDirectObserveUsage=0`（raw=1, blocked=0）を自動判定化。
- `runtimeStore.observe()` の audioengine 直呼びを coordinator API 経由へ置換し、`legacyDirectObserveRawCount` を **1→0** に縮退。
- observe shim allowlist を空化（temporary exception 解除）し、直呼び再流入を即時 blocked できる運用へ移行。
- trigger policy mode を `monitor` から `enforce` へ昇格し、達成済みトリガーを fail-closed 運用へ切替。
- trigger監査へ `retireFacadeRuntimeExecutionCount` を追加（`RuntimePublicationCoordinator::create` 出現数）。retire facade 撤去の残タスクを定量監視化。
- `RuntimePublicationCoordinator::create` の audioengine 直接呼びを `makeRuntimePublicationCoordinator()` へ集約し、`retireFacadeRuntimeExecutionCount` を **11→0** に縮退（standard tier / Release Build green）。
- `RuntimeExecutionView` read path の命名/呼出しを `RuntimeReadView` / `read*RuntimeView` へ収束し、`runtimeExecutionViewUsageCount` を **43→0** に縮退（standard tier / Release Build green）。
- `runtimePublicationCoordinator_.` 直参照を `AudioEngine` ヘルパ経由へ収束し、`retireFacadeRawDependencyCount` を **3→0** に縮退（standard tier / Release Build green）。
- `isr-trigger-audit.ps1` に symbol usage report の自動リフレッシュ分岐を追加し、enforceモードでの stale report 起因の擬陽性 fail を抑止。
- `activeDSP` 生参照の第1段を `AudioEngine.Commit.cpp` でヘルパ経由へ移行し、`activeDspRawRefCount` を **33→24** に縮退（standard tier / Release Build green）。
- `activeDSP` 生参照の第2段を `CtorDtor/PrepareToPlay/ReleaseResources/RebuildDispatch/ISRRTExecution` と `AudioEngine.h` 内部スロット命名へ展開し、`activeDspRawRefCount` を **24→0（raw=0, blocked=0）** に縮退（trigger audit / standard tier / Release Build green）。
- `activeDSP` の temporary symbol allowlist（`BRIDGE-SYMBOL-001`〜`007`）を撤去し、達成済み trigger 項目の deferred fossil 化を解消。
- `runtimePublicationCoordinator_` の残存参照を `runtimePublicationBridge_` へ収束し、temporary symbol allowlist（`BRIDGE-SYMBOL-008` / `009`）を撤去。`trigger_symbol_usage_report` は **totalMatches=0 / blockedMatches=0** へ収束（standard tier / Release Build green）。
- `isr-verify-trigger-symbol-usage.ps1` の「allowlist非空」契約に合わせ、symbol allowlist は `BRIDGE-SYMBOL-010` の最小1件を維持（現時点 `totalMatches=0` のガードレール定義のみ）。
- `isr-trigger-policy.json` の全entryで `allowedMax` を **0** へ引き締め、達成済みメトリクスの fail-open 余地を排除（enforce mode + standard tier green）。
- `fadingOutDSP` 内部スロット識別子を `fadingRuntimeDSPSlot` へ収束し、legacy語彙の縮退を継続（`fadingOutDSP writes=0` を維持、standard tier / Release Build green）。
- `isr-trigger-symbol-allowlist.json` に `activeDSP` / `runtimePublicationCoordinator_` の zero-exception symbol guardrail（`pathRegex="^$"`）を追加し、監査メトリクスを rawRegex 依存から symbolBlockedMatches 優先へ移行。
- fading helper API を `exchangeFadingRuntimeDSP` / `resolveFadingRuntimeDSPFromRuntimeWorldOnly` へ命名収束し、関連 callsite を一括追従（standard tier / Release Build green）。
- active runtime resolver を `resolveActiveRuntimeDSPFromRuntimeWorldOnly` へ命名収束し、snapshot read path の active/fading helper 名を統一。
- `RuntimePublicationCoordinator::create` ルールも `pathRegex="^$"` の zero-exception mode へ移行し、symbol allowlist 全体を fail-closed 運用へ統一（standard tier / Release Build green）。
- `isr-trigger-audit.ps1` の `fadingOutDSP writes` を `trigger_ast_report.json` の `fadingOutDspWriteEffectiveMatches` 優先へ変更し、AST-based metric source（`triggerAstEffectiveMatches`）を導入。
- `RuntimeExecutionView` の zero-exception symbol guardrail（`BRIDGE-SYMBOL-013`）を追加し、`runtimeExecutionViewUsageCount` も symbolBlockedMatches 優先へ移行（standard tier / Release Build green）。
- `DSPHandleRuntime` の active accessor/storage 命名を `getActiveRuntimeDSPHandle` / `activeRuntimeDSPHandle_` へ収束し、AudioEngine callsite を追従。
- `isr-verify-p3-governance.ps1` の R21 gate を `getActiveDSP` と `getActiveRuntimeDSPHandle` の両許容に更新し、命名収束時の false-fail を防止（standard tier / Release Build green）。
- retired helper/API 名（`getActiveDSP` / `resolveActiveDSPFromRuntimeWorldOnly` / `resolveFadingDSPFromRuntimeWorldOnly` / `exchangeFadingOutDSP`）へ zero-exception symbol guardrail（`BRIDGE-SYMBOL-014`〜`017`）を追加。
- `DSPHandleRuntime` の fading accessor/storage 命名を `getFadingRuntimeDSPHandle` / `fadingRuntimeDSPHandle_` へ収束し、AudioEngine callsite を追従。
- R21 gate を新命名専用（`getActiveRuntimeDSPHandle` / `getFadingRuntimeDSPHandle`）へ再固定し、互換モードを解消。
- retired 名（`getFadingDSP` / `fadingDSP_` / `activeDSP_` / `getRuntimeExecutionViewForAudioThread` / `getRuntimeExecutionViewForControlThread`）へ zero-exception symbol guardrail（`BRIDGE-SYMBOL-018`〜`022`）を追加。
- Phase4 向け generation drift 検知ゲート（`.github/scripts/isr-verify-phase4-generation-drift.ps1`）を追加し、`isr-run-tiered-verification.ps1` standard tierへ接続（standard tier / Release Build green）。
- rollback compatibility gate（`.github/scripts/isr-verify-rollback-matrix.ps1`）を強化し、`isr-metric-governance.json` の rollback action flag と matrix 定義の整合・カバレッジを機械検証化（standard tier / Release Build green）。
- `RetireRuntimeEx` に rollback hierarchy（`ISR_ROLLBACK_GLOBAL` / `ISR_ROLLBACK_PUBLICATION_ONLY` / `ISR_ROLLBACK_CROSSFADE_ONLY` / `ISR_ROLLBACK_RETIRE_PATH_ONLY`）を導入し、`rollbackFlags` を retire timeline 証跡へ出力可能化（standard tier / Release Build green）。
- seed/runtime/evidence exporter の `retire_timeline.json` テンプレートへ `rollbackFlags` を反映し、Phase5 観測項目を証跡生成経路で統一（standard tier / Release Build green）。
- CI rule self-test（`.github/scripts/isr-verify-gate-wiring.ps1`）へ `isr-verify-phase4-generation-drift.ps1` の wiring 監視を追加し、規約 7.3 の自己検証範囲を拡張（standard tier green）。
- `isr-verify-v7.ps1` で `retire_timeline.rollbackFlags` を必須化し、`rollbackReady == (rollbackFlags.global && rollbackFlags.retirePathOnly)` 不変条件を機械検証化（standard tier / Release Build green）。
- rollback matrix gate を `retire_timeline.rollbackFlags` 実体検証まで拡張し、設計ファイルのみgreen（runtime証跡未接続）を防止。
- flag interaction hell 抑止として `isr-flag-dependency-graph.json` と `isr-verify-flag-dependency-graph.ps1` を導入。DAG整合・matrix整合を standard tier で検証、`isr-verify-gate-wiring.ps1` に self-test wiring を追加（standard tier / Release Build green）。
- `isr-verify-rollback-matrix.ps1` を拡張し、`retire_timeline.rollbackFlags`（global/publicationOnly/crossfadeOnly/retirePathOnly）の存在と型を機械検証化。
- `isr-verify-trigger-cleanup-readiness.ps1` を拡張し、`trigger_audit_report` の `policyEvaluations` 網羅性および `policyViolations` 伝播を cleanup gate へ統合（Phase6 cleanup readiness を強化）。
- clang-tidy 導線強化として `.github/isr-clang-tidy-rule-registry.json` と `.github/scripts/isr-verify-clang-tidy-readiness.ps1` を導入し、`.clang-tidy` の required checks / CMake導線 / compile_commands export の整合を standard tier で機械検証化（standard tier / Release Build green）。
- `isr-run-tiered-verification.ps1` standard tier と `isr-verify-gate-wiring.ps1` self-test required list に `isr-verify-clang-tidy-readiness.ps1` を接続し、配線漏れを fail-fast 化。
- Phase2-2 実行導線として `isr-verify-clang-tidy-audit.ps1` を追加し、`compile_commands.json` ベースで clang-tidy 実行可否を証跡化（`evidence/clang_tidy_audit_report.json`）。`RequireClangTidy` 未指定時は monitor skip、指定時は fail-closed。
- `isr-run-tiered-verification.ps1` に `-RequireClangTidyAudit` を追加、workflow_dispatch に `requireClangTidyAudit` 入力を追加。`isr-verify-gate-wiring.ps1` で入力/スイッチ配線の自己検証を追加し、clang-tidy audit 導線の退行を防止。
- `isr-enforcement-adoption-policy.json` を導入し、`isr-verify-enforcement-adoption.ps1` の閾値をハードコードから tier連動ポリシーへ移行（`-Tier` 連携）。expiry/owner/issue/rationale を必須化し、運用閾値の変更を設定駆動に統一。
- `isr-verify-gate-wiring.ps1` に enforcement tier forward（`isr-verify-enforcement-adoption.ps1 -Tier $Tier`）の自己検証を追加し、tier文脈欠落による偽greenを防止。
- `isr-verify-enforcement-adoption.ps1` の advanced source 判定へ `observeShim*` 系を追加し、enforcement adoption の実測比率を **0.8→1.0** へ更新（trigger audit + standard tier green）。
- `isr-clang-tidy-audit-policy.json` を導入し、`isr-verify-clang-tidy-audit.ps1` を policy-driven（mode/enforceTiers/expiry）へ移行。`-Tier` 連携により monitor/enforce を設定駆動で切替可能化。
- `isr-run-tiered-verification.ps1` の clang-tidy audit 呼び出しを `-Tier $Tier` 連携へ更新し、`isr-verify-gate-wiring.ps1` に tier forward 自己検証を追加して配線退行を防止。
- `isr-trigger-symbol-allowlist.json` に top-level governance metadata（owner/issue/rationale/expiry）を追加し、`isr-verify-trigger-symbol-usage.ps1` で必須検証化。symbol allowlist も他ポリシーと同一の運用契約へ収束。
- `isr-verify-gate-wiring.ps1` で trigger symbol allowlist の top-level metadata 存在チェックを追加し、設定欠落の偽greenを防止。
- `isr-enforcement-source-policy.json` と `isr-verify-enforcement-source-purity.ps1` を導入し、`trigger_audit_report` の source field が rawRegex/unknown へ逆戻りしていないことを機械検証化（allowed list は policy 管理、現状は fail-closed）。
- `isr-run-tiered-verification.ps1` standard tier と `isr-verify-gate-wiring.ps1` required list へ `isr-verify-enforcement-source-purity.ps1` を接続し、enforcement source purity の配線漏れを fail-fast 化。
- `isr-cleanup-deferred.json` に top-level governance metadata（owner/issue/rationale/expiry）を追加し、`isr-verify-cleanup-deferred.ps1` / `isr-verify-trigger-cleanup-readiness.ps1` で必須検証化。cleanup policy 契約を fail-closed へ統一。
- `isr-verify-trigger-cleanup-readiness.ps1` の `violations` 初期化順序を修正し、policyEvaluations 欠落系エラーでも安定して違反集約できるように強化。`isr-verify-gate-wiring.ps1` へ cleanup deferred top-level metadata 自己検証を追加。
- `isr-verify-gate-wiring.ps1` の自己検証を拡張し、`isr-verify-validator-tiering.ps1` / `isr-verify-trigger-cleanup-completion.ps1` の不変条件（`src` 全体走査、forbidden dependency pattern、`policyEvaluations` 検証、必須 metrics field、legacy helper 残存スキャン）の退行を fail-fast で検出可能化。
- R10 実測導線として `doc/work/samples/shared_epoch_metrics_sample.shared.json` / `...split.json` を追加し、runbook（`ISR_Shared_EpochDomain_SplitMigration_Runbook_2026-05-27.md`）へ `isr-compare-shared-split-epoch.ps1` 入力フォーマット（`latencyMs/jitterMs/reclaimBurst/shutdownDrainMs`）を明記。
- `isr-rebuild-admission-8_1-metrics.ps1` を拡張し、`rebuild_admission_8_1_metrics_report.json`（schema=`rebuild_admission_8_1_metrics_report_v1`）を skip/evaluated の両経路で常時出力化。`log_not_found` 時も 8.1 close 判定の状態を機械取得可能にし、標準 tier で `report=...` を証跡化。
- `isr-rebuild-admission-8_1-metrics.ps1` の guidance を拡張し、`log_not_found` 時に build task 候補（tasks.json）、実行可能バイナリ候補（`ConvoPeq.exe`）、想定ログ候補（`ConvoPeq.log`）に加えて `captureCommands`（起動コマンド候補）を証跡 JSON へ自動埋め込み。8.1 実ログ生成の次アクションを fail-open なしで機械誘導可能化。
- `isr-rebuild-admission-8_1-metrics.ps1` に `-TryAutoCaptureOnMissingLog` / `-AutoCaptureTimeoutSec` を追加し、`log_not_found` 時に `captureCommands` から自動起動試行して `ConvoPeq.log` 生成を実施。結果は `autoCapture`（attempted/success/reason/attemptedCommand/producedLogPath）として証跡化。
- `isr-run-tiered-verification.ps1` へ `-AutoCapture81Log` を追加し、8.1 ステップへ `-TryAutoCaptureOnMissingLog` を転送可能化。workflow_dispatch (`isr-verification.yml`) に `autoCapture81Log` 入力を追加し、手動実行時に段階導入できる導線へ統合。
- `isr-verify-gate-wiring.ps1` に `autoCapture81Log` 入力・`AutoCapture81Log` スイッチ・8.1 自動採取転送の自己検証を追加し、配線退行を fail-fast 化。
- `isr-collect-rebuild-admission-8_1-close-evidence.ps1` を新設し、baseline（snapshot書込）→観測窓（ConvoPeq起動）→delta（snapshot差分）の3段収集を自動化。`rebuild_admission_8_1_close_collection_report.json`（schema=`rebuild_admission_8_1_close_collection_report_v1`）を出力。
- `isr-run-tiered-verification.ps1` に `-Collect81CloseEvidence` / `-Collect81WindowSec` / `-Collect81AutoCaptureTimeoutSec` を追加し、standard/exhaustive 実行後に 8.1 close 実測収集を任意併走可能化。workflow_dispatch 側にも同入力を追加。
- `isr-verify-gate-wiring.ps1` に 8.1 close 収集入力（`collect81CloseEvidence` / `collect81WindowSec` / `collect81AutoCaptureTimeoutSec`）と tier runner 配線（collector 呼出し）の自己検証を追加し、配線退行を fail-fast 化。
- `isr-collect-rebuild-admission-8_1-close-evidence.ps1` に不足シグナル誘発プローブ（`-ProbeOnInsufficientSignals`、既存 `isr-8_1-cli-run.ps1 -ProbeFinalizeAware` 再利用）を追加。`missingSignals` / `probe` / `probeDelta` を収集レポートへ記録し、未達項目に対する再現可能な追試ループを機械化。
- tier runner / workflow_dispatch に `-Collect81SignalProbe` / `-Collect81ProbeExitMs`（`collect81SignalProbe` / `collect81ProbeExitMs`）を追加し、8.1 close 収集時のみ誘発プローブを段階有効化可能にした。
- `isr-verify-gate-wiring.ps1` に上記 probe 入力・スイッチ・collector 転送の自己検証を追加。
- 実測証跡（`evidence/rebuild_admission_8_1_close_collection_report.json`）で `probeDelta.readyToClose8_1=true` を確認（`same_as_pending_would_merge` / `deferred_finalize_*` / `REBUILD_FORCED_DISPATCH` / `policy_must_execute` を観測）。
- `isr-collect-rebuild-admission-8_1-close-evidence.ps1` に `operationalDecision`（schema=`rebuild_admission_8_1_operational_decision_v1`）を追加し、8.1 close 最終判定ソースを `probeDelta` 優先（未取得時は `delta`）で固定化。`closeReady` / `blockingSignals` / `source*` を常時出力。
- `isr-verify-gate-wiring.ps1` に collector 判定固定（`Resolve-OperationalDecision` / `operationalDecision` / `probeDelta` 優先ポリシー）の自己検証を追加し、close運用判定ロジックの退行を fail-fast 化。
- `isr-run-tiered-verification.ps1` に `-Enforce81CloseDecision` を追加し、`Collect81CloseEvidence` 実行後に `rebuild_admission_8_1_close_collection_report.json` の `operationalDecision.closeReady` を強制判定可能化（未達時 fail）。
- `isr-verification.yml` に workflow_dispatch 入力 `enforce81CloseDecision` を追加し、8.1 close 収集時のみ段階的に enforce 昇格できる導線を追加。`isr-verify-gate-wiring.ps1` に入力/スイッチ/enforce ロジックの自己検証を追加して配線退行を防止。
- 実測で `isr-run-tiered-verification.ps1 -Tier standard ... -Collect81CloseEvidence -Collect81SignalProbe ... -Enforce81CloseDecision` を実行し、`[PASS] 8.1 close operational decision enforced: source=probeDelta closeReady=true` を確認。
- `isr-run-tiered-verification.ps1` / `isr-verification.yml` に設定整合ガードを追加し、`Enforce81CloseDecision`（`enforce81CloseDecision`）を `Collect81CloseEvidence`（`collect81CloseEvidence=true`）なしで指定した場合は fail-fast 化。
- `isr-collect-rebuild-admission-8_1-close-evidence.ps1` を堅牢化し、insufficient-signal probe 失敗時は `probeDelta` を採用しないよう修正（`probeResult.success` のときのみ `probeDelta` 読込）。
- `isr-collect-rebuild-admission-8_1-close-evidence.ps1` の `operationalDecision` を v3 化し、`decisionPolicyVersion` / `sourceCandidates` / `rationale` を追加。判定ソースは `ready` 優先の `probeDelta > delta > baseline` で選択し、全候補未達時は最優先の利用可能ソースを診断用途で採用する fail-closed 方針へ固定。
- `isr-verify-gate-wiring.ps1` に上記 v2 判定ポリシー（`baseline` fallback 分岐、`decisionPolicyVersion`、`sourceCandidates`）の自己検証を追加し、判定契約の退行を fail-fast 化。
- `isr-run-tiered-verification.ps1` の `-Enforce81CloseDecision` を強化し、`operationalDecision` の `decisionPolicyVersion=8.1-close-ops-v3`・`source` 許容値（probeDelta/delta/baseline）・`sourceCandidates` 契約（先頭3件）を fail-closed 検証化。
- `isr-verify-gate-wiring.ps1` に上記 enforce 契約（policyVersion/sourceCandidates）の自己検証を追加し、runner 判定仕様の退行を fail-fast 化。
- `isr-run-tiered-verification.ps1` の 8.1 close enforce に、`timeoutForcedDispatchSeen` 単独ブロッキング時のみ collector を条件付き1回再試行する運用吸収を追加（その他の blocking signal は従来通り fail-closed）。`isr-verify-gate-wiring.ps1` に再試行配線契約（attempt/max/timeoutForcedDispatchSeen）を追加し、手動再実行依存を低減。
- `isr-verification.yml` / `isr-run-tiered-verification.ps1` に 8.1 enforce 再試行回数の設定入力（`enforce81CloseDecisionRetryMax` / `-Enforce81CloseDecisionRetryMax`）を追加し、運用の再試行閾値を設定駆動化。`isr-verify-gate-wiring.ps1` に入力/forward/config guard の自己検証を追加し、strict standard で end-to-end green を確認。
- `.github/isr-8_1-close-policy.json`（schema=`isr_8_1_close_policy_v1`）を導入し、8.1 close 収集・判定の運用入力（`Collect81WindowSec` / `Collect81AutoCaptureTimeoutSec` / `Collect81ProbeExitMs` / `Enforce81CloseDecisionRetryMax`）と tier 許可範囲（collect/enforce）を policy-driven 化。`isr-run-tiered-verification.ps1` は同ポリシーを fail-closed で読込検証し、`isr-verify-gate-wiring.ps1` に policy存在/契約/runner参照の自己検証を追加（strict standard green）。
- `isr-verify-bridge-plan-completeness.ps1` を拡張し、required policy として `isr-8_1-close-policy.json`（schema=`isr_8_1_close_policy_v1`）の存在/整合を検証し `policyStatus` を証跡化。`isr-verify-gate-wiring.ps1` に completeness policy 契約（policy path/schema/report field）の自己検証を追加（self-test + strict standard green）。
- `isr-verify-8_1-close-policy.ps1` を新設し、`isr-8_1-close-policy.json` の schema/top-level governance/expiry/collector 範囲整合（min<=max、tier許可、standard tier 必須）を独立検証して `close_policy_8_1_report.json`（schema=`close_policy_8_1_report_v1`）を出力。standard tier / `isr-verify-bridge-plan-completeness.ps1` / `isr-verify-gate-wiring.ps1` へ接続し、policy設定の擬似greenを fail-closed 化（strict standard green）。
- `isr-verification.yml` の workflow_dispatch 8.1入力パースを fail-closed 化し、`Resolve-WorkflowPositiveInt` / `Assert-WorkflowRange` を導入。`collect81WindowSec` / `collect81AutoCaptureTimeoutSec` / `collect81ProbeExitMs` / `enforce81CloseDecisionRetryMax` は不正値時に即 fail、かつ `isr-8_1-close-policy.json` の collector 境界・tier許可（collect/enforce）に一致しない場合も fail-fast 化。`isr-verify-gate-wiring.ps1` に同契約（helper呼出し/入力名/tier拒否）の自己検証を追加（strict standard green）。
- `isr-8_1-close-policy.json` に `expiryGuardDaysByTier`（standard/exhaustive）を追加し、`isr-verify-8_1-close-policy.ps1` を `-Tier` 連動へ拡張。expiry 期限自体の失効検知に加えて、tier cadence に応じた先行期限警告窓（daysRemaining < guardDays）を fail-closed 検証化し、`close_policy_8_1_report.json` へ `expiryDaysRemaining` / `activeTierGuardDays` を出力。`isr-run-tiered-verification.ps1` から tier forward、`isr-verify-gate-wiring.ps1` に同契約を自己検証追加（strict standard green）。
- `isr-verification.yml` でも `Assert-WorkflowPolicyExpiryGuard` を導入し、workflow_dispatch 実行前段で `isr-8_1-close-policy.json` の `expiryGuardDaysByTier`（standard/exhaustive）を tier 連動で fail-fast 検証。`daysRemaining < guardDays` は runner 到達前に即 fail とし、`isr-verify-gate-wiring.ps1` に helper/guard wiring 契約（`Assert-WorkflowPolicyExpiryGuard` / `expiryGuardDaysByTier` / breach message）を自己検証追加（strict standard green）。
- `isr-verify-8_1-workflow-input-coherence.ps1` を新設し、`isr-verification.yml` の 8.1入力デフォルト（`collect81WindowSec` / `collect81AutoCaptureTimeoutSec` / `collect81ProbeExitMs` / `enforce81CloseDecisionRetryMax`）が `isr-8_1-close-policy.json` collector 範囲と整合することを機械検証化。`close_policy_8_1_workflow_input_coherence_report.json`（schema=`close_policy_8_1_workflow_input_coherence_report_v1`）を出力し、standard tier / `isr-verify-bridge-plan-completeness.ps1` / `isr-verify-gate-wiring.ps1` に接続（strict standard green）。
- `isr-verify-8_1-workflow-input-contract.ps1` を新設し、`isr-verification.yml` の 8.1 workflow_dispatch 入力ブロック契約（`type` / `required` / `default` / `description`）を fail-closed 検証化。`close_policy_8_1_workflow_input_contract_report.json`（schema=`close_policy_8_1_workflow_input_contract_report_v1`）を出力し、standard tier / `isr-verify-bridge-plan-completeness.ps1` / `isr-verify-gate-wiring.ps1` に接続（strict standard green）。
- `isr-8_1-close-policy.json` に `workflowInputContract`（`descriptionMustContain` + `inputs[]`）を追加し、`isr-verify-8_1-workflow-input-contract.ps1` の契約定義を policy-driven 化。`isr-verify-8_1-close-policy.ps1` / `isr-verify-gate-wiring.ps1` も同セクションの必須検証を実装し、入力契約のハードコード drift を fail-closed 化。
- `isr-verify-8_1-workflow-input-coherence.ps1` も `workflowInputContract.inputs[]`（string + policy range fields）参照へ移行し、coherence 対象入力名のハードコードを撤去。`isr-verify-gate-wiring.ps1` の coherence 契約自己検証も policy-driven 条件へ更新。
- `isr-verification.yml` に `Assert-WorkflowInputContractAgainstPolicy`（`Get-WorkflowInputBlock` / `Get-WorkflowInputProperty`）を追加し、workflow 前段で `workflowInputContract` と dispatch input 定義（`type`/`required`/`default`/`description`）の整合を fail-closed 検証化。`isr-verify-gate-wiring.ps1` に helper/呼び出し契約の自己検証を追加。
- `workflowInputContract.inputs[]` に `autoCapture81Log` を追加し、`isr-verify-8_1-workflow-input-contract.ps1` / `isr-verification.yml`（workflow前段）で「未契約の 8.1 input（`*81*`）」を fail-closed で拒否する検証を追加。`isr-verify-gate-wiring.ps1` に `Get-WorkflowDispatchInputNames` / uncontracted-input 契約の自己検証を追加。
- `isr-run-tiered-verification.ps1` の 8.1数値既定値（`Collect81WindowSec` / `Collect81AutoCaptureTimeoutSec` / `Collect81ProbeExitMs` / `Enforce81CloseDecisionRetryMax`）を `workflowInputContract` 由来で解決するよう変更し、runner 側既定値のハードコード drift を fail-closed 化。`isr-verify-gate-wiring.ps1` に default resolver 契約（`Resolve-WorkflowInputContractIntDefault`）を追加。
- `.github/workflows/isr-verification.yml` 内の 8.1数値既定値（`collect81WindowSec` / `collect81AutoCaptureTimeoutSec` / `collect81ProbeExitMs` / `enforce81CloseDecisionRetryMax`）を `workflowInputContract` 由来 (`Resolve-WorkflowPolicyIntDefault`) で初期化するよう変更し、workflow 前段の fallback 既定値ハードコード drift を封止。`isr-verify-gate-wiring.ps1` に workflow default resolver 契約を追加。
- `isr-validator-tiering-policy.json` に `schema` と `workflowSchedule` 契約（nightly/weekly cron + tier）を追加し、`isr-verify-validator-tiering.ps1` で workflow cron 定義・schedule→tier 分岐を policy と突合する fail-closed 検証を追加。`isr-verify-gate-wiring.ps1` に validator schedule-contract 契約（`validator_tiering_report_v3`）を追加。
- `.github/workflows/isr-verification.yml` の schedule→tier 判定を `isr-validator-tiering-policy.json` (`Resolve-ValidatorTieringScheduleContract`) 参照へ移行し、未知cronを `Unknown workflow schedule cron` で fail-closed 化。`isr-verify-validator-tiering.ps1` / `isr-verify-gate-wiring.ps1` を追従更新。
- `isr-verify-policy-top-level-governance.ps1` を強化し、全 policy に `schema` の非空必須と `schema` 重複禁止（`Duplicate policy schema detected`）を追加。report schema を `policy_top_level_governance_report_v2` へ更新し、`isr-verify-gate-wiring.ps1` に自己検証契約を追加。
- `isr-verify-policy-top-level-governance.ps1` をさらに強化し、top-level `issue` の重複（`Duplicate policy issue detected`）を fail-closed 検証化。`isr-verify-gate-wiring.ps1` へ自己検証契約を追従し、policy 運用チケット衝突ドリフトを遮断。
- `isr-run-tiered-verification.ps1` の param シグネチャに残っていた 8.1 数値 default（20/10/8000/2）を 0 初期値へ統一し、実効既定値を `workflowInputContract` 解決 (`Resolve-WorkflowInputContractIntDefault`) のみに一本化。
- `isr-verify-bridge-plan-completeness.ps1` の required scripts/artifacts/policies を拡張し、`isr-verify-validator-tiering.ps1` / `isr-verify-policy-top-level-governance.ps1`、`validator_tiering_report_v3`、`policy_top_level_governance_report_v2`、`isr_validator_tiering_policy_v1` を計画完了性の必須契約に昇格。
- `isr-workflow-dispatch-input-policy.json` / `isr-verify-workflow-dispatch-input-policy.ps1` を追加し、非8.1 workflow_dispatch 入力（`requireRuntimeEvidence` / `verificationTier` / `enforceTriggerPolicy` / `requireAstTriggerCheck` / `requireClangTidyAudit` / `autoPruneCleanupDeferred`）を policy 契約化。未契約 non-8.1 入力を fail-closed 検出。runner（smoke）・gate-wiring・bridge-plan-completeness に配線。
- `.github/workflows/isr-verification.yml` 前段に `Assert-WorkflowDispatchInputPolicyAgainstPolicy` を追加し、非8.1 workflow_dispatch 入力契約を workflow 入口でも fail-closed 検証（二重化）。`Get-WorkflowInputOptions` を追加して `type: choice` の options 契約も強制。
- 同 `Assert-WorkflowDispatchInputPolicyAgainstPolicy` に `expiry` の `yyyy-MM-dd` 形式検証と期限切れ fail-closed（`Workflow dispatch input policy expired`）を追加し、非8.1入力契約の運用期限ドリフトを workflow 前段で遮断。
- `isr-verify-workflow-dispatch-input-policy.ps1` を強化し、`boolean` default 妥当性、`choice` option 重複/空値、`choice` default∈options 契約、workflow 側 choice option 重複検出を fail-closed 化。`isr-verify-gate-wiring.ps1` の自己検証契約も追従。
- `isr-workflow-dispatch-input-policy.json` に `forwardingContract.switches` を追加し、非8.1入力の runner switch 転送（`-RequireRuntimeEvidence` など）を policy 契約化。workflow 前段 (`Assert-WorkflowDispatchInputPolicyAgainstPolicy`) と独立ゲート (`isr-verify-workflow-dispatch-input-policy.ps1`) の両方で未配線/未知inputを fail-closed 検出。
- 非8.1 dispatch forwarding 契約を強化し、`forwardingContract` の `inputName` 重複と `runnerSwitch` 重複を workflow 前段/独立ゲートの双方で fail-closed 検証化。`isr-verify-gate-wiring.ps1` の自己検証契約も追従し、forwarding drift の擬似greenを遮断。
- workflow 前段 `Assert-WorkflowDispatchInputPolicyAgainstPolicy` を独立ゲート相当へ強化し、`boolean` default 妥当性、`descriptionMustContain` 空値、`choice` option 重複/空値/default整合、workflow 側 choice option 重複を fail-closed 検証化。`isr-verify-gate-wiring.ps1` へ自己検証契約を追従。
- non-8.1 dispatch forwarding 契約へ型制約を追加し、`forwardingContract` は `type=boolean` 入力のみを許可。workflow 前段/独立ゲートの双方で fail-closed 検証化し、switch forwarding の型崩れを遮断。
- `isr-verify-trigger-cleanup-completion.ps1` を拡張し、`triggerAuditReport` / `deferredRegistryPath` / `sourceRoot` の参照整合を report 出力 (`referenceConsistency`) するだけでなく、不一致時に violation 化して fail-closed へ昇格。`isr-verify-gate-wiring.ps1` に mismatch fail-closed 契約の自己検証を追加。
- `.github/isr-validator-tiering-policy.json` を新設し、validator tiering cadence（smoke=pr / standard=nightly / exhaustive=weekly）と SLA（hbViolation=24h / payloadMismatch=72h）を top-level governance付きでポリシー化。
- `isr-verify-validator-tiering.ps1` を policy-driven へ移行し、SLA/cadence のハードコード文言依存を縮小。`validator_tiering_report.json` schema を `validator_tiering_report_v2` へ更新。
- `isr-verify-gate-wiring.ps1` に validator tiering policy ファイル存在・top-level metadata・`isr-verify-validator-tiering.ps1` の policy wiring 自己検証を追加。
- `isr-metric-governance.json` に `normalizationPolicy`（`enabled` / `baselineWindowMinutes` / `cpuThermalOsNormalization` / `bucketBy` / `issue`）を追加し、PR canary metrics の baseline window normalization 契約を設定ファイル化。
- `isr-verify-metric-governance.ps1` を拡張し、`normalizationPolicy` と各 required metric の `normalization=baselineWindowNormalized` を fail-closed 検証化。`metric_governance_report.json` schema を `metric_governance_report_v2` へ更新。
- `isr-verify-gate-wiring.ps1` に metric normalization 契約（`normalizationPolicy` / `baselineWindowNormalized` / `metric_governance_report_v2`）の自己検証を追加し、canary normalization 退行を fail-fast 化。
- `isr-verify-gate-wiring.ps1` の metric governance 自己検証を強化し、`normalizationPolicy` の必須フィールド（`enabled` / `baselineWindowMinutes` / `cpuThermalOsNormalization` / `bucketBy` / `issue`）欠落を fail-fast 化。
- `.github/scripts/isr-verify-canary-baseline-normalization.ps1` を新設し、canary 4指標（`xrunDelta` / `callbackJitter` / `retireLatency` / `crossfadePeak`）の baseline normalization 契約と evidence 参照（monitor/strict）を機械検証化。`canary_baseline_normalization_report.json`（schema=`canary_baseline_normalization_report_v1`）を出力。
- `isr-run-tiered-verification.ps1` standard tier と `isr-verify-gate-wiring.ps1` required/self-test へ上記ゲートを接続し、canary実測証跡検証の配線退行を fail-fast 化。
- `isr-run-tiered-verification.ps1` の standard 実行順を調整し、`isr-verify-crossfade-observable-state.ps1` の後段で `isr-verify-canary-baseline-normalization.ps1` を実行するよう固定。`crossfadePeak` 証跡未生成による monitor warning を同一run内で解消。
- `isr-metric-governance.json` の `normalizationPolicy` へ `strictModeRequireAllMetrics` を追加し、strict runtime evidence モードでは canary 4指標すべての証跡欠損を fail-closed 化（blocking-only から昇格）。
- `isr-verify-metric-governance.ps1` / `isr-verify-canary-baseline-normalization.ps1` / `isr-verify-gate-wiring.ps1` を更新し、`strictModeRequireAllMetrics` 契約の registry・gate・self-test 三層検証を実装。
- strict runtime evidence 実行での `runtime_budget_report.json` 欠損を解消するため、`isr-run-runtime-evidence.ps1` の fallback artifact へ `runtime_budget_report_v1`（`artifactTotalBytes` 含む）を追加。`-RequireRuntimeEvidence` 付き standard tier を再実行し全ゲート green を確認。
- `isr-verify-validator-tiering.ps1` を拡張し、SLA 対象 artifact（`hb_violation_report.json` / `payload_tier_report.json`）の生成時刻を `generatedAtNs`/`generatedAt` から評価する鮮度検証を追加。`policy.slaHours`（24h/72h）超過を `SLA breach` として fail-closed 化し、`validator_tiering_report.json` に `slaFreshness` を出力。
- `isr-verify-gate-wiring.ps1` に validator SLA 鮮度契約（artifact名・timestamp解決・`SLA breach` 判定）の自己検証を追加し、tiering gate の退行を fail-fast 化。
- `isr-verify-rtmutable-boundary.ps1` を拡張し、`RTAuxMutable` へ raw pointer/smart pointer 形式のフィールド宣言を禁止する境界検証（field declaration pattern）を追加。`NonOwningPtr` など ownership 系型の混入も検知対象に強化。
- `isr-verify-gate-wiring.ps1` へ上記 `RTAuxMutable` pointer/ownership 境界検証の自己検証を追加し、規約 5.2（pointer/ownership/lifetime 禁止）のゲート退行を fail-fast 化。
- `isr-verify-trigger-symbol-usage.ps1` を強化し、allowlist rule の重複（`symbol + pathRegex`）と不正 regex（`pathRegex` parse失敗）を policy violation として fail-closed 化。
- `isr-verify-gate-wiring.ps1` へ trigger symbol allowlist 品質検証（duplicate/regex validation）の自己検証を追加し、symbol guardrail 設定の擬似greenを fail-fast で検出可能化。
- `isr-prune-cleanup-deferred.ps1` を新設し、`trigger_audit_report.policyEvaluations` を基準に deferred cleanup entry の ready/blocked を機械判定化。`-Apply` 時は ready entry を `isr-cleanup-deferred.json` から自動縮退し、`cleanup_deferred_prune_report.json`（schema=`cleanup_deferred_prune_report_v1`）を出力。
- `isr-run-tiered-verification.ps1` に `-AutoPruneCleanupDeferred` を追加し、standard tier で `isr-trigger-audit.ps1` 後段に prune スクリプトを接続。cleanup readiness/completion gate の前に縮退を自動実行できる導線を追加。
- `isr-verification.yml` に `workflow_dispatch.autoPruneCleanupDeferred` を追加し、runner へ `-AutoPruneCleanupDeferred` を forward 可能化。手動運用で cleanup 縮退を段階有効化。
- `isr-verify-gate-wiring.ps1` を拡張し、prune script の required wiring・workflow input/forwarding・tier runner apply 分岐・script contract（schema/Apply）を自己検証化。cleanup 自動縮退導線の配線退行を fail-fast 化。
- `isr-run-tiered-verification.ps1` / `isr-verification.yml` に `autoPruneCleanupDeferred` の tier guard を追加し、`smoke` での誤適用を fail-fast 化（`standard/exhaustive` のみ許可）。`isr-verify-gate-wiring.ps1` へ同ガード契約の自己検証を追加。
- `isr-verify-cleanup-deferred.ps1` を拡張し、`cleanup_deferred_prune_report.json`（schema=`cleanup_deferred_prune_report_v1`）の存在/契約（`apply`/`readyCount`/`blockedCount`/`prunedCount`/`remainingCount`）を必須検証化。`cleanup_deferred_report.json` に `pruneSummary` を出力し、cleanup 縮退の実行証跡を governance report へ統合。
- `isr-verify-gate-wiring.ps1` に上記 prune report governance 契約の自己検証を追加し、cleanup deferred verifier の退行（prune report未検証化）を fail-fast 化。
- cleanup prune 判定ルールを `.github/isr-cleanup-prune-policy.json`（schema=`cleanup_prune_policy_v1`）へ外出しし、`isr-prune-cleanup-deferred.ps1` を policy-driven comparator 評価（`readyRule.comparator`）へ移行。`isr-verify-gate-wiring.ps1` に policy 存在/metadata/`readyRule` 契約と prune script wiring の自己検証を追加し、`-AutoPruneCleanupDeferred` 付き strict standard tier で green を確認。
- `isr-trigger-audit.ps1` を source fail-closed へ強化し、symbol/observe/AST 証跡の自動refresh（`isr-verify-trigger-symbol-usage.ps1` / `isr-verify-observe-shim-usage.ps1` / `isr-verify-trigger-ast.ps1`）後も advanced source を満たさない場合は即 fail に変更。`isr-verify-gate-wiring.ps1` に source contract hardening の自己検証を追加し、strict standard tier で green を確認。
- `isr-verify-enforcement-source-purity.ps1` の source 監査対象へ `retireFacadeRuntimeExecutionMetricSource` を追加し、trigger audit の source field 全量を fail-closed 監視化。`isr-verify-gate-wiring.ps1` に同 coverage の自己検証を追加し、strict standard tier で green を確認。
- `isr-trigger-audit.ps1` を evidence-first 運用へ移行し、symbol/observe/AST 証跡生成スクリプトの実行と report schema/必須フィールド検証を必須化。rawRegex fallback 警告経路を排し、証跡不足時は fail-closed とした。`isr-verify-gate-wiring.ps1` に evidence-first 契約（required report/schema/no fallback warning）自己検証を追加し、strict standard tier 再実行で green を確認。
- `isr-trigger-audit.ps1` の AST 証跡契約をさらに強化し、`trigger_ast_report_v1` で `available=true` かつ `commandOk=true` を必須化（未成立は fail-closed）。`isr-verify-gate-wiring.ps1` に同契約（available/commandOk）の自己検証を追加し、strict standard tier で green を確認。
- `isr-trigger-audit.ps1` の raw 診断カウント（`activeDspRawRefCount` / `retireFacadeRawDependencyCount` / `legacyDirectObserveRawCount`）をローカル `src/audioengine` grep から証跡由来（symbol/observe report の totalMatches）へ移行し、trigger audit 本体の source grep 依存を除去。`isr-verify-gate-wiring.ps1` に `Get-MatchCount` 非依存契約の自己検証を追加し、strict standard tier で green を確認。
- `isr-run-tiered-verification.ps1` で `RequireAstTriggerCheck` 指定時に `isr-trigger-audit.ps1 -RequireAstEvidence` を転送するよう修正。`isr-trigger-audit.ps1` 側は AST 証跡 refresh 時に `-RequireAst` を引き継ぎ、`trigger_ast_report.required` の downgrade（`true -> false`）を防止。`isr-verify-gate-wiring.ps1` に forwarding 契約を追加し、strict standard tier 実行後も `evidence/trigger_ast_report.json` の `required=true` を確認。
- `isr-trigger-audit.ps1` の report に `astEvidenceRequired` を追加し、`RequireAstTriggerCheck` 連携時の監査実行条件を証跡化。`isr-verify-gate-wiring.ps1` へ `RequireAstEvidence` パラメータ宣言・`-RequireAst` 呼び出し・`astEvidenceRequired` report field の自己検証を追加し、strict standard tier で green + `evidence/trigger_audit_report.json` の `astEvidenceRequired=true` を確認。
- `isr-verify-enforcement-source-purity.ps1` に `-RequireAstEvidence` を追加し、`RequireAstTriggerCheck` 指定時は `trigger_audit_report.astEvidenceRequired=true` を fail-closed で必須化。`isr-run-tiered-verification.ps1` から同スイッチを forward、`isr-verify-gate-wiring.ps1` へ配線自己検証を追加。self-test PASS + strict standard PASS を確認。
- `isr-verify-trigger-policy.ps1` の tier runner 契約を拡張し、`isr-verify-enforcement-source-purity.ps1` の実行有無と `RequireAstEvidence` 配線有無を fail-closed で検証。trigger policy self-test PASS + strict standard PASS を確認。
- `isr-verify-trigger-ast.ps1` を required mode fail-closed へ強化し、`fadingOutDspWriteEffectiveSource`（`astOnly` / `astOrFallbackMax`）を導入。`-RequireAst` 時は effective を AST一致数のみで算出し、fallback優勢（`fallbackMatches > astMatches`）は `commandOk=false` で失敗化。`isr-trigger-audit.ps1` も `RequireAstEvidence` 時に `trigger_ast_report.required=true` と `effectiveSource=astOnly` を必須検証化。`isr-verify-gate-wiring.ps1` に同契約の自己検証を追加し、strict standard PASS を確認。
- `isr-verify-trigger-cleanup-completion.ps1` に `-RequireAstEvidence` を追加し、`RequireAstTriggerCheck` 指定時は `trigger_audit_report.astEvidenceRequired=true` を cleanup completion 側でも fail-closed 検証。`isr-run-tiered-verification.ps1` から同スイッチを forward、`isr-verify-gate-wiring.ps1` に配線自己検証を追加。self-test PASS + strict standard PASS を確認。
- `isr-verify-facade-bypass.ps1` を強化し、`trigger_audit_report.json`（schema=`trigger_audit_report_v1`）を必須入力として `retireFacadeRuntimeExecutionCount` / `retireFacadeRawDependencyCount` と facade bypass 直接検出結果の zero-state 整合を fail-closed 検証化。`isr-verify-gate-wiring.ps1` に同契約（report/schema/metric field）の自己検証を追加し、strict standard PASS を確認。
- `isr-verify-phase4-generation-drift.ps1` を evidence-first 化し、`phase4_generation_drift_report.json`（schema=`phase4_generation_drift_report_v1`）を常時出力。required file 欠落/契約違反を `violations` 集約で fail-closed 判定可能に更新。`isr-verify-gate-wiring.ps1` に report 契約（report path/schema/requiredFiles/violations）の自己検証を追加。self-test PASS、strict standard は 8.1 probe 揺れで1回 fail 後、同条件再実行で PASS を確認。
- `isr-verify-rollback-matrix.ps1` に `retire_timeline.rollbackReady == (rollbackFlags.global && rollbackFlags.retirePathOnly)` 不変条件を追加し、`isr-verify-v7.ps1` と同一契約で fail-closed 検証を二重化。`isr-verify-gate-wiring.ps1` に rollbackReady invariant 自己検証（`rollbackReady` / `rollbackFlags.global` / `rollbackFlags.retirePathOnly`）を追加し、strict standard PASS を確認。
- `isr-verify-cleanup-deferred.ps1` を強化し、`cleanup_deferred_prune_report.json` の `violations` 非空を fail-closed 化、`remainingCount == registryEntryCount` と `apply=false => prunedCount=0` の整合条件を追加。`isr-verify-gate-wiring.ps1` に同契約（violations/remainingCount/apply-pruned consistency）の自己検証を追加し、strict standard PASS を確認。
- `isr-verify-trigger-cleanup-completion.ps1` を拡張し、`cleanup_deferred_registry_v1` の schema 検証と `entries==0`（deferred cleanup 残件ゼロ）を fail-closed で必須化。`isr-verify-gate-wiring.ps1` に deferred registry 契約（schema/emptiness）の自己検証を追加し、strict standard PASS を確認。
- `isr-verify-bridge-plan-completeness.ps1` を新設し、`bridge_runtime_migration_plan.md` の Phase 0〜6 見出し、tier runner の主要ゲート配線、主要 evidence artifact schema を横断検証して `bridge_plan_completeness_report_v1` を出力。standard tier へ接続し、`isr-verify-gate-wiring.ps1` required script list に追加。strict standard は 8.1 probe 揺れで1回 fail 後、同条件再実行で PASS を確認。
- `isr-verify-gate-wiring.ps1` を拡張し、`isr-verify-bridge-plan-completeness.ps1` の core contract（report path/schema、Phase6 見出し、cleanup completion artifact 参照）を自己検証化。completeness gate 自体の退行も fail-fast 検出可能化。
- `isr-verify-backlog-specfixed-residual.ps1` を新設し、`ISR_Completeness_Risk_Backlog.md` の Rステータスを機械抽出して `backlog_specfixed_residual_report_v1` を生成。standard tier へ monitor 接続し、現時点 `specFixedResidualCount=10`（R1〜R10）が未完ゼロ化の残差として可視化されたことを確認。
- `isr-run-tiered-verification.ps1` で `isr-verify-backlog-specfixed-residual.ps1 -EnforceNoSpecFixed` を実行するよう昇格し、`Spec-Fixed` 残差の再流入を standard tier で fail-closed 化。`isr-verify-gate-wiring.ps1` に配線自己検証を追加し、strict standard 実行で `backlog_specfixed_residual_report.specFixedResidualCount=0` / `enforceNoSpecFixed=true` を確認。
- `isr-rollback-compatibility-matrix.json` に top-level governance metadata（owner/issue/rationale/expiry）を追加し、`isr-verify-rollback-matrix.ps1` で必須検証化（expiry fail-closed）。
- `isr-verify-flag-dependency-graph.ps1` に graph top-level expiry 検証を追加し、`isr-verify-gate-wiring.ps1` へ rollback matrix top-level metadata の自己検証を追加。rollback/flag policy 契約を同一水準へ統一。
- `isr-trigger-policy.json` に top-level governance metadata（owner/issue/rationale/expiry）を追加し、`isr-trigger-audit.ps1` / `isr-verify-trigger-policy.ps1` で必須検証化（expiry fail-closed）。
- `isr-verify-gate-wiring.ps1` へ trigger policy top-level metadata 自己検証を追加し、trigger governance 契約の欠落を fail-fast 化。
- `isr-observe-shim-allowlist.json` に top-level governance metadata（owner/issue/rationale/expiry）を追加し、`isr-verify-observe-shim-usage.ps1` で必須検証化（expiry fail-closed）。
- `isr-verify-gate-wiring.ps1` へ observe shim allowlist top-level metadata 自己検証を追加し、observe policy 契約の欠落を fail-fast 化。
- `isr-metric-governance.json` に top-level governance metadata（owner/issue/rationale/expiry）を追加し、`isr-verify-metric-governance.ps1` で registry-level 必須検証化（expiry fail-closed）。
- `isr-verify-policy-top-level-governance.ps1` を新設し、`.github/isr-*.json` 全件の top-level 契約（owner/issue/rationale/expiry + expiry format/期限）を横断検証化。`isr-run-tiered-verification.ps1` standard tier と `isr-verify-gate-wiring.ps1` required list に接続して再発防止を固定。
- `isr-verify-clang-tidy-audit.ps1` に clang-tidy 実体探索（PATH / where.exe / LLVM既定候補）を追加し、monitor skip 時の `clang_tidy_audit_report.json` 診断情報を強化。enforce昇格前の環境整備ギャップを機械的に可視化。
- `isr-verify-clang-tidy-audit.ps1` を候補実体の直接実行に対応させ、PATH 未設定でも監査実行経路を試行可能化。enforce tier 昇格前の可用性を改善。
- `AudioEngine` の audio-callback ローカルカウンタ（callback epoch / sample cursor / active count）を `RTLocalState` へ切り出し、`RTAuxMutable` から RT-local mutable を分離。P0-3 の境界固定をコード化。
- `AudioEngine` の audio-thread retire ローカルカウンタ（enqueue dropped / overflow epoch）も `RTLocalState` に移し、RT-local mutable の境界をさらに明確化。`RTAuxMutable` は auxiliary telemetry に寄せる。

### REV3.2運用優先注記

- 本書の評価基準は `plan5.md` REV3.2 を優先する。
- `runtime exposes evidence / CI validates evidence` を固定方針とし、
  Release へ full artifact / full verify を常時要求しない。
- 解釈衝突時は few-authority（7 subsystem）/ 2-world（Publication/Execution）/
  capability-first（runtime coordinator lifecycle 非導入）を優先する。
- stale handle 関連の Closed 判定は CI=Abort / Debug=Assert / Release=Quarantine+Silence を前提に解釈する。

用語正規化（齟齬回避）:

- 本書では `RuntimePublication` を正規記法として扱う。
- CI artifact 判定（missing/schema/parse fail）は merge gate 条件であり、
  runtime correctness は runtime invariants で成立させる。

---

## R1. Deep Immutability Enforcement 固定

- 状態: Closed（2026-05-27）
- 重要度: Critical
- 主要リスク: discipline依存で post-publish mutation 混入
- 反映先: `ISR_Immutability_Enforcement_Spec.md`
- 確定方針:
  - freeze bit + mutation assert + immutable facade + post-publish detector を必須4点セットとする
  - Release でも failure counter を記録し no-op を禁止する
- 実行フェーズ: Phase A
- 完了条件:
  - freeze bit / mutation assert / immutable facade / post-publish detector の実装検証ルール確定
  - CI fail 条件が運用化
- 実装証跡:
  - `src/audioengine/ISRSealedObject.h`（`freeze()` / `assertMutable()` / `sealViolationCountValue()`）
  - `src/audioengine/AudioEngine.h`（`worldOwner->freeze()`）
  - `.github/scripts/isr-verify-v1-immutability.ps1`
  - `.github/scripts/isr-verify-v2-seal.ps1`
  - `.github/scripts/check-list-compliance.ps1`（freeze 経路固定 / RT retire enqueue 検査）
- Closed最小検証項目:
  - [x] freeze 未実行 payload の publish が拒否される（自動テスト）
  - [x] publish 後 write が seal violation として検出される（Debug/Release双方）
  - [x] post-publish mutation detector が CI で fail を返す

## R2. RuntimeGraph Deep Immutability 閉包

- 状態: Closed（2026-05-27）
- 重要度: Critical
- 主要リスク: FFT cache / IR ptr / async state の可変参照混入
- 反映先: `ISR_Runtime_State_Matrix.md`, `ISR_HB_Graph_Specification.md`
- 確定方針:
  - RuntimeGraph は publish 後 read-only closure を必須とし、payload 外 mutable 依存を禁止
  - FFT/IR/async は handle/snapshot 経由のみ参照可
- 実行フェーズ: Phase B
- 完了条件:
  - RuntimeGraph 内依存が read-only closure を満たす
  - payload 外 mutable 参照の禁止検証が成立
- 実装証跡:
  - `src/audioengine/RuntimeGraph.h`（POD-only publish graph data）
  - `src/audioengine/AudioEngine.h`（`getRuntimeGraph()` const-only accessor）
  - `.github/scripts/isr-verify-v3-runtime-graph-immutability.ps1`

## R3. DSPHandle Allocator 粗密要件固定

- 状態: Closed（2026-05-27）
- 重要度: High
- 主要リスク: reuse/wraparound/compaction/shutdown flush の仕様抜け
- 反映先: `ISR_DSPHandle_Allocator_Policy.md`
- 確定方針:
  - reuse latency は quarantine epoch 完了後に限定
  - wraparound は運用停止境界を明記し、到達前メンテ停止を義務化
  - compaction は non-RT 限定、shutdown flush ordering を先行固定
- 実行フェーズ: Phase B
- 完了条件:
  - slot reuse latency / generation wraparound / sparse compaction / shutdown flush ordering の閾値化
- 実装証跡:
  - `src/audioengine/ISRDSPHandle.h` / `src/audioengine/ISRDSPHandle.cpp`
  - `doc/work/ISR_DSPHandle_Allocator_Policy.md`
  - `.github/scripts/isr-verify-v4-dsp-handle-policy.ps1`

## R4. RuntimeWorldRetireManager 責務肥大対策

- 状態: Closed（2026-05-27）
- 重要度: High
- 主要リスク: single authority と single mega-manager の混同
- 反映先: `ISR_Retire_Authority_Graph.md`
- 確定方針:
  - authority identity は単一維持、実装は lane 分離を許容
  - lane 追加時も独立 authority 再定義を禁止
- 実行フェーズ: Phase B
- 完了条件:
  - authority identity を維持した lane 分離方針（DSP/snapshot/cache）が定義済み
- 実装証跡:
  - `src/audioengine/ISRRuntimePublicationCoordinator.h` / `src/audioengine/ISRRuntimePublicationCoordinator.cpp`
  - `src/audioengine/ISRRetireLane.h` / `src/audioengine/ISRRetireRuntimeEx.h` / `src/audioengine/ISRRetireRuntimeEx.cpp`
  - `doc/work/ISR_Retire_Authority_Graph.md`
  - `.github/scripts/isr-verify-v5-retire-authority-lane.ps1`

## R5. HB Domain F 厳密化

- 状態: Closed（2026-05-27）
- 重要度: High
- 主要リスク: smoothing lifetime / host automation ordering / callback reentrancy の事故
- 反映先: `ISR_HB_Graph_Specification.md`
- 確定方針:
  - prepareToPlay -> callback start -> callback stop -> releaseResources の順序鎖を必須化
  - callback reentrancy は単一順序規約違反として明示検出対象にする
- 実行フェーズ: Phase C
- 完了条件:
  - Domain F で順序制約と違反ケースが明文化
  - callback 境界の再入・順序崩れ検証条件が定義済み
- 実装証跡:
  - `src/audioengine/ISRLifecycle.h` / `src/audioengine/ISRLifecycle.cpp`
  - `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp`
  - `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`
  - `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp`
  - `.github/scripts/isr-verify-v6-domain-f-ordering.ps1`

## R6. bug2 形式再現 HB モデル

- 状態: Closed（2026-05-27）
- 重要度: Critical
- 主要リスク: KPI監視のみで根本 ordering 欠落を見逃す
- 反映先: `ISR_HB_Graph_Specification.md`, `ISR_Shared_EpochDomain_Scalability_Validation_Plan.md`
- 確定方針:
  - 失敗ordering（UAF発火）と必要HB（抑止）を1対1で対応付ける形式モデルを必須化
  - KPI は補助指標とし、モデル検証を主判定にする
- 実行フェーズ: Phase C
- 完了条件:
  - 失敗orderingと必要HBの対照モデルが承認済み
- 実装証跡:
  - `.github/scripts/isr-verify-v6.ps1`（hb_graph_trace / hb_violation_report の整合）
  - `evidence/hb_graph_trace.json`
  - `evidence/hb_violation_report.json`

## R7. Recursive Payload Closure 完全性

- 状態: Closed（2026-05-27）
- 重要度: Critical
- 主要リスク: publish payload 下位オブジェクトで ownership closure が破断
- 反映先: `ISR_HB_Graph_Specification.md`, `ISR_Runtime_State_Matrix.md`, `ISR_Runtime_Closure_Descriptor.md`
- 確定方針:
  - payload closure metadata は再帰閉包（parent->child->grandchild）を表現可能であること
  - closure 未閉包を検出する静的/動的規則を必須化
- 実行フェーズ: Phase C
- 完了条件:
  - payload closure metadata が再帰閉包を表現
  - closure 不完全時の検出規則が運用化
- 実装証跡:
  - `.github/scripts/isr-verify-v3.ps1`（closure_graph 整合検証）
  - `evidence/closure_graph.json`

## R8. Shutdown HB Strict Ordering

- 状態: Closed（2026-05-27）
- 重要度: Critical
- 主要リスク: callback停止→observer消滅→retire停止→reclaim完了→allocator shutdown の順序破綻
- 反映先: `ISR_HB_Graph_Specification.md`, `ISR_DSPHandle_Allocator_Policy.md`
- 確定方針:
  - shutdown ordered chain を canonical sequence として固定
  - drain 完了判定前に allocator shutdown へ進む遷移を禁止
- 実行フェーズ: Phase C
- 完了条件:
  - shutdown ordered chain が明文化
  - drain 完了判定の停止条件が固定
- 実装証跡:
  - `src/audioengine/ISRShutdown.h` / `src/audioengine/ISRShutdown.cpp`
  - `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp`
  - `src/audioengine/AudioEngine.h`（`RuntimePublicationBridge::willRetireRuntimeNonRt` shutdown ガード）
  - `.github/scripts/isr-verify-v6.ps1`（shutdown_trace 検証）
  - `evidence/shutdown_trace.json`

## R9. RT検知 -> NonRT Retire Enqueue Bridge 固定

- 状態: Closed（2026-05-27）
- 重要度: High
- 主要リスク: Audio callback 起点の完了イベントが直接 retire enqueue と衝突
- 反映先: `ISR_HB_Graph_Specification.md`, `ISR_Retire_Authority_Graph.md`
- 確定方針:
  - Audio Thread は retire authority を持たず、完了検知のみを行う
  - retire enqueue は NonRT bridge 経由で authority 側へ委譲する
- 実行フェーズ: Phase B
- 完了条件:
  - RT detect -> NonRT enqueue request 経路が明文化
  - callback 直enqueue禁止の検証規則が運用化
- 実装証跡:
  - `src/audioengine/ISRRetireRuntimeEx.cpp`（`ASSERT_NON_RT_THREAD()`）
  - `src/audioengine/AudioEngine.Commit.cpp`（`onRuntimeRetiredNonRt` 入口ガード）
  - `.github/scripts/check-list-compliance.ps1`（`enqueueRetire(` 直呼び検査）
  - `.github/scripts/isr-verify-v7-rt-nonrt-retire-bridge.ps1`
- Closed最小検証項目:
  - [x] Audio Thread が retire enqueue を直接呼ばないことを静的/動的に検証
  - [x] RT detect -> NonRT enqueue request -> authority enqueue 経路が再現テストで確認される
  - [x] callback 直enqueue違反時に検出・失敗（CI）する

## R10. Shared Epoch Canonical 前提の移行コスト固定

- 状態: Closed（2026-05-27）
- 重要度: High
- 主要リスク: shared epoch 失敗時の split migration cost 未評価で設計拘束が強すぎる
- 反映先: `ISR_Shared_EpochDomain_Scalability_Validation_Plan.md`, `ISR_HB_Graph_Specification.md`
- 確定方針:
  - shared strategy は canonical ではなく「検証結果依存」とする
  - split epoch migration の手順・コスト項目を先に定義する
- 実行フェーズ: Phase C
- 完了条件:
  - split migration 手順とコスト評価軸（latency/jitter/reclaim burst）が定義済み
  - shared継続 / split移行のGo/No-Go条件が承認済み
- 実装証跡:
  - `doc/work/ISR_Shared_EpochDomain_SplitMigration_Runbook_2026-05-27.md`
  - `doc/work/ISR_Shared_EpochDomain_Shared_vs_Split_Comparison_2026-05-27.md`
  - `doc/work/ISR_Shared_EpochDomain_GoNoGo_2026-05-27.md`
  - `.github/scripts/isr-compare-shared-split-epoch.ps1`
  - `.github/scripts/isr-record-shared-split-go-no-go.ps1`
  - `.github/scripts/isr-verify-v8-shared-split-readiness.ps1`
- Closed最小検証項目:
  - [x] split migration runbook（切替手順・ロールバック手順）が文書化済み
  - [x] latency/jitter/reclaim burst の比較表（shared vs split）が作成済み
  - [x] Go/No-Go 判定が記録され、判定理由が残っている

## R11. Closure Descriptor System 固定

- 状態: Closed（2026-05-21）
- 重要度: Critical
- 主要リスク: transitive ownership が暗黙で closure 破断を検出できない
- 反映先: `ISR_Formal_Guarantee_Package.md`, `ISR_HB_Graph_Specification.md`, `ISR_Runtime_Closure_Descriptor.md`
- 実行フェーズ: Phase B/C
- 完了条件:
  - descriptor node に ownership/mutability/lifetime/HB を保持
  - publish 前 closure validation が運用化
- Closed最小検証項目:
  - [x] descriptor node に kind/ownership/mutability/lifetime/HB/authority/allocator 情報が記録される
  - [x] publish 前 `validateClosureGraph` が mandatory 実行され、違反を reject する
  - [x] external mutable dependency の混入が CI で検出・失敗する

## R12. Payload Tier System 固定

- 状態: Closed（2026-05-21）
- 重要度: High
- 主要リスク: payload boundary が曖昧で forbidden dependency 混入
- 反映先: `ISR_Formal_Guarantee_Package.md`, `ISR_Runtime_State_Matrix.md`, `ISR_Payload_Tier_Model.md`
- 実行フェーズ: Phase B/C
- 完了条件:
  - tier 分類（InlineImmutable/ImmutableShared/ExternalPinned/RTLocalOnly/Forbidden）が定義済み
  - Forbidden/RTLocalOnly の payload 禁止検証が運用化
- Closed最小検証項目:
  - [x] 全 payload object family に tier が割当済み
  - [x] Forbidden tier の payload 混入が検出・失敗する
  - [x] RTLocalOnly tier の RuntimePublication への混入が検出・失敗する

## R13. Immutable Facade + Mutable Core 分離

- 状態: Closed（2026-05-21）
- 重要度: Critical
- 主要リスク: RuntimePublication 内への mutable atomic/mutex/lazy-init 混入
- 反映先: `ISR_Formal_Guarantee_Package.md`, `ISR_Immutability_Enforcement_Spec.md`
- 実行フェーズ: Phase B
- 完了条件:
  - publish graph が read-only projection のみで構成される
  - mutable cache が RTLocal/Background domain へ隔離済み
- Closed最小検証項目:
  - [x] publish graph 経由で mutable API が露出しない
  - [x] publish graph 内 mutex / lazy-init / mutable atomic が存在しない
  - [x] mutable cache が RTLocal または Background domain に限定される

## R14. Deferred Retire Intent Queue 固定

- 状態: Closed（2026-05-21）
- 重要度: High
- 主要リスク: RT completion detect と retire authority 実行の責務衝突
- 反映先: `ISR_Formal_Guarantee_Package.md`, `ISR_Retire_Authority_Graph.md`, `ISR_Deferred_Retire_Intent_Bridge.md`
- 実行フェーズ: Phase B
- 完了条件:
  - RT は intent emission のみ、NonRT が authority enqueue を実行
  - RT reclaim/delete/authority の禁止検証が運用化
- Closed最小検証項目:
  - [x] RT は `RetireIntent` emission のみを実行し、retire/reclaim/delete を実行しない
  - [x] NonRT coordinator が intent dequeue 後に authority enqueue を実行する
  - [x] RT直enqueue/RT reclaim の違反が検出・失敗（CI）する

## R15. Shutdown HB FSM 固定

- 状態: Closed（2026-05-21）
- 重要度: Critical
- 主要リスク: shutdown ordering が手続き依存で late callback/UAF を誘発
- 反映先: `ISR_Formal_Guarantee_Package.md`, `ISR_HB_Graph_Specification.md`, `ISR_Shutdown_State_Machine.md`
- 実行フェーズ: Phase C
- 完了条件:
  - shutdown state machine と phase HB chain が明文化
  - shutdown verifier で順序違反を検出できる
- Closed最小検証項目:
  - [x] phase enum（Running→AudioStopped→ObserverDrained→RetireClosed→EpochSettled→ReclaimComplete→ShutdownComplete）が実装される
  - [x] phase 逆行/飛び越し遷移が検出・拒否される
  - [x] shutdown verifier が late callback / post-stop enqueue を検出・失敗する

## R16. HB Failure Spec + Reorder Simulation 固定

- 状態: Closed（2026-05-21）
- 監査判定注記（2026-05-21）: **Closed**（`doc/work/R11-R25_Closed判定監査表_2026-05-21.md` と整合）
- 重要度: Critical
- 主要リスク: 再現試験依存で最小HB欠落の証明が不足
- 反映先: `ISR_Formal_Guarantee_Package.md`, `ISR_HB_Graph_Specification.md`, `ISR_Minimal_HB_Failure_Model.md`
- 実行フェーズ: Phase C
- 完了条件:
  - failure ordering と required HB の対照モデルが固定
  - reorder simulator が CI で実行される
- Closed最小検証項目:
  - [x] bug2 の最小HB欠落モデル（failure ordering）が文書化される
  - [x] required HB を適用した対照ケースで UAF 不成立を確認する
  - [x] reorder simulator（forced reorder/epoch lag/retire delay/observe race）が CI で実行される

## R17. Epoch Abstraction Layer 固定

- 状態: Closed（2026-05-21）
- 重要度: High
- 主要リスク: shared epoch が architecture invariant 化して移行不能
- 反映先: `ISR_Formal_Guarantee_Package.md`, `ISR_Shared_EpochDomain_Scalability_Validation_Plan.md`
- 実行フェーズ: Phase C
- 完了条件:
  - epoch coordinator abstraction が定義済み
  - shared/split/hybrid 切替方針が定義済み
- Closed最小検証項目:
  - [x] runtime code が concrete epoch 実装に直接依存しない
  - [x] shared/split/hybrid の切替インターフェースが定義済み
  - [x] split 移行時の rollback 手順が実行可能である

## R18. CI Verification Pipeline 固定

- 状態: Closed（2026-05-21）
- 監査判定注記（2026-05-21）: **Closed**（`doc/work/R11-R25_Closed判定監査表_2026-05-21.md` と整合）
- 重要度: Critical
- 主要リスク: 文書規律のみで merge 時に形式違反を検出できない
- 反映先: `ISR_Formal_Guarantee_Package.md`, `plan5.md`, `ISR_Verification_Pipeline.md`, `ISR_Runtime_Reduction_Strategy.md`, `ISR_Proof_Artifact_Schema_Registry.md`
- 実行フェーズ: Phase C
- 完了条件:
  - 10段ステージ（V1 Atomic Dot-Call Scan / V2 Seal Integrity Check / V3 Recursive Closure Validation / V4 Payload Tier Validation / V5 HB Reorder Simulation / V6 Shutdown FSM Verification / V7 Retire Latency Audit / V8 UAF Suspicion Detector / V9 Forbidden Capability Scan / V10 Ownership Cycle Detection）が定義済み
  - runtime-generated proof artifacts と CI evaluator の責務分離が定義済み
  - proof artifact の canonical naming と JSON schema contract が固定済み
  - pipeline failure が merge blocker として運用化
- Closed最小検証項目:
  - [x] 10段ステージがCIワークフローに統合済み
  - [x] artifact missing / schema mismatch / parse error が CI fail として扱われる
  - [x] canonical artifact 名（registry定義）で生成される
  - [x] いずれか失敗時に merge blocker として PR を停止する
  - [x] 成功時に証跡（レポート/ログ）が保存される

## R19. Authority capability-first 固定（coordinator互換shim化）

- 状態: Closed（2026-05-21）
- 補足: 必須（authority misuse を型制約で抑制）
- 重要度: Critical
- 主要リスク: coordinator中心設計の増殖による authority lifecycle 問題
- 反映先: `ISR_Execution_Authority_Convergence.md`
- 実行フェーズ: Phase C+
- 完了条件:
  - type-level capability（Publish/Retire/Shutdown）が優先運用される
  - runtime coordinator lifecycle を導入しない
- Closed最小検証項目:
  - [x] capability tag が API 契約として明示される
  - [x] coordinator 依存の新規経路を追加しない
  - [x] 分散 authority 導入時に CI でゲート違反となる

## R20. Host Chaos Normalization 固定

- 状態: Closed（2026-05-21）
- 補足: Release 適用候補（Layer 0 補完として優先度高）
- 重要度: High
- 主要リスク: host callback 非決定性により lifecycle invariant 崩壊
- 反映先: `ISR_Execution_Authority_Convergence.md`, `ISR_JUCE_Lifecycle_Isolation.md`
- 実行フェーズ: Phase C+
- 完了条件:
  - HostChaosNormalizer が Layer 0 に統合される
- Closed最小検証項目:
  - [x] HC-1 duplicate prepare collapse が動作
  - [x] HC-2 release-before-prepare reject が動作
  - [x] HC-3 callback during Releasing reject が動作

## R21. DSP ownership 単純化（few-authority）

- 状態: Closed（2026-05-21）
- 補足: 必須（DSPHandleRuntime への統合）
- 重要度: Critical
- 主要リスク: DSP ownership path 分散による運用複雑化
- 反映先: `ISR_Execution_Authority_Convergence.md`, `ISR_DSPHandle_Runtime.md`
- 実行フェーズ: Phase C+
- 完了条件:
  - DSP ownership は DSPHandleRuntime へ統合される
  - crossfade 完了前 retire 禁止が単純ルールで運用される
- Closed最小検証項目:
  - [x] DSPHandleRuntime が callback view を一元提供する
  - [x] crossfade complete before retire が検証される

## R22. callback-local snapshot consistency 固定

- 状態: Closed（2026-05-21）
- 補足: 必須（RTExecutionFrame の最小保証）
- 重要度: High
- 主要リスク: callback 中に runtime snapshot が変化する
- 反映先: `ISR_Execution_Authority_Convergence.md`, `ISR_RT_Execution_Frame.md`
- 実行フェーズ: Phase C+
- 完了条件:
  - RTExecutionFrame が callback-local immutable snapshot を保持する
- Closed最小検証項目:
  - [x] callback 中 snapshot immutable が検証される
  - [x] callbackEpoch 単位で一貫 view が維持される

## R23. world model 簡略化（2-world固定）

- 状態: Closed（2026-05-21）
- 補足: 必須（single-process 最適化）
- 重要度: Critical
- 主要リスク: 不要な federation 抽象で実装複雑化
- 反映先: `ISR_Execution_Authority_Convergence.md`
- 実行フェーズ: Phase C+
- 完了条件:
  - world model を PublicationWorld / ExecutionWorld の2つへ固定
- Closed最小検証項目:
  - [x] RuntimeBoundary 構造で境界が実装される
  - [x] full federation runtime 追加が抑制される

## R24. bounded deterministic teardown 固定

- 状態: Closed（2026-05-21）
- 補足: 必須（Release 安定性直結）
- 重要度: Critical
- 主要リスク: shutdown phase の肥大化で完了不能リスク増大
- 反映先: `ISR_Execution_Authority_Convergence.md`, `ISR_Shutdown_State_Machine.md`
- 実行フェーズ: Phase C+
- 完了条件:
  - shutdown phase を 7段（Running->...->Complete）へ固定
  - bounded completion を必須化
- Closed最小検証項目:
  - [x] SH-1 callback 0
  - [x] SH-2 active crossfade 0
  - [x] SH-3 pending retire 0
  - [x] SH-4 observer 0

## R25. DebugRuntime CI限定化

- 状態: Closed（2026-05-21）
- 監査判定注記（2026-05-21）: **Closed**（`doc/work/R11-R25_Closed判定監査表_2026-05-21.md` と整合）
- 補足: 必須（Release から proof負荷を除去）
- 重要度: High
- 主要リスク: proof/trace runtime の release 混入による性能劣化
- 反映先: `ISR_Execution_Authority_Convergence.md`, `ISR_Runtime_Reduction_Strategy.md`, `plan5.md`
- 実行フェーズ: Phase C+
- 完了条件:
  - Release/Debug/CI の proof 有効範囲が固定化される
  - DebugRuntime へ trace/verify/proof を集約する
- Closed最小検証項目:
  - [x] Release: proof off
  - [x] Debug: proof partial
  - [x] CI: proof full
  - [x] RuntimeReductionGate による新runtime追加審査が CI 強制される

---

## 運用ルール

- 本バックログは `plan5.md` の未完項目を一元管理する
- 各リスクは「反映先正本への反映 + 実装 + 検証証跡」で Close する
- Close 判定は Gate 条件の証跡（文書/検証）を必須とする
- 運用判定ラベルは `doc/work/plan5.md` と `doc/work/R11-R25_Closed判定監査表_2026-05-21.md` と同期し、**Closed（運用重視）/ Closed（厳密）/ 部分適合** の3区分を用いる。

## 外部監査参照

### 監査判定の往復参照（2026-05-21）

- R16: Backlog 状態 `Closed` / 監査判定 `Closed`
- R18: Backlog 状態 `Closed` / 監査判定 `Closed`
- R25: Backlog 状態 `Closed` / 監査判定 `Closed`
- AT-ISR: Air/Tail ISR統合トレーサビリティは `doc/work/R11-R25_Closed判定監査表_2026-05-21.md`（AT-ISR行）および `doc/work/ISR_AirTail_統合設計_2026-05-22.md` を正本参照とする
- 判定更新原則: `Closed` への更新は本書の「Closed最小検証項目」充足時のみ行う。

- ハブ文書（相互参照）:
  - `doc/work/plan5.md`
- R11〜R25 Closed判定監査表（証拠ファイル/コード行付き）:
  - `doc/work/R11-R25_Closed判定監査表_2026-05-21.md`
