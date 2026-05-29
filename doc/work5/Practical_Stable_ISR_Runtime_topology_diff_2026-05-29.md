# Practical Stable ISR Runtime Topology Diff (2026-05-29)

## Topology Diff

この文書は ISR Runtime の topology 差分を、統治規約の提出要件（Documentation Scope Rule）に沿って記録する。

## Scope

- 対象期間: 2026-05-29
- 対象: `.github/scripts` の検証ゲート運用強化
- 非対象: `src/` の実行時アルゴリズム変更

## Delta Summary

- `documentation_scope_rule` に topology差分文書の必須チェックを追加。
- authority inventory 成果物（current/post/diff/report）を必須 artifact として明示。
- gate wiring の自己検査契約に上記チェックを追加。
- `RuntimePublicationCoordinator` の遷移ガードを明確化（`Ready -> Transitioning` / `Transitioning -> Ready` のみ許可）。
- `isr-verify-runtime-coordinator-state-machine.ps1` に遷移ガード契約（Ready/Transitioning 以外は reject）を追加。
- `.github/isr-ai-governance-policy.json` の `residencyTelemetryChecks.requiredTelemetryApplications` に fallback enqueue 即時同期契約を追加。
- `isr-verify-v73-residency-telemetry.ps1` に policyカバレッジ検査（`CI-TELEMETRY-010`）を追加し、実装契約とpolicy宣言のドリフトをfail-closed化。
- `isr-verify-documentation-scope-rule.ps1` に役割別文書の単一解決必須チェックを追加（token一致候補が複数/0なら fail）。
- `isr-verify-gate-wiring.ps1` に上記単一解決契約の自己検査を追加。
- `isr-verify-pr-required-artifacts.ps1` に documentation scope report の `ready=true` / 必須 role 完備チェックを追加。
- `RuntimePublicationCoordinator::markShutdownComplete()` を `ShuttingDown` 状態限定にし、遷移規則（ShuttingDown→terminal）を明示化。
- `isr-verify-taxonomy-phase-mapping.ps1` を追加し、Failure Taxonomy ↔ Phase DoD の対応と失敗時アクション規約を機械検証化。
- `isr-verify-safety-regression.ps1` にノイズ許容規約（1指標のみ +3% 以内、3回再計測中央値）を追加し、`noiseAllowancePolicy` としてレポート化。
- `.github/workflows/isr-verification.yml` の upload-artifact に documentation/taxonomy/state-machine 関連 evidence を追加。
- `isr-verify-gate-wiring.ps1` に workflow artifact 搬送契約の自己検査を追加。
- `isr-verify-pr-required-artifacts.ps1` に `runtime_coordinator_state_machine_report.json` の必須提出/ready検査を追加。
- `isr-verify-pr-sla.ps1` に `requiredNotes` の意味論検証を追加し、`Class-S/A` は inventory diff structural invariant、`Class-B/C` は soak 分数、`Class-D` は BreakGlass report ready を fail-closed 化。
- `.github/workflows/isr-verification.yml` / `isr-run-tiered-verification.ps1` / `isr-workflow-dispatch-input-policy.json` に `declaredPrClass` / `soakMinutes` の forwarding 契約を追加し、manual dispatch 時の `Class-A` 既定固定を緩和。
- `isr-verify-pr-sla.ps1` に `Class-S` の `runtime code change zero` 検知を追加（PR event の base/head git diff による `src/` 変更検出、検知不能時は inventory fallback を明示）し、`runtimeCodeChangeZeroRequired` / `inventoryDiffStructuralInvariantRequired` の policy bool も fail-closed 検証へ昇格。
- `isr-verify-safety-regression.ps1` に Class-D/Class-E の「連続2窓悪化で fail」規約を追加し、`safety_failure_window_history.json` に streak を記録。単発悪化は warning 扱い、2窓連続で fail-closed とする taxonomy-window policy を実装。
- `isr-verify-pr-sla.ps1` に PRクラス宣言必須化を追加（`RequireDeclaredClass`）。`pull_request` 実行時は tier runner から必須化し、`isr-pr-class:Class-*` ラベル解決に失敗した場合は fail-closed とする。
- `isr-verify-gate-wiring.ps1` / `isr-verify-pr-required-artifacts.ps1` を拡張し、`needs-revalidation` ラベル付与フロー（`report.needsRevalidation` + `labelSuggestions`）と `pr_sla_report.json` の SLA再検証フィールド整合を fail-closed で自己検査する契約を追加。
- `isr-verify-pr-sla.ps1` に `Resolve-OpenedAtFromEvent` を追加し、`OpenedAt` 未指定時でも `GITHUB_EVENT_PATH` の `pull_request.created_at` から SLA 期限判定を可能化。`openedAtSource` をレポート化して、期限超過時の `needs-revalidation` 判定を監査可能化。
- `isr-verify-pr-sla.ps1` に PR head freshness 判定（`Resolve-EventHeadSha` + current HEAD 比較）を追加し、評価対象コミット不一致時は `staleEvaluation=true`・`needs-revalidation` 付与・fail-closed で再評価を強制。`eventHeadSha/currentHeadSha` を証跡化。
- `isr-verify-design-docs-coverage.ps1` を新設し、4設計図書（基本計画/規約/詳細設計/タスク分解）の存在と必須トークンを機械検証。`design_docs_coverage_report.json` を PR成果物・workflow upload・gate wiring 契約へ統合し、設計図書準拠を fail-closed 化。

## Runtime Meaning Source Impact

### authority source

- 変更なし（増加なし）
- authority source 自体は不変だが、`RuntimePublicationCoordinator` の状態遷移ガードを強化

### observe path

- 変更なし（増加なし）
- Audio Thread observe 経路への追加なし

### publication path

- `publish(RuntimeWorld*)` 単一路への影響なし
- publish 周辺状態遷移の前提を強化（不正 state からの transition 要求を reject）

### retire ownership

- 変更なし
- retire lane / reclaim ownership への影響なし

## Governance Budget Check (Documentation Delta)

- Authority Migration Budget: pass（増加 0）
- Observe Growth Budget: pass（増加 0）
- Semantic Duplication Budget: pass（新規 semantic source 追加なし）
- Legacy Lifetime Cap: pass（対象外）

## Related Artifacts

- `storage/isr_inventory/current_authority_inventory.json`
- `storage/isr_inventory/post_authority_inventory.json`
- `storage/isr_inventory/inventory_diff_report.json`
- `evidence/authority_inventory_report.json`

## Notes

この文書は topology差分文書の単一提出先として運用し、runtime semantics 差分が生じた場合は同ファイルを追記または新日付版を追加する。
