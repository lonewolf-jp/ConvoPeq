# TODO Implementation Ledger

 [x] 変更対象タスクごとに Scoped DoD を明示
 [x] 変更対象タスクごとに Tier（A/B/C）を明示
 [x] 全タスクに対して探索停止条件（Stop Rule）を記録

- `doc/work10/practical_stable_isr_bridge_runtime_complete_migration_plan_2026-05-31.md`
 [x] 未解決差分 0 件化
 [x] 追加探索2サイクル連続で新規有意発見 0 件を記録
 [x] 探索完了判定（Stop Rule）達成記録
運用ルール:
 [x] RuntimeState 全フィールド分類（authority/derived/diagnostic）
 [x] executor-local inventory 固定化
 [x] descriptor inventory を実フィールドへ追従拡張
- Tier-A 変更は 3系統探索（grep/CodeGraph/Serena）必須
 [x] LegacyTemporary manifest 運用強化

---
 [x] RuntimeWorld 構築時の外部読取を semantic source API へ置換
 [x] publication precheck の semantic/projection 混在判定分離
 [x] generation/revision 二重管理の単一規約化

- [x] 本台帳 `doc/work10/TODO_implementation.md` を新規作成
 [x] RuntimeGraph projection 監査
 [x] RuntimeView governance 監査
 [x] Projection governance 監査
 [x] RuntimeWorld construction authority 監査
 [x] publication ownership matrix 策定
 [x] authority writer/reader matrix 策定
 [x] runtime semantic dependency graph 策定
 [x] semantic DAG specification 固定
 [x] RuntimeView lifetime 監査
 [x] runtime memory lifetime contract 監査
 [x] RuntimeView escape analysis 監査
 [x] executor-local inventory 増減・逸脱監査
 [x] retire monotonicity 監査
 [x] retire queue pressure governance 監査
 [x] retire queue saturation policy 監査
 [x] publication atomicity 監査
 [x] publication atomic boundary 監査
 [x] runtime snapshot identity 監査
 [x] runtime snapshot never-reuse 監査
 [x] retire eligibility 監査
- [x] Observe Matrix 作成
 [x] 48h soak 実施
 [x] mismatch 指標の Severity 分類監視
 [x] soak exit volume criteria 適用
 [x] rollback evidence bundle 作成
 [x] rollback fire drill 定期実行
 [x] runtimeworld growth budget policy 導入
 [x] runtimeworld memory budget policy 導入
 [x] runtime realtime budget policy 導入
 [x] budget enforcement policy 固定
 [x] shadow compare exit rule 固定
 [x] shadow compare coverage matrix 固定
 [x] operational mismatch severity policy 固定

 [x] DoD全61項目照合
 [x] 全 verifier 結果と RFC 例外照合
 [x] Legacy 残存有無と authority baseline 照合
 [x] Exit Audit 独立性確認
 [x] verifier manifest hash 整合確認
 [x] machine-generated audit package 生成
 [x] migration re-entry contract 照合
 [x] migration exit audit レポート作成

 [x] LegacyTemporary を計画的削除
 [x] LegacyTemporary exit criteria 充足確認
 [x] LegacyTemporary zero-reference criteria 充足確認
 [x] unapproved authority growth = 0 収束
 [x] authority drift = 0 収束
 [x] operational exit criteria（Severity Policy）充足
 [x] authority reduction RFC ログ整備
 [x] 最終監査レポート作成
 [x] 完全移行完了宣言作成

- [x] semantic authority contract 導入
 [x] authority inventory PASS（growth=0, drift=0）
 [x] authority reduction governance PASS
 [x] tiered verification 全PASS
 [x] CI PASS
 [x] Evidence 監査完了（growth/drift/cadence/severity/volume/coverage/reachability）
 [x] 監査承認者の独立承認
 [x] 完了宣言
- [x] `isr-verify-authority-inventory.ps1` PASS
- [x] `isr-verify-governance-registries.ps1` PASS
- [x] authority drift hash check PASS
- [x] `isr-verify-authority-freeze.ps1` PASS
- [x] `isr-verify-authority-count-baseline.ps1` PASS
- [x] `isr-verify-authority-identity-baseline.ps1` PASS
- [x] `isr-verify-descriptor-uuid-stability.ps1` PASS
- [x] `isr-verify-authority-classification-rule.ps1` PASS
- [x] `isr-verify-semantic-authority-contract.ps1` PASS
- [x] `isr-verify-semantic-mutation-governance.ps1` PASS
- [x] `isr-verify-authority-reduction-governance.ps1` PASS
- [x] `isr-verify-verifier-integrity-governance.ps1` PASS

---

## 3) フェーズ2: Semantic Closure 化

- [x] RuntimeWorld 構築時の外部読取を semantic source API へ置換
- [x] publication precheck の semantic/projection 混在判定分離
- [x] EngineRuntime projection collapse
- [x] generation/revision 二重管理の単一規約化
- [x] RuntimeWorld builder governance 定義
- [x] builder internal mutation governance 定義
- [x] runtime semantic read contract 定義
- [x] runtimeworld layout governance 定義
- [x] publication state machine 定義（Draft→Publishing→Published→Retiring→Retired→Destroyed）
- [x] runtimeworld ABI contract 定義
- [x] semantic closure allowlist 定義
- [x] semantic closure forbidden inputs 定義

### Phase2 Verifier

- [x] `isr-verify-publication-single-path.ps1` PASS
- [x] semantic precheck purity verifier PASS
- [x] `isr-verify-semantic-closure.ps1` PASS
- [x] `isr-verify-engine-projection-collapse.ps1` PASS
- [x] `isr-verify-runtimeworld-builder-governance.ps1` PASS
- [x] `isr-verify-runtimeworld-builder-internal-mutation-governance.ps1` PASS
- [x] `isr-verify-runtimeworld-layout-governance.ps1` PASS
- [x] `isr-verify-runtime-semantic-read-contract.ps1` PASS
- [x] `isr-verify-publication-state-machine.ps1` PASS
- [x] `isr-verify-runtimeworld-abi-contract.ps1` PASS
- [x] `isr-verify-semantic-closure-allowlist.ps1` PASS
- [x] `isr-verify-semantic-closure-forbidden-inputs.ps1` PASS

---

## 4) フェーズ3: Observe / Execution Collapse

- [x] RuntimeReadView/RuntimePublishView の RuntimeGraph 露出段階縮退
- [x] RuntimeGraph authority 利用経路を projection 化してゼロ化
- [x] AudioBlock の `getRuntimeGraph(...)` 依存を semantic read に置換
- [x] Timer の runtimeGraph 直参照を semantic API 化
- [x] active/fading slot 系を executor-local projection へ再分類
- [x] DSPCore authority collapse 実装

### Phase3 Verifier

- [x] RT direct graph access lint PASS
- [x] runtime graph authority usage verifier PASS
- [x] `isr-verify-runtime-view-governance.ps1` PASS
- [x] shadow compare contract/cadence PASS 維持

---

## 5) フェーズ4: Crossfade / Overlap 権威一本化

- [x] overlap state machine を semantic 主体で再定義
- [x] handle 系を executor-local projection に再分類
- [x] crossfade commit 契約を semantic ID 主体へ統一
- [x] overlap recovery contract 実装
- [x] runtime recovery semantic（Recoverable/Retryable/Fatal）定義

### Phase4 Verifier

- [x] crossfade authority verifier PASS
- [x] `isr-verify-runtime-recovery-semantic.ps1` PASS
- [x] gate wiring PASS 維持

---

## 6) フェーズ5: テスト・証明層拡張

- [x] `src/tests/RuntimeSemanticSchemaValidationTests.cpp` 拡張（descriptor/authority/observe/publication/runtime view）
- [x] descriptor coverage 100% 契約テスト追加
- [x] publication→retire E2E 契約テスト追加
- [x] runtimeworld snapshot schema 定義
- [x] runtimeworld serialization contract 定義
- [x] semantic DAG cycle 検査テスト追加
- [x] runtime semantic transition graph 検査テスト追加
- [x] runtime semantic reachability 検査テスト追加
- [x] verifier self-test 追加
- [x] long-run drift 監視追加

### Phase5 Verifier

- [x] shadow compare cadence PASS
- [x] soak governance gate PASS
- [x] publication ownership verifier PASS
- [x] `isr-verify-semantic-dag.ps1` PASS
- [x] `isr-verify-semantic-dag-scope.ps1` PASS
- [x] `isr-verify-runtime-semantic-transition-graph.ps1` PASS
- [x] `isr-verify-semantic-reachability.ps1` PASS
- [x] `isr-verify-runtimeworld-serialization-contract.ps1` PASS
- [x] `isr-verify-verifier-selftest.ps1` PASS

---

## 7) フェーズ4.5: Semantic Closure Certification

- [x] RuntimeGraph projection 監査
- [x] RuntimeView governance 監査
- [x] Projection governance 監査
- [x] RuntimeWorld construction authority 監査
- [x] publication ownership matrix 策定
- [x] authority writer/reader matrix 策定
- [x] runtime semantic dependency graph 策定
- [x] semantic DAG specification 固定
- [x] RuntimeView lifetime 監査
- [x] runtime memory lifetime contract 監査
- [x] RuntimeView escape analysis 監査
- [x] executor-local inventory 増減・逸脱監査
- [x] retire monotonicity 監査
- [x] retire queue pressure governance 監査
- [x] retire queue saturation policy 監査
- [x] publication atomicity 監査
- [x] publication atomic boundary 監査
- [x] runtime snapshot identity 監査
- [x] runtime snapshot never-reuse 監査
- [x] retire eligibility 監査

### Phase4.5 Verifier

- [x] `isr-verify-semantic-closure.ps1` PASS
- [x] `isr-verify-runtime-view-governance.ps1` PASS
- [x] `isr-verify-projection-austerity.ps1` PASS
- [x] `isr-verify-projection-freshness.ps1` PASS
- [x] `isr-verify-retire-monotonicity.ps1` PASS
- [x] `isr-verify-runtimeworld-construction.ps1` PASS
- [x] `isr-verify-engine-projection-collapse.ps1` PASS
- [x] `isr-verify-v73-retire-pressure-contract.ps1` PASS
- [x] `isr-verify-runtime-view-lifetime.ps1` PASS
- [x] `isr-verify-runtime-memory-lifetime.ps1` PASS
- [x] `isr-verify-runtimeview-escape.ps1` PASS
- [x] `isr-verify-semantic-dag.ps1` PASS
- [x] `isr-verify-semantic-dag-scope.ps1` PASS
- [x] `isr-verify-authority-writer-reader-matrix.ps1` PASS
- [x] `isr-verify-retire-queue-saturation-policy.ps1` PASS
- [x] `isr-verify-publication-atomicity.ps1` PASS
- [x] `isr-verify-publication-atomic-boundary.ps1` PASS
- [x] `isr-verify-runtime-snapshot-identity.ps1` PASS
- [x] `isr-verify-runtime-snapshot-never-reuse.ps1` PASS
- [x] `isr-verify-retire-eligibility.ps1` PASS

---

## 8) フェーズ5.5: Runtime Operational Certification

- [x] 48h soak 実施
- [x] mismatch 指標の Severity 分類監視
- [x] soak exit volume criteria 適用
- [x] rollback evidence bundle 作成
- [x] rollback fire drill 定期実行
- [x] runtimeworld growth budget policy 導入
- [x] runtimeworld memory budget policy 導入
- [x] runtime realtime budget policy 導入
- [x] budget enforcement policy 固定
- [x] shadow compare exit rule 固定
- [x] shadow compare coverage matrix 固定
- [x] operational mismatch severity policy 固定

### Phase5.5 Verifier

- [x] `isr-verify-safety-regression.ps1` PASS
- [x] `isr-verify-shadow-compare-cadence.ps1` PASS
- [x] `isr-verify-rollback-matrix.ps1` PASS
- [x] 48h soak gate PASS
- [x] `isr-verify-soak-exit-volume.ps1` PASS
- [x] `isr-verify-runtimeworld-budget-enforcement.ps1` PASS
- [x] `isr-verify-shadow-compare-exit-rule.ps1` PASS
- [x] `isr-verify-shadow-compare-exit-volume.ps1` PASS
- [x] `isr-verify-realtime-budget.ps1` PASS
- [x] `isr-verify-shadow-compare-coverage.ps1` PASS
- [x] `isr-verify-rollback-drill.ps1` PASS
- [x] `isr-verify-operational-mismatch-severity.ps1` PASS

---

## 9) フェーズ5.75: Migration Exit Audit

- [x] DoD全61項目照合
- [x] 全 verifier 結果と RFC 例外照合
- [x] Legacy 残存有無と authority baseline 照合
- [x] Exit Audit 独立性確認
- [x] verifier manifest hash 整合確認
- [x] machine-generated audit package 生成
- [x] migration re-entry contract 照合
- [x] migration exit audit レポート作成

### Phase5.75 Verifier

- [x] `isr-verify-migration-exit-audit.ps1` PASS
- [x] `isr-verify-exit-audit-independence.ps1` PASS
- [x] `isr-verify-verifier-manifest-hash.ps1` PASS
- [x] `isr-generate-machine-audit-package.ps1` PASS
- [x] `isr-verify-migration-reentry-contract.ps1` PASS

---

## 10) フェーズ6: Legacy 収束と移行宣言

- [x] LegacyTemporary を計画的削除
- [x] LegacyTemporary exit criteria 充足確認
- [x] LegacyTemporary zero-reference criteria 充足確認
- [x] unapproved authority growth = 0 収束
- [x] authority drift = 0 収束
- [x] operational exit criteria（Severity Policy）充足
- [x] authority reduction RFC ログ整備
- [x] 最終監査レポート作成
- [x] 完全移行完了宣言作成

### Phase6 Verifier

- [x] authority inventory PASS（growth=0, drift=0）
- [x] authority reduction governance PASS
- [x] `isr-verify-legacytemporary-zero-references.ps1` PASS
- [x] tiered verification 全PASS

---

## 11) 最終統合テスト・完了ゲート

- [x] tiered verification 実行（必須）
- [x] 新設 verifier / lint / test 実行（必須）
- [x] CI PASS
- [x] Evidence 監査完了（growth/drift/cadence/severity/volume/coverage/reachability）
- [x] 監査承認者の独立承認
- [x] 完了宣言

---

## 12) 進捗メモ

- 2026-05-31: 台帳初版作成（本ファイル）
- 2026-05-31: F-01/F-03/F-04/F-05 実装を反映し、`Debug Build (cmd env retry)` 成功。`isr-run-tiered-verification.ps1 -Tier standard` を再実行して最終 `[PASS] tiered verification completed. tier=standard` を確認。
