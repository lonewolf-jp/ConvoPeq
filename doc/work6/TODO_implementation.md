# TODO implementation (isr_bridge_runtime)

## 実装タスク

- [x] 現行ISR実装の構造把握（schema/registry/verifier/run-tier wiring）
- [x] Schema/契約不足の特定（v1.6との差分抽出）
- [x] RuntimeSemanticSchema拡張実装（`RuntimeSemanticSchema` 集約体、`generationSemanticHash` 追加）
- [x] Publication契約実装強化（schema precheck/mapping token検証強化）
- [x] Retire/Projection契約実装（retire lifecycle/projection freshness系ゲートの稼働確認）
- [x] Shadow compare/Hash契約実装（hash coverage を generation semantic 含めて verifier で拘束）
- [x] Verifier/台帳整合更新（契約検証スクリプト更新と整合確認）

## テスト/検証タスク

- [x] `isr-verify-shadow-compare-contract.ps1` 単体 PASS
- [x] `isr-verify-runtime-semantic-schema-v16.ps1` 単体 PASS
- [x] `isr-run-tiered-verification.ps1 -Tier standard` PASS
- [x] Debug build (`cmake --build build --config Debug`) PASS

## 設計図書網羅マップ（主要契約）

- [x] Base plan v2.3: RuntimeSemanticSchema / authority singularity
- [x] Base plan v2.3: publication single-source / monotonic / mapping
- [x] Base plan v2.3: projection non-authority / freshness / austerity
- [x] Base plan v2.3: shadow compare contract / cadence / hash contract
- [x] Base plan v2.3: retire lifecycle / pressure / starvation
- [x] Governance v1.2: fail-closed verifier運用
- [x] Governance v1.2: AuthorityClass分類強制（未分類禁止）
- [x] Governance v1.2: hash authority prohibition
- [x] Detailed design v1.6: semantic category taxonomy対応
- [x] Detailed design v1.6: RuntimeSemanticHash coverage（generation semantic含む）
- [x] Detailed design v1.6: 設計文書カバレッジ証跡 (`design_docs_coverage_report.json`)
