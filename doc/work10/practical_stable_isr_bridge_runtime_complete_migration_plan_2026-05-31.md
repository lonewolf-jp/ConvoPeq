# Practical Stable ISR Bridge Runtime 完全移行詳細計画（2026-05-31）

## 1. 目的

`doc/work10/notfinished_validation_report_2026-05-31.md` を根拠に、ConvoPeq を **Practical Stable ISR Bridge Runtime 完全到達** 状態へ移行するための実行計画を定義する。

本計画は、実装完了だけでなく、CI・統治・運用監視を含む継続的証明可能な完了条件を対象とする。

## 2. 根拠（監査結論の要点）

- 達成済み寄り
  - Publication / Retire / Freeze / tiered governance 骨格
- 未達中心
  - Authority Collapse（Execution / Observe / Crossfade）
  - Semantic Closure（RuntimeWorld の自己完結）
  - Descriptor 完全性統治（実体フィールドとの一致）
- 追加監査の確定判定
  - LegacyTemporary 統治は「問題としては未確定（懸念は解消寄り）」
  - Shadow Compare の release gate 統合は「問題としては否（統合済み）」

## 3. 完了条件（Definition of Done）

以下を全て満たした時点で「完全移行完了」と判定する。

1. **Single Semantic Authority**: Audio Thread の実行判断が RuntimeWorld semantic のみで決定可能
2. **Observe Collapse 完了**: RT パスで `RuntimeGraph*` 直参照と Engine atomic 直参照がゼロ
3. **RuntimeGraph Projection Collapse 完了**: `RuntimeGraph` が Execution / Observe / Crossfade の authority 判定に使われる箇所がゼロ
4. **Descriptor Closure 完了**: RuntimeState authority field と descriptor inventory が 1:1 で一致
5. **Executor Local Inventory 固定化**: executor-local state が inventory で固定・増殖監視される
6. **Publication Contract 単一路化**: publish / retire の更新経路が Coordinator 管轄に収束
7. **Crossfade Authority 単一化**: overlap / crossfade の権威が semantic 側へ一本化
8. **Semantic Closure Complete**: RuntimeWorld 外状態を参照せず Runtime 意味論（Execution / Observe / Overlap / Publication / Retire）を決定可能
9. **RuntimeView Governance 完了**: View は authority を保持せず、projection のみ公開し、mutable API を持たない
10. **Fail-closed Governance**: tiered verification + 追加 verifier が CI で常時強制
11. **Construction Authority Complete**: RuntimeWorld 生成経路が BuilderToken 経路のみに限定される
12. **EngineRuntime Projection Collapse 完了**: `EngineRuntime` が authority 判定へ使われる箇所がゼロ
13. **DSPCore Authority Zero**: DSPCore は executor としてのみ機能し、authority を保持しない
14. **Descriptor Coverage 100%**: Authority Field Count = Descriptor Count かつ Authority Field UUID = Descriptor UUID
15. **Publication Ownership Matrix 固定化**: runtime publication 変更責務が owner 単位で固定され verifier と一致
16. **Operational Exit Criteria 達成**: 48h soak で mismatch を Severity 管理し Critical/Major=0 を満たす
17. **Runtime Version Taxonomy 固定化**: Generation / Revision / PublicationSequence / Epoch の責務が固定される
18. **Authority Freeze Rule 適用**: Authority の削除・移動・再分類は RFC 承認なしで実施できない
19. **Semantic Ownership Matrix 固定化**: 各 Semantic の owner と責務境界が固定される
20. **RuntimeView Lifetime Governance 完了**: View は publish 境界を越えず、retire 後参照禁止を満たす
21. **Publication→Retire E2E Contract 完了**: Publish→Observe→Retire が機械検証される
22. **Semantic DAG Verified**: Runtime semantic dependency graph が循環依存ゼロで維持される
23. **Authority Count Baseline 固定化**: Expected Authority Count / Expected Descriptor Count が基線化され逸脱不可
24. **Authority Identity Baseline 固定化**: Authority UUID Set / Name Set / Semantic Class Set の差分を検知
25. **Descriptor UUID Stability**: Descriptor UUID は互換性契約なしで変更できない
26. **Builder Governance 完了**: RuntimeWorld Builder API 追加・Authority 注入変更は RFC 必須
27. **Semantic Schema Versioning 適用**: Runtime semantic schema のバージョン運用（vN）と移行契約を固定
28. **Soak Exit Volume Criteria 達成**: 48h soak で最小 publish/retire/crossfade 件数を満たす
29. **Budget Enforcement 固定化**: RuntimeWorld budget 超過時の動作（Fail/Warning）を明示し gate で強制
30. **Semantic Mutation Governance 適用**: semantic 追加/削除/統合/分割は RFC 承認なしで実施できない
31. **RuntimeWorld Layout Governance 適用**: RuntimeWorld 直下フィールド追加は RFC 必須
32. **Semantic Read Contract 完了**: Runtime semantic の読み取り窓口を固定し、無秩序な深掘り参照を禁止
33. **Publication Atomicity Contract 完了**: publication visibility は「完全可視 / 完全不可視」のみを許容
34. **Runtime Snapshot Identity Contract 完了**: snapshot identity は一意で再利用不可（Never Reuse）
35. **Retire Eligibility Contract 完了**: retire は未参照条件（Audio / Observe / Crossfade / ShadowCompare / DebugSnapshot / DiagnosticDump）を満たす場合のみ許可
36. **Shadow Compare Exit Rule 固定化**: Shadow Compare の除去条件（期間・不一致ゼロ・最低イベント数）を明示
37. **Authority Classification Rule 固定化**: authority / derived / diagnostic の分類基準を明文化
38. **Verifier Integrity Governance 適用**: verifier 変更/削除/緩和は RFC 必須
39. **Independent Exit Audit 適用**: Exit Audit 実施者は実装者と分離（self-certification禁止）
40. **Semantic DAG Specification 固定化**: DAG の Scope / Node / Edge 定義を固定する
41. **Semantic Closure Allowlist 固定化**: Semantic Authority Decision に使ってよい外部入力を明示する
42. **Semantic Closure Forbidden Inputs 固定化**: Semantic Authority Decision で絶対に参照禁止の入力種別を固定する
43. **Authority Writer/Reader Matrix 固定化**: Authority ごとに Writer/Reader/Update API/Update Phase を固定する
44. **RuntimeView Escape Analysis 適用**: capture lambda / async task 経由の retire 後アクセスを静的検査で禁止する
45. **Shadow Compare Coverage Matrix 適用**: Publish/Retire/Crossfade/Observe/Rollback/Recovery の全経路通過を終了条件に含める
46. **Rollback Fire Drill Test 適用**: 定期的な rollback 実施テストを必須化し、未使用経路の破損を防ぐ
47. **Realtime Budget 固定化**: Publish Time / Retire Time / Crossfade Commit Time の予算を定義し gate で強制する
48. **Runtime Semantic Transition Graph Verified**: 条件付き遷移を含む実行時グラフで循環・逆流を検査する
49. **Machine-generated Audit Package 適用**: Exit Audit 用の証跡パッケージを自動生成し人依存を低減する
50. **LegacyTemporary Zero-Reference Criteria 適用**: Reader/Writer/Reference Count をゼロ化してから除去を許可する
51. **Operational Mismatch Severity Governance 適用**: 48h mismatch を Severity 別に管理し Critical/Major=0 を必須化する
52. **Semantic Authority Contract 固定化**: Semantic ごとに Owner/Authoritative Writer/Reader/Update Phase/Allowed&Forbidden Dependency/Source&Derived Authority を固定する
53. **Publication State Machine 固定化**: Draft/Publishing/Published/Retiring/Retired/Destroyed の遷移を契約化し逸脱を禁止する
54. **RuntimeWorld ABI Contract 固定化**: フィールド順序・型・配置・互換ルールを固定し ABI 破壊を検知する
55. **Verifier Self-Test Framework 適用**: verifier が違反サンプルを確実に FAIL できることを継続検証する
56. **RuntimeWorld Serialization Contract 固定化**: field order/type/optionality/version migration を固定し dump/audit 互換を守る
57. **Runtime Recovery Semantic 固定化**: Publication/Retire/Crossfade/Shadow failure に対する Recoverable/Retryable/Fatal を定義する
58. **Migration Re-entry Contract 適用**: 移行完了後の Authority/Semantic/Descriptor/Builder 変更時の DoD 再評価条件を固定する
59. **Runtime Semantic Reachability Verified**: 主要終端状態（例: CrossfadeComplete）への到達可能性を機械検証する
60. **Runtime Memory Lifetime Contract 固定化**: RuntimeWorld/DescriptorInventory/PublicationMetadata/CrossfadeMetadata の Create→Publish→Observe→Retire→Destroy を固定する
61. **Authority Reduction Governance 適用**: Authority 増減を RFC 管理し、削減を合法かつ追跡可能に扱う

## 4. 実行フェーズ

## フェーズ1: Authority Inventory 固定化

### フェーズ1の目的

authority / derived / diagnostic の境界を固定し、以降の実装変更の基準を確立する。

### フェーズ1の作業

- `RuntimeState` 全フィールドを分類し、authority inventory を更新
- executor-local inventory を固定化
- `kFieldDescriptors` を実フィールドに追従拡張
- authority inventory hash を固定し、growth だけでなく drift も検出可能にする
- authority freeze rule を導入（Authority 削除/移動/再分類は RFC 必須）
- authority count baseline（Expected Authority Count / Expected Descriptor Count）を固定
- authority identity baseline（Authority UUID / Name / Semantic Class）を固定
- descriptor UUID stability contract を導入（UUID変更は互換性手順必須）
- authority classification rule を導入（分類判定基準の統一）
- semantic authority contract を導入（Owner/Writer/Reader/Update Phase/Dependency/Authority 種別を固定）
- semantic mutation governance を導入（semantic 追加/削除/統合/分割は RFC 必須）
- authority reduction governance を導入（削減を RFC 手順で許可し、基線更新を監査可能化）
- verifier integrity governance を導入（verifier 変更/削除/緩和は RFC 必須）
- `LegacyTemporary` manifest の運用規約を明文化・検証強化

### フェーズ1の成果物

- authority inventory 更新差分
- executor-local inventory 更新差分
- descriptor inventory 更新差分
- authority inventory hash（基準値）
- authority_count_baseline.md
- authority_identity_baseline.md
- descriptor_uuid_stability_contract.md
- runtime_version_taxonomy.md
- runtime_semantic_ownership_matrix.md
- runtime_semantic_schema_versioning.md
- runtime_semantic_compatibility_matrix.md
- authority_classification_rule.md
- semantic_authority_contract.md
- semantic_mutation_governance.md
- authority_reduction_governance.md
- verifier_integrity_governance.md
- 監査ログ（増減理由）

### フェーズ1のゲート

- `isr-verify-authority-inventory.ps1` PASS
- `isr-verify-governance-registries.ps1` PASS
- （拡張）authority drift hash check PASS
- （新設）`isr-verify-authority-freeze.ps1` PASS
- （新設）`isr-verify-authority-count-baseline.ps1` PASS
- （新設）`isr-verify-authority-identity-baseline.ps1` PASS
- （新設）`isr-verify-descriptor-uuid-stability.ps1` PASS
- （新設）`isr-verify-authority-classification-rule.ps1` PASS
- （新設）`isr-verify-semantic-authority-contract.ps1` PASS
- （新設）`isr-verify-semantic-mutation-governance.ps1` PASS
- （新設）`isr-verify-authority-reduction-governance.ps1` PASS
- （新設）`isr-verify-verifier-integrity-governance.ps1` PASS

## フェーズ2: Semantic Closure 化（外部依存の逆流停止）

### フェーズ2の目的

`buildRuntimePublishWorld(...)` が外部 snapshot / atomic に依存する構造を段階排除し、RuntimeWorld 自己完結性を高める。

### フェーズ2の作業

- RuntimeWorld 構築時の外部読取を semantic source API へ置換
- `runPublicationPrecheckNonRt(...)` の semantic / projection 混在判定を分離
- `EngineRuntime` の projection collapse（`world.engine.*` 依存の段階排除）
- generation / revision の二重管理を単一規約へ収束
- RuntimeWorld builder governance を定義（Builder API追加・Authority注入変更はRFC必須）
- builder internal mutation governance を定義（内部 setter / inject 経路変更は RFC 必須）
- runtime semantic read contract を定義（読み取り窓口固定）
- runtimeworld layout governance を定義（直下フィールド追加はRFC必須）
- publication state machine を定義（Draft→Publishing→Published→Retiring→Retired→Destroyed）
- runtimeworld ABI contract を定義（フィールド順序/型/配置の互換契約）
- semantic closure allowlist を定義（許可外入力は拒否）
- semantic closure forbidden inputs を定義（禁止入力は fail-closed）

### フェーズ2の成果物

- semantic-only precheck
- projection-only validation
- generation 単調性ルール
- runtimeworld_builder_governance.md
- runtimeworld_builder_internal_mutation_governance.md
- runtime_semantic_read_contract.md
- runtimeworld_layout_governance.md
- publication_state_machine.md
- runtimeworld_abi_contract.md
- semantic_closure_allowed_external_inputs.md
- semantic_closure_forbidden_inputs.md

### フェーズ2のゲート

- `isr-verify-publication-single-path.ps1` PASS
- （新設）semantic precheck purity verifier PASS
- （新設）`isr-verify-semantic-closure.ps1` PASS
- （新設）`isr-verify-engine-projection-collapse.ps1` PASS
- （新設）`isr-verify-runtimeworld-builder-governance.ps1` PASS
- （新設）`isr-verify-runtimeworld-builder-internal-mutation-governance.ps1` PASS
- （新設）`isr-verify-runtimeworld-layout-governance.ps1` PASS
- （新設）`isr-verify-runtime-semantic-read-contract.ps1` PASS
- （新設）`isr-verify-publication-state-machine.ps1` PASS
- （新設）`isr-verify-runtimeworld-abi-contract.ps1` PASS
- （新設）`isr-verify-semantic-closure-allowlist.ps1` PASS
- （新設）`isr-verify-semantic-closure-forbidden-inputs.ps1` PASS

## フェーズ3: Observe / Execution Collapse

### フェーズ3の目的

RT 観測源を RuntimeWorld に一本化し、実行権威の分散を解消する。

### フェーズ3の作業

- `RuntimeReadView` / `RuntimePublishView` の `RuntimeGraph*` 露出を段階縮退
- `RuntimeGraph` authority 利用経路をゼロ化（projection化）
- `AudioEngine.Processing.AudioBlock.cpp` の `getRuntimeGraph(...)` 依存を semantic read 経由へ置換
- Timer 監視の `runtimeGraph->...` 直参照を semantic API 化
- active / fading slot 系を projection（executor-local）へ格下げ
- DSPCore authority collapse（runtime意味論をDSPCoreから剥離し executor 専用化）

### フェーズ3の成果物

- observe source map（旧→新）
- 直参照ゼロ化パッチ
- DSPCore authority zero 監査レポート

### フェーズ3のゲート

- （新設）RT direct graph access lint PASS
- （新設）runtime graph authority usage verifier PASS
- （新設）`isr-verify-runtime-view-governance.ps1` PASS
- shadow compare contract / cadence PASS 維持

## フェーズ4: Crossfade / Overlap 権威一本化

### フェーズ4の目的

`dspHandleRuntime_` / `crossfadeAuthorityRuntime_` / `activeCrossfadeId_` の並列権威を semantic overlap に収束させる。

### フェーズ4の作業

- overlap state machine（開始 / 進行 / 完了 / 異常）を semantic 主体に再定義
- handle 系は executor-local projection に再分類
- commit 時の crossfade 契約を semantic ID 主体へ統一
- crossfade 異常復旧契約（途中失敗時の overlap semantic 回復手順）を定義
- runtime recovery semantic（Recoverable/Retryable/Fatal）を定義し、個別 recovery を統一する

### フェーズ4の成果物

- overlap 遷移仕様
- overlap_recovery_contract.md
- runtime_recovery_semantic.md
- crossfade authority single-source 実装

### フェーズ4のゲート

- （新設）crossfade authority verifier PASS
- （新設）`isr-verify-runtime-recovery-semantic.ps1` PASS
- 既存 gate wiring PASS 維持

## フェーズ5: テスト・証明層拡張

### フェーズ5の目的

「壊れていない」を継続的に証明する。

### フェーズ5の作業

- `src/tests/RuntimeSemanticSchemaValidationTests.cpp` 拡張（descriptor 完全一致、authority singularization 不変条件、observe collapse 契約、publication ownership 契約、runtime view 契約）
- descriptor coverage 100% 契約テスト（Field Count一致 + UUID一致）
- publication_retire_lifecycle_contract.md を定義し E2E 契約テストを追加
- runtimeworld_snapshot_schema.md を定義し監査/soak用 dump schema を固定化
- runtimeworld serialization contract を定義し dump/audit の互換境界を固定化
- semantic dependency cycle 検査テストを追加（DAG 保証）
- runtime semantic transition graph 検査テストを追加（条件付き遷移の循環/逆流検出）
- runtime semantic reachability 検査テストを追加（主要終端状態への到達性保証）
- verifier self-test を追加（違反サンプルで FAIL を確認）
- long-run drift 監視（soak evidence）追加

### フェーズ5の成果物

- schema / authority / observe / crossfade テスト群
- publication_retire_lifecycle_contract.md
- runtimeworld_snapshot_schema.md
- runtimeworld_serialization_contract.md
- runtime_semantic_transition_graph.md
- runtime_semantic_reachability.md
- verifier_selftest_report.md
- soak evidence JSON と解析レポート

### フェーズ5のゲート

- 既存 shadow compare cadence PASS
- （新設）soak governance gate PASS
- （新設）publication ownership verifier PASS
- （新設）`isr-verify-semantic-dag.ps1` PASS
- （新設）`isr-verify-semantic-dag-scope.ps1` PASS
- （新設）`isr-verify-runtime-semantic-transition-graph.ps1` PASS
- （新設）`isr-verify-semantic-reachability.ps1` PASS
- （新設）`isr-verify-runtimeworld-serialization-contract.ps1` PASS
- （新設）`isr-verify-verifier-selftest.ps1` PASS

## フェーズ4.5: Semantic Closure Certification

### フェーズ4.5の目的

RuntimeWorld 完了判定を人力判定から機械判定へ移し、Projection層の再劣化を防止する。

### フェーズ4.5の作業

- RuntimeGraph projection 監査（authority使用ゼロ）
- RuntimeView 統治監査（authority保持禁止・mutable API禁止）
- Projection governance 監査（projectionの publish / semantic source 化禁止）
- RuntimeWorld construction authority 監査（BuilderToken経路以外を禁止）
- publication ownership matrix を策定し owner 書き込み責務を固定
- authority writer/reader matrix を策定し writer/reader/update api/update phase を固定
- Runtime semantic dependency graph を策定し逆方向依存を禁止
- semantic DAG specification（scope/node/edge）を固定
- RuntimeView lifetime 監査（publish境界外利用禁止、retire後参照禁止）
- runtime memory lifetime contract 監査（RuntimeWorld/DescriptorInventory/PublicationMetadata/CrossfadeMetadata）
- RuntimeView escape analysis（capture lambda / async task 経由の逃避禁止）
- executor-local inventory 増減・逸脱監査
- retire monotonicity 監査（publication generation / retire generation / epoch generation の逆転禁止）
- retire queue pressure governance（Backlog SLO / QueuePressure対応）監査
- retire queue saturation policy（QueueDepth超過時の Fail/Warning/Drop 行動）監査
- publication atomicity 監査（中間可視状態の禁止）
- publication atomic boundary 監査（RuntimeWorld/DescriptorInventory/PublicationMetadata/CrossfadeMetadata の境界を固定）
- runtime snapshot identity 一意性監査
- runtime snapshot identity never-reuse 監査
- retire eligibility 監査（参照中オブジェクトの retire 禁止）

### フェーズ4.5の成果物

- semantic closure certification report
- runtime view governance report
- projection governance report
- runtimeworld construction authority report
- publication ownership matrix
- authority_writer_reader_matrix.md
- runtime semantic dependency graph
- semantic_dag_specification.md
- runtime view lifetime governance report
- runtime_memory_lifetime_contract.md
- runtimeview_escape_analysis_report.md
- retire monotonicity report
- retire backlog SLO report
- retire_queue_saturation_policy.md
- publication_atomicity_contract.md
- publication_atomic_boundary_contract.md
- runtime_snapshot_identity_contract.md
- retire_eligibility_contract.md

### フェーズ4.5のゲート

- `isr-verify-semantic-closure.ps1` PASS
- `isr-verify-runtime-view-governance.ps1` PASS
- `isr-verify-projection-austerity.ps1` PASS
- `isr-verify-projection-freshness.ps1` PASS
- `isr-verify-retire-monotonicity.ps1` PASS
- `isr-verify-runtimeworld-construction.ps1` PASS
- `isr-verify-engine-projection-collapse.ps1` PASS
- `isr-verify-v73-retire-pressure-contract.ps1` PASS
- `isr-verify-runtime-view-lifetime.ps1` PASS
- `isr-verify-runtime-memory-lifetime.ps1` PASS
- `isr-verify-runtimeview-escape.ps1` PASS
- `isr-verify-semantic-dag.ps1` PASS
- `isr-verify-semantic-dag-scope.ps1` PASS
- `isr-verify-authority-writer-reader-matrix.ps1` PASS
- （新設）`isr-verify-retire-queue-saturation-policy.ps1` PASS
- （新設）`isr-verify-publication-atomicity.ps1` PASS
- （新設）`isr-verify-publication-atomic-boundary.ps1` PASS
- （新設）`isr-verify-runtime-snapshot-identity.ps1` PASS
- （新設）`isr-verify-runtime-snapshot-never-reuse.ps1` PASS
- （新設）`isr-verify-retire-eligibility.ps1` PASS

## フェーズ5.5: Runtime Operational Certification

### フェーズ5.5の目的

CI合格だけでなく実運用安定性を満たす「出口条件」を機械的に証明する。

### フェーズ5.5の作業

- 48h soak 実施（nightly相当）
- publication mismatch / retire leak / crossfade mismatch / shadow compare mismatch を Severity 分類付きで定量監視
- soak exit volume criteria を適用（minimum publish/retire/crossfade count を必須化）
- フェーズ3・4の rollback point と復旧手順を固定化し evidence を保存
- rollback fire drill test を定期実行し rollback 経路の実動作を検証
- RuntimeWorld growth budget を導入（新Authority追加時に owner/purpose/replacement plan/review を必須化）
- RuntimeWorld memory budget を導入（Authority Count / RuntimeWorld Size / Publication Cost 上限を規定）
- Realtime budget を導入（Publish Time / Retire Time / Crossfade Commit Time 上限）
- budget enforcement policy を固定（上限超過時の Fail-Closed / Warning 条件を定義）
- shadow compare exit rule を定義（除去条件: 期間 + 不一致ゼロ + 最低イベント数）
- shadow compare coverage matrix を定義（Publish/Retire/Crossfade/Observe/Rollback/Recovery 全経路）

### フェーズ5.5の成果物

- 48h soak certification report
- operational mismatch summary（4指標）
- rollback evidence bundle
- rollback_fire_drill_report.md
- runtimeworld growth budget policy
- runtimeworld memory budget policy
- runtime_realtime_budget_policy.md
- soak_exit_volume_criteria.md
- runtimeworld_budget_enforcement_policy.md
- shadow_compare_exit_rule.md
- shadow_compare_exit_volume_criteria.md
- shadow_compare_coverage_matrix.md
- operational_mismatch_severity_policy.md

### フェーズ5.5のゲート

- `isr-verify-safety-regression.ps1` PASS
- `isr-verify-shadow-compare-cadence.ps1` PASS
- `isr-verify-rollback-matrix.ps1` PASS
- （新設）48h soak gate PASS
- （新設）`isr-verify-soak-exit-volume.ps1` PASS
- （新設）`isr-verify-runtimeworld-budget-enforcement.ps1` PASS
- （新設）`isr-verify-shadow-compare-exit-rule.ps1` PASS
- （新設）`isr-verify-shadow-compare-exit-volume.ps1` PASS
- （新設）`isr-verify-realtime-budget.ps1` PASS
- （新設）`isr-verify-shadow-compare-coverage.ps1` PASS
- （新設）`isr-verify-rollback-drill.ps1` PASS
- （新設）`isr-verify-operational-mismatch-severity.ps1` PASS

## フェーズ5.75: Migration Exit Audit

### フェーズ5.75の目的

「PASSしたから完了」ではなく「監査したから完了」を成立させるため、移行終了直前に最終監査を行う。

### フェーズ5.75の作業

- DoD全61項目の照合
- 全 verifier 結果と RFC 例外の照合
- Legacy 残存有無と RuntimeWorld authority count baseline の照合
- Exit Audit 実施者と実装者の分離を確認（self-certification禁止）
- verifier manifest hash の整合を確認（Verifier Integrity の自己循環回避）
- machine-generated audit package（inventory/hash/descriptor/matrix/DAG/soak/verifier）を生成
- migration re-entry contract（Authority/Semantic/Descriptor/Builder 変更時のDoD再評価条件）を照合
- migration exit audit レポートを作成し、移行完了可否を判定

### フェーズ5.75の成果物

- migration_exit_audit.md
- machine_generated_audit_package.zip
- migration_reentry_contract.md

### フェーズ5.75のゲート

- （新設）`isr-verify-migration-exit-audit.ps1` PASS
- （新設）`isr-verify-exit-audit-independence.ps1` PASS
- （新設）`isr-verify-verifier-manifest-hash.ps1` PASS
- （新設）`isr-generate-machine-audit-package.ps1` PASS
- （新設）`isr-verify-migration-reentry-contract.ps1` PASS

## フェーズ6: Legacy 収束と移行宣言

### フェーズ6の目的

移行残骸をゼロ化し、統治的にも完了状態へ到達する。

### フェーズ6の作業

- `LegacyTemporary` 項目を計画的に削除
- LegacyTemporary exit criteria（Replacement Authority Exists / Verifier Exists / Soak Pass）を満たすことを確認
- LegacyTemporary zero-reference criteria（Reader Count = 0 / Writer Count = 0 / Reference Count = 0）を満たすことを確認
- authority growth（未承認増加）を 0 に収束し、承認済み削減は RFC 追跡下で許可
- authority drift（inventory hash 差分）を 0 に収束
- operational exit criteria（Severity Policy準拠）を満たすまで close しない
- 最終監査レポート（before / after + gate 結果）作成

### フェーズ6の成果物

- legacy 削除完了一覧
- legacytemporary_exit_criteria.md
- legacytemporary_zero_reference_report.md
- authority_reduction_rfc_log.md
- 完全移行完了宣言レポート

### フェーズ6のゲート

- authority inventory PASS（growth = 0, drift = 0）
- authority reduction governance PASS（承認済み削減ログ整合）
- `isr-verify-legacytemporary-zero-references.ps1` PASS
- tiered verification 全PASS

## 5. 優先度と実行順（クリティカルパス）

1. フェーズ1（inventory 固定）
2. フェーズ2（semantic closure）
3. フェーズ3（observe collapse）
4. フェーズ4（crossfade 一本化）
5. フェーズ5（証明層拡張）
6. フェーズ4.5（semantic closure certification）
7. フェーズ5.5（runtime operational certification）
8. フェーズ5.75（migration exit audit）
9. フェーズ6（legacy 収束）

依存関係上、フェーズ1完了前にフェーズ3 / 4へ本格着手しないこと。

## 6. 期間見積（目安）

- フェーズ1–2: 1〜2週間
- フェーズ3–4: 2〜3週間
- フェーズ5–5.5: 1〜2週間
- フェーズ4.5: 0.5〜1週間
- フェーズ5.75: 0.5週間
- フェーズ6: 0.5〜1週間

**合計: 4〜7週間**（既存資産の流用度合いで短縮余地あり）

## 7. リスクと回避策

1. RT経路での隠れ直参照再発: lint + verifier + PR テンプレ必須チェック
2. Descriptor 運用の陳腐化: RuntimeState 変更時に inventory 更新を必須化
3. Crossfade 境界の回 regress: overlap state machine の契約テスト常設
4. Projection が authority 化して再劣化: projection governance gate（austerity / freshness）常設
5. RuntimeView からの権威漏れ再発: runtime view governance verifier + API契約テスト
6. EngineRuntime / DSPCore の authority 再流入: engine projection collapse verifier + DSPCore authority zero監査
7. RuntimeWorld の無秩序な肥大化: runtimeworld growth budget policy + review gate
8. RuntimeView の寿命不整合による参照切れ: runtime view lifetime verifier + lifecycle 契約テスト
9. Soak 試験の母数不足や偏りによる誤判定: minimum publish/retire/crossfade criteria + coverage matrix を gate 化
10. Semantic構造変更の無統治な流入: semantic mutation governance + classification rule + RFC
11. Verifier 変更の自己循環見逃し: verifier manifest hash + independent exit audit
12. 長時間ドリフト見逃し: nightly soak + evidence アーカイブ
13. RuntimeView 逃避参照の検出漏れ: escape analysis verifier を常設
14. Rollback 未使用経路の破損: rollback fire drill を定期実行
15. Realtime 予算逸脱の見逃し: latency budget gate を常設
16. ABI/Serialization 互換破壊: ABI contract + serialization contract + reachability/transition 監査を常設
17. verifier 偽陽性化の見逃し: verifier self-test を CI 常設
18. 移行完了後の再劣化: migration re-entry contract で DoD 再評価を強制

## 8. 最終判定フロー

1. 全フェーズ成果物レビュー完了
2. tiered verification 実行（必須）
3. 新設 verifier / lint / test 実行（必須）
4. evidence 監査（unapproved authority growth = 0, authority drift = 0, cadence pass, severity policy 準拠, soak volume/coverage/reachability criteria 達成）
5. Phase5.75 の migration exit audit 実施と承認
6. 「Practical Stable ISR Bridge Runtime 完全移行完了」宣言

## 9. 参照

- `doc/work10/notfinished_validation_report_2026-05-31.md`
- `doc/work10/notfinished.md`
- `.github/scripts/isr-run-tiered-verification.ps1`
- `.github/scripts/isr-verify-authority-inventory.ps1`
- `.github/scripts/isr-verify-authority-count-baseline.ps1`
- `.github/scripts/isr-verify-authority-identity-baseline.ps1`
- `.github/scripts/isr-verify-authority-freeze.ps1`
- `.github/scripts/isr-verify-authority-classification-rule.ps1`
- `.github/scripts/isr-verify-semantic-authority-contract.ps1`
- `.github/scripts/isr-verify-descriptor-uuid-stability.ps1`
- `.github/scripts/isr-verify-authority-reduction-governance.ps1`
- `.github/scripts/isr-verify-semantic-mutation-governance.ps1`
- `.github/scripts/isr-verify-projection-austerity.ps1`
- `.github/scripts/isr-verify-projection-freshness.ps1`
- `.github/scripts/isr-verify-publication-atomicity.ps1`
- `.github/scripts/isr-verify-publication-atomic-boundary.ps1`
- `.github/scripts/isr-verify-retire-lifecycle-state.ps1`
- `.github/scripts/isr-verify-retire-eligibility.ps1`
- `.github/scripts/isr-verify-retire-queue-saturation-policy.ps1`
- `.github/scripts/isr-verify-v73-retire-pressure-contract.ps1`
- `.github/scripts/isr-verify-safety-regression.ps1`
- `.github/scripts/isr-verify-rollback-matrix.ps1`
- `.github/scripts/isr-verify-runtimeworld-builder-governance.ps1`
- `.github/scripts/isr-verify-runtimeworld-builder-internal-mutation-governance.ps1`
- `.github/scripts/isr-verify-runtimeworld-layout-governance.ps1`
- `.github/scripts/isr-verify-publication-state-machine.ps1`
- `.github/scripts/isr-verify-runtimeworld-abi-contract.ps1`
- `.github/scripts/isr-verify-runtimeworld-budget-enforcement.ps1`
- `.github/scripts/isr-verify-runtime-snapshot-identity.ps1`
- `.github/scripts/isr-verify-runtime-snapshot-never-reuse.ps1`
- `.github/scripts/isr-verify-runtime-semantic-read-contract.ps1`
- `.github/scripts/isr-verify-semantic-closure-allowlist.ps1`
- `.github/scripts/isr-verify-semantic-closure-forbidden-inputs.ps1`
- `.github/scripts/isr-verify-runtime-recovery-semantic.ps1`
- `.github/scripts/isr-verify-soak-exit-volume.ps1`
- `.github/scripts/isr-verify-operational-mismatch-severity.ps1`
- `.github/scripts/isr-verify-realtime-budget.ps1`
- `.github/scripts/isr-verify-runtime-semantic-transition-graph.ps1`
- `.github/scripts/isr-verify-semantic-reachability.ps1`
- `.github/scripts/isr-verify-runtimeworld-serialization-contract.ps1`
- `.github/scripts/isr-verify-runtime-memory-lifetime.ps1`
- `.github/scripts/isr-verify-verifier-selftest.ps1`
- `.github/scripts/isr-verify-migration-exit-audit.ps1`
- `.github/scripts/isr-verify-exit-audit-independence.ps1`
- `.github/scripts/isr-verify-verifier-integrity-governance.ps1`
- `.github/scripts/isr-verify-verifier-manifest-hash.ps1`
- `.github/scripts/isr-generate-machine-audit-package.ps1`
- `.github/scripts/isr-verify-migration-reentry-contract.ps1`
- `.github/scripts/isr-verify-shadow-compare-contract.ps1`
- `.github/scripts/isr-verify-shadow-compare-cadence.ps1`
- `.github/scripts/isr-verify-shadow-compare-exit-rule.ps1`
- `.github/scripts/isr-verify-shadow-compare-exit-volume.ps1`
- `.github/scripts/isr-verify-shadow-compare-coverage.ps1`
- `.github/scripts/isr-verify-runtimeview-escape.ps1`
- `.github/scripts/isr-verify-authority-writer-reader-matrix.ps1`
- `.github/scripts/isr-verify-rollback-drill.ps1`
- `.github/scripts/isr-verify-legacytemporary-zero-references.ps1`
- `.github/scripts/isr-verify-gate-wiring.ps1`
