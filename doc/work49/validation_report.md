# Practical Stable ISR Bridge Runtime 完全移行計画書 検証レポート

**検証日:** 2026-06-19
**検証者:** GitHub Copilot (DeepSeek V4 Flash)
**対象:** doc/work49/basic_plan.md v2.0
**検証方法:** Serena MCP, AiDex MCP, CodeGraph MCP, graphify, semble, Select-String

---

## 1. 総評

**計画書の現状認識 (92-95%) は妥当だが、各 Phase の必要性評価には大幅な見直しが必要。**

計画書が「未実装」として記述した機能の多くが既にコードベースに実装済みであり、
本計画の実質的な残作業量は計画書の記述よりも**大幅に少ない**。

---

## 2. 計画書 vs 実装状況 詳細マッピング

### 2.1 既に実装済みの項目（計画書では未実装/今後と記載）

| 計画書の言及 | 実装状況 | 該当ファイル |
|---|---|---|
| §8: Validator実体化「Placeholder」 | ✗ 誤り。**既に実体化済み**。validatePublication(), validateSemanticConsistency(), validateTopology(), validateResources(), checkNoConflictingTransitions() がフル実装 | RuntimePublicationValidator.h/.cpp |
| §9: Resource Validation | ✗ 誤り。**既に実装済み**。Oversampling(1/2/4/8/16), Dither(0/16/24/**32**), NoiseShaper(0/1/2/**3**) | RuntimePublicationValidator.cpp:102-122 |
| §10: Transition Validation | ✗ 誤り。**既に実装済み**。SmoothOnly/DryAsOld/HardReset の全ロジック完備 | RuntimePublicationValidator.cpp:158-200 |
| §11: Validation Telemetry | ✗ 誤り。**既に実装済み**。6000-6003 の4イベント定義済み、emitValidationEvent() 実装、1秒レート制限完備 | RuntimeHealthMonitor.h/.cpp |
| §12-13: CrossfadeAuthority/Policy | ✗ 誤り。**既に実装済み**。CrossfadePolicy 定義、evaluate() 純粋関数実装 | CrossfadeAuthority.h/.cpp |
| §14: HealthState責務移動 | ✗ 誤り。**既に実装済み**。CrossfadeAuthority は HealthState を参照しない。Orchestrator レベルで Critical 抑制 | RuntimePublicationOrchestrator.cpp:108-115 |
| §15: Emergency Override | ✗ 誤り。**既に実装済み**。DSPTransition::onPublishCompleted() 内で Critical 検知→即 activate/complete/retire | DSPTransition.h:55-75 |
| §16: publishIdleWorldOnly() | ✗ 誤り。**既に実装済み** | AudioEngine.Transition.cpp:10-30 |
| §20: テスト戦略 | ✗ 誤り。**Validator/CrossfadeAuthority のテストは既に存在**。30+ テストケース | PublicationValidatorIsolationTests.cpp |
| §8.2: Topology Validation | ✗ 誤り。runtimeUuid==0 時のチェックは既に実装済み | RuntimePublicationValidator.cpp:74-100 |
| §1.5: Validator Telemetry レート制限 | ✗ 誤り。compareExchangeAtomic ではなく load+store だが、CAS保護は過剰設計とコメント | RuntimeHealthMonitor.cpp:1235-1245 |

### 2.2 計画書に記載はないが既に存在する追加要素

| 要素 | 該当ファイル |
|---|---|
| PublicationAdmission (Decision 7種、PressureLevel 4種) | PublicationAdmission.h/.cpp |
| RuntimePublicationOrchestrator (Admission→Executor→DSPTransition パイプライン) | RuntimePublicationOrchestrator.h/.cpp |
| PublicationExecutor | PublicationExecutor 関連 |
| 出版停滞監視 (isPublicationStalled) | RuntimePublicationOrchestrator.h:117 |
| Deferred Publish TTL (30秒) | RuntimePublicationOrchestrator.h |
| RuntimeStore template (Single Source Of Truth) | src/core/RuntimeStore.h |
| RuntimePublicationCoordinator template (publishWorld() with PublishStageResult) | src/core/RuntimePublicationCoordinator.h |
| RecoveryAction/EscalationTracker/RecoveryBudget | RuntimePolicyEngine.h/.cpp |
| ISRHealthState (Healthy/Degraded/Critical) | RuntimeHealthMonitor.h |
| ShutdownRuntime (完全なFSM) | ISRShutdown.h/.cpp |

### 2.3 未実装の項目（計画書の通り）

| Phase | 項目 | 詳細 | 影響度 |
|---|---|---|---|
| Phase-5 | **AuthoritySource** enum | publishWorld() に発行元種別を渡す仕組みなし | 中（traceability欠如） |
| Phase-1.5 | **AuthorityTelemetry** | callCount[6] のカウンタ未実装 | 低（Validator Telemetryは既存） |
| Phase-6 | **PersistentStateBlock** | publicationSequenceId/epoch/mappedGeneration が Coordinator に分散保持 | 中（統一インターフェース不足） |
| Phase-7 | **deriveAuthorityState()** | PersistentState + RuntimeStore からの再導出関数なし | 高（リカバリの核） |
| Phase-8 | **Recovery Architecture** 完全版 | Step1-6 の体系的手順なし。RecoveryAction は部分的 | 中 |
| Phase-9 | **currentWorld_ 完全撤廃** | ISRRuntimePublicationCoordinator 内の atomic ポインタ存続 | 中 |
| — | **ScopedRuntimePublicationAuthority** | thread_local owner 機構なし | 低（publish 実績から不要の可能性） |

### 2.4 計画書の不正確または不足している記述

1. **§9 Dither 許容値**: 計画書は {0, 16, 24} と記載。実際のコードは {0, 16, 24, **32**} を許容。32 が不足。
2. **§9 NoiseShaper 許容値**: 計画書は {0, 1, 2} と記載。実際のコードは {0, 1, 2, **3**} を許容。3 が不足。
3. **達成率評価**: 現状 92-95% ではなく **95-97%** が妥当。Phase-0〜4 完了後は 99% に近い。

---

## 3. 各 Phase の再評価

### Phase-0: 既存バグ修正

**必要性: ✅ 条件付き**

- 0a (Validatorテスト修正): テストは既に存在する。「修正」が必要かは個別確認要
- 0b (useDryAsOld修正): 未確認。Retire 周りの挙動を検証要
- 0c (Dead Code削除): currentWorld_ 等の評価は必要

### Phase-1: Validator実体化

**必要性: ❌ 不要（既に実装済み）**

### Phase-1.5: Validator Telemetry

**必要性: ❌ 不要（既に実装済み）**

### Phase-2: CrossfadePolicy抽出

**必要性: ❌ 不要（既に実装済み）**

### Phase-2.5: Emergency Override公式化

**必要性: ❌ 不要（既に実装済み）**

### Phase-3: テスト拡充

**必要性: ⚠️ 低優先度**

- Validator/CrossfadeAuthority テストは既存
- Recovery テスト、Property Test は未着手

### Phase-4: Validator網羅率拡充

**必要性: ⚠️ 低優先度**

- 基本網羅は完了。エッジケース拡充は継続的改善

### Phase-5: AuthoritySource導入

**必要性: ✅ 中**

- トレーサビリティ向上。publishWorld() の引数追加

### Phase-6: PersistentStateBlock導入

**必要性: ✅ 中**

- 永続状態の統一インターフェースとして有用

### Phase-7: deriveAuthorityState導入

**必要性: ✅ 高**

- リカバリの要。PersistentState + RuntimeStore からの再導出関数

### Phase-8: Recovery Architecture導入

**必要性: ⚠️ 中**

- RecoveryAction/EscalationTracker は既存。Step1-6 の体系化が不足

### Phase-9: currentWorld_撤廃

**必要性: ⚠️ 低〜中**

- 二重管理解消。ISRRuntimePublicationCoordinator の currentWorld_ を削除し RuntimeStore に統一

### Phase-10: SSOT完成

**必要性: ⚠️ Phase-6〜9 完了後の確認フェーズ**

---

## 4. 推奨フェーズ計画（改訂版）

### Phase-A: 確実な不足対応（工数: 小）

1. AuthoritySource 導入（Phase-5相当）
2. PersistentStateBlock 導入（Phase-6相当）
3. deriveAuthorityState() 実装（Phase-7相当）
4. currentWorld_ → RuntimeStore 統合（Phase-9相当）

### Phase-B: リカバリ体系化（工数: 中）

1. 既存 RecoveryAction と deriveAuthorityState() の接続
2. Recovery Architecture Step1-6 の形骸化
3. Crossfade/Retire Timeout との統合

### Phase-C: テスト拡充（工数: 小〜中）

1. Recovery 障害注入テスト
2. Property Test (10,000〜100,000回ランダムシーケンス)
3. Validator エッジケース拡充

---

## 5. 検証に使用したツール一覧

| ツール | 用途 | 結果 |
|---|---|---|
| Serena MCP (find_symbol) | シンボル構造・参照関係の把握 | 有効 |
| AiDex MCP (aidex_query) | 識別子検索（grep代替） | 有効（278ファイル/12247 items） |
| CodeGraph MCP | ファイル構造把握 | 有効 |
| graphify MCP (god_nodes) | 知識グラフの中心ノード確認 | 有効（Masterplan: 325 edges） |
| semble (semantic search) | 自然語言語セマンティック検索 | 有効 |
| Select-String (grep) | パターン検索（未実装項目の確認） | 有効 |
