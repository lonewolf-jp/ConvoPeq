# Practical Stable ISR Bridge Runtime AI実装統治規約 v2.1（確定版ドラフト）

## 0. 文書情報

- 文書名: AI実装統治規約 v2.1
- 対象: `Practical Stable ISR Bridge Runtime 完全移行詳細計画` の実装作業全般
- 目的: AI実装の逸脱・局所最適・再劣化を防止し、DoD/Verifier/契約群への準拠を担保する
- 適用範囲: 設計変更、実装変更、テスト追加、Verifier追加/修正、運用監査更新

---

## 1. 本規約の目的

本規約は以下を満たす実装を AI に実施させるための統治基準である。

- Practical Stable ISR Bridge Runtime 完全移行詳細計画
- Definition of Done（DoD）全項目（現行最新版）
- フェーズゲート・Verifier群
- Governance Contract群

AI は本規約に反する実装を行ってはならない。

---

## 2. 最上位原則（Top Rules）

### Rule-1: Scoped DoD 準拠 + Global No-Regression

- 変更タスクは **対象スコープDoD** を事前に定義し、その未充足状態で完了宣言してはならない。
- 併せて、対象外DoDを含む既存保証を壊さないこと（Global No-Regression）。

### Rule-2: Verifier回避禁止

以下を禁止する。

- Verifier削除
- Verifier弱体化
- Verifierスキップ
- Verifier条件緩和
- Verifier除外設定

### Rule-3: 探索完了前の実装禁止

AIは改修前に探索を完了しなければならない。探索完了前の実装は禁止。

### Rule-4: 証跡なし完了宣言禁止

以下を提示できない場合、完了宣言を禁止する。

- 調査結果
- 改修箇所一覧
- 影響箇所一覧
- Verifier結果
- Scoped DoD 充足証跡

### Rule-5: 承認責務分離

- 実装担当（AI/人）と監査承認担当（人または独立AI）は分離する。
- 自己認定による「完了」は無効。

---

## 3. MCP探索義務（統合）

### Rule-MCP-1: 最低3系統探索

改修前に最低限以下を実施する。

1. grep/ripgrep 検索
2. CodeGraph 探索
3. Serena 探索

### Rule-MCP-2: 相互照合義務

- 単一系統のみで結論を出してはならない。
- 3系統結果を比較し、差分理由を記録すること。

### Rule-MCP-3: 探索完了条件

以下が揃うまで探索を継続する。

- Reader列挙
- Writer列挙
- Construction経路列挙
- Ownership列挙
- Dependency列挙
- Runtime Flow列挙

### Rule-MCP-4: 探索停止条件（Stop Rule）

無限探索防止のため、次の全条件成立で探索完了とする。

1. 必須マトリクスが全て作成済み
2. 3系統照合の未解決差分が0
3. 追加探索2サイクル連続で新規有意発見0件
4. 未確定仮説が0件（または明示的リスクとして登録済み）

### Rule-MCP-5: ツール障害時のFallback

MCP/検索系ツールが利用不能な場合、以下を提出して暫定進行を許可する。

- 利用不能証跡（日時・エラー）
- 代替探索手順
- 欠落リスク
- 後追い再検証計画

---

## 4. 必須探索対象

### 4.1 Authority探索

対象:

- RuntimeWorld
- RuntimeState
- RuntimeView
- RuntimeReadView
- RuntimePublishView
- RuntimeGraph
- EngineRuntime
- DSPCore

観点:

- Reader / Writer / Constructor / Mutation / Publication / Observe

### 4.2 Descriptor探索

対象:

- `kFieldDescriptors`
- RuntimeState fields
- DescriptorInventory

観点:

- UUID / Count / Mapping

### 4.3 Semantic探索

対象:

- semantic source
- semantic read
- semantic dependency

観点:

- owner / writer / reader / dependency / update phase

### 4.4 Publication探索

対象:

- publish
- retire
- crossfade
- overlap

観点:

- caller / state transition / visibility

---

## 5. 改修対象発掘漏れ防止

### Rule-DISCOVERY-1: 波及調査義務

1箇所修正した場合、最低限以下への波及を確認すること。

- Reader
- Writer
- Constructor
- Test
- Verifier

### Rule-DISCOVERY-2: 参照元再帰探索義務

修正対象の参照元（caller/import/referencer）を再帰探索すること。

### Rule-DISCOVERY-3: 参照先再帰探索義務

修正対象の参照先（callee/dependency）を再帰探索すること。

### Rule-DISCOVERY-4: Call Graph 完全確認

CodeGraphにより Caller/Callee を確認し、変更影響の打ち切りを禁止する。

### Rule-DISCOVERY-5: Construction経路全列挙

以下を全列挙する。

- constructor
- factory
- builder
- publish path
- retire path

---

## 6. 実装禁止事項（Hard Prohibitions）

1. RuntimeGraph authority追加
2. EngineRuntime authority追加
3. DSPCore authority追加
4. RuntimeView mutable API追加
5. RuntimeView authority保持
6. Semantic外判定追加
7. Descriptor迂回
8. BuilderToken迂回
9. Projection authority化
10. LegacyTemporary延命
11. `JUCE/` と `r8brain-free-src/` への編集
12. SEH利用
13. Audio threadでの禁止処理（lock/alloc/io/libm等）

---

## 7. 実装前報告義務（Pre-Implementation Report）

### 7.1 調査報告

- 発見Authority一覧
- 発見Reader一覧
- 発見Writer一覧
- 発見Construction Path一覧
- 発見Dependency一覧
- 発見Verifier一覧

### 7.2 影響分析

- 修正対象
- 関連対象
- 追加修正対象
- テスト対象
- Verifier対象

### 7.3 追加必須

- Scoped DoD（対象DoD項目）
- No-Regression確認対象
- 変更リスクTier（A/B/C）

---

## 8. 実装後報告義務（Post-Implementation Report）

- 変更ファイル一覧
- Authority差分（追加/削除/変更）
- Descriptor差分（UUID/Count/Inventory）
- Semantic差分（Owner/Reader/Writer/Dependency）
- Runtime Flow差分（Publication/Retire/Observe/Crossfade）
- Verifier結果（実行ログ・失敗要因・再実行結果）
- Scoped DoD達成証跡
- No-Regression証跡

---

## 9. 完了宣言条件

以下を全て満たした場合のみ完了可能。

1. Scoped DoD 充足
2. Global No-Regression 確認
3. フェーズゲート通過
4. 必須Verifier PASS
5. MCP探索証跡提出
6. 改修対象一覧提出
7. 波及分析提出
8. Authority/Descriptor/Runtime Flow差分提出
9. CI PASS
10. 監査承認者の独立承認

---

## 10. MCP利用強制規約

### 必須ツール

- Serena MCP
- CodeGraph MCP
- grep/ripgrep

### 推奨順序

1. grep/ripgrep
2. CodeGraph
3. Serena
4. 結果照合
5. 追加探索
6. 実装

### 禁止

- 単一ツールのみで結論
- 単一検索結果のみで実装
- 推測実装
- 「たぶん未使用」による削除

---

## 11. 探索完了判定（提出物）

以下マトリクス提出まで探索終了を禁止する。

- Reader Matrix
- Writer Matrix
- Authority Matrix
- Dependency Matrix
- Construction Matrix
- Publication Matrix
- Retire Matrix
- Observe Matrix
- Crossfade Matrix
- Verifier Matrix

---

## 12. 変更リスクTier運用

### Tier-A（高リスク）

対象例: authority/semantic/publication/retire/ABI/recovery/lifetime

- 3系統探索必須
- フル波及分析必須
- 主要Verifier + 影響Verifier必須

### Tier-B（中リスク）

対象例: テスト拡張・監査文書更新・非中核ロジック

- 2系統以上探索
- 影響Verifier実行

### Tier-C（低リスク）

対象例: typo、コメント、非意味論ドキュメント

- grep中心可
- ただし理由記録必須

---

## 13. Verifier健全性保証

### Rule-VRF-1: Self-Test必須

`isr-verify-verifier-selftest.ps1` により、意図的違反サンプルが FAIL することを確認する。

### Rule-VRF-2: Verifier変更の分離レビュー

Verifier更新PRは実装PRと分離し、独立レビューを必須化する。

### Rule-VRF-3: 失敗時の扱い

Self-Test失敗時は実装マージ禁止。

---

## 14. 削除統治（Deletion Contract）

コード・契約・Verifierの削除は以下全てを満たすこと。

1. 参照0（検索証跡）
2. Caller/Callee 0（CodeGraph証跡）
3. 代替経路存在証跡
4. テスト/Verifier影響評価
5. 承認者サインオフ

---

## 15. CI/運用実行方針（段階実行）

- PR時: 影響範囲Verifier + クリティカルVerifier
- マージ前: tiered full verification
- nightly: full + soak + verifier self-test

---

## 16. 例外管理（Exception Governance）

例外許可には以下が必要。

- 例外理由
- 期限
- 代替抑止策
- 元に戻す条件（Exit Criteria）
- 承認者

期限超過例外は無効。

---

## 17. 規約違反時の措置

- 即時: 完了宣言無効
- 是正: 差分ロールバックまたは追補修正
- 再発防止: 規約追記またはVerifier追加

---

## 18. このv2.1で強化した点（v2.0から）

1. Scoped DoD + No-Regression を明文化
2. 探索停止条件（Stop Rule）を追加
3. ツール障害時Fallbackを追加
4. リスクTier運用を追加
5. Verifier Self-Testを必須化
6. 承認責務分離を明文化
7. 削除統治（Deletion Contract）を追加
8. プロジェクト固有禁止事項を統合

---

## 19. 付録（報告テンプレ要約）

### Pre-Implementation

- Scope / Tier / Scoped DoD
- Discovery Matrix一式
- 影響分析

### Post-Implementation

- 変更差分
- Matrix更新差分
- Verifier結果
- DoD充足証跡
- No-Regression証跡

---

## 20. 最終条項

本規約は、AIが短期の実装修正ではなく、Practical Stable ISR Bridge Runtime の長期安定性を損なわないための拘束規範である。疑義がある場合は、実装より先に統治・検証・証跡を優先すること。
