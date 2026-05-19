# ConvoPeq ISR移行計画（軽量ハブ・正本参照専用）

## 位置づけ

本書は **ISR implementation governance baseline** を維持するためのハブ文書です。
詳細仕様は個別正本に保持し、本書は「要約・優先順位・ゲート管理」のみを扱います。

## 正本ドキュメント（1 topic = 1 authoritative spec）

- 全体方針: `doc/work/ISR改修計画書_修正版_現状認識.md`
- Phase A ガイド: `doc/work/ISR_Phase_A_詳細実装ガイド.md`
- Runtime分類: `doc/work/ISR_Runtime_State_Matrix.md`
- DSP分解: `doc/work/ISR_DSPCore_Decomposition_Analysis.md`
- ownership可視化: `doc/work/ISR_Runtime_Ownership_Graph_完全可視化.md`
- retire authority: `doc/work/ISR_Retire_Authority_Graph.md`
- HB仕様: `doc/work/ISR_HB_Graph_Specification.md`
- immutability enforcement: `doc/work/ISR_Immutability_Enforcement_Spec.md`
- DSPHandle allocator policy: `doc/work/ISR_DSPHandle_Allocator_Policy.md`
- shared EpochDomain scalability検証: `doc/work/ISR_Shared_EpochDomain_Scalability_Validation_Plan.md`
- 未完リスク統制バックログ: `doc/work/ISR_Completeness_Risk_Backlog.md`
- 形式保証パッケージ: `doc/work/ISR_Formal_Guarantee_Package.md`
- 最小フェーズ0（推奨）: `doc/work/ISR_Minimal_Phase0_Recommended.md`

---

## 総合評価（2026-05-20）

| 項目 | 評価 |
| --- | --- |
| 現状コード理解 | 良好 |
| ownership graph | 良好 |
| retire authority | 良好 |
| HB graph | 良好 |
| allocator policy | 良好 |
| enforcement設計 | 改善 |
| implementation governance | 大幅改善 |
| RuntimePublishWorld completeness | 未完（仕様固定・実装未完） |
| 実装開始可能性 | Phase A/B は妥当 |
| ISR完成度 | まだ未完成 |

---

## 設計判断（固定）

- `Spec-Fixed` と `Closed` を厳密分離する
- `1 topic = 1 authoritative spec` を維持し、ハブと正本の責務混在を禁止する
- `single authority != single mega-manager` を維持し、authority identity と lane を分離する
- RuntimePublishWorld は runtime-only publish world とし、GlobalSnapshot / RTLocalState と分離する
- `const化` ではなく sealed-at-publish + CI enforceable で post-publish mutation を禁止する

---

## 実装開始可否の境界（固定）

### Phase A/B 開始: 可（条件付き）

- Runtime Matrix / HB / authority / allocator / enforcement / backlog が正本化済み
- R1〜R18 の最小検証項目が定義済み

### RuntimePublishWorld 最終固定・ISR完成宣言: 不可（現時点）

以下がすべて `Closed` になるまで不可:

- recursive ownership closure
- shutdown HB completeness
- bug2 minimal HB model
- RuntimePublishWorld payload boundary
- RT detect -> NonRT retire bridge 実装検証
- shared/split epoch migration の実測比較

---

## 形式保証と実装順の正本参照

- 形式保証パッケージ（P1〜P8）: `doc/work/ISR_Formal_Guarantee_Package.md`
- 未完リスクとClosed最小検証項目（R1〜R18）: `doc/work/ISR_Completeness_Risk_Backlog.md`
- 当面の実行順（最小フェーズ0）: `doc/work/ISR_Minimal_Phase0_Recommended.md`

本ハブでは P1〜P8 の仕様本文・優先順テーブルを再掲しない。

---

## 重複防止規約

- `plan5.md` は要約とリンクのみ保持
- 正本にある仕様本文を `plan5.md` へ再掲しない
- 結合ファイル化（他文書全文貼り戻し）を禁止
