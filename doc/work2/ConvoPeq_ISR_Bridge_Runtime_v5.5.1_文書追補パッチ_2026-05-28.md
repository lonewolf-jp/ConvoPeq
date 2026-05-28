# ConvoPeq ISR Bridge Runtime v5.5.1 文書追補パッチ

- Date: 2026-05-28
- Purpose: `ConvoPeq_ISR_Bridge_Runtime_v5.5_詳細設計_妥当性監査_2026-05-28.md` の是正提案を最小差分で反映
- Scope: 文書追補のみ（コード変更なし）

---

## 1. 反映対象

1. `doc/work2/ConvoPeq_ISR_Bridge_Runtime_v5.5_詳細設計_FINAL_2026-05-28.md`
2. `doc/work2/ISR_Bridge_Runtime_v5.5_FINAL_FREEZE_2026-05-28.md`
3. `doc/work2/ConvoPeq_ISR_Bridge_Runtime_v5.5_AI_実装統制規約_FINAL_2026-05-28_LINT.md`

---

## 2. 反映内容（監査是正との対応）

### A. 詳細設計書への追補

- publication 命名統一（`appendPublicationIntent` 正規名、`enqueuePublication` legacy alias）
- safe-to-collapse 条件統一（UI-driven transient を含む全成立条件）
- 参照型の「定義元確定」表追加
- 挙動検証の最小定量（shutdown反復/rebuild burst/saturation観測/null test目安）追加
- FREEZE Rule-O〜Z 対応表を追加

### B. FREEZE 仕様への追補

- `shutdownRuntime_` 役割の明確化
- publication 経路の命名統一方針明記
- Telemetry 識別子を単一情報源として固定

### C. AI 実装統制規約（LINT）への追補

- Rule-2/4/11/23 の CI fail-stop 静的検査必須化
- Rule-8 向け `shutdownRuntime_` 役割追補
- Rule-12 向け `saturationEnterCount` / `saturationExitCount` mandatory 化
- Rule-19 向け execution-local 判定基準（3条件）追加

---

## 3. 期待効果

- 仕様解釈の分岐点（命名・条件・観測）を縮小
- トレーサビリティの不足（FREEZE Rule-O〜Z）を解消
- DoD/検証条件の客観性を向上
- 実装前レビューでの合意形成コストを低減

---

## 4. 非変更事項

- 実装コード
- DSPアルゴリズム仕様
- スレッドモデル本体

---

## 5. 適用状態

本パッチ内容は上記3文書へ反映済み（v5.5.1 追補節として追加）。
