# ConvoPeq ISR Bridge Runtime v5.5 詳細設計 妥当性監査レポート

- Date: 2026-05-28
- Target:
  - `doc/work2/ConvoPeq_ISR_Bridge_Runtime_v5.5_詳細設計_FINAL_2026-05-28.md`
- Baseline:
  - `doc/work2/ISR_Bridge_Runtime_v5.5_FINAL_FREEZE_2026-05-28.md`
  - `doc/work2/ConvoPeq_ISR_Bridge_Runtime_v5.5_AI_実装統制規約_FINAL_2026-05-28_LINT.md`
- Verdict: **Medium risk（文書追補で Low へ低減可能）**

---

## 1. 監査サマリー

対象詳細設計は、上位2文書（FREEZE / RULES）と**方向性・数値・導入順序が高い整合**を持つ。
一方で、実装時の解釈分岐を生む欠落（型定義参照、命名統一、テスト閾値）と、トレーサビリティ不足が残る。

---

## 2. 一致点（高重要）

1. 最優先軸（DAW実運用耐性、5優先軸）が一致
2. 段階導入順序（5フェーズ）が一致
3. Admission Gate API / 適用点 / reject時即return が一致
4. Drained完了条件5項目と publicationCoordinator drained 内訳が一致
5. Backpressure 数値（HWM/LWM、clamp 0.75-1.50）が一致
6. memoryPressureScale の許可/禁止入力源が一致
7. saturation 中の安定化方向制約が一致
8. Snapshot lifecycle / finalize determinism / versioning が一致
9. Rebuild collapse（latest wins / must-execute禁止）が一致
10. DoD 10項目が一致

---

## 3. 欠落・弱点

1. FREEZE の Rule-O〜Z と DESIGN の明示対応表がない
2. `enqueuePublication` と `appendPublicationIntent` の命名が曖昧
3. `safe-to-collapse` の条件セットに微差（UI-driven transient の扱い）
4. `RuntimeGeneration` など参照型の定義元が未明示
5. behavior-preserving の定量閾値（音響差分許容）が未定義
6. テレメトリの識別子と必須化レベルが分散

---

## 4. 曖昧点（実装分岐リスク）

- StopAcceptingWork への遷移トリガ
- append→prepare→execute の厳密順序
- saturation の上昇時ステップ値
- fingerprintVersion mismatch 時の処理（reject/rebuild）
- drained後違反の検出方式（assert/telemetry/fail-stop）

---

## 5. テスト不能要件

以下は観測手段または閾値が不足し、現状では客観評価困難。

- finalize determinism（比較インタフェース未定義）
- behavior-preserving（許容誤差未定義）
- stale runtime reuse 不可能（stale判定フック未定義）
- RT禁則不在（CIでの必須検査定義不足）

---

## 6. 是正提案（最小差分）

1. Rule-O〜Z と Rule-0〜32 の**二系統トレーサビリティ表**を追加
2. 用語統一（enqueuePublication alias廃止または正規名固定）
3. safe-to-collapse 条件を3文書で同一集合に統一
4. 参照型の定義元一覧表を追加
5. テレメトリ項目名の単一情報源をFREEZEに追記
6. 検証章に最小定量（反復回数、負荷量、許容誤差）を追記

---

## 7. 結論

現行詳細設計は、**実装着手可能な基礎品質は満たす**。
ただし、上記6項目を文書追補しない場合、フェーズ後半で「仕様解釈差による手戻り」が発生する可能性がある。

推奨: まず文書追補（1〜6）を完了してから実装へ移行する。
