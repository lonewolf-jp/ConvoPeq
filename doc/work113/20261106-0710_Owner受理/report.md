# 限定実走の受理を反映

- 実施: 2026-11-06（Windows 11 x64）
- 対象: RES2-11（TSan）、RES2-12（soak）、RES2-13（TX）
- 結論: Owner の受理判断を最終判断記録 §11 に追記し、各証拠記録・レジスタ・台帳へ反映した。コード・CI・schedule・実走の追加は行っていない。

## 1. 受理された範囲

| 対象 | 受理された内容 | 未実走・未決定 |
|---|---|---|
| TSan | 計装 3 本の実走（PASS 3、報告 0、ランタイム初期化・シンボル確認） | `ISRSemanticValidationTests`、`RetireGraceSemanticsTests`。MKL・IPP の Linux 対応は未承認 |
| soak | `ce386431` から再ビルドした手動 1 回（ALL PASS、744.8 秒） | ランナー登録、費用、管理者、運用責任者、週次化 |
| TX | `[RTM]` と `xRunBuffer` の取得口の確認 | 実機、シナリオ、実施者、実施時期、waiver |

## 2. 状態の境界

- TSan は best-effort を維持し、必須ゲートにはしない。報告 0 件をデータ競合の不存在として扱わない。
- soak は手動実走の証拠として扱う。744.8 秒は本記録の 1 回で、799.9 秒（2026-11-05）とは別の実行である。
- TX は測定準備の受理であり、性能の達成や未測定シナリオへの外挿を示さない。
- いずれも「検証完了」「達成」へ昇格させていない。

## 3. 変更した資料

| 資料 | 更新 |
|---|---|
| `RES2-OWNER-FINAL-DECISION-20261106.md` | §11 を追加（3 件の受理、実行権限の境界） |
| `RES2-TSAN-RUN-EVIDENCE-20261106.md` | §10 を追加（受理、未承認事項、0 件の解釈） |
| `RES2-SOAK-MANUAL-RUN-20261106.md` | §7 を追加（受理、744.8 秒と 799.9 秒の区別、未決定事項） |
| `RES2-TX-MEASUREMENT-PLAN-20261106.md` | §9 を追加（取得口の受理、未指定事項、基準不変） |
| `RES2-REGISTER-FINAL-20261105.md` | RES2-11・12・13 の状態と完了判定に受理を追記 |
| `RESIDUAL-LEDGER-PHASE-E-20261102.md` | §27 を追加 |
| `RES2-VERIFICATION-READINESS-20261106.md` | §6 の承認表を今回の判断に合わせて更新 |

## 4. 変更しなかったもの

RT 経路、Publish、Retire、Epoch、Crossfade、DSP 実装、既定値、CI workflow、schedule、waiver。追加の実走も行っていない。ソース変更がないため `ConvoPeq.md` は再生成せず、FRESH のみ確認する。

## 5. 検証

| 確認 | 結果 |
|---|---|
| 文書のみ | `src`・`CMakeLists.txt`・workflow に差分なし |
| 表の列数 | 不一致 0 |
| 偽漢字 | 0 |
| 日本語 lint | 0 |
| `ConvoPeq.md` | FRESH |

## 6. 次の判断が必要な事項

| 事項 | 内容 |
|---|---|
| TSan の継続 | MKL・IPP の Linux 対応に着手するか、現状で打ち切るか |
| soak の運用 | ランナー登録、費用、管理者、障害時対応、運用責任者の確定 |
| TX の実施 | 実機、シナリオ、実施者、実施時期の指定 |
| waiver | 測定結果が出た後の個別判断 |
