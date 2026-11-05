# RES2-12 soak 運用の設計資料（2026-11-05）

- 基準: `main` = `9dd57529`
- Owner 判断: 週次スケジュールを採用候補とする。実行時間とランナーコストを確認し、失敗時は通知のみ（PR ブロックなし）。設定変更は確認後。

## 1. 現行構成

| 項目 | 内容 | 根拠 |
|---|---|---|
| トリガ | `workflow_dispatch` のみ（手動） | `.github/workflows/soak-ci.yml:19-25` |
| ランナー | `windows-latest` | 同 `:38-40` |
| タイムアウト | 60 分 | 同 `:40` |
| シナリオ | all / s1 / s2b / s3 / s4 / s5 | 同 `:13-22` |
| 実行 | `AudioEngineHarness.exe --soak --scenario=...` | 同 `:111-118` |
| 保存 | 実行ログ（artifact 化は要確認） | 同 |

## 2. 週次化の評価

| 観点 | 週次（候補） | 日次 | push ごと |
|---|---|---|---|
| 検出遅延 | 最大 7 日 | 最大 1 日 | 即時 |
| ランナー時間 | 60 分/週（全シナリオ 1 回） | 60 分/日 | PR ごとに 60 分 |
| PR への影響 | なし（通知のみ） | なし | 大幅な待ち時間 |
| フレーク時の負担 | 週 1 回の調査 | 日次で調査 | 開発停止 |
| 推奨 | **採用候補** | 非推奨 | 非推奨 |

- 推定コスト: `all` 1 回 = 最大 60 分（timeout）。週次で月 4 時間程度。実測時間は初回実行時に記録する（未測定）。
- 通知: GitHub Actions の失敗通知（メール/UI）。PR はブロックしない。Issue の自動作成は行わない（運用判断）。

## 3. 自動化する場合の変更点（未実施）

1. `on:` に `schedule`（例: `cron: '0 18 * * 0'` = 日曜 03:00 JST）を追加
2. `continue-on-error` は付けない（通知は GitHub の失敗通知で足りる）
3. 結果ログを artifact として保存（`actions/upload-artifact`）
4. 実行時間の記録（開始・終了のタイムスタンプ）

## 4. 結論

- 現状は手動のまま。週次化は Owner の確認後に設定変更する。
- 初回の実測（`all` の実行時間）を取得してから頻度を最終決定する。

## 5. Owner 判断（2026-11-05 追加）

- 週次スケジュール実行を採用候補とする。
- 1 回の実行時間・CI コスト・成功率・失敗時の通知方法を確認し、許容範囲であることを条件に設定変更を承認する。
- 失敗は当面 PR ブロックではなく通知扱いとする。
- 現時点で workflow は変更しない（`schedule` を追加していない）。

## 6. 設定変更の採用条件

| 項目 | 条件 | 測定方法 | 現状 |
|---|---|---|---|
| 実行時間 | `all` 1 回が 60 分以内（timeout 内） | 開始・終了のタイムスタンプ | ローカル 13.3 分。CI は skip |
| CI コスト | 月 4 時間以内 | 実行時間 × 週 1 回 | 推定 4 時間 |
| 成功率 | 直近 4 回で `fail` 0（初回は除く） | Actions の履歴 | 実績なし |
| 通知 | 失敗時に GitHub 通知、PR は止めない | 運用確認 | 方針のみ |

- 最初に手動で `all` を 1 回実行し、実行時間と結果を記録する。その値で条件を満たすと確認できた後に、`schedule` の追加を別途承認する。

## 7. CI での実測（2026-11-05）

- `workflow_dispatch` で `scenario=all` を 1 回実行した（run 42044137638）。
- 結果: success、実行時間 0.6 分。ただしログは **skip** である。

> Soak CI skipped
> - Reason: Intel oneAPI not available on this runner.
> - MKLNonUniformConvolver requires mkl.h/ipp.h (Intel oneAPI).

- GitHub の `windows-latest` には oneAPI が無いため、soak はビルドできず、ジョブは「skip を報告して success」になる。**CI での週次実行は、このままでは実測にも回帰検出にもならない。**
- 実行時間の実測は、oneAPI のある開発機（ローカル）で行う必要がある。

### ローカル実測（2026-11-05）

`AudioEngineHarness.exe --soak --scenario=all` を oneAPI のある開発機で実行した。

| シナリオ | 結果 |
|---|---|
| s1 | published 100,000 / accepted 100,000 / rejected 0 / finalSeq 100,003 / backlog 0 |
| s2b | requested 80,000 / accepted 80,000 / rejected 0 / peakBacklog 0 |
| s3 | rapid 300 / drained_to_seq 180,303 / backlog 0 |
| s4 | published 10,001 / epoch 360,606 → 380,608 / pending 0 |
| s5 | samples 158 / peak 287.4 MB / final 287.4 MB（収束） |
| 合計 | issued 180,300 / accepted 180,300 / rejected 0 / peakBacklog 0 / **ALL PASS** |

- 実行時間: **799.9 秒（13.3 分）**。60 分以内の条件を満たす。
- メモリ: 287.4 MB で収束し、増加を続けない。
- これは開発機の値である。CI（oneAPI なし）では同じ結果は得られない。
- 週次化の前提が変わる: 「CI の実行時間 ≤ 60 分」を満たすには、ランナーへ oneAPI を導入するか、セルフホストランナーを使うかの判断が要る。

