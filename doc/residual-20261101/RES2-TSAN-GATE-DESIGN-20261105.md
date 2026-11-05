# RES2-11 TSan 必須化の設計資料（2026-11-05）

- 基準: `main` = `9dd57529`
- Owner 判断: 現時点で必須ゲート化しない。対象拡大・連続成功率・環境依存性を調査し、別判断資料を作成する。

## 1. 現行構成

| 項目 | 内容 | 根拠 |
|---|---|---|
| 実行環境 | `ubuntu-latest`、Clang、Debug、TSan | `.github/workflows/sanitizer-ci.yml:255` |
| 判定 | `continue-on-error: true`（PR をブロックしない） | 同 `:259` |
| タイムアウト | 30 分 | 同 `:258` |
| 対象 | MKL/IPP に依存しないテストのみ（ISR 系）。configure 不能時は graceful skip | 同 コメント・`:283-300` |
| ビルド不能時 | `build.ninja` 不在で skip（成功扱い） | 同 |

## 2. 実績（直近 20 run）

- `sanitizer-ci.yml` の直近 20 run のうち、完了済み 19 run は **すべて success**（1 run は実行中）。
- 失敗・flaky の観測は直近 20 run ではない。
- ただし TSan の "success" は「skip しなかった場合の成功」であり、skip も success に含まれる。skip 率は別途集計が必要（未計測）。

## 3. 必須化の条件（提案）

| 条件 | 内容 | 現状 |
|---|---|---|
| C1 対象の固定 | 必須化するテストのリストを明示し、増減を PR でレビュー | 未実施（対象はコメント記述のみ） |
| C2 連続成功率 | 直近 30 run で失敗 0、かつ skip 0（または skip 理由の記録） | 19/19 success、skip は未集計 |
| C3 環境依存性 | apt/LLVM の取得失敗をリトライで吸収できること | apt リトライあり。実測の失敗率は未計測 |
| C4 実行時間 | 30 分以内に安定して完了 | 実績は timeout 内（個別時間は未集計） |
| C5 失敗時の扱い | 必須化後は PR をブロックする。誤検知時の運用（再実行・除外）を文書化 | 未整備 |

## 4. 必須化の影響

| 案 | 内容 | 利点 | リスク |
|---|---|---|---|
| A 現状維持（推奨） | best-effort のまま | PR を止めない | データ競合の検出がゲートにならない |
| B 段階的必須化 | C1〜C5 を満たした範囲のみ必須 | 検出漏れを減らす | skip の扱いを誤ると形骸化する |
| C 全面必須化 | 全 TSan ジョブを必須 | 最も厳格 | 環境依存で PR が止まる |

## 5. 次に必要な計測

1. 直近 30 run の skip 率と各ジョブの実行時間
2. TSan で検出された既知の競合（0 件か）
3. 対象テストの拡大候補（`ScopedEpochReaderStructuralTests` 等、MKL/IPP 非依存のもの）

## 6. 結論

- 現時点で必須化しない（Owner 判断どおり）。
- 必須化する場合は案 B（範囲限定）を推奨。C1〜C5 の充足を条件とする。

## 7. Owner 判断（2026-11-05 追加）

- 当面 best-effort を維持し、必須ゲート化しない。
- 必須化する場合の対象テスト・失敗時の扱い・安定性評価基準を本資料に整理する。実装は別判断とする。
- CI 設定（workflow）は変更しない。

## 8. 必須化する場合の対象テスト（案）

| 区分 | テスト | 状態 |
|---|---|---|
| 現行 | `ISRSemanticValidationTests` | workflow がビルドする唯一の対象 |
| 追加候補 | `ScopedEpochReaderStructuralTests` | MKL/IPP 非依存の見込み。Linux での configure は未確認 |
| 追加候補 | `ScopedEpochReaderTests` | 同上 |
| 追加候補 | `ISRRuntimeIdentityGeneratorsTests` | 同上 |
| 追加候補 | `MpscBoundedRingTests` | 同上 |
| 追加候補 | `RetireGraceSemanticsTests` | 同上 |

- 候補の採否は、Linux ランナーで `cmake --build ... --target <name>` が通るかで決める。Windows では判定できない。
- 対象リストは workflow に直書きせず、1 か所（例: `cmake/tsan-targets.cmake`）にまとめ、PR で差分が見えるようにする。

## 9. 失敗時の扱い（案）

| 状態 | 失敗時 | skip 時 |
|---|---|---|
| best-effort（現行） | step summary とログに残し、PR は止めない | 成功扱い。理由を step summary に出す |
| 必須化後 | PR を止める。誤検知は再実行 1 回まで | 成功扱いにしない。skip 理由を記録し、環境要因なら除外理由を残す |

- 必須化後にフレークが再発した場合は、該当テストを一時除外し、除外理由と issue 番号を対象リストに書く。除外のまま必須化を維持しない。

## 10. 安定性評価基準

| 基準 | 閾値 | 測定 |
|---|---|---|
| 連続成功率 | 直近 30 run で `fail` 0 | `gh run list --workflow sanitizer-ci.yml` |
| skip | 0（または環境要因として記録済み） | 同上。skip の検出はログの `[TSan] Skipped` |
| 実行時間 | 30 分以内、p95 25 分以内 | ジョブの started/completed |
| 環境依存 | `apt`／`LLVM` 取得失敗がリトライで回復する | 失敗ログ |

- これらを満たしたうえで、必須化の範囲を別途承認する。

