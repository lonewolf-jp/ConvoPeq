# RES2-11 TSan 限定必須化の判断材料（2026-11-06）

- 本資料は 2026-11-06 の Owner 判断で、評価の判断材料として採用された。実行・CI 変更は承認されていない。


- 位置づけ: 実行承認ではない。限定必須化を判断するための材料。
- 前提: 現行設定は `continue-on-error: true`、対象は MKL/IPP 非依存のテストのみ（`.github/workflows/sanitizer-ci.yml:256-259`）。

## 1. 評価条件の充足状況

| # | 条件 | 現状 | 充足に必要なもの |
|---|---|---|---|
| 1 | 対象テストの固定と再評価条件 | 未充足。workflow は `ISRSemanticValidationTests` をビルドするのみで、対象リストが定義されていない | 対象リストのファイル化、変更時の再評価ルール |
| 2 | skip を成功扱いしない。理由・発生率・環境依存性の記録 | 未充足。skip しても success になる。標本 10 run はすべて skip | skip の明示記録、skip 率の集計 |
| 3 | 連続成功率と失敗時の分類 | 一部。直近 30 run は success 26・未完了 4 だが、success の大半は skip | skip を除いた成功率、失敗分類 |
| 4 | 実行時間と CI コスト | 未充足。ジョブは 0.4〜1.1 分（skip のため）。実 TSan の時間は未測定 | TSan が実際に動く環境での実測 |
| 5 | 利用可能／不可能な環境の分離 | 未充足 | 専用環境の定義と、それ以外での扱い |

## 2. 実測データ

| 項目 | 値 | 取得方法 |
|---|---|---|
| 直近 30 run | success 26、未完了 4 | `gh run list --workflow sanitizer-ci.yml --limit 30` |
| 完了 10 run の TSan ジョブ | 0.4〜1.1 分 | `gh run view --json jobs` |
| ログ判定 | `[TSan] Skipped (no build configured — MKL/IPP not available on Linux)` | `gh run view --job <id> --log` |
| 実行された TSan テスト | 0（標本すべて skip） | 同上 |

- skip の原因は 2 段階ある。`Check MKL/IPP availability` は clang++ の有無しか見ないため skip=false になる。しかし Configure が MKL/IPP 不在で失敗し、`build.ninja` が無いため Build ステップが skip する。
- したがって、現行の success は TSan の検出力を何も示さない。

## 3. 対象テストの固定（案）

| 区分 | テスト | 根拠 |
|---|---|---|
| 現行 | `ISRSemanticValidationTests` | workflow がビルドする唯一の対象 |
| 追加候補 | `ScopedEpochReaderStructuralTests` | MKL/IPP 非依存の見込み。Linux configure は未確認 |
| 追加候補 | `ScopedEpochReaderTests` | 同上 |
| 追加候補 | `ISRRuntimeIdentityGeneratorsTests` | 同上 |
| 追加候補 | `MpscBoundedRingTests` | 同上 |
| 追加候補 | `RetireGraceSemanticsTests` | 同上 |

- 対象は 1 か所（例: `cmake/tsan-targets.cmake`）にまとめ、PR の差分で増減が見えるようにする。
- 再評価条件: 対象テストの追加・削除、`ENABLE_TSAN` の設定変更、TSan ランタイムの更新、連続 3 回の失敗。
- 候補の採否は、Linux ランナーで `cmake --build --target <name>` が通るかで決める。Windows では判定できない。

## 4. skip の扱い（案）

| 状態 | 現行 | 提案 |
|---|---|---|
| ビルド不能 | skip を success として報告 | `::warning::` と step summary に「skip（理由）」を明示し、skip 件数を集計できる形にする |
| テスト 0 件 | `[TSan] No TSan-compatible tests executed` を出すが success | テスト 0 件は成功にしない（少なくとも `exit 1`。必須化前は warning でもよい） |
| 記録 | なし | 実行ごとに「対象・実行数・skip 理由」を artifact に保存 |

- skip 率 = skip した run / 全 run。分母は workflow の実行回数。
- 記録項目: run id、日時、対象リストのハッシュ、configure 成否、実行テスト数、失敗数、skip 理由。

## 5. 連続成功率と失敗分類（案）

| 分類 | 例 | 扱い |
|---|---|---|
| 環境要因 | apt/LLVM 取得失敗、configure 失敗 | リトライで回復したら失敗に数えない。回復しなければ環境失敗として記録 |
| TSan 検出 | データ競合の報告 | 失敗。必須化後は PR を止める |
| テスト失敗 | 既存テストの不成立 | 失敗。TSan 固有か切り分ける |
| skip | ビルド不能 | 成功に数えない |

- 連続成功率は「skip を除いた実行」に対する成功の割合で計算する。skip を含めた success 率は参考値に留める。

## 6. 実行時間と CI コスト（見積り）

| 項目 | 値 | 根拠 |
|---|---|---|
| 現行ジョブ | 0.4〜1.1 分（skip） | 実測 |
| TSan 実実行（見積り） | 10〜30 分 | Clang の TSan ビルドとテスト。実測は未取得 |
| timeout | 30 分 | workflow |
| 課金 | 公開リポジトリのため標準ランナーは無料枠 | GitHub の公開リポジトリの扱い |

- 実測は、TSan が動く Linux 環境で 1 回実行して取得する。それまでコストは見積りに留める。

## 7. 環境の分離（案）

| 環境 | TSan | 扱い |
|---|---|---|
| 専用 Linux ランナー（clang、MKL/IPP 不要の構成） | 実行する | 限定必須化の対象。対象テストのみ必須 |
| GitHub-hosted ubuntu | 実行できない（現状 skip） | 必須化しない。参考実行のみ |
| Windows 開発機 | 実行しない | 対象外 |

- 限定必須化は、専用ランナーで対象テストが実際に走ることを確認してから行う。成功しても、完全なスレッド安全性の証明とはしない。

## 8. 判断材料のまとめ

- 現状は条件 1〜5 のすべてが未充足である。特に skip が success になるため、成功率の数字は検出力を表さない。
- 次に必要なのは、専用 Linux 環境で 1 回 TSan を実走させ、対象テスト・実行時間・失敗分類を実測することである。
- その実測が揃うまで、必須化の判断はできない。

## 9. 実走の試行結果（2026-11-06）

- WSL（Ubuntu 26.04、clang 21.1.8、cmake 4.4.4、ninja 1.13.2）で configure を試した。
- 結果: `Could NOT find PkgConfig (missing: PKG_CONFIG_EXECUTABLE)` で停止した。
- 構造上の問題: `ENABLE_TSAN` は `ConvoPeq` ターゲットにのみ `-fsanitize=thread` を付ける。`ISRSemanticValidationTests` は TSan で計装されない。加えて、テストターゲットは `ConvoPeq` に依存するため、Linux では JUCE GUI の全依存が必要になる。
- したがって、現行構成のままでは TSan の診断実走を行ってもテストは TSan で動かない。証拠を得るには、テストターゲットへの TSan 付与（CMakeLists の変更）と、WSL への pkg-config・JUCE 依存の導入が必要で、いずれも追加承認の範囲である。
- 実走は未実施。実測実行時間も未取得（skip 時のジョブ時間 0.4〜1.1 分のみ）。

