# 最終確認: Owner 判断 6 件の現状照合

- 作業日時: 2026-11-24 10:00
- 対象: `main` `ea34495e`（統合クローン `C:\VSC_Project\integ\ConvoPeq`）
- 目的: 2026-11-24 の判断一覧 6 項目が、現在のリポジトリと保護ツリーにどう反映されているかを 1 枚で照合する

## 結論

6 項目すべてが意図どおりの状態にある。保護ツリーは 398 エントリのまま 1 バイトも変わっていない。統合クローンの main は `ea34495e`（判断時点の `75e64841` から 10 マージ進んだ状態）で、クリーン、Release `ctest` 69/69。

判断時点から main が進んだのは、判断 2〜5 を実行した結果である。製品コードの変更はそれぞれ独立した PR で入っており、判断の枠を超えた変更は無い。

## 照合表

| 判断対象 | 判断 | 現在の状態 | 根拠 |
| --- | --- | --- | --- |
| 日常作業の移行先 | `C:\VSC_Project\integ\ConvoPeq` | 移行済み。セッションの作業ディレクトリもこのツリー | PR #279、`doc/work113/20261124-0145_移行記録/report.md` |
| 保護ツリー | 凍結維持・書き込みなし | `4d476d5c` のまま。398 エントリ（変更 43・未追跡 355）、manifest のハッシュが全行一致 | 下記「保護ツリーの再照合」 |
| 保全ブランチ `adfb16f6` | 凍結維持 | ブランチは残置。以後のコミット無し | `git log adfb16f6..HEAD` に当該ブランチの追加コミット無し |
| ASIO 警告 12 行 | 条件付きで独立 PR | PR #275 でマージ。判断関数とテストを分離して実装 | `src/AsioBlacklistSetup.h`、`src/tests/AsioBlacklistSetupTests.cpp`、`src/MainWindow.cpp:310-342` |
| `opencode.json` の protocol 行 | main に載せない | main の graphify エントリに `protocol` キーは無い。ローカルの `.opencode/opencode.json`（gitignore 済み）にのみ存在 | 下記「設定の分離」 |
| B-07 残り検証 | 独立課題として継続 | Issue #278 で完了。PR #284 をマージし、Issue は自動クローズ | `doc/work113/20261124-0800_B07同期/report.md` |
| B-11 証跡検証 | 実測必須 artifact の一覧を先に確定 | 一覧を確定済み（11 種＋live、必須区分、gap 5 件）。実装 5 項目は未着手 | `doc/work113/20261124-0200_B11一覧/report.md` |
| Issue #265 / #266 | #265 を先行 | `acc8c964`（#265）→ `0c35f37c`（#266）の順でマージ。両方クローズ | `git log --oneline` |

## 保護ツリーの再照合

保全時に作った `manifest.tsv`（`%TEMP%\opencode\refl20261124\`）と、いまの保護ツリーから取り直した一覧を全行比較した。

- エントリ数: 398（変更 43・未追跡 355）。`git status --porcelain` の既定表示では未追跡ディレクトリが畳まれて 196 行になるので、数えるときは `-uall` を使う。
- 比較項目: 状態 2 文字、パス、サイズ、SHA-256。
- 結果: 398 行すべて一致。欠損ファイル 0 件、リネーム 0 件。

保護ツリーの HEAD は `4d476d5c` のままで、チェックアウトもされていない。

## ASIO 警告の 4 条件

| 条件 | 結果 | 根拠 |
| --- | --- | --- |
| 通常の GUI 起動で警告が出る | 観測済み。`ASIO blacklist` のタイトルでウィンドウが出る | PR #275 の実機確認、`doc/work113/20261124-0140_ASIO警告/report.md` §5 |
| CLI・テストで不要なダイアログが出ない | `--cli-run --cli-exit-ms 2000` で終了コード 0、ダイアログ 0 件。今回 main で再実行しても 4.2 秒で終了、警告なし | 今回の再実行（`build\ConvoPeq_artefacts\Release\ConvoPeq.exe --cli-run --cli-exit-ms 2000`） |
| 作成失敗とフォールバック失敗の再現 | 実行ファイル側とユーザー側を同名ディレクトリにして両方の失敗を再現し、警告が出ることを確認 | PR #275 の再現手順（退避・復元済み） |
| C-16 のログとフォールバックを壊さない | フォールバック経路とログ 3 行は不変。判断関数の追加のみ | `src/MainWindow.cpp` の差分、`AsioBlacklistSetupTests`（A-1〜A-6 の 25 検証） |

## 設定の分離

main の `opencode.json` の graphify エントリは `protocol` キーを持たない。統合クローンでは `.opencode/opencode.json`（`.gitignore` の `.opencode/` で除外）に上書きを置き、`protocol` と自前グラフ `graphify-out/graph.json` を指定している。`opencode mcp list` で graphify は `connected`。

## 検証

| 項目 | 結果 |
| --- | --- |
| Release `ctest`（main `ea34495e`） | 69/69 合格（194.90 秒） |
| 保護ツリー manifest 全行比較 | 398/398 一致 |
| ASIO 自動実行 | 終了コード 0、`[FAULT]` と drain タイムアウトなし |
| B-07 の opt-in 試験 | `CONVOPEQ_NS_RESTORE_SYNC_TEST=1` と `CONVOPEQ_NS_STATE_RACE_TEST=1` が PASS |

## 未着手として残すもの

- B-11 の実装 5 項目（`measured` の意味修正 → 検証器の必須集合 → `forceFullEvidence` の配線 → telemetry 経路 → Release の HB 扱い）。今回の判断は「一覧の確定まで」であり、そこまで完了している。
- Issue #252（S2 の長期再現性確認と ASIO 初期化の深掘り）。48 時間以上あけた再測定が必要になった段階で実施する。
