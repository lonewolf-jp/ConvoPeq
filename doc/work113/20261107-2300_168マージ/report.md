# PR #168（N1）のマージと main 検証・`ConvoPeq.md` 再生成

- 実施: 2026-11-07 23:00
- 承認: PR #168 のマージ（Owner 判断・2026-11-07）
- 承認条件: (1) 受入対象外・未評価の明記、(2) N1 の回帰なしと既存設計の制約の区別、(3) 原因分析・対応を別 PR 扱いにする、(4) CI CLEAN・ctest 54 / 54・差分とコミットの整合性の維持
- 対象: N1（バイパス切替の 20 ms テールクロスフェード）

## 1. PR #168 のマージ

| 項目 | 値 |
|---|---|
| PR | #168（`fix(conv): バイパス切替のテールクロスフェード（N1）`） |
| マージ方式 | merge commit（#166・#167 と同じ。squash ではない） |
| マージ前 head | `e79a6a3f` |
| PR の CI（`e79a6a3f`） | 14 pass / 1 skip / 0 fail、`CLEAN` |
| マージ commit | `32c9234b` |
| 変更規模 | 15 ファイル、+1,234 / −3。うちソースコードは `src/` の 6 ファイル（+822 / −2）と `CMakeLists.txt`（+2）、残り 8 ファイルは報告書 4 件・CSV 3 件・残存課題レジスタ 1 件 |
| マージ日時 | 2026-11-07 22:58（JST） |
| ローカルの `main` | `32c9234b`（`origin/main` と一致） |

### マージ前の確認

- `git diff --stat main...HEAD` は 15 ファイル、+1,234 / −3。コードは `src/ConvolverProcessor.h`、`ConvolverProcessor.Lifecycle.cpp`、`ConvolverProcessor.Runtime.cpp`、`ConvolverStateRoundTripTests.cpp`、`N1BypassAudioAnalysis.cpp`、`PublishPipelineIntegrationTests.cpp`、`CMakeLists.txt` の 7 ファイル。残り 8 ファイルは報告書 4 件、CSV 3 件、残存課題レジスタ 1 件である。
- 作業ツリーの追跡変更は 0。`N1BypassAudioAnalysis.cpp` は作業ツリーと `HEAD` の blob ハッシュが一致（`dfad851e`）。
- `git ls-remote` の `fix/bypass-tail-crossfade` は `e79a6a3f` で、ローカル `HEAD` と一致。

## 2. 承認条件への対応

| 条件 | 対応 | 確認先 |
|---|---|---|
| 1. 受入対象外・未評価と明記（PASS と混同しない） | 報告書に「受入対象と受入対象外」の表を追加。9 / 9 PASS は T1・T3・T4・T5・T6 に対する結果で、当該項目を含まないと明記 | `doc/work113/20261107-2140_N1自動検証/report.md` §3・§5・§9 |
| 2. N1 の回帰なしと既存設計の制約の区別 | 「N1 の回帰の有無（既存設計の制約との区別）」節を新設。回帰なしの根拠は修正前 `main` との同一条件比較と ctest | 同 §6 |
| 3. 原因分析・対応を N1 に混入させない | 報告書と PR 本文に「別 PR で起票・検討する」と明記。本 PR のコード差分は N1 の 6 ファイルと CMake のみ | 同 §6、PR 本文 |
| 4. CI CLEAN・ctest 54 / 54・整合性の維持 | マージ前 head の CI は `CLEAN`（14 pass / 1 skip / 0 fail）。マージ後の `main` で再ビルド・ハーネス・ctest を実施（§3） | §3 |

マージ前の PR 本文にも、Owner 判断と 4 条件を追記した。

## 3. `main`（`32c9234b`）の検証

| 検証 | コマンド | 結果 |
|---|---|---|
| ビルド | `cmake -S . -B build` → `cmake --build build --config Release -j 8` | 成功（configure 0 / build 0、コンパイル・リンクエラー 0） |
| ハーネス | `build\Release\AudioEngineHarness.exe` | exit 0、`AudioEngineHarness: all publish pipeline tests PASS` |
| `ctest` | `ctest --test-dir build -C Release --output-on-failure` | 54 / 54 PASS（334.24 秒） |
| N1 自動分析の再実行 | `build\Release\AudioEngineHarness.exe --n1-analysis <出力先> post` | 9 / 9 PASS。`metrics_post.csv` は `doc/work113/20261107-2140_N1自動検証/metrics_post.csv` とバイト単位で一致（再現性の確認） |
| CI（`32c9234b`、`main`） | GitHub Actions | 5 ワークフローすべて success。audioengine-lint・list-compliance・ISR Authority Compliance は 22:59、Windows MSVC は 23:15、Push on main は 23:37、Sanitizer は 2026-11-08 00:06（いずれも JST） |

- ハーネスのログに `D5-T1: M4 FAILED` という行が出るが、これは MMCSS 観測テストが記録するマイルストーン名であり、同スイートは `PASS (D5-T1/…/D5-T6)` で終了している。失敗ではない。
- ビルドツリーの再構成が必要だった。`$ENV{IPPROOT}` を設定せずに CMake が再構成されると、IPP の include が `/include` になり `ipp.h` が見つからない。`MKLROOT`・`IPPROOT`・`ONEAPI_ROOT` を設定して `cmake -S . -B build` を実行した。

## 4. `ConvoPeq.md` の再生成

#166・#167・#168 でソースが変わったため、`32c9234b` を基準に再生成した。

| 項目 | 値 |
|---|---|
| 生成元 `main` | `32c9234b` |
| 生成日時 | 2026-11-07 23:01:03 |
| 生成コマンド | `python output_sourcecode_markdown.py` |
| 自動保存されたスナップショット | `doc/ConvoPeqMD/20261107230103/ConvoPeq.md`（CRLF、6,723,042 バイト） |
| 鮮度検査 | `NEWER_SRC_COUNT: 0`、`STATUS: FRESH` |
| サイズ | 6,573,934 バイト（LF、149,108 行） |
| SHA-256 | `4f4b857a5fd317e747a3a35d9a111e86a821820ab0383b0d4bdbc70d631e5224` |
| 前回（`568aa10e` 時点の `ConvoPeq.md`）との差 | 1,054 行追加・8 行削除（#1A・#12・N1 のソースとヘッダの生成時刻） |

生成直後は CRLF（149,108 箇所）だったため、リポジトリの `*.md text eol=lf` に合わせて LF へ正規化した。

### 分割版（1 ファイル 2 MiB 以下）

| ファイル | サイズ（バイト） | SHA-256 |
|---|---|---|
| `ConvoPeq.part01.md` | 2,097,131 | `3faa7afbdf1b1725515675250b54ce92236f486f0d00274a72974f380eb4993a` |
| `ConvoPeq.part02.md` | 2,097,146 | `ea915f4757dcd0252afed59d962a1a74aae1e5dd1fbb72072fc96876326076ff` |
| `ConvoPeq.part03.md` | 2,097,121 | `a636ffe21f872d72da22b5f6248303aabaf433577a970eee969d6d99945bfa65` |
| `ConvoPeq.part04.md` | 282,536 | `f7a21b7acb5420d00a8fd1121d755a73c584e5c9d4aac12cdb7bff35e1484bfd` |

4 分割。最大は 2,097,146 バイトで 2 MiB（2,097,152 バイト）以下。4 ファイルを順に結合すると元ファイルとバイト単位で一致する（`bytes_equal = True`）。分割版はリポジトリに置かず、Dropbox へ納品する（従来と同じ扱い）。

## 5. 記録の更新

| 資料 | 更新 |
|---|---|
| `doc/work113/20261107-2140_N1自動検証/report.md` | 受入対象・受入対象外の区分、回帰の有無と既存設計の制約の区別、Owner 判断とマージ commit を追記 |
| `doc/work113/20261106-2030_残存対応/01-residual-register.md` | 現在の `main` を `32c9234b` へ。#1 残存 A（PR #166）・#12（PR #167）・N1（PR #168）をマージ済みへ。承認範囲の表も更新 |
| `doc/work113/20261106-2030_残存対応/00-owner-decision.md` | 対応状況の表を #1A・#12・N1・`ConvoPeq.md` の現状へ更新（§1 の決定本文は履歴として維持） |
| `doc/work113/20261106-2030_残存対応/04-convopeq-md-sha256.md` | §9 に 3 回目の生成（`32c9234b`）の条件・サイズ・SHA-256・分割版を追記 |

## 6. Dropbox への納品

`/ConvoPeq-168merge-20261107/` に 13 ファイルを配置した（アップロード後に `ConvoPeq.md` をダウンロードし、バイト一致を確認）。

- `ConvoPeq.md` と分割版 4 ファイル（`ConvoPeq.part01.md`〜`part04.md`）
- `report-merge168.md`（本報告書）、`report-n1-analysis.md`（N1 自動検証）
- `comparison.csv`、`metrics_post.csv`、`metrics_pre.csv`
- `01-residual-register.md`、`00-owner-decision.md`、`04-convopeq-md-sha256.md`

同じフォルダに同名の `report.md` が衝突するため、報告書は上記の名前に変更して配置した。

## 7. 状態

| 項目 | 状態 |
|---|---|
| PR #168 | MERGED（`32c9234b`） |
| N1 | 解消済み（受入対象 5 指標 9 / 9 PASS） |
| 受入対象外・未評価 | バイパス解除後の wet 絶対レベル一致（既存設計。必要なら別 PR） |
| `main` | コードは `32c9234b`、本記録と `ConvoPeq.md` は `e7235420` で反映済み。最新の head は git 履歴を参照 |
| `ConvoPeq.md` | `32c9234b` のソースから再生成（FRESH、SHA-256 `4f4b857a…e5224`） |
| 分割版 | 4 ファイル、最大 2,097,146 バイト、結合一致 |

## 8. 反映

| 項目 | 値 |
|---|---|
| PR（本記録と `ConvoPeq.md`） | #169 |
| マージ方式 | squash |
| マージ commit | `e7235420` |
| 変更 | 8 ファイル、+1,240 / −30（`ConvoPeq.md` と記録 7 件） |
| マージ後の `main` の `ConvoPeq.md` | 6,573,934 バイト、CRLF 0（LF のみ）、SHA-256 `4f4b857a5fd317e747a3a35d9a111e86a821820ab0383b0d4bdbc70d631e5224`（生成物と一致） |
| 鮮度検査（`main`） | `NEWER_SRC_COUNT: 0`、`STATUS: FRESH` |
| PR #169 の CI | 必須チェック `Windows MSVC / Release` の pass を確認してマージ（ASan と CodeQL は実行中） |
| 最終記録（本節と CI 結果の追記） | PR #170（文書のみ） |
