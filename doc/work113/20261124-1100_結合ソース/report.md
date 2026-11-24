# 結合ソース ConvoPeq.md の再生成（main `7734a542`）

- 作業日時: 2026-11-24 10:42
- 対象: `main` `7734a542`（統合クローン `C:\VSC_Project\integ\ConvoPeq`）
- 目的: 今後の検討基準を `7734a542` に置くため、参照する結合ソースを同時点へ揃える

## 結果

`python output_sourcecode_markdown.py` で再生成した。生成時刻は 2026-11-24 10:42:29、`ConvoPeq.md` は 7,613,929 バイト、425 セクション。

再生成前（2026-11-24 00:06:06 版）との差は、判断一覧の作業で入った変更と完全に一致した。取りこぼしは無い。

| 種別 | 件数 | 内容 |
| --- | --- | --- |
| 追加 | 4 | `src/AsioBlacklistSetup.h`、`src/learner/SegmentBufferAllocation.h`、`src/tests/AsioBlacklistSetupTests.cpp`、`src/tests/AudioEngineHarness/NoiseShaperRestoreSyncTests.cpp` |
| 変更 | 7 | `CMakeLists.txt`、`src/MainWindow.cpp`、`src/NoiseShaperLearner.cpp`、`src/NoiseShaperLearner.h`、`src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp`、`src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp`、`src/tests/AudioSegmentBufferTests.cpp` |
| 削除 | 0 | — |

追加 4 件は ASIO 警告（#275）、#265、B-07（#284）で増えたファイルだ。変更 7 件は同じ作業の編集と、#266 の CI 修正にあたる。

## 分割版

`ConvoPeq.md` が 2 MB を超えるため、1 ファイル 2 MB 以下に分割した版も生成した。分割は `### 📄`（ファイル単位）の境界で行った。途中で切れるファイルは無い。

| ファイル | サイズ（バイト） | 収録ファイル数 |
| --- | --- | --- |
| `ConvoPeq_part01.md` | 1,922,937 | 91 |
| `ConvoPeq_part02.md` | 1,994,316 | 147 |
| `ConvoPeq_part03.md` | 1,973,144 | 106 |
| `ConvoPeq_part04.md` | 1,875,448 | 81 |

各パートの先頭には、分割番号・生成元の stamp・収録範囲（先頭と末尾のファイル名、件数）を書いたヘッダを置いた。全 425 セクションがどれか 1 つのパートに 1 回だけ現れることを確認している。

## 置き場所

- `ConvoPeq.md`: リポジトリ直下（追跡対象）。従来どおり。
- スナップショットと分割版: `doc/ConvoPeqMD/20261124104229/`。このディレクトリは `.gitignore` の `doc/ConvoPeqMD/` で除外されている。リポジトリは太らない。分割版を追跡すると、再生成のたびに約 7.7 MB が履歴に積まれる。追跡しない判断にした。

## 検証

| 項目 | 結果 |
| --- | --- |
| スナップショット同一性（`--verify-snapshot`） | `STATUS: OK`（名前 stamp・ヘッダ stamp・内容が現行と一致） |
| セクション網羅 | 425 セクションが過不足なく 1 回ずつ |
| パートのサイズ | 4 パートすべて 2,000,000 バイト以下 |
| セクション内容の保持 | 分割前後でセクション本文のバイト数が一致（7,764,783） |
| 作業ツリー | `ConvoPeq.md` 以外に差分なし（分割版は無視対象ディレクトリ） |

## 配布

`ConvoPeq.md`、分割版 4 件、本報告を Dropbox の `/ConvoPeq-source-20261124/` に置いた。
