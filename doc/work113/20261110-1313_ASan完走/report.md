# 調査 C: A-02 追加後の ASan 完走確認

## §1 実施した調査・測定

### 1.1 ビルド構成

| 項目 | 値 |
|---|---|
| ビルドディレクトリ | `build-asan-local` |
| 構成 | `RelWithDebInfo`（`CMakeCache.txt` の `ENABLE_ASAN:BOOL=ON`） |
| ジェネレータ | `Ninja Multi-Config`（`--config RelWithDebInfo`） |
| コンパイラ | MSVC `cl.exe`（VS 18 / MSVC 14.51.36231、`CMAKE_CXX_COMPILER` の実値） |
| ASan ランタイム | `clang_rt.asan_dynamic-x86_64.dll` を実行ファイル隣へコピー |
| 対象 | `AudioEngineHarness`（B-01/B-03/B-04/B-09/B-10 と A-01/A-02 を含む） |

### 1.2 実行コマンド

```cmd
call "%VS%\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_x64=x64
set "MKLROOT=...\mkl\2026.2"
set "IPPROOT=...\ipp\2026.1"
set "LIB=%MKLROOT%\lib;%IPPROOT%\lib;%LIB%"
set "CONVOPEQ_SOURCE_ROOT=C:\VSC_Project\ConvoPeq"
cd /d C:\VSC_Project\ConvoPeq
cmake --build build-asan-local --config RelWithDebInfo --target AudioEngineHarness
copy /y clang_rt.asan_dynamic-x86_64.dll build-asan-local\RelWithDebInfo\
build-asan-local\RelWithDebInfo\AudioEngineHarness.exe
```

`CONVOPEQ_SOURCE_ROOT` は `STG11D18CrossfadeTests` の D18-D が要求する（未設定だと構造検査が失敗する）。

### 1.3 実行結果

| 項目 | 結果 |
|---|---|
| 実行完了 | **した**（中断ではない） |
| 終了コード | **0**（`harness_exit=0`） |
| ログ行数 | 397 行 |
| `ERROR: AddressSanitizer` の出現 | **0 件** |
| 最終行 | `AudioEngineHarness: all publish pipeline tests PASS` |

### 1.4 A-02 のカバレッジ

ハーネスの IR 検査行に、A-01 と A-02 の両方が含まれて PASS している。

```
IRLoadAdmissionTests: PASS (WORK102 A/B/C/D/E/F/G/H/I/L + B01 I-1..I-6 + B01 T-1..T-5 + A-01/A-02 + B04 RG + B10 PB)
```

A-02 は「同じ無音テール入力でプレビュー（`analyzeImpulseResponseFile` 96k）と本適用（`prepareToPlay` 96k + `loadAndSettle`）の受理結果が一致すること」を検査する。ASan 下でこの行が PASS しているため、A-02 が通る経路（`IRConverter::convertFile` のトリム、`LoaderThread` のトリム、プレビュー解析）は ASan 実行に含まれている。

### 1.5 併せて実施した ASan 実行（ターゲット別）

| ターゲット | 実行時刻 | 結果 |
|---|---|---|
| `LearnerStateAuthorityTests`（B-07） | 13:18 | `failures=0`、`authority_exit=0`、ASan 報告 0 件（R-07-A〜E の 9 件 PASS） |
| `LearnedStateMigrationTests`（B-15） | 09:23 | `failures=0`、`migration_exit=0`、ASan 報告 0 件（M-15-1〜13 PASS） |
| `CacheManagerHeaderValidationTests`（B-01/B-02、C-12） | 12:04 | `failures=0`、ASan 報告 0 件（C-12a/b/c PASS） |
| `TailLayerIntegrityTests`（B-12） | 09:23 | `failures=27`、`taillayer_exit=1`。**27 件すべて `RES-F`（CPU 閾値）**で、`RES-F` 以外の失敗は 0 件。ASan 報告 0 件 |

**過去の試行（完了として扱わない）**: 09:23 の一括実行では `LearnerStateAuthorityTests` の起動に失敗していた。実行ファイルのパスを `LearnerStateAuthorityTests_artefacts\RelWithDebInfo\` としていたが、実際の出力先は `build-asan-local\RelWithDebInfo\` であり、`指定されたパスが見つかりません。` と `authority_exit=3` だけが記録されていた。この回は B-07 を検査していない。13:18 に正しいパスで実行し直した結果（上表）を採用する。

`LearnedStateMigrationTests` と `TailLayerIntegrityTests` のバイナリは 09:23 にビルドしたもので、対象ソース（`LearnedStateMigration.cpp` 08:33、`NoiseShaperLearner.cpp` 08:33、`LearnerStateAuthority.h` 08:05）はその後変更されていない。`AudioEngineHarness` は A-02 追加（`IRLoadAdmissionTests.cpp` 12:05）の後に 12:26 に再ビルドしている。

## §2 直接確認できた事実

1. A-02 を含む ASan ハーネスは**最後まで実行され、終了コード 0** だった。
2. sanitizer の出力は **0 件**（`ERROR: AddressSanitizer` の文字列がログに無い）。
3. ハーネスは「all publish pipeline tests PASS」で終了している。
4. A-02 の PASS 行がログに存在する。
5. 以前の試行（A-02 追加直後、12:32）は 102 行まで進んだところで**こちらが停止**させた（`harness_exit=-1`）。停止までの出力にも sanitizer の報告は無かったが、**完走ではない**ため、本 §1 の実行を完了の根拠とする。
6. 09:23 の一括 ASan 実行では `LearnerStateAuthorityTests` が起動できず（`authority_exit=3`）、B-07 は ASan で検査されていなかった。13:18 に再実行して `failures=0`・ASan 報告 0 件を確認した。

## §3 事実から導いた判断

- A-02 を含む ASan 実行は完了しており、sanitizer の報告は無い。A-02 の経路に対する ASan の確認は完了した。
- ただし、これは **ASan が検出できる範囲**（メモリ破壊、未初期化読みの一部）の確認である。データ競合（TSan）やリーク（LSan）の確認ではない。
- `TailLayerIntegrityTests` の 27 件は CPU 閾値（`RES-F`）による失敗である。ASan の実行速度低下が原因と推定する（同一ケースが Release では合格している）。メモリ関連の失敗ではない。

## §4 未確認事項と制約

- TSan は使用できない。実測した根拠は次の 2 点である。B-07 のデータ競合は ASan では検出できない。
  - `cl.exe /fsanitize=thread` は `warning D9002: 不明なオプション '/fsanitize=thread' を無視します` となり、指定が無視される。
  - `clang-cl.exe -fsanitize=thread` は `error: unsupported option '-fsanitize=thread' for target 'x86_64-pc-windows-msvc'` で失敗する。
- リーク検出も使用できない。`ASAN_OPTIONS=detect_leaks=1` を付けて実行すると `AddressSanitizer: detect_leaks is not supported on this platform.` と表示され、終了コード 1 になる（リークさせた試験プログラムで確認）。B-14 のリーク検査は未実施のまま。
- `TailLayerIntegrityTests` は ASan 下で CPU 閾値が成立しないため、機能判定は Release の結果に依存する。
- ハーネスに含まれない他のテストターゲット（例: `PublishPipelineIntegrationTests` 単体、`ISRSoakTests`）は ASan で実行していない。
- 実機の音声デバイスを使う経路（`HeadlessAudioPathVerification`）は ASan 対象外である。

## §5 追加変更が必要な場合の提案（未実施・承認待ち）

- 追加の変更は不要。今回の目的（A-02 を含む ASan 完走の確認）は達成した。
- 参考: ASan 下の `TailLayerIntegrityTests` を意味のある形で回すには、CPU 閾値を ASan 用に分ける（テストの閾値変更）必要があるが、これは閾値の緩和に当たるため提案しない。ASan ではメモリ検査のみを目的とし、機能・性能判定は Release に限定する現在の運用が妥当である。
