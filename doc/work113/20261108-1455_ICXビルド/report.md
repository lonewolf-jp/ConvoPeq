# Intel icx による Release ビルド（build.bat）

- 実施: 2026-11-08 14:10〜14:50
- 対象: `main` = `f69b5a9f`
- 目的: プロジェクトルートの `build.bat` を使い、Intel icx で Release ビルドを行う。
- 実行コマンド: `build.bat Release icx nopause`（作業ディレクトリ `C:\VSC_Project\ConvoPeq`）

## 1. 結果

| 項目 | 値 |
|---|---|
| 終了コード | 0 |
| ターゲット | 678 / 678 |
| 実行ファイル | `build-icx\ConvoPeq_artefacts\Release\ConvoPeq.exe` |
| サイズ | 48,434,176 バイト（46.19 MB） |
| 生成日時 | 2026-11-08 14:18:11（ターゲット 148/678 でリンク） |
| バージョン | 0.6.11 |
| 所要時間 | 約 39 分（14:10:56 開始、14:49:43 終了） |
| エラー | 0（`error:` 0 件、`FAILED:` 0 件） |

ログ: `C:\Users\user\AppData\Local\Temp\opencode\icx\build-icx-release.log`（4,846 行）。

## 2. コンパイラの確認

| 確認 | 結果 |
|---|---|
| `CMakeCache.txt` | `CMAKE_C_COMPILER=icx`、`CMAKE_CXX_COMPILER=icx`、`CMAKE_GENERATOR=Ninja Multi-Config` |
| `compile_commands.json` | 1,920 エントリ中 1,908 が `icx.exe`、12 が `rc.exe`（リソース） |
| 実行中プロセス | `icx.exe` が ninja の子として稼働。その子に `clang-cl.exe`（icx 内部のドライバ） |
| icx の版 | Intel(R) oneAPI DPC++/C++ Compiler 2026.2.0（Build 20261001） |
| 診断の形式 | clang 系（`-Winconsistent-missing-override`、`warnings generated` など） |

`build.bat` の icx 分岐は `-DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx` で構成し、ビルドディレクトリを `build-icx` に分離する。oneAPI は `setvars.bat intel64` で初期化され、MKLROOT・IPPROOT・ONEAPI_ROOT と LIB が設定された。

## 3. ビルド設定

| 設定 | 値 |
|---|---|
| 構成 | Release |
| PGO | なし（normal） |
| `CONVOPEQ_ENABLE_TEST_HOOKS` | OFF（通常の Release） |
| `CONVOPEQ_REQUIRE_MKL` | ON |
| 並列度 | `-j 2`（icx のメモリ使用を考慮した build.bat の既定） |

## 4. 警告

- `warnings generated` のブロックが 235 件。いずれも警告で、ビルドは成功している。
- 主なものは、MSVC ヘッダの非推奨 API 注記、未使用変数、`override` の付け忘れなどである。

## 5. 補足

- 実行ファイルの確認のみを行い、アプリの起動はしていない（稼働中の OpenChamber と二重起動になるため）。
- `build-icx` はリポジトリの追跡対象外（`.gitignore`）。
