# icx による Release ビルド（2026-11-22 10:32 JST）

`build.bat` を使って Intel oneAPI の icx で Release をビルドし、`ConvoPeq.exe` を生成した記録である。テスト用の実行ファイルは作っていない。結果は成功した。所要は 8.4 分。アプリの起動確認まで通っている。

## 1. 結果

| 項目 | 値 |
| --- | --- |
| コマンド | `build.bat Release clean icx app nopause` |
| 終了コード | 0 |
| 所要 | 8.4 分 |
| 生成物 | `build-icx\ConvoPeq_artefacts\Release\ConvoPeq.exe` |
| サイズ | 48,817,152 バイト |
| 更新時刻 | 2026-11-22 10:31:28 |
| SHA-256 先頭 | `302d935c1d0ed0ee` |
| ビルド対象 | `ConvoPeq` のみ（テストターゲットは生成していない） |
| コンパイル手順 | 141 ステップ |
| 警告 | 156 行（icx の警告） |
| エラー | 0 |

## 2. 手順

1. PATH から `C:\Program Files\GNU Octave\Octave-11.3.0\mingw64\bin` を外した。このパスが残っていると、CMake が MSVC の `link.exe` / `lib.exe` ではなく Octave の `ld.exe` / `ar.exe` を検出して構成に失敗する。
2. `build.bat` に `Release clean icx app nopause` を渡した。`clean` は、後述のビルド同一性ゲートが要求する復旧手順である。`app` により `ConvoPeq` ターゲットだけをビルドする。
3. `build.bat` が oneAPI の `setvars.bat` を呼び、`MKLROOT`、`IPPROOT`、`LIB` を設定する。MSVC の `vcvarsall.bat` は呼ばない（icx モードの仕様）。
4. 構成後、`src\tools\build_identity_gate.py` が同一性を刻印し、ビルド前に照合する。今回は clean 後の新規構成なので、新しい刻印で通った。

## 3. ビルド環境

| 項目 | 値 |
| --- | --- |
| コンパイラ | `C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin\icx.exe`（2026.2） |
| ジェネレータ | Ninja Multi-Config |
| CMake | 4.4.5 |
| Release フラグ | `/O2 /Ob2 /DNDEBUG` |
| リンク | LTCG（構成時の表示は `Normal Release build with LTCG`） |
| 並列度 | `-j 2`（build.bat が icx 用に固定） |

`build-icx\CMakeFiles\rules.ninja` のコンパイル行とリンク行は、どちらも `icx.exe` を指している。`CMakeCache.txt` の `CMAKE_CXX_COMPILER` も同じフルパスである。

## 4. 起動確認

生成した exe を CLI モードで 1.5 秒動かした。

| 項目 | 値 |
| --- | --- |
| コマンド | `ConvoPeq.exe --cli-run --cli-exit-ms 1500 --cli-log-file <temp>` |
| 終了コード | 0 |
| 実時間 | 3.5 秒 |
| 音声 | 192 kHz、ブロック 1,024 |
| 処理時間 | 平均 521〜535 µs、最大 900 µs（2 回の計測区間） |
| EQ | `createdHash=0x32ef8b016e60b949 gen=4` |

DLL 依存は `ole32`、`AVRT`、`d3d11` などの Windows 標準で、Intel のランタイム DLL は要求していない（静的リンク）。

## 5. 逸脱と注意

- **clean が必要だった理由**: 既存の `build-icx` は、以前の構成で `compiler_path` に icx のフルパスを刻印していた。今回 `build.bat` が `-DCMAKE_CXX_COMPILER=icx` で構成した。そこでゲートが「刻印はフルパス、現在は `icx`」の不一致を検出して停止した。指示どおり clean で作り直した。コンパイラ自体は同じ 2026.2 だ。
- **表示だけの 2 件**: ログの先頭に `'M' is not recognized` が出る。UTF-8 の日本語 REM 行が、コンソールのコードページ切り替えの途中で分割されるためである。もう 1 件は clean 時の `ERROR: Input redirection is not supported` で、`timeout /t 2` が標準入力のリダイレクト下で失敗したものだ。どちらも終了コード 0 で、生成物に影響しない。対話コンソールから実行すれば出ない。
- ビルドが書き換えたのは `build-icx` だけである。`.gitignore` の `build-*/` に一致し、追跡対象ではない。ソースツリーの未コミット 194 件は触っていない。

## 6. 参照

- ビルドスクリプト: `build.bat`
- 同一性ゲート: `src\tools\build_identity_gate.py`
- ビルドログ: `C:\Users\user\AppData\Local\Temp\opencode\icx_build_clean.log`
- 起動ログ: `C:\Users\user\AppData\Local\Temp\opencode\icx_smoke.log`
