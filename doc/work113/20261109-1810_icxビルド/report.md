# build.bat による Intel icx Release ビルド

- 目的: `build.bat` を用い、Intel oneAPI DPC++/C++ コンパイラ（icx）で Release 構成をビルドする
- 対象ソース: main `c94d3767`（作業ツリーの変更なし）
- ビルドディレクトリ: `build-icx`（`build.bat` の icx モードが使う既存ツリー）

## §1 環境

| 項目 | 値 |
|---|---|
| コンパイラ | Intel oneAPI DPC++/C++ Compiler 2026.2.0（2026.2.0.20261001、`Target: x86_64-pc-windows-msvc`） |
| icx の実体 | `C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin\icx.exe` |
| oneAPI | `C:\Program Files (x86)\Intel\oneAPI`（MKL・IPP は `latest`） |
| ジェネレータ | Ninja Multi-Config（`build.bat` が指定） |
| 構成 | Release |

`build.bat` は icx モードで `setvars.bat intel64` を呼び、`MKLROOT` / `IPPROOT` / `LIB` を自前で設定する。

## §2 手順

1. `build.bat Release icx clean app nopause` を実行する（`app` は `ConvoPeq.exe` のみをビルドし、テストターゲットはビルドしない）
2. `clean` でキャッシュを作り直すため、既定の `CONVOPEQ_CORRECT_POLYPHASE_GAIN=ON` がそのまま適用される（§3・§4）
3. 生成物を確認する

テストターゲットを含む全ビルドは不要との指示による。テストは実行しない。

## §3 引数 `-D` の扱い（実測）

`build.bat` の使い方には `-DVAR[=VALUE]` とあるが、cmd は引数を `=` で分割するため、`-DCONVOPEQ_CORRECT_POLYPHASE_GAIN=ON` と書くと `%1` が `-D...`、`%2` が `ON` になる。結果として CMake には値なしの `-DCONVOPEQ_CORRECT_POLYPHASE_GAIN` だけが渡り、次のエラーで構成に失敗する。

```text
CMake Error: Parse error in command line argument: CONVOPEQ_CORRECT_POLYPHASE_GAIN
 Should be: VAR:type=value
```

引数を引用符で囲むと `%arg:~0,2%` が `"-` になり `-D` と判定されないため、値付きの定義はこの経路では渡せない。今回は値付きの `-D` を使わず、`clean` でキャッシュを作り直して既定値（ON）を適用した。

## §4 試行の経緯とビルド識別ゲート

1. `build.bat Release icx nopause -DCONVOPEQ_CORRECT_POLYPHASE_GAIN=ON` — §3 の CMake 構成エラーで停止
2. キャッシュを ON に揃えたうえで `build.bat Release icx nopause` — ビルド識別ゲートで停止（下記）
3. `build.bat Release icx clean nopause`（テストを含む全ターゲット）— 258/731 で中断（テスト不要の指示）
4. `build.bat Release icx clean app nopause` — 成功（§5）

ゲートの出力:

```text
[GATE-FAIL] build identity mismatch (stamp vs current environment):
           compiler_path: stamp='C:/Program Files (x86)/Intel/oneAPI/compiler/latest/bin/icx.exe' now='icx'
           source_revision: stamp='32de0a50' now='c94d3767+dirty'
[GATE-FAIL] COHERENCE-4 violated.
[GATE-FAIL] recovery: run: build.bat Release clean   (explicit clean recovery)
[GATE-FAIL] refusing to continue (fail-closed; no automatic clean).
```

- stamp は別リビジョン・別コンパイラ指定で書かれた古い識別情報で、`build.bat` が `-DCMAKE_CXX_COMPILER=icx` を渡すため `compiler_path` の表現も変わっていた
- ゲートは fail-closed で自動 clean を行わない。`clean` で `build-icx` を作り直した（MSVC 用の `build/` は対象外）
- `+dirty` は未追跡の `doc/` 報告書によるもので、追跡ソースの変更ではない
- 作り直した構成では `[GATE-OK] build identity stamp written (first configure)` で通過した

## §5 ビルド結果

| 項目 | 値 |
|---|---|
| 実行 | `build.bat Release icx clean app nopause` |
| 結果 | 終了コード 0、`[SUCCESS] Executable created successfully.` |
| 所要 | 18:14:39〜18:22:27（約 7 分 48 秒） |
| 生成物 | `build-icx\ConvoPeq_artefacts\Release\ConvoPeq.exe`（46.19 MB、2026-11-09 18:22:27） |
| バージョン | FileVersion / ProductVersion 0.6.12 |
| 構成 | Release、`CONVOPEQ_CORRECT_POLYPHASE_GAIN=ON`、`CONVOPEQ_ENABLE_LOAD_CHAIN_DIAGNOSTICS=OFF`、PGO なし |
| コンパイラ | `C:/Program Files (x86)/Intel/oneAPI/compiler/latest/bin/icx.exe`（C / C++ とも） |
| コンパイル数 | `ninja -f build-Release.ninja -t commands ConvoPeq` で 138 コマンド中 135 が icx.exe |
| 診断 | エラー 0、警告 83 行（うち 71 は `ISRRetireRouter.h` の `worldReclaimCount` に `override` が無い警告。他は `AudioEngine.Mmcss.cpp` の未使用変数など） |
| テスト | `app` 指定のためテストターゲットはビルドしていない（`*Tests.exe` は生成されていない） |
| ビルドゲート | `[GATE-OK] build identity stamp written (first configure)`、dependency gate は 0 件 / 0 件 |

- テストターゲットを含む全ビルドは 258/731 で中断し、`app` 指定でやり直した（テスト不要の指示による）
- 生成した exe は起動していない（GUI アプリのため）

## §6 検証

- 生成物の存在・サイズ・更新時刻・バージョン情報を確認した
- `ninja -t commands` で Release のコンパイルとリンクのコマンド列を確認し、C/C++ のコンパイルが icx.exe であることを確かめた
- `build-icx/CMakeCache.txt` で `CONVOPEQ_CORRECT_POLYPHASE_GAIN=ON`、コンパイラ、構成を確認した
- テストの実行とアプリの起動は行っていない
