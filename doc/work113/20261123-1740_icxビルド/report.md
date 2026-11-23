# main ブランチを icx でリリースビルドする（ConvoPeq.exe）

作業日: 2026-11-23（JST）。対象: `origin/main` = `db3e7a3a`。作業場所: 独立 worktree `C:\VSC_Project\fix236`（保護対象 `C:\VSC_Project\ConvoPeq` は未変更）。

## 1. 結果

`build.bat Release clean icx app nopause` でビルドし、**`ConvoPeq.exe` を生成した**。テスト用の実行ファイルは作っていない。`app` を指定すると、テストターゲットはビルドされない。

| 項目 | 値 |
| --- | --- |
| 実行ファイル | `build-icx\ConvoPeq_artefacts\Release\ConvoPeq.exe` |
| サイズ | 48,721,920 B |
| 生成時刻 | 2026-11-23 17:37:01 |
| コンパイラ | Intel oneAPI `icx` 2026.2（IntelLLVM 2026.2.0、MSVC 互換コマンドライン） |
| ツールチェーン | Visual Studio 18.12.1（MSVC 14.52.37025）、Windows SDK 10.0.26100.0、CMake 4.4.5、Ninja 1.13.2 |
| ビルド構成 | Ninja Multi-Config、`Release`、PGO なし（`CONVOPEQ_PGO_*` は両方 OFF） |
| ビルド識別スタンプの `source_revision` | `db3e7a3a`（main と一致） |
| 所要 | 約 8 分（構成 18.4 秒 ＋ 139 ステップ） |
| 警告・エラー | コンパイル警告 0 行、エラー 0 件 |

## 2. 手順

1. worktree `fix236` を `origin/main`（`db3e7a3a`）へ同期した。作業ツリーはクリーン。
2. PATH から `Octave` / `mingw64` / `msys` / `MinGW` を含む要素を外した。残すと、CMake が `ld.exe` や `ar.exe` を検出して構成に失敗する。既知の罠である。
3. `build.bat Release clean icx app nopause` を実行した。icx モードは `vcvarsall.bat` を呼ばず、`setvars.bat` で oneAPI を初期化する。
4. 生成物とスタンプを確認し、短い起動確認を行った。

## 3. 途中で止まった点と対処

`clean` を付けずに 1 回目を実行したところ、**ビルド識別ゲートで停止**した。

```
[GATE-FAIL] build identity mismatch (stamp vs current environment):
           compiler_path: stamp='C:/Program Files (x86)/Intel/oneAPI/compiler/latest/bin/icx.exe' now='icx'
           source_revision: stamp='84615903+dirty' now='db3e7a3a'
[GATE-FAIL] recovery: run: build.bat Release clean   (explicit clean recovery)
```

`build-icx` に残っていたのは 2026-11-22 のスタンプで、当時のソースは `84615903+dirty` だった。今回の環境ともソースとも一致しない。仕様どおり fail-closed で止まったのである。表示された指示どおり `clean` を付けて再実行し、通過した。

`clean` により juceaide のサブビルドも作り直された。構成だけで 18.4 秒かかった。全体では約 8 分である。以前の記録（8.4 分）と同じ水準だった。

## 4. 検証

- **成果物**: `[138/139] Linking CXX executable ConvoPeq_artefacts\Release\ConvoPeq.exe` → `[SUCCESS] Executable created successfully.` → `Build Complete!`。`build-icx\ConvoPeq_artefacts\Release\ConvoPeq.exe` が 48,721,920 B で存在する。
- **テスト実行ファイルが無いこと**: `build-icx` 配下の `.exe` は 6 個で、内訳は `ConvoPeq.exe`、`juceaide.exe`（JUCE のコード生成ツール）、CMake のコンパイラ判定用 `CMakeCXXCompilerId.exe` / `CMakeCCompilerId.exe`（本体・JUCE の 2 セット）だけである。`*Tests.exe` は生成されていない。
- **ビルド識別**: `build-icx\CMakeFiles\.build_identity` の `compiler_path` は icx、`source_revision` は `db3e7a3a` で、main と一致する。
- **依存 DLL**: `dumpbin /dependents` の結果は Windows の DLL（`KERNEL32` / `USER32` / `d3d11` / `AVRT` など）だけで、**Intel ランタイム（`libiohlp.dll` など）や MKL の DLL を要求しない**。oneAPI を PATH に通していない通常のシェルから起動できた。
- **起動確認**: `ConvoPeq.exe --cli-run --cli-exit-ms 1500` が終了コード 0、所要 3.7 秒で完了した（ウィンドウは開かない）。

## 5. 制約・注意

- 所要時間とサイズは、この端末での実測値である。icx の成果物は MSVC のものとサイズが違う。MSVC 構成の記録は 48,817,152 B だった。
- `clean` を伴うため、`build-icx` のキャッシュは今回のスタンプで作り直されている。次回、ソースか環境が変わると再びゲートで止まるので、そのときも `clean` を付ける。
- テストは実行していない（テスト用実行ファイルを作らない指示のため）。起動確認は CLI の短時間実行のみである。

## 6. 参照

- `build.bat`（`app` 指定で ConvoPeq.exe のみをビルド）
- `src/tools/build_identity_gate.py`（ビルド識別ゲート）
- ビルドログ: `%TEMP%\opencode\build_icx_main.log`
