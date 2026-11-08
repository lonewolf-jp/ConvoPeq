# プロジェクトバージョンの 0.6.12 への更新

- 実施: 2026-11-08 19:15〜
- 対象: `main` = `6a255fb8`
- 目的: ConvoPeq のバージョンを 0.6.11 から 0.6.12 に更新する。

## 1. 変更内容

バージョンの一元管理は `ProjectMetadata.cmake` の `APP_VERSION_{MAJOR,MINOR,PATCH}` で、`PROJECT_VERSION` を経由して CMake の `project(VERSION ...)`、`JUCE_APPLICATION_VERSION_STRING`、`MainApplication::getApplicationVersion()`、実行ファイルのバージョン情報まで反映される。今回はパッチ番号のみを 11 から 12 に変更した。

| ファイル | 修正 |
|---|---|
| `ProjectMetadata.cmake` | `set(APP_VERSION_PATCH 11)` → `12` |
| `CMakeLists.txt` | 冒頭バナー `# CMakeLists.txt ── v0.6.11 ...` → `v0.6.12` |
| `README.md` | 概要の `ConvoPeq v0.6.11` → `v0.6.12`、ディレクトリマップの `CMakeLists.txt`（2 箇所目）、`ProjectMetadata.cmake`、`ARCHITECTURE.md` の版表記 |
| `ARCHITECTURE.md` | タイトル、§14 の要約、末尾の `*Version: v0.6.11*` → `v0.6.12` |
| `SOUND_PROCESSING.md` | 冒頭の説明文、`**Project**` 行 |
| `BUILD_GUIDE_WINDOWS.md` | 冒頭の構成説明、`**Project**` 行、レイアウト内の `CMakeLists.txt` と `ProjectMetadata.cmake` |

README の `## New in v0.6.12` 見出し（別依頼で更新済み）と整合する。

## 2. 検証

### ローカルビルド

`build.bat Release app clean nopause`（MSVC、vcvars64 環境）を実行した。

| 項目 | 結果 |
|---|---|
| exit code | 0 |
| ビルドターゲット | アプリ限定（テストターゲットなし） |
| 成果物 | `build\ConvoPeq_artefacts\Release\ConvoPeq.exe`（47,606,784 B） |
| 所要 | 約 2.9 分（19:17:56 → 19:20:49、clean から） |
| FileVersion | **0.6.12** |
| ProductVersion | **0.6.12** |
| build identity gate | `[GATE-OK]`（clean で stamp を再作成） |
| 成果物内の旧版残存 | `0.6.11` は 0 件 |

### 参照の残存

- 追跡ファイルの `0.6.11` は `doc/`（過去の作業報告・監査資料）と `ConvoPeq.md`（生成スナップショット）のみ。
- `config/*.json` の `schema_version` / `manifest_version` はスキーマ番号であり、アプリ版とは無関係。

### ConvoPeq.md について

`ConvoPeq.md` は `output_sourcecode_markdown.py` が生成するソース抜粋（2026-11-08 10:08 生成、6.5 MB）で、`CMakeLists.txt` の内容も含む。ただし変更前の時点で `--check` が `STALE`（`build.bat` の更新を検出）と判定しており、継続的に同期される成果物ではない。今回はバージョン変更と無関係な 6.5 MB の差分を避けるため再生成していない。

## 3. CI

コード（`ProjectMetadata.cmake`）を変更するため、必須チェック `Windows MSVC / Release` は通常どおり実行される（文書のみのスキップ対象外）。

### 無関係な一時失敗（Ninja のダウンロード）

`ISR Authority Compliance v6.4` の `authority-runtime-tests` ジョブが `Set up Ninja` で失敗した。

```
Recieved 504 from https://github.com/ninja-build/ninja/releases/download/v1.10.0/ninja-win.zip
```

- 失敗したのは Ninja を GitHub Releases から取得するステップで、`Configure CMake` 以降は実行されていない。
- 2 回再実行したが同じ 504 が続いた。GitHub 側の配信不調であり、バージョン変更とは無関係。
- 必須チェックは `Windows MSVC / Release` のみで、このジョブは必須ではない。
- 恒久対策の案: `windows-latest` には Ninja が同梱されているため、ダウンロードに失敗したら同梱の `ninja` を使うフォールバックをワークフローに追加する（今回は対象外）。

## 4. 変更ファイル

| 種別 | ファイル |
|---|---|
| ビルド | `ProjectMetadata.cmake`、`CMakeLists.txt`（バナー） |
| 文書 | `README.md`、`ARCHITECTURE.md`、`SOUND_PROCESSING.md`、`BUILD_GUIDE_WINDOWS.md` |
| 報告 | 本ファイル |
