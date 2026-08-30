# Token Reduction — 常時運用ルール（2026-07-29 確定・2026-08-10・08-14・08-15 再確認・2026-08-23 再確認・2026-08-27 commandcode環境固定・2026-08-28 ZCode環境でも適用確認）

## ★ 絶対遵守 — 常時かつ必ず3層パイプラインを使用（commandcode環境・ZCode環境ともに）

commandcode環境およびZCode環境では、**headroom (proxy + MCP server + CLI) + context-mode MCP + RTK (WSL版) の3層パイプラインを「常時かつ必ず」使用すること**がユーザーから明示的に指示された（2026-08-28 ZCode環境で再指示・2026-08-30 proxy 常時起動を設定）。**Visual Studio Code 環境のみ例外として proxy は使用しない（下記参照）**。

**【2026-08-27 確定・2026-08-28 更新】トークン削減性能を AI との動作に支障のない範囲で最適化するため、3 系統の役割を適宜最適な形で分担する。**
- **headroom proxy は常時起動（zcode 起動前に稼働）**。ユーザーのスタートアップフォルダに `start-headroom-proxy.bat`（＋最小化 `.lnk`）を配置し、ログオン時に `headroom proxy` を自動起動（port 8787、loopback-only）。クライアントはユーザー環境変数 `ANTHROPIC_BASE_URL=http://127.0.0.1:8787` で全トラフィックを proxy 経由にルーティング（`headroom doctor` で疎通確認可）。`headroom install apply --preset persistent-task` は schtasks 管理者権限が必要なため Startup フォルダ方式を採用。
- **Visual Studio Code 環境は例外: headroom proxy は使用しない**。VS Code 環境では `headroom mcp serve`（MCP server）のみ起動し、`ANTHROPIC_BASE_URL` は未設定（proxy 経由なし・自動圧縮なし）。上記の proxy 常時起動・全トラフィック自動圧縮は ZCode 環境（および commandcode 環境）向け。
- **【v0.37.0 仕様】CLI に `compress` / `retrieve` コマンドは存在しない**。proxy が全トラフィックを自動圧縮（CCR: hash マーカー付き、原文はローカル保存され `headroom_retrieve` で復元可能）。手動の圧縮・復元は MCP ツール `mcp__headroom__headroom_compress` / `mcp__headroom__headroom_retrieve`（＋`headroom_stats`）も利用可。
- proxy がうまく動作しない場合は context-mode MCP を優先して使用する。

## 必須3系統

| 系統 | 役割 | 自動/手動 |
| --- | --- | --- |
| **Headroom (proxy + MCP server + CLI)** | proxy が 8787 で常時起動し全トラフィックを自動圧縮（ANTHROPIC_BASE_URL 経由）。手動の圧縮/復元は MCP ツール `headroom_compress` / `headroom_retrieve`（CLIにcompress/retrieveコマンドは無し）。CLIは `savings` / `doctor` / `perf` / `memory` / `update` を使用 | proxy 自動 / MCP 手動 |
| **Context-Mode MCP** | ファイル分析・並列実行・検索（Read/Grep代替、93-99%削減） | 手動で能動的利用（最優先） |
| **RTK (WSL版)** | CLIコマンド出力を60-90%圧縮 | 手動でprefix付与（常時） |

## 役割分担（最適配分）

| 作業内容 | 使用ツール | 備考 |
| --- | --- | --- |
| ファイル分析/集計/抽出 | **ctx_execute / ctx_execute_file** | 生データをコンテキストに入れない |
| 複数コマンド並列実行 | **ctx_batch_execute** | 1回の呼び出しで最大8並列 |
| 過去内容の検索 | **ctx_search** | セッションメモリ＋インデックス化済みデータ |
| Web取得 | **ctx_fetch_and_index → ctx_search** | 生HTMLはコンテキストに入れない |
| 大きなコンテキスト保存 | **headroom_compress** (MCP) | 復元は **headroom_retrieve** (MCP、hash指定) |
| CLIコマンド | **rtk (WSL版)** | `wsl bash -c '...rtk <cmd>'` |
| ファイル編集 | **Read + Edit** | 編集時のみ通常ツールを使用 |
| コード検索 | AiDex > serena > semble > ctx_execute > WSL CLI | 優先順位順 |

## Context-Mode 活用ルール（能動的）

- ファイル分析/集計/抽出 → **ctx_execute** / **ctx_execute_file**
- 複数コマンド並列 → **ctx_batch_execute**
- 過去内容の検索 → **ctx_search**
- ファイル編集時のみ Read + Edit
- コード検索: AiDex ＞ serena ＞ semble/cocoindex ＞ ctx_execute ＞ WSL CLI(rg/ast-grep/fd/ag)

## フォールバック

- **headroom (proxy/MCP/CLI) が動作しない場合 → context-mode MCP を優先して使用する（無理にheadroomを使わない）**
- proxy 未起動時: スタートアップランチャーがログオンで自動起動（手動なら `headroom proxy`）。`ANTHROPIC_BASE_URL` はユーザー環境変数で永続設定済み。
- headroom MCP サーバ異常終了: ZCode/プラグインが自動再起動（30秒モニター）。
- RTK非対応コマンド: 素通し (rewrite不能時)

## コード検索ツール

| 層 | ツール | 呼び出し方 | 用途 |
| --- | --- | --- | --- |
| MCP#1 | AiDex | `aidex_query/signature/search` | 識別子検索、シグネチャ、セマンティック |
| MCP#2 | serena | `find_symbol/get_symbols_overview` | シンボル探索、参照追跡、宣言特定 |
| MCP#3 | semble | `search/find-related` | 自然言語クエリ検索 (99%削減) |
| CLI#1 | cocoindex | `ccc search/grep/status` | セマンティック検索、AST構造検索 |
| CLI#2 | graphify | `graphify query/path/explain` | ナレッジグラフ探索 |
| WSL | rg/ast-grep/fd(10.3)/ag(2.2)/fzf | `wsl bash -lc "..."` | 最終手段のテキスト/構造検索 |

## WSL統合

- `wsl bash -lc "..."` の内側コマンドは rtk-wsl プラグインが自動rewrite
- bashエイリアス: ls/grep/cat/find/diff → rtk版、fd→fdfind(10.3)

### ctx_batch_execute は MSYS2 環境（WSL 実行可能）— 2026-08-14 確認

ctx_batch_execute のサンドボックスは MSYS2（Git Bash 系）であり、**WSL を呼び出せる**。

- Windows exe は `/c/...` パス形式で直接実行可（`C:\...` バックスラッシュはエスケープされ失敗）
- `wsl.exe bash -c '...'` で WSL 実行可（RTK も使用可）

```bash
# ✅ ctx_batch_execute 内で WSL + RTK
wsl.exe bash -c 'cd /mnt/c/VSC_Project/ConvoPeq && ~/.local/bin/rtk git status'
# ✅ Windows exe（MSYS パス形式）
/c/Users/user/AppData/Roaming/Python/Python314/Scripts/headroom.exe --version
```

## Serena MCP Server (v1.7.0・2026-08-28 動作確認済み)

> 注意: MCPハンドシェイクの serverInfo.version（例: "1.28.1"）は serena 本体ではなく内部の `mcp` Pythonライブラリのバージョン。serena 本体のバージョンは `serena --version` で確認する（uv版 1.7.0 / pip版 Python314 に 1.6.2.dev0 も残留）。

Serena MCP server は OpenCode に接続されています (`--context ide` 使用中)。

### 重要: セッション開始時
- **CRITICAL**: コーディングタスク開始前に `initial_instructions` ツールを呼んで Serena Instructions Manual を読む
- Serena はセッション開始時に connection_prompt を MCP 経由で送信するが、OpenCode が表示しない場合がある

### ツール命名 (OpenCode)
- OpenCodeではMCPツールは `mcp__serena__<tool_name>` の形式で命名される
- 例: `find_symbol` → `mcp__serena__find_symbol`
- `serena_find_symbol` は無効な名前なので使用できない

### 利用可能なツール (32 tools with ide+editing+interactive)
- **シンボル操作**: find_symbol, find_referencing_symbols, find_declaration, find_implementations, get_symbols_overview
- **診断**: get_diagnostics_for_file, get_diagnostics_for_symbol
- **編集**: replace_symbol_body, insert_after_symbol, insert_before_symbol, replace_content, replace_in_files, delete_lines, replace_lines, insert_at_line
- **メモリ**: write_memory, read_memory, list_memories, delete_memory, rename_memory, edit_memory
- **その他**: search_for_pattern, restart_language_server, onboarding, serena_info, open_dashboard, remove_project, list_queryable_projects, query_project

### 注意: onboarding モードは非使用
- 以前の設定で `--add-mode onboarding` を使用していたが、このモードは編集ツールをすべて除外する
- `replace_symbol_body`, `insert_after_symbol`, `insert_before_symbol`, `delete_lines`, `replace_lines`, `insert_at_line` が無効になる
- プロジェクトは既に onboard 済みなので、`onboarding` モードは不要

### ツール使用ルール
- `--context ide` により、OpenCodeの組込み file/shell/search ツールが優先される (serena の create_text_file/read_file/execute_shell_command/find_file/list_dir は無効)
- シンボル操作は serena のツールを使用 (AiDex → serena → semble の優先順位)
- 行番号は0ベース (serena の行番号と異なる)
- ファイルパスはプロジェクトルートからの相対パス
- `ide` context + `single_project: true` により `activate_project` と `get_current_config` は無効

### 設定ファイル
- **opencode.json**: `"command": ["C:/Users/user/.local/bin/serena.exe", "start-mcp-server", "--context", "ide", "--project", "C:/VSC_Project/ConvoPeq", "--mode", "editing", "--add-mode", "interactive"]`
- **serena_config.yml**: `C:\Users\user\.serena\serena_config.yml` (グローバル設定)
- **project.yml**: `C:\VSC_Project\ConvoPeq\.serena\project.yml` (language_servers: [cpp, python, bash])
