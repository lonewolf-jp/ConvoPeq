# Token Reduction — 常時運用ルール（2026-07-29 確定・2026-08-10・08-14・08-15 再確認）

## ★ 絶対遵守 — 全セッションで3層パイプラインを常時使用

以降の全セッションにおいて、**headroom MCP + context-mode MCP + RTK (WSL版) の3層パイプラインを常時使用すること**がユーザーから明示的に指示された。

**【2026-08-15 再確認】「今後は常時かつ必ず」使用する。AI作業に支障のない範囲で適正なトークン削減を行うため、3つの役割分担を適宜最適化する。headroom MCP/CLI がうまく動作しない場合は context-mode MCP を優先して使用する。**

## 必須3系統

| 系統 | 役割 | 自動/手動 |
| --- | --- | --- |
| **Headroom MCP** | コンテキスト内の大きなコンテンツを圧縮（60-95%削減） | 手動で能動的利用 |
| **Context-Mode MCP** | ファイル分析・並列実行・検索（Read/Grep代替、93-99%削減） | 手動で能動的利用（最優先） |
| **RTK (WSL版)** | CLIコマンド出力を60-90%圧縮 | 手動でprefix付与（常時） |

## 役割分担（最適配分）

| 作業内容 | 使用ツール | 備考 |
| --- | --- | --- |
| ファイル分析/集計/抽出 | **ctx_execute / ctx_execute_file** | 生データをコンテキストに入れない |
| 複数コマンド並列実行 | **ctx_batch_execute** | 1回の呼び出しで最大8並列 |
| 過去内容の検索 | **ctx_search** | セッションメモリ＋インデックス化済みデータ |
| Web取得 | **ctx_fetch_and_index → ctx_search** | 生HTMLはコンテキストに入れない |
| 大きなコンテキスト保存 | **headroom_compress** | 復元は headroom_retrieve(hash) |
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

- **headroom MCP が動作しない場合 → context-mode MCP を優先して使用する（無理にheadroomを使わない）**
- proxy起動失敗: ANTHROPIC_BASE_URL未設定 → 直接API (支障なし)
- proxy異常終了: プラグインが自動再起動 + 30秒モニター
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

## Serena MCP Server (v1.7.0)

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
