# Token Reduction — 常時運用ルール（2026-07-29 確定）

## ★ 絶対遵守 — 全セッションで3層パイプラインを常時使用

以降の全セッションにおいて、**headroom MCP + context-mode MCP + RTK (WSL版) の3層パイプラインを常時使用すること**がユーザーから明示的に指示された。

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
