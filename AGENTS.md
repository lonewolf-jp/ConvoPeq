# Token Reduction — 常時運用ルール（2026-07-29 確定・2026-08-10・08-14・08-15 再確認・2026-08-23 再確認・2026-08-27 commandcode環境固定・2026-08-28 ZCode環境でも適用確認・2026-09-20 Xiaomi MIMO Desktop でも再確認・**2026-09-21 Qoder デスクトップ環境で検証（proxy 適用外 → 末尾「Qoder デスクトップ環境」節参照）**・**2026-09-21 Freebuff Desktop 環境で検証（MCP 14 サーバー＋ Agent Skills 131 件・proxy 適用外 → 末尾「Freebuff Desktop 環境」節参照）**）

## ★ 絶対遵守 — 常時かつ必ず3層パイプラインを使用（commandcode環境・ZCode環境・Xiaomi MIMO Desktop 環境ともに）

commandcode環境、ZCode環境、および **Xiaomi MIMO Desktop 環境**では、**headroom (proxy + MCP server + CLI) + context-mode MCP + RTK (WSL版) の3層パイプラインを「常時かつ必ず」使用すること**がユーザーから明示的に指示された（2026-08-28 ZCode環境で再指示・2026-08-30 proxy 常時起動を設定・**2026-09-20 MIMO Desktop で再確認**）。**Visual Studio Code 環境のみ例外として proxy は使用しない（下記参照）**。

**【2026-09-20 MIMO Desktop 確認】** headroom v0.37.0（proxy 8787 稼働・MCP 読込済・`ANTHROPIC_BASE_URL` 経由）、context-mode v1.0.169（doctor PASS）、rtk（WSL `/home/user/.local/bin/rtk`）すべて使用可。役割分担は「ソースで防ぐ（context-mode）→ トラフィック自動圧縮（headroom proxy）→ CLI出所圧縮（rtk）」を原則とし、ダブル圧縮を避ける。詳細は serena memory `tools/token-reduction-pipeline-2026-09-20`。

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

### ctx_batch_execute / ctx_execute は **PowerShell**（MSYS2 ではない）— 2026-09-26 实測訂正

旧記録「サンドボックスは MSYS2（Git Bash 系）」は**この環境では誤り**。実測でサンドボックスが
`C:\Users\user\AppData\Local\Temp\.ctx-mode-XXXX\script.ps1` を生成し、**PowerShell で実行**していた。

- **`/c/...` の MSYS パスは失敗する**（`用語 '/c/Program\' は ... として認識されません`）。
- Windows exe は **call 演算子 `&` + シングルクォート**で書く。
- パイプ / リダイレクト / `Select-Object` は PowerShell 記法 그대로使える。
- `wsl.exe bash -lc '...'` は**単一引用符ごと**渡す（PowerShell がクォートを消費する）。
- 複数行テキストは PowerShell のバッククォート改行 `` `n `` で渡す。
- PowerShell のパス区切りは `\` なので、bash 向けのリダイレクトを直接書くと壊れる。

```powershell
# ✅ ctx_execute / ctx_batch_execute 内で Windows exe
& 'C:\Users\user\AppData\Roaming\Python\Python314\Scripts\headroom.exe' --version
# ✅ WSL + rtk（引用符ごと渡す）
wsl.exe bash -lc 'cd /mnt/c/VSC_Project/ConvoPeq && ~/.local/bin/rtk git status'
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

## Qoder デスクトップ環境（2026-09-21 検証済み）

### 3層パイプラインの適用範囲
Qoder はモデル通信を `ANTHROPIC_BASE_URL` 経由にしないため、**headroom proxy は Qoder のトラフィックを一切圧縮しない**（`headroom savings` Today = `0 / 0 tokens`）。したがって上記「常時必ず proxy」は Qoder では成立せず、実効構成は **context-mode MCP（主）→ rtk (WSL)（CLI 圧縮）→ headroom MCP ツール（手動 compress/retrieve）** の3本。proxy を使えたかのように報告してはいけない。

### MCP サーバー登録と可視化の仕組み
- 登録先は `C:\Users\user\.qoder\settings.json` の `mcpServers`（stdio: `type/command/args/cwd/env`、http: `url/headers`）。保存で自動リロードされ、sqlite `mcp_connection_profiles`（`source_kind='custom'`）に `revision` として反映。**アプリ再起動は不要**。
- エージェントから見えるのは `status=ready`（＝ `protocol_era` が確定）したサーバーだけで、しかも**新しく起動したラン/セッションにのみ**注入される。セッション中は起動時のスナップショットを維持するので、セッション途中で直したサーバーはそのセッションに出ない。
- 接続失敗サーバーの再試行は settings.json を変更して revision を上げる（またはアプリ再起動）。
- 状態確認（読み取り専用）:
  ```bash
  C:/Python314/python.exe -c "import sqlite3;c=sqlite3.connect(r'file:C:/Users/user/AppData/Roaming/com.qoder.app.stable/main.sqlite?mode=ro&immutable=1',uri=True);[print(*r) for r in c.execute(\"select runtime_name,revision,protocol_era from mcp_connection_profiles where source_kind='custom'\")]"
  ```
  ログ: `%APPDATA%\com.qoder.app.stable\logs\<run-dir>\main.log` と `mcp\<serverId>.log`（cold→queued→starting→ready/failed と `failureCode`）、セッションの argv は `%USERPROFILE%\.qoder\logs\runs\<ts>-p<pid>\manifest.json`。
- **リポジトリの `.mcp.json` は読まれるが接続しない**（`--allowed-mcp-server-names` に入らないため）。使わないこと。

### 接続済み（12）/ 意図的に無効（1）
`serena`(32) `aidex`(31) `obscura`(37) `firecrawl`(27) `context_mode`(11) `brave_search`(8) `ddgs`(6) `graphify`(10) `ms_learn`(3) `headroom`(3) `context7`(2) `cocoindex`(1=search)。`semble` は MCP 未登録で CLI のみ。
未接続: `github` — `${GH_TOKEN}` が展開されず 400 になるため `"disabled": true`。GitHub は `C:\Program Files\GitHub CLI\gh.exe` を使う。

### 設定ではまった点（同じ症状が出たらまずこれ）
1. stdio 子プロセスの env はホワイトリスト制で **`APPDATA` を含まない** → `pip install --user` 済みの Python パッケージが `ModuleNotFoundError`（headroom/graphify がこれで落ちた）。対策はサーバー単位で `"env": {"APPDATA": "C:\\Users\\user\\AppData\\Roaming", "PYTHONPATH": "C:\\Users\\user\\AppData\\Roaming\\Python\\Python314\\site-packages"}`。
2. 環境変数 `${VAR}` の展開は非対応（`${QODER_PLUGIN_ROOT}` / `${QODER_PLUGIN_DATA}` / `${user_config.x}` のみ）。
3. **API キーを settings.json に書かない** — 全ランの `manifest.json` argv に平文で残る。方式は `~/.qoder/keys/brave.key`（`env` で参照）と `~/.qoder/keys/firecrawl/.env`（`cwd` をそのディレクトリにして dotenv で読ませる）。両者 `chmod 600` + `icacls /inheritance:r /grant:r <user>:F`。
4. `ccc.exe` は `cwd` 依存（リポジトルートを `cwd` に指定しないと "Not in an initialized project directory"）。
5. 共有 site への `pip install --user` は `mcp` を 2.x に吊り上げて headroom を壊す（`github-mcp-server` も `mcp<2` 必須）。**`mcp<2` を固定**し、Python 系 MCP は `uvx --from '<pkg>[extra]' <entry>` で隔離（ddgs はこれで接続）。
6. obscura は `~/.local/bin/obscura/obscura.exe` が headless browser。`~/.local/bin/obscura.exe` は無関係な argon2 復号ツール。
7. **Dr. Memory はこのマシンで計測不能**（"invasive security software" 干渉・rc=127）。我側の設定では直らない。
8. PATH に無いのでフルパスで呼ぶ: `C:\Program Files\Cppcheck\cppcheck.exe`（2.21.0）、`C:\Program Files\LLVM\bin\clang-tidy.exe`（LLVM 23.1.1）。**Maxima はインストール済み・PATH 未登録だが `maxima` シムあり**（`~/.local/bin/maxima.cmd` と `~/.local/bin/maxima`、実体 `C:\maxima-5.50.0\bin\maxima.bat` = 5.50.0。Octave 11.3.0 は PATH 上、numpy 2.5.2 / scipy 1.18.0 / matplotlib 3.11.1 / pandas 3.0.5 / IPython 9.17.1 / crawl4ai / trafilatura 2.2.0 は使用可）。

### コード検索インデックス状態（2026-09-21 時点）
`ccc` は `.cocoindex_code/`（305,797 chunks / 2,595 files・当日更新）、AiDex は `.aidex/`（当日更新）で新鮮。**tgrep / graphify / AiDex は古い索引を無警告で返す** — 当てる前に必ず鮮度確認する（2026-09-21 時点では tgrep 2,859 files・2h 前、graphify 74,421 nodes・12:58 更新、AiDex `.aidex/index.db` 487 files / 20,564 items・11:26 更新でいずれも新鮮）（graphify の生成は `/graphify` スキル側、`graphify` CLI に build サブコマンドはない）。**tgrep のインデックスは常に `.tgtep/`（プロジェクトルート直下）を使用する**（`--index-path .tgtep`）。`src/.tgrep/` は使わない。

## Freebuff Desktop 環境（2026-09-21 検証済み）

### 3層パイプラインの適用範囲
Freebuff Desktop は推論を `https://www.codebuff.com`（アカウントトークン方式、モデル `deepseek/deepseek-v4-flash`）へ送るため `ANTHROPIC_BASE_URL` を参照しない。**headroom proxy は Freebuff のトラフィックを一切圧縮しない**（proxy 自体は 8787 で稼働中だが `headroom savings` Today = `0 / 0 tokens`）。実効構成は **context-mode MCP（主）→ rtk (WSL)（CLI 圧縮）→ headroom MCP ツール（手動 compress/retrieve）** の3本。proxy が効いたように報告してはいけない。

### MCP サーバー登録と可視化の仕組み
- 登録先は **`C:\Users\user\.agents\mcp.json`**（`mcpServers`）。Codebuff SDK は `{cwd}/.agents/mcp.json` → `{cwd}/../.agents/mcp.json` → `~/.agents/mcp.json` を後勝ちでマージするが、デスクトップのエージェントは homedir の 1 ファイルを使う。**リポジトリの `.mcp.json` や `~/.qoder/settings.json` は読まれない**。
- スキーマ: stdio `{type?: "stdio", command, args[], env{}, cwd?}` / remote `{type: "http"|"sse", url, headers{}}`。**sse は未対応**、url は `https:` か loopback のみ。ツール名は `mcp__<server>__<tool>`。
- 子プロセスの env は **12 変数のホワイトリスト**（APPDATA, HOMEDRIVE, HOMEPATH, LOCALAPPDATA, PATH, PROCESSOR_ARCHITECTURE, SYSTEMDRIVE, SYSTEMROOT, TEMP, USERNAME, USERPROFILE, PROGRAMFILES）。`$VAR` 展開は HOME/PATH/SHELL/USER/LANG/TMPDIR のみで、他は `Missing credential binding` としてそのサーバーが無効化される（`${VAR}` 記法は不可）。
- **API キーはレジストリ読み出しラッパーで渡す**: `C:\Users\user\.local\bin\mcp\{context7,firecrawl,brave-search,github}-mcp.cmd` が `reg query "HKCU\Environment"` で Windows ユーザー環境変数を取得（鍵を設定ファイルに書かない）。github は `GH_TOKEN` → `GITHUB_PERSONAL_ACCESS_TOKEN` に変換。
- **承認 (consent) は人間の操作が必要**: 初回利用時に Electron 本体の「Approve this connector?」ウィンドウが出る。承認状態はサイドカー `~/.freebuff/mcp.json` に `launchHash`/`manifestHash`/`allowedTools` として保存される。**mcp.json を書いた後は新規スレッドを開始**しないとツールが注入されない。同時接続は既定 8（超過分は LRU で切断、全 14 登録でも同時 8）。
- 登録済み 14: `serena`(32) `aidex`(31) `obscura`(37) `firecrawl`(27) `github`(31) `context-mode`(11) `graphify`(10) `brave-search`(8) `ddgs`(6) `headroom`(3) `microsoft-learn`(3=remote) `context7`(2) `semble`(2=uvx) `cocoindex`(1=search)。`cocoindex` / `graphify` / `serena` は `cwd` をリポジトリルートに固定（ccc は cwd 依存）。

### Agent Skills（2026-09-21 検証）
- 探索順（後勝ち）は `~/.claude/skills` → `~/.agents/skills` → `{cwd}/.claude/skills` → `{cwd}/.agents/skills`。**インストール先は `~/.agents/skills`**、エージェントが作るスキルは `~/.freebuff/skills`（グローバル）と `{cwd}/.agents/skills`（プロジェクト）。
- **1 スキル = 1 ディレクトリ + `<dir>/SKILL.md`**。探索は直下 1 階層のみ（ネストした SKILL.md は自動読込されない）。**不備があると無言でスキップ**されるため、追加後は preflight の `Agent Skills` セクションで検査する。
- 規則: ディレクトリ名・frontmatter `name` は `/^[a-z0-9]+(-[a-z0-9]+)*$/`（64 字以内）、`description` 必須（1024 字に切り詰め）。
- 導入済み（全 131 スキルが読み込み可能）。依頼の 5 件はいずれも使用可: `reverse-skill`（router；89 モジュールはネストのため router 経由で読む）、`debugging-code` + `dap` CLI（**`python3` を探すので `~/.local/bin/python3` シム必須**）、`ui-ux-pro-max` ファミリー 7 スキル、`pm-skills`（上流 clone は SKILL.md が無く読めなかったため router SKILL.md を追加作成）、`docmd`（上流にスキルは無し・`docmd` CLI 0.9.5 として使用）。
- 巨大ファミリー（68〜89 スキル）は**個別登録せず router 1 枚経由**にする（name/description は常時文脈に注入されるため）。

### 検証の自動化（再実行は 1 コマンド）
```bash
python scripts/freebuff_preflight.py         # フル（MCP 14 サーバーのハンドシェイクまで・約 90 秒）
python scripts/freebuff_preflight.py --quick # MCP プローブ・索引内容スキャン省略（約 30 秒）
python scripts/freebuff_preflight.py --fix   # 加えて古い索引を更新（ccc index / tgrep index --index-path .tgtep / graphify update .）
```
- 終了コードは必須失敗で 1。判定 `GO` / `GO with warnings` / `NO-GO`。ベースラインは `scripts/freebuff_expectations.json`（ツール数のずれは WARN、ハンドシェイク失敗は FAIL）。
- 単体プローブは `python scripts/freebuff_mcp_probe.py <spec.json> [server ...]`。**必ず Freebuff と同じ 12 変数のホワイトリスト env で起動して判定する**（手元のシェルで起動できても Freebuff で失敗するケースがある）。API キー未取得ならラッパーが `exit /b 1` で落ちるため鍵の配管検証も兼ねる。
- 実測 2026-09-21: フル 116 項目 / 113 pass・1 warn・0 fail / 67 秒（MCP 14 サーバー＋Agent Skills 131 件＋索引＋パイプライン＋メモリ検証を一括検査）。
- **Dr. Memory の代替（この環境のメモリ検証手段）**: clang ASan は `C:\Program Files\LLVM\lib\clang\23\lib\windows` の `clang_rt.asan_dynamic-x86_64.dll` を PATH に足せば動く（実測で heap-buffer-overflow 検出）。プロジェクトは `build.bat Debug msvc -DENABLE_ASAN=ON`、補助的に Application Verifier（`C:\Windows\System32\appverif.exe`）。
- WSL は単一引用符が外側シェルに食われるので **heredoc** を使う: `wsl bash -ls <<'EOS' … EOS`。

### はまった点
1. clang-tidy は `-p` に **ディレクトリ**を要求し、リポジトリ直下の `compile_commands.json` は JSONL 形式で読めない（`json-compilation-database: Expected array`）。配列形式の `compile_commands_clang.json` を `tmp/clangdb/compile_commands.json` にコピーして `clang-tidy -p tmp/clangdb` を使う。
2. cppcheck は JUCE マクロ未設定だと `unknownMacro` が支配的 → `--std=c++17 --suppress=missingIncludeSystem` と `-D` を併用する。
3. **Dr. Memory は実行不可**（`-batch -- <app>` で rc=127 / "interference from invasive security software"）。AGENTS.md 記載どおり環境側要因で、設定では直らない。
4. tgrep / graphify / AiDex は古い索引を無警告で返す。`tgrep status --index-path .tgtep`・`graphify-out/graph.json` の mtime・`aidex_status` で確認してから使う（更新は `tgrep index --index-path .tgtep` / `graphify update .` / `aidex_init`・`aidex_global_refresh`、または preflight の `--fix`）。

詳細・全ツールの実走結果は `doc/tool-inventory-2026-09-21-freebuff.md`。

## OpenChamber / OpenCode 環境（2026-09-26 検証済み・同日修正）

### 3層パイプラインの適用範囲 — headroom proxy は効かない
OpenChamber（OpenCode 2.0.16）はモデル通信を `ANTHROPIC_BASE_URL` 経由にしない。**headroom proxy は OpenChamber のトラフィックを一切圧縮しない**（proxy 自体は 8787 で稼働中だが `headroom savings` Today = `0 / 0 tokens`、`/stats` の `api_requests: 0`）。proxy が効いたように報告してはいけない。実効構成は **context-mode MCP（主）→ rtk（CLI 圧縮）→ headroom MCP ツール（手動 compress/retrieve）** の3本。

### サーバーログが唯一の信頼できる診断手段
`C:\Users\user\.local\share\opencode\log\opencode.log` （OpenCode サーバーログ。セッションのツール出力ではない）。MCP 接続は `message="mcp connected" server=<name> tools=<n>`、プラグイン失敗は `message="failed to load plugin" target=<name> cause=...` で記録される。設定が読めていない時は `message="configuration normalization diagnostic" path=$.mcp.servers.<name> kind=invalid action="skipped malformed recognized value"` が出る。
> `opencode mcp list`（`C:\Users\user\.bun\bin\opencode.exe`）は**この環境のサーバー一覧を一切返さない**（「No MCP servers configured」のまま）。信用しないこと。壊れた config キーが無視される理由もここには出ない。

### MCP 設定スキーマ — `cwd` / `enabled` / `timeout` は使えない（重要）
このビルドの `mcp.servers.<name>` で有効なキーは **`type` / `command` / `disabled` / `environment` / `env` / `url` / `headers` のみ**。
- `cwd`・`enabled`・`timeout` の **1 つでも入れると entry 全体が破棄される**（公式 opencode.ai ドキュメントには載っているが、この実装は未対応）。ログに `kind=invalid action="skipped malformed recognized value"`。
- 有効化は `enabled: true` ではなく **`"disabled": false`**。
- cwd 依存を回避するには、絶対パスを `command` に直接書く（例: graphify は `--graph <絶対パス>`）か、`environment` で環境変数を渡す。
- 検証済みの正しい形:
  ```json
  "graphify": {
    "type": "local",
    "command": [".../graphify-mcp.exe", "--transport", "stdio", "--graph", "C:/VSC_Project/ConvoPeq/graphify-out/graph.json"],
    "disabled": false
  }
  ```

### コンソールウィンドウの抑制 — nowin.py を全 local サーバーに適用（2026-09-26）
OpenChamber は Electron（GUI）アプリだが、MCP サーバーを `CREATE_NO_WINDOW` 無しで起動するため、console サブシステムの実行ファイルや `.cmd` シムを起動するたびにコンソールが点滅する（再接続・監視再起動で繰り返す）。
- **対策**: `C:\Users\user\.local\bin\mcp\nowin.py` をラッパーとして使う。`pythonw.exe`（GUI サブシステム）経由で起動し、子のプロセスを `creationflags=CREATE_NO_WINDOW` (0x08000000) で起こす。fd 0/1/2 をそのまま中継するので stdio MCP トランスポートは無傷。
- **設定例**:
  ```json
  "cocoindex": {
    "type": "local",
    "command": [
      "C:/Python314/pythonw.exe",
      "C:/Users/user/.local/bin/mcp/nowin.py",
      "C:/Users/user/.local/bin/ccc.exe", "mcp"
    ],
    "disabled": false
  }
  ```
- 実装上の留意点:
  1. **pythonw.exe では `sys.stdout` / `sys.stderr` が `None`**（コンソールが無いので）。`sys.stdout.buffer` は `AttributeError`。raw fd の `os.read` / `os.write` を使うこと。
  2. **`.cmd` / `.bat` は CreateProcess で直接実行できない**。ラッパ内で `cmd.exe /d /s /c "<line>"` に変換すること（`CREATE_NO_WINDOW` 付きなので cmd.exe も点滅しない）。`npx` には `npx.exe` が無く `npx.cmd` しか無いので、素の `npx` を渡すと失敗する → `C:\Users\user\AppData\Roaming\npm\npx.cmd` のフルパスに変換済み。
  3. `os.read(fd, n)` は 1 バイト以上来たら返るので対話的プロトコル向き。`read(n)`（厳密 n バイト待ちは）はデッドロックする。
  4. stderr を別スレッドで排出しないと、child が stderr を溜め込んでパイプが詰まって永久に停止する。
- **適用結果**: local サーバー **13 個すべて**をラップ（serena / headroom / aidex / firecrawl / brave / context-mode / obscura / ddgs / graphify / cocoindex / semble / github / mcp-windbg）。remote 2 個（context7 / mslearn）はプロセスを起動しないので無変更。
- **検証**: 13/13 がラッパー経由で initialize + tools/list 成功（serena は 3.4 秒で応答。ラッパー経由の stderr 中継も確認済み）。ロールバック用バックアップは各 config の `.bak-before-nowin`。
- 再適用は冪等（`pythonw.exe` で始まる command はスキップ）。変換スクリプトは永続化済み: `C:\Users\user\.local\bin\mcp\apply-nowin.py`（再実行しても `wrapped 0` になることを確認済み）。
- 高速な再検証: `python C:\Users\user\.local\bin\mcp\mcp-probe.py <spec.json>`（JSON-RPC ハンドシェイクを実際に打って tools/list を取る。spec は config から機械生成する）。

### OpenChamber のプラグイン契約 — 公式 V2 形式は**拒否**される
```
PluginModule.LoadError: Plugin must export a default definition with an id
and an effect or setup function. (cause: SchemaError(Missing key at ["default"]))
```
- **必須**: `export default { id, setup(ctx) {...} }`（または `{ id, effect }`）。
- **不可**: opencode.ai 公式ドキュメントにある名前付きエクスポート形式 `export const X: Plugin = async (ctx) => ({...})`。これは 100% 失敗する。
- したがって **`.opencode/plugins/rtk-wsl/index.ts` の `export default { id: "rtk-wsl", async setup(ctx) {...} }` は正しい契約**であり、V1 の遺物ではない（2026-09-26 訂正）。
- **Vendor 製 context-mode プラグイン（v1.0.169）もこのビルドではロードできない**（同じエラー）。`context-mode upgrade` が `"plugin": ["context-mode"]` を書いても無意味なので削除済み。したがって **context-mode の自動ルーティング（Read/Shell の誘導）は本環境では機能しない** — `ctx_*` ツールは MCP サーバーとして手動で使うこと。
- **rtk プラグインは導入しない**。自動 rewrite にはシェルコマンドごとに子プロセス（`wsl bash -lc`）の起動が必要で、**これがコンソールを毎回点滅させる**（2026-09-26 に一旦実装して撤去）。rtk はコマンドに `rtk <cmd>` と明示的に prefix を付ける運用で足りる。

### MCP サーバー構成（15）
- プロジェクト `C:\VSC_Project\ConvoPeq\opencode.json` に **12** — serena(32) headroom(3) aidex(31) context7(2) firecrawl(27) brave(8) mslearn(3) context-mode(11) obscura(37) ddgs(6) graphify(10) cocoindex(1)
- グローバル `C:\Users\user\.config\opencode\opencode.jsonc` に **3** — semble(2) github(31) mcp-windbg(10)
- 2026-09-26 追加分（インストール済みだが未登録だったもの）: context-mode / obscura / ddgs / graphify / cocoindex。
  - `context-mode` = `C:/Users/user/AppData/Roaming/npm/context-mode.cmd`（`environment` で `CONTEXT_MODE_DIR` を固定）
  - `obscura` = `C:/Users/user/.local/bin/obscura/obscura.exe mcp`（`~/.local/bin/obscura.exe` は無関係な argon2 復号ツール）
  - `ddgs` = `uvx --from "ddgs[mcp]" ddgs mcp` — **共有 site に入れてはいけない**（`mcp>=2.0` が headroom / github-mcp-server を壊す）
  - `graphify` = `graphify-mcp.exe --transport stdio --graph <絶対パス>`
  - `cocoindex` = `ccc.exe mcp`
- **再起動が必要**（config とラッパーの適用には OpenChamber の再起動が要る）。

- **GUI の「MCP サーバー」パネルはグローバル設定だけを見る（仕様・回退ではない）**:
  OpenChamber は `C:\Users\user\.config\opencode\opencode.jsonc` しか読み書きしない
  （`logs\main.log` に `Created config backup: ...opencode.jsonc.openchamber.backup` /
  `Successfully wrote config file: ...opencode.jsonc` のみが残り、プロジェクト設定には触れない）。
  したがってパネルにはグローバル側の 3 個（github / semble / mcp-windbg）しか出ない。これはオンデマンド起動ではない（実測で 15 台が起動時に一斉に接続）。
  一方、项目別 `ConvoPeq\opencode.json` の 12 個は **OpenCode サーバー側が global + project を
  マージして実際にロードする**ので正常に動作する（サーバーログの `mcp connected` 全 15 件で実証）。
  > 「パネルに 12 個出ない = 動いていない」ではない。判定は
  > `Select-String -Path ~\.local\share\opencode\log\opencode.log -Pattern "mcp connected"` で行う。
  > この分離は意図的に残す。**利点は『他のプロジェクト』側だけ**: ConvoPeq を開くときは global 3 + project 12 の 15 個が**すべて起動する**（実測: 起動から +2.1s〜+9.7s に 15 台が並列接続、その後の disconnect/reconnect は 0 件）。ConvoPeq 以外のプロジェクトを開くと、この 12 個は読み込まれない。

### rtk の破損 DB を修復済み
`~/.local/share/rtk/history.db` が `database disk image is malformed: Error code 11` で壊れて `rtk gain` が不能だった → 削除（rtk が再作成）。圧縮機能自体は無影響。
- **ネイティブ `rtk.exe` 0.49.0 が `C:\WINDOWS\system32\rtk.exe` にもある**（WSL 版と同一バージョン）。`rtk rewrite` の実測: `ls -la`→`rtk ls -la` / `cat X`→`rtk read X` / `ast-grep ...`→空（素通し）/ **`wsl bash -lc '...'`→空（ラッパーを解釈しない）**。

### コード検索インデックス（2026-09-26 時点、全て更新済み）
`.cocoindex_code/`（1.87 GB・97,676 chunks・**daemon 生存中で自己更新**）、`graphify-out/graph.json`（87.6 MB・55,419 nodes・**`graphify update .` で LLM 不要の code-only 更新**）、`.tgtep/`（59.8 MB・2,010 files・227,259 trigrams）、`.aidex/`（36.1 MB・484 files・20,787 items）。**4 つとも古い索引を無警告で返す**ので使う前に mtime を確認すること。

### 既存記録の訂正（実測で判明した相違点）
- **tgrep の既定 index path は `.tgrep/`**（`.tgtep/` ではない。リポジトリ固有の慣習）。`status`/`index`/`serve`/検索のたびに `--index-path .tgtep` を付けないと黙って見つからない。
- **tgrep は意味検索ではない**。trigram 索引付き正規表現 grep（ripgrep 互換・埋め込みなし・LLM なし）。
- **clang-tidy に multi-config DB を渡すと 1 ファイルを config 回数だけ解析する**（Ninja Multi-Config では 4 回・27 秒）。高速化には 3 エントリの `compile_commands_clang.json` を `tmp/clangdb/compile_commands.json` にコピーして使う。
- **リポジトリ直下の `compile_commands.json` は現在 JSONL ではない**（2,112 エントリの正しい JSON 配列・5.79 MB）。`json-compilation-database: Expected array` の回避が必要なのは手書き DB のみ。
- **cppcheck 2.22 に `--output-format=json` は無い**（`text`/`sarif`/`xml`/`xmlv2`/`xmlv3` のみ）。また `unknownMacro` ID も無い — JUCE の `#error` ノイズの正しい抑制は `preprocessorErrorDirective`。
- **cppcheck は JUCE ヘッダを compile database 無しで解析できない**（MSVC マクロを `-D` で上書きできないため）。`--project=` 必須。
- **Maxima に `--eval` は無い**（SBCL 層が消費して `Warning: argument eval not recognized`）。使えるのは `--quiet -batch "C:/forward/slash/path.mac"`（**バックスラッシュは消える**）または stdin パイプ（`;` 終端必須）。
- **headroom CLI に `compress`/`retrieve` サブコマンドは無い**（31 サブコマンド確認済み）。これらは MCP ツール限定。CLI は `savings` / `inspect` / `memory` を使う。
- **pwsh → WSL のクォートが最大の実害**。`wsl bash -lc "... 2>&1 ..."` は pwsh が `2>&1` を `C:\dev\null` というファイル名に書き換える。`<<'EOS'` へリドクも pwsh のパーサが拒否。**スクリプトをファイルに書いて `wsl bash /mnt/c/.../s.sh` で実行する**。WSL 内では POSIX パス（`/mnt/c/...`）を使い `C:\...` は使わない。
- **Get-ChildItem -Recurse を `.opencode/plugins/` にかけてはいけない**（node_modules が数万ファイル吐く）。`-Directory` か非 recursive の `-File` を使う。
- **`bufsize=0` の Popen は FileIO を返す**ので `read1()` は無い。`os.read(fd, n)` を使う。
- **OpenChamber を再起動すると `opencode` サーバープロセスが増殖する**（2026-09-26 時点で 2 個生存、各々 15 サーバー全部を起動するため MCP 起動が二重になる）。完全終了してから開き直すこと。

### agent-browser（2026-09-26 導入→同日削除）

- **`opencode-agent-browser` (npm・crottolo製 1.0.0) は使えない**。default export が V1 形状の関数のため、このビルドのプラグイン契約で SchemaError になる（`dist/index.js` 実読で確認）。
- `agent-browser` 本体（vercel-labs 0.38.1）は一時 MCP 登録して動作確認したが、**デーモン/セッションがコール間に死ぬ・コールド起動で数十秒・abort 後もサーバ側が継続する**等、実用に耐えないため **2026-09-26 に撤去**。撤去内容: `opencode.json` の `"agent-browser"` エントリ削除、`npm rm -g agent-browser`、`~/.agent-browser/`（Chrome-for-Testing 154 含む）削除。
- ブラウザ自動化が必要な場合は既存の **`obscura` MCP（37ツール・接続済み）**を使うこと。

### セキュリティ
WSL `~/.bashrc` に平文 `PERPLEXITY_API_KEY` がある。**漏えい扱いとしてローテーションを推奨**。
