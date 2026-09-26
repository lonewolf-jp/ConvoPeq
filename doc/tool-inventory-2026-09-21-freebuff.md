# Tool Inventory — Freebuff Desktop 環境 (2026-09-21 検証)

> 検証者: Buffy (Freebuff Desktop)
> 対象: 本リポジトリ `C:\VSC_Project\ConvoPeq`
> 検証方法: 各ツールの `--version` / `--help` / 実走、MCP は stdio ハンドシェイクを模擬するprober (`tmp/mcp-probe/probe_mcp.py`) で `initialize` → `tools/list` まで実行

## 0. 結論サマリ

| 区分 | 結果 |
|---|---|
| WSL ツール (grep/rg/fd/ag/fzf/sed/awk/ast-grep/rtk) | **全て使用可** |
| Windows CLI (ccc/graphify/semble/cppcheck/clang-tidy/tgrep/obscura/headroom) | **全て使用可** |
| Octave / Maxima / NumPy 系 | **使用可**（Maxima は PATH 外 → フルパス起動） |
| Python Web 系 (crawl4ai/ddgs/trafilatura) | **使用可** |
| MCP 14 サーバー | **本環境の設定ファイルに登録済み・全サーバー起動＆tools/list 成功**（初回利用時に承認ダイアログが必要） |
| Agent Skills | **`~/.agents/skills` の 131 スキルが全て読み込み可能**。依頼の 5 件も使用可（`pm-skills` は要修正→対応済み、`docmd` はスキル非提供のため CLI として使用） |
| Dr. Memory | **使用不可**（環境側の invasive security software 干渉、rc=127） |
| headroom proxy | 稼働中だが **Freebuff のトラフィックは通らない**（後述） |

## 1. 環境識別（重要）

Freebuff Desktop は Codebuff 系のハーネスで、構成が Qoder / Cline と異なる。

| 項目 | 値 |
|---|---|
| アプリ本体 | `C:\Users\user\AppData\Local\Programs\@codebufffreebuff-desktop\` (Electron) |
| エージェント実行体 | `resources\orchestrator\orchestrator.js` を `resources\bun\bun.exe` で起動 |
| オーケストレータ | `http://127.0.0.1:56744`（run ごとに変動）、ログ `%APPDATA%\Freebuff\logs\orchestrator-stderr.log` |
| モデル | `deepseek/deepseek-v4-flash`（`~/.config/manicode/settings.json`） |
| 推論先 | `https://www.codebuff.com`（アカウントトークン方式） |
| スレッド/会話 DB | `<project>\.freebuff\desktop-v2.db`、`~/.config/freebuff-desktop/state.json` |
| MCP 設定ファイル | **`C:\Users\user\.agents\mcp.json`**（＋サイドカー `C:\Users\user\.freebuff\mcp.json`） |

## 2. WSL ツール（Ubuntu-26.04）

`wsl.exe -l -v` → `Ubuntu-26.04 Running`。全て `wsl bash -lc "..."` で使用。

| ツール | パス | バージョン |
|---|---|---|
| grep | /usr/bin/grep | GNU grep |
| rg (ripgrep) | /usr/bin/rg | 15.1.0 |
| fd / fdfind | ~/.local/bin/fd, /usr/bin/fdfind | 10.3.0 |
| ag (the_silver_searcher) | /usr/bin/ag | 2.2.0 |
| fzf | /usr/bin/fzf | 0.67.0 |
| sed | /usr/bin/sed | GNU sed 4.9 |
| awk | /usr/bin/awk | GNU Awk 5.3.2 |
| ast-grep (`sg`) | /usr/local/bin/ast-grep, sg | 0.44.0 |
| rtk | ~/.local/bin/rtk | 0.49.0（累計 3023 コマンド / 57.5% 削減） |

### 呼び出しのコツ（この環境で確認）

- **単一引用符は外側のシェルに食われる**ため `wsl bash -lc 'for c in ...; do echo $c; done'` は `$c` が空になる。**heredoc を使う**のが確実:
  ```bash
  wsl bash -ls <<'EOS'
  cd /mnt/c/VSC_Project/ConvoPeq
  ~/.local/bin/rtk git status
  EOS
  ```
- rtk 経由の圧縮例: `rtk git status` / `rtk grep <pat>` / `rtk find` / `rtk diff` / `rtk gain`。

## 3. Windows CLI ツール

| ツール | パス | バージョン | 使用レシピ |
|---|---|---|---|
| ccc (cocoindex-code) | `C:\Users\user\.local\bin\ccc.exe` | 0.2.41 (uv tool) | `ccc status` / `ccc search "<自然言語>" --limit 10 [--json] [--lang cpp]` / `ccc grep 'foo(\(ARGS*\))' path` / `ccc index` / `ccc doctor`。**cwd 依存**（リポジトリルートで実行） |
| graphify | `...\Python314\Scripts\graphify.exe` | graphify 0.9.64 (pkg graphifyy 0.9.52) | `graphify query "<node>" [--budget N]` / `graphify path A B` / `graphify explain X` / `graphify update <path>`。`graphify-out/graph.json` は 74,421 nodes / 92,084 links（2026-09-21 12:58 更新） |
| semble | `C:\Users\user\.local\bin\semble.exe` | 0.6.0 | `semble search "<query>" ./path --top-k 10 [--format text] [--content docs|config|all]` / `semble find-related <file> <line> ./path` / `semble savings`。初回はモデルDLを伴う |
| cppcheck | `C:\Program Files\Cppcheck\cppcheck.exe` | 2.21.0 | `cppcheck --enable=warning,style --std=c++17 --quiet --suppress=missingIncludeSystem <file>`。JUCE マクロは `-D`/`--project` で設定しないと `unknownMacro` 警告が dominant |
| clang-tidy | `C:\Program Files\LLVM\bin\clang-tidy.exe` | LLVM 23.1.1 | リポジトリ直下の `compile_commands.json` は **JSONL 形式で読めない**。配列形式の `compile_commands_clang.json` を使う: `mkdir -p tmp/clangdb && cp compile_commands_clang.json tmp/clangdb/compile_commands.json && clang-tidy -p tmp/clangdb --quiet -checks='-*,bugprone-*' src/...cpp`（実走で警告取得を確認） |
| tgrep | `C:\Windows\system32\tgrep.exe` | 1.0.4 | `tgrep index [--force]` / `tgrep search <regex> [path]` / `tgrep status`。インデックス: 2,859 files / 265,894 trigrams（2026-09-21 更新） |
| obscura | `C:\Users\user\.local\bin\obscura\obscura.exe` | 0.2.2 | `obscura fetch <url>` / `scrape` / `serve` / `mcp`（Docker 不要の単体ヘッドレスブラウザ。SSRF 対策で loopback は `--allow-private-network` が必要） |
| Dr. Memory | `C:\Program Files (x86)\Dr. Memory\bin\drmemory.exe` | インストール済 | **実行不可**: `-batch -- <app>` で rc=127、「Dr. Memory failed to start the target application, perhaps due to interference from invasive security software」。環境側要因で設定では解決不能（AGENTS.md 記載どおり） |

## 4. 数値計算・Python ライブラリ

| 項目 | 状態 |
|---|---|
| GNU Octave | 11.3.0 `C:\Program Files\GNU Octave\Octave-11.3.0\mingw64\bin\octave-cli.exe`（`--eval` 実走 OK） |
| Maxima | 5.50.0 `C:\maxima-5.50.0\bin\maxima.bat`（**PATH 未登録**。`printf '2+3;\nintegrate(x^2,x);\nquit();\n' \| maxima.bat -q` で動作確認） |
| NumPy / SciPy / Matplotlib / pandas / IPython | 2.5.2 / 1.18.0 / 3.11.1 / 3.0.5 / 9.17.1 |
| crawl4ai | 0.9.3（CLI `crwl`, `crawl4ai-doctor`, `crawl4ai-setup`） |
| trafilatura | 2.2.0（CLI `trafilatura`、fetch+extract 実走 OK） |
| ddgs | 9.16.0（CLI `ddgs`） |
| mcp (Python) | 1.30.0 |

## 5. MCP — Freebuff 固有の仕組み（最重要）

他クライアントと設定ファイルが全く違う。**`.mcp.json` / `.qoder/settings.json` は読まれない。**

### 5.1 設定ファイルと探索

- 読み込み元（`orchestrator.js` の `McpConfigStore`）: `~/.agents/mcp.json`（本体）＋ `~/.freebuff/mcp.json`（サイドカー＝承認状態の保管）
- Codebuff SDK 側（`resources\app.asar`）は `{cwd}/.agents/mcp.json` → `{cwd}/../.agents/mcp.json` → `{homedir}/.agents/mcp.json` の順にマージ（後勝ち）。デスクトップのエージェントは **homedir の 1 ファイル**を使う。
- スキーマ（zod）:
  - stdio: `{"type":"stdio"?, "command": str, "args": [str], "env": {str:str}, "cwd": str?}`
  - remote: `{"type":"http"|"sse", "url": str, "headers": {str:str}}`（**sse は未対応**、url は `https:` または loopback のみ）
- ツール名は `mcp__<server>__<tool>`。`mcp__<server>__*` のワイルドカード許可/拒否ルールあり。

### 5.2 子プロセスの環境（API キーが届かない理由）

stdio サーバーは `cross-spawn` で **ホワイトリスト環境** の下に起動される:

```
APPDATA, HOMEDRIVE, HOMEPATH, LOCALAPPDATA, PATH, PROCESSOR_ARCHITECTURE,
SYSTEMDRIVE, SYSTEMROOT, TEMP, USERNAME, USERPROFILE, PROGRAMFILES
```

そのため `env` ブロックに無い環境変数（`FIRECRAWL_API_KEY` 等）は**継承されない**。さらに `$VAR` 展開は `HOME/PATH/SHELL/USER/LANG/TMPDIR` のみ解決し、それ以外は `Missing credential binding for X` として**そのサーバーが無効化**される（`${VAR}` 記法も非対応）。秘匿値は Electron 本体の secure storage（同意ブリッジ経由）にバインドする設計。

→ 本環境では **API キーをレジストリから読むラッパー** を用意して解決した（`C:\Users\user\.local\bin\mcp\`）:

| ラッパー | 読み出す環境変数 | 起動するサーバー |
|---|---|---|
| `context7-mcp.cmd` | `CONTEXT7_API_KEY` | node `@upstash/context7-mcp/dist/index.js` |
| `firecrawl-mcp.cmd` | `FIRECRAWL_API_KEY` | node `firecrawl-mcp/dist/index.js` |
| `brave-search-mcp.cmd` | `BRAVE_API_KEY` | node `@brave/brave-search-mcp-server/dist/index.js --transport stdio` |
| `github-mcp.cmd` | `GH_TOKEN` → `GITHUB_PERSONAL_ACCESS_TOKEN` | `C:\Users\user\tools\github-mcp-server\github-mcp-server.exe stdio` |

各スクリプトは `reg query "HKCU\Environment" /v <NAME>` でユーザー環境変数を取得する（Windows 環境変数を唯一の真実源に保ち、設定ファイルへ鍵を書かない）。

### 5.3 承認（consent）と同時接続数

- サーバーは `~/.agents/mcp.json` に書いただけでは `enabled: false`。初回利用時に Electron 本体の **同意ウィンドウ「Approve this connector?」** が開き、承認すると `~/.freebuff/mcp.json`（サイドカー）に `launchHash` / `manifestHash` / `allowedTools` が記録され有効化される。**この承認は人間が行う必要がある**（設定だけで回避しない）。
- 接続は `maxConnected: 8`。超過時は LRU で切断される（14 登録でも同時 8）。
- 設定はセッション開始時に読まれるため、**mcp.json を書いた後は新しいスレッド（ラン）を開始する**必要がある。

### 5.4 登録済みサーバー（2026-09-21 / 全 14、tools/list 検証済み）

| サーバー | 実体 | tools | 主なツール |
|---|---|---|---|
| serena | serena.exe 1.7.0 (`--context ide --mode editing --add-mode interactive`) | 32 | find_symbol, find_referencing_symbols, replace_symbol_body, write_memory, get_diagnostics_for_file, initial_instructions |
| aidex | aidex-mcp 2.3.0 (node) | 31 | aidex_query, aidex_signature(s), aidex_status, aidex_init/update, aidex_global_* |
| context-mode | 1.0.169 (node) | 11 | ctx_execute, ctx_execute_file, ctx_batch_execute, ctx_search, ctx_fetch_and_index, ctx_index |
| obscura | 0.2.2 | 37 | browser_navigate/snapshot/click/fill/markdown/screenshot/pdf |
| firecrawl | firecrawl-mcp 3.24.0 | 27 | firecrawl_scrape/search/crawl/extract/agent |
| github | github-mcp-server 1.12.2 | 31 | search_code, get_file_contents, create_pull_request, list_issues |
| headroom | 0.37.0 (`mcp serve`) | 3 | headroom_compress / headroom_retrieve / headroom_stats |
| graphify | graphifyy 0.9.52 | 10 | query_graph, get_node, get_neighbors, shortest_path, god_nodes |
| brave-search | 2.1.4 | 8 | brave_web_search, brave_news_search, brave_llm_context |
| ddgs | ddgs 9.16.0 (uvx) | 6 | search_text/news/images/videos/books, extract_content |
| microsoft-learn | `https://learn.microsoft.com/api/mcp` (http) | 3 | microsoft_docs_search, microsoft_code_sample_search, microsoft_docs_fetch |
| context7 | 4.1.1 (wrapper) | 2 | resolve-library-id, query-docs |
| semble | 0.6.0 (uvx `semble[mcp]`) | 2 | search, find_related |
| cocoindex | ccc mcp 0.2.41 | 1 | search（semantic） |

計 204 ツール。`cocoindex` / `graphify` / `serena` は `cwd` をリポジトリルートに固定している（ccc は cwd 依存、graphify は `graphify-out/graph.json` 相対解決）。

## 6. トークン削減 3 層パイプラインの実効構成

| 層 | Freebuff での実効性 | 根拠 |
|---|---|---|
| headroom proxy (8787) | **Freebuff の通信は通らない** | プロキシは稼働中（`headroom doctor` pass, v0.37.0）だが `headroom savings` の Today は `0 / 0 tokens`。エージェントは `https://www.codebuff.com` へ送信しており `ANTHROPIC_BASE_URL` を参照しない（Electron 版 Claude Desktop と同種の「Desktop が BASE_URL を上書きする」構図） |
| **context-mode MCP** | **有効（主軸）** | 11 ツール登録済み。ファイル分析・並列実行・検索・Web取得をここで処理しコンテキストに生データを入れない |
| **rtk (WSL)** | **有効** | `wsl bash -ls <<'EOS' ... ~/.local/bin/rtk <cmd> ... EOS`。累計 2.0M tokens (57.5%) 削減 |
| headroom MCP ツール | **手動で有効** | `headroom_compress` / `headroom_retrieve` を大きなコンテキストの退避に使用（proxy 自動圧縮が効かないことの代替） |

→ Freebuff では「**context-mode MCP（ソースで防ぐ）→ rtk（CLI 出所で圧縮）→ headroom MCP（手動退避）**」の 3 層で運用し、proxy が圧縮しているかのように報告してはならない。ダブル圧縮回避の原則は他環境と同じ。

## 7. 検証の自動化（再実行手順）

手作業で行ったこの検証は `scripts/freebuff_preflight.py` に落としてある。アップグレード後・マシン移行後・クライアント変更後に同じ 1 コマンドで再確認できる。

```bash
cd /c/VSC_Project/ConvoPeq
python scripts/freebuff_preflight.py            # フル（MCP 14 サーバーのハンドシェイクまで・約 90 秒）
python scripts/freebuff_preflight.py --quick    # MCP プローブと索引内容スキャンを省略（約 30 秒）
python scripts/freebuff_preflight.py --fix      # 加えて古い索引を実際に更新（ccc index / tgrep index / graphify update .）
python scripts/freebuff_preflight.py --json tmp/freebuff-preflight.json
```

- 終了コードは必須チェックに失敗があると 1（作業セッションのゲートに使える）。判定は `GO` / `GO with warnings` / `NO-GO`。
- `scripts/freebuff_expectations.json` がベースライン（サーバー別ツール数・索引の最大経過時間）。**ツール数のずれは WARN、ハンドシェイク失敗は FAIL**。ベースラインは実測に合わせて更新する。
- 単体の MCP プローブは `python scripts/freebuff_mcp_probe.py <spec.json> [server ...]`。spec は `{"name": {"command", "args", "env", "cwd"}}` もしくは `{"name": {"url"}}`。
- **MCP は Freebuff と同じ 12 変数のホワイトリスト env で起動して判定する。** 手元のシェルでは起動できても Freebuff では失敗するケース（環境変数・パス）があるため、これが唯一信頼できる判定方法。API キーのラッパーが鍵を取得できないと `exit /b 1` で落ちるので配管検証も兼ねる。
- 実測（2026-09-21）: フル = 116 項目 / 113 pass・1 warn・0 fail・2 info / 67 秒（スキル検証を追加後）。唯一の WARN は「MCP コネクタ未承認（サイドカー未生成）」＝人間の承認待ちで、これは設計どおり。

## 7.5 Freebuff の制約に対する穴埋め（2026-09-21 実装・検証済み）

| 制約（穴） | 穴埋め | 検証方法 |
|---|---|---|
| MCP 子プロセスが 12 変数しか継承せず **API キーが届かない** | `~/.local/bin/mcp/{context7,firecrawl,brave-search,github}-mcp.cmd` が `reg query "HKCU\Environment"` で読む | `check-keys.cmd` を制限 env で実行 → 4 キーすべて取得（値は長さと SHA-256 先頭 8 文字のみ表示） |
| **承認 (consent) は人間の操作**でしか通せない | 自動化せず、サイドカー `~/.freebuff/mcp.json` を読んで pending / approved / enabled を報告するだけにした | 未承認時の WARN 出力を確認 |
| `maxima` を名前で呼べない | `~/.local/bin/maxima.cmd`（cmd/PowerShell 用）＋ `~/.local/bin/maxima`（bash 用・`cmd.exe /c` は MSYS がスイッチを壊すので直接 exec） | `maxima` で `2+3;` → `5` |
| **Dr. Memory が使えずメモリ検証の空白** | **clang ASan**（`C:\Program Files\LLVM\lib\clang\23\lib\windows` のランタイム DLL を PATH に追加すれば動作）、プロジェクト側は `build.bat Debug msvc -DENABLE_ASAN=ON`、加えて Application Verifier (`C:\Windows\System32\appverif.exe`) | 意図的な heap-buffer-overflow を検出（rc=1, AddressSanitizer レポート） |
| 索引が古くても**無警告で古値**を返す | preflight が経過時間を判定し `--fix` で `ccc index` / `tgrep index` / `graphify update .` を実行 | tgrep の上限を 0 にして再インデックス（rc=0）を確認 |
| `$VAR` が展開されず**黙ってサーバーが無効化**される | preflight が解決可能名（HOME/PATH/SHELL/USER/LANG/TMPDIR）以外の参照を FAIL として検出 | schema チェックに実装済み |

## 8. Agent Skills（2026-09-21 検証）

### 8.1 読み込みの仕組み（orchestrator.js より）

- 探索順（後勝ちでマージ）: `~/.claude/skills` → `~/.agents/skills` → `{cwd}/.claude/skills` → `{cwd}/.agents/skills`。
- 設置先の役割: `installSkillsDir = ~/.agents/skills`（インストール先）、`globalSkillsDir = ~/.freebuff/skills`（エージェントが作るグローバルスキル）、`createSkillsDir = {cwd}/.agents/skills`（プロジェクト用）。
- **1 スキル = 1 ディレクトリ + `<dir>/SKILL.md`**。探索は直下の 1 階層のみで、ネストした SKILL.md は自動では読み込まれない（必要時にエージェントが読む）。
- 検証規則（zod）: ディレクトリ名・frontmatter `name` は `/^[a-z0-9]+(-[a-z0-9]+)*$/`（64 字以内）、`description` は必須（1024 字に切り詰め）、任意で `license` / `disable-model-invocation` / `metadata`（`allowed-tools` など他のキーは黙って無視される）。
- 読み込みに失敗したスキルは**無言でスキップ**され、警告も出ない。

### 8.2 依頼された 5 スキルの状態

| スキル | upstream | 結果 | 内容 |
|---|---|---|---|
| reverse-skill | zhaoxuya520/reverse-skill | **使用可** | `reverse-skill`（router SKILL.md、`name: reverse-skill-router`）＋ `INDEX.md`/`MASTER-ROUTING.md` と約 89 のモジュール（`api-security/` 等）。モジュールはネストしているため個別スキルとしては登録されず、router 経由で読む |
| debug-skill | AlmogBaku/debug-skill | **使用可（1 件修正）** | スキル `debugging-code` と `dap` CLI（v0.4.2）。`dap` は `python3` を名前で呼ぶため Windows では失敗 → `~/.local/bin/python3`（＋`.cmd`）シムを追加して素の `dap debug app.py --break app.py:L` が動作（実測でブレークポイント停止・locals/stack 表示） |
| UI UX Pro Max | nextlevelbuilder/ui-ux-pro-max-skill | **使用可** | ファミリー 7 スキル（`ui-ux-pro-max` `design` `design-system` `ui-styling` `slides` `banner-design` `brand`）が data/references/scripts 込みで導入済み。`scripts/search.py` を実走して設計ガイド（landing パターン等）が返ることを確認 |
| docmd | docmd-io/docmd | **CLI として使用可（スキルは上流に無い）** | 上流リポジトリに SKILL.md は 0 個。npm `@docmd/core` 0.9.5 がグローバル導入済みで、実際に `docmd init` → `docmd build` が完走（サイト・sitemap・llms.txt・OKF を生成）。エージェント連携は `docmd mcp`（stdio MCP）または生成物 `llms.txt` 経由 |
| PM-Skills | product-on-purpose/pm-skills | **使用可（1 件修正）** | リポジトリ clone は `SKILL.md` を持たず**読み込み不能**だった → router `SKILL.md` を新規作成し `pm-skills` として読み込み可能に（68 スキル・7 agents・11 commands・10 workflows への progressive disclosure）。フェーズ別スキル（`define-*`/`deliver-*`/`utility-pm-*`）は 20 件程度が兄弟ディレクトリとして個別導入済み |

### 8.3 なぜ全部を個別スキルにしないか

スキルの `name` + `description` はエージェントの文脈に常時注入される。reverse-skill の 89 モジュールや pm-skills の 68 スキルを全て展開すると、起動時のカタログが数百件に膨らむ。**router SKILL.md を 1 枚置き、必要時に該当 SKILL.md だけを読ませる**のが、機能を落とさずトークンを膨らませない解（両リポジトリが本来想定している使い方でもある）。

### 8.4 検証の自動化

`scripts/freebuff_preflight.py` の `Agent Skills` セクションが、4 つの探索ディレクトリを全走査して**アプリと同じ規則**（ディレクトリ名・SKILL.md の有無・YAML frontmatter の name/description）で合否を出し、依頼 5 件の状態（スキルディレクトリと CLI の版数）も個別に報告する。破損スキルは FAIL（無言スキップは事故の元のため）、期待値は `scripts/freebuff_expectations.json` の `skills` ブロック。

## 9. 未対応・注意点

- **Dr. Memory は使用不可**（rc=127 / -1、セキュリティソフト干渉）。メモリ検証は clang ASan（ランタイム DLL を PATH に追加）または `ENABLE_ASAN=ON` ビルド、Application Verifier を使う（§7.5）。
- **Maxima は PATH 未登録だが `maxima` シムで名前解決可能**（`~/.local/bin/maxima.cmd` と `~/.local/bin/maxima`）。実体は `C:\maxima-5.50.0\bin\maxima.bat`。
- **tgrep / graphify は無警告で古い索引を返す**。使用前に `tgrep status`（Updated xh ago）と `graphify-out/graph.json` の mtime を確認し、必要なら `tgrep index` / `graphify update .`。
- AiDex インデックスは `.aidex/index.db`（487 files / 20,564 items / 288,027 occurrences、2026-09-21 更新）と `~/.aidex/global.db`（3 projects / 1,977 embeddings）。使用前に `aidex_status` / `aidex_global_status` で鮮度確認、必要なら `aidex_init` / `aidex_update` / `aidex_global_refresh`。
- `ccc` のインデックスは 305,853 chunks / 2,596 files（2026-09-21 更新）で新鮮。
- Firecrawl / Brave / Context7 / GitHub は外部 API 依存（レート制限・ネットワーク前提）。
- **MCP 設定変更はセッション開始時のみ反映**。設定後は新しいスレッドを開始し、初回ツール使用時に同意ダイアログを承認すること。
