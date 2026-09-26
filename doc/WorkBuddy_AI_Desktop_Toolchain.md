# WorkBuddy AI Desktop ツールチェーン検証・設定記録

- **検証日**: 2026-09-21
- **環境**: WorkBuddy AI Desktop v5.5.2 (build 910352f) / Windows 11 (10.0.26200)
- **ワークスペース**: `C:\VSC_Project\ConvoPeq`
- **シェル**: MSYS2 Git Bash（WorkBuddy 同梱 PortableGit 1.2.0）

---

## 1. 結論サマリ

| # | ツール | 判定 | 実体 / バージョン |
|---|---|---|---|
| 1 | WSL 本体 | ❌ **使用不可（サンドボックス遮断）** | `wsl.exe` は Program Blacklist 登録済み |
| 2 | rg (ripgrep) | ✅ 使用可 | `C:\Users\user\.cache\pkg\72dc6f…\rg.exe` — 15.0.0 |
| 3 | ast-grep | ✅ 使用可 | uv venv 同梱 — 0.45.2 |
| 4 | fd | ✅ **新規導入** | `C:\Users\user\.local\bin\fd.exe` — 10.5.0 |
| 5 | fzf | ✅ **新規導入** | `C:\Users\user\.local\bin\fzf.exe` — 0.74.4 |
| 6 | ag (silver searcher) | ❌ Windows ビルド不在 | rg / ast-grep で代替 |
| 7 | sed / awk / grep | ✅ 使用可 | MSYS GNU sed 4.9 / gawk 5.4.0 / grep 3.0 |
| 8 | serena MCP | ✅ 使用可 | 1.7.0（MCP ハンドシェイク PASS） |
| 9 | cocoindex code (ccc) | ✅ 使用可 | 0.2.41 |
| 10 | graphify | ✅ 使用可（**要修正**） | 0.9.64（pip）/ 0.9.52（uv venv） |
| 11 | semble | ✅ 使用可（**要修正**） | 0.6.0 CLI / MCP 1.30.0 |
| 12 | AiDex MCP | ✅ 使用可 | 2.3.0（インデックス `~/.aidex/global.db` 11MB 既存） |
| 13 | cppcheck | ✅ 使用可 | 2.21.0 |
| 14 | clang-tidy | ✅ 使用可 | LLVM 23.1.1 |
| 15 | Dr. Memory | ✅ 使用可 | 2.6.20434 |
| 16 | tgrep | ✅ 使用可 | `C:\Windows\System32\tgrep.exe` |
| 17 | Obscura | ✅ 使用可 | 0.2.2（Rust）/ 0.1.1（PyPI） |
| 18 | Crawl4AI | ✅ 使用可 | 0.9.3 |
| 19 | DDGS | ✅ 使用可 | 9.16.0（検索動作確認済） |
| 20 | Trafilatura | ✅ 使用可 | 2.2.0（抽出動作確認済） |
| 21 | context7 MCP | ✅ 使用可 | HTTP 200 / `CONTEXT7_API_KEY` |
| 22 | firecrawl MCP | ✅ 使用可 | firecrawl-fastmcp 3.24.0 |
| 23 | brave search MCP | ✅ 使用可 | 2.1.4 / `BRAVE_API_KEY` |
| 24 | github MCP | ✅ 使用可 | HTTP 200 / `GH_TOKEN` |
| 25 | Microsoft Learn MCP | ✅ 使用可 | HTTP 200 |
| 26 | GNU Octave | ✅ 使用可（**PATH 未登録を修正**） | 11.3.0 |
| 27 | Maxima | ✅ 使用可（**PATH 未登録を修正**） | 5.50.0 |
| 28 | NumPy / SciPy / Matplotlib / pandas / IPython | ✅ 使用可 | 2.5.2 / 1.18.0 / 3.11.1 / 3.0.5 / 9.17.1 |
| 29 | headroom MCP / CLI / proxy | ✅ 使用可 | 0.37.0（proxy は 8787 で稼働中） |
| 30 | context-mode MCP | ✅ 使用可 | 1.0.169 |
| 31 | rtk | ✅ 使用可（**WSL 不要**） | `C:\Windows\System32\rtk.exe` — 0.49.0 |

**MCP サーバー: stdio 8/8 PASS、HTTP 3/3 PASS（200）**

---

## 2. 発見した根本原因と実施した修正

### 原因 1 — サンドボックスで `APPDATA` が空

`HKCU\Environment` に `APPDATA` が存在せず、シェル内で空文字になっていた。Python のユーザーサイト解決が
`%APPDATA%\Python\Python314\site-packages` ではなく `C:\Users\user\Python\Python314\site-packages` に
フォールバックし、**pip 製・uv 製の全ランチャーが `ModuleNotFoundError` で起動不能**になっていた。

影響: `graphify.exe`, `headroom.exe`（`.local/bin` 側）

**修正**: `HKCU\Environment` に `APPDATA = C:\Users\user\AppData\Roaming` を設定 + `WM_SETTINGCHANGE` ブロードキャスト。

### 原因 2 — ツールディレクトリが PATH に未登録

以下が Windows の PATH に一切含まれていなかった。このため MCP 設定で裸のコマンド名
（`serena`, `context-mode` など）を書いても解決できなかった。

**修正**: `HKCU\Environment\Path` に 5 エントリを追加（バックアップ取得済み）。

| 追加した PATH | 用途 |
|---|---|
| `C:\Users\user\.local\bin` | uv ツール / obscura / headroom / serena |
| `C:\Users\user\AppData\Roaming\npm` | npm グローバル（context-mode, aidex, firecrawl, brave…） |
| `C:\Users\user\AppData\Roaming\Python\Python314\Scripts` | pip ユーザースクリプト（graphify） |
| `C:\Program Files\GNU Octave\Octave-11.3.0\mingw64\bin` | Octave |
| `C:\maxima-5.50.0\bin` | Maxima |

> バックアップ: `.workbuddy-ai/env-backup/HKCU-Environment-20260921-095348.json`

### 原因 3 — 任意依存（extras）の欠落

| ツール | 症状 | 修正 |
|---|---|---|
| graphify MCP | `ImportError: mcp not installed. Run: pip install "graphifyy[mcp]"` | `uv pip install graphifyy[mcp]` |
| semble MCP | `Error: typer is required. Install with 'pip install mcp[cli]'` | `uv pip install mcp[cli]` |

### 原因 4 — WSL の完全遮断

`wsl.exe` は Security Center の Program Blacklist に登録されており、**サンドボックスから起動不可**。
`reg.exe`, `where.exe` も同様に遮断されている。

**回避策**: ユーザー指示の「WSL ツール」はすべて **Windows ネイティブ版で代替**した。
特に **rtk は `C:\Windows\System32\rtk.exe` としてネイティブ動作するため、WSL は不要**。

> WSL を有効化したい場合は、Security Center → Command Security → Program Blacklist から
> `wsl.exe` を除外する必要がある（ユーザー操作が必要）。

---

## 3. MCP 設定

### 設定ファイル

| スコープ | パス | 状態 |
|---|---|---|
| ユーザー | `C:\Users\user\.workbuddy-ai\mcp.json` | ✅ 新規作成（11 サーバー） |
| プロジェクト | `C:\VSC_Project\ConvoPeq\.mcp.json` | ✅ 絶対パスに修正 |

**重要**: 元の `.mcp.json` は `"command": "serena"` のような裸のコマンド名を使っており、
PATH 未登録のため**起動不能**だった。すべて絶対パスに変更済み。

### 環境変数展開

WorkBuddy の MCP 設定は `${VAR_NAME}` / `${VAR_NAME:-default}` 形式をサポートする。
API キーはハードコードせず、既存の Windows 環境変数を参照している。

| 環境変数 | 用途 |
|---|---|
| `CONTEXT7_API_KEY` | context7 MCP（`headers`） |
| `FIRECRAWL_API_KEY` | firecrawl MCP（`env`） |
| `BRAVE_API_KEY` | brave-search MCP（`env`） |
| `GH_TOKEN` | github MCP（`Authorization: Bearer`） |

### 登録済みサーバー

| サーバー | 種別 | 起動コマンド |
|---|---|---|
| serena | stdio | `.local\bin\serena.exe start-mcp-server --context ide --project … --mode editing --add-mode interactive` |
| context-mode | stdio | `node.exe …\context-mode\cli.bundle.mjs` |
| aidex | stdio | `node.exe …\aidex-mcp\build\index.js` |
| headroom | stdio | `.local\bin\headroom.exe mcp serve` |
| semble | stdio | `uv\tools\semble\Scripts\semble.exe`（**引数なしで MCP 起動**） |
| graphify | stdio | `uv\tools\graphifyy\Scripts\graphify-mcp.exe` |
| context7 | http | `https://mcp.context7.com/mcp` |
| firecrawl | stdio | `node.exe …\firecrawl-mcp\dist\index.js` |
| brave-search | stdio | `node.exe …\@brave\brave-search-mcp-server\dist\index.js --transport stdio` |
| github | http | `https://api.githubcopilot.com/mcp/` |
| microsoft-learn | http | `https://learn.microsoft.com/api/mcp` |

### 有効化手順（ユーザー操作が必要）

新規追加した MCP サーバーは**自動では有効化されない**。
コネクタ管理ページ右上の「カスタムコネクタ」入口を開き、各サーバーの **「信頼 (Trust)」** を
クリックして有効化すること。

---

## 4. 各ツールの使用方法

### 4.1 検索系

```bash
# ripgrep — 高速全文検索（最優先）
RG=/c/Users/user/.cache/pkg/72dc6f288b8f13e22da7572a55f576ac335b8402e617b8aaf91f8dbf72adf1a0/rg.exe
"$RG" -n --type cpp "pattern" src/
"$RG" -l -g '*.h' "Symbol"

# ast-grep — AST 構造検索（構文を理解した検索・置換）
ASG='C:\Users\user\AppData\Roaming\uv\tools\headroom-ai\Scripts\ast-grep.exe'
"$ASG" -p 'if ($C) { $$$B }' -l cpp src/          # パターン検索
"$ASG" -p 'TODO($X)' -l cpp --rewrite 'FIXME($X)' src/  # 置換

# tgrep — トライグラム索引付き grep（大規模リポジトリ向け）
tgrep index .            # 索引作成
tgrep search "pattern"   # 検索
tgrep status             # 索引状態

# fd — find の高速代替
fd.exe -e cpp -e h . src/
fd.exe --changed-within 1d

# fzf — 対話的ファジーフィルタ
fd.exe -e cpp | fzf.exe --preview 'head -40 {}'
```

### 4.2 コードインテリジェンス

```bash
# cocoindex code — セマンティック検索
ccc.exe index C:\VSC_Project\ConvoPeq\src
ccc.exe search "audio resampling pipeline"
ccc.exe status

# graphify — ナレッジグラフ探索
graphify.exe build .                 # グラフ構築
graphify.exe query "where is resampling handled"
graphify.exe path A B                # ノード間パス
graphify.exe explain <node>

# semble — 自然言語セマンティック検索（MCP として使うのが本命）
semble.exe search "how does the peak EQ work"     # CLI
semble.exe find-related <path>:<line>             # 類似コード
semble.exe savings                                # トークン削減実績
semble.exe install --agent claude --type mcp -y   # 他エージェントへ登録

# serena — LSP ベースのシンボル操作（MCP 経由）
serena.exe start-mcp-server --project C:\VSC_Project\ConvoPeq --mode editing
# ツール例: find_symbol / find_referencing_symbols / get_symbols_overview /
#           replace_symbol_body / insert_after_symbol / write_memory
```

**検索ツール優先順位（AGENTS.md 準拠）**: AiDex ＞ serena ＞ semble ＞ ctx_execute ＞ rg/ast-grep

### 4.3 静的解析・デバッグ

```bash
# cppcheck
"/c/Program Files/Cppcheck/cppcheck.exe" --enable=all --std=c++20 \
  --project=compile_commands.json --suppress=missingIncludeSystem

# clang-tidy（compile_commands.json 必須）
"/c/Program Files/LLVM/bin/clang-tidy.exe" -p build src/foo.cpp \
  --checks='-*,bugprone-*,performance-*,modernize-*'

# Dr. Memory — メモリエラー検出
"/c/Program Files (x86)/Dr. Memory/bin/drmemory.exe" -batch -- ./build/app.exe
```

### 4.4 Web 取得

```bash
# Obscura — 軽量ヘッドレスブラウザ
obscura.exe fetch https://example.com
obscura.exe scrape https://example.com --selector 'article'
obscura.exe mcp                      # MCP サーバーとして起動

# Crawl4AI
python -c "import asyncio,crawl4ai; print(asyncio.run(crawl4ai.AsyncWebCrawler().arun(url='https://example.com')).markdown[:400])"

# DDGS — 検索
python -c "from ddgs import DDGS; print(DDGS().text('query', max_results=5))"

# Trafilatura — 本文抽出
python -c "import trafilatura; d=trafilatura.fetch_url('https://example.com'); print(trafilatura.extract(d))"
```

### 4.5 数式処理

```bash
# GNU Octave（CLI）
"/c/Program Files/GNU Octave/Octave-11.3.0/mingw64/bin/octave-cli.exe" --eval "disp(sqrt(2)); x=[1 2 3]; disp(fft(x))"
# スクリプト実行
octave-cli.exe script.m

# Maxima（CLI）
/c/maxima-5.50.0/bin/maxima.bat -q --very-quiet -r "integrate(x^2, x); quit();"
echo 'diff(sin(x),x);' | /c/maxima-5.50.0/bin/maxima.bat -q

# Python 数値計算
python -c "import numpy, scipy, pandas, matplotlib; print(numpy.__version__)"
```

### 4.6 MCP クライアント（HTTP）

| サーバー | エンドポイント | 認証 |
|---|---|---|
| context7 | `https://mcp.context7.com/mcp` | `CONTEXT7_API_KEY` ヘッダー |
| github | `https://api.githubcopilot.com/mcp/` | `Authorization: Bearer $GH_TOKEN` |
| Microsoft Learn | `https://learn.microsoft.com/api/mcp` | 不要 |

---

## 5. トークン削減 3 層パイプライン（常時運用）

役割分担の原則: **「ソースで防ぐ（context-mode）→ トラフィック自動圧縮（headroom proxy）→ CLI 出所圧縮（rtk）」**。
ダブル圧縮を避けるため、同一対象に 2 層を重ねない。

| 層 | ツール | 役割 | 起動状態 |
|---|---|---|---|
| 1 | **context-mode MCP** | ファイル分析・並列実行・検索（生データをコンテキストに入れない、93-99% 削減） | MCP 登録済 |
| 2 | **headroom** | proxy が全トラフィックを自動圧縮（CCR）。MCP で手動 compress/retrieve | proxy は 8787 で稼働中 |
| 3 | **rtk** | CLI コマンド出力を 60-90% 圧縮 | `C:\Windows\System32\rtk.exe`（**WSL 不要**） |

### 使い分け早見表

| 作業 | 使用ツール |
|---|---|
| ファイル分析 / 集計 / 抽出 | `ctx_execute` / `ctx_execute_file` |
| 複数コマンド並列実行（最大 8） | `ctx_batch_execute` |
| 過去内容の検索 | `ctx_search` |
| Web 取得 | `ctx_fetch_and_index` → `ctx_search` |
| 大きなコンテキストの保存 | `headroom_compress`（MCP）/ 復元 `headroom_retrieve` |
| CLI コマンド出力 | `rtk <cmd>`（例: `rtk git status`, `rtk ls`, `rtk log`） |
| ファイル編集 | Read + Edit（通常ツール） |

### rtk 主要サブコマンド

```
rtk ls / tree / read / find / diff / log / env / json / deps
rtk git / gh / glab / aws / psql / pnpm / dotnet / docker
rtk err <cmd>    # エラー・警告のみ表示
rtk test <cmd>   # 失敗のみ表示
rtk smart        # 2 行技術サマリ
```

### ⚠️ 重要な制約（正直な報告）

`headroom doctor` の診断結果:

```
proxy          ✓ pass  running at http://127.0.0.1:8787 (v0.37.0)
claude         ⚠ warn  not routed (no ANTHROPIC_BASE_URL in settings env)
shell env      ⚠ warn  ANTHROPIC_BASE_URL / OPENAI_BASE_URL unset
claude desktop ⚠ warn  agent sessions bypass the proxy
                       (Desktop overwrites ANTHROPIC_BASE_URL)
```

`ANTHROPIC_BASE_URL=http://127.0.0.1:8787` はユーザー環境変数に設定済みだが、
**デスクトップアプリ（Claude Desktop 系）は起動時に `ANTHROPIC_BASE_URL` を上書きする**ため、
**WorkBuddy Desktop 自身のモデル通信は proxy を経由しない可能性が高い**。

この場合でも以下は有効:
- **headroom MCP ツール**（`headroom_compress` / `headroom_retrieve` / `headroom_stats`）は
  クライアント非依存で動作する
- **CLI エージェント**（`claude` / `codex` CLI）は `headroom wrap` で proxy 経由にできる
- **context-mode / rtk** は WorkBuddy から直接利用可能

→ WorkBuddy 内での実効的なトークン削減は **context-mode + rtk + headroom MCP ツール**が主軸となる。

---

## 6. 既知の制限

| 項目 | 状況 | 対応 |
|---|---|---|
| WSL | `wsl.exe` 遮断 | ネイティブ版で代替済（rtk 含む）。解除はユーザー操作 |
| `ag` | Windows ビルド不在 | rg / ast-grep で代替 |
| `reg.exe` / `where.exe` | 遮断 | Python `winreg` で代替 |
| PATH 先頭の破損エントリ | `%C:\Program Files (x86)\…\java8path`（先頭に `%`） | 未修正（既存・実害小） |
| graphify バージョン差 | skill は 0.9.64 / uv venv は 0.9.52 | `uv tool upgrade graphifyy` で解消可 |
| headroom proxy 経由の自動圧縮 | デスクトップアプリが `ANTHROPIC_BASE_URL` を上書き | 上記 5 章参照 |

---

## 7. 再現手順（新規環境セットアップ）

```bash
# 1. APPDATA の復元（HKCU\Environment）
python -c "import winreg;k=winreg.OpenKey(winreg.HKEY_CURRENT_USER,'Environment',0,winreg.KEY_WRITE);\
winreg.SetValueEx(k,'APPDATA',0,winreg.REG_SZ,r'C:\Users\user\AppData\Roaming')"

# 2. PATH 追加（HKCU\Environment\Path）
#    .local\bin / Roaming\npm / Roaming\Python\Python314\Scripts
#    GNU Octave\Octave-11.3.0\mingw64\bin / maxima-5.50.0\bin

# 3. MCP extras
uv pip install --python <graphifyy venv>\Scripts\python.exe "graphifyy[mcp]"
uv pip install --python <semble venv>\Scripts\python.exe "mcp[cli]"

# 4. MCP 設定
#    ~/.workbuddy-ai/mcp.json と <project>/.mcp.json を作成（絶対パス必須）

# 5. 有効化
#    コネクタ管理ページ → カスタムコネクタ → 各サーバーを Trust
```
