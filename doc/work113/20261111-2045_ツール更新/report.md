# ツール更新（headroom 0.44.0 / graphify 0.9.106）

OpenCode V2 + OpenChamber（Windows 11 x64）環境で、headroom と graphify を最新版へ更新し、MCP・実セッション・機能呼出しまで動作確認した記録である。**headroom の proxy 機能は使用しない**（Owner 指示 2026-11-11）。

## §0 結論

| ツール | 更新前 | 更新後 | 動作確認 |
|---|---|---|---|
| headroom-ai | 0.43.1 | **0.44.0** | MCP 3 tools・実セッション接続・`headroom_stats` 応答 OK |
| graphifyy | 0.9.104 | **0.9.106** | MCP 10 tools・実セッション接続・`graph_stats` 応答 OK・グラフ再生成 OK |

- **最重要の学び**: headroom は v0.43.0 で MCP サーバを SDK 2.x に移行しており、`mcp` extra なしのインストールでは `headroom mcp serve` が `ImportError: MCP SDK not installed` で即死する。`uv tool install "headroom-ai[mcp]==<ver>" --force` が必須。
- graphify は更新後に**スキルの同期**が必要（`graphify install --platform <p>`）。同期しないと `graphify --version` が毎回 warning を出す。
- AGENTS.md にあった 3 件の記述が実測と食い違っていたため訂正した（§5）。

## §1 更新前の状態（実測）

| 項目 | 実測値 |
|---|---|
| uv | 0.13.3（`C:\Users\user\AppData\Roaming\Python\Python315\Scripts\uv.exe`） |
| headroom-ai | 0.43.1（`uv/tools/headroom-ai/Scripts/headroom.exe`） |
| graphifyy | 0.9.104（`uv/tools/graphifyy/Scripts/graphify{,-mcp}.exe`） |
| MCP 構成 | プロジェクト `opencode.json` の 12 サーバー＋グローバル 3。headroom / graphify は `pythonw.exe nowin.py …` 経由 |
| 稼働中 MCP 子プロセス | headroom 4 組・graphify 4 組（計 16 プロセス。nowin ラッパー＋uv python 子） |
| proxy（8787） | **未稼働**（`Test-NetConnection` = False）。スタートアップ bat も無し |
| `.local\bin\headroom.exe` | uv tool の exe と **SHA-256 一致**（47,104 B・0.44.0）。旧記録の「破損」は誤り |

## §2 更新手順（実行したコマンド）

### 2.1 事前: MCP 子プロセスの停止（Windows の exe ロック回避）

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'headroom-ai' -or $_.CommandLine -match 'graphifyy' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

- nowin ラッパー（`pythonw.exe`）と uv python 子を含めて停止する必要がある。停止すると MCP 名前空間が一時的に消える（このセッションでも `headroom` / `graphify` のツールが見えなくなった）。

### 2.2 headroom

```powershell
uv tool install "headroom-ai[mcp]==0.44.0" --force
```

- `uv tool upgrade headroom-ai` は **効かない**: `Nothing to upgrade` ＋ `headroom-ai is pinned to 0.43.1 … reinstall with uv tool install headroom-ai@latest`。
- `[mcp]` extra により tool venv 内に **mcp 2.3.0**（`mcp>=2.2,<3`）が入る。**隔離 venv なので、共有 site-packages の `mcp<2` 制約（headroom / github-mcp-server を壊す）とは矛盾しない。**
- `[proxy]` extra は**入れていない**（proxy 不使用）。`headroom proxy` は `Error: Proxy dependencies not installed … No module named 'fastapi'` で落ちる（実測）。

### 2.3 graphify

```powershell
uv tool install "graphifyy[mcp]==0.9.106" --force
foreach ($p in 'opencode','claude','codex','copilot') { graphify install --platform $p }
```

- インストール直後の `graphify --version` は 4 プラットフォーム分のスキルが 0.9.104 のままで warning を出した。4 回の `install --platform` で解消。
- 0.9.106 では `install` の `requires-python` が `>=3.10,<3.15` にキャップされ、Python 3.15 を選ばなくなった（tree-sitter の wheel が無くソースビルドが失敗するため）。

### 2.4 反映

`opencode.exe` のみ再起動（OpenChamber 本体は触らない）。supervisor が再起動し、MCP 15 サーバーが再接続して設定を再読込する。

## §3 動作確認（実測）

### 3.1 バイナリ

```
headroom, version 0.44.0          # サブコマンド 30 個を実測
graphify 0.9.106                  # warning なし
```

### 3.2 MCP ハンドシェイク（`mcp-probe.py`）

```
[OK] headroom  {'name': 'headroom', 'version': ''}   tools(3): headroom_compress, headroom_retrieve, headroom_stats
[OK] graphify  {'name': 'graphify', 'version': '0.9.106'}  tools(10): query_graph, get_node, get_neighbors, get_community,
                                                            god_nodes, graph_stats, shortest_path, list_prs, get_pr_impact, triage_prs
```

- 更新直後は headroom が `[TIMEOUT/FAIL] JSONDecodeError` になった。原因は上記の `MCP SDK not installed`（`[mcp]` extra 不足）。extra 付き再インストールで解消。

### 3.3 実セッション接続（`opencode.log`）

```
message="mcp connected" server=headroom tools=3   (2026-11-11T11:18:38Z)
message="mcp connected" server=graphify tools=10  (2026-11-11T11:18:44Z)
```

- 再起動後、MCP 子プロセス 8 個（headroom 4・graphify 4）が復帰。

### 3.4 機能呼出し（MCP ツールを実際に呼ぶ）

- `headroom_stats` → 正常応答（`session_duration_seconds` / `compressions: 0` など）。同時に
  `proxy.status: unreachable`（`http://127.0.0.1:8787`）を明示的に返す。**MCP サーバ自体は proxy 無しで動作し、compress/retrieve の実体は proxy 側**という構造が確認できた。
- `graph_stats` → `Nodes: 63,344 / Edges: 80,669 / Communities: 3,318 / EXTRACTED 88% / INFERRED 12%`。
  同時に `HEAD is c94d376, graph built at fb10111: graph may be stale` を返したため、`graphify update .` で再生成した（§3.5）。

### 3.5 グラフ再生成

```
Updating code graph in . (no LLM needed)...
[graphify watch] Incremental: re-extracting 194 of 2729 scanned files that changed since the last build.
AST extraction: 190/190 uncached files (100%) [16 workers]
warning: 3 file(s) had syntax errors and may be partially extracted: src/DeviceSettings.cpp (error region at line 126, 26 symbol(s) extracted), ...
warning: 4 code file(s) yielded no symbols (0.02 MB)
[graphify watch] fail-closed: kept 8 node(s) from 1 file(s) that left the scan corpus ...
[graphify] backed up curated graph (5 files) -> 2026-11-11/
[graphify watch] Rebuilt: 67560 nodes, 85861 edges, 3464 communities
[graphify watch] graph.json, graph.html and GRAPH_REPORT.md updated in graphify-out
Code graph updated. For doc/paper/image changes run /graphify --update in your AI assistant.
exit=0
```

| 項目 | 更新前 | 更新後 |
|---|---|---|
| nodes | 63,344 | **67,560** |
| edges | 80,669 | **85,861** |
| communities | 3,318 | **3,464** |
| build commit | fb10111（古い） | **c94d3767（HEAD と一致）** |
| graph.json | — | 66,809,099 B（mtime 20:27） |

- 更新後の `graph_stats`（MCP）は `Built at commit: c94d3767… (matches HEAD)` を返し、**stale 警告は消えた**。
- 付随警告: 3 ファイルの構文エラー領域（`src/DeviceSettings.cpp:126` ほか）と 4 ファイルのシンボル 0 件は**今回の更新前から存在する抽出限界**であり、ツール更新の失敗ではない。
- コミュニティ数が増えたため保存済みラベル（2,550 件）が古くなり、`graphify label` の再実行を促す警告が出る。**ラベル付けは LLM を使うため本作業では実行していない**（Graphify Cloud も未使用）。

## §4 変更点（学習結果）

### 4.1 headroom 0.43.1 → 0.44.0（Features 13 / Bug Fixes 45 / Dependencies 6）

| 区分 | 内容 | 本環境への影響 |
|---|---|---|
| 破壊的変更 | `all` extra から PyTorch 系 extras を除外（#4067） | なし（`[mcp]` のみ使用） |
| 機能 | `learn` が **OpenCode V2 のセッションスキーマ**を読む（#4199） | あり（本環境は V2） |
| 機能 | `wrap` で **Serena code-memory MCP が opt-in**（#4509） | あり（Serena 使用時） |
| 修正 | `doctor` が IPv6 loopback・大文字小文字違いの proxy URL を認識 | あり（診断精度） |
| セキュリティ | `asgi` がクライアント断でバッファリング停止（GHSA-mc2f-vg4v-2fh4） | proxy 不使用でも更新価値あり |
| その他 | proxy / model-router / dashboard 系（`HEADROOM_MODEL_ROUTER_LOG_ONLY` の shadow mode、`/v1/compress` lossless、per-request `cost_usd`、Redis 可視化、Hermes / Factory Droid wrap） | なし（proxy 不使用） |

- さらに v0.43.0 で **MCP サーバの SDK 2.x 移行**（#4603）と **Claude/Codex/Copilot/skills.sh 向け headroom skill** が入っている。今回の `[mcp]` extra 必須化はこの移行の帰結。

### 4.2 graphify 0.9.104 → 0.9.106（C++ プロジェクトなので C++ 解析の改善が主目的）

| 版 | 内容 |
|---|---|
| 0.9.106 | `install` の `requires-python` キャップ（3.15 回避）／**前方宣言が唯一の定義へ折りたたまれる**／`using Alias = ns::Type;` 経由のメンバ呼出し・継承フィールド・前方宣言ポインタの完成／out-of-line テンプレートメンバの所有者保持／`#if` 後の初期化子内呼出しの復元 |
| 0.9.106 | query の BFS が hash 非依存（`PYTHONHASHSEED` で結果が変わらない）／`--no-doc-demotion`（docs-first コーパス用） |
| 0.9.105 | **class フィールドを通じたメンバ呼び出しの解析**（`client.actions->Execute()`）／watch の除外ディレクトリ pruning（Linux） |

- ノード/エッジ構造が変わるため、更新後は `graphify update .`（LLM 不要）で再生成するのが望ましい。
- Graphify Cloud（`app.graphify.com`）は**使用していない**（`install` が広告を出すだけで、ローカル CLI/MCP のみ）。

## §5 AGENTS.md の訂正（実測との食い違い）

| 旧記述 | 実測 | 対応 |
|---|---|---|
| headroom v0.37.0 | 0.43.1 → **0.44.0** | 更新節を追加 |
| `.local\bin\headroom.exe` は破損（108,318 B・exit 1） | uv tool の exe と **バイト同一**（47,104 B・0.44.0） | 記述を訂正 |
| `mcp` extra の記述は graphify のみ | **headroom も `[mcp]` 必須**（v0.43.0 以降） | 追記 |
| graphify 0.9.69 | **0.9.106**＋スキル同期手順 | 更新 |
| headroom サブコマンド 31 | 0.44.0 で **30**（実測） | 訂正 |

## §6 残置状態・注意

- **proxy は未稼働のまま**（Owner 指示どおり使用しない）。ただしユーザー環境変数 `ANTHROPIC_BASE_URL=http://127.0.0.1:8787` は残っており、`headroom doctor` は「shell env は routed」と表示する。proxy を使わない運用なら、この環境変数は別クライアント（Claude/Codex 等）にとっては誤誘導になり得る（**今回は変更していない**）。
- `headroom` の MCP ツール（compress / retrieve）は proxy 前提。proxy を使わない本環境では `headroom_stats` のようなローカル統計のみ意味を持つ。
- MCP 子プロセスを止めると**セッションが切断される**（今回も 1 回切断し、再開後に作業を継続した）。

## §7 成果物

- 本報告: `doc/work113/20261111-2045_ツール更新/report.md`
- 証跡: `evidence/`（probe 出力、opencode.log 抜粋、headroom doctor、graph_stats、更新ログ）
- AGENTS.md: 「ツール更新 2026-11-11」節を追加＋既存記述の訂正
- Dropbox: `/ConvoPeq-tool-update-20261111/`
