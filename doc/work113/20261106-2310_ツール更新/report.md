# headroom / context-mode / graphify / cocoindex code の更新（2026-11-06）

- 環境: Windows 11 x64、OpenChamber（フロントエンド）+ OpenCode V2（バックエンド）
- 対象: headroom、context-mode、graphify、cocoindex code
- 制約: Graphify Cloud の契約が必要な機能は使用しない（ローカル機能のみ）

## 1. バージョン

| ツール | 更新前 | 更新後 | 配布元 |
|---|---|---|---|
| headroom | 0.42.0 | **0.43.1** | PyPI `headroom-ai`（uv tool） |
| context-mode | 2.1.1 | **2.1.4** | npm `context-mode`（グローバル）+ OpenCode プラグイン |
| graphify | 0.9.102 | **0.9.103** | PyPI `graphifyy`（uv tool） |
| cocoindex code | 0.2.42 | **0.2.43** | PyPI `cocoindex-code`（uv tool） |

## 2. 更新手順

```
uv tool install 'headroom-ai[mcp]==0.43.1' --python 3.14 --force
uv tool install 'graphifyy[mcp]==0.9.103' --python 3.14 --force
uv tool install 'cocoindex-code[embeddings-local]==0.2.43' --python 3.14 --force
npm install -g context-mode@2.1.4
```

### 実行前に必要なこと

MCP サーバーのプロセスが `uv\tools\<name>\Scripts` を掴んでいるため、そのままでは
`failed to remove directory ... (os error 32/5)` で失敗する。次の順で行う。

1. 該当ツールの MCP プロセスを停止する（`headroom.exe`、`graphify-mcp.exe`、`ccc.exe`、`ccc run-daemon` とその python）。
2. 更新する。
3. OpenCode の API で MCP を再接続する（OpenCode サーバーを再起動しなくてよい）。

```
POST /api/experimental/mcp/{server}/disconnect?location[directory]=C:\VSC_Project\ConvoPeq
POST /api/experimental/mcp/{server}/connect?location[directory]=C:\VSC_Project\ConvoPeq
GET  /api/mcp?location[directory]=C:\VSC_Project\ConvoPeq
```

## 3. つまずいた点

| 事象 | 原因 | 対処 |
|---|---|---|
| `headroom mcp serve` が `ImportError: MCP SDK not installed` で即終了 | 0.43.0 で MCP サーバーが MCP SDK 2.x へ移行（破壊的変更）。`mcp` は任意依存 | `headroom-ai[mcp]` で再インストール |
| `ccc search` が `ModuleNotFoundError: No module named 'sentence_transformers'` | 0.2.43 のローカル埋め込みは `embeddings-local` エクストラ | `cocoindex-code[embeddings-local]` で再インストール |
| `uv tool install` が Scripts ディレクトリ削除で失敗 | MCP 本体と `ccc run-daemon` が実行中 | プロセス停止 → 更新 → API で再接続 |
| graphify がスキルの版ずれを警告 | 同梱スキルが 0.9.102 のまま | `graphify install --platform opencode` |

## 4. 動作確認

| 対象 | 確認方法 | 結果 |
|---|---|---|
| headroom | `headroom --version` / `headroom savings` / MCP ハンドシェイク | 0.43.1、3 ツール（compress / retrieve / stats） |
| context-mode | `context-mode --version` / `context-mode doctor` / MCP ハンドシェイク | 2.1.4、OpenCode 行が PASS（parse / list / mcp / hook-registered / hook-runs / hook-seen / skills）、9 ツール |
| graphify | `graphify --version` / `graphify query` / MCP ハンドシェイク | 0.9.103、ローカルグラフで検索成功（57,380 ノード）、10 ツール |
| cocoindex code | `ccc status` / `ccc search` / MCP ハンドシェイク | 0.2.43、105,597 チャンク / 2,212 ファイル、検索成功（exit 0）、1 ツール |

- OpenCode の `/api/mcp`（`C:\VSC_Project\ConvoPeq`）で headroom・graphify・cocoindex・context-mode がすべて `connected`。
- MCP ハンドシェイクは `mcp-probe.py` で initialize + tools/list まで確認。
- context-mode のプラグインは `context-mode@2.1.4`（OpenCode のキャッシュにも 2.1.4 が存在）。
- graphify のスキルを 0.9.103 に更新（`~/.config/opencode/skills/graphify`）。

## 5. 変更点（リリースノートから）

### headroom 0.43.0 / 0.43.1

- **破壊的変更**: MCP サーバーを MCP SDK 2.x へ移行。`mcp` パッケージが必須になった。
- ストリーミングの chat completions でも `headroom_retrieve` に応答する。
- Claude Code / Codex / Copilot / skills.sh 向けの headroom スキルを同梱。
- `headroom savings` がプロキシのモデルルーティング分も表示する。
- 修正: CCR の順序キューが再格納エントリを追い出さない、メモリの relation キー、HNSW の再インデックス、OpenClaw の 12 文字ハッシュ、ONNX Runtime の暗黙探索回避、プロキシのキャッシュ上限、ChatGPT ストリーミングの HTTP/1.1 化、ルーターのタグ付きブロック。

### context-mode 2.1.2 → 2.1.4

- 2.1.2: ctx ツールを持たないサブエージェントが WebFetch/curl を実行できる。
- 2.1.3: `offset`/`limit` 付きの Read がポインタ化されず、そのままモデルへ届く。
- 2.1.4: セキュリティ強化（deny ルールを ctx_execute の全シェル言語と ctx_batch に適用、私有アドレス帯の Fetch 遮断、リダイレクト各段の検査）。`ctx_upgrade` が v1 データを保持。OpenCode 2（Bun）ホスト向けに doctor フック・バックグラウンド起動・読み取り専用 DB・D3 エントリ・読み取りサイズガードを修正。

### graphify 0.9.103

- 抽出器の修正: C# のプリプロセッサ、Solidity のライブラリ呼び出し、YAML のマージキー、Markdown の名前空間参照、Python の組み込み名呼び出し。
- 解決: トップレベル `_main` の呼び出し帰属。
- フック: 内容が変わらないファイルを mtime で stale にしない。
- 性能: 読み込み・解決・抽出のピークメモリ削減、監視の再解決をファイル単位に。
- `graphify update` が内容の変わったファイルだけを再抽出。AST キャッシュのスキーマは 39 に更新。

### cocoindex code 0.2.43

- チャンキング設定やインデックスパラメータが変わったとき、未変更ファイルも再処理・再埋め込みする。
- デーモンのソケット終了処理を安全化（置き換えられたデーモンが新しいソケットを消さない）。
- ユーザー設定ディレクトリをプロジェクトとして自動初期化しない。
- 公式 MCP レジストリ向けの `server.json` とリリースジョブを追加。

## 6. 注意

- headroom の proxy は停止したまま（`headroom savings` の Today は 0）。MCP と CLI のみ使用する方針を維持。
- graphify はローカルグラフのみ使用。Cloud の契約が必要な機能は使っていない。
- graphify の AST キャッシュスキーマが上がったため、次回 `graphify update .` を実行すると未更新分の再抽出が走る。既存の `graph.json` は読み込み・検索とも動作確認済み。
- 実行中の OpenCode サーバーには再接続済み。次回のターンから新しいツール一覧が反映される。
