# context-mode / AiDex の更新と動作確認

- 実施: 2026-11-09 19:00
- 環境: OpenChamber（フロントエンド）+ OpenCode 2.2.6（バックエンド）、Windows 11 x64
- 対象: context-mode 2.1.6 → **2.1.7**、aidex-mcp 2.4.0 → **2.4.1**
- 設定変更: なし（`opencode.json` の MCP エントリとプラグイン指定は不変）

## §1 更新手順

MCP サーバーを止めずに更新すると、実行中の子プロセスが旧コードのまま残る。OpenCode サーバーの再起動はセッションを中断するため、**MCP の disconnect → npm 更新 → connect** で反映した。

```
# 1. 切り離し（OpenChamber の API 経由。opencode サーバーは 127.0.0.1:55696、UI は 63621）
curl -X POST -H 'Origin: openchamber-ui://app' \
  'http://127.0.0.1:63621/api/experimental/mcp/context-mode/disconnect?location%5Bdirectory%5D=C:%5CVSC_Project%5CConvoPeq'
curl -X POST ... /api/experimental/mcp/aidex/disconnect?...
# 2. 更新
npm install -g context-mode@2.1.7 aidex-mcp@2.4.1
# 3. 再接続
curl -X POST ... /api/experimental/mcp/context-mode/connect?...
curl -X POST ... /api/experimental/mcp/aidex/connect?...
```

- disconnect/connect は 204 を返し、子プロセス（node）も終了・再生成された
- `GET /api/mcp?location[directory]=...` で状態を確認（19 サーバーすべて connected）

## §2 動作確認

| 項目 | 結果 |
|---|---|
| context-mode パッケージ版 | 2.1.7（`package.json`） |
| `ctx_doctor` | **全項目 PASS**（D1 version / D2 node / D3 host config / D4 old entries / D5 hook round trip / D5b double-fire / D6 MCP / D7 project / D8 context.db / D9 index 608 sources・1,652 chunks / D10 enforcement / D11 shell） |
| `context-mode doctor --agents opencode`（CLI） | **PASS**（parse / list / mcp / hook-registered / hook-runs / hook-seen / skills） |
| `ctx_stats` / `ctx_execute` | 正常（`ctx-execute-ok`、2.1.7 を返す） |
| プラグイン | `context-mode@2.1.7` をキャッシュから読み込み（`~/.cache/opencode/npm/context-mode@2.1.7/.../opencode/plugin.js`） |
| AiDex サーバー版 | initialize の serverInfo = **2.4.1** |
| `aidex_status` | 正常（ConvoPeq: 516 files / 22,876 items / 39 MB、schema 1.2） |
| `aidex_search` | 正常（hybrid、`SetImpulse` で 20 件） |
| 他の MCP | 19/19 connected（npm 更新の影響なし） |
| グローバル npm | context-mode 2.1.7 / aidex-mcp 2.4.1、他パッケージの版は不変 |

- CLI doctor の claude-code / codex / gemini / qwen / copilot-cli / vscode / kiro / zcode の FAIL は、この環境でそれらを設定していないためで、従来と同じ（opencode 行が PASS ならよい）

## §3 変更点（学習）

### context-mode 2.1.7（2026-11-08）

- **doctor の修正**: D1（更新コマンドがホストごとに正しい）、D3（OpenCode の全設定と tuple 形式のエントリを読む）、D6（in-process プラグインで PASS）、未対応ホストの表示を一本化。ターミナルの環境変数で OpenCode / Pi が「未対応」と誤判定されなくなった
- **`ctx_execute` の堅牢化**: 終了時の回収とパイプの有界ドレイン、`killTree` がホストのイベントループを塞がない（#1337）。大量出力やタイムアウト時の停止が改善
- `policy.json` の `routes.webfetch` と `allowHosts` で WebFetch 経路を無効化できる（#1336）
- `setup` は古い Claude Code プラグインをホスト自身の更新で更新し、失敗時はロールバックではなく WARN で終わる（#1338）
- OpenCode の更新チャネルは npm（`opencode plugin update`）。本環境は npm パッケージ + 固定版プラグインの両方で 2.1.7 になった

### context-mode 2.1.6（前回 2.1.4 から飛ばした分の主なもの）

- OpenCode の `ctx_*` ツールが OpenCode の permission ルールを通る（#1331）
- エージェント設定ディレクトリ配下の skill / rule ファイルがモデルへ全文届く。Read の拒否メッセージが実際に使える経路を指す（#1326）
- post-compact の recall が「最初のプロンプト」ではなく現在のプロンプトを検索する（#1330）
- setup が Node を最初に確認し、古い場合に理由を明示（#1328 / #1329）

### AiDex 2.4.1（2026-11-08・セキュリティリリース）

- **`aidex_screenshot` のコマンドインジェクションを修正**（CWE-78、High、GHSA-xwqr-wrm9-h3jx）
  - Windows: `window_title` が PowerShell 文字列内で `$(...)` を評価し得た。スクリプトを静的化し、値は環境変数で渡す方式に変更
  - macOS: `filename` / `save_path` がシェル文字列に未エスケープで入っていた。`execFileSync` + 引数配列へ変更
  - `filename` は `.png/.jpg/.jpeg` の単純名、`save_path` は既存ディレクトリのみ、数値引数は実行時型検査
- `aidex_global_init` が C/C++ プロジェクト内のネストした CMake/Make/Meson サブビルド（ggml 等）を個別プロジェクトとして登録しなくなった

### AiDex 2.4.0（既に入っていた分の要点）

- Dashboard の control をポーリングではなく **push** で受け取れる（`POST /control/subscribe`、順序保証・失敗 3 回で解除・ローカルネットワーク限定）

## §4 注意点・残課題

- npm 更新時に「changed 278 packages」と表示された（グローバルツリーの再解決）。実行後、19 の MCP はすべて connected、主要パッケージの版は不変であることを確認した
- 旧プラグインのキャッシュが 2.1.0〜2.1.7 まで残っている（各約 6.2 MB、合計約 43 MB）。動作に影響はない。不要なら `~/.cache/opencode/npm/context-mode@2.1.x` を削除できる
- `opencode.json` の `mcp.context-mode` には `enabled: true` が付いたまま。このビルドでは有効に解釈され、実際に接続されている（触っていない）
- context-mode の `serverInfo.version` は `dev` を返す実装のため、版の判定は `package.json` と `ctx_doctor` の D1 で行う
