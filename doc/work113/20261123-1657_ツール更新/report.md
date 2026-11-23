# OpenChamber Desktop でのツール適合性と、headroom・context-mode・graphify の更新

作業日: 2026-11-23（JST）。対象環境: OpenChamber Desktop（OpenCode CLI 2.3.2、Windows 11 x64）。コードは変更していない。

## 1. 結論

3 つとも **この環境でインストールして使用できる**。今回は 3 つとも最新版へ更新し、機能を実測で確認した。

| ツール | 更新前 | 更新後 | 判定 | 主な制約 |
| --- | --- | --- | --- | --- |
| headroom | 0.44.0 | **0.45.0** | 使用可（MCP 3 ツール） | proxy は使わない。MCP は手動 compress / retrieve のみ |
| context-mode | CLI 2.2.0 / プラグイン 2.2.1 | **2.2.2**（両方） | 使用可（`ctx_doctor` 全項目 PASS） | OpenCode のプラグイン版は設定ファイルの固定値を直さないと上がらない |
| graphify | 0.9.115 | **0.9.115**（最新のまま）+ スキル同期 | 使用可（MCP 11 ツール） | Cloud の無料枠は 1 リポジトリ 25,000 ノードまで。本リポジトリは 71,126 ノードで不可 |

## 2. 環境の理解（適合性の根拠）

- OpenChamber Desktop の背後では `opencode-cli.exe 2.3.2 serve --service` が動き、MCP サーバは stdio で起動される。GUI からの起動ではコンソールが点滅するため、ローカル MCP は `C:\Users\user\.local\bin\mcp\nowin.py`（`pythonw.exe` 経由、子を `CREATE_NO_WINDOW` で起動）を挟んでいる。今回の更新でもこの経路は変えていない。
- MCP の設定は 2 か所に分かれる。プロジェクト側 `C:\VSC_Project\ConvoPeq\opencode.json`（headroom / graphify など）と、グローバル側 `C:\Users\user\.config\opencode\opencode.json`（context-mode の MCP エントリと `plugins` の固定値）。**現行ビルドは `opencode.json` を実際に読む**（後述の `context-mode@2.2.2` 切り替えが即座に反映されたことで確認した）。
- ツールの実体は uv tool（headroom・graphify）と npm グローバル（context-mode）である。MCP の起動コマンドは絶対パスなので、バージョンを上げても設定は変えなくてよい。
- context-mode だけは実体が 2 つある。**npm グローバルの CLI / MCP サーバ**と、**OpenCode が `plugins` から読み込む in-process プラグイン**である。`ctx_doctor` の D6 は「in-process plugin: this call ran inside the agent」と返すため、実際にツールを提供しているのはプラグイン側で、MCP エントリは別系統として同時に接続している。

## 3. 更新の手順と結果

### 3.1 headroom 0.44.0 → 0.45.0

1. MCP 子プロセス（`nowin.py` → `headroom.exe mcp serve` → python）を停止した（exe ロックのため）。
2. `uv tool install "headroom-ai[mcp]==0.45.0" --force` を実行した。`[mcp]` を外すと `headroom mcp serve` が `MCP SDK not installed` で落ちるのは従来どおりである。
3. `headroom --version` は 0.45.0、`headroom mcp --help` も正常。

### 3.2 context-mode 2.2.0 / 2.2.1 → 2.2.2

1. `npm install -g context-mode@2.2.2` で CLI と MCP サーバを更新した。
2. `~/.config/opencode/package.json` の依存を `^2.1.0` から `^2.2.2` にし、`npm install` で `node_modules/context-mode` を 2.2.2 にした。
3. それでも `ctx_doctor` の D1 は「2.2.1、更新あり」のままだった。原因は `~/.config/opencode/opencode.json` の `plugins` に **`context-mode@2.2.1` と固定されていた**ことである。`context-mode@2.2.2` へ書き換え、プラグインファイルを触って再読込させると、ログに `loading plugin id=context-mode@2.2.2 entrypoint=.../context-mode@2.2.2/...` が出て、`ctx_doctor` の D1 が PASS（`2.2.2 (ae2e1bb): latest`）になった。
4. 元ファイルは `~/.config/opencode/opencode.json.bak-before-cm222` に退避した。

### 3.3 graphify 0.9.115（最新のまま）とスキル同期

1. PyPI の最新は 0.9.115 で、導入済みと一致していた。更新は不要。
2. スキルを 4 プラットフォームへ同期した（`graphify install --platform opencode|claude|codex|copilot`）。同期前は `graphify --version` がスキル不一致の警告を出していたが、同期後は `graphify 0.9.115` だけを返す。
3. グラフは `graphify-out/graph.json` のメタデータが `"graphify_version":"0.9.115"` で、71,126 ノード・90,063 エッジ・3,449 コミュニティ、`Built at commit 4d476d5c ... (matches HEAD)` だった。再構築は不要である。

## 4. 変更点の学習

### 4.1 headroom 0.44.0 → 0.45.0

- 目立つ追加は proxy / wrap 側である。保存量の出力トークン推定、tray アイコン（任意）、shaper の拡張、テレメトリの分割、`headroom wrap opencode --extra-model` の追加。
- **MCP の stdio 経路に影響する変更は無い**。`mcp` 関連の修正は Streamable HTTP トランスポートの Host / Origin 検査（#4852）で、stdio は対象外である。
- 圧縮側の修正（JSON ブロック走査の上限、呼び出し側が要求していないマーカーを足さない、CCR ハッシュの再起動跨ぎ保持）は、手動の `headroom_compress` / `headroom_retrieve` にも効く。今回の round-trip でも hash が返り、そのまま復元できた。
- 破壊的変更の記載は無い（リリースノートに BREAKING の節が無い）。

### 4.2 context-mode 2.0 系（まとめ）と 2.2.1 / 2.2.2

- 2.0 は全面書き換えで、ツールが 9 個（`ctx_execute` / `ctx_batch` / `ctx_index` / `ctx_search` / `ctx_resume` / `ctx_stats` / `ctx_doctor` / `ctx_upgrade` / `ctx_purge`）になり、**プロジェクトごとの SQLite（`context.db`）** と**全エージェント向けフック**を持つ構成になった。1.x の「MCP サーバだけ」という前提はもう古い。
- 2.2.1: Windows の `mkdir` EPERM を EEXIST と同じ扱いにする修正、`doctor` が 1.x のデータディレクトリを FAIL ではなく注記にする修正、`ctx_search` の source 指定の修正。
- 2.2.2: **OpenCode の skill ツール結果が欠けずにモデルへ届く修正**（#1404）が今回の環境に直接関係する。ほかに、存在しない cwd を最も近い既存の親へ解決する修正（#1409）、フックの書き込みロック待ち（#1401）、Fake-IP の SSRF 判定（#1403）。
- 更新の宛先はエージェントごとに違う。OpenCode は npm 経由（`opencode plugin update`）で、本環境では `plugins` の固定値も上げる必要があった（§3.2）。

### 4.3 graphify 0.9.112 → 0.9.115

- **0.9.115 はセキュリティ修正**（GHSA-g6c9-j4vm-mwq4）。Windows で、リポジトリ内の `.sh` の `source` 行や `.git` の `gitdir:` が UNC パスを指すと、索引時に外向き SMB 認証が発生し NetNTLMv2 応答が漏れ得る問題である。走査ルート外のパスを stat しないようにし、UNC を解決前に拒否する。**Windows 環境では更新の価値が高い**。
- AST キャッシュのスキーマが 50 → 51 に上がった。初回のビルドで該当言語を再抽出する。
- 0.9.112〜114 は Java のオーバーロード分解、Go のレシーバ型解決、Rust・Ruby・Lua・Scala・Elixir の同名衝突の解消、抽出・ビルドの性能改善（ID 正規化の ASCII ファストパス、キャッシュ書き込みの COW 化など）である。C++ 向けの変更は 0.9.111 までのものが中心で、今回の範囲では目立った挙動変更は無い。

## 5. 動作確認（実測）

| 項目 | 方法 | 結果 |
| --- | --- | --- |
| headroom MCP ハンドシェイク | `mcp-probe.py`（nowin 経由、OpenChamber と同じコマンド） | OK、3 ツール（`headroom_compress` / `headroom_retrieve` / `headroom_stats`） |
| headroom compress → retrieve | MCP で 17,607 トークンのログ様テキストを圧縮し、返った hash で復元 | 46 トークン（99.7% 削減）、復元 61,355 B が原文と一致 |
| headroom CLI | `headroom savings` / `headroom doctor` | savings は今日 17,561 トークン削減を計上。doctor は proxy 不達のみ fail（proxy は使わない方針どおり）、Claude / Codex の未経路は warn |
| context-mode CLI | `context-mode --version` / `context-mode doctor` | 2.2.2。opencode の行は PASS（parse / mcp / hook-registered / hook-runs / hook-seen / skills） |
| context-mode 実体 | `ctx_doctor` | 全 11 項目 PASS。D1 は 2.2.2、D9 は 4,090 ソース・103,655 チャンクを索引済み |
| graphify MCP | `graph_stats` / `query_graph` | 71,126 ノード・90,063 エッジ・3,449 コミュニティ、`matches HEAD`。クエリは BFS depth 2 で 10 ノードを返した |
| graphify CLI | `explain` / `diagnose multigraph` | explain は 12 ノードの曖昧一致を列挙（仕様どおり）。diagnose は同一端点の collapse 0 件 |
| graphify スキル | `--version` | 警告なしで `graphify 0.9.115` |

## 6. Graphify Cloud Free Plan の詳細

- Graphify Cloud は `https://app.graphify.com` のホスト型サービスで、**無料プランはカード不要**、**MCP で接続**する。ローカル CLI に認証コマンドは無く、`--backend` の選択肢（gemini / kimi / claude / openai / deepseek / ollama）にも Cloud は無い。つまり Cloud は「CLI の別バックエンド」ではなく、別系統のホスト型グラフである。
- **無料枠の上限は 1 リポジトリ 25,000 ノード**で、CLI のソースにも `_CLOUD_CTA_LARGE_GRAPH_NODES = 25_000` として埋まっている。25,000 以上では無料プランへ案内せず Pro の案内に切り替わる。本リポジトリは 71,126 ノードなので、**無料プランには収まらない**。
- このリポジトリでは既に **Graphify Labs の GitHub App**（Cloud 側の機能）が PR で `Graphify` と `Graphify Formal Verification` のチェックを出しており、無料枠の範囲で PR レビューに使われている。今回の PR でも両方 success / neutral で動作した。
- アカウント作成（メール認証やサインアップ）は本人の操作が要るため、こちらでは行っていない。ホスト型のグラフ MCP をこの環境へ足す場合は、**アカウントを作成して MCP の接続情報を得る**必要があり、かつ 25,000 ノードを超えるリポジトリでは Pro が要る。現状はローカル MCP（11 ツール）で足りている。

## 7. 見つけた問題と対処

1. **headroom の MCP は、更新のためにプロセスを止めると自動では戻らない。** 同じ手順で止めた context-mode は OpenCode が自動再生成した（`mcp connected server=context-mode tools=9`）が、プロジェクト側の設定で動く headroom は再生成されなかった。プロジェクト設定ファイルを同一内容で書き直しても戻らず、グローバル設定の再読込でも戻らなかった。**OpenChamber（または opencode サーバ）の再起動で戻る**。今回は再起動がこのセッション自体を切るため行わず、ハンドシェイクと round-trip で機能を確認した。
2. **context-mode の版数は 3 か所にある。** npm グローバル、`~/.config/opencode/node_modules`、`~/.config/opencode/opencode.json` の `plugins` 固定値である。実際に読み込まれるのは 3 番目で、ここを上げないと `ctx_doctor` の D1 が WARN のまま残る。次回更新時の手順として記録する。
3. **context-mode の MCP エントリに `enabled: true` がある。** 以前のビルド（2.0.16）ではこのキーでエントリ全体が破棄されたが、現行 2.3.2 では `mcp connected server=context-mode tools=9` が記録されており、接続している。互換のため触っていない。将来 MCP が繋がらない症状が出たら、まずこのキーを疑う。
4. **`context-mode doctor` は他のエージェントの行で FAIL する**（claude-code は 1.x の古いエントリ、codex / gemini / qwen / vscode は未設定）。これは本環境（OpenChamber）の不具合ではなく、単に他ツールへ未展開なだけである。opencode の行は PASS なので、終了コード 1 は無視してよい。
5. Graphify Cloud の無料枠は本リポジトリの規模を超える（§6）。

## 8. 未実施・制約

- headroom MCP の**セッション内再接続**は未確認（プロセスの自動再生成が無いため）。次回起動時に `mcp connected server=headroom tools=3` を確認する。
- context-mode の他エージェント（Claude Code / Codex / Gemini / Qwen / VS Code）への展開は未実施。
- Graphify Cloud のホスト型 MCP はアカウント作成が必要なため未接続。無料枠の 25,000 ノード上限も未検証（数値は CLI のソースとドキュメント記載）。
- 0.9.115 の AST スキーマ更新に伴う再抽出は、グラフが既に 0.9.115 で作られていたため発生していない。

## 9. 参照

- headroom: https://github.com/headroomlabs-ai/headroom （v0.45.0、2026-11-22）
- context-mode: https://github.com/mksglu/context-mode （v2.2.2、2026-11-22）
- graphify: https://github.com/Graphify-Labs/graphify （v0.9.115、2026-11-21、GHSA-g6c9-j4vm-mwq4）
- プローブ: `C:\Users\user\.local\bin\mcp\mcp-probe.py`（仕様は `%TEMP%\opencode\probe_tools_20261123.json`）
- round-trip: `%TEMP%\opencode\headroom_roundtrip.py`
