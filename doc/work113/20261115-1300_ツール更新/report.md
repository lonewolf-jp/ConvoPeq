# context-mode と graphify の更新と動作確認

- 実施日時: 2026-11-15 12:55〜13:25 JST
- 環境: OpenCode V2（バックエンド）＋ OpenChamber（フロントエンド）、Windows 11 x64
- 目的: 2 つのツールを最新版へ更新し、動作を確認し、変更点を整理する

---

## 1. context-mode 2.1.7 → 2.2.0

### 1.1 更新

```
npm install -g context-mode@2.2.0
```

`npm view context-mode dist-tags` の `latest` は 2.2.0。npm のグローバル更新のみで完了した（実行中の MCP プロセスは旧コードを保持したままで、ファイル更新は成功）。

### 1.2 動作確認

| 項目 | 結果 |
| --- | --- |
| `context-mode --version` | 2.2.0 |
| `npm list -g --depth=0` | `context-mode@2.2.0` |
| `context-mode doctor` の opencode 行 | **PASS**（parse / list / mcp / hook-registered / hook-runs / hook-seen / skills） |
| MCP ハンドシェイク（別プロセスで起動） | OK、9 tools（ctx_execute / ctx_batch / ctx_index / ctx_search / ctx_resume / ctx_stats / ctx_doctor / ctx_upgrade / ctx_purge） |

`doctor` の全体終了コードは 1 だが、これは claude-code / codex / gemini / qwen / vscode / kiro が未設定で FAIL するため。この環境で使うのは OpenCode だけで、その行は PASS。

### 1.3 変更点（2.1.8〜2.2.0）

**2.1.8**
- OpenCode 2 の subagent ツール集合と in-process replace（#1347, #1348）
- OpenCode の project `.opencode/plugins` ファイルの扱いを修正（#1349）。グローバル入口の判定、重複ロード時の維持、削除すべきファイル名の提示
- `ctx_search` が空の source を未指定として扱い、`offset:0` / `after:0` を受け付ける（#1353, #1354）
- Copilot CLI 検出、Pi の tool_result、`search.mjs` のクォート修正
- 性能: 自ディレクトリ配下のパス解決を省略

**2.1.9**
- Windows の PATH / PATHEXT を大文字小文字を区別せず読む（#1371）
- Node 24.15 未満でも 8 つのサーバーホストで native tool を許可（#1370）
- `hostBin` がチェックを通った PATH 上のファイルを実行（#1335）
- **OpenCode 2 の generate 要求に、context と同じ system payload を渡す（#1369）**
- twin テストの Windows 向け一時ファイル後始末

**2.1.10**
- `doctor` の env 行が `ctx_execute` 用の PATH node とそのバージョンを表示（#1350）
- D5 の境界を node の起動時間に比例させ、in-process ホストは形だけ検査

**2.2.0**
- **OMP（oh-my-pi）を 11 番目の対応エージェントとして追加**（ADR-0026, #1332）。M1〜M8 の一連の実装
- **`tools: mcp__<server>__*` の指定で、そのサーバーの ctx ツールを subagent に許可（#1374）**
- copilot-cli / vscode の hook コマンドを Windows で bare node 始まりにし、`pwsh -Command` が解釈できるようにする（#1375）
- `safeFetch` が全リクエストに user-agent と accept を付け、拒否時にホスト名を出す（#1380）
- sandbox の PATH 候補で stat できないものを飛ばし、解決失敗を明示（#1373）
- 性能: `doctor` が omp のプラグイン一覧を 1 回だけ読む

この環境への影響は OpenCode 2 向けの修正（#1347, #1348, #1369, #1374）が中心。MCP サーバーとして使う形態は変わらない。

---

## 2. graphify 0.9.106 → 0.9.109

### 2.1 更新

```
uv tool install "graphifyy[mcp]==0.9.109" --force
```

`[mcp]` エクストラは必須（無いと `graphify-mcp` が `ImportError: mcp not installed` で起動しない）。

更新前に graphify の MCP プロセス 3 個（`nowin.py` ラッパー 1、`graphify-mcp` 2）を停止した。Windows では実行中の exe がロックされ、置き換えに失敗するため。

スキルの同期:

```
graphify install --platform opencode
graphify install --platform claude
graphify install --platform codex
graphify install --platform copilot
graphify install --project --platform claude
```

### 2.2 動作確認

| 項目 | 結果 |
| --- | --- |
| `graphify --version` | 0.9.109（警告なし） |
| スキルの `.graphify_version` | opencode / claude / codex / copilot / プロジェクト `.claude` すべて 0.9.109 |
| MCP ハンドシェイク | OK、10 tools（query_graph / get_node / get_neighbors / get_community / god_nodes / graph_stats / shortest_path / list_prs / get_pr_impact / triage_prs） |
| `graphify explain "AudioEngine"` | 候補 12 件を列挙（exit 3 = 曖昧。仕様どおり） |
| `graphify god-nodes --top 5` | 取得できた（ConvolverProcessor 497 edges、AudioEngine 437 edges など） |
| `graphify query "audio callback counter"` | 1,163 ノードを探索し、`audioCallbackEpochCounter`（`src/audioengine/AudioEngine.h` L1905）などを返した |
| `graphify update .` | 71,013 nodes / 89,935 edges / 3,449 communities。graph.json 70.4 MB、graph.html と GRAPH_REPORT.md も更新 |

`graphify update .` は LLM を使わないコード専用の再抽出。AST キャッシュの schema が上がったため、変更のあった 226 ファイルのうち 195 を再抽出した。

### 2.3 変更点（0.9.107〜0.9.109）

**0.9.107**
- C++: 折り畳んだ引数なしメンバ宣言（`bar()`）の `definition_file` を、`source_file` と同じ root で記録するよう修正
- 名前が前後の `_` だけ違う宣言を別ノードに分離（C / C++ / JS / TS / PHP / Ruby / Kotlin / Swift / Lua）
- 性能: 読み取り系コマンドが全言語の抽出器を import しない。AST ワーカー数を未キャッシュ量に比例させる
- クラウド: 署名なしの無料プランへの案内に変更。post-build の案内は TTY のみで、`GRAPHIFY_NO_TIPS` / `GRAPHIFY_NO_CTA` を尊重

**0.9.108**
- **MCP serve: グラフ再読み込みを別スレッドで作り、原子的に差し替える**（#5177）。再読み込み中もリクエストが止まらない
- `graphify path` が、同じ長さ・確度なら実ファイル経由の経路を優先（#5153）
- hook の検索ナッジが、グラフが stale かもしれないときに弱まる（#5132）
- C# / Java / SQL の呼び出し解決の修正

**0.9.109**
- C++: コンストラクタ `Foo` とデストラクタ `~Foo`、末尾 `_` の名前を別 id に分離（#5202）
- **ファイル単位の AST 失敗で、前回の良い AST 記録を保持し、その実行を incomplete として次の実行で再試行**（#5191）
- 可変長引数が要素型への型エッジを持つ（Go / Java / C# / Scala / PHP）（#4638）
- シンボリックリンクの走査 root で `source_file` がベース名に潰れる問題を修正（#4624）
- LLM: チャンクのコールバック例外で残りをキャンセル（#5187）、複数断片の応答を拒否して再試行（#5192）、スキルのカバレッジ記録（#2225）
- AST キャッシュ schema は 43 → 46。C++ を含む抽出が再実行される（今回 `update` で反映済み）

Graphify Cloud の契約が必要な機能は使っていない。ローカルの `graph.json` を読む経路のみ。`graphify label`（LLM でコミュニティ名を付ける）は今回実行していない。

---

## 3. 注意

- このセッションの MCP サーバープロセスは更新前のコードを保持している。context-mode は 2.1.7 のまま動いており、graphify は停止したため未再接続（このセッションの graphify ツール名前空間は消えた）。OpenCode を再起動すると新しい版で再接続するが、再起動はセッションを切断するため今回は行っていない。
- `graphify update` で 2 ファイルが構文エラー扱いになった（`src/DeviceSettings.cpp` の 126 行付近、`src/audioengine/AudioEngine.h` の 2209 行付近）。tree-sitter の抽出限界で、ビルドや実行には影響しない。
- コミュニティ名は 448 件がハブ名に置き換わった。LLM で付け直すには `graphify label` を使う（今回は未実行）。
