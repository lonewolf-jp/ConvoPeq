# graphify を 0.9.115 へ更新し、OpenChamber Desktop での利用可否を確認（2026-11-22 10:26 JST）

この環境（OpenChamber Desktop、Windows 11 x64）に入っている graphify を 0.9.111 から 0.9.115 へ更新し、変更点を確認して動作を見た記録である。結論を先に書く。更新は成功し、CLI と MCP の両方が 0.9.115 で動く。OpenChamber への組み込み条件も満たしている。Graphify Cloud は使っていない。

## 1. 環境

- OS: Windows 11 Home、ビルド 26300
- CPU: 11th Gen Intel Core i7-11700K
- シェル: PowerShell 7（OpenChamber のターミナル）
- uv: 0.10.10（`C:\Users\user\.local\bin\uv.exe`）
- Python: graphify のツール環境は 3.14.7（`uv\python\cpython-3.14-windows-x86_64-none`）
- 更新前: graphify 0.9.111
- 更新後: graphify 0.9.115

## 2. 更新手順（実測）

1. graphify の MCP 子プロセスを止めた。実行中の `graphify-mcp.exe` と、それを包む `nowin.py` の python が 8 個あった。exe を掴んだままだと uv が旧ツールを消せず、`os error 32` で失敗する。
2. `uv tool install "graphifyy[mcp]==0.9.115" --force` を実行した。`[mcp]` を付けないと `graphify-mcp` が入らない。
3. スキルを 4 環境へ同期した。`graphify install --platform opencode|claude|codex|copilot`。同期前は `graphify --version` が 4 件の警告を出していた。
4. バージョンを確認した。ツール環境の Python は 3.14.7 のままである。

| 項目 | 値 |
| --- | --- |
| `graphify --version` | 0.9.115 |
| `uv tool list` | graphifyy v0.9.115（graphify、graphify-mcp） |
| ツール環境の Python | 3.14.7（要求は 3.11.5 以上 3.15 未満） |
| 実行ファイル | `C:\Users\user\AppData\Roaming\uv\tools\graphifyy\Scripts\graphify.exe`、`graphify-mcp.exe` |

## 3. 変更点（0.9.112〜0.9.115）

### 3.1 セキュリティ（0.9.115）

- リポジトリを索引するとき、コーパスが指定したパスを無制限に `stat` していた。`.sh` の `source` 行や `.git` の `gitdir:` / `commondir` に UNC パスを書くと、Windows で外向きの SMB 認証が発生し、NetNTLMv2 応答が漏れる可能性があった（GHSA-g6c9-j4vm-mwq4）。
- 0.9.115 は探索をスキャンルート内に制限し、UNC パスを解決前に拒否する。信用できないリポジトリを索引する場面では重要である。

### 3.2 抽出と解決（C++ に関係するもの）

- `std::` 修飾の型（`std::chrono::seconds`、`std::vector<T>`）を、同名のプロジェクト内シンボルへ誤って結び直さないようにした。std 側はソースを持たないスタブになり、テンプレート引数はプロジェクトの型へ解決される（0.9.114）。
- 到達不能な前方宣言が、メンバの宣言と out-of-line 定義の折り畳みを妨げないようにした（0.9.113）。
- AST キャッシュがソースと別ドライブにある場合に include の同一性が空になっていた問題を直した（Windows の別ドライブ構成、0.9.113）。
- `path` は完全一致のノード ID を先に解決する（0.9.115）。`serve` は `call` 関係で絞り込んでもクロスリポジトリの `calls` を落とさない（0.9.114）。

### 3.3 性能（0.9.114、出力は同一）

- 識別子の正規化に ASCII の速路と LRU を入れ、約 1.26 倍。
- キャッシュ書込みの変換を copy-on-write にして約 20% 短縮。
- ファイルを 1 回だけ読み、言語ごとにパーサを再利用する。
- クロスファイル解決の最終段を数回の走査に統合し、約 15% 短縮。
- コミュニティ分割をトポロジのハッシュでキャッシュし、ヒット時は約 9 倍。

### 3.4 パッケージング

- 0.9.113 で最小 Python が 3.11.5 になった（soupsieve のアドバイザリ回避）。3.11.0〜3.11.4 は 0.9.112 に留まる。
- AST キャッシュのスキーマが 50 から 51 に上がった。更新後に最初のビルドで、対象言語のファイルが再抽出される。

## 4. 動作確認

### 4.1 MCP

`mcp-probe.py` でハンドシェイクを打った。

- serverInfo: `graphify` 0.9.115
- tools 11 件: `query_graph`、`get_node`、`get_section`、`get_neighbors`、`get_community`、`god_nodes`、`graph_stats`、`shortest_path`、`list_prs`、`get_pr_impact`、`triage_prs`

続けて `tools/call` で `graph_stats` を実行した。結果は次のとおりで、現在の HEAD と一致する。

```
Nodes: 71126
Edges: 90063
Communities: 3449
EXTRACTED: 89%
INFERRED: 11%
Built at commit: 4d476d5c65261fd975de2606f4d8b5c843cd86e1 (matches HEAD)
```

### 4.2 CLI

- `graphify query "SimplePeakLimiter のソフトニー" --budget 400`: 5.7 秒、出発点 `SimplePeakLimiter`、78 ノード。
- `graphify god-nodes --top 5`: 1.8 秒。1 位は Intel oneAPI の PDF ノード（6,615 エッジ）、2 位 `ConvolverProcessor`（497）、3 位 `AudioEngine`（437）。
- `graphify query "C-24 の再測定で使ったハーネス" --budget 300`: 7.3 秒、442 ノード。

### 4.3 グラフの更新

`graphify update .` を実行した。LLM は使っていない。

- 走査 2,903 ファイルのうち、前回から変わった 2 ファイルだけを再抽出した。
- 結果は 71,126 ノード、90,063 エッジ、3,449 コミュニティ。`graph.json`、`graph.html`、`GRAPH_REPORT.md` が更新された。
- 所要は約 5 分。`graph.json` は 70.6 MB。
- 警告が 2 件出た。ラベルは 2,189 件が保存済みで、コミュニティは 3,449 件に増えたため 343 件をハブ名で改名した（`graphify label` で LLM 再命名が必要）。コーパスから外れた 1 ファイル 8 ノードは、ディスク上に残り除外規則にも合わないため保持した。

## 5. OpenChamber Desktop での利用可否

### 5.1 インストール

- `uv tool` でユーザー領域に入る。管理者権限は要らない。
- 要求は Python 3.11.5 以上 3.15 未満。この環境の uv は 3.14.7 を使う。
- MCP を使うなら `[mcp]` extra が必須。付け忘れると `graphify-mcp` が入らない。
- ネットワークは更新時と、`clone` や `prs` を明示的に使うときだけである。索引そのものはローカルで完結する。

### 5.2 MCP サーバ

プロジェクト設定 `C:\VSC_Project\ConvoPeq\opencode.json` の `mcp.servers.graphify` は次の形で、更新後もそのまま使える。

```
C:\Python315\pythonw.exe
C:\Users\user\.local\bin\mcp\nowin.py
C:/Users/user/AppData/Roaming/uv/tools/graphifyy/Scripts/graphify-mcp.exe
--transport stdio --graph C:/VSC_Project/ConvoPeq/graphify-out/graph.json
```

- `nowin.py` は `pythonw.exe` 経由で子を `CREATE_NO_WINDOW` 付きで起動する。graphify の MCP を起動するたびにコンソールが点滅するのを防ぐ。この環境では既に全ローカル MCP に適用している。
- graph のパスは絶対で書く。`cwd` は OpenChamber のこのビルドでは使えない。
- 実際の接続は、OpenChamber のサーバーログに `mcp connected server=graphify tools=11` として出る。更新前の 0.9.111 では 01:14 にこの行を確認した。0.9.115 は同じコマンドで probe が通ることを確認している。更新のため MCP を止めたので、このセッションの tools は opencode サーバの再起動で戻る。

### 5.3 プラグイン

`graphify install --platform opencode` は `~/.config/opencode/plugins/graphify.js` を書く。中身は `export default { id, async setup(ctx) { ctx.tool.hook("execute.before", ...) } }` で、OpenChamber（OpenCode 2）が要求する契約に合っている。

- セッション最初の shell 実行に、グラフの場所と `graphify query` の案内を前置する。このセッションでも実際に前置を観測した。
- 同じ id の二重ロードを避けるため、id にファイル位置のハッシュを付けている。プロジェクトと上位ディレクトリの両方に置いても衝突しない。
- リマインダ文字列にバッククォートや `$()` を入れない制約がある。ソースにもその注意が書かれている。

### 5.4 Cloud を使わないこと

- 索引、クエリ、MCP はローカルで動く。`graphify install` の完了メッセージは Graphify Cloud の案内を出すが、登録も送信もしない。
- コミュニティ名の LLM 命名（`graphify label`）だけはバックエンドが要る。今回は実行していない。ラベルが古いままでもクエリは動く。

### 5.5 この環境での制約

- 更新前に MCP 子プロセスを止める必要がある（exe ロック）。止めるとそのセッションの graphify ツールは消え、opencode サーバの再起動で戻る。
- 大きなリポジトリでは `graph.json` が 70 MB 程度になる。`graph.html` は 5,000 ノード超で集約表示になる。
- 索引は自動では更新されない。`graphify update .` か watch を使う。古い索引は無警告で返る。
- スキーマ更新（50→51）の後は、変更ファイルの再抽出が走る。全件再抽出ではない。
- クエリは BFS で、既定の budget では切り捨てられる。`--budget` を上げるか、`context_filter=['call']` で絞る。

## 6. 結論

- graphify 0.9.115 は、この環境に追加の常駐サービスも管理者権限もなしで入り、CLI と MCP の両方で動く。
- OpenChamber へは、プロジェクト設定の MCP 1 エントリと、プラグイン 1 ファイルで組み込める。どちらも既存の設定のままでよい。
- 0.9.115 の価値は、Windows の UNC/SMB 脆弱性の修正と、C++ の解決精度の改善である。性能改善は出力を変えない範囲で入っている。
- Cloud 機能は使っていない。

## 7. 参照

- リポジトリ: https://github.com/Graphify-Labs/graphify （旧 `safishamsi/graphify` はここへ転送される）
- リリース: v0.9.112〜v0.9.115
- セキュリティ: GHSA-g6c9-j4vm-mwq4
- 設定: `opencode.json` の `mcp.servers.graphify`
- プラグイン: `C:\Users\user\.config\opencode\plugins\graphify.js`
- グラフ: `graphify-out/graph.json`（更新後 71,126 ノード、90,063 エッジ）
