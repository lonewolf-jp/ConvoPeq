# OpenChamber の復旧（2026-11-07 15:45 更新）

## 結論

OpenChamber の設定ファイルは失われていない。画面が出なかった原因は、OpenCode 2.2.6 のクロスサイト防御で UI の API 呼び出しがすべて 403 になっていたことだった。UI を `http://127.0.0.1:63621` から読む設定に切り替えて復旧し、利用者環境に恒久設定として保存した。

## 1. 症状と原因

- 症状: 画面に「起動に失敗しました」「サーバーは実行中ですが、OpenCode エージェントを読み込めませんでした」と出て、最後のエラーが `agent.list failed (403): unexpected status`。セッション・プロバイダー・MCP・プラグイン・エージェントの各パネルが空になる。
- 原因: OpenCode 2.2.6 で API のクロスサイト防御が入り、許可リストにない Origin からの `/api/*` を 403 で返す。同梱 UI の Origin は `openchamber-ui://app` で許可リストに無いため、全 API が拒否されていた。
- 当該の不具合は OpenChamber の issue #6503 / #6505 として起票されている（OpenCode 2.2.6 の回帰）。
- ログの証拠: `~/.local/share/opencode/log/opencode.log` に 403 が毎分 250 件前後。修正後に画面のエラーは消え、以降の 403 は観測していない（ログの最終更新は 06:37Z）。

## 2. 復旧措置

1. 利用者環境変数 `OPENCHAMBER_ELECTRON_LOAD_SERVER_UI=1` を設定した。これで同梱 UI ではなく `http://127.0.0.1:63621` から UI を読む。Origin が許可リスト内になるため 403 が解消する。
2. OpenChamber を起動し直し、画面を確認した。Document の URL は `http://127.0.0.1:63621/`、プロジェクト・セッション・ゴール・モデル選択（commandcode / DeepSeek V4.1 Flash）が表示され、エラー表示は無い。
3. 設定 4 点を `~/.config/openchamber/backup-20261107-1502/` に退避した。

## 3. 設定の棚卸し

| 対象 | 状態 | 内容 |
|---|---|---|
| `~/.config/opencode/opencode.jsonc` | 正常（44,985 B） | provider・policies・MCP 7 件 |
| `~/.config/opencode/opencode.json` | 正常（466 B） | context-mode 2.1.5 のみ |
| `C:\VSC_Project\ConvoPeq\opencode.json` | 正常（4,039 B） | プロジェクト側 MCP 11 件 |
| `~/.config/openchamber/settings.json` | 正常（101,263 B） | キー 101 個（11/06 の控えと増減なし） |
| `~/.config/openchamber/projects/` | 正常（5 件） | 各ファイル解析成功 |

## 4. `.openchamber` が空である件

- `C:\VSC_Project\ConvoPeq\.openchamber` は空だが、フォルダの更新日時は **11/02 11:39** のまま。今日ファイルが消された形跡はない（フォルダの更新日時は中身の増減で変わる）。
- ここに入るのは共有設定 `project.json` で、内容は worktree のセットアップ/終了コマンド、プロジェクトアクション、下書きスターター、`plansDir`、マジックプロンプト。セッション・プロバイダー・MCP・プラグイン・エージェントの読み込みには使われない。
- ファイルが無い状態はアプリ側で許容される（「missing」として空設定で動く）。今回の 5 つの読み込み失敗とは無関係。
- 過去の内容は Git 管理外で、ローカル控え・ごみ箱・シャドウコピー・Dropbox のリポジトリ控えのいずれにも見つからず、復元できる材料は残っていない。

## 5. 残る確認と注意

- この回避策は環境変数による。OpenChamber 側が Origin を付け替える修正を出したら、変数を外して良い。
- OpenCode を 2.2.6 より前へ戻す回避策もあるが、セッション DB の互換が崩れるため採らなかった。
- `settings.json` に `desktopLocalClientToken` が無い。同梱 UI を直接使う経路では認証に必要だが、今回の構成（63621 経由）では不要。将来この値が必要になったら、アプリが起動時に再発行する。
- OpenCode Desktop V2 と OpenChamber の同時起動は、OpenCode サービスが二重になる。常用はどちらか一方が安全。

## 6. 実行した確認

- UI Automation でウィンドウ内の要素と URL を取得（`http://127.0.0.1:63621/`、エラー文言なし）
- `curl http://127.0.0.1:63621/api/session`（38 KB）、`/api/agent`（7 件）、`/api/plugin`（98 件）、`/api/config/mcp`（8 件）
- `opencode mcp list`（19 サーバー connected）、`opencode models`（632 件、commandcode を含む）
- `opencode.log` の 403 件数の推移（毎分数百件 → 毎分 3 件 → 停止）
