# context-mode 2.1.1 への更新と動作確認

- 実施: 2026-11-05 23:45–23:58（Windows 11 x64、OpenCode 2.2.5、Node 24.22.2）
- 結論: 2.1.0 から 2.1.1 へ更新した。CLI・プラグイン・MCP の 3 つがそろって 2.1.1 になり、`ctx_*` 9 ツールとフックが動作することを確認した。更新の途中で設定ファイルに BOM が付き opencode が読めなくなったため、BOM を除去して復旧した。

## 1. 変更点（2.1.0 → 2.1.1）

リリースノートの記載は 1 項目である。

> fix(claude-code): keep host replace fields, true Read line numbers, one skills dir, no Gemini hooks in the plugin

| 変更 | 内容 | OpenCode への影響 |
|---|---|---|
| host replace fields の保持 | ホストが返す置換フィールドを保つ | なし（Claude Code 向け） |
| Read の行番号 | 置換後の Read で実際の行番号を返す | なし（Claude Code 向け） |
| skills ディレクトリの一本化 | スキルの配置先を 1 つに統一 | あり。`~/.config/opencode/skills/` の 3 スキルがそのまま有効 |
| Gemini フックの除去 | プラグインから Gemini 用フックを外す | なし |

- 2.1.1 は 2026-11-05T13:10Z 公開。OpenCode 向けの機能追加は無く、不具合修正と配布物の整理が中心である。

## 2. 実施した手順

1. グローバル CLI を `npm i -g context-mode@2.1.1` で更新した。
2. `opencode plugin add context-mode@2.1.1` が `Invalid global configuration` で失敗した。原因は `~/.config/opencode/opencode.json` の先頭に付いた UTF-8 BOM で、opencode が設定を読めなくなっていた。
3. BOM を除去し、`plugin` を `context-mode@2.1.1` に書き換えた。
4. `context-mode setup --agents opencode --yes` を再適用し、MCP エントリとスキルを更新した。適用後に PASS を確認。
5. 設定ファイルを再確認し、BOM なし・MCP 定義 1 件・プラグイン 2.1.1 であることを確認した。

## 3. 動作確認

| 確認 | 条件 | 結果 |
|---|---|---|
| CLI の版 | `context-mode --version` | 2.1.1 |
| プラグインの版 | `opencode plugin list --builtin` | 2.1.1（`context-mode@2.1.1`） |
| MCP の版 | `ctx_doctor` の Environment 行 | 2.1.1（`d232ce4`） |
| 導入点検 | `context-mode doctor --agents opencode` | PASS（parse / list / mcp / hook-registered / hook-runs / hook-seen / skills） |
| 実行 | `ctx_execute` で 300 件を集計 | 33 B の答えだけ返却（`{"n":300,"sum":134550,"max":897}`） |
| 索引と検索 | 点検レポート 1 本を索引化 | 11 チャンク。`proxy は使わない` で一致を返却 |
| フック | `webfetch` で外部ページを取得 | 拒否され、`ctx_index`→`ctx_search` へ誘導 |
| 削減量 | `ctx_stats` | このセッションで 98.4% をコンテキスト外へ |
| スキル | `~/.config/opencode/skills/` | `ctx-doctor`・`ctx-stats`・`ctx-upgrade` の 3 つ（配置は 1 か所） |

## 4. 注意点

- **設定ファイルの BOM**。`opencode.json` に BOM が付くと opencode は `Invalid global configuration` で設定全体を拒否する。今回は除去して復旧した。JSON を書き換えるときは BOM なしで保存する。
- **MCP 経由の `ctx_doctor` は起動時の設定を保持する**。設定変更の直後は D3（host config）が FAIL のままになる。CLI の `context-mode doctor` は新しいプロセスで読むため PASS を返す。再起動後に一致する。
- 旧版のキャッシュ（`~/.cache/opencode/npm/context-mode@2.1.0`）は残っている。動作には影響しない。

## 5. ロールバック

1. `npm i -g context-mode@2.1.0` を実行する。
2. `~/.config/opencode/opencode.json` の `plugin` を `context-mode@2.1.0` に戻す。
3. `context-mode setup --agents opencode --yes` を実行する。
4. OpenChamber を再起動する。
