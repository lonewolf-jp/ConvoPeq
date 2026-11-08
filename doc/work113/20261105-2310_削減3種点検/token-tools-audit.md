# トークン削減 3 ツールの設定点検と修正（proxy 不使用版）

- 実施: 2026-11-05 22:45–23:40（Windows 11 x64、OpenCode 2.2.5、OpenChamber）
- 前提: headroom は **MCP と CLI のみ**を使い、**proxy は使わない**。
- 結論: 3 ツールとも稼働している。今回、headroom の proxy（8787）が自動起動していたため停止し、MCP と CLI だけを残した。あわせて context-mode の MCP 二重定義と、headroom プラグインの壊れた実行ファイル解決、rtk の無駄な書き換えを修正した。

## 1. 点検結果

| ツール | 導入 | 稼働 | 見つけた問題 | 処置 |
|---|---|---|---|---|
| rtk | Windows 1.1.0（`C:\WINDOWS\system32\rtk.exe`）と WSL 1.1.0 | プラグイン `rtk.ts` 読込済み。`rtk hook opencode` が書き換え JSON を返す | 書き込み系の git コマンドまで書き換え、トークンが増える場合があった（`git add` で −11 トークン） | 両方の `config.toml` の `exclude_commands` に git の書き込み系 6 種を追加 |
| headroom | uv ツール 0.42.0、MCP 接続済み | MCP 3 ツールが応答。CLI 0.42.0 | **proxy が 8787 で常駐し、ログオン時に自動起動していた**（方針違反）。`headroom-proxy` プラグインも proxy を起動・監視していた | 自動起動を外し、プラグインを無効化し、proxy を停止。MCP と CLI は維持 |
| context-mode | 2.1.0（CLI・プラグイン・MCP） | 9 ツール、フック稼働、doctor PASS | プロジェクト `opencode.json` に v1 時代の MCP 定義が残り、グローバル定義と二重になっていた | プロジェクト側を削除し、`--agent opencode` 付きの 1 定義に統一 |

## 2. 修正の詳細

### headroom の proxy を停止（方針: proxy 不使用）

| 対象 | 処置 |
|---|---|
| ログオン時の自動起動 | `Startup\start-headroom-proxy.bat` とそのバックアップを `C:\Users\user\AppData\Local\headroom-disabled-20261105` へ退避 |
| プラグイン | `.opencode/plugins/headroom-proxy` を `.opencode/plugins-disabled/` へ移動（走査対象外） |
| 稼働中プロセス | supervisor（cmd）と proxy のプロセスツリーを停止 |
| 確認 | 8787 の待受なし、`/health` 接続不可、proxy プロセスなし。60 秒後も再起動しない |

- MCP（`headroom mcp serve`）と CLI は停止していない。`opencode mcp list` で connected、`headroom --version` は 0.42.0。
- 退避した BAT は、方針を戻すときに Startup へ戻せば復旧する。

### context-mode の二重定義を解消

- 削除: `C:\VSC_Project\ConvoPeq\opencode.json` の `mcp.servers.context-mode`（`context-mode.cmd` と `CONTEXT_MODE_DIR`）。
- 残置: `C:\Users\user\.config\opencode\opencode.json` の `mcp.context-mode`（`node server.mjs --agent opencode`）。
- 確認: `opencode debug config` の設定ソースを走査し、`context-mode` を含むファイルが 1 つだけになった。

### headroom プラグインの実行ファイル解決（無効化前の修正）

`<project>\.venv\Scripts\headroom.exe` だけを探していた。この `.venv` は基準 Python（`C:\Python314`）が消えて起動できないため、proxy が落ちると再起動できなかった。解決順を「`HEADROOM_EXE` → uv ツール → `.venv`」に変え、存在しない場合は spawn せず理由を出すようにした。プラグイン自体は今回無効化したが、再有効化したときのために修正を残す。

### rtk の除外設定

`exclude_commands` に `git add`・`git commit`・`git push`・`git switch`・`git checkout`・`git stash` を追加した。出力が小さいコマンドは `rtk ` を付ける分だけトークンが増えるため、書き換えない方が得である。Windows 側（プラグインが使う実体）と WSL 側の両方に適用した。

## 3. 動作確認

| 確認 | 条件 | 結果 |
|---|---|---|
| headroom proxy の停止 | `/health`、待受、プロセス | 接続不可、待受なし、proxy プロセスなし（60 秒後も維持） |
| headroom MCP の生存 | `opencode mcp list` | connected |
| headroom の圧縮（JSON） | 300 件の JSON、55,172 B | 25,501 B（スキーマ化して圧縮） |
| headroom の圧縮（ログ） | 600 行のログ、51,419 B | 946 B（98% 削減、hash で復元可能） |
| headroom CLI | `headroom --version` | 0.42.0 |
| rtk の書き換え | `rtk hook opencode "git log --oneline -3"` | `{"command":"rtk git log --oneline -3"}` |
| rtk の除外 | `rtk hook opencode "git commit -m x"` | `{}`（書き換えない） |
| rtk の実績 | `%LOCALAPPDATA%\rtk\history.db` | 543 件。`git add` → `rtk git add` の記録あり |
| context-mode の圧縮 | `ctx_execute` 85,893 B の出力 | 返答 188 B（99.8% 削減） |
| context-mode の点検 | `context-mode doctor --agents opencode` | PASS（D1〜D11） |
| 3 ツールの同時接続 | `opencode mcp list` | context-mode・headroom ともに connected |

## 4. 役割分担（現行）

| ツール | 担う範囲 | 状態 |
|---|---|---|
| rtk | シェルコマンドの出力を圧縮版コマンドへ差し替える | 有効。書き込み系 git は除外 |
| context-mode | 大きいツール出力をサンドボックスと索引へ逃がし、必要分だけ返す | 有効。出力圧縮の主役 |
| headroom | MCP の `headroom_compress`／`headroom_retrieve`／`headroom_stats` と CLI | 有効。proxy は停止済み |

- headroom の MCP ツールは接続されているが、Code Mode のツール一覧には出ていない。aidex・brave・cocoindex など計 11 サーバーも同様で、ハーネス側の公開範囲の制約である。
- rtk の書き換えは、このセッションのシェル実行が opencode のプラグインフックを通る場合に効く。設定変更後、本セッションの実行では履歴が増えておらず、OpenChamber 側の実行経路ではフックが介在しない可能性がある。CLI 単体では正常に書き換える。

## 5. 再起動で反映される事項

- context-mode の旧 MCP プロセス（`cli.bundle.mjs`）の整理。
- rtk の除外設定。
- headroom プラグインの無効化（すでに `opencode plugin list` から消えている）。

## 6. 未実施・要判断

- 退避した headroom 用 BAT（`C:\Users\user\AppData\Local\headroom-disabled-20261105`）の削除。方針を戻す可能性があるため残した。
- v1 の context-mode データ 1.06 GB（`~/.config/opencode/context-mode`）の削除。
- rtk の履歴 543 件と `tee` の退避ファイルの削除。
