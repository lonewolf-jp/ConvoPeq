# context-mode 2.1.0 への更新と動作確認

- 実施: 2026-11-05 21:40–22:20（Windows 11 x64、OpenCode 2.2.5、OpenChamber、Node v24.22.2）
- 結論: 1.0.169 から 2.1.0 へ更新した。MCP の 9 ツールは稼働し、CLI の `doctor` は PASS、フックは WebFetch を `ctx_index` へ誘導することを実測で確認した。v2 は全面書き換えで、v1 の「MCP サーバー中心」から「フック＋サンドボックス＋プロジェクト別 DB」へ構造が変わっている。

## 1. 更新前後の状態

| 項目 | 更新前 | 更新後 |
|---|---|---|
| npm グローバル CLI | context-mode 1.0.169 | **2.1.0** |
| OpenCode プラグイン | 未読込（`~/.config/opencode/package.json` の依存のみ） | **context-mode@2.1.0**（OpenCode のキャッシュに導入） |
| MCP サーバー | 未登録 | **connected**（`ctx_*` 9 ツール） |
| データ | `~/.config/opencode/context-mode`（1.06 GB、v1 形式） | `~/.context-mode/projects/<hash>/context.db`（3 プロジェクト、104 KB／104 KB／2.3 MB） |
| スキル | なし | `ctx-doctor` / `ctx-stats` / `ctx-upgrade`（`~/.config/opencode/skills/`） |

## 2. v2 の主な変更点（学習結果）

### 提供形態

- v1 は「MCP プラグイン（11 ツール）」、v2 は「10 エージェント対応のプラグイン（9 ツール＋フック＋スキル）」。
- v2 はフックが本体である。エージェントのツール呼び出しを横取りし、大きい出力をサンドボックスと索引へ逃がす。
- Node 要件が **24.15.0 以上**へ上がった（v1 は 22.5 以上）。Windows は Git for Windows が必須（シェルツールが Git Bash で動く）。

### ツール（9 個）

| ツール | 役割 |
|---|---|
| `ctx_execute` | コードやシェルをサンドボックスで実行し、印字された分だけ返す |
| `ctx_batch` | 複数コマンドを実行し、出力を索引化して検索結果を 1 回で返す |
| `ctx_index` | ファイル・テキスト・URL をプロジェクト DB へ索引化 |
| `ctx_search` | 索引を検索し、一致部分だけ返す |
| `ctx_resume` | 圧縮後の状態ブロック（最大 2,048 B）または `seq` 指定の 1 行を返す |
| `ctx_stats` | コンテキストへ入れずに済んだ量を表示 |
| `ctx_doctor` | 導入状態を D1〜D11 で点検し、PASS か修正手順を返す |
| `ctx_upgrade` | 導入・更新・削除 |
| `ctx_purge` | セッション・ソース・プロジェクト単位で索引を削除 |

### フック

| フック | 動き |
|---|---|
| SessionStart | プロジェクト DB を開き、前回の続きがあれば `ctx_resume` を案内 |
| PreToolUse | 拒否リスト、WebFetch、curl／wget、インライン HTTP、ビルドコマンド、16,384 B 超の全ファイル Read を止め、`ctx_execute`／`ctx_index` の実呼び出しを提示。行範囲指定の Read はそのまま通す |
| PostToolUse | 4,096 B 超の結果を丸ごと索引化し、一致チャンクと `ctx_search` の案内に置換 |
| 圧縮時 | 最大 2,048 B の状態ブロックを残し、長い行は `ctx_resume(seq)` のポインタにする |

### 保存と検索

- データは `~/.context-mode` にプロジェクト別の `context.db` として置く。SQLite は Node 標準の `node:sqlite`、全文検索は FTS5。
- 索引は Porter 語幹の表と trigram の表の 2 本。検索は BM25 でそれぞれを順位付けし、RRF（`RRF_K` = 60）で統合する。空振りする前に綴り補正を試す。
- 上限値は `OUTPUT_CAP` 16,384 B、`REPLACE_TRIGGER_BYTES` 4,096 B、`MAX_CHUNK_BYTES` 4,096 B、`SNIPPET_MAX_BYTES` 800 B、`STATE_BUDGET` 2,048 B、`EXEC_TIMEOUT_MS` 600,000、`HOOK_TIMEOUT_MS` 5,000、`DB_TIMEOUT_MS` 5,000。

### 互換性とセキュリティ

- 2.0 が対応するのは 10 エージェント。**Cursor、JetBrains Copilot、Kimi、Codex Desktop、OpenClaw、OMP、Antigravity、Zed は 2.0 非対応**で、これらを使う場合は 1.0.169 に留める必要がある。
- テレメトリは無い。ネットワークは、`ctx_index` の URL 取得、`doctor` のバージョン確認、`ctx_upgrade` の clone と `npm i -g`、利用者自身の導入コマンドの 5 種類だけ。

## 3. 実施した手順

1. `~/.config/opencode` の `opencode.jsonc`・`package.json`・`package-lock.json` を `backup-contextmode-20261105-2142` へ退避した。
2. `opencode plugin add context-mode` を実行した。プラグインは登録されたが、既存依存 `^1.0.169` の範囲内のため 1.0.169 のままだった。
3. `opencode plugin add context-mode@2.1.0` で 2.1.0 を導入した。プラグインの実体は `~/.cache/opencode/npm/context-mode@2.1.0/` に置かれた。
4. `context-mode setup --agents opencode --yes` を実行し、MCP エントリ（`~/.config/opencode/opencode.json`）とスキル 3 本を作成した。あわせて `opencode.jsonc` 側の重複エントリを削除し、プラグインを 1 つに保った。
5. `npm i -g context-mode@2.1.0` で CLI を更新し、`~/.config/opencode` の依存を `^2.1.0` に合わせた。
6. `context-mode doctor --agents opencode` が PASS（parse / list / mcp / hook-registered / hook-runs / hook-seen / skills）になることを確認した。

## 4. 動作確認（実測）

| 確認 | 条件 | 結果 |
|---|---|---|
| MCP 接続 | `opencode mcp list` | `context-mode connected` |
| ツール数 | MCP の tools/list | 9 個（`ctx_execute`〜`ctx_purge`） |
| サンドボックス実行 | JS で 500 行の配列を集計 | 47 B の答えだけ返却（`{"rows":500,"sum":24997,"maxId":30,"maxV":100}`） |
| 大出力の圧縮 | `seq 1 3000` に 22 B を付加（85,893 B） | 返却 188 B（99.8% をコンテキスト外へ）。`ctx_search` の案内付き |
| 索引と検索 | レポート 1 本を索引化 | 11 チャンク。見出し・本文の一致を返却 |
| バッチ | 2 コマンド（計 70,898 B）＋2 クエリ | 一致チャンクを 15,596 B で返却 |
| 状態ブロック | `ctx_resume` | 1,980 B（上限 2,048 B 以内） |
| 統計 | `ctx_stats` | このセッション 67.3%、最大 1 件は 83.9 KB → 189 B |
| フック（WebFetch） | 外部ページの取得 | **拒否され、`ctx_index`→`ctx_search` へ誘導された** |
| フック（Read） | 40,692 B のファイルを全読み | 拒否されず通過（下記の注意点） |
| doctor | CLI（新規プロセス） | **PASS** |

## 5. 注意点

- **全ファイル Read の拒否は効かなかった**。40,692 B のファイルを `limit` なしで読んでも通過した。行範囲を指定した Read は仕様どおり許可されるため、拒否条件は「範囲指定のない全読み」のはずだが、この環境の Read 呼び出しでは一致しなかった。WebFetch の拒否は効いているため、フック自体は稼働している。
- **MCP 経由の `ctx_doctor` は起動時の設定を保持する**。設定を直した直後は D3（host config）が FAIL のままになる。CLI の `context-mode doctor` は新規プロセスで読むため PASS を返す。再起動後に一致する。
- **v1 の旧データ 1.06 GB が残っている**。`~/.config/opencode/context-mode` は v2 から参照されない。削除は不可逆なため実施していない。
- v2 は MCP サーバーを `~/.config/opencode/opencode.json` に、プラグインを npm 仕様 `context-mode@2.1.0` で持つ。`opencode.jsonc` 側には context-mode を置いていない（二重読込を避けるため）。

## 6. ロールバック手順

1. `~/.config/opencode/opencode.json` を削除する（v2 の MCP エントリとプラグイン指定が消える）。
2. `~/.config/opencode/opencode.jsonc` の `plugins` に `"context-mode"` を戻し、`package.json` の依存を `^1.0.169` に戻す。
3. `npm i -g context-mode@1.0.169` を実行する。
4. OpenChamber を再起動する。

退避した設定は `C:\Users\user\.config\opencode\backup-contextmode-20261105-2142` にある。

## 7. 未検証の範囲

- コンテキスト圧縮時の `ctx_resume` の実動作（このセッションでは圧縮が発生していない）。
- opencode 以外の 9 エージェントでの動作。
- 全ファイル Read の拒否（上記のとおり条件が一致しなかった）。
