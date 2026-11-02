# 整理作業ラウンド 3 の報告（2026-11-02・Owner 判断の実行）

Owner 判断（2026-11-02 / APPROVED WITH CONDITIONS）に基づく実施記録。
対象は明示された範囲（worktree 4 本の閉鎖、ローカルブランチ 7 本の削除、退避、読み取り専用の接続確認）に限る。

## 1. worktree の閉鎖

### 1.1 無条件承認 3 本

| worktree | 実行 | 終了コード | 閉鎖後のディレクトリ |
|---|---|---|---|
| `…\f348938043…\crisp-zebra` | `git worktree remove` | 0 | 消滅 |
| `…\Temp\opencode\wt-g3` | `git worktree remove` | 0 | 消滅 |
| `…\Temp\opencode\wt-r16` | `git worktree remove` | 0 | 消滅 |

`git worktree prune -v` も実行（終了コード 0・除去対象なし）。

### 1.2 条件付き承認 `bright-falcon`

条件 1〜5 をすべて満たしたことを確認してから閉鎖した。

| 条件 | 実施内容 | 結果 |
|---|---|---|
| 1. 未コミット 3 ファイルのパッチ保存 | `git diff --binary` の生バイトを保存 | `bright-falcon-uncommitted.patch`（27,687 B / SHA-256 `d8babfdc…`） |
| 2. 未追跡 243 件の退避 | ZIP 化（相対パス保持） | `bright-falcon-untracked.zip`（6,965,024 B / 243 エントリ） |
| 3. 退避物の件数・内容・保存先の検証 | 件数一致・読み戻し照合・保存先の実体確認 | 243 = 243、**不一致 0**、保存先を §2 に記録 |
| 4. 退避完了記録 | `README.md`・`MANIFEST.txt`・`SHA256SUMS.txt` を作成 | §2 の 5 ファイル |
| 5. 閉鎖直前の再確認 | 再 diff のハッシュ比較・件数再計測 | 3 / 243・パッチ SHA **一致**（`d8babfdc…`） |

閉鎖の実行: `git worktree remove` は未コミット/未追跡があるため
`fatal: ... contains modified or untracked files, use --force` で拒否された。
**退避と検証が完了していることを確認済み**のため、`git worktree remove --force` で閉鎖した
（作業内容は §2 の退避物に保全）。閉鎖後のディレクトリは消滅している。

### 1.3 閉鎖後の worktree 一覧

| worktree | HEAD |
|---|---|
| `C:/VSC_Project/ConvoPeq` | `bc56f977`（main） |
| `…\f34893\j195-notion-report` | `bda43034`（detached） |
| `…\f34893\j237-baseline` | `d79b8be4`（detached） |
| `…\f34893\j237-baseline-clean` | `d79b8be4`（detached） |

対象 4 本は一覧から消えている。

## 2. `bright-falcon` の退避物（保存先）

ローカル: `C:\Users\user\AppData\Local\Temp\opencode\archive-bright-falcon-20261102\`
Dropbox: **`/archive-bright-falcon-20261102/`**

| ファイル | サイズ | SHA-256 |
|---|---|---|
| `bright-falcon-uncommitted.patch` | 27,687 B | `d8babfdc1f3a70dda21254ed1b3d36822eb0937104b16375484f1ebcafb7017e` |
| `bright-falcon-untracked.zip` | 6,965,024 B | `f397669e1cbbc368ec553caa2b52ea0659fb90e523968514ea96c015f96c819a` |
| `MANIFEST.txt` | 33,292 B | `106c1df551036780f663c93849e7f7c82f5a5c558b2c1ae949f4e30b8f4b41d7` |
| `README.md` | 2,424 B | `SHA256SUMS.txt` に記載 |
| `SHA256SUMS.txt` | 347 B | 上記 4 件の一覧 |

- パッチの内訳: 3 ファイル / 345 insertions / 11 deletions。
- 退避元: ブランチ `bright-falcon`、HEAD `d79b8be4530826c8a2dc748cac4fc6d6e3222e04`。
- Dropbox への配信は全 5 ファイルで**内容一致を確認済み**（ダウンロードして SHA-256 を照合）。

## 3. ローカルブランチ削除

### 3.1 worktree 閉鎖に伴う 4 本

削除前に **tip と archive タグの対象 SHA の一致**、およびタグのリモート存在を確認した。

| ブランチ | tip | archive タグの対象 | 一致 | タグのリモート | 結果 |
|---|---|---|---|---|---|
| `crisp-zebra` | `bda43034` | `bda43034` | ○ | あり | 削除 |
| `work/g3-snapshot-ignore-20261026` | `812abcc8` | `812abcc8` | ○ | あり | 削除 |
| `work/r16-i4-sync-20261026` | `3f3225d7` | `3f3225d7` | ○ | あり | 削除 |
| `bright-falcon` | `d79b8be4` | `d79b8be4` | ○ | あり | 削除 |

照合 4/4 一致・不一致 0 のため削除を実施。**リモートブランチの追加削除は行っていない。**

### 3.2 `ci/r50` 系の残り 3 本

削除条件を「PR が MERGED」「main の祖先（取り込み済み）」「worktree で使用中でない」の
3 点で再確認した。

| ブランチ | PR | main の祖先 | main..branch | 固有変更 | 結果 |
|---|---|---|---|---|---|
| `ci/r50-probe-cleanup-20261029` | #84 MERGED（`023ae16f`） | はい | 0 | 0 ファイル | 削除 |
| `ci/r50-probe-wait-fix-20261029` | #83 MERGED（`7d2679a6`） | はい | 0 | 0 ファイル | 削除 |
| `ci/r50-vbcable-probe-20261030` | #80 MERGED（`4194accb`） | はい | 0 | 0 ファイル | 削除 |

> 初回の検証では `git diff main..<branch>` の差分数を「残差」と誤って扱い、3 本を中止判定にした。
> main が 88〜96 コミット先行しているため当然に差分が出る指標であり、判定を
> **`git merge-base --is-ancestor <branch> main`**（＝完全に取り込み済み）へ訂正して再実行した。
> 中止判定時は 1 本も削除していない。

**リモートブランチの削除は行っていない**（当該 3 本はいずれもリモートに存在しない）。

## 4. `ci/h2r1`（保留・記録は現状維持）

- 再測定は**実施していない**。CI workflow の追加・変更・恒久化も**行っていない**。
- `H2R1-MEASUREMENT-RECORD-20261102.md` は**変更していない**（本ラウンドで無変更）。
- 再測定には別の Owner 判断（測定目的・CI 定義変更・実行条件・結果の保存方法）が必要である旨を
  記録済みの文書に残している。

## 5. uvx 切替の確定確認（読み取り専用）

| 確認項目 | 結果 |
|---|---|
| リポジトリ設定の `Python314` 参照 | **0**（`Python315` は 11） |
| 新経路プロセスの稼働 | あり（`pythonw` → `nowin.py` → `Roaming\Python\Python315\Scripts\uvx.exe`、生成 11:47:04） |
| 旧経路プロセスの残存 | あり（`C:/Python314/Scripts/uvx.exe`、生成 09:55:17）。設定変更前から動いている別ディレクトリ用の個体 |
| サーバー名 | `ddgs-search`（新旧両経路のハンドシェイクで一致） |
| tools 数 | **6**（ライブの MCP クライアントで確認） |
| tool 名 | `search_text` / `search_images` / `search_news` / `search_videos` / `search_books` / `extract_content`（完全一致） |

- 設定変更・追加インストール・他サーバーの変更は**行っていない**。
- 旧経路が残る理由: セッション worktree
  `…\worktree\f34893\j195-notion-report\opencode.json` が `Python314` を 1 箇所参照している。
  これは `C:\VSC_Project\ConvoPeq` の外にあり、本判断の対象外のため**変更していない**（残件として報告）。

## 6. 現在の状態

| 項目 | 値 |
|---|---|
| ブランチ | `main`（`HEAD` = `origin/main`） |
| ローカルブランチ | **58 本**（65 → 61 → 58） |
| リモートブランチ | **47 本**（本ラウンドで削除なし） |
| アーカイブタグ | **17 個**（ローカル・リモートとも 17） |
| worktree | 4 本（main ＋ detached 3 本） |
| 追跡差分 | 0 |
| 未追跡 | 0 |
| CI workflow / 検査スクリプトの変更 | 0 件 / 0 件 |
| policy の変更 | 既承認 1 行（`c9e74603`）のみ |
| U5 / U3 / U4 | U5 = FAIL・既定 OFF、U3/U4 = HOLD |

## 7. 残存事項

| # | 項目 | 状態 |
|---|---|---|
| 1 | セッション worktree の `opencode.json`（`Python314` 1 箇所） | 対象外のため未変更。必要なら別途承認 |
| 2 | `ci/h2r1` の統合後再測定 | 保留（別の Owner 判断が必要） |
| 3 | 旧経路 ddgs プロセスの終了 | 上記 1 を直さない限り、当該ディレクトリのサーバー再起動で再生成される |
| 4 | U5 有効化・U3/U4 実行・リリース・外部公開 | 未承認のまま維持 |
