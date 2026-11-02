# 整理作業ラウンド 2 の報告（2026-11-02）

Owner 指示（worktree 閉鎖候補の提示 / archive タグ push / `ci/r50` 整理 / 保留 14 件の削除 /
`ci/h2r1` 現状維持と記録 / `opencode.json`・`.gitignore` の是正）に対する実施記録。

**本ラウンドでは worktree を閉鎖していない**（Owner の明示承認が必要なため）。
未承認の CI 変更・リリース・機能有効化も行っていない。

## 1. worktree 閉鎖候補（提示のみ・未実施）

### 1.1 共通の実測

- 4 本すべてに**接続中のプロセスは無い**（`Win32_Process` のコマンドラインを走査。ヒットしたのは
  本調査自身の `pwsh` のみ）。
- `index.lock` / `HEAD.lock` は 4 本とも存在しない。
- stash は 4 本とも 0。

### 1.2 個別の状態

| worktree | ブランチ | HEAD | 未コミット | 未追跡 | 最終更新 | 判定 |
|---|---|---|---|---|---|---|
| `…\f348938043…\crisp-zebra` | `crisp-zebra` | `bda43034` | **0** | **0** | 2026-10-05 | **閉鎖可**（作業なし） |
| `…\Temp\opencode\wt-g3` | `work/g3-snapshot-ignore-20261026` | `812abcc8` | **0** | **0** | 2026-10-26 | **閉鎖可**（一時領域） |
| `…\Temp\opencode\wt-r16` | `work/r16-i4-sync-20261026` | `3f3225d7` | **0** | **0** | 2026-10-26 | **閉鎖可**（一時領域） |
| `…\f348938043…\bright-falcon` | `bright-falcon` | `d79b8be4` | **3 ファイル** | **243 件** | 2026-10-14 | **条件付き閉鎖可**（下記 1.3） |

### 1.3 `bright-falcon` の未保存変更（要判断）

変更 3 ファイルと、その内容が main に存在するかの実測。

| ファイル | 差分（worktree HEAD 比） | main の対応 |
|---|---|---|
| `output_sourcecode_markdown.py` | +147 / −11 | main は `parse_generated_stamp` / `build_snapshot_dir_name` / `resolve_archive_dir` / `archive_snapshot` として**同機能を別実装**で持つ |
| `PublishPipelineIntegrationTests.cpp` | +105 / −0 | `STG8_ONLY` ゲートは main に**存在**（4 箇所） |
| `STG8RecoveryObligationTests.cpp` | +93 / −0 | `hasDeferredRequest`（41 箇所）/ `getMaxDeferredAgeMs`（2）/ `recoveryPublishSeq`（1）/ `deferredOverwriteCount`（5）が main に**存在** |

- 3 ファイルとも main の現行版とは一致しない（worktree 側が 2026-10-07 ベースの初期稿）。
- 追加行のうち main に同名で存在するものは 12/85、14/25、3/49。残りは**実装が置き換わったため
  同一行としては存在しない**が、機能としては main 側にある。
- 未追跡 243 件の多くは `.openchamber/screenshots/*.jpg`（2026-10-13 の画面証跡）。

**推奨**: 閉鎖前に ①未コミット 3 ファイルの `git diff` をパッチとして保存、②未追跡 243 件を
1 つのアーカイブにまとめて Dropbox へ退避、その後に閉鎖。作業そのものは main に置換済みのため
復元する必要は薄いが、証跡として残す価値はある。

### 1.4 閉鎖後の手順（承認後に実施する内容）

1. `git worktree remove <path>`（強制削除オプションは使わない）
2. `git worktree prune` で管理情報を整理
3. ブランチ削除前に `archive/<branch>-<short>` と tip の一致を再確認
4. `git branch -D <branch>`（4 本とも tip は既にタグ保全済み・リモートは削除済み）

## 2. archive タグのリモート保全（実施済み）

- 対象 16 個について、**タグ名・対象 SHA・リモート未存在**を個別に検証（problems = 0）。
- すべて lightweight タグで、対象は削除したブランチの tip と一致。
- `git push origin --tags archive/*` で push（**`--force` は使用していない**）。
- 結果: **リモート 16/16**、欠落 0。加えて `ci/r50` 用の 1 個を追加（下記 3.）。

| 確認項目 | 結果 |
|---|---|
| ローカルの archive タグ | 16（想定 16・余分 0） |
| 対象 SHA の一致 | 16/16 |
| push 前のリモート存在 | 0/16 |
| push 後のリモート存在 | **16/16** |
| force の使用 | なし |

## 3. `ci/r50` の整理（実施済み）

### 3.1 再確認した事実

| 確認項目 | 実測 |
|---|---|
| main の policy | `std::vector<DSPCore\*>\s+deferredDestroy_` に同期済み（`c9e74603`・既承認） |
| `ci/r50` の残差分 | **policy 1 行のみ**で、内容は `std::vector<void\*>` へ**戻す**もの（承認済み変更の巻き戻し） |
| workflow の差分 | **0 ファイル**（`sanitizer-ci.yml` の内容は main に存在） |
| PR | #85 **MERGED**（内容は main に取込済み） |
| 判定 | 未解決の policy 提案ではなく、**陳腐化した旧行** |

### 3.2 実施内容

- tip `fcf383a9` に `archive/ci-r50-asan-vbcable-20261029-fcf383a9` を作成し、リモートへ push。
- ブランチを削除（ローカル・リモートとも）。削除後、ローカル/リモートとも**存在しない**ことを確認。

### 3.3 対象外として残したもの

`ci/r50` 系の他 3 本（`ci/r50-probe-cleanup-20261029` / `ci/r50-probe-wait-fix-20261029` /
`ci/r50-vbcable-probe-20261030`）は、いずれも PR #84 / #83 / #80 で **MERGED**、
リモートには存在せずローカルのみ。本指示の対象外のため**削除していない**。

## 4. 保留 14 ファイルの削除（実施済み）

### 4.1 削除前マニフェスト（再検証済み）

| # | パス | サイズ | SHA-256（先頭 16） |
|---|---|---|---|
| 1 | `.openchamber/convopeq-md-split-20261030/ConvoPeq.part01.md` | 2,097,115 B | `4c1d09212da8aba5` |
| 2 | 同 `ConvoPeq.part02.md` | 2,097,123 B | `83fd7edafe36a602` |
| 3 | 同 `ConvoPeq.part03.md` | 2,097,111 B | `4340b3ae99d14573` |
| 4 | 同 `ConvoPeq.part04.md` | 145,010 B | `898036199cdc01cc` |
| 5 | 同 `SHA256SUMS.txt` | 340 B | `80ce590bd534b4b0` |
| 6 | `doc/ConvoPeq-f8aab49b-split/README.md` | 3,987 B | `f48f4bc84a31dd22` |
| 7 | `doc/ConvoPeq-f8aab49b-split/SHA256SUMS.txt` | 340 B | `80ce590bd534b4b0` |
| 8 | `opencode.json.bak-py314-20261022-201143` | 4,387 B | `d9f1221d6111a788` |
| 9 | `opencode.json.bak-py314-20261022-204607` | 4,387 B | `a6ff625131a3d85a` |
| 10 | `opencode.json.bak-py314-20261022-204647` | 4,387 B | `a6ff625131a3d85a` |
| 11 | `opencode.json.bak-py314-20261022-204708` | 4,387 B | `a6ff625131a3d85a` |
| 12 | `opencode.json.bak-py314-20261022-204731` | 4,387 B | `a6ff625131a3d85a` |
| 13 | `opencode.json.bak-py314-20261022-204802` | 4,410 B | `9b40df06f04837fe` |
| 14 | `vc140.pdb` | 69,632 B | `1b61307f5f9f868f` |

合計 **6,537,003 B**。14 件すべて**未追跡**であることを確認。削除直前にサイズとハッシュを
再検証し、不一致 0 件。

### 4.2 削除結果

- 削除 **14/14**、残存 **0**。
- 空になった `…/convopeq-md-split-20261030/` と `doc/ConvoPeq-f8aab49b-split/` を除去。
- 削除後の `git status`: **未追跡 0 件**、追跡変更は本作業で編集したファイルのみ。
- 旧正規ソース（`f8aab49b…`）のアーカイブは Dropbox `/ConvoPeq-f8aab49b-split/` に残る
  （4 分割のハッシュは削除前に全一致を確認済み）。

## 5. `.gitignore` の是正（実施済み）

### 5.1 差分

```diff
@@ -1,4 +1,3 @@
-```
 # Build artifacts
 *.o
@@ -7,6 +6,7 @@
 *.exe
+*.pdb
 *.out
```

- 先頭行の ``` ``` ``` を除去（`@ -1,4 +1,3`）。**他に ` ``` ` という名前のファイルは無く**
  （追跡 0・未追跡 0）、除外規則としての影響は無い。
- `*.pdb` は既存規則と**重複しない**（`*.pdb` の規則は従来存在せず、`tools/**/bin/` は
  ディレクトリ単位の除外）。追加位置はビルド生成物の並び（`*.exe` の直後、現行 L9）。

### 5.2 影響確認

| 確認項目 | 結果 |
|---|---|
| 追跡済み `.pdb` の扱い | `tools/EtwWaitTimeAnalyzer/bin/Debug/net10.0/EtwWaitTimeAnalyzer.pdb` は**追跡のまま**（git は追跡ファイルに ignore を適用しない） |
| `vc140.pdb` | `.gitignore:9:*.pdb` で除外される |
| `build/` 配下の `.pdb` | 従来どおり `build/` 規則で除外 |
| `ConvoPeq.md` | 除外されない（従来どおり追跡） |
| 新たに無視される追跡ファイル | **0**（`git status` の追跡差分に増加なし） |

## 6. `opencode.json` の uvx 切替（実施済み・検証付き）

### 6.1 変更前の状態

| 項目 | 値 |
|---|---|
| 参照 | `C:/Python314/Scripts/uvx.exe` |
| 実体 | uvx **0.12.10**（単体バイナリ。`C:\Python314` には `python.exe` が無い） |
| 接続 | ddgs は **6 tools** で接続（`opencode.log` 2026-11-02 00:55） |
| サーバー名 | `ddgs-search` |
| tools | `search_text` / `search_images` / `search_news` / `search_videos` / `search_books` / `extract_content` |

### 6.2 切替候補の検証（変更前に実施）

同一の起動系（`pythonw` ＋ `nowin.py` ＋ uvx ＋ `--from "ddgs[mcp]" ddgs mcp`）でハンドシェイクを実行。

| 経路 | uvx 版 | サーバー名 | tools | tool 名の一致 |
|---|---|---|---|---|
| 変更前（Python314） | 0.12.10 | `ddgs-search` | 6 | — |
| 変更後（Python315） | 0.13.3 | `ddgs-search` | **6** | **完全一致** |

### 6.3 変更内容

- `opencode.json` L121 を `C:/Users/user/AppData/Roaming/Python/Python315/Scripts/uvx.exe` へ。
- 差分は **1 行のみ**。JSON は妥当（`parse = OK`、サーバー 12 個）。
- `Python314` の参照は **0**（他 11 箇所は `Python315`）。
- 各サーバーの実行ファイル存在確認: **不存在 0**。

### 6.4 反映の時期

MCP サーバーは**サーバー起動時に設定を読む**ため、本変更は**次回の opencode サーバー再起動時**に
有効になる。現在のセッションの ddgs は旧経路（0.12.10）で稼働中であり、旧経路も動作することを
確認済み。切替による機能差は無い（サーバー名・tools 数・tool 名が一致）。

## 7. `ci/h2r1`（現状維持・記録済み）

- ブランチ `ci/h2r1-release-measurement-20261030`（tip `d7036618`）は**保持**。main への統合・
  恒久化・削除は**行っていない**。
- 測定値・実行 ID・運用コストを `H2R1-MEASUREMENT-RECORD-20261102.md` に記録した。
- 残件として「**統合後の再測定が未実施**」「現行 main を測るには main 側にワークフローが必要」
  を明記した。

## 8. 検証結果

| 対象 | 結果 |
|---|---|
| GitHub `main` とローカル `HEAD` | 一致（後述の commit 後に再確認） |
| 追跡内容 | 全追跡ファイルの `hash-object` と HEAD blob の照合で相違 0（再実行） |
| Dropbox 正規ソース | `ConvoPeq.md`（6,451,711 B / `c12a532e…`）と一致（再確認） |
| CI workflow | 変更 **0 件** |
| CI 検査スクリプト | 変更 **0 件** |
| policy | 本ラウンドで変更なし（既承認 1 行のみ） |
| U5 / U3 / U4 | U5 = FAIL・既定 OFF、U3/U4 = HOLD を維持 |

## 9. 新たに残った Owner 判断事項

| # | 項目 | 内容 |
|---|---|---|
| 1 | worktree 閉鎖 | `crisp-zebra` / `wt-g3` / `wt-r16` の 3 本を閉鎖してよいか（作業なし） |
| 2 | `bright-falcon` の閉鎖 | 未コミット 3 ファイルのパッチ保存と未追跡 243 件の退避を行ったうえで閉鎖してよいか |
| 3 | 閉鎖後のブランチ削除 | 閉鎖後に 4 本を削除してよいか（tip はタグ保全済み・リモート削除済み） |
| 4 | `ci/r50` 系の残り 3 本 | `ci/r50-probe-cleanup` / `-probe-wait-fix` / `-vbcable-probe`（ローカルのみ・PR は MERGED）の削除可否 |
| 5 | `ci/h2r1` の再測定 | 統合後の再測定を行うか（行う場合は CI 定義変更の承認が必要） |
| 6 | `uvx` 切替の確定 | 次回サーバー再起動後に ddgs が 6 tools で接続することを確認するか（確認は再起動後に実施） |
