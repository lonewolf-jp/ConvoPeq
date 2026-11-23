# 保護ツリーの WIP 保全（2026-11-23）

対象: `C:\VSC_Project\ConvoPeq`（HEAD `4d476d5c`、main より 213 コミット遅れ）。目的は、未コミットの作業を**失わない形で確保**し、後から upstream と突き合わせられるようにすることである。

**保護ツリーは一切変更していない。** rebase・reset・checkout・clean・add・stash のいずれも実行していない。読み取りだけで完結させた。

## 1. 保全した状態

| 項目 | 値 |
| --- | --- |
| HEAD | `4d476d5c65261fd975de2606f4d8b5c843cd86e1` |
| 追跡ファイルの変更 | 42 ファイル、+10,074 / −1,015 |
| 未追跡ファイル | 354（`doc/` の報告書、`src/InputInspection.*`、`src/learner/`、テスト 6 本など） |
| status エントリ合計 | 396 |
| status のシグネチャ（SHA-256 先頭 16 桁） | `6e346f32dd6f5db6`（作業の前後で同一） |

## 2. 復元手段（3 系統）

### 2.1 GitHub の保全ブランチ

`wip/preserve-20261123`（commit `adfb16f6`、親 `4d476d5c`、tree `ac05c3c2`）を push した。親が保護ツリーの HEAD と同じなので、**この 1 コミットがそのまま未コミット作業の差分**になる。

```
git fetch origin wip/preserve-20261123
git switch -c restore origin/wip/preserve-20261123
```

### 2.2 追跡ファイルのパッチ

`tracked_changes.patch`（761,731 B、SHA-256 `4b818ba3…`）。`git diff --binary` の出力で、42 ファイルの変更をそのまま含む。

```
git apply --binary tracked_changes.patch
```

### 2.3 未追跡ファイルのアーカイブ

`untracked_files.zip`（20,928,610 B、354 ファイル、SHA-256 `3c91a636…`）。展開すれば未追跡の状態に戻る。ほかに `status_record.json`（HEAD・変更一覧・未追跡一覧）を残した。

3 つとも `%TEMP%\opencode\wip236\` にあり、同じものを Dropbox の `/ConvoPeq-wip-preserve-20261123/` にも置いた。ブランチ・パッチ・アーカイブのどれか 1 つが残れば復元できる。

## 3. 保護ツリーを触らずにコミットを作る方法

複製は `git clone --shared --no-checkout` で行った（`.git` が 973 MB、作業ツリーが 62 GB あるため、通常の clone は取れない）。複製先で次の順に組み立てた。

1. `git read-tree 4d476d5c` でインデックスだけを用意する（作業ツリーは作らない）。
2. 396 ファイルそれぞれについて、`git hash-object -w --path <相対パス> <保護ツリーの実ファイル>` で**ディスク上の内容をそのまま** blob にする。
3. `git update-index --index-info` でインデックスへ反映し、`git write-tree` → `git commit-tree <tree> -p 4d476d5c`。

**つまずいた点**: 複製先に `.gitattributes` が無いと `--path` の属性参照が効かず、clean フィルタ（`text eol=lf`）が適用されない。その状態で作った最初のコミットは CRLF がそのまま入り、差分が 2,123,111 行追加 / 154,666 行削除に膨らんだ。base commit から `.gitattributes` を取り出して複製先の直下に置いたところ、1,969,460 / 1,015 に収まり、追跡 42 ファイルの数値が保護ツリーの `git diff` と一致した（残る大量の追加行は、未追跡の `ConvoPeq.md` スナップショット群によるものである）。

## 4. 忠実性の確認

- 追跡 42 ファイル: 保護ツリーの `git diff --numstat` と保全コミットの `--numstat` が**完全一致**（不一致 0）。
- 未追跡 354 ファイル: `git status` の一覧と保全コミットの新規ファイル数が一致（354）。
- 合計 396 エントリ。コミットの親は `4d476d5c`。
- リモートの ref と tree がローカルの値（`ac05c3c2…`）と一致することを GitHub API で確認した。

## 5. 秘密情報の確認

push の前に、396 ファイルの内容をパターン走査した。対象は AWS のアクセスキー、GitHub のトークン（`ghp_` / `github_pat_`）、`sk-` 形式の API キー、Slack トークン、秘密鍵のヘッダ、`password` / `secret` / `api_key` / `access_token` への代入である。**検出 0 件**。`.env`、`id_rsa`、`*.pem`、`*.pfx`、`*.key`、`credential`、`secret` といった名前のファイルも無かった。これを受けて push している。

なおリポジトリは公開なので、保全ブランチも誰でも読める。今回の走査で秘密情報は無かったが、今後このブランチに追記する場合は同じ確認を行うこと。

## 6. 保護ツリーが無変更であること

作業の前後で次が一致した。

- HEAD: `4d476d5c`
- status エントリ数: 396（42 変更 + 354 未追跡）
- status のシグネチャ: `6e346f32dd6f5db6`

## 7. 次段階（今回は未実施）

1. **upstream との重複 27 ファイルの個別比較**。ローカル変更 42 ファイルのうち 27 が main 側でも変更されている（`DeviceSettings.cpp`、`NoiseShaperLearner.*`、`AudioEngine.*`、`ConvolverProcessor.*`、`EpochDomain.h` など）。ファイル単位で「どちらを採るか」「両方か」を決める。
2. 統合は `4d476d5c` を基点にした一時 worktree で行い、main へは通常のマージか 3-way apply で入れる。保護ツリーは引き続き触らない。
3. 修正作業は独立 worktree で行う。Issue #265（OOM 分岐）と #266（CodeQL 3 件）が対象である。

## 8. 起票した Issue

| # | 内容 |
| ---: | --- |
| 265 | `NoiseShaperLearner` の OOM 失敗経路が到達不能（`create()` が例外を投げる） |
| 266 | CodeQL の自作コード 3 件（`snprintf` 2・workflow `permissions` 1） |

#252（S2 の長期再現性・ASIO 初期化）は従来どおり独立課題として維持する。オープン Issue は 3 件になった。

## 9. 参照

- 残課題の全体: `doc/work113/20261123-1815_残課題整理/report.md`
- 保全ブランチ: `wip/preserve-20261123`（commit `adfb16f6`）
- 復元物: `%TEMP%\opencode\wip236\`（`tracked_changes.patch`、`untracked_files.zip`、`status_record.json`）、Dropbox `/ConvoPeq-wip-preserve-20261123/`
