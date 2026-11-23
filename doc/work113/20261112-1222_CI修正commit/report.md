# CI 修正差分の commit 実行

作業日時: 2026-11-12 12:22
承認: Owner（commit のみ。push・merge・release・tag は未承認）

## 結論

承認された範囲（12 ファイル、+1,705 / −324）を、独立 worktree `C:\VSC_Project\ConvoPeq-ci-fix` の branch `ci-fix/173e6995` に commit した。push はしていない。

## commit 情報

| 項目 | 値 |
| --- | --- |
| commit SHA | `b237f233f5ddaab9ef471bf2afe3e036ceb089b5` |
| parent SHA | `173e69957b046290f7077797782f031ed8633fd0` |
| branch | `ci-fix/173e6995` |
| subject | `fix(isr): restore the lint, atomic-access and proof-scope gates` |
| 変更ファイル | 12（変更 10・新規 2） |
| 差分 | +1,705 / −324 |

新規ファイルは `.github/scripts/test-isr-verify-proof-scope.ps1` と `.github/scripts/test-isr-verify-proof-scope-lexer.ps1`。

## 実行前の確認

- branch が `ci-fix/173e6995`、HEAD が `173e6995`、parent が `c94d3767` であること。
- 作業ツリーの差分が 12 ファイル +1,705 / −324 で、レビュー済み証跡 `ci-fix.patch` とバイト一致すること（`cmp`）。
- stage した 12 ファイルがその差分と一致し、未ステージの残りが 0 件であること。

## commit 後の状態

| 項目 | 値 |
| --- | --- |
| HEAD | `b237f233f5ddaab9ef471bf2afe3e036ceb089b5` |
| branch | `ci-fix/173e6995`（upstream なし） |
| worktree status | クリーン（`git status --porcelain -uall` が 0 件） |
| origin/main | `173e6995`（push していない） |
| commit の差分 | レビュー済み `ci-fix.patch` とバイト一致（`git diff HEAD^ HEAD` を `cmp`） |

## main 作業ツリー

| 項目 | 値 |
| --- | --- |
| HEAD / branch | `173e6995` / `main` |
| 変更 | 34 件（commit 前と同じ） |
| 未追跡 | 184 件（commit 前と同じ。増分は本報告フォルダ） |
| ステージ | 0 件 |

## 未実施（承認範囲外）

- push（`origin/main` は `173e6995` のまま）
- merge、release、tag 作成
- remote CI の再実行
- TSan の検証

## 証跡

- `commit-b237f233.patch`: commit の全差分
- `commit-stat.txt`: `git show --stat HEAD`
- `commit-info.txt`: SHA・parent・author・date・subject
- レビュー時の証跡一式: `doc/work113/20261112-0922_CI修正/`
- Dropbox: `/ConvoPeq-173e6995-commit/ci-fix-20261112/`（レビュー時の証跡 34 ファイル）
