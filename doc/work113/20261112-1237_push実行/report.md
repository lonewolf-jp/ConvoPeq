# CI 修正 commit の push 実行

作業日時: 2026-11-12 12:34〜12:37
承認: 指定 commit を指定ブランチへ通常 push することのみ（merge・release・tag・main への push は未承認）

## 結論

`b237f233f5ddaab9ef471bf2afe3e036ceb089b5` を `origin` の `ci-fix/173e6995` へ push した。新規ブランチとして作成され、force は使っていない。push 後のリモート HEAD とローカル HEAD は一致している。

## 実行前の確認

| # | 確認 | 実測 | 判定 |
| --- | --- | --- | --- |
| 1 | ブランチが `ci-fix/173e6995` | `ci-fix/173e6995` | OK |
| 2 | HEAD が `b237f233…` | `b237f233f5ddaab9ef471bf2afe3e036ceb089b5` | OK |
| 3 | 親が `173e6995…` | `173e69957b046290f7077797782f031ed8633fd0` | OK |
| 4 | worktree が clean | `git status --porcelain -uall` が 0 件 | OK |
| 5 | commit 差分がレビュー済みパッチと一致 | `cmp` で一致 | OK |
| 6 | push 先が `origin`／`ci-fix/173e6995` | `origin` は `github.com/lonewolf-jp/ConvoPeq.git` | OK |
| 7 | 既存履歴を上書きしない | `ls-remote` で当該ブランチは存在せず（新規作成） | OK |

## 実行コマンドと結果

```
git push origin HEAD:refs/heads/ci-fix/173e6995
```

```
To https://github.com/lonewolf-jp/ConvoPeq.git
 * [new branch]        HEAD -> ci-fix/173e6995
```

終了コード 0。`--force` と `--force-with-lease` は使っていない。

## push 後の確認

| 項目 | 値 |
| --- | --- |
| push 前のリモート HEAD | なし（新規ブランチ） |
| push 後のリモート HEAD（`ls-remote`） | `b237f233f5ddaab9ef471bf2afe3e036ceb089b5` |
| ローカル HEAD | `b237f233f5ddaab9ef471bf2afe3e036ceb089b5` |
| 一致 | 一致（remote-tracking も同値） |
| worktree | clean（0 件）、branch は `ci-fix/173e6995` |
| リモートとの差 | 0 commit（`ahead_of_remote=0`） |
| `origin/main` | `173e6995`（変化なし） |

補足: このリポジトリの fetch refspec は `main` のみ（`+refs/heads/main:refs/remotes/origin/main`）のため、push 後に `origin/ci-fix/173e6995` の remote-tracking ref は自動では作られない。確認のため明示的に fetch して `refs/remotes/origin/ci-fix/173e6995` を作成し、`ls-remote` の値と一致することを確認した。この fetch はローカルの参照を増やすだけで、リモートの状態は変えていない。

## 特記事項

GitHub は push 応答で PR 作成用の URL（`https://github.com/lonewolf-jp/ConvoPeq/pull/new/ci-fix/173e6995`）を案内してきたが、PR 作成は承認範囲外のため行っていない。

## 未実施（承認範囲外）

- merge、release、tag 作成
- `main` への push
- remote CI の再実行
- TSan 検証

## 証跡

- `push_output.txt`: `git push` の出力と終了コード
- `post_push.txt`: push 後の `ls-remote`・ローカル HEAD・remote-tracking・worktree 状態
- commit 側の証跡: `doc/work113/20261112-1222_CI修正commit/`
- Dropbox: `/ConvoPeq-b237f233-commit/push-20261112/`
