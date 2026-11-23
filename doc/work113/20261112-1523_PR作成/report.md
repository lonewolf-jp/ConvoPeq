# PR #207 の作成

作業日時: 2026-11-12 15:22
承認: `main` を base、`ci-fix/173e6995` を head とする PR の新規作成のみ（merge・release・tag・main への push は未承認）

## 結論

PR #207 を作成した。state は OPEN、draft ではない。head は `b237f233f5ddaab9ef471bf2afe3e036ceb089b5`、base は `main`（`173e6995`）で、指定どおりに作成されている。PR 作成により CI 6 本が起動した（こちらからの追加操作はしていない）。

## PR 情報

| 項目 | 値 |
| --- | --- |
| PR 番号 | 207 |
| URL | https://github.com/lonewolf-jp/ConvoPeq/pull/207 |
| タイトル | `fix(isr): restore the lint, atomic-access and proof-scope gates` |
| state | OPEN（draft ではない） |
| head | `ci-fix/173e6995` @ `b237f233f5ddaab9ef471bf2afe3e036ceb089b5` |
| base | `main` @ `173e69957b046290f7077797782f031ed8633fd0` |
| 作成日時 | 2026-11-12T06:22:10Z（JST 15:22） |

## 実行前の確認

- 当該 head の PR が 0 件（`gh pr list --head ci-fix/173e6995 --state all`）。
- リモートブランチが `b237f233…`（`git ls-remote`）。
- 本文ファイル `doc/work113/20261112-1300_PR草案/description.md` が存在。
- `gh auth status` が `lonewolf-jp` でログイン済み。

## 実行コマンド

```
cd C:\VSC_Project\ConvoPeq
gh pr create --repo lonewolf-jp/ConvoPeq --base main --head ci-fix/173e6995 \
  --title "fix(isr): restore the lint, atomic-access and proof-scope gates" \
  --body-file "doc/work113/20261112-1300_PR草案/description.md"
```

出力: `https://github.com/lonewolf-jp/ConvoPeq/pull/207`（終了コード 0）。

## 作成後の確認

`gh pr view 207 --json ...` の実測は `pr_view.json` のとおり。

- `headRefOid` = `b237f233f5ddaab9ef471bf2afe3e036ceb089b5`（期待どおり）
- `baseRefOid` = `173e69957b046290f7077797782f031ed8633fd0`（`main` の現在値）
- `state` = OPEN、`isDraft` = false

## PR で起動した CI（15:22 時点）

| workflow | 状態 |
| --- | --- |
| ISR Verification | in_progress |
| Windows MSVC Build | in_progress |
| audioengine-lint | in_progress |
| list-compliance | in_progress |
| ISR Authority Compliance v6.4 | queued |
| Sanitizer CI | in_progress |

PR 作成に伴い自動で起動したもので、こちらからの再実行・修正はしていない。結果の確認は別段階として扱う。

## 未実施（承認範囲外）

- CI 結果の判定・修正
- TSan の追加検証
- merge、release、tag 作成
- `main` への push

## 証跡

- `pr_create_output.txt`: `gh pr create` の出力
- `pr_view.json`: PR のメタデータ（head/base SHA を含む）
- `pr_verify.txt`: 作成直後の確認と CI run 一覧
- `pr_runs.json`: 起動した run の生データ
- 本文: `doc/work113/20261112-1300_PR草案/description.md`
- Dropbox: `/ConvoPeq-b237f233-commit/pr-207-20261112/`
