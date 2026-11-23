# PR 作成の準備（承認待ち）

対象: `ci-fix/173e6995`（commit `b237f233f5ddaab9ef471bf2afe3e036ceb089b5`）
状態: PR は未作成（`gh pr list --head ci-fix/173e6995 --state all` が 0 件、CI run も 0 件）

## 実行場所

`C:\VSC_Project\ConvoPeq`（本文ファイルがある main 作業ツリー）。`--repo` と `--head` を明示するので、どのチェックアウトからでも実行できるが、`--body-file` はカレントディレクトリからの相対パスなので、このディレクトリで実行するのが確実。

## 承認後に実行するコマンド

1 行版（cmd / PowerShell どちらでも可）:

```
cd C:\VSC_Project\ConvoPeq
gh pr create --repo lonewolf-jp/ConvoPeq --base main --head ci-fix/173e6995 --title "fix(isr): restore the lint, atomic-access and proof-scope gates" --body-file "doc/work113/20261112-1300_PR草案/description.md"
```

PowerShell で複数行にする場合（行末はバッククォート）:

```
gh pr create --repo lonewolf-jp/ConvoPeq `
  --base main `
  --head ci-fix/173e6995 `
  --title "fix(isr): restore the lint, atomic-access and proof-scope gates" `
  --body-file "doc/work113/20261112-1300_PR草案/description.md"
```

- `--draft` は付けない想定。ドラフト PR にしたい場合は `--draft` を足す。
- 本文は `description.md` の内容をそのまま使う。

## 作成直後の確認

```
gh pr view <PR番号> --json number,url,state,isDraft,headRefName,headRefOid,baseRefName,baseRefOid
```

期待値:

- `headRefName` = `ci-fix/173e6995`
- `headRefOid` = `b237f233f5ddaab9ef471bf2afe3e036ceb089b5`
- `baseRefName` = `main`
- `state` = `OPEN`

## 作成後の CI

PR ができると、次の 5 系統が `pull_request` で起動する。

| workflow | 備考 |
| --- | --- |
| audioengine-lint | LINT-AE-011 を含む |
| list-compliance | atomic dot-call スキャンを含む |
| ISR Verification | proof-scope 検証を含む（standard tier） |
| Windows MSVC Build | Release ビルドと主要テスト |
| Sanitizer CI | ASan（TSan は best-effort） |

```
gh pr checks <PR番号> --watch
```

## PR 本文

`description.md` に保存した。内容は次のとおり。

- 背景: `173e6995` で赤くなった 3 つのゲート（LINT-AE-011、atomic dot-call 258、proof-scope）
- 変更: mutable 除去と thread_local の RT-SAFE 例外、atomic ヘルパー経由化と weak CAS ヘルパー、proof-scope の関数単位検証と字句走査、自己テスト 20 ケース・境界テスト 38 項目
- 検証: 静的検査 4 種、ISR standard tier、Release/ASan 57/57、Debug 既知 2 件、TSan 3/5 ターゲット警告 0
- 既知の限界: Remote CI 未実行（この PR が起動する）、TSan は部分的、Debug の 2 件は既存

## 承認との関係

- PR 作成の承認は、merge の承認とは別。
- merge・release・tag 作成・`main` への push は、PR 作成の承認に含まれない。
- `workflow_dispatch`（Windows MSVC Build / ISR Verification のみ）は、PR を作らない場合の代替。lint・list-compliance・Sanitizer CI は PR でしか動かない。
