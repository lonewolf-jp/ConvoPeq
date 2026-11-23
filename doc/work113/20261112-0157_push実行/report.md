# push 実行報告 — domain tag 実装 commit `173e6995`

作業日時: 2026-11-12 01:56〜02:0x
対象: commit `173e69957b046290f7077797782f031ed8633fd0` を `origin/main` へ push
範囲: push のみ（merge・release・追加 commit・ソース変更は行っていない）

## 結論

push は成功した。`origin/main` は `c94d3767` から `173e6995` に進み、push した commit は 1 件だけである。ローカルの HEAD と作業ツリー（変更 34・未追跡 157・ステージ 0）は push 前後で変化していない。merge と release は行っていない。

ただし push 後の CI では、`audioengine-lint`、`list-compliance`、定期実行の `ISR Verification` の 3 件が失敗した（親 commit ではいずれも成功）。原因は commit に含めたファイルの記述で、ソース変更と追加 commit が禁じられているため未修正のまま報告する。詳細は「想定外事項」に記す。

## 実行内容

| 項目 | 値 |
| --- | --- |
| push 実行日時 | 2026-11-12 01:57（前確認 01:56:46、後確認 01:57:02） |
| push 元 branch | `main` |
| push 元 commit | `173e69957b046290f7077797782f031ed8633fd0` |
| push 先 remote | `origin`（`https://github.com/lonewolf-jp/ConvoPeq.git`） |
| push 先 branch | `main` |
| push 前の remote HEAD | `c94d376772ee22916f8350cdaa33f8d3787d16bd` |
| push 後の remote HEAD | `173e69957b046290f7077797782f031ed8633fd0` |
| push 対象 commit 数 | 1（`173e6995`、親 `c94d3767`、tree `a1ec450a3ee95b7ba607e170b55ee64da8b726d2`） |
| push 結果 | 成功（`c94d3767..173e6995  main -> main`、終了コード 0） |
| push 後のローカル HEAD | `173e6995`（変化なし） |

実行コマンドは `git push origin main` のみ。`--force` / `--force-with-lease` / rebase / reset / amend は使っていない。

## push 前の確認（指示 3 の 7 項目）

| # | 確認内容 | 実測値 | 判定 |
| --- | --- | --- | --- |
| 1 | 現在の branch が `main` | `main` | OK |
| 2 | HEAD が `173e6995` と一致 | `173e69957b046290f7077797782f031ed8633fd0` | OK |
| 3 | parent が `c94d3767` | `c94d376772ee22916f8350cdaa33f8d3787d16bd` | OK |
| 4 | `origin/main` の最新取得と push 先の確認 | `git fetch origin --prune` が差分なし。remote は `lonewolf-jp/ConvoPeq` の `main` | OK |
| 5 | `origin/main` が `c94d3767` を指す | `c94d376772ee22916f8350cdaa33f8d3787d16bd` | OK |
| 6 | 未反映 commit が `173e6995` の 1 件だけ | `git rev-list --count origin/main..HEAD` = 1、内容も `173e6995` のみ | OK |
| 7 | 未ステージ変更・未追跡が事前報告から変化なし | 変更 34・未追跡 157・ステージ 0（commit 報告時と同じ） | OK |

未コミットのまま保持したもの（push に含めていない）:

- `CMakeLists.txt`（`DomainTagTests` 以外の定義）
- `AGENTS.md`（commit 後に記述を直した分）
- commit 対象から除外した 32 ファイル
- 未追跡ファイル一式（dated な作業記録、分割版、別系統の新規ソースなど）

push の前後を通じて、これらの stage・commit・stash・削除・復元は行っていない。実行したのは `git fetch`、`git rev-parse`、`git status`、`git diff`、`git ls-files`、`git rev-list`、`git log`、`git push` と、本報告のファイル作成だけである。

## push 後の確認（指示 5 の 6 項目）

| # | 確認内容 | 実測値 | 判定 |
| --- | --- | --- | --- |
| 1 | `origin/main` が `173e6995` | `173e69957b046290f7077797782f031ed8633fd0` | OK |
| 2 | remote の commit ID がローカルと一致 | `git ls-remote origin refs/heads/main` = `173e69957b046290f7077797782f031ed8633fd0` | OK |
| 3 | push された commit が 1 件だけ | `git rev-list --count c94d3767..origin/main` = 1、`173e6995` | OK |
| 4 | ローカル HEAD が変化していない | `173e6995` のまま、`git status --porcelain --branch` は `## main...origin/main`（先行・遅延なし） | OK |
| 5 | 未コミット変更・未追跡が変化していない | 変更 34 件・未追跡 157 件が push 前と完全一致（一覧の差分なし） | OK |
| 6 | merge・release を実施していない | HEAD の親は 1 つ（merge commit ではない）、HEAD を指す tag なし、merge／release のコマンドは未実行 | OK |

未追跡は本報告の保存で 157 件から 163 件に増える（増分は `report.md`、`pre_state.txt`、`push_output.txt`、`post_state.txt`、`ci_status.txt`、`atomic_dotcall_local.txt` の 6 ファイル）。push 前後で比較した 157 件の中身は変わっていない。

## 想定外事項

### 1. branch protection の必須チェックを迂回して push された

remote が次の警告を返した。branch protection の必須チェックが未完了のまま push が受理されたことを示す。

```
remote: Bypassed rule violations for refs/heads/main:
remote: - Required status check "Windows MSVC / Release" is expected.
```

push 自体は成功しており（`c94d3767..173e6995  main -> main`、終了コード 0）、reject ではない。必須チェックに当たる `Windows MSVC Build` は push 後の実行で成功している（下記 CI）。

### 2. push した commit で CI の 2 チェックが失敗した

親 `c94d3767` では 8 workflow すべて成功していたので、この commit で新たに生じた失敗である。いずれもソースの修正が必要で、本指示はソース変更と追加 commit を禁じているため手を入れていない。

`audioengine-lint`（run 43063319811）: LINT-AE-011 が 2 件。

- `src/audioengine/ISRDebugRuntime.h:37` — `mutable HBVerifierRuntime hbVerifier_;`
- `src/audioengine/ISRRetireLatency.h:1021` — `thread_local std::uint32_t assigned = 0xFFFFFFFFu;`

規則は「`src/**` に `mutable` / `thread_local` を置かない」で、許容条件は `mutable` が mutex 型か NonRT 契約を書いた POD、`thread_local` は `NOLINT(thread-local)` と `RT-SAFE:` の併記。指摘された 2 行はどちらの条件も満たしていない。

`list-compliance`（run 43063320048）: `[3.1] Strict atomic dot-call scan failed` で 258 件。手元で `.github/scripts/check-src-atomic-dotcall.ps1` を実行しても同じ 258 件になり、内訳は次の 7 ファイル。すべて commit に含まれるファイルである。

| 件数 | ファイル |
| --- | --- |
| 206 | `src/audioengine/ISRRetireLatency.h` |
| 30 | `src/audioengine/WorldLifetimeRecorder.h` |
| 9 | `src/core/ReaderMeasureRegistry.h` |
| 9 | `src/core/ScopedEpochReader.h` |
| 2 | `src/audioengine/ISRDebugRuntime.cpp` |
| 1 | `src/core/EpochDomain.h` |
| 1 | `src/core/SnapshotCoordinator.h` |

規則の趣旨は `convo::consumeAtomic` / `publishAtomic` 等のヘルパー経由にすること。

### 3. 定期実行の `ISR Verification` が失敗した

push 後に走った定期実行（run 43073663375、2026-11-11T18:00:56Z 開始、event=schedule）が `isr-verify-proof-scope.ps1:41` で失敗した。メッセージは `DebugRuntime contract violation: Release short-circuit return not found`。

原因は commit が `src/audioengine/ISRDebugRuntime.cpp` の Release 分岐を変えたことにある。

- 変更前: `if (mode == "Release") { return; }` と `if (mode == "CI") { emitHBTrace(); }`
- 変更後: `if (mode == "Release" && ! isForceEvidenceEmission()) { return; }` と、条件なしの `emitHBTrace();`

ゲートは変更前の 2 つの形を文字列で要求するため、この変更で落ちる。`isForceEvidenceEmission()` は B-11 の実測証跡モード用で、commit に含めた `ISRDebugRuntime.cpp` を丸ごと入れた結果として混入した。

### 4. commit に B-11 系の記述が一部入っていた（範囲の濁り）

`ISRDebugRuntime.cpp` / `.h` は、commit が必要としていた `emitFinalViolationReport()`（A(D1) の測定）と、B-11 の強制証跡モード（`isForceEvidenceEmission`、`g_forceEvidenceEmission`、`mutable HBVerifierRuntime hbVerifier_`）が同じファイルに同居している。ファイル単位でしか分けなかったため、B-11 側も一緒に入った。

この混入は上記 3 のゲート失敗と、`audioengine-lint` の 1 件（`ISRDebugRuntime.h:37`）の直接原因である。行単位で切り分けていれば防げた可能性がある。次の修正では、この 2 ファイルを A(D1) 部分と B-11 部分に分けることを検討する必要がある。

### 5. `AGENTS.md` の記載が古いまま

`AGENTS.md` には「commit 済み `173e6995`（未 push）」と書かれたままである。本指示が作業ツリーの変更を禁じているため、push 済みへの書き換えは行っていない。次回のドキュメント作業で直す必要がある。

## CI

push により `173e6995` を対象とする workflow が起動した（2026-11-11T16:56:58Z＝JST 01:56:58）。2026-11-12 02:31:56 時点の状態は次のとおり。

| workflow | 状態 | 結果 | run |
| --- | --- | --- | --- |
| Windows MSVC Build | completed | success | 43063319686 |
| ISR Authority Compliance v6.4 | completed | success | 43063319902 |
| Push on main | completed | success | 43063318767 |
| audioengine-lint | completed | failure | 43063319811 |
| list-compliance | completed | failure | 43063320048 |
| ISR Verification（定期実行） | completed | failure | 43073663375 |
| Sanitizer CI | in_progress | - | 43063319846 |

branch protection が要求する `Windows MSVC / Release` に当たる `Windows MSVC Build` は成功した。`Push on main` も成功している。`Sanitizer CI` は 03:09:56 時点で実行中で、`Debug TSan (Clang, best-effort)` は成功、`RelWithDebInfo ASan (MSVC, /MD)` が未完了である。状態のスナップショットは `ci_status.txt` に保存した。

失敗した 3 件はいずれも commit に含まれるファイルの記述が原因で、親 `c94d3767` では同じ workflow が成功していた。修正にはソース変更と追加 commit が必要になるが、本指示はどちらも禁じているため行っていない。

## 証跡

- `pre_state.txt`: push 前の branch・HEAD・parent・origin/main・変更／未追跡の全一覧
- `push_output.txt`: `git push origin main` の出力と終了コード
- `post_state.txt`: push 後の remote HEAD・ls-remote・tree・parent 数・変更／未追跡の全一覧
- `ci_status.txt`: push 後の CI 状態スナップショット（2026-11-12 02:31:56）と失敗内容
- `atomic_dotcall_local.txt`: `check-src-atomic-dotcall.ps1` を手元で実行した結果（CI と同じ 258 件）
- Dropbox: `/ConvoPeq-173e6995-commit/push-20261112/` に本報告と証跡 5 件を追加
