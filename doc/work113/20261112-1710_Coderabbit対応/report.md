# CodeRabbit レビュー指摘への対応（PR #207）

作業日時: 2026-11-12 16:50〜17:15
対象: PR #207 の CodeRabbit レビュー 5 件（`isr-verify-proof-scope.ps1` 4 件、境界テスト 1 件）
状態: 修正は worktree に適用済み・**未 commit**。PR は `b237f233` のまま。

## 前提：PR #207 の CI は全件成功

PR 作成で起動した 6 workflow はすべて success だった。

| workflow | 結果 | run |
| --- | --- | --- |
| audioengine-lint | success | 43167657499 |
| list-compliance | success | 43167657385 |
| ISR Authority Compliance v6.4 | success | 43167657336 |
| ISR Verification | success | 43167657403 |
| Windows MSVC Build | success | 43167657517 |
| Sanitizer CI（TSan + ASan） | success | 43167657749 |

PR は `mergeable: MERGEABLE`、`mergeStateStatus: CLEAN`。ただし後述の CodeRabbit 指摘に対応した修正が未反映のため、この状態で merge はしていない。

## 指摘と対応

| # | 指摘 | 対応 |
| --- | --- | --- |
| 1 | 行コメントの行末バックスラッシュ（C++ の行連結）を考慮していない。`// ... \` の次行がコードとして扱われ、コメントアウトされたガードが有効に見える | `//` の走査で、改行直前がバックスラッシュなら次行へ継続するようにした（LF / CRLF 両方） |
| 2 | `emitCIArtifacts()` / `emitHBTrace()` のチェックが文字列内容を残したテキストを見ており、文字列に `emitHBTrace();` や `if (mode == "Release") { return; }` を書くと通ってしまう | 呼び出しの有無は文字列を伏せた本文で判定。条件式は文字列を伏せた本文で構造を探し、条件テキストだけ元の本文から取り出す。関数シグネチャの探索も文字列範囲の外に限定した |
| 3 | `if (mode == "Release" \|\| !isForceEvidenceEmission())` でも部分一致で通る（強制時にも抑制してしまう） | 条件式を正規化して `&&` で分割し、項が `mode == "Release"` と `! isForceEvidenceEmission()` の 2 つだけであることを要求。`\|\|`、全体の `!` 反転（`!(mode == "Release")`）、3 項以上は拒否 |
| 4 | `emitHBTrace()` 内で `emitTrace()` が Release 抑制より前でも通る | 最初の Release return より後に `emitTrace(` があることを要求した |
| 5 | 境界テストの CRLF 変異入力にバックスラッシュが無く、変異が効いていない（両者が同じ結果になる） | 入力にバックスラッシュを入れ、変異前に「正しい走査器がその入力を受け入れる」基準線を確認するようにした |

## 検証

| 検査 | 結果 |
| --- | --- |
| `isr-verify-proof-scope.ps1`（実ツリー） | PASS |
| 自己テスト | **27/27**（20 から 7 件追加） |
| 字句走査の境界テスト | **44/44**（38 から 6 件追加） |
| audioengine-lint | PASS |
| atomic dot-call | 0 件 |
| ISR standard tier | 完走 |

追加した自己テストのうち 4 件は、**修正前の検証器（`b237f233` の版）では FAIL する**ことを実測した。つまり今回の修正がなければ通ってしまう実装を、テストが検出できる。

```
[FAIL] release-guard-with-or                    expected=reject exit=0
[FAIL] hbtrace-call-only-inside-string          expected=reject exit=0
[FAIL] hbtrace-emit-before-guard                expected=reject exit=0
[FAIL] release-guard-commented-out-with-continuation expected=reject exit=0
[PASS] comment-continuation-without-hiding-guard expected=accept exit=0
```

残る 1 件（`comment-continuation-without-hiding-guard`）は正の対照で、新旧どちらでも通る。

## 差分

3 ファイル +353 / −29（未 commit）。`code-review-response.patch` に全差分を保存した。

- `.github/scripts/isr-verify-proof-scope.ps1`: コメント継続、文字列を伏せた判定、条件式の構造検証、`emitTrace` の順序検証
- `.github/scripts/test-isr-verify-proof-scope.ps1`: 回帰フィクスチャ 7 件（`||`、`!(mode == "Release")`、文字列内の呼び出し、順序、コメント継続の 2 件、括弧付きの正例）
- `.github/scripts/test-isr-verify-proof-scope-lexer.ps1`: コメント継続 3 件（LF / CRLF / 継続なし）、CRLF 変異の基準線

## 承認後に実行する手順（準備済み）

現時点では未実行。承認文面を受領後に次を実行する。

1. 差分の最終確認: `git -C C:\VSC_Project\ConvoPeq-ci-fix diff --stat` が 3 ファイル +353 / −29、`code-review-response.patch` と一致すること。
2. `git add` で 3 ファイルを stage し、`git diff --cached --stat` が同じ値であることを確認。
3. commit（メッセージ案）:

   ```
   fix(isr): close the proof-scope verifier gaps found in review

   - continue line comments across a trailing backslash (C++ line splicing)
   - judge call presence and guards on the string-masked body, and skip
     signatures that appear inside string literals
   - require the Release guard to be exactly
     `mode == "Release" && ! isForceEvidenceEmission()` (two terms, no `||`,
     no whole-condition negation); accept optional parentheses
   - require emitHBTrace() to emit after its Release suppression
   - fix the CRLF mutation input in the lexer test and assert the baseline
   ```

4. SHA と parent を記録: `git rev-parse HEAD`、`git rev-parse HEAD^`。
5. push: `git -C C:\VSC_Project\ConvoPeq-ci-fix push origin HEAD:refs/heads/ci-fix/173e6995`（force なし）。
6. 一致確認: `git ls-remote origin refs/heads/ci-fix/173e6995` と `git rev-parse HEAD`、`gh pr view 207 --json headRefOid`。
7. PR 再 CI の起動確認: `gh run list --branch ci-fix/173e6995 --limit 10`（新 commit の 6 系統）。

## 未実施

- commit と push（追加 commit は未承認のため）
- PR 上の再 CI
- merge、release、tag 作成

## 証跡

- `coderabbit_findings.txt`: 指摘 5 件の本文
- `code-review-response.patch`、`diffstat.txt`: 修正差分
- `selftest_25.log`、`lexer_44.log`、`proof_scope.log`、`isr_tier_standard.log`: 再実行ログ
- `old_verifier_discrimination.txt`、`old_verifier_selftest_full.log`: 修正前検証器での識別力の実測
- Dropbox: `/ConvoPeq-b237f233-commit/coderabbit-fix-20261112/`
