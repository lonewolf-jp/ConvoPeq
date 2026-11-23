# CodeRabbit 新規指摘 5 件の修正と条件コンパイル下の fail-closed

作業日時: 2026-11-12 19:00〜21:20
対象: PR #207（head `a05daf76`）に対する CodeRabbit 再レビューの指摘 5 件、および追加調査で見つかった条件コンパイル下の穴
状態: worktree に修正を適用済み・**未 commit**。C++ は変更していない。

## 1. CodeRabbit の指摘 5 件

### A. 無効化された定義の誤選択（Major）

`#if 0` などで除外される定義を、有効な定義として選んでいた。プリプロセッサ条件を走査する処理を追加した。扱いは 3 段階である。

- `#if 0` / `#if false` の領域は**無効**として空白化する（文字位置は変えない）
- `#if defined(X)` のように条件次第で含まれる領域は**曖昧**として記録する
- `#else` / `#elif` は直前の分岐の結果を見て有効・無効・曖昧を決める

関数本体の解決は `Resolve-FunctionBody` に集約した。有効な定義が 1 つなら本体を返す。0 なら未検出。条件次第の領域にある定義と、有効な定義が複数ある場合は**曖昧として FAIL** する。

### B. 大文字小文字を区別しない比較（Major）

`mode == "release"` でもガードとして通っていた。`Test-ReleaseGuardCondition` の比較を `-ceq` / `-ccontains` に変え、`||` の検出も `-cmatch` にした。

### C. トップレベル以外のガード（Minor）

`if (false) { if (mode == "Release") { return; } }` のように、実行されないブロック内のガードでも通っていた。`Get-ConditionalReturns` が波括弧深度を返すようにし、ガードは**関数本体のトップレベル文（深度 1）だけ**を数える。入れ子のガードしか無い場合は専用メッセージで FAIL する。

### D. Release 限定分岐内の呼び出し（Major）

`emitHBTrace()` を `if (mode == "Release") { ... }` の中に置くと、CI / Debug では呼ばれないのに通っていた。呼び出しを囲むブロックの条件式を調べ、次の 2 つを拒否する。

- `mode == "Release"` を含み、`isForceEvidenceEmission` を持たない条件（Release でしか成立しない）
- `if (false)` / `if (0)` のような定数で偽の条件

裸のブロック `{ ... }` の中の呼び出しは無条件に実行される。これは従来どおり受け付ける（実コードの書き方も壊さない）。`if (mode == "CI")` のような Release 以外の分岐も受け付ける。

### E. `emitTrace()` 削除の未検出（Major）

`emitHBTrace()` から `emitTrace()` を削除しても、ガードが残っていれば合格していた。本体に `emitTrace(` が無い場合は FAIL とする。

## 2. 条件コンパイル下の必須ガード・呼び出し（追加の fail-closed）

追加調査で、関数本体の内側にある `#if defined(...)` の中のガードや呼び出しが、まだ合格することを実測した。指摘 A と同じ原則（曖昧なら FAIL）で塞いだ。

対象は 4 種類で、いずれも「条件コンパイル下にしか無い場合は FAIL」とした。

| 対象 | 追加したエラー |
| --- | --- |
| `emitCIArtifacts()` の Release 早期 return | `... is inside conditional compilation; the guard cannot be guaranteed` |
| `emitCIArtifacts()` の `emitHBTrace();` | `... calls emitHBTrace() only inside conditional compilation` |
| `emitHBTrace()` の `emitTrace(...)` | `... emits the trace only inside conditional compilation` |
| `emitFinalViolationReport()` の `emitViolationReport(...)` | `... emits the violation report only inside conditional compilation` |

条件コンパイル下のガードが**別に**あり、無条件の正しいガードもある場合は合格とする（過剰な拒否を避ける）。`#if 0` で無効化されたガードは空白化で消えるため、ガード不在として FAIL する。

## 3. 検証結果

| 検査 | 結果 |
| --- | --- |
| 実ツリーの proof-scope | PASS |
| 自己テスト | **42/42**（27 から 15 件追加） |
| 字句走査の境界テスト | **44/44** |
| audioengine-lint | PASS |
| atomic dot-call | 0 件 |
| ISR standard tier | 完走（PASS） |
| authority inventory 生成 | 成功 |
| 実行時間 | 0.74〜0.75 秒（修正前 0.74 秒） |

追加した回帰フィクスチャは 15 件。

| フィクスチャ | 期待 | 内容 |
| --- | --- | --- |
| `if0-disabled-definition-before-active` | 拒否 | `#if 0` の有効な定義が先、有効な定義はガード無し |
| `release-literal-lowercase` | 拒否 | `mode == "release"` |
| `nested-release-guard` | 拒否 | `if (false) { if (mode == "Release") ... }` |
| `hbtrace-inside-release-block` | 拒否 | `if (mode == "Release") { emitHBTrace(); }` |
| `emittrace-call-missing` | 拒否 | ガードはあるが `emitTrace()` が無い |
| `if0-disabled-bad-definition` | 許可 | `#if 0` 内の不正な定義は無視される（正例） |
| `hbtrace-in-bare-block` | 許可 | 裸のブロック内の呼び出し（正例） |
| `definition-under-conditional-compilation` | 拒否 | `#if defined(...)` で定義が分岐し、有効な定義を決められない |
| `guard-inside-conditional-compilation` | 拒否 | ガードが `#if defined(...)` の中 |
| `hbtrace-inside-conditional-compilation` | 拒否 | `emitHBTrace();` が `#if defined(...)` の中 |
| `emittrace-inside-conditional-compilation` | 拒否 | `emitTrace(...)` が `#if defined(...)` の中 |
| `violation-report-inside-conditional-compilation` | 拒否 | `emitViolationReport(...)` が `#if defined(...)` の中 |
| `guard-disabled-by-if0` | 拒否 | ガードが `#if 0` で無効 |
| `conditional-guard-with-unconditional-guard` | 許可 | 条件付きガードと無条件ガードが併存（正例） |
| `conditional-compilation-elsewhere` | 許可 | 契約と無関係な関数が `#if defined(...)` の中（正例） |

## 4. 修正前の検証器との識別力

修正前（`a05daf76`）の検証器に同じ 42 件を当てると、新規 15 件のうち **12 件**が期待と逆になった。今回の修正が無ければ通ってしまう（または正例を落とす）ことを実測している。

```
[FAIL] if0-disabled-definition-before-active    expected=reject exit=0
[FAIL] release-literal-lowercase                expected=reject exit=0
[FAIL] nested-release-guard                     expected=reject exit=0
[FAIL] hbtrace-inside-release-block             expected=reject exit=0
[FAIL] emittrace-call-missing                   expected=reject exit=0
[FAIL] if0-disabled-bad-definition              expected=accept exit=1
[FAIL] definition-under-conditional-compilation expected=reject exit=0
[FAIL] guard-inside-conditional-compilation     expected=reject exit=0
[FAIL] hbtrace-inside-conditional-compilation   expected=reject exit=0
[FAIL] emittrace-inside-conditional-compilation expected=reject exit=0
[FAIL] violation-report-inside-conditional-compilation expected=reject exit=0
[FAIL] guard-disabled-by-if0                    expected=reject exit=0
```

残る 3 件は正例（`hbtrace-in-bare-block`、`conditional-guard-with-unconditional-guard`、`conditional-compilation-elsewhere`）で、新旧どちらでも通る。

指摘 5 件を個別に再現する probe では、修正前が 5 件すべて `exit=0`（見逃し）、修正後が 5 件すべて `exit=1`（検出）になった。

## 5. 正例の維持

実ツリー（B-11 実装）、旧 R25 実装（`if (mode == "CI")` の中で trace）、B-11、括弧付きの条件、裸のブロック、条件付きガードと無条件ガードの併存、無関係な条件コンパイル。いずれも合格する。

## 6. 変更ファイル

2 ファイル +958 / −78（未 commit）。

- `.github/scripts/isr-verify-proof-scope.ps1`: プリプロセッサ走査、定義の一意性、深度によるトップレベル判定、大文字小文字の区別、呼び出しの到達可能性、`emitTrace()` の必須化、本体内部の条件コンパイル下の必須項目を拒否
- `.github/scripts/test-isr-verify-proof-scope.ps1`: 回帰フィクスチャ 15 件

C++ は変更していないため、ビルドと ctest は再実行していない（`a05daf76` の CI 結果がそのまま有効）。

## 7. 最終差分レビュー

2 ファイルの差分を通読した。追加の欠陥は見つからなかった。

- 追加した関数（12 個）はすべて呼び出し箇所がある。
- 置き換えた `Get-FunctionSpan` / `Get-FunctionBody` / `Get-FunctionBodyMasked` は、`.github/scripts` の他のスクリプトから参照されていない（`grep` で 0 件）。
- `Parser::ParseFile` で 2 ファイルとも構文エラーなし。
- 実行時間は 0.74〜0.75 秒（修正前 0.74 秒）。
- 不具合時のメッセージは、未検出・条件コンパイル下（定義／ガード／呼び出し）・重複定義・入れ子ガード・Release 限定呼び出し・定数偽を区別している。

## 8. 保証範囲

今回の検証器が何を検出し、何を検出しないかを実測した。

| 記述 | 判定 |
| --- | --- |
| `#if 0` の中のガード | 検出（無効として無視） |
| `#if 0` の中の `emitHBTrace();` | 検出（無効として無視） |
| `#if defined(...)` の中のガード | 検出（追加対応） |
| `#if defined(...)` の中の `emitHBTrace();` | 検出（追加対応） |
| `#if defined(...)` の中の `emitTrace(...)` | 検出（追加対応） |
| `#if defined(...)` の中の `emitViolationReport(...)` | 検出（追加対応） |
| `if (isForceEvidenceEmission()) { emitHBTrace(); }` | **未検出（保証外。別途の設計判断）** |
| `ISREvidenceExporter.cpp` の proofLevel 判定を `#if defined(...)` で包む | **未検出（保証外。別途の設計判断）** |

実測ログ:

```
F-guard-inside-conditional-block       exit=1 expected=1 => detected
G-guard-disabled-by-if0                exit=1 expected=1 => detected
H-trace-inside-conditional-block       exit=1 expected=1 => detected
I-trace-disabled-by-if0                exit=1 expected=1 => detected
J-trace-inside-force-branch            exit=0 expected=1 => accepted (documented limit)
exporter proofLevel mapping inside #if defined(...) => exit=0 accepted (documented limit)
```

強制条件だけの分岐（J）は、検証器が保証すべき実行条件の定義に関わるため、指示どおり別途の設計判断とした。exporter 側の `proofLevel` 判定と `isRelease` ブロックは、今回の対象（関数本体のガード・呼び出し）ではないため手を入れていない。同じ fail-closed を適用するなら、`ISREvidenceExporter.cpp` の 6 つの構造チェックにも条件コンパイルの判定を足すことになる。

## 9. 残存課題

1. **強制条件だけの分岐**（§8 の J）は未検出。想定する実行条件をどこまで保証するかの設計判断が要る。
2. **exporter 側の構造チェックは条件コンパイルを見ていない**（§8）。同じ fail-closed を適用するかは別途判断。
3. **自己テストは CI に接続されていない。** 2 つのテストスクリプトはどの workflow からも呼ばれておらず、開発者が手で実行する前提。今回の承認範囲外。
4. **A は条件コンパイル下の定義を一律に拒否する。** ビルド構成で定義を切り替える書き方には厳しいが、指示（曖昧なら FAIL）に沿った挙動。
5. **PSScriptAnalyzer の BOM 警告は未対応。** `.github/scripts/*.ps1` は 200 件中 197 件が BOM なしで、リポジトリの慣習に合わせた。
6. TSan 追加検証、`workflow_dispatch`、CI 手動再実行は行っていない。

## 10. 状態と次の作業

worktree `C:\VSC_Project\ConvoPeq-ci-fix` は 2 ファイル変更・未 commit（HEAD は `a05daf76`）。PR #207 は変更していない。

追加 commit と push は別途承認を得てから行う。その後の PR 再 CI 6 系統と CodeRabbit 再レビュー、merge の判断も別途。

## 11. 証跡

- `cr5-fix.patch`、`diffstat.txt`: 差分
- `selftest_42.log`、`lexer_44.log`、`proof_scope_real_tree.log`: 合格ログ
- `probe.ps1` / `probe_after_fix.log`、`probe_before_fix.ps1` / `probe_before_fix.log`: 指摘 5 件の再現（修正前・修正後）
- `scope_probe.ps1` / `scope_probe.log`: 条件コンパイル下の検出（F〜J）
- `exporter_scope_probe.ps1` / `exporter_scope_probe.log`: exporter 側の未検出の実測
- `old_verifier_discrimination.log`: 修正前の検証器で新規フィクスチャを実行した結果
- `isr_tier_standard.log`、`authority_inventory.log`、`lint_audioengine.log`、`lint_atomic_dotcall.log`
- Dropbox: `/ConvoPeq-b237f233-commit/cr5-fix-20261112/`
