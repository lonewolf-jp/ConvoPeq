# 追加 commit `a05daf76` の CI 結果と CodeRabbit 再レビュー

作業日時: 2026-11-12 18:52〜19:05
対象: PR #207（head `a05daf76`）
状態: CI は 6/6 成功。CodeRabbit のチェックは pass だが、**新 commit に対して新規指摘 5 件**が出ている。修正は未実施（承認外）。

## 1. CI の最終結果（`a05daf76`）

| workflow | run ID | 結果 |
| --- | --- | --- |
| audioengine-lint | 43182860168 | success |
| list-compliance | 43182860161 | success |
| ISR Authority Compliance v6.4 | 43182860139 | success |
| ISR Verification | 43182860171 | success |
| Windows MSVC Build | 43182860149 | success |
| Sanitizer CI | 43182860154 | success |

18:52 に Sanitizer CI が success で完了し、6 系統すべてが `a05daf76` に対して成功した。`gh pr checks 207` の 17 行は 16 行が pass、1 行（`Windows MSVC / probe`）が dispatch 専用のため skipping で、失敗は無い。

PR は `OPEN`、base は `main`、`mergeable: MERGEABLE`、`mergeStateStatus: CLEAN`、head は `a05daf76eda07a6eed6ba2c14049cc1c2a5669f0`。

`b237f233` の 6/6 は前の commit の結果であり、ここでは使っていない。今回の 6/6 は `a05daf76` に対する結果である。

## 2. CodeRabbit 再レビュー

- チェック状態: `pass`（ブロッキング指摘なし）
- レビュー: 2026-11-12T08:31:24Z に `COMMENTED`（新 commit を対象）
- インライン指摘: **5 件、すべて `isr-verify-proof-scope.ps1`**

| # | 位置 | 重大度 | 指摘 |
| --- | --- | --- | --- |
| A | 213 | Major | `#if 0` で無効化された定義を先に置くと `Get-FunctionSpan` がそちらを選び、有効な定義にガードが無くても通る |
| B | 339 | Major | PowerShell の `-eq` / `-contains` は大文字小文字を区別しないため、`mode == "release"` でも通る |
| C | 404 | Minor | `Get-ConditionalReturns` が任意の深さの `if (...) { return; }` を拾うため、実行されないブロック内のガードでも通る |
| D | 411 | Major | Release 早期 return の後、`if (mode == "Release") { emitHBTrace(); }` のように Release 限定の分岐に呼び出しを置いても通る（CI / Debug では呼ばれない） |
| E | 437 | Major | `emitHBTrace()` から `emitTrace()` を削除しても、ガードが残っていればエラーにならない |

補助として PSScriptAnalyzer の `PSUseBOMForUnicodeEncodedFile` 警告が各指摘に付いている（後述）。

## 3. 5 件の再現（実測）

指摘が現在のコードでも成立するかを、検証器の複製と専用フィクスチャで確認した。**5 件すべて、不正な実装に対して検証器が exit 0（合格）を返した**。つまり指摘はすべて有効で、古い指摘ではない。

```
A-if0-disabled-definition      exit=0 expected=1 => MISSED (finding valid)
B-lowercase-release-literal    exit=0 expected=1 => MISSED (finding valid)
C-nested-guard                 exit=0 expected=1 => MISSED (finding valid)
D-trace-in-release-block       exit=0 expected=1 => MISSED (finding valid)
E-emitTrace-missing            exit=0 expected=1 => MISSED (finding valid)
```

各フィクスチャの内容:

- A: `#if 0` 内に正しい `emitCIArtifacts()`、その後に空の有効定義を置く
- B: `if (mode == "release" && ! isForceEvidenceEmission())`
- C: `if (false) { if (mode == "Release" && ...) { return; } }` の後に `emitHBTrace();`
- D: 正しい早期 return の後、`if (mode == "Release") { emitHBTrace(); }`
- E: `emitHBTrace()` にガードだけを残し `emitTrace()` を削除

正の対照（実ツリーの検証）は PASS のままで、既存の自己テスト 27/27 と字句テスト 44/44 も通る。つまり既存テストはこの 5 パターンを網羅していない。

## 4. 重大度と今回の変更との関連

5 件はいずれも今回 `a05daf76` で書き換えた `isr-verify-proof-scope.ps1` の内部にあり、変更そのものに関連する。A・B・D・E は「違反実装が合格してしまう」方向の穴で、検証器の目的（契約違反の検出）を損なう。C は同じ穴だが、実行されないブロックという前提が付くため Minor。

A と D は「定義・呼び出しの到達可能性」を静的テキストで近似していることの限界で、完全に対処するにはプリプロセッサ条件と制御フローの解析が要る。B・C・E は小さな修正で塞げる。

## 5. BOM 警告の扱い

`.github/scripts/*.ps1` は 200 ファイル中 197 が UTF-8 BOM なしで、BOM ありは 3 件だけである。PowerShell 7 は BOM なし UTF-8 を正しく読む（CI も green）。リポジトリの慣習に合わせて BOM なしを維持するのが妥当で、この警告は対応しない。

## 6. 判定と次の作業

merge の判断材料として、CI 6/6 成功は満たしているが、**CodeRabbit の新規指摘 5 件（うち Major 4 件）が残っている**。5 件はいずれも今回の変更ファイル内の実在する穴で、再現も取れている。したがって現時点では merge しない。

次に行うべきは 5 件の修正で、修正には次の承認が要る。

- B: `-eq` → `-ceq`、`-contains` → `-ccontains`、小文字リテラルの回帰フィクスチャ追加
- C: ガードは関数本体のトップレベル文だけを数える
- E: `emitTrace()` が無い場合にエラーを追加
- A: `#if 0` などで無効化された定義を除外する（無効ブロックの除去、または曖昧な定義を拒否）
- D: Release 早期 return の後の到達可能性を検査する（Release 限定分岐に閉じた呼び出しを拒否）

A と D は検証器の設計に踏み込むため、実装方針を決めたうえで着手する。修正後は自己テストと字句テストに回帰フィクスチャを追加し、今回の再現スクリプト（`repro_probe.ps1`）で 5 件が検出されることを確認する。

## 7. 未実施（承認範囲外）

修正、commit、push、merge、release、tag 作成、`main` への push、`workflow_dispatch`、CI の手動再実行、TSan 追加検証。

## 8. 証跡

- `coderabbit_new_findings.txt`: 新規指摘 5 件の本文
- `repro_probe.ps1` / `repro_probe.log`: 5 件の再現スクリプトと結果
- `runs_final.json`: `a05daf76` の run 一覧（6 件すべて success）
- `pr_checks_final.txt`: `gh pr checks 207` の全 19 行
- 前段の証跡: `doc/work113/20261112-1710_Coderabbit対応/`（修正パッチ、自己テスト 27/27、字句テスト 44/44、push 照合）
- Dropbox: `/ConvoPeq-b237f233-commit/coderabbit-rereview-20261112/`
