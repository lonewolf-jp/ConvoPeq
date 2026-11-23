# PR #207 レビュー・マージ報告

- 対象: https://github.com/lonewolf-jp/ConvoPeq/pull/207（`ci-fix/173e6995`）
- 目的: ISR の lint・Atomic Access・Proof Scope の検証ゲート復旧
- 実施日時: 2026-11-14 08:50〜（JST）
- 種別: レビュー指摘の修正・検証・main へのマージ

---

## 1. 結論

PR #207 は、レビュー指摘 **12 件**をすべて修正し、必須 CI を全件成功させたうえで
`main` へマージした。マージ前に `main`（`862565a3`）を取り込んで競合がないことを確認し、
統合後のツリーで 5 つのゲート・Release ビルド・ctest を実測した。

| 項目 | 値 |
| --- | --- |
| PR ブランチ | `ci-fix/173e6995` |
| PR の元 HEAD | `a05daf76eda07a6eed6ba2c14049cc1c2a5669f0` |
| 追加した修正コミット | `40b20757`（レビュー 1 回目）／`821506ca`・`b8a8607b`（Issue #208・#209）／`6f0b55fb`・`74e9aab0`（2 回目）／`25d68e9c`（3 回目） |
| main 取り込みマージ | `743c1972381483229f7e8b8e9f5bda8887abd5a7`（親 = `40b20757` + `862565a3`） |
| main の変更前 HEAD | `862565a319f839d73db306eaaea5e3919d709c14` |
| main の変更後 HEAD | `6e099932bcb4f122a72526f1c9786eecd7fc08b6` |
| マージコミット | `6e099932bcb4f122a72526f1c9786eecd7fc08b6` |
| 変更ファイル（PR 全体） | 12 件 |
| CI | PR head `25d68e9c`: success 14・skipped 2・failure 0／マージ後 `main` `6e099932`: 6/6 success |

---

## 2. 事前確認

| # | 確認項目 | 結果 |
| --- | --- | --- |
| 1 | `main` の最新 HEAD | `862565a3`（C3-D・C9-A・改名の統合後） |
| 2 | PR の最新 HEAD | `a05daf76`（2026-11-12 08:31 更新） |
| 3 | PR の全変更ファイル | 12 件（下記） |
| 4 | `main` との競合・重複 | GitHub の判定は `MERGEABLE` / `CLEAN`。ローカルで `git merge main` を実行しても競合なし。重複は `src/core/EpochDomain.h` の 1 行（PR は CAS の置換、main は `RetireHeadSnapshot` 追加）で、自動マージ後に両方が共存することを確認 |
| 5 | レビュー指摘 | CodeRabbit の指摘は合計 14 件。3 件は `a05daf7` で解決済み、**残り 11 件を本 PR で修正**（§3・§3.2） |
| 6 | CI（PR 作成時点） | 17 チェックすべて success（`a05daf76` / ベース `173e6995`） |

PR の変更ファイル:

| 層 | ファイル |
| --- | --- |
| ゲート | `.github/scripts/isr-verify-proof-scope.ps1`（+450/−29） |
| ゲート | `.github/scripts/test-isr-verify-proof-scope.ps1`（新規 1,018 行） |
| ゲート | `.github/scripts/test-isr-verify-proof-scope-lexer.ps1`（新規 229 行） |
| Atomic Access | `src/audioengine/AtomicAccess.h`（+21/−3） |
| Atomic Access | `src/audioengine/ISRDebugRuntime.cpp` / `.h` |
| Atomic Access | `src/audioengine/ISRRetireLatency.h`（+234/−224） |
| Atomic Access | `src/audioengine/WorldLifetimeRecorder.h`（+38/−36） |
| Atomic Access | `src/core/EpochDomain.h`（+1/−1） |
| Atomic Access | `src/core/ReaderMeasureRegistry.h` / `ScopedEpochReader.h` / `SnapshotCoordinator.h` |

---

## 3. レビュー指摘の扱い

CodeRabbit の指摘は 10 件（3 件は `a05daf7` で解決済みの表示あり）。残る **7 件**が未解決だったため、
すべてコードで修正した（「未解決のままマージしない」という指示に従う）。

| # | 指摘 | 重大度 | 対応 | 検証フィクスチャ |
| --- | --- | --- | --- | --- |
| 1 | 呼び出し・順序の検査が文字列リテラルに一致し得る | Major | `a05daf7`（伏せ字本文で照合、署名探索で文字列範囲を除外） | `release-guard-only-inside-string` / `hbtrace-call-only-inside-string` |
| 2 | Release ガードの真偽関係を検証していない（`&&` → `\|\|`） | Major | `a05daf7`（`\|\|` を拒否、項は 2 個に限定） | `release-guard-with-or` |
| 3 | `#if 0` で無効化された定義を本体として選ぶ | Major | **`40b20757`**（プリプロセッサ走査を追加し、無効領域を空白化。条件次第の定義は「曖昧」として拒否） | `if0-disabled-definition-before-active` / `definition-under-conditional-compilation` / `if0-disabled-bad-definition`（受理） |
| 4 | Release リテラルの比較が大文字小文字を無視する | Major | **`40b20757`**（`-ceq` / `-ccontains`） | `release-literal-lowercase` |
| 5 | 入れ子の Release ガードも数えてしまう | Minor | **`40b20757`**（関数本体内の波括弧深さ 1 のみ許可） | `nested-release-guard` |
| 6 | Release 専用ブロック内の `emitHBTrace()` でも通る | Major | **`40b20757`**（呼び出しを囲む条件を解析し、Release 専用条件・定数偽条件を拒否） | `hbtrace-inside-release-block` / `hbtrace-in-bare-block`（受理） |
| 7 | `emitTrace()` が無くても成功する | Major | **`40b20757`**（呼び出しが無ければエラー） | `emittrace-call-missing` |

条件コンパイル下の必須要素（ガード・trace 呼び出し・違反レポート）も、どのビルドで有効かを
決められないため拒否する。受理側の対照（`conditional-guard-with-unconditional-guard` /
`conditional-compilation-elsewhere`）も用意し、過剰拒否にならないことを固定した。


### 3.2 2 回目のレビュー（2026-11-14・追加 4 件）

修正を push した後に CodeRabbit が再レビューし、4 件の指摘が追加された。いずれも未解決のまま
マージしない方針に従って修正した。

| # | 指摘 | 重大度 | 対応 | 検証フィクスチャ |
| --- | --- | --- | --- | --- |
| 8 | `#if 1 && 0` を有効分岐と誤認する（`^1+\b` が式の先頭に一致） | Major | 定数の判定を完全一致（`^1+$` / `-ceq 'true'`）に変更。式は `unknown` として扱う | `if-expression-not-constant` |
| 9 | ブロックコメント内の `#if 0` を指令として扱い、実在する trace 呼び出しを空白化する | Major | 指令の走査を、コメントと文字列を空白化した同じ長さの伏せ字上で行う（位置は不変） | `directive-inside-block-comment` |
| 10 | `#define` の置換リストがガード・呼び出しとして数えられる | Major | `Get-DirectiveBlankedText` を追加し、指令行（行継続を含む）を文・呼び出しの走査から除外 | `guard-macro-unused` |
| 11 | AC1 の `allDrained` が削除済みの `retireBacklog == 0` を要求している | Minor | 述語に合わせて当該項を削除し、`retireBacklog` が非 0 でも述語とミラーが一致するシナリオを追加 | テスト内シナリオ (4) |
| 12 | 文字列の伏せ字で末尾のバックスラッシュが消え、`#define` の継続行が空白化されない | Minor | 行構造と継続の判定を元テキスト（`-Reference`）で行い、同じ範囲を伏せ字側で空白化 | `continued-macro-line-survives`（ミューテーションで検出力を確認） |

#11 は、以前に「独立課題（未承認）」として起票していた AC1 ミラー乖離と同じ箇所である。
今回はレビュー指摘として未解決のままにできないため、1 行の削除とコメント更新、および
非 0 での一致確認を同じ PR に含めた（`src/tests/ISRSemanticValidationTests.cpp`）。

Proof Scope self-test は 46/46（4 件追加）。Lexer boundary test も全チェック通過。

---

## 4. 変更内容

### 4.1 追加コミット `40b20757`

`.github/scripts/isr-verify-proof-scope.ps1` と `.github/scripts/test-isr-verify-proof-scope.ps1`
の 2 ファイル（+958/−78）。

- プリプロセッサ走査 `Get-PreprocessorRegions`（`#if 0` / `#if 1` / `#else` / `#elif` / `#ifdef`）を追加し、
  無効領域を空白化したうえで関数本体を解決する。
- `Get-FunctionSpans` → `Resolve-FunctionBody` に置き換え、定義が 1 つに定まらない場合
  （`conditional` / `ambiguous`）は理由付きで拒否する。
- `Get-BraceFrames` / `Get-ConditionalFramesContaining` / `Test-ReleaseOnlyCondition` /
  `Test-NeverTrueCondition` を追加し、呼び出しの到達性を検査する。
- `Get-ConditionalReturns` に `Depth` を追加し、トップレベル文のガードだけを数える。
- 文字列リテラルの比較を `-ceq` / `-ccontains` に変更。
- フィクスチャを 15 件追加（27 → 42 件）。

### 4.2 2 回目レビュー対応コミット

| コミット | 内容 |
| --- | --- |
| `6f0b55fb` | Proof Scope 検証器の残り 3 件（定数判定・コメント内指令・マクロ置換リスト）と、フィクスチャ 3 件 |
| `74e9aab0` | AC1 ミラーの述語整合（`retireBacklog` 項の削除、非 0 シナリオ追加、A2 コメント更新） |

### 4.3 main の取り込み

`git merge --no-ff main`（`743c1972`）。`src/core/EpochDomain.h` は自動マージされ、
PR 側の CAS 置換と main 側の `RetireHeadSnapshot` が共存する。競合・手動解決は無し。

---

## 5. 検証（`main` 取り込み後の各 head に対する実測）

| # | 検証 | 結果 |
| --- | --- | --- |
| 1 | `check-audioengine-lint.ps1` | PASS（LINT-AE-001〜014・違反 0） |
| 2 | `check-src-atomic-dotcall.ps1` | PASS（違反 0） |
| 3 | `isr-verify-proof-scope.ps1` | PASS |
| 4 | `test-isr-verify-proof-scope.ps1`（Proof Scope self-test） | **46/46 PASS** |
| 5 | `test-isr-verify-proof-scope-lexer.ps1`（Lexer boundary） | **全チェック PASS**（変異対照 3 件を含む） |
| 6 | Release ビルド | アプリ 96 ターゲット・全 223 ターゲット・エラー 0（修正コミット後は差分 3 ターゲットのみ再ビルド） |
| 7 | Release ctest | **57/57 PASS**（147.26 s・最終コミット後） |
| 8 | `isr-run-tiered-verification.ps1 -Tier standard` | **PASS**（§6.3） |

### 5.1 Release ctest

`ctest -C Release --output-on-failure`（統合後ツリーの build ツリー）。

| 項目 | 結果 |
| --- | --- |
| 結果 | **57/57 PASS**・失敗 0・147.26 s（`74e9aab0` の再ビルド後） |
| 主要テスト | `AudioEngineHarness` 140.44 s・`DomainTagTests`・`D8_2_B_2_Tests`・`TailLayerIntegrityTests` いずれも PASS |
| 登録数 | 57（統合後ツリーに未コミットの `CMakeLists.txt` が無いため、作業ツリーの 61 ではない） |

最終 head `25d68e9c` は、この ctest を実行した `74e9aab0` から **`.ps1` の 2 ファイルだけ**が
進んでいる（C++ のソースと `CMakeLists.txt` は同一）。したがって 57/57 は最終 head の C++ 内容に
対する結果である。CI の Windows MSVC / Release も最終 head で success している。

### 5.2 ISR ドメインタグ・RetireDomain の意味論

PR が意味論を変えていないことを、次の 3 点で確認した。

| 確認 | 結果 |
| --- | --- |
| `RetireDomain.h` / `DeferredDeletionQueue.h` の変更 | **無し**（PR の変更 12 ファイルに含まれない） |
| メモリオーダーの保存 | 変更 7 ファイルすべてで、削除行と追加行の `memory_order_*` の出現数が一致 |
| `EpochDomain.h` の変更 | CAS 1 行を `convo::compareExchangeWeakAtomic` へ置換（`relaxed` / `relaxed` のまま）。ドメイン刻印のコードは無変更 |

**唯一の意味論的変更**は `RetireLatencyRecorder::reset()` の 23 カウンタで、暗黙の `seq_cst` から
`release` へ変わった点である（`publishAtomic` の既定が `release`）。これは単なる機械的置換ではない。
呼び出し箇所を実測すると `AudioEngine` の ctor（`AudioEngine.CtorDtor.cpp:52`、`initialize()` 前・
コーディネータスレッド開始前）と、レコーダ自身の ctor（`ISRRetireLatency.h:1016`）だけで、
いずれも静止状態である。テストからの直接呼び出しは 0 件。したがって並行アクセスは成立せず、
順序を弱めても観測結果は変わらない。**設計判断として記録し、機械的置換として扱わない。**

### 5.3 既知の Debug 失敗と TSan の制約（既存問題との関係）

| 項目 | 確認結果 |
| --- | --- |
| Debug `D8_2_B_2_Tests` のスタック超過 | PR は当該テストとスタック設定を変更していない（変更 12 ファイルに含まれない）。既存問題のまま |
| Debug `TailLayerIntegrityTests` の RES-F 予算超過 | 同上。PR は `TailLayerIntegrityTests.cpp` を変更していない |
| TSan の `ScopedEpochReaderTests` 停止 | PR が原因行としている `ISRRetireLatency.h:1135` / `EpochDomain.h:767` は、PR の差分に**含まれない**（ベースに同一行が存在）。既存の Clang 非互換であることを確認 |
| TSan の `ISRSemanticValidationTests` | Linux で MKL を要する構成上の制約。CI の TSan ジョブは best-effort で success |

---

## 6. 追加で発見した不具合（Issue #208 / #209）

`main` を PR ブランチへ取り込んだ `743c1972` の CI で `ISR Verification` が失敗した。
原因を分類したところ、いずれも **`main` 側の最近の変更にゲートが追随していない**もので、
PR #207 の内容（lint / atomic / proof-scope）とは独立だった。指示（バグは Issues へ起票し、
修正して PR に載せ、マージ後に解決済みにする）に従い、2 件を起票して同じブランチで修正した。

| Issue | 事象 | 原因 | 修正 |
| --- | --- | --- | --- |
| #208 | `isr-verify-runtime-coordinator-state-machine.ps1` が 3 件の violation（`retireBacklogCount_` / `setRetireBacklogCount` 不在） | 承認済み改名（`862565a3`）にゲート 3 本が追随していない | ゲートを新識別子へ更新（`821506ca`） |
| #209 | `isr-verify-v73-shutdown-reclaim.ps1` が CI-SHUTDOWN-006（`ReleaseResources.cpp:707` の `isFullyDrained` callsite 未 allowlist） | `6bb57003` の DRAIN_DIAG が単一 drain authority の外から `isFullyDrained()` を呼んでいた | 取得済み `collectDrainBreakdown()` から 13 条件を計算（`b8a8607b`） |

### 6.1 #208 ゲートの改名追随

| スクリプト | 変更 |
| --- | --- |
| `isr-verify-runtime-coordinator-state-machine.ps1` | `retireBacklogCount_` → `retireAcceptedCount_`、`setRetireBacklogCount` → `setRetireAcceptedCount`（正規表現・メッセージ・契約コメント） |
| `isr-verify-v73-retire-pressure-contract.ps1` | 負の検査に `setRetireAcceptedCount\(` を追加（改名後の setter も production へ再登場させない） |
| `isr-verify-gate-wiring.ps1` | v7.3 スクリプトに `setRetireAcceptedCount` を要求 |

意味論は変えていない。`onRetireAccepted()` の `fetchAddAtomic`、`onRetireConsumed()` の
`old > 0` ガード付き `fetchSubAtomic`、`setRetireAcceptedCount` の `TEST-ONLY` 注記は
いずれも従来どおりで、ゲートが検査する対象の名前だけを現行コードに合わせた。

### 6.2 #209 単一 drain authority の維持

`[DRAIN_DIAG] timedOut` は `runtimePublicationBridge_.isFullyDrained()` を呼んでいた。
policy（`.github/isr-ai-governance-policy.json`）はこの呼び出しを
`AudioEngine.Threading.cpp` の単一 drain authority に限定しており、`ReleaseResources.cpp` からの
呼び出しは設計違反になる。allowlist を広げるのではなく、同じブロックが既に取得している
`collectDrainBreakdown()` から述語と同じ 13 条件を計算する形へ変えた。

- `retireBacklog` は C9-A（2026-11-13）で述語から外れているため**含めない**（ログ表示のみ）。
- 述語（`ShutdownScheduler::isFullyDrained()`）、drain authority、ログの項目は変更していない。

### 6.3 ローカル検証

| 検証 | 結果 |
| --- | --- |
| `isr-verify-runtime-coordinator-state-machine.ps1` | PASS |
| `isr-verify-v73-retire-pressure-contract.ps1` | PASS |
| `isr-verify-gate-wiring.ps1` | PASS |
| `isr-verify-v73-shutdown-reclaim.ps1` | PASS（violations 0） |
| `isr-run-tiered-verification.ps1 -Tier standard -DeclaredClass Class-A -SoakMinutes 0` | **PASS**（`tiered verification completed. tier=standard`） |

## 7. マージ

### 7.1 マージ条件の確認

| 条件 | 結果 |
| --- | --- |
| 必須 CI がすべて成功 | **満たす**（`25d68e9c`・success 14 / skipped 2 / failure 0） |
| レビュー上の未解決問題がない | **満たす**（12 件すべて修正。最終 push 後に新しい指摘なし） |
| 競合がない | **満たす**（`MERGEABLE` / `CLEAN`） |

### 7.2 実行結果

| 項目 | 値 |
| --- | --- |
| マージ方法 | `gh pr merge 207 --merge`（マージコミット） |
| マージ日時 | 2026-11-14T04:26:17Z |
| マージコミット | `6e099932bcb4f122a72526f1c9786eecd7fc08b6` |
| 親 | `862565a3`（main）＋ `25d68e9c`（PR head） |
| `main` の新 HEAD | `6e099932bcb4f122a72526f1c9786eecd7fc08b6` |
| PR の状態 | MERGED |
| Issue #208 / #209 | **CLOSED**（マージで自動解決） |

### 7.3 PR head `25d68e9c` の CI（マージ判定に使った実行）

| ワークフロー / ジョブ | 結果 |
| --- | --- |
| Windows MSVC / Release | success |
| Sanitizer CI（Debug TSan / RelWithDebInfo ASan） | success / success |
| ISR Verification（`isr-verify`） | success |
| audioengine-lint（`lint-audioengine`） | success |
| list-compliance（`compliance-list-md`） | success |
| ISR Authority Compliance v6.4 | success |
| authority-static-checks / authority-runtime-tests | success / success |
| audit-report-verification | success |
| CodeQL（actions / c-cpp / csharp / javascript-typescript） | success |
| Windows MSVC / probe | skipped（dispatch 専用） |
| CodeRabbit | skipped（差分が小さいため） |

### 7.4 マージ後の main（`6e099932`）

| ワークフロー | 結果 |
| --- | --- |
| audioengine-lint | **success**（`862565a3` では failure。本 PR で解消） |
| list-compliance | **success**（`862565a3` では failure。本 PR で解消） |
| ISR Authority Compliance v6.4 | success |
| Windows MSVC Build | success |
| Sanitizer CI | **success**（RelWithDebInfo ASan success・Debug TSan success） |
| CodeQL | **success**（actions / c-cpp / csharp / javascript-typescript すべて success） |

マージ後の `main`（`6e099932`）で 6 ワークフローすべてが success で確定した。
なお `6e099932` のツリーは PR head `25d68e9c` のツリーと同一（`git rev-parse <commit>^{tree}` が
どちらも `d74627d5b85af41120c2a84d63cb4ba08730621b`）であり、PR head の結果と矛盾しない。

### 7.5 ローカル作業ツリー

| 項目 | 結果 |
| --- | --- |
| 本体 worktree の HEAD | `862565a3` → `6e099932`（`git merge --ff-only`） |
| 未コミット変更 | 171 件のまま（他タスク分）。`ReleaseResources.cpp` の `probeStopLearning` 3 行は、退避 → マージ → 再適用で**内容が完全一致**することを確認（行番号のみ +18 シフト） |
| 破棄・上書き | なし |


---

## 8. 未解決事項・残存リスク

| # | 内容 | 状態 |
| --- | --- | --- |
| 1 | Debug の `D8_2_B_2_Tests`（スタック超過）と `TailLayerIntegrityTests`（RES-F 予算） | PR の対象外・既存問題。Release は PASS |
| 2 | TSan の部分カバレッジ（`ScopedEpochReaderTests` は Clang 非互換、`ISRSemanticValidationTests` は Linux で MKL 必須） | 既存制約。CI の TSan ジョブは best-effort で success |
| 3 | `--verify-runtime-evidence-live` の部分一致（B-11 の未コミット作業） | PR の対象外。main では live 証跡モードを使えない |
| 4 | `noteRetireBacklogChanged` / `previousRetireBacklogCount_` の旧名 | 改名課題の範囲外として据え置き |
| 5 | AC1 ミラー乖離 | **本 PR で解消**（レビュー指摘 #11） |

### 残存リスク

- Proof Scope 検証器は静的な字句解析に基づく。C++ の全構文を解釈するわけではないため、
  将来 `#if` の評価式が複雑化した場合は `unknown`（＝拒否）側に倒れる。安全側ではあるが、
  正当なコードを拒否する可能性は残る。
- 検証は Windows の Release 構成が中心で、Debug と Linux の TSan は制約付きの確認にとどまる。
- `isr-verify` の標準 tier はローカルでも通過したが、CI の完全な再現（authority inventory の
  生成や evidence の seed を含む）とは同一ではない。最終判定は CI の結果による。
