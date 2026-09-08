# D135-8/9 Gate G-4.3-T — Dedicated Regression Tests（Work Report）

**Status: PASS**（Debug + Release 実行合格）
**Type:** test-only implementation + execution.
**Tests added:** `src/tests/ISRSemanticValidationTests.cpp`（T1/T2/T7/T8/T10 + main 登録）。
**Production source:** 当初 0 変更の予定 → **ビルドで G-4.2 helper のコンパイルエラーが発覚し、compile-only 修正を1箇所適用**（下記「★ 発覚した production 修正」参照）。

## 実行結果
```
=== G-4.3-T: Build ISRSemanticValidationTests (Debug) ===   → OK (ninja 98/98, 0 error)
=== G-4.3-T: Run ISRSemanticValidationTests (Debug) ===     → DBG_EXIT=0
=== G-4.3-T: Build ISRSemanticValidationTests (Release) === → OK
=== G-4.3-T: Run ISRSemanticValidationTests (Release) ===   → REL_EXIT=0
```
テストは失敗時 `throw std::runtime_error` する構造 → exit 0 = 全テスト（新規 T1/T2/T7/T8/T10 + 既存 C1-C16 / R13 / R18 系）合格。exe 更新時刻 Debug 23:08 / Release 23:10（新規ビルド確認）。

## 追加テスト（G-4.3 固有の identity/capacity/terminal contract）
| T | 内容 | 結果 |
|---|---|---|
| **T1** `testG43_T1_sameIdentityCoalesces` | same handle+same target → COALESCE、liveCount==1、coalescedCount+1、**既存 oblId 再利用（T6）** | ✅ PASS |
| **T2** `testG43_T2_sameHandleDifferentTarget_New` | same handle + 異なる irIdentityHash（601 vs 602）→ 別 obligation、liveCount==2 | ✅ PASS |
| **T7** `testG43_T7_terminalThenSameIdentity_NewAdmission` | submit→resolve(Published)→liveCount 0→同一 identity 再 submit→**NEW（liveCount 1）**（terminal は Live coalesce 候補にならない） | ✅ PASS |
| **T8** `testG43_T8_coalesceKeepsRecoveryGeneration` | coalesce 前後で `popRecoveryRequest()->recoveryGeneration` 同一（再生成なし） | ✅ PASS |
| **T10** `testG43_T10_buildSourceMetadataDriftCoalesces` | 同一 rebuildFingerprint（同一 target）で snapshot-level metadata（sampleRate）だけ変えて再 submit → COALESCE、liveCount==1（identity 不変、D19.4） | ✅ PASS |

既存テストが既にカバー（重複追加せず）：**T3**=C4（distinct handles same target）、**T4**=C3（capacity full + matching → coalesce）、**T5**=C2（capacity full + non-matching → reject）。**T9**（intentId≠RecoveryGeneration）は静的検証済み（`recoveryGeneration = intent.intentId` = 0 件）。**T11**（deterministic race）は production へ synchronization hook を入れる制約により、既存 API で決定論的に作れないため**未実装**（T1-T10 + source-level invariant で終了）。

## ★ 発覚した production 修正（compile-only、意味論不変）
ビルドで `ISRRuntimePublicationCoordinator.cpp` の **G-4.2 helper がコンパイル不能**だった：
```
error C4430/C2146/C2143/C2447 @ cpp:849-850  computeDomainCoverage の戻り値型 ObligationDomains が未解決
error C3861 @ cpp:888  computeDomainCoverage 識別子が見つからない
```
原因：`computeDomainCoverage` は `convo::isr` の anonymous namespace の free function だが、戻り値型 `ObligationDomains` は `RuntimeIntentCoordinator` の**ネストされた member 型**で、無修飾ではそのスコープから参照不能。
修正（1 行、compile-only・非意味論）：anonymous namespace 冒頭に
```cpp
using convo::isr::RuntimeIntentCoordinator;
using ObligationDomains = RuntimeIntentCoordinator::ObligationDomains;
```
を追加。**coalesce / capacity / terminal / delivery / generation の意味論は一切変更なし**（型名の可視化のみ）。

> これは G-4.2 の helper 導入時の未検証コンパイル欠陥であり、G-4.3-T のビルドで初めて露見した。G-4.2/G-4.3 の read-only 監査（A1-A12/R1-R12）は意味論を検証したがビルド検証を含んでいなかった。以後の Gate ではビルド検証を早期に組み込むべき。

## 制約遵守
- 変更ファイル：`src/tests/ISRSemanticValidationTests.cpp`（テスト追加）+ `src/audioengine/ISRRuntimePublicationCoordinator.cpp`（**compile-only 修正 1 箇所**）。
- 未変更：`ISRRuntimePublicationCoordinator.h`、AudioEngine、reclaim、shutdown、publish、durable fallback、CMake、reservation、canSupersede、supersession、stalled/retry。
- production API への test-only accessor 追加なし（T6/T8/T10 は既存 public API `submitRecoveryRequest`/`popRecoveryRequest`/`liveLogicalRecoveryObligationCount`/`recoveryCoalescedCount` のみ使用）。

## STOP
G-4.3-T = PASS（T1/T2/T7/T8/T10 新規 + T3/T4/T5 既存カバー + T9 静的 + T11 対象外）。**G-4.4 / durable overwrite 修正 / 追加テスト拡張には進まない。** compile-only 修正の妥当性について再監査（read-only）を推奨。
