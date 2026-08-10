# work88 P2 実装チェックリスト（REPAIR_PLAN2-dash.md §1.1〜§1.4）

**更新日**: 2026-08-10
**対象**: REPAIR_PLAN2-dash.md の P2-1（§1.1）・P2-4（§1.2）・P2-3（§1.3）・P2-2（§1.4）
**状態**: ✅ 全項目 完了（ビルド + ctest 28/28 合格で検証済み）
**監査補正（2026-08-10・第三者別視点監査反映）**: P2-4 が露出した shutdown admission closure の穴を修正（Recovery shutdown discard — Step A/B/C 実装済み・ctest 28/28 合格）
**対象外**: X1〜X6（P2 後の別タスク — §5 参照）

---

## P2-1 🔴 pendingIntentCount_ の residency accounting 再設計（§1.1）— ✅ 完了

| # | 項目 | 実装箇所 | 状態 |
| --- | ------ | --------- | ------ |
| 1 | 宣言コメントで semantic を固定（Observe/Quarantine/Recovery residency + reservation。Publish/RetireIntent 除外） | `ISRRuntimePublicationCoordinator.h` :384 付近 | ✅ |
| 2 | `submitObserve`: reservation-before-push（fetchAdd → push → 全段失敗 rollback + drop カウンタ） | `ISRRuntimePublicationCoordinator.cpp` `submitObserve` | ✅ |
| 3 | `observeOverflowCounter_` は既存位置（primary push 失敗直後）に維持 | 同上 | ✅ |
| 4 | `submitRecoveryRequest`: reservation-before-push + 失敗時 rollback + `recoveryIntentDropCount_++`（INV-5 silent loss 禁止） | 同上 `submitRecoveryRequest` | ✅ |
| 5 | `popRecoveryRequest`: cur>0 ガード削除 + pop 成功時 fetchSub | 同上 `popRecoveryRequest` | ✅ |
| 6 | `submitQuarantine`: `pendingIntentCount_` のみ reservation。`quarantineResidentCount_` は現状維持（P3 対象） | 同上 `submitQuarantine` | ✅ |
| 7 | `processIntent`: `quarantineFallbackQueue_` pop 無条件 fetchSub / `intentQueue_` pop は Publish 以外 fetchSub（七次 W2） | `ISRRuntimePublicationCoordinator_ProcessIntent.cpp` `processIntent` | ✅ |
| 8 | `drainObserveDeferred`: pop 成功直後・skip 判定前に fetchSub（十次） | 同上 `drainObserveDeferred` | ✅ |
| 9 | `setPendingIntentCount(0)` hard reset 廃止 | `ProcessIntent.cpp` | ✅ |
| 10 | `AudioEngine::isFullyDrained()` の `setPendingIntentCount`/`setPublicationBacklogCount` 上書き廃止。返り値 `!hasDeferredCommit` は維持 | `AudioEngine.Threading.cpp` :117-139 | ✅ |
| 11 | `AudioEngine.Commit.cpp` の `setPendingIntentCount`（RetireIntent 混入）廃止 ×2 箇所。`setRetireBacklogCount` は維持 | `AudioEngine.Commit.cpp` :462, 607 | ✅ |

## P2-4 🟠 isFullyDrained の queue emptiness 検証（§1.2）— ✅ 完了

| # | 項目 | 実装箇所 | 状態 |
| --- | ------ | --------- | ------ |
| 12 | `ShutdownScheduler::isFullyDrained()` に 4 キュー空判定（消費なし）を追加 | `ISRRuntimePublicationCoordinator.cpp` `ShutdownScheduler::isFullyDrained` | ✅ |
| 13 | 判定キュー: `intentQueue_`(sizeApprox) / `observeDeferredRing_`(size) / `quarantineFallbackQueue_`(sizeApprox) / `recoveryIntentQueue_`(size) | 同上 | ✅ |
| 14 | 既存 7 カウンタ == 0 を維持（retireBacklog / publicationBacklog / pendingIntent / fallbackBacklog / reclaimInFlight / deferredRetire / quarantineResident） | 同上 | ✅ |
| 15 | phase-gated コメント（admission closed + producer join 後にのみ authoritative）をコードで固定 | 同上 | ✅ |

## P2-3 🟡 MpscBoundedRing の producer hole テスト固定（§1.3）— ✅ 完了

| # | 項目 | 実装箇所 | 状態 |
| --- | ------ | --------- | ------ |
| 16 | `MpscBoundedRing.h` に `#ifdef CONVO_TESTING` フック（push の CAS 成功直後・payload 書込み前） | `src/MpscBoundedRing.h` `push()` | ✅ |
| 17 | 位置ベース設計（`testHoleBlockPos_`）で既存テストのハング回避 + FIFO テスト両立 | 同上（private メンバ + public test API） | ✅ |
| 18 | public テスト API（`testSetHoleBlock` / `testReleaseHole` / `testResetHole` 等）を `CONVO_MPSC_TEST_HOOKS` 内に定義 | 同上 | ✅ |
| 19 | テスト 4 本（2 スレッド deterministic）: delayed publication / FIFO order / empty pop false / payload publication ordering | `src/tests/MpscBoundedRingTests.cpp` | ✅ |
| 20 | `CMakeLists.txt` で `MpscBoundedRingTests` のみ `CONVO_TESTING=1`（本番ターゲットには定義しない） | `CMakeLists.txt` | ✅ |

## P2-2 🟡 invariant_INV3_INV5 テスト追加（§1.4）— ✅ 完了

| # | 項目 | 実装箇所 | 状態 |
| --- | ------ | --------- | ------ |
| 21 | `src/tests/invariant_INV3_INV5.cpp` 新規作成（INV-3-1 / INV-3-2 / INV-5-1 / INV-5-2 の 4 本） | `src/tests/invariant_INV3_INV5.cpp` | ✅ |
| 22 | `TestEpochProvider`（IEpochProvider スタブ）で router の currentEpoch / minReaderEpoch を制御（INV-3 の安全/非安全を deterministic 化） | 同上 | ✅ |
| 23 | INV-3-1: retire → epoch 安全 → reclaim 完了（isRetired false / reclaimInFlight 0） | 同上 | ✅ |
| 24 | INV-3-2: epoch 非安全 → false（遅延通知）→ retire 実行済み / reclaim 未実行 → epoch 安全化後再試行成功（TOCTOU） | 同上 | ✅ |
| 25 | INV-5-1: 256 回 submit で full → 257 回目 drop（recoveryIntentDropCount=1）+ pendingIntentCount 不変 → pop で消費 | 同上 | ✅ |
| 26 | INV-5-2: `QuarantineService::executeQuarantine` の stateChanged ゲート（null → false / 有効 → true） | 同上 | ✅ |
| 27 | CMake: `add_executable(invariant_INV3_INV5Tests)` + 8 ISR ソース + include/link | `CMakeLists.txt` | ✅ |
| 28 | CMake: ISRSemanticValidationTests と同一の設定一式（MKL link/define/include、cxx_std_20、/utf-8、/EHsc、WIN32 定義、icx LTCG OFF） | `CMakeLists.txt` | ✅ |
| 29 | CMake: `add_test(NAME InvariantINV3INV5 COMMAND invariant_INV3_INV5Tests)` | `CMakeLists.txt` | ✅ |

---

## 検証結果

### ビルド

- `cmake --build build --config Debug --target invariant_INV3_INV5Tests MpscBoundedRingTests`（CMakeLists.txt 変更で自動再構成）: ✅ 成功（83 ステップ）
- warning は既存の無関係なもののみ（C4458 in AudioEngine.Retire.cpp / C4996 in AudioEngine.Processing.Latency.cpp）

### ctest（28/28 合格）

```text
 5/28  MpscBoundedRingTests ................... Passed   0.03 sec   ← P2-3（新テスト4本含む・計10本）
14/28  ISRSemanticValidationRejects ............ Passed   0.41 sec   ← P2-1 回帰（testObserveOverflowEnqueuePath 含む）
15/28  InvariantINV3INV5 ....................... Passed   0.04 sec   ← P2-2（新規）
13/28  RuntimePublicationCoordinatorRejects ..... Passed   0.04 sec   ← P2-1/P2-4 回帰
 2/28  ISRSoakTests ............................ Passed   0.78 sec   ← 回帰
全 28 テスト合格（100%）
```

### 新テスト単体出力

- `invariant_INV3_INV5Tests.exe`: `ALL TESTS PASSED`（exit 0）
- `MpscBoundedRingTests.exe`: `Tests: 10, Failures: 0`（既存 6 + P2-3 新規 4）

---

## P2-4 監査補正（2026-08-10・第三者別視点監査反映）— Recovery shutdown admission closure ✅ 完了

> **背景（監査補正）**: P2-4 の queue emptiness は「検出」としては正しい（queue が実際に non-empty なら
> drain timeout は false-positive ではなく true-positive）。問題は「なぜ shutdown 完了時点で queue が
> non-empty か」= **shutdown admission closure の穴**（StopAcceptingWork が logical flag だけで、
> Recovery admission の linearization point として完全に閉じていない）。修正対象は P2-4 ではなく
> **前段の shutdown admission / Recovery lifecycle**。P2-4 の queue observation は維持する。

| # | 項目 | 実装箇所 | 状態 |
| --- | ------ | --------- | ------ |
| 30 | Step B: `submitRecoveryRequest()` に shutdown gate（`state == ShuttingDown` なら enqueue しない）。reservation 前で評価するため counter 非接触。閉鎖後の submit は `recoveryShutdownDiscardCount_++`（ShutdownDiscard — silent loss 禁止 INV-5） | `ISRRuntimePublicationCoordinator.cpp` `submitRecoveryRequest` | ✅ |
| 31 | Step C: `discardRecoveryRequestsOnShutdown()` 新設 — 残留 Recovery を ShutdownDiscard として明示破棄（popRecoveryRequest が fetchSub するため counter 整合） | `ISRRuntimePublicationCoordinator.cpp` `discardRecoveryRequestsOnShutdown` | ✅ |
| 32 | `stopRebuildThread()` の Builder join 後に `discardRecoveryRequestsOnShutdown()` を呼ぶ（Producer は shutdownCoordinatorLoop で join 済み → 決定的） | `AudioEngine.RebuildDispatch.cpp` `stopRebuildThread` | ✅ |
| 33 | telemetry: `recoveryShutdownDiscardCount_` を新設し drop（queue full）と区別（§8.1 ShutdownDiscard 分離方針）。getter 公開・Critical 昇格対象外 | `ISRRuntimePublicationCoordinator.h` getter + メンバ | ✅ |
| 34 | INV-5-3: requestShutdown() 後の submit → ShutdownDiscard（enqueue なし・pending 不変・drop と区別） | `src/tests/invariant_INV3_INV5.cpp` | ✅ |
| 35 | INV-5-4: discardRecoveryRequestsOnShutdown() が残留を明示 discard（queue empty + pending 0 + discard カウント） | 同上 | ✅ |

> **設計メモ**: Step A（Recovery admission を shutdown で閉じる）は既存 `requestShutdown()`
> （ReleaseResources.cpp:75 / CtorDtor.cpp:102、state=ShuttingDown 確定）が担当。Step B の gate が
> CoordinatorLoop の in-flight phase 中の submit を拒否し（submit 側 gate = admission の最終
> linearization point — Authority Singularization）、Step C の discard が Builder 停止後の残留を
> 決定的に回収する。残余: admission check → reservation の atomicity は X1（RecoveryDurableAdmission +
> lease 方式）で完全化。

---

## 残課題（対象外・§5 参照）

- X1: Recovery full-drop そのもの（最優先）
- X2: Publish completion monotonicity
- X3: shutdownReclaim 二系統
- X4: authority 二重化
- X5: Publish intent residency 専用 counter 未導入
- X6: quarantine intent/resident semantic 分離
