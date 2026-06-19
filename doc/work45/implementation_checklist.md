# Work45 実装チェックリスト

**Plan:** refined_plan.md v1.8.0
**Status:** 全Phase実装完了
**Date:** 2026-06-18

---

## Phase-0a: 既存テストのコンパイルエラー修正 ✅

- [x] `src/tests/PublicationValidatorIsolationTests.cpp` — 既に修正済み（`numSources`, `numDestinations`, `memoryBudgetBytes` 参照なし）

## Phase-0b: useDryAsOld Dormant Bug 除去 ✅

- [x] `src/audioengine/RuntimeBuilder.cpp:288` — `active` → `(policy == convo::TransitionPolicy::DryAsOld)`

## Phase-0c: Dead Code 除去 ✅

- [x] `src/audioengine/CrossfadeRuntime.h` — `setUseDryAsOld()` / `setFirstIrDryPending()` 削除

## Phase-1: RuntimePublicationValidator 実体化 ✅

- [x] `RuntimePublicationValidator.h` — `ValidationFailureReason` enum 追加
- [x] `RuntimeValidationResult` — `failureReason` フィールド追加
- [x] `validateTopology()` — runtimeUuid/hasFadingRuntime/fadingRuntimeUuid 整合性チェック実装
- [x] `validateResources()` — oversampling/dither/noiseShaper バリデーション実装
- [x] `checkNoConflictingTransitions()` — TransitionPolicy 別チェック実装
- [x] `validatePublication()` — failureReason 設定追加

## Phase-1.5: Validator Telemetry ✅

- [x] `RuntimeHealthMonitor.h` — イベントコード 6000-6003 追加
- [x] `ValidationFailureReason` 前方宣言追加
- [x] `#include <array>` 追加
- [x] `emitValidationEvent()` public メソッド追加
- [x] `m_lastValidationEventUs_` + `kValidationEventMinIntervalUs` private メンバ追加
- [x] `RuntimeHealthMonitor.cpp` — `emitValidationEvent()` 実装
- [x] `AudioEngine.Commit.cpp` — validation failure 時に `m_healthMonitor.emitValidationEvent()` 呼び出し

## Phase-2: CrossfadePolicy 抽出 ✅

- [x] P2-1: `CrossfadePolicy` struct 追加（`CrossfadeAuthority.h`）
- [x] P2-2: `evaluate()` シグネチャ変更（engine引数削除 → policy引数追加）
- [x] P2-3: `evaluate()` 実装（engine直読 → policy参照）
- [x] P2-4: `AudioEngine::makeCrossfadePolicy()` factory method 追加
- [x] P2-5: Orchestrator 呼び出し側更新（policy生成 + HealthState Critical抑制）

## Phase-2.5: DSPTransition Emergency Override 公式化 ✅

- [x] P2.5-1: DSPTransition Emergency Override 改善（enqueueHealthEvent + abortCount）
- [x] P2.5-2: イベントコード `EVENT_CROSSFADE_ABORTED_EMERGENCY = 4003` 追加
- [x] `CrossfadeRuntime::emergencyAbortCount()` / `incrementEmergencyAbortCount()` 追加
- [x] `m_emergencyAbortCount_` atomic メンバ追加
- [x] `AudioEngine::enqueueHealthEvent()` 追加
- [x] P2.5-4: `armCrossfadeIfPending` — `crossfadeRuntime_.isPending()` AND条件追加（無限再Arm防止）

## Phase-3: テスト追加 ✅

- [x] P3-1: Validator Reject テスト（11ケース）
  - ValidateTopology_NoRuntimeUuid_Reject
  - ValidateTopology_HasFadingMismatch_Reject
  - ValidateTopology_FadingTransitionMismatch_Reject
  - ValidateResources_OversamplingNotPowerOfTwo_Reject
  - ValidateResources_OversamplingOutOfRange_Reject
  - ValidateResources_DitherInvalid_Reject
  - ValidateResources_NoiseShaperOutOfRange_Reject
  - CheckTransition_HardResetWithFade_Reject
  - CheckTransition_SmoothOnlyNegativeFade_Reject
  - CheckTransition_DryAsOldWithoutFlag_Reject
  - CheckTransition_InactiveWithUseDryAsOld_Reject
  - CheckTransition_UnknownPolicy_Reject
- [x] P3-2: Validator Accept テスト（5ケース）
  - ValidateTopology_Bootstrap_Accept
  - CheckTransition_HardResetNoFade_Accept
  - CheckTransition_DryAsOldValid_Accept
  - CheckTransition_IdleWithFadeRemnant_Accept
  - ValidateResources_ValidOversampling_Accept
- [x] ValidatePublication 統合テスト（3ケース: Topology/Resources/Transition reject）
- [x] P3-3: CrossfadeAuthority Regression テスト（4ケース）
  - DeterministicDecision
  - PolicyChangeChangesDecision
  - SameStructuralHashNoCrossfade
  - OversamplingChangeTriggersCrossfade

## Phase-4: Validator 網羅率拡充 ✅

- [x] P4-1: Validator ルール追加
  - `processingOrder` 0 or 1 チェック
  - `runtimeGeneration > 0 if generation > 0` チェック
  - `sequenceId != 0 if generation > 0` チェック
- [x] P4-2: Builder/Validator/Orchestrator 責務定義文書化

## Appendix B-2: スキーマバージョン更新 ✅

- [x] `ISRRuntimeSemanticSchema.h` — `kRuntimeSemanticSchemaVersion` 8u → 9u

---

## 実装サマリー

| Phase | ファイル | 変更内容 |
|-------|---------|---------|
| 0b | `RuntimeBuilder.cpp` | `useDryAsOld = active` → `(policy == DryAsOld)` |
| 0c | `CrossfadeRuntime.h` | 2つのデッドコードメソッド削除 |
| 1 | `RuntimePublicationValidator.h` | `ValidationFailureReason` enum + `failureReason` 追加 |
| 1 | `RuntimePublicationValidator.cpp` | validateTopology/Resources/checkNoConflictingTransitions 実体化 |
| 1.5 | `RuntimeHealthMonitor.h` | 前方宣言 + イベントコード + emitValidationEvent + array メンバ |
| 1.5 | `RuntimeHealthMonitor.cpp` | emitValidationEvent 実装 |
| 1.5 | `AudioEngine.Commit.cpp` | validation failure → emitValidationEvent 呼び出し |
| 2 | `CrossfadeAuthority.h` | CrossfadePolicy struct + evaluate シグネチャ変更 |
| 2 | `CrossfadeAuthority.cpp` | evaluate 実装（engine依存除去） |
| 2 | `AudioEngine.h` | makeCrossfadePolicy + enqueueHealthEvent + armCrossfadeIfPending修正 |
| 2 | `RuntimePublicationOrchestrator.cpp` | evaluate呼び出し更新 + Critical抑制 |
| 2.5 | `DSPTransition.h` | Emergency Override 公式化（abortCount + enqueueHealthEvent） |
| 2.5 | `CrossfadeRuntime.h` | emergencyAbortCount/incrementEmergencyAbortCount 追加 |
| 3 | `PublicationValidatorIsolationTests.cpp` | 20+テストケース追加 |
| 4 | `RuntimePublicationValidator.cpp` | processingOrder/runtimeGeneration/sequenceId チェック追加 |
| 4 | `RuntimePublicationValidator.h` | Builder/Validator 責務定義文書化 |
| B-2 | `ISRRuntimeSemanticSchema.h` | スキーマバージョン 8→9 |
