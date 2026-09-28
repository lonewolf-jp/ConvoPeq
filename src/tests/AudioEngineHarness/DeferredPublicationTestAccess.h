// DeferredPublicationTestAccess.h
// AudioEngineHarness 統合テスト専用の Friend Test Access。
//
// AudioEngine は private メンバ（runtimeOrchestrator_ / testFadingRuntimePresent_）を
// 持つため、テストから Orchestrator へ到達するには friend 宣言（AudioEngine.h:3515）
// が必要。本クラスは DeferredFlowIntegrationTests / DeferredPublishViewStateMachineTests
// の双方から共有する（グローバル名前空間・AudioEngine.h:122 の前方宣言に合わせる）。
//
// ★ Testing Principle（第13回レビュー反映・design-D4 修正）:
//   RebuildThread ownership（jassert ガード付き）を強制するオブジェクト
//   （DeferredPublishView / RuntimePublicationOrchestrator の deferred 系）は
//   AudioEngineHarness 上の Integration Test で検証する。
//   Standalone 単体テスト（main() + add_executable/add_test）は純粋 Policy
//   （PublicationAdmission::evaluateDeferred 等）にのみ予約する。

#pragma once

#include "audioengine/AudioEngine.h"
#include "audioengine/RuntimePublicationOrchestrator.h"
#include "audioengine/AtomicAccess.h"

class DeferredPublicationTestAccess final
{
public:
    static convo::isr::RuntimePublicationOrchestrator& orchestrator(AudioEngine& e) noexcept
    {
        return *e.runtimeOrchestrator_;
    }

    // Option-2 hook: DeferredFadingActive の「前提条件」（published world に
    // fading runtime が存在）を決定論的に作る。Production の Decision 判定
    // ロジックは一切変更しない（PublicationAdmission.cpp evaluate の
    // if (hasFading) → DeferredFadingActive 分岐はそのまま）。
    static void setFadingRuntimePresent(AudioEngine& e, bool on) noexcept
    {
        convo::publishAtomic(e.testFadingRuntimePresent_, on, std::memory_order_release);
    }

    // ★ STG-8: Coordinator 観測（liveLogicalRecoveryObligationCount /
    //   recoveryRetryRedriveCount）。AudioEngine の friend のため private 到達可能。
    static convo::isr::RuntimeIntentCoordinator& coordinator(AudioEngine& e) noexcept
    {
        return e.runtimePublicationBridge_;
    }

    // ★ STG-8-D3: retire-pressure throttle の test-only 設定。
    //   PublicationAdmission::evaluate の Pressure 分岐（:40-48）を決定論的に発火させる。
    static void setRetirePressureThrottle(AudioEngine& e, bool on) noexcept
    {
        convo::publishAtomic(e.retirePressurePublicationThrottleActive_, on, std::memory_order_release);
    }

    // ★ STG-8-D1: rebuild generation の test-only  bump。
    //   新規 build の submit を伴わないため、deferred slot の generation-stale Discard を
    //   決定論的に発火させる（evaluateDeferred :82-84）。
    static void bumpRebuildGeneration(AudioEngine& e) noexcept
    {
        e.rebuildRequestGeneration.fetch_add(1, std::memory_order_acq_rel);
    }
};
