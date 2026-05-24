#include <JuceHeader.h>
#include "AudioEngine.h"
#include "NoiseShaperLearner.h"

static void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}

void AudioEngine::releaseResources()
{
    ASSERT_NON_RT_THREAD();
    diagLog("[DIAG] releaseResources: enter");

    auto previousState = convo::consumeAtomic(lifecycleState, std::memory_order_acquire);
    for (;;)
    {
        if (previousState == EngineLifecycleState::Destroyed)
        {
            diagLog("[DIAG] releaseResources: ignored in Destroyed state");
            return;
        }

        if (previousState == EngineLifecycleState::Unprepared)
        {
            diagLog("[DIAG] releaseResources: duplicate release ignored (already Unprepared)");
            return;
        }

        if (previousState == EngineLifecycleState::Releasing)
        {
            diagLog("[DIAG] releaseResources: already Releasing");
            return;
        }

        if (convo::compareExchangeAtomic(lifecycleState,
                         previousState,
                         EngineLifecycleState::Releasing,
                         std::memory_order_acq_rel,
                         std::memory_order_acquire))
            break;
    }

    // P0-A0: LifecycleIsolationRuntime integration - enter release phase
    auto lifecycleToken = lifecycleRuntime_.enterRelease();

    setShutdownPhase(ShutdownPhase::StopAcceptingWork, "releaseResources");
    shutdownRuntime_.transitionTo(convo::isr::ShutdownPhase::AudioStopped);

    // 非MT起点の pending rebuild 要求と AsyncUpdater キューをシャットダウン直後に廃棄する。
    // stopRebuildThread より先に実行して handleAsyncUpdate が後から rebuild を発火しないようにする。
    clearRebuildReason(RebuildReason::StructuralFromNonMT);
    clearRebuildReason(RebuildReason::DeferredStructural);
    clearRebuildReason(RebuildReason::DeferredFinalizeAware);
    convo::publishAtomic(deferredFinalizeFirstSeenTicks_, 0, std::memory_order_release);
    cancelPendingUpdate();
    convo::publishAtomic(firstIrDryCrossfadePending, false, std::memory_order_release);
    convo::publishAtomic(dspCrossfadeUseDryAsOld, false, std::memory_order_release);
    convo::publishAtomic(dspCrossfadeStartDelayBlocks, 0, std::memory_order_release);
    convo::publishAtomic(dspCrossfadeDryHoldSamples, 0, std::memory_order_release);
    convo::publishAtomic(dspCrossfadePending, false, std::memory_order_release);
    convo::publishAtomic(latencyResetPending, false, std::memory_order_release);
    convo::publishAtomic(lastIssuedConvolverStructuralHash_, 0, std::memory_order_release);
    convo::publishAtomic(lastCommittedConvolverStructuralHash_, 0, std::memory_order_release);
    convo::publishAtomic(lastCommittedConvolverHasIr_, false, std::memory_order_release);
    convo::publishAtomic(currentSampleRate, 0.0, std::memory_order_release);

    convo::publishAtomic(inputLevelLinear, 0.0f, std::memory_order_release);
    convo::publishAtomic(outputLevelLinear, 0.0f, std::memory_order_release);

    if (noiseShaperLearner)
        noiseShaperLearner->stopLearning();

    resetLearningControlState();
    setShutdownPhase(ShutdownPhase::StopAudio, "releaseResources");

    DSPCore* activeToRelease = nullptr;
    DSPCore* fadingToRelease = nullptr;
    DSPCore* pendingNewToRelease = nullptr;
    DSPCore* pendingCurrentToRelease = nullptr;

    {
        std::lock_guard<std::mutex> lk(rebuildMutex);
        validateDistinctRuntimeSlots("releaseResources.beforeClear",
                         activeDSP,
                         resolveFadingDSPFromRuntimeWorldOnly(getRuntimePublishView().graph),
                         nullptr);

        convo::fetchAddAtomic(rebuildGeneration, 1, std::memory_order_acq_rel);
        {
            auto* const activeRaw = activeDSP.get();
            activeToRelease = (reinterpret_cast<uintptr_t>(activeRaw) == (~static_cast<uintptr_t>(0))) ? nullptr : activeRaw;
        }
        activeDSP = nullptr;

        {
            auto* const fadingRaw = exchangeFadingOutDSP(nullptr);
            fadingToRelease = (reinterpret_cast<uintptr_t>(fadingRaw) == (~static_cast<uintptr_t>(0))) ? nullptr : fadingRaw;
        }
        convo::publishAtomic(dspCrossfadeUseDryAsOld, false, std::memory_order_release);
        dspCrossfadeDryScaleGain.setCurrentAndTargetValue(1.0);
        convo::publishAtomic(queuedFadeTimeSec, 0.03, std::memory_order_release);

        if (hasPendingTask)
        {
            auto* const pendingRaw = pendingTask.currentDSP;
            pendingCurrentToRelease = (reinterpret_cast<uintptr_t>(pendingRaw) == (~static_cast<uintptr_t>(0))) ? nullptr : pendingRaw;
            pendingTask.currentDSP = nullptr;
            hasPendingTask = false;
        }

        convo::publishAtomic(dspCrossfadePending, false, std::memory_order_release);
        dspCrossfadeGain.setCurrentAndTargetValue(1.0);
        refreshCrossfadePreparedSnapshotFromAtomics();
        RuntimePublicationCoordinator::create(RuntimePublicationBridge { *this }, runtimeStore)
            .publishState(nullptr,
                          nullptr,
                          convo::TransitionPolicy::HardReset,
                          0.0,
                          false);

        validateDistinctRuntimeSlots("releaseResources.afterClear",
                         activeDSP,
                         resolveFadingDSPFromRuntimeWorldOnly(getRuntimePublishView().graph),
                         nullptr);
    }

    diagLog("[DIAG] releaseResources: before stopRebuildThread");
    setShutdownPhase(ShutdownPhase::StopWorkers, "releaseResources");
    stopRebuildThread();
    shutdownRuntime_.transitionTo(convo::isr::ShutdownPhase::ObserverDrained);
    diagLog("[DIAG] releaseResources: after stopRebuildThread");

    setShutdownPhase(ShutdownPhase::ForceEpochAdvance, "releaseResources");
    advanceRetireEpoch();
    shutdownRuntime_.transitionTo(convo::isr::ShutdownPhase::RetireClosed);
    shutdownRuntime_.transitionTo(convo::isr::ShutdownPhase::EpochSettled);

    setShutdownPhase(ShutdownPhase::DrainRetire, "releaseResources");

    drainPublicationLogForShutdown();

    if (activeToRelease)
        retireDSP(activeToRelease);
    if (fadingToRelease)
        retireDSP(fadingToRelease);
    if (pendingNewToRelease)
        retireDSP(pendingNewToRelease);
    if (pendingCurrentToRelease)
        retireDSP(pendingCurrentToRelease);

    // shutdown/release シーケンスでは明示的に deferred retire queue をドレインする。
    // 通常タイマー経路は Releasing 中に early-return するため、ここで最終回収を保証する。
    drainDeferredRetireQueues(true);
    shutdownRuntime_.transitionTo(convo::isr::ShutdownPhase::ReclaimComplete);

    const auto activeHandle = dspHandleRuntime_.getActiveDSP();
    const auto fadingHandle = dspHandleRuntime_.getFadingDSP();
    if (!activeHandle.isNull())
    {
        dspHandleRuntime_.retire(activeHandle);
        dspHandleRuntime_.reclaim(activeHandle);
    }
    if (!fadingHandle.isNull() && fadingHandle != activeHandle)
    {
        dspHandleRuntime_.retire(fadingHandle);
        dspHandleRuntime_.reclaim(fadingHandle);
    }
    convo::publishAtomic(activeCrossfadeId_, static_cast<convo::isr::CrossfadeId>(0u), std::memory_order_release);

    diagLog("[DIAG] releaseResources: before ui processor release");
    diagLog("[DIAG] releaseResources: before uiConvolverProcessor.releaseResources");
    uiConvolverProcessor.releaseResources();
    diagLog("[DIAG] releaseResources: after uiConvolverProcessor.releaseResources");

    diagLog("[DIAG] releaseResources: before uiEqEditor.releaseResources");
    uiEqEditor.releaseResources();
    diagLog("[DIAG] releaseResources: after uiEqEditor.releaseResources");

    diagLog("[DIAG] releaseResources: after ui processor release");

    diagLog("[DIAG] releaseResources: skip deferred reclaim (reconfigure phase)");

    RuntimePublicationCoordinator::create(RuntimePublicationBridge { *this }, runtimeStore)
        .clearPublishedRuntimeSnapshotsNonRt();

    const auto pendingRetireCount = [&]() noexcept -> uint32_t
    {
        uint32_t count = 0;
        if (convo::consumeAtomic(audioThreadRetireOverflowPtr, std::memory_order_acquire) != nullptr)
            ++count;

        std::lock_guard<std::mutex> lock(deferredDeleteFallbackMutex);
        count += static_cast<uint32_t>(deferredDeleteFallbackQueue.size());
        return count;
    }();

    const auto activeCrossfadeCount = consumeAtomic(activeCrossfadeId_, std::memory_order_acquire) != static_cast<convo::isr::CrossfadeId>(0u) ? 1u : 0u;
    shutdownRuntime_.setBoundedTeardownCounters(
        convo::consumeAtomic(audioCallbackActiveCount_, std::memory_order_acquire),
        activeCrossfadeCount,
        pendingRetireCount,
        activeEpochObserverCount());

    debugRuntime_.recordHBEdge(300u,
                               400u,
                               static_cast<std::uint64_t>(pendingRetireCount),
                               static_cast<std::uint64_t>(activeCrossfadeCount),
                               static_cast<int>(std::memory_order_acq_rel));

    shutdownRuntime_.transitionTo(convo::isr::ShutdownPhase::ShutdownComplete);
    shutdownRuntime_.emitShutdownTrace();

    emitEvidenceTickNonRt(true);

    convo::publishAtomic(lifecycleState, EngineLifecycleState::Unprepared, std::memory_order_release);
    diagLog("[DIAG] releaseResources: ABOUT_TO_EXIT_SCOPE");

    // P0-A0: LifecycleIsolationRuntime integration - leave release phase
    lifecycleRuntime_.leaveRelease(lifecycleToken);
}
