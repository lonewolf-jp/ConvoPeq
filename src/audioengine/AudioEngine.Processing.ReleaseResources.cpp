#include <JuceHeader.h>
#include "AudioEngine.h"
#include "NoiseShaperLearner.h"

static void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_RELEASE_RESOURCES)

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

    setShutdownPhase(ShutdownPhase::StopAcceptingWork, "releaseResources");

    // 非MT起点の pending rebuild 要求と AsyncUpdater キューをシャットダウン直後に廃棄する。
    // stopRebuildThread より先に実行して handleAsyncUpdate が後から rebuild を発火しないようにする。
    clearRebuildReason(RebuildReason::StructuralFromNonMT);
    clearRebuildReason(RebuildReason::DeferredStructural);
    clearRebuildReason(RebuildReason::DeferredFinalizeAware);
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
        publishCurrentDSP(nullptr);

        activeToRelease = sanitizeRawPtr(activeDSP.get());
        activeDSP = nullptr;

        fadingToRelease = sanitizeRawPtr(exchangeFadingOutDSP(nullptr));
        convo::publishAtomic(dspCrossfadeUseDryAsOld, false, std::memory_order_release);
        dspCrossfadeDryScaleGain.setCurrentAndTargetValue(1.0);
        convo::publishAtomic(queuedFadeTimeSec, 0.03, std::memory_order_release);

        if (hasPendingTask)
        {
            pendingCurrentToRelease = sanitizeRawPtr(pendingTask.currentDSP);
            pendingTask.currentDSP = nullptr;
            hasPendingTask = false;
        }

        convo::publishAtomic(dspCrossfadePending, false, std::memory_order_release);
        dspCrossfadeGain.setCurrentAndTargetValue(1.0);
        refreshCrossfadePreparedSnapshotFromAtomics();
        publicationCoordinator().publishState(nullptr,
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
    diagLog("[DIAG] releaseResources: after stopRebuildThread");

    setShutdownPhase(ShutdownPhase::ForceEpochAdvance, "releaseResources");
    m_epochDomain.advanceEpoch();

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

    diagLog("[DIAG] releaseResources: before ui processor release");
    diagLog("[DIAG] releaseResources: before uiConvolverProcessor.releaseResources");
    uiConvolverProcessor.releaseResources();
    diagLog("[DIAG] releaseResources: after uiConvolverProcessor.releaseResources");

    diagLog("[DIAG] releaseResources: before uiEqEditor.releaseResources");
    uiEqEditor.releaseResources();
    diagLog("[DIAG] releaseResources: after uiEqEditor.releaseResources");

    diagLog("[DIAG] releaseResources: after ui processor release");

    diagLog("[DIAG] releaseResources: skip deferred reclaim (reconfigure phase)");

    publicationCoordinator().clearPublishedRuntimeSnapshotsNonRt();

    convo::publishAtomic(lifecycleState, EngineLifecycleState::Unprepared, std::memory_order_release);
    diagLog("[DIAG] releaseResources: ABOUT_TO_EXIT_SCOPE");
}

#endif
