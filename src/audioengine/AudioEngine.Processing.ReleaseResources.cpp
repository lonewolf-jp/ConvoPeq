#include <JuceHeader.h>
#include "AudioEngine.h"
#include "NoiseShaperLearner.h"

extern std::atomic<bool> gShuttingDown;

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

    auto previousState = lifecycleState.load(std::memory_order_acquire);
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

        if (lifecycleState.compare_exchange_weak(previousState,
                                                 EngineLifecycleState::Releasing,
                                                 std::memory_order_acq_rel,
                                                 std::memory_order_acquire))
            break;
    }

    shutdownInProgress.store(true, std::memory_order_release);
    setShutdownPhase(ShutdownPhase::StopAcceptingWork, "releaseResources");

    // 非MT起点の pending rebuild 要求と AsyncUpdater キューをシャットダウン直後に廃棄する。
    // stopRebuildThread より先に実行して handleAsyncUpdate が後から rebuild を発火しないようにする。
    pendingStructuralRebuildFromNonMT_.store(false, std::memory_order_release);
    cancelPendingUpdate();
    firstIrDryCrossfadePending.store(false, std::memory_order_release);
    dspCrossfadeUseDryAsOld.store(false, std::memory_order_release);
    const bool finalShutdown = gShuttingDown.load(std::memory_order_acquire);
    if (finalShutdown)
    {
        lastIssuedConvolverStructuralHash_.store(0, std::memory_order_release);
        currentSampleRate.store(0.0);
    }

    inputLevelLinear.store(0.0f);
    outputLevelLinear.store(0.0f);

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
                                     sanitizeRawPtr(fadingOutDSP.load(std::memory_order_acquire)),
                                     nullptr);

        rebuildGeneration.fetch_add(1, std::memory_order_relaxed);
        currentDSP.store(nullptr, std::memory_order_release);

        activeToRelease = sanitizeRawPtr(activeDSP);
        activeDSP = nullptr;

        fadingToRelease = sanitizeRawPtr(fadingOutDSP.exchange(nullptr, std::memory_order_acq_rel));
        dspCrossfadeUseDryAsOld.store(false, std::memory_order_release);
        dspCrossfadeDryScaleGain.setCurrentAndTargetValue(1.0);
        queuedFadeTimeSec.store(0.03, std::memory_order_release);

        if (hasPendingTask)
        {
            pendingCurrentToRelease = sanitizeRawPtr(pendingTask.currentDSP);
            pendingTask.currentDSP = nullptr;
            hasPendingTask = false;
        }

        dspCrossfadePending.store(false, std::memory_order_release);
        dspCrossfadeGain.setCurrentAndTargetValue(1.0);
        publishRuntimeTransitionState(nullptr,
                                      nullptr,
                                      convo::TransitionPolicy::HardReset,
                                      0.0,
                                      false);
        publishRuntimeSnapshots(nullptr,
                    nullptr,
                    convo::TransitionPolicy::HardReset,
                    0.0,
                    false);

        validateDistinctRuntimeSlots("releaseResources.afterClear",
                         activeDSP,
                         sanitizeRawPtr(fadingOutDSP.load(std::memory_order_acquire)),
                         nullptr);
    }

    diagLog("[DIAG] releaseResources: before stopRebuildThread");
    setShutdownPhase(ShutdownPhase::StopWorkers, "releaseResources");
    stopRebuildThread();
    diagLog("[DIAG] releaseResources: after stopRebuildThread");

    setShutdownPhase(ShutdownPhase::ForceEpochAdvance, "releaseResources");
    convo::EpochManager::instance().advanceEpoch();
    g_currentEpoch.store(convo::EpochManager::instance().currentEpoch(), std::memory_order_release);

    setShutdownPhase(ShutdownPhase::DrainRetire, "releaseResources");

    {
        std::queue<CommitStaging> abandonedCommits;
        std::lock_guard<std::mutex> lock(deferredCommitMutex);
        std::swap(abandonedCommits, deferredCommitQueue);

        while (!abandonedCommits.empty())
        {
            auto staging = abandonedCommits.front();
            abandonedCommits.pop();

            if (staging.newDSP)
                retireDSP(staging.newDSP);
            if (staging.oldDSP)
                retireDSP(staging.oldDSP);
        }
    }

    if (activeToRelease)
        retireDSP(activeToRelease);
    if (fadingToRelease)
        retireDSP(fadingToRelease);
    if (pendingNewToRelease)
        retireDSP(pendingNewToRelease);
    if (pendingCurrentToRelease)
        retireDSP(pendingCurrentToRelease);

    diagLog("[DIAG] releaseResources: before ui processor release");
    diagLog("[DIAG] releaseResources: before uiConvolverProcessor.releaseResources");
    uiConvolverProcessor.releaseResources();
    diagLog("[DIAG] releaseResources: after uiConvolverProcessor.releaseResources");

    diagLog("[DIAG] releaseResources: before uiEqEditor.releaseResources");
    uiEqEditor.releaseResources();
    diagLog("[DIAG] releaseResources: after uiEqEditor.releaseResources");

    diagLog("[DIAG] releaseResources: after ui processor release");

    diagLog("[DIAG] releaseResources: skip deferred reclaim (reconfigure phase)");

    clearPublishedRuntimeSnapshotsNonRt();

    lifecycleState.store(EngineLifecycleState::Unprepared, std::memory_order_release);
    diagLog("[DIAG] releaseResources: ABOUT_TO_EXIT_SCOPE");
}

#endif
