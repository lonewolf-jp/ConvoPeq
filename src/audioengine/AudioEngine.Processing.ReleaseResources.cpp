#include <JuceHeader.h>
#include "AudioEngine.h"

extern std::atomic<bool> gShuttingDown;

static void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}

static void retireDSP(AudioEngine::DSPCore* dsp)
{
    if (dsp) convo::retireObject(dsp, [](void* p) { delete static_cast<AudioEngine::DSPCore*>(p); });
}

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_RELEASE_RESOURCES)

void AudioEngine::releaseResources()
{
    diagLog("[DIAG] releaseResources: enter");
    shutdownInProgress.store(true, std::memory_order_release);
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

    DSPCore* activeToRelease = nullptr;
    DSPCore* fadingToRelease = nullptr;
    DSPCore* queuedToRelease = nullptr;
    DSPCore* pendingNewToRelease = nullptr;
    DSPCore* pendingCurrentToRelease = nullptr;

    {
        std::lock_guard<std::mutex> lk(rebuildMutex);
        rebuildGeneration.fetch_add(1, std::memory_order_relaxed);
        currentDSP.store(nullptr, std::memory_order_release);

        activeToRelease = sanitizeRawPtr(activeDSP);
        activeDSP = nullptr;

        fadingToRelease = sanitizeRawPtr(fadingOutDSP.exchange(nullptr, std::memory_order_acq_rel));
        queuedToRelease = sanitizeRawPtr(queuedOldDSP.exchange(nullptr, std::memory_order_acq_rel));
        fadeQueued.store(false, std::memory_order_release);
        dspCrossfadeUseDryAsOld.store(false, std::memory_order_release);
        dspCrossfadeDryScaleGain.setCurrentAndTargetValue(1.0);
        queuedFadeTimeSec.store(0.03, std::memory_order_release);
        queuedNextFadeTimeSec.store(0.03, std::memory_order_release);

        if (hasPendingTask)
        {
            pendingNewToRelease = sanitizeRawPtr(pendingTask.newDSP);
            pendingTask.newDSP = nullptr;
            pendingCurrentToRelease = sanitizeRawPtr(pendingTask.currentDSP);
            pendingTask.currentDSP = nullptr;
            hasPendingTask = false;
        }

        dspCrossfadePending.store(false, std::memory_order_release);
        dspCrossfadeGain.setCurrentAndTargetValue(1.0);
    }

    diagLog("[DIAG] releaseResources: before stopRebuildThread");
    stopRebuildThread();
    diagLog("[DIAG] releaseResources: after stopRebuildThread");

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
    if (queuedToRelease)
        retireDSP(queuedToRelease);
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

    diagLog("[DIAG] releaseResources: ABOUT_TO_EXIT_SCOPE");
}

#endif
