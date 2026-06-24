#include <JuceHeader.h>
#include "ConvolverProcessor.h"
#include "convolver/ConvolverProcessor.Internal.h"

#include "audioengine/AtomicAccess.h"

#if defined(CONVOPEQ_ENABLE_CONVOLVER_SPLIT_REBUILD)

// ────────────────────────────────────────────────────────────────
// Rebuild (debounced, incremental)
// ────────────────────────────────────────────────────────────────

void ConvolverProcessor::rebuildAllIRs()
{
    if (isIRLoaded() && !convo::consumeAtomic(isLoading))
    {
        loadImpulseResponse(juce::File(), false);
    }
}

void ConvolverProcessor::postCoalescedChangeNotification()
{
    if (convo::exchangeAtomic(changeNotificationPending, true, std::memory_order_acq_rel)) // acq_rel: acquire で先行状態観測; release で pending=true 公開
        return;

    auto weakThis = juce::WeakReference<ConvolverProcessor>(this);
    const auto dispatchNotification = [weakThis]()
    {
        if (auto* self = weakThis.get())
        {
            convo::publishAtomic(self->changeNotificationPending, false, std::memory_order_release); // release: pending=false 公開; 次回 exchangeAtomic acquire と HB
            self->sendChangeMessage();
        }
    };

    const bool queued = juce::MessageManager::callAsync(dispatchNotification);
    if (!queued)
    {
        if (auto* self = weakThis.get())
            convo::publishAtomic(self->changeNotificationPending, false, std::memory_order_release); // release: pending=false 公開（callAsync失敗時）
    }
}

void ConvolverProcessor::rebuildAllIRsSynchronous(std::function<bool()> shouldCancel)
{
    [[maybe_unused]] auto stageToString = [](IncrementalRebuildJob::Stage stage) -> const char*
    {
        switch (stage)
        {
            case IncrementalRebuildJob::Stage::Idle: return "Idle";
            case IncrementalRebuildJob::Stage::Prepared: return "Prepared";
            case IncrementalRebuildJob::Stage::Building: return "Building";
            case IncrementalRebuildJob::Stage::FinalizingPrepare: return "FinalizingPrepare";
            case IncrementalRebuildJob::Stage::FinalizingApply: return "FinalizingApply";
            case IncrementalRebuildJob::Stage::Done: return "Done";
            default: return "Unknown";
        }
    };

    const IRState* state = acquireIRState();
    if (state && state->ir && state->ir->getNumSamples() > 0 && state->sampleRate > 0.0)
    {
        if (shouldCancel && shouldCancel())
        {
            if (rebuildJob)
                rebuildJob->reset();
            releaseIRState(state);
            return;
        }

        auto runRebuildPath = [&]()
        {
            const double processingSampleRate = convo::consumeAtomic(currentSampleRate, std::memory_order_acquire); // acquire: prepareToPlay/applyNewState の publishAtomic release と HB
            const BuildSnapshot buildSnapshot = captureBuildSnapshot();
            const int clampedPhaseMode = juce::jlimit(static_cast<int>(PhaseMode::AsIs),
                                                      static_cast<int>(PhaseMode::Minimum),
                                                      buildSnapshot.phaseMode);
            const float clampedMixedF1 = juce::jlimit(MIXED_F1_MIN_HZ,
                                                      MIXED_F1_MAX_HZ,
                                                      buildSnapshot.mixedTransitionStartHz);
            const float clampedMixedF2 = juce::jlimit((std::max)(MIXED_F2_MIN_HZ, clampedMixedF1 + 10.0f),
                                                      MIXED_F2_MAX_HZ,
                                                      buildSnapshot.mixedTransitionEndHz);
            LoaderThread loader(*this, *(state->ir), state->sampleRate, processingSampleRate, convo::consumeAtomic(currentBufferSize, std::memory_order_acquire), static_cast<PhaseMode>(clampedPhaseMode), // acquire: prepareToPlay の publishAtomic release と HB
                        clampedMixedF1, clampedMixedF2,
                        convo::consumeAtomic(currentIRScale, std::memory_order_acquire), // acquire: applyNewState の publishAtomic release と HB
                        buildSnapshot);
            loader.externalCancellationCheck = shouldCancel;
            loader.runSynchronously();
        };

        runRebuildPath();

        juce::Logger::writeToLog("[CONV_REBUILD] rebuildAllIRsSynchronous: engine rebuilt"
            " len=" + juce::String(state->ir->getNumSamples())
            + " ch=" + juce::String(state->ir->getNumChannels())
            + " srcSR=" + juce::String(state->sampleRate, 1));
    }

    if (rebuildJob)
        rebuildJob->reset();

    releaseIRState(state);
}

void ConvolverProcessor::IncrementalRebuildJob::reset() noexcept
{
    if (pendingConv != nullptr)
    {
        pendingConv->~StereoConvolver();
        convo::aligned_free(pendingConv);
        pendingConv = nullptr;
    }

    stage = Stage::Idle;
    preparedIR.reset();
    preparedSampleRate = 0.0;
    shouldCancel = nullptr;
    incrementalLoader.reset();
    loaderInitialized = false;
    pendingLoadedIR.setSize(0, 0);
    pendingLoadedSR = 0.0;
    pendingTargetLength = 0;
    pendingDisplayIR.setSize(0, 0);
    pendingScaleFactor = 1.0;
    pendingFile = juce::File();
    pendingIsRebuild = false;
    finalizeApplied = false;
    lastError.clear();
}

bool ConvolverProcessor::runIncrementalBuildStep(IncrementalRebuildJob& job)
{
    [[maybe_unused]] auto stageToString = [](IncrementalRebuildJob::Stage stage) -> const char*
    {
        switch (stage)
        {
            case IncrementalRebuildJob::Stage::Idle: return "Idle";
            case IncrementalRebuildJob::Stage::Prepared: return "Prepared";
            case IncrementalRebuildJob::Stage::Building: return "Building";
            case IncrementalRebuildJob::Stage::FinalizingPrepare: return "FinalizingPrepare";
            case IncrementalRebuildJob::Stage::FinalizingApply: return "FinalizingApply";
            case IncrementalRebuildJob::Stage::Done: return "Done";
            default: return "Unknown";
        }
    };

    if (!job.preparedIR || job.preparedIR->getNumSamples() <= 0 || job.preparedSampleRate <= 0.0)
    {
        job.lastError = "incremental build: prepared payload is not ready";
        return false;
    }

    if (job.shouldCancel && job.shouldCancel())
    {
        job.lastError = "incremental build: canceled by callback";
        return false;
    }

    if (!job.loaderInitialized)
    {
        const double processingSampleRate = convo::consumeAtomic(currentSampleRate, std::memory_order_acquire); // acquire: prepareToPlay/applyNewState の publishAtomic release と HB
        const BuildSnapshot buildSnapshot = captureBuildSnapshot();
        const int clampedPhaseMode = juce::jlimit(static_cast<int>(PhaseMode::AsIs),
                                                  static_cast<int>(PhaseMode::Minimum),
                                                  buildSnapshot.phaseMode);
        const float clampedMixedF1 = juce::jlimit(MIXED_F1_MIN_HZ,
                                                  MIXED_F1_MAX_HZ,
                                                  buildSnapshot.mixedTransitionStartHz);
        const float clampedMixedF2 = juce::jlimit((std::max)(MIXED_F2_MIN_HZ, clampedMixedF1 + 10.0f),
                                                  MIXED_F2_MAX_HZ,
                                                  buildSnapshot.mixedTransitionEndHz);
        job.incrementalLoader = std::make_unique<LoaderThread>(
            *this,
            *(job.preparedIR),
            job.preparedSampleRate,
            processingSampleRate,
            convo::consumeAtomic(currentBufferSize, std::memory_order_acquire), // acquire: prepareToPlay の publishAtomic release と HB
            static_cast<PhaseMode>(clampedPhaseMode),
            clampedMixedF1,
            clampedMixedF2,
            convo::consumeAtomic(currentIRScale, std::memory_order_acquire), // acquire: applyNewState の publishAtomic release と HB
            buildSnapshot);
        job.incrementalLoader->externalCancellationCheck = job.shouldCancel;
        job.loaderInitialized = true;
    }

    auto& loader = *job.incrementalLoader;

    const bool terminal = loader.stepOnce();

    if (loader.hasError())
    {
        job.lastError = loader.getStepResult().errorMessage;
        return false;
    }

    if (job.shouldCancel && job.shouldCancel())
    {
        job.lastError = "incremental build: canceled between steps";
        return false;
    }

    if (!terminal)
    {
        job.stage = IncrementalRebuildJob::Stage::Building;
        job.lastError.clear();
        return true;
    }

    LoaderThread::LoadResult& result = loader.getStepResult();

    if (job.pendingConv != nullptr)
    {
        retireStereoConvolver(std::exchange(job.pendingConv, nullptr), 0);
    }

    job.pendingConv = std::exchange(result.newConv, nullptr);
    job.pendingLoadedIR    = std::move(result.loadedIR);
    job.pendingLoadedSR    = result.loadedSR;
    job.pendingTargetLength = result.targetLength;
    job.pendingDisplayIR   = std::move(result.displayIR);
    job.pendingScaleFactor = result.scaleFactor;
    job.pendingFile        = juce::File();
    job.pendingIsRebuild   = true;
    job.stage              = IncrementalRebuildJob::Stage::FinalizingApply;
    job.finalizeApplied    = false;
    job.lastError.clear();

    return true;
}

bool ConvolverProcessor::runIncrementalFinalizeStep(IncrementalRebuildJob& job)
{
    [[maybe_unused]] auto stageToString = [](IncrementalRebuildJob::Stage stage) -> const char*
    {
        switch (stage)
        {
            case IncrementalRebuildJob::Stage::Idle: return "Idle";
            case IncrementalRebuildJob::Stage::Prepared: return "Prepared";
            case IncrementalRebuildJob::Stage::Building: return "Building";
            case IncrementalRebuildJob::Stage::FinalizingPrepare: return "FinalizingPrepare";
            case IncrementalRebuildJob::Stage::FinalizingApply: return "FinalizingApply";
            case IncrementalRebuildJob::Stage::Done: return "Done";
            default: return "Unknown";
        }
    };

    if (job.stage != IncrementalRebuildJob::Stage::FinalizingApply)
    {
        job.lastError = "incremental finalize: invalid stage";
        return false;
    }

    if (job.pendingConv == nullptr || job.pendingTargetLength <= 0)
    {
        job.lastError = "incremental finalize: pending payload is not ready";
        return false;
    }

    auto loadedIR = std::make_unique<juce::AudioBuffer<double>>(std::move(job.pendingLoadedIR));
    auto displayIR = std::make_unique<juce::AudioBuffer<double>>(std::move(job.pendingDisplayIR));
    StereoConvolver *conv = std::exchange(job.pendingConv, nullptr);

    applyNewState(conv, std::move(loadedIR), job.pendingLoadedSR, job.pendingTargetLength,
                  job.pendingIsRebuild, job.pendingFile, job.pendingScaleFactor, std::move(displayIR));

    job.finalizeApplied = true;
    job.lastError.clear();
    job.stage = IncrementalRebuildJob::Stage::Done;

    return true;
}

void ConvolverProcessor::setUseIncrementalRebuild(bool enable) noexcept
{
    juce::ignoreUnused(enable);
    convo::publishAtomic(useIncrementalRebuild, false, std::memory_order_release); // release: isIncrementalRebuildEnabled 側 acquire と HB
    if (rebuildJob)
        rebuildJob->reset();
}

[[nodiscard]] bool ConvolverProcessor::isIncrementalRebuildEnabled() const noexcept
{
    return convo::consumeAtomic(useIncrementalRebuild, std::memory_order_acquire); // acquire: setUseIncrementalRebuild の publishAtomic release と HB
}

void ConvolverProcessor::invalidatePendingLoads()
{
    if (rebuildJob)
        rebuildJob->reset();
}

#endif // CONVOPEQ_ENABLE_CONVOLVER_SPLIT_REBUILD
