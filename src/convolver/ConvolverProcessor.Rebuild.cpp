#include <JuceHeader.h>
#include "ConvolverProcessor.h"
#include "convolver/ConvolverProcessor.Internal.h"

#if defined(CONVOPEQ_ENABLE_CONVOLVER_SPLIT_REBUILD)

// ────────────────────────────────────────────────────────────────
// Rebuild (debounced, incremental)
// ────────────────────────────────────────────────────────────────

void ConvolverProcessor::rebuildAllIRs()
{
    if (isIRLoaded() && !isLoading.load())
    {
        loadImpulseResponse(juce::File(), false);
    }
}

void ConvolverProcessor::postCoalescedChangeNotification()
{
    if (changeNotificationPending.exchange(true, std::memory_order_acq_rel))
        return;

    auto weakThis = juce::WeakReference<ConvolverProcessor>(this);
    const auto dispatchNotification = [weakThis]()
    {
        if (auto* self = weakThis.get())
        {
            self->changeNotificationPending.store(false, std::memory_order_release);
            self->sendChangeMessage();
        }
    };

    const bool queued = juce::MessageManager::callAsync(dispatchNotification);
    if (!queued)
    {
        if (auto* self = weakThis.get())
            self->changeNotificationPending.store(false, std::memory_order_release);
    }
}

void ConvolverProcessor::requestDebouncedRebuild()
{
    debugDebouncedRebuildRequestCount.fetch_add(1, std::memory_order_relaxed);

    if (!isIRLoaded())
    {
        if (isLoading.load(std::memory_order_acquire) || isRebuilding.load(std::memory_order_acquire))
        {
            debugDebouncedRebuildDeferredAfterLoadCount.fetch_add(1, std::memory_order_relaxed);
            rebuildPendingAfterLoad.store(true, std::memory_order_release);
        }
        return;
    }

    if (!isIRFinalized())
    {
        debugDebouncedRebuildDeferredAfterLoadCount.fetch_add(1, std::memory_order_relaxed);
        rebuildPendingAfterLoad.store(true, std::memory_order_release);
        return;
    }

    const std::uint64_t token = rebuildDebounceToken.fetch_add(1, std::memory_order_acq_rel) + 1;
    auto weakThis = juce::WeakReference<ConvolverProcessor>(this);

    const int debounceMs = juce::jlimit(REBUILD_DEBOUNCE_MIN_MS,
                                        REBUILD_DEBOUNCE_MAX_MS,
                                        rebuildDebounceMs.load(std::memory_order_acquire));

    juce::Timer::callAfterDelay(debounceMs, [weakThis, token]()
    {
        if (auto* self = weakThis.get())
        {
            if (self->rebuildDebounceToken.load(std::memory_order_acquire) != token)
                return;

            if (!self->isIRLoaded() || self->isLoading.load(std::memory_order_acquire))
                return;

            self->debugDebouncedRebuildTriggeredCount.fetch_add(1, std::memory_order_relaxed);
            self->loadImpulseResponse(juce::File());
        }
    });

    debugDebouncedRebuildScheduledCount.fetch_add(1, std::memory_order_relaxed);
}

void ConvolverProcessor::rebuildAllIRsSynchronous(std::function<bool()> shouldCancel)
{
    auto stageToString = [](IncrementalRebuildJob::Stage stage) -> const char*
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

        auto runLegacyPath = [&]()
        {
            const double processingSampleRate = currentSampleRate.load(std::memory_order_acquire);
            LoaderThread loader(*this, *(state->ir), state->sampleRate, processingSampleRate, currentBufferSize.load(std::memory_order_acquire), getPhaseMode(),
                        mixedTransitionStartHz.load(std::memory_order_acquire), mixedTransitionEndHz.load(std::memory_order_acquire),
                        mixedPreRingTau.load(std::memory_order_acquire), currentIRScale.load(std::memory_order_acquire));
            loader.externalCancellationCheck = shouldCancel;
            loader.runSynchronously();
        };

        runLegacyPath();
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
    auto stageToString = [](IncrementalRebuildJob::Stage stage) -> const char*
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
        const double processingSampleRate = currentSampleRate.load(std::memory_order_acquire);
        job.incrementalLoader = std::make_unique<LoaderThread>(
            *this,
            *(job.preparedIR),
            job.preparedSampleRate,
            processingSampleRate,
            currentBufferSize.load(std::memory_order_acquire),
            getPhaseMode(),
            mixedTransitionStartHz.load(std::memory_order_acquire),
            mixedTransitionEndHz.load(std::memory_order_acquire),
            mixedPreRingTau.load(std::memory_order_acquire),
            currentIRScale.load(std::memory_order_acquire));
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
    auto stageToString = [](IncrementalRebuildJob::Stage stage) -> const char*
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

    auto loadedIRShared = std::make_shared<juce::AudioBuffer<double>>(std::move(job.pendingLoadedIR));
    auto displayIRShared = std::make_shared<juce::AudioBuffer<double>>(std::move(job.pendingDisplayIR));
    StereoConvolver *conv = std::exchange(job.pendingConv, nullptr);

    applyNewState(conv, loadedIRShared, job.pendingLoadedSR, job.pendingTargetLength,
                  job.pendingIsRebuild, job.pendingFile, job.pendingScaleFactor, displayIRShared);

    job.finalizeApplied = true;
    job.lastError.clear();
    job.stage = IncrementalRebuildJob::Stage::Done;

    return true;
}

void ConvolverProcessor::setUseIncrementalRebuild(bool enable) noexcept
{
    juce::ignoreUnused(enable);
    useIncrementalRebuild.store(false, std::memory_order_release);
    if (rebuildJob)
        rebuildJob->reset();
}

bool ConvolverProcessor::isIncrementalRebuildEnabled() const noexcept
{
    return useIncrementalRebuild.load(std::memory_order_acquire);
}

void ConvolverProcessor::invalidatePendingLoads()
{
    if (rebuildJob)
        rebuildJob->reset();
}

#endif // CONVOPEQ_ENABLE_CONVOLVER_SPLIT_REBUILD
