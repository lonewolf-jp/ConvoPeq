#include <JuceHeader.h>
#include "AudioEngine.h"

namespace {
static void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}
}

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_TIMER_CALLBACK)

void AudioEngine::timerCallback()
{
    {
        const auto ts = getRuntimeTransitionStateForDebug();
        const int active = ts.active ? 1 : 0;
        const int policy = static_cast<int>(ts.policy);
        const uint64_t currentPtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ts.current));
        const uint64_t nextPtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ts.next));
        const double fadeSec = ts.fadeTimeSec;

        if (active != debugLastReportedTransitionActive
            || policy != debugLastReportedTransitionPolicy
            || currentPtr != debugLastReportedTransitionCurrentPtr
            || nextPtr != debugLastReportedTransitionNextPtr
            || absNoLibm(fadeSec - debugLastReportedTransitionFadeSec) > 1e-9)
        {
            debugLastReportedTransitionActive = active;
            debugLastReportedTransitionPolicy = policy;
            debugLastReportedTransitionCurrentPtr = currentPtr;
            debugLastReportedTransitionNextPtr = nextPtr;
            debugLastReportedTransitionFadeSec = fadeSec;

            diagLog("[VERIFY] transition state active=" + juce::String(active)
                + " policy=" + juce::String(policy)
                + " current=0x" + juce::String::toHexString(static_cast<juce::int64>(currentPtr))
                + " next=0x" + juce::String::toHexString(static_cast<juce::int64>(nextPtr))
                + " fadeSec=" + juce::String(fadeSec, 6));
        }
    }

    // フェイルセーフ: current snapshot が欠落した状態を放置すると
    // EQ変更が演算経路へ乗らないため、Message Thread 側で自己修復する。
    if (!shutdownInProgress.load(std::memory_order_acquire)
        && currentDSP.load(std::memory_order_acquire) != nullptr
        && !m_coordinator.isFading()
        && m_coordinator.getCurrent() == nullptr)
    {
        diagLog("[VERIFY] snapshot bootstrap: current was null, requesting worker snapshot refresh");
        if (!enqueueSnapshotCommand())
            diagLog("[VERIFY] snapshot bootstrap: enqueueSnapshotCommand failed");
    }

    {
        const uint32_t observedVersion = debugAppliedEqHashVersion.load(std::memory_order_acquire);
        debugObservedEqHashVersion = observedVersion;

        const uint64_t createdHash = debugLastCreatedEqHash.load(std::memory_order_acquire);
        const uint64_t appliedHash = debugLastAppliedEqHash.load(std::memory_order_acquire);
        const uint64_t createBlockCounter = debugLastCreateAudioBlockCounter.load(std::memory_order_acquire);
        const uint64_t nowBlockCounter = m_audioBlockCounter.load(std::memory_order_acquire);
        const uint64_t processedBlocksSinceCreate = (nowBlockCounter >= createBlockCounter)
            ? (nowBlockCounter - createBlockCounter)
            : 0;
        const int dspReady = (currentDSP.load(std::memory_order_acquire) != nullptr) ? 1 : 0;
        const int coordIsFading = m_coordinator.isFading() ? 1 : 0;
        const int updateFadeReturned = coordIsFading;
        const int fromNull = (m_coordinator.getCurrent() == nullptr) ? 1 : 0;
        const int toNull = -1;

        const bool eqMismatch = (createdHash != 0 && createdHash != appliedHash && dspReady == 1);
        const bool recoveryEligible = eqMismatch
            && coordIsFading == 0
            && processedBlocksSinceCreate >= 4;
        if (recoveryEligible)
        {
            if (debugLastRecoveryAttemptCreatedEqHash != createdHash)
            {
                debugRecoveryRetryCountForCurrentHash = 0;
                debugRecoverySuppressedForCurrentHash = false;
            }

            const bool firstAttemptForThisHash = (debugLastRecoveryAttemptCreatedEqHash != createdHash);
            const bool retryAfterProgress = (nowBlockCounter > debugLastRecoveryAttemptAudioBlockCounter + 256);
            if (firstAttemptForThisHash || retryAfterProgress)
            {
                debugLastRecoveryAttemptCreatedEqHash = createdHash;
                debugLastRecoveryAttemptAudioBlockCounter = nowBlockCounter;

                if (debugRecoveryRetryCountForCurrentHash < 3)
                {
                    ++debugRecoveryRetryCountForCurrentHash;
#ifdef _DEBUG
                    const uint64_t workerRecv = m_workerThread.getCommandsReceived();
                    const uint64_t workerSnap = m_workerThread.getSnapshotsCreated();
                    const uint64_t workerDrop = m_workerThread.getCommandsDropped();
#else
                    const uint64_t workerRecv = 0;
                    const uint64_t workerSnap = 0;
                    const uint64_t workerDrop = 0;
#endif
                    diagLog("[VERIFY] snapshot recovery: retry=" + juce::String(debugRecoveryRetryCountForCurrentHash)
                        + " forcing reapply createdHash=0x"
                        + juce::String::toHexString(static_cast<juce::int64>(createdHash))
                        + " appliedHash=0x"
                        + juce::String::toHexString(static_cast<juce::int64>(appliedHash))
                        + " blocksSinceCreate=" + juce::String(static_cast<juce::int64>(processedBlocksSinceCreate))
                        + " workerRecv=" + juce::String(static_cast<juce::int64>(workerRecv))
                        + " workerSnap=" + juce::String(static_cast<juce::int64>(workerSnap))
                        + " workerDrop=" + juce::String(static_cast<juce::int64>(workerDrop))
                        + " genNow=" + juce::String(static_cast<juce::int64>(m_generationManager.getCurrentGeneration())));

                    if (!enqueueSnapshotCommand())
                        diagLog("[VERIFY] snapshot recovery: enqueueSnapshotCommand failed");
                }
                else if (!debugRecoverySuppressedForCurrentHash)
                {
                    debugRecoverySuppressedForCurrentHash = true;
                    diagLog("[VERIFY] snapshot recovery: suppressed for createdHash=0x"
                        + juce::String::toHexString(static_cast<juce::int64>(createdHash))
                        + " after 3 retries (waiting for next hash change)");
                }
            }
        }
        else
        {
            debugRecoveryRetryCountForCurrentHash = 0;
            debugRecoverySuppressedForCurrentHash = false;
        }

        if (createdHash != debugLastReportedCreatedEqHash ||
            appliedHash != debugLastReportedAppliedEqHash ||
            dspReady != debugLastReportedDspReady)
        {
            debugLastReportedCreatedEqHash = createdHash;
            debugLastReportedAppliedEqHash = appliedHash;
            debugLastReportedDspReady = dspReady;
            diagLog("[VERIFY] EQ reflection createdHash=0x"
                + juce::String::toHexString(static_cast<juce::int64>(createdHash))
                + " appliedHash=0x"
                + juce::String::toHexString(static_cast<juce::int64>(appliedHash))
                + " matched=" + juce::String((int)(createdHash == appliedHash))
                + " ver=" + juce::String((int)observedVersion)
                + " blocksSinceCreate=" + juce::String(static_cast<juce::int64>(processedBlocksSinceCreate))
                + " dspReady=" + juce::String(dspReady)
                + " coordFading=" + juce::String(coordIsFading)
                + " updRet=" + juce::String(updateFadeReturned)
                + " fromNull=" + juce::String(fromNull)
                + " toNull=" + juce::String(toNull));
        }
    }

    if (!shutdownInProgress.load(std::memory_order_acquire)
        && deferredStructuralRebuildPending_.load(std::memory_order_acquire))
    {
        const int64_t dueTicks = deferredStructuralRebuildDueTicks_.load(std::memory_order_acquire);
        const int64_t nowTicks = juce::Time::getHighResolutionTicks();

        if (dueTicks > 0 && nowTicks >= dueTicks)
        {
            deferredStructuralRebuildPending_.store(false, std::memory_order_release);
            deferredStructuralRebuildDueTicks_.store(0, std::memory_order_release);

            if (uiConvolverProcessor.isIRLoaded())
            {
                diagLog("[DIAG] timerCallback: issuing deferred Structural rebuild after prepared IR apply");
                requestRebuild(convo::RebuildKind::Structural);

                ++pendingIRGeneration;
                setIRChangeFlag();

                const LearningCommand cmd {
                    LearningCommand::Type::IRChanged,
                    false,
                    pendingLearningMode.load(std::memory_order_acquire),
                    pendingIRGeneration
                };

                if (!enqueueLearningCommand(cmd))
                {
                    DBG("[AudioEngine] timerCallback: deferred command queue overflow");
                }
            }
        }
    }

    if (!shutdownInProgress.load(std::memory_order_acquire)
        && deferredFinalizeAwareRebuildPending_.load(std::memory_order_acquire))
    {
        const int queuedGeneration = rebuildGeneration.load(std::memory_order_acquire);
        const int committedGeneration = lastCommittedRebuildGeneration.load(std::memory_order_acquire);
        const bool outstandingRebuild = queuedGeneration > committedGeneration;
        const bool irLoaded = uiConvolverProcessor.isIRLoaded();
        const bool irFinalized = uiConvolverProcessor.isIRFinalized();
        const bool irLoading = uiConvolverProcessor.isLoadingIR();
        const bool structuralDeferred = deferredStructuralRebuildPending_.load(std::memory_order_acquire);
        const bool pendingIrChange = m_pendingIRChange.load(std::memory_order_acquire);

        // IR 遷移が完全に落ち着いてから 1 回だけ再構築を発火する。
        if ((!irLoaded || irFinalized)
            && !irLoading
            && !structuralDeferred
            && !pendingIrChange
            && !outstandingRebuild)
        {
            deferredFinalizeAwareRebuildPending_.store(false, std::memory_order_release);

            const double sr = currentSampleRate.load(std::memory_order_acquire);
            if (!m_isRestoringState && sr > 0.0)
            {
                diagLog("[DIAG] timerCallback: issuing deferred finalize-aware rebuild");
                requestRebuild(sr, maxSamplesPerBlock.load(std::memory_order_acquire));
            }
        }
    }

    processLearningCommands();
    processDeferredLearningActions();

    if (!shutdownInProgress.load(std::memory_order_acquire) &&
        sanitizeRawPtr(fadingOutDSP.load(std::memory_order_acquire)) == nullptr &&
        !dspCrossfadePending.load(std::memory_order_acquire) &&
        fadeQueued.exchange(false, std::memory_order_acq_rel))
    {
        if (auto* queued = sanitizeRawPtr(queuedOldDSP.exchange(nullptr, std::memory_order_acq_rel)))
        {
            const double fadeSec = queuedNextFadeTimeSec.load(std::memory_order_acquire);
            queuedFadeTimeSec.store(fadeSec, std::memory_order_release);
            fadingOutDSP.store(queued, std::memory_order_release);
            dspCrossfadePending.store(true, std::memory_order_release);
            setIRChangeFlag();
        }
    }

    // Grace period に基づく安全なリリース遅延を実行する。
    processDeferredReleases();

    if (m_coordinator.tryCompleteFade())
    {
        sendChangeMessage();
    }

    // 内部プロセッサのクリーンアップを実行する。
    if (auto* dsp = currentDSP.load(std::memory_order_acquire))
    {
        dsp->eq.cleanup();
        dsp->convolver.cleanup();

        const bool activeFixed4Tap = (dsp->noiseShaperType == NoiseShaperType::Fixed4Tap);
        const bool activeFixed15Tap = (dsp->noiseShaperType == NoiseShaperType::Fixed15Tap);
        const bool activeDitherEnabled = (dsp->ditherBitDepth > 0);

        if (activeFixed4Tap && activeDitherEnabled)
        {
            dsp->fixedNoiseShaper.setDiagnosticsWindowSamples(
                static_cast<uint32_t>(fixedNoiseWindowSamples.load(std::memory_order_relaxed)));

            const uint32 now = juce::Time::getMillisecondCounter();
            const uint32 intervalMs = static_cast<uint32>(
                std::max(250, fixedNoiseLogIntervalMs.load(std::memory_order_relaxed)));
            if ((now - fixedNoiseLastLogMs) >= intervalMs)
            {
                fixedNoiseLastLogMs = now;
                const auto diag = dsp->fixedNoiseShaper.getDiagnostics();
                if (diag.windowSamples > 0)
                {
                    DBG_LOG(
                        "[Fixed4Tap] bitDepth=" + juce::String(diag.bitDepth)
                        + " rmsL=" + juce::String(diag.rmsErrorL, 9)
                        + " rmsR=" + juce::String(diag.rmsErrorR, 9)
                        + " peak=" + juce::String(diag.peakAbsError, 9)
                        + " windowSamples=" + juce::String((int)diag.windowSamples));
                }
                else
                {
                    DBG_LOG(
                        "[Fixed4Tap] waiting for diagnostics window"
                        " (bitDepth=" + juce::String(dsp->ditherBitDepth)
                        + ", targetWindow=" + juce::String(fixedNoiseWindowSamples.load(std::memory_order_relaxed))
                        + ")");
                }
            }
        }
        else if (activeFixed15Tap && activeDitherEnabled)
        {
            dsp->fixed15TapNoiseShaper.setDiagnosticsWindowSamples(
                static_cast<uint32_t>(fixedNoiseWindowSamples.load(std::memory_order_relaxed)));

            const uint32 now = juce::Time::getMillisecondCounter();
            const uint32 intervalMs = static_cast<uint32>(
                std::max(250, fixedNoiseLogIntervalMs.load(std::memory_order_relaxed)));
            if ((now - fixedNoiseLastLogMs) >= intervalMs)
            {
                fixedNoiseLastLogMs = now;
                const auto diag = dsp->fixed15TapNoiseShaper.getDiagnostics();
                if (diag.windowSamples > 0)
                {
                    DBG_LOG(
                        "[Fixed15Tap] bitDepth=" + juce::String(diag.bitDepth)
                        + " rmsL=" + juce::String(diag.rmsErrorL, 9)
                        + " rmsR=" + juce::String(diag.rmsErrorR, 9)
                        + " peak=" + juce::String(diag.peakAbsError, 9)
                        + " windowSamples=" + juce::String((int)diag.windowSamples));
                }
                else
                {
                    DBG_LOG(
                        "[Fixed15Tap] waiting for diagnostics window"
                        " (bitDepth=" + juce::String(dsp->ditherBitDepth)
                        + ", targetWindow=" + juce::String(fixedNoiseWindowSamples.load(std::memory_order_relaxed))
                        + ")");
                }
            }
        }
    }

    // UI用プロセッサのクリーンアップ
    uiEqEditor.cleanup();
    uiConvolverProcessor.cleanup();
}

#endif // CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_TIMER_CALLBACK
