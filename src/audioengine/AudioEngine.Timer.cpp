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
        const int latencyDelta = ts.latencyDeltaSamples;

        if (active != debugLastReportedTransitionActive
            || policy != debugLastReportedTransitionPolicy
            || currentPtr != debugLastReportedTransitionCurrentPtr
            || nextPtr != debugLastReportedTransitionNextPtr
            || absNoLibm(fadeSec - debugLastReportedTransitionFadeSec) > 1e-9
            || latencyDelta != debugLastReportedTransitionLatencyDeltaSamples)
        {
            debugLastReportedTransitionActive = active;
            debugLastReportedTransitionPolicy = policy;
            debugLastReportedTransitionCurrentPtr = currentPtr;
            debugLastReportedTransitionNextPtr = nextPtr;
            debugLastReportedTransitionLatencyDeltaSamples = latencyDelta;

            diagLog("[VERIFY] transition state active=" + juce::String(active)
                + " policy=" + juce::String(policy)
                + " current=0x" + juce::String::toHexString(static_cast<juce::int64>(currentPtr))
                + " next=0x" + juce::String::toHexString(static_cast<juce::int64>(nextPtr))
                + " fadeSec=" + juce::String(fadeSec, 6)
                + " latencyDelta=" + juce::String(latencyDelta));
        }

        const int alignWritePos = convo::consumeAtomic(debugLatencyAlignWritePos, std::memory_order_acquire);
        const int alignReadOld = convo::consumeAtomic(debugLatencyAlignReadOld, std::memory_order_acquire);
        const int alignReadNew = convo::consumeAtomic(debugLatencyAlignReadNew, std::memory_order_acquire);
        const int alignDelayOld = convo::consumeAtomic(debugLatencyAlignDelayOld, std::memory_order_acquire);
        const int alignDelayNew = convo::consumeAtomic(debugLatencyAlignDelayNew, std::memory_order_acquire);

        if (active != 0
            && (alignWritePos != debugLastReportedLatencyAlignWritePos
                || alignReadOld != debugLastReportedLatencyAlignReadOld
                || alignReadNew != debugLastReportedLatencyAlignReadNew
                || alignDelayOld != debugLastReportedLatencyAlignDelayOld
                || alignDelayNew != debugLastReportedLatencyAlignDelayNew
                || (alignDelayOld - alignDelayNew) != latencyDelta))
        {
            debugLastReportedLatencyAlignWritePos = alignWritePos;
            debugLastReportedLatencyAlignReadOld = alignReadOld;
            debugLastReportedLatencyAlignReadNew = alignReadNew;
            debugLastReportedLatencyAlignDelayOld = alignDelayOld;
            debugLastReportedLatencyAlignDelayNew = alignDelayNew;

            diagLog("[VERIFY] latency align writePos=" + juce::String(alignWritePos)
                + " readOld=" + juce::String(alignReadOld)
                + " readNew=" + juce::String(alignReadNew)
                + " delayOld=" + juce::String(alignDelayOld)
                + " delayNew=" + juce::String(alignDelayNew)
                + " deltaFromDelay=" + juce::String(alignDelayOld - alignDelayNew)
                + " transitionDelta=" + juce::String(latencyDelta));
        }
    }

    {
        const auto runtimePublishView = getRuntimePublishView();
        const uint64_t revision = runtimeRevision(runtimePublishView.engine, runtimePublishView.graph);
        const uint64_t currentUuid = runtimeCurrentUuid(runtimePublishView.engine, runtimePublishView.graph);
        const uint64_t fadingUuid = runtimeFadingUuid(runtimePublishView.engine, runtimePublishView.graph);
        const uint64_t transitionCurrentUuid = runtimeTransitionCurrentUuid(runtimePublishView.engine, runtimePublishView.graph);
        const uint64_t transitionNextUuid = runtimeTransitionNextUuid(runtimePublishView.engine, runtimePublishView.graph);

        if (revision != debugLastReportedRuntimeSnapshotRevision
            || currentUuid != debugLastReportedRuntimePublishCurrentUuid
            || fadingUuid != debugLastReportedRuntimePublishFadingUuid
            || transitionCurrentUuid != debugLastReportedRuntimePublishTransitionCurrentUuid
            || transitionNextUuid != debugLastReportedRuntimePublishTransitionNextUuid)
        {
            debugLastReportedRuntimeSnapshotRevision = revision;
            debugLastReportedRuntimePublishCurrentUuid = currentUuid;
            debugLastReportedRuntimePublishFadingUuid = fadingUuid;
            debugLastReportedRuntimePublishTransitionCurrentUuid = transitionCurrentUuid;
            debugLastReportedRuntimePublishTransitionNextUuid = transitionNextUuid;

            diagLog("[VERIFY] runtime publish rev=" + juce::String(static_cast<juce::int64>(revision))
                + " currentUuid=" + juce::String(static_cast<juce::int64>(currentUuid))
                + " fadingUuid=" + juce::String(static_cast<juce::int64>(fadingUuid))
                + " transition(" + juce::String(static_cast<juce::int64>(transitionCurrentUuid))
                + "->" + juce::String(static_cast<juce::int64>(transitionNextUuid)) + ")");
        }
    }

    {
        const auto lifecycle = getRuntimeLifecycleDiagnostics();
        const auto rebuild = getRebuildDispatchDiagnostics();
        const auto convRebuild = uiConvolverProcessor.getRebuildAutomationDiagnostics();
        const int shutdownPhaseValue = static_cast<int>(convo::consumeAtomic(shutdownPhase, std::memory_order_acquire));

        if (lifecycle.publishCount != debugLastReportedRuntimePublishCount
            || lifecycle.retireCount != debugLastReportedRuntimeRetireCount
            || lifecycle.reclaimCount != debugLastReportedRuntimeReclaimCount
            || rebuild.requestCount != debugLastReportedRebuildRequestCount
            || rebuild.queuedCount != debugLastReportedRebuildQueuedCount
            || rebuild.blockedPendingDuplicateCount != debugLastReportedRebuildBlockedPendingDuplicateCount
            || rebuild.blockedRecentDuplicateCount != debugLastReportedRebuildBlockedRecentDuplicateCount
            || rebuild.runtimeQueueFullCount != debugLastReportedRebuildRuntimeQueueFullCount
            || rebuild.drainedCommandCount != debugLastReportedRebuildDrainedCommandCount
            || rebuild.matchedRuntimeCommandCount != debugLastReportedRebuildMatchedRuntimeCommandCount
            || rebuild.taskSnapshotFallbackCount != debugLastReportedRebuildTaskSnapshotFallbackCount
            || convRebuild.requestCount != debugLastReportedConvolverRebuildRequestCount
            || convRebuild.deferredAfterLoadCount != debugLastReportedConvolverRebuildDeferredAfterLoadCount
            || convRebuild.scheduledCount != debugLastReportedConvolverRebuildScheduledCount
            || convRebuild.triggeredCount != debugLastReportedConvolverRebuildTriggeredCount
            || shutdownPhaseValue != debugLastReportedShutdownPhase)
        {
            debugLastReportedRuntimePublishCount = lifecycle.publishCount;
            debugLastReportedRuntimeRetireCount = lifecycle.retireCount;
            debugLastReportedRuntimeReclaimCount = lifecycle.reclaimCount;
            debugLastReportedRebuildRequestCount = rebuild.requestCount;
            debugLastReportedRebuildQueuedCount = rebuild.queuedCount;
            debugLastReportedRebuildBlockedPendingDuplicateCount = rebuild.blockedPendingDuplicateCount;
            debugLastReportedRebuildBlockedRecentDuplicateCount = rebuild.blockedRecentDuplicateCount;
            debugLastReportedRebuildRuntimeQueueFullCount = rebuild.runtimeQueueFullCount;
            debugLastReportedRebuildDrainedCommandCount = rebuild.drainedCommandCount;
            debugLastReportedRebuildMatchedRuntimeCommandCount = rebuild.matchedRuntimeCommandCount;
            debugLastReportedRebuildTaskSnapshotFallbackCount = rebuild.taskSnapshotFallbackCount;
            debugLastReportedConvolverRebuildRequestCount = convRebuild.requestCount;
            debugLastReportedConvolverRebuildDeferredAfterLoadCount = convRebuild.deferredAfterLoadCount;
            debugLastReportedConvolverRebuildScheduledCount = convRebuild.scheduledCount;
            debugLastReportedConvolverRebuildTriggeredCount = convRebuild.triggeredCount;
            debugLastReportedShutdownPhase = shutdownPhaseValue;

            diagLog("[VERIFY] tx counters lifecycle(pub/ret/reclaim)="
                + juce::String(static_cast<juce::int64>(lifecycle.publishCount)) + "/"
                + juce::String(static_cast<juce::int64>(lifecycle.retireCount)) + "/"
                + juce::String(static_cast<juce::int64>(lifecycle.reclaimCount))
                + " rebuild(req/queued/blockP/blockR/queueFull/drain/match/fallback)="
                + juce::String(static_cast<juce::int64>(rebuild.requestCount)) + "/"
                + juce::String(static_cast<juce::int64>(rebuild.queuedCount)) + "/"
                + juce::String(static_cast<juce::int64>(rebuild.blockedPendingDuplicateCount)) + "/"
                + juce::String(static_cast<juce::int64>(rebuild.blockedRecentDuplicateCount)) + "/"
                + juce::String(static_cast<juce::int64>(rebuild.runtimeQueueFullCount)) + "/"
                + juce::String(static_cast<juce::int64>(rebuild.drainedCommandCount)) + "/"
                + juce::String(static_cast<juce::int64>(rebuild.matchedRuntimeCommandCount)) + "/"
                + juce::String(static_cast<juce::int64>(rebuild.taskSnapshotFallbackCount))
                + " convDebounce(req/defer/sched/trigger)="
                + juce::String(static_cast<juce::int64>(convRebuild.requestCount)) + "/"
                + juce::String(static_cast<juce::int64>(convRebuild.deferredAfterLoadCount)) + "/"
                + juce::String(static_cast<juce::int64>(convRebuild.scheduledCount)) + "/"
                + juce::String(static_cast<juce::int64>(convRebuild.triggeredCount))
                + " shutdownPhase="
                + juce::String(shutdownPhaseToString(static_cast<ShutdownPhase>(shutdownPhaseValue))));
        }
    }

    // フェイルセーフ: current snapshot が欠落した状態を放置すると
    // EQ変更が演算経路へ乗らないため、Message Thread 側で自己修復する。
    const auto runtimePublishView = getRuntimePublishView();
    auto* currentDspForRuntime = resolveCurrentDSPFromRuntimePublish(runtimePublishView.graph, runtimePublishView.engine);
    auto* fadingDspForRuntime = resolveFadingDSPFromRuntimePublish(runtimePublishView.graph, runtimePublishView.engine);

    // T1: 公開済みRuntimeへのNonRTからの可変更新を避けるため、
    // Timerからのdither内部状態更新は行わない。

    if (!isShutdownInProgress()
        && currentDspForRuntime != nullptr
        && !m_coordinator.isFading()
        && m_coordinator.getCurrent() == nullptr)
    {
        diagLog("[VERIFY] snapshot bootstrap: current was null, requesting worker snapshot refresh");
        if (!enqueueSnapshotCommand())
            diagLog("[VERIFY] snapshot bootstrap: enqueueSnapshotCommand failed");
    }

    {
        const uint32_t observedVersion = convo::consumeAtomic(debugAppliedEqHashVersion, std::memory_order_acquire);
        debugObservedEqHashVersion = observedVersion;

        const uint64_t createdHash = convo::consumeAtomic(debugLastCreatedEqHash, std::memory_order_acquire);
        const uint64_t appliedHash = convo::consumeAtomic(debugLastAppliedEqHash, std::memory_order_acquire);
        const uint64_t createBlockCounter = convo::consumeAtomic(debugLastCreateAudioBlockCounter, std::memory_order_acquire);
        const uint64_t nowBlockCounter = convo::consumeAtomic(m_audioBlockCounter, std::memory_order_acquire);
        const uint64_t processedBlocksSinceCreate = (nowBlockCounter >= createBlockCounter)
            ? (nowBlockCounter - createBlockCounter)
            : 0;
        const int dspReady = (currentDspForRuntime != nullptr) ? 1 : 0;
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

    if (!isShutdownInProgress()
        && hasRebuildReason(RebuildReason::DeferredStructural))
    {
        const int64_t dueTicks = convo::consumeAtomic(deferredStructuralRebuildDueTicks_, std::memory_order_acquire);
        const int64_t nowTicks = juce::Time::getHighResolutionTicks();

        if (dueTicks > 0 && nowTicks >= dueTicks)
        {
            clearRebuildReason(RebuildReason::DeferredStructural);
            convo::publishAtomic(deferredStructuralRebuildDueTicks_, 0, std::memory_order_release);

            if (uiConvolverProcessor.isIRLoaded())
            {
                diagLog("[DIAG] timerCallback: issuing deferred Structural rebuild after prepared IR apply");
                requestRebuild(convo::RebuildKind::Structural);

                ++pendingIRGeneration;
                setIRChangeFlag();

                const LearningCommand cmd {
                    LearningCommand::Type::IRChanged,
                    false,
                    convo::consumeAtomic(pendingLearningMode, std::memory_order_acquire),
                    pendingIRGeneration
                };

                if (!enqueueLearningCommand(cmd))
                {
                    DBG("[AudioEngine] timerCallback: deferred command queue overflow");
                }
            }
        }
    }

    if (!isShutdownInProgress()
        && hasRebuildReason(RebuildReason::DeferredFinalizeAware))
    {
        const int queuedGeneration = convo::consumeAtomic(rebuildGeneration, std::memory_order_acquire);
        const int committedGeneration = convo::consumeAtomic(lastCommittedRebuildGeneration, std::memory_order_acquire);
        const bool outstandingRebuild = queuedGeneration > committedGeneration;
        const bool irLoaded = uiConvolverProcessor.isIRLoaded();
        const bool irFinalized = uiConvolverProcessor.isIRFinalized();
        const bool irLoading = uiConvolverProcessor.isLoadingIR();
        const bool structuralDeferred = hasRebuildReason(RebuildReason::DeferredStructural);
        const bool pendingIrChange = convo::consumeAtomic(m_pendingIRChange, std::memory_order_acquire);

        // IR 遷移が完全に落ち着いてから 1 回だけ再構築を発火する。
        if ((!irLoaded || irFinalized)
            && !irLoading
            && !structuralDeferred
            && !pendingIrChange
            && !outstandingRebuild)
        {
            clearRebuildReason(RebuildReason::DeferredFinalizeAware);

            const double sr = convo::consumeAtomic(currentSampleRate, std::memory_order_acquire);
            if (!m_isRestoringState && sr > 0.0)
            {
                diagLog("[DIAG] timerCallback: issuing deferred finalize-aware rebuild");
                requestRebuild(sr, convo::consumeAtomic(maxSamplesPerBlock, std::memory_order_acquire));
            }
        }
    }

    processLearningCommands();
    processDeferredLearningActions();

    const bool hasFading = (fadingDspForRuntime != nullptr);
    const bool hasPendingCrossfade = runtimeCrossfadePending(runtimePublishView.engine, runtimePublishView.graph);

    // Grace period に基づく安全なリリース遅延を実行する。
    processDeferredReleases();

    const bool fadeCompleted = m_coordinator.tryCompleteFade();
    if (fadeCompleted)
    {
        if (auto* done = sanitizeRawPtr(exchangeFadingOutDSP(nullptr)))
            retireDSP(done);
        publishAtomic(dspCrossfadePending, false);
        refreshCrossfadePreparedSnapshotFromAtomics();
        sendChangeMessage();
    }

    if (!m_coordinator.isFading())
    {
        if (auto* done = sanitizeRawPtr(exchangeFadingOutDSP(nullptr)))
            retireDSP(done);
    }

    if (!isShutdownInProgress()
        && !hasFading
        && !hasPendingCrossfade)
    {
        bool hasDeferredCommits = false;
        {
            std::lock_guard<std::mutex> lock(deferredCommitMutex);
            hasDeferredCommits = !deferredCommitQueue.empty();
        }

        if (hasDeferredCommits)
            triggerAsyncUpdate();
    }

    // 内部プロセッサのクリーンアップを実行する。
    if (auto* dsp = currentDspForRuntime)
    {
        dsp->eqState->cleanupForRuntime();
        dsp->convolverState->cleanupForRuntime();

        const bool activeFixed4Tap = (dsp->noiseShaperType == NoiseShaperType::Fixed4Tap);
        const bool activeFixed15Tap = (dsp->noiseShaperType == NoiseShaperType::Fixed15Tap);
        const bool activeDitherEnabled = (dsp->ditherBitDepth > 0);

        if (activeFixed4Tap && activeDitherEnabled)
        {
            dsp->fixedNoiseShaper.setDiagnosticsWindowSamples(
                static_cast<uint32_t>(convo::consumeAtomic(fixedNoiseWindowSamples, std::memory_order_acquire)));

            const uint32 now = juce::Time::getMillisecondCounter();
            const uint32 intervalMs = static_cast<uint32>(
                std::max(250, convo::consumeAtomic(fixedNoiseLogIntervalMs, std::memory_order_acquire)));
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
                        + ", targetWindow=" + juce::String(convo::consumeAtomic(fixedNoiseWindowSamples, std::memory_order_acquire))
                        + ")");
                }
            }
        }
        else if (activeFixed15Tap && activeDitherEnabled)
        {
            dsp->fixed15TapNoiseShaper.setDiagnosticsWindowSamples(
                static_cast<uint32_t>(convo::consumeAtomic(fixedNoiseWindowSamples, std::memory_order_acquire)));

            const uint32 now = juce::Time::getMillisecondCounter();
            const uint32 intervalMs = static_cast<uint32>(
                std::max(250, convo::consumeAtomic(fixedNoiseLogIntervalMs, std::memory_order_acquire)));
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
                        + ", targetWindow=" + juce::String(convo::consumeAtomic(fixedNoiseWindowSamples, std::memory_order_acquire))
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
