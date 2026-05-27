#include <JuceHeader.h>
#include "AudioEngine.h"

namespace {
void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}
}

void AudioEngine::timerCallback()
{
    const auto runtimeReadView = readControlRuntimeView();
    const auto& runtimePublishView = runtimeReadView.runtimePublish;
    const auto* runtimeGraph = getRuntimeGraph(runtimeReadView);
    const auto* currentSnapshot = getRuntimeSnapshot(runtimeReadView);

    emitEvidenceTickNonRt(false);

    {
        const auto ts = runtimePublishView.transition;
        const int active = ts.active ? 1 : 0;
        const int policy = static_cast<int>(ts.policy);
        const uint64_t currentPtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ts.current));
        const uint64_t nextPtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ts.next));
        const double fadeSec = ts.fadeTimeSec;
        const int latencyDelta = ts.latencyDeltaSamples;

        if (active != rtAuxMutable_.debugLastReportedTransitionActive
            || policy != rtAuxMutable_.debugLastReportedTransitionPolicy
            || currentPtr != rtAuxMutable_.debugLastReportedTransitionCurrentPtr
            || nextPtr != rtAuxMutable_.debugLastReportedTransitionNextPtr
            || absNoLibm(fadeSec - rtAuxMutable_.debugLastReportedTransitionFadeSec) > 1e-9
            || latencyDelta != rtAuxMutable_.debugLastReportedTransitionLatencyDeltaSamples)
        {
            rtAuxMutable_.debugLastReportedTransitionActive = active;
            rtAuxMutable_.debugLastReportedTransitionPolicy = policy;
            rtAuxMutable_.debugLastReportedTransitionCurrentPtr = currentPtr;
            rtAuxMutable_.debugLastReportedTransitionNextPtr = nextPtr;
            rtAuxMutable_.debugLastReportedTransitionLatencyDeltaSamples = latencyDelta;

#if JUCE_DEBUG
            diagLog("[VERIFY] transition state active=" + juce::String(active)
                + " policy=" + juce::String(policy)
                + " current=0x" + juce::String::toHexString(static_cast<juce::int64>(currentPtr))
                + " next=0x" + juce::String::toHexString(static_cast<juce::int64>(nextPtr))
                + " fadeSec=" + juce::String(fadeSec, 6)
                + " latencyDelta=" + juce::String(latencyDelta));
#endif
        }

    }

    {
        const uint64_t revision = (runtimeGraph != nullptr) ? runtimeGraph->generation : 0;
        const uint64_t currentUuid = (runtimeGraph != nullptr) ? runtimeGraph->runtimeUuid : 0;
        const uint64_t fadingUuid = (runtimeGraph != nullptr) ? runtimeGraph->fadingRuntimeUuid : 0;
        const uint64_t transitionCurrentUuid = (runtimeGraph != nullptr) ? runtimeGraph->transitionCurrentRuntimeUuid : 0;
        const uint64_t transitionNextUuid = (runtimeGraph != nullptr) ? runtimeGraph->transitionNextRuntimeUuid : 0;

        if (revision != rtAuxMutable_.debugLastReportedRuntimeSnapshotRevision
            || currentUuid != rtAuxMutable_.debugLastReportedRuntimePublishCurrentUuid
            || fadingUuid != rtAuxMutable_.debugLastReportedRuntimePublishFadingUuid
            || transitionCurrentUuid != rtAuxMutable_.debugLastReportedRuntimePublishTransitionCurrentUuid
            || transitionNextUuid != rtAuxMutable_.debugLastReportedRuntimePublishTransitionNextUuid)
        {
            rtAuxMutable_.debugLastReportedRuntimeSnapshotRevision = revision;
            rtAuxMutable_.debugLastReportedRuntimePublishCurrentUuid = currentUuid;
            rtAuxMutable_.debugLastReportedRuntimePublishFadingUuid = fadingUuid;
            rtAuxMutable_.debugLastReportedRuntimePublishTransitionCurrentUuid = transitionCurrentUuid;
            rtAuxMutable_.debugLastReportedRuntimePublishTransitionNextUuid = transitionNextUuid;

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
        const auto eqCacheMiss = getEqCacheMissDiagnostics();
        const auto convRebuild = uiConvolverProcessor.getRebuildAutomationDiagnostics();
        const int shutdownPhaseValue = static_cast<int>(convo::consumeAtomic(shutdownPhase, std::memory_order_acquire));

        if (lifecycle.publishCount != rtAuxMutable_.debugLastReportedRuntimePublishCount
            || lifecycle.retireCount != rtAuxMutable_.debugLastReportedRuntimeRetireCount
            || lifecycle.reclaimCount != rtAuxMutable_.debugLastReportedRuntimeReclaimCount
            || rebuild.requestCount != rtAuxMutable_.debugLastReportedRebuildRequestCount
            || rebuild.queuedCount != rtAuxMutable_.debugLastReportedRebuildQueuedCount
            || rebuild.blockedPendingDuplicateCount != rtAuxMutable_.debugLastReportedRebuildBlockedPendingDuplicateCount
            || rebuild.blockedRecentDuplicateCount != rtAuxMutable_.debugLastReportedRebuildBlockedRecentDuplicateCount
            || rebuild.runtimeQueueFullCount != rtAuxMutable_.debugLastReportedRebuildRuntimeQueueFullCount
            || rebuild.drainedCommandCount != rtAuxMutable_.debugLastReportedRebuildDrainedCommandCount
            || rebuild.matchedRuntimeCommandCount != rtAuxMutable_.debugLastReportedRebuildMatchedRuntimeCommandCount
            || rebuild.taskSnapshotFallbackCount != rtAuxMutable_.debugLastReportedRebuildTaskSnapshotFallbackCount
            || eqCacheMiss.snapshotCreateMissCount != rtAuxMutable_.debugLastReportedEqCacheSnapshotCreateMissCount
            || eqCacheMiss.runtimeLookupMissCount != rtAuxMutable_.debugLastReportedEqCacheRuntimeLookupMissCount
            || convRebuild.requestCount != rtAuxMutable_.debugLastReportedConvolverRebuildRequestCount
            || convRebuild.deferredAfterLoadCount != rtAuxMutable_.debugLastReportedConvolverRebuildDeferredAfterLoadCount
            || convRebuild.scheduledCount != rtAuxMutable_.debugLastReportedConvolverRebuildScheduledCount
            || convRebuild.triggeredCount != rtAuxMutable_.debugLastReportedConvolverRebuildTriggeredCount
            || shutdownPhaseValue != rtAuxMutable_.debugLastReportedShutdownPhase)
        {
            rtAuxMutable_.debugLastReportedRuntimePublishCount = lifecycle.publishCount;
            rtAuxMutable_.debugLastReportedRuntimeRetireCount = lifecycle.retireCount;
            rtAuxMutable_.debugLastReportedRuntimeReclaimCount = lifecycle.reclaimCount;
            rtAuxMutable_.debugLastReportedRebuildRequestCount = rebuild.requestCount;
            rtAuxMutable_.debugLastReportedRebuildQueuedCount = rebuild.queuedCount;
            rtAuxMutable_.debugLastReportedRebuildBlockedPendingDuplicateCount = rebuild.blockedPendingDuplicateCount;
            rtAuxMutable_.debugLastReportedRebuildBlockedRecentDuplicateCount = rebuild.blockedRecentDuplicateCount;
            rtAuxMutable_.debugLastReportedRebuildRuntimeQueueFullCount = rebuild.runtimeQueueFullCount;
            rtAuxMutable_.debugLastReportedRebuildDrainedCommandCount = rebuild.drainedCommandCount;
            rtAuxMutable_.debugLastReportedRebuildMatchedRuntimeCommandCount = rebuild.matchedRuntimeCommandCount;
            rtAuxMutable_.debugLastReportedRebuildTaskSnapshotFallbackCount = rebuild.taskSnapshotFallbackCount;
            rtAuxMutable_.debugLastReportedEqCacheSnapshotCreateMissCount = eqCacheMiss.snapshotCreateMissCount;
            rtAuxMutable_.debugLastReportedEqCacheRuntimeLookupMissCount = eqCacheMiss.runtimeLookupMissCount;
            rtAuxMutable_.debugLastReportedConvolverRebuildRequestCount = convRebuild.requestCount;
            rtAuxMutable_.debugLastReportedConvolverRebuildDeferredAfterLoadCount = convRebuild.deferredAfterLoadCount;
            rtAuxMutable_.debugLastReportedConvolverRebuildScheduledCount = convRebuild.scheduledCount;
            rtAuxMutable_.debugLastReportedConvolverRebuildTriggeredCount = convRebuild.triggeredCount;
            rtAuxMutable_.debugLastReportedShutdownPhase = shutdownPhaseValue;

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
                + " eqCacheMiss(create/lookup)="
                + juce::String(static_cast<juce::int64>(eqCacheMiss.snapshotCreateMissCount)) + "/"
                + juce::String(static_cast<juce::int64>(eqCacheMiss.runtimeLookupMissCount))
                + " convDebounce(req/defer/sched/trigger)="
                + juce::String(static_cast<juce::int64>(convRebuild.requestCount)) + "/"
                + juce::String(static_cast<juce::int64>(convRebuild.deferredAfterLoadCount)) + "/"
                + juce::String(static_cast<juce::int64>(convRebuild.scheduledCount)) + "/"
                + juce::String(static_cast<juce::int64>(convRebuild.triggeredCount))
                + " shutdownPhase="
                + juce::String(shutdownPhaseToString(static_cast<ShutdownPhase>(shutdownPhaseValue))));
        }
    }

    // 回復経路: current snapshot が欠落した状態を放置すると
    // EQ変更が演算経路へ乗らないため、Message Thread 側で自己修復する。
    auto* currentDspForRuntime = (runtimeGraph != nullptr)
        ? static_cast<DSPCore*>(runtimeGraph->activeNode)
        : nullptr;
    auto* fadingDspForRuntime = resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeGraph);

    // T1: 公開済みRuntimeへのNonRTからの可変更新を避けるため、
    // Timerからのdither内部状態更新は行わない。

    {
        const uint64_t currentEqHash = (currentSnapshot != nullptr) ? currentSnapshot->eqCoeffHash : 0;
        const bool hasEqHash = (currentEqHash != 0);
        const bool lookupMiss = hasEqHash && !eqCacheManager.containsNonRt(currentEqHash);

        if (lookupMiss)
        {
            const bool missEdge = (!rtAuxMutable_.debugEqCacheLookupMissLatched)
                || (rtAuxMutable_.debugEqCacheLookupMissLatchedHash != currentEqHash);

            if (missEdge)
            {
                convo::fetchAddAtomic(rtAuxMutable_.eqCacheRuntimeLookupMissCountNonRt,
                                      static_cast<std::uint64_t>(1),
                                      std::memory_order_acq_rel);
                rtAuxMutable_.debugEqCacheLookupMissLatched = true;
                rtAuxMutable_.debugEqCacheLookupMissLatchedHash = currentEqHash;
            }
        }
        else
        {
            rtAuxMutable_.debugEqCacheLookupMissLatched = false;
            rtAuxMutable_.debugEqCacheLookupMissLatchedHash = 0;
        }
    }

    if (!isShutdownInProgress()
        && currentDspForRuntime != nullptr
        && !m_coordinator.isFading()
        && currentSnapshot == nullptr)
    {
        diagLog("[VERIFY] snapshot bootstrap: current was null, requesting worker snapshot refresh");
        if (!enqueueSnapshotCommand())
            diagLog("[VERIFY] snapshot bootstrap: enqueueSnapshotCommand failed");
    }

    {
        const uint64_t createdHash = convo::consumeAtomic(rtAuxMutable_.debugLastCreatedEqHash, std::memory_order_acquire);
        const int dspReady = (currentDspForRuntime != nullptr) ? 1 : 0;
        const int coordIsFading = m_coordinator.isFading() ? 1 : 0;
        const int updateFadeReturned = coordIsFading;
        const int fromNull = (currentSnapshot == nullptr) ? 1 : 0;
        const int toNull = -1;

        if (createdHash != rtAuxMutable_.debugLastReportedCreatedEqHash ||
            dspReady != rtAuxMutable_.debugLastReportedDspReady)
        {
            rtAuxMutable_.debugLastReportedCreatedEqHash = createdHash;
            rtAuxMutable_.debugLastReportedDspReady = dspReady;
            diagLog("[VERIFY] EQ reflection createdHash=0x"
                + juce::String::toHexString(static_cast<juce::int64>(createdHash))
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
        const uint64_t intentId = nextRebuildTelemetryIntentId();
        const int64_t dueTicks = convo::consumeAtomic(deferredStructuralRebuildDueTicks_, std::memory_order_acquire);
        const int64_t nowTicks = juce::Time::getHighResolutionTicks();

        if (dueTicks > 0 && nowTicks >= dueTicks)
        {
            clearRebuildReason(RebuildReason::DeferredStructural);
            convo::publishAtomic(deferredStructuralRebuildDueTicks_, 0, std::memory_order_release);

            if (uiConvolverProcessor.isIRLoaded())
            {
                diagLog("[DIAG] timerCallback: issuing deferred Structural rebuild after prepared IR apply");
                emitRebuildTelemetry(RebuildTelemetryEvent::Deferred,
                                     intentId,
                                     RebuildTelemetryReason::DeferredStructuralDue,
                                     RebuildTelemetryDecision::Released,
                                     0,
                                     0,
                                     RebuildTelemetryClass::Structural,
                                     RebuildTelemetryPolicy::NA,
                                     "deferred_structural");
                submitRebuildIntent(convo::RebuildKind::Structural,
                                    RebuildTelemetryReason::DeferredStructuralRebuildRequested,
                                    RebuildTelemetryClass::Structural,
                                    RebuildTelemetryPolicy::Replaceable);

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

    if (isShutdownInProgress())
        emitEvidenceTickNonRt(true);

    if (!isShutdownInProgress()
        && hasRebuildReason(RebuildReason::DeferredFinalizeAware))
    {
        const uint64_t intentId = nextRebuildTelemetryIntentId();
        static constexpr int kFinalizeDeferMaxDurationMs = 2000;
        const int64_t nowTicks = juce::Time::getHighResolutionTicks();
        const int64_t ticksPerSecond = juce::Time::getHighResolutionTicksPerSecond();
        const int64_t maxDeferTicks = (ticksPerSecond * kFinalizeDeferMaxDurationMs) / 1000;

        int64_t firstSeenTicks = convo::consumeAtomic(deferredFinalizeFirstSeenTicks_, std::memory_order_acquire);
        if (firstSeenTicks <= 0)
        {
            convo::publishAtomic(deferredFinalizeFirstSeenTicks_, nowTicks, std::memory_order_release);
            firstSeenTicks = nowTicks;
        }

        const int64_t elapsedTicks = (nowTicks >= firstSeenTicks) ? (nowTicks - firstSeenTicks) : 0;
        const bool timedOut = (maxDeferTicks > 0) && (elapsedTicks >= maxDeferTicks);
        const double elapsedMs = (ticksPerSecond > 0)
            ? (static_cast<double>(elapsedTicks) * 1000.0 / static_cast<double>(ticksPerSecond))
            : 0.0;

        const int queuedGeneration = convo::consumeAtomic(rebuildGeneration, std::memory_order_acquire);
        const int committedGeneration = convo::consumeAtomic(lastCommittedRebuildGeneration, std::memory_order_acquire);
        const bool outstandingRebuild = queuedGeneration > committedGeneration;
        const bool irLoaded = uiConvolverProcessor.isIRLoaded();
        const bool irFinalized = uiConvolverProcessor.isIRFinalized();
        const bool irLoading = uiConvolverProcessor.isLoadingIR();
        const bool structuralDeferred = hasRebuildReason(RebuildReason::DeferredStructural);
        const bool pendingIrChange = convo::consumeAtomic(m_pendingIRChange, std::memory_order_acquire);

        // IR 遷移が完全に落ち着いてから 1 回だけ再構築を発火する。
        const bool finalizeReady = (!irLoaded || irFinalized)
            && !irLoading
            && !structuralDeferred
            && !pendingIrChange
            && !outstandingRebuild;

        if (finalizeReady || timedOut)
        {
            clearRebuildReason(RebuildReason::DeferredFinalizeAware);
            convo::publishAtomic(deferredFinalizeFirstSeenTicks_, 0, std::memory_order_release);

            const double sr = convo::consumeAtomic(currentSampleRate, std::memory_order_acquire);
            if (!m_isRestoringState && sr > 0.0)
            {
                if (timedOut && !finalizeReady)
                {
                    diagLog("[DIAG] timerCallback: finalize defer timeout reached, forcing rebuild dispatch");
                    emitRebuildTelemetry(RebuildTelemetryEvent::ForcedDispatch,
                                         intentId,
                                         RebuildTelemetryReason::DeferredFinalizeRebuildRequested,
                                         RebuildTelemetryDecision::Dispatched,
                                         0,
                                         0,
                                         RebuildTelemetryClass::FinalizeAware,
                                         RebuildTelemetryPolicy::MustExecute,
                                         "deferred_finalize_timeout",
                                         elapsedMs);
                }
                else
                {
                    diagLog("[DIAG] timerCallback: issuing deferred finalize-aware rebuild");
                    emitRebuildTelemetry(RebuildTelemetryEvent::Deferred,
                                         intentId,
                                         RebuildTelemetryReason::DeferredFinalizeReady,
                                         RebuildTelemetryDecision::Released,
                                         0,
                                         0,
                                         RebuildTelemetryClass::FinalizeAware,
                                         RebuildTelemetryPolicy::NA,
                                         "deferred_finalize_aware");
                }

                submitRebuildIntent(convo::RebuildKind::Structural,
                                    RebuildTelemetryReason::DeferredFinalizeRebuildRequested,
                                    RebuildTelemetryClass::FinalizeAware,
                                    RebuildTelemetryPolicy::MustExecute);
            }
        }
    }

    processLearningCommands();
    processDeferredLearningActions();

    const bool hasFading = (fadingDspForRuntime != nullptr);
    const bool hasPendingCrossfade = (runtimeGraph != nullptr) ? runtimeGraph->dspCrossfadePending : false;

    // Grace period に基づく安全なリリース遅延を実行する。
    processDeferredReleases();

    const bool fadeCompleted = m_coordinator.tryCompleteFade();
    if (fadeCompleted)
    {
        const auto activeCrossfadeId = convo::consumeAtomic(activeCrossfadeId_, std::memory_order_acquire);
        if (activeCrossfadeId != 0u)
        {
            dspHandleRuntime_.endCrossfade(activeCrossfadeId);
            crossfadeAuthorityRuntime_.unregisterCrossfade(activeCrossfadeId);
            convo::publishAtomic(activeCrossfadeId_, static_cast<convo::isr::CrossfadeId>(0u), std::memory_order_release);
        }

        auto* const doneRaw1 = exchangeFadingRuntimeDSP(nullptr);
        if (auto* done = (reinterpret_cast<uintptr_t>(doneRaw1) == (~static_cast<uintptr_t>(0))) ? nullptr : doneRaw1)
            retireDSP(done);
        publishAtomic(dspCrossfadePending, false, std::memory_order_release);
        publishAtomic(firstIrDryCrossfadePending, false, std::memory_order_release);
        publishAtomic(dspCrossfadeUseDryAsOld, false, std::memory_order_release);
        publishAtomic(dspCrossfadeStartDelayBlocks, 0, std::memory_order_release);
        publishAtomic(dspCrossfadeDryHoldSamples, 0, std::memory_order_release);
        refreshCrossfadePreparedSnapshotFromAtomics();

        // Phase4: フェード完了時の RuntimeGraph を idle 状態へ同期する。
        // これにより AudioThread は atomic fallback ではなく publish world だけで
        // crossfade 状態を正しく観測できる。
        auto* currentAfterFade = (runtimeGraph != nullptr)
            ? static_cast<DSPCore*>(runtimeGraph->activeNode)
            : nullptr;
        if (currentAfterFade != nullptr)
        {
            makeRuntimePublicationCoordinator()
                .publishState(currentAfterFade,
                              nullptr,
                              convo::TransitionPolicy::SmoothOnly,
                              0.0,
                              false);
        }

        sendChangeMessage();
    }

    if (!m_coordinator.isFading())
    {
        auto* const doneRaw2 = exchangeFadingRuntimeDSP(nullptr);
        if (auto* done = (reinterpret_cast<uintptr_t>(doneRaw2) == (~static_cast<uintptr_t>(0))) ? nullptr : doneRaw2)
            retireDSP(done);
    }

    if (!isShutdownInProgress()
        && !hasFading
        && !hasPendingCrossfade)
    {
        const bool hasDeferredCommits = hasPendingPublicationIntents();

        if (hasDeferredCommits)
            triggerAsyncUpdate();
    }

    // 内部プロセッサのクリーンアップを実行する。
    if (auto* dsp = currentDspForRuntime)
    {
        dsp->eqState->cleanupForRuntime();
        dsp->convolverState->cleanupForRuntime();

        // M2: PsychoacousticDither の RNG リングは Audio Thread 外で補充する。
        // timerCallback は Message Thread で実行されるため、RT 制約に抵触しない。
        const bool activePsychoacoustic = (dsp->noiseShaperType == NoiseShaperType::Psychoacoustic);
        if (activePsychoacoustic && dsp->ditherBitDepth > 0)
            dsp->dither.refillRandomRingNonRt();

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
            if ((now - rtAuxMutable_.fixedNoiseLastLogMs) >= intervalMs)
            {
                rtAuxMutable_.fixedNoiseLastLogMs = now;
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
            if ((now - rtAuxMutable_.fixedNoiseLastLogMs) >= intervalMs)
            {
                rtAuxMutable_.fixedNoiseLastLogMs = now;
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
