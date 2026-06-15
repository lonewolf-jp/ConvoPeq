#include <JuceHeader.h>
#include "AudioEngine.h"
#include "core/RuntimeReaderContext.h"
#include "RuntimeBuilder.h"
#include "RuntimePublicationOrchestrator.h"
#include "DSPLifetimeManager.h"
#include "../NoiseShaperLearner.h"  // ★ Work39: Restore Learner Rollback 用

namespace {
void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}
}

void AudioEngine::timerCallback()
{
    const convo::RuntimeReaderContext messageCtx{ messageThreadRcuReader, convo::ObserveChannel::Message };
    const auto runtimeReadHandle = makeRuntimeReadHandle(messageCtx);
    const auto* runtimeWorld = getRuntimeWorldFromReadHandle(runtimeReadHandle);
    const bool transitionActive = hasFadingRuntimeInWorld(runtimeReadHandle);
    const auto* currentSnapshot = getRuntimeSnapshotFromReadHandle(runtimeReadHandle);

    emitEvidenceTickNonRt(false);

    {
        const int active = transitionActive ? 1 : 0;
        const int policy = (runtimeWorld != nullptr)
            ? runtimeWorld->execution.transitionPolicy
            : static_cast<int>(convo::TransitionPolicy::SmoothOnly);
        auto* currentRuntime = resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle);
        auto* fadingRuntime = resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle);
        const uint64_t currentPtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(currentRuntime));
        const uint64_t nextPtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(fadingRuntime));
        const double fadeSec = (runtimeWorld != nullptr)
            ? runtimeWorld->overlap.fadeTimeSec
            : 0.0;
        const int latencyDelta = (runtimeWorld != nullptr)
            ? runtimeWorld->execution.latencyCompensationSamples
            : 0;

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
        auto* currentRuntime = resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle);
        auto* fadingRuntime = resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle);
        const uint64_t revision = (runtimeWorld != nullptr) ? runtimeWorld->generation : static_cast<std::uint64_t>(0);
        const uint64_t currentUuid = (currentRuntime != nullptr) ? currentRuntime->runtimeUuid : 0;
        const uint64_t fadingUuid = (fadingRuntime != nullptr) ? fadingRuntime->runtimeUuid : 0;
        const uint64_t transitionCurrentUuid = currentUuid;
        const uint64_t transitionNextUuid = (fadingRuntime != nullptr) ? fadingRuntime->runtimeUuid : 0;

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
    auto* currentDspForRuntime = resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle);
    auto* fadingDspForRuntime = resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle);

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
        diagLog("[VERIFY] snapshot bootstrap: current was null, creating snapshot from current state");
        const auto snapshotGeneration = (runtimeWorld != nullptr)
            ? runtimeWorld->generation
            : convo::consumeAtomic(lastCommittedRuntimeGeneration_, std::memory_order_acquire);
        createSnapshotFromCurrentState(snapshotGeneration);
        diagLog("[VERIFY] snapshot bootstrap: createSnapshotFromCurrentState done");
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

        const int queuedGeneration = convo::consumeAtomic(rebuildRequestGeneration, std::memory_order_acquire);
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
    const bool hasPendingCrossfade = hasPendingCrossfadeInWorld(runtimeReadHandle)
        || shouldUseDryAsOldInWorld(runtimeReadHandle);

    // Grace period に基づく安全なリリース遅延を実行する。
    processDeferredReleases();

    const bool fadeCompleted = m_coordinator.tryCompleteFade();
    if (fadeCompleted)
    {
        // ★ P1-C: 完了した crossfade の ID を SPSC 経由で消費
        //   notifyFadeComplete は AudioThread 側（advanceFade 内）で呼ばれる想定。
        //   現状は Timer 主導の tryCompleteFade のため、ここで SPSC に投入し即消費する。
        const auto completedId = convo::consumeAtomic(activeCrossfadeId_, std::memory_order_acquire);
        if (completedId != 0u)
        {
            crossfadeRuntime_.notifyFadeComplete(completedId);
            convo::isr::CompletedFadeEvent ev;
            if (crossfadeRuntime_.consumeCompletedFade(ev))
            {
                // ★ SPSC を経由することで将来的な AudioThread 主導への移行が容易
                dspHandleRuntime_.endCrossfade(ev.id);
                crossfadeAuthorityRuntime_.unregisterCrossfade(ev.id);
            }
            convo::publishAtomic(activeCrossfadeId_, static_cast<convo::isr::CrossfadeId>(0u), std::memory_order_release);
        }

        auto* const doneRaw1 = exchangeFadingRuntimeDSP(nullptr);
        if (auto* done = (reinterpret_cast<uintptr_t>(doneRaw1) == (~static_cast<uintptr_t>(0))) ? nullptr : doneRaw1)
        {
            DSPLifetimeManager lifetimeMgr(*this);
            lifetimeMgr.retire(done);
        }
        crossfadeRuntime_.complete();
        crossfadeRuntime_.setStartDelayBlocks(0);
        crossfadeRuntime_.setDryHoldSamples(0);
        refreshCrossfadePreparedSnapshotFromAtomics();

        // Phase4: フェード完了時の RuntimeGraph を idle 状態へ同期する。
        // これにより AudioThread は atomic fallback ではなく publish world だけで
        // crossfade 状態を正しく観測できる。
        auto* currentAfterFade = resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle);
        if (currentAfterFade != nullptr)
        {
            // Migrated to publishWorld() with pre-built RuntimePublishWorld (Sprint-2 P1-A)
            auto coordinator = makeRuntimePublicationCoordinator();
            auto worldBuilder = convo::RuntimeBuilder(*this);
            worldBuilder.setHealthStateRef(getHealthStateRef());
            auto worldOwner = worldBuilder.buildRuntimePublishWorld(currentAfterFade,
                                                                     nullptr,
                                                                     convo::TransitionPolicy::SmoothOnly,
                                                                     0.0,
                                                                     false);
            coordinator.publishWorld(std::move(worldOwner));
        }

        sendChangeMessage();
    }

    if (!m_coordinator.isFading())
    {
        auto* const doneRaw2 = exchangeFadingRuntimeDSP(nullptr);
        if (auto* done = (reinterpret_cast<uintptr_t>(doneRaw2) == (~static_cast<uintptr_t>(0))) ? nullptr : doneRaw2)
        {
            DSPLifetimeManager lifetimeMgr(*this);
            lifetimeMgr.retire(done);
        }
    }

    if (!isShutdownInProgress()
        && !hasFading
        && !hasPendingCrossfade)
    {
        // [PR-3] Deferred commits via Orchestrator
        if (runtimeOrchestrator_ != nullptr && runtimeOrchestrator_->hasDeferredRequest())
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

    // ★ P1-8: RuntimeHealthMonitor tick（shutdown 中はスキップ）
    if (!isShutdownInProgress()) {
        m_healthMonitor.tick();
    }

    // UI用プロセッサのクリーンアップ
    uiEqEditor.cleanup();
    uiConvolverProcessor.cleanup();
}

// ★ P1-8: HealthMonitor コールバック実装
void AudioEngine::onHealthEvent(const convo::HealthEvent& event) noexcept
{
    diagLog("[HEALTH] eventCode=" + juce::String(static_cast<int>(event.eventCode))
        + " severity=" + juce::String(static_cast<int>(event.severity))
        + " value=" + juce::String(static_cast<juce::int64>(event.value)));

    // ★ A-1: Reader Exhaustion → Admission 強制停止 + 診断ダンプ
    if (event.eventCode == convo::EVENT_READER_SLOT_USAGE
        && event.severity == convo::HealthEvent::Severity::Error)
    {
        diagLog("[HEALTH] Reader slot exhaustion detected, forcing admission stop"
            + juce::String(" slot=") + juce::String(static_cast<int>(event.slot))
            + juce::String(" value=") + juce::String(static_cast<juce::int64>(event.value))
            + juce::String(" readerIndex=") + juce::String(static_cast<int>(event.readerIndex))
            + juce::String(" residencyUs=") + juce::String(static_cast<juce::int64>(event.residencyTimeUs)));

        // Admission 強制停止（HealthState Critical で既に停止するが、念のため直接設定）
        convo::publishAtomic(retirePressureAdmissionStrict_, true, std::memory_order_release);
        // [work37 Phase 9.41] 抑制開始時刻を記録
        if (convo::consumeAtomic(suppressionStartUs_, std::memory_order_acquire) == 0)
            convo::publishAtomic(suppressionStartUs_, convo::getCurrentTimeUs(), std::memory_order_release);

        // 強制診断ダンプ
        emitEvidenceTickNonRt(true);
        worldLifecycleAudit_.tryDumpPeriodic();
        return;
    }

    // ★ A-1: Publication Stall → 停滞中の publish を強制ドレイン
    if (event.eventCode == convo::EVENT_PUBLICATION_STALL
        && event.severity == convo::HealthEvent::Severity::Error)
    {
        diagLog("[HEALTH] Publication stall detected, draining deferred publish");

        if (runtimeOrchestrator_) {
            runtimeOrchestrator_->clearDeferredForShutdown();
        }

        // 強制診断ダンプ
        emitEvidenceTickNonRt(true);
        return;
    }

    // ★ Work38: Retire Stall / Retire Age Critical → 即時遮断のみ（回復は PolicyEngine に委譲）
    if ((event.eventCode == convo::EVENT_RETIRE_STALL
         || event.eventCode == convo::EVENT_RETIRE_AGE_CRITICAL)
        && event.severity == convo::HealthEvent::Severity::Error)
    {
        diagLog("[HEALTH] Retire stall detected, throttling rebuild");

        // retirePressureAdmissionStrict_ を即時設定（PolicyEngine 評価より先に遮断）
        convo::publishAtomic(retirePressureAdmissionStrict_, true, std::memory_order_release);
        // [work37 Phase 9.41] 抑制開始時刻を記録
        if (convo::consumeAtomic(suppressionStartUs_, std::memory_order_acquire) == 0)
            convo::publishAtomic(suppressionStartUs_, convo::getCurrentTimeUs(), std::memory_order_release);

        // ★ tryReclaimResources + emitEvidenceTickNonRt は削除
        //    PolicyEngine の evaluateAggregate → Recover Action に委譲
        return;
    }

    // ★ Practical-5: Crossfade Timeout 回復処理
    if (event.eventCode == convo::EVENT_CROSSFADE_TIMEOUT)
    {
        diagLog("[HEALTH] Crossfade timeout detected, initiating recovery");

        // 1. 滞留中の fading DSP を強制退役
        auto* doneRaw = exchangeFadingRuntimeDSP(nullptr);
        if (doneRaw != nullptr
            && reinterpret_cast<uintptr_t>(doneRaw) != (~static_cast<uintptr_t>(0)))
        {
            DSPLifetimeManager lifetime(*this);
            lifetime.retire(doneRaw);
        }

        // 2. アクティブな crossfade ID を取得して unregister
        const auto activeId = convo::consumeAtomic(activeCrossfadeId_, std::memory_order_acquire);
        if (activeId != 0u)
        {
            crossfadeAuthorityRuntime_.unregisterCrossfade(activeId);
            convo::publishAtomic(activeCrossfadeId_, uint64_t{0}, std::memory_order_release);
        }

        // 3. CrossfadeRuntime を complete 状態に戻す（pending=false）
        crossfadeRuntime_.complete();

        // ★ A-4: publish 前準備 — publishIdleWorldOnly は前準備を含まない
        crossfadeRuntime_.setDryHoldSamples(0);
        refreshCrossfadePreparedSnapshotFromAtomics();

        // ★ A-4: Idle world publish — AudioThread が正しく idle 状態を観測できるよう発行
        {
            const convo::RuntimeReaderContext messageCtx{
                messageThreadRcuReader, convo::ObserveChannel::Message };
            const auto runtimeReadHandle = makeRuntimeReadHandle(messageCtx);
            auto* currentAfterFade =
                resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle);
            (void)publishIdleWorldOnly(currentAfterFade,
                convo::TransitionPolicy::HardReset);
        }

        diagLog("[HEALTH] Crossfade timeout recovery completed");
    }

    // ★ A-3: EVENT_READER_STUCK — Evidence 出力のみ（Shutdown Authority は collectDrainAudit が担当）
    if (event.eventCode == convo::EVENT_READER_STUCK)
    {
        diagLog("[HEALTH] Reader stuck detected"
            + juce::String(" readerIndex=") + juce::String(static_cast<int>(event.readerIndex))
            + juce::String(" residencyUs=") + juce::String(static_cast<juce::int64>(event.residencyTimeUs))
            + juce::String(" pendingRetire=") + juce::String(static_cast<juce::int64>(event.value)));

        // 強制診断ダンプ
        emitEvidenceTickNonRt(true);
    }
}

// [work37 Phase 4.1/9.1] RecoveryAction 実行 — PolicyEngine からの Action を実行
void AudioEngine::executeRecoveryAction(convo::RecoveryAction action) noexcept
{
    switch (action) {
        case convo::RecoveryAction::Throttle:
            convo::publishAtomic(retirePressureAdmissionStrict_, true,
                                 std::memory_order_release);
            // [work37 Phase 9.41] 抑制開始時刻を記録
            if (convo::consumeAtomic(suppressionStartUs_, std::memory_order_acquire) == 0)
                convo::publishAtomic(suppressionStartUs_, convo::getCurrentTimeUs(),
                                     std::memory_order_release);
            break;

        case convo::RecoveryAction::Recover:
            // [work37 Phase 9.5] 能動的回復試行 — drain + reclaim + 滞留 publish 解除
            tryReclaimResources();
            drainDeferredRetireQueues(false);
            if (runtimeOrchestrator_)
                runtimeOrchestrator_->clearDeferredForShutdown();
            break;

        case convo::RecoveryAction::Restore:
        {
            // [work39 Phase 1] Epoch Recovery + Learner Rollback + Idle World
            // Step1: Epoch Recovery
            if (retireRuntimeEx_.canRollback()) {
                retireRuntimeEx_.setRollbackMode(convo::isr::EpochMode::Split);
                retireRuntimeEx_.requestRollback();
                ++m_restoreGeneration_;
            }
            // 強制回復（Recover との差別化: 二重実行でも安全）
            tryReclaimResources();
            drainDeferredRetireQueues(false);
            // Learner Rollback
            if (lastKnownGoodNoiseShaper_.isValid && noiseShaperLearner)
                noiseShaperLearner->setState(lastKnownGoodNoiseShaper_.state);
            // DeferredPublicationFlush
            if (runtimeOrchestrator_)
                runtimeOrchestrator_->clearDeferredForShutdown();
            // Step2（publishIdleWorldOnly）は閉ループ制御後 (Phase 6)
            m_restorePhase_ = convo::RestorePhase::EpochRecoveryIssued;
            break;
        }

        case convo::RecoveryAction::Safe:
            // [work37 Phase 9.34] EnterSafeMode — Safe Mode World 発行
            diagLog("[RECOVERY] EnterSafeMode: stopping learner");
            stopNoiseShaperLearning();
            // 現在の DSP 設定で Safe Mode 動作 (Convolver バイパス + Learner停止 + admission再開)
            convo::publishAtomic(retirePressureAdmissionStrict_, false,
                                 std::memory_order_release);
            break;

        case convo::RecoveryAction::Critical:
            // 全面新規 publish 拒否
            convo::publishAtomic(retirePressureAdmissionStrict_, true,
                                 std::memory_order_release);
            m_healthMonitor.requestEmergencyDrain();
            break;

        default:
            break;
    }
    diagLog("[RECOVERY] execute action=" + juce::String(static_cast<int>(action)));
}

// [work39 Phase 6] Suppression Probe — CAS reserve
bool AudioEngine::tryReserveProbeBudget() noexcept
{
    uint32_t expected = 1;
    if (convo::compareExchangeAtomic(m_probeBudget_, expected, uint32_t{0},
                                     std::memory_order_acq_rel, std::memory_order_acquire)) {
        m_probeState_.publishSeqBefore = convo::consumeAtomic(
            rtAuxMutable_.runtimePublishCount, std::memory_order_acquire);
        m_probeState_.pendingRetireBefore = convo::consumeAtomic(
            retireQueueDepth_, std::memory_order_acquire);
        m_probeState_.retireAgeBefore = static_cast<uint64_t>(
            convo::consumeAtomic(reclaimLatency_, std::memory_order_acquire));
        m_probeState_.startedUs = convo::getCurrentTimeUs();
        convo::publishAtomic(m_lastProbeUs_, m_probeState_.startedUs,
                             std::memory_order_release);
        m_probeState_.reserveState = AudioEngine::ProbeState::ReserveState::Reserved;
        return true;
    }
    return false;
}

// [work39 Phase 6] Suppression Probe — commit or rollback
void AudioEngine::commitOrRollbackProbe(bool publishSucceeded, uint64_t seqAfter) noexcept
{
    if (publishSucceeded && seqAfter > m_probeState_.publishSeqBefore) {
        m_probeState_.reserveState = AudioEngine::ProbeState::ReserveState::Committed;
    } else {
        convo::fetchAddAtomic(m_probeBudget_, uint32_t{1}, std::memory_order_acq_rel);
        m_probeState_.reserveState = AudioEngine::ProbeState::ReserveState::RolledBack;
        ++m_probeState_.failureCount;
    }
}
