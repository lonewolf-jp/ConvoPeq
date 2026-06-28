#include <JuceHeader.h>
#include "AudioEngine.h"
#include "DiagnosticsConfig.h"
#include "core/RuntimeReaderContext.h"
#include "RuntimeBuilder.h"
#include "RuntimePublicationOrchestrator.h"
#include "DSPLifetimeManager.h"
#include "../NoiseShaperLearner.h"  // ★ Work39: Restore Learner Rollback 用

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#endif

namespace {

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
juce::String diagPrefix(uint64_t currentGeneration)
{
    const auto now = juce::Time::getCurrentTime();
    const auto timestamp = now.formatted("%H:%M:%S.")
        + juce::String(now.getMilliseconds()).paddedLeft('0', 3);
    const auto ticks = juce::Time::getHighResolutionTicks();
    return "[" + timestamp + "]"
        + " Gen=" + juce::String(static_cast<juce::int64>(currentGeneration))
        + " Ticks=" + juce::String(static_cast<juce::int64>(ticks));
}

#endif // CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS

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

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // ★ Timer exec開始時刻（timerCallback全体の実行時間計測用）
    //    関数スコープのstatic変数。末尾のexecブロックと共有。
    static double s_timerExecStartMs = 0.0;
    s_timerExecStartMs = juce::Time::getMillisecondCounterHiRes();

    // ★ Timer jitter計測（timerCallback先頭）
    {
        static double s_prevCallbackMs = 0.0;
        const double nowMs = juce::Time::getMillisecondCounterHiRes();
        const double expectedMs = static_cast<double>(timerPeriodMs_);

        if (s_prevCallbackMs > 0.0) {
            const double timerIntervalMs = nowMs - s_prevCallbackMs;
            const double jitterMs = timerIntervalMs - expectedMs;
            const double jitterThreshold = std::max(20.0, expectedMs * 0.1);
            if (std::abs(jitterMs) > jitterThreshold) {
                const int estimatedMissed = static_cast<int>(std::abs(jitterMs) / expectedMs);
                const uint64_t gen = (runtimeWorld != nullptr)
                    ? static_cast<uint64_t>(runtimeWorld->generation) : 0;
                const auto seq = convo::fetchAddAtomic(diagSequenceCounter(), uint64_t{1}, std::memory_order_acq_rel) + 1u;
                diagLog(diagPrefix(gen) + " [Seq=" + juce::String(static_cast<juce::int64>(seq))
                    + "] [TIMER] jitter: interval=" + juce::String(timerIntervalMs, 2) + "ms"
                    + " (expected=" + juce::String(expectedMs, 0) + "ms, delta=" + juce::String(jitterMs, 2) + "ms"
                    + ", estimatedMissed=" + juce::String(estimatedMissed) + ")");
            }
        }
        s_prevCallbackMs = nowMs;
    }
#endif

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
        else
        {
            // ★ Log world pointer every 10th tick to detect stale cache
            static int tickCounter = 0;
            if (++tickCounter % 10 == 0 && runtimeWorld != nullptr)
            {
                diagLog("[VERIFY] worldPtr=0x" + juce::String::toHexString(static_cast<juce::int64>(reinterpret_cast<uintptr_t>(runtimeWorld)))
                    + " rev=" + juce::String(static_cast<juce::int64>(revision)));
            }
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

    // AdaptiveCoeff authority divergence: worldGen vs live bankGen
    {
        const int worldGen = (runtimeWorld != nullptr)
            ? static_cast<int>(runtimeWorld->coefficient.adaptiveCoeffGeneration)
            : -1;
        const int liveBankIdx = convo::consumeAtomic(currentAdaptiveCoeffBankIndex, std::memory_order_acquire);
        const auto& bank = getAdaptiveCoeffBankForIndex(liveBankIdx);
        const int liveBankGen = static_cast<int>(convo::consumeAtomic(bank.generation, std::memory_order_acquire));

        if (worldGen != rtAuxMutable_.debugLastReportedWorldGen
            || liveBankGen != rtAuxMutable_.debugLastReportedBankGen
            || liveBankIdx != rtAuxMutable_.debugLastReportedCoeffBankIdx)
        {
            rtAuxMutable_.debugLastReportedWorldGen = worldGen;
            rtAuxMutable_.debugLastReportedBankGen = liveBankGen;
            rtAuxMutable_.debugLastReportedCoeffBankIdx = liveBankIdx;

            diagLog("[COEFF_AUTH] worldGen=" + juce::String(worldGen)
                + " bankGen=" + juce::String(liveBankGen)
                + " bankIdx=" + juce::String(liveBankIdx)
                + " lag=" + juce::String(liveBankGen - worldGen));
        }
    }

    // Adaptive bank switch count (ISR-safe atomic counter from DSPCore, read on Message Thread)
    // ★ 2026-06-23: DSP instance UUID を出力し、OLD DSP と NEW DSP の切り替えを区別可能に。
    //   runtimeUuid は DSPCore 構築時に一度設定され不変のため Timer スレッドから安全に読取可。
    {
        auto* dsp = resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle);
        if (dsp != nullptr)
        {
            const uint64_t count = dsp->adaptiveBankSwitchCount.load(std::memory_order_relaxed);
            if (count != rtAuxMutable_.debugLastReportedBankSwitchCount)
            {
                rtAuxMutable_.debugLastReportedBankSwitchCount = count;
                diagLog("[ADAPTIVE_SWITCH] dspUuid=" + juce::String(static_cast<juce::int64>(dsp->runtimeUuid))
                    + " count=" + juce::String(static_cast<juce::int64>(count)));
            }
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

    // ★ XFADE start 検出: crossfadeRuntime_ が非pending→pendingに遷移した瞬間を記録
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    {
        static bool s_prevPending = false;
        const bool nowPending = crossfadeRuntime_.isPending();
        if (nowPending && !s_prevPending) {
            const double expectedSec = crossfadeRuntime_.getQueuedFadeTimeSec();
            const uint64_t gen = (runtimeWorld != nullptr) ? static_cast<uint64_t>(runtimeWorld->generation) : 0;
            const auto seq = convo::fetchAddAtomic(diagSequenceCounter(), uint64_t{1}, std::memory_order_acq_rel) + 1u;
            diagLog(diagPrefix(gen) + " [Seq=" + juce::String(static_cast<juce::int64>(seq))
                + "] [XFADE] start expected=" + juce::String(expectedSec, 3) + "s");
        }
        s_prevPending = nowPending;
    }
#endif

    // Grace period に基づく安全なリリース遅延を実行する。
    processDeferredReleases();

    const bool fadeCompleted = m_coordinator.tryCompleteFade();
    if (fadeCompleted)
    {
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        // ★ XFADE completed ログ（既存のfade完了ブロックに追記）
        {
            const double elapsedSec = static_cast<double>(crossfadeRuntime_.getFadeAgeUs()) / 1'000'000.0;
            const double expectedSec = crossfadeRuntime_.getQueuedFadeTimeSec();
            const uint64_t gen = (runtimeWorld != nullptr) ? static_cast<uint64_t>(runtimeWorld->generation) : 0;
            const auto seq = convo::fetchAddAtomic(diagSequenceCounter(), uint64_t{1}, std::memory_order_acq_rel) + 1u;
            diagLog(diagPrefix(gen) + " [Seq=" + juce::String(static_cast<juce::int64>(seq))
                + "] [XFADE] completed: elapsed=" + juce::String(elapsedSec, 3) + "s"
                + " expected=" + juce::String(expectedSec, 3) + "s");
        }
#endif
        // ★ PR2/PR4: Authority の Registry から active crossfade を取得
        auto records = crossfadeAuthorityRuntime_.getActiveCrossfades();
        if (!records.empty())
        {
            // 単一 Crossfade 前提を表明
            jassert(records.size() == 1);
            const auto xfadeId = records.front().id;
            crossfadeRuntime_.notifyFadeComplete(xfadeId);
            convo::isr::CompletedFadeEvent ev;
            while (crossfadeRuntime_.consumeCompletedFade(ev))
            {
                dspHandleRuntime_.endCrossfade(ev.id);
                crossfadeAuthorityRuntime_.unregisterCrossfade(ev.id);
            }
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

    // ★ Phase1: OverflowRing 定期 drain — Coordinator 経由で一元管理
    //   50ms周期のtimerCallbackごとにdrainOverflowRingを呼出
    //   Coordinator が retry/age/deferred を管理
    {
        if (retireRuntime_.getOverflowRing())
        {
            const auto drainResult = runtimePublicationBridge_.drainOverflowRing(
                *retireRuntime_.getOverflowRing(), retireRuntime_, false);
            if (drainResult.reinjectedCount > 0)
            {
                m_retireRouter->tryReclaim();
            }
        }
    }

    // UI用プロセッサのクリーンアップ
    uiEqEditor.cleanup();
    uiConvolverProcessor.cleanup();

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // ★ 計測ログ: Backpressure 診断（変化時のみ [BACKPRESSURE] として出力）
    {
        const auto backpressure = getRuntimeBackpressureTelemetry();
        if (backpressure.retireQueueDepth != rtAuxMutable_.debugLastReportedRetireQueueDepth
            || backpressure.fallbackQueueDepth != rtAuxMutable_.debugLastReportedFallbackQueueDepth
            || backpressure.quarantineResident != rtAuxMutable_.debugLastReportedQuarantineResident
            || backpressure.retirePressureLevel != rtAuxMutable_.debugLastReportedRetirePressureLevel)
        {
            rtAuxMutable_.debugLastReportedRetireQueueDepth = backpressure.retireQueueDepth;
            rtAuxMutable_.debugLastReportedFallbackQueueDepth = backpressure.fallbackQueueDepth;
            rtAuxMutable_.debugLastReportedQuarantineResident = backpressure.quarantineResident;
            rtAuxMutable_.debugLastReportedRetirePressureLevel = backpressure.retirePressureLevel;

            const uint64_t gen = (runtimeWorld != nullptr) ? static_cast<uint64_t>(runtimeWorld->generation) : 0;
            diagLog(diagPrefix(gen) + " [BACKPRESSURE] retireDepth="
                + juce::String(static_cast<juce::int64>(backpressure.retireQueueDepth))
                + " fallback=" + juce::String(static_cast<juce::int64>(backpressure.fallbackQueueDepth))
                + " quarantine=" + juce::String(static_cast<juce::int64>(backpressure.quarantineResident))
                + " pressureLevel=" + juce::String(backpressure.retirePressureLevel));
        }
    }

    // ★ メモリ情報のキャッシュ
    // MEMブロックで毎秒更新。XRUN消費ループではこのキャッシュを参照して
    // GetProcessMemoryInfo() の多重呼び出しを避ける。
    static ProcessMemoryInfo s_cachedMemInfo {};

    // ★ 計測ログ: Memory 定期ログ（1秒ごと・常時出力）+ PageFault Delta警告
    {
        static int64_t lastMemLogTicks = 0;
        static uint64_t s_prevPageFaults = 0;
        static double s_pfEwmaAvg = 0.0;
        static int s_pfSampleCount = 0;
        static constexpr int kEwmaWarmupSamples = 10;
        static constexpr uint64_t kAbsoluteThreshold = 50000;

        const auto nowTicks = juce::Time::getHighResolutionTicks();
        const auto ticksPerSec = juce::Time::getHighResolutionTicksPerSecond();
        if (nowTicks - lastMemLogTicks >= ticksPerSec)
        {
            lastMemLogTicks = nowTicks;
            const auto memInfo = getProcessMemoryInfo();
            s_cachedMemInfo = memInfo;  // ★ XRUN消費ループ用にキャッシュ更新
            const uint64_t gen = (runtimeWorld != nullptr) ? static_cast<uint64_t>(runtimeWorld->generation) : 0;
            const auto seq = convo::fetchAddAtomic(diagSequenceCounter(), uint64_t{1}, std::memory_order_acq_rel) + 1u;

            // Delta計算（初回は0）
            const uint64_t pfDelta = (s_prevPageFaults > 0)
                ? (memInfo.pageFaultCount - s_prevPageFaults) : 0;
            s_prevPageFaults = memInfo.pageFaultCount;

            // ★【重要】PageFault警告判定はEWMA更新の前に行う
            const double thresholdRel = (s_pfSampleCount >= kEwmaWarmupSamples)
                ? std::max(1000.0, s_pfEwmaAvg * 5.0) : 0.0;
            const uint64_t threshold = std::max<uint64_t>(kAbsoluteThreshold,
                static_cast<uint64_t>(thresholdRel));
            const bool warnAbsolute = (pfDelta > kAbsoluteThreshold);
            const bool warnRelative = (s_pfSampleCount >= kEwmaWarmupSamples)
                && (pfDelta > static_cast<uint64_t>(thresholdRel));

            // ★ MEMログ（Seq+Delta+Pagefile付き）
            diagLog(diagPrefix(gen) + " [Seq=" + juce::String(static_cast<juce::int64>(seq))
                + "] [MEM] Private=" + juce::String(static_cast<juce::int64>(memInfo.privateUsageMB)) + "MB"
                + " WS=" + juce::String(static_cast<juce::int64>(memInfo.workingSetMB)) + "MB"
                + " Pagefile=" + juce::String(static_cast<juce::int64>(memInfo.pagefileUsageMB)) + "MB"
                + " PageFaults=" + juce::String(static_cast<juce::int64>(memInfo.pageFaultCount))
                + " Delta=+" + juce::String(static_cast<juce::int64>(pfDelta)));

            // ★ PageFault警告（absolute OR relative）
            if (pfDelta > 0 && (warnAbsolute || warnRelative)) {
                const auto warnSeq = convo::fetchAddAtomic(diagSequenceCounter(), uint64_t{1}, std::memory_order_acq_rel) + 1u;
                diagLog(diagPrefix(gen) + " [Seq=" + juce::String(static_cast<juce::int64>(warnSeq))
                    + "] [WARN] PageFault surge: +" + juce::String(static_cast<juce::int64>(pfDelta))
                    + " faults (EWMA=" + juce::String(s_pfEwmaAvg, 0)
                    + ", threshold=" + juce::String(static_cast<juce::int64>(threshold)) + ")");
            }

            // ★ EWMA更新（警告判定の後）+ クリッピング
            const double clippedDelta = std::min<double>(static_cast<double>(pfDelta), 50000.0);
            if (s_pfSampleCount < kEwmaWarmupSamples) {
                s_pfEwmaAvg = (s_pfEwmaAvg * static_cast<double>(s_pfSampleCount) + clippedDelta)
                    / static_cast<double>(s_pfSampleCount + 1);
                ++s_pfSampleCount;
            } else {
                s_pfEwmaAvg = 0.05 * clippedDelta + 0.95 * s_pfEwmaAvg;
            }

            // ★ WS減少警告（absolute AND relative）
            //    前回のWSをstatic保持し、5%以上かつ50MB以上の減少を検出
            {
                static uint64_t s_prevWsMB = 0;
                if (s_prevWsMB > 0 && memInfo.workingSetMB < s_prevWsMB) {
                    const uint64_t wsDropMB = s_prevWsMB - memInfo.workingSetMB;
                    const double wsDropPct = static_cast<double>(wsDropMB) / static_cast<double>(s_prevWsMB) * 100.0;
                    static constexpr uint64_t kMinWsDropMB = 50;
                    static constexpr double kMinWsDropPct = 5.0;
                    if (wsDropMB >= kMinWsDropMB && wsDropPct >= kMinWsDropPct) {
                        const auto warnSeq = convo::fetchAddAtomic(diagSequenceCounter(), uint64_t{1}, std::memory_order_acq_rel) + 1u;
                        diagLog(diagPrefix(gen) + " [Seq=" + juce::String(static_cast<juce::int64>(warnSeq))
                            + "] [WARN] WS dropped: " + juce::String(static_cast<juce::int64>(wsDropMB)) + "MB"
                            + " (" + juce::String(wsDropPct, 1) + "%)"
                            + " from " + juce::String(static_cast<juce::int64>(s_prevWsMB)) + "MB"
                            + " to " + juce::String(static_cast<juce::int64>(memInfo.workingSetMB)) + "MB");
                    }
                }
                s_prevWsMB = memInfo.workingSetMB;
            }
        }
    }

    // ★ 計測ログ: Audio callback 1秒サマリ（CBSUMMARY）
    //    intervalMax を主指標、callbackMax を副指標とする。
    //    変化があった秒のみ出力（通常時0行）。
    {
        static int64_t lastCbSummaryTicks = 0;
        const auto nowTicks = juce::Time::getHighResolutionTicks();
        const auto ticksPerSec = juce::Time::getHighResolutionTicksPerSecond();
        if (nowTicks - lastCbSummaryTicks >= ticksPerSec)
        {
            lastCbSummaryTicks = nowTicks;
            const uint32_t ivMaxUs = convo::exchangeAtomic(intervalMaxUs_, 0u, std::memory_order_relaxed);
            const uint32_t cbMaxUs = convo::exchangeAtomic(callbackMaxUs_, 0u, std::memory_order_relaxed);
            const uint32_t cbCount = convo::exchangeAtomic(callbackCount_, 0u, std::memory_order_relaxed);

            const uint32_t expectedCbCount = (currentSampleRate.load(std::memory_order_relaxed) > 0.0
                && maxSamplesPerBlock.load(std::memory_order_relaxed) > 0)
                ? static_cast<uint32_t>(currentSampleRate.load(std::memory_order_relaxed)
                    / static_cast<double>(maxSamplesPerBlock.load(std::memory_order_relaxed)))
                : 0;
            const int32_t lossCount = static_cast<int32_t>(expectedCbCount) - static_cast<int32_t>(cbCount);

            static uint32_t s_prevIntervalMaxUs = 0;
            static uint32_t s_prevCbMaxUs = 0;
            const bool hasChange = (ivMaxUs != s_prevIntervalMaxUs)
                || (cbMaxUs != s_prevCbMaxUs);
            s_prevIntervalMaxUs = ivMaxUs;
            s_prevCbMaxUs = cbMaxUs;

            if (hasChange && (ivMaxUs > 0 || cbMaxUs > 0 || cbCount > 0)) {
                const uint64_t gen = (runtimeWorld != nullptr) ? static_cast<uint64_t>(runtimeWorld->generation) : 0;
                const auto seq = convo::fetchAddAtomic(diagSequenceCounter(), uint64_t{1}, std::memory_order_acq_rel) + 1u;
                diagLog(diagPrefix(gen) + " [Seq=" + juce::String(static_cast<juce::int64>(seq))
                    + "] [CBSUMMARY] intervalMax=" + juce::String(static_cast<double>(ivMaxUs) / 1000.0, 3) + "ms"
                    + " callbackMax=" + juce::String(static_cast<double>(cbMaxUs) / 1000.0, 3) + "ms"
                    + " (expected=" + juce::String(static_cast<juce::int64>(expectedCbCount))
                    + " actual=" + juce::String(static_cast<juce::int64>(cbCount))
                    + " loss=" + juce::String(static_cast<juce::int64>(lossCount)) + ")");
            }
        }
    }

    // ★ 計測ログ: WORLD count（Active/Fading/Retired）
    {
        auto* activeDsp = resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle);
        auto* fadingDsp = resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle);
        const auto backpressure = getRuntimeBackpressureTelemetry();
        const int activeCount = (activeDsp != nullptr) ? 1 : 0;
        const int fadingCount = (fadingDsp != nullptr) ? 1 : 0;

        if (activeCount != rtAuxMutable_.debugLastReportedWorldActiveCount
            || fadingCount != rtAuxMutable_.debugLastReportedWorldFadingCount
            || backpressure.retireQueueDepth != rtAuxMutable_.debugLastReportedRetireQueueDepth)
        {
            rtAuxMutable_.debugLastReportedWorldActiveCount = activeCount;
            rtAuxMutable_.debugLastReportedWorldFadingCount = fadingCount;

            const uint64_t gen = (runtimeWorld != nullptr) ? static_cast<uint64_t>(runtimeWorld->generation) : 0;
            diagLog(diagPrefix(gen) + " [WORLD] Active=" + juce::String(activeCount)
                + " Fading=" + juce::String(fadingCount)
                + " RetireQueue=" + juce::String(static_cast<juce::int64>(backpressure.retireQueueDepth))
                + " Quarantine=" + juce::String(static_cast<juce::int64>(backpressure.quarantineResident)));
        }
    }

    // ★ 計測ログ: XRUN リングバッファ消費（100ms毎）
    {
        // XRUN drop count を読み取り・リセット
        const uint64_t xRunDropped = convo::exchangeAtomic(rtAuxMutable_.xRunDropCount, 0, std::memory_order_acq_rel);
        if (xRunDropped > 0)
        {
            const uint64_t gen = (runtimeWorld != nullptr) ? static_cast<uint64_t>(runtimeWorld->generation) : 0;
            diagLog(diagPrefix(gen) + " [XRUN] dropped="
                + juce::String(static_cast<juce::int64>(xRunDropped))
                + " events (ring buffer full)");
        }

        // ★ メモリはキャッシュから取得（毎回 GetProcessMemoryInfo を呼ばない）
        //    初回は MEM ブロックより先に XRUN が発生する可能性があるため、
        //    キャッシュが空ならここで初期化する。
        if (s_cachedMemInfo.privateUsageMB == 0 && s_cachedMemInfo.workingSetMB == 0)
        {
            s_cachedMemInfo = getProcessMemoryInfo();
        }

        XRunEvent ev;
        while (xRunBuffer.pop(ev))
        {
            const auto backpressure = getRuntimeBackpressureTelemetry();
            const auto lifecycle = getRuntimeLifecycleDiagnostics();
            const uint64_t gen = (runtimeWorld != nullptr) ? static_cast<uint64_t>(runtimeWorld->generation) : 0;

            // ★ ACTIVATE イベント（callbackMs==0 && intervalMs==0 が ACTIVATE）
            if (ev.callbackMs == 0.0 && ev.intervalMs == 0.0)
            {
                diagLog(diagPrefix(gen) + " [ACTIVATE] EventGen=" + juce::String(ev.generation));
                continue;
            }

            // ★ XRUN イベント — メモリ情報はキャッシュを使用
            const juce::String xrunId = "XRUN#" + juce::String(static_cast<juce::int64>(ev.sequenceNumber));
            const auto* activeDsp = resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle);
            const auto* fadingDsp = resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle);
            const int activeCount = (activeDsp != nullptr) ? 1 : 0;
            const int fadingCount = (fadingDsp != nullptr) ? 1 : 0;

            diagLog(diagPrefix(gen) + " [" + xrunId + "]"
                + " Callback=" + juce::String(ev.callbackMs, 2) + "ms"
                + " Interval=" + juce::String(ev.intervalMs, 2) + "ms"
                + " Expected=" + juce::String(ev.expectedMs, 2) + "ms"
                + " EventGen=" + juce::String(ev.generation)
                + " RetireDepth=" + juce::String(static_cast<juce::int64>(ev.retireQueueDepth))
                + " Private=" + juce::String(static_cast<juce::int64>(s_cachedMemInfo.privateUsageMB)) + "MB"
                + " WS=" + juce::String(static_cast<juce::int64>(s_cachedMemInfo.workingSetMB)) + "MB"
                + " World=" + juce::String(activeCount) + "/" + juce::String(fadingCount)
                + "/" + juce::String(static_cast<juce::int64>(backpressure.retireQueueDepth))
                + " Pressure=" + juce::String(backpressure.retirePressureLevel)
                + " PublishTotal=" + juce::String(static_cast<juce::int64>(lifecycle.publishCount)));
        }
    }

    // ★ timerCallback 末尾 — 実行時間計測
    {
        static double s_timerStartMs = 0.0;
        if (s_timerStartMs > 0.0) {
            const double execMs = juce::Time::getMillisecondCounterHiRes() - s_timerExecStartMs;
            if (execMs > 10.0) {
                const uint64_t gen = (runtimeWorld != nullptr) ? static_cast<uint64_t>(runtimeWorld->generation) : 0;
                const auto seq = convo::fetchAddAtomic(diagSequenceCounter(), uint64_t{1}, std::memory_order_acq_rel) + 1u;
                diagLog(diagPrefix(gen) + " [Seq=" + juce::String(static_cast<juce::int64>(seq))
                    + "] [TIMER] exec=" + juce::String(execMs, 3) + "ms");
            }
        }
        s_timerExecStartMs = juce::Time::getMillisecondCounterHiRes();
    }
#endif // CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
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

        // 2. ★ PR2/PR4: Authority の Registry から全 active レコードを取得
        auto records = crossfadeAuthorityRuntime_.getActiveCrossfades();
        jassert(records.size() <= 1);
        if (records.size() > 1) {
            diagLog("[DIAG] Crossfade: multiple active crossfades detected (count="
                + juce::String(static_cast<int>(records.size()))
                + "), clearing all via timeout recovery");
        }
        for (const auto& record : records)
            crossfadeAuthorityRuntime_.unregisterCrossfade(record.id);

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

    // ★ A-3: EVENT_READER_STUCK — Evidence 出力 + Reader quarantine + High優先度Retire
    if (event.eventCode == convo::EVENT_READER_STUCK)
    {
        diagLog("[HEALTH] Reader stuck detected"
            + juce::String(" readerIndex=") + juce::String(static_cast<int>(event.readerIndex))
            + juce::String(" residencyUs=") + juce::String(static_cast<juce::int64>(event.residencyTimeUs))
            + juce::String(" pendingRetire=") + juce::String(static_cast<juce::int64>(event.value)));

        // ★ Phase3: この Reader を quarantine（stuck Reader の epoch を safe-epoch 計算から除外）
        const bool immediate = m_retireRouter->quarantineReader(event.readerIndex);
        diagLog("[PHASE3] quarantineReader idx=" + juce::String(static_cast<int>(event.readerIndex))
            + " immediate=" + (immediate ? "true" : "false"));

        // ★ Phase5: quarantine された Reader が参照していた slot の Retire を High 優先度で投入
        //   （stuck Reader の epoch が safe-epoch から除外されたため、Reclaim が進行可能）
        if (immediate && event.slot != std::numeric_limits<uint32_t>::max())
        {
            // 即座 quarantine 成功 → 該当 slot の RetireIntent を High 優先度で発行
            convo::isr::RetireIntent highIntent{};
            highIntent.dspSlot = event.slot;
            highIntent.generation = event.readerEpoch;
            highIntent.retireEpoch = m_retireRouter->currentEpoch();
            highIntent.isValid = true;
            highIntent.priority = convo::isr::RetirePriority::High;
            retireRuntime_.emitRetireIntent(highIntent);
            diagLog("[PHASE5] High priority retire emitted for slot="
                + juce::String(static_cast<int>(event.slot)));
        }

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

