#include <JuceHeader.h>
#include "AudioEngine.h"
#include "core/RuntimeReaderContext.h"
#include "DSPLifetimeManager.h"
#include "RuntimeBuilder.h"
#include "NoiseShaperLearner.h"

namespace {
void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}
}

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

    // P0-A0: LifecycleIsolationRuntime integration - enter release phase
    auto lifecycleToken = lifecycleRuntime_.enterRelease();

    setShutdownPhase(ShutdownPhase::StopAcceptingWork, "releaseResources");
    shutdownRuntime_.transitionTo(convo::isr::ShutdownPhase::AudioStopped);
    runtimePublicationBridge_.requestShutdown();

    // 非MT起点の pending rebuild 要求と AsyncUpdater キューをシャットダウン直後に廃棄する。
    // stopRebuildThread より先に実行して handleAsyncUpdate が後から rebuild を発火しないようにする。
    clearRebuildReason(RebuildReason::StructuralFromNonMT);
    clearRebuildReason(RebuildReason::DeferredStructural);
    clearRebuildReason(RebuildReason::DeferredFinalizeAware);
    convo::publishAtomic(deferredFinalizeFirstSeenTicks_, 0, std::memory_order_release);
    cancelPendingUpdate();
    crossfadeRuntime_.reset();
    convo::publishAtomic(latencyResetPending, false, std::memory_order_release);
    convo::publishAtomic(lastIssuedConvolverStructuralHash_, 0, std::memory_order_release);
    convo::publishAtomic(lastCommittedConvolverStructuralHash_, 0, std::memory_order_release);
    convo::publishAtomic(lastCommittedConvolverHasIr_, false, std::memory_order_release);
    convo::publishAtomic(currentSampleRate, 0.0, std::memory_order_release);

    convo::publishAtomic(inputLevelLinear, 0.0f, std::memory_order_release);
    convo::publishAtomic(outputLevelLinear, 0.0f, std::memory_order_release);

    if (noiseShaperLearner)
    {
        juce::Logger::writeToLog("[AudioEngine] releaseResources: stopping learner");
        noiseShaperLearner->stopLearning();
    }

    resetLearningControlState();
    setShutdownPhase(ShutdownPhase::StopAudio, "releaseResources");

    DSPCore* activeToRelease = nullptr;
    DSPCore* fadingToRelease = nullptr;
    DSPCore* pendingNewToRelease = nullptr;
    DSPCore* pendingCurrentToRelease = nullptr;

    // ★ [PR-A2] DSPLifetimeManager 経由で retire (lifetime は lock 外でも参照可能にする)
    DSPLifetimeManager lifetimeForShutdown(*this);

    {
        std::lock_guard<std::mutex> lk(rebuildMutex);
        const convo::RuntimeReaderContext messageCtx{ messageThreadRcuReader, convo::ObserveChannel::Message };
        const auto runtimeReadHandle = makeRuntimeReadHandle(messageCtx);
        validateDistinctRuntimeSlots("releaseResources.beforeClear",
                 getActiveRuntimeDSP(),
                 resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle),
                         nullptr);

        convo::fetchAddAtomic(rebuildRequestGeneration, 1, std::memory_order_acq_rel);
        {
            auto* const activeRaw = getActiveRuntimeDSP();
            activeToRelease = (reinterpret_cast<uintptr_t>(activeRaw) == (~static_cast<uintptr_t>(0))) ? nullptr : activeRaw;
        }
        setActiveRuntimeDSP(nullptr);

        {
            auto* const fadingRaw = exchangeFadingRuntimeDSP(nullptr);
            fadingToRelease = (reinterpret_cast<uintptr_t>(fadingRaw) == (~static_cast<uintptr_t>(0))) ? nullptr : fadingRaw;
        }
        crossfadeRuntime_.reset();
        refreshCrossfadePreparedSnapshotFromAtomics();

        if (hasPendingTask)
        {
            auto* const pendingRaw = pendingTask.currentDSP;
            pendingCurrentToRelease = (reinterpret_cast<uintptr_t>(pendingRaw) == (~static_cast<uintptr_t>(0))) ? nullptr : pendingRaw;
            pendingTask.currentDSP = nullptr;
            hasPendingTask = false;
        }

        // Migrated to publishWorld() with pre-built RuntimePublishWorld (Sprint-2 P1-A)
        {
            auto coordinator = makeRuntimePublicationCoordinator();
            auto worldBuilder = convo::RuntimeBuilder(*this);
            worldBuilder.setHealthStateRef(getHealthStateRef());
            auto worldOwner = worldBuilder.buildRuntimePublishWorld(nullptr,
                                                                     nullptr,
                                                                     convo::TransitionPolicy::HardReset,
                                                                     0.0,
                                                                     false);
            coordinator.publishWorld(std::move(worldOwner));
        }

        validateDistinctRuntimeSlots("releaseResources.afterClear",
                 getActiveRuntimeDSP(),
                 resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle),
                         nullptr);
    }

    diagLog("[DIAG] releaseResources: before stopRebuildThread");
    setShutdownPhase(ShutdownPhase::StopWorkers, "releaseResources");
    stopRebuildThread();
    shutdownRuntime_.transitionTo(convo::isr::ShutdownPhase::ObserverDrained);
    diagLog("[DIAG] releaseResources: after stopRebuildThread");

    setShutdownPhase(ShutdownPhase::ForceEpochAdvance, "releaseResources");
    advanceRetireEpoch();
    shutdownRuntime_.transitionTo(convo::isr::ShutdownPhase::RetireClosed);
    shutdownRuntime_.transitionTo(convo::isr::ShutdownPhase::EpochSettled);

    setShutdownPhase(ShutdownPhase::DrainRetire, "releaseResources");

    // ★ Practical-7: Graceful Drain Phase（最大5秒間のポーリング待機）
    {
        constexpr int kGracefulDrainMaxMs = 5000;
        constexpr int kGracefulDrainPollMs = 10;
        int waitedMs = 0;
        while (waitedMs < kGracefulDrainMaxMs)
        {
            if (m_retireRouter->pendingRetireCount() == 0
                && m_retireRouter->activeReaderCount() == 0)
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(kGracefulDrainPollMs));
            waitedMs += kGracefulDrainPollMs;
            m_retireRouter->publishEpoch();
            m_retireRouter->tryReclaim();
        }
        if (waitedMs >= kGracefulDrainMaxMs)
        {
            diagLog("[AUDIT] releaseResources: graceful drain timeout after "
                + juce::String(kGracefulDrainMaxMs)
                + "ms, pendingRetireCount="
                + juce::String(static_cast<int>(m_retireRouter->pendingRetireCount()))
                + " — forcing drain");
        }
    }

    // [P1 Phase1-B] drainPublicationLogForShutdown removed

    if (activeToRelease)
        lifetimeForShutdown.retire(activeToRelease);
    if (fadingToRelease)
        lifetimeForShutdown.retire(fadingToRelease);
    if (pendingNewToRelease)
        lifetimeForShutdown.retire(pendingNewToRelease);
    if (pendingCurrentToRelease)
        lifetimeForShutdown.retire(pendingCurrentToRelease);

    // shutdown/release シーケンスでは明示的に deferred retire queue をドレインする。
    // 通常タイマー経路は Releasing 中に early-return するため、ここで最終回収を保証する。
    drainDeferredRetireQueues(true);
    shutdownRuntime_.transitionTo(convo::isr::ShutdownPhase::ReclaimComplete);

    // ★ C-2: EmergencyDrain — Optional 最終手段（デフォルトはスキップ）
    //   常に EmergencyDrain フェーズを経由（ReclaimComplete+1=EmergencyDrain のため単一遷移）
    //   CONVOPEQ_EMERGENCY_DRAIN が定義されている場合のみ有効な処理を実行。
    //   Reader slot の epoch/depth 強制書き換えは一切禁止。
    shutdownRuntime_.transitionTo(convo::isr::ShutdownPhase::EmergencyDrain);
#ifdef CONVOPEQ_EMERGENCY_DRAIN
    {
        diagLog("[DIAG] releaseResources: EmergencyDrain phase enter");

        constexpr int kEmergencyDrainMaxMs = 500;
        const auto emergencyStartMs = juce::Time::getMillisecondCounterHiRes();

        // Deferred publish クリア
        if (runtimeOrchestrator_)
            runtimeOrchestrator_->clearDeferredForShutdown();
        diagLog("[DIAG] releaseResources: EmergencyDrain — cleared deferred publish");

        // 安全な tryReclaim（drainAll 禁止）
        {
            const auto preReclaimPending = m_retireRouter->pendingRetireCount();
            m_epochDomain.tryReclaim();
            const auto postReclaimPending = m_retireRouter->pendingRetireCount();
            diagLog("[DIAG] releaseResources: EmergencyDrain — tryReclaim done (pending "
                + juce::String(static_cast<int>(preReclaimPending)) + " → "
                + juce::String(static_cast<int>(postReclaimPending)) + ")");
        }

        // Crossfade timeout recovery の強制実行
        if (crossfadeRuntime_.isPending())
        {
            diagLog("[DIAG] releaseResources: EmergencyDrain — forcing crossfade recovery");
            crossfadeRuntime_.reset();
            convo::publishAtomic(activeCrossfadeId_, uint64_t{0}, std::memory_order_release);
        }

        const auto emergencyElapsedMs = juce::Time::getMillisecondCounterHiRes() - emergencyStartMs;
        diagLog("[DIAG] releaseResources: EmergencyDrain phase completed in "
            + juce::String(emergencyElapsedMs, 1) + "ms");
        emitEvidenceTickNonRt(true);
    }
#else
    {
        // DiagnosticMode: evidence 出力のみ
        const auto audit = collectDrainAudit();
        if (!audit.isAllZero() || audit.stuckReaderCount > 0)
        {
            diagLog("[DIAG] releaseResources: EmergencyDrain (diagnostic only) — "
                "pendingPub=" + juce::String(static_cast<int64>(audit.pendingPublication)) +
                " pendingRetire=" + juce::String(static_cast<int64>(audit.pendingRetire)) +
                " stuckReaders=" + juce::String(static_cast<int64>(audit.stuckReaderCount)));
        }
    }
#endif // CONVOPEQ_EMERGENCY_DRAIN

    // ★ P3: VerifyDrained — 最終監査フェーズ
    shutdownRuntime_.transitionTo(convo::isr::ShutdownPhase::VerifyDrained);
    diagLog("[DIAG] releaseResources: VerifyDrained — collecting drain audit");

    const auto activeHandle = dspHandleRuntime_.getActiveRuntimeDSPHandle();
    const auto fadingHandle = dspHandleRuntime_.getFadingRuntimeDSPHandle();
    if (!activeHandle.isNull())
    {
        dspHandleRuntime_.retire(activeHandle);
        dspHandleRuntime_.reclaim(activeHandle);
    }
    if (!fadingHandle.isNull() && fadingHandle != activeHandle)
    {
        dspHandleRuntime_.retire(fadingHandle);
        dspHandleRuntime_.reclaim(fadingHandle);
    }
    convo::publishAtomic(activeCrossfadeId_, static_cast<convo::isr::CrossfadeId>(0u), std::memory_order_release);

    diagLog("[DIAG] releaseResources: before ui processor release");
    diagLog("[DIAG] releaseResources: before uiConvolverProcessor.releaseResources");
    uiConvolverProcessor.releaseResources();
    diagLog("[DIAG] releaseResources: after uiConvolverProcessor.releaseResources");

    diagLog("[DIAG] releaseResources: before uiEqEditor.releaseResources");
    uiEqEditor.releaseResources();
    diagLog("[DIAG] releaseResources: after uiEqEditor.releaseResources");

    diagLog("[DIAG] releaseResources: after ui processor release");

    diagLog("[DIAG] releaseResources: skip deferred reclaim (reconfigure phase)");

    auto runtimePublicationCoordinator = makeRuntimePublicationCoordinator();
    runtimePublicationCoordinator.requestShutdownClearNonRt();
    runtimePublicationCoordinator.clearPublishedRuntimeSnapshotsNonRt();

    const bool drainedWithinBudget = waitForDrain(2000, 2);
    const bool timedOut = !drainedWithinBudget;

    if (timedOut) {
        // ★ A-3: VerifyDrained で Reader 異常を検出 → markTimedOut に ReaderActive を伝達
        auto audit = collectDrainAudit();
        auto reason = audit.stuckReaderCount > 0
            ? convo::isr::ShutdownBlockingReason::ReaderActive
            : convo::isr::ShutdownBlockingReason::Unknown;
        shutdownRuntime_.markTimedOut(reason);
    }

    // ★ 改善③: World Consistency 診断は VerifyDrained では常に実行（タイムアウト有無に依存しない）
    {
        const auto audit = collectDrainAudit();
        const auto cs = audit.verifyWorldConsistency();
        if (cs != convo::isr::RuntimeDrainAudit::ConsistencyState::Consistent) {
            diagLog("[AUDIT] VerifyDrained: world consistency="
                + juce::String(static_cast<int>(cs))
                + " published=" + juce::String(static_cast<juce::int64>(audit.publishedCount))
                + " retired=" + juce::String(static_cast<juce::int64>(audit.retiredCount))
                + " active=" + juce::String(static_cast<juce::int64>(audit.activeWorldCount)));
            // ★ B-2: HealthState を診断情報として出力
            diagLog("[AUDIT] VerifyDrained: healthState="
                + juce::String(static_cast<int>(audit.healthState))
                + " activeReaders=" + juce::String(static_cast<juce::int64>(audit.activeReaderCount))
                + " stuckReaders=" + juce::String(static_cast<juce::int64>(audit.stuckReaderCount)));
            emitEvidenceTickNonRt(true);
        }
    }

    if (!drainedWithinBudget || !isFullyDrained())
    {
        if (timedOut)
            diagLog("[DIAG] releaseResources: drain timeout reached, performing safe tryReclaim (drainAll skipped)");

        // [P1 Phase1-B] drainPublicationLogForShutdown removed
        drainDeferredRetireQueues(true);
        m_epochDomain.tryReclaim();  // ★ P1-2: drainAll 禁止 → 安全な tryReclaim
    }

    m_coordinator.finalizeShutdown(timedOut);  // ★ P1-2: 二段構えの正常系

    // ★ A-2.7: ReleaseResources の DrainAudit 統合
    const auto currentShutdownPhase = shutdownRuntime_.getPhase();
    const bool traceSafe = (currentShutdownPhase >= convo::isr::ShutdownPhase::EpochSettled);
    const auto audit = collectDrainAudit();
    if (!drainedWithinBudget || !audit.isAllZero()) {
        diagLog("[ISR][Shutdown] Drain incomplete: "
                "pendingPub=" + juce::String(static_cast<int64>(audit.pendingPublication)) +
                " pendingRetire=" + juce::String(static_cast<int64>(audit.pendingRetire)) +
                " crossfade=" + juce::String(static_cast<int64>(audit.activeCrossfadeCount)) +
                " routerPendingRetire=" + juce::String(static_cast<int64>(audit.routerPendingRetire)) +
                " maxDeferredAgeMs=" + juce::String(static_cast<int64>(audit.maxDeferredAgeMs)) +
                " deferred=" + juce::String(static_cast<int64>(audit.deferredPublish)) +
                " quarantine=" + juce::String(static_cast<int64>(audit.quarantineResident)) +
                " oldestAgeMs=" + juce::String(static_cast<int64>(audit.oldestPendingAgeMs)) +
                " (observation only)");
        if (traceSafe) {
            const auto evidenceRoot = std::filesystem::current_path() / "evidence";
            retireRuntimeEx_.emitRetireTrace(evidenceRoot / "retire_trace_shutdown_last.json");
        }
    }
    if (audit.quarantineResident > 0) {
        diagLog("[ISR][Shutdown] Drain complete but quarantine residents remain: "
                + juce::String(static_cast<int64>(audit.quarantineResident)));
    }

    runtimePublicationBridge_.markShutdownComplete();

    const auto pendingRetireCount = [&]() noexcept -> uint32_t
    {
        return m_retireRouter->pendingRetireCount();
    }();

    const auto activeCrossfadeCount = consumeAtomic(activeCrossfadeId_, std::memory_order_acquire) != static_cast<convo::isr::CrossfadeId>(0u) ? 1u : 0u;
    shutdownRuntime_.setBoundedTeardownCounters(
        convo::consumeAtomic(rtLocalState_.audioCallbackActiveCount, std::memory_order_acquire),
        activeCrossfadeCount,
        pendingRetireCount,
        activeEpochObserverCount());

    debugRuntime_.recordHBEdge(300u,
                               400u,
                               static_cast<std::uint64_t>(pendingRetireCount),
                               static_cast<std::uint64_t>(activeCrossfadeCount),
                               static_cast<int>(std::memory_order_acq_rel));

    shutdownRuntime_.transitionTo(convo::isr::ShutdownPhase::ShutdownComplete);
    shutdownRuntime_.emitShutdownTrace();

    emitEvidenceTickNonRt(true);

    convo::publishAtomic(lifecycleState, EngineLifecycleState::Unprepared, std::memory_order_release);
    diagLog("[DIAG] releaseResources: ABOUT_TO_EXIT_SCOPE");

    // P0-A0: LifecycleIsolationRuntime integration - leave release phase
    lifecycleRuntime_.leaveRelease(lifecycleToken);
}
