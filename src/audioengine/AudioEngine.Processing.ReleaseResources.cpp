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
    if (!drainedWithinBudget || !isFullyDrained())
    {
        if (!drainedWithinBudget)
            diagLog("[DIAG] releaseResources: drain timeout reached, performing one emergency reclaim boost path");

        // [P1 Phase1-B] drainPublicationLogForShutdown removed
        drainDeferredRetireQueues(true);
        m_epochDomain.drainAll();
    }

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
