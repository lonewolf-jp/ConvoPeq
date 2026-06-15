#include <JuceHeader.h>
#include "AudioEngine.h"
#include "core/RuntimeReaderContext.h"
#include "RuntimePublicationOrchestrator.h"
#include "NoiseShaperLearner.h"
#include "ISRRetireRouter.h"
#include "DSPLifetimeManager.h"

namespace {
void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}
}

// ★ 静的メンバ定義: 全局一意 Engine インスタンスID カウンタ
std::atomic<uint64_t> AudioEngine::s_nextEngineInstanceId_{0};

AudioEngine::AudioEngine()
    : eqCacheManager(*this)
    , uiEqEditor(*this)
#pragma warning(push)
#pragma warning(disable : 4996) // [[deprecated]] — transitional, SnapshotCoordinator EpochDomain (P1-7)
    , m_coordinator(m_epochDomain)
#pragma warning(pop)
    , m_workerThread(m_commandBuffer, m_generationManager, &affinityManager)
{
    // ★ engineInstanceId 初期化 (全局一意)
    engineInstanceId_ = s_nextEngineInstanceId_.fetch_add(1, std::memory_order_relaxed) + 1; // NOLINT(atomic-dot-call): relaxed counter

    // [work21] ISRRetireRouter初期化
    m_retireRouter = std::make_unique<convo::isr::ISRRetireRouter>(m_epochDomain);
    // [PR-1.5] RuntimePublicationOrchestrator 初期化 (engineInstanceId を注入)
    runtimeOrchestrator_ = std::make_unique<convo::isr::RuntimePublicationOrchestrator>(*this, engineInstanceId_);

    // [P1 Phase1-B] PublicationIntent/PublicationLog initialization removed
    uiConvolverProcessor.setRcuProvider(*this);
    uiConvolverProcessor.setRetireCoordinator(&runtimePublicationBridge_);
    // Route EQ retirement through coordinator
    uiEqEditor.setRetireCoordinator(&runtimePublicationBridge_);

    // ★ P1-8: RuntimeHealthMonitor 初期化
    m_healthMonitor.setRetireRouter(m_retireRouter.get());
    m_healthMonitor.setOrchestrator(runtimeOrchestrator_.get());
    m_healthMonitor.setRetireHighWatermarkRef(&retireHighWatermark_);
    m_healthMonitor.setCrossfadeRuntime(&crossfadeRuntime_);
    m_healthMonitor.setCrossfadeEventDropRef(crossfadeRuntime_.getCrossfadeEventDropCountRef());
    // ★ Work38: Retire Reclaim Latency 監視用参照（型安全）
    //   reclaimLatency_ は AudioEngine の atomic<double> — オーバーロード解決により double* 版が呼ばれる
    m_healthMonitor.setMaxRetireAgeRef(&reclaimLatency_);
    // ★ Practical-4: Reader Slot 使用率監視用参照
    //   activeReaderCount は ISRRetireRouter 経由で取得（HealthMonitor が直接読む）
    m_healthMonitor.setOverflowCountRef(m_retireRouter->getOverflowCountRef());
    m_healthMonitor.setEventCallback(
        [this](const convo::HealthEvent& ev) { onHealthEvent(ev); });

    // [work37 Phase 4.1] PolicyEngine Action Callback
    m_healthMonitor.setActionCallback(
        [this](convo::RecoveryAction action) { executeRecoveryAction(action); });

    // [work39 Phase 6] RestoreStep2 Callback — publishIdleWorldOnly(HardReset)
    m_healthMonitor.setRestoreStep2Callback([this]() {
        if (convo::consumeAtomic(m_lastHardResetGeneration_, std::memory_order_acquire)
            != convo::consumeAtomic(m_restoreGeneration_, std::memory_order_acquire)) {
            const convo::RuntimeReaderContext ctx{
                messageThreadRcuReader, convo::ObserveChannel::Message };
            const auto handle = makeRuntimeReadHandle(ctx);
            auto* dsp = resolveActiveRuntimeDSPFromRuntimeWorldOnly(handle);
            if (dsp != nullptr) {
                (void)publishIdleWorldOnly(dsp, convo::TransitionPolicy::HardReset);
                convo::publishAtomic(m_lastHardResetGeneration_,
                    convo::consumeAtomic(m_restoreGeneration_, std::memory_order_acquire),
                    std::memory_order_release);
                convo::publishAtomic(m_restorePhase_, convo::RestorePhase::IdleWorldPublished,
                    std::memory_order_release);
            }
        }
    });

    // ★ P1-B: Admission に HealthState 参照を設定
    runtimeOrchestrator_->setAdmissionHealthStateRef(m_healthMonitor.getHealthStateRef());
}

AudioEngine::~AudioEngine()
{
    diagLog("[DIAG] ~AudioEngine: enter");
    // Shutdown sequence (list.md 12.1.2):
    // 1) stop callbacks/workers, 2) detach published runtime pointers,
    // 3) retire captured runtimes, 4) force epoch advance, 5) deterministic drain/reclaim.
    // 以後の順序を固定して、終了時の reclaim レースを防止する。
    setShutdownPhase(ShutdownPhase::StopAcceptingWork, "~AudioEngine");
    convo::publishAtomic(lifecycleState, EngineLifecycleState::Releasing, std::memory_order_release); // release: isShuttingDown の acquire と HB
    cancelPendingUpdate();
    runtimePublicationBridge_.requestShutdown();

    // 終了順序を固定化して、終了時フリーズを防ぐ。
    setShutdownPhase(ShutdownPhase::StopAudio, "~AudioEngine");
    stopTimer();

    setShutdownPhase(ShutdownPhase::StopWorkers, "~AudioEngine");
    // releaseResources が未実行の異常系でも worker 終了を保証する。
    stopRebuildThread();

    // まず rebuild thread 側へ終了を通知し、pending task を破棄して
    // 終了時に重い再構築へ入る経路を閉じる。
    // pending task を破棄して進行中 rebuild を obsolete にし、thread を停止する。
    DSPCore* activeToRelease = nullptr;
    DSPCore* fadingToRelease = nullptr;
    {
        std::lock_guard<std::mutex> lock(rebuildMutex);
        const convo::RuntimeReaderContext messageCtx{ messageThreadRcuReader, convo::ObserveChannel::Message };
        const auto runtimeReadHandle = makeRuntimeReadHandle(messageCtx);
        validateDistinctRuntimeSlots("~AudioEngine.beforeClear",
                 getActiveRuntimeDSP(),
                         resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle),
                         nullptr);

        convo::fetchAddAtomic(rebuildRequestGeneration, 1, std::memory_order_acq_rel); // acq_rel: rebuild observer の acquire と HB

        // active runtime slot / fading runtime slot はここでスロットを切り離すだけにして、
        // 実体の解放は retireDSP() → deferred delete / epoch drain に寄せる。
        {
            constexpr uintptr_t kInvalidAllOnes = ~static_cast<uintptr_t>(0);
            DSPCore* activeRaw = getActiveRuntimeDSP();
            activeToRelease = (reinterpret_cast<uintptr_t>(activeRaw) == kInvalidAllOnes) ? nullptr : activeRaw;
        }
        setActiveRuntimeDSP(nullptr);
        {
            auto* const fadingRaw = exchangeFadingRuntimeDSP(nullptr);
            fadingToRelease = (reinterpret_cast<uintptr_t>(fadingRaw) == (~static_cast<uintptr_t>(0))) ? nullptr : fadingRaw;
        }

        validateDistinctRuntimeSlots("~AudioEngine.afterClear",
                 getActiveRuntimeDSP(),
                 resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle),
                         nullptr);

        // pendingTask.currentDSP は worker 側の未コミット生成物なので、
        // ここで回収して以後の commit 経路に残さない。
        if (hasPendingTask)
        {
            if (pendingTask.currentDSP)
            {
                DSPLifetimeManager lifetimeMgr(*this);
                lifetimeMgr.retire(pendingTask.currentDSP);
                pendingTask.currentDSP = nullptr;
            }

            hasPendingTask = false;
        }
    }

    // [P1 Phase1-B] drainPublicationLogForShutdown removed

    {
        DSPLifetimeManager lifetimeMgr(*this);
        if (activeToRelease) lifetimeMgr.retire(activeToRelease);
        if (fadingToRelease) lifetimeMgr.retire(fadingToRelease);
    }

    uiConvolverProcessor.removeChangeListener(this);
    uiEqEditor.removeChangeListener(this);

    // Note: stopRebuildThread は releaseResources() で呼ばれる。
    // dtor が releaseResources 経由で呼ばれる場合、stopRebuildThread は既に完了している。
    // dtor が直接呼ばれる場合（例：ホストが releaseResources を呼ばない異常系）、
    // rebuildThreadShouldExit が既に true なので thread ループは速やかに終了する。

    // Snapshot worker を停止。
    shutdownWorkerThread();

    setShutdownPhase(ShutdownPhase::ForceEpochAdvance, "~AudioEngine");
    m_retireRouter->publishEpoch();

    // ★ Practical-7: Graceful Drain Phase — pendingRetireCount が 0 になるまでポーリング待機
    //   最大 5 秒間のみ待機し、タイムアウト時は強制 drain にフォールバック。
    setShutdownPhase(ShutdownPhase::DrainRetire, "~AudioEngine");
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
            // tick: reclaim を進めて pendingRetire の消化を促進
            m_retireRouter->publishEpoch();
            m_retireRouter->tryReclaim();
        }
        if (waitedMs >= kGracefulDrainMaxMs)
        {
            diagLog("[AUDIT] Graceful drain timeout after " + juce::String(kGracefulDrainMaxMs)
                + "ms, pendingRetireCount="
                + juce::String(static_cast<int>(m_retireRouter->pendingRetireCount()))
                + " — forcing drain");
        }
    }

    auto runtimePublicationCoordinator = makeRuntimePublicationCoordinator();
    runtimePublicationCoordinator.requestShutdownClearNonRt();
    runtimePublicationCoordinator.clearPublishedRuntimeSnapshotsNonRt();
    drainDeferredRetireQueues(true);
    m_epochDomain.drainAll();
    runtimePublicationBridge_.markShutdownComplete();

    // ...既存の解放処理...
    if (latencyBufOldL) { convo::aligned_free(latencyBufOldL); latencyBufOldL = nullptr; }
    if (latencyBufOldR) { convo::aligned_free(latencyBufOldR); latencyBufOldR = nullptr; }
    if (latencyBufNewL) { convo::aligned_free(latencyBufNewL); latencyBufNewL = nullptr; }
    if (latencyBufNewR) { convo::aligned_free(latencyBufNewR); latencyBufNewR = nullptr; }
    latencyBufSize = 0;
    setShutdownPhase(ShutdownPhase::Destroy, "~AudioEngine");
    convo::publishAtomic(lifecycleState, EngineLifecycleState::Destroyed, std::memory_order_release); // release: isShuttingDown の acquire と HB
    diagLog("[DIAG] ~AudioEngine: shutdown sequence complete exit");
}

// [work37 Phase 9.16/9.44] 正常 publish 完了時 — RollbackToLastHealthyWorld + LearnerRollback
void AudioEngine::notifyHealthyPublication(uint64_t worldId) noexcept
{
    convo::publishAtomic(lastHealthyWorldId_, worldId, std::memory_order_release);
    convo::publishAtomic(lastHealthyPublicationTimestampUs_, convo::getCurrentTimeUs(),
                         std::memory_order_release);
    // [work37 Phase 9.44] Learner 正常状態を定期保存
    if (noiseShaperLearner && noiseShaperLearner->isRunning()) {
        convo::NoiseShaperLearnerState current;
        noiseShaperLearner->getState(current);
        lastKnownGoodNoiseShaper_.state = current;
        lastKnownGoodNoiseShaper_.timestampUs = convo::getCurrentTimeUs();
        lastKnownGoodNoiseShaper_.publicationSequence =
            convo::consumeAtomic(publicationSequenceCounter_, std::memory_order_acquire);
        lastKnownGoodNoiseShaper_.isValid = true;
    }
}
