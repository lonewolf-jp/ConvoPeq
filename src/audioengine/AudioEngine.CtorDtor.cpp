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
    : uiEqEditor(*this)
    , eqCacheManager(*this)
#pragma warning(push)
#pragma warning(disable : 4996) // [[deprecated]] — transitional, SnapshotCoordinator EpochDomain (P1-7)
    , m_coordinator(m_epochDomain)
#pragma warning(pop)
    , m_workerThread(m_commandBuffer, m_generationManager, &affinityManager)
    , worldAuthority_(runtimePublicationBridge_)
    , shutdownRuntime_(runtimePublicationBridge_)  // ★ dash2 §2.2 (Step 14 — Authority Singularization 完了): constructor 固定注入（setReclaimAuthority 廃止 — immutable association）
{
    // ★ engineInstanceId 初期化 (全局一意)
    engineInstanceId_ = s_nextEngineInstanceId_.fetch_add(1, std::memory_order_relaxed) + 1; // NOLINT(atomic-dot-call): relaxed counter

    // [work21] ISRRetireRouter初期化
    // ★ T1 (D100.4): reference observer に telemetry を配線（onRelease → releaseObserved 転送・
    //   sampler の outstanding 推定 acquireObserved - releaseObserved を正しくする）。
    worldRetirementReference_.setTelemetry(&worldRetirementTelemetry_);
    m_retireRouter = std::make_unique<convo::isr::ISRRetireRouter>(m_epochDomain, &worldRetirementReference_);
    // ★ BUG-015/027 (work88): SnapshotCoordinator の退避移送先を Router に接続
    //   （Category A — Router API 経由で RetireQuarantineStore へ移送。直接保持はしない）
    m_coordinator.setRetireSink(m_retireRouter.get());
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

    // ★ B14: Vyukov MPSC Retire Queue 初期化
    worldAuthority_.lifetime().initQueue();
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
    shutdownCoordinatorLoop();  // ★ FUTURE-9: join Coordinator Worker (defensive)
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
        // 実体の解放は retireDSPHandleForRuntime() → deferred delete / epoch drain に寄せる。
        {
            // ★ BUG-051: sentinel (uintptr_t)-1 は書き込まれない（常に nullptr or 有効 ptr）。
            activeToRelease = getActiveRuntimeDSP();
        }
        setActiveRuntimeDSP(nullptr);
        {
            // ★ B-1: CAS-based fading slot clear
            DSPCore* current = convo::consumeAtomic(fadingRuntimeDSPSlot, std::memory_order_acquire);
            if (current != nullptr
                && convo::compareExchangeAtomic(fadingRuntimeDSPSlot, current,
                                                 static_cast<DSPCore*>(nullptr),
                                                 std::memory_order_acq_rel,
                                                 std::memory_order_acquire))
                fadingToRelease = current;
            else
                fadingToRelease = nullptr;
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
            publishRetryReady = false;
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
    // ★ work88 (X3 §6.3 / INV-X3-4 / INV-ISR-04): CloseReaderRegistration（系統2 — ~AudioEngine）。
    //   releaseResources 未実行の異常系 shutdown でも、graceful drain 前に reader 新規登録を封じる
    //   （0 に達した後の再登録を構造的に排除）。登録済み slot の enter/exit は継続可能。
    m_epochDomain.closeReaderRegistration();
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

    // ★ work88 (X4-B §6.4 / X4-B-7): shutdown clear を RuntimeWorldAuthority 経由に一本化
    //   （一時生成 makeRuntimePublishAuthority() は X4-B-8 で廃止）。詳細は ReleaseResources 側に同様。
    worldAuthority_.requestShutdownClearNonRt();
    auto* clearedWorld = worldAuthority_.clearPublishedRuntimeSnapshotsNonRt();
    if (clearedWorld != nullptr)
    {
        RuntimePublicationBridge clearBridge{ *this, runtimePublicationValidator_ };
        clearBridge.retirePublishedRuntimeWorldNonRt(clearedWorld, true);
    }
    drainDeferredRetireQueues(true);

    // ★ 15-P-4-5-FIX: Drain residual RetireIntents (slot-state System 1) before pointer-lifetime
    //   drain (drainAll). OverflowRing may hold entries not yet processed because the RT commit
    //   path (the only caller of dequeuePendingRetireIntents()) has stopped.
    drainPendingRetireIntentsForShutdown();

    // ★ 15-P-5: 完全 drain（D + Q + E + Terminal）。m_epochDomain.drainAll() は D のみのため、
    //   TerminalReclaimAuthority に保持された World（stuck reader ケースの clearedWorld 等）が
    //   漏れる。quiescence（activeReaderCount==0）確立時のみ m_retireRouter->drainAll() で
    //   全 store を強制解放する。stuck reader が残る場合は UAF 回避のため D のみ（従来動作）に
    //   フォールバックし、epoch-gated drain（drainTerminalReclaim）に委ねる。
    // ★ 15-P-5 FIX: stuck-reader fallback は D のみにしない。Audio Thread は停止済みのため、
    //   Q + E + T の drainAllQuarantineStore（drainAllUnsafe ベース、epoch 非依存）も強制実行する。
    //   drainAllUnsafe は Audio Thread 停止後のみ呼ばれる契約を満たす（RetireQuarantineStore.h:59参照）。
    if (m_retireRouter->activeReaderCount() == 0)
        m_retireRouter->drainAll();
    else
    {
        diagLog("[DRAIN] Destructor stuck-reader fallback: activeReaderCount > 0 — draining D + Q + E + T");
        m_epochDomain.drainAll();           // D only (safe — no live readers can access D slots in dtor)
        m_retireRouter->drainAllQuarantineStore();  // Q + E + T force-drain (epoch-agnostic, Audio Thread stopped)
    }
    runtimePublicationBridge_.markShutdownComplete();

    // ★ 15-P-5: Post-shutdown Faulted state diagnostic
    //   markShutdownComplete 後、Coordinator が Faulted 状態である場合は shutdown 異常を記録。
    if (runtimePublicationBridge_.getState() == convo::isr::RuntimeIntentCoordinator::CoordinatorState::Faulted)
        diagLog("[FAULT] ~AudioEngine: coordinator in Faulted state after markShutdownComplete — "
                "residual intents may remain in System 1 queues");

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
