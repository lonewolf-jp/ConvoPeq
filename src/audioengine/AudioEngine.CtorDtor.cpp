#include <JuceHeader.h>
#include "AudioEngine.h"
#include "core/RuntimeReaderContext.h"
#include "RuntimePublicationOrchestrator.h"
#include "NoiseShaperLearner.h"
#include "ISRRetireRouter.h"

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
    m_healthMonitor.setEventCallback(
        [this](const convo::HealthEvent& ev) { onHealthEvent(ev); });

    // 必要な初期化処理があればここに追加
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
                retireDSP(pendingTask.currentDSP);
                pendingTask.currentDSP = nullptr;
            }

            hasPendingTask = false;
        }
    }

    // [P1 Phase1-B] drainPublicationLogForShutdown removed

    if (activeToRelease) retireDSP(activeToRelease);
    if (fadingToRelease) retireDSP(fadingToRelease);

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

    // Shutdown 時は EBR 回収を試みる。
    setShutdownPhase(ShutdownPhase::DrainRetire, "~AudioEngine");
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
