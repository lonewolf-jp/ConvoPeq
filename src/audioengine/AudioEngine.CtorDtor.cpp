#include <JuceHeader.h>
#include "AudioEngine.h"
#include "NoiseShaperLearner.h"

namespace {
static void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}
}

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_CTOR_DTOR)

AudioEngine::AudioEngine()
    : eqCacheManager(*this)
    , uiEqEditor(*this)
    , m_coordinator(m_epochDomain)
    , m_workerThread(m_commandBuffer, m_generationManager, &affinityManager)
{
    publicationLogSentinel = new PublicationIntent();
    convo::publishAtomic(publicationLog.head, publicationLogSentinel, std::memory_order_release); // release: commitPublishedState の acquire と HB
    convo::publishAtomic(publicationLog.consumedTail, publicationLogSentinel, std::memory_order_release); // release: drainPublicationLog の acquire と HB
    convo::publishAtomic(publicationLog.retiredHead, publicationLogSentinel, std::memory_order_release); // release: drainPublicationLog の acquire と HB

    uiConvolverProcessor.setRcuProvider(*this);
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
        validateDistinctRuntimeSlots("~AudioEngine.beforeClear",
                         activeDSP,
                         resolveFadingDSPFromRuntimeWorldOnly(getRuntimePublishView().graph),
                         nullptr);

        convo::fetchAddAtomic(rebuildGeneration, 1, std::memory_order_acq_rel); // acq_rel: rebuild observer の acquire と HB

        // Audio Thread から参照される公開ポインタを明示的に外す。
        publishCurrentDSP(nullptr);

        {
            constexpr uintptr_t kInvalidAllOnes = ~static_cast<uintptr_t>(0);
            DSPCore* activeRaw = activeDSP.get();
            activeToRelease = (reinterpret_cast<uintptr_t>(activeRaw) == kInvalidAllOnes) ? nullptr : activeRaw;
        }
        activeDSP = nullptr;
        fadingToRelease = sanitizeRawPtr(exchangeFadingOutDSP(nullptr));

        validateDistinctRuntimeSlots("~AudioEngine.afterClear",
                         activeDSP,
                         resolveFadingDSPFromRuntimeWorldOnly(getRuntimePublishView().graph),
                         nullptr);

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

    drainPublicationLogForShutdown();

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
    m_epochDomain.advanceEpoch();

    // Shutdown 時は EBR 回収を試みる。
    setShutdownPhase(ShutdownPhase::DrainRetire, "~AudioEngine");
    publicationCoordinator().clearPublishedRuntimeSnapshotsNonRt();
    drainDeferredRetireQueues(true);
    m_epochDomain.drainAll();

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

#endif // defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_CTOR_DTOR)
