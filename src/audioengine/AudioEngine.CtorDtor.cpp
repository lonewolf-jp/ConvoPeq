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
    , m_coordinator(m_epochCore)
    , m_workerThread(m_commandBuffer, m_coordinator, m_generationManager, &affinityManager)
{
    uiConvolverProcessor.setRcuProvider(*this);
    // 必要な初期化処理があればここに追加
}

AudioEngine::~AudioEngine()
{
    diagLog("[DIAG] ~AudioEngine: enter");
    setShutdownPhase(ShutdownPhase::StopAcceptingWork, "~AudioEngine");
    convo::publishAtomic(lifecycleState, EngineLifecycleState::Releasing, std::memory_order_release);
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
                         sanitizeRawPtr(loadFadingOutDSP()),
                         nullptr);

        rebuildGeneration.fetch_add(1, std::memory_order_acq_rel);

        // Audio Thread から参照される公開ポインタを明示的に外す。
        publishCurrentDSP(nullptr);

        activeToRelease = sanitizeRawPtr(activeDSP);
        activeDSP = nullptr;
        fadingToRelease = sanitizeRawPtr(exchangeFadingOutDSP(nullptr));

        validateDistinctRuntimeSlots("~AudioEngine.afterClear",
                         activeDSP,
                         sanitizeRawPtr(loadFadingOutDSP()),
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

    {
        std::queue<CommitStaging> abandonedCommits;
        std::lock_guard<std::mutex> lock(deferredCommitMutex);
        std::swap(abandonedCommits, deferredCommitQueue);

        while (!abandonedCommits.empty())
        {
            auto staging = abandonedCommits.front();
            abandonedCommits.pop();

            if (staging.newDSP)
                retireDSP(staging.newDSP);
            if (staging.oldDSP)
                retireDSP(staging.oldDSP);
        }
    }

    if (activeToRelease) retireDSP(activeToRelease);
    if (fadingToRelease) retireDSP(fadingToRelease);

    uiConvolverProcessor.removeChangeListener(this);
    uiEqEditor.removeChangeListener(this);
    uiConvolverProcessor.removeListener(this);

    // Note: stopRebuildThread は releaseResources() で呼ばれる。
    // dtor が releaseResources 経由で呼ばれる場合、stopRebuildThread は既に完了している。
    // dtor が直接呼ばれる場合（例：ホストが releaseResources を呼ばない異常系）、
    // rebuildThreadShouldExit が既に true なので thread ループは速やかに終了する。

    // Snapshot worker を停止。
    shutdownWorkerThread();

    setShutdownPhase(ShutdownPhase::ForceEpochAdvance, "~AudioEngine");
    convo::EpochManager::instance().advanceEpoch();
    convo::publishAtomic(g_currentEpoch, convo::EpochManager::instance().currentEpoch(), std::memory_order_release);

    // Shutdown 時は EBR 回収を試みる。
    setShutdownPhase(ShutdownPhase::DrainRetire, "~AudioEngine");
    clearPublishedRuntimeSnapshotsNonRt();
    drainDeferredRetireQueues(true);

    // ...既存の解放処理...
    if (latencyBufOldL) { _aligned_free(latencyBufOldL); latencyBufOldL = nullptr; }
    if (latencyBufOldR) { _aligned_free(latencyBufOldR); latencyBufOldR = nullptr; }
    if (latencyBufNewL) { _aligned_free(latencyBufNewL); latencyBufNewL = nullptr; }
    if (latencyBufNewR) { _aligned_free(latencyBufNewR); latencyBufNewR = nullptr; }
    latencyBufSize = 0;
    setShutdownPhase(ShutdownPhase::Destroy, "~AudioEngine");
    convo::publishAtomic(lifecycleState, EngineLifecycleState::Destroyed, std::memory_order_release);
    diagLog("[DIAG] ~AudioEngine: shutdown sequence complete exit");
}

#endif // defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_CTOR_DTOR)
