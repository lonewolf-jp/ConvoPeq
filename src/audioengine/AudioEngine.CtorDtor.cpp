#include <JuceHeader.h>
#include "AudioEngine.h"
#include "NoiseShaperLearner.h"

extern std::atomic<bool> gShuttingDown;

namespace {
static void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}
}

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_CTOR_DTOR)

AudioEngine::AudioEngine()
    : uiEqEditor(*this)
    , m_coordinator(m_epochCore)
    , m_workerThread(m_commandBuffer, m_coordinator, m_generationManager, &affinityManager)
{
    gShuttingDown.store(false, std::memory_order_release);
    uiConvolverProcessor.setRcuProvider(this);
    // 必要な初期化処理があればここに追加
}

AudioEngine::~AudioEngine()
{
    diagLog("[DIAG] ~AudioEngine: enter");
    shutdownInProgress.store(true, std::memory_order_release);
    gShuttingDown.store(true, std::memory_order_release);
    cancelPendingUpdate();

    // 終了順序を固定化して、終了時フリーズを防ぐ。
    stopTimer();

    // まず rebuild thread 側へ終了を通知し、pending task を破棄して
    // 終了時に重い再構築へ入る経路を閉じる。
    // pending task を破棄して進行中 rebuild を obsolete にし、thread を停止する。
    DSPCore* activeToRelease = nullptr;
    DSPCore* fadingToRelease = nullptr;
    DSPCore* queuedToRelease = nullptr;
    {
        std::lock_guard<std::mutex> lock(rebuildMutex);
        rebuildGeneration.fetch_add(1, std::memory_order_relaxed);

        // Audio Thread から参照される公開ポインタを明示的に外す。
        currentDSP.store(nullptr, std::memory_order_release);

        activeToRelease = sanitizeRawPtr(activeDSP);
        activeDSP = nullptr;
        fadingToRelease = sanitizeRawPtr(fadingOutDSP.exchange(nullptr, std::memory_order_acq_rel));
        queuedToRelease = sanitizeRawPtr(queuedOldDSP.exchange(nullptr, std::memory_order_acq_rel));
        fadeQueued.store(false, std::memory_order_release);

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
    stopRebuildThread();

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
    if (queuedToRelease) retireDSP(queuedToRelease);

    uiConvolverProcessor.removeChangeListener(this);
    uiEqEditor.removeChangeListener(this);
    uiConvolverProcessor.removeListener(this);

    // Snapshot worker を先に停止。
    shutdownWorkerThread();

    // Shutdown 時は EBR 回収を試みる。
    convo::EBRQueue::instance().tryReclaim();

    // ...既存の解放処理...
    if (latencyBufOldL) { _aligned_free(latencyBufOldL); latencyBufOldL = nullptr; }
    if (latencyBufOldR) { _aligned_free(latencyBufOldR); latencyBufOldR = nullptr; }
    if (latencyBufNewL) { _aligned_free(latencyBufNewL); latencyBufNewL = nullptr; }
    if (latencyBufNewR) { _aligned_free(latencyBufNewR); latencyBufNewR = nullptr; }
    latencyBufSize = 0;
    diagLog("[DIAG] ~AudioEngine: exit");
}

#endif // defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_CTOR_DTOR)
