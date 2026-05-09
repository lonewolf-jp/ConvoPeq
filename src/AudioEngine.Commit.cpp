#include <JuceHeader.h>
#include "AudioEngine.h"

extern std::atomic<bool> gShuttingDown;

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_COMMIT_PREPARE)

namespace
{
static void retireDSP(AudioEngine::DSPCore* dsp)
{
    if (dsp) convo::retireObject(dsp, [](void* p) { delete static_cast<AudioEngine::DSPCore*>(p); });
}
}

void AudioEngine::prepareCommit(DSPCore* newDSP, int generation)
{
    if (newDSP == nullptr)
        return;

    if (shutdownInProgress.load(std::memory_order_acquire) ||
        gShuttingDown.load(std::memory_order_acquire))
    {
        retireDSP(newDSP);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(deferredCommitMutex);

        if (shutdownInProgress.load(std::memory_order_acquire) ||
            gShuttingDown.load(std::memory_order_acquire))
        {
            retireDSP(newDSP);
            return;
        }

        deferredCommitQueue.push(CommitStaging { newDSP, nullptr, generation });
    }

    triggerAsyncUpdate();
}

#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_COMMIT_EXECUTE)
void AudioEngine::executeCommit()
{
    std::queue<CommitStaging> localQueue;

    {
        std::lock_guard<std::mutex> lock(deferredCommitMutex);
        std::swap(localQueue, deferredCommitQueue);
    }

    while (!localQueue.empty())
    {
        auto staging = localQueue.front();
        localQueue.pop();

        if (staging.newDSP == nullptr)
            continue;

        if (shutdownInProgress.load(std::memory_order_acquire) ||
            gShuttingDown.load(std::memory_order_acquire))
        {
            retireDSP(staging.newDSP);
            continue;
        }

        commitNewDSP(staging.newDSP, staging.generation);
    }
}
#endif
