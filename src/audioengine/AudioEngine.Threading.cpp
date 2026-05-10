#include <JuceHeader.h>
#include "AudioEngine.h"

extern std::atomic<bool> gShuttingDown;

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_THREADING_ADVANCE)

void AudioEngine::advanceRcuEpoch() noexcept
{
    convo::EpochManager::instance().advanceEpoch();
}

#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_THREADING_PUBLISH)

uint64_t AudioEngine::publishRcuEpoch() noexcept
{
    return convo::EpochManager::instance().currentEpoch();
}

#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_THREADING_ENTER)

void AudioEngine::enterRcuReader(int readerIndex) noexcept
{
    convo::EpochManager::instance().enter(readerIndex);
}

#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_THREADING_EXIT)

void AudioEngine::exitRcuReader(int readerIndex) noexcept
{
    convo::EpochManager::instance().exit(readerIndex);
}

#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_THREADING_RECLAIM)

void AudioEngine::tryReclaimResources() noexcept
{
    convo::EBRQueue::instance().tryReclaim();
}

#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_THREADING_DEFERRED)

void AudioEngine::processDeferredReleases()
{
    if (shutdownInProgress.load(std::memory_order_acquire))
        return;

    if (gShuttingDown.load(std::memory_order_acquire))
    {
        // For simple EBR in minimal config, just try one last reclaim or rely on cleanup at termination
        convo::EBRQueue::instance().tryReclaim();
        return;
    }

    convo::EBRQueue::instance().tryReclaim();
}

#endif
