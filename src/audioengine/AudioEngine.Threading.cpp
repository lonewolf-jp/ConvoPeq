#include <JuceHeader.h>
#include "AudioEngine.h"

extern std::atomic<bool> gShuttingDown;

namespace
{
    constexpr size_t kDeferredReclaimBudget = 16;
    constexpr size_t kReclaimBacklogWarnThreshold = 128;
}

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
    g_runtimeReclaimCount.fetch_add(1, std::memory_order_relaxed);
    auto& queue = convo::EBRQueue::instance();
    queue.tryReclaim(kDeferredReclaimBudget);
    const size_t backlog = queue.getPendingRetiredCount();
    if (backlog >= kReclaimBacklogWarnThreshold)
    {
        juce::Logger::writeToLog("[DIAG] EBR backlog warning pending=" + juce::String(static_cast<juce::int64>(backlog)));
    }
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
        g_runtimeReclaimCount.fetch_add(1, std::memory_order_relaxed);
        convo::EBRQueue::instance().tryReclaim();
        return;
    }

    g_runtimeReclaimCount.fetch_add(1, std::memory_order_relaxed);
    auto& queue = convo::EBRQueue::instance();
    queue.tryReclaim(kDeferredReclaimBudget);
    const size_t backlog = queue.getPendingRetiredCount();
    if (backlog >= kReclaimBacklogWarnThreshold)
    {
        juce::Logger::writeToLog("[DIAG] deferred reclaim backlog pending=" + juce::String(static_cast<juce::int64>(backlog)));
    }
}

#endif
