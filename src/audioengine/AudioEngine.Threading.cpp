#include <JuceHeader.h>
#include "AudioEngine.h"

namespace
{
    constexpr size_t kReclaimBacklogWarnThreshold = 128;
}

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
    g_runtimeReclaimCount.fetch_add(1, std::memory_order_acq_rel);
    g_deletionQueue.reclaim(m_epochCore);
}

#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_THREADING_DEFERRED)

void AudioEngine::drainDeferredRetireQueues(bool allowDuringShutdown) noexcept
{
    if (!allowDuringShutdown && isShutdownInProgress())
        return;

    auto flushAudioThreadRetireOverflow = [this]() noexcept
    {
        auto* overflow = convo::exchangeAtomic(audioThreadRetireOverflowPtr, nullptr, std::memory_order_acq_rel);
        if (overflow == nullptr)
            return;

        const uint64_t overflowEpoch = convo::exchangeAtomic(audioThreadRetireOverflowEpoch, 0, std::memory_order_acq_rel);
        const uint64_t epoch = overflowEpoch != 0 ? overflowEpoch : m_epochCore.current();
        if (!g_deletionQueue.enqueue(
                overflow,
                [](void* p) { std::default_delete<DSPCore>{}(static_cast<DSPCore*>(p)); },
                epoch))
        {
            std::lock_guard<std::mutex> lock(deferredDeleteFallbackMutex);
            deferredDeleteFallbackQueue.push_back(DeferredDeleteFallbackEntry {
                overflow,
                [](void* p) { std::default_delete<DSPCore>{}(static_cast<DSPCore*>(p)); },
                epoch
            });
        }
    };

    auto flushDeferredDeleteFallbackQueue = [this]() noexcept
    {
        std::vector<DeferredDeleteFallbackEntry> pending;
        {
            std::lock_guard<std::mutex> lock(deferredDeleteFallbackMutex);
            if (deferredDeleteFallbackQueue.empty())
                return;
            pending.swap(deferredDeleteFallbackQueue);
        }

        for (auto& entry : pending)
        {
            const uint64_t epoch = entry.epoch != 0 ? entry.epoch : m_epochCore.current();
            if (!g_deletionQueue.enqueue(entry.ptr, entry.deleter, epoch))
            {
                std::lock_guard<std::mutex> lock(deferredDeleteFallbackMutex);
                deferredDeleteFallbackQueue.push_back(entry);
            }
        }
    };

    flushAudioThreadRetireOverflow();
    flushDeferredDeleteFallbackQueue();
    g_deletionQueue.reclaim(m_epochCore);
    m_coordinator.reclaim(m_epochCore);

    const uint64_t dropped = convo::consumeAtomic(audioThreadRetireEnqueueDropped, std::memory_order_acquire);
    if (dropped >= kReclaimBacklogWarnThreshold)
    {
        juce::Logger::writeToLog("[DIAG] deferred reclaim enqueue drops=" + juce::String(static_cast<juce::int64>(dropped)));
    }
}

void AudioEngine::processDeferredReleases()
{
    drainDeferredRetireQueues(false);
}

#endif
