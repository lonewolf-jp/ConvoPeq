#include <JuceHeader.h>
#include "AudioEngine.h"

namespace
{
    constexpr size_t kReclaimBacklogWarnThreshold = 128;
}

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_THREADING_PUBLISH)

uint64_t AudioEngine::publishRcuEpoch() noexcept
{
    return currentRetireEpoch();
}

uint64_t AudioEngine::publishRetireEpoch() noexcept
{
    return m_epochDomain.publish();
}

uint64_t AudioEngine::currentRetireEpoch() const noexcept
{
    return m_epochDomain.current();
}

uint64_t AudioEngine::advanceRetireEpoch() noexcept
{
    return m_epochDomain.advanceEpoch();
}

bool AudioEngine::enqueueRetireEpochBounded(void* ptr, void (*deleter)(void*), uint64_t epoch) noexcept
{
    return m_epochDomain.enqueueRetire(ptr, deleter, epoch);
}

uint32_t AudioEngine::activeEpochObserverCount() const noexcept
{
    return m_epochDomain.activeReaderCount();
}

#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_THREADING_ENTER)

void AudioEngine::enterRcuReader(int readerIndex) noexcept
{
    m_epochDomain.enterReader(readerIndex);
}

#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_THREADING_EXIT)

void AudioEngine::exitRcuReader(int readerIndex) noexcept
{
    m_epochDomain.exitReader(readerIndex);
}

#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_THREADING_RECLAIM)

void AudioEngine::tryReclaimResources() noexcept
{
    convo::fetchAddAtomic(g_runtimeReclaimCount, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
    m_epochDomain.reclaimRetired();
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
        const uint64_t epoch = overflowEpoch != 0 ? overflowEpoch : currentRetireEpoch();
        if (!enqueueRetireEpochBounded(
                overflow,
                [](void* p)
                {
                    auto* core = static_cast<DSPCore*>(p);
                    core->~DSPCore();
                    convo::aligned_free(core);
                },
                epoch))
        {
            std::lock_guard<std::mutex> lock(deferredDeleteFallbackMutex);
            deferredDeleteFallbackQueue.push_back(DeferredDeleteFallbackEntry {
                overflow,
                [](void* p)
                {
                    auto* core = static_cast<DSPCore*>(p);
                    core->~DSPCore();
                    convo::aligned_free(core);
                },
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
            const uint64_t epoch = entry.epoch != 0 ? entry.epoch : currentRetireEpoch();
            if (!enqueueRetireEpochBounded(entry.ptr, entry.deleter, epoch))
            {
                std::lock_guard<std::mutex> lock(deferredDeleteFallbackMutex);
                deferredDeleteFallbackQueue.push_back(entry);
            }
        }
    };

    flushAudioThreadRetireOverflow();
    flushDeferredDeleteFallbackQueue();
    m_epochDomain.reclaimRetired();
    m_coordinator.reclaim(m_epochDomain);

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
