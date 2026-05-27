#include <JuceHeader.h>
#include "AudioEngine.h"

namespace
{
    constexpr size_t kReclaimBacklogWarnThreshold = 128;
}

void AudioEngine::destroyDSPCoreNode(void* p) noexcept
{
    auto* core = static_cast<DSPCore*>(p);
    core->~DSPCore();
    convo::aligned_free(core);
}

[[nodiscard]] uint64_t AudioEngine::publishRcuEpoch() noexcept
{
    return currentRetireEpoch();
}

[[nodiscard]] uint64_t AudioEngine::publishRetireEpoch() noexcept
{
    return m_epochDomain.publish();
}

[[nodiscard]] uint64_t AudioEngine::currentRetireEpoch() const noexcept
{
    return m_epochDomain.current();
}

uint64_t AudioEngine::advanceRetireEpoch() noexcept
{
    return m_epochDomain.advanceEpoch();
}

[[nodiscard]] bool AudioEngine::enqueueRetireEpochBounded(void* ptr, void (*deleter)(void*), uint64_t epoch) noexcept
{
    return m_epochDomain.enqueueRetire(ptr, deleter, epoch);
}

[[nodiscard]] uint32_t AudioEngine::activeEpochObserverCount() const noexcept
{
    return m_epochDomain.activeReaderCount();
}

void AudioEngine::enterRcuReader(int readerIndex) noexcept
{
    m_epochDomain.enterReader(readerIndex);
}

void AudioEngine::exitRcuReader(int readerIndex) noexcept
{
    m_epochDomain.exitReader(readerIndex);
}

void AudioEngine::tryReclaimResources() noexcept
{
    convo::fetchAddAtomic(rtAuxMutable_.runtimeReclaimCount, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
    m_epochDomain.reclaimRetired();
}

void AudioEngine::drainDeferredRetireQueues(bool allowDuringShutdown) noexcept
{
    if (!allowDuringShutdown && isShutdownInProgress())
        return;

    auto flushAudioThreadRetireOverflow = [this]() noexcept
    {
        auto* overflow = convo::exchangeAtomic(audioThreadRetireOverflowPtr, nullptr, std::memory_order_acq_rel);
        if (overflow == nullptr)
            return;

        const uint64_t overflowEpoch = convo::exchangeAtomic(rtLocalState_.audioThreadRetireOverflowEpoch, 0, std::memory_order_acq_rel);
        const uint64_t epoch = overflowEpoch != 0 ? overflowEpoch : currentRetireEpoch();
        if (!enqueueRetireEpochBounded(
                overflow,
                &AudioEngine::destroyDSPCoreNode,
                epoch))
        {
            std::lock_guard<std::mutex> lock(deferredDeleteFallbackMutex);
            deferredDeleteFallbackQueue.push_back(DeferredDeleteFallbackEntry {
                overflow,
                &AudioEngine::destroyDSPCoreNode,
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

    const uint64_t dropped = convo::consumeAtomic(rtLocalState_.audioThreadRetireEnqueueDropped, std::memory_order_acquire);
    if (dropped >= kReclaimBacklogWarnThreshold)
    {
        juce::Logger::writeToLog("[DIAG] deferred reclaim enqueue drops=" + juce::String(static_cast<juce::int64>(dropped)));
    }
}

void AudioEngine::processDeferredReleases()
{
    drainDeferredRetireQueues(false);
}
