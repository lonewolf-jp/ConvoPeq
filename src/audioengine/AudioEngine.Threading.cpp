#include <JuceHeader.h>
#include "AudioEngine.h"
#include "RuntimePublicationOrchestrator.h"

//==============================================================================
// [P0-15] AudioEngine.Threading.cpp — 3PR分割済み.
// 共通定数は AudioEngine.Retire.cpp に移動.
// 本ファイルには非3PR関数のみ残す.
//==============================================================================

void AudioEngine::destroyDSPCoreNode(void* p) noexcept
{
    auto* core = static_cast<DSPCore*>(p);
    core->~DSPCore();
    convo::aligned_free(core);
}

bool AudioEngine::shouldRejectRebuildAdmissionForPressure() const noexcept
{
    return convo::consumeAtomic(retirePressureAdmissionStrict_, std::memory_order_acquire);
}

bool AudioEngine::isFullyDrained() noexcept
{
    const bool hasDeferredCommit = (runtimeOrchestrator_ != nullptr && runtimeOrchestrator_->hasDeferredRequest());
    runtimePublicationBridge_.setPendingIntentCount(hasDeferredCommit ? 1u : 0u);
    runtimePublicationBridge_.setPublicationBacklogCount(hasDeferredCommit ? 1u : 0u);

    const std::uint64_t fallbackDepth = convo::consumeAtomic(fallbackQueueDepth_, std::memory_order_acquire);
    const std::uint64_t retireDepth = convo::consumeAtomic(retireQueueDepth_, std::memory_order_acquire);
    runtimePublicationBridge_.setFallbackBacklogCount(fallbackDepth);
    runtimePublicationBridge_.setRetireBacklogCount(retireDepth);
    runtimePublicationBridge_.setDeferredRetireResidencyCount(fallbackDepth);

    return !hasDeferredCommit && runtimePublicationBridge_.isFullyDrained();
}

bool AudioEngine::waitForDrain(int timeoutMs, int pollIntervalMs) noexcept
{
    ASSERT_NON_RT_THREAD();

    const int boundedTimeoutMs = juce::jlimit(1, 10000, timeoutMs);
    const int boundedPollIntervalMs = juce::jlimit(1, 5, pollIntervalMs);

    const double startMs = juce::Time::getMillisecondCounterHiRes();
    while (!isFullyDrained())
    {
        drainDeferredRetireQueues(true);

        const double elapsedMs = juce::Time::getMillisecondCounterHiRes() - startMs;
        if (elapsedMs >= static_cast<double>(boundedTimeoutMs))
            return false;

        juce::Thread::sleep(boundedPollIntervalMs);
    }

    return true;
}

void AudioEngine::processDeferredReleases()
{
    drainDeferredRetireQueues(false);
}
