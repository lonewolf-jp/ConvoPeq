#include <JuceHeader.h>
#include <algorithm>
#include "AudioEngine.h"
#include "ISRDSPQuarantine.h"
#include "RuntimeDrainAudit.h"
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

// ★ A-1.6: 3系統の隔離を1トランザクションとして実行（1 truth + 2 projections）
bool AudioEngine::quarantineSlot(uint32_t slot, uint64_t generation,
                                  convo::isr::QuarantineReason reason) noexcept
{
    ASSERT_NON_RT_THREAD();

    // Step 1: Truth store 更新（唯一の隔離判定）
    const bool applied = dspQuarantineManager_.quarantineHandle(slot, generation, reason);

    // Step 2: Truth 確認（既に隔離済みの場合は何もしない）
    if (!applied)
        return false;

    // Step 3: Projection 更新（truth を反映）
    dspHandleRuntime_.quarantineSlot(slot);
    retireRuntimeEx_.quarantine(slot);

    return true;
}

// ★ A-2.5: collectDrainAudit — shutdown 完了条件の監査構造体を収集
convo::isr::RuntimeDrainAudit AudioEngine::collectDrainAudit() noexcept
{
    return convo::isr::RuntimeDrainAudit{
        .pendingPublication = runtimePublicationBridge_.getPublicationBacklogCount(),
        .pendingRetire = retireRuntime_.pendingIntentCount(),
        .activeCrossfadeCount = crossfadeRuntime_.isPending() ? 1u : 0u,
        .routerPendingRetire = 0u,  // ★ B-2: m_retireRouter->pendingRetireCount() 追加予定
        .maxDeferredAgeMs = runtimeOrchestrator_
            ? runtimeOrchestrator_->getMaxDeferredAgeMs() : 0u,
        .deferredPublish = (runtimeOrchestrator_
            && runtimeOrchestrator_->hasDeferredRequest()) ? 1u : 0u,
        .quarantineResident = dspQuarantineManager_.residentCount(),
        .oldestPendingAgeMs = static_cast<uint64_t>(
            std::max(0.0, convo::consumeAtomic(oldestPendingAge_, std::memory_order_acquire))),
        .maxQuarantineAgeSec = dspQuarantineManager_.getMaxEntryAgeSec()
    };
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
