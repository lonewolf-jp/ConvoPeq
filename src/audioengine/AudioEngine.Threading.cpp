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
    // 既存: retire queue pressure チェック
    if (convo::consumeAtomic(retirePressureAdmissionStrict_, std::memory_order_acquire))
        return true;

    // ★ S-2: HealthState Critical の場合も Rebuild を拒否
    if (m_healthMonitor.getHealthState() == convo::ISRHealthState::Critical)
        return true;

    return false;
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
    // ★ detectStuckReaders は1回だけ呼び出し、2つのフィールドで再利用（二重呼出の改善①）
    const auto readerStuckInfo = m_retireRouter
        ? m_retireRouter->detectStuckReaders(10)
        : convo::StuckReaderInfo{};

    return convo::isr::RuntimeDrainAudit{
        .pendingPublication = runtimePublicationBridge_.getPublicationBacklogCount(),
        .pendingRetire = retireRuntime_.pendingIntentCount(),
        .activeCrossfadeCount = crossfadeRuntime_.isPending() ? 1u : 0u,
        .routerPendingRetire = static_cast<uint64_t>(m_retireRouter->pendingRetireCount())
            + convo::consumeAtomic(fallbackQueueDepth_, std::memory_order_acquire),  // ★ P1-9: ring+fallback 合計
        .maxDeferredAgeMs = runtimeOrchestrator_
            ? runtimeOrchestrator_->getMaxDeferredAgeMs() : 0u,
        .deferredPublish = (runtimeOrchestrator_
            && runtimeOrchestrator_->hasDeferredRequest()) ? 1u : 0u,
        .quarantineResident = dspQuarantineManager_.residentCount(),
        .oldestPendingAgeMs = static_cast<uint64_t>(
            std::max(0.0, convo::consumeAtomic(oldestPendingAge_, std::memory_order_acquire))),
        .maxQuarantineAgeSec = dspQuarantineManager_.getMaxEntryAgeSec(),
        // ★ C-1: WorldLifecycleAudit から World カウンタ取得
        .activeWorldCount = worldLifecycleAudit_.activeWorldCount(),
        .publishedCount = worldLifecycleAudit_.publishedCount(),
        .retiredCount = worldLifecycleAudit_.retiredCount(),
        // ★ A-2/A-3: Reader 状態収集（detectStuckReaders は1回のみ）
        .activeReaderCount = m_retireRouter ? m_retireRouter->activeReaderCount() : 0u,
        .stuckReaderCount = readerStuckInfo.isStuck ? 1u : 0u,
        .maxReaderResidencyUs = readerStuckInfo.residencyTimeUs,
        // ★ B-2: HealthState 診断情報
        .healthState = m_healthMonitor.getHealthState(),
        // ★ A-2: EBR Queue Visibility 統計
        .reclaimAttemptCount = m_retireRouter
            ? m_retireRouter->reclaimAttemptCount() : 0,
        .reclaimSuccessCount = m_retireRouter
            ? m_retireRouter->reclaimSuccessCount() : 0,
        .overflowCount = m_retireRouter
            ? m_retireRouter->overflowCount() : 0
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
    // ★ P1-4: waitForDrain は AudioStopped 以降でのみ呼ばれる。
    //   新しい ShutdownPhase が追加された場合はここに追加すること。
    [[maybe_unused]] const auto phase = shutdownRuntime_.getPhase();
    jassert(phase == convo::isr::ShutdownPhase::AudioStopped
         || phase == convo::isr::ShutdownPhase::ObserverDrained
         || phase == convo::isr::ShutdownPhase::RetireClosed
         || phase == convo::isr::ShutdownPhase::EpochSettled
         || phase == convo::isr::ShutdownPhase::ReclaimComplete
         || phase == convo::isr::ShutdownPhase::EmergencyDrain     // ★ C-2
         || phase == convo::isr::ShutdownPhase::TimedOut
         || phase == convo::isr::ShutdownPhase::Failed
         || phase == convo::isr::ShutdownPhase::ShutdownComplete);

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
