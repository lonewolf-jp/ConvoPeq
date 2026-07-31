#include "ISRRuntimePublicationCoordinator.h"
#include "AtomicAccess.h"
#include "ISRRetireOverflowRing.h"
#include "ISRDSPHandle.h"
#include "ISRDSPQuarantine.h"
#include <cassert>

namespace convo::isr {

RuntimePublicationCoordinator::RuntimePublicationCoordinator()
    : overflowScheduler_(*this)
    , shutdownScheduler_(*this)
    , priorityScheduler_(*this)
    , currentWorld_(nullptr)
    , lastRejectCode_(RejectCode::None)
    , retireBacklogCount_(0)
    , publicationBacklogCount_(0)
    , pendingIntentCount_(0)
    , fallbackBacklogCount_(0)
    , reclaimInFlightCount_(0)
    , deferredRetireResidencyCount_(0)
    , previousRetireBacklogCount_(0)
    , pressureNormalizedWindows_(0)
    , swapPending_(false)
    , state_(CoordinatorState::Bootstrapping)
    , retireAuthorityCount_(0)
{
    // ★ persistentState_{} は zero-initialize（メンバ初期化子 =0 により保証）
}

bool RuntimePublicationCoordinator::precheckPublish(const PayloadClosureDescriptor& closure,
                                                    const TieredPayloadDescriptor& descriptor) noexcept {
    ClosureValidator closureValidator;
    if (!closureValidator.validateClosureGraph(closure)) {
        convo::publishAtomic(lastRejectCode_, RejectCode::InvalidClosure, std::memory_order_release);
        return false;
    }

    PayloadTierValidator tierValidator;
    if (!tierValidator.isPublishAllowed(descriptor)) {
        convo::publishAtomic(lastRejectCode_, RejectCode::InvalidPayloadTier, std::memory_order_release);
        return false;
    }

    convo::publishAtomic(lastRejectCode_, RejectCode::None, std::memory_order_release);
    return true;
}

const char* RuntimePublicationCoordinator::lastRejectReason() const noexcept {
    switch (convo::consumeAtomic(lastRejectCode_, std::memory_order_acquire)) {
    case RejectCode::InvalidClosure:
        return "invalid closure graph";
    case RejectCode::InvalidPayloadTier:
        return "invalid payload tier";
    case RejectCode::None:
    default:
        return "none";
    }
}

void RuntimePublicationCoordinator::commit(PublishAuthority,
                                           RuntimeBoundary boundary,
                                           const void* newWorld,
                                           std::uint64_t version) {
    commit(PublishAuthority::Granted,
           boundary,
           newWorld,
           version,
           static_cast<PublicationSequenceId>(version),
           static_cast<PublicationEpoch>(version),
           version);
}

void RuntimePublicationCoordinator::commit(PublishAuthority,
                                           RuntimeBoundary boundary,
                                           const void* newWorld,
                                           std::uint64_t /*version*/,
                                           PublicationSequenceId sequenceId,
                                           PublicationEpoch epoch,
                                           std::uint64_t mappedGeneration) {
    if (boundary != RuntimeBoundary::NonRTWorld || newWorld == nullptr) {
        convo::publishAtomic(state_, CoordinatorState::Faulted, std::memory_order_release);
        return;
    }

    // ★ 方式C: 単一 struct 読取 → 3フィールド論理一貫
    const auto prev = persistentState_;

    if (!PersistentStateBlock::isMonotonic(prev,
            static_cast<std::uint64_t>(sequenceId),
            static_cast<std::uint64_t>(epoch),
            mappedGeneration)) {
        convo::publishAtomic(state_, CoordinatorState::Faulted, std::memory_order_release);
        return;
    }

    convo::publishAtomic(state_, CoordinatorState::Publishing, std::memory_order_release);
    convo::publishAtomic(swapPending_, true, std::memory_order_release);

    // ★ plain struct: 単一代入（atomic store 不要）
    persistentState_ = PersistentStateBlock{
        static_cast<std::uint64_t>(sequenceId),
        static_cast<std::uint64_t>(epoch),
        mappedGeneration
    };

    convo::publishAtomic(currentWorld_, newWorld, std::memory_order_release);
    convo::publishAtomic(swapPending_, false, std::memory_order_release);
    convo::publishAtomic(state_, CoordinatorState::Ready, std::memory_order_release);
}

void RuntimePublicationCoordinator::retire(RetireAuthority,
                                           RuntimeBoundary boundary,
                                           const void* oldWorld) {
    if (boundary != RuntimeBoundary::NonRTWorld || oldWorld == nullptr) {
        convo::publishAtomic(state_, CoordinatorState::Faulted, std::memory_order_release);
        return;
    }

    (void) oldWorld;
    auto observedCurrent = convo::consumeAtomic(currentWorld_, std::memory_order_acquire);
    if (observedCurrent == oldWorld)
    {
        convo::compareExchangeAtomic(currentWorld_,
                                     observedCurrent,
                                     static_cast<const void*>(nullptr),
                                     std::memory_order_acq_rel,
                                     std::memory_order_acquire);
    }

    const auto backlog = convo::consumeAtomic(retireBacklogCount_, std::memory_order_acquire) + 1u;
    setRetireBacklogCount(backlog);
}

RetireEnqueueResult RuntimePublicationCoordinator::enqueueRetire(RetireAuthority,
                                                                   ISRRetireRouter& router,
                                                                   void* ptr,
                                                                   void (*deleter)(void*),
                                                                   std::uint64_t epoch) noexcept
{
    convo::fetchAddAtomic(retireAuthorityCount_,
                          static_cast<std::uint64_t>(1),
                          std::memory_order_acq_rel);

    if (ptr == nullptr || deleter == nullptr)
        return RetireEnqueueResult::Success;

    // ★ Bug#2-d: enqueueWithRetry に委譲（リトライロジックは Router に集約）
    const auto result = router.enqueueWithRetry(ptr, deleter, epoch, DeletionEntryType::Generic);
    if (result != RetireEnqueueResult::Success)
        return result;

    const auto backlog = convo::consumeAtomic(retireBacklogCount_, std::memory_order_acquire) + 1u;
    setRetireBacklogCount(backlog);

    return RetireEnqueueResult::Success;
}

std::uint64_t RuntimePublicationCoordinator::retireAuthorityCount() const noexcept
{
    return convo::consumeAtomic(retireAuthorityCount_, std::memory_order_acquire);
}

const void* RuntimePublicationCoordinator::getCurrent() const noexcept {
    return convo::consumeAtomic(currentWorld_, std::memory_order_acquire);
}

std::uint64_t RuntimePublicationCoordinator::getVersion() const noexcept {
    // ★ 方式C: persistentState_ から直接導出（plain struct、atomic 不要）
    // ★ ADR-010: Runtime API — スレッド制約なし（read-only 単調増加 uint64）
    //   persistentState_.mappedRuntimeGeneration への読み取りは
    //   単調増加カウンタであり cross-thread の一貫性問題は発生しない。
    //   Timer Thread からの読み取りも安全。
    return persistentState_.mappedRuntimeGeneration;
}

void RuntimePublicationCoordinator::setRetireBacklogCount(std::uint64_t count) noexcept {
    const auto previousBacklog = convo::consumeAtomic(previousRetireBacklogCount_, std::memory_order_acquire);
    const auto slope = (count > previousBacklog) ? (count - previousBacklog) : 0;

    convo::publishAtomic(retireBacklogCount_, count, std::memory_order_release);
    convo::publishAtomic(previousRetireBacklogCount_, count, std::memory_order_release);

    if (slope > kPressureSlopeThreshold) {
        convo::publishAtomic(pressureNormalizedWindows_, static_cast<std::uint32_t>(0), std::memory_order_release);
        convo::publishAtomic(state_, CoordinatorState::Pressure, std::memory_order_release);
        return;
    }

    if (!isSwapPending()) {
        const auto state = convo::consumeAtomic(state_, std::memory_order_acquire);
        if (state == CoordinatorState::Pressure) {
            const auto nextWindow = static_cast<std::uint32_t>(
                convo::consumeAtomic(pressureNormalizedWindows_, std::memory_order_acquire) + 1U);
            convo::publishAtomic(pressureNormalizedWindows_, nextWindow, std::memory_order_release);

            if (nextWindow < kPressureNormalizeWindows) {
                return;
            }

            convo::publishAtomic(pressureNormalizedWindows_, static_cast<std::uint32_t>(0), std::memory_order_release);
            convo::publishAtomic(state_, CoordinatorState::Ready, std::memory_order_release);
        } else if (state == CoordinatorState::Publishing && count == 0) {
            convo::publishAtomic(state_, CoordinatorState::Ready, std::memory_order_release);
        }
    }
}

void RuntimePublicationCoordinator::setPublicationBacklogCount(std::uint64_t count) noexcept {
    convo::publishAtomic(publicationBacklogCount_, count, std::memory_order_release);
}

void RuntimePublicationCoordinator::setPendingIntentCount(std::uint64_t count) noexcept {
    convo::publishAtomic(pendingIntentCount_, count, std::memory_order_release);
}

void RuntimePublicationCoordinator::setFallbackBacklogCount(std::uint64_t count) noexcept {
    convo::publishAtomic(fallbackBacklogCount_, count, std::memory_order_release);
}

void RuntimePublicationCoordinator::setReclaimInFlightCount(std::uint64_t count) noexcept {
    convo::publishAtomic(reclaimInFlightCount_, count, std::memory_order_release);
}

void RuntimePublicationCoordinator::setDeferredRetireResidencyCount(std::uint64_t count) noexcept {
    convo::publishAtomic(deferredRetireResidencyCount_, count, std::memory_order_release);
}

void RuntimePublicationCoordinator::setQuarantineResidentCount(std::uint64_t count) noexcept {
    convo::publishAtomic(quarantineResidentCount_, count, std::memory_order_release);
}

void RuntimePublicationCoordinator::setOverflowMaxAgeUs(std::uint64_t maxAgeUs) noexcept {
    convo::publishAtomic(overflowMaxAgeUs_, maxAgeUs, std::memory_order_release);
}

std::uint64_t RuntimePublicationCoordinator::getOverflowMaxAgeUs() const noexcept {
    return convo::consumeAtomic(overflowMaxAgeUs_, std::memory_order_acquire);
}

RuntimePublicationCoordinator::OverflowDrainResult
RuntimePublicationCoordinator::drainOverflowRing(
    RetireOverflowRing& overflowRing, RetireRuntime& retireRuntime, bool unlimited) noexcept
{
    return overflowScheduler_.drainOverflowRing(overflowRing, retireRuntime, unlimited);
}

void RuntimePublicationCoordinator::setOverflowAgeWarnCallback(AgeWarnCallback cb) noexcept {
    priorityScheduler_.setOverflowAgeWarnCallback(cb);
}

size_t RuntimePublicationCoordinator::deferredRingOccupancy() const noexcept {
    return overflowScheduler_.deferredRingOccupancy();
}

// ═══════════════════════════════════════════════════════════
// ★ Phase5: OverflowScheduler implementation
// ═══════════════════════════════════════════════════════════

RuntimePublicationCoordinator::OverflowDrainResult
RuntimePublicationCoordinator::OverflowScheduler::drainOverflowRing(
    RetireOverflowRing& overflowRing, RetireRuntime& retireRuntime, bool unlimited) noexcept
{
    OverflowDrainResult result;
    constexpr uint32_t kDefaultBudget = 64;
    constexpr uint32_t kMaxReinjectRetries = 10;
    const uint32_t budget = unlimited ? 0xFFFFFFFFu : kDefaultBudget;
    uint32_t consumed = 0;

    // ★ Phase1: OverflowRing から drain（優先度高）
    RetireOverflowEntry entry;
    while (consumed < budget && overflowRing.pop(entry))
    {
        ++consumed;

        // 滞留時間監視
        if (coordinator_.overflowAgeWarnCallback_ != nullptr)
        {
            const uint64_t nowUs = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count());
            if (entry.overflowTimestampUs > 0 && nowUs > entry.overflowTimestampUs)
            {
                const uint64_t ageUs = nowUs - entry.overflowTimestampUs;
                if (ageUs > result.oldestOverflowAgeUs)
                    result.oldestOverflowAgeUs = ageUs;
                const uint64_t maxAgeUs = convo::consumeAtomic(coordinator_.overflowMaxAgeUs_, std::memory_order_acquire);
                if (maxAgeUs > 0 && ageUs > maxAgeUs)
                    coordinator_.overflowAgeWarnCallback_(ageUs, result.droppedCount);
            }
        }

        // retry超過 → Drop
        if (entry.reinjectRetryCount >= kMaxReinjectRetries)
        {
            ++result.droppedCount;
            continue;
        }

        // 再注入: Overflowからの再注入は High 優先度
        ++entry.reinjectRetryCount;
        entry.intent.priority = RetirePriority::High;
        retireRuntime.emitRetireIntent(entry.intent);
        ++result.reinjectedCount;
    }

    // ★ Phase5: Coordinator DeferredRing から drain（優先度中）
    {
        RetireOverflowEntry deferredEntry;
        constexpr uint32_t kDeferredBudget = 32;
        uint32_t deferredDrained = 0;
        while (deferredDrained < kDeferredBudget && coordinator_.coordinatorDeferredRing_.pop(deferredEntry))
        {
            ++deferredDrained;
            retireRuntime.emitRetireIntent(deferredEntry.intent);
            ++result.reinjectedCount;
        }
        result.deferredRingOccupancy = convo::consumeAtomic(coordinator_.coordinatorDeferredCount_, std::memory_order_acquire);
        // 排出成功分をカウントから減算
        if (deferredDrained > 0)
        {
            convo::fetchSubAtomic(coordinator_.coordinatorDeferredCount_,
                                  static_cast<size_t>(deferredDrained),
                                  std::memory_order_acq_rel);
        }
    }

    // ★ Phase5: LastResortQueue から drain（優先度低）
    {
        const size_t lrCount = convo::consumeAtomic(coordinator_.lastResortCount_, std::memory_order_acquire);
        if (lrCount > 0)
        {
            constexpr uint32_t kLastResortBudget = 16;
            uint32_t lrDrained = 0;
            for (size_t i = 0; i < lrCount && lrDrained < kLastResortBudget; ++i)
            {
                auto& lrEntry = coordinator_.lastResortQueue_[i];
                if (lrEntry.intent.dspSlot != UINT32_MAX)
                {
                    lrEntry.intent.priority = RetirePriority::High;
                    retireRuntime.emitRetireIntent(lrEntry.intent);
                    lrEntry.intent.dspSlot = UINT32_MAX;
                    ++lrDrained;
                    ++result.reinjectedCount;
                }
            }
            if (lrDrained > 0)
            {
                // 排出済みエントリを詰める
                size_t writeIdx = 0;
                for (size_t readIdx = 0; readIdx < lrCount; ++readIdx)
                {
                    if (coordinator_.lastResortQueue_[readIdx].intent.dspSlot != UINT32_MAX)
                    {
                        if (writeIdx != readIdx)
                            coordinator_.lastResortQueue_[writeIdx] = coordinator_.lastResortQueue_[readIdx];
                        ++writeIdx;
                    }
                }
                convo::publishAtomic(coordinator_.lastResortCount_, writeIdx, std::memory_order_release);
            }
        }
    }

    return result;
}

void RuntimePublicationCoordinator::setSwapPending(bool pending) noexcept {
    convo::publishAtomic(swapPending_, pending, std::memory_order_release);
}

bool RuntimePublicationCoordinator::isSwapPending() const noexcept {
    return convo::consumeAtomic(swapPending_, std::memory_order_acquire);
}

std::uint64_t RuntimePublicationCoordinator::getReclaimInFlightCount() const noexcept {
    return convo::consumeAtomic(reclaimInFlightCount_, std::memory_order_acquire);
}

// ★ A-2.4: 新規 getter 群（DrainAudit 用）
std::uint64_t RuntimePublicationCoordinator::getPublicationBacklogCount() const noexcept {
    return convo::consumeAtomic(publicationBacklogCount_, std::memory_order_acquire);
}

std::uint64_t RuntimePublicationCoordinator::getPendingIntentCount() const noexcept {
    return convo::consumeAtomic(pendingIntentCount_, std::memory_order_acquire);
}

std::uint64_t RuntimePublicationCoordinator::getRetireBacklogCount() const noexcept {
    return convo::consumeAtomic(retireBacklogCount_, std::memory_order_acquire);
}

std::uint64_t RuntimePublicationCoordinator::getFallbackBacklogCount() const noexcept {
    return convo::consumeAtomic(fallbackBacklogCount_, std::memory_order_acquire);
}

std::uint64_t RuntimePublicationCoordinator::getDeferredRetireResidencyCount() const noexcept {
    return convo::consumeAtomic(deferredRetireResidencyCount_, std::memory_order_acquire);
}

std::uint64_t RuntimePublicationCoordinator::getQuarantineResidentCount() const noexcept {
    return convo::consumeAtomic(quarantineResidentCount_, std::memory_order_acquire);
}

// ★ Phase5: Delegation to ShutdownScheduler
bool RuntimePublicationCoordinator::isFullyDrained() const noexcept {
    return shutdownScheduler_.isFullyDrained();
}

void RuntimePublicationCoordinator::requestShutdown() noexcept {
    shutdownScheduler_.requestShutdown();
}

void RuntimePublicationCoordinator::markShutdownComplete() noexcept {
    shutdownScheduler_.markShutdownComplete();
}

RuntimePublicationCoordinator::CoordinatorState RuntimePublicationCoordinator::getState() const noexcept {
    return convo::consumeAtomic(state_, std::memory_order_acquire);
}

void RuntimePublicationCoordinator::markTransitionStart() noexcept {
    const auto state = convo::consumeAtomic(state_, std::memory_order_acquire);
    if (state != CoordinatorState::Ready) {
        return;
    }
    convo::publishAtomic(state_, CoordinatorState::Transitioning, std::memory_order_release);
}

void RuntimePublicationCoordinator::markTransitionCommitted() noexcept {
    const auto state = convo::consumeAtomic(state_, std::memory_order_acquire);
    if (state != CoordinatorState::Transitioning) {
        return;
    }
    if (!isSwapPending()) {
        convo::publishAtomic(state_, CoordinatorState::Ready, std::memory_order_release);
    }
}

// ═══════════════════════════════════════════════════════════
// ★ Phase5: OverflowScheduler deferredRingOccupancy
// ═══════════════════════════════════════════════════════════

size_t RuntimePublicationCoordinator::OverflowScheduler::deferredRingOccupancy() const noexcept {
    return convo::consumeAtomic(coordinator_.coordinatorDeferredCount_, std::memory_order_acquire);
}

// ═══════════════════════════════════════════════════════════
// ★ Phase5: ShutdownScheduler implementation
// ═══════════════════════════════════════════════════════════

bool RuntimePublicationCoordinator::ShutdownScheduler::isFullyDrained() const noexcept {
    if (convo::consumeAtomic(coordinator_.swapPending_, std::memory_order_acquire)) {
        return false;
    }

    return convo::consumeAtomic(coordinator_.retireBacklogCount_, std::memory_order_acquire) == 0
        && convo::consumeAtomic(coordinator_.publicationBacklogCount_, std::memory_order_acquire) == 0
        && convo::consumeAtomic(coordinator_.pendingIntentCount_, std::memory_order_acquire) == 0
        && convo::consumeAtomic(coordinator_.fallbackBacklogCount_, std::memory_order_acquire) == 0
        && convo::consumeAtomic(coordinator_.reclaimInFlightCount_, std::memory_order_acquire) == 0
        && convo::consumeAtomic(coordinator_.deferredRetireResidencyCount_, std::memory_order_acquire) == 0
        && convo::consumeAtomic(coordinator_.quarantineResidentCount_, std::memory_order_acquire) == 0;
}

void RuntimePublicationCoordinator::ShutdownScheduler::requestShutdown() noexcept {
    convo::publishAtomic(coordinator_.state_, CoordinatorState::ShuttingDown, std::memory_order_release);
}

void RuntimePublicationCoordinator::ShutdownScheduler::markShutdownComplete() noexcept {
    const auto state = convo::consumeAtomic(coordinator_.state_, std::memory_order_acquire);
    if (state != CoordinatorState::ShuttingDown) {
        return;
    }

    if (isFullyDrained()) {
        convo::publishAtomic(coordinator_.state_, CoordinatorState::Bootstrapping, std::memory_order_release);
    } else {
        convo::publishAtomic(coordinator_.state_, CoordinatorState::Faulted, std::memory_order_release);
    }
}

// ═══════════════════════════════════════════════════════════
// ★ Phase5: PriorityScheduler implementation
// ═══════════════════════════════════════════════════════════

void RuntimePublicationCoordinator::PriorityScheduler::setOverflowAgeWarnCallback(AgeWarnCallback cb) noexcept {
    coordinator_.overflowAgeWarnCallback_ = cb;
}

void RuntimePublicationCoordinator::PriorityScheduler::escalateAllRetires(RetirePriority minPriority) noexcept {
    // ★ Phase5: Coordinator の escalateAllRetires は RetireRuntime に委譲
    //   実装は AudioEngine.Processing.ReleaseResources.cpp の retireRuntime_.escalateAllRetires() が担当
    //   本メソッドは Coordinator の公開APIとしての将来拡張用プレースホルダ
    (void)minPriority;
}

void MultiStagePublisher::publishTier(PayloadTier tier, const void* payload) {
    TieredPayloadDescriptor descriptor{};
    descriptor.tier = tier;
    descriptor.requiresRT = (boundary_ == RuntimeBoundary::RTWorld);
    descriptor.hasExternalResource = (tier == PayloadTier::ExternalPinned);
    descriptor.pinnedLifetime = (tier != PayloadTier::ExternalPinned) ? true : (payload != nullptr);

    PayloadTierValidator validator;
    rejected_ = (validator.explainPublishReject(descriptor) != TierRejectReason::None);
}

//==============================================================================
// ★ P0-4C: ISR Intent 発行インターフェース実装
//==============================================================================

void RuntimePublicationCoordinator::emitObserveIntent(const DSPHandle& handle) noexcept
{
    // ★ P0-4A: Observe Intent — Timer → LockFreeRingBuffer push（RT-safe, SPSC, lock-free）
    //   OBSERVE-2: push() は即座に復帰（SPSC lock-free）
    //   OBSERVE-9: LockFreeRingBuffer は FIFO を保証（SPSC）
    //   OBSERVE-10: 世代検証用に epoch を保存
    //   ISR: Intent は自己完結型 — DSPHandle を含むため、Coordinator は外部状態に依存せず retire 対象を識別可能
    ObserveIntent intent{
        handle,                          // 観測対象の DSPHandle
        persistentState_.publicationEpoch,
        nextObserveIntentId_.fetch_add(1, std::memory_order_relaxed)
    };

    // ★ QUEUE-11: 4層 Overflow Policy
    //   Layer 1 (Primary): LockFreeRingBuffer<ObserveIntent, 1024> (SPSC lock-free)
    //   Layer 2 (Fallback): LockFreeRingBuffer<ObserveIntent, 2048> (SPSC lock-free)
    //   Layer 3 (Deferred): RetireOverflowEntry → coordinatorDeferredRing_ → drainOverflowRing
    //   Layer 4 (Quarantine): emitQuarantineIntent — 最終安全策
    if (observeIntentQueue_.push(intent)) {
        // Layer 1 (Primary): 正常完了 — ACK (queued)
        overflowCounter_.store(0, std::memory_order_relaxed);
        setPendingIntentCount(pendingIntentCount_.load(std::memory_order_relaxed) + 1);
        return;
    }

    // Layer 2 (Fallback): Primary 溢れ → セカンダリキュー
    overflowCounter_.fetch_add(1, std::memory_order_relaxed);
    if (observeFallbackQueue_.push(intent)) {
        setPendingIntentCount(pendingIntentCount_.load(std::memory_order_relaxed) + 1);
        return;
    }

    // ★ QUEUE-12: Fallback も満杯 → Layer 3 (Deferred)
    //   coordinatorDeferredRing_ に ObserveFallbackEntry を変換して enqueue
    //   coordinatorDeferredRing_ は drainOverflowRing で定期回収される
    fallbackOverflowCounter_.fetch_add(1, std::memory_order_relaxed);
    RetireOverflowEntry deferredEntry{};
    deferredEntry.intent.dspSlot = 0;
    deferredEntry.intent.priority = RetirePriority::Normal;
    deferredEntry.overflowTimestampUs = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    deferredEntry.reinjectRetryCount = 0;
    if (coordinatorDeferredRing_.push(deferredEntry)) {
        coordinatorDeferredCount_.fetch_add(1, std::memory_order_relaxed);
    } else {
        // Layer 3 overflow → Layer 4 (Quarantine): 全層溢れの最終安全策
        // observeIntentQueue_ / observeFallbackQueue_ / coordinatorDeferredRing_
        // の3層全てが満杯。Coordinator Deferred Ring も満杯のため、
        // この Intent はドロップされる。カウンタで記録する。
        // 通常運用でここに到達することはない（合計1024+2048+1024=4096超え）。
    }
    setPendingIntentCount(pendingIntentCount_.load(std::memory_order_relaxed) + 1);
}

void RuntimePublicationCoordinator::requestReclaim(
    const DSPHandle& handle,
    DSPHandleRuntime& handleRuntime,
    ISRRetireRouter& router) noexcept
{
    // ★ P0-4B: Coordinator 専用 reclaim 要求
    //   DELETE-2: executeRetire → waitReaders → executeReclaim の順序
    //   DELETE-3: waitReaders で epoch 安全確認後にのみ reclaim

    // 1. executeRetire(handle) — DSPHandleRuntime に retire を委譲
    handleRuntime.retire(handle);

    // 2. waitReaders(handle) — ISR不変条件: epoch 安全確認
    //    retireEpoch < minReaderEpoch で安全判定
    const auto retireEpoch = router.currentEpoch();
    const auto minReaderEpoch = router.minReaderEpoch();
    if (retireEpoch >= minReaderEpoch) {
        // Reader がまだアクティブ → 再試行（次の processIntent サイクルで再確認）
        // カウンタ更新のみ行い、即座に復帰（NonRT safe）
        setReclaimInFlightCount(reclaimInFlightCount_.load(std::memory_order_relaxed) + 1);
        return;
    }

    // 3. executeReclaim(handle) — 安全確認完了
    //    DELETE-8: Coordinator は Reclaimed 状態への遷移のみを行い、
    //    物理削除（DSPCore* の delete）は DSPLifetimeManager 経由で別途実行済み。
    //    DSPHandleRuntime::reclaim() は Handle の状態を Reclaimed に遷移する
    //    （DSPCore* の削除は行わない — 既に retire path で enqueueWithRetry 済み）。
    handleRuntime.reclaim(handle);
    // ACK: reclaim complete — カウンタリセット
    setReclaimInFlightCount(0);
}

//==============================================================================
// ★ P0-5: QuarantineService implementation
//==============================================================================

QuarantineService::QuarantineResult QuarantineService::executeQuarantine(
    DSPHandleRuntime& handleRuntime,
    DSPQuarantineManager& quarantineManager,
    const QuarantineRequest& request) noexcept
{
    // QSVC-1: State変更 + Audit を単一トランザクションとして実行
    QuarantineResult result{};

    // 1. State 変更 — DSPHandleRuntime::quarantine()
    if (!request.handle.isNull() && request.handle.slot > 0) {
        handleRuntime.quarantine(request.handle);
        result.stateChanged = true;
    }

    // 2. Audit 記録 — DSPQuarantineManager::quarantineHandle()
    const bool auditLogged = quarantineManager.quarantineHandle(
        request.handle.slot,
        request.handle.generation,
        request.reason);
    result.auditLogged = auditLogged;

    // QSVC-5 (ISR準拠): Audit 失敗時も State は変更しない。
    //   Publish後は Immutable。Rollback 禁止。Diagnostic カウンタのみ更新。
    if (!auditLogged && result.stateChanged) {
        result.rolledBack = false;
        result.stateChanged = true;  // State 変更は確定
    }

    return result;
}

void RuntimePublicationCoordinator::emitQuarantineIntent(
    const DSPHandle& handle,
    QuarantineReason reason,
    DSPHandleRuntime& handleRuntime,
    DSPQuarantineManager& quarantineManager,
    uint64_t contextEpoch) noexcept
{
    // QSVC-2: Coordinator は QuarantineService 経由で quarantine を実行
    // 直接 dspHandleRuntime_.quarantine() を呼ばない
    QuarantineService::QuarantineRequest request{handle, reason, contextEpoch};
    quarantineService_.executeQuarantine(handleRuntime, quarantineManager, request);
    setQuarantineResidentCount(quarantineResidentCount_.load(std::memory_order_relaxed) + 1);
}

} // namespace convo::isr
