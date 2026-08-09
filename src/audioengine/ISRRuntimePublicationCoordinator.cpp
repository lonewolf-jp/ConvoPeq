#include "ISRRuntimePublicationCoordinator.h"
#include "AtomicAccess.h"
#include "ISRRetireOverflowRing.h"
#include "ISRDSPHandle.h"
#include "ISRDSPQuarantine.h"
#include <cassert>
#include "AudioEngine.h"  // FUTURE-4: RuntimeState (downcast currentWorld_)

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
    // ★ FUTURE-4: persistentState_ removed — metadata derived from currentWorld_ (RuntimeState::publication)
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

    // ★ FUTURE-4: prev metadata derived from currentWorld_ (RuntimeState::publication), not persistentState_
    const auto prevWorld = static_cast<const RuntimeState*>(
        convo::consumeAtomic(currentWorld_, std::memory_order_acquire));
    const auto prevSeqId = prevWorld ? prevWorld->publication.sequenceId : PublicationSequenceId{0};
    const auto prevEpoch  = prevWorld ? prevWorld->publication.epoch : PublicationEpoch{0};
    const auto prevGen    = prevWorld ? prevWorld->publication.mappedRuntimeGeneration : std::uint64_t{0};

    const bool hasPrevious = prevSeqId != 0 || prevEpoch != 0 || prevGen != 0;
    if (hasPrevious) {
        if (!(static_cast<std::uint64_t>(sequenceId) > static_cast<std::uint64_t>(prevSeqId)
              && static_cast<std::uint64_t>(epoch) > static_cast<std::uint64_t>(prevEpoch)
              && mappedGeneration > prevGen)) {
            convo::publishAtomic(state_, CoordinatorState::Faulted, std::memory_order_release);
            return;
        }
    }

    convo::publishAtomic(state_, CoordinatorState::Publishing, std::memory_order_release);
    convo::publishAtomic(swapPending_, true, std::memory_order_release);

    // ★ FUTURE-4 (publish-time freeze): bake incoming publication semantics onto newWorld,
    //   replacing the former persistentState_ cache; Reader derives metadata via currentWorld_->publication.
    auto* pubWorld = const_cast<RuntimeState*>(static_cast<const RuntimeState*>(newWorld));
    pubWorld->publication = PublicationSemantic{sequenceId, epoch,
        static_cast<PublicationGeneration>(mappedGeneration), prevSeqId};
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
    // ★ FUTURE-4: derive from currentWorld_ (RuntimeState::publication.mappedRuntimeGeneration)
    const auto world = static_cast<const RuntimeState*>(
        convo::consumeAtomic(currentWorld_, std::memory_order_acquire));
    return world ? world->publication.mappedRuntimeGeneration : std::uint64_t{0};
}

PublicationEpoch RuntimePublicationCoordinator::currentPublicationEpoch() const noexcept {
    // ★ FUTURE-4: latest publicationEpoch derived from currentWorld_ (RuntimeState::publication.epoch)
    const auto world = static_cast<const RuntimeState*>(
        convo::consumeAtomic(currentWorld_, std::memory_order_acquire));
    return world ? world->publication.epoch : PublicationEpoch{0};
}

// ★ A-1: sequence derived from currentWorld_ (RuntimeState::publication.sequenceId).
//   Read-only Authority accessor — RuntimeWorldAuthority::sequence() delegates here.
PublicationSequenceId RuntimePublicationCoordinator::currentPublicationSequenceId() const noexcept {
    const auto world = static_cast<const RuntimeState*>(
        convo::consumeAtomic(currentWorld_, std::memory_order_acquire));
    return world ? world->publication.sequenceId : PublicationSequenceId{0};
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
    RetireOverflowRing& overflowRing, LifetimeState& retireRuntime, bool unlimited) noexcept
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
    RetireOverflowRing& overflowRing, LifetimeState& retireRuntime, bool unlimited) noexcept
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
    // ★ Phase5: Coordinator の escalateAllRetires は LifetimeState に委譲
    //   実装は AudioEngine.Processing.ReleaseResources.cpp の worldAuthority_.lifetime().escalateAllRetires() が担当
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

void RuntimePublicationCoordinator::submitObserve(const DSPHandle& handle) noexcept
{
    // ★ P0-4A: Observe Intent — Timer → enqueue（RT-safe, lock-free）
    //   OBSERVE-2: push() は即座に復帰（lock-free）
    //   OBSERVE-9: FIFO を保証（共通 intentQueue_ = MpscBoundedRing, reservation order）
    //   OBSERVE-10: 世代検証用に epoch を保存
    //   ISR: Intent は自己完結型 — DSPHandle を含むため、Coordinator は外部状態に依存せず retire 対象を識別可能
    //
    // ★ work88 (FUTURE-10 / Phase 7): Observe 統合 — 共通 intentQueue_ (MPSC) を primary に。
    //   cross-type FIFO で Publish/Quarantine と同じキューを共有（ObserveIntentHandler が
    //   kDispatchTable 経由で処理）。SPSC 専用リング（observeIntentQueue_/observeFallbackQueue_）
    //   への複数 Producer 依存を排除（MPSC 実態の潜在競合を構造的に解消）。
    const auto world = static_cast<const RuntimeState*>(
        convo::consumeAtomic(currentWorld_, std::memory_order_acquire));
    const auto currentEpoch = world ? world->publication.epoch : PublicationEpoch{0};
    const auto intentId = nextObserveIntentId_.fetch_add(1, std::memory_order_relaxed);

    RuntimePublicationCoordinator::Intent intent{};
    intent.type = RuntimePublicationCoordinator::IntentType::Observe;
    intent.payload.observe = RuntimePublicationCoordinator::ObservePayload{handle, currentEpoch};
    intent.sequenceId = intentId;
    if (intentQueue_.push(intent)) {
        setPendingIntentCount(pendingIntentCount_.load(std::memory_order_relaxed) + 1);
        return;
    }

    // intentQueue_ full → observeDeferredRing_（FUTURE-8 overflow 専用）へ退避。
    //   回御は processIntent の drainObserveDeferred（QUEUE-16）。
    observeOverflowCounter_.fetch_add(1, std::memory_order_relaxed);
    ObserveIntent fallbackIntent{ handle, currentEpoch, intentId };
    if (observeDeferredRing_.push(fallbackIntent)) {
        setPendingIntentCount(pendingIntentCount_.load(std::memory_order_relaxed) + 1);
        return;
    }

    // 全層溢れ（intentQueue_ + deferred ring）→ drop カウンタ。
    //   Observe は観測情報のため後発 Observe で補完可能（三次レビュー policy 表:
    //   Observe は条件付き drop / coalesce 可）。Publish/Quarantine とは異なり
    //   state transition の喪失ではないため許容。
    observeFallbackOverflowCounter_.fetch_add(1, std::memory_order_relaxed);
}

bool RuntimePublicationCoordinator::requestReclaim(
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
        // ★ work88 (六次レビュー — TOCTOU 修正): 呼出し元へ「遅延」を通知。
        //   呼出し元（requestReclaimHandle / drainDeferredRetireQueues）が handle を
        //   再試行リストへ戻す（slot リーク防止）。
        return false;
    }

    // 3. executeReclaim(handle) — 安全確認完了
    //    DELETE-8: Coordinator は Reclaimed 状態への遷移のみを行い、
    //    物理削除（DSPCore* の delete）は DSPLifetimeManager 経由で別途実行済み。
    //    DSPHandleRuntime::reclaim() は Handle の状態を Reclaimed に遷移する
    //    （DSPCore* の削除は行わない — 既に retire path で enqueueWithRetry 済み）。
    handleRuntime.reclaim(handle);
    // ACK: reclaim complete — カウンタリセット
    setReclaimInFlightCount(0);
    return true;
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

    // ★ FUTURE-3/QSVC-5: Audit 失敗時も State は変更しない。Publish後は Immutable。Rollback 廃止。
    //   Diagnostic カウンタのみ更新。（stateChanged は既に確定。rolledBack は削除 ── New World Publish が復旧担う。）

    return result;
}

// ★ FUTURE-3: Coordinator は Recovery Request enqueue のみ。Rollback ではない — New World の Immutable Publish が復旧担う。
//   Builder → Validate → Publish は Builder Loop が popRecoveryRequest() で消費（Admission 判定なし）。
//   transport-only: saturate 時は drop。Decision Authority を持たない。
// ★ FUTURE-3 (work88): buildSource（RuntimeBuildSnapshot 値コピー）を payload に内包。
//   quarantinedHandle だけでは resolve() 不能（ISRDSPHandle.cpp:69）なため、build 入力は
//   値コピーした snapshot から引当する（epoch 逆引き不要 — lifetime を構造的に解決）。
void RuntimePublicationCoordinator::submitRecoveryRequest(const DSPHandle& quarantinedHandle,
                                                          const convo::RuntimeBuildSnapshot& buildSource) noexcept
{
    const auto world = static_cast<const RuntimeState*>(
        convo::consumeAtomic(currentWorld_, std::memory_order_acquire));
    const auto currentEpoch = world ? world->publication.epoch : PublicationEpoch{0};

    RecoveryIntent intent{
        quarantinedHandle,
        currentEpoch,
        nextRecoveryIntentId_.fetch_add(1, std::memory_order_relaxed),
        buildSource
    };

    // ★ work88 (六次レビュー — INV-5: Recovery drop 禁止):
    //   recoveryIntentQueue_ は SPSC（Producer=CoordinatorLoop, Consumer=Builder Loop）。
    //   push 失敗（full = Builder が遅延）は Recovery Intent の drop を意味し、INV-5 違反。
    //   drop された場合、pendingIntentCount_ を増やさない（pop 時 -1 と整合し、shutdown の
    //   isFullyDrained が false のままハングするのを防ぐ）。drop は診断カウンタに記録。
    if (recoveryIntentQueue_.push(intent)) {
        setPendingIntentCount(pendingIntentCount_.load(std::memory_order_relaxed) + 1);
    } else {
        // ★ drop 記録 — Recovery Intent が失われるため Critical 相当の診断。静かに破棄しない。
        //   HealthMonitor が RecoveryDrop を監視し、再発時は ISRHealthState 昇格を駆動する。
        convo::fetchAddAtomic(recoveryIntentDropCount_, uint64_t{1}, std::memory_order_release);
    }
}

std::optional<RuntimePublicationCoordinator::RecoveryIntent>
RuntimePublicationCoordinator::popRecoveryRequest() noexcept
{
    RecoveryIntent intent{};
    if (!recoveryIntentQueue_.pop(intent))
        return std::nullopt;              // transport-only pop: empty は Builder 消費の前提
    // ★ 監査指摘 (work88): processIntent が pendingIntentCount を 0 にリセットした後に pop が
    //   減算すると uint64 underflow（巨大値）→ isFullyDrained の pendingIntentCount==0 が false
    //   になりシャットダウンがハングし得る。0 未満へは減算しないガードを追加。
    const auto cur = pendingIntentCount_.load(std::memory_order_relaxed);
    if (cur > 0)
        setPendingIntentCount(cur - 1);
    return intent;
}

void RuntimePublicationCoordinator::submitQuarantine(
    const DSPHandle& handle,
    QuarantineReason reason,
    DSPHandleRuntime& handleRuntime,
    DSPQuarantineManager& quarantineManager,
    uint64_t contextEpoch) noexcept
{
    // ★ A3 Step 4: sync → async. submitQuarantine no longer executes directly;
    //   it enqueues a Quarantine Intent onto the common intentQueue_.
    //   QSVC-2: Execution delegated to QuarantineService via QuarantineIntentHandler
    //   (kDispatchTable) in processIntent — Coordinator retains Decision Authority over enqueue only.
    //   (handleRuntime/quarantineManager params retained for API stability; the handler
    //    re-sources them through HandlerContext.engine. Step 6 removes these redundant params.)
    (void)handleRuntime;
    (void)quarantineManager;

    const std::uint64_t seqId = nextIntentId_.fetch_add(1, std::memory_order_relaxed);
    RuntimePublicationCoordinator::Intent intent{};
    intent.type = RuntimePublicationCoordinator::IntentType::Quarantine;
    intent.payload.quarantine = RuntimePublicationCoordinator::QuarantinePayload{handle, reason, contextEpoch};
    intent.sequenceId = seqId;

    if (intentQueue_.push(intent)) {
        setQuarantineResidentCount(quarantineResidentCount_.load(std::memory_order_relaxed) + 1);
        setPendingIntentCount(pendingIntentCount_.load(std::memory_order_relaxed) + 1);
        return;
    }

    // ★ work88 (FUTURE-10 / 三次レビュー policy 表): Quarantine intent の drop は禁止。
    //   quarantine 検出は安全要件（bad DSP のアクセス禁止）であり、drop されると
    //   quarantined DSP が永久に retire されず RT からアクセス不能なメモリが残存する。
    //   intentQueue_ full 時は Quarantine 専用 fallback ring へ退避（drop しない）。
    if (quarantineFallbackQueue_.push(intent)) {
        setQuarantineResidentCount(quarantineResidentCount_.load(std::memory_order_relaxed) + 1);
        setPendingIntentCount(pendingIntentCount_.load(std::memory_order_relaxed) + 1);
        return;
    }

    // fallback も full → 絶対に静かに破棄しない。drop カウンタを増やして診断に残す。
    //   （AudioEngine 側の HealthMonitor が quarantineFallbackDropCount を監視し、
    //    ISRHealthState::Critical 昇格 / controlled shutdown を駆動する。）
    convo::fetchAddAtomic(quarantineFallbackDropCount_, uint64_t{1}, std::memory_order_release);
}

} // namespace convo::isr
