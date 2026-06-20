#include "ISRRuntimePublicationCoordinator.h"
#include "AtomicAccess.h"

namespace convo::isr {

RuntimePublicationCoordinator::RuntimePublicationCoordinator()
    : currentWorld_(nullptr)
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

    if (router.enqueueRetire(ptr, deleter, epoch, DeletionEntryType::Generic) != RetireEnqueueResult::Success)
        return RetireEnqueueResult::QueueFull;

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

bool RuntimePublicationCoordinator::isFullyDrained() const noexcept {
    if (convo::consumeAtomic(swapPending_, std::memory_order_acquire)) {
        return false;
    }

    return convo::consumeAtomic(retireBacklogCount_, std::memory_order_acquire) == 0
        && convo::consumeAtomic(publicationBacklogCount_, std::memory_order_acquire) == 0
        && convo::consumeAtomic(pendingIntentCount_, std::memory_order_acquire) == 0
        && convo::consumeAtomic(fallbackBacklogCount_, std::memory_order_acquire) == 0
        && convo::consumeAtomic(reclaimInFlightCount_, std::memory_order_acquire) == 0
        && convo::consumeAtomic(deferredRetireResidencyCount_, std::memory_order_acquire) == 0;
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

void RuntimePublicationCoordinator::requestShutdown() noexcept {
    convo::publishAtomic(state_, CoordinatorState::ShuttingDown, std::memory_order_release);
}

void RuntimePublicationCoordinator::markShutdownComplete() noexcept {
    const auto state = convo::consumeAtomic(state_, std::memory_order_acquire);
    if (state != CoordinatorState::ShuttingDown) {
        return;
    }

    if (isFullyDrained()) {
        convo::publishAtomic(state_, CoordinatorState::Bootstrapping, std::memory_order_release);
    } else {
        convo::publishAtomic(state_, CoordinatorState::Faulted, std::memory_order_release);
    }
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

void PublicationBuffer::enqueue(const void* world) {
    if (world == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(guard_);
    queued_.push_back(world);
}

void PublicationBuffer::retireOld() {
    std::lock_guard<std::mutex> lock(guard_);
    if (!queued_.empty()) {
        queued_.erase(queued_.begin());
    }
}

std::size_t PublicationBuffer::size() noexcept {
    std::lock_guard<std::mutex> lock(guard_);
    return queued_.size();
}

} // namespace convo::isr
