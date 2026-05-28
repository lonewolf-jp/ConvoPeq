#include "ISRRuntimePublicationCoordinator.h"
#include "AtomicAccess.h"

namespace convo::isr {

RuntimePublicationCoordinator::RuntimePublicationCoordinator()
    : currentWorld_(nullptr)
    , version_(0)
    , lastRejectCode_(RejectCode::None)
    , retireBacklogCount_(0)
    , publicationBacklogCount_(0)
    , pendingIntentCount_(0)
    , fallbackBacklogCount_(0)
    , reclaimInFlightCount_(0)
    , deferredRetireResidencyCount_(0)
    , swapPending_(false)
{
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
    if (boundary != RuntimeBoundary::NonRTWorld || newWorld == nullptr) {
        return;
    }

    convo::publishAtomic(swapPending_, true, std::memory_order_release);
    convo::publishAtomic(currentWorld_, newWorld, std::memory_order_release);
    convo::publishAtomic(version_, version, std::memory_order_release);
    convo::publishAtomic(swapPending_, false, std::memory_order_release);
}

void RuntimePublicationCoordinator::retire(RetireAuthority,
                                           RuntimeBoundary boundary,
                                           const void* oldWorld) {
    if (boundary != RuntimeBoundary::NonRTWorld || oldWorld == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(retireGuard_);
    if (retiredWorlds_.size() >= kMaxRetiredWorldResidency) {
        retiredWorlds_.erase(retiredWorlds_.begin());
    }
    retiredWorlds_.push_back(oldWorld);
    convo::publishAtomic(retireBacklogCount_, static_cast<std::uint64_t>(retiredWorlds_.size()), std::memory_order_release);
}

const void* RuntimePublicationCoordinator::getCurrent() const noexcept {
    return convo::consumeAtomic(currentWorld_, std::memory_order_acquire);
}

std::uint64_t RuntimePublicationCoordinator::getVersion() const noexcept {
    return convo::consumeAtomic(version_, std::memory_order_acquire);
}

void RuntimePublicationCoordinator::setRetireBacklogCount(std::uint64_t count) noexcept {
    convo::publishAtomic(retireBacklogCount_, count, std::memory_order_release);
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
