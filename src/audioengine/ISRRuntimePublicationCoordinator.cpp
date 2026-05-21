#include "ISRRuntimePublicationCoordinator.h"
#include "AtomicAccess.h"

namespace convo::isr {

RuntimePublicationCoordinator::RuntimePublicationCoordinator() : currentWorld_(nullptr), version_(0), lastRejectCode_(RejectCode::None) {}

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

    convo::publishAtomic(currentWorld_, newWorld, std::memory_order_release);
    convo::publishAtomic(version_, version, std::memory_order_release);
}

void RuntimePublicationCoordinator::retire(RetireAuthority,
                                           RuntimeBoundary boundary,
                                           const void* oldWorld) {
    if (boundary != RuntimeBoundary::NonRTWorld || oldWorld == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(retireGuard_);
    retiredWorlds_.push_back(oldWorld);
}

const void* RuntimePublicationCoordinator::getCurrent() const noexcept {
    return convo::consumeAtomic(currentWorld_, std::memory_order_acquire);
}

std::uint64_t RuntimePublicationCoordinator::getVersion() const noexcept {
    return convo::consumeAtomic(version_, std::memory_order_acquire);
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
