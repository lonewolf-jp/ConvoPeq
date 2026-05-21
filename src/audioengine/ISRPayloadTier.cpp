#include "ISRPayloadTier.h"

namespace convo {
namespace isr {

bool PayloadTierValidator::isValidTier(uint32_t tierValue) const noexcept
{
    return tierValue <= static_cast<uint32_t>(PayloadTier::Forbidden);
}

bool PayloadTierValidator::validateTierSequence(const std::vector<PayloadTier>& tiers) const noexcept
{
    if (tiers.empty()) {
        return false;
    }

    uint32_t prev = 0;
    bool first = true;
    for (const auto& tier : tiers) {
        const auto v = static_cast<uint32_t>(tier);
        if (!isValidTier(v) || tier == PayloadTier::Forbidden) {
            return false;
        }

        if (!first && v < prev) {
            return false;
        }

        prev = v;
        first = false;
    }

    for (const auto& dep : dependencies_) {
        bool hasFrom = false;
        bool hasTo = false;
        for (const auto& tier : tiers) {
            if (tier == dep.first) hasFrom = true;
            if (tier == dep.second) hasTo = true;
        }
        if (hasTo && !hasFrom) {
            return false;
        }
    }

    return true;
}

void PayloadTierValidator::registerTierDependency(PayloadTier from, PayloadTier to)
{
    dependencies_.emplace_back(from, to);
}

TierRejectReason PayloadTierValidator::explainPublishReject(const TieredPayloadDescriptor& descriptor) const noexcept
{
    if (!isValidTier(static_cast<uint32_t>(descriptor.tier))) {
        return TierRejectReason::InvalidTier;
    }

    if (descriptor.tier == PayloadTier::Forbidden) {
        return TierRejectReason::ForbiddenTier;
    }

    if (descriptor.tier == PayloadTier::RTLocalOnly) {
        return TierRejectReason::RTLocalLeak;
    }

    if (descriptor.requiresRT && descriptor.tier == PayloadTier::ExternalPinned && !descriptor.pinnedLifetime) {
        return TierRejectReason::ExternalPinnedWithoutLifetime;
    }

    if (descriptor.hasExternalResource && descriptor.tier == PayloadTier::InlineImmutable) {
        return TierRejectReason::InlineImmutableWithExternalResource;
    }

    return TierRejectReason::None;
}

bool PayloadTierValidator::isPublishAllowed(const TieredPayloadDescriptor& descriptor) const noexcept
{
    return explainPublishReject(descriptor) == TierRejectReason::None;
}

bool PayloadTierValidator::isDeferredReclaimRequired(const TieredPayloadDescriptor& descriptor) const noexcept
{
    if (descriptor.tier == PayloadTier::ExternalPinned) {
        return true;
    }

    return descriptor.hasExternalResource;
}

}  // namespace isr
}  // namespace convo
