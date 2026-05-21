#pragma once

#include <cstdint>
#include <vector>

namespace convo {
namespace isr {

/**
 * ISR 10層 Architecture Layer 4: Payload Tier
 * publication の多層 staged publish を categorize する。
 */

/**
 * Payload tier enumeration
 * Tier の値が大きいほど、より後の publication stage
 */
enum class PayloadTier : uint32_t
{
    InlineImmutable,
    ImmutableShared,
    ExternalPinned,
    RTLocalOnly,
    Forbidden,
    Invalid = 0xFFFFFFFFu
};

struct TieredPayloadDescriptor
{
    PayloadTier tier = PayloadTier::Invalid;
    bool requiresRT = false;
    bool hasExternalResource = false;
    bool pinnedLifetime = false;
};

enum class TierRejectReason : uint8_t
{
    None = 0,
    InvalidTier,
    ForbiddenTier,
    RTLocalLeak,
    ExternalPinnedWithoutLifetime,
    InlineImmutableWithExternalResource
};

/**
 * Payload tier validator
 * publication order を enforce
 */
class PayloadTierValidator
{
public:
    // tier value を検証
    bool isValidTier(uint32_t tierValue) const noexcept;

    // tier sequence を検証（ordered publish か確認）
    bool validateTierSequence(const std::vector<PayloadTier>& tiers) const noexcept;

    // tier 依存グラフを register
    void registerTierDependency(PayloadTier from, PayloadTier to);
    TierRejectReason explainPublishReject(const TieredPayloadDescriptor& descriptor) const noexcept;
    bool isPublishAllowed(const TieredPayloadDescriptor& descriptor) const noexcept;
    bool isDeferredReclaimRequired(const TieredPayloadDescriptor& descriptor) const noexcept;

private:
    std::vector<std::pair<PayloadTier, PayloadTier>> dependencies_;
};

}  // namespace isr
}  // namespace convo
