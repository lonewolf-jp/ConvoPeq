#pragma once
#include "ISRRetireLane.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>

namespace convo::isr {

enum class EpochMode : std::uint32_t {
    Shared = 0,
    Split = 1,
    Hybrid = 2
};

struct EpochStrategyDescriptor {
    EpochMode activeMode{EpochMode::Shared};
    EpochMode rollbackMode{EpochMode::Shared};
    bool rollbackReady{true};
};

struct RollbackFlagDescriptor {
    bool globalEnabled{true};
    bool publicationOnlyEnabled{false};
    bool crossfadeOnlyEnabled{false};
    bool retirePathOnlyEnabled{true};
};

enum class RetireLifecycleState : std::uint32_t {
    Visible = 0,
    CompareEligible,
    TelemetryRetained,
    ReplayRetainedOptional,
    ReclaimEligible,
    Reclaimed
};

class RetireRuntimeEx {
public:
    RetireRuntimeEx();
    void emitIntent(std::uint32_t slot, std::uint32_t generation);
    void enqueueRetire(std::uint32_t slot);
    void settleEpoch(std::uint32_t slot);
    void reclaim(std::uint32_t slot);
    void quarantine(std::uint32_t slot);
    void setEpochMode(EpochMode mode) noexcept;
    [[nodiscard]] EpochMode getEpochMode() const noexcept;
    void setRollbackMode(EpochMode mode) noexcept;
    [[nodiscard]] EpochMode getRollbackMode() const noexcept;
    void setRollbackFlags(bool globalEnabled,
                          bool publicationOnlyEnabled,
                          bool crossfadeOnlyEnabled,
                          bool retirePathOnlyEnabled) noexcept;
    [[nodiscard]] RollbackFlagDescriptor describeRollbackFlags() const noexcept;
    [[nodiscard]] bool canRollback() const noexcept;
    void requestRollback() noexcept;
    [[nodiscard]] EpochStrategyDescriptor describeEpochStrategy() const noexcept;
    [[nodiscard]] std::uint64_t getQuarantineResidentCount() const noexcept;
    [[nodiscard]] RetireLifecycleState lifecycleStateOf(std::uint32_t slot) const noexcept;
    [[nodiscard]] static bool isGracePeriodCompleted(std::uint64_t worldGeneration,
                                                     std::uint64_t maxObservedGeneration,
                                                     std::uint32_t audioCallbackActiveCount) noexcept
    {
        return (maxObservedGeneration > worldGeneration) || (audioCallbackActiveCount == 0u);
    }
    [[nodiscard]] static bool canTransitionRetirePendingToFree(bool graceCompleted,
                                                               bool pendingIntentOwned,
                                                               bool authoritativeOwnershipReleased) noexcept
    {
        return graceCompleted && pendingIntentOwned && authoritativeOwnershipReleased;
    }

    [[nodiscard]] RetireLane laneOf(std::uint32_t slot) const noexcept;
    void emitRetireTimeline(const std::filesystem::path& outputPath) const;

private:
    static constexpr std::size_t kMaxSlots = 256;
    std::array<std::atomic<uint32_t>, kMaxSlots> laneBySlot_{};
    std::array<std::atomic<std::uint32_t>, kMaxSlots> lifecycleStateBySlot_{};
    std::array<std::atomic<std::uint64_t>, 5> laneCounters_{};
    std::array<std::atomic<std::uint64_t>, 6> lifecycleCounters_{};
    std::atomic<std::uint64_t> totalTransitions_{0};
    std::atomic<std::uint64_t> quarantineResidentCount_{0};
    std::atomic<std::uint32_t> epochModeRaw_{static_cast<std::uint32_t>(EpochMode::Shared)};
    std::atomic<std::uint32_t> rollbackModeRaw_{static_cast<std::uint32_t>(EpochMode::Shared)};
    std::atomic<bool> rollbackGlobalEnabled_{true};
    std::atomic<bool> rollbackPublicationOnlyEnabled_{false};
    std::atomic<bool> rollbackCrossfadeOnlyEnabled_{false};
    std::atomic<bool> rollbackRetirePathOnlyEnabled_{true};
    std::atomic<bool> rollbackReady_{true};
};

} // namespace convo::isr
