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
    [[nodiscard]] bool canRollback() const noexcept;
    void requestRollback() noexcept;
    [[nodiscard]] EpochStrategyDescriptor describeEpochStrategy() const noexcept;

    [[nodiscard]] RetireLane laneOf(std::uint32_t slot) const noexcept;
    void emitRetireTimeline(const std::filesystem::path& outputPath) const;

private:
    static constexpr std::size_t kMaxSlots = 256;
    std::array<std::atomic<uint32_t>, kMaxSlots> laneBySlot_{};
    std::array<std::atomic<std::uint64_t>, 5> laneCounters_{};
    std::atomic<std::uint64_t> totalTransitions_{0};
    std::atomic<std::uint32_t> epochModeRaw_{static_cast<std::uint32_t>(EpochMode::Shared)};
    std::atomic<std::uint32_t> rollbackModeRaw_{static_cast<std::uint32_t>(EpochMode::Shared)};
    std::atomic<bool> rollbackReady_{true};
};

} // namespace convo::isr
