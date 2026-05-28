#include "ISRRetireRuntimeEx.h"
#include "AtomicAccess.h"
#include "DspNumericPolicy.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace convo::isr {

namespace {
constexpr std::uint32_t toRaw(RetireLane lane) noexcept
{
    return static_cast<std::uint32_t>(lane);
}

constexpr std::uint32_t toRaw(EpochMode mode) noexcept
{
    return static_cast<std::uint32_t>(mode);
}

RetireLane fromRaw(std::uint32_t raw) noexcept
{
    switch (static_cast<RetireLane>(raw)) {
    case RetireLane::RTIntent:
    case RetireLane::Coordination:
    case RetireLane::Epoch:
    case RetireLane::Reclaim:
    case RetireLane::Quarantine:
        return static_cast<RetireLane>(raw);
    default:
        return RetireLane::RTIntent;
    }
}

EpochMode epochModeFromRaw(std::uint32_t raw) noexcept
{
    switch (raw) {
    case static_cast<std::uint32_t>(EpochMode::Shared):
        return EpochMode::Shared;
    case static_cast<std::uint32_t>(EpochMode::Split):
        return EpochMode::Split;
    case static_cast<std::uint32_t>(EpochMode::Hybrid):
        return EpochMode::Hybrid;
    default:
        return EpochMode::Shared;
    }
}

const char* epochModeName(EpochMode mode) noexcept
{
    switch (mode) {
    case EpochMode::Shared: return "shared";
    case EpochMode::Split: return "split";
    case EpochMode::Hybrid: return "hybrid";
    default: return "shared";
    }
}

bool readEnvFlag(const char* name, bool defaultValue) noexcept
{
    if (name == nullptr || name[0] == '\0') {
        return defaultValue;
    }

    if (const char* raw = std::getenv(name); raw != nullptr) {
        const char first = raw[0];
        if (first == '1' || first == 't' || first == 'T' || first == 'y' || first == 'Y') {
            return true;
        }
        if (first == '0' || first == 'f' || first == 'F' || first == 'n' || first == 'N') {
            return false;
        }
    }

    return defaultValue;
}
} // namespace

RetireRuntimeEx::RetireRuntimeEx()
{
    for (auto& lane : laneBySlot_) {
        convo::publishAtomic(lane, toRaw(RetireLane::RTIntent), std::memory_order_relaxed);
    }

    for (auto& counter : laneCounters_) {
        convo::publishAtomic(counter, static_cast<std::uint64_t>(0), std::memory_order_relaxed);
    }

    const bool globalEnabled = readEnvFlag("ISR_ROLLBACK_GLOBAL", true);
    const bool publicationOnlyEnabled = readEnvFlag("ISR_ROLLBACK_PUBLICATION_ONLY", false);
    const bool crossfadeOnlyEnabled = readEnvFlag("ISR_ROLLBACK_CROSSFADE_ONLY", false);
    const bool retirePathOnlyEnabled = readEnvFlag("ISR_ROLLBACK_RETIRE_PATH_ONLY", true);

    convo::publishAtomic(totalTransitions_, static_cast<std::uint64_t>(0), std::memory_order_relaxed);
    convo::publishAtomic(rollbackModeRaw_, toRaw(EpochMode::Shared), std::memory_order_relaxed);
    convo::publishAtomic(rollbackGlobalEnabled_, globalEnabled, std::memory_order_relaxed);
    convo::publishAtomic(rollbackPublicationOnlyEnabled_, publicationOnlyEnabled, std::memory_order_relaxed);
    convo::publishAtomic(rollbackCrossfadeOnlyEnabled_, crossfadeOnlyEnabled, std::memory_order_relaxed);
    convo::publishAtomic(rollbackRetirePathOnlyEnabled_, retirePathOnlyEnabled, std::memory_order_relaxed);
    convo::publishAtomic(rollbackReady_, (globalEnabled && retirePathOnlyEnabled), std::memory_order_relaxed);
    convo::publishAtomic(quarantineResidentCount_, static_cast<std::uint64_t>(0), std::memory_order_relaxed);
}

void RetireRuntimeEx::emitIntent(std::uint32_t slot, std::uint32_t generation) {
    if (slot >= laneBySlot_.size()) {
        return;
    }

    const auto lane = (generation == 0u) ? RetireLane::Quarantine : RetireLane::RTIntent;
    convo::publishAtomic(laneBySlot_[slot], toRaw(lane), std::memory_order_release);
    (void)convo::fetchAddAtomic(laneCounters_[static_cast<std::size_t>(lane)], static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
    (void)convo::fetchAddAtomic(totalTransitions_, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
}

void RetireRuntimeEx::enqueueRetire(std::uint32_t slot) {
    ASSERT_NON_RT_THREAD();
    if (slot >= laneBySlot_.size()) {
        return;
    }
    convo::publishAtomic(laneBySlot_[slot], toRaw(RetireLane::Coordination), std::memory_order_release);
    (void)convo::fetchAddAtomic(laneCounters_[static_cast<std::size_t>(RetireLane::Coordination)], static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
    (void)convo::fetchAddAtomic(totalTransitions_, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
}

void RetireRuntimeEx::settleEpoch(std::uint32_t slot) {
    if (slot >= laneBySlot_.size()) {
        return;
    }
    convo::publishAtomic(laneBySlot_[slot], toRaw(RetireLane::Epoch), std::memory_order_release);
    (void)convo::fetchAddAtomic(laneCounters_[static_cast<std::size_t>(RetireLane::Epoch)], static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
    (void)convo::fetchAddAtomic(totalTransitions_, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
}

void RetireRuntimeEx::reclaim(std::uint32_t slot) {
    if (slot >= laneBySlot_.size()) {
        return;
    }
    const RetireLane previousLane = laneOf(slot);
    convo::publishAtomic(laneBySlot_[slot], toRaw(RetireLane::Reclaim), std::memory_order_release);
    (void)convo::fetchAddAtomic(laneCounters_[static_cast<std::size_t>(RetireLane::Reclaim)], static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
    (void)convo::fetchAddAtomic(totalTransitions_, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
    if (previousLane == RetireLane::Quarantine)
    {
        const auto resident = convo::consumeAtomic(quarantineResidentCount_, std::memory_order_acquire);
        if (resident > 0)
            convo::fetchSubAtomic(quarantineResidentCount_, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
    }
}

void RetireRuntimeEx::quarantine(std::uint32_t slot) {
    if (slot >= laneBySlot_.size()) {
        return;
    }
    const RetireLane previousLane = laneOf(slot);
    convo::publishAtomic(laneBySlot_[slot], toRaw(RetireLane::Quarantine), std::memory_order_release);
    (void)convo::fetchAddAtomic(laneCounters_[static_cast<std::size_t>(RetireLane::Quarantine)], static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
    (void)convo::fetchAddAtomic(totalTransitions_, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
    if (previousLane != RetireLane::Quarantine)
        convo::fetchAddAtomic(quarantineResidentCount_, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
}

void RetireRuntimeEx::setEpochMode(EpochMode mode) noexcept {
    convo::publishAtomic(epochModeRaw_, static_cast<std::uint32_t>(mode), std::memory_order_release);
}

EpochMode RetireRuntimeEx::getEpochMode() const noexcept {
    return epochModeFromRaw(convo::consumeAtomic(epochModeRaw_, std::memory_order_acquire));
}

void RetireRuntimeEx::setRollbackMode(EpochMode mode) noexcept {
    convo::publishAtomic(rollbackModeRaw_, toRaw(mode), std::memory_order_release);
}

EpochMode RetireRuntimeEx::getRollbackMode() const noexcept {
    return epochModeFromRaw(convo::consumeAtomic(rollbackModeRaw_, std::memory_order_acquire));
}

void RetireRuntimeEx::setRollbackFlags(bool globalEnabled,
                                       bool publicationOnlyEnabled,
                                       bool crossfadeOnlyEnabled,
                                       bool retirePathOnlyEnabled) noexcept
{
    convo::publishAtomic(rollbackGlobalEnabled_, globalEnabled, std::memory_order_release);
    convo::publishAtomic(rollbackPublicationOnlyEnabled_, publicationOnlyEnabled, std::memory_order_release);
    convo::publishAtomic(rollbackCrossfadeOnlyEnabled_, crossfadeOnlyEnabled, std::memory_order_release);
    convo::publishAtomic(rollbackRetirePathOnlyEnabled_, retirePathOnlyEnabled, std::memory_order_release);
    convo::publishAtomic(rollbackReady_, (globalEnabled && retirePathOnlyEnabled), std::memory_order_release);
}

RollbackFlagDescriptor RetireRuntimeEx::describeRollbackFlags() const noexcept
{
    return RollbackFlagDescriptor{
        .globalEnabled = convo::consumeAtomic(rollbackGlobalEnabled_, std::memory_order_acquire),
        .publicationOnlyEnabled = convo::consumeAtomic(rollbackPublicationOnlyEnabled_, std::memory_order_acquire),
        .crossfadeOnlyEnabled = convo::consumeAtomic(rollbackCrossfadeOnlyEnabled_, std::memory_order_acquire),
        .retirePathOnlyEnabled = convo::consumeAtomic(rollbackRetirePathOnlyEnabled_, std::memory_order_acquire)
    };
}

bool RetireRuntimeEx::canRollback() const noexcept {
    return convo::consumeAtomic(rollbackReady_, std::memory_order_acquire)
        && convo::consumeAtomic(rollbackGlobalEnabled_, std::memory_order_acquire)
        && convo::consumeAtomic(rollbackRetirePathOnlyEnabled_, std::memory_order_acquire);
}

void RetireRuntimeEx::requestRollback() noexcept {
    if (!canRollback()) {
        return;
    }

    setEpochMode(getRollbackMode());
}

EpochStrategyDescriptor RetireRuntimeEx::describeEpochStrategy() const noexcept {
    return EpochStrategyDescriptor{
        .activeMode = getEpochMode(),
        .rollbackMode = getRollbackMode(),
        .rollbackReady = canRollback()
    };
}

std::uint64_t RetireRuntimeEx::getQuarantineResidentCount() const noexcept {
    return convo::consumeAtomic(quarantineResidentCount_, std::memory_order_acquire);
}

RetireLane RetireRuntimeEx::laneOf(std::uint32_t slot) const noexcept {
    if (slot >= laneBySlot_.size()) {
        return RetireLane::Quarantine;
    }
    return fromRaw(convo::consumeAtomic(laneBySlot_[slot], std::memory_order_acquire));
}

void RetireRuntimeEx::emitRetireTimeline(const std::filesystem::path& outputPath) const {
    std::error_code ec;
    std::filesystem::create_directories(outputPath.parent_path(), ec);

    std::ofstream file(outputPath, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        return;
    }

    const auto strategy = describeEpochStrategy();
    const auto rollbackFlags = describeRollbackFlags();

    file << "{\n";
    file << "  \"schema\": \"retire_timeline_v1\",\n";
    file << "  \"epochMode\": \"" << epochModeName(strategy.activeMode) << "\",\n";
    file << "  \"rollbackMode\": \"" << epochModeName(strategy.rollbackMode) << "\",\n";
    file << "  \"rollbackReady\": " << (strategy.rollbackReady ? "true" : "false") << ",\n";
    file << "  \"rollbackFlags\": {\n";
    file << "    \"global\": " << (rollbackFlags.globalEnabled ? "true" : "false") << ",\n";
    file << "    \"publicationOnly\": " << (rollbackFlags.publicationOnlyEnabled ? "true" : "false") << ",\n";
    file << "    \"crossfadeOnly\": " << (rollbackFlags.crossfadeOnlyEnabled ? "true" : "false") << ",\n";
    file << "    \"retirePathOnly\": " << (rollbackFlags.retirePathOnlyEnabled ? "true" : "false") << "\n";
    file << "  },\n";
    file << "  \"totalTransitions\": "
         << convo::consumeAtomic(totalTransitions_, std::memory_order_acquire) << ",\n";
    file << "  \"laneCounters\": {\n";
    file << "    \"rtIntent\": " << convo::consumeAtomic(laneCounters_[static_cast<std::size_t>(RetireLane::RTIntent)], std::memory_order_acquire) << ",\n";
    file << "    \"coordination\": " << convo::consumeAtomic(laneCounters_[static_cast<std::size_t>(RetireLane::Coordination)], std::memory_order_acquire) << ",\n";
    file << "    \"epoch\": " << convo::consumeAtomic(laneCounters_[static_cast<std::size_t>(RetireLane::Epoch)], std::memory_order_acquire) << ",\n";
    file << "    \"reclaim\": " << convo::consumeAtomic(laneCounters_[static_cast<std::size_t>(RetireLane::Reclaim)], std::memory_order_acquire) << ",\n";
    file << "    \"quarantine\": " << convo::consumeAtomic(laneCounters_[static_cast<std::size_t>(RetireLane::Quarantine)], std::memory_order_acquire) << "\n";
    file << "  }\n";
    file << "}\n";
}

} // namespace convo::isr
