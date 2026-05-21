#pragma once

#include <atomic>
#include <array>
#include <cstdint>
#include <vector>
#include <filesystem>
#include <mutex>

namespace convo {
namespace isr {

/**
 * ISR 10層 Architecture Layer 6: Happens-Before Graph
 * RT/NonRT boundary の memory order guarantee を記述・検証
 */

/**
 * HB edge の方向性（from -> to）
 */
struct HBNode
{
    uint32_t nodeId;
    uint64_t epochId;
};

/**
 * HB edge（transition の memory order）
 */
struct HBEdge
{
    HBNode fromNode;
    HBNode toNode;
    int memoryOrder;  // std::memory_order value
};

/**
 * HB trace event
 */
struct HBTraceEvent
{
    uint64_t timestamp;
    HBEdge edge;
    bool isRelease;
    bool isAcquire;
};

/**
 * HB trace runtime（artifact emission）
 */
class HBTraceRuntime
{
public:
    // Record HB edge
    void recordEdge(const HBEdge& edge) noexcept;

    // Emit HB trace to file
    void emitTrace(const std::filesystem::path& outputPath) const;
    bool validateMonotonicTimestamps() const noexcept;
    std::vector<HBTraceEvent> snapshotEvents() const;

private:
    std::vector<HBTraceEvent> traceEvents_;
    mutable std::mutex traceMutex_;
    std::atomic<uint64_t> eventCount_{0};
};

/**
 * HB runtime core (verification)
 */
class HBRuntimeCore
{
public:
    // Verify HB guarantee between two nodes
    bool verifyHBGuarantee(const HBNode& from, const HBNode& to) const noexcept;
    void registerEdge(const HBEdge& edge);

private:
    std::vector<HBEdge> edges_;
};

enum class HBReorderScenario : uint8_t
{
    ForcedReorder = 0,
    EpochLag,
    RetireDelay,
    ObserveRace
};

struct HBScenarioResult
{
    const char* name;
    bool passed;
};

/**
 * HB verifier (CI/Debug build only)
 */
class HBVerifierRuntime
{
public:
    // Full validation of HB graph consistency
    bool validateHBGraph() const noexcept;
    void emitViolationReport(const std::filesystem::path& outputPath) const;
    bool simulateReorderScenario(HBReorderScenario scenario) const noexcept;
    std::array<HBScenarioResult, 4> runScenarioSuite() const noexcept;

    void setTraceRuntime(const HBTraceRuntime* traceRuntime) noexcept;

private:
    bool hasTraceRuntimeBound() const noexcept;
    bool hasAnyEvent() const noexcept;
    bool checkEpochOrder() const noexcept;
    bool checkBoundedEpochLag(std::uint64_t maxLag) const noexcept;
    bool checkRetireDelayBound(std::uint64_t maxNs) const noexcept;
    bool checkObserveRaceFree() const noexcept;

    const HBTraceRuntime* traceRuntime_ = nullptr;
};

}  // namespace isr
}  // namespace convo
