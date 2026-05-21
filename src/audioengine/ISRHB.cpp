#include "ISRHB.h"
#include "AtomicAccess.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <unordered_map>

namespace convo {
namespace isr {

void HBTraceRuntime::recordEdge(const HBEdge& edge) noexcept
{
    const auto nowNs = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());

    HBTraceEvent event{
        .timestamp = nowNs,
        .edge = edge,
        .isRelease = edge.memoryOrder == static_cast<int>(std::memory_order_release)
                  || edge.memoryOrder == static_cast<int>(std::memory_order_acq_rel),
        .isAcquire = edge.memoryOrder == static_cast<int>(std::memory_order_acquire)
                  || edge.memoryOrder == static_cast<int>(std::memory_order_acq_rel)
    };

    {
        std::lock_guard<std::mutex> lock(traceMutex_);
        traceEvents_.push_back(event);
    }

    (void)convo::fetchAddAtomic(eventCount_, uint64_t{1}, std::memory_order_relaxed);
}

void HBTraceRuntime::emitTrace(const std::filesystem::path& outputPath) const
{
    std::error_code ec;
    std::filesystem::create_directories(outputPath.parent_path(), ec);

    std::ofstream file(outputPath, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        return;
    }

    std::vector<HBTraceEvent> snapshot;
    {
        std::lock_guard<std::mutex> lock(traceMutex_);
        snapshot = traceEvents_;
    }

    file << "{\n";
    file << "  \"schema\": \"hb_trace_v1\",\n";
    file << "  \"eventCount\": "
         << convo::consumeAtomic(eventCount_, std::memory_order_acquire)
         << ",\n";
    file << "  \"events\": [\n";

    for (std::size_t i = 0; i < snapshot.size(); ++i) {
        const auto& e = snapshot[i];
        file << "    {\"ts\": " << e.timestamp
             << ", \"from\": " << e.edge.fromNode.nodeId
             << ", \"to\": " << e.edge.toNode.nodeId
             << ", \"fromEpoch\": " << e.edge.fromNode.epochId
             << ", \"toEpoch\": " << e.edge.toNode.epochId
             << ", \"mo\": " << e.edge.memoryOrder
             << ", \"release\": " << (e.isRelease ? "true" : "false")
             << ", \"acquire\": " << (e.isAcquire ? "true" : "false")
             << "}";
        if (i + 1u < snapshot.size()) {
            file << ",";
        }
        file << "\n";
    }

    file << "  ]\n";
    file << "}\n";
}

bool HBTraceRuntime::validateMonotonicTimestamps() const noexcept
{
    std::lock_guard<std::mutex> lock(traceMutex_);
    if (traceEvents_.empty()) {
        return true;
    }

    uint64_t prev = traceEvents_.front().timestamp;
    for (std::size_t i = 1; i < traceEvents_.size(); ++i) {
        if (traceEvents_[i].timestamp < prev) {
            return false;
        }
        prev = traceEvents_[i].timestamp;
    }

    return true;
}

std::vector<HBTraceEvent> HBTraceRuntime::snapshotEvents() const
{
    std::lock_guard<std::mutex> lock(traceMutex_);
    return traceEvents_;
}

bool HBRuntimeCore::verifyHBGuarantee(const HBNode& from, const HBNode& to) const noexcept
{
    if (from.epochId > to.epochId) {
        return false;
    }

    if (from.nodeId == to.nodeId) {
        return true;
    }

    for (const auto& edge : edges_) {
        if (edge.fromNode.nodeId == from.nodeId && edge.toNode.nodeId == to.nodeId
            && edge.fromNode.epochId <= edge.toNode.epochId) {
            return true;
        }
    }

    return false;
}

void HBRuntimeCore::registerEdge(const HBEdge& edge)
{
    edges_.push_back(edge);
}

bool HBVerifierRuntime::validateHBGraph() const noexcept
{
    if (!hasTraceRuntimeBound()) {
        return true;
    }

    return traceRuntime_->validateMonotonicTimestamps() && checkEpochOrder();
}

void HBVerifierRuntime::emitViolationReport(const std::filesystem::path& outputPath) const
{
    std::error_code ec;
    std::filesystem::create_directories(outputPath.parent_path(), ec);

    std::ofstream file(outputPath, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        return;
    }

    const bool valid = validateHBGraph();
    const auto scenarios = runScenarioSuite();

    file << "{\n";
    file << "  \"schema\": \"hb_violation_report_v1\",\n";
    file << "  \"status\": \"" << (valid ? "ok" : "violation") << "\",\n";
    file << "  \"violations\": [";
    bool wroteViolation = false;
    if (!valid) {
        file << "\"non-monotonic timestamp detected\"";
        wroteViolation = true;
    }

    for (const auto& scenario : scenarios) {
        if (!scenario.passed) {
            if (wroteViolation) {
                file << ",";
            }
            file << "\"scenario_failed:" << scenario.name << "\"";
            wroteViolation = true;
        }
    }

    file << "],\n";
    file << "  \"scenarioResults\": [\n";
    for (std::size_t i = 0; i < scenarios.size(); ++i) {
        const auto& scenario = scenarios[i];
        file << "    {\"name\": \"" << scenario.name << "\", \"result\": \""
             << (scenario.passed ? "pass" : "fail") << "\"}";
        if (i + 1u < scenarios.size()) {
            file << ",";
        }
        file << "\n";
    }
    file << "  ]\n";
    file << "}\n";
}

bool HBVerifierRuntime::simulateReorderScenario(HBReorderScenario scenario) const noexcept
{
    if (!hasTraceRuntimeBound() || !hasAnyEvent()) {
        return false;
    }

    const bool valid = validateHBGraph();
    if (!valid) {
        return false;
    }

    switch (scenario) {
    case HBReorderScenario::ForcedReorder:
        return checkEpochOrder();
    case HBReorderScenario::EpochLag:
        return checkBoundedEpochLag(4u);
    case HBReorderScenario::RetireDelay:
        return checkRetireDelayBound(50'000'000u);
    case HBReorderScenario::ObserveRace:
        return checkObserveRaceFree();
    default:
        return false;
    }
}

std::array<HBScenarioResult, 4> HBVerifierRuntime::runScenarioSuite() const noexcept
{
    return {{
        {"forced_reorder", simulateReorderScenario(HBReorderScenario::ForcedReorder)},
        {"epoch_lag", simulateReorderScenario(HBReorderScenario::EpochLag)},
        {"retire_delay", simulateReorderScenario(HBReorderScenario::RetireDelay)},
        {"observe_race", simulateReorderScenario(HBReorderScenario::ObserveRace)}
    }};
}

void HBVerifierRuntime::setTraceRuntime(const HBTraceRuntime* traceRuntime) noexcept
{
    traceRuntime_ = traceRuntime;
}

bool HBVerifierRuntime::hasTraceRuntimeBound() const noexcept
{
    return traceRuntime_ != nullptr;
}

bool HBVerifierRuntime::hasAnyEvent() const noexcept
{
    if (!hasTraceRuntimeBound()) {
        return false;
    }

    const auto events = traceRuntime_->snapshotEvents();
    return !events.empty();
}

bool HBVerifierRuntime::checkEpochOrder() const noexcept
{
    if (!hasTraceRuntimeBound()) {
        return false;
    }

    const auto events = traceRuntime_->snapshotEvents();
    for (const auto& e : events) {
        if (e.edge.toNode.epochId < e.edge.fromNode.epochId) {
            return false;
        }
    }

    return true;
}

bool HBVerifierRuntime::checkBoundedEpochLag(std::uint64_t maxLag) const noexcept
{
    if (!hasTraceRuntimeBound()) {
        return false;
    }

    const auto events = traceRuntime_->snapshotEvents();
    for (const auto& e : events) {
        if (e.edge.toNode.epochId >= e.edge.fromNode.epochId) {
            if ((e.edge.toNode.epochId - e.edge.fromNode.epochId) > maxLag) {
                return false;
            }
        }
    }

    return true;
}

bool HBVerifierRuntime::checkRetireDelayBound(std::uint64_t maxNs) const noexcept
{
    if (!hasTraceRuntimeBound()) {
        return false;
    }

    const auto events = traceRuntime_->snapshotEvents();
    if (events.size() < 2u) {
        return true;
    }

    for (std::size_t i = 1; i < events.size(); ++i) {
        const auto prev = events[i - 1u].timestamp;
        const auto curr = events[i].timestamp;
        if (curr > prev && (curr - prev) > maxNs) {
            return false;
        }
    }

    return true;
}

bool HBVerifierRuntime::checkObserveRaceFree() const noexcept
{
    if (!hasTraceRuntimeBound()) {
        return false;
    }

    const auto events = traceRuntime_->snapshotEvents();
    std::unordered_map<std::uint64_t, std::uint32_t> sinkByEpoch;

    for (const auto& e : events) {
        const std::uint64_t key = (static_cast<std::uint64_t>(e.edge.toNode.nodeId) << 32u)
            | static_cast<std::uint64_t>(e.edge.toNode.epochId & 0xFFFFFFFFu);
        const std::uint32_t from = e.edge.fromNode.nodeId;

        const auto iter = sinkByEpoch.find(key);
        if (iter == sinkByEpoch.end()) {
            sinkByEpoch.emplace(key, from);
            continue;
        }

        if (iter->second != from) {
            return false;
        }
    }

    return true;
}

}  // namespace isr
}  // namespace convo
