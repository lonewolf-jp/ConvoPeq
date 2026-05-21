#include "ISRLifecycle.h"
#include "AtomicAccess.h"
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace convo {
namespace isr {

// ============================================================================
// LifecycleIsolationRuntime
// ============================================================================

LifecycleIsolationRuntime::LifecycleIsolationRuntime()
{
}

LifecycleIsolationRuntime::~LifecycleIsolationRuntime() = default;

LifecycleToken LifecycleIsolationRuntime::enterPrepare(int sampleRate, int blockSize)
{
    std::lock_guard<std::mutex> guard(nonRtGuard_);

    auto currentPhase = convo::consumeAtomic(phase_, std::memory_order_acquire);

    if (currentPhase == LifecyclePhase::Prepared) {
        const int currentRate = convo::consumeAtomic(lastPreparedSampleRate_, std::memory_order_acquire);
        const int currentBlock = convo::consumeAtomic(lastPreparedBlockSize_, std::memory_order_acquire);

        if (currentRate == sampleRate && currentBlock == blockSize) {
            (void)convo::fetchAddAtomic(duplicatePrepareCollapsed_, uint32_t{1}, std::memory_order_acq_rel);
            uint64_t epochId = convo::consumeAtomic(epochCounter_, std::memory_order_acquire);
            return LifecycleToken{ epochId, LifecyclePhase::Prepared };
        }
    }

    validateTransition(currentPhase, LifecyclePhase::Preparing);

    auto newPhase = transitionTo(LifecyclePhase::Preparing);
    convo::publishAtomic(lastPreparedSampleRate_, sampleRate, std::memory_order_release);
    convo::publishAtomic(lastPreparedBlockSize_, blockSize, std::memory_order_release);
    uint64_t epochId = convo::consumeAtomic(epochCounter_, std::memory_order_acquire);

    return LifecycleToken{ epochId, newPhase };
}

void LifecycleIsolationRuntime::leavePrepare(LifecycleToken token)
{
    std::lock_guard<std::mutex> guard(nonRtGuard_);

    auto currentPhase = convo::consumeAtomic(phase_, std::memory_order_acquire);
    if (currentPhase != LifecyclePhase::Preparing) {
        std::abort();
    }

    transitionTo(LifecyclePhase::Prepared);
}

LifecycleToken LifecycleIsolationRuntime::enterAudioCallback()
{
    auto currentPhase = convo::consumeAtomic(phase_, std::memory_order_acquire);
    if (currentPhase == LifecyclePhase::Releasing
        || currentPhase == LifecyclePhase::Released
        || currentPhase == LifecyclePhase::Shutdown) {
        (void)convo::fetchAddAtomic(hostChaosViolations_, uint32_t{1}, std::memory_order_acq_rel);
        std::abort();
    }

    if (currentPhase != LifecyclePhase::Prepared && currentPhase != LifecyclePhase::AudioRunning) {
        std::abort();
    }

    transitionTo(LifecyclePhase::AudioRunning);
    uint64_t epochId = convo::consumeAtomic(epochCounter_, std::memory_order_acquire);

    return LifecycleToken{ epochId, LifecyclePhase::AudioRunning };
}

void LifecycleIsolationRuntime::leaveAudioCallback(LifecycleToken token)
{
    auto currentPhase = convo::consumeAtomic(phase_, std::memory_order_acquire);
    if (currentPhase != LifecyclePhase::AudioRunning) {
        std::abort();
    }

    transitionTo(LifecyclePhase::Prepared);
}

LifecycleToken LifecycleIsolationRuntime::enterRelease()
{
    std::lock_guard<std::mutex> guard(nonRtGuard_);

    auto currentPhase = convo::consumeAtomic(phase_, std::memory_order_acquire);
    if (currentPhase == LifecyclePhase::Uninitialized || currentPhase == LifecyclePhase::Preparing) {
        (void)convo::fetchAddAtomic(hostChaosViolations_, uint32_t{1}, std::memory_order_acq_rel);
        std::abort();
    }

    validateTransition(currentPhase, LifecyclePhase::Releasing);

    auto newPhase = transitionTo(LifecyclePhase::Releasing);
    uint64_t epochId = convo::consumeAtomic(epochCounter_, std::memory_order_acquire);

    return LifecycleToken{ epochId, newPhase };
}

void LifecycleIsolationRuntime::leaveRelease(LifecycleToken token)
{
    std::lock_guard<std::mutex> guard(nonRtGuard_);

    auto currentPhase = convo::consumeAtomic(phase_, std::memory_order_acquire);
    if (currentPhase != LifecyclePhase::Releasing) {
        std::abort();
    }

    transitionTo(LifecyclePhase::Released);
}

void LifecycleIsolationRuntime::shutdown()
{
    std::lock_guard<std::mutex> guard(nonRtGuard_);

    auto currentPhase = convo::consumeAtomic(phase_, std::memory_order_acquire);
    if (currentPhase == LifecyclePhase::Shutdown) {
        std::abort();
    }

    transitionTo(LifecyclePhase::Shutdown);
}

LifecyclePhase LifecycleIsolationRuntime::current() const noexcept
{
    return convo::consumeAtomic(phase_, std::memory_order_acquire);
}

void LifecycleIsolationRuntime::assertAudioRunning() const noexcept
{
    auto currentPhase = convo::consumeAtomic(phase_, std::memory_order_acquire);
    if (currentPhase != LifecyclePhase::AudioRunning) {
        std::abort();
    }
}

void LifecycleIsolationRuntime::validateTransition(LifecyclePhase from, LifecyclePhase to)
{
    bool valid = false;

    switch (from) {
    case LifecyclePhase::Uninitialized:
        valid = (to == LifecyclePhase::Preparing || to == LifecyclePhase::Shutdown);
        break;
    case LifecyclePhase::Preparing:
        valid = (to == LifecyclePhase::Prepared || to == LifecyclePhase::Shutdown);
        break;
    case LifecyclePhase::Prepared:
        valid = (to == LifecyclePhase::Preparing || to == LifecyclePhase::AudioRunning || to == LifecyclePhase::Releasing
                 || to == LifecyclePhase::Shutdown);
        break;
    case LifecyclePhase::AudioRunning:
        valid = (to == LifecyclePhase::Prepared || to == LifecyclePhase::Shutdown);
        break;
    case LifecyclePhase::Releasing:
        valid = (to == LifecyclePhase::Released || to == LifecyclePhase::Shutdown);
        break;
    case LifecyclePhase::Released:
        valid = (to == LifecyclePhase::Preparing || to == LifecyclePhase::Shutdown);
        break;
    case LifecyclePhase::Shutdown:
        valid = false; // once-only
        break;
    }

    if (!valid) {
        std::abort();
    }
}

LifecyclePhase LifecycleIsolationRuntime::transitionTo(LifecyclePhase next)
{
    auto previous = convo::consumeAtomic(phase_, std::memory_order_acquire);
    validateTransition(previous, next);

    convo::publishAtomic(phase_, next, std::memory_order_release);

    // Record transition
    {
        std::lock_guard<std::mutex> guard(traceGuard_);
        uint64_t now_ns = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        uint64_t epochId = convo::consumeAtomic(epochCounter_, std::memory_order_acquire);
        transitions_.push_back({ previous, next, epochId, now_ns });
    }

    // Increment epoch on key transitions
    if (next == LifecyclePhase::Prepared || next == LifecyclePhase::Released) {
        (void)convo::fetchAddAtomic(epochCounter_, uint64_t{1}, std::memory_order_release);
    }

    return next;
}

void LifecycleIsolationRuntime::emitPhaseTrace(const std::filesystem::path& outputPath)
{
    std::lock_guard<std::mutex> guard(traceGuard_);

    std::stringstream ss;
    ss << "{\n";
    ss << "  \"schema\": \"lifecycle_phase_trace_v1\",\n";
    ss << "  \"transitions\": [\n";

    for (size_t i = 0; i < transitions_.size(); ++i) {
        const auto& t = transitions_[i];
        ss << "    {\n";
        ss << "      \"from\": \"" << static_cast<int>(t.from) << "\",\n";
        ss << "      \"to\": \"" << static_cast<int>(t.to) << "\",\n";
        ss << "      \"epochId\": " << t.epochId << ",\n";
        ss << "      \"timestamp_ns\": " << t.timestamp_ns << "\n";
        ss << "    }";
        if (i + 1 < transitions_.size()) {
            ss << ",";
        }
        ss << "\n";
    }

    ss << "  ],\n";
    ss << "  \"invariant_violations\": {\n";
    ss << "    \"hostChaosViolations\": " << convo::consumeAtomic(hostChaosViolations_, std::memory_order_acquire) << ",\n";
    ss << "    \"duplicatePrepareCollapsed\": " << convo::consumeAtomic(duplicatePrepareCollapsed_, std::memory_order_acquire) << "\n";
    ss << "  }\n";
    ss << "}\n";

    std::ofstream ofs(outputPath);
    ofs << ss.str();
    ofs.close();
}

// ============================================================================
// LifecycleBarrierRuntime
// ============================================================================

LifecycleBarrierRuntime::LifecycleBarrierRuntime(LifecycleIsolationRuntime& lifecycleRuntime)
    : lifecycleRuntime_(lifecycleRuntime)
{
}

void LifecycleBarrierRuntime::publishPreparedBarrier()
{
    const auto phase = lifecycleRuntime_.current();
    if (phase != LifecyclePhase::Prepared && phase != LifecyclePhase::AudioRunning) {
        std::abort();
    }
}

void LifecycleBarrierRuntime::publishReleasingBarrier()
{
    const auto phase = lifecycleRuntime_.current();
    if (phase != LifecyclePhase::Releasing && phase != LifecyclePhase::Released) {
        std::abort();
    }
}

void LifecycleBarrierRuntime::publishShutdownBarrier()
{
    const auto phase = lifecycleRuntime_.current();
    if (phase != LifecyclePhase::Shutdown) {
        std::abort();
    }
}

} // namespace isr
} // namespace convo
