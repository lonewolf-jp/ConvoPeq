#include "ISRShutdown.h"
#include "AtomicAccess.h"
#include "RuntimeDrainAudit.h"  // ★ P2-B: getPrimaryBlockingReason

#include <filesystem>
#include <fstream>

namespace convo {
namespace isr {

ShutdownRuntime::ShutdownRuntime() = default;
ShutdownRuntime::~ShutdownRuntime() = default;

void ShutdownRuntime::initiateShutdown()
{
    transitionTo(ShutdownPhase::AudioStopped);
}

ShutdownPhase ShutdownRuntime::getPhase() const noexcept
{
    return convo::consumeAtomic(phase_, std::memory_order_acquire);
}

ShutdownPhase ShutdownRuntime::getLastNonTerminalPhase() const noexcept
{
    return convo::consumeAtomic(lastNonTerminalPhase_, std::memory_order_acquire);
}

ShutdownBlockingReason ShutdownRuntime::getBlockingReason() const noexcept
{
    return convo::consumeAtomic(blockingReason_, std::memory_order_acquire);
}

void ShutdownRuntime::markTimedOut(ShutdownBlockingReason reason) noexcept
{
    // ★ P2-B: 阻害要因を保存
    convo::publishAtomic(blockingReason_, reason, std::memory_order_release);
    // ★ P1-1: 現在の phase を保存してから上書き
    convo::publishAtomic(lastNonTerminalPhase_,
                         convo::consumeAtomic(phase_, std::memory_order_acquire),
                         std::memory_order_release);
    convo::publishAtomic(phase_, ShutdownPhase::TimedOut, std::memory_order_release);
}

void ShutdownRuntime::markFailed(ShutdownBlockingReason reason) noexcept
{
    // ★ P2-B: 阻害要因を保存
    convo::publishAtomic(blockingReason_, reason, std::memory_order_release);
    convo::publishAtomic(lastNonTerminalPhase_,
                         convo::consumeAtomic(phase_, std::memory_order_acquire),
                         std::memory_order_release);
    convo::publishAtomic(phase_, ShutdownPhase::Failed, std::memory_order_release);
}

void ShutdownRuntime::advancePhase() noexcept
{
    const ShutdownPhase current = convo::consumeAtomic(phase_, std::memory_order_acquire);

    // ★ P1-1: terminal 状態からは advance しない
    if (isTerminalPhase(current))
        return;

    ShutdownPhase next = current;
    switch (current) {
        case ShutdownPhase::Running:
            next = ShutdownPhase::AudioStopped;
            break;
        case ShutdownPhase::AudioStopped:
            next = ShutdownPhase::ObserverDrained;
            break;
        case ShutdownPhase::ObserverDrained:
            next = ShutdownPhase::RetireClosed;
            break;
        case ShutdownPhase::RetireClosed:
            next = ShutdownPhase::EpochSettled;
            break;
        case ShutdownPhase::EpochSettled:
            next = ShutdownPhase::ReclaimComplete;
            break;
        case ShutdownPhase::ReclaimComplete:
            // ★ C-2: CONVOPEQ_EMERGENCY_DRAIN 有効時のみ EmergencyDrain を経由
            next = ShutdownPhase::VerifyDrained;
            break;
        case ShutdownPhase::EmergencyDrain:        // ★ C-2
            next = ShutdownPhase::VerifyDrained;
            break;
        case ShutdownPhase::VerifyDrained:
            next = ShutdownPhase::ShutdownComplete;
            break;
        case ShutdownPhase::TimedOut:
        case ShutdownPhase::Failed:
        case ShutdownPhase::ShutdownComplete:
        default:
            return;
    }

    (void)transitionTo(next);
}

bool ShutdownRuntime::transitionTo(ShutdownPhase target) noexcept
{
    const auto current = convo::consumeAtomic(phase_, std::memory_order_acquire);
    const auto c = static_cast<int>(current);
    const auto t = static_cast<int>(target);

    // ★ P1-1: TimedOut(6)/Failed(7) を ShutdownComplete(8) の前に挿入したため、
    //   ReclaimComplete(5)→ShutdownComplete(8) のような terminal 状態をスキップする
    //   遷移を許可する。terminal 状態のみをスキップする遷移は許容。
    bool allowed = (t == c || t == c + 1);
    if (!allowed && t > c + 1) {
        // terminal 状態のみをスキップしているか確認
        allowed = true;
        for (int i = c + 1; i < t; ++i) {
            if (!isTerminalPhase(static_cast<ShutdownPhase>(i))) {
                allowed = false;
                break;
            }
        }
    }

    if (!allowed) {
        (void)convo::fetchAddAtomic(transitionViolations_, uint32_t{1}, std::memory_order_acq_rel);
        return false;
    }

    convo::publishAtomic(phase_, target, std::memory_order_release);
    return true;
}

bool ShutdownRuntime::isShutdownInProgress() const noexcept
{
    const ShutdownPhase current = convo::consumeAtomic(phase_, std::memory_order_acquire);
    return current != ShutdownPhase::Running && !isTerminalPhase(current);
}

void ShutdownRuntime::emitShutdownTrace() const
{
    const auto outputPath = std::filesystem::current_path() / "evidence" / "shutdown_trace.json";
    std::error_code ec;
    std::filesystem::create_directories(outputPath.parent_path(), ec);

    std::ofstream file(outputPath, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        return;
    }

    const auto phase = convo::consumeAtomic(phase_, std::memory_order_acquire);
    const auto violations = convo::consumeAtomic(transitionViolations_, std::memory_order_acquire);
    const auto sh1 = convo::consumeAtomic(sh1CallbackCount_, std::memory_order_acquire);
    const auto sh2 = convo::consumeAtomic(sh2ActiveCrossfade_, std::memory_order_acquire);
    const auto sh3 = convo::consumeAtomic(sh3PendingRetire_, std::memory_order_acquire);
    const auto sh4 = convo::consumeAtomic(sh4ObserverCount_, std::memory_order_acquire);
    const auto sh5 = convo::consumeAtomic(sh5LateCallbackCount_, std::memory_order_acquire);
    const auto sh6 = convo::consumeAtomic(sh6PostStopEnqueueCount_, std::memory_order_acquire);
    const auto reason = convo::consumeAtomic(blockingReason_, std::memory_order_acquire);  // ★ P2-B

    const char* phaseName = "Running";
    switch (phase) {
    case ShutdownPhase::Running: phaseName = "Running"; break;
    case ShutdownPhase::AudioStopped: phaseName = "AudioStopped"; break;
    case ShutdownPhase::ObserverDrained: phaseName = "ObserverDrained"; break;
    case ShutdownPhase::RetireClosed: phaseName = "RetireClosed"; break;
    case ShutdownPhase::EpochSettled: phaseName = "EpochSettled"; break;
    case ShutdownPhase::ReclaimComplete: phaseName = "ReclaimComplete"; break;
    case ShutdownPhase::EmergencyDrain: phaseName = "EmergencyDrain"; break;  // ★ C-2
    case ShutdownPhase::VerifyDrained: phaseName = "VerifyDrained"; break;
    case ShutdownPhase::TimedOut: phaseName = "TimedOut"; break;
    case ShutdownPhase::Failed: phaseName = "Failed"; break;
    case ShutdownPhase::ShutdownComplete: phaseName = "ShutdownComplete"; break;
    }

    const char* reasonName = "None";
    switch (reason) {
    case ShutdownBlockingReason::None: reasonName = "None"; break;
    case ShutdownBlockingReason::PendingPublication: reasonName = "PendingPublication"; break;
    case ShutdownBlockingReason::PendingRetire: reasonName = "PendingRetire"; break;
    case ShutdownBlockingReason::ActiveCrossfade: reasonName = "ActiveCrossfade"; break;
    case ShutdownBlockingReason::DeferredPublish: reasonName = "DeferredPublish"; break;
    case ShutdownBlockingReason::QuarantineResident: reasonName = "QuarantineResident"; break;
    case ShutdownBlockingReason::RouterPendingRetire: reasonName = "RouterPendingRetire"; break;
    case ShutdownBlockingReason::ReaderActive: reasonName = "ReaderActive"; break;
    case ShutdownBlockingReason::Unknown: reasonName = "Unknown"; break;
    }

    const bool boundedComplete = (sh1 == 0u && sh2 == 0u && sh3 == 0u && sh4 == 0u && sh5 == 0u && sh6 == 0u);

    file << "{\n";
    file << "  \"schema\": \"shutdown_trace_v2\",\n";
    file << "  \"phase\": " << static_cast<int>(phase) << ",\n";
    file << "  \"phaseName\": \"" << phaseName << "\",\n";
    file << "  \"blockingReason\": \"" << reasonName << "\",\n";  // ★ P2-B
    file << "  \"blockingReasonCode\": " << static_cast<int>(reason) << ",\n";
    file << "  \"transitionViolations\": " << violations << ",\n";
    file << "  \"sh1_callbackCount\": " << sh1 << ",\n";
    file << "  \"sh2_activeCrossfade\": " << sh2 << ",\n";
    file << "  \"sh3_pendingRetire\": " << sh3 << ",\n";
    file << "  \"sh4_observerCount\": " << sh4 << ",\n";
    file << "  \"sh5_lateCallbackCount\": " << sh5 << ",\n";
    file << "  \"sh6_postStopEnqueueCount\": " << sh6 << ",\n";
    file << "  \"verified\": " << ((violations == 0 && boundedComplete) ? "true" : "false") << "\n";
    file << "}\n";
}

void ShutdownRuntime::setBoundedTeardownCounters(uint32_t callbackCount,
                                                 uint32_t activeCrossfade,
                                                 uint32_t pendingRetire,
                                                 uint32_t observerCount) noexcept
{
    convo::publishAtomic(sh1CallbackCount_, callbackCount, std::memory_order_release);
    convo::publishAtomic(sh2ActiveCrossfade_, activeCrossfade, std::memory_order_release);
    convo::publishAtomic(sh3PendingRetire_, pendingRetire, std::memory_order_release);
    convo::publishAtomic(sh4ObserverCount_, observerCount, std::memory_order_release);
}

void ShutdownRuntime::markLateCallback() noexcept
{
    (void)convo::fetchAddAtomic(sh5LateCallbackCount_, uint32_t{1}, std::memory_order_acq_rel);
}

void ShutdownRuntime::markPostStopEnqueue() noexcept
{
    (void)convo::fetchAddAtomic(sh6PostStopEnqueueCount_, uint32_t{1}, std::memory_order_acq_rel);
}

}  // namespace isr
}  // namespace convo
