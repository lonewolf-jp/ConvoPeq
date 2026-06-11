#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>

namespace convo {
namespace isr {

/**
 * ISR 10層 Architecture Layer 8: Shutdown FSM
 * coordinated shutdown sequence と barrier transition
 */

/**
 * Shutdown phase
 */
enum class ShutdownPhase : uint8_t
{
    Running,
    AudioStopped,
    ObserverDrained,
    RetireClosed,
    EpochSettled,
    ReclaimComplete,
    VerifyDrained,    // ★ P3: 最終監査フェーズ
    TimedOut,
    Failed,
    ShutdownComplete
};

/**
 * Shutdown runtime FSM
 */
class ShutdownRuntime
{
public:
    ShutdownRuntime();
    ~ShutdownRuntime();

    // Initiate shutdown sequence
    void initiateShutdown();

    // Check current shutdown phase
    ShutdownPhase getPhase() const noexcept;

    // ★ P1-1: enum 順序非依存の terminal 判定
    static bool isTerminalPhase(ShutdownPhase p) noexcept {
        return p == ShutdownPhase::ShutdownComplete
            || p == ShutdownPhase::TimedOut
            || p == ShutdownPhase::Failed;
    }

    // ★ P1-1: TimedOut/Failed 上書き前の最終フェーズを取得（障害解析用）
    ShutdownPhase getLastNonTerminalPhase() const noexcept;

    // NonRT: advance shutdown phase
    void advancePhase() noexcept;
    bool transitionTo(ShutdownPhase target) noexcept;

    // RT: check if shutdown in progress
    bool isShutdownInProgress() const noexcept;

    // ★ P1-1: タイムアウト・異常終了を記録（transitionTo をバイパスして直接 store）
    void markTimedOut() noexcept;
    void markFailed() noexcept;

    // Emit final shutdown trace
    void emitShutdownTrace() const;

    // Update bounded teardown counters (SH-1..SH-4)
    void setBoundedTeardownCounters(uint32_t callbackCount,
                                    uint32_t activeCrossfade,
                                    uint32_t pendingRetire,
                                    uint32_t observerCount) noexcept;

    // SH-5/SH-6: detect callbacks/enqueue after stop transition
    void markLateCallback() noexcept;
    void markPostStopEnqueue() noexcept;

private:
    std::atomic<ShutdownPhase> phase_{ShutdownPhase::Running};
    // ★ P1-1: TimedOut/Failed 上書き前の最終フェーズ（障害解析用）
    std::atomic<ShutdownPhase> lastNonTerminalPhase_{ShutdownPhase::Running};
    std::atomic<uint32_t> transitionViolations_{0};
    std::atomic<uint32_t> sh1CallbackCount_{0};
    std::atomic<uint32_t> sh2ActiveCrossfade_{0};
    std::atomic<uint32_t> sh3PendingRetire_{0};
    std::atomic<uint32_t> sh4ObserverCount_{0};
    std::atomic<uint32_t> sh5LateCallbackCount_{0};
    std::atomic<uint32_t> sh6PostStopEnqueueCount_{0};
};

}  // namespace isr
}  // namespace convo
