#pragma once

#include <atomic>
#include <cstdint>
#include <vector>
#include <array>
#include <mutex>

namespace convo {
namespace isr {

/**
 * ISR 10層 Architecture Layer 7: RetireIntent
 * RT から retire intent を emit し、NonRT側で coordination
 */

/**
 * Retire intention descriptor
 */
struct RetireIntent
{
    uint32_t dspSlot;
    uint64_t generation;  // ★ B-1: 64bit化
    uint64_t retireEpoch;
    bool isValid;
};

/**
 * Retire runtime
 */
class RetireRuntime
{
public:
    // Generic helper used internally / non-RT paths.
    void emitRetireIntent(const RetireIntent& intent) noexcept;

    // Preferred API for runtime retire intent publication from commit path.
    void emitRetireIntentRT(const RetireIntent& intent) noexcept;

    // NonRT: dequeue retire intents
    std::vector<RetireIntent> dequeuePendingRetireIntents() noexcept;

    [[nodiscard]] std::uint64_t pendingIntentCount() const noexcept;
    [[nodiscard]] std::uint64_t overflowCount() const noexcept;
    [[nodiscard]] std::uint64_t droppedIntentCount() const noexcept;

    // ★ P1: Fallback queue metrics
    [[nodiscard]] std::size_t fallbackOccupancy() const noexcept;
    [[nodiscard]] std::size_t fallbackHighWatermark() const noexcept;
    [[nodiscard]] std::uint64_t fallbackOverflowCount() const noexcept;

    // ★ C-1: overflow 継続時間追跡
    [[nodiscard]] std::uint64_t overflowStartTimestamp() const noexcept;
    [[nodiscard]] std::uint64_t lastOverflowTicks() const noexcept;
    [[nodiscard]] std::uint64_t overflowWindowCounter() const noexcept;
    [[nodiscard]] std::uint64_t lastOverflowWindowCount() const noexcept;

    // NonRT: acknowledge retire coordination
    void acknowledgeRetireCoordination(const RetireIntent& intent);

private:
    // Lock-free queue (using atomics)
    std::atomic<uint64_t> retireIntentHead_{0};
    std::atomic<uint64_t> retireIntentTail_{0};

    static constexpr size_t RETIRE_INTENT_QUEUE_SIZE = 256;
    RetireIntent retireIntentQueue_[RETIRE_INTENT_QUEUE_SIZE];
    std::array<std::atomic<uint64_t>, RETIRE_INTENT_QUEUE_SIZE> acknowledgeGeneration_{};  // ★ B-1: 64bit化
    std::atomic<uint64_t> acknowledgedCount_{0};
    std::atomic<uint64_t> overflowCount_{0};
    std::atomic<uint64_t> droppedIntentCount_{0};

    // ★ P1: Bounded Fallback Queue (mutex-protected, 上限 retireHighWatermark*2)
    static constexpr size_t FALLBACK_QUEUE_CAPACITY = 4096;
    RetireIntent fallbackQueue_[FALLBACK_QUEUE_CAPACITY];
    std::atomic<size_t> fallbackCount_{0};
    std::atomic<size_t> fallbackHighWatermark_{0};
    std::atomic<uint64_t> fallbackOverflowCount_{0};
    mutable std::mutex fallbackMutex_;

    // ★ C-1: overflow 継続時間追跡
    std::atomic<uint64_t> lastOverflowTicks_{0};
    std::atomic<uint64_t> overflowStartTimestamp_{0};
    std::atomic<uint64_t> overflowWindowCounter_{0};
    std::atomic<uint64_t> lastOverflowWindowCount_{0};
    std::atomic<uint64_t> lastOverflowWindowResetTicks_{0};
};

}  // namespace isr
}  // namespace convo
