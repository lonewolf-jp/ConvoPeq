#pragma once

#include <atomic>
#include <cstdint>
#include <vector>
#include <array>

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
    uint32_t generation;
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

    // NonRT: acknowledge retire coordination
    void acknowledgeRetireCoordination(const RetireIntent& intent);

private:
    // Lock-free queue (using atomics)
    std::atomic<uint64_t> retireIntentHead_{0};
    std::atomic<uint64_t> retireIntentTail_{0};

    static constexpr size_t RETIRE_INTENT_QUEUE_SIZE = 256;
    RetireIntent retireIntentQueue_[RETIRE_INTENT_QUEUE_SIZE];
    std::array<std::atomic<uint32_t>, RETIRE_INTENT_QUEUE_SIZE> acknowledgeGeneration_{};
    std::atomic<uint64_t> acknowledgedCount_{0};
};

}  // namespace isr
}  // namespace convo
