#include "ISRRetire.h"
#include "AtomicAccess.h"

#include <algorithm>

namespace convo {
namespace isr {

void RetireRuntime::emitRetireIntent(const RetireIntent& intent) noexcept
{
    uint64_t tail = convo::consumeAtomic(retireIntentTail_, std::memory_order_relaxed);
    uint64_t nextTail = (tail + 1) % RETIRE_INTENT_QUEUE_SIZE;

    uint64_t head = convo::consumeAtomic(retireIntentHead_, std::memory_order_acquire);
    if (nextTail == head) {
        (void)convo::fetchAddAtomic(overflowCount_, uint64_t{1}, std::memory_order_acq_rel);
        (void)convo::fetchAddAtomic(droppedIntentCount_, uint64_t{1}, std::memory_order_acq_rel);
        return;
    }

    retireIntentQueue_[tail] = intent;
    convo::publishAtomic(retireIntentTail_, nextTail, std::memory_order_release);
}

void RetireRuntime::emitRetireIntentRT(const RetireIntent& intent) noexcept
{
    emitRetireIntent(intent);
}

std::vector<RetireIntent> RetireRuntime::dequeuePendingRetireIntents() noexcept
{
    std::vector<RetireIntent> result;

    uint64_t head = convo::consumeAtomic(retireIntentHead_, std::memory_order_acquire);
    uint64_t tail = convo::consumeAtomic(retireIntentTail_, std::memory_order_acquire);

    while (head != tail) {
        result.push_back(retireIntentQueue_[head]);
        head = (head + 1) % RETIRE_INTENT_QUEUE_SIZE;
    }

    std::stable_sort(result.begin(), result.end(), [](const RetireIntent& lhs, const RetireIntent& rhs) noexcept {
        if (lhs.retireEpoch != rhs.retireEpoch)
            return lhs.retireEpoch < rhs.retireEpoch;

        if (lhs.generation != rhs.generation)
            return lhs.generation < rhs.generation;

        return lhs.dspSlot < rhs.dspSlot;
    });

    convo::publishAtomic(retireIntentHead_, head, std::memory_order_release);
    return result;
}

std::uint64_t RetireRuntime::pendingIntentCount() const noexcept
{
    const uint64_t head = convo::consumeAtomic(retireIntentHead_, std::memory_order_acquire);
    const uint64_t tail = convo::consumeAtomic(retireIntentTail_, std::memory_order_acquire);
    if (tail >= head)
        return tail - head;
    return (RETIRE_INTENT_QUEUE_SIZE - head) + tail;
}

std::uint64_t RetireRuntime::overflowCount() const noexcept
{
    return convo::consumeAtomic(overflowCount_, std::memory_order_acquire);
}

std::uint64_t RetireRuntime::droppedIntentCount() const noexcept
{
    return convo::consumeAtomic(droppedIntentCount_, std::memory_order_acquire);
}

void RetireRuntime::acknowledgeRetireCoordination(const RetireIntent& intent)
{
    const auto idx = static_cast<std::size_t>(intent.dspSlot % RETIRE_INTENT_QUEUE_SIZE);
    convo::publishAtomic(acknowledgeGeneration_[idx], intent.generation, std::memory_order_release);
    (void)convo::fetchAddAtomic(acknowledgedCount_, uint64_t{1}, std::memory_order_acq_rel);
}

}  // namespace isr
}  // namespace convo
