#include "ISRRetire.h"
#include "AtomicAccess.h"

namespace convo {
namespace isr {

void RetireRuntime::emitRetireIntent(const RetireIntent& intent) noexcept
{
    uint64_t tail = convo::consumeAtomic(retireIntentTail_, std::memory_order_relaxed);
    uint64_t nextTail = (tail + 1) % RETIRE_INTENT_QUEUE_SIZE;

    uint64_t head = convo::consumeAtomic(retireIntentHead_, std::memory_order_acquire);
    if (nextTail == head) {
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

    convo::publishAtomic(retireIntentHead_, head, std::memory_order_release);
    return result;
}

void RetireRuntime::acknowledgeRetireCoordination(const RetireIntent& intent)
{
    const auto idx = static_cast<std::size_t>(intent.dspSlot % RETIRE_INTENT_QUEUE_SIZE);
    convo::publishAtomic(acknowledgeGeneration_[idx], intent.generation, std::memory_order_release);
    (void)convo::fetchAddAtomic(acknowledgedCount_, uint64_t{1}, std::memory_order_acq_rel);
}

}  // namespace isr
}  // namespace convo
