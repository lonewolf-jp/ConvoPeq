#include "ISRRetire.h"
#include "AtomicAccess.h"

#include <algorithm>
#include <chrono>

namespace convo {
namespace isr {

void RetireRuntime::emitRetireIntent(const RetireIntent& intent) noexcept
{
    uint64_t tail = convo::consumeAtomic(retireIntentTail_, std::memory_order_relaxed);
    uint64_t nextTail = (tail + 1) % RETIRE_INTENT_QUEUE_SIZE;

    uint64_t head = convo::consumeAtomic(retireIntentHead_, std::memory_order_acquire);
    if (nextTail == head) {
        // ★ P1: MPSC満杯 → Fallback queue へ退避
        {
            std::lock_guard<std::mutex> lock(fallbackMutex_);
            if (fallbackCount_.load(std::memory_order_relaxed) < FALLBACK_QUEUE_CAPACITY)
            {
                const size_t idx = fallbackCount_.load(std::memory_order_relaxed);
                fallbackQueue_[idx] = intent;
                convo::publishAtomic(fallbackCount_, idx + 1, std::memory_order_release);

                // 最大使用量を更新
                size_t prevHwm = convo::consumeAtomic(fallbackHighWatermark_, std::memory_order_relaxed);
                while (idx + 1 > prevHwm
                    && !convo::compareExchangeAtomic(fallbackHighWatermark_, prevHwm, idx + 1,
                        std::memory_order_release, std::memory_order_relaxed)) {}

                (void)convo::fetchAddAtomic(overflowCount_, uint64_t{1}, std::memory_order_acq_rel);
                // droppedIntentCount_ はインクリメントしない（保存成功）
            }
            else
            {
                // Fallback も満杯 → overflow としてカウント
                (void)convo::fetchAddAtomic(fallbackOverflowCount_, uint64_t{1}, std::memory_order_acq_rel);
                (void)convo::fetchAddAtomic(overflowCount_, uint64_t{1}, std::memory_order_acq_rel);
                (void)convo::fetchAddAtomic(droppedIntentCount_, uint64_t{1}, std::memory_order_acq_rel);
            }
        }

        // ★ C-1: overflowStartTimestamp_ を初回のみ設定（CAS）
        uint64_t expected = 0;
        convo::compareExchangeAtomic(overflowStartTimestamp_, expected,
            static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()),
            std::memory_order_release);

        (void)convo::fetchAddAtomic(overflowWindowCounter_, uint64_t{1},
            std::memory_order_release);

        convo::publishAtomic(lastOverflowTicks_,
            static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()),
            std::memory_order_release);
        return;
    }

    // ★ C-1: success（キュー空きあり）: overflow が継続中ならタイムスタンプをリセット
    uint64_t prevStart = convo::exchangeAtomic(overflowStartTimestamp_, uint64_t{0},
        std::memory_order_release);
    (void)prevStart;

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

    // 1. Drain MPSC queue
    uint64_t head = convo::consumeAtomic(retireIntentHead_, std::memory_order_acquire);
    uint64_t tail = convo::consumeAtomic(retireIntentTail_, std::memory_order_acquire);

    while (head != tail) {
        result.push_back(retireIntentQueue_[head]);
        head = (head + 1) % RETIRE_INTENT_QUEUE_SIZE;
    }

    convo::publishAtomic(retireIntentHead_, head, std::memory_order_release);

    // 2. ★ P1: Drain Fallback queue
    {
        std::lock_guard<std::mutex> lock(fallbackMutex_);
        const size_t fbCount = convo::consumeAtomic(fallbackCount_, std::memory_order_acquire);
        for (size_t i = 0; i < fbCount; ++i) {
            result.push_back(fallbackQueue_[i]);
        }
        convo::publishAtomic(fallbackCount_, size_t{0}, std::memory_order_release);
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

// ★ C-1: overflow 継続時間追跡 getter
std::uint64_t RetireRuntime::overflowStartTimestamp() const noexcept
{
    return convo::consumeAtomic(overflowStartTimestamp_, std::memory_order_acquire);
}

std::uint64_t RetireRuntime::lastOverflowTicks() const noexcept
{
    return convo::consumeAtomic(lastOverflowTicks_, std::memory_order_acquire);
}

std::uint64_t RetireRuntime::overflowWindowCounter() const noexcept
{
    return convo::consumeAtomic(overflowWindowCounter_, std::memory_order_acquire);
}

std::uint64_t RetireRuntime::lastOverflowWindowCount() const noexcept
{
    return convo::consumeAtomic(lastOverflowWindowCount_, std::memory_order_acquire);
}

// ★ P1: Fallback queue metrics
std::size_t RetireRuntime::fallbackOccupancy() const noexcept
{
    return convo::consumeAtomic(fallbackCount_, std::memory_order_acquire);
}

std::size_t RetireRuntime::fallbackHighWatermark() const noexcept
{
    return convo::consumeAtomic(fallbackHighWatermark_, std::memory_order_acquire);
}

std::uint64_t RetireRuntime::fallbackOverflowCount() const noexcept
{
    return convo::consumeAtomic(fallbackOverflowCount_, std::memory_order_acquire);
}

}  // namespace isr
}  // namespace convo
