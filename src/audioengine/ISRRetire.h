#pragma once

#include <atomic>
#include <cstdint>
#include <vector>
#include <array>
#include <mutex>

#include "AtomicAccess.h"
#include "ISRAuthorityClass.h"  // RetirePriority

namespace convo {
namespace isr {

// ★ Phase 1: 前方宣言（ISRRetireOverflowRing.h で完全定義）
class RetireOverflowRing;

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
    // ★ Phase 5: 優先度フィールド（デフォルト Normal）
    RetirePriority priority{RetirePriority::Normal};
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

    // ★ Phase5: 全保留中Intentの優先度を底上げ（Shutdown時の Critical 一括昇格用）
    void escalateAllRetires(RetirePriority minPriority) noexcept;
    [[nodiscard]] std::uint64_t fallbackOverflowCount() const noexcept;

    // ★ C-1: overflow 継続時間追跡
    [[nodiscard]] std::uint64_t overflowStartTimestamp() const noexcept;
    [[nodiscard]] std::uint64_t lastOverflowTicks() const noexcept;
    [[nodiscard]] std::uint64_t overflowWindowCounter() const noexcept;
    [[nodiscard]] std::uint64_t lastOverflowWindowCount() const noexcept;

    // NonRT: acknowledge retire coordination
    void acknowledgeRetireCoordination(const RetireIntent& intent);

    // ★ Phase 1: OverflowRing 連携
    void setOverflowRing(RetireOverflowRing* ring) noexcept { overflowRing_ = ring; }
    [[nodiscard]] RetireOverflowRing* getOverflowRing() const noexcept { return overflowRing_; }

    // ★ Phase 1: OverflowRing 救済統計
    [[nodiscard]] std::uint64_t quarantineRescuedCount() const noexcept
    {
        return convo::consumeAtomic(quarantineRescuedCount_, std::memory_order_acquire);
    }

private:
    // ★ Phase 1: OverflowRing（純粋保存ストア、Coordinator管理）
    RetireOverflowRing* overflowRing_ = nullptr;

    // Lock-free queue (using atomics)
    std::atomic<uint64_t> retireIntentHead_{0};
    std::atomic<uint64_t> retireIntentTail_{0};

    static constexpr size_t RETIRE_INTENT_QUEUE_SIZE = 256;
    RetireIntent retireIntentQueue_[RETIRE_INTENT_QUEUE_SIZE];
    std::array<std::atomic<uint64_t>, RETIRE_INTENT_QUEUE_SIZE> acknowledgeGeneration_{};  // ★ B-1: 64bit化
    std::atomic<uint64_t> acknowledgedCount_{0};
    std::atomic<uint64_t> overflowCount_{0};
    std::atomic<uint64_t> droppedIntentCount_{0};

    // ★ Phase 1: OverflowRing救済カウンタ
    std::atomic<uint64_t> quarantineRescuedCount_{0};

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
