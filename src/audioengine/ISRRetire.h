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
 *
 * ★ B14: isValid 廃止 — slot.sequence が slot 状態の唯一の Authority。
 *    無効な intent (tombstone) は dspSlot == UINT32_MAX で識別する。
 *    これにより atomic<bool> 問題が完全に回避され、RetireIntent は
 *    trivially copyable かつ 64 バイト以下を維持できる。
 */
struct RetireIntent
{
    uint32_t dspSlot;
    uint64_t generation;  // ★ B-1: 64bit化
    uint64_t retireEpoch;
    // ★ Phase 5: 優先度フィールド（デフォルト Normal）
    RetirePriority priority{RetirePriority::Normal};
};

/**
 * ★ B14: Vyukov MPSC Queue slot
 */
struct RetireSlot {
    RetireIntent payload;
    std::atomic<uint64_t> sequence{0};
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

    // ★ B14: Vyukov MPSC 新 API
    void initQueue() noexcept;
    bool dequeueOne(RetireIntent& out) noexcept;
    bool dequeueFallback(RetireIntent& out) noexcept;

    // NonRT: dequeue retire intents (backward compat, built on dequeueOne/dequeueFallback)
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

    // ★ B14: Queue Pressure 診断 (HealthMonitor 用)
    [[nodiscard]] uint64_t approxQueueDepth() const noexcept
    {
        const uint64_t enqueued = convo::consumeAtomic(enqueueTicket_, std::memory_order_acquire);
        const uint64_t consumed = convo::consumeAtomic(dequeuePos_, std::memory_order_acquire);
        const uint64_t mainPending = (enqueued > consumed) ? (enqueued - consumed) : 0;
        const uint64_t fbPending = convo::consumeAtomic(fallbackCount_, std::memory_order_relaxed);
        return mainPending + fbPending;
    }

private:
    // ★ Phase 1: OverflowRing（純粋保存ストア、Coordinator管理）
    RetireOverflowRing* overflowRing_ = nullptr;

    // ★ B14: Vyukov MPSC Queue
    //    Producer: fetch_add(ticket) → slot.sequence で spin → payload 書き込み → sequence++
    //    Consumer: dequeuePos_ から slot.sequence 確認 → payload 読取 → sequence += SIZE で解放
    std::atomic<uint64_t> enqueueTicket_{0};
    std::atomic<uint64_t> dequeuePos_{0};  // Consumer 専有だが HealthMonitor が読むため atomic

    static constexpr size_t RETIRE_INTENT_QUEUE_SIZE = 256;
    RetireSlot slots_[RETIRE_INTENT_QUEUE_SIZE];
    std::array<std::atomic<uint64_t>, RETIRE_INTENT_QUEUE_SIZE> acknowledgeGeneration_{};  // ★ B-1: 64bit化
    std::atomic<uint64_t> acknowledgedCount_{0};
    std::atomic<uint64_t> overflowCount_{0};
    std::atomic<uint64_t> droppedIntentCount_{0};

    // ★ Phase 1: OverflowRing救済カウンタ
    std::atomic<uint64_t> quarantineRescuedCount_{0};

    // ★ B14: Fallback — head/tail 管理の循環バッファ (mutex保護, O(1) head移動)
    static constexpr size_t FALLBACK_QUEUE_CAPACITY = 4096;
    RetireIntent fallbackQueue_[FALLBACK_QUEUE_CAPACITY];
    size_t fallbackHead_{0};
    std::atomic<size_t> fallbackCount_{0};
    std::atomic<size_t> fallbackQueuePeak_{0};
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
