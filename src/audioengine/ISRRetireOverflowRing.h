// ISRRetireOverflowRing.h
// Phase 1: SPSC Lock-Free Overflow Ring — 純粋保存専用ストア
//
// ★ 責務: tryPush / pop / residentCount / drainAll のみ
//   retry/age/deferred/priority は一切判断しない → Coordinator の責務
//
// ★ ADR-001: SPSC 前提
//   Producer: Audio Callback（processBlock）のみ
//   Consumer: Coordinator (NonRT Timer) のみ
//   複数Producerが必要な場合は MPSC Ring への置き換えが必要
//
// ★ 設計: LockFreeRingBuffer.h テンプレートを直接流用
//   - trivially copyable 制約
//   - power-of-2 容量制約
//   - SPSC acquire/release ordering

#pragma once

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <vector>

#include "../LockFreeRingBuffer.h"
#include "ISRRetire.h"

namespace convo {
namespace isr {

// LockFreeRingBuffer準拠: trivially copyable, power-of-2 capacity
// メタデータ（overflowTimestampUs, reinjectRetryCount）は保持するが、
// これらを使ったスケジューリング判断は Coordinator が行う。
struct RetireOverflowEntry
{
    RetireIntent intent;
    uint64_t overflowTimestampUs;    // Coordinator が滞留監視に使用
    uint16_t reinjectRetryCount;     // Coordinator が retry 管理に使用
};
// sizeof(RetireOverflowEntry) = 24 (RetireIntent) + 8 + 2 = 34 → pad to 40 bytes
// trivially copyable ✅ (all fields are primitive types)

/**
 * RetireOverflowRing — 純粋保存専用ストア
 *
 * 責務: tryPush / pop / residentCount / drainAll のみ
 * retry/age/deferred/priority は一切判断しない
 *
 * SPSC前提 (ADR-001):
 *   Producer: Audio Callback（processBlock）のみ
 *   Consumer: Coordinator (NonRT Timer) のみ
 */
class RetireOverflowRing
{
public:
    // Ring容量: 2^14 (power-of-2必須, LockFreeRingBuffer制約)
    static constexpr size_t kRingCapacity = 16384;

    // RT-safe: SPSC lock-free push（Audio Callback のみ）
    // 戻り値: true=成功, false=Ring満杯
    [[nodiscard]] bool tryPush(const RetireOverflowEntry& entry) noexcept
    {
        if (ring_.push(entry)) {
            return true;
        }
        // Ring満杯 → 呼出元で droppedIntentCount_++ + onHealthEvent
        return false;
    }

    // NonRT: FIFO pop（Coordinator が drainOverflowRing() 内で呼出）
    [[nodiscard]] bool pop(RetireOverflowEntry& out) noexcept
    {
        return ring_.pop(out);
    }

    // 現在の滞留数
    [[nodiscard]] size_t residentCount() const noexcept
    {
        return ring_.size();
    }

    // 累計overflow回数（デバッグ診断用）
    [[nodiscard]] uint64_t totalOverflowCount() const noexcept
    {
        return convo::consumeAtomic(totalOverflowCount_, std::memory_order_acquire);
    }

    // Shutdown用: 全エントリ強制排出
    // Coordinator が排出後のエントリを Coordinator の DeferredRing で管理
    void drainAll(std::vector<RetireOverflowEntry>& out) noexcept
    {
        RetireOverflowEntry entry;
        while (ring_.pop(entry)) {
            out.push_back(entry);
        }
    }

    // リセット（テスト・Shutdown完了後）
    void clear() noexcept
    {
        ring_.clear();
        convo::publishAtomic(totalOverflowCount_, uint64_t{0}, std::memory_order_release);
    }

    void incrementOverflowCount() noexcept
    {
        convo::fetchAddAtomic(totalOverflowCount_, uint64_t{1}, std::memory_order_release);
    }

private:
    // ★ 単一Ring — deferredRing_ は Coordinator が保持（Phase 5）
    LockFreeRingBuffer<RetireOverflowEntry, kRingCapacity> ring_;
    std::atomic<uint64_t> totalOverflowCount_{0};
};

} // namespace isr
} // namespace convo
