#pragma once

// [work37 Phase 1.5/8.1] フォールバックキュー拡張:
//   - retryCount 追加（最大3回）
//   - SoftLimit(1000)/HardLimit(2000) 二段構成
//   - estimatedBytes 追跡
//   - overflow 通知（PolicyEngine 連携用）

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>
#include "audioengine/AtomicAccess.h"

namespace convo {

struct DeferredRetireFallbackEntry
{
    void* ptr = nullptr;
    void (*deleter)(void*) = nullptr;
    std::uint64_t epoch = 0;
    int retryCount = 0;              // ★ work37: 再試行回数
    std::size_t estimatedSize = 0;   // ★ work37: 推定メモリサイズ
};

class DeferredRetireFallbackQueue
{
public:
    // ★ work37: SoftLimit(1000) 超過→PolicyEngineにCritical昇格要求
    //   HardLimit(2000) 超過→強制ドロップ（リーク許容）
    static constexpr std::size_t kFallbackSoftLimit = 1000;
    static constexpr std::size_t kFallbackHardLimit = 2000;
    static constexpr int kFallbackMaxRetries = 3;
    static constexpr std::size_t kFallbackMemoryLimit = 50 * 1024 * 1024; // 50MB

    using OverflowCallback = std::function<void()>;

    DeferredRetireFallbackQueue() = default;
    DeferredRetireFallbackQueue(const DeferredRetireFallbackQueue&) = delete;
    DeferredRetireFallbackQueue& operator=(const DeferredRetireFallbackQueue&) = delete;

    // ★ work37: 戻り値を bool に変更。HardLimit超過時はドロップして false を返す。
    //   SoftLimit超過時は notifyOverflow() を呼び出す。
    [[nodiscard]] bool push(DeferredRetireFallbackEntry entry)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.size() >= kFallbackHardLimit)
            return false;  // 強制ドロップ（リーク許容）
        if (queue_.size() >= kFallbackSoftLimit)
            ++softLimitOverflowCount_;  // PolicyEngine 連携用
        estimatedBytes_ += entry.estimatedSize;
        queue_.push_back(entry);
        totalPushCount_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    [[nodiscard]] std::vector<DeferredRetireFallbackEntry> popAll()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<DeferredRetireFallbackEntry> pending;
        pending.swap(queue_);
        estimatedBytes_ = 0;
        return pending;
    }

    [[nodiscard]] std::size_t size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    [[nodiscard]] bool empty() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    // ★ work37: 推定メモリ使用量
    [[nodiscard]] std::size_t estimatedBytes() const noexcept
    {
        return convo::consumeAtomic(estimatedBytes_, std::memory_order_acquire);
    }

    // ★ work37: SoftLimit 超過回数
    [[nodiscard]] uint64_t overflowCount() const noexcept
    {
        return convo::consumeAtomic(softLimitOverflowCount_, std::memory_order_acquire);
    }

    // ★ work37: 超過率（回数ベース）
    [[nodiscard]] double overflowRate() const noexcept
    {
        const auto count = convo::consumeAtomic(softLimitOverflowCount_, std::memory_order_acquire);
        const auto total = convo::consumeAtomic(totalPushCount_, std::memory_order_acquire);
        return (total > 0) ? static_cast<double>(count) / total : 0.0;
    }

private:
    mutable std::mutex mutex_;
    std::vector<DeferredRetireFallbackEntry> queue_;
    std::atomic<std::size_t> estimatedBytes_{0};
    std::atomic<uint64_t> softLimitOverflowCount_{0};
    std::atomic<uint64_t> totalPushCount_{0};
};

} // namespace convo
