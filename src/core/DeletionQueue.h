//==============================================================================
// DeletionQueue.h
// スナップショット非同期解放のためのキュー（内部 epoch 記録）
// v13.0 設計ロック準拠
//==============================================================================
#pragma once

#include <cstdint>
#include <atomic>
#include <array>
#include <mutex>
#include "../DeferredDeletionQueue.h"

namespace convo {

class DeletionQueue {
public:
    void enqueue(void* ptr, void (*deleter)(void*), uint64_t epoch, DeletionEntryType type);
    // ★ C-09: 容量超過で破棄した件数（診断用）。0 以外は「解放できずに捨てた」ことを意味する。
    [[nodiscard]] uint64_t overflowDropCount() noexcept
    {
        return overflowDrops.load(std::memory_order_relaxed);
    }
    // [P1-21] epoch-free API: minReaderEpoch を直接受け取る (EpochDomain非依存)
    void reclaim(uint64_t minReaderEpoch);

private:
    struct Entry {
        void* ptr = nullptr;
        void (*deleter)(void*) = nullptr;
        uint64_t epoch = 0;
        DeletionEntryType type = DeletionEntryType::Generic;
    };

    static constexpr size_t kCapacity = 128;
    std::array<Entry, kCapacity> queue{};
    size_t count = 0;
    std::mutex mutex;
    std::atomic<uint64_t> overflowDrops { 0 };
};

} // namespace convo
