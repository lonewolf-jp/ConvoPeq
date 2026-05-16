//==============================================================================
// DeletionQueue.h
// スナップショット非同期解放のためのキュー（内部 epoch 記録）
// v13.0 設計ロック準拠
//==============================================================================
#pragma once

#include <cstdint>
#include <array>
#include <mutex>
#include "EpochCore.h"
#include "../DeferredDeletionQueue.h"

namespace convo {

class DeletionQueue {
public:
    void enqueue(void* ptr, void (*deleter)(void*), uint64_t epoch, DeletionEntryType type);
    void reclaim(const EpochCore& core);

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
};

} // namespace convo
