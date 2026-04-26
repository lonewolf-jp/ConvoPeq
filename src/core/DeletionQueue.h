//==============================================================================
// DeletionQueue.h
// スナップショット非同期解放のためのキュー（内部 epoch 記録）
// v13.0 設計ロック準拠
//==============================================================================
#pragma once

#include <cstdint>
#include <vector>
#include <mutex>
#include "EpochCore.h"
#include "../DeferredDeletionQueue.h"

namespace convo {

class DeletionQueue {
public:
    void enqueue(void* ptr, void (*deleter)(void*), uint64_t epoch, DeletionEntryType type);
    void enqueue(void* ptr, void (*deleter)(void*), uint64_t epoch);
    void reclaim(const EpochCore& core);
    void reclaimAllIgnoringEpoch();

private:
    struct Entry {
        void* ptr = nullptr;
        void (*deleter)(void*) = nullptr;
        uint64_t epoch = 0;
        DeletionEntryType type = DeletionEntryType::Generic;
    };

    std::vector<Entry> queue;
    std::mutex mutex;
};

} // namespace convo
