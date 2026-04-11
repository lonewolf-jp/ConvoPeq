//==============================================================================
// core/ReaderEpoch.h
// スナップショット専用軽量 epoch 管理（リーダースロット機能完全削除）
//==============================================================================
#pragma once

#include <atomic>
#include <cstdint>

namespace convo {

class SnapshotEpoch {
public:
    static uint64_t advance() noexcept {
        return s_epoch.fetch_add(1, std::memory_order_acq_rel) + 1;
    }

    static uint64_t get() noexcept {
        return s_epoch.load(std::memory_order_acquire);
    }

private:
    static inline std::atomic<uint64_t> s_epoch{1};
};

} // namespace convo
