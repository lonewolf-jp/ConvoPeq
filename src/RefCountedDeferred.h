#pragma once

#include <atomic>
#include "DeferredDeletionQueue.h"

// グローバル epoch（AudioEngine から設定される）
extern std::atomic<uint64_t> g_currentEpoch;

template <typename T>
class RefCountedDeferred {
public:
    void addRef() const {
        refCount.fetch_add(1, std::memory_order_relaxed);
    }

    void release() const {
        if (refCount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::atomic_thread_fence(std::memory_order_acquire);
            g_deletionQueue.enqueue(
                const_cast<T*>(static_cast<const T*>(this)),
                [](void* p) { delete static_cast<T*>(p); },
                g_currentEpoch.load(std::memory_order_acquire)
            );
        }
    }

    // tryAddRef: オブジェクトがまだ有効な場合のみ参照カウントを増やす
    bool tryAddRef() const noexcept {
        int count = refCount.load(std::memory_order_relaxed);
        while (count > 0) {
            if (refCount.compare_exchange_weak(count, count + 1,
                    std::memory_order_acquire, std::memory_order_relaxed))
                return true;
        }
        return false;
    }

    int getRefCount() const noexcept {
        return refCount.load(std::memory_order_relaxed);
    }

protected:
    RefCountedDeferred() = default;
    ~RefCountedDeferred() = default;

private:
    mutable std::atomic<int> refCount{1};
};
