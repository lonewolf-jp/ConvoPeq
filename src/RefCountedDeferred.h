#pragma once

#include <atomic>
#include "DeferredDeletionQueue.h"
#include "core/EpochManager.h"

#include "audioengine/AtomicAccess.h"

template <typename T>
class RefCountedDeferred {
public:
    void addRef() {
        refCount.fetch_add(1, std::memory_order_acq_rel);
    }

    void release() {
        if (refCount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::atomic_thread_fence(std::memory_order_acquire);
            g_deletionQueue.enqueue(
                static_cast<T*>(this),
                [](void* p) { std::default_delete<T>{}(static_cast<T*>(p)); },
                convo::EpochManager::instance().currentEpoch()
            );
        }
    }

    // tryAddRef: オブジェクトがまだ有効な場合のみ参照カウントを増やす
    bool tryAddRef() noexcept {
        int count = convo::consumeAtomic(refCount, std::memory_order_acquire);
        while (count > 0) {
            if (refCount.compare_exchange_weak(count, count + 1,
                    std::memory_order_acq_rel, std::memory_order_acquire))
                return true;
        }
        return false;
    }

protected:
    RefCountedDeferred() = default;
    ~RefCountedDeferred() = default;

private:
    std::atomic<int> refCount{1};
};
