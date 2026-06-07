#pragma once

// [work21 Phase-D] RefCountedDeferred — Router-based retire only.
// Old release(EpochDomain&) removed — use release(IEpochProvider&) instead.

#include <atomic>
#include <memory>
#include "core/IEpochProvider.h"

#include "audioengine/AtomicAccess.h"

template <typename T>
class RefCountedDeferred {
public:
    void addRef() {
        convo::fetchAddAtomic(refCount, 1, std::memory_order_acq_rel);
    }

    // [work21 Phase-D] IEpochProvider 経由版 (EpochDomain型露出回避)
    void release(convo::IEpochProvider& provider) {
        if (convo::fetchSubAtomic(refCount, 1, std::memory_order_acq_rel) == 1) {
            std::atomic_thread_fence(std::memory_order_acquire);
            provider.enqueueRetire(
                static_cast<T*>(this),
                [](void* p) { std::default_delete<T>{}(static_cast<T*>(p)); },
                provider.currentEpoch()
            );
        }
    }

    // tryAddRef: オブジェクトがまだ有効な場合のみ参照カウントを増やす
    bool tryAddRef() noexcept {
        int count = convo::consumeAtomic(refCount, std::memory_order_acquire); // acquire: 直前の addRef/release の acq_rel と HB し最新カウントを観測
        while (count > 0) {
                if (convo::compareExchangeAtomic(refCount,
                    count,
                    count + 1,
                    std::memory_order_acq_rel,  // 成功時 acq_rel: acquire で release の acq_rel と HB; release で次の release/tryAddRef の acquire と HB
                    std::memory_order_acquire)) // 失敗時 acquire: 最新 refCount を観測して CAS を再試行
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
