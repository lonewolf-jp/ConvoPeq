#pragma once

#include <atomic>
#include "core/EpochDomain.h"

#include "audioengine/AtomicAccess.h"

template <typename T>
class RefCountedDeferred {
public:
    void addRef() {
        convo::fetchAddAtomic(refCount, 1, std::memory_order_acq_rel); // acq_rel: acquire で直前の release/addRef と HB し最新カウントを観測; release で release の acquire と HB
    }

    void release(convo::EpochDomain& epochDomain) {
        if (convo::fetchSubAtomic(refCount, 1, std::memory_order_acq_rel) == 1) { // acq_rel: acquire で全 addRef/tryAddRef の release と HB し最後の参照を確認; release で fence の acquire と HB
            std::atomic_thread_fence(std::memory_order_acquire); // acquire: 上記 fetchSub acq_rel の release 側と HB し、他スレッドの全 addRef/release を可視化してから retire
            epochDomain.enqueueRetire(
                static_cast<T*>(this),
                [](void* p) { std::default_delete<T>{}(static_cast<T*>(p)); },
                epochDomain.currentEpoch()
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
