#pragma once

// [work21 Phase-D] RefCountedDeferred — Router-based retire only.
// Old release(EpochDomain&) removed — use release(IEpochProvider&) instead.
//
// [work37 Phase 1.3] enqueueRetire 戻り値チェック追加。
//   canBlock() 判定により RT スレッドからは tryReclaim をスキップ。
//
// [work69 Phase B08] RetirePolicy による dispatch 追加。
//   release(RetirePolicy::Immediate) は Shutdown 時に EBR を迂回する。

#include <atomic>
#include <memory>
#include "core/IEpochProvider.h"
#include "DspNumericPolicy.h"

#include "audioengine/AtomicAccess.h"

// ★ B08: Retire policy — EBR 経由または即時 delete
enum class RetirePolicy {
    Epoch,      // 通常: EBR 経由 (IEpochProvider& が必要)
    Immediate   // Shutdown: EBR を迂回し即時 delete
};

template <typename T>
class RefCountedDeferred {
public:
    void addRef() {
        convo::fetchAddAtomic(refCount, 1, std::memory_order_acq_rel);
    }

    // [work37 Phase 1.3] enqueueRetire 戻り値をチェック。RT から呼ばれ得るため、
    //   canBlock() (Non-RT) の場合のみ tryReclaim 再試行を行う。
    //   RT からの失敗は HealthMonitor overflowCount 監視に委ねる。
    void release(convo::IEpochProvider& provider) {
        if (convo::fetchSubAtomic(refCount, 1, std::memory_order_acq_rel) == 1) {
            std::atomic_thread_fence(std::memory_order_acquire);
            if (!provider.enqueueRetire(
                    static_cast<T*>(this),
                    [](void* p) { std::default_delete<T>{}(static_cast<T*>(p)); },
                    provider.currentEpoch())) {
                // canBlock() が false (RT) なら tryReclaim 禁止
                if (!convo::numeric_policy::isAudioThread()) {
                    provider.tryReclaim();
                    (void)provider.enqueueRetire(
                        static_cast<T*>(this),
                        [](void* p) { std::default_delete<T>{}(static_cast<T*>(p)); },
                        provider.currentEpoch());
                }
                // 再試行失敗は HealthMonitor overflowCount 監視に委ねる
            }
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

    // ★ B08: Shutdown 専用 — RetireRouter を経由せず即時 delete
    //    使用条件: AudioEngine::ShutdownPhase::Destroy 以上 (Shutdown 時のみ)
    //    Runtime publish 後の呼び出しは禁止 (EBR 迂回により Audio Thread 参照中破棄のリスク)
    [[nodiscard]] bool release(RetirePolicy policy) noexcept {
        if (convo::fetchSubAtomic(refCount, 1, std::memory_order_acq_rel) == 1) {
            std::atomic_thread_fence(std::memory_order_acquire);
            if (policy == RetirePolicy::Immediate) {
                delete static_cast<T*>(this);
            }
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
