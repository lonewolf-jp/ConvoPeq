#pragma once

// ★ R-1: RefCountedDeferred — Router-based retire via IRetireRouter.
//   release(IRetireRouter&): NonRT, リトライ込み（Router::retire 経由）
//   releaseRT(IRetireRouter&): RT-safe, リトライなし（Router::retireRT 経由）。戻り値 bool
//   releaseDirect(): Shutdown 専用。RetireRouter を経由せず即時 delete。
//
// releaseDirect() 使用条件:
//   AudioEngine::ShutdownPhase::Destroy 以上 (Shutdown 時のみ)
//   Runtime publish 後の呼び出しは禁止 (EBR 迂回により Audio Thread 参照中破棄のリスク)

#include <atomic>
#include <memory>
#include "core/IRetireRouter.h"
#include "DspNumericPolicy.h"
#include "audioengine/AtomicAccess.h"

template <typename T>
class RefCountedDeferred {
public:
    void addRef() {
        convo::fetchAddAtomic(refCount, 1, std::memory_order_acq_rel);
    }

    // ★ R-1: NonRT release — Router::retire 経由（リトライ・QueuePressure 通知は Router 内部で完結）
    void release(convo::IRetireRouter& router) {
        if (convo::fetchSubAtomic(refCount, 1, std::memory_order_acq_rel) == 1) {
            std::atomic_thread_fence(std::memory_order_acquire);
            router.retire(
                static_cast<T*>(this),
                [](void* p) { std::default_delete<T>{}(static_cast<T*>(p)); });
        }
    }

    // ★ R-1: RT-safe release — Router::retireRT 経由（単発 enqueue、リトライなし）
    //   戻り値: true=成功/false=QueueFull
    [[nodiscard]] bool releaseRT(convo::IRetireRouter& router) noexcept {
        if (convo::fetchSubAtomic(refCount, 1, std::memory_order_acq_rel) == 1) {
            std::atomic_thread_fence(std::memory_order_acquire);
            return router.retireRT(
                static_cast<T*>(this),
                [](void* p) { std::default_delete<T>{}(static_cast<T*>(p)); });
        }
        return true; // Not the last ref, success by definition
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

    // ★ R-1: Shutdown 専用 — RetireRouter を経由せず即時 delete
    //    使用条件: AudioEngine::ShutdownPhase::Destroy 以上 (Shutdown 時のみ)
    //    Runtime publish 後の呼び出しは禁止 (EBR 迂回により Audio Thread 参照中破棄のリスク)
    [[nodiscard]] bool releaseDirect() noexcept {
        if (convo::fetchSubAtomic(refCount, 1, std::memory_order_acq_rel) == 1) {
            std::atomic_thread_fence(std::memory_order_acquire);
            delete static_cast<T*>(this);
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
