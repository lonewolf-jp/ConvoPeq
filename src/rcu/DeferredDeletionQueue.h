//==============================================================================
// DeferredDeletionQueue.h
// ConvoPeq RCU v17.15 - Vyukov 型 Bounded MPSC キュー
// 
// 設計:
//   - Dmitry Vyukov の bounded MPMC queue アルゴリズムに準拠
//   - sequence 配列を外部に配置し、Retired エントリは trivially copyable
//   - enqueue(): 必ず成功（スピン＋yield）
//   - try_enqueue(): スロット確保できた場合のみ true
//   - dequeue(): 消費者が呼ぶ（ReclaimerThread）
//==============================================================================
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace convo {

struct Retired {
    void* ptr;
    void (*deleter)(void*);
    uint64_t epoch;
};

static_assert(std::is_trivially_copyable_v<Retired>, "Retired must be trivially copyable");
static_assert(std::is_trivially_destructible_v<Retired>, "Retired must be trivially destructible");

class DeferredDeletionQueue {
public:
    DeferredDeletionQueue();
    
    void enqueue(const Retired& r) noexcept;      // always succeeds (spin + yield)
    bool try_enqueue(const Retired& r) noexcept;  // non-blocking, may fail if full
    bool dequeue(Retired& r) noexcept;            // consumer only
    size_t size() const noexcept;

private:
    static constexpr size_t CAPACITY = 65536;
    static constexpr size_t MASK = CAPACITY - 1;
    static constexpr int SPIN_LIMIT = 1024;

    alignas(64) std::atomic<size_t> m_head {0};
    alignas(64) std::atomic<size_t> m_tail {0};
    alignas(64) Retired m_buffer[CAPACITY];
    alignas(64) std::atomic<uint64_t> m_sequence[CAPACITY] {};
};

inline void cpu_pause() noexcept {
#if defined(_MSC_VER)
    _mm_pause();
#elif defined(__GNUC__) || defined(__clang__)
    #if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
    #elif defined(__aarch64__)
        __asm__ volatile("yield" ::: "memory");
    #else
        __asm__ volatile("" ::: "memory");
    #endif
#else
    __asm__ volatile("" ::: "memory");
#endif
}

} // namespace convo
