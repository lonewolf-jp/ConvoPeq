// LockFreeRingBuffer.h
// SPSCロックフリーリングバッファ（RT安全・64byteアライン）
#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <cstring>
#include <cassert>
#include <utility>

#include "audioengine/AtomicAccess.h"

// T: trivially copyable型のみ
// Capacity: 2の冪

// C4324: alignas指定子によって構造体がパッドされた
// alignas(64) はキャッシュライン分離に必須のため、警告を抑制する
#ifdef _MSC_VER
#  pragma warning(push) // C4324 suppression scope begin: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
#  pragma warning(disable : 4324) // Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
#endif

template<typename T, size_t Capacity>
class LockFreeRingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
    alignas(64) T buffer[Capacity];
    alignas(64) std::atomic<size_t> writeIndex{0};
    alignas(64) std::atomic<size_t> readIndex{0};
    static constexpr size_t MASK = Capacity - 1;
public:
    bool push(const T& item) noexcept {
        size_t w = convo::consumeAtomic(writeIndex, std::memory_order_acquire);
        size_t r = convo::consumeAtomic(readIndex, std::memory_order_acquire);
        if ((w - r) >= Capacity) return false; // full
        buffer[w & MASK] = item;
        convo::publishAtomic(writeIndex, w + 1, std::memory_order_release);
        return true;
    }
    template<typename Writer>
    bool pushWithWriter(Writer&& writer) noexcept {
        size_t w = convo::consumeAtomic(writeIndex, std::memory_order_acquire);
        size_t r = convo::consumeAtomic(readIndex, std::memory_order_acquire);
        if ((w - r) >= Capacity) return false; // full
        std::forward<Writer>(writer)(buffer[w & MASK]);
        convo::publishAtomic(writeIndex, w + 1, std::memory_order_release);
        return true;
    }
    bool pop(T& item) noexcept {
        size_t r = convo::consumeAtomic(readIndex, std::memory_order_acquire);
        size_t w = convo::consumeAtomic(writeIndex, std::memory_order_acquire);
        if (r == w) return false; // empty
        // Memory ordering contract (SPSC):
        // - Producer writes buffer slot BEFORE publishing via convo::publishAtomic(writeIndex, release)
        // - Consumer reads writeIndex(acquire) BEFORE reading buffer slot
        // This guarantees the element is fully written before it is read.
        //
        // NOTE:
        // - The copy is non-atomic; T must be trivially copyable.
        // - Do NOT use with types that have internal pointers, ownership,
        //   or non-trivial invariants.
        // - Do NOT rely on this pattern for multi-producer/consumer scenarios.
        //
        // Real-time safety: ensures no torn reads under proper memory ordering.
        // This relies on the producer's writeIndex release and consumer's acquire
        // to establish a happens-before relationship between write and read.
        item = buffer[r & MASK];
        convo::publishAtomic(readIndex, r + 1, std::memory_order_release);
        return true;
    }
    size_t size() const noexcept {
        size_t w = convo::consumeAtomic(writeIndex, std::memory_order_acquire);
        size_t r = convo::consumeAtomic(readIndex, std::memory_order_acquire);
        return w - r;
    }
    // 注意: この関数はスレッドセーフではない。
    // プロデューサーとコンシューマーが完全に停止している状態でのみ呼び出すこと。
    void clear() noexcept {
        convo::publishAtomic(writeIndex, 0, std::memory_order_seq_cst);
        convo::publishAtomic(readIndex, 0, std::memory_order_seq_cst);
    }
};

#ifdef _MSC_VER
#  pragma warning(pop) // C4324 suppression scope end: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
#endif
