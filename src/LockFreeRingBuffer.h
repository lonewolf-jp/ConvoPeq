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
        // SPSC HB 契約: acquire で最新の readIndex を観測して満杯判定;
        //               バッファ書き込み後に writeIndex を release し pop() の acquire と HB 形成。
        size_t w = convo::consumeAtomic(writeIndex, std::memory_order_acquire);
        size_t r = convo::consumeAtomic(readIndex, std::memory_order_acquire);
        if ((w - r) >= Capacity) return false; // full
        buffer[w & MASK] = item;
        convo::publishAtomic(writeIndex, w + 1, std::memory_order_release);
        return true;
    }
    template<typename Writer>
    bool pushWithWriter(Writer&& writer) noexcept {
        // SPSC HB 契約: push() と同じ acquire/release 対。
        //               writer() 完了後に writeIndex release で pop() の acquire と HB 形成。
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
        // ★ work92 C-5 (big 2-5): 読取順序固定 — writeIndex を先に読む。
        //   旧実装 (r を先に読む) では、直後に producer が w を進めた場合 w - r が
        //   実占有数より大きくなり得、整数 wrap で巨大値を返す窓があった。
        //   w 先読みにより w は「読取時点以前の値」に固定され、r はその後進むため
        //   計算結果は実占有数以下に飽和する（過大評価は起きない）。
        //   acquire × 2: push/pop の release と HB し、一貫した（ベストエフォート）占有数を算出。
        size_t w = convo::consumeAtomic(writeIndex, std::memory_order_acquire);
        size_t r = convo::consumeAtomic(readIndex, std::memory_order_acquire);
        return (w >= r) ? (w - r) : 0;
    }
    // 注意: この関数はスレッドセーフではない。
    // プロデューサーとコンシューマーが完全に停止している状態でのみ呼び出すこと。
    // 停止後の単独再初期化なので seq_cst は不要で、release で十分。
    void clear() noexcept {
        convo::publishAtomic(writeIndex, 0, std::memory_order_release);
        convo::publishAtomic(readIndex, 0, std::memory_order_release);
    }
};

#ifdef _MSC_VER
#  pragma warning(pop) // C4324 suppression scope end: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
#endif
