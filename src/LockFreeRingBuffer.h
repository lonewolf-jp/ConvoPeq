// LockFreeRingBuffer.h
// SPSCロックフリーリングバッファ（RT安全・64byteアライン）
#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <cstring>
#include <cassert>

// T: trivially copyable型のみ
// Capacity: 2の冪

// C4324: alignas指定子によって構造体がパッドされた
// alignas(64) はキャッシュライン分離に必須のため、警告を抑制する
#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable: 4324)
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
        size_t w = writeIndex.load(std::memory_order_relaxed);
        size_t r = readIndex.load(std::memory_order_acquire);
        if ((w - r) >= Capacity) return false; // full
        buffer[w & MASK] = item;
        writeIndex.store(w + 1, std::memory_order_release);
        return true;
    }
    bool pop(T& item) noexcept {
        size_t r = readIndex.load(std::memory_order_relaxed);
        size_t w = writeIndex.load(std::memory_order_acquire);
        if (r == w) return false; // empty
        item = buffer[r & MASK];
        readIndex.store(r + 1, std::memory_order_release);
        return true;
    }
    size_t size() const noexcept {
        size_t w = writeIndex.load(std::memory_order_acquire);
        size_t r = readIndex.load(std::memory_order_acquire);
        return w - r;
    }
    void clear() noexcept {
        writeIndex.store(0, std::memory_order_relaxed);
        readIndex.store(0, std::memory_order_relaxed);
    }
};

#ifdef _MSC_VER
#  pragma warning(pop)
#endif
