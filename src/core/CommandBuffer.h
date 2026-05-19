//==============================================================================
// CommandBuffer.h
// SPSC lock-free ring buffer for parameter update commands
//==============================================================================
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "audioengine/AtomicAccess.h"

namespace convo {

struct ParameterCommand {
    enum class Type : uint8_t {
        ParameterChanged
    };

    Type type = Type::ParameterChanged;
    uint64_t generation = 0;

    ParameterCommand() = default;
    ParameterCommand(Type t, uint64_t gen) noexcept : type(t), generation(gen) {}
};

template <typename T, size_t Capacity>
#ifdef _MSC_VER
#pragma warning(push) // C4324 suppression scope begin: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
#pragma warning(disable : 4324) // Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
#endif
class SPSCRingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

public:
    bool push(const T& item) noexcept {
        // SPSC HB 契約:
        // acquire: 直前の pop() の readIndex release と HB し、最新の読み取り位置を観測して満杯判定。
        // acquire: writeIndex の自己読み取り — 書き込み側のみが更新するため、
        //          relaxed でも安全だが acquire で統一し HB 根拠を明確化。
        const size_t w = convo::consumeAtomic(writeIndex, std::memory_order_acquire);
        const size_t r = convo::consumeAtomic(readIndex, std::memory_order_acquire);
        if ((w - r) >= Capacity)
            return false;

        buffer[w & mask] = item;
        // release: バッファへの書き込みが完了した後に writeIndex を公開し、
        //          pop() 側の acquire と HB を形成してデータが可視化される。
        convo::publishAtomic(writeIndex, w + 1, std::memory_order_release);
        return true;
    }

    bool pop(T& item) noexcept {
        // SPSC HB 契約:
        // acquire: readIndex の自己読み取り — 消費側のみが更新するため relaxed でも安全だが acquire で統一。
        // acquire: push() の writeIndex release と HB し、最新の書き込み位置を観測して空判定。
        const size_t r = convo::consumeAtomic(readIndex, std::memory_order_acquire);
        const size_t w = convo::consumeAtomic(writeIndex, std::memory_order_acquire);
        if (r == w)
            return false;

        item = buffer[r & mask];
        // release: バッファ読み取り完了後に readIndex を公開し、
        //          push() 側の acquire と HB を形成してスロット再利用を安全化。
        convo::publishAtomic(readIndex, r + 1, std::memory_order_release);
        return true;
    }

private:
    static constexpr size_t mask = Capacity - 1;

    alignas(64) T buffer[Capacity]{};
    alignas(64) std::atomic<size_t> writeIndex{0};
    alignas(64) std::atomic<size_t> readIndex{0};
};
#ifdef _MSC_VER
#pragma warning(pop) // C4324 suppression scope end: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
#endif

using CommandBuffer = SPSCRingBuffer<ParameterCommand, 1024>;

} // namespace convo
