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
#pragma warning(push)
#pragma warning(disable : 4324) // alignas による意図的なパディング警告を抑制
#endif
class SPSCRingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

public:
    bool push(const T& item) noexcept {
        const size_t w = convo::consumeAtomic(writeIndex, std::memory_order_acquire);
        const size_t r = convo::consumeAtomic(readIndex, std::memory_order_acquire);
        if ((w - r) >= Capacity)
            return false;

        buffer[w & mask] = item;
        convo::publishAtomic(writeIndex, w + 1, std::memory_order_release);
        return true;
    }

    bool pop(T& item) noexcept {
        const size_t r = convo::consumeAtomic(readIndex, std::memory_order_acquire);
        const size_t w = convo::consumeAtomic(writeIndex, std::memory_order_acquire);
        if (r == w)
            return false;

        item = buffer[r & mask];
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
#pragma warning(pop)
#endif

using CommandBuffer = SPSCRingBuffer<ParameterCommand, 1024>;

} // namespace convo
