#pragma once

#include <array>
#include <atomic>
#include <cstddef>

#include "AtomicAccess.h"
#include "RuntimeCommand.h"

namespace convo {

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324)
#endif

class RuntimeCommandQueue {
public:
    static constexpr std::size_t capacity = 128;

    // enqueue はメッセージスレッド専用（SPSC）のため mutex 不要
    bool enqueue(const EngineCommand& cmd) noexcept
    {
        const std::size_t write = convo::consumeAtomic(writeIndex, std::memory_order_acquire);
        const std::size_t read = convo::consumeAtomic(readIndex, std::memory_order_acquire);
        if ((write - read) >= capacity)
            return false;

        buffer[write & mask] = cmd;
        convo::publishAtomic(writeIndex, write + 1, std::memory_order_release);
        return true;
    }

    bool tryDequeue(EngineCommand& out) noexcept
    {
        const std::size_t read = convo::consumeAtomic(readIndex, std::memory_order_acquire);
        const std::size_t write = convo::consumeAtomic(writeIndex, std::memory_order_acquire);
        if (read == write)
            return false;

        out = buffer[read & mask];
        convo::publishAtomic(readIndex, read + 1, std::memory_order_release);
        return true;
    }

    int drainCoalesced(EngineCommand* outBuffer, int outCapacity) noexcept
    {
        if (outBuffer == nullptr || outCapacity <= 0)
            return 0;

        int count = 0;
        EngineCommand cmd {};
        while (tryDequeue(cmd))
        {
            bool replaced = false;
            for (int i = 0; i < count; ++i)
            {
                if (outBuffer[i].type == cmd.type)
                {
                    outBuffer[i] = cmd;
                    replaced = true;
                    break;
                }
            }

            if (!replaced && count < outCapacity)
                outBuffer[count++] = cmd;
        }

        return count;
    }

    void clear() noexcept
    {
        convo::publishAtomic(readIndex, convo::consumeAtomic(writeIndex, std::memory_order_acquire), std::memory_order_release);
    }

private:
    static constexpr std::size_t mask = capacity - 1;
    static_assert((capacity & mask) == 0, "RuntimeCommandQueue capacity must be a power of two");

    alignas(64) std::array<EngineCommand, capacity> buffer {};
    alignas(64) std::atomic<std::size_t> writeIndex { 0 };
    alignas(64) std::atomic<std::size_t> readIndex { 0 };
};

#ifdef _MSC_VER
#pragma warning(pop)
#endif

} // namespace convo
