#pragma once
#include <atomic>
#include <array>
#include <cstdint>
#include <limits>

#include "audioengine/AtomicAccess.h"

namespace convo {

class EpochManager
{
public:
    static constexpr int kMaxThreads = 64;
    static constexpr uint64_t kInactiveEpoch = std::numeric_limits<uint64_t>::max();
    static constexpr uint64_t kReservedEpoch = std::numeric_limits<uint64_t>::max() - 1;

    struct ThreadRecord
    {
        std::atomic<uint64_t> epoch { kInactiveEpoch };
    };

    static EpochManager& instance()
    {
        static EpochManager inst;
        return inst;
    }

    // ===== Reader API =====

    int registerThread()
    {
        for (int i = 0; i < kMaxThreads; ++i)
        {
            uint64_t expected = kInactiveEpoch;
            if (convo::compareExchangeAtomic(threads[i].epoch, expected, kReservedEpoch))
                return i;
        }
        return -1; // fatal: too many threads
    }

    bool reserveThread(int tid)
    {
        if (tid < 0 || tid >= kMaxThreads)
            return false;

        uint64_t expected = kInactiveEpoch;
        return convo::compareExchangeAtomic(threads[tid].epoch, expected, kReservedEpoch);
    }

    void enter(int tid)
    {
        if (tid < 0 || tid >= kMaxThreads) return;
        uint64_t e = convo::consumeAtomic(globalEpoch, std::memory_order_acquire);
        convo::publishAtomic(threads[tid].epoch, e, std::memory_order_release);
    }

    void exit(int tid)
    {
        if (tid < 0 || tid >= kMaxThreads) return;
        convo::publishAtomic(threads[tid].epoch, kInactiveEpoch, std::memory_order_release);
    }

    // ===== Writer API =====

    uint64_t currentEpoch() const
    {
        return convo::consumeAtomic(globalEpoch, std::memory_order_acquire);
    }

    void advanceEpoch()
    {
        convo::fetchAddAtomic(globalEpoch, static_cast<uint64_t>(1), std::memory_order_acq_rel);
    }

    uint64_t minActiveEpoch() const
    {
        uint64_t minE = convo::consumeAtomic(globalEpoch, std::memory_order_acquire);

        for (const auto& t : threads)
        {
            uint64_t e = convo::consumeAtomic(t.epoch, std::memory_order_acquire);
            if (e != kInactiveEpoch && e != kReservedEpoch)
            {
                if (isOlder(e, minE))
                    minE = e;
            }
        }
        return minE;
    }

    static inline bool isOlder(uint64_t a, uint64_t b) noexcept
    {
        return static_cast<int64_t>(a - b) < 0;
    }

private:
    EpochManager() : globalEpoch(1) {}
    ~EpochManager() = default;

    std::atomic<uint64_t> globalEpoch;
    std::array<ThreadRecord, kMaxThreads> threads;
};

} // namespace convo
