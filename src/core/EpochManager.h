#pragma once
#include <atomic>
#include <array>
#include <cstdint>
#include <limits>

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
            if (threads[i].epoch.compare_exchange_strong(expected, kReservedEpoch))
                return i;
        }
        return -1; // fatal: too many threads
    }

    void enter(int tid)
    {
        if (tid < 0 || tid >= kMaxThreads) return;
        uint64_t e = globalEpoch.load(std::memory_order_acquire);
        threads[tid].epoch.store(e, std::memory_order_release);
    }

    void exit(int tid)
    {
        if (tid < 0 || tid >= kMaxThreads) return;
        threads[tid].epoch.store(kInactiveEpoch, std::memory_order_release);
    }

    // ===== Writer API =====

    uint64_t currentEpoch() const
    {
        return globalEpoch.load(std::memory_order_acquire);
    }

    void advanceEpoch()
    {
        globalEpoch.fetch_add(1, std::memory_order_acq_rel);
    }

    uint64_t minActiveEpoch() const
    {
        uint64_t minE = globalEpoch.load(std::memory_order_acquire);

        for (const auto& t : threads)
        {
            uint64_t e = t.epoch.load(std::memory_order_acquire);
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
