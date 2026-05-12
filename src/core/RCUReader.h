#pragma once

#include "EpochManager.h"
#include <atomic>

namespace convo {

/**
 * Epoch-based RCU Reader guard.
 * Usage:
 *   convo::RCUReader reader;
 *   convo::RCUReaderGuard guard(reader);
 *   // ... safe to read atomic pointers ...
 */
class RCUReader
{
public:
    RCUReader() = default;

    void enter() noexcept
    {
        const uint32_t previousDepth = nestingDepth.fetch_add(1, std::memory_order_acq_rel);
        if (previousDepth > 0)
            return;

        const int tid = acquireThreadSlot();
        if (tid >= 0)
            EpochManager::instance().enter(tid);
        else
            nestingDepth.fetch_sub(1, std::memory_order_acq_rel);
    }

    void exit() noexcept
    {
        const uint32_t previousDepth = nestingDepth.fetch_sub(1, std::memory_order_acq_rel);
        if (previousDepth == 0)
        {
            nestingDepth.store(0, std::memory_order_release);
            return;
        }

        if (previousDepth > 1)
            return;

        const int tid = activeThreadId.exchange(-1, std::memory_order_acq_rel);
        if (tid >= 0)
        {
            EpochManager::instance().exit(tid);
            preferredThreadId.store(tid, std::memory_order_release);
        }
    }

private:
    int acquireThreadSlot() noexcept
    {
        const int activeTid = activeThreadId.load(std::memory_order_acquire);
        if (activeTid >= 0)
            return activeTid;

        auto& manager = EpochManager::instance();
        const int preferredTid = preferredThreadId.load(std::memory_order_acquire);
        int reservedTid = -1;
        if (preferredTid >= 0 && manager.reserveThread(preferredTid))
        {
            reservedTid = preferredTid;
        }
        else
        {
            reservedTid = manager.registerThread();
        }

        activeThreadId.store(reservedTid, std::memory_order_release);
        return reservedTid;
    }

    std::atomic<int> preferredThreadId { -1 };
    std::atomic<int> activeThreadId { -1 };
    std::atomic<uint32_t> nestingDepth { 0 };
};

class RCUReaderGuard
{
public:
    explicit RCUReaderGuard(RCUReader& r) : reader(r) { reader.enter(); }
    ~RCUReaderGuard() { reader.exit(); }

    RCUReaderGuard(const RCUReaderGuard&) = delete;
    RCUReaderGuard& operator=(const RCUReaderGuard&) = delete;

private:
    RCUReader& reader;
};

} // namespace convo
