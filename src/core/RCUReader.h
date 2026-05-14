#pragma once

#include "EpochManager.h"
#include <atomic>
#include <functional>
#include <thread>

#include "audioengine/AtomicAccess.h"

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
    RCUReader(const RCUReader&) = delete;
    RCUReader& operator=(const RCUReader&) = delete;
    RCUReader(RCUReader&&) = delete;
    RCUReader& operator=(RCUReader&&) = delete;

    void enter() noexcept
    {
        const uint32_t previousDepth = convo::fetchAddAtomic(nestingDepth, static_cast<uint32_t>(1), std::memory_order_acq_rel);
        if (previousDepth > 0)
        {
            return;
        }

        const uint64_t threadToken = currentThreadToken();
        uint64_t expectedOwner = 0;
        if (!convo::compareExchangeAtomic(ownerThreadToken,
                                          expectedOwner,
                                          threadToken,
                                          std::memory_order_acq_rel,
                                          std::memory_order_acquire)
            && expectedOwner != threadToken)
        {
            convo::fetchSubAtomic(nestingDepth, static_cast<uint32_t>(1), std::memory_order_acq_rel);
            return;
        }

        const int tid = acquireThreadSlot();
        if (tid >= 0)
            EpochManager::instance().enter(tid);
        else
        {
            convo::fetchSubAtomic(nestingDepth, static_cast<uint32_t>(1), std::memory_order_acq_rel);
            uint64_t expectedOwnerOnRelease = threadToken;
            convo::compareExchangeAtomic(ownerThreadToken,
                                         expectedOwnerOnRelease,
                                         static_cast<uint64_t>(0),
                                         std::memory_order_acq_rel,
                                         std::memory_order_acquire);
        }
    }

    void exit() noexcept
    {
        const uint32_t previousDepth = convo::fetchSubAtomic(nestingDepth, static_cast<uint32_t>(1), std::memory_order_acq_rel);
        if (previousDepth == 0)
        {
            convo::publishAtomic(nestingDepth, 0, std::memory_order_release);
            return;
        }

        if (previousDepth > 1)
            return;

        if (convo::consumeAtomic(ownerThreadToken, std::memory_order_acquire) != currentThreadToken())
            return;

        const int tid = convo::exchangeAtomic(activeThreadId, -1, std::memory_order_acq_rel);
        if (tid >= 0)
        {
            EpochManager::instance().exit(tid);
            convo::publishAtomic(preferredThreadId, tid, std::memory_order_release);
        }
        convo::publishAtomic(ownerThreadToken, static_cast<uint64_t>(0), std::memory_order_release);
    }

private:
    static uint64_t currentThreadToken() noexcept
    {
        return static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    }

    int acquireThreadSlot() noexcept
    {
        const int activeTid = convo::consumeAtomic(activeThreadId, std::memory_order_acquire);
        if (activeTid >= 0)
            return activeTid;

        auto& manager = EpochManager::instance();
        const int preferredTid = convo::consumeAtomic(preferredThreadId, std::memory_order_acquire);
        int reservedTid = -1;
        if (preferredTid >= 0 && manager.reserveThread(preferredTid))
        {
            reservedTid = preferredTid;
        }
        else
        {
            reservedTid = manager.registerThread();
        }

        convo::publishAtomic(activeThreadId, reservedTid, std::memory_order_release);
        return reservedTid;
    }

    std::atomic<int> preferredThreadId { -1 };
    std::atomic<int> activeThreadId { -1 };
    std::atomic<uint32_t> nestingDepth { 0 };
    std::atomic<uint64_t> ownerThreadToken { 0 };
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
