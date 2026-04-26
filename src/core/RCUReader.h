#pragma once

#include "EpochManager.h"

namespace convo {

/**
 * Epoch-based RCU Reader guard.
 * Usage:
 *   thread_local convo::RCUReader reader;
 *   convo::RCUReaderGuard guard(reader);
 *   // ... safe to read atomic pointers ...
 */
class RCUReader
{
public:
    RCUReader()
    {
        // One-time registration for this thread
        static thread_local int tid = EpochManager::instance().registerThread();
        threadId = tid;
    }

    void enter() noexcept
    {
        EpochManager::instance().enter(threadId);
    }

    void exit() noexcept
    {
        EpochManager::instance().exit(threadId);
    }

private:
    int threadId;
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
