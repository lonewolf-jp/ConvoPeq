//==============================================================================
// WorkerThread.h
// Debounced snapshot update worker
//==============================================================================
#pragma once

#include <atomic>
#include <chrono>
#include <thread>
#include "CommandBuffer.h"

#include "audioengine/AtomicAccess.h"

class GenerationManager;
class ThreadAffinityManager;

namespace convo {

struct WorkerThreadConfig {
    int debounceDelayMs = 50;
    int idleSleepMs = 10;
};

class WorkerThread {
public:
    WorkerThread(CommandBuffer& cmdBuf,
                 GenerationManager& genManager,
                 const ThreadAffinityManager* affinityMgr,
                 const WorkerThreadConfig& config = WorkerThreadConfig());
    ~WorkerThread();

    void start();
    void stop();

#ifdef _DEBUG
    uint64_t getCommandsReceived() const noexcept { return convo::consumeAtomic(commandsReceived, std::memory_order_acquire); }   // acquire: run() の fetchAdd acq_rel と HB し最新カウントを観測
    uint64_t getSnapshotsCreated() const noexcept { return convo::consumeAtomic(snapshotsCreated, std::memory_order_acquire); }   // acquire: run() の fetchAdd acq_rel と HB
    uint64_t getCommandsDropped() const noexcept { return convo::consumeAtomic(commandsDropped, std::memory_order_acquire); }    // acquire: run() の fetchAdd acq_rel と HB
#endif

private:
    void run();

    CommandBuffer& commandBuffer;
    GenerationManager& generationManager;
    const ThreadAffinityManager* affinityManager = nullptr;
    WorkerThreadConfig config;



    std::atomic<bool> running{false};
    std::atomic<bool> pendingFlush{false};
    std::thread thread;

#ifdef _DEBUG
    std::atomic<uint64_t> commandsReceived{0};
    std::atomic<uint64_t> snapshotsCreated{0};
    std::atomic<uint64_t> commandsDropped{0};
#endif
};

} // namespace convo
