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

class SnapshotCoordinator;

using SnapshotCreatorCallback = void (*)(void* userData, uint64_t generation);

struct WorkerThreadConfig {
    int debounceDelayMs = 50;
    int idleSleepMs = 10;
};

class WorkerThread {
public:
    WorkerThread(CommandBuffer& cmdBuf,
                 SnapshotCoordinator& coordinator,
                 GenerationManager& genManager,
                 const ThreadAffinityManager* affinityMgr,
                 const WorkerThreadConfig& config = WorkerThreadConfig());
    ~WorkerThread();

    void start();
    void stop();

    void setSnapshotCreator(SnapshotCreatorCallback callback, void* userData) noexcept {
        convo::publishAtomic(callbackFunc, callback, std::memory_order_release);
        convo::publishAtomic(callbackUserData, userData, std::memory_order_release);
    }

#ifdef _DEBUG
    uint64_t getCommandsReceived() const noexcept { return convo::consumeAtomic(commandsReceived, std::memory_order_acquire); }
    uint64_t getSnapshotsCreated() const noexcept { return convo::consumeAtomic(snapshotsCreated, std::memory_order_acquire); }
    uint64_t getCommandsDropped() const noexcept { return convo::consumeAtomic(commandsDropped, std::memory_order_acquire); }
#endif

private:
    void run();

    CommandBuffer& commandBuffer;
    SnapshotCoordinator& coordinator;
    GenerationManager& generationManager;
    const ThreadAffinityManager* affinityManager = nullptr;
    WorkerThreadConfig config;

    std::atomic<SnapshotCreatorCallback> callbackFunc{nullptr};
    std::atomic<void*> callbackUserData{nullptr};

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
