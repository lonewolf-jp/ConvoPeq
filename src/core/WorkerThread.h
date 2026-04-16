//==============================================================================
// WorkerThread.h
// Debounced snapshot update worker
//==============================================================================
#pragma once

#include <atomic>
#include <chrono>
#include <thread>
#include "CommandBuffer.h"

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
    void requestStop() noexcept;
    void flush();

    void setSnapshotCreator(SnapshotCreatorCallback callback, void* userData) noexcept {
        callbackFunc.store(callback, std::memory_order_release);
        callbackUserData.store(userData, std::memory_order_release);
    }

    void setDebounceDelayMs(int ms) noexcept {
        if (ms < 0)
            ms = 0;
        config.debounceDelayMs = ms;
    }

#ifdef _DEBUG
    uint64_t getCommandsReceived() const noexcept { return commandsReceived.load(std::memory_order_relaxed); }
    uint64_t getSnapshotsCreated() const noexcept { return snapshotsCreated.load(std::memory_order_relaxed); }
    uint64_t getCommandsDropped() const noexcept { return commandsDropped.load(std::memory_order_relaxed); }
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
