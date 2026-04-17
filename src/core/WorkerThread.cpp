//==============================================================================
// WorkerThread.cpp
//==============================================================================

#include "WorkerThread.h"
#include "SnapshotCoordinator.h"
#include "ThreadAffinityManager.h"
#include "../GenerationManager.h"

#include <chrono>
#include <thread>

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
 #include <xmmintrin.h>
 #include <pmmintrin.h>
#endif

extern std::atomic<bool> gShuttingDown;

namespace convo {

WorkerThread::WorkerThread(CommandBuffer& cmdBuf,
                           SnapshotCoordinator& coord,
                           GenerationManager& genMgr,
                   const ThreadAffinityManager* affinityMgr,
                           const WorkerThreadConfig& cfg)
    : commandBuffer(cmdBuf),
      coordinator(coord),
      generationManager(genMgr),
    affinityManager(affinityMgr),
      config(cfg)
{
}

WorkerThread::~WorkerThread()
{
    stop();
}

void WorkerThread::start()
{
    if (thread.joinable())
        return;

    running.store(true, std::memory_order_release);
    pendingFlush.store(false, std::memory_order_release);
    thread = std::thread(&WorkerThread::run, this);
}

void WorkerThread::stop()
{
    requestStop();
    pendingFlush.store(true, std::memory_order_release);

    if (thread.joinable())
        thread.join();
}

void WorkerThread::requestStop() noexcept
{
    running.store(false, std::memory_order_release);
}

void WorkerThread::flush()
{
    pendingFlush.store(true, std::memory_order_release);
}

void WorkerThread::run()
{
    if (affinityManager != nullptr)
        affinityManager->applyCurrentThreadPolicy(ThreadType::Worker);

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

    bool hasPending = false;
    uint64_t pendingGeneration = 0;
    auto lastCommandTime = std::chrono::steady_clock::now();

    while (running.load(std::memory_order_acquire)) {
        if (::gShuttingDown.load(std::memory_order_acquire))
            break;

        ParameterCommand cmd;
        bool poppedAny = false;
        bool flushRequested = pendingFlush.exchange(false, std::memory_order_acq_rel);

        while (commandBuffer.pop(cmd)) {
            poppedAny = true;
#ifdef _DEBUG
            commandsReceived.fetch_add(1, std::memory_order_relaxed);
#endif
            pendingGeneration = cmd.generation; // 最新の世代のみ保持
            lastCommandTime = std::chrono::steady_clock::now();
        }

        if (poppedAny)
            hasPending = true;

        // flush 要求または pending のデバウンス満了時にスナップショットを生成する。
        if (flushRequested || hasPending) {
            const auto now = std::chrono::steady_clock::now();
            const auto elapsedMs = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                now - lastCommandTime).count());

            if (flushRequested || elapsedMs >= config.debounceDelayMs) {
                hasPending = false;

                SnapshotCreatorCallback cb = callbackFunc.load(std::memory_order_acquire);
                void* userData = callbackUserData.load(std::memory_order_acquire);

                if (cb && userData && generationManager.isCurrentGeneration(pendingGeneration)) {
                    cb(userData, pendingGeneration);
#ifdef _DEBUG
                    snapshotsCreated.fetch_add(1, std::memory_order_relaxed);
#endif
                }
#ifdef _DEBUG
                else {
                    commandsDropped.fetch_add(1, std::memory_order_relaxed);
                }
#endif
            }
        }

        if (!poppedAny) {
            const int sleepChunks = (config.idleSleepMs > 0)
                ? ((config.idleSleepMs + 1) / 2)
                : 0;

            for (int i = 0; i < sleepChunks; ++i) {
                if (!running.load(std::memory_order_acquire) ||
                    ::gShuttingDown.load(std::memory_order_acquire))
                    break;

                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }
    }
}

} // namespace convo
