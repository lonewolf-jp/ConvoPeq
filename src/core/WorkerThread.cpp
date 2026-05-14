//==============================================================================
// WorkerThread.cpp
//==============================================================================

#include "WorkerThread.h"
#include "SnapshotCoordinator.h"
#include "ThreadAffinityManager.h"
#include "../GenerationManager.h"

#include <chrono>
#include <thread>

#include "audioengine/AtomicAccess.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
 #include <xmmintrin.h>
 #include <pmmintrin.h>
#endif

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

    convo::publishAtomic(running, true, std::memory_order_release);
    convo::publishAtomic(pendingFlush, false, std::memory_order_release);
    thread = std::thread(&WorkerThread::run, this);
}

void WorkerThread::stop()
{
    convo::publishAtomic(running, false, std::memory_order_release);
    convo::publishAtomic(pendingFlush, true, std::memory_order_release);

    if (thread.joinable())
        thread.join();
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
    uint64_t latestCommandGeneration = 0;
    auto lastCommandTime = std::chrono::steady_clock::now();

    while (convo::consumeAtomic(running, std::memory_order_acquire)) {
        ParameterCommand cmd;
        bool poppedAny = false;
        bool flushRequested = convo::exchangeAtomic(pendingFlush, false, std::memory_order_acq_rel);

        while (commandBuffer.pop(cmd)) {
            poppedAny = true;
#ifdef _DEBUG
            commandsReceived.fetch_add(1, std::memory_order_acq_rel);
#endif
            latestCommandGeneration = cmd.generation; // 最新の世代のみ保持
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

                SnapshotCreatorCallback cb = convo::consumeAtomic(callbackFunc, std::memory_order_acquire);
                void* userData = convo::consumeAtomic(callbackUserData, std::memory_order_acquire);

                if (cb && userData && generationManager.isCurrentGeneration(latestCommandGeneration)) {
                    cb(userData, latestCommandGeneration);
#ifdef _DEBUG
                    snapshotsCreated.fetch_add(1, std::memory_order_acq_rel);
#endif
                }
#ifdef _DEBUG
                else {
                    commandsDropped.fetch_add(1, std::memory_order_acq_rel);
                }
#endif
            }
        }

        if (!poppedAny) {
            const int sleepChunks = (config.idleSleepMs > 0)
                ? ((config.idleSleepMs + 1) / 2)
                : 0;

            for (int i = 0; i < sleepChunks; ++i) {
                if (!convo::consumeAtomic(running, std::memory_order_acquire))
                    break;

                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }
    }
}

} // namespace convo
