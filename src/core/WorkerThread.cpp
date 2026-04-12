//==============================================================================
// WorkerThread.cpp
//==============================================================================

#include "WorkerThread.h"
#include "SnapshotCoordinator.h"
#include "ThreadAffinityManager.h"
#include "../AudioEngine.h"
#include "../GenerationManager.h"

#include <chrono>
#include <thread>

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
 #include <xmmintrin.h>
 #include <pmmintrin.h>
#endif

namespace convo {

WorkerThread::WorkerThread(CommandBuffer& cmdBuf,
                           SnapshotCoordinator& coord,
                           GenerationManager& genMgr,
                   AudioEngine& engine,
                           const WorkerThreadConfig& cfg)
    : commandBuffer(cmdBuf),
      coordinator(coord),
      generationManager(genMgr),
    audioEngine(engine),
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
    running.store(false, std::memory_order_release);
    pendingFlush.store(true, std::memory_order_release);

    if (thread.joinable())
        thread.join();
}

void WorkerThread::flush()
{
    pendingFlush.store(true, std::memory_order_release);
}

void WorkerThread::run()
{
    audioEngine.getAffinityManager().applyCurrentThreadPolicy(ThreadType::Worker);

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

    bool hasPending = false;
    uint64_t pendingGeneration = 0;
    auto lastCommandTime = std::chrono::steady_clock::now();

    while (running.load(std::memory_order_acquire)) {
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

        // flush 要求があれば、コマンドがなくても処理を試みる
        if (flushRequested || poppedAny) {
            hasPending = true; // poppedAny が true なら pendingGeneration は有効
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
            std::this_thread::sleep_for(std::chrono::milliseconds(config.idleSleepMs));
        }
    }
}

} // namespace convo
