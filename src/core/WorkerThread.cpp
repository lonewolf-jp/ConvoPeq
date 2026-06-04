//==============================================================================
// WorkerThread.cpp
//==============================================================================

#include "WorkerThread.h"
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
                           GenerationManager& genMgr,
                   const ThreadAffinityManager* affinityMgr,
                           const WorkerThreadConfig& cfg)
    : commandBuffer(cmdBuf),
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

    convo::publishAtomic(running, true, std::memory_order_release);    // release: run() の running acquire と HB しスレッド起動を公知
    convo::publishAtomic(pendingFlush, false, std::memory_order_release); // release: run() の exchangeAtomic acq_rel と HB し初期状態を公知
    thread = std::thread(&WorkerThread::run, this);
}

void WorkerThread::stop()
{
    convo::publishAtomic(running, false, std::memory_order_release);    // release: run() の running acquire と HB しスレッド停止を通知
    convo::publishAtomic(pendingFlush, true, std::memory_order_release); // release: run() の exchangeAtomic acq_rel と HB し flush 要求を公知

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

    while (convo::consumeAtomic(running, std::memory_order_acquire)) {  // acquire: start()/stop() の publishAtomic release と HB
        ParameterCommand cmd;
        bool poppedAny = false;
        bool flushRequested = convo::exchangeAtomic(pendingFlush, false, std::memory_order_acq_rel); // acq_rel: acquire で stop() の release と HB; release で次回観測と HB

        while (commandBuffer.pop(cmd)) {
            poppedAny = true;
#ifdef _DEBUG
            convo::fetchAddAtomic(commandsReceived, 1, std::memory_order_acq_rel); // acq_rel: getCommandsReceived acquire と HB — カウント単調増加を保証
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

                // SnapshotCreatorCallback removed (#3.2.8): snapshot creation is now
                // handled via submitRebuildIntent() directly from the intent source.
                // The worker thread no longer needs to create snapshots on demand.
                (void)generationManager;
                (void)latestCommandGeneration;
            }
        }

        if (!poppedAny) {
            const int sleepChunks = (config.idleSleepMs > 0)
                ? ((config.idleSleepMs + 1) / 2)
                : 0;

            for (int i = 0; i < sleepChunks; ++i) {
                if (!convo::consumeAtomic(running, std::memory_order_acquire))  // acquire: stop() の publishAtomic release と HB — 早期終了判定
                    break;

                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }
    }
}

} // namespace convo
