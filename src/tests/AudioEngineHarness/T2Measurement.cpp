// T2Measurement.cpp
// D101-9 Phase 9-B Step 5-III-B: T2 Short-Stall Measurement
//   Runs AudioEngineHarness with an intentional reader stall of `stallMs` milliseconds,
//   capturing [D101_9_T5_OBS] telemetry logs via stdout.
//
//   Called from PublishPipelineIntegrationTests.cpp main() with --t2 flag.
//   The --t2=<stall_ms> flag triggers runT2ShortStallMeasurement(stallMs, durationSec).
//
//   T2 procedure:
//     1. Start engine, bootstrap settle (as T1)
//     2. Publish normally for 5 seconds (pre-stall baseline)
//     3. Enter reader stall: call enterRcuReader() but delay exitRcuReader() for stallMs ms
//     4. During stall, continue publishing to generate pressure
//     5. After stallMs, exit reader (recover) and observe recovery
//     6. Publish normally for 5 more seconds (post-stall recovery)
//
//   Acceptance criteria:
//     - During stall: activeReaders > 0, minEpoch stagnates, Q_resident/E_resident may rise
//     - After recovery: minEpoch advances, Q/E drain, Terminal stays at 0 (expected for short stalls)

#include "AudioEngineHarness.h"
#include "audioengine/AudioEngine.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <string>

// ★ Shared JUCE Logger that writes to stderr.
//   By default, Logger::writeToLog() and DBG() use outputDebugString() on Windows,
//   which requires a debugger attached. This custom logger ensures output to stderr.
class StderrLogger : public juce::Logger
{
public:
    void logMessage(const juce::String& message) override
    {
        std::fprintf(stderr, "%s\n", message.toStdString().c_str());
        std::fflush(stderr);
    }
};

bool runT2ShortStallMeasurement(int stallMs, int durationSec)
{
    // ★ Install stderr logger so DBG() and Logger::writeToLog() output goes to stderr
    static StderrLogger stderrLogger;
    juce::Logger::setCurrentLogger(&stderrLogger);

    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "T2: FAIL: harness start failed\n");
        return false;
    }

    // ★ Start MessageManager so AudioEngine's juce::Timer::timerCallback() fires
    juce::MessageManager* messageManager = juce::MessageManager::getInstance();

    AudioEngine& e = h.engine();

    // Launch measurement loop on a background thread
    std::atomic<bool> measurementDone{false};
    std::thread measurementThread([&h, &e, stallMs, durationSec, &measurementDone, messageManager]() {
        // Wait for bootstrap publish to settle
        std::fprintf(stderr, "T2: waiting for bootstrap settle...\n");
        auto settleStart = std::chrono::steady_clock::now();
        while (e.getPublicationBacklogCount() != 0)
        {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - settleStart);
            if (elapsed.count() > 30)
            {
                std::fprintf(stderr, "T2: WARN: bootstrap settle timeout (backlog=%llu)\n",
                    static_cast<unsigned long long>(e.getPublicationBacklogCount()));
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        std::fprintf(stderr, "T2: bootstrap settled\n");

        auto runStart = std::chrono::steady_clock::now();
        uint64_t publishCount = 0;

        // ★ Resolve active DSP from published world (publishIdleWorldOnly returns false for nullptr)
        AudioEngine::DSPCore* activeDSP = nullptr;
        if (const auto* w = e.observePublishedWorld())
            activeDSP = static_cast<AudioEngine::DSPCore*>(w->engine.current);
        if (activeDSP == nullptr)
        {
            std::fprintf(stderr, "T2: FAIL: could not resolve active DSP\n");
            messageManager->stopDispatchLoop();
            return true;
        }
        std::fprintf(stderr, "T2: active DSP resolved\n");

        // Phase 1: Pre-stall baseline publish (5 seconds)
        std::fprintf(stderr, "T2: Phase 1 — pre-stall baseline publish (5s)\n");
        auto phase1End = runStart + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < phase1End)
        {
            (void)e.publishIdleWorldOnly(activeDSP, convo::TransitionPolicy::SmoothOnly);
            publishCount++;
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }

        // Phase 2: Reader stall — enter reader, hold, then publish during stall
        std::fprintf(stderr, "T2: Phase 2 — reader stall (%dms)\n", stallMs);
        auto& router = e.getRetireRouter();
        // ★ CRITICAL: Do NOT use registerReaderThread() — it performs a linear scan
        //   for the first kInactiveEpoch slot, which can collide with ConvolverProcessor's
        //   GlobalGuard slots (indices 2/3 via enterGlobalReader(2/3)). During a stall,
        //   ConvolverProcessor operations will enter/exit reader slot 2 or 3, overwriting
        //   the stall reader's epoch and breaking the epoch-gated reclaim safety invariant.
        //   Fix: reserve a slot at index 4+ (beyond ConvolverProcessor's reserved 2/3).
        int readerIndex = 4;
        if (!router.reserveReaderThread(readerIndex))
        {
            // Fallback: try registerReaderThread if slot 4 is taken
            readerIndex = router.registerReaderThread();
            std::fprintf(stderr, "T2: WARN: slot 4 unavailable, using registerReaderThread() idx=%d\n", readerIndex);
        }
        router.enterReader(readerIndex);

        std::fprintf(stderr, "T2: reader entered (index=%d), activeReaders should be > 0\n", readerIndex);

        // Publish rapidly during stall to generate pressure that will show up
        // in the next [D101_9_T5_OBS] timer callback (100ms interval).
        auto stallStart = std::chrono::steady_clock::now();
        auto stallEnd = stallStart + std::chrono::milliseconds(stallMs);
        while (std::chrono::steady_clock::now() < stallEnd)
        {
            (void)e.publishIdleWorldOnly(activeDSP, convo::TransitionPolicy::SmoothOnly);
            publishCount++;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // Phase 3: Recovery — exit reader
        std::fprintf(stderr, "T2: Phase 3 — reader recovery\n");
        router.exitReader(readerIndex);
        std::fprintf(stderr, "T2: reader exited\n");

        // Phase 4: Post-stall recovery publish (remainder of duration)
        auto totalEnd = runStart + std::chrono::seconds(durationSec);
        std::fprintf(stderr, "T2: Phase 4 — post-stall recovery publish (%ds remaining)\n",
            static_cast<int>(std::chrono::duration_cast<std::chrono::seconds>(totalEnd - std::chrono::steady_clock::now()).count()));
        while (std::chrono::steady_clock::now() < totalEnd)
        {
            (void)e.publishIdleWorldOnly(activeDSP, convo::TransitionPolicy::SmoothOnly);
            publishCount++;
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }

        std::fprintf(stderr, "T2: measurement complete. publishes=%llu stallMs=%d\n",
            static_cast<unsigned long long>(publishCount), stallMs);

        // Signal the MessageManager dispatch loop on the main thread to stop
        messageManager->stopDispatchLoop();
        measurementDone = true;

        return true;
    });

    // ★ Main thread runs the MessageManager dispatch loop
    messageManager->runDispatchLoop();

    // Wait for the measurement thread to finish
    if (measurementThread.joinable())
        measurementThread.join();

    juce::Logger::setCurrentLogger(nullptr);
    return true;
}
