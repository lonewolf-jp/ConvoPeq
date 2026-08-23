// T3Measurement.cpp
// D101-9 Phase 9-B Step 5-III-C: T3 Long-Stall Measurement
//   Runs AudioEngineHarness with an intentional reader stall of `stallSec` seconds,
//   capturing [D101_9_T5_OBS] telemetry logs via stdout.
//
//   Called from PublishPipelineIntegrationTests.cpp main() with --t3 flag.
//   The --t3=<stall_sec> flag triggers runT3LongStallMeasurement(stallSec).
//
//   T3 procedure:
//     Phase 1: Pre-stall baseline publish (3 seconds)
//     Phase 2: Reader stall — enter reader, hold for stallSec, publish rapidly
//     Phase 3: Reader recovery — exit reader, observe Terminal drain
//     Phase 4: Post-recovery stable publish (3 seconds)
//
//   T3 extends T2 to long stalls (1s, 5s, 10s, 30s) to observe the full causal chain:
//     stall → minEpoch stagnation → Q/E pressure → Terminal arrival → Terminal growth
//     → reader recovery → minEpoch advance → Terminal drain

#include "AudioEngineHarness.h"
#include "audioengine/AudioEngine.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

// ★ Shared JUCE Logger that writes to stderr.
class StderrLogger : public juce::Logger
{
public:
    void logMessage(const juce::String& message) override
    {
        std::fprintf(stderr, "%s\n", message.toStdString().c_str());
        std::fflush(stderr);
    }
};

bool runT3LongStallMeasurement(int stallSec)
{
    // ★ Install stderr logger so DBG() and Logger::writeToLog() output goes to stderr
    static StderrLogger stderrLogger;
    juce::Logger::setCurrentLogger(&stderrLogger);

    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "T3: FAIL: harness start failed\n");
        return false;
    }

    // ★ Start MessageManager so AudioEngine's juce::Timer::timerCallback() fires
    juce::MessageManager* messageManager = juce::MessageManager::getInstance();

    AudioEngine& e = h.engine();

    // Launch measurement loop on a background thread
    std::atomic<bool> measurementDone{false};
    std::thread measurementThread([&h, &e, stallSec, &measurementDone, messageManager]() {
        // Wait for bootstrap publish to settle
        std::fprintf(stderr, "T3: waiting for bootstrap settle...\n");
        auto settleStart = std::chrono::steady_clock::now();
        while (e.getPublicationBacklogCount() != 0)
        {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - settleStart);
            if (elapsed.count() > 30)
            {
                std::fprintf(stderr, "T3: WARN: bootstrap settle timeout (backlog=%llu)\n",
                    static_cast<unsigned long long>(e.getPublicationBacklogCount()));
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        std::fprintf(stderr, "T3: bootstrap settled\n");

        auto runStart = std::chrono::steady_clock::now();
        uint64_t publishCount = 0;

        // ★ Resolve active DSP from published world (publishIdleWorldOnly returns false for nullptr)
        AudioEngine::DSPCore* activeDSP = nullptr;
        if (const auto* w = e.observePublishedWorld())
            activeDSP = static_cast<AudioEngine::DSPCore*>(w->engine.current);
        if (activeDSP == nullptr)
        {
            std::fprintf(stderr, "T3: FAIL: could not resolve active DSP\n");
            messageManager->stopDispatchLoop();
            return true;
        }
        std::fprintf(stderr, "T3: active DSP resolved\n");

        // Phase 1: Pre-stall baseline publish (3 seconds)
        std::fprintf(stderr, "T3: Phase 1 — pre-stall baseline publish (3s)\n");
        auto phase1End = runStart + std::chrono::seconds(3);
        while (std::chrono::steady_clock::now() < phase1End)
        {
            (void)e.publishIdleWorldOnly(activeDSP, convo::TransitionPolicy::SmoothOnly);
            publishCount++;
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }

        // Phase 2: Reader stall — enter reader, hold for stallSec, publish rapidly
        std::fprintf(stderr, "T3: Phase 2 — reader stall (%ds)\n", stallSec);
        auto& router = e.getRetireRouter();
        // ★ CRITICAL: Do NOT use registerReaderThread() — it performs a linear scan
        //   for the first kInactiveEpoch slot, which can collide with ConvolverProcessor's
        //   GlobalGuard slots (indices 2/3 via enterGlobalReader(2/3)). During a long
        //   stall, ConvolverProcessor operations will enter/exit reader slot 2 or 3,
        //   overwriting the stall reader's epoch and breaking the epoch-gated reclaim
        //   safety invariant (minEpoch advances instead of stagnating).
        //   Fix: reserve a slot at index 4+ (beyond ConvolverProcessor's reserved 2/3)
        //   to guarantee no collision.
        int readerIndex = 4;
        if (!router.reserveReaderThread(readerIndex))
        {
            // Fallback: try registerReaderThread if slot 4 is taken
            readerIndex = router.registerReaderThread();
            std::fprintf(stderr, "T3: WARN: slot 4 unavailable, using registerReaderThread() idx=%d\n", readerIndex);
        }
        router.enterReader(readerIndex);

        std::fprintf(stderr, "T3: reader entered (index=%d), activeReaders should be > 0\n", readerIndex);

        // Publish rapidly during stall to generate pressure that will show up
        // in the [D101_9_T5_OBS] timer callback (100ms interval).
        auto stallStart = std::chrono::steady_clock::now();
        auto stallEnd = stallStart + std::chrono::seconds(stallSec);
        while (std::chrono::steady_clock::now() < stallEnd)
        {
            (void)e.publishIdleWorldOnly(activeDSP, convo::TransitionPolicy::SmoothOnly);
            publishCount++;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // Phase 3: Recovery — exit reader, observe drain
        std::fprintf(stderr, "T3: Phase 3 — reader recovery\n");
        router.exitReader(readerIndex);
        std::fprintf(stderr, "T3: reader exited, waiting for Terminal drain...\n");

        // Phase 4: Post-recovery publish until Terminal drains
        auto recoveryStart = std::chrono::steady_clock::now();
        const auto maxRecoveryWait = std::chrono::seconds(stallSec + 10);
        while (std::chrono::steady_clock::now() < recoveryStart + maxRecoveryWait)
        {
            (void)e.publishIdleWorldOnly(activeDSP, convo::TransitionPolicy::SmoothOnly);
            publishCount++;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        std::fprintf(stderr, "T3: measurement complete. publishes=%llu stallSec=%d\n",
            static_cast<unsigned long long>(publishCount), stallSec);

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
