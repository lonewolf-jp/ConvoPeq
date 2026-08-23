// T1Measurement.cpp
// D101-9 Phase 9-B Step 5-III-B: T1 Baseline Measurement
//   Runs AudioEngineHarness in normal operation for a specified duration,
//   capturing [D101_9_T5_OBS] telemetry logs via stderr.
//
//   Called from PublishPipelineIntegrationTests.cpp main() with --t1 flag.
//   The --t1 flag triggers runT1BaselineMeasurement(durationSec).

#include "AudioEngineHarness.h"
#include "audioengine/AudioEngine.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

// ★ T1 fix: JUCE Logger that writes to stderr.
//   By default, Logger::writeToLog() and DBG() use outputDebugString() on Windows,
//   which requires a debugger attached. In a console app without a debugger, the output
//   is lost. This custom logger ensures all diagnostic output goes to stderr.
class StderrLogger : public juce::Logger
{
public:
    void logMessage(const juce::String& message) override
    {
        std::fprintf(stderr, "%s\n", message.toStdString().c_str());
        std::fflush(stderr);
    }
};

// T1: D101-9 Phase 9-B Step 5-III-B
//   Runs the AudioEngine in normal operation (no reader stall) for `durationSec` seconds.
//   The 100ms timerCallback emits [D101_9_T5_OBS] logs to stderr automatically.
//   No shutdown is performed — T1 measures the running engine baseline.
bool runT1BaselineMeasurement(int durationSec)
{
    // ★ T1 fix: Install a stderr logger so DBG() and Logger::writeToLog() output goes
    //   to stderr instead of OutputDebugString (which requires a debugger).
    static StderrLogger stderrLogger;
    juce::Logger::setCurrentLogger(&stderrLogger);

    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "T1: FAIL: harness start failed\n");
        return false;
    }

    // ★ T1 fix: Start MessageManager so AudioEngine's juce::Timer::timerCallback() fires.
    //   AudioEngine inherits juce::Timer and startTimer(100) is called in initialize().
    //   A JUCE Timer's callback is dispatched via the MessageManager's event loop, but
    //   the MessageManager must be created (getInstance) and runDispatchLoop() called on
    //   the SAME thread — that thread becomes the "message thread".
    //   Strategy: create MessageManager on the main thread (which becomes the message thread),
    //   then run the measurement loop on a background thread and run the dispatch loop
    //   on the main thread. When the measurement is done, call stopDispatchLoop() from
    //   the measurement thread to unblock runDispatchLoop() on the main thread.
    juce::MessageManager* messageManager = juce::MessageManager::getInstance();

    AudioEngine& e = h.engine();

    // Launch measurement loop on a background thread
    std::atomic<bool> measurementDone{false};
    std::thread measurementThread([&h, &e, durationSec, &measurementDone, messageManager]() {
        AudioEngineHarness& harness = h;

        // Wait for bootstrap publish to settle
        std::fprintf(stderr, "T1: waiting for bootstrap settle...\n");
        auto settleStart = std::chrono::steady_clock::now();
        while (e.getPublicationBacklogCount() != 0)
        {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - settleStart);
            if (elapsed.count() > 30)
            {
                std::fprintf(stderr, "T1: WARN: bootstrap settle timeout (backlog=%llu)\n",
                    static_cast<unsigned long long>(e.getPublicationBacklogCount()));
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        std::fprintf(stderr, "T1: bootstrap settled, starting %d-second measurement\n", durationSec);

        // ★ T1 condition: normal operation — periodic low-cadence publish (no reader stall)
        //   Publish every 500ms simulates light DAW idle operation.
        //   The audio thread (audioLoop) continuously processes audio blocks → normal retire cycle.
        //   The timerCallback (100ms) captures [D101_9_T5_OBS] telemetry.
        auto start = std::chrono::steady_clock::now();
        uint64_t publishCount = 0;

        // ★ Resolve active DSP from published world (publishIdleWorldOnly returns false for nullptr)
        AudioEngine::DSPCore* activeDSP = nullptr;
        if (const auto* w = e.observePublishedWorld())
            activeDSP = static_cast<AudioEngine::DSPCore*>(w->engine.current);

        while (true)
        {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start);
            if (elapsed.count() >= durationSec)
                break;

            // Issue a periodic idle publish at low cadence (simulates normal DAW operation)
            (void)e.publishIdleWorldOnly(activeDSP, convo::TransitionPolicy::SmoothOnly);
            publishCount++;

            // Sleep until next publish interval (but check time continuously for duration limit)
            auto nextPublish = now + std::chrono::milliseconds(500);
            while (std::chrono::steady_clock::now() < nextPublish)
            {
                auto checkTime = std::chrono::steady_clock::now();
                auto remaining = std::chrono::duration_cast<std::chrono::seconds>(checkTime - start);
                if (remaining.count() >= durationSec)
                    goto done;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }

    done:
        std::fprintf(stderr, "T1: measurement complete. publishes=%llu duration=%ds\n",
            static_cast<unsigned long long>(publishCount), durationSec);

        // ★ T1: shutdown NOT performed. Observe terminal state for drainAll accounting.
        //   The [D101_9_T5_OBS] logs are already captured via stderr during the run.
        //   We do NOT call h.stop() — that would trigger shutdown drain.
        //   T1 is about running-engine baseline, not shutdown behavior.
        //   The process exit will dump all logs captured during the run.

        std::fprintf(stderr, "T1: DONE (engine left running for log capture)\n");

        // Signal the MessageManager dispatch loop on the main thread to stop.
        messageManager->stopDispatchLoop();
        measurementDone = true;
    });

    // ★ Main thread runs the MessageManager dispatch loop — this is required for
    //   juce::Timer::timerCallback() to fire. Timer callbacks are dispatched here.
    messageManager->runDispatchLoop();

    // Wait for the measurement thread to finish
    if (measurementThread.joinable())
        measurementThread.join();
    // ★ T1 fix: Restore default logger (no-op, stderrLogger is static)
    juce::Logger::setCurrentLogger(nullptr);
    // Note: intentionally not calling h.stop() to avoid shutdown drain in T1.
    // The process will exit and all stderr output is captured by the test runner.
    return true;
}
