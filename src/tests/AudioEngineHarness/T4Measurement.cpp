// T4Measurement.cpp
// D101-9 Phase 9-B Step 5-III-E: T4 Repeated-Publish Measurement
//   Varies the publish load (sleep interval) during a FIXED 30 s reader stall to measure
//   the load-dependence of λ_terminal and validate the T3 empirical model
//   K ≈ λ_terminal × T_stall − (C_D + C_Q + C_E) across publish rates.
//
//   Called from PublishPipelineIntegrationTests.cpp main() with --t4=<interval_us>.
//
//   Spec cases (interval is the TARGET; actual publish rate is ground truth):
//     T4-A: 5000 us sleep   (~200 pub/s target)
//     T4-B: 3333 us sleep   (~300 pub/s target — T3 baseline equivalent)
//     T4-C: 1000 us sleep   (~1000 pub/s target)
//   Post-hoc: λ_publish = stallPublishes / actualStallDuration.
//
//   Stall duration is fixed at 30 s: T3-30s showed Terminal arrival at ~20 s, so 30 s
//   covers arrival → growth → peak for every load level. 60 s/120 s not required yet.
//
//   Methodology locked from T3 (slot-collision fix, Step 5-III-C §0):
//     - The stall reader MUST hold reader slot 4 (ConvolverProcessor::GlobalGuard owns
//       slots 2/3 via enterGlobalReader(2)/(3); registerReaderThread() linear scan may
//       grab slot 2 and get its epoch overwritten mid-stall).
//     - ★ Step 5-III-E hardening: if reserveReaderThread(4) fails, the run is INVALID
//       and is aborted immediately. NO fallback to registerReaderThread() — a fallback
//       slot could silently invalidate the epoch-stagnation premise.
//
//   Phases (unified with T3 for T3/T4 comparability):
//     Phase 1: baseline 3 s @ 500 ms
//     Phase 2: reader enter(slot 4) → 30 s stalled repeated publish at target load
//     Phase 3: reader exit
//     Phase 4: recovery ~10 s @ 100 ms (T3-30s drained fully within 79 ms of exit,
//              so 10 s is ample; mechanism identical to T3)
//
//   Run-level accounting is printed as a machine-parseable line:
//     T4_SUMMARY: intervalUs=<u> stallSec=<s> baselinePublishes=<n> stallPublishes=<n>
//                 actualStallUs=<u> lambdaPublish=<f> recoveryPublishes=<n> totalPublishes=<n>
//
//   Observation: existing [D101_9_T5_OBS] (100 ms timer tick) used as-is. Primary metrics
//   per Step 5-III-E §11: pendingRetire / terminalStoreCount / terminalResident /
//   terminalPeakResident. Q_resident OBS field is an aggregate (store + auxiliary counter)
//   and must NOT be used to infer Q capacity.

#include "AudioEngineHarness.h"
#include "audioengine/AudioEngine.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

// ★ Shared JUCE Logger that writes to stderr (same as T1/T2/T3).
class StderrLogger : public juce::Logger
{
public:
    void logMessage(const juce::String& message) override
    {
        std::fprintf(stderr, "%s\n", message.toStdString().c_str());
        std::fflush(stderr);
    }
};

// ★ File-scope constants (usable inside the measurement-thread lambda without capture)
inline constexpr int kT4StallSec = 30;          // Step 5-III-E: fixed 30 s stall
inline constexpr int kT4StallReaderSlot = 4;    // MANDATORY — see file header

bool runT4RepeatedPublishMeasurement(int intervalUs)
{
    // ★ Install stderr logger so DBG() and Logger::writeToLog() output goes to stderr
    static StderrLogger stderrLogger;
    juce::Logger::setCurrentLogger(&stderrLogger);

    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "T4: FAIL: harness start failed\n");
        return false;
    }

    // ★ Start MessageManager so AudioEngine's juce::Timer::timerCallback() fires
    //   (same main-thread dispatch-loop strategy as T1/T2/T3).
    juce::MessageManager* messageManager = juce::MessageManager::getInstance();

    AudioEngine& e = h.engine();

    std::atomic<bool> measurementDone{false};
    std::thread measurementThread([&h, &e, intervalUs, &measurementDone, messageManager]() {
        // Wait for bootstrap publish to settle
        std::fprintf(stderr, "T4: waiting for bootstrap settle...\n");
        auto settleStart = std::chrono::steady_clock::now();
        while (e.getPublicationBacklogCount() != 0)
        {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - settleStart);
            if (elapsed.count() > 30)
            {
                std::fprintf(stderr, "T4: WARN: bootstrap settle timeout (backlog=%llu)\n",
                    static_cast<unsigned long long>(e.getPublicationBacklogCount()));
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        std::fprintf(stderr, "T4: bootstrap settled\n");

        auto runStart = std::chrono::steady_clock::now();
        uint64_t baselinePublishes = 0;
        uint64_t stallPublishes = 0;
        uint64_t recoveryPublishes = 0;

        // ★ Resolve active DSP from published world (publishIdleWorldOnly no-ops on nullptr)
        AudioEngine::DSPCore* activeDSP = nullptr;
        if (const auto* w = e.observePublishedWorld())
            activeDSP = static_cast<AudioEngine::DSPCore*>(w->engine.current);
        if (activeDSP == nullptr)
        {
            std::fprintf(stderr, "T4: INVALID: could not resolve active DSP — aborting run\n");
            messageManager->stopDispatchLoop();
            measurementDone = true;
            return false;
        }
        std::fprintf(stderr, "T4: active DSP resolved\n");

        // ── Phase 1: Pre-stall baseline publish (3 s @ 500 ms) ──
        std::fprintf(stderr, "T4: Phase 1 — pre-stall baseline publish (3s)\n");
        auto phase1End = runStart + std::chrono::seconds(3);
        while (std::chrono::steady_clock::now() < phase1End)
        {
            (void)e.publishIdleWorldOnly(activeDSP, convo::TransitionPolicy::SmoothOnly);
            baselinePublishes++;
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }

        // ── Phase 2: Reader stall (fixed 30 s) with repeated publish at target load ──
        std::fprintf(stderr, "T4: Phase 2 — reader stall (%ds), target interval=%dus\n",
            kT4StallSec, intervalUs);
        auto& router = e.getRetireRouter();

        // ★★ MANDATORY slot 4 (Step 5-III-E §4): reserve failure ⇒ INVALID run.
        //   No fallback — a linear-scan slot could collide with GlobalGuard slots 2/3
        //   and silently invalidate the epoch-stagnation premise (see T3 evidence §0).
        if (!router.reserveReaderThread(kT4StallReaderSlot))
        {
            std::fprintf(stderr,
                "T4: INVALID: reserveReaderThread(%d) failed — measurement aborted "
                "(no fallback per Step 5-III-E methodology)\n", kT4StallReaderSlot);
            messageManager->stopDispatchLoop();
            measurementDone = true;
            return false;
        }
        router.enterReader(kT4StallReaderSlot);
        std::fprintf(stderr, "T4: reader entered (index=%d), activeReaders should be > 0\n",
            kT4StallReaderSlot);

        auto stallStart = std::chrono::steady_clock::now();
        auto stallEnd = stallStart + std::chrono::seconds(kT4StallSec);
        while (std::chrono::steady_clock::now() < stallEnd)
        {
            (void)e.publishIdleWorldOnly(activeDSP, convo::TransitionPolicy::SmoothOnly);
            stallPublishes++;
            if (intervalUs > 0)
                std::this_thread::sleep_for(std::chrono::microseconds(intervalUs));
        }
        const auto actualStallUs = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - stallStart).count();

        // ── Phase 3: Reader exit ──
        std::fprintf(stderr, "T4: Phase 3 — reader recovery\n");
        router.exitReader(kT4StallReaderSlot);
        std::fprintf(stderr, "T4: reader exited, waiting for Terminal drain...\n");

        // ★ Print run-level accounting BEFORE recovery so λ_publish survives even if a
        //   later stage hangs. λ_publish = stallPublishes / actualStallDuration.
        const double lambdaPublish =
            static_cast<double>(stallPublishes) * 1e6 / static_cast<double>(actualStallUs);
        std::fprintf(stderr,
            "T4_SUMMARY: intervalUs=%d stallSec=%d baselinePublishes=%llu "
            "stallPublishes=%llu actualStallUs=%lld lambdaPublish=%.2f\n",
            intervalUs, kT4StallSec,
            static_cast<unsigned long long>(baselinePublishes),
            static_cast<unsigned long long>(stallPublishes),
            static_cast<long long>(actualStallUs),
            lambdaPublish);

        // ── Phase 4: Recovery publish (~10 s @ 100 ms) — same mechanism as T3 ──
        auto recoveryStart = std::chrono::steady_clock::now();
        const auto maxRecoveryWait = std::chrono::seconds(10);
        while (std::chrono::steady_clock::now() < recoveryStart + maxRecoveryWait)
        {
            (void)e.publishIdleWorldOnly(activeDSP, convo::TransitionPolicy::SmoothOnly);
            recoveryPublishes++;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        const uint64_t totalPublishes =
            baselinePublishes + stallPublishes + recoveryPublishes;
        std::fprintf(stderr,
            "T4: measurement complete. totalPublishes=%llu recoveryPublishes=%llu stallSec=%d\n",
            static_cast<unsigned long long>(totalPublishes),
            static_cast<unsigned long long>(recoveryPublishes),
            kT4StallSec);

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
