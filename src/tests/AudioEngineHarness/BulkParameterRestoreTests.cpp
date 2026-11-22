// BulkParameterRestoreTests.cpp
// Issue #236 regression: a bulk parameter restore must collapse its structural
// setters into the single rebuild that endBulkParameterRestore() issues, and a
// setter called outside a bulk restore must still dispatch its own rebuild.
//
// Run: AudioEngineHarness.exe --bulk-restore-contract
// CTest: BulkParameterRestoreContract
//
// Observables are existing public readers only (no production change, no new
// counter, no new log prefix):
//   getRebuildDispatchDiagnostics().requestCount
//       incremented when requestRebuild() reaches the dispatch path
//       (AudioEngine.RebuildDispatch.cpp, before the queue decision).
//   getPublicationBacklogCount()
//       used to wait until the publisher is idle before a delta is taken.
//
// Contract:
//   A  setInputHeadroomDb() outside a bulk restore dispatches a rebuild and the
//      value is applied.
//   B  the six setters guarded for #236 (input headroom, output makeup, dither
//      bit depth, noise shaper type, oversampling factor, oversampling type)
//      dispatch nothing while a bulk restore is open.
//   C  endBulkParameterRestore(true) dispatches the collapsed rebuild and every
//      restored value is readable.
//   D  a setter called after the bulk restore dispatches again: the flag is
//      cleared, so the suppression cannot leak into normal editing.
//
// Negative control (mutation): replacing the `!m_isRestoringState` guards with
// `true` (the pre-fix behaviour) makes B report dispatches inside the restore
// and fail; D still passes.

#include "AudioEngineHarness.h"

#include "audioengine/AudioEngine.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <thread>

#if JUCE_WINDOWS
    #include <windows.h>
#endif

namespace {

constexpr double kSampleRate = 48000.0;
constexpr int    kBlockSize  = 512;

void pumpMessages() noexcept
{
   #if JUCE_WINDOWS
    MSG msg {};
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE))
    {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
   #endif
}

void sleepPump(int ms)
{
    const auto start = std::chrono::steady_clock::now();
    const auto budget = std::chrono::milliseconds(ms);
    while (std::chrono::steady_clock::now() - start < budget)
    {
        pumpMessages();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

long long dispatchRequests(AudioEngine& engine) noexcept
{
    return static_cast<long long>(engine.getRebuildDispatchDiagnostics().requestCount);
}

// Wait until the engine has been quiet for stableMs: no new dispatch request and
// no publication backlog. Returns false on timeout.
bool waitQuiet(AudioEngine& engine, int stableMs, int timeoutMs)
{
    const auto start = std::chrono::steady_clock::now();
    auto lastChange = start;
    long long last = dispatchRequests(engine);

    while (std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - start).count() < timeoutMs)
    {
        pumpMessages();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        const long long now = dispatchRequests(engine);
        if (now != last)
        {
            last = now;
            lastChange = std::chrono::steady_clock::now();
            continue;
        }

        const auto quietMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - lastChange).count();
        if (quietMs >= stableMs && engine.getPublicationBacklogCount() == 0)
            return true;
    }

    return false;
}

bool waitDispatchIncrease(AudioEngine& engine, long long base, int timeoutMs, long long& observed)
{
    const auto start = std::chrono::steady_clock::now();
    for (;;)
    {
        pumpMessages();
        observed = dispatchRequests(engine);
        if (observed > base)
            return true;

        if (std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count() >= timeoutMs)
            return false;

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

// Keep the probe inside the setter's own clamp range, so a target that clamps
// back to the current value can never look like a missing restore.
float shiftHeadroomDb(float current) noexcept
{
    return current > -11.0f ? current - 1.0f : current + 1.0f;
}

float shiftMakeupDb(float current) noexcept
{
    return current <= 11.0f ? current + 1.0f : current - 1.0f;
}

int nextDitherBitDepth(int current) noexcept
{
    return current == 24 ? 16 : 24;
}

int nextOversamplingFactor(int current) noexcept
{
    return current == 2 ? 4 : 2;
}

AudioEngine::NoiseShaperType nextNoiseShaperType(AudioEngine::NoiseShaperType current) noexcept
{
    return current == AudioEngine::NoiseShaperType::Fixed4Tap
        ? AudioEngine::NoiseShaperType::Psychoacoustic
        : AudioEngine::NoiseShaperType::Fixed4Tap;
}

AudioEngine::OversamplingType nextOversamplingType(AudioEngine::OversamplingType current) noexcept
{
    return current == AudioEngine::OversamplingType::IIR
        ? AudioEngine::OversamplingType::LinearPhase
        : AudioEngine::OversamplingType::IIR;
}

} // namespace

int runBulkParameterRestoreContract()
{
    // The rebuild dispatch path is Message Thread only, so the harness needs a
    // message manager owned by this thread. It is released explicitly after the
    // harness is gone, so JUCE shuts down after every engine object.
    auto juceInitialiser = std::make_unique<juce::ScopedJuceInitialiser_GUI>();

    int failures = 0;
    const auto fail = [&failures](const char* what, long long value)
    {
        ++failures;
        std::fprintf(stderr, "[BULK-RESTORE] FAIL: %s (value=%lld)\n", what, value);
    };

    {
        AudioEngineHarness harness;
        if (!harness.start(kSampleRate, kBlockSize))
        {
            fail("harness start", 0);
            juceInitialiser.reset();
            return 1;
        }

        AudioEngine& engine = harness.engine();

        // ---- A: a setter outside a bulk restore still dispatches a rebuild ----
        if (!waitQuiet(engine, 300, 20000))
        {
            fail("engine did not become quiet before check A", dispatchRequests(engine));
            harness.stop();
            juceInitialiser.reset();
            return 1;
        }

        {
            const float target = shiftHeadroomDb(engine.getInputHeadroomDb());
            const long long base = dispatchRequests(engine);
            engine.setInputHeadroomDb(target);

            long long after = base;
            if (!waitDispatchIncrease(engine, base, 5000, after))
                fail("A: setter outside a bulk restore dispatched no rebuild", after - base);

            if (std::abs(engine.getInputHeadroomDb() - target) > 1.0e-4f)
                fail("A: input headroom value not applied", 0);
        }

        // ---- B/C: the six guarded setters collapse into one rebuild ----
        if (!waitQuiet(engine, 300, 20000))
        {
            fail("engine did not become quiet before check B", dispatchRequests(engine));
            harness.stop();
            juceInitialiser.reset();
            return 1;
        }

        const float targetHeadroom = shiftHeadroomDb(engine.getInputHeadroomDb());
        const float targetMakeup = shiftMakeupDb(engine.getOutputMakeupDb());
        const int targetDither = nextDitherBitDepth(engine.getDitherBitDepth());
        const auto targetShaper = nextNoiseShaperType(engine.getNoiseShaperType());
        const int targetOversampling = nextOversamplingFactor(engine.getOversamplingFactor());
        const auto targetOversamplingType = nextOversamplingType(engine.getOversamplingType());

        {
            const long long base = dispatchRequests(engine);

            engine.beginBulkParameterRestore();
            engine.setInputHeadroomDb(targetHeadroom);
            engine.setOutputMakeupDb(targetMakeup);
            engine.setDitherBitDepth(targetDither);
            engine.setNoiseShaperType(targetShaper);
            engine.setOversamplingFactor(targetOversampling);
            engine.setOversamplingType(targetOversamplingType);

            // Give an unwanted dispatch time to surface while the restore is open.
            sleepPump(400);
            const long long during = dispatchRequests(engine);
            if (during != base)
                fail("B: setters inside a bulk restore dispatched rebuilds", during - base);

            engine.endBulkParameterRestore(true);

            long long after = during;
            if (!waitDispatchIncrease(engine, base, 5000, after))
                fail("C: endBulkParameterRestore dispatched no rebuild", after - base);
        }

        // ---- C: every restored value is readable ----
        if (std::abs(engine.getInputHeadroomDb() - targetHeadroom) > 1.0e-4f)
            fail("C: input headroom not restored", 0);
        if (std::abs(engine.getOutputMakeupDb() - targetMakeup) > 1.0e-4f)
            fail("C: output makeup not restored", 0);
        if (engine.getDitherBitDepth() != targetDither)
            fail("C: dither bit depth not restored", engine.getDitherBitDepth());
        if (engine.getNoiseShaperType() != targetShaper)
            fail("C: noise shaper type not restored",
                 static_cast<long long>(engine.getNoiseShaperType()));
        if (engine.getOversamplingFactor() != targetOversampling)
            fail("C: oversampling factor not restored", engine.getOversamplingFactor());
        if (engine.getOversamplingType() != targetOversamplingType)
            fail("C: oversampling type not restored",
                 static_cast<long long>(engine.getOversamplingType()));

        // ---- D: the suppression does not leak into normal editing ----
        if (!waitQuiet(engine, 300, 20000))
        {
            fail("engine did not become quiet before check D", dispatchRequests(engine));
            harness.stop();
            juceInitialiser.reset();
            return 1;
        }

        {
            const float target = shiftHeadroomDb(engine.getInputHeadroomDb());
            const long long base = dispatchRequests(engine);
            engine.setInputHeadroomDb(target);

            long long after = base;
            if (!waitDispatchIncrease(engine, base, 5000, after))
                fail("D: setter after a bulk restore dispatched no rebuild", after - base);
        }

        harness.stop();
    } // the engine and the harness are destroyed before JUCE shuts down

    juceInitialiser.reset();

    std::fprintf(stderr, "[BULK-RESTORE] %s failures=%d\n",
                 failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
