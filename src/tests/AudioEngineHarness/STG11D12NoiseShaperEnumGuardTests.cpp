// STG11D12NoiseShaperEnumGuardTests.cpp - STG-11-D12-2 regression.
//
// Target defect (STG-11-D12-2):
//   AudioEngine::setNoiseShaperType stored its argument unconditionally.
//   The authoritative legal set is 0..3 (NoiseShaperType in src/core/Types.h,
//   matched by RuntimePublicationValidator.cpp:128-131), but the setter had no
//   range normalisation. The session restore path (AudioEngine.StateIO.cpp:
//   107-113) carries an inline guard, while the device settings XML restore
//   (DeviceSettings.cpp:1140-1141) casts unchecked, so the two restore paths
//   disagreed and an out-of-range enum reached the setter. Same structure as
//   D12-1: the bad value was written to the noiseShaperType /
//   m_currentNoiseShaperType atomics before any publish decision,
//   captureBuildParameterSnapshot (RebuildDispatch.cpp:49) re-read it on every
//   rebuild, validateResources rejected the world, publishWorld discarded it,
//   and nothing restored the value, so the rebuild/publish path stayed
//   permanently rejected.
//
// Fix:
//   Reject out-of-range enums at the setter boundary, keeping the previous
//   value (the D7 / D10 / D12-1 contract). No clamp. The StateIO.cpp inline
//   guard is kept as defence in depth (its negative control stays intact) and
//   the DeviceSettings.cpp cast is left untouched (now unreachable).
//
// Test contract:
//   D12-2-A  Invalid negative enum (-1) via setter and via session restore
//            leaves the authoritative value unchanged.
//   D12-2-B  Invalid enum above the range (4, 99) via setter and via session
//            restore leaves the authoritative value unchanged.
//   D12-2-C  Valid enums (0, 1, 2, 3) are accepted and round-trip unchanged.
//   D12-2-D  After an invalid enum, a subsequent normal structural parameter
//            change still reaches a committed publication. This is the core
//            regression for the sticky-reject defect.
//
// Negative control:
//   Removing the setter guard makes D12-2-A, D12-2-B and D12-2-D fail.
//
// No new CTest registration: sub-test of AudioEngineHarness, STG-8/9/D3-D5/D7-D12
// shape. The fixture is owned by the harness (AudioEngine ~4MB must not live on
// TU stack).
// =============================================================================

#include <cstdio>
#include <chrono>
#include <thread>

#include "audioengine/AudioEngine.h"
#include "AudioEngineHarness.h"

namespace {

using NoiseShaper = AudioEngine::NoiseShaperType;

[[nodiscard]] int toInt(NoiseShaper type) noexcept
{
    return static_cast<int>(type);
}

template <typename Predicate>
bool waitUntil(double timeoutSeconds, Predicate pred)
{
    const auto deadline = std::chrono::steady_clock::now()
                        + std::chrono::milliseconds(static_cast<long long>(timeoutSeconds * 1000.0));
    while (std::chrono::steady_clock::now() < deadline)
    {
        if (pred())
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return pred();
}

// ── D12-2-A: invalid negative enum never reaches the authoritative value ──
bool checkD12T1NegativeEnumRejected()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-11-D12-2 T-1] harness start failed\n");
        return false;
    }
    h.stop();
    AudioEngine& e = h.engine();

    // Establish a known legal baseline through the setter.
    e.setNoiseShaperType(NoiseShaper::Fixed4Tap);
    if (e.getNoiseShaperType() != NoiseShaper::Fixed4Tap)
    {
        std::fprintf(stderr, "D12-2-A: legal baseline Fixed4Tap not accepted (got %d)\n",
                     toInt(e.getNoiseShaperType()));
        return false;
    }

    bool ok = true;

    // Path 1: the setter itself must reject.
    e.setNoiseShaperType(static_cast<NoiseShaper>(-1));
    if (e.getNoiseShaperType() != NoiseShaper::Fixed4Tap)
    {
        std::fprintf(stderr, "D12-2-A: setter accepted -1 (value now %d)\n",
                     toInt(e.getNoiseShaperType()));
        ok = false;
    }

    // Path 2: the session restore path must not be a way around the setter.
    // (StateIO.cpp carries its own inline guard; this asserts the parity holds.)
    const juce::ValueTree saved = e.getCurrentState();
    if (!saved.isValid())
    {
        std::fprintf(stderr, "D12-2-A: getCurrentState invalid\n");
        return false;
    }
    {
        juce::ValueTree v = saved.createCopy();
        v.setProperty("noiseShaperType", -1, nullptr);
        e.requestLoadState(v);
        if (e.getNoiseShaperType() != NoiseShaper::Fixed4Tap)
        {
            std::fprintf(stderr, "D12-2-A: session restore applied -1 (value now %d)\n",
                         toInt(e.getNoiseShaperType()));
            ok = false;
        }
    }

    if (ok)
        std::printf("STG11D12NoiseShaperEnumGuardTests: D12-2-A PASS (negative enum rejected)\n");
    return ok;
}

// ── D12-2-B: invalid above-range enums never reach the authoritative value ──
bool checkD12T2AboveRangeEnumRejected()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-11-D12-2 T-2] harness start failed\n");
        return false;
    }
    h.stop();
    AudioEngine& e = h.engine();

    e.setNoiseShaperType(NoiseShaper::Fixed4Tap);
    if (e.getNoiseShaperType() != NoiseShaper::Fixed4Tap)
    {
        std::fprintf(stderr, "D12-2-B: legal baseline Fixed4Tap not accepted (got %d)\n",
                     toInt(e.getNoiseShaperType()));
        return false;
    }

    bool ok = true;
    const juce::ValueTree saved = e.getCurrentState();
    if (!saved.isValid())
    {
        std::fprintf(stderr, "D12-2-B: getCurrentState invalid\n");
        return false;
    }

    for (const int bad : { 4, 99 })
    {
        e.setNoiseShaperType(static_cast<NoiseShaper>(bad));
        if (e.getNoiseShaperType() != NoiseShaper::Fixed4Tap)
        {
            std::fprintf(stderr, "D12-2-B: setter accepted %d (value now %d)\n",
                         bad, toInt(e.getNoiseShaperType()));
            ok = false;
        }

        juce::ValueTree v = saved.createCopy();
        v.setProperty("noiseShaperType", bad, nullptr);
        e.requestLoadState(v);
        if (e.getNoiseShaperType() != NoiseShaper::Fixed4Tap)
        {
            std::fprintf(stderr, "D12-2-B: session restore applied %d (value now %d)\n",
                         bad, toInt(e.getNoiseShaperType()));
            ok = false;
        }
    }

    if (ok)
        std::printf("STG11D12NoiseShaperEnumGuardTests: D12-2-B PASS (above-range enums rejected)\n");
    return ok;
}

// ── D12-2-C: valid enums are accepted and round-trip ──
bool checkD12T3ValidRoundTrip()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-11-D12-2 T-3] harness start failed\n");
        return false;
    }
    h.stop();
    AudioEngine& e = h.engine();

    bool ok = true;
    const NoiseShaper kValid[] = {
        NoiseShaper::Psychoacoustic,
        NoiseShaper::Fixed4Tap,
        NoiseShaper::Adaptive9thOrder,
        NoiseShaper::Fixed15Tap,
    };
    for (const NoiseShaper legal : kValid)
    {
        e.setNoiseShaperType(legal);
        if (e.getNoiseShaperType() != legal)
        {
            std::fprintf(stderr, "D12-2-C: setter rejected legal %d (value now %d)\n",
                         toInt(legal), toInt(e.getNoiseShaperType()));
            ok = false;
            continue;
        }

        // Round-trip through the session serialisation.
        const juce::ValueTree saved = e.getCurrentState();
        e.setNoiseShaperType(legal == NoiseShaper::Fixed4Tap
                             ? NoiseShaper::Psychoacoustic
                             : NoiseShaper::Fixed4Tap);
        e.requestLoadState(saved);
        if (e.getNoiseShaperType() != legal)
        {
            std::fprintf(stderr, "D12-2-C: round-trip lost legal %d (value now %d)\n",
                         toInt(legal), toInt(e.getNoiseShaperType()));
            ok = false;
        }
    }

    if (ok)
        std::printf("STG11D12NoiseShaperEnumGuardTests: D12-2-C PASS (valid enums round-trip)\n");
    return ok;
}

// ── D12-2-D: a later structural change still publishes after an invalid enum ──
// Same sticky-reject core as D12-1-C. Before the setter guard, the invalid enum
// stayed in the atomics, every subsequent rebuild produced a world that
// validateResources rejected, and the committed publication sequence stopped
// advancing for the rest of the engine lifetime.
bool checkD12T4PublishRecoversAfterInvalid()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-11-D12-2 T-4] harness start failed\n");
        return false;
    }
    AudioEngine& e = h.engine();

    auto committedSeq = [&e] {
        return static_cast<unsigned long long>(e.getLastCommittedPublicationSequence());
    };

    // Let the startup publications settle so the baseline is stable.
    unsigned long long previous = committedSeq();
    const bool settled = waitUntil(20.0, [&] {
        const unsigned long long now = committedSeq();
        if (now == previous)
        {
            previous = now;
            return true;
        }
        previous = now;
        return false;
    });
    if (!settled)
    {
        std::fprintf(stderr, "D12-2-D: startup publications did not settle\n");
        return false;
    }

    // A legal value, to prove the pipeline is live before the invalid input.
    e.setNoiseShaperType(NoiseShaper::Fixed15Tap);
    if (!waitUntil(20.0, [&] { return committedSeq() > previous; }))
    {
        std::fprintf(stderr, "D12-2-D: legal noise shaper change did not publish (seq=%llu)\n",
                     committedSeq());
        return false;
    }
    previous = committedSeq();

    // Inject an invalid enum through the setter (the DeviceSettings XML path
    // reaches the same setter, so this covers the defect path).
    e.setNoiseShaperType(static_cast<NoiseShaper>(7));
    if (e.getNoiseShaperType() != NoiseShaper::Fixed15Tap)
    {
        std::fprintf(stderr, "D12-2-D: invalid enum was stored (value now %d)\n",
                     toInt(e.getNoiseShaperType()));
        return false;
    }

    // A normal structural parameter change must still reach a committed world.
    e.setSaturationAmount(0.5f);
    if (!waitUntil(20.0, [&] { return committedSeq() > previous; }))
    {
        std::fprintf(stderr, "D12-2-D: publish stayed closed after an invalid enum (seq=%llu)\n",
                     committedSeq());
        return false;
    }

    // The published world must carry an in-set noise shaper type.
    const auto* world = e.observePublishedWorld();
    if (world == nullptr)
    {
        std::fprintf(stderr, "D12-2-D: no published world\n");
        return false;
    }
    const int published = world->resource.noiseShaperType;
    if (published < 0 || published > 3)
    {
        std::fprintf(stderr, "D12-2-D: published world carries noiseShaperType=%d\n", published);
        return false;
    }

    std::printf("STG11D12NoiseShaperEnumGuardTests: D12-2-D PASS (publish survives an invalid enum)\n");
    return true;
}

} // namespace

int runSTG11D12NoiseShaperEnumGuardTests()
{
    bool ok = true;

    if (!checkD12T1NegativeEnumRejected())
    {
        std::fprintf(stderr, "FAIL: D12-2-A negative enum rejected\n");
        ok = false;
    }
    if (!checkD12T2AboveRangeEnumRejected())
    {
        std::fprintf(stderr, "FAIL: D12-2-B above-range enums rejected\n");
        ok = false;
    }
    if (!checkD12T3ValidRoundTrip())
    {
        std::fprintf(stderr, "FAIL: D12-2-C valid round-trip\n");
        ok = false;
    }
    if (!checkD12T4PublishRecoversAfterInvalid())
    {
        std::fprintf(stderr, "FAIL: D12-2-D publish survives invalid enum\n");
        ok = false;
    }

    if (ok)
        std::printf("STG11D12NoiseShaperEnumGuardTests: PASS (D12-2-A/D12-2-B/D12-2-C/D12-2-D)\n");
    return ok ? 0 : 1;
}
