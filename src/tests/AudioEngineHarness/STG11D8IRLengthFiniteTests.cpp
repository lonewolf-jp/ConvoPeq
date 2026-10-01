// STG11D8IRLengthFiniteTests.cpp - STG-11-D8-1 regression (T1 .. T3).
//
// Target defect (STG-11-D8-1):
//   ConvolverProcessor::setTargetIRLength / applyAutoDetectedIRLength stored
//   the juce::jlimit result unconditionally. jlimit passes NaN through
//   (comparisons are false -> input returned), so a corrupted session's NaN
//   irLength was stored as-is. computeTargetIRLength then computes
//   (int)(sampleRate * NaN) -> INT_MIN -> min(..., kMaxIRCap) -> max(..., 1),
//   i.e. target 1 sample, and the LoaderThread trims the IR to 1 sample:
//   silent user-data destruction.
//
//   Fix: reject non-finite inputs at both setters (existing
//   convo::numeric_policy::isFinite, bit-pattern, fp:fast safe), keeping the
//   previous finite value.
//
// Test contract:
//   D8-T1  NaN irLength via setState -> finite value retained;
//           computeTargetIRLength sane (not 1 from NaN).
//   D8-T2  Valid round-trip (1.5s set -> save -> restore -> match).
//   D8-T3  +Inf / -Inf rejected as well.
//   Negative control: with the guards removed, D8-T1 FAILs (NaN stored).
//
// No new CTest registration: sub-test of AudioEngineHarness.
// Fixture owned by the harness.
// =============================================================================

#include <cmath>
#include <cstdio>
#include <limits>

#include "audioengine/AudioEngine.h"
#include "AudioEngineHarness.h"
#include "ConvolverProcessor.h"

namespace {

bool checkD8T1NaNRejected()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-11-D8 T-1] harness start failed\n");
        return false;
    }
    h.stop();
    ConvolverProcessor& conv = h.engine().getConvolverProcessor();

    // Establish a known finite baseline.
    conv.setTargetIRLength(1.5f);
    const float baseline = conv.getTargetIRLength();
    if (!(baseline > 0.0f && baseline < 100.0f))
    {
        std::fprintf(stderr, "D8-T1: baseline not finite (%f)\n", baseline);
        return false;
    }

    // Corrupted session: NaN irLength through the real setState path.
    juce::ValueTree v("Convolver");
    v.setProperty("irLength", std::numeric_limits<float>::quiet_NaN(), nullptr);
    conv.setState(v);

    const float after = conv.getTargetIRLength();
    if (!(after > 0.0f && after < 100.0f))
    {
        std::fprintf(stderr, "D8-T1: NaN irLength stored (%f), baseline was %f\n", after, baseline);
        return false;
    }

    // Downstream consequence (verified by code trace, not called here because
    // computeTargetIRLength is private): a stored NaN makes
    // (int)(sampleRate * NaN) -> INT_MIN -> min(..., kMaxIRCap) -> max(..., 1),
    // i.e. the LoaderThread trims the IR to 1 sample. Keeping the stored value
    // finite therefore preserves the whole downstream chain.

    // NaN auto-detected length is rejected the same way.
    juce::ValueTree v2("Convolver");
    v2.setProperty("irLengthManualOverride", false, nullptr);
    v2.setProperty("autoDetectedIRLength", std::numeric_limits<float>::quiet_NaN(), nullptr);
    conv.setState(v2);
    const float autoAfter = conv.getAutoDetectedIRLength();
    if (!(autoAfter > 0.0f && autoAfter < 1000.0f))
    {
        std::fprintf(stderr, "D8-T1: NaN autoDetectedIRLength stored (%f)\n", autoAfter);
        return false;
    }

    std::printf("STG11D8IRLengthFiniteTests: D8-T1 PASS (NaN irLength rejected)\n");
    return true;
}

bool checkD8T2ValidRoundTrip()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-11-D8 T-2] harness start failed\n");
        return false;
    }
    h.stop();
    ConvolverProcessor& conv = h.engine().getConvolverProcessor();

    conv.setTargetIRLength(1.5f);
    juce::ValueTree saved("Convolver");
    saved.setProperty("irLength", conv.getTargetIRLength(), nullptr);
    saved.setProperty("irLengthManualOverride", true, nullptr);
    conv.setTargetIRLength(0.5f);
    conv.setState(saved);

    const float restored = conv.getTargetIRLength();
    if (!(restored > 1.49f && restored < 1.51f))
    {
        std::fprintf(stderr, "D8-T2: valid round-trip broken (%f)\n", restored);
        return false;
    }
    std::printf("STG11D8IRLengthFiniteTests: D8-T2 PASS (valid round-trip kept)\n");
    return true;
}

bool checkD8T3InfRejected()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-11-D8 T-3] harness start failed\n");
        return false;
    }
    h.stop();
    ConvolverProcessor& conv = h.engine().getConvolverProcessor();

    conv.setTargetIRLength(1.5f);

    juce::ValueTree vPos("Convolver");
    vPos.setProperty("irLength", std::numeric_limits<float>::infinity(), nullptr);
    conv.setState(vPos);
    const float afterPos = conv.getTargetIRLength();
    if (!(afterPos > 0.0f && afterPos < 100.0f))
    {
        std::fprintf(stderr, "D8-T3: +Inf irLength stored (%f)\n", afterPos);
        return false;
    }

    juce::ValueTree vNeg("Convolver");
    vNeg.setProperty("irLength", -std::numeric_limits<float>::infinity(), nullptr);
    conv.setState(vNeg);
    const float afterNeg = conv.getTargetIRLength();
    if (!(afterNeg > 0.0f && afterNeg < 100.0f))
    {
        std::fprintf(stderr, "D8-T3: -Inf irLength stored (%f)\n", afterNeg);
        return false;
    }

    std::printf("STG11D8IRLengthFiniteTests: D8-T3 PASS (Inf rejected)\n");
    return true;
}

} // namespace

int runSTG11D8IRLengthFiniteTests()
{
    bool ok = true;
    if (!checkD8T1NaNRejected())
    {
        std::fprintf(stderr, "FAIL: D8-T1 NaN rejected\n");
        ok = false;
    }
    if (!checkD8T2ValidRoundTrip())
    {
        std::fprintf(stderr, "FAIL: D8-T2 valid round-trip\n");
        ok = false;
    }
    if (!checkD8T3InfRejected())
    {
        std::fprintf(stderr, "FAIL: D8-T3 Inf rejected\n");
        ok = false;
    }
    if (ok)
        std::printf("STG11D8IRLengthFiniteTests: PASS (D8-T1/D8-T2/D8-T3)\n");
    return ok ? 0 : 1;
}
