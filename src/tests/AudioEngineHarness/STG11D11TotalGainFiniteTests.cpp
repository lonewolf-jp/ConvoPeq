// STG11D11TotalGainFiniteTests.cpp - STG-11-D11-1 regression (T1 .. T3).
//
// Target defect (STG-11-D11-1):
//   EQProcessor::setTotalGain stored the juce::jlimit result unconditionally.
//   jlimit passes NaN through, so a corrupted session's NaN totalGain was
//   stored as-is (EQState.totalGainDb + totalGainDbTarget/totalGainTarget
//   atomics). prepareToPlay then ran
//   smoothTotalGain.setCurrentAndTargetValue(decibelsToGain(NaN)), NaN-ing the
//   RT gain ramp (NaN audio output). The Processing.cpp abs-gate cannot undo
//   this because NaN comparisons are false.
//
//   Fix: reject non-finite inputs at setTotalGain (existing
//   convo::numeric_policy::isFinite), keeping the previous finite value.
//
// Test contract:
//   D11-T1 NaN totalGain via setState -> finite retained.
//   D11-T2 Valid round-trip kept.
//   D11-T3 +Inf / -Inf rejected.
//   Negative control: with the guard removed, D11-T1 FAILs.
//
// No new CTest registration: sub-test of AudioEngineHarness.
// Fixture owned by the harness.
// =============================================================================

#include <cstdio>
#include <limits>

#include "audioengine/AudioEngine.h"
#include "AudioEngineHarness.h"
#include "eqprocessor/EQProcessor.h"

namespace {

static EQProcessor& eqOf(AudioEngine& e) { return e.getEQProcessor(); }

bool checkD11T1NaNRejected()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-11-D11 T-1] harness start failed\n");
        return false;
    }
    h.stop();
    EQProcessor& eq = eqOf(h.engine());

    eq.setTotalGain(3.0f);
    const float baseline = eq.getTotalGain();
    if (!(baseline > 2.9f && baseline < 3.1f))
    {
        std::fprintf(stderr, "D11-T1: baseline not established (%f)\n", baseline);
        return false;
    }

    juce::ValueTree v("EQ");
    v.setProperty("totalGain", std::numeric_limits<float>::quiet_NaN(), nullptr);
    eq.setState(v);

    const float after = eq.getTotalGain();
    if (!(after > 2.9f && after < 3.1f))
    {
        std::fprintf(stderr, "D11-T1: NaN totalGain stored (%f), baseline was %f\n", after, baseline);
        return false;
    }
    std::printf("STG11D11TotalGainFiniteTests: D11-T1 PASS (NaN totalGain rejected)\n");
    return true;
}

bool checkD11T2ValidRoundTrip()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-11-D11 T-2] harness start failed\n");
        return false;
    }
    h.stop();
    EQProcessor& eq = eqOf(h.engine());

    eq.setTotalGain(6.0f);
    juce::ValueTree saved("EQ");
    saved.setProperty("totalGain", eq.getTotalGain(), nullptr);
    eq.setTotalGain(-3.0f);
    eq.setState(saved);

    const float restored = eq.getTotalGain();
    if (!(restored > 5.9f && restored < 6.1f))
    {
        std::fprintf(stderr, "D11-T2: valid round-trip broken (%f)\n", restored);
        return false;
    }
    std::printf("STG11D11TotalGainFiniteTests: D11-T2 PASS (valid round-trip kept)\n");
    return true;
}

bool checkD11T3InfRejected()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-11-D11 T-3] harness start failed\n");
        return false;
    }
    h.stop();
    EQProcessor& eq = eqOf(h.engine());

    eq.setTotalGain(3.0f);

    juce::ValueTree vPos("EQ");
    vPos.setProperty("totalGain", std::numeric_limits<float>::infinity(), nullptr);
    eq.setState(vPos);
    if (!(eq.getTotalGain() > 2.9f && eq.getTotalGain() < 3.1f))
    {
        std::fprintf(stderr, "D11-T3: +Inf totalGain stored (%f)\n", eq.getTotalGain());
        return false;
    }

    juce::ValueTree vNeg("EQ");
    vNeg.setProperty("totalGain", -std::numeric_limits<float>::infinity(), nullptr);
    eq.setState(vNeg);
    if (!(eq.getTotalGain() > 2.9f && eq.getTotalGain() < 3.1f))
    {
        std::fprintf(stderr, "D11-T3: -Inf totalGain stored (%f)\n", eq.getTotalGain());
        return false;
    }
    std::printf("STG11D11TotalGainFiniteTests: D11-T3 PASS (Inf rejected)\n");
    return true;
}

} // namespace

int runSTG11D11TotalGainFiniteTests()
{
    bool ok = true;
    if (!checkD11T1NaNRejected())
    {
        std::fprintf(stderr, "FAIL: D11-T1 NaN rejected\n");
        ok = false;
    }
    if (!checkD11T2ValidRoundTrip())
    {
        std::fprintf(stderr, "FAIL: D11-T2 valid round-trip\n");
        ok = false;
    }
    if (!checkD11T3InfRejected())
    {
        std::fprintf(stderr, "FAIL: D11-T3 Inf rejected\n");
        ok = false;
    }
    if (ok)
        std::printf("STG11D11TotalGainFiniteTests: PASS (D11-T1/D11-T2/D11-T3)\n");
    return ok ? 0 : 1;
}
