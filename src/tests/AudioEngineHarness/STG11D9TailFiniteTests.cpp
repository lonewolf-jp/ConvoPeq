// STG11D9TailFiniteTests.cpp - STG-11-D9-1 regression (T1 .. T3).
//
// Target defect (STG-11-D9-1):
//   ConvolverProcessor::setTailStrength / setTailStartSec stored the
//   juce::jlimit result unconditionally. jlimit passes NaN through, so a
//   corrupted session's NaN was stored as-is and flowed into the MKL tail
//   computation (whose jlimits also pass NaN): NaN layer gains -> NaN RT
//   audio output, NaN tailStartSec -> distorted layer geometry + NaN damping.
//
//   Fix: reject non-finite inputs at both setters (existing
//   convo::numeric_policy::isFinite), keeping the previous finite value.
//
// Test contract:
//   D9-T1  NaN tailStrength / tailStartSec via setState -> finite retained.
//   D9-T2  Valid round-trip kept.
//   D9-T3  +Inf / -Inf rejected.
//   Negative control: with the guards removed, D9-T1 FAILs.
//
// No new CTest registration: sub-test of AudioEngineHarness.
// Fixture owned by the harness.
// =============================================================================

#include <cstdio>
#include <limits>

#include "audioengine/AudioEngine.h"
#include "AudioEngineHarness.h"
#include "ConvolverProcessor.h"

namespace {

bool checkD9T1NaNRejected()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-11-D9 T-1] harness start failed\n");
        return false;
    }
    h.stop();
    ConvolverProcessor& conv = h.engine().getConvolverProcessor();

    conv.setTailStrength(1.0f);
    conv.setTailStartSec(0.2f);

    juce::ValueTree v("Convolver");
    v.setProperty("tailStrength", std::numeric_limits<float>::quiet_NaN(), nullptr);
    v.setProperty("tailStartSec", std::numeric_limits<float>::quiet_NaN(), nullptr);
    conv.setState(v);

    bool ok = true;
    const float s = conv.getTailStrength();
    if (!(s > 0.0f && s < 10.0f))
    {
        std::fprintf(stderr, "D9-T1: NaN tailStrength stored (%f)\n", s);
        ok = false;
    }
    const float t = conv.getTailStartSec();
    if (!(t > 0.0f && t < 10.0f))
    {
        std::fprintf(stderr, "D9-T1: NaN tailStartSec stored (%f)\n", t);
        ok = false;
    }
    if (ok)
        std::printf("STG11D9TailFiniteTests: D9-T1 PASS (NaN tail values rejected)\n");
    return ok;
}

bool checkD9T2ValidRoundTrip()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-11-D9 T-2] harness start failed\n");
        return false;
    }
    h.stop();
    ConvolverProcessor& conv = h.engine().getConvolverProcessor();

    conv.setTailStrength(1.25f);
    conv.setTailStartSec(0.3f);
    juce::ValueTree saved("Convolver");
    saved.setProperty("tailStrength", conv.getTailStrength(), nullptr);
    saved.setProperty("tailStartSec", conv.getTailStartSec(), nullptr);
    conv.setTailStrength(0.5f);
    conv.setTailStartSec(0.1f);
    conv.setState(saved);

    bool ok = true;
    if (!(conv.getTailStrength() > 1.24f && conv.getTailStrength() < 1.26f))
    {
        std::fprintf(stderr, "D9-T2: tailStrength round-trip broken (%f)\n", conv.getTailStrength());
        ok = false;
    }
    if (!(conv.getTailStartSec() > 0.29f && conv.getTailStartSec() < 0.31f))
    {
        std::fprintf(stderr, "D9-T2: tailStartSec round-trip broken (%f)\n", conv.getTailStartSec());
        ok = false;
    }
    if (ok)
        std::printf("STG11D9TailFiniteTests: D9-T2 PASS (valid round-trip kept)\n");
    return ok;
}

bool checkD9T3InfRejected()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-11-D9 T-3] harness start failed\n");
        return false;
    }
    h.stop();
    ConvolverProcessor& conv = h.engine().getConvolverProcessor();

    conv.setTailStrength(1.0f);
    conv.setTailStartSec(0.2f);

    juce::ValueTree v("Convolver");
    v.setProperty("tailStrength", std::numeric_limits<float>::infinity(), nullptr);
    v.setProperty("tailStartSec", -std::numeric_limits<float>::infinity(), nullptr);
    conv.setState(v);

    bool ok = true;
    if (!(conv.getTailStrength() > 0.0f && conv.getTailStrength() < 10.0f))
    {
        std::fprintf(stderr, "D9-T3: +Inf tailStrength stored (%f)\n", conv.getTailStrength());
        ok = false;
    }
    if (!(conv.getTailStartSec() > 0.0f && conv.getTailStartSec() < 10.0f))
    {
        std::fprintf(stderr, "D9-T3: -Inf tailStartSec stored (%f)\n", conv.getTailStartSec());
        ok = false;
    }
    if (ok)
        std::printf("STG11D9TailFiniteTests: D9-T3 PASS (Inf rejected)\n");
    return ok;
}

} // namespace

int runSTG11D9TailFiniteTests()
{
    bool ok = true;
    if (!checkD9T1NaNRejected())
    {
        std::fprintf(stderr, "FAIL: D9-T1 NaN rejected\n");
        ok = false;
    }
    if (!checkD9T2ValidRoundTrip())
    {
        std::fprintf(stderr, "FAIL: D9-T2 valid round-trip\n");
        ok = false;
    }
    if (!checkD9T3InfRejected())
    {
        std::fprintf(stderr, "FAIL: D9-T3 Inf rejected\n");
        ok = false;
    }
    if (ok)
        std::printf("STG11D9TailFiniteTests: PASS (D9-T1/D9-T2/D9-T3)\n");
    return ok ? 0 : 1;
}
