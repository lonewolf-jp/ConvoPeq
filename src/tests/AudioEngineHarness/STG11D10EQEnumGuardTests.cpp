// STG11D10EQEnumGuardTests.cpp - STG-11-D10-1 regression (T1 .. T3).
//
// Target defect (STG-11-D10-1):
//   EQProcessor::setBandType / setBandChannelMode stored session ints as
//   enums without range validation. Out-of-range values (corrupted / tampered /
//   future-version sessions) became runtime state: an OOB band type falls
//   through calcSVFCoeffs' switch to zero coefficients (band silenced), and an
//   OOB channel mode matches none of the processing equality chains (band
//   unprocessed). Same defect class as STG-11-D7 (unvalidated session enums).
//   RT NaN guards contain propagation (no crash), but the corrupted load
//   silently disables the band.
//
//   Fix: range guards at both setters (D7 pattern). Out-of-range values keep
//   the current value.
//
// Test contract:
//   D10-T1 Out-of-range type/channel via setState -> values unchanged.
//   D10-T2 Valid round-trip (all 5 types x all 5 channels).
//   D10-T3 Boundaries (min/max applied, min-1/max+1 rejected).
//   Negative control: with the guards removed, D10-T1 FAILs.
//
// No new CTest registration: sub-test of AudioEngineHarness.
// Fixture owned by the harness.
// =============================================================================

#include <cstdio>

#include "audioengine/AudioEngine.h"
#include "AudioEngineHarness.h"
#include "eqprocessor/EQProcessor.h"

namespace {

static EQProcessor& eqOf(AudioEngine& e) { return e.getEQProcessor(); }

bool checkD10T1OutOfRangeRejected()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-11-D10 T-1] harness start failed\n");
        return false;
    }
    h.stop();
    EQProcessor& eq = eqOf(h.engine());

    eq.setBandType(0, EQBandType::Peaking);
    eq.setBandChannelMode(0, EQChannelMode::Stereo);

    juce::ValueTree v("EQ");
    juce::ValueTree band("Band");
    band.setProperty("index", 0, nullptr);
    band.setProperty("type", 99, nullptr);
    band.setProperty("channel", -7, nullptr);
    v.addChild(band, -1, nullptr);
    eq.setState(v);

    bool ok = true;
    if (eq.getBandType(0) != EQBandType::Peaking)
    {
        std::fprintf(stderr, "D10-T1: OOB band type applied (%d)\n",
                     static_cast<int>(eq.getBandType(0)));
        ok = false;
    }
    if (eq.getBandChannelMode(0) != EQChannelMode::Stereo)
    {
        std::fprintf(stderr, "D10-T1: OOB channel mode applied (%d)\n",
                     static_cast<int>(eq.getBandChannelMode(0)));
        ok = false;
    }
    if (ok)
        std::printf("STG11D10EQEnumGuardTests: D10-T1 PASS (OOB enums rejected)\n");
    return ok;
}

bool checkD10T2ValidRoundTrip()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-11-D10 T-2] harness start failed\n");
        return false;
    }
    h.stop();
    EQProcessor& eq = eqOf(h.engine());

    static const EQBandType kTypes[] = {
        EQBandType::LowShelf, EQBandType::Peaking,
        EQBandType::HighShelf, EQBandType::LowPass,
        EQBandType::HighPass
    };
    static const EQChannelMode kChannels[] = {
        EQChannelMode::Stereo, EQChannelMode::Left,
        EQChannelMode::Right, EQChannelMode::Mid,
        EQChannelMode::Side
    };
    bool ok = true;
    for (int ti = 0; ti < 5 && ok; ++ti)
    {
        for (int ci = 0; ci < 5 && ok; ++ci)
        {
            juce::ValueTree v("EQ");
            juce::ValueTree band("Band");
            band.setProperty("index", 0, nullptr);
            band.setProperty("type", static_cast<int>(kTypes[ti]), nullptr);
            band.setProperty("channel", static_cast<int>(kChannels[ci]), nullptr);
            v.addChild(band, -1, nullptr);
            eq.setState(v);
            if (eq.getBandType(0) != kTypes[ti]
                || eq.getBandChannelMode(0) != kChannels[ci])
            {
                std::fprintf(stderr, "D10-T2: valid type/channel not restored (t=%d c=%d)\n", ti, ci);
                ok = false;
            }
        }
    }
    if (ok)
        std::printf("STG11D10EQEnumGuardTests: D10-T2 PASS (valid round-trip kept)\n");
    return ok;
}

bool checkD10T3Boundaries()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-11-D10 T-3] harness start failed\n");
        return false;
    }
    h.stop();
    EQProcessor& eq = eqOf(h.engine());

    eq.setBandType(0, EQBandType::Peaking);
    eq.setBandChannelMode(0, EQChannelMode::Stereo);

    // min-1 / max+1 rejected (defaults retained).
    {
        juce::ValueTree v("EQ");
        juce::ValueTree band("Band");
        band.setProperty("index", 0, nullptr);
        band.setProperty("type", -1, nullptr);
        band.setProperty("channel", 5, nullptr);
        v.addChild(band, -1, nullptr);
        eq.setState(v);
    }
    bool ok = true;
    if (eq.getBandType(0) != EQBandType::Peaking
        || eq.getBandChannelMode(0) != EQChannelMode::Stereo)
    {
        std::fprintf(stderr, "D10-T3: boundary-adjacent OOB applied\n");
        ok = false;
    }

    // min / max accepted.
    {
        juce::ValueTree v("EQ");
        juce::ValueTree band("Band");
        band.setProperty("index", 0, nullptr);
        band.setProperty("type", 0, nullptr);    // LowShelf
        band.setProperty("channel", 4, nullptr); // Side
        v.addChild(band, -1, nullptr);
        eq.setState(v);
    }
    if (eq.getBandType(0) != EQBandType::LowShelf
        || eq.getBandChannelMode(0) != EQChannelMode::Side)
    {
        std::fprintf(stderr, "D10-T3: boundary min/max rejected\n");
        ok = false;
    }

    if (ok)
        std::printf("STG11D10EQEnumGuardTests: D10-T3 PASS (boundaries exact)\n");
    return ok;
}

} // namespace

int runSTG11D10EQEnumGuardTests()
{
    bool ok = true;
    if (!checkD10T1OutOfRangeRejected())
    {
        std::fprintf(stderr, "FAIL: D10-T1 OOB rejected\n");
        ok = false;
    }
    if (!checkD10T2ValidRoundTrip())
    {
        std::fprintf(stderr, "FAIL: D10-T2 valid round-trip\n");
        ok = false;
    }
    if (!checkD10T3Boundaries())
    {
        std::fprintf(stderr, "FAIL: D10-T3 boundaries\n");
        ok = false;
    }
    if (ok)
        std::printf("STG11D10EQEnumGuardTests: PASS (D10-T1/D10-T2/D10-T3)\n");
    return ok ? 0 : 1;
}
