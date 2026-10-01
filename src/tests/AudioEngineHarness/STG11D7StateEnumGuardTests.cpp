// STG11D7StateEnumGuardTests.cpp - STG-11-D7-1 regression (T1 .. T3).
//
// Target defect (STG-11-D7-1):
//   AudioEngine::requestLoadState restored convHCFilterMode / convLCFilterMode /
//   eqLPFFilterMode (+ processingOrder / analyzerSource) through unchecked casts.
//   Out-of-range values (corrupted / tampered / future-version sessions) were
//   published straight into atomics. The three filter modes are used as raw
//   array indices by OutputFilter::process on the RT audio thread
//   (hcCoeff[3][2] / lcCoeff[2] / lpCoeff[3][2], no bounds check), so an
//   out-of-range value causes an OOB read (silent wrong coefficients, worst
//   case AV crash). Same defect class as work92 B-3 (big 2-10), which guarded
//   noiseShaperType / oversamplingType but left these sites open. The range
//   validator (validatePresetStateTreeForDebug) exists but has zero callers.
//
//   Fix: range guards at the six cast sites (B-3 pattern). Out-of-range values
//   keep the current runtime value (default).
//
// Test contract:
//   D7-T1  Corrupted session (OOB enum values) -> modes stay at defaults.
//   D7-T2  Valid round-trip unchanged (getCurrentState -> requestLoadState).
//   D7-T3  Boundary values (min/max applied, min-1/max+1 rejected).
//   Negative control: with the guards removed, D7-T1 FAILs.
//
// No new CTest registration: sub-test of AudioEngineHarness, STG-8/9/D3-D5 shape.
// The fixture is owned by the harness (AudioEngine ~4MB must not live on TU stack).
// =============================================================================

#include <cstdio>

#include "audioengine/AudioEngine.h"
#include "AudioEngineHarness.h"

namespace {

bool checkD7T1CorruptedSessionRejected()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-11-D7 T-1] harness start failed\n");
        return false;
    }
    h.stop();
    AudioEngine& e = h.engine();

    // Baseline: a valid saved state restores cleanly first.
    const juce::ValueTree saved = e.getCurrentState();
    if (!saved.isValid())
    {
        std::fprintf(stderr, "D7-T1: getCurrentState invalid\n");
        return false;
    }

    // Corrupt the six enum properties with out-of-range values.
    juce::ValueTree corrupted = saved.createCopy();
    corrupted.setProperty("processingOrder", 7, nullptr);
    corrupted.setProperty("analyzerSource", -3, nullptr);
    corrupted.setProperty("convHCFilterMode", 5, nullptr);
    corrupted.setProperty("convLCFilterMode", 9, nullptr);
    corrupted.setProperty("eqLPFFilterMode", -1, nullptr);
    e.requestLoadState(corrupted);

    // All six must retain their pre-load values (defaults on a fresh engine).
    bool ok = true;
    const auto order = convo::consumeAtomic(e.currentProcessingOrder, std::memory_order_acquire);
    if (order != convo::ProcessingOrder::ConvolverThenEQ)
    {
        std::fprintf(stderr, "D7-T1: processingOrder changed to %d (expected 0)\n",
                     static_cast<int>(order));
        ok = false;
    }
    if (e.getAnalyzerSource() != AudioEngine::AnalyzerSource::Output)
    {
        std::fprintf(stderr, "D7-T1: analyzerSource changed (expected Output)\n");
        ok = false;
    }
    if (e.getConvHCFilterMode() != convo::HCMode::Natural)
    {
        std::fprintf(stderr, "D7-T1: convHCFilterMode changed to %d (expected 1)\n",
                     static_cast<int>(e.getConvHCFilterMode()));
        ok = false;
    }
    if (e.getConvLCFilterMode() != convo::LCMode::Natural)
    {
        std::fprintf(stderr, "D7-T1: convLCFilterMode changed to %d (expected 0)\n",
                     static_cast<int>(e.getConvLCFilterMode()));
        ok = false;
    }
    if (e.getEqLPFFilterMode() != convo::HCMode::Natural)
    {
        std::fprintf(stderr, "D7-T1: eqLPFFilterMode changed to %d (expected 1)\n",
                     static_cast<int>(e.getEqLPFFilterMode()));
        ok = false;
    }
    if (ok)
        std::printf("STG11D7StateEnumGuardTests: D7-T1 PASS (OOB session values rejected)\n");
    return ok;
}

bool checkD7T2ValidRoundTrip()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-11-D7 T-2] harness start failed\n");
        return false;
    }
    h.stop();
    AudioEngine& e = h.engine();

    // Set non-default but valid values through the public setters.
    e.setConvHCFilterMode(convo::HCMode::Soft);
    e.setConvLCFilterMode(convo::LCMode::Soft);
    e.setEqLPFFilterMode(convo::HCMode::Sharp);

    // Round-trip through serialization.
    const juce::ValueTree saved = e.getCurrentState();
    e.setConvHCFilterMode(convo::HCMode::Natural);
    e.setConvLCFilterMode(convo::LCMode::Natural);
    e.setEqLPFFilterMode(convo::HCMode::Natural);
    e.requestLoadState(saved);

    bool ok = true;
    if (e.getConvHCFilterMode() != convo::HCMode::Soft)
    {
        std::fprintf(stderr, "D7-T2: convHCFilterMode not restored\n");
        ok = false;
    }
    if (e.getConvLCFilterMode() != convo::LCMode::Soft)
    {
        std::fprintf(stderr, "D7-T2: convLCFilterMode not restored\n");
        ok = false;
    }
    if (e.getEqLPFFilterMode() != convo::HCMode::Sharp)
    {
        std::fprintf(stderr, "D7-T2: eqLPFFilterMode not restored\n");
        ok = false;
    }
    if (ok)
        std::printf("STG11D7StateEnumGuardTests: D7-T2 PASS (valid round-trip kept)\n");
    return ok;
}

bool checkD7T3Boundaries()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-11-D7 T-3] harness start failed\n");
        return false;
    }
    h.stop();
    AudioEngine& e = h.engine();

    const juce::ValueTree saved = e.getCurrentState();
    bool ok = true;

    // min-1 / max+1 rejected for each mode (defaults retained).
    struct Case { const char* key; int bad; };
    static const Case kBad[] = {
        { "convHCFilterMode", -1 }, { "convHCFilterMode", 3 },
        { "convLCFilterMode", -1 }, { "convLCFilterMode", 2 },
        { "eqLPFFilterMode", -1 }, { "eqLPFFilterMode", 3 },
        { "processingOrder", -1 }, { "processingOrder", 2 },
        { "analyzerSource", -1 }, { "analyzerSource", 2 },
    };
    for (const Case& c : kBad)
    {
        juce::ValueTree v = saved.createCopy();
        v.setProperty(c.key, c.bad, nullptr);
        e.requestLoadState(v);
    }
    if (e.getConvHCFilterMode() != convo::HCMode::Natural
        || e.getConvLCFilterMode() != convo::LCMode::Natural
        || e.getEqLPFFilterMode() != convo::HCMode::Natural
        || convo::consumeAtomic(e.currentProcessingOrder, std::memory_order_acquire)
               != convo::ProcessingOrder::ConvolverThenEQ
        || e.getAnalyzerSource() != AudioEngine::AnalyzerSource::Output)
    {
        std::fprintf(stderr, "D7-T3: boundary-adjacent OOB value applied\n");
        ok = false;
    }

    // min / max accepted.
    {
        juce::ValueTree v = saved.createCopy();
        v.setProperty("convHCFilterMode", 0, nullptr);   // Sharp
        v.setProperty("convLCFilterMode", 1, nullptr);   // Soft
        v.setProperty("eqLPFFilterMode", 2, nullptr);    // Soft
        v.setProperty("processingOrder", 1, nullptr);    // EQThenConvolver
        v.setProperty("analyzerSource", 0, nullptr);     // Input
        e.requestLoadState(v);
    }
    if (e.getConvHCFilterMode() != convo::HCMode::Sharp
        || e.getConvLCFilterMode() != convo::LCMode::Soft
        || e.getEqLPFFilterMode() != convo::HCMode::Soft
        || convo::consumeAtomic(e.currentProcessingOrder, std::memory_order_acquire)
               != convo::ProcessingOrder::EQThenConvolver
        || e.getAnalyzerSource() != AudioEngine::AnalyzerSource::Input)
    {
        std::fprintf(stderr, "D7-T3: boundary min/max value rejected\n");
        ok = false;
    }

    if (ok)
        std::printf("STG11D7StateEnumGuardTests: D7-T3 PASS (boundaries exact)\n");
    return ok;
}

} // namespace

int runSTG11D7StateEnumGuardTests()
{
    bool ok = true;
    if (!checkD7T1CorruptedSessionRejected())
    {
        std::fprintf(stderr, "FAIL: D7-T1 corrupted session rejected\n");
        ok = false;
    }
    if (!checkD7T2ValidRoundTrip())
    {
        std::fprintf(stderr, "FAIL: D7-T2 valid round-trip\n");
        ok = false;
    }
    if (!checkD7T3Boundaries())
    {
        std::fprintf(stderr, "FAIL: D7-T3 boundaries\n");
        ok = false;
    }
    if (ok)
        std::printf("STG11D7StateEnumGuardTests: PASS (D7-T1/D7-T2/D7-T3)\n");
    return ok ? 0 : 1;
}
