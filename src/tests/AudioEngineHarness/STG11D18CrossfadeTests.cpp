// STG11D18CrossfadeTests.cpp - STG-11-D18-1 regression (T-CFX-A .. T-CFX-D).
//
// Target defect (STG-11-D18-1):
//   DSPTransition::onPublishCompleted registered a crossfade record, began the
//   crossfade, and restarted the ramp even when the fading-slot claim failed
//   and the old DSP was retired immediately. The phantom record claimed a fade
//   the audio never performs; its later completion resurrected the retired DSP
//   (toHandle -> Active) and overwrote activeHandle with a stale value, so the
//   next publish used a retired handle as oldHandle (UAF risk via
//   fading-resurrection of a reclaimed object).
//
//   Fix (NonRT only, 3 points):
//     F-A. DSPHandleRuntime::activate no longer clears the fading handle, so an
//          overlapping publish does not destroy the in-flight fade's retire path.
//     F-B. onPublishCompleted claims FIRST; on claim failure it retires oldDSP
//          and skips register/begin (no phantom record). ramp.start stays
//          unconditional (audible behavior preserved).
//     F-C. endCrossfade guards stale writes: toHandle -> Active only from
//          CrossfadingIn; activeHandle only from fromHandle/toHandle; fading
//          null only from fromHandle/null.
//
// Test contract (all deterministic, no timing, no real DSP objects):
//   T-CFX-A  Overlap completion: begin(h1,h2) -> retire(h2) -> activate(h3) ->
//            endCrossfade(id1) keeps h2 Retired, active==h3, h1 Retired.
//   T-CFX-B  activate() preserves a fading handle set by beginCrossfade.
//   T-CFX-C  Normal completion unchanged: begin(h1,h2) -> endCrossfade ->
//            h1 Retired, h2 Active, active==h2, fading null.
//   T-CFX-D  Structural: onPublishCompleted claims before registering
//            (source order), and endCrossfade contains the staleness guards.
//   Negative control: temporarily reverting each fix makes its test FAIL.
//
// Fixture: standalone DSPHandleRuntime + CrossfadeAuthorityRuntime instances
// (default-constructible, instance-local state). Dummy distinct pointers via
// create(void*) — never dereferenced by the exercised paths. No engine needed.
// New CTest target: none (AudioEngineHarness sub-test).
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <ios>
#include <sstream>
#include <string>

#include "audioengine/ISRDSPHandle.h"

namespace {

using convo::isr::CrossfadeAuthorityRuntime;
using convo::isr::CrossfadeId;
using convo::isr::DSPHandle;
using convo::isr::DSPHandleRuntime;

// Distinct dummy identities. create(void*) stores the pointer without
// dereferencing it on any path exercised here.
static int s_dummyA = 1;
static int s_dummyB = 2;
static int s_dummyC = 3;

bool checkD18AOverlapCompletion()
{
    DSPHandleRuntime rt;
    CrossfadeAuthorityRuntime auth;

    const DSPHandle h1 = rt.create(&s_dummyA);
    const DSPHandle h2 = rt.create(&s_dummyB);
    const DSPHandle h3 = rt.create(&s_dummyC);
    if (h1.isNull() || h2.isNull() || h3.isNull())
    {
        std::fprintf(stderr, "D18-A: handle creation failed\n");
        return false;
    }

    // In-flight fade R1 -> R2 (mirrors onPublishCompleted claimed path).
    const CrossfadeId id1 = auth.registerCrossfade(h1, h2);
    rt.beginCrossfade(h1, h2, id1);

    // Overlapping R3 publish: claim fails -> R2 retired immediately,
    // fading handle preserved (F-A), no phantom record (F-B).
    rt.retire(h2);
    rt.activate(h3);

    // Completion of the in-flight fade must not resurrect R2 (F-C).
    rt.endCrossfade(id1);

    bool ok = true;
    if (!rt.isRetired(h1))
    {
        std::fprintf(stderr, "D18-A: h1 not Retired\n");
        ok = false;
    }
    if (!rt.isRetired(h2))
    {
        std::fprintf(stderr, "D18-A: h2 resurrected (expected Retired)\n");
        ok = false;
    }
    if (!(rt.getActiveRuntimeDSPHandle() == h3))
    {
        std::fprintf(stderr, "D18-A: active handle clobbered (expected h3)\n");
        ok = false;
    }
    if (!(rt.resolve(h3).valid))
    {
        std::fprintf(stderr, "D18-A: h3 not Active\n");
        ok = false;
    }
    if (ok)
        std::printf("STG11D18CrossfadeTests: D18-A PASS (no resurrection, no stale active)\n");
    return ok;
}

bool checkD18BActivatePreservesFadingHandle()
{
    DSPHandleRuntime rt;
    CrossfadeAuthorityRuntime auth;

    const DSPHandle h1 = rt.create(&s_dummyA);
    const DSPHandle h2 = rt.create(&s_dummyB);
    const DSPHandle h3 = rt.create(&s_dummyC);
    if (h1.isNull() || h2.isNull() || h3.isNull())
    {
        std::fprintf(stderr, "D18-B: handle creation failed\n");
        return false;
    }

    const CrossfadeId id1 = auth.registerCrossfade(h1, h2);
    (void)id1;
    rt.beginCrossfade(h1, h2, 7u);
    if (!(rt.getFadingRuntimeDSPHandle() == h1))
    {
        std::fprintf(stderr, "D18-B: fading handle not set by beginCrossfade\n");
        return false;
    }

    // A newer publish activates h3. The in-flight fading handle must survive (F-A).
    rt.activate(h3);
    if (!(rt.getFadingRuntimeDSPHandle() == h1))
    {
        std::fprintf(stderr, "D18-B: activate() destroyed the in-flight fading handle\n");
        return false;
    }
    if (!(rt.getActiveRuntimeDSPHandle() == h3))
    {
        std::fprintf(stderr, "D18-B: activate() did not publish h3\n");
        return false;
    }
    std::printf("STG11D18CrossfadeTests: D18-B PASS (fading handle preserved)\n");
    return true;
}

bool checkD18CNormalCompletionUnchanged()
{
    DSPHandleRuntime rt;
    CrossfadeAuthorityRuntime auth;

    const DSPHandle h1 = rt.create(&s_dummyA);
    const DSPHandle h2 = rt.create(&s_dummyB);
    if (h1.isNull() || h2.isNull())
    {
        std::fprintf(stderr, "D18-C: handle creation failed\n");
        return false;
    }

    const CrossfadeId id1 = auth.registerCrossfade(h1, h2);
    rt.beginCrossfade(h1, h2, id1);
    // Mirrors production order (DSPTransition step 1 activates before completion).
    rt.activate(h2);
    rt.endCrossfade(id1);

    bool ok = true;
    if (!rt.isRetired(h1))
    {
        std::fprintf(stderr, "D18-C: h1 not Retired\n");
        ok = false;
    }
    if (rt.isRetired(h2))
    {
        std::fprintf(stderr, "D18-C: h2 not Active\n");
        ok = false;
    }
    if (!(rt.getActiveRuntimeDSPHandle() == h2))
    {
        std::fprintf(stderr, "D18-C: active handle not h2\n");
        ok = false;
    }
    if (!rt.getFadingRuntimeDSPHandle().isNull())
    {
        std::fprintf(stderr, "D18-C: fading handle not cleared\n");
        ok = false;
    }
    if (ok)
        std::printf("STG11D18CrossfadeTests: D18-C PASS (normal completion unchanged)\n");
    return ok;
}

// -- structural source check (D8_1 precedent): the inspected tree is explicit.
//    CONVOPEQ_SOURCE_ROOT must name the worktree to inspect (CMake passes
//    CMAKE_CURRENT_SOURCE_DIR). There is deliberately no absolute-path fallback:
//    a missing root fails the test instead of silently inspecting another
//    working tree (V-08). ---
const char* productionSourceRoot()
{
    const char* root = std::getenv("CONVOPEQ_SOURCE_ROOT");
    return (root != nullptr && root[0] != '\0') ? root : nullptr;
}

bool readProductionSource(const char* relativePath, std::string& out)
{
    const char* root = productionSourceRoot();
    if (root == nullptr)
    {
        std::fprintf(stderr, "D18-D: CONVOPEQ_SOURCE_ROOT is not set\n");
        return false;
    }
    const std::string path = std::string(root) + "/" + relativePath;
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
    {
        std::fprintf(stderr, "D18-D: cannot open %s\n", path.c_str());
        return false;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return !out.empty();
}

bool checkD18DStructuralOrdering()
{
    const char* root = productionSourceRoot();
    std::printf("STG11D18CrossfadeTests: source root = %s\n", root != nullptr ? root : "(unset)");
    std::string tr;
    if (!readProductionSource("src/audioengine/DSPTransition.h", tr))
    {
        std::fprintf(stderr, "D18-D: cannot open DSPTransition.h\n");
        return false;
    }
    // Claim must precede registerCrossfade textually (claim-first repair F-B).
    const std::size_t claimPos = tr.find("claimFadingRuntimeDSP(oldDSP)");
    const std::size_t regPos = tr.find("registerCrossfade(oldHandle, newHandle)");
    const std::size_t beginPos = tr.find("beginCrossfade(oldHandle, newHandle");
    if (claimPos == std::string::npos || regPos == std::string::npos
        || beginPos == std::string::npos)
    {
        std::fprintf(stderr, "D18-D: claim/register/begin call sites missing\n");
        return false;
    }
    if (!(claimPos < regPos && claimPos < beginPos))
    {
        std::fprintf(stderr, "D18-D: claim does not precede register/begin\n");
        return false;
    }

    std::string isr;
    if (!readProductionSource("src/audioengine/ISRDSPHandle.cpp", isr))
    {
        std::fprintf(stderr, "D18-D: cannot open ISRDSPHandle.cpp\n");
        return false;
    }
    // Staleness guards must exist in endCrossfade (F-C): the toHandle guard
    // referencing CrossfadingIn, the active-handle fromHandle/toHandle rule,
    // and the fading-handle fromHandle rule.
    const std::size_t ecPos = isr.find("void DSPHandleRuntime::endCrossfade(CrossfadeId id)");
    if (ecPos == std::string::npos)
    {
        std::fprintf(stderr, "D18-D: endCrossfade body missing\n");
        return false;
    }
    const std::string ecBody = isr.substr(ecPos, 4000);
    if (ecBody.find("CrossfadingIn") == std::string::npos
        || ecBody.find("fromHandle") == std::string::npos
        || ecBody.find("fadingRuntimeDSPHandle_") == std::string::npos
        || ecBody.find("STG-11-D18") == std::string::npos)
    {
        std::fprintf(stderr, "D18-D: endCrossfade staleness guards missing\n");
        return false;
    }
    const std::size_t actPos = isr.find("void DSPHandleRuntime::activate(DSPHandle handle)");
    if (actPos == std::string::npos)
    {
        std::fprintf(stderr, "D18-D: activate body missing\n");
        return false;
    }
    const std::string actBody = isr.substr(actPos, 1200);
    if (actBody.find("fadingRuntimeDSPHandle_") != std::string::npos)
    {
        std::fprintf(stderr, "D18-D: activate still touches the fading handle\n");
        return false;
    }
    std::printf("STG11D18CrossfadeTests: D18-D PASS (claim-first + guards structural)\n");
    return true;
}

} // namespace

int runSTG11D18CrossfadeTests()
{
    bool ok = true;
    if (!checkD18AOverlapCompletion())
    {
        std::fprintf(stderr, "FAIL: D18-A overlap completion\n");
        ok = false;
    }
    if (!checkD18BActivatePreservesFadingHandle())
    {
        std::fprintf(stderr, "FAIL: D18-B activate preserves fading handle\n");
        ok = false;
    }
    if (!checkD18CNormalCompletionUnchanged())
    {
        std::fprintf(stderr, "FAIL: D18-C normal completion unchanged\n");
        ok = false;
    }
    if (!checkD18DStructuralOrdering())
    {
        std::fprintf(stderr, "FAIL: D18-D structural ordering\n");
        ok = false;
    }
    if (ok)
        std::printf("STG11D18CrossfadeTests: PASS (D18-A/D18-B/D18-C/D18-D)\n");
    return ok ? 0 : 1;
}
