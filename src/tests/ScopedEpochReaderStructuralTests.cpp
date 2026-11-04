// =============================================================================
// ScopedEpochReaderStructuralTests.cpp — R-22: structural enforcement for the
// A-closure reader-safety contract.
//
// Why this test exists:
//   PR #59 fixed a use-after-free class by making every non-RT reader of
//   EQProcessor's published EQState hold a convo::ScopedEpochReader across the
//   load + dereference. Nothing prevents a future reader from being added
//   without the guard, which would silently reintroduce the defect.
//
//   This test reads the production sources and asserts the contract:
//     1. every listed non-RT reader body acquires the guard (either an explicit
//        "ScopedEpochReader" or "acquireReadGuard()") and that acquisition appears
//        BEFORE the first "loadCurrentState" in that body;
//     2. getEQState() stays the only raw-pointer accessor (no guard inside);
//     3. the two known raw-pointer call sites hold "acquireReadGuard".
//
//   The list of functions is the contract: adding a new reader means adding it
//   here, which forces the author to look at the guard requirement.
//
// Oracle: source text only. No production code, no test hooks, no timing.
// =============================================================================

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef CONVOPEQ_REPO_ROOT
#error "CONVOPEQ_REPO_ROOT must be defined by the build (repository root path)"
#endif

namespace
{

std::string readFile (const std::string& relativePath)
{
    const std::string path = std::string (CONVOPEQ_REPO_ROOT) + "/" + relativePath;
    std::ifstream in (path, std::ios::binary);

    if (! in.is_open())
        return {};

    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Extracts the body of "EQProcessor::<name>(" by brace matching. Returns an
// empty string when the function is not found.
std::string functionBody (const std::string& source, const std::string& name)
{
    const std::string signature = "EQProcessor::" + name + "(";
    const std::size_t sigPos = source.find (signature);

    if (sigPos == std::string::npos)
        return {};

    const std::size_t openBrace = source.find ('{', sigPos);

    if (openBrace == std::string::npos)
        return {};

    int depth = 0;

    for (std::size_t i = openBrace; i < source.size(); ++i)
    {
        if (source[i] == '{')
            ++depth;
        else if (source[i] == '}')
        {
            --depth;

            if (depth == 0)
                return source.substr (openBrace, i - openBrace + 1);
        }
    }

    return {};
}

struct Check
{
    bool ok = false;
    std::string detail;
};

// R-13: generic brace-matched body lookup for a fully-qualified signature prefix
// (e.g. "RuntimeReadHandle makeRuntimeReadHandle(").
std::string bodyOfMarker (const std::string& source, const std::string& marker)
{
    const std::size_t sigPos = source.find (marker);

    if (sigPos == std::string::npos)
        return {};

    const std::size_t openBrace = source.find ('{', sigPos);

    if (openBrace == std::string::npos)
        return {};

    int depth = 0;

    for (std::size_t i = openBrace; i < source.size(); ++i)
    {
        if (source[i] == '{')
            ++depth;
        else if (source[i] == '}')
        {
            --depth;

            if (depth == 0)
                return source.substr (openBrace, i - openBrace + 1);
        }
    }

    return {};
}

// R-13: the guard must be acquired BEFORE the published-world load in the body.
Check guardBeforeWorld (const std::string& source, const std::string& marker,
                        const std::string& guardNeedle, const std::string& loadNeedle)
{
    const std::string body = bodyOfMarker (source, marker);

    if (body.empty())
        return { false, "function body not found: " + marker };

    const std::size_t guard = body.find (guardNeedle);
    const std::size_t load = body.find (loadNeedle);

    if (guard == std::string::npos)
        return { false, marker + ": missing guard '" + guardNeedle + "'" };

    if (load == std::string::npos)
        return { false, marker + ": missing world load '" + loadNeedle + "' (contract changed?)" };

    if (! (guard < load))
        return { false, marker + ": guard must appear BEFORE '" + loadNeedle + "'" };

    return { true, marker + ": guard precedes the world load" };
}

Check guardBeforeLoad (const std::string& source, const std::string& name)
{
    const std::string body = functionBody (source, name);

    if (body.empty())
        return { false, "function body not found: " + name };

    // The body may acquire the guard directly (an explicit ScopedEpochReader) or
    // through the EQProcessor::acquireReadGuard() accessor; both are the same contract.
    const std::size_t directGuard = body.find ("ScopedEpochReader");
    const std::size_t accessorGuard = body.find ("acquireReadGuard");
    std::size_t guard = std::min (directGuard, accessorGuard);
    const std::size_t load = body.find ("loadCurrentState");

    if (directGuard == std::string::npos && accessorGuard == std::string::npos)
        return { false, name + ": no ScopedEpochReader/acquireReadGuard in the body" };

    if (load == std::string::npos)
        return { false, name + ": no loadCurrentState in the body (contract changed?)" };

    if (! (guard < load))
        return { false, name + ": guard acquisition must appear BEFORE loadCurrentState" };

    return { true, name + ": guard precedes the load" };
}

Check containsText (const std::string& text, const std::string& needle, const std::string& what)
{
    if (text.find (needle) == std::string::npos)
        return { false, what + ": missing '" + needle + "'" };

    return { true, what + ": '" + needle + "' present" };
}

} // namespace

int main()
{
    const std::string core = readFile ("src/eqprocessor/EQProcessor.Core.cpp");
    const std::string coefficients = readFile ("src/eqprocessor/EQProcessor.Coefficients.cpp");
    const std::string eqResponse = readFile ("src/audioengine/AudioEngine.EQResponse.cpp");
    const std::string rebuild = readFile ("src/audioengine/AudioEngine.RebuildDispatch.cpp");
    const std::string engineHeader = readFile ("src/audioengine/AudioEngine.h");
    const std::string orchestrator = readFile ("src/audioengine/RuntimePublicationOrchestrator.cpp");
    const std::string publication = readFile ("src/audioengine/AudioEngine.Publication.cpp");

    if (core.empty() || coefficients.empty() || eqResponse.empty() || rebuild.empty()
        || engineHeader.empty() || orchestrator.empty() || publication.empty())
    {
        std::fprintf (stderr, "FAIL: cannot read one or more production sources\n");
        return 1;
    }

    int failures = 0;

    // 1) Non-RT readers must hold the guard before the load.
    const std::vector<std::pair<const std::string*, std::string>> guardedReaders = {
        { &core, "getState" },
        { &core, "reset" },
        { &core, "prepareToPlay" },
        { &core, "getTotalGain" },
        { &core, "getBandType" },
        { &core, "getBandChannelMode" },
        { &core, "getBandParams" },
        { &coefficients, "updateBandNode" },
    };

    for (const auto& entry : guardedReaders)
    {
        const Check c = guardBeforeLoad (*entry.first, entry.second);

        if (! c.ok)
        {
            std::fprintf (stderr, "FAIL: %s\n", c.detail.c_str());
            ++failures;
        }
        else
        {
            std::printf ("  [PASS] %s\n", c.detail.c_str());
        }
    }

    // 2) getEQState() stays a raw-pointer accessor (no guard inside; the caller owns it).
    {
        const std::string body = functionBody (core, "getEQState");

        if (body.empty())
        {
            std::fprintf (stderr, "FAIL: getEQState body not found\n");
            ++failures;
        }
        else if (body.find ("ScopedEpochReader") != std::string::npos)
        {
            std::fprintf (stderr, "FAIL: getEQState must not take a guard internally (raw pointer contract)\n");
            ++failures;
        }
        else
        {
            std::printf ("  [PASS] getEQState: raw-pointer accessor unchanged\n");
        }
    }

    // 3) Known raw-pointer call sites must hold acquireReadGuard().
    {
        const Check a = containsText (eqResponse, "acquireReadGuard", "AudioEngine.EQResponse.cpp");
        const Check b = containsText (rebuild, "acquireReadGuard", "AudioEngine.RebuildDispatch.cpp");

        for (const Check* c : { &a, &b })
        {
            if (! c->ok)
            {
                std::fprintf (stderr, "FAIL: %s\n", c->detail.c_str());
                ++failures;
            }
            else
            {
                std::printf ("  [PASS] %s\n", c->detail.c_str());
            }
        }
    }

    // 4) R-13: published-world readers must enter a reader scope BEFORE the load.
    //    These three producers run on the Message thread, the RebuildThread and
    //    CoordinatorLoop workers; without a registered epoch the message-thread
    //    reclaimer can free the world inside the load→deref window.
    {
        const Check checks[] = {
            guardBeforeWorld (engineHeader, "RuntimeReadHandle makeRuntimeReadHandle(",
                              "observeCurrentRuntime", "consumeWorldHandle"),
            guardBeforeWorld (orchestrator, "RuntimePublicationOrchestrator::trySubmitImpl(",
                              "RCUReaderGuard submitReadGuard", "observePublishedWorld"),
            guardBeforeWorld (publication, "AudioEngine::makePublishDecisionSnapshot(",
                              "RCUReaderGuard decisionGuard", "observePublishedWorld"),
        };

        for (const auto& c : checks)
        {
            if (! c.ok)
            {
                std::fprintf (stderr, "FAIL: %s\n", c.detail.c_str());
                ++failures;
            }
            else
            {
                std::printf ("  [PASS] %s\n", c.detail.c_str());
            }
        }

        // Fail-closed: makeRuntimeReadHandle must consult the guard result.
        const std::string handleBody = bodyOfMarker (engineHeader, "RuntimeReadHandle makeRuntimeReadHandle(");

        if (handleBody.find ("rootEnterSucceeded()") == std::string::npos)
        {
            std::fprintf (stderr, "FAIL: makeRuntimeReadHandle must fail closed on guard failure\n");
            ++failures;
        }
        else
        {
            std::printf ("  [PASS] makeRuntimeReadHandle: fail-closed on guard failure\n");
        }
    }

    if (failures != 0)
    {
        std::fprintf (stderr, "ScopedEpochReaderStructuralTests: %d violation(s)\n", failures);
        return 1;
    }

    std::printf ("ScopedEpochReaderStructuralTests: PASS (reader-safety contract enforced)\n");
    return 0;
}
