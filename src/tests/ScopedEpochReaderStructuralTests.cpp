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
//     1. every listed non-RT reader body contains "ScopedEpochReader" and it
//        appears BEFORE the first "loadCurrentState" in that body;
//     2. getEQState() stays the only raw-pointer accessor (no guard inside);
//     3. the two known raw-pointer call sites hold "acquireReadGuard".
//
//   The list of functions is the contract: adding a new reader means adding it
//   here, which forces the author to look at the guard requirement.
//
// Oracle: source text only. No production code, no test hooks, no timing.
// =============================================================================

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

Check guardBeforeLoad (const std::string& source, const std::string& name)
{
    const std::string body = functionBody (source, name);

    if (body.empty())
        return { false, "function body not found: " + name };

    const std::size_t guard = body.find ("ScopedEpochReader");
    const std::size_t load = body.find ("loadCurrentState");

    if (guard == std::string::npos)
        return { false, name + ": no ScopedEpochReader in the body" };

    if (load == std::string::npos)
        return { false, name + ": no loadCurrentState in the body (contract changed?)" };

    if (! (guard < load))
        return { false, name + ": ScopedEpochReader must appear BEFORE loadCurrentState" };

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

    if (core.empty() || coefficients.empty() || eqResponse.empty() || rebuild.empty())
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

    if (failures != 0)
    {
        std::fprintf (stderr, "ScopedEpochReaderStructuralTests: %d violation(s)\n", failures);
        return 1;
    }

    std::printf ("ScopedEpochReaderStructuralTests: PASS (reader-safety contract enforced)\n");
    return 0;
}
