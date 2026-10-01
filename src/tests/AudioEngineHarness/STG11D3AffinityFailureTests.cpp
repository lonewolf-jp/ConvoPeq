// STG11D3AffinityFailureTests.cpp - STG-11-D3 / Candidate A regression (D3-T1 .. D3-T4).
//
// Target defect (STG-11-D3):
//   Failure branches of AudioEngine::applyMmcssPriority() executed diagLog()
//   outside the diagnostics guard, so the RT (audio thread) acquired a mutex and
//   allocated on the heap. F4 (SetThreadAffinityMask failure) had no guard at all,
//   so RT reached diagLog -> asyncSink -> std::mutex -> juce::String -> heap even
//   with Diagnostics OFF.
//
//   Candidate A: the RT side performs a lock-free atomic record only, and the
//   existing NonRT execution point (timerCallback) diagnoses it through the
//   existing diagLog backend. Failure policy / priorityApplied / caller return
//   value handling are unchanged.
//
// Test contract (Owner-specified four quadrants):
//   D3-T1  Diagnostics OFF + MMCSS success : failure path does not reach a backend
//   D3-T2  Diagnostics OFF + MMCSS failure : record -> NonRT diagnosis
//   D3-T3  Diagnostics ON  + MMCSS success : existing guarded success logs preserved
//   D3-T4  Diagnostics ON  + MMCSS failure : failure path backend-free even when ON
//
// Verification method:
//   - structural: reads the real production source and extracts the failure
//     branches of applyMmcssPriority (F1=SetPriorityClass, F2=SetThreadPriority,
//     F4=SetThreadAffinityMask) by brace matching, then asserts that no backend
//     token (diagLog / juce::String / mutex / asyncSink / s_logMutex / Logger /
//     heap) occurs inside them. The judgement is per statement in the failure
//     branch, not per guard, so it holds for Diagnostics ON and OFF alike (this
//     is the structural proof of INV-D3-3).
//   - functional: drives both ends of the transport with synthetic values through
//     the same private production function. No OS failure injection (Owner
//     instruction) and no new production mechanism.
//   - ordering (D3-T5): a release store publishes only what is sequenced BEFORE
//     it, so the recorder must publish the payload first, then the count (which
//     is the per-record publication marker through convo::fetchAddAtomic at
//     acq_rel), then the observed flag. D3-T5 asserts the recorder and reader
//     orders structurally and the visibility contract deterministically.
//   - Existing oracles are untouched. Production logic changes are Candidate A only.
// =============================================================================

#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "audioengine/AudioEngine.h"
#include "AudioEngineHarness.h"
#include "DeferredPublicationTestAccess.h"

namespace {

// -- production source reader (follows the D8_1_WrapperCacheTests precedent) ---
bool readProductionSource(const char* relativePath, std::string& out)
{
    FILE* f = std::fopen(relativePath, "r");
    if (!f)
    {
        char absPath[512];
        std::snprintf(absPath, sizeof(absPath), "C:/VSC_Project/ConvoPeq/%s", relativePath);
        f = std::fopen(absPath, "r");
    }
    if (!f)
        return false;
    char buf[4096];
    out.clear();
    size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0)
        out.append(buf, n);
    std::fclose(f);
    return !out.empty();
}

// Strip // line comments and /* */ block comments from a snippet so that
// structural ordering checks cannot be satisfied or broken by prose. Comment
// stripping is quote-aware so that a "//" inside a string literal survives.
std::string stripComments(const std::string& src)
{
    std::string out;
    out.reserve(src.size());
    enum class St { code, lineComment, blockComment, dq, sq };
    St st = St::code;
    for (std::size_t i = 0; i < src.size(); ++i)
    {
        const char c = src[i];
        const char n = (i + 1 < src.size()) ? src[i + 1] : '\0';
        switch (st)
        {
        case St::code:
            if (c == '/' && n == '/') { st = St::lineComment; ++i; }
            else if (c == '/' && n == '*') { st = St::blockComment; ++i; }
            else if (c == '"') { st = St::dq; out += c; }
            else if (c == '\'') { st = St::sq; out += c; }
            else { out += c; }
            break;
        case St::lineComment:
            if (c == '\n') { st = St::code; out += c; }
            break;
        case St::blockComment:
            if (c == '*' && n == '/') { st = St::code; ++i; }
            else if (c == '\n') { out += c; }
            break;
        case St::dq:
            out += c;
            if (c == '\\' && n != '\0') { out += n; ++i; }
            else if (c == '"') { st = St::code; }
            break;
        case St::sq:
            out += c;
            if (c == '\\' && n != '\0') { out += n; ++i; }
            else if (c == '\'') { st = St::code; }
            break;
        }
    }
    return out;
}

// One atomic access, in source order: the convo:: wrapper used and the target
// member. The position in the vector is the publication order, which is what
// the ordering test asserts on.
struct AtomicAccess { std::string wrapper; std::string member; };

// Collect, in source order, every atomic target member referenced by a
// convo:: wrapper call inside a snippet: "publishAtomic(affinityFailureLastMask_"
// yields { "convo::publishAtomic(", "affinityFailureLastMask_" }. Comments are
// stripped first, so prose can neither satisfy nor break an ordering assertion.
std::vector<AtomicAccess> atomicCallOrder(const std::string& snippet)
{
    const std::string code = stripComments(snippet);
    static const char* const kWrappers[] = {
        "convo::publishAtomic(", "convo::consumeAtomic(", "convo::fetchAddAtomic(",
        "convo::exchangeAtomic(", "convo::fetchSubAtomic(", "convo::compareExchangeAtomic("
    };
    std::vector<AtomicAccess> order;
    for (std::size_t pos = 0; pos < code.size(); ++pos)
    {
        for (const char* w : kWrappers)
        {
            const std::size_t wl = std::string(w).size();
            if (code.compare(pos, wl, w) != 0)
                continue;
            std::size_t p = pos + wl;
            while (p < code.size() && (code[p] == ' ' || code[p] == '\n' || code[p] == '\t'))
                ++p;
            std::size_t e = p;
            while (e < code.size() && (std::isalnum(static_cast<unsigned char>(code[e]))
                                      || code[e] == '_'))
                ++e;
            if (e > p)
                order.push_back({ w, code.substr(p, e - p) });
            pos = (e > pos) ? e - 1 : pos;
            break;
        }
    }
    return order;
}

// Index of the first access to member, or -1.
int indexOfMember(const std::vector<AtomicAccess>& order, const char* member)
{
    for (std::size_t i = 0; i < order.size(); ++i)
        if (order[i].member == member)
            return static_cast<int>(i);
    return -1;
}

// Wrapper used by the first access to member, or "" when absent.
std::string wrapperOfMember(const std::vector<AtomicAccess>& order, const char* member)
{
    for (const AtomicAccess& a : order)
        if (a.member == member)
            return a.wrapper;
    return std::string();
}

// Extract a function body from a signature anchor to the matching closing brace.
// bodyStart receives the offset of the opening brace inside src, so that
// positions found in the returned body can be mapped back onto src.
bool extractFunctionBody(const std::string& src, const char* anchor, std::string& body,
                         std::size_t* bodyStart = nullptr)
{
    const std::size_t start = src.find(anchor);
    if (start == std::string::npos)
        return false;
    const std::size_t open = src.find('{', start);
    if (open == std::string::npos)
        return false;
    int depth = 0;
    for (std::size_t i = open; i < src.size(); ++i)
    {
        if (src[i] == '{')
            ++depth;
        else if (src[i] == '}')
        {
            if (--depth == 0)
            {
                body = src.substr(open, i - open + 1);
                if (bodyStart)
                    *bodyStart = open;
                return true;
            }
        }
    }
    return false;
}

// Extract an "if (<cond>) { ... }" block inside a body by brace matching.
// Only block form is accepted, so a stray ";" or "{" between the condition and
// the brace is rejected. The last match wins if a condition occurs more than once.
bool extractIfBlock(const std::string& body, const std::string& condition,
                    std::string& found)
{
    bool any = false;
    std::size_t search = 0;
    for (;;)
    {
        const std::size_t condPos = body.find(condition, search);
        if (condPos == std::string::npos)
            break;
        search = condPos + 1;
        const std::size_t open = body.find('{', condPos);
        if (open == std::string::npos)
            break;
        bool notBlockForm = false;
        for (std::size_t i = condPos; i < open; ++i)
        {
            if (body[i] == '{' || body[i] == ';')
            {
                notBlockForm = true;
                break;
            }
        }
        if (notBlockForm)
            continue;
        int depth = 0;
        for (std::size_t i = open; i < body.size(); ++i)
        {
            if (body[i] == '{')
                ++depth;
            else if (body[i] == '}')
            {
                if (--depth == 0)
                {
                    found = body.substr(open, i - open + 1);
                    any = true;
                    search = i;
                    break;
                }
            }
        }
    }
    return any;
}

// Backend / allocation tokens that must not be RT reachable.
const char* const kForbiddenOnRtFailure[] = {
    "diagLog(",        // logging backend entry point
    "DBG(",            // JUCE debug stream
    "asyncSink",       // mutex + SPSC push holding
    "s_logMutex",      // MP-to-SPSC mutex
    "flushLogBuffer",  // NonRT drain
    "juce::String",    // heap allocation
    "std::mutex",      // lock
    "lock_guard",      // lock
    "Logger::",        // logging backend
    "OutputDebugString",
    "malloc(", "calloc(", "realloc(", "free(",
    "new ",            // heap allocation
    "make_unique", "make_shared"
};

// Raw std::atomic API. D3 must use only the existing ConvoPeq wrappers.
const char* const kForbiddenRawAtomic[] = {
    ".load(", ".store(", ".fetch_add(", ".fetch_sub(", ".fetch_or(",
    ".fetch_xor(", ".exchange(", ".compare_exchange"
};

bool containsAny(const std::string& s, const char* const* tokens, std::size_t count,
                 const char** whichOut)
{
    for (std::size_t i = 0; i < count; ++i)
    {
        if (s.find(tokens[i]) != std::string::npos)
        {
            if (whichOut)
                *whichOut = tokens[i];
            return true;
        }
    }
    return false;
}

// Mark every character of src as "only reachable with Diagnostics ON" or not.
// A proper preprocessor conditional stack is used so that unrelated #if / #endif
// pairs elsewhere in the translation unit cannot corrupt the mapping, and so
// that #else inside the diagnostics guard correctly un-guards its branch.
std::vector<bool> markDiagnosticsGuarded(const std::string& src)
{
    std::vector<bool> guarded(src.size(), false);
    struct Cond { bool isDiagnostics; bool active; };
    std::vector<Cond> stack;

    auto anyActive = [&stack]() {
        for (const Cond& c : stack)
            if (c.active)
                return true;
        return false;
    };

    std::size_t i = 0;
    while (i < src.size())
    {
        const std::size_t eol = src.find('\n', i);
        const std::size_t end = (eol == std::string::npos) ? src.size() : eol;
        const std::string line = src.substr(i, end - i);

        const std::size_t hash = line.find_first_not_of(" \t");
        const bool isDirective = (hash != std::string::npos) && (line[hash] == '#');
        const bool opensCond = isDirective
            && (line.compare(hash, 3, "#if") == 0)
            && (line.compare(hash, 5, "#ifdef") != 0)
            && (line.compare(hash, 6, "#ifndef") != 0);
        const bool isElse = isDirective
            && (line.compare(hash, 5, "#else") == 0 || line.compare(hash, 5, "#elif") == 0);
        const bool closesCond = isDirective && (line.compare(hash, 6, "#endif") == 0);

        const bool active = anyActive();
        for (std::size_t k = i; k < end; ++k)
            guarded[k] = active;

        if (opensCond)
        {
            stack.push_back({ line.find("CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS") != std::string::npos,
                              true });
        }
        else if (isElse)
        {
            if (!stack.empty())
            {
                // #else flips the diagnostics guard; #elif conservatively
                // deactivates so an assertion can never be silently satisfied.
                stack.back().active = (line.compare(hash, 5, "#else") == 0)
                    ? !stack.back().isDiagnostics
                    : false;
            }
        }
        else if (closesCond)
        {
            if (!stack.empty())
                stack.pop_back();
        }

        if (eol == std::string::npos)
            break;
        i = end + 1;
    }
    return guarded;
}

using Access = DeferredPublicationTestAccess;

} // namespace

// =============================================================================
// D3-T1 / D3-T4 (structural core): the three failure branches reach no backend.
// Holds with Diagnostics ON and OFF alike, proven by brace extraction.
// =============================================================================
static bool checkD3T1FailureBranchesAreRtSafe()
{
    std::string src;
    if (!readProductionSource("src/audioengine/AudioEngine.Timer.cpp", src))
    {
        std::fprintf(stderr, "D3-T1: cannot open AudioEngine.Timer.cpp\n");
        return false;
    }
    std::string body;
    std::size_t bodyStart = 0;
    if (!extractFunctionBody(src, "bool AudioEngine::applyMmcssPriority()", body, &bodyStart))
    {
        std::fprintf(stderr, "D3-T1: cannot extract applyMmcssPriority body\n");
        return false;
    }

    // Three failure branches. F4 is the primary D3 target; F1 and F2 carry the
    // same RT-reachability defect and are covered by the same transport.
    struct Branch { const char* label; const char* condition; };
    static const Branch kBranches[] = {
        { "F1 SetPriorityClass failure", "if (pcResult == 0)" },
        { "F2 SetThreadPriority failure", "if (tpResult == 0)" },
        { "F4 SetThreadAffinityMask failure", "if (prevMask == 0)" },
    };
    for (const Branch& br : kBranches)
    {
        std::string blk;
        if (!extractIfBlock(body, br.condition, blk))
        {
            std::fprintf(stderr, "D3-T1: cannot extract failure branch: %s\n", br.label);
            return false;
        }
        const char* which = nullptr;
        if (containsAny(blk, kForbiddenOnRtFailure,
                        sizeof(kForbiddenOnRtFailure) / sizeof(kForbiddenOnRtFailure[0]), &which))
        {
            std::fprintf(stderr, "D3-T1: %s reaches RT-forbidden backend '%s'\n",
                         br.label, which ? which : "?");
            return false;
        }
        if (blk.find("recordAffinityFailure(") == std::string::npos)
        {
            std::fprintf(stderr, "D3-T1: %s has no recordAffinityFailure replacement\n", br.label);
            return false;
        }
        std::printf("  D3-T1: %-38s RT-safe (backend-free, record-only)\n", br.label);
    }

    // INV-D3-3: no recordAffinityFailure call sits inside a diagnostics guard,
    // so Diagnostics OFF is never a precondition for the failure path.
    const std::vector<bool> guarded = markDiagnosticsGuarded(src);
    for (std::size_t pos = body.find("recordAffinityFailure(");
         pos != std::string::npos;
         pos = body.find("recordAffinityFailure(", pos + 1))
    {
        if (guarded[bodyStart + pos])
        {
            std::fprintf(stderr, "D3-T1: recordAffinityFailure call is inside a diagnostics guard\n");
            return false;
        }
    }

    // INV-D3-1/2: the recorder body has no backend, no allocation, no raw atomic.
    std::string recBody;
    if (!extractFunctionBody(src, "void AudioEngine::recordAffinityFailure(", recBody))
    {
        std::fprintf(stderr, "D3-T1: cannot extract recordAffinityFailure body\n");
        return false;
    }
    const char* which2 = nullptr;
    if (containsAny(recBody, kForbiddenOnRtFailure,
                    sizeof(kForbiddenOnRtFailure) / sizeof(kForbiddenOnRtFailure[0]), &which2))
    {
        std::fprintf(stderr, "D3-T1: recordAffinityFailure body reaches '%s'\n",
                     which2 ? which2 : "?");
        return false;
    }
    if (containsAny(recBody, kForbiddenRawAtomic,
                    sizeof(kForbiddenRawAtomic) / sizeof(kForbiddenRawAtomic[0]), &which2))
    {
        std::fprintf(stderr, "D3-T1: recordAffinityFailure uses raw atomic API '%s'\n",
                     which2 ? which2 : "?");
        return false;
    }
    if (recBody.find("convo::publishAtomic(") == std::string::npos
        || recBody.find("convo::fetchAddAtomic(") == std::string::npos)
    {
        std::fprintf(stderr, "D3-T1: recordAffinityFailure does not use convo:: atomic wrappers\n");
        return false;
    }
    std::printf("  D3-T1: recordAffinityFailure = convo:: wrappers only, no backend/allocation\n");

    // The NonRT report keeps the existing diagLog backend.
    std::string repBody;
    if (!extractFunctionBody(src, "void AudioEngine::reportAffinityFailureIfRecorded()", repBody))
    {
        std::fprintf(stderr, "D3-T1: cannot extract reportAffinityFailureIfRecorded body\n");
        return false;
    }
    if (repBody.find("diagLog(") == std::string::npos)
    {
        std::fprintf(stderr, "D3-T1: NonRT diagnosis lost its diagLog backend\n");
        return false;
    }

    // The NonRT diagnosis runs from the existing execution point, no new thread.
    std::string cbBody;
    if (!extractFunctionBody(src, "void AudioEngine::timerCallback()", cbBody))
    {
        std::fprintf(stderr, "D3-T1: cannot extract timerCallback body\n");
        return false;
    }
    if (cbBody.find("reportAffinityFailureIfRecorded();") == std::string::npos)
    {
        std::fprintf(stderr, "D3-T1: reportAffinityFailureIfRecorded not called from timerCallback\n");
        return false;
    }

    // INV-D3-4: the observation creates no new publication, retire, or recovery
    // authority, and no new queue ownership.
    static const char* const kNewAuthority[] = {
        "std::thread", "std::jthread", "juce::Timer", "Timer::",
        "LockFreeRingBuffer", "enqueueDeferredDelete", "enqueueRetire",
        "DeferredDeletionQueue", "RuntimePublication", "Recovery"
    };
    for (const char* tok : kNewAuthority)
    {
        if (recBody.find(tok) != std::string::npos)
        {
            std::fprintf(stderr, "D3-T1: record creates new authority '%s'\n", tok);
            return false;
        }
    }
    // INV-D3-5: lossy-coalescing semantics are explicit. The count is monotonic,
    // the snapshot fields are last-wins, and no ownership token is transported.
    if (recBody.find("affinityFailureCount_") == std::string::npos
        || recBody.find("affinityFailureLastMask_") == std::string::npos
        || recBody.find("affinityFailureLastError_") == std::string::npos
        || recBody.find("affinityFailureLastKind_") == std::string::npos)
    {
        std::fprintf(stderr, "D3-T1: coalescing state fields missing from recorder\n");
        return false;
    }
    if (repBody.find("affinityFailureReportedCount_") == std::string::npos)
    {
        std::fprintf(stderr, "D3-T1: NonRT diagnosis has no once-only bookkeeping\n");
        return false;
    }
    std::printf("  D3-T1: NonRT report uses existing diagLog + existing timerCallback (no new authority)\n");
    std::printf("  D3-T1: lossy-coalescing semantics explicit (monotonic count + last-wins snapshot)\n");

    std::printf("STG11D3AffinityFailureTests: D3-T1/D3-T4 PASS (RT failure paths backend-free, ON/OFF independent)\n");
    return true;
}

// =============================================================================
// D3-T1 completion (structural): with Diagnostics OFF the success path has no
// backend either, and the whole RT function takes no lock directly.
// =============================================================================
static bool checkD3T1DiagnosticsOffSuccessHasNoBackend()
{
    std::string src;
    if (!readProductionSource("src/audioengine/AudioEngine.Timer.cpp", src))
    {
        std::fprintf(stderr, "D3-T1b: cannot open AudioEngine.Timer.cpp\n");
        return false;
    }
    std::string body;
    std::size_t bodyStart = 0;
    if (!extractFunctionBody(src, "bool AudioEngine::applyMmcssPriority()", body, &bodyStart))
    {
        std::fprintf(stderr, "D3-T1b: cannot extract applyMmcssPriority body\n");
        return false;
    }
    const std::vector<bool> guarded = markDiagnosticsGuarded(src);
    static const char* const kBackendTokens[] = { "diagLog(", "juce::String" };
    for (const char* tok : kBackendTokens)
    {
        for (std::size_t pos = body.find(tok); pos != std::string::npos;
             pos = body.find(tok, pos + 1))
        {
            if (!guarded[bodyStart + pos])
            {
                std::fprintf(stderr, "D3-T1b: '%s' reachable with Diagnostics OFF\n", tok);
                return false;
            }
        }
    }
    static const char* const kDirectLocks[] = { "std::mutex", "lock_guard", "s_logMutex", "asyncSink" };
    const char* which = nullptr;
    if (containsAny(body, kDirectLocks, 4, &which))
    {
        std::fprintf(stderr, "D3-T1b: applyMmcssPriority directly touches '%s'\n",
                     which ? which : "?");
        return false;
    }
    std::printf("STG11D3AffinityFailureTests: D3-T1b PASS (Diagnostics OFF success path has no backend)\n");
    return true;
}

// =============================================================================
// D3-T3 (structural): Diagnostics ON + success non-regression. The existing
// guarded success logs are preserved.
// =============================================================================
static bool checkD3T3SuccessLogPreserved()
{
    std::string src;
    if (!readProductionSource("src/audioengine/AudioEngine.Timer.cpp", src))
    {
        std::fprintf(stderr, "D3-T3: cannot open AudioEngine.Timer.cpp\n");
        return false;
    }
    if (src.find("[AFFINITY] AudioThread pinned mask=0x") == std::string::npos)
    {
        std::fprintf(stderr, "D3-T3: affinity success log lost\n");
        return false;
    }
    if (src.find("[AFFINITY] P/E cores: AudioThread affinity skipped") == std::string::npos)
    {
        std::fprintf(stderr, "D3-T3: heterogeneous-core skip log lost\n");
        return false;
    }
    if (src.find("[NATIVE_RT] applied: win32Prio=") == std::string::npos)
    {
        std::fprintf(stderr, "D3-T3: NativeRT applied log lost\n");
        return false;
    }
    // Success-path change is limited to the guard restructure; the applied log
    // stays guarded so Diagnostics OFF stays RT-safe.
    const std::vector<bool> guarded = markDiagnosticsGuarded(src);
    for (const char* tok : { "[NATIVE_RT] applied", "[AFFINITY] AudioThread pinned",
                             "[AFFINITY] P/E cores" })
    {
        const std::size_t pos = src.find(tok);
        if (pos == std::string::npos || !guarded[pos])
        {
            std::fprintf(stderr, "D3-T3: success log '%s' is not diagnostics-guarded\n", tok);
            return false;
        }
    }
    std::printf("STG11D3AffinityFailureTests: D3-T3 PASS (existing guarded success logs preserved)\n");
    return true;
}

// =============================================================================
// D3-T2 / D3-T4 (functional): record -> state -> NonRT diagnosis, using the one
// flag-independent mechanism. The fixture is owned by the harness so the stack
// footprint of this TU stays small. No OS failure injection.
// =============================================================================
static bool checkD3T2RecordAndDiagnose()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-11-D3 T-2] harness start failed\n");
        return false;
    }
    h.stop();
    AudioEngine& e = h.engine();

    // Clean initial state. A report with nothing recorded must be a no-op.
    if (Access::affinityFailureObserved(e) || Access::affinityFailureCount(e) != 0)
    {
        std::fprintf(stderr, "D3-T2: initial state not clean\n");
        return false;
    }
    Access::reportAffinityFailure(e);
    if (Access::affinityFailureReportedCount(e) != 0)
    {
        std::fprintf(stderr, "D3-T2: report without record changed state\n");
        return false;
    }

    // First failure, recorded through the same production function and the same
    // wrapper path the RT side uses.
    Access::recordAffinityFailure(e, 0, 0xAB, 5);
    if (!Access::affinityFailureObserved(e)
        || Access::affinityFailureCount(e) != 1
        || Access::affinityFailureLastKind(e) != 0
        || Access::affinityFailureLastMask(e) != 0xAB
        || Access::affinityFailureLastError(e) != 5)
    {
        std::fprintf(stderr, "D3-T2: record state mismatch\n");
        return false;
    }

    // The NonRT diagnosis advances exactly once per recorded count.
    Access::reportAffinityFailure(e);
    if (Access::affinityFailureReportedCount(e) != 1)
    {
        std::fprintf(stderr, "D3-T2: first report did not advance\n");
        return false;
    }
    Access::reportAffinityFailure(e);
    if (Access::affinityFailureReportedCount(e) != 1)
    {
        std::fprintf(stderr, "D3-T2: duplicate report advanced (must be once-only)\n");
        return false;
    }

    // Second failure: last-wins snapshot plus a monotonic count, so it is
    // diagnosed again. That is the declared lossy-coalescing behaviour.
    Access::recordAffinityFailure(e, 0, 0xCD, 6);
    if (Access::affinityFailureCount(e) != 2
        || Access::affinityFailureLastMask(e) != 0xCD
        || Access::affinityFailureLastError(e) != 6)
    {
        std::fprintf(stderr, "D3-T2: second record state mismatch (coalescing broken)\n");
        return false;
    }
    Access::reportAffinityFailure(e);
    if (Access::affinityFailureReportedCount(e) != 2)
    {
        std::fprintf(stderr, "D3-T2: second report did not advance\n");
        return false;
    }

    // Third failure exercises the kind field, so the NativeRT failures travel
    // over the same transport.
    Access::recordAffinityFailure(e, 2, 0, 87);
    if (Access::affinityFailureCount(e) != 3
        || Access::affinityFailureLastKind(e) != 2
        || Access::affinityFailureLastError(e) != 87)
    {
        std::fprintf(stderr, "D3-T2: third record (kind=2) state mismatch\n");
        return false;
    }
    Access::reportAffinityFailure(e);
    if (Access::affinityFailureReportedCount(e) != 3)
    {
        std::fprintf(stderr, "D3-T2: third report did not advance\n");
        return false;
    }

    std::printf("STG11D3AffinityFailureTests: D3-T2/D3-T4 PASS (record + once-per-count NonRT diagnosis)\n");
    return true;
}

// =============================================================================
// D3-T5 (structural): publication ordering in the RT recorder.
//   A release store publishes only what is sequenced BEFORE it. Marking the
//   record before writing the payload therefore leaves the payload outside the
//   happens-before edge and the reader cannot rely on the acquire. Required
//   order is therefore payload -> count -> observed.
//
//   count is the per-record publication marker: convo::fetchAddAtomic is
//   acq_rel (a release operation), so an acquire load of count reading N
//   synchronizes-with the increment that produced N and consequently exposes
//   the payload of record #N. observed is a monotonic fast-out flag and must
//   be the last store, never the payload carrier.
// =============================================================================
static bool checkD3T5RecorderPublicationOrder()
{
    std::string src;
    if (!readProductionSource("src/audioengine/AudioEngine.Timer.cpp", src))
    {
        std::fprintf(stderr, "D3-T5: cannot open AudioEngine.Timer.cpp\n");
        return false;
    }
    std::string recBody;
    if (!extractFunctionBody(src, "void AudioEngine::recordAffinityFailure(", recBody))
    {
        std::fprintf(stderr, "D3-T5: cannot extract recordAffinityFailure body\n");
        return false;
    }

    const std::vector<AtomicAccess> order = atomicCallOrder(recBody);

    static const char* const kPayload[] = {
        "affinityFailureLastKind_", "affinityFailureLastMask_", "affinityFailureLastError_"
    };
    const int iCount = indexOfMember(order, "affinityFailureCount_");
    const int iObserved = indexOfMember(order, "affinityFailureObserved_");
    if (iCount < 0 || iObserved < 0)
    {
        std::fprintf(stderr, "D3-T5: recorder must touch count and observed\n");
        return false;
    }

    // (1) Every payload store is sequenced BEFORE the count increment.
    for (const char* p : kPayload)
    {
        const int i = indexOfMember(order, p);
        if (i < 0)
        {
            std::fprintf(stderr, "D3-T5: recorder lost payload member %s\n", p);
            return false;
        }
        if (i > iCount)
        {
            std::fprintf(stderr, "D3-T5: payload %s is published AFTER the count increment "
                                 "(index %d > %d): the release/acquire pair would not cover it\n",
                         p, i, iCount);
            return false;
        }
    }

    // (2) The count increment is after the payload and before observed.
    if (!(iCount < iObserved))
    {
        std::fprintf(stderr, "D3-T5: observed must be the last store (count=%d observed=%d)\n",
                     iCount, iObserved);
        return false;
    }

    // (3) observed is the very last atomic access in the recorder.
    if (iObserved != static_cast<int>(order.size()) - 1)
    {
        std::fprintf(stderr, "D3-T5: observed is not the last recorder access (%d of %zu)\n",
                     iObserved, order.size());
        return false;
    }

    // (4) The count must be a monotonic RMW (release operation), never a
    //     plain store, otherwise the acquire on the reader side gains nothing.
    if (wrapperOfMember(order, "affinityFailureCount_") != "convo::fetchAddAtomic(")
    {
        std::fprintf(stderr, "D3-T5: count must use convo::fetchAddAtomic (acq_rel) for "
                             "monotonicity and release publication\n");
        return false;
    }
    if (wrapperOfMember(order, "affinityFailureObserved_") != "convo::publishAtomic(")
    {
        std::fprintf(stderr, "D3-T5: observed must use convo::publishAtomic (release)\n");
        return false;
    }

    // (5) No raw atomic API, so the memory orders above are the whole contract.
    const char* which = nullptr;
    if (containsAny(recBody, kForbiddenRawAtomic,
                    sizeof(kForbiddenRawAtomic) / sizeof(kForbiddenRawAtomic[0]), &which))
    {
        std::fprintf(stderr, "D3-T5: recorder uses raw atomic API '%s'\n", which ? which : "?");
        return false;
    }

    std::printf("  D3-T5: recorder order = payload -> count(fetchAdd/acq_rel) -> observed(publish/release)\n");
    return true;
}

// =============================================================================
// D3-T5 (structural): read order in the NonRT report.
//   observed is a fast-out only; the snapshot reads must follow the count
//   acquire so that count == N implies "record #N's payload is visible".
// =============================================================================
static bool checkD3T5ReportReadOrder()
{
    std::string src;
    if (!readProductionSource("src/audioengine/AudioEngine.Timer.cpp", src))
    {
        std::fprintf(stderr, "D3-T5r: cannot open AudioEngine.Timer.cpp\n");
        return false;
    }
    std::string repBody;
    if (!extractFunctionBody(src, "void AudioEngine::reportAffinityFailureIfRecorded()", repBody))
    {
        std::fprintf(stderr, "D3-T5r: cannot extract reportAffinityFailureIfRecorded body\n");
        return false;
    }

    const std::vector<AtomicAccess> order = atomicCallOrder(repBody);
    const int iCount = indexOfMember(order, "affinityFailureCount_");
    if (iCount < 0)
    {
        std::fprintf(stderr, "D3-T5r: report does not read count\n");
        return false;
    }
    static const char* const kSnapshot[] = {
        "affinityFailureLastKind_", "affinityFailureLastMask_", "affinityFailureLastError_"
    };
    for (const char* s : kSnapshot)
    {
        const int i = indexOfMember(order, s);
        if (i < 0)
        {
            std::fprintf(stderr, "D3-T5r: report does not read snapshot member %s\n", s);
            return false;
        }
        if (i < iCount)
        {
            std::fprintf(stderr, "D3-T5r: snapshot %s is read BEFORE the count acquire "
                                 "(index %d < %d): payload visibility is not guaranteed\n",
                         s, i, iCount);
            return false;
        }
    }

    // Every read that carries the snapshot or the count must be an acquire.
    if (wrapperOfMember(order, "affinityFailureCount_") != "convo::consumeAtomic(")
    {
        std::fprintf(stderr, "D3-T5r: count must be read with convo::consumeAtomic (acquire)\n");
        return false;
    }
    static const char* const kRequiredAcquire[] = {
        "affinityFailureCount_", "affinityFailureLastKind_",
        "affinityFailureLastMask_", "affinityFailureLastError_"
    };
    for (const char* m : kRequiredAcquire)
    {
        if (wrapperOfMember(order, m) != "convo::consumeAtomic(")
        {
            std::fprintf(stderr, "D3-T5r: %s must be read with convo::consumeAtomic (acquire)\n", m);
            return false;
        }
    }
    // The acquire is spelled out, not left to a default that could change.
    if (repBody.find("std::memory_order_acquire") == std::string::npos)
    {
        std::fprintf(stderr, "D3-T5r: report does not spell out memory_order_acquire\n");
        return false;
    }

    const char* which = nullptr;
    if (containsAny(repBody, kForbiddenRawAtomic,
                    sizeof(kForbiddenRawAtomic) / sizeof(kForbiddenRawAtomic[0]), &which))
    {
        std::fprintf(stderr, "D3-T5r: report uses raw atomic API '%s'\n", which ? which : "?");
        return false;
    }

    std::printf("  D3-T5: report order = observed(fast-out) -> count(acquire) -> snapshot(acquire)\n");
    return true;
}

// =============================================================================
// D3-T5 (deterministic, functional): record -> observed acquire -> payload
//   visibility, and the multi-record contract from the Owner: count is
//   monotonic, the snapshot is last-wins, an older snapshot is never confirmed
//   as a newer count's snapshot.
//
//   Single-threaded observation of the publication contract: after each record
//   the reader observes count == k AND a snapshot belonging to record #k. Since
//   the recorder publishes the payload before incrementing, any observation of
//   count == k must already expose record #k's payload. A reorder that marked
//   the record first would let a reader observe count == k with a stale or
//   default payload, which this oracle rejects.
// =============================================================================
static bool checkD3T5RecordThenVisible()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-11-D3 T-5] harness start failed\n");
        return false;
    }
    h.stop();
    AudioEngine& e = h.engine();

    if (Access::affinityFailureCount(e) != 0 || Access::affinityFailureObserved(e))
    {
        std::fprintf(stderr, "D3-T5: initial state not clean\n");
        return false;
    }

    // Five records with distinct payloads. Each round asserts the exact
    // contract: observed is set, count is exactly k (monotonic, no lost or
    // phantom increment), and the snapshot is the one written by record #k
    // (last-wins, never an older one).
    for (int k = 1; k <= 5; ++k)
    {
        const std::uint32_t kind = static_cast<std::uint32_t>(k % 3);
        const std::uint64_t mask = 0x100u + static_cast<std::uint64_t>(k);
        const std::uint32_t err = static_cast<std::uint32_t>(100 + k);
        Access::recordAffinityFailure(e, kind, mask, err);

        if (!Access::affinityFailureObserved(e))
        {
            std::fprintf(stderr, "D3-T5: round %d did not set observed\n", k);
            return false;
        }
        const std::uint64_t count = Access::affinityFailureCount(e);
        if (count != static_cast<std::uint64_t>(k))
        {
            std::fprintf(stderr, "D3-T5: count is not monotonic (expected %d, got %llu)\n",
                         k, (unsigned long long)count);
            return false;
        }
        // count == k implies record #k's payload is visible.
        if (Access::affinityFailureLastKind(e) != kind
            || Access::affinityFailureLastMask(e) != mask
            || Access::affinityFailureLastError(e) != err)
        {
            std::fprintf(stderr, "D3-T5: round %d snapshot is stale (kind=%u mask=0x%llX err=%u, "
                                 "expected kind=%u mask=0x%llX err=%u)\n",
                         k,
                         Access::affinityFailureLastKind(e),
                         (unsigned long long)Access::affinityFailureLastMask(e),
                         Access::affinityFailureLastError(e),
                         kind, (unsigned long long)mask, err);
            return false;
        }

        // The NonRT diagnosis must advance by exactly one per record, i.e. the
        // count it acts on is never behind the observed snapshot.
        const std::uint64_t before = Access::affinityFailureReportedCount(e);
        Access::reportAffinityFailure(e);
        const std::uint64_t after = Access::affinityFailureReportedCount(e);
        if (after != before + 1)
        {
            std::fprintf(stderr, "D3-T5: round %d report advanced %llu -> %llu (expected +1)\n",
                         k, (unsigned long long)before, (unsigned long long)after);
            return false;
        }
    }

    // A report with no new record must not advance (count is the contract).
    const std::uint64_t settled = Access::affinityFailureReportedCount(e);
    Access::reportAffinityFailure(e);
    if (Access::affinityFailureReportedCount(e) != settled)
    {
        std::fprintf(stderr, "D3-T5: report advanced without a new record\n");
        return false;
    }

    std::printf("STG11D3AffinityFailureTests: D3-T5 PASS (payload published before count, "
                "count monotonic, snapshot last-wins)\n");
    return true;
}

// Entry point called from the harness main (PublishPipelineIntegrationTests.cpp).
// No new CTest registration: this is a sub-test of the AudioEngineHarness
// executable, matching the STG-8 and STG-9 shape.
int runSTG11D3AffinityFailureTests()
{
    bool ok = true;
    if (!checkD3T1FailureBranchesAreRtSafe())
    {
        std::fprintf(stderr, "FAIL: D3-T1/D3-T4 RT failure paths backend-free\n");
        ok = false;
    }
    if (!checkD3T1DiagnosticsOffSuccessHasNoBackend())
    {
        std::fprintf(stderr, "FAIL: D3-T1b diagnostics-OFF success path\n");
        ok = false;
    }
    if (!checkD3T2RecordAndDiagnose())
    {
        std::fprintf(stderr, "FAIL: D3-T2/D3-T4 record and diagnose\n");
        ok = false;
    }
    if (!checkD3T3SuccessLogPreserved())
    {
        std::fprintf(stderr, "FAIL: D3-T3 success logs preserved\n");
        ok = false;
    }
    if (!checkD3T5RecorderPublicationOrder())
    {
        std::fprintf(stderr, "FAIL: D3-T5 recorder publication order\n");
        ok = false;
    }
    if (!checkD3T5ReportReadOrder())
    {
        std::fprintf(stderr, "FAIL: D3-T5 report read order\n");
        ok = false;
    }
    if (!checkD3T5RecordThenVisible())
    {
        std::fprintf(stderr, "FAIL: D3-T5 record then visible\n");
        ok = false;
    }
    if (ok)
        std::printf("STG11D3AffinityFailureTests: PASS (D3-T1/D3-T2/D3-T3/D3-T4/D3-T5)\n");
    return ok ? 0 : 1;
}
