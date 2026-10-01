// STG11D4SuccessObservationTests.cpp - STG-11-D4 / OBS-1 regression (D4-T1 .. D4-T6).
//
// Target defect (STG-11-D4, recorded as OBS-1 in the D3 gate):
//   The three Diagnostics ON success branches of AudioEngine::applyMmcssPriority()
//   executed diagLog() directly on the RT (audio thread):
//     S1  [NATIVE_RT] applied (NativeRT success)
//     S2  [AFFINITY] AudioThread pinned (affinity success)
//     S3  [AFFINITY] P/E cores skipped (heterogeneous-core path)
//   diagLog -> asyncSink -> std::mutex -> juce::String -> heap on RT.
//
//   Fix (same idea as D3 Candidate A, adapted - not mechanically copied):
//   the RT side performs a lock-free atomic record only
//   (recordSuccessObserved, convo:: wrappers), and the existing NonRT execution
//   point (timerCallback) diagnoses it through the existing diagLog backend
//   (reportSuccessIfRecorded). Success policy / priorityApplied / caller return
//   value handling are unchanged.
//
// Test contract (Owner-specified):
//   D4-T1  Diagnostics ON + NativeRT success: no backend reachable from RT paths
//   D4-T2  Diagnostics ON + affinity success: pinned info kept on the NonRT side
//   D4-T3  Diagnostics ON + hetero path: skipped meaning kept
//   D4-T4  Diagnostics OFF: existing OFF semantics unbroken
//   D4-T5  RT structural audit on the success path
//   D4-T6  D1 / D2 / D3 regression guard (D3 ordering intact, read-only check)
//
// Verification method:
//   - structural: reads the real production source and extracts the success
//     branches of applyMmcssPriority by brace matching, then asserts that no
//     backend token occurs inside them, and that the NonRT report keeps the
//     original wording under the diagnostics guard.
//   - functional: drives both ends of the transport with synthetic values through
//     the same private production function. No OS injection, no new mechanism.
//   - D3 files, functions, members, tests and ordering are untouched; D4-T6
//     asserts the D3 publication order is still intact (read-only).
//   - Existing oracles are untouched. Production logic changes are D4 only.
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

// Strip // line comments and /* */ block comments, quote-aware.
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

// One atomic access, in source order.
struct AtomicAccess { std::string wrapper; std::string member; };

// Collect every atomic target member referenced by a convo:: wrapper call,
// in source order. Comments are stripped first.
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

int indexOfMember(const std::vector<AtomicAccess>& order, const char* member)
{
    for (std::size_t i = 0; i < order.size(); ++i)
        if (order[i].member == member)
            return static_cast<int>(i);
    return -1;
}

std::string wrapperOfMember(const std::vector<AtomicAccess>& order, const char* member)
{
    for (const AtomicAccess& a : order)
        if (a.member == member)
            return a.wrapper;
    return std::string();
}

// Extract a function body from a signature anchor to the matching closing brace.
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
const char* const kForbiddenOnRt[] = {
    "diagLog(",
    "DBG(",
    "asyncSink",
    "s_logMutex",
    "flushLogBuffer",
    "juce::String",
    "std::mutex",
    "lock_guard",
    "unique_lock",
    "shared_lock",
    "condition_variable",
    "Logger::",
    "OutputDebugString",
    "malloc(", "calloc(", "realloc(", "free(",
    "new ",
    "make_unique", "make_shared"
};

// Raw std::atomic API. D4 must use only the existing ConvoPeq wrappers.
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

// Proper preprocessor conditional stack: #else flips the diagnostics guard,
// #elif conservatively deactivates.
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
// D4-T1 / D4-T5 (structural core): the three success sites reach no backend.
// Holds with Diagnostics ON and OFF alike, proven by brace extraction.
// S1 = NativeRT applied, S2 = affinity pinned, S3 = hetero skipped.
// =============================================================================
static bool checkD4T1SuccessSitesAreRtSafe()
{
    std::string src;
    if (!readProductionSource("src/audioengine/AudioEngine.Timer.cpp", src))
    {
        std::fprintf(stderr, "D4-T1: cannot open AudioEngine.Timer.cpp\n");
        return false;
    }
    std::string body;
    if (!extractFunctionBody(src, "bool AudioEngine::applyMmcssPriority()", body))
    {
        std::fprintf(stderr, "D4-T1: cannot extract applyMmcssPriority body\n");
        return false;
    }

    // The old success wordings must NOT occur inside the RT function at all.
    // They belong to the NonRT report now. Presence here means the Diagnostics
    // ON success path still reaches the logging backend on RT.
    static const char* const kOldWording[] = {
        "[NATIVE_RT] applied: win32Prio=",
        "[AFFINITY] AudioThread pinned mask=0x",
        "[AFFINITY] P/E cores: AudioThread affinity skipped"
    };
    for (const char* w : kOldWording)
    {
        if (body.find(w) != std::string::npos)
        {
            std::fprintf(stderr, "D4-T1: success wording '%s' still inside RT function "
                                 "(ON success path reaches logging backend)\n", w);
            return false;
        }
    }

    struct Site { const char* label; const char* condition; };
    static const Site kSites[] = {
        { "S1 NativeRT applied", "if (nativeRtOk)" },
        { "S2 affinity pinned", "recordSuccessObserved(2," },
    };
    for (const Site& s : kSites)
    {
        std::string blk;
        if (s.condition[0] == 'i')
        {
            if (!extractIfBlock(body, s.condition, blk))
            {
                std::fprintf(stderr, "D4-T1: cannot extract success site: %s\n", s.label);
                return false;
            }
        }
        else
        {
            // S2 is an else-branch record call: take a window around the call.
            const std::size_t pos = body.find(s.condition);
            if (pos == std::string::npos)
            {
                std::fprintf(stderr, "D4-T1: %s has no recordSuccessObserved replacement "
                                     "(old diagLog path still present)\n", s.label);
                return false;
            }
            blk = body.substr(pos > 200 ? pos - 200 : 0, 400);
        }
        const char* which = nullptr;
        if (containsAny(blk, kForbiddenOnRt,
                        sizeof(kForbiddenOnRt) / sizeof(kForbiddenOnRt[0]), &which))
        {
            std::fprintf(stderr, "D4-T1: %s reaches RT-forbidden backend '%s'\n",
                         s.label, which ? which : "?");
            return false;
        }
        std::printf("  D4-T1: %-24s RT-safe (backend-free, record-only)\n", s.label);
    }

    // S3: the hetero else-branch must record, not log.
    {
        const std::size_t pos = body.find("recordSuccessObserved(3,");
        if (pos == std::string::npos)
        {
            std::fprintf(stderr, "D4-T1: S3 hetero-skipped has no recordSuccessObserved replacement\n");
            return false;
        }
        std::string window = body.substr(pos > 200 ? pos - 200 : 0, 400);
        const char* which = nullptr;
        if (containsAny(window, kForbiddenOnRt,
                        sizeof(kForbiddenOnRt) / sizeof(kForbiddenOnRt[0]), &which))
        {
            std::fprintf(stderr, "D4-T1: S3 hetero-skipped reaches RT-forbidden backend '%s'\n",
                         which ? which : "?");
            return false;
        }
        std::printf("  D4-T1: %-24s RT-safe (backend-free, record-only)\n", "S3 hetero skipped");
    }

    // All three record calls exist and sit outside any diagnostics guard.
    const std::vector<bool> guarded = markDiagnosticsGuarded(src);
    std::size_t bodyStart = 0;
    {
        std::string tmp;
        std::size_t bs = 0;
        if (!extractFunctionBody(src, "bool AudioEngine::applyMmcssPriority()", tmp, &bs))
        {
            std::fprintf(stderr, "D4-T1: cannot locate applyMmcssPriority for guard mapping\n");
            return false;
        }
        bodyStart = bs;
    }
    static const char* const kRecordCalls[] = {
        "recordSuccessObserved(1,", "recordSuccessObserved(2,", "recordSuccessObserved(3,"
    };
    for (const char* rc : kRecordCalls)
    {
        const std::size_t pos = body.find(rc);
        if (pos == std::string::npos)
        {
            std::fprintf(stderr, "D4-T1: missing record call %s\n", rc);
            return false;
        }
        if (guarded[bodyStart + pos])
        {
            std::fprintf(stderr, "D4-T1: %s is inside a diagnostics guard\n", rc);
            return false;
        }
    }

    std::printf("STG11D4SuccessObservationTests: D4-T1 PASS (success sites backend-free, ON/OFF independent)\n");
    return true;
}

// =============================================================================
// D4-T5 (structural): the recorder body has no backend, no allocation,
// no raw atomic; it uses convo:: wrappers in payload -> count -> observed order.
// =============================================================================
static bool checkD4T5RecorderIsRtSafeAndOrdered()
{
    std::string src;
    if (!readProductionSource("src/audioengine/AudioEngine.Timer.cpp", src))
    {
        std::fprintf(stderr, "D4-T5: cannot open AudioEngine.Timer.cpp\n");
        return false;
    }
    std::string recBody;
    if (!extractFunctionBody(src, "void AudioEngine::recordSuccessObserved(", recBody))
    {
        std::fprintf(stderr, "D4-T5: cannot extract recordSuccessObserved body\n");
        return false;
    }

    const char* which = nullptr;
    if (containsAny(recBody, kForbiddenOnRt,
                    sizeof(kForbiddenOnRt) / sizeof(kForbiddenOnRt[0]), &which))
    {
        std::fprintf(stderr, "D4-T5: recorder body reaches '%s'\n", which ? which : "?");
        return false;
    }
    if (containsAny(recBody, kForbiddenRawAtomic,
                    sizeof(kForbiddenRawAtomic) / sizeof(kForbiddenRawAtomic[0]), &which))
    {
        std::fprintf(stderr, "D4-T5: recorder uses raw atomic API '%s'\n", which ? which : "?");
        return false;
    }

    const std::vector<AtomicAccess> order = atomicCallOrder(recBody);
    static const char* const kPayload[] = { "successKind_", "successA_", "successB_", "successC_" };
    const int iCount = indexOfMember(order, "successCount_");
    const int iObserved = indexOfMember(order, "successObserved_");
    if (iCount < 0 || iObserved < 0)
    {
        std::fprintf(stderr, "D4-T5: recorder must touch count and observed\n");
        return false;
    }
    for (const char* p : kPayload)
    {
        const int i = indexOfMember(order, p);
        if (i < 0)
        {
            std::fprintf(stderr, "D4-T5: recorder lost payload member %s\n", p);
            return false;
        }
        if (i > iCount)
        {
            std::fprintf(stderr, "D4-T5: payload %s is published AFTER the count increment\n", p);
            return false;
        }
    }
    if (!(iCount < iObserved))
    {
        std::fprintf(stderr, "D4-T5: observed must be the last store\n");
        return false;
    }
    if (iObserved != static_cast<int>(order.size()) - 1)
    {
        std::fprintf(stderr, "D4-T5: observed is not the last recorder access\n");
        return false;
    }
    if (wrapperOfMember(order, "successCount_") != "convo::fetchAddAtomic(")
    {
        std::fprintf(stderr, "D4-T5: count must use convo::fetchAddAtomic (acq_rel)\n");
        return false;
    }
    if (wrapperOfMember(order, "successObserved_") != "convo::publishAtomic(")
    {
        std::fprintf(stderr, "D4-T5: observed must use convo::publishAtomic (release)\n");
        return false;
    }

    // No new authority in the recorder.
    static const char* const kNewAuthority[] = {
        "std::thread", "std::jthread", "juce::Timer", "Timer::",
        "LockFreeRingBuffer", "enqueueDeferredDelete", "enqueueRetire",
        "DeferredDeletionQueue", "RuntimePublication", "Recovery"
    };
    for (const char* tok : kNewAuthority)
    {
        if (recBody.find(tok) != std::string::npos)
        {
            std::fprintf(stderr, "D4-T5: recorder creates new authority '%s'\n", tok);
            return false;
        }
    }

    std::printf("  D4-T5: recorder = payload -> count(fetchAdd/acq_rel) -> observed(publish/release)\n");
    std::printf("  D4-T5: recorder = convo:: wrappers only, no backend/allocation/authority\n");
    return true;
}

// =============================================================================
// D4-T5 (structural): read order in the NonRT report.
// =============================================================================
static bool checkD4T5ReportReadOrder()
{
    std::string src;
    if (!readProductionSource("src/audioengine/AudioEngine.Timer.cpp", src))
    {
        std::fprintf(stderr, "D4-T5r: cannot open AudioEngine.Timer.cpp\n");
        return false;
    }
    std::string repBody;
    if (!extractFunctionBody(src, "void AudioEngine::reportSuccessIfRecorded()", repBody))
    {
        std::fprintf(stderr, "D4-T5r: cannot extract reportSuccessIfRecorded body\n");
        return false;
    }

    const std::vector<AtomicAccess> order = atomicCallOrder(repBody);
    const int iCount = indexOfMember(order, "successCount_");
    if (iCount < 0)
    {
        std::fprintf(stderr, "D4-T5r: report does not read count\n");
        return false;
    }
    static const char* const kSnapshot[] = { "successKind_", "successA_", "successB_", "successC_" };
    for (const char* s : kSnapshot)
    {
        const int i = indexOfMember(order, s);
        if (i < 0)
        {
            std::fprintf(stderr, "D4-T5r: report does not read snapshot member %s\n", s);
            return false;
        }
        if (i < iCount)
        {
            std::fprintf(stderr, "D4-T5r: snapshot %s is read BEFORE the count acquire\n", s);
            return false;
        }
    }
    static const char* const kRequiredAcquire[] = {
        "successCount_", "successKind_", "successA_", "successB_", "successC_"
    };
    for (const char* m : kRequiredAcquire)
    {
        if (wrapperOfMember(order, m) != "convo::consumeAtomic(")
        {
            std::fprintf(stderr, "D4-T5r: %s must be read with convo::consumeAtomic (acquire)\n", m);
            return false;
        }
    }
    if (repBody.find("std::memory_order_acquire") == std::string::npos)
    {
        std::fprintf(stderr, "D4-T5r: report does not spell out memory_order_acquire\n");
        return false;
    }
    const char* which = nullptr;
    if (containsAny(repBody, kForbiddenRawAtomic,
                    sizeof(kForbiddenRawAtomic) / sizeof(kForbiddenRawAtomic[0]), &which))
    {
        std::fprintf(stderr, "D4-T5r: report uses raw atomic API '%s'\n", which ? which : "?");
        return false;
    }

    // The report runs from the existing execution point, no new thread.
    std::string cbBody;
    if (!extractFunctionBody(src, "void AudioEngine::timerCallback()", cbBody))
    {
        std::fprintf(stderr, "D4-T5r: cannot extract timerCallback body\n");
        return false;
    }
    if (cbBody.find("reportSuccessIfRecorded();") == std::string::npos)
    {
        std::fprintf(stderr, "D4-T5r: reportSuccessIfRecorded not called from timerCallback\n");
        return false;
    }

    std::printf("  D4-T5: report order = observed(fast-out) -> count(acquire) -> snapshot(acquire)\n");
    std::printf("  D4-T5: report runs from existing timerCallback (no new authority)\n");
    return true;
}

// =============================================================================
// D4-T2 / D4-T3 (structural): the NonRT report keeps the original wording,
// under the diagnostics guard.
// =============================================================================
static bool checkD4T2T3WordingPreserved()
{
    std::string src;
    if (!readProductionSource("src/audioengine/AudioEngine.Timer.cpp", src))
    {
        std::fprintf(stderr, "D4-T2: cannot open AudioEngine.Timer.cpp\n");
        return false;
    }
    std::string repBody;
    if (!extractFunctionBody(src, "void AudioEngine::reportSuccessIfRecorded()", repBody))
    {
        std::fprintf(stderr, "D4-T2: cannot extract reportSuccessIfRecorded body\n");
        return false;
    }
    static const char* const kWording[] = {
        "[NATIVE_RT] applied: win32Prio=",
        "[AFFINITY] AudioThread pinned mask=0x",
        "[AFFINITY] P/E cores: AudioThread affinity skipped"
    };
    for (const char* w : kWording)
    {
        if (repBody.find(w) == std::string::npos)
        {
            std::fprintf(stderr, "D4-T2: report lost original wording '%s'\n", w);
            return false;
        }
    }
    // toHexString formatting for the mask payload stays on the NonRT side.
    if (repBody.find("toHexString") == std::string::npos)
    {
        std::fprintf(stderr, "D4-T2: report lost hex formatting for mask payload\n");
        return false;
    }
    // The emission must be diagnostics-guarded (OFF semantics preserved).
    const std::vector<bool> guarded = markDiagnosticsGuarded(src);
    for (const char* w : kWording)
    {
        const std::size_t pos = src.find(w);
        if (pos == std::string::npos || !guarded[pos])
        {
            std::fprintf(stderr, "D4-T2: wording '%s' is not diagnostics-guarded\n", w);
            return false;
        }
    }
    if (repBody.find("diagLog(") == std::string::npos)
    {
        std::fprintf(stderr, "D4-T2: report lost its diagLog backend\n");
        return false;
    }

    std::printf("STG11D4SuccessObservationTests: D4-T2/D4-T3 PASS (original wording kept, guarded, NonRT)\n");
    return true;
}

// =============================================================================
// D4-T4 (structural): Diagnostics OFF semantics unbroken.
//   Every diagLog / juce::String left in applyMmcssPriority must be guarded,
//   and the function must take no lock directly.
// =============================================================================
static bool checkD4T4DiagnosticsOffSemantics()
{
    std::string src;
    if (!readProductionSource("src/audioengine/AudioEngine.Timer.cpp", src))
    {
        std::fprintf(stderr, "D4-T4: cannot open AudioEngine.Timer.cpp\n");
        return false;
    }
    std::string body;
    std::size_t bodyStart = 0;
    if (!extractFunctionBody(src, "bool AudioEngine::applyMmcssPriority()", body, &bodyStart))
    {
        std::fprintf(stderr, "D4-T4: cannot extract applyMmcssPriority body\n");
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
                std::fprintf(stderr, "D4-T4: '%s' reachable with Diagnostics OFF\n", tok);
                return false;
            }
        }
    }
    static const char* const kDirectLocks[] = { "std::mutex", "lock_guard", "s_logMutex", "asyncSink" };
    const char* which = nullptr;
    if (containsAny(body, kDirectLocks, 4, &which))
    {
        std::fprintf(stderr, "D4-T4: applyMmcssPriority directly touches '%s'\n",
                     which ? which : "?");
        return false;
    }
    std::printf("STG11D4SuccessObservationTests: D4-T4 PASS (Diagnostics OFF semantics kept)\n");
    return true;
}

// =============================================================================
// D4-T2/T4 functional + D4 ordering contract (deterministic):
//   record -> observed acquire -> payload visibility; count monotonic;
//   snapshot last-wins; an older snapshot is never confirmed as a newer
//   count's snapshot. Five rounds across the three kinds.
// =============================================================================
static bool checkD4RecordThenVisible()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-11-D4] harness start failed\n");
        return false;
    }
    h.stop();
    AudioEngine& e = h.engine();

    if (Access::successObserved(e) || Access::successCount(e) != 0)
    {
        std::fprintf(stderr, "D4: initial state not clean\n");
        return false;
    }
    Access::reportSuccess(e);
    if (Access::successReportedCount(e) != 0)
    {
        std::fprintf(stderr, "D4: report without record changed state\n");
        return false;
    }

    // kind cycles 1 -> 2 -> 3 -> 1 -> 2 with distinct payloads, so every round
    // asserts the exact kind/payload mapping (last-wins, never an older one).
    for (int k = 1; k <= 5; ++k)
    {
        const std::uint32_t kind = static_cast<std::uint32_t>((k - 1) % 3 + 1);
        const std::uint64_t a = 0x1000u + static_cast<std::uint64_t>(k) * 0x11u;
        const std::uint64_t b = 0x2000u + static_cast<std::uint64_t>(k) * 0x23u;
        const std::uint64_t c = (kind == 1) ? (0x3000u + static_cast<std::uint64_t>(k)) : 0u;
        Access::recordSuccessObserved(e, kind, a, b, c);

        if (!Access::successObserved(e))
        {
            std::fprintf(stderr, "D4: round %d did not set observed\n", k);
            return false;
        }
        if (Access::successCount(e) != static_cast<std::uint64_t>(k))
        {
            std::fprintf(stderr, "D4: count is not monotonic at round %d\n", k);
            return false;
        }
        // count == k implies record #k's payload is visible.
        if (Access::successKind(e) != kind
            || Access::successA(e) != a
            || Access::successB(e) != b
            || Access::successC(e) != c)
        {
            std::fprintf(stderr, "D4: round %d snapshot is stale\n", k);
            return false;
        }

        const std::uint64_t before = Access::successReportedCount(e);
        Access::reportSuccess(e);
        if (Access::successReportedCount(e) != before + 1)
        {
            std::fprintf(stderr, "D4: round %d report did not advance by exactly one\n", k);
            return false;
        }
    }

    const std::uint64_t settled = Access::successReportedCount(e);
    Access::reportSuccess(e);
    if (Access::successReportedCount(e) != settled)
    {
        std::fprintf(stderr, "D4: report advanced without a new record\n");
        return false;
    }

    std::printf("STG11D4SuccessObservationTests: D4 functional PASS (record + once-per-count NonRT diagnosis)\n");
    return true;
}

// =============================================================================
// D4-T6 (structural, read-only): the D3 failure transport is untouched.
//   Publication order payload -> count -> observed and the reader order still
//   hold. This guards the D3 authority boundary without modifying D3 tests.
// =============================================================================
static bool checkD4T6D3BoundaryIntact()
{
    std::string src;
    if (!readProductionSource("src/audioengine/AudioEngine.Timer.cpp", src))
    {
        std::fprintf(stderr, "D4-T6: cannot open AudioEngine.Timer.cpp\n");
        return false;
    }
    std::string recBody;
    if (!extractFunctionBody(src, "void AudioEngine::recordAffinityFailure(", recBody))
    {
        std::fprintf(stderr, "D4-T6: D3 recorder missing (boundary violated)\n");
        return false;
    }
    const std::vector<AtomicAccess> order = atomicCallOrder(recBody);
    const int iCount = indexOfMember(order, "affinityFailureCount_");
    const int iObserved = indexOfMember(order, "affinityFailureObserved_");
    static const char* const kPayload[] = {
        "affinityFailureLastKind_", "affinityFailureLastMask_", "affinityFailureLastError_"
    };
    if (iCount < 0 || iObserved < 0 || !(iCount < iObserved))
    {
        std::fprintf(stderr, "D4-T6: D3 recorder order broken\n");
        return false;
    }
    for (const char* p : kPayload)
    {
        const int i = indexOfMember(order, p);
        if (i < 0 || i > iCount)
        {
            std::fprintf(stderr, "D4-T6: D3 payload %s misplaced\n", p);
            return false;
        }
    }
    if (wrapperOfMember(order, "affinityFailureCount_") != "convo::fetchAddAtomic(")
    {
        std::fprintf(stderr, "D4-T6: D3 count wrapper changed\n");
        return false;
    }

    std::string repBody;
    if (!extractFunctionBody(src, "void AudioEngine::reportAffinityFailureIfRecorded()", repBody))
    {
        std::fprintf(stderr, "D4-T6: D3 report missing (boundary violated)\n");
        return false;
    }
    const std::vector<AtomicAccess> rorder = atomicCallOrder(repBody);
    const int riCount = indexOfMember(rorder, "affinityFailureCount_");
    if (riCount < 0)
    {
        std::fprintf(stderr, "D4-T6: D3 report does not read count\n");
        return false;
    }
    for (const char* p : kPayload)
    {
        const int i = indexOfMember(rorder, p);
        if (i < 0 || i < riCount)
        {
            std::fprintf(stderr, "D4-T6: D3 report snapshot %s misplaced\n", p);
            return false;
        }
    }

    std::printf("STG11D4SuccessObservationTests: D4-T6 PASS (D3 failure transport untouched)\n");
    return true;
}

// Entry point called from the harness main (PublishPipelineIntegrationTests.cpp).
// No new CTest registration: this is a sub-test of the AudioEngineHarness
// executable, matching the STG-8, STG-9, STG-11-D3 shape.
int runSTG11D4SuccessObservationTests()
{
    bool ok = true;
    if (!checkD4T1SuccessSitesAreRtSafe())
    {
        std::fprintf(stderr, "FAIL: D4-T1 success sites backend-free\n");
        ok = false;
    }
    if (!checkD4T5RecorderIsRtSafeAndOrdered())
    {
        std::fprintf(stderr, "FAIL: D4-T5 recorder RT-safe and ordered\n");
        ok = false;
    }
    if (!checkD4T5ReportReadOrder())
    {
        std::fprintf(stderr, "FAIL: D4-T5 report read order\n");
        ok = false;
    }
    if (!checkD4T2T3WordingPreserved())
    {
        std::fprintf(stderr, "FAIL: D4-T2/D4-T3 wording preserved\n");
        ok = false;
    }
    if (!checkD4T4DiagnosticsOffSemantics())
    {
        std::fprintf(stderr, "FAIL: D4-T4 diagnostics-OFF semantics\n");
        ok = false;
    }
    if (!checkD4RecordThenVisible())
    {
        std::fprintf(stderr, "FAIL: D4 record then visible\n");
        ok = false;
    }
    if (!checkD4T6D3BoundaryIntact())
    {
        std::fprintf(stderr, "FAIL: D4-T6 D3 boundary intact\n");
        ok = false;
    }
    if (ok)
        std::printf("STG11D4SuccessObservationTests: PASS (D4-T1/D4-T2/D4-T3/D4-T4/D4-T5/D4-T6)\n");
    return ok ? 0 : 1;
}
