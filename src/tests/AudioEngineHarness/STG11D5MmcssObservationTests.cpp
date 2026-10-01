// STG11D5MmcssObservationTests.cpp - STG-11-D5 regression (D5-T1 .. D5-T8).
//
// Target defect (STG-11-D5 = OBS-D4-1 / OBS-D4-2):
//   The Diagnostics ON MMCSS logging sites in AudioEngine.Mmcss.cpp executed
//   diagLog() directly on the RT (audio thread):
//     M1  [MMCSS-*] registered (primary success)
//     M2  [MMCSS-*] already registered by JUCE/driver (expected path)
//     M3  [MMCSS-*] registered (fallback)
//     M4  [MMCSS-*] FAILED (all paths exhausted)
//     M5  [MMCSS] reverted on Audio Thread (shutdown path)
//   Mmcss.cpp's file-local diagLog runs DBG + juce::Logger::writeToLog with
//   juce::String construction on the caller thread.
//
//   Fix (same idea as D3 Candidate A / D4, adapted - not mechanically copied):
//   the RT side performs a lock-free atomic record only
//   (recordMmcssEventObserved, convo:: wrappers), and the existing NonRT
//   execution point (timerCallback) diagnoses it through the existing
//   Mmcss.cpp diagLog backend (reportMmcssEventIfRecorded).
//   The MMCSS OS calls themselves (AvSetMmThreadCharacteristicsW /
//   AvSetMmThreadPriority / AvRevertMmThreadCharacteristics) stay on the
//   owning audio thread by architectural necessity (calling-thread
//   registration per Microsoft Learn + driver-owned ASIO threads +
//   same-thread revert requirement). See the Contract Audit for the
//   safe design contract (C-D5-OS1..4). Registration / revert / fallback /
//   return-value semantics are unchanged.
//
// Test contract (Owner-specified):
//   D5-T1  RT MMCSS success/failure logging backend-free
//   D5-T2  RT revert logging backend-free
//   D5-T3  Diagnostics ON/OFF both RT-safe
//   D5-T4  thread-affinity / lifetime contract kept
//   D5-T5  failure / return / fallback semantics unchanged
//   D5-T6  D3/D4 observation ordering unchanged (read-only)
//   D5-T7  D1/D2/D3/D4 regression (via existing suites, Gate)
//   D5-T8  negative control (old RT diagLog restored -> FAIL)
//   D5-T9  Debug full CTest (Gate) / D5-T10 Release full CTest (Gate)
//   D5-T11 raw atomic audit / D5-T12 RT lock audit / D5-T13 authority audit (Gate)
//
// Verification method:
//   - structural: reads the real Mmcss.cpp source, extracts the registration
//     and revert function bodies by brace matching, and asserts that no
//     backend token occurs in RT-reachable code while the NonRT report keeps
//     the original wording under the diagnostics guard.
//   - functional: drives both ends of the transport with synthetic values.
//   - D3/D4 files, functions, members, tests and orderings are untouched;
//     D5-T6 asserts both still hold (read-only).
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

struct AtomicAccess { std::string wrapper; std::string member; };

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
// D5-T1 / D5-T2 (structural core): M1-M5 reach no backend on RT.
// The old wordings must not occur in the RT functions at all; the record
// calls must exist outside any diagnostics guard.
// =============================================================================
static bool checkD5T1MmcssSitesAreRtSafe()
{
    std::string src;
    if (!readProductionSource("src/audioengine/AudioEngine.Mmcss.cpp", src))
    {
        std::fprintf(stderr, "D5-T1: cannot open AudioEngine.Mmcss.cpp\n");
        return false;
    }
    std::string regBody;
    if (!extractFunctionBody(src, "bool AudioEngine::tryApplyMmcssForSelfManagedThread()", regBody))
    {
        std::fprintf(stderr, "D5-T1: cannot extract tryApplyMmcssForSelfManagedThread body\n");
        return false;
    }
    std::string revBody;
    if (!extractFunctionBody(src, "void AudioEngine::revertMmcssOnAudioThread()", revBody))
    {
        std::fprintf(stderr, "D5-T1: cannot extract revertMmcssOnAudioThread body\n");
        return false;
    }

    // The old wordings must NOT occur in the RT functions at all.
    static const char* const kOldWording[] = {
        "] registered: task=",
        "] already registered by JUCE/driver",
        "] registered (fallback): task=",
        "] FAILED: primary err=",
        "[MMCSS] reverted on Audio Thread"
    };
    for (const char* w : kOldWording)
    {
        if (regBody.find(w) != std::string::npos || revBody.find(w) != std::string::npos)
        {
            std::fprintf(stderr, "D5-T1: old wording '%s' still inside RT function\n", w);
            return false;
        }
    }

    // No backend token in either RT-reachable body.
    const char* which = nullptr;
    if (containsAny(regBody, kForbiddenOnRt,
                    sizeof(kForbiddenOnRt) / sizeof(kForbiddenOnRt[0]), &which))
    {
        std::fprintf(stderr, "D5-T1: registration path reaches RT-forbidden backend '%s'\n",
                     which ? which : "?");
        return false;
    }
    if (containsAny(revBody, kForbiddenOnRt,
                    sizeof(kForbiddenOnRt) / sizeof(kForbiddenOnRt[0]), &which))
    {
        std::fprintf(stderr, "D5-T2: revert path reaches RT-forbidden backend '%s'\n",
                     which ? which : "?");
        return false;
    }
    std::printf("  D5-T1: M1 registered                 RT-safe (backend-free, record-only)\n");
    std::printf("  D5-T1: M2 already-registered         RT-safe (backend-free, record-only)\n");
    std::printf("  D5-T1: M3 fallback                   RT-safe (backend-free, record-only)\n");
    std::printf("  D5-T1: M4 FAILED                     RT-safe (backend-free, record-only)\n");
    std::printf("  D5-T2: M5 reverted                   RT-safe (backend-free, record-only)\n");

    // All five record calls exist outside any diagnostics guard.
    const std::vector<bool> guarded = markDiagnosticsGuarded(src);
    static const char* const kRecordCalls[] = {
        "recordMmcssEventObserved(1,", "recordMmcssEventObserved(2,",
        "recordMmcssEventObserved(3,", "recordMmcssEventObserved(4,",
        "recordMmcssEventObserved(5,"
    };
    for (const char* rc : kRecordCalls)
    {
        const std::size_t pos = src.find(rc);
        if (pos == std::string::npos)
        {
            std::fprintf(stderr, "D5-T1: missing record call %s\n", rc);
            return false;
        }
        if (guarded[pos])
        {
            std::fprintf(stderr, "D5-T1: %s is inside a diagnostics guard\n", rc);
            return false;
        }
    }

    std::printf("STG11D5MmcssObservationTests: D5-T1/D5-T2 PASS (MMCSS sites backend-free, ON/OFF independent)\n");
    return true;
}

// =============================================================================
// D5 recorder / report ordering (structural, D3/D4 contract reused).
// =============================================================================
static bool checkD5RecorderIsRtSafeAndOrdered()
{
    std::string src;
    if (!readProductionSource("src/audioengine/AudioEngine.Mmcss.cpp", src))
    {
        std::fprintf(stderr, "D5: cannot open AudioEngine.Mmcss.cpp\n");
        return false;
    }
    std::string recBody;
    if (!extractFunctionBody(src, "void AudioEngine::recordMmcssEventObserved(", recBody))
    {
        std::fprintf(stderr, "D5: cannot extract recordMmcssEventObserved body\n");
        return false;
    }

    const char* which = nullptr;
    if (containsAny(recBody, kForbiddenOnRt,
                    sizeof(kForbiddenOnRt) / sizeof(kForbiddenOnRt[0]), &which))
    {
        std::fprintf(stderr, "D5: recorder body reaches '%s'\n", which ? which : "?");
        return false;
    }
    if (containsAny(recBody, kForbiddenRawAtomic,
                    sizeof(kForbiddenRawAtomic) / sizeof(kForbiddenRawAtomic[0]), &which))
    {
        std::fprintf(stderr, "D5: recorder uses raw atomic API '%s'\n", which ? which : "?");
        return false;
    }

    const std::vector<AtomicAccess> order = atomicCallOrder(recBody);
    static const char* const kPayload[] = { "mmcssKind_", "mmcssA_", "mmcssB_", "mmcssC_", "mmcssD_" };
    const int iCount = indexOfMember(order, "mmcssCount_");
    const int iObserved = indexOfMember(order, "mmcssObserved_");
    if (iCount < 0 || iObserved < 0)
    {
        std::fprintf(stderr, "D5: recorder must touch count and observed\n");
        return false;
    }
    for (const char* p : kPayload)
    {
        const int i = indexOfMember(order, p);
        if (i < 0)
        {
            std::fprintf(stderr, "D5: recorder lost payload member %s\n", p);
            return false;
        }
        if (i > iCount)
        {
            std::fprintf(stderr, "D5: payload %s is published AFTER the count increment\n", p);
            return false;
        }
    }
    if (!(iCount < iObserved) || iObserved != static_cast<int>(order.size()) - 1)
    {
        std::fprintf(stderr, "D5: observed must be the last recorder access\n");
        return false;
    }
    if (wrapperOfMember(order, "mmcssCount_") != "convo::fetchAddAtomic(")
    {
        std::fprintf(stderr, "D5: count must use convo::fetchAddAtomic (acq_rel)\n");
        return false;
    }
    if (wrapperOfMember(order, "mmcssObserved_") != "convo::publishAtomic(")
    {
        std::fprintf(stderr, "D5: observed must use convo::publishAtomic (release)\n");
        return false;
    }

    static const char* const kNewAuthority[] = {
        "std::thread", "std::jthread", "juce::Timer", "Timer::",
        "LockFreeRingBuffer", "enqueueDeferredDelete", "enqueueRetire",
        "DeferredDeletionQueue", "RuntimePublication", "Recovery"
    };
    for (const char* tok : kNewAuthority)
    {
        if (recBody.find(tok) != std::string::npos)
        {
            std::fprintf(stderr, "D5: recorder creates new authority '%s'\n", tok);
            return false;
        }
    }

    std::printf("  D5: recorder = payload -> count(fetchAdd/acq_rel) -> observed(publish/release)\n");
    return true;
}

static bool checkD5ReportReadOrder()
{
    std::string src;
    if (!readProductionSource("src/audioengine/AudioEngine.Mmcss.cpp", src))
    {
        std::fprintf(stderr, "D5r: cannot open AudioEngine.Mmcss.cpp\n");
        return false;
    }
    std::string repBody;
    if (!extractFunctionBody(src, "void AudioEngine::reportMmcssEventIfRecorded()", repBody))
    {
        std::fprintf(stderr, "D5r: cannot extract reportMmcssEventIfRecorded body\n");
        return false;
    }

    const std::vector<AtomicAccess> order = atomicCallOrder(repBody);
    const int iCount = indexOfMember(order, "mmcssCount_");
    if (iCount < 0)
    {
        std::fprintf(stderr, "D5r: report does not read count\n");
        return false;
    }
    static const char* const kSnapshot[] = { "mmcssKind_", "mmcssA_", "mmcssB_", "mmcssC_", "mmcssD_" };
    for (const char* s : kSnapshot)
    {
        const int i = indexOfMember(order, s);
        if (i < 0 || i < iCount)
        {
            std::fprintf(stderr, "D5r: snapshot %s misplaced (must follow count acquire)\n", s);
            return false;
        }
        if (wrapperOfMember(order, s) != "convo::consumeAtomic(")
        {
            std::fprintf(stderr, "D5r: %s must be read with convo::consumeAtomic (acquire)\n", s);
            return false;
        }
    }
    if (wrapperOfMember(order, "mmcssCount_") != "convo::consumeAtomic(")
    {
        std::fprintf(stderr, "D5r: count must be read with convo::consumeAtomic (acquire)\n");
        return false;
    }
    if (repBody.find("std::memory_order_acquire") == std::string::npos)
    {
        std::fprintf(stderr, "D5r: report does not spell out memory_order_acquire\n");
        return false;
    }
    const char* which = nullptr;
    if (containsAny(repBody, kForbiddenRawAtomic,
                    sizeof(kForbiddenRawAtomic) / sizeof(kForbiddenRawAtomic[0]), &which))
    {
        std::fprintf(stderr, "D5r: report uses raw atomic API '%s'\n", which ? which : "?");
        return false;
    }

    // The report runs from the existing execution point, no new thread.
    std::string timerSrc;
    if (!readProductionSource("src/audioengine/AudioEngine.Timer.cpp", timerSrc))
    {
        std::fprintf(stderr, "D5r: cannot open AudioEngine.Timer.cpp\n");
        return false;
    }
    std::string cbBody;
    if (!extractFunctionBody(timerSrc, "void AudioEngine::timerCallback()", cbBody))
    {
        std::fprintf(stderr, "D5r: cannot extract timerCallback body\n");
        return false;
    }
    if (cbBody.find("reportMmcssEventIfRecorded();") == std::string::npos)
    {
        std::fprintf(stderr, "D5r: reportMmcssEventIfRecorded not called from timerCallback\n");
        return false;
    }

    std::printf("  D5: report order = observed(fast-out) -> count(acquire) -> snapshot(acquire)\n");
    std::printf("  D5: report runs from existing timerCallback (no new authority)\n");
    return true;
}

// =============================================================================
// D5-T3 wording + guard: the NonRT report keeps the original MMCSS wording,
// under the diagnostics guard (OFF semantics preserved).
// =============================================================================
static bool checkD5WordingPreserved()
{
    std::string src;
    if (!readProductionSource("src/audioengine/AudioEngine.Mmcss.cpp", src))
    {
        std::fprintf(stderr, "D5-T3: cannot open AudioEngine.Mmcss.cpp\n");
        return false;
    }
    std::string repBody;
    if (!extractFunctionBody(src, "void AudioEngine::reportMmcssEventIfRecorded()", repBody))
    {
        std::fprintf(stderr, "D5-T3: cannot extract reportMmcssEventIfRecorded body\n");
        return false;
    }
    static const char* const kWording[] = {
        "] registered: task=",
        "] already registered by JUCE/driver",
        "] registered (fallback): task=",
        "] FAILED: primary err=",
        "[MMCSS] reverted on Audio Thread"
    };
    for (const char* w : kWording)
    {
        if (repBody.find(w) == std::string::npos)
        {
            std::fprintf(stderr, "D5-T3: report lost original wording '%s'\n", w);
            return false;
        }
    }
    if (repBody.find("diagLog(") == std::string::npos)
    {
        std::fprintf(stderr, "D5-T3: report lost its diagLog backend\n");
        return false;
    }
    const std::vector<bool> guarded = markDiagnosticsGuarded(src);
    for (const char* w : kWording)
    {
        const std::size_t pos = src.find(w);
        if (pos == std::string::npos || !guarded[pos])
        {
            std::fprintf(stderr, "D5-T3: wording '%s' is not diagnostics-guarded\n", w);
            return false;
        }
    }

    // OFF semantics: with the flag off, the RT bodies take no lock directly,
    // and neither RT nor report body touches the sink or mutex itself.
    // (The report calls diagLog; asyncSink lives inside diagLog's callee,
    // never as text in these bodies.)
    std::string regBody;
    if (!extractFunctionBody(src, "bool AudioEngine::tryApplyMmcssForSelfManagedThread()", regBody))
    {
        std::fprintf(stderr, "D5-T3: cannot extract registration body\n");
        return false;
    }
    static const char* const kDirectLocks[] = { "std::mutex", "lock_guard", "s_logMutex", "asyncSink" };
    const char* which = nullptr;
    if (containsAny(regBody, kDirectLocks, 4, &which))
    {
        std::fprintf(stderr, "D5-T3: registration body directly touches '%s'\n",
                     which ? which : "?");
        return false;
    }
    if (containsAny(repBody, kDirectLocks, 4, &which))
    {
        std::fprintf(stderr, "D5-T3: report body directly touches '%s'\n",
                     which ? which : "?");
        return false;
    }

    std::printf("STG11D5MmcssObservationTests: D5-T3 PASS (wording kept, guarded, OFF safe)\n");
    return true;
}

// =============================================================================
// D5-T4 (structural): thread-affinity / lifetime contract kept.
//   (a) t_mmcssHandle / t_mmcssTaskIndex / t_mmcssTried stay thread_local.
//   (b) AvRevert stays in revertMmcssOnAudioThread (same thread).
//   (c) Message Thread path only sets the flag (no Av* on NonRT).
//   (d) No AvRevert / AvSetMm introduced outside the two audio-thread functions.
// =============================================================================
static bool checkD5T4ThreadContract()
{
    std::string src;
    if (!readProductionSource("src/audioengine/AudioEngine.Mmcss.cpp", src))
    {
        std::fprintf(stderr, "D5-T4: cannot open AudioEngine.Mmcss.cpp\n");
        return false;
    }
    if (src.find("thread_local HANDLE t_mmcssHandle") == std::string::npos
        || src.find("thread_local DWORD  t_mmcssTaskIndex") == std::string::npos
        || src.find("thread_local bool   t_mmcssTried") == std::string::npos)
    {
        std::fprintf(stderr, "D5-T4: thread_local MMCSS state changed\n");
        return false;
    }
    std::string revBody;
    if (!extractFunctionBody(src, "void AudioEngine::revertMmcssOnAudioThread()", revBody))
    {
        std::fprintf(stderr, "D5-T4: cannot extract revert body\n");
        return false;
    }
    if (revBody.find("::AvRevertMmThreadCharacteristics(t_mmcssHandle)") == std::string::npos)
    {
        std::fprintf(stderr, "D5-T4: same-thread AvRevert moved out of revertMmcssOnAudioThread\n");
        return false;
    }
    if (revBody.find("t_mmcssTried = false") == std::string::npos)
    {
        std::fprintf(stderr, "D5-T4: re-registration path (t_mmcssTried reset) changed\n");
        return false;
    }
    // No Av* calls anywhere else in this TU (single registration + single revert).
    int avSetCount = 0, avRevertCount = 0, avPrioCount = 0;
    for (std::size_t p = src.find("AvSetMmThreadCharacteristicsW"); p != std::string::npos;
         p = src.find("AvSetMmThreadCharacteristicsW", p + 1))
        ++avSetCount;
    for (std::size_t p = src.find("AvRevertMmThreadCharacteristics"); p != std::string::npos;
         p = src.find("AvRevertMmThreadCharacteristics", p + 1))
        ++avRevertCount;
    for (std::size_t p = src.find("AvSetMmThreadPriority"); p != std::string::npos;
         p = src.find("AvSetMmThreadPriority", p + 1))
        ++avPrioCount;
    // All AvSetMmThreadCharacteristicsW calls go through the tryTask wrapper,
    // so the TU holds exactly one definition-side mention; AvSetMmThreadPriority
    // appears at the primary + fallback sites; AvRevert appears once (revert fn).
    // Comments are stripped first so prose cannot satisfy the count.
    const std::string code = stripComments(src);
    auto countIn = [&code](const char* tok) {
        int n = 0;
        for (std::size_t p = code.find(tok); p != std::string::npos; p = code.find(tok, p + 1))
            ++n;
        return n;
    };
    if (countIn("AvSetMmThreadCharacteristicsW") != 1
        || countIn("AvRevertMmThreadCharacteristics") != 1
        || countIn("AvSetMmThreadPriority") != 2)
    {
        std::fprintf(stderr, "D5-T4: Av* call sites changed (set=%d revert=%d prio=%d)\n",
                     countIn("AvSetMmThreadCharacteristicsW"),
                     countIn("AvRevertMmThreadCharacteristics"),
                     countIn("AvSetMmThreadPriority"));
        return false;
    }
    (void)avSetCount; (void)avRevertCount; (void)avPrioCount;

    // Message Thread side still only sets the flag.
    std::string relSrc;
    if (!readProductionSource("src/audioengine/AudioEngine.Processing.ReleaseResources.cpp", relSrc))
    {
        std::fprintf(stderr, "D5-T4: cannot open ReleaseResources.cpp\n");
        return false;
    }
    if (relSrc.find("mmcssShutdownRequested, true") == std::string::npos)
    {
        std::fprintf(stderr, "D5-T4: Message-Thread flag path changed\n");
        return false;
    }
    if (relSrc.find("AvRevertMmThreadCharacteristics") != std::string::npos
        || relSrc.find("AvSetMmThreadCharacteristics") != std::string::npos)
    {
        std::fprintf(stderr, "D5-T4: Av* introduced on the Message Thread path\n");
        return false;
    }

    std::printf("STG11D5MmcssObservationTests: D5-T4 PASS (thread-affinity/lifetime contract kept)\n");
    return true;
}

// =============================================================================
// D5-T5 (structural): failure / return / fallback semantics unchanged.
// =============================================================================
static bool checkD5T5SemanticsUnchanged()
{
    std::string src;
    if (!readProductionSource("src/audioengine/AudioEngine.Mmcss.cpp", src))
    {
        std::fprintf(stderr, "D5-T5: cannot open AudioEngine.Mmcss.cpp\n");
        return false;
    }
    std::string regBody;
    if (!extractFunctionBody(src, "bool AudioEngine::tryApplyMmcssForSelfManagedThread()", regBody))
    {
        std::fprintf(stderr, "D5-T5: cannot extract registration body\n");
        return false;
    }
    // t_mmcssTried once-guard intact.
    if (regBody.find("if (t_mmcssTried)") == std::string::npos
        || regBody.find("t_mmcssTried = true") == std::string::npos)
    {
        std::fprintf(stderr, "D5-T5: once-per-thread guard changed\n");
        return false;
    }
    // applyMmcssPriority still unconditional on the first pass.
    if (regBody.find("applyMmcssPriority();") == std::string::npos)
    {
        std::fprintf(stderr, "D5-T5: applyMmcssPriority call moved or removed\n");
        return false;
    }
    // Error-code classification intact: 5/183/1552 -> true.
    if (regBody.find("ERROR_ACCESS_DENIED") == std::string::npos
        || regBody.find("ERROR_ALREADY_EXISTS") == std::string::npos
        || regBody.find("ERROR_NO_MORE_ITEMS") == std::string::npos)
    {
        std::fprintf(stderr, "D5-T5: already-registered error classification changed\n");
        return false;
    }
    // Fallback chain intact: 1531 -> fallback1 -> fallback2, in order.
    const std::size_t fb1 = regBody.find("attemptFallback(fallback1, 1u)");
    const std::size_t fb2 = regBody.find("attemptFallback(fallback2, 2u)");
    if (fb1 == std::string::npos || fb2 == std::string::npos || !(fb1 < fb2))
    {
        std::fprintf(stderr, "D5-T5: fallback chain order changed\n");
        return false;
    }
    if (regBody.find("ERROR_INVALID_TASK_NAME") == std::string::npos)
    {
        std::fprintf(stderr, "D5-T5: fallback trigger changed\n");
        return false;
    }
    // Return-literal sites, verified against the real source:
    //   true x6 = JuceManaged/None + primary + already + fallback-lambda +
    //             fallback1 call-site + fallback2 call-site.
    //   false x4 = NativeRT-passthrough + lambda null-guard +
    //              lambda exhausted + all-exhausted.
    //   (The t_mmcssTried early-out returns a handle comparison, counted separately
    //   by the once-guard check above.)
    int retTrue = 0, retFalse = 0;
    for (std::size_t p = regBody.find("return true"); p != std::string::npos;
         p = regBody.find("return true", p + 1))
        ++retTrue;
    for (std::size_t p = regBody.find("return false"); p != std::string::npos;
         p = regBody.find("return false", p + 1))
        ++retFalse;
    if (retTrue != 6 || retFalse != 4)
    {
        std::fprintf(stderr, "D5-T5: return-value sites changed (true=%d false=%d)\n", retTrue, retFalse);
        return false;
    }

    std::printf("STG11D5MmcssObservationTests: D5-T5 PASS (semantics unchanged)\n");
    return true;
}

// =============================================================================
// D5 functional (deterministic): record -> observed -> payload visibility,
// count monotonic, snapshot last-wins across all five kinds.
// =============================================================================
static bool checkD5RecordThenVisible()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-11-D5] harness start failed\n");
        return false;
    }
    h.stop();
    AudioEngine& e = h.engine();

    if (Access::mmcssObserved(e) || Access::mmcssCount(e) != 0)
    {
        std::fprintf(stderr, "D5: initial state not clean\n");
        return false;
    }
    Access::reportMmcssEvent(e);
    if (Access::mmcssReportedCount(e) != 0)
    {
        std::fprintf(stderr, "D5: report without record changed state\n");
        return false;
    }

    // Five rounds, one per kind, with distinct payloads.
    for (int k = 1; k <= 5; ++k)
    {
        const std::uint32_t kind = static_cast<std::uint32_t>(k);
        const std::uint64_t a = static_cast<std::uint64_t>((k - 1) % 2); // policy alternates
        const std::uint64_t b = static_cast<std::uint64_t>((k - 1) % 3); // task selector
        const std::uint64_t c = 100u + static_cast<std::uint64_t>(k);
        const std::uint64_t d = 200u + static_cast<std::uint64_t>(k);
        Access::recordMmcssEvent(e, kind, a, b, c, d);

        if (!Access::mmcssObserved(e))
        {
            std::fprintf(stderr, "D5: round %d did not set observed\n", k);
            return false;
        }
        if (Access::mmcssCount(e) != static_cast<std::uint64_t>(k))
        {
            std::fprintf(stderr, "D5: count is not monotonic at round %d\n", k);
            return false;
        }
        if (Access::mmcssKind(e) != kind || Access::mmcssA(e) != a
            || Access::mmcssB(e) != b || Access::mmcssC(e) != c || Access::mmcssD(e) != d)
        {
            std::fprintf(stderr, "D5: round %d snapshot is stale\n", k);
            return false;
        }

        const std::uint64_t before = Access::mmcssReportedCount(e);
        Access::reportMmcssEvent(e);
        if (Access::mmcssReportedCount(e) != before + 1)
        {
            std::fprintf(stderr, "D5: round %d report did not advance by exactly one\n", k);
            return false;
        }
    }

    const std::uint64_t settled = Access::mmcssReportedCount(e);
    Access::reportMmcssEvent(e);
    if (Access::mmcssReportedCount(e) != settled)
    {
        std::fprintf(stderr, "D5: report advanced without a new record\n");
        return false;
    }

    std::printf("STG11D5MmcssObservationTests: D5 functional PASS (record + once-per-count NonRT diagnosis)\n");
    return true;
}

// =============================================================================
// D5-T6 (structural, read-only): D3 failure and D4 success transports intact.
// =============================================================================
static bool checkD5T6PriorBoundariesIntact()
{
    std::string timerSrc;
    if (!readProductionSource("src/audioengine/AudioEngine.Timer.cpp", timerSrc))
    {
        std::fprintf(stderr, "D5-T6: cannot open AudioEngine.Timer.cpp\n");
        return false;
    }
    std::string d3rec;
    if (!extractFunctionBody(timerSrc, "void AudioEngine::recordAffinityFailure(", d3rec))
    {
        std::fprintf(stderr, "D5-T6: D3 recorder missing (boundary violated)\n");
        return false;
    }
    const std::vector<AtomicAccess> d3order = atomicCallOrder(d3rec);
    const int d3c = indexOfMember(d3order, "affinityFailureCount_");
    const int d3o = indexOfMember(d3order, "affinityFailureObserved_");
    if (d3c < 0 || d3o < 0 || !(d3c < d3o)
        || wrapperOfMember(d3order, "affinityFailureCount_") != "convo::fetchAddAtomic(")
    {
        std::fprintf(stderr, "D5-T6: D3 recorder order broken\n");
        return false;
    }
    std::string d4rec;
    if (!extractFunctionBody(timerSrc, "void AudioEngine::recordSuccessObserved(", d4rec))
    {
        std::fprintf(stderr, "D5-T6: D4 recorder missing (boundary violated)\n");
        return false;
    }
    const std::vector<AtomicAccess> d4order = atomicCallOrder(d4rec);
    const int d4c = indexOfMember(d4order, "successCount_");
    const int d4o = indexOfMember(d4order, "successObserved_");
    if (d4c < 0 || d4o < 0 || !(d4c < d4o)
        || wrapperOfMember(d4order, "successCount_") != "convo::fetchAddAtomic(")
    {
        std::fprintf(stderr, "D5-T6: D4 recorder order broken\n");
        return false;
    }
    // D3/D4 reports still hooked from the existing execution point.
    std::string cbBody;
    if (!extractFunctionBody(timerSrc, "void AudioEngine::timerCallback()", cbBody))
    {
        std::fprintf(stderr, "D5-T6: cannot extract timerCallback body\n");
        return false;
    }
    if (cbBody.find("reportAffinityFailureIfRecorded();") == std::string::npos
        || cbBody.find("reportSuccessIfRecorded();") == std::string::npos
        || cbBody.find("reportMmcssEventIfRecorded();") == std::string::npos)
    {
        std::fprintf(stderr, "D5-T6: report hook set changed\n");
        return false;
    }

    std::printf("STG11D5MmcssObservationTests: D5-T6 PASS (D3/D4 transports untouched)\n");
    return true;
}

// Entry point called from the harness main (PublishPipelineIntegrationTests.cpp).
// No new CTest registration: this is a sub-test of the AudioEngineHarness
// executable, matching the STG-8, STG-9, STG-11-D3, STG-11-D4 shape.
int runSTG11D5MmcssObservationTests()
{
    bool ok = true;
    if (!checkD5T1MmcssSitesAreRtSafe())
    {
        std::fprintf(stderr, "FAIL: D5-T1/D5-T2 MMCSS sites backend-free\n");
        ok = false;
    }
    if (!checkD5RecorderIsRtSafeAndOrdered())
    {
        std::fprintf(stderr, "FAIL: D5 recorder RT-safe and ordered\n");
        ok = false;
    }
    if (!checkD5ReportReadOrder())
    {
        std::fprintf(stderr, "FAIL: D5 report read order\n");
        ok = false;
    }
    if (!checkD5WordingPreserved())
    {
        std::fprintf(stderr, "FAIL: D5-T3 wording preserved\n");
        ok = false;
    }
    if (!checkD5T4ThreadContract())
    {
        std::fprintf(stderr, "FAIL: D5-T4 thread contract\n");
        ok = false;
    }
    if (!checkD5T5SemanticsUnchanged())
    {
        std::fprintf(stderr, "FAIL: D5-T5 semantics unchanged\n");
        ok = false;
    }
    if (!checkD5RecordThenVisible())
    {
        std::fprintf(stderr, "FAIL: D5 record then visible\n");
        ok = false;
    }
    if (!checkD5T6PriorBoundariesIntact())
    {
        std::fprintf(stderr, "FAIL: D5-T6 prior boundaries intact\n");
        ok = false;
    }
    if (ok)
        std::printf("STG11D5MmcssObservationTests: PASS (D5-T1/D5-T2/D5-T3/D5-T4/D5-T5/D5-T6)\n");
    return ok ? 0 : 1;
}
