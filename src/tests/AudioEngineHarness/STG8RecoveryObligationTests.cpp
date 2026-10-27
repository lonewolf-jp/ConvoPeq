// STG8RecoveryObligationTests.cpp — STG-8-D1/D2/D3 regression.
//   Recovery obligation 終端欠落 3 件の回帰テスト（AudioEngineHarness 内 subtest）。
//   新 CTest 登録なし（既存 harness exe のサブテストとして run 側から呼出）。
//   方針: 実エンジン経路のみ（R30 vehicle の submitRecoveryIntent 直呼び pattern）。
//   obligation ID を数値で追跡せず、liveLogicalRecoveryObligationCount (L)・
//   hasDeferredRequest・deferredOverwriteCount・recoveryRetryRedriveCount・
//   publication sequence の公開観測のみで oracle を構成する。
//   L は Live のみを数えるため、L==0 ⟺ 当該 obligation 終端済みと同値。

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <thread>
#include <vector>
#include <fstream>

#include "AudioEngineHarness.h"
#include "DeferredPublicationTestAccess.h"

#if JUCE_WINDOWS
#include <windows.h>

// R8 diagnostic: unbuffered per-test trace. A 0xC0000005 in this suite loses the unflushed
// stdout tail, so the last stderr line is the reliable marker of the running test.
struct Stg8Trace
{
    explicit Stg8Trace(const char* n) noexcept : name(n)
    {
        std::fprintf(stderr, "[STG8-TRACE] enter %s\n", name);
        std::fflush(stderr);
    }
    ~Stg8Trace() noexcept
    {
        std::fprintf(stderr, "[STG8-TRACE] leave %s\n", name);
        std::fflush(stderr);
    }
    const char* name;
};

#endif

namespace {

static void stg8PumpMessages() noexcept
{
#if JUCE_WINDOWS
    MSG msg {};
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE))
    {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
#endif
}

static void stg8SleepPump(int ms)
{
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(ms))
    {
        stg8PumpMessages();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

static bool stg8WaitUntil(double timeoutSec, const std::function<bool()>& pred)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeoutSec);
    while (std::chrono::steady_clock::now() < deadline)
    {
        stg8PumpMessages();
        if (pred())
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    stg8PumpMessages();
    return pred();
}

static std::uint64_t stg8LiveCount(AudioEngine& e)
{
    return DeferredPublicationTestAccess::coordinator(e).liveLogicalRecoveryObligationCount();
}

static std::uint64_t stg8RedriveCount(AudioEngine& e)
{
    return DeferredPublicationTestAccess::coordinator(e).recoveryRetryRedriveCount();
}

// authoritative published runtime の成立待ち（R30 vehicle と同一前提）。
static bool stg8EnsureAuthoritative(AudioEngine& e)
{
    for (int i = 0; i < 3000; ++i)
    {
        const auto* w = e.observePublishedWorld();
        if (w != nullptr && w->engine.current != nullptr && e.hasAuthoritativePublishedRuntime())
            return true;
        stg8PumpMessages();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}

static convo::isr::DSPHandle stg8RegisterActive(AudioEngine& e)
{
    const auto* w = e.observePublishedWorld();
    if (w == nullptr || w->engine.current == nullptr)
        return convo::isr::DSPHandle::null();
    return e.registerDSPHandleForRuntime(static_cast<AudioEngine::DSPCore*>(w->engine.current));
}

//==============================================================================
// ★ STG-8-D1: deferred-discard が recovery obligation を終端化する。
//   fading ON → recovery defer (slot=O1/gen1, L=1) → 新規 submit を伴わない
//   generation bump → watchdog tick で evaluateDeferred-Discard。
//   期待: !hasDeferred && L==0（O1 は ResolvedStaleSuperseded と同値）。
//   修正前は Discard が resolve しないため L==1 のまま残留し FAIL する。
//==============================================================================
static bool checkSTG81DiscardTerminalizesRecovery()
{
    Stg8Trace trace("checkSTG81DiscardTerminalizesRecovery");
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-8-D1] harness start failed\n");
        return false;
    }
    AudioEngine& e = h.engine();
    auto& orch = DeferredPublicationTestAccess::orchestrator(e);
    bool ok = true;

    if (!stg8EnsureAuthoritative(e))
    {
        std::fprintf(stderr, "[STG-8-D1] FAIL: no authoritative runtime\n");
        return false;
    }
    stg8SleepPump(1000);   // baseline tail 消化（R30 と同一）
    if (stg8LiveCount(e) != 0)
    {
        std::fprintf(stderr, "[STG-8-D1] FAIL: setup L=%llu want 0\n",
                     (unsigned long long)stg8LiveCount(e));
        return false;
    }

    DeferredPublicationTestAccess::setFadingRuntimePresent(e, true);
    auto snapshot = e.getCurrentBuildSnapshotForRecovery();
    snapshot.sealed = true;
    const auto h1 = stg8RegisterActive(e);
    if (h1.isNull())
    {
        std::fprintf(stderr, "[STG-8-D1] FAIL: null handle\n");
        return false;
    }
    e.submitRecoveryIntent(h1, snapshot);
    if (!stg8WaitUntil(30.0, [&] { return stg8LiveCount(e) == 1; }))
    {
        std::fprintf(stderr, "[STG-8-D1] FAIL: recovery not admitted\n");
        return false;
    }
    if (!stg8WaitUntil(45.0, [&] { return orch.hasDeferredRequest(); }))
    {
        std::fprintf(stderr, "[STG-8-D1] FAIL: deferred state not reached\n");
        return false;
    }

    // 新規 submit を伴わない generation bump。以後の tick は Discard 一択
    // （generation 不一致 → Ready 不可、他 submit なし → overwrite 不可）。
    DeferredPublicationTestAccess::bumpRebuildGeneration(e);
    if (!stg8WaitUntil(45.0, [&] { return !orch.hasDeferredRequest() && stg8LiveCount(e) == 0; }))
    {
        std::fprintf(stderr, "[STG-8-D1] FAIL: discard leaked (hasDeferred=%d L=%llu)\n",
                     orch.hasDeferredRequest() ? 1 : 0, (unsigned long long)stg8LiveCount(e));
        ok = false;
    }
    // 安定性: 終端後に L が戻らないこと。
    stg8SleepPump(3000);
    if (stg8LiveCount(e) != 0)
    {
        std::fprintf(stderr, "[STG-8-D1] FAIL: L resurrected to %llu\n",
                     (unsigned long long)stg8LiveCount(e));
        ok = false;
    }

    DeferredPublicationTestAccess::setFadingRuntimePresent(e, false);
    h.stop();
    if (!ok)
        return false;
    std::printf("STG8RecoveryObligationTests: PASS (STG-8-D1 discard terminalizes recovery)\n");
    return true;
}

//==============================================================================
// ★ STG-8-D2: deferred-overwrite が追い出された旧 obligation だけを終端化する。
//   fading ON 維持 → O1 defer (L=1) → 別 handle O2 submit (L=2) → Builder が O2 を
//   build して submit → overwrite で O1 追い出し。
//   期待: L==1（O2 のみ Live）&& hasDeferred（O2 保持）。O2 は fading により
//   retention loop で deferred のまま残るため end-state は安定。
//   修正前は O1 が残留し L==2 のまま FAIL する。
//==============================================================================
static bool checkSTG82OverwriteTerminalizesEvicted()
{
    Stg8Trace trace("checkSTG82OverwriteTerminalizesEvicted");
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-8-D2] harness start failed\n");
        return false;
    }
    AudioEngine& e = h.engine();
    auto& orch = DeferredPublicationTestAccess::orchestrator(e);
    bool ok = true;

    if (!stg8EnsureAuthoritative(e))
    {
        std::fprintf(stderr, "[STG-8-D2] FAIL: no authoritative runtime\n");
        return false;
    }
    stg8SleepPump(1000);
    if (stg8LiveCount(e) != 0)
    {
        std::fprintf(stderr, "[STG-8-D2] FAIL: setup L=%llu want 0\n",
                     (unsigned long long)stg8LiveCount(e));
        return false;
    }

    DeferredPublicationTestAccess::setFadingRuntimePresent(e, true);
    auto snapshot = e.getCurrentBuildSnapshotForRecovery();
    snapshot.sealed = true;
    const auto h1 = stg8RegisterActive(e);
    // h2 は O1 と別 identity にするための synthetic handle（RECOVERY-6 は null のみ拒否し、
    // build 入力は buildSource 値コピーから引当するため build 経路は正常動作する。
    // 同一 DSP の二重登録は同一 handle を返すため使えない）。
    const auto h2 = convo::isr::DSPHandle{91, 1};
    if (h1.isNull() || h2.isNull() || h1 == h2)
    {
        std::fprintf(stderr, "[STG-8-D2] FAIL: handles not distinct\n");
        return false;
    }
    e.submitRecoveryIntent(h1, snapshot);
    if (!stg8WaitUntil(30.0, [&] { return stg8LiveCount(e) == 1; }))
    {
        std::fprintf(stderr, "[STG-8-D2] FAIL: O1 not admitted\n");
        return false;
    }
    if (!stg8WaitUntil(45.0, [&] { return orch.hasDeferredRequest(); }))
    {
        std::fprintf(stderr, "[STG-8-D2] FAIL: O1 deferred state not reached\n");
        return false;
    }
    const auto ow0 = orch.deferredOverwriteCount();
    e.submitRecoveryIntent(h2, snapshot);
    if (!stg8WaitUntil(60.0, [&] { return stg8LiveCount(e) == 2; }))
    {
        std::fprintf(stderr, "[STG-8-D2] FAIL: O2 not admitted\n");
        return false;
    }
    // O2 build→submit→overwrite（O1 追い出し）。O2 は fading で retention 維持。
    if (!stg8WaitUntil(120.0, [&] { return stg8LiveCount(e) == 1 && orch.hasDeferredRequest(); }))
    {
        std::fprintf(stderr, "[STG-8-D2] FAIL: evicted leak (L=%llu hasDeferred=%d overwriteDelta=%llu)\n",
                     (unsigned long long)stg8LiveCount(e), orch.hasDeferredRequest() ? 1 : 0,
                     (unsigned long long)(orch.deferredOverwriteCount() - ow0));
        ok = false;
    }
    // 安定性: retention loop 中も L==1 && slot 保持。
    stg8SleepPump(8000);
    if (stg8LiveCount(e) != 1 || !orch.hasDeferredRequest())
    {
        std::fprintf(stderr, "[STG-8-D2] FAIL: unstable (L=%llu hasDeferred=%d)\n",
                     (unsigned long long)stg8LiveCount(e), orch.hasDeferredRequest() ? 1 : 0);
        ok = false;
    }

    DeferredPublicationTestAccess::setFadingRuntimePresent(e, false);
    h.stop();
    if (!ok)
        return false;
    std::printf("STG8RecoveryObligationTests: PASS (STG-8-D2 overwrite terminalizes evicted)\n");
    return true;
}

//==============================================================================
// ★ STG-8-D2b: 同一 obligation の再 defer は終端化しない（coalesce guard）。
//   fading ON → O1 defer (L=1) → 同一 handle 再 submit（同 O1 の新表現）。
//   期待: L==1 維持 && hasDeferred（過終端なし）。修正前後とも PASS する
//   べき guard 証明テスト。
//==============================================================================
static int stg8EnvInt(const char* name, int fallback)
{
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0')
        return fallback;
    return std::atoi(v);
}

// ---------------------------------------------------------------------------
// T-STG8-D2b correlation instrumentation (2026-10-26, owner-approved).
//   D2b previously sampled only the two observable atoms (hasDeferred / L). A "bad" interval
//   therefore could not be told apart from a legitimate in-flight publication: hasDeferred_ is
//   cleared by finishView() and re-armed by the next enqueueDeferred(), so it is false for the
//   whole duration of a publication attempt.
//   These samples add the forward-progress counters so a bad interval can be classified as
//   (A) in-flight publication (counters advance) or (B) liveness loss (counters stalled).
//   Every value comes from a const/atomic getter: nothing here mutates engine state.
// ---------------------------------------------------------------------------
struct D2bSample
{
    long long          tMs             = 0;
    int                deferred        = 0;
    unsigned long long live            = 0;
    unsigned long long submitted       = 0;
    unsigned long long published       = 0;
    unsigned long long retired         = 0;
    int                stage           = 0;   // RuntimePublicationState::StuckStage as int
    unsigned long long maxDeferredAgeMs = 0;
    int                stalled         = 0;
    unsigned long long pending         = 0;
    unsigned long long backlog         = 0;
    // F-4 diagnostic (2026-10-27): three monotonic counters that distinguish "the deferred slot
    //   was overwritten" from "the obligation was terminalized", and show whether a redrive was
    //   in flight during the streak. All three are read through accessors whose backing is
    //   std::atomic (orchestrator: deferredOverwriteCount_, lastRecoveryPublishSeq_; coordinator:
    //   recoveryRetryRedriveCount), so sampling them from the test thread is data-race free.
    unsigned long long overwrite       = 0;   // deferred-overwrite count (monotonic)
    unsigned long long recSeq          = 0;   // last recovery publish sequence
    unsigned long long redrive         = 0;   // recovery retry / redrive count (monotonic)
};

template <typename OrchestratorT>
static D2bSample stg8D2bSample(AudioEngine& e, OrchestratorT& orch, long long tMs) noexcept
{
    D2bSample s {};
    s.tMs = tMs;
    s.deferred = orch.hasDeferredRequest() ? 1 : 0;
    s.live = stg8LiveCount(e);
    const auto& prog = orch.stateOwner().state().progress;
    s.submitted = prog.submittedCount;
    s.published = prog.publishedCount;
    s.retired   = prog.retiredCount;
    s.stage     = static_cast<int>(prog.detectStuckStage());
    s.maxDeferredAgeMs = orch.getMaxDeferredAgeMs();
    s.stalled   = orch.isPublicationStalled() ? 1 : 0;
    s.pending   = orch.getPendingIntentCount();
    s.backlog   = orch.getPublicationBacklogCount();
    // F-4 diagnostic: counter reads only (no assertion, no wait, no behaviour change).
    s.overwrite = orch.deferredOverwriteCount();
    s.recSeq    = static_cast<unsigned long long>(orch.recoveryPublishSeq());
    s.redrive   = stg8RedriveCount(e);
    return s;
}

static void stg8D2bWriteCsv(const char* path, int iteration,
                            const std::vector<D2bSample>& samples)
{
    if (path == nullptr || *path == '\0' || samples.empty())
        return;

    bool needHeader = true;
    {
        std::ifstream probe(path);
        if (probe.good() && probe.peek() != std::ifstream::traits_type::eof())
            needHeader = false;
    }

    std::ofstream f(path, std::ios::app);
    if (!f)
        return;
    if (needHeader)
        f << "iter,t_ms,deferred,live,submitted,published,retired,stage,max_age_ms,stalled,pending,backlog\n";
    for (const auto& x : samples)
        f << iteration << ',' << x.tMs << ',' << x.deferred << ',' << x.live << ','
          << x.submitted << ',' << x.published << ',' << x.retired << ',' << x.stage << ','
          << x.maxDeferredAgeMs << ',' << x.stalled << ',' << x.pending << ',' << x.backlog << '\n';
}

// One D2b iteration on a fresh harness. Returns false on failure.
//   trace: sample the two observable atoms during the retention window and print changes.
static bool stg8RunD2bOnce(int iteration, bool trace)
{
    // Named differently from the bool `trace` parameter (r8 used the same name in the
    // functions that have no such parameter).
    Stg8Trace traceScope("checkSTG82bSameObligationNotResolved");
    const char* tag = (iteration >= 0) ? "STG-8-D2b" : "STG-8-D2b";

    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[%s] harness start failed (iter %d)\n", tag, iteration);
        return false;
    }
    AudioEngine& e = h.engine();
    auto& orch = DeferredPublicationTestAccess::orchestrator(e);
    bool ok = true;

    if (!stg8EnsureAuthoritative(e))
    {
        std::fprintf(stderr, "[%s] FAIL: no authoritative runtime (iter %d)\n", tag, iteration);
        return false;
    }
    stg8SleepPump(1000);

    DeferredPublicationTestAccess::setFadingRuntimePresent(e, true);
    auto snapshot = e.getCurrentBuildSnapshotForRecovery();
    snapshot.sealed = true;
    const auto h1 = stg8RegisterActive(e);
    if (h1.isNull())
    {
        std::fprintf(stderr, "[%s] FAIL: null handle (iter %d)\n", tag, iteration);
        return false;
    }
    e.submitRecoveryIntent(h1, snapshot);
    const auto d2bWindowT0 = std::chrono::steady_clock::now();
    if (!stg8WaitUntil(30.0, [&] { return stg8LiveCount(e) == 1; }))
    {
        std::fprintf(stderr, "[%s] FAIL: O1 not admitted (iter %d)\n", tag, iteration);
        return false;
    }
    if (!stg8WaitUntil(45.0, [&] { return orch.hasDeferredRequest(); }))
    {
        std::fprintf(stderr, "[%s] FAIL: O1 deferred state not reached (iter %d)\n", tag, iteration);
        return false;
    }
    e.submitRecoveryIntent(h1, snapshot);
    // R8 note (kept from r8-stall-clear-obligation-fix, 2026-10-24): the retention window must
    //   stay clear of the production bounded-drop watchdogs. Both are 30 s but run on different
    //   clocks: the publication stall threshold counts from the last committed publication
    //   progress (which precedes this test's T0 by the setup gap), while the deferred TTL counts
    //   from the obligation's creation. A 25 s window left only ~5 s of margin, so a slow setup
    //   could let the stall watchdog fire inside the window. 10 s keeps the same semantics with
    //   ~19 s of margin. Re-baseline the stall observation here (the same call prepareToPlay()
    //   uses) so the window cannot be cut short by the setup gap.
    orch.resetProgressObservation();


    // Retention window (see the R8 note in the original test): 10 s keeps clear of the 30 s
    // publication-stall watchdog and the 30 s deferred TTL.
    //
    // T-STG8-D2b triage (2026-10-26): measured with CONVOPEQ_STG8_D2B_TRACE=1, the deferred
    // flag legitimately cycles 1 -> 0 -> 1 inside this window (dips of 5-16 ms = one audio
    // block) while the live obligation count stays 1. A single sample at the end therefore
    // reported "over-terminalized" whenever it landed inside a re-arm dip. The verdict is
    // now about a DURABLE loss:
    //   - a continuous non-retained streak >= kDurableBadMs fails;
    //   - a bad end snapshot is given kRecoveryBudgetMs to recover before failing.
    constexpr long long kDurableBadMs = 250;     // ~23 audio blocks; a real loss is permanent
    constexpr long long kRecoveryBudgetMs = 1000;
    long long maxBadStreakMs = 0;
    std::vector<D2bSample> samples;          // correlation log (trace only)
    D2bSample firstBad {};
    D2bSample lastBad {};
    long long firstBadMs = -1;
    long long lastBadMs = -1;
    int       deferredTransitions = 0;
    int       badSamples = 0;
    {
        const auto t0 = std::chrono::steady_clock::now();
        bool bad = false;
        auto badSince = t0;
        int  prevDeferred = -1;
        while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(10000))
        {
            stg8PumpMessages();
            std::this_thread::sleep_for(std::chrono::milliseconds(5));

            const auto now = std::chrono::steady_clock::now();
            const long long tMs =
                std::chrono::duration_cast<std::chrono::milliseconds>(now - t0).count();
            const auto s = stg8D2bSample(e, orch, tMs);
            if (trace)
                samples.push_back(s);
            if (prevDeferred >= 0 && s.deferred != prevDeferred)
                ++deferredTransitions;
            prevDeferred = s.deferred;

            if (s.live != 1 || s.deferred == 0)
            {
                ++badSamples;
                if (!bad)
                {
                    bad = true;
                    badSince = now;
                    firstBad = s;
                    firstBadMs = tMs;
                }
                lastBad = s;
                lastBadMs = tMs;
                maxBadStreakMs = std::max<long long>(
                    maxBadStreakMs,
                    std::chrono::duration_cast<std::chrono::milliseconds>(now - badSince).count());
            }
            else
            {
                bad = false;
            }
        }
    }

    // ---- correlation summary (observation only; the D2b verdict is unchanged) ----
    if (trace && !samples.empty())
    {
        const auto& f = samples.front();
        const auto& l = samples.back();
        unsigned long long maxAge = 0;
        unsigned long long maxPending = 0;
        unsigned long long maxBacklog = 0;
        int stalledSamples = 0;
        for (const auto& s : samples)
        {
            maxAge = std::max(maxAge, s.maxDeferredAgeMs);
            maxPending = std::max(maxPending, s.pending);
            maxBacklog = std::max(maxBacklog, s.backlog);
            if (s.stalled != 0)
                ++stalledSamples;
        }
        std::printf("[D2b-SUM] iter=%d samples=%zu window_ms=%lld transitions=%d bad=%d"
                    " longestBad_ms=%lld overwrite=%llu recSeq=%llu redrive=%llu\n",
                    iteration, samples.size(), l.tMs, deferredTransitions, badSamples,
                    maxBadStreakMs, l.overwrite, l.recSeq, l.redrive);
        std::printf("[D2b-CNT] iter=%d start{t=%lld sub=%llu pub=%llu ret=%llu stage=%d age=%llu}"
                    " end{t=%lld sub=%llu pub=%llu ret=%llu stage=%d age=%llu}\n",
                    iteration, f.tMs, f.submitted, f.published, f.retired, f.stage,
                    f.maxDeferredAgeMs, l.tMs, l.submitted, l.published, l.retired,
                    l.stage, l.maxDeferredAgeMs);
        std::printf("[D2b-BAD] iter=%d streak[%lld..%lld] sub=+%llu pub=+%llu ret=+%llu"
                    " stage %d->%d age %llu->%llu stalled=%d maxAge=%llu maxPending=%llu"
                    " maxBacklog=%llu overwrite=%llu->%llu recSeq=%llu->%llu redrive=%llu->%llu\n",
                    iteration, firstBadMs, lastBadMs,
                    lastBad.submitted - firstBad.submitted,
                    lastBad.published - firstBad.published,
                    lastBad.retired - firstBad.retired,
                    firstBad.stage, lastBad.stage,
                    firstBad.maxDeferredAgeMs, lastBad.maxDeferredAgeMs,
                    stalledSamples, maxAge, maxPending, maxBacklog,
                    firstBad.overwrite, lastBad.overwrite,
                    firstBad.recSeq, lastBad.recSeq,
                    firstBad.redrive, lastBad.redrive);
        const char* csvPath = std::getenv("CONVOPEQ_STG8_D2B_CSV");
        if (csvPath != nullptr && *csvPath != '\0')
            stg8D2bWriteCsv(csvPath, iteration, samples);
    }

    // Single snapshot decides the verdict; the diagnostic prints THIS snapshot only.
    auto liveFinal = stg8LiveCount(e);
    auto deferredFinal = orch.hasDeferredRequest();
    long long endBadMs = 0;

    if (liveFinal != 1 || !deferredFinal)
    {
        // Give the obligation a bounded chance to be re-armed before calling it durable.
        const auto t0 = std::chrono::steady_clock::now();
        for (;;)
        {
            stg8PumpMessages();
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            const auto now = std::chrono::steady_clock::now();
            endBadMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - t0).count();
            liveFinal = stg8LiveCount(e);
            deferredFinal = orch.hasDeferredRequest();
            if (liveFinal == 1 && deferredFinal)
                break;                       // recovered: transient
            if (endBadMs >= kRecoveryBudgetMs)
                break;                       // durable loss
        }
    }

    const bool durableLoss =
        (maxBadStreakMs >= kDurableBadMs) || (liveFinal != 1 || !deferredFinal);

    if (durableLoss)
    {
        const long long d2bWindowMs = static_cast<long long>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - d2bWindowT0).count());
        std::fprintf(stderr,
                     "[STG-8-D2b] FAIL: over-terminalized (iter %d, snapshot L=%llu hasDeferred=%d,"
                     " maxBadStreakMs=%lld, endBadMs=%lld, windowMs=%lld, trace=%d)\n",
                     iteration,
                     (unsigned long long)liveFinal, deferredFinal ? 1 : 0,
                     maxBadStreakMs, endBadMs, d2bWindowMs, trace ? 1 : 0);
        ok = false;
    }
    else if (maxBadStreakMs > 0 || endBadMs > 0)
    {
        std::printf("[STG-8-D2b] note iter=%d: transient re-arm dip recovered"
                    " (maxBadStreakMs=%lld, endBadMs=%lld)\n",
                    iteration, maxBadStreakMs, endBadMs);
    }

    // ---- (A)/(B): once fading clears, is the obligation resolved exactly once? ----
    //   Observation only (never fails). Forward progress immediately after the retention
    //   window supports (A) in-flight publication; a live obligation that never drains inside
    //   the budget supports (B) liveness loss. The verdict above is not affected.
    {
        const auto pre = stg8D2bSample(e, orch, 0);
        DeferredPublicationTestAccess::setFadingRuntimePresent(e, false);
        const auto t0 = std::chrono::steady_clock::now();
        constexpr long long kResolveBudgetMs = 15000;
        long long resolvedMs = -1;
        unsigned long long maxLiveObserved = pre.live;
        unsigned long long maxAgeObserved = pre.maxDeferredAgeMs;
        for (;;)
        {
            stg8PumpMessages();
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            const auto now = std::chrono::steady_clock::now();
            const long long ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(now - t0).count();
            const auto liveNow = stg8LiveCount(e);
            maxLiveObserved = std::max(maxLiveObserved, liveNow);
            maxAgeObserved = std::max(maxAgeObserved, orch.getMaxDeferredAgeMs());
            if (liveNow == 0) { resolvedMs = ms; break; }
            if (ms >= kResolveBudgetMs) break;
        }
        const auto post = stg8D2bSample(e, orch, resolvedMs);
        std::printf("[D2b-RES] iter=%d resolved=%d atMs=%lld maxLive=%llu maxAge_ms=%llu"
                    " sub=+%llu pub=+%llu ret=+%llu stage=%d deferred=%d\n",
                    iteration, resolvedMs >= 0 ? 1 : 0, resolvedMs, maxLiveObserved,
                    maxAgeObserved, post.submitted - pre.submitted,
                    post.published - pre.published, post.retired - pre.retired,
                    post.stage, post.deferred);
    }

    h.stop();
    return ok;
}

static bool checkSTG82bSameObligationNotResolved()
{
    const int repeats = stg8EnvInt("CONVOPEQ_STG8_D2B_REPEAT", 1);
    const bool trace = stg8EnvInt("CONVOPEQ_STG8_D2B_TRACE", 0) != 0;

    int failures = 0;
    for (int i = 0; i < repeats; ++i)
    {
        if (!stg8RunD2bOnce(i, trace))
            ++failures;
        if (repeats == 1)
            break;
    }

    if (failures != 0)
    {
        std::fprintf(stderr, "[STG-8-D2b] iterations=%d failures=%d\n", repeats, failures);
        return false;
    }
    if (repeats > 1)
        std::printf("STG8RecoveryObligationTests: PASS (STG-8-D2b same obligation retained, %d iterations)\n",
                    repeats);
    else
        std::printf("STG8RecoveryObligationTests: PASS (STG-8-D2b same obligation retained)\n");
    return true;
}

//==============================================================================
// STG-8-CLEAR: clearDeferredForShutdown() must terminalize the recovery obligation.
//   Reproduction test for the R8 stall-clear asymmetry: a deferred publish with a live
//   obligation is cleared (mid-run drain path). Before the fix, clearDeferredForShutdown()
//   released the slot (hasDeferred=0) but left the obligation live (L=1). After the fix,
//   the obligation is resolved (L=0) via resolveRecoveryObligation(StaleSuperseded).
//   This test calls clearDeferredForShutdown() directly to exercise the path without
//   waiting for the 30 s stall watchdog.
//==============================================================================
static bool checkSTG8ClearTerminalizesObligation()
{
    Stg8Trace trace("checkSTG8ClearTerminalizesObligation");
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-8-CLEAR] harness start failed\n");
        return false;
    }
    AudioEngine& e = h.engine();
    auto& orch = DeferredPublicationTestAccess::orchestrator(e);
    bool ok = true;

    if (!stg8EnsureAuthoritative(e))
    {
        std::fprintf(stderr, "[STG-8-CLEAR] FAIL: no authoritative runtime\n");
        return false;
    }
    stg8SleepPump(1000);

    DeferredPublicationTestAccess::setFadingRuntimePresent(e, true);
    auto snapshot = e.getCurrentBuildSnapshotForRecovery();
    snapshot.sealed = true;
    const auto h1 = stg8RegisterActive(e);
    if (h1.isNull())
    {
        std::fprintf(stderr, "[STG-8-CLEAR] FAIL: null handle\n");
        return false;
    }
    e.submitRecoveryIntent(h1, snapshot);
    if (!stg8WaitUntil(30.0, [&] { return stg8LiveCount(e) == 1; }))
    {
        std::fprintf(stderr, "[STG-8-CLEAR] FAIL: O1 not admitted\n");
        return false;
    }
    if (!stg8WaitUntil(45.0, [&] { return orch.hasDeferredRequest(); }))
    {
        std::fprintf(stderr, "[STG-8-CLEAR] FAIL: O1 deferred state not reached\n");
        return false;
    }

    // Pre-condition: obligation is live, deferred request is retained.
    if (stg8LiveCount(e) != 1 || !orch.hasDeferredRequest())
    {
        std::fprintf(stderr, "[STG-8-CLEAR] FAIL: pre-condition (L=%llu hasDeferred=%d)\n",
                     (unsigned long long)stg8LiveCount(e), orch.hasDeferredRequest() ? 1 : 0);
        ok = false;
    }

    // Invoke the clear path (mid-run drain, not actual shutdown).
    orch.clearDeferredForShutdown();
    stg8SleepPump(500);

    // Post-condition: deferred request released AND obligation terminalized (L=0).
    if (orch.hasDeferredRequest())
    {
        std::fprintf(stderr, "[STG-8-CLEAR] FAIL: hasDeferred still true after clear\n");
        ok = false;
    }
    if (stg8LiveCount(e) != 0)
    {
        std::fprintf(stderr, "[STG-8-CLEAR] FAIL: obligation NOT terminalized (L=%llu, expected 0)\n",
                     (unsigned long long)stg8LiveCount(e));
        ok = false;
    }

    DeferredPublicationTestAccess::setFadingRuntimePresent(e, false);
    h.stop();
    if (!ok)
        return false;
    std::printf("STG8RecoveryObligationTests: PASS (STG-8-CLEAR clear terminalizes obligation)\n");
    return true;
}

//==============================================================================
// R8-2: repeated clear must be idempotent (no double terminalization / double release).
//   Calls clearDeferredForShutdown() three times total. After the first call the
//   obligation is terminal (L=0); the 2nd and 3rd calls must be no-ops:
//     - L stays exactly 0 (a second resolve would underflow the live counter,
//       since resolve() performs the single -1 through the id-based table),
//     - hasDeferred stays false (the slot must not be resurrected).
//==============================================================================
static bool checkSTG8ClearIsIdempotent()
{
    Stg8Trace trace("checkSTG8ClearIsIdempotent");
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[R8-2] harness start failed\n");
        return false;
    }
    AudioEngine& e = h.engine();
    auto& orch = DeferredPublicationTestAccess::orchestrator(e);
    bool ok = true;

    if (!stg8EnsureAuthoritative(e))
    {
        std::fprintf(stderr, "[R8-2] FAIL: no authoritative runtime\n");
        return false;
    }
    stg8SleepPump(1000);

    DeferredPublicationTestAccess::setFadingRuntimePresent(e, true);
    auto snapshot = e.getCurrentBuildSnapshotForRecovery();
    snapshot.sealed = true;
    const auto h1 = stg8RegisterActive(e);
    if (h1.isNull())
    {
        std::fprintf(stderr, "[R8-2] FAIL: null handle\n");
        return false;
    }
    e.submitRecoveryIntent(h1, snapshot);
    if (!stg8WaitUntil(30.0, [&] { return stg8LiveCount(e) == 1; }))
    {
        std::fprintf(stderr, "[R8-2] FAIL: O1 not admitted\n");
        return false;
    }
    if (!stg8WaitUntil(45.0, [&] { return orch.hasDeferredRequest(); }))
    {
        std::fprintf(stderr, "[R8-2] FAIL: deferred state not reached\n");
        return false;
    }

    // First clear: obligation terminalized.
    orch.clearDeferredForShutdown();
    stg8SleepPump(300);
    if (stg8LiveCount(e) != 0 || orch.hasDeferredRequest())
    {
        std::fprintf(stderr, "[R8-2] FAIL: first clear (L=%llu hasDeferred=%d)\n",
                     (unsigned long long)stg8LiveCount(e), orch.hasDeferredRequest() ? 1 : 0);
        ok = false;
    }

    // Repeat twice: must be a no-op.
    for (int i = 0; i < 2; ++i)
    {
        orch.clearDeferredForShutdown();
        stg8SleepPump(100);
    }

    if (stg8LiveCount(e) != 0)
    {
        std::fprintf(stderr, "[R8-2] FAIL: repeated clear changed L (L=%llu, expected 0)\n",
                     (unsigned long long)stg8LiveCount(e));
        ok = false;
    }
    if (orch.hasDeferredRequest())
    {
        std::fprintf(stderr, "[R8-2] FAIL: repeated clear resurrected deferred slot\n");
        ok = false;
    }

    DeferredPublicationTestAccess::setFadingRuntimePresent(e, false);
    h.stop();
    if (!ok)
        return false;
    std::printf("STG8RecoveryObligationTests: PASS (R8-2 repeated clear is idempotent)\n");
    return true;
}

//==============================================================================
// R8-3: terminal reason must match the actual state.
//   A mid-run clear is NOT a shutdown, so it must terminalize the obligation as
//   StaleSuperseded, never ShutdownDiscarded.
//   recoveryObligationShutdownDiscardCount() is incremented only by the
//   ShutdownDiscarded outcome (ISRRuntimePublicationCoordinator.cpp resolve path),
//   so it must be unchanged across a mid-run clear. If it moves, clearIsShutdown
//   was computed wrongly and the reason recorded is wrong.
//==============================================================================
static bool checkSTG8ClearTerminalReasonIsStale()
{
    Stg8Trace trace("checkSTG8ClearTerminalReasonIsStale");
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[R8-3] harness start failed\n");
        return false;
    }
    AudioEngine& e = h.engine();
    auto& orch = DeferredPublicationTestAccess::orchestrator(e);
    auto& coord = DeferredPublicationTestAccess::coordinator(e);
    bool ok = true;

    if (!stg8EnsureAuthoritative(e))
    {
        std::fprintf(stderr, "[R8-3] FAIL: no authoritative runtime\n");
        return false;
    }
    stg8SleepPump(1000);

    DeferredPublicationTestAccess::setFadingRuntimePresent(e, true);
    auto snapshot = e.getCurrentBuildSnapshotForRecovery();
    snapshot.sealed = true;
    const auto h1 = stg8RegisterActive(e);
    if (h1.isNull())
    {
        std::fprintf(stderr, "[R8-3] FAIL: null handle\n");
        return false;
    }
    e.submitRecoveryIntent(h1, snapshot);
    if (!stg8WaitUntil(30.0, [&] { return stg8LiveCount(e) == 1; }))
    {
        std::fprintf(stderr, "[R8-3] FAIL: O1 not admitted\n");
        return false;
    }
    if (!stg8WaitUntil(45.0, [&] { return orch.hasDeferredRequest(); }))
    {
        std::fprintf(stderr, "[R8-3] FAIL: deferred state not reached\n");
        return false;
    }

    const auto discardBefore = coord.recoveryObligationShutdownDiscardCount();

    orch.clearDeferredForShutdown();   // mid-run drain: not an actual shutdown
    stg8SleepPump(300);

    const auto discardAfter = coord.recoveryObligationShutdownDiscardCount();

    if (stg8LiveCount(e) != 0)
    {
        std::fprintf(stderr, "[R8-3] FAIL: obligation not terminalized (L=%llu)\n",
                     (unsigned long long)stg8LiveCount(e));
        ok = false;
    }
    if (discardAfter != discardBefore)
    {
        std::fprintf(stderr,
                     "[R8-3] FAIL: mid-run clear counted as ShutdownDiscarded (%llu -> %llu)\n",
                     (unsigned long long)discardBefore, (unsigned long long)discardAfter);
        ok = false;
    }

    DeferredPublicationTestAccess::setFadingRuntimePresent(e, false);
    h.stop();
    if (!ok)
        return false;
    std::printf("STG8RecoveryObligationTests: PASS (R8-3 mid-run clear is StaleSuperseded)\n");
    return true;
}

//==============================================================================
// ★ STG-8-D3: RejectedPressure の transport recovery が signal→redrive で再送される。
//   fading OFF・throttle ON → recovery submit (L=1, Transport) → Builder build→
//   submit → RejectedPressure →（修正: signal）→ throttle OFF → adjudicate→None→
//   redrive→再 Transport→再 build→Accepted→Published。
//   期待: redriveCount 増加 && L==0 && sequence 前進。
//   修正前は redrive 0 件・L==1 のまま停滞し FAIL する。
//==============================================================================
//==============================================================================
// R8-12 (P1): the real 30 s publication-stall watchdog, driven by the wall clock.
//   R8-9 ages m_lastProgressTimestampUs (synthetic clock) so the suite stays fast, and
//   R8-8 pushes the health event directly. Neither proves that the *production* threshold
//   (kPublicationStallThresholdUs == 30 s) is reached by real elapsed time and that the
//   engine's own timer tick then drives requestDeferredClear -> RebuildThread drain ->
//   clearDeferredForShutdown. This test waits the full threshold out in real time.
//   PASS: stalled first observed at >= 30 s real time; obligation terminalized (L=0);
//         deferred slot drained; reason is NOT ShutdownDiscarded (engine still running).
//==============================================================================
static bool checkR8RealTimeStallDetectorTerminalizes()
{
    Stg8Trace trace("checkR8RealTimeStallDetectorTerminalizes");
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[R8-12] harness start failed\n");
        return false;
    }

    AudioEngine& e = h.engine();
    auto& orch = DeferredPublicationTestAccess::orchestrator(e);
    auto& coord = DeferredPublicationTestAccess::coordinator(e);
    bool ok = true;

    if (!stg8EnsureAuthoritative(e))
    {
        std::fprintf(stderr, "[R8-12] FAIL: no authoritative runtime\n");
        return false;
    }
    stg8SleepPump(1000);

    DeferredPublicationTestAccess::setFadingRuntimePresent(e, true);
    auto snapshot = e.getCurrentBuildSnapshotForRecovery();
    snapshot.sealed = true;
    const auto h1 = stg8RegisterActive(e);
    if (h1.isNull())
    {
        std::fprintf(stderr, "[R8-12] FAIL: null handle\n");
        return false;
    }
    e.submitRecoveryIntent(h1, snapshot);

    if (!stg8WaitUntil(45.0, [&] { return orch.hasDeferredRequest(); }))
    {
        std::fprintf(stderr, "[R8-12] FAIL: deferred state not reached\n");
        return false;
    }

    const auto discardBefore = coord.recoveryObligationShutdownDiscardCount();

    // Real time only: no ageProgress(), no manufactured health event. Baseline the progress
    // observation first (the call prepareToPlay() uses) so the 30 s threshold is measured
    // from T0 rather than from the last committed publication before the setup.
    orch.resetProgressObservation();
    const bool stalledAtBaseline = orch.isPublicationStalled();
    const auto stallStart = std::chrono::steady_clock::now();
    const bool stalled = stg8WaitUntil(90.0, [&] { return orch.isPublicationStalled(); });
    const double stalledAfterSec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - stallStart).count();

    if (!stalled)
    {
        std::fprintf(stderr, "[R8-12] FAIL: detector never reported stalled (waited %.1f s)\n",
                     stalledAfterSec);
        DeferredPublicationTestAccess::setFadingRuntimePresent(e, false);
        h.stop();
        return false;
    }

    if (stalledAtBaseline)
    {
        std::fprintf(stderr, "[R8-12] FAIL: detector already latched right after the baseline\n");
        ok = false;
    }

    // The health monitor evaluates on its own ~1 s tick, so the report lands at or just after
    // the 30 s threshold. A 25 s floor still catches a detector that fires materially early
    // (a 10 s threshold, or a clock that was not baselined) without policing tick granularity.
    if (stalledAfterSec < 25.0)
    {
        std::fprintf(stderr,
                     "[R8-12] FAIL: stalled reported after %.2f s, far before the 30 s threshold\n",
                     stalledAfterSec);
        ok = false;
    }

    if (!stg8WaitUntil(60.0, [&] {
            return !orch.hasDeferredRequest() && stg8LiveCount(e) == 0;
        }))
    {
        std::fprintf(stderr,
                     "[R8-12] FAIL: real-time stall clear did not terminalize "
                     "(hasDeferred=%d L=%llu)\n",
                     orch.hasDeferredRequest() ? 1 : 0,
                     (unsigned long long)stg8LiveCount(e));
        ok = false;
    }

    const auto discardAfter = coord.recoveryObligationShutdownDiscardCount();
    if (discardAfter != discardBefore)
    {
        std::fprintf(stderr,
                     "[R8-12] FAIL: mid-run clear counted as ShutdownDiscarded (%llu -> %llu)\n",
                     (unsigned long long)discardBefore, (unsigned long long)discardAfter);
        ok = false;
    }

    DeferredPublicationTestAccess::setFadingRuntimePresent(e, false);
    h.stop();

    if (!ok)
        return false;

    std::printf("STG8RecoveryObligationTests: PASS "
                "(R8-12 real-time 30 s stall detector -> StaleSuperseded, L=0, "
                "stalledAfter=%.2fs)\n", stalledAfterSec);
    return true;
}

static bool checkSTG83PressureSignalRedrives()
{
    Stg8Trace trace("checkSTG83PressureSignalRedrives");
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-8-D3] harness start failed\n");
        return false;
    }
    AudioEngine& e = h.engine();
    bool ok = true;

    if (!stg8EnsureAuthoritative(e))
    {
        std::fprintf(stderr, "[STG-8-D3] FAIL: no authoritative runtime\n");
        return false;
    }
    stg8SleepPump(1000);
    if (stg8LiveCount(e) != 0)
    {
        std::fprintf(stderr, "[STG-8-D3] FAIL: setup L=%llu want 0\n",
                     (unsigned long long)stg8LiveCount(e));
        return false;
    }

    DeferredPublicationTestAccess::setFadingRuntimePresent(e, false);
    DeferredPublicationTestAccess::setRetirePressureThrottle(e, true);
    auto snapshot = e.getCurrentBuildSnapshotForRecovery();
    snapshot.sealed = true;
    const auto h1 = stg8RegisterActive(e);
    if (h1.isNull())
    {
        std::fprintf(stderr, "[STG-8-D3] FAIL: null handle\n");
        return false;
    }
    const auto* w0 = e.observePublishedWorld();
    const long long baseSeq = (w0 != nullptr)
        ? static_cast<long long>(w0->publication.sequenceId) : 0LL;
    const auto redrive0 = stg8RedriveCount(e);

    e.submitRecoveryIntent(h1, snapshot);
    if (!stg8WaitUntil(30.0, [&] { return stg8LiveCount(e) == 1; }))
    {
        std::fprintf(stderr, "[STG-8-D3] FAIL: recovery not admitted\n");
        DeferredPublicationTestAccess::setRetirePressureThrottle(e, false);
        return false;
    }
    // Builder build→submit→RejectedPressure→signal→adjudicate→redrive を待つ。
    // throttle ON 期間は短く保つ（budget 4 消費を避けるため redrive 初発で解除）。
    if (!stg8WaitUntil(90.0, [&] { return stg8RedriveCount(e) > redrive0; }))
    {
        std::fprintf(stderr, "[STG-8-D3] FAIL: no redrive (count=%llu L=%llu)\n",
                     (unsigned long long)stg8RedriveCount(e), (unsigned long long)stg8LiveCount(e));
        ok = false;
    }
    DeferredPublicationTestAccess::setRetirePressureThrottle(e, false);
    // 圧力解除後の再送 build→Accepted→Published（L==0＋sequence 前進）。
    if (!stg8WaitUntil(120.0, [&] { return stg8LiveCount(e) == 0; }))
    {
        std::fprintf(stderr, "[STG-8-D3] FAIL: not completed (L=%llu)\n",
                     (unsigned long long)stg8LiveCount(e));
        ok = false;
    }
    const auto* w1 = e.observePublishedWorld();
    const long long seq1 = (w1 != nullptr)
        ? static_cast<long long>(w1->publication.sequenceId) : 0LL;
    if (ok && seq1 <= baseSeq)
    {
        std::fprintf(stderr, "[STG-8-D3] FAIL: sequence not advanced (%lld <= %lld)\n", seq1, baseSeq);
        ok = false;
    }

    h.stop();
    if (!ok)
        return false;
    std::printf("STG8RecoveryObligationTests: PASS (STG-8-D3 pressure signal redrives)\n");
    return true;
}

//==============================================================================
// R8-4: the shutdown branch must record ShutdownDiscarded (not StaleSuperseded).
//   Forces EngineLifecycleState::Releasing so isShutdownInProgress() is true, then runs
//   the clear path and asserts recoveryObligationShutdownDiscardCount() advances by
//   exactly one while the obligation is terminalized (L=0, hasDeferred=0).
//   The original lifecycle state is restored before the harness is stopped.
//==============================================================================
static bool checkSTG8ClearUsesShutdownReason()
{
    Stg8Trace trace("checkSTG8ClearUsesShutdownReason");
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[R8-4] harness start failed\n");
        return false;
    }
    AudioEngine& e = h.engine();
    auto& orch = DeferredPublicationTestAccess::orchestrator(e);
    auto& coord = DeferredPublicationTestAccess::coordinator(e);
    bool ok = true;

    if (!stg8EnsureAuthoritative(e))
    {
        std::fprintf(stderr, "[R8-4] FAIL: no authoritative runtime\n");
        return false;
    }
    stg8SleepPump(1000);

    DeferredPublicationTestAccess::setFadingRuntimePresent(e, true);
    auto snapshot = e.getCurrentBuildSnapshotForRecovery();
    snapshot.sealed = true;
    const auto h1 = stg8RegisterActive(e);
    if (h1.isNull())
    {
        std::fprintf(stderr, "[R8-4] FAIL: null handle\n");
        return false;
    }
    e.submitRecoveryIntent(h1, snapshot);
    if (!stg8WaitUntil(30.0, [&] { return stg8LiveCount(e) == 1; }))
    {
        std::fprintf(stderr, "[R8-4] FAIL: O1 not admitted\n");
        return false;
    }
    if (!stg8WaitUntil(45.0, [&] { return orch.hasDeferredRequest(); }))
    {
        std::fprintf(stderr, "[R8-4] FAIL: deferred state not reached\n");
        return false;
    }

    const int savedState = DeferredPublicationTestAccess::getLifecycleStateRaw(e);
    const auto discardBefore = coord.recoveryObligationShutdownDiscardCount();

    // Pretend the engine is releasing, so clearIsShutdown == true.
    DeferredPublicationTestAccess::setLifecycleStateRaw(e, 3 /* Releasing */);

    orch.clearDeferredForShutdown();
    stg8SleepPump(300);

    const auto discardAfter = coord.recoveryObligationShutdownDiscardCount();

    // Restore immediately so nothing downstream observes a fake shutdown.
    DeferredPublicationTestAccess::setLifecycleStateRaw(e, savedState);

    if (stg8LiveCount(e) != 0)
    {
        std::fprintf(stderr, "[R8-4] FAIL: obligation not terminalized (L=%llu)\n",
                     (unsigned long long)stg8LiveCount(e));
        ok = false;
    }
    if (orch.hasDeferredRequest())
    {
        std::fprintf(stderr, "[R8-4] FAIL: hasDeferred still true after clear\n");
        ok = false;
    }
    // Shutdown branch must be classified as ShutdownDiscarded: counter increments by 1.
    if (discardAfter != discardBefore + 1)
    {
        std::fprintf(stderr,
                     "[R8-4] FAIL: shutdown clear not recorded as ShutdownDiscarded "
                     "(%llu -> %llu, expected +1)\n",
                     (unsigned long long)discardBefore, (unsigned long long)discardAfter);
        ok = false;
    }

    DeferredPublicationTestAccess::setFadingRuntimePresent(e, false);
    h.stop();
    if (!ok)
        return false;
    std::printf("STG8RecoveryObligationTests: PASS (R8-4 shutdown clear records ShutdownDiscarded)\n");
    return true;
}

//==============================================================================
// R8-5: production entry point. The 30 s publication-stall watchdog does not call
// clearDeferredForShutdown() directly: AudioEngine.Timer.cpp:1847 calls
// requestDeferredClear(), which latches deferredClearRequested_ + notifies the
// RebuildThread, and the clear then runs inside drainDeferredClearIfRequested().
// This test drives that same entry point and proves the obligation is terminalized
// by the drain, so the latch -> notify -> drain -> clear chain is covered rather
// than only the direct call.
//==============================================================================
static bool checkSTG8RequestDeferredClearDrains()
{
    Stg8Trace trace("checkSTG8RequestDeferredClearDrains");
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[R8-5] harness start failed\n");
        return false;
    }
    AudioEngine& e = h.engine();
    auto& orch = DeferredPublicationTestAccess::orchestrator(e);
    auto& coord = DeferredPublicationTestAccess::coordinator(e);
    bool ok = true;

    if (!stg8EnsureAuthoritative(e))
    {
        std::fprintf(stderr, "[R8-5] FAIL: no authoritative runtime\n");
        return false;
    }
    stg8SleepPump(1000);

    DeferredPublicationTestAccess::setFadingRuntimePresent(e, true);
    auto snapshot = e.getCurrentBuildSnapshotForRecovery();
    snapshot.sealed = true;
    const auto h1 = stg8RegisterActive(e);
    if (h1.isNull())
    {
        std::fprintf(stderr, "[R8-5] FAIL: null handle\n");
        return false;
    }
    e.submitRecoveryIntent(h1, snapshot);
    if (!stg8WaitUntil(30.0, [&] { return stg8LiveCount(e) == 1; }))
    {
        std::fprintf(stderr, "[R8-5] FAIL: O1 not admitted\n");
        return false;
    }
    if (!stg8WaitUntil(45.0, [&] { return orch.hasDeferredRequest(); }))
    {
        std::fprintf(stderr, "[R8-5] FAIL: deferred state not reached\n");
        return false;
    }

    const auto discardBefore = coord.recoveryObligationShutdownDiscardCount();

    // Exactly what the 30 s stall watchdog does (AudioEngine.Timer.cpp:1847).
    orch.requestDeferredClear();

    // The RebuildThread performs the clear on its next wake.
    if (!stg8WaitUntil(45.0, [&] { return stg8LiveCount(e) == 0; }))
    {
        std::fprintf(stderr, "[R8-5] FAIL: obligation not terminalized by drain (L=%llu)\n",
                     (unsigned long long)stg8LiveCount(e));
        ok = false;
    }
    if (orch.hasDeferredRequest())
    {
        std::fprintf(stderr, "[R8-5] FAIL: hasDeferred still true after drain\n");
        ok = false;
    }
    // Mid-run drain: must be StaleSuperseded, so the shutdown counter stays flat.
    if (coord.recoveryObligationShutdownDiscardCount() != discardBefore)
    {
        std::fprintf(stderr, "[R8-5] FAIL: drain classified as ShutdownDiscarded\n");
        ok = false;
    }

    DeferredPublicationTestAccess::setFadingRuntimePresent(e, false);
    h.stop();
    if (!ok)
        return false;
    std::printf("STG8RecoveryObligationTests: PASS (R8-5 requestDeferredClear drain terminalizes)\n");
    return true;
}

} // namespace

// main 側（PublishPipelineIntegrationTests.cpp）から呼ばれるエントリ。
// 新 CTest 登録なし（AudioEngineHarness 既存 exe のサブテスト）。
//==============================================================================
// ★ R8-10 (P1): C1 under contention — the exact window the fix targets.
//   A foreign thread hammers requestDeferredClear() (the timer's entry point) while the
//   main thread runs stop() -> stopRebuildThread() -> join(). Pre-fix, the C1 branch was
//   taken as soon as rebuildThreadShouldExit was published (before join), so the spammer
//   called clearDeferredForShutdown() while the RebuildThread could still be inside
//   processDeferredAdmission() - two threads on the non-atomic deferred members.
//   Post-fix the spammer can only latch until join() has returned.
//   Oracle: the teardown must leave exactly one terminalization (L == 0, never negative),
//   no deferred residue, and no crash.
//==============================================================================
static bool checkR8C1RaceWithTeardown()
{
    Stg8Trace trace("checkR8C1RaceWithTeardown");

    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[R8-10] harness start failed\n");
        return false;
    }

    AudioEngine& e = h.engine();
    auto& orch = DeferredPublicationTestAccess::orchestrator(e);

    if (!stg8EnsureAuthoritative(e))
    {
        std::fprintf(stderr, "[R8-10] FAIL: no authoritative runtime\n");
        return false;
    }
    stg8SleepPump(1000);

    DeferredPublicationTestAccess::setFadingRuntimePresent(e, true);
    auto snapshot = e.getCurrentBuildSnapshotForRecovery();
    snapshot.sealed = true;
    const auto h1 = stg8RegisterActive(e);
    if (h1.isNull())
    {
        std::fprintf(stderr, "[R8-10] FAIL: null handle\n");
        return false;
    }
    e.submitRecoveryIntent(h1, snapshot);

    if (!stg8WaitUntil(45.0, [&] { return orch.hasDeferredRequest(); }))
    {
        std::fprintf(stderr, "[R8-10] FAIL: deferred state not reached\n");
        return false;
    }

    std::atomic<long long> spamCalls{0};
    std::thread spammer([&]
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        // Hammer the timer's entry point only while the RebuildThread has NOT provably
        // stopped. Once rebuildThreadStopped is published (right after join) the engine is
        // being torn down and continuing to call into it would be a test-side UB, not the
        // product window under test.
        while (!DeferredPublicationTestAccess::rebuildThreadStoppedOf(e)
               && std::chrono::steady_clock::now() < deadline)
        {
            orch.requestDeferredClear();
            spamCalls.fetch_add(1, std::memory_order_relaxed);
            std::this_thread::yield();
        }
    });

    // Let it hammer while the worker is alive, then tear down underneath it.
    stg8SleepPump(500);
    h.stop();
    spammer.join();

    bool ok = true;

    // Exactly one terminalization: the live count must be 0, not negative, and no deferred
    // residue may survive the teardown.
    const auto liveAfter = stg8LiveCount(e);
    if (liveAfter != 0)
    {
        std::fprintf(stderr, "[R8-10] FAIL: live count after teardown = %llu (want 0)\n",
                     (unsigned long long)liveAfter);
        ok = false;
    }
    if (orch.hasDeferredRequest())
    {
        std::fprintf(stderr, "[R8-10] FAIL: deferred residue after teardown\n");
        ok = false;
    }
    if (spamCalls.load() <= 0)
    {
        std::fprintf(stderr, "[R8-10] FAIL: spammer never ran (test vacuous)\n");
        ok = false;
    }

    if (!ok)
        return false;

    std::printf("STG8RecoveryObligationTests: PASS "
                "(R8-10 clear requests race teardown: single terminalization, no residue, "
                "%lld requests)\n",
                spamCalls.load());
    return true;
}


//   Stage 2 of the watchdog plan. `RuntimeHealthMonitor::tick()` calls
//   checkPublicationStall(), which reports EVENT_PUBLICATION_STALL when there is pending
//   work AND `isPublicationStalled()` (no committed publication for >= 30 s). Rather than
//   blocking a thread for 30 s, the friend seam ages the progress timestamp; the detector
//   itself (threshold comparison, engine timer, event transport, handler, latch, drain) is
//   the production path.
//   Pre-fix behaviour: the clear would leave the obligation live (L=1, hasDeferred=0).
//==============================================================================
namespace convo { namespace isr {
struct R8PublicationProgressTestAccess
{
    static void ageProgress(RuntimePublicationOrchestrator& o, std::uint64_t ageUs) noexcept
    {
        convo::publishAtomic(o.m_lastProgressTimestampUs,
                             convo::getCurrentTimeUs() - ageUs,
                             std::memory_order_release);
    }
};
} } // namespace convo::isr

static bool checkR8PublicationStallDetectorTerminalizes()
{
    Stg8Trace trace("checkR8PublicationStallDetectorTerminalizes");

    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[R8-9] harness start failed\n");
        return false;
    }

    AudioEngine& e = h.engine();
    auto& orch = DeferredPublicationTestAccess::orchestrator(e);
    auto& coord = DeferredPublicationTestAccess::coordinator(e);
    bool ok = true;

    if (!stg8EnsureAuthoritative(e))
    {
        std::fprintf(stderr, "[R8-9] FAIL: no authoritative runtime\n");
        return false;
    }
    stg8SleepPump(1000);

    DeferredPublicationTestAccess::setFadingRuntimePresent(e, true);
    auto snapshot = e.getCurrentBuildSnapshotForRecovery();
    snapshot.sealed = true;
    const auto h1 = stg8RegisterActive(e);
    if (h1.isNull())
    {
        std::fprintf(stderr, "[R8-9] FAIL: null handle\n");
        return false;
    }
    e.submitRecoveryIntent(h1, snapshot);

    if (!stg8WaitUntil(45.0, [&] { return orch.hasDeferredRequest(); }))
    {
        std::fprintf(stderr, "[R8-9] FAIL: deferred state not reached\n");
        return false;
    }

    const auto discardBefore = coord.recoveryObligationShutdownDiscardCount();

    // Age the publication-progress observation past the production threshold (30 s) and let
    // the engine's own health tick fire. 31 s > kPublicationStallThresholdUs.
    convo::isr::R8PublicationProgressTestAccess::ageProgress(orch, 31ull * 1000ull * 1000ull);

    // The detector must now report the stalled state (elapsed >= 30 s).
    if (!orch.isPublicationStalled())
    {
        std::fprintf(stderr, "[R8-9] FAIL: detector did not report stalled after aging\n");
        DeferredPublicationTestAccess::setFadingRuntimePresent(e, false);
        h.stop();
        return false;
    }

    if (!stg8WaitUntil(45.0, [&] {
            return !orch.hasDeferredRequest() && stg8LiveCount(e) == 0;
        }))
    {
        std::fprintf(stderr,
                     "[R8-9] FAIL: stall-detector clear did not terminalize "
                     "(hasDeferred=%d L=%llu)\n",
                     orch.hasDeferredRequest() ? 1 : 0,
                     (unsigned long long)stg8LiveCount(e));
        ok = false;
    }

    const auto discardAfter = coord.recoveryObligationShutdownDiscardCount();
    if (discardAfter != discardBefore)
    {
        std::fprintf(stderr,
                     "[R8-9] FAIL: mid-run clear counted as ShutdownDiscarded (%llu -> %llu)\n",
                     (unsigned long long)discardBefore, (unsigned long long)discardAfter);
        ok = false;
    }

    DeferredPublicationTestAccess::setFadingRuntimePresent(e, false);
    h.stop();

    if (!ok)
        return false;
    std::printf("STG8RecoveryObligationTests: PASS "
                "(R8-9 30s stall detector -> StaleSuperseded, L=0)\n");
    return true;
}


//==============================================================================
// ★ R8-8 (P1): watchdog → handler → latch → terminalization, end to end.
//   Stage 1 of the "real 30 s stall" plan: instead of waiting out the detector, a genuine
//   EVENT_PUBLICATION_STALL health event is pushed through the engine's own async transport
//   (enqueueHealthEvent → timer tick → onHealthEvent → requestDeferredClear → RebuildThread
//   drain). The 30 s *detection* is the only part not covered here.
//   Pre-fix behaviour: the clear left the obligation live (L=1, hasDeferred=0).
//==============================================================================
static bool checkR8WatchdogEventTerminalizesDeferred()
{
    Stg8Trace trace("checkR8WatchdogEventTerminalizesDeferred");
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[R8-8] harness start failed\n");
        return false;
    }

    AudioEngine& e = h.engine();
    auto& orch = DeferredPublicationTestAccess::orchestrator(e);
    auto& coord = DeferredPublicationTestAccess::coordinator(e);
    bool ok = true;

    if (!stg8EnsureAuthoritative(e))
    {
        std::fprintf(stderr, "[R8-8] FAIL: no authoritative runtime\n");
        return false;
    }
    stg8SleepPump(1000);

    DeferredPublicationTestAccess::setFadingRuntimePresent(e, true);
    auto snapshot = e.getCurrentBuildSnapshotForRecovery();
    snapshot.sealed = true;
    const auto h1 = stg8RegisterActive(e);
    if (h1.isNull())
    {
        std::fprintf(stderr, "[R8-8] FAIL: null handle\n");
        return false;
    }
    e.submitRecoveryIntent(h1, snapshot);

    if (!stg8WaitUntil(45.0, [&] { return orch.hasDeferredRequest(); }))
    {
        std::fprintf(stderr, "[R8-8] FAIL: deferred state not reached\n");
        return false;
    }

    const auto discardBefore = coord.recoveryObligationShutdownDiscardCount();

    // Drive the production async path: SPSC queue -> timer tick -> onHealthEvent.
    convo::HealthEvent stall{};
    stall.timestampUs = convo::getCurrentTimeUs();
    stall.severity = convo::HealthEvent::Severity::Error;
    stall.eventCode = convo::EVENT_PUBLICATION_STALL;
    e.enqueueHealthEvent(stall);

    if (!stg8WaitUntil(45.0, [&] {
            return !orch.hasDeferredRequest() && stg8LiveCount(e) == 0;
        }))
    {
        std::fprintf(stderr,
                     "[R8-8] FAIL: watchdog clear did not terminalize "
                     "(hasDeferred=%d L=%llu)\n",
                     orch.hasDeferredRequest() ? 1 : 0,
                     (unsigned long long)stg8LiveCount(e));
        ok = false;
    }

    const auto discardAfter = coord.recoveryObligationShutdownDiscardCount();
    if (discardAfter != discardBefore)
    {
        std::fprintf(stderr,
                     "[R8-8] FAIL: mid-run watchdog clear counted as ShutdownDiscarded "
                     "(%llu -> %llu)\n",
                     (unsigned long long)discardBefore, (unsigned long long)discardAfter);
        ok = false;
    }

    DeferredPublicationTestAccess::setFadingRuntimePresent(e, false);
    h.stop();

    if (!ok)
        return false;
    std::printf("STG8RecoveryObligationTests: PASS (R8-8 watchdog event -> StaleSuperseded, L=0)\n");
    return true;
}


//   rebuildThreadShouldExit is published BEFORE join(); it therefore does not prove the
//   thread stopped. rebuildThreadStopped is published AFTER join() and is the only flag the
//   C1 synchronous clear is allowed to trust. Observed invariants:
//     while running : shouldExit=false, stopped=false, rebuild id valid, this thread is not
//                     the rebuild thread
//     after stop()  : shouldExit=true,  stopped=true,  rebuild id empty (get_id() after join),
//                     isRebuildThread() false for every thread
//==============================================================================
static bool checkR8C1StoppedFlagOrdering()
{
    Stg8Trace trace("checkR8C1StoppedFlagOrdering");
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[R8-6] harness start failed\n");
        return false;
    }

    AudioEngine& e = h.engine();

    if (DeferredPublicationTestAccess::rebuildThreadShouldExitOf(e))
    {
        std::fprintf(stderr, "[R8-6] FAIL: shouldExit set while running\n");
        return false;
    }
    if (DeferredPublicationTestAccess::rebuildThreadStoppedOf(e))
    {
        std::fprintf(stderr, "[R8-6] FAIL: stopped set while running\n");
        return false;
    }
    if (DeferredPublicationTestAccess::rebuildThreadIdIsEmpty(e))
    {
        std::fprintf(stderr, "[R8-6] FAIL: no rebuild thread id while running\n");
        return false;
    }
    if (DeferredPublicationTestAccess::isRebuildThreadOf(e))
    {
        std::fprintf(stderr, "[R8-6] FAIL: test thread claims RebuildThread ownership\n");
        return false;
    }
    if (DeferredPublicationTestAccess::rebuildThreadTokenOf(e) == 0)
    {
        std::fprintf(stderr, "[R8-6] FAIL: no ownership token while running\n");
        return false;
    }

    h.stop();

    if (!DeferredPublicationTestAccess::rebuildThreadShouldExitOf(e))
    {
        std::fprintf(stderr, "[R8-6] FAIL: shouldExit not set after stop\n");
        return false;
    }
    if (!DeferredPublicationTestAccess::rebuildThreadStoppedOf(e))
    {
        std::fprintf(stderr, "[R8-6] FAIL: stopped not set after join\n");
        return false;
    }
    if (!DeferredPublicationTestAccess::rebuildThreadIdIsEmpty(e))
    {
        std::fprintf(stderr, "[R8-6] FAIL: rebuild id still valid after join\n");
        return false;
    }
    if (DeferredPublicationTestAccess::isRebuildThreadOf(e))
    {
        std::fprintf(stderr, "[R8-6] FAIL: ownership still claimable after join\n");
        return false;
    }
    if (DeferredPublicationTestAccess::rebuildThreadTokenOf(e) != 0)
    {
        std::fprintf(stderr, "[R8-6] FAIL: ownership token not cleared on exit\n");
        return false;
    }

    std::printf("STG8RecoveryObligationTests: PASS "
                "(R8-6 stopped published after join; ownership unclaimable after)\n");
    return true;
}

//==============================================================================
// ★ R8-7 (P1): the RebuildThread contract is enforced in Release, not only by jassert.
//   Preconditions: a deferred slot is live (hasDeferred == true, L == 1) and this thread is
//   NOT the RebuildThread. The plain deferred members must be unreachable:
//     peekDeferred()                -> nullopt   (and hasDeferred unchanged)
//     tryConsumeRedriveBudget()     -> false
//     drainDeferredClearIfRequested -> false (no latch consumed)
//   Then a generation bump lets the real RebuildThread discard the obligation, which proves
//   the refusal did not strand the engine.
//==============================================================================
static bool checkR8ForeignThreadRefused()
{
    Stg8Trace trace("checkR8ForeignThreadRefused");
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[R8-7] harness start failed\n");
        return false;
    }

    AudioEngine& e = h.engine();
    auto& orch = DeferredPublicationTestAccess::orchestrator(e);

    if (!stg8EnsureAuthoritative(e))
    {
        std::fprintf(stderr, "[R8-7] FAIL: no authoritative runtime\n");
        return false;
    }
    stg8SleepPump(1000);

    DeferredPublicationTestAccess::setFadingRuntimePresent(e, true);
    auto snapshot = e.getCurrentBuildSnapshotForRecovery();
    snapshot.sealed = true;
    const auto h1 = stg8RegisterActive(e);
    if (h1.isNull())
    {
        std::fprintf(stderr, "[R8-7] FAIL: null handle\n");
        return false;
    }
    e.submitRecoveryIntent(h1, snapshot);

    if (!stg8WaitUntil(45.0, [&] { return orch.hasDeferredRequest(); }))
    {
        std::fprintf(stderr, "[R8-7] FAIL: deferred state not reached\n");
        return false;
    }

    bool ok = true;
    if (DeferredPublicationTestAccess::isRebuildThreadOf(e))
    {
        std::fprintf(stderr, "[R8-7] FAIL: test thread is the RebuildThread (test invalid)\n");
        return false;
    }

    const auto view = orch.peekDeferred();
    if (view.has_value())
    {
        std::fprintf(stderr, "[R8-7] FAIL: foreign thread obtained a DeferredPublishView\n");
        ok = false;
    }
    if (!orch.hasDeferredRequest())
    {
        std::fprintf(stderr, "[R8-7] FAIL: refusal consumed the deferred slot\n");
        ok = false;
    }
    if (orch.tryConsumeRedriveBudget())
    {
        std::fprintf(stderr, "[R8-7] FAIL: foreign thread consumed redrive budget\n");
        ok = false;
    }
    if (orch.drainDeferredClearIfRequested())
    {
        std::fprintf(stderr, "[R8-7] FAIL: foreign thread drained the clear latch\n");
        ok = false;
    }

    // Cleanup through the real owner: generation bump => Discard on the next tick.
    DeferredPublicationTestAccess::bumpRebuildGeneration(e);
    if (!stg8WaitUntil(45.0, [&] { return !orch.hasDeferredRequest() && stg8LiveCount(e) == 0; }))
    {
        std::fprintf(stderr, "[R8-7] FAIL: obligation not terminalized after refusal "
                             "(hasDeferred=%d L=%llu)\n",
                     orch.hasDeferredRequest() ? 1 : 0,
                     (unsigned long long)stg8LiveCount(e));
        ok = false;
    }

    h.stop();
    if (ok)
        std::printf("STG8RecoveryObligationTests: PASS "
                    "(R8-7 foreign thread refused: peek=nullopt, slot not consumed, no stranding)\n");
    return ok;
}
int runSTG8RecoveryObligationTests()
{
    bool stg8ok = true;
    // ★ STG-8-D1 (Discard 終端・新 CTest 登録なし)
    if (!checkSTG81DiscardTerminalizesRecovery())
    {
        std::fprintf(stderr, "FAIL: checkSTG81DiscardTerminalizesRecovery\n");
        stg8ok = false;
    }
    // ★ STG-8-D2 (overwrite 終端・新 CTest 登録なし)
    if (!checkSTG82OverwriteTerminalizesEvicted())
    {
        std::fprintf(stderr, "FAIL: checkSTG82OverwriteTerminalizesEvicted\n");
        stg8ok = false;
    }
    // ★ STG-8-D2b (同一 obligation 維持・新 CTest 登録なし)
    if (!checkSTG82bSameObligationNotResolved())
    {
        std::fprintf(stderr, "FAIL: checkSTG82bSameObligationNotResolved\n");
        stg8ok = false;
    }
    if (!checkSTG8ClearTerminalizesObligation())
    {
        std::fprintf(stderr, "FAIL: checkSTG8ClearTerminalizesObligation\n");
        stg8ok = false;
    }
    // R8-2: repeated clear must not double-terminalize / double-release.
    if (!checkSTG8ClearIsIdempotent())
    {
        std::fprintf(stderr, "FAIL: checkSTG8ClearIsIdempotent\n");
        stg8ok = false;
    }
    // R8-3: mid-run clear must record StaleSuperseded, not ShutdownDiscarded.
    if (!checkSTG8ClearTerminalReasonIsStale())
    {
        std::fprintf(stderr, "FAIL: checkSTG8ClearTerminalReasonIsStale\n");
        stg8ok = false;
    }
    // R8-4: the shutdown branch must record ShutdownDiscarded.
    if (!checkSTG8ClearUsesShutdownReason())
    {
        std::fprintf(stderr, "FAIL: checkSTG8ClearUsesShutdownReason\n");
        stg8ok = false;
    }
    // R8-5: the production entry point (watchdog -> requestDeferredClear -> drain).
    if (!checkSTG8RequestDeferredClearDrains())
    {
        std::fprintf(stderr, "FAIL: checkSTG8RequestDeferredClearDrains\n");
        stg8ok = false;
    }
    // R8-6: rebuildThreadStopped is published only after join(); it is the C1 predicate.
    if (!checkR8C1StoppedFlagOrdering())
    {
        std::fprintf(stderr, "FAIL: checkR8C1StoppedFlagOrdering\n");
        stg8ok = false;
    }
    // R8-7: the RebuildThread contract is enforced in Release (refuse, not jassert-only).
    if (!checkR8ForeignThreadRefused())
    {
        std::fprintf(stderr, "FAIL: checkR8ForeignThreadRefused\n");
        stg8ok = false;
    }
    // R8-8: watchdog event -> handler -> latch -> terminalization (stage 1 of the 30 s plan).
    if (!checkR8WatchdogEventTerminalizesDeferred())
    {
        std::fprintf(stderr, "FAIL: checkR8WatchdogEventTerminalizesDeferred\n");
        stg8ok = false;
    }
    // R8-9: the real 30 s stall detector, driven with a synthetic clock (stage 2).
    if (!checkR8PublicationStallDetectorTerminalizes())
    {
        std::fprintf(stderr, "FAIL: checkR8PublicationStallDetectorTerminalizes\n");
        stg8ok = false;
    }
    // R8-10: clear requests racing teardown (the exact window the C1 fix targets).
    if (!checkR8C1RaceWithTeardown())
    {
        std::fprintf(stderr, "FAIL: checkR8C1RaceWithTeardown\n");
        stg8ok = false;
    }
    // ★ STG-8-D3 (pressure signal redrive・新 CTest 登録なし)
    // R8-12: the real 30 s stall watchdog on the wall clock (no synthetic aging).
    if (!checkR8RealTimeStallDetectorTerminalizes())
    {
        std::fprintf(stderr, "FAIL: checkR8RealTimeStallDetectorTerminalizes\n");
        stg8ok = false;
    }
    if (!checkSTG83PressureSignalRedrives())
    {
        std::fprintf(stderr, "FAIL: checkSTG83PressureSignalRedrives\n");
        stg8ok = false;
    }
    if (!stg8ok)
        return 1;
    return 0;
}
