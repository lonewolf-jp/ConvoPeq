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
                    " longestBad_ms=%lld\n",
                    iteration, samples.size(), l.tMs, deferredTransitions, badSamples,
                    maxBadStreakMs);
        std::printf("[D2b-CNT] iter=%d start{t=%lld sub=%llu pub=%llu ret=%llu stage=%d age=%llu}"
                    " end{t=%lld sub=%llu pub=%llu ret=%llu stage=%d age=%llu}\n",
                    iteration, f.tMs, f.submitted, f.published, f.retired, f.stage,
                    f.maxDeferredAgeMs, l.tMs, l.submitted, l.published, l.retired,
                    l.stage, l.maxDeferredAgeMs);
        std::printf("[D2b-BAD] iter=%d streak[%lld..%lld] sub=+%llu pub=+%llu ret=+%llu"
                    " stage %d->%d age %llu->%llu stalled=%d maxAge=%llu maxPending=%llu"
                    " maxBacklog=%llu\n",
                    iteration, firstBadMs, lastBadMs,
                    lastBad.submitted - firstBad.submitted,
                    lastBad.published - firstBad.published,
                    lastBad.retired - firstBad.retired,
                    firstBad.stage, lastBad.stage,
                    firstBad.maxDeferredAgeMs, lastBad.maxDeferredAgeMs,
                    stalledSamples, maxAge, maxPending, maxBacklog);
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
// ★ STG-8-D3: RejectedPressure の transport recovery が signal→redrive で再送される。
//   fading OFF・throttle ON → recovery submit (L=1, Transport) → Builder build→
//   submit → RejectedPressure →（修正: signal）→ throttle OFF → adjudicate→None→
//   redrive→再 Transport→再 build→Accepted→Published。
//   期待: redriveCount 増加 && L==0 && sequence 前進。
//   修正前は redrive 0 件・L==1 のまま停滞し FAIL する。
//==============================================================================
static bool checkSTG83PressureSignalRedrives()
{
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

} // namespace

// main 側（PublishPipelineIntegrationTests.cpp）から呼ばれるエントリ。
// 新 CTest 登録なし（AudioEngineHarness 既存 exe のサブテスト）。
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
    // ★ STG-8-D3 (pressure signal redrive・新 CTest 登録なし)
    if (!checkSTG83PressureSignalRedrives())
    {
        std::fprintf(stderr, "FAIL: checkSTG83PressureSignalRedrives\n");
        stg8ok = false;
    }
    if (!stg8ok)
        return 1;
    return 0;
}
