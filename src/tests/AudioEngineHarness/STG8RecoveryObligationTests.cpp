// STG8RecoveryObligationTests.cpp — STG-8-D1/D2/D3 regression.
//   Recovery obligation 終端欠落 3 件の回帰テスト（AudioEngineHarness 内 subtest）。
//   新 CTest 登録なし（既存 harness exe のサブテストとして run 側から呼出）。
//   方針: 実エンジン経路のみ（R30 vehicle の submitRecoveryIntent 直呼び pattern）。
//   obligation ID を数値で追跡せず、liveLogicalRecoveryObligationCount (L)・
//   hasDeferredRequest・deferredOverwriteCount・recoveryRetryRedriveCount・
//   publication sequence の公開観測のみで oracle を構成する。
//   L は Live のみを数えるため、L==0 ⟺ 当該 obligation 終端済みと同値。

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <thread>

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
static bool checkSTG82bSameObligationNotResolved()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-8-D2b] harness start failed\n");
        return false;
    }
    AudioEngine& e = h.engine();
    auto& orch = DeferredPublicationTestAccess::orchestrator(e);
    bool ok = true;

    if (!stg8EnsureAuthoritative(e))
    {
        std::fprintf(stderr, "[STG-8-D2b] FAIL: no authoritative runtime\n");
        return false;
    }
    stg8SleepPump(1000);

    DeferredPublicationTestAccess::setFadingRuntimePresent(e, true);
    auto snapshot = e.getCurrentBuildSnapshotForRecovery();
    snapshot.sealed = true;
    const auto h1 = stg8RegisterActive(e);
    if (h1.isNull())
    {
        std::fprintf(stderr, "[STG-8-D2b] FAIL: null handle\n");
        return false;
    }
    e.submitRecoveryIntent(h1, snapshot);
    if (!stg8WaitUntil(30.0, [&] { return stg8LiveCount(e) == 1; }))
    {
        std::fprintf(stderr, "[STG-8-D2b] FAIL: O1 not admitted\n");
        return false;
    }
    if (!stg8WaitUntil(45.0, [&] { return orch.hasDeferredRequest(); }))
    {
        std::fprintf(stderr, "[STG-8-D2b] FAIL: O1 deferred state not reached\n");
        return false;
    }
    // 同一 handle 再 submit（coalesce → 同 O1 の新表現）。第二 build→submit に25秒猶予。
    e.submitRecoveryIntent(h1, snapshot);
    stg8SleepPump(25000);
    if (stg8LiveCount(e) != 1 || !orch.hasDeferredRequest())
    {
        std::fprintf(stderr, "[STG-8-D2b] FAIL: over-terminalized (L=%llu hasDeferred=%d)\n",
                     (unsigned long long)stg8LiveCount(e), orch.hasDeferredRequest() ? 1 : 0);
        ok = false;
    }

    DeferredPublicationTestAccess::setFadingRuntimePresent(e, false);
    h.stop();
    if (!ok)
        return false;
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
