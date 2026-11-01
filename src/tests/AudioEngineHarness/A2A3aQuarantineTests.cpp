// A2A3aQuarantineTests.cpp
// ★ A2 → A3-a (2026-11-01): 二重解放（V1）と誤状態処理（V2）の分離検証。
//
// 実行: A2A3A_TEST=1 AudioEngineHarness.exe
//   （既定スイートには影響しない opt-in 入口）

#include <JuceHeader.h>

#include "audioengine/ISRDSPHandle.h"

#include <atomic>
#include <cstdio>
#include <thread>

using convo::isr::DSPHandleRuntime;

namespace {

int g_failures = 0;

void expectTrue(bool condition, const char* what)
{
    if (condition)
    {
        std::printf("[A2A3A] ok   %s\n", what);
    }
    else
    {
        std::printf("[A2A3A] FAIL %s\n", what);
        ++g_failures;
    }
}

} // namespace

int runA2A3aQuarantineTests()
{
    DSPHandleRuntime rt;

    alignas(64) static int payloadA {};
    alignas(64) static int payloadB {};
    alignas(64) static int payloadC {};

    // ---- V2: pin 残留では破棄せず、スロット単位で deferred に記録する -------------
    const auto handleA = rt.create(&payloadA);
    const bool createdA = !handleA.isNull();
    expectTrue(createdA, "create returns a handle");

    rt.quarantine(handleA);
    {
        auto pin = rt.tryPin(handleA);
        const bool pinned = static_cast<bool>(pin);
        expectTrue(pinned, "a pin can be taken");

        const bool destroyedWhilePinned = rt.destroyQuarantineSlot(handleA.slot, 0);
        expectTrue(!destroyedWhilePinned, "destroy with a live pin returns false (deferred)");

        const bool marked = rt.isQuarantineDeferred(handleA.slot);
        expectTrue(marked, "the slot is marked deferred (A3-a per-slot mask)");

        const auto deferredWhilePinned = rt.deferredQuarantineDestroyCount();
        expectTrue(deferredWhilePinned == 1, "deferred count == 1 (gate contract preserved)");
    }

    // ---- V1: pin 解放後、retry が 1 回だけ回収する ------------------------------
    const auto reclaimedFirst = rt.retryDeferredQuarantineDestroy();
    expectTrue(reclaimedFirst == 1, "retry destroys the marked slot exactly once");

    const bool stillMarked = rt.isQuarantineDeferred(handleA.slot);
    expectTrue(!stillMarked, "the mark is cleared after reclaim");

    const auto deferredAfterReclaim = rt.deferredQuarantineDestroyCount();
    expectTrue(deferredAfterReclaim == 0, "deferred count returns to 0");

    // V1: 2 回目の retry で二重解放しない
    const auto reclaimedSecond = rt.retryDeferredQuarantineDestroy();
    expectTrue(reclaimedSecond == 0, "a second retry is a no-op (no double free)");

    const bool destroyedAgain = rt.destroyQuarantineSlot(handleA.slot, 0);
    expectTrue(!destroyedAgain, "destroy on a reclaimed slot returns false");

    // ---- A3-a: マークされていない Quarantined スロットは retry が触らない --------
    const auto handleB = rt.create(&payloadB);
    rt.quarantine(handleB);

    const auto deferredUnmarked = rt.deferredQuarantineDestroyCount();
    expectTrue(deferredUnmarked == 0,
               "an unmarked quarantine does not raise the deferred count");

    const auto reclaimedUnmarked = rt.retryDeferredQuarantineDestroy();
    expectTrue(reclaimedUnmarked == 0, "retry ignores unmarked slots");

    // ---- V1: 2 スレッド同時破棄で勝者が 1 人だけ ---------------------------------
    const auto handleC = rt.create(&payloadC);
    rt.quarantine(handleC);
    const auto slotC = handleC.slot;

    std::atomic<int> winners { 0 };
    auto attempt = [&rt, slotC, &winners]() {
        if (rt.destroyQuarantineSlot(slotC, 0))
            winners.fetch_add(1, std::memory_order_relaxed);
    };

    std::thread t1(attempt);
    std::thread t2(attempt);
    t1.join();
    t2.join();

    const int winCount = winners.load();
    expectTrue(winCount == 1, "concurrent destroy has exactly one winner");

    const auto deferredC = rt.deferredQuarantineDestroyCount();
    expectTrue(deferredC == 0, "no deferred residue after concurrent reclaim");

    std::printf("[A2A3A] RESULT failures=%d\n", g_failures);
    std::fflush(stdout);
    return g_failures == 0 ? 0 : 1;
}
