// DeferredFlowIntegrationTests.cpp
// View/Orchestrator deferred-flow integration test（ADR-C4 / design-D4）。
//
// 検証対象: submitPublishRequest が deferred 自動 enqueue し、RebuildThread が
//   consume → finishView → submit し直す Ready 経路の終端状態。
//   Orchestrator は AudioEngine の private メンバ（friend 限定）のため、
//   テスト専用 Friend Test Access（DeferredPublicationTestAccess）を経由して観測する。
//
// 方針:
//   - 直接 peek/consume/discard は RebuildThread 専有（jassert 付き）のため
//     テストスレッドからは呼ばない。エンジンの実経路（requestRebuild →
//     submitPublishRequest → CoordinatorLoop → executePublish）を駆動し、
//     未ガードの atomic 観測 API（hasDeferredRequest / getPublicationBacklogCount /
//     deferredOverwriteCount）で Ready 経路の終端状態（slot drain）を証明する。
//   - MovedFrom / copy 禁止は型レベル（static_assert）で証明する。
//   - デストラクタ fail-fast（Valid のまま View 破棄 → jassert）は
//     RuntimePublicationOrchestrator.cpp のデストラクタ実装（jassert）により
//     コード検査で担保（テストスレッドからはスレッド専有 jassert に阻まれて到達不能）。

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <functional>
#include <chrono>
#include <thread>
#include <type_traits>

#include "AudioEngineHarness.h"
#include "DeferredPublicationTestAccess.h"
#include "audioengine/RuntimePublicationState.h"

// ── 型レベル証明: DeferredPublishView は move-only（Phase-1 再整列の実体）──
static_assert(!std::is_copy_constructible_v<convo::isr::DeferredPublishView>,
              "DeferredPublishView must be move-only (Phase-1)");
static_assert(!std::is_copy_assignable_v<convo::isr::DeferredPublishView>,
              "DeferredPublishView must be move-only (Phase-1)");
static_assert(std::is_move_constructible_v<convo::isr::DeferredPublishView>,
              "DeferredPublishView must be move-constructible (Phase-1)");
static_assert(std::is_move_assignable_v<convo::isr::DeferredPublishView>,
              "DeferredPublishView must be move-assignable (Phase-1)");

namespace {

bool waitUntil(double timeoutSec, const std::function<bool()>& pred)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeoutSec);
    while (std::chrono::steady_clock::now() < deadline)
    {
        if (pred())
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return pred();
}

// ── 1. Deferred → Ready → Drain の全状態遷移（決定論的証明）──
//   ADR-C4: enqueueDeferred → peekDeferred → consume → finishView → submit。
//   前提条件フック（testFadingRuntimePresent）で DeferredFadingActive を決定論的に
//   発生させ、実 rebuild → RebuildThread の実 submitPublishRequest を deferred に落とす。
//   検証: [Deferred → hasDeferred==true] → [前提解除 → Ready → consume → finishView
//   → hasDeferred==false] → [resubmit publish 完了（seq 前進）]。
bool testDeferredReadyPathDrainsSlot()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
        return false;

    AudioEngine& e = h.engine();
    auto& orch = DeferredPublicationTestAccess::orchestrator(e);

    const auto* w0 = e.observePublishedWorld();
    const unsigned long long baseSeq = w0 ? static_cast<unsigned long long>(w0->publication.sequenceId) : 0ULL;

    // 前提条件: fading runtime 存在 → submit は DeferredFadingActive で deferred に入る
    DeferredPublicationTestAccess::setFadingRuntimePresent(e, true);

    // 実 rebuild → RebuildThread が実 request で submitPublishRequest を呼ぶ → deferred enqueue
    e.requestRebuild(convo::RebuildKind::Structural);

    // [Deferred → hasDeferred==true]: 決定論的に到達する
    const bool sawDeferred = waitUntil(45.0, [&] { return orch.hasDeferredRequest(); });
    if (!sawDeferred)
    {
        std::fprintf(stderr, "FAIL: Deferred state not reached (hasDeferredRequest stayed false)\n");
        return false;
    }

    // 前提条件解除 → 次 tick の processDeferredAdmission が Ready → consume → finishView → resubmit
    DeferredPublicationTestAccess::setFadingRuntimePresent(e, false);

    // [Ready → consume → finishView → hasDeferred==false]: slot drain（Ownership 解放）
    const bool drained = waitUntil(30.0, [&] {
        return !orch.hasDeferredRequest() && orch.getPublicationBacklogCount() == 0;
    });
    if (!drained)
    {
        std::fprintf(stderr, "FAIL: not drained (hasDeferred=%d backlog=%llu)\n",
                     orch.hasDeferredRequest() ? 1 : 0,
                     static_cast<unsigned long long>(orch.getPublicationBacklogCount()));
        return false;
    }

    // consume で取り出した request が resubmit され publish 完了（seq 前進）
    const bool progressed = waitUntil(30.0, [&] {
        const auto* w = e.observePublishedWorld();
        return w != nullptr && w->publication.sequenceId > baseSeq;
    });
    if (!progressed)
    {
        std::fprintf(stderr, "FAIL: deferred resubmit did not publish (base seq=%llu)\n",
                     static_cast<unsigned long long>(baseSeq));
        return false;
    }

    std::printf("INFO: Deferred->hasDeferred==true->Ready->consume->finishView->drain proved\n");
    return true;
}

// ── 2. Deferred サイクル後の backlog 完全排出 ──
//   前提条件フックで 2 回の deferred サイクルを回し、終端で slot と backlog が
//   完全に排出されること（Slot/Ownership リークなし）を検証する。
bool testDeferredBacklogDrainsCompletely()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
        return false;

    AudioEngine& e = h.engine();
    auto& orch = DeferredPublicationTestAccess::orchestrator(e);

    for (int cycle = 0; cycle < 2; ++cycle)
    {
        DeferredPublicationTestAccess::setFadingRuntimePresent(e, true);
        e.requestRebuild(convo::RebuildKind::Structural);

        if (!waitUntil(45.0, [&] { return orch.hasDeferredRequest(); }))
        {
            std::fprintf(stderr, "FAIL: cycle %d did not defer\n", cycle);
            return false;
        }
        DeferredPublicationTestAccess::setFadingRuntimePresent(e, false);

        if (!waitUntil(30.0, [&] {
                return !orch.hasDeferredRequest() && orch.getPublicationBacklogCount() == 0;
            }))
        {
            std::fprintf(stderr, "FAIL: cycle %d not drained (backlog=%llu)\n", cycle,
                         static_cast<unsigned long long>(orch.getPublicationBacklogCount()));
            return false;
        }
    }

    std::printf("INFO: 2x deferred cycles drained (no slot/ownership leak)\n");
    return true;
}

} // namespace

// main 側（PublishPipelineIntegrationTests.cpp）から呼ばれるエントリ
int runDeferredFlowIntegrationTests()
{
    if (!testDeferredReadyPathDrainsSlot())
    {
        std::fprintf(stderr, "FAIL: testDeferredReadyPathDrainsSlot\n");
        return 1;
    }
    if (!testDeferredBacklogDrainsCompletely())
    {
        std::fprintf(stderr, "FAIL: testDeferredBacklogDrainsCompletely\n");
        return 1;
    }
    std::printf("DeferredFlowIntegrationTests: PASS\n");
    return 0;
}
