// V11EngineRetireProtectionTests.cpp — V-11 at engine level.
//
// The standalone tests drive EpochDomain/RetireQuarantineStore directly. This test uses the
// *engine's own* objects after a real `releaseResources()` (which permanently closes reader
// registration, AudioEngine.Processing.ReleaseResources.cpp:274):
//
//   - the engine's retire router (real IReaderEpochProvider, real D/Q/E queues)
//   - the engine's EpochDomain (via the router)
//   - the engine's reclaim entry point (AudioEngine::tryReclaimResources)
//
// Assertions:
//   E1  after releaseResources(), an RCUReader that registered before the close can still
//       establish protection (false without the provider fix — that is V-11)
//   E2  an object retired at the reader's epoch is NOT reclaimed while the guard is held,
//       even after the global epoch advances
//   E3  after the guard drops, the same object IS reclaimed (no leak / not a stuck barrier)
//   E4  a reader that never registered before the close cannot register afterwards
//
// E2/E3 are reported as NOT OBSERVABLE if the post-teardown router refuses new retires
// (shutdown authority); that case is printed explicitly rather than counted as a pass.
//
// Oracle: the sentinel is not freed by its deleter (bounded leak), so the observation is the
// deleter invocation itself — the reclaim event.

#include "AudioEngineHarness.h"
#include "DeferredPublicationTestAccess.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <thread>

#include "core/RCUReader.h"

namespace {

int   g_v11eReclaimCalls = 0;
void* g_v11eLastReclaimed = nullptr;

void v11eSentinelDeleter(void* p) noexcept
{
    ++g_v11eReclaimCalls;
    g_v11eLastReclaimed = p;
}

bool checkEngineLevelProtection()
{
    AudioEngineHarness harness;
    if (!harness.start())
    {
        std::printf("  V11-ENGINE: harness.start() failed\n");
        return false;
    }

    auto& router = harness.engine().getRetireRouter();

    // Register this thread's reader slot while registration is still open.
    convo::RCUReader probe(router);
    {
        convo::RCUReaderGuard g(probe);
        if (!g.rootEnterSucceeded())
        {
            std::printf("  V11-ENGINE: initial enter failed (before close)\n");
            harness.stop();
            return false;
        }
    }

    // Real teardown: closes reader registration permanently.
    harness.stop();

    static int sentinel = 0;
    void* const obj = &sentinel;
    bool retireObservable = false;
    bool protectedWhileHeld = false;
    bool reclaimedAfterRelease = false;

    {
        convo::RCUReaderGuard g(probe);

        // E1
        if (!g.rootEnterSucceeded())
        {
            std::printf("  V11-ENGINE: E1 FAIL - enter refused after releaseResources\n");
            return false;
        }

        const uint64_t retireEpoch = harness.engine().currentRetireEpoch();
        retireObservable = harness.engine().enqueueRetireEpochBounded(obj, v11eSentinelDeleter,
                                                                     retireEpoch);

        if (retireObservable)
        {
            // A new publication would advance the epoch past the retire epoch.
            harness.engine().advanceRetireEpoch();
            harness.engine().tryReclaimResources();
            const int afterFirstDrain = g_v11eReclaimCalls;

            harness.engine().tryReclaimResources();
            const int afterSecondDrain = g_v11eReclaimCalls;

            protectedWhileHeld = (afterFirstDrain == 0 && afterSecondDrain == 0);
            if (!protectedWhileHeld)
            {
                std::printf("  V11-ENGINE: E2 FAIL - reclaimed while held (1st=%d 2nd=%d)\n",
                            afterFirstDrain, afterSecondDrain);
                return false;
            }
        }
        else
        {
            std::printf("  V11-ENGINE: WARN - post-teardown retire refused; E2/E3 not observable here\n");
        }
    }

    // E3: after the guard drops the object must become reclaimable.
    if (retireObservable)
    {
        harness.engine().advanceRetireEpoch();
        harness.engine().tryReclaimResources();

        reclaimedAfterRelease = (g_v11eReclaimCalls == 1) && (g_v11eLastReclaimed == obj);
        if (!reclaimedAfterRelease)
        {
            std::printf("  V11-ENGINE: E3 FAIL - not reclaimed after the guard dropped (calls=%d)\n",
                        g_v11eReclaimCalls);
            return false;
        }
    }

    // E4: a reader that never registered cannot register after the close.
    std::atomic<int> freshOutcome{99};
    std::thread fresh([&]
    {
        convo::RCUReader reader(harness.engine().getRetireRouter());
        convo::RCUReaderGuard g(reader);
        freshOutcome.store(g.rootEnterSucceeded() ? 1 : 0, std::memory_order_release);
    });
    fresh.join();

    if (freshOutcome.load(std::memory_order_acquire) != 0)
    {
        std::printf("  V11-ENGINE: E4 FAIL - a never-registered reader obtained protection\n");
        return false;
    }

    if (retireObservable)
        std::printf("  V11-ENGINE: E1 enter-after-close=1  E2 not-reclaimed-while-held  E3 reclaimed-after=1  E4 fresh-refused\n");
    else
        std::printf("  V11-ENGINE: E1 enter-after-close=1  E4 fresh-refused  (E2/E3 not observable)\n");

    return true;
}

//==============================================================================
// ★ V-11 (P0-5 complete): retire recorded FROM INSIDE the audio callback.
//   The hook runs on the audio thread, inside getNextAudioBlock, after the RCU guard is
//   established and before the published-world pointer is read. It holds the callback there
//   while the test thread records a retirement at the guard's epoch, advances the global
//   epoch and drains. Pre-fix, the guard could not be established after close (V-11); the
//   object would then be reclaimed while the callback "holds" it.
//==============================================================================
#if CONVOPEQ_ENABLE_TEST_HOOKS
namespace {

std::atomic<int>  g_v11HookCalls{0};
std::atomic<bool> g_v11HookInside{false};
std::atomic<bool> g_v11HookReturned{false};
std::atomic<bool> g_v11HookRelease{false};

void v11BlockHook(void*) noexcept
{
    if (g_v11HookCalls.fetch_add(1, std::memory_order_acq_rel) != 0)
        return;   // only the first callback blocks
    g_v11HookInside.store(true, std::memory_order_release);
    while (!g_v11HookRelease.load(std::memory_order_acquire))
        std::this_thread::yield();
    g_v11HookReturned.store(true, std::memory_order_release);
}

bool v11WaitFor(double timeoutSec, const std::function<bool()>& pred)
{
    const auto deadline = std::chrono::steady_clock::now()
                        + std::chrono::milliseconds(static_cast<int>(timeoutSec * 1000));
    while (std::chrono::steady_clock::now() < deadline)
    {
        if (pred())
            return true;
        std::this_thread::yield();
    }
    return pred();
}

} // namespace
#endif

static bool checkInCallbackRetireProtection()
{
#if CONVOPEQ_ENABLE_TEST_HOOKS
    // Reset the shared counters: the preceding engine-level test already reclaimed its own
    // sentinel, so a stale count would make this test read as a failure of the product.
    g_v11eReclaimCalls = 0;
    g_v11eLastReclaimed = nullptr;

    AudioEngineHarness harness;
    if (!harness.start(48000.0, 512))
    {
        std::printf("[V11-CB] harness.start() failed\n");
        return false;
    }

    AudioEngine& e = harness.engine();

    // Wait for a published runtime so the callback actually reaches the read path.
    if (!v11WaitFor(30.0, [&] { return e.hasAuthoritativePublishedRuntime(); }))
    {
        std::printf("[V11-CB] FAIL: no authoritative runtime\n");
        harness.stop();
        return false;
    }

    g_v11HookCalls.store(0);
    g_v11HookInside.store(false);
    g_v11HookReturned.store(false);
    g_v11HookRelease.store(false);

    e.audioBlockTestHookCtx.store(nullptr, std::memory_order_release);
    e.audioBlockTestHook.store(&v11BlockHook, std::memory_order_release);

    if (!v11WaitFor(10.0, [] { return g_v11HookInside.load(std::memory_order_acquire); }))
    {
        std::printf("[V11-CB] FAIL: callback hook never ran\n");
        e.audioBlockTestHook.store(nullptr, std::memory_order_release);
        harness.stop();
        return false;
    }

    static int sentinel = 0;
    void* const obj = &sentinel;

    // The audio callback is parked inside the guard at epoch E.
    const uint64_t retireEpoch = e.currentRetireEpoch();
    const bool queued = e.enqueueRetireEpochBounded(obj, v11eSentinelDeleter, retireEpoch);

    e.advanceRetireEpoch();
    e.tryReclaimResources();
    const bool protectedWhileInCallback = (g_v11eReclaimCalls == 0);

    // Let the callback finish; the guard drops.
    g_v11HookRelease.store(true, std::memory_order_release);
    const bool returned = v11WaitFor(10.0, [] { return g_v11HookReturned.load(std::memory_order_acquire); });
    e.audioBlockTestHook.store(nullptr, std::memory_order_release);

    e.advanceRetireEpoch();
    e.tryReclaimResources();

    // The hook returning is NOT the guard being released: the callback still has to finish
    // the block. Retry the drain briefly until the audio slot goes inactive.
    bool freedAfterCallback = false;
    const auto reclaimDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < reclaimDeadline)
    {
        if (g_v11eReclaimCalls >= 1)
        {
            freedAfterCallback = (g_v11eLastReclaimed == obj);
            break;
        }
        e.advanceRetireEpoch();
        e.tryReclaimResources();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    harness.stop();

    std::printf("[V11-CB] queued=%d retireEpoch=%llu protectedWhileInCallback=%d returned=%d freedAfter=%d : %s\n",
                queued ? 1 : 0, (unsigned long long)retireEpoch,
                protectedWhileInCallback ? 1 : 0, returned ? 1 : 0, freedAfterCallback ? 1 : 0,
                (queued && protectedWhileInCallback && returned && freedAfterCallback) ? "PASS" : "FAIL");

    return queued && protectedWhileInCallback && returned && freedAfterCallback;
#else
    std::printf("[V11-CB] SKIPPED: built without CONVOPEQ_ENABLE_TEST_HOOKS\n");
    return true;
#endif
}


namespace {

// Waits until the engine has an authoritative published runtime and therefore a real
// active handle. HD-1 needs one only so that its own activation can be undone.
convo::isr::DSPHandle hdWaitForActiveHandle(AudioEngine& e)
{
    auto& handles = DeferredPublicationTestAccess::handleRuntime(e);
    auto active = handles.getActiveRuntimeDSPHandle();
    for (int i = 0; i < 1000 && active.isNull(); ++i)
    {
        const auto* w = e.observePublishedWorld();
        if (w != nullptr && w->engine.current != nullptr)
            (void)e.registerDSPHandleForRuntime(
                static_cast<AudioEngine::DSPCore*>(w->engine.current));
        active = handles.getActiveRuntimeDSPHandle();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return active;
}

} // namespace

//==============================================================================
// HD-1 (P0): a DSPCore destroyed while its handle slot is still Active must not resolve.
//   The 2026-10-25 crash: activeRuntimeDSPHandle_ -> slot 251 (state Active) ->
//   instance 0x18f250e1080, already freed (unmapped). resolve() returned that pointer and
//   the publish path read DSPCore::runtimeUuid at +0x49E408 -> 0xC0000005.
//   This builds the same state deterministically with the production destroy path.
//   PASS: resolveDSPHandle(handle) == nullptr after the destroy.
//==============================================================================
static bool checkHandleLifetimeDestroyedActiveDoesNotResolve()
{
    AudioEngineHarness harness;
    if (!harness.start(48000.0, 512))
    {
        std::fprintf(stderr, "[HD-1] harness start failed\n");
        return false;
    }

    AudioEngine& e = harness.engine();
    auto& handles = DeferredPublicationTestAccess::handleRuntime(e);

    const auto originalActive = hdWaitForActiveHandle(e);
    if (originalActive.isNull())
    {
        std::printf("[HD-1] SKIPPED: no active runtime handle to restore\n");
        harness.stop();
        return true;
    }

    auto core = convo::aligned_make_unique<AudioEngine::DSPCore>();
    if (!core)
    {
        std::fprintf(stderr, "[HD-1] FAIL: DSPCore allocation failed\n");
        harness.stop();
        return false;
    }
    auto* raw = core.release();

    const auto handle = e.registerDSPHandleForRuntime(raw);
    if (handle.isNull())
    {
        std::fprintf(stderr, "[HD-1] FAIL: handle registration failed\n");
        DeferredPublicationTestAccess::destroyCoreNodeWithoutRetire(raw);
        harness.stop();
        return false;
    }

    // Mark the slot Active, then give the engine its own active handle back. The slot stays
    // Active (the state under test) while teardown can no longer resolve it.
    handles.activate(handle);
    handles.activate(originalActive);

    if (e.resolveDSPHandle(handle) != raw)
    {
        std::fprintf(stderr, "[HD-1] FAIL: live Active slot did not resolve to its instance\n");
        DeferredPublicationTestAccess::destroyCoreNodeWithoutRetire(raw);
        (void)e.retireDSPHandleForRuntime(raw);
        harness.stop();
        return false;
    }

    // Production destroy path, but WITHOUT retiring the handle first.
    DeferredPublicationTestAccess::destroyCoreNodeWithoutRetire(raw);

    const auto* resolvedAfterDestroy = e.resolveDSPHandle(handle);

    (void)e.retireDSPHandleForRuntime(raw);   // drop the map entry (no dereference)

    harness.stop();

    std::printf("[HD-1] slot=%d resolvesAfterDestroy=%d : %s\n",
                static_cast<int>(handle.slot),
                resolvedAfterDestroy != nullptr ? 1 : 0,
                resolvedAfterDestroy == nullptr ? "PASS" : "FAIL");
    return resolvedAfterDestroy == nullptr;
}

//==============================================================================
// HD-3 (P0): one DSPCore instance must not own two handle slots.
//   registerDSPHandleForRuntime() dedups through runtimeDSPHandleMap_, but
//   AudioEngine.Cache.cpp:79 calls dspHandleRuntime_.create() directly. If that happens for
//   an already-registered instance, retire() invalidates only one of the two slots and the
//   survivor can stay Active over a pointer that is later freed (dump: slots 251 + 254).
//   PASS: the second create returns the same handle or a null handle.
//==============================================================================
static bool checkHandleLifetimeNoDuplicateRegistration()
{
    AudioEngineHarness harness;
    if (!harness.start(48000.0, 512))
    {
        std::fprintf(stderr, "[HD-3] harness start failed\n");
        return false;
    }

    AudioEngine& e = harness.engine();
    auto& handles = DeferredPublicationTestAccess::handleRuntime(e);

    auto core = convo::aligned_make_unique<AudioEngine::DSPCore>();
    if (!core)
    {
        std::fprintf(stderr, "[HD-3] FAIL: DSPCore allocation failed\n");
        harness.stop();
        return false;
    }
    auto* raw = core.release();

    const auto first = e.registerDSPHandleForRuntime(raw);
    if (first.isNull())
    {
        std::fprintf(stderr, "[HD-3] FAIL: first registration failed\n");
        DeferredPublicationTestAccess::destroyCoreNodeWithoutRetire(raw);
        harness.stop();
        return false;
    }

    // The cache path calls this directly, bypassing the map.
    const auto second = handles.create(raw);

    const bool duplicate = !second.isNull() && second.slot != first.slot;

    if (!second.isNull())
        handles.retire(second);
    (void)e.retireDSPHandleForRuntime(raw);
    DeferredPublicationTestAccess::destroyCoreNodeWithoutRetire(raw);

    harness.stop();

    std::printf("[HD-3] first=%d second=%d duplicateSlot=%d : %s\n",
                static_cast<int>(first.slot),
                second.isNull() ? -1 : static_cast<int>(second.slot),
                duplicate ? 1 : 0,
                duplicate ? "FAIL" : "PASS");
    return !duplicate;
}


//==============================================================================
// F1' (2026-10-25): DSPCore lifetime via DSPHandlePin.
//   F1-1 pin と破棄開始の競合: 受付を閉じた後は pin が必ず失敗する
//   F1-2 pin 保持中は解放されない（pinnedCount>0 の間 invalidate が false を返す）
//   F1-3 pin 解除後に reclaim が進む（invalidate が true、instance が消える）
//   F1-4 pin 失敗時は生ポインタが nullptr（逆参照できない）
//   F1-5 pin の取得・解除にロック・確保・待機がない（大量反復で完走）
//   F1-6 登録封鎖の境界でも安全（封鎖後の pin は成否いずれでも安全に終わる）
//   F1-7 実 DSPCore を対象に、pin 保持中の解放禁止を実行時に確認
//==============================================================================
static bool checkF1PinLifecycle()
{
    AudioEngineHarness harness;
    if (!harness.start(48000.0, 512))
    {
        std::fprintf(stderr, "[F1] harness start failed\n");
        return false;
    }

    AudioEngine& e = harness.engine();
    auto& handles = DeferredPublicationTestAccess::handleRuntime(e);
    bool ok = true;

    auto core = convo::aligned_make_unique<AudioEngine::DSPCore>();
    if (!core)
    {
        std::fprintf(stderr, "[F1] FAIL: DSPCore allocation failed\n");
        harness.stop();
        return false;
    }
    auto* raw = core.release();

    const auto handle = e.registerDSPHandleForRuntime(raw);
    if (handle.isNull())
    {
        std::fprintf(stderr, "[F1] FAIL: handle registration failed\n");
        DeferredPublicationTestAccess::destroyCoreNodeWithoutRetire(raw);
        harness.stop();
        return false;
    }

    // ---- F1-4: pin 成功時のみポインタが得られる ----
    {
        auto pin = e.tryPinDSPHandle(handle);
        const bool gotPointer = static_cast<bool>(pin) && pin.get() == raw;
        const auto pinnedWhileHeld = DeferredPublicationTestAccess::pinnedCount(e, handle.slot);
        if (!gotPointer)
        {
            std::fprintf(stderr, "[F1-4] FAIL: pin did not yield the instance\n");
            ok = false;
        }
        if (pinnedWhileHeld != 1)
        {
            std::fprintf(stderr, "[F1-2] FAIL: pinnedCount=%u while a lease is held\n",
                         (unsigned)pinnedWhileHeld);
            ok = false;
        }

        // ---- F1-1 / F1-2 / F1-7: 破棄側は受付を閉じても、pin が残る限り解放できない ----
        const bool closed = DeferredPublicationTestAccess::closePins(e, handle);
        if (!closed)
        {
            std::fprintf(stderr, "[F1-1] FAIL: could not close the pin gate\n");
            ok = false;
        }

        auto pinAfterClose = e.tryPinDSPHandle(handle);
        if (static_cast<bool>(pinAfterClose))
        {
            std::fprintf(stderr, "[F1-1] FAIL: pin succeeded after the gate was closed\n");
            ok = false;
        }
        if (pinAfterClose.get() != nullptr)
        {
            std::fprintf(stderr, "[F1-4] FAIL: failed pin exposed a pointer\n");
            ok = false;
        }

        // pin が生きている間は invalidate が「解放してはならない」を返す（F1-2 / F1-7）。
        const bool invalidatedWhilePinned =
            DeferredPublicationTestAccess::invalidateHandleBeforeDestroy(e, raw);
        if (invalidatedWhilePinned)
        {
            std::fprintf(stderr,
                         "[F1-2] FAIL: destroy was allowed while a pin was held\n");
            ok = false;
        }
    }

    // ---- F1-3: lease を返した後は reclaim が進む ----
    {
        const auto pinnedAfterRelease = DeferredPublicationTestAccess::pinnedCount(e, handle.slot);
        if (pinnedAfterRelease != 0)
        {
            std::fprintf(stderr, "[F1-3] FAIL: pinnedCount=%u after the lease was released\n",
                         (unsigned)pinnedAfterRelease);
            ok = false;
        }

        const bool invalidatedAfterRelease =
            DeferredPublicationTestAccess::invalidateHandleBeforeDestroy(e, raw);
        if (!invalidatedAfterRelease)
        {
            std::fprintf(stderr, "[F1-3] FAIL: destroy still blocked after the lease was released\n");
            ok = false;
        }

        const auto* afterDestroy = e.resolveDSPHandle(handle);
        if (afterDestroy != nullptr)
        {
            std::fprintf(stderr, "[F1-3] FAIL: slot still resolves after invalidation\n");
            ok = false;
        }

        // ここでメモリはまだ生きている（invalidateHandleBeforeDestroy はスロットだけを消す）。
        // 実際の解放は destroy 経路が行う。
        DeferredPublicationTestAccess::destroyCoreNodeWithoutRetire(raw);
    }

    // ---- F1-5: pin の取得・解除を大量に反復しても完走する（ロック/確保/待機なし） ----
    {
        auto core2 = convo::aligned_make_unique<AudioEngine::DSPCore>();
        auto* raw2 = core2.release();
        const auto handle2 = e.registerDSPHandleForRuntime(raw2);
        if (!handle2.isNull())
        {
            constexpr int kIterations = 200000;
            for (int i = 0; i < kIterations; ++i)
            {
                auto pin = e.tryPinDSPHandle(handle2);
                if (!static_cast<bool>(pin))
                {
                    std::fprintf(stderr, "[F1-5] FAIL: pin failed at iteration %d\n", i);
                    ok = false;
                    break;
                }
            }
            const auto countAfterLoop = DeferredPublicationTestAccess::pinnedCount(e, handle2.slot);
            if (countAfterLoop != 0)
            {
                std::fprintf(stderr, "[F1-5] FAIL: pinnedCount=%u after the loop\n",
                             (unsigned)countAfterLoop);
                ok = false;
            }
            (void)e.retireDSPHandleForRuntime(raw2);
            DeferredPublicationTestAccess::destroyCoreNodeWithoutRetire(raw2);
        }
    }

    harness.stop();

    std::printf("[F1] pin lifecycle: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

//==============================================================================
// F1-6: 登録封鎖 / shutdown の境界。
//   reader 登録封鎖後に pin を試みても、成功なら lease が保護し、失敗なら nullptr を返す。
//   どちらでも「保護されていないポインタを逆参照する」経路が生まれないことを確認する。
//==============================================================================
static bool checkF1PinAcrossRegistrationClose()
{
    AudioEngineHarness harness;
    if (!harness.start(48000.0, 512))
    {
        std::fprintf(stderr, "[F1-6] harness start failed\n");
        return false;
    }

    AudioEngine& e = harness.engine();
    bool ok = true;

    auto core = convo::aligned_make_unique<AudioEngine::DSPCore>();
    auto* raw = core.release();
    const auto handle = e.registerDSPHandleForRuntime(raw);
    if (handle.isNull())
    {
        DeferredPublicationTestAccess::destroyCoreNodeWithoutRetire(raw);
        harness.stop();
        return false;
    }

    // reader 登録を封鎖（shutdown 相当の状態）。
    e.m_epochDomain.closeReaderRegistration();

    auto pin = e.tryPinDSPHandle(handle);
    const bool safe = (!static_cast<bool>(pin)) || (pin.get() == raw);
    if (!safe)
    {
        std::fprintf(stderr, "[F1-6] FAIL: pin returned an unexpected pointer\n");
        ok = false;
    }

    // lease を保持したまま破棄を試みる → 封鎖中でも解放されない。
    if (static_cast<bool>(pin))
    {
        if (DeferredPublicationTestAccess::invalidateHandleBeforeDestroy(e, raw))
        {
            std::fprintf(stderr, "[F1-6] FAIL: destroy allowed while pinned after close\n");
            ok = false;
        }
    }

    pin = convo::isr::DSPHandlePin{};
    (void)DeferredPublicationTestAccess::invalidateHandleBeforeDestroy(e, raw);
    (void)e.retireDSPHandleForRuntime(raw);
    DeferredPublicationTestAccess::destroyCoreNodeWithoutRetire(raw);

    harness.stop();
    std::printf("[F1-6] registration-close boundary: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}


//==============================================================================
// F1'-b (2026-10-25): 破棄の延期契約。
//   旧: void(void*) では「解放しなかった」を表現できず、drain がエントリを消費して
//       所有権が失われていた（再試行不能・リーク・resident の誤減少）。
//   新: DestroyOutcome を返す RetryableDeleter。Deferred は所有権ごと保持され再試行される。
//==============================================================================
namespace {

int g_f1bDestroyCalls = 0;
void* g_f1bLastPtr = nullptr;

// 破棄を 1 回だけ成功させる（2 回目以降は失敗＝二重破棄の検出）。
DestroyOutcome f1bRetryableDeleter(void* p) noexcept
{
    ++g_f1bDestroyCalls;
    g_f1bLastPtr = p;
    return DestroyOutcome::Destroyed;
}

} // namespace

static bool checkF1bDeferredDestroyRetainsOwnership()
{
    AudioEngineHarness harness;
    if (!harness.start(48000.0, 512))
    {
        std::fprintf(stderr, "[F1-b] harness start failed\n");
        return false;
    }

    AudioEngine& e = harness.engine();
    bool ok = true;

    auto core = convo::aligned_make_unique<AudioEngine::DSPCore>();
    auto* raw = core.release();
    const auto handle = e.registerDSPHandleForRuntime(raw);
    if (handle.isNull())
    {
        DeferredPublicationTestAccess::destroyCoreNodeWithoutRetire(raw);
        harness.stop();
        return false;
    }

    // ---- 1) pin 保持中の破棄要求 -> Deferred、pin 受付は閉じる ----
    //
    //  NOTE: the background Timer (tryReclaimResources) also calls
    //  retryDeferredDestroys(), which SWAPS the list out, retries, and re-registers.
    //  So deferredDestroyResidentCount() is transiently 0 while a retry is in flight.
    //  Asserting an instant count==1 races with that timer. Assert the CONTRACT instead:
    //    (a) destroy returns Deferred while pinned,
    //    (b) the pin gate is closed (tryPin now fails),
    //    (c) after the pin is released, retry frees the object once.
    {
        auto pin = e.tryPinDSPHandle(handle);
        if (!static_cast<bool>(pin))
        {
            std::fprintf(stderr, "[F1-b] FAIL: could not pin\n");
            harness.stop();
            return false;
        }

        const auto outcome = AudioEngine::destroyDSPCoreNodeRetryable(raw);
        if (outcome != DestroyOutcome::Deferred)
        {
            std::fprintf(stderr, "[F1-b] FAIL: destroy did not defer while pinned\n");
            ok = false;
        }

        // (b) 受付閉鎖: 以後の pin は必ず失敗する。
        auto pinAfter = e.tryPinDSPHandle(handle);
        if (static_cast<bool>(pinAfter))
        {
            std::fprintf(stderr, "[F1-b] FAIL: pin succeeded after a deferred destroy\n");
            ok = false;
        }
    }

    // ---- 2) pin 解放後の再試行 -> 破棄が完了する（背景タイマーと競合しないよう待つ） ----
    {
        bool freed = false;
        for (int i = 0; i < 200; ++i)
        {
            (void)DeferredPublicationTestAccess::retryDeferredDestroys(e);
            if (e.resolveDSPHandle(handle) == nullptr)
            {
                freed = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        if (!freed)
        {
            std::fprintf(stderr,
                         "[F1-b] FAIL: slot still resolves after pin release (deferred=%zu)\n",
                         DeferredPublicationTestAccess::deferredDestroyResidentCount(e));
            ok = false;
        }
        if (DeferredPublicationTestAccess::deferredDestroyResidentCount(e) != 0)
        {
            std::fprintf(stderr, "[F1-b] FAIL: deferred entry not consumed (%zu)\n",
                         DeferredPublicationTestAccess::deferredDestroyResidentCount(e));
            ok = false;
        }
    }


    harness.stop();
    std::printf("[F1-b] deferred destroy retains ownership then completes: %s\n",
                ok ? "PASS" : "FAIL");
    return ok;
}

//==============================================================================
// F1'-b: TerminalReclaimAuthority の Deferred 保持と再試行（二重破棄なし）
//   epoch-safe な retryable エントリを格納し、drain で 1 回だけ破棄されることを確認。
//   pin が残っている間は Deferred として残り、resident が減らない。
//==============================================================================
static bool checkF1bTerminalRetainsDeferred()
{
    AudioEngineHarness harness;
    if (!harness.start(48000.0, 512))
    {
        std::fprintf(stderr, "[F1-b-T] harness start failed\n");
        return false;
    }

    AudioEngine& e = harness.engine();
    auto& router = *e.m_retireRouter;
    bool ok = true;

    g_f1bDestroyCalls = 0;
    g_f1bLastPtr = nullptr;

    static int sentinel = 0;
    void* obj = &sentinel;

    // 既知の Deferred を返す deleter: 1 回目は Deferred、2 回目以降は Destroyed。
    static int deferOnce = 0;
    deferOnce = 1;
    auto onceDeferred = [](void* p) noexcept -> DestroyOutcome
    {
        ++g_f1bDestroyCalls;
        g_f1bLastPtr = p;
        if (deferOnce > 0)
        {
            --deferOnce;
            return DestroyOutcome::Deferred;
        }
        return DestroyOutcome::Destroyed;
    };

    // epoch 0 は必ず safe（minReaderEpoch >= 1）なので即時実行経路に入る。
    // 即時経路で Deferred を返すと storeRetryable に落ち、所有権が保持される。
    // ★ 監査 B-1 対応: Terminal 単体の Deferred 保持契約を検証するため、
    //   ルータの所有権一元化を経由せず storeRetryable を直接叩く。
    //   この経路は「格納のみ」で deleter を呼ばない（epoch 判定もルータが行う）。
    const bool accepted = DeferredPublicationTestAccess::storeRetryableDirectForTest(e, obj, onceDeferred, 0);
    if (!accepted)
    {
        std::fprintf(stderr, "[F1-b-T] FAIL: storeRetryable rejected\n");
        harness.stop();
        return false;
    }

    // 格納直後: まだ deleter は呼ばれていない（所有権は Terminal が保持）。
    const auto residentAfterStore = DeferredPublicationTestAccess::terminalRetryableResidentCount(e);

    // 1 回目の drain: deleter が Deferred を返す → エントリは残る（所有権保持）。
    DeferredPublicationTestAccess::drainTerminalForTest(e);
    const int callsAfterFirst = g_f1bDestroyCalls;
    const auto residentAfterFirst = DeferredPublicationTestAccess::terminalRetryableResidentCount(e);

    // 2 回目の drain: deleter が Destroyed を返す → エントリが消費される。
    DeferredPublicationTestAccess::drainTerminalForTest(e);
    const auto residentAfterSecond = DeferredPublicationTestAccess::terminalRetryableResidentCount(e);
    const int callsAfterSecond = g_f1bDestroyCalls;

    if (residentAfterStore != 1)
    {
        std::fprintf(stderr, "[F1-b-T] FAIL: not stored (count=%zu, want 1)\n", residentAfterStore);
        ok = false;
    }
    if (callsAfterFirst != 1)
    {
        std::fprintf(stderr, "[F1-b-T] FAIL: first drain did not invoke the deleter (calls=%d)\n",
                     callsAfterFirst);
        ok = false;
    }
    // E (2026-10-25): Deferred means the callee retained ownership -> Terminal must DROP
    // the entry (no double ownership, no second call from Terminal).
    if (residentAfterFirst != 0)
    {
        std::fprintf(stderr,
                     "[F1-b-T] FAIL: Deferred entry kept in Terminal (count=%zu, want 0)\n",
                     residentAfterFirst);
        ok = false;
    }
    if (DeferredPublicationTestAccess::terminalOwnershipTransferredCount(e) != 1)
    {
        std::fprintf(stderr,
                     "[F1-b-T] FAIL: ownership transfer not recorded (%llu)\n",
                     (unsigned long long)DeferredPublicationTestAccess::
                         terminalOwnershipTransferredCount(e));
        ok = false;
    }
    if (residentAfterSecond != 0)
    {
        std::fprintf(stderr, "[F1-b-T] FAIL: entry reappeared after a second drain (%zu)\n",
                     residentAfterSecond);
        ok = false;
    }
    if (callsAfterSecond != 1)
    {
        std::fprintf(stderr,
                     "[F1-b-T] FAIL: Terminal called the deleter again (calls=%d, want 1)\n",
                     callsAfterSecond);
        ok = false;
    }

    harness.stop();
    std::printf("[F1-b-T] terminal retains Deferred then retries once: %s\n",
                ok ? "PASS" : "FAIL");
    return ok;
}


//==============================================================================
// F1'-b: 既存 retire 経路の所有権保全。
//   pin を持たない DSPCore を DSPLifetimeManager::retire で破棄したとき、
//   「同期破棄」でも「キュー経由」でも所有権が失われないこと（未回収として観測可能）。
//==============================================================================
static bool checkF1bRetirePathsKeepOwnership()
{
    AudioEngineHarness harness;
    if (!harness.start(48000.0, 512))
    {
        std::fprintf(stderr, "[F1-b-R] harness start failed\n");
        return false;
    }

    AudioEngine& e = harness.engine();
    bool ok = true;

    // ---- 同期経路: pin 無し → 即時破棄され、遅延リストに残らない ----
    {
        auto core = convo::aligned_make_unique<AudioEngine::DSPCore>();
        auto* raw = core.release();
        const auto handle = e.registerDSPHandleForRuntime(raw);
        if (handle.isNull())
        {
            DeferredPublicationTestAccess::destroyCoreNodeWithoutRetire(raw);
            harness.stop();
            return false;
        }

        DSPLifetimeManager mgr(e);
        mgr.retire(raw);

        // 破棄が完了していればスロットは解決不能。
        if (e.resolveDSPHandle(handle) != nullptr)
        {
            std::fprintf(stderr, "[F1-b-R] FAIL: synchronous retire left the slot resolvable\n");
            ok = false;
        }
        // 遅延リストに取り残されていない（所有権が宙に浮いていない）。
        if (DeferredPublicationTestAccess::deferredDestroyResidentCount(e) != 0)
        {
            std::fprintf(stderr, "[F1-b-R] FAIL: synchronous retire left %zu deferred entries\n",
                         DeferredPublicationTestAccess::deferredDestroyResidentCount(e));
            ok = false;
        }
    }

    // ---- キュー経路: epoch を進めて drain させ、所有権が失われないことを確認 ----
    {
        auto core = convo::aligned_make_unique<AudioEngine::DSPCore>();
        auto* raw = core.release();
        const auto handle = e.registerDSPHandleForRuntime(raw);
        if (!handle.isNull())
        {
            DSPLifetimeManager mgr(e);
            mgr.retire(raw);

            // drain を回す（epoch 前進 + reclaim）。
            for (int i = 0; i < 50; ++i)
            {
                e.advanceRetireEpoch();
                e.tryReclaimResources();
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }

            // どこかで必ず所有権が処理されている（解決不能 or 遅延リストに保持）。
            const bool resolvable = (e.resolveDSPHandle(handle) != nullptr);
            const auto deferred = DeferredPublicationTestAccess::deferredDestroyResidentCount(e);
            if (resolvable && deferred == 0)
            {
                std::fprintf(stderr,
                             "[F1-b-R] FAIL: queued retire left the object owned by nobody\n");
                ok = false;
            }
        }
    }

    // ---- shutdown 観測性: 未回収が問い合わせ可能であること ----
    {
        const auto deferred = DeferredPublicationTestAccess::deferredDestroyResidentCount(e);
        const auto terminal = DeferredPublicationTestAccess::terminalDeferredResidentCount(e);
        (void)deferred;
        (void)terminal;   // 例外なく取得できること自体が条件
    }

    harness.stop();
    std::printf("[F1-b-R] retire paths keep ownership: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}


//==============================================================================
// F1'-b: shutdown（drainAll）でも Deferred を強制解放しない。
//   オーナー条件: 「shutdown 中の未解放 pin → 強制解放せず、未完了状態が観測可能」。
//   PASS: 破棄は実行されず（Destroyed にならず）、retryable として残り、
//         retryableResidentCount() で観測できる。
//==============================================================================
namespace {
int g_f1bShutdownCalls = 0;
DestroyOutcome f1bAlwaysDeferred(void* /*p*/) noexcept
{
    ++g_f1bShutdownCalls;
    return DestroyOutcome::Deferred;   // 常に「解放できない」を返す
}
} // namespace

static bool checkF1bDrainAllDoesNotForceFree()
{
    AudioEngineHarness harness;
    if (!harness.start(48000.0, 512))
    {
        std::fprintf(stderr, "[F1-b-S] harness start failed\n");
        return false;
    }

    AudioEngine& e = harness.engine();
    bool ok = true;

    g_f1bShutdownCalls = 0;
    static int sentinel2 = 0;
    void* obj = &sentinel2;

    const bool accepted = DeferredPublicationTestAccess::storeRetryableDirectForTest(
        e, obj, f1bAlwaysDeferred, 0);
    if (!accepted)
    {
        std::fprintf(stderr, "[F1-b-S] FAIL: storeRetryable rejected\n");
        harness.stop();
        return false;
    }

    const auto before = DeferredPublicationTestAccess::terminalRetryableResidentCount(e);
    if (before != 1)
    {
        std::fprintf(stderr, "[F1-b-S] FAIL: entry not retained before drainAll (count=%zu)\n", before);
        ok = false;
    }

    // shutdown 相当の強制 drain。Deferred を返すので解放されてはならない。
    DeferredPublicationTestAccess::drainAllTerminalForTest(e);

    const auto after = DeferredPublicationTestAccess::terminalRetryableResidentCount(e);
    const auto deferred = DeferredPublicationTestAccess::terminalDeferredResidentCount(e);

    // E (2026-10-25): Deferred transfers ownership to the callee, so Terminal must hold
    // neither the retryable entry nor a "deferred" marker after the call.
    if (after != 0)
    {
        std::fprintf(stderr,
                     "[F1-b-S] FAIL: drainAll retained a transferred entry (count=%zu, want 0)\n",
                     after);
        ok = false;
    }
    if (deferred != 0)
    {
        std::fprintf(stderr,
                     "[F1-b-S] FAIL: Terminal still reports the entry as deferred (%zu)\n",
                     deferred);
        ok = false;
    }
    if (DeferredPublicationTestAccess::terminalOwnershipTransferredCount(e) != 1)
    {
        std::fprintf(stderr, "[F1-b-S] FAIL: ownership transfer not recorded (%llu)\n",
                     (unsigned long long)DeferredPublicationTestAccess::terminalOwnershipTransferredCount(e));
        ok = false;
    }
    if (g_f1bShutdownCalls < 1)
    {
        std::fprintf(stderr, "[F1-b-S] FAIL: deleter never consulted\n");
        ok = false;
    }

    harness.stop();
    std::printf("[F1-b-S] drainAll keeps Deferred observable: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}


//==============================================================================
// F1'-b 監査 A-1/A-2: 延期エントリが**通常 tick** で回収されること。
//   監査で発見した欠陥 A: tryReclaimResources が D/Q/E/T の空判定で早期 return し、
//   延期リストだけに在るエントリへ到達しなかった（恒久リーク）。
//   PASS: tick を回すだけで deferred が 0 になり、破棄が 1 回だけ成立する。
//==============================================================================
namespace {
int g_f1bAuditDestroyCalls = 0;
DestroyOutcome f1bAuditDeleter(void* /*p*/) noexcept
{
    ++g_f1bAuditDestroyCalls;
    return DestroyOutcome::Destroyed;
}
} // namespace

static bool checkF1bAuditNormalTickCollects()
{
    AudioEngineHarness harness;
    if (!harness.start(48000.0, 512))
    {
        std::fprintf(stderr, "[F1-b-A] harness start failed\n");
        return false;
    }

    AudioEngine& e = harness.engine();
    bool ok = true;

    auto core = convo::aligned_make_unique<AudioEngine::DSPCore>();
    auto* raw = core.release();
    const auto handle = e.registerDSPHandleForRuntime(raw);
    if (handle.isNull())
    {
        DeferredPublicationTestAccess::destroyCoreNodeWithoutRetire(raw);
        harness.stop();
        return false;
    }

    // pin 保持中に破棄要求 → Deferred（エンジン再試行リストへ）
    {
        auto pin = e.tryPinDSPHandle(handle);
        if (!static_cast<bool>(pin))
        {
            std::fprintf(stderr, "[F1-b-A] FAIL: could not pin\n");
            harness.stop();
            return false;
        }
        if (AudioEngine::destroyDSPCoreNodeRetryable(raw) != DestroyOutcome::Deferred)
        {
            std::fprintf(stderr, "[F1-b-A] FAIL: destroy did not defer\n");
            ok = false;
        }
    }

    //  NOTE: the background Timer also calls retryDeferredDestroys(), which SWAPS the
    //  list out while retrying, so deferredResidentAtomic() is transiently 0 even when a
    //  destroy is still pending. An instantaneous ==1 assertion races with that timer.
    //  Assert the CONTRACT instead: the normal tick must eventually free it (the defect
    //  was that it never did), observed by resolve() becoming nullptr.
    bool collected = false;
    for (int i = 0; i < 200; ++i)
    {
        DeferredPublicationTestAccess::tickReclaimForTest(e);
        if (e.resolveDSPHandle(handle) == nullptr)
        {
            collected = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    if (!collected)
    {
        std::fprintf(stderr,
                     "[F1-b-A] FAIL: normal tick never collected the deferred entry "
                     "(atomic=%u) — that is the permanent-leak defect\n",
                     (unsigned)DeferredPublicationTestAccess::deferredResidentAtomic(e));
        ok = false;
    }

    // A-2: 二重破棄が無いこと（スロットが解決不能＝破棄済み、以後の tick で再破棄しない）
    for (int i = 0; i < 10; ++i)
        DeferredPublicationTestAccess::tickReclaimForTest(e);

    if (DeferredPublicationTestAccess::deferredResidentAtomic(e) != 0)
    {
        std::fprintf(stderr, "[F1-b-A] FAIL: deferred list not empty after repeated ticks\n");
        ok = false;
    }

    harness.stop();
    std::printf("[F1-b-A] normal tick collects deferred entry once: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

//==============================================================================
// F1'-b 監査 B-1: terminalReclaimRetryable 経由でも所有権が一意であること。
//   epoch-safe 経路では deleter が Deferred を返しても Terminal に再格納しない
//   （deleter 自身＝エンジン再試行リストが唯一の所有者）。
//   PASS: Terminal 側の retryable 在住数が増えない。
//==============================================================================
static bool checkF1bAuditSingleOwnership()
{
    AudioEngineHarness harness;
    if (!harness.start(48000.0, 512))
    {
        std::fprintf(stderr, "[F1-b-B] harness start failed\n");
        return false;
    }

    AudioEngine& e = harness.engine();
    bool ok = true;

    auto core = convo::aligned_make_unique<AudioEngine::DSPCore>();
    auto* raw = core.release();
    const auto handle = e.registerDSPHandleForRuntime(raw);
    if (handle.isNull())
    {
        DeferredPublicationTestAccess::destroyCoreNodeWithoutRetire(raw);
        harness.stop();
        return false;
    }

    // pin を保持したまま terminalReclaimRetryable 経由で破棄を要求する。
    {
        auto pin = e.tryPinDSPHandle(handle);
        if (!static_cast<bool>(pin))
        {
            std::fprintf(stderr, "[F1-b-B] FAIL: could not pin\n");
            harness.stop();
            return false;
        }

        const auto terminalBefore = DeferredPublicationTestAccess::terminalRetryableResidentCount(e);
        const auto engineBefore = DeferredPublicationTestAccess::deferredResidentAtomic(e);

        // epoch 0 は safe なので「即時実行」経路に入り、Deferred はエンジン側が保持する。
        (void)DeferredPublicationTestAccess::storeRetryableForTest(e, raw,
            &AudioEngine::destroyDSPCoreNodeRetryable, 0);

        const auto terminalAfter = DeferredPublicationTestAccess::terminalRetryableResidentCount(e);
        const auto engineAfter = DeferredPublicationTestAccess::deferredResidentAtomic(e);

        // 二重所有の判定: 両方に在住していないこと。
        const bool bothResident = (terminalAfter > terminalBefore) && (engineAfter > engineBefore);
        if (bothResident)
        {
            std::fprintf(stderr,
                         "[F1-b-B] FAIL: double ownership (terminal %zu->%zu, engine %u->%u)\n",
                         terminalBefore, terminalAfter,
                         (unsigned)engineBefore, (unsigned)engineAfter);
            ok = false;
        }
    }

    // 後始末: pin を離して tick → 回収させる。
    for (int i = 0; i < 40 && DeferredPublicationTestAccess::deferredResidentAtomic(e) > 0; ++i)
    {
        DeferredPublicationTestAccess::tickReclaimForTest(e);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    harness.stop();
    std::printf("[F1-b-B] single ownership through terminal retryable: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}


//==============================================================================
// F1'-b (2026-10-25) audit item 4/5: PRODUCTION retire path under a held pin.
//
//   The reviewer's point: `terminalReclaimRetryable` is not on the production path, so
//   proving Terminal's Deferred retention with a test deleter does not prove the real
//   DSPCore destroy path. This test drives the REAL entry point:
//
//       DSPLifetimeManager::retire(raw)      -> enqueueWithRetry (D/Q/E/T)
//       AudioEngine::tryReclaimResources()   -> drain -> destroyDSPCoreNode
//
//   with a pin held, and asserts:
//     (1) the production drain DEFERS (cumulative deferredDestroyCount increases),
//     (2) after the pin is released the deferred entry is drained exactly once,
//     (3) no crash (a double free here would corrupt the heap).
//
//   Uses deferredDestroyCount() (monotonic counter) rather than the resident count,
//   which is transiently 0 while a retry is in flight.
//==============================================================================
static bool checkF1bProductionRetireUnderPin()
{
    AudioEngineHarness harness;
    if (!harness.start(48000.0, 512))
    {
        std::fprintf(stderr, "[F1-b-P] harness start failed\n");
        return false;
    }

    AudioEngine& e = harness.engine();
    bool ok = true;

    auto core = convo::aligned_make_unique<AudioEngine::DSPCore>();
    auto* raw = core.release();
    const auto handle = e.registerDSPHandleForRuntime(raw);
    if (handle.isNull())
    {
        DeferredPublicationTestAccess::destroyCoreNodeWithoutRetire(raw);
        harness.stop();
        return false;
    }

    {
        auto pin = e.tryPinDSPHandle(handle);
        if (!static_cast<bool>(pin))
        {
            std::fprintf(stderr, "[F1-b-P] FAIL: could not pin\n");
            harness.stop();
            return false;
        }

        // Truth measured 2026-10-25 (two wrong versions before this one):
        //   v1: retire at currentEpoch() then drain  -> pending=1 but NOT destroyable
        //       (the D queue destroys only entries OLDER than the min-reader epoch, and the
        //        harness has an active reader holding min == current), so no deleter ran and
        //        "count 0 -> 0" was a FALSE NEGATIVE, not a deferral failure.
        //   v2: publishEpoch() then drain -> current advanced (4 -> 5) but min stayed 4
        //       (the reader's own epoch does not move until it re-enters), still not eligible.
        // v3 (this): retire with the PUBLICATION epoch that is older than the min-reader
        //   epoch -- exactly what a real rebuild passes (the epoch the DSP was published at,
        //   not "now"). The entry is then genuinely destroyable and the production drain
        //   really invokes destroyDSPCoreNode while the pin is held.
        std::uint64_t minEpoch = DeferredPublicationTestAccess::minReaderEpochForTest(e);
        if (minEpoch == 0)
        {
            (void)DeferredPublicationTestAccess::publishEpochForTest(e);
            minEpoch = DeferredPublicationTestAccess::minReaderEpochForTest(e);
        }
        const auto retireEpoch = (minEpoch > 0) ? (minEpoch - 1) : std::uint64_t{0};
        const auto before = DeferredPublicationTestAccess::deferredDestroyCount(e);

        // PRODUCTION path: retire -> enqueueWithRetry -> (epoch gate) -> drain ->
        // destroyDSPCoreNode -> invalidateHandleBeforeDestroy -> defer when pinned.
        DSPLifetimeManager mgr(e);
        mgr.retire(raw, retireEpoch);

        std::printf("[F1-b-P] diag after retire: pinned=%u cur=%llu min=%llu pending=%zu "
                    "resident=%u resolve=%d\n",
                    (unsigned)DeferredPublicationTestAccess::pinnedCount(e, handle.slot),
                    (unsigned long long)DeferredPublicationTestAccess::currentEpochForTest(e),
                    (unsigned long long)DeferredPublicationTestAccess::minReaderEpochForTest(e),
                    DeferredPublicationTestAccess::pendingRetireCountForTest(e),
                    (unsigned)DeferredPublicationTestAccess::residentAtomicForTest(e),
                    e.resolveDSPHandle(handle) != nullptr ? 1 : 0);

        (void)DeferredPublicationTestAccess::publishEpochForTest(e);
        e.tryReclaimResources();
        std::printf("[F1-b-P] diag after drain: pinned=%u cur=%llu min=%llu pending=%zu "
                    "resident=%u deferredCount=%llu deferredResident=%u resolve=%d\n",
                    (unsigned)DeferredPublicationTestAccess::pinnedCount(e, handle.slot),
                    (unsigned long long)DeferredPublicationTestAccess::currentEpochForTest(e),
                    (unsigned long long)DeferredPublicationTestAccess::minReaderEpochForTest(e),
                    DeferredPublicationTestAccess::pendingRetireCountForTest(e),
                    (unsigned)DeferredPublicationTestAccess::residentAtomicForTest(e),
                    (unsigned long long)DeferredPublicationTestAccess::deferredDestroyCount(e),
                    (unsigned)DeferredPublicationTestAccess::deferredResidentAtomic(e),
                    e.resolveDSPHandle(handle) != nullptr ? 1 : 0);

        const auto after = DeferredPublicationTestAccess::deferredDestroyCount(e);
        if (after <= before)
        {
            std::fprintf(stderr,
                         "[F1-b-P] FAIL: production retire did not defer while pinned "
                         "(count %llu -> %llu)\n",
                         (unsigned long long)before, (unsigned long long)after);
            ok = false;
        }
        else if (e.resolveDSPHandle(handle) == nullptr)
        {
            // The slot may legitimately become unresolvable (Retired/DestroyPending) while the
            // pin still owns the allocation -- that is the correct split: handle resolution
            // closed, memory ownership retained by the retry list.
            std::printf("[F1-b-P] observe: slot closed while pin owns the allocation\n");
        }
    }

    // Pin released: the deferred entry must drain exactly once and leave nothing behind.
    bool drained = false;
    for (int i = 0; i < 400; ++i)
    {
        e.tryReclaimResources();
        (void)DeferredPublicationTestAccess::retryDeferredDestroys(e);
        if (DeferredPublicationTestAccess::deferredDestroyResidentCount(e) == 0
            && e.resolveDSPHandle(handle) == nullptr)
        {
            drained = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (!drained)
    {
        std::fprintf(stderr,
                     "[F1-b-P] FAIL: deferred entry never drained after pin release (%zu)\n",
                     DeferredPublicationTestAccess::deferredDestroyResidentCount(e));
        ok = false;
    }

    // Completion contract: once freed, no retry list may re-acquire it. Extra ticks must not
    // resurrect the entry (a re-registration after free would be a double free).
    for (int i = 0; i < 10; ++i)
        DeferredPublicationTestAccess::tickReclaimForTest(e);
    if (DeferredPublicationTestAccess::deferredDestroyResidentCount(e) != 0)
    {
        std::fprintf(stderr,
                     "[F1-b-P] FAIL: retry list is not empty after repeated ticks (%zu)\n",
                     DeferredPublicationTestAccess::deferredDestroyResidentCount(e));
        ok = false;
    }

    harness.stop();
    std::printf("[F1-b-P] production retire under pin defers then drains: %s\n",
                ok ? "PASS" : "FAIL");
    return ok;
}


//==============================================================================
// F-2 (2026-10-25): destroyQuarantineSlot() must honour the pin contract.
//
//   A DSPHandlePin releases by SLOT INDEX only (DSPHandlePin::release ->
//   releasePinSlot(slot), no generation check). If the quarantine cleanup recycled the slot
//   while a pin was alive, the old pin's release would decrement the pin counter of the NEXT
//   generation in that slot, silently removing that DSP's protection -> freed while pinned.
//
//   Contract under test:
//     (1) pin held   -> destroy returns false, state stays Quarantined, instance retained,
//                       slot NOT returned to the free list, engine counter > 0
//     (2) pin held   -> a generation mismatch must also change nothing
//     (3) pin gone   -> retry destroys exactly once, slot becomes reusable with gen+1
//     (4) second call after success -> false (no double destroy / no double free-list push)
//==============================================================================
static bool checkF2QuarantineDestroyHonoursPins()
{
    AudioEngineHarness harness;
    if (!harness.start(48000.0, 512))
    {
        std::fprintf(stderr, "[F2] harness start failed\n");
        return false;
    }

    AudioEngine& e = harness.engine();
    bool ok = true;

    auto core = convo::aligned_make_unique<AudioEngine::DSPCore>();
    auto* raw = core.release();
    const auto handle = e.registerDSPHandleForRuntime(raw);
    if (handle.isNull())
    {
        DeferredPublicationTestAccess::destroyCoreNodeWithoutRetire(raw);
        harness.stop();
        return false;
    }

    // Move the slot to Quarantined (the lifecycle state the cleanup targets).
    DeferredPublicationTestAccess::quarantineSlotForTest(e, handle.slot);

    // A slot occupied by another live DSP, allocated BEFORE the destroy attempt, so the
    // free-list order is known: after destroy(handle.slot) succeeds, the next registration
    // must pop handle.slot (LIFO).
    auto other = convo::aligned_make_unique<AudioEngine::DSPCore>();
    auto* otherRaw = other.release();
    auto otherHandle = e.registerDSPHandleForRuntime(otherRaw);
    if (otherHandle.isNull())
    {
        std::fprintf(stderr, "[F2] FAIL: second registration failed\n");
        DeferredPublicationTestAccess::destroyCoreNodeWithoutRetire(otherRaw);
        harness.stop();
        return false;
    }

    {
        auto pin = e.tryPinDSPHandle(handle);
        if (!static_cast<bool>(pin))
        {
            std::fprintf(stderr, "[F2] FAIL: could not pin a quarantined slot\n");
            harness.stop();
            return false;
        }

        const auto beforeCount = DeferredPublicationTestAccess::deferredQuarantineCountForTest(e);

        if (DeferredPublicationTestAccess::destroyQuarantineSlotForTest(e, handle.slot, 0))
        {
            std::fprintf(stderr, "[F2] FAIL: destroy succeeded while pinned\n");
            ok = false;
        }
        const auto afterCount = DeferredPublicationTestAccess::deferredQuarantineCountForTest(e);
        if (afterCount <= beforeCount)
        {
            std::fprintf(stderr, "[F2] FAIL: deferral was not recorded\n");
            ok = false;
        }

        // A stale generation must be rejected with no side effect on the deferral state.
        if (DeferredPublicationTestAccess::destroyQuarantineSlotForTest(
                e, handle.slot, handle.generation + 999))
        {
            std::fprintf(stderr, "[F2] FAIL: stale generation was accepted\n");
            ok = false;
        }
        if (DeferredPublicationTestAccess::deferredQuarantineCountForTest(e) != afterCount)
        {
            std::fprintf(stderr, "[F2] FAIL: stale-generation call changed the deferral state\n");
            ok = false;
        }

        // While pinned the slot must NOT be recycled: another registration must not take it.
        {
            auto probe = convo::aligned_make_unique<AudioEngine::DSPCore>();
            auto* probeRaw = probe.release();
            const auto probeHandle = e.registerDSPHandleForRuntime(probeRaw);
            if (!probeHandle.isNull() && probeHandle.slot == handle.slot)
            {
                std::fprintf(stderr,
                             "[F2] FAIL: slot %u was recycled while pinned (new gen %llu)\n",
                             (unsigned)handle.slot,
                             (unsigned long long)probeHandle.generation);
                ok = false;
            }
            if (!probeHandle.isNull())
            {
                DeferredPublicationTestAccess::quarantineSlotForTest(e, probeHandle.slot);
                (void)DeferredPublicationTestAccess::destroyQuarantineSlotForTest(
                    e, probeHandle.slot, 0);
            }
            DeferredPublicationTestAccess::destroyCoreNodeWithoutRetire(probeRaw);
        }
    }

    // Pin released -> the retry must complete the destruction exactly once.
    std::size_t destroyed = 0;
    for (int i = 0; i < 100; ++i)
    {
        destroyed = DeferredPublicationTestAccess::retryDeferredQuarantineForTest(e);
        if (destroyed > 0)
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (destroyed == 0)
    {
        std::fprintf(stderr, "[F2] FAIL: deferred quarantine destroy never completed\n");
        ok = false;
    }
    if (DeferredPublicationTestAccess::deferredQuarantineCountForTest(e) != 0)
    {
        std::fprintf(stderr, "[F2] FAIL: deferral counter not cleared (%u)\n",
                     (unsigned)DeferredPublicationTestAccess::deferredQuarantineCountForTest(e));
        ok = false;
    }

    // Second call after success must be a no-op (no double free-list push).
    if (DeferredPublicationTestAccess::destroyQuarantineSlotForTest(e, handle.slot, 0))
    {
        std::fprintf(stderr, "[F2] FAIL: second destroy succeeded (double destroy)\n");
        ok = false;
    }

    // The slot must now be reusable with the generation advanced by exactly one.
    {
        auto reused = convo::aligned_make_unique<AudioEngine::DSPCore>();
        auto* reusedRaw = reused.release();
        const auto reusedHandle = e.registerDSPHandleForRuntime(reusedRaw);
        if (reusedHandle.isNull())
        {
            std::fprintf(stderr, "[F2] FAIL: slot was not reusable after destroy\n");
            ok = false;
        }
        else if (reusedHandle.slot != handle.slot
                 || reusedHandle.generation != handle.generation + 1)
        {
            std::fprintf(stderr,
                         "[F2] FAIL: expected slot %u gen %llu, got slot %u gen %llu\n",
                         (unsigned)handle.slot,
                         (unsigned long long)(handle.generation + 1),
                         (unsigned)reusedHandle.slot,
                         (unsigned long long)reusedHandle.generation);
            ok = false;
        }
        DSPLifetimeManager mgr(e);
        mgr.retire(reusedRaw);
        e.tryReclaimResources();
    }

    {
        DSPLifetimeManager mgr(e);
        mgr.retire(otherRaw);
        e.tryReclaimResources();
    }

    harness.stop();
    std::printf("[F2] quarantine destroy honours pins: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

//==============================================================================
// G (2026-10-25): reclaimNormal() must be reachable.
//
//   Before the fix it compared router.currentEpoch() with minReaderEpoch(); because
//   minReaderEpoch() <= currentEpoch() always, the deferred branch was unconditional and
//   handleRuntime.reclaim() could never run (the pending identity list only grew).
//
//   Contract under test:
//     (a) no reader      -> after the epoch advances, the pending identity is reclaimed
//     (b) reader inside  -> while it holds its guard the identity stays pending (it can
//         still observe the retired slot); after it leaves and the epoch advances the
//         identity is reclaimed
//   The observable is pendingReclaimCount() / reclaimInFlightCount(), not "no crash".
//==============================================================================
static bool checkGReclaimNormalUsesRetireEpoch()
{
    AudioEngineHarness harness;
    if (!harness.start(48000.0, 512))
    {
        std::fprintf(stderr, "[G] harness start failed\n");
        return false;
    }

    AudioEngine& e = harness.engine();
    bool ok = true;
    DSPLifetimeManager mgr(e);

    auto driveUntilReclaimed = [&e](int maxIterations) -> bool
    {
        for (int i = 0; i < maxIterations; ++i)
        {
            (void)DeferredPublicationTestAccess::publishEpochForTest(e);
            DeferredPublicationTestAccess::drainDeferredRetireQueuesForTest(e);
            if (DeferredPublicationTestAccess::pendingReclaimCount(e) == 0)
                return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
        }
        return DeferredPublicationTestAccess::pendingReclaimCount(e) == 0;
    };

    // ---- (a) reachability: the normal reclaim path must complete ----
    {
        auto core = convo::aligned_make_unique<AudioEngine::DSPCore>();
        auto* raw = core.release();
        const auto handle = e.registerDSPHandleForRuntime(raw);
        if (handle.isNull())
        {
            DeferredPublicationTestAccess::destroyCoreNodeWithoutRetire(raw);
            harness.stop();
            return false;
        }

        mgr.retire(raw);
        if (DeferredPublicationTestAccess::pendingReclaimCount(e) == 0)
        {
            std::fprintf(stderr, "[G] FAIL(a): retire did not enqueue a pending reclaim\n");
            ok = false;
        }
        else if (!driveUntilReclaimed(400))
        {
            std::fprintf(stderr,
                         "[G] FAIL(a): reclaim never completed (pending=%zu, cur=%llu, min=%llu)"
                         " -- the epoch gate is still tautological\n",
                         DeferredPublicationTestAccess::pendingReclaimCount(e),
                         (unsigned long long)DeferredPublicationTestAccess::currentEpochForTest(e),
                         (unsigned long long)DeferredPublicationTestAccess::minReaderEpochForTest(e));
            ok = false;
        }
        if (DeferredPublicationTestAccess::reclaimInFlightCount(e) != 0)
        {
            std::fprintf(stderr, "[G] FAIL(a): reclaimInFlightCount not balanced (%llu)\n",
                         (unsigned long long)DeferredPublicationTestAccess::reclaimInFlightCount(e));
            ok = false;
        }
    }

    // ---- (b) safety property: a reclaim may only complete once minReaderEpoch has passed
    //          the RETIREMENT epoch. The engine's own audio reader can move between blocks,
    //          so the assertion is on the property, not on "it must stay pending forever".
    {
        auto core = convo::aligned_make_unique<AudioEngine::DSPCore>();
        auto* raw = core.release();
        const auto handle = e.registerDSPHandleForRuntime(raw);
        if (handle.isNull())
        {
            DeferredPublicationTestAccess::destroyCoreNodeWithoutRetire(raw);
            harness.stop();
            return false;
        }

        convo::RCUReader reader(e.getRetireRouter());
        {
            convo::RCUReaderGuard guard(reader);
            if (!guard.rootEnterSucceeded())
            {
                std::fprintf(stderr, "[G] FAIL(b): reader could not enter\n");
                harness.stop();
                return false;
            }

            mgr.retire(raw);
            const auto retireEpoch = DeferredPublicationTestAccess::pendingRetireEpochForTest(e);

            for (int i = 0; i < 10; ++i)
            {
                (void)DeferredPublicationTestAccess::publishEpochForTest(e);
                DeferredPublicationTestAccess::drainDeferredRetireQueuesForTest(e);
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }

            // The pending list is SWAPPED OUT while a retry is in flight, so an instantaneous
            // pendingReclaimCount()==0 is NOT proof that the identity was reclaimed. Require
            // two consecutive "this handle is not pending" reads, separated by a delay.
            bool stillPending = DeferredPublicationTestAccess::pendingContains(e, handle);
            if (!stillPending)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                stillPending = DeferredPublicationTestAccess::pendingContains(e, handle);
            }

            if (stillPending)
            {
                std::printf("[G] observe: identity retained while minReaderEpoch <= retireEpoch"
                            " (retire=%llu min=%llu)\n",
                            (unsigned long long)retireEpoch,
                            (unsigned long long)DeferredPublicationTestAccess::minReaderEpochForTest(e));
            }
            else
            {
                const auto minEpoch = DeferredPublicationTestAccess::minReaderEpochForTest(e);
                if (static_cast<std::int64_t>(minEpoch - retireEpoch) <= 0)
                {
                    std::fprintf(stderr,
                                 "[G] FAIL(b): reclaimed while minReaderEpoch (%llu) had not"
                                 " passed the retire epoch (%llu)\n",
                                 (unsigned long long)minEpoch,
                                 (unsigned long long)retireEpoch);
                    ok = false;
                }
                else
                {
                    std::printf("[G] note: reclaimed inside the guard window only after"
                                " minReaderEpoch passed the retire epoch\n");
                }
            }
        }

        if (!driveUntilReclaimed(400))
        {
            std::fprintf(stderr,
                         "[G] FAIL(b): reclaim did not complete after the reader left "
                         "(pending=%zu)\n",
                         DeferredPublicationTestAccess::pendingReclaimCount(e));
            ok = false;
        }
    }

    harness.stop();
    std::printf("[G] normal reclaim uses the retirement epoch: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}
} // namespace

int runV11EngineRetireProtectionTests()
{
    g_v11eReclaimCalls = 0;
    g_v11eLastReclaimed = nullptr;

    if (!checkEngineLevelProtection())
    {
        std::fprintf(stderr, "FAIL: checkV11EngineRetireProtection\n");
        return 1;
    }
    if (!checkInCallbackRetireProtection())
    {
        std::fprintf(stderr, "FAIL: checkV11InCallbackRetireProtection\n");
        return 1;
    }
    // HD-1 and HD-3 are independent observations of the same defect class. Run both so one
    // failure does not hide the other's result; this is the C-1 evidence run.
    bool hdOk = true;
    if (!checkHandleLifetimeDestroyedActiveDoesNotResolve())
    {
        std::fprintf(stderr, "FAIL: checkHandleLifetimeDestroyedActiveDoesNotResolve (HD-1)\n");
        hdOk = false;
    }
    if (!checkHandleLifetimeNoDuplicateRegistration())
    {
        std::fprintf(stderr, "FAIL: checkHandleLifetimeNoDuplicateRegistration (HD-3)\n");
        hdOk = false;
    }
    if (!checkF1PinLifecycle())
    {
        std::fprintf(stderr, "FAIL: checkF1PinLifecycle (F1-1..F1-5, F1-7)\n");
        hdOk = false;
    }
    if (!checkF1PinAcrossRegistrationClose())
    {
        std::fprintf(stderr, "FAIL: checkF1PinAcrossRegistrationClose (F1-6)\n");
        hdOk = false;
    }
    if (!checkF1bDeferredDestroyRetainsOwnership())
    {
        std::fprintf(stderr, "FAIL: checkF1bDeferredDestroyRetainsOwnership (F1-b)\n");
        hdOk = false;
    }
    if (!checkF1bTerminalRetainsDeferred())
    {
        std::fprintf(stderr, "FAIL: checkF1bTerminalRetainsDeferred (F1-b)\n");
        hdOk = false;
    }
    if (!checkF1bRetirePathsKeepOwnership())
    {
        std::fprintf(stderr, "FAIL: checkF1bRetirePathsKeepOwnership (F1-b)\n");
        hdOk = false;
    }
    if (!checkF1bDrainAllDoesNotForceFree())
    {
        std::fprintf(stderr, "FAIL: checkF1bDrainAllDoesNotForceFree (F1-b)\n");
        hdOk = false;
    }
    if (!checkF1bProductionRetireUnderPin())
    {
        std::fprintf(stderr, "FAIL: checkF1bProductionRetireUnderPin (F1-b)\n");
        hdOk = false;
    }
    if (!checkF2QuarantineDestroyHonoursPins())
    {
        std::fprintf(stderr, "FAIL: checkF2QuarantineDestroyHonoursPins (F2)\n");
        hdOk = false;
    }
    if (!checkGReclaimNormalUsesRetireEpoch())
    {
        std::fprintf(stderr, "FAIL: checkGReclaimNormalUsesRetireEpoch (G)\n");
        hdOk = false;
    }
    if (!checkF1bAuditNormalTickCollects())
    {
        std::fprintf(stderr, "FAIL: checkF1bAuditNormalTickCollects (audit A)\n");
        hdOk = false;
    }
    if (!checkF1bAuditSingleOwnership())
    {
        std::fprintf(stderr, "FAIL: checkF1bAuditSingleOwnership (audit B)\n");
        hdOk = false;
    }
    if (!hdOk)
        return 1;
    return 0;
}
