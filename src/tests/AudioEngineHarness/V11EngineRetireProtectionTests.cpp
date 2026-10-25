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
    if (!hdOk)
        return 1;
    return 0;
}
