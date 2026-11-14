// D8-2-B-2_Tests.cpp
// D8-2-B-2: Test Infrastructure — Configurable Epoch Provider + T1/T2/T3/T4/T7/T8
//
// Test-first: ownership disposition observability for D/Q/E/T pathways.
// No production code changes. Test-only helpers only (anonymous namespace).
//
// T5 (QueueFull)     = DEFERRED (dead code; Terminal is growable)
// T6 (Shutdown)       = DEFERRED (no production path added)
// T9 (DSPLifetimeMgr) = DEFERRED (source audit only, see follow-up)
// T10 (RT boundary)   = NOT EXERCISABLE: resetFadeStateAndRetireTarget has no production
//                       caller (updateFade removed 2026-07-28). The reachable path
//                       (switchImmediate) is verified instead; see test_T10_化)

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <thread>

#include "DspNumericPolicy.h"          // ★ T13: ScopedThreadRole（RT 違反ケース）

#include "audioengine/AtomicAccess.h"
#include "audioengine/ISRAuthorityClass.h"   // RetireEnqueueResult
#include "audioengine/ISRRetireRouter.h"

#include "core/IEpochProvider.h"
#include "core/EpochDomain.h"          // ★ T11: 実 provider（実 DeferredDeletionQueue）で C3 を駆動する
#include "core/SnapshotCoordinator.h"
#include "core/SnapshotFactory.h"
#include "core/SnapshotParams.h"
#include "core/GlobalSnapshot.h"

#include "DeferredDeletionQueue.h"              // DeletionEntryType

namespace {

//============================================================================
// ConfigurableEpochProvider — test-only IEpochProvider stub
//
// All control via atomics (no std::function — RT-safe by design).
// Merges the patterns from:
//   - RetireGraceSemanticsTests.cpp (TestProvider: enqueueRetire=false)
//   - D8_1_WrapperCacheTests.cpp    (TestEpochProvider: enqueueRetire=true)
//============================================================================

class ConfigurableEpochProvider final : public convo::IEpochProvider
{
public:
    // ── Control (settable per-test) ───────────────────────────────────────

    /// When true: enqueueRetire / enqueueRetireTyped return true (D owns).
    /// When false: return false → triggers Q/E/T escalation.
    std::atomic<bool> enqueueRetireResult{true};

    /// minReaderEpoch returned by getMinReaderEpoch().
    /// Entries with epoch < minReaderEpoch are drainable (isOlder).
    /// Default 0 → entries with epoch 0 are NOT drainable (isOlder(0,0)=false).
    std::atomic<uint64_t> minReaderEpoch_{0};

    // ── Observability counters ─────────────────────────────────────────────

    std::atomic<int>       enqueueRetireCallCount{0};
    std::atomic<int>       enqueueRetireTypedCallCount{0};
    std::atomic<int>       tryReclaimCount{0};
    std::atomic<int>       publishEpochCount{0};
    std::atomic<uint64_t>  currentEpoch_{0};

    /// Last ptr accepted by D (enqueueRetire returned true).
    /// Used for cleanup of D-path test objects.
    std::atomic<void*> lastEnqueuedPtr{nullptr};

    // ── IEpochProvider interface (RT-safe atomics only) ────────────────────

    // ── Retire API (IRetireProvider) ──

    bool enqueueRetire(void* ptr, void (*deleter)(void*), uint64_t epoch) noexcept override
    {
        ++enqueueRetireCallCount;
        (void)deleter; (void)epoch;
        if (enqueueRetireResult.load(std::memory_order_acquire))
        {
            lastEnqueuedPtr.store(ptr, std::memory_order_release);
            return true;   // D accepts
        }
        return false;      // D rejects → caller falls through to Q
    }

    bool enqueueRetireTyped(void* ptr, void (*deleter)(void*), uint64_t epoch,
                            DeletionEntryType type) noexcept override
    {
        ++enqueueRetireTypedCallCount;
        (void)deleter; (void)epoch; (void)type;
        if (enqueueRetireResult.load(std::memory_order_acquire))
        {
            lastEnqueuedPtr.store(ptr, std::memory_order_release);
            return true;
        }
        return false;
    }

    void tryReclaim() noexcept override
    {
        ++tryReclaimCount;
    }

    std::uint32_t pendingRetireCount() const noexcept override
    {
        return 0;  // stub: no real D queue
    }

    void drainAll() noexcept override
    {
        // stub: no-op
    }

    // ── Reader API (IReaderEpochProvider) — stub (no real readers) ──

    int registerReaderThread() noexcept override                { return 0; }
    bool reserveReaderThread(int /*readerIndex*/) noexcept override { return true; }
    void enterReader(int /*readerIndex*/) noexcept override     {}
    void exitReader(int /*readerIndex*/) noexcept override      {}

    uint64_t currentEpoch() const noexcept override
    {
        return currentEpoch_.load(std::memory_order_acquire);
    }

    std::uint32_t activeReaderCount() const noexcept override
    {
        return 0;
    }

    int readerCapacity() const noexcept override                { return 1; }

    uint64_t getMinReaderEpoch() const noexcept override
    {
        return minReaderEpoch_.load(std::memory_order_acquire);
    }

    // ── Publication API (IPublicationProvider) ──

    uint64_t publishEpoch() noexcept override
    {
        ++publishEpochCount;
        return ++currentEpoch_;
    }

    // ── IEpochProvider additional virtuals (defaults are fine, override for completeness) ──

    convo::ReaderSlotDetail getReaderSlotDetail(int /*readerIndex*/) const noexcept override
    {
        return convo::ReaderSlotDetail{};
    }

    convo::StuckReaderInfo detectStuckReaders(uint64_t /*stuckThreshold*/) const noexcept override
    {
        return convo::StuckReaderInfo{};
    }

    uint64_t reclaimAttemptCount() const noexcept override  { return 0; }
    uint64_t reclaimSuccessCount()  const noexcept override  { return 0; }
    uint64_t pendingRetireBytes()   const noexcept override  { return 0; }
    uint64_t worldReclaimCount()    const noexcept override  { return 0; }

    void setReferenceObserver(void* /*observer*/) noexcept override {}
    bool quarantineReader(int /*readerIndex*/) noexcept override     { return false; }
    void unquarantineAllReaders() noexcept override                   {}
    int  quarantinedReaderCount() const noexcept override             { return 0; }
};

//============================================================================
// Counting deleter infrastructure (test-only, mirrors D8_1_WrapperCacheTests.cpp)
//============================================================================

struct TestObject
{
    int value = 42;
};

static std::atomic<int> g_deleteCount{0};
static int              g_totalCreated{0};

static void countingDeleter(void* ptr) noexcept
{
    ++g_deleteCount;
    delete static_cast<TestObject*>(ptr);
}

static void resetDeleteCount() noexcept
{
    g_deleteCount.store(0, std::memory_order_release);
}

static int getDeleteCount() noexcept
{
    return g_deleteCount.load(std::memory_order_acquire);
}

static void noopDeleter(void*) noexcept {}

} // anonymous namespace

//============================================================================
// Test-only friend access for T10 (mirrors ISRRetireRouter's RetireGraceSemanticsTestAccess).
//   SnapshotCoordinator declares `friend class SnapshotCoordinatorTestAccess;` (test-only access,
//   no behaviour change). This definition lives in the test translation unit.
//============================================================================
namespace convo {
class SnapshotCoordinatorTestAccess {
public:
    static void resetFadeStateAndRetireTarget(SnapshotCoordinator& c) noexcept
    {
        c.resetFadeStateAndRetireTarget();
    }

    // ★ C2 テスト用: target スロットの現在値（RT 違反時に残っていることの確認に使う）
    static GlobalSnapshot* targetSlot(SnapshotCoordinator& c) noexcept
    {
        return c.m_slots.loadTarget(std::memory_order_acquire);
    }
};
} // namespace convo

//============================================================================
// T1 — SnapshotCoordinator → D (Success)
//
//   SnapshotCoordinator.switchImmediate → enqueueWithRetry → IEpochProvider::enqueueRetire = true
//   → Success (D owns)
//
// Expect:
//   enqueueRetireCallCount >= 1   (D accepted)
//   quarantineResidentCount == 0  (Q not used)
//   emergency == 0, terminal == 0
//============================================================================

static bool test_T1_SnapshotD()
{
    std::printf("[T1] SnapshotCoordinator → D (Success)...\n");

    ConfigurableEpochProvider provider;         // enqueueRetireResult = true (default)
    convo::isr::ISRRetireRouter router(provider);

    auto coordinator = std::make_unique<convo::SnapshotCoordinator>(provider);
    coordinator->setRetireSink(&router);

    auto* snap1 = convo::SnapshotFactory::create(convo::SnapshotParams{});
    auto* snap2 = convo::SnapshotFactory::create(convo::SnapshotParams{});
    if (!snap1 || !snap2)
    {
        std::printf("  FAIL: SnapshotFactory::create returned null\n");
        convo::SnapshotFactory::destroy(snap1);
        convo::SnapshotFactory::destroy(snap2);
        return false;
    }

    coordinator->switchImmediate(snap1);        // first call: no old current
    coordinator->switchImmediate(snap2);        // retires snap1 via enqueueWithRetry

    // ── Verify D owns (no Q/E/T) ──
    const int  retireCalls = provider.enqueueRetireCallCount.load(std::memory_order_acquire);
    const auto qResidents  = router.quarantineResidentCount();
    const auto eResidents  = router.emergencyQuarantineResidentCount();
    const auto tResidents  = router.terminalReclaimResidentCount();

    bool pass = true;
    if (retireCalls < 1)
    {
        std::printf("  FAIL: enqueueRetireCallCount=%d, expected >= 1\n", retireCalls);
        pass = false;
    }
    if (qResidents != 0)
    {
        std::printf("  FAIL: quarantineResidentCount=%zu, expected 0\n", qResidents);
        pass = false;
    }
    if (eResidents != 0)
    {
        std::printf("  FAIL: emergencyQuarantineResidentCount=%zu, expected 0\n", eResidents);
        pass = false;
    }
    if (tResidents != 0)
    {
        std::printf("  FAIL: terminalReclaimResidentCount=%zu, expected 0\n", tResidents);
        pass = false;
    }

    // ── Cleanup ──
    // Force Q path so destructor's retireCurrentAndTarget goes to Q (not D leak)
    provider.enqueueRetireResult.store(false);
    void* dPtr = provider.lastEnqueuedPtr.exchange(nullptr, std::memory_order_acq_rel);
    if (dPtr)
        convo::SnapshotFactory::destroy(static_cast<convo::GlobalSnapshot*>(dPtr));   // snap1 was D-owned
    coordinator.reset();           // destructor: snap2 → Q (enqueueRetireResult=false)
    router.drainAllQuarantineStore(); // drains Q → snapshotDeleter → SnapshotFactory::destroy

    if (pass) std::printf("  PASS: D owns, Q/E/T empty\n");
    return pass;
}

//============================================================================
// T2 — SnapshotCoordinator → Q (QueuePressure fallback)
//
//   enqueueRetire = false → retry fails → quarantineRetireSink → ISRRetireRouter::Q
//
// Expect:
//   enqueueRetireCallCount == 2  (initial + retry)
//   quarantineResidentCount == 1 (Q owns exactly 1)
//   emergency == 0, terminal == 0
//   deleteCount == 0              (not deleted while Q owns)
//============================================================================

static bool test_T2_SnapshotQ()
{
    std::printf("[T2] SnapshotCoordinator → Q (QueuePressure fallback)...\n");

    ConfigurableEpochProvider provider;
    provider.enqueueRetireResult.store(false);   // force Q fallback
    convo::isr::ISRRetireRouter router(provider);

    auto coordinator = std::make_unique<convo::SnapshotCoordinator>(provider);
    coordinator->setRetireSink(&router);

    auto* snap1 = convo::SnapshotFactory::create(convo::SnapshotParams{});
    auto* snap2 = convo::SnapshotFactory::create(convo::SnapshotParams{});
    if (!snap1 || !snap2)
    {
        std::printf("  FAIL: SnapshotFactory::create returned null\n");
        convo::SnapshotFactory::destroy(snap1);
        convo::SnapshotFactory::destroy(snap2);
        return false;
    }

    coordinator->switchImmediate(snap1);        // first: no old current
    coordinator->switchImmediate(snap2);        // retires snap1 via enqueueWithRetry → false → Q

    // ── Verify Q owns (D rejected, exactly one transfer to Q) ──
    const int  retryCount     = provider.tryReclaimCount;       // called once between retries
    const int  retireCalls    = provider.enqueueRetireCallCount.load(std::memory_order_acquire);
    const auto qResidents     = router.quarantineResidentCount();
    const auto eResidents     = router.emergencyQuarantineResidentCount();
    const auto tResidents     = router.terminalReclaimResidentCount();
    const auto dPending       = router.pendingRetireCount();

    bool pass = true;
    if (retireCalls != 2)
    {
        std::printf("  FAIL: enqueueRetireCallCount=%d, expected 2 (initial + retry)\n", retireCalls);
        pass = false;
    }
    if (qResidents != 1)
    {
        std::printf("  FAIL: quarantineResidentCount=%zu, expected 1\n", qResidents);
        pass = false;
    }
    if (eResidents != 0)
    {
        std::printf("  FAIL: emergencyQuarantineResidentCount=%zu, expected 0\n", eResidents);
        pass = false;
    }
    if (tResidents != 0)
    {
        std::printf("  FAIL: terminalReclaimResidentCount=%zu, expected 0\n", tResidents);
        pass = false;
    }
    if (dPending != 0)
    {
        std::printf("  FAIL: pendingRetireCount=%u, expected 0 (D did not accept)\n", dPending);
        pass = false;
    }

    // ── Cleanup ──
    // snap2 will be retired to Q by destructor; drain all Q entries
    coordinator.reset();           // destructor: snap2 → Q (enqueueRetireResult already false)
    router.drainAllQuarantineStore(); // drains Q (snap1 + snap2) → snapshotDeleter → destroy

    if (pass) std::printf("  PASS: Q owns exactly 1, D/E/T empty, no double-transfer\n");
    return pass;
}

//============================================================================
// T3 — ISRRetireRouter → E (EmergencyQuarantine escalation)
//
//   Q = full (512) → D fails → Q rejects → E accepts
//
// Expect:
//   result == QueuePressure       (Q or E accepted)
//   emergencyQuarantineResidentCount == 1  (E owns the entry)
//   quarantineResidentCount     == 513     (512 Q + 1 E)
//   terminalReclaimResidentCount == 0
//   deleteCount == 0
//============================================================================

static bool test_T3_RouterE()
{
    std::printf("[T3] ISRRetireRouter Q → E (EmergencyQuarantine)...\n");
    resetDeleteCount();

    constexpr int kQCapacity = 512;

    ConfigurableEpochProvider provider;
    provider.enqueueRetireResult.store(false);   // force Q/E/T path
    convo::isr::ISRRetireRouter router(provider);

    // ── Pre-fill Q to capacity via direct quarantineRetire ──
    for (int i = 0; i < kQCapacity; ++i)
    {
        void* ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x1000 + i));
        if (!router.quarantineRetire(ptr, noopDeleter, /*epoch=*/0,
                                     DeletionEntryType::Generic,
                                     "T3:pre-fill Q", /*pubSeq=*/0, /*gen=*/0))
        {
            std::printf("  FAIL: pre-fill Q entry %d rejected (capacity reached early)\n", i);
            router.drainAllQuarantineStore();
            return false;
        }
    }

    const auto qBefore = router.quarantineResidentCount();
    if (qBefore != static_cast<std::size_t>(kQCapacity))
    {
        std::printf("  FAIL: Q pre-fill = %zu, expected %d\n", qBefore, kQCapacity);
        router.drainAllQuarantineStore();
        return false;
    }

    // ── Exercise: enqueueWithRetry should escalate D→Q(full)→E ──
    auto* obj = new TestObject{42};
    auto result = router.enqueueWithRetry(obj, countingDeleter, /*epoch=*/0, DeletionEntryType::Generic);

    const auto qAfter = router.quarantineResidentCount();
    const auto eCount = router.emergencyQuarantineResidentCount();
    const auto tCount = router.terminalReclaimResidentCount();
    const int  delCount = getDeleteCount();

    bool pass = true;
    if (result != convo::isr::RetireEnqueueResult::QueuePressure)
    {
        std::printf("  FAIL: result=%d, expected QueuePressure(%d)\n",
                    static_cast<int>(result),
                    static_cast<int>(convo::isr::RetireEnqueueResult::QueuePressure));
        pass = false;
    }
    if (eCount != 1)
    {
        std::printf("  FAIL: emergencyQuarantineResidentCount=%zu, expected 1\n", eCount);
        pass = false;
    }
    if (tCount != 0)
    {
        std::printf("  FAIL: terminalReclaimResidentCount=%zu, expected 0\n", tCount);
        pass = false;
    }
    if (delCount != 0)
    {
        std::printf("  FAIL: deleteCount=%d, expected 0 (entry stored in E, not deleted)\n", delCount);
        pass = false;
    }

    // ── Cleanup ──
    router.drainAllQuarantineStore();   // Q (512 noop) + E (1 counting) → deleteCount == 1

    if (pass) std::printf("  PASS: E owns 1, Q=512 full, T empty, deleteCount==0\n");
    return pass;
}

//============================================================================
// T4 — ISRRetireRouter → T (TerminalReclaim)
//
//   Q = full (512) + E = full (512) → D fails → Q rejects → E rejects → T accepts
//   TerminalReclaimAuthority stores() is growable → ALWAYS succeeds (T owns).
//
// Expect:
//   result == TerminalReclaim
//   terminalReclaimResidentCount == 1  (T owns the entry)
//   emergencyQuarantineResidentCount == 512  (E full)
//   deleteCount == 0
//============================================================================

static bool test_T4_RouterT()
{
    std::printf("[T4] ISRRetireRouter Q+E → T (TerminalReclaim)...\n");
    resetDeleteCount();

    constexpr int kQCapacity = 512;
    constexpr int kECapacity = 512;

    ConfigurableEpochProvider provider;
    provider.enqueueRetireResult.store(false);   // force Q/E/T path
    convo::isr::ISRRetireRouter router(provider);

    // ── Pre-fill Q to capacity ──
    for (int i = 0; i < kQCapacity; ++i)
    {
        void* ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x1000 + i));
        if (!router.quarantineRetire(ptr, noopDeleter, 0, DeletionEntryType::Generic,
                                     "T4:pre-fill Q", 0, 0))
        {
            std::printf("  FAIL: pre-fill Q entry %d rejected\n", i);
            router.drainAllQuarantineStore();
            return false;
        }
    }

    // ── Pre-fill E to capacity ──
    for (int i = 0; i < kECapacity; ++i)
    {
        void* ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x2000 + i));
        if (!router.emergencyQuarantine(ptr, noopDeleter, 0, DeletionEntryType::Generic,
                                        "T4:pre-fill E", 0, 0))
        {
            std::printf("  FAIL: pre-fill E entry %d rejected\n", i);
            router.drainAllQuarantineStore();
            return false;
        }
    }

    const auto eBefore = router.emergencyQuarantineResidentCount();
    if (eBefore != static_cast<std::size_t>(kECapacity))
    {
        std::printf("  FAIL: E pre-fill = %zu, expected %d\n", eBefore, kECapacity);
        router.drainAllQuarantineStore();
        return false;
    }

    // ── Exercise: enqueueWithRetry should escalate to Terminal ──
    auto* obj = new TestObject{42};
    auto result = router.enqueueWithRetry(obj, countingDeleter, 0, DeletionEntryType::Generic);

    const auto tCount = router.terminalReclaimResidentCount();
    const int  delCount = getDeleteCount();

    bool pass = true;
    if (result != convo::isr::RetireEnqueueResult::TerminalReclaim)
    {
        std::printf("  FAIL: result=%d, expected TerminalReclaim(%d)\n",
                    static_cast<int>(result),
                    static_cast<int>(convo::isr::RetireEnqueueResult::TerminalReclaim));
        pass = false;
    }
    if (tCount != 1)
    {
        std::printf("  FAIL: terminalReclaimResidentCount=%zu, expected 1\n", tCount);
        pass = false;
    }
    if (delCount != 0)
    {
        std::printf("  FAIL: deleteCount=%d, expected 0 (T stores, not deletes yet)\n", delCount);
        pass = false;
    }

    // ── Cleanup ──
    router.drainAllQuarantineStore();
    const int finalDelCount = getDeleteCount();
    if (finalDelCount != 1)
    {
        std::printf("  FAIL: after drain deleteCount=%d, expected 1 (Terminal owns, delete exactly once)\n",
                    finalDelCount);
        pass = false;
    }

    if (pass) std::printf("  PASS: T owns 1, growable store accepted, deleteCount==1 after drain\n");
    return pass;
}

//============================================================================
// T7 — SnapshotCoordinator ownership conservation
//
//   Verifies: caller → D OR caller → Q, never both.
//   D path: enqueueRetire=true → D accepts, Q empty.
//   Q path: enqueueRetire=false → D rejects, Q stores exactly 1.
//
//   Invariant: admitted = 1, ownership is in exactly ONE destination.
//============================================================================

static bool test_T7_SnapshotOwnershipConservation()
{
    std::printf("[T7] SnapshotCoordinator ownership conservation...\n");
    bool pass = true;

    // ── Sub-test A: D path ──
    {
        ConfigurableEpochProvider provider;  // enqueueRetireResult = true (default)
        convo::isr::ISRRetireRouter router(provider);
        auto coordinator = std::make_unique<convo::SnapshotCoordinator>(provider);
        coordinator->setRetireSink(&router);

        auto* snap1 = convo::SnapshotFactory::create(convo::SnapshotParams{});
        auto* snap2 = convo::SnapshotFactory::create(convo::SnapshotParams{});

        coordinator->switchImmediate(snap1);
        coordinator->switchImmediate(snap2);

        const int  dAccepts   = provider.enqueueRetireCallCount.load(std::memory_order_acquire);
        const auto qResidents = router.quarantineResidentCount();

        // D owns snap1: enqueueRetireCount >= 1, Q empty
        const bool dOwns = (dAccepts >= 1) && (qResidents == 0);
        if (!dOwns)
        {
            std::printf("  FAIL [A]: D path: D accepts=%d, Q=%zu — expected D≥1, Q=0\n",
                        dAccepts, qResidents);
            pass = false;
        }
        else
        {
            std::printf("  PASS [A]: D owns snap1 (enqueueRetireCount=%d), Q=0\n", dAccepts);
        }

        // Cleanup: snap1 was D-path (stub accepted, never stored) → manual destroy
        // snap2 goes to Q via destructor (enqueueRetireResult=false) → drainAllQuarantineStore
        provider.enqueueRetireResult.store(false);
        void* dPtr = provider.lastEnqueuedPtr.exchange(nullptr, std::memory_order_acq_rel);
        if (dPtr)
            convo::SnapshotFactory::destroy(static_cast<convo::GlobalSnapshot*>(dPtr));
        coordinator.reset();           // destructor: snap2 → Q
        router.drainAllQuarantineStore(); // drains Q → snapshotDeleter → destroy snap2
    }

    // ── Sub-test B: Q path ──
    {
        ConfigurableEpochProvider provider;
        provider.enqueueRetireResult.store(false);  // force Q
        convo::isr::ISRRetireRouter router(provider);
        auto coordinator = std::make_unique<convo::SnapshotCoordinator>(provider);
        coordinator->setRetireSink(&router);

        auto* snap1 = convo::SnapshotFactory::create(convo::SnapshotParams{});
        auto* snap2 = convo::SnapshotFactory::create(convo::SnapshotParams{});

        coordinator->switchImmediate(snap1);
        coordinator->switchImmediate(snap2);

        const int  dRejects    = provider.enqueueRetireCallCount.load(std::memory_order_acquire);
        const auto qResidents  = router.quarantineResidentCount();

        // Q owns snap1: D rejected (count==2 from retry), Q has exactly 1
        const bool qOwns = (qResidents == 1) && (dRejects == 2);
        if (!qOwns)
        {
            std::printf("  FAIL [B]: Q path: D calls=%d, Q=%zu — expected D=2(retry), Q=1\n",
                        dRejects, qResidents);
            pass = false;
        }
        else
        {
            std::printf("  PASS [B]: Q owns snap1 (D rejected=%d), Q=1, exactly one transfer\n", dRejects);
        }

        // Cleanup
        coordinator.reset();
        router.drainAllQuarantineStore();  // destroys snap1 + snap2 from Q
    }

    if (pass) std::printf("  PASS: ownership conserved — ptr in exactly one destination in all paths\n");
    return pass;
}

//============================================================================
// T8 — ISRRetireRouter ownership conservation (counting deleter)
//
//   Verifies:
//     - Entry admitted to ONE store (Q, E, or T) — no double transfer.
//     - After drain: deleteCount == exactly 1 (exactly-once deletion).
//============================================================================

static bool test_T8_RouterOwnershipConservation()
{
    std::printf("[T8] ISRRetireRouter ownership conservation (counting deleter)...\n");
    bool pass = true;

    // ── Sub-test A: Q path ──
    {
        resetDeleteCount();
        constexpr int kQCapacity = 512;

        ConfigurableEpochProvider provider;
        provider.enqueueRetireResult.store(false);
        convo::isr::ISRRetireRouter router(provider);

        // Pre-fill Q so the test entry is NOT stored in Q (forces E? No — for Q path test,
        // we want the entry TO go to Q, so no pre-fill needed.)
        // Actually, for pure Q path: no pre-fill, enqueueWithRetry → Q stores
        auto* obj = new TestObject{42};
        auto result = router.enqueueWithRetry(obj, countingDeleter, 0, DeletionEntryType::Generic);

        const auto qCount = router.quarantineResidentCount();
        const int  delBeforeDrain = getDeleteCount();

        bool subPass = true;
        if (result != convo::isr::RetireEnqueueResult::QueuePressure)
        {
            std::printf("  FAIL [A]: result=%d, expected QueuePressure\n", static_cast<int>(result));
            subPass = false;
        }
        if (qCount != 1)
        {
            std::printf("  FAIL [A]: Q resident=%zu, expected 1\n", qCount);
            subPass = false;
        }
        if (delBeforeDrain != 0)
        {
            std::printf("  FAIL [A]: deleteCount=%d before drain, expected 0\n", delBeforeDrain);
            subPass = false;
        }

        // Drain → deleter must fire exactly once
        router.drainAllQuarantineStore();
        const int delAfterDrain = getDeleteCount();
        if (delAfterDrain != 1)
        {
            std::printf("  FAIL [A]: deleteCount=%d after drain, expected 1\n", delAfterDrain);
            subPass = false;
        }

        if (subPass) std::printf("  PASS [A]: Q path — admitted 1, deleted exactly once\n");
        else pass = false;
    }

    // ── Sub-test B: E path ──
    {
        resetDeleteCount();
        constexpr int kQCapacity = 512;

        ConfigurableEpochProvider provider;
        provider.enqueueRetireResult.store(false);
        convo::isr::ISRRetireRouter router(provider);

        // Pre-fill Q to capacity → 513th entry escalates to E
        for (int i = 0; i < kQCapacity; ++i)
        {
            void* ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x3000 + i));
            router.quarantineRetire(ptr, noopDeleter, 0, DeletionEntryType::Generic, "T8:B prefill", 0, 0);
        }

        auto* obj = new TestObject{42};
        auto result = router.enqueueWithRetry(obj, countingDeleter, 0, DeletionEntryType::Generic);

        const auto eCount = router.emergencyQuarantineResidentCount();
        const int  delBeforeDrain = getDeleteCount();

        bool subPass = true;
        if (result != convo::isr::RetireEnqueueResult::QueuePressure)
        {
            std::printf("  FAIL [B]: result=%d, expected QueuePressure\n", static_cast<int>(result));
            subPass = false;
        }
        if (eCount != 1)
        {
            std::printf("  FAIL [B]: E resident=%zu, expected 1\n", eCount);
            subPass = false;
        }
        if (delBeforeDrain != 0)
        {
            std::printf("  FAIL [B]: deleteCount=%d before drain, expected 0\n", delBeforeDrain);
            subPass = false;
        }

        router.drainAllQuarantineStore();
        const int delAfterDrain = getDeleteCount();
        if (delAfterDrain != 1)
        {
            std::printf("  FAIL [B]: deleteCount=%d after drain, expected 1\n", delAfterDrain);
            subPass = false;
        }

        if (subPass) std::printf("  PASS [B]: E path — admitted 1, deleted exactly once\n");
        else pass = false;
    }

    // ── Sub-test C: T path ──
    {
        resetDeleteCount();
        constexpr int kQCapacity = 512;
        constexpr int kECapacity = 512;

        ConfigurableEpochProvider provider;
        provider.enqueueRetireResult.store(false);
        convo::isr::ISRRetireRouter router(provider);

        // Pre-fill Q + E to capacity → 1025th entry escalates to T
        for (int i = 0; i < kQCapacity; ++i)
        {
            void* ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x4000 + i));
            router.quarantineRetire(ptr, noopDeleter, 0, DeletionEntryType::Generic, "T8:C prefill Q", 0, 0);
        }
        for (int i = 0; i < kECapacity; ++i)
        {
            void* ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x5000 + i));
            router.emergencyQuarantine(ptr, noopDeleter, 0, DeletionEntryType::Generic, "T8:C prefill E", 0, 0);
        }

        auto* obj = new TestObject{42};
        auto result = router.enqueueWithRetry(obj, countingDeleter, 0, DeletionEntryType::Generic);

        const auto tCount = router.terminalReclaimResidentCount();
        const int  delBeforeDrain = getDeleteCount();

        bool subPass = true;
        if (result != convo::isr::RetireEnqueueResult::TerminalReclaim)
        {
            std::printf("  FAIL [C]: result=%d, expected TerminalReclaim(%d)\n",
                        static_cast<int>(result),
                        static_cast<int>(convo::isr::RetireEnqueueResult::TerminalReclaim));
            subPass = false;
        }
        if (tCount != 1)
        {
            std::printf("  FAIL [C]: T resident=%zu, expected 1\n", tCount);
            subPass = false;
        }
        if (delBeforeDrain != 0)
        {
            std::printf("  FAIL [C]: deleteCount=%d before drain, expected 0\n", delBeforeDrain);
            subPass = false;
        }

        router.drainAllQuarantineStore();
        const int delAfterDrain = getDeleteCount();
        if (delAfterDrain != 1)
        {
            std::printf("  FAIL [C]: deleteCount=%d after drain, expected 1\n", delAfterDrain);
            subPass = false;
        }

        if (subPass) std::printf("  PASS [C]: T path — admitted 1 (growable), deleted exactly once\n");
        else pass = false;
    }

    if (pass) std::printf("  PASS: all paths conserve ownership, deleteCount==1 after drain\n");
    return pass;
}




//============================================================================
// T10 — SnapshotCoordinator RT boundary: resetFadeStateAndRetireTarget
//
//   Requirement (D8-2-B-1 :28):
//     resetFadeStateAndRetireTarget only calls enqueueRetire (D), never quarantine.
//
//   The function is private and has NO production caller (updateFade was removed 2026-07-28,
//   SnapshotCoordinator.h:118), so the public API cannot reach it. This test therefore calls it
//   DIRECTLY through the test-only friend SnapshotCoordinatorTestAccess (SnapshotCoordinator.h),
//   which is the same pattern ISRRetireRouter uses for RetireGraceSemanticsTestAccess.
//
//   What is asserted (T10-a / T10-b):
//     publishEpoch delta       == 1   (one retirement epoch, taken inside the function)
//     enqueueRetire delta      == 1   (D receives the target)
//     lastEnqueuedPtr          == the target that was in the target slot
//     Q / E / T residents      == 0/0/0   (quarantine never touched)
//     router.pendingRetireCount() == 0    (the provider is the D stand-in)
//     isFading()               == false   (resetToIdle ran)
//
//   What is NOT asserted (T10-c): that the same holds when called from an RT context. There is no
//   such caller in the current design, so the RT boundary has no reachable observation point.
//============================================================================

static bool test_T10_ResetFadeStateAndRetireTarget()
{
    std::printf("[T10] resetFadeStateAndRetireTarget — direct call (D only, no quarantine)...\n");

    ConfigurableEpochProvider provider;              // enqueueRetireResult = true (D accepts)
    convo::isr::ISRRetireRouter router(provider);

    auto coordinator = std::make_unique<convo::SnapshotCoordinator>(provider);
    coordinator->setRetireSink(&router);

    auto* a = convo::SnapshotFactory::create(convo::SnapshotParams{});
    auto* b = convo::SnapshotFactory::create(convo::SnapshotParams{});
    if (!a || !b)
    {
        std::printf("  FAIL: SnapshotFactory::create returned null\n");
        convo::SnapshotFactory::destroy(a);
        convo::SnapshotFactory::destroy(b);
        return false;
    }

    // PRE-CONDITION, built with the public API only.
    //   startFade(a, fadeSamples=100) with an empty current slot forwards to switchImmediate(a)
    //   (SnapshotCoordinator.cpp:60-64), so `a` becomes current.
    //   startFade(b, fadeSamples=100) now takes the real fade path: it moves `b` INTO THE TARGET SLOT
    //   (SnapshotCoordinator.cpp:71) and starts a fade, retiring `a` from the current slot.
    //   -> the target slot holds `b`, which is exactly the state resetFadeStateAndRetireTarget()
    //      consumes. Without this step the slot is empty and the call proves nothing.
    coordinator->startFade(a, /*fadeSamples=*/100);
    coordinator->startFade(b, /*fadeSamples=*/100);
    if (!coordinator->isFading())
    {
        std::printf("  FAIL: pre-condition not met - no fade in progress (target slot may be empty)\n");
        coordinator->finalizeShutdown(/*timedOut=*/true);
        coordinator.reset();
        router.drainAllQuarantineStore();
        return false;
    }

    const int publishBefore = provider.publishEpochCount.load(std::memory_order_acquire);
    const int retireBefore  = provider.enqueueRetireCallCount.load(std::memory_order_acquire);

    convo::SnapshotCoordinatorTestAccess::resetFadeStateAndRetireTarget(*coordinator);

    const int publishDelta = provider.publishEpochCount.load(std::memory_order_acquire) - publishBefore;
    const int retireDelta  = provider.enqueueRetireCallCount.load(std::memory_order_acquire) - retireBefore;
    const auto qResidents  = router.quarantineResidentCount();
    const auto eResidents  = router.emergencyQuarantineResidentCount();
    const auto tResidents  = router.terminalReclaimResidentCount();
    const auto dPending    = router.pendingRetireCount();
    const bool fading      = coordinator->isFading();
    void* const lastPtr    = provider.lastEnqueuedPtr.load(std::memory_order_acquire);

    bool pass = true;

    // T10-a: exactly one D retirement, and it is the target that was in the target slot.
    if (publishDelta != 1)
    {
        std::printf("  FAIL: publishEpoch delta=%d, expected 1\n", publishDelta);
        pass = false;
    }
    if (retireDelta != 1)
    {
        std::printf("  FAIL: enqueueRetire delta=%d, expected 1 (D only)\n", retireDelta);
        pass = false;
    }
    if (lastPtr != static_cast<void*>(b))
    {
        std::printf("  FAIL: enqueueRetire received %p, expected the target-slot entry %p\n",
                    lastPtr, static_cast<void*>(b));
        pass = false;
    }
    // T10-b: quarantine is never touched.
    if (qResidents != 0 || eResidents != 0 || tResidents != 0)
    {
        std::printf("  FAIL: quarantine used (Q=%zu E=%zu T=%zu), expected 0/0/0\n",
                    qResidents, eResidents, tResidents);
        pass = false;
    }
    if (dPending != 0)
    {
        std::printf("  FAIL: pendingRetireCount=%u, expected 0\n", dPending);
        pass = false;
    }
    if (fading)
    {
        std::printf("  FAIL: isFading()=true, expected resetToIdle\n");
        pass = false;
    }

    coordinator->finalizeShutdown(/*timedOut=*/true);
    coordinator.reset();
    router.drainAllQuarantineStore();

    if (pass)
        std::printf("  PASS: direct call retired the target via D only"
                    " (publishEpoch=%d, enqueueRetire=%d, Q/E/T=0)"
                    " [T10-c RT boundary: no reachable caller, not asserted]\n",
                    publishDelta, retireDelta);
    return pass;
}

//============================================================================
// T11 — completeFade (C3) の要求・完了 1 対 1（2026-11-10 Owner 承認・テスト限定）
//
//   C3 は製品経路で到達しないため、既存テストで回帰を担保する。
//   実 SnapshotCoordinator・実 EpochDomain（実 DeferredDeletionQueue）・実 recorder を使う。
//   quarantine 経由でも記録が重複しないことを併せて確認する。
//============================================================================

namespace {

// enqueueRetire を常に拒否する provider（quarantine 移送の検査用）
class FullQueueEpochProvider final : public convo::EpochDomain
{
public:
    // ★ domain tag (v2.9): テスト用 provider は本番帰属を持たないため Unknown を明示する。
    FullQueueEpochProvider() : convo::EpochDomain (convo::RetireDomainId::Unknown) {}

    bool enqueueRetire(void*, void (*)(void*), uint64_t) noexcept override { return false; }
    bool enqueueRetireTyped(void*, void (*)(void*), uint64_t, DeletionEntryType) noexcept override { return false; }
};

// D/Q/E/T を epoch-gated に drain する
void drainRetireQueues(convo::EpochDomain& dom, convo::isr::ISRRetireRouter& router) noexcept
{
    for (int i = 0; i < 4; ++i)
    {
        dom.publishEpoch();
        router.tryReclaim();
    }
}

} // namespace

static bool test_T11_CompleteFadeRecords()
{
    std::printf("[T11] completeFade (C3) request/completion 1:1 (real queue)...\n");

    auto& rec = convo::isr::RetireLatencyRecorder::instance();
    bool pass = true;

    // --- (1) 正常系: D キューが受領する ---
    {
        rec.reset();
        // ★ D8 stack fix (2026-11-14): EpochDomain は約 245 KB。Debug(/Od) は
        //   スコープごとに別スロットを確保するため、複数をスタックに置くと 1 MB を超える。
        auto domStorage = std::make_unique<convo::EpochDomain>(convo::RetireDomainId::Unknown);
        auto& dom = *domStorage;
        convo::isr::ISRRetireRouter router(dom);
        convo::SnapshotCoordinator coord(dom);
        coord.setRetireSink(&router);

        auto* a = convo::SnapshotFactory::create(convo::SnapshotParams{});
        auto* b = convo::SnapshotFactory::create(convo::SnapshotParams{});
        if (a == nullptr || b == nullptr)
        {
            std::printf("  FAIL: SnapshotFactory::create returned null\n");
            return false;
        }

        coord.startFade(a, /*fadeSamples=*/100);   // current = a
        coord.startFade(b, /*fadeSamples=*/100);   // target  = b
        const auto before = rec.snapshot();

        coord.advanceFade(1000);
        const bool completed = coord.tryCompleteFade();   // 旧 current (a) を退役 = C3
        const auto after = rec.snapshot();

        if (! completed)
        {
            std::printf("  FAIL: tryCompleteFade did not complete\n");
            pass = false;
        }
        if (after.requests != before.requests + 1)
        {
            std::printf("  FAIL: requests=%llu, expected %llu\n",
                        (unsigned long long) after.requests,
                        (unsigned long long) (before.requests + 1));
            pass = false;
        }

        drainRetireQueues(dom, router);
        const auto drained = rec.snapshot();
        if (drained.completions != drained.requests || drained.unmatchedCompletions != 0)
        {
            std::printf("  FAIL: req=%llu comp=%llu unmatched=%llu\n",
                        (unsigned long long) drained.requests,
                        (unsigned long long) drained.completions,
                        (unsigned long long) drained.unmatchedCompletions);
            pass = false;
        }
        if (drained.cancelled != 0)
        {
            std::printf("  FAIL: cancelled=%llu, expected 0\n",
                        (unsigned long long) drained.cancelled);
            pass = false;
        }

        // 後始末は C5 経路（finalizeShutdown → retireCurrentAndTarget）で行う
        coord.finalizeShutdown(/*timedOut=*/false);
        drainRetireQueues(dom, router);
        const auto fin = rec.snapshot();
        if (fin.completions != fin.requests || fin.unmatchedCompletions != 0)
        {
            std::printf("  FAIL (finalize): req=%llu comp=%llu unmatched=%llu\n",
                        (unsigned long long) fin.requests,
                        (unsigned long long) fin.completions,
                        (unsigned long long) fin.unmatchedCompletions);
            pass = false;
        }
        if (fin.recordMisses != 0)
        {
            std::printf("  FAIL: recordMisses=%llu, expected 0\n",
                        (unsigned long long) fin.recordMisses);
            pass = false;
        }
    }

    // --- (2) queue 失敗 → quarantine 移送: 記録が重複しない ---
    {
        rec.reset();
        // ★ D8 stack fix (2026-11-14): FullQueueEpochProvider も EpochDomain を継承（約 245 KB）。
        auto fdomStorage = std::make_unique<FullQueueEpochProvider>();
        auto& fdom = *fdomStorage;
        convo::isr::ISRRetireRouter frouter(fdom);
        convo::SnapshotCoordinator fcoord(fdom);
        fcoord.setRetireSink(&frouter);

        auto* a = convo::SnapshotFactory::create(convo::SnapshotParams{});
        auto* b = convo::SnapshotFactory::create(convo::SnapshotParams{});
        if (a == nullptr || b == nullptr)
        {
            std::printf("  FAIL: SnapshotFactory::create returned null (Q case)\n");
            return false;
        }

        fcoord.startFade(a, 100);
        fcoord.startFade(b, 100);
        const auto before = rec.snapshot();
        fcoord.advanceFade(1000);
        const bool completed = fcoord.tryCompleteFade();   // C3 → enqueue 失敗 → Q 移送
        const auto after = rec.snapshot();

        if (! completed)
        {
            std::printf("  FAIL: tryCompleteFade (Q) did not complete\n");
            pass = false;
        }
        if (after.requests != before.requests + 1)
        {
            std::printf("  FAIL (Q): requests=%llu, expected %llu (duplicate record?)\n",
                        (unsigned long long) after.requests,
                        (unsigned long long) (before.requests + 1));
            pass = false;
        }

        drainRetireQueues(fdom, frouter);
        const auto drained = rec.snapshot();
        if (drained.completions != drained.requests || drained.unmatchedCompletions != 0)
        {
            std::printf("  FAIL (Q): req=%llu comp=%llu unmatched=%llu\n",
                        (unsigned long long) drained.requests,
                        (unsigned long long) drained.completions,
                        (unsigned long long) drained.unmatchedCompletions);
            pass = false;
        }
    }

    if (pass)
        std::printf("  PASS: C3 request/completion 1:1, no duplicate record on quarantine\n");
    return pass;
}

//============================================================================
// T12 — finalizeShutdown が current と target の両方を記録する（C5 の 2 か所）
//
//   fade 完了前に finalizeShutdown を呼ぶと、current と target の両方が
//   retireCurrentAndTarget() で退役する。要求 2 件・完了 2 件を確認する。
//============================================================================
static bool test_T12_FinalizeRetiresTargetAndCurrent()
{
    std::printf("[T12] finalizeShutdown retires current + target (C5 both sites)...\n");

    auto& rec = convo::isr::RetireLatencyRecorder::instance();
    bool pass = true;

    rec.reset();
    // ★ D8 stack fix (2026-11-14): EpochDomain は約 245 KB。Debug(/Od) は
    //   スコープごとに別スロットを確保するため、複数をスタックに置くと 1 MB を超える。
    auto domStorage = std::make_unique<convo::EpochDomain>(convo::RetireDomainId::Unknown);
    auto& dom = *domStorage;
    convo::isr::ISRRetireRouter router(dom);
    convo::SnapshotCoordinator coord(dom);
    coord.setRetireSink(&router);

    auto* a = convo::SnapshotFactory::create(convo::SnapshotParams{});
    auto* b = convo::SnapshotFactory::create(convo::SnapshotParams{});
    if (a == nullptr || b == nullptr)
    {
        std::printf("  FAIL: SnapshotFactory::create returned null\n");
        return false;
    }

    coord.startFade(a, /*fadeSamples=*/100);   // current = a
    coord.startFade(b, /*fadeSamples=*/100);   // target  = b（fade は完了させない）
    const auto before = rec.snapshot();

    coord.finalizeShutdown(/*timedOut=*/false);   // current=a と target=b を退役
    const auto after = rec.snapshot();

    if (after.requests != before.requests + 2)
    {
        std::printf("  FAIL: requests=%llu, expected %llu (current+target)\n",
                    (unsigned long long) after.requests,
                    (unsigned long long) (before.requests + 2));
        pass = false;
    }

    drainRetireQueues(dom, router);
    const auto fin = rec.snapshot();
    if (fin.completions != fin.requests || fin.unmatchedCompletions != 0)
    {
        std::printf("  FAIL: req=%llu comp=%llu unmatched=%llu\n",
                    (unsigned long long) fin.requests,
                    (unsigned long long) fin.completions,
                    (unsigned long long) fin.unmatchedCompletions);
        pass = false;
    }
    if (fin.recordMisses != 0)
    {
        std::printf("  FAIL: recordMisses=%llu, expected 0\n",
                    (unsigned long long) fin.recordMisses);
        pass = false;
    }

    if (pass)
        std::printf("  PASS: current + target retired, request/completion 2:2, unmatched 0\n");
    return pass;
}

//============================================================================
// T13 — C2 (resetFadeStateAndRetireTarget) Non-RT 専用化（2026-11-10 Owner 承認）
//
//   1. Non-RT 成功: D が受領、要求 1・完了 1
//   2. Q 移送: enqueue 失敗 → Q が受領、要求 1・完了 1（Q drain 後）
//   3. Q 満杯 → E が受領（所有権の重複なし）
//   4. Q/E 満杯 → T が受領（所有権の重複なし）
//   5. RT 違反: target・fade 状態・所有権・記録を変更しない（target が slot に残る）
//   6. RT 違反後の Non-RT 再呼出し: 正常退役（要求 1・完了 1）
//   7. 二重呼出し: 二重移送・二重記録なし
//============================================================================

namespace {

void fillQuarantine(convo::isr::ISRRetireRouter& router, int n, uint64_t epoch) noexcept
{
    static int dummy = 0;
    for (int i = 0; i < n; ++i)
        (void) router.quarantineRetire(&dummy, +[](void*) noexcept {}, epoch,
                                       DeletionEntryType::Generic, "t13-fill");
}

void fillEmergency(convo::isr::ISRRetireRouter& router, int n, uint64_t epoch) noexcept
{
    static int dummy2 = 0;
    for (int i = 0; i < n; ++i)
        (void) router.emergencyQuarantine(&dummy2, +[](void*) noexcept {}, epoch,
                                          DeletionEntryType::Generic, "t13-fill", 0, 0);
}

} // namespace

static bool test_T13_C2NonRtOnly()
{
    std::printf("[T13] C2 Non-RT only: success/Q/E/T/RT-violation/double-call...\n");

    auto& rec = convo::isr::RetireLatencyRecorder::instance();
    bool pass = true;
    const auto rtBefore = convo::SnapshotCoordinator::rtViolationCount();

    // --- (1) Non-RT 成功 ---
    {
        rec.reset();
        // ★ D8 stack fix (2026-11-14): EpochDomain は約 245 KB。Debug(/Od) は
        //   スコープごとに別スロットを確保するため、複数をスタックに置くと 1 MB を超える。
        auto domStorage = std::make_unique<convo::EpochDomain>(convo::RetireDomainId::Unknown);
        auto& dom = *domStorage;
        convo::isr::ISRRetireRouter router(dom);
        convo::SnapshotCoordinator coord(dom);
        coord.setRetireSink(&router);

        auto* a = convo::SnapshotFactory::create(convo::SnapshotParams{});
        auto* b = convo::SnapshotFactory::create(convo::SnapshotParams{});
        coord.startFade(a, 100);
        coord.startFade(b, 100);            // target = b

        const auto before = rec.snapshot();
        convo::SnapshotCoordinatorTestAccess::resetFadeStateAndRetireTarget(coord);
        const auto after = rec.snapshot();

        if (after.requests != before.requests + 1)
        {
            std::printf("  FAIL (success): requests=%llu expected %llu\n",
                        (unsigned long long) after.requests,
                        (unsigned long long) (before.requests + 1));
            pass = false;
        }
        if (convo::SnapshotCoordinatorTestAccess::targetSlot(coord) != nullptr)
        {
            std::printf("  FAIL (success): target slot not cleared\n");
            pass = false;
        }
        drainRetireQueues(dom, router);
        const auto fin = rec.snapshot();
        if (fin.completions != fin.requests || fin.unmatchedCompletions != 0 || fin.cancelled != 0)
        {
            std::printf("  FAIL (success): req=%llu comp=%llu un=%llu cancelled=%llu\n",
                        (unsigned long long) fin.requests, (unsigned long long) fin.completions,
                        (unsigned long long) fin.unmatchedCompletions, (unsigned long long) fin.cancelled);
            pass = false;
        }
        coord.finalizeShutdown(false);
        drainRetireQueues(dom, router);
    }

    // --- (2) Q 移送 ---
    {
        rec.reset();
        // ★ D8 stack fix (2026-11-14): FullQueueEpochProvider も EpochDomain を継承（約 245 KB）。
        auto fdomStorage = std::make_unique<FullQueueEpochProvider>();
        auto& fdom = *fdomStorage;
        convo::isr::ISRRetireRouter router(fdom);
        convo::SnapshotCoordinator coord(fdom);
        coord.setRetireSink(&router);

        auto* a = convo::SnapshotFactory::create(convo::SnapshotParams{});
        auto* b = convo::SnapshotFactory::create(convo::SnapshotParams{});
        coord.startFade(a, 100);
        coord.startFade(b, 100);

        const auto before = rec.snapshot();
        const auto qBefore = router.quarantineResidentCount();
        convo::SnapshotCoordinatorTestAccess::resetFadeStateAndRetireTarget(coord);
        const auto after = rec.snapshot();
        const auto qAfter = router.quarantineResidentCount();

        if (qAfter != qBefore + 1)
        {
            std::printf("  FAIL (Q): quarantine resident %zu → %zu\n", qBefore, qAfter);
            pass = false;
        }
        if (after.requests != before.requests + 1)
        {
            std::printf("  FAIL (Q): requests=%llu expected %llu\n",
                        (unsigned long long) after.requests,
                        (unsigned long long) (before.requests + 1));
            pass = false;
        }
        drainRetireQueues(fdom, router);
        const auto fin = rec.snapshot();
        if (fin.completions != fin.requests || fin.unmatchedCompletions != 0)
        {
            std::printf("  FAIL (Q): req=%llu comp=%llu un=%llu\n",
                        (unsigned long long) fin.requests, (unsigned long long) fin.completions,
                        (unsigned long long) fin.unmatchedCompletions);
            pass = false;
        }
    }

    // --- (3) Q 満杯 → E ---
    {
        rec.reset();
        // ★ D8 stack fix (2026-11-14): FullQueueEpochProvider も EpochDomain を継承（約 245 KB）。
        auto fdomStorage = std::make_unique<FullQueueEpochProvider>();
        auto& fdom = *fdomStorage;
        convo::isr::ISRRetireRouter router(fdom);
        convo::SnapshotCoordinator coord(fdom);
        coord.setRetireSink(&router);

        const uint64_t keepEpoch = fdom.currentEpoch();
        fillQuarantine(router, 512, keepEpoch);

        auto* a = convo::SnapshotFactory::create(convo::SnapshotParams{});
        auto* b = convo::SnapshotFactory::create(convo::SnapshotParams{});
        coord.startFade(a, 100);
        coord.startFade(b, 100);

        const auto eBefore = router.emergencyQuarantineResidentCount();
        convo::SnapshotCoordinatorTestAccess::resetFadeStateAndRetireTarget(coord);
        const auto eAfter = router.emergencyQuarantineResidentCount();

        if (eAfter != eBefore + 1)
        {
            std::printf("  FAIL (Q→E): emergency resident %zu → %zu\n", eBefore, eAfter);
            pass = false;
        }
        router.drainAllQuarantineStore();   // 後始末（noop deleter）
    }

    // --- (4) Q/E 満杯 → T ---
    {
        rec.reset();
        // ★ D8 stack fix (2026-11-14): FullQueueEpochProvider も EpochDomain を継承（約 245 KB）。
        auto fdomStorage = std::make_unique<FullQueueEpochProvider>();
        auto& fdom = *fdomStorage;
        convo::isr::ISRRetireRouter router(fdom);
        convo::SnapshotCoordinator coord(fdom);
        coord.setRetireSink(&router);

        const uint64_t keepEpoch = fdom.currentEpoch();
        fillQuarantine(router, 512, keepEpoch);
        fillEmergency(router, 512, keepEpoch);

        auto* a = convo::SnapshotFactory::create(convo::SnapshotParams{});
        auto* b = convo::SnapshotFactory::create(convo::SnapshotParams{});
        coord.startFade(a, 100);
        coord.startFade(b, 100);

        const auto tBefore = router.terminalReclaimResidentCount();
        const auto syncBefore = rec.snapshot().paths[
            static_cast<std::size_t>(convo::isr::RetireLatencyRecorder::Path::Sync)].count;
        convo::SnapshotCoordinatorTestAccess::resetFadeStateAndRetireTarget(coord);
        const auto tAfter = router.terminalReclaimResidentCount();
        const auto syncAfter = rec.snapshot().paths[
            static_cast<std::size_t>(convo::isr::RetireLatencyRecorder::Path::Sync)].count;

        // T は epoch 安全なら同期破棄する（resident は増えない）。どちらでも所有権は T。
        if (tAfter != tBefore + 1 && syncAfter != syncBefore + 1)
        {
            std::printf("  FAIL (Q/E->T): terminal resident %zu -> %zu, sync %llu -> %llu\n",
                        tBefore, tAfter, (unsigned long long) syncBefore, (unsigned long long) syncAfter);
            pass = false;
        }
        router.drainAllQuarantineStore();
    }

    // --- (5) RT 違反: 状態を変更しない ---
    {
        rec.reset();
        // ★ D8 stack fix (2026-11-14): EpochDomain は約 245 KB。Debug(/Od) は
        //   スコープごとに別スロットを確保するため、複数をスタックに置くと 1 MB を超える。
        auto domStorage = std::make_unique<convo::EpochDomain>(convo::RetireDomainId::Unknown);
        auto& dom = *domStorage;
        convo::isr::ISRRetireRouter router(dom);
        convo::SnapshotCoordinator coord(dom);
        coord.setRetireSink(&router);

        auto* a = convo::SnapshotFactory::create(convo::SnapshotParams{});
        auto* b = convo::SnapshotFactory::create(convo::SnapshotParams{});
        coord.startFade(a, 100);
        coord.startFade(b, 100);

        const auto before = rec.snapshot();
        const auto* targetBefore = convo::SnapshotCoordinatorTestAccess::targetSlot(coord);
        const bool fadingBefore = coord.isFading();
        const auto qBefore = router.quarantineResidentCount();

        std::thread rt([&coord]
        {
            convo::numeric_policy::ScopedThreadRole role(convo::numeric_policy::ThreadRole::AudioRealtime);
            convo::SnapshotCoordinatorTestAccess::resetFadeStateAndRetireTarget(coord);
        });
        rt.join();

        const auto after = rec.snapshot();
        const auto* targetAfter = convo::SnapshotCoordinatorTestAccess::targetSlot(coord);

        if (convo::SnapshotCoordinator::rtViolationCount() != rtBefore + 1)
        {
            std::printf("  FAIL (RT): rtViolationCount=%llu expected %llu\n",
                        (unsigned long long) convo::SnapshotCoordinator::rtViolationCount(),
                        (unsigned long long) (rtBefore + 1));
            pass = false;
        }
        if (targetAfter != targetBefore || targetAfter == nullptr)
        {
            std::printf("  FAIL (RT): target slot changed (before=%p after=%p)\n",
                        (const void*) targetBefore, (const void*) targetAfter);
            pass = false;
        }
        if (coord.isFading() != fadingBefore)
        {
            std::printf("  FAIL (RT): fade state changed\n");
            pass = false;
        }
        if (after.requests != before.requests || after.completions != before.completions)
        {
            std::printf("  FAIL (RT): record changed (req %llu→%llu comp %llu→%llu)\n",
                        (unsigned long long) before.requests, (unsigned long long) after.requests,
                        (unsigned long long) before.completions, (unsigned long long) after.completions);
            pass = false;
        }
        if (router.quarantineResidentCount() != qBefore)
        {
            std::printf("  FAIL (RT): quarantine resident changed\n");
            pass = false;
        }

        // --- (6) RT 違反後の Non-RT 再呼出し ---
        const auto before2 = rec.snapshot();
        convo::SnapshotCoordinatorTestAccess::resetFadeStateAndRetireTarget(coord);
        const auto after2 = rec.snapshot();
        if (after2.requests != before2.requests + 1)
        {
            std::printf("  FAIL (RT→NonRT): requests=%llu expected %llu\n",
                        (unsigned long long) after2.requests,
                        (unsigned long long) (before2.requests + 1));
            pass = false;
        }
        if (convo::SnapshotCoordinatorTestAccess::targetSlot(coord) != nullptr)
        {
            std::printf("  FAIL (RT→NonRT): target not retired\n");
            pass = false;
        }
        drainRetireQueues(dom, router);
        const auto fin = rec.snapshot();
        if (fin.completions != fin.requests || fin.unmatchedCompletions != 0)
        {
            std::printf("  FAIL (RT→NonRT): req=%llu comp=%llu un=%llu\n",
                        (unsigned long long) fin.requests, (unsigned long long) fin.completions,
                        (unsigned long long) fin.unmatchedCompletions);
            pass = false;
        }

        // --- (7) 二重呼出し ---
        const auto before3 = rec.snapshot();
        convo::SnapshotCoordinatorTestAccess::resetFadeStateAndRetireTarget(coord);   // 2 回目
        const auto after3 = rec.snapshot();
        if (after3.requests != before3.requests)
        {
            std::printf("  FAIL (double): requests incremented on empty target (%llu → %llu)\n",
                        (unsigned long long) before3.requests, (unsigned long long) after3.requests);
            pass = false;
        }
        coord.finalizeShutdown(false);
        drainRetireQueues(dom, router);
    }

    if (pass)
        std::printf("  PASS: Non-RT only contract, Q/E/T chain, RT violation keeps state\n");
    return pass;
}

//============================================================================
// main — test runner (mirrors RetireGraceSemanticsTests.cpp pattern)
//============================================================================
int main()
{
    bool ok = true;

    ok &= test_T1_SnapshotD();
    ok &= test_T2_SnapshotQ();
    ok &= test_T3_RouterE();
    ok &= test_T4_RouterT();
    ok &= test_T7_SnapshotOwnershipConservation();
    ok &= test_T8_RouterOwnershipConservation();
    ok &= test_T10_ResetFadeStateAndRetireTarget();
    ok &= test_T11_CompleteFadeRecords();
    ok &= test_T12_FinalizeRetiresTargetAndCurrent();
    ok &= test_T13_C2NonRtOnly();

    if (ok)
    {
        std::printf("\n========================================\n");
        std::printf("D8-2-B-2 Tests PASS\n");
        std::printf("  T1  Snapshot → D      PASS\n");
        std::printf("  T2  Snapshot → Q      PASS\n");
        std::printf("  T3  Router   → E      PASS\n");
        std::printf("  T4  Router   → T      PASS\n");
        std::printf("  T7  Snapshot conservation  PASS\n");
        std::printf("  T8  Router   conservation  PASS\n");
        std::printf("  T5  DEFERRED (dead code QueueFull)\n");
        std::printf("  T6  DEFERRED (no production Shutdown path)\n");
        std::printf("  T9  DEFERRED (source audit only)\n");
        std::printf("  T10 resetFadeStateAndRetireTarget  PASS (direct call, T10-c unobservable)\n");
        std::printf("  T11 completeFade (C3) record 1:1  PASS (real queue + quarantine)\n");
        std::printf("  T12 finalizeShutdown current+target  PASS (C5 both sites)\n");
        std::printf("  T13 C2 Non-RT only (Q/E/T + RT)      PASS\n");
        std::printf("========================================\n");
        return 0;
    }
    else
    {
        std::printf("\nD8-2-B-2 Tests FAIL\n");
        return 1;
    }
}
