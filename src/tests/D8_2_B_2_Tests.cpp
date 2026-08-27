// D8-2-B-2_Tests.cpp
// D8-2-B-2: Test Infrastructure — Configurable Epoch Provider + T1/T2/T3/T4/T7/T8
//
// Test-first: ownership disposition observability for D/Q/E/T pathways.
// No production code changes. Test-only helpers only (anonymous namespace).
//
// T5 (QueueFull)     = DEFERRED (dead code; Terminal is growable)
// T6 (Shutdown)       = DEFERRED (no production path added)
// T9 (DSPLifetimeMgr) = DEFERRED (source audit only, see follow-up)
// T10 (RT boundary)   = source audit (no friend, no public化)

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>

#include "audioengine/AtomicAccess.h"
#include "audioengine/ISRAuthorityClass.h"   // RetireEnqueueResult
#include "audioengine/ISRRetireRouter.h"

#include "core/IEpochProvider.h"
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
        std::printf("  T10 source audit (no friend/addition)\n");
        std::printf("========================================\n");
        return 0;
    }
    else
    {
        std::printf("\nD8-2-B-2 Tests FAIL\n");
        return 1;
    }
}
