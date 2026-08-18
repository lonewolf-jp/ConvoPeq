// =============================================================================
// StuckReaderFallbackDrainTests.cpp — 15-P-7: Regression test for the stuck-reader
//   fallback path added in 15-P-5 (AudioEngine.CtorDtor.cpp drainAllQuarantineStore).
//
// Verifies the ownership invariant:
//   "stuck reader が存在しても、shutdown 後に Q/E/Terminal の ownership が残留しない"
//
// The test:
//   1. Creates EpochDomain + ISRRetireRouter
//   2. Registers a reader and enters it (stuck reader — never exits)
//   3. Pushes entries into Q (via quarantineRetire), E (via emergencyQuarantine),
//      and Terminal (via terminalReclaim) directly
//   4. Calls drainAllQuarantineStore() — the stuck-reader fallback path
//   5. Verifies all Q + E + Terminal resident entries are 0 (no ownership leak)
//   6. Verifies double-drain is safe (idempotent)
//   7. Verifies no leaks / double-frees (deleter called exactly once per ptr)
//
// No production API changes. No test hooks added. Uses only existing public APIs.
// =============================================================================

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "audioengine/ISRRetireRouter.h"
#include "core/EpochDomain.h"

using convo::isr::ISRRetireRouter;
using convo::EpochDomain;

// ── Test observer: counts how many times a deleter is invoked ──
struct DeleterTracker {
    std::atomic<int> invokeCount{0};
};

// Global vector to track all allocations and verify no leaks/double-frees
struct TestObject {
    int id;
    DeleterTracker* tracker;
    static std::atomic<int> aliveCount;

    TestObject(int i, DeleterTracker* t) : id(i), tracker(t) {
        ++aliveCount;
    }
    ~TestObject() {
        --aliveCount;
    }
};
std::atomic<int> TestObject::aliveCount{0};

static void testDeleter(void* p) noexcept {
    auto* obj = static_cast<TestObject*>(p);
    ++obj->tracker->invokeCount;
    delete obj;
}

// ── Helper: fill Q to resident capacity ──
// RetireQuarantineStore::kMaxQuarantinedEntries is 512.
// We push entries via quarantineRetire (stores in Q).
// When Q is full, quarantineRetire returns false.
// NOTE: When quarantineRetire returns false, the store did NOT accept the ptr.
//   The contract says "caller must NOT delete" — but in the real code, the caller
//   would escalate to emergencyQuarantine or terminalReclaim. In this test, we
//   delete the object ourselves since we're just testing the drain path.
//   We use a separate dummy tracker for the overflow object so it doesn't affect
//   the deleter count verification for the stored entries.
static int fillQuarantineStore(ISRRetireRouter& router, int startId, DeleterTracker& tracker) {
    int count = 0;
    uint64_t epoch = router.currentEpoch() + 1000;  // epoch in future → not safe → stays in Q
    while (true) {
        auto* obj = new TestObject(startId + count, &tracker);
        bool stored = router.quarantineRetire(
            obj, testDeleter, epoch,
            DeletionEntryType::Generic, "15-P-7:Q-fill");
        if (!stored) {
            // Q full — store did NOT accept ptr. Delete it ourselves with a dummy tracker.
            DeleterTracker overflowTracker;
            obj->tracker = &overflowTracker;
            testDeleter(obj);
            break;
        }
        ++count;
    }
    return count;
}

// ── Helper: fill E (EmergencyQuarantineStore) ──
// EmergencyQuarantineStore is also a RetireQuarantineStore with kMaxQuarantinedEntries = 512.
// We push directly via emergencyQuarantine (stores in E).
static int fillEmergencyStore(ISRRetireRouter& router, int startId, DeleterTracker& tracker) {
    int count = 0;
    uint64_t epoch = router.currentEpoch() + 1000;
    while (true) {
        auto* obj = new TestObject(startId + count, &tracker);
        bool stored = router.emergencyQuarantine(
            obj, testDeleter, epoch,
            DeletionEntryType::Generic, "15-P-7:E-fill");
        if (!stored) {
            // E full — store did NOT accept ptr. Delete it ourselves with a dummy tracker.
            DeleterTracker overflowTracker;
            obj->tracker = &overflowTracker;
            testDeleter(obj);
            break;
        }
        ++count;
    }
    return count;
}

// ── Helper: fill Terminal (TerminalReclaimAuthority) ──
// TerminalReclaimAuthority is growable (std::vector) — store() ALWAYS returns true.
// We use an epoch in the future so terminalReclaim does NOT destroy immediately
// (it checks isOlder(epoch, minReaderEpoch) — future epoch is NOT older → stored).
static int fillTerminalStore(ISRRetireRouter& router, int startId, DeleterTracker& tracker, int n) {
    int count = 0;
    uint64_t epoch = router.currentEpoch() + 1000;  // future epoch → NOT epoch-safe → stored
    for (int i = 0; i < n; ++i) {
        auto* obj = new TestObject(startId + i, &tracker);
        bool stored = router.terminalReclaim(
            obj, testDeleter, epoch,
            DeletionEntryType::Generic, "15-P-7:Terminal-fill");
        if (stored) ++count;
    }
    return count;
}

// ── Test 1: Stuck reader + Q/E/Terminal populated → drainAllQuarantineStore clears all ──
[[nodiscard]] bool testStuckReaderFallbackDrainsAllStores() {
    TestObject::aliveCount = 0;
    EpochDomain epochDomain;
    ISRRetireRouter router(epochDomain);

    // Step 1: Register a reader and enter it — this creates a "stuck reader"
    //   activeReaderCount() > 0 → the stuck-reader fallback path is taken
    int readerIdx = epochDomain.registerReaderThread("TestReader");
    if (readerIdx < 0) return false;  // registration failed
    epochDomain.enterReader(readerIdx);

    // Verify stuck reader is active
    if (router.activeReaderCount() == 0) return false;  // stuck reader not active

    DeleterTracker trackerQ, trackerE, trackerT;

    // Step 2: Fill Q (quarantineRetire stores into m_retireQuarantine)
    int qCount = fillQuarantineStore(router, 1, trackerQ);
    if (qCount == 0) return false;  // couldn't fill Q
    // quarantineResidentCount() = Q + E; after filling only Q, it should equal qCount
    if (router.quarantineResidentCount() != static_cast<std::size_t>(qCount)) return false;

    // Step 3: Fill E (emergencyQuarantine stores into m_emergencyQuarantine)
    int eCount = fillEmergencyStore(router, 1000, trackerE);
    if (eCount == 0) return false;  // couldn't fill E
    // After filling E, quarantineResidentCount() = Q + E = qCount + eCount
    if (router.quarantineResidentCount() != static_cast<std::size_t>(qCount + eCount)) return false;
    if (router.emergencyQuarantineResidentCount() != static_cast<std::size_t>(eCount)) return false;

    // Step 4: Fill Terminal (terminalReclaim stores into m_terminalReclaim)
    int tCount = 50;
    fillTerminalStore(router, 2000, trackerT, tCount);
    if (router.terminalReclaimResidentCount() != static_cast<std::size_t>(tCount)) return false;

    // Step 5: Call drainAllQuarantineStore() — this is the 15-P-5 fix path
    router.drainAllQuarantineStore();

    // Step 6: Verify ALL stores are empty
    if (router.quarantineResidentCount() != 0) return false;
    if (router.emergencyQuarantineResidentCount() != 0) return false;
    if (router.terminalReclaimResidentCount() != 0) return false;

    // Step 7: Verify deleters called exactly once per entry
    if (trackerQ.invokeCount != qCount) return false;
    if (trackerE.invokeCount != eCount) return false;
    if (trackerT.invokeCount != tCount) return false;

    // Step 8: Verify no leaks (all TestObjects destroyed)
    if (TestObject::aliveCount != 0) return false;

    // Cleanup: exit the stuck reader
    epochDomain.exitReader(readerIdx);
    return true;
}

// ── Test 2: Double drain is safe (idempotent) ──
[[nodiscard]] bool testDoubleDrainIsSafe() {
    TestObject::aliveCount = 0;
    EpochDomain epochDomain;
    ISRRetireRouter router(epochDomain);

    int readerIdx = epochDomain.registerReaderThread("TestReader2");
    if (readerIdx < 0) return false;
    epochDomain.enterReader(readerIdx);

    DeleterTracker tracker;

    // Fill Q
    int qCount = fillQuarantineStore(router, 1, tracker);
    if (qCount == 0) return false;

    // Fill Terminal with 10 entries
    fillTerminalStore(router, 1000, tracker, 10);

    // First drain
    router.drainAllQuarantineStore();
    if (router.quarantineResidentCount() != 0) return false;
    if (router.emergencyQuarantineResidentCount() != 0) return false;
    if (router.terminalReclaimResidentCount() != 0) return false;

    // Second drain — should be a no-op (all stores empty)
    router.drainAllQuarantineStore();
    if (router.quarantineResidentCount() != 0) return false;
    if (router.emergencyQuarantineResidentCount() != 0) return false;
    if (router.terminalReclaimResidentCount() != 0) return false;

    // No double-free: deleters called exactly once
    if (tracker.invokeCount != qCount + 10) return false;
    if (TestObject::aliveCount != 0) return false;

    epochDomain.exitReader(readerIdx);
    return true;
}

// ── Test 3: No stuck reader (activeReaderCount == 0) — drainAll + drainAllQuarantineStore both work ──
[[nodiscard]] bool testNoStuckReaderDrainWorks() {
    TestObject::aliveCount = 0;
    EpochDomain epochDomain;
    ISRRetireRouter router(epochDomain);

    // No reader registered — activeReaderCount() == 0
    if (router.activeReaderCount() != 0) return false;

    DeleterTracker tracker;
    fillTerminalStore(router, 1, tracker, 30);
    if (router.terminalReclaimResidentCount() != 30) return false;

    // drainAll() calls both provider_->drainAll() (D) and drainAllQuarantineStore() (Q+E+Terminal)
    router.drainAll();

    if (router.terminalReclaimResidentCount() != 0) return false;
    if (router.quarantineResidentCount() != 0) return false;
    if (router.emergencyQuarantineResidentCount() != 0) return false;
    if (tracker.invokeCount != 30) return false;
    if (TestObject::aliveCount != 0) return false;

    return true;
}

// ── Test 4: Shutdown after drain — completion invariant holds ──
[[nodiscard]] bool testShutdownCompletesAfterDrain() {
    EpochDomain epochDomain;
    ISRRetireRouter router(epochDomain);

    int readerIdx = epochDomain.registerReaderThread("TestReader4");
    if (readerIdx < 0) return false;
    epochDomain.enterReader(readerIdx);

    // Push some entries into Q and Terminal
    DeleterTracker tracker;
    fillQuarantineStore(router, 1, tracker);
    fillTerminalStore(router, 2000, tracker, 20);

    // Drain
    router.drainAllQuarantineStore();

    // Verify clean state
    if (router.quarantineResidentCount() != 0) return false;
    if (router.terminalReclaimResidentCount() != 0) return false;

    // Now exit the stuck reader — activeReaderCount should be 0
    epochDomain.exitReader(readerIdx);
    if (router.activeReaderCount() != 0) return false;

    return true;
}

// ── Test 5: Ownership transfer — no ptr held by two authorities ──
// Verifies that when we push to Q then E then Terminal, the Q+E counts
// match exactly what we pushed (no double-counting / no lost entries).
[[nodiscard]] bool testOwnershipTransferNoLeaks() {
    EpochDomain epochDomain;
    ISRRetireRouter router(epochDomain);

    // No stuck reader — activeReaderCount == 0
    if (router.activeReaderCount() != 0) return false;

    DeleterTracker trackerQ, trackerE, trackerT;

    int qCount = fillQuarantineStore(router, 1, trackerQ);
    int eCount = fillEmergencyStore(router, 1000, trackerE);
    int tCount = 20;
    fillTerminalStore(router, 2000, trackerT, tCount);

    // Verify counts match exactly (ownership is in the right place)
    // quarantineResidentCount() = Q + E combined
    if (router.quarantineResidentCount() != static_cast<std::size_t>(qCount + eCount)) return false;
    if (router.emergencyQuarantineResidentCount() != static_cast<std::size_t>(eCount)) return false;
    if (router.terminalReclaimResidentCount() != static_cast<std::size_t>(tCount)) return false;

    // Total objects alive should be Q + E + T (all still alive, not yet drained)
    if (TestObject::aliveCount != qCount + eCount + tCount) return false;

    // Drain everything
    router.drainAllQuarantineStore();

    // All counts must be 0
    if (router.quarantineResidentCount() != 0) return false;
    if (router.emergencyQuarantineResidentCount() != 0) return false;
    if (router.terminalReclaimResidentCount() != 0) return false;

    // All deleters called exactly once
    if (trackerQ.invokeCount != qCount) return false;
    if (trackerE.invokeCount != eCount) return false;
    if (trackerT.invokeCount != tCount) return false;

    // No leaks, no double-frees
    if (TestObject::aliveCount != 0) return false;

    return true;
}

// ── Test 6: Stuck reader + drainAll() path (not just drainAllQuarantineStore) ──
// drainAll() calls provider_->drainAll() (D) AND drainAllQuarantineStore() (Q+E+Terminal).
// With a stuck reader, D is epoch-gated but Q+E+Terminal are epoch-agnostic.
[[nodiscard]] bool testStuckReaderDrainAllPath() {
    EpochDomain epochDomain;
    ISRRetireRouter router(epochDomain);

    int readerIdx = epochDomain.registerReaderThread("TestReader6");
    if (readerIdx < 0) return false;
    epochDomain.enterReader(readerIdx);

    DeleterTracker tracker;
    // Fill Terminal with entries using future epoch (so they're stored, not destroyed)
    int tCount = 15;
    fillTerminalStore(router, 1, tracker, tCount);
    if (router.terminalReclaimResidentCount() != static_cast<std::size_t>(tCount)) return false;

    // drainAll() — even with stuck reader, drainAllQuarantineStore is called
    router.drainAll();

    // Terminal should be drained (epoch-agnostic)
    if (router.terminalReclaimResidentCount() != 0) return false;
    if (router.quarantineResidentCount() != 0) return false;
    if (router.emergencyQuarantineResidentCount() != 0) return false;
    if (tracker.invokeCount != tCount) return false;
    if (TestObject::aliveCount != 0) return false;

    epochDomain.exitReader(readerIdx);
    return true;
}

int main() {
    int failures = 0;

    if (!testStuckReaderFallbackDrainsAllStores()) {
        std::fprintf(stderr, "FAIL: testStuckReaderFallbackDrainsAllStores\n");
        ++failures;
    }
    if (!testDoubleDrainIsSafe()) {
        std::fprintf(stderr, "FAIL: testDoubleDrainIsSafe\n");
        ++failures;
    }
    if (!testNoStuckReaderDrainWorks()) {
        std::fprintf(stderr, "FAIL: testNoStuckReaderDrainWorks\n");
        ++failures;
    }
    if (!testShutdownCompletesAfterDrain()) {
        std::fprintf(stderr, "FAIL: testShutdownCompletesAfterDrain\n");
        ++failures;
    }
    if (!testOwnershipTransferNoLeaks()) {
        std::fprintf(stderr, "FAIL: testOwnershipTransferNoLeaks\n");
        ++failures;
    }
    if (!testStuckReaderDrainAllPath()) {
        std::fprintf(stderr, "FAIL: testStuckReaderDrainAllPath\n");
        ++failures;
    }

    if (failures == 0) {
        std::printf("15-P-7: All StuckReaderFallbackDrain tests PASS (%d tests)\n", 6);
    } else {
        std::fprintf(stderr, "15-P-7: %d test(s) FAILED\n", failures);
        throw std::runtime_error("StuckReaderFallbackDrain tests failed");
    }

    return 0;
}
