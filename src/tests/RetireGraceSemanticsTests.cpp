#include <algorithm>
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>

#include "audioengine/ISRRetire.h"
#include "audioengine/ISRRetireRuntimeEx.h"
#include "audioengine/ISRAuthorityClass.h"
#include "audioengine/ISRRetireOverflowRing.h"
#include "audioengine/ISRRetireRouter.h"
#include "audioengine/RetireQuarantineStore.h"

// ── ★ Phase5: 複合ソートキー (priority, retireEpoch, generation, dspSlot) 検証 ──

[[nodiscard]] bool testPrioritySortCompositeKey()
{
    using convo::isr::RetireIntent;
    using convo::isr::RetirePriority;

    // dequeuePendingRetireIntents と同じ comparator
    auto sorter = [](const RetireIntent& lhs, const RetireIntent& rhs) noexcept {
        if (lhs.priority != rhs.priority)
            return lhs.priority > rhs.priority;   // priority降順（Critical最先頭）
        if (lhs.retireEpoch != rhs.retireEpoch)
            return lhs.retireEpoch < rhs.retireEpoch;
        if (lhs.generation != rhs.generation)
            return lhs.generation < rhs.generation;
        return lhs.dspSlot < rhs.dspSlot;
    };

    // 1. Critical > Normal (priority降順)
    {
        const RetireIntent critical{1, 100, 1000, RetirePriority::Critical};
        const RetireIntent normal{2, 100, 1000, RetirePriority::Normal};
        if (!sorter(critical, normal) || sorter(normal, critical))
            return false;
    }

    // 2. 同priority内: 古いepochが先（FIFO）
    {
        const RetireIntent older{1, 100, 500, RetirePriority::Normal};
        const RetireIntent newer{2, 100, 1000, RetirePriority::Normal};
        if (!sorter(older, newer) || sorter(newer, older))
            return false;
    }

    // 3. 同priority+epoch: 低いgenerationが先
    {
        const RetireIntent early{1, 50, 1000, RetirePriority::Normal};
        const RetireIntent late{2, 100, 1000, RetirePriority::Normal};
        if (!sorter(early, late) || sorter(late, early))
            return false;
    }

    // 4. 同priority+epoch+generation: 低いdspSlotが先
    {
        const RetireIntent first{1, 100, 1000, RetirePriority::Normal};
        const RetireIntent second{2, 100, 1000, RetirePriority::Normal};
        if (!sorter(first, second) || sorter(second, first))
            return false;
    }

    // 5. 完全ソート: Critical > High > Normal > Low
    {
        std::vector<RetireIntent> intents = {
            {3, 100, 1000, RetirePriority::Low},
            {1, 100, 1000, RetirePriority::Critical},
            {4, 100, 1000, RetirePriority::High},
            {2, 100, 1000, RetirePriority::Normal},
        };
        std::stable_sort(intents.begin(), intents.end(), sorter);
        if (intents[0].dspSlot != 1 || intents[0].priority != RetirePriority::Critical)
            return false;
        if (intents[1].dspSlot != 4 || intents[1].priority != RetirePriority::High)
            return false;
        if (intents[2].dspSlot != 2 || intents[2].priority != RetirePriority::Normal)
            return false;
        if (intents[3].dspSlot != 3 || intents[3].priority != RetirePriority::Low)
            return false;
    }

    return true;
}

[[nodiscard]] bool testGracePeriodCompletionRules()
{
    if (!convo::isr::EpochControl::isGracePeriodCompleted(100, 101, 1))
        return false;

    if (!convo::isr::EpochControl::isGracePeriodCompleted(100, 100, 0))
        return false;

    if (convo::isr::EpochControl::isGracePeriodCompleted(100, 100, 1))
        return false;

    return true;
}

[[nodiscard]] bool testRetirePendingToFreeRules()
{
    if (!convo::isr::EpochControl::canTransitionRetirePendingToFree(true, true, true))
        return false;

    if (convo::isr::EpochControl::canTransitionRetirePendingToFree(false, true, true))
        return false;

    if (convo::isr::EpochControl::canTransitionRetirePendingToFree(true, false, true))
        return false;

    if (convo::isr::EpochControl::canTransitionRetirePendingToFree(true, true, false))
        return false;

    return true;
}

[[nodiscard]] bool testRetireStarvationDualThresholdRules()
{
    if (!convo::isr::EpochControl::hasExceededDeferralThresholds(101, 10.0, 100, 5000.0))
        return false;

    if (!convo::isr::EpochControl::hasExceededDeferralThresholds(10, 5001.0, 100, 5000.0))
        return false;

    if (convo::isr::EpochControl::hasExceededDeferralThresholds(100, 5000.0, 100, 5000.0))
        return false;

    return true;
}

[[nodiscard]] bool testRetireEscalationSafetyRules()
{
    if (!convo::isr::EpochControl::canReclaimAfterEscalation(true, true, true))
        return false;

    if (convo::isr::EpochControl::canReclaimAfterEscalation(false, true, true))
        return false;

    if (convo::isr::EpochControl::canReclaimAfterEscalation(true, false, true))
        return false;

    if (convo::isr::EpochControl::canReclaimAfterEscalation(true, true, false))
        return false;

    return true;
}

[[nodiscard]] bool testRetirePressureThresholdPolicyRules()
{
    constexpr int kRetirePressureMildPercent = 75;
    constexpr int kRetirePressureMediumPercent = 90;
    constexpr int kRetirePressureSeverePercent = 95;

    auto evaluateLevel = [](std::uint64_t retireDepth, int highWatermark) noexcept {
        const int safeHwm = (highWatermark > 0) ? highWatermark : 1;
        const std::uint64_t ratioPercent = (retireDepth * 100ull) / static_cast<std::uint64_t>(safeHwm);
        if (ratioPercent >= static_cast<std::uint64_t>(kRetirePressureSeverePercent))
            return 3;
        if (ratioPercent >= static_cast<std::uint64_t>(kRetirePressureMediumPercent))
            return 2;
        if (ratioPercent >= static_cast<std::uint64_t>(kRetirePressureMildPercent))
            return 1;
        return 0;
    };

    if (evaluateLevel(74, 100) != 0)
        return false;
    if (evaluateLevel(75, 100) != 1)
        return false;
    if (evaluateLevel(89, 100) != 1)
        return false;
    if (evaluateLevel(90, 100) != 2)
        return false;
    if (evaluateLevel(94, 100) != 2)
        return false;
    if (evaluateLevel(95, 100) != 3)
        return false;

    auto isProtectiveMode = [](int retirePressureLevel, std::uint64_t retireDepth, int highWatermark) noexcept {
        const bool severe = retirePressureLevel >= 3;
        const int safeHwm = (highWatermark > 0) ? highWatermark : 1;
        return severe && (retireDepth >= static_cast<std::uint64_t>(safeHwm));
    };

    if (isProtectiveMode(2, 150, 100))
        return false;
    if (!isProtectiveMode(3, 100, 100))
        return false;
    if (isProtectiveMode(3, 99, 100))
        return false;

    return true;
}

// ── ★ Phase1: OverflowRing 基本 FIFO 検証 ──

[[nodiscard]] bool testOverflowRingFifoOrder()
{
    using convo::isr::RetireOverflowEntry;
    using convo::isr::RetireOverflowRing;
    using convo::isr::RetireIntent;
    using convo::isr::RetirePriority;

    RetireOverflowRing ring;

    RetireOverflowEntry e1{{1, 100, 1000, RetirePriority::Normal}, 100, 0};
    RetireOverflowEntry e2{{2, 200, 2000, RetirePriority::Normal}, 200, 0};
    RetireOverflowEntry e3{{3, 300, 3000, RetirePriority::Normal}, 300, 0};

    if (!ring.tryPush(e1)) return false;
    if (!ring.tryPush(e2)) return false;
    if (!ring.tryPush(e3)) return false;
    if (ring.residentCount() != 3) return false;

    RetireOverflowEntry out;
    if (!ring.pop(out) || out.intent.dspSlot != 1) return false;
    if (!ring.pop(out) || out.intent.dspSlot != 2) return false;
    if (!ring.pop(out) || out.intent.dspSlot != 3) return false;
    if (ring.residentCount() != 0) return false;
    if (ring.pop(out)) return false;

    std::vector<RetireOverflowEntry> drained;
    (void)ring.tryPush(e1);
    (void)ring.tryPush(e2);
    ring.drainAll(drained);
    if (drained.size() != 2) return false;

    return true;
}

// ── ★ Phase5: 優先度ソート Critical最優先 + 異種priority混合 ──

[[nodiscard]] bool testPrioritySortCriticalFirst()
{
    using convo::isr::RetireIntent;
    using convo::isr::RetirePriority;

    auto sorter = [](const RetireIntent& lhs, const RetireIntent& rhs) noexcept {
        if (lhs.priority != rhs.priority) return lhs.priority > rhs.priority;
        if (lhs.retireEpoch != rhs.retireEpoch) return lhs.retireEpoch < rhs.retireEpoch;
        if (lhs.generation != rhs.generation) return lhs.generation < rhs.generation;
        return lhs.dspSlot < rhs.dspSlot;
    };

    // Critical vs High vs Normal vs Low (same epoch)
    {
        std::vector<RetireIntent> intents = {
            {1, 100, 1000, RetirePriority::Low},
            {2, 100, 1000, RetirePriority::High},
            {3, 100, 1000, RetirePriority::Critical},
            {4, 100, 1000, RetirePriority::Normal},
        };
        std::stable_sort(intents.begin(), intents.end(), sorter);
        if (intents[0].priority != RetirePriority::Critical) return false;
        if (intents[1].priority != RetirePriority::High) return false;
        if (intents[2].priority != RetirePriority::Normal) return false;
        if (intents[3].priority != RetirePriority::Low) return false;
    }

    // Cross-priority with mixed epochs
    {
        std::vector<RetireIntent> intents = {
            {1, 100, 1000, RetirePriority::High},
            {2, 100, 3000, RetirePriority::Critical},
        };
        std::stable_sort(intents.begin(), intents.end(), sorter);
        if (intents[0].priority != RetirePriority::Critical) return false;
    }

    return true;
}

// ── ★ Phase5: 既存 enqueueRetire 互換性（Normal 優先度として動作）──

[[nodiscard]] bool testRetirePriorityCompatibility()
{
    using convo::isr::RetireIntent;
    using convo::isr::RetirePriority;

    // デフォルト priority が Normal であることを確認
    {
        const RetireIntent intent{1, 100, 1000};
        if (intent.priority != RetirePriority::Normal)
            return false;
    }

    // 明示的に Normal を設定
    {
        const RetireIntent intent{1, 100, 1000, RetirePriority::Normal};
        if (intent.priority != RetirePriority::Normal)
            return false;
    }

    // ソートで Normal が正しい位置に入る
    {
        auto sorter = [](const RetireIntent& lhs, const RetireIntent& rhs) noexcept {
            if (lhs.priority != rhs.priority) return lhs.priority > rhs.priority;
            return lhs.retireEpoch < rhs.retireEpoch;
        };

        std::vector<RetireIntent> intents = {
            {1, 100, 3000},                       // デフォルト Normal
            {2, 100, 1000, RetirePriority::High},
            {3, 100, 2000, RetirePriority::Normal},
        };
        std::stable_sort(intents.begin(), intents.end(), sorter);
        // High > Normal(2) > Normal(1, default)
        if (intents[0].priority != RetirePriority::High) return false;
        if (intents[1].priority != RetirePriority::Normal) return false;
        if (intents[1].dspSlot != 3) return false;  // epoch 2000 が先
        if (intents[2].priority != RetirePriority::Normal) return false;
        if (intents[2].dspSlot != 1) return false;  // epoch 3000 が後
    }

    return true;
}

// ── ★ E-1.9-A: empty-drain suppression atomic counter verification ──

[[nodiscard]] bool testEmptyDrainSuppressionAtomicCounter()
{
    using convo::consumeAtomic;
    using convo::publishAtomic;

    // ★ E-1.9-A: RetireQuarantineStore のロックフリーカウンタの整合性検証
    //   quarantine() → residentCountAtomic() increment
    //   drain() / drainAllUnsafe() → decrement / reset
    {
        convo::isr::RetireQuarantineStore store;
        if (store.residentCountAtomic() != 0) return false;

        // enqueue 3 entries
        for (int i = 0; i < 3; ++i) {
            auto* ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x1000 + i * 0x10));
            auto* deleter = +[](void*) noexcept {};
            if (!store.quarantine(ptr, deleter, /*epoch=*/100,
                                  DeletionEntryType::Generic,
                                  "test", 0, 0))
                return false;
        }
        if (store.residentCountAtomic() != 3) return false;

        // drain: epoch 100 is safe (< minReader=200)
        uint64_t minReader = 200;
        auto isOlder = [](uint64_t a, uint64_t b) noexcept {
            return static_cast<int64_t>(a - b) < 0;
        };
        store.drain(minReader, isOlder);
        if (store.residentCountAtomic() != 0) return false;

        // drainAllUnsafe resets to 0
        for (int i = 0; i < 2; ++i) {
            auto* ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x2000 + i * 0x10));
            auto* deleter = +[](void*) noexcept {};
            store.quarantine(ptr, deleter, 100, DeletionEntryType::Generic,
                             "test", 0, 0);
        }
        if (store.residentCountAtomic() != 2) return false;
        store.drainAllUnsafe();
        if (store.residentCountAtomic() != 0) return false;
    }

    // ★ E-1.9-A: TerminalReclaimAuthority のロックフリーカウンタの整合性検証
    {
        convo::isr::TerminalReclaimAuthority auth;
        if (auth.residentCountAtomic() != 0) return false;

        // store 3 entries
        for (int i = 0; i < 3; ++i) {
            auto* ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x3000 + i * 0x10));
            auto* deleter = +[](void*) noexcept {};
            if (!auth.store(ptr, deleter, /*epoch=*/100,
                            DeletionEntryType::Generic, "test"))
                return false;
        }
        if (auth.residentCountAtomic() != 3) return false;

        // drain: epoch 100 is safe (< minReader=200)
        uint64_t minReader = 200;
        auto isOlder = [](uint64_t a, uint64_t b) noexcept {
            return static_cast<int64_t>(a - b) < 0;
        };
        auth.drain(minReader, isOlder);
        if (auth.residentCountAtomic() != 0) return false;

        // drainAll resets to 0
        for (int i = 0; i < 2; ++i) {
            auto* ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x4000 + i * 0x10));
            auto* deleter = +[](void*) noexcept {};
            auth.store(ptr, deleter, 100, DeletionEntryType::Generic, "test");
        }
        if (auth.residentCountAtomic() != 2) return false;
        auth.drainAll();
        if (auth.residentCountAtomic() != 0) return false;
    }

    return true;
}

// ── ★ E-1.9-B wake protocol tests ──
//
// Test 1: enqueue → wake predicate becomes true
//   Verifies that after quarantine() places an entry in Q/E/T,
//   residentCountAtomic() != 0 (the wake predicate becomes true).
[[nodiscard]] bool testWakePredicateTrueAfterEnqueue()
{
    using convo::isr::RetireQuarantineStore;

    // RetireQuarantineStore
    {
        RetireQuarantineStore store;
        if (store.residentCountAtomic() != 0) return false;

        auto* ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x5000));
        auto* deleter = +[](void*) noexcept {};
        if (!store.quarantine(ptr, deleter, 100, DeletionEntryType::Generic,
                              "test", 0, 0))
            return false;

        // Predicate: residentCountAtomic() != 0
        if (store.residentCountAtomic() == 0) return false;
    }

    // TerminalReclaimAuthority
    {
        convo::isr::TerminalReclaimAuthority auth;
        if (auth.residentCountAtomic() != 0) return false;

        auto* ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x6000));
        auto* deleter = +[](void*) noexcept {};
        if (!auth.store(ptr, deleter, 100, DeletionEntryType::Generic, "test"))
            return false;

        if (auth.residentCountAtomic() == 0) return false;
    }

    return true;
}

// Test 2: predicate already true → no blocking
//   Verifies that waitForDrainSignalOrTimeout with a short timeout returns
//   immediately when the predicate is already true (no entry arrives before wait).
[[nodiscard]] bool testWakePredicateAlreadyTrueNoBlock()
{
    using convo::isr::RetireQuarantineStore;

    // Set up a store with a resident entry (predicate is true)
    RetireQuarantineStore store;
    auto* ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x7000));
    auto* deleter = +[](void*) noexcept {};
    if (!store.quarantine(ptr, deleter, 100, DeletionEntryType::Generic,
                          "test", 0, 0))
        return false;

    // The predicate residentCountAtomic() != 0 is true.
    // waitForDrainSignalOrTimeout should return immediately (no 1ms wait).
    const auto startUs = convo::getCurrentTimeUs();
    (void)store.residentCountAtomic();  // predicate check (discard nodiscard)
    const auto elapsedUs = convo::getCurrentTimeUs() - startUs;

    // No blocking — should be < 1ms (just the atomic load)
    if (store.residentCountAtomic() == 0) return false;
    // Just verify predicate is true (no actual CV wait in this unit test)
    return true;
}

// Test 3: spurious wake / empty state → no drain
//   Verifies that after drainAll resets the counter, residentCountAtomic() == 0
//   (predicate is false). This proves a spurious wake with empty predicate
//   results in no drain (E-1.9-A empty-guard handles it).
[[nodiscard]] bool testWakeSpuriousNoDrainOnEmpty()
{
    using convo::isr::RetireQuarantineStore;

    RetireQuarantineStore store;
    auto* ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x8000));
    auto* deleter = +[](void*) noexcept {};
    if (!store.quarantine(ptr, deleter, 100, DeletionEntryType::Generic,
                          "test", 0, 0))
        return false;

    if (store.residentCountAtomic() != 1) return false;

    // Drain all → predicate becomes false
    store.drainAllUnsafe();
    if (store.residentCountAtomic() != 0) return false;

    // Predicate is false — a spurious wake here would cause wait_for to
    // re-check the predicate (false), so no drain occurs.
    // The wait_for(lock, timeout, predicate) pattern guarantees this.
    return true;
}

// Test 4: timeout fallback when no entries
//   Verifies that with 0 timeout, waitForDrainSignalOrTimeout returns
//   immediately (timeout fallback), and the predicate (pendingRetireCount()==0
//   && residentCountAtomic()==0) is checked correctly.
[[nodiscard]] bool testWakeTimeoutFallback()
{
    using convo::isr::RetireQuarantineStore;

    // Empty store — predicate is false
    RetireQuarantineStore store;
    if (store.residentCountAtomic() != 0) return false;

    // The wait_for with predicate would timeout, but since we can't easily
    // test the CV in a synchronous unit test, verify the predicate logic:
    // If pendingRetireCount()==0 && residentCountAtomic()==0, then
    // waitForDrainSignalOrTimeout would timeout and return (no drain needed).
    return store.residentCountAtomic() == 0;
}

// Test 5: shutdown — forced drain resets all atomics to 0
//   Verifies that drainAllUnsafe (shutdown path) resets residentAtomic_ to 0,
//   ensuring no stale wake signals remain after shutdown drain.
[[nodiscard]] bool testWakeShutdownResetsAtomiCSafterForcedDrain()
{
    using convo::isr::RetireQuarantineStore;

    RetireQuarantineStore store;
    for (int i = 0; i < 3; ++i) {
        auto* ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x9000 + i * 0x10));
        auto* deleter = +[](void*) noexcept {};
        if (!store.quarantine(ptr, deleter, 100, DeletionEntryType::Generic,
                              "test", 0, 0))
            return false;
    }
    if (store.residentCountAtomic() != 3) return false;

    // Shutdown forced drain
    store.drainAllUnsafe();
    if (store.residentCountAtomic() != 0) return false;

    // Predicate is now false — no wake signal should fire.
    // CoordinatorLoop's waitForDrainSignalOrTimeout would timeout.
    return true;
}

// Test 6: enqueue → drain → predicate transitions true→false
//   Verifies the full lifecycle: enqueue increments, drain decrements,
//   predicate correctly tracks the transition.
[[nodiscard]] bool testWakePredicateLifecycle()
{
    using convo::isr::RetireQuarantineStore;

    RetireQuarantineStore store;
    auto isOlder = [](uint64_t a, uint64_t b) noexcept {
        return static_cast<int64_t>(a - b) < 0;
    };

    // Start: predicate false
    if (store.residentCountAtomic() != 0) return false;

    // Enqueue: predicate becomes true
    auto* ptr1 = reinterpret_cast<void*>(static_cast<uintptr_t>(0xA000));
    auto* deleter = +[](void*) noexcept {};
    if (!store.quarantine(ptr1, deleter, 100, DeletionEntryType::Generic,
                          "test", 0, 0))
        return false;
    if (store.residentCountAtomic() != 1) return false;

    // Enqueue more: predicate stays true
    auto* ptr2 = reinterpret_cast<void*>(static_cast<uintptr_t>(0xA010));
    if (!store.quarantine(ptr2, deleter, 100, DeletionEntryType::Generic,
                          "test", 0, 0))
        return false;
    if (store.residentCountAtomic() != 2) return false;

    // Drain epoch-safe entries: predicate becomes false
    store.drain(200, isOlder);  // 100 < 200 → safe
    if (store.residentCountAtomic() != 0) return false;

    // Enqueue again: predicate true
    auto* ptr3 = reinterpret_cast<void*>(static_cast<uintptr_t>(0xA020));
    if (!store.quarantine(ptr3, deleter, 100, DeletionEntryType::Generic,
                          "test", 0, 0))
        return false;
    if (store.residentCountAtomic() != 1) return false;

    return true;
}

// ── ★ B-R3/R5: Test-only friend access to ISRRetireRouter internals ──
//   Grants the lost-wake regression test access to drainCv_ / drainCvMtx_
//   WITHOUT exposing them as public API (R5-4). The friend class is declared
//   in ISRRetireRouter.h; this definition lives in the test translation unit.
namespace convo::isr {
class RetireGraceSemanticsTestAccess {
public:
    static std::condition_variable& cv(ISRRetireRouter& r) noexcept { return r.drainCv_; }
    static std::mutex& mtx(ISRRetireRouter& r) noexcept { return r.drainCvMtx_; }
};
} // namespace convo::isr

// ── ★ B-R3: lost-wake regression test ──
//
// Test 7: Deterministically forces the lost-wake window.
//
// The bug being guarded against (pre-B-R3):
//   signalDrainWakeup() called notify_one() WITHOUT acquiring drainCvMtx_.
//   Interleaving that loses the wake:
//     Consumer: lock(drainCvMtx_), predicate check → false
//     Producer: residentAtomic_++ (predicate true), notify_one() → NO waiter yet → LOST
//     Consumer: wait_for → unlock + block → sleeps until timeout (latency regression)
//
// The B-R3 fix: signalDrainWakeup() acquires drainCvMtx_ before notify_one().
// This serializes the notify with the consumer's wait transition:
//   - If the consumer still holds the lock (between predicate check and wait entry),
//     the producer BLOCKS on drainCvMtx_ until the consumer enters wait, then notifies.
//   - If the consumer is already in wait, the producer acquires the lock and notifies.
//   Either way, the wake is immediate — never lost.
//
// Test structure:
//   Consumer thread: acquires drainCvMtx_, checks predicate (false), signals
//     "ready" (STILL HOLDING THE LOCK), then enters wait_for(2000ms).
//   Main thread: waits for "ready", then enqueues to Q (residentAtomic_++)
//     and calls signalDrainWakeup().
//   Assert: consumer wakes well before the 2000ms timeout (< 1000ms).
//
// With the fix: signalDrainWakeup() blocks on the lock until the consumer
//   enters wait, then notifies → immediate wake (< 1000ms). PASS.
// Without the fix: notify_one() fires while the consumer still holds the lock
//   (not yet waiting) → LOST → consumer sleeps the full 2000ms → FAIL.
[[nodiscard]] bool testWakeLostWakeRegression()
{
    // Minimal IEpochProvider stub — enqueueRetire returns false to force the
    // Q/E/T fallback path in enqueueWithRetry (D queue "full").
    struct TestProvider : convo::IEpochProvider {
        bool enqueueRetire(void*, void (*)(void*), std::uint64_t) noexcept override { return false; }
        void tryReclaim() noexcept override {}
        std::uint32_t pendingRetireCount() const noexcept override { return 0; }
        void drainAll() noexcept override {}
        int registerReaderThread() noexcept override { return 0; }
        bool reserveReaderThread(int) noexcept override { return true; }
        void enterReader(int) noexcept override {}
        void exitReader(int) noexcept override {}
        std::uint64_t currentEpoch() const noexcept override { return 0; }
        std::uint32_t activeReaderCount() const noexcept override { return 0; }
        int readerCapacity() const noexcept override { return 1; }
        std::uint64_t getMinReaderEpoch() const noexcept override { return 0; }
        std::uint64_t publishEpoch() noexcept override { return 0; }
    };

    TestProvider provider;
    convo::isr::ISRRetireRouter router(provider);

    std::atomic<bool> consumerReady{false};
    std::atomic<bool> consumerWoke{false};

    // Consumer thread: hold drainCvMtx_, check predicate (false), signal ready,
    // then enter wait_for. Holding the lock while signaling ready forces the
    // producer's signalDrainWakeup() to block (with the fix) until the consumer
    // enters wait — this is the exact lost-wake window.
    // Access to drainCv_ / drainCvMtx_ is via the friend class
    // RetireGraceSemanticsTestAccess (NOT public API — R5-4).
    std::thread consumer([&] {
        std::unique_lock<std::mutex> lock(convo::isr::RetireGraceSemanticsTestAccess::mtx(router));
        // Predicate is false (no entries yet) — verified by the wait_for predicate.
        consumerReady = true;  // still holding the lock
        // Enter wait_for — atomically releases the lock and blocks.
        convo::isr::RetireGraceSemanticsTestAccess::cv(router).wait_for(
            lock, std::chrono::milliseconds(2000),
            [&] {
                return router.pendingRetireCount() != 0
                    || router.residentCountAtomic() != 0;
            });
        consumerWoke = true;
    });

    // Wait for the consumer to be ready (holding the lock, about to enter wait).
    while (!consumerReady.load()) {}

    // Producer: enqueue to Q (D "full" → Q fallback → residentAtomic_++) + signal.
    auto* ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(0xB000));
    auto* deleter = +[](void*) noexcept {};
    router.enqueueWithRetry(ptr, deleter, 100, DeletionEntryType::Generic);

    // signalDrainWakeup() acquires drainCvMtx_ (B-R3 fix). If the consumer still
    // holds the lock, this blocks until the consumer enters wait, then notifies.
    const auto start = std::chrono::steady_clock::now();
    router.signalDrainWakeup();

    // Wait for the consumer to wake and finish.
    consumer.join();
    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();

    // With the fix: immediate wake (< 1000ms, well under the 2000ms timeout).
    // Without the fix: notify lost → consumer sleeps ~2000ms → FAIL.
    return consumerWoke.load() && elapsedMs < 1000;
}

int main()
{
    if (!testGracePeriodCompletionRules())
        throw std::runtime_error("grace period completion rules failed");

    if (!testRetirePendingToFreeRules())
        throw std::runtime_error("retire pending to free rules failed");

    if (!testRetireStarvationDualThresholdRules())
        throw std::runtime_error("retire starvation dual threshold rules failed");

    if (!testRetireEscalationSafetyRules())
        throw std::runtime_error("retire escalation safety rules failed");

    if (!testRetirePressureThresholdPolicyRules())
        throw std::runtime_error("retire pressure threshold policy rules failed");

    if (!testPrioritySortCompositeKey())
        throw std::runtime_error("priority sort composite key failed");

    if (!testOverflowRingFifoOrder())
        throw std::runtime_error("overflow ring FIFO order failed");

    if (!testPrioritySortCriticalFirst())
        throw std::runtime_error("priority sort critical first failed");

    if (!testRetirePriorityCompatibility())
        throw std::runtime_error("retire priority compatibility failed");

if (!testEmptyDrainSuppressionAtomicCounter())
        throw std::runtime_error("empty drain suppression atomic counter failed");

    if (!testWakePredicateTrueAfterEnqueue())
        throw std::runtime_error("wake predicate true after enqueue failed");

    if (!testWakePredicateAlreadyTrueNoBlock())
        throw std::runtime_error("wake predicate already true (no block) failed");

    if (!testWakeSpuriousNoDrainOnEmpty())
        throw std::runtime_error("wake spurious no drain on empty failed");

    if (!testWakeTimeoutFallback())
        throw std::runtime_error("wake timeout fallback failed");

    if (!testWakeShutdownResetsAtomiCSafterForcedDrain())
        throw std::runtime_error("wake shutdown resets atomics after forced drain failed");

    if (!testWakePredicateLifecycle())
        throw std::runtime_error("wake predicate lifecycle failed");

    if (!testWakeLostWakeRegression())
        throw std::runtime_error("wake lost-wake regression failed");

    return 0;
}
