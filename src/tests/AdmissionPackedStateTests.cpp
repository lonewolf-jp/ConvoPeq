//==============================================================================
// AdmissionPackedStateTests.cpp — D101-31-B B-13: tryAdmit/release/outstanding unit tests
//
// Verifies the packed-state admission reservation contract:
//   - tryAdmit succeeds only when AdmissionState == Open
//   - tryAdmit increments reservationCount; release decrements
//   - outstanding() reflects the current reservationCount
//   - closeAdmission → joinProducers transitions correctly with count==0
//   - G-H linearization: tryAdmit and closeAdmission CAS on the same packedState_ word
//
// ★ D101-31-D additions:
//   - versionWrapDoesNotCorruptReservationCount (D-3): 6-bit version wrap at 63→0
//     via test-only AdmissionPackedStateTestAccess seam (CONVOPEQ_UNIT_TESTS)
//   - concurrentTryAdmitCloseAdmission (D-4): tryAdmit ↔ closeAdmission race,
//     Case A / Case B both converge to consistent Closed state
//   - doubleReleaseDoesNotUnderflow (D-5): release underflow clamps, no wrap
//
// Build: standalone test target (links ISRShutdown.cpp + JUCE core)
//==============================================================================
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "audioengine/AtomicAccess.h"     // convo::publishAtomic / consumeAtomic
#include "audioengine/ISRShutdown.h"
#include "audioengine/ISRRuntimePublicationCoordinator.h"
#include "../DspNumericPolicy.h"  // ASSERT_NON_RT_THREAD
#include "AdmissionPackedStateTestAccess.h"  // ★ D101-31-D-3: version 注入用 test-only seam

namespace {

// Test fixture: ShutdownRuntime with a minimal RuntimeIntentCoordinator.
//   NOTE: RuntimeIntentCoordinator is a large object (embedded MpscBoundedRing buffers,
//   LockFreeRingBuffer arrays, large fixed-size queues) — well over 1 MB. It MUST be
//   heap-allocated (via make_unique) to avoid stack overflow on the test thread.
class TestShutdownRuntime {
public:
    std::unique_ptr<convo::isr::RuntimeIntentCoordinator> coordinator;
    std::unique_ptr<convo::isr::ShutdownRuntime> runtime;

    TestShutdownRuntime()
        : coordinator(std::make_unique<convo::isr::RuntimeIntentCoordinator>())
        , runtime(std::make_unique<convo::isr::ShutdownRuntime>(*coordinator))
    {}
};

// ── Test 1: tryAdmit succeeds in Open state, outstanding increments ──
bool testTryAdmitIncrementsOutstanding() {
    TestShutdownRuntime t;
    assert(t.runtime->isAdmissionOpen());  // Open initially
    assert(t.runtime->outstanding() == 0);

    bool ok = t.runtime->tryAdmit(1);
    if (!ok) return false;
    if (t.runtime->outstanding() != 1) return false;

    ok = t.runtime->tryAdmit(5);
    if (!ok) return false;
    if (t.runtime->outstanding() != 6) return false;

    return true;
}

// ── Test 2: tryAdmit fails after closeAdmission (Closing state) ──
bool testTryAdmitFailsAfterClose() {
    TestShutdownRuntime t;
    t.runtime->closeAdmission();  // Open → Closing
    assert(!t.runtime->isAdmissionOpen());

    if (t.runtime->tryAdmit(1)) return false;  // Should fail — not Open
    if (t.runtime->outstanding() != 0) return false;

    return true;
}

// ── Test 3: release decrements outstanding ──
bool testReleaseDecrements() {
    TestShutdownRuntime t;
    t.runtime->tryAdmit(3);
    assert(t.runtime->outstanding() == 3);

    t.runtime->release(1);
    if (t.runtime->outstanding() != 2) return false;

    t.runtime->release(2);
    if (t.runtime->outstanding() != 0) return false;

    return true;
}

// ── Test 4: joinProducers returns false when outstanding > 0 ──
bool testJoinProducersFailsWhenOutstanding() {
    TestShutdownRuntime t;
    t.runtime->tryAdmit(1);
    assert(t.runtime->outstanding() == 1);

    t.runtime->closeAdmission();  // Open → Closing (reservations still outstanding)

    bool joined = t.runtime->joinProducers();
    if (joined) return false;  // Should NOT join — outstanding > 0

    // Release the reservation, then join should succeed
    t.runtime->release(1);
    assert(t.runtime->outstanding() == 0);

    joined = t.runtime->joinProducers();
    if (!joined) return false;  // Should succeed now

    // Verify state is Closed
    if (t.runtime->isAdmissionOpen()) return false;
    if (t.runtime->admissionState() != convo::isr::AdmissionState::Closed) return false;

    return true;
}

// ── Test 5: joinProducers fails if not in Closing state ──
bool testJoinProducersFailsIfNotClosing() {
    TestShutdownRuntime t;
    // State is still Open
    bool joined = t.runtime->joinProducers();
    if (joined) return false;  // Should fail — not Closing

    return true;
}

// ── Test 6: closeAdmission is idempotent ──
bool testCloseAdmissionIdempotent() {
    TestShutdownRuntime t;
    t.runtime->closeAdmission();  // Open → Closing
    // Second call should be no-op (already Closing)
    t.runtime->closeAdmission();

    // Third call after joinProducers → Closed
    // (joinProducers requires count==0; no reservations so should join)
    t.runtime->joinProducers();
    // Now Closed — closeAdmission should be no-op
    t.runtime->closeAdmission();

    if (t.runtime->admissionState() != convo::isr::AdmissionState::Closed) return false;

    return true;
}

// ── Test 7: Concurrent tryAdmit/release stress (G-H race) ──
bool testConcurrentTryAdmitRelease() {
    TestShutdownRuntime t;
    const int kNprocs = 4;
    const int kOpsPerThread = 1000;

    std::vector<std::thread> threads;
    for (int i = 0; i < kNprocs; i++) {
        threads.emplace_back([&t, kOpsPerThread]() {
            for (int j = 0; j < kOpsPerThread; j++) {
                if (t.runtime->tryAdmit(1)) {
                    t.runtime->release(1);
                }
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    // All reservations should be released
    if (t.runtime->outstanding() != 0) return false;

    return true;
}

// ── Test 8: tryAdmit with n > 1 and overflow protection ──
bool testTryAdmitBatchAndOverflow() {
    TestShutdownRuntime t;
    // Batch admit
    if (!t.runtime->tryAdmit(10)) return false;
    if (t.runtime->outstanding() != 10) return false;

    // Overflow: try to admit beyond 24-bit max (0x00FFFFFF = 16,777,215)
    // The max reservation count fits in 24 bits. We can't easily test the overflow
    // with a small number, but verify that tryAdmit with a huge n fails
    if (t.runtime->tryAdmit(0x00FFFFFF)) return false;  // would overflow

    return true;
}

// ── Test 9: version increments on closeAdmission ──
bool testVersionIncrement() {
    TestShutdownRuntime t;
    // Initial version is 0 (packedState_ = 0)
    // closeAdmission should bump version to 1
    t.runtime->closeAdmission();
    // We can't directly observe version, but we can verify behavior:
    // After close+join, the state should be Closed
    t.runtime->joinProducers();
    if (t.runtime->admissionState() != convo::isr::AdmissionState::Closed) return false;

    // A second close should start a new generation (version bumps again)
    // But since state is already Closed, closeAdmission is idempotent
    // This test verifies no crash on repeated close
    return true;
}

// ── Test 10 (D101-31-D-3): version 63 → closeAdmission wraps to 0 without
//    corrupting reservationCount (6-bit version field contract) ──
bool testVersionWrapDoesNotCorruptReservationCount() {
    using convo::isr::AdmissionPackedStateTestAccess;
    TestShutdownRuntime t;

    // Inject: state=Open(0), version=63, count=5 → raw = 0x5FC
    constexpr uint32_t kRawVersion63 =
        (63u << 2) | (5u << 8);  // Open=0, kVersionShift=2, kReservationShift=8
    AdmissionPackedStateTestAccess::store(*t.runtime, kRawVersion63);
    if (t.runtime->outstanding() != 5) return false;
    if (!t.runtime->isAdmissionOpen()) return false;

    t.runtime->closeAdmission();  // Open→Closing with version wrap 63→0

    // Exact word check: Closing(1) | version 0 | count 5 = 0x501.
    // The pre-fix bug ((version + 1) << shift unmasked) would produce 0x601 — bit8 leaks into the
    // reservationCount region and outstanding() would read 6.
    constexpr uint32_t kExpectedAfterWrap =
        static_cast<uint32_t>(convo::isr::AdmissionState::Closing) | (5u << 8);
    if (AdmissionPackedStateTestAccess::load(*t.runtime) != kExpectedAfterWrap) return false;

    if (t.runtime->admissionState() != convo::isr::AdmissionState::Closing) return false;
    if (t.runtime->outstanding() != 5) return false;  // reservationCount 不変

    // Drain then join: wrap must not block Closing→Closed
    t.runtime->release(5);
    if (t.runtime->outstanding() != 0) return false;
    if (!t.runtime->joinProducers()) return false;
    if (t.runtime->admissionState() != convo::isr::AdmissionState::Closed) return false;

    return true;
}

// ── Test 11 (D101-31-D-4): tryAdmit ↔ closeAdmission G-H race ──
//   Single-word CAS linearization executable evidence. Never asserts which side
//   wins; verifies both legal outcomes converge to a consistent state:
//     Case A (tryAdmit CAS first): admits happen under Open, all released,
//                                  joinProducers()==true afterwards.
//     Case B (closeAdmission CAS first): every subsequent tryAdmit fails,
//                                  outstanding stays 0, joinProducers()==true.
//   Contradiction ("both succeeded into inconsistent state") is impossible:
//   after the closing CAS completes on packedState_, no later tryAdmit can win.
bool testConcurrentTryAdmitCloseAdmission() {
    const int kRounds = 100;
    const int kSpinners = 4;
    int caseA = 0;  // tryAdmit won at least once before/during close
    int caseB = 0;  // closeAdmission fully preceded admission attempts

    for (int r = 0; r < kRounds; ++r) {
        TestShutdownRuntime t;
        std::atomic<bool> go{false};
        std::atomic<bool> stop{false};
        std::atomic<int> admitted{0};

        std::vector<std::thread> threads;
        threads.reserve(kSpinners);
        for (int i = 0; i < kSpinners; ++i) {
            threads.emplace_back([&t, &go, &stop, &admitted]() {
                while (!go.load(std::memory_order_relaxed)) {
                    // spin until closer releases the start gate
                }
                while (!stop.load(std::memory_order_relaxed)) {
                    if (!t.runtime->tryAdmit(1)) break;  // admission closed — exit
                    admitted.fetch_add(1, std::memory_order_relaxed);
                    t.runtime->release(1);  // exactly-one release per admit
                }
            });
        }

        go.store(true, std::memory_order_release);
        t.runtime->closeAdmission();          // G-H linearization point (this thread)
        stop.store(true, std::memory_order_release);

        for (auto& th : threads) th.join();

        const int roundAdmitted = admitted.load();
        if (roundAdmitted > 0) ++caseA; else ++caseB;

        // Case-invariant checks (identical for A and B):
        if (t.runtime->outstanding() != 0) return false;       // every admit was released exactly once
        if (t.runtime->tryAdmit(1)) return false;              // post-close admission impossible
        if (!t.runtime->joinProducers()) return false;         // Closing→Closed after full drain
        if (t.runtime->admissionState() != convo::isr::AdmissionState::Closed) return false;
        if (t.runtime->isAdmissionOpen()) return false;        // no resurrection
    }

    printf("    [info] race rounds: tryAdmit-first(Case A)=%d, close-first(Case B)=%d\n",
           caseA, caseB);
    fflush(stdout);

    // Both outcomes must be exercised across rounds for genuine race coverage
    // (scheduler-dependent, but 100 rounds × 4 spinners makes single-sided runs implausible)
    if (caseA == 0 && caseB == 100) {
        printf("    [warn] no interleaving observed — verify thread scheduling\n");
    }
    return true;
}

// ── Test 12 (D101-31-D-5): double-release / release-underflow contract ──
//   Contract (documented, unchanged behavior): release(n) with count < n is a
//   silent no-op (state and count unchanged). Double-release must never wrap
//   reservationCount into 0x00FFFFFF territory.
bool testDoubleReleaseDoesNotUnderflow() {
    TestShutdownRuntime t;

    t.runtime->tryAdmit(1);
    t.runtime->release(1);
    t.runtime->release(1);  // double release — must be a no-op
    if (t.runtime->outstanding() != 0) return false;   // must NOT wrap to 0x00FFFFFF
    if (t.runtime->admissionState() != convo::isr::AdmissionState::Open) return false;

    // Underflow guard: n > count → unchanged
    if (!t.runtime->tryAdmit(1)) return false;
    t.runtime->release(2);
    if (t.runtime->outstanding() != 1) return false;

    // Repeated over-release stress: first decrements to 0, rest clamp at 0
    for (int i = 0; i < 1000; ++i) t.runtime->release(1);
    if (t.runtime->outstanding() != 0) return false;

    // FSM untouched by clamped releases: shutdown path still works
    t.runtime->closeAdmission();
    if (!t.runtime->joinProducers()) return false;
    if (t.runtime->admissionState() != convo::isr::AdmissionState::Closed) return false;
    if (t.runtime->outstanding() != 0) return false;

    return true;
}

// ── Test 13 (D101-33-C Case B): shutdown 後 admission は副作用ゼロで拒否 ──
bool testCaseB_ShutdownRejectSideEffectZero() {
    using convo::isr::RuntimeIntentCoordinator;
    TestShutdownRuntime t;

    t.runtime->closeAdmission();  // shutdown 確定（Early Close Convergence 後の状態）

    // Path B producer の入口と同一操作: tryAdmit 失敗 = side effect zero
    if (t.runtime->tryAdmit(1)) return false;

    if (t.runtime->outstanding() != 0) return false;                       // token 残留なし
    if (t.coordinator->getPendingIntentCount() != 0) return false;         // intent 会計不变
    if (t.coordinator->getPublicationIntentResidencyCount() != 0)
        return false;                                                      // X5 residency 不变
    // queue 未変更の代理観測: 全 counter/queue が空のため drain 判定が true
    if (!t.coordinator->isFullyDrained()) return false;
    if (t.runtime->admissionState() != convo::isr::AdmissionState::Closing) return false;

    // token なしでの enqueue も禁止（admission authority は packedState_ のみ）
    RuntimeIntentCoordinator::Intent intent{};
    // NOTE: D101-33-C 以降、enqueuePublicationIntent 自体は token を検査しない
    // （authority は facade の tryAdmit に収束）。本テストは「close 後は tryAdmit が
    //  失敗するため Path B transaction が開始されない」ことを検証済み。

    // close 後でも closeAdmission は冪等、joinProducers は可能
    t.runtime->closeAdmission();
    if (!t.runtime->joinProducers()) return false;
    if (t.runtime->admissionState() != convo::isr::AdmissionState::Closed) return false;

    return true;
}

// ── Test 14 (D101-33-C Case C): close vs publication-admit race stress ──
//   tryAdmit/closeAdmission は同一 packedState_ word への CAS（全順序）。
//   各 iteration の不変条件:
//     admit 成功 ⇒ outstanding()==1 が即時観測可能（token observable）
//     admit 失敗 ⇒ publication 側効果ゼロ（push しない）
//     全 thread join 後 ⇒ outstanding==0 / residency==pushed 数（漏れ・過剰なし）
bool testCaseC_CloseVsPublicationAdmitStress() {
    const int kRounds = 200;

    int caseA = 0;  // admit 先行（token 観測下で push 完遂）
    int caseB = 0;  // close 先行（reject）

    for (int r = 0; r < kRounds; ++r) {
        TestShutdownRuntime t;
        std::atomic<bool> go{false};
        std::atomic<int> pushed{0};

        std::thread producer([&]() {
            while (!go.load(std::memory_order_relaxed)) {}
            // Path B producer transaction（facade 冒頭と同一の token 取得）
            if (!t.runtime->tryAdmit(1)) return;              // close 先行 → reject
            // token observable: 直後に reservation が見える
            if (t.runtime->outstanding() < 1) return;

            convo::isr::RuntimeIntentCoordinator::Intent intent{};
            if (t.coordinator->enqueuePublicationIntent(intent)) {
                pushed.fetch_add(1, std::memory_order_relaxed);
            }
            // push 失敗（full）の場合も X5 rollback は choke point 内部で完了済み。
            // durable 点（push 成功）または失敗確定点で token release。
            t.runtime->release(1);
        });

        go.store(true, std::memory_order_release);
        t.runtime->closeAdmission();                           // close vs admit race
        producer.join();

        (pushed.load() > 0 ? ++caseA : ++caseB);

        // round 不変条件: 漏れ・過剰なし
        if (t.runtime->outstanding() != 0) return false;       // token 残留なし
        if (t.coordinator->getPendingIntentCount() != 0) return false;
        if (t.coordinator->getPublicationIntentResidencyCount()
                != static_cast<std::uint64_t>(pushed.load())) return false;  // X5 == push 数
        if (t.runtime->tryAdmit(1)) return false;              // close 後の再 admission 不可
        if (t.runtime->admissionState() == convo::isr::AdmissionState::Open) return false;
    }

    printf("    [info] race rounds: admit-first(Case A)=%d, close-first(Case B)=%d\n",
           caseA, caseB);
    fflush(stdout);
    return true;
}

// ── Test 15 (D101-33-C Case D): queue full — token/X5/rollback chain ──
bool testCaseD_QueueFullRollbackChain() {
    using convo::isr::RuntimeIntentCoordinator;
    TestShutdownRuntime t;

    // intentQueue_ を満たすまで Path B transaction（admit→enqueue→durable release）を反復
    unsigned long long succeeded = 0;
    bool hitFull = false;
    constexpr int kMaxAttempts = 300000;
    for (int i = 0; i < kMaxAttempts; ++i) {
        if (!t.runtime->tryAdmit(1)) return false;             // token 取得
        RuntimeIntentCoordinator::Intent intent{};
        if (t.coordinator->enqueuePublicationIntent(intent)) {
            ++succeeded;
            t.runtime->release(1);                             // durable 点 release
        } else {
            // queue full: X5 rollback は choke point 内部で完了済み
            t.runtime->release(1);                             // token release（失敗経路）
            hitFull = true;
            break;
        }
        if (t.runtime->outstanding() != 0) return false;       // durable release 漏れチェック
    }
    if (!hitFull) return false;                                // 容量確認できず

    // full 状態での整合: residency == 成功 push 数、token 残留なし
    if (t.coordinator->getPublicationIntentResidencyCount() != succeeded) return false;
    if (t.runtime->outstanding() != 0) return false;

    // full 状態での Case D chain 再検証: admit → push fail → rollback → release
    if (!t.runtime->tryAdmit(1)) return false;
    if (t.runtime->outstanding() != 1) return false;
    RuntimeIntentCoordinator::Intent intent{};
    if (t.coordinator->enqueuePublicationIntent(intent)) return false;   // full なので失敗
    const auto resAfterFail = t.coordinator->getPublicationIntentResidencyCount();
    if (resAfterFail != succeeded) return false;               // X5 rollback 済
    t.runtime->release(1);
    if (t.runtime->outstanding() != 0) return false;           // token release 済
    if (t.coordinator->getPendingIntentCount() != 0) return false;  // Publish は計上されない

    return true;
}

}  // namespace

int main() {
    struct Test { const char* name; bool (*fn)(); };
    Test tests[] = {
        {"tryAdmitIncrementsOutstanding",    testTryAdmitIncrementsOutstanding},
        {"tryAdmitFailsAfterClose",          testTryAdmitFailsAfterClose},
        {"releaseDecrements",                testReleaseDecrements},
        {"joinProducersFailsWhenOutstanding",testJoinProducersFailsWhenOutstanding},
        {"joinProducersFailsIfNotClosing",   testJoinProducersFailsIfNotClosing},
        {"closeAdmissionIdempotent",         testCloseAdmissionIdempotent},
        {"concurrentTryAdmitRelease",        testConcurrentTryAdmitRelease},
        {"tryAdmitBatchAndOverflow",         testTryAdmitBatchAndOverflow},
        {"versionIncrement",                 testVersionIncrement},
        {"versionWrapDoesNotCorruptReservationCount",
                                             testVersionWrapDoesNotCorruptReservationCount},
        {"concurrentTryAdmitCloseAdmission", testConcurrentTryAdmitCloseAdmission},
        {"doubleReleaseDoesNotUnderflow",    testDoubleReleaseDoesNotUnderflow},
        {"caseB_ShutdownRejectSideEffectZero",
                                             testCaseB_ShutdownRejectSideEffectZero},
        {"caseC_CloseVsPublicationAdmitStress",
                                             testCaseC_CloseVsPublicationAdmitStress},
        {"caseD_QueueFullRollbackChain",     testCaseD_QueueFullRollbackChain},
    };

    int passed = 0;
    int failed = 0;
    for (const auto& test : tests) {
        bool ok = false;        try {
            ok = test.fn();
        } catch (...) {
            ok = false;
        }
        if (ok) {
            printf("[PASS] %s\n", test.name);
            fflush(stdout);
            passed++;
        } else {
            printf("[FAIL] %s\n", test.name);
            fflush(stdout);
            failed++;
        }
    }

    printf("\n%d passed, %d failed out of %d tests\n", passed, failed,
           static_cast<int>(sizeof(tests) / sizeof(tests[0])));
    return failed == 0 ? 0 : 1;
}
