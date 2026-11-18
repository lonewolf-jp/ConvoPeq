// ============================================================================
// WORK113 Queue/Gate/Drain — substrate-level ThreadSanitizer harness.
//
// 目的: Windows / MSVC では TSan を実行できないため、MKL/IPP/JUCE 非依存の
//       shutdown 基盤（EpochDomain / DeferredDeletionQueue / RetireQuarantineStore）を
//       Linux clang + TSan で動かし、データ競合の有無を実測する。
//
// 対象は「実際のヘッダ」（コピーではない）。テスト用シームは
//   CONVOPEQ_ENABLE_DRAIN_TEST_FACTORY （DrainCapability::makeForTest）のみ。
//
// ビルド: clang++ -std=c++20 -fsanitize=thread -O1 -g -pthread \
//           -DCONVOPEQ_ENABLE_DRAIN_TEST_FACTORY \
//           -I<repo>/src tsan_substrate.cpp -o tsan_substrate
//
// 陽性対照（TSan が競合を検出できることを確認するビルド）:
//   同じコマンドに -DTSAN_POSITIVE_CONTROL を追加する。
// ============================================================================
#include "core/EpochDomain.h"
#include "audioengine/RetireQuarantineStore.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <thread>
#include <vector>

namespace {

std::atomic<int> g_del { 0 };

void noopDeleter (void*) noexcept { g_del.fetch_add (1, std::memory_order_relaxed); }
void countedDeleter (void*) noexcept { g_del.fetch_add (1, std::memory_order_relaxed); }

int g_failures = 0;

void check (bool cond, const char* what)
{
    if (! cond)
    {
        std::printf ("FAIL: %s\n", what);
        ++g_failures;
    }
}

// ── T1: publish gate と publishEpoch の競合 ─────────────────────────────────
//   4 本の publisher が publishEpoch() を回し続ける中で gate を閉じる。
//   gate 閉鎖後は epoch が進まないこと、遅延 publish が計数されることを見る。
void t1_publishGate()
{
    auto domain = std::make_unique<convo::EpochDomain> (convo::RetireDomainId::EngineISR);

    std::atomic<bool> stop { false };
    std::vector<std::thread> pubs;
    for (int t = 0; t < 4; ++t)
        pubs.emplace_back ([&]
        {
            while (! stop.load (std::memory_order_acquire))
                (void) domain->publishEpoch();
        });

    std::this_thread::sleep_for (std::chrono::milliseconds (30));
    const bool idle = domain->closePublishersAndWaitIdle (std::chrono::milliseconds (2000));
    const std::uint64_t epochAtFence = domain->epochAtFence();

    // gate 閉鎖後も publisher は回り続ける（= 遅延 publish の計数対象）。
    std::this_thread::sleep_for (std::chrono::milliseconds (20));
    stop.store (true, std::memory_order_release);
    for (auto& t : pubs)
        t.join();

    check (idle, "T1: publishers must become idle");
    check (domain->currentEpoch() == epochAtFence,
           "T1: epoch must not advance after the gate closed");
    check (domain->latePublisherAttempts() > 0,
           "T1: late publisher attempts must be counted");
    std::printf ("T1 publish gate: attempts=%u epoch=%llu\n",
                 static_cast<unsigned> (domain->latePublisherAttempts()),
                 static_cast<unsigned long long> (domain->currentEpoch()));
}

// ── T2: reader admission / fence の競合 ─────────────────────────────────────
//   reader が register → enter → exit を回す中で fence を閉じる。
void t2_readerFence()
{
    auto domain = std::make_unique<convo::EpochDomain> (convo::RetireDomainId::EngineISR);

    std::atomic<bool> stop { false };
    std::atomic<int> entered { 0 };
    std::vector<std::thread> readers;
    for (int t = 0; t < 4; ++t)
        readers.emplace_back ([&]
        {
            while (! stop.load (std::memory_order_acquire))
            {
                const int slot = domain->registerReaderThread ("tsan-reader");
                if (slot < 0)
                {
                    std::this_thread::yield();
                    continue;
                }
                domain->enterReader (slot);
                entered.fetch_add (1, std::memory_order_relaxed);
                domain->exitReader (slot);
            }
        });

    std::this_thread::sleep_for (std::chrono::milliseconds (30));
    domain->closeReaderRegistration();
    domain->engageShutdownReaderFence();
    const bool admissionsIdle = domain->waitForReaderAdmissions (std::chrono::milliseconds (2000));
    const bool publishersIdle = domain->closePublishersAndWaitIdle (std::chrono::milliseconds (2000));
    std::this_thread::sleep_for (std::chrono::milliseconds (20));
    stop.store (true, std::memory_order_release);
    for (auto& t : readers)
        t.join();

    check (entered.load() > 0, "T2: readers must have entered at least once");
    check (admissionsIdle, "T2: reader admissions must become idle");
    check (publishersIdle, "T2: publishers must become idle");
    check (domain->activeReaderCount() == 0, "T2: no active reader after the fence");
    check (domain->admissionPendingSlots() == 0, "T2: no pending admission slot");
    check (domain->enterAfterFence() > 0, "T2: enter-after-fence attempts must be counted");
    std::printf ("T2 reader fence: entered=%d enterAfterFence=%u reacquireAfterFence=%u\n",
                 entered.load(),
                 static_cast<unsigned> (domain->enterAfterFence()),
                 static_cast<unsigned> (domain->reacquireAfterFence()));
}

// ── T3: producer / consumer（enqueue と tryReclaim の同時実行）──────────────
void t3_queueProducerConsumer()
{
    auto domain = std::make_unique<convo::EpochDomain> (convo::RetireDomainId::EngineISR);

    constexpr int kObjects = 4096;
    auto objects = std::make_unique<int[]> (kObjects);
    std::atomic<int> enqueued { 0 };
    std::atomic<bool> producerDone { false };
    const int deletedBefore = g_del.load();

    std::thread producer ([&]
    {
        for (int i = 0; i < kObjects; ++i)
        {
            // epoch を 1 つ手前（安全側）にして reclaim の対象にする。
            const std::uint64_t epoch = domain->currentEpoch();
            if (domain->enqueueRetire (&objects[static_cast<std::size_t> (i)], countedDeleter, epoch))
                enqueued.fetch_add (1, std::memory_order_relaxed);
            if ((i & 31) == 0)
                domain->publishEpoch();
        }
        producerDone.store (true, std::memory_order_release);
    });

    std::thread consumer ([&]
    {
        while (! producerDone.load (std::memory_order_acquire))
        {
            (void) domain->tryReclaim();
            std::this_thread::yield();
        }
        (void) domain->tryReclaim();
    });

    producer.join();
    consumer.join();

    check (enqueued.load() == kObjects, "T3: every enqueue must succeed");
    // 保存則: 生存する entry と reclaim された entry の合計は enqueue 受理件数に等しい
    //   （消失も重複も許さない）。reclaim は g_del を増やすので、差分で数える。
    const std::uint32_t pendingT3 = domain->pendingRetireCount();
    const std::uint32_t reclaimedT3 =
        static_cast<std::uint32_t> (g_del.load() - deletedBefore);
    check (reclaimedT3 + pendingT3 == static_cast<std::uint32_t> (kObjects),
           "T3: enqueued == reclaimed + pending (conservation)");
    std::printf ("T3 queue: enqueued=%d reclaimed=%u pending=%u (sum=%u)\n",
                 enqueued.load(), reclaimedT3, pendingT3,
                 reclaimedT3 + pendingT3);
}

// ── T4: takeForShutdown と producer の競合（queue 寿命の受け渡し）───────────
void t4_takeDuringEnqueue()
{
    auto domain = std::make_unique<convo::EpochDomain> (convo::RetireDomainId::EngineISR);

    constexpr int kObjects = 400000;
    auto objects = std::make_unique<int[]> (kObjects);
    std::atomic<bool> stop { false };
    std::atomic<int> accepted { 0 };

    std::thread producer ([&]
    {
        for (int i = 0; i < kObjects && ! stop.load (std::memory_order_acquire); ++i)
        {
            if (domain->enqueueRetire (&objects[static_cast<std::size_t> (i)], noopDeleter,
                                       domain->currentEpoch()))
                accepted.fetch_add (1, std::memory_order_relaxed);
        }
    });

    // producer が回り続けている途中で引き取る（行数を十分に多くしてある）。
    std::this_thread::sleep_for (std::chrono::milliseconds (1));
    auto taken = domain->takeForShutdown();   // ← producer と同時
    stop.store (true, std::memory_order_release);
    producer.join();

    // 取り出し後は enqueue が成立しない（queue の所有は移っている）。
    static int lateObject = 0;
    const bool lateAccepted = domain->enqueueRetire (&lateObject, noopDeleter, domain->currentEpoch());
    check (taken != nullptr, "T4: takeForShutdown must hand over the queue");
    check (! lateAccepted, "T4: enqueue after take must be refused");
    // 保存則: producer が受理された件数と、引き取られた queue に入っていた件数が一致する
    //   （受理の漂白・重複なしの直接確認）。
    const std::uint32_t inTakenQueue = taken != nullptr ? taken->sizeApprox() : 0u;
    check (accepted.load() == static_cast<int> (inTakenQueue),
           "T4: accepted entries must all be inside the taken queue");
    std::printf ("T4 take: accepted=%d takenQueue=%u\n",
                 accepted.load(), static_cast<unsigned> (inTakenQueue));
}

// ── T5: checked drain（capability 付き）と reader の競合 ────────────────────
void t5_checkedDrainWithReaders()
{
    auto domain = std::make_unique<convo::EpochDomain> (convo::RetireDomainId::EngineISR);

    constexpr int kObjects = 512;
    auto objects = std::make_unique<int[]> (kObjects);
    for (int i = 0; i < kObjects; ++i)
        (void) domain->enqueueRetire (&objects[static_cast<std::size_t> (i)], countedDeleter,
                                      domain->currentEpoch());

    domain->closeReaderRegistration();
    domain->engageShutdownReaderFence();
    const bool admissionsIdle = domain->waitForReaderAdmissions (std::chrono::milliseconds (2000));
    const bool publishersIdle = domain->closePublishersAndWaitIdle (std::chrono::milliseconds (2000));

    std::atomic<bool> stop { false };
    std::atomic<int> refused { 0 };
    std::atomic<int> attempts { 0 };
    std::thread lateReader ([&]
    {
        while (! stop.load (std::memory_order_acquire))
        {
            attempts.fetch_add (1, std::memory_order_relaxed);
            if (domain->registerReaderThread ("late") < 0)
                refused.fetch_add (1, std::memory_order_relaxed);
            std::this_thread::yield();
        }
    });

    // ★ 決定性: TSan 無効のビルドでは main が速く、reader が 1 回も試行しないうちに
    //   stop されることがある（refused == 0 で誤判定になる）。最低 1 回試行するまで待つ。
    for (int spin = 0; spin < 20000 && attempts.load (std::memory_order_acquire) == 0; ++spin)
        std::this_thread::sleep_for (std::chrono::microseconds (100));

    auto queue = domain->borrow();
    bool completed = false;
    if (queue != nullptr)
    {
        auto cap = convo::DrainCapability::makeForTest (true, true, true, true,
                                                       domain->epochAtFence(), 1, 1);
        completed = (domain->drainQueue (*queue, std::move (cap))
                     == convo::EpochDomain::DrainResult::Completed);
    }

    stop.store (true, std::memory_order_release);
    lateReader.join();

    check (admissionsIdle, "T5: admissions must be idle before the drain");
    check (publishersIdle, "T5: publishers must be idle before the drain");
    check (queue != nullptr, "T5: queue must be borrowable");
    check (completed, "T5: checked drain must complete");
    check (attempts.load() > 0, "T5: the late reader must have attempted registration");
    check (refused.load() > 0, "T5: late reader registration must be refused");
    check (domain->drainRemainingAfter() == 0, "T5: nothing must remain in the queue");
    std::printf ("T5 drain: completed=%d remaining=%llu refused=%d deleted=%d\n",
                 completed ? 1 : 0,
                 static_cast<unsigned long long> (domain->drainRemainingAfter()),
                 refused.load(), g_del.load());
}

// ── T6: RetireQuarantineStore の store / drain 同時実行 ─────────────────────
void t6_quarantineStoreConcurrency()
{
    convo::isr::RetireQuarantineStore store;
    constexpr int kObjects = 1024;
    auto objects = std::make_unique<int[]> (kObjects);
    std::atomic<bool> stop { false };

    std::thread storer ([&]
    {
        for (int i = 0; i < kObjects; ++i)
            (void) store.quarantine (&objects[static_cast<std::size_t> (i)], noopDeleter,
                                     static_cast<std::uint64_t> (i), DeletionEntryType::Generic,
                                     "tsan", 0, 0);
        stop.store (true, std::memory_order_release);
    });

    std::thread drainer ([&]
    {
        while (! stop.load (std::memory_order_acquire))
        {
            // すべての epoch を安全とみなして drain する（deleter の同時実行を誘発）。
            store.drain (std::uint64_t{1} << 40,
                         [] (std::uint64_t a, std::uint64_t b) { return a < b; },
                         convo::RetireDomainId::EngineISR);
            std::this_thread::yield();
        }
    });

    storer.join();
    drainer.join();
    store.drain (std::uint64_t{1} << 40,
                 [] (std::uint64_t a, std::uint64_t b) { return a < b; },
                 convo::RetireDomainId::EngineISR);

    check (store.residentCount() == 0, "T6: quarantine store must be empty at the end");
    std::printf ("T6 quarantine: resident=%zu overflow=%u\n",
                 store.residentCount(), static_cast<unsigned> (store.overflowCount()));
}

} // namespace

#if defined(TSAN_POSITIVE_CONTROL)
// 陽性対照: TSan が実際に競合を検出できることを確かめる。
// このビルドだけ -DTSAN_POSITIVE_CONTROL を付け、実行すると
// "WARNING: ThreadSanitizer: data race" が出ることを確認する。
void positiveControl()
{
    static int sharedCounter = 0;   // 意図的に非 atomic
    std::thread a ([] { for (int i = 0; i < 100000; ++i) ++sharedCounter; });
    std::thread b ([] { for (int i = 0; i < 100000; ++i) ++sharedCounter; });
    a.join();
    b.join();
    std::printf ("positive control counter=%d\n", sharedCounter);
}
#endif

int main()
{
    std::printf ("== WORK113 substrate TSan harness ==\n");
#if defined(TSAN_POSITIVE_CONTROL)
    positiveControl();
#endif
    t1_publishGate();
    t2_readerFence();
    t3_queueProducerConsumer();
    t4_takeDuringEnqueue();
    t5_checkedDrainWithReaders();
    t6_quarantineStoreConcurrency();

    std::printf ("== failures=%d deleted=%d ==\n", g_failures, g_del.load());
    return g_failures == 0 ? 0 : 1;
}
