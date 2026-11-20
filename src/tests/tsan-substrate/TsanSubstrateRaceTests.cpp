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
#include "core/RCUReader.h"
#include "audioengine/RetireQuarantineStore.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <functional>
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

// ★ bounded wait: 条件が成立するまで待つ（上限付き）。タイムアウトは false。
//   固定 sleep だけで「競合区間に到達した」ことを仮定しないための補助。
bool waitUntil (const std::function<bool()>& pred, std::chrono::milliseconds timeout)
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline)
    {
        if (pred())
            return true;
        std::this_thread::sleep_for (std::chrono::microseconds (100));
    }
    return pred();
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

    // publisher が実際に実行されていることを、固定 sleep ではなく epoch 前進で確認する。
    const std::uint64_t epochBeforeClose = domain->currentEpoch();
    check (waitUntil ([&] {
               return domain->currentEpoch() != epochBeforeClose;
           }, std::chrono::milliseconds (2000)),
           "T1: publishers must be running before the gate closes");

    const bool idle = domain->closePublishersAndWaitIdle (std::chrono::milliseconds (2000));
    const std::uint64_t epochAtFence = domain->epochAtFence();

    // gate 閉鎖後の遅延 publish が計数されるまで待つ（固定 sleep 不使用）。
    check (waitUntil ([&] {
               return domain->latePublisherAttempts() > 0;
           }, std::chrono::milliseconds (2000)),
           "T1: late publisher attempts must be counted after the gate closed");

    stop.store (true, std::memory_order_release);
    for (auto& t : pubs)
        t.join();

    check (idle, "T1: publishers must become idle");
    check (domain->currentEpoch() == epochAtFence,
           "T1: epoch must not advance after the gate closed");
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

    // fence を閉じる前に、reader が実際に入場したことを待つ。
    check (waitUntil ([&] { return entered.load (std::memory_order_acquire) > 0; },
                      std::chrono::milliseconds (2000)),
           "T2: readers must have entered before the fence closes");

    domain->closeReaderRegistration();
    domain->engageShutdownReaderFence();

    // fence 閉鎖後の入場試行が拒否されたことを待つ。
    check (waitUntil ([&] { return domain->enterAfterFence() > 0; },
                      std::chrono::milliseconds (2000)),
           "T2: enter-after-fence attempts must be observed");

    const bool admissionsIdle = domain->waitForReaderAdmissions (std::chrono::milliseconds (2000));
    const bool publishersIdle = domain->closePublishersAndWaitIdle (std::chrono::milliseconds (2000));
    stop.store (true, std::memory_order_release);
    for (auto& t : readers)
        t.join();
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

    // producer が少なくとも 1 件を受け入れたことを確認してから引き取る（固定 sleep 不使用）。
    check (waitUntil ([&] { return accepted.load (std::memory_order_acquire) > 0; },
                      std::chrono::milliseconds (2000)),
           "T4: producer must enqueue at least once before the take");
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

// ── T7: reader exit と shutdown 側観測の競合 ─────────────────────
//   reader が enter/exit を繰り返す間に、shutdown 側の読み取り（activeReaderCount /
//   activeDepthReaderCount / admission）と epoch 前進を同時に回す。
//   exitReader の provider アクセスと静止判定の読み取りが競合しないことを
//   TSan で見る。終了後に活動数が 0 に戻ることも確かめる。
void t7_exitVsShutdownObservation()
{
    auto domain = std::make_unique<convo::EpochDomain> (convo::RetireDomainId::EngineISR);

    // ★ 前処理: ReaderMeasureRegistry の同名タグの枠を単スレッドで作っておく。
    //   既存の計測レジストリは、タグを書く前に used を公開するため、
    //   同名タグの初回 claim を複数スレッドで競合させると strncmp 嵌めで
    //   data race になる（別 Issue で管理。ここでは T7 が目的とする競合だけを実行する）。
    {
        // 記録を閉じたこと（タグの登録ができたこと）を closed_ で確かめる。
        // claim に失敗すると close されず、同名タグの初回 claim 競合が残る。
        const std::uint64_t closedBefore =
            convo::consumeAtomic (convo::ReaderMeasureRegistry::closed_, std::memory_order_acquire);
        convo::RCUReader warmup (*domain);
        if (warmup.enter())
            warmup.exit();
        const std::uint64_t closedAfter =
            convo::consumeAtomic (convo::ReaderMeasureRegistry::closed_, std::memory_order_acquire);
        const bool warmupClosed = (closedAfter == closedBefore + 1);
        check (warmupClosed, "T7: warmup must close a measure record (tag registration)");
        if (! warmupClosed)
            return;   // タグが未登録のまま並行 claim を始めない（別 Issue の race を混ぜない）
    }

    std::atomic<bool> stop { false };
    std::atomic<int> exitsDone { 0 };
    std::atomic<int> observations { 0 };
    std::atomic<bool> sawActivity { false };
    std::atomic<int> exitWindowSamples { 0 };    // depth < activity（exit 実行中）を見た回数

#if defined(CONVOPEQ_ENABLE_DRAIN_TEST_FACTORY)
    // ★ 決定論的な exit 窓: 1 回だけ、depth デクリメント後の reader を止める。
    struct ExitSeamCtx
    {
        std::atomic<bool> fired { false };
        std::atomic<bool> reached { false };
        std::atomic<bool> release { false };
    };
    static ExitSeamCtx seam;
    seam.fired.store (false);
    seam.reached.store (false);
    seam.release.store (false);
    convo::EpochDomain::setReaderExitSeam ([] (void* p) noexcept
    {
        auto* s = static_cast<ExitSeamCtx*> (p);
        if (s == nullptr)
            return;
        bool expected = false;
        if (! s->fired.compare_exchange_strong (expected, true))
            return;
        s->reached.store (true, std::memory_order_release);
        while (! s->release.load (std::memory_order_acquire))
            std::this_thread::yield();
    }, &seam);
#endif

    std::vector<std::thread> readers;
    for (int t = 0; t < 3; ++t)
        readers.emplace_back ([&]
        {
            convo::RCUReader reader (*domain);
            while (! stop.load (std::memory_order_acquire))
            {
                if (! reader.enter())
                {
                    std::this_thread::yield();
                    continue;
                }
                reader.exit();
                (void) exitsDone.fetch_add (1, std::memory_order_relaxed);
            }
        });

    std::thread shutdownSide ([&]
    {
        while (! stop.load (std::memory_order_acquire))
        {
            (void) domain->publishEpoch();
            const std::uint32_t observedActivity = domain->activeReaderCount();
            const std::uint32_t observedDepth = domain->activeDepthReaderCount();
            if (observedActivity > 0)
                sawActivity.store (true, std::memory_order_release);
            // ★ 活動しているのに depth が少ない = exit 実行中を抽出できた。
            if (observedActivity > observedDepth)
                exitWindowSamples.fetch_add (1, std::memory_order_release);
            (void) domain->readerAdmissionsActive();
            (void) domain->admissionPendingSlots();
            (void) observations.fetch_add (1, std::memory_order_release);
        }
    });

    // ★ observer が reader の稼働中に実際の観測を完了したことを確認してから停める。
    //   これが無いと、observer が一度も実行されなくてもカウンタが 0 に戻るだけで PASS してしまう。
    const bool observedWhileRunning = waitUntil ([&]
    {
        return observations.load (std::memory_order_acquire) > 0
            && exitsDone.load (std::memory_order_acquire) > 0
            && sawActivity.load (std::memory_order_acquire);
    }, std::chrono::milliseconds (2000));
    check (observedWhileRunning, "T7: observer must complete an iteration while readers run");

    // ★ exit 実行中のサンプルは、ストレスループの偶然に頼らず、
    //   seam で 1 人を depth デクリメント後に止めて確定的に取る。
    //   止まっている reader は depth 0・活動 1 なので、observer は必ず activity > depth を見る。
    const int windowsBefore = exitWindowSamples.load (std::memory_order_acquire);
    const bool seamReached = waitUntil ([&]
    {
        return seam.reached.load (std::memory_order_acquire);
    }, std::chrono::milliseconds (5000));
    check (seamReached, "T7: the exit seam must be reached");
    if (seamReached)
    {
        const bool sampledWhilePaused = waitUntil ([&]
        {
            return exitWindowSamples.load (std::memory_order_acquire) > windowsBefore;
        }, std::chrono::milliseconds (2000));
        check (sampledWhilePaused,
               "T7: the observer must sample the paused exit (activity > depth)");
    }
    seam.release.store (true, std::memory_order_release);
    check (sawActivity.load (std::memory_order_acquire),
           "T7: the observer must see non-zero reader activity");

    stop.store (true, std::memory_order_release);

    for (auto& r : readers)
        r.join();
    shutdownSide.join();

    check (exitsDone.load() > 0, "T7: readers must complete exits");
    check (observations.load() > 0, "T7: observer iterations must be recorded");
#if defined(CONVOPEQ_ENABLE_DRAIN_TEST_FACTORY)
    convo::EpochDomain::setReaderExitSeam (nullptr, nullptr);
#endif
    check (domain->activeReaderCount() == 0, "T7: activity must return to 0");
    check (domain->activeDepthReaderCount() == 0, "T7: depth must return to 0");
    std::printf ("T7 exit vs shutdown: exits=%d observations=%d activity=%u depth=%u\n",
                 exitsDone.load(), observations.load(),
                 static_cast<unsigned> (domain->activeReaderCount()),
                 static_cast<unsigned> (domain->activeDepthReaderCount()));
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
    t7_exitVsShutdownObservation();

    std::printf ("== failures=%d deleted=%d ==\n", g_failures, g_del.load());
    return g_failures == 0 ? 0 : 1;
}
