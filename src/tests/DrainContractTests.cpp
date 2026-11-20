//==============================================================================
// DrainContractTests.cpp — WORK113 Queue/Gate/Drain の契約テスト（単一 TU・テスト専用）
//
//   - publish gate の期限判定（active 中の waitForIdle は false、close 後の入場拒否）
//   - reader fence / reader admission（register / reserve / reacquire の拒否と計数）
//   - reader admission と shutdown の競合（fence 閉鎖中に admission 中の reader がいる）
//   - drainQueue の判定（reader 非静止・capability 不正・epoch 不一致・完了）
//   - capability の single-use
//   - 隔離の一意化と自己参照保持（非空 queue・deleter は実行しない）
//   - checked drain の deleter exactly-once（非空 queue・失敗時は保持）
//
//   本ファイルは CONVOPEQ_ENABLE_DRAIN_TEST_FACTORY=1 でビルドされ、
//   DrainCapability::makeForTest() だけが本番ヘッダに現れる（本番ビルドには存在しない）。
//   makeForTest は EpochDomain の下位契約（drainQueue）の検証にだけ使う。
//   ShutdownRuntime の発行経路は invariant_INV3_INV5Tests が実 API で検証する。
//==============================================================================

#include "core/EpochDomain.h"
#include "core/DrainCapability.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <thread>

namespace
{

using namespace std::chrono_literals;

int g_failures = 0;
std::atomic<int> g_deleted { 0 };

void countingDeleter (void*) { g_deleted.fetch_add (1, std::memory_order_relaxed); }

void check (bool ok, const char* what)
{
    if (! ok)
    {
        std::fprintf (stderr, "[drain-test] FAIL: %s\n", what);
        ++g_failures;
    }
}

void waitUntil (const std::atomic<bool>& flag)
{
    while (! flag.load (std::memory_order_acquire))
        std::this_thread::yield();
}

// ★ 上限付きの待機。タイムアウトは false。
//   チェックポイントに到達しない回帰でテストがハングしないように使う。
bool waitUntilBounded (const std::atomic<bool>& flag, std::chrono::milliseconds timeout)
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline)
    {
        if (flag.load (std::memory_order_acquire))
            return true;
        std::this_thread::sleep_for (std::chrono::microseconds (100));
    }
    return flag.load (std::memory_order_acquire);
}

void testPublishGate()
{
    convo::EpochDomain::EpochPublishGate gate;

    check (gate.tryEnter(), "gate: tryEnter while open");
    gate.close();

    // active publisher が残っている間は、期限内の idle を観測できない。
    check (! gate.waitForIdle (std::chrono::milliseconds (50)),
           "gate: waitForIdle must time out while a publisher is active");

    // close 後の新規入場は拒否される。
    check (! gate.tryEnter(), "gate: tryEnter after close must fail");

    gate.exit();
    check (gate.waitForIdle (std::chrono::milliseconds (50)),
           "gate: waitForIdle must succeed after exit");
}

void testReaderFence (convo::EpochDomain& domain)
{
    domain.engageShutdownReaderFence();

    check (domain.registerReaderThread ("drain-test") == -1,
           "fence: registerReaderThread must be refused");
    check (! domain.reserveReaderThread (0),
           "fence: reserveReaderThread must be refused");
    check (! domain.reacquireReaderSlot (0),
           "fence: reacquireReaderSlot must be refused");
    check (domain.enterAfterFence() >= 2u, "fence: enterAfterFence counter");
    check (domain.reacquireAfterFence() >= 1u, "fence: reacquireAfterFence counter");

    // fence は reader の入場だけを止める。NonRT の enqueue は継続できる。
    static int sentinel = 0;
    auto noopDeleter = [] (void*) noexcept {};
    check (domain.enqueueRetire (&sentinel, noopDeleter, domain.currentEpoch()),
           "fence: enqueueRetire must still succeed");
    domain.drainAll();
}

//==============================================================================
// shutdown の強制解放除外（retire drain exclusion）
//   除外中は再入場を拒否し、解除後は従来どおり再入場できること。
//   fence（永久）とは異なり、V11 E1 の規則を壊さないことを固定する。
//==============================================================================
void testRetireDrainExclusion()
{
    auto domain = std::make_unique<convo::EpochDomain> (convo::RetireDomainId::Unknown);

    const int slot = domain->registerReaderThread ("drain-exclusion");
    check (slot >= 0, "exclusion: registerReaderThread must succeed before close");
    if (slot < 0)
        return;

    // registerReaderThread が予約（admission）を保持しているので、そのまま enter する。
    // 一度 exit すれば slot は inactive に戻り、reacquire 可能になる。
    domain->enterReader (slot);
    domain->exitReader (slot);

    domain->beginRetireDrainExclusion();
    check (domain->retireDrainExclusionActive(), "exclusion: flag must be visible");
    check (! domain->reacquireReaderSlot (slot),
           "exclusion: reacquireReaderSlot must be refused during the exclusion");
    check (domain->reacquireDuringDrainExclusion() >= 1u,
           "exclusion: refusal must be counted");
    check (domain->readerAdmissionsActive() == 0u,
           "exclusion: a refused reacquire must not hold an admission");

    domain->endRetireDrainExclusion();
    check (! domain->retireDrainExclusionActive(), "exclusion: flag must clear");

    const bool reacquired = domain->reacquireReaderSlot (slot);
    check (reacquired, "exclusion: reacquireReaderSlot must succeed after the exclusion ends");
    if (reacquired)
    {
        // reacquire は slot を depth 0 で予約し、admission を保持したままである。
        // enter せずに exit すると、admission が残って下流の静止判定を壊す。
        domain->enterReader (slot);
        check (domain->activeDepthReaderCount() >= 1u,
               "exclusion: depth must be nonzero after enter");
        domain->exitReader (slot);
        check (domain->activeDepthReaderCount() == 0u,
               "exclusion: depth must return to 0 after exit");
        check (domain->readerAdmissionsActive() == 0u,
               "exclusion: no admission must remain after exit");
        check (domain->admissionPendingSlots() == 0u,
               "exclusion: no pending slot must remain after exit");
    }
}

//==============================================================================
// admission と shutdown の競合（決定論的 interleaving）
//   Reader が admission を取得した直後（fence 判定の前）に停止し、その間に
//   shutdown が fence / gate を閉じる。shutdown は入場処理の完了を待ち、
//   checked drain はその状態で成功しない。
//==============================================================================
struct SeamContext
{
    std::atomic<bool> reached { false };
    std::atomic<bool> release { false };
};

void admissionSeam (void* ctxPtr)
{
    auto* ctx = static_cast<SeamContext*> (ctxPtr);
    ctx->reached.store (true, std::memory_order_release);
    waitUntil (ctx->release);
}

void testAdmissionRace()
{
    auto domain = std::make_unique<convo::EpochDomain> (convo::RetireDomainId::Unknown);

    SeamContext ctx;
    std::atomic<int> slot { -1 };
    std::atomic<bool> registered { false };
    std::atomic<bool> reacquired { true };

    std::thread reader ([&]
    {
        const int idx = domain->registerReaderThread ("race");
        slot.store (idx, std::memory_order_release);
        registered.store (true, std::memory_order_release);
        if (idx < 0)
            return;

        domain->enterReader (idx);   // admission 終了（depth 1）
        domain->exitReader (idx);    // inactive（ownerThreadId は保持）

        domain->setReaderAdmissionSeam (&admissionSeam, &ctx);
        reacquired.store (domain->reacquireReaderSlot (idx), std::memory_order_release);
        domain->setReaderAdmissionSeam (nullptr, nullptr);
    });

    waitUntil (registered);
    waitUntil (ctx.reached);          // reader は admission 取得後・fence 判定前で停止

    // shutdown 側: fence / admission gate / publish gate を閉じる。
    domain->engageShutdownReaderFence();
    domain->closeReaderRegistration();

    // 入場処理中の reader が残っているため、静止確認は成立しない（待つ）。
    check (! domain->waitForReaderAdmissions (std::chrono::milliseconds (100)),
           "race: shutdown must wait for an in-flight reader admission");
    check (domain->readerAdmissionsActive() >= 1u, "race: in-flight admission is counted");
    check (domain->activeReaderCount() == 0, "race: in-flight admission is not an active reader");
    check (domain->closePublishersAndWaitIdle (std::chrono::milliseconds (50)),
           "race: publishers must become idle");

    auto queue = domain->borrow();
    check (queue != nullptr, "race: borrow");
    if (queue == nullptr)
    {
        ctx.release.store (true, std::memory_order_release);
        reader.join();
        return;
    }

    {
        auto cap = convo::DrainCapability::makeForTest (true, true, true, true,
                                                        domain->epochAtFence(), 1, 1);
        check (domain->drainQueue (*queue, std::move (cap))
                   == convo::EpochDomain::DrainResult::ReadersNotQuiescent,
               "race: checked drain must fail while a reader admission is in flight");
    }

    // reader を再開する。fence は閉じているため入場できない。
    ctx.release.store (true, std::memory_order_release);
    reader.join();

    check (! reacquired.load(), "race: reader must be refused after the fence closed");
    check (domain->reacquireAfterFence() >= 1u, "race: refused reacquire is counted");
    check (domain->waitForReaderAdmissions (std::chrono::milliseconds (500)),
           "race: admissions become idle after the reader is refused");
    check (domain->admissionPendingSlots() == 0, "race: no reserved slot remains");
    check (domain->registerReaderThread ("late") == -1,
           "race: new registration after the fence must be refused");
    check (domain->reacquireReaderSlot (slot.load()) == false,
           "race: later reacquire must still be refused");
    check (domain->activeReaderCount() == 0, "race: no active reader after the race");

    {
        auto cap = convo::DrainCapability::makeForTest (true, true, true, true,
                                                        domain->epochAtFence(), 1, 1);
        check (domain->drainQueue (*queue, std::move (cap))
                   == convo::EpochDomain::DrainResult::Completed,
               "race: checked drain succeeds after quiescence");
    }
}

//==============================================================================
// 予約〜enter の間（depth 0・slot 予約済み）に shutdown が入る場合。
//   admission として数えられるため shutdown は待ち、enter 後は active reader と
//   して drain が拒否される。
//==============================================================================
void testReservedNotEntered()
{
    auto domain = std::make_unique<convo::EpochDomain> (convo::RetireDomainId::Unknown);

    std::atomic<int> slot { -1 };
    std::atomic<bool> registered { false };
    std::atomic<bool> released { false };
    std::atomic<bool> entered { false };

    std::thread reader ([&]
    {
        const int idx = domain->registerReaderThread ("reserved");
        slot.store (idx, std::memory_order_release);
        registered.store (true, std::memory_order_release);
        if (idx < 0)
            return;

        waitUntil (released);        // 予約だけ保持（admission 中・depth 0）
        domain->enterReader (idx);
        entered.store (true, std::memory_order_release);
    });

    waitUntil (registered);
    check (slot.load() >= 0, "reserved: registration");
    check (domain->admissionPendingSlots() == 1u, "reserved: reservation is an admission");

    domain->engageShutdownReaderFence();
    domain->closeReaderRegistration();

    check (! domain->waitForReaderAdmissions (std::chrono::milliseconds (100)),
           "reserved: shutdown must wait for the reservation");
    check (domain->activeReaderCount() == 0, "reserved: reserved slot is not active");
    check (domain->closePublishersAndWaitIdle (std::chrono::milliseconds (50)),
           "reserved: publishers must become idle");

    auto queue = domain->borrow();
    check (queue != nullptr, "reserved: borrow");
    if (queue == nullptr)
    {
        released.store (true, std::memory_order_release);
        reader.join();
        return;
    }

    {
        auto cap = convo::DrainCapability::makeForTest (true, true, true, true,
                                                        domain->epochAtFence(), 1, 1);
        check (domain->drainQueue (*queue, std::move (cap))
                   == convo::EpochDomain::DrainResult::ReadersNotQuiescent,
               "reserved: checked drain must fail while a reservation is open");
    }

    released.store (true, std::memory_order_release);
    waitUntil (entered);

    check (domain->waitForReaderAdmissions (std::chrono::milliseconds (500)),
           "reserved: admission ends at enterReader");
    check (domain->admissionPendingSlots() == 0, "reserved: reservation flag cleared");
    check (domain->activeReaderCount() == 1, "reserved: entered reader is active");

    {
        auto cap = convo::DrainCapability::makeForTest (true, true, true, true,
                                                        domain->epochAtFence(), 1, 1);
        check (domain->drainQueue (*queue, std::move (cap))
                   == convo::EpochDomain::DrainResult::ReadersNotQuiescent,
               "reserved: checked drain must fail while the reader is active");
    }

    domain->exitReader (slot.load());
    reader.join();

    {
        auto cap = convo::DrainCapability::makeForTest (true, true, true, true,
                                                        domain->epochAtFence(), 1, 1);
        check (domain->drainQueue (*queue, std::move (cap))
                   == convo::EpochDomain::DrainResult::Completed,
               "reserved: checked drain succeeds after the reader exits");
    }
}

void testDrainContract()
{
    auto domain = std::make_unique<convo::EpochDomain> (convo::RetireDomainId::Unknown);
    auto queue = domain->borrow();
    check (queue != nullptr, "drain: borrow");

    if (! queue)
        return;

    // 非空 queue で deleter の exactly-once を確認する。
    static int sentinel = 0;
    g_deleted.store (0, std::memory_order_relaxed);
    check (domain->enqueueRetire (&sentinel, countingDeleter, domain->currentEpoch()),
           "drain: enqueue sentinel");

    // (1) reader 登録が開いたまま → ReadersNotQuiescent（drain しない・保持）。
    {
        auto cap = convo::DrainCapability::makeForTest (true, true, true, true,
                                                        domain->currentEpoch(), 1, 1);
        const auto r = domain->drainQueue (*queue, std::move (cap));
        check (r == convo::EpochDomain::DrainResult::ReadersNotQuiescent,
               "drain: open reader registration must be rejected");
        check (g_deleted.load() == 0, "drain: rejected drain must not run the deleter");
        check (queue->sizeApprox() == 1u, "drain: rejected drain must retain the entry");
    }

    // (2) fence + 登録遮断 + gate 閉鎖（idle 確認）→ Completed（deleter 1 回）。
    domain->engageShutdownReaderFence();
    domain->closeReaderRegistration();
    check (domain->closePublishersAndWaitIdle (std::chrono::milliseconds (50)),
           "drain: publishers must become idle");
    {
        auto cap = convo::DrainCapability::makeForTest (true, true, true, true,
                                                        domain->epochAtFence(), 1, 1);
        const auto r = domain->drainQueue (*queue, std::move (cap));
        check (r == convo::EpochDomain::DrainResult::Completed,
               "drain: quiescent state must complete");
        check (g_deleted.load() == 1, "drain: deleter must run exactly once");
        check (queue->sizeApprox() == 0u, "drain: queue must be empty after the drain");
    }

    // (3) capability の前提フラグが欠ける → CapabilityInvalid。
    {
        auto cap = convo::DrainCapability::makeForTest (false, true, true, true,
                                                        domain->epochAtFence(), 1, 1);
        const auto r = domain->drainQueue (*queue, std::move (cap));
        check (r == convo::EpochDomain::DrainResult::CapabilityInvalid,
               "drain: missing producers-joined flag must be rejected");
    }

    // (4) epoch 不一致 → ReadersNotQuiescent。
    {
        auto cap = convo::DrainCapability::makeForTest (true, true, true, true,
                                                        domain->epochAtFence() + 1000, 1, 1);
        const auto r = domain->drainQueue (*queue, std::move (cap));
        check (r == convo::EpochDomain::DrainResult::ReadersNotQuiescent,
               "drain: epoch mismatch must be rejected");
    }

    // (5) capability は single-use。
    {
        auto cap = convo::DrainCapability::makeForTest (true, true, true, true,
                                                        domain->epochAtFence(), 1, 1);
        check (cap.consume(), "drain: first consume");
        check (! cap.consume(), "drain: second consume must fail");
    }

    // (6) 二重 consume の capability では drain しない（AlreadyConsumed）。
    {
        auto cap = convo::DrainCapability::makeForTest (true, true, true, true,
                                                        domain->epochAtFence(), 1, 1);
        check (cap.consume(), "drain: consume before reuse");
        const auto r = domain->drainQueue (*queue, std::move (cap));
        check (r == convo::EpochDomain::DrainResult::AlreadyConsumed,
               "drain: reused capability must be rejected");
    }
}

void testQuarantine()
{
    auto domain = std::make_unique<convo::EpochDomain> (convo::RetireDomainId::Unknown);
    auto local = domain->borrow();
    check (local != nullptr, "quarantine: borrow");

    if (! local)
        return;

    static int sentinel = 0;
    g_deleted.store (0, std::memory_order_relaxed);
    check (domain->enqueueRetire (&sentinel, countingDeleter, domain->currentEpoch()),
           "quarantine: enqueue sentinel");

    const auto countBefore = convo::EpochDomain::quarantineCount();
    const auto pendingBefore = convo::EpochDomain::quarantinePendingEntries();
    const auto pendingInQueue = static_cast<std::uint64_t> (local->sizeApprox());
    check (pendingInQueue == 1u, "quarantine: queue must hold one entry");
    auto* raw = local.get();

    const auto first = domain->quarantineQueue (local);
    const auto second = domain->quarantineQueue (local);
    local.reset();

    check (first == convo::EpochDomain::QuarantineResult::Registered,
           "quarantine: first registration");
    check (second == convo::EpochDomain::QuarantineResult::AlreadyRegistered,
           "quarantine: second registration must be refused");
    check (convo::EpochDomain::quarantineCount() == countBefore + 1u,
           "quarantine: count must increase once");
    check (convo::EpochDomain::quarantinePendingEntries() == pendingBefore + pendingInQueue,
           "quarantine: pending entries must accumulate once");
    check (convo::EpochDomain::quarantinePendingEntries() > pendingBefore,
           "quarantine: pending increment must not be vacuous");
    check (g_deleted.load() == 0, "quarantine: deleter must not run");
    check (raw != nullptr && raw->sizeApprox() == static_cast<std::uint32_t> (pendingInQueue),
           "quarantine: queue must be retained by the self reference");
}

} // namespace

//==============================================================================
// ★ WORK113 R1 (2026-11-19 Owner 承認): exit 処理中（depth==0・provider 後処理が残る）に
//   drainQueue が静止を主張しないこと。
//   seam で depth デクリメント直後に止め、同じ構成で exit 完了後に
//   Completed になることを対照に取る（他の条件が成立していることの証拠）。
//==============================================================================
void testExitInProgressBlocksDrain()
{
    auto domain = std::make_unique<convo::EpochDomain> (convo::RetireDomainId::Unknown);

    struct Seam
    {
        std::atomic<bool> fired { false };      // 1 回だけ止めるためのフラグ
        std::atomic<bool> reached { false };
        std::atomic<bool> release { false };
    };

    static Seam seam;
    seam.fired.store (false);
    seam.reached.store (false);
    seam.release.store (false);

    convo::EpochDomain::setReaderExitSeam ([] (void* p) noexcept
    {
        // ★ 1 回だけ止める（他の exit を止めない）。
        //   seam 自体はここで解除しない（解除は join 後）。
        auto* s2 = static_cast<Seam*> (p);
        if (s2 == nullptr)
            return;
        bool expected = false;
        if (! s2->fired.compare_exchange_strong (expected, true))
            return;
        s2->reached.store (true, std::memory_order_release);
        while (! s2->release.load (std::memory_order_acquire))
            std::this_thread::yield();
    }, &seam);

    std::atomic<int> slot { -1 };
    std::atomic<bool> entered { false };
    std::atomic<bool> goExit { false };

    std::thread reader ([&]
    {
        const int idx = domain->registerReaderThread ("exit-window");
        slot.store (idx, std::memory_order_release);
        if (idx < 0)
            return;
        domain->enterReader (idx);
        entered.store (true, std::memory_order_release);
        waitUntil (goExit);
        domain->exitReader (idx);          // seam で depth デクリメント直後に止まる
    });

    // タイムアウト時は worker を解放して join し、seam を外してから FAIL にする。
    auto cleanupWorker = [&]
    {
        goExit.store (true, std::memory_order_release);
        seam.release.store (true, std::memory_order_release);
        if (reader.joinable())
            reader.join();
        // ★ 解除は join の後。実行中の exitReader が旧 fn と nullptr の ctx を
        //   組み合わせて呼ぶ経路を作らない。
        convo::EpochDomain::setReaderExitSeam (nullptr, nullptr);
    };

    if (! waitUntilBounded (entered, std::chrono::milliseconds (2000)))
    {
        check (false, "exit-window: reader did not enter (registration failed or enter stalled)");
        cleanupWorker();
        return;
    }
    check (slot.load() >= 0, "exit-window: registration");

    // shutdown 側の前提を整える（fence・登録閉鎖・bublisher idle）。
    domain->engageShutdownReaderFence();
    domain->closeReaderRegistration();
    check (domain->closePublishersAndWaitIdle (std::chrono::milliseconds (50)),
           "exit-window: publishers must become idle");

    auto queue = domain->borrow();
    check (queue != nullptr, "exit-window: borrow");
    if (queue == nullptr)
    {
        cleanupWorker();
        return;
    }

    // 拒否が「entry を処理しない」ことの証拠として sentinel を 1 件積む。
    static int sentinel = 0;
    g_deleted.store (0, std::memory_order_relaxed);
    check (domain->enqueueRetire (&sentinel, countingDeleter, domain->currentEpoch()),
           "exit-window: enqueue sentinel");

    goExit.store (true, std::memory_order_release);
    if (! waitUntilBounded (seam.reached, std::chrono::milliseconds (2000)))
    {
        check (false, "exit-window: exit seam was not reached");
        cleanupWorker();
        return;
    }

    // 窓の観測：depth は 0、活動数は 1 以上、admission は idle。
    const std::uint32_t depthReaders = domain->activeDepthReaderCount();
    const std::uint32_t activity = domain->activeReaderCount();
    check (depthReaders == 0u, "exit-window: depth is 0 inside the window");
    check (activity >= 1u, "exit-window: exit-in-progress is counted as activity");
    check (domain->readerAdmissionsActive() == 0u, "exit-window: admission active is 0");
    check (domain->admissionPendingSlots() == 0u, "exit-window: no pending admission slot");

    {
        auto cap = convo::DrainCapability::makeForTest (true, true, true, true,
                                                        domain->epochAtFence(), 1, 1);
        check (domain->drainQueue (*queue, std::move (cap))
                   == convo::EpochDomain::DrainResult::ReadersNotQuiescent,
               "exit-window: checked drain must fail while exit is in progress");
    }
    check (g_deleted.load() == 0, "exit-window: rejected drain must not run the deleter");
    check (queue->sizeApprox() == 1u, "exit-window: rejected drain must retain the entry");

    // exit を完了させる。同じ構成で Completed になれば、上の拒否は exit カウンタが理由だ。
    seam.release.store (true, std::memory_order_release);
    reader.join();
    convo::EpochDomain::setReaderExitSeam (nullptr, nullptr);

    check (domain->activeReaderCount() == 0u, "exit-window: activity returns to 0 after exit");

    {
        auto cap = convo::DrainCapability::makeForTest (true, true, true, true,
                                                        domain->epochAtFence(), 1, 1);
        check (domain->drainQueue (*queue, std::move (cap))
                   == convo::EpochDomain::DrainResult::Completed,
               "exit-window: checked drain succeeds after exit completes");
    }
    check (g_deleted.load() == 1, "exit-window: completed drain must run the deleter exactly once");
    check (queue->sizeApprox() == 0u, "exit-window: queue must be empty after the drain");
}

//==============================================================================
// ★ WORK113 R1: exitReader の早期 return（previousDepth == 0 / > 1）でも
//   exit カウンタが元に戻ること。
//   差（activeReaderCount - activeDepthReaderCount）が exit 実行中の件数である。
//==============================================================================
void testExitCounterRestoredOnEarlyReturns()
{
    auto domain = std::make_unique<convo::EpochDomain> (convo::RetireDomainId::Unknown);

    const int idx = domain->registerReaderThread ("early-return");
    check (idx >= 0, "early: registration");
    if (idx < 0)
        return;

    auto exitInProgress = [&domain]() -> std::uint32_t
    {
        const std::uint32_t activity = domain->activeReaderCount();
        const std::uint32_t depthReaders = domain->activeDepthReaderCount();
        return activity > depthReaders ? activity - depthReaders : 0u;
    };

    domain->enterReader (idx);          // depth 1
    domain->enterReader (idx);          // depth 2（ネスト）
    check (domain->activeDepthReaderCount() == 1u, "early: nested reader counts once");

    domain->exitReader (idx);           // previousDepth == 2 → 早期 return
    check (exitInProgress() == 0u, "early: nested exit restores the counter");
    check (domain->activeReaderCount() == 1u, "early: nested exit keeps activity 1");

    domain->exitReader (idx);           // previousDepth == 1 → 通常終了
    check (exitInProgress() == 0u, "early: outer exit restores the counter");
    check (domain->activeReaderCount() == 0u, "early: activity returns to 0");

    domain->exitReader (idx);           // previousDepth == 0 → underflow の早期 return
    check (exitInProgress() == 0u, "early: underflow exit restores the counter");
    check (domain->activeReaderCount() == 0u, "early: activity stays 0 after underflow");
}

int main()
{
    testPublishGate();

    {
        auto domain = std::make_unique<convo::EpochDomain> (convo::RetireDomainId::Unknown);
        testReaderFence (*domain);
    }

    testAdmissionRace();
    testReservedNotEntered();
    testExitInProgressBlocksDrain();
    testExitCounterRestoredOnEarlyReturns();
    testRetireDrainExclusion();
    testDrainContract();
    testQuarantine();

    if (g_failures != 0)
    {
        std::fprintf (stderr, "[drain-test] %d failure(s)\n", g_failures);
        return 1;
    }

    std::printf ("[drain-test] ALL PASS (gate / fence / admission / drain / capability / quarantine)\n");
    return 0;
}
