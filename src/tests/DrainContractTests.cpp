//==============================================================================
// DrainContractTests.cpp — WORK113 Queue/Gate/Drain の契約テスト（単一 TU・テスト専用）
//
//   - publish gate の期限判定（active 中の waitForIdle は false、close 後の入場拒否）
//   - reader fence（register / reserve / reacquire の拒否とカウンタ）
//   - drainQueue の判定（reader 非静止・capability 不正・epoch 不一致・完了）
//   - capability の single-use
//   - 隔離の一意化と自己参照保持
//
//   本ファイルは CONVOPEQ_ENABLE_DRAIN_TEST_FACTORY=1 でビルドされ、
//   DrainCapability::makeForTest() だけが本番ヘッダに現れる（本番ビルドには存在しない）。
//==============================================================================

#include "core/EpochDomain.h"
#include "core/DrainCapability.h"

#include <chrono>
#include <cstdio>
#include <memory>

namespace
{

int g_failures = 0;

void check (bool ok, const char* what)
{
    if (! ok)
    {
        std::fprintf (stderr, "[drain-test] FAIL: %s\n", what);
        ++g_failures;
    }
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
    const auto noopDeleter = [] (void*) noexcept {};
    check (domain.enqueueRetire (&sentinel, noopDeleter, domain.currentEpoch()),
           "fence: enqueueRetire must still succeed");
    domain.drainAll();
}

void testDrainContract()
{
    auto domain = std::make_unique<convo::EpochDomain> (convo::RetireDomainId::Unknown);
    auto queue = domain->borrow();
    check (queue != nullptr, "drain: borrow");

    if (! queue)
        return;

    // (1) reader 登録が開いたまま → ReadersNotQuiescent（drain しない）。
    {
        auto cap = convo::DrainCapability::makeForTest (true, true, true, true,
                                                        domain->currentEpoch(), 1, 1);
        const auto r = domain->drainQueue (*queue, std::move (cap));
        check (r == convo::EpochDomain::DrainResult::ReadersNotQuiescent,
               "drain: open reader registration must be rejected");
    }

    // (2) fence + 登録遮断 + gate 閉鎖（idle 確認）→ Completed。
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
}

void testQuarantine()
{
    auto domain = std::make_unique<convo::EpochDomain> (convo::RetireDomainId::Unknown);
    auto local = domain->borrow();
    check (local != nullptr, "quarantine: borrow");

    if (! local)
        return;

    const auto countBefore = convo::EpochDomain::quarantineCount();
    const auto pendingBefore = convo::EpochDomain::quarantinePendingEntries();
    const auto pendingInQueue = static_cast<std::uint64_t> (local->sizeApprox());
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
    check (raw != nullptr && raw->sizeApprox() == static_cast<std::uint32_t> (pendingInQueue),
           "quarantine: queue must be retained by the self reference");
}

} // namespace

int main()
{
    testPublishGate();

    {
        auto domain = std::make_unique<convo::EpochDomain> (convo::RetireDomainId::Unknown);
        testReaderFence (*domain);
    }

    testDrainContract();
    testQuarantine();

    if (g_failures != 0)
    {
        std::fprintf (stderr, "[drain-test] %d failure(s)\n", g_failures);
        return 1;
    }

    std::printf ("[drain-test] ALL PASS (gate / fence / drain / capability / quarantine)\n");
    return 0;
}
