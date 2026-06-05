#include <stdexcept>
#include <vector>
#include <cstring>
#include <limits>
#include <cstdio>

#include "audioengine/ISRClosure.h"
#include "audioengine/ISRPayloadTier.h"
#include "audioengine/ISRRuntimePublicationCoordinator.h"

namespace {

[[nodiscard]] bool testInvalidClosureRejected()
{
    convo::isr::RuntimePublicationCoordinator coordinator;

    convo::isr::PayloadClosureDescriptor invalid {};
    invalid.closureId = 0; // invalid by contract

    convo::isr::TieredPayloadDescriptor descriptor {};
    descriptor.tier = convo::isr::PayloadTier::InlineImmutable;
    descriptor.requiresRT = false;
    descriptor.hasExternalResource = false;
    descriptor.pinnedLifetime = true;

    if (coordinator.precheckPublish(invalid, descriptor))
        return false;

    if (std::strcmp(coordinator.lastRejectReason(), "invalid closure graph") != 0)
        return false;

    return true;
}

[[nodiscard]] bool testInvalidTierRejected()
{
    convo::isr::RuntimePublicationCoordinator coordinator;

    convo::isr::PayloadClosureDescriptor closure {};
    closure.closureId = 1;
    closure.nodes.push_back(convo::isr::ClosureNodeRef {
        1u,
        static_cast<std::uint32_t>(convo::isr::PayloadTier::InlineImmutable),
        1u,
        1u,
        1u,
        1u,
        1u,
        1u,
        1u
    });

    convo::isr::TieredPayloadDescriptor descriptor {};
    descriptor.tier = convo::isr::PayloadTier::Forbidden; // invalid by publish policy
    descriptor.requiresRT = false;
    descriptor.hasExternalResource = false;
    descriptor.pinnedLifetime = true;

    if (coordinator.precheckPublish(closure, descriptor))
        return false;

    if (std::strcmp(coordinator.lastRejectReason(), "invalid payload tier") != 0)
        return false;

    return true;
}

[[nodiscard]] bool testCoordinatorCommitAndMonotonicityContract()
{
    convo::isr::RuntimePublicationCoordinator coordinator;

    int world1 = 1;
    int world2 = 2;

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       &world1,
                       1,
                       1,
                       1,
                       1);

    if (coordinator.getCurrent() != &world1)
        return false;
    if (coordinator.getVersion() != 1)
        return false;
    if (coordinator.getState() != convo::isr::RuntimePublicationCoordinator::CoordinatorState::Ready)
        return false;

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       &world2,
                       2,
                       2,
                       2,
                       2);
    if (coordinator.getCurrent() != &world2)
        return false;
    if (coordinator.getVersion() != 2)
        return false;

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       &world1,
                       1,
                       1,
                       1,
                       1);

    if (coordinator.getCurrent() != &world2)
        return false;
    if (coordinator.getState() != convo::isr::RuntimePublicationCoordinator::CoordinatorState::Faulted)
        return false;

    return true;
}

[[nodiscard]] bool testCoordinatorRejectEpochRollbackContract()
{
    convo::isr::RuntimePublicationCoordinator coordinator;

    int world1 = 1;
    int world2 = 2;

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       &world1,
                       1,
                       1,
                       5,
                       10);

    if (coordinator.getCurrent() != &world1)
        return false;

    // sequence は増加しても epoch rollback は fail-closed
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       &world2,
                       2,
                       2,
                       4,
                       11);

    if (coordinator.getCurrent() != &world1)
        return false;

    return coordinator.getState() == convo::isr::RuntimePublicationCoordinator::CoordinatorState::Faulted;
}

[[nodiscard]] bool testCoordinatorRejectMappedGenerationRollbackOnEpochAdvance()
{
    convo::isr::RuntimePublicationCoordinator coordinator;

    int world1 = 1;
    int world2 = 2;

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       &world1,
                       1,
                       10,
                       10,
                       100);

    if (coordinator.getCurrent() != &world1)
        return false;

    // epoch advance 時の mapped generation rollback は fail-closed
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       &world2,
                       2,
                       11,
                       11,
                       99);

    if (coordinator.getCurrent() != &world1)
        return false;

    return coordinator.getState() == convo::isr::RuntimePublicationCoordinator::CoordinatorState::Faulted;
}

[[nodiscard]] bool testCoordinatorRejectEpochReuseContract()
{
    convo::isr::RuntimePublicationCoordinator coordinator;

    int world1 = 1;
    int world2 = 2;

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       &world1,
                       1,
                       100,
                       100,
                       1000);

    if (coordinator.getCurrent() != &world1)
        return false;

    // sequence が進んでも epoch reuse は strict monotonic 契約違反
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       &world2,
                       2,
                       101,
                       100,
                       1001);

    if (coordinator.getCurrent() != &world1)
        return false;

    return coordinator.getState() == convo::isr::RuntimePublicationCoordinator::CoordinatorState::Faulted;
}

[[nodiscard]] bool testCoordinatorRejectMappedGenerationReuseContract()
{
    convo::isr::RuntimePublicationCoordinator coordinator;

    int world1 = 1;
    int world2 = 2;

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       &world1,
                       1,
                       200,
                       200,
                       5000);

    if (coordinator.getCurrent() != &world1)
        return false;

    // epoch が進んでも mapped generation reuse は strict monotonic 契約違反
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       &world2,
                       2,
                       201,
                       201,
                       5000);

    if (coordinator.getCurrent() != &world1)
        return false;

    return coordinator.getState() == convo::isr::RuntimePublicationCoordinator::CoordinatorState::Faulted;
}

[[nodiscard]] bool testCoordinatorRejectWraparoundContract()
{
    convo::isr::RuntimePublicationCoordinator coordinator;

    int world1 = 1;
    int world2 = 2;
    int world3 = 3;

    constexpr std::uint64_t maxValue = std::numeric_limits<std::uint64_t>::max();

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       &world1,
                       maxValue - 1,
                       maxValue - 1,
                       maxValue - 1,
                       maxValue - 1);

    if (coordinator.getCurrent() != &world1)
        return false;

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       &world2,
                       maxValue,
                       maxValue,
                       maxValue,
                       maxValue);

    if (coordinator.getCurrent() != &world2)
        return false;

    // wraparound（max -> 0）は strict monotonic 契約違反
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       &world3,
                       0,
                       0,
                       0,
                       0);

    if (coordinator.getCurrent() != &world2)
        return false;

    return coordinator.getState() == convo::isr::RuntimePublicationCoordinator::CoordinatorState::Faulted;
}

[[nodiscard]] bool testCoordinatorDrainAndShutdownContract()
{
    convo::isr::RuntimePublicationCoordinator coordinator;
    int world = 1;
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       &world,
                       1,
                       1,
                       1,
                       1);

    coordinator.setRetireBacklogCount(0);
    coordinator.setPublicationBacklogCount(0);
    coordinator.setPendingIntentCount(0);
    coordinator.setFallbackBacklogCount(0);
    coordinator.setReclaimInFlightCount(0);
    coordinator.setDeferredRetireResidencyCount(0);
    coordinator.setSwapPending(false);

    if (!coordinator.isFullyDrained())
        return false;

    coordinator.requestShutdown();
    coordinator.markShutdownComplete();

    if (coordinator.getState() != convo::isr::RuntimePublicationCoordinator::CoordinatorState::Bootstrapping)
        return false;

    return true;
}

[[nodiscard]] bool testShutdownCompleteFailsWhenNotDrained()
{
    convo::isr::RuntimePublicationCoordinator coordinator;
    int world = 1;

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       &world,
                       1,
                       1,
                       1,
                       1);

    coordinator.setRetireBacklogCount(1); // drained 条件を破る
    coordinator.setPublicationBacklogCount(0);
    coordinator.setPendingIntentCount(0);
    coordinator.setFallbackBacklogCount(0);
    coordinator.setReclaimInFlightCount(0);
    coordinator.setDeferredRetireResidencyCount(0);
    coordinator.setSwapPending(false);

    if (coordinator.isFullyDrained())
        return false;

    coordinator.requestShutdown();
    coordinator.markShutdownComplete();

    return coordinator.getState() == convo::isr::RuntimePublicationCoordinator::CoordinatorState::Faulted;
}

[[nodiscard]] bool testPressureStateNormalizationContract()
{
    convo::isr::RuntimePublicationCoordinator coordinator;
    int world = 1;

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       &world,
                       1,
                       1,
                       1,
                       1);

    // slope > threshold で Pressure へ遷移
    coordinator.setRetireBacklogCount(9);
    if (coordinator.getState() != convo::isr::RuntimePublicationCoordinator::CoordinatorState::Pressure)
        return false;

    // swapPending 中は normalization しない
    coordinator.setSwapPending(true);
    coordinator.setRetireBacklogCount(0);
    if (coordinator.getState() != convo::isr::RuntimePublicationCoordinator::CoordinatorState::Pressure)
        return false;

    // swapPending 解除後、3 window で Ready へ復帰
    coordinator.setSwapPending(false);
    coordinator.setRetireBacklogCount(0);
    if (coordinator.getState() != convo::isr::RuntimePublicationCoordinator::CoordinatorState::Pressure)
        return false;
    coordinator.setRetireBacklogCount(0);
    if (coordinator.getState() != convo::isr::RuntimePublicationCoordinator::CoordinatorState::Pressure)
        return false;
    coordinator.setRetireBacklogCount(0);

    return coordinator.getState() == convo::isr::RuntimePublicationCoordinator::CoordinatorState::Ready;
}

[[nodiscard]] bool testShutdownCompleteFailsWhenSwapPending()
{
    convo::isr::RuntimePublicationCoordinator coordinator;
    int world = 1;

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       &world,
                       1,
                       1,
                       1,
                       1);

    coordinator.setRetireBacklogCount(0);
    coordinator.setPublicationBacklogCount(0);
    coordinator.setPendingIntentCount(0);
    coordinator.setFallbackBacklogCount(0);
    coordinator.setReclaimInFlightCount(0);
    coordinator.setDeferredRetireResidencyCount(0);
    coordinator.setSwapPending(true); // drained 条件を破る

    if (coordinator.isFullyDrained())
        return false;

    coordinator.requestShutdown();
    coordinator.markShutdownComplete();

    return coordinator.getState() == convo::isr::RuntimePublicationCoordinator::CoordinatorState::Faulted;
}

// --- P4: Generation / ActivationEpoch 契約 ---
// generation 増加時は activationEpoch も必ず増加する (+1 以上)。
// 同一 generation での activationEpoch 単独変更は禁止。
[[nodiscard]] bool testP4SameGenerationEpochChangeRejected()
{
    convo::isr::RuntimePublicationCoordinator coordinator;

    int world1 = 1;
    int world2 = 2;

    // 初回 commit: gen=100, epoch=100
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       &world1,
                       1,
                       100,
                       100,
                       100);

    if (coordinator.getState() != convo::isr::RuntimePublicationCoordinator::CoordinatorState::Ready)
        return false;

    // 同一 generation (100) で epoch のみ変更 (101) → 禁止 (generation 不変で epoch 変更)
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       &world2,
                       2,
                       100,
                       101,
                       100);

    // world1 が維持され、Faulted になるべき
    if (coordinator.getCurrent() != &world1)
        return false;

    return coordinator.getState() == convo::isr::RuntimePublicationCoordinator::CoordinatorState::Faulted;
}

// --- P20: Fail-Closed Rollback ---
// reject 時に system state がロールバックされることを確認。
// coordinator は契約違反時に Faulted に遷移する（fail-closed）が、
// currentWorld と version は reject 前の値を維持する。
// 副作用（callback, telemetry）は reject 経路では発生しない。
[[nodiscard]] bool testP20RejectPreservesWorldState()
{
    convo::isr::RuntimePublicationCoordinator coordinator;

    int world1 = 1;
    int world2 = 2;

    // 初回 commit
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       &world1,
                       1, 1, 1, 1);

    if (coordinator.getCurrent() != &world1)
        return false;
    if (coordinator.getVersion() != 1)
        return false;

    // 不正な commit（epoch rollback）で reject されるはず
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       &world2,
                       2, 2, 0, 2);

    // state は Faulted に遷移する（fail-closed）: これは意図された動作
    if (coordinator.getState() != convo::isr::RuntimePublicationCoordinator::CoordinatorState::Faulted)
        return false;

    // currentWorld が reject 前の値（world1）を維持している
    if (coordinator.getCurrent() != &world1)
        return false;

    // version が reject 前の値（1）を維持している
    if (coordinator.getVersion() != 1)
        return false;

    return true;
}

} // namespace

int main()
{
    try
    {
    if (!testInvalidClosureRejected())
        throw std::runtime_error("invalid closure must be rejected");

    if (!testInvalidTierRejected())
        throw std::runtime_error("invalid tier must be rejected");

    if (!testCoordinatorCommitAndMonotonicityContract())
        throw std::runtime_error("coordinator monotonic commit contract failed");

    if (!testCoordinatorRejectEpochRollbackContract())
        throw std::runtime_error("coordinator epoch rollback contract failed");

    if (!testCoordinatorRejectMappedGenerationRollbackOnEpochAdvance())
        throw std::runtime_error("coordinator mapped generation rollback contract failed");

    if (!testCoordinatorRejectEpochReuseContract())
        throw std::runtime_error("coordinator epoch reuse contract failed");

    if (!testCoordinatorRejectMappedGenerationReuseContract())
        throw std::runtime_error("coordinator mapped generation reuse contract failed");

    if (!testCoordinatorRejectWraparoundContract())
        throw std::runtime_error("coordinator wraparound contract failed");

    if (!testCoordinatorDrainAndShutdownContract())
        throw std::runtime_error("coordinator drain and shutdown contract failed");

    if (!testShutdownCompleteFailsWhenNotDrained())
        throw std::runtime_error("coordinator shutdown not-drained contract failed");

    if (!testPressureStateNormalizationContract())
        throw std::runtime_error("coordinator pressure normalization contract failed");

    if (!testShutdownCompleteFailsWhenSwapPending())
        throw std::runtime_error("coordinator shutdown swap-pending contract failed");

    // --- P4 契約テスト群 ---
    if (!testP4SameGenerationEpochChangeRejected())
        throw std::runtime_error("P4: same-generation epoch change must be rejected");

    // --- P20 ロールバックテスト群 ---
    if (!testP20RejectPreservesWorldState())
        throw std::runtime_error("P20: reject must preserve world state");

    return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "TEST FAILED: %s\n", e.what());
        return 1;
    }
}
