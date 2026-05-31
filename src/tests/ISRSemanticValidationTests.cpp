#include <stdexcept>
#include <vector>
#include <cstring>

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

    if (coordinator.getCurrent() != nullptr)
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
    if (coordinator.getCurrent() != nullptr)
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

    if (coordinator.getCurrent() != nullptr)
        return false;
    if (coordinator.getState() != convo::isr::RuntimePublicationCoordinator::CoordinatorState::Faulted)
        return false;

    return true;
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

} // namespace

int main()
{
    if (!testInvalidClosureRejected())
        throw std::runtime_error("invalid closure must be rejected");

    if (!testInvalidTierRejected())
        throw std::runtime_error("invalid tier must be rejected");

    if (!testCoordinatorCommitAndMonotonicityContract())
        throw std::runtime_error("coordinator monotonic commit contract failed");

    if (!testCoordinatorDrainAndShutdownContract())
        throw std::runtime_error("coordinator drain and shutdown contract failed");

    return 0;
}
