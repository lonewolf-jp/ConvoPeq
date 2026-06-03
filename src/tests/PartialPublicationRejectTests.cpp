#include <cstdint>
#include <memory>
#include <stdexcept>

#include "core/RuntimePublicationCoordinator.h"
#include "AlignedAllocation.h"

namespace {

struct Candidate
{
    std::uint64_t token = 0;
    std::uint64_t generation = 0;
    std::uint64_t publicationSequence = 0;
    std::uint64_t previousSequenceId = 0;
    std::uint64_t mappedRuntimeGeneration = 0;
    bool semanticComplete = true;
    int routingProcessingOrder = 0;
    bool routingEqBypassed = false;
    bool routingConvBypassed = false;
    bool graphEqBypassed = false;
    bool graphConvBypassed = false;
    std::uint64_t graphRuntimeUuid = 0;
    std::uint64_t graphFadingRuntimeUuid = 0;
    std::uint64_t topologyRuntimeUuid = 0;
    bool topologyHasFadingRuntime = false;
    std::uint64_t topologyFadingRuntimeUuid = 0;
    bool executionTransitionActive = false;
    int executionTransitionPolicy = 0;
    int executionCrossfadeStartDelayBlocks = 0;
    int executionCrossfadeDryHoldSamples = 0;
};

struct TestWorld
{
    std::uint64_t token = 0;
    std::uint64_t generation = 0;
    std::uint64_t publicationSequence = 0;
    std::uint64_t previousSequenceId = 0;
    std::uint64_t mappedRuntimeGeneration = 0;
    bool semanticComplete = true;
    int routingProcessingOrder = 0;
    bool routingEqBypassed = false;
    bool routingConvBypassed = false;
    bool graphEqBypassed = false;
    bool graphConvBypassed = false;
    std::uint64_t graphRuntimeUuid = 0;
    std::uint64_t graphFadingRuntimeUuid = 0;
    std::uint64_t topologyRuntimeUuid = 0;
    bool topologyHasFadingRuntime = false;
    std::uint64_t topologyFadingRuntimeUuid = 0;
    bool executionTransitionActive = false;
    int executionTransitionPolicy = 0;
    int executionCrossfadeStartDelayBlocks = 0;
    int executionCrossfadeDryHoldSamples = 0;
};

struct TestBridge
{
    [[nodiscard]] bool validatePublicationNonRt(const TestWorld& world) noexcept
    {
        if (!world.semanticComplete)
            return false;
        if (world.generation == 0)
            return false;
        if (world.publicationSequence == 0)
            return false;
        if (world.previousSequenceId >= world.publicationSequence)
            return false;
        if (world.mappedRuntimeGeneration != world.generation)
            return false;

        constexpr int kMinProcessingOrder = 0;
        constexpr int kMaxProcessingOrder = 1;
        if (world.routingProcessingOrder < kMinProcessingOrder
            || world.routingProcessingOrder > kMaxProcessingOrder)
            return false;

        constexpr int kMinTransitionPolicy = 0;
        constexpr int kMaxTransitionPolicy = 2;
        if (world.executionTransitionPolicy < kMinTransitionPolicy
            || world.executionTransitionPolicy > kMaxTransitionPolicy)
            return false;
        if (world.executionCrossfadeStartDelayBlocks < 0
            || world.executionCrossfadeDryHoldSamples < 0)
            return false;

        if (world.routingEqBypassed != world.graphEqBypassed)
            return false;
        if (world.routingConvBypassed != world.graphConvBypassed)
            return false;

        const bool hasGraphActiveNode = (world.graphRuntimeUuid != 0);
        const bool hasGraphFadingNode = (world.graphFadingRuntimeUuid != 0);
        if (hasGraphActiveNode != (world.topologyRuntimeUuid != 0))
            return false;
        if (hasGraphFadingNode != world.topologyHasFadingRuntime)
            return false;
        if (world.topologyRuntimeUuid != world.graphRuntimeUuid)
            return false;
        if (world.topologyFadingRuntimeUuid != world.graphFadingRuntimeUuid)
            return false;
        if (world.executionTransitionActive != world.topologyHasFadingRuntime)
            return false;

        return true;
    }

    void didPublishRuntimeNonRt(const TestWorld&) noexcept {}
    void willRetireRuntimeNonRt(const TestWorld*) noexcept {}
    void retireRuntimePublishWorldNonRt(TestWorld* world, bool) noexcept
    {
        convo::AlignedObjectDeleter<TestWorld>{}(world);
    }
};

convo::aligned_unique_ptr<TestWorld> createWorld(const Candidate& c)
{
    auto w = convo::aligned_make_unique<TestWorld>();
    w->token = c.token;
    w->generation = c.generation;
    w->publicationSequence = c.publicationSequence;
    w->previousSequenceId = c.previousSequenceId;
    w->mappedRuntimeGeneration = c.mappedRuntimeGeneration;
    w->semanticComplete = c.semanticComplete;
    w->routingProcessingOrder = c.routingProcessingOrder;
    w->routingEqBypassed = c.routingEqBypassed;
    w->routingConvBypassed = c.routingConvBypassed;
    w->graphEqBypassed = c.graphEqBypassed;
    w->graphConvBypassed = c.graphConvBypassed;
    w->graphRuntimeUuid = c.graphRuntimeUuid;
    w->graphFadingRuntimeUuid = c.graphFadingRuntimeUuid;
    w->topologyRuntimeUuid = c.topologyRuntimeUuid;
    w->topologyHasFadingRuntime = c.topologyHasFadingRuntime;
    w->topologyFadingRuntimeUuid = c.topologyFadingRuntimeUuid;
    w->executionTransitionActive = c.executionTransitionActive;
    w->executionTransitionPolicy = c.executionTransitionPolicy;
    w->executionCrossfadeStartDelayBlocks = c.executionCrossfadeStartDelayBlocks;
    w->executionCrossfadeDryHoldSamples = c.executionCrossfadeDryHoldSamples;
    return w;
}

[[nodiscard]] bool testRejectIncompleteSemanticWorld()
{
    using Coordinator = convo::RuntimePublicationCoordinator<TestWorld, const Candidate*, TestBridge>;
    using Store = Coordinator::Store;

    Store store;
    auto coordinator = Coordinator::create(TestBridge{}, store);

    Candidate complete {};
    complete.token = 1;
    complete.generation = 100;
    complete.publicationSequence = 1;
    complete.previousSequenceId = 0;
    complete.mappedRuntimeGeneration = 100;
    complete.semanticComplete = true;
    complete.routingProcessingOrder = 0;
    complete.routingEqBypassed = false;
    complete.routingConvBypassed = false;
    complete.graphEqBypassed = false;
    complete.graphConvBypassed = false;
    complete.graphRuntimeUuid = 10;
    complete.graphFadingRuntimeUuid = 0;
    complete.topologyRuntimeUuid = 10;
    complete.topologyHasFadingRuntime = false;
    complete.topologyFadingRuntimeUuid = 0;
    complete.executionTransitionActive = false;
    complete.executionTransitionPolicy = 0;
    complete.executionCrossfadeStartDelayBlocks = 0;
    complete.executionCrossfadeDryHoldSamples = 0;

    Candidate incomplete = complete;
    incomplete.token = 2;
    incomplete.generation = 101;
    incomplete.publicationSequence = 2;
    incomplete.previousSequenceId = 1;
    incomplete.mappedRuntimeGeneration = 101;
    incomplete.semanticComplete = false;

    coordinator.publishWorld(createWorld(complete));
    const TestWorld* first = Coordinator::consumeWorldHandle(store);
    if (first == nullptr)
        return false;
    if (first->generation != 100 || first->publicationSequence != 1)
        return false;

    // Incomplete world must be rejected and current published world must stay unchanged.
    coordinator.publishWorld(createWorld(incomplete));
    const TestWorld* afterReject = Coordinator::consumeWorldHandle(store);
    if (afterReject != first)
        return false;

    coordinator.requestShutdownClearNonRt();
    coordinator.clearPublishedRuntimeSnapshotsNonRt();
    return true;
}

[[nodiscard]] bool testRejectRuntimeGraphAuthorityMismatch()
{
    using Coordinator = convo::RuntimePublicationCoordinator<TestWorld, const Candidate*, TestBridge>;
    using Store = Coordinator::Store;

    Store store;
    auto coordinator = Coordinator::create(TestBridge{}, store);

    Candidate complete {};
    complete.token = 1;
    complete.generation = 200;
    complete.publicationSequence = 10;
    complete.previousSequenceId = 9;
    complete.mappedRuntimeGeneration = 200;
    complete.semanticComplete = true;
    complete.routingProcessingOrder = 0;
    complete.routingEqBypassed = false;
    complete.routingConvBypassed = true;
    complete.graphEqBypassed = false;
    complete.graphConvBypassed = true;
    complete.graphRuntimeUuid = 20;
    complete.graphFadingRuntimeUuid = 0;
    complete.topologyRuntimeUuid = 20;
    complete.topologyHasFadingRuntime = false;
    complete.topologyFadingRuntimeUuid = 0;
    complete.executionTransitionActive = false;
    complete.executionTransitionPolicy = 0;
    complete.executionCrossfadeStartDelayBlocks = 0;
    complete.executionCrossfadeDryHoldSamples = 0;

    Candidate authorityMismatch = complete;
    authorityMismatch.token = 2;
    authorityMismatch.generation = 201;
    authorityMismatch.publicationSequence = 11;
    authorityMismatch.previousSequenceId = 10;
    authorityMismatch.mappedRuntimeGeneration = 201;
    authorityMismatch.graphEqBypassed = true; // routingEqBypassed=false と不整合

    coordinator.publishWorld(createWorld(complete));
    const TestWorld* first = Coordinator::consumeWorldHandle(store);
    if (first == nullptr)
        return false;

    coordinator.publishWorld(createWorld(authorityMismatch));
    const TestWorld* afterReject = Coordinator::consumeWorldHandle(store);
    if (afterReject != first)
        return false;

    if (afterReject->generation != 200 || afterReject->publicationSequence != 10)
        return false;

    coordinator.requestShutdownClearNonRt();
    coordinator.clearPublishedRuntimeSnapshotsNonRt();
    return true;
}

[[nodiscard]] bool testRejectTransitionSemanticMismatch()
{
    using Coordinator = convo::RuntimePublicationCoordinator<TestWorld, const Candidate*, TestBridge>;
    using Store = Coordinator::Store;

    Store store;
    auto coordinator = Coordinator::create(TestBridge{}, store);

    Candidate complete {};
    complete.token = 1;
    complete.generation = 300;
    complete.publicationSequence = 20;
    complete.previousSequenceId = 19;
    complete.mappedRuntimeGeneration = 300;
    complete.semanticComplete = true;
    complete.routingProcessingOrder = 0;
    complete.routingEqBypassed = false;
    complete.routingConvBypassed = false;
    complete.graphEqBypassed = false;
    complete.graphConvBypassed = false;
    complete.graphRuntimeUuid = 30;
    complete.graphFadingRuntimeUuid = 31;
    complete.topologyRuntimeUuid = 30;
    complete.topologyHasFadingRuntime = true;
    complete.topologyFadingRuntimeUuid = 31;
    complete.executionTransitionActive = true;
    complete.executionTransitionPolicy = 0;
    complete.executionCrossfadeStartDelayBlocks = 0;
    complete.executionCrossfadeDryHoldSamples = 0;

    Candidate transitionMismatch = complete;
    transitionMismatch.token = 2;
    transitionMismatch.generation = 301;
    transitionMismatch.publicationSequence = 21;
    transitionMismatch.previousSequenceId = 20;
    transitionMismatch.mappedRuntimeGeneration = 301;
    transitionMismatch.executionTransitionActive = false; // topologyHasFadingRuntime=true と不整合

    coordinator.publishWorld(createWorld(complete));
    const TestWorld* first = Coordinator::consumeWorldHandle(store);
    if (first == nullptr)
        return false;

    coordinator.publishWorld(createWorld(transitionMismatch));
    const TestWorld* afterReject = Coordinator::consumeWorldHandle(store);
    if (afterReject != first)
        return false;

    if (afterReject->generation != 300 || afterReject->publicationSequence != 20)
        return false;

    coordinator.requestShutdownClearNonRt();
    coordinator.clearPublishedRuntimeSnapshotsNonRt();
    return true;
}

[[nodiscard]] bool testRejectPublicationSequenceRollback()
{
    using Coordinator = convo::RuntimePublicationCoordinator<TestWorld, const Candidate*, TestBridge>;
    using Store = Coordinator::Store;

    Store store;
    auto coordinator = Coordinator::create(TestBridge{}, store);

    Candidate complete {};
    complete.token = 1;
    complete.generation = 400;
    complete.publicationSequence = 30;
    complete.previousSequenceId = 29;
    complete.mappedRuntimeGeneration = 400;
    complete.semanticComplete = true;
    complete.routingProcessingOrder = 0;
    complete.routingEqBypassed = false;
    complete.routingConvBypassed = false;
    complete.graphEqBypassed = false;
    complete.graphConvBypassed = false;
    complete.graphRuntimeUuid = 40;
    complete.graphFadingRuntimeUuid = 0;
    complete.topologyRuntimeUuid = 40;
    complete.topologyHasFadingRuntime = false;
    complete.topologyFadingRuntimeUuid = 0;
    complete.executionTransitionActive = false;
    complete.executionTransitionPolicy = 0;
    complete.executionCrossfadeStartDelayBlocks = 0;
    complete.executionCrossfadeDryHoldSamples = 0;

    Candidate rollback = complete;
    rollback.token = 2;
    rollback.generation = 401;
    rollback.publicationSequence = 31;
    rollback.previousSequenceId = 31; // previous >= current は reject
    rollback.mappedRuntimeGeneration = 401;

    coordinator.publishWorld(createWorld(complete));
    const TestWorld* first = Coordinator::consumeWorldHandle(store);
    if (first == nullptr)
        return false;

    coordinator.publishWorld(createWorld(rollback));
    const TestWorld* afterReject = Coordinator::consumeWorldHandle(store);
    if (afterReject != first)
        return false;

    if (afterReject->generation != 400 || afterReject->publicationSequence != 30)
        return false;

    coordinator.requestShutdownClearNonRt();
    coordinator.clearPublishedRuntimeSnapshotsNonRt();
    return true;
}

[[nodiscard]] bool testRejectMappedRuntimeGenerationMismatch()
{
    using Coordinator = convo::RuntimePublicationCoordinator<TestWorld, const Candidate*, TestBridge>;
    using Store = Coordinator::Store;

    Store store;
    auto coordinator = Coordinator::create(TestBridge{}, store);

    Candidate complete {};
    complete.token = 1;
    complete.generation = 500;
    complete.publicationSequence = 40;
    complete.previousSequenceId = 39;
    complete.mappedRuntimeGeneration = 500;
    complete.semanticComplete = true;
    complete.routingProcessingOrder = 0;
    complete.routingEqBypassed = false;
    complete.routingConvBypassed = false;
    complete.graphEqBypassed = false;
    complete.graphConvBypassed = false;
    complete.graphRuntimeUuid = 50;
    complete.graphFadingRuntimeUuid = 0;
    complete.topologyRuntimeUuid = 50;
    complete.topologyHasFadingRuntime = false;
    complete.topologyFadingRuntimeUuid = 0;
    complete.executionTransitionActive = false;
    complete.executionTransitionPolicy = 0;
    complete.executionCrossfadeStartDelayBlocks = 0;
    complete.executionCrossfadeDryHoldSamples = 0;

    Candidate mappingMismatch = complete;
    mappingMismatch.token = 2;
    mappingMismatch.generation = 501;
    mappingMismatch.publicationSequence = 41;
    mappingMismatch.previousSequenceId = 40;
    mappingMismatch.mappedRuntimeGeneration = 999; // generation と不整合

    coordinator.publishWorld(createWorld(complete));
    const TestWorld* first = Coordinator::consumeWorldHandle(store);
    if (first == nullptr)
        return false;

    coordinator.publishWorld(createWorld(mappingMismatch));
    const TestWorld* afterReject = Coordinator::consumeWorldHandle(store);
    if (afterReject != first)
        return false;

    if (afterReject->generation != 500 || afterReject->publicationSequence != 40)
        return false;

    coordinator.requestShutdownClearNonRt();
    coordinator.clearPublishedRuntimeSnapshotsNonRt();
    return true;
}

[[nodiscard]] bool testRejectInvalidRoutingProcessingOrder()
{
    using Coordinator = convo::RuntimePublicationCoordinator<TestWorld, const Candidate*, TestBridge>;
    using Store = Coordinator::Store;

    Store store;
    auto coordinator = Coordinator::create(TestBridge{}, store);

    Candidate complete {};
    complete.token = 1;
    complete.generation = 600;
    complete.publicationSequence = 50;
    complete.previousSequenceId = 49;
    complete.mappedRuntimeGeneration = 600;
    complete.semanticComplete = true;
    complete.routingProcessingOrder = 0;
    complete.routingEqBypassed = false;
    complete.routingConvBypassed = false;
    complete.graphEqBypassed = false;
    complete.graphConvBypassed = false;
    complete.graphRuntimeUuid = 60;
    complete.graphFadingRuntimeUuid = 0;
    complete.topologyRuntimeUuid = 60;
    complete.topologyHasFadingRuntime = false;
    complete.topologyFadingRuntimeUuid = 0;
    complete.executionTransitionActive = false;
    complete.executionTransitionPolicy = 0;
    complete.executionCrossfadeStartDelayBlocks = 0;
    complete.executionCrossfadeDryHoldSamples = 0;

    Candidate invalidRouting = complete;
    invalidRouting.token = 2;
    invalidRouting.generation = 601;
    invalidRouting.publicationSequence = 51;
    invalidRouting.previousSequenceId = 50;
    invalidRouting.mappedRuntimeGeneration = 601;
    invalidRouting.routingProcessingOrder = 2; // schema上の上限(1)超過

    coordinator.publishWorld(createWorld(complete));
    const TestWorld* first = Coordinator::consumeWorldHandle(store);
    if (first == nullptr)
        return false;

    coordinator.publishWorld(createWorld(invalidRouting));
    const TestWorld* afterReject = Coordinator::consumeWorldHandle(store);
    if (afterReject != first)
        return false;

    if (afterReject->generation != 600 || afterReject->publicationSequence != 50)
        return false;

    coordinator.requestShutdownClearNonRt();
    coordinator.clearPublishedRuntimeSnapshotsNonRt();
    return true;
}

[[nodiscard]] bool testRejectInvalidExecutionPolicy()
{
    using Coordinator = convo::RuntimePublicationCoordinator<TestWorld, const Candidate*, TestBridge>;
    using Store = Coordinator::Store;

    Store store;
    auto coordinator = Coordinator::create(TestBridge{}, store);

    Candidate complete {};
    complete.token = 1;
    complete.generation = 700;
    complete.publicationSequence = 60;
    complete.previousSequenceId = 59;
    complete.mappedRuntimeGeneration = 700;
    complete.semanticComplete = true;
    complete.routingProcessingOrder = 0;
    complete.routingEqBypassed = false;
    complete.routingConvBypassed = false;
    complete.graphEqBypassed = false;
    complete.graphConvBypassed = false;
    complete.graphRuntimeUuid = 70;
    complete.graphFadingRuntimeUuid = 0;
    complete.topologyRuntimeUuid = 70;
    complete.topologyHasFadingRuntime = false;
    complete.topologyFadingRuntimeUuid = 0;
    complete.executionTransitionActive = false;
    complete.executionTransitionPolicy = 0;
    complete.executionCrossfadeStartDelayBlocks = 0;
    complete.executionCrossfadeDryHoldSamples = 0;

    Candidate invalidExecution = complete;
    invalidExecution.token = 2;
    invalidExecution.generation = 701;
    invalidExecution.publicationSequence = 61;
    invalidExecution.previousSequenceId = 60;
    invalidExecution.mappedRuntimeGeneration = 701;
    invalidExecution.executionTransitionPolicy = 3; // schema上の上限(2)超過

    coordinator.publishWorld(createWorld(complete));
    const TestWorld* first = Coordinator::consumeWorldHandle(store);
    if (first == nullptr)
        return false;

    coordinator.publishWorld(createWorld(invalidExecution));
    const TestWorld* afterReject = Coordinator::consumeWorldHandle(store);
    if (afterReject != first)
        return false;

    if (afterReject->generation != 700 || afterReject->publicationSequence != 60)
        return false;

    coordinator.requestShutdownClearNonRt();
    coordinator.clearPublishedRuntimeSnapshotsNonRt();
    return true;
}

} // namespace

int main()
{
    if (!testRejectIncompleteSemanticWorld())
        throw std::runtime_error("partial publication reject contract failed");

    if (!testRejectRuntimeGraphAuthorityMismatch())
        throw std::runtime_error("runtime graph authority reject contract failed");

    if (!testRejectTransitionSemanticMismatch())
        throw std::runtime_error("transition semantic mismatch reject contract failed");

    if (!testRejectPublicationSequenceRollback())
        throw std::runtime_error("publication sequence rollback reject contract failed");

    if (!testRejectMappedRuntimeGenerationMismatch())
        throw std::runtime_error("mapped runtime generation mismatch reject contract failed");

    if (!testRejectInvalidRoutingProcessingOrder())
        throw std::runtime_error("invalid routing processing order reject contract failed");

    if (!testRejectInvalidExecutionPolicy())
        throw std::runtime_error("invalid execution policy reject contract failed");

    return 0;
}
