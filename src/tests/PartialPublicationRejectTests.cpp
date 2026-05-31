#include <cstdint>
#include <memory>
#include <stdexcept>

#include "core/RuntimePublicationCoordinator.h"

namespace {

struct Candidate
{
    std::uint64_t token = 0;
    std::uint64_t generation = 0;
    std::uint64_t publicationSequence = 0;
    bool semanticComplete = true;
};

struct TestWorld
{
    std::uint64_t token = 0;
    std::uint64_t generation = 0;
    std::uint64_t publicationSequence = 0;
    bool semanticComplete = true;
};

struct TestBridge
{
    std::unique_ptr<TestWorld> buildRuntimePublishWorld(const Candidate* current,
                                                        const Candidate* next,
                                                        convo::TransitionPolicy,
                                                        double,
                                                        bool) noexcept
    {
        auto world = std::make_unique<TestWorld>();
        const Candidate* source = (next != nullptr) ? next : current;
        if (source != nullptr)
        {
            world->token = source->token;
            world->generation = source->generation;
            world->publicationSequence = source->publicationSequence;
            world->semanticComplete = source->semanticComplete;
        }
        return world;
    }

    [[nodiscard]] bool validatePublicationNonRt(const TestWorld& world) noexcept
    {
        if (!world.semanticComplete)
            return false;
        if (world.generation == 0)
            return false;
        if (world.publicationSequence == 0)
            return false;
        return true;
    }

    void didPublishRuntimeNonRt(const TestWorld&) noexcept {}
    void willRetireRuntimeNonRt(const TestWorld*) noexcept {}
    void retireRuntimePublishWorldNonRt(TestWorld* world, bool) noexcept { delete world; }
};

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
    complete.semanticComplete = true;

    Candidate incomplete = complete;
    incomplete.token = 2;
    incomplete.generation = 101;
    incomplete.publicationSequence = 2;
    incomplete.semanticComplete = false;

    coordinator.publishState(&complete, nullptr, convo::TransitionPolicy::SmoothOnly, 0.0, false);
    const TestWorld* first = Coordinator::observeWorldHandle(store);
    if (first == nullptr)
        return false;
    if (first->generation != 100 || first->publicationSequence != 1)
        return false;

    // Incomplete world must be rejected and current published world must stay unchanged.
    coordinator.publishState(&incomplete, nullptr, convo::TransitionPolicy::SmoothOnly, 0.0, false);
    const TestWorld* afterReject = Coordinator::observeWorldHandle(store);
    if (afterReject != first)
        return false;

    coordinator.clearPublishedRuntimeSnapshotsNonRt();
    return true;
}

} // namespace

int main()
{
    if (!testRejectIncompleteSemanticWorld())
        throw std::runtime_error("partial publication reject contract failed");

    return 0;
}
