#include <cstdint>
#include <stdexcept>
#include <memory>

#include "core/RuntimePublicationCoordinator.h"

namespace {

struct Candidate
{
    std::uint64_t token = 0;
    std::uint64_t generation = 0;
};

struct TestWorld
{
    std::uint64_t token = 0;
    std::uint64_t generation = 0;
    std::uint64_t publicationSequence = 0;
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
        const Candidate* source = (current != nullptr) ? current : next;
        world->token = (source != nullptr) ? source->token : 0;
        world->generation = (source != nullptr) ? source->generation : 0;
        world->publicationSequence = nextSequence_ + 1;
        return world;
    }

    [[nodiscard]] bool validatePublicationNonRt(const TestWorld& world) noexcept
    {
        // Reject duplicated token (re-publish).
        if (lastAcceptedToken_ != 0 && world.token == lastAcceptedToken_)
            return false;

        // Reject rollback / non-monotonic generation.
        if (lastAcceptedGeneration_ != 0 && world.generation <= lastAcceptedGeneration_)
            return false;

        return true;
    }

    void didPublishRuntimeNonRt(const TestWorld& world) noexcept
    {
        lastAcceptedToken_ = world.token;
        lastAcceptedGeneration_ = world.generation;
        ++nextSequence_;
        publishedCount_++;
    }

    void willRetireRuntimeNonRt(const TestWorld* world) noexcept
    {
        if (world == nullptr)
            return;

        retiredCount_++;
    }

    void retireRuntimePublishWorldNonRt(TestWorld* world, bool) noexcept
    {
        delete world;
    }

    [[nodiscard]] std::uint64_t publishedCount() const noexcept { return publishedCount_; }
    [[nodiscard]] std::uint64_t retiredCount() const noexcept { return retiredCount_; }

private:
    std::uint64_t lastAcceptedToken_ = 0;
    std::uint64_t lastAcceptedGeneration_ = 0;
    std::uint64_t nextSequence_ = 0;
    std::uint64_t publishedCount_ = 0;
    std::uint64_t retiredCount_ = 0;
};

[[nodiscard]] bool testRejectRepublishAndRollback()
{
    using Coordinator = convo::RuntimePublicationCoordinator<TestWorld, const Candidate*, TestBridge>;
    using Store = Coordinator::Store;

    Store store;
    auto coordinator = Coordinator::create(TestBridge{}, store);

    Candidate c100 { 100, 100 };
    Candidate c101 { 101, 101 };

    coordinator.publishState(&c100, nullptr, convo::TransitionPolicy::SmoothOnly, 0.0, false);
    const TestWorld* first = Coordinator::observeWorldHandle(store);
    if (first == nullptr || first->generation != 100 || first->publicationSequence != 1)
        return false;

    // Re-publish same token should be rejected.
    coordinator.publishState(&c100, nullptr, convo::TransitionPolicy::SmoothOnly, 0.0, false);
    const TestWorld* afterRepublish = Coordinator::observeWorldHandle(store);
    if (afterRepublish != first || afterRepublish->publicationSequence != 1)
        return false;

    // Monotonic increase should pass.
    coordinator.publishState(&c101, nullptr, convo::TransitionPolicy::SmoothOnly, 0.0, false);
    const TestWorld* second = Coordinator::observeWorldHandle(store);
    if (second == nullptr || second == first || second->generation != 101 || second->publicationSequence != 2)
        return false;

    // Rollback (101 -> 100) should be rejected.
    coordinator.publishState(&c100, nullptr, convo::TransitionPolicy::SmoothOnly, 0.0, false);
    const TestWorld* afterRollback = Coordinator::observeWorldHandle(store);
    if (afterRollback != second || afterRollback->publicationSequence != 2)
        return false;

    coordinator.clearPublishedRuntimeSnapshotsNonRt();
    return true;
}

} // namespace

int main()
{
    if (!testRejectRepublishAndRollback())
        throw std::runtime_error("RuntimePublicationCoordinator rejection contract failed");

    return 0;
}
