#include <cstdint>
#include <stdexcept>
#include <memory>

#include "core/RuntimePublicationCoordinator.h"
#include "AlignedAllocation.h"

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

    void sealRecursively() noexcept {}
};

struct TestBridge
{
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
        convo::AlignedObjectDeleter<TestWorld>{}(world);
    }

    [[nodiscard]] std::uint64_t publishedCount() const noexcept { return publishedCount_; }
    [[nodiscard]] std::uint64_t retiredCount() const noexcept { return retiredCount_; }

private:
    std::uint64_t lastAcceptedToken_ = 0;
    std::uint64_t lastAcceptedGeneration_ = 0;
    std::uint64_t publishedCount_ = 0;
    std::uint64_t retiredCount_ = 0;
};

[[nodiscard]] bool testRejectRepublishAndRollback()
{
    using Coordinator = convo::RuntimePublicationCoordinator<TestWorld, const Candidate*, TestBridge>;
    using Store = Coordinator::Store;

    Store store;
    auto coordinator = Coordinator::create(TestBridge{}, store);

    auto w1 = convo::aligned_make_unique<TestWorld>();
    w1->token = 100;
    w1->generation = 100;
    w1->publicationSequence = 1;
    coordinator.publishWorld(std::move(w1));

    const TestWorld* first = Coordinator::consumeWorldHandle(store);
    if (first == nullptr || first->generation != 100 || first->publicationSequence != 1)
        return false;

    // Re-publish same token should be rejected.
    auto w2 = convo::aligned_make_unique<TestWorld>();
    w2->token = 100;
    w2->generation = 100;
    w2->publicationSequence = 2;
    coordinator.publishWorld(std::move(w2));

    const TestWorld* afterRepublish = Coordinator::consumeWorldHandle(store);
    if (afterRepublish != first || afterRepublish->publicationSequence != 1)
        return false;

    // Monotonic increase should pass.
    auto w3 = convo::aligned_make_unique<TestWorld>();
    w3->token = 101;
    w3->generation = 101;
    w3->publicationSequence = 2;
    coordinator.publishWorld(std::move(w3));

    const TestWorld* second = Coordinator::consumeWorldHandle(store);
    if (second == nullptr || second == first || second->generation != 101 || second->publicationSequence != 2)
        return false;

    // Rollback (101 -> 100) should be rejected.
    auto w4 = convo::aligned_make_unique<TestWorld>();
    w4->token = 100;
    w4->generation = 100;
    w4->publicationSequence = 3;
    coordinator.publishWorld(std::move(w4));

    const TestWorld* afterRollback = Coordinator::consumeWorldHandle(store);
    if (afterRollback != second || afterRollback->publicationSequence != 2)
        return false;

    coordinator.requestShutdownClearNonRt();
    coordinator.clearPublishedRuntimeSnapshotsNonRt();
    return true;
}

[[nodiscard]] bool testClearRequiresShutdownRequest()
{
    using Coordinator = convo::RuntimePublicationCoordinator<TestWorld, const Candidate*, TestBridge>;
    using Store = Coordinator::Store;

    Store store;
    auto coordinator = Coordinator::create(TestBridge{}, store);

    auto w1 = convo::aligned_make_unique<TestWorld>();
    w1->token = 100;
    w1->generation = 100;
    w1->publicationSequence = 1;
    coordinator.publishWorld(std::move(w1));

    const TestWorld* published = Coordinator::consumeWorldHandle(store);
    if (published == nullptr)
        return false;

    // shutdown request なし clear は no-op（publish(nullptr) を許可しない）
    coordinator.clearPublishedRuntimeSnapshotsNonRt();
    const TestWorld* stillPublished = Coordinator::consumeWorldHandle(store);
    if (stillPublished != published)
        return false;

    coordinator.requestShutdownClearNonRt();
    coordinator.clearPublishedRuntimeSnapshotsNonRt();

    const TestWorld* afterShutdownClear = Coordinator::consumeWorldHandle(store);
    return afterShutdownClear == nullptr;
}

} // namespace

int main()
{
    if (!testRejectRepublishAndRollback())
        throw std::runtime_error("RuntimePublicationCoordinator rejection contract failed");

    if (!testClearRequiresShutdownRequest())
        throw std::runtime_error("RuntimePublicationCoordinator shutdown clear contract failed");

    return 0;
}
