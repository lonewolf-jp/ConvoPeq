// ★ B2: OwnerChannel unit tests (ADR-D3). JUCE-independent.
//   Verifies ownership transfer semantics in isolation, BEFORE any publish-path wiring
//   (B3): single-transfer, single-take, key isolation, no-overwrite, no-leak, no-double-free.

#include <cstdint>
#include <memory>
#include <stdexcept>

#include "audioengine/OwnerChannel.h"

// Move-only mock; unique_ptr<MockOwner> never move-constructs a MockOwner (it only
// transfers the internal pointer), so deleting MockOwner's move-ctor is safe and
// catches accidental copies.
struct MockOwner {
    int id;
    static int alive;
    explicit MockOwner(int i) : id(i) { ++alive; }
    ~MockOwner() { --alive; }
    MockOwner(const MockOwner&) = delete;
    MockOwner& operator=(const MockOwner&) = delete;
    MockOwner(MockOwner&&) = delete;
    MockOwner& operator=(MockOwner&&) = delete;
};
int MockOwner::alive = 0;

using Channel = convo::isr::OwnerChannel<std::unique_ptr<MockOwner>>;

// 1. enqueue -> take -> single-transfer (2nd take returns nullptr).
[[nodiscard]] bool testOwnerChannelBasicTransfer() {
    MockOwner::alive = 0;
    Channel ch;
    if (!ch.enqueue({1, 0, 0}, std::make_unique<MockOwner>(1)))
        return false;
    auto got = ch.take({1, 0, 0});
    if (!got || got->id != 1)
        return false;
    if (ch.take({1, 0, 0}))                 // second take -> drained -> nullptr
        return false;
    return MockOwner::alive == 1;           // `got` is the sole live owner
}

// 2. Wrong key: take(wrongKey) returns nullptr and does NOT drain the real owner.
[[nodiscard]] bool testOwnerChannelWrongKey() {
    MockOwner::alive = 0;
    Channel ch;
    ch.enqueue({11, 0, 5}, std::make_unique<MockOwner>(9));
    if (ch.take({99, 0, 5}))                // wrong seqId -> nullptr, owner retained
        return false;
    auto got = ch.take({11, 0, 5});          // correct key still present
    return got && got->id == 9;
}

// 3. Overwrite rejected: re-enqueue same key returns false; caller keeps its owner;
//    take returns the FIRST owner (never the rejected duplicate).
[[nodiscard]] bool testOwnerChannelOverwriteRejected() {
    MockOwner::alive = 0;
    Channel ch;
    ch.enqueue({5, 0, 0}, std::make_unique<MockOwner>(1));
    std::unique_ptr<MockOwner> dup = std::make_unique<MockOwner>(2);
    if (ch.enqueue({5, 0, 0}, std::move(dup)))
        return false;                      // reject duplicate key
    if (!dup || dup->id != 2)              // caller still owns the rejected owner
        return false;
    auto got = ch.take({5, 0, 0});
    return got && got->id == 1;            // first owner, not the dup
}

// 4. Lifetime: take + scope exit destroys the owner exactly once (no leak/double-free).
[[nodiscard]] bool testOwnerChannelLifetime() {
    MockOwner::alive = 0;
    {
        Channel ch;
        ch.enqueue({42, 1, 7}, std::make_unique<MockOwner>(5));
        auto got = ch.take({42, 1, 7});
        if (!got)
            return false;
        // got + ch destroyed at scope exit
    }
    return MockOwner::alive == 0;          // drained + destroyed, no leak
}

// 5. Stress: 100k enqueue/take round-trips, correct id each time, clean at end.
[[nodiscard]] bool testOwnerChannelStress100k() {
    MockOwner::alive = 0;
    Channel ch;
    for (std::uint64_t i = 1; i <= 100000; ++i) {
        const convo::isr::OwnerChannelKey key{ i, 0, i };
        if (!ch.enqueue(key, std::make_unique<MockOwner>(static_cast<int>(i))))
            return false;                  // 1 in-flight: channel never fills
        auto got = ch.take(key);
        if (!got || got->id != static_cast<int>(i))
            return false;
    }
    return MockOwner::alive == 0;          // every owner drained & destroyed
}

// 6. B3 backpressure: fill channel to capacity -> next enqueue returns false (caller
//    keeps owner, no silent drop); after a take, the channel accepts again.
[[nodiscard]] bool testOwnerChannelFullBackpressure() {
    MockOwner::alive = 0;
    Channel ch;
    constexpr std::size_t kCapacity = 256;
    for (std::size_t i = 0; i < kCapacity; ++i) {
        if (!ch.enqueue({ static_cast<std::uint64_t>(i + 1), 0, i },
                        std::make_unique<MockOwner>(static_cast<int>(i + 1))))
            return false;                  // distinct keys: must all be accepted up to capacity
    }
    if (ch.size() != kCapacity)
        return false;

    std::unique_ptr<MockOwner> extra = std::make_unique<MockOwner>(9999);
    if (ch.enqueue({ static_cast<std::uint64_t>(kCapacity + 1), 0, kCapacity },
                   std::move(extra)))
        return false;                      // full -> explicit reject (no overwrite, no silent drop)
    if (!extra || extra->id != 9999)       // caller retains the rejected owner
        return false;

    // after draining one, the channel accepts again (recovery)
    auto got = ch.take({ 1, 0, 0 });
    if (!got || got->id != 1)
        return false;
    if (ch.size() != kCapacity - 1)
        return false;
    if (!ch.enqueue({ static_cast<std::uint64_t>(kCapacity + 1), 0, kCapacity },
                    std::make_unique<MockOwner>(9999)))
        return false;
    return true;
}

// 7. drainAllNonRt: drains all residual owners via callback (no key needed).
//    - all slots drained → callback count matches enqueue count
//    - ownership relinquished: re-drain is no-op (slots_ empty after drain)
//    - single-transfer: callback receives each owner exactly once (no double-fire)
[[nodiscard]] bool testOwnerChannelDrainAllNonRt() {
    MockOwner::alive = 0;
    Channel ch;
    constexpr std::size_t kFill = 5;       // fill a few slots (distinct keys)
    for (std::size_t i = 0; i < kFill; ++i) {
        const convo::isr::OwnerChannelKey key{ i + 1, 0, i };
        if (!ch.enqueue(key, std::make_unique<MockOwner>(static_cast<int>(i + 1))))
            return false;
    }
    if (ch.size() != kFill)
        return false;

    // drainAllNonRt: callback must fire for each enqueued owner exactly once.
    int drained = 0;
    std::size_t count = ch.drainAllNonRt([&](const MockOwner* raw) {
        if (raw == nullptr) return;        // defensive
        ++drained;
        // ownership: callback receives the raw Owner* (not re-wrap); caller
        // owns the deletion semantics. Here we just count — the mock's dtor
        // runs when the test's unique_ptr scope ends.
    });
    if (count != kFill || drained != static_cast<int>(kFill))
        return false;

    // re-drain: all slots now nullptr -> no-op (single-transfer proven)
    std::size_t count2 = ch.drainAllNonRt([&](const MockOwner*) {});
    if (count2 != 0)
        return false;

    // slots_ fully drained (size() walks the same full scan)
    if (ch.size() != 0)
        return false;

    return true;                            // drained exactly kFill owners, re-drain no-op
}

// 8. drainAllNonRt does NOT touch wrong-key isolation: drain then enqueue(take) still works.
[[nodiscard]] bool testOwnerChannelDrainThenReenqueue() {
    MockOwner::alive = 0;
    Channel ch;
    ch.enqueue({7, 0, 0}, std::make_unique<MockOwner>(1));
    ch.drainAllNonRt([&](const MockOwner*) {});   // drain the owner
    if (ch.size() != 0)
        return false;

    // channel is reusable after drain (empty slot recycled)
    if (!ch.enqueue({7, 0, 0}, std::make_unique<MockOwner>(2)))
        return false;
    auto got = ch.take({7, 0, 0});
    return got && got->id == 2;
}

int main() {
    if (!testOwnerChannelBasicTransfer())     throw std::runtime_error("OwnerChannel basic transfer failed");
    if (!testOwnerChannelWrongKey())          throw std::runtime_error("OwnerChannel wrong-key failed");
    if (!testOwnerChannelOverwriteRejected()) throw std::runtime_error("OwnerChannel overwrite-reject failed");
    if (!testOwnerChannelLifetime())          throw std::runtime_error("OwnerChannel lifetime failed");
    if (!testOwnerChannelStress100k())        throw std::runtime_error("OwnerChannel stress 100k failed");
    if (!testOwnerChannelFullBackpressure())  throw std::runtime_error("OwnerChannel full backpressure failed");
    if (!testOwnerChannelDrainAllNonRt())     throw std::runtime_error("OwnerChannel drainAllNonRt failed");
    if (!testOwnerChannelDrainThenReenqueue()) throw std::runtime_error("OwnerChannel drain-then-reenqueue failed");
    return 0;
}
