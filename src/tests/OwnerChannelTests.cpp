// ★ B2: OwnerChannel unit tests (ADR-D3). JUCE-independent.
//   Verifies ownership transfer semantics in isolation, BEFORE any publish-path wiring
//   (B3): single-transfer, single-take, key isolation, no-overwrite, no-leak, no-double-free.
// ★ STG-11-D15-1: multi-producer regression (tests 9/10/11). The channel is SPSC by
//   design; production serializes its physical producers in the facade. These tests
//   mirror that composition (producers share a mutex, consumer never locks) and prove
//   exact ownership accounting under concurrency. Run the hammer with
//   useSerialization=false to reproduce the defect (negative control).

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <atomic>
#include <thread>
#include <mutex>
#include <vector>

#include "audioengine/OwnerChannel.h"

// Move-only mock; unique_ptr<MockOwner> never move-constructs a MockOwner (it only
// transfers the internal pointer), so deleting MockOwner's move-ctor is safe and
// catches accidental copies.
// ★ STG-11-D15-1: alive is atomic because the multi-producer tests construct and
//   destroy owners on several threads concurrently.
struct MockOwner {
    int id;
    static std::atomic<int> alive;
    explicit MockOwner(int i) : id(i) { ++alive; }
    ~MockOwner() { --alive; }
    MockOwner(const MockOwner&) = delete;
    MockOwner& operator=(const MockOwner&) = delete;
    MockOwner(MockOwner&&) = delete;
    MockOwner& operator=(MockOwner&&) = delete;
};
std::atomic<int> MockOwner::alive{ 0 };

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

// ★ STG-11-D15-1: multi-producer regression (Test A / B / C).
//
// Contract under test: physical producer threads (>1) are serialized BEFORE
// touching the channel (production: AudioEngine::enqueueRuntimePublicationFireAndForget
// under ownerChannelProducerMutex_), so the channel observes a single logical
// producer and its SPSC contract holds. These tests mirror that composition:
// producers share a test-local mutex; the consumer never locks (RT lock-free
// parity). Run the hammer with useSerialization=false to reproduce the defect
// (negative control): same-key double-accept strands an owner (alive != 0) and
// colliding probes lose key/owner correspondence.
namespace d15 {

constexpr int kHammerRounds = 2000;

struct HammerOutcome {
    int roundsWon = 0;
    bool mappingIntact = true;
};

// One hammer round: both producers enqueue the SAME key. Identical keys hash to
// the same probe start, so without serialization the two threads almost surely
// collide on the same free slot within kHammerRounds. With serialization,
// exactly one wins per round (single-thread overwrite-reject semantics).
void hammerSameKey(Channel& ch, std::mutex& producerMutex, bool useSerialization,
                   HammerOutcome& out) {
    // 8 racers released behind a ready-gate: with this many threads spinning on
    // the same atomic, at least two land inside the slot load->store window in
    // essentially every batch of rounds, so the unserialized run fails reliably
    // while the serialized run passes deterministically.
    constexpr int kRacers = 8;
    for (int round = 0; round < kHammerRounds; ++round) {
        const convo::isr::OwnerChannelKey key{ 9000, 0, 0 };
        std::atomic<int> ready{ 0 };
        std::atomic<int> go{ 0 };
        std::atomic<int> winners{ 0 };
        std::vector<std::unique_ptr<MockOwner>> losers(kRacers);
        std::vector<std::thread> threads;
        threads.reserve(kRacers);
        // Pre-create owners BEFORE the gate so allocation jitter cannot mask
        // the race window.
        std::vector<std::unique_ptr<MockOwner>> owners;
        owners.reserve(kRacers);
        for (int t = 0; t < kRacers; ++t)
            owners.push_back(std::make_unique<MockOwner>(t + 1));

        for (int t = 0; t < kRacers; ++t) {
            threads.emplace_back([&, t] {
                ready.fetch_add(1, std::memory_order_acq_rel);
                while (go.load(std::memory_order_acquire) == 0) {}
                bool accepted;
                if (useSerialization) {
                    std::lock_guard<std::mutex> lock(producerMutex);
                    accepted = ch.enqueue(key, std::move(owners[t]));
                } else {
                    accepted = ch.enqueue(key, std::move(owners[t]));
                }
                if (accepted)
                    winners.fetch_add(1, std::memory_order_acq_rel);
                else
                    losers[t] = std::move(owners[t]);
            });
        }
        while (ready.load(std::memory_order_acquire) < kRacers) {}
        go.store(1, std::memory_order_release);
        for (auto& th : threads)
            th.join();

        // Exactly one winner per round under correct serialization.
        if (winners.load(std::memory_order_acquire) != 1)
            out.mappingIntact = false;
        else
            ++out.roundsWon;
        // Drain whatever is present so the next round starts clean; stranded
        // owners (multi-accept) stay behind and trip the alive check.
        auto got = ch.take(key);
        if (winners.load(std::memory_order_acquire) == 1 && !got)
            out.mappingIntact = false;
        // losers retain their owners (destroyed at scope exit); winner above.
    }
}

} // namespace d15

// 9. Test A — multi-producer enqueue: serialized producers keep exact ownership
//    accounting and key/owner correspondence across 2000 same-key race rounds.
[[nodiscard]] bool testOwnerChannelMultiProducerSerialized() {
    MockOwner::alive = 0;
    Channel ch;
    std::mutex producerMutex;   // mirrors ownerChannelProducerMutex_ in production
    d15::HammerOutcome out{};
    d15::hammerSameKey(ch, producerMutex, /*useSerialization=*/true, out);
    if (!out.mappingIntact)
        return false;
    if (out.roundsWon != d15::kHammerRounds)
        return false;
    if (ch.size() != 0)
        return false;
    return MockOwner::alive == 0;
}

// 10. Test B — producer-side rollback: enqueue then take-back (intent-queue-full
//     rollback shape) recovers caller ownership with nothing stranded.
[[nodiscard]] bool testOwnerChannelProducerRollback() {
    MockOwner::alive = 0;
    Channel ch;
    const convo::isr::OwnerChannelKey key{ 4242, 3, 9 };
    if (!ch.enqueue(key, std::make_unique<MockOwner>(77)))
        return false;
    // Rollback: the producer takes back its own (unique) key.
    auto recovered = ch.take(key);
    if (!recovered || recovered->id != 77)
        return false;
    if (ch.size() != 0)
        return false;
    recovered.reset();
    if (MockOwner::alive != 0)
        return false;
    // Rollback of an absent key is a no-op returning nullptr.
    auto absent = ch.take(key);
    return absent == nullptr;
}

// 11. Test C — producers + consumer concurrency: 2 serialized producers and one
//     lock-free consumer deliver every owner exactly once with ids intact.
[[nodiscard]] bool testOwnerChannelProducersConsumer() {
    MockOwner::alive = 0;
    Channel ch;
    std::mutex producerMutex;   // producers serialized; consumer never locks
    constexpr int kItemsPerProducer = 500;
    std::atomic<bool> abortFlag{ false };
    std::atomic<int> delivered{ 0 };
    std::atomic<bool> mismatch{ false };

    auto producer = [&](int baseId) {
        for (int i = 0; i < kItemsPerProducer && !abortFlag.load(std::memory_order_acquire); ++i) {
            const convo::isr::OwnerChannelKey key{
                static_cast<std::uint64_t>(baseId + i), 1, 7 };
            auto owner = std::make_unique<MockOwner>(baseId + i);
            bool accepted = false;
            int attempts = 0;
            while (!accepted && !abortFlag.load(std::memory_order_acquire)
                   && attempts < 5000000) {
                {
                    std::lock_guard<std::mutex> lock(producerMutex);
                    accepted = ch.enqueue(key, std::move(owner));
                }
                if (!accepted) {
                    // Slot still held by an undrained owner (consumer is behind):
                    // yield and retry with a fresh owner. The rejected owner is
                    // destroyed here, so alive accounting stays exact.
                    owner = std::make_unique<MockOwner>(baseId + i);
                    ++attempts;
                    std::this_thread::yield();
                }
            }
            if (!accepted) {
                mismatch.store(true, std::memory_order_release);
                abortFlag.store(true, std::memory_order_release);
                return;
            }
        }
    };

    std::thread consumer([&] {
        int wantA = 1000000;
        int wantB = 2000000;
        const int total = 2 * kItemsPerProducer;
        int spins = 0;
        while (delivered.load(std::memory_order_acquire) < total && spins < 20000000
               && !abortFlag.load(std::memory_order_acquire)) {
            bool progress = false;
            for (int base : { 1000000, 2000000 }) {
                int& want = (base == 1000000) ? wantA : wantB;
                if (want >= base + kItemsPerProducer)
                    continue;
                const convo::isr::OwnerChannelKey key{
                    static_cast<std::uint64_t>(want), 1, 7 };
                auto got = ch.take(key);   // lock-free, as on the RT consumer path
                if (got) {
                    if (got->id != want)
                        mismatch.store(true, std::memory_order_release);
                    ++want;
                    delivered.fetch_add(1, std::memory_order_acq_rel);
                    progress = true;
                }
            }
            if (!progress) {
                ++spins;
                std::this_thread::yield();
            }
        }
        if (delivered.load(std::memory_order_acquire) != 2 * kItemsPerProducer)
            abortFlag.store(true, std::memory_order_release);   // let producers stop
    });

    std::thread pA(producer, 1000000);
    std::thread pB(producer, 2000000);
    pA.join();
    pB.join();
    consumer.join();

    if (abortFlag.load(std::memory_order_acquire))
        return false;

    if (mismatch.load(std::memory_order_acquire))
        return false;
    if (delivered.load(std::memory_order_acquire) != 2 * kItemsPerProducer)
        return false;
    if (ch.size() != 0)
        return false;
    return MockOwner::alive == 0;
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
    if (!testOwnerChannelMultiProducerSerialized()) throw std::runtime_error("OwnerChannel multi-producer serialized failed");
    if (!testOwnerChannelProducerRollback()) throw std::runtime_error("OwnerChannel producer rollback failed");
    if (!testOwnerChannelProducersConsumer()) throw std::runtime_error("OwnerChannel producers-consumer failed");
    return 0;
}
