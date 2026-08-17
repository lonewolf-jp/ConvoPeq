#pragma once
// ★ B2: Lock-free SPSC owner-transfer channel (ADR-D3 Step 5-3 owner leg).
//   Single Producer (Non-RT publish thread) -> Single Consumer (ISR/audio thread).
//   Transfers SOLE OWNERSHIP of a RuntimeStateOwner across the RT boundary.
//
//   Key invariant (B2-design): key = (sequenceId, epoch, mappedGeneration), NOT sequenceId
//   alone — so future cancel / retry / overflow / replay cannot collide across attempts.
//   All three fields are already present on Intent.payload.publish + intent.sequenceId,
//   so no new data is introduced.
//
//   API is intentionally minimal (Single Transfer):
//     enqueue(key, owner&&)  -> false if key already queued or full (caller keeps owner)
//     take(key)             -> nullptr if absent/mismatch; slot drained on hit (no 2nd take)
//   No lookup/contains/peek: a slot yields its owner exactly once.
//
//   OwnerPtr must be a unique_ptr-like type whose deleter_type is stateless and
//   default-constructible (std::unique_ptr<T> or convo::aligned_unique_ptr<T>), so a
//   drained raw pointer can be re-wrapped as OwnerPtr on take().
//
//   B2 scope: publish path is NOT touched. This header is structurally standalone and
//   JUCE-independent; it is unit-tested in isolation before any publish-path wiring (B3).

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include "AtomicAccess.h"

namespace convo::isr {

struct OwnerChannelKey {
    std::uint64_t sequenceId{0};
    std::uint32_t epoch{0};
    std::uint64_t mappedGeneration{0};
    bool operator==(const OwnerChannelKey&) const noexcept = default;
};

template <class OwnerPtr>
class OwnerChannel {
    using Owner = typename OwnerPtr::element_type;
    static constexpr std::size_t kCapacity = 256;            // >> max in-flight publishes
    static constexpr std::size_t kMask = kCapacity - 1;      // power-of-two probe

    struct Slot {
        OwnerChannelKey key{};
        // non-null => slot holds an owner for `key`.
        // key is written (sequenced-before) the release-store of owner, so a single-transfer
        // consumer always observes a consistent (key, owner) pair.
        std::atomic<Owner*> owner{nullptr};
    };

    Slot slots_[kCapacity];

    static constexpr std::size_t hashOf(const OwnerChannelKey& k) noexcept {
        // Knuth mixing of the composite key -> deterministic probe start.
        std::size_t h = k.sequenceId;
        h ^= static_cast<std::size_t>(k.epoch) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= k.mappedGeneration + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }

public:
    OwnerChannel() noexcept = default;

    // Single producer. Sole ownership transfer. Returns false if `key` is already
    // queued (no overwrite) or the channel is full — caller retains ownership.
    bool enqueue(const OwnerChannelKey& key, OwnerPtr&& owner) noexcept {
        if (!owner)
            return false;
        Owner* const raw = owner.get();
        const std::size_t base = hashOf(key);
        for (std::size_t i = 0; i < kCapacity; ++i) {
            Slot& s = slots_[(base + i) & kMask];
            if (convo::consumeAtomic(s.owner, std::memory_order_acquire) != nullptr) {
                if (s.key == key)
                    return false;        // already enqueued -> reject (no overwrite)
                continue;                 // collision with a different key -> keep probing
            }
            // free slot (SPSC: sole producer on this path):
            // write key BEFORE publishing owner so the consumer sees a consistent pair.
            s.key = key;
            convo::publishAtomic(s.owner, raw, std::memory_order_release);  // publish slot as occupied
            owner.release();                                // ownership transferred to slot
            return true;
        }
        return false;                                       // channel full
    }

    // Single consumer. Claim+drain the owner for `key` exactly once (single-transfer).
    // Returns nullptr if no matching owner is present — the slot is NOT drained on a
    // key mismatch, so the owner remains takeable for its actual key.
    OwnerPtr take(const OwnerChannelKey& key) noexcept {
        Owner* raw = nullptr;
        const std::size_t base = hashOf(key);
        for (std::size_t i = 0; i < kCapacity; ++i) {
            Slot& s = slots_[(base + i) & kMask];
            Owner* const seen = convo::consumeAtomic(s.owner, std::memory_order_acquire);
            if (seen == nullptr || s.key != key)
                continue;                                   // empty or different key
            // match: single-transfer drain (SPSC: sole consumer)
            convo::publishAtomic(s.owner, static_cast<Owner*>(nullptr), std::memory_order_release);
            raw = seen;
            break;
        }
        // re-wrap the raw pointer with a default-constructed (stateless) deleter so the
        // caller receives a fully-formed owner with the correct destruction semantics.
        return OwnerPtr(raw, typename OwnerPtr::deleter_type{});
    }

    // ★ 15-P-CROSS-IMPLEMENTATION-1: GAP-CROSS-1 fix — OwnerChannel terminal drain.
    //   Drain all residual owners (Non-RT phase, producer/consumer quiescent).
    //   Ownership is *relinquished* (not released) — each raw Owner* is handed to `reclaim`
    //   which MUST transfer ownership to an existing retire authority. It must NOT delete.
    //   Caller contract: enqueue(producer) and take(consumer) MUST be quiescent — i.e.
    //   shutdown has joined the producer/consumer before this is called.
    //   Uses the same consume->publish(nullptr,release) single-transfer pattern as take()
    //   (so re-drain is a no-op: slots_ seen nullptr after the first drain).
    //   s.key is NOT reset — key-matching is irrelevant for a full scan; slot emptiness
    //   is determined by owner==nullptr (matches take()'s empty-slot check).
    template <class Fn>
    std::size_t drainAllNonRt(Fn&& reclaim) noexcept
    {
        std::size_t reclaimed = 0;
        for (std::size_t i = 0; i < kCapacity; ++i) {
            Slot& s = slots_[i];
            Owner* const raw = consumeAtomic(s.owner, std::memory_order_acquire);
            if (raw != nullptr) {
                publishAtomic(s.owner, static_cast<Owner*>(nullptr),
                              std::memory_order_release);   // single-transfer (same as take)
                reclaim(raw);
                ++reclaimed;
            }
        }
        return reclaimed;
    }

    // B2 diagnostic only (not on the publish hot path): occupied-slot count.
    std::size_t size() const noexcept {
        std::size_t n = 0;
        for (const auto& s : slots_)
            if (convo::consumeAtomic(s.owner, std::memory_order_acquire) != nullptr)
                ++n;
        return n;
    }
};

} // namespace convo::isr
