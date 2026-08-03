#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include "ISRRuntimePublicationCoordinator.h"  // RuntimePublicationCoordinator
#include "ISRRetire.h"  // ★ A2: LifetimeState (sole owner of lifetime/retire state)
#include "OwnerChannel.h"               // ★ B3 (Option C): OwnerChannel holder
#include "AtomicAccess.h"               // ★ convo helpers: publishAtomic / consumeAtomic
#include "../AlignedAllocation.h"       // ★ B3: aligned_unique_ptr<const RuntimeState>

// ★ B3 (Option C): RuntimeWorldAuthority owns the OwnerChannel by value; the owner
//   is the aligned owner of the mutable RuntimeWorld (RuntimeState). Global fwd-decl
//   (FrozenRuntimeWorld.h precedent) so this header need not include AudioEngine.h
//   (which would cycle); aligned_unique_ptr<const T> only requires T* (pointer), not
//   the complete type, for a member declaration.
struct RuntimeState;

namespace convo::isr {

// ★ ADR-D3: PendingPublishRegistry — owner of newWorld during the Step 5-3 async
//   enqueue→commit lifetime gap. registerPublish() populates the gap at the enqueue
//   Producer (after releaseState, keyed on the builder-baked publication.sequenceId);
//   ISR PublishExecutor resolves it at commit via lookup() and drops the entry via
//   unregister() once currentWorld_ takes ownership. Registry ≠ Authority.
//   Lock-free (audio-thread safe): registerPublish is Non-RT; lookup/unregister may run
//   on the ISR/audio thread via ProcessIntent → PublishExecutor::executePublish.
class PendingPublishRegistry {
    static constexpr std::size_t kPendingPublishCapacity = 64;  // >> async enqueue→commit gap
    struct Entry {
        std::atomic<PublicationSequenceId> seqId{0};
        std::atomic<const void*> world{nullptr};
    };
    Entry entries_[kPendingPublishCapacity];
    std::atomic<std::size_t> cursor_{0};

public:
    void registerPublish(PublicationSequenceId seqId, const void* sealedWorld) noexcept {
        if (sealedWorld == nullptr || seqId == 0)
            return;
        const auto index = cursor_.fetch_add(1, std::memory_order_relaxed) % kPendingPublishCapacity;
        convo::publishAtomic(entries_[index].seqId, seqId, std::memory_order_release);
        convo::publishAtomic(entries_[index].world, sealedWorld, std::memory_order_release);
    }

    const void* lookup(PublicationSequenceId seqId) const noexcept {
        for (std::size_t i = 0; i < kPendingPublishCapacity; ++i) {
            const auto seq = convo::consumeAtomic(entries_[i].seqId, std::memory_order_acquire);
            if (seq == seqId) {
                auto* world = convo::consumeAtomic(entries_[i].world, std::memory_order_acquire);
                if (convo::consumeAtomic(entries_[i].seqId, std::memory_order_acquire) == seqId)
                    return world;  // unchanged → belongs to this publish
            }
        }
        return nullptr;
    }

    void unregister(PublicationSequenceId seqId) noexcept {
        for (std::size_t i = 0; i < kPendingPublishCapacity; ++i) {
            if (convo::consumeAtomic(entries_[i].seqId, std::memory_order_acquire) == seqId) {
                convo::publishAtomic(entries_[i].seqId, static_cast<PublicationSequenceId>(0), std::memory_order_release);
                convo::publishAtomic(entries_[i].world, static_cast<const void*>(nullptr), std::memory_order_release);
            }
        }
    }
};

// ★ A-1: RuntimeWorldAuthority — ISR Authority Surface (mutable API only).
//   Delegate-first: wraps the existing RuntimePublicationCoordinator and forwards
//   the Authority surface (epoch / sequence / world-read / commit).
//   Per the (A) invariant, NO diagnostic / observe / metric / health APIs leak
//   through here — those remain on the coordinator's DrainAudit surface, owned by
//   RuntimeWorldAuthority only where they are StateOwner-owned (TelemetryRecorder).
//   Migration stage 1: behavior is identical (delegates to the live coordinator);
//   callers migrate in A-2/A-3 to route through this Adapter. RuntimeWorld is the
//   single source of truth for publication metadata (Epoch / Sequence / Generation).
class RuntimeWorldAuthority
{
public:
    explicit RuntimeWorldAuthority(RuntimePublicationCoordinator& coordinator) noexcept
        : coordinator_(coordinator) {}

    // ── Authority: publication metadata (derived from currentWorld_) ──
    [[nodiscard]] PublicationEpoch currentEpoch() const noexcept
    {
        return coordinator_.currentPublicationEpoch();
    }

    [[nodiscard]] PublicationSequenceId sequence() const noexcept
    {
        return coordinator_.currentPublicationSequenceId();
    }

    // ── Authority: world snapshot source (read) ──
    [[nodiscard]] const void* getCurrent() const noexcept
    {
        return coordinator_.getCurrent();
    }

    [[nodiscard]] std::uint64_t getVersion() const noexcept
    {
        return coordinator_.getVersion();
    }

    // ── Authority: publish/commit ──
    void commit(PublishAuthority auth,
                RuntimeBoundary boundary,
                const void* newWorld,
                std::uint64_t version) noexcept
    {
        coordinator_.commit(auth, boundary, newWorld, version);
    }

    void commit(PublishAuthority auth,
                RuntimeBoundary boundary,
                const void* newWorld,
                std::uint64_t version,
                PublicationSequenceId sequenceId,
                PublicationEpoch epoch,
                std::uint64_t mappedGeneration) noexcept
    {
        coordinator_.commit(auth, boundary, newWorld, version, sequenceId, epoch, mappedGeneration);
    }

    // ★ ADR-D3: gap registry for async publish (non-owning handle; owner during enqueue→commit).
    [[nodiscard]] PendingPublishRegistry& registry() noexcept { return registry_; }
    [[nodiscard]] const PendingPublishRegistry& registry() const noexcept { return registry_; }

// ★ A2: sole owner of Lifetime/retire state (formerly AudioEngine::retireRuntime_ / retireRuntimeEx_).
    //   AudioEngine / ISR Handler reach epoch control only via worldAuthority_.lifetime().
    LifetimeState& lifetime() noexcept { return lifetime_; }
    const LifetimeState& lifetime() const noexcept { return lifetime_; }

    // ── ★ B3 (Option C): Owner channel — sole owner-transfer point across the RT boundary. ──
    //   Owned HERE by value (not by ISR coordinator / not by AudioEngine) because:
    //   (a) the owner is an aligned_unique_ptr<const RuntimeState> (complete type required
    //       only on this side), (b) the ISR coordinator header stays ignorant of RuntimeState
    //       (no circular include), (c) AudioEngine = orchestrator, not transport detail owner.
    //   Producer (enqueue, Non-RT): commitRuntimePublication → worldAuthority_.ownerChannel().
    //   Consumer (take, ISR/audio): executePublish → authority.ownerChannel().take(key).
    //   B3 invariant: take() is the sole Owner-consumption point; Owner held locally
    //   (RuntimeStateOwner) until authority.commit() success, then released.
    using RuntimeOwner = convo::aligned_unique_ptr<const RuntimeState>;
    using OwnerChannelType = convo::isr::OwnerChannel<RuntimeOwner>;

    [[nodiscard]] OwnerChannelType& ownerChannel() noexcept { return ownerChannel_; }


private:
    RuntimePublicationCoordinator& coordinator_;
    LifetimeState lifetime_;
    PendingPublishRegistry registry_;
    OwnerChannelType ownerChannel_;   // ★ B3 (Option C): owned by value here.
};

} // namespace convo::isr
