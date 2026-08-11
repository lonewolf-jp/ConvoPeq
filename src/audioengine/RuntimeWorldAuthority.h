#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include "ISRRuntimePublicationCoordinator.h"  // RuntimeIntentCoordinator
#include "ISRRetire.h"  // ★ A2: LifetimeState (sole owner of lifetime/retire state)
#include "OwnerChannel.h"               // ★ B3 (Option C): OwnerChannel holder
#include "AtomicAccess.h"               // ★ convo helpers: publishAtomic / consumeAtomic
#include "core/RuntimeStore.h"          // ★ X4-B: RuntimeStore<RuntimeState, RuntimeWorldAuthority>（CRTP 所有）
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
//   Delegate-first: wraps the existing RuntimeIntentCoordinator and forwards
//   the Authority surface (epoch / sequence / world-read / commit).
//   Per the (A) invariant, NO diagnostic / observe / metric / health APIs leak
//   through here — those remain on the coordinator's DrainAudit surface, owned by
//   RuntimeWorldAuthority only where they are StateOwner-owned (TelemetryRecorder).
//   Migration stage 1: behavior is identical (delegates to the live coordinator);
//   callers migrate in A-2/A-3 to route through this Adapter. RuntimeWorld is the
//   single source of truth for publication metadata (Epoch / Sequence / Generation).
//
// ★ work88 (X4 §6.4 — Phase 0 Authority matrix / INV-X4-1〜8・A〜C):
//   各操作の唯一の Authority を明文化する（Authority Singularization）。
//   INV-X4-1: Intent enqueue / dispatch → RuntimeIntentCoordinator only
//   INV-X4-2: Publish execution → PublishExecutor → RuntimeWorldAuthority（sole physical
//             publish gateway。Bootstrap / shutdown clear は lifecycle-controlled publish）
//   INV-X4-3: RuntimeStore::publishAndSwap() は RuntimeWorldAuthority-owned WriteAccess のみ。
//             RuntimePublishAuthority は Store / WriteAccess / publishAndSwap / 代替 authority /
//             write capability を一切所有しない（二階層化禁止）。
//   INV-X4-4: RT / Audio callback から RuntimeStateOwner / unique_ptr / WriteAccess を
//             新規取得・破棄しない（RT は観測主体・実行主体でない）
//   INV-X4-5: X4-B 後、RuntimeWorldAuthority::RuntimeStore 以外の write-capable RuntimeStore を作らない
//   INV-X4-6: publish transaction 完了後、currentWorld_ と RuntimeStore::current は同一
//             PublicationIdentity（sequenceId + publicationEpoch + mappedGeneration）
//   INV-X4-7: currentWorld_ と RuntimeStore::current を独立した authoritative source として
//             扱う API を禁止（getCurrent() を consumeWorldHandle の置換先にしない）
//   INV-X4-8: currentWorld_ = metadata observation alias（non-owning）/ RuntimeStore::current =
//             physical publication source。交換可能として扱う API を禁止
//   INV-X4-A: currentWorld_ is observation-only（RuntimeWorld 取得元として使わない）
//   INV-X4-B: RuntimeStore::current が唯一の物理 RuntimeWorld source
//   INV-X4-C: RT API は currentWorld_ から RuntimeWorld の ownership/lifetime を導出しない
//   ★ X4-B は write authority singularization（RuntimeWorldAuthority が Store を所有）。
//     read-source singularization（currentWorld_ 廃止）は Future（二十四次レビュー §27-C）。
//     現状は PublishExecutor が sole gateway + 一時生成 Coordinator が唯一の store-swap のため
//     INV-X4-3 は de facto 成立（X4-B で物理所有へ一本化する）。
class RuntimeWorldAuthority
{
public:
    // ★ work88 (X4-B §6.4 / 二十次・二十一次レビュー): RuntimeWorldAuthority が物理 RuntimeStore を
    //   value 所有する（write authority singularization — INV-X4-3 / INV-X4-5）。Owner = 自身（CRTP）。
    //   WriteAccess は Store より後に宣言（C++ 逆順破棄で writeAccess_ → runtimeStore_ の順に破棄 —
    //   WriteAccess が生きている間に Store が破棄されない）。RuntimeWorldAuthority 自体の move/copy は
    //   禁止（WriteAccess は Store への非所有参照を持ち、move すると参照先が分離する — 十九次レビュー）。
    using Store = convo::RuntimeStore<RuntimeState, RuntimeWorldAuthority>;
    using WriteAccess = typename Store::WriteAccess;

    explicit RuntimeWorldAuthority(RuntimeIntentCoordinator& coordinator) noexcept
        : coordinator_(coordinator)
        , runtimeStore_()
        , writeAccess_(runtimeStore_.acquireWriteAccess())   // ★ X4-B-3: Authority が Store を構築し WriteAccess を取得
        , ownerChannel_()
        , lifetime_()
        , registry_()
    {
    }

    RuntimeWorldAuthority(const RuntimeWorldAuthority&) = delete;
    RuntimeWorldAuthority& operator=(const RuntimeWorldAuthority&) = delete;
    RuntimeWorldAuthority(RuntimeWorldAuthority&&) = delete;
    RuntimeWorldAuthority& operator=(RuntimeWorldAuthority&&) = delete;

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

    // ── ★ work88 (X4-B §6.4 / X4-B-9): read API — RuntimeStore::current が唯一の物理
    //   RuntimeWorld source（INV-X4-B）。getCurrent() は置換先にしない（INV-X4-7 —
    //   currentWorld_ は metadata observation alias）。ReadToken は opaque トークン。
    struct ReadToken
    {
    private:
        friend class RuntimeWorldAuthority;
        constexpr ReadToken() noexcept = default;
    };

    [[nodiscard]] ReadToken acquireReadToken() const noexcept { return ReadToken{}; }

    [[nodiscard]] const RuntimeState* consumeWorldHandle(const ReadToken&) const noexcept
    {
        return runtimeStore_.observe();
    }

    [[nodiscard]] const RuntimeState* consumeWorldHandle() const noexcept
    {
        return runtimeStore_.observe();
    }

    [[nodiscard]] const RuntimeState* observePublishedWorld() const noexcept
    {
        return runtimeStore_.observe();
    }

    // ── ★ work88 (X4-B §6.4 / X4-B-4): publish — semantic publication transaction の唯一の
    //   execution boundary（commit + publishAndSwap を束ねる・commit-before-swap ordering — Test 7）。
    //   retire は publish() に戻さない（Lifetime の責務 — X3/retire と分離）。戻り値は previous
    //   （oldWorld）— retire 対象を caller（PublishExecutor / Bootstrap）に明示する。
    //   ★ seal / validate は caller が RuntimeState 完全型（AudioEngine.h）で実行する
    //     （RuntimeWorldAuthority.h は RuntimeState を前方宣言のみ — 循環 include 回避）。
    //   ★ seal-before-bake ordering（監査軽微指摘3）: caller の sealRecursively() の後に commit
    //     内で publication metadata を bake（pubWorld->publication 書込み）する。seal は RT reader
    //     向けの論理的不変性フラグ（メモリ保護ではない）ため bake は問題なく実行でき、かつ
    //     bake → publishAndSwap の順序により bake 完了前に world が観測されることはない。
    struct PublishMetadata
    {
        RuntimeBoundary boundary;
        std::uint64_t version;
        PublicationSequenceId sequenceId;
        PublicationEpoch epoch;
        std::uint64_t mappedGeneration;
    };

    [[nodiscard]] RuntimeState* publish(RuntimeOwner&& owner, const PublishMetadata& metadata,
                                        bool* committed = nullptr) noexcept
    {
        if (!owner)
        {
            if (committed != nullptr) *committed = false;
            return nullptr;                                 // validate: owner null → Failed
        }
        const auto* newWorld = owner.get();
        if (newWorld == nullptr)
        {
            if (committed != nullptr) *committed = false;
            return nullptr;                                 // validate: null world → Failed
        }
        // ★ work88（監査軽微指摘2）: seqId==0 は producer が保証しない（構造的に到達不能）。
        //   Debug で即時検出。失敗時は owner を無条件消費（破棄）して nullptr を返すため、
        //   caller は committed=false を確認して *newWorld を deref してはならない
        //   （戻り値 nullptr は「初回 publish の oldWorld」と曖昧 — dangling deref 防止）。
        assert(metadata.sequenceId != 0);
        if (metadata.sequenceId == 0)
        {
            if (committed != nullptr) *committed = false;
            return nullptr;                                 // validate: metadata invalid → Rejected
        }
        // commit metadata（ISR currentWorld_ 更新 + publication bake）— commit-before-swap（Test 7）
        coordinator_.commit(PublishAuthority::Granted, metadata.boundary, newWorld, metadata.version,
                            metadata.sequenceId, metadata.epoch, metadata.mappedGeneration);
        // ★ work88（監査軽微指摘2）: commit が Faulted（monotonicity violation 等）なら swap しない。
        //   FIFO producer により到達不能だが、commit 失敗後の物理 swap による currentWorld_
        //   （metadata alias）と Store::current の不一致を構造的に排除する（transaction 原子性）。
        if (coordinator_.getState() == RuntimeIntentCoordinator::CoordinatorState::Faulted)
        {
            if (committed != nullptr) *committed = false;
            return nullptr;
        }
        // physical store swap — 唯一の WriteAccess（INV-X4-3）
        auto* next = const_cast<RuntimeState*>(owner.release());
        std::atomic_thread_fence(std::memory_order_release);
        auto* oldWorld = writeAccess_.publishAndSwap(next); // previous（oldWorld）を caller へ返す
        if (committed != nullptr) *committed = true;
        return oldWorld;
    }

    // ── ★ work88 (X4-B §6.4 / X4-B-7): shutdown clear — publish() と統合しない（null 公開 =
    //   world クリアは別 semantic・二十次レビュー §15）。戻り値の oldWorld は caller が retire する。
    void requestShutdownClearNonRt() noexcept { shutdownClearRequested_ = true; }

    [[nodiscard]] RuntimeState* clearPublishedRuntimeSnapshotsNonRt() noexcept
    {
        if (!shutdownClearRequested_)
            return nullptr;
        shutdownClearRequested_ = false;
        return writeAccess_.publishAndSwap(nullptr);
    }

private:
    // ★ member order（二十一次レビュー①・確定）: runtimeStore_ を writeAccess_ より先に宣言。
    //   C++ 逆順破棄により writeAccess_ → runtimeStore_ の順で破棄（WriteAccess が生きている間に
    //   Store が破棄されない）。
    RuntimeIntentCoordinator& coordinator_;   // commit metadata（ISR currentWorld_）委譲先
    Store runtimeStore_;                       // ★ X4-B-2: Authority が物理 Store を所有
    WriteAccess writeAccess_;                  // ★ X4-B-3: Store の唯一の WriteAccess
    OwnerChannelType ownerChannel_;            // ★ B3 (Option C): owned by value here.
    LifetimeState lifetime_;
    PendingPublishRegistry registry_;
    bool shutdownClearRequested_ = false;
};

// ★ work88 (X4-B §6.4 / 十九次レビュー): Authority の move/copy 禁止をコンパイル時固定。
//   WriteAccess は Store への非所有参照を持ち、move すると参照先が分離する（Store は
//   non-copyable/non-movable）。static_assert はクラス完了後に置く。
static_assert(!std::is_copy_constructible_v<RuntimeWorldAuthority>,
    "RuntimeWorldAuthority must not be copy-constructible (owns RuntimeStore/WriteAccess)");
static_assert(!std::is_copy_assignable_v<RuntimeWorldAuthority>,
    "RuntimeWorldAuthority must not be copy-assignable");
static_assert(!std::is_move_constructible_v<RuntimeWorldAuthority>,
    "RuntimeWorldAuthority must not be move-constructible");
static_assert(!std::is_move_assignable_v<RuntimeWorldAuthority>,
    "RuntimeWorldAuthority must not be move-assignable");

} // namespace convo::isr
