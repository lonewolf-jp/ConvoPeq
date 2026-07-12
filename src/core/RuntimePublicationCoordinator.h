#pragma once

#include <atomic>
#include <type_traits>
#include <utility>
#include <cstdint>
#include "RuntimeTransition.h"
#include "core/RuntimeStore.h"
#include "AlignedAllocation.h"
// New components are injected from AudioEngine level (not included directly to avoid circular deps)

namespace convo {

// ★ P0-1: PublishStageResult — Coordinator が返す最小限の結果
enum class PublishStageResult : uint8_t {
    Success,
    Rejected,
    Failed
};

struct RuntimeBuildSnapshot;

template <typename World, typename Handle, typename Bridge>
class RuntimePublicationCoordinator final
{
public:
    struct ReadToken final
    {
    private:
        friend class RuntimePublicationCoordinator<World, Handle, Bridge>;
        constexpr ReadToken() noexcept = default;
    };

    using Store = RuntimeStore<World, RuntimePublicationCoordinator<World, Handle, Bridge>>;
    using WriteAccess = typename Store::WriteAccess;

    RuntimePublicationCoordinator(const RuntimePublicationCoordinator&) = delete;
    RuntimePublicationCoordinator& operator=(const RuntimePublicationCoordinator&) = delete;
    RuntimePublicationCoordinator(RuntimePublicationCoordinator&&) noexcept = default;
    RuntimePublicationCoordinator& operator=(RuntimePublicationCoordinator&&) noexcept = default;

    explicit RuntimePublicationCoordinator(Bridge&& bridge, WriteAccess&& writeAccess) noexcept
        : bridge_(std::move(bridge))
        , writeAccess_(std::move(writeAccess))
    {
    }

    [[nodiscard]] static RuntimePublicationCoordinator create(Bridge&& bridge,
                                                              Store& store) noexcept
    {
        return RuntimePublicationCoordinator { std::move(bridge), store.acquireWriteAccess() };
    }

    [[nodiscard]] static const World* consumePublishedWorld(const Store& store) noexcept
    {
        return store.observe();
    }

    [[nodiscard]] static ReadToken acquireReadToken(const Store&) noexcept
    {
        return ReadToken{};
    }

    [[nodiscard]] static const World* consumePublishedWorld(const Store& store,
                                                            const ReadToken&) noexcept
    {
        return consumePublishedWorld(store);
    }

    [[nodiscard]] static const World* consumeWorldHandle(const Store& store) noexcept
    {
        return consumePublishedWorld(store);
    }

    [[nodiscard]] static const World* consumeWorldHandle(const Store& store,
                                                         const ReadToken& token) noexcept
    {
        return consumePublishedWorld(store, token);
    }

    void clearPublishedRuntimeSnapshotsNonRt() noexcept
    {
        if (!shutdownClearRequested_)
            return;

        shutdownClearRequested_ = false;

        auto* world = writeAccess_.publishAndSwap(nullptr);
        bridge_.retireRuntimePublishWorldNonRt(world, true);
    }

    void requestShutdownClearNonRt() noexcept
    {
        shutdownClearRequested_ = true;
    }

    // ★ v8.3: const World を受け入れる — INV-11 コンパイル時保証
    //   sealRecursively() は論理的に const 操作（freeze = 不変性の確定）のため、
    //   内部で const_cast を使用。Builder から publish 完了まではこの 1 箇所のみ。
    [[nodiscard]] PublishStageResult publishWorld(convo::aligned_unique_ptr<const World> worldOwner) noexcept
    {
        if (!worldOwner)
            return PublishStageResult::Failed;

        // [PR-5] Immutable 化: publish 前に sealRecursively() で全フィールドを frozen にする
        // const_cast: Builder → Coordinator 間で唯一の非 const 操作。seal 後に不変。
        const_cast<World*>(worldOwner.get())->sealRecursively();

        if constexpr (requires(Bridge bridge, const World& world) { bridge.validatePublicationNonRt(world); })
        {
            if (!bridge_.validatePublicationNonRt(*worldOwner))
            {
                auto* rejectedWorld = const_cast<World*>(worldOwner.release());
                bridge_.retireRuntimePublishWorldNonRt(rejectedWorld, false);
                return PublishStageResult::Rejected;
            }
        }

        auto* newWorld = const_cast<World*>(worldOwner.release());
        std::atomic_thread_fence(std::memory_order_release);
        auto* oldWorld = writeAccess_.publishAndSwap(newWorld);

        if (oldWorld == nullptr && newWorld == nullptr) {
            // publishAndSwap が nullptr→nullptr の場合は異常
            return PublishStageResult::Failed;
        }

        if constexpr (requires(Bridge bridge, const World& world) { bridge.didPublishRuntimeNonRt(world); })
        {
            bridge_.didPublishRuntimeNonRt(*newWorld);
        }

        if constexpr (requires(Bridge bridge, const World* world) { bridge.willRetireRuntimeNonRt(world); })
        {
            bridge_.willRetireRuntimeNonRt(oldWorld);
        }

        bridge_.retireRuntimePublishWorldNonRt(oldWorld, false);

        return PublishStageResult::Success;
    }

private:
    Bridge bridge_;
    WriteAccess writeAccess_;
    bool shutdownClearRequested_ = false;
};

} // namespace convo
