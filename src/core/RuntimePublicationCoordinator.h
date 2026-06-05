#pragma once

#include <atomic>
#include <type_traits>
#include <utility>
#include "RuntimeTransition.h"
#include "core/RuntimeStore.h"
#include "AlignedAllocation.h"
// New components are injected from AudioEngine level (not included directly to avoid circular deps)

namespace convo {

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

    void publishWorld(convo::aligned_unique_ptr<World> worldOwner) noexcept
    {
        if (!worldOwner)
            return;

        // [PR-5] Immutable 化: publish 前に sealRecursively() で全フィールドを frozen にする
        worldOwner->sealRecursively();

        if constexpr (requires(Bridge bridge, const World& world) { bridge.validatePublicationNonRt(world); })
        {
            if (!bridge_.validatePublicationNonRt(*worldOwner))
            {
                auto* rejectedWorld = worldOwner.release();
                bridge_.retireRuntimePublishWorldNonRt(rejectedWorld, false);
                return;
            }
        }

        auto* newWorld = worldOwner.release();
        std::atomic_thread_fence(std::memory_order_release);
        auto* oldWorld = writeAccess_.publishAndSwap(newWorld);

        if constexpr (requires(Bridge bridge, const World& world) { bridge.didPublishRuntimeNonRt(world); })
        {
            bridge_.didPublishRuntimeNonRt(*newWorld);
        }

        if constexpr (requires(Bridge bridge, const World* world) { bridge.willRetireRuntimeNonRt(world); })
        {
            bridge_.willRetireRuntimeNonRt(oldWorld);
        }

        bridge_.retireRuntimePublishWorldNonRt(oldWorld, false);
    }

private:
    Bridge bridge_;
    WriteAccess writeAccess_;
    bool shutdownClearRequested_ = false;
};

} // namespace convo
