#pragma once

#include <atomic>
#include <type_traits>
#include <utility>

#include "AlignedAllocation.h"
#include "RuntimeTransition.h"
#include "core/RuntimeStore.h"

namespace convo {

template <typename World, typename Handle, typename Bridge>
class RuntimePublicationCoordinator final
{
public:
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

    [[nodiscard]] static const World* observePublishedWorld(const Store& store) noexcept
    {
        return store.observe();
    }

    void clearPublishedRuntimeSnapshotsNonRt() noexcept
    {
        auto* world = writeAccess_.publishAndSwap(nullptr);
        bridge_.retireRuntimePublishWorldNonRt(world, true);
    }

    void publishState(Handle current,
                      Handle next,
                      convo::TransitionPolicy policy,
                      double fadeTimeSec,
                      bool active) noexcept
    {
        auto worldOwner = bridge_.buildRuntimePublishWorld(current,
                                                           next,
                                                           policy,
                                                           fadeTimeSec,
                                                           active);

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
};

} // namespace convo
