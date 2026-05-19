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

    [[nodiscard]] static RuntimePublicationCoordinator create(Bridge bridge,
                                                              Store& store) noexcept
    {
        return RuntimePublicationCoordinator { std::move(bridge), store.acquireWriteAccess() };
    }

    void clearPublishedRuntimeSnapshotsNonRt() noexcept
    {
        auto* world = writeAccess_.publishAndSwap(nullptr);
        bridge_.retireRuntimePublishWorldNonRt(world);
        bridge_.resetRuntimeGraphRevisionNonRt();
    }

    void publishState(Handle current,
                      Handle next,
                      convo::TransitionPolicy policy,
                      double fadeTimeSec,
                      bool active) noexcept
    {
        auto worldOwner = bridge_.buildWorldAndPublishTransition(current,
                                                                 next,
                                                                 policy,
                                                                 fadeTimeSec,
                                                                 active);
        auto* newWorld = worldOwner.release();
        std::atomic_thread_fence(std::memory_order_release);
        auto* oldWorld = writeAccess_.publishAndSwap(newWorld);
        bridge_.retireRuntimePublishWorldNonRt(oldWorld);
    }

    void adoptAndPublish(Handle newCurrent,
                         Handle transitionNext,
                         convo::TransitionPolicy policy,
                         double fadeTimeSec,
                         bool active) noexcept
    {
        auto worldOwner = bridge_.adoptAndBuildPublishWorld(newCurrent,
                                                            transitionNext,
                                                            policy,
                                                            fadeTimeSec,
                                                            active);
        auto* newWorld = worldOwner.release();
        std::atomic_thread_fence(std::memory_order_release);
        auto* oldWorld = writeAccess_.publishAndSwap(newWorld);
        bridge_.retireRuntimePublishWorldNonRt(oldWorld);
    }

private:
    Bridge bridge_;
    WriteAccess writeAccess_;
};

} // namespace convo
