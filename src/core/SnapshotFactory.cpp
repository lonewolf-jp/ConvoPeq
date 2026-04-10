//==============================================================================
// SnapshotFactory.cpp
//==============================================================================
#include "SnapshotFactory.h"
#include <atomic>
#include <cassert>

namespace convo {

namespace {
#ifdef _DEBUG
    std::atomic<int> g_liveSnapshotCount{0};
    std::atomic<uint64_t> g_nextSnapshotId{1};
#endif
}

const GlobalSnapshot* SnapshotFactory::create(const SnapshotParams& params)
{
#ifdef _DEBUG
    const uint64_t snapId = g_nextSnapshotId.fetch_add(1, std::memory_order_relaxed);
#endif

    GlobalSnapshot* snap = new GlobalSnapshot(params);

#ifdef _DEBUG
    const_cast<GlobalSnapshot*>(snap)->snapshotId = snapId;
    const_cast<GlobalSnapshot*>(snap)->alive.store(true, std::memory_order_release);
    g_liveSnapshotCount.fetch_add(1, std::memory_order_relaxed);
#endif

    return snap;
}

void SnapshotFactory::destroy(const GlobalSnapshot* snap) noexcept
{
    if (!snap) return;

#ifdef _DEBUG
    const_cast<GlobalSnapshot*>(snap)->alive.store(false, std::memory_order_release);
    g_liveSnapshotCount.fetch_sub(1, std::memory_order_relaxed);
#endif

    delete snap;
}

#ifdef _DEBUG
int SnapshotFactory::getLiveSnapshotCount() noexcept
{
    return g_liveSnapshotCount.load(std::memory_order_relaxed);
}
#endif

} // namespace convo
