//==============================================================================
// SnapshotSlotStore.h
// current/target スナップショットの atomic ポインタペアを保持する。
// all access is via project-approved atomic wrappers (AtomicAccess.h).
//==============================================================================
#pragma once

#include <atomic>
#include "GlobalSnapshot.h"
#include "audioengine/AtomicAccess.h"

namespace convo {

/// current/target スナップショットの atomic ポインタペアを保持する純粋ストレージ型。
/// メモリオーダーは呼び出し元が指定し、HB コメントも呼び出し元が記述する。
/// コピー・ムーブ不可（std::atomic を内包するため）。
class SnapshotSlotStore {
public:
    SnapshotSlotStore() noexcept = default;
    ~SnapshotSlotStore() noexcept = default;

    SnapshotSlotStore(const SnapshotSlotStore&) = delete;
    SnapshotSlotStore& operator=(const SnapshotSlotStore&) = delete;
    SnapshotSlotStore(SnapshotSlotStore&&) = delete;
    SnapshotSlotStore& operator=(SnapshotSlotStore&&) = delete;

    /// 初期値 nullptr を release で公開する（SnapshotCoordinator コンストラクタ専用）。
    void initializeSlots() noexcept
    {
        // release ×2: current/target の初期化完了後に他スレッドが acquire で安全に観測できるよう HB を形成する。
        convo::publishAtomic(m_current, nullptr, std::memory_order_release);
        convo::publishAtomic(m_target,  nullptr, std::memory_order_release);
    }

    GlobalSnapshot* loadCurrent(std::memory_order order) const noexcept
    {
        return convo::consumeAtomic(m_current, order);
    }

    GlobalSnapshot* loadTarget(std::memory_order order) const noexcept
    {
        return convo::consumeAtomic(m_target, order);
    }

    GlobalSnapshot* exchangeCurrent(GlobalSnapshot* newVal, std::memory_order order) noexcept
    {
        return convo::exchangeAtomic(m_current, newVal, order);
    }

    GlobalSnapshot* exchangeTarget(GlobalSnapshot* newVal, std::memory_order order) noexcept
    {
        return convo::exchangeAtomic(m_target, newVal, order);
    }

private:
    std::atomic<GlobalSnapshot*> m_current{nullptr};
    std::atomic<GlobalSnapshot*> m_target{nullptr};
};

} // namespace convo
