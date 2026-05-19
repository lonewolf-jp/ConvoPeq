//==============================================================================
// SnapshotRetireManager.h - Phase 5 (retire authority 分離)
// GlobalSnapshot の RCU 遅延解放を一元管理する唯一の retire 経路。
// v13.0 設計ロック準拠
//==============================================================================
#pragma once

#include "DeletionQueue.h"
#include "SnapshotFactory.h"

namespace convo {

/// GlobalSnapshot の RCU 遅延解放を担当する唯一の retire 経路。
/// retire() で DeletionQueue にエントリを追加し、
/// reclaim() で EpochDomain に基づく実際の解放を行う。
///
/// スレッド安全性: retire() / reclaim() はいずれも内部 mutex で保護される。
class SnapshotRetireManager {
public:
    SnapshotRetireManager() = default;
    ~SnapshotRetireManager() = default;

    // コピー・ムーブ禁止（内部 mutex 起因）
    SnapshotRetireManager(const SnapshotRetireManager&) = delete;
    SnapshotRetireManager& operator=(const SnapshotRetireManager&) = delete;
    SnapshotRetireManager(SnapshotRetireManager&&) = delete;
    SnapshotRetireManager& operator=(SnapshotRetireManager&&) = delete;

    /// @p snap を RCU retire キューへ追加する。
    /// @p snap == nullptr の場合は何もしない。
    /// @p epoch : 安全に解放できる最小 epoch (呼び出し側が epochDomain.publish()/current() で取得)
    void retire(GlobalSnapshot* snap, uint64_t epoch)
    {
        if (!snap) return;
        m_queue.enqueue(
            snap,
            [](void* ptr) { SnapshotFactory::destroy(static_cast<GlobalSnapshot*>(ptr)); },
            epoch,
            DeletionEntryType::Generic);
    }

    /// EpochDomain に基づき、retire 済みエントリのうち安全に解放可能なものを解放する。
    void reclaim(const EpochDomain& domain)
    {
        m_queue.reclaim(domain);
    }

private:
    DeletionQueue m_queue;
};

} // namespace convo
