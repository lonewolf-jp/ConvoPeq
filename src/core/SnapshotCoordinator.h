//==============================================================================
// SnapshotCoordinator.h
// 状態遷移の唯一の入口。atomic スナップショットポインタを管理する。
// v13.0 設計ロック準拠
//==============================================================================
#pragma once

#include <atomic>
#include "GlobalSnapshot.h"
#include "DeletionQueue.h"
#include "ReaderEpoch.h"
#include "SnapshotFactory.h"

namespace convo {

class SnapshotCoordinator {
public:
    SnapshotCoordinator() noexcept
        : m_current(nullptr)
    {
        m_current.store(nullptr, std::memory_order_relaxed);
    }

    ~SnapshotCoordinator() noexcept {
        // 最後のスナップショットを安全に破棄
        const GlobalSnapshot* snap = m_current.load(std::memory_order_acquire);
        if (snap) {
            SnapshotFactory::destroy(snap);
        }
    }

    // 現在のスナップショットを取得（RCU reader guard なしでも読み取り可能）
    const GlobalSnapshot* getCurrent() const noexcept {
        return m_current.load(std::memory_order_acquire);
    }

    // 新しいスナップショットに即座に切り替え
    void switchImmediate(const GlobalSnapshot* newSnap) noexcept {
        const GlobalSnapshot* oldSnap = m_current.exchange(newSnap, std::memory_order_release);
        if (oldSnap) {
            uint64_t newEpoch = ReaderEpoch::advanceGlobalEpoch();
            m_deletionQueue.enqueue(
                const_cast<GlobalSnapshot*>(oldSnap),
                [](void* ptr) { SnapshotFactory::destroy(static_cast<GlobalSnapshot*>(ptr)); },
                newEpoch
            );
        }
    }

    // 削除待機エントリを再利用可能のマークまでクリーンアップ
    void reclaim(uint64_t minEpoch) noexcept {
        m_deletionQueue.reclaim(minEpoch);
    }

private:
    std::atomic<const GlobalSnapshot*> m_current{nullptr};
    DeletionQueue m_deletionQueue;
};

} // namespace convo
