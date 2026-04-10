//==============================================================================
// SnapshotFactory.h
// GlobalSnapshot の生成と破棄を担当する唯一の物理層
// v13.0 設計ロック準拠
//==============================================================================
#pragma once

#include "GlobalSnapshot.h"
#include "SnapshotParams.h"

namespace convo {

class DeletionQueue;      // friend 宣言のため前方宣言
class SnapshotCoordinator; // friend 宣言のため前方宣言

class SnapshotFactory {
public:
    // 新しいスナップショットを生成（Audio Thread 禁止）
    static const GlobalSnapshot* create(const SnapshotParams& params);

#ifdef _DEBUG
    static int getLiveSnapshotCount() noexcept;
#endif

private:
    friend class DeletionQueue;
    friend class SnapshotCoordinator;

    // DeletionQueue / SnapshotCoordinator のみが呼び出し可能
    static void destroy(const GlobalSnapshot* snap) noexcept;

    SnapshotFactory() = delete;
};

} // namespace convo
