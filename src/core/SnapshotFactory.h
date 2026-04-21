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

    /**
     * 差分ビルドによるスナップショット生成。
     * 現在のスナップショットと実質同一内容の場合、nullptr を返す。
     *
     * @param pending    新しいパラメータ
     * @param current    現在のスナップショット（nullptr 可）
     * @param generation 世代番号
     * @param sampleRate 現在のサンプルレート（将来の拡張用、現時点では未使用）
     */
    static const GlobalSnapshot* createImpl(
        const SnapshotParams& pending,
        const GlobalSnapshot* current,
        uint64_t generation,
        double sampleRate) noexcept;

    /**
     * 高速否定用ハッシュを計算する。
     * ここではビット厳密な値を使って高速に不一致候補を落とす。
     */
    static uint64_t computeContentHash(const SnapshotParams& params) noexcept;

    /**
     * 実質的な等価判定。
     * ハッシュ一致時の衝突回避として、許容誤差を含む厳密比較を行う。
     */
    static bool areSnapshotsEquivalent(const SnapshotParams& params,
                                       const GlobalSnapshot& snapshot) noexcept;

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
