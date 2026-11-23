#pragma once

// ★ B-07: 学習器の可変状態（optimizer）を学習スレッドに所有させ、
//   他のスレッドからは「復旧要求」と「公開スナップショット」だけをやり取りする。
//
//   契約:
//     * enqueueRestoreIfRunning() は任意スレッドから呼べる。学習スレッドが
//       動作中のときだけ最新の 1 件を保持し、takePendingRestore() が世代境界で
//       1 回だけ取り出す。復旧は「最後に来た要求が勝つ」。
//     * 停止中の適用は呼出スレッドが直接行う。競合しないことを保証するため、
//       startLearning() は同じ mutex を取って Idle -> Starting を遷移させる。
//     * publishSnapshot()/readSnapshot() はスナップショットの受け渡しだけを行い、
//       optimizer の内部状態は決して共有しない。
//
//   このクラスは JUCE に依存しない（単体テストで実際に複数スレッドから叩ける）。

#include <cstdint>
#include <mutex>
#include <optional>

namespace convo
{

template <typename StateT>
class LearnerStateAuthority
{
public:
    // 呼出側が停止中と判断したときの即時適用を、開始遷移と排他するためのロック。
    // コールバックの中でこのクラスの他のメソッドを呼ぶとデッドロックするため、
    // 呼出側は「状態の適用」だけを行うこと。
    template <typename Fn>
    void withExclusive (Fn&& fn)
    {
        const std::lock_guard<std::mutex> lock (mutex_);
        fn();
    }

    // 学習スレッドが動作中のときだけ、最新の復旧要求として保持する。
    // 戻り値: true = 保持した（学習スレッドが境界で適用する）、false = 停止中。
    bool enqueueRestoreIfRunning (const StateT& state, bool workerActive)
    {
        const std::lock_guard<std::mutex> lock (mutex_);
        if (! workerActive)
            return false;

        pending_ = state;
        hasPending_ = true;
        ++submitted_;
        return true;
    }

    // withExclusive() の中で使う版（ロックを二重に取らない）。
    void enqueueRestoreLocked (const StateT& state)
    {
        pending_ = state;
        hasPending_ = true;
        ++submitted_;
    }

    // 学習スレッドの世代境界で呼ぶ。保持している復旧要求を 1 回だけ取り出す。
    bool takePendingRestore (StateT& out)
    {
        const std::lock_guard<std::mutex> lock (mutex_);
        if (! hasPending_)
            return false;

        out = *pending_;
        pending_.reset();
        hasPending_ = false;
        ++applied_;
        return true;
    }

    // 公開スナップショット。UI とメッセージスレッドはこれだけを読む。
    void publishSnapshot (const StateT& state)
    {
        const std::lock_guard<std::mutex> lock (mutex_);
        snapshot_ = state;
        hasSnapshot_ = true;
    }

    bool readSnapshot (StateT& out) const
    {
        const std::lock_guard<std::mutex> lock (mutex_);
        if (! hasSnapshot_)
            return false;

        out = snapshot_;
        return true;
    }

    void clearPending()
    {
        const std::lock_guard<std::mutex> lock (mutex_);
        pending_.reset();
        hasPending_ = false;
    }

    bool hasPendingRestore() const
    {
        const std::lock_guard<std::mutex> lock (mutex_);
        return hasPending_;
    }

    std::uint64_t submittedCount() const
    {
        const std::lock_guard<std::mutex> lock (mutex_);
        return submitted_;
    }

    std::uint64_t appliedCount() const
    {
        const std::lock_guard<std::mutex> lock (mutex_);
        return applied_;
    }

private:
    mutable std::mutex mutex_;
    std::optional<StateT> pending_;
    bool hasPending_ = false;
    StateT snapshot_ {};
    bool hasSnapshot_ = false;
    std::uint64_t submitted_ = 0;
    std::uint64_t applied_ = 0;
};

} // namespace convo
