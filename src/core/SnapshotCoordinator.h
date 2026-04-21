//==============================================================================
// SnapshotCoordinator.h - Phase 4 (Audio Thread safe fade)
// 状態遷移の唯一の入口。atomic スナップショットポインタを管理する。
// v13.0 設計ロック準拠
//==============================================================================
#pragma once

#include <atomic>
#include <cstdint>
#include "GlobalSnapshot.h"
#include "DeletionQueue.h"
#include "ReaderEpoch.h"
#include "SnapshotFactory.h"

namespace convo {

enum class FadeState : uint8_t {
    Idle,
    FadingIn
};

// Audio Thread safe equal-power approximation (sin(pi/2*x), libm free)
static inline float equalPowerSinApprox(float x) noexcept
{
    const float t = x * 1.5707963267948966f;
    const float t2 = t * t;
    return t * (1.0f + t2 * (-1.0f / 6.0f + t2 * (1.0f / 120.0f + t2 * (-1.0f / 5040.0f + t2 * (1.0f / 362880.0f)))));
}

class SnapshotCoordinator {
public:
    SnapshotCoordinator() noexcept
        : m_current(nullptr)
    {
        m_current.store(nullptr, std::memory_order_relaxed);
        m_target.store(nullptr, std::memory_order_relaxed);
        m_fadeAlpha.store(1.0, std::memory_order_relaxed);
        m_fadeState.store(FadeState::Idle, std::memory_order_relaxed);
        m_fadeTotalSamples.store(0, std::memory_order_relaxed);
        m_fadeRemainingSamples.store(0, std::memory_order_relaxed);
        m_fadeCompleted.store(false, std::memory_order_relaxed);
    }

    ~SnapshotCoordinator() noexcept {
        const GlobalSnapshot* snap = m_current.load(std::memory_order_acquire);
        if (snap)
            SnapshotFactory::destroy(snap);
        snap = m_target.load(std::memory_order_acquire);
        if (snap)
            SnapshotFactory::destroy(snap);
    }

    const GlobalSnapshot* getCurrent() const noexcept {
        return m_current.load(std::memory_order_acquire);
    }

    void switchImmediate(const GlobalSnapshot* newSnap) noexcept {
        abortFade();
        const GlobalSnapshot* oldSnap = m_current.exchange(newSnap, std::memory_order_release);
        if (oldSnap) {
            uint64_t newEpoch = SnapshotEpoch::advance();
            m_deletionQueue.enqueue(
                const_cast<GlobalSnapshot*>(oldSnap),
                [](void* ptr) { SnapshotFactory::destroy(static_cast<GlobalSnapshot*>(ptr)); },
                newEpoch,
                DeletionEntryType::Generic
            );
        }
    }

    void startFade(const GlobalSnapshot* target, int fadeSamples) noexcept;

    bool updateFade(float& outAlpha,
                    const GlobalSnapshot*& outCurrent,
                    const GlobalSnapshot*& outTarget) noexcept
    {
        const FadeState state = m_fadeState.load(std::memory_order_acquire);
        if (state == FadeState::Idle)
        {
            outAlpha = 1.0f;
            outCurrent = m_current.load(std::memory_order_acquire);
            outTarget = nullptr;
            return false;
        }

        outCurrent = m_current.load(std::memory_order_acquire);
        outTarget = m_target.load(std::memory_order_acquire);
        if (outTarget == nullptr)
        {
            abortFade();
            outAlpha = 1.0f;
            outTarget = nullptr;
            return false;
        }

        const double alpha = m_fadeAlpha.load(std::memory_order_acquire);
        outAlpha = equalPowerSinApprox(static_cast<float>(alpha));
        return true;
    }

    void advanceFade(int numSamples) noexcept;
    bool tryCompleteFade() noexcept;

    void reclaim(uint64_t minEpoch) noexcept
    {
        m_deletionQueue.reclaim(minEpoch);
    }

    bool isFading() const noexcept
    {
        return m_fadeState.load(std::memory_order_acquire) != FadeState::Idle;
    }

private:
    void requestFadeCompletion() noexcept;
    void abortFade() noexcept;
    void completeFade() noexcept;

    std::atomic<const GlobalSnapshot*> m_current{nullptr};
    std::atomic<const GlobalSnapshot*> m_target{nullptr};
    std::atomic<double> m_fadeAlpha{1.0};
    std::atomic<FadeState> m_fadeState{FadeState::Idle};
    std::atomic<int> m_fadeTotalSamples{0};
    std::atomic<int> m_fadeRemainingSamples{0};
    std::atomic<bool> m_fadeCompleted{false};
    DeletionQueue m_deletionQueue;
};

} // namespace convo
