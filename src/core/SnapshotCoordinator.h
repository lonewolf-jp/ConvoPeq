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
#include "EpochCore.h"
#include "SnapshotFactory.h"

#include "audioengine/AtomicAccess.h"

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
    explicit SnapshotCoordinator(EpochCore& epochCore) noexcept
        : m_epochCore(epochCore), m_current(nullptr)
    {
        convo::publishAtomic(m_current, nullptr, std::memory_order_release);
        convo::publishAtomic(m_target, nullptr, std::memory_order_release);
        convo::publishAtomic(m_fadeAlpha, 1.0, std::memory_order_release);
        convo::publishAtomic(m_fadeState, FadeState::Idle, std::memory_order_release);
        convo::publishAtomic(m_fadeTotalSamples, 0, std::memory_order_release);
        convo::publishAtomic(m_fadeRemainingSamples, 0, std::memory_order_release);
        convo::publishAtomic(m_fadeCompleted, false, std::memory_order_release);
    }

    ~SnapshotCoordinator() noexcept {
        GlobalSnapshot* snap = convo::consumeAtomic(m_current, std::memory_order_acquire);
        if (snap)
            SnapshotFactory::destroy(snap);
        snap = convo::consumeAtomic(m_target, std::memory_order_acquire);
        if (snap)
            SnapshotFactory::destroy(snap);
    }

    GlobalSnapshot* getCurrent() const noexcept {
        return convo::consumeAtomic(m_current, std::memory_order_acquire);
    }

    void switchImmediate(GlobalSnapshot* newSnap) noexcept {
        abortFade();
        GlobalSnapshot* oldSnap = convo::exchangeAtomic(m_current, newSnap, std::memory_order_release);
        if (oldSnap) {
            uint64_t newEpoch = m_epochCore.publish();
            m_deletionQueue.enqueue(
                oldSnap,
                [](void* ptr) { SnapshotFactory::destroy(static_cast<GlobalSnapshot*>(ptr)); },
                newEpoch,
                DeletionEntryType::Generic
            );
        }
    }

    void startFade(GlobalSnapshot* target, int fadeSamples) noexcept;

    bool updateFade(float& outAlpha,
                    const GlobalSnapshot*& outCurrent,
                    const GlobalSnapshot*& outTarget) noexcept
    {
        const FadeState state = convo::consumeAtomic(m_fadeState, std::memory_order_acquire);
        if (state == FadeState::Idle)
        {
            outAlpha = 1.0f;
            outCurrent = convo::consumeAtomic(m_current, std::memory_order_acquire);
            outTarget = nullptr;
            return false;
        }

        outCurrent = convo::consumeAtomic(m_current, std::memory_order_acquire);
        outTarget = convo::consumeAtomic(m_target, std::memory_order_acquire);
        if (outTarget == nullptr)
        {
            abortFade();
            outAlpha = 1.0f;
            outTarget = nullptr;
            return false;
        }

        const double alpha = convo::consumeAtomic(m_fadeAlpha, std::memory_order_acquire);
        outAlpha = equalPowerSinApprox(static_cast<float>(alpha));
        return true;
    }

    void advanceFade(int numSamples) noexcept;
    bool tryCompleteFade() noexcept;

    bool isFading() const noexcept
    {
        return convo::consumeAtomic(m_fadeState, std::memory_order_acquire) != FadeState::Idle;
    }

private:
    void requestFadeCompletion() noexcept;
    void abortFade() noexcept;
    void completeFade() noexcept;

    EpochCore& m_epochCore;
    std::atomic<GlobalSnapshot*> m_current{nullptr};
    std::atomic<GlobalSnapshot*> m_target{nullptr};
    std::atomic<double> m_fadeAlpha{1.0};
    std::atomic<FadeState> m_fadeState{FadeState::Idle};
    std::atomic<int> m_fadeTotalSamples{0};
    std::atomic<int> m_fadeRemainingSamples{0};
    std::atomic<bool> m_fadeCompleted{false};
    DeletionQueue m_deletionQueue;
};

} // namespace convo
