//==============================================================================
// SnapshotCoordinator.h - Phase 4 (Audio Thread safe fade)
// 状態遷移の唯一の入口。atomic スナップショットポインタを管理する。
// v13.0 設計ロック準拠
//
// [work21 P1-7] 本クラスは IEpochProvider 経由で EBR操作を行う。
// publish/retire/reclaim は provider (通常 ISRRetireRouter) に委譲。
//==============================================================================
#pragma once

#include <cstdint>
#include "ObservedRuntime.h"
#include "SnapshotFadeState.h"
#include "SnapshotSlotStore.h"
#include "IEpochProvider.h"
#include "SnapshotFactory.h"

namespace convo {

// P0-1 ObserveToken formalization:
// - SnapshotCoordinator は observe enter/exit をトークン化して返す責務のみを持つ。
// - 実体型は ObservedRuntime（ObserveToken の互換エイリアス）を使用する。
// - publish / retire / graph mutation / ownership 管理は本APIの責務外。

// Audio Thread safe equal-power approximation (sin(pi/2*x), libm free)
static inline float equalPowerSinApprox(float x) noexcept
{
    const float t = x * 1.5707963267948966f;
    const float t2 = t * t;
    return t * (1.0f + t2 * (-1.0f / 6.0f + t2 * (1.0f / 120.0f + t2 * (-1.0f / 5040.0f + t2 * (1.0f / 362880.0f)))));
}

class SnapshotCoordinator {
public:
    // [work21 P1-7] IEpochProvider 経由でEBR操作 (EpochDomain型非依存)
    explicit SnapshotCoordinator(IEpochProvider& provider) noexcept
        : m_epochProvider(&provider)
    {
        m_slots.initializeSlots();
        m_fade.initialize();
    }

    ~SnapshotCoordinator() noexcept {
        constexpr auto snapshotDeleter = [](void* ptr) noexcept
        {
            SnapshotFactory::destroy(static_cast<GlobalSnapshot*>(ptr));
        };

        const uint64_t retireEpoch = m_epochProvider->publishEpoch();

        GlobalSnapshot* snap = m_slots.exchangeCurrent(nullptr, std::memory_order_acq_rel);
        if (snap)
        {
            m_epochProvider->enqueueRetire(snap, snapshotDeleter, retireEpoch);
        }

        snap = m_slots.exchangeTarget(nullptr, std::memory_order_acq_rel);
        if (snap)
        {
            m_epochProvider->enqueueRetire(snap, snapshotDeleter, retireEpoch);
        }

        m_epochProvider->tryReclaim();
    }

    // observeCurrentRuntime:
    // - reader guard を保持した ObserveToken 相当（ObservedRuntime）を返す。
    // - 呼び出し側は本トークン寿命内で snapshot を参照する。
    ObservedRuntime observeCurrentRuntime(RCUReader& reader) const noexcept {
        ObservedRuntime observed(reader);
        // acquire: switchImmediate/publishNew の m_current release と HB し最新スナップを観測。
        observed.ptr = m_slots.loadCurrent(std::memory_order_acquire);
        return observed;
    }

    void switchImmediate(GlobalSnapshot* newSnap) noexcept {
        constexpr auto snapshotDeleter = [](void* ptr) noexcept
        {
            SnapshotFactory::destroy(static_cast<GlobalSnapshot*>(ptr));
        };

        resetFadeStateAndRetireTarget();
        // release: 新スナップを公開し、observeCurrentRuntime/updateFade の acquire と HB 。
        //          旧ポインタ回収は release で十分（publishNew と同一 NonRT スレッドから呼ぶ前提）。
        GlobalSnapshot* oldSnap = m_slots.exchangeCurrent(newSnap, std::memory_order_release);
        if (oldSnap) {
            uint64_t newEpoch = m_epochProvider->publishEpoch();
            m_epochProvider->enqueueRetire(oldSnap, snapshotDeleter, newEpoch);
        }
    }

    void startFade(GlobalSnapshot* target, int fadeSamples) noexcept;

    // [P1-21] epoch-free: uint64_t epoch を受け取るように変更
    void reclaim(uint64_t) noexcept {
        m_epochProvider->tryReclaim();
    }

    bool updateFade(float& outAlpha,
                    const GlobalSnapshot*& outCurrent,
                    const GlobalSnapshot*& outTarget) noexcept
    {
        // acquire: SnapshotFadeState::start/resetToIdle の release と HB してフェード状態を観測。
        const FadeState state = m_fade.state();
        if (state == FadeState::Idle)
        {
            outAlpha = 1.0f;
            // acquire: switchImmediate/publishNew release と HB し、最新スナップを観測。
            outCurrent = m_slots.loadCurrent(std::memory_order_acquire);
            outTarget = nullptr;
            return false;
        }

        // acquire ×2: m_current/m_target の release と HB し、フェード中の両スナップを観測。
        outCurrent = m_slots.loadCurrent(std::memory_order_acquire);
        outTarget = m_slots.loadTarget(std::memory_order_acquire);
        if (outTarget == nullptr)
        {
            resetFadeStateAndRetireTarget();
            outAlpha = 1.0f;
            outTarget = nullptr;
            return false;
        }

        // acquire: advanceFade/reset/complete の alpha release と HB し最新アルファ値を観測。
        const double alpha = m_fade.alpha();
        outAlpha = equalPowerSinApprox(static_cast<float>(alpha));
        return true;
    }

    void advanceFade(int numSamples) noexcept;
    bool tryCompleteFade() noexcept;

    bool isFading() const noexcept
    {
        return m_fade.isFading();
    }

private:
    void resetFadeStateAndRetireTarget() noexcept;
    void completeFade() noexcept;

    IEpochProvider* m_epochProvider;
    SnapshotSlotStore m_slots;
    SnapshotFadeState m_fade;
};

} // namespace convo
