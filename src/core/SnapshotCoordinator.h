//==============================================================================
// SnapshotCoordinator.h - Phase 4 (Audio Thread safe fade)
// 状態遷移の唯一の入口。atomic スナップショットポインタを管理する。
// v13.0 設計ロック準拠
//==============================================================================
#pragma once

#include <cstdint>
#include "ObservedRuntime.h"
#include "SnapshotFadeState.h"
#include "SnapshotRetireManager.h"
#include "SnapshotSlotStore.h"

namespace convo {

// Audio Thread safe equal-power approximation (sin(pi/2*x), libm free)
static inline float equalPowerSinApprox(float x) noexcept
{
    const float t = x * 1.5707963267948966f;
    const float t2 = t * t;
    return t * (1.0f + t2 * (-1.0f / 6.0f + t2 * (1.0f / 120.0f + t2 * (-1.0f / 5040.0f + t2 * (1.0f / 362880.0f)))));
}

class SnapshotCoordinator {
public:
    explicit SnapshotCoordinator(EpochDomain& epochDomain) noexcept
        : m_epochDomain(&epochDomain)
    {
        m_slots.initializeSlots();
        m_fade.initialize();
    }

    ~SnapshotCoordinator() noexcept {
        const uint64_t retireEpoch = m_epochDomain->publish();

        // acq_rel ×2: acquire → 直前の publishNew/switchImmediate release と HB して旧ポインタ取得；
        //              release → null を公開し後続 acquire と HB してダブルフリーを防止。
        GlobalSnapshot* snap = m_slots.exchangeCurrent(nullptr, std::memory_order_acq_rel);
        m_retire.retire(snap, retireEpoch);

        // acq_rel: startFade の m_target release と HB し、target を回収して null を公開。
        snap = m_slots.exchangeTarget(nullptr, std::memory_order_acq_rel);
        m_retire.retire(snap, retireEpoch);

        m_retire.reclaim(*m_epochDomain);
    }

    ObservedRuntime observeCurrentRuntime(int readerIndex) const noexcept {
        ObservedRuntime observed(*m_epochDomain, readerIndex);
        // acquire: switchImmediate/publishNew の m_current release と HB し最新スナップを観測。
        observed.ptr = m_slots.loadCurrent(std::memory_order_acquire);
        return observed;
    }

    void switchImmediate(GlobalSnapshot* newSnap) noexcept {
        resetFadeStateAndRetireTarget();
        // release: 新スナップを公開し、observeCurrent/updateFade の acquire と HB 。
        //          旧ポインタ回収は release で十分（publishNew と同一 NonRT スレッドから呼ぶ前提）。
        GlobalSnapshot* oldSnap = m_slots.exchangeCurrent(newSnap, std::memory_order_release);
        if (oldSnap) {
            uint64_t newEpoch = m_epochDomain->publish();
            m_retire.retire(oldSnap, newEpoch);
        }
    }

    void startFade(GlobalSnapshot* target, int fadeSamples) noexcept;

    void reclaim(const EpochDomain& core) noexcept {
        m_retire.reclaim(core);
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

    EpochDomain* m_epochDomain;
    SnapshotSlotStore m_slots;
    SnapshotFadeState m_fade;
    SnapshotRetireManager m_retire;
};

} // namespace convo
