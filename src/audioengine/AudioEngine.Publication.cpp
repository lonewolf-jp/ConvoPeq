#include <JuceHeader.h>
#include "AudioEngine.h"
#include "CrossfadeAuthority.h"  // ★ CrossfadePolicy 完全型（makeCrossfadePolicy）

//==============================================================================
// [P0-15] Publication PR: Epoch publication operations
//         (publish / current / advanceEpoch) → Router::publishEpoch()
//         Part of AudioEngine.Threading.cpp 3-way split.
//==============================================================================

[[nodiscard]] uint64_t AudioEngine::snapshotRcuEpoch() noexcept
{
    return currentRetireEpoch();
}

[[nodiscard]] uint64_t AudioEngine::markRetireEpoch() noexcept
{
    return m_retireRouter->publishEpoch();
}

[[nodiscard]] uint64_t AudioEngine::currentRetireEpoch() const noexcept
{
    return m_retireRouter->currentEpoch();
}

uint64_t AudioEngine::advanceRetireEpoch() noexcept
{
    return m_retireRouter->publishEpoch();
}

// ★ Phase-2: NonRT → Policy 生成（7個の atomic を acquire で一括読み取り）
[[nodiscard]] convo::isr::CrossfadePolicy AudioEngine::makeCrossfadePolicy() const noexcept
{
    convo::isr::CrossfadePolicy p;
    p.irFadeTimeSec       = convo::consumeAtomic(m_irFadeTimeSec,       std::memory_order_acquire);
    p.phaseFadeTimeSec    = convo::consumeAtomic(m_phaseFadeTimeSec,    std::memory_order_acquire);
    p.tailFadeTimeSec     = convo::consumeAtomic(m_tailFadeTimeSec,     std::memory_order_acquire);
    p.osFadeTimeSec       = convo::consumeAtomic(m_osFadeTimeSec,       std::memory_order_acquire);
    p.irLengthFadeTimeSec = convo::consumeAtomic(m_irLengthFadeTimeSec, std::memory_order_acquire);
    p.directHeadFadeTimeSec = convo::consumeAtomic(m_directHeadFadeTimeSec, std::memory_order_acquire);
    p.nucFilterFadeTimeSec  = convo::consumeAtomic(m_nucFilterFadeTimeSec,  std::memory_order_acquire);
    // ★ HealthState は Policy に入れない — Orchestrator または DSPTransition が判断する
    return p;
}

// ★ B4: Producer 共通の Decision snapshot 生成（Decision + Handle を一括生成）。
//   oldHandle == null（idle publish #4/#5/#6）は crossfade 判定をスキップし old DSP retire 意図なし。
//   Rebuild (#7) のみ current active DSP handle を渡す（old DSP を retire する意図）。
//   判定ロジックは Orchestrator の 3-step (evaluate → null fallback → Critical 抑制) と同一。
[[nodiscard]] convo::isr::RuntimeIntentCoordinator::PublishDecisionSnapshot AudioEngine::makePublishDecisionSnapshot(
    const RuntimePublishWorld* newWorld,
    const convo::isr::DSPHandle& newHandle,
    const convo::isr::DSPHandle& oldHandle) const noexcept
{
    convo::isr::RuntimeIntentCoordinator::PublishDecisionSnapshot snapshot;
    snapshot.newHandle = newHandle;
    snapshot.oldHandle = oldHandle;

    if (newWorld == nullptr || oldHandle.isNull())
    {
        snapshot.needsCrossfade = false;
        snapshot.fadeTimeSec = 0.0;
        snapshot.oldHasIR = false;
        snapshot.newHasIR = (newWorld != nullptr) ? newWorld->dspProjection.irLoaded : false;
        return snapshot;
    }

    const auto* oldWorld = observePublishedWorld();
    if (oldWorld == nullptr)
    {
        snapshot.needsCrossfade = false;
        snapshot.fadeTimeSec = 0.0;
        snapshot.oldHasIR = false;
        snapshot.newHasIR = newWorld->dspProjection.irLoaded;
    }
    else
    {
        convo::isr::CrossfadeAuthority crossfade;
        const auto decision = crossfade.evaluate(*oldWorld, *newWorld, makeCrossfadePolicy());
        snapshot.needsCrossfade = decision.needsCrossfade;
        snapshot.oldHasIR = decision.oldHasIR;
        snapshot.newHasIR = decision.newHasIR;
        snapshot.fadeTimeSec = decision.fadeTimeSec;
    }

    // HealthState Critical 時は crossfade を強制抑制（Orchestrator Step 2b と同じ）
    if (snapshot.needsCrossfade)
    {
        auto ref = getHealthStateRef();
        if (ref != nullptr)
        {
            auto health = convo::consumeAtomic(*ref, std::memory_order_acquire);
            if (health == convo::ISRHealthState::Critical)
            {
                snapshot.needsCrossfade = false;
                snapshot.fadeTimeSec = 0.0;
            }
        }
    }
    return snapshot;
}
