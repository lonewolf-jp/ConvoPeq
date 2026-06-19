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
