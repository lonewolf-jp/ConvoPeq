#include "PublicationAdmission.h"
#include "AudioEngine.h"

namespace convo::isr {

PublicationAdmission::Decision PublicationAdmission::evaluate(
    const PublishRequest& req, AudioEngine& engine,
    const convo::RuntimeReaderContext& ctx) const noexcept
{
    // 1. Shutdown check
    if (engine.isShutdownInProgress())
        return Decision::RejectedShutdown;

    // 2. Generation staleness check
    const int currentGen = convo::consumeAtomic(
        engine.rebuildRequestGeneration, std::memory_order_acquire);
    if (req.generation != currentGen)
        return Decision::RejectedStaleGeneration;

    // 3. DSP finalized check (from sealedSnapshot, not DSPCore*)
    if (req.sealedSnapshot.irLoaded && !req.sealedSnapshot.irFinalized)
        return Decision::RejectedNotFinalized;

    // 4. HealthState check (P1-B: HealthMonitor の統合 HealthState を参照)
    if (m_healthStateRef) {
        auto health = convo::consumeAtomic(*m_healthStateRef, std::memory_order_acquire);
        if (health == ISRHealthState::Critical) {
            return Decision::RejectedPressure;
        }
        if (health == ISRHealthState::Degraded) {
            // Degraded 時は低優先度 publish を拒否（現状一律 RejectedPressure）
            return Decision::RejectedPressure;
        }
    }

    // 5. Pressure / throttle check (P1-6: Adaptive Backpressure)
    const bool pressureActive = convo::consumeAtomic(
        engine.retirePressurePublicationThrottleActive_, std::memory_order_acquire);
    if (pressureActive) {
        // ★ P1-6: Pressure レベル段階制御
        // RejectLowPriority: timer/crossfade publish を拒否
        // RejectMostRequests: bootstrap以外の全publish拒否
        // 現状は一律 RejectedPressure で対応
        return Decision::RejectedPressure;
    }

    // 5. Fading active check → defer
    const bool hasFading = engine.hasFadingRuntimeInWorld(
        engine.makeRuntimeReadHandle(ctx));
    if (hasFading)
        return Decision::DeferredFadingActive;

    return Decision::Accepted;
}

} // namespace convo::isr
