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

    // 4. HealthState check (Practical-9: Admission Circuit Breaker)
    if (m_healthStateRef) {
        auto health = convo::consumeAtomic(*m_healthStateRef, std::memory_order_acquire);
        if (health == ISRHealthState::Critical) {
            // Critical: 全 publish 拒否（フェイルクローズ）
            return Decision::RejectedPressure;
        }
        if (health == ISRHealthState::Degraded) {
            // Degraded: 低優先度 publish を拒否
            //   generation==0 は存在しない（初回は 1）ため、一律 RejectedLowPriority は不可。
            //   代わりに RejectedPressure を返し、Coordinator 側で間引き制御に委ねる。
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
