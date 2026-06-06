#include "PublicationAdmission.h"
#include "AudioEngine.h"

namespace convo::isr {

PublicationAdmission::Decision PublicationAdmission::evaluate(
    const PublishRequest& req, AudioEngine& engine) const noexcept
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

    // 4. Pressure / throttle check
    if (convo::consumeAtomic(engine.retirePressurePublicationThrottleActive_,
                             std::memory_order_acquire))
        return Decision::RejectedPressure;

    // 5. Fading active check → defer
    const bool hasFading = engine.hasFadingRuntimeInWorld(
        engine.readControlRuntimeHandle());
    if (hasFading)
        return Decision::DeferredFadingActive;

    return Decision::Accepted;
}

} // namespace convo::isr
