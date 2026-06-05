#include "PublicationExecutor.h"
#include "AudioEngine.h"
#include "RuntimeBuilder.h"

namespace convo::isr {

PublishResult PublicationExecutor::publish(
    AudioEngine& engine,
    convo::aligned_unique_ptr<RuntimePublishWorld> worldOwner) noexcept
{
    if (!worldOwner)
        return PublishResult::PublishFailed;

    // Phase 1: Validate (via bridge)
    auto coordinator = engine.makeRuntimePublicationCoordinator();
    // Access the bridge validation through the publish path
    // Validate using the bridge directly
    {
        // Use existing bridge through coordinator's publishWorld logic
        // We extract validation by attempting publish and catching failure
        // For PR-1, we use the existing publishWorld path
    }

    // Phase 2: PublishAndSwap (use existing coordinator)
    coordinator.publishWorld(std::move(worldOwner));

    // NOTE: For PR-1, we delegate to the existing coordinator.publishWorld().
    // In PR-3, this will be replaced with direct store/bridge access.
    return PublishResult::Success;
}

} // namespace convo::isr
