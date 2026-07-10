#include <JuceHeader.h>
#include "AudioEngine.h"
#include "RuntimeBuilder.h"
#include "RuntimePublicationOrchestrator.h"

// ★ A-4: Idle world publish 統一関数（publish only）
//   責務は publishWorld のみ。setDryHoldSamples/resfreshSnapshot は含まない。
//   前準備は各経路の呼び出し側で実行すること。
//   Returns: true=publish 実行, false=shutdown guard または nullptr で skip
bool AudioEngine::publishIdleWorldOnly(
    AudioEngine::DSPCore* currentAfterFade,
    convo::TransitionPolicy idlePolicy) noexcept
{
    // Shutdown guard（publishWorld パスに明示的な guard がないため）
    if (isShutdownInProgress())
        return false;
    if (currentAfterFade == nullptr)
        return false;

    // Idle world 発行 — 呼び出し側ですべての前準備を完了している前提
    auto coordinator = makeRuntimePublicationCoordinator();
    auto worldBuilder = convo::RuntimeBuilder(*this);
    worldBuilder.setHealthStateRef(getHealthStateRef());
    auto worldOwner = worldBuilder.buildRuntimePublishWorld(
        currentAfterFade, nullptr, idlePolicy, 0.0, false);
    const auto pubResult = commitRuntimePublication(coordinator, std::move(worldOwner),
                             RegistrationContext::needsRegistration(currentAfterFade));
    juce::ignoreUnused(pubResult);
    return true;
}
