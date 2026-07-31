#include "ISRRuntimePublicationCoordinator.h"
#include "DSPLifetimeManager.h"
#include "AudioEngine.h"

namespace convo::isr {

void RuntimePublicationCoordinator::processIntent(
    AudioEngine& engine,
    DSPLifetimeManager& lifetimeMgr) noexcept
{
    ObserveIntent intent;
    while (observeIntentQueue_.pop(intent)) {
        const auto currentEpoch = persistentState_.publicationEpoch;
        if (intent.epoch < currentEpoch || intent.handle.isNull()) {
            continue;
        }
        lifetimeMgr.retireByHandle(intent.handle);
    }

    while (observeFallbackQueue_.pop(intent)) {
        const auto currentEpoch = persistentState_.publicationEpoch;
        if (intent.epoch < currentEpoch || intent.handle.isNull()) {
            continue;
        }
        lifetimeMgr.retireByHandle(intent.handle);
    }

    engine.markReceiptReclaimComplete();

    setPendingIntentCount(0);
}

} // namespace convo::isr
