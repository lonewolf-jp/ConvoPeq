#include "DSPLifetimeManager.h"
#include "AudioEngine.h"

DSPLifetimeManager::DSPLifetimeManager(AudioEngine& engine) noexcept
    : engine_(engine)
    , router_(engine_.m_retireRouter.get())
{
}

DSPLifetimeManager::DSPLifetimeManager(AudioEngine& engine, convo::isr::ISRRetireRouter* router) noexcept
    : engine_(engine)
    , router_(router)
{
}

void DSPLifetimeManager::activate(void* dsp) noexcept
{
    if (dsp == nullptr)
        return;
    engine_.registerDSPHandleForRuntime(static_cast<AudioEngine::DSPCore*>(dsp));
}

void DSPLifetimeManager::beginCrossfade(convo::isr::DSPHandle from, convo::isr::DSPHandle to, convo::isr::CrossfadeId id) noexcept
{
    juce::ignoreUnused(from, to, id);
}

void DSPLifetimeManager::retire(void* dsp) noexcept
{
    retire(dsp, 0);
}

void DSPLifetimeManager::retire(void* dsp, uint64_t publicationEpoch) noexcept
{
    if (dsp == nullptr)
        return;

    const bool retired = engine_.retireDSPHandleForRuntime(static_cast<AudioEngine::DSPCore*>(dsp));
    if (!retired)
        return;

    if (router_ == nullptr)
        return;

    const auto epoch = (publicationEpoch > 0)
        ? publicationEpoch
        : router_->currentEpoch();

    const auto result = router_->enqueueWithRetry(
        dsp, &AudioEngine::destroyDSPCoreNode,
        epoch,
        DeletionEntryType::Generic);
    juce::ignoreUnused(result);

    convo::fetchAddAtomic(currentRetiringGeneration_,
        static_cast<uint64_t>(1),
        std::memory_order_acq_rel);
}

void DSPLifetimeManager::retireByHandle(convo::isr::DSPHandle handle) noexcept
{
    if (handle.isNull())
        return;

    AudioEngine::DSPCore* toDelete = nullptr;

    {
        std::lock_guard<std::mutex> lock(engine_.runtimeDSPHandleMapMutex_);
        for (auto it = engine_.runtimeDSPHandleMap_.begin();
             it != engine_.runtimeDSPHandleMap_.end(); ++it)
        {
            if (it->second == handle)
            {
                toDelete = it->first;
                engine_.runtimeDSPHandleMap_.erase(it);
                break;
            }
        }
    }

    if (toDelete == nullptr)
        return;

    engine_.dspHandleRuntime_.retire(handle);

    if (router_ == nullptr)
        return;

    const auto epoch = router_->currentEpoch();
    const auto result = router_->enqueueWithRetry(
        toDelete, &AudioEngine::destroyDSPCoreNode,
        epoch,
        DeletionEntryType::Generic);
    juce::ignoreUnused(result);

    convo::fetchAddAtomic(currentRetiringGeneration_,
        static_cast<uint64_t>(1),
        std::memory_order_acq_rel);
}

void DSPLifetimeManager::retireDeferred() noexcept
{
    convo::consumeAtomic(currentRetiringGeneration_, std::memory_order_acquire);
}

void* DSPLifetimeManager::getActive() const noexcept
{
    return engine_.getActiveRuntimeDSP();
}

void DSPLifetimeManager::destroyRolledBackDSP(void* dsp) noexcept
{
    if (dsp == nullptr)
        return;
    AudioEngine::destroyDSPCoreNode(dsp);
    convo::fetchAddAtomic(currentRetiringGeneration_,
        static_cast<uint64_t>(1),
        std::memory_order_acq_rel);
}

[[nodiscard]] uint64_t DSPLifetimeManager::retiringGeneration() const noexcept
{
    return convo::consumeAtomic(currentRetiringGeneration_,
                                std::memory_order_acquire);
}
