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
    // ★ BUG-015/027 (work88): enqueue 失敗（QueuePressure/QueueFull）は enqueueWithRetry
    //   内部で RetireQuarantineStore へ移送済み（directDelete しない — RT 参照中の UAF 排除）。
    //   Shutdown はシャットダウン経路が処理。二重移送（double-quarantine → double-free）を
    //   避けるため、ここでは追加の quarantineRetire を呼ばない。滞留監視は
    //   AudioEngine 側の quarantineResidentCount / quarantineOverflowCount で行う。
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
        // ★ FUTURE-6 (work88): unordered_map 線形走査 → DSPHandleTable::findAndEraseByHandle
        //   （value 一致 + key 取得 + erase を一括。O(n)・固定 512 上限）
        void* rawKey = nullptr;
        if (engine_.runtimeDSPHandleMap_.findAndEraseByHandle(handle, rawKey))
            toDelete = static_cast<AudioEngine::DSPCore*>(rawKey);
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
    // ★ BUG-015/027 (work88): 同上 — enqueueWithRetry 内部で退避ストアへ移送済み。
    //   二重移送を避けるため追加処置なし（directDelete 禁止）。
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
