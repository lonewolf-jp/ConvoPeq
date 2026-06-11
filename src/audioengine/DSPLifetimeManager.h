#pragma once

#include "AudioEngine.h"
#include "ISRRetireRouter.h"

// DSPLifetimeManager: DSP の activation / crossfade / retire を一元管理する。
// Publication 完了後に NonRT で非同期的に呼ばれる。
//
// ★Phase-B: ISRRetireRouter 経由で EpochDomain に直接 enqueueRetire する。
//   retireDSP() のラッパではなく、Router → EpochDomain へ直接委譲する。
class DSPLifetimeManager {
public:
    explicit DSPLifetimeManager(AudioEngine& engine) noexcept
        : engine_(engine), router_(engine.m_retireRouter.get()) {}

    explicit DSPLifetimeManager(AudioEngine& engine, convo::isr::ISRRetireRouter* router) noexcept
        : engine_(engine), router_(router) {}

    // Authority: DSPLifetimeManager (Lifecycle Authority)
    void activate(AudioEngine::DSPCore* dsp) noexcept
    {
        if (dsp == nullptr) return;
        engine_.setActiveRuntimeDSP(dsp);
    }

    // Authority: CrossfadeAuthorityRuntime
    convo::isr::CrossfadeId beginCrossfade(convo::isr::DSPHandle from, convo::isr::DSPHandle to) noexcept
    {
        return engine_.dspHandleRuntime_.beginCrossfade(from, to);
    }

    // Authority: DSPLifetimeManager (Lifecycle Authority)
    // Retire pipeline: DSPLifetimeManager → ISRRetireRouter → EpochDomain
    void retire(AudioEngine::DSPCore* dsp) noexcept
    {
        if (dsp == nullptr) return;
        // 1. Release DSP handle (must happen before enqueue)
        if (!engine_.retireDSPHandleForRuntime(dsp))
            return;

        // 2. Route through ISRRetireRouter → EpochDomain
        const uint64_t epoch = router_->publishEpoch();
        router_->enqueueRetire(static_cast<void*>(dsp),
                               &AudioEngine::destroyDSPCoreNode,
                               epoch);

        convo::fetchAddAtomic(engine_.rtAuxMutable_.runtimeRetireCount,
                              static_cast<std::uint64_t>(1),
                              std::memory_order_acq_rel);
    }

    void retireDeferred() noexcept
    {
        // deferred queue drain: handled by AudioEngine threading
    }

    AudioEngine::DSPCore* getActive() const noexcept { return engine_.getActiveRuntimeDSP(); }

private:
    AudioEngine& engine_;
    convo::isr::ISRRetireRouter* router_;
};
