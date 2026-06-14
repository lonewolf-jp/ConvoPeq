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
    // [work37 Phase 1.1] enqueueRetire の戻り値をチェックし、失敗時に tryReclaim + 再試行
    void retire(AudioEngine::DSPCore* dsp) noexcept
    {
        if (dsp == nullptr) return;
        // 1. Release DSP handle (must happen before enqueue)
        if (!engine_.retireDSPHandleForRuntime(dsp))
            return;

        // 2. Route through ISRRetireRouter → EpochDomain
        // ★ S-1: publishEpoch() → currentEpoch() に変更。retire が epoch を進めない。
        const uint64_t epoch = router_->currentEpoch();
        if (!router_->enqueueRetire(static_cast<void*>(dsp),
                                    &AudioEngine::destroyDSPCoreNode,
                                    epoch)) {
            // ★ work37: 初回失敗 → tryReclaim で backlog 消化後に再試行
            router_->tryReclaim();
            if (!router_->enqueueRetire(static_cast<void*>(dsp),
                                        &AudioEngine::destroyDSPCoreNode,
                                        epoch)) {
                // 再試行失敗は HealthMonitor overflowCount 監視に委ねる（ベストエフォート）
                return;
            }
        }

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
