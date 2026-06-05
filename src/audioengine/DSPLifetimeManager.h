#pragma once

#include "AudioEngine.h"

// DSPLifetimeManager: DSP の activation / crossfade / retire を一元管理する。
// Publication 完了後に NonRT で非同期的に呼ばれる。
//
// ★Phase-A: AudioEngine のラッパーに過ぎない。
//   真の分離は Phase-B/C で行う。
class DSPLifetimeManager {
public:
    explicit DSPLifetimeManager(AudioEngine& engine) noexcept : engine_(engine) {}

    void activate(AudioEngine::DSPCore* dsp) noexcept
    {
        if (dsp == nullptr) return;
        engine_.setActiveRuntimeDSP(dsp);
    }

    convo::isr::CrossfadeId beginCrossfade(convo::isr::DSPHandle from, convo::isr::DSPHandle to) noexcept
    {
        return engine_.dspHandleRuntime_.beginCrossfade(from, to);
    }

    void retire(AudioEngine::DSPCore* dsp) noexcept
    {
        if (dsp == nullptr) return;
        engine_.retireDSP(dsp);
    }

    void retireDeferred() noexcept
    {
        // deferred queue drain: handled by AudioEngine threading
    }

    AudioEngine::DSPCore* getActive() const noexcept { return engine_.getActiveRuntimeDSP(); }

private:
    AudioEngine& engine_;
};
