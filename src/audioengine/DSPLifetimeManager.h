#pragma once

#include "ISRRetireRouter.h"

class AudioEngine;

class DSPLifetimeManager {
public:
    explicit DSPLifetimeManager(AudioEngine& engine) noexcept;
    explicit DSPLifetimeManager(AudioEngine& engine, convo::isr::ISRRetireRouter* router) noexcept;

    void activate(void* dsp) noexcept;
    void beginCrossfade(convo::isr::DSPHandle from, convo::isr::DSPHandle to, convo::isr::CrossfadeId id) noexcept;
    void retire(void* dsp) noexcept;
    void retire(void* dsp, uint64_t publicationEpoch) noexcept;
    void retireByHandle(convo::isr::DSPHandle handle) noexcept;
    void retireDeferred() noexcept;
    void* getActive() const noexcept;
    void destroyRolledBackDSP(void* dsp) noexcept;
    [[nodiscard]] uint64_t retiringGeneration() const noexcept;

private:
    AudioEngine& engine_;
    convo::isr::ISRRetireRouter* router_;
    std::atomic<uint64_t> currentRetiringGeneration_{0};
};
