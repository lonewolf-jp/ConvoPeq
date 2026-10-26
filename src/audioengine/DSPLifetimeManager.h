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

    // V-11 §10.6-2 (2026-10-26): shutdown-only ownership transfer WITHOUT touching the
    // slot. The caller has already reclaimed the slot under a quiescence permit.
    //   - erases the runtimeDSPHandleMap_ entry for `handle` (full handle compare, so it
    //     is generation-safe) under runtimeDSPHandleMapMutex_;
    //   - enqueues the DSPCore into the EBR retire path (destroyDSPCoreNode).
    // It must NOT call retire()/reclaim(): a Reclaimed slot would go back to Retired and
    // be pushed into the free list twice.
    // Returns true when an entry was found and the ownership moved.
    [[nodiscard]] bool retireDetachedCoreForShutdown(convo::isr::DSPHandle handle) noexcept;
    void retireDeferred() noexcept;
    void* getActive() const noexcept;
    void destroyRolledBackDSP(void* dsp) noexcept;
    [[nodiscard]] uint64_t retiringGeneration() const noexcept;

private:
    AudioEngine& engine_;
    convo::isr::ISRRetireRouter* router_;
    std::atomic<uint64_t> currentRetiringGeneration_{0};
};
