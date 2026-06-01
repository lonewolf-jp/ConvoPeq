#pragma once

#include <cstdint>
#include "RuntimeGraph.h"

namespace convo {

enum class TransitionPolicy : uint8_t
{
    SmoothOnly = 0,
    HardReset,
    DryAsOld
};

// Runtime publish/crossfade の観測用ステート。
// 所有権は持たず、ポインタは状態可視化のためにのみ保持する。
struct TransitionState
{
    // AuthorityClass::Derived (mirrors authoritative transition fact for observation)
    void* current = nullptr;
    void* next = nullptr;
    TransitionPolicy policy = TransitionPolicy::SmoothOnly;
    double fadeTimeSec = 0.0;
    int latencyDeltaSamples = 0;
    bool active = false;
};

// Phase-2 migration target: single runtime snapshot publish for audio-thread reads.
// This keeps ownership out-of-band and only mirrors the runtime-visible state.
struct EngineRuntime
{
    // AuthorityClass::Derived
    void* current = nullptr;
    std::uint64_t currentRuntimeUuid = 0;
    void* fading = nullptr;
    std::uint64_t fadingRuntimeUuid = 0;
    TransitionState transition {};
    std::uint64_t transitionCurrentRuntimeUuid = 0;
    std::uint64_t transitionNextRuntimeUuid = 0;
    int latencyDelayOld = 0;
    int latencyDelayNew = 0;
    bool latencyResetPending = false;
    bool dspCrossfadePending = false;
    bool dspCrossfadeUseDryAsOld = false;
    bool firstIrDryCrossfadePending = false;
    int processingOrder = 0;
    bool eqBypassed = false;
    bool convBypassed = false;
    bool softClipEnabled = false;
    double saturationAmount = 0.0;
    double inputHeadroomGain = 1.0;
    double outputMakeupGain = 1.0;
    double convolverInputTrimGain = 1.0;
    std::uint64_t retireBacklog = 0;
    std::uint64_t deferredResidency = 0;
    bool rebuildWorkerRunning = false;
    int adaptiveCoeffBankIndex = -1;
    std::uint32_t adaptiveCoeffGeneration = 0;
    std::uint64_t eqCoeffHash = 0;
    double queuedFadeTimeSec = 0.0;
    int dspCrossfadeStartDelayBlocks = 0;
    int dspCrossfadeDryHoldSamples = 0;
    double dryScaleTarget = 1.0;           // dry-as-old crossfade の IRスケール目標値
    std::uint64_t revision = 0;
};

} // namespace convo
