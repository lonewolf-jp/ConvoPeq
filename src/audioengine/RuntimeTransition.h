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
    double queuedFadeTimeSec = 0.0;
    int dspCrossfadeStartDelayBlocks = 0;
    int dspCrossfadeDryHoldSamples = 0;
    double dryScaleTarget = 1.0;           // dry-as-old crossfade の IRスケール目標値
    std::uint64_t revision = 0;
};

} // namespace convo
