#pragma once

#include <cstdint>

namespace convo {

// IMMUTABLE_RUNTIME: publish 後に変更しない Phase-1 骨格。
struct RuntimeGraph
{
    // IMMUTABLE_RUNTIME: runtime identity / publish generation
    std::uint64_t runtimeUuid = 0;
    std::uint64_t fadingRuntimeUuid = 0;
    std::uint64_t transitionCurrentRuntimeUuid = 0;
    std::uint64_t transitionNextRuntimeUuid = 0;
    std::uint64_t generation = 0;

    // IMMUTABLE_RUNTIME: graph node pointers (visibility only, no ownership)
    void* activeNode = nullptr;
    void* fadingNode = nullptr;

    // IMMUTABLE_RUNTIME: core processing metadata
    double sampleRate = 0.0;
    int ditherBitDepth = 0;
    int noiseShaperType = 0;
    int oversamplingFactor = 1;

    // IMMUTABLE_RUNTIME: processing mode flags
    bool eqBypassed = false;
    bool convBypassed = false;
    bool softClipEnabled = false;
    bool dspCrossfadePending = false;
    bool dspCrossfadeUseDryAsOld = false;
    bool firstIrDryCrossfadePending = false;
    double queuedFadeTimeSec = 0.0;
    int dspCrossfadeStartDelayBlocks = 0;
    int dspCrossfadeDryHoldSamples = 0;
    double dryScaleTarget = 1.0;           // dry-as-old crossfade の IRスケール目標値
    int latencyDelayOld = 0;
    int latencyDelayNew = 0;
    bool latencyResetPending = false;

    // IMMUTABLE_RUNTIME: gain / saturation parameters
    double saturationAmount = 0.0;
    double inputHeadroomGain = 1.0;
    double outputMakeupGain = 1.0;
    double convolverInputTrimGain = 1.0;

    // IMMUTABLE_RUNTIME: adaptive capture snapshot metadata
    int adaptiveCoeffBankIndex = -1;
    std::uint32_t adaptiveCoeffGeneration = 0;
    std::uint64_t captureSessionId = 0;

    // IMMUTABLE_RUNTIME: EQ AGC coefficient table snapshot view
    const double* eqAgcAttackCoeffTable = nullptr;
    const double* eqAgcReleaseCoeffTable = nullptr;
    const double* eqAgcSmoothCoeffTable = nullptr;
    int eqAgcCoeffTableCapacity = 0;
};

} // namespace convo
