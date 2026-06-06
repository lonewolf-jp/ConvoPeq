#pragma once

#include <cstdint>
#include <type_traits>

namespace convo {

struct BuildInput final {
    double sampleRate = 0.0;
    int blockSize = 0;
    int ditherBitDepth = 0;
    int oversamplingFactor = 0;
    int oversamplingType = 0;
    int noiseShaperType = 0;
    int processingOrder = 0;
    bool eqBypassed = false;
    bool convBypassed = false;
    bool softClipEnabled = false;
    double saturationAmount = 0.0;
    double inputHeadroomGain = 1.0;
    double outputMakeupGain = 1.0;
    double convolverInputTrimGain = 1.0;
};

struct RuntimeBuildFingerprint
{
    std::uint32_t fingerprintVersion = 1;
    std::uint64_t irIdentityHash = 0;
    std::uint64_t convolutionConfigHash = 0;
    std::uint64_t dspParameterHash = 0;
    double sampleRate = 0.0;
    int blockSize = 0;
};

struct RuntimeBuildSnapshot
{
    int generation = 0;
    BuildInput buildInput {};
    std::uint64_t convolverFingerprint = 0;
    RuntimeBuildFingerprint rebuildFingerprint {};
    bool sealed = false;

    // [PR-2] DSP semantic projection snapshot values
    // These fields are populated from DSPCore when snapshot is created,
    // and consumed by RuntimeBuilder::buildRuntimePublishWorld() to
    // construct dspProjection without DSPCore direct reads.
    bool irLoaded = false;
    bool irFinalized = false;
    std::uint64_t structuralHash = 0;
    int oversamplingFactor = 1;
    double sampleRate = 48000.0;
    int baseLatencySamples = 0;
};

static_assert(std::is_same_v<decltype(RuntimeBuildSnapshot{}.buildInput), BuildInput>,
              "RuntimeBuildSnapshot must use convo::BuildInput as the sole semantic input descriptor.");

[[nodiscard]] inline bool isRuntimeBuildSnapshotSealedAndCompatible(const RuntimeBuildSnapshot& snapshot,
                                                                    const RuntimeBuildSnapshot& other) noexcept
{
    if (!snapshot.sealed || !other.sealed)
        return false;

    if (snapshot.rebuildFingerprint.fingerprintVersion != other.rebuildFingerprint.fingerprintVersion)
        return false;

    return snapshot.buildInput.sampleRate == other.buildInput.sampleRate
        && snapshot.buildInput.blockSize == other.buildInput.blockSize
        && snapshot.buildInput.ditherBitDepth == other.buildInput.ditherBitDepth
        && snapshot.buildInput.oversamplingFactor == other.buildInput.oversamplingFactor
        && snapshot.buildInput.oversamplingType == other.buildInput.oversamplingType
        && snapshot.buildInput.noiseShaperType == other.buildInput.noiseShaperType
        && snapshot.buildInput.processingOrder == other.buildInput.processingOrder
        && snapshot.buildInput.eqBypassed == other.buildInput.eqBypassed
        && snapshot.buildInput.convBypassed == other.buildInput.convBypassed
        && snapshot.buildInput.softClipEnabled == other.buildInput.softClipEnabled
        && snapshot.buildInput.saturationAmount == other.buildInput.saturationAmount
        && snapshot.buildInput.inputHeadroomGain == other.buildInput.inputHeadroomGain
        && snapshot.buildInput.outputMakeupGain == other.buildInput.outputMakeupGain
        && snapshot.buildInput.convolverInputTrimGain == other.buildInput.convolverInputTrimGain
        && snapshot.convolverFingerprint == other.convolverFingerprint
        && snapshot.rebuildFingerprint.irIdentityHash == other.rebuildFingerprint.irIdentityHash
        && snapshot.rebuildFingerprint.convolutionConfigHash == other.rebuildFingerprint.convolutionConfigHash
        && snapshot.rebuildFingerprint.dspParameterHash == other.rebuildFingerprint.dspParameterHash;
}

} // namespace convo
