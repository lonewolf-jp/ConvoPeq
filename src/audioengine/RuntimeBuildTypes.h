#pragma once

#include <cstdint>

namespace convo {

struct BuildInput {
    double sampleRate = 0.0;
    int blockSize = 0;
    int ditherBitDepth = 0;
    int oversamplingFactor = 0;
    int oversamplingType = 0;
    int noiseShaperType = 0;
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
};

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
        && snapshot.convolverFingerprint == other.convolverFingerprint
        && snapshot.rebuildFingerprint.irIdentityHash == other.rebuildFingerprint.irIdentityHash
        && snapshot.rebuildFingerprint.convolutionConfigHash == other.rebuildFingerprint.convolutionConfigHash
        && snapshot.rebuildFingerprint.dspParameterHash == other.rebuildFingerprint.dspParameterHash;
}

} // namespace convo
