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

} // namespace convo
