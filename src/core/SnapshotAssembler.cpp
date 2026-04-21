//==============================================================================
// SnapshotAssembler.cpp
//==============================================================================
#include "SnapshotAssembler.h"

namespace convo {

SnapshotParams SnapshotAssembler::assemble(
    const ConvolverState* conv,
    uint64_t convStateId,
    const EQParameters& eq,
    const std::array<double, 9>& nsCoeffs,
    double inputHeadroomGain,
    double outputMakeupGain,
    double convInputTrimGain,
    bool convBypass,
    bool eqBypass,
    bool softClipEnabled,
    float saturationAmount,
    ProcessingOrder processingOrder,
    OversamplingType oversamplingType,
    int oversamplingFactor,
    int ditherBitDepth,
    NoiseShaperType noiseShaperType,
    uint64_t generation,
    double sampleRate,          // v2.3 フェーズ1 追加
    int maxBlockSize,           // v2.3 フェーズ1 追加
    uint64_t eqCoeffHash        // v2.3 フェーズ1 追加
) noexcept
{
    SnapshotParams params;
    params.convState = conv;
    params.convStateId = convStateId;
    params.eqParams = eq;
    params.nsCoeffs = nsCoeffs;
    params.inputHeadroomGain = inputHeadroomGain;
    params.outputMakeupGain = outputMakeupGain;
    params.convInputTrimGain = convInputTrimGain;
    params.convBypass = convBypass;
    params.eqBypass = eqBypass;
    params.softClipEnabled = softClipEnabled;
    params.saturationAmount = saturationAmount;
    params.processingOrder = processingOrder;
    params.oversamplingType = oversamplingType;
    params.oversamplingFactor = oversamplingFactor;
    params.ditherBitDepth = ditherBitDepth;
    params.noiseShaperType = noiseShaperType;
    params.generation = generation;
    params.sampleRate = sampleRate;         // v2.3 フェーズ1 追加
    params.maxBlockSize = maxBlockSize;     // v2.3 フェーズ1 追加
    params.eqCoeffHash = eqCoeffHash;       // v2.3 フェーズ1 追加
    return params;
}

} // namespace convo
