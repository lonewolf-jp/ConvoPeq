//==============================================================================
// GlobalSnapshot.cpp
//==============================================================================
#include "GlobalSnapshot.h"

namespace convo {

GlobalSnapshot::GlobalSnapshot(const SnapshotParams& params) noexcept
    : convStateId(params.convStateId)
    , eqParams(params.eqParams)
    , nsCoeffs(params.nsCoeffs)
    , contentHash(params.contentHash)
    , inputHeadroomGain(params.inputHeadroomGain)
    , outputMakeupGain(params.outputMakeupGain)
    , convInputTrimGain(params.convInputTrimGain)
    , convBypass(params.convBypass)
    , eqBypass(params.eqBypass)
    , processingOrder(params.processingOrder)
    , softClipEnabled(params.softClipEnabled)
    , saturationAmount(params.saturationAmount)
    , oversamplingType(params.oversamplingType)
    , oversamplingFactor(params.oversamplingFactor)
    , ditherBitDepth(params.ditherBitDepth)
    , noiseShaperType(params.noiseShaperType)
    , generation(params.generation)
    , sampleRate(params.sampleRate)
    , maxBlockSize(params.maxBlockSize)
    , eqCoeffHash(params.eqCoeffHash)
{
#ifdef _DEBUG
    alive.store(true, std::memory_order_relaxed);
#endif
}

GlobalSnapshot::~GlobalSnapshot() = default;

} // namespace convo
