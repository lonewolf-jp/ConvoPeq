//==============================================================================
// GlobalSnapshot.cpp
//==============================================================================
#include "GlobalSnapshot.h"
#include "../ConvolverState.h"

namespace convo {

GlobalSnapshot::GlobalSnapshot(const SnapshotParams& params) noexcept
    : convState(params.convState)
    , convStateId(params.convStateId)
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
    , eqCoeffHash(params.eqCoeffHash)
{
    if (convState) {
        convState->snapshotRefCount.fetch_add(1, std::memory_order_relaxed);
    }
#ifdef _DEBUG
    alive.store(true, std::memory_order_relaxed);
#endif
}

GlobalSnapshot::~GlobalSnapshot()
{
    if (convState) {
#ifdef _DEBUG
        const int old = convState->snapshotRefCount.fetch_sub(1, std::memory_order_relaxed);
        jassert(old > 0);
#else
        convState->snapshotRefCount.fetch_sub(1, std::memory_order_relaxed);
#endif
    }
}

} // namespace convo
