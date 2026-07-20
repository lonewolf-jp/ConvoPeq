#include "BandHelper.h"
#include "EQAnalysisMath.h"
#include "EQProcessor.h"

BandCollection BandHelper::collectActiveBands(
    const EQProcessor& processor,
    const EQProcessor::EQState& state,
    double processingRate)
{
    BandCollection result;
    result.bands.reserve(EQProcessor::NUM_BANDS);

    float maxActiveQ = 0.0f;
    float maxTotalQ = 0.0f;

    for (int i = 0; i < EQProcessor::NUM_BANDS; ++i)
    {
        if (!state.bands[i].enabled)
            continue;

        const EQBandType type = state.bandTypes[i];
        const float gain = state.bands[i].gain;
        const double freq = static_cast<double>(state.bands[i].frequency);
        const double q = static_cast<double>(state.bands[i].q);

        // EQProcessor の static SVF 関数を利用
        const EQCoeffsSVF svf = processor.calcSVFCoeffs(type, static_cast<float>(freq),
                                                          gain, static_cast<float>(q), processingRate);
        const EQCoeffsBiquad bq = processor.svfToDisplayBiquad(svf);

        const bool boosting = EQAnalysisMath::isBoostingBand(type, gain);

        BandInfo info;
        info.index = i;
        info.freq = freq;
        info.q = q;
        info.type = type;
        info.gain = gain;
        info.biquad = bq;
        info.isBoosting = boosting;

        result.bands.push_back(info);

        if (boosting && static_cast<float>(q) > maxActiveQ)
            maxActiveQ = static_cast<float>(q);

        if (static_cast<float>(q) > maxTotalQ)
            maxTotalQ = static_cast<float>(q);
    }

    result.maxActiveQ = maxActiveQ;
    result.maxTotalQ = maxTotalQ;
    return result;
}
