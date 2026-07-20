#include "UpperBoundEstimator.h"

UpperBoundEstimate UpperBoundEstimator::estimateMax(const std::vector<MergedSample>& samples)
{
    UpperBoundEstimate result;
    if (samples.empty())
        return result;

    size_t maxIdx = 0;
    double maxVal = samples[0].upperBoundDb;
    for (size_t i = 1; i < samples.size(); ++i)
    {
        if (samples[i].upperBoundDb > maxVal)
        {
            maxVal = samples[i].upperBoundDb;
            maxIdx = i;
        }
    }

    result.maxDb = static_cast<float>(maxVal);
    result.freqHz = static_cast<float>(samples[maxIdx].freqHz);
    result.sampleIndex = static_cast<int>(maxIdx);
    return result;
}
