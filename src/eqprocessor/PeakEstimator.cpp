#include "PeakEstimator.h"
#include "EQAnalysisMath.h"
#include <algorithm>
#include <cmath>

PeakEstimate PeakEstimator::estimate(const std::vector<MergedSample>& samples)
{
    PeakEstimate result;
    if (samples.empty())
        return result;

    const int peakIdx = findGlobalPeak(samples);
    if (peakIdx < 0)
        return result;

    const auto& peak = samples[static_cast<size_t>(peakIdx)];
    result.rawDb = static_cast<float>(EQAnalysisMath::linearToDb(peak.linearMagnitude));
    result.rawFreqHz = static_cast<float>(peak.freqHz);
    result.rawSampleIndex = peakIdx;

    // 放物線補間（端点では補間なし）
    if (peakIdx > 0 && peakIdx < static_cast<int>(samples.size()) - 1)
    {
        const auto& prev = samples[static_cast<size_t>(peakIdx) - 1];
        const auto& next = samples[static_cast<size_t>(peakIdx) + 1];

        const double x0 = std::log2(prev.freqHz);
        const double y0 = EQAnalysisMath::linearToDb(prev.linearMagnitude);
        const double x1 = std::log2(peak.freqHz);
        const double y1 = result.rawDb;
        const double x2 = std::log2(next.freqHz);
        const double y2 = EQAnalysisMath::linearToDb(next.linearMagnitude);

        const double interpolated = interpolateParabolic(x0, y0, x1, y1, x2, y2);

        // 補間値が妥当なら採用（異常値ガード）
        if (std::isfinite(interpolated) && interpolated >= y1)
        {
            result.interpolatedDb = static_cast<float>(interpolated);
            // 補間周波数: Lagrange から x_peak を計算
            const double denom = y0 - 2.0 * y1 + y2;
            if (std::abs(denom) > 1e-12)
            {
                const double delta = 0.5 * (y0 - y2) / denom;
                const double xPeak = x1 + delta;
                result.interpolatedFreqHz = static_cast<float>(std::pow(2.0, xPeak));
            }
            else
            {
                result.interpolatedFreqHz = result.rawFreqHz;
            }
        }
        else
        {
            result.interpolatedDb = result.rawDb;
            result.interpolatedFreqHz = result.rawFreqHz;
        }
    }
    else
    {
        // 端点: 補間なし
        result.interpolatedDb = result.rawDb;
        result.interpolatedFreqHz = result.rawFreqHz;
    }

    return result;
}

double PeakEstimator::interpolateParabolic(double x0, double y0,
                                            double x1, double y1,
                                            double x2, double y2)
{
    // Lagrange 二次補間（一般3点、不等間隔対応）
    // x_peak = 0.5 * (y0*(x1^2-x2^2) + y1*(x2^2-x0^2) + y2*(x0^2-x1^2))
    //              / (y0*(x1-x2) + y1*(x2-x0) + y2*(x0-x1))
    const double denom = y0 * (x1 - x2) + y1 * (x2 - x0) + y2 * (x0 - x1);
    if (std::abs(denom) < 1e-12)
        return y1;  // ゼロ除算防止

    const double numer = 0.5 * (y0 * (x1 * x1 - x2 * x2)
                               + y1 * (x2 * x2 - x0 * x0)
                               + y2 * (x0 * x0 - x1 * x1));
    const double xPeak = numer / denom;

    // Re-evaluate Lagrange at xPeak for the interpolated value
    const double l0 = ((xPeak - x1) * (xPeak - x2)) / ((x0 - x1) * (x0 - x2));
    const double l1 = ((xPeak - x0) * (xPeak - x2)) / ((x1 - x0) * (x1 - x2));
    const double l2 = ((xPeak - x0) * (xPeak - x1)) / ((x2 - x0) * (x2 - x1));
    const double interpolated = l0 * y0 + l1 * y1 + l2 * y2;

    if (!std::isfinite(interpolated))
        return y1;

    return interpolated;
}

int PeakEstimator::findGlobalPeak(const std::vector<MergedSample>& samples)
{
    if (samples.empty())
        return -1;

    int maxIdx = 0;
    double maxMag = samples[0].linearMagnitude;
    for (size_t i = 1; i < samples.size(); ++i)
    {
        if (samples[i].linearMagnitude > maxMag)
        {
            maxMag = samples[i].linearMagnitude;
            maxIdx = static_cast<int>(i);
        }
    }
    return maxIdx;
}
