#include "EQResponseSampler.h"
#include "EQAnalysisMath.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

//==============================================================================
// 1点評価
//==============================================================================
MergedSample EQResponseSampler::evaluate(double freqHz, const BandCollection& bands) const
{
    const double w = 2.0 * juce::MathConstants<double>::pi * freqHz / processingRate_;

    double linearMag = 0.0, upperBoundDb = 0.0;
    EQAnalysisMath::computeSampleResponse(
        bands.bands.data(), bands.bands.size(),
        w, isParallel_, linearMag, upperBoundDb);

    MergedSample s;
    s.freqHz = freqHz;
    s.linearMagnitude = linearMag;
    s.upperBoundDb = upperBoundDb;
    s.origin.type = EQProcessor::SampleOrigin::Unknown;
    s.origin.bandIndex = -1;
    s.origin.sampleIndex = -1;
    return s;
}

//==============================================================================
// 粗探索600点
//==============================================================================
CoarseScanResult EQResponseSampler::runCoarse(const BandCollection& bands) const
{
    CoarseScanResult result;
    result.samples.reserve(static_cast<size_t>(kCoarsePoints));

    constexpr double kTwentyOverLog10 = 8.685889638065036; // 20.0 / ln(10)
    constexpr double kEpsilon = 1e-6;
    const std::complex<double> kOne(1.0, 0.0);

    for (int i = 0; i < kCoarsePoints; ++i)
    {
        const double t = static_cast<double>(i) / static_cast<double>(kCoarsePoints - 1);
        const double freqHz = 10.0 * std::pow(maxFreq_ / 10.0, t);
        const double w = 2.0 * juce::MathConstants<double>::pi * freqHz / processingRate_;

        MergedSample sample;
        sample.freqHz = freqHz;
        sample.origin.type = EQProcessor::SampleOrigin::Coarse;
        sample.origin.bandIndex = -1;
        sample.origin.sampleIndex = i;

        if (isParallel_)
        {
            std::complex<double> parallelSum(1.0, 0.0);
            double logBound = 0.0;

            for (size_t j = 0; j < bands.bands.size(); ++j)
            {
                const auto& band = bands.bands[j];
                const auto H = EQAnalysisMath::biquadResponse(band.biquad, w);
                parallelSum += H - kOne;

                const double delta = std::abs(H - kOne);
                if (std::isfinite(delta))
                {
                    if (delta > kEpsilon)
                        logBound += std::log1p(delta);
                    result.bandMaxDelta[j] = std::max(result.bandMaxDelta[j], delta);
                }

                const double mag = std::abs(H);
                if (mag > result.bandMaxMagnitude[j])
                    result.bandMaxMagnitude[j] = mag;
            }

            sample.linearMagnitude = std::abs(parallelSum);
            sample.upperBoundDb = kTwentyOverLog10 * logBound;
        }
        else
        {
            double productMag = 1.0;
            double logBound = 0.0;

            for (size_t j = 0; j < bands.bands.size(); ++j)
            {
                const auto& band = bands.bands[j];
                const auto H = EQAnalysisMath::biquadResponse(band.biquad, w);
                const double mag = std::abs(H);
                productMag *= mag;

                const double delta = std::abs(H - kOne);
                if (std::isfinite(delta))
                {
                    if (delta > kEpsilon)
                        logBound += std::log1p(delta);
                    result.bandMaxDelta[j] = std::max(result.bandMaxDelta[j], delta);
                }

                if (mag > result.bandMaxMagnitude[j])
                    result.bandMaxMagnitude[j] = mag;
            }

            sample.linearMagnitude = productMag;
            sample.upperBoundDb = kTwentyOverLog10 * logBound;
        }

        result.samples.push_back(sample);
    }

    return result;
}

//==============================================================================
// 候補Band判定: measured 用
//==============================================================================
std::vector<const BandInfo*> EQResponseSampler::findMeasuredCandidates(
    const BandCollection& bands) const
{
    std::vector<const BandInfo*> candidates;
    for (const auto& band : bands.bands)
    {
        if (band.isBoosting)
            candidates.push_back(&band);
    }
    return candidates;
}

//==============================================================================
// 候補Band判定: upperBound 用（max|Hi-1| > 0.1）
//==============================================================================
std::vector<const BandInfo*> EQResponseSampler::findUpperBoundCandidates(
    const BandCollection& bands,
    const std::array<double, 20>& bandMaxDelta) const
{
    std::vector<const BandInfo*> candidates;
    for (size_t j = 0; j < bands.bands.size(); ++j)
    {
        if (bandMaxDelta[j] > kDeltaThreshold)
            candidates.push_back(&bands.bands[j]);
    }
    return candidates;
}

//==============================================================================
// 適応サンプリング（union統合+比例配分）
//==============================================================================
AdaptiveScanResult EQResponseSampler::runAdaptive(
    const BandCollection& bands,
    const std::vector<const BandInfo*>& measuredCands,
    const std::vector<const BandInfo*>& upperBoundCands,
    const CoarseScanResult& coarseResult) const
{
    AdaptiveScanResult result;

    // 候補Bandの範囲を収集（measured + upperBound 両方）
    struct RangeEntry {
        double start;
        double end;
    };
    std::vector<RangeEntry> ranges;

    auto addRange = [&](const BandInfo* band) {
        auto r = band->searchRange(maxFreq_);
        if (r.second > r.first)
            ranges.push_back({r.first, r.second});
    };

    for (auto* b : measuredCands) addRange(b);
    for (auto* b : upperBoundCands) addRange(b);

    if (ranges.empty())
        return result;

    // ソート
    std::sort(ranges.begin(), ranges.end(),
        [](const RangeEntry& a, const RangeEntry& b) { return a.start < b.start; });

    // Union統合（重複マージ）
    struct MergedRange {
        double start;
        double end;
        double length;
    };
    std::vector<MergedRange> merged;
    merged.push_back({ranges[0].start, ranges[0].end, 0.0});

    for (size_t i = 1; i < ranges.size(); ++i)
    {
        if (ranges[i].start <= merged.back().end)
            merged.back().end = std::max(merged.back().end, ranges[i].end);
        else
            merged.push_back({ranges[i].start, ranges[i].end, 0.0});
    }

    // 各区間の対数長を計算
    double totalLogLength = 0.0;
    for (auto& mr : merged)
    {
        mr.length = std::log2(mr.end / mr.start);
        totalLogLength += mr.length;
    }

    if (totalLogLength <= 0.0)
        return result;

    // 各区間に比例配分
    for (const auto& mr : merged)
    {
        const int numPoints = std::max(4, static_cast<int>(
            std::round(kAdaptivePoints * mr.length / totalLogLength)));

        for (int j = 0; j < numPoints; ++j)
        {
            const double t = static_cast<double>(j) / static_cast<double>(numPoints - 1);
            const double freqHz = mr.start * std::pow(mr.end / mr.start, t);
            auto s = evaluate(freqHz, bands);
            s.origin.type = EQProcessor::SampleOrigin::Adaptive;
            s.origin.bandIndex = -1;
            s.origin.sampleIndex = static_cast<int>(result.samples.size());
            result.samples.push_back(s);
        }
    }

    return result;
}
