//==============================================================================
// EQBoundExcessBenchmark.cpp — Week2 実IRベンチマーク (EQ編)
//
// 目的:
//   boundExcessDb = max(0, upperBound.gainDb - measured.gainDb) の分布を
//   実測し、upperBound の過大評価量を定量化する。
//
// 測定項目:
//   - boundExcessDb: 平均・中央値・95%tile・最大値
//   - CPU時間 / コール
//   - 候補Band数
//   - algorithm / filterStructure
//
// 使い方:
//   EQBoundExcessBenchmark [--quick] [--json]
//
// 設計:
//   JUCE/AudioEngine 非依存。EQAnalysisUnitTests.cpp と同一パターンで
//   inline 実装を使用する。
//==============================================================================
#include <cmath>
#include <complex>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <algorithm>
#include <vector>
#include <array>
#include <cstdint>
#include <cfloat>
#include <chrono>
#include <random>
#include <map>

//==============================================================================
// 型定義（EQAnalysisTypes.h / EQProcessor.h から抽出）
//==============================================================================
namespace {

constexpr double kPi = 3.14159265358979323846;

enum class EQBandType : uint8_t {
    Peaking, LowShelf, HighShelf, LowPass, HighPass
};

struct EQCoeffsBiquad {
    double b0 = 1.0, b1 = 0.0, b2 = 0.0;
    double a0 = 1.0, a1 = 0.0, a2 = 0.0;
};

struct SampleOrigin {
    enum Type : uint8_t { Unknown = 0, Coarse = 1, Adaptive = 2, Union = 3 };
    Type type = Coarse;
    int bandIndex = -1;
    int sampleIndex = -1;
};

struct MergedSample {
    double freqHz;
    double linearMagnitude;
    double upperBoundDb;
    SampleOrigin origin;
};

struct PeakEstimate {
    float interpolatedDb = 0.0f;
    float interpolatedFreqHz = 0.0f;
    float rawDb = 0.0f;
    float rawFreqHz = 0.0f;
    int rawSampleIndex = -1;
};

struct UpperBoundEstimate {
    float maxDb = 0.0f;
    float freqHz = 0.0f;
    int sampleIndex = -1;
};

struct BandInfo {
    int index;
    double freq;
    double q;
    EQBandType type;
    float gain;
    EQCoeffsBiquad biquad;
    bool isBoosting;

    std::pair<double, double> searchRange(double maxFreq) const noexcept {
        switch (type) {
            case EQBandType::Peaking:
                return { std::max(10.0, freq / 4.0), std::min(maxFreq, freq * 4.0) };
            case EQBandType::LowShelf:
                return { 10.0, std::min(maxFreq, freq * 2.0) };
            case EQBandType::HighShelf:
                return { std::max(10.0, freq / 2.0), maxFreq };
            default:
                return { std::max(10.0, freq / 4.0), std::min(maxFreq, freq * 4.0) };
        }
    }
};

struct BandCollection {
    std::vector<BandInfo> bands;
    float maxActiveQ = 0.0f;
    float maxTotalQ = 0.0f;
};

struct CoarseScanResult {
    std::vector<MergedSample> samples;
    std::array<double, 20> bandMaxDelta{};
    std::array<double, 20> bandMaxMagnitude{};
};

struct AdaptiveScanResult {
    std::vector<MergedSample> samples;
};

//==============================================================================
// 数学ヘルパー
//==============================================================================
double linearToDb(double linear) noexcept {
    return (linear > 1e-18) ? 20.0 * std::log10(linear) : -DBL_MAX;
}
double dbToLinear(double db) noexcept {
    return std::pow(10.0, db / 20.0);
}
bool isBoostingBand(EQBandType type, float gain) noexcept {
    if (!(gain > 0.01f)) return false;
    switch (type) {
        case EQBandType::Peaking:
        case EQBandType::LowShelf:
        case EQBandType::HighShelf:
            return gain > 0.01f;
        default:
            return false;
    }
}

std::complex<double> biquadResponse(const EQCoeffsBiquad& c, double w) noexcept {
    const std::complex<double> z(std::cos(w), std::sin(w));
    const std::complex<double> z2 = z * z;
    const std::complex<double> num = c.b0 * z2 + c.b1 * z + c.b2;
    const std::complex<double> den = c.a0 * z2 + c.a1 * z + c.a2;
    const double denNorm = std::norm(den);
    if (denNorm < 1e-18) return std::complex<double>(1.0, 0.0);
    return num / den;
}

void computeSampleResponse(
    const BandInfo* bands, size_t numBands,
    double normalizedFreq, bool isParallel,
    double& outLinearMagnitude, double& outUpperBoundDb) noexcept
{
    constexpr double kTwentyOverLog10 = 8.685889638065036;
    constexpr double kEpsilon = 1e-6;
    const std::complex<double> kOne(1.0, 0.0);
    double logBound = 0.0;

    if (isParallel) {
        std::complex<double> parallelSum(1.0, 0.0);
        for (size_t i = 0; i < numBands; ++i) {
            const auto H = biquadResponse(bands[i].biquad, normalizedFreq);
            parallelSum += H - kOne;
            const double delta = std::abs(H - kOne);
            if (std::isfinite(delta) && delta > kEpsilon)
                logBound += std::log1p(delta);
        }
        outLinearMagnitude = std::abs(parallelSum);
        outUpperBoundDb = kTwentyOverLog10 * logBound;
    } else {
        double productMag = 1.0;
        for (size_t i = 0; i < numBands; ++i) {
            const auto H = biquadResponse(bands[i].biquad, normalizedFreq);
            productMag *= std::abs(H);
            const double delta = std::abs(H - kOne);
            if (std::isfinite(delta) && delta > kEpsilon)
                logBound += std::log1p(delta);
        }
        outLinearMagnitude = productMag;
        outUpperBoundDb = kTwentyOverLog10 * logBound;
    }
}

//==============================================================================
// Audio EQ Cookbook 係数
//==============================================================================
EQCoeffsBiquad calcPeakingBiquad(double freq, double gainDb, double q, double sr) noexcept {
    EQCoeffsBiquad c{};
    const double A = std::pow(10.0, gainDb / 40.0);
    const double w0 = 2.0 * kPi * freq / sr;
    const double cosw0 = std::cos(w0);
    const double alpha = std::sin(w0) / (2.0 * q);
    const double a0 = 1.0 + alpha / A;
    if (std::abs(a0) < 1e-15) { c.b0 = 1.0; c.a0 = 1.0; return c; }
    c.b0 = 1.0 + alpha * A;
    c.b1 = -2.0 * cosw0;
    c.b2 = 1.0 - alpha * A;
    c.a0 = a0;
    c.a1 = -2.0 * cosw0;
    c.a2 = 1.0 - alpha / A;
    return c;
}

EQCoeffsBiquad calcLowShelfBiquad(double freq, double gainDb, double q, double sr) noexcept {
    EQCoeffsBiquad c{};
    const double A = std::pow(10.0, gainDb / 40.0);
    const double w0 = 2.0 * kPi * freq / sr;
    const double cosw0 = std::cos(w0);
    const double sinw0 = std::sin(w0);
    const double alpha = sinw0 / (2.0 * q);
    const double sqrtA = std::sqrt(A);
    const double twoSqrtAAlpha = 2.0 * sqrtA * alpha;
    const double a0 = (A + 1.0) + (A - 1.0) * cosw0 + twoSqrtAAlpha;
    if (std::abs(a0) < 1e-15) { c.b0 = 1.0; c.a0 = 1.0; return c; }
    c.b0 = A * ((A + 1.0) - (A - 1.0) * cosw0 + twoSqrtAAlpha);
    c.b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cosw0);
    c.b2 = A * ((A + 1.0) - (A - 1.0) * cosw0 - twoSqrtAAlpha);
    c.a0 = a0;
    c.a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cosw0);
    c.a2 = (A + 1.0) + (A - 1.0) * cosw0 - twoSqrtAAlpha;
    return c;
}

EQCoeffsBiquad calcHighShelfBiquad(double freq, double gainDb, double q, double sr) noexcept {
    EQCoeffsBiquad c{};
    const double A = std::pow(10.0, gainDb / 40.0);
    const double w0 = 2.0 * kPi * freq / sr;
    const double cosw0 = std::cos(w0);
    const double sinw0 = std::sin(w0);
    const double alpha = sinw0 / (2.0 * q);
    const double sqrtA = std::sqrt(A);
    const double twoSqrtAAlpha = 2.0 * sqrtA * alpha;
    const double a0 = (A + 1.0) - (A - 1.0) * cosw0 + twoSqrtAAlpha;
    if (std::abs(a0) < 1e-15) { c.b0 = 1.0; c.a0 = 1.0; return c; }
    c.b0 = A * ((A + 1.0) + (A - 1.0) * cosw0 + twoSqrtAAlpha);
    c.b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cosw0);
    c.b2 = A * ((A + 1.0) + (A - 1.0) * cosw0 - twoSqrtAAlpha);
    c.a0 = a0;
    c.a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cosw0);
    c.a2 = (A + 1.0) - (A - 1.0) * cosw0 - twoSqrtAAlpha;
    return c;
}

EQCoeffsBiquad calcBypassBiquad() noexcept {
    EQCoeffsBiquad c{};
    c.b0 = 1.0; c.a0 = 1.0;
    return c;
}

//==============================================================================
// EQResponseSampler 実装（stateless）
//==============================================================================
class EQResponseSampler {
public:
    EQResponseSampler(double processingRate, bool isParallel) noexcept
        : processingRate_(processingRate), isParallel_(isParallel),
          nyquist_(processingRate * 0.5),
          maxFreq_(std::min(20000.0, nyquist_)) {}

    static constexpr int kCoarsePoints = 600;
    static constexpr int kAdaptivePoints = 128;
    static constexpr double kDeltaThreshold = 0.1;

    MergedSample evaluate(double freqHz, const BandCollection& bands) const {
        const double w = 2.0 * kPi * freqHz / processingRate_;
        double linearMag = 0.0, upperBoundDb = 0.0;
        computeSampleResponse(bands.bands.data(), bands.bands.size(),
                              w, isParallel_, linearMag, upperBoundDb);
        MergedSample s;
        s.freqHz = freqHz;
        s.linearMagnitude = linearMag;
        s.upperBoundDb = upperBoundDb;
        s.origin.type = SampleOrigin::Unknown;
        s.origin.bandIndex = -1;
        s.origin.sampleIndex = -1;
        return s;
    }

    CoarseScanResult runCoarse(const BandCollection& bands) const {
        CoarseScanResult result;
        result.samples.reserve(static_cast<size_t>(kCoarsePoints));
        constexpr double kTwentyOverLog10 = 8.685889638065036;
        constexpr double kEpsilon = 1e-6;
        const std::complex<double> kOne(1.0, 0.0);

        for (int i = 0; i < kCoarsePoints; ++i) {
            const double t = static_cast<double>(i) / static_cast<double>(kCoarsePoints - 1);
            const double freqHz = 10.0 * std::pow(maxFreq_ / 10.0, t);
            const double w = 2.0 * kPi * freqHz / processingRate_;

            MergedSample sample;
            sample.freqHz = freqHz;
            sample.origin.type = SampleOrigin::Coarse;
            sample.origin.bandIndex = -1;
            sample.origin.sampleIndex = i;

            if (isParallel_) {
                std::complex<double> parallelSum(1.0, 0.0);
                double logBound = 0.0;
                for (size_t j = 0; j < bands.bands.size(); ++j) {
                    const auto& band = bands.bands[j];
                    const auto H = biquadResponse(band.biquad, w);
                    parallelSum += H - kOne;
                    const double delta = std::abs(H - kOne);
                    if (std::isfinite(delta)) {
                        if (delta > kEpsilon) logBound += std::log1p(delta);
                        result.bandMaxDelta[j] = std::max(result.bandMaxDelta[j], delta);
                    }
                    const double mag = std::abs(H);
                    if (mag > result.bandMaxMagnitude[j])
                        result.bandMaxMagnitude[j] = mag;
                }
                sample.linearMagnitude = std::abs(parallelSum);
                sample.upperBoundDb = kTwentyOverLog10 * logBound;
            } else {
                double productMag = 1.0;
                double logBound = 0.0;
                for (size_t j = 0; j < bands.bands.size(); ++j) {
                    const auto& band = bands.bands[j];
                    const auto H = biquadResponse(band.biquad, w);
                    const double mag = std::abs(H);
                    productMag *= mag;
                    const double delta = std::abs(H - kOne);
                    if (std::isfinite(delta)) {
                        if (delta > kEpsilon) logBound += std::log1p(delta);
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

    std::vector<const BandInfo*> findMeasuredCandidates(const BandCollection& bands) const {
        std::vector<const BandInfo*> candidates;
        for (const auto& band : bands.bands) {
            if (band.isBoosting) candidates.push_back(&band);
        }
        return candidates;
    }

    std::vector<const BandInfo*> findUpperBoundCandidates(
        const BandCollection& bands, const std::array<double, 20>& bandMaxDelta) const
    {
        std::vector<const BandInfo*> candidates;
        for (size_t j = 0; j < bands.bands.size(); ++j) {
            if (bandMaxDelta[j] > kDeltaThreshold)
                candidates.push_back(&bands.bands[j]);
        }
        return candidates;
    }

    AdaptiveScanResult runAdaptive(
        const BandCollection& bands,
        const std::vector<const BandInfo*>& measuredCands,
        const std::vector<const BandInfo*>& upperBoundCands,
        const CoarseScanResult& coarseResult) const
    {
        AdaptiveScanResult result;
        struct RangeEntry { double start; double end; };
        std::vector<RangeEntry> ranges;

        auto addRange = [&](const BandInfo* band) {
            auto r = band->searchRange(maxFreq_);
            if (r.second > r.first) ranges.push_back({r.first, r.second});
        };
        for (auto* b : measuredCands) addRange(b);
        for (auto* b : upperBoundCands) addRange(b);
        if (ranges.empty()) return result;

        std::sort(ranges.begin(), ranges.end(),
            [](const RangeEntry& a, const RangeEntry& b) { return a.start < b.start; });

        struct MergedRange { double start; double end; double length; };
        std::vector<MergedRange> merged;
        merged.push_back({ranges[0].start, ranges[0].end, 0.0});
        for (size_t i = 1; i < ranges.size(); ++i) {
            if (ranges[i].start <= merged.back().end)
                merged.back().end = std::max(merged.back().end, ranges[i].end);
            else
                merged.push_back({ranges[i].start, ranges[i].end, 0.0});
        }

        double totalLogLength = 0.0;
        for (auto& mr : merged) {
            mr.length = std::log2(mr.end / mr.start);
            totalLogLength += mr.length;
        }
        if (totalLogLength <= 0.0) return result;

        int totalCandBands = static_cast<int>(measuredCands.size() + upperBoundCands.size());
        for (const auto& mr : merged) {
            const int numPoints = std::max(4, static_cast<int>(
                std::round(kAdaptivePoints * mr.length / totalLogLength)));
            for (int j = 0; j < numPoints; ++j) {
                const double t = static_cast<double>(j) / static_cast<double>(numPoints - 1);
                const double freqHz = mr.start * std::pow(mr.end / mr.start, t);
                auto s = evaluate(freqHz, bands);
                s.origin.type = SampleOrigin::Adaptive;
                s.origin.bandIndex = -1;
                s.origin.sampleIndex = static_cast<int>(result.samples.size());
                result.samples.push_back(s);
            }
        }
        (void)coarseResult; // used by production for candidate optimization, not needed here
        return result;
    }

private:
    double processingRate_;
    bool isParallel_;
    double nyquist_;
    double maxFreq_;
};

//==============================================================================
// merge パイプライン
//==============================================================================
std::vector<MergedSample> mergeAndSort(
    const CoarseScanResult& coarse, const AdaptiveScanResult& adaptive)
{
    std::vector<MergedSample> result;
    result.reserve(coarse.samples.size() + adaptive.samples.size());
    for (const auto& s : coarse.samples) result.push_back(s);
    for (const auto& s : adaptive.samples) result.push_back(s);
    std::stable_sort(result.begin(), result.end(),
        [](const MergedSample& a, const MergedSample& b) { return a.freqHz < b.freqHz; });
    return result;
}

std::vector<MergedSample> deduplicate(const std::vector<MergedSample>& sorted) {
    if (sorted.empty()) return {};
    std::vector<MergedSample> result;
    result.reserve(sorted.size());
    result.push_back(sorted[0]);
    for (size_t i = 1; i < sorted.size(); ++i) {
        if (sorted[i].freqHz == result.back().freqHz) {
            if (result.back().origin.type == SampleOrigin::Coarse
                && sorted[i].origin.type == SampleOrigin::Adaptive)
                result.back() = sorted[i];
        } else {
            result.push_back(sorted[i]);
        }
    }
    return result;
}

void renumber(std::vector<MergedSample>& samples) {
    for (size_t i = 0; i < samples.size(); ++i)
        samples[i].origin.sampleIndex = static_cast<int>(i);
}

//==============================================================================
// PeakEstimator / UpperBoundEstimator
//==============================================================================
int findGlobalPeak(const std::vector<MergedSample>& samples) {
    if (samples.empty()) return -1;
    int maxIdx = 0;
    double maxMag = samples[0].linearMagnitude;
    for (size_t i = 1; i < samples.size(); ++i) {
        if (samples[i].linearMagnitude > maxMag) {
            maxMag = samples[i].linearMagnitude;
            maxIdx = static_cast<int>(i);
        }
    }
    return maxIdx;
}

double interpolateParabolic(double x0, double y0, double x1, double y1,
                             double x2, double y2) noexcept
{
    const double denom = y0 * (x1 - x2) + y1 * (x2 - x0) + y2 * (x0 - x1);
    if (std::abs(denom) < 1e-12) return y1;
    const double numer = 0.5 * (y0 * (x1 * x1 - x2 * x2)
                                + y1 * (x2 * x2 - x0 * x0)
                                + y2 * (x0 * x0 - x1 * x1));
    const double xPeak = numer / denom;
    const double l0 = ((xPeak - x1) * (xPeak - x2)) / ((x0 - x1) * (x0 - x2));
    const double l1 = ((xPeak - x0) * (xPeak - x2)) / ((x1 - x0) * (x1 - x2));
    const double l2 = ((xPeak - x0) * (xPeak - x1)) / ((x2 - x0) * (x2 - x1));
    const double interpolated = l0 * y0 + l1 * y1 + l2 * y2;
    if (!std::isfinite(interpolated)) return y1;
    return interpolated;
}

PeakEstimate estimate(const std::vector<MergedSample>& samples) {
    PeakEstimate result;
    if (samples.empty()) return result;
    const int peakIdx = findGlobalPeak(samples);
    if (peakIdx < 0) return result;
    const auto& peak = samples[static_cast<size_t>(peakIdx)];
    result.rawDb = static_cast<float>(linearToDb(peak.linearMagnitude));
    result.rawFreqHz = static_cast<float>(peak.freqHz);
    result.rawSampleIndex = peakIdx;

    if (peakIdx > 0 && peakIdx < static_cast<int>(samples.size()) - 1) {
        const auto& prev = samples[static_cast<size_t>(peakIdx) - 1];
        const auto& next = samples[static_cast<size_t>(peakIdx) + 1];
        const double x0 = std::log2(prev.freqHz);
        const double y0 = linearToDb(prev.linearMagnitude);
        const double x1 = std::log2(peak.freqHz);
        const double y1 = result.rawDb;
        const double x2 = std::log2(next.freqHz);
        const double y2 = linearToDb(next.linearMagnitude);
        const double interp = interpolateParabolic(x0, y0, x1, y1, x2, y2);
        if (std::isfinite(interp) && interp >= y1) {
            result.interpolatedDb = static_cast<float>(interp);
            const double denom = y0 - 2.0 * y1 + y2;
            if (std::abs(denom) > 1e-12) {
                const double delta = 0.5 * (y0 - y2) / denom;
                result.interpolatedFreqHz = static_cast<float>(std::pow(2.0, x1 + delta));
            } else {
                result.interpolatedFreqHz = result.rawFreqHz;
            }
        } else {
            result.interpolatedDb = result.rawDb;
            result.interpolatedFreqHz = result.rawFreqHz;
        }
    } else {
        result.interpolatedDb = result.rawDb;
        result.interpolatedFreqHz = result.rawFreqHz;
    }
    return result;
}

UpperBoundEstimate estimateMax(const std::vector<MergedSample>& samples) {
    UpperBoundEstimate result;
    if (samples.empty()) return result;
    size_t maxIdx = 0;
    double maxVal = samples[0].upperBoundDb;
    for (size_t i = 1; i < samples.size(); ++i) {
        if (samples[i].upperBoundDb > maxVal) {
            maxVal = samples[i].upperBoundDb;
            maxIdx = i;
        }
    }
    result.maxDb = static_cast<float>(maxVal);
    result.freqHz = static_cast<float>(samples[maxIdx].freqHz);
    result.sampleIndex = static_cast<int>(maxIdx);
    return result;
}

//==============================================================================
// テスト構成生成
//==============================================================================
struct BenchmarkConfig {
    std::string label;
    bool isParallel;
    double sampleRate;
    int numBands;
};

struct BenchmarkResult {
    std::string label;
    bool isParallel;
    double sampleRate;
    int numBands;
    int totalCoarseSamples;
    int totalAdaptiveSamples;
    int totalMergedSamples;
    int measuredCandCount;
    int ubCandCount;
    double measuredGainDb;
    double measuredRawGainDb;
    double upperBoundGainDb;
    double boundExcessDb;
    double elapsedUs;
    bool hasNanOrInf;
};

//==============================================================================
// 構成スイープ
//==============================================================================
std::vector<BandCollection> generateConfigurations(
    const BenchmarkConfig& cfg, std::mt19937& rng)
{
    std::vector<BandCollection> configs;
    std::uniform_real_distribution<double> freqDist(50.0, 18000.0);
    std::uniform_real_distribution<double> gainDist(-12.0, 18.0);
    std::uniform_real_distribution<double> qDist(0.5, 10.0);

    // 5 random seeds per config
    for (int seed = 0; seed < 5; ++seed) {
        rng.seed(static_cast<unsigned>(seed * 1000 + cfg.numBands));

        BandCollection bc;
        float maxActiveQ = 0.0f;
        float maxTotalQ = 0.0f;

        for (int i = 0; i < cfg.numBands; ++i) {
            BandInfo bi;
            bi.index = i;
            bi.freq = freqDist(rng);
            bi.q = qDist(rng);
            bi.gain = static_cast<float>(gainDist(rng));

            // Assign type with some variation
            int typeChoice = i % 4;
            switch (typeChoice) {
                case 0: bi.type = EQBandType::Peaking; break;
                case 1: bi.type = EQBandType::LowShelf; break;
                case 2: bi.type = EQBandType::HighShelf; break;
                case 3: bi.type = EQBandType::Peaking; break; // more peaking
            }

            bi.isBoosting = isBoostingBand(bi.type, bi.gain);

            // Generate Biquad coefficients
            switch (bi.type) {
                case EQBandType::Peaking:
                    bi.biquad = calcPeakingBiquad(bi.freq, bi.gain, bi.q, cfg.sampleRate);
                    break;
                case EQBandType::LowShelf:
                    bi.biquad = calcLowShelfBiquad(bi.freq, bi.gain, bi.q, cfg.sampleRate);
                    break;
                case EQBandType::HighShelf:
                    bi.biquad = calcHighShelfBiquad(bi.freq, bi.gain, bi.q, cfg.sampleRate);
                    break;
                default:
                    bi.biquad = calcPeakingBiquad(bi.freq, bi.gain, bi.q, cfg.sampleRate);
                    break;
            }

            if (bi.isBoosting && static_cast<float>(bi.q) > maxActiveQ)
                maxActiveQ = static_cast<float>(bi.q);
            if (static_cast<float>(bi.q) > maxTotalQ)
                maxTotalQ = static_cast<float>(bi.q);

            bc.bands.push_back(bi);
        }
        bc.maxActiveQ = maxActiveQ;
        bc.maxTotalQ = maxTotalQ;
        configs.push_back(bc);
    }

    // Add edge cases: all peaking, all shelf, etc.
    // (1) All peaking +12dB Q=1
    {
        BandCollection bc;
        float maq = 0.0f, mtq = 0.0f;
        for (int i = 0; i < cfg.numBands; ++i) {
            BandInfo bi;
            bi.index = i;
            bi.freq = 200.0 * std::pow(40000.0 / 200.0, static_cast<double>(i) / std::max(1.0, static_cast<double>(cfg.numBands - 1)));
            bi.q = 1.0;
            bi.gain = 12.0f;
            bi.type = EQBandType::Peaking;
            bi.isBoosting = true;
            bi.biquad = calcPeakingBiquad(bi.freq, bi.gain, bi.q, cfg.sampleRate);
            if (static_cast<float>(bi.q) > maq) maq = static_cast<float>(bi.q);
            if (static_cast<float>(bi.q) > mtq) mtq = static_cast<float>(bi.q);
            bc.bands.push_back(bi);
        }
        bc.maxActiveQ = maq;
        bc.maxTotalQ = mtq;
        configs.push_back(bc);
    }

    // (2) All Peaking +24dB Q=10 (worst-case)
    {
        BandCollection bc;
        float maq2 = 0.0f, mtq2 = 0.0f;
        for (int i = 0; i < cfg.numBands; ++i) {
            BandInfo bi;
            bi.index = i;
            bi.freq = 200.0 * std::pow(40000.0 / 200.0, static_cast<double>(i) / std::max(1.0, static_cast<double>(cfg.numBands - 1)));
            bi.q = 10.0;
            bi.gain = 24.0f;
            bi.type = EQBandType::Peaking;
            bi.isBoosting = true;
            bi.biquad = calcPeakingBiquad(bi.freq, bi.gain, bi.q, cfg.sampleRate);
            bc.bands.push_back(bi);
            if (static_cast<float>(bi.q) > maq2) maq2 = static_cast<float>(bi.q);
            if (static_cast<float>(bi.q) > mtq2) mtq2 = static_cast<float>(bi.q);
        }
        bc.maxActiveQ = maq2;
        bc.maxTotalQ = mtq2;
        configs.push_back(bc);
    }

    // (3) Opposite phase: +12dB + +12dB at same frequency
    {
        BandCollection bc;
        for (int i = 0; i < std::min(2, cfg.numBands); ++i) {
            BandInfo bi;
            bi.index = i;
            bi.freq = 1000.0;
            bi.q = 1.0;
            bi.gain = 12.0f;
            bi.type = EQBandType::Peaking;
            bi.isBoosting = true;
            bi.biquad = calcPeakingBiquad(bi.freq, bi.gain, bi.q, cfg.sampleRate);
            bc.bands.push_back(bi);
        }
        bc.maxActiveQ = 1.0f;
        bc.maxTotalQ = 1.0f;
        configs.push_back(bc);
    }

    // (4) LowShelf +18dB single band (Shelf peak detection test)
    if (cfg.numBands >= 1) {
        BandCollection bc;
        BandInfo bi;
        bi.index = 0;
        bi.freq = 1000.0;
        bi.q = 0.707;
        bi.gain = 18.0f;
        bi.type = EQBandType::LowShelf;
        bi.isBoosting = true;
        bi.biquad = calcLowShelfBiquad(bi.freq, bi.gain, bi.q, cfg.sampleRate);
        bc.bands.push_back(bi);
        bc.maxActiveQ = 0.707f;
        bc.maxTotalQ = 0.707f;
        configs.push_back(bc);
    }

    return configs;
}

//==============================================================================
// 測定
//==============================================================================
BenchmarkResult runBenchmark(
    const BenchmarkConfig& cfg,
    const BandCollection& bands,
    int configIdx)
{
    BenchmarkResult result;
    result.label = cfg.label + "_cfg" + std::to_string(configIdx);
    result.isParallel = cfg.isParallel;
    result.sampleRate = cfg.sampleRate;
    result.numBands = cfg.numBands;

    const EQResponseSampler sampler(cfg.sampleRate, cfg.isParallel);

    auto start = std::chrono::high_resolution_clock::now();

    // Full pipeline
    const auto coarseResult = sampler.runCoarse(bands);
    const auto measCands = sampler.findMeasuredCandidates(bands);
    const auto ubCands = sampler.findUpperBoundCandidates(bands, coarseResult.bandMaxDelta);
    const auto adaptiveResult = sampler.runAdaptive(bands, measCands, ubCands, coarseResult);

    auto merged = mergeAndSort(coarseResult, adaptiveResult);
    merged = deduplicate(merged);
    renumber(merged);

    const auto measured = estimate(merged);
    const auto upperBound = estimateMax(merged);

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    result.measuredGainDb = measured.interpolatedDb;
    result.measuredRawGainDb = measured.rawDb;
    result.upperBoundGainDb = upperBound.maxDb;
    result.boundExcessDb = std::max<double>(0.0, upperBound.maxDb - static_cast<double>(measured.interpolatedDb));

    result.totalCoarseSamples = static_cast<int>(coarseResult.samples.size());
    result.totalAdaptiveSamples = static_cast<int>(adaptiveResult.samples.size());
    result.totalMergedSamples = static_cast<int>(merged.size());
    result.measuredCandCount = static_cast<int>(measCands.size());
    result.ubCandCount = static_cast<int>(ubCands.size());

    // Check for NaN/Inf
    result.hasNanOrInf = !std::isfinite(measured.interpolatedDb)
                       || !std::isfinite(upperBound.maxDb)
                       || !std::isfinite(result.boundExcessDb);

    return result;
}

//==============================================================================
// 統計
//==============================================================================
struct Stats {
    int count = 0;
    int nanCount = 0;
    double mean = 0.0;
    double median = 0.0;
    double p95 = 0.0;
    double max = 0.0;
    double min = 0.0;
    double meanCpuUs = 0.0;
    double meanMeasuredCands = 0.0;
    double meanUbCands = 0.0;
    double meanMergedSamples = 0.0;
};

Stats computeStats(const std::vector<BenchmarkResult>& results, const std::string& filter) {
    std::vector<double> values;
    Stats stats;
    double totalCpu = 0.0;
    double totalMeasCands = 0.0;
    double totalUbCands = 0.0;
    double totalMerged = 0.0;

    for (const auto& r : results) {
        if (!filter.empty() && r.label.find(filter) == std::string::npos)
            continue;
        values.push_back(r.boundExcessDb);
        totalCpu += r.elapsedUs;
        totalMeasCands += r.measuredCandCount;
        totalUbCands += r.ubCandCount;
        totalMerged += r.totalMergedSamples;
        if (r.hasNanOrInf) ++stats.nanCount;
    }

    stats.count = static_cast<int>(values.size());
    if (values.empty()) return stats;

    std::sort(values.begin(), values.end());
    stats.min = values.front();
    stats.max = values.back();

    double sum = 0.0;
    for (double v : values) sum += v;
    stats.mean = sum / values.size();
    stats.median = values[values.size() / 2];
    stats.p95 = values[static_cast<size_t>(values.size() * 0.95)];
    stats.meanCpuUs = totalCpu / values.size();
    stats.meanMeasuredCands = totalMeasCands / values.size();
    stats.meanUbCands = totalUbCands / values.size();
    stats.meanMergedSamples = totalMerged / values.size();

    return stats;
}

void printResult(const BenchmarkResult& r) {
    std::cout
        << std::left << std::setw(35) << r.label.substr(0, 34)
        << " | " << (r.isParallel ? "Par" : "Ser")
        << " | SR=" << std::right << std::setw(6) << static_cast<int>(r.sampleRate)
        << " | B=" << std::setw(2) << r.numBands
        << " | meas=" << std::setw(6) << std::fixed << std::setprecision(2) << r.measuredGainDb
        << " | raw=" << std::setw(6) << r.measuredRawGainDb
        << " | ub=" << std::setw(6) << r.upperBoundGainDb
        << " | excess=" << std::setw(7) << r.boundExcessDb
        << " | candM=" << r.measuredCandCount << " candU=" << r.ubCandCount
        << " | CP=" << r.elapsedUs << "us"
        << (r.hasNanOrInf ? " *** NaN/Inf ***" : "")
        << std::endl;
}

void printStats(const Stats& s, const std::string& title) {
    std::cout << "\n--- " << title << " ---\n"
              << "  Configs: " << s.count
              << " (NaN: " << s.nanCount << ")\n"
              << "  boundExcessDb: min=" << s.min
              << " mean=" << s.mean
              << " median=" << s.median
              << " p95=" << s.p95
              << " max=" << s.max << "\n"
              << "  Avg CPU: " << s.meanCpuUs << " us/call\n"
              << "  Avg cand: measured=" << s.meanMeasuredCands
              << " upperBound=" << s.meanUbCands << "\n"
              << "  Avg merged samples: " << s.meanMergedSamples << "\n";
}

} // anonymous namespace

//==============================================================================
// main
//==============================================================================
int main(int argc, char** argv)
{
    bool quickMode = false;
    bool jsonMode = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--quick") quickMode = true;
        if (arg == "--json") jsonMode = true;
    }

    std::cout << "============================================================\n";
    std::cout << "EQ BoundExcess Benchmark  (Week2)\n";
    std::cout << "  upperBound over-measurement quantification\n";
    std::cout << "============================================================\n";

    std::vector<BenchmarkConfig> configs;
    std::mt19937 rng(42);

    // Serial
    configs.push_back({"Ser_1b", false, 48000, 1});
    configs.push_back({"Ser_3b", false, 48000, 3});
    configs.push_back({"Ser_5b", false, 48000, 5});
    configs.push_back({"Ser_10b", false, 48000, 10});
    configs.push_back({"Ser_20b", false, 48000, 20});
    if (!quickMode) {
        configs.push_back({"Ser_20b_96k", false, 96000, 20});
        configs.push_back({"Ser_20b_192k", false, 192000, 20});
    }

    // Parallel
    configs.push_back({"Par_1b", true, 48000, 1});
    configs.push_back({"Par_3b", true, 48000, 3});
    configs.push_back({"Par_5b", true, 48000, 5});
    configs.push_back({"Par_10b", true, 48000, 10});
    configs.push_back({"Par_20b", true, 48000, 20});
    if (!quickMode) {
        configs.push_back({"Par_20b_96k", true, 96000, 20});
        configs.push_back({"Par_20b_192k", true, 192000, 20});
    }

    std::vector<BenchmarkResult> allResults;
    int totalConfigs = 0;

    for (const auto& cfg : configs) {
        auto bandConfigs = generateConfigurations(cfg, rng);
        int idx = 0;
        for (auto& bands : bandConfigs) {
            auto result = runBenchmark(cfg, bands, idx);
            printResult(result);
            allResults.push_back(result);
            ++totalConfigs;
            ++idx;
        }
    }

    // 全体統計
    auto allStats = computeStats(allResults, "");
    printStats(allStats, "ALL CONFIGS");

    // Serial vs Parallel
    auto serStats = computeStats(allResults, "Ser_");
    auto parStats = computeStats(allResults, "Par_");
    printStats(serStats, "SERIAL ONLY");
    printStats(parStats, "PARALLEL ONLY");

    // Parallel opposite phase specific
    auto oppStats = computeStats(allResults, "Opposite");
    printStats(oppStats, "OPPOSITE PHASE");

    std::cout << "\n============================================================\n";
    std::cout << "Total: " << totalConfigs << " configurations\n";
    std::cout << "============================================================\n";

    // Summary
    std::cout << "\n=== SUMMARY ===\n";
    std::cout << "boundExcessDb (ALL):  mean=" << allStats.mean
              << "  p95=" << allStats.p95
              << "  max=" << allStats.max << "\n";
    std::cout << "boundExcessDb (SER):  mean=" << serStats.mean
              << "  p95=" << serStats.p95
              << "  max=" << serStats.max << "\n";
    std::cout << "boundExcessDb (PAR):  mean=" << parStats.mean
              << "  p95=" << parStats.p95
              << "  max=" << parStats.max << "\n";
    std::cout << "NaN/Inf configs: " << allStats.nanCount << "/" << allStats.count << "\n";
    std::cout << "Avg CPU: " << allStats.meanCpuUs << " us\n";

    return (allStats.nanCount > 0) ? 1 : 0;
}
