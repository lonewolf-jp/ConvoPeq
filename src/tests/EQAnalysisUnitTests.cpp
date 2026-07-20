//==============================================================================
// EQAnalysisUnitTests.cpp — ★ v14.47 3層リファクタリング 単体テスト
//
// PeakEstimator / UpperBoundEstimator / AnalysisMerge / EQResponseSampler
// の各コンポーネントを検証する。
//
// JUCE/AudioEngine に依存しない純粋数学テスト。
// EQProcessorMaxGainTests.cpp と同一パターンで inline 実装を使用。
//==============================================================================
#include <cmath>
#include <complex>
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <cfloat>
#include <cstdint>
#include <numbers>

namespace {

constexpr double kPi = 3.14159265358979323846;

int g_testsPassed = 0;
int g_testsFailed = 0;

void check(bool condition, const std::string& label)
{
    if (condition)
        ++g_testsPassed;
    else
        ++g_testsFailed, std::cerr << "[FAIL] " << label << "\n";
}

//==============================================================================
// 最小限の型定義（EQAnalysisTypes.h / EQProcessor.h から抽出）
//==============================================================================

enum class EQBandType : uint8_t {
    Peaking, LowShelf, HighShelf, LowPass, HighPass
};

struct EQCoeffsBiquad {
    double b0 = 1.0, b1 = 0.0, b2 = 0.0;
    double a0 = 1.0, a1 = 0.0, a2 = 0.0;
};

struct SampleOrigin {
    enum Type { Coarse, Adaptive };
    Type type = Coarse;
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
};

struct BandCollection {
    std::vector<BandInfo> bands;
    float maxActiveQ = 0.0f;
    float maxTotalQ = 0.0f;
};

//==============================================================================
// 数学ヘルパー（EQAnalysisMath.h から抽出）
//==============================================================================

double linearToDb(double linear) noexcept {
    return (linear > 1e-18) ? 20.0 * std::log10(linear)
                            : -DBL_MAX;
}

double dbToLinear(double db) noexcept {
    return std::pow(10.0, db / 20.0);
}

bool isBoostingBand(EQBandType type, float gain) noexcept {
    if (!(gain > 0.01f))
        return false;
    switch (type) {
        case EQBandType::Peaking:
        case EQBandType::LowShelf:
        case EQBandType::HighShelf:
            return gain > 0.01f;
        case EQBandType::LowPass:
        case EQBandType::HighPass:
            return false;
    }
    return false;
}

std::complex<double> biquadResponse(const EQCoeffsBiquad& c, double w) noexcept
{
    const std::complex<double> z(std::cos(w), std::sin(w));
    const std::complex<double> z2 = z * z;
    const std::complex<double> num = c.b0 * z2 + c.b1 * z + c.b2;
    const std::complex<double> den = c.a0 * z2 + c.a1 * z + c.a2;
    const double denNorm = std::norm(den);
    if (denNorm < 1e-18)
        return std::complex<double>(1.0, 0.0);
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

    if (isParallel)
    {
        std::complex<double> parallelSum(1.0, 0.0);
        for (size_t i = 0; i < numBands; ++i)
        {
            const auto H = biquadResponse(bands[i].biquad, normalizedFreq);
            parallelSum += H - kOne;
            const double delta = std::abs(H - kOne);
            if (std::isfinite(delta) && delta > kEpsilon)
                logBound += std::log1p(delta);
        }
        outLinearMagnitude = std::abs(parallelSum);
        outUpperBoundDb = kTwentyOverLog10 * logBound;
    }
    else
    {
        double productMag = 1.0;
        for (size_t i = 0; i < numBands; ++i)
        {
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
// PeakEstimator 実装（PeakEstimator.cpp から抽出）
//==============================================================================

int findGlobalPeak(const std::vector<MergedSample>& samples)
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

double interpolateParabolic(double x0, double y0,
                            double x1, double y1,
                            double x2, double y2) noexcept
{
    const double denom = y0 * (x1 - x2) + y1 * (x2 - x0) + y2 * (x0 - x1);
    if (std::abs(denom) < 1e-12)
        return y1;

    const double numer = 0.5 * (y0 * (x1 * x1 - x2 * x2)
                               + y1 * (x2 * x2 - x0 * x0)
                               + y2 * (x0 * x0 - x1 * x1));
    const double xPeak = numer / denom;

    const double l0 = ((xPeak - x1) * (xPeak - x2)) / ((x0 - x1) * (x0 - x2));
    const double l1 = ((xPeak - x0) * (xPeak - x2)) / ((x1 - x0) * (x1 - x2));
    const double l2 = ((xPeak - x0) * (xPeak - x1)) / ((x2 - x0) * (x2 - x1));
    const double interpolated = l0 * y0 + l1 * y1 + l2 * y2;

    if (!std::isfinite(interpolated))
        return y1;

    return interpolated;
}

PeakEstimate estimate(const std::vector<MergedSample>& samples)
{
    PeakEstimate result;
    if (samples.empty())
        return result;

    const int peakIdx = findGlobalPeak(samples);
    if (peakIdx < 0)
        return result;

    const auto& peak = samples[static_cast<size_t>(peakIdx)];
    result.rawDb = static_cast<float>(linearToDb(peak.linearMagnitude));
    result.rawFreqHz = static_cast<float>(peak.freqHz);
    result.rawSampleIndex = peakIdx;

    // 放物線補間（端点では補間なし）
    if (peakIdx > 0 && peakIdx < static_cast<int>(samples.size()) - 1)
    {
        const auto& prev = samples[static_cast<size_t>(peakIdx) - 1];
        const auto& next = samples[static_cast<size_t>(peakIdx) + 1];

        const double x0 = std::log2(prev.freqHz);
        const double y0 = linearToDb(prev.linearMagnitude);
        const double x1 = std::log2(peak.freqHz);
        const double y1 = result.rawDb;
        const double x2 = std::log2(next.freqHz);
        const double y2 = linearToDb(next.linearMagnitude);

        const double interpolated = interpolateParabolic(x0, y0, x1, y1, x2, y2);

        if (std::isfinite(interpolated) && interpolated >= y1)
        {
            result.interpolatedDb = static_cast<float>(interpolated);
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
        result.interpolatedDb = result.rawDb;
        result.interpolatedFreqHz = result.rawFreqHz;
    }

    return result;
}

//==============================================================================
// UpperBoundEstimator 実装（UpperBoundEstimator.cpp から抽出）
//==============================================================================

UpperBoundEstimate estimateMax(const std::vector<MergedSample>& samples)
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

//==============================================================================
// AnalysisMerge 実装（AnalysisMerge.h から抽出）
//==============================================================================

std::vector<MergedSample> mergeAndSort(
    const std::vector<MergedSample>& coarse,
    const std::vector<MergedSample>& adaptive)
{
    std::vector<MergedSample> result;
    result.reserve(coarse.size() + adaptive.size());

    for (const auto& s : coarse)
        result.push_back(s);
    for (const auto& s : adaptive)
        result.push_back(s);

    std::stable_sort(result.begin(), result.end(),
        [](const MergedSample& a, const MergedSample& b) {
            return a.freqHz < b.freqHz;
        });

    return result;
}

std::vector<MergedSample> deduplicate(const std::vector<MergedSample>& sorted)
{
    if (sorted.empty())
        return {};

    std::vector<MergedSample> result;
    result.reserve(sorted.size());
    result.push_back(sorted[0]);

    for (size_t i = 1; i < sorted.size(); ++i)
    {
        if (sorted[i].freqHz == result.back().freqHz)
        {
            if (result.back().origin.type == SampleOrigin::Coarse
                && sorted[i].origin.type == SampleOrigin::Adaptive)
            {
                result.back() = sorted[i];
            }
        }
        else
        {
            result.push_back(sorted[i]);
        }
    }

    return result;
}

void renumber(std::vector<MergedSample>& samples)
{
    for (size_t i = 0; i < samples.size(); ++i)
        samples[i].origin.sampleIndex = static_cast<int>(i);
}

//==============================================================================
// TEST GROUP 1: PeakEstimator.interpolateParabolic (10 cases)
//   Lagrange 3点不等間隔放物線補間の検証
//==============================================================================

void testInterpolateParabolic_symmetric()
{
    // 対称放物線 y = 4 - (x-2)^2, peak at x=2, y=4
    // Points: (1, 3), (2, 4), (3, 3)
    const double y = interpolateParabolic(1.0, 3.0, 2.0, 4.0, 3.0, 3.0);
    check(std::abs(y - 4.0) < 1e-10,
          "interpParabolic: symmetric peak ≈ 4, got " + std::to_string(y));
}

void testInterpolateParabolic_leftShifted()
{
    // 左寄せ放物線 y = 4 - (x-1)^2, peak at x=1, y=4
    // Points: (0, 3), (1, 4), (2, 3)
    const double y = interpolateParabolic(0.0, 3.0, 1.0, 4.0, 2.0, 3.0);
    check(std::abs(y - 4.0) < 1e-10,
          "interpParabolic: left-shifted peak ≈ 4, got " + std::to_string(y));
}

void testInterpolateParabolic_rightShifted()
{
    // 右寄せ放物線 y = 4 - (x-3)^2, peak at x=3, y=4
    // Points: (2, 3), (3, 4), (4, 3)
    const double y = interpolateParabolic(2.0, 3.0, 3.0, 4.0, 4.0, 3.0);
    check(std::abs(y - 4.0) < 1e-10,
          "interpParabolic: right-shifted peak ≈ 4, got " + std::to_string(y));
}

void testInterpolateParabolic_flat()
{
    // フラット（全て同じy値）→ 分母 ≒ 0 → y1 を返す
    const double y = interpolateParabolic(1.0, 3.0, 2.0, 3.0, 3.0, 3.0);
    check(std::abs(y - 3.0) < 1e-12,
          "interpParabolic: flat returns y1=3, got " + std::to_string(y));
}

void testInterpolateParabolic_ascending()
{
    // 単調増加 → 分母=0に近い → y1 を返す
    const double y = interpolateParabolic(1.0, 1.0, 2.0, 2.0, 3.0, 3.0);
    check(std::abs(y - 2.0) < 1e-12,
          "interpParabolic: ascending returns y1=2, got " + std::to_string(y));
}

void testInterpolateParabolic_narrow()
{
    // 鋭いピーク y = 10 - 5*(x+0.3)^2, peak at x=-0.3, y=10
    // Points: (-1, 10-5*0.49=7.55), (0, 10-5*0.09=9.55), (1, 10-5*1.69=1.55) → not symmetric
    // Let's use simple one: y = 10 - (x-2)^2 * 10, points around x=2
    // At x=1.9: 10 - 0.01*10 = 9.9, x=2.0: 10, x=2.1: 10 - 0.01*10 = 9.9
    // That's too narrow. Let me use: y = 5 - (x-2)^2, peak at (2,5)
    // Points: (1.5, 5-0.25=4.75), (2, 5), (2.5, 5-0.25=4.75)
    const double y = interpolateParabolic(1.5, 4.75, 2.0, 5.0, 2.5, 4.75);
    check(std::abs(y - 5.0) < 1e-10,
          "interpParabolic: narrow peak ≈ 5, got " + std::to_string(y));
}

void testInterpolateParabolic_wide()
{
    // 広いピーク y = 3 - 0.2*(x-2)^2, peak at (2,3)
    // Points: (1, 3-0.2=2.8), (2, 3), (3, 3-0.2=2.8)
    const double y = interpolateParabolic(1.0, 2.8, 2.0, 3.0, 3.0, 2.8);
    check(std::abs(y - 3.0) < 1e-10,
          "interpParabolic: wide peak ≈ 3, got " + std::to_string(y));
}

void testInterpolateParabolic_unevenSpacing()
{
    // 不等間隔: x values not equally spaced
    // y = 10 - (x-2)^2
    // Uneven x: 1, 2, 4 (not 1, 2, 3)
    // At x=1: 10-1=9, x=2: 10, x=4: 10-4=6
    const double y = interpolateParabolic(1.0, 9.0, 2.0, 10.0, 4.0, 6.0);
    check(std::abs(y - 10.0) < 1e-9,
          "interpParabolic: uneven spacing peak ≈ 10, got " + std::to_string(y));
}

void testInterpolateParabolic_nonQuadratic()
{
    // 完全な二次曲線でない場合でも妥当な値を返す
    // y = 5*sin(x) の近似: (1, 5*sin(1)=4.207), (2, 5*sin(2)=4.546), (3, 5*sin(3)=0.706)
    const double y0 = 5.0 * std::sin(1.0);
    const double y1 = 5.0 * std::sin(2.0);
    const double y2 = 5.0 * std::sin(3.0);
    const double y = interpolateParabolic(1.0, y0, 2.0, y1, 3.0, y2);
    // Should be finite and non-NaN
    check(std::isfinite(y),
          "interpParabolic: non-quadratic finite, got " + std::to_string(y));
}

void testInterpolateParabolic_denomBoundary()
{
    // 分母が非常に小さいケースを検証
    // 3点がほぼ同一直線上 → denom ≈ 0 → y1 を返す
    const double y = interpolateParabolic(1.0, 1.000000000001, 2.0, 1.0, 3.0, 0.999999999999);
    check(std::isfinite(y),
          "interpParabolic: near-zero denom finite, got " + std::to_string(y));
}

//==============================================================================
// TEST GROUP 2: PeakEstimator.estimate (9 cases)
//==============================================================================

void testEstimate_empty()
{
    const auto result = estimate({});
    check(result.interpolatedDb == 0.0f, "estimate: empty returns 0dB");
    check(result.rawSampleIndex == -1, "estimate: empty sampleIndex=-1");
}

void testEstimate_singleSample()
{
    std::vector<MergedSample> samples = {
        {1000.0, 4.0, 10.0, {SampleOrigin::Adaptive, 0}}
    };
    const auto result = estimate(samples);
    check(std::abs(result.rawDb - 12.041) < 0.001,  // 20*log10(4) ≈ 12.041
          "estimate: single sample rawDb ≈ 12.041, got " + std::to_string(result.rawDb));
    check(result.interpolatedDb == result.rawDb,
          "estimate: single sample no interpolation");
    check(result.rawSampleIndex == 0, "estimate: single sample index=0");
}

void testEstimate_twoSamples()
{
    // 2 samples → peak at one edge → no interpolation
    std::vector<MergedSample> samples = {
        {100.0, 1.0, 0.0, {SampleOrigin::Coarse, 0}},
        {1000.0, 4.0, 5.0, {SampleOrigin::Adaptive, 1}}
    };
    const auto result = estimate(samples);
    // peak at index 1 (right edge) → raw=20*log10(4)=12.041, no interpolation
    check(result.rawSampleIndex == 1, "estimate: two samples peak at right edge");
    check(std::abs(result.rawDb - 12.041) < 0.001,
          "estimate: two samples rawDb ≈ 12.041");
    check(result.interpolatedDb == result.rawDb,
          "estimate: two samples no interpolation (edge)");
}

void testEstimate_threeSamples_interpolation()
{
    // 3 samples, peak in middle → should interpolate
    // Peaking around 1000Hz: |H| ≈ 4 at peak
    // Samples: 800Hz (|H|=3), 1000Hz (|H|=4), 1200Hz (|H|=3)
    // In dB: 800Hz=9.542dB, 1000Hz=12.041dB, 1200Hz=9.542dB
    std::vector<MergedSample> samples = {
        {800.0, 3.0, 0.0, {SampleOrigin::Coarse, 0}},
        {1000.0, 4.0, 0.0, {SampleOrigin::Coarse, 1}},
        {1200.0, 3.0, 0.0, {SampleOrigin::Coarse, 2}}
    };
    const auto result = estimate(samples);
    check(result.rawSampleIndex == 1, "estimate: 3-samples peak at center");
    check(std::abs(result.rawDb - 12.041) < 0.001,
          "estimate: 3-samples rawDb ≈ 12.041");
    // Interpolated should be >= raw (parabolic peak)
    check(result.interpolatedDb >= result.rawDb,
          "estimate: interpolatedDb >= rawDb (" +
          std::to_string(result.interpolatedDb) + " >= " + std::to_string(result.rawDb) + ")");
    check(std::isfinite(result.interpolatedDb),
          "estimate: interpolatedDb finite");
    check(std::isfinite(result.interpolatedFreqHz),
          "estimate: interpolatedFreqHz finite");
}

void testEstimate_decreasing_noPeak()
{
    // 単調減少 → 最大値は最初のサンプル（左端）→ 補間なし
    std::vector<MergedSample> samples = {
        {100.0, 5.0, 0.0, {SampleOrigin::Coarse, 0}},
        {500.0, 3.0, 0.0, {SampleOrigin::Coarse, 1}},
        {1000.0, 1.5, 0.0, {SampleOrigin::Coarse, 2}},
        {5000.0, 1.0, 0.0, {SampleOrigin::Coarse, 3}}
    };
    const auto result = estimate(samples);
    check(result.rawSampleIndex == 0, "estimate: decreasing peak at left edge");
    check(result.interpolatedDb == result.rawDb,
          "estimate: decreasing no interpolation");
    check(std::abs(result.rawDb - 13.979) < 0.001,  // 20*log10(5) ≈ 13.979
          "estimate: decreasing rawDb ≈ 13.979, got " + std::to_string(result.rawDb));
}

void testEstimate_increasing_rightEdge()
{
    // 単調増加 → max at right edge → 補間なし
    std::vector<MergedSample> samples = {
        {100.0, 1.0, 0.0, {SampleOrigin::Coarse, 0}},
        {500.0, 2.0, 0.0, {SampleOrigin::Coarse, 1}},
        {1000.0, 4.0, 0.0, {SampleOrigin::Coarse, 2}}
    };
    const auto result = estimate(samples);
    check(result.rawSampleIndex == 2, "estimate: increasing peak at right edge");
    check(result.interpolatedDb == result.rawDb,
          "estimate: increasing no interpolation");
}

void testEstimate_plateau_firstPeak()
{
    // プラトー（同値の最大値）→ 最初の最大値が選択される
    std::vector<MergedSample> samples = {
        {100.0, 4.0, 0.0, {SampleOrigin::Coarse, 0}},
        {500.0, 4.0, 0.0, {SampleOrigin::Coarse, 1}},
        {1000.0, 2.0, 0.0, {SampleOrigin::Coarse, 2}}
    };
    const auto result = estimate(samples);
    check(result.rawSampleIndex == 0,
          "estimate: plateau picks first (index=0), got " + std::to_string(result.rawSampleIndex));
}

void testEstimate_sharpPeak_largeInterpolation()
{
    // 鋭いピーク → 補間の改善が顕著に出る
    // Points: (x0=2.0, y0=-0.5), (x1=2.5, y1=3.0), (x2=3.0, y2=-0.5)
    // In log2 freq space: x values are log2(freq)
    // This simulates a sharp peak in log-frequency + dB space
    const double y0 = 20.0 * std::log10(2.0);    // ~6.02dB
    const double y1 = 20.0 * std::log10(5.0);    // ~13.98dB
    const double y2 = 20.0 * std::log10(3.0);    // ~9.54dB

    std::vector<MergedSample> samples = {
        {std::pow(2.0, 1.0), 2.0, 0.0, {SampleOrigin::Coarse, 0}},  // log2=1.0
        {std::pow(2.0, 2.0), 5.0, 0.0, {SampleOrigin::Coarse, 1}},  // log2=2.0
        {std::pow(2.0, 3.0), 3.0, 0.0, {SampleOrigin::Coarse, 2}}   // log2=3.0
    };
    const auto result = estimate(samples);
    check(result.rawSampleIndex == 1, "estimate: sharp peak at center");
    check(std::isfinite(result.interpolatedDb), "estimate: sharp peak interpolatedDb finite");
    // Interpolated should be >= raw
    check(result.interpolatedDb >= result.rawDb,
          "estimate: sharp peak interpolated >= raw");
}

void testEstimate_twoPeaks_chooseHighest()
{
    // 2つのピーク → 高い方を選択
    std::vector<MergedSample> samples = {
        {100.0, 2.0, 0.0, {SampleOrigin::Coarse, 0}},
        {500.0, 5.0, 0.0, {SampleOrigin::Coarse, 1}},  // 高いピーク (20*log10(5)≈13.98dB)
        {1000.0, 3.0, 0.0, {SampleOrigin::Coarse, 2}},
        {2000.0, 4.0, 0.0, {SampleOrigin::Coarse, 3}},  // 低いピーク
        {5000.0, 2.0, 0.0, {SampleOrigin::Coarse, 4}}
    };
    const auto result = estimate(samples);
    check(result.rawSampleIndex == 1,
          "estimate: two peaks choose highest (index=1), got " + std::to_string(result.rawSampleIndex));
    check(std::abs(result.rawDb - 13.979) < 0.001,
          "estimate: two peaks rawDb ≈ 13.979, got " + std::to_string(result.rawDb));
}

//==============================================================================
// TEST GROUP 3: UpperBoundEstimator.estimateMax (6 cases)
//==============================================================================

void testEstimateMax_empty()
{
    const auto result = estimateMax({});
    check(result.maxDb == 0.0f, "estimateMax: empty returns 0dB");
    check(result.sampleIndex == -1, "estimateMax: empty index=-1");
}

void testEstimateMax_single()
{
    std::vector<MergedSample> samples = {
        {1000.0, 0.0, 15.5, {SampleOrigin::Adaptive, 0}}
    };
    const auto result = estimateMax(samples);
    check(std::abs(result.maxDb - 15.5f) < 0.001f,
          "estimateMax: single returns 15.5, got " + std::to_string(result.maxDb));
    check(result.sampleIndex == 0, "estimateMax: single index=0");
}

void testEstimateMax_ramp()
{
    // 単調増加 → 最後の要素
    std::vector<MergedSample> samples = {
        {100.0, 0.0, 1.0, {SampleOrigin::Coarse, 0}},
        {200.0, 0.0, 2.0, {SampleOrigin::Coarse, 1}},
        {500.0, 0.0, 5.0, {SampleOrigin::Coarse, 2}},
        {1000.0, 0.0, 10.0, {SampleOrigin::Coarse, 3}}
    };
    const auto result = estimateMax(samples);
    check(std::abs(result.maxDb - 10.0f) < 0.001f,
          "estimateMax: ramp max=10, got " + std::to_string(result.maxDb));
    check(result.sampleIndex == 3, "estimateMax: ramp index=3");
    check(std::abs(result.freqHz - 1000.0f) < 0.001f,
          "estimateMax: ramp freq=1000");
}

void testEstimateMax_singlePeak()
{
    // 中央にピーク
    std::vector<MergedSample> samples = {
        {100.0, 0.0, 2.0, {SampleOrigin::Coarse, 0}},
        {500.0, 0.0, 20.0, {SampleOrigin::Coarse, 1}},
        {1000.0, 0.0, 3.0, {SampleOrigin::Coarse, 2}}
    };
    const auto result = estimateMax(samples);
    check(std::abs(result.maxDb - 20.0f) < 0.001f,
          "estimateMax: peak max=20, got " + std::to_string(result.maxDb));
    check(result.sampleIndex == 1, "estimateMax: peak index=1");
    check(std::abs(result.freqHz - 500.0f) < 0.001f,
          "estimateMax: peak freq=500");
}

void testEstimateMax_allSame()
{
    // 全て同じ値 → 最初の要素
    std::vector<MergedSample> samples = {
        {100.0, 0.0, 5.0, {SampleOrigin::Coarse, 0}},
        {200.0, 0.0, 5.0, {SampleOrigin::Coarse, 1}},
        {300.0, 0.0, 5.0, {SampleOrigin::Coarse, 2}}
    };
    const auto result = estimateMax(samples);
    check(std::abs(result.maxDb - 5.0f) < 0.001f,
          "estimateMax: allSame max=5, got " + std::to_string(result.maxDb));
    check(result.sampleIndex == 0, "estimateMax: allSame index=0 (first)");
}

void testEstimateMax_negative()
{
    // 負の値（全てカット）→ 最大値（最小の負）
    std::vector<MergedSample> samples = {
        {100.0, 0.0, -10.0, {SampleOrigin::Coarse, 0}},
        {500.0, 0.0, -3.0, {SampleOrigin::Coarse, 1}},
        {1000.0, 0.0, -8.0, {SampleOrigin::Coarse, 2}}
    };
    const auto result = estimateMax(samples);
    check(std::abs(result.maxDb + 3.0f) < 0.001f,
          "estimateMax: negative max=-3, got " + std::to_string(result.maxDb));
    check(result.sampleIndex == 1,
          "estimateMax: negative index=1");
}

//==============================================================================
// TEST GROUP 4: mergeAndSort (4 cases)
//==============================================================================

void testMergeAndSort_empty()
{
    auto result = mergeAndSort({}, {});
    check(result.empty(), "mergeAndSort: empty returns empty");
}

void testMergeAndSort_nonOverlapping()
{
    // coarse: 500Hz, 1000Hz; adaptive: 750Hz, 1250Hz
    std::vector<MergedSample> coarse = {
        {500.0, 1.0, 0.0, {SampleOrigin::Coarse, 0}},
        {1000.0, 2.0, 0.0, {SampleOrigin::Coarse, 1}}
    };
    std::vector<MergedSample> adaptive = {
        {750.0, 1.5, 0.0, {SampleOrigin::Adaptive, 2}},
        {1250.0, 1.2, 0.0, {SampleOrigin::Adaptive, 3}}
    };
    auto result = mergeAndSort(coarse, adaptive);
    check(result.size() == 4, "mergeAndSort: non-overlapping size=4");
    // All frequencies should be in ascending order
    for (size_t i = 1; i < result.size(); ++i)
        check(result[i-1].freqHz < result[i].freqHz,
              "mergeAndSort: order at index " + std::to_string(i));
}

void testMergeAndSort_interleaved()
{
    // coarse が adaptive より高い周波数 → stable_sort で正順に
    // Input: coarse={1000, 2000}, adaptive={500, 1500}
    // Expected: 500(adaptive), 1000(coarse), 1500(adaptive), 2000(coarse)
    std::vector<MergedSample> coarse = {
        {1000.0, 1.0, 0.0, {SampleOrigin::Coarse, 0}},
        {2000.0, 2.0, 0.0, {SampleOrigin::Coarse, 1}}
    };
    std::vector<MergedSample> adaptive = {
        {500.0, 1.0, 0.0, {SampleOrigin::Adaptive, 2}},
        {1500.0, 1.5, 0.0, {SampleOrigin::Adaptive, 3}}
    };
    auto result = mergeAndSort(coarse, adaptive);
    check(result.size() == 4, "mergeAndSort: interleaved size=4");
    for (size_t i = 1; i < result.size(); ++i)
        check(result[i-1].freqHz < result[i].freqHz,
              "mergeAndSort: interleaved order at index " + std::to_string(i));
    // 先頭が adaptive（周波数500）であることを確認
    check(result[0].origin.type == SampleOrigin::Adaptive,
          "mergeAndSort: first element is adaptive");
}

void testMergeAndSort_stableOrder()
{
    // 同周波数の場合、stable_sort により coarse → adaptive 順が維持される
    std::vector<MergedSample> coarse = {
        {1000.0, 1.0, 0.0, {SampleOrigin::Coarse, 0}}
    };
    std::vector<MergedSample> adaptive = {
        {1000.0, 2.0, 0.0, {SampleOrigin::Adaptive, 1}}
    };
    auto result = mergeAndSort(coarse, adaptive);
    check(result.size() == 2, "mergeAndSort: same freq size=2");
    // stable_sort: coarse が先、adaptive が後
    check(result[0].origin.type == SampleOrigin::Coarse,
          "mergeAndSort: stable coarse first");
    check(result[1].origin.type == SampleOrigin::Adaptive,
          "mergeAndSort: stable adaptive second");
}

//==============================================================================
// TEST GROUP 5: deduplicate (3 cases)
//==============================================================================

void testDeduplicate_noDuplicates()
{
    std::vector<MergedSample> sorted = {
        {100.0, 1.0, 0.0, {SampleOrigin::Coarse, 0}},
        {500.0, 2.0, 0.0, {SampleOrigin::Adaptive, 1}},
        {1000.0, 3.0, 0.0, {SampleOrigin::Coarse, 2}}
    };
    auto result = deduplicate(sorted);
    check(result.size() == 3, "deduplicate: no dupes size=3");
}

void testDeduplicate_adaptiveOverwritesCoarse()
{
    // 同一周波数: coarse が先、adaptive が後 → adaptive で上書き
    std::vector<MergedSample> sorted = {
        {500.0, 1.0, 0.0, {SampleOrigin::Coarse, 0}},
        {500.0, 2.0, 10.0, {SampleOrigin::Adaptive, 1}},
        {1000.0, 3.0, 5.0, {SampleOrigin::Coarse, 2}}
    };
    auto result = deduplicate(sorted);
    check(result.size() == 2, "deduplicate: adaptive overwrite size=2");
    check(result[0].linearMagnitude == 2.0,
          "deduplicate: adaptive magnitude=2, got " + std::to_string(result[0].linearMagnitude));
    check(result[0].origin.type == SampleOrigin::Adaptive,
          "deduplicate: origin type is Adaptive");
}

void testDeduplicate_multipleDupes()
{
    // 複数の重複を含む複合ケース
    std::vector<MergedSample> sorted = {
        {100.0, 1.0, 0.0, {SampleOrigin::Coarse, 0}},
        {100.0, 1.5, 1.0, {SampleOrigin::Adaptive, 1}},  // adaptive overwrites
        {500.0, 2.0, 2.0, {SampleOrigin::Coarse, 2}},
        {500.0, 2.5, 3.0, {SampleOrigin::Adaptive, 3}},  // adaptive overwrites
        {1000.0, 3.0, 4.0, {SampleOrigin::Coarse, 4}}     // unique
    };
    auto result = deduplicate(sorted);
    check(result.size() == 3, "deduplicate: multiple dupes size=3");
    check(result[0].origin.type == SampleOrigin::Adaptive, "deduplicate: 100Hz type=Adaptive");
    check(result[1].origin.type == SampleOrigin::Adaptive, "deduplicate: 500Hz type=Adaptive");
    check(result[2].origin.type == SampleOrigin::Coarse, "deduplicate: 1000Hz type=Coarse");
}

//==============================================================================
// TEST GROUP 6: renumber (3 cases)
//==============================================================================

void testRenumber_sequential()
{
    std::vector<MergedSample> samples = {
        {100.0, 1.0, 0.0, {SampleOrigin::Coarse, 5}},
        {500.0, 2.0, 0.0, {SampleOrigin::Adaptive, 10}},
        {1000.0, 3.0, 0.0, {SampleOrigin::Coarse, 99}}
    };
    renumber(samples);
    check(samples.size() == 3, "renumber: size=3");
    check(samples[0].origin.sampleIndex == 0,
          "renumber: [0].sampleIndex=0, got " + std::to_string(samples[0].origin.sampleIndex));
    check(samples[1].origin.sampleIndex == 1,
          "renumber: [1].sampleIndex=1, got " + std::to_string(samples[1].origin.sampleIndex));
    check(samples[2].origin.sampleIndex == 2,
          "renumber: [2].sampleIndex=2, got " + std::to_string(samples[2].origin.sampleIndex));
}

void testRenumber_singleElement()
{
    std::vector<MergedSample> samples = {
        {1000.0, 1.0, 0.0, {SampleOrigin::Coarse, 42}}
    };
    renumber(samples);
    check(samples.size() == 1, "renumber: single size=1");
    check(samples[0].origin.sampleIndex == 0,
          "renumber: single index=0, got " + std::to_string(samples[0].origin.sampleIndex));
}

void testRenumber_empty()
{
    std::vector<MergedSample> samples;
    renumber(samples);
    check(samples.empty(), "renumber: empty stays empty");
}

//==============================================================================
// TEST GROUP 7: findGlobalPeak (4 cases)
//==============================================================================

void testFindGlobalPeak_empty()
{
    const int idx = findGlobalPeak({});
    check(idx == -1, "findGlobalPeak: empty returns -1");
}

void testFindGlobalPeak_first()
{
    std::vector<MergedSample> samples = {
        {100.0, 5.0, 0.0, {SampleOrigin::Coarse, 0}},
        {500.0, 3.0, 0.0, {SampleOrigin::Coarse, 1}},
        {1000.0, 2.0, 0.0, {SampleOrigin::Coarse, 2}}
    };
    const int idx = findGlobalPeak(samples);
    check(idx == 0, "findGlobalPeak: first at index 0");
}

void testFindGlobalPeak_last()
{
    std::vector<MergedSample> samples = {
        {100.0, 1.0, 0.0, {SampleOrigin::Coarse, 0}},
        {500.0, 2.0, 0.0, {SampleOrigin::Coarse, 1}},
        {1000.0, 10.0, 0.0, {SampleOrigin::Coarse, 2}}
    };
    const int idx = findGlobalPeak(samples);
    check(idx == 2, "findGlobalPeak: last at index 2");
}

void testFindGlobalPeak_middle()
{
    std::vector<MergedSample> samples = {
        {100.0, 1.0, 0.0, {SampleOrigin::Coarse, 0}},
        {500.0, 10.0, 0.0, {SampleOrigin::Coarse, 1}},
        {1000.0, 2.0, 0.0, {SampleOrigin::Coarse, 2}}
    };
    const int idx = findGlobalPeak(samples);
    check(idx == 1, "findGlobalPeak: middle at index 1");
}

//==============================================================================
// TEST GROUP 8: linearToDb / dbToLinear roundtrip (4 cases)
//==============================================================================

void testLinearToDb_unit()
{
    check(std::abs(linearToDb(1.0)) < 1e-12, "linearToDb: 1.0 → 0dB");
}

void testLinearToDb_positive()
{
    // +12dB → linear ≈ 3.981
    const double lin = dbToLinear(12.0);
    const double db = linearToDb(lin);
    check(std::abs(db - 12.0) < 1e-9, "linearToDb: roundtrip +12dB, got " + std::to_string(db));
}

void testLinearToDb_negative()
{
    const double lin = dbToLinear(-6.0);
    const double db = linearToDb(lin);
    check(std::abs(db + 6.0) < 1e-9, "linearToDb: roundtrip -6dB, got " + std::to_string(db));
}

void testLinearToDb_zero()
{
    const double db = linearToDb(0.0);
    check(db <= -1e100, "linearToDb: 0.0 → -inf, got " + std::to_string(db));
}

//==============================================================================
// TEST GROUP 9: isBoostingBand (4 cases)
//==============================================================================

void testIsBoosting_peakingPositive()
{
    check(isBoostingBand(EQBandType::Peaking, 3.0f) == true,
          "isBoosting: Peaking +3dB = true");
}

void testIsBoosting_peakingNegative()
{
    check(isBoostingBand(EQBandType::Peaking, -3.0f) == false,
          "isBoosting: Peaking -3dB = false");
}

void testIsBoosting_lowPass()
{
    check(isBoostingBand(EQBandType::LowPass, 12.0f) == false,
          "isBoosting: LowPass +12dB = false");
}

void testIsBoosting_boundary()
{
    check(isBoostingBand(EQBandType::Peaking, 0.01f) == false,
          "isBoosting: Peaking +0.01dB boundary = false");
    check(isBoostingBand(EQBandType::Peaking, 0.010001f) == true,
          "isBoosting: Peaking +0.010001dB = true (just above threshold)");
}

//==============================================================================
// TEST GROUP 10: computeSampleResponse (3 cases)
//==============================================================================

void testComputeSampleResponse_bypass()
{
    // バイパス: linearMag=1, upperBound=0
    BandInfo band{0, 1000.0, 1.0, EQBandType::Peaking, 0.0f, {1.0,0.0,0.0,1.0,0.0,0.0}, false};
    double mag = 0.0, ub = 999.0;
    computeSampleResponse(&band, 1, 0.1, false, mag, ub);
    check(std::abs(mag - 1.0) < 1e-12,
          "computeSampleResponse: bypass serial mag=1, got " + std::to_string(mag));
    check(std::abs(ub) < 1e-12,
          "computeSampleResponse: bypass serial ub=0, got " + std::to_string(ub));
}

void testComputeSampleResponse_serial()
{
    // 2-band serial with positive gain
    // Band1: Peaking +6dB → gain_linear ≈ 2.0
    // Band2: Peaking +6dB → gain_linear ≈ 2.0
    // At resonance: |H_total| ≈ 4.0
    const double gainLin = dbToLinear(6.0); // ≈ 2.0
    // H(z) = G*z^2 / z^2 = G (constant magnitude at all frequencies)
    const BandInfo bandArr[] = {
        {0, 1000.0, 1.0, EQBandType::Peaking, 6.0f,
         {gainLin, 0.0, 0.0, 1.0, 0.0, 0.0}, true},
        {1, 1000.0, 1.0, EQBandType::Peaking, 6.0f,
         {gainLin, 0.0, 0.0, 1.0, 0.0, 0.0}, true}
    };
    double mag = 0.0, ub = 0.0;
    computeSampleResponse(bandArr, 2, 0.1, false, mag, ub);
    check(std::abs(mag - gainLin * gainLin) < 1e-9,
          "computeSampleResponse: serial product mag=" + std::to_string(mag) +
          " expected=" + std::to_string(gainLin * gainLin));
    check(std::isfinite(ub), "computeSampleResponse: serial ub finite");
    check(ub > 0.0, "computeSampleResponse: serial ub > 0");
}

void testComputeSampleResponse_parallel()
{
    // 2-band parallel: H_total = 1 + (H1-1) + (H2-1) = H1 + H2 - 1
    // H1=2, H2=2 → H_total=3 (constant magnitude filter)
    const BandInfo bandArr2[] = {
        {0, 1000.0, 1.0, EQBandType::Peaking, 6.0f,
         {2.0, 0.0, 0.0, 1.0, 0.0, 0.0}, true},
        {1, 1000.0, 1.0, EQBandType::Peaking, 6.0f,
         {2.0, 0.0, 0.0, 1.0, 0.0, 0.0}, true}
    };
    double mag = 0.0, ub = 0.0;
    computeSampleResponse(bandArr2, 2, 0.1, true, mag, ub);
    check(std::abs(mag - 3.0) < 1e-9,
          "computeSampleResponse: parallel sum mag=3, got " + std::to_string(mag));
}

//==============================================================================
// MAIN
//==============================================================================
} // namespace

int main()
{
    std::cout << "[EQAnalysisUnitTests] Start — 53 tests\n";

    // Group 1: interpolateParabolic (10)
    testInterpolateParabolic_symmetric();
    testInterpolateParabolic_leftShifted();
    testInterpolateParabolic_rightShifted();
    testInterpolateParabolic_flat();
    testInterpolateParabolic_ascending();
    testInterpolateParabolic_narrow();
    testInterpolateParabolic_wide();
    testInterpolateParabolic_unevenSpacing();
    testInterpolateParabolic_nonQuadratic();
    testInterpolateParabolic_denomBoundary();

    // Group 2: estimate (9)
    testEstimate_empty();
    testEstimate_singleSample();
    testEstimate_twoSamples();
    testEstimate_threeSamples_interpolation();
    testEstimate_decreasing_noPeak();
    testEstimate_increasing_rightEdge();
    testEstimate_plateau_firstPeak();
    testEstimate_sharpPeak_largeInterpolation();
    testEstimate_twoPeaks_chooseHighest();

    // Group 3: estimateMax (6)
    testEstimateMax_empty();
    testEstimateMax_single();
    testEstimateMax_ramp();
    testEstimateMax_singlePeak();
    testEstimateMax_allSame();
    testEstimateMax_negative();

    // Group 4: mergeAndSort (4)
    testMergeAndSort_empty();
    testMergeAndSort_nonOverlapping();
    testMergeAndSort_interleaved();
    testMergeAndSort_stableOrder();

    // Group 5: deduplicate (3)
    testDeduplicate_noDuplicates();
    testDeduplicate_adaptiveOverwritesCoarse();
    testDeduplicate_multipleDupes();

    // Group 6: renumber (3)
    testRenumber_sequential();
    testRenumber_singleElement();
    testRenumber_empty();

    // Group 7: findGlobalPeak (4)
    testFindGlobalPeak_empty();
    testFindGlobalPeak_first();
    testFindGlobalPeak_last();
    testFindGlobalPeak_middle();

    // Group 8: linearToDb/dbToLinear (4)
    testLinearToDb_unit();
    testLinearToDb_positive();
    testLinearToDb_negative();
    testLinearToDb_zero();

    // Group 9: isBoostingBand (4)
    testIsBoosting_peakingPositive();
    testIsBoosting_peakingNegative();
    testIsBoosting_lowPass();
    testIsBoosting_boundary();

    // Group 10: computeSampleResponse (3)
    testComputeSampleResponse_bypass();
    testComputeSampleResponse_serial();
    testComputeSampleResponse_parallel();

    // Group 11: biquadResponse (3) — quick sanity checks
    // (Already tested in EQProcessorMaxGainTests, just basic here)
    {
        EQCoeffsBiquad bypass{1.0, 0.0, 0.0, 1.0, 0.0, 0.0};
        const auto H = biquadResponse(bypass, 0.0);
        check(std::abs(H.real() - 1.0) < 1e-12, "biquadResponse: bypass DC real=1");
        check(std::abs(H.imag()) < 1e-12, "biquadResponse: bypass DC imag=0");
        check(std::abs(std::abs(H) - 1.0) < 1e-12, "biquadResponse: bypass DC mag=1");
    }

    const int total = g_testsPassed + g_testsFailed;
    std::cout << "[EQAnalysisUnitTests] Passed: " << g_testsPassed
              << ", Failed: " << g_testsFailed
              << " (Total: " << total << "/53)\n";
    return (g_testsFailed == 0) ? 0 : 1;
}
