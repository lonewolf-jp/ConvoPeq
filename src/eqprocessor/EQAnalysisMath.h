#pragma once

#include "EQProcessor.h"
#include "EQAnalysisTypes.h"
#include <complex>
#include <cmath>
#include <limits>

//==============================================================================
// EQAnalysisMath — EQ 解析で使用する共有数学関数群
//
// 設計方針:
// - すべての関数は純粋関数（inline 自由関数）
// - biquadResponse を1度だけ評価し measured + upperBound を同時算出
//
// v14.47: computeEstimatedMaxGainComplex() リファクタリング
//==============================================================================

namespace EQAnalysisMath {

//==============================================================================
// Biquad 周波数応答 H(e^{jω})
//==============================================================================

inline std::complex<double> biquadResponse(const EQCoeffsBiquad& c, double w) noexcept
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

//==============================================================================
// 1評価点の measured + upperBound を同時計算
// ★ biquadResponse を1度だけ評価し両方を同時算出（二重評価防止）
//==============================================================================

inline void computeSampleResponse(
    const BandInfo* bands, size_t numBands,
    double normalizedFreq, bool isParallel,
    double& outLinearMagnitude, double& outUpperBoundDb) noexcept
{
    constexpr double kTwentyOverLog10 = 8.685889638065036; // 20.0 / ln(10)
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
// ゲイン変換
//==============================================================================

inline double linearToDb(double linear) noexcept {
    return (linear > 1e-18) ? 20.0 * std::log10(linear)
                            : -std::numeric_limits<double>::infinity();
}

inline double dbToLinear(double db) noexcept {
    return std::pow(10.0, db / 20.0);
}

//==============================================================================
// バンド判定
//==============================================================================

inline bool isBoostingBand(EQBandType type, float gain) noexcept
{
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

} // namespace EQAnalysisMath
