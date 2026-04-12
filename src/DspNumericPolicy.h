//==============================================================================
#pragma once

#include <bit>      // std::bit_cast (C++20)
#include <cstdint>  // uint64_t, uint32_t
#include <immintrin.h>

namespace convo::numeric_policy
{
    // ─────────────────────────────────────────────────────────────
    // デノーマル除去のしきい値（一箇所で定義）
    // ─────────────────────────────────────────────────────────────
    inline constexpr double kDenormThresholdDouble = 1.0e-20;
    inline constexpr float  kDenormThresholdFloat  = 1.0e-20f;

    // Audio Thread 内でのデノーマル除去しきい値 (Double と同じ値を使用)
    // 使用箇所: OutputFilter, MKLNonUniformConvolver, EQProcessor, CustomInputOversampler,
    //           PsychoacousticDither, UltraHighRateDCBlocker など
    inline constexpr double kDenormThresholdAudioState = kDenormThresholdDouble;

    // 入力サニタイズ用のデノーマルしきい値 (Double と同じ値を使用)
    inline constexpr double kDenormThresholdInputSanitize = kDenormThresholdDouble;

    // ─────────────────────────────────────────────────────────────
    // しきい値に対応するビットパターンを constexpr で取得（C++20）
    // ─────────────────────────────────────────────────────────────
    inline constexpr uint64_t denormThresholdBitsDouble() noexcept
    {
        return std::bit_cast<uint64_t>(kDenormThresholdDouble);
    }

    inline constexpr uint32_t denormThresholdBitsFloat() noexcept
    {
        return std::bit_cast<uint32_t>(kDenormThresholdFloat);
    }
}

// ─────────────────────────────────────────────────────────────────
// デノーマル除去ヘルパー関数（libm 非依存・分岐レス）
// ─────────────────────────────────────────────────────────────────

inline double killDenormal(double x) noexcept
{
    constexpr uint64_t kExpMask = 0x7FF0000000000000ULL;
    constexpr uint64_t kFracMask = 0x000FFFFFFFFFFFFFULL;

    const uint64_t bits = std::bit_cast<uint64_t>(x);
    const bool isSubnormal = ((bits & kExpMask) == 0ULL) && ((bits & kFracMask) != 0ULL);
    return isSubnormal ? 0.0 : x;
}

inline float killDenormal(float x) noexcept
{
    constexpr uint32_t kExpMask = 0x7F800000U;
    constexpr uint32_t kFracMask = 0x007FFFFFU;

    const uint32_t bits = std::bit_cast<uint32_t>(x);
    const bool isSubnormal = ((bits & kExpMask) == 0U) && ((bits & kFracMask) != 0U);
    return isSubnormal ? 0.0f : x;
}

inline double saturateAVX2(double x, double minVal, double maxVal) noexcept
{
#if defined(__AVX2__) || defined(_M_AVX2)
    __m128d vx = _mm_load_sd(&x);
    __m128d vMin = _mm_load_sd(&minVal);
    __m128d vMax = _mm_load_sd(&maxVal);
    vx = _mm_max_sd(vx, vMin);
    vx = _mm_min_sd(vx, vMax);
    return _mm_cvtsd_f64(vx);
#else
    if (x < minVal)
        return minVal;
    if (x > maxVal)
        return maxVal;
    return x;
#endif
}

#if defined(__AVX2__)
inline __m256d killDenormalV(__m256d v) noexcept
{
    constexpr double kDenormThreshold = convo::numeric_policy::kDenormThresholdDouble;
    const __m256d vThreshold = _mm256_set1_pd(kDenormThreshold);
    const __m256d vSignMask = _mm256_set1_pd(-0.0);
    __m256d vAbs = _mm256_andnot_pd(vSignMask, v);
    __m256d vMask = _mm256_cmp_pd(vAbs, vThreshold, _CMP_GE_OQ);
    return _mm256_and_pd(v, vMask);
}

inline __m128d killDenormalV(__m128d v) noexcept
{
    constexpr double kDenormThreshold = convo::numeric_policy::kDenormThresholdDouble;
    const __m128d vThreshold = _mm_set1_pd(kDenormThreshold);
    const __m128d vSignMask = _mm_set1_pd(-0.0);
    __m128d vAbs = _mm_andnot_pd(vSignMask, v);
    __m128d vMask = _mm_cmp_pd(vAbs, vThreshold, _CMP_GE_OQ);
    return _mm_and_pd(v, vMask);
}
#endif

// ─────────────────────────────────────────────────────────────────
// LinearRamp — juce::SmoothedValue<double> の線形モード代替実装
//
// juce::SmoothedValue<double> と同一セマンティクスを持つが、
// JUCE 内部の仮想関数・テンプレート依存を排除した軽量実装。
//
// スレッド安全性:
//   reset() / setCurrentAndTargetValue() は prepareToPlay 等の
//   非 Audio Thread から呼ぶこと（既存 SmoothedValue と同じ規約）。
//   setTargetValue() / getNextValue() / isSmoothing() は
//   Audio Thread のみから呼ぶこと。
// ─────────────────────────────────────────────────────────────────
namespace convo {

struct LinearRamp
{
    double current    = 0.0;
    double target     = 0.0;
    double step       = 0.0;
    int    remaining  = 0;
    int    totalSteps = 1;  ///< reset() で確定する定数（Audio Thread 中は不変）

    explicit LinearRamp(double initialValue = 0.0) noexcept
        : current(initialValue), target(initialValue) {}

    /// prepareToPlay 等の非 Audio Thread から呼ぶ。ランプ長をサンプル数で確定する。
    void reset(double sampleRate, double timeSec) noexcept
    {
        const int steps = static_cast<int>(sampleRate * timeSec + 0.5);
        totalSteps = (steps > 0) ? steps : 1;
    }

    /// current と target を同じ値に設定し、ランプを無効化する。
    void setCurrentAndTargetValue(double v) noexcept
    {
        current = target = v;
        step      = 0.0;
        remaining = 0;
    }

    /// 目標値を設定してランプを開始する。Audio Thread からのみ呼ぶこと。
    /// juce::SmoothedValue と同一セマンティクス:
    ///   ランプ中は残りステップ数、停止中は totalSteps を分母に使用。
    void setTargetValue(double v) noexcept
    {
        if (v == target) return;
        target = v;
        const int steps = (remaining > 0) ? remaining : totalSteps;
        step      = (target - current) / static_cast<double>(steps);
        remaining = steps;
    }

    /// 1 サンプル進めて新しい current を返す。Audio Thread のみ。
    inline double getNextValue() noexcept
    {
        if (remaining <= 0)
            return current;
        current += step;
        if (--remaining <= 0)
            current = target;  // 最終ステップで浮動小数点誤差なく収束
        return current;
    }

    double getCurrentValue() const noexcept { return current; }
    double getTargetValue()  const noexcept { return target;  }
    bool   isSmoothing()     const noexcept { return remaining > 0; }
};

} // namespace convo
