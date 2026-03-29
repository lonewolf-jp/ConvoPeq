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
