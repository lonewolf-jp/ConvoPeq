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
    constexpr uint64_t kThresholdBits = convo::numeric_policy::denormThresholdBitsDouble();
    union { double d; uint64_t u; } v { x };
    constexpr uint64_t kSignMask = 0x8000000000000000ULL;
    uint64_t absBits = v.u & ~kSignMask;
    v.u = (absBits < kThresholdBits) ? 0ULL : v.u;
    return v.d;
}

inline float killDenormal(float x) noexcept
{
    constexpr uint32_t kThresholdBits = convo::numeric_policy::denormThresholdBitsFloat();
    union { float f; uint32_t u; } v { x };
    constexpr uint32_t kSignMask = 0x80000000U;
    uint32_t absBits = v.u & ~kSignMask;
    v.u = (absBits < kThresholdBits) ? 0U : v.u;
    return v.f;
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
