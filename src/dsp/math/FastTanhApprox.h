#pragma once

#include <immintrin.h>
#include <cstdint>
#include <type_traits>

//==============================================================================
// FastTanhApprox — Tanh 近似の共通ユーティリティ
//
// ★ ISR Runtime 準拠（Single Semantic Source）:
//   DSPCoreDouble（SoftClip）・EQProcessor（Saturation）のすべてが
//   同一実装を参照する。係数と閾値は Policy テンプレートで注入し、
//   将来の独立チューニングに備える。
//
// 使用方法:
//   double y = convo::dsp::fastTanh<SoftClipPolicy>(x);
//   __m128d yv = convo::dsp::fastTanhV128<EQSaturationPolicy>(xv);
//
//==============================================================================

namespace convo::dsp {

//==============================================================================
// デフォルトポリシー — 現行コードの 27/9 係数 + 閾値 4.5 を維持
//   f(x) = x * (27 + x²) / (27 + 9*x²)
//   3次/2次 Padé近似。x=3 で厳密に 1.0 に収束。
//==============================================================================
struct DefaultFastTanhPolicy {
    static constexpr double clipThreshold = 4.5;

    // ★ R-3: PascalCase 定数（fastTanhV256 用）
    //   27/9 Padé: x*(27 + x²) / (27 + 9*x²)
    //   = x*(27 + x²*1 + 0*x⁴) / (27 + x²*9 + 0*x⁴ + 0*x⁶)
    static constexpr double ClipThreshold = 4.5;
    static constexpr double NumA = 27.0;
    static constexpr double NumB = 1.0;
    static constexpr double NumC = 0.0;
    static constexpr double DenA = 27.0;
    static constexpr double DenB = 9.0;
    static constexpr double DenC = 0.0;

    // ★ 各 Policy は static compute(x, x2) を提供する。
    //    これにより係数構造に依存しない任意の多項式を表現可能。
    [[nodiscard]] static double compute(double x, double x2) noexcept {
        return x * (27.0 + x2) / (27.0 + 9.0 * x2);
    }

    // SSE2 版: __m128d を取るオーバーロード
    [[nodiscard]] static __m128d compute(__m128d x, __m128d x2) noexcept {
        const auto vNine = _mm_set1_pd(9.0);
        const auto vTwentySeven = _mm_set1_pd(27.0);
        const auto num = _mm_mul_pd(x, _mm_add_pd(vTwentySeven, x2));
        const auto den = _mm_add_pd(vTwentySeven, _mm_mul_pd(vNine, x2));
        return _mm_div_pd(num, den);
    }
};

//==============================================================================
// 高次 Padé 近似ポリシー — DSPCoreDouble の SoftClip 用（10395 係数）
//   f(x) = x*(10395 + x²*(1260 + 21*x²)) / (10395 + x²*(4725 + x²*(210 + x²)))
//   5次/6次 Padé近似。x=4.5 で約 0.99927 に収束。
//==============================================================================
struct SoftClipPadéPolicy {
    static constexpr double clipThreshold = 4.5;

    // ★ R-3: PascalCase 定数（fastTanhV256 用）
    //   Policy は係数と閾値のみを保持するデータ構造。
    //   SIMD 演算は FastTanhApprox 側の fastTanhV128<Policy>() / fastTanhV256<Policy>()
    //   がこれらの定数を読み込んで実行する。
    static constexpr double ClipThreshold = 4.5;
    static constexpr double NumA = 10395.0;
    static constexpr double NumB = 1260.0;
    static constexpr double NumC = 21.0;
    static constexpr double DenA = 10395.0;
    static constexpr double DenB = 4725.0;
    static constexpr double DenC = 210.0;

    [[nodiscard]] static double compute(double x, double x2) noexcept {
        const double num = x * (10395.0 + x2 * (1260.0 + x2 * 21.0));
        const double den = 10395.0 + x2 * (4725.0 + x2 * (210.0 + x2));
        return num / den;
    }

    [[nodiscard]] static __m128d compute(__m128d x, __m128d x2) noexcept {
        const auto v10395 = _mm_set1_pd(10395.0);
        const auto v1260  = _mm_set1_pd(1260.0);
        const auto v21    = _mm_set1_pd(21.0);
        const auto v4725  = _mm_set1_pd(4725.0);
        const auto v210   = _mm_set1_pd(210.0);
        const auto num = _mm_mul_pd(x, _mm_add_pd(v10395,
            _mm_mul_pd(x2, _mm_add_pd(v1260, _mm_mul_pd(x2, v21)))));
        const auto den = _mm_add_pd(v10395,
            _mm_mul_pd(x2, _mm_add_pd(v4725, _mm_mul_pd(x2, _mm_add_pd(v210, x2)))));
        return _mm_div_pd(num, den);
    }
};

//==============================================================================
// fastTanh — Policy ベース Scalar 版
//==============================================================================
template<class Policy = DefaultFastTanhPolicy>
[[nodiscard]] inline double fastTanh(double x) noexcept
{
    if (x >= Policy::clipThreshold) return 1.0;
    if (x <= -Policy::clipThreshold) return -1.0;
    return Policy::compute(x, x * x);
}

//==============================================================================
// fastTanhV128 — Policy ベース SSE2 版
//==============================================================================
template<class Policy = DefaultFastTanhPolicy>
[[nodiscard]] inline __m128d fastTanhV128(__m128d x) noexcept
{
    const auto vClipHigh = _mm_set1_pd(Policy::clipThreshold);
    const auto vClipLow  = _mm_set1_pd(-Policy::clipThreshold);
    const auto xClamped = _mm_min_pd(_mm_max_pd(x, vClipLow), vClipHigh);
    return Policy::compute(xClamped, _mm_mul_pd(xClamped, xClamped));
}

//==============================================================================
// ★ Policy 要件検証用ヘルパー
//   Policy は以下の static constexpr メンバを提供しなければならない:
//     ClipThreshold, NumA, NumB, NumC, DenA, DenB, DenC
//==============================================================================
namespace detail {
template<class P, class = void>
struct has_fast_tanh_policy_constants : std::false_type {};

template<class P>
struct has_fast_tanh_policy_constants<P, std::void_t<
    decltype(P::ClipThreshold),
    decltype(P::NumA), decltype(P::NumB), decltype(P::NumC),
    decltype(P::DenA), decltype(P::DenB), decltype(P::DenC)
>> : std::true_type {};
} // namespace detail

//==============================================================================
// fastTanhV256 — Policy ベース AVX2 版
//==============================================================================
#if defined(__AVX2__) || defined(__FMA__)
template<class Policy = DefaultFastTanhPolicy>
    requires detail::has_fast_tanh_policy_constants<Policy>::value
[[nodiscard]] inline __m256d fastTanhV256(__m256d x) noexcept
{
    const auto vClipHigh = _mm256_set1_pd(Policy::ClipThreshold);
    const auto vClipLow  = _mm256_set1_pd(-Policy::ClipThreshold);
    const auto xClamped = _mm256_min_pd(_mm256_max_pd(x, vClipLow), vClipHigh);
    const auto x2 = _mm256_mul_pd(xClamped, xClamped);

    // ★ Policy が直接保持する static constexpr 係数を参照
    //   （Coefficients 構造体のネストを避け、constexpr 最適化を促進）
    const auto vNumA = _mm256_set1_pd(Policy::NumA);
    const auto vNumB = _mm256_set1_pd(Policy::NumB);
    const auto vNumC = _mm256_set1_pd(Policy::NumC);
    const auto vDenA = _mm256_set1_pd(Policy::DenA);
    const auto vDenB = _mm256_set1_pd(Policy::DenB);
    const auto vDenC = _mm256_set1_pd(Policy::DenC);

    const auto num = _mm256_mul_pd(xClamped, _mm256_add_pd(vNumA,
        _mm256_mul_pd(x2, _mm256_add_pd(vNumB, _mm256_mul_pd(x2, vNumC)))));
    const auto den = _mm256_add_pd(vDenA,
        _mm256_mul_pd(x2, _mm256_add_pd(vDenB,
            _mm256_mul_pd(x2, _mm256_add_pd(vDenC, x2)))));
    return _mm256_div_pd(num, den);
}
#endif

} // namespace convo::dsp
