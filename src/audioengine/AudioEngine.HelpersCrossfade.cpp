#include <immintrin.h>
#include <JuceHeader.h>

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_HELPERS_CROSSFADE)

//====================================================
// AVX2 クロスフェード（double）
//====================================================
[[maybe_unused]] static inline void crossfadeAVX2(
    double* dstL, double* dstR,
    const double* newL, const double* newR,
    const double* oldL, const double* oldR,
    int numSamples,
    double gStart,
    double gStep)
{
    int i = 0;
    for (; i + 3 < numSamples; i += 4)
    {
        __m256d g = _mm256_set_pd(
            gStart + gStep * (i + 3),
            gStart + gStep * (i + 2),
            gStart + gStep * (i + 1),
            gStart + gStep * (i + 0)
        );
        __m256d gOld = _mm256_sub_pd(_mm256_set1_pd(1.0), g);
        __m256d nL = _mm256_loadu_pd(newL + i);
        __m256d oL = _mm256_loadu_pd(oldL + i);
        __m256d nR = _mm256_loadu_pd(newR + i);
        __m256d oR = _mm256_loadu_pd(oldR + i);
        __m256d outL = _mm256_add_pd(_mm256_mul_pd(nL, g), _mm256_mul_pd(oL, gOld));
        __m256d outR = _mm256_add_pd(_mm256_mul_pd(nR, g), _mm256_mul_pd(oR, gOld));
        _mm256_storeu_pd(dstL + i, outL);
        _mm256_storeu_pd(dstR + i, outR);
    }
    for (; i < numSamples; ++i)
    {
        double g = gStart + gStep * i;
        double gOld = 1.0 - g;
        dstL[i] = newL[i] * g + oldL[i] * gOld;
        dstR[i] = newR[i] * g + oldR[i] * gOld;
    }
}

//==============================================================================
// 等電力クロスフェード用近似関数（Audio Thread安全・libm不使用）
//==============================================================================
[[maybe_unused]] static inline float equalPowerSinFloat(float x) noexcept
{
    x = juce::jlimit(0.0f, 1.0f, x);
    const float t = x * 1.5707963267948966f;  // pi/2
    const float t2 = t * t;
    return t * (1.0f + t2 * (-1.0f/6.0f + t2 * (1.0f/120.0f
             + t2 * (-1.0f/5040.0f + t2 * (1.0f/362880.0f)))));
}

[[maybe_unused]] static inline double equalPowerSinDouble(double x) noexcept
{
    x = juce::jlimit(0.0, 1.0, x);
    const double t = x * 1.5707963267948966;
    const double t2 = t * t;
    return t * (1.0 + t2 * (-1.0/6.0 + t2 * (1.0/120.0
             + t2 * (-1.0/5040.0 + t2 * (1.0/362880.0)))));
}

#endif // defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_HELPERS_CROSSFADE)
