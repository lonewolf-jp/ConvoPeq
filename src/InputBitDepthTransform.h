//==============================================================================
#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>

#if JUCE_INTEL
 #include <immintrin.h>
#endif

#if JUCE_DSP_USE_INTEL_MKL
 #include <mkl.h>
#endif

namespace convo::input_transform
{
    // document/input_bitdepth_transform.txt の仕様に合わせた共通定数
    static constexpr double kHeadroomScale    = 0.988553; // about -0.1 dB
    static constexpr double kDenormThreshold  = 1.0e-25;

    inline void sanitizeAndLimit(double* data, int numSamples) noexcept
    {
        for (int i = 0; i < numSamples; ++i)
        {
            double v = data[i];
            if (!std::isfinite(v) || std::abs(v) < kDenormThreshold)
                v = 0.0;
            data[i] = std::clamp(v, -1.0, 1.0);
        }
    }

    inline void applyHighQuality64BitTransform(double* data, int numSamples) noexcept
    {
        if (data == nullptr || numSamples <= 0)
            return;

        // 1) ヘッドルーム確保 (-0.1dB)
#if JUCE_DSP_USE_INTEL_MKL
        cblas_dscal(numSamples, kHeadroomScale, data, 1);
#else
        for (int i = 0; i < numSamples; ++i)
            data[i] *= kHeadroomScale;
#endif

        // 2) デノーマル/NaN対策 + 範囲制限
        sanitizeAndLimit(data, numSamples);
    }

    inline void convertFloatToDoubleHighQuality(const float* src, double* dst, int numSamples) noexcept
    {
        if (src == nullptr || dst == nullptr || numSamples <= 0)
            return;

#if defined(__AVX2__)
        int i = 0;
        const int vEnd = (numSamples / 8) * 8;
        for (; i < vEnd; i += 8)
        {
            const __m256 vf = _mm256_loadu_ps(src + i);
            const __m128 lo = _mm256_castps256_ps128(vf);
            const __m128 hi = _mm256_extractf128_ps(vf, 1);
            _mm256_storeu_pd(dst + i,     _mm256_cvtps_pd(lo));
            _mm256_storeu_pd(dst + i + 4, _mm256_cvtps_pd(hi));
        }
        for (; i < numSamples; ++i)
            dst[i] = static_cast<double>(src[i]);
#else
        for (int i = 0; i < numSamples; ++i)
            dst[i] = static_cast<double>(src[i]);
#endif

        applyHighQuality64BitTransform(dst, numSamples);
    }

    inline void convertDoubleToDoubleHighQuality(const double* src, double* dst, int numSamples) noexcept
    {
        if (src == nullptr || dst == nullptr || numSamples <= 0)
            return;

        if (src != dst)
            std::memcpy(dst, src, static_cast<size_t>(numSamples) * sizeof(double));

        applyHighQuality64BitTransform(dst, numSamples);
    }
}
