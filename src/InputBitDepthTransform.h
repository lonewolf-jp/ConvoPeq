//==============================================================================
#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>

#include "DspNumericPolicy.h"

 #include <immintrin.h>

 #include <mkl.h>

namespace convo::input_transform
{
    // document/input_bitdepth_transform.txt の仕様に合わせた共通定数
    static constexpr double kHeadroomScale    = 0.988553; // about -0.1 dB
    static constexpr double kDenormThreshold  = convo::numeric_policy::kDenormThresholdInputSanitize;

    inline bool isFiniteAndAboveThresholdMask(double value, double threshold) noexcept
    {
        const __m128d v = _mm_set1_pd(value);
        const __m128d diff = _mm_sub_pd(v, v);
        const __m128d finiteMask = _mm_cmpeq_pd(diff, _mm_setzero_pd());

        const __m128d signMask = _mm_set1_pd(-0.0);
        const __m128d absV = _mm_andnot_pd(signMask, v);
        const __m128d thresholdV = _mm_set1_pd(threshold);
        const __m128d denormalMask = _mm_cmplt_pd(absV, thresholdV);

        const __m128d validMask = _mm_andnot_pd(denormalMask, finiteMask);
        return _mm_movemask_pd(validMask) == 0x3;
    }

    inline void sanitizeAndLimit(double* __restrict data, int numSamples) noexcept
    {
        // data は ScopedAlignedPtr<double> 由来なので 64byte アライン保証
        const __m256d vMax     = _mm256_set1_pd(1.0);
        const __m256d vMin     = _mm256_set1_pd(-1.0);
        const __m256d vZero    = _mm256_setzero_pd();
        const __m256d vThresh  = _mm256_set1_pd(kDenormThreshold);
        const __m256d vSignMask = _mm256_set1_pd(-0.0); // 符号ビットマスク

        int i = 0;
        const int vEnd = numSamples / 4 * 4;
        for (; i < vEnd; i += 4)
        {
            __m256d v = _mm256_load_pd(data + i);           // アライン保証→loadu→load に変更可能

            // NaN/Inf チェック: x == x は NaN 以外で true (ORD = Ordered predicate)
            __m256d nanMask     = _mm256_cmp_pd(v, v, _CMP_ORD_Q);

            // |v| < kDenormThreshold チェック
            __m256d vAbs        = _mm256_andnot_pd(vSignMask, v);          // |v|
            __m256d normMask    = _mm256_cmp_pd(vAbs, vThresh, _CMP_GE_OQ); // |v| >= thresh

            // NaN かつデノーマル未満なら 0 に
            __m256d validMask   = _mm256_and_pd(nanMask, normMask);
            v = _mm256_and_pd(v, validMask);                // 無効サンプル→0.0

            // [-1, 1] クランプ
            v = _mm256_min_pd(_mm256_max_pd(v, vMin), vMax);
            _mm256_store_pd(data + i, v);
        }
        for (; i < numSamples; ++i)
        {
            double v = data[i];
            if (!isFiniteAndAboveThresholdMask(v, kDenormThreshold)) v = 0.0;
            data[i] = std::clamp(v, -1.0, 1.0);
        }
    }

    inline void applyHighQuality64BitTransform(double* data, int numSamples, double gain = 1.0) noexcept
    {
        if (data == nullptr || numSamples <= 0)
            return;

        // 1) ヘッドルーム確保 (Dynamic)
        // クランプ前に適用することで、>0dBFSの入力を救済する
        if (std::abs(gain - 1.0) > 1e-9)
            cblas_dscal(numSamples, gain, data, 1);

        // 2) デノーマル/NaN対策 + 範囲制限
        sanitizeAndLimit(data, numSamples);
    }

    inline void convertFloatToDoubleHighQuality(const float* src, double* dst, int numSamples, double gain = 1.0) noexcept
    {
        if (src == nullptr || dst == nullptr || numSamples <= 0)
            return;

        int i = 0;
        const int vEnd = (numSamples / 8) * 8;
        for (; i < vEnd; i += 8)
        {
            const __m256 vf = _mm256_loadu_ps(src + i);
            const __m128 lo = _mm256_castps256_ps128(vf);
            const __m128 hi = _mm256_extractf128_ps(vf, 1);
            _mm256_store_pd(dst + i,     _mm256_cvtps_pd(lo));
            _mm256_store_pd(dst + i + 4, _mm256_cvtps_pd(hi));
        }
        for (; i < numSamples; ++i)
            dst[i] = static_cast<double>(src[i]);

        applyHighQuality64BitTransform(dst, numSamples, gain);
    }

    inline void convertDoubleToDoubleHighQuality(const double* src, double* dst, int numSamples, double gain = 1.0) noexcept
    {
        if (src == nullptr || dst == nullptr || numSamples <= 0)
            return;

        if (src != dst)
            std::memcpy(dst, src, static_cast<size_t>(numSamples) * sizeof(double));

        applyHighQuality64BitTransform(dst, numSamples, gain);
    }
}
