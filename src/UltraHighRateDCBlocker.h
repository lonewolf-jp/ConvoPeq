//============================================================================
#pragma once
// UltraHighRateDCBlocker.h
// 1次IIR DC Blocker（超高サンプリングレート対応）
//============================================================================

#include <cmath>
#include <immintrin.h>
#include <juce_core/juce_core.h>

#include "DspNumericPolicy.h"

namespace convo
{
class UltraHighRateDCBlocker
{
private:
    double m_prev_x = 0.0;
    double m_prev_y = 0.0;
    double m_R = 0.999999;

public:
    // 利用ルール:
    // - init() は std::exp() を使用するため Audio Thread から呼ばないこと。
    //   prepareToPlay() / リビルド処理などの非Audio Threadで初期化する。
    // - process() は Audio Thread で呼び出し可能（ブロッキング処理・動的確保なし）。
    // - init() 呼び出し後は内部状態が reset() されるため、再初期化時は先行状態を引き継がない。
    void init(double sampleRate, double cutoffHz) noexcept
    {
        if (!std::isfinite(sampleRate) || sampleRate <= 0.0 ||
            !std::isfinite(cutoffHz) || cutoffHz <= 0.0)
        {
            m_R = 0.999999;
            reset();
            return;
        }

        m_R = std::exp(-2.0 * juce::MathConstants<double>::pi * cutoffHz / sampleRate);
        if (! std::isfinite(m_R) || m_R <= 0.0 || m_R >= 1.0)
            m_R = 0.999999;
        reset();
    }

    void reset() noexcept
    {
        m_prev_x = 0.0;
        m_prev_y = 0.0;
    }

    void loadState() noexcept
    {
        px_local = m_prev_x;
        py_local = m_prev_y;
    }

    void saveState() noexcept
    {
        m_prev_x = px_local;
        m_prev_y = py_local;
    }

    inline void processSample(double& sample) noexcept
    {
        const double r = m_R;
        constexpr double kDenormalThreshold = convo::numeric_policy::kDenormThresholdAudioState;

        const double curr_x = sample;
        double curr_y = curr_x - px_local + r * py_local;

        if (!isFiniteAndAboveThresholdMask(curr_y, kDenormalThreshold))
            curr_y = 0.0;

        px_local = curr_x;
        py_local = curr_y;
        sample = curr_y;
    }

    void process(double* data, int numSamples) noexcept
    {
        double px = m_prev_x;
        double py = m_prev_y;
        const double r = m_R;
        constexpr double kDenormalThreshold = convo::numeric_policy::kDenormThresholdAudioState;

        int i = 0;
        const int vEnd = numSamples / 4 * 4;

        if (vEnd > 0)
        {
            const double r2 = r * r;
            const double r3 = r2 * r;
            const double r4 = r3 * r;
            const __m256d vR = _mm256_set1_pd(r);
            const __m256d vR2 = _mm256_set1_pd(r2);
            const __m256d vPrevYFactors = _mm256_set_pd(r4, r3, r2, r);
            const __m256d vThresh = _mm256_set1_pd(kDenormalThreshold);
            const __m256d vInfThresh = _mm256_set1_pd(1.0e100);
            const __m256d vSignMask = _mm256_set1_pd(-0.0);

            for (; i < vEnd; i += 4)
            {
                __m256d vx = _mm256_load_pd(data + i);

                __m256d t = _mm256_permute4x64_pd(vx, _MM_SHUFFLE(2, 1, 0, 0));
                __m256d vpx = _mm256_set1_pd(px);
                __m256d v_prev_x = _mm256_blend_pd(t, vpx, 1);

                __m256d vu = _mm256_sub_pd(vx, v_prev_x);

                __m256d v_shift1 = _mm256_permute4x64_pd(vu, _MM_SHUFFLE(2, 1, 0, 0));
                v_shift1 = _mm256_blend_pd(v_shift1, _mm256_setzero_pd(), 1);
                __m256d vs1 = _mm256_fmadd_pd(vR, v_shift1, vu);

                __m256d v_shift2 = _mm256_permute4x64_pd(vs1, _MM_SHUFFLE(1, 0, 0, 0));
                v_shift2 = _mm256_blend_pd(v_shift2, _mm256_setzero_pd(), 3);
                __m256d vy = _mm256_fmadd_pd(vR2, v_shift2, vs1);

                __m256d vpy = _mm256_set1_pd(py);
                vy = _mm256_fmadd_pd(vPrevYFactors, vpy, vy);

                __m256d abs_y = _mm256_andnot_pd(vSignMask, vy);
                __m256d mask = _mm256_cmp_pd(abs_y, vThresh, _CMP_GE_OQ);
                mask = _mm256_and_pd(mask, _mm256_cmp_pd(abs_y, vInfThresh, _CMP_LT_OQ));
                vy = _mm256_and_pd(vy, mask);

                _mm256_store_pd(data + i, vy);

                _mm_storeh_pd(&px, _mm256_extractf128_pd(vx, 1));
                _mm_storeh_pd(&py, _mm256_extractf128_pd(vy, 1));
            }
        }

        for (; i < numSamples; ++i)
        {
            const double curr_x = data[i];
            double curr_y = curr_x - px + r * py;

            if (!isFiniteAndAboveThresholdMask(curr_y, kDenormalThreshold))
                curr_y = 0.0;

            px = curr_x;
            py = curr_y;
            data[i] = curr_y;
        }

        m_prev_x = isFiniteAndBelowThresholdMask(px, 1.0e15) ? px : 0.0;
        m_prev_y = isFiniteAndBelowThresholdMask(py, 1.0e15) ? py : 0.0;
    }

private:
    static inline bool isFiniteAndAboveThresholdMask(double value, double threshold) noexcept
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

    static inline bool isFiniteAndBelowThresholdMask(double value, double threshold) noexcept
    {
        const __m128d v = _mm_set1_pd(value);
        const __m128d diff = _mm_sub_pd(v, v);
        const __m128d finiteMask = _mm_cmpeq_pd(diff, _mm_setzero_pd());

        const __m128d signMask = _mm_set1_pd(-0.0);
        const __m128d absV = _mm_andnot_pd(signMask, v);
        const __m128d thresholdV = _mm_set1_pd(threshold);
        const __m128d belowMask = _mm_cmplt_pd(absV, thresholdV);

        const __m128d validMask = _mm_and_pd(finiteMask, belowMask);
        return _mm_movemask_pd(validMask) == 0x3;
    }

    double px_local = 0.0;
    double py_local = 0.0;
};
}
