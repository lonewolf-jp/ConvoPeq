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
    double m_state = 0.0;
    double m_alpha = 1.0e-6;

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
            m_alpha = 1.0e-6;
            reset();
            return;
        }

        const double omega = 2.0 * juce::MathConstants<double>::pi * cutoffHz / sampleRate;
        const double alpha = -std::expm1(-omega);
        if (!std::isfinite(alpha) || alpha <= 0.0 || alpha >= 1.0)
            m_alpha = 1.0e-6;
        else
            m_alpha = alpha;
        reset();
    }

    void reset() noexcept
    {
        m_state = 0.0;
    }

    void loadState() noexcept
    {
    }

    void saveState() noexcept
    {
    }

    inline void processSample(double& sample) noexcept
    {
        const double alpha = m_alpha;
        constexpr double kDenormalThreshold = convo::numeric_policy::kDenormThresholdAudioState;

        const double x = sample;
        m_state = m_state + alpha * (x - m_state);
        double y = x - m_state;

        if (!isFiniteAndAboveThresholdMask(y, kDenormalThreshold))
            y = 0.0;
        if (!isFiniteAndAboveThresholdMask(m_state, kDenormalThreshold))
            m_state = 0.0;

        sample = y;
    }

    void process(double* data, int numSamples) noexcept
    {
        if (data == nullptr || numSamples <= 0)
            return;

        double state = m_state;
        const double alpha = m_alpha;
        constexpr double kDenormalThreshold = convo::numeric_policy::kDenormThresholdAudioState;

        for (int i = 0; i < numSamples; ++i)
        {
            const double x = data[i];
            state = state + alpha * (x - state);
            double y = x - state;

            if (!isFiniteAndAboveThresholdMask(y, kDenormalThreshold))
                y = 0.0;
            if (!isFiniteAndAboveThresholdMask(state, kDenormalThreshold))
                state = 0.0;

            data[i] = y;
        }

        m_state = isFiniteAndBelowThresholdMask(state, 1.0e15) ? state : 0.0;
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
};
}
