#include <JuceHeader.h>
#include <immintrin.h>
#include "AudioEngine.h"
#include "EQProcessor.h"

namespace {
// EQ応答曲線計算用補助関数
void calcMagnitudesForBand(const EQCoeffsBiquad& c,
                           const std::complex<double>* zArr,
                           float* outMagSq,
                           int numPoints) noexcept
{
    const __m256d vB0 = _mm256_set1_pd(c.b0);
    const __m256d vB1 = _mm256_set1_pd(c.b1);
    const __m256d vB2 = _mm256_blend_pd(_mm256_set1_pd(c.b2), _mm256_setzero_pd(), 0b1010); // [b2, 0, b2, 0]
    const __m256d vA0 = _mm256_set1_pd(c.a0);
    const __m256d vA1 = _mm256_set1_pd(c.a1);
    const __m256d vA2 = _mm256_blend_pd(_mm256_set1_pd(c.a2), _mm256_setzero_pd(), 0b1010); // [a2, 0, a2, 0]
    const __m256d vDenEpsilon = _mm256_set1_pd(1.0e-36);

    int i = 0;
    for (; i <= numPoints - 2; i += 2)
    {
        // z = [re0, im0, re1, im1]
        __m256d z = _mm256_loadu_pd(reinterpret_cast<const double*>(zArr + i));

        // --- z^2 ---
        __m256d z_swapped = _mm256_permute_pd(z, 5); // [im0, re0, im1, re1]
        __m256d z_re_sq_parts = _mm256_mul_pd(z, z); // [re0^2, im0^2, re1^2, im1^2]
        __m256d z_im_part_parts = _mm256_mul_pd(z, z_swapped); // [re0*im0, im0*re0, re1*im1, im1*re1]

        __m256d z2_re_lanes = _mm256_hsub_pd(z_re_sq_parts, z_re_sq_parts); // [re0^2-im0^2, re0^2-im0^2, re1^2-im1^2, re1^2-im1^2]
        __m256d z2_im_lanes = _mm256_add_pd(z_im_part_parts, z_im_part_parts); // [2*re0*im0, 2*re0*im0, 2*re1*im1, 2*re1*im1]

        __m256d z2 = _mm256_unpacklo_pd(z2_re_lanes, z2_im_lanes); // [re(z0^2), im(z0^2), re(z1^2), im(z1^2)]

        // --- Polynomial Evaluation ---
        __m256d numV = _mm256_add_pd(_mm256_fmadd_pd(vB1, z, vB2), _mm256_mul_pd(vB0, z2));
        __m256d denV = _mm256_add_pd(_mm256_fmadd_pd(vA1, z, vA2), _mm256_mul_pd(vA0, z2));

        // --- Norm Squared: |v|^2 = re^2 + im^2 ---
        __m256d numSq = _mm256_mul_pd(numV, numV);
        __m256d denSq = _mm256_mul_pd(denV, denV);
        __m256d numNorm = _mm256_hadd_pd(numSq, numSq); // [|num0|^2, |num0|^2, |num1|^2, |num1|^2]
        __m256d denNorm = _mm256_hadd_pd(denSq, denSq);

        denNorm = _mm256_max_pd(denNorm, vDenEpsilon);
        __m256d magSqV = _mm256_div_pd(numNorm, denNorm);

        // --- Store ---
        __m128 magSqF = _mm256_cvtpd_ps(_mm256_permute4x64_pd(magSqV, 0xD8)); // Extract lanes 0 and 2 to [magSq0, magSq1, ?, ?]
        _mm_storel_pi(reinterpret_cast<__m64*>(outMagSq + i), magSqF); // Store 2 floats
    }

    for (; i < numPoints; ++i)
        outMagSq[i] = EQProcessor::getMagnitudeSquared(c, zArr[i]);
}
}

void AudioEngine::calcEQResponseCurve(float* outMagnitudesL,
                                     float* outMagnitudesR,
                                     const std::complex<double>* zArray,
                                     int numPoints,
                                     double sampleRate)
{
    constexpr float kEQGainEpsilon = 0.01f;
    constexpr float kEQUnityGainEpsilon = 1.0e-5f;

    const double sr = sampleRate;
    if (sr <= 0.0)
    {
        for (int i = 0; i < numPoints; ++i)
        {
            if (outMagnitudesL) outMagnitudesL[i] = 1.0f;
            if (outMagnitudesR) outMagnitudesR[i] = 1.0f;
        }
        return;
    }

    // ── 最適化: 有効なバンドの係数をスタック上で事前に計算 ──
    // UIスレッドでの計算負荷を下げるため、無効なバンドやゲイン0のバンドは除外する
    struct ActiveBand {
        EQCoeffsBiquad coeffs;
        EQChannelMode mode;
    };
    ActiveBand activeBands[EQProcessor::NUM_BANDS];
    int numActiveBands = 0;

    // 状態スナップショットを取得して一貫性を確保
    auto eqState = uiEqEditor.getEQState();

    if (eqState == nullptr)
    {
        if (outMagnitudesL) std::fill_n(outMagnitudesL, numPoints, 1.0f);
        if (outMagnitudesR) std::fill_n(outMagnitudesR, numPoints, 1.0f);
        return;
    }

    for (int band = 0; band < EQProcessor::NUM_BANDS; ++band)
    {
        const auto& params = eqState->bands[band];
        if (!params.enabled) continue;

        EQBandType type = eqState->bandTypes[band];

        // LowPass/HighPass以外でゲインがほぼ0の場合はスキップ
        if (type != EQBandType::LowPass && type != EQBandType::HighPass &&
            std::abs(params.gain) < kEQGainEpsilon)
            continue;

        // 【Fix Bug #EQ-Display】 Audio EQ Cookbook (calcBiquadCoeffs) の代わりに
        // calcSVFCoeffs → svfToDisplayBiquad を使用する。
        //
        // 背景:
        //   実際の音声処理は TPT SVF (calcSVFCoeffs) を使用しており、
        //   Peaking の場合 k = 1/(Q·A)、LowShelf は g = tan()/√A 等、
        //   Cookbook とは異なるパラメータ化を採用している。
        //   その結果、以前の calcBiquadCoeffs (RBJ Cookbook: alpha = sin(w0)/(2Q))
        //   は実際の SVF フィルタとバンド幅が微妙に異なり、
        //   総合応答曲線が個別バンド曲線の積と一致しないという表示上の誤りが
        //   生じていた。
        //
        // 修正:
        //   calcSVFCoeffs → svfToDisplayBiquad のパスは SVF の z 域伝達関数を
        //   厳密に等価 biquad へ変換する（updateEQData の個別バンド曲線と同一）。
        //   これにより「総合曲線 = 個別バンド曲線の積 = 実際の DSP 処理」が
        //   三者完全一致する。
        activeBands[numActiveBands++] = {
            EQProcessor::svfToDisplayBiquad(
                EQProcessor::calcSVFCoeffs(type, params.frequency, params.gain, params.q, sr)),
            eqState->bandChannelModes[band]
        };
    }

    float totalGainLinear = 1.0f;
    if (!uiEqEditor.getAGCEnabled())
    {
        totalGainLinear = juce::Decibels::decibelsToGain(eqState->totalGainDb);
    }

    // ── 最適化: 有効なバンドがない、かつトータルゲインが0dBの場合は計算をスキップ ──
    if (numActiveBands == 0 && std::abs(totalGainLinear - 1.0f) < kEQUnityGainEpsilon)
    {
        if (outMagnitudesL) std::fill_n(outMagnitudesL, numPoints, 1.0f);
        if (outMagnitudesR) std::fill_n(outMagnitudesR, numPoints, 1.0f);
        return;
    }

    const float totalGainSq = totalGainLinear * totalGainLinear;

    float* totalMagSqL = eqTotalMagSqLBuffer.data();
    float* totalMagSqR = eqTotalMagSqRBuffer.data();
    float* bandMagSq = eqBandMagSqBuffer.data();

    const __m256 vTotalGainSq = _mm256_set1_ps(totalGainSq);
    int i = 0;
    const int vEnd = numPoints / 8 * 8;
    for (; i < vEnd; i += 8)
    {
        _mm256_storeu_ps(totalMagSqL + i, vTotalGainSq);
        _mm256_storeu_ps(totalMagSqR + i, vTotalGainSq);
    }
    for (; i < numPoints; ++i)
    {
        totalMagSqL[i] = totalGainSq;
        totalMagSqR[i] = totalGainSq;
    }

    for (int b = 0; b < numActiveBands; ++b)
    {
        const auto& band = activeBands[b];
        calcMagnitudesForBand(band.coeffs, zArray, bandMagSq, numPoints);

        i = 0;
        if (band.mode == EQChannelMode::Stereo)
        {
            for (; i < vEnd; i += 8)
            {
                __m256 vBand = _mm256_loadu_ps(bandMagSq + i);
                __m256 vL = _mm256_loadu_ps(totalMagSqL + i);
                __m256 vR = _mm256_loadu_ps(totalMagSqR + i);
                _mm256_storeu_ps(totalMagSqL + i, _mm256_mul_ps(vL, vBand));
                _mm256_storeu_ps(totalMagSqR + i, _mm256_mul_ps(vR, vBand));
            }
        }
        else if (band.mode == EQChannelMode::Left)
        {
            for (; i < vEnd; i += 8)
            {
                __m256 vBand = _mm256_loadu_ps(bandMagSq + i);
                __m256 vL = _mm256_loadu_ps(totalMagSqL + i);
                _mm256_storeu_ps(totalMagSqL + i, _mm256_mul_ps(vL, vBand));
            }
        }
        else // Right
        {
            for (; i < vEnd; i += 8)
            {
                __m256 vBand = _mm256_loadu_ps(bandMagSq + i);
                __m256 vR = _mm256_loadu_ps(totalMagSqR + i);
                _mm256_storeu_ps(totalMagSqR + i, _mm256_mul_ps(vR, vBand));
            }
        }

        for (; i < numPoints; ++i)
        {
            float magSq = bandMagSq[i];
            if (!std::isfinite(magSq)) magSq = 1.0f;
            if (band.mode == EQChannelMode::Stereo || band.mode == EQChannelMode::Left)
                totalMagSqL[i] *= magSq;
            if (band.mode == EQChannelMode::Stereo || band.mode == EQChannelMode::Right)
                totalMagSqR[i] *= magSq;
        }
    }

    if (outMagnitudesL)
    {
        int j = 0;
        const int vEndSqrt = numPoints / 8 * 8;
        const __m256 vZero = _mm256_setzero_ps();
        for (; j < vEndSqrt; j += 8)
        {
            __m256 v = _mm256_loadu_ps(totalMagSqL + j);
            v = _mm256_max_ps(v, vZero);
            _mm256_storeu_ps(outMagnitudesL + j, _mm256_sqrt_ps(v));
        }
        for (; j < numPoints; ++j)
            outMagnitudesL[j] = std::sqrt(std::max(0.0f, totalMagSqL[j]));
        for (int k = 0; k < numPoints; ++k)
            if (!std::isfinite(outMagnitudesL[k])) outMagnitudesL[k] = 1.0f;
    }

    if (outMagnitudesR)
    {
        int j = 0;
        const int vEndSqrt = numPoints / 8 * 8;
        const __m256 vZero = _mm256_setzero_ps();
        for (; j < vEndSqrt; j += 8)
        {
            __m256 v = _mm256_loadu_ps(totalMagSqR + j);
            v = _mm256_max_ps(v, vZero);
            _mm256_storeu_ps(outMagnitudesR + j, _mm256_sqrt_ps(v));
        }
        for (; j < numPoints; ++j)
            outMagnitudesR[j] = std::sqrt(std::max(0.0f, totalMagSqR[j]));
        for (int k = 0; k < numPoints; ++k)
            if (!std::isfinite(outMagnitudesR[k])) outMagnitudesR[k] = 1.0f;
    }
}
