//============================================================================
#define _USE_MATH_DEFINES
#include <JuceHeader.h>
#include <immintrin.h>
#include <cmath>

#include "LoudnessMeter.h"

//============================================================================
void LoudnessMeter::prepare(double sr, int maxBlockSize)
{
    sampleRate = sr;
    blockCounter = 0;
    preparedBlockSize = maxBlockSize;

    // ★ [work74 FIX-02] サンプルレートに応じてK-weighting係数を再計算
    updateCoefficients(sr);

    const int required = maxBlockSize * 2;
    if (required > filterWorkCapacity || !filterWorkBuffer)
    {
        filterWorkBuffer = convo::makeAlignedArray<double>(static_cast<size_t>(required));
        filterWorkCapacity = required;
    }
    // RingBufferStorageを動的確保（これによりDSPCoreのサイズが98KB肥大化するのを防ぐ）
    if (!ringBufferStorage)
    {
        auto* mem = convo::aligned_malloc(sizeof(RingBufferStorage), 64);
        if (mem)
            ringBufferStorage = convo::ScopedAlignedPtr<RingBufferStorage>(
                new (mem) RingBufferStorage());
    }
    reset();
}

void LoudnessMeter::reset() noexcept
{
    for (int ch = 0; ch < kMaxChannels; ++ch)
    {
        std::memset(&preFilterState[ch], 0, sizeof(KWeightingState));
        std::memset(&rlbFilterState[ch], 0, sizeof(KWeightingState));
    }
    blockCounter = 0;
    if (ringBufferStorage) ringBufferStorage->ringBuffer.clear();
}

void LoudnessMeter::processBlock(const double* dataL, const double* dataR, int numSamples) noexcept
{
    if (numSamples <= 0 || !filterWorkBuffer) return;

    double* fl = filterWorkBuffer.get();
    double* fr = fl + numSamples;
    if (!fl) return;

    // Stage 1: Pre-filter per channel
    for (int n = 0; n < numSamples; ++n)
    {
        fl[n] = processKWeightingStage(preFilterCoeffs, preFilterState[0], dataL[n]);
        fr[n] = (dataR != nullptr)
            ? processKWeightingStage(preFilterCoeffs, preFilterState[1], dataR[n])
            : processKWeightingStage(preFilterCoeffs, preFilterState[1], dataL[n]);
    }

    // Stage 2: RLB filter per channel
    for (int n = 0; n < numSamples; ++n)
    {
        fl[n] = processKWeightingStage(rlbFilterCoeffs, rlbFilterState[0], fl[n]);
        fr[n] = processKWeightingStage(rlbFilterCoeffs, rlbFilterState[1], fr[n]);
    }

    // Compute mean square + peak (processed signal)
    double sumSqL = 0.0, sumSqR = 0.0;
    double peakL = 0.0, peakR = 0.0;
    int i = 0;
#if defined(__AVX2__)
    const int vEnd = numSamples / 4 * 4;
    __m256d vSumL = _mm256_setzero_pd();
    __m256d vSumR = _mm256_setzero_pd();
    __m256d vPeakL = _mm256_setzero_pd();
    __m256d vPeakR = _mm256_setzero_pd();
    const __m256d vSignMask = _mm256_set1_pd(-0.0);
    for (; i < vEnd; i += 4)
    {
        __m256d vL = _mm256_loadu_pd(fl + i);
        __m256d vR = _mm256_loadu_pd(fr + i);
        vSumL = _mm256_fmadd_pd(vL, vL, vSumL);
        vSumR = _mm256_fmadd_pd(vR, vR, vSumR);
        vPeakL = _mm256_max_pd(vPeakL, _mm256_andnot_pd(vSignMask, vL));
        vPeakR = _mm256_max_pd(vPeakR, _mm256_andnot_pd(vSignMask, vR));
    }
    // Reduce
    __m128d loL = _mm256_castpd256_pd128(vSumL);
    __m128d hiL = _mm256_extractf128_pd(vSumL, 1);
    __m128d sumL128 = _mm_add_pd(loL, hiL);
    sumL128 = _mm_hadd_pd(sumL128, sumL128);
    _mm_store_sd(&sumSqL, sumL128);

    __m128d loR = _mm256_castpd256_pd128(vSumR);
    __m128d hiR = _mm256_extractf128_pd(vSumR, 1);
    __m128d sumR128 = _mm_add_pd(loR, hiR);
    sumR128 = _mm_hadd_pd(sumR128, sumR128);
    _mm_store_sd(&sumSqR, sumR128);

    loL = _mm256_castpd256_pd128(vPeakL);
    hiL = _mm256_extractf128_pd(vPeakL, 1);
    __m128d pL128 = _mm_max_pd(loL, hiL);
    pL128 = _mm_max_sd(pL128, _mm_unpackhi_pd(pL128, pL128));
    _mm_store_sd(&peakL, pL128);

    loR = _mm256_castpd256_pd128(vPeakR);
    hiR = _mm256_extractf128_pd(vPeakR, 1);
    __m128d pR128 = _mm_max_pd(loR, hiR);
    pR128 = _mm_max_sd(pR128, _mm_unpackhi_pd(pR128, pR128));
    _mm_store_sd(&peakR, pR128);
#endif
    for (; i < numSamples; ++i)
    {
        const double l = fl[i], r = fr[i];
        sumSqL += l * l;
        sumSqR += r * r;
        const double al = std::abs(l), ar = std::abs(r);
        if (al > peakL) peakL = al;
        if (ar > peakR) peakR = ar;
    }

    // チャンネル重み (L=1.0, R=1.0) 適用
    const double meanSquare = (sumSqL * kChannelWeightStereo[0] + sumSqR * kChannelWeightStereo[1])
                            / static_cast<double>(numSamples);
    const double peakLinear = std::max(peakL, peakR);

    // RingBuffer publish
    BlockPower bp;
    bp.meanSquare = meanSquare;
    bp.peakLinear = peakLinear;
    bp.blockIndex = blockCounter++;
    if (ringBufferStorage) ringBufferStorage->ringBuffer.push(bp);
}

//============================================================================
// ★ [work74 FIX-02] K-weighting フィルタ係数のサンプルレート依存計算
//
// ITU-R BS.1770-4 Annex B のアナログ伝達関数に基づき、
// pre-warped bilinear transform でIIR係数をサンプルレート毎に計算する。
//
// 検証結果:
//   Stage 1 (Pre-filter) は中心周波数 f₀=1500Hz, Q=1/√2, G=+4dB の
//   標準ハイシェルフフィルタ (Audio EQ Cookbook) と判明。
//   Stage 2 (RLB) は fc=38Hz, Q=0.50 の標準ハイパスフィルタ。
//
// 参考実装: libebur128 (MIT license)
//   一次仕様: ITU-R BS.1770-4 Annex B
//============================================================================
void LoudnessMeter::updateCoefficients(double fs)
{
    if (fs <= 0.0)
        return;

    // ── Stage 2: RLB filter (High-pass, fc=38Hz, Q=0.50) ──
    //   H(s) = s² / (s² + E·s + 1),  E = 1.99004745483398
    //   → Audio EQ Cookbook 高域通過 (High-pass) 公式
    {
        const double w0 = 2.0 * M_PI * 38.0 / fs;
        const double cosW0 = std::cos(w0);
        const double sinW0 = std::sin(w0);
        const double alpha = sinW0 / (2.0 * 0.50); // Q=0.50

        // ★ 注意: High-pass は (1+cosΩ) を使用。Low-pass の (1-cosΩ) と混同しないこと。
        const double b0 = (1.0 + cosW0) / 2.0;
        const double b1 = -(1.0 + cosW0);
        const double b2 = (1.0 + cosW0) / 2.0;
        const double a0 = 1.0 + alpha;
        const double a1 = -2.0 * cosW0;
        const double a2 = 1.0 - alpha;

        // a0=1 に正規化
        // processKWeightingStage の DF1 形式: y = b'*x + ... - A1*y1 - A2*y2
        // A1 = a1/a0, A2 = a2/a0 (a1,a2 は Cookbook 形式)
        const double invA0 = 1.0 / a0;
        rlbFilterCoeffs[0] = b0 * invA0; // b0'
        rlbFilterCoeffs[1] = b1 * invA0; // b1'
        rlbFilterCoeffs[2] = b2 * invA0; // b2'
        rlbFilterCoeffs[3] = a1 * invA0; // A1 = a1/a0 (DF1 で -coeffs[3]*y1)
        rlbFilterCoeffs[4] = a2 * invA0; // A2 = a2/a0 (DF1 で -coeffs[4]*y2)
    }

    // ── Stage 1: Pre-filter (High-shelf, f₀=1500Hz, Q=1/√2, G=+4dB) ──
    //   BS.1770-4 Table 1 の48kHz係数を標準ハイシェルフbiquad公式で再現確認済み:
    //     b0=1.535, b1=-2.692, b2=1.198, a1=-1.691, a2=0.732
    //   → 任意 fs で Audio EQ Cookbook 高域シェルフ公式を使用
    {
        constexpr double kPreFreq = 1500.0;    // 中心周波数 [Hz] (BS.1770準拠)
        constexpr double kPreGainDb = 4.0;       // 高域ゲイン [+dB]
        constexpr double kPreQ = 0.7071067811865476; // 1/√2

        const double w0 = 2.0 * M_PI * kPreFreq / fs;
        const double cosW0 = std::cos(w0);
        const double sinW0 = std::sin(w0);
        const double A = std::pow(10.0, kPreGainDb / 40.0);
        const double alpha = sinW0 / (2.0 * kPreQ);
        const double sqrtA = std::sqrt(A);

        // High-shelf 公式 (Audio EQ Cookbook)
        const double b0 = A * ((A + 1.0) + (A - 1.0) * cosW0 + 2.0 * sqrtA * alpha);
        const double b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cosW0);
        const double b2 = A * ((A + 1.0) + (A - 1.0) * cosW0 - 2.0 * sqrtA * alpha);
        const double a0 = (A + 1.0) - (A - 1.0) * cosW0 + 2.0 * sqrtA * alpha;
        const double a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cosW0);
        const double a2 = (A + 1.0) - (A - 1.0) * cosW0 - 2.0 * sqrtA * alpha;

        const double invA0 = 1.0 / a0;
        preFilterCoeffs[0] = b0 * invA0;
        preFilterCoeffs[1] = b1 * invA0;
        preFilterCoeffs[2] = b2 * invA0;
        preFilterCoeffs[3] = a1 * invA0; // A1 = a1/a0 (DF1 で -coeffs[3]*y1)
        preFilterCoeffs[4] = a2 * invA0; // A2 = a2/a0 (DF1 で -coeffs[4]*y2)
    }
}
