//============================================================================
#include <JuceHeader.h>
#include <immintrin.h>

#include "LoudnessMeter.h"

//============================================================================
void LoudnessMeter::prepare(double sr, int maxBlockSize)
{
    sampleRate = sr;
    blockCounter = 0;
    preparedBlockSize = maxBlockSize;
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
        fl[n] = processKWeightingStage(kPreBiquad, preFilterState[0], dataL[n]);
        fr[n] = (dataR != nullptr)
            ? processKWeightingStage(kPreBiquad, preFilterState[1], dataR[n])
            : processKWeightingStage(kPreBiquad, preFilterState[1], dataL[n]);
    }

    // Stage 2: RLB filter per channel
    for (int n = 0; n < numSamples; ++n)
    {
        fl[n] = processKWeightingStage(kRlbBiquad, rlbFilterState[0], fl[n]);
        fr[n] = processKWeightingStage(kRlbBiquad, rlbFilterState[1], fr[n]);
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
