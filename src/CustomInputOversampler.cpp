//============================================================================
#include <JuceHeader.h>
#include "CustomInputOversampler.h"
#include "DspNumericPolicy.h"

#include <bit>
#include <cmath>
#include <cstring>
#include <immintrin.h>

#include "audioengine/AtomicAccess.h"

namespace
{
    [[maybe_unused]] static constexpr int kMaxChannels = 2;

    inline double fastAbs(double x) noexcept
    {
        uint64_t bits = std::bit_cast<uint64_t>(x);
        bits &= 0x7FFFFFFFFFFFFFFFULL;
        return std::bit_cast<double>(bits);
    }

    inline bool isBadSample(double x) noexcept
    {
        const uint64_t bits = std::bit_cast<uint64_t>(x);
        const uint64_t exp = bits & 0x7FF0000000000000ULL;
        if (exp == 0x7FF0000000000000ULL)
            return true;

        constexpr uint64_t limit = 0x4340000000000000ULL;
        return (bits & 0x7FFFFFFFFFFFFFFFULL) > limit;
    }

    constexpr double kDenormThreshold = convo::numeric_policy::kDenormThresholdAudioState;

#if defined(__AVX2__)
    /// AVX2 版バッチ isBadSample: 4要素を1SIMD命令でチェック
    /// halfband 非連続インデックスでも set_pd 後に一括チェック可能
    [[maybe_unused]] inline bool isBadSampleV(__m256d v) noexcept
    {
        // NaN 検出: _CMP_UNORD_Q — v のいずれかが NaN で true
        const __m256d vNanMask = _mm256_cmp_pd(v, v, _CMP_UNORD_Q);
        // Inf/絶対値 > limit 検出
        const __m256d vAbs = _mm256_andnot_pd(_mm256_set1_pd(-0.0), v);
        const __m256d vInfMask = _mm256_cmp_pd(vAbs, _mm256_set1_pd(1e20), _CMP_GT_OQ);
        return _mm256_movemask_pd(_mm256_or_pd(vNanMask, vInfMask)) != 0;
    }

    /// @brief Stride-2 で4要素を半帯域履歴バッファからロード
    /// @param ptr  history のベースアドレス（呼出側: history + (base - convParity)）
    /// @return     { ptr[0], ptr[-2], ptr[-4], ptr[-6] } の順
    /// @note       係数 coeffs[r+0..r+3] との FMA で正しい畳み込み順序になる
    inline __m256d loadStride2(const double* ptr) noexcept
    {
        __m128d v0 = _mm_loadu_pd(ptr - 6);    // { ptr[-6], ptr[-5] }
        __m128d v1 = _mm_loadu_pd(ptr - 4);    // { ptr[-4], ptr[-3] }
        __m128d v2 = _mm_loadu_pd(ptr - 2);    // { ptr[-2], ptr[-1] }
        __m128d v3 = _mm_loadu_pd(ptr);        // { ptr[0],  ptr[1]  }
        // low:  ptr[0], ptr[-2]  ← v3 と v2 の low element
        __m128d vLow  = _mm_unpacklo_pd(v3, v2);
        // high: ptr[-4], ptr[-6] ← v1 と v0 の low element
        __m128d vHigh = _mm_unpacklo_pd(v1, v0);
        return _mm256_insertf128_pd(
            _mm256_castpd128_pd256(vLow), vHigh, 1);
        // = { ptr[0], ptr[-2], ptr[-4], ptr[-6] }
    }
#endif
}

CustomInputOversampler::~CustomInputOversampler()
{
    release();
}

int CustomInputOversampler::sanitizeRatio(int ratio) noexcept
{
    if (ratio >= 8) return 8;
    if (ratio >= 4) return 4;
    if (ratio >= 2) return 2;
    return 1;
}

int CustomInputOversampler::tapsForStage(int stageIndex, Preset preset) noexcept
{
    if (preset == Preset::LinearPhase)
    {
        static constexpr int taps[3] = { 1023, 255, 63 };
        return taps[juce::jlimit(0, 2, stageIndex)];
    }

    static constexpr int taps[3] = { 511, 127, 31 };
    return taps[juce::jlimit(0, 2, stageIndex)];
}

double CustomInputOversampler::attenuationForStage(int stageIndex, Preset preset) noexcept
{
    if (preset == Preset::LinearPhase)
    {
        static constexpr double attenuation[3] = { 160.0, 140.0, 120.0 };
        return attenuation[juce::jlimit(0, 2, stageIndex)];
    }

    static constexpr double attenuation[3] = { 140.0, 110.0, 90.0 };
    return attenuation[juce::jlimit(0, 2, stageIndex)];
}

void CustomInputOversampler::clearStage(Stage& stage) noexcept
{
    stage.convCoeffs.reset();
    stage.convCoeffsReversed.reset();
    for (int ch = 0; ch < kMaxChannels; ++ch)
    {
        stage.upHistory[ch].reset();
        stage.downHistory[ch].reset();
    }
    stage.upHistorySize = 0;
    stage.downHistorySize = 0;
}

void CustomInputOversampler::release() noexcept
{
    for (auto& stage : stages)
        clearStage(stage);

    for (int ch = 0; ch < kMaxChannels; ++ch)
    {
        workA[ch].reset();
        workB[ch].reset();
        blockChannels[ch] = nullptr;
    }

    workCapacity = 0;
    upsampleRatio = 1;
    activePreset = Preset::IIRLike;
    numStages = 0;
    maxInputBlockSize = 0;
    maxUpsampledBlockSize = 0;
    convo::publishAtomic(corruptionDetected, false, std::memory_order_release);
    convo::publishAtomic(consecutiveCorruptionAutoClearCount, static_cast<std::uint32_t>(0), std::memory_order_release);
    convo::publishAtomic(hardFallbackActive, false, std::memory_order_release);
}

double CustomInputOversampler::besselI0(double x) noexcept
{
    double sum = 1.0;
    double term = 1.0;
    const double xx = x * x;
    for (int n = 1; n < 100; ++n)
    {
        term *= xx / (4.0 * static_cast<double>(n) * static_cast<double>(n));
        sum += term;
        if (term < sum * 1.0e-18)
            break;
    }
    return sum;
}

double CustomInputOversampler::dotProductAvx2(const double* __restrict x,
                                              const double* __restrict coeffs,
                                              int n) noexcept
{
    // x: upHistory buffer with offset, not guaranteed to be aligned. Use loadu.
    // coeffs: convCoeffsReversed buffer, guaranteed to be 64-byte aligned. Use load.
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    __m256d acc2 = _mm256_setzero_pd();
    __m256d acc3 = _mm256_setzero_pd();

    int i = 0;
    // Unroll 4x to hide FMA latency (16 elements per main loop)
    for (; i <= n - 16; i += 16)
    {
        // Guard: prefetch only when within buffer bounds (safety fix, not performance optimization)
        if (i + 64 < n) {
            _mm_prefetch(reinterpret_cast<const char*>(x + i + 64), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(coeffs + i + 64), _MM_HINT_T0);
        }

        acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i),      _mm256_load_pd(coeffs + i),      acc0);
        acc1 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i + 4),  _mm256_load_pd(coeffs + i + 4),  acc1);
        acc2 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i + 8),  _mm256_load_pd(coeffs + i + 8),  acc2);
        acc3 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i + 12), _mm256_load_pd(coeffs + i + 12), acc3);
    }
    // Handle remaining blocks of 4
    for (; i <= n - 4; i += 4)
        acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i), _mm256_load_pd(coeffs + i), acc0);

    // Reduction of the four accumulators
    acc0 = _mm256_add_pd(acc0, acc1);
    acc2 = _mm256_add_pd(acc2, acc3);
    acc0 = _mm256_add_pd(acc0, acc2);

    // SIMD horizontal reduction: vextractf128 + hadd (avoid store-to-stack + scalar adds)
    __m128d vLo = _mm256_castpd256_pd128(acc0);
    __m128d vHi = _mm256_extractf128_pd(acc0, 1);
    __m128d vSum = _mm_add_pd(vLo, vHi);
    vSum = _mm_hadd_pd(vSum, vSum);
    double sum = _mm_cvtsd_f64(vSum);

    // Scalar remainder
    for (; i < n; ++i)
        sum += x[i] * coeffs[i];

    if (isBadSample(sum))
        sum = 0.0;
    else if (fastAbs(sum) < kDenormThreshold)
        sum = 0.0;

    return sum;
}

#if defined(__AVX2__) && defined(__FMA__)
double CustomInputOversampler::dotProductDecimateAvx2(
    const double* __restrict history,
    const double* __restrict coeffs,
    int convCount) noexcept
{
    // 8-way unroll (32 elements/iteration) for Skylake+ FMA pipeline saturation
    // FMA latency=4, throughput=0.5 → need 4/0.5=8 independent accumulators
    __assume(convCount >= 8);
    __assume(convCount >= 0);

    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    __m256d acc2 = _mm256_setzero_pd();
    __m256d acc3 = _mm256_setzero_pd();
    __m256d acc4 = _mm256_setzero_pd();
    __m256d acc5 = _mm256_setzero_pd();
    __m256d acc6 = _mm256_setzero_pd();
    __m256d acc7 = _mm256_setzero_pd();

    int r = 0;
    const int unrollEnd = (convCount / 32) * 32;
    for (; r < unrollEnd; r += 32)
    {
        acc0 = _mm256_fmadd_pd(loadStride2(history - (r << 1)),       _mm256_loadu_pd(coeffs + r),       acc0);
        acc1 = _mm256_fmadd_pd(loadStride2(history - ((r +  4) << 1)), _mm256_loadu_pd(coeffs + r +  4), acc1);
        acc2 = _mm256_fmadd_pd(loadStride2(history - ((r +  8) << 1)), _mm256_loadu_pd(coeffs + r +  8), acc2);
        acc3 = _mm256_fmadd_pd(loadStride2(history - ((r + 12) << 1)), _mm256_loadu_pd(coeffs + r + 12), acc3);
        acc4 = _mm256_fmadd_pd(loadStride2(history - ((r + 16) << 1)), _mm256_loadu_pd(coeffs + r + 16), acc4);
        acc5 = _mm256_fmadd_pd(loadStride2(history - ((r + 20) << 1)), _mm256_loadu_pd(coeffs + r + 20), acc5);
        acc6 = _mm256_fmadd_pd(loadStride2(history - ((r + 24) << 1)), _mm256_loadu_pd(coeffs + r + 24), acc6);
        acc7 = _mm256_fmadd_pd(loadStride2(history - ((r + 28) << 1)), _mm256_loadu_pd(coeffs + r + 28), acc7);
    }

    // 4-element blocks remainder (16 elements)
    const int simdEnd = (convCount / 4) * 4;
    for (; r < simdEnd; r += 4)
    {
        __m256d vS = loadStride2(history - (r << 1));
        __m256d vC = _mm256_loadu_pd(coeffs + r);
        acc0 = _mm256_fmadd_pd(vS, vC, acc0);
    }

    // Tree reduction: acc0+acc1, acc2+acc3, acc4+acc5, acc6+acc7
    acc0 = _mm256_add_pd(acc0, acc1);
    acc2 = _mm256_add_pd(acc2, acc3);
    acc4 = _mm256_add_pd(acc4, acc5);
    acc6 = _mm256_add_pd(acc6, acc7);
    // acc0+acc2, acc4+acc6
    acc0 = _mm256_add_pd(acc0, acc2);
    acc4 = _mm256_add_pd(acc4, acc6);
    // acc0+acc4
    acc0 = _mm256_add_pd(acc0, acc4);

    // SIMD horizontal reduction: vextractf128 + hadd
    __m128d vLo = _mm256_castpd256_pd128(acc0);
    __m128d vHi = _mm256_extractf128_pd(acc0, 1);
    __m128d vSum = _mm_add_pd(vLo, vHi);
    vSum = _mm_hadd_pd(vSum, vSum);
    double result = _mm_cvtsd_f64(vSum);

    // Scalar remainder (handles non-4-multiple convCount safely)
    for (; r < convCount; ++r)
        result += coeffs[r] * history[-(r << 1)];

    return result;
}
#endif

bool CustomInputOversampler::prepareStage(Stage& stage, int taps, double attenuationDb, int stageInputMax)
{
    clearStage(stage);

    stage.taps = juce::jmax(3, taps | 1);
    stage.centerTap = (stage.taps - 1) / 2;
    stage.centerParity = stage.centerTap & 1;
    stage.convParity = 1 - stage.centerParity;
    stage.maxInputSamples = stageInputMax;
    stage.maxOutputSamples = stageInputMax * 2;

    auto rawCoeffs = convo::makeAlignedArray_nothrow<double>(static_cast<size_t>(stage.taps));
    if (!rawCoeffs) { clearStage(stage); return false; }

    const double beta = (attenuationDb > 50.0)
                      ? (0.1102 * (attenuationDb - 8.7))
                      : ((attenuationDb >= 21.0) ? (0.5842 * std::pow(attenuationDb - 21.0, 0.4) + 0.07886 * (attenuationDb - 21.0))
                                                  : 0.0);
    const double i0Beta = besselI0(beta);
    const int M = stage.centerTap;

    for (int n = 0; n < stage.taps; ++n)
    {
        const double t = static_cast<double>(n - M);
        const double sinc = (n == M)
                          ? 0.5
                          : (std::sin(juce::MathConstants<double>::pi * 0.5 * t) / (juce::MathConstants<double>::pi * t));
        const double frac = static_cast<double>(n - M) / static_cast<double>(M);
        const double window = besselI0(beta * std::sqrt(juce::jmax(0.0, 1.0 - frac * frac))) / i0Beta;
        rawCoeffs[n] = sinc * window;
    }

    for (int n = 0; n < stage.taps; ++n)
    {
        if (n != stage.centerTap && ((n & 1) == stage.centerParity))
            rawCoeffs[n] = 0.0;
    }

    double sum = 0.0;
    for (int i = 0; i < stage.taps; ++i)
        sum += rawCoeffs[i];
    if (std::abs(sum) > 1.0e-20)
    {
        const double inv = 1.0 / sum;
        for (int i = 0; i < stage.taps; ++i)
            rawCoeffs[i] *= inv;
    }

    rawCoeffs[stage.centerTap] = 0.5;
    double nonCenterSum = 0.0;
    for (int i = 0; i < stage.taps; ++i)
        if (i != stage.centerTap) nonCenterSum += rawCoeffs[i];
    if (std::abs(nonCenterSum) > 1.0e-20)
    {
        const double scale = 0.5 / nonCenterSum;
        for (int i = 0; i < stage.taps; ++i)
        {
            if (i != stage.centerTap)
                rawCoeffs[i] *= scale;
        }
    }
    rawCoeffs[stage.centerTap] = 0.5;

    stage.convCount = (stage.taps - stage.convParity + 1) / 2;
    stage.convCoeffs = convo::makeAlignedArray_nothrow<double>(static_cast<size_t>(stage.convCount));
    stage.convCoeffsReversed = convo::makeAlignedArray_nothrow<double>(static_cast<size_t>(stage.convCount));

    if (!stage.convCoeffs || !stage.convCoeffsReversed)
    {
        clearStage(stage);
        return false;
    }

    for (int r = 0; r < stage.convCount; ++r)
    {
        const int k = stage.convParity + (r << 1);
        stage.convCoeffs[r] = (k < stage.taps) ? rawCoeffs[k] : 0.0;
        stage.convCoeffsReversed[stage.convCount - 1 - r] = stage.convCoeffs[r];
    }

    stage.centerCoeff = rawCoeffs[stage.centerTap];
    stage.centerDelayInput = (stage.centerTap - stage.centerParity) / 2;
    stage.historyUpKeep = juce::jmax(stage.convCount - 1, stage.centerDelayInput);
    // loadStride2 が ptr[-6] までアクセスするため、historyDownKeep に +6 マージンを追加
    // ref: doc/work46/bug.md (Bug #1)
    stage.historyDownKeep = juce::jmax(stage.centerTap, stage.convParity + ((stage.convCount - 1) << 1) + 6);

    stage.upHistorySize = stage.historyUpKeep + stage.maxInputSamples + 16;
    stage.downHistorySize = stage.historyDownKeep + stage.maxOutputSamples + 16;

    for (int ch = 0; ch < kMaxChannels; ++ch)
    {
        stage.upHistory[ch] = convo::makeAlignedArray_nothrow<double>(static_cast<size_t>(stage.upHistorySize));
        stage.downHistory[ch] = convo::makeAlignedArray_nothrow<double>(static_cast<size_t>(stage.downHistorySize));
        if (!stage.upHistory[ch] || !stage.downHistory[ch])
        {
            clearStage(stage);
            return false;
        }
        juce::FloatVectorOperations::clear(stage.upHistory[ch].get(), stage.upHistorySize);
        juce::FloatVectorOperations::clear(stage.downHistory[ch].get(), stage.downHistorySize);
    }
    return true;
}

bool CustomInputOversampler::prepareSingleStage(int taps, double attenDb, int stageInputMax) noexcept
{
    release();
    upsampleRatio = 2;
    numStages = 1;
    maxInputBlockSize = stageInputMax;
    maxUpsampledBlockSize = stageInputMax * 2;
    if (!prepareStage(stages[0], taps, attenDb, stageInputMax))
    {
        release();
        return false;
    }
    workCapacity = maxUpsampledBlockSize;
    for (int ch = 0; ch < kMaxChannels; ++ch) {
        workA[ch] = convo::makeAlignedArray_nothrow<double>(static_cast<size_t>(workCapacity));
        workB[ch] = convo::makeAlignedArray_nothrow<double>(static_cast<size_t>(workCapacity));
        if (workA[ch]) juce::FloatVectorOperations::clear(workA[ch].get(), workCapacity);
        if (workB[ch]) juce::FloatVectorOperations::clear(workB[ch].get(), workCapacity);
        blockChannels[ch] = workA[ch].get();
    }
    return true;
}

void CustomInputOversampler::prepare(int newMaxInputBlockSize, int ratio, Preset preset)
{
    release();

    maxInputBlockSize = juce::jmax(1, newMaxInputBlockSize);
    upsampleRatio = sanitizeRatio(ratio);
    activePreset = preset;
    numStages = (upsampleRatio == 8) ? 3 : ((upsampleRatio == 4) ? 2 : ((upsampleRatio == 2) ? 1 : 0));
    maxUpsampledBlockSize = maxInputBlockSize * upsampleRatio;

    int stageInputMax = maxInputBlockSize;
    for (int i = 0; i < numStages; ++i)
    {
        if (!prepareStage(stages[i], tapsForStage(i, activePreset), attenuationForStage(i, activePreset), stageInputMax))
        {
            release();
            return;
        }
        stageInputMax *= 2;
    }

    workCapacity = juce::jmax(1, maxUpsampledBlockSize);
    for (int ch = 0; ch < kMaxChannels; ++ch)
    {
        workA[ch] = convo::makeAlignedArray_nothrow<double>(static_cast<size_t>(workCapacity));
        workB[ch] = convo::makeAlignedArray_nothrow<double>(static_cast<size_t>(workCapacity));
        if (!workA[ch] || !workB[ch])
        {
            release();
            return;
        }
        juce::FloatVectorOperations::clear(workA[ch].get(), workCapacity);
        juce::FloatVectorOperations::clear(workB[ch].get(), workCapacity);
        blockChannels[ch] = workA[ch].get();
    }
}

void CustomInputOversampler::reset() noexcept
{
    for (int i = 0; i < numStages; ++i)
    {
        auto& stage = stages[i];
        for (int ch = 0; ch < kMaxChannels; ++ch)
        {
            if (stage.upHistory[ch]) juce::FloatVectorOperations::clear(stage.upHistory[ch].get(), stage.upHistorySize);
            if (stage.downHistory[ch]) juce::FloatVectorOperations::clear(stage.downHistory[ch].get(), stage.downHistorySize);
        }
    }

    convo::publishAtomic(corruptionDetected, false, std::memory_order_release);
    convo::publishAtomic(consecutiveCorruptionAutoClearCount, static_cast<std::uint32_t>(0), std::memory_order_release);
    convo::publishAtomic(hardFallbackActive, false, std::memory_order_release);
}

void CustomInputOversampler::clearAllStages() noexcept
{
    for (int i = 0; i < numStages; ++i)
    {
        auto& stage = stages[i];
        for (int ch = 0; ch < kMaxChannels; ++ch)
        {
            if (stage.upHistory[ch])
                juce::FloatVectorOperations::clear(stage.upHistory[ch].get(), stage.upHistorySize);
            if (stage.downHistory[ch])
                juce::FloatVectorOperations::clear(stage.downHistory[ch].get(), stage.downHistorySize);
        }
    }

    convo::publishAtomic(corruptionDetected, false, std::memory_order_release);
}

void CustomInputOversampler::markCorruptionDetected() noexcept
{
    convo::fetchAddAtomic(corruptionEventCount, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
    convo::publishAtomic(corruptionDetected, true, std::memory_order_release);
}

void CustomInputOversampler::interpolateStage(const Stage& stage,
                                              const double* input,
                                              int inputSamples,
                                              double* output,
                                              int channel) noexcept
{
    double* history = stage.upHistory[channel].get();
    if (history == nullptr || input == nullptr || output == nullptr)
        return;

    const int keep = stage.historyUpKeep;
    const int capacity = stage.upHistorySize;
    juce::FloatVectorOperations::copy(history + keep, input, inputSamples);

    for (int n = 0; n < inputSamples; ++n)
    {
        const int idx = keep + n;
        if (idx < (stage.convCount - 1) || idx >= capacity)
        {
            markCorruptionDetected();
            output[n * 2 + 0] = 0.0;
            output[n * 2 + 1] = 0.0;
            continue;
        }

        const double* xWindow = history + idx - (stage.convCount - 1);
        double convValue = 0.0;
        bool bad = false;

#if defined(__AVX2__)
        if (stage.convCount >= 4)
        {
            convValue = dotProductAvx2(xWindow, stage.convCoeffsReversed.get(), stage.convCount);
            if (isBadSample(convValue))
                bad = true;
        }
        else
#endif
        {
            for (int r = 0; r < stage.convCount; ++r)
            {
                const double x = xWindow[r];
                if (isBadSample(x))
                {
                    bad = true;
                    break;
                }
                convValue += stage.convCoeffsReversed[r] * x;
            }
        }

        double centerValue = 0.0;
        if (idx >= stage.centerDelayInput)
            centerValue = stage.centerCoeff * history[idx - stage.centerDelayInput];
        else
            bad = true;

        if (bad || isBadSample(centerValue))
        {
            markCorruptionDetected();
            output[n * 2 + 0] = 0.0;
            output[n * 2 + 1] = 0.0;
            continue;
        }

        convValue *= 2.0;
        if (fastAbs(convValue) < kDenormThreshold) convValue = 0.0;
        if (fastAbs(centerValue) < kDenormThreshold) centerValue = 0.0;

        const int outBase = n << 1;

        output[outBase + stage.convParity] = convValue;
        output[outBase + stage.centerParity] = centerValue;
    }

    std::memmove(history, history + inputSamples, static_cast<size_t>(keep) * sizeof(double));
}

void CustomInputOversampler::decimateStage(const Stage& stage,
                                           const double* __restrict input,
                                           int inputSamples,
                                           double* __restrict output,
                                           int channel) noexcept
{
    double* __restrict history = stage.downHistory[channel].get();
    if (history == nullptr || input == nullptr || output == nullptr)
        return;

    const int keep = stage.historyDownKeep;
    const int capacity = stage.downHistorySize;

    // ── [既存] サイレンス最適化パス（変更なし） ──
    bool inputSilent = true;
    for (int i = 0; i < inputSamples; ++i)
    {
        if (fastAbs(input[i]) > kDenormThreshold)
        {
            inputSilent = false;
            break;
        }
    }

    if (inputSilent)
    {
        bool historySilent = true;
        for (int i = 0; i < keep; ++i)
        {
            if (fastAbs(history[i]) > kDenormThreshold)
            {
                historySilent = false;
                break;
            }
        }

        if (historySilent)
        {
            const int outSamples = inputSamples >> 1;
            juce::FloatVectorOperations::clear(output, outSamples);
            juce::FloatVectorOperations::clear(history, keep);
            return;
        }
    }

    juce::FloatVectorOperations::copy(history + keep, input, inputSamples);

    const int outSamples = inputSamples >> 1;
    const double* __restrict coeffs = stage.convCoeffs.get();

    // [Safety Guard] outSamples == 0 の場合、baseMax が不正になるため早期return
    if (outSamples <= 0)
        return;

    // ── P1-1: nループ外部へのバウンドチェック完全外出し ──
    // base = keep + (n << 1) は n に単調増加 → global min/max を事前計算
    const int baseMax = keep + ((outSamples - 1) << 1);

    // centerTap用: base - centerTap >= 0 → keep >= centerTap; base < capacity → baseMax < capacity
    const bool centerTapOk = (keep >= stage.centerTap) && (baseMax < capacity);

    // convタップ範囲: 最小convIndex = n=0,r=convCount-1; 最大convIndex = n=outSamples-1,r=0
    // 注: globalMinConvIdx は loadStride2 の ptr[-6] を考慮しないスカラー最低位置。
    //     AVX2 パスの実最低アクセスは index 0 となる（prepareStage の +6 マージン保証）。
    //     globalMinConvIdx >= 0 はスカラー経路の安全条件として十分であり、
    //     +6 マージンは prepareStage 側で historyDownKeep に組み込まれている。
    const int globalMinConvIdx = keep - stage.convParity - ((stage.convCount - 1) << 1);
    const int globalMaxConvIdx = baseMax - stage.convParity;
    const bool convTapOk = (globalMinConvIdx >= 0) && (globalMaxConvIdx < capacity);

    if (!centerTapOk || !convTapOk || stage.convCount <= 0)
    {
        // ブロック全体が境界違反 → 全出力0クリア
        std::memset(output, 0, static_cast<size_t>(outSamples) * sizeof(double));
        markCorruptionDetected();
        return;
    }

    // ── nループ（境界安全100%保証済み） ──
    for (int n = 0; n < outSamples; ++n)
    {
        const int base = keep + (n << 1);

        // centerCoeffは動的な履歴値に依存 → nループ内でチェック
        const double centerSample = history[base - stage.centerTap];
        double acc = stage.centerCoeff * centerSample;
        if (isBadSample(acc))
        {
            output[n] = 0.0;
            markCorruptionDetected();
            continue;
        }

        // ── P1-2: stride-2 dot product（AVX2） ──
#if defined(__AVX2__) && defined(__FMA__)
        if (stage.convCount >= 8)
        {
            // 8重アンロール dotProductDecimateAvx2（convCount>=8 で効果的）
            __assume(stage.convCount >= 8);
            acc += dotProductDecimateAvx2(
                history + (base - stage.convParity),
                coeffs,
                stage.convCount);
        }
        else if (stage.convCount >= 4)
        {
            // convCount 4〜7: 簡易AVX2（unrollなしのloadStride2）
            __m256d vAcc = _mm256_setzero_pd();
            int r = 0;
            const int simdEnd = (stage.convCount / 4) * 4;
            for (; r < simdEnd; r += 4)
            {
                __m256d vS = loadStride2(
                    history + (base - stage.convParity) - (r << 1));
                __m256d vC = _mm256_loadu_pd(coeffs + r);
                vAcc = _mm256_fmadd_pd(vS, vC, vAcc);
            }
            // 水平加算（vextractf128 + hadd）
            __m128d vLo = _mm256_castpd256_pd128(vAcc);
            __m128d vHi = _mm256_extractf128_pd(vAcc, 1);
            __m128d vSum = _mm_add_pd(vLo, vHi);
            vSum = _mm_hadd_pd(vSum, vSum);
            acc += _mm_cvtsd_f64(vSum);
            // スカラー剰余
            for (; r < stage.convCount; ++r)
                acc += coeffs[r] * history[base - stage.convParity - (r << 1)];
        }
        else
#endif
        {
            // convCount < 4: スカラーパス（バリデーション済み、チェックなし）
            for (int r = 0; r < stage.convCount; ++r)
                acc += coeffs[r] * history[base - stage.convParity - (r << 1)];
        }

        // ── 事後処理（bad sample + denormal） ──
        if (isBadSample(acc))
        {
            output[n] = 0.0;
            markCorruptionDetected();
        }
        else
        {
            if (fastAbs(acc) < kDenormThreshold) acc = 0.0;
            output[n] = acc;
        }
    }

    // ── [既存] 履歴シフト（変更なし） ──
    std::memmove(history, history + inputSamples, static_cast<size_t>(keep) * sizeof(double));
}

juce::dsp::AudioBlock<double> CustomInputOversampler::processUp(juce::dsp::AudioBlock<double>& inputBlock, int numChannels) noexcept
{

    if (convo::consumeAtomic(hardFallbackActive, std::memory_order_acquire))
    {
        const int channels = juce::jlimit(1, kMaxChannels, numChannels);
        const int inSamples = static_cast<int>(inputBlock.getNumSamples());
        blockChannels[0] = inputBlock.getChannelPointer(0);
        blockChannels[1] = (channels > 1) ? inputBlock.getChannelPointer(1)
                                          : blockChannels[0];
        blockChannelView[0] = blockChannels[0].get();
        blockChannelView[1] = blockChannels[1].get();
        return { blockChannelView, static_cast<size_t>(channels), static_cast<size_t>(inSamples) };
    }

    const int channels = juce::jlimit(1, kMaxChannels, numChannels);
    const int inSamples = static_cast<int>(inputBlock.getNumSamples());

    // [Safety Guard] 入力サイズが準備された容量を超えている場合は空ブロックを返す
    if (inSamples > maxInputBlockSize)
    {
        // バッファサイズ不足時は無音を返す (AudioEngine側で再構築をリクエストする)
        return {};
    }

    if (upsampleRatio <= 1 || numStages == 0)
    {
        blockChannels[0] = inputBlock.getChannelPointer(0);
        blockChannels[1] = (channels > 1) ? inputBlock.getChannelPointer(1)
                                          : blockChannels[0];
        blockChannelView[0] = blockChannels[0].get();
        blockChannelView[1] = blockChannels[1].get();
        return { blockChannelView, static_cast<size_t>(channels), static_cast<size_t>(inSamples) };
    }

    double* currIn[2] = { inputBlock.getChannelPointer(0),
                          (channels > 1) ? inputBlock.getChannelPointer(1) : inputBlock.getChannelPointer(0) };
    int currSamples = inSamples;

    for (int stageIndex = 0; stageIndex < numStages; ++stageIndex)
    {
        const bool writeToA = ((stageIndex & 1) == 0);
        double* stageOut[2] = { writeToA ? workA[0].get() : workB[0].get(),
                                writeToA ? workA[1].get() : workB[1].get() };

        for (int ch = 0; ch < channels; ++ch)
            interpolateStage(stages[stageIndex], currIn[ch], currSamples, stageOut[ch], ch);

        currIn[0] = stageOut[0];
        currIn[1] = stageOut[1];
        currSamples <<= 1;
    }

    blockChannels[0] = currIn[0];
    blockChannels[1] = currIn[1];
    blockChannelView[0] = blockChannels[0].get();
    blockChannelView[1] = blockChannels[1].get();
    return { blockChannelView, static_cast<size_t>(channels), static_cast<size_t>(currSamples) };
}

void CustomInputOversampler::processDown(const juce::dsp::AudioBlock<double>& upsampledBlock,
                                         juce::dsp::AudioBlock<double>& outputBlock,
                                         int numChannels) noexcept
{
    if (convo::consumeAtomic(hardFallbackActive, std::memory_order_acquire))
    {
        const int channels = juce::jlimit(1, kMaxChannels, numChannels);
        const int targetSamples = static_cast<int>(outputBlock.getNumSamples());
        const int copySamples = std::min(targetSamples, static_cast<int>(upsampledBlock.getNumSamples()));
        for (int ch = 0; ch < channels; ++ch)
        {
            double* dst = outputBlock.getChannelPointer(ch);
            const double* src = upsampledBlock.getChannelPointer(ch);
            if (copySamples > 0)
                std::memcpy(dst, src, static_cast<size_t>(copySamples) * sizeof(double));
            if (copySamples < targetSamples)
                juce::FloatVectorOperations::clear(dst + copySamples, targetSamples - copySamples);
        }
        for (int ch = channels; ch < static_cast<int>(outputBlock.getNumChannels()); ++ch)
            juce::FloatVectorOperations::clear(outputBlock.getChannelPointer(ch), targetSamples);
        return;
    }

    if (convo::exchangeAtomic(corruptionDetected, false, std::memory_order_acq_rel))
    {
        convo::fetchAddAtomic(corruptionAutoClearCount, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
        const std::uint32_t consecutive = convo::fetchAddAtomic(consecutiveCorruptionAutoClearCount,
                                                                static_cast<std::uint32_t>(1),
                                                                std::memory_order_acq_rel) + 1u;
        if (consecutive >= kHardFallbackAutoClearThreshold)
            convo::publishAtomic(hardFallbackActive, true, std::memory_order_release);
        clearAllStages();
        outputBlock.clear();
        return;
    }

    convo::publishAtomic(consecutiveCorruptionAutoClearCount, static_cast<std::uint32_t>(0), std::memory_order_release);

    const int channels = juce::jlimit(1, kMaxChannels, numChannels);
    const int targetSamples = static_cast<int>(outputBlock.getNumSamples());

    // [Safety Guard] 入力サイズが準備された容量を超えている場合は処理をスキップし、出力をクリアする
    if (upsampledBlock.getNumSamples() > static_cast<size_t>(maxUpsampledBlockSize))
    {
        markCorruptionDetected();
        outputBlock.clear();
        return;
    }

    if (upsampleRatio <= 1 || numStages == 0)
    {
        for (int ch = 0; ch < channels; ++ch)
        {
            double* dst = outputBlock.getChannelPointer(ch);
            const double* src = upsampledBlock.getChannelPointer(ch);
            std::memcpy(dst, src, static_cast<size_t>(targetSamples) * sizeof(double));
        }
        return;
    }

    const double* currIn[2] = { upsampledBlock.getChannelPointer(0),
                                (channels > 1) ? upsampledBlock.getChannelPointer(1) : upsampledBlock.getChannelPointer(0) };
    int currSamples = static_cast<int>(upsampledBlock.getNumSamples());

    for (int stageIndex = numStages - 1; stageIndex >= 0; --stageIndex)
    {
        const bool writeToA = (((numStages - 1 - stageIndex) & 1) == 0);
        double* stageOut[2] = { writeToA ? workA[0].get() : workB[0].get(),
                                writeToA ? workA[1].get() : workB[1].get() };

        for (int ch = 0; ch < channels; ++ch)
            decimateStage(stages[stageIndex], currIn[ch], currSamples, stageOut[ch], ch);

        currIn[0] = stageOut[0];
        currIn[1] = stageOut[1];
        currSamples >>= 1;
    }

    for (int ch = 0; ch < channels; ++ch)
    {
        double* dst = outputBlock.getChannelPointer(ch);
        const double* src = currIn[ch];
        if (dst != src)
            std::memcpy(dst, src, static_cast<size_t>(targetSamples) * sizeof(double));
    }
}
