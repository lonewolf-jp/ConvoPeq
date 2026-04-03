//============================================================================
#include <JuceHeader.h>
#include "CustomInputOversampler.h"
#include "DspNumericPolicy.h"

#include <bit>
#include <cmath>
#include <cstring>
#include <immintrin.h>

namespace
{
    static constexpr int kMaxChannels = 2;

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
    corruptionDetected.store(false, std::memory_order_release);
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
        _mm_prefetch(reinterpret_cast<const char*>(x + i + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(coeffs + i + 64), _MM_HINT_T0);

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

    alignas(64) double partial[4];
    _mm256_store_pd(partial, acc0);
    double sum = partial[0] + partial[1] + partial[2] + partial[3];

    // Scalar remainder
    for (; i < n; ++i)
        sum += x[i] * coeffs[i];

    if (isBadSample(sum))
        sum = 0.0;
    else if (fastAbs(sum) < kDenormThreshold)
        sum = 0.0;

    return sum;
}

void CustomInputOversampler::prepareStage(Stage& stage, int taps, double attenuationDb, int stageInputMax)
{
    clearStage(stage);

    stage.taps = juce::jmax(3, taps | 1);
    stage.centerTap = (stage.taps - 1) / 2;
    stage.centerParity = stage.centerTap & 1;
    stage.convParity = 1 - stage.centerParity;
    stage.maxInputSamples = stageInputMax;
    stage.maxOutputSamples = stageInputMax * 2;

    convo::ScopedAlignedPtr<double> rawCoeffs(static_cast<double*>(convo::aligned_malloc(static_cast<size_t>(stage.taps) * sizeof(double), 64)));

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
    stage.convCoeffs.reset(static_cast<double*>(convo::aligned_malloc(static_cast<size_t>(stage.convCount) * sizeof(double), 64)));
    stage.convCoeffsReversed.reset(static_cast<double*>(convo::aligned_malloc(static_cast<size_t>(stage.convCount) * sizeof(double), 64)));

    if (!stage.convCoeffs || !stage.convCoeffsReversed)
    {
        clearStage(stage);
        return;
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
    stage.historyDownKeep = juce::jmax(stage.centerTap, stage.convParity + ((stage.convCount - 1) << 1));

    stage.upHistorySize = stage.historyUpKeep + stage.maxInputSamples + 16;
    stage.downHistorySize = stage.historyDownKeep + stage.maxOutputSamples + 16;

    for (int ch = 0; ch < kMaxChannels; ++ch)
    {
        stage.upHistory[ch].reset(static_cast<double*>(convo::aligned_malloc(static_cast<size_t>(stage.upHistorySize) * sizeof(double), 64)));
        stage.downHistory[ch].reset(static_cast<double*>(convo::aligned_malloc(static_cast<size_t>(stage.downHistorySize) * sizeof(double), 64)));
        if (stage.upHistory[ch]) juce::FloatVectorOperations::clear(stage.upHistory[ch].get(), stage.upHistorySize);
        if (stage.downHistory[ch]) juce::FloatVectorOperations::clear(stage.downHistory[ch].get(), stage.downHistorySize);
    }
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
        prepareStage(stages[i], tapsForStage(i, activePreset), attenuationForStage(i, activePreset), stageInputMax);
        stageInputMax *= 2;
    }

    workCapacity = juce::jmax(1, maxUpsampledBlockSize);
    for (int ch = 0; ch < kMaxChannels; ++ch)
    {
        workA[ch].reset(static_cast<double*>(convo::aligned_malloc(static_cast<size_t>(workCapacity) * sizeof(double), 64)));
        workB[ch].reset(static_cast<double*>(convo::aligned_malloc(static_cast<size_t>(workCapacity) * sizeof(double), 64)));
        if (workA[ch]) juce::FloatVectorOperations::clear(workA[ch].get(), workCapacity);
        if (workB[ch]) juce::FloatVectorOperations::clear(workB[ch].get(), workCapacity);
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

    corruptionDetected.store(false, std::memory_order_release);
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

    corruptionDetected.store(false, std::memory_order_release);
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
            corruptionDetected.store(true, std::memory_order_relaxed);
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
            corruptionDetected.store(true, std::memory_order_relaxed);
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
                                           const double* input,
                                           int inputSamples,
                                           double* output,
                                           int channel) noexcept
{
    double* history = stage.downHistory[channel].get();
    if (history == nullptr || input == nullptr || output == nullptr)
        return;

    const int keep = stage.historyDownKeep;
    const int capacity = stage.downHistorySize;
    juce::FloatVectorOperations::copy(history + keep, input, inputSamples);

    const int outSamples = inputSamples >> 1;
    const double* coeffs = stage.convCoeffs.get();
    for (int n = 0; n < outSamples; ++n)
    {
        const int base = keep + (n << 1);
        if (base < stage.centerTap || base >= capacity)
        {
            output[n] = 0.0;
            corruptionDetected.store(true, std::memory_order_relaxed);
            continue;
        }

        double acc = stage.centerCoeff * history[base - stage.centerTap];
        if (isBadSample(acc))
        {
            output[n] = 0.0;
            corruptionDetected.store(true, std::memory_order_relaxed);
            continue;
        }

        bool bad = false;
        bool usedAvxPath = false;

#if defined(__AVX2__) && defined(__FMA__)
        if (stage.convCount >= 4)
        {
            usedAvxPath = true;
            __m256d vAcc = _mm256_setzero_pd();
            int r = 0;
            const int simdEnd = (stage.convCount / 4) * 4;
            for (; r < simdEnd; r += 4)
            {
                const int idx0 = base - stage.convParity - ((r + 0) << 1);
                const int idx1 = base - stage.convParity - ((r + 1) << 1);
                const int idx2 = base - stage.convParity - ((r + 2) << 1);
                const int idx3 = base - stage.convParity - ((r + 3) << 1);
                if (idx0 < 0 || idx0 >= capacity || idx1 < 0 || idx1 >= capacity ||
                    idx2 < 0 || idx2 >= capacity || idx3 < 0 || idx3 >= capacity)
                {
                    bad = true;
                    break;
                }

                const double s0 = history[idx0];
                const double s1 = history[idx1];
                const double s2 = history[idx2];
                const double s3 = history[idx3];
                if (isBadSample(s0) || isBadSample(s1) || isBadSample(s2) || isBadSample(s3))
                {
                    bad = true;
                    break;
                }

                const __m256d vSamples = _mm256_set_pd(s3, s2, s1, s0);
                const __m256d vCoeffs  = _mm256_loadu_pd(coeffs + r);
                vAcc = _mm256_fmadd_pd(vSamples, vCoeffs, vAcc);
            }

            if (!bad)
            {
                alignas(64) double partial[4];
                _mm256_store_pd(partial, vAcc);
                acc += partial[0] + partial[1] + partial[2] + partial[3];

                for (; r < stage.convCount; ++r)
                {
                    const int idx = base - stage.convParity - (r << 1);
                    if (idx < 0 || idx >= capacity)
                    {
                        bad = true;
                        break;
                    }
                    const double x = history[idx];
                    if (isBadSample(x))
                    {
                        bad = true;
                        break;
                    }
                    acc += coeffs[r] * x;
                }
            }
        }
#endif

        if (!usedAvxPath || bad)
        {
            bad = false;
            acc = stage.centerCoeff * history[base - stage.centerTap];
            if (isBadSample(acc))
                bad = true;

            if (!bad)
            {
                for (int r = 0; r < stage.convCount; ++r)
                {
                    const int idx = base - stage.convParity - (r << 1);
                    if (idx < 0 || idx >= capacity)
                    {
                        bad = true;
                        break;
                    }
                    const double x = history[idx];
                    if (isBadSample(x))
                    {
                        bad = true;
                        break;
                    }
                    acc += coeffs[r] * x;
                }
            }
        }

        if (bad)
        {
            output[n] = 0.0;
            corruptionDetected.store(true, std::memory_order_relaxed);
            continue;
        }

        if (fastAbs(acc) < kDenormThreshold) acc = 0.0;
        output[n] = acc;
    }

    std::memmove(history, history + inputSamples, static_cast<size_t>(keep) * sizeof(double));
}

juce::dsp::AudioBlock<double> CustomInputOversampler::processUp(const juce::dsp::AudioBlock<double>& inputBlock, int numChannels) noexcept
{
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
        blockChannels[0] = const_cast<double*>(inputBlock.getChannelPointer(0));
        blockChannels[1] = (channels > 1) ? const_cast<double*>(inputBlock.getChannelPointer(1))
                                          : blockChannels[0];
        return { blockChannels, static_cast<size_t>(channels), static_cast<size_t>(inSamples) };
    }

    const double* currIn[2] = { inputBlock.getChannelPointer(0),
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

    blockChannels[0] = const_cast<double*>(currIn[0]);
    blockChannels[1] = const_cast<double*>(currIn[1]);
    return { blockChannels, static_cast<size_t>(channels), static_cast<size_t>(currSamples) };
}

void CustomInputOversampler::processDown(const juce::dsp::AudioBlock<double>& upsampledBlock,
                                         juce::dsp::AudioBlock<double>& outputBlock,
                                         int numChannels) noexcept
{
    if (corruptionDetected.exchange(false, std::memory_order_acq_rel))
    {
        clearAllStages();
        outputBlock.clear();
        return;
    }

    const int channels = juce::jlimit(1, kMaxChannels, numChannels);
    const int targetSamples = static_cast<int>(outputBlock.getNumSamples());

    // [Safety Guard] 入力サイズが準備された容量を超えている場合は処理をスキップし、出力をクリアする
    if (upsampledBlock.getNumSamples() > static_cast<size_t>(maxUpsampledBlockSize))
    {
        corruptionDetected.store(true, std::memory_order_relaxed);
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
