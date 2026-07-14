//============================================================================
#include <JuceHeader.h>
#include <cmath>
#include <cstring>
#include <immintrin.h>

#include "TruePeakDetector.h"

//============================================================================
TruePeakDetector::~TruePeakDetector()
{
    for (auto& stage : stages)
    {
        stage.convCoeffs.reset();
        stage.convCoeffsReversed.reset();
        for (int ch = 0; ch < kMaxChannels; ++ch)
            stage.upHistory[ch].reset();
    }
    upsampleBuffer.reset();
}

void TruePeakDetector::prepare(double sampleRate, int maxBlockSize, int taps)
{
    convo::publishAtomic(currentSampleRate, sampleRate, std::memory_order_release);

    // ★ レイアウト [Stage0L | Stage0R | Stage1L | Stage1R] = 2N+2N+4N+4N = 12N
    //    constexpr 導出: Stage0Channels(2)*UpsampleFactor1(2) + Stage1Channels(2)*UpsampleFactor2(4) = 12
    constexpr int kStage0Channels = 2;
    constexpr int kStage1Channels = 2;
    constexpr int kUpsampleFactor1 = 2;
    constexpr int kUpsampleFactor2 = 4;
    constexpr int kWorkBufferMultiplier =
        kStage0Channels * kUpsampleFactor1 +
        kStage1Channels * kUpsampleFactor2;

    const int upBufferSize = maxBlockSize * kWorkBufferMultiplier;
    if (upBufferSize > bufferCapacity || !upsampleBuffer)
    {
        upsampleBuffer = convo::makeAlignedArray<double>(static_cast<size_t>(upBufferSize));
        bufferCapacity = upBufferSize;
    }

    // 2段の2倍OSで4倍を構成
    int stageInputMax = maxBlockSize;
    for (int i = 0; i < 2; ++i)
    {
        const int stageTaps = (i == 0) ? taps : std::max(15, taps / 2);
        prepareStage(stages[i], stageTaps, kDefaultAttenuationDb, stageInputMax);
        stageInputMax *= 2;
    }
    upsampledCapacity = maxBlockSize * kWorkBufferMultiplier;
    reset();
}

void TruePeakDetector::reset() noexcept
{
    peakHold = 0.0;
    for (auto& stage : stages)
    {
        for (int ch = 0; ch < kMaxChannels; ++ch)
        {
            if (stage.upHistory[ch])
                juce::FloatVectorOperations::clear(stage.upHistory[ch].get(), stage.upHistorySize);
        }
    }
}

// ★ scanPeak: |buf[i]| の最大値を求めるヘルパー (匿名名前空間、static linkage)
//    RT 初回初期化 (guard variable) を避けるため static local は使用しない。
namespace {

double scanPeak(const double* buf, int n) noexcept
{
    double peak = 0.0;
#if defined(__AVX2__)
    // ローカル変数 _mm256_set1_pd は 1 命令 (vsetpd) で実質ゼロコスト
    const __m256d signMask = _mm256_set1_pd(-0.0);
    __m256d vPeak = _mm256_setzero_pd();
    int i = 0;
    for (; i <= n - 4; i += 4) {
        __m256d v = _mm256_andnot_pd(signMask, _mm256_loadu_pd(buf + i));
        vPeak = _mm256_max_pd(vPeak, v);
    }
    alignas(32) double tmp[4];
    _mm256_store_pd(tmp, vPeak);
    for (int j = 0; j < 4; ++j) if (tmp[j] > peak) peak = tmp[j];
    for (; i < n; ++i) { double v = std::abs(buf[i]); if (v > peak) peak = v; }
#else
    for (int i = 0; i < n; ++i) { double v = std::abs(buf[i]); if (v > peak) peak = v; }
#endif
    return peak;
}

} // anonymous namespace

double TruePeakDetector::processBlock(const double* dataL, const double* dataR, int numSamples) noexcept
{
    if (numSamples <= 0 || !upsampleBuffer)
        return 0.0;

    double* work = upsampleBuffer.get();
    const int up1Samples = numSamples * 2;
    const int up2Samples = numSamples * 4;

    // オフセット: work 領域のレイアウト
    //   [ Stage0 L | Stage0 R | Stage1 L | Stage1 R ]
    //   Stage0 は zero-offset、それ以外は up1Samples/up2Samples に依存する runtime 値
    constexpr int kStage0LOffset = 0;
    const int   kStage0ROffset = up1Samples;
    const int   kStage1LOffset = up1Samples * 2;
    const int   kStage1ROffset = up1Samples * 2 + up2Samples;

    // Stage 0: 1x -> 2x (L)
    interpolateStage(stages[0], dataL, numSamples, work + kStage0LOffset, 0);
    // Stage 0: 1x -> 2x (R)
    if (dataR != nullptr)
        interpolateStage(stages[0], dataR, numSamples, work + kStage0ROffset, 1);
    else
        interpolateStage(stages[0], dataL, numSamples, work + kStage0ROffset, 1);

    // Stage 1: 2x -> 4x (L + R)
    interpolateStage(stages[1], work + kStage0LOffset, up1Samples, work + kStage1LOffset, 0);  // L
    interpolateStage(stages[1], work + kStage0ROffset, up1Samples, work + kStage1ROffset, 1);  // R

    // Peak scan: L/R 別領域で独立実行
    double peakL = scanPeak(work + kStage1LOffset, up2Samples);
    double peakR = scanPeak(work + kStage1ROffset, up2Samples);
    double peak = std::max(peakL, peakR);

    // ピークホールド（指数平滑）
    if (peak > peakHold)
        peakHold = peak;
    else
        peakHold *= 0.999; // 減衰時定数: 約1000サンプル

    return peakHold;
}

//==============================================================================
// 内部実装: Kaiser窓 FIR halfband フィルタ
//==============================================================================
double TruePeakDetector::besselI0(double x) noexcept
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

double TruePeakDetector::dotProductAvx2(const double* __restrict x,
                                        const double* __restrict coeffs,
                                        int n) noexcept
{
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    __m256d acc2 = _mm256_setzero_pd();
    __m256d acc3 = _mm256_setzero_pd();
    int i = 0;
    for (; i <= n - 16; i += 16)
    {
        _mm_prefetch(reinterpret_cast<const char*>(x + i + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(coeffs + i + 64), _MM_HINT_T0);
        acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i),      _mm256_load_pd(coeffs + i),      acc0);
        acc1 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i + 4),  _mm256_load_pd(coeffs + i + 4),  acc1);
        acc2 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i + 8),  _mm256_load_pd(coeffs + i + 8),  acc2);
        acc3 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i + 12), _mm256_load_pd(coeffs + i + 12), acc3);
    }
    for (; i <= n - 4; i += 4)
        acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i), _mm256_load_pd(coeffs + i), acc0);
    acc0 = _mm256_add_pd(acc0, acc1);
    acc2 = _mm256_add_pd(acc2, acc3);
    acc0 = _mm256_add_pd(acc0, acc2);
    __m128d vLo = _mm256_castpd256_pd128(acc0);
    __m128d vHi = _mm256_extractf128_pd(acc0, 1);
    __m128d vSum = _mm_add_pd(vLo, vHi);
    vSum = _mm_hadd_pd(vSum, vSum);
    double sum = _mm_cvtsd_f64(vSum);
    for (; i < n; ++i)
        sum += x[i] * coeffs[i];
    return sum;
}

void TruePeakDetector::prepareStage(Stage& stage, int taps, double attenuationDb, int stageInputMax)
{
    stage.convCoeffs.reset();
    stage.convCoeffsReversed.reset();
    for (int ch = 0; ch < kMaxChannels; ++ch)
        stage.upHistory[ch].reset();

    stage.taps = juce::jmax(3, taps | 1);
    stage.centerTap = (stage.taps - 1) / 2;
    stage.centerParity = stage.centerTap & 1;
    stage.convParity = 1 - stage.centerParity;
    stage.maxInputSamples = stageInputMax;
    stage.maxOutputSamples = stageInputMax * 2;

    auto rawCoeffs = convo::makeAlignedArray<double>(static_cast<size_t>(stage.taps));
    if (!rawCoeffs) return;

    const double beta = (attenuationDb > 50.0)
        ? (0.1102 * (attenuationDb - 8.7))
        : ((attenuationDb >= 21.0)
            ? (0.5842 * std::pow(attenuationDb - 21.0, 0.4) + 0.07886 * (attenuationDb - 21.0))
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
    stage.convCoeffs = convo::makeAlignedArray<double>(static_cast<size_t>(stage.convCount));
    stage.convCoeffsReversed = convo::makeAlignedArray<double>(static_cast<size_t>(stage.convCount));
    if (!stage.convCoeffs || !stage.convCoeffsReversed) return;

    for (int r = 0; r < stage.convCount; ++r)
    {
        const int k = stage.convParity + (r << 1);
        stage.convCoeffs[r] = (k < stage.taps) ? rawCoeffs[k] : 0.0;
        stage.convCoeffsReversed[stage.convCount - 1 - r] = stage.convCoeffs[r];
    }

    stage.centerCoeff = rawCoeffs[stage.centerTap];
    stage.centerDelayInput = (stage.centerTap - stage.centerParity) / 2;
    stage.historyUpKeep = juce::jmax(stage.convCount - 1, stage.centerDelayInput);
    stage.upHistorySize = stage.historyUpKeep + stage.maxInputSamples + 16;

    for (int ch = 0; ch < kMaxChannels; ++ch)
    {
        stage.upHistory[ch] = convo::makeAlignedArray<double>(static_cast<size_t>(stage.upHistorySize));
        if (stage.upHistory[ch])
            juce::FloatVectorOperations::clear(stage.upHistory[ch].get(), stage.upHistorySize);
    }
}

void TruePeakDetector::interpolateStage(const Stage& stage,
                                        const double* input, int inputSamples,
                                        double* output, int channel) noexcept
{
    auto* history = stage.upHistory[channel].get();
    if (!history || !stage.convCoeffs) return;

    const int histLen = stage.historyUpKeep;
    const int convCnt = stage.convCount;
    const int centerDelay = stage.centerDelayInput;
    const double cCoeff = stage.centerCoeff;
    const int convParity = stage.convParity;

    // 履歴シフト
    std::memmove(history, history + inputSamples, static_cast<size_t>(histLen) * sizeof(double));
    std::memcpy(history + histLen, input, static_cast<size_t>(inputSamples) * sizeof(double));

    for (int n = 0; n < inputSamples; ++n)
    {
        const double* base = history + histLen + n - centerDelay;
        const double even = base[0] * cCoeff + dotProductAvx2(base - convParity, stage.convCoeffsReversed.get(), convCnt);
        const double odd  = base[1] * cCoeff + dotProductAvx2(base - 1 + convParity, stage.convCoeffsReversed.get(), convCnt);
        output[n * 2]     = even;
        output[n * 2 + 1] = odd;
    }
}
