#include <JuceHeader.h>
#include "AudioEngine.h"
#include "InputBitDepthTransform.h"
#include "core/RCUReader.h"

extern std::atomic<bool> gShuttingDown;

static thread_local convo::RCUReader tls_rcuReader;

static void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}

static void retireDSP(AudioEngine::DSPCore* dsp)
{
    if (dsp) convo::retireObject(dsp, [](void* p) { delete static_cast<AudioEngine::DSPCore*>(p); });
}

template <typename T>
static inline T* sanitizeRawPtr(T* ptr) noexcept
{
    constexpr uintptr_t kInvalidAllOnes = ~static_cast<uintptr_t>(0);
    return (reinterpret_cast<uintptr_t>(ptr) == kInvalidAllOnes) ? nullptr : ptr;
}

namespace
{
    static constexpr std::array<double, convo::FixedNoiseShaper::ORDER> kFixedNoiseShaperTunedCoeffs
    {
        0.46, 0.28, 0.17, 0.09
    };

    static constexpr std::array<double, convo::Fixed15TapNoiseShaper::ORDER> kFixed15TapNoiseShaperTunedCoeffs
    {
        2.033, -2.165, 1.959, -1.590, 1.221, -0.886, 0.604, -0.389, 0.235, -0.132, 0.068, -0.031, 0.012, -0.004, 0.001, 0.0
    };

    static constexpr std::array<double, kAdaptiveNoiseShaperOrder> kDefaultAdaptiveNoiseShaperCoeffs
    {
        0.82, -0.68, 0.55, -0.43, 0.33, -0.25, 0.18, -0.12, 0.07
    };

    inline double absNoLibm(double x) noexcept
    {
        union { double d; uint64_t u; } v { x };
        v.u &= 0x7FFFFFFFFFFFFFFFULL;
        return v.d;
    }

    inline double absDiffNoLibm(double a, double b) noexcept
    {
        return absNoLibm(a - b);
    }

    inline bool isFiniteNoLibm(double x) noexcept
    {
        union { double d; uint64_t u; } v { x };
        return ((v.u >> 52) & 0x7FFu) != 0x7FFu;
    }

    inline bool isFiniteAndAbsBelowNoLibm(double x, double threshold) noexcept
    {
        return isFiniteNoLibm(x) && (absNoLibm(x) < threshold);
    }

    static_assert(CustomInputOversampler::isLinearPhaseFIR
                  && CustomInputOversampler::isSymmetricUpDown,
                  "Oversampling latency formula assumes symmetric linear-phase FIR with identical up/down taps");

    inline double estimateOversamplingLatencySamplesImpl(int oversamplingFactor,
                                                         AudioEngine::OversamplingType oversamplingType,
                                                         double baseSampleRate) noexcept
    {
        if (oversamplingFactor <= 1 || baseSampleRate <= 0.0)
            return 0.0;

        const int numStages = (oversamplingFactor == 8) ? 3 : ((oversamplingFactor == 4) ? 2 : ((oversamplingFactor == 2) ? 1 : 0));
        if (numStages <= 0)
            return 0.0;

        const int* taps = nullptr;
        static constexpr int iirLikeTaps[3] = { 511, 127, 31 };
        static constexpr int linearPhaseTaps[3] = { 1023, 255, 63 };
        taps = (oversamplingType == AudioEngine::OversamplingType::LinearPhase) ? linearPhaseTaps : iirLikeTaps;

        double totalLatencyBaseSamples = 0.0;
        for (int stage = 0; stage < numStages; ++stage)
        {
            const double stageRate = baseSampleRate * static_cast<double>(1 << (stage + 1));
            const double groupDelaySamplesAtStageRate = static_cast<double>(taps[stage] - 1); // up + down
            const double delayBaseSamples = groupDelaySamplesAtStageRate * (baseSampleRate / stageRate);
            totalLatencyBaseSamples += delayBaseSamples;
        }

        return totalLatencyBaseSamples;
    }
}

inline void pushAdaptiveCaptureBlocks(LockFreeRingBuffer<AudioBlock, 4096>* captureQueue,
                                          const double* left,
                                          const double* right,
                                          int numSamples,
                                          int sampleRateHz,
                                          int bitDepth,
                                          int adaptiveCoeffBankIndex,
                                          uint64_t captureSessionId) noexcept
    {
        if (captureQueue == nullptr || left == nullptr || numSamples <= 0)
            return;

        static std::atomic<uint64_t> dropCount { 0 };

        static constexpr int kBlockSize = 256;
        for (int offset = 0; offset < numSamples; offset += kBlockSize)
        {
            const int currentBlockSize = std::min(kBlockSize, numSamples - offset);
            const double* srcL = left + offset;
            const double* srcR = (right != nullptr) ? (right + offset) : srcL;

            if (!captureQueue->pushWithWriter([&](AudioBlock& block) noexcept
            {
                block.numSamples = currentBlockSize;
                block.sampleRateHz = sampleRateHz;
                block.bitDepth = bitDepth;
                block.adaptiveCoeffBankIndex = adaptiveCoeffBankIndex;
                block.sessionId = captureSessionId;

                const int simdCount = currentBlockSize & ~3;
                int i = 0;

                for (; i < simdCount; i += 4)
                {
                    __m256d v = _mm256_loadu_pd(srcL + i);
                    _mm256_storeu_pd(block.L + i, v);
                }
                for (; i < currentBlockSize; ++i)
                    block.L[i] = srcL[i];

                i = 0;
                for (; i < simdCount; i += 4)
                {
                    __m256d v = _mm256_loadu_pd(srcR + i);
                    _mm256_storeu_pd(block.R + i, v);
                }
                for (; i < currentBlockSize; ++i)
                    block.R[i] = srcR[i];
            }))
            {
                dropCount.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

namespace TanhApprox {
    constexpr double NUM_A = 10395.0;
    constexpr double NUM_B = 1260.0;
    constexpr double NUM_C = 21.0;
    constexpr double DEN_A = 10395.0;
    constexpr double DEN_B = 4725.0;
    constexpr double DEN_C = 210.0;
    constexpr double CLIP_THRESHOLD = 4.5;
}


inline void applyGainRamp(double* __restrict data, int numSamples,
                              double startGain, double increment) noexcept
    {
        __m256d vGain = _mm256_set_pd(startGain + 3.0 * increment,
                                       startGain + 2.0 * increment,
                                       startGain + increment,
                                       startGain);
        const __m256d vInc4 = _mm256_set1_pd(4.0 * increment);

        int i = 0;
        const int vEnd = numSamples / 4 * 4;
        for (; i < vEnd; i += 4)
        {
            __m256d vData = _mm256_loadu_pd(data + i);
            _mm256_storeu_pd(data + i, _mm256_mul_pd(vData, vGain));
            vGain = _mm256_add_pd(vGain, vInc4);
        }

        double gain = startGain + static_cast<double>(i) * increment;
        for (; i < numSamples; ++i) { data[i] *= gain; gain += increment; }
    }

inline void applyGainRamp(float* __restrict data, int numSamples,
                              float startGain, float increment) noexcept
    {
        __m256 vGain = _mm256_set_ps(startGain + 7.0f * increment,
                                     startGain + 6.0f * increment,
                                     startGain + 5.0f * increment,
                                     startGain + 4.0f * increment,
                                     startGain + 3.0f * increment,
                                     startGain + 2.0f * increment,
                                     startGain + increment,
                                     startGain);
        const __m256 vInc8 = _mm256_set1_ps(8.0f * increment);

        int i = 0;
        const int vEnd = numSamples / 8 * 8;
        for (; i < vEnd; i += 8)
        {
            __m256 vData = _mm256_loadu_ps(data + i);
            _mm256_storeu_ps(data + i, _mm256_mul_ps(vData, vGain));
            vGain = _mm256_add_ps(vGain, vInc8);
        }

        float gain = startGain + static_cast<float>(i) * increment;
        for (; i < numSamples; ++i) { data[i] *= gain; gain += increment; }
    }

inline void scaleBlockFallback(double* data, int numSamples, double gain) noexcept
    {
        int i = 0;
        const int vEnd = numSamples / 4 * 4;
        const __m256d vGain = _mm256_set1_pd(gain);
        for (; i < vEnd; i += 4)
        {
            __m256d vData = _mm256_loadu_pd(data + i);
            _mm256_storeu_pd(data + i, _mm256_mul_pd(vData, vGain));
        }
        for (; i < numSamples; ++i)
            data[i] *= gain;
    }

static inline double fastTanh(double x) noexcept
{
    using namespace TanhApprox;

    if (x >= CLIP_THRESHOLD) return 1.0;
    if (x <= -CLIP_THRESHOLD) return -1.0;
    const double x2 = x * x;

    const double num = x * (NUM_A + x2 * (NUM_B + x2 * NUM_C));
    const double den = DEN_A + x2 * (DEN_B + x2 * (DEN_C + x2));
    return num / den;
}

static inline double musicalSoftClipScalar(double x, double threshold, double knee, double asymmetry) noexcept
{
    const double abs_x = absNoLibm(x);
    const double clip_start = threshold - knee;

    // 安全対策: kneeが極端に小さい場合のゼロ除算防止
    if (knee < 1.0e-9) return (x > threshold) ? threshold : ((x < -threshold) ? -threshold : x);

    // 閾値以下はリニア
    if (abs_x < clip_start)
        return x;

    const double sign = (x > 0.0) ? 1.0 : -1.0;

    // ソフトニー領域 (ブレンド率計算)
    double knee_shape = 1.0;
    if (abs_x < threshold + knee)
    {
        // 3次多項式でスムーズなニー
        const double t = (abs_x - clip_start) / (2.0 * knee);
        knee_shape = t * t * (3.0 - 2.0 * t); // Smoothstep
    }

    const double linear = abs_x;
    // tanhによるソフトクリッピングカーブ
    const double clipped = threshold + knee * fastTanh((abs_x - threshold) / knee);

    const double asymmetric_gain = 1.0 - asymmetry * (1.0 - sign) * 0.5 * knee_shape;
    return sign * (linear * (1.0 - knee_shape) + clipped * knee_shape) * asymmetric_gain;
}

static void softClipBlockAVX2(double* __restrict data, int numSamples,
                               double threshold, double knee, double asymmetry,
                               double& prevSampleInOut) noexcept
{
    const double clip_start = threshold - knee;
    jassert(knee > 1.0e-9);

    const __m256d vClipStart   = _mm256_set1_pd(clip_start);
    const __m256d vThreshold   = _mm256_set1_pd(threshold);
    const __m256d vKnee        = _mm256_set1_pd(knee);
    const __m256d vAsym        = _mm256_set1_pd(asymmetry);

    // Pre-calculate reciprocals for division optimization
    const __m256d vRecipKnee   = _mm256_set1_pd(1.0 / knee);
    const __m256d vRecipKnee2  = _mm256_set1_pd(1.0 / (2.0 * knee));

    // Constants for fastTanh and smoothstep
    const __m256d vOne         = _mm256_set1_pd(1.0);
    const __m256d vMinusOne    = _mm256_set1_pd(-1.0);
    const __m256d vTwo         = _mm256_set1_pd(2.0);
    const __m256d vThree       = _mm256_set1_pd(3.0);
    const __m256d vNegThree    = _mm256_set1_pd(-3.0);
    const __m256d vHalf        = _mm256_set1_pd(0.5);

    const __m256d vNumA        = _mm256_set1_pd(TanhApprox::NUM_A);
    const __m256d vNumB        = _mm256_set1_pd(TanhApprox::NUM_B);
    const __m256d vNumC        = _mm256_set1_pd(TanhApprox::NUM_C);
    const __m256d vDenB        = _mm256_set1_pd(TanhApprox::DEN_B);
    const __m256d vDenC        = _mm256_set1_pd(TanhApprox::DEN_C);
    const __m256d vZero        = _mm256_setzero_pd();
    const __m256d vSignMask    = _mm256_set1_pd(-0.0);

    // ── インターサンプルピーク用ブロック間状態 ─────────────────────────────
    // 前ブロック末尾のクリップ済み出力を保持。ブロック先頭の中点チェックに使用。
    double prevScalar = prevSampleInOut;

    int i = 0;
    const int vEnd = numSamples / 4 * 4;
    for (; i < vEnd; i += 4)
    {
            __m256d x    = _mm256_loadu_pd(data + i);

        // ── インターサンプルピーク近似 (線形中点プリゲイン) ──────────────────
        // 連続する2サンプルの中点 mid = (prev + x[n]) / 2 を推定し、
        // DACの帯域制限再構成で発生しうる inter-sample ピークを事前に抑制する。
        // prevVec = [prevScalar, x[0], x[1], x[2]] を構築して vectorize する。
        // 注: intra-vector の prev には RAW 入力を使用（AVX2 逐次依存を回避）。
        {
            const __m128d xLow       = _mm256_castpd256_pd128(x);                         // [x0, x1]
            const __m128d xHigh      = _mm256_extractf128_pd(x, 1);                       // [x2, x3]
            const __m128d prevLow128 = _mm_unpacklo_pd(_mm_set_sd(prevScalar), xLow);     // [prevScalar, x0]
            const __m128d prevHigh128= _mm_shuffle_pd(xLow, xHigh, 0x1);                  // [x1, x2]
            const __m256d prevVec    = _mm256_set_m128d(prevHigh128, prevLow128);          // [prevScalar, x0, x1, x2]

            const __m256d midVec     = _mm256_mul_pd(_mm256_add_pd(prevVec, x), vHalf);   // (prev+x)/2
            const __m256d absMidVec  = _mm256_andnot_pd(vSignMask, midVec);               // |mid|

            // mid が threshold を超える箇所にのみプリゲインを適用
            const __m256d vTiny      = _mm256_set1_pd(1e-15);
            const __m256d needMidClip= _mm256_cmp_pd(absMidVec, vThreshold, _CMP_GT_OQ);
            const __m256d safeAbsMid = _mm256_max_pd(absMidVec, vTiny);
            const __m256d midGainRaw = _mm256_div_pd(vThreshold, safeAbsMid);             // threshold / |mid|
            const __m256d midGain    = _mm256_blendv_pd(vOne, midGainRaw, needMidClip);   // gain=1 or scaled
            x = _mm256_mul_pd(x, midGain);  // プリゲイン適用後の x をソフトクリッパーに渡す
        }
        // ─────────────────────────────────────────────────────────────────────

        __m256d absX = _mm256_andnot_pd(vSignMask, x);

        // Check if any sample in the vector needs clipping
        __m256d needClip = _mm256_cmp_pd(absX, vClipStart, _CMP_GT_OQ);
        // Jitter対策: 条件分岐を削除し、常に一定の計算負荷をかける。
        // データ依存の負荷変動（静かなパートは軽く、大音量で重くなる）を防ぐ。

        // --- sign ---
        __m256d maskSignPos = _mm256_cmp_pd(x, vZero, _CMP_GT_OQ);
        __m256d sign = _mm256_blendv_pd(vMinusOne, vOne, maskSignPos);

        // --- knee_shape (smoothstep) ---
        __m256d t = _mm256_mul_pd(_mm256_sub_pd(absX, vClipStart), vRecipKnee2);
        t = _mm256_min_pd(_mm256_max_pd(t, vZero), vOne); // clamp t to [0,1]
        __m256d t2 = _mm256_mul_pd(t, t);
        __m256d ks = _mm256_mul_pd(t2, _mm256_fnmadd_pd(vTwo, t, vThree)); // t2 * (3 - 2*t)

        // --- fastTanh ---
        __m256d arg = _mm256_mul_pd(_mm256_sub_pd(absX, vThreshold), vRecipKnee);
        __m256d satHi    = _mm256_cmp_pd(arg, vThree,    _CMP_GE_OQ);
        __m256d satLo    = _mm256_cmp_pd(arg, vNegThree, _CMP_LE_OQ);
        __m256d arg2     = _mm256_mul_pd(arg, arg);

        __m256d num      = _mm256_mul_pd(arg,
                            _mm256_fmadd_pd(arg2,
                                _mm256_fmadd_pd(arg2, vNumC, vNumB),
                            vNumA));
        __m256d den      = _mm256_fmadd_pd(arg2,
                            _mm256_fmadd_pd(arg2,
                                _mm256_fmadd_pd(arg2, vDenC, vDenB),
                            vDenC),
                           vNumA);
        __m256d tanhVal  = _mm256_div_pd(num, den);
        tanhVal = _mm256_blendv_pd(tanhVal, vOne,      satHi);
        tanhVal = _mm256_blendv_pd(tanhVal, vMinusOne, satLo);

        // clipped = threshold + knee * tanh(...)
        __m256d clipped = _mm256_fmadd_pd(vKnee, tanhVal, vThreshold);

        // --- blend linear / clipped ---
        __m256d linear  = absX;
        __m256d mixed   = _mm256_fmadd_pd(_mm256_sub_pd(clipped, linear), ks, linear);

        // --- asymmetry ---
        __m256d factor = _mm256_mul_pd(vAsym, _mm256_sub_pd(vOne, sign));
        factor = _mm256_mul_pd(factor, vHalf);
        factor = _mm256_mul_pd(factor, ks);
        __m256d asymmetric_gain = _mm256_sub_pd(vOne, factor);

        __m256d result = _mm256_mul_pd(sign, _mm256_mul_pd(mixed, asymmetric_gain));

        // Blend with original x for samples that didn't need clipping
        result = _mm256_blendv_pd(x, result, needClip);
            _mm256_storeu_pd(data + i, result);

        // 次イテレーションに向けてクリップ済み末尾出力を保持
        prevScalar = data[i + 3];
    }

    // Scalar remainder（インターサンプルピークチェック込み）
    for (; i < numSamples; ++i)
    {
        // 線形中点で inter-sample ピークを推定し、超過時はプリゲインで抑制
        const double mid    = (prevScalar + data[i]) * 0.5;
        const double absMid = absNoLibm(mid);
        double x = data[i];
        if (absMid > threshold)
            x *= threshold / absMid;

        if (absNoLibm(x) > clip_start)
            x = musicalSoftClipScalar(x, threshold, knee, asymmetry);

        data[i] = x;
        prevScalar = x;  // クリップ済み出力を次サンプルの prev として保持
    }

    // ブロック間状態を更新
    prevSampleInOut = prevScalar;
}

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_LATENCY_QUERY)

double AudioEngine::getProcessingSampleRate() const
{
    const double sr = currentSampleRate.load();
    if (sr <= 0.0) return 0.0;

    int factor = manualOversamplingFactor.load();
    int actualFactor = 1;

    if (factor > 0)
    {
        if (factor == 1 || factor == 2 || factor == 4 || factor == 8)
            actualFactor = factor;
    }
    else
    {
        // Auto
        if (sr <= 96000.0)       actualFactor = 8;
        else if (sr <= 192000.0) actualFactor = 4;
        else if (sr <= 384000.0) actualFactor = 2;
        else                     actualFactor = 1;
    }

    // 制限: サンプルレートに応じた最大倍率を適用
    int maxFactor = 1;
    if (sr <= 96000.0)       maxFactor = 8;
    else if (sr <= 192000.0) maxFactor = 4;
    else if (sr <= 384000.0) maxFactor = 2;

    actualFactor = std::min(actualFactor, maxFactor);

    return sr * static_cast<double>(actualFactor);
}

int AudioEngine::getCurrentLatencySamples() const
{
    return getCurrentLatencyBreakdown().totalLatencyBaseRateSamples;
}

int AudioEngine::getTotalLatencySamples() const
{
    return getCurrentLatencyBreakdown().totalLatencyBaseRateSamples;
}

AudioEngine::LatencyBreakdown AudioEngine::getCurrentLatencyBreakdown() const
{
    LatencyBreakdown breakdown;

    auto* dsp = currentDSP.load(std::memory_order_acquire);
    if (dsp == nullptr)
        return breakdown;

    const int osFactor = static_cast<int>(dsp->oversamplingFactor);
    const int safeOsFactor = std::max(1, osFactor);

    const auto toBaseRateSamples = [safeOsFactor](int processingRateSamples) -> int
    {
        return juce::jmax(0,
            static_cast<int>(std::lround(static_cast<double>(processingRateSamples)
                                         / static_cast<double>(safeOsFactor))));
    };

    breakdown.oversamplingLatencyBaseRateSamples = juce::jmax(0,
        static_cast<int>(std::lround(estimateOversamplingLatencySamples(
            safeOsFactor,
            oversamplingType.load(std::memory_order_acquire),
            currentSampleRate.load(std::memory_order_acquire)))));

    if (!convBypassActive.load(std::memory_order_relaxed))
    {
        auto convBreakdown = dsp->convolver.getLatencyBreakdown();

        // DSP側が再構築中などで 0 を返す瞬間は、UI側コンボルバーのスナップショットを使う。
        if (convBreakdown.algorithmLatencySamples == 0 &&
            convBreakdown.irPeakLatencySamples == 0 &&
            convBreakdown.totalLatencySamples == 0)
        {
            convBreakdown = uiConvolverProcessor.getLatencyBreakdown();
        }

        breakdown.convolverAlgorithmLatencyBaseRateSamples = toBaseRateSamples(convBreakdown.algorithmLatencySamples);
        breakdown.convolverIRPeakLatencyBaseRateSamples = toBaseRateSamples(convBreakdown.irPeakLatencySamples);
        breakdown.convolverTotalLatencyBaseRateSamples = toBaseRateSamples(convBreakdown.totalLatencySamples);
    }

    breakdown.totalLatencyBaseRateSamples = juce::jmax(0,
        breakdown.oversamplingLatencyBaseRateSamples
      + breakdown.convolverTotalLatencyBaseRateSamples);

    return breakdown;
}

double AudioEngine::getCurrentLatencyMs() const
{
    const double sr = currentSampleRate.load(std::memory_order_acquire);
    if (sr <= 0.0)
        return 0.0;

    const int totalSamples = getCurrentLatencyBreakdown().totalLatencyBaseRateSamples;
    const double totalMs = (static_cast<double>(totalSamples) * 1000.0) / sr;
    return static_cast<double>(juce::roundToInt(totalMs));
}

#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_DSP_LATENCY)

double AudioEngine::estimateOversamplingLatencySamples(int oversamplingFactor,
                                                       OversamplingType oversamplingType,
                                                       double baseSampleRate) noexcept
{
    return estimateOversamplingLatencySamplesImpl(oversamplingFactor, oversamplingType, baseSampleRate);
}

#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_DSP_PREPARE)

AudioEngine::DSPCore::DSPCore() = default;

void AudioEngine::DSPCore::prepare(double newSampleRate, int samplesPerBlock, int bitDepth, int manualOversamplingFactor, OversamplingType oversamplingType, NoiseShaperType selectedNoiseShaperType, AudioEngine* owner)
{
    this->sampleRate = newSampleRate;
    this->noiseShaperType = selectedNoiseShaperType;
    this->ownerEngine = owner;
    convolver.setRcuProvider(ownerEngine);

    int targetFactor = 1;
    if (manualOversamplingFactor > 0)
    {
        targetFactor = manualOversamplingFactor;
    }
    else
    {
        // 自動設定 (デフォルト)
        if (newSampleRate >= 705600)
            targetFactor = 1;
        else if (newSampleRate >= 352800)
            targetFactor =  2;
        else if (newSampleRate >= 176400)
            targetFactor =  4;
        else if (newSampleRate >= 88200)
            targetFactor = 8;
         else
             targetFactor = 8;
    }

    // 制限: サンプルレートに応じた最大倍率を適用
    int maxFactor = 1;
    if (newSampleRate <= 96000.0)       maxFactor = 8;
    else if (newSampleRate <= 192000.0) maxFactor = 4;
    else if (newSampleRate <= 384000.0) maxFactor = 2;

    targetFactor = std::min(targetFactor, maxFactor);

    size_t factorLog2 = 0;
    if (targetFactor >= 8)      factorLog2 = 3;
    else if (targetFactor >= 4) factorLog2 = 2;
    else if (targetFactor >= 2) factorLog2 = 1;
    else                        factorLog2 = 0;

    oversamplingFactor = (size_t)1 << factorLog2;

    // ==================================================================
    // 【Issue 3 完全修正】内部最大バッファサイズの計算（推奨A）
    // 固定で SAFE_MAX_BLOCK_SIZE × 8 を確保
    // 理由:
    //   ・OS=8x時のupBlockサイズを完全にカバー
    //   ・RCU再構築（IRロード・プリセット切替・OS変更）ごとにresizeしない
    //   ・MKLAllocator + 64byteアライメントの最適化が最大限活きる
    //   ・将来16x OS対応もこの定数1箇所変更だけで済む
    // ==================================================================
    constexpr int MAX_OS_FACTOR = 8;
    // [FIX] Ensure we cover the requested block size even if it exceeds SAFE_MAX_BLOCK_SIZE
    const int inputMaxBlock     = std::max(SAFE_MAX_BLOCK_SIZE, samplesPerBlock);
    const int internalMaxBlock  = inputMaxBlock * MAX_OS_FACTOR;

    maxSamplesPerBlock   = inputMaxBlock;
    maxInternalBlockSize = internalMaxBlock;

// === 【パッチ3】raw aligned_malloc確保（message threadのみ・64byte保証）===
    const int newRequired = internalMaxBlock;
    if (newRequired > alignedCapacity || !alignedL || !alignedR)
    {
        // Exception-safe allocation using local ScopedAlignedPtr
        auto newL = convo::ScopedAlignedPtr<double>(static_cast<double*>(convo::aligned_malloc(
            static_cast<size_t>(newRequired) * sizeof(double), 64)));
        auto newR = convo::ScopedAlignedPtr<double>(static_cast<double*>(convo::aligned_malloc(
            static_cast<size_t>(newRequired) * sizeof(double), 64)));

        // 明示的ゼロクリア（Denormal/NaN防止）
        juce::FloatVectorOperations::clear(newL.get(), newRequired);
        juce::FloatVectorOperations::clear(newR.get(), newRequired);

        // Commit (noexcept move)
        alignedL = std::move(newL);
        alignedR = std::move(newR);
        alignedCapacity = newRequired;
    }

    if (newRequired > dryBypassCapacityDouble || !dryBypassBufferDoubleL || !dryBypassBufferDoubleR)
    {
        auto newDryL = convo::ScopedAlignedPtr<double>(static_cast<double*>(convo::aligned_malloc(
            static_cast<size_t>(newRequired) * sizeof(double), 64)));
        auto newDryR = convo::ScopedAlignedPtr<double>(static_cast<double*>(convo::aligned_malloc(
            static_cast<size_t>(newRequired) * sizeof(double), 64)));
        juce::FloatVectorOperations::clear(newDryL.get(), newRequired);
        juce::FloatVectorOperations::clear(newDryR.get(), newRequired);
        dryBypassBufferDoubleL = std::move(newDryL);
        dryBypassBufferDoubleR = std::move(newDryR);
        dryBypassCapacityDouble = newRequired;
    }

    bypassFadeGainDouble.reset(newSampleRate, 0.005);
    bypassFadeGainDouble.setCurrentAndTargetValue(1.0);
    bypassedDouble = false;

    const auto osPreset = (oversamplingType == OversamplingType::LinearPhase)
                        ? CustomInputOversampler::Preset::LinearPhase
                        : CustomInputOversampler::Preset::IIRLike;
    oversampling.prepare(inputMaxBlock, static_cast<int>(oversamplingFactor), osPreset);

    const double processingRate = newSampleRate * static_cast<double>(oversamplingFactor);
    const int processingBlockSize = samplesPerBlock * static_cast<int>(oversamplingFactor);

    // プロセッサの準備
    // Convolverには実際のブロックサイズを渡す (パーティションサイズ決定やLoaderThreadで使用)
    convolver.prepareToPlay(processingRate, processingBlockSize);

    // EQも内部最大サイズで準備（より安全）
    eq.prepareToPlay(processingRate, internalMaxBlock);

    // 出力段(processOutput)で実行されるため、オーバーサンプリング前のレートとサイズを使用する
    // 【最適化】UltraHighRateDCBlocker の init() は sampleRate + cutoffHz を受け取る
    dcBlockerL.init(newSampleRate, 3.0);
    dcBlockerR.init(newSampleRate, 3.0);

    // 入力段用DCBlockerの準備
    inputDCBlockerL.init(newSampleRate, 3.0);
    inputDCBlockerR.init(newSampleRate, 3.0);

    // オーバーサンプリング後のDC除去用 (1Hzカットオフ)
    osDCBlockerL.init(processingRate, 1.0);
    osDCBlockerR.init(processingRate, 1.0);

    // ノイズシェーパーの準備 (出力段で行うため元のサンプルレート)
    if (selectedNoiseShaperType == NoiseShaperType::Psychoacoustic)
        dither.prepare(newSampleRate, bitDepth);
    else if (selectedNoiseShaperType == NoiseShaperType::Fixed4Tap)
    {
        fixedNoiseShaper.setCoefficients(kFixedNoiseShaperTunedCoeffs);
        fixedNoiseShaper.prepare(newSampleRate, bitDepth);
    }
    else if (selectedNoiseShaperType == NoiseShaperType::Fixed15Tap)
    {
        fixed15TapNoiseShaper.setCoefficients(kFixed15TapNoiseShaperTunedCoeffs);
        fixed15TapNoiseShaper.prepare(newSampleRate, bitDepth);
    }
    else
    {
        adaptiveNoiseShaper.prepare(bitDepth);
        adaptiveNoiseShaper.setCoefficients(kDefaultAdaptiveNoiseShaperCoeffs.data(), kAdaptiveNoiseShaperOrder);
        activeAdaptiveCoeffGeneration = 0;
        activeAdaptiveCoeffBankIndex = -1;
    }
    this->ditherBitDepth = bitDepth; // DSPCoreのメンバーに保存

    // 出力周波数フィルターの係数を事前計算 (processingRate: OS後のレート)
    // filter.txt: ハイカット/ローカット(①) / ローパス/ハイパス(②) の全モード分を一括生成
    outputFilter.prepare(processingRate);

    // 【Issue 5】Fade-inカウンタをリセット
    fadeInSamplesLeft.store(0, std::memory_order_relaxed);

    // 初期状態は固定レイテンシなし
    setFixedLatencySamples(0);
}

void AudioEngine::DSPCore::setFixedLatencySamples(int samples)
{
    const int clamped = std::max(0, samples);
    fixedLatencySamples = clamped;
    fixedLatencyWritePos = 0;

    const int requiredSize = clamped + std::max(1, maxInternalBlockSize) + 2;
    if (requiredSize > fixedLatencyBufferSize || !fixedLatencyBufferL || !fixedLatencyBufferR)
    {
        auto newDelayL = convo::ScopedAlignedPtr<double>(static_cast<double*>(convo::aligned_malloc(
            static_cast<size_t>(requiredSize) * sizeof(double), 64)));
        auto newDelayR = convo::ScopedAlignedPtr<double>(static_cast<double*>(convo::aligned_malloc(
            static_cast<size_t>(requiredSize) * sizeof(double), 64)));

        juce::FloatVectorOperations::clear(newDelayL.get(), requiredSize);
        juce::FloatVectorOperations::clear(newDelayR.get(), requiredSize);

        fixedLatencyBufferL = std::move(newDelayL);
        fixedLatencyBufferR = std::move(newDelayR);
        fixedLatencyBufferSize = requiredSize;
    }
    else if (fixedLatencyBufferSize > 0)
    {
        juce::FloatVectorOperations::clear(fixedLatencyBufferL.get(), fixedLatencyBufferSize);
        juce::FloatVectorOperations::clear(fixedLatencyBufferR.get(), fixedLatencyBufferSize);
    }
}

void AudioEngine::DSPCore::reset()
{
    convolver.reset();
    eq.reset();
    dcBlockerL.reset();
    dcBlockerR.reset();
    inputDCBlockerL.reset();
    inputDCBlockerR.reset();
    osDCBlockerL.reset();
    osDCBlockerR.reset();
    dither.reset();
    fixedNoiseShaper.reset();
    adaptiveNoiseShaper.reset();
    oversampling.reset();
    outputFilter.reset();
    activeAdaptiveCoeffGeneration = 0;
    activeAdaptiveCoeffBankIndex = -1;

    // 【パッチ3】rawバッファクリア（alignedCapacity使用）
    if (alignedL && alignedCapacity > 0)
        juce::FloatVectorOperations::clear(alignedL.get(), alignedCapacity);
    if (alignedR && alignedCapacity > 0)
        juce::FloatVectorOperations::clear(alignedR.get(), alignedCapacity);
    if (dryBypassBufferDoubleL && dryBypassCapacityDouble > 0)
        juce::FloatVectorOperations::clear(dryBypassBufferDoubleL.get(), dryBypassCapacityDouble);
    if (dryBypassBufferDoubleR && dryBypassCapacityDouble > 0)
        juce::FloatVectorOperations::clear(dryBypassBufferDoubleR.get(), dryBypassCapacityDouble);

    bypassFadeGainDouble.setCurrentAndTargetValue(1.0);
    bypassedDouble = false;

    fixedLatencyWritePos = 0;
    if (fixedLatencyBufferL && fixedLatencyBufferSize > 0)
        juce::FloatVectorOperations::clear(fixedLatencyBufferL.get(), fixedLatencyBufferSize);
    if (fixedLatencyBufferR && fixedLatencyBufferSize > 0)
        juce::FloatVectorOperations::clear(fixedLatencyBufferR.get(), fixedLatencyBufferSize);

    // インターサンプルピーク用ブロック間状態をリセット
    softClipPrevSample[0] = 0.0;
    softClipPrevSample[1] = 0.0;
}

#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_PREPARE_TO_PLAY)

void AudioEngine::prepareToPlay (int samplesPerBlockExpected, double sampleRate)
  {
      diagLog("[DIAG] prepareToPlay: enter spb=" + juce::String(samplesPerBlockExpected) + " sr=" + juce::String(sampleRate, 2));

    gShuttingDown.store(false, std::memory_order_release);
    shutdownInProgress.store(false, std::memory_order_release);

    // releaseResources() で停止済みの場合に備えて、必要なら rebuild thread を再起動する。
    if (!rebuildThread.joinable())
    {
        {
            std::lock_guard<std::mutex> lock(rebuildMutex);
            rebuildThreadShouldExit.store(false, std::memory_order_release);
            hasPendingTask = false;
            pendingTask = RebuildTask{};
        }
        rebuildThread = std::thread(&AudioEngine::rebuildThreadLoop, this);
    }

    // --- AudioEngine::prepareToPlay ---
    // ※本関数は「AudioThread停止中のみ呼ぶ」ことがJUCE AudioSource仕様上の前提です。
    //   これを破るとバッファfree競合・data raceの危険があります。

    // パラメータ検証 (Parameter Validation)
    double safeSampleRate = sampleRate;
    if (safeSampleRate <= 0.0 || safeSampleRate > SAFE_MAX_SAMPLE_RATE || !std::isfinite(safeSampleRate)) {
        jassertfalse;
        safeSampleRate = 48000.0;
    }
    if (samplesPerBlockExpected <= 0) {
        jassertfalse;
        samplesPerBlockExpected = 512;
    }
    const int bufferSize = samplesPerBlockExpected;

    // サンプルレート・ブロックサイズ変更検知
    const bool rateChanged = (std::abs(currentSampleRate.load() - safeSampleRate) > 1e-6);
    const bool blockSizeChanged = (maxSamplesPerBlock.load() != bufferSize);

    maxSamplesPerBlock.store(bufferSize);
    currentSampleRate.store(safeSampleRate);
    dspCrossfadeGain.reset(safeSampleRate, 0.03);
    dspCrossfadeGain.setCurrentAndTargetValue(1.0);
    dspCrossfadePending.store(false, std::memory_order_release);
    selectAdaptiveCoeffBankForCurrentSettings();

    dspCrossfadeFloatBuffer.setSize(2, std::max(SAFE_MAX_BLOCK_SIZE, bufferSize), false, false, true);
    dspCrossfadeDoubleBuffer.setSize(2, std::max(SAFE_MAX_BLOCK_SIZE, bufferSize), false, false, true);

    analyzerFifo.prepare(2, FIFO_SIZE);
    inputLevelLinear.store(0.0f);
    outputLevelLinear.store(0.0f);

    eqBypassActive.store(eqBypassRequested.load(std::memory_order_relaxed), std::memory_order_relaxed);
    convBypassActive.store(convBypassRequested.load(std::memory_order_relaxed), std::memory_order_relaxed);

    // --- レイテンシ整合バッファの再確保 ---
    // ※本関数はAudioThread停止中のみ呼ぶこと！
    if (latencyBufOldL) { _aligned_free(latencyBufOldL); latencyBufOldL = nullptr; }
    if (latencyBufOldR) { _aligned_free(latencyBufOldR); latencyBufOldR = nullptr; }
    if (latencyBufNewL) { _aligned_free(latencyBufNewL); latencyBufNewL = nullptr; }
    if (latencyBufNewR) { _aligned_free(latencyBufNewR); latencyBufNewR = nullptr; }

    // 最大遅延（2秒上限・kMaxLatencySamples制限）
    // +blockSizeはwrap安全余裕（リングバッファwrap時の読み出し安全域）
    const int maxDelay = std::min(kMaxLatencySamples, static_cast<int>(safeSampleRate * 2.0));
    latencyBufSize = maxDelay + bufferSize + 2;

    latencyBufOldL = (double*)_aligned_malloc(sizeof(double) * latencyBufSize, 64);
    latencyBufOldR = (double*)_aligned_malloc(sizeof(double) * latencyBufSize, 64);
    latencyBufNewL = (double*)_aligned_malloc(sizeof(double) * latencyBufSize, 64);
    latencyBufNewR = (double*)_aligned_malloc(sizeof(double) * latencyBufSize, 64);

    // malloc失敗時は安全フェイル
    if (!latencyBufOldL || !latencyBufOldR || !latencyBufNewL || !latencyBufNewR) {
        latencyBufSize = 0;
        return;
    }

    std::memset(latencyBufOldL, 0, sizeof(double) * latencyBufSize);
    std::memset(latencyBufOldR, 0, sizeof(double) * latencyBufSize);
    std::memset(latencyBufNewL, 0, sizeof(double) * latencyBufSize);
    std::memset(latencyBufNewR, 0, sizeof(double) * latencyBufSize);

    latencyWritePos = 0;
    latencyDelayOld.store(0, std::memory_order_release);
    latencyDelayNew.store(0, std::memory_order_release);
    latencyResetPending.store(false, std::memory_order_release);

    latencyDelayOld_RT = 0;
    latencyDelayNew_RT = 0;

    // 初回IRロード前でも currentDSP を常に有効にし、DSP->DSP クロスフェードへ統一する。
    if (currentDSP.load(std::memory_order_acquire) == nullptr && activeDSP == nullptr)
    {
        DSPCore* placeholderDSP = new DSPCore();
        placeholderDSP->convolver.setVisualizationEnabled(false);
        placeholderDSP->prepare(safeSampleRate,
                                bufferSize,
                                ditherBitDepth.load(std::memory_order_relaxed),
                                manualOversamplingFactor.load(std::memory_order_relaxed),
                                oversamplingType.load(std::memory_order_relaxed),
                                noiseShaperType.load(std::memory_order_relaxed),
                                this);
        placeholderDSP->convolver.setBypass(true);

        // Use same formula as actual convolver L0 partition size (MKLNonUniformConvolver.cpp:534)
        // to ensure placeholder bypass latency matches NUC algorithm latency during DSP transition
        int predictedLatency = juce::nextPowerOfTwo(std::max(bufferSize, 64));
        predictedLatency = juce::jlimit(0, latencyBufSize - 1, predictedLatency);
        placeholderDSP->setFixedLatencySamples(predictedLatency);

        activeDSP = placeholderDSP;
        currentDSP.store(placeholderDSP, std::memory_order_release);
    }

    // --- DSP再ビルド判定・同期 ---
    uiConvolverProcessor.prepareToPlay(safeSampleRate, bufferSize);
    if (rateChanged)
        uiConvolverProcessor.invalidatePendingLoads();
    if (rateChanged || blockSizeChanged || currentDSP.load(std::memory_order_acquire) == nullptr) {
        if (juce::MessageManager::getInstance()->isThisTheMessageThread()) {
            requestRebuild(safeSampleRate, bufferSize);
        } else {
            requestRebuild(convo::RebuildKind::Structural);
        }
    }
        diagLog("[DIAG] prepareToPlay: exit currentSR=" + juce::String(currentSampleRate.load(), 2) + " maxSPB=" + juce::String(maxSamplesPerBlock.load()));
    }

#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_RELEASE_RESOURCES)

void AudioEngine::releaseResources()
{
    diagLog("[DIAG] releaseResources: enter");
    shutdownInProgress.store(true, std::memory_order_release);
    firstIrDryCrossfadePending.store(false, std::memory_order_release);
    dspCrossfadeUseDryAsOld.store(false, std::memory_order_release);
        const bool finalShutdown = gShuttingDown.load(std::memory_order_acquire);
    if (finalShutdown)
    {
        lastIssuedConvolverStructuralHash_.store(0, std::memory_order_release);
        currentSampleRate.store(0.0);
    }

    inputLevelLinear.store(0.0f);
    outputLevelLinear.store(0.0f);

    if (noiseShaperLearner)
        noiseShaperLearner->stopLearning();

    resetLearningControlState();

    DSPCore* activeToRelease = nullptr;
    DSPCore* fadingToRelease = nullptr;
    DSPCore* queuedToRelease = nullptr;
    DSPCore* pendingNewToRelease = nullptr;
    DSPCore* pendingCurrentToRelease = nullptr;

    {
        std::lock_guard<std::mutex> lk(rebuildMutex);
        rebuildGeneration.fetch_add(1, std::memory_order_relaxed);
        currentDSP.store(nullptr, std::memory_order_release);

        activeToRelease = sanitizeRawPtr(activeDSP);
        activeDSP = nullptr;

        fadingToRelease = sanitizeRawPtr(fadingOutDSP.exchange(nullptr, std::memory_order_acq_rel));
        queuedToRelease = sanitizeRawPtr(queuedOldDSP.exchange(nullptr, std::memory_order_acq_rel));
        fadeQueued.store(false, std::memory_order_release);
        dspCrossfadeUseDryAsOld.store(false, std::memory_order_release);
        dspCrossfadeDryScaleGain.setCurrentAndTargetValue(1.0);
        queuedFadeTimeSec.store(0.03, std::memory_order_release);
        queuedNextFadeTimeSec.store(0.03, std::memory_order_release);

        if (hasPendingTask)
        {
            pendingNewToRelease = sanitizeRawPtr(pendingTask.newDSP);
            pendingTask.newDSP = nullptr;
            pendingCurrentToRelease = sanitizeRawPtr(pendingTask.currentDSP);
            pendingTask.currentDSP = nullptr;
            hasPendingTask = false;
        }

        dspCrossfadePending.store(false, std::memory_order_release);
        dspCrossfadeGain.setCurrentAndTargetValue(1.0);
    }

    diagLog("[DIAG] releaseResources: before stopRebuildThread");
    stopRebuildThread();
    diagLog("[DIAG] releaseResources: after stopRebuildThread");

    {
        std::queue<CommitStaging> abandonedCommits;
        std::lock_guard<std::mutex> lock(deferredCommitMutex);
        std::swap(abandonedCommits, deferredCommitQueue);

        while (!abandonedCommits.empty())
        {
            auto staging = abandonedCommits.front();
            abandonedCommits.pop();

            if (staging.newDSP)
                retireDSP(staging.newDSP);
            if (staging.oldDSP)
                retireDSP(staging.oldDSP);
        }
    }

    if (activeToRelease)
        retireDSP(activeToRelease);
    if (fadingToRelease)
        retireDSP(fadingToRelease);
    if (queuedToRelease)
        retireDSP(queuedToRelease);
    if (pendingNewToRelease)
        retireDSP(pendingNewToRelease);
    if (pendingCurrentToRelease)
        retireDSP(pendingCurrentToRelease);

    diagLog("[DIAG] releaseResources: before ui processor release");
    diagLog("[DIAG] releaseResources: before uiConvolverProcessor.releaseResources");
    uiConvolverProcessor.releaseResources();
    diagLog("[DIAG] releaseResources: after uiConvolverProcessor.releaseResources");

    diagLog("[DIAG] releaseResources: before uiEqEditor.releaseResources");
    uiEqEditor.releaseResources();
    diagLog("[DIAG] releaseResources: after uiEqEditor.releaseResources");

    diagLog("[DIAG] releaseResources: after ui processor release");

    diagLog("[DIAG] releaseResources: skip deferred reclaim (reconfigure phase)");

    diagLog("[DIAG] releaseResources: ABOUT_TO_EXIT_SCOPE");
}

#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_AUDIO_BLOCK)

void AudioEngine::getNextAudioBlock (const juce::AudioSourceChannelInfo& bufferToFill)
{
    const juce::ScopedNoDenormals noDenormals;
    m_audioBlockCounter.fetch_add(1, std::memory_order_release);

    // 入力検証 (Input Validation)
    if (bufferToFill.buffer == nullptr)
        return;

    const int numSamples = bufferToFill.numSamples;
    const int startSample = bufferToFill.startSample;
    auto* buffer = bufferToFill.buffer;

    // 事前サニティチェック: 絶対的な上限 (1<<20 ≒ 100万サンプル) で明らかな破損データを弾く。
    // DSPCore の maxSamplesPerBlock は prepareToPlay() でホスト指定値を反映して設定されるため、
    // ここで SAFE_MAX_BLOCK_SIZE (65536) を使うと、131072 等の正当なブロックを誤って拒否する。
    // 【Bug Fix】SAFE_MAX_BLOCK_SIZE による早期リジェクトを廃止し、dsp->maxSamplesPerBlock で
    //            正確なチェックを行う (下記 DSPCore 取得後)。
    constexpr int ABSOLUTE_MAX_BLOCK_SIZE = 1 << 20; // 破損データ検出用上限
    if (numSamples <= 0 || numSamples > ABSOLUTE_MAX_BLOCK_SIZE)
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    // startSampleの妥当性チェック
    if (startSample < 0 || startSample + numSamples > buffer->getNumSamples())
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    // Epoch tracking for lock-free Audio Thread safety
    convo::RCUReaderGuard rcuGuard(tls_rcuReader);

    DSPCore* dsp = currentDSP.load(std::memory_order_acquire);
    if (dsp == nullptr)
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    if (dsp != nullptr)
    {
        // DSPCore 固有の上限チェック
        // DSPCore::prepare() でホスト指定の samplesPerBlock を反映した maxSamplesPerBlock が設定される。
        // dsp は RCU で公開済みのため maxSamplesPerBlock は Audio Thread から安全に読み出せる。
        if (numSamples > dsp->maxSamplesPerBlock)
        {
            bufferToFill.clearActiveBufferRegion();
            return;
        }

        // 安全対策: サンプルレート不整合チェック
        // DSPのサンプルレートとエンジンの現在のサンプルレートが一致しない場合、
        // レート変更処理中とみなし、グリッチを防ぐために無音を出力する。
        const double engineSampleRate = currentSampleRate.load(std::memory_order_relaxed);
        if (absDiffNoLibm(dsp->sampleRate, engineSampleRate) > 1e-6)
        {
            // 不整合時はレベルメーターもリセットして誤表示を防ぐ
            inputLevelLinear.store(0.0f);
            outputLevelLinear.store(0.0f);
            bufferToFill.clearActiveBufferRegion();
            return;
        }

        // パラメータのロード
        // 【Parameter安全設計】
        // Audio ThreadではAtomic変数の読み取りのみを行い、ロックやメモリ確保を伴う処理は行わない。
        // 構造変更が必要な場合は、別途フラグやUIスレッド経由で再構築を行う。
        // ── Audio Thread 最適化: GlobalSnapshot を優先し、fallback で atomics を読む ──
        const convo::GlobalSnapshot* snap = m_coordinator.getCurrent();
        const bool eqBypassed               = (snap != nullptr) ? snap->eqBypass : eqBypassRequested.load(std::memory_order_acquire);
        const bool convBypassed             = (snap != nullptr) ? snap->convBypass : convBypassRequested.load(std::memory_order_acquire);
        const ProcessingOrder order         = (snap != nullptr) ? snap->processingOrder : currentProcessingOrder.load(std::memory_order_relaxed);
        const AnalyzerSource analyzerSource = currentAnalyzerSource.load(std::memory_order_relaxed);
        const bool analyzerEnabledNow       = analyzerEnabled.load(std::memory_order_relaxed);
        const bool softClip                 = (snap != nullptr) ? snap->softClipEnabled : softClipEnabled.load(std::memory_order_relaxed);
        const float satAmt                  = (snap != nullptr) ? snap->saturationAmount : saturationAmount.load(std::memory_order_relaxed);
        const double headroomGain           = (snap != nullptr) ? snap->inputHeadroomGain : inputHeadroomGain.load(std::memory_order_relaxed);
        const double makeupGain             = (snap != nullptr) ? snap->outputMakeupGain : outputMakeupGain.load(std::memory_order_relaxed);
        const double convInputTrimGain      = (snap != nullptr) ? snap->convInputTrimGain : convolverInputTrimGain.load(std::memory_order_relaxed);
        const convo::HCMode hcMode      = convHCFilterMode.load(std::memory_order_relaxed);
        const convo::LCMode lcMode      = convLCFilterMode.load(std::memory_order_relaxed);
        const convo::HCMode lpfMode     = eqLPFFilterMode.load(std::memory_order_relaxed);
        const int adaptiveCoeffBankIndex    = currentAdaptiveCoeffBankIndex.load(std::memory_order_acquire);
        const auto& adaptiveCoeffBank       = getAdaptiveCoeffBankForIndex(adaptiveCoeffBankIndex);
        const bool adaptiveCaptureEnabled   = noiseShaperLearner && noiseShaperLearner->isRunning();

        // RCU スナップショット取得：generation と active ポインタはダブルバッファリングにより一貫性が保証される
        const uint32_t genSnapshot = adaptiveCoeffBank.generation.load(std::memory_order_acquire);
        const CoeffSet* safeAdaptiveSet = AudioEngine::getActiveCoeffSet(adaptiveCoeffBank);
        // safeAdaptiveSet は、genSnapshot 時点で有効な係数セットを指す。
        // Writer が後で切り替えても、このポインタの指す内容は不変である。

        // 念のため nullptr チェック
        if (!safeAdaptiveSet) {
            // フォールバック：デフォルト係数を使用するなどの処理（必要に応じて）
            // ここでは単に nullptr のまま処理を続行（process() 側で対処）
        }
        const uint32_t adaptiveGenAfter = genSnapshot; // 互換性のため変数名を維持

        // UI表示用: 比較なしで直接ストア（ロード→比較→ストアより高速）
        eqBypassActive.store(eqBypassed, std::memory_order_relaxed);
        convBypassActive.store(convBypassed, std::memory_order_relaxed);

        DSPCore::ProcessingState procState {
            .eqBypassed               = eqBypassed,
            .convBypassed             = convBypassed,
            .order                    = order,
            .analyzerSource           = analyzerSource,
            .analyzerEnabled          = analyzerEnabledNow,
            .softClipEnabled          = softClip,
            .saturationAmount         = satAmt,
            .inputHeadroomGain        = headroomGain,
            .outputMakeupGain         = makeupGain,
            .convolverInputTrimGain   = convInputTrimGain,
            .convHCMode               = hcMode,
            .convLCMode               = lcMode,
            .eqLPFMode                = lpfMode,
            .adaptiveCoeffBankIndex   = adaptiveCoeffBankIndex,
            .adaptiveCoeffSet         = safeAdaptiveSet,
            .adaptiveCoeffGeneration  = adaptiveGenAfter,
            .adaptiveCaptureSampleRateHz = static_cast<int>(dsp->sampleRate + 0.5),
            .adaptiveCaptureBitDepth  = dsp->ditherBitDepth,
            .captureSessionId         = dsp->currentCaptureSessionId,
            .adaptiveCaptureQueue     = adaptiveCaptureEnabled ? &audioCaptureQueue : nullptr
        };

        if (m_coordinator.isFading())
            m_coordinator.advanceFade(numSamples);
        debugLastCoordinatorIsFading.store(m_coordinator.isFading() ? 1 : 0, std::memory_order_relaxed);

        float snapshotAlpha = 1.0f;
        const convo::GlobalSnapshot* snapshotFrom = nullptr;
        const convo::GlobalSnapshot* snapshotTo = nullptr;
        const bool updateFadeReturned = m_coordinator.updateFade(snapshotAlpha, snapshotFrom, snapshotTo);
        debugLastUpdateFadeReturned.store(updateFadeReturned ? 1 : 0, std::memory_order_relaxed);
        debugLastSnapshotFromNull.store(snapshotFrom == nullptr ? 1 : 0, std::memory_order_relaxed);
        debugLastSnapshotToNull.store(snapshotTo == nullptr ? 1 : 0, std::memory_order_relaxed);

        const bool snapshotFading = updateFadeReturned
            && snapshotTo != nullptr;

        if (snapshotFading)
        {
            const int fadeChannels = std::min(dspCrossfadeFloatBuffer.getNumChannels(), buffer->getNumChannels());
            for (int ch = 0; ch < fadeChannels; ++ch)
                dspCrossfadeFloatBuffer.clear(ch, 0, numSamples);

            juce::AudioSourceChannelInfo oldInfo(&dspCrossfadeFloatBuffer, 0, numSamples);
            processWithSnapshot(oldInfo, snapshotFrom, true);
            processWithSnapshot(bufferToFill, snapshotTo, false);

            const float gNew = snapshotAlpha;
            const float gOld = 1.0f - snapshotAlpha;
            const int outChannels = std::min(buffer->getNumChannels(), dspCrossfadeFloatBuffer.getNumChannels());
            float* dstL = (outChannels > 0) ? buffer->getWritePointer(0, startSample) : nullptr;
            float* dstR = (outChannels > 1) ? buffer->getWritePointer(1, startSample) : nullptr;
            const float* oldL = (outChannels > 0) ? dspCrossfadeFloatBuffer.getReadPointer(0, 0) : nullptr;
            const float* oldR = (outChannels > 1) ? dspCrossfadeFloatBuffer.getReadPointer(1, 0) : nullptr;

            for (int i = 0; i < numSamples; ++i)
            {
                if (dstL != nullptr)
                    dstL[i] = dstL[i] * gNew + oldL[i] * gOld;
                if (dstR != nullptr)
                    dstR[i] = dstR[i] * gNew + oldR[i] * gOld;
            }

            return;
        }

        DSPCore* fading = sanitizeRawPtr(fadingOutDSP.load(std::memory_order_acquire));
        bool useDryAsOld = dspCrossfadeUseDryAsOld.load(std::memory_order_acquire);
        int pendingFadeDelayBlocks = dspCrossfadeStartDelayBlocks.load(std::memory_order_acquire);
        if (fading != nullptr
            && !useDryAsOld
            && dspCrossfadePending.load(std::memory_order_acquire)
            && pendingFadeDelayBlocks > 0)
        {
            dspCrossfadeStartDelayBlocks.store(pendingFadeDelayBlocks - 1, std::memory_order_release);

            auto fadingState = procState;
            fadingState.analyzerEnabled = false;
            fadingState.adaptiveCaptureQueue = nullptr;

            std::atomic<float> fadingInputMeter { 0.0f };
            std::atomic<float> fadingOutputMeter { 0.0f };
            fading->process(bufferToFill, analyzerFifo, inputLevelLinear, outputLevelLinear, fadingState);
            return;
        }

        if ((fading != nullptr || firstIrDryCrossfadePending.load(std::memory_order_acquire))
            && dspCrossfadePending.exchange(false, std::memory_order_acq_rel))
        {
            const double fadeSec = std::max(0.001, queuedFadeTimeSec.load(std::memory_order_acquire));
            dspCrossfadeGain.reset(std::max(1.0, dsp->sampleRate), fadeSec);
            dspCrossfadeGain.setCurrentAndTargetValue(0.0);
            dspCrossfadeGain.setTargetValue(1.0);

            // レイテンシ整合値を Audio Thread スナップショットへ反映する。
            latencyDelayOld_RT = latencyDelayOld.load(std::memory_order_acquire);
            latencyDelayNew_RT = latencyDelayNew.load(std::memory_order_acquire);

            if (firstIrDryCrossfadePending.exchange(false, std::memory_order_acq_rel))
            {
                dspCrossfadeUseDryAsOld.store(true, std::memory_order_release);
                useDryAsOld = true;
            }
        }

        const bool canCrossfade = (fading != nullptr || useDryAsOld)
            && dspCrossfadeGain.isSmoothing()
            && dspCrossfadeFloatBuffer.getNumChannels() >= 2
            && dspCrossfadeFloatBuffer.getNumSamples() >= numSamples;

        if (canCrossfade)
        {
            juce::AudioSourceChannelInfo fadeInfo(&dspCrossfadeFloatBuffer, 0, numSamples);
            dspCrossfadeFloatBuffer.clear(0, 0, numSamples);
            dspCrossfadeFloatBuffer.clear(1, 0, numSamples);

            auto fadingState = procState;
            fadingState.analyzerEnabled = false;
            fadingState.adaptiveCaptureQueue = nullptr;

            std::atomic<float> fadingInputMeter { 0.0f };
            std::atomic<float> fadingOutputMeter { 0.0f };
            if (useDryAsOld)
            {
                const int outChannels = std::min(2, buffer->getNumChannels());
                if (outChannels > 0)
                    juce::FloatVectorOperations::copy(dspCrossfadeFloatBuffer.getWritePointer(0, 0), buffer->getReadPointer(0, startSample), numSamples);
                if (outChannels > 1)
                    juce::FloatVectorOperations::copy(dspCrossfadeFloatBuffer.getWritePointer(1, 0), buffer->getReadPointer(1, startSample), numSamples);
            }
            else
            {
                // EBR: lifetime managed by RCUReader
                fading->processToBuffer(bufferToFill, dspCrossfadeFloatBuffer, analyzerFifo,
                                       fadingInputMeter, fadingOutputMeter, fadingState);
            }
            dsp->process(bufferToFill, analyzerFifo, inputLevelLinear, outputLevelLinear, procState);

            const int outChannels = std::min(2, buffer->getNumChannels());
            float* dstL = (outChannels > 0) ? buffer->getWritePointer(0, startSample) : nullptr;
            float* dstR = (outChannels > 1) ? buffer->getWritePointer(1, startSample) : nullptr;
            const float* oldL = (outChannels > 0) ? dspCrossfadeFloatBuffer.getReadPointer(0, 0) : nullptr;
            const float* oldR = (outChannels > 1) ? dspCrossfadeFloatBuffer.getReadPointer(1, 0) : nullptr;

            const int bufferSize = latencyBufSize;
            int writePos = latencyWritePos;
            const int delayOld = latencyDelayOld_RT;
            const int delayNew = latencyDelayNew_RT;
            if (latencyResetPending.exchange(false, std::memory_order_acq_rel))
            {
                if (latencyBufOldL) std::memset(latencyBufOldL, 0, sizeof(double) * bufferSize);
                if (latencyBufOldR) std::memset(latencyBufOldR, 0, sizeof(double) * bufferSize);
                if (latencyBufNewL) std::memset(latencyBufNewL, 0, sizeof(double) * bufferSize);
                if (latencyBufNewR) std::memset(latencyBufNewR, 0, sizeof(double) * bufferSize);
                writePos = 0;
            }
            for (int i = 0; i < numSamples; ++i)
            {
                latencyBufOldL[writePos] = (oldL != nullptr) ? static_cast<double>(oldL[i]) : 0.0;
                latencyBufOldR[writePos] = (oldR != nullptr) ? static_cast<double>(oldR[i]) : 0.0;
                latencyBufNewL[writePos] = (dstL != nullptr) ? static_cast<double>(dstL[i]) : 0.0;
                latencyBufNewR[writePos] = (dstR != nullptr) ? static_cast<double>(dstR[i]) : 0.0;

                int readOld = writePos - delayOld;
                int readNew = writePos - delayNew;
                while (readOld < 0) readOld += bufferSize;
                while (readOld >= bufferSize) readOld -= bufferSize;
                while (readNew < 0) readNew += bufferSize;
                while (readNew >= bufferSize) readNew -= bufferSize;

                const double alignedOldL = latencyBufOldL[readOld];
                const double alignedOldR = latencyBufOldR[readOld];
                const double alignedNewL = latencyBufNewL[readNew];
                const double alignedNewR = latencyBufNewR[readNew];

                const double gNew = dspCrossfadeGain.getNextValue();
                const double dryScale = useDryAsOld ? dspCrossfadeDryScaleGain.getNextValue() : 1.0;
                const double gOld = 1.0 - gNew;
                const double dryScaledL = alignedOldL * dryScale;
                const double dryScaledR = alignedOldR * dryScale;
                if (dstL != nullptr)
                    dstL[i] = static_cast<float>(alignedNewL * gNew + dryScaledL * gOld);
                if (dstR != nullptr)
                    dstR[i] = static_cast<float>(alignedNewR * gNew + dryScaledR * gOld);

                writePos++;
                if (writePos >= bufferSize)
                    writePos = 0;
            }
            latencyWritePos = writePos;

            if (!useDryAsOld)
            {
                // EBR: fading lifetime managed by RCUReaderGuard
            }

            if (!dspCrossfadeGain.isSmoothing())
            {
                if (auto* done = sanitizeRawPtr(fadingOutDSP.exchange(nullptr, std::memory_order_acq_rel)))
                    retireDSP(done);
                dspCrossfadeGain.setCurrentAndTargetValue(1.0);
                dspCrossfadeDryScaleGain.setCurrentAndTargetValue(1.0);
                dspCrossfadeUseDryAsOld.store(false, std::memory_order_release);
            }
        }
        else
        {
            // 通常パス（クロスフェードなし）：RCU で dsp の生存が保証されるため addRef/release 不要
            dsp->process(bufferToFill, analyzerFifo, inputLevelLinear, outputLevelLinear, procState);

            if (fading != nullptr && !dspCrossfadeGain.isSmoothing())
            {
                if (auto* done = sanitizeRawPtr(fadingOutDSP.exchange(nullptr, std::memory_order_acq_rel)))
                    retireDSP(done);
            }
            if (!dspCrossfadeGain.isSmoothing())
                dspCrossfadeUseDryAsOld.store(false, std::memory_order_release);
        }
    }

}

#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_BLOCK_DOUBLE)

void AudioEngine::processBlockDouble (juce::AudioBuffer<double>& buffer)
{
    const juce::ScopedNoDenormals noDenormals;
    m_audioBlockCounter.fetch_add(1, std::memory_order_release);

    // ★ 追加: RCU ガードで現在の DSP を保護する
    convo::RCUReaderGuard rcuGuard(tls_rcuReader);
    const int numSamples = buffer.getNumSamples();
    // 事前サニティチェック (getNextAudioBlock と同様)
    constexpr int ABSOLUTE_MAX_BLOCK_SIZE = 1 << 20;
    if (numSamples <= 0 || numSamples > ABSOLUTE_MAX_BLOCK_SIZE)
    {
        buffer.clear();
        return;
    }

    DSPCore* dsp = currentDSP.load(std::memory_order_acquire);
    if (dsp == nullptr)
    {
        buffer.clear();
        return;
    }

    // AudioThread入口で、現在のDSPが持つ全てのNUCのガードをチェック（デバッグ時のみ）
        #ifdef NUC_DEBUG_GUARDS
        {
        dsp->convolver.debugCheckNucGuards();
        }
    #endif

    // --- ProcessingStateを現行設計で初期化 ---
    const bool eqBypassed = eqBypassActive.load(std::memory_order_relaxed);
    const bool convBypassed = convBypassActive.load(std::memory_order_relaxed);
    const ProcessingOrder order = currentProcessingOrder.load(std::memory_order_relaxed);
    const bool analyzerEnabledNow = analyzerEnabled.load(std::memory_order_relaxed);
    const AnalyzerSource analyzerSourceNow = currentAnalyzerSource.load(std::memory_order_relaxed);
    const convo::HCMode hcMode = convHCFilterMode.load(std::memory_order_relaxed);
    const convo::LCMode lcMode = convLCFilterMode.load(std::memory_order_relaxed);
    const convo::HCMode lpfMode = eqLPFFilterMode.load(std::memory_order_relaxed);
    const int adaptiveCoeffBankIndex = currentAdaptiveCoeffBankIndex.load(std::memory_order_acquire);
    const auto& adaptiveCoeffBank = getAdaptiveCoeffBankForIndex(adaptiveCoeffBankIndex);
    const uint32_t genSnapshot = adaptiveCoeffBank.generation.load(std::memory_order_acquire);
    const CoeffSet* safeAdaptiveSet = AudioEngine::getActiveCoeffSet(adaptiveCoeffBank);
    const uint32_t adaptiveGenAfter = genSnapshot;
    const bool adaptiveCaptureEnabled = noiseShaperLearner && noiseShaperLearner->isRunning();

    DSPCore::ProcessingState procState {
        eqBypassed,
        convBypassed,
        order,
        analyzerSourceNow,
        analyzerEnabledNow,
        softClipEnabled.load(std::memory_order_relaxed),
        saturationAmount.load(std::memory_order_relaxed),
        inputHeadroomGain.load(std::memory_order_relaxed),
        outputMakeupGain.load(std::memory_order_relaxed),
        convolverInputTrimGain.load(std::memory_order_relaxed),
        hcMode,
        lcMode,
        lpfMode,
        adaptiveCoeffBankIndex,
        safeAdaptiveSet,
        adaptiveGenAfter,
        static_cast<int>(dsp->sampleRate + 0.5),
        dsp->ditherBitDepth,
        dsp->currentCaptureSessionId,
        adaptiveCaptureEnabled ? &audioCaptureQueue : nullptr
    };

    // DSPCore 固有の上限チェック (getNextAudioBlock と同様)
    if (numSamples > dsp->maxSamplesPerBlock)
    {
        buffer.clear();
        return;
    }

    // EQ スナップショットフェードを進める (getNextAudioBlock と同等)
    if (m_coordinator.isFading())
        m_coordinator.advanceFade(numSamples);

        debugLastCoordinatorIsFading.store(m_coordinator.isFading() ? 1 : 0, std::memory_order_relaxed);

        float snapshotAlpha = 1.0f;
        const convo::GlobalSnapshot* snapshotFrom = nullptr;
        const convo::GlobalSnapshot* snapshotTo = nullptr;
        const bool updateFadeReturned = m_coordinator.updateFade(snapshotAlpha, snapshotFrom, snapshotTo);
        debugLastUpdateFadeReturned.store(updateFadeReturned ? 1 : 0, std::memory_order_relaxed);
        debugLastSnapshotFromNull.store(snapshotFrom == nullptr ? 1 : 0, std::memory_order_relaxed);
        debugLastSnapshotToNull.store(snapshotTo == nullptr ? 1 : 0, std::memory_order_relaxed);
        (void) snapshotAlpha;

    const double engineSampleRate = currentSampleRate.load(std::memory_order_relaxed);
    if (absDiffNoLibm(dsp->sampleRate, engineSampleRate) > 1e-6)
    {
        inputLevelLinear.store(0.0f);
        // --- クロスフェード・遅延整合処理（現行設計に準拠） ---
        DSPCore* fading = sanitizeRawPtr(fadingOutDSP.load(std::memory_order_acquire));
        bool useDryAsOld = dspCrossfadeUseDryAsOld.load(std::memory_order_acquire);
        int pendingFadeDelayBlocks = dspCrossfadeStartDelayBlocks.load(std::memory_order_acquire);
        if (fading != nullptr
            && !useDryAsOld
            && dspCrossfadePending.load(std::memory_order_acquire)
            && pendingFadeDelayBlocks > 0)
        {
            dspCrossfadeStartDelayBlocks.store(pendingFadeDelayBlocks - 1, std::memory_order_release);

            auto fadingState = procState;
            fadingState.analyzerEnabled = false;
            fadingState.adaptiveCaptureQueue = nullptr;

            std::atomic<float> fadingInputMeter { 0.0f };
            std::atomic<float> fadingOutputMeter { 0.0f };
            fading->processDouble(buffer, analyzerFifo, inputLevelLinear, outputLevelLinear, fadingState);
            return;
        }

        if ((fading != nullptr || firstIrDryCrossfadePending.load(std::memory_order_acquire))
            && dspCrossfadePending.exchange(false, std::memory_order_acq_rel))
        {
            const double fadeSec = std::max(0.001, queuedFadeTimeSec.load(std::memory_order_acquire));
            dspCrossfadeGain.reset(std::max(1.0, dsp->sampleRate), fadeSec);
            dspCrossfadeGain.setCurrentAndTargetValue(0.0);
            dspCrossfadeGain.setTargetValue(1.0);

            // レイテンシ整合値を Audio Thread スナップショットへ反映する。
            latencyDelayOld_RT = latencyDelayOld.load(std::memory_order_acquire);
            latencyDelayNew_RT = latencyDelayNew.load(std::memory_order_acquire);

            if (firstIrDryCrossfadePending.exchange(false, std::memory_order_acq_rel))
            {
                dspCrossfadeUseDryAsOld.store(true, std::memory_order_release);
                useDryAsOld = true;
            }
        }

        const bool canCrossfade = (fading != nullptr || useDryAsOld)
            && dspCrossfadeGain.isSmoothing()
            && dspCrossfadeDoubleBuffer.getNumChannels() >= 2
            && dspCrossfadeDoubleBuffer.getNumSamples() >= numSamples;

        if (canCrossfade)
        {
            // 旧DSPの出力をバッファに生成
            dspCrossfadeDoubleBuffer.clear(0, 0, numSamples);
            dspCrossfadeDoubleBuffer.clear(1, 0, numSamples);

            auto fadingState = procState;
            fadingState.analyzerEnabled = false;
            fadingState.adaptiveCaptureQueue = nullptr;

            std::atomic<float> fadingInputMeter { 0.0f };
            std::atomic<float> fadingOutputMeter { 0.0f };
            if (useDryAsOld)
            {
                const int outChannels = std::min(2, buffer.getNumChannels());
                if (outChannels > 0)
                    juce::FloatVectorOperations::copy(dspCrossfadeDoubleBuffer.getWritePointer(0, 0), buffer.getReadPointer(0, 0), numSamples);
                if (outChannels > 1)
                    juce::FloatVectorOperations::copy(dspCrossfadeDoubleBuffer.getWritePointer(1, 0), buffer.getReadPointer(1, 0), numSamples);
            }
            else
            {
                // EBR: lifetime managed by RCUReader
                fading->processDoubleToBuffer(buffer, dspCrossfadeDoubleBuffer, analyzerFifo,
                                              fadingInputMeter, fadingOutputMeter, fadingState);
            }
            dsp->processDouble(buffer, analyzerFifo, inputLevelLinear, outputLevelLinear, procState);

            const int outChannels = std::min(2, buffer.getNumChannels());
            double* dstL = (outChannels > 0) ? buffer.getWritePointer(0, 0) : nullptr;
            double* dstR = (outChannels > 1) ? buffer.getWritePointer(1, 0) : nullptr;
            const double* oldL = (outChannels > 0) ? dspCrossfadeDoubleBuffer.getReadPointer(0, 0) : nullptr;
            const double* oldR = (outChannels > 1) ? dspCrossfadeDoubleBuffer.getReadPointer(1, 0) : nullptr;

            // 遅延整合バッファを使ったクロスフェード
            const int bufferSize = latencyBufSize;
            const int delayOld = latencyDelayOld_RT;
            const int delayNew = latencyDelayNew_RT;
            double gNew = dspCrossfadeGain.getCurrentValue();
            const double gTarget = dspCrossfadeGain.getTargetValue();
            const double dg = (gTarget - gNew) / numSamples;
            // resetPendingはAudioThreadで1回だけ処理
            if (latencyResetPending.exchange(false, std::memory_order_acq_rel)) {
                if (latencyBufOldL) std::memset(latencyBufOldL, 0, sizeof(double) * bufferSize);
                if (latencyBufOldR) std::memset(latencyBufOldR, 0, sizeof(double) * bufferSize);
                if (latencyBufNewL) std::memset(latencyBufNewL, 0, sizeof(double) * bufferSize);
                if (latencyBufNewR) std::memset(latencyBufNewR, 0, sizeof(double) * bufferSize);
                latencyWritePos = 0;
            }
            for (int i = 0; i < numSamples; ++i)
            {
                latencyBufOldL[latencyWritePos] = (oldL != nullptr) ? oldL[i] : 0.0;
                latencyBufOldR[latencyWritePos] = (oldR != nullptr) ? oldR[i] : 0.0;
                latencyBufNewL[latencyWritePos] = (dstL != nullptr) ? dstL[i] : 0.0;
                latencyBufNewR[latencyWritePos] = (dstR != nullptr) ? dstR[i] : 0.0;

                int readOld = latencyWritePos - delayOld;
                int readNew = latencyWritePos - delayNew;
                // 完全wrap
                while (readOld < 0) readOld += bufferSize;
                while (readOld >= bufferSize) readOld -= bufferSize;
                while (readNew < 0) readNew += bufferSize;
                while (readNew >= bufferSize) readNew -= bufferSize;

                const double alignedOldL = latencyBufOldL[readOld];
                const double alignedOldR = latencyBufOldR[readOld];
                const double alignedNewL = latencyBufNewL[readNew];
                const double alignedNewR = latencyBufNewR[readNew];

                gNew += dg;
                const double gOld = 1.0 - gNew;

                if (dstL) dstL[i] = alignedNewL * gNew + alignedOldL * gOld;
                if (dstR) dstR[i] = alignedNewR * gNew + alignedOldR * gOld;

                latencyWritePos++;
                if (latencyWritePos >= bufferSize)
                    latencyWritePos = 0;
            }
            if (!useDryAsOld)
            {
                // EBR: managed by RCUReader
            }

            if (!dspCrossfadeGain.isSmoothing())
            {
                if (auto* done = sanitizeRawPtr(fadingOutDSP.exchange(nullptr, std::memory_order_acq_rel)))
                    retireDSP(done);
                dspCrossfadeGain.setCurrentAndTargetValue(1.0);
                dspCrossfadeDryScaleGain.setCurrentAndTargetValue(1.0);
                dspCrossfadeUseDryAsOld.store(false, std::memory_order_release);
            }
            return;
        }

        // --- 通常パス（クロスフェードなし） ---
        dsp->processDouble(buffer, analyzerFifo, inputLevelLinear, outputLevelLinear, procState);

        if (fading != nullptr && !dspCrossfadeGain.isSmoothing())
        {
            if (auto* done = sanitizeRawPtr(fadingOutDSP.exchange(nullptr, std::memory_order_acq_rel)))
                retireDSP(done);
        }
        if (!dspCrossfadeGain.isSmoothing())
            dspCrossfadeUseDryAsOld.store(false, std::memory_order_release);
        return;
    }

    // --- クロスフェード開始時: スナップショット取得・RT競合ゼロ設計 ---
    DSPCore* fading = sanitizeRawPtr(fadingOutDSP.load(std::memory_order_acquire));
    bool useDryAsOld = dspCrossfadeUseDryAsOld.load(std::memory_order_acquire);
    int pendingFadeDelayBlocks = dspCrossfadeStartDelayBlocks.load(std::memory_order_acquire);
    if (fading != nullptr
        && !useDryAsOld
        && dspCrossfadePending.load(std::memory_order_acquire)
        && pendingFadeDelayBlocks > 0)
    {
        dspCrossfadeStartDelayBlocks.store(pendingFadeDelayBlocks - 1, std::memory_order_release);

        auto fadingState = procState;
        fadingState.analyzerEnabled = false;
        fadingState.adaptiveCaptureQueue = nullptr;

        std::atomic<float> fadingInputMeter { 0.0f };
        std::atomic<float> fadingOutputMeter { 0.0f };
        fading->processDouble(buffer, analyzerFifo, inputLevelLinear, outputLevelLinear, fadingState);
        return;
    }

    if ((fading != nullptr || firstIrDryCrossfadePending.load(std::memory_order_acquire))
        && dspCrossfadePending.exchange(false, std::memory_order_acq_rel))
    {
        // queuedFadeTimeSecはcommitNewDSPでセット済み、ここでスナップショット
        const double fadeSec = std::max(0.001, queuedFadeTimeSec.load(std::memory_order_acquire));
        // latencyDelayOld/New, latencyWritePos, latencyBuf*もcommitNewDSPでセット済み
        // AudioThread側は読み取り専用、atomic不要
        dspCrossfadeGain.reset(std::max(1.0, dsp->sampleRate), fadeSec);
        dspCrossfadeGain.setCurrentAndTargetValue(0.0);
        dspCrossfadeGain.setTargetValue(1.0);

        // レイテンシ整合値を Audio Thread スナップショットへ反映する。
        latencyDelayOld_RT = latencyDelayOld.load(std::memory_order_acquire);
        latencyDelayNew_RT = latencyDelayNew.load(std::memory_order_acquire);

        if (firstIrDryCrossfadePending.exchange(false, std::memory_order_acq_rel))
        {
            dspCrossfadeUseDryAsOld.store(true, std::memory_order_release);
            useDryAsOld = true;
        }
    }

    const bool canCrossfade = (fading != nullptr || useDryAsOld)
        && dspCrossfadeGain.isSmoothing()
        && dspCrossfadeDoubleBuffer.getNumChannels() >= 2
        && dspCrossfadeDoubleBuffer.getNumSamples() >= numSamples;

    if (canCrossfade)
    {
        // --- wrap安全・スナップショット設計 ---
        dspCrossfadeDoubleBuffer.clear(0, 0, numSamples);
        dspCrossfadeDoubleBuffer.clear(1, 0, numSamples);

        auto fadingState = procState;
        fadingState.analyzerEnabled = false;
        fadingState.adaptiveCaptureQueue = nullptr;

        std::atomic<float> fadingInputMeter { 0.0f };
        std::atomic<float> fadingOutputMeter { 0.0f };
        if (useDryAsOld)
        {
            const int outChannels = std::min(2, buffer.getNumChannels());
            if (outChannels > 0)
                juce::FloatVectorOperations::copy(dspCrossfadeDoubleBuffer.getWritePointer(0, 0), buffer.getReadPointer(0, 0), numSamples);
            if (outChannels > 1)
                juce::FloatVectorOperations::copy(dspCrossfadeDoubleBuffer.getWritePointer(1, 0), buffer.getReadPointer(1, 0), numSamples);
        }
        else
        {
            // EBR: managed by RCUReader
            fading->processDoubleToBuffer(buffer, dspCrossfadeDoubleBuffer, analyzerFifo,
                                          fadingInputMeter, fadingOutputMeter, fadingState);
        }
        dsp->processDouble(buffer, analyzerFifo, inputLevelLinear, outputLevelLinear, procState);

        // スナップショット（commitNewDSPでセット済み、ここでは読み取り専用）
        const int outChannels = std::min(2, buffer.getNumChannels());
        double* dstL = (outChannels > 0) ? buffer.getWritePointer(0, 0) : nullptr;
        double* dstR = (outChannels > 1) ? buffer.getWritePointer(1, 0) : nullptr;
        const double* oldL = (outChannels > 0) ? dspCrossfadeDoubleBuffer.getReadPointer(0, 0) : nullptr;
        const double* oldR = (outChannels > 1) ? dspCrossfadeDoubleBuffer.getReadPointer(1, 0) : nullptr;

        const int bufferSize = latencyBufSize;
        int writePos = latencyWritePos;
        // ===== 遅延整合スナップショット取得（AudioThread）=====
        // ===== RT snapshot値を使用 =====
        const int delayOld = latencyDelayOld_RT;
        const int delayNew = latencyDelayNew_RT;
        // ===== resetPending処理（AudioThreadのみ）=====
        if (latencyResetPending.exchange(false, std::memory_order_acq_rel)) {
            if (latencyBufOldL) std::memset(latencyBufOldL, 0, sizeof(double) * bufferSize);
            if (latencyBufOldR) std::memset(latencyBufOldR, 0, sizeof(double) * bufferSize);
            if (latencyBufNewL) std::memset(latencyBufNewL, 0, sizeof(double) * bufferSize);
            if (latencyBufNewR) std::memset(latencyBufNewR, 0, sizeof(double) * bufferSize);
            writePos = 0;
        }
        for (int i = 0; i < numSamples; ++i) {
            latencyBufOldL[writePos] = (oldL != nullptr) ? oldL[i] : 0.0;
            latencyBufOldR[writePos] = (oldR != nullptr) ? oldR[i] : 0.0;
            latencyBufNewL[writePos] = (dstL != nullptr) ? dstL[i] : 0.0;
            latencyBufNewR[writePos] = (dstR != nullptr) ? dstR[i] : 0.0;

            int readOld = writePos - delayOld;
            int readNew = writePos - delayNew;
            // 完全wrap
            while (readOld < 0) readOld += bufferSize;
            while (readOld >= bufferSize) readOld -= bufferSize;
            while (readNew < 0) readNew += bufferSize;
            while (readNew >= bufferSize) readNew -= bufferSize;

            const double alignedOldL = latencyBufOldL[readOld];
            const double alignedOldR = latencyBufOldR[readOld];
            const double alignedNewL = latencyBufNewL[readNew];
            const double alignedNewR = latencyBufNewR[readNew];

            const double gNew = dspCrossfadeGain.getNextValue();
            const double gOld = 1.0 - gNew;

            if (dstL) dstL[i] = alignedNewL * gNew + alignedOldL * gOld;
            if (dstR) dstR[i] = alignedNewR * gNew + alignedOldR * gOld;

            writePos = (writePos + 1);
            if (writePos == bufferSize)
                writePos = 0;
        }
        latencyWritePos = writePos;
        if (!useDryAsOld)
        {
            // EBR: managed by RCUReader
        }

        if (!dspCrossfadeGain.isSmoothing())
        {
            if (auto* done = sanitizeRawPtr(fadingOutDSP.exchange(nullptr, std::memory_order_acq_rel)))
                retireDSP(done);
            dspCrossfadeGain.setCurrentAndTargetValue(1.0);
            dspCrossfadeUseDryAsOld.store(false, std::memory_order_release);
        }
    }
    else
    {
        dsp->processDouble(buffer, analyzerFifo, inputLevelLinear, outputLevelLinear, procState);

        if (fading != nullptr && !dspCrossfadeGain.isSmoothing())
        {
            if (auto* done = sanitizeRawPtr(fadingOutDSP.exchange(nullptr, std::memory_order_acq_rel)))
                retireDSP(done);
        }
        if (!dspCrossfadeGain.isSmoothing())
            dspCrossfadeUseDryAsOld.store(false, std::memory_order_release);
    }
}

#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_SNAPSHOT)

void AudioEngine::processWithSnapshot(const juce::AudioSourceChannelInfo& bufferToFill,
                                      const convo::GlobalSnapshot* snap,
                                      bool isFadingTarget)
{
    if (snap == nullptr)
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    DSPCore* dsp = currentDSP.load(std::memory_order_acquire);
    if (dsp == nullptr)
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    const uint64_t hash = snap->eqCoeffHash;
    if (hash != debugLastAppliedEqHash.load(std::memory_order_relaxed))
    {
        debugLastAppliedEqHash.store(hash, std::memory_order_relaxed);
        debugAppliedEqHashVersion.fetch_add(1u, std::memory_order_relaxed);
    }

    const bool eqBypassed = snap->eqBypass;
    const bool convBypassed = snap->convBypass;
    const ProcessingOrder order = snap->processingOrder;
    const bool softClip = snap->softClipEnabled;
    const float satAmt = snap->saturationAmount;
    const double headroomGain = snap->inputHeadroomGain;
    const double makeupGain = snap->outputMakeupGain;
    const double convInputTrimGain = snap->convInputTrimGain;

    if (!isFadingTarget)
    {
        eqBypassActive.store(eqBypassed, std::memory_order_relaxed);
        convBypassActive.store(convBypassed, std::memory_order_relaxed);
    }

    const AnalyzerSource analyzerSource = currentAnalyzerSource.load(std::memory_order_relaxed);
    const bool analyzerEnabledNow = analyzerEnabled.load(std::memory_order_relaxed);
    const convo::HCMode hcMode = convHCFilterMode.load(std::memory_order_relaxed);
    const convo::LCMode lcMode = convLCFilterMode.load(std::memory_order_relaxed);
    const convo::HCMode lpfMode = eqLPFFilterMode.load(std::memory_order_relaxed);
    const int adaptiveCoeffBankIndex = currentAdaptiveCoeffBankIndex.load(std::memory_order_acquire);
    const auto& adaptiveCoeffBank = getAdaptiveCoeffBankForIndex(adaptiveCoeffBankIndex);
    const bool adaptiveCaptureEnabled = noiseShaperLearner && noiseShaperLearner->isRunning();

    // RCU snapshot acquisition: generation and active pointer are kept consistent by double buffering.
    const uint32_t genSnapshot = adaptiveCoeffBank.generation.load(std::memory_order_acquire);
    const CoeffSet* safeAdaptiveSet = AudioEngine::getActiveCoeffSet(adaptiveCoeffBank);

    if (!safeAdaptiveSet) {
        // Keep nullptr and let DSPCore::process handle the fallback path.
    }
    const uint32_t adaptiveGenAfter = genSnapshot;

    DSPCore::ProcessingState procState {
        .eqBypassed               = eqBypassed,
        .convBypassed             = convBypassed,
        .order                    = order,
        .analyzerSource           = analyzerSource,
        .analyzerEnabled          = isFadingTarget ? false : analyzerEnabledNow,
        .softClipEnabled          = softClip,
        .saturationAmount         = satAmt,
        .inputHeadroomGain        = headroomGain,
        .outputMakeupGain         = makeupGain,
        .convolverInputTrimGain   = convInputTrimGain,
        .convHCMode               = hcMode,
        .convLCMode               = lcMode,
        .eqLPFMode                = lpfMode,
        .adaptiveCoeffBankIndex   = adaptiveCoeffBankIndex,
        .adaptiveCoeffSet         = safeAdaptiveSet,
        .adaptiveCoeffGeneration  = adaptiveGenAfter,
        .adaptiveCaptureSampleRateHz = static_cast<int>(dsp->sampleRate + 0.5),
        .adaptiveCaptureBitDepth  = dsp->ditherBitDepth,
        .captureSessionId         = dsp->currentCaptureSessionId,
        .adaptiveCaptureQueue     = adaptiveCaptureEnabled ? &audioCaptureQueue : nullptr
    };

    std::atomic<float> fadingInputMeter { 0.0f };
    std::atomic<float> fadingOutputMeter { 0.0f };
    auto& inMeter = isFadingTarget ? fadingInputMeter : inputLevelLinear;
    auto& outMeter = isFadingTarget ? fadingOutputMeter : outputLevelLinear;
    dsp->process(bufferToFill, analyzerFifo, inMeter, outMeter, procState);
}

#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_TO_BUFFER)

void AudioEngine::DSPCore::processToBuffer(const juce::AudioSourceChannelInfo& source,
                                          juce::AudioBuffer<float>& destination,
                                          LockFreeAudioRingBuffer& analyzerFifo,
                                          std::atomic<float>& inputLevelLinear,
                                          std::atomic<float>& outputLevelLinear,
                                          const ProcessingState& state)
{
    const int numSamples = source.numSamples;
    const int numChannels = std::min(2, source.buffer != nullptr ? source.buffer->getNumChannels() : 0);

    if (source.buffer == nullptr || numSamples <= 0 || destination.getNumSamples() < numSamples)
    {
        destination.clear();
        return;
    }

    for (int ch = 0; ch < numChannels; ++ch)
    {
        const float* src = source.buffer->getReadPointer(ch, source.startSample);
        float* dst = destination.getWritePointer(ch, 0);
        juce::FloatVectorOperations::copy(dst, src, numSamples);
    }
    for (int ch = numChannels; ch < destination.getNumChannels(); ++ch)
        destination.clear(ch, 0, numSamples);

    juce::AudioSourceChannelInfo destinationInfo(&destination, 0, numSamples);
    process(destinationInfo, analyzerFifo, inputLevelLinear, outputLevelLinear, state);
}

void AudioEngine::DSPCore::processDoubleToBuffer(const juce::AudioBuffer<double>& source,
                                                 juce::AudioBuffer<double>& destination,
                                                 LockFreeAudioRingBuffer& analyzerFifo,
                                                 std::atomic<float>& inputLevelLinear,
                                                 std::atomic<float>& outputLevelLinear,
                                                 const ProcessingState& state)
{
    const int numSamples = source.getNumSamples();
    const int numChannels = std::min(2, source.getNumChannels());

    if (numSamples <= 0 || destination.getNumSamples() < numSamples)
    {
        destination.clear();
        return;
    }

    for (int ch = 0; ch < numChannels; ++ch)
    {
        const double* src = source.getReadPointer(ch, 0);
        double* dst = destination.getWritePointer(ch, 0);
        juce::FloatVectorOperations::copy(dst, src, numSamples);
    }
    for (int ch = numChannels; ch < destination.getNumChannels(); ++ch)
        destination.clear(ch, 0, numSamples);

    processDouble(destination, analyzerFifo, inputLevelLinear, outputLevelLinear, state);
}

#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_DSP_FLOAT)

void AudioEngine::DSPCore::process(const juce::AudioSourceChannelInfo& bufferToFill,
                                  LockFreeAudioRingBuffer& analyzerFifo,
                                  std::atomic<float>& inputLevelLinear,
                                  std::atomic<float>& outputLevelLinear,
                                  const ProcessingState& state) // ProcessingState構造体でパラメータを受け取る
{
    const int numSamples = bufferToFill.numSamples;

    // バッファサイズ超過ガード (Buffer Overrun Protection)
    if (numSamples > maxSamplesPerBlock)
    {
        // この状況は通常発生しないが、万が一ホストが予期せぬサイズのバッファを渡してきた場合の安全策
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    // ==================================================================
    // 【Issue 3 追加防御】オーバーラン即検出（リリースでも有効）
    // オーバーサンプリング有効時にupBlockサイズが内部バッファを超えないことを保証
    // ==================================================================
    if (oversamplingFactor > 1)
    {
        const int expectedUpSize = numSamples * static_cast<int>(oversamplingFactor);

        // Fix: Releaseビルドでも確実にチェックし、バッファ破壊を防ぐ
        if (expectedUpSize > maxInternalBlockSize)
        {
            bufferToFill.clearActiveBufferRegion(); // 無音を出力
            return;
        }
    }

    // ── 入力処理 + Raw Input Analyzer Tap ──
    // processInput() がヘッドルームゲイン適用前の raw レベルを返す。
    // analyzerInputTap=true の場合、同関数内で pre-gain の FIFO プッシュも行う。
    const bool inputTap = state.analyzerEnabled && (state.analyzerSource == AnalyzerSource::Input);
    const float rawInputLinear = processInput(bufferToFill, numSamples, state.inputHeadroomGain,
                                              inputTap, analyzerFifo);

    //----------------------------------------------------------
    // AudioBlockの構築 (AlignedBufferを使用)
    // ※ この時点では既にヘッドルームゲイン適用済み
    //----------------------------------------------------------
    double* channels[2] = { alignedL.get(), alignedR.get() };
    juce::dsp::AudioBlock<double> processBlock(channels, 2, numSamples);

    //----------------------------------------------------------
    // 入力レベル記録 (raw: ヘッドルームゲイン適用前の値を使用)
    //----------------------------------------------------------
    inputLevelLinear.store(rawInputLinear, std::memory_order_relaxed);

    //----------------------------------------------------------
    // オーバーサンプリング処理ブロック
    //----------------------------------------------------------
    // バッファ全体ではなく、有効なサンプル数のみをラップする (重要)
    juce::dsp::AudioBlock<double> originalBlock = processBlock; // 元サイズを保存

    // アップサンプリング
    if (oversamplingFactor > 1)
    {
        processBlock = oversampling.processUp(originalBlock, static_cast<int>(originalBlock.getNumChannels()));

        // [Safety Guard] サイズ超過またはエラー(空ブロック)の場合は無音にして中断
        if (processBlock.getNumSamples() == 0 || processBlock.getNumSamples() > static_cast<size_t>(maxInternalBlockSize))
        {
            jassertfalse;
            bufferToFill.clearActiveBufferRegion();
            return;
        }

        // オーバーサンプリング直後に高精度DC除去を適用
        // これにより、後段のDSP処理（Convolver/EQ）にクリーンな信号を渡す
        const int numOSSamples = (int)processBlock.getNumSamples();
        if (processBlock.getNumChannels() > 0)
            osDCBlockerL.process(processBlock.getChannelPointer(0), numOSSamples);
        if (processBlock.getNumChannels() > 1)
            osDCBlockerR.process(processBlock.getChannelPointer(1), numOSSamples);
    }

    // ── Analyzer Input Tap は processInput() 内で pre-gain 済みデータをプッシュ済み ──
    // (oversampling後のここではなく、ゲイン適用前の生データを表示するため移動)

    int numProcSamples = (int)processBlock.getNumSamples();
    int numProcChannels = (int)processBlock.getNumChannels(); // 通常は2

    const convo::GlobalSnapshot* snap = ownerEngine ? ownerEngine->m_coordinator.getCurrent() : nullptr;
    const bool useSnapshotEq = (snap != nullptr);
    const convo::EQParameters* eqParamsToUse = nullptr;
    const EQCoeffCache* eqCacheToUse = nullptr;
    if (useSnapshotEq && ownerEngine != nullptr)
    {
        const uint64_t hash = snap->eqCoeffHash;
        eqParamsToUse = &snap->eqParams;
        eqCacheToUse = ownerEngine->eqCacheManager.get(hash);
        if (hash != ownerEngine->debugLastAppliedEqHash.load(std::memory_order_relaxed))
        {
            ownerEngine->debugLastAppliedEqHash.store(hash, std::memory_order_relaxed);
            ownerEngine->debugAppliedEqHashVersion.fetch_add(1u, std::memory_order_relaxed);
        }
    }
    else if (ownerEngine != nullptr)
    {
        uint64_t fallbackHash = 0;
        const convo::EQParameters& fallbackParams = ownerEngine->getLatestEqParamsFallback(fallbackHash);
        eqParamsToUse = &fallbackParams;
        eqCacheToUse = ownerEngine->eqCacheManager.get(fallbackHash);
        if (fallbackHash != 0 && fallbackHash != ownerEngine->debugLastAppliedEqHash.load(std::memory_order_relaxed))
        {
            ownerEngine->debugLastAppliedEqHash.store(fallbackHash, std::memory_order_relaxed);
            ownerEngine->debugAppliedEqHashVersion.fetch_add(1u, std::memory_order_relaxed);
        }
    }

    //----------------------------------------------------------
    // DSP処理チェーン (Dynamic Processing Order)
    //----------------------------------------------------------
    // EQバイパス要求を毎ブロック反映（実効バイパスはEQProcessor内の短フェードで切替）
    eq.setBypass(state.eqBypassed);

    // プロセッサには AudioBlock を直接渡す (AudioBuffer作成によるmalloc回避)
    if (state.order == ProcessingOrder::ConvolverThenEQ) // stateから読み出し
    {
        // 1. Convolver
        if (!state.convBypassed) // stateから読み出し
            convolver.process(processBlock);
        // 2. EQ
        if (!state.eqBypassed)
        {
            if (eqParamsToUse != nullptr)
            {
                eq.process(processBlock, *eqParamsToUse, eqCacheToUse);
            }
            else
            {
                eq.process(processBlock);
            }
        }
        else
        {
            eq.process(processBlock);
        }
    }
    else
    {
        // 1. EQ
        if (!state.eqBypassed)
        {
            if (eqParamsToUse != nullptr)
            {
                eq.process(processBlock, *eqParamsToUse, eqCacheToUse);
            }
            else
            {
                eq.process(processBlock);
            }
        }
        else
        {
            eq.process(processBlock);
        }
        // 2. Convolver
        if (!state.convBypassed) // stateから読み出し
        {
            // EQ→Conv 時: コンボルバー入力トリムを適用してから畳み込む
            if (state.convolverInputTrimGain != 1.0)
            {
                for (size_t ch = 0; ch < processBlock.getNumChannels(); ++ch)
                {
                    double* ptr = processBlock.getChannelPointer(ch);
                    scaleBlockFallback(ptr, (int)processBlock.getNumSamples(), state.convolverInputTrimGain);
                }
            }
            convolver.process(processBlock);
        }
    }

    // ─── 出力周波数フィルター ──────────────────────────────────────
    // filter.txt: DSP処理チェーンの直後・出力メイクアップゲイン適用前に挿入
    // ① コンボルバー最終段: ハイカット(Sharp/Natural/Soft) + ローカット(Natural/Soft)
    // ② EQ最終段         : ハイパス(固定 20Hz) + ローパス(Sharp/Natural/Soft)
    // ① コンボルバー最終段: NUC が irFreqDomain に焼き込み済み → IIR 不要
    {
        const bool convActive = !state.convBypassed;
        const bool eqActive   = !state.eqBypassed;
        if (convActive || eqActive)
        {
            const bool convIsLast = convActive &&
                (!eqActive || state.order == ProcessingOrder::EQThenConvolver);
            // ① conv-last の場合は NUC 内部で処理済みのため IIR をスキップ
            if (!convIsLast)
            {
                outputFilter.process(processBlock, /*convIsLast=*/false,
                                     state.convHCMode, state.convLCMode, state.eqLPFMode);
            }
        }
    }

    //----------------------------------------------------------
    // 出力メイクアップゲイン
    // [Bug Fix] processDouble() にのみ存在し float path に欠落していたため追加。
    // 配置: OutputFilter 直後・SoftClipper 直前。processDouble() と同一位置・同一ロジック。
    //----------------------------------------------------------
    for (size_t ch = 0; ch < processBlock.getNumChannels(); ++ch)
    {
        double* ptr = processBlock.getChannelPointer(ch);
        scaleBlockFallback(ptr, (int)processBlock.getNumSamples(), state.outputMakeupGain);
    }

    //----------------------------------------------------------
    // ソフトクリッピング (Soft Clipping)
    // 配置: ダウンサンプリング前に行うことで、倍音成分の折り返しノイズ(エイリアシング)を低減する。
    //----------------------------------------------------------
    if (state.softClipEnabled) // stateから読み出し
    {
        const double sat = static_cast<double>(state.saturationAmount); // stateから読み出し
        const double CLIP_THRESHOLD = 0.95 - 0.45 * sat;
        const double CLIP_KNEE      = 0.05 + 0.35 * sat;
        const double CLIP_ASYMMETRY = 0.10 * sat;

        for (int ch = 0; ch < numProcChannels; ++ch)
        {
            double* data = processBlock.getChannelPointer(ch);
            // ブロック間インターサンプルピーク状態をチャンネルごとに渡す
            softClipBlockAVX2(data, numProcSamples, CLIP_THRESHOLD, CLIP_KNEE, CLIP_ASYMMETRY,
                               softClipPrevSample[ch < 2 ? ch : 1]);
        }
    }

    //----------------------------------------------------------

    // ダウンサンプリング (結果は processBuffer に書き戻される)
    if (oversamplingFactor > 1)
    {
        oversampling.processDown(processBlock, originalBlock, static_cast<int>(originalBlock.getNumChannels()));
        processBlock = originalBlock;
    }

    // ── Analyzer Output Tap (Post-DSP, Post-Downsampling) ──
    // オーバーサンプリング有効時でも、UI へはベースレートのデータを供給する。
    if (state.analyzerEnabled && state.analyzerSource == AnalyzerSource::Output)
    {
        pushToFifo(processBlock, analyzerFifo);
    }

    //----------------------------------------------------------
    // 出力レベル計算 (DC除去後のクリーンな信号で計測)
    //----------------------------------------------------------
    // オーバーサンプリング有効時は、ダウンサンプリング後の信号(originalBlock)を使用する
    const float outputLinear = measureLevel(originalBlock);
    outputLevelLinear.store(outputLinear, std::memory_order_relaxed);

    processOutput(bufferToFill, numSamples, state);

    // === 【Issue 5 追加】新DSP切り替え時のFade-in Ramp（最終出力に適用）===
    {
        int fadeLeft = fadeInSamplesLeft.load(std::memory_order_relaxed);
        if (fadeLeft > 0)
        {
            const int rampThisBlock = std::min(numSamples, fadeLeft);
            const float gainStep = 1.0f / static_cast<float>(FADE_IN_SAMPLES);
            const float startGain = static_cast<float>(FADE_IN_SAMPLES - fadeLeft) * gainStep;
            auto* buffer = bufferToFill.buffer;
            const int startSample = bufferToFill.startSample;
            const int numChannels = buffer->getNumChannels();

            for (int ch = 0; ch < numChannels; ++ch)
                applyGainRamp(buffer->getWritePointer(ch, startSample), rampThisBlock, startGain, gainStep);

            fadeInSamplesLeft.store(fadeLeft - rampThisBlock, std::memory_order_relaxed);
        }
    }
}

#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_DSP_DOUBLE)

void AudioEngine::DSPCore::processDouble(juce::AudioBuffer<double>& buffer,
                                         LockFreeAudioRingBuffer& analyzerFifo,
                                         std::atomic<float>& inputLevelLinear,
                                         std::atomic<float>& outputLevelLinear,
                                         const ProcessingState& state)
{
    const int numSamples = buffer.getNumSamples();

    if (numSamples > maxSamplesPerBlock)
    {
        buffer.clear();
        return;
    }

    if (oversamplingFactor > 1)
    {
        const int expectedUpSize = numSamples * static_cast<int>(oversamplingFactor);
        if (expectedUpSize > maxInternalBlockSize)
        {
            buffer.clear();
            return;
        }
    }

    // ── 入力処理 + Raw Input Analyzer Tap ──
    const bool inputTapD = state.analyzerEnabled && (state.analyzerSource == AnalyzerSource::Input);
    const float rawInputLinearD = processInputDouble(buffer, numSamples, state.inputHeadroomGain,
                                                     inputTapD, analyzerFifo);
    inputLevelLinear.store(rawInputLinearD, std::memory_order_relaxed);

    const bool requestedFullBypass = state.eqBypassed && state.convBypassed;
    if (requestedFullBypass != bypassedDouble)
    {
        bypassFadeGainDouble.setTargetValue(requestedFullBypass ? 0.0 : 1.0);
        bypassedDouble = requestedFullBypass;
    }

    double* channels[2] = { alignedL.get(), alignedR.get() };
    juce::dsp::AudioBlock<double> processBlock(channels, 2, numSamples);
    juce::dsp::AudioBlock<double> originalBlock = processBlock;

    if (dryBypassBufferDoubleL && dryBypassBufferDoubleR && dryBypassCapacityDouble >= numSamples)
    {
        juce::FloatVectorOperations::copy(dryBypassBufferDoubleL.get(), alignedL.get(), numSamples);
        juce::FloatVectorOperations::copy(dryBypassBufferDoubleR.get(), alignedR.get(), numSamples);
    }

    if (oversamplingFactor > 1)
    {
        processBlock = oversampling.processUp(originalBlock, static_cast<int>(originalBlock.getNumChannels()));

        if (processBlock.getNumSamples() == 0 || processBlock.getNumSamples() > static_cast<size_t>(maxInternalBlockSize))
        {
            jassertfalse;
            buffer.clear();
            return;
        }

        const int numOSSamples = static_cast<int>(processBlock.getNumSamples());
        if (processBlock.getNumChannels() > 0)
            osDCBlockerL.process(processBlock.getChannelPointer(0), numOSSamples);
        if (processBlock.getNumChannels() > 1)
            osDCBlockerR.process(processBlock.getChannelPointer(1), numOSSamples);
    }

    // ── Analyzer Input Tap は processInputDouble() 内で pre-gain データをプッシュ済み ──

    const int numProcSamples = static_cast<int>(processBlock.getNumSamples());
    const int numProcChannels = static_cast<int>(processBlock.getNumChannels());

    const convo::GlobalSnapshot* snap = ownerEngine ? ownerEngine->m_coordinator.getCurrent() : nullptr;
    const bool useSnapshotEq = (snap != nullptr);
    const convo::EQParameters* eqParamsToUse = nullptr;
    const EQCoeffCache* eqCacheToUse = nullptr;
    if (useSnapshotEq && ownerEngine != nullptr)
    {
        const uint64_t hash = snap->eqCoeffHash;
        eqParamsToUse = &snap->eqParams;
        eqCacheToUse = ownerEngine->eqCacheManager.get(hash);
        if (hash != ownerEngine->debugLastAppliedEqHash.load(std::memory_order_relaxed))
        {
            ownerEngine->debugLastAppliedEqHash.store(hash, std::memory_order_relaxed);
            ownerEngine->debugAppliedEqHashVersion.fetch_add(1u, std::memory_order_relaxed);
        }
    }
    else if (ownerEngine != nullptr)
    {
        uint64_t fallbackHash = 0;
        const convo::EQParameters& fallbackParams = ownerEngine->getLatestEqParamsFallback(fallbackHash);
        eqParamsToUse = &fallbackParams;
        eqCacheToUse = ownerEngine->eqCacheManager.get(fallbackHash);
        if (fallbackHash != 0 && fallbackHash != ownerEngine->debugLastAppliedEqHash.load(std::memory_order_relaxed))
        {
            ownerEngine->debugLastAppliedEqHash.store(fallbackHash, std::memory_order_relaxed);
            ownerEngine->debugAppliedEqHashVersion.fetch_add(1u, std::memory_order_relaxed);
        }
    }

    eq.setBypass(state.eqBypassed);

    if (state.order == ProcessingOrder::ConvolverThenEQ)
    {
        if (!state.convBypassed)
            convolver.process(processBlock);
        if (!state.eqBypassed)
        {
            if (eqParamsToUse != nullptr)
            {
                eq.process(processBlock, *eqParamsToUse, eqCacheToUse);
            }
            else
            {
                eq.process(processBlock);
            }
        }
        else
        {
            eq.process(processBlock);
        }
    }
    else
    {
        if (!state.eqBypassed)
        {
            if (eqParamsToUse != nullptr)
            {
                eq.process(processBlock, *eqParamsToUse, eqCacheToUse);
            }
            else
            {
                eq.process(processBlock);
            }
        }
        else
        {
            eq.process(processBlock);
        }
        if (!state.convBypassed)
        {
            // EQ→Conv 時: コンボルバー入力トリムを適用してから畳み込む
            if (state.convolverInputTrimGain != 1.0)
            {
                for (size_t ch = 0; ch < processBlock.getNumChannels(); ++ch)
                {
                    double* ptr = processBlock.getChannelPointer(ch);
                    scaleBlockFallback(ptr, (int)processBlock.getNumSamples(), state.convolverInputTrimGain);
                }
            }
            convolver.process(processBlock);
        }
    }

    // ─── 出力周波数フィルター ──────────────────────────────────────
    // ① conv-last: NUC irFreqDomain 焼き込み済み → IIR スキップ
    // ② eq-last:   引き続き IIR で処理
    {
        const bool convActive = !state.convBypassed;
        const bool eqActive   = !state.eqBypassed;
        if (convActive || eqActive)
        {
            const bool convIsLast = convActive &&
                (!eqActive || state.order == ProcessingOrder::EQThenConvolver);
            if (!convIsLast)
            {
                outputFilter.process(processBlock, /*convIsLast=*/false,
                                     state.convHCMode, state.convLCMode, state.eqLPFMode);
            }
        }
    }

    // Output Makeup Gain
    for (size_t ch = 0; ch < processBlock.getNumChannels(); ++ch)
    {
        double* ptr = processBlock.getChannelPointer(ch);
        scaleBlockFallback(ptr, (int)processBlock.getNumSamples(), state.outputMakeupGain);
    }

    if (state.softClipEnabled)
    {
        const double sat = static_cast<double>(state.saturationAmount);
        const double CLIP_THRESHOLD = 0.95 - 0.45 * sat;
        const double CLIP_KNEE      = 0.05 + 0.35 * sat;
        const double CLIP_ASYMMETRY = 0.10 * sat;

        for (int ch = 0; ch < numProcChannels; ++ch)
        {
            double* data = processBlock.getChannelPointer(ch);
            // ブロック間インターサンプルピーク状態をチャンネルごとに渡す
            softClipBlockAVX2(data, numProcSamples, CLIP_THRESHOLD, CLIP_KNEE, CLIP_ASYMMETRY,
                               softClipPrevSample[ch < 2 ? ch : 1]);
        }
    }

    const bool bypassBlendRequested = bypassFadeGainDouble.isSmoothing() || requestedFullBypass;
    if (oversamplingFactor == 1
        && dryBypassBufferDoubleL
        && dryBypassBufferDoubleR
        && dryBypassCapacityDouble >= numSamples
        && bypassBlendRequested)
    {
        double* wetL = (numProcChannels > 0) ? processBlock.getChannelPointer(0) : nullptr;
        double* wetR = (numProcChannels > 1) ? processBlock.getChannelPointer(1) : nullptr;
        const double* dryL = dryBypassBufferDoubleL.get();
        const double* dryR = dryBypassBufferDoubleR.get();
        for (int i = 0; i < numProcSamples; ++i)
        {
            const double gWet = bypassFadeGainDouble.getNextValue();
            const double gDry = 1.0 - gWet;
            if (wetL != nullptr)
                wetL[i] = wetL[i] * gWet + dryL[i] * gDry;
            if (wetR != nullptr)
                wetR[i] = wetR[i] * gWet + dryR[i] * gDry;
        }
    }

    if (oversamplingFactor > 1)
    {
        oversampling.processDown(processBlock, originalBlock, static_cast<int>(originalBlock.getNumChannels()));
        processBlock = originalBlock;

        if (bypassBlendRequested)
        {
            double* wetL = processBlock.getNumChannels() > 0 ? processBlock.getChannelPointer(0) : nullptr;
            double* wetR = processBlock.getNumChannels() > 1 ? processBlock.getChannelPointer(1) : nullptr;
            for (int i = 0; i < numSamples; ++i)
            {
                const double gWet = bypassFadeGainDouble.getNextValue();
                if (wetL != nullptr)
                    wetL[i] *= gWet;
                if (wetR != nullptr)
                    wetR[i] *= gWet;
            }
        }
    }

    // ── Analyzer Output Tap (Post-DSP, Post-Downsampling) ──
    // オーバーサンプリング有効時でも、UI へはベースレートのデータを供給する。
    if (state.analyzerEnabled && state.analyzerSource == AnalyzerSource::Output)
        pushToFifo(processBlock, analyzerFifo);

    const float outputLinear = measureLevel(originalBlock);
    outputLevelLinear.store(outputLinear, std::memory_order_relaxed);

    processOutputDouble(buffer, numSamples, state);

    int fadeLeft = fadeInSamplesLeft.load(std::memory_order_relaxed);
    if (fadeLeft > 0)
    {
        const int rampThisBlock = std::min(numSamples, fadeLeft);
        const double gainStep = 1.0 / static_cast<double>(FADE_IN_SAMPLES);
        const double startGain = static_cast<double>(FADE_IN_SAMPLES - fadeLeft) * gainStep;
        const int numChannels = buffer.getNumChannels();

        for (int ch = 0; ch < numChannels; ++ch)
            applyGainRamp(buffer.getWritePointer(ch), rampThisBlock, startGain, gainStep);

        fadeInSamplesLeft.store(fadeLeft - rampThisBlock, std::memory_order_relaxed);
    }
}

#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_DSP_IO)

float AudioEngine::DSPCore::measureLevel (const juce::dsp::AudioBlock<const double>& block) const noexcept
{
    double maxLevel = 0.0;
    const int numChannels = (int)block.getNumChannels();
    const int numSamples = (int)block.getNumSamples();

    for (int ch = 0; ch < numChannels; ++ch)
    {
        auto range = juce::FloatVectorOperations::findMinAndMax(block.getChannelPointer(ch), numSamples);
        const double level = std::max(absNoLibm(range.getStart()), absNoLibm(range.getEnd()));
        if (level > maxLevel) maxLevel = level;
    }

    // 【Fix Bug #8】Audio Thread 内での gainToDecibels (std::log10 / libm) 呼び出しを排除。
    // linear gain をそのまま返す。dB 変換は UI Thread 側の getInputLevel() / getOutputLevel() で行う。
    return static_cast<float>(maxLevel);
}

void AudioEngine::DSPCore::pushToFifo(const juce::dsp::AudioBlock<const double>& block,
                                      LockFreeAudioRingBuffer& analyzerFifo) const noexcept
{
    analyzerFifo.push(block);
}

float AudioEngine::DSPCore::processInput(const juce::AudioSourceChannelInfo& bufferToFill, int numSamples,
                                          double headroomGain,
                                          bool analyzerInputTap,
                                          LockFreeAudioRingBuffer& analyzerFifo) noexcept
{
    auto* buffer = bufferToFill.buffer;
    const int startSample = bufferToFill.startSample;
    const int effectiveInputChannels = std::min(buffer->getNumChannels(), 2);

    for (int ch = 0; ch < effectiveInputChannels; ++ch)
    {
        const float* src = buffer->getReadPointer(ch, startSample);
        double* dst = (ch == 0) ? alignedL.get() : alignedR.get();
        // Convert without gain (gain=1.0), but with sanitization
        convo::input_transform::convertFloatToDoubleHighQuality(src, dst, numSamples, 1.0);
    }

    // 入力がないチャンネル、または余剰チャンネルはクリア
    // ただし、Mono->Stereo展開を行う場合はCh 1のクリアをスキップする (直後に上書きされるため)
    // ロジック整理:
    // 1. 入力が1chで出力が2ch以上の場合 -> Ch 1 (R) はコピーされるのでクリア不要。Ch 2以降をクリア。
    // 2. それ以外 -> 入力チャンネル数以降をすべてクリア。
    // AlignedBufferは常に2ch分 (L/R) 用意されていると仮定
    const bool expandMono = (effectiveInputChannels == 1);
    const int clearStartCh = expandMono ? 2 : effectiveInputChannels;

    for (int ch = clearStartCh; ch < 2; ++ch)
    {
        double* dst = (ch == 0) ? alignedL.get() : alignedR.get();
        juce::FloatVectorOperations::clear(dst, numSamples);
    }

    // ── Mono -> Stereo 展開 ──
    // 入力が1chのみで、処理バッファが2ch以上ある場合、LchをRchにコピーする
    // これにより、モノラルマイク入力時などでもステレオ処理として扱えるようにし、
    // 後段のステレオエフェクト（Convolver等）での片側無音を防ぐ。
    if (expandMono)
    {
        const double* src = alignedL.get();
        double* dst = alignedR.get();
        // 高速なメモリコピー (double配列)
        juce::FloatVectorOperations::copy(dst, src, numSamples);
    }

    // 2. Measure level before gain
    double* channels[2] = { alignedL.get(), alignedR.get() };
    juce::dsp::AudioBlock<double> block(channels, 2, numSamples);
    const float inputLevel = measureLevel(block);

    // ── Analyzer Input Tap (Pre-Gain / Raw Input) ──
    // ヘッドルームゲイン適用前の raw 入力データを FIFO へプッシュ。
    // インプットスペアナ・レベルメーターが "入力されたデータそのもの" を表示するために必須。
    if (analyzerInputTap)
        pushToFifo(block, analyzerFifo);

    // 3. Apply headroom gain
    if (absDiffNoLibm(headroomGain, 1.0) > 1e-9)
    {
        for (int ch = 0; ch < 2; ++ch)
            scaleBlockFallback(block.getChannelPointer(ch), numSamples, headroomGain);
    }

    // ── 入力段DC除去 (ブロックモード) ──
    // 旧: 1 サンプルずつ 4 次 IIR を呼出 (~20 ops/sample)
    // 新: UltraHighRateDCBlocker.process(ptr, N) でブロック単位処理 (~4 ops/sample)
    double* lPtr = alignedL.get();
    double* rPtr = alignedR.get();
    inputDCBlockerL.process(lPtr, numSamples);
    inputDCBlockerR.process(rPtr, numSamples);

    return inputLevel;
}

float AudioEngine::DSPCore::processInputDouble(const juce::AudioBuffer<double>& buffer, int numSamples,
                                               double headroomGain,
                                               bool analyzerInputTap,
                                               LockFreeAudioRingBuffer& analyzerFifo) noexcept
{
    const int effectiveInputChannels = std::min(buffer.getNumChannels(), 2);

    for (int ch = 0; ch < effectiveInputChannels; ++ch)
    {
        const double* src = buffer.getReadPointer(ch);
        double* dst = (ch == 0) ? alignedL.get() : alignedR.get();
        // Convert without gain (gain=1.0), but with sanitization
        convo::input_transform::convertDoubleToDoubleHighQuality(src, dst, numSamples, 1.0);
    }

    const bool expandMono = (effectiveInputChannels == 1);
    const int clearStartCh = expandMono ? 2 : effectiveInputChannels;

    for (int ch = clearStartCh; ch < 2; ++ch)
    {
        double* dst = (ch == 0) ? alignedL.get() : alignedR.get();
        juce::FloatVectorOperations::clear(dst, numSamples);
    }

    if (expandMono)
        std::memcpy(alignedR.get(), alignedL.get(), numSamples * sizeof(double));

    // 2. Measure level before gain
    double* channels[2] = { alignedL.get(), alignedR.get() };
    juce::dsp::AudioBlock<double> block(channels, 2, numSamples);
    const float inputLevel = measureLevel(block);

    // ── Analyzer Input Tap (Pre-Gain / Raw Input) ──
    if (analyzerInputTap)
        pushToFifo(block, analyzerFifo);

    // 3. Apply headroom gain
    if (absDiffNoLibm(headroomGain, 1.0) > 1e-9)
    {
        for (int ch = 0; ch < 2; ++ch)
            scaleBlockFallback(block.getChannelPointer(ch), numSamples, headroomGain);
    }

    // ── 入力段DC除去 (ブロックモード) ──
    double* lPtr = alignedL.get();
    double* rPtr = alignedR.get();
    inputDCBlockerL.process(lPtr, numSamples);
    inputDCBlockerR.process(rPtr, numSamples);

    return inputLevel;
}

void AudioEngine::DSPCore::applyFixedLatencyDelay(double* dataL, double* dataR, int numSamples) noexcept
{
    if (fixedLatencySamples <= 0 || fixedLatencyBufferSize <= 0 || dataL == nullptr)
        return;

    const int delay = std::min(fixedLatencySamples, fixedLatencyBufferSize - 1);
    int writePos = fixedLatencyWritePos;
    const int bufferSize = fixedLatencyBufferSize;
    double* delayL = fixedLatencyBufferL.get();
    double* delayR = fixedLatencyBufferR.get();

    for (int i = 0; i < numSamples; ++i)
    {
        delayL[writePos] = dataL[i];
        if (dataR != nullptr)
            delayR[writePos] = dataR[i];

        int readPos = writePos - delay;
        while (readPos < 0)
            readPos += bufferSize;

        dataL[i] = delayL[readPos];
        if (dataR != nullptr)
            dataR[i] = delayR[readPos];

        ++writePos;
        if (writePos >= bufferSize)
            writePos = 0;
    }

    fixedLatencyWritePos = writePos;
}

double AudioEngine::DSPCore::musicalSoftClip(double x, double threshold, double knee, double asymmetry) noexcept
{
    return musicalSoftClipScalar(x, threshold, knee, asymmetry);
}

void AudioEngine::DSPCore::processOutput(const juce::AudioSourceChannelInfo& bufferToFill,
                                         int numSamples,
                                         const ProcessingState& state) noexcept
{
    auto* buffer = bufferToFill.buffer;
    const int startSample = bufferToFill.startSample;
    constexpr double kOutputHeadroom = 0.8912509381337456; // -1.0dB

    // ビット深度に基づくディザリング判定
    // ユーザー設定に従い、32-bit (float/int) でもディザリングを適用する。
    const bool applyDither = (ditherBitDepth > 0);
    const int numChannels = std::min(2, buffer->getNumChannels());

    // ── ループフュージョンによる最適化 ──
    // DC除去、ディザリング、変換・クランプを1つのループに統合し、キャッシュ効率を最大化する。
    // 各処理の結果をレジスタ内で次の処理に渡すことで、メモリへの書き戻しを削減する。

    double* dataL = (numChannels > 0) ? alignedL.get() : nullptr;
    double* dataR = (numChannels > 1) ? alignedR.get() : nullptr;
    float* dstL = (numChannels > 0) ? buffer->getWritePointer(0, startSample) : nullptr;
    float* dstR = (numChannels > 1) ? buffer->getWritePointer(1, startSample) : nullptr;

    // ── Step 1: DC除去 (ブロックモード・AVX2最適化) ──────────────────────────
    // process() は内部で m_prev_x/m_prev_y を直接読み書きするため
    // loadState()/saveState() は不要。NaN/Inf・デノーマルも内部でゼロ化する。
    dcBlockerL.process(dataL, numSamples);
    if (dataR) dcBlockerR.process(dataR, numSamples);

    // ── Step 2: NaN/Inf サニタイズ (NSフィルタ状態保護) ─────────────────────
    // DC除去後に残存する NaN/Inf をAVX2でゼロ化（libm完全排除）。
    {
        const __m256d vInf = _mm256_set1_pd(1.0e300);
        int i = 0;
        const int vEnd = numSamples / 4 * 4;
        for (; i < vEnd; i += 4)
        {
            __m256d vL = _mm256_loadu_pd(dataL + i);
            __m256d nanMaskL = _mm256_cmp_pd(vL, vL, _CMP_ORD_Q); // NaN以外=true
            __m256d infMaskL = _mm256_cmp_pd(_mm256_andnot_pd(_mm256_set1_pd(-0.0), vL), vInf, _CMP_LT_OQ);
            __m256d maskL = _mm256_and_pd(nanMaskL, infMaskL);
            vL = _mm256_and_pd(vL, maskL); // NaN/Inf → 0
            _mm256_storeu_pd(dataL + i, vL);

            if (dataR)
            {
                __m256d vR = _mm256_loadu_pd(dataR + i);
                __m256d nanMaskR = _mm256_cmp_pd(vR, vR, _CMP_ORD_Q);
                __m256d infMaskR = _mm256_cmp_pd(_mm256_andnot_pd(_mm256_set1_pd(-0.0), vR), vInf, _CMP_LT_OQ);
                __m256d maskR = _mm256_and_pd(nanMaskR, infMaskR);
                vR = _mm256_and_pd(vR, maskR);
                _mm256_storeu_pd(dataR + i, vR);
            }
        }
        for (; i < numSamples; ++i)
        {
            double v = dataL[i];
            if (!isFiniteAndAbsBelowNoLibm(v, 1.0e300)) v = 0.0; // 最終fallback（極稀）
            dataL[i] = v;
            if (dataR)
            {
                v = dataR[i];
                if (!isFiniteAndAbsBelowNoLibm(v, 1.0e300)) v = 0.0;
                dataR[i] = v;
            }
        }
    }

    pushAdaptiveCaptureBlocks(state.adaptiveCaptureQueue,
                              dataL,
                              dataR,
                              numSamples,
                              state.adaptiveCaptureSampleRateHz,
                              state.adaptiveCaptureBitDepth,
                              state.adaptiveCoeffBankIndex,
                              state.captureSessionId);

    // ── Step 3: ディザリング / ヘッドルーム適用 ─────────────────────────────
    if (noiseShaperType == NoiseShaperType::Adaptive9thOrder
        && state.adaptiveCoeffSet != nullptr
        && (activeAdaptiveCoeffBankIndex != state.adaptiveCoeffBankIndex
            || activeAdaptiveCoeffGeneration != state.adaptiveCoeffGeneration))
    {
        adaptiveNoiseShaper.applyMatchedCoefficients(state.adaptiveCoeffSet->k, kAdaptiveNoiseShaperOrder);
        activeAdaptiveCoeffBankIndex = state.adaptiveCoeffBankIndex;
        activeAdaptiveCoeffGeneration = state.adaptiveCoeffGeneration;
    }

    if (applyDither)
    {
        if (noiseShaperType == NoiseShaperType::Fixed4Tap)
        {
            fixedNoiseShaper.processStereoBlock(dataL, dataR, numSamples, kOutputHeadroom);
        }
        else if (noiseShaperType == NoiseShaperType::Fixed15Tap)
        {
            fixed15TapNoiseShaper.processStereoBlock(dataL, dataR, numSamples, kOutputHeadroom);
        }
        else if (noiseShaperType == NoiseShaperType::Adaptive9thOrder)
        {
            adaptiveNoiseShaper.processStereoBlock(dataL, dataR, numSamples, kOutputHeadroom);
        }
        else
        {
            dither.processStereoBlock(dataL, dataR, numSamples, kOutputHeadroom);
        }
    }
    else
    {
        // ヘッドルームのみ適用 (コンパイラが AVX2 自動ベクトル化可能な独立ループ)
        for (int i = 0; i < numSamples; ++i) dataL[i] *= kOutputHeadroom;
        if (dataR) for (int i = 0; i < numSamples; ++i) dataR[i] *= kOutputHeadroom;
    }

    // ── Step 3.5: ディザリング後・出力直前のNaN/Infサニタイズ ─────────────
    {
        const __m256d vInf = _mm256_set1_pd(1.0e300);
        int i = 0;
        const int vEnd = numSamples / 4 * 4;
        for (; i < vEnd; i += 4)
        {
            __m256d vL = _mm256_loadu_pd(dataL + i);
            __m256d nanMaskL = _mm256_cmp_pd(vL, vL, _CMP_ORD_Q);
            __m256d infMaskL = _mm256_cmp_pd(_mm256_andnot_pd(_mm256_set1_pd(-0.0), vL), vInf, _CMP_LT_OQ);
            __m256d maskL = _mm256_and_pd(nanMaskL, infMaskL);
            vL = _mm256_and_pd(vL, maskL);
            _mm256_storeu_pd(dataL + i, vL);

            if (dataR)
            {
                __m256d vR = _mm256_loadu_pd(dataR + i);
                __m256d nanMaskR = _mm256_cmp_pd(vR, vR, _CMP_ORD_Q);
                __m256d infMaskR = _mm256_cmp_pd(_mm256_andnot_pd(_mm256_set1_pd(-0.0), vR), vInf, _CMP_LT_OQ);
                __m256d maskR = _mm256_and_pd(nanMaskR, infMaskR);
                vR = _mm256_and_pd(vR, maskR);
                _mm256_storeu_pd(dataR + i, vR);
            }
        }
        for (; i < numSamples; ++i)
        {
            double v = dataL[i];
            if (!isFiniteAndAbsBelowNoLibm(v, 1.0e300)) v = 0.0;
            dataL[i] = v;
            if (dataR)
            {
                v = dataR[i];
                if (!isFiniteAndAbsBelowNoLibm(v, 1.0e300)) v = 0.0;
                dataR[i] = v;
            }
        }
    }

    // パススルーDSP等に固定レイテンシを付与する。
    applyFixedLatencyDelay(dataL, dataR, numSamples);

    // ── Step 4: double→float 変換・クランプ ─────────────────────────────────
    // juce::jlimit<double>(-1.0, 1.0, x) は JUCE 8.0.12 で存在確認済み (template)。
    // Step 2 のサニタイズ後は dataL/dataR に NaN/Inf がないため、
    // ここでの追加 isfinite チェックは不要。
    for (int i = 0; i < numSamples; ++i)
           dstL[i] = static_cast<float>(juce::jlimit(-kOutputHeadroom, kOutputHeadroom, dataL[i]));
    if (dstR)
        for (int i = 0; i < numSamples; ++i)
              dstR[i] = static_cast<float>(juce::jlimit(-kOutputHeadroom, kOutputHeadroom, dataR[i]));

    // 3ch以降は使用しないためクリア (ゴミデータ出力防止)
    for (int ch = numChannels; ch < buffer->getNumChannels(); ++ch)
        buffer->clear(ch, startSample, numSamples);
}

void AudioEngine::DSPCore::processOutputDouble(juce::AudioBuffer<double>& buffer,
                                               int numSamples,
                                               const ProcessingState& state) noexcept
{
    constexpr double kOutputHeadroom = 0.8912509381337456; // -1.0dB
    const bool applyDither = (ditherBitDepth > 0);
    const int numChannels = std::min(2, buffer.getNumChannels());
    double* dataL = (numChannels > 0) ? alignedL.get() : nullptr;
    double* dataR = (numChannels > 1) ? alignedR.get() : nullptr;

    dcBlockerL.processStereo(dataL, dataR, numSamples, dcBlockerR);

    {
        const __m256d vInf = _mm256_set1_pd(1.0e300);
        int i = 0;
        const int vEnd = numSamples / 4 * 4;
        for (; i < vEnd; i += 4)
        {
            __m256d vL = _mm256_loadu_pd(dataL + i);
            __m256d nanMaskL = _mm256_cmp_pd(vL, vL, _CMP_ORD_Q);
            __m256d infMaskL = _mm256_cmp_pd(_mm256_andnot_pd(_mm256_set1_pd(-0.0), vL), vInf, _CMP_LT_OQ);
            _mm256_storeu_pd(dataL + i, _mm256_and_pd(vL, _mm256_and_pd(nanMaskL, infMaskL)));

            if (dataR != nullptr)
            {
                __m256d vR = _mm256_loadu_pd(dataR + i);
                __m256d nanMaskR = _mm256_cmp_pd(vR, vR, _CMP_ORD_Q);
                __m256d infMaskR = _mm256_cmp_pd(_mm256_andnot_pd(_mm256_set1_pd(-0.0), vR), vInf, _CMP_LT_OQ);
                _mm256_storeu_pd(dataR + i, _mm256_and_pd(vR, _mm256_and_pd(nanMaskR, infMaskR)));
            }
        }

        for (; i < numSamples; ++i)
        {
            if (!isFiniteAndAbsBelowNoLibm(dataL[i], 1.0e300))
                dataL[i] = 0.0;

            if (dataR != nullptr && !isFiniteAndAbsBelowNoLibm(dataR[i], 1.0e300))
                dataR[i] = 0.0;
        }
    }

    pushAdaptiveCaptureBlocks(state.adaptiveCaptureQueue,
                              dataL,
                              dataR,
                              numSamples,
                              state.adaptiveCaptureSampleRateHz,
                              state.adaptiveCaptureBitDepth,
                              state.adaptiveCoeffBankIndex,
                              state.captureSessionId);

    if (noiseShaperType == NoiseShaperType::Adaptive9thOrder
        && state.adaptiveCoeffSet != nullptr
        && (activeAdaptiveCoeffBankIndex != state.adaptiveCoeffBankIndex
            || activeAdaptiveCoeffGeneration != state.adaptiveCoeffGeneration))
    {
        adaptiveNoiseShaper.applyMatchedCoefficients(state.adaptiveCoeffSet->k, kAdaptiveNoiseShaperOrder);
        activeAdaptiveCoeffBankIndex = state.adaptiveCoeffBankIndex;
        activeAdaptiveCoeffGeneration = state.adaptiveCoeffGeneration;
    }

    if (applyDither)
    {
        if (noiseShaperType == NoiseShaperType::Fixed4Tap)
            fixedNoiseShaper.processStereoBlock(dataL, dataR, numSamples, kOutputHeadroom);
        else if (noiseShaperType == NoiseShaperType::Fixed15Tap)
            fixed15TapNoiseShaper.processStereoBlock(dataL, dataR, numSamples, kOutputHeadroom);
        else if (noiseShaperType == NoiseShaperType::Adaptive9thOrder)
            adaptiveNoiseShaper.processStereoBlock(dataL, dataR, numSamples, kOutputHeadroom);
        else
            dither.processStereoBlock(dataL, dataR, numSamples, kOutputHeadroom);
    }
    else
    {
        for (int i = 0; i < numSamples; ++i)
        {
            dataL[i] *= kOutputHeadroom;
            if (dataR != nullptr)
                dataR[i] *= kOutputHeadroom;
        }
    }

    // ── Step 3.5: ディザリング後・出力直前のNaN/Infサニタイズ ─────────────
    {
        const __m256d vInf = _mm256_set1_pd(1.0e300);
        int i = 0;
        const int vEnd = numSamples / 4 * 4;
        for (; i < vEnd; i += 4)
        {
            __m256d vL = _mm256_loadu_pd(dataL + i);
            __m256d nanMaskL = _mm256_cmp_pd(vL, vL, _CMP_ORD_Q);
            __m256d infMaskL = _mm256_cmp_pd(_mm256_andnot_pd(_mm256_set1_pd(-0.0), vL), vInf, _CMP_LT_OQ);
            _mm256_storeu_pd(dataL + i, _mm256_and_pd(vL, _mm256_and_pd(nanMaskL, infMaskL)));

            if (dataR != nullptr)
            {
                __m256d vR = _mm256_loadu_pd(dataR + i);
                __m256d nanMaskR = _mm256_cmp_pd(vR, vR, _CMP_ORD_Q);
                __m256d infMaskR = _mm256_cmp_pd(_mm256_andnot_pd(_mm256_set1_pd(-0.0), vR), vInf, _CMP_LT_OQ);
                _mm256_storeu_pd(dataR + i, _mm256_and_pd(vR, _mm256_and_pd(nanMaskR, infMaskR)));
            }
        }

        for (; i < numSamples; ++i)
        {
            if (!isFiniteAndAbsBelowNoLibm(dataL[i], 1.0e300))
                dataL[i] = 0.0;

            if (dataR != nullptr && !isFiniteAndAbsBelowNoLibm(dataR[i], 1.0e300))
                dataR[i] = 0.0;
        }
    }

    {
        const __m256d vLimit = _mm256_set1_pd(kOutputHeadroom);
        const __m256d vNegLimit = _mm256_set1_pd(-kOutputHeadroom);
        int i = 0;
        const int vEnd = (numSamples / 4) * 4;
        for (; i < vEnd; i += 4)
        {
            __m256d vL = _mm256_loadu_pd(dataL + i);
            vL = _mm256_min_pd(_mm256_max_pd(vL, vNegLimit), vLimit);
            _mm256_storeu_pd(dataL + i, vL);

            if (dataR != nullptr)
            {
                __m256d vR = _mm256_loadu_pd(dataR + i);
                vR = _mm256_min_pd(_mm256_max_pd(vR, vNegLimit), vLimit);
                _mm256_storeu_pd(dataR + i, vR);
            }
        }

        for (; i < numSamples; ++i)
        {
            dataL[i] = juce::jlimit(-kOutputHeadroom, kOutputHeadroom, dataL[i]);
            if (dataR != nullptr)
                dataR[i] = juce::jlimit(-kOutputHeadroom, kOutputHeadroom, dataR[i]);
        }
    }

    // パススルーDSP等に固定レイテンシを付与する。
    applyFixedLatencyDelay(dataL, dataR, numSamples);

    juce::FloatVectorOperations::copy(buffer.getWritePointer(0, 0), dataL, numSamples);
    if (numChannels > 1 && dataR != nullptr)
        juce::FloatVectorOperations::copy(buffer.getWritePointer(1, 0), dataR, numSamples);

    for (int channel = numChannels; channel < buffer.getNumChannels(); ++channel)
        buffer.clear(channel, 0, numSamples);
}

#endif
