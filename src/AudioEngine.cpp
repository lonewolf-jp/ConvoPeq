#include <immintrin.h> // AVX2

//====================================================
// AVX2 クロスフェード（double）
//====================================================
static inline void crossfadeAVX2(
    double* dstL, double* dstR,
    const double* newL, const double* newR,
    const double* oldL, const double* oldR,
    int numSamples,
    double gStart,
    double gStep)
{
    int i = 0;
    for (; i + 3 < numSamples; i += 4)
    {
        __m256d g = _mm256_set_pd(
            gStart + gStep * (i + 3),
            gStart + gStep * (i + 2),
            gStart + gStep * (i + 1),
            gStart + gStep * (i + 0)
        );
        __m256d gOld = _mm256_sub_pd(_mm256_set1_pd(1.0), g);
        __m256d nL = _mm256_loadu_pd(newL + i);
        __m256d oL = _mm256_loadu_pd(oldL + i);
        __m256d nR = _mm256_loadu_pd(newR + i);
        __m256d oR = _mm256_loadu_pd(oldR + i);
        __m256d outL = _mm256_add_pd(_mm256_mul_pd(nL, g), _mm256_mul_pd(oL, gOld));
        __m256d outR = _mm256_add_pd(_mm256_mul_pd(nR, g), _mm256_mul_pd(oR, gOld));
        _mm256_storeu_pd(dstL + i, outL);
        _mm256_storeu_pd(dstR + i, outR);
    }
    for (; i < numSamples; ++i)
    {
        double g = gStart + gStep * i;
        double gOld = 1.0 - g;
        dstL[i] = newL[i] * g + oldL[i] * gOld;
        dstR[i] = newR[i] * g + oldR[i] * gOld;
    }
}
//============================================================================
// AudioEngine.cpp  ── v0.2 (JUCE 8.0.12対応)
// AudioEngineの実装
//============================================================================

#include <JuceHeader.h>
#include "AudioEngine.h"
#include "InputBitDepthTransform.h"
#include "OutputFilter.h"

extern std::atomic<bool> gShuttingDown;

// fastTanh 高精度 Padé 近似用の定数
//----------------------------------------------------------------------------
namespace TanhApprox {
    // Padé [5/4] 近似係数
    constexpr double NUM_A = 10395.0;
    constexpr double NUM_B = 1260.0;
    constexpr double NUM_C = 21.0;
    constexpr double DEN_A = 10395.0;
    constexpr double DEN_B = 4725.0;
    constexpr double DEN_C = 210.0;

    // 近似有効範囲
    constexpr double CLIP_THRESHOLD = 4.5;
}

// ダブルバッファモデル用：DSPCore 削除ヘルパー（単純な delete）
static void deleteDSP(AudioEngine::DSPCore* dsp)
{
    if (dsp) delete dsp;
}

// ダブルバッファモデル用：EQCoeffCache 削除ヘルパー
static void deleteEQCache(EQCoeffCache* cache)
{
    if (cache) delete cache;
}

static void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}

//==============================================================================
// 等電力クロスフェード用近似関数（Audio Thread安全・libm不使用）
//==============================================================================
static inline float equalPowerSinFloat(float x) noexcept
{
    x = juce::jlimit(0.0f, 1.0f, x);
    const float t = x * 1.5707963267948966f;  // π/2
    const float t2 = t * t;
    return t * (1.0f + t2 * (-1.0f/6.0f + t2 * (1.0f/120.0f
             + t2 * (-1.0f/5040.0f + t2 * (1.0f/362880.0f)))));
}

static inline double equalPowerSinDouble(double x) noexcept
{
    x = juce::jlimit(0.0, 1.0, x);
    const double t = x * 1.5707963267948966;
    const double t2 = t * t;
    return t * (1.0 + t2 * (-1.0/6.0 + t2 * (1.0/120.0
             + t2 * (-1.0/5040.0 + t2 * (1.0/362880.0)))));
}

//==============================================================================

// =============================================================
// Rebuild request coalescing (Stage 3)
// =============================================================
void AudioEngine::requestRebuild(convo::RebuildKind kind) noexcept
{
    if (shutdownInProgress.load(std::memory_order_acquire) ||
        gShuttingDown.load(std::memory_order_acquire))
        return;

    if (kind == convo::RebuildKind::None)
        return;

    if (kind == convo::RebuildKind::IRContent)
    {
        if (!uiConvolverProcessor.isIRFinalized())
            return;

        const int64_t nowTicks = juce::Time::getHighResolutionTicks();
        const int64_t lastTicks = lastIRContentRebuildTicks_.load(std::memory_order_relaxed);
        const int64_t minDelta = juce::Time::getHighResolutionTicksPerSecond() / 5; // 200ms

        if (lastTicks > 0 && (nowTicks - lastTicks) < minDelta)
            return;

        lastIRContentRebuildTicks_.store(nowTicks, std::memory_order_relaxed);
    }

    const uint32_t mask = convo::toMask(kind);
    const uint32_t prev = pendingRebuildMask_.fetch_or(mask, std::memory_order_acq_rel);

    if (prev == 0)
        triggerAsyncUpdate();
}

void AudioEngine::handleAsyncUpdate()
{
    if (shutdownInProgress.load(std::memory_order_acquire) ||
        gShuttingDown.load(std::memory_order_acquire))
        return;

    executeCommit();
    processRebuildRequestsInternal();
}

void AudioEngine::prepareCommit(DSPCore* newDSP, int generation)
{
    if (newDSP == nullptr)
        return;

    if (shutdownInProgress.load(std::memory_order_acquire) ||
        gShuttingDown.load(std::memory_order_acquire))
    {
        deleteDSP(newDSP);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(deferredCommitMutex);

        if (shutdownInProgress.load(std::memory_order_acquire) ||
            gShuttingDown.load(std::memory_order_acquire))
        {
            deleteDSP(newDSP);
            return;
        }

        deferredCommitQueue.push(CommitStaging { newDSP, nullptr, generation });
    }

    triggerAsyncUpdate();
}

void AudioEngine::executeCommit()
{
    std::queue<CommitStaging> localQueue;

    {
        std::lock_guard<std::mutex> lock(deferredCommitMutex);
        std::swap(localQueue, deferredCommitQueue);
    }

    while (!localQueue.empty())
    {
        auto staging = localQueue.front();
        localQueue.pop();

        if (staging.newDSP == nullptr)
            continue;

        if (shutdownInProgress.load(std::memory_order_acquire) ||
            gShuttingDown.load(std::memory_order_acquire))
        {
            deleteDSP(staging.newDSP);
            continue;
        }

        commitNewDSP(staging.newDSP, staging.generation);
    }
}

void AudioEngine::processRebuildRequestsInternal()
{
    // 1. mask 取得（完全 drain）
    const uint32_t mask = pendingRebuildMask_.exchange(0, std::memory_order_acq_rel);
    if (mask == 0)
        return;

    // 2. 現在の DSP パラメータ取得
    const double sr = currentSampleRate.load(std::memory_order_acquire);
    const int bs = maxSamplesPerBlock.load(std::memory_order_acquire);

    // 3. 無効状態 → 再投入（重要）
    if (sr <= 0.0 || bs <= 0)
    {
        pendingRebuildMask_.fetch_or(mask, std::memory_order_release);
        return;
    }

    // 4. 優先度制御
    // =============================

    // --- HIGH: Structural ---
    if (mask & static_cast<uint32_t>(convo::RebuildKind::Structural))
    {
        requestRebuild(sr, bs);
        return; // 他は defer
    }

    // --- MID: IRContent ---
    if (mask & static_cast<uint32_t>(convo::RebuildKind::IRContent))
    {
        requestRebuild(sr, bs);
        return;
    }

    // --- LOW: Runtime / UIOnly ---
    // 現状は何もしない（将来拡張ポイント）
}


// ==================================================================
// 段階 2+3：thread_local スロット管理および定数
// ==================================================================
static constexpr size_t INVALID_SLOT = SIZE_MAX;

// ==================================================================
// B19: requestLoadState の例外安全性確保用 RAII ガード
// ==================================================================
class RestoreStateGuard {
public:
    explicit RestoreStateGuard(bool& flag) noexcept : m_flag(flag) {
        m_flag = true;
    }
    ~RestoreStateGuard() noexcept {
        m_flag = false;
    }
    RestoreStateGuard(const RestoreStateGuard&) = delete;
    RestoreStateGuard& operator=(const RestoreStateGuard&) = delete;
private:
    bool& m_flag;
};

// グローバルインスタンスの定義
std::atomic<bool> gShuttingDown{false};

template <typename T>
static inline T* sanitizeRawPtr(T* ptr) noexcept
{
    constexpr uintptr_t kInvalidAllOnes = ~static_cast<uintptr_t>(0);
    return (reinterpret_cast<uintptr_t>(ptr) == kInvalidAllOnes) ? nullptr : ptr;
}

// ==================================================================
// 以下、既存の AudioEngine 実装
// ==================================================================

namespace
{
    static constexpr std::array<double, convo::FixedNoiseShaper::ORDER> kFixedNoiseShaperTunedCoeffs
    {
        0.46, 0.28, 0.17, 0.09
    };

    static constexpr std::array<double, convo::Fixed15TapNoiseShaper::ORDER> kFixed15TapNoiseShaperTunedCoeffs
    {
        // 15th-order noise shaper coefficients (psychoacoustically optimized)
        2.033, -2.165, 1.959, -1.590, 1.221, -0.886, 0.604, -0.389, 0.235, -0.132, 0.068, -0.031, 0.012, -0.004, 0.001, 0.0
    };

    static constexpr std::array<double, kAdaptiveNoiseShaperOrder> kDefaultAdaptiveNoiseShaperCoeffs
    {
        0.82, -0.68, 0.55, -0.43, 0.33, -0.25, 0.18, -0.12, 0.07
    };

    static constexpr std::array<double, kAdaptiveNoiseShaperSampleRateBankCount> kAdaptiveSupportedSampleRatesHz
    {
        44100.0, 48000.0, 88200.0, 96000.0, 176400.0,
        192000.0, 352800.0, 384000.0, 705600.0, 768000.0
    };

    inline double absNoLibm(double x) noexcept
    {
        union { double d; uint64_t u; } v { x };
        v.u &= 0x7FFFFFFFFFFFFFFFULL;
        return v.d;
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

    inline double absDiffNoLibm(double a, double b) noexcept
    {
        return absNoLibm(a - b);
    }

    inline int clampAdaptiveBankIndex(int bankIndex) noexcept
    {
        if (bankIndex < 0)
            return 0;

        if (bankIndex >= kAdaptiveNoiseShaperSampleRateBankCount)
            return kAdaptiveNoiseShaperSampleRateBankCount - 1;

        return bankIndex;
    }

    inline juce::String makeAdaptiveCoeffPropertyName(double sampleRate, int coeffIndex)
    {
        return "adaptiveCoeff_" + juce::String(static_cast<int>(sampleRate + 0.5)) + "_" + juce::String(coeffIndex);
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

    inline bool isAligned64(const void* ptr) noexcept
    {
        return (reinterpret_cast<std::uintptr_t>(ptr) & static_cast<std::uintptr_t>(63)) == 0;
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

    // AVX2 helper to calculate magnitude squared for a biquad over an array of complex frequencies
    static void calcMagnitudesForBand(const EQCoeffsBiquad& c,
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

// Padé近似による高速tanh (std::exp回避)
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

// Helper for soft clipping (Scalar version)
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

// prevSampleInOut: 前ブロック末尾のクリップ済み出力サンプル（ブロック間インターサンプルピーク検出用）
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

// コンストラクタ
//--------------------------------------------------------------

AudioEngine::AudioEngine()
    : uiEqEditor(*this)
    , m_workerThread(m_commandBuffer, *this, m_generationManager, &affinityManager)
{
    gShuttingDown.store(false, std::memory_order_release);
    uiConvolverProcessor.setRcuProvider(this);
    // 必要な初期化処理があればここに追加
}


AudioEngine::~AudioEngine()
{
    diagLog("[DIAG] ~AudioEngine: enter");
    shutdownInProgress.store(true, std::memory_order_release);
    gShuttingDown.store(true, std::memory_order_release);
    cancelPendingUpdate();

    // 終了順序を固定化して、終了時フリーズを防ぐ。
    stopTimer();

    // まず rebuild thread 側へ終了を通知し、pending task を破棄して
    // 終了時に重い再構築へ入る経路を閉じる。
    // pending task を破棄して進行中 rebuild を obsolete にし、thread を停止する。
    DSPCore* activeToRelease = nullptr;
    DSPCore* fadingToRelease = nullptr;
    DSPCore* queuedToRelease = nullptr;
    {
        std::lock_guard<std::mutex> lock(rebuildMutex);
        rebuildGeneration.fetch_add(1, std::memory_order_relaxed);

        // Audio Thread から参照される公開ポインタを明示的に外す。
        currentDSP.store(nullptr, std::memory_order_release);

        activeToRelease = sanitizeRawPtr(activeDSP);
        activeDSP = nullptr;
        fadingToRelease = sanitizeRawPtr(fadingOutDSP.exchange(nullptr, std::memory_order_acq_rel));
        queuedToRelease = sanitizeRawPtr(queuedOldDSP.exchange(nullptr, std::memory_order_acq_rel));
        fadeQueued.store(false, std::memory_order_release);

        if (hasPendingTask)
        {
            if (pendingTask.newDSP)
            {
                deleteDSP(pendingTask.newDSP);
                pendingTask.newDSP = nullptr;
            }

            if (pendingTask.currentDSP)
            {
                deleteDSP(pendingTask.currentDSP);
                pendingTask.currentDSP = nullptr;
            }

            hasPendingTask = false;
        }
    }
    stopRebuildThread();

    {
        std::queue<CommitStaging> abandonedCommits;
        std::lock_guard<std::mutex> lock(deferredCommitMutex);
        std::swap(abandonedCommits, deferredCommitQueue);

        while (!abandonedCommits.empty())
        {
            auto staging = abandonedCommits.front();
            abandonedCommits.pop();

            if (staging.newDSP)
                deleteDSP(staging.newDSP);
            if (staging.oldDSP)
                deleteDSP(staging.oldDSP);
        }
    }

    if (activeToRelease) deleteDSP(activeToRelease);
    if (fadingToRelease) deleteDSP(fadingToRelease);
    if (queuedToRelease) deleteDSP(queuedToRelease);

    uiConvolverProcessor.removeChangeListener(this);
    uiEqEditor.removeChangeListener(this);
    uiConvolverProcessor.removeListener(this);

    // Snapshot worker を先に停止。
    shutdownWorkerThread();

    // RCU v17.15: ReclaimerThread が全ての解放を担当するため、
    // ここでの明示的な回収処理は不要。

    // ...既存の解放処理...
    if (latencyBufOldL) { _aligned_free(latencyBufOldL); latencyBufOldL = nullptr; }
    if (latencyBufOldR) { _aligned_free(latencyBufOldR); latencyBufOldR = nullptr; }
    if (latencyBufNewL) { _aligned_free(latencyBufNewL); latencyBufNewL = nullptr; }
    if (latencyBufNewR) { _aligned_free(latencyBufNewR); latencyBufNewR = nullptr; }
    latencyBufSize = 0;
    diagLog("[DIAG] ~AudioEngine: exit");
}

// 以降の初期化コードはAudioEngineコンストラクタ本体へ移動すること



void AudioEngine::startNoiseShaperLearning(NoiseShaperLearner::LearningMode mode, bool resume)
{
    if (noiseShaperLearner == nullptr)
        return;

    pendingLearningMode.store(mode, std::memory_order_release);
    selectAdaptiveCoeffBankForCurrentSettings();

    if (noiseShaperType.load(std::memory_order_acquire) != NoiseShaperType::Adaptive9thOrder)
        setNoiseShaperType(NoiseShaperType::Adaptive9thOrder);

    const LearningCommand cmd {
        LearningCommand::Type::Start,
        resume,
        mode,
        pendingIRGeneration
    };

    if (!enqueueLearningCommand(cmd))
    {
        DBG("[AudioEngine] startNoiseShaperLearning: command queue overflow");
        return;
    }
}

void AudioEngine::stopNoiseShaperLearning()
{
    const LearningCommand cmd {
        LearningCommand::Type::Stop,
        false,
        pendingLearningMode.load(std::memory_order_acquire),
        pendingIRGeneration
    };

    if (!enqueueLearningCommand(cmd))
    {
        DBG("[AudioEngine] stopNoiseShaperLearning: command queue overflow");
    }

    if (noiseShaperLearner)
        noiseShaperLearner->stopLearning();
}

void AudioEngine::setNoiseShaperLearningMode(NoiseShaperLearner::LearningMode mode)
{
    pendingLearningMode.store(mode, std::memory_order_release);
    selectAdaptiveCoeffBankForCurrentSettings();
    if (noiseShaperLearner)
        noiseShaperLearner->setLearningMode(mode);
}

bool AudioEngine::isNoiseShaperLearning() const
{
    return noiseShaperLearner && noiseShaperLearner->isRunning();
}

const NoiseShaperLearner::Progress& AudioEngine::getNoiseShaperLearningProgress() const
{
    jassert(noiseShaperLearner);
    return noiseShaperLearner->getProgress();
}

NoiseShaperLearner::Settings AudioEngine::getNoiseShaperLearnerSettings() const
{
    if (noiseShaperLearner)
        return noiseShaperLearner->getSettings();
    return {};
}

void AudioEngine::setNoiseShaperLearnerSettings(const NoiseShaperLearner::Settings& settings)
{
    if (noiseShaperLearner)
        noiseShaperLearner->setSettings(settings);
}

int AudioEngine::copyNoiseShaperLearningHistory(double* outScores, int maxPoints) const noexcept
{
    return noiseShaperLearner ? noiseShaperLearner->copyBestScoreHistory(outScores, maxPoints) : 0;
}

const char* AudioEngine::getNoiseShaperLearningError() const noexcept
{
    if (noiseShaperLearner == nullptr)
        return nullptr;
    return noiseShaperLearner->getErrorMessage();
}

int AudioEngine::getAdaptiveSampleRateBankCount() noexcept
{
    return kAdaptiveNoiseShaperSampleRateBankCount;
}

double AudioEngine::getAdaptiveSampleRateBankHz(int bankIndex) noexcept
{
    return kAdaptiveSupportedSampleRatesHz[static_cast<size_t>(clampAdaptiveBankIndex(bankIndex))];
}

void AudioEngine::initialiseAdaptiveCoeffBanks() noexcept
{
    for (int srBank = 0; srBank < kAdaptiveNoiseShaperSampleRateBankCount; ++srBank)
    {
        double sr = getAdaptiveSampleRateBankHz(srBank);
        for (int bdIdx = 0; bdIdx < kAdaptiveBitDepthCount; ++bdIdx)
        {
            for (int modeIdx = 0; modeIdx < kLearningModeCount; ++modeIdx)
            {
                int bankIndex = (srBank * kAdaptiveBitDepthCount + bdIdx) * kLearningModeCount + modeIdx;
                auto& bank = adaptiveCoeffBanks[static_cast<size_t>(bankIndex)];
                bank.sampleRateHz = sr;

                for (int coeffIndex = 0; coeffIndex < kAdaptiveNoiseShaperOrder; ++coeffIndex)
                {
                    const double coefficient = kDefaultAdaptiveNoiseShaperCoeffs[static_cast<size_t>(coeffIndex)];
                    bank.coeffSetA.k[coeffIndex] = coefficient;
                    bank.coeffSetB.k[coeffIndex] = coefficient;
                }

                bank.activeIndex.store(0, std::memory_order_relaxed);
                bank.generation.store(1u, std::memory_order_relaxed);
                bank.writeLock.store(false, std::memory_order_relaxed);
            }
        }
    }
}

int AudioEngine::resolveAdaptiveCoeffBankIndex(double sampleRate) noexcept
{
    int bestIndex = 0;
    double bestDistance = std::numeric_limits<double>::max();

    for (int bankIndex = 0; bankIndex < kAdaptiveNoiseShaperSampleRateBankCount; ++bankIndex)
    {
        const double distance = std::abs(sampleRate - getAdaptiveSampleRateBankHz(bankIndex));
        if (distance < bestDistance)
        {
            bestDistance = distance;
            bestIndex = bankIndex;
        }
    }

    return bestIndex;
}

int AudioEngine::getAdaptiveBitDepthIndex(int bitDepth) noexcept
{
    if (bitDepth <= 16) return 0;
    if (bitDepth <= 24) return 1;
    return 2;
}

int AudioEngine::getAdaptiveCoeffBankIndex(double sampleRate, int bitDepth, NoiseShaperLearner::LearningMode mode) noexcept
{
    const int srBank = resolveAdaptiveCoeffBankIndex(sampleRate);
    const int bdIdx  = getAdaptiveBitDepthIndex(bitDepth);
    const int modeIdx = static_cast<int>(mode);
    return (srBank * kAdaptiveBitDepthCount + bdIdx) * kLearningModeCount + modeIdx;
}

AudioEngine::AdaptiveCoeffBankSlot& AudioEngine::getAdaptiveCoeffBankForIndex(int bankIndex) noexcept
{
    if (bankIndex < 0) bankIndex = 0;
    if (bankIndex >= kAdaptiveNoiseShaperSampleRateBankCount * kAdaptiveBitDepthCount * kLearningModeCount) bankIndex = kAdaptiveNoiseShaperSampleRateBankCount * kAdaptiveBitDepthCount * kLearningModeCount - 1;
    return adaptiveCoeffBanks[static_cast<size_t>(bankIndex)];
}

const AudioEngine::AdaptiveCoeffBankSlot& AudioEngine::getAdaptiveCoeffBankForIndex(int bankIndex) const noexcept
{
    if (bankIndex < 0) bankIndex = 0;
    if (bankIndex >= kAdaptiveNoiseShaperSampleRateBankCount * kAdaptiveBitDepthCount * kLearningModeCount) bankIndex = kAdaptiveNoiseShaperSampleRateBankCount * kAdaptiveBitDepthCount * kLearningModeCount - 1;
    return adaptiveCoeffBanks[static_cast<size_t>(bankIndex)];
}

void AudioEngine::selectAdaptiveCoeffBankForCurrentSettings() noexcept
{
    const double sr = currentSampleRate.load(std::memory_order_acquire);
    const int bd   = ditherBitDepth.load(std::memory_order_acquire);
    const auto mode = pendingLearningMode.load(std::memory_order_acquire);

    const int newBankIndex = getAdaptiveCoeffBankIndex(sr, bd, mode);

    if (newBankIndex != currentAdaptiveCoeffBankIndex.load(std::memory_order_acquire))
    {
        currentAdaptiveCoeffBankIndex.store(newBankIndex, std::memory_order_release);

        // 必要なら学習スレッドや表示側に通知（非同期で安全）
        if (noiseShaperLearner)
        {
            // callAsync などで UI 更新を依頼（Audio Thread からは直接呼ばない）
            juce::MessageManager::callAsync([this, newBankIndex]() {
                if (auto* learner = noiseShaperLearner.get())
                    learner->onCoeffBankChanged(newBankIndex);
            });
        }
    }
}

void AudioEngine::getCurrentAdaptiveCoefficients(double* outCoeffs, int maxCoefficients) const noexcept
{
    if (outCoeffs == nullptr || maxCoefficients <= 0)
        return;

    const auto& bank = getAdaptiveCoeffBankForIndex(
        currentAdaptiveCoeffBankIndex.load(std::memory_order_acquire));

    // RCU 世代検証（Message Thread なので失敗時はリトライ）
    for (int retry = 0; retry < 3; ++retry)
    {
        const uint32_t genBefore = bank.generation.load(std::memory_order_acquire);
        const auto* coeffSet = AudioEngine::getActiveCoeffSet(bank);
        const uint32_t genAfter = bank.generation.load(std::memory_order_acquire);

        if (genBefore == genAfter)
        {
            const int limit = std::min(kAdaptiveNoiseShaperOrder, maxCoefficients);
            for (int i = 0; i < limit; ++i)
                outCoeffs[i] = coeffSet->k[i];
            return;
        }
    }
}

void AudioEngine::setCurrentAdaptiveCoefficients(const double* coeffs, int numCoefficients)
{
    if (coeffs == nullptr || numCoefficients <= 0)
        return;

    // 学習中は UI からの係数更新を拒否（競合防止）
    if (isNoiseShaperLearning())
    {
        DBG_LOG("[AudioEngine] Coefficient update rejected during learning");
        return;
    }

    const int bankIndex = currentAdaptiveCoeffBankIndex.load(std::memory_order_acquire);
    double stagedCoefficients[kAdaptiveNoiseShaperOrder] = {};
    getCurrentAdaptiveCoefficients(stagedCoefficients, kAdaptiveNoiseShaperOrder);

    const int limit = std::min(kAdaptiveNoiseShaperOrder, numCoefficients);
    for (int i = 0; i < limit; ++i)
        stagedCoefficients[i] = coeffs[i];

    publishCoeffsToBank(bankIndex, stagedCoefficients);
}

void AudioEngine::getAdaptiveCoefficientsForSampleRate(double sampleRate, double* outCoeffs, int maxCoefficients) const noexcept
{
    if (outCoeffs == nullptr || maxCoefficients <= 0)
        return;

    const auto& bank = getAdaptiveCoeffBankForIndex(resolveAdaptiveCoeffBankIndex(sampleRate));

    // RCU 世代検証（Message Thread なので失敗時はリトライ）
    for (int retry = 0; retry < 3; ++retry)
    {
        const uint32_t genBefore = bank.generation.load(std::memory_order_acquire);
        const auto* coeffSet = AudioEngine::getActiveCoeffSet(bank);
        const uint32_t genAfter = bank.generation.load(std::memory_order_acquire);

        if (genBefore == genAfter)
        {
            const int limit = std::min(kAdaptiveNoiseShaperOrder, maxCoefficients);
            for (int i = 0; i < limit; ++i)
                outCoeffs[i] = coeffSet->k[i];
            return;
        }
    }
}

void AudioEngine::setAdaptiveCoefficientsForSampleRate(double sampleRate, const double* coeffs, int numCoefficients)
{
    if (coeffs == nullptr || numCoefficients <= 0)
        return;

    // 学習中は UI からの係数更新を拒否（競合防止）
    if (isNoiseShaperLearning())
    {
        DBG_LOG("[AudioEngine] Coefficient update rejected during learning");
        return;
    }

    const int bankIndex = resolveAdaptiveCoeffBankIndex(sampleRate);
    double stagedCoefficients[kAdaptiveNoiseShaperOrder] = {};
    getAdaptiveCoefficientsForSampleRate(sampleRate, stagedCoefficients, kAdaptiveNoiseShaperOrder);

    const int limit = std::min(kAdaptiveNoiseShaperOrder, numCoefficients);
    for (int i = 0; i < limit; ++i)
        stagedCoefficients[i] = coeffs[i];

    publishCoeffsToBank(bankIndex, stagedCoefficients);
}

void AudioEngine::getAdaptiveCoefficientsForSampleRateAndBitDepth(double sampleRate, int bitDepth, double* outCoeffs, int maxCoefficients) const noexcept
{
    if (outCoeffs == nullptr || maxCoefficients <= 0)
        return;

    const auto mode = pendingLearningMode.load(std::memory_order_acquire);
    const int bank = getAdaptiveCoeffBankIndex(sampleRate, bitDepth, mode);
    const auto& slot = getAdaptiveCoeffBankForIndex(bank);

    // RCU 世代検証（Message Thread なので失敗時はリトライ）
    for (int retry = 0; retry < 3; ++retry)
    {
        const uint32_t genBefore = slot.generation.load(std::memory_order_acquire);
        const CoeffSet* active = AudioEngine::getActiveCoeffSet(slot);
        const uint32_t genAfter = slot.generation.load(std::memory_order_acquire);

        if (genBefore == genAfter)
        {
            if (active)
            {
                const int copyCount = std::min(maxCoefficients, kAdaptiveNoiseShaperOrder);
                std::memcpy(outCoeffs, active->k, static_cast<size_t>(copyCount) * sizeof(double));
            }
            else
            {
                // デフォルト（初期値）
                std::memcpy(outCoeffs, kDefaultAdaptiveNoiseShaperCoeffs.data(), sizeof(kDefaultAdaptiveNoiseShaperCoeffs));
            }
            return;
        }
    }
}

void AudioEngine::setAdaptiveCoefficientsForSampleRateAndBitDepth(double sampleRate, int bitDepth, const double* coeffs, int numCoefficients)
{
    if (coeffs == nullptr || numCoefficients <= 0)
        return;

    // 学習中は UI からの係数更新を拒否（競合防止）
    if (isNoiseShaperLearning())
    {
        DBG_LOG("[AudioEngine] Coefficient update rejected during learning");
        return;
    }

    const auto mode = pendingLearningMode.load(std::memory_order_acquire);
    const int bankIndex = getAdaptiveCoeffBankIndex(sampleRate, bitDepth, mode);
    double stagedCoefficients[kAdaptiveNoiseShaperOrder] = {};
    getAdaptiveCoefficientsForSampleRateAndBitDepth(sampleRate, bitDepth, stagedCoefficients, kAdaptiveNoiseShaperOrder);

    const int limit = std::min(kAdaptiveNoiseShaperOrder, numCoefficients);
    for (int i = 0; i < limit; ++i)
        stagedCoefficients[i] = coeffs[i];

    publishCoeffsToBank(bankIndex, stagedCoefficients);
}

void AudioEngine::setAdaptiveAutosaveCallback(std::function<void()> callback)
{
    const std::scoped_lock lock(adaptiveAutosaveCallbackMutex);
    adaptiveAutosaveCallback = std::move(callback);
}

void AudioEngine::requestAdaptiveAutosave()
{
    std::function<void()> callbackCopy;
    {
        const std::scoped_lock lock(adaptiveAutosaveCallbackMutex);
        callbackCopy = adaptiveAutosaveCallback;
    }

    if (callbackCopy)
        callbackCopy();
}

void AudioEngine::publishCoeffs(const double* coeffs)
{
    const double sr = currentSampleRate.load(std::memory_order_acquire);
    const int bd  = ditherBitDepth.load(std::memory_order_acquire);
    const auto mode = pendingLearningMode.load(std::memory_order_acquire);
    const int bank = getAdaptiveCoeffBankIndex(sr, bd, mode);

    publishCoeffsToBank(bank, coeffs);
}

void AudioEngine::publishCoeffsToBank(int bankIndex, const double* coeffs)
{
    if (coeffs == nullptr)
        return;

    // 【安全性向上】Audio Thread からの呼び出しをブロック（デバッグビルド）
    // Message Thread またはオフラインレンダリング時のみ許可
    jassert (juce::MessageManager::existsAndIsCurrentThread());

    auto& bank = getAdaptiveCoeffBankForIndex(bankIndex);

    // RAII ガードを使用し、例外発生時もロックが解放されることを保証
    CoeffSetWriteLockGuard guard(bank);

    // 非アクティブバッファを予約（最大 100 回リトライ）
    // 学習スレッド/UI スレッドのみなので yield は安全
    for (int retry = 0; retry < 100; ++retry)
    {
        if (guard.acquire())
            break;
        std::this_thread::yield();
    }

    // 予約に失敗した場合は更新をスキップ（稀なケース）
    if (!guard.isAcquired())
    {
        DBG_LOG("[AudioEngine] Failed to acquire coeff write lock (bank="
                + juce::String(bankIndex) + ")");
        return;
    }

    // 非アクティブバッファに書き込み
    CoeffSet* inactive = getReservedInactiveCoeffSet(bank);

    for (int i = 0; i < kAdaptiveNoiseShaperOrder; ++i)
        inactive->k[i] = coeffs[i];

    // コミット（ロック解放を含む）
    // commit() を呼ぶことで、デストラクタでの二重解放を回避
    guard.commit();
}

bool AudioEngine::getAdaptiveNoiseShaperState(int bankIndex, NoiseShaperLearner::State& outState) const noexcept
{
    const auto& bank = getAdaptiveCoeffBankForIndex(bankIndex);
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(bank.stateMutex));
    outState = bank.state;
    return true;
}

void AudioEngine::setAdaptiveNoiseShaperState(int bankIndex, const NoiseShaperLearner::State& inState) noexcept
{
    auto& bank = getAdaptiveCoeffBankForIndex(bankIndex);
    std::lock_guard<std::mutex> lock(bank.stateMutex);
    bank.state = inState;
}

void AudioEngine::initialize()
{
    firstIrDryCrossfadePending.store(false, std::memory_order_release);
    firstIrDryCrossfadeDone.store(false, std::memory_order_release);
    dspCrossfadeUseDryAsOld.store(false, std::memory_order_release);
    dspCrossfadeDryHoldSamples.store(0, std::memory_order_release);
    dspCrossfadeDryScaleGain.reset(48000.0, 0.060);
    dspCrossfadeDryScaleGain.setCurrentAndTargetValue(1.0);

    // ダブルバッファ初期状態：alpha=1.0, previousValid=false
    m_views[0].alpha = 1.0f;
    m_views[0].previousValid = false;
    m_views[1].alpha = 1.0f;
    m_views[1].previousValid = false;
    m_activeIndex.store(0, std::memory_order_relaxed);

    // Start worker thread
    rebuildThread = std::thread(&AudioEngine::rebuildThreadLoop, this);

    // 初期DSP構築 (デフォルト設定)
    maxSamplesPerBlock.store(SAFE_MAX_BLOCK_SIZE);
    currentSampleRate.store(48000.0);

    m_fadeFloatBuffer.setSize(2, SAFE_MAX_BLOCK_SIZE, false, false, true);
    m_fadeDoubleBuffer.setSize(2, SAFE_MAX_BLOCK_SIZE, false, false, true);

    // オーディオデバイスがまだ開始していない段階でも、IRロード側には実用的な既定値を渡す。
    // SAFE_MAX_BLOCK_SIZE をそのまま使うと不要に巨大な一時NUCを組んでメモリ使用量が跳ねるため、
    uiEqEditor.addChangeListener(this);
    uiConvolverProcessor.addListener(this);

    // タイマー開始 (100ms間隔)
    // - DSP再構築リクエストのポーリング (Audio Threadからの依頼を処理)
    // - ガベージコレクション
    startTimer(100);

void AudioEngine::initWorkerThread()
{
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());
    m_workerThread.start();
}

void AudioEngine::shutdownWorkerThread()
{
    m_workerThread.stop();
}

bool AudioEngine::enqueueSnapshotCommand() noexcept
{
    auto makeDebounceKey = [](uint64_t seed, uint64_t value) noexcept -> uint64_t
    {
        // 64-bit mix (no libm, no allocation)
        constexpr uint64_t kMul = 0x9E3779B185EBCA87ull;
        seed ^= value + kMul + (seed << 6) + (seed >> 2);
        return seed;
    };

    const auto* mm = juce::MessageManager::getInstanceWithoutCreating();
    if (mm != nullptr && mm->isThisTheMessageThread())
    {
        uint64_t eqHash = 0;
        if (const auto* eqState = uiEqEditor.getEQStateSnapshot())
            eqHash = EQProcessor::computeParamsHash(eqState->toEQParameters());

        uint64_t key = 0xD6E8FEB86659FD93ull;
        key = makeDebounceKey(key, eqHash);
        key = makeDebounceKey(key, static_cast<uint64_t>(m_pendingIRChange.load(std::memory_order_acquire)));
        key = makeDebounceKey(key, static_cast<uint64_t>(m_pendingNSChange.load(std::memory_order_acquire)));
        key = makeDebounceKey(key, static_cast<uint64_t>(m_pendingAGCChange.load(std::memory_order_acquire)));
        key = makeDebounceKey(key, static_cast<uint64_t>(m_currentEqBypass.load(std::memory_order_relaxed)));
        key = makeDebounceKey(key, static_cast<uint64_t>(m_currentConvBypass.load(std::memory_order_relaxed)));
        key = makeDebounceKey(key, static_cast<uint64_t>(m_currentProcessingOrder.load(std::memory_order_relaxed)));
        key = makeDebounceKey(key, static_cast<uint64_t>(m_currentSoftClipEnabled.load(std::memory_order_relaxed)));
        key = makeDebounceKey(key, static_cast<uint64_t>(m_currentOversamplingFactor.load(std::memory_order_relaxed)));
        key = makeDebounceKey(key, static_cast<uint64_t>(m_currentOversamplingType.load(std::memory_order_relaxed)));
        key = makeDebounceKey(key, static_cast<uint64_t>(m_currentDitherBitDepth.load(std::memory_order_relaxed)));
        key = makeDebounceKey(key, static_cast<uint64_t>(m_currentNoiseShaperType.load(std::memory_order_relaxed)));
        key = makeDebounceKey(key, static_cast<uint64_t>(m_currentInputHeadroomDb.load(std::memory_order_relaxed)));
        key = makeDebounceKey(key, static_cast<uint64_t>(m_currentOutputMakeupDb.load(std::memory_order_relaxed)));
        key = makeDebounceKey(key, static_cast<uint64_t>(m_currentConvInputTrimDb.load(std::memory_order_relaxed)));
        key = makeDebounceKey(key, static_cast<uint64_t>(m_currentSaturationAmount.load(std::memory_order_relaxed)));

        const bool hasLastKey = hasLastEnqueuedSnapshotDebounceKey_.load(std::memory_order_acquire);
        const uint64_t lastKey = lastEnqueuedSnapshotDebounceKey_.load(std::memory_order_acquire);
        if (hasLastKey && lastKey == key)
        {
            diagLog("[VERIFY] enqueue snapshot debounced: identical snapshot intent");
            return true;
        }

        const uint64_t generation = m_generationManager.bumpGeneration();
        const convo::ParameterCommand cmd(convo::ParameterCommand::Type::ParameterChanged, generation);
        if (!m_commandBuffer.push(cmd))
        {
            DBG("AudioEngine: CommandBuffer full, dropping parameter change command");
            return false;
        }

        lastEnqueuedSnapshotDebounceKey_.store(key, std::memory_order_release);
        hasLastEnqueuedSnapshotDebounceKey_.store(true, std::memory_order_release);
        return true;
    }

    const uint64_t generation = m_generationManager.bumpGeneration();
    const convo::ParameterCommand cmd(convo::ParameterCommand::Type::ParameterChanged, generation);
    if (!m_commandBuffer.push(cmd))
    {
        DBG("AudioEngine: CommandBuffer full, dropping parameter change command");
        return false;
    }
    return true;
}

void AudioEngine::onSnapshotRequired(void* userData, uint64_t generation)
{
    auto* self = static_cast<AudioEngine*>(userData);
    if (self == nullptr)
        return;

    if (self->shutdownInProgress.load(std::memory_order_acquire))
        return;

    // 新しいアーキテクチャでは publishEngineState を使用
    auto newState = self->buildCurrentState();
    self->publishEngineState(std::move(newState), 0.0f);
}



void AudioEngine::debugAssertNotAudioThread() const
{
    // Worker Thread 専用チェック。
    // 現状は簡易的に Message Thread でないことを確認する。
    // （Worker Thread は Message Thread ではないため、このチェックで十分）
    jassert(!juce::MessageManager::getInstance()->isThisTheMessageThread());
}

bool AudioEngine::waitForAudioBlockBoundary(uint64_t observedCounter, uint32_t timeoutMs) const noexcept
{
    const uint32_t startMs = juce::Time::getMillisecondCounter();
    while (!rebuildThreadShouldExit.load(std::memory_order_acquire))
    {
        if (m_audioBlockCounter.load(std::memory_order_acquire) != observedCounter)
            return true;

        if ((juce::Time::getMillisecondCounter() - startMs) >= timeoutMs)
            return false;

        juce::Thread::sleep(1);
    }

    return false;
}

AudioEngine::EQCacheManager::EQCacheManager()
{
    cacheMapPtr.store(new CacheMap(), std::memory_order_release);
}

bool AudioEngine::EQCacheManager::tryEnqueueDeferredMap(CacheMap* map) noexcept
{
    if (map == nullptr)
        return true;

    deleteEQCache((EQCoeffCache*)map);
    return true;
}

void AudioEngine::EQCacheManager::drainDeferredMapsUnderLock() noexcept
{
    if (enqueueFallbackMaps.empty())
        return;

    auto out = enqueueFallbackMaps.begin();
    for (auto it = enqueueFallbackMaps.begin(); it != enqueueFallbackMaps.end(); ++it)
    {
        if (!tryEnqueueDeferredMap(*it))
            *out++ = *it;
    }

    enqueueFallbackMaps.erase(out, enqueueFallbackMaps.end());
}

void AudioEngine::EQCacheManager::storeNewMap(CacheMap* newMap) noexcept
{
    auto* old = cacheMapPtr.exchange(newMap, std::memory_order_acq_rel);
    if (old == nullptr)
        return;

    deleteDSP((AudioEngine::DSPCore*)old);
}

EQCoeffCache* AudioEngine::EQCacheManager::getOrCreate(const convo::EQParameters& params,
                                                       double sampleRate,
                                                       int maxBlockSize,
                                                       uint64_t generation)
{
    const uint64_t hash = EQProcessor::computeParamsHash(params);
    const CacheMap* currentMap = loadMap();
    if (currentMap == nullptr)
        return nullptr;

    auto it = currentMap->map.find(hash);
    if (it != currentMap->map.end())
        return it->second;

    EQCoeffCache* cache = EQProcessor::createCoeffCache(params, sampleRate, maxBlockSize, generation);
    if (cache == nullptr)
        return nullptr;

    std::lock_guard<std::mutex> lock(writeMutex);

    drainDeferredMapsUnderLock();

    // Lock取得中に他スレッドが同じハッシュを追加した可能性を再確認
    currentMap = loadMap();
    if (currentMap == nullptr)
    {
        deleteEQCache(cache);
        return nullptr;
    }

    it = currentMap->map.find(hash);
    if (it != currentMap->map.end())
    {
        // 先に追加されたキャッシュを採用し、新規作成分を破棄
        deleteEQCache(cache);
        return it->second;
    }

    CacheMap* newMap = nullptr;
    try
    {
        newMap = new CacheMap(*currentMap);
    }
    catch (const std::bad_alloc&)
    {
        deleteEQCache(cache);
        return nullptr;
    }

    newMap->map[hash] = cache;
    storeNewMap(newMap);

    return cache;
}

EQCoeffCache* AudioEngine::EQCacheManager::get(uint64_t hash) const noexcept
{
    const CacheMap* currentMap = loadMap();
    if (currentMap == nullptr)
        return nullptr;

    const auto it = currentMap->map.find(hash);
    return (it != currentMap->map.end()) ? it->second : nullptr;
}

void AudioEngine::EQCacheManager::releaseCache(EQCoeffCache* cache) noexcept
{
    if (cache != nullptr)
        deleteEQCache(cache);
}

AudioEngine::EQCacheManager::~EQCacheManager()
{
    std::lock_guard<std::mutex> lock(writeMutex);

    CacheMap* currentMap = cacheMapPtr.exchange(nullptr, std::memory_order_acq_rel);
    if (currentMap != nullptr)
        deleteDSP((AudioEngine::DSPCore*)currentMap);

    for (auto* map : enqueueFallbackMaps)
        deleteEQCache((EQCoeffCache*)map);

    enqueueFallbackMaps.clear();
}

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

//--------------------------------------------------------------
// DSPCore Implementation
//--------------------------------------------------------------
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

//--------------------------------------------------------------
// requestRebuild - DSPグラフの再構築 (Message Thread)
//--------------------------------------------------------------
void AudioEngine::requestRebuild(double sampleRate, int samplesPerBlock)
{
    // UIコンポーネント(uiEqEditor等)へのアクセスやMKLメモリ確保を行うため、必ずMessage Threadで実行すること
    jassert (juce::MessageManager::getInstance()->isThisTheMessageThread());

    if (deferredStructuralRebuildPending_.load(std::memory_order_acquire))
    {
        const int64_t dueTicks = deferredStructuralRebuildDueTicks_.load(std::memory_order_acquire);
        const int64_t nowTicks = juce::Time::getHighResolutionTicks();
        const bool uiHasIr = uiConvolverProcessor.isIRLoaded();
        const bool activeHasIr = (activeDSP != nullptr) && activeDSP->convolver.isIRLoaded();

        if (dueTicks > 0 && nowTicks < dueTicks && uiHasIr && !activeHasIr)
        {
            diagLog("[DIAG] requestRebuild(sr,bs): SUPPRESSED direct rebuild during deferred Structural window SR="
                + juce::String(sampleRate, 2));
            return;
        }
    }

    const int publishedFftSize = uiConvolverProcessor.getActiveCacheFFTSize();
    const int targetFftSize = uiConvolverProcessor.getTargetUpgradeFFTSize();
    const bool suppressIntermediateMixedPhasePublish =
        uiConvolverProcessor.isProgressiveUpgradeEnabled()
        && uiConvolverProcessor.getPhaseMode() == ConvolverProcessor::PhaseMode::Mixed
        && publishedFftSize > 0
        && publishedFftSize < targetFftSize;

    if (suppressIntermediateMixedPhasePublish)
    {
        diagLog("[DIAG] requestRebuild(sr,bs): SUPPRESSED intermediate progressive mixed-phase publish fft="
                + juce::String(publishedFftSize)
                + " targetFFT=" + juce::String(targetFftSize)
                + " SR=" + juce::String(sampleRate, 2));
        return;
    }

    if (noiseShaperLearner && noiseShaperLearner->isRunning())
        noiseShaperLearner->stopLearning();

    // 新しいDSPコアを作成
    DSPCore* newDSP = new DSPCore();
    newDSP->convolver.setVisualizationEnabled(false); // DSP用は可視化データ不要

    // UIプロセッサから状態をコピー
    // EQ状態は Snapshot 経由で反映するため、ここでの直接同期は行わない。
    newDSP->convolver.syncStateFrom(uiConvolverProcessor);

    // キャプチャ用変数
    int ditherDepth = ditherBitDepth.load();
    int osFactor = manualOversamplingFactor.load();
    OversamplingType osType = oversamplingType.load();
    NoiseShaperType nsType = noiseShaperType.load();
    DSPCore* current = activeDSP; // 現在のアクティブDSPをキャプチャ
    int generation = 0;

    RebuildTask task;
    task.newDSP = newDSP;
    task.currentDSP = current;
    task.sampleRate = sampleRate;
    task.samplesPerBlock = samplesPerBlock;
    task.ditherDepth = ditherDepth;
    task.manualOversamplingFactor = osFactor;
    task.oversamplingType = osType;
    task.noiseShaperType = nsType;

        DSPCore* dspToDestroy = nullptr; // To be destroyed outside the lock
    DSPCore* currentToRelease = nullptr;
    bool queued = false;
    bool blockedAsDuplicate = false;
    bool blockedAsRecentDuplicate = false;
    {
        std::lock_guard<std::mutex> lock(rebuildMutex);

        if (hasPendingTask)
        {
            const bool sameAsPending =
                std::abs(pendingTask.sampleRate - sampleRate) <= 1.0e-6
                && pendingTask.samplesPerBlock == samplesPerBlock
                && pendingTask.ditherDepth == ditherDepth
                && pendingTask.manualOversamplingFactor == osFactor
                && pendingTask.oversamplingType == osType
                && pendingTask.noiseShaperType == nsType;

            if (sameAsPending)
            {
                blockedAsDuplicate = true;
            }
            else
            {
                dspToDestroy = pendingTask.newDSP;
                currentToRelease = pendingTask.currentDSP;
            }
        }

        if (!blockedAsDuplicate)
        {
            const bool sameAsLastQueued =
                std::abs(lastQueuedTaskSignature.sampleRate - sampleRate) <= 1.0e-6
                && lastQueuedTaskSignature.samplesPerBlock == samplesPerBlock
                && lastQueuedTaskSignature.ditherDepth == ditherDepth
                && lastQueuedTaskSignature.manualOversamplingFactor == osFactor
                && lastQueuedTaskSignature.oversamplingType == osType
                && lastQueuedTaskSignature.noiseShaperType == nsType;

            if (sameAsLastQueued)
            {
                const int64_t nowTicks = juce::Time::getHighResolutionTicks();
                const int64_t minDeltaTicks = juce::Time::getHighResolutionTicksPerSecond() / 5; // 200ms
                if (lastQueuedTaskTicks > 0 && (nowTicks - lastQueuedTaskTicks) < minDeltaTicks)
                    blockedAsRecentDuplicate = true;
            }
        }

        if (!blockedAsDuplicate && !blockedAsRecentDuplicate)
        {
            generation = ++rebuildGeneration;
            task.generation = generation;
            pendingTask = task;
            hasPendingTask = true;
            lastQueuedTaskSignature = task;
            lastQueuedTaskTicks = juce::Time::getHighResolutionTicks();
            queued = true;
        }
    }

    if (queued)
    {
        rebuildCV.notify_all();
        diagLog("[DIAG] requestRebuild(sr,bs): task queued generation=" + juce::String(generation)
            + " SR=" + juce::String(sampleRate, 2));
    }
    else
    {
        if (blockedAsRecentDuplicate)
        {
            diagLog("[DIAG] requestRebuild(sr,bs): BLOCKED duplicate recent task SR="
                + juce::String(sampleRate, 2));
        }
        else
        {
            diagLog("[DIAG] requestRebuild(sr,bs): BLOCKED duplicate pending task SR="
                + juce::String(sampleRate, 2));
        }
    }

    // Destroy orphaned DSP objects outside the lock.
    if (dspToDestroy)
        deleteDSP(dspToDestroy);
    if (currentToRelease)
        deleteDSP(currentToRelease);

    if (!queued)
    {
        deleteDSP(newDSP);
        if (current)
        {
            // EBR: lifetime managed by RCUReader
        }
    }
}

void AudioEngine::stopRebuildThread()
{
    // exit フラグを立てる（predicate が次に評価された時に break する）
    rebuildThreadShouldExit.store(true, std::memory_order_release);

    // 待機中のスレッドを確実に起こす
    rebuildCV.notify_all();

    if (rebuildThread.joinable())
        rebuildThread.join();
}

void AudioEngine::rebuildThreadLoop()
{
    affinityManager.applyCurrentThreadPolicy(ThreadType::HeavyBackground);

    // Set denormal handling modes for this thread. This is crucial for performance
    // in MKL VML and AVX/SSE operations, which can be significantly slowed down
    // by subnormal numbers. This setting is thread-local.
    vmlSetMode(VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    rebuildThreadIsRunning.store(true, std::memory_order_release);

    while (true)
    {
        try
        {
            RebuildTask task;
            {
                std::unique_lock<std::mutex> lock(rebuildMutex);
                rebuildCV.wait(lock, [this] { return hasPendingTask || rebuildThreadShouldExit.load(); });

                if (rebuildThreadShouldExit.load()) break;

                // Copy task and clear pendingTask pointers to transfer ownership
                task = pendingTask;
                pendingTask.newDSP = nullptr;
                pendingTask.currentDSP = nullptr;

                hasPendingTask = false;
            }

            if (task.newDSP == nullptr)
            {
                jassertfalse;
                continue;
            }

            struct DSPGuard
            {
                DSPCore* ptr;
                ~DSPGuard()
                {
                    if (ptr != nullptr)
                        deleteDSP(ptr);
                }
            } dspGuard { task.newDSP };

            // Helper to check obsolescence
            const auto isObsolete = [&] {
                return isRebuildObsolete(task.generation) || rebuildThreadShouldExit.load();
            };

            if (isObsolete())
                continue;

            // 1. Prepare (メモリ確保)
            task.newDSP->prepare(task.sampleRate, task.samplesPerBlock, task.ditherDepth, task.manualOversamplingFactor, task.oversamplingType, task.noiseShaperType, this);

            if (isObsolete())
                continue;

            // 2. Reuse Logic
            //
            // 【task.currentDSP (= 旧 activeDSP) の安全性証明】
            //
            // task.currentDSP は RCU 設計により、退役後も一定期間（Epoch）生存が
            // 保証される。rebuild タスクはこの期間内に完了することを前提としている。
            // (通常、数ミリ秒〜数十ミリ秒。EBR Queue の遅延削除により安全性確保)
            //
            // 【shared_ptr 不採用の理由】
            //     Audio Thread は std::shared_ptr の参照カウント操作を禁止している
            //     (コーディング規約)。currentDSP の RCU 設計はこの制約に基づく。
            bool irReused = false;
            if (task.currentDSP)
            {

                if (std::abs(task.currentDSP->sampleRate - task.sampleRate) < 1e-6 &&
                    task.currentDSP->oversamplingFactor == task.newDSP->oversamplingFactor &&
                    task.currentDSP->convolver.getCurrentBufferSize() == task.newDSP->convolver.getCurrentBufferSize())
                {
                    // IRの生成条件が一致しているか確認
                    if (task.newDSP->convolver.getIRName() == task.currentDSP->convolver.getIRName() &&
                        task.newDSP->convolver.getPhaseMode() == task.currentDSP->convolver.getPhaseMode() &&
                        std::abs(task.newDSP->convolver.getMixedTransitionStartHz() - task.currentDSP->convolver.getMixedTransitionStartHz()) < 0.001f &&
                        std::abs(task.newDSP->convolver.getMixedTransitionEndHz() - task.currentDSP->convolver.getMixedTransitionEndHz()) < 0.001f &&
                        std::abs(task.newDSP->convolver.getMixedPreRingTau() - task.currentDSP->convolver.getMixedPreRingTau()) < 0.001f &&
                        task.newDSP->convolver.getExperimentalDirectHeadEnabled() == task.currentDSP->convolver.getExperimentalDirectHeadEnabled() &&
                        std::abs(task.newDSP->convolver.getTargetIRLength() - task.currentDSP->convolver.getTargetIRLength()) < 0.001f)
                    {
                        // 既存のConvolutionエンジンを共有（クローン回避・グリッチ防止）
                        task.newDSP->convolver.shareConvolutionEngineFrom(task.currentDSP->convolver);
                        irReused = true;
                    }
                }
            }

            // 3. Rebuild IR if needed (Heavy operation)
            if (!irReused && task.newDSP->convolver.getIRLength() > 0)
            {
                if (isObsolete())
                    continue;
                task.newDSP->convolver.rebuildAllIRsSynchronous(isObsolete);
            }

            if (isObsolete())
                continue;

            // 4. Refresh Latency (Prevent pitch slide during fade-in)
            task.newDSP->convolver.refreshLatency();

            // 5. Fade In
            task.newDSP->fadeInSamplesLeft.store(DSPCore::FADE_IN_SAMPLES, std::memory_order_relaxed);

            // 6. Commit on Message Thread
            // Release ownership from guard, pass to commitNewDSP
            DSPCore* dspToCommit = dspGuard.ptr;
            dspGuard.ptr = nullptr;
            prepareCommit(dspToCommit, task.generation);
        }
        catch (const std::exception& e)
        {
            DBG("AudioEngine::rebuildThreadLoop exception: " << e.what());
            juce::ignoreUnused(e);
        }
        catch (...)
        {
            DBG("AudioEngine::rebuildThreadLoop unknown exception");
        }
    }

    rebuildThreadIsRunning.store(false, std::memory_order_release);
}

void AudioEngine::commitNewDSP(DSPCore* newDSP, int generation)
{
    DSPCore* dspToTrash = nullptr;
    uint64_t retireEpoch = 0;
    bool scheduleDryAsOldCrossfade = false;
    double dryAsOldFadeTimeSec = 0.0;

    // Lock to ensure the check and commit are atomic with respect to new rebuild requests.
    {
        std::lock_guard<std::mutex> lock(rebuildMutex);

        // 古いリクエストの結果であれば破棄 (Race condition対策)
        if (generation != rebuildGeneration.load(std::memory_order_relaxed))
        {
            deleteDSP(newDSP);
            return;
        }

        // 公開不変条件:
        // IR を実際に使う構成では finalized 済みのみ公開する。
        // 一方、IR 未ロード時のパススルーDSPまで弾くと起動直後に無音化するため許可する。
        if (newDSP == nullptr
            || (newDSP->convolver.isIRLoaded() && !newDSP->convolver.isIRFinalized()))
        {
            DBG("[AudioEngine] commitNewDSP: rejected non-finalized DSP publish");
            if (newDSP != nullptr)
                deleteDSP(newDSP);
            return;
        }

        // 1. 旧 DSP を安全にキャプチャしてから新 DSP を公開する
        dspToTrash = activeDSP;

        // 2. Update the atomic raw pointer for the Audio Thread (Wait-free)
        currentDSP.store(newDSP, std::memory_order_release);

        // 3. EBR：エポックを進める
        convo::EpochManager::instance().advanceEpoch();
        retireEpoch = convo::EpochManager::instance().currentEpoch();
        g_currentEpoch.store(retireEpoch, std::memory_order_release);

        // 4. Take ownership of the new DSP
        activeDSP = newDSP;

        const uint64_t newSessionId = globalCaptureSessionId.fetch_add(1, std::memory_order_acq_rel) + 1;
        if (newDSP != nullptr)
            newDSP->currentCaptureSessionId = newSessionId;

        // この世代の publish が完了したので outstanding rebuild 窓を閉じる。
        lastCommittedRebuildGeneration.store(generation, std::memory_order_release);
    }


    // 5. 初回IRロード時（旧DSPなし）: dry を旧信号としてクロスフェード予約
    if (dspToTrash == nullptr
        && newDSP != nullptr
        && newDSP->convolver.isIRLoaded()
        && !firstIrDryCrossfadeDone.load(std::memory_order_acquire))
    {
        // 初回のみ dry -> IR を明示的にフェードし、立ち上がりノイズを抑制する。
        scheduleDryAsOldCrossfade = true;
        dryAsOldFadeTimeSec = std::max(0.001, m_irFadeTimeSec.load(std::memory_order_relaxed));

        const int newLatency = newDSP->convolver.getTotalLatencySamples();
        int dOld = std::min(newLatency, latencyBufSize - 1); // dry 側を遅延させて整合
        const int dNew = 0;
        latencyDelayOld.store(dOld, std::memory_order_release);
        latencyDelayNew.store(dNew, std::memory_order_release);
        latencyResetPending.store(true, std::memory_order_release);
    }

    diagLog("[DIAG] commitNewDSP: entry gen=" + juce::String(generation)
        + " dspToTrash=" + (dspToTrash != nullptr ? juce::String(dspToTrash->convolver.isIRLoaded() ? "IR" : "passthrough") : "null")
        + " irLoaded=" + (newDSP != nullptr ? juce::String((int)newDSP->convolver.isIRLoaded()) : "n/a"));
    // 5. RCU deferred release：旧 DSP を grace period 後に解放する
    if (dspToTrash != nullptr)
    {
        // --- クロスフェード判定・遅延整合処理 ---
        bool needsCrossfade = false;
        double fadeTimeSec = 0.0;

        if (newDSP != nullptr && dspToTrash != nullptr)
        {
            const bool oldHasIR = dspToTrash->convolver.isIRLoaded();
            const bool newHasIR = newDSP->convolver.isIRLoaded();
            const bool hasAudibleConvolverTransition = oldHasIR || newHasIR;
            const bool irPresenceChanged = (oldHasIR != newHasIR);

            // 1. オーバーサンプリング倍率変更
            if (hasAudibleConvolverTransition
                && newDSP->oversamplingFactor != dspToTrash->oversamplingFactor)
            {
                needsCrossfade = true;
                fadeTimeSec = std::max(fadeTimeSec, m_osFadeTimeSec.load(std::memory_order_relaxed));
            }

            // 2. 構造ハッシュ比較によるその他の変更検出
            // 両者とも IR 未ロード（実質 dry/passthrough）の場合、
            // 構造ハッシュ差だけでクロスフェードを発火させる必要はない。
            // 起動直後の設定同期で不要なフェード連打が起きるのを防ぐ。
            if (hasAudibleConvolverTransition)
            {
                const uint64_t oldHash = dspToTrash->convolver.getStructuralHash();
                const uint64_t newHash = newDSP->convolver.getStructuralHash();
                diagLog("[DIAG] commitNewDSP: hashes oldHash=" + juce::String((int64)oldHash) + " newHash=" + juce::String((int64)newHash) + " needsCF=" + juce::String((int)needsCrossfade));
                if (oldHash != newHash)
                {
                    needsCrossfade = true;
                    const double baseIrFade = m_irFadeTimeSec.load(std::memory_order_relaxed);

                    // IRの有無が切り替わる遷移（passthrough->IR / IR->passthrough）は
                    // 長時間フェードにすると二重処理時間が伸び、再生を圧迫しやすい。
                    // そのため短いフェード時間に制限し、他の長時間系は適用しない。
                    if (irPresenceChanged)
                    {
                        fadeTimeSec = std::max(fadeTimeSec, std::clamp(baseIrFade, 0.001, 0.010));
                    }
                    else
                    {
                        fadeTimeSec = std::max(fadeTimeSec, baseIrFade);
                        fadeTimeSec = std::max(fadeTimeSec, m_irLengthFadeTimeSec.load(std::memory_order_relaxed));
                        fadeTimeSec = std::max(fadeTimeSec, m_phaseFadeTimeSec.load(std::memory_order_relaxed));
                        fadeTimeSec = std::max(fadeTimeSec, m_directHeadFadeTimeSec.load(std::memory_order_relaxed));
                        fadeTimeSec = std::max(fadeTimeSec, m_nucFilterFadeTimeSec.load(std::memory_order_relaxed));
                        fadeTimeSec = std::max(fadeTimeSec, m_tailFadeTimeSec.load(std::memory_order_relaxed));
                    }
                }
            }
            else
            {
                diagLog("[DIAG] commitNewDSP: skip crossfade for passthrough->passthrough");
            }

            // --- レイテンシ差・バッファ初期化はクロスフェード時のみ ---
            if (needsCrossfade)
            {
                const int oldLatency = dspToTrash->convolver.getTotalLatencySamples();
                const int newLatency = newDSP->convolver.getTotalLatencySamples();
                const int targetLatency = std::max(oldLatency, newLatency);
                int dOld = targetLatency - oldLatency;
                int dNew = targetLatency - newLatency;
                dOld = std::min(dOld, latencyBufSize - 1);
                dNew = std::min(dNew, latencyBufSize - 1);
                latencyDelayOld.store(dOld, std::memory_order_release);
                latencyDelayNew.store(dNew, std::memory_order_release);
                // ★ resetはAudioThreadで1回だけ行う
                latencyResetPending.store(true, std::memory_order_release);

                if (!oldHasIR && newHasIR)
                    dspCrossfadeStartDelayBlocks.store(std::max(0, m_crossfadeStartDelayBlocks.load(std::memory_order_relaxed)), std::memory_order_release);
                else
                    dspCrossfadeStartDelayBlocks.store(0, std::memory_order_release);
            }
            else
            {
                // クロスフェード不要時は絶対に遅延値・バッファを触らない
                dspCrossfadeStartDelayBlocks.store(0, std::memory_order_release);
            }

            // デフォルト値（fadeTimeSec==0なら30ms）
            if (fadeTimeSec <= 0.0)
                fadeTimeSec = 0.030;
        }

        // --- クロスフェードdeduplication・スナップショット ---
        if (needsCrossfade)
        {
            const bool isFadingActive = (sanitizeRawPtr(fadingOutDSP.load(std::memory_order_acquire)) != nullptr) ||
                                        dspCrossfadePending.load(std::memory_order_acquire) ||
                                        dspCrossfadeUseDryAsOld.load(std::memory_order_acquire);
            if (isFadingActive)
            {
                if (auto* prev = sanitizeRawPtr(queuedOldDSP.exchange(dspToTrash, std::memory_order_acq_rel)))
                    deleteDSP(prev);
                queuedNextFadeTimeSec.store(fadeTimeSec, std::memory_order_release);
                fadeQueued.store(true, std::memory_order_release);
            }
            else
            {
                if (auto* oldFading = sanitizeRawPtr(fadingOutDSP.exchange(dspToTrash, std::memory_order_acq_rel)))
                    deleteDSP(oldFading);
                queuedFadeTimeSec.store(fadeTimeSec, std::memory_order_release);
                dspCrossfadePending.store(true, std::memory_order_release);
                setIRChangeFlag();
            }
        }
        else if (dspToTrash)
        {
            // クロスフェード不要時は即時解放
            deleteDSP(dspToTrash);
        }
    }

    if (scheduleDryAsOldCrossfade)
    {
        queuedFadeTimeSec.store(dryAsOldFadeTimeSec, std::memory_order_release);
        dspCrossfadeDryHoldSamples.store(std::max(1, maxSamplesPerBlock.load(std::memory_order_acquire)), std::memory_order_release);
        // dry スケーリング: passthrough (1.0) -> IR scale (~0.133) へ 60ms で ramp
        dspCrossfadeDryScaleGain.reset(std::max(1.0, currentSampleRate.load(std::memory_order_acquire)), 0.060);
        dspCrossfadeDryScaleGain.setCurrentAndTargetValue(1.0);
        dspCrossfadeDryScaleGain.setTargetValue(uiConvolverProcessor.getCurrentIRScale());
        firstIrDryCrossfadePending.store(true, std::memory_order_release);
        dspCrossfadePending.store(true, std::memory_order_release);
        firstIrDryCrossfadeDone.store(true, std::memory_order_release);
        setIRChangeFlag();

        diagLog("[DIAG] commitNewDSP: first-load dry->IR crossfade armed fadeSec="
            + juce::String(dryAsOldFadeTimeSec, 3)
            + " irName=" + newDSP->convolver.getIRName());
    }

    if (newDSP != nullptr)
    {
        diagLog("[DIAG] commitNewDSP: before setMixedPhaseState state="
            + juce::String(newDSP->convolver.getMixedPhaseState()));
        uiConvolverProcessor.setMixedPhaseState(newDSP->convolver.getMixedPhaseState());
        diagLog("[DIAG] commitNewDSP: after setMixedPhaseState");
    }

    const LearningCommand cmd {
        LearningCommand::Type::DSPReady,
        false,
        pendingLearningMode.load(std::memory_order_acquire),
        static_cast<uint64_t>(generation)
    };

    diagLog("[DIAG] commitNewDSP: before enqueueLearningCommand");
    if (!enqueueLearningCommand(cmd))
    {
        DBG("[AudioEngine] commitNewDSP: command queue overflow");
        diagLog("[DIAG] commitNewDSP: enqueueLearningCommand overflow");
    }
    else
    {
        diagLog("[DIAG] commitNewDSP: enqueueLearningCommand ok");
    }

    // NOTE: rebuild 完了通知の唯一の発火点。
    // sendChangeMessage() は commitNewDSP() でのみ rebuild 用途で呼ぶ。
    // それ以外の sendChangeMessage() はフェード完了・UIパラメータ変更・
    // 状態復元など rebuild とは独立したイベント用途。
    diagLog("[DIAG] commitNewDSP: before sendChangeMessage");
    sendChangeMessage();
    diagLog("[DIAG] commitNewDSP: after sendChangeMessage");
}

void AudioEngine::processLearningCommands() noexcept
{
    if (learnerDispatchOverflow.load(std::memory_order_acquire))
    {
        const LearnerDispatchAction last = lastFailedAction.load(std::memory_order_acquire);
        if (enqueueLearnerDispatch(last))
            learnerDispatchOverflow.store(false, std::memory_order_release);
    }

    LearningCommand cmd;
    while (dequeueLearningCommand(cmd))
    {
        switch (cmd.type)
        {
            case LearningCommand::Type::Start:
            {
                requestedLearningMode = cmd.mode;
                requestedLearningResume = cmd.resume;
                requestedLearningGeneration = cmd.irGeneration;

                auto* dsp = currentDSP.load(std::memory_order_acquire);
                // irGeneration チェックを削除: DSP が有効かつ型が適切であれば即座に学習開始可能
                const bool dspReady = (dsp != nullptr)
                    && (dsp->noiseShaperType == NoiseShaperType::Adaptive9thOrder);

                if (!dspReady)
                {
                    learningRuntimeState = LearningRuntimeState::WaitingForDSP;
                    break;
                }

                if (learningRuntimeState == LearningRuntimeState::Running)
                {
                    const LearnerDispatchAction stopAction {
                        LearnerDispatchAction::Type::Stop,
                        false,
                        requestedLearningMode
                    };

                    if (!enqueueLearnerDispatch(stopAction))
                    {
                        DBG("[AudioEngine] processLearningCommands: learner stop queue overflow");
                    }
                }

                const LearnerDispatchAction startAction {
                    LearnerDispatchAction::Type::Start,
                    requestedLearningResume,
                    requestedLearningMode
                };

                if (enqueueLearnerDispatch(startAction))
                    learningRuntimeState = LearningRuntimeState::Running;
                else
                {
                    learningRuntimeState = LearningRuntimeState::WaitingForDSP;
                    DBG("[AudioEngine] processLearningCommands: learner start queue overflow");
                }
                break;
            }

            case LearningCommand::Type::Stop:
            {
                requestedLearningResume = false;
                requestedLearningGeneration = currentIRGeneration;

                const LearnerDispatchAction stopAction {
                    LearnerDispatchAction::Type::Stop,
                    false,
                    requestedLearningMode
                };

                if (!enqueueLearnerDispatch(stopAction))
                {
                    DBG("[AudioEngine] processLearningCommands: learner stop queue overflow");
                }

                learningRuntimeState = LearningRuntimeState::Idle;
                break;
            }

            case LearningCommand::Type::IRChanged:
            {
                const bool shouldRestart = (learningRuntimeState != LearningRuntimeState::Idle);
                requestedLearningGeneration = cmd.irGeneration;

                const LearnerDispatchAction stopAction {
                    LearnerDispatchAction::Type::Stop,
                    false,
                    requestedLearningMode
                };

                if (!enqueueLearnerDispatch(stopAction))
                {
                    DBG("[AudioEngine] processLearningCommands: learner stop queue overflow");
                }

                if (shouldRestart)
                {
                    requestedLearningResume = false;
                    learningRuntimeState = LearningRuntimeState::WaitingForDSP;
                }
                else
                {
                    learningRuntimeState = LearningRuntimeState::Idle;
                }
                break;
            }

            case LearningCommand::Type::DSPReady:
            {
                currentIRGeneration = cmd.irGeneration;

                // irGeneration チェックを削除: WaitingForDSP 状態であれば遅延なく学習開始
                if (learningRuntimeState == LearningRuntimeState::WaitingForDSP)
                {
                    const LearnerDispatchAction startAction {
                        LearnerDispatchAction::Type::Start,
                        requestedLearningResume,
                        requestedLearningMode
                    };

                    if (enqueueLearnerDispatch(startAction))
                    {
                        learningRuntimeState = LearningRuntimeState::Running;
                    }
                    else
                    {
                        DBG("[AudioEngine] processLearningCommands: DSPReady learner start queue overflow");
                    }
                }
                break;
            }
        }
    }
}

void AudioEngine::processDeferredLearningActions()
{
    LearnerDispatchAction action;
    while (dequeueLearnerDispatch(action))
    {
        if (noiseShaperLearner == nullptr)
            continue;

        if (action.type == LearnerDispatchAction::Type::Stop)
        {
            noiseShaperLearner->stopLearning();
            continue;
        }

        noiseShaperLearner->setLearningMode(action.mode);
        noiseShaperLearner->startLearning(action.resume);
    }
}

void AudioEngine::resetLearningControlState() noexcept
{
    learningCommandWrite = 0;
    learningCommandRead = 0;
    learnerDispatchWrite = 0;
    learnerDispatchRead = 0;
    learnerDispatchOverflow.store(false, std::memory_order_release);
    lastFailedAction.store(LearnerDispatchAction {}, std::memory_order_release);
    learningRuntimeState = LearningRuntimeState::Idle;
    requestedLearningMode = pendingLearningMode.load(std::memory_order_acquire);
    requestedLearningResume = false;
    requestedLearningGeneration = pendingIRGeneration;
    currentIRGeneration = pendingIRGeneration;
}

void AudioEngine::timerCallback()
{
    processRebuildRequestsInternal();

    // フェイルセーフ: current snapshot が欠落した状態を放置すると
    // EQ変更が演算経路へ乗らないため、Message Thread 側で自己修復する。
    if (!shutdownInProgress.load(std::memory_order_acquire)
        && currentDSP.load(std::memory_order_acquire) != nullptr
        && !isFading()
        && m_activeIndex.load(std::memory_order_acquire) < 0)  // ダブルバッファ初期化チェック
    {
        diagLog("[VERIFY] snapshot bootstrap: current was null, requesting worker snapshot refresh");
        if (!enqueueSnapshotCommand())
            diagLog("[VERIFY] snapshot bootstrap: enqueueSnapshotCommand failed");
    }

    {
        const uint32_t observedVersion = debugAppliedEqHashVersion.load(std::memory_order_acquire);
        debugObservedEqHashVersion = observedVersion;

        const uint64_t createdHash = debugLastCreatedEqHash.load(std::memory_order_acquire);
        const uint64_t appliedHash = debugLastAppliedEqHash.load(std::memory_order_acquire);
        const uint64_t createBlockCounter = debugLastCreateAudioBlockCounter.load(std::memory_order_acquire);
        const uint64_t nowBlockCounter = m_audioBlockCounter.load(std::memory_order_acquire);
        const uint64_t processedBlocksSinceCreate = (nowBlockCounter >= createBlockCounter)
            ? (nowBlockCounter - createBlockCounter)
            : 0;
        const int dspReady = (currentDSP.load(std::memory_order_acquire) != nullptr) ? 1 : 0;
        const int coordIsFading = debugLastCoordinatorIsFading.load(std::memory_order_acquire);
        const int updateFadeReturned = debugLastUpdateFadeReturned.load(std::memory_order_acquire);
        const int fromNull = debugLastSnapshotFromNull.load(std::memory_order_acquire);
        const int toNull = debugLastSnapshotToNull.load(std::memory_order_acquire);

        const bool eqMismatch = (createdHash != 0 && createdHash != appliedHash && dspReady == 1);
        const bool recoveryEligible = eqMismatch
            && coordIsFading == 0
            && processedBlocksSinceCreate >= 4;
        if (recoveryEligible)
        {
            if (debugLastRecoveryAttemptCreatedEqHash != createdHash)
            {
                debugRecoveryRetryCountForCurrentHash = 0;
                debugRecoverySuppressedForCurrentHash = false;
            }

            const bool firstAttemptForThisHash = (debugLastRecoveryAttemptCreatedEqHash != createdHash);
            const bool retryAfterProgress = (nowBlockCounter > debugLastRecoveryAttemptAudioBlockCounter + 256);
            if (firstAttemptForThisHash || retryAfterProgress)
            {
                debugLastRecoveryAttemptCreatedEqHash = createdHash;
                debugLastRecoveryAttemptAudioBlockCounter = nowBlockCounter;

                if (debugRecoveryRetryCountForCurrentHash < 3)
                {
                    ++debugRecoveryRetryCountForCurrentHash;
#ifdef _DEBUG
                    const uint64_t workerRecv = m_workerThread.getCommandsReceived();
                    const uint64_t workerSnap = m_workerThread.getSnapshotsCreated();
                    const uint64_t workerDrop = m_workerThread.getCommandsDropped();
#else
                    const uint64_t workerRecv = 0;
                    const uint64_t workerSnap = 0;
                    const uint64_t workerDrop = 0;
#endif
                    diagLog("[VERIFY] snapshot recovery: retry=" + juce::String(debugRecoveryRetryCountForCurrentHash)
                        + " forcing reapply createdHash=0x"
                        + juce::String::toHexString(static_cast<juce::int64>(createdHash))
                        + " appliedHash=0x"
                        + juce::String::toHexString(static_cast<juce::int64>(appliedHash))
                        + " blocksSinceCreate=" + juce::String(static_cast<juce::int64>(processedBlocksSinceCreate))
                        + " workerRecv=" + juce::String(static_cast<juce::int64>(workerRecv))
                        + " workerSnap=" + juce::String(static_cast<juce::int64>(workerSnap))
                        + " workerDrop=" + juce::String(static_cast<juce::int64>(workerDrop))
                        + " genNow=" + juce::String(static_cast<juce::int64>(m_generationManager.getCurrentGeneration())));

                    if (!enqueueSnapshotCommand())
                        diagLog("[VERIFY] snapshot recovery: enqueueSnapshotCommand failed");
                }
                else if (!debugRecoverySuppressedForCurrentHash)
                {
                    debugRecoverySuppressedForCurrentHash = true;
                    diagLog("[VERIFY] snapshot recovery: suppressed for createdHash=0x"
                        + juce::String::toHexString(static_cast<juce::int64>(createdHash))
                        + " after 3 retries (waiting for next hash change)");
                }
            }
        }
        else
        {
            debugRecoveryRetryCountForCurrentHash = 0;
            debugRecoverySuppressedForCurrentHash = false;
        }

        if (createdHash != debugLastReportedCreatedEqHash ||
            appliedHash != debugLastReportedAppliedEqHash ||
            dspReady != debugLastReportedDspReady)
        {
            debugLastReportedCreatedEqHash = createdHash;
            debugLastReportedAppliedEqHash = appliedHash;
            debugLastReportedDspReady = dspReady;
            diagLog("[VERIFY] EQ reflection createdHash=0x"
                + juce::String::toHexString(static_cast<juce::int64>(createdHash))
                + " appliedHash=0x"
                + juce::String::toHexString(static_cast<juce::int64>(appliedHash))
                + " matched=" + juce::String((int)(createdHash == appliedHash))
                + " ver=" + juce::String((int)observedVersion)
                + " blocksSinceCreate=" + juce::String(static_cast<juce::int64>(processedBlocksSinceCreate))
                + " dspReady=" + juce::String(dspReady)
                + " coordFading=" + juce::String(coordIsFading)
                + " updRet=" + juce::String(updateFadeReturned)
                + " fromNull=" + juce::String(fromNull)
                + " toNull=" + juce::String(toNull));
        }
    }

    if (!shutdownInProgress.load(std::memory_order_acquire)
        && deferredStructuralRebuildPending_.load(std::memory_order_acquire))
    {
        const int64_t dueTicks = deferredStructuralRebuildDueTicks_.load(std::memory_order_acquire);
        const int64_t nowTicks = juce::Time::getHighResolutionTicks();

        if (dueTicks > 0 && nowTicks >= dueTicks)
        {
            deferredStructuralRebuildPending_.store(false, std::memory_order_release);
            deferredStructuralRebuildDueTicks_.store(0, std::memory_order_release);

            if (uiConvolverProcessor.isIRLoaded())
            {
                diagLog("[DIAG] timerCallback: issuing deferred Structural rebuild after prepared IR apply");
                requestRebuild(convo::RebuildKind::Structural);

                ++pendingIRGeneration;
                setIRChangeFlag();

                const LearningCommand cmd {
                    LearningCommand::Type::IRChanged,
                    false,
                    pendingLearningMode.load(std::memory_order_acquire),
                    pendingIRGeneration
                };

                if (!enqueueLearningCommand(cmd))
                {
                    DBG("[AudioEngine] timerCallback: deferred command queue overflow");
                }
            }
        }
    }

    if (!shutdownInProgress.load(std::memory_order_acquire)
        && deferredFinalizeAwareRebuildPending_.load(std::memory_order_acquire))
    {
        const int queuedGeneration = rebuildGeneration.load(std::memory_order_acquire);
        const int committedGeneration = lastCommittedRebuildGeneration.load(std::memory_order_acquire);
        const bool outstandingRebuild = queuedGeneration > committedGeneration;
        const bool irLoaded = uiConvolverProcessor.isIRLoaded();
        const bool irFinalized = uiConvolverProcessor.isIRFinalized();
        const bool irLoading = uiConvolverProcessor.isLoadingIR();
        const bool structuralDeferred = deferredStructuralRebuildPending_.load(std::memory_order_acquire);
        const bool pendingIrChange = m_pendingIRChange.load(std::memory_order_acquire);

        // IR 遷移が完全に落ち着いてから 1 回だけ再構築を発火する。
        if ((!irLoaded || irFinalized)
            && !irLoading
            && !structuralDeferred
            && !pendingIrChange
            && !outstandingRebuild)
        {
            deferredFinalizeAwareRebuildPending_.store(false, std::memory_order_release);

            const double sr = currentSampleRate.load(std::memory_order_acquire);
            if (!m_isRestoringState && sr > 0.0)
            {
                diagLog("[DIAG] timerCallback: issuing deferred finalize-aware rebuild");
                requestRebuild(sr, maxSamplesPerBlock.load(std::memory_order_acquire));
            }
        }
    }

    processLearningCommands();
    processDeferredLearningActions();

    if (!shutdownInProgress.load(std::memory_order_acquire) &&
        sanitizeRawPtr(fadingOutDSP.load(std::memory_order_acquire)) == nullptr &&
        !dspCrossfadePending.load(std::memory_order_acquire) &&
        fadeQueued.exchange(false, std::memory_order_acq_rel))
    {
        if (auto* queued = sanitizeRawPtr(queuedOldDSP.exchange(nullptr, std::memory_order_acq_rel)))
        {
            const double fadeSec = queuedNextFadeTimeSec.load(std::memory_order_acquire);
            queuedFadeTimeSec.store(fadeSec, std::memory_order_release);
            fadingOutDSP.store(queued, std::memory_order_release);
            dspCrossfadePending.store(true, std::memory_order_release);
            setIRChangeFlag();
        }
    }

    // Grace period に基づく安全なリリース遅延は、
    // RCU v17.15 では ReclaimerThread が自動的に担当するため不要。
    // processDeferredReleases(); // 削除

    const auto& view = getActiveView();
    if (view.previousValid && view.alpha >= 1.0f)
    {
        sendChangeMessage();
    }

    // 内部プロセッサのクリーンアップを実行する。
    if (auto* dsp = currentDSP.load(std::memory_order_acquire))
    {
        dsp->eq.cleanup();
        dsp->convolver.cleanup();

        const bool activeFixed4Tap = (dsp->noiseShaperType == NoiseShaperType::Fixed4Tap);
        const bool activeFixed15Tap = (dsp->noiseShaperType == NoiseShaperType::Fixed15Tap);
        const bool activeDitherEnabled = (dsp->ditherBitDepth > 0);

        if (activeFixed4Tap && activeDitherEnabled)
        {
            dsp->fixedNoiseShaper.setDiagnosticsWindowSamples(
                static_cast<uint32_t>(fixedNoiseWindowSamples.load(std::memory_order_relaxed)));

            const uint32 now = juce::Time::getMillisecondCounter();
            const uint32 intervalMs = static_cast<uint32>(
                std::max(250, fixedNoiseLogIntervalMs.load(std::memory_order_relaxed)));
            if ((now - fixedNoiseLastLogMs) >= intervalMs)
            {
                fixedNoiseLastLogMs = now;
                const auto diag = dsp->fixedNoiseShaper.getDiagnostics();
                if (diag.windowSamples > 0)
                {
                    DBG_LOG(
                        "[Fixed4Tap] bitDepth=" + juce::String(diag.bitDepth)
                        + " rmsL=" + juce::String(diag.rmsErrorL, 9)
                        + " rmsR=" + juce::String(diag.rmsErrorR, 9)
                        + " peak=" + juce::String(diag.peakAbsError, 9)
                        + " windowSamples=" + juce::String((int)diag.windowSamples));
                }
                else
                {
                    DBG_LOG(
                        "[Fixed4Tap] waiting for diagnostics window"
                        " (bitDepth=" + juce::String(dsp->ditherBitDepth)
                        + ", targetWindow=" + juce::String(fixedNoiseWindowSamples.load(std::memory_order_relaxed))
                        + ")");
                }
            }
        }
        else if (activeFixed15Tap && activeDitherEnabled)
        {
            dsp->fixed15TapNoiseShaper.setDiagnosticsWindowSamples(
                static_cast<uint32_t>(fixedNoiseWindowSamples.load(std::memory_order_relaxed)));

            const uint32 now = juce::Time::getMillisecondCounter();
            const uint32 intervalMs = static_cast<uint32>(
                std::max(250, fixedNoiseLogIntervalMs.load(std::memory_order_relaxed)));
            if ((now - fixedNoiseLastLogMs) >= intervalMs)
            {
                fixedNoiseLastLogMs = now;
                const auto diag = dsp->fixed15TapNoiseShaper.getDiagnostics();
                if (diag.windowSamples > 0)
                {
                    DBG_LOG(
                        "[Fixed15Tap] bitDepth=" + juce::String(diag.bitDepth)
                        + " rmsL=" + juce::String(diag.rmsErrorL, 9)
                        + " rmsR=" + juce::String(diag.rmsErrorR, 9)
                        + " peak=" + juce::String(diag.peakAbsError, 9)
                        + " windowSamples=" + juce::String((int)diag.windowSamples));
                }
                else
                {
                    DBG_LOG(
                        "[Fixed15Tap] waiting for diagnostics window"
                        " (bitDepth=" + juce::String(dsp->ditherBitDepth)
                        + ", targetWindow=" + juce::String(fixedNoiseWindowSamples.load(std::memory_order_relaxed))
                        + ")");
                }
            }
        }
    }

    // UI用プロセッサのクリーンアップ
    uiEqEditor.cleanup();
    uiConvolverProcessor.cleanup();
}

void AudioEngine::changeListenerCallback(juce::ChangeBroadcaster* source)
{
    if (source == &uiEqEditor)
    {
        // UI (SpectrumAnalyzerComponent など) が EQ 編集を即時反映できるよう通知する。
        // 実 DSP 反映は従来どおり requestRebuild() 経由で行う。
        sendChangeMessage();
        requestRebuild(convo::RebuildKind::Structural);
    }
}

void AudioEngine::convolverParamsChanged(ConvolverProcessor* processor)
{
    if (processor == &uiConvolverProcessor)
    {
        const bool suppressIntermediateMixedPhasePublish =
            uiConvolverProcessor.isProgressiveUpgradeEnabled()
            && uiConvolverProcessor.getPhaseMode() == ConvolverProcessor::PhaseMode::Mixed
            && uiConvolverProcessor.getActiveCacheFFTSize() > 0
            && uiConvolverProcessor.getActiveCacheFFTSize() < uiConvolverProcessor.getTargetUpgradeFFTSize();

        if (suppressIntermediateMixedPhasePublish)
        {
            diagLog("[DIAG] convolverParamsChanged: SUPPRESSED intermediate progressive mixed-phase publish fft="
                    + juce::String(uiConvolverProcessor.getActiveCacheFFTSize())
                    + " targetFFT=" + juce::String(uiConvolverProcessor.getTargetUpgradeFFTSize())
                    + " irName=" + uiConvolverProcessor.getIRName());
            return;
        }

        bool needsStructuralRebuild = false;
        double srForRebuild = 0.0;
        uint64_t uiStructuralHash = 0;
        bool uiHasIrForRebuild = false;
        bool dspHasIrForRebuild = false;

        {
            std::lock_guard<std::mutex> lk(rebuildMutex);
            diagLog("[DIAG] convolverParamsChanged: enter");
            if (activeDSP)
            {
                activeDSP->convolver.syncParametersFrom(uiConvolverProcessor);

                const bool uiHasIr = uiConvolverProcessor.isIRLoaded();
                const bool dspHasIr = activeDSP->convolver.isIRLoaded();
                uiHasIrForRebuild = uiHasIr;
                dspHasIrForRebuild = dspHasIr;

                if (uiHasIr)
                    uiStructuralHash = uiConvolverProcessor.getStructuralHash();

                needsStructuralRebuild = (uiHasIr != dspHasIr);

                if (!needsStructuralRebuild && uiHasIr)
                {
                    needsStructuralRebuild =
                        activeDSP->convolver.getIRName() != uiConvolverProcessor.getIRName()
                     || activeDSP->convolver.getIRLength() != uiConvolverProcessor.getIRLength()
                     || activeDSP->convolver.getPhaseMode() != uiConvolverProcessor.getPhaseMode()
                     || activeDSP->convolver.getExperimentalDirectHeadEnabled() != uiConvolverProcessor.getExperimentalDirectHeadEnabled()
                     || std::abs(activeDSP->convolver.getTargetIRLength() - uiConvolverProcessor.getTargetIRLength()) > 0.001f;
                }
            }
            else
            {
                needsStructuralRebuild = uiConvolverProcessor.isIRLoaded();
                uiHasIrForRebuild = needsStructuralRebuild;
                dspHasIrForRebuild = false;
                if (needsStructuralRebuild)
                    uiStructuralHash = uiConvolverProcessor.getStructuralHash();
            }

            if (needsStructuralRebuild)
                srForRebuild = currentSampleRate.load(std::memory_order_acquire);
        }

        // 同一構造ハッシュで再通知が来ても、重い Structural rebuild を再発火させない。
        // これにより IR 読み込み後の rebuild 連鎖（CPU スパイク）を抑止する。
        if (needsStructuralRebuild && uiStructuralHash != 0)
        {
            const uint64_t prevHash = lastIssuedConvolverStructuralHash_.load(std::memory_order_acquire);
            if (prevHash == uiStructuralHash)
            {
                diagLog("[DIAG] convolverParamsChanged: BLOCKED by hash dedup hash="
                    + juce::String::toHexString((int64_t) uiStructuralHash));
                needsStructuralRebuild = false;
            }
            else
                lastIssuedConvolverStructuralHash_.store(uiStructuralHash, std::memory_order_release);
                        diagLog("[DIAG] convolverParamsChanged: requestRebuild Structural hash="
                            + juce::String::toHexString((int64_t) uiStructuralHash)
                            + " irName=" + uiConvolverProcessor.getIRName());
        }

        if (needsStructuralRebuild && uiHasIrForRebuild && !dspHasIrForRebuild)
        {
            const int64_t nowTicks = juce::Time::getHighResolutionTicks();
            const int64_t appliedTicks = uiConvolverProcessor.getLastPreparedIRApplyTicks();
            const int64_t minDeltaTicks = juce::Time::getHighResolutionTicksPerSecond() / 5; // 200ms

            if (appliedTicks > 0 && (nowTicks - appliedTicks) < minDeltaTicks)
            {
                deferredStructuralRebuildPending_.store(true, std::memory_order_release);
                deferredStructuralRebuildDueTicks_.store(appliedTicks + minDeltaTicks, std::memory_order_release);
                pendingRebuildMask_.fetch_and(~convo::toMask(convo::RebuildKind::Structural), std::memory_order_acq_rel);
                needsStructuralRebuild = false;

                diagLog("[DIAG] convolverParamsChanged: DEFERRED Structural rebuild after prepared IR apply and cleared pending Structural bit");
            }
        }

        if (needsStructuralRebuild)
        {
            requestRebuild(convo::RebuildKind::Structural);
        }

        if (needsStructuralRebuild && srForRebuild > 0.0)
        {
            ++pendingIRGeneration;
            setIRChangeFlag();

            const LearningCommand cmd {
                LearningCommand::Type::IRChanged,
                false,
                pendingLearningMode.load(std::memory_order_acquire),
                pendingIRGeneration
            };

            if (!enqueueLearningCommand(cmd))
            {
                DBG("[AudioEngine] convolverParamsChanged: command queue overflow");
            }
        }
    }
}

//--------------------------------------------------------------
// releaseResources
// デバイス停止時に呼ばれる（Audio Thread停止後）
// JUCE v8.0.12 完全対応版（MMCSSはJUCEが自動管理）
//--------------------------------------------------------------
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
                deleteDSP(staging.newDSP);
            if (staging.oldDSP)
                deleteDSP(staging.oldDSP);
        }
    }

    if (activeToRelease)
        deleteDSP(activeToRelease);
    if (fadingToRelease)
        deleteDSP(fadingToRelease);
    if (queuedToRelease)
        deleteDSP(queuedToRelease);
    if (pendingNewToRelease)
        deleteDSP(pendingNewToRelease);
    if (pendingCurrentToRelease)
        deleteDSP(pendingCurrentToRelease);

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

//--------------------------------------------------------------
// getNextAudioBlock - オーディオ処理コールバック (Audio Thread)
// リアルタイム制約 (Real-time Constraints)
//    1. メモリ割り当て禁止 (No memory allocation): new, malloc, vector::resize, AudioBuffer::setSize 等はNG。
//    2. ロック禁止 (No locks): Mutex, CriticalSection 等によるブロックはNG。
//    3. システムコール禁止 (No system calls): ファイルI/O, コンソール出力(printf) 等はNG。
//    4. 待機禁止 (No waiting): sleep や 重い計算によるストールを避ける。IRの再ロードもNG。
//    5. 禁止API: AudioBlock::allocate, AudioBlock::copyFrom (確保伴うもの), FFT::performFrequencyOnlyForwardTransform (事前確保なしはNG)
//    6. std::vector使用時は、必ず AudioBuffer / 生ポインタを wrap する形で使用すること。
//    7. MMCSS設定禁止: AvSetMmThreadCharacteristics 等の呼び出しは禁止。
//--------------------------------------------------------------
void AudioEngine::getNextAudioBlock (const juce::AudioSourceChannelInfo& bufferToFill)
{
    const juce::ScopedNoDenormals noDenormals;
    m_audioBlockCounter.fetch_add(1, std::memory_order_release);

    // 入力検証
    if (bufferToFill.buffer == nullptr)
        return;

    const int numSamples = bufferToFill.numSamples;
    const int startSample = bufferToFill.startSample;
    auto* buffer = bufferToFill.buffer;

    constexpr int ABSOLUTE_MAX_BLOCK_SIZE = 1 << 20;
    if (numSamples <= 0 || numSamples > ABSOLUTE_MAX_BLOCK_SIZE)
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    if (startSample < 0 || startSample + numSamples > buffer->getNumSamples())
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    // ========================================================================
    // ダブルバッファモデル：スロット取得（lock-free）
    // ========================================================================
    int idx = m_activeIndex.load(std::memory_order_acquire);
    const convo::EngineView& view = m_views[idx];

    const convo::EngineState& cur  = view.current;
    const convo::EngineState& prev = view.previous;
    const float alpha = view.alpha;

    // ========================================================================
    // フェード処理または通常処理
    // ========================================================================
    if (view.previousValid && alpha < 1.0f)
    {
        // クロスフェード中：prev と cur をブレンド
        m_tmpA.clear();
        m_tmpB.clear();

        processWithState(m_tmpA, prev, 0, numSamples, 1.0f - alpha);
        processWithState(m_tmpB, cur,  0, numSamples, alpha);

        const int numChannels = buffer->getNumChannels();
        for (int ch = 0; ch < numChannels; ++ch)
        {
            buffer->addFrom(ch, startSample, m_tmpA, ch, 0, numSamples);
            buffer->addFrom(ch, startSample, m_tmpB, ch, 0, numSamples);
        }
    }
    else
    {
        // 通常処理：current のみ
        processWithState(*buffer, cur, startSample, numSamples, 1.0f);
    }
}

// ============================================================================
// CONTROL THREAD: 状態構築ヘルパー
// ============================================================================

convo::EngineState AudioEngine::buildCurrentState() noexcept 
{
    convo::EngineState state;
    
    // 1. ConvolverProcessor の状態をシリアライズ
    if (auto* convProc = uiConvolverProcessor.getConvolverState())
    {
        // ConvolverState をバイト列にシリアライズ
        // TODO: 実際のシリアライズロジックを実装
        // 例：convProc->serializeTo(state.dspBlob, convo::EngineState::kDSPCoreSize);
        std::memset(state.dspBlob, 0, sizeof(state.dspBlob));
    }
    
    // 2. EQEditor の状態をシリアライズ
    if (const auto* eqState = uiEqEditor.getEQStateSnapshot())
    {
        // EQParameters または EQState をバイト列にシリアライズ
        // TODO: 実際のシリアライズロジックを実装
        // 例：eqState->serializeTo(state.eqBlob, convo::EngineState::kEQStateSize);
        std::memset(state.eqBlob, 0, sizeof(state.eqBlob));
    }
    
    // 3. スナップショット情報（ノイズシェイパー係数など）をシリアライズ
    // TODO: 実際のシリアライズロジックを実装
    std::memset(state.snapBlob, 0, sizeof(state.snapBlob));
    
    state.generation++;
    state.isValid = true;
    
    return state;
}

// ============================================================================
// CONTROL THREAD: 唯一の書き込みパス
// ============================================================================

void AudioEngine::publishEngineState(convo::EngineState&& newState, float fadeTimeSec) 
{
    // CRITICAL: Single-Writer Guarantee (この関数は単一スレッドからのみ呼ばれる前提)
    
    // 1. 非アクティブスロットの取得
    int active = m_activeIndex.load(std::memory_order_acquire);
    int write  = 1 - active;
    
    convo::EngineView& dst = m_views[write];
    const convo::EngineView& src = m_views[active];
    
    // 2. previous 状態のセットアップ (フェードありの場合)
    if (fadeTimeSec > 0.0f) 
    {
        if (src.previousValid) 
        {
            dst.previous.copyFrom(src.previous);
            dst.previousValid = true;
        } 
        else 
        {
            dst.previous.copyFrom(src.current);
            dst.previousValid = true;
        }
        dst.alpha = 0.0f;
    } 
    else 
    {
        // フェードなし：即時切り替え
        dst.previousValid = false;
        dst.alpha = 1.0f;
    }
    
    // 3. current 状態の更新 (完全コピー)
    dst.current.copyFrom(newState);
    
    // 4. 公開 (Atomic Swap)
    m_activeIndex.exchange(write, std::memory_order_acq_rel);
}

void AudioEngine::advanceFade(float step) 
{
    // クロスフェード進行用 (Control Thread で定期的に呼ぶ)
    int active = m_activeIndex.load(std::memory_order_acquire);
    const convo::EngineView& src = m_views[active];
    
    // フェード中でない場合は何もしない
    if (!src.previousValid || src.alpha >= 1.0f) 
        return;
    
    int write  = 1 - active;
    convo::EngineView& dst = m_views[write];
    
    // current は不変なのでコピー
    dst.current.copyFrom(src.current);
    
    // alpha を線形補間
    float newAlpha = src.alpha + step;
    if (newAlpha >= 1.0f) 
    {
        dst.alpha = 1.0f;
        dst.previousValid = false;
    } 
    else 
    {
        dst.previous.copyFrom(src.previous);
        dst.previousValid = true;
        dst.alpha = newAlpha;
    }
    
    // 公開
    m_activeIndex.exchange(write, std::memory_order_release);
}

// ============================================================================
// AUDIO THREAD: DSP 処理実体
// ============================================================================

void AudioEngine::processWithState(juce::AudioBuffer<float>& output, 
                                   const convo::EngineState& state, 
                                   int startSample, 
                                   int numSamples, 
                                   float gain) 
{
    // 実際の DSP 処理パイプライン
    // state.dspBlob, state.eqBlob, state.snapBlob を参照して処理を行う
    
    // 1. ゲイン適用（フェード用）
    if (gain != 1.0f && gain > 0.0f) 
    {
        for (int ch = 0; ch < output.getNumChannels(); ++ch) 
        {
            output.applyGain(ch, startSample, numSamples, gain);
        }
    }
    
    // 2. Convolver 処理
    // TODO: state.dspBlob から ConvolverState をデシリアライズし、処理を実行
    // 例：ConvolverProcessor::process(output, state.dspBlob, startSample, numSamples);
    
    // 3. EQ 処理
    // TODO: state.eqBlob から EQState をデシリアライズし、処理を実行
    // 例：EQProcessor::process(output, state.eqBlob, startSample, numSamples);
    
    // 4. Noise Shaper / Dither 処理
    // TODO: state.snapBlob からパラメータを読み取り、処理を実行
    
    // 注意：この関数は Audio Thread 内で呼ばれるため、malloc/new/lock は厳禁
}

