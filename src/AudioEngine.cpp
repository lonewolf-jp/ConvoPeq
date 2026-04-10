//============================================================================
// AudioEngine.cpp  ── v0.2 (JUCE 8.0.12対応)
//
// AudioEngineの実装
//============================================================================

#include <JuceHeader.h>
#include "AudioEngine.h"
#include "InputBitDepthTransform.h"
#include "OutputFilter.h"
#include "DeferredDeletionQueue.h"
#include "RefCountedDeferred.h"
#include "core/SnapshotAssembler.h"
#include <Windows.h>
#include <cmath>
#include <cstdint>
#include <complex>
#include <algorithm>
#include <mutex>
#include <condition_variable>
#include <xmmintrin.h>
#include <immintrin.h>

// ==================================================================
// 段階 2+3：thread_local スロット管理および定数
// ==================================================================
thread_local size_t tls_readerSlot = SIZE_MAX;
static constexpr size_t INVALID_SLOT = SIZE_MAX;
static constexpr uint64_t kIdleEpoch = 0;
static constexpr size_t RCU_BACKLOG_WARNING_THRESHOLD = 128;

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
DeferredDeletionQueue g_deletionQueue;
std::atomic<uint64_t> g_currentEpoch{1};

// ==================================================================
// RCU 実装
// ==================================================================

// -------------------------------------------------------------------
// スレッド登録
// -------------------------------------------------------------------
size_t AudioEngine::registerReader()
{
    size_t slot = nextReaderSlot.fetch_add(1, std::memory_order_relaxed);
    if (slot >= MAX_READERS) {
        jassertfalse;
        slot = MAX_READERS - 1;
    }
    readerEpochs[slot].store(0, std::memory_order_relaxed);
    return slot;
}

void AudioEngine::unregisterReader(size_t slot)
{
    if (slot < MAX_READERS) {
        readerEpochs[slot].store(kIdleEpoch, std::memory_order_release);
    }
}

void AudioEngine::updateReaderEpoch(size_t slot, uint64_t epoch)
{
    if (slot < MAX_READERS) {
        readerEpochs[slot].store(epoch, std::memory_order_release);
    }
}

void AudioEngine::processDeferredReleases()
{
    // B22: 旧 SPSC キューの重複処理を完全に削除し、MPMC g_deletionQueue に一本化。
    // この関数は Timer コールバックから呼ばれる（非 RT スレッド）
    const uint64_t minEpoch = getMinReaderEpoch();

    // 最後にクリーンアップした epoch から進んでいない場合はスキップ
    if (minEpoch <= lastReclaimedEpoch)
        return;

    g_deletionQueue.reclaim(minEpoch);
    lastReclaimedEpoch = minEpoch;
}

// ==================================================================
// 段階 2+3 実装（追加）
// ==================================================================

size_t AudioEngine::getOrRegisterCurrentThreadSlot()
{
    if (tls_readerSlot != INVALID_SLOT)
        return tls_readerSlot;

    size_t slot = registerReader();
    if (slot >= MAX_READERS) {
        jassertfalse;
        tls_readerSlot = INVALID_SLOT;
        return INVALID_SLOT;
    }
    tls_readerSlot = slot;
    return slot;
}

void AudioEngine::enterReader(size_t slot, uint64_t epoch)
{
    if (slot >= MAX_READERS) return;

    // Publish epoch with release semantics
    readerEpochs[slot].store(epoch, std::memory_order_release);
    // ★ 強化：seq_cst fence により publish 以降のロードが store より後に見えることを完全保証
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

void AudioEngine::exitReader(size_t slot)
{
    if (slot >= MAX_READERS) return;
    readerEpochs[slot].store(kIdleEpoch, std::memory_order_release);
}

uint64_t AudioEngine::getMinReaderEpoch() const noexcept
{
    uint64_t minEpoch = std::numeric_limits<uint64_t>::max();
    bool hasActiveReader = false;

    for (size_t i = 0; i < MAX_READERS; ++i) {
        const uint64_t e = readerEpochs[i].load(std::memory_order_acquire);
        if (e != kIdleEpoch) {
            hasActiveReader = true;
            if (SafeStateSwapper::isOlder(e, minEpoch))
                minEpoch = e;
        }
    }

    // Audio Thread の epoch も考慮
    const uint64_t ae = audioThreadEpoch.load(std::memory_order_acquire);
    if (ae != kIdleEpoch) {
        hasActiveReader = true;
        if (SafeStateSwapper::isOlder(ae, minEpoch))
            minEpoch = ae;
    }

    if (!hasActiveReader)
        return globalEpoch.load(std::memory_order_acquire);

    return minEpoch;
}

// epoch 比較ヘルパー（isOlder）はヘッダ内でインライン定義済み

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

struct AudioEngine::DSPCore; // Forward declaration

// Padé近似による高速tanh (std::exp回避)
// 精度: |x| < 3.0 で誤差 1e-4 以下
static inline double fastTanh(double x) noexcept
{
    if (x >= 3.0) return 1.0;
    if (x <= -3.0) return -1.0;
    const double x2 = x * x;
    return x * (27.0 + x2) / (27.0 + 9.0 * x2);
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
    const __m256d v27          = _mm256_set1_pd(27.0);
    const __m256d v9           = _mm256_set1_pd(9.0);
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
        __m256d num      = _mm256_mul_pd(arg, _mm256_add_pd(v27, arg2));
        __m256d den      = _mm256_add_pd(v27, _mm256_mul_pd(v9, arg2));
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
    : m_workerThread(m_commandBuffer, m_coordinator, m_generationManager)
{
    initialiseAdaptiveCoeffBanks();
    selectAdaptiveCoeffBankForCurrentSettings();

    initialiseThreadAffinityMasks();
    noiseShaperLearner = std::make_unique<NoiseShaperLearner>(*this, audioCaptureQueue);
    noiseShaperLearner->setLearningMode(pendingLearningMode.load(std::memory_order_acquire));

    // デフォルトサンプルレート (0 = 未初期化/デバイスなし)
    currentSampleRate.store(0.0);

    // バッファ初期化
    audioFifoBuffer.setSize (2, FIFO_SIZE);
    currentDSP.store(nullptr);
    // 段階 2+3：globalEpoch を 1 から開始（0 は kIdleEpoch と区別するため）
    globalEpoch.store(1, std::memory_order_relaxed);
    dspCrossfadeGain.reset(48000.0, 0.03);
    dspCrossfadeGain.setCurrentAndTargetValue(1.0);

    m_currentInputHeadroomDb.store(inputHeadroomDb.load(std::memory_order_relaxed), std::memory_order_relaxed);
    m_currentOutputMakeupDb.store(outputMakeupDb.load(std::memory_order_relaxed), std::memory_order_relaxed);
    m_currentConvInputTrimDb.store(convolverInputTrimDb.load(std::memory_order_relaxed), std::memory_order_relaxed);
    m_currentEqBypass.store(eqBypassRequested.load(std::memory_order_relaxed), std::memory_order_relaxed);
    m_currentConvBypass.store(convBypassRequested.load(std::memory_order_relaxed), std::memory_order_relaxed);
    m_currentProcessingOrder.store(currentProcessingOrder.load(std::memory_order_relaxed), std::memory_order_relaxed);
    m_currentSoftClipEnabled.store(softClipEnabled.load(std::memory_order_relaxed), std::memory_order_relaxed);
    m_currentSaturationAmount.store(saturationAmount.load(std::memory_order_relaxed), std::memory_order_relaxed);
    m_currentOversamplingFactor.store(manualOversamplingFactor.load(std::memory_order_relaxed), std::memory_order_relaxed);
    m_currentOversamplingType.store(oversamplingType.load(std::memory_order_relaxed), std::memory_order_relaxed);
    m_currentDitherBitDepth.store(ditherBitDepth.load(std::memory_order_relaxed), std::memory_order_relaxed);
    m_currentNoiseShaperType.store(noiseShaperType.load(std::memory_order_relaxed), std::memory_order_relaxed);

    // Phase 2/3: 初期スナップショットを生成して SnapshotCoordinator に登録する
    {
        convo::SnapshotParams initParams;
        initParams.inputHeadroomGain = inputHeadroomGain.load(std::memory_order_relaxed);
        initParams.outputMakeupGain  = outputMakeupGain.load(std::memory_order_relaxed);
        initParams.convInputTrimGain = convolverInputTrimGain.load(std::memory_order_relaxed);
        initParams.eqBypass          = eqBypassRequested.load(std::memory_order_relaxed);
        initParams.convBypass        = convBypassRequested.load(std::memory_order_relaxed);
        initParams.processingOrder   = static_cast<convo::ProcessingOrder>(
                                           static_cast<int>(currentProcessingOrder.load(std::memory_order_relaxed)));
        initParams.softClipEnabled   = softClipEnabled.load(std::memory_order_relaxed);
        initParams.saturationAmount  = saturationAmount.load(std::memory_order_relaxed);
        initParams.oversamplingType  = static_cast<convo::OversamplingType>(
                                           static_cast<int>(oversamplingType.load(std::memory_order_relaxed)));
        initParams.oversamplingFactor = manualOversamplingFactor.load(std::memory_order_relaxed);
        initParams.ditherBitDepth    = ditherBitDepth.load(std::memory_order_relaxed);
        initParams.noiseShaperType   = static_cast<convo::NoiseShaperType>(
                                           static_cast<int>(noiseShaperType.load(std::memory_order_relaxed)));
        initParams.generation        = m_generationManager.bumpGeneration();
        const convo::GlobalSnapshot* initSnap = convo::SnapshotFactory::create(initParams);
        m_coordinator.switchImmediate(initSnap);
    }
}



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

void AudioEngine::initialiseThreadAffinityMasks() noexcept
{
    DWORD_PTR processAffinityMask = 0;
    DWORD_PTR systemAffinityMask = 0;

    if (!::GetProcessAffinityMask(::GetCurrentProcess(), &processAffinityMask, &systemAffinityMask))
        return;

    DWORD_PTR firstMask = 0;
    DWORD_PTR secondMask = 0;

    for (DWORD_PTR bit = 1; bit != 0; bit <<= 1)
    {
        if ((processAffinityMask & bit) == 0)
            continue;

        if (firstMask == 0)
            firstMask = bit;
        else
        {
            secondMask = bit;
            break;
        }
    }

    audioThreadAffinityMask = static_cast<std::uintptr_t>(firstMask);
    noiseLearnerThreadAffinityMask = static_cast<std::uintptr_t>(secondMask != 0 ? secondMask : firstMask);
    const DWORD_PTR nonAudioMask = processAffinityMask & ~firstMask;
    nonAudioThreadAffinityMask = static_cast<std::uintptr_t>(nonAudioMask != 0 ? nonAudioMask : (secondMask != 0 ? secondMask : firstMask));
}

void AudioEngine::pinCurrentThreadToNoiseLearnerCoreIfNeeded() const noexcept
{
    if (noiseLearnerThreadAffinityMask == 0)
        return;

    ::SetThreadAffinityMask(::GetCurrentThread(), static_cast<DWORD_PTR>(noiseLearnerThreadAffinityMask));
}

void AudioEngine::pinCurrentThreadToNonAudioCoresIfNeeded() const noexcept
{
    static thread_local bool isPinned = false;
    if (isPinned)
        return;

    if (nonAudioThreadAffinityMask == 0)
        return;

    ::SetThreadAffinityMask(::GetCurrentThread(), static_cast<DWORD_PTR>(nonAudioThreadAffinityMask));
    isPinned = true;
}

void AudioEngine::initialize()
{
    // ==================================================================
    // 段階 1：RCU 基盤の初期化
    // ==================================================================
    // B22: 旧 SPSC キュー (queueWrite/queueRead/overflowList) は廃止。
    //      g_deletionQueue は静的初期化される。
    // readerEpochs と globalEpoch は静的初期化で 0

    // Start worker thread
    rebuildThread = std::thread(&AudioEngine::rebuildThreadLoop, this);

    // 初期DSP構築 (デフォルト設定)
    // 安全対策: バッファサイズを余裕を持って確保 (SAFE_MAX_BLOCK_SIZE)
    // これにより、デバイス初期化前やバッファサイズ変更時の不整合による音切れ/無音を防ぐ
    requestRebuild(48000.0, SAFE_MAX_BLOCK_SIZE);
    maxSamplesPerBlock.store(SAFE_MAX_BLOCK_SIZE);
    currentSampleRate.store(48000.0);

    // オーディオデバイスがまだ開始していない段階でも、IRロード側には実用的な既定値を渡す。
    // SAFE_MAX_BLOCK_SIZE をそのまま使うと不要に巨大な一時NUCを組んでメモリ使用量が跳ねるため、
    // ローダー用の暫定値は一般的な 48kHz / 512samples に固定する。
    uiConvolverProcessor.prepareToPlay(48000.0, 512);

    uiConvolverProcessor.addChangeListener(this);
    uiEqProcessor.addChangeListener(this);
    uiConvolverProcessor.addListener(this);
    uiEqProcessor.addListener(this);

    // タイマー開始 (100ms間隔)
    // - DSP再構築リクエストのポーリング (Audio Threadからの依頼を処理)
    // - ガベージコレクション
    startTimer(100);

    m_workerThread.setSnapshotCreator(&AudioEngine::onSnapshotRequired, this);
    initWorkerThread();
}

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
    if (self != nullptr)
        self->createSnapshotFromCurrentState(generation);
}

void AudioEngine::createSnapshotFromCurrentState(uint64_t generation)
{
    const ConvolverState* convState = uiConvolverProcessor.getConvolverState();

    convo::EQParameters eqParams;
    if (const auto* eqState = uiEqProcessor.getEQStateSnapshot())
        eqParams = eqState->toEQParameters();

    std::array<double, 9> nsCoeffs{};
    for (int i = 0; i < kAdaptiveNoiseShaperOrder; ++i)
        nsCoeffs[static_cast<size_t>(i)] = kDefaultAdaptiveNoiseShaperCoeffs[static_cast<size_t>(i)];

    const double inputHeadroomGainValue = juce::Decibels::decibelsToGain(static_cast<double>(m_currentInputHeadroomDb.load(std::memory_order_relaxed)));
    const double outputMakeupGainValue = juce::Decibels::decibelsToGain(static_cast<double>(m_currentOutputMakeupDb.load(std::memory_order_relaxed)));
    const double convInputTrimGainValue = juce::Decibels::decibelsToGain(static_cast<double>(m_currentConvInputTrimDb.load(std::memory_order_relaxed)));
    const bool convBypass = m_currentConvBypass.load(std::memory_order_relaxed);
    const bool eqBypass = m_currentEqBypass.load(std::memory_order_relaxed);
    const bool softClip = m_currentSoftClipEnabled.load(std::memory_order_relaxed);
    const float satAmount = m_currentSaturationAmount.load(std::memory_order_relaxed);
    const convo::ProcessingOrder order = m_currentProcessingOrder.load(std::memory_order_relaxed);
    const convo::OversamplingType osType = m_currentOversamplingType.load(std::memory_order_relaxed);
    const int osFactor = m_currentOversamplingFactor.load(std::memory_order_relaxed);
    const int bitDepth = m_currentDitherBitDepth.load(std::memory_order_relaxed);
    const convo::NoiseShaperType nsType = m_currentNoiseShaperType.load(std::memory_order_relaxed);

    convo::SnapshotParams params = convo::SnapshotAssembler::assemble(
        convState,
        eqParams,
        nsCoeffs,
        inputHeadroomGainValue,
        outputMakeupGainValue,
        convInputTrimGainValue,
        convBypass,
        eqBypass,
        softClip,
        satAmount,
        order,
        osType,
        osFactor,
        bitDepth,
        nsType,
        generation);

    const convo::GlobalSnapshot* newSnap = convo::SnapshotFactory::create(params);
    m_coordinator.switchImmediate(newSnap);
}



AudioEngine::~AudioEngine()
{
    shutdownWorkerThread();

    stopTimer();

    // 1. Stop Audio Thread access immediately
    // Prevent Audio Thread from acquiring new pointer.
    currentDSP.store(nullptr, std::memory_order_release);

    // 2. Stop worker thread
    // Ensure no new DSPs are being built or committed.
    rebuildThreadShouldExit.store(true);
    rebuildCV.notify_one();
    if (rebuildThread.joinable())
        rebuildThread.join();

    // 3. Remove listeners
    uiConvolverProcessor.removeChangeListener(this);
    uiEqProcessor.removeChangeListener(this);
    uiConvolverProcessor.removeListener(this);
    uiEqProcessor.removeListener(this);

    // 4. Release Active DSP
    if (activeDSP)
    {
        activeDSP->convolver.forceCleanup();
        activeDSP->release();
        activeDSP = nullptr;
    }

    if (auto* fading = fadingOutDSP.exchange(nullptr, std::memory_order_acq_rel))
        fading->release();

    // 5. RCU リリースキューを最終解放する。
    //    ここでは Audio Thread / rebuildThread ともに停止済みのため、安全に全解放できる。
    // B22: 旧 SPSC キューの処理を削除し、g_deletionQueue の最終 reclaim を行う。
    const uint64_t finalEpoch = std::numeric_limits<uint64_t>::max();
    g_deletionQueue.reclaim(finalEpoch);
}

//--------------------------------------------------------------
// FIFOからデータ読み出し (UI Thread)
//--------------------------------------------------------------
void AudioEngine::readFromFifo(float* dest, int numSamples)
{
    // 単一のリーダーを保証: 複数のUIコンポーネント（アナライザーなど）からの同時読み出しを防ぐ。
	//
    // Note: 書き込み側 (DSPCore::pushToFifo / Audio Thread) はこのロックを使用しないため、ロックフリーです。
    //       したがって、ここでロックを取得してもオーディオスレッドをブロックする恐れはありません (Deadlock Free)。
    const juce::ScopedLock sl(fifoReadLock);

    int start1, size1, start2, size2;
    audioFifo.prepareToRead(numSamples, start1, size1, start2, size2);

    // 実際に読み取れるサンプル数を計算 (FIFO内の有効データ量に依存)
    // prepareToRead は numSamples 分の領域を返さない場合がある (FIFO不足時)
    const int actualRead = size1 + size2;
    const bool hasRightChannel = (audioFifoBuffer.getNumChannels() > 1);

    // AVX2 L+R 平均化ヘルパー
    auto mixToMono = [](const float* srcL, const float* srcR, float* dst, int n) noexcept
    {
        const __m256 half = _mm256_set1_ps(0.5f);
        int i = 0;
        const int vEnd = n / 8 * 8;
        for (; i < vEnd; i += 8)
        {
            __m256 vL  = _mm256_loadu_ps(srcL + i);
            __m256 vR  = _mm256_loadu_ps(srcR + i);
            __m256 avg = _mm256_mul_ps(_mm256_add_ps(vL, vR), half);
            _mm256_storeu_ps(dst + i, avg);
        }
        for (; i < n; ++i) dst[i] = (srcL[i] + srcR[i]) * 0.5f;
    };

    if (size1 > 0)
    {
        const float* srcL = audioFifoBuffer.getReadPointer(0, start1);
        const float* srcR = hasRightChannel ? audioFifoBuffer.getReadPointer(1, start1) : srcL;
        mixToMono(srcL, srcR, dest, size1);
    }

    if (size2 > 0)
    {
        const float* srcL = audioFifoBuffer.getReadPointer(0, start2);
        const float* srcR = hasRightChannel ? audioFifoBuffer.getReadPointer(1, start2) : srcL;
        mixToMono(srcL, srcR, dest + size1, size2);
    }

    // 実際に読み取った分だけFIFOを進める
    if (actualRead > 0)
        audioFifo.finishedRead(actualRead);

    // 足りない分はゼロ埋め (グリッチ防止)
    if (actualRead < numSamples)
        juce::FloatVectorOperations::clear(dest + actualRead, numSamples - actualRead);
}

//--------------------------------------------------------------
// FIFOからデータをスキップ (Latency対策)
//--------------------------------------------------------------
void AudioEngine::skipFifo(int numSamples)
{
    const juce::ScopedLock sl(fifoReadLock);
    int start1, size1, start2, size2;
    audioFifo.prepareToRead(numSamples, start1, size1, start2, size2);
    const int total = size1 + size2;
    if (total > 0)
        audioFifo.finishedRead(total);
}

//--------------------------------------------------------------
// EQ応答曲線計算
// 現在のEQ設定に基づき、周波数ごとのトータルゲイン応答（マグニチュード）を計算する
//--------------------------------------------------------------
void AudioEngine::calcEQResponseCurve(float* outMagnitudesL,
                                     float* outMagnitudesR,
                                     const std::complex<double>* zArray,
                                     int numPoints,
                                     double sampleRate)
{
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
    auto eqState = uiEqProcessor.getEQState();

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
            std::abs(params.gain) < EQ_GAIN_EPSILON)
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
    if (!uiEqProcessor.getAGCEnabled())
    {
        totalGainLinear = juce::Decibels::decibelsToGain(eqState->totalGainDb);
    }

    // ── 最適化: 有効なバンドがない、かつトータルゲインが0dBの場合は計算をスキップ ──
    if (numActiveBands == 0 && std::abs(totalGainLinear - 1.0f) < EQ_UNITY_GAIN_EPSILON)
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

//--------------------------------------------------------------
// getProcessingSampleRate
//--------------------------------------------------------------
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

double AudioEngine::estimateOversamplingLatencySamples(int oversamplingFactor,
                                                       OversamplingType oversamplingType,
                                                       double baseSampleRate) noexcept
{
    return estimateOversamplingLatencySamplesImpl(oversamplingFactor, oversamplingType, baseSampleRate);
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

//--------------------------------------------------------------
// prepareToPlay
//--------------------------------------------------------------
void AudioEngine::prepareToPlay (int samplesPerBlockExpected, double sampleRate)
{
    // パラメータ検証 (Parameter Validation)
    // 不正なパラメータから保護
    double safeSampleRate = sampleRate;
    if (safeSampleRate <= 0.0 || safeSampleRate > SAFE_MAX_SAMPLE_RATE || !std::isfinite(safeSampleRate))
    {
        jassertfalse; // デバッグビルドで警告
        safeSampleRate = 48000.0; // デフォルト値
    }

    if (samplesPerBlockExpected <= 0)
    {
        jassertfalse;
        samplesPerBlockExpected = 512; // フォールバックして続行
    }

    // ASIO同期: デバイスの実際のブロックサイズを使用する
    // SAFE_MAX_BLOCK_SIZEで固定すると、ASIOのブロックサイズ変更に追従できず破綻するため。
    const int bufferSize = samplesPerBlockExpected;

    // サンプルレート変更検知
    const bool rateChanged = (std::abs(currentSampleRate.load() - safeSampleRate) > 1e-6);
    // ブロックサイズ変更検知 (コンボルバーのパーティションサイズ最適化のため)
    const bool blockSizeChanged = (maxSamplesPerBlock.load() != bufferSize);

    maxSamplesPerBlock.store(bufferSize);
    currentSampleRate.store(safeSampleRate);
    dspCrossfadeGain.reset(safeSampleRate, 0.03);
    dspCrossfadeGain.setCurrentAndTargetValue(1.0);
    dspCrossfadePending.store(false, std::memory_order_release);
    selectAdaptiveCoeffBankForCurrentSettings();

    dspCrossfadeFloatBuffer.setSize(2, std::max(SAFE_MAX_BLOCK_SIZE, bufferSize), false, false, true);
    dspCrossfadeDoubleBuffer.setSize(2, std::max(SAFE_MAX_BLOCK_SIZE, bufferSize), false, false, true);


    audioFifo.reset();

    // レベルメーターのリセット
    inputLevelLinear.store(0.0f);
    outputLevelLinear.store(0.0f);

    // ===== bypass 状態の初期化 =====
    // 再生中のリアルタイムな更新は getNextAudioBlock() で行われる
    eqBypassActive.store (eqBypassRequested.load (std::memory_order_relaxed), std::memory_order_relaxed);
    convBypassActive.store (convBypassRequested.load (std::memory_order_relaxed), std::memory_order_relaxed);

    // [FIX] サンプルレートまたはブロックサイズが変わった場合、あるいはDSPが未初期化の場合、
    // ここでDSPを再ビルドする。
    // 以前はtimerCallback()が毎50msに再ビルドしていたが、それは誤りだった。
    // prepareToPlayはMessageThreadから呼ばれるため、requestRebuild()を直接呼べる。

    // [FIX] uiConvolverProcessor の currentBufferSize を必ず最新の bufferSize に同期する。
    // この呼び出しが欠けていると currentBufferSize == 0 のまま残り、
    // IR読み込み時に LoaderThread が blockSize=0 でMKLConvolver::setup(0,...) を呼んで
    // numPartitions = (irLen - 1) / 0 → ゼロ除算クラッシュが発生する。
    // prepareToPlay は Message Thread から呼ばれることが保証されているので安全。
    uiConvolverProcessor.prepareToPlay(safeSampleRate, bufferSize);

    if (rateChanged)
        uiConvolverProcessor.invalidatePendingLoads();

    if (rateChanged || blockSizeChanged || currentDSP.load(std::memory_order_acquire) == nullptr)
    {
        if (juce::MessageManager::getInstance()->isThisTheMessageThread())
        {
            requestRebuild(safeSampleRate, bufferSize);
        }
        else
        {
            // フォールバック: メッセージスレッド以外から呼ばれた場合は
            // フラグを立てtimerCallbackに委ねる (通常は到達しない)
            rebuildRequested.store(true, std::memory_order_release);
        }
    }
}

//--------------------------------------------------------------
// DSPCore Implementation
//--------------------------------------------------------------
AudioEngine::DSPCore::DSPCore() = default;

void AudioEngine::DSPCore::prepare(double newSampleRate, int samplesPerBlock, int bitDepth, int manualOversamplingFactor, OversamplingType oversamplingType, NoiseShaperType selectedNoiseShaperType)
{
    this->sampleRate = newSampleRate;
    this->noiseShaperType = selectedNoiseShaperType;

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

    // インターサンプルピーク用ブロック間状態をリセット
    softClipPrevSample[0] = 0.0;
    softClipPrevSample[1] = 0.0;
}

//--------------------------------------------------------------
// requestRebuild - DSPグラフの再構築 (Message Thread)
//--------------------------------------------------------------
void AudioEngine::requestRebuild(double sampleRate, int samplesPerBlock)
{
    // UIコンポーネント(uiEqProcessor等)へのアクセスやMKLメモリ確保を行うため、必ずMessage Threadで実行すること
    jassert (juce::MessageManager::getInstance()->isThisTheMessageThread());

    if (noiseShaperLearner && noiseShaperLearner->isRunning())
        noiseShaperLearner->stopLearning();

    // 新しいDSPコアを作成
    DSPCore* newDSP = new DSPCore();
    newDSP->convolver.setVisualizationEnabled(false); // DSP用は可視化データ不要

    // UIプロセッサから状態をコピー
    newDSP->eq.syncStateFrom(uiEqProcessor); // 最適化: ValueTreeを経由せず直接同期
    newDSP->convolver.syncStateFrom(uiConvolverProcessor);

    // キャプチャ用変数
    int ditherDepth = ditherBitDepth.load();
    int osFactor = manualOversamplingFactor.load();
    OversamplingType osType = oversamplingType.load();
    NoiseShaperType nsType = noiseShaperType.load();
    DSPCore* current = activeDSP; // 現在のアクティブDSPをキャプチャ
    if (current != nullptr)
        current->addRef();
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
    {
        std::lock_guard<std::mutex> lock(rebuildMutex);

        // Increment generation and create task inside the lock to ensure atomicity
        generation = ++rebuildGeneration;
        task.generation = generation;

        // If a task is already pending, move it out to be destroyed outside the lock.
        // This prevents holding the lock during a potentially slow DSPCore destruction.
        if (hasPendingTask)
        {
            dspToDestroy = pendingTask.newDSP;
            currentToRelease = pendingTask.currentDSP;
        }

        pendingTask = task;
        hasPendingTask = true;
    }
    rebuildCV.notify_one();

    // Destroy the orphaned DSP from the superseded task outside the lock.
    if (dspToDestroy)
        dspToDestroy->release();
    if (currentToRelease)
        currentToRelease->release();
}

void AudioEngine::rebuildThreadLoop()
{
    // Set denormal handling modes for this thread. This is crucial for performance
    // in MKL VML and AVX/SSE operations, which can be significantly slowed down
    // by subnormal numbers. This setting is thread-local.
    vmlSetMode(VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

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
                        ptr->release();
                }
            } dspGuard { task.newDSP };

            // Helper to check obsolescence
            const auto isObsolete = [&] {
                return isRebuildObsolete(task.generation) || rebuildThreadShouldExit.load();
            };

            if (isObsolete())
                continue;

            // 1. Prepare (メモリ確保)
            task.newDSP->prepare(task.sampleRate, task.samplesPerBlock, task.ditherDepth, task.manualOversamplingFactor, task.oversamplingType, task.noiseShaperType);

            if (isObsolete())
                continue;

            // 2. Reuse Logic
            //
            // 【task.currentDSP (= 旧 activeDSP) の安全性証明】
            //
            // task.currentDSP は requestRebuild() 側で addRef 済みであり、
            // rebuild タスクの寿命全体を通じて参照カウントで保持される。
            // activeDSP/currentDSP が差し替わっても、このローカル参照が残る限り
            // UAF は発生しない。
            //
            // 【shared_ptr 不採用の理由】
            //     Audio Thread は std::shared_ptr の参照カウント操作を禁止している
            //     (コーディング規約)。currentDSP の RCU 設計はこの制約に基づく。
            bool irReused = false;
            if (task.currentDSP)
            {
                // 既存のDSPからAGCの状態を引き継ぐ
                task.newDSP->eq.syncGlobalStateFrom(task.currentDSP->eq);

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
            if (! juce::MessageManager::callAsync([weakSelf = juce::WeakReference<AudioEngine>(this), newDSP = dspToCommit, generation = task.generation] {
                if (auto* self = weakSelf.get())
                {
                    self->commitNewDSP(newDSP, generation);
                }
                else
                {
                    // Engine is gone, delete the orphan DSP
                    newDSP->release();
                }
            }))
            {
                // MessageManager failed (e.g. shutting down), prevent leak
                dspToCommit->release();
            }
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
}

void AudioEngine::commitNewDSP(DSPCore* newDSP, int generation)
{
    DSPCore* dspToTrash = nullptr;
    uint64_t retireEpoch = 0;

    // Lock to ensure the check and commit are atomic with respect to new rebuild requests.
    {
        std::lock_guard<std::mutex> lock(rebuildMutex);

        // 古いリクエストの結果であれば破棄 (Race condition対策)
        if (generation != rebuildGeneration.load(std::memory_order_relaxed))
        {
            newDSP->release();
            return;
        }

        // 1. 旧 DSP を安全にキャプチャしてから新 DSP を公開する（段階 3）
        dspToTrash = activeDSP;

        // 2. Update the atomic raw pointer for the Audio Thread (Wait-free)
        currentDSP.store(newDSP, std::memory_order_release);

        // 3. 段階 3：エポックを進め、旧 DSP の retire epoch を記録する
        //    fetch_add は旧値を返すため、retireEpoch = 公開直前のグローバルエポック
        retireEpoch = globalEpoch.fetch_add(1, std::memory_order_acq_rel);
        g_currentEpoch.store(retireEpoch, std::memory_order_release);

        // 4. Take ownership of the new DSP
        activeDSP = newDSP;

        const uint64_t newSessionId = globalCaptureSessionId.fetch_add(1, std::memory_order_acq_rel) + 1;
        if (newDSP != nullptr)
            newDSP->currentCaptureSessionId = newSessionId;
    }

    // 5. RCU deferred release：旧 DSP を grace period 後に解放する
    if (dspToTrash != nullptr)
    {
        if (newDSP != nullptr && dspToTrash->oversamplingFactor != newDSP->oversamplingFactor)
        {
            dspToTrash->addRef();
            if (auto* oldFading = fadingOutDSP.exchange(dspToTrash, std::memory_order_acq_rel))
                oldFading->release();
            dspCrossfadePending.store(true, std::memory_order_release);
        }

        // 段階 3：RCU キューに登録し、grace period 経過後に processDeferredReleases が release() する
        dspToTrash->release();
    }

    if (newDSP != nullptr)
    {
        uiConvolverProcessor.setMixedPhaseState(newDSP->convolver.getMixedPhaseState());
    }

    const LearningCommand cmd {
        LearningCommand::Type::DSPReady,
        false,
        pendingLearningMode.load(std::memory_order_acquire),
        static_cast<uint64_t>(generation)
    };

    if (!enqueueLearningCommand(cmd))
    {
        DBG("[AudioEngine] commitNewDSP: command queue overflow");
    }
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
    processLearningCommands();
    processDeferredLearningActions();

    // Grace period に基づく安全なリリース遅延を実行する。
    processDeferredReleases();

    // UIプロセッサからの構造変更（プリセットロード、IRロードなど）を検知
    // タイマーコールバック内では、UIの状態を直接使用せずに、
    // Atomic変数から安全に読み取る
    const double sr = currentSampleRate.load();
    const int bs = maxSamplesPerBlock.load();

    // [FIX] rebuildRequestedフラグが立っているときだけ再ビルドを実行する。
    // 以前は毎50ms無条件でrequestRebuild()を呼んでいたため、
    // 新DSPのfadeInSamplesLeft(2048サンプル≈42ms)によるフェードが
    // 50ms周期でリセットされ続け、プチプチノイズの原因となっていた。
    if (sr > 0.0 && rebuildRequested.exchange(false, std::memory_order_acq_rel))
    {
        requestRebuild(sr, bs);
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
    uiEqProcessor.cleanup();
    uiConvolverProcessor.cleanup();
}

void AudioEngine::changeListenerCallback(juce::ChangeBroadcaster* source)
{
    // UIプロセッサからの構造変更（プリセットロード、IRロードなど）を検知
    if (source == &uiEqProcessor || source == &uiConvolverProcessor)
    {
        const double sr = currentSampleRate.load();
        if (sr <= 0.0) return;

        // DSPグラフを安全に再構築
        requestRebuild(sr, maxSamplesPerBlock.load());

        // UIに更新を通知 (MainWindowが受け取る)
        sendChangeMessage();
    }
}

void AudioEngine::eqBandChanged(EQProcessor* processor, int bandIndex)
{
    if (processor == &uiEqProcessor)
    {
        std::lock_guard<std::mutex> lk(rebuildMutex);
        if (activeDSP)
            activeDSP->eq.syncBandNodeFrom(uiEqProcessor, bandIndex);
        enqueueSnapshotCommand();
    }
}

void AudioEngine::eqGlobalChanged(EQProcessor* processor)
{
    if (processor == &uiEqProcessor)
    {
        std::lock_guard<std::mutex> lk(rebuildMutex);
        if (activeDSP) {
            // syncGlobalStateFrom は AGC の実行状態も上書きしてしまうため、
            // UIからの変更通知では、UIが管理するパラメータのみを個別に設定する。
            // これにより、アクティブなDSPのAGC状態がリセットされるのを防ぐ。
            activeDSP->eq.setTotalGain(uiEqProcessor.getTotalGain());
            activeDSP->eq.setAGCEnabled(uiEqProcessor.getAGCEnabled());
            activeDSP->eq.setNonlinearSaturation(uiEqProcessor.getNonlinearSaturation());
            activeDSP->eq.setFilterStructure(uiEqProcessor.getFilterStructure());
        }
        enqueueSnapshotCommand();
    }
}

void AudioEngine::convolverParamsChanged(ConvolverProcessor* processor)
{
    if (processor == &uiConvolverProcessor)
    {
        bool needsStructuralRebuild = false;
        double srForRebuild = 0.0;
        int bsForRebuild = 0;

        {
            std::lock_guard<std::mutex> lk(rebuildMutex);
            if (activeDSP)
            {
                activeDSP->convolver.syncParametersFrom(uiConvolverProcessor);

                const bool uiHasIr = uiConvolverProcessor.isIRLoaded();
                const bool dspHasIr = activeDSP->convolver.isIRLoaded();

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
            }

            if (needsStructuralRebuild)
            {
                srForRebuild = currentSampleRate.load(std::memory_order_acquire);
                bsForRebuild = maxSamplesPerBlock.load(std::memory_order_acquire);
            }
        }

        if (needsStructuralRebuild && srForRebuild > 0.0)
        {
            ++pendingIRGeneration;

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

            requestRebuild(srForRebuild, bsForRebuild);
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
    // サンプルレートをリセット (描画停止用)
    currentSampleRate.store(0.0);

    // レベルをリセット
    inputLevelLinear.store(0.0f);
    outputLevelLinear.store(0.0f);

    if (noiseShaperLearner)
        noiseShaperLearner->stopLearning();

    resetLearningControlState();

    // ==================================================================
    // 【Issue 2 完全解消】手動MMCSS revertを削除
    // 理由:
    //   1. JUCE 8.0.12 の setMMCSSModeEnabled() が内部で管理
    //   2. mmcssHandle はローカル変数だったため未定義エラー発生
    //   3. 手動revertは不要・リークリスクあり → JUCEに任せる
    // ==================================================================

    // rebuildGeneration インクリメント・currentDSP・activeDSP の変更を
    // rebuildMutex で保護し、commitNewDSP() との競合を防ぐ。
    {
        std::lock_guard<std::mutex> lk(rebuildMutex);

        // rebuildThread の進行中タスクを obsolete にする。
        // rebuildThread は isObsolete() チェックで task.currentDSP へのアクセスを
        // 早期に打ち切るため、dangling pointer アクセスのウィンドウを最小化する。
        // 安全な最終保証は ~AudioEngine() の rebuildThread.join() が担う。
        rebuildGeneration.fetch_add(1, std::memory_order_relaxed);

        // 1. Stop Audio Thread access
        currentDSP.store(nullptr, std::memory_order_release);

        // 2. activeDSP を deferred release キューへ移す。
        if (activeDSP)
        {
            activeDSP->release();
            activeDSP = nullptr;
        }

        if (auto* fading = fadingOutDSP.exchange(nullptr, std::memory_order_acq_rel))
            fading->release();

        dspCrossfadePending.store(false, std::memory_order_release);
        dspCrossfadeGain.setCurrentAndTargetValue(1.0);
    }

    // 3. Release UI Processor Resources
    uiConvolverProcessor.releaseResources();
    uiEqProcessor.releaseResources();
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


    // Audio Thread Epoch 管理
    const uint64_t epoch = globalEpoch.load(std::memory_order_acquire);
    audioThreadEpoch.store(epoch, std::memory_order_release);
    std::atomic_thread_fence(std::memory_order_seq_cst);

    // RAII により関数終了時に必ず audioThreadEpoch をリセット
    struct EpochGuard {
        std::atomic<uint64_t>& ae;
        ~EpochGuard() { ae.store(0, std::memory_order_release); }
    } epochGuard { audioThreadEpoch };

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

        // [Bug Fix] RCU 世代検証によるデータ競合の完全排除
        // 1. 取得前の世代をスナップショット
        const uint32_t adaptiveGenBefore = adaptiveCoeffBank.generation.load(std::memory_order_acquire);
        // 2. ポインタを取得
        const CoeffSet* adaptiveSet = AudioEngine::getActiveCoeffSet(adaptiveCoeffBank);
        // 3. 取得後の世代をスナップショット
        const uint32_t adaptiveGenAfter = adaptiveCoeffBank.generation.load(std::memory_order_acquire);

        // 取得中に世代が変化した場合、ポインタが指すバッファが Writer によって再利用されている可能性がある。
        // 安全のため、このブロックでは係数更新をスキップする（nullptr を渡す）。
        CoeffSet localAdaptiveSet {};
        const CoeffSet* safeAdaptiveSet = nullptr;
        if (adaptiveSet != nullptr && adaptiveGenBefore == adaptiveGenAfter)
        {
            localAdaptiveSet = *adaptiveSet;
            if (adaptiveCoeffBank.generation.load(std::memory_order_acquire) == adaptiveGenAfter)
                safeAdaptiveSet = &localAdaptiveSet;
        }

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

        DSPCore* fading = fadingOutDSP.load(std::memory_order_acquire);
        if (fading != nullptr && dspCrossfadePending.exchange(false, std::memory_order_acq_rel))
        {
            dspCrossfadeGain.reset(std::max(1.0, dsp->sampleRate), 0.03);
            dspCrossfadeGain.setCurrentAndTargetValue(0.0);
            dspCrossfadeGain.setTargetValue(1.0);
        }

        const bool canCrossfade = (fading != nullptr)
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
            fading->addRef();
            fading->processToBuffer(bufferToFill, dspCrossfadeFloatBuffer, audioFifo, audioFifoBuffer,
                                   fadingInputMeter, fadingOutputMeter, fadingState);
            dsp->process(bufferToFill, audioFifo, audioFifoBuffer, inputLevelLinear, outputLevelLinear, procState);

            const int outChannels = std::min(2, buffer->getNumChannels());
            float* dstL = (outChannels > 0) ? buffer->getWritePointer(0, startSample) : nullptr;
            float* dstR = (outChannels > 1) ? buffer->getWritePointer(1, startSample) : nullptr;
            const float* oldL = (outChannels > 0) ? dspCrossfadeFloatBuffer.getReadPointer(0, 0) : nullptr;
            const float* oldR = (outChannels > 1) ? dspCrossfadeFloatBuffer.getReadPointer(1, 0) : nullptr;
            for (int i = 0; i < numSamples; ++i)
            {
                const double gNew = dspCrossfadeGain.getNextValue();
                const double gOld = 1.0 - gNew;
                if (dstL != nullptr)
                    dstL[i] = static_cast<float>(dstL[i] * gNew + oldL[i] * gOld);
                if (dstR != nullptr)
                    dstR[i] = static_cast<float>(dstR[i] * gNew + oldR[i] * gOld);
            }

            fading->release();

            if (!dspCrossfadeGain.isSmoothing())
            {
                if (auto* done = fadingOutDSP.exchange(nullptr, std::memory_order_acq_rel))
                    done->release();
                dspCrossfadeGain.setCurrentAndTargetValue(1.0);
            }
        }
        else
        {
            // 通常パス（クロスフェードなし）：RCU で dsp の生存が保証されるため addRef/release 不要
            dsp->process(bufferToFill, audioFifo, audioFifoBuffer, inputLevelLinear, outputLevelLinear, procState);

            if (fading != nullptr && !dspCrossfadeGain.isSmoothing())
            {
                if (auto* done = fadingOutDSP.exchange(nullptr, std::memory_order_acq_rel))
                    done->release();
            }
        }
    }

}

void AudioEngine::processBlockDouble (juce::AudioBuffer<double>& buffer)
{
    const juce::ScopedNoDenormals noDenormals;

    const int numSamples = buffer.getNumSamples();
    // 事前サニティチェック (getNextAudioBlock と同様)
    constexpr int ABSOLUTE_MAX_BLOCK_SIZE = 1 << 20;
    if (numSamples <= 0 || numSamples > ABSOLUTE_MAX_BLOCK_SIZE)
    {
        buffer.clear();
        return;
    }

    // Audio Thread Epoch 管理
    const uint64_t epoch = globalEpoch.load(std::memory_order_acquire);
    audioThreadEpoch.store(epoch, std::memory_order_release);
    std::atomic_thread_fence(std::memory_order_seq_cst);

    // RAII により関数終了時に必ず audioThreadEpoch をリセット
    struct EpochGuard {
        std::atomic<uint64_t>& ae;
        ~EpochGuard() { ae.store(0, std::memory_order_release); }
    } epochGuard { audioThreadEpoch };

    DSPCore* dsp = currentDSP.load(std::memory_order_acquire);
    if (dsp == nullptr)
    {
        buffer.clear();
        return;
    }

    // DSPCore 固有の上限チェック (getNextAudioBlock と同様)
    if (numSamples > dsp->maxSamplesPerBlock)
    {
        buffer.clear();
        return;
    }

    const double engineSampleRate = currentSampleRate.load(std::memory_order_relaxed);
    if (absDiffNoLibm(dsp->sampleRate, engineSampleRate) > 1e-6)
    {
        inputLevelLinear.store(0.0f);
        outputLevelLinear.store(0.0f);
        buffer.clear();
        return;
    }

    // ── Audio Thread 最適化: 全アトミック変数をスナップショット構築（load 回数最小化） ──
    // すべての読み取りをここに集中させ、processDouble() 内では追加の .load() を一切行わない。
    const bool eqBypassed               = eqBypassRequested.load(std::memory_order_acquire);
    const bool convBypassed             = convBypassRequested.load(std::memory_order_acquire);
    const ProcessingOrder order         = currentProcessingOrder.load(std::memory_order_relaxed);
    const AnalyzerSource analyzerSource = currentAnalyzerSource.load(std::memory_order_relaxed);
    const bool analyzerEnabledNow       = analyzerEnabled.load(std::memory_order_relaxed);
    const bool softClip                 = softClipEnabled.load(std::memory_order_relaxed);
    const float satAmt                  = saturationAmount.load(std::memory_order_relaxed);
    const double headroomGain           = inputHeadroomGain.load(std::memory_order_relaxed);
    const double makeupGain             = outputMakeupGain.load(std::memory_order_relaxed);
    const double convInputTrimGain      = convolverInputTrimGain.load(std::memory_order_relaxed);
    const convo::HCMode hcMode      = convHCFilterMode.load(std::memory_order_relaxed);
    const convo::LCMode lcMode      = convLCFilterMode.load(std::memory_order_relaxed);
    const convo::HCMode lpfMode     = eqLPFFilterMode.load(std::memory_order_relaxed);
    const int adaptiveCoeffBankIndex    = currentAdaptiveCoeffBankIndex.load(std::memory_order_acquire);
    const auto& adaptiveCoeffBank       = getAdaptiveCoeffBankForIndex(adaptiveCoeffBankIndex);
    const bool adaptiveCaptureEnabled   = noiseShaperLearner && noiseShaperLearner->isRunning();

    // [Bug Fix] RCU 世代検証によるデータ競合の完全排除
    // 1. 取得前の世代をスナップショット
    const uint32_t adaptiveGenBefore = adaptiveCoeffBank.generation.load(std::memory_order_acquire);
    // 2. ポインタを取得
    const CoeffSet* adaptiveSet = AudioEngine::getActiveCoeffSet(adaptiveCoeffBank);
    // 3. 取得後の世代をスナップショット
    const uint32_t adaptiveGenAfter = adaptiveCoeffBank.generation.load(std::memory_order_acquire);

    // 取得中に世代が変化した場合、ポインタが指すバッファが Writer によって再利用されている可能性がある。
    // 安全のため、このブロックでは係数更新をスキップする（nullptr を渡す）。
    CoeffSet localAdaptiveSet {};
    const CoeffSet* safeAdaptiveSet = nullptr;
    if (adaptiveSet != nullptr && adaptiveGenBefore == adaptiveGenAfter)
    {
        localAdaptiveSet = *adaptiveSet;
        if (adaptiveCoeffBank.generation.load(std::memory_order_acquire) == adaptiveGenAfter)
            safeAdaptiveSet = &localAdaptiveSet;
    }

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

    DSPCore* fading = fadingOutDSP.load(std::memory_order_acquire);
    if (fading != nullptr && dspCrossfadePending.exchange(false, std::memory_order_acq_rel))
    {
        dspCrossfadeGain.reset(std::max(1.0, dsp->sampleRate), 0.03);
        dspCrossfadeGain.setCurrentAndTargetValue(0.0);
        dspCrossfadeGain.setTargetValue(1.0);
    }

    const bool canCrossfade = (fading != nullptr)
        && dspCrossfadeGain.isSmoothing()
        && dspCrossfadeDoubleBuffer.getNumChannels() >= 2
        && dspCrossfadeDoubleBuffer.getNumSamples() >= numSamples;

    if (canCrossfade)
    {
        dspCrossfadeDoubleBuffer.clear(0, 0, numSamples);
        dspCrossfadeDoubleBuffer.clear(1, 0, numSamples);

        auto fadingState = procState;
        fadingState.analyzerEnabled = false;
        fadingState.adaptiveCaptureQueue = nullptr;

        std::atomic<float> fadingInputMeter { 0.0f };
        std::atomic<float> fadingOutputMeter { 0.0f };
        fading->addRef();
        fading->processDoubleToBuffer(buffer, dspCrossfadeDoubleBuffer, audioFifo, audioFifoBuffer,
                                      fadingInputMeter, fadingOutputMeter, fadingState);
        dsp->processDouble(buffer, audioFifo, audioFifoBuffer, inputLevelLinear, outputLevelLinear, procState);

        const int outChannels = std::min(2, buffer.getNumChannels());
        double* dstL = (outChannels > 0) ? buffer.getWritePointer(0, 0) : nullptr;
        double* dstR = (outChannels > 1) ? buffer.getWritePointer(1, 0) : nullptr;
        const double* oldL = (outChannels > 0) ? dspCrossfadeDoubleBuffer.getReadPointer(0, 0) : nullptr;
        const double* oldR = (outChannels > 1) ? dspCrossfadeDoubleBuffer.getReadPointer(1, 0) : nullptr;
        for (int i = 0; i < numSamples; ++i)
        {
            const double gNew = dspCrossfadeGain.getNextValue();
            const double gOld = 1.0 - gNew;
            if (dstL != nullptr)
                dstL[i] = dstL[i] * gNew + oldL[i] * gOld;
            if (dstR != nullptr)
                dstR[i] = dstR[i] * gNew + oldR[i] * gOld;
        }
        fading->release();

        if (!dspCrossfadeGain.isSmoothing())
        {
            if (auto* done = fadingOutDSP.exchange(nullptr, std::memory_order_acq_rel))
                done->release();
            dspCrossfadeGain.setCurrentAndTargetValue(1.0);
        }
    }
    else
    {
        dsp->processDouble(buffer, audioFifo, audioFifoBuffer, inputLevelLinear, outputLevelLinear, procState);

        if (fading != nullptr && !dspCrossfadeGain.isSmoothing())
        {
            if (auto* done = fadingOutDSP.exchange(nullptr, std::memory_order_acq_rel))
                done->release();
        }
    }
}

void AudioEngine::DSPCore::processToBuffer(const juce::AudioSourceChannelInfo& source,
                                          juce::AudioBuffer<float>& destination,
                                          juce::AbstractFifo& audioFifo,
                                          juce::AudioBuffer<float>& audioFifoBuffer,
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
    process(destinationInfo, audioFifo, audioFifoBuffer, inputLevelLinear, outputLevelLinear, state);
}

void AudioEngine::DSPCore::process(const juce::AudioSourceChannelInfo& bufferToFill,
                                  juce::AbstractFifo& audioFifo,
                                  juce::AudioBuffer<float>& audioFifoBuffer,
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
                                              inputTap, audioFifo, audioFifoBuffer);

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
        eq.process(processBlock);
    }
    else
    {
        // 1. EQ
        eq.process(processBlock);
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

    // ── Analyzer Output Tap (Post-DSP) ──
    if (state.analyzerEnabled && state.analyzerSource == AnalyzerSource::Output)
    {
        pushToFifo(processBlock, audioFifo, audioFifoBuffer);
    }

    // ダウンサンプリング (結果は processBuffer に書き戻される)
    if (oversamplingFactor > 1)
    {
        oversampling.processDown(processBlock, originalBlock, static_cast<int>(originalBlock.getNumChannels()));
        processBlock = originalBlock;
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

void AudioEngine::DSPCore::processDoubleToBuffer(const juce::AudioBuffer<double>& source,
                                                 juce::AudioBuffer<double>& destination,
                                                 juce::AbstractFifo& audioFifo,
                                                 juce::AudioBuffer<float>& audioFifoBuffer,
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

    processDouble(destination, audioFifo, audioFifoBuffer, inputLevelLinear, outputLevelLinear, state);
}

void AudioEngine::DSPCore::processDouble(juce::AudioBuffer<double>& buffer,
                                         juce::AbstractFifo& audioFifo,
                                         juce::AudioBuffer<float>& audioFifoBuffer,
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
                                                     inputTapD, audioFifo, audioFifoBuffer);
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

    eq.setBypass(state.eqBypassed);

    if (state.order == ProcessingOrder::ConvolverThenEQ)
    {
        if (!state.convBypassed)
            convolver.process(processBlock);
        eq.process(processBlock);
    }
    else
    {
        eq.process(processBlock);
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

    if (state.analyzerEnabled && state.analyzerSource == AnalyzerSource::Output)
        pushToFifo(processBlock, audioFifo, audioFifoBuffer);

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
                                      juce::AbstractFifo& audioFifo,
                                      juce::AudioBuffer<float>& audioFifoBuffer) const noexcept
{
    const int numSamples = (int)block.getNumSamples();

    // 安全性確認: FIFOバッファのサイズが初期化時から変更されていないことを保証
    jassert (audioFifoBuffer.getNumSamples() == audioFifo.getTotalSize());

    // FIFO空き容量チェック (Overflow Protection)
    // 部分書き込みは波形不連続（グリッチ）の原因となるため、
    // ブロック全体が書き込めない場合は書き込みをスキップする (All or Nothing)
    if (audioFifo.getFreeSpace() < numSamples)
        return;

    int start1, size1, start2, size2;
    audioFifo.prepareToWrite(numSamples, start1, size1, start2, size2);

    if (size1 + size2 < numSamples)
    {
        // getFreeSpace() チェックを通過したにもかかわらず、prepareToWrite() が
        // 要求されたサンプル数より少ない領域しか返さなかった場合の防御的措置。
        // SPSCキューでは理論上到達しないはずだが、部分書き込みを確実に防ぐためにチェックする。
        jassertfalse;
        return;
    }

    const double* l = block.getChannelPointer(0);
    const double* r = (block.getNumChannels() > 1) ? block.getChannelPointer(1) : nullptr;
    const bool hasRightChannel = (audioFifoBuffer.getNumChannels() > 1);

    // AVX2 double→float 変換ヘルパー (4 doubles → 4 floats)
    auto convertBlock = [&](const double* srcL, const double* srcR,
                             float* dstL, float* dstR, int n) noexcept
    {
        int i = 0;
        const int vEnd = n / 4 * 4;
        for (; i < vEnd; i += 4)
        {
            __m256d vL = _mm256_loadu_pd(srcL + i);
            __m128  fL = _mm256_cvtpd_ps(vL);
            _mm_storeu_ps(dstL + i, fL);
            if (dstR && srcR)
            {
                __m256d vR = _mm256_loadu_pd(srcR + i);
                _mm_storeu_ps(dstR + i, _mm256_cvtpd_ps(vR));
            }
            else if (dstR)
            {
                _mm_storeu_ps(dstR + i, fL); // モノ → ステレオ
            }
        }
        for (; i < n; ++i)
        {
            dstL[i] = static_cast<float>(srcL[i]);
            if (dstR) dstR[i] = srcR ? static_cast<float>(srcR[i]) : dstL[i];
        }
    };

    if (size1 > 0)
    {
        convertBlock(l, r,
                     audioFifoBuffer.getWritePointer(0, start1),
                     hasRightChannel ? audioFifoBuffer.getWritePointer(1, start1) : nullptr,
                     size1);
        l += size1;
        if (r != nullptr) r += size1;
    }

    if (size2 > 0)
    {
        convertBlock(l, r,
                     audioFifoBuffer.getWritePointer(0, start2),
                     hasRightChannel ? audioFifoBuffer.getWritePointer(1, start2) : nullptr,
                     size2);
    }

    audioFifo.finishedWrite(size1 + size2);
}

float AudioEngine::DSPCore::processInput(const juce::AudioSourceChannelInfo& bufferToFill, int numSamples,
                                          double headroomGain,
                                          bool analyzerInputTap,
                                          juce::AbstractFifo& audioFifo,
                                          juce::AudioBuffer<float>& audioFifoBuffer) noexcept
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
        pushToFifo(block, audioFifo, audioFifoBuffer);

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
                                               juce::AbstractFifo& audioFifo,
                                               juce::AudioBuffer<float>& audioFifoBuffer) noexcept
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
        pushToFifo(block, audioFifo, audioFifoBuffer);

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
// 閾値を超えた信号を滑らかにクリップし、真空管アンプのような温かみのある歪みを加える。
// @param x 入力信号
// @param threshold クリッピングが開始される閾値 (正の値)
// @param knee 閾値周辺のカーブの滑らかさ（ニー） (正の値)
// @param asymmetry 非対称性の量。負の波形をより強くクリップし、偶数次倍音を生成する。
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

    juce::FloatVectorOperations::copy(buffer.getWritePointer(0, 0), dataL, numSamples);
    if (numChannels > 1 && dataR != nullptr)
        juce::FloatVectorOperations::copy(buffer.getWritePointer(1, 0), dataR, numSamples);

    for (int channel = numChannels; channel < buffer.getNumChannels(); ++channel)
        buffer.clear(channel, 0, numSamples);
}

void AudioEngine::setEqBypassRequested (bool shouldBypass)
{
    eqBypassRequested.store (shouldBypass, std::memory_order_release);
    m_currentEqBypass.store(shouldBypass, std::memory_order_release);
    uiEqProcessor.setBypass(shouldBypass);
    applyDefaultsForCurrentMode();
    enqueueSnapshotCommand();
    sendChangeMessage();
}

void AudioEngine::setConvolverBypassRequested (bool shouldBypass)
{
    convBypassRequested.store (shouldBypass, std::memory_order_release);
    m_currentConvBypass.store(shouldBypass, std::memory_order_release);
    uiConvolverProcessor.setBypass(shouldBypass);
    applyDefaultsForCurrentMode();
    enqueueSnapshotCommand();
    sendChangeMessage();
}

void AudioEngine::setConvolverUseMinPhase(bool useMinPhase)
{
    setConvolverPhaseMode(useMinPhase ? ConvolverProcessor::PhaseMode::Minimum
                                      : ConvolverProcessor::PhaseMode::AsIs);
}

bool AudioEngine::getConvolverUseMinPhase() const
{
    return getConvolverPhaseMode() == ConvolverProcessor::PhaseMode::Minimum;
}

void AudioEngine::setConvolverPhaseMode(ConvolverProcessor::PhaseMode mode)
{
    uiConvolverProcessor.setPhaseMode(mode);
}

ConvolverProcessor::PhaseMode AudioEngine::getConvolverPhaseMode() const
{
    return uiConvolverProcessor.getPhaseMode();
}

void AudioEngine::requestEqPreset (int presetIndex)
{
    uiEqProcessor.loadPreset (presetIndex);
    sendChangeMessage();
}

void AudioEngine::requestEqPresetFromText(const juce::File& file)
{
    if (uiEqProcessor.loadFromTextFile(file))
        sendChangeMessage();
}

void AudioEngine::requestConvolverPreset(const juce::File& irFile)
{
    uiConvolverProcessor.loadIR(irFile);
}

void AudioEngine::requestLoadState (const juce::ValueTree& state)
{
    // B19: RAII ガードを使用して、例外発生時も確実にフラグを戻す
    RestoreStateGuard guard(m_isRestoringState);

    // ─── Step 1: モード・バイパス状態を先に復元 ────────────────────────────
    if (state.hasProperty("processingOrder"))
        currentProcessingOrder.store((ProcessingOrder)(int)state.getProperty("processingOrder"));

    if (state.hasProperty("eqBypassed"))
    {
        bool bypassed = state.getProperty("eqBypassed");
        eqBypassRequested.store(bypassed, std::memory_order_release);
        uiEqProcessor.setBypass(bypassed);
    }

    if (state.hasProperty("convBypassed"))
    {
        bool bypassed = state.getProperty("convBypassed");
        convBypassRequested.store(bypassed, std::memory_order_release);
        uiConvolverProcessor.setBypass(bypassed);
    }

    // ─── Step 2: ゲイン値を復元 (モード依存クランプが正しく適用される) ─────
    // NOTE: ここで guard を破棄せず、関数終了まで維持することで
    //       setInputHeadroomDb 等の内部で呼ばれる applyDefaults を抑制し続ける。
    //       (旧実装では 4059 行目で false に戻していたが、B19 では安全のため全域カバー)

    if (state.hasProperty("inputHeadroomDb"))
        setInputHeadroomDb(state.getProperty("inputHeadroomDb"));

    if (state.hasProperty("outputMakeupDb"))
        setOutputMakeupDb(state.getProperty("outputMakeupDb"));

    if (state.hasProperty("convolverInputTrimDb"))
        setConvolverInputTrimDb(state.getProperty("convolverInputTrimDb"));

    if (state.hasProperty("ditherBitDepth"))
        setDitherBitDepth(static_cast<int>(state.getProperty("ditherBitDepth")));

    if (state.hasProperty("noiseShaperType"))
        setNoiseShaperType((NoiseShaperType)(int)state.getProperty("noiseShaperType"));

    {
        bool hasBankedAdaptiveCoefficients = false;

        for (int bankIndex = 0; bankIndex < getAdaptiveSampleRateBankCount(); ++bankIndex)
        {
            const double bankSampleRate = getAdaptiveSampleRateBankHz(bankIndex);
            double adaptiveCoefficients[kAdaptiveNoiseShaperOrder] = {};
            bool hasBankCoefficients = false;

            getAdaptiveCoefficientsForSampleRate(bankSampleRate, adaptiveCoefficients, kAdaptiveNoiseShaperOrder);
            for (int coeffIndex = 0; coeffIndex < kAdaptiveNoiseShaperOrder; ++coeffIndex)
            {
                const auto propertyName = makeAdaptiveCoeffPropertyName(bankSampleRate, coeffIndex);
                if (state.hasProperty(propertyName))
                {
                    adaptiveCoefficients[coeffIndex] = static_cast<double>(state.getProperty(propertyName));
                    hasBankCoefficients = true;
                    hasBankedAdaptiveCoefficients = true;
                }
            }

            if (hasBankCoefficients)
                setAdaptiveCoefficientsForSampleRate(bankSampleRate, adaptiveCoefficients, kAdaptiveNoiseShaperOrder);
        }

        if (!hasBankedAdaptiveCoefficients)
        {
            double legacyAdaptiveCoefficients[kAdaptiveNoiseShaperOrder] = {};
            bool hasLegacyAdaptiveCoefficients = false;

            for (int coeffIndex = 0; coeffIndex < kAdaptiveNoiseShaperOrder; ++coeffIndex)
            {
                const auto propertyName = "adaptiveCoeff" + juce::String(coeffIndex);
                if (state.hasProperty(propertyName))
                {
                    legacyAdaptiveCoefficients[coeffIndex] = static_cast<double>(state.getProperty(propertyName));
                    hasLegacyAdaptiveCoefficients = true;
                }
            }

            if (hasLegacyAdaptiveCoefficients)
            {
                for (int bankIndex = 0; bankIndex < getAdaptiveSampleRateBankCount(); ++bankIndex)
                    setAdaptiveCoefficientsForSampleRate(getAdaptiveSampleRateBankHz(bankIndex),
                                                         legacyAdaptiveCoefficients,
                                                         kAdaptiveNoiseShaperOrder);
            }
        }
    }

    if (state.hasProperty("oversamplingFactor"))
        setOversamplingFactor(static_cast<int>(state.getProperty("oversamplingFactor")));

    if (state.hasProperty("oversamplingType"))
        setOversamplingType((OversamplingType)(int)state.getProperty("oversamplingType"));

    // --- NoiseShaperLearner Settings ---
    if (state.hasProperty("cmaesRestarts") || state.hasProperty("coeffSafetyMargin") || state.hasProperty("enableStabilityCheck"))
    {
        auto s = getNoiseShaperLearnerSettings();
        if (state.hasProperty("cmaesRestarts"))
            s.cmaesRestarts = static_cast<int>(state.getProperty("cmaesRestarts"));
        if (state.hasProperty("coeffSafetyMargin"))
            s.coeffSafetyMargin = static_cast<double>(state.getProperty("coeffSafetyMargin"));
        if (state.hasProperty("enableStabilityCheck"))
            s.enableStabilityCheck = static_cast<bool>(state.getProperty("enableStabilityCheck"));
        setNoiseShaperLearnerSettings(s);
    }

    // ─── Step 3: その他のグローバル設定 ─────────────────────────────────────
    if (state.hasProperty("softClipEnabled"))
        setSoftClipEnabled(state.getProperty("softClipEnabled"));

    if (state.hasProperty("saturationAmount"))
        setSaturationAmount(state.getProperty("saturationAmount"));

    if (state.hasProperty("analyzerSource"))
        setAnalyzerSource((AnalyzerSource)(int)state.getProperty("analyzerSource"));

    // 出力周波数フィルターモードの読み込み
    if (state.hasProperty("convHCFilterMode"))
        setConvHCFilterMode((convo::HCMode)(int)state.getProperty("convHCFilterMode"));
    if (state.hasProperty("convLCFilterMode"))
        setConvLCFilterMode((convo::LCMode)(int)state.getProperty("convLCFilterMode"));
    if (state.hasProperty("eqLPFFilterMode"))
        setEqLPFFilterMode((convo::HCMode)(int)state.getProperty("eqLPFFilterMode"));

    // ─── Step 4: サブプロセッサ状態の復元 ───────────────────────────────────
    auto eqState = state.getChildWithName ("EQ");
    if (eqState.isValid())
        uiEqProcessor.setState (eqState);

    auto convState = state.getChildWithName ("Convolver");
    if (convState.isValid())
        uiConvolverProcessor.setState (convState);

    // UI更新通知
    sendChangeMessage();
}

juce::ValueTree AudioEngine::getCurrentState() const
{
    juce::ValueTree state ("Preset");

    // グローバル設定の保存
    state.setProperty("processingOrder", (int)currentProcessingOrder.load(), nullptr);
    state.setProperty("softClipEnabled", softClipEnabled.load(), nullptr);
    state.setProperty("saturationAmount", saturationAmount.load(), nullptr);
    state.setProperty("inputHeadroomDb", inputHeadroomDb.load(), nullptr);
    state.setProperty("outputMakeupDb", outputMakeupDb.load(), nullptr);
    state.setProperty("analyzerSource", (int)currentAnalyzerSource.load(), nullptr);
    state.setProperty("convolverInputTrimDb", convolverInputTrimDb.load(), nullptr);
    state.setProperty("ditherBitDepth", ditherBitDepth.load(), nullptr);
    state.setProperty("noiseShaperType", (int)noiseShaperType.load(), nullptr);
    state.setProperty("oversamplingFactor", manualOversamplingFactor.load(), nullptr);
    state.setProperty("oversamplingType", (int)oversamplingType.load(), nullptr);

    // NoiseShaperLearner Settings
    {
        auto s = getNoiseShaperLearnerSettings();
        state.setProperty("cmaesRestarts", s.cmaesRestarts.load(), nullptr);
        state.setProperty("coeffSafetyMargin", s.coeffSafetyMargin.load(), nullptr);
        state.setProperty("enableStabilityCheck", s.enableStabilityCheck.load(), nullptr);
    }

    state.setProperty("eqBypassed", eqBypassRequested.load(), nullptr);
    state.setProperty("convBypassed", convBypassRequested.load(), nullptr);
    // 出力周波数フィルターモードの保存
    state.setProperty("convHCFilterMode", (int)convHCFilterMode.load(), nullptr);
    state.setProperty("convLCFilterMode", (int)convLCFilterMode.load(), nullptr);
    state.setProperty("eqLPFFilterMode",  (int)eqLPFFilterMode.load(), nullptr);

    for (int bankIndex = 0; bankIndex < getAdaptiveSampleRateBankCount(); ++bankIndex)
    {
        const double bankSampleRate = getAdaptiveSampleRateBankHz(bankIndex);
        double adaptiveCoefficients[kAdaptiveNoiseShaperOrder] = {};
        getAdaptiveCoefficientsForSampleRate(bankSampleRate, adaptiveCoefficients, kAdaptiveNoiseShaperOrder);

        for (int coeffIndex = 0; coeffIndex < kAdaptiveNoiseShaperOrder; ++coeffIndex)
            state.setProperty(makeAdaptiveCoeffPropertyName(bankSampleRate, coeffIndex),
                              adaptiveCoefficients[coeffIndex],
                              nullptr);
    }

    state.addChild (uiEqProcessor.getState(), -1, nullptr);
    state.addChild (uiConvolverProcessor.getState(), -1, nullptr);
    return state;
}

void AudioEngine::setInputHeadroomDb(float db)
{
    // コンボルバーが先頭に来る場合 (Conv→PEQ / Conv only) は -6dB 上限で入力保護する。
    // EQ が先頭またはコンボルバーがバイパスされている場合は 0dB まで許容する。
    const bool convBypassed = convBypassRequested.load(std::memory_order_relaxed);
    const bool eqBypassed   = eqBypassRequested.load(std::memory_order_relaxed);
    const ProcessingOrder order = currentProcessingOrder.load(std::memory_order_relaxed);
    const bool convIsFirst = !convBypassed && (order == ProcessingOrder::ConvolverThenEQ || eqBypassed);
    const float maxDb = convIsFirst ? -6.0f : 0.0f;
    float clampedDb = juce::jlimit(-12.0f, maxDb, db);
    if (std::abs(inputHeadroomDb.load() - clampedDb) > 1e-5f)
    {
        inputHeadroomDb.store(clampedDb);
        inputHeadroomGain.store(juce::Decibels::decibelsToGain((double)clampedDb));
        m_currentInputHeadroomDb.store(clampedDb, std::memory_order_relaxed);
        enqueueSnapshotCommand();
    }
}

float AudioEngine::getInputHeadroomDb() const
{
    return inputHeadroomDb.load();
}

void AudioEngine::setOutputMakeupDb(float db)
{
    // Output makeup は全モード共通で 0..12 dB
    const float clampedDb = juce::jlimit(0.0f, 12.0f, db);
    if (std::abs(outputMakeupDb.load() - clampedDb) > 1e-5f)
    {
        outputMakeupDb.store(clampedDb);
        outputMakeupGain.store(juce::Decibels::decibelsToGain((double)clampedDb));
        m_currentOutputMakeupDb.store(clampedDb, std::memory_order_relaxed);
        enqueueSnapshotCommand();
    }
}

float AudioEngine::getOutputMakeupDb() const
{
    return outputMakeupDb.load();
}

void AudioEngine::setProcessingOrder(ProcessingOrder order)
{
    currentProcessingOrder.store(order);
    m_currentProcessingOrder.store(order, std::memory_order_relaxed);
    enqueueSnapshotCommand();
    applyDefaultsForCurrentMode();
}

void AudioEngine::setConvolverInputTrimDb(float db)
{
    // 範囲: -12..0 dB (0dB = トリムなし / -12dB = 最大保護)
    float clampedDb = juce::jlimit(-12.0f, 0.0f, db);
    if (std::abs(convolverInputTrimDb.load() - clampedDb) > 1e-5f)
    {
        convolverInputTrimDb.store(clampedDb);
        convolverInputTrimGain.store(juce::Decibels::decibelsToGain((double)clampedDb));
        m_currentConvInputTrimDb.store(clampedDb, std::memory_order_relaxed);
        enqueueSnapshotCommand();
    }
}

float AudioEngine::getConvolverInputTrimDb() const
{
    return convolverInputTrimDb.load();
}

void AudioEngine::applyDefaultsForCurrentMode()
{
    if (m_isRestoringState) return; // プリセットロード中はデフォルトリセットを抑制する

    const bool eqBypassed  = eqBypassRequested.load(std::memory_order_relaxed);
    const bool convBypassed = convBypassRequested.load(std::memory_order_relaxed);
    const ProcessingOrder order = currentProcessingOrder.load(std::memory_order_relaxed);

    float newInputHeadroomDb = 0.0f;
    float newOutputMakeupDb = 0.0f;
    float newConvTrimDb = 0.0f;

    if (convBypassed && !eqBypassed)
    {
        newInputHeadroomDb = 0.0f;
        newOutputMakeupDb = 0.0f;
        newConvTrimDb = 0.0f;
    }
    else if (!convBypassed && order == ProcessingOrder::EQThenConvolver && !eqBypassed)
    {
        newInputHeadroomDb = 0.0f;
        newOutputMakeupDb = 10.0f;
        newConvTrimDb = -6.0f;
    }
    else if (eqBypassed && !convBypassed)
    {
        newInputHeadroomDb = -6.0f;
        newOutputMakeupDb = 12.0f;
        newConvTrimDb = 0.0f;
    }
    else
    {
        newInputHeadroomDb = -6.0f;
        newOutputMakeupDb = 12.0f;
        newConvTrimDb = 0.0f;
    }

    inputHeadroomDb.store(newInputHeadroomDb, std::memory_order_relaxed);
    outputMakeupDb.store(newOutputMakeupDb, std::memory_order_relaxed);
    convolverInputTrimDb.store(newConvTrimDb, std::memory_order_relaxed);
    inputHeadroomGain.store(juce::Decibels::decibelsToGain(static_cast<double>(newInputHeadroomDb)), std::memory_order_relaxed);
    outputMakeupGain.store(juce::Decibels::decibelsToGain(static_cast<double>(newOutputMakeupDb)), std::memory_order_relaxed);
    convolverInputTrimGain.store(juce::Decibels::decibelsToGain(static_cast<double>(newConvTrimDb)), std::memory_order_relaxed);

    m_currentInputHeadroomDb.store(newInputHeadroomDb, std::memory_order_relaxed);
    m_currentOutputMakeupDb.store(newOutputMakeupDb, std::memory_order_relaxed);
    m_currentConvInputTrimDb.store(newConvTrimDb, std::memory_order_relaxed);
    enqueueSnapshotCommand();
}

void AudioEngine::setDitherBitDepth(int bitDepth)
{
    if (ditherBitDepth.load() != bitDepth)
    {
        const bool adaptiveLearningActive = (noiseShaperType.load(std::memory_order_relaxed) == NoiseShaperType::Adaptive9thOrder)
            && noiseShaperLearner
            && noiseShaperLearner->isRunning();

        if (adaptiveLearningActive)
        {
            stopNoiseShaperLearning();
            noiseShaperLearner->setErrorMessage("Learning stopped due to bit depth change. Please restart learning.");
        }

        ditherBitDepth.store(bitDepth);
        m_currentDitherBitDepth.store(bitDepth, std::memory_order_relaxed);
        DBG_LOG("Dither Bit Depth changed: " + juce::String(bitDepth));
        enqueueSnapshotCommand();

        selectAdaptiveCoeffBankForCurrentSettings();

        // UI側（学習ウィンドウ）が即座に反映できるように通知
        sendChangeMessage();

        const double sr = currentSampleRate.load();
        if (sr > 0.0)
        {
            requestRebuild(sr, maxSamplesPerBlock.load());
        }
    }
}

int AudioEngine::getDitherBitDepth() const
{
    return ditherBitDepth.load();
}

void AudioEngine::setNoiseShaperType(NoiseShaperType type)
{
    if (noiseShaperType.load() != type)
    {
        noiseShaperType.store(type);
        m_currentNoiseShaperType.store(type, std::memory_order_relaxed);
        if (type != NoiseShaperType::Adaptive9thOrder)
        {
            stopNoiseShaperLearning();
        }
        else
        {
            if (noiseShaperLearner)
                noiseShaperLearner->stopLearning();

            noiseShaperLearner = std::make_unique<NoiseShaperLearner>(*this, audioCaptureQueue);
            noiseShaperLearner->setLearningMode(pendingLearningMode.load(std::memory_order_acquire));
            resetLearningControlState();
        }

        juce::String typeName = "Psychoacoustic";
        if (type == NoiseShaperType::Fixed4Tap)
            typeName = "Fixed4Tap";
        else if (type == NoiseShaperType::Adaptive9thOrder)
            typeName = "Adaptive9thOrder";

        DBG_LOG("Noise Shaper changed: " + typeName);
        enqueueSnapshotCommand();
        const double sr = currentSampleRate.load();
        if (sr > 0.0)
            requestRebuild(sr, maxSamplesPerBlock.load());
    }
}

AudioEngine::NoiseShaperType AudioEngine::getNoiseShaperType() const
{
    return noiseShaperType.load();
}

void AudioEngine::setFixedNoiseLogIntervalMs(int intervalMs) noexcept
{
    fixedNoiseLogIntervalMs.store(juce::jlimit(250, 10000, intervalMs), std::memory_order_relaxed);
}

int AudioEngine::getFixedNoiseLogIntervalMs() const noexcept
{
    return fixedNoiseLogIntervalMs.load(std::memory_order_relaxed);
}

void AudioEngine::setFixedNoiseWindowSamples(int windowSamples) noexcept
{
    fixedNoiseWindowSamples.store(juce::jlimit(256, 262144, windowSamples), std::memory_order_relaxed);
}

int AudioEngine::getFixedNoiseWindowSamples() const noexcept
{
    return fixedNoiseWindowSamples.load(std::memory_order_relaxed);
}

void AudioEngine::setSoftClipEnabled(bool enabled)
{
    softClipEnabled.store(enabled, std::memory_order_relaxed);
    m_currentSoftClipEnabled.store(enabled, std::memory_order_relaxed);
    enqueueSnapshotCommand();
}

bool AudioEngine::isSoftClipEnabled() const
{
    return softClipEnabled.load(std::memory_order_relaxed);
}

void AudioEngine::setSaturationAmount(float amount)
{
    const float clamped = juce::jlimit(0.0f, 1.0f, amount);
    saturationAmount.store(clamped, std::memory_order_relaxed);
    m_currentSaturationAmount.store(clamped, std::memory_order_relaxed);
    enqueueSnapshotCommand();
}

float AudioEngine::getSaturationAmount() const
{
    return saturationAmount.load(std::memory_order_relaxed);
}

void AudioEngine::setOversamplingFactor(int factor)
{
    // 0=Auto, 1, 2, 4, 8
    int newFactor = 0;
    if (factor == 1 || factor == 2 || factor == 4 || factor == 8)
    {
        newFactor = factor;
    }

    if (manualOversamplingFactor.load() != newFactor)
    {
        manualOversamplingFactor.store(newFactor);
        m_currentOversamplingFactor.store(newFactor, std::memory_order_relaxed);
        enqueueSnapshotCommand();
        const double sr = currentSampleRate.load();
        if (sr > 0.0)
        {
            requestRebuild(sr, maxSamplesPerBlock.load());
        }
    }
}

int AudioEngine::getOversamplingFactor() const
{
    return manualOversamplingFactor.load();
}

void AudioEngine::setOversamplingType(OversamplingType type)
{
    oversamplingType.store(type);
    m_currentOversamplingType.store(type, std::memory_order_relaxed);
    enqueueSnapshotCommand();
    const double sr = currentSampleRate.load();
    if (sr > 0.0)
    {
        requestRebuild(sr, maxSamplesPerBlock.load());
    }
}

AudioEngine::OversamplingType AudioEngine::getOversamplingType() const
{
    return oversamplingType.load();
}

//──────────────────────────────────────────────────────────────────────────
// 出力周波数フィルターモード Setter / Getter (Message Thread)
//──────────────────────────────────────────────────────────────────────────
void AudioEngine::setConvHCFilterMode(convo::HCMode mode) noexcept
{
    convHCFilterMode.store(mode, std::memory_order_relaxed);
    // NUC irFreqDomain を再焼き込みするため、uiConvolverProcessor を再構築する。
    // DSPCore::convolver は次回 requestRebuild 時に syncStateFrom + rebuildAllIRsSynchronous で追従する。
    uiConvolverProcessor.setNUCFilterModes(
        convHCFilterMode.load(std::memory_order_relaxed),
        convLCFilterMode.load(std::memory_order_relaxed));
}

convo::HCMode AudioEngine::getConvHCFilterMode() const noexcept
{
    return convHCFilterMode.load(std::memory_order_relaxed);
}

void AudioEngine::setConvLCFilterMode(convo::LCMode mode) noexcept
{
    convLCFilterMode.store(mode, std::memory_order_relaxed);
    // HC と組み合わせて NUC を再構築
    uiConvolverProcessor.setNUCFilterModes(
        convHCFilterMode.load(std::memory_order_relaxed),
        convLCFilterMode.load(std::memory_order_relaxed));
}

convo::LCMode AudioEngine::getConvLCFilterMode() const noexcept
{
    return convLCFilterMode.load(std::memory_order_relaxed);
}

void AudioEngine::setEqLPFFilterMode(convo::HCMode mode) noexcept
{
    eqLPFFilterMode.store(mode, std::memory_order_relaxed);
}

convo::HCMode AudioEngine::getEqLPFFilterMode() const noexcept
{
    return eqLPFFilterMode.load(std::memory_order_relaxed);
}
