//============================================================================
// AudioEngine.cpp  ── v0.2 (JUCE 8.0.12対応)
//
// AudioEngineの実装
//============================================================================
#include "AudioEngine.h"
#include "InputBitDepthTransform.h"
#include <cmath>
#include <complex>
#include <algorithm>
#include <mutex>
#include <condition_variable>
#include <xmmintrin.h>
#include <immintrin.h>

namespace
{
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
    const double abs_x = std::abs(x);
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
                               double threshold, double knee, double asymmetry) noexcept
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

    int i = 0;
    const int vEnd = numSamples / 4 * 4;
    for (; i < vEnd; i += 4)
    {
            __m256d x    = _mm256_loadu_pd(data + i);
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
    }

    // Scalar remainder
    for (; i < numSamples; ++i)
    {
        if (std::abs(data[i]) > clip_start)
            data[i] = musicalSoftClipScalar(data[i], threshold, knee, asymmetry);
    }
}

// コンストラクタ
//--------------------------------------------------------------
AudioEngine::AudioEngine()
{
    // デフォルトサンプルレート (0 = 未初期化/デバイスなし)
    currentSampleRate.store(0.0);

    // バッファ初期化
    audioFifoBuffer.setSize (2, FIFO_SIZE);
    currentDSP.store(nullptr);
}



void AudioEngine::initialize()
{
    // Start worker thread
    rebuildThread = std::thread(&AudioEngine::rebuildThreadLoop, this);

    // 初期DSP構築 (デフォルト設定)
    // 安全対策: バッファサイズを余裕を持って確保 (SAFE_MAX_BLOCK_SIZE)
    // これにより、デバイス初期化前やバッファサイズ変更時の不整合による音切れ/無音を防ぐ
    requestRebuild(48000.0, SAFE_MAX_BLOCK_SIZE);
    maxSamplesPerBlock.store(SAFE_MAX_BLOCK_SIZE);
    currentSampleRate.store(48000.0);

    uiConvolverProcessor.addChangeListener(this);
    uiEqProcessor.addChangeListener(this);
    uiConvolverProcessor.addListener(this);
    uiEqProcessor.addListener(this);

    // タイマー開始 (100ms間隔)
    // - DSP再構築リクエストのポーリング (Audio Threadからの依頼を処理)
    // - ガベージコレクション
    startTimer(100);
}



AudioEngine::~AudioEngine()
{
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
        delete activeDSP;
        activeDSP = nullptr;
    }

    // 5. Explicit Generation Sweep (Trash Bin Cleanup)
    // Clear all pending and old DSPs.
    {
        juce::ScopedLock sl(trashBinLock);
        for (auto* p : trashBinPending) delete p;
        trashBinPending.clear();

        for (auto& entry : trashBin) delete entry.first;
        trashBin.clear();
    }
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

        activeBands[numActiveBands++] = {
            EQProcessor::calcBiquadCoeffs(type, params.frequency, params.gain, params.q, sr),
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

    std::vector<float> totalMagSqL(numPoints);
    std::vector<float> totalMagSqR(numPoints);
    std::vector<float> bandMagSq(numPoints);

    const __m256 vTotalGainSq = _mm256_set1_ps(totalGainSq);
    int i = 0;
    const int vEnd = numPoints / 8 * 8;
    for (; i < vEnd; i += 8)
    {
        _mm256_storeu_ps(totalMagSqL.data() + i, vTotalGainSq);
        _mm256_storeu_ps(totalMagSqR.data() + i, vTotalGainSq);
    }
    for (; i < numPoints; ++i)
    {
        totalMagSqL[i] = totalGainSq;
        totalMagSqR[i] = totalGainSq;
    }

    for (int b = 0; b < numActiveBands; ++b)
    {
        const auto& band = activeBands[b];
        calcMagnitudesForBand(band.coeffs, zArray, bandMagSq.data(), numPoints);

        i = 0;
        if (band.mode == EQChannelMode::Stereo)
        {
            for (; i < vEnd; i += 8)
            {
                __m256 vBand = _mm256_loadu_ps(bandMagSq.data() + i);
                __m256 vL = _mm256_loadu_ps(totalMagSqL.data() + i);
                __m256 vR = _mm256_loadu_ps(totalMagSqR.data() + i);
                _mm256_storeu_ps(totalMagSqL.data() + i, _mm256_mul_ps(vL, vBand));
                _mm256_storeu_ps(totalMagSqR.data() + i, _mm256_mul_ps(vR, vBand));
            }
        }
        else if (band.mode == EQChannelMode::Left)
        {
            for (; i < vEnd; i += 8)
            {
                __m256 vBand = _mm256_loadu_ps(bandMagSq.data() + i);
                __m256 vL = _mm256_loadu_ps(totalMagSqL.data() + i);
                _mm256_storeu_ps(totalMagSqL.data() + i, _mm256_mul_ps(vL, vBand));
            }
        }
        else // Right
        {
            for (; i < vEnd; i += 8)
            {
                __m256 vBand = _mm256_loadu_ps(bandMagSq.data() + i);
                __m256 vR = _mm256_loadu_ps(totalMagSqR.data() + i);
                _mm256_storeu_ps(totalMagSqR.data() + i, _mm256_mul_ps(vR, vBand));
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

    for (i = 0; i < numPoints; ++i)
    {
        if (outMagnitudesL)
        {
            float val = std::sqrt(totalMagSqL[i]);
            outMagnitudesL[i] = std::isfinite(val) ? val : 1.0f;
        }
        if (outMagnitudesR)
        {
            float val = std::sqrt(totalMagSqR[i]);
            outMagnitudesR[i] = std::isfinite(val) ? val : 1.0f;
        }
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

void AudioEngine::DSPCore::prepare(double newSampleRate, int samplesPerBlock, int bitDepth, int manualOversamplingFactor, OversamplingType oversamplingType)
{
    this->sampleRate = newSampleRate;

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

    // ディザの準備 (出力段で行うため元のサンプルレート)
    dither.prepare(newSampleRate, bitDepth);
    this->ditherBitDepth = bitDepth; // DSPCoreのメンバーに保存

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
    oversampling.reset();

    // 【パッチ3】rawバッファクリア（alignedCapacity使用）
    if (alignedL && alignedCapacity > 0)
        juce::FloatVectorOperations::clear(alignedL.get(), alignedCapacity);
    if (alignedR && alignedCapacity > 0)
        juce::FloatVectorOperations::clear(alignedR.get(), alignedCapacity);
}

//--------------------------------------------------------------
// requestRebuild - DSPグラフの再構築 (Message Thread)
//--------------------------------------------------------------
void AudioEngine::requestRebuild(double sampleRate, int samplesPerBlock)
{
    // UIコンポーネント(uiEqProcessor等)へのアクセスやMKLメモリ確保を行うため、必ずMessage Threadで実行すること
    jassert (juce::MessageManager::getInstance()->isThisTheMessageThread());

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

    DSPCore* dspToDestroy = nullptr; // To be destroyed outside the lock
    {
        std::lock_guard<std::mutex> lock(rebuildMutex);

        // Increment generation and create task inside the lock to ensure atomicity
        generation = ++rebuildGeneration;
        task.generation = generation;

        // If a task is already pending, move it out to be destroyed outside the lock.
        // This prevents holding the lock during a potentially slow DSPCore destruction.
        if (hasPendingTask)
            dspToDestroy = pendingTask.newDSP;

        pendingTask = task;
        hasPendingTask = true;
    }
    rebuildCV.notify_one();

    // Destroy the orphaned DSP from the superseded task outside the lock.
    if (dspToDestroy)
        delete dspToDestroy;
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

            // Use unique_ptr to ensure deletion if we continue/break/throw before commit
            std::unique_ptr<DSPCore> dspGuard(task.newDSP);

            // Helper to check obsolescence
            const auto isObsolete = [&] {
                return isRebuildObsolete(task.generation) || rebuildThreadShouldExit.load();
            };

            if (isObsolete()) continue;

            // 1. Prepare (メモリ確保)
            task.newDSP->prepare(task.sampleRate, task.samplesPerBlock, task.ditherDepth, task.manualOversamplingFactor, task.oversamplingType);

            if (isObsolete()) continue;

            // 2. Reuse Logic
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
                        task.newDSP->convolver.getUseMinPhase() == task.currentDSP->convolver.getUseMinPhase() &&
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
                if (isObsolete()) continue;
                task.newDSP->convolver.rebuildAllIRsSynchronous(isObsolete);
            }

            if (isObsolete()) continue;

            // 4. Refresh Latency (Prevent pitch slide during fade-in)
            task.newDSP->convolver.refreshLatency();

            // 5. Fade In
            task.newDSP->fadeInSamplesLeft.store(DSPCore::FADE_IN_SAMPLES, std::memory_order_relaxed);

            // 6. Commit on Message Thread
            // Release ownership from guard, pass to commitNewDSP
            DSPCore* dspToCommit = dspGuard.release();
            if (! juce::MessageManager::callAsync([weakSelf = juce::WeakReference<AudioEngine>(this), newDSP = dspToCommit, generation = task.generation] {
                if (auto* self = weakSelf.get())
                {
                    self->commitNewDSP(newDSP, generation);
                }
                else
                {
                    // Engine is gone, delete the orphan DSP
                    delete newDSP;
                }
            }))
            {
                // MessageManager failed (e.g. shutting down), prevent leak
                delete dspToCommit;
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

    // Lock to ensure the check and commit are atomic with respect to new rebuild requests.
    {
        std::lock_guard<std::mutex> lock(rebuildMutex);

        // 古いリクエストの結果であれば破棄 (Race condition対策)
        if (generation != rebuildGeneration.load(std::memory_order_relaxed))
        {
            delete newDSP;
            return;
        }

        // 1. Update the atomic raw pointer for the Audio Thread (Wait-free)
        currentDSP.store(newDSP, std::memory_order_release);

        // 2. Move the previous active DSP to a temporary variable to be trashed later.
        dspToTrash = activeDSP;

        // 3. Take ownership of the new DSP
        activeDSP = newDSP;
    }

    // 4. Move the old DSP to the trash bin outside the main lock.
    if (dspToTrash != nullptr)
    {
        const juce::ScopedLock sl(trashBinLock);
        try
        {
            trashBinPending.push_back(dspToTrash);
        }
        catch (...)
        {
            // Fallback: if we can't enqueue for later deletion, delete immediately.
            // This prevents a memory leak in case of std::bad_alloc.
            delete dspToTrash;
        }
    }
}

void AudioEngine::timerCallback()
{
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

    std::vector<DSPCore*> toDelete;

    {
        const juce::ScopedLock sl(trashBinLock);
        const uint32 now = juce::Time::getMillisecondCounter();

        // 1. Move pending items to main trash bin with timestamp
        for (const auto& p : trashBinPending)
            trashBin.push_back({p, now});
        trashBinPending.clear();

        // 2. Identify items to delete (older than 10000ms)
        // This ensures that any Audio Thread processing cycle (worst case: 65536/8000Hz = 8.2s)
        // that might have started using the pointer has finished.
        for (auto it = trashBin.begin(); it != trashBin.end(); )
        {
            // Unsigned arithmetic handles wrap-around correctly
            if ((now - it->second) > 10000)
            {
                toDelete.push_back(it->first);
                it = trashBin.erase(it);
            }
            else { ++it; }
        }

        // 3. Size limit (Max 10 items) - メモリ爆発防止
        // 【Fix Bug #1】古いアイテムのみ強制削除する。
        // サイズ超過時も2秒の猶予期間を尊重するため、最も古いアイテム(front)から
        // 削除する。ただし、通常はステップ2の時間ベース削除で十分であるため、
        // ここには滅多に到達しない。高速IR切り替えによる異常蓄積時のみの安全弁として機能する。
        while (trashBin.size() > 30)
        {
            // 最も古い(frontの)アイテムを削除する。
            // これはすでに10秒の猶予を超えているか、超えていなければ次のtimerCallbackで削除される。
            toDelete.push_back(trashBin.front().first);
            trashBin.erase(trashBin.begin());
        }
    }

    // Lock解放後にデストラクタを実行 (stopThread等の重い処理をロック外で行う)
    for (auto* p : toDelete)
        delete p;
    toDelete.clear();

    // 3. 内部プロセッサのクリーンアップを実行
    // 現在アクティブなDSPの内部ゴミ箱も掃除する
    // [Fix Bug C] cleanup() を trashBinLock の外で、currentDSP 経由で一度だけ呼ぶ
    if (auto* dsp = currentDSP.load(std::memory_order_acquire))
    {
        dsp->eq.cleanup();
        dsp->convolver.cleanup();
    }

    // [FIX] trashBin内のDSPCoreに対してもcleanup()を呼び、リソース解放の遅延を防ぐ
    // DSPCoreがtrashBinに移動されると、currentDSPではなくなるためタイマーからの
    // cleanup()呼び出しが止まってしまう。最終的にデストラクタでforceCleanup()が
    // 呼ばれるためリークはしないが、最大10秒間リソースが解放されない期間が発生する。
    // これを防ぐため、trashBin内のインスタンスも明示的にクリーンアップする。
    {
        const juce::ScopedLock sl(trashBinLock);
        for (const auto& entry : trashBin)
            if (entry.first != nullptr) {
                entry.first->eq.cleanup();
                entry.first->convolver.cleanup();
            }
        for (auto* p : trashBinPending)
            if (p != nullptr) {
                p->eq.cleanup();
                p->convolver.cleanup();
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
        if (activeDSP)
            activeDSP->eq.syncBandNodeFrom(uiEqProcessor, bandIndex);
    }
}

void AudioEngine::eqGlobalChanged(EQProcessor* processor)
{
    if (processor == &uiEqProcessor)
    {
        if (activeDSP) {
            // syncGlobalStateFrom は AGC の実行状態も上書きしてしまうため、
            // UIからの変更通知では、UIが管理するパラメータのみを個別に設定する。
            // これにより、アクティブなDSPのAGC状態がリセットされるのを防ぐ。
            activeDSP->eq.setTotalGain(uiEqProcessor.getTotalGain());
            activeDSP->eq.setAGCEnabled(uiEqProcessor.getAGCEnabled());
        }
    }
}

void AudioEngine::convolverParamsChanged(ConvolverProcessor* processor)
{
    if (processor == &uiConvolverProcessor)
    {
        if (activeDSP)
            activeDSP->convolver.syncParametersFrom(uiConvolverProcessor);
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

    // ==================================================================
    // 【Issue 2 完全解消】手動MMCSS revertを削除
    // 理由:
    //   1. JUCE 8.0.12 の setMMCSSModeEnabled() が内部で管理
    //   2. mmcssHandle はローカル変数だったため未定義エラー発生
    //   3. 手動revertは不要・リークリスクあり → JUCEに任せる
    // ==================================================================

    // 1. Stop Audio Thread access
    currentDSP.store(nullptr, std::memory_order_release);

    // 2. Release Active DSP (triggers destructors of DSPCore and its members)
    // 3. Clear Trash Bin (release old DSPs)
    {
        const juce::ScopedLock sl(trashBinLock);

        // 旧 DSP 群は task.currentDSP ではないため即時削除可能
        for (auto& entry : trashBin) delete entry.first;
        trashBin.clear();
        for (auto* p : trashBinPending) delete p;
        trashBinPending.clear();

        // activeDSP (= task.currentDSP) は即時削除禁止。
        // Rebuild Thread が task.currentDSP として現在アクセス中の可能性がある。
        // trashBin に移動し、timerCallback() の 2000ms GC で安全に削除。
        // 最終的に ~AudioEngine() の rebuildThread.join() 後にも削除される。
        if (activeDSP)
        {
            trashBin.push_back({activeDSP, juce::Time::getMillisecondCounter()});
            activeDSP = nullptr;
        }
    }

    // 4. Release UI Processor Resources
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
    // === パッチ1: モード設定をAudio Thread開始時に1回だけ実行 ===
    static thread_local bool threadInitialized = false;
    if (!threadInitialized) {
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
        threadInitialized = true;
    }
    // ========================================================

    // 入力検証 (Input Validation)
    if (bufferToFill.buffer == nullptr)
        return;

    const int numSamples = bufferToFill.numSamples;
    const int startSample = bufferToFill.startSample;
    auto* buffer = bufferToFill.buffer;

    // サンプル数の妥当性チェック
    // maxSamplesPerBlock.load() (Atomic) の代わりに定数 SAFE_MAX_BLOCK_SIZE を使用する。
    // これにより、Message Threadでの更新との競合を回避し、DSPCoreのバッファ確保サイズ(SAFE_MAX_BLOCK_SIZE)に基づく安全なチェックを行う。
    if (numSamples <= 0 || numSamples > SAFE_MAX_BLOCK_SIZE)
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


    // DSPコアの取得 (Atomic Load - Raw Pointer)
    // shared_ptrの参照カウント操作(atomic RMW)を回避し、完全なWait-freeを実現
    DSPCore* dsp = currentDSP.load(std::memory_order_acquire);

    if (dsp != nullptr)
    {
        // 安全対策: サンプルレート不整合チェック
        // DSPのサンプルレートとエンジンの現在のサンプルレートが一致しない場合、
        // レート変更処理中とみなし、グリッチを防ぐために無音を出力する。
        const double engineSampleRate = currentSampleRate.load(std::memory_order_relaxed);
        if (std::abs(dsp->sampleRate - engineSampleRate) > 1e-6)
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
        const bool eqBypassed = eqBypassRequested.load(std::memory_order_acquire);
        const bool convBypassed = convBypassRequested.load(std::memory_order_acquire);
        const ProcessingOrder order = currentProcessingOrder.load(std::memory_order_relaxed);
        const AnalyzerSource analyzerSource = currentAnalyzerSource.load(std::memory_order_relaxed);
        const bool softClip = softClipEnabled.load(std::memory_order_relaxed);
        const float satAmt = saturationAmount.load(std::memory_order_relaxed);
        const double headroomGain = inputHeadroomGain.load(std::memory_order_relaxed);

        const double makeupGain = outputMakeupGain.load(std::memory_order_relaxed);
        // UI表示用の状態更新
        if (eqBypassActive.load(std::memory_order_relaxed) != eqBypassed)
            eqBypassActive.store(eqBypassed, std::memory_order_relaxed);
        if (convBypassActive.load(std::memory_order_relaxed) != convBypassed)
            convBypassActive.store(convBypassed, std::memory_order_relaxed);

        // 処理委譲
        dsp->process(bufferToFill, audioFifo, audioFifoBuffer, inputLevelLinear, outputLevelLinear,
                     { .eqBypassed = eqBypassed,
                       .convBypassed = convBypassed,
                       .order = order,
                       .analyzerSource = analyzerSource,
                       .softClipEnabled = softClip,
                       .saturationAmount = satAmt,
                       .inputHeadroomGain = headroomGain,
                       .outputMakeupGain = makeupGain }); // スマートポインタでDSPを呼び出し
    }
    else
    {
        bufferToFill.clearActiveBufferRegion();
    }
}

void AudioEngine::processBlockDouble (juce::AudioBuffer<double>& buffer)
{
    static thread_local bool threadInitialized = false;
    if (!threadInitialized) {
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
        threadInitialized = true;
    }

    const int numSamples = buffer.getNumSamples();
    if (numSamples <= 0 || numSamples > SAFE_MAX_BLOCK_SIZE)
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

    const double engineSampleRate = currentSampleRate.load(std::memory_order_relaxed);
    if (std::abs(dsp->sampleRate - engineSampleRate) > 1e-6)
    {
        inputLevelLinear.store(0.0f);
        outputLevelLinear.store(0.0f);
        buffer.clear();
        return;
    }

    const bool eqBypassed = eqBypassRequested.load(std::memory_order_acquire);
    const bool convBypassed = convBypassRequested.load(std::memory_order_acquire);
    const ProcessingOrder order = currentProcessingOrder.load(std::memory_order_relaxed);
    const AnalyzerSource analyzerSource = currentAnalyzerSource.load(std::memory_order_relaxed);
    const bool softClip = softClipEnabled.load(std::memory_order_relaxed);
    const float satAmt = saturationAmount.load(std::memory_order_relaxed);
    const double headroomGain = inputHeadroomGain.load(std::memory_order_relaxed);
    const double makeupGain = outputMakeupGain.load(std::memory_order_relaxed);

    if (eqBypassActive.load(std::memory_order_relaxed) != eqBypassed)
        eqBypassActive.store(eqBypassed, std::memory_order_relaxed);
    if (convBypassActive.load(std::memory_order_relaxed) != convBypassed)
        convBypassActive.store(convBypassed, std::memory_order_relaxed);

    dsp->processDouble(buffer, audioFifo, audioFifoBuffer, inputLevelLinear, outputLevelLinear,
                       { .eqBypassed = eqBypassed,
                         .convBypassed = convBypassed,
                         .order = order,
                         .analyzerSource = analyzerSource,
                         .softClipEnabled = softClip,
                         .saturationAmount = satAmt,
                       .inputHeadroomGain = headroomGain,
                       .outputMakeupGain = makeupGain });
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

    processInput(bufferToFill, numSamples, state.inputHeadroomGain);

    //----------------------------------------------------------
    // AudioBlockの構築 (AlignedBufferを使用)
    //----------------------------------------------------------
    double* channels[2] = { alignedL.get(), alignedR.get() };
    juce::dsp::AudioBlock<double> processBlock(channels, 2, numSamples);

    //----------------------------------------------------------
    // 入力レベル計算
    //----------------------------------------------------------
    const float inputLinear = measureLevel(processBlock);
    inputLevelLinear.store(inputLinear, std::memory_order_relaxed);

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

    // ── Analyzer Input Tap (Pre-DSP) ──
    if (state.analyzerSource == AnalyzerSource::Input)
    {
        pushToFifo(processBlock, audioFifo, audioFifoBuffer);
    }

    int numProcSamples = (int)processBlock.getNumSamples();
    int numProcChannels = (int)processBlock.getNumChannels(); // 通常は2

    //----------------------------------------------------------
    // DSP処理チェーン (Dynamic Processing Order)
    //----------------------------------------------------------
    // プロセッサには AudioBlock を直接渡す (AudioBuffer作成によるmalloc回避)
    if (state.order == ProcessingOrder::ConvolverThenEQ) // stateから読み出し
    {
        // 1. Convolver
        if (!state.convBypassed) // stateから読み出し
            convolver.process(processBlock);
        // 2. EQ
        if (!state.eqBypassed) // stateから読み出し
            eq.process(processBlock);
    }
    else
    {
        // 1. EQ
        if (!state.eqBypassed) // stateから読み出し
            eq.process(processBlock);
        // 2. Convolver
        if (!state.convBypassed) // stateから読み出し
            convolver.process(processBlock);
    }

    // Output Makeup Gain
    for (size_t ch = 0; ch < processBlock.getNumChannels(); ++ch)
        cblas_dscal((int)processBlock.getNumSamples(), state.outputMakeupGain, processBlock.getChannelPointer(ch), 1);

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
        const double CLIP_START = CLIP_THRESHOLD - CLIP_KNEE;

        for (int ch = 0; ch < numProcChannels; ++ch)
        {
            double* data = processBlock.getChannelPointer(ch);
            softClipBlockAVX2(data, numProcSamples, CLIP_THRESHOLD, CLIP_KNEE, CLIP_ASYMMETRY);
        }
    }

    //----------------------------------------------------------

    // ── Analyzer Output Tap (Post-DSP) ──
    if (state.analyzerSource == AnalyzerSource::Output)
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

    processOutput(bufferToFill, numSamples);

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

    processInputDouble(buffer, numSamples, state.inputHeadroomGain);

    double* channels[2] = { alignedL.get(), alignedR.get() };
    juce::dsp::AudioBlock<double> processBlock(channels, 2, numSamples);

    const float inputLinear = measureLevel(processBlock);
    inputLevelLinear.store(inputLinear, std::memory_order_relaxed);

    juce::dsp::AudioBlock<double> originalBlock = processBlock;

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

    if (state.analyzerSource == AnalyzerSource::Input)
        pushToFifo(processBlock, audioFifo, audioFifoBuffer);

    const int numProcSamples = static_cast<int>(processBlock.getNumSamples());
    const int numProcChannels = static_cast<int>(processBlock.getNumChannels());

    if (state.order == ProcessingOrder::ConvolverThenEQ)
    {
        if (!state.convBypassed)
            convolver.process(processBlock);
        if (!state.eqBypassed)
            eq.process(processBlock);
    }
    else
    {
        if (!state.eqBypassed)
            eq.process(processBlock);
        if (!state.convBypassed)
            convolver.process(processBlock);
    }

    // Output Makeup Gain
    for (size_t ch = 0; ch < processBlock.getNumChannels(); ++ch)
        cblas_dscal((int)processBlock.getNumSamples(), state.outputMakeupGain, processBlock.getChannelPointer(ch), 1);

    if (state.softClipEnabled)
    {
        const double sat = static_cast<double>(state.saturationAmount);
        const double CLIP_THRESHOLD = 0.95 - 0.45 * sat;
        const double CLIP_KNEE      = 0.05 + 0.35 * sat;
        const double CLIP_ASYMMETRY = 0.10 * sat;
        const double CLIP_START = CLIP_THRESHOLD - CLIP_KNEE;

        for (int ch = 0; ch < numProcChannels; ++ch)
        {
            double* data = processBlock.getChannelPointer(ch);
            softClipBlockAVX2(data, numProcSamples, CLIP_THRESHOLD, CLIP_KNEE, CLIP_ASYMMETRY);
        }
    }

    if (state.analyzerSource == AnalyzerSource::Output)
        pushToFifo(processBlock, audioFifo, audioFifoBuffer);

    if (oversamplingFactor > 1)
    {
        oversampling.processDown(processBlock, originalBlock, static_cast<int>(originalBlock.getNumChannels()));
        processBlock = originalBlock;
    }

    const float outputLinear = measureLevel(originalBlock);
    outputLevelLinear.store(outputLinear, std::memory_order_relaxed);

    processOutputDouble(buffer, numSamples);

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
        const double level = std::max(std::abs(range.getStart()), std::abs(range.getEnd()));
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

void AudioEngine::DSPCore::processInput(const juce::AudioSourceChannelInfo& bufferToFill, int numSamples, double headroomGain) noexcept
{
    auto* buffer = bufferToFill.buffer;
    const int startSample = bufferToFill.startSample;
    const int effectiveInputChannels = std::min(buffer->getNumChannels(), 2);

    for (int ch = 0; ch < effectiveInputChannels; ++ch)
    {
        const float* src = buffer->getReadPointer(ch, startSample);
        double* dst = (ch == 0) ? alignedL.get() : alignedR.get();
        convo::input_transform::convertFloatToDoubleHighQuality(src, dst, numSamples, headroomGain);
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
        std::memcpy(dst, src, numSamples * sizeof(double));
    }

    // ── 入力段DC除去 (ブロックモード) ──
    // 旧: 1 サンプルずつ 4 次 IIR を呼出 (~20 ops/sample)
    // 新: UltraHighRateDCBlocker.process(ptr, N) でブロック単位処理 (~4 ops/sample)
    double* lPtr = alignedL.get();
    double* rPtr = alignedR.get();
    inputDCBlockerL.process(lPtr, numSamples);
    inputDCBlockerR.process(rPtr, numSamples);
}

void AudioEngine::DSPCore::processInputDouble(const juce::AudioBuffer<double>& buffer, int numSamples, double headroomGain) noexcept
{
    const int effectiveInputChannels = std::min(buffer.getNumChannels(), 2);

    for (int ch = 0; ch < effectiveInputChannels; ++ch)
    {
        const double* src = buffer.getReadPointer(ch);
        double* dst = (ch == 0) ? alignedL.get() : alignedR.get();
        convo::input_transform::convertDoubleToDoubleHighQuality(src, dst, numSamples, headroomGain);
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

    // ── 入力段DC除去 (ブロックモード) ──
    double* lPtr = alignedL.get();
    double* rPtr = alignedR.get();
    inputDCBlockerL.process(lPtr, numSamples);
    inputDCBlockerR.process(rPtr, numSamples);
}

// 音楽的なソフトクリッピング関数
// 閾値を超えた信号を滑らかにクリップし、真空管アンプのような温かみのある歪みを加える。
// @param x 入力信号
// @param threshold クリッピングが開始される閾値 (正の値)
// @param knee 閾値周辺のカーブの滑らかさ（ニー） (正の値)
// @param asymmetry 非対称性の量。負の波形をより強くクリップし、偶数次倍音を生成する。
double AudioEngine::DSPCore::musicalSoftClip(double x, double threshold, double knee, double asymmetry) noexcept
{
    return musicalSoftClipScalar(x, threshold, knee, asymmetry);
}

void AudioEngine::DSPCore::processOutput(const juce::AudioSourceChannelInfo& bufferToFill, int numSamples) noexcept
{
    auto* buffer = bufferToFill.buffer;
    const int startSample = bufferToFill.startSample;
    constexpr double kOutputHeadroom = 0.988553; // 約 -0.1dB

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

    // DCブロッカーの状態をロード
    dcBlockerL.loadState();
    if (dataR) dcBlockerR.loadState();

    const __m256d vMax = _mm256_set1_pd(1.0);
    const __m256d vMin = _mm256_set1_pd(-1.0);
    const __m256d vHeadroom = applyDither ? _mm256_set1_pd(1.0)
                                          : _mm256_set1_pd(kOutputHeadroom);

    for (int i = 0; i < numSamples; ++i)
    {
        // --- DC除去 ---
        double sampleL = dataL[i];
        dcBlockerL.processSample(sampleL);

        double sampleR = 0.0;
        if (dataR)
        {
            sampleR = dataR[i];
            dcBlockerR.processSample(sampleR);
        }

        // --- ディザリング ---
        if (applyDither)
        {
            sampleL = dither.process(sampleL * kOutputHeadroom, 0);
            if (dataR)
                sampleR = dither.process(sampleR * kOutputHeadroom, 1);
        }

        // --- 変換・クランプ ---
        double finalL = sampleL;
        double finalR = sampleR;

        if (!applyDither)
        {
            finalL *= kOutputHeadroom;
            if (dataR)
                finalR *= kOutputHeadroom;
        }

        if (!std::isfinite(finalL)) finalL = 0.0;
        if (!std::isfinite(finalR)) finalR = 0.0;

        dstL[i] = static_cast<float>(juce::jlimit(-1.0, 1.0, finalL));
        if (dstR)
            dstR[i] = static_cast<float>(juce::jlimit(-1.0, 1.0, finalR));
    }

    // DCブロッカーの状態を保存
    dcBlockerL.saveState();
    if (dataR) dcBlockerR.saveState();

    // 3ch以降は使用しないためクリア (ゴミデータ出力防止)
    for (int ch = numChannels; ch < buffer->getNumChannels(); ++ch)
        buffer->clear(ch, startSample, numSamples);
}

void AudioEngine::DSPCore::processOutputDouble(juce::AudioBuffer<double>& buffer, int numSamples) noexcept
{
    constexpr double kOutputHeadroom = 0.988553; // 約 -0.1dB
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
    {
        if (ch < 2)
        {
            double* data = (ch == 0) ? alignedL.get() : alignedR.get();
            auto& blocker = (ch == 0) ? dcBlockerL : dcBlockerR;
            double* dst = buffer.getWritePointer(ch);

            // 【最適化】ブロックモード DC 除去 (1次IIR)
            blocker.process(data, numSamples);

            // NaN 対策: vmaxpd(NaN, -1.0)→-1.0 (Intel仕様: 第1op=NaN→第2opを返す)
            // Inf 対策: min/max でクランプ済み
            // デノーマル対策: FTZ/DAZ が有効なため vEpsilon blendv チェックは省略する。
            {
                int i = 0;
                const int vEnd = numSamples / 16 * 16;
                const __m256d vMax      = _mm256_set1_pd(1.0);
                const __m256d vMin      = _mm256_set1_pd(-1.0);
                const __m256d vHeadroom = _mm256_set1_pd(kOutputHeadroom);

                for (; i < vEnd; i += 16)
                {
                    __m256d v0 = _mm256_mul_pd(_mm256_loadu_pd(data + i),      vHeadroom);
                    __m256d v1 = _mm256_mul_pd(_mm256_loadu_pd(data + i + 4),  vHeadroom);
                    __m256d v2 = _mm256_mul_pd(_mm256_loadu_pd(data + i + 8),  vHeadroom);
                    __m256d v3 = _mm256_mul_pd(_mm256_loadu_pd(data + i + 12), vHeadroom);

                    v0 = _mm256_min_pd(_mm256_max_pd(v0, vMin), vMax);
                    v1 = _mm256_min_pd(_mm256_max_pd(v1, vMin), vMax);
                    v2 = _mm256_min_pd(_mm256_max_pd(v2, vMin), vMax);
                    v3 = _mm256_min_pd(_mm256_max_pd(v3, vMin), vMax);

                    _mm256_storeu_pd(dst + i,      v0);
                    _mm256_storeu_pd(dst + i + 4,  v1);
                    _mm256_storeu_pd(dst + i + 8,  v2);
                    _mm256_storeu_pd(dst + i + 12, v3);
                }

                for (; i < (numSamples / 4 * 4); i += 4)
                {
                    __m256d v = _mm256_mul_pd(_mm256_loadu_pd(data + i), vHeadroom);
                    v = _mm256_min_pd(_mm256_max_pd(v, vMin), vMax);
                    _mm256_storeu_pd(dst + i, v);
                }

                for (; i < numSamples; ++i)
                {
                    double v = data[i] * kOutputHeadroom;
                    if (!std::isfinite(v)) v = 0.0;
                    dst[i] = juce::jlimit(-1.0, 1.0, v);
                }
            }
        }
        else
        {
            buffer.clear(ch, 0, numSamples);
        }
    }
}

void AudioEngine::setEqBypassRequested (bool shouldBypass) noexcept
{
    eqBypassRequested.store (shouldBypass, std::memory_order_release);
}

void AudioEngine::setConvolverBypassRequested (bool shouldBypass) noexcept
{
    convBypassRequested.store (shouldBypass, std::memory_order_release);
}

void AudioEngine::setConvolverUseMinPhase(bool useMinPhase)
{
    uiConvolverProcessor.setUseMinPhase(useMinPhase);
}

bool AudioEngine::getConvolverUseMinPhase() const
{
    return uiConvolverProcessor.getUseMinPhase();
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
    uiConvolverProcessor.loadImpulseResponse (irFile);
}

void AudioEngine::requestLoadState (const juce::ValueTree& state)
{
    // グローバル設定の読み込み
    if (state.hasProperty("processingOrder"))
        setProcessingOrder((ProcessingOrder)(int)state.getProperty("processingOrder"));

    if (state.hasProperty("softClipEnabled"))
        setSoftClipEnabled(state.getProperty("softClipEnabled"));

    if (state.hasProperty("saturationAmount"))
        setSaturationAmount(state.getProperty("saturationAmount"));

    if (state.hasProperty("inputHeadroomDb"))
        setInputHeadroomDb(state.getProperty("inputHeadroomDb"));

    if (state.hasProperty("outputMakeupDb"))
        setOutputMakeupDb(state.getProperty("outputMakeupDb"));

    if (state.hasProperty("analyzerSource"))
        setAnalyzerSource((AnalyzerSource)(int)state.getProperty("analyzerSource"));

    if (state.hasProperty("eqBypassed"))
    {
        bool bypassed = state.getProperty("eqBypassed");
        setEqBypassRequested(bypassed);
        uiEqProcessor.setBypass(bypassed);
    }

    if (state.hasProperty("convBypassed"))
    {
        bool bypassed = state.getProperty("convBypassed");
        setConvolverBypassRequested(bypassed);
        // ConvolverProcessor::setState でも設定される可能性があるが、
        // 整合性を保つためにここでも設定する
        uiConvolverProcessor.setBypass(bypassed);
    }

    // EQ
    auto eqState = state.getChildWithName ("EQ");
    if (eqState.isValid())
        uiEqProcessor.setState (eqState);

    // Convolver
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
    state.setProperty("eqBypassed", eqBypassRequested.load(), nullptr);
    state.setProperty("convBypassed", convBypassRequested.load(), nullptr);

    state.addChild (uiEqProcessor.getState(), -1, nullptr);
    state.addChild (uiConvolverProcessor.getState(), -1, nullptr);
    return state;
}

void AudioEngine::setInputHeadroomDb(float db)
{
    float clampedDb = juce::jlimit(-12.0f, -6.0f, db);
    if (std::abs(inputHeadroomDb.load() - clampedDb) > 1e-5f)
    {
        inputHeadroomDb.store(clampedDb);
        inputHeadroomGain.store(juce::Decibels::decibelsToGain((double)clampedDb));
    }
}

float AudioEngine::getInputHeadroomDb() const
{
    return inputHeadroomDb.load();
}

void AudioEngine::setOutputMakeupDb(float db)
{
    float clampedDb = juce::jlimit(12.0f, 17.0f, db);
    if (std::abs(outputMakeupDb.load() - clampedDb) > 1e-5f)
    {
        outputMakeupDb.store(clampedDb);
        outputMakeupGain.store(juce::Decibels::decibelsToGain((double)clampedDb));
    }
}

float AudioEngine::getOutputMakeupDb() const
{
    return outputMakeupDb.load();
}

void AudioEngine::setDitherBitDepth(int bitDepth)
{
    if (ditherBitDepth.load() != bitDepth)
    {
        ditherBitDepth.store(bitDepth);
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

void AudioEngine::setSoftClipEnabled(bool enabled)
{
    softClipEnabled.store(enabled, std::memory_order_relaxed);
}

bool AudioEngine::isSoftClipEnabled() const
{
    return softClipEnabled.load(std::memory_order_relaxed);
}

void AudioEngine::setSaturationAmount(float amount)
{
    saturationAmount.store(juce::jlimit(0.0f, 1.0f, amount), std::memory_order_relaxed);
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
