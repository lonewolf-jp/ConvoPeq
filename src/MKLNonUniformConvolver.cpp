//============================================================================
// MKLNonUniformConvolver.cpp  ── v2.1 (JUCE 8.0.12 / Intel oneMKL + IPP 対応)
//
// v2.1 変更点: Audio Thread 内 FFT を MKL DFTI → Intel IPP に換装。
//
// ■ IPP FFT 換装の設計根拠:
//   MKL DFTI は内部スレッドプールへのアクセスを行う可能性があり、
//   DFTI_THREAD_LIMIT=1 / mkl_set_num_threads(1) を設定しても
//   スレッド同期オーバーヘッドがゼロにならない場合がある。
//   Intel IPP FFT (ippsFFTFwd_RToCCS_64f / ippsFFTInv_CCSToR_64f) は
//   完全シングルスレッド設計であり、Audio Thread のリアルタイム性を最大化する。
//
// ■ バッファ互換性:
//   IPP CCS 出力形式: [re0,im0,re1,im1,...] (kFftLength/2+1)*2 doubles
//   MKL DFTI_COMPLEX_COMPLEX 出力と同一レイアウト。
//   → fdlBuf / irFreqDomain の既存 AVX2 複素乗算コードは無変更で動作する。
//
// ■ 正規化:
//   IPP_FFT_DIV_INV_BY_N フラグ使用 → IFFT 時に 1/N 正規化自動適用。
//   MKL の DFTI_BACKWARD_SCALE = 1/N 設定と完全等価。
//
// ■ ワークバッファ事前確保:
//   SetImpulse() 内で ippsFFTGetSize_R_64f が返す sizeWork 分を事前確保。
//   sizeWork == 0 の場合のみ fftWorkBuf = nullptr (IPP が外部バッファ不要)。
//   Audio Thread (processLayerBlock / Add の分散ループ) でのメモリ確保はゼロ。
//
// ■ 継続使用する MKL 機能 (VML/BLAS: Message Thread のみ):
//   mkl_malloc / mkl_free   : オーディオデータバッファ確保
//   vdMul (MKL VML)         : applySpectrumFilter での周波数ゲイン適用
//   cblas_dscal (MKL BLAS)  : IR スケーリング
//
//============================================================================

#include <JuceHeader.h>
#include "MKLNonUniformConvolver.h"
#include "MKLRealTimeSetup.h"
#include "AlignedAllocation.h"
#include "DspNumericPolicy.h"

#include <mkl.h>        // mkl_malloc, mkl_free, mkl_set_num_threads
#include <mkl_vml.h>    // vdMul
#include <mkl_cblas.h>  // cblas_dscal
#include <ipp.h>       // ippsFFTFwd_RToCCS_64f, ippsFFTInv_CCSToR_64f (MKL DFTI 代替)
#include <cstring>
#include <algorithm>
#include <cmath>
#include <cstdint>

#include <immintrin.h>  // AVX2

namespace convo
{

namespace
{
inline double hsum256_pd(__m256d v) noexcept
{
    const __m128d hi = _mm256_extractf128_pd(v, 1);
    const __m128d lo = _mm256_castpd256_pd128(v);
    const __m128d sum = _mm_add_pd(lo, hi);
    const __m128d shuf = _mm_shuffle_pd(sum, sum, 0x1);
    return _mm_cvtsd_f64(_mm_add_sd(sum, shuf));
}

inline bool isFiniteAndAboveThresholdMask(double value, double threshold) noexcept
{
    const __m128d v = _mm_set1_pd(value);
    const __m128d diff = _mm_sub_pd(v, v);
    const __m128d finiteMask = _mm_cmpeq_pd(diff, _mm_setzero_pd());

    const __m128d signMask = _mm_set1_pd(-0.0);
    const __m128d absV = _mm_andnot_pd(signMask, v);
    const __m128d thresholdV = _mm_set1_pd(threshold);
    const __m128d denormalMask = _mm_cmplt_pd(absV, thresholdV);

    const __m128d validMask = _mm_andnot_pd(denormalMask, finiteMask);
    return _mm_movemask_pd(validMask) == 0x3;
}

inline double computeTailGainForBin(int k,
                                    int numBins,
                                    double sampleRate,
                                    double startHz,
                                    double strength) noexcept
{
    if (strength <= 0.0 || sampleRate <= 0.0 || numBins <= 1)
        return 1.0;

    const double nyquist = sampleRate * 0.5;
    if (nyquist <= 1.0)
        return 1.0;

    const double safeStartHz = std::max(0.0, std::min(startHz, nyquist - 1.0));
    const double f = static_cast<double>(k) * nyquist / static_cast<double>(numBins - 1);

    if (f <= safeStartHz)
        return 1.0;

    const double denom = std::max(1.0, nyquist - safeStartHz);
    const double x = juce::jlimit(0.0, 1.0, (f - safeStartHz) / denom);
    return std::exp(-strength * x * x);
}
} // namespace

//==============================================================================
// Layer::freeAll
//==============================================================================
void MKLNonUniformConvolver::Layer::freeAll() noexcept
{
    // [v2.1] IPP FFT リソース解放
    // fftSpec は fftSpecBuf 内を指すポインタなので、fftSpecBuf の解放で自動的に無効化。
    if (fftSpecBuf)
    {
        ippsFree(fftSpecBuf);
        fftSpecBuf = nullptr;
        fftSpec    = nullptr;
    }
    if (fftWorkBuf)
    {
        ippsFree(fftWorkBuf);
        fftWorkBuf = nullptr;
    }
    descriptorCommitted = false;

    if (irFreqDomain)  { mkl_free(irFreqDomain);  irFreqDomain  = nullptr; }
    if (fdlBuf)        { mkl_free(fdlBuf);         fdlBuf        = nullptr; }
    if (fftTimeBuf)    { mkl_free(fftTimeBuf);     fftTimeBuf    = nullptr; }
    if (fftOutBuf)     { mkl_free(fftOutBuf);      fftOutBuf     = nullptr; }
    if (prevInputBuf)  { mkl_free(prevInputBuf);   prevInputBuf  = nullptr; }
    if (accumBuf)      { mkl_free(accumBuf);       accumBuf      = nullptr; }
    if (inputAccBuf)   { mkl_free(inputAccBuf);    inputAccBuf   = nullptr; }
    if (tailOutputBuf) { mkl_free(tailOutputBuf);  tailOutputBuf = nullptr; }

    fftSize = partSize = numParts = numPartsIR = 0;
    fdlMask = complexSize = partStride = 0;
    fdlIndex = inputPos = partsPerCallback = nextPart = 0;
    tailOutputPos    = 0;
    baseFdlIdxSaved  = 0;
    distributing     = false;
    isImmediate      = false;
}

//==============================================================================
// コンストラクタ / デストラクタ
//==============================================================================
MKLNonUniformConvolver::MKLNonUniformConvolver()
{
    // MKL VML / CBLAS (applySpectrumFilter) のスレッド数を制限。
    // IPP は単一スレッド設計のため、この設定は MKL 依存部のみに影響する。
    mkl_set_num_threads(1);
}

MKLNonUniformConvolver::~MKLNonUniformConvolver()
{
    releaseAllLayers();
}

//==============================================================================
// applySpectrumFilter  ─ Message Thread のみ
//==============================================================================
void MKLNonUniformConvolver::applySpectrumFilter(const FilterSpec& spec) noexcept
{
    const double fs      = spec.sampleRate;
    const double nyquist = fs * 0.5;
    const int tailMode   = juce::jlimit(0, 1, spec.tailMode);
    const double tailStartHz = juce::jlimit(20.0, 20000.0, static_cast<double>(spec.tailRolloffStartHz));
    const double baseTailStrength = juce::jlimit(0.0, 2.0, static_cast<double>(spec.tailRolloffStrength));
    const double partitionStrength = juce::jlimit(0.0, 2.0, static_cast<double>(spec.partitionTailStrength));

    const double hcFcStart = (fs <= 48000.0) ? 18000.0 : 22000.0;
    const double hcFcEnd   = nyquist;

    const double lcFcEnd   = (spec.lcMode == LCMode::Soft) ?  6.0 :  8.0;
    const double lcFcStart = (spec.lcMode == LCMode::Soft) ? 15.0 : 18.0;

    for (int li = 0; li < m_numActiveLayers; ++li)
    {
        Layer& l = m_layers[li];
        if (!l.irFreqDomain) continue;

        const int N      = l.fftSize;
        const int halfN  = N / 2;
        const int cSize  = l.complexSize;
        const int stride = l.partStride;

        convo::ScopedAlignedPtr<double> gain(
            static_cast<double*>(mkl_malloc(static_cast<size_t>(cSize) * sizeof(double), 64)));
        if (!gain.get())
        {
            juce::Logger::writeToLog("MKLNonUniformConvolver: OOM in applySpectrumFilter for layer " + juce::String(li));
            continue;
        }
        std::fill_n(gain.get(), cSize, 1.0);

        // ── HC ゲイン ──
        {
            const int kStart = static_cast<int>(std::round(hcFcStart * N / fs));
            const int kEnd   = std::min(halfN,
                                        static_cast<int>(std::round(hcFcEnd * N / fs)));

            for (int k = 0; k < cSize; ++k)
            {
                if (k <= kStart)
                {
                    // パスバンド: ゲイン 1.0
                }
                else if (k <= kEnd)
                {
                    const double denom = static_cast<double>(kEnd - kStart);
                    const double x     = static_cast<double>(k - kStart) / denom;

                    switch (spec.hcMode)
                    {
                    case HCMode::Sharp:
                        gain[k] = 1.0 / std::sqrt(1.0 + std::pow(x, 8.0));
                        break;
                    case HCMode::Natural:
                        gain[k] = 0.5 * (1.0 + std::cos(
                            juce::MathConstants<double>::pi * x));
                        break;
                    case HCMode::Soft:
                        gain[k] = std::exp(-4.60517 * x * x);
                        break;
                    }
                }
                else
                {
                    gain[k] = 0.0;
                }
            }
        }

        // ── LC ゲイン ──
        {
            const int kEnd   = static_cast<int>(std::round(lcFcEnd   * N / fs));
            const int kStart = static_cast<int>(std::round(lcFcStart * N / fs));

            for (int k = 0; k < cSize; ++k)
            {
                if (k <= kEnd)
                {
                    gain[k] = 0.0;
                }
                else if (k < kStart)
                {
                    const double denom = static_cast<double>(
                        std::max(1, kStart - kEnd));
                    const double x     = static_cast<double>(k - kEnd) / denom;
                    const double g_lc  = 0.5 * (1.0 - std::cos(
                        juce::MathConstants<double>::pi * x));
                    gain[k] *= g_lc;
                }
            }
        }

        // ── Tail ゲイン ──
        double layerTailStrength = baseTailStrength;
        if (tailMode == 1)
            layerTailStrength = (li == 0) ? 0.0 : juce::jlimit(0.0, 4.0, baseTailStrength * partitionStrength);

        if (layerTailStrength > 0.0)
        {
            for (int k = 0; k < cSize; ++k)
                gain[k] *= computeTailGainForBin(k, cSize, fs, tailStartHz, layerTailStrength);
        }

        // ── 全パーティションの irFreqDomain に gain[] を適用 ──
        {
            convo::ScopedAlignedPtr<double> gainIL(
                static_cast<double*>(mkl_malloc(static_cast<size_t>(cSize) * 2 * sizeof(double), 64)));
            if (!gainIL.get()) continue;
            for (int k = 0; k < cSize; ++k)
                gainIL.get()[2 * k] = gainIL.get()[2 * k + 1] = gain[k];

            for (int p = 0; p < l.numParts; ++p)
            {
                double* slot = l.irFreqDomain + p * stride;
                vdMul(cSize * 2, slot, gainIL.get(), slot);
            }
        }
    }
}

//==============================================================================
void MKLNonUniformConvolver::releaseAllLayers() noexcept
{
    for (int i = 0; i < kNumLayers; ++i)
        m_layers[i].freeAll();
    m_numActiveLayers = 0;
    m_latency         = 0;

    if (m_ringBuf) { mkl_free(m_ringBuf); m_ringBuf = nullptr; }
    m_ringSize = m_ringMask = m_ringWrite = m_ringRead = m_ringAvail = 0;

    if (m_directIRRev)   { mkl_free(m_directIRRev);   m_directIRRev = nullptr; }
    if (m_directHistory) { mkl_free(m_directHistory); m_directHistory = nullptr; }
    if (m_directWindow)  { mkl_free(m_directWindow);  m_directWindow = nullptr; }
    if (m_directOutBuf)  { mkl_free(m_directOutBuf);  m_directOutBuf = nullptr; }

    m_directTapCount = 0;
    m_directHistLen  = 0;
    m_directMaxBlock = 0;
    m_directPendingSamples = 0;
    m_directEnabled  = false;
}

//==============================================================================
// SetImpulse  ─ Message Thread のみ
//==============================================================================
bool MKLNonUniformConvolver::SetImpulse(const double* impulse, int irLen, int blockSize, double scale,
                                        bool enableDirectHead,
                                        const FilterSpec* filterSpec)
{
    m_ready.store(false, std::memory_order_release);

    if (impulse == nullptr || irLen <= 0 || blockSize <= 0)
        return false;

    releaseAllLayers();

    // ────────────────────────────────────────────────
    // 先頭 Direct Form 設定
    // ────────────────────────────────────────────────
    constexpr int kMaxDirectTaps = 32;
    const int directPart = juce::nextPowerOfTwo(std::max(blockSize, 64));
    m_directTapCount = (enableDirectHead ? std::min(irLen, std::min(directPart, kMaxDirectTaps)) : 0);
    m_directHistLen  = std::max(0, m_directTapCount - 1);
    m_directMaxBlock = std::max(blockSize, 1);
    m_directPendingSamples = 0;
    m_directEnabled  = (m_directTapCount > 0);

    if (m_directEnabled)
    {
        m_directIRRev   = static_cast<double*>(mkl_malloc(static_cast<size_t>(m_directTapCount) * sizeof(double), 64));
        m_directHistory = (m_directHistLen > 0)
            ? static_cast<double*>(mkl_malloc(static_cast<size_t>(m_directHistLen) * sizeof(double), 64))
            : nullptr;
        m_directWindow  = static_cast<double*>(mkl_malloc(static_cast<size_t>(m_directHistLen + m_directMaxBlock) * sizeof(double), 64));
        m_directOutBuf  = static_cast<double*>(mkl_malloc(static_cast<size_t>(m_directMaxBlock) * sizeof(double), 64));

        if (!m_directIRRev || !m_directWindow || !m_directOutBuf || (m_directHistLen > 0 && !m_directHistory))
        {
            releaseAllLayers();
            return false;
        }

        if (m_directHistLen > 0)
            memset(m_directHistory, 0, static_cast<size_t>(m_directHistLen) * sizeof(double));
        memset(m_directOutBuf, 0, static_cast<size_t>(m_directMaxBlock) * sizeof(double));

        for (int i = 0; i < m_directTapCount; ++i)
            m_directIRRev[i] = impulse[m_directTapCount - 1 - i] * scale;
    }

    convo::ScopedAlignedPtr<double> impulseForFft(
        static_cast<double*>(mkl_malloc(static_cast<size_t>(irLen) * sizeof(double), 64)));
    if (!impulseForFft.get())
    {
        releaseAllLayers();
        return false;
    }

    memcpy(impulseForFft.get(), impulse, static_cast<size_t>(irLen) * sizeof(double));

    if (m_directEnabled)
        memset(impulseForFft.get(), 0, static_cast<size_t>(m_directTapCount) * sizeof(double));

    // NOTE: vmlSetMode はここでは呼ばない (MainApplication::initialise() 設定済み)。

    // ────────────────────────────────────────────────
    // レイヤー構成決定 (Non-Uniform Partitioned Convolution)
    // ────────────────────────────────────────────────
    const int l0Part = juce::nextPowerOfTwo(std::max(blockSize, 64));
    const int l1Part = l0Part * 8;
    const int l2Part = l1Part * 8;

    const int l0Len = std::min(irLen, kL0MaxParts * l0Part);

    const int l1Offset = l0Len;
    const int l1Len    = std::max(0, std::min(irLen - l0Len, kL1MaxParts * l1Part));

    const int l2Offset = l0Len + l1Len;
    const int l2Len    = std::max(0, irLen - l0Len - l1Len);

    struct LayerCfg { int offset; int len; int partSize; bool immediate; };
    const LayerCfg cfgs[kNumLayers] = {
        { 0,        l0Len, l0Part, true  },
        { l1Offset, l1Len, l1Part, false },
        { l2Offset, l2Len, l2Part, false },
    };

    m_numActiveLayers = 0;

    // ────────────────────────────────────────────────
    // 各レイヤーを初期化
    // ────────────────────────────────────────────────
    for (int li = 0; li < kNumLayers; ++li)
    {
        if (cfgs[li].len <= 0)
            continue;

        Layer& l = m_layers[m_numActiveLayers];
        l.descriptorCommitted = false;

        l.partSize    = cfgs[li].partSize;
        l.fftSize     = l.partSize * 2;
        l.isImmediate = cfgs[li].immediate;

        l.complexSize = l.fftSize / 2 + 1;
        l.partStride  = (l.complexSize * 2 + 7) & ~7;

        l.numPartsIR = (cfgs[li].len + l.partSize - 1) / l.partSize;
        l.numParts   = juce::nextPowerOfTwo(l.numPartsIR);
        l.fdlMask    = l.numParts - 1;

        // ── IPP FFT スペック初期化 (Message Thread で事前確保) ──
        //
        // [v2.1] MKL DFTI_DESCRIPTOR_HANDLE の代替。
        // fftSize = 2 * partSize は常に 2 の冪なので FFT (DFT より高速) を使用可能。
        // IPP_FFT_DIV_INV_BY_N: IFFT 時に 1/N 正規化を自動適用
        //   (旧: DftiSetValue(DFTI_BACKWARD_SCALE, 1.0/fftSize) と等価)
        // ippAlgHintFast: 速度優先 (精度を一切犠牲にしない範囲でのヒント)
        {
            // fftSize = 2^order を求める (fftSize は必ず 2 の冪)
            int order = 0;
            {
                int tmp = l.fftSize;
                while (tmp > 1) { tmp >>= 1; ++order; }
            }

            int sizeSpec = 0, sizeInit = 0, sizeWork = 0;
            const IppStatus getSt = ippsFFTGetSize_R_64f(
                order, IPP_FFT_DIV_INV_BY_N, ippAlgHintFast,
                &sizeSpec, &sizeInit, &sizeWork);
            if (getSt != ippStsNoErr)
            {
                juce::Logger::writeToLog("MKLNonUniformConvolver: ippsFFTGetSize_R_64f failed for layer "
                                         + juce::String(li) + " (status=" + juce::String(static_cast<int>(getSt)) + ")");
                releaseAllLayers();
                return false;
            }

            // スペックバッファ確保 (IPP FFT スペックのメモリオーナー)
            l.fftSpecBuf = ippsMalloc_8u(sizeSpec);
            if (!l.fftSpecBuf)
            {
                juce::Logger::writeToLog("MKLNonUniformConvolver: ippsMalloc_8u(sizeSpec) failed for layer " + juce::String(li));
                releaseAllLayers();
                return false;
            }

            // 初期化用一時バッファ (不要なら nullptr)
            Ipp8u* initBuf = (sizeInit > 0) ? ippsMalloc_8u(sizeInit) : nullptr;

            const IppStatus initSt = ippsFFTInit_R_64f(
                &l.fftSpec, order, IPP_FFT_DIV_INV_BY_N, ippAlgHintFast,
                l.fftSpecBuf, initBuf);

            if (initBuf)
            {
                ippsFree(initBuf);
                initBuf = nullptr;
            }

            if (initSt != ippStsNoErr || l.fftSpec == nullptr)
            {
                juce::Logger::writeToLog("MKLNonUniformConvolver: ippsFFTInit_R_64f failed for layer "
                                         + juce::String(li) + " (status=" + juce::String(static_cast<int>(initSt)) + ")");
                releaseAllLayers();
                return false;
            }

            // ワークバッファ確保 (Audio Thread での動的確保を防ぐため事前確保)
            // sizeWork == 0 の場合 nullptr のまま (IPP が外部バッファ不要)
            // sizeWork > 0 かつ確保失敗 → リアルタイム安全でないため初期化失敗とする
            if (sizeWork > 0)
            {
                l.fftWorkBuf = ippsMalloc_8u(sizeWork);
                if (!l.fftWorkBuf)
                {
                    juce::Logger::writeToLog("MKLNonUniformConvolver: ippsMalloc_8u(sizeWork=" + juce::String(sizeWork)
                                             + ") failed for layer " + juce::String(li));
                    releaseAllLayers();
                    return false;
                }
            }
            // else: sizeWork == 0 → l.fftWorkBuf は nullptr のまま (正常)

            l.descriptorCommitted = true;
        }

        // ── バッファ確保 (すべて mkl_malloc 64byte アライン) ──
        const size_t irBufSize  = static_cast<size_t>(l.numParts) * l.partStride;
        const size_t fdlBufSize = static_cast<size_t>(l.numParts) * 2 * l.partStride;

        l.irFreqDomain = static_cast<double*>(mkl_malloc(irBufSize  * sizeof(double), 64));
        l.fdlBuf       = static_cast<double*>(mkl_malloc(fdlBufSize * sizeof(double), 64));
        l.fftTimeBuf   = static_cast<double*>(mkl_malloc(l.fftSize   * sizeof(double), 64));
        l.fftOutBuf    = static_cast<double*>(mkl_malloc(l.fftSize   * sizeof(double), 64));
        l.prevInputBuf = static_cast<double*>(mkl_malloc(l.partSize  * sizeof(double), 64));
        l.accumBuf     = static_cast<double*>(mkl_malloc(l.partStride * sizeof(double), 64));
        l.inputAccBuf  = static_cast<double*>(mkl_malloc(l.partSize  * sizeof(double), 64));

        if (!l.isImmediate)
            l.tailOutputBuf = static_cast<double*>(mkl_malloc(l.partSize * sizeof(double), 64));

        if (!l.irFreqDomain || !l.fdlBuf || !l.fftTimeBuf ||
            !l.fftOutBuf || !l.prevInputBuf || !l.accumBuf || !l.inputAccBuf ||
            (!l.isImmediate && !l.tailOutputBuf))
        {
            releaseAllLayers();
            return false;
        }

        // ゼロ初期化
        juce::FloatVectorOperations::clear(l.irFreqDomain, irBufSize);
        juce::FloatVectorOperations::clear(l.fdlBuf,       fdlBufSize);
        juce::FloatVectorOperations::clear(l.fftTimeBuf,   l.fftSize);
        juce::FloatVectorOperations::clear(l.fftOutBuf,    l.fftSize);
        juce::FloatVectorOperations::clear(l.prevInputBuf, l.partSize);
        juce::FloatVectorOperations::clear(l.accumBuf,     l.partStride);
        juce::FloatVectorOperations::clear(l.inputAccBuf,  l.partSize);
        if (l.tailOutputBuf)
            juce::FloatVectorOperations::clear(l.tailOutputBuf, l.partSize);

        // ── IR プリコンピュート ──
        // tempFreq は CCS 出力用に (fftSize + 2) 分確保
        // (IPP CCS 形式: (N/2+1)*2 = N+2 doubles)
        double* tempTime = static_cast<double*>(mkl_malloc(l.fftSize          * sizeof(double), 64));
        double* tempFreq = static_cast<double*>(mkl_malloc((l.fftSize + 2)    * sizeof(double), 64));
        if (!tempTime || !tempFreq)
        {
            if (tempTime) mkl_free(tempTime);
            if (tempFreq) mkl_free(tempFreq);
            releaseAllLayers();
            return false;
        }

        const double* irSrc    = impulseForFft.get() + cfgs[li].offset;
        const int     irRemain = cfgs[li].len;

        for (int p = 0; p < l.numParts; ++p)
        {
            memset(tempTime, 0, l.fftSize * sizeof(double));

            if (p < l.numPartsIR)
            {
                const int copyStart = p * l.partSize;
                const int copyLen   = std::min(l.partSize, irRemain - copyStart);
                if (copyLen > 0)
                    memcpy(tempTime, irSrc + copyStart, copyLen * sizeof(double));
            }

            // [v2.1] Forward FFT: real → CCS
            // IPP CCS 出力: [re0,im0,re1,im1,...] ← MKL DFTI_COMPLEX_COMPLEX と同一レイアウト
            ippsFFTFwd_RToCCS_64f(tempTime, tempFreq, l.fftSpec, l.fftWorkBuf);

            // interleaved complex として irFreqDomain に格納
            memcpy(l.irFreqDomain + p * l.partStride, tempFreq,
                   l.complexSize * 2 * sizeof(double));

            if (scale != 1.0)
                cblas_dscal(l.complexSize * 2, scale, l.irFreqDomain + p * l.partStride, 1);
        }

        // Backward FFT のウォームアップ
        // Audio Thread での初回実行時の遅延 (IPP テーブル生成等) を事前消化する。
        // [v2.1] IFFT: CCS → real (IPP_FFT_DIV_INV_BY_N により 1/N 正規化済み)
        ippsFFTInv_CCSToR_64f(tempFreq, tempTime, l.fftSpec, l.fftWorkBuf);

        mkl_free(tempTime);
        mkl_free(tempFreq);

        // [最適化2] IR パーティションを逆順に並び替える (forward アクセス最適化)
        if (l.numPartsIR > 1)
        {
            double* swapBuf = static_cast<double*>(mkl_malloc(
                static_cast<size_t>(l.partStride) * sizeof(double), 64));
            if (swapBuf)
            {
                for (int pf = 0; pf < l.numPartsIR / 2; ++pf)
                {
                    const int pb = l.numPartsIR - 1 - pf;
                    double* slotF = l.irFreqDomain + pf * l.partStride;
                    double* slotB = l.irFreqDomain + pb * l.partStride;
                    memcpy(swapBuf, slotF, l.partStride * sizeof(double));
                    memcpy(slotF,   slotB, l.partStride * sizeof(double));
                    memcpy(slotB,   swapBuf, l.partStride * sizeof(double));
                }
                mkl_free(swapBuf);
            }
        }

        // ── 非 Immediate レイヤーのコールバックあたりパーティション数 ──
        if (!l.isImmediate)
        {
            const int blocksPerPart = (l.partSize + std::max(blockSize, 1) - 1) / std::max(blockSize, 1);
            l.partsPerCallback = std::max(1,
                (l.numPartsIR + blocksPerPart - 1) / blocksPerPart);
            l.partsPerCallback = std::min(l.partsPerCallback, l.numPartsIR);
        }

        l.fdlIndex       = 0;
        l.inputPos       = 0;
        l.nextPart       = 0;
        l.tailOutputPos  = 0;
        l.baseFdlIdxSaved = 0;
        l.distributing   = false;

        ++m_numActiveLayers;
    }

    if (m_numActiveLayers == 0)
        return false;

    // ────────────────────────────────────────────────
    // 出力リングバッファ確保 (L0 専用)
    // ────────────────────────────────────────────────
    const int l0PartSize  = m_layers[0].partSize;
    const int numPartsIR  = (irLen + blockSize - 1) / blockSize;
    const int numParts    = juce::nextPowerOfTwo(numPartsIR);
    const int baseSize    = numParts * 2;
    const int margin      = juce::nextPowerOfTwo(blockSize);
    const int rSize       = juce::nextPowerOfTwo(baseSize + margin);
    const int minSize     = juce::nextPowerOfTwo(l0PartSize * 4 + blockSize * 4);
    const int finalSize   = std::max(rSize, minSize);
    m_ringBuf = static_cast<double*>(mkl_malloc(finalSize * sizeof(double), 64));
    if (!m_ringBuf)
    {
        releaseAllLayers();
        return false;
    }

    memset(m_ringBuf, 0, finalSize * sizeof(double));
    m_ringSize  = finalSize;
    m_ringMask  = finalSize - 1;
    m_ringWrite = 0;
    m_ringRead  = 0;
    m_ringAvail = 0;

    m_latency = m_layers[0].partSize;

    if (filterSpec != nullptr)
        applySpectrumFilter(*filterSpec);

    m_ready.store(true, std::memory_order_release);
    return true;
}

bool MKLNonUniformConvolver::areFftDescriptorsCommitted() const noexcept
{
    if (!m_ready.load(std::memory_order_acquire) || m_numActiveLayers <= 0)
        return false;

    for (int li = 0; li < m_numActiveLayers; ++li)
    {
        const Layer& l = m_layers[li];
        // [v2.1] fftHandle → fftSpec
        if (l.fftSpec == nullptr || !l.descriptorCommitted)
            return false;
    }

    return true;
}

//==============================================================================
// processDirectBlock  ─ Audio Thread
//==============================================================================
void MKLNonUniformConvolver::processDirectBlock(const double* input, int numSamples) noexcept
{
    if (!m_directEnabled || numSamples <= 0 || m_directOutBuf == nullptr || m_directWindow == nullptr || m_directIRRev == nullptr)
        return;

    if (numSamples > m_directMaxBlock)
    {
        jassertfalse;
        m_directPendingSamples = 0;
        return;
    }

    constexpr double kDenormalThreshold = convo::numeric_policy::kDenormThresholdAudioState;
    juce::FloatVectorOperations::clear(m_directOutBuf, numSamples);

    int processed = 0;
    while (processed < numSamples)
    {
        const int chunk = std::min(numSamples - processed, m_directMaxBlock);

        if (m_directHistLen > 0)
            juce::FloatVectorOperations::copy(m_directWindow, m_directHistory, m_directHistLen);

        if (input)
            juce::FloatVectorOperations::copy(m_directWindow + m_directHistLen, input + processed, chunk);
        else
            memset(m_directWindow + m_directHistLen, 0, static_cast<size_t>(chunk) * sizeof(double));

        for (int n = 0; n < chunk; ++n)
        {
            const double* x = m_directWindow + n;
            __m256d sum0 = _mm256_setzero_pd();
            __m256d sum1 = _mm256_setzero_pd();
            int k = 0;
            const int vEnd8 = (m_directTapCount / 8) * 8;

            for (; k < vEnd8; k += 8)
            {
                const __m256d h0 = _mm256_load_pd(m_directIRRev + k);
                const __m256d x0 = _mm256_loadu_pd(x + k);
                const __m256d h1 = _mm256_load_pd(m_directIRRev + k + 4);
                const __m256d x1 = _mm256_loadu_pd(x + k + 4);
                sum0 = _mm256_fmadd_pd(h0, x0, sum0);
                sum1 = _mm256_fmadd_pd(h1, x1, sum1);
            }

            double y = hsum256_pd(_mm256_add_pd(sum0, sum1));
            for (; k < m_directTapCount; ++k)
                y += m_directIRRev[k] * x[k];

            if (!isFiniteAndAboveThresholdMask(y, kDenormalThreshold))
                y = 0.0;

            m_directOutBuf[processed + n] = y;
        }

        if (m_directHistLen > 0)
            juce::FloatVectorOperations::copy(m_directHistory, m_directWindow + chunk, m_directHistLen);

        processed += chunk;
    }

    m_directPendingSamples = numSamples;
}

//==============================================================================
// processLayerBlock  ─ Audio Thread (L0 専用)
//
// 処理:
//   1. Overlap-Save 形式で fftTimeBuf を組み立てる [prevInput | currentInput]
//   2. Forward FFT → FDL の現在スロットへ格納
//   3. FDL × IR の複素乗算積算 (AVX2 FMA)
//   4. Backward FFT
//   5. 有効出力 (後半 partSize サンプル) をリングバッファへ書き込み (ringWrite)
//   6. FDL インデックスを進める
//==============================================================================
void MKLNonUniformConvolver::processLayerBlock(Layer& l) noexcept
{
    // B7: Audio Thread 内での初回 FFT 実行による遅延を防止（Message Thread で warmup 済みを保証）
    MKLRealTime::warmupLayer(l);

    // ── 1. [prevInput | currentInput] を fftTimeBuf に配置 (Overlap-Save) ──
    juce::FloatVectorOperations::copy(l.fftTimeBuf,              l.prevInputBuf, l.partSize);
    juce::FloatVectorOperations::copy(l.fftTimeBuf + l.partSize, l.inputAccBuf,  l.partSize);
    juce::FloatVectorOperations::copy(l.prevInputBuf, l.inputAccBuf, l.partSize);

    // ── 2. Forward FFT ──
    // [v2.1] ippsFFTFwd_RToCCS_64f: real → CCS interleaved complex
    // CCS 出力形式: [re0,im0,re1,im1,...] ← 既存 AVX2 複素乗算と完全互換
    // partStride >= complexSize*2 = fftSize+2 を確保済み (SetImpulse で検証)
    double* currentFDLSlot = l.fdlBuf + l.fdlIndex * l.partStride;
    ippsFFTFwd_RToCCS_64f(l.fftTimeBuf, currentFDLSlot, l.fftSpec, l.fftWorkBuf);

    // [最適化2] Linearized ring buffer: mirror write
    double* mirrorFDLSlot = l.fdlBuf + (l.fdlIndex + l.numParts) * l.partStride;
    memcpy(mirrorFDLSlot, currentFDLSlot, l.partStride * sizeof(double));

    // ── 3. 複素乗算積算 (FDL × IR) → accumBuf ──
    memset(l.accumBuf, 0, l.partStride * sizeof(double));

    const double* fdlBase = l.fdlBuf;
    const double* irBase  = l.irFreqDomain;
    double*       dst     = l.accumBuf;

    const int linStart   = l.fdlIndex - l.numPartsIR + 1 + l.numParts;
    const double* fdlLin = fdlBase + linStart * l.partStride;

    for (int p = 0; p < l.numPartsIR; ++p)
    {
        const double* srcA = fdlLin + p * l.partStride;
        const double* srcB = irBase + p * l.partStride;

        if (p + 1 < l.numPartsIR)
        {
            _mm_prefetch((const char*)(srcA + l.partStride),     _MM_HINT_T1);
            _mm_prefetch((const char*)(srcB + l.partStride),     _MM_HINT_T1);
        }
        if (p + 2 < l.numPartsIR)
        {
            _mm_prefetch((const char*)(srcA + 2 * l.partStride), _MM_HINT_T1);
            _mm_prefetch((const char*)(srcB + 2 * l.partStride), _MM_HINT_T1);
        }

        int k = 0;
        const int vEnd8 = (l.complexSize / 8) * 8;
        const int vEnd4 = (l.complexSize / 4) * 4;

        for (; k < vEnd8; k += 8)
        {
            _mm_prefetch((const char*)(srcA + 2 * k + 64), _MM_HINT_T0);
            _mm_prefetch((const char*)(srcB + 2 * k + 64), _MM_HINT_T0);

            __m256d acc0 = _mm256_load_pd(dst  + 2 * k);
            __m256d acc1 = _mm256_load_pd(dst  + 2 * k + 4);
            __m256d a0   = _mm256_load_pd(srcA + 2 * k);
            __m256d a1   = _mm256_load_pd(srcA + 2 * k + 4);
            __m256d b0   = _mm256_load_pd(srcB + 2 * k);
            __m256d b1   = _mm256_load_pd(srcB + 2 * k + 4);

            __m256d a0_re = _mm256_movedup_pd(a0);
            __m256d a0_im = _mm256_permute_pd(a0, 0xF);
            acc0 = _mm256_fmadd_pd(a0_re, b0, acc0);
            __m256d b0_sw = _mm256_permute_pd(b0, 0x5);
            acc0 = _mm256_addsub_pd(acc0, _mm256_mul_pd(a0_im, b0_sw));

            __m256d a1_re = _mm256_movedup_pd(a1);
            __m256d a1_im = _mm256_permute_pd(a1, 0xF);
            acc1 = _mm256_fmadd_pd(a1_re, b1, acc1);
            __m256d b1_sw = _mm256_permute_pd(b1, 0x5);
            acc1 = _mm256_addsub_pd(acc1, _mm256_mul_pd(a1_im, b1_sw));

            _mm256_store_pd(dst + 2 * k,     acc0);
            _mm256_store_pd(dst + 2 * k + 4, acc1);

            __m256d acc2 = _mm256_load_pd(dst  + 2 * k + 8);
            __m256d acc3 = _mm256_load_pd(dst  + 2 * k + 12);
            __m256d a2   = _mm256_load_pd(srcA + 2 * k + 8);
            __m256d a3   = _mm256_load_pd(srcA + 2 * k + 12);
            __m256d b2   = _mm256_load_pd(srcB + 2 * k + 8);
            __m256d b3   = _mm256_load_pd(srcB + 2 * k + 12);

            __m256d a2_re = _mm256_movedup_pd(a2);
            __m256d a2_im = _mm256_permute_pd(a2, 0xF);
            acc2 = _mm256_fmadd_pd(a2_re, b2, acc2);
            __m256d b2_sw = _mm256_permute_pd(b2, 0x5);
            acc2 = _mm256_addsub_pd(acc2, _mm256_mul_pd(a2_im, b2_sw));

            __m256d a3_re = _mm256_movedup_pd(a3);
            __m256d a3_im = _mm256_permute_pd(a3, 0xF);
            acc3 = _mm256_fmadd_pd(a3_re, b3, acc3);
            __m256d b3_sw = _mm256_permute_pd(b3, 0x5);
            acc3 = _mm256_addsub_pd(acc3, _mm256_mul_pd(a3_im, b3_sw));

            _mm256_store_pd(dst + 2 * k + 8,  acc2);
            _mm256_store_pd(dst + 2 * k + 12, acc3);
        }

        for (; k < vEnd4; k += 4)
        {
            __m256d acc0 = _mm256_load_pd(dst  + 2 * k);
            __m256d acc1 = _mm256_load_pd(dst  + 2 * k + 4);
            __m256d a0   = _mm256_load_pd(srcA + 2 * k);
            __m256d a1   = _mm256_load_pd(srcA + 2 * k + 4);
            __m256d b0   = _mm256_load_pd(srcB + 2 * k);
            __m256d b1   = _mm256_load_pd(srcB + 2 * k + 4);

            __m256d a0_re = _mm256_movedup_pd(a0);
            __m256d a0_im = _mm256_permute_pd(a0, 0xF);
            acc0 = _mm256_fmadd_pd(a0_re, b0, acc0);
            __m256d b0_sw = _mm256_permute_pd(b0, 0x5);
            acc0 = _mm256_addsub_pd(acc0, _mm256_mul_pd(a0_im, b0_sw));

            __m256d a1_re = _mm256_movedup_pd(a1);
            __m256d a1_im = _mm256_permute_pd(a1, 0xF);
            acc1 = _mm256_fmadd_pd(a1_re, b1, acc1);
            __m256d b1_sw = _mm256_permute_pd(b1, 0x5);
            acc1 = _mm256_addsub_pd(acc1, _mm256_mul_pd(a1_im, b1_sw));

            _mm256_store_pd(dst + 2 * k,     acc0);
            _mm256_store_pd(dst + 2 * k + 4, acc1);
        }

        for (; k < l.complexSize; ++k)
        {
            const double ar = srcA[2 * k],     ai = srcA[2 * k + 1];
            const double br = srcB[2 * k],     bi = srcB[2 * k + 1];
            dst[2 * k]     += ar * br - ai * bi;
            dst[2 * k + 1] += ar * bi + ai * br;
        }
    }

    // ── 4. Backward FFT ──
    // IFFT 前にデノーマル対策 (accumBuf の複素データに適用)
#if defined(__AVX2__)
    for (int k = 0; k < l.partStride; k += 4) {
        __m256d v = _mm256_load_pd(&l.accumBuf[k]);
        v = killDenormalV(v);
        _mm256_store_pd(&l.accumBuf[k], v);
    }
#else
    for (int k = 0; k < l.partStride; ++k)
        l.accumBuf[k] = killDenormal(l.accumBuf[k]);
#endif
    // [v2.1] ippsFFTInv_CCSToR_64f: CCS → real
    // IPP_FFT_DIV_INV_BY_N により 1/N 正規化自動適用 (旧 DFTI_BACKWARD_SCALE と等価)
    ippsFFTInv_CCSToR_64f(l.accumBuf, l.fftOutBuf, l.fftSpec, l.fftWorkBuf);

    // ── 5. Overlap-Save: 有効出力をリングへ書き込み ──
    ringWrite(l.fftOutBuf + l.partSize, l.partSize);

    // ── 6. FDL インデックスを進める ──
    l.fdlIndex = (l.fdlIndex + 1) & l.fdlMask;
}

//==============================================================================
// ringWrite  ─ Audio Thread (L0 専用)
//==============================================================================
void MKLNonUniformConvolver::ringWrite(const double* src, int n) noexcept
{
    if (n <= 0 || m_ringBuf == nullptr || src == nullptr) return;

    const int first = std::min(n, m_ringSize - m_ringWrite);
    juce::FloatVectorOperations::copy(m_ringBuf + m_ringWrite, src, first);
    if (n > first)
        juce::FloatVectorOperations::copy(m_ringBuf, src + first, n - first);

    m_ringWrite = (m_ringWrite + n) & m_ringMask;

    const int nextAvail = m_ringAvail + n;
    if (nextAvail > m_ringSize)
    {
        const int overflow = nextAvail - m_ringSize;
        m_ringRead = (m_ringRead + overflow) & m_ringMask;
        m_ringAvail = m_ringSize;
        m_ringWrite = (m_ringWrite + overflow) & m_ringMask;
        m_ringOverflowCount.fetch_add(1, std::memory_order_relaxed);
        if (overflowCallback)
            overflowCallback(overflowUserData);
    }
    else
    {
        m_ringAvail = nextAvail;
    }
}

//==============================================================================
// ringRead  ─ Audio Thread (L0 専用)
//==============================================================================
int MKLNonUniformConvolver::ringRead(double* dst, int n) noexcept
{
    if (n <= 0 || m_ringBuf == nullptr) return 0;

    const int toRead = std::min(n, m_ringAvail);
    if (toRead == 0)
    {
        if (dst) memset(dst, 0, n * sizeof(double));
        return 0;
    }

    const int first = std::min(toRead, m_ringSize - m_ringRead);

    if (dst)
    {
        juce::FloatVectorOperations::copy(dst, m_ringBuf + m_ringRead, first);
        if (toRead > first)
            juce::FloatVectorOperations::copy(dst + first, m_ringBuf, toRead - first);

        if (toRead < n)
            memset(dst + toRead, 0, (n - toRead) * sizeof(double));
    }

    m_ringRead  = (m_ringRead + toRead) & m_ringMask;
    m_ringAvail -= toRead;
    return toRead;
}

//==============================================================================
// Add  ─ Audio Thread
//==============================================================================
void MKLNonUniformConvolver::Add(const double* input, int numSamples)
{
    if (!m_ready.load(std::memory_order_acquire) || numSamples <= 0)
        return;

    processDirectBlock(input, numSamples);

    for (int li = 0; li < m_numActiveLayers; ++li)
    {
        Layer& l = m_layers[li];

        // B7: Audio Thread 内での初回 FFT 実行による遅延を防止（Message Thread で warmup 済みを保証）
        MKLRealTime::warmupLayer(l);

        int consumed = 0;
        while (consumed < numSamples)
        {
            const int toFill = std::min(numSamples - consumed, l.partSize - l.inputPos);
            if (input)
                memcpy(l.inputAccBuf + l.inputPos, input + consumed, toFill * sizeof(double));
            else
                memset(l.inputAccBuf + l.inputPos, 0, toFill * sizeof(double));
            l.inputPos += toFill;
            consumed   += toFill;

            if (l.inputPos >= l.partSize)
            {
                l.inputPos = 0;

                if (l.isImmediate)
                {
                    processLayerBlock(l);
                }
                else
                {
                    jassert(consumed <= numSamples);

                    juce::FloatVectorOperations::copy(l.fftTimeBuf,              l.prevInputBuf, l.partSize);
                    juce::FloatVectorOperations::copy(l.fftTimeBuf + l.partSize, l.inputAccBuf,  l.partSize);
                    juce::FloatVectorOperations::copy(l.prevInputBuf, l.inputAccBuf, l.partSize);

                    // [v2.1] L1/L2 Forward FFT: real → CCS
                    double* currentFDLSlot = l.fdlBuf + l.fdlIndex * l.partStride;
                    ippsFFTFwd_RToCCS_64f(l.fftTimeBuf, currentFDLSlot, l.fftSpec, l.fftWorkBuf);

                    // [最適化2] mirror write
                    double* mirrorFDLSlot = l.fdlBuf + (l.fdlIndex + l.numParts) * l.partStride;
                    juce::FloatVectorOperations::copy(mirrorFDLSlot, currentFDLSlot, l.partStride);

                    l.fdlIndex = (l.fdlIndex + 1) & l.fdlMask;

                    // [Bug2 fix] FDL スナップショット保存
                    l.baseFdlIdxSaved = (l.fdlIndex - 1 + l.numParts) & l.fdlMask;

                    memset(l.accumBuf, 0, l.partStride * sizeof(double));
                    l.nextPart    = 0;
                    l.distributing = true;
                }
            }
        } // while consumed < numSamples

        // ────────────────────────────────────────────────────────────────────
        // [Bug3 fix] 分散計算ループ: 毎コールバック実行
        // ────────────────────────────────────────────────────────────────────
        if (!l.isImmediate && l.distributing)
        {
            const int endPart  = std::min(l.nextPart + l.partsPerCallback, l.numPartsIR);

            const double* fdlBase    = l.fdlBuf;
            const double* irBase     = l.irFreqDomain;
            double*       dst        = l.accumBuf;
            const int     baseFdlIdx = l.baseFdlIdxSaved;

            const int linStart   = baseFdlIdx - l.numPartsIR + 1 + l.numParts;
            const double* fdlLin = fdlBase + linStart * l.partStride;

            for (int p = l.nextPart; p < endPart; ++p)
            {
                const double* srcA = fdlLin + p * l.partStride;
                const double* srcB = irBase + p * l.partStride;

                if (p + 1 < endPart)
                {
                    _mm_prefetch((const char*)(srcA + l.partStride), _MM_HINT_T1);
                    _mm_prefetch((const char*)(srcB + l.partStride), _MM_HINT_T1);
                }
                if (p + 2 < endPart)
                {
                    _mm_prefetch((const char*)(srcA + 2 * l.partStride), _MM_HINT_T1);
                    _mm_prefetch((const char*)(srcB + 2 * l.partStride), _MM_HINT_T1);
                }

                int k = 0;
                const int vEnd8 = (l.complexSize / 8) * 8;
                const int vEnd4 = (l.complexSize / 4) * 4;

                for (; k < vEnd8; k += 8)
                {
                    _mm_prefetch((const char*)(srcA + 2 * k + 128), _MM_HINT_T0);
                    _mm_prefetch((const char*)(srcB + 2 * k + 128), _MM_HINT_T0);

                    __m256d acc0 = _mm256_load_pd(dst  + 2 * k);
                    __m256d acc1 = _mm256_load_pd(dst  + 2 * k + 4);
                    __m256d a0   = _mm256_load_pd(srcA + 2 * k);
                    __m256d a1   = _mm256_load_pd(srcA + 2 * k + 4);
                    __m256d b0   = _mm256_load_pd(srcB + 2 * k);
                    __m256d b1   = _mm256_load_pd(srcB + 2 * k + 4);

                    __m256d a0_re = _mm256_movedup_pd(a0);
                    __m256d a0_im = _mm256_permute_pd(a0, 0xF);
                    acc0 = _mm256_fmadd_pd(a0_re, b0, acc0);
                    __m256d b0_sw = _mm256_permute_pd(b0, 0x5);
                    acc0 = _mm256_addsub_pd(acc0, _mm256_mul_pd(a0_im, b0_sw));

                    __m256d a1_re = _mm256_movedup_pd(a1);
                    __m256d a1_im = _mm256_permute_pd(a1, 0xF);
                    acc1 = _mm256_fmadd_pd(a1_re, b1, acc1);
                    __m256d b1_sw = _mm256_permute_pd(b1, 0x5);
                    acc1 = _mm256_addsub_pd(acc1, _mm256_mul_pd(a1_im, b1_sw));

                    _mm256_store_pd(dst + 2 * k,     acc0);
                    _mm256_store_pd(dst + 2 * k + 4, acc1);

                    __m256d acc2 = _mm256_load_pd(dst  + 2 * k + 8);
                    __m256d acc3 = _mm256_load_pd(dst  + 2 * k + 12);
                    __m256d a2   = _mm256_load_pd(srcA + 2 * k + 8);
                    __m256d a3   = _mm256_load_pd(srcA + 2 * k + 12);
                    __m256d b2   = _mm256_load_pd(srcB + 2 * k + 8);
                    __m256d b3   = _mm256_load_pd(srcB + 2 * k + 12);

                    __m256d a2_re = _mm256_movedup_pd(a2);
                    __m256d a2_im = _mm256_permute_pd(a2, 0xF);
                    acc2 = _mm256_fmadd_pd(a2_re, b2, acc2);
                    __m256d b2_sw = _mm256_permute_pd(b2, 0x5);
                    acc2 = _mm256_addsub_pd(acc2, _mm256_mul_pd(a2_im, b2_sw));

                    __m256d a3_re = _mm256_movedup_pd(a3);
                    __m256d a3_im = _mm256_permute_pd(a3, 0xF);
                    acc3 = _mm256_fmadd_pd(a3_re, b3, acc3);
                    __m256d b3_sw = _mm256_permute_pd(b3, 0x5);
                    acc3 = _mm256_addsub_pd(acc3, _mm256_mul_pd(a3_im, b3_sw));

                    _mm256_store_pd(dst + 2 * k + 8,  acc2);
                    _mm256_store_pd(dst + 2 * k + 12, acc3);
                }

                for (; k < vEnd4; k += 4)
                {
                    __m256d acc0 = _mm256_load_pd(dst  + 2 * k);
                    __m256d acc1 = _mm256_load_pd(dst  + 2 * k + 4);
                    __m256d a0   = _mm256_load_pd(srcA + 2 * k);
                    __m256d a1   = _mm256_load_pd(srcA + 2 * k + 4);
                    __m256d b0   = _mm256_load_pd(srcB + 2 * k);
                    __m256d b1   = _mm256_load_pd(srcB + 2 * k + 4);

                    __m256d a0_re = _mm256_movedup_pd(a0);
                    __m256d a0_im = _mm256_permute_pd(a0, 0xF);
                    acc0 = _mm256_fmadd_pd(a0_re, b0, acc0);
                    __m256d b0_sw = _mm256_permute_pd(b0, 0x5);
                    acc0 = _mm256_addsub_pd(acc0, _mm256_mul_pd(a0_im, b0_sw));

                    __m256d a1_re = _mm256_movedup_pd(a1);
                    __m256d a1_im = _mm256_permute_pd(a1, 0xF);
                    acc1 = _mm256_fmadd_pd(a1_re, b1, acc1);
                    __m256d b1_sw = _mm256_permute_pd(b1, 0x5);
                    acc1 = _mm256_addsub_pd(acc1, _mm256_mul_pd(a1_im, b1_sw));

                    _mm256_store_pd(dst + 2 * k,     acc0);
                    _mm256_store_pd(dst + 2 * k + 4, acc1);
                }

                for (; k < l.complexSize; ++k)
                {
                    const double ar = srcA[2 * k],     ai = srcA[2 * k + 1];
                    const double br = srcB[2 * k],     bi = srcB[2 * k + 1];
                    dst[2 * k]     += ar * br - ai * bi;
                    dst[2 * k + 1] += ar * bi + ai * br;
                }
            }

            l.nextPart = endPart;

            // ── 全パーティション累積完了 → IFFT → tailOutputBuf へコピー ──
            if (l.nextPart >= l.numPartsIR)
            {
                // [v2.1] Backward FFT: CCS → real (Audio Thread 内で再初期化禁止の制約はIPPも同様)
                ippsFFTInv_CCSToR_64f(l.accumBuf, l.fftOutBuf, l.fftSpec, l.fftWorkBuf);

                memcpy(l.tailOutputBuf, l.fftOutBuf + l.partSize, l.partSize * sizeof(double));
                l.tailOutputPos = 0;

                l.distributing = false;
                l.nextPart     = 0;
            }
        }

    } // for each layer
}

//==============================================================================
// Get  ─ Audio Thread
//==============================================================================
int MKLNonUniformConvolver::Get(double* output, int numSamples)
{
    if (!m_ready.load(std::memory_order_acquire) || numSamples <= 0)
    {
        if (output && numSamples > 0)
            memset(output, 0, numSamples * sizeof(double));
        return 0;
    }

    const int got = ringRead(output, numSamples);

    auto isAligned64 = [](const void* ptr) noexcept
    {
        return (reinterpret_cast<std::uintptr_t>(ptr) & static_cast<std::uintptr_t>(63)) == 0;
    };

    auto addFallback = [](int n, double* dst, const double* src) noexcept
    {
#if defined(__AVX2__)
        int i = 0;
        const int vEnd = (n / 4) * 4;
        const bool aligned = ((reinterpret_cast<std::uintptr_t>(dst) & static_cast<std::uintptr_t>(31)) == 0)
            && ((reinterpret_cast<std::uintptr_t>(src) & static_cast<std::uintptr_t>(31)) == 0);

        for (; i < vEnd; i += 4)
        {
            const __m256d a = aligned ? _mm256_load_pd(dst + i) : _mm256_loadu_pd(dst + i);
            const __m256d b = aligned ? _mm256_load_pd(src + i) : _mm256_loadu_pd(src + i);
            if (aligned)
                _mm256_store_pd(dst + i, _mm256_add_pd(a, b));
            else
                _mm256_storeu_pd(dst + i, _mm256_add_pd(a, b));
        }
        for (; i < n; ++i)
            dst[i] += src[i];
#else
        for (int i = 0; i < n; ++i)
            dst[i] += src[i];
#endif
    };

    // suppress unused warning
    (void)isAligned64;

    // ── Direct 出力 ──
    if (m_directEnabled && m_directOutBuf != nullptr)
    {
        const int toAdd = std::min(numSamples, m_directPendingSamples);
        if (toAdd > 0)
        {
            if (output != nullptr)
                addFallback(toAdd, output, m_directOutBuf);

            memset(m_directOutBuf, 0, static_cast<size_t>(toAdd) * sizeof(double));
            m_directPendingSamples = 0;
        }
    }

    // ── L1/L2 出力 ──
    for (int li = 1; li < m_numActiveLayers; ++li)
    {
        Layer& l = m_layers[li];
        if (l.tailOutputBuf == nullptr) continue;

        const int remaining = l.partSize - l.tailOutputPos;
        if (remaining <= 0) continue;

        const int toAdd = std::min(numSamples, remaining);

        if (output != nullptr)
        {
            const double* tailPtr = l.tailOutputBuf + l.tailOutputPos;
            addFallback(toAdd, output, tailPtr);
        }

        l.tailOutputPos += toAdd;
    }

    return got;
}

//==============================================================================
// Reset  ─ Message Thread
//==============================================================================
void MKLNonUniformConvolver::Reset()
{
    for (int li = 0; li < m_numActiveLayers; ++li)
    {
        Layer& l = m_layers[li];
        if (l.irFreqDomain == nullptr) continue;

        const size_t fdlBufSize = static_cast<size_t>(l.numParts) * 2 * l.partStride;
        juce::FloatVectorOperations::clear(l.fdlBuf,       fdlBufSize);
        juce::FloatVectorOperations::clear(l.fftTimeBuf,   l.fftSize);
        juce::FloatVectorOperations::clear(l.fftOutBuf,    l.fftSize);
        juce::FloatVectorOperations::clear(l.prevInputBuf, l.partSize);
        juce::FloatVectorOperations::clear(l.accumBuf,     l.partStride);
        juce::FloatVectorOperations::clear(l.inputAccBuf,  l.partSize);

        if (l.tailOutputBuf)
            juce::FloatVectorOperations::clear(l.tailOutputBuf, l.partSize);

        l.fdlIndex        = 0;
        l.inputPos        = 0;
        l.nextPart        = 0;
        l.tailOutputPos   = 0;
        l.baseFdlIdxSaved = 0;
        l.distributing    = false;
    }

    if (m_ringBuf)
        juce::FloatVectorOperations::clear(m_ringBuf, m_ringSize);
    m_ringWrite = 0;
    m_ringRead  = 0;
    m_ringAvail = 0;

    if (m_directHistLen > 0 && m_directHistory)
        memset(m_directHistory, 0, static_cast<size_t>(m_directHistLen) * sizeof(double));
    if (m_directOutBuf && m_directMaxBlock > 0)
        memset(m_directOutBuf, 0, static_cast<size_t>(m_directMaxBlock) * sizeof(double));
    m_directPendingSamples = 0;
}

} // namespace convo
