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
// D2: /fp:fast の影響を回避するため、DSP コアファイルで float_control を precise に指定
#if defined(_MSC_VER)
#pragma float_control(precise, on)
#endif

#include <JuceHeader.h>
#include "MKLNonUniformConvolver.h"
#include "DiagnosticsConfig.h"  // ★ work70: DIAG_MKL_MALLOC, convo::diag, getProcessMemoryInfo

#include "AlignedAllocation.h"
#include "DspNumericPolicy.h"
#include "AtomicAccess.h"  // convo::consumeAtomic

// absNoLibm — 標準ライブラリ abs を経由せずビット操作で |x| を求める (RT-safe)
[[nodiscard]] constexpr inline double absNoLibm(double x) noexcept
{
    auto bits = std::bit_cast<std::uint64_t>(x);
    bits &= 0x7FFFFFFFFFFFFFFFULL;
    return std::bit_cast<double>(bits);
}
#include <mkl.h>        // mkl_malloc, mkl_free, mkl_set_num_threads
#include <mkl_vml.h>    // vdMul
#include <mkl_cblas.h>  // cblas_dscal
#include <ipp.h>       // ippsFFTFwd_RToCCS_64f, ippsFFTInv_CCSToR_64f (MKL DFTI 代替)
#include <cstring>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

#pragma comment(lib, "psapi.lib")  // ★ work70: getProcessMemoryInfo

#include <immintrin.h>  // AVX2

#include "audioengine/AtomicAccess.h"

namespace convo
{

#if JUCE_DEBUG
std::atomic<int> MKLNonUniformConvolver::debugWarmupGuardCountStorage_ { 0 };

std::atomic<int>& MKLNonUniformConvolver::debugWarmupGuardCount() noexcept
{
    return debugWarmupGuardCountStorage_;
}
#endif

// ★ work70: 診断用静的変数
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
std::atomic<uint32_t> MKLNonUniformConvolver::liveCount { 0 };
std::atomic<uint64_t> MKLNonUniformConvolver::globalDiagSeq { 0 };
#endif

struct IppFFTPlan
{
    int order = 0;
    int fftSize = 0;
    int sizeWork = 0;
    IppsFFTSpec_R_64f* fftSpec = nullptr;
    Ipp8u* fftSpecBuf = nullptr;

    ~IppFFTPlan()
    {
        if (fftSpecBuf)
            ippsFree(fftSpecBuf);
    }
};

class IppFFTPlanCache
{
public:
    static const IppFFTPlan* getOrCreate(int order)
    {
        ASSERT_NON_RT_THREAD();
        std::lock_guard<std::mutex> lock(getMutex());
        auto& cache = getCache();
        const auto it = cache.find(order);
        if (it != cache.end())
            return it->second.get();

        auto plan = createPlan(order);
        if (!plan)
            return nullptr;

        auto* ptr = plan.get();
        cache.emplace(order, std::move(plan));
        return ptr;
    }

private:
    static std::unordered_map<int, std::unique_ptr<IppFFTPlan>> cacheStorage_;
    static std::mutex cacheMutex_;

    static std::unordered_map<int, std::unique_ptr<IppFFTPlan>>& getCache()
    {
        return cacheStorage_;
    }

    static std::mutex& getMutex()
    {
        return cacheMutex_;
    }

    static std::unique_ptr<IppFFTPlan> createPlan(int order)
    {
        int sizeSpec = 0, sizeInit = 0, sizeWork = 0;
        const IppStatus getSt = ippsFFTGetSize_R_64f(
            order, IPP_FFT_DIV_INV_BY_N, ippAlgHintFast,
            &sizeSpec, &sizeInit, &sizeWork);
        if (getSt != ippStsNoErr)
            return nullptr;

        std::unique_ptr<IppFFTPlan> plan = std::make_unique<IppFFTPlan>();
        plan->order = order;
        plan->fftSize = 1 << order;
        plan->sizeWork = sizeWork;

        plan->fftSpecBuf = ippsMalloc_8u(sizeSpec);
        if (!plan->fftSpecBuf)
            return nullptr;

        Ipp8u* initBuf = (sizeInit > 0) ? ippsMalloc_8u(sizeInit) : nullptr;
        const IppStatus initSt = ippsFFTInit_R_64f(
            &plan->fftSpec, order, IPP_FFT_DIV_INV_BY_N, ippAlgHintFast,
            plan->fftSpecBuf, initBuf);

        if (initBuf)
            ippsFree(initBuf);

        if (initSt != ippStsNoErr || plan->fftSpec == nullptr)
            return nullptr;

        return plan;
    }
};

std::unordered_map<int, std::unique_ptr<IppFFTPlan>> IppFFTPlanCache::cacheStorage_{};
std::mutex IppFFTPlanCache::cacheMutex_{};

namespace
{
// [Mem-Fix] 本プロジェクトは AVX2 必須環境 (x64 / Intel or AMD64, AVX2 保証) のため、
// split-complex (SoA) カーネルを唯一の実装として一本化する。
// AoS 側の非split-complexフォールバック分岐は撤去済み。
#ifndef __AVX2__
#error "MKLNonUniformConvolver requires AVX2 (see coding standard: CPU must support AVX2)."
#endif

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

inline void deinterleaveComplex(const double* srcInterleaved, double* dstReal, double* dstImag, int complexSize) noexcept
{
    for (int k = 0; k < complexSize; ++k)
    {
        dstReal[k] = srcInterleaved[2 * k];
        dstImag[k] = srcInterleaved[2 * k + 1];
    }
}

inline void interleaveComplex(const double* srcReal, const double* srcImag, double* dstInterleaved, int complexSize) noexcept
{
    for (int k = 0; k < complexSize; ++k)
    {
        dstInterleaved[2 * k] = srcReal[k];
        dstInterleaved[2 * k + 1] = srcImag[k];
    }
}

inline void accumulateSplitComplex(const double* srcAReal,
                                   const double* srcAImag,
                                   const double* srcBReal,
                                   const double* srcBImag,
                                   double* dstReal,
                                   double* dstImag,
                                   int complexSize) noexcept
{
#if defined(__AVX2__)
    int k = 0;
    const int vEnd = (complexSize / 4) * 4;
    for (; k < vEnd; k += 4)
    {
        // ★ icx アライメント対策: complexSize が奇数の場合、SoA バッファの行オフセットが
        //    32バイト境界に乗らないケースがある。全ポインタを unaligned load で安全に読む。
        //    dstReal/dstImag (accumReal/accumImag) は 64バイトアラインかつ k が 4 の倍数の
        //    ため常に 32バイトアラインだが、store も unaligned に統一して安全側に倒す。
        __m256d ar = _mm256_loadu_pd(srcAReal + k);
        __m256d ai = _mm256_loadu_pd(srcAImag + k);
        __m256d br = _mm256_loadu_pd(srcBReal + k);
        __m256d bi = _mm256_loadu_pd(srcBImag + k);

        __m256d dr = _mm256_loadu_pd(dstReal + k);
        __m256d di = _mm256_loadu_pd(dstImag + k);

        dr = _mm256_add_pd(dr, _mm256_sub_pd(_mm256_mul_pd(ar, br), _mm256_mul_pd(ai, bi)));
        di = _mm256_add_pd(di, _mm256_add_pd(_mm256_mul_pd(ar, bi), _mm256_mul_pd(ai, br)));

        _mm256_storeu_pd(dstReal + k, dr);
        _mm256_storeu_pd(dstImag + k, di);
    }

    for (; k < complexSize; ++k)
    {
        dstReal[k] += srcAReal[k] * srcBReal[k] - srcAImag[k] * srcBImag[k];
        dstImag[k] += srcAReal[k] * srcBImag[k] + srcAImag[k] * srcBReal[k];
    }
#else
    for (int k = 0; k < complexSize; ++k)
    {
        dstReal[k] += srcAReal[k] * srcBReal[k] - srcAImag[k] * srcBImag[k];
        dstImag[k] += srcAReal[k] * srcBImag[k] + srcAImag[k] * srcBReal[k];
    }
#endif
}
} // namespace

//==============================================================================
// ★ work70: 診断ログ出力（無名名前空間、juce::Logger::writeToLog のラッパー）
//==============================================================================
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
namespace {

void diagLogNonRt(const juce::String& message) noexcept
{
    juce::Logger::writeToLog(message);
}

/// IR_RELEASE ログ出力（static free function）。
/// 呼び出し元は解放前のスナップショット beforeSnap を引数で渡す。
void logIrRelease(
    const MKLNonUniformConvolver* nuc,
    uint64_t diagSeq,
    uint64_t beforeMkl,
    uint32_t beforeLost,
    const ProcessMemoryInfo& beforeOs,
    const NucDiagnosticsSnapshot& beforeSnap,
    const ProcessMemoryInfo& afterReleaseOs,
    uint32_t liveBefore) noexcept
{
    const uint64_t afterMkl = convo::diag::allocatedBytes();
    const uint32_t afterLost = convo::diag::lostFreeCount();
    const int64_t deltaMkl = static_cast<int64_t>(afterMkl) - static_cast<int64_t>(beforeMkl);
    const int32_t deltaLost = static_cast<int32_t>(afterLost) - static_cast<int32_t>(beforeLost);

    diagLogNonRt(juce::String::formatted(
        "[IR_RELEASE] NUC#%p seq=%llu "
        "MKL: before=%lluMB after=%lluMB delta=%lldMB "
        "LayersBefore=%d TotalBefore=%.0fMB(persistent) "
        "lostFree=%u(+%d) liveBefore=%u | "
        "OS: beforePrivate=%lluMB afterPrivate=%lluMB",
        (void*)nuc,
        (unsigned long long)diagSeq,
        (unsigned long long)(beforeMkl / (1024*1024)),
        (unsigned long long)(afterMkl / (1024*1024)),
        (long long)(deltaMkl / (1024*1024)),
        beforeSnap.numActiveLayers,
        beforeSnap.totalBytes() / (1024.0 * 1024.0),
        (unsigned)afterLost, (int)deltaLost,
        (unsigned)liveBefore,
        (unsigned long long)beforeOs.privateUsageMB,
        (unsigned long long)afterReleaseOs.privateUsageMB));
}

} // namespace
#endif

//==============================================================================
// Layer::freeAll
//==============================================================================
void MKLNonUniformConvolver::Layer::freeAll() noexcept
{
    // [v2.2] FFT plan はサイズ単位の共有キャッシュ管理。
    // レイヤー側は所有権のみ解放し、スペック実体はキャッシュ側で保持する。
    fftPlanOwner.reset();
    fftSpec = nullptr;
    if (fftWorkBuf)
    {
        ippsFree(fftWorkBuf);
        fftWorkBuf = nullptr;
    }
    descriptorCommitted = false;

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // ★ work70: freeTracked を使用（allocSizes からサイズ取得）
    freeTracked(irFreqDomain,  allocSizes.irFreqDomain);
    freeTracked(irFreqReal,    allocSizes.irFreqReal);
    freeTracked(irFreqImag,    allocSizes.irFreqImag);
    freeTracked(fdlBuf,        allocSizes.fdlBuf);
    freeTracked(fdlReal,       allocSizes.fdlReal);
    freeTracked(fdlImag,       allocSizes.fdlImag);
    freeTracked(fftTimeBuf,    allocSizes.fftTimeBuf);
    freeTracked(fftOutBuf,     allocSizes.fftOutBuf);
    freeTracked(prevInputBuf,  allocSizes.prevInputBuf);
    freeTracked(accumBuf,      allocSizes.accumBuf);
    freeTracked(accumReal,     allocSizes.accumReal);
    freeTracked(accumImag,     allocSizes.accumImag);
    freeTracked(inputAccBuf,   allocSizes.inputAccBuf);
    freeTracked(tailOutputBuf, allocSizes.tailOutputBuf);
    freeTracked(delayLineBuf,  allocSizes.delayLineBuf);   // ★ Bug#1 B13 delayLineBuf 追跡
    allocSizes = {};
#else
    if (irFreqDomain)  { mkl_free(irFreqDomain);  irFreqDomain  = nullptr; }
    if (irFreqReal)    { mkl_free(irFreqReal);    irFreqReal    = nullptr; }
    if (irFreqImag)    { mkl_free(irFreqImag);    irFreqImag    = nullptr; }
    if (fdlBuf)        { mkl_free(fdlBuf);         fdlBuf        = nullptr; }
    if (fdlReal)       { mkl_free(fdlReal);       fdlReal       = nullptr; }
    if (fdlImag)       { mkl_free(fdlImag);       fdlImag       = nullptr; }
    if (fftTimeBuf)    { mkl_free(fftTimeBuf);     fftTimeBuf    = nullptr; }
    if (fftOutBuf)     { mkl_free(fftOutBuf);      fftOutBuf     = nullptr; }
    if (prevInputBuf)  { mkl_free(prevInputBuf);   prevInputBuf  = nullptr; }
    if (accumBuf)      { mkl_free(accumBuf);       accumBuf      = nullptr; }
    if (accumReal)     { mkl_free(accumReal);      accumReal     = nullptr; }
    if (accumImag)     { mkl_free(accumImag);      accumImag     = nullptr; }
    if (inputAccBuf)   { mkl_free(inputAccBuf);    inputAccBuf   = nullptr; }
    if (tailOutputBuf) { mkl_free(tailOutputBuf);  tailOutputBuf = nullptr; }
    if (delayLineBuf)  { mkl_free(delayLineBuf);   delayLineBuf  = nullptr; }
#endif

    outputDelaySamples = 0;
    delayLineCapacity  = 0;
    delayWriteCursor   = 0;
    delayReadCursor    = 0;

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
    // ★ [work74 FIX-01] スレッドローカル版を使用（MKLRealTimeSetup と一貫性維持）
    mkl_set_num_threads_local(1);
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    liveCount.fetch_add(1, std::memory_order_relaxed);
#endif
}

MKLNonUniformConvolver::~MKLNonUniformConvolver()
{
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint32_t oldLive = liveCount.fetch_sub(1, std::memory_order_relaxed);
    (void)oldLive;
    jassert(oldLive > 0);
#endif
    #ifdef NUC_DEBUG_GUARDS
    checkGuards();
    #endif
    releaseAllLayers();
}

//==============================================================================
// applySpectrumFilter  ─ Message Thread のみ
//==============================================================================
void MKLNonUniformConvolver::applySpectrumFilter(const FilterSpec& spec) noexcept
{
    const double fs      = spec.sampleRate;
    const double nyquist = fs * 0.5;

    const double hcFcStart = (fs <= 48000.0) ? 18000.0 : 22000.0;
    const double hcFcEnd   = nyquist;

    const double lcFcEnd   = (spec.lcMode == LCMode::Soft) ?  6.0 :  8.0;
    const double lcFcStart = (spec.lcMode == LCMode::Soft) ? 15.0 : 18.0;

    convo::ScopedAlignedPtr<double> reusableGain;
    int reusableGainCapacity = 0;

    for (int li = 0; li < m_numActiveLayers; ++li)
    {
        Layer& l = m_layers[li];
        if (!l.irFreqReal || !l.irFreqImag) continue;

        const int N      = l.fftSize;
        const int halfN  = N / 2;
        const int cSize  = l.complexSize;

        if (reusableGainCapacity < cSize || reusableGain.get() == nullptr)
        {
            reusableGain.reset(static_cast<double*>(mkl_malloc(static_cast<size_t>(cSize) * sizeof(double), 64)));
            reusableGainCapacity = (reusableGain.get() != nullptr) ? cSize : 0;
        }
        if (!reusableGain.get())
        {
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
            juce::Logger::writeToLog("MKLNonUniformConvolver: OOM in applySpectrumFilter for layer " + juce::String(li));
#endif
            continue;
        }
        double* gain = reusableGain.get();
        std::fill_n(gain, cSize, 1.0);

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
                // ★ L-02: hcFcEnd = nyquist のため k > kEnd は発生しない。
                // hcFcEnd が nyquist 未満に変更された場合は else-if 分割を再検討すること。
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

        // [Mem-Fix] gain[] は実数値(振幅のみ)のフィルタなので、実部・虚部それぞれに
        // 同一ゲインを掛けるだけでよい。interleave/deinterleaveもAoS経由も不要。
        for (int p = 0; p < l.numParts; ++p)
        {
            double* re = l.irFreqReal + static_cast<size_t>(p) * l.complexSize;
            double* im = l.irFreqImag + static_cast<size_t>(p) * l.complexSize;
            vdMul(cSize, re, gain, re);
            vdMul(cSize, im, gain, im);
        }
    }
}

//==============================================================================
void MKLNonUniformConvolver::releaseAllLayers() noexcept
{
    #ifdef NUC_DEBUG_GUARDS
        checkGuards();
        // ... 既存の checkPtr 群（変更なし）...
        auto checkPtr = [](double *&ptr){
            if (ptr != nullptr){
                auto addr = reinterpret_cast<uintptr_t>(ptr);
                if (addr == 0xFFFFFFFFFFFFFFFFULL ||
                    (addr & 0xFFFFFFFF00000000) == 0xCDCDCDCD00000000ULL ||
                     addr < 0x10000)
            {
                __debugbreak();
            }
        }
    };
    checkPtr(m_ringBuf);
    checkPtr(m_directIRRev);
    checkPtr(m_directHistory);
    checkPtr(m_directWindow);
    checkPtr(m_directOutBuf);
    #endif

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // ★ work70: 解放前にスナップショットとサイズを取得
    const uint64_t beforeMkl = convo::diag::allocatedBytes();
    const uint32_t beforeLost = convo::diag::lostFreeCount();
    const auto beforeOs = getProcessMemoryInfo();

    // 解放前の NUC 状態を取得（TotalBefore / LayersBefore 用）
    // ★ freeAll で Layer が初期化される前に呼ぶこと
    NucDiagnosticsSnapshot beforeSnap;
    beforeSnap.numActiveLayers = m_numActiveLayers;
    beforeSnap.isReady = convo::consumeAtomic(m_ready, std::memory_order_acquire);
    for (int li = 0; li < kNumLayers; ++li) {
        const Layer& l = m_layers[li];
        uint64_t irF = 0, fdl = 0, acc = 0, tail = 0;
        irF += addIfAlive(l.irFreqDomain, l.allocSizes.irFreqDomain, "irFreqDomain");
        irF += addIfAlive(l.irFreqReal,   l.allocSizes.irFreqReal,   "irFreqReal");
        irF += addIfAlive(l.irFreqImag,   l.allocSizes.irFreqImag,   "irFreqImag");
        fdl += addIfAlive(l.fdlBuf,       l.allocSizes.fdlBuf,       "fdlBuf");
        fdl += addIfAlive(l.fdlReal,      l.allocSizes.fdlReal,      "fdlReal");
        fdl += addIfAlive(l.fdlImag,      l.allocSizes.fdlImag,      "fdlImag");
        acc += addIfAlive(l.fftTimeBuf,   l.allocSizes.fftTimeBuf,   "fftTimeBuf");
        acc += addIfAlive(l.fftOutBuf,    l.allocSizes.fftOutBuf,    "fftOutBuf");
        acc += addIfAlive(l.prevInputBuf, l.allocSizes.prevInputBuf, "prevInputBuf");
        acc += addIfAlive(l.accumBuf,     l.allocSizes.accumBuf,     "accumBuf");
        acc += addIfAlive(l.accumReal,    l.allocSizes.accumReal,    "accumReal");
        acc += addIfAlive(l.accumImag,    l.allocSizes.accumImag,    "accumImag");
        acc += addIfAlive(l.inputAccBuf,  l.allocSizes.inputAccBuf,  "inputAccBuf");
        tail += addIfAlive(l.tailOutputBuf,l.allocSizes.tailOutputBuf,"tailOutputBuf");
        beforeSnap.layerBufs[li] = irF + fdl + acc + tail;
        beforeSnap.irFreqBytes   += irF;
        beforeSnap.fdlBytes      += fdl;
        beforeSnap.accumBytes    += acc;
        beforeSnap.tailBytes     += tail;
    }
    const uint32_t liveBefore = liveCount.load(std::memory_order_relaxed);

    // NUC レベルバッファの解放サイズを事前退避（ガード後、解放前に）
    const size_t ringBufBytes    = static_cast<size_t>(m_ringSize) * sizeof(double);
    const size_t directIRBytes   = static_cast<size_t>(m_directTapCount) * sizeof(double);
    const size_t directHistBytes = static_cast<size_t>(m_directHistLen) * sizeof(double);
    const size_t directWinBytes  = static_cast<size_t>(m_directHistLen + m_directMaxBlock) * sizeof(double);
    const size_t directOutBytes  = static_cast<size_t>(m_directMaxBlock) * sizeof(double);
#endif

    for (int i = 0; i < kNumLayers; ++i)
        m_layers[i].freeAll();
    m_numActiveLayers = 0;
    m_latency         = 0;

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // ★ work70: NUC レベルバッファも freeTracked で解放
    freeTracked(m_ringBuf,       ringBufBytes);
    freeTracked(m_directIRRev,   directIRBytes);
    freeTracked(m_directHistory, directHistBytes);
    freeTracked(m_directWindow,  directWinBytes);
    freeTracked(m_directOutBuf,  directOutBytes);
#else
    if (m_ringBuf) { mkl_free(m_ringBuf); m_ringBuf = nullptr; }
    if (m_directIRRev)   { mkl_free(m_directIRRev);   m_directIRRev = nullptr; }
    if (m_directHistory) { mkl_free(m_directHistory); m_directHistory = nullptr; }
    if (m_directWindow)  { mkl_free(m_directWindow);  m_directWindow = nullptr; }
    if (m_directOutBuf)  { mkl_free(m_directOutBuf);  m_directOutBuf = nullptr; }
#endif

    m_ringSize = m_ringMask = m_ringWrite = m_ringRead = m_ringAvail = 0;

    m_directTapCount = 0;
    m_directHistLen  = 0;
    m_directMaxBlock = 0;
    m_directPendingSamples = 0;
    m_directEnabled  = false;

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const auto afterReleaseOs = getProcessMemoryInfo();
    logIrRelease(this, kDiagSeqReserved,
                 beforeMkl, beforeLost, beforeOs,
                 beforeSnap, afterReleaseOs, liveBefore);
#endif
    m_tailEnabled = true;
    m_tailStrength = 1.0;
    for (int i = 0; i < kNumLayers; ++i)
        m_tailLayerGain[i] = 1.0;
}

//==============================================================================
// ★ work70: getDiagnostics  ─ Message Thread のみ
//==============================================================================
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
NucDiagnosticsSnapshot MKLNonUniformConvolver::getDiagnostics() const noexcept
{
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());

    NucDiagnosticsSnapshot snap{};
    snap.numActiveLayers = m_numActiveLayers;
    snap.isReady = convo::consumeAtomic(m_ready, std::memory_order_acquire);

    for (int li = 0; li < kNumLayers; ++li)
    {
        const Layer& l = m_layers[li];
        uint64_t irFreq = 0, fdl = 0, accum = 0, tail = 0;

        irFreq += addIfAlive(l.irFreqDomain, l.allocSizes.irFreqDomain, "irFreqDomain");
        irFreq += addIfAlive(l.irFreqReal,   l.allocSizes.irFreqReal,   "irFreqReal");
        irFreq += addIfAlive(l.irFreqImag,   l.allocSizes.irFreqImag,   "irFreqImag");
        fdl    += addIfAlive(l.fdlBuf,       l.allocSizes.fdlBuf,       "fdlBuf");
        fdl    += addIfAlive(l.fdlReal,      l.allocSizes.fdlReal,      "fdlReal");
        fdl    += addIfAlive(l.fdlImag,      l.allocSizes.fdlImag,      "fdlImag");
        accum  += addIfAlive(l.fftTimeBuf,   l.allocSizes.fftTimeBuf,   "fftTimeBuf");
        accum  += addIfAlive(l.fftOutBuf,    l.allocSizes.fftOutBuf,    "fftOutBuf");
        accum  += addIfAlive(l.prevInputBuf, l.allocSizes.prevInputBuf, "prevInputBuf");
        accum  += addIfAlive(l.accumBuf,     l.allocSizes.accumBuf,     "accumBuf");
        accum  += addIfAlive(l.accumReal,    l.allocSizes.accumReal,    "accumReal");
        accum  += addIfAlive(l.accumImag,    l.allocSizes.accumImag,    "accumImag");
        accum  += addIfAlive(l.inputAccBuf,  l.allocSizes.inputAccBuf,  "inputAccBuf");
        tail   += addIfAlive(l.tailOutputBuf,l.allocSizes.tailOutputBuf,"tailOutputBuf");

        snap.layerBufs[li]  = irFreq + fdl + accum + tail;
        snap.irFreqBytes   += irFreq;
        snap.fdlBytes      += fdl;
        snap.accumBytes    += accum;
        snap.tailBytes     += tail;
    }

    snap.directBytes = addIfAlive(m_directIRRev,
        static_cast<size_t>(m_directTapCount) * sizeof(double), "directIRRev");
    snap.ringBytes   = addIfAlive(m_ringBuf,
        static_cast<size_t>(m_ringSize) * sizeof(double), "ringBuf");
    return snap;
}
#endif

//==============================================================================
// SetImpulse  ─ Message Thread のみ
//==============================================================================
bool MKLNonUniformConvolver::SetImpulse(const double* impulse, int irLen, int blockSize, double scale,
                                        bool enableDirectHead,
                                        const FilterSpec* filterSpec)
{
    convo::publishAtomic(m_ready, false, std::memory_order_release);

    if (impulse == nullptr || irLen <= 0 || blockSize <= 0)
        return false;

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t diagSeq   = globalDiagSeq.fetch_add(1, std::memory_order_relaxed) + kDiagSeqFirstRuntime;
    const uint64_t beforeMkl = convo::diag::allocatedBytes();
    const uint32_t beforeLost = convo::diag::lostFreeCount();
#endif
    releaseAllLayers();

    // tailMode: 0=Air Absorption, 1=Layer Tail Contouring, 2=Bypass
    const int tailMode = (filterSpec != nullptr) ? juce::jlimit(0, 2, filterSpec->tailMode) : 1;
    const bool tailEnabled = (tailMode != 2) && ((filterSpec != nullptr) ? filterSpec->tailEnabled : true);
    const double sampleRateForTail = (filterSpec != nullptr) ? filterSpec->sampleRate : 48000.0;
    double tailStartSec = (filterSpec != nullptr) ? juce::jlimit(0.01, 0.80, filterSpec->tailStartSeconds) : 0.085;
    const double userTailStrength = (filterSpec != nullptr) ? juce::jlimit(0.0, 2.0, filterSpec->tailStrength) : 1.0;
    double tailStrength = userTailStrength;
    int tailL1L2Mult = (filterSpec != nullptr) ? juce::jlimit(2, 16, filterSpec->tailL1L2Multiplier) : 8;

    double layer1Gain = 1.0;
    double layer2Gain = 1.0;

    const double strength01 = juce::jlimit(0.0, 1.0, userTailStrength * 0.5);

    // Tail profile redesign:
    // - Air Absorption: tail gain is reduced by layer and HF damping is applied to L1/L2 spectra.
    // - Layer Tail Contouring: enforce robust lower bounds and shape layer gains for contour clarity.
    if (!tailEnabled)
    {
        tailStrength = 0.0;
        layer1Gain = 0.0;
        layer2Gain = 0.0;
    }
    else if (tailMode == 0)
    {
        // Air Absorption mode: preserve early reflections while progressively damping late layers.
        tailStartSec = juce::jlimit(0.01, 0.80, std::max(tailStartSec, 0.055));
        tailL1L2Mult = juce::jlimit(2, 16, std::max(tailL1L2Mult, 6));
        tailStrength = juce::jlimit(0.0, 2.0, userTailStrength);

        layer1Gain = juce::jlimit(0.0, 2.0, tailStrength * (0.95 - 0.25 * strength01));
        layer2Gain = juce::jlimit(0.0, 2.0, tailStrength * (0.80 - 0.45 * strength01));
    }
    else if (tailMode == 1)
    {
        tailStartSec = juce::jlimit(0.01, 0.80, std::max(tailStartSec, 0.12));
        tailStrength = juce::jlimit(0.0, 2.0, std::max(tailStrength, 1.25));
        // tailL1L2Mult 最小値を 12→8 に緩和（案A）。数学的に出力は同一であり、
        // L2 small buffers (fftTimeBuf/fftOutBuf 等) が 12.4MB → 5.5MB (56%減)。
        // 83ms IR 環境では L1/L2 が生成されないため影響ゼロ。
        tailL1L2Mult = juce::jlimit(2, 16, std::max(tailL1L2Mult, 8));

        layer1Gain = juce::jlimit(0.0, 2.0, tailStrength * (1.05 + 0.20 * strength01));
        layer2Gain = juce::jlimit(0.0, 2.0, tailStrength * (0.82 + 0.12 * strength01));
    }
    else
    {
        // Bypass mode (defensive path)
        tailStrength = 0.0;
        layer1Gain = 0.0;
        layer2Gain = 0.0;
    }

    m_tailEnabled = tailEnabled;
    m_tailStrength = tailEnabled ? tailStrength : 0.0;
    m_maxBlockSize = blockSize;
    m_tailLayerGain[0] = 1.0;
    m_tailLayerGain[1] = layer1Gain;
    m_tailLayerGain[2] = layer2Gain;

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
        m_directIRRev   = static_cast<double*>(DIAG_MKL_MALLOC(static_cast<size_t>(m_directTapCount) * sizeof(double), 64));
        m_directHistory = (m_directHistLen > 0)
            ? static_cast<double*>(DIAG_MKL_MALLOC(static_cast<size_t>(m_directHistLen) * sizeof(double), 64))
            : nullptr;
        m_directWindow  = static_cast<double*>(DIAG_MKL_MALLOC(static_cast<size_t>(m_directHistLen + m_directMaxBlock) * sizeof(double), 64));
        m_directOutBuf  = static_cast<double*>(DIAG_MKL_MALLOC(static_cast<size_t>(m_directMaxBlock) * sizeof(double), 64));

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
    const int l1Part = l0Part * tailL1L2Mult;
    const int l2Part = l1Part * tailL1L2Mult;

    const int l0MaxLen = kL0MaxParts * l0Part;
    const int l0LenByTailStart = static_cast<int>(std::llround(tailStartSec * sampleRateForTail));
    const int l0LenTarget = juce::jlimit(l0Part, l0MaxLen, l0LenByTailStart);
    const int l0Len = std::min(irLen, tailEnabled ? l0LenTarget : l0MaxLen);

    const int l1Offset = l0Len;
    const int l1Len    = tailEnabled ? std::max(0, std::min(irLen - l0Len, kL1MaxParts * l1Part)) : 0;

    const int l2Offset = l0Len + l1Len;
    const int l2Len    = tailEnabled ? std::max(0, irLen - l0Len - l1Len) : 0;

    struct LayerCfg { int offset; int len; int partSize; bool immediate; };
    const LayerCfg cfgs[kNumLayers] = {
        { 0,        l0Len, l0Part, true  },
        { l1Offset, l1Len, l1Part, false },
        { l2Offset, l2Len, l2Part, false },
    };

    m_numActiveLayers = 0;

    // ────────────────────────────────────────────────
    // 各レイヤーの遅延補償用エントリを保持
    int prevLayerTotalSamples = 0;  // ★ B13: 先行レイヤーの IR 総長

    // ────────────────────────────────────────────────
    // 各レイヤーを初期化
    // ────────────────────────────────────────────────
    for (int li = 0; li < kNumLayers; ++li)
    {
        if (cfgs[li].len <= 0)
            continue;

        Layer& l = m_layers[m_numActiveLayers];
        l.descriptorCommitted = false;
        convo::publishAtomic(l.warmupCompleted, false, std::memory_order_release);

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

            if (const IppFFTPlan* plan = IppFFTPlanCache::getOrCreate(order); plan != nullptr)
                l.fftPlanOwner = std::cref(*plan);
            else
                l.fftPlanOwner.reset();

            if (!l.fftPlanOwner.has_value() || l.fftPlanOwner->get().fftSpec == nullptr)
            {
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
                juce::Logger::writeToLog("MKLNonUniformConvolver: FFT plan cache creation failed for layer "
                                         + juce::String(li) + " (order=" + juce::String(order) + ")");
#endif
                releaseAllLayers();
                return false;
            }

            l.fftSpec = l.fftPlanOwner->get().fftSpec;

            // ワークバッファ確保 (Audio Thread での動的確保を防ぐため事前確保)
            // sizeWork == 0 の場合 nullptr のまま (IPP が外部バッファ不要)
            // sizeWork > 0 かつ確保失敗 → リアルタイム安全でないため初期化失敗とする
            if (l.fftPlanOwner->get().sizeWork > 0)
            {
                l.fftWorkBuf = ippsMalloc_8u(l.fftPlanOwner->get().sizeWork);
                if (!l.fftWorkBuf)
                {
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
                    juce::Logger::writeToLog("MKLNonUniformConvolver: ippsMalloc_8u(sizeWork=" + juce::String(l.fftPlanOwner->get().sizeWork)
                                             + ") failed for layer " + juce::String(li));
#endif
                    releaseAllLayers();
                    return false;
                }
            }
            // else: sizeWork == 0 → l.fftWorkBuf は nullptr のまま (正常)

            l.descriptorCommitted = true;
        }

        // ── バッファ確保 (すべて mkl_malloc 64byte アライン) ──
        // [Mem-Fix] irFreqDomain/fdlBuf は永続履歴ではなく使い捨てスクラッチ。
        // irFreqDomain: 1パーティション分、fdlBuf: current+mirrorの2スロット分のみ。
        const size_t irBufSize  = static_cast<size_t>(l.partStride);
        const size_t fdlBufSize = static_cast<size_t>(l.partStride) * 2;
        const size_t irSoaSize  = static_cast<size_t>(l.numParts) * static_cast<size_t>(l.complexSize);
        const size_t fdlSoaSize = static_cast<size_t>(l.numParts) * 2 * static_cast<size_t>(l.complexSize);

        l.irFreqDomain = static_cast<double*>(DIAG_MKL_MALLOC(irBufSize  * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
l.allocSizes.irFreqDomain = irBufSize * sizeof(double);
#endif
        l.irFreqReal   = static_cast<double*>(DIAG_MKL_MALLOC(irSoaSize  * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
l.allocSizes.irFreqReal = irSoaSize * sizeof(double);
#endif
        l.irFreqImag   = static_cast<double*>(DIAG_MKL_MALLOC(irSoaSize  * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
l.allocSizes.irFreqImag = irSoaSize * sizeof(double);
#endif
        l.fdlBuf       = static_cast<double*>(DIAG_MKL_MALLOC(fdlBufSize * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
l.allocSizes.fdlBuf = fdlBufSize * sizeof(double);
#endif
        l.fdlReal      = static_cast<double*>(DIAG_MKL_MALLOC(fdlSoaSize * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
l.allocSizes.fdlReal = fdlSoaSize * sizeof(double);
#endif
        l.fdlImag      = static_cast<double*>(DIAG_MKL_MALLOC(fdlSoaSize * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
l.allocSizes.fdlImag = fdlSoaSize * sizeof(double);
#endif
        l.fftTimeBuf   = static_cast<double*>(DIAG_MKL_MALLOC(l.fftSize   * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
l.allocSizes.fftTimeBuf = l.fftSize * sizeof(double);
#endif
        l.fftOutBuf    = static_cast<double*>(DIAG_MKL_MALLOC(l.fftSize   * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
l.allocSizes.fftOutBuf = l.fftSize * sizeof(double);
#endif
        l.prevInputBuf = static_cast<double*>(DIAG_MKL_MALLOC(l.partSize  * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
l.allocSizes.prevInputBuf = l.partSize * sizeof(double);
#endif
        l.accumBuf     = static_cast<double*>(DIAG_MKL_MALLOC(l.partStride * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
l.allocSizes.accumBuf = l.partStride * sizeof(double);
#endif
        l.accumReal    = static_cast<double*>(DIAG_MKL_MALLOC(l.complexSize * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
l.allocSizes.accumReal = l.complexSize * sizeof(double);
#endif
        l.accumImag    = static_cast<double*>(DIAG_MKL_MALLOC(l.complexSize * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
l.allocSizes.accumImag = l.complexSize * sizeof(double);
#endif
        l.inputAccBuf  = static_cast<double*>(DIAG_MKL_MALLOC(l.partSize  * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
l.allocSizes.inputAccBuf = l.partSize * sizeof(double);
#endif

        if (!l.isImmediate)
        {
            l.tailOutputBuf = static_cast<double*>(DIAG_MKL_MALLOC(l.partSize * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
            l.allocSizes.tailOutputBuf = l.partSize * sizeof(double);   // ★ Bug#7: isImmediate ガード内に移動
#endif
        }

        if (!l.irFreqDomain || !l.irFreqReal || !l.irFreqImag || !l.fdlBuf || !l.fdlReal || !l.fdlImag || !l.fftTimeBuf ||
            !l.fftOutBuf || !l.prevInputBuf || !l.accumBuf || !l.accumReal || !l.accumImag || !l.inputAccBuf ||
            (!l.isImmediate && !l.tailOutputBuf))
        {
            releaseAllLayers();
            return false;
        }

        // ゼロ初期化
        juce::FloatVectorOperations::clear(l.irFreqDomain, irBufSize);
        juce::FloatVectorOperations::clear(l.irFreqReal,   irSoaSize);
        juce::FloatVectorOperations::clear(l.irFreqImag,   irSoaSize);
        juce::FloatVectorOperations::clear(l.fdlBuf,       fdlBufSize);
        juce::FloatVectorOperations::clear(l.fdlReal,      fdlSoaSize);
        juce::FloatVectorOperations::clear(l.fdlImag,      fdlSoaSize);
        juce::FloatVectorOperations::clear(l.fftTimeBuf,   l.fftSize);
        juce::FloatVectorOperations::clear(l.fftOutBuf,    l.fftSize);
        juce::FloatVectorOperations::clear(l.prevInputBuf, l.partSize);
        juce::FloatVectorOperations::clear(l.accumBuf,     l.partStride);
        juce::FloatVectorOperations::clear(l.accumReal,    l.complexSize);
        juce::FloatVectorOperations::clear(l.accumImag,    l.complexSize);
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

            // [Mem-Fix] irFreqDomain は 1 パーティション分のスクラッチのため、オフセット0(先頭)へ書き込む。
            memcpy(l.irFreqDomain, tempFreq, l.complexSize * 2 * sizeof(double));

            if (scale != 1.0)
                cblas_dscal(l.complexSize * 2, scale, l.irFreqDomain, 1);

            deinterleaveComplex(l.irFreqDomain,
                        l.irFreqReal + static_cast<size_t>(p) * l.complexSize,
                        l.irFreqImag + static_cast<size_t>(p) * l.complexSize,
                        l.complexSize);
        }

        // Backward FFT のウォームアップ
        // Audio Thread での初回実行時の遅延 (IPP テーブル生成等) を事前消化する。
        // [v2.1] IFFT: CCS → real (IPP_FFT_DIV_INV_BY_N により 1/N 正規化済み)
        ippsFFTInv_CCSToR_64f(tempFreq, tempTime, l.fftSpec, l.fftWorkBuf);
        convo::publishAtomic(l.warmupCompleted, true, std::memory_order_release);

        mkl_free(tempTime);
        mkl_free(tempFreq);

        // [Mem-Fix] IR パーティションを逆順に並び替える (forward アクセス最適化)
        // irFreqDomain はスクラッチ化されたため、swap対象は SoA (irFreqReal/irFreqImag) のみでよい。
        if (l.numPartsIR > 1)
        {
            double* swapSoA = static_cast<double*>(mkl_malloc(
                static_cast<size_t>(l.complexSize) * sizeof(double), 64));
            if (swapSoA)
            {
                for (int pf = 0; pf < l.numPartsIR / 2; ++pf)
                {
                    const int pb = l.numPartsIR - 1 - pf;

                    // irFreqReal swap (SoA)
                    double* realF = l.irFreqReal + static_cast<size_t>(pf) * l.complexSize;
                    double* realB = l.irFreqReal + static_cast<size_t>(pb) * l.complexSize;
                    memcpy(swapSoA, realF, l.complexSize * sizeof(double));
                    memcpy(realF,   realB, l.complexSize * sizeof(double));
                    memcpy(realB,   swapSoA, l.complexSize * sizeof(double));

                    // irFreqImag swap (SoA)
                    double* imagF = l.irFreqImag + static_cast<size_t>(pf) * l.complexSize;
                    double* imagB = l.irFreqImag + static_cast<size_t>(pb) * l.complexSize;
                    memcpy(swapSoA, imagF, l.complexSize * sizeof(double));
                    memcpy(imagF,   imagB, l.complexSize * sizeof(double));
                    memcpy(imagB,   swapSoA, l.complexSize * sizeof(double));
                }
            }
            if (swapSoA) mkl_free(swapSoA);
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
        l.outputDelaySamples = 0;

        // ★ B13: 遅延補償リングバッファ設定 (L1/L2)
        if (prevLayerTotalSamples > 0) {
            l.outputDelaySamples = prevLayerTotalSamples;
            l.delayLineCapacity = ((prevLayerTotalSamples + l.partSize + m_maxBlockSize + 15) / 16) * 16;
            const size_t delayLineBytes = static_cast<size_t>(l.delayLineCapacity) * sizeof(double);
            l.delayLineBuf = static_cast<double*>(DIAG_MKL_MALLOC(delayLineBytes, 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
            l.allocSizes.delayLineBuf = delayLineBytes;
#endif
            if (l.delayLineBuf == nullptr) {
                releaseAllLayers();
                return false;
            }
            juce::FloatVectorOperations::clear(l.delayLineBuf, l.delayLineCapacity);
            jassert(l.outputDelaySamples > 0);
        }

        ++m_numActiveLayers;

        // ★ B13: 先行レイヤーの IR 総長を累積 (次レイヤーの outputDelaySamples 用)
        prevLayerTotalSamples += cfgs[li].len;
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
    m_ringBuf = static_cast<double*>(DIAG_MKL_MALLOC(finalSize * sizeof(double), 64));
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

    if (tailEnabled && tailMode == 0)
    {
        const double startNorm = juce::jlimit(0.65, 1.55, tailStartSec / 0.085);
        const double dampingBase = (0.35 + 1.10 * strength01) * startNorm;

        for (int li = 1; li < m_numActiveLayers; ++li)
        {
            Layer& l = m_layers[li];
            if (!l.irFreqReal || !l.irFreqImag || l.complexSize <= 0)
                continue;

            const double layerWeight = (li == 1) ? 1.0 : 1.6;
            const double dampingCoeff = dampingBase * layerWeight;

            // [Mem-Fix] gain は実数値(振幅のみ)のためinterleaved配列は不要。
            convo::ScopedAlignedPtr<double> gainReal(
                static_cast<double*>(mkl_malloc(static_cast<size_t>(l.complexSize) * sizeof(double), 64)));
            if (!gainReal.get())
                continue;

            const double denom = static_cast<double>(std::max(1, l.complexSize - 1));
            for (int k = 0; k < l.complexSize; ++k)
            {
                const double fNorm = static_cast<double>(k) / denom;
                const double hfTilt = std::exp(-dampingCoeff * fNorm * fNorm);
                gainReal.get()[k] = hfTilt;
            }

            // [Mem-Fix] SoA (irFreqReal/irFreqImag) に直接ゲインを適用する。
            for (int p = 0; p < l.numParts; ++p)
            {
                double* re = l.irFreqReal + static_cast<size_t>(p) * l.complexSize;
                double* im = l.irFreqImag + static_cast<size_t>(p) * l.complexSize;
                vdMul(l.complexSize, re, gainReal.get(), re);
                vdMul(l.complexSize, im, gainReal.get(), im);
            }
        }
    }

    convo::publishAtomic(m_ready, true, std::memory_order_release);
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t afterMkl = convo::diag::allocatedBytes();
    const uint32_t afterLost = convo::diag::lostFreeCount();

    const int l0p = m_numActiveLayers >= 1 ? m_layers[0].partSize : 0;
    const int l1p = m_numActiveLayers >= 2 ? m_layers[1].partSize : 0;
    const int l2p = m_numActiveLayers >= 3 ? m_layers[2].partSize : 0;

    // IR_LOAD
    diagLogNonRt(juce::String::formatted(
        "[IR_LOAD] NUC#%p seq=%llu irLen=%d blockSize=%d "
        "Layers=%d L0Part=%d L1Part=%d L2Part=%d "
        "directTaps=%d ringSize=%d "
        "MKL: before=%lluMB after=%lluMB delta=%lldMB "
        "lostFree=%u(+%d) live=%u",
        (void*)this,
        (unsigned long long)diagSeq,
        irLen, blockSize,
        m_numActiveLayers, l0p, l1p, l2p,
        m_directTapCount, m_ringSize,
        (unsigned long long)(beforeMkl / (1024*1024)),
        (unsigned long long)(afterMkl / (1024*1024)),
        (long long)((int64_t)(afterMkl) - (int64_t)(beforeMkl)) / (1024*1024),
        (unsigned)afterLost, (int)((int32_t)(afterLost) - (int32_t)(beforeLost)),
        (unsigned)liveCount.load(std::memory_order_relaxed)));

    // IR_LAYOUT (1 回の getDiagnostics で Layer 情報 + 種別内訳を取得)
    const auto __snap = getDiagnostics();
    diagLogNonRt(juce::String::formatted(
        "[IR_LAYOUT] NUC#%p seq=%llu "
        "IRFreq=%.0fMB FDL=%.0fMB Accum=%.0fMB Tail=%.0fMB "
        "Direct=%.0fMB Ring=%.0fMB Total=%.0fMB(persistent data buffers only) | "
        "L0=%.0fMB L1=%.0fMB L2=%.0fMB",
        (void*)this,
        (unsigned long long)diagSeq,
        __snap.irFreqBytes / (1024.0*1024.0),
        __snap.fdlBytes    / (1024.0*1024.0),
        __snap.accumBytes  / (1024.0*1024.0),
        __snap.tailBytes   / (1024.0*1024.0),
        __snap.directBytes / (1024.0*1024.0),
        __snap.ringBytes   / (1024.0*1024.0),
        __snap.totalBytes() / (1024.0*1024.0),
        __snap.layerBufs[0] / (1024.0*1024.0),
        __snap.layerBufs[1] / (1024.0*1024.0),
        __snap.layerBufs[2] / (1024.0*1024.0)));
#endif


    return true;
}

bool MKLNonUniformConvolver::areFftDescriptorsCommitted() const noexcept
{
    if (!convo::consumeAtomic(m_ready, std::memory_order_acquire) || m_numActiveLayers <= 0)
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
#ifdef NUC_DEBUG_GUARDS
    checkGuards();
#endif
#if JUCE_DEBUG
    if (!convo::consumeAtomic(l.warmupCompleted, std::memory_order_acquire))
    convo::fetchAddAtomic(debugWarmupGuardCount(), 1, std::memory_order_acq_rel);
#endif

    // ── 1. [prevInput | currentInput] を fftTimeBuf に配置 (Overlap-Save) ──
    juce::FloatVectorOperations::copy(l.fftTimeBuf,              l.prevInputBuf, l.partSize);
    juce::FloatVectorOperations::copy(l.fftTimeBuf + l.partSize, l.inputAccBuf,  l.partSize);
    juce::FloatVectorOperations::copy(l.prevInputBuf, l.inputAccBuf, l.partSize);

    // ── 2. Forward FFT ──
    // [v2.1] ippsFFTFwd_RToCCS_64f: real → CCS interleaved complex
    // CCS 出力形式: [re0,im0,re1,im1,...] ← 既存 AVX2 複素乗算と完全互換
    // [Mem-Fix] fdlBuf は使い捨てスクラッチ (current=offset0 / mirror=offset partStride)。
    // 永続履歴は fdlReal/fdlImag (SoA) 側にのみ保持する。
    double* currentFDLSlot = l.fdlBuf;
    ippsFFTFwd_RToCCS_64f(l.fftTimeBuf, currentFDLSlot, l.fftSpec, l.fftWorkBuf);

    deinterleaveComplex(currentFDLSlot,
                        l.fdlReal + static_cast<size_t>(l.fdlIndex) * l.complexSize,
                        l.fdlImag + static_cast<size_t>(l.fdlIndex) * l.complexSize,
                        l.complexSize);

    // [最適化2] Linearized ring buffer: mirror write
    double* mirrorFDLSlot = l.fdlBuf + l.partStride;
    memcpy(mirrorFDLSlot, currentFDLSlot, l.partStride * sizeof(double));

    const int mirrorIndex = l.fdlIndex + l.numParts;
    deinterleaveComplex(mirrorFDLSlot,
                        l.fdlReal + static_cast<size_t>(mirrorIndex) * l.complexSize,
                        l.fdlImag + static_cast<size_t>(mirrorIndex) * l.complexSize,
                        l.complexSize);

    // ── 3. 複素乗算積算 (FDL × IR) → accumBuf ──
    memset(l.accumReal, 0, static_cast<size_t>(l.complexSize) * sizeof(double));
    memset(l.accumImag, 0, static_cast<size_t>(l.complexSize) * sizeof(double));

    // [Mem-Fix] AoS(fdlBuf/irFreqDomain) 経由の読み出しとダミーprefetchを廃止し、
    // SoA (fdlReal/fdlImag, irFreqReal/irFreqImag) のみを読む一本化されたパスにする。
    const int linStart = l.fdlIndex - l.numPartsIR + 1 + l.numParts;

    for (int p = 0; p < l.numPartsIR; ++p)
    {
        const int index = linStart + p;
        const double* srcARe = l.fdlReal    + static_cast<size_t>(index) * l.complexSize;
        const double* srcAIm = l.fdlImag    + static_cast<size_t>(index) * l.complexSize;
        const double* srcBRe = l.irFreqReal + static_cast<size_t>(p)     * l.complexSize;
        const double* srcBIm = l.irFreqImag + static_cast<size_t>(p)     * l.complexSize;

        if (p + 1 < l.numPartsIR)
        {
            _mm_prefetch((const char*)(l.fdlReal    + static_cast<size_t>(index + 1) * l.complexSize), _MM_HINT_T1);
            _mm_prefetch((const char*)(l.irFreqReal + static_cast<size_t>(p + 1)     * l.complexSize), _MM_HINT_T1);
        }

        accumulateSplitComplex(srcARe, srcAIm, srcBRe, srcBIm, l.accumReal, l.accumImag, l.complexSize);
    }

    memset(l.accumBuf, 0, l.partStride * sizeof(double));
    interleaveComplex(l.accumReal, l.accumImag, l.accumBuf, l.complexSize);

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
#ifdef NUC_DEBUG_GUARDS
    checkGuards();
#endif
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
        // [BUG-02] m_ringWrite は直前の更新で既に正しい位置にある。overflow 時の追加更新は不要。
        // m_ringWrite = (m_ringWrite + overflow) & m_ringMask;
        convo::fetchAddAtomic(m_ringOverflowCount, 1, std::memory_order_acq_rel);
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
    #ifdef NUC_DEBUG_GUARDS
        checkGuards();
        // 内部バッファの簡単な健全性チェック
        if (m_ringBuf && reinterpret_cast<uintptr_t>(m_ringBuf) % 64 != 0) __debugbreak();
    #endif
    if (!convo::consumeAtomic(m_ready, std::memory_order_acquire) || numSamples <= 0)
        return;

    processDirectBlock(input, numSamples);

    for (int li = 0; li < m_numActiveLayers; ++li)
    {
        Layer& l = m_layers[li];

#if JUCE_DEBUG
        if (!convo::consumeAtomic(l.warmupCompleted, std::memory_order_acquire))
            convo::fetchAddAtomic(debugWarmupGuardCount(), 1, std::memory_order_acq_rel);
#endif

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
                    // ★ [P0-3] Release 安全ガード: 超過時は clamp
                    if (consumed > numSamples) [[unlikely]]
                    {
                        consumed = numSamples;
                    }

                    juce::FloatVectorOperations::copy(l.fftTimeBuf,              l.prevInputBuf, l.partSize);
                    juce::FloatVectorOperations::copy(l.fftTimeBuf + l.partSize, l.inputAccBuf,  l.partSize);
                    juce::FloatVectorOperations::copy(l.prevInputBuf, l.inputAccBuf, l.partSize);

                    // [v2.1] L1/L2 Forward FFT: real → CCS
                    // [Mem-Fix] fdlBuf は使い捨てスクラッチ (current=offset0 / mirror=offset partStride)。
                    double* currentFDLSlot = l.fdlBuf;
                    ippsFFTFwd_RToCCS_64f(l.fftTimeBuf, currentFDLSlot, l.fftSpec, l.fftWorkBuf);

                    deinterleaveComplex(currentFDLSlot,
                                        l.fdlReal + static_cast<size_t>(l.fdlIndex) * l.complexSize,
                                        l.fdlImag + static_cast<size_t>(l.fdlIndex) * l.complexSize,
                                        l.complexSize);

                    // [最適化2] mirror write
                    double* mirrorFDLSlot = l.fdlBuf + l.partStride;
                    juce::FloatVectorOperations::copy(mirrorFDLSlot, currentFDLSlot, l.partStride);

                    const int mirrorIndex = l.fdlIndex + l.numParts;
                    deinterleaveComplex(mirrorFDLSlot,
                                        l.fdlReal + static_cast<size_t>(mirrorIndex) * l.complexSize,
                                        l.fdlImag + static_cast<size_t>(mirrorIndex) * l.complexSize,
                                        l.complexSize);

                    l.fdlIndex = (l.fdlIndex + 1) & l.fdlMask;

                    // [Bug2 fix] FDL スナップショット保存
                    l.baseFdlIdxSaved = (l.fdlIndex - 1 + l.numParts) & l.fdlMask;

                    memset(l.accumReal, 0, static_cast<size_t>(l.complexSize) * sizeof(double));
                    memset(l.accumImag, 0, static_cast<size_t>(l.complexSize) * sizeof(double));
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
            const int baseFdlIdx = l.baseFdlIdxSaved;
            const int linStart   = baseFdlIdx - l.numPartsIR + 1 + l.numParts;

            // [Mem-Fix] AoS(fdlBuf/irFreqDomain)経由の読み出しを廃止し、
            // SoA (fdlReal/fdlImag, irFreqReal/irFreqImag) のみを読む一本化されたパスにする。
            for (int p = l.nextPart; p < endPart; ++p)
            {
                const int index = linStart + p;
                const double* srcARe = l.fdlReal    + static_cast<size_t>(index) * l.complexSize;
                const double* srcAIm = l.fdlImag    + static_cast<size_t>(index) * l.complexSize;
                const double* srcBRe = l.irFreqReal + static_cast<size_t>(p)     * l.complexSize;
                const double* srcBIm = l.irFreqImag + static_cast<size_t>(p)     * l.complexSize;

                if (p + 1 < endPart)
                {
                    _mm_prefetch((const char*)(l.fdlReal    + static_cast<size_t>(index + 1) * l.complexSize), _MM_HINT_T1);
                    _mm_prefetch((const char*)(l.irFreqReal + static_cast<size_t>(p + 1)     * l.complexSize), _MM_HINT_T1);
                }

                accumulateSplitComplex(srcARe, srcAIm, srcBRe, srcBIm, l.accumReal, l.accumImag, l.complexSize);
            }

            l.nextPart = endPart;

            memset(l.accumBuf, 0, l.partStride * sizeof(double));
            interleaveComplex(l.accumReal, l.accumImag, l.accumBuf, l.complexSize);

            // ── 全パーティション累積完了 → IFFT → tailOutputBuf へコピー ──
            if (l.nextPart >= l.numPartsIR)
            {
                // [v2.1] Backward FFT: CCS → real (Audio Thread 内で再初期化禁止の制約はIPPも同様)
                ippsFFTInv_CCSToR_64f(l.accumBuf, l.fftOutBuf, l.fftSpec, l.fftWorkBuf);

                memcpy(l.tailOutputBuf, l.fftOutBuf + l.partSize, l.partSize * sizeof(double));
                l.tailOutputPos = 0;

                // ★ B13: 遅延補償リングバッファに書き込み
                if (l.delayLineBuf != nullptr)
                    delayLineWrite(l, l.tailOutputBuf, l.partSize);

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
    #ifdef NUC_DEBUG_GUARDS
        checkGuards();
    #endif
    if (!convo::consumeAtomic(m_ready, std::memory_order_acquire) || numSamples <= 0)
    {
        if (output && numSamples > 0)
            memset(output, 0, numSamples * sizeof(double));
        return 0;
    }

    const int got = ringRead(output, numSamples);

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

    [[maybe_unused]] auto addScaledFallback = [&addFallback](int n, double* dst, const double* src, double gain) noexcept
    {
        if (absNoLibm(gain - 1.0) < 1.0e-12)
        {
            addFallback(n, dst, src);
            return;
        }
        for (int i = 0; i < n; ++i)
            dst[i] += src[i] * gain;
    };

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

    // ── L1/L2 出力 (B13: 遅延補償リングバッファ経由) ──
    for (int li = 1; li < m_numActiveLayers; ++li)
    {
        Layer& l = m_layers[li];
        if (l.delayLineBuf == nullptr) continue;

        if (output != nullptr)
        {
            const double layerGain = m_tailEnabled
                ? m_tailLayerGain[juce::jlimit(0, kNumLayers - 1, li)]
                : 0.0;
            delayLineReadAdd(l, output, numSamples, layerGain);
        }
    }

    return got;
}

//==============================================================================
// ★ B13: delayLineWrite — 遅延補償リングバッファ書き込み (Add / IFFT完了時)
//==============================================================================
void MKLNonUniformConvolver::delayLineWrite(Layer& l, const double* src, int n) noexcept
{
    const size_t writeOffset = static_cast<size_t>(l.delayWriteCursor % static_cast<uint64_t>(l.delayLineCapacity));
    const int remain = l.delayLineCapacity - static_cast<int>(writeOffset);
    const int first = std::min(n, remain);
    juce::FloatVectorOperations::copy(l.delayLineBuf + writeOffset, src, first);
    if (first < n)
        juce::FloatVectorOperations::copy(l.delayLineBuf, src + first, n - first);
    l.delayWriteCursor += static_cast<uint64_t>(n);
}

//==============================================================================
// ★ B13: delayLineReadAdd — 遅延補償リングバッファ読み出し + 加算 (Get)
//==============================================================================
void MKLNonUniformConvolver::delayLineReadAdd(Layer& l, double* dst, int numSamples, double gain) noexcept
{
    if (l.delayLineBuf == nullptr || l.delayLineCapacity <= 0 || dst == nullptr)
        return;

    // ★ readCursor = max(readCursor, writeCursor - outputDelaySamples)
    const uint64_t maxRead = (l.delayWriteCursor >= static_cast<uint64_t>(l.outputDelaySamples))
        ? (l.delayWriteCursor - static_cast<uint64_t>(l.outputDelaySamples))
        : 0;
    const uint64_t actualReadStart = std::max(l.delayReadCursor, maxRead);

    // ★ Writer がまだ outputDelaySamples 分先に進んでいない → スキップ
    if (actualReadStart + static_cast<uint64_t>(numSamples) > l.delayWriteCursor)
        return;

    // ★ リングバッファ読み出し
    const size_t readOffset = static_cast<size_t>(actualReadStart % static_cast<uint64_t>(l.delayLineCapacity));
    const int first = std::min(numSamples, l.delayLineCapacity - static_cast<int>(readOffset));
    if (first > 0) {
        const double* src = l.delayLineBuf + readOffset;
        if (absNoLibm(gain - 1.0) < 1.0e-12)
            for (int i = 0; i < first; ++i) dst[i] += src[i];
        else
            for (int i = 0; i < first; ++i) dst[i] += src[i] * gain;
    }
    if (first < numSamples) {
        const double* src = l.delayLineBuf;
        const int second = numSamples - first;
        if (absNoLibm(gain - 1.0) < 1.0e-12)
            for (int i = 0; i < second; ++i) dst[first + i] += src[i];
        else
            for (int i = 0; i < second; ++i) dst[first + i] += src[i] * gain;
    }

    l.delayReadCursor = actualReadStart + static_cast<uint64_t>(numSamples);
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

        // [Mem-Fix] fdlBuf は 2*partStride のスクラッチに縮小されているため、
        // クリアサイズも実際の確保サイズに合わせる (旧サイズのままだと範囲外書き込みになる)。
        const size_t fdlBufSize = static_cast<size_t>(l.partStride) * 2;
        const size_t fdlSoaSize = static_cast<size_t>(l.numParts) * 2 * static_cast<size_t>(l.complexSize);
        juce::FloatVectorOperations::clear(l.fdlBuf,       fdlBufSize);
        juce::FloatVectorOperations::clear(l.fdlReal,      fdlSoaSize);
        juce::FloatVectorOperations::clear(l.fdlImag,      fdlSoaSize);
        juce::FloatVectorOperations::clear(l.fftTimeBuf,   l.fftSize);
        juce::FloatVectorOperations::clear(l.fftOutBuf,    l.fftSize);
        juce::FloatVectorOperations::clear(l.prevInputBuf, l.partSize);
        juce::FloatVectorOperations::clear(l.accumBuf,     l.partStride);
        juce::FloatVectorOperations::clear(l.accumReal,    l.complexSize);
        juce::FloatVectorOperations::clear(l.accumImag,    l.complexSize);
        juce::FloatVectorOperations::clear(l.inputAccBuf,  l.partSize);

        if (l.tailOutputBuf)
            juce::FloatVectorOperations::clear(l.tailOutputBuf, l.partSize);

        l.fdlIndex        = 0;
        l.inputPos        = 0;
        l.nextPart        = 0;
        l.tailOutputPos   = 0;
        l.baseFdlIdxSaved = 0;
        l.distributing    = false;

        // ★ B13: 遅延補償リセット (状態のみ、構成情報は保持)
        l.resetDelayAlignment();
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
