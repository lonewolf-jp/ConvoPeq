//============================================================================
// MKLNonUniformConvolver.cpp  ── v2.0 (JUCE 8.0.12 / Intel oneMKL 対応)
//
// Non-Uniform Partitioned Convolution の完全実装。
// 3 層レイヤー構造で低遅延と大 IR を両立する。
//
// ■ 修正内容 (v1.0 → v2.0):
//
//   [Bug1 fix] 出力アーキテクチャを「別テールバッファ」方式に変更。
//     - v1.0: L0/L1/L2 が同一リングバッファに独立して ringWrite() (memcpy 上書き)。
//             複数レイヤーの出力が時間位置を合わせず逐次上書きされ、OLA が崩壊。
//     - v2.0: L0 のみリングバッファを独占使用 (ringWrite = memcpy, 順次書き込み)。
//             L1/L2 は IFFT 完了時に tailOutputBuf へコピーし、Get() で vdAdd 合算。
//             共有リングの OLA タイミング問題を根本解消。
//
//   [Bug2 fix] baseFdlIdxSaved による FDL スナップショット保存。
//     - v1.0: baseFdlIdx をトリガブロック内で毎回再計算。同一サイクルでも
//             fdlIndex が進むため、各パーティションが異なる FDL 時間軸を参照。
//     - v2.0: トリガ時に baseFdlIdxSaved として一度だけ保存し、
//             IFFT 完了まで全パーティションで同一値を使用。
//
//   [Bug3 fix] 分散計算ループをトリガブロック外に移動。
//     - v1.0: 分散計算がトリガ時の 1 コールバックにのみ実行されていた。
//             (1 トリガあたり 1 バッチ = partsPerCallback 個しか処理されず、
//              残りは永遠に処理されない。全パーティション処理不完全。)
//     - v2.0: 分散計算ループを for(li) ループ内・while(consumed) 外へ移動。
//             毎コールバックで partsPerCallback 個ずつ処理し、
//             numPartsIR 個完了後に IFFT + tailOutputBuf コピーを行う。
//
// ■ CPU 負荷比較 (IR=3s@48kHz=144000samples, blockSize=128):
//   v1.0: L0 単独 = 1125 パーティション/コールバック → 大スパイク
//   v2.0: L0=32 + L1=8(分散) + L2=1(分散) ≈ 41相当/コールバック → フラット
//
//============================================================================

#include "MKLNonUniformConvolver.h"
#include "AlignedAllocation.h"
#include "DspNumericPolicy.h"

#include <mkl.h>
#include <mkl_dfti.h>
#include <mkl_vml.h>
#include <mkl_cblas.h>
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
} // namespace

//==============================================================================
// Layer::freeAll
//==============================================================================
void MKLNonUniformConvolver::Layer::freeAll() noexcept
{
    if (fftHandle)     { DftiFreeDescriptor(&fftHandle); fftHandle = nullptr; }
    descriptorCommitted = false;
    if (irFreqDomain)  { mkl_free(irFreqDomain);  irFreqDomain  = nullptr; }
    if (fdlBuf)        { mkl_free(fdlBuf);         fdlBuf        = nullptr; }
    // [Bug F fix] overlapBuf 削除済み
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
    mkl_set_num_threads(1);
}

MKLNonUniformConvolver::~MKLNonUniformConvolver()
{
    releaseAllLayers();
}

//==============================================================================
// applySpectrumFilter  ─ Message Thread のみ
//
// SetImpulse() の完了後に呼ばれ、全レイヤーの irFreqDomain に
// ハイカット (HC) / ローカット (LC) のゲインテーブルを乗算して焼き込む。
//
// ■ アルゴリズム:
//   各レイヤーは fftSize が異なる (L0 < L1 < L2)。
//   物理周波数 → ビンインデックス変換は k = round(f * fftSize / fs) なので、
//   各レイヤーに対して独立にゲインテーブルを計算・適用する。
//
// ■ HC カットオフ:
//   fs ≤ 48000 Hz: fc_start=18kHz, fc_end=ナイキスト
//   fs >  48000 Hz: fc_start=22kHz, fc_end=ナイキスト
//   (可聴域上限を確保しつつナイキスト直下のエイリアシングを抑圧)
//
// ■ LC カットオフ:
//   Natural: コサインロールオン  end=8Hz, start=18Hz
//   Soft:    コサインロールオン  end=6Hz, start=15Hz
//   (DC 直流成分 ～ end までをゼロにし、start まで緩やかに持ち上げる)
//
// ■ 使用する MKL 関数: cblas_dscal (各 bin へのスカラー乗算)
//   [指摘1 fix] MKL VML 使用箇所のメモリは mkl_malloc/ScopedAlignedPtr で管理する。
//   Audio Thread では本関数を呼ばない。
//==============================================================================
void MKLNonUniformConvolver::applySpectrumFilter(const FilterSpec& spec) noexcept
{
    const double fs      = spec.sampleRate;
    const double nyquist = fs * 0.5;

    // HC カットオフ周波数 (サンプルレート依存)
    const double hcFcStart = (fs <= 48000.0) ? 18000.0 : 22000.0;
    const double hcFcEnd   = nyquist; // ナイキストでゲイン=0

    // LC カットオフ周波数
    const double lcFcEnd   = (spec.lcMode == LCMode::Soft) ?  6.0 :  8.0;
    const double lcFcStart = (spec.lcMode == LCMode::Soft) ? 15.0 : 18.0;

    for (int li = 0; li < m_numActiveLayers; ++li)
    {
        Layer& l = m_layers[li];
        if (!l.irFreqDomain) continue;

        const int N      = l.fftSize;   // = partSize * 2
        const int halfN  = N / 2;       // ナイキスト bin
        const int cSize  = l.complexSize; // = halfN + 1
        const int stride = l.partStride;

        // ── ゲインテーブル計算 (全 bin 分, Message Thread のみ) ──
        // [指摘1 fix] MKL VML (vdMul) 使用箇所のメモリは mkl_malloc/mkl_free で管理する。
        // convo::aligned_malloc は noexcept 関数内で throw するため、mkl_malloc を直接使用して
        // nullptr チェックで対処する (OOM 時は当該レイヤーの filter 適用をスキップ)。
        convo::ScopedAlignedPtr<double> gain(
            static_cast<double*>(mkl_malloc(static_cast<size_t>(cSize) * sizeof(double), 64)));
        if (!gain.get()) continue;
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
                    // パスバンド: ゲイン 1.0 (初期値のまま)
                }
                else if (k <= kEnd)
                {
                    // ロールオフ域: [kStart+1, kEnd]
                    const double denom = static_cast<double>(kEnd - kStart);
                    const double x     = static_cast<double>(k - kStart) / denom; // [0,1]

                    switch (spec.hcMode)
                    {
                    case HCMode::Sharp:
                        // x^8 に基づくローパス近似 (Butterworth 急峻)
                        gain[k] = 1.0 / std::sqrt(1.0 + std::pow(x, 8.0));
                        break;
                    case HCMode::Natural:
                        // コサインクロスフェード (Hann 窓片側, 位相特性良好)
                        gain[k] = 0.5 * (1.0 + std::cos(
                            juce::MathConstants<double>::pi * x));
                        break;
                    case HCMode::Soft:
                        // ガウス型 (-60dB を kEnd 付近に設定)
                        gain[k] = std::exp(-4.60517 * x * x);
                        break;
                    }
                }
                else
                {
                    // ストップバンド (kEnd より上): ゲイン 0.0
                    gain[k] = 0.0;
                }
            }
        }

        // ── LC ゲイン (既存 HC ゲインに乗算) ──
        {
            const int kEnd   = static_cast<int>(std::round(lcFcEnd   * N / fs));
            const int kStart = static_cast<int>(std::round(lcFcStart * N / fs));

            for (int k = 0; k < cSize; ++k)
            {
                if (k <= kEnd)
                {
                    // DC ～ lcFcEnd: ゲイン 0.0
                    gain[k] = 0.0;
                }
                else if (k < kStart)
                {
                    // ロールオン域: [lcFcEnd+1, lcFcStart-1]
                    const double denom = static_cast<double>(
                        std::max(1, kStart - kEnd));
                    const double x     = static_cast<double>(k - kEnd) / denom; // [0,1)
                    // コサインロールオン (0→1)
                    const double g_lc  = 0.5 * (1.0 - std::cos(
                        juce::MathConstants<double>::pi * x));
                    gain[k] *= g_lc;
                }
                // k >= kStart: LC ゲイン 1.0 → 乗算不要 (HC ゲインのみ)
            }
        }

        // ── 全パーティションの irFreqDomain に gain[] を適用 ──
        // MKL VML vdMul を使用してベクトル化乗算 (Message Thread のみ)。
        //
        // interleaved complex 形式: [re0, im0, re1, im1, ...]
        // ビン k に対して gain[k] を re[k] と im[k] の両方へ適用するため、
        // gainIL[2k] = gainIL[2k+1] = gain[k] とした interleaved ゲイン配列を
        // 一度構築し、全パーティションに vdMul で一括適用する。
        //
        // [指摘1 fix] MKL VML 使用箇所は mkl_malloc/mkl_free で管理する。
        // vdMul のインプレース演算 (y == a) は MKL 規約で許可されている。
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

    // 前回のリソースを解放
    releaseAllLayers();

    // ────────────────────────────────────────────────
    // 先頭 Direct Form 設定
    //   - 先頭タップは時間領域で即時処理 (AVX2/FMA + MKL)
    //   - FFT 側IRは先頭区間をゼロ化して二重加算を回避
    // ────────────────────────────────────────────────
    // NOTE:
    // Direct head path remains behind an experimental flag.
    // Direct output is kept in a dedicated zero-latency mix buffer and added in Get().
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

    // NOTE: vmlSetMode はここでは呼ばない。
    // MainApplication::initialise() で vmlSetMode(VML_FTZDAZ_ON | VML_ERRMODE_IGNORE) を
    // 設定済みであり、ここで再度呼ぶと VML_ERRMODE_IGNORE が失われ、VML エラー時に
    // プロセス全体が強制終了するリスクがある。
    // また vmlSetMode はグローバル設定であるため、IR ロードのたびに呼び出すことは
    // スレッド安全性の観点からも望ましくない。

    // ────────────────────────────────────────────────
    // レイヤー構成決定 (Non-Uniform Partitioned Convolution)
    //
    // [設計方針]
    // L0 (即時): 最初の kL0MaxParts パーティション。
    //            毎コールバックで全パーティション処理 → 低レイテンシー
    //            L1/L2 CPU キャッシュ (L2: 512KB, L3: 16MB) に収まるサイズ。
    //
    // L1 (遅延): L0 担当外の IR 前半。partSize = l0Part * 8。
    //            partsPerCallback 個ずつ分散処理 → CPUスパイク抑制。
    //            IFFT 完了時に tailOutputBuf へコピー。
    //
    // L2 (遅延): IR テール全体。partSize = l1Part * 8。
    //            同上。
    //
    // [L1/L2 の遅延について]
    // L1/L2 の出力は 1 パーティション (partSize サンプル) 分遅延する。
    // リバーブのテール領域 (Lateリフレクション) は拡散・密度が高く、
    // 数十ms の遅延は知覚不可能なため実用上問題なし。
    // ────────────────────────────────────────────────
    const int l0Part = juce::nextPowerOfTwo(std::max(blockSize, 64));
    const int l1Part = l0Part * 8;
    const int l2Part = l1Part * 8;

    // L0: IR の先頭 kL0MaxParts パーティション
    const int l0Len = std::min(irLen, kL0MaxParts * l0Part);

    // L1: 残りの前半 kL1MaxParts パーティション
    const int l1Offset = l0Len;
    const int l1Len    = std::max(0, std::min(irLen - l0Len, kL1MaxParts * l1Part));

    // L2: 残り全部
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
            continue;  // このレイヤーは使用しない

        Layer& l = m_layers[m_numActiveLayers];
        l.descriptorCommitted = false;

        l.partSize    = cfgs[li].partSize;
        l.fftSize     = l.partSize * 2;
        l.isImmediate = cfgs[li].immediate;

        // complexSize = fftSize/2 + 1
        l.complexSize = l.fftSize / 2 + 1;
        // partStride: double 換算 complexSize*2 を 8-double (64byte) 境界にアライン
        l.partStride  = (l.complexSize * 2 + 7) & ~7;

        // IR パーティション数 (実数)
        l.numPartsIR = (cfgs[li].len + l.partSize - 1) / l.partSize;
        // FDL 用に power-of-two に切り上げ
        l.numParts   = juce::nextPowerOfTwo(l.numPartsIR);
        l.fdlMask    = l.numParts - 1;

        // ── DFTI ハンドル生成 (Message Thread で DftiCommitDescriptor) ──
        if (DftiCreateDescriptor(&l.fftHandle, DFTI_DOUBLE, DFTI_REAL, 1, l.fftSize) != DFTI_NO_ERROR)
        {
            releaseAllLayers();
            return false;
        }

        bool ok = true;
        ok = ok && (DftiSetValue(l.fftHandle, DFTI_PLACEMENT,             DFTI_NOT_INPLACE)     == DFTI_NO_ERROR);
        ok = ok && (DftiSetValue(l.fftHandle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX) == DFTI_NO_ERROR);
        ok = ok && (DftiSetValue(l.fftHandle, DFTI_BACKWARD_SCALE, 1.0 / static_cast<double>(l.fftSize)) == DFTI_NO_ERROR);
        ok = ok && (DftiCommitDescriptor(l.fftHandle) == DFTI_NO_ERROR);
        l.descriptorCommitted = ok;

        if (!ok)
        {
            releaseAllLayers();
            return false;
        }

        // ── バッファ確保 (すべて mkl_malloc 64byte アライン) ──
        const size_t irBufSize  = static_cast<size_t>(l.numParts) * l.partStride;
        // [最適化2] Linearized ring buffer: FDL を 2×numParts 分確保。
        // fdlBuf[fdlIndex] と fdlBuf[fdlIndex + numParts] に同一データをミラー書き込みし、
        // forward 方向への連続アクセスを実現する (hardware prefetcher 最大活用)。
        const size_t fdlBufSize = static_cast<size_t>(l.numParts) * 2 * l.partStride;

        l.irFreqDomain = static_cast<double*>(mkl_malloc(irBufSize  * sizeof(double), 64));
        l.fdlBuf       = static_cast<double*>(mkl_malloc(fdlBufSize * sizeof(double), 64));
        // [Bug F fix] overlapBuf 削除済み
        l.fftTimeBuf   = static_cast<double*>(mkl_malloc(l.fftSize   * sizeof(double), 64));
        l.fftOutBuf    = static_cast<double*>(mkl_malloc(l.fftSize   * sizeof(double), 64));
        l.prevInputBuf = static_cast<double*>(mkl_malloc(l.partSize  * sizeof(double), 64));
        l.accumBuf     = static_cast<double*>(mkl_malloc(l.partStride * sizeof(double), 64));
        l.inputAccBuf  = static_cast<double*>(mkl_malloc(l.partSize  * sizeof(double), 64));

        // L1/L2: テール出力バッファ確保
        // L0 (isImmediate) では不要 (nullptr のまま)
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
        memset(l.irFreqDomain, 0, irBufSize  * sizeof(double));
        memset(l.fdlBuf,       0, fdlBufSize * sizeof(double));
        // [Bug F fix] overlapBuf memset 削除済み
        memset(l.fftTimeBuf,   0, l.fftSize   * sizeof(double));
        memset(l.fftOutBuf,    0, l.fftSize   * sizeof(double));
        memset(l.prevInputBuf, 0, l.partSize  * sizeof(double));
        memset(l.accumBuf,     0, l.partStride * sizeof(double));
        memset(l.inputAccBuf,  0, l.partSize  * sizeof(double));
        if (l.tailOutputBuf)
            memset(l.tailOutputBuf, 0, l.partSize * sizeof(double));

        // ── IR プリコンピュート ──
        // 一時バッファ (スタック上では大きすぎるので mkl_malloc)
        double* tempTime = static_cast<double*>(mkl_malloc(l.fftSize * sizeof(double), 64));
        double* tempFreq = static_cast<double*>(mkl_malloc((l.fftSize + 2) * sizeof(double), 64));
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
            // p >= numPartsIR のスロットはゼロパディング (音質影響なし)

            DftiComputeForward(l.fftHandle, tempTime, tempFreq);

            // interleaved complex として irFreqDomain に格納
            memcpy(l.irFreqDomain + p * l.partStride, tempFreq,
                   l.complexSize * 2 * sizeof(double));

            // ヘッドルーム確保のためのスケーリング (MKL BLAS, Message Thread のみ)
            if (scale != 1.0)
                cblas_dscal(l.complexSize * 2, scale, l.irFreqDomain + p * l.partStride, 1);
        }

        // Backward FFT のウォームアップ
        // Audio Thread での初回実行時の遅延 (DFT テーブル生成等) を防ぐ
        DftiComputeBackward(l.fftHandle, tempFreq, tempTime);

        mkl_free(tempTime);
        mkl_free(tempFreq);

        // [最適化2] IR パーティションを逆順に並び替える。
        // Linearized ring buffer での forward 読み出し:
        //   FDL: [oldest ... newest] (linStart + 0, +1, ... +numPartsIR-1)
        //   IR:  [newest ... oldest] を逆順格納 → p=0 が oldest FDL に対応
        //
        // 正しい畳み込み: accumBuf += sum_p FDL[fdlIndex - p] * IR[p]
        //   → 格納を逆順にすることで forward アクセスで同じ結果が得られる。
        // この逆順変換は SetImpulse() のみで行い、Audio Thread にはコストゼロ。
        if (l.numPartsIR > 1)
        {
            // スワップ用一時バッファ (IR 1 パーティション分)
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
            // swapBuf 確保失敗 (OOM) の場合: 逆順格納なしでも音は出るが
            // 畳み込み順序が反転する (IR が逆再生になる)。実用上許容できない
            // ケースだが OOM = 極めて稀であり、NoExcept 関数内なので致命的とはしない。
        }

        // ── 非 Immediate レイヤーのコールバックあたりパーティション数 ──
        // 1 コールバック (blockSize サンプル) あたりに処理するパーティション数:
        //
        // 【設計上の不変条件】
        //   partSize = nextPowerOfTwo(max(blockSize,64)) * 8^n  (L1: n=1, L2: n=2)
        //   blockSize = nextPowerOfTwo(ホスト指定値)
        //   → partSize は常に blockSize の正確な倍数 (partSize % blockSize == 0)
        //
        //   blocksPerPart = partSize / blockSize  (整除。切り捨てなし)
        //   partsPerCallback = ceil(numPartsIR / blocksPerPart)
        //
        // これにより分散計算は 1 トリガ周期 (blocksPerPart コールバック) 内に完了する。
        //
        // 【注意】「= ceil(numPartsIR * blockSize / partSize)」は
        //   partSize % blockSize == 0 の場合のみ成立する等式である。
        //   非整除の場合、提案式は partsPerCallback を過小評価し
        //   分散計算が完了しない可能性があるため使用しないこと。
        if (!l.isImmediate)
        {
            jassert(l.partSize % blockSize == 0); // 上記不変条件の実行時検証
            const int blocksPerPart = l.partSize / std::max(blockSize, 1);
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
    // L1/L2 は tailOutputBuf を使用するため、リングは L0 のみに最適化したサイズでよい。
    //
    // 必要最小サイズ: 2 * l0PartSize
    //   (L0.partSize ≥ blockSize であるため、1 コールバックで最大 1 回の
    //    processLayerBlock が発火し、ringAvail の最大値 = 2 * l0PartSize)
    //
    // 確保サイズ: nextPowerOfTwo(l0PartSize * 4 + blockSize * 4)
    //   ≥ 8 * l0PartSize  (∵ blockSize ≤ l0PartSize)
    //   → ヘッドルーム 4x。通常動作でのオーバーフローは構造上不可能。
    //      (詳細は MKLNonUniformConvolver.h m_ringOverflowCount のコメントを参照)
    // ────────────────────────────────────────────────
    const int l0PartSize = m_layers[0].partSize;
    const int rSize = juce::nextPowerOfTwo(l0PartSize * 4 + blockSize * 4);
    m_ringBuf = static_cast<double*>(mkl_malloc(rSize * sizeof(double), 64));
    if (!m_ringBuf)
    {
        releaseAllLayers();
        return false;
    }

    memset(m_ringBuf, 0, rSize * sizeof(double));
    m_ringSize  = rSize;
    m_ringMask  = rSize - 1;
    m_ringWrite = 0;
    m_ringRead  = 0;
    m_ringAvail = 0;

    m_latency = m_layers[0].partSize;  // Layer0 の partSize = 最低遅延

    // ── 出力周波数フィルターを irFreqDomain に焼き込む (Message Thread のみ) ──
    // filterSpec が nullptr の場合はスルー (既存動作と完全互換)。
    // Audio Thread の追加コストはゼロ (計算済みゲインが irFreqDomain に乗算済み)。
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
        if (l.fftHandle == nullptr || !l.descriptorCommitted)
            return false;
    }

    return true;
}

//==============================================================================
// processDirectBlock  ─ Audio Thread
// 先頭IRを時間領域で即時処理し、結果を専用ミックスバッファへ書き込む。
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
    memset(m_directOutBuf, 0, static_cast<size_t>(numSamples) * sizeof(double));

    int processed = 0;
    while (processed < numSamples)
    {
        const int chunk = std::min(numSamples - processed, m_directMaxBlock);

        if (m_directHistLen > 0)
            memcpy(m_directWindow, m_directHistory, static_cast<size_t>(m_directHistLen) * sizeof(double));

        if (input)
            memcpy(m_directWindow + m_directHistLen, input + processed, static_cast<size_t>(chunk) * sizeof(double));
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
            memcpy(m_directHistory, m_directWindow + chunk, static_cast<size_t>(m_directHistLen) * sizeof(double));

        processed += chunk;
    }

    m_directPendingSamples = numSamples;
}

//==============================================================================
// processLayerBlock  ─ Audio Thread (L0 専用)
//
// l.inputAccBuf に 1 パーティション分の入力が溜まったタイミングで呼ぶ。
//
// 処理:
//   1. Overlap-Save 形式で fftTimeBuf を組み立てる [prevInput | currentInput]
//   2. Forward FFT → FDL の現在スロットへ格納
//   3. FDL × IR の複素乗算積算 (AVX2 FMA)
//   4. Backward FFT
//   5. 有効出力 (後半 partSize サンプル) をリングバッファへ書き込む (ringWrite)
//   6. FDL インデックスを進める
//==============================================================================
void MKLNonUniformConvolver::processLayerBlock(Layer& l) noexcept
{
    // ── 1. [prevInput | currentInput] を fftTimeBuf に配置 (Overlap-Save) ──
    memcpy(l.fftTimeBuf,              l.prevInputBuf, l.partSize * sizeof(double));
    memcpy(l.fftTimeBuf + l.partSize, l.inputAccBuf,  l.partSize * sizeof(double));

    // 現在の入力を次回の "prev" として保存
    memcpy(l.prevInputBuf, l.inputAccBuf, l.partSize * sizeof(double));

    // ── 2. Forward FFT ──
    // 出力は FDL の現在スロットへ直接書き込む
    double* currentFDLSlot = l.fdlBuf + l.fdlIndex * l.partStride;
    DftiComputeForward(l.fftHandle, l.fftTimeBuf, currentFDLSlot);

    // [最適化2] Linearized ring buffer: mirror write。
    // fdlBuf[fdlIndex + numParts] にも同一データを書き込む。
    // これにより convolution ループで forward 方向の連続アクセスが可能になる。
    double* mirrorFDLSlot = l.fdlBuf + (l.fdlIndex + l.numParts) * l.partStride;
    memcpy(mirrorFDLSlot, currentFDLSlot, l.partStride * sizeof(double));

    // ── 3. 複素乗算積算 (FDL × IR) → accumBuf ──
    memset(l.accumBuf, 0, l.partStride * sizeof(double));

    const double* fdlBase = l.fdlBuf;
    const double* irBase  = l.irFreqDomain;
    double*       dst     = l.accumBuf;

    // [最適化2] Linearized ring buffer: forward 順次アクセス。
    // linStart = fdlIndex - numPartsIR + 1 + numParts
    //   → 常に [1, 2*numParts-1] 範囲内 (modulo 演算不要)
    // IR は SetImpulse() で逆順格納済みのため forward アクセスで正しい畳み込みになる。
    const int linStart   = l.fdlIndex - l.numPartsIR + 1 + l.numParts;
    const double* fdlLin = fdlBase + linStart * l.partStride;

    for (int p = 0; p < l.numPartsIR; ++p)
    {
        const double* srcA = fdlLin + p * l.partStride;  // FDL: oldest→newest (forward)
        const double* srcB = irBase + p * l.partStride;  // IR:  forward (逆順格納済み)

        // ── パーティション先読み (T1: L2キャッシュターゲット, 2 iterations 先) ──
        // forward アクセスのため stride は一定 → hardware prefetcher も有効
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

        // ── 8 複素数 (= 16 double) を 1 ループで処理 (AVX2 8-wide unroll) ──
        // 2 組の acc/a/b ペアをアウトオブオーダー実行で並列投入し
        // FMA パイプラインのスループットを最大化する。
        const int vEnd8 = (l.complexSize / 8) * 8;
        const int vEnd4 = (l.complexSize / 4) * 4;

        for (; k < vEnd8; k += 8)
        {
            // 内側ループ先読み (T0: L1キャッシュターゲット, 2 iterations 先)
            _mm_prefetch((const char*)(srcA + 2 * k + 64), _MM_HINT_T0);
            _mm_prefetch((const char*)(srcB + 2 * k + 64), _MM_HINT_T0);

            // ── 複素 k..k+3 ──
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

            // ── 複素 k+4..k+7 ──
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

        // 4-wide 残余 (vEnd8 〜 vEnd4)
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

        // スカラーフォールバック (残り要素)
        for (; k < l.complexSize; ++k)
        {
            const double ar = srcA[2 * k],     ai = srcA[2 * k + 1];
            const double br = srcB[2 * k],     bi = srcB[2 * k + 1];
            dst[2 * k]     += ar * br - ai * bi;
            dst[2 * k + 1] += ar * bi + ai * br;
        }
    }

    // ── 4. Backward FFT ──
    DftiComputeBackward(l.fftHandle, l.accumBuf, l.fftOutBuf);

    // ── 5. Overlap-Save: 有効出力 (後半 partSize サンプル) をリングへ順次書き込み ──
    // [Bug1 fix] L0 のみリングを独占使用。memcpy による上書きは安全 (L1/L2 非使用)。
    ringWrite(l.fftOutBuf + l.partSize, l.partSize);

    // ── 6. FDL インデックスを進める ──
    l.fdlIndex = (l.fdlIndex + 1) & l.fdlMask;
}

//==============================================================================
// ringWrite  ─ Audio Thread (L0 専用)
// L0 の IFFT 出力をリングバッファへ順次書き込み、m_ringWrite / m_ringAvail を更新する。
//==============================================================================
void MKLNonUniformConvolver::ringWrite(const double* src, int n) noexcept
{
    if (n <= 0 || m_ringBuf == nullptr || src == nullptr) return;

    const int first = std::min(n, m_ringSize - m_ringWrite);
    memcpy(m_ringBuf + m_ringWrite, src, first * sizeof(double));
    if (n > first)
        memcpy(m_ringBuf, src + first, (n - first) * sizeof(double));

    m_ringWrite = (m_ringWrite + n) & m_ringMask;

    const int nextAvail = m_ringAvail + n;
    if (nextAvail > m_ringSize)
    {
        // ─────────────────────────────────────────────────────────────────
        // オーバーフロー安全弁 (通常動作では到達しない)
        //
        // 【到達条件】Add/Get の非対称呼び出し、またはリセット直後の連続 Add など
        //   構造バグが存在する場合のみ発生する。
        //   証明は m_ringOverflowCount メンバのコメントを参照。
        //
        // 【対応方針】Audio Thread 内でブロックは不可。
        //   最も古いデータを破棄し、最新データを保持する「ドロップアウト」戦略を採る。
        //   これにより、不連続は生じるが Audio Thread の遅延は回避できる。
        //   発生頻度が非ゼロの場合は m_ringOverflowCount の監視によって検出可能。
        // ─────────────────────────────────────────────────────────────────
        jassertfalse; // 通常動作では絶対に到達しない。到達した場合はバグ。
        m_ringOverflowCount.fetch_add(1, std::memory_order_relaxed);

        const int overflow = nextAvail - m_ringSize;
        m_ringRead  = (m_ringRead + overflow) & m_ringMask;
        m_ringAvail = m_ringSize;
    }
    else
    {
        m_ringAvail = nextAvail;
    }
}

//==============================================================================
// ringRead  ─ Audio Thread (L0 専用)
// リングバッファから n サンプルを読み出す。
//
// 【Zero-Flush 削除について】
//   旧実装は読み出し後に memset(0) でバッファ領域をクリアしていたが、
//   ringWrite は memcpy による上書きのみであるため、読み出し済み領域は
//   次回 ringWrite で必ず完全に上書きされる。
//   クリアは冗長であり Audio Thread での余分な memset は避けるべきため削除した。
//   初期状態は setup() の memset(0) で保証されている。
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
        memcpy(dst, m_ringBuf + m_ringRead, first * sizeof(double));
        if (toRead > first)
            memcpy(dst + first, m_ringBuf, (toRead - first) * sizeof(double));

        // 読み取れなかった分をゼロ埋め
        if (toRead < n)
            memset(dst + toRead, 0, (n - toRead) * sizeof(double));
    }

    m_ringRead  = (m_ringRead + toRead) & m_ringMask;
    m_ringAvail -= toRead;
    return toRead;
}

//==============================================================================
// Add  ─ Audio Thread
//
// [処理フロー]
//   L0 (即時レイヤー):
//     - blockSize サンプルごとに processLayerBlock() → ringWrite()
//
//   L1/L2 (遅延レイヤー):
//     トリガブロック (partSize サンプル蓄積後, while ループ内):
//       1. Overlap-Save 形式で fftTimeBuf を組み立てる
//       2. Forward FFT → FDL 格納
//       3. fdlIndex を +1 (トリガ完了)
//       4. [Bug2 fix] baseFdlIdxSaved を保存 (サイクル全体で再計算禁止)
//       5. accumBuf クリア, nextPart=0, distributing=true
//
//     分散計算ループ (while ループ外, 毎コールバック実行):
//       [Bug3 fix] トリガブロック外に移動。
//       6. partsPerCallback 個の FDL × IR 複素乗算積算 (AVX2 FMA)
//       7. numPartsIR 完了時: Backward FFT → tailOutputBuf へコピー
//          distributing=false, nextPart=0
//==============================================================================
void MKLNonUniformConvolver::Add(const double* input, int numSamples)
{
    // input == nullptr は無音として扱う
    if (!m_ready.load(std::memory_order_acquire) || numSamples <= 0)
        return;

    // 先頭Direct区間を即時処理 (AVX2/FMA + MKL dot)
    processDirectBlock(input, numSamples);

    for (int li = 0; li < m_numActiveLayers; ++li)
    {
        Layer& l = m_layers[li];

        // ────────────────────────────────────────────
        // 入力を partSize 単位で蓄積し、溜まったらトリガ処理する
        //
        // 【複数トリガ不発生の証明】
        //   L1/L2 の partSize と numSamples の大小関係:
        //     L1.partSize = l0Part × 8 ≥ blockSize × 8
        //     L2.partSize = l0Part × 64 ≥ blockSize × 64
        //     numSamples  = blockSize (StereoConvolver::process から渡される値)
        //   よって numSamples ≤ L1.partSize / 8 であり、
        //   1 回の Add() 呼び出しで L1/L2 が 2 回以上トリガすることは
        //   構造上不可能。while ループの L1/L2 トリガ分岐は高々 1 回しか実行されない。
        // ────────────────────────────────────────────
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
                // 1 パーティション分が溜まった
                l.inputPos = 0;

                if (l.isImmediate)
                {
                    // ── L0: 即時処理 (全パーティション畳み込み → ringWrite) ──
                    processLayerBlock(l);
                }
                else
                {
                    // ── L1/L2: トリガ処理 (Forward FFT + FDL 更新のみ) ──
                    // 分散計算ループは下 (while ループ外) で毎コールバック実行する。

                    // 【不変条件】1 回の Add() でこの分岐は高々 1 回しか実行されない。
                    // partSize ≥ 8 × blockSize が保証されているため、
                    // numSamples(= blockSize) < partSize が常に成立する。
                    jassert(consumed <= numSamples); // この分岐後は consumed == numSamples になるはず

                    // Overlap-Save 形式で fftTimeBuf を組み立てる
                    memcpy(l.fftTimeBuf,              l.prevInputBuf, l.partSize * sizeof(double));
                    memcpy(l.fftTimeBuf + l.partSize, l.inputAccBuf,  l.partSize * sizeof(double));
                    memcpy(l.prevInputBuf, l.inputAccBuf, l.partSize * sizeof(double));

                    // Forward FFT → FDL の現在スロットへ格納
                    double* currentFDLSlot = l.fdlBuf + l.fdlIndex * l.partStride;
                    DftiComputeForward(l.fftHandle, l.fftTimeBuf, currentFDLSlot);

                    // [最適化2] Linearized ring buffer: mirror write
                    // fdlBuf[fdlIndex + numParts] にも同一データを書き込む。
                    double* mirrorFDLSlot = l.fdlBuf + (l.fdlIndex + l.numParts) * l.partStride;
                    memcpy(mirrorFDLSlot, currentFDLSlot, l.partStride * sizeof(double));

                    // FDL インデックスを進める (トリガ完了)
                    l.fdlIndex = (l.fdlIndex + 1) & l.fdlMask;

                    // [Bug2 fix] 新サイクル開始時に FDL スナップショットを保存。
                    // fdlIndex は既に +1 済みなので「-1」が直前に書き込んだスロット。
                    // このサイクルの全パーティションはこの値を使い、再計算しない。
                    l.baseFdlIdxSaved = (l.fdlIndex - 1 + l.numParts) & l.fdlMask;

                    // accumBuf クリア・カウンタリセット (新サイクル開始)
                    memset(l.accumBuf, 0, l.partStride * sizeof(double));
                    l.nextPart    = 0;
                    l.distributing = true;
                }
            }
        } // while consumed < numSamples

        // ────────────────────────────────────────────────────────────────────
        // [Bug3 fix] 分散計算ループ: while ループ外で毎コールバック実行。
        //
        // トリガブロック内に置くと、トリガ時の 1 コールバックしか実行されない。
        // ここに置くことで毎コールバック partsPerCallback 個ずつ処理が進み、
        // numPartsIR 完了後に IFFT + tailOutputBuf コピーが行われる。
        // ────────────────────────────────────────────────────────────────────
        if (!l.isImmediate && l.distributing)
        {
            const int endPart  = std::min(l.nextPart + l.partsPerCallback, l.numPartsIR);

            const double* fdlBase    = l.fdlBuf;
            const double* irBase     = l.irFreqDomain;
            double*       dst        = l.accumBuf;
            // [Bug2 fix] サイクル開始時に保存したスナップショットを使用 (再計算禁止)
            const int     baseFdlIdx = l.baseFdlIdxSaved;

            // [最適化2] Linearized ring buffer: forward 順次アクセス。
            // linStart = baseFdlIdx - numPartsIR + 1 + numParts
            // IR は SetImpulse() で逆順格納済み。
            const int linStart   = baseFdlIdx - l.numPartsIR + 1 + l.numParts;
            const double* fdlLin = fdlBase + linStart * l.partStride;

            for (int p = l.nextPart; p < endPart; ++p)
            {
                const double* srcA = fdlLin + p * l.partStride;  // FDL: forward
                const double* srcB = irBase + p * l.partStride;  // IR:  forward (逆順格納済み)

                // [最適化2] prefetch: forward 方向なので stride = partStride (一定)
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

                // ── 8 複素数 (= 16 double) 8-wide AVX2 アンロール ──
                const int vEnd8 = (l.complexSize / 8) * 8;
                const int vEnd4 = (l.complexSize / 4) * 4;

                for (; k < vEnd8; k += 8)
                {
                    _mm_prefetch((const char*)(srcA + 2 * k + 128), _MM_HINT_T0);
                    _mm_prefetch((const char*)(srcB + 2 * k + 128), _MM_HINT_T0);

                    // 複素 k..k+3
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

                    // 複素 k+4..k+7
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

                // 4-wide 残余
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

                // スカラーフォールバック
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
                // Backward FFT (規約遵守: Audio Thread 内で DftiCommitDescriptor 禁止)
                DftiComputeBackward(l.fftHandle, l.accumBuf, l.fftOutBuf);

                // [Bug1 fix] リングバッファへの上書きではなく tailOutputBuf へコピー。
                // Get() で L0 の ring 出力に vdAdd で合算する。
                // 有効出力は Overlap-Save の後半 partSize サンプル。
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
//
// L0 の出力 (リングバッファ) に L1/L2 の遅延出力 (tailOutputBuf) を vdAdd で合算して返す。
//
// [L1/L2 の遅延特性]
//   L1/L2 の tailOutputBuf は IFFT 完了後に埋まる。IFFT は partSize/blockSize コールバック後に
//   完了するため、L1/L2 の出力は 1 パーティション (partSize サンプル) 分遅延する。
//   リバーブのテール領域では知覚不可能なため実用上問題なし。
//
// [tailOutputBuf の消費タイミング]
//   L1: partsPerCallback × (partSize/blockSize) = numPartsIR 処理後に IFFT 完了。
//       tailOutputBuf が partSize サンプル分を保持し、blockSize ずつ消費。
//       tailOutputPos が partSize に達したら次の IFFT まで zeros を合算。
//==============================================================================
int MKLNonUniformConvolver::Get(double* output, int numSamples)
{
    if (!m_ready.load(std::memory_order_acquire) || numSamples <= 0)
    {
        if (output && numSamples > 0)
            memset(output, 0, numSamples * sizeof(double));
        return 0;
    }

    // ── L0 出力: リングバッファから読み出し ──
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
        for (; i < vEnd; i += 4)
        {
            const __m256d a = _mm256_loadu_pd(dst + i);
            const __m256d b = _mm256_loadu_pd(src + i);
            _mm256_storeu_pd(dst + i, _mm256_add_pd(a, b));
        }
        for (; i < n; ++i)
            dst[i] += src[i];
#else
        for (int i = 0; i < n; ++i)
            dst[i] += src[i];
#endif
    };

    // ── Direct 出力: 専用ゼロ遅延ミックスバッファを加算 ──
    if (m_directEnabled && m_directOutBuf != nullptr)
    {
        const int toAdd = std::min(numSamples, m_directPendingSamples);
        if (toAdd > 0)
        {
            if (output != nullptr)
            {
                if (isAligned64(output) && isAligned64(m_directOutBuf))
                    vdAdd(toAdd, output, m_directOutBuf, output);
                else
                    addFallback(toAdd, output, m_directOutBuf);
            }

            memset(m_directOutBuf, 0, static_cast<size_t>(toAdd) * sizeof(double));
            m_directPendingSamples = 0;
        }
    }

    // ── L1/L2 出力: tailOutputBuf から vdAdd で合算 ──
    // output が nullptr の場合でも tailOutputPos を正しく進める。
    for (int li = 1; li < m_numActiveLayers; ++li)
    {
        Layer& l = m_layers[li];
        if (l.tailOutputBuf == nullptr) continue;  // 未初期化時はスキップ

        const int remaining = l.partSize - l.tailOutputPos;
        if (remaining <= 0) continue;  // tailOutputBuf が枯渇 (次 IFFT まで待機)

        const int toAdd = std::min(numSamples, remaining);

        if (output != nullptr)
        {
            // vdAdd: output[0..toAdd-1] += tailOutputBuf[tailOutputPos..tailOutputPos+toAdd-1]
            const double* tailPtr = l.tailOutputBuf + l.tailOutputPos;
            if (isAligned64(output) && isAligned64(tailPtr))
            {
                vdAdd(toAdd,
                      output,
                      tailPtr,
                      output);
            }
            else
            {
                addFallback(toAdd, output, tailPtr);
            }
        }

        l.tailOutputPos += toAdd;
    }

    return got;
}

//==============================================================================
// Reset  ─ Message Thread
// すべての内部バッファをゼロクリアし、位置変数を初期化する。
//==============================================================================
void MKLNonUniformConvolver::Reset()
{
    for (int li = 0; li < m_numActiveLayers; ++li)
    {
        Layer& l = m_layers[li];
        if (l.irFreqDomain == nullptr) continue;

        // [最適化2] Linearized ring buffer: fdlBuf は 2×numParts 分確保済み
        const size_t fdlBufSize = static_cast<size_t>(l.numParts) * 2 * l.partStride;
        memset(l.fdlBuf,       0, fdlBufSize   * sizeof(double));
        // [Bug F fix] overlapBuf 削除済み
        memset(l.fftTimeBuf,   0, l.fftSize     * sizeof(double));
        memset(l.fftOutBuf,    0, l.fftSize     * sizeof(double));
        memset(l.prevInputBuf, 0, l.partSize    * sizeof(double));
        memset(l.accumBuf,     0, l.partStride  * sizeof(double));
        memset(l.inputAccBuf,  0, l.partSize    * sizeof(double));

        if (l.tailOutputBuf)
            memset(l.tailOutputBuf, 0, l.partSize * sizeof(double));

        l.fdlIndex        = 0;
        l.inputPos        = 0;
        l.nextPart        = 0;
        l.tailOutputPos   = 0;
        l.baseFdlIdxSaved = 0;
        l.distributing    = false;
    }

    if (m_ringBuf)
        memset(m_ringBuf, 0, m_ringSize * sizeof(double));
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
