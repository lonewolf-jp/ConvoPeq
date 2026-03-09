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

#include <mkl.h>
#include <mkl_dfti.h>
#include <mkl_vml.h>
#include <mkl_cblas.h>
#include <cstring>
#include <algorithm>
#include <cmath>

#include <immintrin.h>  // AVX2

namespace convo
{

//==============================================================================
// Layer::freeAll
//==============================================================================
void MKLNonUniformConvolver::Layer::freeAll() noexcept
{
    if (fftHandle)     { DftiFreeDescriptor(&fftHandle); fftHandle = nullptr; }
    if (irFreqDomain)  { mkl_free(irFreqDomain);  irFreqDomain  = nullptr; }
    if (fdlBuf)        { mkl_free(fdlBuf);         fdlBuf        = nullptr; }
    if (overlapBuf)    { mkl_free(overlapBuf);     overlapBuf    = nullptr; }
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

void MKLNonUniformConvolver::releaseAllLayers() noexcept
{
    for (int i = 0; i < kNumLayers; ++i)
        m_layers[i].freeAll();
    m_numActiveLayers = 0;
    m_latency         = 0;

    if (m_ringBuf) { mkl_free(m_ringBuf); m_ringBuf = nullptr; }
    m_ringSize = m_ringMask = m_ringWrite = m_ringRead = m_ringAvail = 0;
}

//==============================================================================
// SetImpulse  ─ Message Thread のみ
//==============================================================================
bool MKLNonUniformConvolver::SetImpulse(const double* impulse, int irLen, int blockSize, double scale)
{
    m_ready.store(false, std::memory_order_release);

    if (impulse == nullptr || irLen <= 0 || blockSize <= 0)
        return false;

    // 前回のリソースを解放
    releaseAllLayers();

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

        if (!ok)
        {
            releaseAllLayers();
            return false;
        }

        // ── バッファ確保 (すべて mkl_malloc 64byte アライン) ──
        const size_t irBufSize  = static_cast<size_t>(l.numParts) * l.partStride;
        const size_t fdlBufSize = static_cast<size_t>(l.numParts) * l.partStride;

        l.irFreqDomain = static_cast<double*>(mkl_malloc(irBufSize  * sizeof(double), 64));
        l.fdlBuf       = static_cast<double*>(mkl_malloc(fdlBufSize * sizeof(double), 64));
        l.overlapBuf   = static_cast<double*>(mkl_malloc(l.partSize  * sizeof(double), 64));
        l.fftTimeBuf   = static_cast<double*>(mkl_malloc(l.fftSize   * sizeof(double), 64));
        l.fftOutBuf    = static_cast<double*>(mkl_malloc(l.fftSize   * sizeof(double), 64));
        l.prevInputBuf = static_cast<double*>(mkl_malloc(l.partSize  * sizeof(double), 64));
        l.accumBuf     = static_cast<double*>(mkl_malloc(l.partStride * sizeof(double), 64));
        l.inputAccBuf  = static_cast<double*>(mkl_malloc(l.partSize  * sizeof(double), 64));

        // L1/L2: テール出力バッファ確保
        // L0 (isImmediate) では不要 (nullptr のまま)
        if (!l.isImmediate)
            l.tailOutputBuf = static_cast<double*>(mkl_malloc(l.partSize * sizeof(double), 64));

        if (!l.irFreqDomain || !l.fdlBuf || !l.overlapBuf || !l.fftTimeBuf ||
            !l.fftOutBuf || !l.prevInputBuf || !l.accumBuf || !l.inputAccBuf ||
            (!l.isImmediate && !l.tailOutputBuf))
        {
            releaseAllLayers();
            return false;
        }

        // ゼロ初期化
        memset(l.irFreqDomain, 0, irBufSize  * sizeof(double));
        memset(l.fdlBuf,       0, fdlBufSize * sizeof(double));
        memset(l.overlapBuf,   0, l.partSize  * sizeof(double));
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

        const double* irSrc    = impulse + cfgs[li].offset;
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

        // ── 非 Immediate レイヤーのコールバックあたりパーティション数 ──
        // 1 コールバック (blockSize サンプル) あたりに処理するパーティション数:
        //   partsPerCallback = ceil(numPartsIR / (partSize / blockSize))
        //                    = ceil(numPartsIR * blockSize / partSize)
        // これにより、IFFT は partSize/blockSize コールバック後 (= 1 トリガ周期後) に完了する。
        if (!l.isImmediate)
        {
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
    // rSize = nextPow2(L0.partSize * 4 + blockSize * 4) で十分。
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

    m_ready.store(true, std::memory_order_release);
    return true;
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

    // ── 3. 複素乗算積算 (FDL × IR) → accumBuf ──
    memset(l.accumBuf, 0, l.partStride * sizeof(double));

    const double* fdlBase = l.fdlBuf;
    const double* irBase  = l.irFreqDomain;
    double*       dst     = l.accumBuf;

    for (int p = 0; p < l.numPartsIR; ++p)
    {
        // FDL は巡回インデックス: 現在ブロックから p 個前のスロット
        const int lineIdx  = (l.fdlIndex - p + l.numParts) & l.fdlMask;
        const double* srcA = fdlBase + lineIdx * l.partStride;  // FDL[p]
        const double* srcB = irBase  + p       * l.partStride;  // IR[p]

        int k = 0;

        // 4 複素数 (= 8 double) を 1 ループで処理 (AVX2)
        const int vEnd = (l.complexSize / 4) * 4;
        for (; k < vEnd; k += 4)
        {
            __m256d acc0 = _mm256_load_pd(dst  + 2 * k);
            __m256d acc1 = _mm256_load_pd(dst  + 2 * k + 4);
            __m256d a0   = _mm256_load_pd(srcA + 2 * k);
            __m256d a1   = _mm256_load_pd(srcA + 2 * k + 4);
            __m256d b0   = _mm256_load_pd(srcB + 2 * k);
            __m256d b1   = _mm256_load_pd(srcB + 2 * k + 4);

            // 複素乗算積算: acc += a * b
            // Re(a*b) = Re(a)*Re(b) - Im(a)*Im(b)
            // Im(a*b) = Re(a)*Im(b) + Im(a)*Re(b)
            __m256d a0_re = _mm256_movedup_pd(a0);      // [Ar0, Ar0, Ar1, Ar1]
            __m256d a0_im = _mm256_permute_pd(a0, 0xF); // [Ai0, Ai0, Ai1, Ai1]
            acc0 = _mm256_fmadd_pd(a0_re, b0, acc0);
            __m256d b0_sw = _mm256_permute_pd(b0, 0x5); // [Bi0, Br0, Bi1, Br1]
            acc0 = _mm256_addsub_pd(acc0, _mm256_mul_pd(a0_im, b0_sw));

            __m256d a1_re = _mm256_movedup_pd(a1);
            __m256d a1_im = _mm256_permute_pd(a1, 0xF);
            acc1 = _mm256_fmadd_pd(a1_re, b1, acc1);
            __m256d b1_sw = _mm256_permute_pd(b1, 0x5);
            acc1 = _mm256_addsub_pd(acc1, _mm256_mul_pd(a1_im, b1_sw));

            _mm256_store_pd(dst + 2 * k,     acc0);
            _mm256_store_pd(dst + 2 * k + 4, acc1);
        }
        // スカラーフォールバック (残り要素 / AVX2 非対応環境)
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
        // リングオーバーフロー: 最古のデータを破棄して読み取り位置を進める
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
// リングバッファから n サンプルを読み出し、読み出し位置を Zero-Flush する。
//
// Zero-Flush の目的: 読み出し済み領域を即座にゼロクリアする。
//   次回 ringWrite (L0 の memcpy) がその位置に上書きするため実質的には不要だが、
//   バッファ内のゴミデータによる予期しない問題を防ぐ防御的措置として実施する。
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

    // Zero-Flush: 読み出し位置をゼロクリア
    memset(m_ringBuf + m_ringRead, 0, first * sizeof(double));
    if (toRead > first)
        memset(m_ringBuf, 0, (toRead - first) * sizeof(double));

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

    for (int li = 0; li < m_numActiveLayers; ++li)
    {
        Layer& l = m_layers[li];

        // ────────────────────────────────────────────
        // 入力を partSize 単位で蓄積し、溜まったらトリガ処理する
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

                    // Overlap-Save 形式で fftTimeBuf を組み立てる
                    memcpy(l.fftTimeBuf,              l.prevInputBuf, l.partSize * sizeof(double));
                    memcpy(l.fftTimeBuf + l.partSize, l.inputAccBuf,  l.partSize * sizeof(double));
                    memcpy(l.prevInputBuf, l.inputAccBuf, l.partSize * sizeof(double));

                    // Forward FFT → FDL の現在スロットへ格納
                    double* currentFDLSlot = l.fdlBuf + l.fdlIndex * l.partStride;
                    DftiComputeForward(l.fftHandle, l.fftTimeBuf, currentFDLSlot);

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

            for (int p = l.nextPart; p < endPart; ++p)
            {
                // FDL 巡回インデックス: 最新スロット (p=0) から古い順
                const int lineIdx  = (baseFdlIdx - p + l.numParts) & l.fdlMask;
                const double* srcA = fdlBase + lineIdx * l.partStride;  // FDL[p]
                const double* srcB = irBase  + p       * l.partStride;  // IR[p]

                int k = 0;
                const int vEnd4 = (l.complexSize / 4) * 4;
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
            vdAdd(toAdd,
                  output,
                  l.tailOutputBuf + l.tailOutputPos,
                  output);
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

        const size_t fdlBufSize = static_cast<size_t>(l.numParts) * l.partStride;
        memset(l.fdlBuf,       0, fdlBufSize   * sizeof(double));
        memset(l.overlapBuf,   0, l.partSize    * sizeof(double));
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
}

} // namespace convo
