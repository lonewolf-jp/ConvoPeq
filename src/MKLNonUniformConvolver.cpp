//============================================================================
// MKLNonUniformConvolver.cpp  ── v1.0 (JUCE 8.0.12 / Intel oneMKL 対応)
//
// Non-Uniform Partitioned Convolution の完全実装。
// 3 層レイヤー構造で低遅延と大 IR を両立する。
//============================================================================

#include "MKLNonUniformConvolver.h"

#include <mkl.h>
#include <mkl_dfti.h>
#include <mkl_vml.h>
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

    fftSize = partSize = numParts = numPartsIR = 0;
    fdlMask = complexSize = partStride = 0;
    fdlIndex = inputPos = partsPerCallback = nextPart = 0;
    isImmediate = false;
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

    // VML 高精度モード設定 (Message Thread で一度だけ)
    vmlSetMode(VML_HA | VML_FTZDAZ_ON);

    // ────────────────────────────────────────────────
    // レイヤー構成決定 (品質安定化修正版)
    // 旧3レイヤー(非即時レイヤー含む)は時間整合されていない出力が混入し、
    // IRロード後に原音由来のブツ切れ音を発生させるため、現状は単一レイヤーで運用する。
    // ※ NUC経路自体は維持し、今後は時間整合付き多層合成へ拡張可能。
    // Jitter対策: 最小パーティションサイズを256から64へ引き下げ。
    // 低レイテンシー設定時(64/128 samples)の蓄積→スパイク処理を防ぎ、コールバックごとの負荷を均一化する。
    // ────────────────────────────────────────────────
    const int l0Part = juce::nextPowerOfTwo(std::max(blockSize, 64));

    struct LayerCfg { int offset; int len; int partSize; bool immediate; };
    LayerCfg cfgs[kNumLayers];

    cfgs[0].offset    = 0;
    cfgs[0].len       = irLen;
    cfgs[0].partSize  = l0Part;
    cfgs[0].immediate = true;

    // L1/L2 は無効化 (時間整合付き実装まで)
    // TODO: 将来的に多層レイヤー化 (Non-Uniform Partitioned Convolution) を再有効化する際は、
    //       各レイヤーの出力タイミングを整合させるための遅延補正と、リングバッファへの加算合成 (accumulate)
    //       の実装が必要です。現在は品質安定化のため単一レイヤー (L0) のみを使用しています。
    cfgs[1].offset = cfgs[1].len = cfgs[1].partSize = 0;
    cfgs[1].immediate = false;
    cfgs[2].offset = cfgs[2].len = cfgs[2].partSize = 0;
    cfgs[2].immediate = false;

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
        l.complexSize  = l.fftSize / 2 + 1;
        // partStride: double 換算 complexSize*2 を 8-double (64byte) 境界にアライン
        l.partStride   = (l.complexSize * 2 + 7) & ~7;

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
        const size_t irBufSize  = static_cast<size_t>(l.numParts)  * l.partStride;
        const size_t fdlBufSize = static_cast<size_t>(l.numParts)  * l.partStride;

        l.irFreqDomain  = static_cast<double*>(mkl_malloc(irBufSize  * sizeof(double), 64));
        l.fdlBuf        = static_cast<double*>(mkl_malloc(fdlBufSize * sizeof(double), 64));
        l.overlapBuf    = static_cast<double*>(mkl_malloc(l.partSize  * sizeof(double), 64));
        l.fftTimeBuf    = static_cast<double*>(mkl_malloc(l.fftSize   * sizeof(double), 64));
        l.fftOutBuf     = static_cast<double*>(mkl_malloc(l.fftSize   * sizeof(double), 64));
        l.prevInputBuf  = static_cast<double*>(mkl_malloc(l.partSize  * sizeof(double), 64));
        l.accumBuf      = static_cast<double*>(mkl_malloc(l.partStride * sizeof(double), 64));
        l.inputAccBuf   = static_cast<double*>(mkl_malloc(l.partSize  * sizeof(double), 64));

        if (!l.irFreqDomain || !l.fdlBuf || !l.overlapBuf || !l.fftTimeBuf ||
            !l.fftOutBuf || !l.prevInputBuf || !l.accumBuf || !l.inputAccBuf)
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

        const double* irSrc   = impulse + cfgs[li].offset;
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

            // ヘッドルーム確保のためのスケーリング (MKL BLAS)
            if (scale != 1.0)
                cblas_dscal(l.complexSize * 2, scale, l.irFreqDomain + p * l.partStride, 1);
        }

        // 【追加】Backward FFT のウォームアップ
        // Audio Thread での初回実行時の遅延（テーブル生成やメモリ確保）を防ぐために
        // ここで一度実行しておく。
        DftiComputeBackward(l.fftHandle, tempFreq, tempTime);

        mkl_free(tempTime);
        mkl_free(tempFreq);

        // ── 非 Immediate レイヤーのコールバックあたりパーティション数 ──
        // 1 ブロック (blockSize サンプル) で処理するパーティション数を決定
        // = ceil(numPartsIR / (partSize / blockSize))
        // ただし最低 1, 最大 numPartsIR
        if (!l.isImmediate)
        {
            const int blocksPerPart = l.partSize / std::max(blockSize, 1);
            l.partsPerCallback = std::max(1,
                (l.numPartsIR + blocksPerPart - 1) / blocksPerPart);
            l.partsPerCallback = std::min(l.partsPerCallback, l.numPartsIR);
        }

        l.fdlIndex = 0;
        l.inputPos = 0;
        l.nextPart = 0;

        ++m_numActiveLayers;
    }

    if (m_numActiveLayers == 0)
        return false;

    // ────────────────────────────────────────────────
    // 出力リングバッファ確保
    // サイズ = 2^ceil(log2(maxPartSize * 4 + blockSize * 4))
    // ※ 将来多層化する際は、最大パーティションサイズに合わせて調整が必要。
    //    また、ringWriteは上書き(memcpy)のため、多層化時は加算(accumulate)への変更が必須。
    // ────────────────────────────────────────────────
    int maxPartSize = m_layers[0].partSize; // 現在は単一レイヤーなのでL0が最大
    int rSize = juce::nextPowerOfTwo(maxPartSize * 4 + blockSize * 4);
    m_ringBuf  = static_cast<double*>(mkl_malloc(rSize * sizeof(double), 64));
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
// processLayerBlock  ─ Audio Thread
// l.inputAccBuf に 1 パーティション分の入力が溜まったタイミングで呼ぶ。
// (または非即時レイヤーの定期呼び出し)
//
// 処理:
//   1. Overlap-Save 形式で fftTimeBuf を組み立てる
//   2. Forward FFT → FDL に格納
//   3. FDL と IR の複素乗算積算 (AVX2 FMA)
//   4. Backward FFT
//   5. Overlap-Add して出力リングバッファへ書き込む
//==============================================================================
void MKLNonUniformConvolver::processLayerBlock(Layer& l) noexcept
{
    // ── 1. [prevInput | currentInput] を fftTimeBuf に配置 (Overlap-Save) ──
    memcpy(l.fftTimeBuf,                l.prevInputBuf,  l.partSize * sizeof(double));
    memcpy(l.fftTimeBuf + l.partSize,   l.inputAccBuf,   l.partSize * sizeof(double));

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
        const int lineIdx = (l.fdlIndex - p + l.numParts) & l.fdlMask;
        const double* srcA = fdlBase + lineIdx       * l.partStride;  // FDL[p]
        const double* srcB = irBase  + p             * l.partStride;  // IR[p]

        int k = 0;

        // 4 複素数 (= 8 double) を 1 ループで処理
        const int vEnd = (l.complexSize / 4) * 4;
        for (; k < vEnd; k += 4)
        {
            // accumBuf の 8 doubles (4 複素数) を load (アライン保証)
            __m256d acc0 = _mm256_load_pd(dst + 2 * k);
            __m256d acc1 = _mm256_load_pd(dst + 2 * k + 4);

            // FDL の 8 doubles をロード
            __m256d a0 = _mm256_load_pd(srcA + 2 * k);
            __m256d a1 = _mm256_load_pd(srcA + 2 * k + 4);

            // IR の 8 doubles をロード
            __m256d b0 = _mm256_load_pd(srcB + 2 * k);
            __m256d b1 = _mm256_load_pd(srcB + 2 * k + 4);

            // 複素乗算積算: acc += a * b
            // Re(a*b) = Re(a)*Re(b) - Im(a)*Im(b)
            // Im(a*b) = Re(a)*Im(b) + Im(a)*Re(b)
            // AVX2 を使った実装 (movedup / permute で Re/Im を分離)

            // --- 4 複素数ブロック 0 ---
            __m256d a0_re = _mm256_movedup_pd(a0);          // [Ar0, Ar0, Ar1, Ar1]
            __m256d a0_im = _mm256_permute_pd(a0, 0xF);     // [Ai0, Ai0, Ai1, Ai1]
            // acc += a_re * b  (Re の寄与)
            acc0 = _mm256_fmadd_pd(a0_re, b0, acc0); // FMA
            // acc += addsub(a_im * b_swap)  (Im の寄与)
            __m256d b0_sw = _mm256_permute_pd(b0, 0x5);     // [Bi0, Br0, Bi1, Br1]
            __m256d t0    = _mm256_mul_pd(a0_im, b0_sw);
            acc0 = _mm256_addsub_pd(acc0, t0);

            // --- 4 複素数ブロック 1 ---
            __m256d a1_re = _mm256_movedup_pd(a1);
            __m256d a1_im = _mm256_permute_pd(a1, 0xF);
            acc1 = _mm256_fmadd_pd(a1_re, b1, acc1); // FMA
            __m256d b1_sw = _mm256_permute_pd(b1, 0x5);
            __m256d t1    = _mm256_mul_pd(a1_im, b1_sw);
            acc1 = _mm256_addsub_pd(acc1, t1);

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

    // ── 5. Overlap-Add → 出力リングバッファ書き込み ──
    // Overlap-Save の場合、有効出力は後半 partSize サンプル
    const double* validOut = l.fftOutBuf + l.partSize;

    // overlapBuf との加算 (Overlap-Add)
    // overlapBuf は前回の IFFT 後半サンプルの "尾" を保持する
    // 今回は validOut に overlapBuf を加算してからリングへ書き込む
    // ──→ 実際には Overlap-Save なので overlapBuf は不要。
    //      ただし将来 Overlap-Add 方式に切り替える際のフックとして残す。
    //      現状は validOut をそのままリングに書き込む。
    ringWrite(validOut, l.partSize);

    // ── 6. FDL インデックスを進める ──
    l.fdlIndex = (l.fdlIndex + 1) & l.fdlMask;
}

//==============================================================================
// ringWrite  ─ Audio Thread
//==============================================================================
void MKLNonUniformConvolver::ringWrite(const double* src, int n) noexcept
{
    if (n <= 0 || m_ringBuf == nullptr) return;

    const int first = std::min(n, m_ringSize - m_ringWrite);
    memcpy(m_ringBuf + m_ringWrite, src, first * sizeof(double));
    if (n > first)
        memcpy(m_ringBuf, src + first, (n - first) * sizeof(double));

    m_ringWrite  = (m_ringWrite + n) & m_ringMask;

    const int nextAvail = m_ringAvail + n;
    if (nextAvail > m_ringSize)
    {
        const int overflow = nextAvail - m_ringSize;
        m_ringRead = (m_ringRead + overflow) & m_ringMask;
        m_ringAvail = m_ringSize;
    }
    else
    {
        m_ringAvail = nextAvail;
    }
}

//==============================================================================
// ringRead  ─ Audio Thread
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

    if (dst)
    {
        const int first = std::min(toRead, m_ringSize - m_ringRead);
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
//==============================================================================
void MKLNonUniformConvolver::Add(const double* input, int numSamples)
{
    // input == nullptr は無音として扱うため、ここでのチェックは削除
    if (!m_ready.load(std::memory_order_acquire) || numSamples <= 0)
        return;

    // すべてのアクティブレイヤーに対して処理を行う
    for (int li = 0; li < m_numActiveLayers; ++li)
    {
        Layer& l = m_layers[li];

        // ────────────────────────────────────────────
        // 入力を partSize 単位で蓄積し、溜まったら処理する
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
                    // L0: 即時処理 (Audio Thread 内で全パーティション畳み込み)
                    processLayerBlock(l);
                }
                else
                {
                    // L1/L2: 遅延処理
                    // Forward FFT + FDL 更新だけ今すぐ行い、
                    // 畳み込み計算は次の数コールバックに分散する

                    // Forward FFT → FDL 格納
                    memcpy(l.fftTimeBuf,              l.prevInputBuf, l.partSize * sizeof(double));
                    memcpy(l.fftTimeBuf + l.partSize, l.inputAccBuf,  l.partSize * sizeof(double));
                    memcpy(l.prevInputBuf, l.inputAccBuf, l.partSize * sizeof(double));

                    double* currentFDLSlot = l.fdlBuf + l.fdlIndex * l.partStride;
                    DftiComputeForward(l.fftHandle, l.fftTimeBuf, currentFDLSlot);
                    l.fdlIndex = (l.fdlIndex + 1) & l.fdlMask;

                    // 畳み込み計算を負荷分散: partsPerCallback パーティションずつ処理
                    const int end = std::min(l.nextPart + l.partsPerCallback, l.numPartsIR);

                    // accumBuf を初期化 (nextPart == 0 のときのみクリア)
                    if (l.nextPart == 0)
                        memset(l.accumBuf, 0, l.partStride * sizeof(double));

                    const double* fdlBase = l.fdlBuf;
                    const double* irBase  = l.irFreqDomain;
                    double*       dst     = l.accumBuf;

                    // 現在の FDL インデックス基準 (1 つ前: fdlIndex は既に +1 済み)
                    const int baseFdlIdx = (l.fdlIndex - 1 + l.numParts) & l.fdlMask;

                    for (int p = l.nextPart; p < end; ++p)
                    {
                        const int lineIdx = (baseFdlIdx - p + l.numParts) & l.fdlMask;
                        const double* srcA = fdlBase + lineIdx * l.partStride;
                        const double* srcB = irBase  + p       * l.partStride;

                        int k = 0;
                        const int vEnd4 = (l.complexSize / 4) * 4;
                        for (; k < vEnd4; k += 4)
                        {
                            __m256d acc0 = _mm256_load_pd(dst + 2 * k);
                            __m256d acc1 = _mm256_load_pd(dst + 2 * k + 4);
                            __m256d a0   = _mm256_load_pd(srcA + 2 * k);
                            __m256d a1   = _mm256_load_pd(srcA + 2 * k + 4);
                            __m256d b0   = _mm256_load_pd(srcB + 2 * k);
                            __m256d b1   = _mm256_load_pd(srcB + 2 * k + 4);

                            __m256d a0_re = _mm256_movedup_pd(a0);
                            __m256d a0_im = _mm256_permute_pd(a0, 0xF);
                            acc0 = _mm256_fmadd_pd(a0_re, b0, acc0); // FMA
                            __m256d b0_sw = _mm256_permute_pd(b0, 0x5);
                            acc0 = _mm256_addsub_pd(acc0, _mm256_mul_pd(a0_im, b0_sw));

                            __m256d a1_re = _mm256_movedup_pd(a1);
                            __m256d a1_im = _mm256_permute_pd(a1, 0xF);
                            acc1 = _mm256_fmadd_pd(a1_re, b1, acc1); // FMA
                            __m256d b1_sw = _mm256_permute_pd(b1, 0x5);
                            acc1 = _mm256_addsub_pd(acc1, _mm256_mul_pd(a1_im, b1_sw));

                            _mm256_store_pd(dst + 2 * k,     acc0);
                            _mm256_store_pd(dst + 2 * k + 4, acc1);
                        }
                        for (; k < l.complexSize; ++k)
                        {
                            const double ar = srcA[2*k], ai = srcA[2*k+1];
                            const double br = srcB[2*k], bi = srcB[2*k+1];
                            dst[2*k]   += ar*br - ai*bi;
                            dst[2*k+1] += ar*bi + ai*br;
                        }
                    }

                    l.nextPart = end;

                    // 全パーティションの積算が終わったら IFFT して出力へ
                    if (l.nextPart >= l.numPartsIR)
                    {
                        DftiComputeBackward(l.fftHandle, l.accumBuf, l.fftOutBuf);
                        ringWrite(l.fftOutBuf + l.partSize, l.partSize);
                        l.nextPart = 0;
                    }
                }
            }
        } // while consumed < numSamples
    } // for each layer
}

//==============================================================================
// Get  ─ Audio Thread
//==============================================================================
int MKLNonUniformConvolver::Get(double* output, int numSamples)
{
    // output == nullptr の場合もリングバッファを進める(データ破棄)ため、ここでのチェックは削除
    if (!m_ready.load(std::memory_order_acquire) || numSamples <= 0)
    {
        if (output && numSamples > 0)
            memset(output, 0, numSamples * sizeof(double));
        return 0;
    }

    return ringRead(output, numSamples);
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

        const size_t fdlBufSize = static_cast<size_t>(l.numParts) * l.partStride;
        memset(l.fdlBuf,       0, fdlBufSize   * sizeof(double));
        memset(l.overlapBuf,   0, l.partSize    * sizeof(double));
        memset(l.fftTimeBuf,   0, l.fftSize     * sizeof(double));
        memset(l.fftOutBuf,    0, l.fftSize     * sizeof(double));
        memset(l.prevInputBuf, 0, l.partSize    * sizeof(double));
        memset(l.accumBuf,     0, l.partStride  * sizeof(double));
        memset(l.inputAccBuf,  0, l.partSize    * sizeof(double));

        l.fdlIndex = 0;
        l.inputPos = 0;
        l.nextPart = 0;
    }

    if (m_ringBuf)
        memset(m_ringBuf, 0, m_ringSize * sizeof(double));
    m_ringWrite = 0;
    m_ringRead  = 0;
    m_ringAvail = 0;
}

} // namespace convo
