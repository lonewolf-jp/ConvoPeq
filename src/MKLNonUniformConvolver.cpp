//============================================================================
// MKLNonUniformConvolver.cpp  ── v1.0 (JUCE 8.0.12 / Intel oneMKL 対応)
//
// Non-Uniform Partitioned Convolution の完全実装。
// 3 層レイヤー構造で低遅延と大 IR を両立する。
//============================================================================

#include "MKLNonUniformConvolver.h"

#if JUCE_DSP_USE_INTEL_MKL

#include <mkl.h>
#include <mkl_dfti.h>
#include <mkl_vml.h>
#include <cstring>
#include <algorithm>
#include <cmath>

#if JUCE_INTEL
#include <immintrin.h>  // AVX2
#endif

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
    irOffset = 0;
    fdlIndex = inputPos = 0;
    processedBlocks = 0;
    pendingBaseFdlIdx = 0;
    pendingNextPart = 0;
    preferredPartsPerCall = 1;
    pendingBlockStartSample = 0;
    pendingActive = false;
    isImmediate = false;
}
MKLNonUniformConvolver::MKLNonUniformConvolver()
{
    mkl_set_num_threads(1);
}

MKLNonUniformConvolver::~MKLNonUniformConvolver()
{
    releaseAllLayers();
    if (m_ringBuf) { mkl_free(m_ringBuf); m_ringBuf = nullptr; }
}

void MKLNonUniformConvolver::releaseAllLayers() noexcept
{
    for (int i = 0; i < kNumLayers; ++i)
        m_layers[i].freeAll();
    m_numActiveLayers = 0;
    m_latency         = 0;
}

//==============================================================================
// SetImpulse  ─ Message Thread のみ
//==============================================================================
bool MKLNonUniformConvolver::SetImpulse(const double* impulse, int irLen, int blockSize)
{
    m_ready.store(false, std::memory_order_release);

    if (impulse == nullptr || irLen <= 0 || blockSize <= 0)
        return false;

    // 前回のリソースを解放
    releaseAllLayers();
    if (m_ringBuf) { mkl_free(m_ringBuf); m_ringBuf = nullptr; }
    m_ringSize = m_ringMask = 0;
    m_outputReadSample = 0;
    m_inputSampleCursor = 0;
    m_lastNonSilentInputSample = -(1LL << 60);
    m_irLength = 0;
    m_zeroInputFastPathActive = false;

    auto fail = [this]() -> bool
    {
        releaseAllLayers();
        if (m_ringBuf) { mkl_free(m_ringBuf); m_ringBuf = nullptr; }
        m_ringSize = m_ringMask = 0;
        m_outputReadSample = 0;
        m_inputSampleCursor = 0;
        m_lastNonSilentInputSample = -(1LL << 60);
        m_irLength = 0;
        m_zeroInputFastPathActive = false;
        return false;
    };

    // VML 高精度モード設定 (Message Thread で一度だけ)
    vmlSetMode(VML_HA | VML_FTZDAZ_ON);

    // ────────────────────────────────────────────────
    // レイヤー構成決定 (時間整合付き 3 層 NUC)
    //   L0: 即時レイヤー (最小パーティション)
    //   L1: 中域レイヤー
    //   L2: 末尾レイヤー
    // 各レイヤーは元 IR 内の offset を保持し、出力時に時間整合して合成する。
    // ────────────────────────────────────────────────
    const int l0Part = juce::nextPowerOfTwo(std::max(blockSize, 256));
    const int l1Part = juce::nextPowerOfTwo(std::max(2048, l0Part * 2));
    const int l2Part = juce::nextPowerOfTwo(std::max(8192, l1Part * 2));

    const int l0Len = std::min(irLen, l1Part * 2);
    const int remainAfterL0 = irLen - l0Len;
    const int l1Len = std::min(remainAfterL0, l2Part);
    const int l2Len = irLen - l0Len - l1Len;

    struct LayerCfg { int offset; int len; int partSize; bool immediate; };
    LayerCfg cfgs[kNumLayers];

    cfgs[0].offset    = 0;
    cfgs[0].len       = l0Len;
    cfgs[0].partSize  = l0Part;
    cfgs[0].immediate = true;

    cfgs[1].offset    = l0Len;
    cfgs[1].len       = l1Len;
    cfgs[1].partSize  = l1Part;
    cfgs[1].immediate = false;

    cfgs[2].offset    = l0Len + l1Len;
    cfgs[2].len       = l2Len;
    cfgs[2].partSize  = l2Part;
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
        l.irOffset    = cfgs[li].offset;
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
            return fail();

        bool ok = true;
        ok = ok && (DftiSetValue(l.fftHandle, DFTI_PLACEMENT,             DFTI_NOT_INPLACE)     == DFTI_NO_ERROR);
        ok = ok && (DftiSetValue(l.fftHandle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX) == DFTI_NO_ERROR);
        ok = ok && (DftiSetValue(l.fftHandle, DFTI_BACKWARD_SCALE, 1.0 / static_cast<double>(l.fftSize)) == DFTI_NO_ERROR);
        ok = ok && (DftiCommitDescriptor(l.fftHandle) == DFTI_NO_ERROR);

        if (!ok)
            return fail();

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
            return fail();

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
            return fail();
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
            // p >= numPartsIR のスロットはゼロパディング

            DftiComputeForward(l.fftHandle, tempTime, tempFreq);

            // interleaved complex として irFreqDomain に格納
            memcpy(l.irFreqDomain + p * l.partStride, tempFreq,
                   l.complexSize * 2 * sizeof(double));
        }

        mkl_free(tempTime);
        mkl_free(tempFreq);

        l.fdlIndex = 0;
        l.inputPos = 0;
        l.processedBlocks = 0;
        l.preferredPartsPerCall = std::max(1,
            static_cast<int>((static_cast<int64_t>(l.numPartsIR) * blockSize + l.partSize - 1) / l.partSize));
        l.pendingBaseFdlIdx = 0;
        l.pendingNextPart = 0;
        l.pendingBlockStartSample = 0;
        l.pendingActive = false;

        ++m_numActiveLayers;
    }

    if (m_numActiveLayers == 0)
        return fail();

    m_latency = m_layers[0].partSize;  // Layer0 の partSize = 基準レイテンシ

    // ────────────────────────────────────────────────
    // 出力リングバッファ確保
    // 時間整合付き合成のため、IR 長 + 基準遅延 + 余裕分を確保する。
    // ────────────────────────────────────────────────
    const int maxLookahead = irLen + m_latency + l2Part + blockSize * 4;
    const int rSize = juce::nextPowerOfTwo(std::max(maxLookahead, blockSize * 8));
    m_ringBuf  = static_cast<double*>(mkl_malloc(static_cast<size_t>(rSize) * sizeof(double), 64));
    if (!m_ringBuf)
        return fail();

    memset(m_ringBuf, 0, static_cast<size_t>(rSize) * sizeof(double));
    m_ringSize  = rSize;
    m_ringMask  = rSize - 1;
    m_outputReadSample = 0;
    m_inputSampleCursor = 0;
    m_lastNonSilentInputSample = -(1LL << 60);
    m_irLength = irLen;
    m_zeroInputFastPathActive = false;

    m_ready.store(true, std::memory_order_release);
    return true;
}
void MKLNonUniformConvolver::accumulateLayerProducts(Layer& l, int baseFdlIdx, int startPart, int endPart) noexcept
{
    if (startPart >= endPart)
        return;

    const double* fdlBase = l.fdlBuf;
    const double* irBase  = l.irFreqDomain;
    double*       dst     = l.accumBuf;

    for (int p = startPart; p < endPart; ++p)
    {
        const int lineIdx = (baseFdlIdx - p + l.numParts) & l.fdlMask;
        const double* srcA = fdlBase + lineIdx * l.partStride;
        const double* srcB = irBase  + p       * l.partStride;

        int k = 0;
#if defined(__AVX2__)
        const int vEnd = (l.complexSize / 4) * 4;
        for (; k < vEnd; k += 4)
        {
            __m256d acc0 = _mm256_load_pd(dst + 2 * k);
            __m256d acc1 = _mm256_load_pd(dst + 2 * k + 4);
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
#endif
        for (; k < l.complexSize; ++k)
        {
            const double ar = srcA[2 * k], ai = srcA[2 * k + 1];
            const double br = srcB[2 * k], bi = srcB[2 * k + 1];
            dst[2 * k]     += ar * br - ai * bi;
            dst[2 * k + 1] += ar * bi + ai * br;
        }
    }
}
void MKLNonUniformConvolver::processLayerBlock(Layer& l) noexcept
{
    // ── 1. [prevInput | currentInput] を fftTimeBuf に配置 (Overlap-Save) ──
    memcpy(l.fftTimeBuf,                l.prevInputBuf,  l.partSize * sizeof(double));
    memcpy(l.fftTimeBuf + l.partSize,   l.inputAccBuf,   l.partSize * sizeof(double));

    // 現在の入力を次回の "prev" として保存
    memcpy(l.prevInputBuf, l.inputAccBuf, l.partSize * sizeof(double));

    // ── 2. Forward FFT ──
    const int baseFdlIdx = l.fdlIndex;
    double* currentFDLSlot = l.fdlBuf + baseFdlIdx * l.partStride;
    DftiComputeForward(l.fftHandle, l.fftTimeBuf, currentFDLSlot);

    // ── 3. 複素乗算積算 (FDL × IR) → accumBuf ──
    memset(l.accumBuf, 0, l.partStride * sizeof(double));
    accumulateLayerProducts(l, baseFdlIdx, 0, l.numPartsIR);

    // ── 4. Backward FFT ──
    DftiComputeBackward(l.fftHandle, l.accumBuf, l.fftOutBuf);

    // ── 5. Overlap-Save 有効出力を時間整合付きでリングへ加算 ──
    const double* validOut = l.fftOutBuf + l.partSize;
    const int64_t blockStartSample = l.processedBlocks * static_cast<int64_t>(l.partSize);
    const int64_t writeStartSample = blockStartSample + static_cast<int64_t>(l.irOffset + m_latency);
    ringAddAt(writeStartSample, validOut, l.partSize);

    // ── 6. 状態更新 ──
    l.fdlIndex = (l.fdlIndex + 1) & l.fdlMask;
    ++l.processedBlocks;
}
void MKLNonUniformConvolver::ringAddAt(int64_t startSample, const double* src, int n) noexcept
{
    if (n <= 0 || src == nullptr || m_ringBuf == nullptr || m_ringSize <= 0)
        return;

    const int64_t readHead = m_outputReadSample;
    int srcOffset = 0;

    // 既に読み出し済み区間に書こうとした分は破棄
    if (startSample < readHead)
    {
        srcOffset = static_cast<int>(readHead - startSample);
        if (srcOffset >= n)
            return;

        startSample = readHead;
        n -= srcOffset;
    }

    // リング容量を超える未来位置への書き込みはクリップ
    const int64_t writableEnd = readHead + static_cast<int64_t>(m_ringSize);
    if (startSample >= writableEnd)
        return;

    if (startSample + static_cast<int64_t>(n) > writableEnd)
        n = static_cast<int>(writableEnd - startSample);

    if (n <= 0)
        return;

    const int writePos = static_cast<int>(startSample & m_ringMask);
    const int first = std::min(n, m_ringSize - writePos);

    cblas_daxpy(first, 1.0, src + srcOffset, 1, m_ringBuf + writePos, 1);

    if (n > first)
        cblas_daxpy(n - first, 1.0, src + srcOffset + first, 1, m_ringBuf, 1);
}
int MKLNonUniformConvolver::ringRead(double* dst, int n) noexcept
{
    if (n <= 0 || dst == nullptr || m_ringBuf == nullptr)
        return 0;

    const int readPos = static_cast<int>(m_outputReadSample & m_ringMask);
    const int first = std::min(n, m_ringSize - readPos);

    memcpy(dst, m_ringBuf + readPos, first * sizeof(double));
    memset(m_ringBuf + readPos, 0, first * sizeof(double));

    if (n > first)
    {
        memcpy(dst + first, m_ringBuf, (n - first) * sizeof(double));
        memset(m_ringBuf, 0, (n - first) * sizeof(double));
    }

    m_outputReadSample += static_cast<int64_t>(n);
    return n;
}
void MKLNonUniformConvolver::Add(const double* input, int numSamples)
{
    if (!m_ready.load(std::memory_order_acquire) || numSamples <= 0)
        return;

    static constexpr double kSilenceThreshold = 1.0e-15;

    bool inputSilent = (input == nullptr);
    if (!inputSilent)
    {
        for (int i = 0; i < numSamples; ++i)
        {
            const double v = input[i];
            if (v > kSilenceThreshold || v < -kSilenceThreshold)
            {
                inputSilent = false;
                break;
            }
            inputSilent = true;
        }
    }

    const int64_t blockStartCursor = m_inputSampleCursor;
    if (!inputSilent)
        m_lastNonSilentInputSample = blockStartCursor + static_cast<int64_t>(numSamples) - 1;

    bool anyPending = false;
    for (int li = 0; li < m_numActiveLayers; ++li)
    {
        if (m_layers[li].pendingActive)
        {
            anyPending = true;
            break;
        }
    }

    const int64_t tailEndSample = m_lastNonSilentInputSample + static_cast<int64_t>(m_irLength) + static_cast<int64_t>(m_latency) + 1;
    const bool tailExpired = (m_outputReadSample >= tailEndSample);

    if (inputSilent && !anyPending && tailExpired)
    {
        if (!m_zeroInputFastPathActive)
        {
            for (int li = 0; li < m_numActiveLayers; ++li)
            {
                Layer& l = m_layers[li];
                if (!l.fdlBuf) continue;

                const size_t fdlBufSize = static_cast<size_t>(l.numParts) * l.partStride;
                memset(l.fdlBuf, 0, fdlBufSize * sizeof(double));
                memset(l.prevInputBuf, 0, l.partSize * sizeof(double));
                memset(l.inputAccBuf, 0, l.partSize * sizeof(double));
                memset(l.accumBuf, 0, l.partStride * sizeof(double));

                l.pendingActive = false;
                l.pendingNextPart = 0;
                l.pendingBaseFdlIdx = 0;
            }
            m_zeroInputFastPathActive = true;
        }

        // 無音中はFFTを回さず、タイムラインだけ進める
        for (int li = 0; li < m_numActiveLayers; ++li)
        {
            Layer& l = m_layers[li];
            const int total = l.inputPos + numSamples;
            const int blocks = total / l.partSize;
            l.inputPos = total - blocks * l.partSize;
            if (blocks > 0)
            {
                l.fdlIndex = (l.fdlIndex + blocks) & l.fdlMask;
                l.processedBlocks += static_cast<int64_t>(blocks);
            }
        }

        m_inputSampleCursor += static_cast<int64_t>(numSamples);
        return;
    }

    m_zeroInputFastPathActive = false;

    auto finalizePendingJob = [this](Layer& l) noexcept
    {
        DftiComputeBackward(l.fftHandle, l.accumBuf, l.fftOutBuf);
        const int64_t writeStartSample = l.pendingBlockStartSample + static_cast<int64_t>(l.irOffset + m_latency);
        ringAddAt(writeStartSample, l.fftOutBuf + l.partSize, l.partSize);
        l.pendingActive = false;
        l.pendingNextPart = 0;
    };

    for (int li = 0; li < m_numActiveLayers; ++li)
    {
        Layer& l = m_layers[li];

        int consumed = 0;
        while (consumed < numSamples)
        {
            const int toFill = std::min(numSamples - consumed, l.partSize - l.inputPos);

            if (input != nullptr)
                memcpy(l.inputAccBuf + l.inputPos, input + consumed, toFill * sizeof(double));
            else
                memset(l.inputAccBuf + l.inputPos, 0, toFill * sizeof(double));

            l.inputPos += toFill;
            consumed   += toFill;

            if (l.inputPos < l.partSize)
                continue;

            l.inputPos = 0;

            if (l.isImmediate)
            {
                processLayerBlock(l);
                continue;
            }

            // 非即時レイヤー: Forward FFT は即時、積算は分割実行
            memcpy(l.fftTimeBuf,              l.prevInputBuf, l.partSize * sizeof(double));
            memcpy(l.fftTimeBuf + l.partSize, l.inputAccBuf,  l.partSize * sizeof(double));
            memcpy(l.prevInputBuf, l.inputAccBuf, l.partSize * sizeof(double));

            const int baseFdlIdx = l.fdlIndex;
            double* currentFDLSlot = l.fdlBuf + baseFdlIdx * l.partStride;
            DftiComputeForward(l.fftHandle, l.fftTimeBuf, currentFDLSlot);
            l.fdlIndex = (l.fdlIndex + 1) & l.fdlMask;

            // 万一前ジョブが残っていた場合は即時完了して整合を維持
            if (l.pendingActive)
            {
                accumulateLayerProducts(l, l.pendingBaseFdlIdx, l.pendingNextPart, l.numPartsIR);
                l.pendingNextPart = l.numPartsIR;
                finalizePendingJob(l);
            }

            memset(l.accumBuf, 0, l.partStride * sizeof(double));
            l.pendingBaseFdlIdx = baseFdlIdx;
            l.pendingNextPart = 0;
            l.pendingBlockStartSample = l.processedBlocks * static_cast<int64_t>(l.partSize);
            l.pendingActive = true;
            ++l.processedBlocks;
        }

        if (!l.isImmediate && l.pendingActive)
        {
            const int dynamicBudget = std::max(1,
                static_cast<int>((static_cast<int64_t>(l.numPartsIR) * numSamples + l.partSize - 1) / l.partSize));
            const int partBudget = std::max(l.preferredPartsPerCall, dynamicBudget);
            const int endPart = std::min(l.pendingNextPart + partBudget, l.numPartsIR);

            if (endPart > l.pendingNextPart)
            {
                accumulateLayerProducts(l, l.pendingBaseFdlIdx, l.pendingNextPart, endPart);
                l.pendingNextPart = endPart;
            }

            if (l.pendingNextPart >= l.numPartsIR)
                finalizePendingJob(l);
        }
    }

    m_inputSampleCursor += static_cast<int64_t>(numSamples);
}
int MKLNonUniformConvolver::Get(double* output, int numSamples)
{
    if (!m_ready.load(std::memory_order_acquire) || output == nullptr || numSamples <= 0)
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
        l.processedBlocks = 0;
        l.pendingBaseFdlIdx = 0;
        l.pendingNextPart = 0;
        l.pendingBlockStartSample = 0;
        l.pendingActive = false;
    }

    if (m_ringBuf)
        memset(m_ringBuf, 0, static_cast<size_t>(m_ringSize) * sizeof(double));

    m_outputReadSample = 0;
    m_inputSampleCursor = 0;
    m_lastNonSilentInputSample = -(1LL << 60);
    m_zeroInputFastPathActive = false;
}
} // namespace convo

#endif // JUCE_DSP_USE_INTEL_MKL
