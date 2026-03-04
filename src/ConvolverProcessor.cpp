//============================================================================
// ConvolverProcessor.cpp  ── v0.2 (JUCE 8.0.12対応)
//
// コンボリューションプロセッサーの実装
//============================================================================
#include "ConvolverProcessor.h"
#include "InputBitDepthTransform.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <utility>
#include <cstring>
#include <limits>
#include <new>

#include "WDL/fft.h" // WDL Double precision FFT
#include "CDSPResampler.h"
#include "AlignedAllocation.h" // For convo::MKLAllocator

#if JUCE_DSP_USE_INTEL_MKL
#include <mkl.h>
#include <mkl_vml.h>
#endif

#if JUCE_INTEL
 #include <xmmintrin.h>
 #include <pmmintrin.h>
 #include <immintrin.h> // For AVX2
#endif

#if JUCE_DSP_USE_INTEL_MKL
namespace
{
    struct DftiGuard
    {
        DFTI_DESCRIPTOR_HANDLE* handle = nullptr;

        explicit DftiGuard(DFTI_DESCRIPTOR_HANDLE* h) noexcept : handle(h) {}

        ~DftiGuard()
        {
            if (handle != nullptr && *handle != nullptr)
            {
                DftiFreeDescriptor(handle);
                *handle = nullptr;
            }
        }

        DftiGuard(const DftiGuard&) = delete;
        DftiGuard& operator=(const DftiGuard&) = delete;
    };
}
#endif

#if JUCE_DSP_USE_INTEL_MKL
//--------------------------------------------------------------
// MKLConvolver Implementation
// Uniform Partitioned Convolution using Overlap-Save
//--------------------------------------------------------------
bool ConvolverProcessor::MKLConvolver::setup(int partSize, const double* ir, int irLen)
{
    // [FIX] Guard against zero/negative partition size (causes integer divide-by-zero)
    // This happens when uiConvolverProcessor.prepareToPlay() was never called,
    // leaving currentBufferSize = 0, which propagates as partSize = 0.
    if (partSize <= 0 || irLen <= 0 || ir == nullptr)
    {
        DBG("MKLConvolver::setup: invalid parameters (partSize=" << partSize
            << ", irLen=" << irLen << "). Returning false to fall back to WDL.");
        return false;
    }

    partitionSize = partSize;
    fftSize = 1;
    while (fftSize < partitionSize * 2) fftSize <<= 1;

    // Calculate number of partitions
    numPartitions = (irLen + partitionSize - 1) / partitionSize;
    if (numPartitions == 0) numPartitions = 1;

    // Latency of UPC is one partition
    latency = partitionSize;

    // Setup MKL DFTI
    if (DftiCreateDescriptor(&fftHandle, DFTI_DOUBLE, DFTI_REAL, 1, fftSize) != DFTI_NO_ERROR) return false;
    DftiSetValue(fftHandle, DFTI_PLACEMENT, DFTI_NOT_INPLACE); // Explicit buffers
    DftiSetValue(fftHandle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX); // Ensure standard complex array format
    DftiSetValue(fftHandle, DFTI_BACKWARD_SCALE, 1.0 / static_cast<double>(fftSize));
    if (DftiCommitDescriptor(fftHandle) != DFTI_NO_ERROR) return false;

    // Allocate buffers
    // IR Freq Domain: (fftSize/2 + 1) complex numbers * numPartitions
    int complexSize = fftSize / 2 + 1;
    size_t irBufSize = static_cast<size_t>(complexSize) * 2 * numPartitions; // 2 doubles per complex
    irFreqDomain.reset(static_cast<double*>(convo::aligned_malloc(irBufSize * sizeof(double), 64)));
    juce::FloatVectorOperations::clear(irFreqDomain.get(), static_cast<int>(irBufSize));

    // FDL Lines
    fdlLines.reset(static_cast<MKL_Complex16*>(convo::aligned_malloc(irBufSize * sizeof(double), 64)));
    juce::FloatVectorOperations::clear(reinterpret_cast<double*>(fdlLines.get()), static_cast<int>(irBufSize));

    // Work Buffers
    fftBuffer.reset(static_cast<double*>(convo::aligned_malloc(fftSize * sizeof(double), 64)));
    inputBuffer.reset(static_cast<double*>(convo::aligned_malloc(partitionSize * sizeof(double), 64)));
    outputBuffer.reset(static_cast<double*>(convo::aligned_malloc(partitionSize * sizeof(double), 64)));
    prevBlock.reset(static_cast<double*>(convo::aligned_malloc(partitionSize * sizeof(double), 64)));
    mulTemp.reset(static_cast<MKL_Complex16*>(convo::aligned_malloc(complexSize * sizeof(MKL_Complex16), 64)));

    juce::FloatVectorOperations::clear(inputBuffer.get(), partitionSize);
    juce::FloatVectorOperations::clear(outputBuffer.get(), partitionSize);
    juce::FloatVectorOperations::clear(prevBlock.get(), partitionSize);

    // Precompute IR Partitions
    convo::ScopedAlignedPtr<double> tempTime(static_cast<double*>(convo::aligned_malloc(fftSize * sizeof(double), 64)));
    convo::ScopedAlignedPtr<double> tempFreq(static_cast<double*>(convo::aligned_malloc((fftSize + 2) * sizeof(double), 64)));

    for (int p = 0; p < numPartitions; ++p)
    {
        juce::FloatVectorOperations::clear(tempTime.get(), fftSize);
        int copyLen = std::min(partitionSize, irLen - p * partitionSize);
        if (copyLen > 0)
            std::memcpy(tempTime.get(), ir + p * partitionSize, copyLen * sizeof(double));

        DftiComputeForward(fftHandle, tempTime.get(), tempFreq.get());

        // Copy to IR buffer (interleaved complex)
        double* dest = irFreqDomain.get() + p * complexSize * 2;
        std::memcpy(dest, tempFreq.get(), complexSize * 2 * sizeof(double));
    }

    inputBufferPos = 0;
    outputBufferPos = 0;
    fdlIndex = 0;
    return true;
}

void ConvolverProcessor::MKLConvolver::process(const double* in, double* out, int numSamples)
{
    int processed = 0;
    while (processed < numSamples)
    {
        // Fill input buffer
        int toWrite = std::min(numSamples - processed, partitionSize - inputBufferPos);
        std::memcpy(inputBuffer.get() + inputBufferPos, in + processed, toWrite * sizeof(double));
        inputBufferPos += toWrite;

        // Process block if full
        if (inputBufferPos == partitionSize)
        {
            // Construct FFT input: [PrevBlock, CurrentBlock]
            std::memcpy(fftBuffer.get(), prevBlock.get(), partitionSize * sizeof(double));
            std::memcpy(fftBuffer.get() + partitionSize, inputBuffer.get(), partitionSize * sizeof(double));

            // Save current to prev
            std::memcpy(prevBlock.get(), inputBuffer.get(), partitionSize * sizeof(double));
            inputBufferPos = 0;

            // Forward FFT
            // Note: Output of Real->Complex is packed CCS or Permuted.
            // DFTI_NOT_INPLACE with CCE format is standard.
            // We use a temp buffer for frequency domain calculation.
            // Actually, let's use the fdlLines buffer slot for the current block.

            int complexSize = fftSize / 2 + 1;
            MKL_Complex16* currentFDL = reinterpret_cast<MKL_Complex16*>(fdlLines.get()) + fdlIndex * complexSize;

            DftiComputeForward(fftHandle, fftBuffer.get(), currentFDL);

            // Convolution (FDL)
            juce::FloatVectorOperations::clear(reinterpret_cast<double*>(mulTemp.get()), complexSize * 2); // Clear accumulator

            const double* fdlBase = reinterpret_cast<const double*>(fdlLines.get());
            const double* irBase = reinterpret_cast<const double*>(irFreqDomain.get());
            double* dstBase = reinterpret_cast<double*>(mulTemp.get());

            for (int p = 0; p < numPartitions; ++p)
            {
                int lineIdx = (fdlIndex - p + numPartitions) % numPartitions;
                const double* srcA = fdlBase + lineIdx * complexSize * 2;
                const double* srcB = irBase + p * complexSize * 2;
                double* dst = dstBase;

                int k = 0;
#if defined(__AVX2__)
                const int vEnd = (complexSize >> 1) << 1;
                for (; k < vEnd; k += 2)
                {
                    // Load Accumulator (Aligned)
                    __m256d acc = _mm256_load_pd(dst + 2 * k);
                    // Load FDL (Unaligned)
                    __m256d a = _mm256_loadu_pd(srcA + 2 * k);
                    // Load IR (Unaligned)
                    __m256d b = _mm256_loadu_pd(srcB + 2 * k);

                    // Complex Multiply Accumulate: Acc += A * B
                    // Re = Ar*Br - Ai*Bi
                    // Im = Ar*Bi + Ai*Br

                    __m256d a_re = _mm256_movedup_pd(a);
                    __m256d a_im = _mm256_permute_pd(a, 0xF);

                    acc = _mm256_fmadd_pd(a_re, b, acc);

                    __m256d b_swap = _mm256_permute_pd(b, 0x5);
                    __m256d term2 = _mm256_mul_pd(a_im, b_swap);

                    acc = _mm256_addsub_pd(acc, term2);

                    _mm256_store_pd(dst + 2 * k, acc);
                }
#endif
                for (; k < complexSize; ++k)
                {
                    double a_re = srcA[2 * k];
                    double a_im = srcA[2 * k + 1];
                    double b_re = srcB[2 * k];
                    double b_im = srcB[2 * k + 1];

                    dst[2 * k]     += a_re * b_re - a_im * b_im;
                    dst[2 * k + 1] += a_re * b_im + a_im * b_re;
                }
            }

            // Backward FFT
            DftiComputeBackward(fftHandle, mulTemp.get(), fftBuffer.get());

            // Overlap-Save: Output is the last partitionSize samples
            std::memcpy(outputBuffer.get(), fftBuffer.get() + partitionSize, partitionSize * sizeof(double));
            outputBufferPos = 0;

            // Advance FDL
            fdlIndex = (fdlIndex + 1) % numPartitions;
        }

        // Read from output buffer.
        // [FIX] toRead is bounded by toWrite, NOT by (numSamples - processed).
        // Using (numSamples - processed) caused an infinite loop when
        // outputBufferPos == partitionSize (output exhausted) but
        // inputBufferPos < partitionSize (input not yet full, so no FFT fired).
        // In that case toRead == 0 while toWrite > 0, so processed never advanced.
        // The correct invariant is: input and output advance by the same amount per
        // iteration (startup latency is absorbed by outputBuffer being pre-zeroed).
        int toRead = std::min(toWrite, partitionSize - outputBufferPos);
        if (toRead > 0)
        {
            std::memcpy(out + processed, outputBuffer.get() + outputBufferPos, toRead * sizeof(double));
            outputBufferPos += toRead;
        }
        // Zero-fill for any startup latency gap (output buffer exhausted before input fills)
        if (toRead < toWrite)
            std::memset(out + processed + toRead, 0, (toWrite - toRead) * sizeof(double));

        processed += toWrite; // Advance by input consumed, not output produced
    }
}

void ConvolverProcessor::MKLConvolver::reset()
{
    if (inputBuffer) juce::FloatVectorOperations::clear(inputBuffer.get(), partitionSize);
    if (outputBuffer) juce::FloatVectorOperations::clear(outputBuffer.get(), partitionSize);
    if (prevBlock) juce::FloatVectorOperations::clear(prevBlock.get(), partitionSize);
    if (fdlLines) juce::FloatVectorOperations::clear(reinterpret_cast<double*>(fdlLines.get()), (fftSize / 2 + 1) * 2 * numPartitions);
    inputBufferPos = 0;
    outputBufferPos = 0;
    fdlIndex = 0;
}

ConvolverProcessor::MKLConvolver::~MKLConvolver()
{
    if (fftHandle) DftiFreeDescriptor(&fftHandle);
}
#endif

void ConvolverProcessor::StereoConvolver::reset()
{
#if JUCE_DSP_USE_INTEL_MKL
    if (useMKL)
    {
        if (mklConvolvers[0]) mklConvolvers[0]->reset();
        if (mklConvolvers[1]) mklConvolvers[1]->reset();
        return;
    }
#endif
    convolvers[0].Reset();
    convolvers[1].Reset();
}

void ConvolverProcessor::StereoConvolver::process(int channel, const double* in, double* out, int numSamples)
{
#if JUCE_DSP_USE_INTEL_MKL
    if (useMKL && mklConvolvers[channel])
    {
        mklConvolvers[channel]->process(in, out, numSamples);
        return;
    }
#endif
    // WDL Path
    WDL_FFT_REAL* inputs[1] = { const_cast<WDL_FFT_REAL*>(in) };
    convolvers[channel].Add(inputs, numSamples, 1);
    int avail = convolvers[channel].Avail(numSamples);
    if (avail < numSamples)
    {
        // Should not happen if prewarmed correctly, but fill with silence if it does
        std::memset(out, 0, numSamples * sizeof(double));
        return;
    }
    WDL_FFT_REAL** outputs = convolvers[channel].Get();
    if (outputs && outputs[0])
    {
        std::memcpy(out, outputs[0], numSamples * sizeof(double));
    }
    else
    {
        std::memset(out, 0, numSamples * sizeof(double));
    }
    convolvers[channel].Advance(numSamples);
}

// 前方宣言
static juce::AudioBuffer<double> convertToMinimumPhase(const juce::AudioBuffer<double>& linearIR, const std::function<bool()>& shouldExit = nullptr, bool* wasCancelled = nullptr);

// スレッドキャンセル確認用ヘルパー関数
static bool checkCancellation(const std::function<bool()>& shouldExit, bool* wasCancelled) noexcept
{
    if (shouldExit && shouldExit())
    {
        if (wasCancelled)
            *wasCancelled = true;
        return true;
    }
    return false;
}

// AudioBufferの容量を現在のサイズに合わせて縮小するヘルパー
// JUCEのsetSize()は容量を縮小しないため、メモリ使用量を最適化するために使用する
static void shrinkToFit(juce::AudioBuffer<double>& buffer)
{
    if (buffer.getNumSamples() == 0 || buffer.getNumChannels() == 0)
        return;

    juce::AudioBuffer<double> newBuffer(buffer.getNumChannels(), buffer.getNumSamples());
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
        newBuffer.copyFrom(ch, 0, buffer, ch, 0, buffer.getNumSamples());

    buffer = std::move(newBuffer);
}

// リサンプリング用ヘルパー
static juce::AudioBuffer<double> resampleIR(const juce::AudioBuffer<double>& inputIR, double inputSR, double targetSR, const std::function<bool()>& shouldExit)
{
    if (inputSR <= 0.0 || targetSR <= 0.0 || std::abs(inputSR - targetSR) <= 1e-6)
        return inputIR;

    const double ratio = targetSR / inputSR;
    const int inLength = inputIR.getNumSamples();

    // 出力長オーバーフローの安全チェック
    const double expectedLen = inLength * ratio + 2.0;
    if (expectedLen > static_cast<double>(std::numeric_limits<int>::max()))
        return {};

    const int maxOutLen = static_cast<int>(expectedLen);

    juce::AudioBuffer<double> resampled(inputIR.getNumChannels(), maxOutLen);
    resampled.clear();

    constexpr double transBand = 2.0;
    constexpr double stopBandAtten = 140.0;
    constexpr r8b::EDSPFilterPhaseResponse phase = r8b::fprLinearPhase;

    int maxLength = 0;
    for (int ch = 0; ch < inputIR.getNumChannels(); ++ch)
    {
        if (checkCancellation(shouldExit, nullptr)) return {};

        auto resampler = std::make_unique<r8b::CDSPResampler>(inputSR, targetSR, inLength, transBand, stopBandAtten, phase);

        const double* inPtr = inputIR.getReadPointer(ch);
        double* outPtr = resampled.getWritePointer(ch);

        int done = 0;
        int inputProcessed = 0;
        int iterations = 0;
        constexpr int maxIterations = 1000000; // 無限ループ防止のための安全カウンター
        constexpr int CHUNK_SIZE = 4096; // キャンセル応答性を高めるためのチャンクサイズ

        // 入力をチャンク分割して処理 (キャンセルチェックを頻繁に行うため)
        while (inputProcessed < inLength && done < maxOutLen && ++iterations < maxIterations)
        {
            if (checkCancellation(shouldExit, nullptr)) return {};

            int chunk = std::min(CHUNK_SIZE, inLength - inputProcessed);
            double* r8bOutput = nullptr;

            const int generated = resampler->process(const_cast<double*>(inPtr + inputProcessed), chunk, r8bOutput);
            inputProcessed += chunk;

            if (generated > 0)
            {
                const int toCopy = std::min(generated, maxOutLen - done);
                std::memcpy(outPtr + done, r8bOutput, toCopy * sizeof(double));
                done += toCopy;
            }
        }

        // 残りの出力をフラッシュ (r8brainのレイテンシー分など)
        while (done < maxOutLen && ++iterations < maxIterations)
        {
            if (checkCancellation(shouldExit, nullptr)) return {};
            double* r8bOutput = nullptr;
            const int generated = resampler->process(nullptr, 0, r8bOutput);

            if (generated <= 0) break;

            const int toCopy = std::min(generated, maxOutLen - done);
            std::memcpy(outPtr + done, r8bOutput, toCopy * sizeof(double));
            done += toCopy;
        }
        maxLength = std::max(maxLength, done);
    }
    resampled.setSize(inputIR.getNumChannels(), maxLength, true, true, true);
    shrinkToFit(resampled); // 余分なキャパシティを解放

    // コンボリューション用のIRリサンプリングでは、サンプルレート比率の逆数をゲインとして適用する。
    // Upsampling (ratio > 1.0) -> Gain < 1.0 (減衰)
    // Downsampling (ratio < 1.0) -> Gain > 1.0 (増幅)
    // これにより、畳み込み積分のDCゲイン（総エネルギー）がサンプルレート変更前後で維持される。
    if (ratio > 0.0)
        resampled.applyGain(1.0 / ratio);

    return resampled;
}

// -------------------------------------------------------------------------
// 非対称Tukey窓ヘルパー
// -------------------------------------------------------------------------
static double calculate_post_alpha(int n_taps)
{
    if (n_taps <= 0) return 0.05;
    double log2n = std::log2(static_cast<double>(n_taps));
    double alpha = 0.05 + 0.033 * (log2n - 10.0);
    return std::max(0.05, std::min(0.25, alpha));
}

static void applyAsymmetricTukey(double* data, int numSamples)
{
    if (numSamples <= 0) return;

    // 1. ピーク位置の検出
    auto* start = data;
    auto* end = data + numSamples;
    auto it = std::max_element(start, end, [](double a, double b){
        return std::abs(a) < std::abs(b);
    });
    int peakIndex = static_cast<int>(std::distance(start, it));

    // 2. アルファ値の計算
    const double alpha_pre = 0.05;
    const double alpha_post = calculate_post_alpha(numSamples);
    const double pi = juce::MathConstants<double>::pi;

#if JUCE_DSP_USE_INTEL_MKL
    convo::ScopedAlignedPtr<double> window_vals(static_cast<double*>(convo::aligned_malloc(numSamples * sizeof(double), 64)));

    // Initialize to 1.0
    for (int i = 0; i < numSamples; ++i)
        window_vals[i] = 1.0;

    // Pre-peak part
    if (peakIndex > 0)
    {
        const int pre_taper_len = static_cast<int>(std::floor(peakIndex * alpha_pre));
        if (pre_taper_len > 0)
        {
            convo::ScopedAlignedPtr<double> cos_args(static_cast<double*>(convo::aligned_malloc(pre_taper_len * sizeof(double), 64)));
            for (int i = 0; i < pre_taper_len; ++i)
                cos_args[i] = pi * (static_cast<double>(i) / (peakIndex * alpha_pre) - 1.0);

            vdCos(pre_taper_len, cos_args.get(), window_vals.get());

            for (int i = 0; i < pre_taper_len; ++i)
                window_vals[i] = 0.5 * (1.0 + window_vals[i]);
        }
    }

    // Post-peak part
    const double dist_to_end = static_cast<double>(numSamples - 1 - peakIndex);
    if (dist_to_end > 0)
    {
        const int post_taper_start_idx = peakIndex + static_cast<int>(std::ceil(dist_to_end * (1.0 - alpha_post)));
        const int post_taper_len = numSamples - post_taper_start_idx;
        if (post_taper_len > 0)
        {
            convo::ScopedAlignedPtr<double> cos_args(static_cast<double*>(convo::aligned_malloc(post_taper_len * sizeof(double), 64)));
            double* post_window_vals = window_vals.get() + post_taper_start_idx;

            for (int i = 0; i < post_taper_len; ++i)
            {
                const double x_post = static_cast<double>(post_taper_start_idx + i - peakIndex) / dist_to_end;
                cos_args[i] = pi * ((x_post - (1.0 - alpha_post)) / alpha_post);
            }

            vdCos(post_taper_len, cos_args.get(), post_window_vals);

            for (int i = 0; i < post_taper_len; ++i)
                post_window_vals[i] = 0.5 * (1.0 + post_window_vals[i]);
        }
    }

    // Apply window
    vdMul(numSamples, data, window_vals.get(), data);
#else
    // 3. 窓関数の適用
    for (int i = 0; i < numSamples; ++i)
    {
        double window_val = 1.0;
        if (i < peakIndex)
        {
            // --- 左側 (開始点からピークまで) ---
            if (peakIndex > 0)
            {
                double x_pre = static_cast<double>(i) / static_cast<double>(peakIndex);
                if (x_pre < alpha_pre)
                {
                    window_val = 0.5 * (1.0 + std::cos(pi * (x_pre / alpha_pre - 1.0)));
                }
            }
        }
        else
        {
            // --- 右側 (ピークから終了点まで) ---
            if (dist_to_end > 0)
            {
                double x_post = static_cast<double>(i - peakIndex) / dist_to_end;
                if (x_post > (1.0 - alpha_post))
                {
                    double phase = (x_post - (1.0 - alpha_post)) / alpha_post;
                    window_val = 0.5 * (1.0 + std::cos(pi * phase));
                }
            }
        }
        data[i] *= window_val;
    }
#endif
}

// 2の累乗へ切り上げ (Helper)
static inline int nextPow2(int x)
{
    if (x <= 0) return 1;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

struct ConvolverSizing
{
    int firstPartition;
    int maxFFTSize;
};

// マスターリング専用 sizing 計算
static inline ConvolverSizing computeMasteringSizing(int internalBlockSize, int irLength)
{
    ConvolverSizing s{};

    // FP = nextPow2(internalBlock * 4)
    // FPは 4096〜16384 に制限（キャッシュ最適帯域）
    int fp = nextPow2(internalBlockSize * 4);
    fp = std::clamp(fp, 4096, 16384);
    s.firstPartition = fp;

    // MFS = nextPow2(clamp(irInternal / 4, FP, 131072))
    int mfsBase = irLength / 4;
    constexpr int kMFSUpper = 131072;
    mfsBase = std::clamp(mfsBase, s.firstPartition, kMFSUpper);
    s.maxFFTSize = nextPow2(mfsBase);

    // WDL安全制約
    if (s.maxFFTSize < s.firstPartition)
        s.maxFFTSize = s.firstPartition;
    if (s.maxFFTSize < internalBlockSize)
        s.maxFFTSize = nextPow2(internalBlockSize);

    return s;
}

//--------------------------------------------------------------
// 高精度型 DC Blocker (1次IIR)
// 超高サンプリングレート（OSR）対応
//--------------------------------------------------------------
class UltraHighRateDCBlocker {
private:
    double m_prev_x = 0.0;
    double m_prev_y = 0.0;
    double m_R = 0.999999; // デフォルト値

public:
    // サンプリングレートに合わせて R を計算
    void init(double sampleRate, double cutoffHz) {
        // R = exp(-2 * PI * cutoff / sampleRate)
        m_R = std::exp(-2.0 * juce::MathConstants<double>::pi * cutoffHz / sampleRate);
    }

    // 64byteアライメントされたバッファを高速処理
    void process(double* data, int numSamples) {
        double px = m_prev_x;
        double py = m_prev_y;
        double r = m_R;
        constexpr double kDenormalThreshold = 1.0e-20;

        for (int i = 0; i < numSamples; ++i) {
            double curr_x = data[i];
            // 高精度演算 (64bit double)
            double curr_y = curr_x - px + r * py;

            if (std::abs(curr_y) < kDenormalThreshold) curr_y = 0.0;

            px = curr_x;
            py = curr_y;
            data[i] = curr_y;
        }
        m_prev_x = px;
        m_prev_y = py;
    }
};

//--------------------------------------------------------------
// LoaderThread クラス定義
// IRの読み込み、処理、State作成をバックグラウンドで行う
//--------------------------------------------------------------
class ConvolverProcessor::LoaderThread : public juce::Thread
{
public:
    // ファイルからロードする場合のコンストラクタ
    LoaderThread(ConvolverProcessor& p, const juce::File& f, double sr, int bs, bool minPhase)
        : Thread("IRLoader"), owner(p), weakOwner(&p), file(f), sampleRate(sr), blockSize(bs), useMinPhase(minPhase), isRebuild(false)
    {}

    // メモリからリビルドする場合のコンストラクタ
    LoaderThread(ConvolverProcessor& p, const juce::AudioBuffer<double>& src, double srcSR, double sr, int bs, bool minPhase)
        : Thread("IRRebuilder"), owner(p), weakOwner(&p), sourceIR(src), sourceSampleRate(srcSR), sampleRate(sr), blockSize(bs), useMinPhase(minPhase), isRebuild(true)
    {}

    ~LoaderThread() override
    {
        stopThread(4000);
    }

    std::function<bool()> externalCancellationCheck;

    struct LoadResult
    {
        juce::AudioBuffer<double> loadedIR;
        double loadedSR = 0.0;
        int targetLength = 0;
        juce::AudioBuffer<double> displayIR;
        StereoConvolver::Ptr newConv;
        bool success = false;
        juce::String errorMessage;
    };

    void run() override
    {
        juce::ScopedNoDenormals noDenormals; // バックグラウンド処理でのDenormal対策

#if JUCE_INTEL
        // MKL/AVX最適化のためにFTZ/DAZフラグを明示的に設定
        // ScopedNoDenormalsでも設定されるが、MKLの要件として明示しておく
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

#if JUCE_DSP_USE_INTEL_MKL
        // VML (Vector Math Library) のDenormal扱いをゼロに設定
        // vdHypot, vdLn 等のパフォーマンス低下を防ぐ
        vmlSetMode(VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
#endif

        // メモリ確保失敗時の例外処理: std::terminate() を防ぐために try-catch で囲む
        // 早期終了時にフラグを確実にリセットするためのRAIIヘルパー
        struct FlagResetter {
            ConvolverProcessor& p;
            juce::WeakReference<ConvolverProcessor> weakP;
            const juce::Thread& t;
            bool success = false;
            ~FlagResetter() {
                if (!success && !t.threadShouldExit()) { // 正常終了またはスレッド中断以外の場合
                    auto wp = weakP;
                    juce::MessageManager::callAsync([wp] {
                        if (auto* o = wp.get()) {
                            o->isLoading.store(false);
                            o->isRebuilding.store(false);
                        }
                    });
                }
            }
        } resetter { owner, weakOwner, *this };

        LoadResult result = performLoad(this);

        if (result.success && !threadShouldExit())
        {
            // 6. メインスレッドで適用
            auto wp = weakOwner;

            // shared_ptrで管理 (Lambdaコピー時のAudioBufferディープコピー回避)
            auto loadedIRPtr = std::make_shared<juce::AudioBuffer<double>>(std::move(result.loadedIR));
            auto displayIRPtr = std::make_shared<juce::AudioBuffer<double>>(std::move(result.displayIR));
            StereoConvolver::Ptr newConvPtr = result.newConv;

            juce::MessageManager::callAsync([wp, newConvPtr, loadedIRPtr, loadedSR = result.loadedSR, targetLength = result.targetLength, isRebuild = this->isRebuild, file = this->file, displayIRPtr]()
            {
                if (auto* o = wp.get())
                {
                    o->applyNewState(newConvPtr, *loadedIRPtr, loadedSR, targetLength, isRebuild, file, *displayIRPtr);
                }
            });

            resetter.success = true;
        }
        else if (!result.success && result.errorMessage.isNotEmpty() && !threadShouldExit())
        {
            // エラー発生時: メインスレッドでエラー処理を行う
            auto wp = weakOwner;
            const juce::String error = result.errorMessage;
            juce::MessageManager::callAsync([wp, error]()
            {
                if (auto* o = wp.get())
                {
                    o->handleLoadError(error);
                }
            });
        }
    }

    LoadResult performLoad(juce::Thread* thread)
    {
        LoadResult result;

        // キャンセル判定用ラムダ: スレッド自身の終了フラグ または 外部コールバックをチェック
        auto shouldStop = [thread, this]() -> bool {
            if (thread && thread->threadShouldExit()) return true;
            if (externalCancellationCheck && externalCancellationCheck()) return true;
            return false;
        };

        try
        {
            owner.setLoadingProgress(0.0f);

            // 1. IRデータの取得 (ファイル読み込み or メモリコピー)
            if (isRebuild)
            {
                result.loadedIR = std::move(sourceIR); // 最適化: コピーではなくムーブ
                result.loadedSR = sourceSampleRate;
            }
            else
            {
                if (!file.existsAsFile()) return result;

                juce::AudioFormatManager formatManager;
                formatManager.registerBasicFormats();
                std::unique_ptr<juce::AudioFormatReader> reader(formatManager.createReaderFor(file));

                if (!reader) return result;

                // サイズの妥当性チェック (lengthInSamples が int の範囲を超える場合への対策)
                const int64 fileLength = reader->lengthInSamples;
                const int numChannels = static_cast<int>(reader->numChannels);
                static constexpr int64 MAX_FILE_LENGTH = 2147483647;  // int の最大値

                if (fileLength > MAX_FILE_LENGTH) {
                    DBG("LoaderThread: ファイルサイズが大きすぎます。");
                    return result;
                }
                if (numChannels <= 0) {
                    DBG("LoaderThread: チャンネル数が不正です。");
                    return result;
                }

                // AudioFormatReader::read は float のみ対応のため、一時バッファを使用
                juce::AudioBuffer<float> tempFloatBuffer(numChannels, static_cast<int>(fileLength));
                reader->read(&tempFloatBuffer, 0, static_cast<int>(fileLength), 0, true, true);

                // convo::input_transform::convertFloatToDoubleHighQuality はアライメント済みストア命令(_mm256_store_pd)を使用するため、
                // 出力先バッファは32byteアライメントされている必要がある。
                // juce::AudioBuffer はアライメントを保証しないため、一時的なアライメント済みバッファに変換後、コピーする。
                convo::ScopedAlignedPtr<double> tempAlignedBuffer(static_cast<double*>(convo::aligned_malloc(
                    static_cast<size_t>(fileLength) * sizeof(double), 64)));

                if (!tempAlignedBuffer)
                {
                    result.errorMessage = "Failed to allocate temporary buffer for IR loading.";
                    DBG("LoaderThread: " << result.errorMessage);
                    return result;
                }

                result.loadedIR.setSize(numChannels, static_cast<int>(fileLength));
                for (int ch = 0; ch < numChannels; ++ch)
                {
                    const float* src = tempFloatBuffer.getReadPointer(ch);
                    // アライメント済みの一時バッファに変換
                    convo::input_transform::convertFloatToDoubleHighQuality(src, tempAlignedBuffer.get(), static_cast<int>(fileLength));
                    // 結果を juce::AudioBuffer にコピー
                    result.loadedIR.copyFrom(ch, 0, tempAlignedBuffer.get(), static_cast<int>(fileLength));
                }
                result.loadedSR = reader->sampleRate;
            }

            if (checkCancellation(shouldStop, nullptr) || result.loadedIR.getNumSamples() == 0 || result.loadedIR.getNumChannels() == 0) return result;

            // 2. Auto Makeup (Energy Normalization)
            // IRのエネルギー(RMS)を測定し、入力信号と出力信号の音量感が一致するようにゲインを自動補正する。
            // その後、-6dBの安全マージンを適用する。
            // リビルド時は既に正規化されていると仮定するためスキップします。
            if (!isRebuild)
            {
                double maxChannelEnergy = 0.0;
                for (int ch = 0; ch < result.loadedIR.getNumChannels(); ++ch)
                {
                    const double* data = result.loadedIR.getReadPointer(ch);
#if JUCE_DSP_USE_INTEL_MKL
                    double energy = cblas_ddot(result.loadedIR.getNumSamples(), data, 1, data, 1);
#else
                    double energy = 0.0;
                    for (int i = 0; i < result.loadedIR.getNumSamples(); ++i)
                        energy += data[i] * data[i];
#endif

                    if (energy > maxChannelEnergy)
                        maxChannelEnergy = energy;
                }

                if (maxChannelEnergy > 1.0e-18)
                {
                    // Makeup Gain = 1.0 / RMS_IR
                    // Safety Margin = 0.501187 (-6.0dB)
                    const double makeup = 1.0 / std::sqrt(maxChannelEnergy);
                    const double safetyMargin = 0.5011872336272722;
                    result.loadedIR.applyGain(makeup * safetyMargin);
                }
            }

            // 3. 末尾の無音カット (Denormal対策 & 効率化)
            // IR末尾の極小値(Denormal領域)をカットすることで、畳み込み負荷とDenormal発生リスクを低減
            if (result.loadedIR.getNumSamples() > 0)
            {
                int newLength = result.loadedIR.getNumSamples();
                const int channels = result.loadedIR.getNumChannels();
                const double threshold = 1.0e-15; // -300dB (double精度における実質的な無音)

                while (newLength > 0)
                {
                    bool isSilent = true;
                    for (int ch = 0; ch < channels; ++ch)
                    {
                        if (std::abs(result.loadedIR.getSample(ch, newLength - 1)) > threshold)
                        {
                            isSilent = false;
                            break;
                        }
                    }
                    if (!isSilent) break;
                    newLength--;
                }

                if (newLength < result.loadedIR.getNumSamples())
                {
                    result.loadedIR.setSize(channels, std::max(1, newLength), true);
                    shrinkToFit(result.loadedIR); // 末尾カット後の余分なメモリを解放
                }
            }

            // 4. リサンプリング (SR不一致の場合)
            // IRのサンプルレートがターゲットと異なる場合、ピッチズレを防ぐためにリサンプリングする
            if (result.loadedSR > 0.0 && sampleRate > 0.0 &&
                std::abs(result.loadedSR - sampleRate) > 1e-6)
            {
                auto resampled = resampleIR(result.loadedIR, result.loadedSR, sampleRate, shouldStop);

                if (resampled.getNumSamples() == 0)
                {
                    // キャンセルされたか、エラーで0長になった場合
                    if (!checkCancellation(shouldStop, nullptr))
                    {
                        DBG("LoaderThread: Resampling failed (produced 0 samples or overflow).");
                    }
                    return result;
                }

                result.loadedIR = std::move(resampled);
                result.loadedSR = sampleRate;
            }

            // 5. 高精度型 DC Blocker (1次IIR)
            // WDLコンボルバー直前に置くため、位相回転を最小限に抑えつつDCを除去する
            // 超高サンプリングレート（OSR）対応
            if (result.loadedSR > 0.0 && result.loadedIR.getNumSamples() > 0)
            {
                for (int ch = 0; ch < result.loadedIR.getNumChannels(); ++ch)
                {
                    UltraHighRateDCBlocker dcBlocker;
                    // カットオフ周波数は 1.0Hz に設定 (超低域ノイズ除去)
                    dcBlocker.init(result.loadedSR, 1.0);

                    double* data = result.loadedIR.getWritePointer(ch);
                    const int numSamples = result.loadedIR.getNumSamples();
                    dcBlocker.process(data, numSamples);
                }
            }

            if (checkCancellation(shouldStop, nullptr)) return result;

            // 6. Asymmetric Tukey Window (Peak-based)
            // IRデータの先頭と末尾を滑らかにする「ピーク位置基準の非対称tukey窓」を適用
            if (result.loadedIR.getNumSamples() > 0)
            {
                const int numSamples = result.loadedIR.getNumSamples();
                for (int ch = 0; ch < result.loadedIR.getNumChannels(); ++ch)
                {
                    applyAsymmetricTukey(result.loadedIR.getWritePointer(ch), numSamples);
                }
            }

            if (checkCancellation(shouldStop, nullptr)) return result;

            if (checkCancellation(shouldStop, nullptr)) return result;

            // 7. ターゲット長計算とトリミング
            result.targetLength = owner.computeTargetIRLength(sampleRate, result.loadedIR.getNumSamples());
            juce::AudioBuffer<double> trimmed(result.loadedIR.getNumChannels(), result.targetLength);
            trimmed.clear();

            int copySamples = (std::min)(result.targetLength, result.loadedIR.getNumSamples());
            for (int ch = 0; ch < result.loadedIR.getNumChannels(); ++ch)
            {
                trimmed.copyFrom(ch, 0, result.loadedIR, ch, 0, copySamples);
                // フェードアウト
                int fade = 256;
                if (copySamples > fade)
                    trimmed.applyGainRamp(ch, copySamples - fade, fade, 1.0, 0.0);
            }

            if (checkCancellation(shouldStop, nullptr)) return result;

            // 8. MinPhase変換 (オプション)
            bool conversionSuccessful = false;
            if (useMinPhase)
            {
                bool wasCancelled = false;
                auto minPhaseIR = convertToMinimumPhase(trimmed, shouldStop, &wasCancelled);

                if (wasCancelled) return result;

                // 変換成功チェック: 全サンプルが有限値かつ十分なエネルギーがある場合のみ適用
                bool allFinite = (minPhaseIR.getNumSamples() > 0 && minPhaseIR.getNumChannels() > 0);
                double maxAbs = 0.0;
                if (allFinite)
                {
                    for (int ch = 0; ch < minPhaseIR.getNumChannels() && allFinite; ++ch)
                    {
                        const double* ptr = minPhaseIR.getReadPointer(ch);
                        for (int i = 0; i < minPhaseIR.getNumSamples(); ++i)
                        {
                            const double v = ptr[i];
                            if (!std::isfinite(v))
                            {
                                allFinite = false;
                                break;
                            }
                            maxAbs = (std::max)(maxAbs, std::abs(v));
                        }
                    }
                }

                if (allFinite && maxAbs > 1.0e-12)
                {
                    trimmed = minPhaseIR;
                    conversionSuccessful = true;
                }
                // 変換に失敗または無音になった場合は、元のtrimmed(Linear Phase)を使用する
            }

            if (checkCancellation(shouldStop, nullptr)) return result;

            // 9.ピーク位置検出 (レイテンシー補正用)
            // Linear Phaseの場合、ピークが遅れてやってくるため、その分Dryを遅らせる必要がある
            // MinPhase変換に失敗した場合も、Linear Phaseとして扱う必要があるためピーク検出を行う
            int irPeakLatency = 0;
            if (trimmed.getNumChannels() > 0)
            {
                if (!useMinPhase || (useMinPhase && !conversionSuccessful))
                {
                    // 全チャンネルの中で最大振幅を持つサンプルの位置を探す
                    double maxMag = 0.0;
                    for (int ch = 0; ch < trimmed.getNumChannels(); ++ch)
                    {
                        const double* data = trimmed.getReadPointer(ch);
                        for (int i = 0; i < result.targetLength; ++i)
                        {
                            double mag = std::abs(data[i]);
                            if (mag > maxMag)
                            {
                                maxMag = mag;
                                irPeakLatency = i;
                            }
                        }
                    }
                }
            }

            // 10. 新しいConvolutionの構築 (Non-uniform Partitioned Convolution)
            result.newConv = std::make_shared<StereoConvolver>();

            // IRデータを格納するアラインされたバッファを準備 (Rebuild用に保持)
            convo::ScopedAlignedPtr<double> irL(static_cast<double*>(convo::aligned_malloc(result.targetLength * sizeof(double), 64)));
            convo::ScopedAlignedPtr<double> irR(static_cast<double*>(convo::aligned_malloc(result.targetLength * sizeof(double), 64)));

            // 安全対策: チャンネル数チェック
            if (trimmed.getNumChannels() == 0) return result;

            const double* srcL = trimmed.getReadPointer(0);
            const double* srcR = (trimmed.getNumChannels() > 1) ? trimmed.getReadPointer(1) : srcL;

            // データを一度だけコピー
            std::memcpy(irL.get(), srcL, result.targetLength * sizeof(double));
            std::memcpy(irR.get(), srcR, result.targetLength * sizeof(double));

            // 10. 新しいConvolutionの構築 (initメソッドを使用して安全に初期化)
            // prepareToPlayとロジックを統一し、WDLエンジンのプリウォーミングも行う
            int internalBlockSize = juce::nextPowerOfTwo(blockSize);
            auto sizing = computeMasteringSizing(internalBlockSize, result.targetLength);

            result.newConv->init(irL.release(), irR.release(), result.targetLength, sampleRate, irPeakLatency, sizing.maxFFTSize, internalBlockSize, sizing.firstPartition, blockSize, owner.isMklEnabled());

            // Display用コピーを作成 (move前に)
            if (owner.isVisualizationEnabled())
                result.displayIR = trimmed;

            if (checkCancellation(shouldStop, nullptr)) return result;

            result.success = true;
            return result;
        }
        catch (const std::bad_alloc&)
        {
            result.errorMessage = "IR too large (Out of Memory)";
            DBG("LoaderThread: " << result.errorMessage);
            return result;
        }
        catch (const std::exception& e)
        {
            result.errorMessage = "Error loading IR: " + juce::String(e.what());
            DBG("LoaderThread: " << result.errorMessage);
            return result;
        }
        catch (...)
        {
            result.errorMessage = "Unknown error loading IR";
            DBG("LoaderThread: " << result.errorMessage);
            return result;
        }
    }

    void runSynchronously()
    {
        juce::ScopedNoDenormals noDenormals;
        // 同期実行のため、スレッドキャンセルチェックは行わない (nullptrを渡す)
        LoadResult result = performLoad(nullptr);

        if (result.success)
        {
            owner.applyNewState(result.newConv, result.loadedIR, result.loadedSR, result.targetLength, isRebuild, file, result.displayIR);
        }
    }
private:
    ConvolverProcessor& owner;
    juce::WeakReference<ConvolverProcessor> weakOwner;
    juce::File file;
    juce::AudioBuffer<double> sourceIR;
    double sourceSampleRate = 0.0;
    double sampleRate;
    int blockSize;
    bool useMinPhase;
    bool isRebuild;
};

//--------------------------------------------------------------
// コンストラクタ
//--------------------------------------------------------------
ConvolverProcessor::ConvolverProcessor()
    : mixSmoother(1.0f)
{
    startTimer(500);
}

//--------------------------------------------------------------
// デストラクタ
//--------------------------------------------------------------
ConvolverProcessor::~ConvolverProcessor()
{
    stopTimer();
    forceCleanup();
    // スレッドを停止
    activeLoader.reset();
    convolution.store(nullptr);
    activeConvolution.reset();

#if JUCE_DSP_USE_INTEL_MKL
    if (fftHandle) {
        DftiFreeDescriptor(&fftHandle);
        fftHandle = nullptr;
    }
#endif
}

void ConvolverProcessor::timerCallback()
{
    cleanup();
}

//--------------------------------------------------------------
// prepareToPlay
//--------------------------------------------------------------
void ConvolverProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
#if JUCE_DSP_USE_INTEL_MKL
    // 旧descriptor未解放防止
    if (fftHandle) {
        DftiFreeDescriptor(&fftHandle);
        fftHandle = nullptr;
        fftHandleSize = 0;
    }
#endif

    const bool rateChanged = (std::abs(currentSampleRate.load() - sampleRate) > 1e-6);
    const bool blockChanged = (currentBufferSize != samplesPerBlock);

    currentBufferSize = samplesPerBlock;

    // 最初にサンプルレートを更新（oldValueを保存）
    currentSampleRate.store(sampleRate, std::memory_order_release);

    // ProcessSpec設定
    juce::dsp::ProcessSpec spec;
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = static_cast<juce::uint32>(MAX_BLOCK_SIZE);
    spec.numChannels = 2;  // ステレオ

    currentSpec = spec;

    // 既存のコンボリューション状態の確認
    auto* conv = convolution.load(std::memory_order_acquire);
    if (conv) {
        // FIX: Oversampling x8時のblockSize/partitionSize不整合対策
        // WDL_ConvolutionEngine_Div に正しい knownBlockSize を渡すために、エンジンを再構築する。
        // 既存のエンジンは他スレッドで共有されている可能性があるため、複製して差し替える。
        const int internalBlockSize = juce::nextPowerOfTwo(samplesPerBlock);

        if ((rateChanged || blockChanged) && conv->irDataLength > 0)
        {
            auto newConv = std::make_shared<StereoConvolver>();

            // バッファ確保とコピー
            convo::ScopedAlignedPtr<double> irL(static_cast<double*>(convo::aligned_malloc(conv->irDataLength * sizeof(double), 64)));
            convo::ScopedAlignedPtr<double> irR(static_cast<double*>(convo::aligned_malloc(conv->irDataLength * sizeof(double), 64)));
            std::memcpy(irL.get(), conv->irData[0], conv->irDataLength * sizeof(double));
            std::memcpy(irR.get(), conv->irData[1], conv->irDataLength * sizeof(double));

            auto sizing = computeMasteringSizing(internalBlockSize, conv->irDataLength);

            // 新しいブロックサイズで初期化
            newConv->init(irL.release(), irR.release(),
                          conv->irDataLength, sampleRate, conv->irLatency, sizing.maxFFTSize, internalBlockSize, sizing.firstPartition, samplesPerBlock, enableMKL.load());

            // 差し替え
            convolution.store(newConv.get(), std::memory_order_release);

            if (activeConvolution)
            {
                const juce::ScopedLock sl(trashBinLock);
                trashBin.push_back({activeConvolution, juce::Time::getMillisecondCounter()});
            }
            activeConvolution = newConv;
        }
    }

    // DelayLine準備
    if (delayBufferCapacity < DELAY_BUFFER_SIZE)
    {
        delayBuffer[0].reset(static_cast<double*>(convo::aligned_malloc(DELAY_BUFFER_SIZE * sizeof(double), 64)));
        delayBuffer[1].reset(static_cast<double*>(convo::aligned_malloc(DELAY_BUFFER_SIZE * sizeof(double), 64)));
        delayBufferCapacity = DELAY_BUFFER_SIZE;
    }
    // バッファクリア
    juce::FloatVectorOperations::clear(delayBuffer[0].get(), DELAY_BUFFER_SIZE);
    juce::FloatVectorOperations::clear(delayBuffer[1].get(), DELAY_BUFFER_SIZE);
    delayWritePos = 0;

    // Dryバッファ確保
    if (dryBufferCapacity < MAX_BLOCK_SIZE)
    {
        dryBufferStorage[0].reset(static_cast<double*>(convo::aligned_malloc(MAX_BLOCK_SIZE * sizeof(double), 64)));
        dryBufferStorage[1].reset(static_cast<double*>(convo::aligned_malloc(MAX_BLOCK_SIZE * sizeof(double), 64)));
        dryBufferCapacity = MAX_BLOCK_SIZE;
    }
    double* dryChs[2] = { dryBufferStorage[0].get(), dryBufferStorage[1].get() };
    dryBuffer.setDataToReferTo(dryChs, 2, MAX_BLOCK_SIZE);
    dryBuffer.clear();

    if (smoothingBufferCapacity < MAX_BLOCK_SIZE)
    {
        smoothingBufferStorage[0].reset(static_cast<double*>(convo::aligned_malloc(MAX_BLOCK_SIZE * sizeof(double), 64)));
        smoothingBufferStorage[1].reset(static_cast<double*>(convo::aligned_malloc(MAX_BLOCK_SIZE * sizeof(double), 64)));
        smoothingBufferCapacity = MAX_BLOCK_SIZE;
    }
    double* smoothChs[2] = { smoothingBufferStorage[0].get(), smoothingBufferStorage[1].get() };
    smoothingBuffer.setDataToReferTo(smoothChs, 2, MAX_BLOCK_SIZE);
    smoothingBuffer.clear();

    if (oldDryBufferCapacity < MAX_BLOCK_SIZE)
    {
        oldDryBufferStorage[0].reset(static_cast<double*>(convo::aligned_malloc(MAX_BLOCK_SIZE * sizeof(double), 64)));
        oldDryBufferStorage[1].reset(static_cast<double*>(convo::aligned_malloc(MAX_BLOCK_SIZE * sizeof(double), 64)));
        oldDryBufferCapacity = MAX_BLOCK_SIZE;
    }
    double* oldDryChs[2] = { oldDryBufferStorage[0].get(), oldDryBufferStorage[1].get() };
    oldDryBuffer.setDataToReferTo(oldDryChs, 2, MAX_BLOCK_SIZE);
    oldDryBuffer.clear();

    // Wetバッファ確保
    if (wetBufferCapacity < MAX_BLOCK_SIZE)
    {
        wetBufferStorage[0].reset(static_cast<double*>(convo::aligned_malloc(MAX_BLOCK_SIZE * sizeof(double), 64)));
        wetBufferStorage[1].reset(static_cast<double*>(convo::aligned_malloc(MAX_BLOCK_SIZE * sizeof(double), 64)));
        wetBufferCapacity = MAX_BLOCK_SIZE;
    }
    juce::FloatVectorOperations::clear(wetBufferStorage[0].get(), MAX_BLOCK_SIZE);
    juce::FloatVectorOperations::clear(wetBufferStorage[1].get(), MAX_BLOCK_SIZE);

    // スムージング時間の設定
    currentSmoothingTimeSec = smoothingTimeSec.load();
    mixSmoother.reset(sampleRate, currentSmoothingTimeSec);
    // 初期化: 現在のターゲット値を設定し、不要なフェードインや未初期化状態を防ぐ
    mixSmoother.setCurrentAndTargetValue(static_cast<double>(mixTarget.load()));
    // ダミー呼び出し: 内部状態の確実な初期化 (メモリ確保リスクの排除)
    (void)mixSmoother.getNextValue();

    // レイテンシースムーサーの初期化
    // 100msのスムージング時間でクリックノイズを防止
    latencySmoother.reset(sampleRate, 0.1);
    // ドップラー効果対策のクロスフェード用 (20ms)
    crossfadeGain.reset(sampleRate, 0.02);
    crossfadeGain.setCurrentAndTargetValue(1.0);

    // 既にIRがロードされている場合は、初期値をそのレイテンシーに合わせる (起動時のスライド防止)
    if (conv)
    {
        const int initialLatency = juce::jmin(conv->latency + conv->irLatency, MAX_TOTAL_DELAY);
        latencySmoother.setCurrentAndTargetValue(static_cast<double>(initialLatency));
    }
    else
    {
        latencySmoother.setCurrentAndTargetValue(0.0);
    }
    oldDelay = latencySmoother.getTargetValue();

    isPrepared.store(true, std::memory_order_release);
}

void ConvolverProcessor::releaseResources()
{
    forceCleanup();
    // 【パッチ2】LoaderThread を先に停止し、解放後の非同期コールバックを防ぐ
    // activeLoader.reset() → stopThread(4000) → ~LoaderThread() の順で安全に停止される。
    // これを省略すると、ローダーが releaseResources() 完了後に callAsync() で
    // convolution ポインタや isPrepared フラグを書き換え、Use-After-Free の原因になる。
    activeLoader.reset();

    // バッファの解放
    delayBuffer[0].reset();
    delayBuffer[1].reset();
    delayBufferCapacity = 0;

    dryBufferStorage[0].reset();
    dryBufferStorage[1].reset();
    dryBufferCapacity = 0;

    oldDryBufferStorage[0].reset();
    oldDryBufferStorage[1].reset();
    oldDryBufferCapacity = 0;

    smoothingBufferStorage[0].reset();
    smoothingBufferStorage[1].reset();
    smoothingBufferCapacity = 0;

    cachedFFTBuffer.reset();
    cachedFFTBufferCapacity = 0;

    dryBuffer.setSize(0, 0);
    smoothingBuffer.setSize(0, 0);

#if JUCE_DSP_USE_INTEL_MKL
    if (fftHandle) {
        DftiFreeDescriptor(&fftHandle);
        fftHandle = nullptr;
        fftHandleSize = 0;
    }
#endif

    // Release active convolution engine
    convolution.store(nullptr, std::memory_order_release);
    activeConvolution.reset();

    {
        const juce::ScopedLock sl(trashBinLock);
        trashBin.clear();
    }

    isPrepared.store(false, std::memory_order_release);
}

void ConvolverProcessor::reset()
{
    auto* conv = convolution.load(std::memory_order_acquire);
    if (conv)
    {
        conv->reset();
    }
    // リングバッファのクリア
    if (delayBuffer[0]) juce::FloatVectorOperations::clear(delayBuffer[0].get(), DELAY_BUFFER_SIZE);
    if (delayBuffer[1]) juce::FloatVectorOperations::clear(delayBuffer[1].get(), DELAY_BUFFER_SIZE);
    delayWritePos = 0;

    dryBuffer.clear();
    smoothingBuffer.clear();
    mixSmoother.setCurrentAndTargetValue(static_cast<double>(mixTarget.load()));
    latencySmoother.setCurrentAndTargetValue(latencySmoother.getTargetValue());
}

void ConvolverProcessor::rebuildAllIRs()
{
    if (isIRLoaded() && !isLoading.load())
    {
        // リビルドモードでロード (現在のoriginalIRを使用)
        // Message Threadから呼ばれることを想定
        loadImpulseResponse(juce::File(), false);
    }
}

void ConvolverProcessor::rebuildAllIRsSynchronous(std::function<bool()> shouldCancel)
{
    if (originalIR.getNumSamples() > 0 && originalIRSampleRate > 0.0)
    {
        // リビルドモードでローダーを作成し、同期的に実行
        LoaderThread loader(*this, originalIR, originalIRSampleRate, currentSpec.sampleRate, currentBufferSize, useMinPhase.load());
        loader.externalCancellationCheck = shouldCancel;
        loader.runSynchronously();
    }
}

//--------------------------------------------------------------
// StereoConvolver Copy Constructor
//--------------------------------------------------------------
//--------------------------------------------------------------
// Minimum Phase 変換ヘルパー
// ケプストラム法 (Homomorphic Filtering) による最小位相復元
// 目的: 振幅特性（周波数応答の絶対値）を保ったまま、エネルギーを時間軸の前方に集中させ、レイテンシーとプリリンギングを低減する。
// アルゴリズム手順:
//   1. FFT -> 周波数領域へ
//   2. 対数マグニチュード計算 (位相情報を捨てる)
//   3. IFFT -> ケプストラム領域 (Real Cepstrum) へ
//   4. 因果的ウィンドウ適用 (負の時間をゼロにし、正の時間を2倍にする) -> 最小位相ケプストラム
//   5. FFT -> 解析信号の対数スペクトルへ
//   6. 複素指数変換 (exp) -> 最小位相スペクトルへ
//   7. IFFT -> 時間領域の最小位相IRへ
//
// 精度向上:
//   JUCEのFFTはfloatのみですが、対数・指数演算や窓関数処理をdoubleで行うことで
//   計算誤差（特にexp時の発散や微小値の消失）を抑制します。
//--------------------------------------------------------------
// Note: この関数は LoaderThread (バックグラウンド) で実行されるため、FFTのメモリ確保や計算負荷はAudio Threadに影響しません。
static juce::AudioBuffer<double> convertToMinimumPhase(const juce::AudioBuffer<double>& linearIR, const std::function<bool()>& shouldExit, bool* wasCancelled)
{
    if (wasCancelled) *wasCancelled = false;

    const int numSamples = linearIR.getNumSamples();
    if (numSamples <= 0 || linearIR.getNumChannels() < 1) return {};
    // ゼロパディングを含めて十分なサイズを確保 (4倍程度が安全)
    const int fftSize = juce::nextPowerOfTwo(numSamples * 4);

    // メモリ使用量過多を防ぐためのFFTサイズ制限
    static constexpr int MAX_MINPHASE_FFT_SIZE = 2097152; // 2^21
    if (fftSize > MAX_MINPHASE_FFT_SIZE)
    {
        DBG("convertToMinimumPhase: fftSize (" << fftSize << ") exceeds limit. Skipping min-phase conversion to prevent excessive memory usage.");
        return {}; // 失敗/スキップを通知するために空のバッファを返す
    }

    juce::AudioBuffer<double> minPhaseIR(linearIR.getNumChannels(), numSamples);

#if JUCE_DSP_USE_INTEL_MKL
    // MKL DFTI は自然順序で扱えるため、WDL_fft の permute 順序問題を回避できる。
    DFTI_DESCRIPTOR_HANDLE dfti = nullptr;
    DftiGuard dftiGuard { &dfti };

    const MKL_LONG len = static_cast<MKL_LONG>(fftSize);
    // --- Descriptor Creation and Configuration ---
    // Each step is checked to prevent using an invalid handle.
    if (DftiCreateDescriptor(&dfti, DFTI_DOUBLE, DFTI_COMPLEX, 1, len) != DFTI_NO_ERROR)
    {
        DBG("convertToMinimumPhase: DftiCreateDescriptor failed.");
        return {};
    }

    if (DftiSetValue(dfti, DFTI_PLACEMENT, DFTI_INPLACE) != DFTI_NO_ERROR)
    {
        DBG("convertToMinimumPhase: DftiSetValue(DFTI_PLACEMENT) failed.");
        return {};
    }

    if (DftiSetValue(dfti, DFTI_BACKWARD_SCALE, 1.0 / static_cast<double>(fftSize)) != DFTI_NO_ERROR)
    {
        DBG("convertToMinimumPhase: DftiSetValue(DFTI_BACKWARD_SCALE) failed.");
        return {};
    }

    if (DftiCommitDescriptor(dfti) != DFTI_NO_ERROR)
    {
        DBG("convertToMinimumPhase: DftiCommitDescriptor failed.");
        return {};
    }

    convo::ScopedAlignedPtr<MKL_Complex16> spectrum(static_cast<MKL_Complex16*>(convo::aligned_malloc(
        static_cast<size_t>(fftSize) * sizeof(MKL_Complex16), 64)));
    if (!spectrum)
        return {};

    for (int ch = 0; ch < linearIR.getNumChannels(); ++ch)
    {
        if (checkCancellation(shouldExit, wasCancelled))
            return {};

        const double* src = linearIR.getReadPointer(ch);
        for (int i = 0; i < fftSize; ++i)
        {
            spectrum.get()[i].real = (i < numSamples) ? src[i] : 0.0;
            spectrum.get()[i].imag = 0.0;
        }

        // 1) FFT
        if (DftiComputeForward(dfti, spectrum.get()) != DFTI_NO_ERROR) {
            DBG("convertToMinimumPhase: DftiComputeForward (1) failed.");
            return {};
        }

        // 2) log|H(w)|
        {
            convo::ScopedAlignedPtr<double> mag(static_cast<double*>(convo::aligned_malloc(fftSize * sizeof(double), 64)));

            // Calculate magnitude: |H(w)|
            vzAbs(fftSize, spectrum.get(), mag.get());

            // Clamp to avoid log(0)
            for (int i = 0; i < fftSize; ++i)
                mag[i] = std::max(mag[i], 1.0e-300);

            // Calculate log magnitude: log|H(w)|
            vdLn(fftSize, mag.get(), mag.get());

            for (int i = 0; i < fftSize; ++i)
                { spectrum.get()[i].real = mag[i]; spectrum.get()[i].imag = 0.0; }
        }
        // 3) IFFT -> real cepstrum
        if (DftiComputeBackward(dfti, spectrum.get()) != DFTI_NO_ERROR) {
            DBG("convertToMinimumPhase: DftiComputeBackward (1) failed.");
            return {};
        }

        // 4) causal lifter
        const int half = fftSize / 2;
        spectrum.get()[0].imag = 0.0;
        for (int i = 1; i < half; ++i)
        {
            spectrum.get()[i].real *= 2.0;
            spectrum.get()[i].imag = 0.0;
        }
        spectrum.get()[half].imag = 0.0;
        for (int i = half + 1; i < fftSize; ++i)
        {
            spectrum.get()[i].real = 0.0;
            spectrum.get()[i].imag = 0.0;
        }

        // 5) FFT
        if (DftiComputeForward(dfti, spectrum.get()) != DFTI_NO_ERROR) {
            DBG("convertToMinimumPhase: DftiComputeForward (2) failed.");
            return {};
        }

        // 6) complex exp
        {
            // Clamp inputs to prevent overflow/underflow in vzExp
            for (int i = 0; i < fftSize; ++i)
            {
                spectrum.get()[i].real = juce::jlimit(-50.0, 50.0, spectrum.get()[i].real);
                spectrum.get()[i].imag = juce::jlimit(-50.0, 50.0, spectrum.get()[i].imag);
            }

            vzExp(fftSize, spectrum.get(), spectrum.get());

            for (int i = 0; i < fftSize; ++i)
                if (!std::isfinite(spectrum.get()[i].real) || !std::isfinite(spectrum.get()[i].imag)) return {};
        }
        // 7) IFFT -> minimum-phase IR
        if (DftiComputeBackward(dfti, spectrum.get()) != DFTI_NO_ERROR) {
            DBG("convertToMinimumPhase: DftiComputeBackward (2) failed.");
            return {};
        }

        double* dst = minPhaseIR.getWritePointer(ch);
        for (int i = 0; i < numSamples; ++i)
        {
            double v = spectrum.get()[i].real;
            if (!std::isfinite(v))
                return {};
            if (std::abs(v) < 1.0e-18)
                v = 0.0;
            dst[i] = v;
        }
    }

    return minPhaseIR;
#else
    // 非MKLビルドでは最小位相変換を無効化 (Linear Phaseへフォールバック)。
    return {};
#endif
}

//--------------------------------------------------------------
// loadImpulseResponse（Message Thread）
//--------------------------------------------------------------
bool ConvolverProcessor::loadImpulseResponse(const juce::File& irFile, bool optimizeForRealTime)
{
    // ファイル指定あり: 新規ロード
    // ファイル指定なし: 現在のデータでリビルド (SR変更時など)
    bool isRebuild = (irFile == juce::File());

    if (isRebuild)
    {
        if (isRebuilding.exchange(true, std::memory_order_acquire))
        {
            DBG("ConvolverProcessor::rebuild (via loadImpulseResponse) already in progress, skipping");
            return true;
        }
        if (originalIR.getNumSamples() == 0 || originalIRSampleRate <= 0.0)
        {
            isRebuilding.store(false, std::memory_order_release);
            return false;
        }
    }

    if (!isRebuild && !irFile.existsAsFile())
    {
        return false;
    }

    isLoading.store(true);
    lastError.clear(); // 新しいロード開始時にエラーをクリア

    // 既存のローダーを停止してゴミ箱へ退避 (即時resetによるブロックを回避)
    if (activeLoader)
    {
        activeLoader->signalThreadShouldExit();
        loaderTrashBin.push_back(std::move(activeLoader));
    }

    // 新しいローダーを作成して開始
    if (isRebuild)
    {
        activeLoader = std::make_unique<LoaderThread>(*this, originalIR, originalIRSampleRate, currentSpec.sampleRate, currentBufferSize, useMinPhase.load());
    }
    else
    {
        activeLoader = std::make_unique<LoaderThread>(*this, irFile, currentSpec.sampleRate, currentBufferSize, useMinPhase.load());
        currentIrOptimized.store(optimizeForRealTime);
    }

    activeLoader->startThread();

    return true;
}

//--------------------------------------------------------------
// applyNewState (Message Thread Callback)
// ローダースレッド完了後に呼ばれる
//--------------------------------------------------------------
void ConvolverProcessor::applyNewState(StereoConvolver::Ptr newConv,
                                       const juce::AudioBuffer<double>& loadedIR,
                                       double loadedSR,
                                       int targetLength,
                                       bool isRebuild,
                                       const juce::File& file,
                                       const juce::AudioBuffer<double>& displayIR)
{
    // 元データの更新 (新規ロード時のみ)
    if (!isRebuild)
    {
        originalIR = loadedIR;
        originalIRSampleRate = loadedSR;
        {
            const juce::ScopedLock sl(irFileLock);
            currentIrFile = file;
        }
        irName = file.getFileNameWithoutExtension();
    }

    // スナップショット更新 (表示用)
    // LoaderThreadで計算済みの displayIR (trimmed & min-phased) を使用
    if (visualizationEnabled)
    {
        createWaveformSnapshot(displayIR);
        // 表示用には現在のサンプルレートを使用 (loadedSRはリサンプリング後のレート)
        createFrequencyResponseSnapshot(displayIR, loadedSR);
    }

    // 安全に差し替え (Atomic Swap)
    convolution.store(newConv.get(), std::memory_order_release);

    if (activeConvolution)
    {
        DBG("ConvolverProcessor: Enqueueing old StereoConvolver to trash bin.");
        const juce::ScopedLock sl(trashBinLock);
        trashBin.push_back({activeConvolution, juce::Time::getMillisecondCounter()}); // 古いオブジェクトをゴミ箱へ
    }
    activeConvolution = newConv;

    // 現在の有効なIR長を更新
    irLength = targetLength;
    currentSampleRate.store(currentSpec.sampleRate);

    isLoading.store(false);
    isRebuilding.store(false, std::memory_order_release); // Reset rebuild flag
    sendChangeMessage();
}

void ConvolverProcessor::handleLoadError(const juce::String& error)
{
    lastError = error;
    isLoading.store(false);
    isRebuilding.store(false, std::memory_order_release);
    // UIに通知してエラーメッセージを表示させる
    sendChangeMessage();
}

void ConvolverProcessor::cleanup()
{
    // LoaderThread のクリーンアップ (Message Thread Only)
    // 終了したスレッドのみを削除する (waitForThreadToExit(0) はブロックしない)
    for (auto it = loaderTrashBin.begin(); it != loaderTrashBin.end(); )
    {
        if ((*it)->waitForThreadToExit(0))
            it = loaderTrashBin.erase(it);
        else
            ++it;
    }

    // 【Leak Fix】LoaderThreadの異常蓄積防止
    // スレッドが終了しない場合でも、一定数を超えたら強制削除してメモリを解放する。
    // [FIX] ~LoaderThread() は stopThread(4000) をブロック呼出するため、
    //       メッセージスレッド上で直接 erase() すると最大4秒UIが凍結しクラッシュする。
    //       超過スレッドは detach されたバックグラウンドスレッドで破棄する。
    while (loaderTrashBin.size() > 2)
    {
        std::unique_ptr<LoaderThread> staleLoader = std::move(loaderTrashBin.front());
        loaderTrashBin.erase(loaderTrashBin.begin());
        // ~LoaderThread() の stopThread(4000) をメッセージスレッドの外で実行する
        std::thread([l = std::move(staleLoader)]() mutable {
            l.reset(); // blocks in stopThread(4000) on background thread
        }).detach();
    }

    // StereoConvolver のクリーンアップ (Worker Threadと競合するためロックが必要)
    juce::ScopedTryLock lock(trashBinLock);
    if (!lock.isLocked()) return;

    const uint32 now = juce::Time::getMillisecondCounter();
    std::vector<StereoConvolver::Ptr> toRelease;

    for (auto it = trashBin.begin(); it != trashBin.end(); )
    {
        uint32 age = (now >= it->second) ?
                     (now - it->second) :
                     (std::numeric_limits<uint32>::max() - it->second + now);

        if (age > 800 || it->first.use_count() <= 1)
        {
            toRelease.push_back(std::move(it->first));
            it = trashBin.erase(it);
        }
        else
        {
            ++it;
        }
    }

    while (trashBin.size() > 3)
    {
        toRelease.push_back(std::move(trashBin.back().first));
        trashBin.pop_back();
    }
}

void ConvolverProcessor::forceCleanup()
{
    // This method is for eager cleanup of non-blocking resources.
    // The blocking cleanup of LoaderThreads is handled by the destructor.

    std::vector<std::pair<StereoConvolver::Ptr, uint32>> temp;
    {
        juce::ScopedLock lock(trashBinLock);
        temp.swap(trashBin);
    }
    // temp is destroyed here, releasing any old StereoConvolver instances.
}

//--------------------------------------------------------------
// computeTargetIRLength
// 1.0秒固定長を計算し、最大長で制限する
//--------------------------------------------------------------
int ConvolverProcessor::computeTargetIRLength(double sampleRate, int /*originalLength*/) const
{
    const double targetIRTimeSec = targetIRLengthSec.load();
    static constexpr int kMaxIRCap = MAX_IR_LATENCY;

    int target = static_cast<int>(sampleRate * targetIRTimeSec);

    target = (std::min)(target, kMaxIRCap);
    target = (std::max)(target, 1); // Ensure at least 1 sample

    return target;
}

//--------------------------------------------------------------
// applySmoothing (Helper)
// 1/6オクターブスムージングを適用する
//--------------------------------------------------------------
static void applySmoothing(std::vector<float>& magnitudes, int fftSize)
{
    if (magnitudes.empty()) return;

    std::vector<float> smoothed = magnitudes;
    const float bandwidth = 1.0f / 6.0f; // 1/6 octave
    const float factor = std::pow(2.0f, bandwidth * 0.5f);

    // DC(0)はスキップ
    for (size_t i = 1; i < magnitudes.size(); ++i)
    {
        float sum = 0.0f;
        int count = 0;

        // ウィンドウ範囲の決定
        int startBin = static_cast<int>(static_cast<float>(i) / factor);
        int endBin   = static_cast<int>(static_cast<float>(i) * factor);

        startBin = (std::max)(1, startBin); // DCを含めない
        endBin   = (std::min)(static_cast<int>(magnitudes.size()) - 1, endBin);

        for (int j = startBin; j <= endBin; ++j)
        {
            sum += magnitudes[j];
            count++;
        }

        if (count > 0)
            smoothed[i] = sum / static_cast<float>(count);
    }

    magnitudes = smoothed;
}

//--------------------------------------------------------------
// createWaveformSnapshot
//--------------------------------------------------------------
void ConvolverProcessor::createWaveformSnapshot (const juce::AudioBuffer<double>& irBuffer)
{
    irWaveform.assign(WAVEFORM_POINTS, 0.0f);

    const int numSamples = irBuffer.getNumSamples();
    const int numChannels = irBuffer.getNumChannels();

    if (numSamples <= 0 || numChannels <= 0)
        return;

    const int samplesPerPoint = (std::max)(1, numSamples / WAVEFORM_POINTS);

    float maxAbs = 0.0f;

    for (int i = 0; i < WAVEFORM_POINTS; ++i)
    {
        float peak = 0.0f;
        int startSample = i * samplesPerPoint;
        int endSample = (std::min)(numSamples, startSample + samplesPerPoint);

        // 全チャンネルのピークを取得
        for (int ch = 0; ch < numChannels; ++ch)
            for (int j = startSample; j < endSample; ++j)
                peak = (std::max)(peak, static_cast<float>(std::abs(irBuffer.getReadPointer(ch)[j])));

        irWaveform[i] = peak;
        maxAbs = (std::max)(maxAbs, peak);
    }

    // 正規化 (表示用)
    if (maxAbs > 0.0f)
        for (float& val : irWaveform) val /= maxAbs;
}

//--------------------------------------------------------------
// createFrequencyResponseSnapshot
// IRの周波数特性（マグニチュード）を計算する
//--------------------------------------------------------------
void ConvolverProcessor::createFrequencyResponseSnapshot(const juce::AudioBuffer<double>& irBuffer, double sampleRate)
{
    irSpectrumSampleRate = sampleRate;
    irMagnitudeSpectrum.clear();

    const int numSamples = irBuffer.getNumSamples();
    if (numSamples <= 0 || irBuffer.getNumChannels() < 1) return;

    // IRの長さに応じてFFTサイズを決定 (固定サイズではなく適応させる)
    // ただし、極端に巨大なIRの場合はパフォーマンスを考慮して上限を設ける (例: 65536)
    int fftSize = juce::nextPowerOfTwo(numSamples);
    const int maxFFTSize = 65536;
    if (fftSize > maxFFTSize) fftSize = maxFFTSize;
    if (fftSize < 512) fftSize = 512;

    // キャッシュされたバッファを再利用 (メモリ確保のオーバーヘッド削減)
    if (cachedFFTBufferCapacity < fftSize * 2)
    {
        cachedFFTBuffer.reset(static_cast<float*>(convo::aligned_malloc(fftSize * 2 * sizeof(float), 64)));
        cachedFFTBufferCapacity = fftSize * 2;
    }

    juce::FloatVectorOperations::clear(cachedFFTBuffer.get(), fftSize * 2);

    // チャンネル0 (Lch) の特性を使用する
    const double* src = irBuffer.getReadPointer(0);
    const int copyLen = (std::min)(numSamples, fftSize);
    float* dst = cachedFFTBuffer.get();

#if JUCE_DSP_USE_INTEL_MKL
    // MKL FFT (One-shot)
    if (fftHandle && fftHandleSize != fftSize)
    {
        DftiFreeDescriptor(&fftHandle);
        fftHandle = nullptr;
        fftHandleSize = 0;
    }

    if (!fftHandle)
    {
        if (DftiCreateDescriptor(&fftHandle, DFTI_SINGLE, DFTI_COMPLEX, 1, fftSize) != DFTI_NO_ERROR) return;
        if (DftiSetValue(fftHandle, DFTI_PLACEMENT, DFTI_INPLACE) != DFTI_NO_ERROR) { DftiFreeDescriptor(&fftHandle); fftHandle = nullptr; return; }
        if (DftiCommitDescriptor(fftHandle) != DFTI_NO_ERROR) { DftiFreeDescriptor(&fftHandle); fftHandle = nullptr; return; }
        fftHandleSize = fftSize;
    }

    // Double -> Complex Float conversion
    for (int i = 0; i < copyLen; ++i) {
        dst[2 * i] = static_cast<float>(src[i]);
        dst[2 * i + 1] = 0.0f;
    }
    // Zero pad
    for (int i = copyLen; i < fftSize; ++i) {
        dst[2 * i] = 0.0f;
        dst[2 * i + 1] = 0.0f;
    }

    if (DftiComputeForward(fftHandle, dst) != DFTI_NO_ERROR) return;

    // Calculate magnitude in-place (compacting to start of buffer)
    const int numBins = fftSize / 2 + 1;
    // MKL vcAbs: complex float -> magnitude float
    // Use the latter part of the buffer as temporary storage to avoid overwriting input before reading.
    // Ensure 64-byte alignment for MKL output (16 floats)
    // dst is 64-byte aligned. fftSize is a multiple of 16.
    // Offset by fftSize + 16 floats ensures alignment and no overlap with input (fftSize + 2 floats).
    float* magBuf = dst + fftSize + 16;
    vcAbs(numBins, reinterpret_cast<const MKL_Complex8*>(dst), magBuf);
    std::memcpy(dst, magBuf, numBins * sizeof(float));
#else
    juce::dsp::FFT fft(static_cast<int>(std::log2(fftSize)));
    // Double -> Float conversion for display FFT
    for (int i = 0; i < copyLen; ++i)
        dst[i] = static_cast<float>(src[i]);

    fft.performFrequencyOnlyForwardTransform(cachedFFTBuffer);
    const int numBins = fftSize / 2 + 1;
#endif

    // スムーシング適用 (Linear Magnitudeに対して行う)
    std::vector<float> linearMags(cachedFFTBuffer.get(), cachedFFTBuffer.get() + numBins);
    applySmoothing(linearMags, fftSize);

    // マグニチュード(dB)に変換して格納
    irMagnitudeSpectrum.resize(numBins);

    for (int i = 0; i < numBins; ++i)
    {
        float mag = linearMags[i];
        irMagnitudeSpectrum[i] = (mag > 1e-9f) ? juce::Decibels::gainToDecibels(mag) : -100.0f;
    }
}

//--------------------------------------------------------------
// State Management
//--------------------------------------------------------------
juce::ValueTree ConvolverProcessor::getState() const
{
    juce::ValueTree v ("Convolver");
    v.setProperty ("mix", mixTarget.load(), nullptr);
    v.setProperty ("bypassed", bypassed.load(), nullptr);
    v.setProperty ("useMinPhase", useMinPhase.load(), nullptr);
    v.setProperty ("smoothingTime", smoothingTimeSec.load(), nullptr);
    v.setProperty ("irLength", targetIRLengthSec.load(), nullptr);
    {
        const juce::ScopedLock sl(irFileLock);
        v.setProperty ("irPath", currentIrFile.getFullPathName(), nullptr);
    }
    return v;
}

void ConvolverProcessor::setState (const juce::ValueTree& v)
{
    if (v.hasProperty ("mix")) setMix (v.getProperty ("mix"));
    if (v.hasProperty ("bypassed")) setBypass (v.getProperty ("bypassed"));
    if (v.hasProperty ("useMinPhase")) setUseMinPhase (v.getProperty ("useMinPhase"));
    if (v.hasProperty ("smoothingTime")) setSmoothingTime (v.getProperty ("smoothingTime"));
    if (v.hasProperty ("irLength")) setTargetIRLength (v.getProperty ("irLength"));

    if (v.hasProperty ("irPath"))
    {
        juce::File fileToLoad; // ロード対象のファイルを保持
        juce::String path = v.getProperty ("irPath").toString();
        if (path.isNotEmpty())
        {
            juce::File f (path);
            if (f.existsAsFile())
            {
                // ロック内では currentIrFile との比較のみを行い、ロック外で loadImpulseResponse を呼ぶ
                const juce::ScopedLock sl(irFileLock);
                if (f != currentIrFile)
                    fileToLoad = f;
            }
            else
            {
                // IRファイルが見つからない場合のエラーハンドリング
                juce::NativeMessageBox::showAsync(
                    juce::MessageBoxOptions()
                        .withIconType(juce::MessageBoxIconType::WarningIcon)
                        .withTitle("IR File Not Found")
                        .withMessage("The Impulse Response file specified in the preset could not be found:\n" + path + "\n\nThe previous IR will be kept.")
                        .withButton("OK"),
                    nullptr);
            }
        }

        // ロックの外でロードを実行
        if (fileToLoad.existsAsFile())
            loadImpulseResponse(fileToLoad);
    }
}

//--------------------------------------------------------------
// syncStateFrom
//--------------------------------------------------------------
void ConvolverProcessor::syncStateFrom(const ConvolverProcessor& other)
{
    jassert (juce::MessageManager::getInstance()->isThisTheMessageThread());

    // パラメータの同期
    mixTarget.store(other.mixTarget.load(), std::memory_order_release);
    bypassed.store(other.bypassed.load(), std::memory_order_release);
    useMinPhase.store(other.useMinPhase.load(), std::memory_order_release);
    smoothingTimeSec.store(other.smoothingTimeSec.load(), std::memory_order_release);
    targetIRLengthSec.store(other.targetIRLengthSec.load(), std::memory_order_release);

    // サンプルレート変更時にリビルドできるよう、元のIR情報をコピーする
    // これにより、新しいDSPコアがIRをリサンプリングするためのソース素材を持つことが保証されます。
    originalIR = other.originalIR;
    originalIRSampleRate = other.originalIRSampleRate;
    {
        const juce::ScopedLock sl(irFileLock);
        currentIrFile = other.currentIrFile;
    }
    irName = other.irName;
    irLength = other.irLength;

    // ▼ クローンを作らない (prepareToPlayが正しいレートでSCを生成するため)
    // 現在の実装では毎回クローンを作成し即廃棄している（LEAK 1）
    // SCはDSPCore::prepare()内のprepareToPlay、またはrebuildAllIRsSynchronousで生成する
    // activeConvolution / convolution はnullptrのままにする
    convolution.store(nullptr, std::memory_order_release);
    activeConvolution.reset();
}

void ConvolverProcessor::syncParametersFrom(const ConvolverProcessor& other)
{
    jassert (juce::MessageManager::getInstance()->isThisTheMessageThread());

    // 軽量なランタイムパラメータのみ同期 (AudioBufferのコピーを避ける)
    // 注意:
    //   useMinPhase / targetIRLengthSec はIR再構築を伴う構造変更パラメータのため、
    //   ここで同期すると requestRebuild() 側のIR再利用判定が誤って成立し、
    //   古い畳み込み実体が再利用される恐れがある。
    //   これらは UIプロセッサのロード完了通知(sendChangeMessage)経由で
    //   requestRebuild() に反映させる。
    mixTarget.store(other.mixTarget.load(), std::memory_order_release);
    bypassed.store(other.bypassed.load(), std::memory_order_release);
    smoothingTimeSec.store(other.smoothingTimeSec.load(), std::memory_order_release);

    // サンプルレートが一致する場合のみ Convolution オブジェクトを同期する。
    // オーバーサンプリング中は DSP側のレート(Nx) != UI側のレート(1x) となるため、
    // UI側のオブジェクトをコピーするとピッチズレやレイテンシー不整合が発生する。
    if (std::abs(currentSampleRate.load() - other.currentSampleRate.load()) < 1e-6)
    {
        auto* otherConv = other.convolution.load(std::memory_order_acquire);
        auto* expectedConv = convolution.load(std::memory_order_acquire);

        if (otherConv != expectedConv && otherConv != nullptr)
        {
            shareConvolutionEngineFrom(other);
        }
    }
}

void ConvolverProcessor::copyConvolutionEngineFrom(const ConvolverProcessor& other)
{
    // Only copy the heavy engine part and related IR metadata
    auto otherConv = other.activeConvolution;
    StereoConvolver::Ptr newConv = (otherConv != nullptr) ? otherConv->clone() : nullptr;
    convolution.store(newConv.get(), std::memory_order_release);

    if (activeConvolution)
    {
        const juce::ScopedLock sl(trashBinLock);
        trashBin.push_back({activeConvolution, juce::Time::getMillisecondCounter()});
    }
    activeConvolution = newConv;

    irLength = other.irLength;
}

void ConvolverProcessor::shareConvolutionEngineFrom(const ConvolverProcessor& other)
{
    // Share the active convolution engine (Shared Pointer copy)
    auto otherConv = other.activeConvolution;
    convolution.store(otherConv.get(), std::memory_order_release);

    if (activeConvolution)
    {
        const juce::ScopedLock sl(trashBinLock);
        trashBin.push_back({activeConvolution, juce::Time::getMillisecondCounter()});
    }
    activeConvolution = otherConv;

    irLength = other.irLength;
}

void ConvolverProcessor::refreshLatency()
{
    auto* conv = convolution.load(std::memory_order_acquire);
    if (conv)
    {
        const int totalLatency = juce::jmin(conv->latency + conv->irLatency, MAX_TOTAL_DELAY);
        latencySmoother.setCurrentAndTargetValue(static_cast<double>(totalLatency));
    }
    else
    {
        latencySmoother.setCurrentAndTargetValue(0.0);
    }
}

//--------------------------------------------------------------
// process (Audio Thread)
// リアルタイム制約 (Real-time Constraints)
//    - メモリ確保なし (No Malloc)
//    - ロックなし (No Lock)
//    - ファイルI/Oなし (No I/O)
//    - 待機なし (No Wait): IR再ロード等はMessage Threadで行う (Audio Threadでの待機は厳禁)
//    - RCU (Read-Copy-Update) パターンにより、ロックフリーで安全にパラメータ/IRを更新
//--------------------------------------------------------------
void ConvolverProcessor::process(juce::dsp::AudioBlock<double>& block)
{
    // (A) Denormal対策 (重要)
    juce::ScopedNoDenormals noDenormals;

    // ── Step 1: RCU State Load (Lock-free / Wait-free) ──
    // Raw pointer load (No ref counting)
    auto* conv = convolution.load(std::memory_order_acquire);

    // ── Step 2: 処理実行可能かチェック ──
    // バイパス、未準備、IR未ロードの場合はスルー
    if (!isPrepared.load(std::memory_order_acquire) || bypassed.load(std::memory_order_relaxed) || !conv)
    {
        return;
    }

    // レイテンシー補正の更新 (必要な場合のみ)
    {
        // 処理遅延(ブロックサイズ) + IR遅延(ピーク位置)
        const int calculatedLatency = conv->latency + conv->irLatency;

        // 安全対策: 要求される遅延が最大許容値を超えていないかデバッグ時にチェック
        jassert(calculatedLatency <= MAX_TOTAL_DELAY);

        const int totalLatency = juce::jmin(calculatedLatency, MAX_TOTAL_DELAY);

        // ターゲット値が変更された場合のみ更新
        if (std::abs(latencySmoother.getTargetValue() - static_cast<double>(totalLatency)) > 0.5) // 0.5サンプル以上の変化でトリガー
        {
            // ドップラー効果対策: クロスフェードを開始
            // クロスフェード中はターゲット更新を保留し、不連続なジャンプ（クリック）を防ぐ
            if (!crossfadeGain.isSmoothing())
            {
                oldDelay = latencySmoother.getCurrentValue();
                crossfadeGain.setCurrentAndTargetValue(0.0); // 古いディレイパスのゲインを0に設定
                crossfadeGain.setTargetValue(1.0);           // 新しいディレイパスのゲインを1に設定
                latencySmoother.setTargetValue(static_cast<double>(totalLatency));
            }
        }
    }

    // processBufferのチャンネル数を使用 (最大2ch)
    const int procChannels = (std::min)((int)block.getNumChannels(), 2);
    const int numSamples = (int)block.getNumSamples();

    // ── Step 3: バッファサイズ安全対策 (Bounds Check) ──
    if (numSamples <= 0 || procChannels == 0 || numSamples > dryBuffer.getNumSamples())
        return;

    // ── Step 4: パラメータ更新と最適化 ──
    // Audio Threadでのみ setTargetValue() を呼ぶことでスレッドセーフティを確保
    const double targetMixValue = static_cast<double>(mixTarget.load(std::memory_order_relaxed));
    if (std::abs(mixSmoother.getTargetValue() - targetMixValue) > 1.0e-5)
    {
        mixSmoother.setTargetValue(targetMixValue);
    }

    // Smoothing Timeの更新 (Audio Thread-safe)
    // UIスレッドで変更された値を検出し、SmoothedValueのランプタイムを再設定する。
    // reset()は内部係数を再計算するだけで、メモリ確保やロックは行わないため安全。
    // Smoothing Timeの更新
    const double newSmoothingTime = smoothingTimeSec.load(std::memory_order_relaxed);
    if (std::abs(currentSmoothingTimeSec - newSmoothingTime) > 0.0001)
    {
        // reset()を呼ぶと現在値がリセットされる可能性があるため、
        // 現在値とターゲット値を保持したままランプ時間のみ更新する手順を踏む
        // これにより、スムージング時間の変更時に音量が飛ぶのを防ぐ
        double currentVal = mixSmoother.getCurrentValue();
        double targetVal = mixSmoother.getTargetValue();
        mixSmoother.reset(currentSpec.sampleRate, newSmoothingTime);
        mixSmoother.setCurrentAndTargetValue(currentVal); // Restore current value
        mixSmoother.setTargetValue(targetVal);
        currentSmoothingTimeSec = newSmoothingTime;
    }

    const bool isSmoothing = mixSmoother.isSmoothing();

    // ── 最適化: 処理内容をミックス比率に応じて決定 ──
    const bool needsConvolution = isSmoothing || targetMixValue > 0.001;
    const bool needsDrySignal   = isSmoothing || targetMixValue < 0.999;

    // ── Step 5: Dry信号生成 ──
    // DelayLineの内部状態（履歴）を維持するため、Dry信号が不要な場合(100% Wet)でも常に処理を実行する。
    // これにより、Mixパラメータ変更時に過去のDry信号が正しく再生されるようにする。
    {
        // 1. 入力をリングバッファに書き込む (Push)
        // 常にブロック単位で書き込むため、AVX2で最適化可能
        int wPos = delayWritePos;
        for (int ch = 0; ch < procChannels; ++ch)
        {
            const double* src = block.getChannelPointer(ch);
            double* buf = delayBuffer[ch].get();

            // リングバッファの境界処理 (2分割コピー)
            int samplesFirst = std::min(numSamples, DELAY_BUFFER_SIZE - wPos);
            int samplesSecond = numSamples - samplesFirst;

            std::memcpy(buf + wPos, src, samplesFirst * sizeof(double));
            if (samplesSecond > 0)
                std::memcpy(buf, src + samplesFirst, samplesSecond * sizeof(double));
        }

        // 書き込み位置の更新は後で行う（読み出しで現在の位置を使うため）

        if (crossfadeGain.isSmoothing())
        {
            // --- クロスフェード処理 ---
            const double newDelay = latencySmoother.getTargetValue();

            // サブサンプル精度読み出し用ヘルパー (Catmull-Rom Interpolation)
            auto readInterpolated = [&](double delay, double* dst, int ch)
            {
                const double* srcBuf = delayBuffer[ch].get();
                double rPos = static_cast<double>(delayWritePos) - delay;

                // rPos を [0, DELAY_BUFFER_SIZE) に正規化
                // floor(rPos) が p1 (t=0) のインデックスとなる
                rPos -= std::floor(rPos / DELAY_BUFFER_SIZE) * DELAY_BUFFER_SIZE;

                const int iRead = static_cast<int>(rPos);
                const double frac = rPos - iRead;

                // 最適化: ほぼ整数の場合は高速パス (memcpy)
                if (std::abs(frac) < 1.0e-6)
                {
                    int rPosInt = iRead; // frac ~ 0.0
                    int samplesFirst = std::min(numSamples, DELAY_BUFFER_SIZE - rPosInt);
                    std::memcpy(dst, srcBuf + rPosInt, samplesFirst * sizeof(double));
                    if (numSamples > samplesFirst)
                        std::memcpy(dst + samplesFirst, srcBuf, (numSamples - samplesFirst) * sizeof(double));
                    return;
                }
                else if (std::abs(frac - 1.0) < 1.0e-6)
                {
                    int rPosInt = (iRead + 1) & DELAY_BUFFER_MASK; // frac ~ 1.0
                    int samplesFirst = std::min(numSamples, DELAY_BUFFER_SIZE - rPosInt);
                    std::memcpy(dst, srcBuf + rPosInt, samplesFirst * sizeof(double));
                    if (numSamples > samplesFirst)
                        std::memcpy(dst + samplesFirst, srcBuf, (numSamples - samplesFirst) * sizeof(double));
                    return;
                }

                // Catmull-Rom 係数 (ブロック内で一定)
                const double t = frac;
                const double t2 = t * t;
                const double t3 = t2 * t;
                const double w0 = -0.5 * t3 + t2 - 0.5 * t;
                const double w1 =  1.5 * t3 - 2.5 * t2 + 1.0;
                const double w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
                const double w3 =  0.5 * t3 - 0.5 * t2;

                int i = 0;
                // 境界チェック: 読み出し範囲がバッファ境界を跨がない場合のみ高速化
                if (iRead >= 1 && iRead + numSamples + 2 < DELAY_BUFFER_SIZE)
                {
                    const double* s = srcBuf + iRead;
#if defined(__AVX2__)
                    const __m256d vw0 = _mm256_set1_pd(w0);
                    const __m256d vw1 = _mm256_set1_pd(w1);
                    const __m256d vw2 = _mm256_set1_pd(w2);
                    const __m256d vw3 = _mm256_set1_pd(w3);

                    // AVX2 最適化ループ
                    for (; i <= numSamples - 4; i += 4)
                    {
                        __m256d p0 = _mm256_loadu_pd(s + i - 1);
                        __m256d p1 = _mm256_loadu_pd(s + i);
                        __m256d p2 = _mm256_loadu_pd(s + i + 1);
                        __m256d p3 = _mm256_loadu_pd(s + i + 2);
                        __m256d sum = _mm256_mul_pd(p0, vw0);
                        sum = _mm256_fmadd_pd(p1, vw1, sum);
                        sum = _mm256_fmadd_pd(p2, vw2, sum);
                        sum = _mm256_fmadd_pd(p3, vw3, sum);
                        _mm256_storeu_pd(dst + i, sum);
                    }
#endif
                    // スカラー残余処理 (AVX2ループ後、または非AVX2ビルド時)
                    for (; i < numSamples; ++i)
                        dst[i] = w0 * s[i - 1] + w1 * s[i] + w2 * s[i + 1] + w3 * s[i + 2];
                }
                else
                {
                    // バッファラップアラウンド対応 (低速パス)
                    for (; i < numSamples; ++i)
                    {
                        int idx = iRead + i;
                        double p0 = srcBuf[(idx - 1) & DELAY_BUFFER_MASK];
                        double p1 = srcBuf[(idx    ) & DELAY_BUFFER_MASK];
                        double p2 = srcBuf[(idx + 1) & DELAY_BUFFER_MASK];
                        double p3 = srcBuf[(idx + 2) & DELAY_BUFFER_MASK];
                        dst[i] = w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3;
                    }
                }
            };

            // 1. 古いディレイからの信号を oldDryBuffer に読み出す
            for (int ch = 0; ch < procChannels; ++ch)
                readInterpolated(oldDelay, oldDryBuffer.getWritePointer(ch), ch);

            // 2. 新しいディレイからの信号を dryBuffer に読み出す
            for (int ch = 0; ch < procChannels; ++ch)
                readInterpolated(newDelay, dryBuffer.getWritePointer(ch), ch);

            // 3. 2つの信号をクロスフェードして dryBuffer に書き込む
#if defined(__AVX2__)
            const double startFadeInGain = crossfadeGain.getCurrentValue();
            crossfadeGain.skip(numSamples);
            const double endFadeInGain = crossfadeGain.getCurrentValue();
            const double fadeInInc = (endFadeInGain - startFadeInGain) / static_cast<double>(numSamples);

            for (int ch = 0; ch < procChannels; ++ch)
            {
                double* newSamples = dryBuffer.getWritePointer(ch);
                const double* oldSamples = oldDryBuffer.getReadPointer(ch);

                __m256d vGain = _mm256_set_pd(startFadeInGain + 3.0 * fadeInInc,
                                              startFadeInGain + 2.0 * fadeInInc,
                                              startFadeInGain + fadeInInc,
                                              startFadeInGain);
                const __m256d vInc = _mm256_set1_pd(4.0 * fadeInInc);
                const __m256d vOne = _mm256_set1_pd(1.0);

                int i = 0;
                for (; i <= numSamples - 4; i += 4)
                {
                    const __m256d vOld = _mm256_loadu_pd(oldSamples + i);
                    const __m256d vNew = _mm256_loadu_pd(newSamples + i);
                    const __m256d vFadeOutGain = _mm256_sub_pd(vOne, vGain);

                    // out = new * fadeIn + old * fadeOut
                    const __m256d vOut = _mm256_fmadd_pd(vNew, vGain, _mm256_mul_pd(vOld, vFadeOutGain));
                    _mm256_storeu_pd(newSamples + i, vOut);

                    vGain = _mm256_add_pd(vGain, vInc);
                }

                double currentGain = startFadeInGain + static_cast<double>(i) * fadeInInc;
                for (; i < numSamples; ++i)
                {
                    newSamples[i] = newSamples[i] * currentGain + oldSamples[i] * (1.0 - currentGain);
                    currentGain += fadeInInc;
                }
            }
#else
            // Fallback for non-AVX2
            for (int i = 0; i < numSamples; ++i)
            {
                const double fadeInGain = crossfadeGain.getNextValue();
                const double fadeOutGain = 1.0 - fadeInGain;
                for (int ch = 0; ch < procChannels; ++ch)
                {
                    dryBuffer.setSample(ch, i, dryBuffer.getSample(ch, i) * fadeInGain + oldDryBuffer.getSample(ch, i) * fadeOutGain);
                }
            }
#endif

            if (!crossfadeGain.isSmoothing())
            {
                latencySmoother.setCurrentAndTargetValue(latencySmoother.getTargetValue());
                oldDelay = latencySmoother.getCurrentValue();
            }
        }
        else
        {
            // 安定時はブロック処理で最適化
            // 遅延量は整数とみなす (補間なしの高速コピー)
            int delayInt = static_cast<int>(latencySmoother.getCurrentValue() + 0.5);

            // 読み出し開始位置
            int rPos = (delayWritePos - delayInt) & DELAY_BUFFER_MASK;
            // 負の補正 (念のため)
            if (rPos < 0) rPos += DELAY_BUFFER_SIZE;

            for (int ch = 0; ch < procChannels; ++ch)
            {
                double* srcBuf = delayBuffer[ch].get();
                double* dstBuf = dryBuffer.getWritePointer(ch);

                // リングバッファからの読み出し (2分割コピー)
                int samplesFirst = std::min(numSamples, DELAY_BUFFER_SIZE - rPos);
                int samplesSecond = numSamples - samplesFirst;

                // AVX2最適化コピー (memcpyは通常最適化されているが、明示的なループ展開も可)
                // ここではmemcpyを使用 (コンパイラがAVX命令を使用する)
                std::memcpy(dstBuf, srcBuf + rPos, samplesFirst * sizeof(double));
                if (samplesSecond > 0)
                    std::memcpy(dstBuf + samplesFirst, srcBuf, samplesSecond * sizeof(double));
            }
        }

        // 書き込み位置を更新
        delayWritePos = (delayWritePos + numSamples) & DELAY_BUFFER_MASK;
    }

    // ── Step 6 & 7: Wet信号生成 & Mix (Fused & Optimized) ──
    // 常にコンボリューションを実行し、エンジンの内部状態(オーバーラップバッファ)を維持する。
    // これにより、Mixを0%から上げた際のグリッチを防ぐ。
    // WDL_ConvolutionEngineを使用

    const double headroom = CONVOLUTION_HEADROOM_GAIN;

    const double* wetGains = nullptr;
    const double* dryGains = nullptr;

    // スムーシングゲインの計算
    if (isSmoothing)
    {
        // Audio Threadでのメモリ確保を避けるため、事前に確保したメンバ変数のバッファを使用
        double* wg = smoothingBuffer.getWritePointer(0);
        double* dg = smoothingBuffer.getWritePointer(1);

        for (int i = 0; i < numSamples; ++i)
        {
            const double mix = mixSmoother.getNextValue();
            wg[i] = mix * headroom;
            dg[i] = 1.0 - mix;
        }
        wetGains = wg;
        dryGains = dg;
    }

    // 追加防御:
    // WDL呼び出しサイズを量子化し、プリウォーム済みサイズを超える長さは必ず分割する。
    const int quantizedCallSamples = juce::jmax(1, conv->callQuantumSamples);
    const int prewarmedMaxSamples = juce::jmax(1, conv->prewarmedMaxSamples);
    const int guardedCallSamples = juce::jmin(quantizedCallSamples, prewarmedMaxSamples);

    // WDLへの呼び出しサイズを固定化 (内部再確保防止)
    // AudioEngineは、processに渡すnumSamplesがguardedCallSamplesの倍数であることを保証する。
    // この前提が崩れると、最後のチャンクが小さくなり、WDL内部で再確保が発生する可能性がある。
    const int callLen = guardedCallSamples;
    jassert(numSamples % callLen == 0 && "ConvolverProcessor::process: numSamples must be a multiple of the guarded call size.");

    for (int ch = 0; ch < procChannels; ++ch)
    {
        const double wetG = needsConvolution ? (targetMixValue * headroom) : 0.0;
        const double dryG = needsDrySignal ? (1.0 - targetMixValue) : 0.0;
        const double* inputBase = block.getChannelPointer(ch);
        double* wetBase = wetBufferStorage[ch].get(); // Use temp buffer for wet signal
        const double* dryBase = dryBuffer.getReadPointer(ch);
        double* dstBase = block.getChannelPointer(ch);

        int processed = 0;
        while (processed < numSamples)
        {
            // 1. Process Convolution (Unified Interface)
            const double* input = inputBase + processed;
            double* wetOut = wetBase + processed;

            conv->process(ch, input, wetOut, callLen);

            // Note: StereoConvolver::process guarantees output is written to wetOut
            const double* wdlOut = wetOut;
            int validWetSamples = callLen; // Assumed valid after process

            // 3. Mix (Fused Loop: Copy + Gain + Mix)
            double* dst = dstBase + processed;
            const double* dry = dryBase + processed;

            if (isSmoothing)
            {
                const double* wetChunkGains = wetGains + processed;
                const double* dryChunkGains = dryGains + processed;

                int i = 0;
#if defined(__AVX2__)
                const int vLoop = validWetSamples / 16 * 16; // Unroll 4x (16 doubles)
                if (vLoop > 0)
                {
                    for (; i < vLoop; i += 16)
                    {
                        _mm_prefetch(reinterpret_cast<const char*>(wdlOut + i + 64), _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char*>(dry + i + 64), _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char*>(wetChunkGains + i + 64), _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char*>(dryChunkGains + i + 64), _MM_HINT_T0);

                        // 1
                        __m256d vWet0 = _mm256_loadu_pd(wdlOut + i);
                        __m256d vDry0 = _mm256_loadu_pd(dry + i);
                        __m256d vWetG0 = _mm256_loadu_pd(wetChunkGains + i);
                        __m256d vDryG0 = _mm256_loadu_pd(dryChunkGains + i);
                        __m256d vOut0 = _mm256_fmadd_pd(vWet0, vWetG0, _mm256_mul_pd(vDry0, vDryG0));
                        _mm256_storeu_pd(dst + i, vOut0);

                        // 2
                        __m256d vWet1 = _mm256_loadu_pd(wdlOut + i + 4);
                        __m256d vDry1 = _mm256_loadu_pd(dry + i + 4);
                        __m256d vWetG1 = _mm256_loadu_pd(wetChunkGains + i + 4);
                        __m256d vDryG1 = _mm256_loadu_pd(dryChunkGains + i + 4);
                        __m256d vOut1 = _mm256_fmadd_pd(vWet1, vWetG1, _mm256_mul_pd(vDry1, vDryG1));
                        _mm256_storeu_pd(dst + i + 4, vOut1);

                        // 3
                        __m256d vWet2 = _mm256_loadu_pd(wdlOut + i + 8);
                        __m256d vDry2 = _mm256_loadu_pd(dry + i + 8);
                        __m256d vWetG2 = _mm256_loadu_pd(wetChunkGains + i + 8);
                        __m256d vDryG2 = _mm256_loadu_pd(dryChunkGains + i + 8);
                        __m256d vOut2 = _mm256_fmadd_pd(vWet2, vWetG2, _mm256_mul_pd(vDry2, vDryG2));
                        _mm256_storeu_pd(dst + i + 8, vOut2);

                        // 4
                        __m256d vWet3 = _mm256_loadu_pd(wdlOut + i + 12);
                        __m256d vDry3 = _mm256_loadu_pd(dry + i + 12);
                        __m256d vWetG3 = _mm256_loadu_pd(wetChunkGains + i + 12);
                        __m256d vDryG3 = _mm256_loadu_pd(dryChunkGains + i + 12);
                        __m256d vOut3 = _mm256_fmadd_pd(vWet3, vWetG3, _mm256_mul_pd(vDry3, vDryG3));
                        _mm256_storeu_pd(dst + i + 12, vOut3);
                    }
                }
                // Handle remaining multiples of 4
                for (; i < (validWetSamples / 4 * 4); i += 4)
                {
                    __m256d vWet = _mm256_loadu_pd(wdlOut + i);
                    __m256d vDry = _mm256_loadu_pd(dry + i);
                    __m256d vWetG = _mm256_loadu_pd(wetChunkGains + i);
                    __m256d vDryG = _mm256_loadu_pd(dryChunkGains + i);
                    __m256d vOut = _mm256_fmadd_pd(vWet, vWetG, _mm256_mul_pd(vDry, vDryG));
                    _mm256_storeu_pd(dst + i, vOut);
                }
#endif
                for (; i < validWetSamples; ++i)
                {
                    dst[i] = wdlOut[i] * wetChunkGains[i] + dry[i] * dryChunkGains[i];
                }

                // Wetが無効な区間
                for (; i < callLen; ++i)
                {
                    dst[i] = dry[i] * dryChunkGains[i];
                }
            }
            else
            {
                // 定常状態 (99%のケース) -> AVX2最適化
                int i = 0;

#if defined(__AVX2__)
                const __m256d vWetG = _mm256_set1_pd(wetG);
                const __m256d vDryG = _mm256_set1_pd(dryG);

                const int vLoop = validWetSamples / 16 * 16; // Unroll 4x
                if (vLoop > 0)
                {
                    for (; i < vLoop; i += 16)
                    {
                        _mm_prefetch(reinterpret_cast<const char*>(wdlOut + i + 64), _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char*>(dry + i + 64), _MM_HINT_T0);

                        // 1
                        __m256d vWet0 = _mm256_loadu_pd(wdlOut + i);
                        __m256d vDry0 = _mm256_loadu_pd(dry + i);
                        __m256d vOut0 = _mm256_fmadd_pd(vWet0, vWetG, _mm256_mul_pd(vDry0, vDryG));
                        _mm256_storeu_pd(dst + i, vOut0);

                        // 2
                        __m256d vWet1 = _mm256_loadu_pd(wdlOut + i + 4);
                        __m256d vDry1 = _mm256_loadu_pd(dry + i + 4);
                        __m256d vOut1 = _mm256_fmadd_pd(vWet1, vWetG, _mm256_mul_pd(vDry1, vDryG));
                        _mm256_storeu_pd(dst + i + 4, vOut1);

                        // 3
                        __m256d vWet2 = _mm256_loadu_pd(wdlOut + i + 8);
                        __m256d vDry2 = _mm256_loadu_pd(dry + i + 8);
                        __m256d vOut2 = _mm256_fmadd_pd(vWet2, vWetG, _mm256_mul_pd(vDry2, vDryG));
                        _mm256_storeu_pd(dst + i + 8, vOut2);

                        // 4
                        __m256d vWet3 = _mm256_loadu_pd(wdlOut + i + 12);
                        __m256d vDry3 = _mm256_loadu_pd(dry + i + 12);
                        __m256d vOut3 = _mm256_fmadd_pd(vWet3, vWetG, _mm256_mul_pd(vDry3, vDryG));
                        _mm256_storeu_pd(dst + i + 12, vOut3);
                    }
                }
                for (; i < (validWetSamples / 4 * 4); i += 4)
                {
                    __m256d vWet = _mm256_loadu_pd(wdlOut + i);
                    __m256d vDry = _mm256_loadu_pd(dry + i);
                    __m256d vOut = _mm256_fmadd_pd(vWet, vWetG, _mm256_mul_pd(vDry, vDryG));
                    _mm256_storeu_pd(dst + i, vOut);
                }
#endif
                // 残りの有効なWetサンプル (Scalar)
                for (; i < validWetSamples; ++i)
                {
                    dst[i] = wdlOut[i] * wetG + dry[i] * dryG;
                }

                // Wetが無効な区間 (初期レイテンシー等) -> Dryのみ出力
                for (; i < callLen; ++i)
                {
                    dst[i] = dry[i] * dryG;
                }
            }

            processed += callLen;
        }
    }
}
/*
    // 補足: 元のコードにあった以下のブロックは削除・統合されました。
    // - convolutionBuffer への memcpy
    // - convolutionBuffer.applyGain
    // - 3パターンの分岐 (DryOnly, WetOnly, Mix)

    // 新しいコードはこれらを1つのループで行い、AVX2で高速化しています。
*/

//--------------------------------------------------------------
// setMix
//--------------------------------------------------------------
void ConvolverProcessor::setMix(float mixAmount)
{
    // 0.0 ~ 1.0 にクランプ
    float newVal = juce::jlimit(0.0f, 1.0f, mixAmount);
    if (std::abs(mixTarget.load() - newVal) > 1.0e-5f)
    {
        mixTarget.store(newVal);
        listeners.call(&Listener::convolverParamsChanged, this);
    }
}

float ConvolverProcessor::getMix() const
{
    return mixTarget.load();
}

void ConvolverProcessor::setBypass(bool shouldBypass)
{
    if (bypassed.load() != shouldBypass)
    {
        bypassed.store(shouldBypass);
        listeners.call(&Listener::convolverParamsChanged, this);
    }
}

void ConvolverProcessor::setTargetIRLength(float timeSec)
{
    float clampedTime = juce::jlimit(IR_LENGTH_MIN_SEC, IR_LENGTH_MAX_SEC, timeSec);
    if (std::abs(targetIRLengthSec.load() - clampedTime) > 1e-5f)
    {
        targetIRLengthSec.store(clampedTime);
        listeners.call(&Listener::convolverParamsChanged, this);

        // IRがロードされている場合、メモリ上のデータを使ってリビルドする (Disk I/O回避)
        if (isIRLoaded())
        {
            loadImpulseResponse(juce::File()); // 空のファイルを渡すとリビルドモードになる
        }
    }
}

void ConvolverProcessor::setSmoothingTime(float timeSec)
{
    float clampedTime = juce::jlimit(SMOOTHING_TIME_MIN_SEC, SMOOTHING_TIME_MAX_SEC, timeSec);
    if (std::abs(smoothingTimeSec.load() - clampedTime) > 1e-5f)
    {
        smoothingTimeSec.store(clampedTime);
        listeners.call(&Listener::convolverParamsChanged, this);
    }
}

float ConvolverProcessor::getTargetIRLength() const
{
    return targetIRLengthSec.load();
}

float ConvolverProcessor::getSmoothingTime() const
{
    return smoothingTimeSec.load();
}

void ConvolverProcessor::setUseMinPhase(bool shouldUseMinPhase)
{
    if (useMinPhase.load() != shouldUseMinPhase)
    {
        useMinPhase.store(shouldUseMinPhase);
        listeners.call(&Listener::convolverParamsChanged, this);

        // 設定変更時にIRがロード済みなら再ロードして変換を適用
        if (isIRLoaded())
        {
            loadImpulseResponse(juce::File()); // リビルドモード
        }
    }
}
