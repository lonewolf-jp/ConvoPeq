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
#include <deque>

#include "CDSPResampler.h"
#include "AlignedAllocation.h" // For convo::MKLAllocator

#include <mkl.h>
#include <mkl_vml.h>

 #include <xmmintrin.h>
 #include <pmmintrin.h>
 #include <immintrin.h> // For AVX2

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

static bool applyAsymmetricTukey(double* data, int numSamples)
{
    if (numSamples <= 0) return true; // no-op: エラーではない

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

    convo::ScopedAlignedPtr<double> window_vals(static_cast<double*>(convo::aligned_malloc(static_cast<size_t>(numSamples) * sizeof(double), 64)));
    if (!window_vals) return false; // メモリ確保失敗: 呼び出し元に伝播

    // ── 3. 窓関数バッファを 1.0 で初期化 ──
    std::fill_n(window_vals.get(), numSamples, 1.0);

    // Pre-peak part
    if (peakIndex > 0)
    {
        const int pre_taper_len = static_cast<int>(std::floor(peakIndex * alpha_pre));
        if (pre_taper_len > 0)
        {
            convo::ScopedAlignedPtr<double> cos_args(static_cast<double*>(convo::aligned_malloc(static_cast<size_t>(pre_taper_len) * sizeof(double), 64)));
            if (!cos_args) return false; // メモリ確保失敗: 呼び出し元に伝播

            // ── cos引数計算: cos_args[i] = scale * i + offset ──
            // [Bug A fix] vdLinearFrac(n, nullptr, nullptr, scale≠0, ...) は
            // scalea≠0 の場合 a[i] を読みに行き nullptr デリファレンス → Access Violation。
            // SEH は /EHsc では捕捉されないため LoaderThread が強制終了する。
            // → スカラーループで等差数列を生成する。
            const double scale = pi / (peakIndex * alpha_pre);
            const double offset = -pi;
            for (int i = 0; i < pre_taper_len; ++i)
                cos_args.get()[i] = scale * static_cast<double>(i) + offset;

            vdCos(pre_taper_len, cos_args.get(), window_vals.get());

            // ── 窓関数値計算 (MKL VML) ──
            // window_vals[i] = 0.5 * (1.0 + window_vals[i]) = 0.5 * window_vals[i] + 0.5
            vdLinearFrac(pre_taper_len, window_vals.get(), nullptr, 0.5, 0.5, 0.0, 1.0, window_vals.get());
        }
    }

    // Post-peak part
    const double dist_to_end = static_cast<double>(numSamples - 1 - peakIndex);
    if (dist_to_end > 1.0e-9) // ゼロ除算防止
    {
        const int post_taper_start_idx = peakIndex + static_cast<int>(std::ceil(dist_to_end * (1.0 - alpha_post)));
        const int post_taper_len = numSamples - post_taper_start_idx;
        if (post_taper_len > 0)
        {
            convo::ScopedAlignedPtr<double> cos_args(static_cast<double*>(convo::aligned_malloc(static_cast<size_t>(post_taper_len) * sizeof(double), 64)));
            if (!cos_args) return false; // メモリ確保失敗: 呼び出し元に伝播
            double* post_window_vals = window_vals.get() + post_taper_start_idx;

            const double scale = (pi / alpha_post) / dist_to_end;
            const double offset = (pi / alpha_post) * (((double)post_taper_start_idx - (double)peakIndex) / dist_to_end - (1.0 - alpha_post));
            // [Bug A fix] vdLinearFrac(n, nullptr, nullptr, scale≠0, ...) → スカラーループ
            for (int i = 0; i < post_taper_len; ++i)
                cos_args.get()[i] = scale * static_cast<double>(i) + offset;

            vdCos(post_taper_len, cos_args.get(), post_window_vals);

            vdLinearFrac(post_taper_len, post_window_vals, nullptr, 0.5, 0.5, 0.0, 1.0, post_window_vals);
        }
    }

    // Apply window
    vdMul(numSamples, data, window_vals.get(), data);
    return true;
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

    // 畳み込みエンジン安全制約
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
    LoaderThread(ConvolverProcessor& p, const juce::AudioBuffer<double>& src, double srcSR, double sr, int bs, bool minPhase, double scale)
        : Thread("IRRebuilder"), owner(p), weakOwner(&p), sourceIR(src), sourceSampleRate(srcSR), sampleRate(sr), blockSize(bs), useMinPhase(minPhase), isRebuild(true), scaleFactor(scale)
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
        StereoConvolver* newConv = nullptr;
        bool success = false;
        double scaleFactor = 1.0;
        juce::String errorMessage;
    };

    void run() override
    {
        juce::ScopedNoDenormals noDenormals; // バックグラウンド処理でのDenormal対策

        // MKL/AVX最適化のためにFTZ/DAZフラグを明示的に設定
        // ScopedNoDenormalsでも設定されるが、MKLの要件として明示しておく
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

        // VML (Vector Math Library) のDenormal扱いをゼロに設定
        // vdHypot, vdLn 等のパフォーマンス低下を防ぐ
        vmlSetMode(VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);

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

        // [指摘3 fix] デッドコード削除。
        // performLoad() は非同期パス (callAsync → finalizeNUCEngineOnMessageThread) で
        // 常に result.success = false を返すため、以前の if (result.success ...) ブロックは
        // 永遠に実行されなかった。実際の後処理はメッセージスレッド側の callAsync で行われる。
        if (result.newConv)
        {
            result.newConv->release();
            result.newConv = nullptr;
        }

        if (!result.success && result.errorMessage.isNotEmpty() && !threadShouldExit())
        {
            // エラー発生時: メインスレッドでエラー処理を行う
            auto wp = weakOwner;
            const juce::String error = result.errorMessage;
            juce::MessageManager::callAsync([wp, error]()
            {
                if (auto* o = wp.get())
                    o->handleLoadError(error);
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
                result.scaleFactor = this->scaleFactor;
            }
            else
            {
                if (!file.existsAsFile())
                {
                    result.errorMessage = "IR file not found: " + file.getFullPathName();
                    return result;
                }

                juce::AudioFormatManager formatManager;
                formatManager.registerBasicFormats();
                std::unique_ptr<juce::AudioFormatReader> reader(formatManager.createReaderFor(file));

                if (!reader)
                {
                    result.errorMessage = "Unsupported audio format or corrupted file: " + file.getFileName();
                    return result;
                }

                // サイズの妥当性チェック (lengthInSamples が int の範囲を超える場合への対策)
                const int64 fileLength = reader->lengthInSamples;
                const int numChannels = static_cast<int>(reader->numChannels);
                static constexpr int64 MAX_FILE_LENGTH = 2147483647;  // int の最大値

                if (fileLength > MAX_FILE_LENGTH) {
                    result.errorMessage = "IR file is too large (exceeds 2GB samples limit).";
                    DBG("LoaderThread: " << result.errorMessage);
                    return result;
                }
                if (numChannels <= 0) {
                    result.errorMessage = "Invalid channel count in IR file.";
                    DBG("LoaderThread: " << result.errorMessage);
                    return result;
                }

                // AudioFormatReader::read は float のみ対応のため、一時バッファを使用
                juce::AudioBuffer<float> tempFloatBuffer(numChannels, static_cast<int>(fileLength));
                if (!reader->read(&tempFloatBuffer, 0, static_cast<int>(fileLength), 0, true, true))
                {
                    result.errorMessage = "Failed to read audio data from file.";
                    DBG("LoaderThread: " << result.errorMessage);
                    return result;
                }

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

            // 2. [Bug D fix] scaleFactor の計算をここから Step 2' (trimmed バッファ確定後) に移動。
            // 旧実装は生 IR (ウィンドウ・トリム・MinPhase 変換前) のエネルギーで計算していたため、
            // 実際に SetImpulse() に渡すデータのエネルギーと乖離し、ヘッドルームが不正確だった。

            // 3. 末尾の無音カット (Denormal対策 & 効率化)
            // IR末尾の極小値(Denormal領域)をカットすることで、畳み込み負荷とDenormal発生リスクを低減
            if (result.loadedIR.getNumSamples() > 0)
            {
                const int numSamples = result.loadedIR.getNumSamples();
                const int numChannels = result.loadedIR.getNumChannels();
                const double threshold = 1.0e-15; // -300dB (double精度における実質的な無音)

                int newLength = 0; // Assume all silent if nothing found

                if (numChannels > 0)
                {
                    const double* ch0_ptr = result.loadedIR.getReadPointer(0);
                    const double* ch1_ptr = (numChannels > 1) ? result.loadedIR.getReadPointer(1) : nullptr;

                    const __m256d vThreshold = _mm256_set1_pd(threshold);
                    const __m256d vSignMask = _mm256_set1_pd(-0.0);

                    int i = numSamples;
                    bool found = false;

                    // Process in chunks of 4 from the end using AVX2
                    for (; i >= 4; i -= 4)
                    {
                        __m256d v0 = _mm256_loadu_pd(ch0_ptr + i - 4);
                        __m256d abs_v0 = _mm256_andnot_pd(vSignMask, v0);
                        __m256d mask = _mm256_cmp_pd(abs_v0, vThreshold, _CMP_GT_OQ);

                        if (ch1_ptr)
                        {
                            __m256d v1 = _mm256_loadu_pd(ch1_ptr + i - 4);
                            __m256d abs_v1 = _mm256_andnot_pd(vSignMask, v1);
                            __m256d mask1 = _mm256_cmp_pd(abs_v1, vThreshold, _CMP_GT_OQ);
                            mask = _mm256_or_pd(mask, mask1);
                        }

                        if (_mm256_testz_pd(mask, mask) == 0) // if not all zero
                        {
                            // Non-silent sample found in this chunk. Find the exact one.
                            for (int j = i - 1; j >= i - 4; --j)
                            {
                                if (std::abs(ch0_ptr[j]) > threshold || (ch1_ptr && std::abs(ch1_ptr[j]) > threshold))
                                {
                                    newLength = j + 1;
                                    found = true;
                                    break;
                                }
                            }
                            if (found) break;
                        }
                    }

                    if (!found)
                    {
                        // Check remaining samples (scalar)
                        for (int j = i - 1; j >= 0; --j)
                        {
                            if (std::abs(ch0_ptr[j]) > threshold || (ch1_ptr && std::abs(ch1_ptr[j]) > threshold))
                            {
                                newLength = j + 1;
                                break;
                            }
                        }
                    }
                }

                if (newLength < numSamples)
                {
                    result.loadedIR.setSize(numChannels, std::max(1, newLength), true);
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
                        result.errorMessage = "Resampling failed (unknown error).";
                    }
                    return result;
                }

                result.loadedIR = std::move(resampled);
                result.loadedSR = sampleRate;
            }

            // 5. 高精度型 DC Blocker (1次IIR)
            // NUCコンボルバー直前に置くため、位相回転を最小限に抑えつつDCを除去する
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
                    // [Bug D fix] メモリ確保失敗時はエラーを伝播してロードを中断する。
                    // 旧実装はエラーを無視してウィンドウ未適用のIRのままリターンしていたため、
                    // IR末端の不連続点がクリックノイズを引き起こす可能性があった。
                    if (!applyAsymmetricTukey(result.loadedIR.getWritePointer(ch), numSamples))
                    {
                        result.errorMessage = "Failed to allocate Tukey window buffer (Out of Memory).";
                        DBG("LoaderThread: " << result.errorMessage);
                        return result;
                    }
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

            // 2'. Auto Makeup (Energy Normalization) — trimmed バッファ確定後に計算
            // [Bug D fix] ウィンドウ・トリム・MinPhase 変換がすべて完了した trimmed バッファで
            // エネルギーを計測することで、SetImpulse() に渡す実データと一致した scaleFactor を得る。
            // リビルド時は既に正規化済みのためスキップ。
            if (!isRebuild)
            {
                double maxChannelEnergy = 0.0;
                const int nSamp = trimmed.getNumSamples();
                for (int ch = 0; ch < trimmed.getNumChannels(); ++ch)
                {
                    const double* data = trimmed.getReadPointer(ch);
                    const double energy = cblas_ddot(nSamp, data, 1, data, 1);
                    if (energy > maxChannelEnergy)
                        maxChannelEnergy = energy;
                }
                if (maxChannelEnergy > 1.0e-18)
                {
                    // Makeup Gain = 1.0 / RMS_IR  (Safety Margin = -6dB)
                    const double makeup = 1.0 / std::sqrt(maxChannelEnergy);
                    constexpr double safetyMargin = 0.5011872336272722;
                    result.scaleFactor = makeup * safetyMargin;
                }
                else
                {
                    DBG("LoaderThread: IR energy is too low (near silence), skipping Auto Makeup.");
                    result.scaleFactor = 1.0;
                }
            }

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
            // prepareToPlayとロジックを統一し、NUCエンジンを同条件で構築する
            int internalBlockSize = juce::nextPowerOfTwo(blockSize);
            auto sizing = computeMasteringSizing(internalBlockSize, result.targetLength);

            // Display用コピーを作成 (move前に)
            if (owner.isVisualizationEnabled()) {
                result.displayIR = trimmed;
                // 表示用にはスケールを適用しておく (処理用IRはSetImpulseでスケールされる)
                result.displayIR.applyGain(result.scaleFactor);
            }

            if (thread == nullptr) // Synchronous mode (Worker Thread)
            {
                void* mem = convo::aligned_malloc(sizeof(StereoConvolver), 64);
                new (mem) StereoConvolver();
                result.newConv = static_cast<StereoConvolver*>(mem);
                result.newConv->addRef();

                if (result.newConv->init(irL.release(), irR.release(), result.targetLength, sampleRate, irPeakLatency,
                                         sizing.maxFFTSize, internalBlockSize, sizing.firstPartition, blockSize, result.scaleFactor))
                {
                    result.success = true;
                }
                else
                {
                    result.newConv->release();
                    result.newConv = nullptr;
                    result.success = false;
                    result.errorMessage = "Failed to initialize NUC engine (Memory allocation or MKL setup failed).";
                }
                return result;
            }
            else // Async mode (Loader Thread)
            {
                // std::function (callAsync) requires CopyConstructible, so we cannot capture move-only types directly.
                // We wrap them in a shared_ptr.
                struct AsyncState {
                    convo::ScopedAlignedPtr<double> irL;
                    convo::ScopedAlignedPtr<double> irR;
                    std::shared_ptr<juce::AudioBuffer<double>> loadedIR;
                    std::shared_ptr<juce::AudioBuffer<double>> displayIR;
                };

                auto state = std::make_shared<AsyncState>();
                state->irL = std::move(irL);
                state->irR = std::move(irR);
                state->loadedIR = std::make_shared<juce::AudioBuffer<double>>(std::move(result.loadedIR));
                state->displayIR = std::make_shared<juce::AudioBuffer<double>>(std::move(result.displayIR));

                juce::MessageManager::callAsync([weakOwner = this->weakOwner, state,
                                                 length   = result.targetLength,
                                                 sr       = sampleRate,
                                                 peak     = irPeakLatency,
                                                 maxFFT   = sizing.maxFFTSize,
                                                 known    = internalBlockSize,
                                                 first    = sizing.firstPartition,
                                                 callQ    = blockSize,
                                                 isReb    = isRebuild,
                                                 file     = file,
                                                 scale    = result.scaleFactor]() mutable
                {
                    if (auto* owner = weakOwner.get())
                    {
                        owner->finalizeNUCEngineOnMessageThread(std::move(state->irL),
                                                                std::move(state->irR),
                                                                length, sr, peak, maxFFT, known, first, callQ, isReb, file,
                                                                scale, state->loadedIR, state->displayIR);
                    }
                });

                result.success = false; // finalize側でapplyNewStateを実行するため、ここではスキップ
                return result;
            }

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
        auto loadedIRShared = std::make_shared<juce::AudioBuffer<double>>(std::move(result.loadedIR));
        auto displayIRShared = std::make_shared<juce::AudioBuffer<double>>(std::move(result.displayIR));
        owner.applyNewState(result.newConv, loadedIRShared, result.loadedSR, result.targetLength, isRebuild, file, result.scaleFactor, displayIRShared);
        }
        else
        {
            // [FIX] Clean up leaked StereoConvolver if load failed or cancelled
            if (result.newConv) result.newConv->release();
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
    double scaleFactor = 1.0;
};

//--------------------------------------------------------------
// コンストラクタ
//--------------------------------------------------------------
ConvolverProcessor::ConvolverProcessor()
    : mixSmoother(1.0f)
{
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
    if (activeConvolution) { activeConvolution->release(); activeConvolution = nullptr; }

    if (fftHandle) {
        DftiFreeDescriptor(&fftHandle);
        fftHandle = nullptr;
    }
}

void ConvolverProcessor::timerCallback()
{
    // IR切り替え時のクロスフェードが完了したら、古いコンボルバーを安全に破棄する
    // [Bug G fix] wetCrossfade.isSmoothing() は非スレッドセーフ(AudioThread が同時に getNextValue() を呼ぶ)。
    // wetCrossfadeActive アトミックフラグで代替する。
    if (!wetCrossfadeActive.load(std::memory_order_acquire))
    {
        auto* doneFading = fadingOutConvolution.exchange(nullptr);
        if (doneFading != nullptr)
        {
            const juce::ScopedLock sl(trashBinLock);
            // 参照カウントを解放するために release() を呼ぶ
            trashBin.push_back({doneFading, juce::Time::getMillisecondCounter()});
        }
    }

    cleanup();
}

//--------------------------------------------------------------
// prepareToPlay
//--------------------------------------------------------------
void ConvolverProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    // 旧descriptor未解放防止
    if (fftHandle) {
        DftiFreeDescriptor(&fftHandle);
        fftHandle = nullptr;
        fftHandleSize = 0;
    }

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
        // MKL NUC に正しい knownBlockSize を渡すために、エンジンを再構築する。
        // 既存のエンジンは他スレッドで共有されている可能性があるため、複製して差し替える。
        const int internalBlockSize = juce::nextPowerOfTwo(samplesPerBlock);

        if ((rateChanged || blockChanged) && conv->irDataLength > 0)
        {
            // clone() は古いパラメータで複製するため、ここでは使えない。
            // 新しい sampleRate/blockSize で再初期化する必要があるため、
            // MKL規約に準拠した方法で手動で構築する。
            // [Bug 5 fix] newConv を try ブロック外で宣言し、irL/irR の aligned_malloc が
            // std::bad_alloc を投げた場合でも catch 節で release() を呼べるようにする。
            // (try スコープ内で宣言すると catch 節からアクセス不可 → リーク)
            StereoConvolver* newConv = nullptr;
            try
            {
                void* mem = convo::aligned_malloc(sizeof(StereoConvolver), 64);
                new (mem) StereoConvolver();
                newConv = static_cast<StereoConvolver*>(mem);
                newConv->addRef();

                convo::ScopedAlignedPtr<double> irL(static_cast<double*>(convo::aligned_malloc(conv->irDataLength * sizeof(double), 64)));
                convo::ScopedAlignedPtr<double> irR(static_cast<double*>(convo::aligned_malloc(conv->irDataLength * sizeof(double), 64)));
                std::memcpy(irL.get(), conv->irData[0], conv->irDataLength * sizeof(double));
                std::memcpy(irR.get(), conv->irData[1], conv->irDataLength * sizeof(double));

                auto sizing = computeMasteringSizing(internalBlockSize, conv->irDataLength);

                if (newConv->init(irL.release(), irR.release(),
                                  conv->irDataLength, sampleRate, conv->irLatency, sizing.maxFFTSize, internalBlockSize, sizing.firstPartition, samplesPerBlock, conv->storedScale))
                {
                    convolution.store(newConv, std::memory_order_release);

                    if (activeConvolution)
                    {
                        const juce::ScopedLock sl(trashBinLock);
                        trashBin.push_back({activeConvolution, juce::Time::getMillisecondCounter()});
                    }
                    activeConvolution = newConv;
                }
                else
                {
                    DBG("ConvolverProcessor::prepareToPlay: NUC re-init failed (MKL alloc?). Keeping existing engine.");
                    newConv->release();
                    newConv = nullptr;
                }
            }
            catch (const std::bad_alloc&)
            {
                // [Bug 5 fix] 例外発生時に newConv (refCount=1) を確実に解放する。
                if (newConv != nullptr)
                {
                    newConv->release();
                    newConv = nullptr;
                }
                DBG("ConvolverProcessor::prepareToPlay: NUC re-init failed (std::bad_alloc). Keeping existing engine.");
            }
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

    if (oldWetBufferCapacity < MAX_BLOCK_SIZE)
    {
        oldWetBufferStorage[0].reset(static_cast<double*>(convo::aligned_malloc(MAX_BLOCK_SIZE * sizeof(double), 64)));
        oldWetBufferStorage[1].reset(static_cast<double*>(convo::aligned_malloc(MAX_BLOCK_SIZE * sizeof(double), 64)));
        oldWetBufferCapacity = MAX_BLOCK_SIZE;
    }
    double* oldWetChs[2] = { oldWetBufferStorage[0].get(), oldWetBufferStorage[1].get() };
    oldWetBuffer.setDataToReferTo(oldWetChs, 2, MAX_BLOCK_SIZE);
    oldWetBuffer.clear();

    if (crossfadeRampBufferCapacity < MAX_BLOCK_SIZE)
    {
        crossfadeRampBuffer.reset(static_cast<double*>(convo::aligned_malloc(MAX_BLOCK_SIZE * sizeof(double), 64)));
        crossfadeRampBufferCapacity = MAX_BLOCK_SIZE;
    }

    wetCrossfade.reset(sampleRate, 0.02); // 20ms crossfade
    wetCrossfade.setCurrentAndTargetValue(1.0);

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

    // レイテンシー補正の初期化
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

    oldWetBufferStorage[0].reset();
    oldWetBufferStorage[1].reset();
    oldWetBufferCapacity = 0;

    crossfadeRampBuffer.reset();
    crossfadeRampBufferCapacity = 0;

    smoothingBufferStorage[0].reset();
    smoothingBufferStorage[1].reset();
    smoothingBufferCapacity = 0;

    cachedFFTBuffer.reset();
    cachedFFTBufferCapacity = 0;

    dryBuffer.setSize(0, 0);
    smoothingBuffer.setSize(0, 0);

    if (fftHandle) {
        DftiFreeDescriptor(&fftHandle);
        fftHandle = nullptr;
        fftHandleSize = 0;
    }

    // Release active convolution engine
    convolution.store(nullptr, std::memory_order_release);
    if (activeConvolution) { activeConvolution->release(); activeConvolution = nullptr; }
    auto* fading = fadingOutConvolution.exchange(nullptr);
    if (fading) fading->release();

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

void ConvolverProcessor::postCoalescedChangeNotification()
{
    if (changeNotificationPending.exchange(true, std::memory_order_acq_rel))
        return;

    auto weakThis = juce::WeakReference<ConvolverProcessor>(this);
    juce::MessageManager::callAsync([weakThis]()
    {
        if (auto* self = weakThis.get())
        {
            self->changeNotificationPending.store(false, std::memory_order_release);
            self->sendChangeMessage();
        }
    });
}

void ConvolverProcessor::requestDebouncedRebuild()
{
    if (!isIRLoaded())
        return;

    const int token = rebuildDebounceToken.fetch_add(1, std::memory_order_acq_rel) + 1;
    auto weakThis = juce::WeakReference<ConvolverProcessor>(this);

    const int debounceMs = juce::jlimit(REBUILD_DEBOUNCE_MIN_MS,
                                        REBUILD_DEBOUNCE_MAX_MS,
                                        rebuildDebounceMs.load(std::memory_order_acquire));

    juce::Timer::callAfterDelay(debounceMs, [weakThis, token]()
    {
        if (auto* self = weakThis.get())
        {
            if (self->rebuildDebounceToken.load(std::memory_order_acquire) != token)
                return;

            if (!self->isIRLoaded() || self->isLoading.load(std::memory_order_acquire))
                return;

            self->loadImpulseResponse(juce::File());
        }
    });
}

void ConvolverProcessor::rebuildAllIRsSynchronous(std::function<bool()> shouldCancel)
{
    // [Bug E fix] originalIR は std::atomic<std::shared_ptr<T>> なので .load() でスナップショットを取得。
    // rebuildThread と Message Thread の同時アクセスによるデータレースを防ぐ。
    auto snap = originalIR.load();
    if (snap && snap->getNumSamples() > 0 && originalIRSampleRate.load(std::memory_order_acquire) > 0.0)
    {
        // リビルドモードでローダーを作成し、同期的に実行
        LoaderThread loader(*this, *snap, originalIRSampleRate.load(std::memory_order_acquire), currentSpec.sampleRate, currentBufferSize, useMinPhase.load(), currentIRScale.load(std::memory_order_acquire));
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
    static constexpr int MAX_MINPHASE_FFT_SIZE = 8388608; // 2^23
    if (fftSize > MAX_MINPHASE_FFT_SIZE)
    {
        DBG("convertToMinimumPhase: fftSize (" << fftSize << ") exceeds limit. Skipping min-phase conversion to prevent excessive memory usage.");
        return {}; // 失敗/スキップを通知するために空のバッファを返す
    }

    juce::AudioBuffer<double> minPhaseIR(linearIR.getNumChannels(), numSamples);

    // MKL DFTI は自然順序で扱えるため、旧FFT経路の permute 順序問題を回避できる。
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
        auto snapIR = originalIR.load();
        if (!snapIR || snapIR->getNumSamples() == 0 || originalIRSampleRate.load(std::memory_order_acquire) <= 0.0)
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
        auto snapIR2 = originalIR.load(); // [Bug E fix] 最新スナップショットを再取得
        activeLoader = std::make_unique<LoaderThread>(*this, *snapIR2, originalIRSampleRate.load(std::memory_order_acquire), currentSpec.sampleRate, currentBufferSize, useMinPhase.load(), currentIRScale.load(std::memory_order_acquire));
    }
    else
    {
        activeLoader = std::make_unique<LoaderThread>(*this, irFile, currentSpec.sampleRate, currentBufferSize, useMinPhase.load());
        currentIrOptimized.store(optimizeForRealTime);
    }

    activeLoader->startThread();

    return true;
}

void ConvolverProcessor::handleLoadError(const juce::String& error)
{
    lastError = error;
    isLoading.store(false);
    isRebuilding.store(false, std::memory_order_release);
    // UIに通知してエラーメッセージを表示させる
    postCoalescedChangeNotification();
}

void ConvolverProcessor::cleanup()
{
    // LoaderThread のクリーンアップ (Message Thread Only)
    // 終了したスレッドのみを削除する (waitForThreadToExit(0) はブロックしない)
    for (auto it = loaderTrashBin.begin(); it != loaderTrashBin.end(); )
    {
        if ((*it)->waitForThreadToExit(0))
        {
            // [Fix] スレッドは終了済みのため、reset() を直接呼んでブロックしない。
            // JUCE の stopThread() は isThreadRunning() == false の場合に即リターンするため、
            // わざわざ detached スレッドで実行する必要はない。
            it->reset();
            it = loaderTrashBin.erase(it);
        }
        else
        {
            ++it;
        }
    }

    // 【Leak Fix】LoaderThreadの異常蓄積防止
    // スレッドが終了しない場合でも、一定数を超えたら強制削除してメモリを解放する。
    // [FIX] detached thread はプロセス終了時に未定義動作を引き起こすため、
    //       同期的なチェックと削除に切り替える。
    while (loaderTrashBin.size() > 2)
    {
        // 最も古いスレッドが終了しているか非ブロックで確認
        if (loaderTrashBin.front() && loaderTrashBin.front()->waitForThreadToExit(0))
        {
            // 終了済みなら安全に削除 (unique_ptrのデストラクタが呼ばれる)
            loaderTrashBin.pop_front();
        }
        else
        {
            // 終了していないスレッドが見つかったら、今回はここまで。次回タイマーで再試行。
            break;
        }
    }

    // StereoConvolver のクリーンアップ (Worker Threadと競合するためロックが必要)
    juce::ScopedTryLock lock(trashBinLock);
    if (!lock.isLocked()) return;

    const uint32 now = juce::Time::getMillisecondCounter();
    std::vector<StereoConvolver*> toRelease;

    for (auto it = trashBin.begin(); it != trashBin.end(); )
    {
        uint32 age = (now >= it->second) ?
                     (now - it->second) :
                     (std::numeric_limits<uint32>::max() - it->second + now);

        if (age > 10000)
        {
            toRelease.push_back(it->first);
            it = trashBin.erase(it);
        }
        else
        {
            ++it;
        }
    }

    // 安全性優先: 年齢条件(age > 10000ms)を満たしたもののみ回収する。
    // サイズ超過のみを理由とした強制解放は行わない。

    for (auto* p : toRelease) p->release();
}

void ConvolverProcessor::forceCleanup()
{
    // This method is for eager cleanup of non-blocking resources.
    // The blocking cleanup of LoaderThreads is handled by the destructor.

    std::vector<std::pair<StereoConvolver*, uint32>> stereoConvolversToDelete;
    {
        juce::ScopedLock lock(trashBinLock);
        stereoConvolversToDelete.swap(trashBin);
    }

    // 【Fix】LoaderThread のクリーンアップ漏れ防止
    // DSPCore破棄時やreleaseResources時に、残っているローダースレッドを
    // メインスレッドをブロックせずに破棄する。
    std::deque<std::unique_ptr<LoaderThread>> loadersToDelete;
    loadersToDelete.swap(loaderTrashBin);
    if (activeLoader)
        loadersToDelete.push_back(std::move(activeLoader));

    // [FIX] detached thread はプロセス終了時に未定義動作を引き起こすため、
    // シャットダウンシーケンスでは同期的にスレッドを停止する。
    // stopThread() はスレッドが既に終了している場合は即座にリターンするため、
    // ここでブロッキング呼び出しを行っても安全。
    for (auto& loader : loadersToDelete)
    {
        if (loader)
            loader->stopThread(4000);
    }
    loadersToDelete.clear(); // unique_ptrのデストラクタが呼ばれ、スレッドがクリーンアップされる

    for (auto& entry : stereoConvolversToDelete) entry.first->release();
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

std::vector<float> ConvolverProcessor::getIRWaveform() const
{
    const juce::ScopedLock sl(visualizationDataLock);
    return irWaveform;
}

std::vector<float> ConvolverProcessor::getIRMagnitudeSpectrum() const
{
    const juce::ScopedLock sl(visualizationDataLock);
    return irMagnitudeSpectrum;
}

double ConvolverProcessor::getIRSpectrumSampleRate() const
{
    const juce::ScopedLock sl(visualizationDataLock);
    return irSpectrumSampleRate;
}

//--------------------------------------------------------------
// createWaveformSnapshot
//--------------------------------------------------------------
void ConvolverProcessor::createWaveformSnapshot (const juce::AudioBuffer<double>& irBuffer)
{
    // Lock to protect access from the UI thread
    const juce::ScopedLock sl(visualizationDataLock);

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
    // Lock to protect access from the UI thread
    const juce::ScopedLock sl(visualizationDataLock);

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

    // スムージング適用 (Linear Magnitudeに対して行う)
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
    v.setProperty ("rebuildDebounceMs", rebuildDebounceMs.load(std::memory_order_acquire), nullptr);
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
    if (v.hasProperty ("rebuildDebounceMs")) setRebuildDebounceMs (static_cast<int>(v.getProperty("rebuildDebounceMs")));

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
                // IRファイルが見つからない場合のエラーハンドリング。
                // lastErrorに情報を設定し、UI側で再リンクを促す。
                // これにより、UIスレッドをブロックせずに非同期で対応できる。
                lastError = "IR not found: " + f.getFileName();
                // UIに通知してエラーメッセージを表示させる
                postCoalescedChangeNotification();
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
    rebuildDebounceMs.store(other.rebuildDebounceMs.load(std::memory_order_acquire), std::memory_order_release);

    // サンプルレート変更時にリビルドできるよう、元のIR情報を共有する
    // [Bug E fix] std::atomic<shared_ptr>::store() でアトミックに代入。
    originalIR.store(other.originalIR.load());
    originalIRSampleRate.store(other.originalIRSampleRate.load(std::memory_order_acquire), std::memory_order_release);
    {
        const juce::ScopedLock sl(irFileLock);
        currentIrFile = other.currentIrFile;
    }
    irName = other.irName;
    irLength.store(other.irLength.load(std::memory_order_acquire), std::memory_order_release);
    currentIRScale.store(other.currentIRScale.load(std::memory_order_acquire), std::memory_order_release);

    // NUC フィルターモードも同期する (rebuildAllIRsSynchronous で使用)
    nucHCMode.store(other.nucHCMode.load(std::memory_order_acquire), std::memory_order_release);
    nucLCMode.store(other.nucLCMode.load(std::memory_order_acquire), std::memory_order_release);

    // クローンを作らない (prepareToPlayが正しいレートでSCを生成するため)
    // SCはDSPCore::prepare()内のprepareToPlay、またはrebuildAllIRsSynchronousで生成する
    // activeConvolution / convolution はnullptrのままにする
    convolution.store(nullptr, std::memory_order_release);
    if (activeConvolution) { activeConvolution->release(); activeConvolution = nullptr; }
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
    rebuildDebounceMs.store(other.rebuildDebounceMs.load(std::memory_order_acquire), std::memory_order_release);

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

void ConvolverProcessor::shareConvolutionEngineFrom(const ConvolverProcessor& other)
{
    // Share the active convolution engine (Shared Pointer copy)
    auto* otherConv = other.convolution.load(std::memory_order_acquire);
    if (otherConv) otherConv->addRef();
    convolution.store(otherConv, std::memory_order_release);

    if (activeConvolution)
    {
        const juce::ScopedLock sl(trashBinLock);
        trashBin.push_back({activeConvolution, juce::Time::getMillisecondCounter()});
    }
    activeConvolution = otherConv;

    irLength.store(other.irLength.load(std::memory_order_acquire), std::memory_order_release);
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
    static constexpr double kLatencyRetargetThresholdSamples = 2.0;

    // ── デノーマル対策 ──
    // Audio Threadは専用スレッドだが、JUCEの内部実装はgetNextAudioBlock()呼び出し前に
    // FTZ/DAZを保証しない。ScopedNoDenormalsでMXCSRのFTZ/DAZビットを関数スコープで保護する。
    juce::ScopedNoDenormals noDenormals;
    // MKL VMLのデノーマル設定 (thread_localによりスレッドごとに1回のみ呼び出し)
    // vmlSetModeはスレッド局所設定。メモリ確保なし・ロックなし・wait-free。
    {
        thread_local bool vmlDenormalInitialized = false;
        if (!vmlDenormalInitialized)
        {
            vmlSetMode(VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
            vmlDenormalInitialized = true;
        }
    }

    // ── Step 1: RCU State Load (Lock-free / Wait-free) ──
    // Raw pointer load (No ref counting)
    auto* conv = convolution.load(std::memory_order_acquire);
    auto* oldConv = fadingOutConvolution.load(std::memory_order_acquire);

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
        if (std::abs(latencySmoother.getTargetValue() - static_cast<double>(totalLatency)) >= kLatencyRetargetThresholdSamples)
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

    // [Bug 1 fix] wetCrossfade ペンディングリセットの処理。
    // applyNewState() (Message Thread) は wetCrossfade フィールドへ直接書き込まず、
    // このフラグを経由して Audio Thread に委譲する。これにより Message Thread との
    // データ競合 (setCurrentAndTargetValue vs getNextValue/isSmoothing) を解消する。
    if (wetCrossfadeResetPending.exchange(false, std::memory_order_acq_rel))
    {
        wetCrossfade.setCurrentAndTargetValue(0.0);
        wetCrossfade.setTargetValue(1.0);
        // wetCrossfadeActive はすでに applyNewState() で store(true) 済み。
        // 初回ロードで oldConv が null の場合も wetCrossfade を初期化しておくことで、
        // 次の IR ロード時に正確なフェードイン開始値から始められる。
    }

    const bool isCrossfading = (oldConv != nullptr && wetCrossfadeActive.load(std::memory_order_acquire));
    int activeWetCrossfadeSamples = 0;
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
            double* delayFadeRamp = crossfadeRampBuffer.get();
            int activeDelayCrossfadeSamples = 0;

            for (; activeDelayCrossfadeSamples < numSamples; ++activeDelayCrossfadeSamples)
            {
                delayFadeRamp[activeDelayCrossfadeSamples] = crossfadeGain.getNextValue();
                if (!crossfadeGain.isSmoothing())
                {
                    ++activeDelayCrossfadeSamples;
                    break;
                }
            }
            for (int i = activeDelayCrossfadeSamples; i < numSamples; ++i)
                delayFadeRamp[i] = 1.0;

            // サブサンプル精度読み出し用ヘルパー (Catmull-Rom Interpolation)
            auto readInterpolated = [&](double delay, double* dst, int ch, int samplesToRead)
            {
                if (samplesToRead <= 0)
                    return;

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
                    int samplesFirst = std::min(samplesToRead, DELAY_BUFFER_SIZE - rPosInt);
                    std::memcpy(dst, srcBuf + rPosInt, samplesFirst * sizeof(double));
                    if (samplesToRead > samplesFirst)
                        std::memcpy(dst + samplesFirst, srcBuf, (samplesToRead - samplesFirst) * sizeof(double));
                    return;
                }
                else if (std::abs(frac - 1.0) < 1.0e-6)
                {
                    int rPosInt = (iRead + 1) & DELAY_BUFFER_MASK; // frac ~ 1.0
                    int samplesFirst = std::min(samplesToRead, DELAY_BUFFER_SIZE - rPosInt);
                    std::memcpy(dst, srcBuf + rPosInt, samplesFirst * sizeof(double));
                    if (samplesToRead > samplesFirst)
                        std::memcpy(dst + samplesFirst, srcBuf, (samplesToRead - samplesFirst) * sizeof(double));
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
                if (iRead >= 1 && iRead + samplesToRead + 2 < DELAY_BUFFER_SIZE)
                {
                    const double* s = srcBuf + iRead;
#if defined(__AVX2__)
                    const __m256d vw0 = _mm256_set1_pd(w0);
                    const __m256d vw1 = _mm256_set1_pd(w1);
                    const __m256d vw2 = _mm256_set1_pd(w2);
                    const __m256d vw3 = _mm256_set1_pd(w3);

                    // AVX2 最適化ループ
                    for (; i <= samplesToRead - 4; i += 4)
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
                    for (; i < samplesToRead; ++i)
                        dst[i] = w0 * s[i - 1] + w1 * s[i] + w2 * s[i + 1] + w3 * s[i + 2];
                }
                else
                {
                    // バッファラップアラウンド対応 (低速パス)
                    for (; i < samplesToRead; ++i)
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
            if (activeDelayCrossfadeSamples > 0)
            {
                for (int ch = 0; ch < procChannels; ++ch)
                    readInterpolated(oldDelay, oldDryBuffer.getWritePointer(ch), ch, activeDelayCrossfadeSamples);
            }

            // 2. 新しいディレイからの信号を dryBuffer に読み出す
            for (int ch = 0; ch < procChannels; ++ch)
                readInterpolated(newDelay, dryBuffer.getWritePointer(ch), ch, numSamples);

            // 3. 2つの信号をクロスフェードして dryBuffer に書き込む
            if (activeDelayCrossfadeSamples > 0)
            {
                for (int ch = 0; ch < procChannels; ++ch)
                {
                    double* newSamples = dryBuffer.getWritePointer(ch);
                    const double* oldSamples = oldDryBuffer.getReadPointer(ch);
                    const double* fadeInRamp = delayFadeRamp;
#if defined(__AVX2__)
                    int i = 0;
                    const int vEnd = activeDelayCrossfadeSamples / 4 * 4;
                    const __m256d vOne = _mm256_set1_pd(1.0);
                    for (; i < vEnd; i += 4)
                    {
                        const __m256d vFade = _mm256_loadu_pd(fadeInRamp + i);
                        const __m256d vNew = _mm256_loadu_pd(newSamples + i);
                        const __m256d vOld = _mm256_loadu_pd(oldSamples + i);
                        const __m256d vOut = _mm256_add_pd(_mm256_mul_pd(vNew, vFade),
                                                           _mm256_mul_pd(vOld, _mm256_sub_pd(vOne, vFade)));
                        _mm256_storeu_pd(newSamples + i, vOut);
                    }
                    for (; i < activeDelayCrossfadeSamples; ++i)
                        newSamples[i] = newSamples[i] * fadeInRamp[i] + oldSamples[i] * (1.0 - fadeInRamp[i]);
#else
                    for (int i = 0; i < activeDelayCrossfadeSamples; ++i)
                        newSamples[i] = newSamples[i] * fadeInRamp[i] + oldSamples[i] * (1.0 - fadeInRamp[i]);
#endif
                }
            }

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
    // MKL NUC を使用

    if (isCrossfading)
    {
        double* ramp = crossfadeRampBuffer.get();
        for (; activeWetCrossfadeSamples < numSamples; ++activeWetCrossfadeSamples)
        {
            ramp[activeWetCrossfadeSamples] = wetCrossfade.getNextValue();
            if (!wetCrossfade.isSmoothing())
            {
                ++activeWetCrossfadeSamples;
                break;
            }
        }
        for (int i = activeWetCrossfadeSamples; i < numSamples; ++i)
            ramp[i] = 1.0;

        if (!wetCrossfade.isSmoothing())
            wetCrossfadeActive.store(false, std::memory_order_release);
    }

    const double headroom = CONVOLUTION_HEADROOM_GAIN;

    const double* wetGains = nullptr;
    const double* dryGains = nullptr;

    // スムージングゲインの計算
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
    // NUC呼び出しサイズを量子化し、呼び出し長のばらつきを抑える。
    const int quantizedCallSamples = juce::jmax(1, conv->callQuantumSamples);
    const int prewarmedMaxSamples = juce::jmax(1, conv->prewarmedMaxSamples);
    const int guardedCallSamples = juce::jmin(quantizedCallSamples, prewarmedMaxSamples);

    // NUCへの呼び出しサイズは guardedCallSamples を基準量子として使用する。
    // 実デバイスの可変ブロック長に対応するため、末尾は chunkSamples (< callLen) で安全に処理する。
    // これにより、非倍数ブロックでも無音化せず連続再生を維持する。
    const int callLen = guardedCallSamples;

    constexpr int kSmallMixThreshold = 192;

    auto mixSmoothingSmall = [](double* dst,
                                const double* wet,
                                const double* dry,
                                const double* wetGain,
                                const double* dryGain,
                                int n) noexcept
    {
#if defined(__AVX2__)
        int i = 0;
        const int vEnd = n / 4 * 4;
        for (; i < vEnd; i += 4)
        {
            const __m256d vWet = _mm256_loadu_pd(wet + i);
            const __m256d vDry = _mm256_loadu_pd(dry + i);
            const __m256d vWG = _mm256_loadu_pd(wetGain + i);
            const __m256d vDG = _mm256_loadu_pd(dryGain + i);
            const __m256d vOut = _mm256_add_pd(_mm256_mul_pd(vWet, vWG), _mm256_mul_pd(vDry, vDG));
            _mm256_storeu_pd(dst + i, vOut);
        }
        for (; i < n; ++i)
            dst[i] = wet[i] * wetGain[i] + dry[i] * dryGain[i];
#else
        for (int i = 0; i < n; ++i)
            dst[i] = wet[i] * wetGain[i] + dry[i] * dryGain[i];
#endif
    };

    auto mixSteadySmall = [](double* dst,
                             const double* wet,
                             const double* dry,
                             double wetG,
                             double dryG,
                             int n) noexcept
    {
#if defined(__AVX2__)
        int i = 0;
        const int vEnd = n / 4 * 4;
        const __m256d vWG = _mm256_set1_pd(wetG);
        const __m256d vDG = _mm256_set1_pd(dryG);
        for (; i < vEnd; i += 4)
        {
            const __m256d vWet = _mm256_loadu_pd(wet + i);
            const __m256d vDry = _mm256_loadu_pd(dry + i);
            const __m256d vOut = _mm256_add_pd(_mm256_mul_pd(vWet, vWG), _mm256_mul_pd(vDry, vDG));
            _mm256_storeu_pd(dst + i, vOut);
        }
        for (; i < n; ++i)
            dst[i] = wet[i] * wetG + dry[i] * dryG;
#else
        for (int i = 0; i < n; ++i)
            dst[i] = wet[i] * wetG + dry[i] * dryG;
#endif
    };

    auto scaleDrySmall = [](double* dst,
                            const double* dry,
                            const double* gain,
                            int n) noexcept
    {
#if defined(__AVX2__)
        int i = 0;
        const int vEnd = n / 4 * 4;
        for (; i < vEnd; i += 4)
        {
            const __m256d vDry = _mm256_loadu_pd(dry + i);
            const __m256d vGain = _mm256_loadu_pd(gain + i);
            _mm256_storeu_pd(dst + i, _mm256_mul_pd(vDry, vGain));
        }
        for (; i < n; ++i)
            dst[i] = dry[i] * gain[i];
#else
        for (int i = 0; i < n; ++i)
            dst[i] = dry[i] * gain[i];
#endif
    };


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
            const int chunkSamples = juce::jmin(callLen, numSamples - processed);

            // 1. Process Convolution (Unified Interface)
            const double* input = inputBase + processed;
            double* wetOut = wetBase + processed;

            conv->process(ch, input, wetOut, chunkSamples);

            // クロスフェード処理
            if (isCrossfading)
            {
                const int crossfadeSamplesThisChunk = juce::jlimit(0, chunkSamples, activeWetCrossfadeSamples - processed);
                if (crossfadeSamplesThisChunk > 0)
                {
                    double* oldWetOut = oldWetBuffer.getWritePointer(ch, processed);
                    oldConv->process(ch, input, oldWetOut, crossfadeSamplesThisChunk);

                    const double* fadeInRamp = crossfadeRampBuffer.get() + processed;
#if defined(__AVX2__)
                    int i = 0;
                    const int vEnd = crossfadeSamplesThisChunk / 4 * 4;
                    const __m256d vOne = _mm256_set1_pd(1.0);
                    for (; i < vEnd; i += 4)
                    {
                        const __m256d vFade = _mm256_loadu_pd(fadeInRamp + i);
                        const __m256d vNew = _mm256_loadu_pd(wetOut + i);
                        const __m256d vOld = _mm256_loadu_pd(oldWetOut + i);
                        const __m256d vOut = _mm256_add_pd(_mm256_mul_pd(vNew, vFade),
                                                           _mm256_mul_pd(vOld, _mm256_sub_pd(vOne, vFade)));
                        _mm256_storeu_pd(wetOut + i, vOut);
                    }
                    for (; i < crossfadeSamplesThisChunk; ++i)
                    {
                        const double fade = fadeInRamp[i];
                        wetOut[i] = wetOut[i] * fade + oldWetOut[i] * (1.0 - fade);
                    }
#else
                    for (int i = 0; i < crossfadeSamplesThisChunk; ++i)
                    {
                        const double fade = fadeInRamp[i];
                        wetOut[i] = wetOut[i] * fade + oldWetOut[i] * (1.0 - fade);
                    }
#endif
                }
            }

            // Note: StereoConvolver::process guarantees output is written to wetOut
            const double* wetSignal = wetOut;
            int validWetSamples = chunkSamples; // Assumed valid after process

            // 3. Mix (Fused Loop: Copy + Gain + Mix)
            double* dst = dstBase + processed;
            const double* dry = dryBase + processed;

            if (isSmoothing)
            {
                if (validWetSamples > 0)
                {
                    if (validWetSamples < kSmallMixThreshold)
                    {
                        mixSmoothingSmall(dst, wetSignal, dry,
                                          wetGains + processed,
                                          dryGains + processed,
                                          validWetSamples);
                    }
                    else
                    {
                        // MKL VML を使用した Dry/Wet ミキシング (スムージング時)
                        // dst = (wet * wetGains) + (dry * dryGains)
                        // oldDryBuffer を一時的なスクラッチバッファとして使用
                        double* temp = oldDryBuffer.getWritePointer(ch);
                        vdMul(validWetSamples, wetSignal, wetGains + processed, temp);
                        vdMul(validWetSamples, dry, dryGains + processed, dst);
                        vdAdd(validWetSamples, dst, temp, dst);
                    }
                }

                // Wet信号が無効な区間 (畳み込みの初期レイテンシーなど)
                if (chunkSamples > validWetSamples)
                {
                    const int remainder = chunkSamples - validWetSamples;
                    if (remainder < kSmallMixThreshold)
                    {
                        scaleDrySmall(dst + validWetSamples,
                                      dry + validWetSamples,
                                      dryGains + processed + validWetSamples,
                                      remainder);
                    }
                    else
                    {
                        vdMul(remainder, dry + validWetSamples, dryGains + processed + validWetSamples, dst + validWetSamples);
                    }
                }
            }
            else
            {
                // 定常状態 (99%のケース) -> MKL BLAS を使用して最適化
                if (validWetSamples > 0)
                {
                    if (validWetSamples < kSmallMixThreshold)
                    {
                        mixSteadySmall(dst, wetSignal, dry, wetG, dryG, validWetSamples);
                    }
                    else
                    {
                        // dst = dryG * dry + wetG * wetSignal
                        // cblas_daxpby (y := alpha*x + beta*y) を使用
                        // 1. dst に dry をコピー
                        cblas_dcopy(validWetSamples, dry, 1, dst, 1);
                        // 2. dst = wetG * wetSignal + dryG * dst (dstはdryで初期化済み)
                        cblas_daxpby(validWetSamples, wetG, wetSignal, 1, dryG, dst, 1);
                    }
                }

                // Wetが無効な区間 (初期レイテンシー等) -> Dryのみ出力
                if (chunkSamples > validWetSamples)
                {
                    const int remainder = chunkSamples - validWetSamples;
                    if (remainder < kSmallMixThreshold)
                    {
                        for (int i = 0; i < remainder; ++i)
                            dst[validWetSamples + i] = dry[validWetSamples + i] * dryG;
                    }
                    else
                    {
                        cblas_dcopy(remainder, dry + validWetSamples, 1, dst + validWetSamples, 1);
                        cblas_dscal(remainder, dryG, dst + validWetSamples, 1);
                    }
                }
            }

            processed += chunkSamples;
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

        // IRがロード済みなら、短時間の連続操作をまとめて1回だけリビルドする
        requestDebouncedRebuild();
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

void ConvolverProcessor::setRebuildDebounceMs(int ms)
{
    const int clampedMs = juce::jlimit(REBUILD_DEBOUNCE_MIN_MS, REBUILD_DEBOUNCE_MAX_MS, ms);
    rebuildDebounceMs.store(clampedMs, std::memory_order_release);
}

int ConvolverProcessor::getRebuildDebounceMs() const
{
    return rebuildDebounceMs.load(std::memory_order_acquire);
}

int ConvolverProcessor::getLatencySamples() const
{
    if (auto* conv = convolution.load(std::memory_order_acquire))
        return conv->latency;
    return 0;
}

void ConvolverProcessor::setUseMinPhase(bool shouldUseMinPhase)
{
    if (useMinPhase.load() != shouldUseMinPhase)
    {
        useMinPhase.store(shouldUseMinPhase);
        listeners.call(&Listener::convolverParamsChanged, this);

        // 連続切り替えをまとめ、最新状態のみで再構築する
        requestDebouncedRebuild();
    }
}

//==============================================================================
// setNUCFilterModes  ─ Message Thread のみ
//
// HC/LC モードを atomic に保存し、IR がロード済みなら非同期で NUC を再構築する。
// 再構築は rebuildAllIRs() 経由でクロスフェード付きで行われる。
//==============================================================================
void ConvolverProcessor::setNUCFilterModes(convo::HCMode hcMode, convo::LCMode lcMode)
{
    const int newHC = static_cast<int>(hcMode);
    const int newLC = static_cast<int>(lcMode);

    const bool changed = (nucHCMode.exchange(newHC) != newHC) ||
                         (nucLCMode.exchange(newLC) != newLC);

    if (changed)
        requestDebouncedRebuild();
}

//==============================================================================
// finalizeNUCEngineOnMessageThread
// LoaderThreadから委譲されたNUCエンジン構築（メッセージスレッド専用）
//==============================================================================
void ConvolverProcessor::finalizeNUCEngineOnMessageThread(convo::ScopedAlignedPtr<double> irL,
                                                          convo::ScopedAlignedPtr<double> irR,
                                                          int length,
                                                          double sr,
                                                          int peakDelay,
                                                          int maxFFTSize,
                                                          int knownBlockSize,
                                                          int firstPartition,
                                                          int preferredCallSize,
                                                          bool isRebuild,
                                                          const juce::File& irFile,
                                                          double scaleFactor,
                                                          std::shared_ptr<juce::AudioBuffer<double>> loadedIR,
                                                          std::shared_ptr<juce::AudioBuffer<double>> displayIR)
{
    // ここはMessage Thread上で実行されるためMKL規約を完全に遵守する
    // メモリ確保失敗に備えて try-catch を使用する
    try
    {
        void* mem = convo::aligned_malloc(sizeof(StereoConvolver), 64);
        new (mem) StereoConvolver();
        auto* newConv = static_cast<StereoConvolver*>(mem);
        newConv->addRef();

        if (newConv->init(irL.release(), irR.release(), length, sr, peakDelay,
                          maxFFTSize, knownBlockSize, firstPartition, preferredCallSize, scaleFactor,
                          [&]() -> const convo::FilterSpec* {
                              // NUC フィルタースペックを Message Thread 上で構築
                              static thread_local convo::FilterSpec spec;
                              spec.sampleRate = sr;
                              spec.hcMode = static_cast<convo::HCMode>(nucHCMode.load(std::memory_order_acquire));
                              spec.lcMode = static_cast<convo::LCMode>(nucLCMode.load(std::memory_order_acquire));
                              return &spec;
                          }()))
        {
            applyNewState(newConv, loadedIR, sr, length, isRebuild, irFile, scaleFactor, displayIR);
        }
        else
        {
            newConv->release();
            handleLoadError("Failed to initialize NUC engine (Memory allocation or MKL setup failed).");
        }
    }
    catch (const std::bad_alloc&)
    {
        handleLoadError("Failed to initialize NUC engine (Memory allocation or MKL setup failed).");
    }
}

void ConvolverProcessor::applyNewState(StereoConvolver* newConv,
                                       std::shared_ptr<juce::AudioBuffer<double>> loadedIR,
                                       double loadedSR,
                                       int targetLength,
                                       bool isRebuild,
                                       const juce::File& file,
                                       double scaleFactor,
                                       std::shared_ptr<juce::AudioBuffer<double>> displayIR)
{
    // 元データの更新 (新規ロード時のみ)
    if (!isRebuild)
    {
        originalIR.store(loadedIR); // [Bug E fix] atomic store
        originalIRSampleRate.store(loadedSR, std::memory_order_release);  // [Bug 4 fix] atomic store
        {
            const juce::ScopedLock sl(irFileLock);
            currentIrFile = file;
        }
        irName = file.getFileNameWithoutExtension();
        currentIRScale.store(scaleFactor, std::memory_order_release);  // [Bug 4 fix] atomic store
    }

    // スナップショット更新 (表示用)
    if (visualizationEnabled)
    {
        createWaveformSnapshot(*displayIR);
        createFrequencyResponseSnapshot(*displayIR, loadedSR);
    }

    auto* convToFadeOut = activeConvolution;
    activeConvolution = newConv;

    // Audio Threadが新しいコンボルバーを参照するようにアトミックに更新
    convolution.store(newConv, std::memory_order_release);

    // [Bug C fix] wet フェードイン (0→1) は初回ロードを含む常に起動する。
    // 旧実装は convToFadeOut != nullptr のブロック内のみで起動していたため、
    // 初回ロード時には wetCrossfade が開始されず wet 信号がフルレベルで瞬間出力されていた。
    // [Bug G fix] wetCrossfadeActive を Message Thread 側の通知フラグとして立てる。
    // [Bug 1 fix] wetCrossfade フィールドへの直接書き込みをここで行うと Audio Thread の
    //             getNextValue()/isSmoothing() と競合 (データ競合) する。
    //             代わりに wetCrossfadeResetPending を立てて、Audio Thread の process() 先頭で
    //             初期化させる (Audio Thread のみが wetCrossfade フィールドを操作する設計)。
    wetCrossfadeResetPending.store(true, std::memory_order_release);
    wetCrossfadeActive.store(true, std::memory_order_release);

    // 古いコンボルバーがあれば、フェードアウトも開始する
    if (convToFadeOut != nullptr)
    {
        // 既に別のクロスフェードが進行中だった場合、その古いエンジンは即座に破棄リストへ
        auto* interruptedFade = fadingOutConvolution.exchange(convToFadeOut);
        if (interruptedFade != nullptr)
        {
            const juce::ScopedLock sl(trashBinLock);
            trashBin.push_back({interruptedFade, juce::Time::getMillisecondCounter()});
        }
    }

    irLength.store(targetLength, std::memory_order_release);
    currentSampleRate.store(currentSpec.sampleRate);

    isLoading.store(false);
    isRebuilding.store(false, std::memory_order_release);
    postCoalescedChangeNotification();
}

void ConvolverProcessor::StereoConvolver::reset()
{
    if (nucConvolvers[0]) nucConvolvers[0]->Reset();
    if (nucConvolvers[1]) nucConvolvers[1]->Reset();
}

void ConvolverProcessor::StereoConvolver::process(int channel, const double* in, double* out, int numSamples)
{
    if (channel < 0 || channel >= 2 || !nucConvolvers[channel])
    {
        std::memset(out, 0, numSamples * sizeof(double));
        return;
    }

    nucConvolvers[channel]->Add(in, numSamples);
    const int got = nucConvolvers[channel]->Get(out, numSamples);
    if (got < numSamples)
        std::memset(out + got, 0, (numSamples - got) * sizeof(double));
}
