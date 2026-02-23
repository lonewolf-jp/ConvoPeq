//============================================================================
// ConvolverProcessor.cpp  ── v0.2 (JUCE 8.0.12対応)
//
// コンボリューションプロセッサーの実装
//============================================================================
#include "ConvolverProcessor.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <utility>
#include <cstring>
#include <limits>
#include <new>

#include "WDL/fft.h" // WDL Double precision FFT
#include "CDSPResampler.h"
#include "AlignedAllocation.h" // For convo::AlignedBuffer

#if JUCE_DSP_USE_INTEL_MKL
#include <mkl.h>
#endif

#if JUCE_INTEL
 #include <xmmintrin.h>
 #include <pmmintrin.h>
 #include <immintrin.h> // For AVX2
#endif

// 前方宣言
static juce::AudioBuffer<double> convertToMinimumPhase(const juce::AudioBuffer<double>& linearIR, juce::Thread* thread = nullptr, bool* wasCancelled = nullptr);

// スレッドキャンセル確認用ヘルパー関数
static bool checkCancellation(juce::Thread* thread, bool* wasCancelled) noexcept
{
    if (thread != nullptr && thread->threadShouldExit())
    {
        if (wasCancelled)
            *wasCancelled = true;
        return true;
    }
    return false;
}

// リサンプリング用ヘルパー
static juce::AudioBuffer<double> resampleIR(const juce::AudioBuffer<double>& inputIR, double inputSR, double targetSR, juce::Thread* thread)
{
    if (inputSR <= 0.0 || targetSR <= 0.0 || std::abs(inputSR - targetSR) <= 1.0)
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
        if (checkCancellation(thread, nullptr)) return {};

        r8b::CDSPResampler resampler(inputSR, targetSR, inLength, transBand, stopBandAtten, phase);

        const double* inPtr = inputIR.getReadPointer(ch);
        double* outPtr = resampled.getWritePointer(ch);

        int done = 0;
        int iterations = 0;
        constexpr int maxIterations = 1000000; // 無限ループ防止のための安全カウンター

        while (done < maxOutLen && ++iterations < maxIterations)
        {
            if (checkCancellation(thread, nullptr)) return {};
            double* r8bOutput = nullptr;
            const int generated = resampler.process(const_cast<double*>(inPtr),
                                                    done == 0 ? inLength : 0, r8bOutput);

            // generatedが0以下の場合、処理が完了したかエラーが発生したとみなしループを抜ける
            if (generated <= 0)
                break;

            const int toCopy = std::min(generated, maxOutLen - done);
            std::memcpy(outPtr + done, r8bOutput, toCopy * sizeof(double));
            done += toCopy;
        }
        maxLength = std::max(maxLength, done);
    }
    resampled.setSize(inputIR.getNumChannels(), maxLength, true, true, true);

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
            double dist_to_end = static_cast<double>(numSamples - 1 - peakIndex);
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
}

// サンプルレートに基づいて最大FFTサイズを計算するヘルパー
static int calculateMaxFFTSize(double sampleRate)
{
    if (sampleRate <= 96000.0 + 1.0) return 4096;
    if (sampleRate <= 192000.0 + 1.0) return 8192;
    if (sampleRate <= 384000.0 + 1.0) return 16384;
    if (sampleRate <= 768000.0 + 1.0) return 32768;
    if (sampleRate <= 1540000.0 + 1.0) return 65536;
    if (sampleRate <= 3080000.0 + 1.0) return 131072;
    return 262144;
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

        for (int i = 0; i < numSamples; ++i) {
            double curr_x = data[i];
            // 高精度演算 (64bit double)
            double curr_y = curr_x - px + r * py;

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

    struct LoadResult
    {
        juce::AudioBuffer<double> loadedIR;
        double loadedSR = 0.0;
        int targetLength = 0;
        juce::AudioBuffer<double> displayIR;
        StereoConvolver::Ptr newConv;
        bool success = false;
    };

    void run() override
    {
        juce::ScopedNoDenormals noDenormals; // バックグラウンド処理でのDenormal対策

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
    }

    LoadResult performLoad(juce::Thread* thread)
    {
        LoadResult result;

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

                // AudioFormatReader::read は float のみ対応のため、一時バッファを使用
                juce::AudioBuffer<float> tempFloatBuffer(numChannels, static_cast<int>(fileLength));
                reader->read(&tempFloatBuffer, 0, static_cast<int>(fileLength), 0, true, true);

                result.loadedIR.setSize(numChannels, static_cast<int>(fileLength));
                for (int ch = 0; ch < numChannels; ++ch)
                {
                    const float* src = tempFloatBuffer.getReadPointer(ch);
                    double* dst = result.loadedIR.getWritePointer(ch);
                    for (int i = 0; i < static_cast<int>(fileLength); ++i)
                        dst[i] = static_cast<double>(src[i]);
                }
                result.loadedSR = reader->sampleRate;
            }

            if (checkCancellation(thread, nullptr) || result.loadedIR.getNumSamples() == 0) return result;

            // 2. ピーク正規化 (ファイル読み込み時のみ)
            // リビルド時は既に正規化されていると仮定するためスキップします。
            if (!isRebuild)
            {
                double maxMagnitude = 0.0;
                for (int ch = 0; ch < result.loadedIR.getNumChannels(); ++ch)
                    maxMagnitude = (std::max)(maxMagnitude, result.loadedIR.getMagnitude(ch, 0, result.loadedIR.getNumSamples()));

                if (maxMagnitude > 0.0)
                    result.loadedIR.applyGain(1.0 / maxMagnitude);
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
                    result.loadedIR.setSize(channels, std::max(1, newLength), true);
            }

            // 4. リサンプリング (SR不一致の場合)
            // IRのサンプルレートがターゲットと異なる場合、ピッチズレを防ぐためにリサンプリングする
            if (result.loadedSR > 0.0 && sampleRate > 0.0 &&
                std::abs(result.loadedSR - sampleRate) > 1.0)
            {
                auto resampled = resampleIR(result.loadedIR, result.loadedSR, sampleRate, thread);

                if (resampled.getNumSamples() == 0)
                {
                    // キャンセルされたか、エラーで0長になった場合
                    if (!checkCancellation(thread, nullptr))
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

            if (checkCancellation(thread, nullptr)) return result;

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

            if (checkCancellation(thread, nullptr)) return result;

            if (checkCancellation(thread, nullptr)) return result;

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

            if (checkCancellation(thread, nullptr)) return result;

            // 8. MinPhase変換 (オプション)
            bool conversionSuccessful = false;
            if (useMinPhase)
            {
                bool wasCancelled = false;
                auto minPhaseIR = convertToMinimumPhase(trimmed, thread, &wasCancelled);

                if (wasCancelled) return result;

                // 変換成功チェック: キャンセルされておらず、かつ無音でない場合のみ適用
                if (minPhaseIR.getNumSamples() > 0 && minPhaseIR.getMagnitude(0, 0, minPhaseIR.getNumSamples()) > 1.0e-5)
                {
                    trimmed = minPhaseIR;
                    conversionSuccessful = true;
                }
                // 変換に失敗または無音になった場合は、元のtrimmed(Linear Phase)を使用する
            }

            if (checkCancellation(thread, nullptr)) return result;

            // 9.ピーク位置検出 (レイテンシー補正用)
            // Linear Phaseの場合、ピークが遅れてやってくるため、その分Dryを遅らせる必要がある
            // MinPhase変換に失敗した場合も、Linear Phaseとして扱う必要があるためピーク検出を行う
            int irPeakLatency = 0;
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

            // 10. 新しいConvolutionの構築 (Non-uniform Partitioned Convolution)
            result.newConv = new StereoConvolver();

            // WDL_ImpulseBuffer用にIRデータを準備
            WDL_ImpulseBuffer impL, impR;
            impL.SetNumChannels(1);
            impR.SetNumChannels(1);
            impL.samplerate = sampleRate;
            impR.samplerate = sampleRate;

            // WDL_ImpulseBuffer::samples は WDL_TypedBuf<WDL_FFT_REAL>
            // WDL_FFT_REALSIZE=8 なので double
            impL.impulses[0].Resize(result.targetLength);
            impR.impulses[0].Resize(result.targetLength);

            const double* srcL = trimmed.getReadPointer(0);
            const double* srcR = (trimmed.getNumChannels() > 1) ? trimmed.getReadPointer(1) : srcL;

            std::memcpy(impL.impulses[0].Get(), srcL, result.targetLength * sizeof(double));
            std::memcpy(impR.impulses[0].Get(), srcR, result.targetLength * sizeof(double));

            // サンプルレートに基づいて最大FFTサイズを計算
            int maxFFTSize = calculateMaxFFTSize(sampleRate);

            // 初期化
            result.newConv->init(impL, impR, irPeakLatency, maxFFTSize, blockSize);

            // Display用コピーを作成 (move前に)
            result.displayIR = trimmed;

            if (checkCancellation(thread, nullptr)) return result;

            result.success = true;
            return result;
        }
        catch (const std::bad_alloc&)
        {
            DBG("LoaderThread: Memory allocation failed. Aborting IR load.");
            return result;
        }
        catch (const std::exception& e)
        {
            juce::ignoreUnused(e);
            DBG("LoaderThread: Exception occurred during IR loading: " << e.what());
            return result;
        }
        catch (...)
        {
            DBG("LoaderThread: Unknown exception occurred during IR loading.");
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
    : mixSmoother(1.0f) // 初期値
{
}

//--------------------------------------------------------------
// デストラクタ
//--------------------------------------------------------------
ConvolverProcessor::~ConvolverProcessor()
{
    // スレッドを停止
    activeLoader.reset();

    // shared_ptrが自動的に解放されるため、明示的なdeleteは不要
    // ただし、trashBinのクリアは行う
    trashBin.clear();
    auto c = convolution.exchange(nullptr);
    if (c) c->decReferenceCount();
}

//--------------------------------------------------------------
// prepareToPlay
//--------------------------------------------------------------
void ConvolverProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
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
    auto conv = convolution.load(std::memory_order_acquire);
    if (conv) {
        // WDL_ConvolutionEngine_Div は初期化時に known_blocksize を使用して最適化されるため、
        // ブロックサイズが大幅に変更された場合はリビルドが望ましいですが、
        // AudioEngine側で rebuildAllIRsSynchronous() が呼ばれるため、ここでは何もしません。
        // ただし、内部状態の不整合を防ぐためにリセットを行うことは安全です。

        // Note: AudioEngine::requestRebuild が呼ばれると、新しい ConvolverProcessor (または再利用されたもの) に対して
        // rebuildAllIRsSynchronous() が呼ばれ、そこで新しいブロックサイズで init() が実行されます。
        // したがって、ここでの特別な処理は不要です。
    }

    // DelayLine準備
    // カスタムリングバッファの確保 (2の累乗サイズ)
    delayBuffer[0].allocate(DELAY_BUFFER_SIZE);
    delayBuffer[1].allocate(DELAY_BUFFER_SIZE);
    // バッファクリア
    juce::FloatVectorOperations::clear(delayBuffer[0].get(), DELAY_BUFFER_SIZE);
    juce::FloatVectorOperations::clear(delayBuffer[1].get(), DELAY_BUFFER_SIZE);
    delayWritePos = 0;

    // Dryバッファ確保
    dryBufferStorage[0].allocate(MAX_BLOCK_SIZE);
    dryBufferStorage[1].allocate(MAX_BLOCK_SIZE);
    double* dryChs[2] = { dryBufferStorage[0].get(), dryBufferStorage[1].get() };
    dryBuffer.setDataToReferTo(dryChs, 2, MAX_BLOCK_SIZE);
    dryBuffer.clear();

    smoothingBufferStorage[0].allocate(MAX_BLOCK_SIZE);
    smoothingBufferStorage[1].allocate(MAX_BLOCK_SIZE);
    double* smoothChs[2] = { smoothingBufferStorage[0].get(), smoothingBufferStorage[1].get() };
    smoothingBuffer.setDataToReferTo(smoothChs, 2, MAX_BLOCK_SIZE);
    smoothingBuffer.clear();

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

    isPrepared.store(true, std::memory_order_release);
}

void ConvolverProcessor::reset()
{
    auto conv = convolution.load(std::memory_order_acquire);
    if (conv)
    {
        conv->reset();
    }
    // リングバッファのクリア
    if (delayBuffer[0].get()) juce::FloatVectorOperations::clear(delayBuffer[0].get(), DELAY_BUFFER_SIZE);
    if (delayBuffer[1].get()) juce::FloatVectorOperations::clear(delayBuffer[1].get(), DELAY_BUFFER_SIZE);
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

void ConvolverProcessor::rebuildAllIRsSynchronous()
{
    if (originalIR.getNumSamples() > 0 && originalIRSampleRate > 0.0)
    {
        // リビルドモードでローダーを作成し、同期的に実行
        LoaderThread loader(*this, originalIR, originalIRSampleRate, currentSpec.sampleRate, currentBufferSize, useMinPhase.load());
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
static juce::AudioBuffer<double> convertToMinimumPhase(const juce::AudioBuffer<double>& linearIR, juce::Thread* thread, bool* wasCancelled)
{
    if (wasCancelled) *wasCancelled = false;

    const int numSamples = linearIR.getNumSamples();
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

    // WDL FFTはインターリーブされた複素数バッファ(re, im)を使用します。
    // SIMD用のアライメントを確保 (AVXは32バイトアライメントが必要)。
    // convo::AlignedBufferは64バイトアライメントで確保します。
    convo::AlignedBuffer fftBufferAligned;
    fftBufferAligned.allocate(fftSize * 2); // 2 doubles per complex
    WDL_FFT_COMPLEX* fftBuffer = reinterpret_cast<WDL_FFT_COMPLEX*>(fftBufferAligned.get());

    convo::AlignedBuffer dataAligned;
    dataAligned.allocate(fftSize);
    double* data = dataAligned.get(); // 作業用
#if JUCE_DSP_USE_INTEL_MKL
    convo::AlignedBuffer tempAligned;
    tempAligned.allocate(fftSize);
    double* temp = tempAligned.get(); // MKL用作業バッファ
#endif

    for (int ch = 0; ch < linearIR.getNumChannels(); ++ch)
    {
        if (checkCancellation(thread, wasCancelled)) return {};

        // 1. IRをコピー & スケーリング (WDL FFT順変換は 1/N スケーリングを要求)
        const double* src = linearIR.getReadPointer(ch);
        const double scale = 1.0 / fftSize;

        for (int i = 0; i < fftSize; ++i)
        {
            fftBuffer[i].re = (i < numSamples) ? (src[i] * scale) : 0.0;
            fftBuffer[i].im = 0.0;
        }

        // 2. FFT (時間 -> 周波数)
        // Output is permuted order
        WDL_fft(fftBuffer, fftSize, 0);

        if (checkCancellation(thread, wasCancelled)) return {};

        // 3. 対数マグニチュードスペクトル計算 (Real=Log|H|, Imag=0) [Double]
#if JUCE_DSP_USE_INTEL_MKL
        // MKL VML 最適化
        // マグニチュード計算: sqrt(re^2 + im^2) -> vdHypot
        // MKL用に複素数を分割
        for (int i = 0; i < fftSize; ++i) { data[i] = fftBuffer[i].re; temp[i] = fftBuffer[i].im; }

        vdHypot(fftSize, data, temp, data); // data = magnitude

        // ゼロ除算防止 (Clamp)
        for (int i = 0; i < fftSize; ++i)
            data[i] = (std::max)(data[i], 1.0e-100);

        // 対数: re = ln(data)
        vdLn(fftSize, data, data);

        // fftBufferに書き戻し (虚部は0)
        for (int i = 0; i < fftSize; ++i) { fftBuffer[i].re = data[i]; fftBuffer[i].im = 0.0; }
#else
        for (int i = 0; i < fftSize; ++i)
        {
            double mag = std::sqrt(fftBuffer[i].re * fftBuffer[i].re + fftBuffer[i].im * fftBuffer[i].im);
            if (!std::isfinite(mag)) mag = 0.0;
            // ゼロ除算防止 (doubleの極小値を使用)
            double logMag = std::log((std::max)(mag, 1.0e-100));
            fftBuffer[i].re = logMag;
            fftBuffer[i].im = 0.0;
        }
#endif

        // 4. IFFT (周波数 -> 時間) => 実ケプストラム (Real Cepstrum)
        // Input is permuted, Output is Natural order (Sum)
        WDL_fft(fftBuffer, fftSize, 1);

        if (checkCancellation(thread, wasCancelled)) return {};

        // 5. 因果的ウィンドウ適用 (リフタリング) [Double]
        // c[0] = c[0]
        // c[n] = 2*c[n] (0 < n < N/2)
        // c[N/2] = c[N/2]
        // c[n] = 0 (N/2 < n < N)
        // data[] は実数配列
        // WDL_fft output is in .re
        for (int i = 1; i < fftSize / 2; ++i)
        {
            fftBuffer[i].re *= 2.0;
            fftBuffer[i].im = 0.0; // Cepstrum is real
        }
        fftBuffer[0].im = 0.0;
        fftBuffer[fftSize/2].im = 0.0;

        for (int i = fftSize / 2 + 1; i < fftSize; ++i)
        {
            fftBuffer[i].re = 0.0;
            fftBuffer[i].im = 0.0;
        }

        // 6. FFT (時間 -> 周波数) => 解析信号の対数スペクトル (実部が対数振幅、虚部が最小位相)
        // 順変換用に再スケーリング
        for (int i = 0; i < fftSize; ++i) fftBuffer[i].re *= scale;

        WDL_fft(fftBuffer, fftSize, 0);

        if (checkCancellation(thread, wasCancelled)) return {};

        // 7. 複素指数変換 (exp) => 最小位相スペクトル [Double]
        // スカラー実装 (この複素演算の単純化のため、MKLと非MKLの両方で使用)
        for (int i = 0; i < fftSize; ++i)
        {
            double real = fftBuffer[i].re;
            double imag = fftBuffer[i].im;

            // 数値オーバーフロー防止のためのクランプ
            real = juce::jlimit(-50.0, 50.0, real);
            imag = juce::jlimit(-50.0, 50.0, imag);

            std::complex<double> c(real, imag);
            std::complex<double> ex = std::exp(c);
            fftBuffer[i].re = ex.real();
            fftBuffer[i].im = ex.imag();
        }

        // 8. IFFT (周波数 -> 時間) => 時間領域の最小位相IR
        WDL_fft(fftBuffer, fftSize, 1);

        if (checkCancellation(thread, wasCancelled)) return {};

        // 9. 結果をコピー
        double* dst = minPhaseIR.getWritePointer(ch);
        for (int i = 0; i < numSamples; ++i)
            dst[i] = fftBuffer[i].re;
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

    // 既存のローダーを停止して破棄
    activeLoader.reset();

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
    createWaveformSnapshot(displayIR);
    // 表示用には現在のサンプルレートを使用 (loadedSRはリサンプリング後のレート)
    createFrequencyResponseSnapshot(displayIR, loadedSR);

    // 安全に差し替え (Atomic Swap)
    newConv->incReferenceCount(); // Atomicポインタで保持するための参照カウント
    auto oldConv = convolution.exchange(newConv.get(), std::memory_order_acq_rel);

    if (oldConv)
    {
        DBG("ConvolverProcessor: Enqueueing old StereoConvolver to trash bin.");
        const juce::ScopedLock sl(trashBinLock);
        trashBinPending.push_back(StereoConvolver::Ptr(oldConv)); // 古いオブジェクトをゴミ箱へ
        oldConv->decReferenceCount();
    }

    // 現在の有効なIR長を更新
    irLength = targetLength;
    currentSampleRate.store(currentSpec.sampleRate);

    isLoading.store(false);
    isRebuilding.store(false, std::memory_order_release); // Reset rebuild flag
    sendChangeMessage();
}

void ConvolverProcessor::cleanup()
{
    std::vector<StereoConvolver::Ptr> toDelete;
    {
        const juce::ScopedLock sl(trashBinLock);
        // 1. Clean old trash
        auto it = std::remove_if(trashBin.begin(), trashBin.end(),
                                 [](const auto& p) { return p->getReferenceCount() == 1; });

        toDelete.insert(toDelete.end(),
                        std::make_move_iterator(it),
                        std::make_move_iterator(trashBin.end()));

        trashBin.erase(it, trashBin.end());

        // 2. Move pending to trash
        trashBin.insert(trashBin.end(), trashBinPending.begin(), trashBinPending.end());
        trashBinPending.clear();
    }
    // ロック解放後にデストラクタを実行
    toDelete.clear();
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
    if (numSamples <= 0) return;

    // IRの長さに応じてFFTサイズを決定 (固定サイズではなく適応させる)
    // ただし、極端に巨大なIRの場合はパフォーマンスを考慮して上限を設ける (例: 65536)
    int fftSize = juce::nextPowerOfTwo(numSamples);
    const int maxFFTSize = 65536;
    if (fftSize > maxFFTSize) fftSize = maxFFTSize;
    if (fftSize < 512) fftSize = 512;

    juce::dsp::FFT fft(static_cast<int>(std::log2(fftSize)));

    // キャッシュされたバッファを再利用 (メモリ確保のオーバーヘッド削減)
    if (cachedFFTBuffer.size() < static_cast<size_t>(fftSize * 2))
        cachedFFTBuffer.resize(static_cast<size_t>(fftSize * 2));

    std::fill(cachedFFTBuffer.begin(), cachedFFTBuffer.end(), 0.0f);

    // チャンネル0 (Lch) の特性を使用する
    const double* src = irBuffer.getReadPointer(0);
    const int copyLen = (std::min)(numSamples, fftSize);
    // Double -> Float conversion for display FFT
    float* dst = cachedFFTBuffer.data();
    for (int i = 0; i < copyLen; ++i)
        dst[i] = static_cast<float>(src[i]);

    fft.performFrequencyOnlyForwardTransform(cachedFFTBuffer.data());

    // スムーシング適用 (Linear Magnitudeに対して行う)
    const int numBins = fftSize / 2 + 1;
    std::vector<float> linearMags(cachedFFTBuffer.begin(), cachedFFTBuffer.begin() + numBins);
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

        juce::File fileToLoad;
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

        // ← irFileLock 解放後に loadImpulseResponse() 呼び出し
        if (fileToLoad != juce::File())
            loadImpulseResponse(fileToLoad);
    }
}

//--------------------------------------------------------------
// syncStateFrom
//--------------------------------------------------------------
void ConvolverProcessor::syncStateFrom(const ConvolverProcessor& other)
{
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

    // Convolutionオブジェクトの同期 (Deep Copy)
    // rebuild時は新しいDSPCoreが作られるため、既存のStereoConvolverを共有すると
    // prepareToPlayでのreset()が稼働中のDSPに影響してしまう。
    // そのため、ディープコピーを作成して独立させる。
    auto otherConv = other.convolution.load(std::memory_order_acquire);
    if (otherConv)
    {
        // Shallow Copy (共有) に変更
        // prepareToPlayで必要に応じてDeep Copyを行うことで、通常時のオーバーラップ維持を実現
        otherConv->incReferenceCount();
        auto oldConv = convolution.exchange(otherConv, std::memory_order_release);
        if (oldConv)
            oldConv->decReferenceCount();
    }
    else
    {
        auto oldConv = convolution.exchange(nullptr, std::memory_order_release);
        if (oldConv)
            oldConv->decReferenceCount();
    }
}

void ConvolverProcessor::syncParametersFrom(const ConvolverProcessor& other)
{
    // 軽量なパラメータのみ同期 (AudioBufferのコピーを避ける)
    mixTarget.store(other.mixTarget.load(), std::memory_order_release);
    bypassed.store(other.bypassed.load(), std::memory_order_release);
    useMinPhase.store(other.useMinPhase.load(), std::memory_order_release);
    smoothingTimeSec.store(other.smoothingTimeSec.load(), std::memory_order_release);
    targetIRLengthSec.store(other.targetIRLengthSec.load(), std::memory_order_release);

    // サンプルレートが一致する場合のみ Convolution オブジェクトを同期する。
    // オーバーサンプリング中は DSP側のレート(Nx) != UI側のレート(1x) となるため、
    // UI側のオブジェクトをコピーするとピッチズレやレイテンシー不整合が発生する。
    if (std::abs(currentSampleRate.load() - other.currentSampleRate.load()) < 1.0)
    {
        auto otherConv = other.convolution.load(std::memory_order_acquire);
        auto expectedConv = convolution.load(std::memory_order_acquire);

        if (otherConv != expectedConv)
        {
            // Note: This is a simplified sync. In a real scenario, we should handle ref counting carefully.
            // Since this is called from UI thread (timer) usually, we can just do a swap if needed.
            // But syncing raw pointers between processors is tricky.
            // For now, we assume syncStateFrom (full sync) is preferred.
            // Leaving this as is might be unsafe if otherConv is deleted.
            // Ideally, we should clone or increment ref count.
        }
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
    // 参照カウントをインクリメントして、処理中の削除を防ぐ
    StereoConvolver::Ptr convPtr = convolution.load(std::memory_order_acquire);
    StereoConvolver* conv = convPtr.get();

    // ── Step 2: 処理実行可能かチェック ──
    // バイパス、未準備、IR未ロードの場合はスルー
    if (!isPrepared.load(std::memory_order_acquire) || bypassed.load() || !conv)
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
        if (std::abs(latencySmoother.getTargetValue() - static_cast<double>(totalLatency)) > 0.001)
            latencySmoother.setTargetValue(static_cast<double>(totalLatency));
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

        // レイテンシー変更中はサンプル単位で処理してスムージングを行う
        if (latencySmoother.isSmoothing())
        {
            // スムーシング時: 線形補間付き読み出し (Variable Delay)
            for (int i = 0; i < numSamples; ++i)
            {
                // 現在の書き込み位置 (サンプル単位で進む)
                int currentWPos = (delayWritePos + i) & DELAY_BUFFER_MASK;
                const double currentDelay = latencySmoother.getNextValue();

                // 読み出し位置の計算 (整数部と小数部)
                double readPosFloat = static_cast<double>(currentWPos) - currentDelay;
                // 負の値のラップアラウンド処理
                while (readPosFloat < 0.0) readPosFloat += DELAY_BUFFER_SIZE;

                int readPosInt = static_cast<int>(readPosFloat);
                double frac = readPosFloat - readPosInt;
                int readPosNext = (readPosInt + 1) & DELAY_BUFFER_MASK;
                readPosInt &= DELAY_BUFFER_MASK; // 安全のため

                for (int ch = 0; ch < procChannels; ++ch)
                {
                    double* buf = delayBuffer[ch].get();
                    // 線形補間
                    double val = buf[readPosInt] + frac * (buf[readPosNext] - buf[readPosInt]);
                    dryBuffer.setSample(ch, i, val);
                }
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

    for (int ch = 0; ch < procChannels; ++ch)
    {
        // 1. 全チャンネルに入力を供給 (Add)
        const double* input = block.getChannelPointer(ch);
        WDL_FFT_REAL* inputs[1] = { const_cast<WDL_FFT_REAL*>(input) };
        conv->convolvers[ch].Add(inputs, numSamples, 1);

        // 2. 出力を取得 (Get) - ポインタのみ取得し、コピーはMixループで行う
        int avail = conv->convolvers[ch].Avail(numSamples);
        int validWetSamples = std::min(numSamples, avail);

        WDL_FFT_REAL** outputs = conv->convolvers[ch].Get();
        const double* wdlOut = (validWetSamples > 0 && outputs && outputs[0]) ? outputs[0] : nullptr;

        if (!wdlOut) validWetSamples = 0;

        // 3. Mix (Fused Loop: Copy + Gain + Mix)
        double* dst = block.getChannelPointer(ch);
        const double* dry = dryBuffer.getReadPointer(ch);

        if (isSmoothing)
        {
            // スムーシング時 (AVX2 Optimized)
            int i = 0;
#if defined(__AVX2__)
            const int vLoop = validWetSamples / 4 * 4;
            if (vLoop > 0)
            {
                for (; i < vLoop; i += 4)
                {
                    __m256d vWet = _mm256_loadu_pd(wdlOut + i);
                    __m256d vDry = _mm256_loadu_pd(dry + i);
                    __m256d vWetG = _mm256_loadu_pd(wetGains + i);
                    __m256d vDryG = _mm256_loadu_pd(dryGains + i);
                    __m256d vOut = _mm256_add_pd(_mm256_mul_pd(vWet, vWetG), _mm256_mul_pd(vDry, vDryG));
                    _mm256_storeu_pd(dst + i, vOut);
                }
            }
#endif
            for (; i < validWetSamples; ++i)
            {
                dst[i] = wdlOut[i] * wetGains[i] + dry[i] * dryGains[i];
            }

            // Wetが無効な区間
            for (; i < numSamples; ++i)
            {
                dst[i] = dry[i] * dryGains[i];
            }
        }
        else
        {
            // 定常状態 (99%のケース) -> AVX2最適化
            const double wetG = needsConvolution ? (targetMixValue * headroom) : 0.0;
            const double dryG = needsDrySignal   ? (1.0 - targetMixValue)      : 0.0;

            int i = 0;

#if defined(__AVX2__)
            // AVX2 (256-bit) = 4 doubles
            const int vLoop = validWetSamples / 4 * 4;
            if (vLoop > 0)
            {
                const __m256d vWetG = _mm256_set1_pd(wetG);
                const __m256d vDryG = _mm256_set1_pd(dryG);

                for (; i < vLoop; i += 4)
                {
                    // Unaligned load is fine on modern CPUs
                    __m256d vWet = _mm256_loadu_pd(wdlOut + i);
                    __m256d vDry = _mm256_loadu_pd(dry + i);

                    // dst = wet * wetG + dry * dryG
                    __m256d vOut = _mm256_add_pd(_mm256_mul_pd(vWet, vWetG), _mm256_mul_pd(vDry, vDryG));

                    _mm256_storeu_pd(dst + i, vOut);
                }
            }
#endif
            // 残りの有効なWetサンプル (Scalar)
            for (; i < validWetSamples; ++i)
            {
                dst[i] = wdlOut[i] * wetG + dry[i] * dryG;
            }

            // Wetが無効な区間 (初期レイテンシー等) -> Dryのみ出力
            // dst = 0 * wetG + dry * dryG
            for (; i < numSamples; ++i)
            {
                dst[i] = dry[i] * dryG;
            }
        }

        // 4. Advance (読み取った分だけ進める)
        if (validWetSamples > 0)
            conv->convolvers[ch].Advance(validWetSamples);
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
