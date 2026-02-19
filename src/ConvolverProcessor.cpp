//============================================================================
// ConvolverProcessor.cpp  ── v0.1 (JUCE 8.0.12対応)
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

#include <juce_dsp/juce_dsp.h>

#include "AudioFFT.h" // Double precision FFT
#include "CDSPResampler.h"

// Forward declaration
static juce::AudioBuffer<double> convertToMinimumPhase(const juce::AudioBuffer<double>& linearIR, juce::Thread* thread = nullptr, bool* wasCancelled = nullptr);

// Helper function to check for thread cancellation
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

// Helper for resampling
static juce::AudioBuffer<double> resampleIR(const juce::AudioBuffer<double>& inputIR, double inputSR, double targetSR, juce::Thread* thread)
{
    if (inputSR <= 0.0 || targetSR <= 0.0 || std::abs(inputSR - targetSR) <= 1.0)
        return inputIR;

    const double ratio = targetSR / inputSR;
    const int inLength = inputIR.getNumSamples();
    const int maxOutLen = static_cast<int>(inLength * ratio + 2.0);

    juce::AudioBuffer<double> resampled(inputIR.getNumChannels(), maxOutLen);
    resampled.clear();

    constexpr double transBand = 2.0;
    constexpr double stopBandAtten = 140.0;
    constexpr r8b::EDSPFilterPhaseResponse phase = r8b::fprLinearPhase;

    int finalLength = 0;
    for (int ch = 0; ch < inputIR.getNumChannels(); ++ch)
    {
        if (checkCancellation(thread, nullptr)) return {};

        r8b::CDSPResampler resampler(inputSR, targetSR, inLength, transBand, stopBandAtten, phase);

        const double* inPtr = inputIR.getReadPointer(ch);
        double* outPtr = resampled.getWritePointer(ch);

        int done = 0;
        while (done < maxOutLen)
        {
            if (checkCancellation(thread, nullptr)) return {};
            double* r8bOutput = nullptr;
            const int generated = resampler.process(const_cast<double*>(inPtr),
                                                    done == 0 ? inLength : 0, r8bOutput);

            if (generated == 0) break;

            const int toCopy = std::min(generated, maxOutLen - done);
            std::memcpy(outPtr + done, r8bOutput, toCopy * sizeof(double));
            done += toCopy;
        }
        finalLength = done;
    }
    resampled.setSize(inputIR.getNumChannels(), finalLength, true, true, true);
    return resampled;
}

//--------------------------------------------------------------
// LoaderThread クラス定義
// IRの読み込み、処理、State作成をバックグラウンドで行う
//--------------------------------------------------------------
class ConvolverProcessor::LoaderThread : public juce::Thread
{
public:
    // ファイルからロードする場合のコンストラクタ
    LoaderThread(ConvolverProcessor& p, const juce::File& f, double sr, int bs, bool minPhase)
        : Thread("IRLoader"), owner(p), file(f), sampleRate(sr), blockSize(bs), useMinPhase(minPhase), isRebuild(false)
    {}

    // メモリからリビルドする場合のコンストラクタ
    LoaderThread(ConvolverProcessor& p, const juce::AudioBuffer<double>& src, double srcSR, double sr, int bs, bool minPhase)
        : Thread("IRRebuilder"), owner(p), sourceIR(src), sourceSampleRate(srcSR), sampleRate(sr), blockSize(bs), useMinPhase(minPhase), isRebuild(true)
    {}

    ~LoaderThread() override
    {
        stopThread(4000);
    }

    void run() override
    {
        juce::ScopedNoDenormals noDenormals; // バックグラウンド処理でのDenormal対策

        // BUG #16: Wrap in try-catch to prevent std::terminate() on std::bad_alloc
        // BUG FIX: Ensure isLoading/isRebuilding are reset on early exit (unless thread is stopping)
        struct FlagResetter {
            ConvolverProcessor& p;
            const juce::Thread& t;
            bool success = false;
            ~FlagResetter() {
                if (!success && !t.threadShouldExit()) {
                    juce::WeakReference<ConvolverProcessor> weakOwner(&p);
                    juce::MessageManager::callAsync([weakOwner] {
                        if (auto* o = weakOwner.get()) {
                            o->isLoading.store(false);
                            o->isRebuilding.store(false);
                        }
                    });
                }
            }
        } resetter { owner, *this };

        try
        {

        owner.setLoadingProgress(0.0f);
        // IRデータ
        juce::AudioBuffer<double> loadedIR; // IRデータ
        double loadedSR = 0.0;             // サンプルレート

        // 1. IRデータの取得 (ファイル読み込み or メモリコピー)
        if (isRebuild)
        {
            loadedIR = sourceIR;
            loadedSR = sourceSampleRate;
        }
        else
        {
            if (!file.existsAsFile()) return;

            juce::AudioFormatManager formatManager;
            formatManager.registerBasicFormats();
            std::unique_ptr<juce::AudioFormatReader> reader(formatManager.createReaderFor(file));

            if (!reader) return;

            // サイズの妥当性チェック (lengthInSamples が int の範囲を超える場合への対策)
            const int64 fileLength = reader->lengthInSamples;
            const int numChannels = static_cast<int>(reader->numChannels);
            static constexpr int64 MAX_FILE_LENGTH = 2147483647;  // int の最大値

            if (fileLength > MAX_FILE_LENGTH) {
                DBG("LoaderThread: ファイルサイズが大きすぎます。");
                return;
            }

            // AudioFormatReader::read は float のみ対応のため、一時バッファを使用
            juce::AudioBuffer<float> tempFloatBuffer(numChannels, static_cast<int>(fileLength));
            reader->read(&tempFloatBuffer, 0, static_cast<int>(fileLength), 0, true, true);

            loadedIR.setSize(numChannels, static_cast<int>(fileLength));
            for (int ch = 0; ch < numChannels; ++ch)
            {
                const float* src = tempFloatBuffer.getReadPointer(ch);
                double* dst = loadedIR.getWritePointer(ch);
                for (int i = 0; i < static_cast<int>(fileLength); ++i)
                    dst[i] = static_cast<double>(src[i]);
            }
            loadedSR = reader->sampleRate;
        }

        if (threadShouldExit() || loadedIR.getNumSamples() == 0) return;

        // 1.5. リサンプリング (SR不一致の場合)
        // IRのサンプルレートがターゲットと異なる場合、ピッチズレを防ぐためにリサンプリングする
        if (loadedSR > 0.0 && sampleRate > 0.0 && std::abs(loadedSR - sampleRate) > 1.0)
        {
            auto resampled = resampleIR(loadedIR, loadedSR, sampleRate, this);
            if (resampled.getNumSamples() == 0) return; // Cancelled

            loadedIR = std::move(resampled);
            loadedSR = sampleRate;
        }

        // 1.6. 末尾の無音カット (Denormal対策 & 効率化)
        // IR末尾の極小値(Denormal領域)をカットすることで、畳み込み負荷とDenormal発生リスクを低減
        if (loadedIR.getNumSamples() > 0)
        {
            int newLength = loadedIR.getNumSamples();
            const int channels = loadedIR.getNumChannels();
            const double threshold = 1.0e-15; // -300dB (double精度における実質的な無音)

            while (newLength > 0)
            {
                bool isSilent = true;
                for (int ch = 0; ch < channels; ++ch)
                {
                    if (std::abs(loadedIR.getSample(ch, newLength - 1)) > threshold)
                    {
                        isSilent = false;
                        break;
                    }
                }
                if (!isSilent) break;
                newLength--;
            }

            if (newLength < loadedIR.getNumSamples())
                loadedIR.setSize(channels, std::max(1, newLength), true);
        }

        // 2. ピーク正規化 (ファイル読み込み時のみ)
        // リビルド時は既に正規化されていると仮定するためスキップします。
        if (!isRebuild)
        {
            double maxMagnitude = 0.0;
            for (int ch = 0; ch < loadedIR.getNumChannels(); ++ch)
                maxMagnitude = (std::max)(maxMagnitude, loadedIR.getMagnitude(ch, 0, loadedIR.getNumSamples()));

            if (maxMagnitude > 0.0)
                loadedIR.applyGain(1.0 / maxMagnitude);
        }

        if (threadShouldExit()) return;

        // 3. ターゲット長計算とトリミング
        int targetLength = owner.computeTargetIRLength(sampleRate, loadedIR.getNumSamples());
        juce::AudioBuffer<double> trimmed(loadedIR.getNumChannels(), targetLength);
        trimmed.clear();

        int copySamples = (std::min)(targetLength, loadedIR.getNumSamples());
        for (int ch = 0; ch < loadedIR.getNumChannels(); ++ch)
        {
            trimmed.copyFrom(ch, 0, loadedIR, ch, 0, copySamples);
            // フェードアウト
            int fade = 256;
            if (copySamples > fade)
                trimmed.applyGainRamp(ch, copySamples - fade, fade, 1.0, 0.0);
        }

        if (threadShouldExit()) return;

        // 4. MinPhase変換 (オプション)
        if (useMinPhase)
        {
            bool wasCancelled = false;
            auto minPhaseIR = convertToMinimumPhase(trimmed, this, &wasCancelled);

            if (wasCancelled) return;

            // 変換成功チェック: キャンセルされておらず、かつ無音でない場合のみ適用
            if (minPhaseIR.getNumSamples() > 0 && minPhaseIR.getMagnitude(0, 0, minPhaseIR.getNumSamples()) > 1.0e-5)
            {
                trimmed = minPhaseIR;
            }
            // 変換に失敗または無音になった場合は、元のtrimmed(Linear Phase)を使用する
        }

        if (threadShouldExit()) return;

        // ピーク位置検出 (レイテンシー補正用)
        // Linear Phaseの場合、ピークが遅れてやってくるため、その分Dryを遅らせる必要がある
        int irPeakLatency = 0;
        if (!useMinPhase)
        {
            // 全チャンネルの中で最大振幅を持つサンプルの位置を探す
            double maxMag = -1.0;
            for (int ch = 0; ch < trimmed.getNumChannels(); ++ch)
            {
                const double* data = trimmed.getReadPointer(ch);
                for (int i = 0; i < targetLength; ++i)
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

        // 5. 新しいConvolutionの構築 (Non-Uniform Partitioning)
        auto newConv = std::make_shared<StereoConvolver>();

        // FFTConvolver用にIRデータを準備
        // FFTConvolver::init は Sample* (double*) を要求するため、trimmedバッファ (double) からコピーする。
        std::vector<fftconvolver::Sample> irL(targetLength);
        std::vector<fftconvolver::Sample> irR(targetLength);

        const double* srcL = trimmed.getReadPointer(0);
        const double* srcR = (trimmed.getNumChannels() > 1) ? trimmed.getReadPointer(1) : srcL;

        // 型安全なコピー（Sample型が何であれ動作する）
        for (int i = 0; i < targetLength; ++i)
        {
            irL[i] = static_cast<fftconvolver::Sample>(srcL[i]);
            irR[i] = static_cast<fftconvolver::Sample>(srcR[i]);
        }

        // 初期化 (blockSizeをセグメントサイズとして使用)
        newConv->init(blockSize, irL.data(), irR.data(), targetLength, irPeakLatency);

        // Display用コピーを作成 (move前に)
        juce::AudioBuffer<double> displayIR = trimmed;

        if (threadShouldExit()) return;

        // 6. メインスレッドで適用
        // WeakReferenceを使って、Processorが削除されていたら実行しないようにする
        juce::WeakReference<ConvolverProcessor> weakOwner(&owner);

        // shared_ptrで管理 (Lambdaコピー時のAudioBufferディープコピー回避 & メモリ寿命管理)
        auto loadedIRPtr = std::make_shared<juce::AudioBuffer<double>>(std::move(loadedIR));
        auto displayIRPtr = std::make_shared<juce::AudioBuffer<double>>(std::move(displayIR));

        juce::MessageManager::callAsync([weakOwner, newConv, loadedIRPtr, loadedSR, targetLength, isRebuild = this->isRebuild, file = this->file, displayIRPtr]()
        {
            if (weakOwner)
            {
                weakOwner->applyNewState(newConv, *loadedIRPtr, loadedSR, targetLength, isRebuild, file, *displayIRPtr);
            }
        });

        resetter.success = true;

        }
        catch (const std::bad_alloc&)
        {
            DBG("LoaderThread: Memory allocation failed. Aborting IR load.");
        }
    }

private:
    ConvolverProcessor& owner;
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
    spec.maximumBlockSize = static_cast<juce::uint32>(samplesPerBlock);
    spec.numChannels = 2;  // ステレオ

    currentSpec = spec;

    // 既存のコンボリューション状態をリセット (オーバーラップバッファのクリア)
    auto conv = convolution.load();
    if (conv) {
        conv->reset();
    }

    // DelayLine準備
    // 最大レイテンシーを多めに確保 (e.g., 2秒)
    // JUCEの仕様上、prepare()の前にsetMaximumDelayInSamples()を呼ぶ必要がある。
    // これにより、prepare()が十分なメモリを事前に確保する。
    delayLine.setMaximumDelayInSamples(MAX_TOTAL_DELAY);
    delayLine.prepare(spec);
    delayLine.setDelay(0.0f);

    // Dryバッファ確保
    dryBuffer.setSize(2, samplesPerBlock);
    dryBuffer.clear();

    // スムージング時間の設定
    currentSmoothingTimeSec = smoothingTimeSec.load();
    mixSmoother.reset(sampleRate, currentSmoothingTimeSec);

    convolutionBuffer.setSize(2, samplesPerBlock);
    convolutionBuffer.clear();

    isPrepared.store(true, std::memory_order_release);
}

void ConvolverProcessor::reset()
{
    auto conv = convolution.load();
    if (conv)
    {
        conv->convolvers[0].resetInput();
        conv->convolvers[1].resetInput();
    }
    delayLine.reset();
    dryBuffer.clear();
    convolutionBuffer.clear();
    mixSmoother.setCurrentAndTargetValue(static_cast<double>(mixTarget.load()));
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

//--------------------------------------------------------------
// StereoConvolver Copy Constructor
//--------------------------------------------------------------
ConvolverProcessor::StereoConvolver::StereoConvolver(const StereoConvolver& other)
{
    // キャッシュされたIRデータを使って初期化 (Deep Copy)
    init(other.blockSize, other.irL.data(), other.irR.data(), other.irL.size(), other.irLatency);
}

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

    // BUG #16: Add fftSize cap to prevent excessive memory allocation
    static constexpr int MAX_MINPHASE_FFT_SIZE = 2097152; // 2^21
    if (fftSize > MAX_MINPHASE_FFT_SIZE)
    {
        DBG("convertToMinimumPhase: fftSize (" << fftSize << ") exceeds limit. Skipping min-phase conversion to prevent excessive memory usage.");
        return {}; // Return empty buffer to signal failure/skip
    }

    // Double精度FFTを使用 (audiofft::AudioFFT)
    // FFTCONVOLVER_USE_DOUBLE が定義されているため、Sample は double
    audiofft::AudioFFT fft;
    fft.init(fftSize);

    juce::AudioBuffer<double> minPhaseIR(linearIR.getNumChannels(), numSamples);

    // Split-Complex バッファ (audiofft用)
    std::vector<double> re(fftSize);
    std::vector<double> im(fftSize);
    std::vector<double> data(fftSize); // 作業用

    // audiofft の IFFT は 2/N スケーリングが必要 (Oouraの場合)
    // ただし AudioFFT::ifft 内部でスケーリングされる実装になっているか確認が必要
    // AudioFFT.cpp を見ると、OouraFFT::ifft で 2.0/size スケーリングされている。

    for (int ch = 0; ch < linearIR.getNumChannels(); ++ch)
    {
        if (checkCancellation(thread, wasCancelled)) return {};

        // 1. IRをコピー (Realパート)
        const double* src = linearIR.getReadPointer(ch);
        std::fill(data.begin(), data.end(), 0.0);
        std::memcpy(data.data(), src, numSamples * sizeof(double));

        // 2. FFT (Time -> Freq)
        fft.fft(data.data(), re.data(), im.data());

        if (checkCancellation(thread, wasCancelled)) return {};

        // 3. 対数マグニチュードスペクトル計算 (Real=Log|H|, Imag=0) [Double]
        for (int i = 0; i < fftSize; ++i)
        {
            double mag = std::sqrt(re[i] * re[i] + im[i] * im[i]);
            if (!std::isfinite(mag)) mag = 0.0;
            // ゼロ除算防止 (doubleの極小値を使用)
            double logMag = std::log((std::max)(mag, 1.0e-100));
            re[i] = logMag;
            im[i] = 0.0;
        }

        // 4. IFFT (Freq -> Time) => 実ケプストラム (Real Cepstrum)
        fft.ifft(data.data(), re.data(), im.data());

        if (checkCancellation(thread, wasCancelled)) return {};

        // 5. 因果的ウィンドウ適用 (リフタリング) [Double]
        // c[0] = c[0]
        // c[n] = 2*c[n] (0 < n < N/2)
        // c[N/2] = c[N/2]
        // c[n] = 0 (N/2 < n < N)
        // data[] は実数配列
        for (int i = 1; i < fftSize / 2; ++i)
            data[i] *= 2.0;

        for (int i = fftSize / 2 + 1; i < fftSize; ++i)
            data[i] = 0.0;

        // 6. FFT (Time -> Freq) => 解析信号の対数スペクトル (実部が対数振幅、虚部が最小位相)
        fft.fft(data.data(), re.data(), im.data());

        if (checkCancellation(thread, wasCancelled)) return {};

        // 7. 複素指数変換 (exp) => 最小位相スペクトル [Double]
        for (int i = 0; i < fftSize; ++i)
        {
            double real = re[i];
            double imag = im[i];
            // 数値オーバーフロー防止のためのクランプ
            real = juce::jlimit(-50.0, 50.0, real);
            imag = juce::jlimit(-50.0, 50.0, imag);

            std::complex<double> c(real, imag);
            std::complex<double> ex = std::exp(c);
            re[i] = ex.real();
            im[i] = ex.imag();
        }

        // 8. IFFT (Freq -> Time) => 時間領域の最小位相IR
        fft.ifft(data.data(), re.data(), im.data());

        if (checkCancellation(thread, wasCancelled)) return {};

        // 9. 結果をコピー
        double* dst = minPhaseIR.getWritePointer(ch);
        for (int i = 0; i < numSamples; ++i)
            dst[i] = data[i];
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
        activeLoader = std::make_unique<LoaderThread>(*this, originalIR, originalIRSampleRate, currentSpec.sampleRate, currentSpec.maximumBlockSize, useMinPhase.load());
    }
    else
    {
        activeLoader = std::make_unique<LoaderThread>(*this, irFile, currentSpec.sampleRate, currentSpec.maximumBlockSize, useMinPhase.load());
        currentIrOptimized.store(optimizeForRealTime);
    }

    activeLoader->startThread();

    return true;
}

//--------------------------------------------------------------
// applyNewState (Message Thread Callback)
// ローダースレッド完了後に呼ばれる
//--------------------------------------------------------------
void ConvolverProcessor::applyNewState(std::shared_ptr<StereoConvolver> newConv,
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
    auto oldConv = convolution.exchange(newConv, std::memory_order_acq_rel);

    if (oldConv)
    {
        const juce::ScopedLock sl(trashBinLock);
        trashBin.push_back(std::move(oldConv));

        // ゴミ箱のサイズ制限 (メモリ肥大化防止)
        // Audio Threadが参照していないオブジェクトのみを削除し、
        // Audio Threadでのメモリ解放(free)を防ぐ。
        trashBin.erase(std::remove_if(trashBin.begin(), trashBin.end(),
                                      [](const auto& p) { return p.use_count() == 1; }), trashBin.end());
    }

    // 現在の有効なIR長を更新
    irLength = targetLength;
    currentSampleRate.store(currentSpec.sampleRate);

    isLoading.store(false);
    isRebuilding.store(false, std::memory_order_release); // Reset rebuild flag
    sendChangeMessage();
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

    // IRパスの自動復元はここでは行いません。
    // AudioEngine::rebuild時にsetStateが呼ばれた際の無限ループや二重読み込みを防ぐため、
    // IRの読み込みはユーザーアクションまたは明示的なプリセットロード処理でのみ行います。
    /*
    if (v.hasProperty ("irPath"))
    {
        juce::File f (v.getProperty ("irPath").toString());
        if (f.existsAsFile() && f != currentIrFile)
            loadImpulseResponse (f);
    }
    */
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

    // BUG #15: サンプルレート変更時にリビルドできるよう、元のIR情報をコピーする
    // This ensures the new DSP core has the source material to resample the IR.
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
        auto newConv = std::make_shared<StereoConvolver>(*otherConv);
        convolution.store(newConv, std::memory_order_release);
    }
    else
    {
        convolution.store(nullptr, std::memory_order_release);
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

    // Convolutionオブジェクトのポインタ同期 (Atomic shared_ptr copy is cheap)
    auto otherConv = other.convolution.load(std::memory_order_acquire);
    auto expectedConv = convolution.load(std::memory_order_acquire);

    if (otherConv != expectedConv)
    {
        convolution.compare_exchange_strong(expectedConv, otherConv, std::memory_order_acq_rel, std::memory_order_acquire);
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
    auto conv = convolution.load(std::memory_order_acquire);

    if (conv)
    {
        // 処理遅延(ブロックサイズ) + IR遅延(ピーク位置)
        const int totalLatency = (std::min)(conv->latency + conv->irLatency, MAX_TOTAL_DELAY);
        delayLine.setDelay(static_cast<float>(totalLatency));
        currentLatency.store(totalLatency);
    }

    // ── Step 2: 処理実行可能かチェック ──
    // バイパス、未準備、IR未ロードの場合はスルー
    if (!isPrepared.load(std::memory_order_acquire) || bypassed.load() || !conv)
    {
        return;
    }

    // processBufferのチャンネル数を使用 (最大2ch)
    const int procChannels = (std::min)((int)block.getNumChannels(), 2);
    const int numSamples = (int)block.getNumSamples();

    // ── Step 3: バッファサイズ安全対策 (Bounds Check) ──
    if (numSamples <= 0 || procChannels == 0 || numSamples > dryBuffer.getNumSamples() || numSamples > convolutionBuffer.getNumSamples())
        return;

    // ── Step 4: パラメータ更新と最適化 ──
    // Audio Threadでのみ setTargetValue() を呼ぶことでスレッドセーフティを確保
    const double targetMixValue = static_cast<double>(mixTarget.load(std::memory_order_relaxed));
    if (std::abs(mixSmoother.getTargetValue() - targetMixValue) > 0.001)
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
        double currentVal = mixSmoother.getCurrentValue();
        double targetVal = mixSmoother.getTargetValue();
        mixSmoother.reset(currentSpec.sampleRate, newSmoothingTime);
        mixSmoother.setCurrentAndTargetValue(currentVal);
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
        // Dry信号をdryBufferにコピーし、レイテンシー分遅延 (inputBlockは 'block' を直接使用)
        juce::dsp::AudioBlock<const double> inputBlock(block);
        juce::dsp::AudioBlock<double> outputBlock(dryBuffer.getArrayOfWritePointers(), procChannels, numSamples);
        juce::dsp::ProcessContextNonReplacing<double> delayContext(inputBlock, outputBlock);
        delayLine.process(delayContext);
    }

    // ── Step 6: Wet信号生成 ──
    // 常にコンボリューションを実行し、FFTConvolverの内部状態(オーバーラップバッファ)を維持する。
    // これにより、Mixを0%から上げた際のグリッチを防ぐ。
    // FFTConvolverを使用 (double精度)
    for (int ch = 0; ch < procChannels; ++ch)
    {
        const double* input = block.getChannelPointer(ch);
        double* dst = convolutionBuffer.getWritePointer(ch);
        conv->convolvers[ch].process(input, dst, numSamples);
    }

    // Wet信号に-6dBのヘッドルームを確保 (より保守的なクリッピング防止)
    // BUG #20: double精度で、かつ必要な範囲にのみゲインを適用
    if (needsConvolution)
        convolutionBuffer.applyGain(0, numSamples, CONVOLUTION_HEADROOM_GAIN);

    // ── Step 7: Dry/Wet Mix ──
    if (!needsConvolution) // 100% Dry
    {
        for (int ch = 0; ch < procChannels; ++ch)
        {
            const double* src = dryBuffer.getReadPointer(ch);
            double* dst = block.getChannelPointer(ch);
            std::memcpy(dst, src, numSamples * sizeof(double));
        }
    }
    else if (!needsDrySignal) // 100% Wet
    {
        for (int ch = 0; ch < procChannels; ++ch)
        {
            const double* wetSrc = convolutionBuffer.getReadPointer(ch);
            double* dst = block.getChannelPointer(ch);
            for (int i = 0; i < numSamples; ++i)
                dst[i] = wetSrc[i];
        }
    }
    else
    {
        // 0% < Mix < 100% または スムージング中
        if (mixSmoother.isSmoothing())
        {
            const double* wetPtrs[2] = { nullptr };
            const double* dryPtrs[2] = { nullptr };
            double* dstPtrs[2] = { nullptr };

            for (int ch = 0; ch < procChannels; ++ch)
            {
                wetPtrs[ch] = convolutionBuffer.getReadPointer(ch);
                dryPtrs[ch] = dryBuffer.getReadPointer(ch);
                dstPtrs[ch] = block.getChannelPointer(ch);
            }

            for (int i = 0; i < numSamples; ++i)
            {
                const double mixValue = mixSmoother.getNextValue();
                const double wetGain = std::sin(mixValue * juce::MathConstants<double>::halfPi);
                const double dryGain = std::cos(mixValue * juce::MathConstants<double>::halfPi);

                for (int ch = 0; ch < procChannels; ++ch)
                {
                    dstPtrs[ch][i] = wetPtrs[ch][i] * wetGain + dryPtrs[ch][i] * dryGain;
                }
            }
        }
        else
        {
            const double mixValue = targetMixValue;
            const double wetGain = std::sin(mixValue * juce::MathConstants<double>::halfPi);
            const double dryGain = std::cos(mixValue * juce::MathConstants<double>::halfPi);

            for (int ch = 0; ch < procChannels; ++ch)
            {
                const double* wet = convolutionBuffer.getReadPointer(ch);
                const double* dry = dryBuffer.getReadPointer(ch);
                double* dst = block.getChannelPointer(ch);

                for (int i = 0; i < numSamples; ++i)
                    dst[i] = wet[i] * wetGain + dry[i] * dryGain;
            }
        }
    }
}

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
