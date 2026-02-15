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

// Forward declaration
static juce::AudioBuffer<float> convertToMinimumPhase(const juce::AudioBuffer<float>& linearIR, juce::Thread* thread = nullptr, bool* wasCancelled = nullptr);

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
    LoaderThread(ConvolverProcessor& p, const juce::AudioBuffer<float>& src, double srcSR, double sr, int bs, bool minPhase)
        : Thread("IRRebuilder"), owner(p), sourceIR(src), sourceSampleRate(srcSR), sampleRate(sr), blockSize(bs), useMinPhase(minPhase), isRebuild(true)
    {}

    ~LoaderThread() override
    {
        stopThread(4000);
    }

    void run() override
    {
        juce::AudioBuffer<float> loadedIR;
        double loadedSR = 0.0;

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

            loadedIR.setSize(static_cast<int>(reader->numChannels), static_cast<int>(reader->lengthInSamples));
            reader->read(&loadedIR, 0, static_cast<int>(reader->lengthInSamples), 0, true, true);
            loadedSR = reader->sampleRate;
        }

        if (threadShouldExit() || loadedIR.getNumSamples() == 0) return;

        // 1.5. リサンプリング (SR不一致の場合)
        // IRのサンプルレートがターゲットと異なる場合、ピッチズレを防ぐためにリサンプリングする
        if (loadedSR > 0.0 && sampleRate > 0.0 && std::abs(loadedSR - sampleRate) > 1.0)
        {
            const double ratio = loadedSR / sampleRate;
            const int newLength = static_cast<int>(std::ceil(loadedIR.getNumSamples() * (sampleRate / loadedSR)));
            juce::AudioBuffer<float> resampled(loadedIR.getNumChannels(), newLength);

            juce::LagrangeInterpolator interpolator;
            for (int ch = 0; ch < loadedIR.getNumChannels(); ++ch)
            {
                resampled.clear(ch, 0, newLength); // processは加算するためクリア必須
                interpolator.reset();
                interpolator.process(ratio, loadedIR.getReadPointer(ch), resampled.getWritePointer(ch), newLength);
            }

            loadedIR = std::move(resampled); // Move semantics to avoid copy
            loadedSR = sampleRate;
        }

        // 2. ピーク正規化 (ファイル読み込み時のみ)
        // リビルド時は既に正規化されていると仮定するためスキップします。
        if (!isRebuild)
        {
            float maxMagnitude = 0.0f;
            for (int ch = 0; ch < loadedIR.getNumChannels(); ++ch)
                maxMagnitude = std::max(maxMagnitude, loadedIR.getMagnitude(ch, 0, loadedIR.getNumSamples()));

            if (maxMagnitude > 0.0f)
                loadedIR.applyGain(1.0f / maxMagnitude);
        }

        if (threadShouldExit()) return;

        // 3. ターゲット長計算とトリミング
        // 表示用に常に1000ms(1.0s)分のバッファを確保するため、loadedSRではなく現在のsampleRateを使用する
        int targetLength = ConvolverProcessor::computeTargetIRLength(sampleRate, loadedIR.getNumSamples());
        juce::AudioBuffer<float> trimmed(loadedIR.getNumChannels(), targetLength);
        trimmed.clear();

        int copySamples = std::min(targetLength, loadedIR.getNumSamples());
        for (int ch = 0; ch < loadedIR.getNumChannels(); ++ch)
        {
            trimmed.copyFrom(ch, 0, loadedIR, ch, 0, copySamples);
            // フェードアウト
            int fade = 256;
            if (copySamples > fade)
                trimmed.applyGainRamp(ch, copySamples - fade, fade, 1.0f, 0.0f);
        }

        if (threadShouldExit()) return;

        // 4. MinPhase変換 (オプション)
        if (useMinPhase)
        {
            bool wasCancelled = false;
            auto minPhaseIR = convertToMinimumPhase(trimmed, this, &wasCancelled);

            if (wasCancelled) return;

            // 変換成功チェック: キャンセルされておらず、かつ無音でない場合のみ適用
            if (minPhaseIR.getNumSamples() > 0 && minPhaseIR.getMagnitude(0, 0, minPhaseIR.getNumSamples()) > 1.0e-5f)
            {
                trimmed = minPhaseIR;
            }
            // 変換に失敗または無音になった場合は、元のtrimmed(Linear Phase)を使用する
        }

        if (threadShouldExit()) return;

        // 5. 新しいConvolutionの構築 (Non-Uniform Partitioning)
        double irSeconds = (double)targetLength / sampleRate;
        auto newConv = owner.createConfiguredConvolution(sampleRate, blockSize, irSeconds);

        auto stereoMode = (trimmed.getNumChannels() >= 2) ? juce::dsp::Convolution::Stereo::yes : juce::dsp::Convolution::Stereo::no;

        // Display用コピーを作成 (move前に)
        juce::AudioBuffer<float> displayIR = trimmed;

        // AudioBufferを直接渡すオーバーロードを使用
        // これにより、生ポインタの管理やフォーマットの誤解釈を防ぐ
        newConv->loadImpulseResponse(std::move(trimmed),
                                     sampleRate,
                                     stereoMode,
                                     juce::dsp::Convolution::Trim::no,
                                     juce::dsp::Convolution::Normalise::no);

        if (threadShouldExit()) return;

        // 6. メインスレッドで適用
        // WeakReferenceを使って、Processorが削除されていたら実行しないようにする
        juce::WeakReference<ConvolverProcessor> weakOwner(&owner);

        // ✅ shared_ptrで管理 (Lambdaコピー時のAudioBufferディープコピー回避 & メモリ寿命管理)
        auto loadedIRPtr = std::make_shared<juce::AudioBuffer<float>>(std::move(loadedIR));
        auto displayIRPtr = std::make_shared<juce::AudioBuffer<float>>(std::move(displayIR));

        juce::MessageManager::callAsync([weakOwner, newConv, loadedIRPtr, loadedSR, targetLength, isRebuild = this->isRebuild, file = this->file, displayIRPtr]()
        {
            if (weakOwner)
            {
                weakOwner->applyNewState(newConv, *loadedIRPtr, loadedSR, targetLength, isRebuild, file, *displayIRPtr);
            }
        });
    }

private:
    ConvolverProcessor& owner;
    juce::File file;
    juce::AudioBuffer<float> sourceIR;
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

    // ✅ 最初にサンプルレートを更新（oldValueを保存）
    double oldSampleRate = currentSampleRate.exchange(sampleRate, std::memory_order_acq_rel);

    // ProcessSpec設定
    juce::dsp::ProcessSpec spec;
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = static_cast<juce::uint32>(samplesPerBlock);
    spec.numChannels = 2;  // ステレオ

    currentSpec = spec;

    // サンプルレート変更検知とState再構築ロジック
    if (isIRLoaded())
    {
        if (std::abs(oldSampleRate - sampleRate) > 1.0)
        {
            // サンプルレート変更時: 現在のStateは維持しつつ、裏で再構築をリクエストする
            // (音切れを防ぐため、再構築完了まで古いSRのまま動作させるか、あるいはクロスフェードさせるのが理想だが、
            //  ここでは即座に切り替えるための準備を行う)

            // リロードフラグは使わず、直接再構築をリクエスト
            // Audio ThreadからMessage Threadへの非同期呼び出し
            if (!isLoading.load() && isIRLoaded())
            {
                juce::WeakReference<ConvolverProcessor> weakThis (this);
                auto task = [weakThis]() {
                    if (weakThis) weakThis->loadImpulseResponse(juce::File(), false); // File空 = リビルドモード
                };

                if (juce::MessageManager::getInstance()->isThisTheMessageThread())
                    task();
                else
                    juce::MessageManager::callAsync(task);
            }
        }
        else
        {
            auto conv = convolution.load();
            if (conv) {
                conv->prepare(spec);
                conv->reset();
            }
        }
    }

    // DelayLine準備
    // 最大レイテンシーを多めに確保 (e.g., 2秒)
    delayLine.prepare(spec);
    delayLine.setMaximumDelayInSamples(static_cast<int>(sampleRate * 2.0));

    // Dryバッファ確保
    dryBuffer.setSize(2, samplesPerBlock);
    dryBuffer.clear();

    mixSmoother.reset(sampleRate, 0.05); // 50msでスムージング
    mixSmoother.setCurrentAndTargetValue(mixTarget.load());

    convolutionBuffer.setSize(2, samplesPerBlock);
    convolutionBuffer.clear();

    isPrepared.store(true, std::memory_order_release);
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
static juce::AudioBuffer<float> convertToMinimumPhase(const juce::AudioBuffer<float>& linearIR, juce::Thread* thread, bool* wasCancelled)
{
    if (wasCancelled) *wasCancelled = false;

    const int numSamples = linearIR.getNumSamples();
    // ゼロパディングを含めて十分なサイズを確保 (4倍程度が安全)
    const int fftSize = juce::nextPowerOfTwo(numSamples * 4);

    juce::dsp::FFT fft(static_cast<int>(std::log2(fftSize)));

    juce::AudioBuffer<float> minPhaseIR(linearIR.getNumChannels(), numSamples);

    // JUCE FFT用バッファ (Float)
    // AudioBuffer does not guarantee 32-byte alignment required for AVX2.
    // Use HeapBlock with explicit alignment.
    juce::HeapBlock<std::complex<float>> fftDataBlock;
    fftDataBlock.malloc(fftSize, 32);
    auto* fftDataFloat = fftDataBlock.getData();

    // 計算用バッファ (Double) - 精度向上のため
    std::vector<std::complex<double>> fftDataDouble(fftSize);

    // JUCE dsp::FFT inverse transform is already scaled by 1/N.
    // No manual scaling required.

    for (int ch = 0; ch < linearIR.getNumChannels(); ++ch)
    {
        // キャンセルチェック
        if (thread != nullptr && thread->threadShouldExit())
        {
            if (wasCancelled) *wasCancelled = true;
            return {};
        }

        // 1. IRをコピー (Realパート)
        const float* src = linearIR.getReadPointer(ch);
        for (int i = 0; i < numSamples; ++i)
            fftDataFloat[i] = std::complex<float>(src[i], 0.0f);
        for (int i = numSamples; i < fftSize; ++i)
            fftDataFloat[i] = std::complex<float>(0.0f, 0.0f);

        // 2. FFT (Time -> Freq)
        fft.perform(fftDataFloat, fftDataFloat, false);
        if (thread != nullptr && thread->threadShouldExit())
        {
            if (wasCancelled) *wasCancelled = true;
            return {};
        }

        // Doubleへ変換
        for (int i = 0; i < fftSize; ++i)
            fftDataDouble[i] = std::complex<double>(fftDataFloat[i]);

        // 3. 対数マグニチュードスペクトル計算 (Real=Log|H|, Imag=0) [Double]
        for (int i = 0; i < fftSize; ++i)
        {
            double mag = std::abs(fftDataDouble[i]);
            if (!std::isfinite(mag)) mag = 0.0;
            // ゼロ除算防止 (doubleの極小値を使用)
            double logMag = std::log(std::max(mag, 1.0e-100));
            fftDataDouble[i] = std::complex<double>(logMag, 0.0);
        }

        // Floatへ戻す
        for (int i = 0; i < fftSize; ++i)
            fftDataFloat[i] = std::complex<float>(static_cast<float>(fftDataDouble[i].real()),
                                                  static_cast<float>(fftDataDouble[i].imag()));

        // 4. IFFT (Freq -> Time) => 実ケプストラム (Real Cepstrum)
        fft.perform(fftDataFloat, fftDataFloat, true);
        if (thread != nullptr && thread->threadShouldExit())
        {
            if (wasCancelled) *wasCancelled = true;
            return {};
        }

        // Doubleへ変換 & スケーリング
        for (int i = 0; i < fftSize; ++i)
            fftDataDouble[i] = std::complex<double>(fftDataFloat[i]);

        // 5. 因果的ウィンドウ適用 (リフタリング) [Double]
        // c[0] = c[0]
        // c[n] = 2*c[n] (0 < n < N/2)
        // c[N/2] = c[N/2]
        // c[n] = 0 (N/2 < n < N)
        for (int i = 1; i < fftSize / 2; ++i)
            fftDataDouble[i] *= 2.0;

        for (int i = fftSize / 2 + 1; i < fftSize; ++i)
            fftDataDouble[i] = 0.0;

        // Floatへ戻す
        for (int i = 0; i < fftSize; ++i)
            fftDataFloat[i] = std::complex<float>(static_cast<float>(fftDataDouble[i].real()),
                                                  static_cast<float>(fftDataDouble[i].imag()));

        // 6. FFT (Time -> Freq) => 解析信号の対数スペクトル (実部が対数振幅、虚部が最小位相)
        fft.perform(fftDataFloat, fftDataFloat, false);
        if (thread != nullptr && thread->threadShouldExit())
        {
            if (wasCancelled) *wasCancelled = true;
            return {};
        }

        // Doubleへ変換
        for (int i = 0; i < fftSize; ++i)
            fftDataDouble[i] = std::complex<double>(fftDataFloat[i]);

        // 7. 複素指数変換 (exp) => 最小位相スペクトル [Double]
        for (int i = 0; i < fftSize; ++i)
        {
            double real = fftDataDouble[i].real();
            double imag = fftDataDouble[i].imag();
            // 数値オーバーフロー防止のためのクランプ
            real = juce::jlimit(-50.0, 50.0, real);
            imag = juce::jlimit(-50.0, 50.0, imag);
            fftDataDouble[i] = std::exp(std::complex<double>(real, imag));
        }

        // Floatへ戻す
        for (int i = 0; i < fftSize; ++i)
            fftDataFloat[i] = static_cast<std::complex<float>>(fftDataDouble[i]);

        // 8. IFFT (Freq -> Time) => 時間領域の最小位相IR
        fft.perform(fftDataFloat, fftDataFloat, true);
        if (thread != nullptr && thread->threadShouldExit())
        {
            if (wasCancelled) *wasCancelled = true;
            return {};
        }

        // 9. 結果をコピー
        float* dst = minPhaseIR.getWritePointer(ch);
        for (int i = 0; i < numSamples; ++i)
            dst[i] = static_cast<float>(fftDataFloat[i].real());
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

    if (!isRebuild && !irFile.existsAsFile())
    {
        return false;
    }

    if (isRebuild && originalIR.getNumSamples() == 0)
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
void ConvolverProcessor::applyNewState(std::shared_ptr<juce::dsp::Convolution> newConv,
                                       const juce::AudioBuffer<float>& loadedIR,
                                       double loadedSR,
                                       int targetLength,
                                       bool isRebuild,
                                       const juce::File& file,
                                       juce::AudioBuffer<float> displayIR)
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
    // ✅ 修正: 表示用には現在のサンプルレートを使用 (loadedSRはリサンプリング後のレート)
    createFrequencyResponseSnapshot(displayIR, loadedSR);

    // 安全に差し替え (Atomic Swap)
    auto oldConv = convolution.exchange(newConv);

    if (oldConv)
    {
        const juce::ScopedLock sl(trashBinLock);
        trashBin.push_back(oldConv);

        // ゴミ箱のサイズ制限 (メモリ肥大化防止)
        if (trashBin.size() > 5)
        {
            trashBin.erase(trashBin.begin());
        }
    }

    // 現在の有効なIR長を更新
    irLength = targetLength;
    currentSampleRate.store(currentSpec.sampleRate);

    isLoading.store(false);
    sendChangeMessage();
}

//--------------------------------------------------------------
// computeTargetIRLength
// 1.0秒固定長を計算し、最大長で制限する
//--------------------------------------------------------------
int ConvolverProcessor::computeTargetIRLength(double sampleRate, int originalLength)
{
    static constexpr double kTargetIRTimeSec = 1.0;
    static constexpr int kMaxIRCap = 524288;

    int target = static_cast<int>(sampleRate * kTargetIRTimeSec);

    target = std::min(target, kMaxIRCap);

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

        startBin = std::max(1, startBin); // DCを含めない
        endBin   = std::min(static_cast<int>(magnitudes.size()) - 1, endBin);

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
void ConvolverProcessor::createWaveformSnapshot (const juce::AudioBuffer<float>& irBuffer)
{
    irWaveform.assign(WAVEFORM_POINTS, 0.0f);

    const int numSamples = irBuffer.getNumSamples();
    const int numChannels = irBuffer.getNumChannels();

    if (numSamples <= 0 || numChannels <= 0)
        return;

    const int samplesPerPoint = std::max(1, numSamples / WAVEFORM_POINTS);

    float maxAbs = 0.0f;

    for (int i = 0; i < WAVEFORM_POINTS; ++i)
    {
        float peak = 0.0f;
        int startSample = i * samplesPerPoint;
        int endSample = std::min(numSamples, startSample + samplesPerPoint);

        // 全チャンネルのピークを取得
        for (int ch = 0; ch < numChannels; ++ch)
            for (int j = startSample; j < endSample; ++j)
                peak = std::max(peak, std::abs(irBuffer.getReadPointer(ch)[j]));

        irWaveform[i] = peak;
        maxAbs = std::max(maxAbs, peak);
    }

    // 正規化 (表示用)
    if (maxAbs > 0.0f)
        for (float& val : irWaveform) val /= maxAbs;
}

//--------------------------------------------------------------
// createFrequencyResponseSnapshot
// IRの周波数特性（マグニチュード）を計算する
//--------------------------------------------------------------
void ConvolverProcessor::createFrequencyResponseSnapshot(const juce::AudioBuffer<float>& irBuffer, double sampleRate)
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

    // ✅ キャッシュされたバッファを再利用 (メモリ確保のオーバーヘッド削減)
    if (cachedFFTBuffer.size() < static_cast<size_t>(fftSize * 2))
        cachedFFTBuffer.resize(static_cast<size_t>(fftSize * 2));

    std::fill(cachedFFTBuffer.begin(), cachedFFTBuffer.end(), 0.0f);

    // チャンネル0 (Lch) の特性を使用する
    const float* src = irBuffer.getReadPointer(0);
    const int copyLen = std::min(numSamples, fftSize);
    std::memcpy(cachedFFTBuffer.data(), src, copyLen * sizeof(float));

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
// createConfiguredConvolution
//--------------------------------------------------------------
std::shared_ptr<juce::dsp::Convolution> ConvolverProcessor::createConfiguredConvolution(double sampleRate, int maxBlockSize, double irSeconds)
{
    auto conv = std::make_shared<juce::dsp::Convolution>();

    juce::dsp::ProcessSpec spec;
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = static_cast<juce::uint32>(maxBlockSize);
    spec.numChannels = 2;
    conv->prepare(spec);

    // Non-Uniform Partitioning
    // Note: JUCEのConvolutionクラスは自動的にパーティションサイズを決定します。
    // 高負荷環境（192kHzなど）で手動最適化が必要な場合は、ここで head/tail サイズを計算し、
    // カスタム実装のConvolutionエンジンに渡す設計を検討してください。
    // 現状のJUCE標準APIでは自動設定に任せるのが最良です。

    (void)irSeconds; // Unused parameter suppression

    return conv;
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

    // Convolutionオブジェクトの同期 (Atomic)
    auto otherConv = other.convolution.load(std::memory_order_acquire);
    auto expectedConv = convolution.load(std::memory_order_acquire);

    if (otherConv != expectedConv)
    {
        // Compare-and-swap で安全に更新
        convolution.compare_exchange_strong(expectedConv, otherConv,
                                           std::memory_order_acq_rel,
                                           std::memory_order_acquire);
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
void ConvolverProcessor::process(juce::AudioBuffer<double>& buffer, int numSamples)
{
    // (A) Denormal対策 (重要)
    juce::ScopedNoDenormals noDenormals;

    // ── (B) 無音ブロック最適化 ──
    // 入力が無音の場合は処理をスキップ
    bool isSilent = true;
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
    {
        if (buffer.getMagnitude(ch, 0, numSamples) > 1.0e-8)
        {
            isSilent = false;
            break;
        }
    }
    if (isSilent) return;

    // ── Step 1: RCU State Load (Lock-free / Wait-free) ──
    auto conv = convolution.load(std::memory_order_acquire);

    if (conv)
    {
        const int latency = conv->getLatency();
        delayLine.setDelay(static_cast<float>(latency));
        currentLatency.store(latency);
    }

    // ── Step 2: 処理実行可能かチェック ──
    // バイパス、未準備、IR未ロードの場合はスルー
    if (!isPrepared.load(std::memory_order_acquire) || bypassed.load() || !conv)
    {
        return;
    }

    // ✅ 修正: processBufferのチャンネル数を使用 (最大2ch)
    const int procChannels = std::min(buffer.getNumChannels(), 2);

    // ── Step 3: バッファサイズ安全対策 (Bounds Check) ──
    if (numSamples <= 0 || procChannels == 0 || numSamples > dryBuffer.getNumSamples() || numSamples > convolutionBuffer.getNumSamples())
        return;

    // ── Step 4: パラメータ更新と最適化 ──
    mixSmoother.setTargetValue(mixTarget.load(std::memory_order_relaxed));

    const float currentMix = mixSmoother.getTargetValue();
    const bool isSmoothing = mixSmoother.isSmoothing();

    // ── 最適化: 処理内容をミックス比率に応じて決定 ──
    const bool needsConvolution = isSmoothing || currentMix > 0.001f;
    const bool needsDrySignal   = isSmoothing || currentMix < 0.999f;

    // ── Step 5: Dry信号生成 (必要な場合のみ) ──
    if (needsDrySignal)
    {
        // Dry信号をdryBufferにコピーし、レイテンシー分遅延
        juce::dsp::AudioBlock<const double> inputBlock(buffer.getArrayOfReadPointers(), procChannels, numSamples);
        juce::dsp::AudioBlock<double> outputBlock(dryBuffer.getArrayOfWritePointers(), procChannels, numSamples);
        juce::dsp::ProcessContextNonReplacing<double> delayContext(inputBlock, outputBlock);
        delayLine.process(delayContext);
    }

    // ── Step 6: Wet信号生成 (必要な場合のみ) ──
    if (needsConvolution)
    {
        // Convolutionはfloatのみ対応のため、double -> float変換
        for (int ch = 0; ch < procChannels; ++ch)
        {
            const double* src = buffer.getReadPointer(ch);
            float* dst = convolutionBuffer.getWritePointer(ch);
            for (int i = 0; i < numSamples; ++i)
                dst[i] = static_cast<float>(src[i]);
        }

        // convolutionBufferの実際のチャンネル数（2）を使用
        juce::dsp::AudioBlock<float> block(convolutionBuffer.getArrayOfWritePointers(), 2, numSamples);
        juce::dsp::ProcessContextReplacing<float> context(block);

        conv->process(context);

        // Wet信号に-6dBのヘッドルームを確保 (より保守的なクリッピング防止)
        convolutionBuffer.applyGain(0.5f);
    }

    // ── Step 7: Dry/Wet Mix ──
    if (!needsConvolution) // 100% Dry
    {
        for (int ch = 0; ch < procChannels; ++ch)
            buffer.copyFrom(ch, 0, dryBuffer, ch, 0, numSamples);
    }
    else if (!needsDrySignal) // 100% Wet
    {
        for (int ch = 0; ch < procChannels; ++ch)
        {
            const float* wetSrc = convolutionBuffer.getReadPointer(ch);
            double* dst = buffer.getWritePointer(ch);
            for (int i = 0; i < numSamples; ++i)
                dst[i] = static_cast<double>(wetSrc[i]);
        }
    }
    else
    {
        // 0% < Mix < 100% または スムージング中
        if (mixSmoother.isSmoothing())
        {
            const float* wetPtrs[2] = { nullptr };
            const double* dryPtrs[2] = { nullptr };
            double* dstPtrs[2] = { nullptr };

            for (int ch = 0; ch < procChannels; ++ch)
            {
                wetPtrs[ch] = convolutionBuffer.getReadPointer(ch);
                dryPtrs[ch] = dryBuffer.getReadPointer(ch);
                dstPtrs[ch] = buffer.getWritePointer(ch);
            }

            for (int i = 0; i < numSamples; ++i)
            {
                const double mixValue = static_cast<double>(mixSmoother.getNextValue());
                const double wetGain = std::sin(mixValue * juce::MathConstants<double>::halfPi);
                const double dryGain = std::cos(mixValue * juce::MathConstants<double>::halfPi);

                for (int ch = 0; ch < procChannels; ++ch)
                {
                    dstPtrs[ch][i] = static_cast<double>(wetPtrs[ch][i]) * wetGain + dryPtrs[ch][i] * dryGain;
                }
            }
        }
        else
        {
            const double mixValue = mixSmoother.getTargetValue();
            const double wetGain = std::sin(mixValue * juce::MathConstants<double>::halfPi);
            const double dryGain = std::cos(mixValue * juce::MathConstants<double>::halfPi);

            for (int ch = 0; ch < procChannels; ++ch)
            {
                const float* wet = convolutionBuffer.getReadPointer(ch);
                const double* dry = dryBuffer.getReadPointer(ch);
                double* dst = buffer.getWritePointer(ch);

                for (int i = 0; i < numSamples; ++i)
                    dst[i] = static_cast<double>(wet[i]) * wetGain + dry[i] * dryGain;
            }
        }
    }
}

//--------------------------------------------------------------
// rebuild (Message Thread / Helper)
//--------------------------------------------------------------
void ConvolverProcessor::rebuild(double sampleRate, int maxBlockSize, double irSeconds)
{
    // この関数はLoaderThreadから呼び出されることを想定しているが、
    // 実際にはLoaderThread内でcreateConfiguredConvolutionを使用しているため、
    // ここではインターフェースとしての実装を提供する。
    // 必要に応じて、同期的なリビルド処理を実装することも可能。
    (void)sampleRate; (void)maxBlockSize; (void)irSeconds;
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
        sendChangeMessage();
    }
}

float ConvolverProcessor::getMix() const
{
    return mixTarget.load();
}

void ConvolverProcessor::setUseMinPhase(bool shouldUseMinPhase)
{
    if (useMinPhase.load() != shouldUseMinPhase)
    {
        useMinPhase.store(shouldUseMinPhase);
        sendChangeMessage();

        // 設定変更時にIRがロード済みなら再ロードして変換を適用
        juce::File fileToLoad;
        {
            const juce::ScopedLock sl(irFileLock);
            fileToLoad = currentIrFile;
        }
        if (fileToLoad.existsAsFile())
        {
            loadImpulseResponse(fileToLoad);
        }
    }
}
