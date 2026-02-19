//============================================================================
// AudioEngine.cpp  ── v0.1 (JUCE 8.0.12対応)
//
// AudioEngineの実装
//============================================================================
#include "AudioEngine.h"
#include <cmath>
#include <complex>
#include <algorithm>

//--------------------------------------------------------------
// コンストラクタ
//--------------------------------------------------------------
AudioEngine::AudioEngine()
{
    // デフォルトサンプルレート (0 = 未初期化/デバイスなし)
    currentSampleRate.store(0);

    // バッファ初期化
    audioFifoBuffer.setSize (2, FIFO_SIZE);
    audioFifoBuffer.clear();
}

void AudioEngine::initialize()
{
    // 初期DSP構築 (デフォルト設定)
    // 安全対策: バッファサイズを余裕を持って確保 (8192)
    // これにより、デバイス初期化前やバッファサイズ変更時の不整合による音切れ/無音を防ぐ
    requestRebuild(48000.0, SAFE_MAX_BLOCK_SIZE);
    maxSamplesPerBlock.store(SAFE_MAX_BLOCK_SIZE);
    currentSampleRate.store(48000);

    uiConvolverProcessor.addChangeListener(this);
    uiEqProcessor.addChangeListener(this);
    uiConvolverProcessor.addListener(this);
    uiEqProcessor.addListener(this);
}

AudioEngine::~AudioEngine()
{
    uiConvolverProcessor.removeChangeListener(this);
    uiEqProcessor.removeChangeListener(this);
    uiConvolverProcessor.removeListener(this);
    uiEqProcessor.removeListener(this);

    // 進行中のコールバックが完了するのを待つため、DSPを無効化
    // これにより、changeListenerCallback内でdsp->...へのアクセスを防ぐ
    currentDSP.store(nullptr);
}

//--------------------------------------------------------------
// FIFOからデータ読み出し (UI Thread)
//--------------------------------------------------------------
void AudioEngine::readFromFifo(float* dest, int numSamples)
{
    int start1, size1, start2, size2;
    audioFifo.prepareToRead(numSamples, start1, size1, start2, size2);

    // 実際に読み取れるサンプル数を計算 (FIFO内の有効データ量に依存)
    // prepareToRead は numSamples 分の領域を返さない場合がある (FIFO不足時)
    const int actualRead = size1 + size2;

    if (size1 > 0)
    {
        const float* srcL = audioFifoBuffer.getReadPointer(0, start1);
        const float* srcR = audioFifoBuffer.getReadPointer(1, start1);
        for (int i = 0; i < size1; ++i)
            dest[i] = (srcL[i] + srcR[i]) * 0.5f;
    }

    if (size2 > 0)
    {
        const float* srcL = audioFifoBuffer.getReadPointer(0, start2);
        const float* srcR = audioFifoBuffer.getReadPointer(1, start2);
        for (int i = 0; i < size2; ++i)
            dest[size1 + i] = (srcL[i] + srcR[i]) * 0.5f;
    }

    // 実際に読み取った分だけFIFOを進める
    if (actualRead > 0)
        audioFifo.finishedRead(actualRead);

    // 足りない分はゼロ埋め (グリッチ防止)
    if (actualRead < numSamples)
        std::memset(dest + actualRead, 0, (numSamples - actualRead) * sizeof(float));
}

//--------------------------------------------------------------
// EQ応答曲線計算
// 現在のEQ設定に基づき、周波数ごとのトータルゲイン応答（マグニチュード）を計算する
//--------------------------------------------------------------
void AudioEngine::calcEQResponseCurve(float* outMagnitudesL,
                                     float* outMagnitudesR,
                                     const std::complex<double>* zArray,
                                     int numPoints)
{
    const int sr = currentSampleRate.load();
    if (sr <= 0)
    {
        for (int i = 0; i < numPoints; ++i)
        {
            if (outMagnitudesL) outMagnitudesL[i] = 1.0f;
            if (outMagnitudesR) outMagnitudesR[i] = 1.0f;
        }
        return;
    }

    // ── 最適化: 有効なバンドの係数をスタック上で事前に計算 ──
    // UIスレッドでの計算負荷を下げるため、無効なバンドやゲイン0のバンドは除外する
    struct ActiveBand {
        EQCoeffsBiquad coeffs;
        EQChannelMode mode;
    };
    ActiveBand activeBands[EQProcessor::NUM_BANDS];
    int numActiveBands = 0;

    // 状態スナップショットを取得して一貫性を確保
    auto eqState = uiEqProcessor.getEQState();

    for (int band = 0; band < EQProcessor::NUM_BANDS; ++band)
    {
        const auto& params = eqState->bands[band];
        if (!params.enabled) continue;

        EQBandType type = eqState->bandTypes[band];

        // LowPass/HighPass以外でゲインがほぼ0の場合はスキップ
        if (type != EQBandType::LowPass && type != EQBandType::HighPass &&
            std::abs(params.gain) < EQ_GAIN_EPSILON)
            continue;

        activeBands[numActiveBands++] = {
            EQProcessor::calcBiquadCoeffs(type, params.frequency, params.gain, params.q, sr),
            eqState->bandChannelModes[band]
        };
    }

    float totalGainLinear = 1.0f;
    if (!uiEqProcessor.getAGCEnabled())
    {
        totalGainLinear = juce::Decibels::decibelsToGain(eqState->totalGainDb);
    }

    // ── 最適化: 有効なバンドがない、かつトータルゲインが0dBの場合は計算をスキップ ──
    if (numActiveBands == 0 && std::abs(totalGainLinear - 1.0f) < EQ_UNITY_GAIN_EPSILON)
    {
        if (outMagnitudesL) std::fill_n(outMagnitudesL, numPoints, 1.0f);
        if (outMagnitudesR) std::fill_n(outMagnitudesR, numPoints, 1.0f);
        return;
    }

    const float totalGainSq = totalGainLinear * totalGainLinear;

    for (int i = 0; i < numPoints; ++i)
    {
        // 各バンドの応答を計算
        // 二乗マグニチュードを積算して最後にsqrtすることで、ループ内のsqrtを回避
        float totalMagSqL = totalGainSq;
        float totalMagSqR = totalGainSq;

        // 事前計算された z (e^jw) を使用
        // これにより、ここで sin/cos を計算する必要がなくなる
        const std::complex<double> z = zArray[i];

        for (int b = 0; b < numActiveBands; ++b)
        {
            const auto& band = activeBands[b];
            float magSq = EQProcessor::getMagnitudeSquared(band.coeffs, z);

            // 数値安定性のため、NaN/Infの伝播を防止
            if (!std::isfinite(magSq))
                magSq = 1.0f;

            if (band.mode == EQChannelMode::Stereo || band.mode == EQChannelMode::Left)
                totalMagSqL *= magSq;
            if (band.mode == EQChannelMode::Stereo || band.mode == EQChannelMode::Right)
                totalMagSqR *= magSq;
        }

        // 最終的なNaNチェック
        if (outMagnitudesL)
        {
            float val = std::sqrt(totalMagSqL);
            outMagnitudesL[i] = std::isfinite(val) ? val : 1.0f;
        }
        if (outMagnitudesR)
        {
            float val = std::sqrt(totalMagSqR);
            outMagnitudesR[i] = std::isfinite(val) ? val : 1.0f;
        }
    }
}

//--------------------------------------------------------------
// prepareToPlay
//--------------------------------------------------------------
void AudioEngine::prepareToPlay (int samplesPerBlockExpected, double sampleRate)
{
    // パラメータ検証 (Parameter Validation)
    // 不正なパラメータから保護
    if (sampleRate <= 0.0 || sampleRate > SAFE_MAX_SAMPLE_RATE)
    {
        jassertfalse; // デバッグビルドで警告
        currentSampleRate.store(48000); // デフォルト値
        return; // 初期化をスキップ
    }

    if (samplesPerBlockExpected <= 0)
    {
        jassertfalse;
        samplesPerBlockExpected = 512; // フォールバックして続行
    }

    // ASIO同期: デバイスの実際のブロックサイズを使用する
    // SAFE_MAX_BLOCK_SIZEで固定すると、ASIOのブロックサイズ変更に追従できず破綻するため。
    const int bufferSize = samplesPerBlockExpected;

    // サンプルレート変更検知
    const bool rateChanged = (std::abs(currentSampleRate.load() - sampleRate) > 1.0);
    // ブロックサイズ変更検知 (FFTConvolverのパーティションサイズ最適化のため)
    const bool blockSizeChanged = (maxSamplesPerBlock.load() != bufferSize);

    // UI用プロセッサのサンプルレートも更新 (IR表示やパラメータ管理のため)
    uiConvolverProcessor.prepareToPlay(sampleRate, bufferSize);
    uiEqProcessor.prepareToPlay(static_cast<int>(sampleRate), bufferSize);

    if (rateChanged || blockSizeChanged)
    {
        uiConvolverProcessor.rebuildAllIRs();
    }

    maxSamplesPerBlock.store(bufferSize);
    // DSP再構築 (RT安全化: 新しいDSPを作成してスワップ)
    requestRebuild(sampleRate, bufferSize);

    currentSampleRate.store(static_cast<int>(sampleRate));
    uiConvolverProcessor.setBypass(convBypassActive.load (std::memory_order_relaxed));
    audioFifo.reset();

    // レベルメーターのリセット
    inputLevelDb.store(LEVEL_METER_MIN_DB);
    outputLevelDb.store(LEVEL_METER_MIN_DB);

    // ===== bypass 状態の初期化 =====
    // 再生中のリアルタイムな更新は getNextAudioBlock() で行われる
    eqBypassActive.store (eqBypassRequested.load (std::memory_order_relaxed), std::memory_order_relaxed);
    convBypassActive.store (convBypassRequested.load (std::memory_order_relaxed), std::memory_order_relaxed);

    // ConvolverProcessorの状態も同期させておく（念のため）
    uiConvolverProcessor.setBypass(convBypassActive.load (std::memory_order_relaxed));
}

//--------------------------------------------------------------
// requestRebuild - DSPグラフの再構築 (Message Thread)
//--------------------------------------------------------------
void AudioEngine::requestRebuild(double sampleRate, int samplesPerBlock)
{
    // 新しいDSPコアを作成
    auto newDSP = std::make_shared<DSPCore>();

    // UIプロセッサから状態をコピー
    newDSP->eq.syncStateFrom(uiEqProcessor); // 最適化: ValueTreeを経由せず直接同期
    newDSP->convolver.syncStateFrom(uiConvolverProcessor);

    // 準備
    newDSP->prepare(sampleRate, samplesPerBlock, ditherBitDepth.load(), manualOversamplingFactor.load(), oversamplingType.load());

    commitNewDSP(newDSP);
}

void AudioEngine::commitNewDSP(std::shared_ptr<DSPCore> newDSP)
{
    // Atomic Swap
    auto oldDSP = currentDSP.exchange(newDSP, std::memory_order_acq_rel);

    if (oldDSP)
    {
        // 古いDSPをゴミ箱へ (Audio Threadが使用中の可能性があるため即削除しない)
        const juce::ScopedLock sl(trashBinLock);
        trashBin.push_back(std::move(oldDSP));

        // ゴミ箱の掃除 (Garbage Collection)
        // Audio Threadが参照していない(use_count == 1)オブジェクトのみを削除する。
        // これにより、Audio Thread内でのデストラクタ実行(ロックやメモリ解放)を確実に防ぐ。
        trashBin.erase(std::remove_if(trashBin.begin(), trashBin.end(), [](const auto& p) { return p.use_count() == 1; }), trashBin.end());
    }
}

void AudioEngine::changeListenerCallback(juce::ChangeBroadcaster* source)
{
    // UIプロセッサからの構造変更（プリセットロード、IRロードなど）を検知
    if (source == &uiEqProcessor || source == &uiConvolverProcessor)
    {
        // DSPグラフを安全に再構築
        requestRebuild(currentSampleRate.load(), maxSamplesPerBlock.load());

        // UIに更新を通知 (MainWindowが受け取る)
        sendChangeMessage();
    }
}

void AudioEngine::eqBandChanged(EQProcessor* processor, int bandIndex)
{
    if (processor == &uiEqProcessor)
    {
        auto dsp = currentDSP.load();
        if (dsp)
            dsp->eq.syncBandNodeFrom(uiEqProcessor, bandIndex);
    }
}

void AudioEngine::eqGlobalChanged(EQProcessor* processor)
{
    if (processor == &uiEqProcessor)
    {
        auto dsp = currentDSP.load();
        if (dsp)
            dsp->eq.syncGlobalStateFrom(uiEqProcessor);
    }
}

void AudioEngine::convolverParamsChanged(ConvolverProcessor* processor)
{
    if (processor == &uiConvolverProcessor)
    {
        auto dsp = currentDSP.load();
        if (dsp)
            dsp->convolver.syncParametersFrom(uiConvolverProcessor);
    }
}

//--------------------------------------------------------------
// releaseResources
//--------------------------------------------------------------
void AudioEngine::releaseResources()
{
    // サンプルレートをリセット (描画停止用)
    currentSampleRate.store(0);

    // レベルをリセット
    inputLevelDb.store(-120.0f);
    outputLevelDb.store(-120.0f);
}

//--------------------------------------------------------------
// getNextAudioBlock - オーディオ処理コールバック (Audio Thread)
// リアルタイム制約 (Real-time Constraints)
//    1. メモリ割り当て禁止 (No memory allocation): new, malloc, vector::resize, AudioBuffer::setSize 等はNG。
//    2. ロック禁止 (No locks): Mutex, CriticalSection 等によるブロックはNG。
//    3. システムコール禁止 (No system calls): ファイルI/O, コンソール出力(printf) 等はNG。
//    4. 待機禁止 (No waiting): sleep や 重い計算によるストールを避ける。IRの再ロードもNG。
//    5. 禁止API: AudioBlock::allocate, AudioBlock::copyFrom (確保伴うもの), FFT::performFrequencyOnlyForwardTransform (事前確保なしはNG)
//    6. std::vector使用時は、必ず AudioBuffer / 生ポインタを wrap する形で使用すること。
//--------------------------------------------------------------
void AudioEngine::getNextAudioBlock (const juce::AudioSourceChannelInfo& bufferToFill)
{
    // (A) Denormal対策 (重要: CPU負荷スパイク防止)
    juce::ScopedNoDenormals noDenormals;

    // 入力検証 (Input Validation)
    if (bufferToFill.buffer == nullptr)
        return;

    const int numSamples = bufferToFill.numSamples;
    const int startSample = bufferToFill.startSample;
    auto* buffer = bufferToFill.buffer;

    // サンプル数の妥当性チェック
    if (numSamples <= 0 || numSamples > maxSamplesPerBlock.load())
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    // startSampleの妥当性チェック
    if (startSample < 0 || startSample + numSamples > buffer->getNumSamples())
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    // DSPコアの取得 (Atomic Load)
    auto dsp = currentDSP.load(std::memory_order_acquire);

    if (dsp)
    {
        // パラメータのロード
        const bool eqBypassed = eqBypassRequested.load(std::memory_order_relaxed);
        const bool convBypassed = convBypassRequested.load(std::memory_order_relaxed);
        const ProcessingOrder order = currentProcessingOrder.load(std::memory_order_relaxed);
        const AnalyzerSource analyzerSource = currentAnalyzerSource.load(std::memory_order_relaxed);
        const bool softClip = softClipEnabled.load(std::memory_order_relaxed);
        const float satAmt = saturationAmount.load(std::memory_order_relaxed);

        // UI表示用の状態更新
        eqBypassActive.store(eqBypassed, std::memory_order_relaxed);
        convBypassActive.store(convBypassed, std::memory_order_relaxed);

        // 処理委譲
        dsp->process(bufferToFill, audioFifo, audioFifoBuffer, inputLevelDb, outputLevelDb, {eqBypassed, convBypassed, order, analyzerSource, softClip, satAmt});
    }
    else
    {
        bufferToFill.clearActiveBufferRegion();
    }
}

//--------------------------------------------------------------
// DSPCore Implementation
//--------------------------------------------------------------
AudioEngine::DSPCore::DSPCore() = default;

void AudioEngine::DSPCore::prepare(double sampleRate, int samplesPerBlock, int bitDepth, int manualOversamplingFactor, OversamplingType oversamplingType)
{
    maxSamplesPerBlock = samplesPerBlock;

    size_t factorLog2 = 0;
    if (manualOversamplingFactor > 0)
    {
        // Manual override
        if (manualOversamplingFactor == 8)      factorLog2 = 3;
        else if (manualOversamplingFactor == 4) factorLog2 = 2;
        else if (manualOversamplingFactor == 2) factorLog2 = 1;
        else                                    factorLog2 = 0; // 1x
    }
    else
    {
        // Auto mode (default)
        // 44.1k, 48k -> 4x
        // 88.2k, 96k -> 2x
        // >= 176.4k  -> 1x (None)
        if (sampleRate < 80000.0)      factorLog2 = 2; // 4x
        else if (sampleRate < 160000.0) factorLog2 = 1; // 2x
        else                            factorLog2 = 0; // 1x
    }

    oversamplingFactor = (size_t)1 << factorLog2;

    if (factorLog2 > 0)
    {
        auto filterType = (oversamplingType == OversamplingType::LinearPhase)
                          ? juce::dsp::Oversampling<double>::filterHalfBandFIREquiripple
                          : juce::dsp::Oversampling<double>::filterHalfBandPolyphaseIIR;
        oversampling = std::make_unique<juce::dsp::Oversampling<double>>(2, factorLog2, filterType);
        oversampling->initProcessing(samplesPerBlock);
    }
    else
    {
        oversampling.reset();
    }

    const double processingRate = sampleRate * static_cast<double>(oversamplingFactor);
    const int processingBlockSize = samplesPerBlock * static_cast<int>(oversamplingFactor);

    // プロセッサの準備
    convolver.prepareToPlay(processingRate, processingBlockSize);
    eq.prepareToPlay(static_cast<int>(processingRate), processingBlockSize);
    dcBlockerL.prepare(processingRate, processingBlockSize);
    dcBlockerR.prepare(processingRate, processingBlockSize);

    // ディザの準備 (出力段で行うため元のサンプルレート)
    dither.prepare(sampleRate, bitDepth);
    this->ditherBitDepth = bitDepth; // DSPCoreのメンバーに保存

    // バッファ確保 (Message Threadで実行されるため安全)
    processBuffer.setSize(2, samplesPerBlock);
}

void AudioEngine::DSPCore::reset()
{
    convolver.reset();
    eq.reset();
    dcBlockerL.reset();
    dcBlockerR.reset();
    dither.reset();
    if (oversampling)
        oversampling->reset();
    processBuffer.clear();
}

void AudioEngine::DSPCore::process(const juce::AudioSourceChannelInfo& bufferToFill,
                                  juce::AbstractFifo& audioFifo,
                                  juce::AudioBuffer<float>& audioFifoBuffer,
                                  std::atomic<float>& inputLevelDb,
                                  std::atomic<float>& outputLevelDb,
                                  const ProcessingState& state) // ProcessingState構造体でパラメータを受け取る
{
    const int numSamples = bufferToFill.numSamples;

    // バッファサイズ超過ガード (Buffer Overrun Protection)
    if (numSamples > maxSamplesPerBlock)
    {
        // この状況は通常発生しないが、万が一ホストが予期せぬサイズのバッファを渡してきた場合の安全策
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    processInput(bufferToFill, numSamples);

    //----------------------------------------------------------
    // 入力レベル計算
    //----------------------------------------------------------
    const float inputDb = measureLevel(processBuffer, numSamples);
    inputLevelDb.store(inputDb, std::memory_order_relaxed);

    // ── Analyzer Input Tap (Pre-DSP) ──
    if (state.analyzerSource == AnalyzerSource::Input)
    {
        pushToFifo(processBuffer, numSamples, audioFifo, audioFifoBuffer);
    }

    //----------------------------------------------------------
    // オーバーサンプリング処理ブロック
    //----------------------------------------------------------
    // バッファ全体ではなく、有効なサンプル数のみをラップする (重要)
    juce::dsp::AudioBlock<double> block = juce::dsp::AudioBlock<double>(processBuffer).getSubBlock(0, numSamples);
    juce::dsp::AudioBlock<double> processBlock = block;

    // アップサンプリング
    if (oversampling)
    {
        processBlock = oversampling->processSamplesUp(block);
    }

    int numProcSamples = (int)processBlock.getNumSamples();
    int numProcChannels = (int)processBlock.getNumChannels(); // 通常は2

    //----------------------------------------------------------
    // DSP処理チェーン (Dynamic Processing Order)
    //----------------------------------------------------------
    // プロセッサには AudioBlock を直接渡す (AudioBuffer作成によるmalloc回避)
    if (state.order == ProcessingOrder::ConvolverThenEQ) // stateから読み出し
    {
        // 1. Convolver
        if (!state.convBypassed) // stateから読み出し
            convolver.process(processBlock);
        // 2. EQ
        if (!state.eqBypassed) // stateから読み出し
            eq.process(processBlock);
    }
    else
    {
        // 1. EQ
        if (!state.eqBypassed) // stateから読み出し
            eq.process(processBlock);
        // 2. Convolver
        if (!state.convBypassed) // stateから読み出し
            convolver.process(processBlock);
    }

    //----------------------------------------------------------
    // ソフトクリッピング (Soft Clipping)
    // 配置: ダウンサンプリング前に行うことで、倍音成分の折り返しノイズ(エイリアシング)を低減する。
    //----------------------------------------------------------
    if (state.softClipEnabled) // stateから読み出し
    {
        const double sat = static_cast<double>(state.saturationAmount); // stateから読み出し
        const double CLIP_THRESHOLD = 0.95 - 0.45 * sat;
        const double CLIP_KNEE      = 0.05 + 0.35 * sat;
        const double CLIP_ASYMMETRY = 0.10 * sat;
        const double CLIP_START = CLIP_THRESHOLD - CLIP_KNEE;

        for (int ch = 0; ch < numProcChannels; ++ch)
        {
            double* data = processBlock.getChannelPointer(ch);
            for (int i = 0; i < numProcSamples; ++i)
            {
                if (std::abs(data[i]) > CLIP_START)
                    data[i] = musicalSoftClip(data[i], CLIP_THRESHOLD, CLIP_KNEE, CLIP_ASYMMETRY);
            }
        }
    }

    //----------------------------------------------------------
    // DCオフセット除去 (DC Offset Removal)
    // 目的: フィルタ処理後のDC成分を除去し、スピーカー保護とヘッドルーム確保を行う。
    // 配置: ソフトクリップの後、ダウンサンプリング前に行う。
    //----------------------------------------------------------
    for (int ch = 0; ch < numProcChannels; ++ch)
    {
        if (ch > 1) break; // DCBlockerは現在ステレオ(L/R)のみ対応

        auto* data = processBlock.getChannelPointer(ch);
        auto& blocker = (ch == 0) ? dcBlockerL : dcBlockerR;

        for (int i = 0; i < numProcSamples; ++i)
            data[i] = blocker.process(data[i]);
    }

    // ダウンサンプリング (結果は processBuffer に書き戻される)
    if (oversampling)
    {
        oversampling->processSamplesDown(block);
    }

    //----------------------------------------------------------
    // 出力レベル計算 (DC除去後のクリーンな信号で計測)
    //----------------------------------------------------------
    const float outputDb = measureLevel(processBuffer, numSamples);
    outputLevelDb.store(outputDb, std::memory_order_relaxed);

    // ── Analyzer Output Tap (Post-DSP) ──
    if (state.analyzerSource == AnalyzerSource::Output)
    {
        pushToFifo(processBuffer, numSamples, audioFifo, audioFifoBuffer);
    }

    processOutput(bufferToFill, numSamples);
}

float AudioEngine::DSPCore::measureLevel (const juce::AudioBuffer<SampleType>& buffer, int numSamples) const noexcept
{
    float maxLevel = 0.0f;
    const int numChannels = buffer.getNumChannels();

    for (int ch = 0; ch < numChannels; ++ch)
    {
        const float level = static_cast<float>(buffer.getMagnitude(ch, 0, numSamples));
        maxLevel = std::max(maxLevel, level);
    }

    return (maxLevel > LEVEL_METER_MIN_MAG) ? juce::Decibels::gainToDecibels(maxLevel) : LEVEL_METER_MIN_DB;
}

void AudioEngine::DSPCore::pushToFifo(const juce::AudioBuffer<SampleType>& buffer, int numSamples,
                                      juce::AbstractFifo& audioFifo,
                                      juce::AudioBuffer<float>& audioFifoBuffer) const
{
    const int procChannels = buffer.getNumChannels();
    const double* l = buffer.getReadPointer(0);
    const double* r = (procChannels > 1) ? buffer.getReadPointer(1) : nullptr;

    // FIFO空き容量チェック (Overflow Protection)
    if (audioFifo.getFreeSpace() >= numSamples)
    {
        int start1, size1, start2, size2;
        audioFifo.prepareToWrite(numSamples, start1, size1, start2, size2);

        // 第1セグメント書き込み
        if (size1 > 0)
        {
            float* destL = audioFifoBuffer.getWritePointer(0, start1);
            float* destR = audioFifoBuffer.getWritePointer(1, start1);
            for (int i = 0; i < size1; ++i)
            {
                destL[i] = static_cast<float>(l[i]);
                destR[i] = (r != nullptr) ? static_cast<float>(r[i]) : destL[i];
            }
            l += size1;
            if (r != nullptr) r += size1;
        }

        // 第2セグメント書き込み
        if (size2 > 0)
        {
            float* destL = audioFifoBuffer.getWritePointer(0, start2);
            float* destR = audioFifoBuffer.getWritePointer(1, start2);
            for (int i = 0; i < size2; ++i)
            {
                destL[i] = static_cast<float>(l[i]);
                destR[i] = (r != nullptr) ? static_cast<float>(r[i]) : destL[i];
            }
        }

        audioFifo.finishedWrite(size1 + size2);
    }
}

void AudioEngine::DSPCore::processInput(const juce::AudioSourceChannelInfo& bufferToFill, int numSamples)
{
    auto* buffer = bufferToFill.buffer;
    const int startSample = bufferToFill.startSample;
    const int effectiveInputChannels = std::min(buffer->getNumChannels(), 2);
    const int procChannels = processBuffer.getNumChannels();

    //----------------------------------------------------------
    // 入力データを processBuffer (double) にコピー
    //----------------------------------------------------------
    // ループ分割による分岐排除と最適化
    for (int ch = 0; ch < effectiveInputChannels; ++ch)
    {
        // float (I/O) -> double (DSP) conversion
        const float* src = buffer->getReadPointer(ch, startSample);
        double* dst = processBuffer.getWritePointer(ch);
        for (int i = 0; i < numSamples; ++i)
            dst[i] = static_cast<double>(src[i]);
    }

    // 入力がないチャンネル、または余剰チャンネルはクリア
    // ただし、Mono->Stereo展開を行う場合はCh 1のクリアをスキップする (直後に上書きされるため)
    int clearStartCh = effectiveInputChannels;
    if (effectiveInputChannels == 1 && procChannels > 1)
        clearStartCh = 2;

    for (int ch = clearStartCh; ch < procChannels; ++ch)
    {
        processBuffer.clear(ch, 0, numSamples);
    }

    // ── Mono -> Stereo 展開 ──
    // 入力が1chのみで、処理バッファが2ch以上ある場合、LchをRchにコピーする
    // これにより、モノラルマイク入力時などでもステレオ処理として扱えるようにし、
    // 後段のステレオエフェクト（Convolver等）での片側無音を防ぐ。
    if (effectiveInputChannels == 1 && procChannels > 1)
    {
        const double* src = processBuffer.getReadPointer(0);
        double* dst = processBuffer.getWritePointer(1);
        std::memcpy(dst, src, numSamples * sizeof(double));
    }
}

double AudioEngine::DSPCore::musicalSoftClip(double x, double threshold, double knee, double asymmetry) noexcept
{
    const double abs_x = std::abs(x);
    const double clip_start = threshold - knee;

    // 閾値以下はリニア
    if (abs_x < clip_start)
        return x;

    const double sign = (x > 0.0) ? 1.0 : -1.0;

    // ソフトニー領域 (ブレンド率計算)
    double knee_shape = 1.0;
    if (abs_x < threshold + knee)
    {
        // 3次多項式でスムーズなニー
        const double t = (abs_x - clip_start) / (2.0 * knee);
        knee_shape = t * t * (3.0 - 2.0 * t); // Smoothstep
    }

    const double linear = abs_x;
    // tanhによるソフトクリッピングカーブ
    const double clipped = threshold + knee * std::tanh((abs_x - threshold) / knee);

    // 非対称性の追加（真空管風）
    const double asymmetric_factor = 1.0 + asymmetry * sign * knee_shape;
    return sign * (linear * (1.0 - knee_shape) + clipped * knee_shape) * asymmetric_factor;
}

void AudioEngine::DSPCore::processOutput(const juce::AudioSourceChannelInfo& bufferToFill, int numSamples)
{
    auto* buffer = bufferToFill.buffer;
    const int startSample = bufferToFill.startSample;
    const int procChannels = processBuffer.getNumChannels();

    // ビット深度に基づくディザリング判定
    // 32-bit (float/int) 以上はディザ不要。24-bit/16-bit (int) はディザ必要。
    const bool applyDither = (ditherBitDepth < 32);

    //----------------------------------------------------------
    // 出力バッファへコピー & ディザリング (Output & TPDF Dither)
    // 目的: double -> float/int 変換時の量子化歪みを低減。
    //       リバーブテール等の微小信号の消失を防ぎ、聴感上のS/N比を改善。
    //----------------------------------------------------------
    for (int ch = 0; ch < buffer->getNumChannels(); ++ch)
    {
        if (ch < procChannels)
        {
            // double (DSP) -> float (I/O) conversion
            // ディザリング適用 (Psychoacoustic Noise Shaping)
            const double* src = processBuffer.getReadPointer(ch);
            float* dst = buffer->getWritePointer(ch, startSample);

            if (applyDither)
            {
                for (int i = 0; i < numSamples; ++i)
                    dst[i] = static_cast<float>(dither.process(src[i], ch));
            }
            else
            {
                for (int i = 0; i < numSamples; ++i)
                    dst[i] = static_cast<float>(src[i]);
            }
        }
        else
        {
            buffer->clear (ch, startSample, numSamples);
        }
    }
}

void AudioEngine::setEqBypassRequested (bool shouldBypass) noexcept
{
    eqBypassRequested.store (shouldBypass, std::memory_order_relaxed);
}

void AudioEngine::setConvolverBypassRequested (bool shouldBypass) noexcept
{
    convBypassRequested.store (shouldBypass, std::memory_order_relaxed);
}

void AudioEngine::setConvolverUseMinPhase(bool useMinPhase)
{
    uiConvolverProcessor.setUseMinPhase(useMinPhase);
}

bool AudioEngine::getConvolverUseMinPhase() const
{
    return uiConvolverProcessor.getUseMinPhase();
}

void AudioEngine::requestEqPreset (int presetIndex) noexcept
{
    uiEqProcessor.loadPreset (presetIndex);
    sendChangeMessage();
}

void AudioEngine::requestEqPresetFromText(const juce::File& file) noexcept
{
    if (uiEqProcessor.loadFromTextFile(file))
        sendChangeMessage();
}

void AudioEngine::requestConvolverPreset (const juce::File& irFile) noexcept
{
    uiConvolverProcessor.loadImpulseResponse (irFile);
}

void AudioEngine::requestLoadState (const juce::ValueTree& state)
{
    // EQ
    auto eqState = state.getChildWithName ("EQ");
    if (eqState.isValid())
        uiEqProcessor.setState (eqState);

    // Convolver
    auto convState = state.getChildWithName ("Convolver");
    if (convState.isValid())
        uiConvolverProcessor.setState (convState);
}

juce::ValueTree AudioEngine::getCurrentState() const
{
    juce::ValueTree state ("Preset");
    state.addChild (uiEqProcessor.getState(), -1, nullptr);
    state.addChild (uiConvolverProcessor.getState(), -1, nullptr);
    return state;
}

void AudioEngine::setDitherBitDepth(int bitDepth)
{
    if (ditherBitDepth.load() != bitDepth)
    {
        ditherBitDepth.store(bitDepth);
        requestRebuild(currentSampleRate.load(), maxSamplesPerBlock.load());
    }
}

int AudioEngine::getDitherBitDepth() const
{
    return ditherBitDepth.load();
}

void AudioEngine::setSoftClipEnabled(bool enabled)
{
    softClipEnabled.store(enabled, std::memory_order_relaxed);
}

bool AudioEngine::isSoftClipEnabled() const
{
    return softClipEnabled.load(std::memory_order_relaxed);
}

void AudioEngine::setSaturationAmount(float amount)
{
    saturationAmount.store(juce::jlimit(0.0f, 1.0f, amount), std::memory_order_relaxed);
}

float AudioEngine::getSaturationAmount() const
{
    return saturationAmount.load(std::memory_order_relaxed);
}

void AudioEngine::setOversamplingFactor(int factor)
{
    // 0=Auto, 1, 2, 4, 8
    int newFactor = 0;
    if (factor == 1 || factor == 2 || factor == 4 || factor == 8)
    {
        newFactor = factor;
    }

    if (manualOversamplingFactor.load() != newFactor)
    {
        manualOversamplingFactor.store(newFactor);
        requestRebuild(currentSampleRate.load(), maxSamplesPerBlock.load());
    }
}

int AudioEngine::getOversamplingFactor() const
{
    return manualOversamplingFactor.load();
}

void AudioEngine::setOversamplingType(OversamplingType type)
{
    oversamplingType.store(type);
    requestRebuild(currentSampleRate.load(), maxSamplesPerBlock.load());
}

AudioEngine::OversamplingType AudioEngine::getOversamplingType() const
{
    return oversamplingType.load();
}
