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
    audioFifoBuffer.setSize (1, FIFO_SIZE);
    audioFifoBuffer.clear();
}

void AudioEngine::initialize()
{
    // 初期DSP構築 (デフォルト設定)
    // 安全対策: バッファサイズを余裕を持って確保 (8192)
    // これにより、デバイス初期化前やバッファサイズ変更時の不整合による音切れ/無音を防ぐ
    rebuild(48000.0, SAFE_MAX_BLOCK_SIZE);
    maxSamplesPerBlock.store(SAFE_MAX_BLOCK_SIZE);
    currentSampleRate.store(48000);

    uiConvolverProcessor.addChangeListener(this);
    uiEqProcessor.addChangeListener(this);
}

AudioEngine::~AudioEngine()
{
    uiConvolverProcessor.removeChangeListener(this);
    uiEqProcessor.removeChangeListener(this);

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
        std::memcpy(dest,
                   audioFifoBuffer.getReadPointer(0, start1),
                   size1 * sizeof(float));
    }

    if (size2 > 0)
    {
        std::memcpy(dest + size1,
                   audioFifoBuffer.getReadPointer(0, start2),
                   size2 * sizeof(float));
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

    for (int band = 0; band < EQProcessor::NUM_BANDS; ++band)
    {
        auto params = uiEqProcessor.getBandParams(band);
        if (!params.enabled) continue;

        EQBandType type = uiEqProcessor.getBandType(band);

        // LowPass/HighPass以外でゲインがほぼ0の場合はスキップ
        if (type != EQBandType::LowPass && type != EQBandType::HighPass &&
            std::abs(params.gain) < EQ_GAIN_EPSILON)
            continue;

        activeBands[numActiveBands++] = {
            EQProcessor::calcBiquadCoeffs(type, params.frequency, params.gain, params.q, sr),
            uiEqProcessor.getBandChannelMode(band)
        };
    }

    float totalGainLinear = 1.0f;
    if (!uiEqProcessor.getAGCEnabled())
    {
        totalGainLinear = juce::Decibels::decibelsToGain(uiEqProcessor.getTotalGain());
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

            if (band.mode == EQChannelMode::Stereo || band.mode == EQChannelMode::Left)
                totalMagSqL *= magSq;
            if (band.mode == EQChannelMode::Stereo || band.mode == EQChannelMode::Right)
                totalMagSqR *= magSq;
        }

        if (outMagnitudesL) outMagnitudesL[i] = std::sqrt(totalMagSqL);
        if (outMagnitudesR) outMagnitudesR[i] = std::sqrt(totalMagSqR);
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

    // 安全対策: バッファサイズを余裕を持って確保
    // ホストが通知より大きなバッファを渡してくる場合に備え、最低でも SAFE_MAX_BLOCK_SIZE を確保する
    const int safeBufferSize = std::max(samplesPerBlockExpected, SAFE_MAX_BLOCK_SIZE);

    // UI用プロセッサのサンプルレートも更新 (IR表示やパラメータ管理のため)
    uiConvolverProcessor.prepareToPlay(sampleRate, safeBufferSize);
    uiEqProcessor.prepareToPlay(static_cast<int>(sampleRate), safeBufferSize);

    maxSamplesPerBlock.store(safeBufferSize);
    // DSP再構築 (RT安全化: 新しいDSPを作成してスワップ)
    rebuild(sampleRate, safeBufferSize);

    currentSampleRate.store(static_cast<int>(sampleRate));
    uiConvolverProcessor.setBypass(convBypassActive.load (std::memory_order_relaxed));
    audioFifo.reset();

    // ===== bypass 状態の初期化 =====
    // 再生中のリアルタイムな更新は getNextAudioBlock() で行われる
    eqBypassActive.store (eqBypassRequested.load (std::memory_order_relaxed), std::memory_order_relaxed);
    convBypassActive.store (convBypassRequested.load (std::memory_order_relaxed), std::memory_order_relaxed);

    // ConvolverProcessorの状態も同期させておく（念のため）
    uiConvolverProcessor.setBypass(convBypassActive.load (std::memory_order_relaxed));
}

//--------------------------------------------------------------
// rebuild - DSPグラフの再構築 (Message Thread)
//--------------------------------------------------------------
void AudioEngine::rebuild(double sampleRate, int samplesPerBlock)
{
    // 新しいDSPコアを作成
    auto newDSP = std::make_shared<DSPCore>();

    // UIプロセッサから状態をコピー
    newDSP->eq.setState(uiEqProcessor.getState());
    newDSP->convolver.syncStateFrom(uiConvolverProcessor);

    // 準備
    newDSP->prepare(sampleRate, samplesPerBlock);

    // Atomic Swap
    auto oldDSP = currentDSP.exchange(newDSP);

    if (oldDSP)
    {
        // 古いDSPをゴミ箱へ (Audio Threadが使用中の可能性があるため即削除しない)
        const juce::ScopedLock sl(trashBinLock);
        trashBin.push_back(oldDSP);

        // ゴミ箱の掃除 (古いものから削除してメモリ肥大化を防止)
        if (trashBin.size() > 5)
            trashBin.erase(trashBin.begin());
    }
}

void AudioEngine::changeListenerCallback(juce::ChangeBroadcaster* source)
{
    // UIプロセッサからの変更を現在のDSPに反映
    if (source == &uiEqProcessor)
    {
        // EQの場合は構造変更の可能性があるため、安全にリビルドを行う (RCUパターン)
        rebuild(currentSampleRate.load(), maxSamplesPerBlock.load());
        // DSPが有効な場合のみ変更通知を送る (デストラクタ競合回避)
        sendChangeMessage();
    }
    else if (source == &uiConvolverProcessor)
    {
        // ✅ 修正: EQと同様にrebuildを使用
        rebuild(currentSampleRate.load(), maxSamplesPerBlock.load());
        sendChangeMessage();
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

        // UI表示用の状態更新
        eqBypassActive.store(eqBypassed, std::memory_order_relaxed);
        convBypassActive.store(convBypassed, std::memory_order_relaxed);

        // 処理委譲
        dsp->process(bufferToFill, audioFifo, audioFifoBuffer, inputLevelDb, outputLevelDb, eqBypassed, convBypassed, order, analyzerSource);
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

void AudioEngine::DSPCore::prepare(double sampleRate, int samplesPerBlock)
{
    maxSamplesPerBlock = samplesPerBlock;

    // プロセッサの準備
    convolver.prepareToPlay(sampleRate, samplesPerBlock);
    eq.prepareToPlay(static_cast<int>(sampleRate), samplesPerBlock);

    dcBlockerL.prepare(sampleRate);
    dcBlockerR.prepare(sampleRate);

    // バッファ確保 (Message Threadで実行されるため安全)
    processBuffer.setSize(2, samplesPerBlock);
}

void AudioEngine::DSPCore::process(const juce::AudioSourceChannelInfo& bufferToFill,
                                  juce::AbstractFifo& audioFifo,
                                  juce::AudioBuffer<float>& audioFifoBuffer,
                                  std::atomic<float>& inputLevelDb,
                                  std::atomic<float>& outputLevelDb,
                                  bool eqBypassed,
                                  bool convBypassed,
                                  ProcessingOrder order,
                                  AnalyzerSource analyzerSource)
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
    inputLevelDb.store(inputDb);

    // ── Analyzer Input Tap (Pre-DSP) ──
    if (analyzerSource == AnalyzerSource::Input)
    {
        pushToFifo(processBuffer, numSamples, audioFifo, audioFifoBuffer);
    }

    //----------------------------------------------------------
    // DSP処理チェーン (Dynamic Processing Order)
    //----------------------------------------------------------
    if (order == ProcessingOrder::ConvolverThenEQ)
    {
        // 1. Convolver
        if (!convBypassed)
            convolver.process(processBuffer, numSamples);
        // 2. EQ
        if (!eqBypassed)
            eq.process(processBuffer, numSamples);
    }
    else
    {
        // 1. EQ
        if (!eqBypassed)
            eq.process(processBuffer, numSamples);
        // 2. Convolver
        if (!convBypassed)
            convolver.process(processBuffer, numSamples);
    }

    //----------------------------------------------------------
    // DCオフセット除去 (DC Offset Removal)
    // 目的: フィルタ処理後のDC成分を除去し、スピーカー保護とヘッドルーム確保を行う。
    // 配置: 最終段で行うことで、レベルメーターの正確性を担保する。
    //----------------------------------------------------------
    const int procChannels = processBuffer.getNumChannels();
    for (int ch = 0; ch < procChannels; ++ch)
    {
        if (ch > 1) break; // DCBlockerは現在ステレオ(L/R)のみ対応

        auto* data = processBuffer.getWritePointer(ch);
        auto& blocker = (ch == 0) ? dcBlockerL : dcBlockerR;

        for (int i = 0; i < numSamples; ++i)
            data[i] = blocker.process(data[i]);
    }

    //----------------------------------------------------------
    // 出力レベル計算 (DC除去後のクリーンな信号で計測)
    //----------------------------------------------------------
    const float outputDb = measureLevel(processBuffer, numSamples);
    outputLevelDb.store(outputDb);

    // ── Analyzer Output Tap (Post-DSP) ──
    if (analyzerSource == AnalyzerSource::Output)
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

inline void AudioEngine::DSPCore::writeSampleToFifo(float* dest, int index, const double* l, const double* r) const noexcept
{
    // Mono mix for spectrum analyzer
    float val = static_cast<float>(l[index]);
    if (r != nullptr) val = (val + static_cast<float>(r[index])) * 0.5f;
    dest[index] = val;
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
            float* dest = audioFifoBuffer.getWritePointer(0, start1);
            for (int i = 0; i < size1; ++i)
                writeSampleToFifo(dest, i, l, r);
            l += size1;
            if (r != nullptr) r += size1;
        }

        // 第2セグメント書き込み
        if (size2 > 0)
        {
            float* dest = audioFifoBuffer.getWritePointer(0, start2);
            for (int i = 0; i < size2; ++i)
                writeSampleToFifo(dest, i, l, r);
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
    for (int ch = 0; ch < procChannels; ++ch)
    {
        if (ch < effectiveInputChannels)
        {
            // float (I/O) -> double (DSP) conversion
            const float* src = buffer->getReadPointer(ch, startSample);
            double* dst = processBuffer.getWritePointer(ch);
            for (int i = 0; i < numSamples; ++i)
                dst[i] = static_cast<double>(src[i]);
        }
        else
        {
            // 入力がないチャンネル、または余剰チャンネルはクリア
            processBuffer.clear(ch, 0, numSamples);
        }
    }

    // ── Mono -> Stereo 展開 ──
    // 入力が1chのみで、処理バッファが2ch以上ある場合、LchをRchにコピーする
    // これにより、モノラルマイク入力時などでもステレオ処理として扱えるようにし、
    // 後段のステレオエフェクト（Convolver等）での片側無音を防ぐ。
    if (effectiveInputChannels == 1 && procChannels > 1)
    {
        processBuffer.copyFrom(1, 0, processBuffer, 0, 0, numSamples);
    }
}

void AudioEngine::DSPCore::processOutput(const juce::AudioSourceChannelInfo& bufferToFill, int numSamples)
{
    auto* buffer = bufferToFill.buffer;
    const int startSample = bufferToFill.startSample;
    const int procChannels = processBuffer.getNumChannels();

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
            // ディザリング適用 (TPDF)
            const double* src = processBuffer.getReadPointer(ch);
            float* dst = buffer->getWritePointer(ch, startSample);

            // ソフトクリッピング (tanh) による安全リミッターと、24bitターゲットのTPDFディザを適用
            // 過大入力時のデジタルクリップを防ぎつつ、量子化ノイズを低減してS/N比を改善する
            if (ch == 0 || ch == 1)
            {
                auto& dither = (ch == 0) ? ditherL : ditherR;
                for (int i = 0; i < numSamples; ++i)
                {
                    float sample = static_cast<float>(src[i]);
                    // ✅ 閾値ベースのソフトクリッピング (Option 1)
                    if (std::abs(sample) > 0.8f)
                        sample = std::tanh(sample);
                    dst[i] = dither.process(sample);
                }
            }
            else
            {
                // その他のチャンネルは単純変換
                for (int i = 0; i < numSamples; ++i)
                {
                    float sample = static_cast<float>(src[i]);
                    if (std::abs(sample) > 0.8f)
                        sample = std::tanh(sample);
                    dst[i] = sample;
                }
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
