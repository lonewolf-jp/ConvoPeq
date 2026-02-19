//============================================================================
#pragma once
// AudioEngine.h  ── v0.1 (JUCE 8.0.12対応)
//
// オーディオエンジン - AudioSource実装
//
// ■ JUCE AudioSource仕様:
//   - getNextAudioBlock(...) : オーディオ処理コールバック
//   - prepareToPlay(...) : 再生準備（バッファ確保など）
//   - releaseResources() : リソース解放（再生停止時）
//
// ■ スレッド安全性とリアルタイム制約:
//   - getNextAudioBlock: Audio Thread で実行されます。
//     - リアルタイム制約があります（ブロック不可、ロック不可、メモリ割り当て不可、IR再ロード不可）。
//   - prepareToPlay / releaseResources: Audio Thread の開始前/終了後に Message Thread から呼ばれます。
//   - パラメータ設定: Message Thread から呼ばれます。std::atomic を使用して Audio Thread と安全に同期します。
//   - readFromFifo: Message Thread (Timer) から呼ばれます。Lock-free FIFO を介してデータを取得します。
//============================================================================

#include <JuceHeader.h>
#include <atomic>
#include <cstring>
#include <array>
#include <juce_dsp/juce_dsp.h>

#include "ConvolverProcessor.h"
#include "EQProcessor.h"
#include "PsychoacousticDither.h"

class AudioEngine : public juce::AudioSource,
                  public juce::ChangeBroadcaster,
                  private juce::ChangeListener,
                  private EQProcessor::Listener,
                  private ConvolverProcessor::Listener
{
public:
    using SampleType = double; // 内部DSP精度 (JUCE Best Practice)

    enum class ProcessingOrder
    {
        ConvolverThenEQ,
        EQThenConvolver
    };

    enum class AnalyzerSource
    {
        Input,
        Output
    };

    enum class OversamplingType
    {
        IIR,
        LinearPhase
    };

    class Listener
    {
    public:
        virtual ~Listener() = default;
        virtual void eqSettingsChanged() = 0;
    };

    // FIFO設定
    static constexpr int FIFO_SIZE = 32768;  // Lock-free FIFO サイズ (推奨: FFTサイズ * 8)

    // ── 安全性制限 ──
    static constexpr double SAFE_MIN_SAMPLE_RATE = 8000.0;
    static constexpr double SAFE_MAX_SAMPLE_RATE = 384000.0;
    static constexpr int    SAFE_MAX_BLOCK_SIZE  = 8192;

    //----------------------------------------------------------
    // コンストラクタ
    //----------------------------------------------------------
    AudioEngine();
    ~AudioEngine() override;
    void initialize();

    //----------------------------------------------------------
    // AudioSource インターフェース
    //----------------------------------------------------------
    void prepareToPlay (int samplesPerBlockExpected, double sampleRate) override;
    void releaseResources() override;
    void getNextAudioBlock (const juce::AudioSourceChannelInfo& bufferToFill) override;
    void changeListenerCallback(juce::ChangeBroadcaster* source) override;
    void eqBandChanged(EQProcessor* processor, int bandIndex) override;
    void eqGlobalChanged(EQProcessor* processor) override;
    void convolverParamsChanged(ConvolverProcessor* processor) override;

    //----------------------------------------------------------
    // 外部インターフェース (Message Thread)
    //----------------------------------------------------------
    ConvolverProcessor& getConvolverProcessor() { return uiConvolverProcessor; }
    EQProcessor& getEQProcessor() { return uiEqProcessor; }

    int getSampleRate() const { return currentSampleRate.load(); }

    float getInputLevel()  const { return inputLevelDb.load(); }
    float getOutputLevel() const { return outputLevelDb.load(); }

    // UIスレッドから呼び出し。FIFOからデータを取得する。
    int getFifoNumReady() const { return audioFifo.getNumReady(); }
    void readFromFifo(float* dest, int numSamples);

    void calcEQResponseCurve(float* outMagnitudesL, float* outMagnitudesR, const std::complex<double>* zArray, int numPoints);

    // パラメータ設定 (Thread-safe)
    void setEqBypassRequested (bool shouldBypass) noexcept;
    void setConvolverBypassRequested (bool shouldBypass) noexcept;

    void setConvolverUseMinPhase(bool useMinPhase);
    bool getConvolverUseMinPhase() const;

    void requestEqPreset (int presetIndex) noexcept;
    void requestEqPresetFromText(const juce::File& file) noexcept;
    void requestConvolverPreset (const juce::File& irFile) noexcept;

    void requestLoadState (const juce::ValueTree& state);
    juce::ValueTree getCurrentState() const;

    void setProcessingOrder(ProcessingOrder order) { currentProcessingOrder.store(order); }
    ProcessingOrder getProcessingOrder() const { return currentProcessingOrder.load(); }

    void setAnalyzerSource(AnalyzerSource source) { currentAnalyzerSource.store(source); }
    AnalyzerSource getAnalyzerSource() const { return currentAnalyzerSource.load(); }

    void setDitherBitDepth(int bitDepth);
    int getDitherBitDepth() const;

    void setSoftClipEnabled(bool enabled);
    bool isSoftClipEnabled() const;

    void setSaturationAmount(float amount);
    float getSaturationAmount() const;

    void setOversamplingFactor(int factor);
    int getOversamplingFactor() const;

    void setOversamplingType(OversamplingType type);
    OversamplingType getOversamplingType() const;

private:
    //==============================================================================
    // 内部クラス定義
    //----------------------------------------------------------
    // DC除去フィルタ (4次バターワースハイパスフィルタ)
    // 目的: EQやConvolver処理で発生しうるDCオフセットを除去し、スピーカー保護とヘッドルーム確保を行う。
    // 特徴: カットオフ周波数 3Hz。20Hz帯域の位相歪みを低減 (-24dB/oct)。
    // 実装: JUCE IIR Filter (2次x2段)
    //----------------------------------------------------------
    class DCBlocker
    {
    public:
        void prepare(double sampleRate) noexcept
        {
            // 4次バターワースハイパスフィルタ（3Hz、-24dB/oct）
            // 20Hz帯域の位相歪みを低減
            spec.sampleRate = sampleRate;
            spec.maximumBlockSize = SAFE_MAX_BLOCK_SIZE;
            spec.numChannels = 1;

            // 2次バターワース × 2段 = 4次フィルタ
            auto coeffs = juce::dsp::IIR::Coefficients<double>::makeHighPass(
                sampleRate,
                3.0,    // カットオフ周波数: 3Hz
                0.707   // Q値（バターワース）
            );

            for (auto& filter : filters)
            {
                filter.coefficients = coeffs;
                filter.prepare(spec);
                filter.reset();
            }
        }

        void reset() noexcept
        {
            for (auto& filter : filters)
                filter.reset();
        }

        double process(double input) noexcept
        {
            double output = input;

            // 2段縦続接続で4次特性を実現
            for (auto& filter : filters)
                output = filter.processSample(output);

            // Denormal対策
            static constexpr double DENORMAL_THRESHOLD = 1.0e-15;
            if (std::abs(output) < DENORMAL_THRESHOLD)
                output = 0.0;

            return output;
        }
    private:
        juce::dsp::ProcessSpec spec;
        std::array<juce::dsp::IIR::Filter<double>, 2> filters;
    };

    //----------------------------------------------------------
    // DSPコア (Audio Threadで実行される処理のコンテナ)
    //----------------------------------------------------------
    struct DSPCore
    {
        struct ProcessingState
        {
            bool eqBypassed;
            bool convBypassed;
            ProcessingOrder order;
            AnalyzerSource analyzerSource;
            bool softClipEnabled;
            float saturationAmount;
        };

        DSPCore();

        void prepare(double sampleRate, int samplesPerBlock, int bitDepth, int manualOversamplingFactor, OversamplingType oversamplingType);
        void reset();
        void process(const juce::AudioSourceChannelInfo& bufferToFill, juce::AbstractFifo& audioFifo,
                     juce::AudioBuffer<float>& audioFifoBuffer, std::atomic<float>& inputLevelDb,
                     std::atomic<float>& outputLevelDb, const ProcessingState& state);

        ConvolverProcessor convolver;
        EQProcessor eq;
        DCBlocker dcBlockerL, dcBlockerR;
        PsychoacousticDither dither;

        std::unique_ptr<juce::dsp::Oversampling<double>> oversampling;
        size_t oversamplingFactor = 1;
        int ditherBitDepth = 0; // DSPCore内でディザリング判定に使用

        juce::AudioBuffer<SampleType> processBuffer;
        int maxSamplesPerBlock = 0;

        // Helpers
        float measureLevel (const juce::AudioBuffer<SampleType>& buffer, int numSamples) const noexcept;
        void pushToFifo(const juce::AudioBuffer<SampleType>& buffer, int numSamples,
                        juce::AbstractFifo& audioFifo,
                        juce::AudioBuffer<float>& audioFifoBuffer) const;
        void processInput(const juce::AudioSourceChannelInfo& bufferToFill, int numSamples);
        void processOutput(const juce::AudioSourceChannelInfo& bufferToFill, int numSamples);
    private:
        static double musicalSoftClip(double x, double threshold, double knee, double asymmetry) noexcept;
    };

    //----------------------------------------------------------
    // 処理チェーンコンポーネント
    //----------------------------------------------------------
    // UI/State管理用のインスタンス (Audio Threadでは使用しない)
    ConvolverProcessor uiConvolverProcessor;
    EQProcessor uiEqProcessor;

    juce::AbstractFifo audioFifo { FIFO_SIZE };
    juce::AudioBuffer<float> audioFifoBuffer;

    //----------------------------------------------------------
    // 状態管理
    //----------------------------------------------------------
    std::atomic<std::shared_ptr<DSPCore>> currentDSP;
    std::vector<std::shared_ptr<DSPCore>> trashBin; // 古いDSPの保持用 (Audio Threadでの削除防止)
    juce::CriticalSection trashBinLock;

    std::atomic<int>   currentSampleRate{48000};
    std::atomic<float> inputLevelDb{-120.0f};
    std::atomic<float> outputLevelDb{-120.0f};
    std::atomic<int>   maxSamplesPerBlock{4096};

    std::atomic<bool> eqBypassRequested { false };
    std::atomic<bool> convBypassRequested { false };
    std::atomic<bool> eqBypassActive   { false };
    std::atomic<bool> convBypassActive { false };
    std::atomic<ProcessingOrder> currentProcessingOrder{ProcessingOrder::ConvolverThenEQ};
    std::atomic<AnalyzerSource> currentAnalyzerSource { AnalyzerSource::Output };
    std::atomic<int> ditherBitDepth { 0 }; // 0 = 未初期化 (DeviceSettingsで最大値に設定される)
    std::atomic<bool> softClipEnabled { true };
    std::atomic<float> saturationAmount { 0.5f };
    std::atomic<int> manualOversamplingFactor { 0 }; // 0=Auto, 1=1x, 2=2x, 4=4x, 8=8x
    std::atomic<OversamplingType> oversamplingType { OversamplingType::IIR };

    // dB変換時の下限値
    static constexpr float LEVEL_METER_MIN_DB = -120.0f;
    static constexpr float LEVEL_METER_MIN_MAG = 1e-6f;

    // EQ応答曲線計算用の定数
    static constexpr float EQ_GAIN_EPSILON = 0.01f;          // ゲインがこれ以下なら無視
    static constexpr float EQ_UNITY_GAIN_EPSILON = 1.0e-5f;  // 1.0との比較用

    //----------------------------------------------------------
    // ヘルパー関数
    //----------------------------------------------------------
    void requestRebuild(double sampleRate, int samplesPerBlock);
    void commitNewDSP(std::shared_ptr<DSPCore> newDSP);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AudioEngine)
};
