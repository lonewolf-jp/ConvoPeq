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

#include "ConvolverProcessor.h"
#include "EQProcessor.h"

class AudioEngine : public juce::AudioSource,
                  public juce::ChangeBroadcaster,
                  private juce::ChangeListener
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

    class Listener
    {
    public:
        virtual ~Listener() = default;
        virtual void eqSettingsChanged() = 0;
    };

    // FIFO設定
    static constexpr int FIFO_SIZE = 16384;  // Lock-free FIFO サイズ

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

private:
    //==============================================================================
    // 内部クラス定義
    //----------------------------------------------------------
    // DC除去フィルタ (1次ハイパスフィルタ)
    // 目的: EQやConvolver処理で発生しうるDCオフセットを除去し、スピーカー保護とヘッドルーム確保を行う。
    // 特徴: カットオフ周波数 ~5Hz。可聴域(20Hz~)での位相歪みは無視できるレベル。
    // 実装: シンプルな1次IIRフィルタ (y[n] = alpha * (y[n-1] + x[n] - x[n-1]))
    //----------------------------------------------------------
    class DCBlocker
    {
    public:
        void prepare(double sampleRate) noexcept
        {
            // 1次ハイパスフィルタ (カットオフ ~5Hz)
            // ref: https://www.dsprelated.com/showarticle/58.php
            static constexpr double CUTOFF_FREQ = 5.0;
            const double rc = 1.0 / (2.0 * juce::MathConstants<double>::pi * CUTOFF_FREQ);
            alpha = rc / (rc + 1.0 / sampleRate);
            reset();
        }

        void reset() noexcept { x1 = y1 = 0.0; }

        double process(double input) noexcept
        {
            // y[n] = alpha * (y[n-1] + x[n] - x[n-1])
            const double output = alpha * (y1 + input - x1);
            x1 = input;
            y1 = output;

            // Denormal対策: 状態変数が極小値になったら0にする
            static constexpr double DENORMAL_THRESHOLD = 1.0e-15;
            y1 = (std::abs(y1) < DENORMAL_THRESHOLD) ? 0.0 : y1;
            x1 = (std::abs(x1) < DENORMAL_THRESHOLD) ? 0.0 : x1;

            return y1;
        }
    private:
        double alpha = 0.0, x1 = 0.0, y1 = 0.0;
    };

    //----------------------------------------------------------
    // ディザリング (TPDF: Triangular Probability Density Function)
    // 目的: 浮動小数点数(32/64bit)から整数フォーマット(16/24bit)へ変換する際の量子化歪みを低減する。
    // 特徴: 三角分布の確率密度関数を持つノイズを加えることで、量子化誤差を信号に依存しないホワイトノイズに変調する。
    // 効果: リバーブのテールなど、微小レベルの信号の消失を防ぎ、聴感上のS/N比を改善する。
    //----------------------------------------------------------
    class TPDFDither
    {
    public:
        static constexpr int DEFAULT_BIT_DEPTH = 24;

        TPDFDither() = default;

        void setTargetBitDepth(int bits) noexcept
        {
            // LSB (Least Significant Bit) のレベル
            ditherAmount = 1.0f / static_cast<float>(std::pow(2.0f, bits - 1));
        }

        float process(float input) noexcept
        {
            // 2つの独立した乱数の差 = 三角分布 (TPDF)
            const float r1 = random.nextFloat() * 2.0f - 1.0f;
            const float r2 = random.nextFloat() * 2.0f - 1.0f;
            const float dither = (r1 - r2) * ditherAmount;
            return input + dither;
        }
    private:
        juce::Random random;
        float ditherAmount = 1.0f / 8388608.0f; // Default 24-bit
    };

    //----------------------------------------------------------
    // DSPコア (Audio Threadで実行される処理のコンテナ)
    //----------------------------------------------------------
    struct DSPCore
    {
        DSPCore();

        void prepare(double sampleRate, int samplesPerBlock);
        void process(const juce::AudioSourceChannelInfo& bufferToFill,
                     juce::AbstractFifo& audioFifo,
                     juce::AudioBuffer<float>& audioFifoBuffer,
                     std::atomic<float>& inputLevelDb,
                     std::atomic<float>& outputLevelDb,
                     bool eqBypassed,
                     bool convBypassed,
                     ProcessingOrder order,
                     AnalyzerSource analyzerSource);

        ConvolverProcessor convolver;
        EQProcessor eq;
        DCBlocker dcBlockerL, dcBlockerR;
        TPDFDither ditherL, ditherR;

        juce::AudioBuffer<SampleType> processBuffer;
        int maxSamplesPerBlock = 0;

        // Helpers
        float measureLevel (const juce::AudioBuffer<SampleType>& buffer, int numSamples) const noexcept;
        void writeSampleToFifo(float* dest, int index, const double* l, const double* r) const noexcept;
        void pushToFifo(const juce::AudioBuffer<SampleType>& buffer, int numSamples,
                        juce::AbstractFifo& audioFifo,
                        juce::AudioBuffer<float>& audioFifoBuffer) const;
        void processInput(const juce::AudioSourceChannelInfo& bufferToFill, int numSamples);
        void processOutput(const juce::AudioSourceChannelInfo& bufferToFill, int numSamples);
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

    // dB変換時の下限値
    static constexpr float LEVEL_METER_MIN_DB = -120.0f;
    static constexpr float LEVEL_METER_MIN_MAG = 1e-6f;

    // EQ応答曲線計算用の定数
    static constexpr float EQ_GAIN_EPSILON = 0.01f;          // ゲインがこれ以下なら無視
    static constexpr float EQ_UNITY_GAIN_EPSILON = 1.0e-5f;  // 1.0との比較用

    //----------------------------------------------------------
    // ヘルパー関数
    //----------------------------------------------------------
    void rebuild(double sampleRate, int samplesPerBlock);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AudioEngine)
};
