//============================================================================
#pragma once
// AudioEngine.h  ── v0.2 (JUCE 8.0.12対応)
//
// オーディオエンジン - AudioSource実装
//
// ■ JUCE AudioSource仕様:
//   - getNextAudioBlock(...) : オーディオ処理コールバック
//   - prepareToPlay(...) : 再生準備（バッファ確保など）
//   - releaseResources() : リソース解放（再生停止時）
//
// ■ スレッド安全性とリアルタイム制約:
//   - getNextAudioBlock: Audio Threadで実行されます。
//     - リアルタイム制約があります（ブロック不可、ロック不可、メモリ割り当て不可、IR再ロード不可）。
//   - prepareToPlay / releaseResources: Audio Thread の開始前/終了後に Message Thread から呼ばれます。
//   - パラメータ設定: Message Thread から呼ばれます。std::atomic を使用して Audio Thread と安全に同期します (RCUパターン)。
//   - readFromFifo: Message Thread (Timer) から呼ばれます。FIFOバッファからデータを取得します。
//============================================================================

#include <JuceHeader.h>
#include <atomic>
#include <cstring>
#include <array>
#include <vector>
#include <juce_dsp/juce_dsp.h>

#include "AlignedAllocation.h"
#include "ConvolverProcessor.h"
#include "EQProcessor.h"
#include "PsychoacousticDither.h"

class AudioEngine : public juce::AudioSource,
                  public juce::ChangeBroadcaster,
                  private juce::ChangeListener,
                  private EQProcessor::Listener,
                  private ConvolverProcessor::Listener,
                   private juce::Timer
{
public:
    using SampleType = double; // 内部DSP精度 (JUCE推奨)

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
    static constexpr int    SAFE_MAX_BLOCK_SIZE  = 65536; // 8x Oversampling対応のため拡張

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
    void timerCallback() override;

    //----------------------------------------------------------
    // 外部インターフェース (Message Thread)
    //----------------------------------------------------------
    ConvolverProcessor& getConvolverProcessor() { return uiConvolverProcessor; }
    EQProcessor& getEQProcessor() { return uiEqProcessor; }

    double getSampleRate() const { return currentSampleRate.load(); }
    double getProcessingSampleRate() const;

    float getInputLevel()  const { return inputLevelDb.load(); }
    float getOutputLevel() const { return outputLevelDb.load(); }




    int getFifoNumReady() const { return audioFifo.getNumReady(); }
    void readFromFifo(float* dest, int numSamples);

    void calcEQResponseCurve(float* outMagnitudesL, float* outMagnitudesR, const std::complex<double>* zArray, int numPoints, double sampleRate);

    // パラメータ設定 (Thread-safe)
    void setEqBypassRequested (bool shouldBypass) noexcept;
    void setConvolverBypassRequested (bool shouldBypass) noexcept;

    void setConvolverUseMinPhase(bool useMinPhase);
    bool getConvolverUseMinPhase() const;

    void requestEqPreset (int presetIndex);
    void requestEqPresetFromText(const juce::File& file);
    void requestConvolverPreset (const juce::File& irFile);

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
        void prepare(double sampleRate, int blockSize) noexcept
        {
            // 4次バターワースハイパスフィルタ（3Hz、-24dB/oct）
            // 20Hz帯域の位相歪みを低減
            spec.sampleRate = sampleRate;
            spec.maximumBlockSize = static_cast<juce::uint32>(blockSize);
            spec.numChannels = 1;

            // 4次 Linkwitz-Riley ハイパスフィルタ (2次バターワース x 2段)
            // 特徴:
            // 1. カットオフ周波数(3Hz)で-6dBの減衰 (Butterworthは-3dB)
            // 2. ストップバンド(DC付近)での減衰量がButterworthより大きく、DCブロック性能が高い
            // 3. Q=0.707の段を重ねるため、過渡応答のリンギングが少ない
            auto coeffs = juce::dsp::IIR::Coefficients<double>::makeHighPass(
                sampleRate,
                3.0,    // カットオフ周波数: 3Hz
                0.7071067811865476 // Q = 1/sqrt(2)
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
    // 高精度型 DC Blocker (1次IIR)
    // 超高サンプリングレート（OSR）対応
    //----------------------------------------------------------
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
            reset();
        }

        void reset() {
            m_prev_x = 0.0;
            m_prev_y = 0.0;
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

            if (std::abs(curr_y) < 1.0e-20) curr_y = 0.0; // Anti-Denormal Trick
                data[i] = curr_y;
            }
            m_prev_x = px;
             m_prev_y = py;
        }
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

        using Ptr = std::shared_ptr<DSPCore>;

        DSPCore();

        void prepare(double sampleRate, int samplesPerBlock, int bitDepth, int manualOversamplingFactor, OversamplingType oversamplingType);
        void reset();
        void process(const juce::AudioSourceChannelInfo& bufferToFill, juce::AbstractFifo& audioFifo,
                     juce::AudioBuffer<float>& audioFifoBuffer, std::atomic<float>& inputLevelDb,
                     std::atomic<float>& outputLevelDb, const ProcessingState& state);

        ConvolverProcessor convolver;
        EQProcessor eq;
        DCBlocker dcBlockerL, dcBlockerR;
        DCBlocker inputDCBlockerL, inputDCBlockerR;
         UltraHighRateDCBlocker osDCBlockerL, osDCBlockerR; // Oversampling後のDC除去用
        ::convo::PsychoacousticDither dither;

        std::unique_ptr<juce::dsp::Oversampling<double>> oversampling;
        size_t oversamplingFactor = 1;
        int ditherBitDepth = 0; // DSPCore内でディザリング判定に使用
        double sampleRate = 0.0;

        std::vector<double, convo::MKLAllocator<double>> alignedL, alignedR; // Aligned buffers for processing
        int maxSamplesPerBlock = 0;               // 入力側最大ブロックサイズ (SAFE_MAX_BLOCK_SIZE)

        // ─────────────────────────────────────────────────────────────
        // 【Issue 3 修正】内部処理用最大バッファサイズ
        // 理由: Oversampling有効時（最大8x）、processSamplesUp()後の
        //      ブロックサイズがSAFE_MAX×8になるため。
        //      固定で×8確保することでRCU再構築時のresizeを完全排除。
        //      メモリ増加 ≈ 8.4MB（現代PCでは無視できるレベル）
        // ─────────────────────────────────────────────────────────────
        int maxInternalBlockSize = 0;             // OS考慮後の最大サイズ（常にSAFE_MAX×8）

        // ==================================================================
        // 【Issue 5 修正】新DSP切り替え時のFade-in Ramp
        // 用途: 新DSPCoreがゼロスタートしても出力が0→1に滑らかに立ち上がる
        // 効果: 短い無音/グリッチを完全に滑らかに解消（20〜42msランプ）
        // Audio Thread内完全lock-free・no-alloc
        // ==================================================================
        std::atomic<int> fadeInSamplesLeft {0};
        static constexpr int FADE_IN_SAMPLES = 2048; // 42ms @ 48kHz

        // Helpers
        float measureLevel (const juce::dsp::AudioBlock<const double>& block) const noexcept;
        void pushToFifo(const juce::dsp::AudioBlock<const double>& block,
                        juce::AbstractFifo& audioFifo,
                        juce::AudioBuffer<float>& audioFifoBuffer) const noexcept;
        void processInput(const juce::AudioSourceChannelInfo& bufferToFill, int numSamples) noexcept;
        void processOutput(const juce::AudioSourceChannelInfo& bufferToFill, int numSamples) noexcept;
    private:
        static double musicalSoftClip(double x, double threshold, double knee, double asymmetry) noexcept;
    };

    //----------------------------------------------------------
    // 処理チェーンコンポーネント
    //----------------------------------------------------------
     // UI/State管理用のインスタンス (Audio Threadでは使用しない)
    ConvolverProcessor  uiConvolverProcessor;
    EQProcessor uiEqProcessor;

    juce::AbstractFifo audioFifo { FIFO_SIZE };
    juce::AudioBuffer<float> audioFifoBuffer;
    juce::CriticalSection fifoReadLock;

    //----------------------------------------------------------
    // 状態管理
    //----------------------------------------------------------
    std::atomic<DSPCore*> currentDSP { nullptr }; // Raw pointer for Audio Thread (Lock-free)
    DSPCore::Ptr activeDSP; // Ownership holder for Message Thread
    std::vector<std::pair<DSPCore::Ptr, uint32>> trashBin; // Time-based garbage collection
    std::vector<DSPCore::Ptr> trashBinPending; // 新しく追加されたゴミ (次回のタイマーコールバックまで保持)
    juce::CriticalSection trashBinLock;

    std::atomic<double> currentSampleRate{48000.0};
    std::atomic<float> inputLevelDb{-120.0f};
    std::atomic<float> outputLevelDb{-120.0f};
    std::atomic<int>   maxSamplesPerBlock{4096};

    std::atomic<bool> eqBypassRequested { false };
    std::atomic<bool> convBypassRequested { false };
    std::atomic<bool> eqBypassActive   { false };
    std::atomic<bool> convBypassActive { false };
    std::atomic<bool> rebuildRequested { false };
    std::atomic<ProcessingOrder> currentProcessingOrder{ProcessingOrder::ConvolverThenEQ};
    std::atomic<AnalyzerSource> currentAnalyzerSource { AnalyzerSource::Output };
    std::atomic<int> ditherBitDepth { 0 }; // 0 = 未初期化 (DeviceSettingsで最大値に設定される)
    std::atomic<bool> softClipEnabled { true };
    std::atomic<float> saturationAmount { 0.5f };
    std::atomic<int> manualOversamplingFactor { 0 }; // 0=Auto, 1=1x, 2=2x, 4=4x, 8=8x
    std::atomic<OversamplingType> oversamplingType { OversamplingType::IIR };

    // dB変換時の下限値
    static constexpr float LEVEL_METER_MIN_DB  = -120.0f;
    static constexpr float LEVEL_METER_MIN_MAG = 1e-6f;

    // EQ応答曲線計算用の定数
    static constexpr float EQ_GAIN_EPSILON = 0.01f;          // ゲインがこれ以下なら無視
    static constexpr float EQ_UNITY_GAIN_EPSILON = 1.0e-5f;  // 1.0との比較用

    //----------------------------------------------------------
    // ヘルパー関数
    //----------------------------------------------------------
    // Note: This function performs memory allocation (including MKL) and other blocking operations
    // such as IR resampling. It MUST only be called from the message thread.
    // The prepareToPlay() method ensures this by using MessageManager::callAsync if necessary.
    void requestRebuild(double sampleRate, int samplesPerBlock);
    void commitNewDSP(DSPCore::Ptr newDSP);

    JUCE_DECLARE_WEAK_REFERENCEABLE(AudioEngine)
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AudioEngine)
};
