
//============================================================================
#pragma once
//============================================================================

static constexpr int kAdaptiveNoiseShaperOrder = 9;
static constexpr int kAdaptiveNoiseShaperSampleRateBankCount = 10;

// BitDepth も一緒に管理するための拡張（16/24/32 の3段階）
static constexpr int kAdaptiveBitDepthCount = 3;
static constexpr int kAdaptiveBitDepthValues[kAdaptiveBitDepthCount] = {16, 24, 32};
static constexpr int kLearningModeCount = 6;

// ストリーミング信号キャプチャ用 AudioBlock（2ch, 256サンプル）
struct AudioBlock {
    double L[256];
    double R[256];
    int numSamples = 0;
    int sampleRateHz = 0;
    int bitDepth = 0;
    int adaptiveCoeffBankIndex = 0;
};

// RT/Worker間ダブルバッファ係数連携（RCU）
struct CoeffSet {
    static constexpr int kDim = kAdaptiveNoiseShaperOrder;
    double k[CoeffSet::kDim] = {};
};

#include <JuceHeader.h>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <array>
#include <functional>
#include <vector>
#include <juce_dsp/juce_dsp.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <immintrin.h>

#include "AlignedAllocation.h"
#include "CustomInputOversampler.h"
#include "ConvolverProcessor.h"
#include "EQProcessor.h"
#include "PsychoacousticDither.h"
#include "FixedNoiseShaper.h"
#include "LatticeNoiseShaper.h"
#include "OutputFilter.h"
#include "DspNumericPolicy.h"
#include "UltraHighRateDCBlocker.h"
#include "LockFreeRingBuffer.h"
#include "NoiseShaperLearner.h"

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

    enum class NoiseShaperType
    {
        Psychoacoustic = 0,
        Fixed4Tap = 1,
        Adaptive9thOrder = 2
    };

    class Listener
    {
     public:
         virtual ~Listener() = default;
        virtual void eqSettingsChanged() = 0;
    };

    // FIFO設定
    static constexpr int FIFO_SIZE = 1048576;  // Lock-free FIFO サイズ (2^20, SAFE_MAX_BLOCK_SIZE * 8x OS をカバー)

    // ── 安全性制限 ──
    static constexpr double SAFE_MIN_SAMPLE_RATE = 8000.0;
    static constexpr double SAFE_MAX_SAMPLE_RATE = 768000.0;
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
    void processBlockDouble (juce::AudioBuffer<double>& buffer);
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
    struct LatencyBreakdown
    {
        int oversamplingLatencyBaseRateSamples = 0;
        int convolverAlgorithmLatencyBaseRateSamples = 0;
        int convolverIRPeakLatencyBaseRateSamples = 0;
        int convolverTotalLatencyBaseRateSamples = 0;
        int totalLatencyBaseRateSamples = 0;
    };

    LatencyBreakdown getCurrentLatencyBreakdown() const;
    int getCurrentLatencySamples() const;
    double getCurrentLatencyMs() const;

    // 【Fix Bug #8】gainToDecibels (std::log10 / libm) を Audio Thread から排除。
    // Audio Thread は linear gain を inputLevelLinear / outputLevelLinear に格納し、
    // getter (UI Thread) で dB 変換する。
    float getInputLevel() const
    {
        const float linear = inputLevelLinear.load(std::memory_order_relaxed);
        return (linear > LEVEL_METER_MIN_MAG)
               ? juce::Decibels::gainToDecibels(linear)
               : LEVEL_METER_MIN_DB;
    }
    float getOutputLevel() const
    {
        const float linear = outputLevelLinear.load(std::memory_order_relaxed);
        return (linear > LEVEL_METER_MIN_MAG)
               ? juce::Decibels::gainToDecibels(linear)
               : LEVEL_METER_MIN_DB;
    }




    int getFifoNumReady() const { return audioFifo.getNumReady(); }
    void readFromFifo(float* dest, int numSamples);
    void skipFifo(int numSamples);

    void calcEQResponseCurve(float* outMagnitudesL, float* outMagnitudesR, const std::complex<double>* zArray, int numPoints, double sampleRate);

    // パラメータ設定 (Thread-safe)
    void setEqBypassRequested (bool shouldBypass);
    void setConvolverBypassRequested (bool shouldBypass);
    bool isEqBypassRequested() const noexcept { return eqBypassRequested.load(std::memory_order_relaxed); }
    bool isConvolverBypassRequested() const noexcept { return convBypassRequested.load(std::memory_order_relaxed); }

    void setConvolverPhaseMode(ConvolverProcessor::PhaseMode mode);
    ConvolverProcessor::PhaseMode getConvolverPhaseMode() const;

    void setConvolverUseMinPhase(bool useMinPhase);
    bool getConvolverUseMinPhase() const;

    void requestEqPreset (int presetIndex);
    void requestEqPresetFromText(const juce::File& file);
    void requestConvolverPreset (const juce::File& irFile);

    void requestLoadState (const juce::ValueTree& state);
    juce::ValueTree getCurrentState() const;

    void setProcessingOrder(ProcessingOrder order);
    ProcessingOrder getProcessingOrder() const { return currentProcessingOrder.load(); }

    void setAnalyzerSource(AnalyzerSource source) { currentAnalyzerSource.store(source); }
    AnalyzerSource getAnalyzerSource() const { return currentAnalyzerSource.load(); }
    void setAnalyzerEnabled(bool enabled) noexcept { analyzerEnabled.store(enabled, std::memory_order_release); }
    bool isAnalyzerEnabled() const noexcept { return analyzerEnabled.load(std::memory_order_acquire); }

    void setInputHeadroomDb(float db);
    float getInputHeadroomDb() const;

    void setOutputMakeupDb(float db);
    float getOutputMakeupDb() const;

    void setConvolverInputTrimDb(float db);
    float getConvolverInputTrimDb() const;

    void setDitherBitDepth(int bitDepth);
    int getDitherBitDepth() const;

    void setNoiseShaperType(NoiseShaperType type);
    NoiseShaperType getNoiseShaperType() const;
    void setFixedNoiseLogIntervalMs(int intervalMs) noexcept;
    int getFixedNoiseLogIntervalMs() const noexcept;
    void setFixedNoiseWindowSamples(int windowSamples) noexcept;
    int getFixedNoiseWindowSamples() const noexcept;

    void setSoftClipEnabled(bool enabled);
    bool isSoftClipEnabled() const;

    void setSaturationAmount(float amount);
    float getSaturationAmount() const;

    void setOversamplingFactor(int factor);
    int getOversamplingFactor() const;

    void setOversamplingType(OversamplingType type);
    OversamplingType getOversamplingType() const;

    // ────────────────────────────────────────────────────────────────
    // 出力周波数フィルター設定 (Thread-safe)
    //
    // convHCMode / convLCMode: ① コンボルバー最終段の場合に使用
    // eqLPFMode              : ② EQ最終段の場合に使用
    // ────────────────────────────────────────────────────────────────
    void setConvHCFilterMode(convo::HCMode mode) noexcept;
    convo::HCMode getConvHCFilterMode() const noexcept;

    void setConvLCFilterMode(convo::LCMode mode) noexcept;
    convo::LCMode getConvLCFilterMode() const noexcept;

    void setEqLPFFilterMode(convo::HCMode mode) noexcept;
    convo::HCMode getEqLPFFilterMode() const noexcept;

    // --- Adaptiveノイズシェイパー学習サポート ---
    void startNoiseShaperLearning(NoiseShaperLearner::LearningMode mode, bool resume = false);
    void stopNoiseShaperLearning();
    void setNoiseShaperLearningMode(NoiseShaperLearner::LearningMode mode);
    bool isNoiseShaperLearning() const;
    const NoiseShaperLearner::Progress& getNoiseShaperLearningProgress() const;
    int copyNoiseShaperLearningHistory(float* outScores, int maxPoints) const noexcept;
    // 学習ワーカーが記録したエラーメッセージを返す（UI 表示用）。エラーなしは nullptr。
    const char* getNoiseShaperLearningError() const noexcept;
    static int getAdaptiveSampleRateBankCount() noexcept;
    static double getAdaptiveSampleRateBankHz(int bankIndex) noexcept;
    void getCurrentAdaptiveCoefficients(double* outCoeffs, int maxCoefficients) const noexcept;
    void setCurrentAdaptiveCoefficients(const double* coeffs, int numCoefficients);
    void getAdaptiveCoefficientsForSampleRate(double sampleRate, double* outCoeffs, int maxCoefficients) const noexcept;
    void setAdaptiveCoefficientsForSampleRate(double sampleRate, const double* coeffs, int numCoefficients);
    void getAdaptiveCoefficientsForSampleRateAndBitDepth(double sampleRate, int bitDepth, double* outCoeffs, int maxCoefficients) const noexcept;
    void setAdaptiveCoefficientsForSampleRateAndBitDepth(double sampleRate, int bitDepth, const double* coeffs, int numCoefficients);
    void setAdaptiveAutosaveCallback(std::function<void()> callback);
    void requestAdaptiveAutosave();
    // NoiseShaperLearner から学習済み係数を受け取るコールバック (Worker Thread)
    void publishCoeffs(const double* coeffs);

    // --- Adaptive ノイズシェイパー係数インデックス計算（UI スレッドからアクセス可能） ---
    static int getAdaptiveCoeffBankIndex(double sampleRate, int bitDepth, NoiseShaperLearner::LearningMode mode) noexcept;

    bool getAdaptiveNoiseShaperState(int bankIndex, NoiseShaperLearner::State& outState) const noexcept;
    void setAdaptiveNoiseShaperState(int bankIndex, const NoiseShaperLearner::State& inState) noexcept;

private:
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
            bool analyzerEnabled;
            bool softClipEnabled;
            float saturationAmount;
            double inputHeadroomGain;
            double outputMakeupGain;
            double convolverInputTrimGain; // EQThenConvolver 時のコンボルバー入力トリム
            // 出力周波数フィルターモード
            convo::HCMode convHCMode;  // ① ハイカットモード
            convo::LCMode convLCMode;  // ① ローカットモード
            convo::HCMode eqLPFMode;   // ② EQローパスモード
            int adaptiveCoeffBankIndex;
            const CoeffSet* adaptiveCoeffSet;
            uint32_t adaptiveCoeffGeneration;
            int adaptiveCaptureSampleRateHz;
            int adaptiveCaptureBitDepth;
            LockFreeRingBuffer<AudioBlock, 4096>* adaptiveCaptureQueue;
        };

DSPCore();
        DSPCore(const DSPCore&) = delete;
        DSPCore& operator=(const DSPCore&) = delete;

    ~DSPCore()
    {
        // Explicitly clean up convolver resources to ensure no WDL memory is leaked,
        // especially for instances that are destroyed from the trash bin.
        convolver.forceCleanup();
    }

    void prepare(double sampleRate, int samplesPerBlock, int bitDepth, int manualOversamplingFactor, OversamplingType oversamplingType, NoiseShaperType selectedNoiseShaperType);
    void reset();
    void process(const juce::AudioSourceChannelInfo& bufferToFill, juce::AbstractFifo& audioFifo,
                 juce::AudioBuffer<float>& audioFifoBuffer, std::atomic<float>& inputLevelLinear,
                 std::atomic<float>& outputLevelLinear, const ProcessingState& state);
    void processDouble(juce::AudioBuffer<double>& buffer,
                       juce::AbstractFifo& audioFifo,
                       juce::AudioBuffer<float>& audioFifoBuffer,
                       std::atomic<float>& inputLevelLinear,
                       std::atomic<float>& outputLevelLinear,
                       const ProcessingState& state);
        ConvolverProcessor convolver;
        EQProcessor eq;
        // 【最適化】出力 / 入力 DC 除去を UltraHighRateDCBlocker (1次IIR, ブロックモード) に統一。
        // 旧 DCBlocker (4次 Butterworth, サンプル単位) は 1 サンプルあたり ~20 演算を要したが、
        // 1次 IIR は ~4 演算で済みかつ process(data, N) ブロック呼び出しによりメモリアクセスも効率化。
        // DC 除去の目的 (3Hz 以下のカット) には 1 次で十分。
        convo::UltraHighRateDCBlocker dcBlockerL, dcBlockerR;
        convo::UltraHighRateDCBlocker inputDCBlockerL, inputDCBlockerR;
        convo::UltraHighRateDCBlocker osDCBlockerL, osDCBlockerR; // Oversampling後のDC除去用
        ::convo::PsychoacousticDither dither;
        ::convo::FixedNoiseShaper fixedNoiseShaper;
        LatticeNoiseShaper adaptiveNoiseShaper;
        // 出力周波数フィルター (① ハイカット/ローカット / ② ローパス/ハイパス)
        convo::OutputFilter outputFilter;

        CustomInputOversampler oversampling;
        size_t oversamplingFactor = 1;
        int ditherBitDepth = 0; // DSPCore内でディザリング判定に使用
        NoiseShaperType noiseShaperType = NoiseShaperType::Psychoacoustic;
        uint32_t activeAdaptiveCoeffGeneration = 0;
        int activeAdaptiveCoeffBankIndex = -1;
        double sampleRate = 0.0;

    // 【パッチ3】MKL用rawアライメントバッファ（vector完全排除・ガイドライン厳守）
        convo::ScopedAlignedPtr<double> alignedL;
        convo::ScopedAlignedPtr<double> alignedR;
        int alignedCapacity = 0;                  // 現在確保済み容量（再確保判定用）

        int maxSamplesPerBlock = 0;               // 入力側最大ブロックサイズ (SAFE_MAX_BLOCK_SIZE)

        // ─────────────────────────────────────────────────────────────
        // 【Issue 3 修正】内部処理用最大バッファサイズ
        // 理由: Oversampling有効時（最大8x）、processSamplesUp()後の
        //      ブロックサイズがSAFE_MAX×8になるため。
        //      固定で×8確保することでRCU再構築時のresizeを完全排除。
        //      メモリ増加 ≈ 8.4MB（現代PCでは無視できるレベル）
        // ─────────────────────────────────────────────────────────────
        int maxInternalBlockSize = 0;             // OS考慮後の最大サイズ（常にSAFE_MAX×8）
        std::atomic<int> fadeInSamplesLeft {0};
        static constexpr int FADE_IN_SAMPLES = 2048; // 42ms @ 48kHz

        // インターサンプルピーク近似: 前ブロック末尾のクリップ済み出力 (L/R)
        // softClipBlockAVX2() へブロック間状態を渡すために保持する。
        double softClipPrevSample[2] = {0.0, 0.0};

        // Helpers
        float measureLevel (const juce::dsp::AudioBlock<const double>& block) const noexcept;
        void pushToFifo(const juce::dsp::AudioBlock<const double>& block,
                        juce::AbstractFifo& audioFifo,
                        juce::AudioBuffer<float>& audioFifoBuffer) const noexcept;
        // analyzerInputTap=true の場合、ヘッドルームゲイン適用前の raw 入力を
        // audioFifo / audioFifoBuffer にプッシュする。
        // これにより、インプットスペアナ/レベルメーターがヘッドルーム非適用の
        // "入力されたデータそのもの" を表示できる。
        float processInput(const juce::AudioSourceChannelInfo& bufferToFill, int numSamples,
                           double headroomGain,
                           bool analyzerInputTap,
                           juce::AbstractFifo& audioFifo,
                           juce::AudioBuffer<float>& audioFifoBuffer) noexcept;
        void processOutput(const juce::AudioSourceChannelInfo& bufferToFill,
                           int numSamples,
                           const ProcessingState& state) noexcept;
        float processInputDouble(const juce::AudioBuffer<double>& buffer, int numSamples,
                                 double headroomGain,
                                 bool analyzerInputTap,
                                 juce::AbstractFifo& audioFifo,
                                 juce::AudioBuffer<float>& audioFifoBuffer) noexcept;
        void processOutputDouble(juce::AudioBuffer<double>& buffer,
                                 int numSamples,
                                 const ProcessingState& state) noexcept;
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
    DSPCore* activeDSP = nullptr; // Ownership holder for Message Thread (Raw pointer)
    std::vector<std::pair<DSPCore*, uint32>> trashBin; // Time-based garbage collection for old DSPs
    juce::CriticalSection trashBinLock;

    std::atomic<double> currentSampleRate{48000.0};
    // 【Fix Bug #8】linear gain を格納 (dB変換はgetInputLevel/getOutputLevelで行う)
    std::atomic<float> inputLevelLinear{0.0f};
    std::atomic<float> outputLevelLinear{0.0f};
    std::atomic<int>   maxSamplesPerBlock{4096};

    std::atomic<bool> eqBypassRequested { false };
    std::atomic<bool> convBypassRequested { false };
    std::atomic<bool> eqBypassActive   { false };
    std::atomic<bool> convBypassActive { false };
    std::atomic<bool> rebuildRequested { false };
    std::atomic<ProcessingOrder> currentProcessingOrder{ProcessingOrder::ConvolverThenEQ};
    std::atomic<AnalyzerSource> currentAnalyzerSource { AnalyzerSource::Output };
    std::atomic<bool> analyzerEnabled { false };
    std::atomic<int> ditherBitDepth { 0 }; // 0 = 未初期化 (DeviceSettingsで最大値に設定される)
    std::atomic<NoiseShaperType> noiseShaperType { NoiseShaperType::Psychoacoustic };
    std::atomic<bool> pendingNoiseShaperLearningStart { false };
    std::atomic<bool> pendingNoiseShaperLearningResume { false };
    std::atomic<NoiseShaperLearner::LearningMode> pendingLearningMode { NoiseShaperLearner::LearningMode::Short };
    std::atomic<int> fixedNoiseLogIntervalMs { 2000 };
    std::atomic<int> fixedNoiseWindowSamples { 8192 };
    std::atomic<bool> softClipEnabled { true };
    std::atomic<float> saturationAmount { 0.2f };
    std::atomic<int> manualOversamplingFactor { 0 }; // 0=Auto, 1=1x, 2=2x, 4=4x, 8=8x
    std::atomic<OversamplingType> oversamplingType { OversamplingType::IIR };
    std::atomic<float> inputHeadroomDb { -6.0f };
    std::atomic<double> inputHeadroomGain { 0.5011872336272722 }; // -6dB
    std::atomic<float> outputMakeupDb { 12.0f };
    std::atomic<double> outputMakeupGain { 3.981071705534972 }; // +12dB (unity: -6dB input headroom + -6dB IR safety margin)
    std::atomic<int> rebuildGeneration { 0 }; // 非同期リビルドの競合防止用
    std::atomic<float> convolverInputTrimDb { 0.0f };
    std::atomic<double> convolverInputTrimGain { 1.0 }; // 0 dB (EQThenConvolver時にコンボルバー入力に適用)
    bool m_isRestoringState { false }; // requestLoadState 中はデフォルトリセットを抑制 (Message Thread のみ)
    uint32 fixedNoiseLastLogMs = 0;

    // 出力周波数フィルターモード (Thread-safe)
    std::atomic<convo::HCMode> convHCFilterMode { convo::HCMode::Natural }; // ① ハイカット
    std::atomic<convo::LCMode> convLCFilterMode { convo::LCMode::Natural }; // ① ローカット
    std::atomic<convo::HCMode> eqLPFFilterMode  { convo::HCMode::Natural }; // ② EQローパス

    // dB変換時の下限値
    static constexpr float LEVEL_METER_MIN_DB  = -120.0f;
    static constexpr float LEVEL_METER_MIN_MAG = 1e-6f;

    // EQ応答曲線計算用の定数
    static constexpr float EQ_GAIN_EPSILON = 0.01f;          // ゲインがこれ以下なら無視
    static constexpr float EQ_UNITY_GAIN_EPSILON = 1.0e-5f;  // 1.0との比較用

    // EQ応答曲線計算用ワークバッファ (Message Thread/UI Threadで再利用)
    std::vector<float> eqTotalMagSqLBuffer;
    std::vector<float> eqTotalMagSqRBuffer;
    std::vector<float> eqBandMagSqBuffer;
    //----------------------------------------------------------
    // プライベートヘルパー (Message Thread のみ)
    //----------------------------------------------------------
    void applyDefaultsForCurrentMode();

    //----------------------------------------------------------
    // ヘルパー関数
    //----------------------------------------------------------
    // Note: This function performs memory allocation (including MKL) and other blocking operations
    // such as IR resampling. It MUST only be called from the message thread.
    // The prepareToPlay() method ensures this by using MessageManager::callAsync if necessary.
    void requestRebuild(double sampleRate, int samplesPerBlock);
    void commitNewDSP(DSPCore* newDSP, int generation);
    bool isRebuildObsolete(int generation) const { return generation != rebuildGeneration.load(); }

    // Worker thread for rebuilds
    void rebuildThreadLoop();
    std::thread rebuildThread;
    std::mutex rebuildMutex;
    std::condition_variable rebuildCV;
    std::atomic<bool> rebuildThreadShouldExit { false };
    bool hasPendingTask = false;

    struct RebuildTask {
        DSPCore* newDSP = nullptr;
        DSPCore* currentDSP = nullptr;
        double sampleRate;
        int samplesPerBlock;
        int ditherDepth;
        int manualOversamplingFactor;
        OversamplingType oversamplingType;
        NoiseShaperType noiseShaperType;
        int generation;
    };
    RebuildTask pendingTask;

    // --- Adaptiveノイズシェイパー学習用メンバー ---
    struct AdaptiveCoeffBankSlot
    {
        double sampleRateHz = 0.0;
        CoeffSet coeffSetA {};
        CoeffSet coeffSetB {};
        std::atomic<const CoeffSet*> current { nullptr };
        std::atomic<uint32_t> generation { 1u };
        NoiseShaperLearner::State state {};
        std::mutex stateMutex;
    };

    std::unique_ptr<NoiseShaperLearner> noiseShaperLearner;
    LockFreeRingBuffer<AudioBlock, 4096> audioCaptureQueue;
    std::array<AdaptiveCoeffBankSlot, kAdaptiveNoiseShaperSampleRateBankCount * kAdaptiveBitDepthCount * kLearningModeCount> adaptiveCoeffBanks {};
    std::atomic<int> currentAdaptiveCoeffBankIndex { 1 };
    std::uintptr_t audioThreadAffinityMask = 0;
    std::uintptr_t noiseLearnerThreadAffinityMask = 0;
    std::uintptr_t nonAudioThreadAffinityMask = 0;
    std::mutex adaptiveAutosaveCallbackMutex;
    std::function<void()> adaptiveAutosaveCallback;
    void initialiseAdaptiveCoeffBanks() noexcept;
    static int resolveAdaptiveCoeffBankIndex(double sampleRate) noexcept;
    static int getAdaptiveBitDepthIndex(int bitDepth) noexcept;
    AdaptiveCoeffBankSlot& getAdaptiveCoeffBankForIndex(int bankIndex) noexcept;
    const AdaptiveCoeffBankSlot& getAdaptiveCoeffBankForIndex(int bankIndex) const noexcept;
    void selectAdaptiveCoeffBankForCurrentSettings() noexcept;
    void initialiseThreadAffinityMasks() noexcept;
    void pinCurrentThreadToAudioCoreIfNeeded() noexcept;
    void pinCurrentThreadToNoiseLearnerCoreIfNeeded() const noexcept;
    void pinCurrentThreadToNonAudioCoresIfNeeded() const noexcept;
    void publishCoeffsToBank(int bankIndex, const double* coeffs);

    friend class NoiseShaperLearner;

    JUCE_DECLARE_WEAK_REFERENCEABLE(AudioEngine)
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AudioEngine)
};
