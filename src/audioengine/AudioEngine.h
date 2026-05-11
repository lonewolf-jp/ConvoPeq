
//============================================================================
#pragma once
//============================================================================

#include <cstdint>

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
    std::uint64_t sessionId = 0;
};

// RT/Worker間ダブルバッファ係数連携（RCU）
struct CoeffSet {
    static constexpr int kDim = kAdaptiveNoiseShaperOrder;
    double k[kDim] = {};
};

#include <JuceHeader.h>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <array>
#include <functional>
#include <limits>
#include <memory>
#include <unordered_map>
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
#include "core/RCUReader.h"
#include "core/EBRQueue.h"
#include "EQEditProcessor.h"
#include "PsychoacousticDither.h"
#include "FixedNoiseShaper.h"
#include "Fixed15TapNoiseShaper.h"
#include "LatticeNoiseShaper.h"
#include "OutputFilter.h"
#include "DspNumericPolicy.h"
#include "UltraHighRateDCBlocker.h"
#include "LockFreeRingBuffer.h"
#include "LockFreeAudioRingBuffer.h"
#include "NoiseShaperLearnerTypes.h"
#include "GenerationManager.h"
#include "RuntimeBuildTypes.h"
#include "RuntimeTransition.h"
#include "RuntimeCommandQueue.h"
#include "core/Types.h"
#include "core/SnapshotCoordinator.h"
#include "core/EpochCore.h"
#include "core/CommandBuffer.h"
#include "core/ThreadAffinityManager.h"
#include "core/WorkerThread.h"
#include "core/RebuildTypes.h"

class NoiseShaperLearner;

inline std::atomic<std::uint64_t> g_runtimePublishCount { 0 };
inline std::atomic<std::uint64_t> g_runtimeRetireCount { 0 };
inline std::atomic<std::uint64_t> g_runtimeReclaimCount { 0 };

// デバッグビルド時のみログを出力するマクロ
#if defined(JUCE_DEBUG) && !defined(NDEBUG)
    #define DBG_LOG(msg) juce::Logger::writeToLog(msg)
#else
    #define DBG_LOG(msg) ((void)0)
#endif

inline double absNoLibm(double x) noexcept
{
    union { double d; std::uint64_t u; } v { x };
    v.u &= 0x7FFFFFFFFFFFFFFFULL;
    return v.d;
}

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
                                     private ConvolverProcessor::Listener,
                                     private juce::Timer,
                                     private juce::AsyncUpdater
{
public:
    using SampleType = double; // 内部DSP精度 (JUCE推奨)

    using ProcessingOrder = convo::ProcessingOrder;
    using OversamplingType = convo::OversamplingType;   // ここに追加
    using NoiseShaperType = convo::NoiseShaperType;     // ここに追加

    enum class AnalyzerSource
    {
        Input,
        Output
    };

    struct EngineParameterSnapshot
    {
        bool eqBypassed = false;
        bool convBypassed = false;
        ProcessingOrder order = ProcessingOrder::ConvolverThenEQ;
        AnalyzerSource analyzerSource = AnalyzerSource::Output;
        bool analyzerEnabled = false;
        bool softClipEnabled = false;
        float saturationAmount = 0.0f;
        double inputHeadroomGain = 1.0;
        double outputMakeupGain = 1.0;
        double convolverInputTrimGain = 1.0;
        convo::HCMode convHCMode = convo::HCMode::Natural;
        convo::LCMode convLCMode = convo::LCMode::Natural;
        convo::HCMode eqLPFMode = convo::HCMode::Natural;
        int adaptiveCoeffBankIndex = 0;
        const CoeffSet* adaptiveCoeffSet = nullptr;
        uint32_t adaptiveCoeffGeneration = 0;
        bool adaptiveCaptureEnabled = false;
    };

    //----------------------------------------------------------
     // DSPコア (Audio Threadで実行される処理のコンテナ)
    //----------------------------------------------------------
    struct DSPCore
    {
        struct DCBlockerRuntimeState
        {
            convo::UltraHighRateDCBlocker outputL, outputR;
            convo::UltraHighRateDCBlocker inputL, inputR;
            convo::UltraHighRateDCBlocker oversampledL, oversampledR;

            void init(double sampleRate, double processingRate) noexcept
            {
                outputL.init(sampleRate, 3.0);
                outputR.init(sampleRate, 3.0);
                inputL.init(sampleRate, 3.0);
                inputR.init(sampleRate, 3.0);
                oversampledL.init(processingRate, 1.0);
                oversampledR.init(processingRate, 1.0);
            }

            void reset() noexcept
            {
                outputL.reset();
                outputR.reset();
                inputL.reset();
                inputR.reset();
                oversampledL.reset();
                oversampledR.reset();
            }
        };

        struct ConvolverRuntimeState
        {
            ConvolverProcessor* processor = nullptr;

            void bind(ConvolverProcessor& value) noexcept
            {
                processor = &value;
            }

            ConvolverProcessor& ref() noexcept
            {
                jassert(processor != nullptr);
                return *processor;
            }

            const ConvolverProcessor& ref() const noexcept
            {
                jassert(processor != nullptr);
                return *processor;
            }

            void prepare(AudioEngine* ownerEngine, double processingRate, int processingBlockSize) noexcept
            {
                auto& proc = ref();
                proc.setRcuProvider(ownerEngine);
                proc.prepareToPlay(processingRate, processingBlockSize);
            }

            void resetForRuntime() noexcept
            {
                ref().reset();
            }

            void cleanupForRuntime() noexcept
            {
                ref().cleanup();
            }
        };

        struct EQRuntimeState
        {
            EQProcessor* processor = nullptr;

            void bind(EQProcessor& value) noexcept
            {
                processor = &value;
            }

            EQProcessor& ref() noexcept
            {
                jassert(processor != nullptr);
                return *processor;
            }

            const EQProcessor& ref() const noexcept
            {
                jassert(processor != nullptr);
                return *processor;
            }

            void prepare(double processingRate, int internalMaxBlock) noexcept
            {
                ref().prepareToPlay(processingRate, internalMaxBlock);
            }

            void resetForRuntime() noexcept
            {
                ref().reset();
            }

            void cleanupForRuntime() noexcept
            {
                ref().cleanup();
            }
        };

        struct RampRuntimeState
        {
            std::atomic<int> fadeInSamplesLeft {0};
            convo::LinearRamp bypassFadeGainDouble;
            bool bypassedDouble = false;

            void prepare(double sampleRate) noexcept
            {
                bypassFadeGainDouble.reset(sampleRate, 0.005);
                bypassFadeGainDouble.setCurrentAndTargetValue(1.0);
                bypassedDouble = false;
                fadeInSamplesLeft.store(0, std::memory_order_relaxed);
            }

            void resetForRuntime() noexcept
            {
                bypassFadeGainDouble.setCurrentAndTargetValue(1.0);
                bypassedDouble = false;
                fadeInSamplesLeft.store(0, std::memory_order_relaxed);
            }
        };

        struct HistoryRuntimeState
        {
            convo::ScopedAlignedPtr<double> fixedLatencyBufferL;
            convo::ScopedAlignedPtr<double> fixedLatencyBufferR;
            int fixedLatencyBufferSize = 0;
            int fixedLatencyWritePos = 0;
            int fixedLatencySamples = 0;
            double softClipPrevSample[2] = {0.0, 0.0};

            void clearSoftClipHistory() noexcept
            {
                softClipPrevSample[0] = 0.0;
                softClipPrevSample[1] = 0.0;
            }

            void configureFixedLatencySamples(int samples, int maxInternalBlockSize) noexcept
            {
                const int clamped = std::max(0, samples);
                fixedLatencySamples = clamped;
                fixedLatencyWritePos = 0;

                const int requiredSize = clamped + std::max(1, maxInternalBlockSize) + 2;
                if (requiredSize > fixedLatencyBufferSize || !fixedLatencyBufferL || !fixedLatencyBufferR)
                {
                    auto newDelayL = convo::ScopedAlignedPtr<double>(static_cast<double*>(convo::aligned_malloc(
                        static_cast<size_t>(requiredSize) * sizeof(double), 64)));
                    auto newDelayR = convo::ScopedAlignedPtr<double>(static_cast<double*>(convo::aligned_malloc(
                        static_cast<size_t>(requiredSize) * sizeof(double), 64)));

                    juce::FloatVectorOperations::clear(newDelayL.get(), requiredSize);
                    juce::FloatVectorOperations::clear(newDelayR.get(), requiredSize);

                    fixedLatencyBufferL = std::move(newDelayL);
                    fixedLatencyBufferR = std::move(newDelayR);
                    fixedLatencyBufferSize = requiredSize;
                }
                else if (fixedLatencyBufferSize > 0)
                {
                    juce::FloatVectorOperations::clear(fixedLatencyBufferL.get(), fixedLatencyBufferSize);
                    juce::FloatVectorOperations::clear(fixedLatencyBufferR.get(), fixedLatencyBufferSize);
                }
            }

            void resetForRuntime() noexcept
            {
                fixedLatencyWritePos = 0;
                if (fixedLatencyBufferL && fixedLatencyBufferSize > 0)
                    juce::FloatVectorOperations::clear(fixedLatencyBufferL.get(), fixedLatencyBufferSize);
                if (fixedLatencyBufferR && fixedLatencyBufferSize > 0)
                    juce::FloatVectorOperations::clear(fixedLatencyBufferR.get(), fixedLatencyBufferSize);
                clearSoftClipHistory();
            }
        };

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
            uint64_t captureSessionId;
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

    void prepare(double sampleRate, int samplesPerBlock, int bitDepth, int manualOversamplingFactor, OversamplingType oversamplingType, NoiseShaperType selectedNoiseShaperType, AudioEngine* owner);
    void setFixedLatencySamples(int samples);
    void reset();
        void process(const juce::AudioSourceChannelInfo& bufferToFill, LockFreeAudioRingBuffer& analyzerFifo,
             std::atomic<float>& inputLevelLinear,
                 std::atomic<float>& outputLevelLinear, const ProcessingState& state);
    void processToBuffer(const juce::AudioSourceChannelInfo& source,
                         juce::AudioBuffer<float>& destination,
                 LockFreeAudioRingBuffer& analyzerFifo,
                         std::atomic<float>& inputLevelLinear,
                         std::atomic<float>& outputLevelLinear,
                         const ProcessingState& state);
    void processDouble(juce::AudioBuffer<double>& buffer,
                   LockFreeAudioRingBuffer& analyzerFifo,
                       std::atomic<float>& inputLevelLinear,
                       std::atomic<float>& outputLevelLinear,
                       const ProcessingState& state);
    void processDoubleToBuffer(const juce::AudioBuffer<double>& source,
                               juce::AudioBuffer<double>& destination,
                       LockFreeAudioRingBuffer& analyzerFifo,
                               std::atomic<float>& inputLevelLinear,
                               std::atomic<float>& outputLevelLinear,
                               const ProcessingState& state);
        ConvolverProcessor convolver;
        EQProcessor eq;
        // A-12 first step: mutable DC blocker state is detached from DSP graph members.
        std::unique_ptr<DCBlockerRuntimeState> dcBlockerState;
        std::unique_ptr<ConvolverRuntimeState> convolverState;
        std::unique_ptr<EQRuntimeState> eqState;
        std::unique_ptr<RampRuntimeState> rampState;
        std::unique_ptr<HistoryRuntimeState> historyState;

        DCBlockerRuntimeState& dcBlockers() noexcept
        {
            jassert(dcBlockerState != nullptr);
            return *dcBlockerState;
        }

        const DCBlockerRuntimeState& dcBlockers() const noexcept
        {
            jassert(dcBlockerState != nullptr);
            return *dcBlockerState;
        }

        ConvolverProcessor& convolverRt() noexcept
        {
            jassert(convolverState != nullptr);
            return convolverState->ref();
        }

        const ConvolverProcessor& convolverRt() const noexcept
        {
            jassert(convolverState != nullptr);
            return convolverState->ref();
        }

        EQProcessor& eqRt() noexcept
        {
            jassert(eqState != nullptr);
            return eqState->ref();
        }

        const EQProcessor& eqRt() const noexcept
        {
            jassert(eqState != nullptr);
            return eqState->ref();
        }

        RampRuntimeState& ramps() noexcept
        {
            jassert(rampState != nullptr);
            return *rampState;
        }

        const RampRuntimeState& ramps() const noexcept
        {
            jassert(rampState != nullptr);
            return *rampState;
        }

        HistoryRuntimeState& histories() noexcept
        {
            jassert(historyState != nullptr);
            return *historyState;
        }

        const HistoryRuntimeState& histories() const noexcept
        {
            jassert(historyState != nullptr);
            return *historyState;
        }

        ::convo::PsychoacousticDither dither;
        ::convo::FixedNoiseShaper fixedNoiseShaper;
        ::convo::Fixed15TapNoiseShaper fixed15TapNoiseShaper;
        LatticeNoiseShaper adaptiveNoiseShaper;
        // 出力周波数フィルター (① ハイカット/ローカット / ② ローパス/ハイパス)
        convo::OutputFilter outputFilter;

        CustomInputOversampler oversampling;
        size_t oversamplingFactor = 1;
        OversamplingType activeOversamplingType = OversamplingType::IIR;
        int ditherBitDepth = 0; // DSPCore内でディザリング判定に使用
        NoiseShaperType noiseShaperType = NoiseShaperType::Psychoacoustic;
        uint32_t activeAdaptiveCoeffGeneration = 0;
        int activeAdaptiveCoeffBankIndex = -1;
        uint64_t currentCaptureSessionId = 0;
        std::uint64_t runtimeUuid = 0;
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
        static constexpr int FADE_IN_SAMPLES = 2048; // 42ms @ 48kHz
        AudioEngine* ownerEngine = nullptr;

        // RCU Reader Support (Forward to ownerEngine)
        uint64_t publishRcuEpoch() noexcept { return ownerEngine ? ownerEngine->publishRcuEpoch() : 1; }
        void enterRcuReader(int tid) noexcept { if (ownerEngine) ownerEngine->enterRcuReader(tid); }
        void exitRcuReader(int tid) noexcept { if (ownerEngine) ownerEngine->exitRcuReader(tid); }
        static thread_local size_t tls_readerSlot;

        // B2: processDouble 用のバイパスフェード状態
        convo::ScopedAlignedPtr<double> dryBypassBufferDoubleL;
        convo::ScopedAlignedPtr<double> dryBypassBufferDoubleR;
        int dryBypassCapacityDouble = 0;

        // Helpers
        float measureLevel (const juce::dsp::AudioBlock<const double>& block) const noexcept;
        void applyFixedLatencyDelay(double* dataL, double* dataR, int numSamples) noexcept;
        void pushToFifo(const juce::dsp::AudioBlock<const double>& block,
                        LockFreeAudioRingBuffer& analyzerFifo) const noexcept;
        // analyzerInputTap=true の場合、ヘッドルームゲイン適用前の raw 入力を
        // analyzerFifo にプッシュする。
        // これにより、インプットスペアナ/レベルメーターがヘッドルーム非適用の
        // "入力されたデータそのもの" を表示できる。
        float processInput(const juce::AudioSourceChannelInfo& bufferToFill, int numSamples,
                           double headroomGain,
                           bool analyzerInputTap,
                       LockFreeAudioRingBuffer& analyzerFifo) noexcept;
        void processOutput(const juce::AudioSourceChannelInfo& bufferToFill,
                           int numSamples,
                           const ProcessingState& state) noexcept;
        float processInputDouble(const juce::AudioBuffer<double>& buffer, int numSamples,
                                 double headroomGain,
                                 bool analyzerInputTap,
                           LockFreeAudioRingBuffer& analyzerFifo) noexcept;
        void processOutputDouble(juce::AudioBuffer<double>& buffer,
                                 int numSamples,
                                 const ProcessingState& state) noexcept;
    private:
        static double musicalSoftClip(double x, double threshold, double knee, double asymmetry) noexcept;
    };

    class Listener
    {
     public:
         virtual ~Listener() = default;
        virtual void eqSettingsChanged() = 0;
    };

    // FIFO設定
    static constexpr int FIFO_SIZE = 1048576;  // Lock-free FIFO サイズ (2^20, SAFE_MAX_BLOCK_SIZE * 8x OS をカバー)

    // EQ応答曲線計算用の定数
    static constexpr int   NUM_DISPLAY_BARS = 128;
    static constexpr float EQ_GAIN_EPSILON = 0.01f;          // ゲインがこれ以下なら無視
    static constexpr float EQ_UNITY_GAIN_EPSILON = 1.0e-5f;  // 1.0との比較用

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
    static thread_local size_t tls_readerSlot;

    //----------------------------------------------------------
    // AudioSource インターフェース
    //----------------------------------------------------------
    void prepareToPlay (int samplesPerBlockExpected, double sampleRate) override;
    void releaseResources() override;
    void getNextAudioBlock (const juce::AudioSourceChannelInfo& bufferToFill) override;

    // ==================================================================
    // EBR (Epoch-Based Reclamation) 基盤
    // ==================================================================
    void advanceRcuEpoch() noexcept;
    void tryReclaimResources() noexcept;

    // RCU Reader Support
    uint64_t publishRcuEpoch() noexcept;
    void enterRcuReader(int readerIndex) noexcept;
    void exitRcuReader(int readerIndex) noexcept;

    void processBlockDouble (juce::AudioBuffer<double>& buffer);
    void changeListenerCallback(juce::ChangeBroadcaster* source) override;
    void convolverParamsChanged(ConvolverProcessor* processor) override;
    void timerCallback() override;

    //----------------------------------------------------------
    // 外部インターフェース (Message Thread)
    //----------------------------------------------------------
    ConvolverProcessor& getConvolverProcessor() { return uiConvolverProcessor; }
    const ConvolverProcessor& getConvolverProcessor() const { return uiConvolverProcessor; }
    EQEditProcessor& getEQProcessor() { return uiEqEditor; }
    const ThreadAffinityManager& getAffinityManager() const noexcept { return affinityManager; }

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
    int getTotalLatencySamples() const;  // PDC 用エイリアス (getCurrentLatencySamples と同値)
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




    int getFifoNumReady() const { return analyzerFifo.getAvailableSamples(); }
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
    void beginBulkParameterRestore() noexcept;
    void endBulkParameterRestore(bool requestRebuildNow = true) noexcept;

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
    void requestSnapshotForNoiseShaper();
    void commitAGCChange();
    void requestRebuild(convo::RebuildKind kind) noexcept;
    void setFixedNoiseLogIntervalMs(int intervalMs) noexcept;
    int getFixedNoiseLogIntervalMs() const noexcept;
    void setFixedNoiseWindowSamples(int windowSamples) noexcept;
    int getFixedNoiseWindowSamples() const noexcept;

    void setIRFadeSamples(int samples) noexcept
    {
        const int clamped = juce::jlimit(0, 96000, samples);
        m_irFadeSamples.store(clamped, std::memory_order_relaxed);

        const double sr = currentSampleRate.load(std::memory_order_acquire);
        if (sr > 0.0)
        {
            const double fadeSec = (clamped > 0)
                ? (static_cast<double>(clamped) / sr)
                : 0.001;
            m_irFadeTimeSec.store(fadeSec, std::memory_order_relaxed);
        }
    }
    void setEQFadeSamples(int samples) noexcept { m_eqFadeSamples.store(samples, std::memory_order_relaxed); }
    int getIRFadeSamples() const noexcept { return m_irFadeSamples.load(std::memory_order_relaxed); }
    int getEQFadeSamples() const noexcept { return m_eqFadeSamples.load(std::memory_order_relaxed); }
    void setCrossfadeStartDelayBlocks(int blocks) noexcept;
    int getCrossfadeStartDelayBlocks() const noexcept;
    bool isFading() const noexcept { return m_coordinator.isFading(); }
    const convo::SnapshotCoordinator& getSnapshotCoordinator() const noexcept { return m_coordinator; }
    void setIRChangeFlag() noexcept { m_pendingIRChange.store(true, std::memory_order_release); }

    // UI 操作が実際の音声演算へ反映されたかを確認するための診断値。
    uint64_t getLastCreatedEqHashForDebug() const noexcept { return debugLastCreatedEqHash.load(std::memory_order_acquire); }
    uint64_t getLastAppliedEqHashForDebug() const noexcept { return debugLastAppliedEqHash.load(std::memory_order_acquire); }
    convo::TransitionState getRuntimeTransitionStateForDebug() const noexcept
    {
        convo::TransitionState snapshot{};
        snapshot.current = reinterpret_cast<void*>(static_cast<uintptr_t>(debugRuntimeTransitionCurrentPtr.load(std::memory_order_acquire)));
        snapshot.next = reinterpret_cast<void*>(static_cast<uintptr_t>(debugRuntimeTransitionNextPtr.load(std::memory_order_acquire)));
        int rawPolicy = debugRuntimeTransitionPolicy.load(std::memory_order_acquire);
        if (rawPolicy < static_cast<int>(convo::TransitionPolicy::SmoothOnly)
            || rawPolicy > static_cast<int>(convo::TransitionPolicy::DryAsOld))
        {
            rawPolicy = static_cast<int>(convo::TransitionPolicy::SmoothOnly);
        }
        snapshot.policy = static_cast<convo::TransitionPolicy>(rawPolicy);
        snapshot.fadeTimeSec = debugRuntimeTransitionFadeSec.load(std::memory_order_acquire);
        snapshot.latencyDeltaSamples = debugRuntimeTransitionLatencyDeltaSamples.load(std::memory_order_acquire);
        snapshot.active = (debugRuntimeTransitionActive.load(std::memory_order_acquire) != 0);
        return snapshot;
    }

    const convo::EQParameters& getLatestEqParamsFallback(uint64_t& outHash) const noexcept
    {
        const int index = latestEqFallbackReadIndex.load(std::memory_order_acquire);
        outHash = latestEqHashForFallback[(size_t) index];
        return latestEqParamsForFallback[(size_t) index];
    }

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
    void startNoiseShaperLearning(convo::NoiseShaperLearningMode mode, bool resume = false);
    void stopNoiseShaperLearning();
    void setNoiseShaperLearningMode(convo::NoiseShaperLearningMode mode);
    convo::NoiseShaperLearningMode getNoiseShaperLearningMode() const { return pendingLearningMode.load(std::memory_order_acquire); }
    bool isNoiseShaperLearning() const;
    const convo::NoiseShaperLearnerProgress& getNoiseShaperLearningProgress() const;
    int copyNoiseShaperLearningHistory(double* outScores, int maxPoints) const noexcept;
    // 学習ワーカーが記録したエラーメッセージを返す（UI 表示用）。エラーなしは nullptr。
    const char* getNoiseShaperLearningError() const noexcept;
    static int getAdaptiveSampleRateBankCount() noexcept;

    struct RuntimeLifecycleDiagnostics
    {
        std::uint64_t publishCount = 0;
        std::uint64_t retireCount = 0;
        std::uint64_t reclaimCount = 0;
    };

    RuntimeLifecycleDiagnostics getRuntimeLifecycleDiagnostics() const noexcept
    {
        return {
            g_runtimePublishCount.load(std::memory_order_acquire),
            g_runtimeRetireCount.load(std::memory_order_acquire),
            g_runtimeReclaimCount.load(std::memory_order_acquire)
        };
    }

    struct RebuildDispatchDiagnostics
    {
        std::uint64_t requestCount = 0;
        std::uint64_t queuedCount = 0;
        std::uint64_t blockedPendingDuplicateCount = 0;
        std::uint64_t blockedRecentDuplicateCount = 0;
        std::uint64_t runtimeQueueFullCount = 0;
        std::uint64_t drainedCommandCount = 0;
        std::uint64_t matchedRuntimeCommandCount = 0;
        std::uint64_t taskSnapshotFallbackCount = 0;
    };

    RebuildDispatchDiagnostics getRebuildDispatchDiagnostics() const noexcept
    {
        return {
            debugRebuildDispatchRequestCount.load(std::memory_order_acquire),
            debugRebuildDispatchQueuedCount.load(std::memory_order_acquire),
            debugRebuildDispatchBlockedPendingDuplicateCount.load(std::memory_order_acquire),
            debugRebuildDispatchBlockedRecentDuplicateCount.load(std::memory_order_acquire),
            debugRebuildDispatchRuntimeQueueFullCount.load(std::memory_order_acquire),
            debugRebuildDispatchDrainedCommandCount.load(std::memory_order_acquire),
            debugRebuildDispatchMatchedRuntimeCommandCount.load(std::memory_order_acquire),
            debugRebuildDispatchTaskSnapshotFallbackCount.load(std::memory_order_acquire)
        };
    }

    // --- NoiseShaperLearner Settings ---
    convo::NoiseShaperLearnerSettings getNoiseShaperLearnerSettings() const;
    void setNoiseShaperLearnerSettings(const convo::NoiseShaperLearnerSettings& settings);

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
    static int getAdaptiveCoeffBankIndex(double sampleRate, int bitDepth, convo::NoiseShaperLearningMode mode) noexcept;

    bool getAdaptiveNoiseShaperState(int bankIndex, convo::NoiseShaperLearnerState& outState) const noexcept;
    void setAdaptiveNoiseShaperState(int bankIndex, const convo::NoiseShaperLearnerState& inState) noexcept;

private:
    //==================================================
    // クロスフェード用レイテンシ整合バッファ（最大2秒@48kHz, double精度, L/R独立）
    // 64byte aligned double* buffers (RT外確保)
    // ===== 遅延整合バッファ・スレッド安全 =====
    double* latencyBufOldL = nullptr;
    double* latencyBufOldR = nullptr;
    double* latencyBufNewL = nullptr;
    double* latencyBufNewR = nullptr;
    int latencyBufSize = 0;
    // AudioThread専用（atomic不要）
    int latencyWritePos = 0;
    // 遅延値はatomicで管理（MessageThread→AudioThread）
    std::atomic<int> latencyDelayOld { 0 };
    std::atomic<int> latencyDelayNew { 0 };
    // AudioThread snapshot（フェード単位で固定）
    int latencyDelayOld_RT = 0;
    int latencyDelayNew_RT = 0;
    // バッファリセット要求（MessageThread→AudioThread）
    std::atomic<bool> latencyResetPending { false };
    static constexpr int kMaxLatencySamples = 1536000; // 最大2秒@768kHz対応
    static constexpr int MAX_LATENCY_ALIGN_SAMPLES = 96000 * 2; // 2秒@48kHz
    class EQCacheManager
    {
    public:
        EQCacheManager();
        EQCoeffCache* getOrCreate(const convo::EQParameters& params,
                                  double sampleRate,
                                  int maxBlockSize,
                                  uint64_t generation);
        EQCoeffCache* get(uint64_t hash) const noexcept;
        void releaseCache(EQCoeffCache* cache) noexcept;
        ~EQCacheManager();

    private:
        struct CacheMap
        {
            CacheMap() = default;

            CacheMap(const CacheMap& other)
            {
                for (const auto& entry : other.map)
                {
                    if (entry.second != nullptr)
                    {
                        // EBR: Using RefCountedDeferred for cache objects as they are shared
                        entry.second->addRef();
                    }

                    map.emplace(entry.first, entry.second);
                }
            }

            ~CacheMap()
            {
                for (auto& entry : map)
                {
                    if (entry.second != nullptr)
                        entry.second->release();
                }
            }

            std::unordered_map<uint64_t, EQCoeffCache*> map;
        };

        const CacheMap* loadMap() const noexcept
        {
            return cacheMapPtr.load(std::memory_order_acquire);
        }

        void storeNewMap(CacheMap* newMap) noexcept;
        void drainDeferredMapsUnderLock() noexcept;
        bool tryEnqueueDeferredMap(CacheMap* map) noexcept;

        mutable std::mutex writeMutex;
        std::atomic<CacheMap*> cacheMapPtr { nullptr };
        std::vector<CacheMap*> enqueueFallbackMaps;
    };

    static double estimateOversamplingLatencySamples(int oversamplingFactor,
                                                     OversamplingType oversamplingType,
                                                     double baseSampleRate) noexcept;

public:
    //----------------------------------------------------------
    // 処理チェーンコンポーネント
    //----------------------------------------------------------
     // UI/State管理用のインスタンス (Audio Threadでは使用しない)
    ConvolverProcessor  uiConvolverProcessor;
    EQEditProcessor uiEqEditor;

    LockFreeAudioRingBuffer analyzerFifo;

    //----------------------------------------------------------
    // 状態管理
    //----------------------------------------------------------
    std::atomic<DSPCore*> currentDSP { nullptr }; // Raw pointer for Audio Thread (Lock-free)
    DSPCore* activeDSP = nullptr; // Ownership holder for Message Thread (Raw pointer)
    std::atomic<DSPCore*> fadingOutDSP { nullptr }; // D2: DSP切替クロスフェード用
    std::atomic<convo::RuntimePublishState*> runtimePublishState { nullptr };
    std::atomic<std::uint64_t> runtimePublishRevision { 0 };
    std::atomic<convo::EngineRuntime*> engineRuntimeState { nullptr };
    std::atomic<std::uint64_t> engineRuntimeRevision { 0 };

        // フェードキューイング機構（DSP構造切替用）
        std::atomic<DSPCore*> queuedOldDSP { nullptr };
        std::atomic<double> queuedFadeTimeSec { 0.030 };      // 現在開始するフェード時間
        std::atomic<double> queuedNextFadeTimeSec { 0.030 };  // キュー待機中の次フェード時間
        std::atomic<bool> fadeQueued { false };

        // モード別フェード時間（秒）
        std::atomic<double> m_irFadeTimeSec { 0.080 };
        std::atomic<double> m_irLengthFadeTimeSec { 0.050 };
        std::atomic<double> m_phaseFadeTimeSec { 0.060 };
        std::atomic<double> m_directHeadFadeTimeSec { 0.010 };
        std::atomic<double> m_nucFilterFadeTimeSec { 0.030 };
        std::atomic<double> m_tailFadeTimeSec { 0.030 };
        std::atomic<double> m_osFadeTimeSec { 0.030 };
        std::atomic<int> m_crossfadeStartDelayBlocks { 1 };

    std::atomic<bool> dspCrossfadePending { false };
    std::atomic<int> dspCrossfadeStartDelayBlocks { 0 };
    std::atomic<bool> firstIrDryCrossfadePending { false }; // 初回IRロード時に dry を旧信号として使用
    std::atomic<bool> firstIrDryCrossfadeDone { false };    // アプリ起動後の初回1回のみ有効
    std::atomic<bool> dspCrossfadeUseDryAsOld { false };    // Audio Thread 実行中フラグ
    std::atomic<int> dspCrossfadeDryHoldSamples { 0 };      // dry-as-old開始直後の旧信号優先ホールド
    convo::LinearRamp dspCrossfadeGain;
    convo::LinearRamp dspCrossfadeDryScaleGain;             // dry信号をIRスケールに合わせるため（60ms ramp）
    juce::AudioBuffer<float> dspCrossfadeFloatBuffer;
    juce::AudioBuffer<double> dspCrossfadeDoubleBuffer;

    std::atomic<double> currentSampleRate{48000.0};
    // 【Fix Bug #8】linear gain を格納 (dB変換はgetInputLevel/getOutputLevelで行う)
    std::atomic<float> inputLevelLinear{0.0f};
    std::atomic<float> outputLevelLinear{0.0f};
    std::atomic<int>   maxSamplesPerBlock{4096};

    std::atomic<bool> eqBypassRequested { false };
    std::atomic<bool> convBypassRequested { false };
    std::atomic<bool> eqBypassActive   { false };
    std::atomic<bool> convBypassActive { false };
    std::atomic<bool> pendingStructuralRebuildFromNonMT_{ false };
    std::atomic<bool> deferredStructuralRebuildPending_ { false };
    std::atomic<int64_t> deferredStructuralRebuildDueTicks_ { 0 };
    std::atomic<bool> deferredFinalizeAwareRebuildPending_ { false };
    // 同一IR構造に対する Structural rebuild の多重発火を抑止する。
    // 値は「直近で rebuild を要求した UI 側 Convolver 構造ハッシュ」。
    std::atomic<uint64_t> lastIssuedConvolverStructuralHash_{ 0 };
    EQCacheManager eqCacheManager;
    std::atomic<ProcessingOrder> currentProcessingOrder{ProcessingOrder::ConvolverThenEQ};
    std::atomic<AnalyzerSource> currentAnalyzerSource { AnalyzerSource::Output };
    std::atomic<bool> analyzerEnabled { false };
    std::atomic<int> ditherBitDepth { 0 }; // 0 = 未初期化 (DeviceSettingsで最大値に設定される)
    std::atomic<NoiseShaperType> noiseShaperType { NoiseShaperType::Psychoacoustic };

    // 【False Sharing 防止】頻繁な UI 更新変数を独立キャッシュラインへ配置
    struct LearningCommand
    {
        enum class Type : uint8_t { Start, Stop, IRChanged, DSPReady };

        Type type = Type::Stop;
        bool resume = false;
        convo::NoiseShaperLearningMode mode = convo::NoiseShaperLearningMode::Short;
        uint64_t irGeneration = 0;
    };

    struct LearnerDispatchAction
    {
        enum class Type : uint8_t { Start, Stop };

        Type type = Type::Stop;
        bool resume = false;
        convo::NoiseShaperLearningMode mode = convo::NoiseShaperLearningMode::Short;
    };

    enum class LearningRuntimeState : uint8_t { Idle, WaitingForDSP, Running };

    static constexpr uint32_t learningCommandBufferSize = 128;
    static constexpr uint32_t learningCommandBufferMask = learningCommandBufferSize - 1;
    static constexpr uint32_t learnerDispatchBufferSize = 32;
    static constexpr uint32_t learnerDispatchBufferMask = learnerDispatchBufferSize - 1;

    #pragma warning(push)
    #pragma warning(disable: 4324)
    alignas(64) std::atomic<convo::NoiseShaperLearningMode> pendingLearningMode { convo::NoiseShaperLearningMode::Short };
    alignas(64) std::atomic<uint64_t> globalCaptureSessionId { 1 };
    #pragma warning(pop)

    LearningCommand learningCommandBuffer[learningCommandBufferSize] {};
    LearnerDispatchAction learnerDispatchBuffer[learnerDispatchBufferSize] {};
    std::atomic<bool> learnerDispatchOverflow { false };
    std::atomic<LearnerDispatchAction> lastFailedAction {};

    #pragma warning(push)
    #pragma warning(disable: 4324)
    alignas(64) std::atomic<uint32_t> learningCommandWrite { 0 };  // Message/UI thread only
    alignas(64) std::atomic<uint32_t> learningCommandRead { 0 };   // Audio thread only
    alignas(64) std::atomic<uint32_t> learnerDispatchWrite { 0 };  // Audio thread only
    alignas(64) std::atomic<uint32_t> learnerDispatchRead { 0 };   // Message thread only
    #pragma warning(pop)

    LearningRuntimeState learningRuntimeState = LearningRuntimeState::Idle;
    convo::NoiseShaperLearningMode requestedLearningMode = convo::NoiseShaperLearningMode::Short;
    bool requestedLearningResume = false;
    uint64_t requestedLearningGeneration = 0;
    uint64_t currentIRGeneration = 0; // Audio thread only
    uint64_t pendingIRGeneration = 0; // Message/UI thread only

    std::atomic<int> fixedNoiseLogIntervalMs { 2000 };
    std::atomic<int> fixedNoiseWindowSamples { 8192 };
    std::atomic<bool> softClipEnabled { true };

    #pragma warning(push)
    #pragma warning(disable: 4324)
    alignas(64) std::atomic<float> saturationAmount { 0.1f };
    #pragma warning(pop)

    std::atomic<int> manualOversamplingFactor { 0 }; // 0=Auto, 1=1x, 2=2x, 4=4x, 8=8x
    std::atomic<OversamplingType> oversamplingType { OversamplingType::IIR };

    #pragma warning(push)
    #pragma warning(disable: 4324)
    alignas(64) std::atomic<float> inputHeadroomDb { -6.0f };
    alignas(64) std::atomic<double> inputHeadroomGain { 0.5011872336272722 }; // -6dB
    alignas(64) std::atomic<float> outputMakeupDb { 12.0f };
    alignas(64) std::atomic<double> outputMakeupGain { 3.981071705534972 }; // +12dB
    #pragma warning(pop)

    std::atomic<int> rebuildGeneration { 0 }; // 非同期リビルドの競合防止用
    std::atomic<int> lastCommittedRebuildGeneration { 0 }; // commit 完了済み世代

    #pragma warning(push)
    #pragma warning(disable: 4324)
    alignas(64) std::atomic<float> convolverInputTrimDb { 0.0f };
    alignas(64) std::atomic<double> convolverInputTrimGain { 1.0 }; // 0 dB
    #pragma warning(pop)

    bool m_isRestoringState { false }; // requestLoadState 中はデフォルトリセットを抑制 (Message Thread のみ)
    uint32 fixedNoiseLastLogMs = 0;

    // 出力周波数フィルターモード (Thread-safe)
    std::atomic<convo::HCMode> convHCFilterMode { convo::HCMode::Natural }; // ① ハイカット
    std::atomic<convo::LCMode> convLCFilterMode { convo::LCMode::Natural }; // ① ローカット
    std::atomic<convo::HCMode> eqLPFFilterMode  { convo::HCMode::Natural }; // ② EQローパス

    // dB変換時の下限値
    static constexpr float LEVEL_METER_MIN_DB  = -120.0f;
    static constexpr float LEVEL_METER_MIN_MAG = 1e-6f;

    // EQ応答曲線計算用ワークバッファ (Message Thread/UI Threadで再利用)
    std::array<float, NUM_DISPLAY_BARS> eqTotalMagSqLBuffer;
    std::array<float, NUM_DISPLAY_BARS> eqTotalMagSqRBuffer;
    std::array<float, NUM_DISPLAY_BARS> eqBandMagSqBuffer;
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
    void prepareCommit(DSPCore* newDSP, int generation);
    void executeCommit();
    bool isRebuildObsolete(int generation) const { return generation != rebuildGeneration.load(); }
    bool enqueueLearningCommand(const LearningCommand& cmd) noexcept;
    bool dequeueLearningCommand(LearningCommand& cmd) noexcept;
    bool enqueueLearnerDispatch(const LearnerDispatchAction& action) noexcept;
    bool dequeueLearnerDispatch(LearnerDispatchAction& action) noexcept;
    void processLearningCommands() noexcept;
    void processDeferredLearningActions();
    void resetLearningControlState() noexcept;
    bool enqueueSnapshotCommand() noexcept;
    void processWithSnapshot(const juce::AudioSourceChannelInfo& bufferToFill,
                             const convo::GlobalSnapshot* snap,
                             bool isFadingTarget);
    bool waitForAudioBlockBoundary(uint64_t observedCounter, uint32_t timeoutMs) const noexcept;
    void handleAsyncUpdate() override;

    static void onSnapshotRequired(void* userData, uint64_t generation);
    void createSnapshotFromCurrentState(uint64_t generation);
    void initWorkerThread();
    void shutdownWorkerThread();

    enum class EngineLifecycleState : int
    {
        Unprepared = 0,
        Preparing,
        Prepared,
        Releasing,
        Destroyed
    };

    enum class ShutdownPhase : int
    {
        Running = 0,
        StopAcceptingWork,
        StopAudio,
        StopWorkers,
        ForceEpochAdvance,
        DrainRetire,
        Destroy
    };

    static const char* shutdownPhaseToString(ShutdownPhase phase) noexcept
    {
        switch (phase)
        {
            case ShutdownPhase::Running: return "RUNNING";
            case ShutdownPhase::StopAcceptingWork: return "STOP_ACCEPTING_WORK";
            case ShutdownPhase::StopAudio: return "STOP_AUDIO";
            case ShutdownPhase::StopWorkers: return "STOP_WORKERS";
            case ShutdownPhase::ForceEpochAdvance: return "FORCE_EPOCH_ADVANCE";
            case ShutdownPhase::DrainRetire: return "DRAIN_RETIRE";
            case ShutdownPhase::Destroy: return "DESTROY";
        }
        return "UNKNOWN";
    }

    void setShutdownPhase(ShutdownPhase nextPhase, const char* origin) noexcept
    {
        const ShutdownPhase previous = shutdownPhase.exchange(nextPhase, std::memory_order_acq_rel);
        if (previous == nextPhase)
            return;

        const juce::String log = "[DIAG] shutdown phase: "
            + juce::String(shutdownPhaseToString(previous))
            + " -> "
            + juce::String(shutdownPhaseToString(nextPhase))
            + " at "
            + juce::String(origin != nullptr ? origin : "unknown");
        DBG(log);
        juce::Logger::writeToLog(log);
    }

    // Worker thread for rebuilds
    void rebuildThreadLoop();
    void stopRebuildThread();
    std::thread rebuildThread;
    std::mutex rebuildMutex;
    std::condition_variable rebuildCV;
    std::atomic<bool> rebuildThreadShouldExit { false };
    std::atomic<bool> rebuildThreadIsRunning { false };
    std::atomic<bool> shutdownInProgress { false };
    std::atomic<ShutdownPhase> shutdownPhase { ShutdownPhase::Running };
    std::atomic<EngineLifecycleState> lifecycleState { EngineLifecycleState::Unprepared };
    bool hasPendingTask = false;

    struct RebuildTask {
        DSPCore* currentDSP = nullptr;
        convo::BuildInput buildInput {};
        ConvolverProcessor::BuildSnapshot convolverBuildSnapshot {};
        int generation = 0;
    };
    RebuildTask pendingTask;
    RebuildTask lastQueuedTaskSignature;
    int64_t lastQueuedTaskTicks = 0;

    // --- Commit 2段階化：CommitStaging と deferredCommitQueue ---
    struct CommitStaging {
        DSPCore* newDSP = nullptr;
        DSPCore* oldDSP = nullptr;
        int generation = 0;
    };

    std::queue<CommitStaging> deferredCommitQueue;
    std::mutex deferredCommitMutex;
    convo::TransitionState runtimeTransitionState {};

    std::vector<DSPCore*> deferredDeleteQueue;  // generation mismatch された DSP を defer release

    // --- Adaptiveノイズシェイパー学習用メンバー ---
    struct AdaptiveCoeffBankSlot
    {
        double sampleRateHz = 0.0;
        CoeffSet coeffSetA {};
        CoeffSet coeffSetB {};
        std::atomic<int> activeIndex { 0 };   // 0 = A, 1 = B
        std::atomic<uint32_t> generation { 1u };
        convo::NoiseShaperLearnerState state {};
        std::mutex stateMutex;
        std::atomic<bool> writeLock { false };  // CAS用書き込みロック
    };

    LockFreeRingBuffer<AudioBlock, 4096> audioCaptureQueue;
    std::unique_ptr<NoiseShaperLearner> noiseShaperLearner;
    ThreadAffinityManager affinityManager;
    std::array<AdaptiveCoeffBankSlot, kAdaptiveNoiseShaperSampleRateBankCount * kAdaptiveBitDepthCount * kLearningModeCount> adaptiveCoeffBanks {};
    std::atomic<int> currentAdaptiveCoeffBankIndex { 1 };
    std::mutex adaptiveAutosaveCallbackMutex;
    std::function<void()> adaptiveAutosaveCallback;
    void initialiseAdaptiveCoeffBanks() noexcept;
    static int resolveAdaptiveCoeffBankIndex(double sampleRate) noexcept;
    static int getAdaptiveBitDepthIndex(int bitDepth) noexcept;
    AdaptiveCoeffBankSlot& getAdaptiveCoeffBankForIndex(int bankIndex) noexcept;
    const AdaptiveCoeffBankSlot& getAdaptiveCoeffBankForIndex(int bankIndex) const noexcept;
    void selectAdaptiveCoeffBankForCurrentSettings() noexcept;
    void publishCoeffsToBank(int bankIndex, const double* coeffs);



    void debugAssertNotAudioThread() const;

    inline void publishRuntimeTransitionState(DSPCore* current,
                                              DSPCore* next,
                                              convo::TransitionPolicy policy,
                                              double fadeTimeSec,
                                              bool active,
                                              int latencyDeltaSamples = 0) noexcept
    {
        runtimeTransitionState.current = current;
        runtimeTransitionState.next = next;
        runtimeTransitionState.policy = policy;
        runtimeTransitionState.fadeTimeSec = fadeTimeSec;
        runtimeTransitionState.latencyDeltaSamples = latencyDeltaSamples;
        runtimeTransitionState.active = active;

        debugRuntimeTransitionActive.store(active ? 1 : 0, std::memory_order_release);
        debugRuntimeTransitionPolicy.store(static_cast<int>(policy), std::memory_order_release);
        debugRuntimeTransitionCurrentPtr.store(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(current)), std::memory_order_release);
        debugRuntimeTransitionNextPtr.store(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(next)), std::memory_order_release);
        debugRuntimeTransitionFadeSec.store(fadeTimeSec, std::memory_order_release);
        debugRuntimeTransitionLatencyDeltaSamples.store(latencyDeltaSamples, std::memory_order_release);
    }

    inline convo::RuntimePublishState makeRuntimePublishState(DSPCore* current,
                                                             DSPCore* next,
                                                             convo::TransitionPolicy policy,
                                                             double fadeTimeSec,
                                                             bool active) const noexcept
    {
        convo::RuntimePublishState state {};
        const auto getRuntimeUuid = [](DSPCore* dsp) noexcept -> std::uint64_t
        {
            return dsp != nullptr ? dsp->runtimeUuid : 0;
        };
        DSPCore* fading = sanitizeRawPtr(fadingOutDSP.load(std::memory_order_acquire));
        DSPCore* queuedOld = sanitizeRawPtr(queuedOldDSP.load(std::memory_order_acquire));
        state.current = current;
        state.currentRuntimeUuid = getRuntimeUuid(current);
        state.transition.current = current;
        state.transition.next = next;
        state.transitionCurrentRuntimeUuid = getRuntimeUuid(current);
        state.transitionNextRuntimeUuid = getRuntimeUuid(next);
        state.latencyDelayOld = latencyDelayOld.load(std::memory_order_acquire);
        state.latencyDelayNew = latencyDelayNew.load(std::memory_order_acquire);
        state.transition.policy = policy;
        state.transition.fadeTimeSec = fadeTimeSec;
        state.transition.latencyDeltaSamples = state.latencyDelayOld - state.latencyDelayNew;
        state.transition.active = active;
        state.fading = fading;
        state.fadingRuntimeUuid = getRuntimeUuid(fading);
        state.queuedOld = queuedOld;
        state.queuedOldRuntimeUuid = getRuntimeUuid(queuedOld);
        state.latencyResetPending = latencyResetPending.load(std::memory_order_acquire);
        state.dspCrossfadePending = dspCrossfadePending.load(std::memory_order_acquire);
        state.dspCrossfadeUseDryAsOld = dspCrossfadeUseDryAsOld.load(std::memory_order_acquire);
        state.fadeQueued = fadeQueued.load(std::memory_order_acquire);
        state.queuedFadeTimeSec = queuedFadeTimeSec.load(std::memory_order_acquire);
        state.queuedNextFadeTimeSec = queuedNextFadeTimeSec.load(std::memory_order_acquire);
        state.dspCrossfadeStartDelayBlocks = dspCrossfadeStartDelayBlocks.load(std::memory_order_acquire);
        state.dspCrossfadeDryHoldSamples = dspCrossfadeDryHoldSamples.load(std::memory_order_acquire);
        return state;
    }

    inline const convo::RuntimePublishState* getRuntimePublishState() const noexcept
    {
        return runtimePublishState.load(std::memory_order_acquire);
    }

    inline convo::EngineRuntime makeEngineRuntimeState(const convo::RuntimePublishState& state) const noexcept
    {
        convo::EngineRuntime runtime {};
        runtime.current = state.current;
        runtime.currentRuntimeUuid = state.currentRuntimeUuid;
        runtime.fading = state.fading;
        runtime.fadingRuntimeUuid = state.fadingRuntimeUuid;
        runtime.queuedOld = state.queuedOld;
        runtime.queuedOldRuntimeUuid = state.queuedOldRuntimeUuid;
        runtime.transition = state.transition;
        runtime.transitionCurrentRuntimeUuid = state.transitionCurrentRuntimeUuid;
        runtime.transitionNextRuntimeUuid = state.transitionNextRuntimeUuid;
        runtime.latencyDelayOld = state.latencyDelayOld;
        runtime.latencyDelayNew = state.latencyDelayNew;
        runtime.latencyResetPending = state.latencyResetPending;
        runtime.dspCrossfadePending = state.dspCrossfadePending;
        runtime.dspCrossfadeUseDryAsOld = state.dspCrossfadeUseDryAsOld;
        runtime.fadeQueued = state.fadeQueued;
        runtime.queuedFadeTimeSec = state.queuedFadeTimeSec;
        runtime.queuedNextFadeTimeSec = state.queuedNextFadeTimeSec;
        runtime.dspCrossfadeStartDelayBlocks = state.dspCrossfadeStartDelayBlocks;
        runtime.dspCrossfadeDryHoldSamples = state.dspCrossfadeDryHoldSamples;
        return runtime;
    }

    inline const convo::EngineRuntime* getEngineRuntimeState() const noexcept
    {
        return engineRuntimeState.load(std::memory_order_acquire);
    }

    // =============================
    // Fallback/compatibility layer (Phase-out):
    //  - 旧API。EngineRuntime未移行の読取経路や互換維持のために残す。
    //  - cpp 側からの直接利用がゼロになった時点で削除可能。
    // =============================
    inline bool runtimeCrossfadePending(const convo::RuntimePublishState* runtimePublish,
                                        const convo::EngineRuntime* engineRuntime) const noexcept
    {
        if (engineRuntime != nullptr)
            return engineRuntime->dspCrossfadePending;
        return runtimePublish != nullptr && runtimePublish->dspCrossfadePending;
    }
    inline bool runtimeCrossfadeUseDryAsOld(const convo::RuntimePublishState* runtimePublish,
                                            const convo::EngineRuntime* engineRuntime) const noexcept
    {
        if (engineRuntime != nullptr)
            return engineRuntime->dspCrossfadeUseDryAsOld;
        return runtimePublish != nullptr && runtimePublish->dspCrossfadeUseDryAsOld;
    }
    inline double runtimeQueuedFadeTimeSec(const convo::RuntimePublishState* runtimePublish,
                                           const convo::EngineRuntime* engineRuntime) const noexcept
    {
        if (engineRuntime != nullptr)
            return engineRuntime->queuedFadeTimeSec;
        return runtimePublish != nullptr ? runtimePublish->queuedFadeTimeSec : 0.0;
    }
    inline DSPCore* runtimePublishedCurrentDSP(const convo::RuntimePublishState* runtimePublish,
                                               const convo::EngineRuntime* engineRuntime) const noexcept
    {
        if (engineRuntime != nullptr)
            return static_cast<DSPCore*>(engineRuntime->current);
        return runtimePublish != nullptr ? static_cast<DSPCore*>(runtimePublish->current) : nullptr;
    }
    inline std::uint64_t runtimeRevision(const convo::RuntimePublishState* runtimePublish,
                                         const convo::EngineRuntime* engineRuntime) const noexcept
    {
        if (engineRuntime != nullptr)
            return engineRuntime->revision;
        return runtimePublish != nullptr ? runtimePublish->revision : 0;
    }
    inline std::uint64_t runtimeCurrentUuid(const convo::RuntimePublishState* runtimePublish,
                                            const convo::EngineRuntime* engineRuntime) const noexcept
    {
        if (engineRuntime != nullptr)
            return engineRuntime->currentRuntimeUuid;
        return runtimePublish != nullptr ? runtimePublish->currentRuntimeUuid : 0;
    }
    inline std::uint64_t runtimeFadingUuid(const convo::RuntimePublishState* runtimePublish,
                                           const convo::EngineRuntime* engineRuntime) const noexcept
    {
        if (engineRuntime != nullptr)
            return engineRuntime->fadingRuntimeUuid;
        return runtimePublish != nullptr ? runtimePublish->fadingRuntimeUuid : 0;
    }
    inline std::uint64_t runtimeQueuedOldUuid(const convo::RuntimePublishState* runtimePublish,
                                              const convo::EngineRuntime* engineRuntime) const noexcept
    {
        if (engineRuntime != nullptr)
            return engineRuntime->queuedOldRuntimeUuid;
        return runtimePublish != nullptr ? runtimePublish->queuedOldRuntimeUuid : 0;
    }
    inline std::uint64_t runtimeTransitionCurrentUuid(const convo::RuntimePublishState* runtimePublish,
                                                      const convo::EngineRuntime* engineRuntime) const noexcept
    {
        if (engineRuntime != nullptr)
            return engineRuntime->transitionCurrentRuntimeUuid;
        return runtimePublish != nullptr ? runtimePublish->transitionCurrentRuntimeUuid : 0;
    }
    inline std::uint64_t runtimeTransitionNextUuid(const convo::RuntimePublishState* runtimePublish,
                                                   const convo::EngineRuntime* engineRuntime) const noexcept
    {
        if (engineRuntime != nullptr)
            return engineRuntime->transitionNextRuntimeUuid;
        return runtimePublish != nullptr ? runtimePublish->transitionNextRuntimeUuid : 0;
    }

    inline void publishEngineRuntimeState(const convo::EngineRuntime& state) noexcept
    {
        auto* snapshot = new convo::EngineRuntime(state);
        snapshot->revision = engineRuntimeRevision.fetch_add(1, std::memory_order_acq_rel) + 1;
        auto* old = engineRuntimeState.exchange(snapshot, std::memory_order_acq_rel);
        if (old != nullptr)
            convo::retireObject(old, [](void* p) { delete static_cast<convo::EngineRuntime*>(p); });
    }

    inline DSPCore* resolveCurrentDSPFromRuntimePublish(const convo::RuntimePublishState* runtimePublish) const noexcept
    {
        DSPCore* atomicCurrent = currentDSP.load(std::memory_order_acquire);
        const auto* engineRuntime = getEngineRuntimeState();
        const auto atomicCurrentUuid = atomicCurrent != nullptr ? atomicCurrent->runtimeUuid : 0;

        if (engineRuntime != nullptr)
        {
            auto* publishedCurrent = static_cast<DSPCore*>(engineRuntime->current);
            const auto publishedCurrentUuid = engineRuntime->currentRuntimeUuid;
            if (publishedCurrent != nullptr
                && atomicCurrent != nullptr
                && publishedCurrentUuid != 0
                && publishedCurrentUuid == atomicCurrentUuid)
                return publishedCurrent;

            if (atomicCurrent == nullptr)
                return nullptr;
        }

        if (runtimePublish != nullptr)
        {
            auto* publishedCurrent = static_cast<DSPCore*>(runtimePublish->current);
            const auto publishedCurrentUuid = runtimePublish->currentRuntimeUuid;
            // Lifetime は atomic 側が最終的な事実。publish が stale の場合は atomic を優先する。
            if (publishedCurrent != nullptr
                && atomicCurrent != nullptr
                && publishedCurrentUuid != 0
                && publishedCurrentUuid == atomicCurrentUuid)
                return publishedCurrent;

            if (atomicCurrent == nullptr)
                return nullptr;
        }

        return atomicCurrent;
    }

    inline DSPCore* resolveFadingDSPFromRuntimePublish(const convo::RuntimePublishState* runtimePublish) const noexcept
    {
        DSPCore* atomicFading = sanitizeRawPtr(fadingOutDSP.load(std::memory_order_acquire));
        const auto* engineRuntime = getEngineRuntimeState();
        const auto atomicFadingUuid = atomicFading != nullptr ? atomicFading->runtimeUuid : 0;

        if (engineRuntime != nullptr)
        {
            auto* publishedFading = static_cast<DSPCore*>(engineRuntime->fading);
            const auto publishedFadingUuid = engineRuntime->fadingRuntimeUuid;
            if (publishedFading != nullptr
                && atomicFading != nullptr
                && publishedFadingUuid != 0
                && publishedFadingUuid == atomicFadingUuid)
                return publishedFading;

            if (atomicFading == nullptr)
                return nullptr;
        }

        if (runtimePublish != nullptr)
        {
            auto* publishedFading = static_cast<DSPCore*>(runtimePublish->fading);
            const auto publishedFadingUuid = runtimePublish->fadingRuntimeUuid;
            // fadingOutDSP は audio thread 側で消えるため、publish 単独参照は stale を拾う可能性がある。
            if (publishedFading != nullptr
                && atomicFading != nullptr
                && publishedFadingUuid != 0
                && publishedFadingUuid == atomicFadingUuid)
                return publishedFading;

            if (atomicFading == nullptr)
                return nullptr;
        }

        return atomicFading;
    }

    inline void publishRuntimePublishState(const convo::RuntimePublishState& state) noexcept
    {
        auto* snapshot = new convo::RuntimePublishState(state);
        snapshot->revision = runtimePublishRevision.fetch_add(1, std::memory_order_acq_rel) + 1;
        auto* old = runtimePublishState.exchange(snapshot, std::memory_order_acq_rel);
        g_runtimePublishCount.fetch_add(1, std::memory_order_relaxed);
        if (old != nullptr)
            convo::retireObject(old, [](void* p) { delete static_cast<convo::RuntimePublishState*>(p); });

        // Phase-2 dual-write: keep EngineRuntime snapshot in sync while migrating readers.
        auto engineState = makeEngineRuntimeState(*snapshot);
        publishEngineRuntimeState(engineState);
    }

    // A-6 Phase 1: commit 系の寿命遷移を helper へ集約し、
    // current/fading/queued の更新規約を 1 箇所で追えるようにする。
    inline void publishCurrentDSPAndTakeOwnership(DSPCore* newDSP) noexcept
    {
        currentDSP.store(newDSP, std::memory_order_release);
        activeDSP = newDSP;
    }

    inline void logUnexpectedRuntimeTransition(const char* origin,
                                               DSPCore* current,
                                               DSPCore* candidate) const noexcept
    {
        const auto currentUuid = (current != nullptr) ? current->runtimeUuid : 0;
        const auto candidateUuid = (candidate != nullptr) ? candidate->runtimeUuid : 0;
        const juce::String message = "[DIAG] runtime transition anomaly origin="
            + juce::String(origin != nullptr ? origin : "unknown")
            + " currentUuid=" + juce::String(static_cast<juce::int64>(currentUuid))
            + " candidateUuid=" + juce::String(static_cast<juce::int64>(candidateUuid));
        DBG(message);
        juce::Logger::writeToLog(message);
    }

    inline void logRuntimeTransitionEvent(const char* origin,
                                          DSPCore* primary,
                                          DSPCore* secondary = nullptr) const noexcept
    {
        const auto getUuid = [](DSPCore* dsp) noexcept -> std::uint64_t
        {
            return (dsp != nullptr) ? dsp->runtimeUuid : 0;
        };

        auto* atomicCurrent = currentDSP.load(std::memory_order_acquire);
        auto* fading = sanitizeRawPtr(fadingOutDSP.load(std::memory_order_acquire));
        auto* queued = sanitizeRawPtr(queuedOldDSP.load(std::memory_order_acquire));
        const auto* published = getRuntimePublishState();
        const auto* engineRuntime = getEngineRuntimeState();
        const auto revision = runtimeRevision(published, engineRuntime);
        const auto publishedCurrentUuid = runtimeCurrentUuid(published, engineRuntime);
        const auto publishedFadingUuid = runtimeFadingUuid(published, engineRuntime);
        const auto publishedQueuedUuid = runtimeQueuedOldUuid(published, engineRuntime);

        const juce::String message = "[DIAG] runtime transition event origin="
            + juce::String(origin != nullptr ? origin : "unknown")
            + " primaryUuid=" + juce::String(static_cast<juce::int64>(getUuid(primary)))
            + " secondaryUuid=" + juce::String(static_cast<juce::int64>(getUuid(secondary)))
            + " activeUuid=" + juce::String(static_cast<juce::int64>(getUuid(activeDSP)))
            + " atomicCurrentUuid=" + juce::String(static_cast<juce::int64>(getUuid(atomicCurrent)))
            + " fadingUuid=" + juce::String(static_cast<juce::int64>(getUuid(fading)))
            + " queuedUuid=" + juce::String(static_cast<juce::int64>(getUuid(queued)))
            + " publishRev=" + juce::String(static_cast<juce::int64>(revision))
            + " publishCurrentUuid=" + juce::String(static_cast<juce::int64>(publishedCurrentUuid))
            + " publishFadingUuid=" + juce::String(static_cast<juce::int64>(publishedFadingUuid))
            + " publishQueuedUuid=" + juce::String(static_cast<juce::int64>(publishedQueuedUuid));
        DBG(message);
        juce::Logger::writeToLog(message);
    }

    inline bool validateDistinctRuntimeSlots(const char* origin,
                                             DSPCore* active,
                                             DSPCore* fading,
                                             DSPCore* queued) const noexcept
    {
        const bool activeEqualsFading = (active != nullptr && active == fading);
        const bool activeEqualsQueued = (active != nullptr && active == queued);
        const bool fadingEqualsQueued = (fading != nullptr && fading == queued);

        if (!activeEqualsFading && !activeEqualsQueued && !fadingEqualsQueued)
            return true;

        const auto getUuid = [](DSPCore* dsp) noexcept -> std::uint64_t
        {
            return (dsp != nullptr) ? dsp->runtimeUuid : 0;
        };

        const juce::String message = "[DIAG] runtime slot overlap origin="
            + juce::String(origin != nullptr ? origin : "unknown")
            + " activeUuid=" + juce::String(static_cast<juce::int64>(getUuid(active)))
            + " fadingUuid=" + juce::String(static_cast<juce::int64>(getUuid(fading)))
            + " queuedUuid=" + juce::String(static_cast<juce::int64>(getUuid(queued)));
        DBG(message);
        juce::Logger::writeToLog(message);
        jassert(!activeEqualsFading && !activeEqualsQueued && !fadingEqualsQueued);
        return false;
    }

    // Audio Thread でも使えるように、ログ出力を行わない軽量版。
    inline bool validateDistinctRuntimeSlotsRT(DSPCore* active,
                                               DSPCore* fading,
                                               DSPCore* queued) const noexcept
    {
        const bool activeEqualsFading = (active != nullptr && active == fading);
        const bool activeEqualsQueued = (active != nullptr && active == queued);
        const bool fadingEqualsQueued = (fading != nullptr && fading == queued);

        jassert(!activeEqualsFading && !activeEqualsQueued && !fadingEqualsQueued);
        return !activeEqualsFading && !activeEqualsQueued && !fadingEqualsQueued;
    }

    inline void replaceQueuedOldDSPAndRetirePrevious(DSPCore* dsp) noexcept
    {
        validateDistinctRuntimeSlots("replaceQueuedOldDSPAndRetirePrevious.before",
                                     activeDSP,
                                     sanitizeRawPtr(fadingOutDSP.load(std::memory_order_acquire)),
                                     sanitizeRawPtr(queuedOldDSP.load(std::memory_order_acquire)));

        if (auto* prev = sanitizeRawPtr(queuedOldDSP.exchange(dsp, std::memory_order_acq_rel)))
        {
            if (prev == dsp)
            {
                logUnexpectedRuntimeTransition("replaceQueuedOldDSPAndRetirePrevious", prev, dsp);
                jassert(prev != dsp);
                return;
            }

            retireDSP(prev);
        }

        validateDistinctRuntimeSlots("replaceQueuedOldDSPAndRetirePrevious.after",
                                     activeDSP,
                                     sanitizeRawPtr(fadingOutDSP.load(std::memory_order_acquire)),
                                     sanitizeRawPtr(queuedOldDSP.load(std::memory_order_acquire)));
        logRuntimeTransitionEvent("replaceQueuedOldDSPAndRetirePrevious", dsp);
    }

    inline void replaceFadingOutDSPAndRetirePrevious(DSPCore* dsp) noexcept
    {
        validateDistinctRuntimeSlots("replaceFadingOutDSPAndRetirePrevious.before",
                                     activeDSP,
                                     sanitizeRawPtr(fadingOutDSP.load(std::memory_order_acquire)),
                                     sanitizeRawPtr(queuedOldDSP.load(std::memory_order_acquire)));

        if (auto* prev = sanitizeRawPtr(fadingOutDSP.exchange(dsp, std::memory_order_acq_rel)))
        {
            if (prev == dsp)
            {
                logUnexpectedRuntimeTransition("replaceFadingOutDSPAndRetirePrevious", prev, dsp);
                jassert(prev != dsp);
                return;
            }

            retireDSP(prev);
        }

        validateDistinctRuntimeSlots("replaceFadingOutDSPAndRetirePrevious.after",
                                     activeDSP,
                                     sanitizeRawPtr(fadingOutDSP.load(std::memory_order_acquire)),
                                     sanitizeRawPtr(queuedOldDSP.load(std::memory_order_acquire)));
        logRuntimeTransitionEvent("replaceFadingOutDSPAndRetirePrevious", dsp);
    }

    inline void retireRuntimeImmediately(DSPCore* dsp) noexcept
    {
        if (dsp == nullptr)
            return;

        const auto* published = getRuntimePublishState();
        const auto* engineRuntime = getEngineRuntimeState();
        auto* publishedCurrent = runtimePublishedCurrentDSP(published, engineRuntime);
        auto* atomicCurrent = currentDSP.load(std::memory_order_acquire);
        if (dsp == activeDSP || dsp == atomicCurrent || dsp == publishedCurrent)
        {
            logUnexpectedRuntimeTransition("retireRuntimeImmediately", atomicCurrent, dsp);
            jassert(dsp != activeDSP && dsp != atomicCurrent && dsp != publishedCurrent);
            return;
        }

        logRuntimeTransitionEvent("retireRuntimeImmediately", dsp);
        retireDSP(dsp);
    }

    inline void publishSmoothTransitionState(DSPCore* nextDSP,
                                             DSPCore* previousDSP,
                                             double fadeTimeSec,
                                             int latencyDeltaSamples) noexcept
    {
        if (nextDSP == nullptr || nextDSP == previousDSP)
        {
            logUnexpectedRuntimeTransition("publishSmoothTransitionState", nextDSP, previousDSP);
            jassert(nextDSP != nullptr && nextDSP != previousDSP);
        }

        publishRuntimeTransitionState(nextDSP,
                                      previousDSP,
                                      convo::TransitionPolicy::SmoothOnly,
                                      fadeTimeSec,
                                      true,
                                      latencyDeltaSamples);
        logRuntimeTransitionEvent("publishSmoothTransitionState", nextDSP, previousDSP);
    }

    inline void queueDeferredSmoothTransition(DSPCore* previousDSP,
                                              double fadeTimeSec) noexcept
    {
        if (previousDSP == nullptr || previousDSP == activeDSP)
        {
            logUnexpectedRuntimeTransition("queueDeferredSmoothTransition", activeDSP, previousDSP);
            jassert(previousDSP != nullptr && previousDSP != activeDSP);
        }

        replaceQueuedOldDSPAndRetirePrevious(previousDSP);
        queuedNextFadeTimeSec.store(fadeTimeSec, std::memory_order_release);
        fadeQueued.store(true, std::memory_order_release);
        publishRuntimePublishState(makeRuntimePublishState(activeDSP,
                                                           previousDSP,
                                                           convo::TransitionPolicy::SmoothOnly,
                                                           fadeTimeSec,
                                                           true));
        validateDistinctRuntimeSlots("queueDeferredSmoothTransition",
                                     activeDSP,
                                     sanitizeRawPtr(fadingOutDSP.load(std::memory_order_acquire)),
                                     sanitizeRawPtr(queuedOldDSP.load(std::memory_order_acquire)));
        logRuntimeTransitionEvent("queueDeferredSmoothTransition", activeDSP, previousDSP);
    }

    inline void startImmediateSmoothTransition(DSPCore* previousDSP,
                                               double fadeTimeSec) noexcept
    {
        if (previousDSP == nullptr || previousDSP == activeDSP)
        {
            logUnexpectedRuntimeTransition("startImmediateSmoothTransition", activeDSP, previousDSP);
            jassert(previousDSP != nullptr && previousDSP != activeDSP);
        }

        replaceFadingOutDSPAndRetirePrevious(previousDSP);
        queuedFadeTimeSec.store(fadeTimeSec, std::memory_order_release);
        dspCrossfadePending.store(true, std::memory_order_release);
        setIRChangeFlag();
        publishRuntimePublishState(makeRuntimePublishState(activeDSP,
                                                           previousDSP,
                                                           convo::TransitionPolicy::SmoothOnly,
                                                           fadeTimeSec,
                                                           true));
        validateDistinctRuntimeSlots("startImmediateSmoothTransition",
                                     activeDSP,
                                     sanitizeRawPtr(fadingOutDSP.load(std::memory_order_acquire)),
                                     sanitizeRawPtr(queuedOldDSP.load(std::memory_order_acquire)));
        logRuntimeTransitionEvent("startImmediateSmoothTransition", activeDSP, previousDSP);
    }

    inline void publishHardResetForCurrentDSP() noexcept
    {
        if (activeDSP == nullptr)
        {
            logUnexpectedRuntimeTransition("publishHardResetForCurrentDSP", nullptr, nullptr);
            jassert(activeDSP != nullptr);
        }

        publishRuntimePublishState(makeRuntimePublishState(activeDSP,
                                                           nullptr,
                                                           convo::TransitionPolicy::HardReset,
                                                           0.0,
                                                           false));
        publishRuntimeTransitionState(activeDSP,
                                      nullptr,
                                      convo::TransitionPolicy::HardReset,
                                      0.0,
                                      false,
                                      0);
        validateDistinctRuntimeSlots("publishHardResetForCurrentDSP",
                                     activeDSP,
                                     sanitizeRawPtr(fadingOutDSP.load(std::memory_order_acquire)),
                                     sanitizeRawPtr(queuedOldDSP.load(std::memory_order_acquire)));
        logRuntimeTransitionEvent("publishHardResetForCurrentDSP", activeDSP);
    }

    inline void armDryAsOldCrossfadeForCurrentDSP(double fadeTimeSec,
                                                  int latencyDeltaSamples,
                                                  double targetIrScale) noexcept
    {
        if (activeDSP == nullptr)
        {
            logUnexpectedRuntimeTransition("armDryAsOldCrossfadeForCurrentDSP", nullptr, nullptr);
            jassert(activeDSP != nullptr);
        }

        queuedFadeTimeSec.store(fadeTimeSec, std::memory_order_release);
        publishRuntimeTransitionState(activeDSP,
                                      nullptr,
                                      convo::TransitionPolicy::DryAsOld,
                                      fadeTimeSec,
                                      true,
                                      latencyDeltaSamples);
        dspCrossfadeDryHoldSamples.store(std::max(1, maxSamplesPerBlock.load(std::memory_order_acquire)), std::memory_order_release);
        dspCrossfadeDryScaleGain.reset(std::max(1.0, currentSampleRate.load(std::memory_order_acquire)), 0.060);
        dspCrossfadeDryScaleGain.setCurrentAndTargetValue(1.0);
        dspCrossfadeDryScaleGain.setTargetValue(targetIrScale);
        firstIrDryCrossfadePending.store(true, std::memory_order_release);
        dspCrossfadePending.store(true, std::memory_order_release);
        firstIrDryCrossfadeDone.store(true, std::memory_order_release);
        setIRChangeFlag();
        publishRuntimePublishState(makeRuntimePublishState(activeDSP,
                                                           nullptr,
                                                           convo::TransitionPolicy::DryAsOld,
                                                           fadeTimeSec,
                                                           true));
        validateDistinctRuntimeSlots("armDryAsOldCrossfadeForCurrentDSP",
                                     activeDSP,
                                     sanitizeRawPtr(fadingOutDSP.load(std::memory_order_acquire)),
                                     sanitizeRawPtr(queuedOldDSP.load(std::memory_order_acquire)));
        logRuntimeTransitionEvent("armDryAsOldCrossfadeForCurrentDSP", activeDSP);
    }

    inline EngineParameterSnapshot captureAudioThreadParameterSnapshot(const convo::GlobalSnapshot* snap,
                                                                       bool isFadingTarget = false) const noexcept
    {
        EngineParameterSnapshot snapshot {};
        snapshot.eqBypassed = (snap != nullptr) ? snap->eqBypass : eqBypassRequested.load(std::memory_order_acquire);
        snapshot.convBypassed = (snap != nullptr) ? snap->convBypass : convBypassRequested.load(std::memory_order_acquire);
        snapshot.order = (snap != nullptr) ? snap->processingOrder : currentProcessingOrder.load(std::memory_order_relaxed);
        snapshot.softClipEnabled = (snap != nullptr) ? snap->softClipEnabled : softClipEnabled.load(std::memory_order_relaxed);
        snapshot.saturationAmount = (snap != nullptr) ? snap->saturationAmount : saturationAmount.load(std::memory_order_relaxed);
        snapshot.inputHeadroomGain = (snap != nullptr) ? snap->inputHeadroomGain : inputHeadroomGain.load(std::memory_order_relaxed);
        snapshot.outputMakeupGain = (snap != nullptr) ? snap->outputMakeupGain : outputMakeupGain.load(std::memory_order_relaxed);
        snapshot.convolverInputTrimGain = (snap != nullptr) ? snap->convInputTrimGain : convolverInputTrimGain.load(std::memory_order_relaxed);
        snapshot.analyzerSource = currentAnalyzerSource.load(std::memory_order_relaxed);
        snapshot.analyzerEnabled = isFadingTarget ? false : analyzerEnabled.load(std::memory_order_relaxed);
        snapshot.convHCMode = convHCFilterMode.load(std::memory_order_relaxed);
        snapshot.convLCMode = convLCFilterMode.load(std::memory_order_relaxed);
        snapshot.eqLPFMode = eqLPFFilterMode.load(std::memory_order_relaxed);
        snapshot.adaptiveCoeffBankIndex = currentAdaptiveCoeffBankIndex.load(std::memory_order_acquire);
        const auto& adaptiveCoeffBank = getAdaptiveCoeffBankForIndex(snapshot.adaptiveCoeffBankIndex);
        snapshot.adaptiveCoeffGeneration = adaptiveCoeffBank.generation.load(std::memory_order_acquire);
        snapshot.adaptiveCoeffSet = getActiveCoeffSet(adaptiveCoeffBank);
        snapshot.adaptiveCaptureEnabled = isNoiseShaperLearning();
        return snapshot;
    }

    inline DSPCore::ProcessingState buildAudioThreadProcessingState(DSPCore* dsp,
                                                                     const EngineParameterSnapshot& snapshot) noexcept
    {
        return DSPCore::ProcessingState {
            .eqBypassed = snapshot.eqBypassed,
            .convBypassed = snapshot.convBypassed,
            .order = snapshot.order,
            .analyzerSource = snapshot.analyzerSource,
            .analyzerEnabled = snapshot.analyzerEnabled,
            .softClipEnabled = snapshot.softClipEnabled,
            .saturationAmount = snapshot.saturationAmount,
            .inputHeadroomGain = snapshot.inputHeadroomGain,
            .outputMakeupGain = snapshot.outputMakeupGain,
            .convolverInputTrimGain = snapshot.convolverInputTrimGain,
            .convHCMode = snapshot.convHCMode,
            .convLCMode = snapshot.convLCMode,
            .eqLPFMode = snapshot.eqLPFMode,
            .adaptiveCoeffBankIndex = snapshot.adaptiveCoeffBankIndex,
            .adaptiveCoeffSet = snapshot.adaptiveCoeffSet,
            .adaptiveCoeffGeneration = snapshot.adaptiveCoeffGeneration,
            .adaptiveCaptureSampleRateHz = static_cast<int>(dsp->sampleRate + 0.5),
            .adaptiveCaptureBitDepth = dsp->ditherBitDepth,
            .captureSessionId = dsp->currentCaptureSessionId,
            .adaptiveCaptureQueue = snapshot.adaptiveCaptureEnabled ? &audioCaptureQueue : nullptr
        };
    }

    inline bool updateAudioThreadSnapshotFade(int numSamples,
                                              float& snapshotAlpha,
                                              const convo::GlobalSnapshot*& snapshotFrom,
                                              const convo::GlobalSnapshot*& snapshotTo) noexcept
    {
        if (m_coordinator.isFading())
            m_coordinator.advanceFade(numSamples);

        const bool updateFadeReturned = m_coordinator.updateFade(snapshotAlpha, snapshotFrom, snapshotTo);
        return updateFadeReturned;
    }

    inline void applySafeSilentFallback(const juce::AudioSourceChannelInfo& bufferToFill) noexcept
    {
        inputLevelLinear.store(0.0f, std::memory_order_relaxed);
        outputLevelLinear.store(0.0f, std::memory_order_relaxed);
        bufferToFill.clearActiveBufferRegion();
    }

    inline void applySafeSilentFallback(juce::AudioBuffer<double>& buffer) noexcept
    {
        inputLevelLinear.store(0.0f, std::memory_order_relaxed);
        outputLevelLinear.store(0.0f, std::memory_order_relaxed);
        buffer.clear();
    }

    inline void armCrossfadeIfPending(DSPCore* dsp,
                                      bool hasFading,
                                      bool& useDryAsOld,
                                      const convo::RuntimePublishState* runtimePublish) noexcept
    {
        const auto* engineRuntime = getEngineRuntimeState();
        const bool atomicPendingCrossfade = dspCrossfadePending.load(std::memory_order_acquire);
        const bool hasPendingCrossfade = atomicPendingCrossfade
            || runtimeCrossfadePending(runtimePublish, engineRuntime);

        if ((hasFading || firstIrDryCrossfadePending.load(std::memory_order_acquire))
            && hasPendingCrossfade
            && dspCrossfadePending.exchange(false, std::memory_order_acq_rel))
        {
            const double atomicQueuedFadeSec = queuedFadeTimeSec.load(std::memory_order_acquire);
            const double fadeSecFromPublish = runtimeQueuedFadeTimeSec(runtimePublish, engineRuntime);
            const double fadeSec = std::max(0.001,
                atomicQueuedFadeSec > 0.0 ? atomicQueuedFadeSec : fadeSecFromPublish);
            dspCrossfadeGain.reset(std::max(1.0, dsp->sampleRate), fadeSec);
            dspCrossfadeGain.setCurrentAndTargetValue(0.0);
            dspCrossfadeGain.setTargetValue(1.0);

            latencyDelayOld_RT = latencyDelayOld.load(std::memory_order_acquire);
            latencyDelayNew_RT = latencyDelayNew.load(std::memory_order_acquire);

            if (firstIrDryCrossfadePending.exchange(false, std::memory_order_acq_rel))
            {
                dspCrossfadeUseDryAsOld.store(true, std::memory_order_release);
                useDryAsOld = true;
            }
        }
    }

    template <typename ProcessFn>
    inline bool processCrossfadeDelayGateIfPending(DSPCore* fading,
                                                   bool useDryAsOld,
                                                   bool hasPendingCrossfade,
                                                   int pendingFadeDelayBlocks,
                                                   ProcessFn processFn) noexcept
    {
        if (fading != nullptr
            && !useDryAsOld
            && hasPendingCrossfade
            && pendingFadeDelayBlocks > 0)
        {
            dspCrossfadeStartDelayBlocks.store(pendingFadeDelayBlocks - 1, std::memory_order_release);
            processFn();
            return true;
        }

        return false;
    }

    static inline DSPCore::ProcessingState makeCrossfadeAuxState(const DSPCore::ProcessingState& procState) noexcept
    {
        auto fadingState = procState;
        fadingState.analyzerEnabled = false;
        fadingState.adaptiveCaptureQueue = nullptr;
        return fadingState;
    }

    inline void finalizeCrossfadeMixPath(bool resetDryScaleGain) noexcept
    {
        if (!dspCrossfadeGain.isSmoothing())
        {
            validateDistinctRuntimeSlotsRT(
                sanitizeRawPtr(currentDSP.load(std::memory_order_acquire)),
                sanitizeRawPtr(fadingOutDSP.load(std::memory_order_acquire)),
                sanitizeRawPtr(queuedOldDSP.load(std::memory_order_acquire)));

            if (auto* done = sanitizeRawPtr(fadingOutDSP.exchange(nullptr, std::memory_order_acq_rel)))
                retireDSP(done);

            validateDistinctRuntimeSlotsRT(
                sanitizeRawPtr(currentDSP.load(std::memory_order_acquire)),
                sanitizeRawPtr(fadingOutDSP.load(std::memory_order_acquire)),
                sanitizeRawPtr(queuedOldDSP.load(std::memory_order_acquire)));

            dspCrossfadeGain.setCurrentAndTargetValue(1.0);
            if (resetDryScaleGain)
                dspCrossfadeDryScaleGain.setCurrentAndTargetValue(1.0);
            dspCrossfadeUseDryAsOld.store(false, std::memory_order_release);
        }
    }

    inline void cleanupCrossfadeDirectPath(DSPCore* fading) noexcept
    {
        if (fading != nullptr && !dspCrossfadeGain.isSmoothing())
        {
            validateDistinctRuntimeSlotsRT(
                sanitizeRawPtr(currentDSP.load(std::memory_order_acquire)),
                sanitizeRawPtr(fadingOutDSP.load(std::memory_order_acquire)),
                sanitizeRawPtr(queuedOldDSP.load(std::memory_order_acquire)));

            if (auto* done = sanitizeRawPtr(fadingOutDSP.exchange(nullptr, std::memory_order_acq_rel)))
                retireDSP(done);

            validateDistinctRuntimeSlotsRT(
                sanitizeRawPtr(currentDSP.load(std::memory_order_acquire)),
                sanitizeRawPtr(fadingOutDSP.load(std::memory_order_acquire)),
                sanitizeRawPtr(queuedOldDSP.load(std::memory_order_acquire)));
        }
        if (!dspCrossfadeGain.isSmoothing())
            dspCrossfadeUseDryAsOld.store(false, std::memory_order_release);
    }

    inline void resetLatencyBuffersIfPending(int bufferSize, int& writePos) noexcept
    {
        if (latencyResetPending.exchange(false, std::memory_order_acq_rel))
        {
            if (latencyBufOldL) std::memset(latencyBufOldL, 0, sizeof(double) * bufferSize);
            if (latencyBufOldR) std::memset(latencyBufOldR, 0, sizeof(double) * bufferSize);
            if (latencyBufNewL) std::memset(latencyBufNewL, 0, sizeof(double) * bufferSize);
            if (latencyBufNewR) std::memset(latencyBufNewR, 0, sizeof(double) * bufferSize);
            writePos = 0;
        }
    }

    inline int estimateRuntimeLatencyBaseRateSamples(const DSPCore* dsp,
                                                     bool convolverBypassed) const noexcept
    {
        if (dsp == nullptr)
            return 0;

        const double baseSampleRate = currentSampleRate.load(std::memory_order_acquire) > 0.0
            ? currentSampleRate.load(std::memory_order_acquire)
            : dsp->sampleRate;

        const int osFactor = std::max(1, static_cast<int>(dsp->oversamplingFactor));
        const auto toBaseRateSamples = [osFactor](int processingRateSamples) -> int
        {
            return juce::jmax(0,
                static_cast<int>(std::lround(static_cast<double>(processingRateSamples)
                    / static_cast<double>(osFactor))));
        };

        int totalLatency = juce::jmax(0,
            static_cast<int>(std::lround(estimateOversamplingLatencySamples(
                osFactor,
                dsp->activeOversamplingType,
                baseSampleRate))));

        if (!convolverBypassed)
        {
            const auto convBreakdown = dsp->convolver.getLatencyBreakdown();
            totalLatency += toBaseRateSamples(convBreakdown.totalLatencySamples);
        }

        return juce::jmax(0, totalLatency);
    }

    static inline int wrapLatencyIndex(int index, int bufferSize) noexcept
    {
        while (index < 0) index += bufferSize;
        while (index >= bufferSize) index -= bufferSize;
        return index;
    }

    template <typename SampleType, typename MixFn>
    inline void runLatencyAlignedCrossfadeMixLoop(SampleType* dstL,
                                                  SampleType* dstR,
                                                  const SampleType* oldL,
                                                  const SampleType* oldR,
                                                  int numSamples,
                                                  MixFn mixFn) noexcept
    {
        const int bufferSize = latencyBufSize;
        int writePos = latencyWritePos;
        const int delayOld = latencyDelayOld_RT;
        const int delayNew = latencyDelayNew_RT;

        resetLatencyBuffersIfPending(bufferSize, writePos);

        if (numSamples > 0)
        {
            const int firstReadOld = wrapLatencyIndex(writePos - delayOld, bufferSize);
            const int firstReadNew = wrapLatencyIndex(writePos - delayNew, bufferSize);
            debugLatencyAlignWritePos.store(writePos, std::memory_order_release);
            debugLatencyAlignReadOld.store(firstReadOld, std::memory_order_release);
            debugLatencyAlignReadNew.store(firstReadNew, std::memory_order_release);
            debugLatencyAlignDelayOld.store(delayOld, std::memory_order_release);
            debugLatencyAlignDelayNew.store(delayNew, std::memory_order_release);
        }

        for (int i = 0; i < numSamples; ++i)
        {
            latencyBufOldL[writePos] = (oldL != nullptr) ? static_cast<double>(oldL[i]) : 0.0;
            latencyBufOldR[writePos] = (oldR != nullptr) ? static_cast<double>(oldR[i]) : 0.0;
            latencyBufNewL[writePos] = (dstL != nullptr) ? static_cast<double>(dstL[i]) : 0.0;
            latencyBufNewR[writePos] = (dstR != nullptr) ? static_cast<double>(dstR[i]) : 0.0;

            const int readOld = wrapLatencyIndex(writePos - delayOld, bufferSize);
            const int readNew = wrapLatencyIndex(writePos - delayNew, bufferSize);

            const double alignedOldL = latencyBufOldL[readOld];
            const double alignedOldR = latencyBufOldR[readOld];
            const double alignedNewL = latencyBufNewL[readNew];
            const double alignedNewR = latencyBufNewR[readNew];

            const double gNew = dspCrossfadeGain.getNextValue();
            mixFn(dstL, dstR, i, gNew, alignedOldL, alignedOldR, alignedNewL, alignedNewR);

            ++writePos;
            if (writePos >= bufferSize)
                writePos = 0;
        }

        latencyWritePos = writePos;
    }

    friend class NoiseShaperLearner;
    friend class EQEditProcessor;

//==============================================================================
// インラインヘルパー関数（Adaptive 係数アクセス）
//==============================================================================

// Audio Thread 用：現在アクティブな係数セットを取得（ロックフリー）
static inline const CoeffSet* getActiveCoeffSet(const AdaptiveCoeffBankSlot& slot) noexcept
{
    return (slot.activeIndex.load(std::memory_order_acquire) == 0)
           ? &slot.coeffSetA
           : &slot.coeffSetB;
}

// 書き込み側用：非アクティブバッファの予約（CAS）
static inline bool reserveInactiveCoeffSet(AdaptiveCoeffBankSlot& slot) noexcept
{
    bool expected = false;
    return slot.writeLock.compare_exchange_strong(expected, true,
                                                   std::memory_order_acquire,
                                                   std::memory_order_relaxed);
}

// 書き込み側用：予約した非アクティブセットへのポインタ取得
static inline CoeffSet* getReservedInactiveCoeffSet(AdaptiveCoeffBankSlot& slot) noexcept
{
    int active = slot.activeIndex.load(std::memory_order_acquire);
    return (active == 0) ? &slot.coeffSetB : &slot.coeffSetA;
}

// DSP ポインタのセンチネル値（全ビット 1）をチェックし、無効値なら nullptr を返す
template <typename T>
static inline T* sanitizeRawPtr(T* ptr) noexcept
{
    constexpr uintptr_t kInvalidAllOnes = ~static_cast<uintptr_t>(0);
    return (reinterpret_cast<uintptr_t>(ptr) == kInvalidAllOnes) ? nullptr : ptr;
}

// DSPCore の解放は必ず EBR retire キュー経由で行う
static inline void retireDSP(DSPCore* dsp) noexcept
{
    if (dsp)
    {
        g_runtimeRetireCount.fetch_add(1, std::memory_order_relaxed);
        convo::retireObject(dsp, [](void* p) { delete static_cast<DSPCore*>(p); });
    }
}

//==============================================================================
// RAII ガードクラス（例外安全性確保＋commit() 最適化）
//==============================================================================
class CoeffSetWriteLockGuard
{
public:
    explicit CoeffSetWriteLockGuard(AdaptiveCoeffBankSlot& s) noexcept
        : slot(s), acquired(false), committed(false) {}

    ~CoeffSetWriteLockGuard() noexcept
    {
        // commit() が呼ばれていない場合のみ、ロックを解放
        if (acquired && !committed)
            slot.writeLock.store(false, std::memory_order_release);
    }

    bool acquire() noexcept
    {
        bool expected = false;
        acquired = slot.writeLock.compare_exchange_strong(expected, true,
                                                           std::memory_order_acquire,
                                                           std::memory_order_relaxed);
        return acquired;
    }

    // commit() を呼ぶことで、デストラクタでのロック解放をスキップ
    void commit() noexcept
    {
        if (!acquired || committed)
            return;

        int oldActive = slot.activeIndex.load(std::memory_order_relaxed);
        slot.activeIndex.store(1 - oldActive, std::memory_order_release);
        slot.generation.fetch_add(1u, std::memory_order_acq_rel);
        slot.writeLock.store(false, std::memory_order_release);
        committed = true;
    }

    bool isAcquired() const noexcept { return acquired; }
    bool isCommitted() const noexcept { return committed; }

    CoeffSetWriteLockGuard(const CoeffSetWriteLockGuard&) = delete;
    CoeffSetWriteLockGuard& operator=(const CoeffSetWriteLockGuard&) = delete;

    private:
        AdaptiveCoeffBankSlot& slot;
        bool acquired;
        bool committed;
    };

    // リリースキューに溜まったエントリを解放可能なものから処理する
    void processDeferredReleases();

    // ==================================================================
    // スナップショット基盤（Phase 2）
    // ==================================================================
    convo::EpochCore m_epochCore;
    convo::SnapshotCoordinator m_coordinator;
    GenerationManager m_generationManager;

    // ==================================================================
    // Phase 3: コマンドバッファ + ワーカースレッド
    // ==================================================================
    convo::RuntimeCommandQueue m_runtimeCommandQueue;
    convo::CommandBuffer m_commandBuffer;
    convo::WorkerThread m_workerThread;
    std::atomic<std::uint64_t> debugRebuildDispatchRequestCount { 0 };
    std::atomic<std::uint64_t> debugRebuildDispatchQueuedCount { 0 };
    std::atomic<std::uint64_t> debugRebuildDispatchBlockedPendingDuplicateCount { 0 };
    std::atomic<std::uint64_t> debugRebuildDispatchBlockedRecentDuplicateCount { 0 };
    std::atomic<std::uint64_t> debugRebuildDispatchRuntimeQueueFullCount { 0 };
    std::atomic<std::uint64_t> debugRebuildDispatchDrainedCommandCount { 0 };
    std::atomic<std::uint64_t> debugRebuildDispatchMatchedRuntimeCommandCount { 0 };
    std::atomic<std::uint64_t> debugRebuildDispatchTaskSnapshotFallbackCount { 0 };

    std::atomic<float> m_currentInputHeadroomDb { -6.0f };
    std::atomic<float> m_currentOutputMakeupDb { 12.0f };
    std::atomic<float> m_currentConvInputTrimDb { 0.0f };
    std::atomic<bool> m_currentEqBypass { false };
    std::atomic<bool> m_currentConvBypass { false };
    std::atomic<convo::ProcessingOrder> m_currentProcessingOrder { convo::ProcessingOrder::ConvolverThenEQ };
    std::atomic<bool> m_currentSoftClipEnabled { true };
    std::atomic<float> m_currentSaturationAmount { 0.1f };
    std::atomic<int> m_currentOversamplingFactor { 0 };
    std::atomic<convo::OversamplingType> m_currentOversamplingType { convo::OversamplingType::IIR };
    std::atomic<int> m_currentDitherBitDepth { 24 };
    std::atomic<convo::NoiseShaperType> m_currentNoiseShaperType { convo::NoiseShaperType::Psychoacoustic };

    static constexpr int DEFAULT_IR_FADE_SAMPLES = 2048;
    static constexpr int DEFAULT_EQ_FADE_SAMPLES = 256;
    static constexpr int DEFAULT_NS_FADE_SAMPLES = 1024;
    static constexpr int DEFAULT_AGC_FADE_SAMPLES = 128;
    std::atomic<int> m_irFadeSamples{ DEFAULT_IR_FADE_SAMPLES };
    std::atomic<int> m_eqFadeSamples{ DEFAULT_EQ_FADE_SAMPLES };
    std::atomic<int> m_nsFadeSamples{ DEFAULT_NS_FADE_SAMPLES };
    std::atomic<int> m_agcFadeSamples{ DEFAULT_AGC_FADE_SAMPLES };
    std::atomic<bool> m_pendingIRChange{ false };
    std::atomic<bool> m_pendingNSChange{ false };
    std::atomic<bool> m_pendingAGCChange{ false };
    std::atomic<uint64_t> m_audioBlockCounter{ 0 };

    std::array<convo::EQParameters, 2> latestEqParamsForFallback{};
    std::array<uint64_t, 2> latestEqHashForFallback{ 0, 0 };
    std::atomic<int> latestEqFallbackReadIndex{ 0 };

    // Audio Thread -> Message Thread 反映確認用 (RT安全: atomic store/fetch_add のみ)
    std::atomic<uint64_t> debugLastCreatedEqHash{ 0 };
    std::atomic<uint64_t> debugLastCreateAudioBlockCounter{ 0 };
    std::atomic<uint64_t> debugLastAppliedEqHash{ 0 };
    std::atomic<uint32_t> debugAppliedEqHashVersion{ 0 };
    std::atomic<uint64_t> lastEnqueuedSnapshotDebounceKey_{ 0 };
    std::atomic<bool> hasLastEnqueuedSnapshotDebounceKey_{ false };
    uint32_t debugObservedEqHashVersion{ 0 }; // timerCallback (Message Thread) 専用
    uint64_t debugLastReportedCreatedEqHash{ std::numeric_limits<uint64_t>::max() }; // timerCallback 専用
    uint64_t debugLastReportedAppliedEqHash{ std::numeric_limits<uint64_t>::max() }; // timerCallback 専用
    int debugLastReportedDspReady{ -1 }; // timerCallback 専用
    uint64_t debugLastRecoveryAttemptCreatedEqHash{ 0 }; // timerCallback 専用
    uint64_t debugLastRecoveryAttemptAudioBlockCounter{ 0 }; // timerCallback 専用
    int debugRecoveryRetryCountForCurrentHash{ 0 }; // timerCallback 専用
    bool debugRecoverySuppressedForCurrentHash{ false }; // timerCallback 専用
    std::atomic<int> debugRuntimeTransitionActive{ 0 };
    std::atomic<int> debugRuntimeTransitionPolicy{ 0 };
    std::atomic<uint64_t> debugRuntimeTransitionCurrentPtr{ 0 };
    std::atomic<uint64_t> debugRuntimeTransitionNextPtr{ 0 };
    std::atomic<double> debugRuntimeTransitionFadeSec{ 0.0 };
    std::atomic<int> debugRuntimeTransitionLatencyDeltaSamples{ 0 };
    std::atomic<int> debugLatencyAlignWritePos{ 0 };
    std::atomic<int> debugLatencyAlignReadOld{ 0 };
    std::atomic<int> debugLatencyAlignReadNew{ 0 };
    std::atomic<int> debugLatencyAlignDelayOld{ 0 };
    std::atomic<int> debugLatencyAlignDelayNew{ 0 };
    int debugLastReportedTransitionActive{ -1 }; // timerCallback 専用
    int debugLastReportedTransitionPolicy{ -1 }; // timerCallback 専用
    uint64_t debugLastReportedTransitionCurrentPtr{ std::numeric_limits<uint64_t>::max() }; // timerCallback 専用
    uint64_t debugLastReportedTransitionNextPtr{ std::numeric_limits<uint64_t>::max() }; // timerCallback 専用
    double debugLastReportedTransitionFadeSec{ -1.0 }; // timerCallback 専用
    int debugLastReportedTransitionLatencyDeltaSamples{ std::numeric_limits<int>::max() }; // timerCallback 専用
    int debugLastReportedLatencyAlignWritePos{ std::numeric_limits<int>::max() }; // timerCallback 専用
    int debugLastReportedLatencyAlignReadOld{ std::numeric_limits<int>::max() }; // timerCallback 専用
    int debugLastReportedLatencyAlignReadNew{ std::numeric_limits<int>::max() }; // timerCallback 専用
    int debugLastReportedLatencyAlignDelayOld{ std::numeric_limits<int>::max() }; // timerCallback 専用
    int debugLastReportedLatencyAlignDelayNew{ std::numeric_limits<int>::max() }; // timerCallback 専用
    uint64_t debugLastReportedRuntimePublishCount{ std::numeric_limits<uint64_t>::max() }; // timerCallback 専用
    uint64_t debugLastReportedRuntimeRetireCount{ std::numeric_limits<uint64_t>::max() }; // timerCallback 専用
    uint64_t debugLastReportedRuntimeReclaimCount{ std::numeric_limits<uint64_t>::max() }; // timerCallback 専用
    uint64_t debugLastReportedRebuildRequestCount{ std::numeric_limits<uint64_t>::max() }; // timerCallback 専用
    uint64_t debugLastReportedRebuildQueuedCount{ std::numeric_limits<uint64_t>::max() }; // timerCallback 専用
    uint64_t debugLastReportedRebuildBlockedPendingDuplicateCount{ std::numeric_limits<uint64_t>::max() }; // timerCallback 専用
    uint64_t debugLastReportedRebuildBlockedRecentDuplicateCount{ std::numeric_limits<uint64_t>::max() }; // timerCallback 専用
    uint64_t debugLastReportedRebuildRuntimeQueueFullCount{ std::numeric_limits<uint64_t>::max() }; // timerCallback 専用
    uint64_t debugLastReportedRebuildDrainedCommandCount{ std::numeric_limits<uint64_t>::max() }; // timerCallback 専用
    uint64_t debugLastReportedRebuildMatchedRuntimeCommandCount{ std::numeric_limits<uint64_t>::max() }; // timerCallback 専用
    uint64_t debugLastReportedRebuildTaskSnapshotFallbackCount{ std::numeric_limits<uint64_t>::max() }; // timerCallback 専用
    uint64_t debugLastReportedRuntimePublishRevision{ std::numeric_limits<uint64_t>::max() }; // timerCallback 専用
    uint64_t debugLastReportedRuntimePublishCurrentUuid{ std::numeric_limits<uint64_t>::max() }; // timerCallback 専用
    uint64_t debugLastReportedRuntimePublishFadingUuid{ std::numeric_limits<uint64_t>::max() }; // timerCallback 専用
    uint64_t debugLastReportedRuntimePublishQueuedOldUuid{ std::numeric_limits<uint64_t>::max() }; // timerCallback 専用
    uint64_t debugLastReportedRuntimePublishTransitionCurrentUuid{ std::numeric_limits<uint64_t>::max() }; // timerCallback 専用
    uint64_t debugLastReportedRuntimePublishTransitionNextUuid{ std::numeric_limits<uint64_t>::max() }; // timerCallback 専用
    uint64_t debugLastReportedConvolverRebuildRequestCount{ std::numeric_limits<uint64_t>::max() }; // timerCallback 専用
    uint64_t debugLastReportedConvolverRebuildDeferredAfterLoadCount{ std::numeric_limits<uint64_t>::max() }; // timerCallback 専用
    uint64_t debugLastReportedConvolverRebuildScheduledCount{ std::numeric_limits<uint64_t>::max() }; // timerCallback 専用
    uint64_t debugLastReportedConvolverRebuildTriggeredCount{ std::numeric_limits<uint64_t>::max() }; // timerCallback 専用
    int debugLastReportedShutdownPhase{ -1 }; // timerCallback 専用

    juce::AudioBuffer<float> m_fadeFloatBuffer;
    juce::AudioBuffer<double> m_fadeDoubleBuffer;

    // ==================================================================

    JUCE_DECLARE_WEAK_REFERENCEABLE(AudioEngine)
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AudioEngine)
};

inline bool AudioEngine::enqueueLearningCommand(const LearningCommand& cmd) noexcept
{
    const uint32_t currentWrite = learningCommandWrite.load(std::memory_order_relaxed);
    const uint32_t currentRead = learningCommandRead.load(std::memory_order_acquire);
    const uint32_t next = (currentWrite + 1u) & learningCommandBufferMask;
    if (next == currentRead)
    {
        jassertfalse;
        return false;
    }

    learningCommandBuffer[currentWrite] = cmd;
    std::atomic_thread_fence(std::memory_order_release);
    learningCommandWrite.store(next, std::memory_order_release);
    return true;
}

inline bool AudioEngine::dequeueLearningCommand(LearningCommand& cmd) noexcept
{
    const uint32_t currentRead = learningCommandRead.load(std::memory_order_relaxed);
    const uint32_t currentWrite = learningCommandWrite.load(std::memory_order_acquire);
    if (currentRead == currentWrite)
        return false;

    std::atomic_thread_fence(std::memory_order_acquire);
    cmd = learningCommandBuffer[currentRead];
    learningCommandRead.store((currentRead + 1u) & learningCommandBufferMask, std::memory_order_release);
    return true;
}

inline bool AudioEngine::enqueueLearnerDispatch(const LearnerDispatchAction& action) noexcept
{
    const uint32_t currentWrite = learnerDispatchWrite.load(std::memory_order_relaxed);
    const uint32_t currentRead = learnerDispatchRead.load(std::memory_order_acquire);
    const uint32_t next = (currentWrite + 1u) & learnerDispatchBufferMask;
    if (next == currentRead)
    {
        lastFailedAction.store(action, std::memory_order_release);
        learnerDispatchOverflow.store(true, std::memory_order_release);
        return false;
    }

    learnerDispatchBuffer[currentWrite] = action;
    std::atomic_thread_fence(std::memory_order_release);
    learnerDispatchWrite.store(next, std::memory_order_release);
    learnerDispatchOverflow.store(false, std::memory_order_relaxed);
    return true;
}

inline bool AudioEngine::dequeueLearnerDispatch(LearnerDispatchAction& action) noexcept
{
    const uint32_t currentRead = learnerDispatchRead.load(std::memory_order_relaxed);
    const uint32_t currentWrite = learnerDispatchWrite.load(std::memory_order_acquire);
    if (currentRead == currentWrite)
        return false;

    std::atomic_thread_fence(std::memory_order_acquire);
    action = learnerDispatchBuffer[currentRead];
    learnerDispatchRead.store((currentRead + 1u) & learnerDispatchBufferMask, std::memory_order_release);
    return true;
}
