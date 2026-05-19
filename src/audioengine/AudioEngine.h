
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
#include <type_traits>
#include <vector>
#include <utility>
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
#include "AtomicAccess.h"
#include "DeferredDeletionQueue.h"
#include "core/Types.h"
#include "core/SnapshotCoordinator.h"
#include "core/EpochDomain.h"
#include "core/RuntimePublicationCoordinator.h"
#include "core/RuntimeStore.h"
#include "core/CommandBuffer.h"
#include "core/ThreadAffinityManager.h"
#include "core/WorkerThread.h"
#include "core/RebuildTypes.h"

class NoiseShaperLearner;
class AudioEngine;

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

struct RuntimeState
{
    convo::EngineRuntime engine {};
    convo::RuntimeGraph graph {};
    std::uint64_t generation = 0;
    std::uint64_t runtimeVersion = 0;  // Monotonically increasing version number
    std::uint64_t transitionId = 0;    // Unique identifier per crossfade
};

using RuntimePublishWorld = RuntimeState;

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
                jassert(ownerEngine != nullptr);
                proc.setRcuProvider(*ownerEngine);
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
            int fadeInSamplesLeft = 0;
            convo::LinearRamp bypassFadeGainDouble;
            bool bypassedDouble = false;

            void prepare(double sampleRate) noexcept
            {
                bypassFadeGainDouble.reset(sampleRate, 0.005);
                bypassFadeGainDouble.setCurrentAndTargetValue(1.0);
                bypassedDouble = false;
                fadeInSamplesLeft = 0;
            }

            void resetForRuntime() noexcept
            {
                bypassFadeGainDouble.setCurrentAndTargetValue(1.0);
                bypassedDouble = false;
                fadeInSamplesLeft = 0;
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
                    auto newDelayL = convo::makeAlignedArray<double>(static_cast<size_t>(requiredSize));
                    auto newDelayR = convo::makeAlignedArray<double>(static_cast<size_t>(requiredSize));

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
             std::atomic<float>* inputLevelLinear,
             std::atomic<float>* outputLevelLinear, const ProcessingState& state);
    void processToBuffer(const juce::AudioSourceChannelInfo& source,
                         juce::AudioBuffer<float>& destination,
                 LockFreeAudioRingBuffer& analyzerFifo,
                 std::atomic<float>* inputLevelLinear,
                 std::atomic<float>* outputLevelLinear,
                         const ProcessingState& state);
    void processDouble(juce::AudioBuffer<double>& buffer,
                   LockFreeAudioRingBuffer& analyzerFifo,
                   std::atomic<float>* inputLevelLinear,
                   std::atomic<float>* outputLevelLinear,
                       const ProcessingState& state);
    void processDoubleToBuffer(const juce::AudioBuffer<double>& source,
                               juce::AudioBuffer<double>& destination,
                       LockFreeAudioRingBuffer& analyzerFifo,
                       std::atomic<float>* inputLevelLinear,
                       std::atomic<float>* outputLevelLinear,
                               const ProcessingState& state);
        ConvolverProcessor convolver;
        EQProcessor eq;
        // DC blocker state is detached from DSP graph members.
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

    //----------------------------------------------------------
    // AudioSource インターフェース
    //----------------------------------------------------------
    void prepareToPlay (int samplesPerBlockExpected, double sampleRate) override;
    void releaseResources() override;
    void getNextAudioBlock (const juce::AudioSourceChannelInfo& bufferToFill) override;

    // ==================================================================
    // EBR (Epoch-Based Reclamation) 基盤
    // ==================================================================
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
    ThreadAffinityManager& getAffinityManager() noexcept { return affinityManager; }
    const ThreadAffinityManager& getAffinityManager() const noexcept { return affinityManager; }

    // ========================================================
    // EQ Parameter Wrappers (Thread-safe delegation to uiEqEditor)
    // ========================================================
    void setEQBandGain(int band, float gainDb) { uiEqEditor.setBandGain(band, gainDb); }
    void setEQBandFrequency(int band, float freq) { uiEqEditor.setBandFrequency(band, freq); }
    void setEQBandQ(int band, float q) { uiEqEditor.setBandQ(band, q); }
    void setEQBandEnabled(int band, bool enabled) { uiEqEditor.setBandEnabled(band, enabled); }
    void setEQBandType(int band, EQBandType type) { uiEqEditor.setBandType(band, type); }
    void setEQBandChannelMode(int band, EQChannelMode mode) { uiEqEditor.setBandChannelMode(band, mode); }
    void setEQTotalGain(float gainDb) { uiEqEditor.setTotalGain(gainDb); }
    void setEQAGCEnabled(bool enabled) { uiEqEditor.setAGCEnabled(enabled); }
    void setEQNonlinearSaturation(float value) noexcept { uiEqEditor.setNonlinearSaturation(value); }
    void setEQFilterStructure(EQProcessor::FilterStructure mode) noexcept { uiEqEditor.setFilterStructure(mode); }

    EQBandParams getEQBandParams(int band) const { return uiEqEditor.getBandParams(band); }
    EQBandType getEQBandType(int band) const { return uiEqEditor.getBandType(band); }
    EQChannelMode getEQBandChannelMode(int band) const { return uiEqEditor.getBandChannelMode(band); }
    float getEQTotalGain() const { return uiEqEditor.getTotalGain(); }
    bool isEQAGCEnabled() const { return uiEqEditor.getAGCEnabled(); }
    float getEQNonlinearSaturation() const noexcept { return uiEqEditor.getNonlinearSaturation(); }
    EQProcessor::FilterStructure getEQFilterStructure() const noexcept { return uiEqEditor.getFilterStructure(); }
    const void* getEQStateSnapshot() const { return uiEqEditor.getEQStateSnapshot(); }
    void resetEQToDefaults() { uiEqEditor.reset(); }

    // acquire: prepareToPlay/releaseResources の release と HB し、有効なサンプルレートを取得。
    double getSampleRate() const { return consumeAtomic(currentSampleRate, std::memory_order_acquire); }
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

    // 【Fix Bug #8】gainToDecibels (std::log10 / libm) を Audio Thread から排除。
    // Audio Thread は linear gain を inputLevelLinear / outputLevelLinear に格納し、
    // getter (UI Thread) で dB 変換する。
    // acquire: Audio Thread の release publishAtomic (inputLevelLinear/outputLevelLinear) と HB
    //          し、最新のレベル値を UI Thread から安全に取得する。
    float getInputLevel() const
    {
        const float linear = consumeAtomic(inputLevelLinear, std::memory_order_acquire);
        return (linear > LEVEL_METER_MIN_MAG)
               ? juce::Decibels::gainToDecibels(linear)
               : LEVEL_METER_MIN_DB;
    }
    float getOutputLevel() const
    {
        const float linear = consumeAtomic(outputLevelLinear, std::memory_order_acquire);
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
    // 以下 4 getter は acquire: 対応 setter の release publishAtomic/compareExchange と HB し、
    // Message Thread から bypass 状態の最新値を安全に観測する。
    bool isEqBypassRequested() const noexcept { return consumeAtomic(eqBypassRequested, std::memory_order_acquire); }
    bool isConvolverBypassRequested() const noexcept { return consumeAtomic(convBypassRequested, std::memory_order_acquire); }
    bool isEQBypassed() const noexcept { return consumeAtomic(eqBypassActive, std::memory_order_acquire); }
    bool isConvolverBypassed() const noexcept { return consumeAtomic(convBypassActive, std::memory_order_acquire); }

    void setConvolverPhaseMode(ConvolverProcessor::PhaseMode mode);
    ConvolverProcessor::PhaseMode getConvolverPhaseMode() const;

    void requestEqPreset (int presetIndex);
    void requestEqPresetFromText(const juce::File& file);
    void requestConvolverPreset (const juce::File& irFile);

    void requestLoadState (const juce::ValueTree& state);
    juce::ValueTree getCurrentState() const;
    void beginBulkParameterRestore() noexcept;
    void endBulkParameterRestore(bool requestRebuildNow = true) noexcept;

    void setProcessingOrder(ProcessingOrder order);
    // acquire: setProcessingOrder の release と HB し、最新の ProcessingOrder を観測。
    ProcessingOrder getProcessingOrder() const { return consumeAtomic(currentProcessingOrder, std::memory_order_acquire); }

    // release: UI スレッドからの設定を Audio Thread が acquire で観測できるよう公開。
    // acquire: 対応 setter の release と HB し、最新値を取得。
    void setAnalyzerSource(AnalyzerSource source) { publishAtomic(currentAnalyzerSource, source, std::memory_order_release); }
    AnalyzerSource getAnalyzerSource() const { return consumeAtomic(currentAnalyzerSource, std::memory_order_acquire); }
    void setAnalyzerEnabled(bool enabled) noexcept { publishAtomic(analyzerEnabled, enabled, std::memory_order_release); }
    bool isAnalyzerEnabled() const noexcept { return consumeAtomic(analyzerEnabled, std::memory_order_acquire); }

    void setInputHeadroomDb(float db);
    float getInputHeadroomDb() const;

    void setOutputMakeupDb(float db);
    float getOutputMakeupDb() const;

    void setConvolverInputTrimDb(float db);
    float getConvolverInputTrimDb() const;

    // Audio Thread command queue 経路を廃止し、
    // Message Thread 上の UI staging -> snapshot/rebuild 経路に統一する。
    void setConvolverMix(float value) noexcept
    {
        ASSERT_NON_RT_THREAD();
        uiConvolverProcessor.setMix(value);
        enqueueSnapshotCommand();
    }

    void setConvolverSmoothingTime(float timeSec) noexcept
    {
        ASSERT_NON_RT_THREAD();
        uiConvolverProcessor.setSmoothingTime(timeSec);
        enqueueSnapshotCommand();
    }

    void setConvolverTargetIRLength(float timeSec, bool manualOverride = false) noexcept;
    void setConvolverMixedTransitionStartHz(float hz) noexcept;
    void setConvolverMixedTransitionEndHz(float hz) noexcept;
    void setConvolverMixedPreRingTau(float tau) noexcept;
    void setConvolverRebuildDebounceMs(int ms) noexcept;
    // Tail* setter wrappers migrated to pendingOverride (ConvolverProcessor direct only)

    // Convolver State Tree & Cache Settings
    juce::ValueTree getConvolverStateTree() const;
    void setConvolverStateTree(const juce::ValueTree& state);
    int getConvolverTargetUpgradeFFTSize() const;
    void setConvolverTargetUpgradeFFTSize(int fftSize);
    bool isConvolverProgressiveUpgradeEnabled() const;
    void setConvolverEnableProgressiveUpgrade(bool enabled);
    int getConvolverMaxCacheEntries() const;
    void setConvolverMaxCacheEntries(int maxEntries);
    void clearConvolverCache();

    void setDitherBitDepth(int bitDepth);
    int getDitherBitDepth() const;

    void setNoiseShaperType(NoiseShaperType type);
    NoiseShaperType getNoiseShaperType() const;
    void requestSnapshotForNoiseShaper();
    void requestRebuild(convo::RebuildKind kind) noexcept;
    void setFixedNoiseLogIntervalMs(int intervalMs) noexcept;
    int getFixedNoiseLogIntervalMs() const noexcept;
    void setFixedNoiseWindowSamples(int windowSamples) noexcept;
    int getFixedNoiseWindowSamples() const noexcept;

    void setIRFadeSamples(int samples) noexcept
    {
        const int clamped = juce::jlimit(0, 96000, samples);
        // release: m_irFadeSamples の更新を Audio Thread が acquire で観測できるよう公開。
        publishAtomic(m_irFadeSamples, clamped, std::memory_order_release);

        // acquire: prepareToPlay release と HB し、有効な currentSampleRate を取得。
        const double sr = consumeAtomic(currentSampleRate, std::memory_order_acquire);
        if (sr > 0.0)
        {
            const double fadeSec = (clamped > 0)
                ? (static_cast<double>(clamped) / sr)
                : 0.001;
            // release: m_irFadeTimeSec の更新を Audio Thread が acquire で観測できるよう公開。
            publishAtomic(m_irFadeTimeSec, fadeSec, std::memory_order_release);
        }
    }
    // release: 各 setXxx の release と HB する getter は Audio Thread が acquire で観測。
    void setEQFadeSamples(int samples) noexcept { publishAtomic(m_eqFadeSamples, samples, std::memory_order_release); }
    int getIRFadeSamples() const noexcept { return consumeAtomic(m_irFadeSamples, std::memory_order_acquire); }
    int getEQFadeSamples() const noexcept { return consumeAtomic(m_eqFadeSamples, std::memory_order_acquire); }
    bool isFading() const noexcept { return m_coordinator.isFading(); }
    convo::ObservedRuntime observeCurrentRuntime() const noexcept
    {
        return m_coordinator.observeCurrentRuntime(kControlEpochReaderIndex);
    }
    // release: IR 変更フラグを Audio Thread の acquire で観測できるよう公開。
    void setIRChangeFlag() noexcept { publishAtomic(m_pendingIRChange, true, std::memory_order_release); }

    // acquire: Audio Thread の release (debugLastCreatedEqHash publish) と HB し、診断ハッシュを取得。
    uint64_t getLastCreatedEqHashForDebug() const noexcept { return consumeAtomic(debugLastCreatedEqHash, std::memory_order_acquire); }

    const convo::EQParameters& getLatestEqParamsFallback(uint64_t& outHash) const noexcept
    {
        // acquire: latestEqFallbackReadIndex の release publish と HB し、最新のインデックスを取得。
        const int index = consumeAtomic(latestEqFallbackReadIndex, std::memory_order_acquire);
        outHash = latestEqHashForFallback[(size_t) index];
        return latestEqParamsForFallback[(size_t) index];
    }

    void setSoftClipEnabled(bool enabled);
    bool isSoftClipEnabled() const;

    void setSaturationAmount(float amount);
    float getSaturationAmount() const;

    bool isShutdownInProgress() const noexcept
    {
        // acquire: lifecycleState の setShutdownPhase/releaseResources release 側と HB し、
        //          Releasing/Destroyed 遷移を各スレッドから安全に観測する。
        const auto state = consumeAtomic(lifecycleState, std::memory_order_acquire);
        return state == EngineLifecycleState::Releasing
            || state == EngineLifecycleState::Destroyed;
    }

    void drainDeferredRetireQueues(bool allowDuringShutdown) noexcept;

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
    // acquire: setNoiseShaperLearningMode の release と HB し、最新の LearningMode を取得。
    convo::NoiseShaperLearningMode getNoiseShaperLearningMode() const { return consumeAtomic(pendingLearningMode, std::memory_order_acquire); }
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
            consumeAtomic(g_runtimePublishCount),
            consumeAtomic(g_runtimeRetireCount),
            consumeAtomic(g_runtimeReclaimCount)
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
            consumeAtomic(debugRebuildDispatchRequestCount),
            consumeAtomic(debugRebuildDispatchQueuedCount),
            consumeAtomic(debugRebuildDispatchBlockedPendingDuplicateCount),
            consumeAtomic(debugRebuildDispatchBlockedRecentDuplicateCount),
            consumeAtomic(debugRebuildDispatchRuntimeQueueFullCount),
            consumeAtomic(debugRebuildDispatchDrainedCommandCount),
            consumeAtomic(debugRebuildDispatchMatchedRuntimeCommandCount),
            consumeAtomic(debugRebuildDispatchTaskSnapshotFallbackCount)
        };
    }

    // --- NoiseShaperLearner Settings ---
    convo::NoiseShaperLearnerSettings getNoiseShaperLearnerSettings() const;
    void setNoiseShaperLearnerSettings(const convo::NoiseShaperLearnerSettings& settings);

    static double getAdaptiveSampleRateBankHz(int bankIndex) noexcept;
    void getCurrentAdaptiveCoefficients(double* outCoeffs, int maxCoefficients) const noexcept;
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
    static constexpr int kAudioEpochReaderIndex = 0;
    static constexpr int kControlEpochReaderIndex = 1;
    static constexpr int kCommitProducerEpochReaderIndex = 2;
    static constexpr int kCommitConsumerEpochReaderIndex = 3;

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
    int dspCrossfadeStartDelayBlocks_RT = 0;
    bool dspCrossfadeArmed_RT = false;
    // バッファリセット要求（MessageThread→AudioThread）
    std::atomic<bool> latencyResetPending { false };
    static constexpr int kMaxLatencySamples = 1536000; // 最大2秒@768kHz対応
    static constexpr int MAX_LATENCY_ALIGN_SAMPLES = 96000 * 2; // 2秒@48kHz
    class EQCacheManager
    {
    public:
        explicit EQCacheManager(AudioEngine& ownerIn) noexcept;
        EQCoeffCache* getOrCreate(const convo::EQParameters& params,
                                  double sampleRate,
                                  int maxBlockSize,
                                  uint64_t generation);
        EQCoeffCache* get(uint64_t hash) noexcept;
        void releaseCache(EQCoeffCache* cache) noexcept;
        ~EQCacheManager();

    private:
        struct CacheMap
        {
            explicit CacheMap(AudioEngine& ownerIn) noexcept
                : owner(&ownerIn)
            {
            }

            CacheMap(const CacheMap& other)
                : owner(other.owner)
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
                jassert(owner != nullptr);
                for (auto& entry : map)
                {
                    if (entry.second != nullptr)
                        entry.second->release(owner->epochDomain());
                }
            }

            AudioEngine* owner = nullptr;
            std::unordered_map<uint64_t, EQCoeffCache*> map;
        };

        const CacheMap* loadMap() noexcept
        {
            return AudioEngine::consumeAtomicPtr(cacheMapPtr);
        }

        void storeNewMap(CacheMap* newMap) noexcept;
        void drainDeferredMapsUnderLock() noexcept;
        bool tryEnqueueDeferredMap(CacheMap* map) noexcept;

        AudioEngine& owner;
        std::mutex writeMutex;
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
    convo::NonOwningPtr<DSPCore> activeDSP { nullptr }; // Non-RT ownership holder
    convo::NonOwningPtr<DSPCore> fadingOutDSP { nullptr }; // Non-RT fading slot holder

    inline DSPCore* loadCurrentDSP(std::memory_order order = std::memory_order_acquire) const noexcept
    {
        juce::ignoreUnused(order);
        return activeDSP.get();
    }

    inline DSPCore* exchangeCurrentDSP(DSPCore* value,
                                       std::memory_order order = std::memory_order_acq_rel) noexcept
    {
        juce::ignoreUnused(order);
        DSPCore* previous = activeDSP.get();
        activeDSP = value;
        return previous;
    }

    inline void publishCurrentDSP(DSPCore* value,
                                  std::memory_order order = std::memory_order_release) noexcept
    {
        juce::ignoreUnused(order);
        activeDSP = value;
    }

    inline DSPCore* loadFadingOutDSP(std::memory_order order = std::memory_order_acquire) const noexcept
    {
        juce::ignoreUnused(order);
        return fadingOutDSP.get();
    }

    inline DSPCore* exchangeFadingOutDSP(DSPCore* value,
                                         std::memory_order order = std::memory_order_acq_rel) noexcept
    {
        juce::ignoreUnused(order);
        DSPCore* previous = fadingOutDSP.get();
        fadingOutDSP = value;
        return previous;
    }

    inline void publishFadingOutDSP(DSPCore* value,
                                    std::memory_order order = std::memory_order_release) noexcept
    {
        juce::ignoreUnused(order);
        fadingOutDSP = value;
    }
    struct RuntimePublishView
    {
        const convo::RuntimeGraph* graph = nullptr;
    };

    struct CrossfadePreparedSnapshot
    {
        bool pending = false;
        bool useDryAsOld = false;
        bool firstIrDryCrossfadePending = false;
        double fadeTimeSec = 0.0;
        int latencyDelayOld = 0;
        int latencyDelayNew = 0;
        int startDelayBlocks = 0;
        int dryHoldSamples = 0;
        bool latencyResetPending = false;
        double dryScaleTarget = 1.0;       // dry-as-old crossfade の IR スケール目標値
    };

    class RuntimePublicationBridge;
    using DspCoreHandle = DSPCore*;

    std::atomic<std::uint64_t> runtimeGraphRevision { 0 };
    std::array<CrossfadePreparedSnapshot, 2> crossfadePreparedSnapshots_ {};
    std::atomic<int> crossfadePreparedSnapshotIndex_ { 0 };

        std::atomic<double> queuedFadeTimeSec { 0.030 };      // 現在開始するフェード時間

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
    std::atomic<double> dspCrossfadeDryScaleTarget { 1.0 }; // dry-as-old crossfade の IR スケール目標値
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

    enum class RebuildReason : uint32_t
    {
        None = 0u,
        StructuralFromNonMT = 1u << 0,
        DeferredStructural = 1u << 1,
        DeferredFinalizeAware = 1u << 2
    };

    std::atomic<uint32_t> rebuildReasonFlags_ { 0u };
    std::atomic<int64_t> deferredStructuralRebuildDueTicks_ { 0 };
    std::atomic<bool> pendingChangeNotification { false };
    // 同一IR構造に対する Structural rebuild の多重発火を抑止する。
    // 値は「直近で rebuild を要求した UI 側 Convolver 構造ハッシュ」。
    std::atomic<uint64_t> lastIssuedConvolverStructuralHash_{ 0 };
    // activeDSP 実体を直接読まずに判定するための、commit済み Convolver 構造スナップショット。
    std::atomic<uint64_t> lastCommittedConvolverStructuralHash_{ 0 };
    std::atomic<bool> lastCommittedConvolverHasIr_{ false };
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

    #pragma warning(push) // C4324 suppression scope begin: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
    #pragma warning(disable : 4324) // Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
    alignas(64) std::atomic<convo::NoiseShaperLearningMode> pendingLearningMode { convo::NoiseShaperLearningMode::Short };
    alignas(64) std::atomic<uint64_t> globalCaptureSessionId { 1 };
    #pragma warning(pop) // C4324 suppression scope end: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容

    LearningCommand learningCommandBuffer[learningCommandBufferSize] {};
    LearnerDispatchAction learnerDispatchBuffer[learnerDispatchBufferSize] {};
    std::atomic<bool> learnerDispatchOverflow { false };
    std::atomic<LearnerDispatchAction> lastFailedAction {};

    #pragma warning(push) // C4324 suppression scope begin: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
    #pragma warning(disable : 4324) // Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
    alignas(64) std::atomic<uint32_t> learningCommandWrite { 0 };  // Message/UI thread only
    alignas(64) std::atomic<uint32_t> learningCommandRead { 0 };   // Audio thread only
    alignas(64) std::atomic<uint32_t> learnerDispatchWrite { 0 };  // Audio thread only
    alignas(64) std::atomic<uint32_t> learnerDispatchRead { 0 };   // Message thread only
    #pragma warning(pop) // C4324 suppression scope end: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容

    LearningRuntimeState learningRuntimeState = LearningRuntimeState::Idle;
    convo::NoiseShaperLearningMode requestedLearningMode = convo::NoiseShaperLearningMode::Short;
    bool requestedLearningResume = false;
    uint64_t requestedLearningGeneration = 0;
    uint64_t currentIRGeneration = 0; // Audio thread only
    uint64_t pendingIRGeneration = 0; // Message/UI thread only

    std::atomic<int> fixedNoiseLogIntervalMs { 2000 };
    std::atomic<int> fixedNoiseWindowSamples { 8192 };
    std::atomic<bool> softClipEnabled { true };

    #pragma warning(push) // C4324 suppression scope begin: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
    #pragma warning(disable : 4324) // Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
    alignas(64) std::atomic<float> saturationAmount { 0.1f };
    #pragma warning(pop) // C4324 suppression scope end: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容

    std::atomic<int> manualOversamplingFactor { 0 }; // 0=Auto, 1=1x, 2=2x, 4=4x, 8=8x
    std::atomic<OversamplingType> oversamplingType { OversamplingType::IIR };

    #pragma warning(push) // C4324 suppression scope begin: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
    #pragma warning(disable : 4324) // Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
    alignas(64) std::atomic<float> inputHeadroomDb { -6.0f };
    alignas(64) std::atomic<double> inputHeadroomGain { 0.5011872336272722 }; // -6dB
    alignas(64) std::atomic<float> outputMakeupDb { 12.0f };
    alignas(64) std::atomic<double> outputMakeupGain { 3.981071705534972 }; // +12dB
    #pragma warning(pop) // C4324 suppression scope end: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容

    std::atomic<int> rebuildGeneration { 0 }; // 非同期リビルドの競合防止用
    std::atomic<int> lastCommittedRebuildGeneration { 0 }; // commit 完了済み世代

    #pragma warning(push) // C4324 suppression scope begin: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
    #pragma warning(disable : 4324) // Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
    alignas(64) std::atomic<float> convolverInputTrimDb { 0.0f };
    alignas(64) std::atomic<double> convolverInputTrimGain { 1.0 }; // 0 dB
    #pragma warning(pop) // C4324 suppression scope end: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容

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
    // acquire: commitNewDSP/requestRebuild の rebuildGeneration 更新 release と HB し、
    //          リビルド世代が古いか否かを各スレッドから安全に判定。
    bool isRebuildObsolete(int generation) const { return generation != consumeAtomic(rebuildGeneration, std::memory_order_acquire); }
    bool enqueueLearningCommand(const LearningCommand& cmd) noexcept;
    bool dequeueLearningCommand(LearningCommand& cmd) noexcept;
    bool enqueueLearnerDispatch(const LearnerDispatchAction& action) noexcept;
    bool dequeueLearnerDispatch(LearnerDispatchAction& action) noexcept;
    void processLearningCommands() noexcept;
    void processDeferredLearningActions();
    void resetLearningControlState() noexcept;
    bool enqueueSnapshotCommand() noexcept;
    void appendPublicationIntent(DSPCore* newDSP, int generation, int epochReaderIndex) noexcept;
    void drainPublicationLogForShutdown() noexcept;
    bool hasPendingPublicationIntents() noexcept;
    bool hasPublicationLogPending() noexcept;
    void processWithSnapshot(const juce::AudioSourceChannelInfo& bufferToFill,
                             const convo::GlobalSnapshot* snap,
                             bool isFadingTarget);
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
        // acq_rel: 取得側 acquire で前フェーズの操作を観測し、
        //          解放側 release で新フェーズを全スレッドに公開。
        const ShutdownPhase previous = exchangeAtomic(shutdownPhase, nextPhase, std::memory_order_acq_rel);
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

    // --- Commit 2段階化：PublicationLog と commit staging ---
    struct PublicationIntent {
        DSPCore* newDSP = nullptr;
        int generation = 0;
        std::atomic<PublicationIntent*> next { nullptr };
    };

    struct PublicationLog {
        std::atomic<PublicationIntent*> head { nullptr };
        std::atomic<PublicationIntent*> consumedTail { nullptr };
        std::atomic<PublicationIntent*> retiredHead { nullptr };
    };

    PublicationLog publicationLog;
    PublicationIntent* publicationLogSentinel = nullptr;
    std::atomic<bool> commitDrainInProgress { false };
    struct DeferredDeleteFallbackEntry
    {
        void* ptr = nullptr;
        void (*deleter)(void*) = nullptr;
        uint64_t epoch = 0;
    };
    std::mutex deferredDeleteFallbackMutex;
    std::vector<DeferredDeleteFallbackEntry> deferredDeleteFallbackQueue;

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
    static int resolveAdaptiveCoeffBankIndex(double sampleRate) noexcept;
    static int getAdaptiveBitDepthIndex(int bitDepth) noexcept;
    AdaptiveCoeffBankSlot& getAdaptiveCoeffBankForIndex(int bankIndex) noexcept;
    const AdaptiveCoeffBankSlot& getAdaptiveCoeffBankForIndex(int bankIndex) const noexcept;
    void selectAdaptiveCoeffBankForCurrentSettings() noexcept;
    void publishCoeffsToBank(int bankIndex, const double* coeffs);

    static constexpr uint32_t rebuildReasonMask(RebuildReason reason) noexcept
    {
        return static_cast<uint32_t>(reason);
    }

    bool hasRebuildReason(RebuildReason reason) const noexcept
    {
        // acquire: setRebuildReason の acq_rel release-side と HB し、最新の flags を観測。
        const uint32_t flags = convo::consumeAtomic(rebuildReasonFlags_, std::memory_order_acquire);
        return (flags & rebuildReasonMask(reason)) != 0u;
    }

    bool setRebuildReason(RebuildReason reason) noexcept
    {
        // acq_rel: 取得側 acquire で直前の clearRebuildReason を観測し、
        //          解放側 release で新 flag を hasRebuildReason の acquire に公開。
        const uint32_t oldFlags = convo::fetchOrAtomic(rebuildReasonFlags_, rebuildReasonMask(reason), std::memory_order_acq_rel);
        return (oldFlags & rebuildReasonMask(reason)) == 0u;
    }

    bool clearRebuildReason(RebuildReason reason) noexcept
    {
        // acq_rel: 取得側 acquire で直前の setRebuildReason を観測し、
        //          解放側 release でクリア後の状態を hasRebuildReason acquire に公開。
        const uint32_t oldFlags = convo::fetchAndAtomic(rebuildReasonFlags_, static_cast<uint32_t>(~rebuildReasonMask(reason)), std::memory_order_acq_rel);
        return (oldFlags & rebuildReasonMask(reason)) != 0u;
    }



    void debugAssertNotAudioThread() const;

    inline convo::EngineRuntime makeEngineRuntimeState(DSPCore* current,
                                                        DSPCore* next,
                                                        convo::TransitionPolicy policy,
                                                        double fadeTimeSec,
                                                        bool active) noexcept
    {
        // IR-7 (Chapter 8): State transition precondition validation.
        jassert(current != nullptr); // Current DSP must always be valid at publish time.
        jassert(fadeTimeSec >= 0.0);

        refreshCrossfadePreparedSnapshotFromAtomics();
        const auto prepared = consumeCrossfadePreparedSnapshot();

        // IR-7: Crossfade parameter sanity checks (invariant enforcement).
        jassert(prepared.fadeTimeSec >= 0.0);
        jassert(prepared.latencyDelayOld >= 0 && prepared.latencyDelayNew >= 0);
        jassert(prepared.startDelayBlocks >= 0);
        jassert(prepared.dryHoldSamples >= 0);

        convo::EngineRuntime runtime {};
        const auto getRuntimeUuid = [](DSPCore* dsp) noexcept -> std::uint64_t
        {
            return dsp != nullptr ? dsp->runtimeUuid : 0;
        };

        const auto runtimePublishView = getRuntimePublishView();
        DSPCore* fading = resolveFadingDSPFromRuntimeWorldOnly(runtimePublishView.graph);

        runtime.current = current;
        runtime.currentRuntimeUuid = getRuntimeUuid(current);
        runtime.transition.current = current;
        runtime.transition.next = next;
        runtime.transitionCurrentRuntimeUuid = getRuntimeUuid(current);
        runtime.transitionNextRuntimeUuid = getRuntimeUuid(next);
        runtime.latencyDelayOld = prepared.latencyDelayOld;
        runtime.latencyDelayNew = prepared.latencyDelayNew;
        runtime.transition.policy = policy;
        runtime.transition.fadeTimeSec = fadeTimeSec;
        runtime.transition.latencyDeltaSamples = runtime.latencyDelayOld - runtime.latencyDelayNew;
        runtime.transition.active = active;
        runtime.fading = fading;
        runtime.fadingRuntimeUuid = getRuntimeUuid(fading);
        runtime.latencyResetPending = prepared.latencyResetPending;
        runtime.dspCrossfadePending = prepared.pending;
        runtime.dspCrossfadeUseDryAsOld = prepared.useDryAsOld;
        runtime.firstIrDryCrossfadePending = prepared.firstIrDryCrossfadePending;
        runtime.queuedFadeTimeSec = prepared.fadeTimeSec;
        runtime.dspCrossfadeStartDelayBlocks = prepared.startDelayBlocks;
        runtime.dspCrossfadeDryHoldSamples = prepared.dryHoldSamples;
        runtime.dryScaleTarget = prepared.dryScaleTarget;
        return runtime;
    }

    inline CrossfadePreparedSnapshot consumeCrossfadePreparedSnapshot() const noexcept
    {
        // acquire: publishCrossfadePreparedSnapshot の release と HB し、
        //          最新のクロスフェードスナップショットスロットを取得。
        const int slot = static_cast<int>(convo::consumeAtomic(crossfadePreparedSnapshotIndex_, std::memory_order_acquire) & 1u);
        return crossfadePreparedSnapshots_[slot];
    }

    inline void publishCrossfadePreparedSnapshot(const CrossfadePreparedSnapshot& snapshot) noexcept
    {
        // acquire: 現スロットの読み出しは直前の公開を観測。
        const int currentSlot = convo::consumeAtomic(crossfadePreparedSnapshotIndex_, std::memory_order_acquire) & 1;
        const int nextSlot = currentSlot ^ 1;
        crossfadePreparedSnapshots_[nextSlot] = snapshot;
        // release: 新スロット書き込み完了を consumeCrossfadePreparedSnapshot の acquire に公開。
        convo::publishAtomic(crossfadePreparedSnapshotIndex_, nextSlot, std::memory_order_release);
    }

    inline void refreshCrossfadePreparedSnapshotFromAtomics() noexcept
    {
        const CrossfadePreparedSnapshot snapshot {
            .pending = consumeAtomic(dspCrossfadePending),
            .useDryAsOld = consumeAtomic(dspCrossfadeUseDryAsOld),
            .firstIrDryCrossfadePending = consumeAtomic(firstIrDryCrossfadePending),
            .fadeTimeSec = consumeAtomic(queuedFadeTimeSec),
            .latencyDelayOld = consumeAtomic(latencyDelayOld),
            .latencyDelayNew = consumeAtomic(latencyDelayNew),
            .startDelayBlocks = consumeAtomic(dspCrossfadeStartDelayBlocks),
            .dryHoldSamples = consumeAtomic(dspCrossfadeDryHoldSamples),
            .latencyResetPending = consumeAtomic(latencyResetPending),
            .dryScaleTarget = consumeAtomic(dspCrossfadeDryScaleTarget)
        };

        publishCrossfadePreparedSnapshot(snapshot);
    }

    inline RuntimePublishView getRuntimePublishView() const noexcept
    {
        const auto* world = runtimeStore.observe();
        return RuntimePublishView { world != nullptr ? &world->graph : nullptr };
    }

    inline convo::RuntimeGraph makeRuntimeGraphState(const convo::EngineRuntime& state) const noexcept
    {
        convo::RuntimeGraph graph {};
        graph.activeNode = state.current;
        graph.fadingNode = state.fading;

        auto* current = static_cast<DSPCore*>(state.current);
        if (current != nullptr)
        {
            graph.runtimeUuid = current->runtimeUuid;
            graph.sampleRate = current->sampleRate;
            graph.ditherBitDepth = current->ditherBitDepth;
            graph.noiseShaperType = static_cast<int>(current->noiseShaperType);
            graph.oversamplingFactor = static_cast<int>(current->oversamplingFactor);
            auto& eq = current->eqRt();
            graph.eqAgcAttackCoeffTable = eq.getAgcAttackCoeffTable();
            graph.eqAgcReleaseCoeffTable = eq.getAgcReleaseCoeffTable();
            graph.eqAgcSmoothCoeffTable = eq.getAgcSmoothCoeffTable();
            graph.eqAgcCoeffTableCapacity = eq.getAgcCoeffTableCapacity();
        }

        auto* fading = static_cast<DSPCore*>(state.fading);
        if (fading != nullptr)
            graph.fadingRuntimeUuid = fading->runtimeUuid;

        graph.transitionCurrentRuntimeUuid = state.transitionCurrentRuntimeUuid;
        graph.transitionNextRuntimeUuid = state.transitionNextRuntimeUuid;
        graph.latencyDelayOld = state.latencyDelayOld;
        graph.latencyDelayNew = state.latencyDelayNew;
        graph.latencyResetPending = state.latencyResetPending;

        const auto observedSnapshot = m_coordinator.observeCurrentRuntime(kControlEpochReaderIndex);
        const auto* snapshot = observedSnapshot.get();
        if (snapshot != nullptr)
        {
            graph.eqBypassed = snapshot->eqBypass;
            graph.convBypassed = snapshot->convBypass;
            graph.softClipEnabled = snapshot->softClipEnabled;
            graph.saturationAmount = static_cast<double>(snapshot->saturationAmount);
            graph.inputHeadroomGain = snapshot->inputHeadroomGain;
            graph.outputMakeupGain = snapshot->outputMakeupGain;
            graph.convolverInputTrimGain = snapshot->convInputTrimGain;
            graph.oversamplingFactor = snapshot->oversamplingFactor;
            graph.ditherBitDepth = snapshot->ditherBitDepth;
            graph.noiseShaperType = static_cast<int>(snapshot->noiseShaperType);
            graph.sampleRate = snapshot->sampleRate;
        }

        graph.dspCrossfadePending = state.dspCrossfadePending;
        graph.dspCrossfadeUseDryAsOld = state.dspCrossfadeUseDryAsOld;
        graph.firstIrDryCrossfadePending = state.firstIrDryCrossfadePending;
        graph.queuedFadeTimeSec = state.queuedFadeTimeSec;
        graph.dspCrossfadeStartDelayBlocks = state.dspCrossfadeStartDelayBlocks;
        graph.dspCrossfadeDryHoldSamples = state.dspCrossfadeDryHoldSamples;
        graph.dryScaleTarget = state.dryScaleTarget;

        return graph;
    }

    inline bool runtimeCrossfadePendingWorldOnly(const convo::RuntimeGraph* runtimeGraph) const noexcept
    {
        return runtimeGraph != nullptr
            ? runtimeGraph->dspCrossfadePending
            : false;
    }
    inline bool runtimeCrossfadeUseDryAsOldWorldOnly(const convo::RuntimeGraph* runtimeGraph) const noexcept
    {
        return runtimeGraph != nullptr
            ? runtimeGraph->dspCrossfadeUseDryAsOld
            : false;
    }
    inline DSPCore* runtimePublishedCurrentDSP(const convo::RuntimeGraph* runtimeGraph = nullptr) const noexcept
    {
        if (runtimeGraph != nullptr)
            return static_cast<DSPCore*>(runtimeGraph->activeNode);
        return nullptr;
    }
    inline std::uint64_t runtimeRevision(const convo::RuntimeGraph* runtimeGraph = nullptr) const noexcept
    {
        if (runtimeGraph != nullptr)
            return runtimeGraph->generation;
        return 0;
    }
    inline std::uint64_t runtimeCurrentUuid(const convo::RuntimeGraph* runtimeGraph = nullptr) const noexcept
    {
        if (runtimeGraph != nullptr)
            return runtimeGraph->runtimeUuid;
        return 0;
    }
    inline std::uint64_t runtimeFadingUuid(const convo::RuntimeGraph* runtimeGraph = nullptr) const noexcept
    {
        if (runtimeGraph != nullptr)
            return runtimeGraph->fadingRuntimeUuid;
        return 0;
    }
    inline std::uint64_t runtimeTransitionCurrentUuid(const convo::RuntimeGraph* runtimeGraph = nullptr) const noexcept
    {
        if (runtimeGraph != nullptr)
            return runtimeGraph->transitionCurrentRuntimeUuid;
        return 0;
    }
    inline std::uint64_t runtimeTransitionNextUuid(const convo::RuntimeGraph* runtimeGraph = nullptr) const noexcept
    {
        if (runtimeGraph != nullptr)
            return runtimeGraph->transitionNextRuntimeUuid;
        return 0;
    }
    inline double runtimeSampleRateWorldOnly(const convo::RuntimeGraph* runtimeGraph) const noexcept
    {
        return runtimeGraph != nullptr
            ? runtimeGraph->sampleRate
            : 0.0;
    }

    template <typename T>
    static inline void publishAtomicPtr(std::atomic<T*>& dst, T* value) noexcept
    {
        // release: pointer を consumeAtomicPtr acquire に公開。
        convo::publishAtomic(dst, value, std::memory_order_release);
    }

    template <typename T>
    static inline T* consumeAtomicPtr(const std::atomic<T*>& src) noexcept
    {
        // acquire: publishAtomicPtr release と HB し、最新の pointer を取得。
        return convo::consumeAtomic(src, std::memory_order_acquire);
    }

    template <typename T, typename U,
              typename = std::enable_if_t<std::is_convertible_v<U, T*>>>
    static inline T* exchangeAtomicPtr(std::atomic<T*>& dst, U&& value) noexcept
    {
        // acq_rel: 旧 pointer を acquire で取得し、新 pointer を release で公開。
        return convo::exchangeAtomic(dst, static_cast<T*>(std::forward<U>(value)), std::memory_order_acq_rel);
    }

    template <typename T>
    static inline void publishAtomic(std::atomic<T>& dst,
                                     T value,
                                     std::memory_order order = std::memory_order_release) noexcept
    {
        // order (release/relaxed): callerの consumeAtomic acquire と HB し、値を公開。
        convo::publishAtomic(dst, value, order);
    }

    template <typename T>
    static inline T consumeAtomic(const std::atomic<T>& src,
                                  std::memory_order order = std::memory_order_acquire) noexcept
    {
        // order (acquire/relaxed): callerの publishAtomic release と HB し、値を取得。
        return convo::consumeAtomic(src, order);
    }

    template <typename T>
    static inline T exchangeAtomic(std::atomic<T>& dst,
                                   T value,
                                   std::memory_order order = std::memory_order_acq_rel) noexcept
    {
        // order (acq_rel): 旧値を acquire で取得し、新値を release で公開。
        return convo::exchangeAtomic(dst, value, order);
    }

    template <typename T, typename U,
              typename = std::enable_if_t<std::is_integral_v<T> && std::is_convertible_v<U, T>>>
    static inline T fetchAddAtomic(std::atomic<T>& dst,
                                   U value,
                                   std::memory_order order = std::memory_order_acq_rel) noexcept
    {
        // order (acq_rel): 旧値を acquire で辻取り、新値を release で公開。世代カウンター advance、etc.に使用。
        return convo::fetchAddAtomic(dst, value, order);
    }

    // RT helper は publish world が無い場合に side-channel atomic へフォールバックしない。
    // RuntimeGraph が確立していない期間は無音フェイルセーフへ落とし、
    // publish world 非依存の DSP 参照は Audio Thread から段階的に除去する。
    inline DSPCore* resolveCurrentDSPFromRuntimeWorldOnly(const convo::RuntimeGraph* runtimeGraph) const noexcept
    {
        if (runtimeGraph == nullptr)
            return nullptr;

        auto* graphCurrent = static_cast<DSPCore*>(runtimeGraph->activeNode);
        const auto graphCurrentUuid = runtimeGraph->runtimeUuid;
        if (graphCurrent != nullptr
            && graphCurrentUuid != 0)
            return graphCurrent;

        return nullptr;
    }

    inline DSPCore* resolveFadingDSPFromRuntimeWorldOnly(const convo::RuntimeGraph* runtimeGraph) const noexcept
    {
        if (runtimeGraph == nullptr)
            return nullptr;

        auto* graphFading = static_cast<DSPCore*>(runtimeGraph->fadingNode);
        const auto graphFadingUuid = runtimeGraph->fadingRuntimeUuid;
        if (graphFading != nullptr
            && graphFadingUuid != 0)
            return graphFading;

        return nullptr;
    }

    // commit 系の寿命遷移を helper へ集約し、
    // current/fading の更新規約を 1 箇所で追えるようにする。
    inline void publishCurrentDSPAndTakeOwnership(DSPCore* newDSP) noexcept
    {
        publishCurrentDSP(newDSP);
        activeDSP = newDSP;
    }

    inline std::uint64_t reserveNextRuntimeGraphGeneration() noexcept
    {
        // acq_rel: fetch-add で generation counter を advance し、他スレッドへ新 generation を公開。
        //          acquire 側で旧 generation 参照により stale request 検出を回避。
        const auto nextGraphGeneration = convo::fetchAddAtomic(runtimeGraphRevision,
                                                               static_cast<std::uint64_t>(1),
                                                               std::memory_order_acq_rel) + 1;
        // acq_rel: publish count を increment し、runtime world 公開イベント数を追跡。
        convo::fetchAddAtomic(g_runtimePublishCount,
                              static_cast<std::uint64_t>(1),
                              std::memory_order_acq_rel);
        return nextGraphGeneration;
    }

    static inline std::uint64_t reserveNextRuntimeVersion() noexcept
    {
        static std::atomic<std::uint64_t> s_nextRuntimeVersion { 1 };
        // acq_rel: runtime version counter を increment し、複数 world lifecycle across に unique ID を割り当て。
        return convo::fetchAddAtomic(s_nextRuntimeVersion,
                                     static_cast<std::uint64_t>(1),
                                     std::memory_order_acq_rel);
    }

    //=== RuntimePublicationCoordinator NonRT helper API ===//
    // AudioEngine 内部の publish/retire helper（NonRT 専用）。

    inline void retireRuntimePublishWorldNonRt(RuntimePublishWorld* world) noexcept
    {
        if (world == nullptr)
            return;

        enqueueDeferredDeleteNonRt(world, [](void* p)
        {
            auto* ptr = static_cast<RuntimePublishWorld*>(p);
            ptr->~RuntimePublishWorld();
            convo::aligned_free(ptr);
        });
    }

    [[nodiscard]] convo::aligned_unique_ptr<RuntimePublishWorld>
    buildRuntimePublishWorld(DspCoreHandle current,
                             DspCoreHandle next,
                             convo::TransitionPolicy policy,
                             double fadeTimeSec,
                             bool active) noexcept
    {
        const auto nextGraphGeneration = reserveNextRuntimeGraphGeneration();

        auto engineState = makeEngineRuntimeState(current, next, policy, fadeTimeSec, active);
        engineState.revision = nextGraphGeneration;
        auto graphState = makeRuntimeGraphState(engineState);
        graphState.generation = nextGraphGeneration;

        auto worldOwner = convo::aligned_make_unique<RuntimePublishWorld>();
        worldOwner->generation = nextGraphGeneration;
        worldOwner->engine = engineState;
        worldOwner->graph = graphState;
        // runtimeVersion: monotonically increasing version number (magna_carta.md Section 2)
        worldOwner->runtimeVersion = reserveNextRuntimeVersion();
        // transitionId: unique per crossfade event
        worldOwner->transitionId = nextGraphGeneration + (active ? 0x1000000000000000ULL : 0);

        return worldOwner;
    }

    // publishState 用 helper: world を構築して返す。
    [[nodiscard]] convo::aligned_unique_ptr<RuntimePublishWorld>
    buildWorldAndPublishTransition(DspCoreHandle current,
                                   DspCoreHandle next,
                                   convo::TransitionPolicy policy,
                                   double fadeTimeSec,
                                   bool active) noexcept
    {
        return buildRuntimePublishWorld(current, next, policy, fadeTimeSec, active);
    }

    // adoptAndPublish 用 helper: publishCurrentDSP（DSP 採用）+
    // buildRuntimePublishWorld を 1 本に集約して world を構築して返す。
    [[nodiscard]] convo::aligned_unique_ptr<RuntimePublishWorld>
    adoptAndBuildPublishWorld(DspCoreHandle newCurrent,
                              DspCoreHandle transitionNext,
                              convo::TransitionPolicy policy,
                              double fadeTimeSec,
                              bool active) noexcept
    {
        publishCurrentDSPAndTakeOwnership(newCurrent);
        return buildRuntimePublishWorld(newCurrent, transitionNext, policy, fadeTimeSec, active);
    }

    //=== End RuntimePublicationCoordinator NonRT helper API ===//

    class RuntimePublicationBridge final
    {
    public:
        explicit RuntimePublicationBridge(AudioEngine& engine) noexcept
            : engine_(&engine)
        {
        }

        [[nodiscard]] convo::aligned_unique_ptr<RuntimePublishWorld>
        buildWorldAndPublishTransition(DspCoreHandle current,
                                       DspCoreHandle next,
                                       convo::TransitionPolicy policy,
                                       double fadeTimeSec,
                                       bool active) noexcept
        {
            return engine_->buildWorldAndPublishTransition(current, next, policy, fadeTimeSec, active);
        }

        [[nodiscard]] convo::aligned_unique_ptr<RuntimePublishWorld>
        adoptAndBuildPublishWorld(DspCoreHandle newCurrent,
                                  DspCoreHandle transitionNext,
                                  convo::TransitionPolicy policy,
                                  double fadeTimeSec,
                                  bool active) noexcept
        {
            return engine_->adoptAndBuildPublishWorld(newCurrent, transitionNext, policy, fadeTimeSec, active);
        }

        void retireRuntimePublishWorldNonRt(RuntimePublishWorld* world) noexcept
        {
            engine_->retireRuntimePublishWorldNonRt(world);
        }

        void resetRuntimeGraphRevisionNonRt() noexcept
        {
            convo::publishAtomic(engine_->runtimeGraphRevision,
                                 static_cast<std::uint64_t>(0),
                                 std::memory_order_release);
        }

    private:
        AudioEngine* engine_ = nullptr;
    };

    using RuntimePublicationCoordinator = convo::RuntimePublicationCoordinator<RuntimePublishWorld,
                                                                               DspCoreHandle,
                                                                               RuntimePublicationBridge>;

    static_assert(!std::is_copy_constructible_v<RuntimePublicationCoordinator>,
                  "RuntimePublicationCoordinator must remain move-only");
    static_assert(!std::is_copy_assignable_v<RuntimePublicationCoordinator>,
                  "RuntimePublicationCoordinator must remain move-only");
    static_assert(std::is_move_constructible_v<RuntimePublicationCoordinator>,
                  "RuntimePublicationCoordinator must remain move-constructible");

    using RuntimePublishStore = RuntimePublicationCoordinator::Store;

    RuntimePublishStore runtimeStore;

    [[nodiscard]] inline RuntimePublicationCoordinator publicationCoordinator() noexcept
    {
        return RuntimePublicationCoordinator::create(RuntimePublicationBridge { *this }, runtimeStore);
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

        DSPCore* current = activeDSP.get();
        const auto runtimePublishView = getRuntimePublishView();
        const auto* runtimeGraph = runtimePublishView.graph;
        auto* fading = resolveFadingDSPFromRuntimeWorldOnly(runtimeGraph);
        const auto revision = runtimeRevision(runtimeGraph);
        const auto publishedCurrentUuid = runtimeCurrentUuid(runtimeGraph);
        const auto publishedFadingUuid = runtimeFadingUuid(runtimeGraph);

        const juce::String message = "[DIAG] runtime transition event origin="
            + juce::String(origin != nullptr ? origin : "unknown")
            + " primaryUuid=" + juce::String(static_cast<juce::int64>(getUuid(primary)))
            + " secondaryUuid=" + juce::String(static_cast<juce::int64>(getUuid(secondary)))
            + " currentUuid=" + juce::String(static_cast<juce::int64>(getUuid(current)))
            + " fadingUuid=" + juce::String(static_cast<juce::int64>(getUuid(fading)))
            + " publishRev=" + juce::String(static_cast<juce::int64>(revision))
            + " publishCurrentUuid=" + juce::String(static_cast<juce::int64>(publishedCurrentUuid))
            + " publishFadingUuid=" + juce::String(static_cast<juce::int64>(publishedFadingUuid));
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

    inline void replaceFadingOutDSPAndRetirePrevious(DSPCore* dsp) noexcept
    {
        DSPCore* atomicCurrent = activeDSP.get();
        const auto runtimePublishView = getRuntimePublishView();
        const auto* runtimeGraph = runtimePublishView.graph;
        validateDistinctRuntimeSlots("replaceFadingOutDSPAndRetirePrevious.before",
                                     atomicCurrent,
                                     resolveFadingDSPFromRuntimeWorldOnly(runtimeGraph),
                                     nullptr);

        if (auto* prev = sanitizeRawPtr(exchangeFadingOutDSP(dsp)))
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
                                     atomicCurrent,
                                     resolveFadingDSPFromRuntimeWorldOnly(runtimeGraph),
                                     nullptr);
        logRuntimeTransitionEvent("replaceFadingOutDSPAndRetirePrevious", dsp);
    }

    inline void retireRuntimeImmediately(DSPCore* dsp) noexcept
    {
        if (dsp == nullptr)
            return;

        const auto runtimePublishView = getRuntimePublishView();
        const auto* runtimeGraph = runtimePublishView.graph;
        auto* publishedCurrent = runtimePublishedCurrentDSP(runtimeGraph);
        DSPCore* atomicCurrent = activeDSP.get();
        if (dsp == atomicCurrent || dsp == publishedCurrent)
        {
            logUnexpectedRuntimeTransition("retireRuntimeImmediately", atomicCurrent, dsp);
            jassert(dsp != atomicCurrent && dsp != publishedCurrent);
            return;
        }

        logRuntimeTransitionEvent("retireRuntimeImmediately", dsp);
        retireDSP(dsp);
    }

    inline void publishSmoothTransitionState(DSPCore* nextDSP,
                                             DSPCore* previousDSP,
                                             double fadeTimeSec) noexcept
    {
        if (nextDSP == nullptr || nextDSP == previousDSP)
        {
            logUnexpectedRuntimeTransition("publishSmoothTransitionState", nextDSP, previousDSP);
            jassert(nextDSP != nullptr && nextDSP != previousDSP);
        }

        publicationCoordinator().publishState(nextDSP,
                              previousDSP,
                              convo::TransitionPolicy::SmoothOnly,
                              fadeTimeSec,
                              true);
        logRuntimeTransitionEvent("publishSmoothTransitionState", nextDSP, previousDSP);
    }

    inline void startImmediateSmoothTransition(DSPCore* previousDSP,
                                               double fadeTimeSec) noexcept
    {
        DSPCore* atomicCurrent = activeDSP.get();
        const auto runtimePublishView = getRuntimePublishView();
        const auto* runtimeGraph = runtimePublishView.graph;
        if (previousDSP == nullptr || previousDSP == atomicCurrent)
        {
            logUnexpectedRuntimeTransition("startImmediateSmoothTransition", atomicCurrent, previousDSP);
            jassert(previousDSP != nullptr && previousDSP != atomicCurrent);
        }

        const double rampSampleRate = std::max(1.0,
            (atomicCurrent != nullptr) ? atomicCurrent->sampleRate : consumeAtomic(currentSampleRate, std::memory_order_acquire));
        dspCrossfadeGain.reset(rampSampleRate, std::max(0.001, fadeTimeSec));
        dspCrossfadeGain.setCurrentAndTargetValue(0.0);
        // setTargetValue(1.0) は Audio Thread の armCrossfadeIfPending で呼ぶ (C1-2)

        replaceFadingOutDSPAndRetirePrevious(previousDSP);
        // release: dspCrossfade フラグ群を Audio Thread acquire で観測できるよう公開。
        publishAtomic(dspCrossfadeUseDryAsOld, false, std::memory_order_release);
        publishAtomic(firstIrDryCrossfadePending, false, std::memory_order_release);
        publishAtomic(queuedFadeTimeSec, fadeTimeSec, std::memory_order_release);
        publishAtomic(dspCrossfadePending, true, std::memory_order_release);
        setIRChangeFlag();
        publicationCoordinator().publishState(atomicCurrent,
                              previousDSP,
                              convo::TransitionPolicy::SmoothOnly,
                              fadeTimeSec,
                              true);
        validateDistinctRuntimeSlots("startImmediateSmoothTransition",
                                     atomicCurrent,
                                     resolveFadingDSPFromRuntimeWorldOnly(runtimeGraph),
                                     nullptr);
        logRuntimeTransitionEvent("startImmediateSmoothTransition", atomicCurrent, previousDSP);
    }

    inline void publishHardResetForCurrentDSP() noexcept
    {
        DSPCore* atomicCurrent = activeDSP.get();
        const auto runtimePublishView = getRuntimePublishView();
        const auto* runtimeGraph = runtimePublishView.graph;
        if (atomicCurrent == nullptr)
        {
            logUnexpectedRuntimeTransition("publishHardResetForCurrentDSP", nullptr, nullptr);
            jassert(atomicCurrent != nullptr);
        }

        publishAtomic(dspCrossfadePending, false, std::memory_order_release);
        publishAtomic(dspCrossfadeUseDryAsOld, false, std::memory_order_release);
        publishAtomic(firstIrDryCrossfadePending, false, std::memory_order_release);
        // release: dspCrossfade delay/sample パラメータをクリアし、Audio Thread acquire で取得。
        publishAtomic(dspCrossfadeStartDelayBlocks, 0, std::memory_order_release);
        publishAtomic(dspCrossfadeDryHoldSamples, 0, std::memory_order_release);
        publicationCoordinator().publishState(atomicCurrent,
                              nullptr,
                              convo::TransitionPolicy::HardReset,
                              0.0,
                              false);
        validateDistinctRuntimeSlots("publishHardResetForCurrentDSP",
                                     atomicCurrent,
                                     resolveFadingDSPFromRuntimeWorldOnly(runtimeGraph),
                                     nullptr);
        logRuntimeTransitionEvent("publishHardResetForCurrentDSP", atomicCurrent);
    }

    inline void armDryAsOldCrossfadeForCurrentDSP(double fadeTimeSec,
                                                  double targetIrScale) noexcept
    {
        DSPCore* atomicCurrent = activeDSP.get();
        const auto runtimePublishView = getRuntimePublishView();
        const auto* runtimeGraph = runtimePublishView.graph;
        if (atomicCurrent == nullptr)
        {
            logUnexpectedRuntimeTransition("armDryAsOldCrossfadeForCurrentDSP", nullptr, nullptr);
            jassert(atomicCurrent != nullptr);
        }

        const double rampSampleRate = std::max(1.0,
            // acquire: prepareToPlay/setSampleRate release と HB し、有効な currentSampleRate を取得。
            (atomicCurrent != nullptr) ? atomicCurrent->sampleRate : consumeAtomic(currentSampleRate, std::memory_order_acquire));
        dspCrossfadeGain.reset(rampSampleRate, std::max(0.001, fadeTimeSec));
        dspCrossfadeGain.setCurrentAndTargetValue(0.0);
        // setTargetValue(1.0) は Audio Thread の armCrossfadeIfPending で呼ぶ (C1-2)

        // release: queuedFadeTimeSec を Audio Thread acquire で観測できるよう公開。
        publishAtomic(queuedFadeTimeSec, fadeTimeSec, std::memory_order_release);
        // release: dryHoldSamples を設定。maxSamplesPerBlock を acquire で取得し有効性確保。
        publishAtomic(dspCrossfadeDryHoldSamples,
                  std::max(1, consumeAtomic(maxSamplesPerBlock, std::memory_order_acquire)));
        // acquire: currentSampleRate を取得し機率設定の確定幾体計置を後繿ゲイン設定削減価を確実。
        dspCrossfadeDryScaleGain.reset(std::max(1.0, consumeAtomic(currentSampleRate, std::memory_order_acquire)), 0.060);
        dspCrossfadeDryScaleGain.setCurrentAndTargetValue(1.0);
        // setTargetValue(targetIrScale) は Audio Thread の armCrossfadeIfPending で呼ぶ (C1-2)
        // release: dry scale target を設定。
        publishAtomic(dspCrossfadeDryScaleTarget, targetIrScale, std::memory_order_release);
        // release: dry-as-old crossfade フラグ群を Audio Thread acquire で観測。
        publishAtomic(dspCrossfadeUseDryAsOld, true, std::memory_order_release);
        publishAtomic(firstIrDryCrossfadePending, true, std::memory_order_release);
        publishAtomic(dspCrossfadePending, true, std::memory_order_release);
        publishAtomic(firstIrDryCrossfadeDone, true, std::memory_order_release);
        setIRChangeFlag();
        publicationCoordinator().publishState(atomicCurrent,
                              nullptr,
                              convo::TransitionPolicy::DryAsOld,
                              fadeTimeSec,
                              true);
        validateDistinctRuntimeSlots("armDryAsOldCrossfadeForCurrentDSP",
                                     atomicCurrent,
                                     resolveFadingDSPFromRuntimeWorldOnly(runtimeGraph),
                                     nullptr);
        logRuntimeTransitionEvent("armDryAsOldCrossfadeForCurrentDSP", atomicCurrent);
    }

    inline void queueCoalescedChangeNotification() noexcept
    {
        // acq_rel: pendingChangeNotification を交換し、true にならばasync update をトリガー。
        if (!exchangeAtomic(pendingChangeNotification, true))
            triggerAsyncUpdate();
    }

    inline EngineParameterSnapshot captureAudioThreadParameterSnapshot(const convo::GlobalSnapshot* snap,
                                                                       bool isFadingTarget = false) const noexcept
    {
        EngineParameterSnapshot snapshot {};
        // acquire: setXxx の release を観測しパラメータ最新値を取得。
        snapshot.eqBypassed = (snap != nullptr) ? snap->eqBypass : consumeAtomic(eqBypassRequested, std::memory_order_acquire);
        snapshot.convBypassed = (snap != nullptr) ? snap->convBypass : consumeAtomic(convBypassRequested, std::memory_order_acquire);
        snapshot.order = (snap != nullptr) ? snap->processingOrder : consumeAtomic(currentProcessingOrder, std::memory_order_acquire);
        snapshot.softClipEnabled = (snap != nullptr) ? snap->softClipEnabled : consumeAtomic(softClipEnabled, std::memory_order_acquire);
        snapshot.saturationAmount = (snap != nullptr) ? snap->saturationAmount : consumeAtomic(saturationAmount, std::memory_order_acquire);
        snapshot.inputHeadroomGain = (snap != nullptr) ? snap->inputHeadroomGain : consumeAtomic(inputHeadroomGain, std::memory_order_acquire);
        snapshot.outputMakeupGain = (snap != nullptr) ? snap->outputMakeupGain : consumeAtomic(outputMakeupGain, std::memory_order_acquire);
        snapshot.convolverInputTrimGain = (snap != nullptr) ? snap->convInputTrimGain : consumeAtomic(convolverInputTrimGain, std::memory_order_acquire);
        snapshot.analyzerSource = consumeAtomic(currentAnalyzerSource, std::memory_order_acquire);
        snapshot.analyzerEnabled = isFadingTarget ? false : consumeAtomic(analyzerEnabled, std::memory_order_acquire);
        snapshot.convHCMode = consumeAtomic(convHCFilterMode, std::memory_order_acquire);
        snapshot.convLCMode = consumeAtomic(convLCFilterMode, std::memory_order_acquire);
        snapshot.eqLPFMode = consumeAtomic(eqLPFFilterMode, std::memory_order_acquire);
        snapshot.adaptiveCoeffBankIndex = consumeAtomic(currentAdaptiveCoeffBankIndex, std::memory_order_acquire);
        const auto& adaptiveCoeffBank = getAdaptiveCoeffBankForIndex(snapshot.adaptiveCoeffBankIndex);
        snapshot.adaptiveCoeffGeneration = consumeAtomic(adaptiveCoeffBank.generation, std::memory_order_acquire);
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
        bufferToFill.clearActiveBufferRegion();
    }

    inline void applySafeSilentFallback(juce::AudioBuffer<double>& buffer) noexcept
    {
        buffer.clear();
    }

    inline void armCrossfadeIfPending(bool hasFading,
                                      bool& useDryAsOld,
                                      const convo::RuntimeGraph* runtimeGraph) noexcept
    {
        const bool hasPendingCrossfade = (runtimeGraph != nullptr && runtimeGraph->dspCrossfadePending);

        if (!hasPendingCrossfade)
        {
            dspCrossfadeArmed_RT = false;
            dspCrossfadeStartDelayBlocks_RT = 0;
            return;
        }

        const bool firstLoadDryPending = (runtimeGraph != nullptr && runtimeGraph->firstIrDryCrossfadePending);

        if ((hasFading || firstLoadDryPending)
            && !dspCrossfadeArmed_RT)
        {
            dspCrossfadeArmed_RT = true;
            latencyDelayOld_RT        = (runtimeGraph != nullptr) ? runtimeGraph->latencyDelayOld              : 0;
            latencyDelayNew_RT        = (runtimeGraph != nullptr) ? runtimeGraph->latencyDelayNew              : 0;
            dspCrossfadeStartDelayBlocks_RT = (runtimeGraph != nullptr) ? runtimeGraph->dspCrossfadeStartDelayBlocks : 0;

            // C1-2: activate のみ Audio Thread で実行 (reset/setCurrentAndTargetValue は Message Thread 側)
            dspCrossfadeGain.setTargetValue(1.0);

            if (firstLoadDryPending)
            {
                useDryAsOld = true;
                if (runtimeGraph != nullptr)
                    dspCrossfadeDryScaleGain.setTargetValue(runtimeGraph->dryScaleTarget);
            }
        }
    }

    template <typename ProcessFn>
    inline bool processCrossfadeDelayGateIfPending(DSPCore* fading,
                                                   bool useDryAsOld,
                                                   bool hasPendingCrossfade,
                                                   ProcessFn processFn) noexcept
    {
        if (fading != nullptr
            && !useDryAsOld
            && hasPendingCrossfade
            && dspCrossfadeStartDelayBlocks_RT > 0)
        {
            --dspCrossfadeStartDelayBlocks_RT;
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

    inline void finalizeCrossfadeMixPath(DSPCore* current,
                                         DSPCore* fading,
                                         bool resetDryScaleGain) noexcept
    {
        if (!dspCrossfadeGain.isSmoothing())
        {
            validateDistinctRuntimeSlotsRT(current, fading, nullptr);

            if (resetDryScaleGain)
            {
                dspCrossfadeDryScaleGain.current = 1.0;
                dspCrossfadeDryScaleGain.target = 1.0;
                dspCrossfadeDryScaleGain.step = 0.0;
                dspCrossfadeDryScaleGain.remaining = 0;
            }
        }
    }

    inline void cleanupCrossfadeDirectPath(DSPCore* current,
                                           DSPCore* fading) noexcept
    {
        if (fading != nullptr && !dspCrossfadeGain.isSmoothing())
        {
            validateDistinctRuntimeSlotsRT(current, fading, nullptr);
        }
    }

    inline void resetLatencyBuffersIfPending(int bufferSize,
                                             int& writePos,
                                             const convo::RuntimeGraph* runtimeGraph) noexcept
    {
        const bool runtimeResetPending = (runtimeGraph != nullptr && runtimeGraph->latencyResetPending);
        if (runtimeResetPending)
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

        const double cachedSampleRate = consumeAtomic(currentSampleRate, std::memory_order_acquire);
        const double baseSampleRate = cachedSampleRate > 0.0
            ? cachedSampleRate
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
                                                  const convo::RuntimeGraph* runtimeGraph,
                                                  MixFn mixFn) noexcept
    {
        const int bufferSize = latencyBufSize;
        int writePos = latencyWritePos;
        const int delayOld = latencyDelayOld_RT;
        const int delayNew = latencyDelayNew_RT;

        resetLatencyBuffersIfPending(bufferSize, writePos, runtimeGraph);

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
    // RuntimePublicationCoordinator は AudioEngine のネストクラスであるため
    // C++11 以降は自動的にプライベートメンバへアクセス可能。friend 宣言は不要。

//==============================================================================
// インラインヘルパー関数（Adaptive 係数アクセス）
//==============================================================================

// Audio Thread 用：現在アクティブな係数セットを取得（ロックフリー）
static inline const CoeffSet* getActiveCoeffSet(const AdaptiveCoeffBankSlot& slot) noexcept
{
    // acquire: CoeffSetWriteLockGuard::commit の publishAtomic release と HB し、アクティブインデックスを確定取得。
    const int activeIdx = consumeAtomic(slot.activeIndex, std::memory_order_acquire);
    // Memory barrier ensures coeffSetA/coeffSetB contents are fully visible
    std::atomic_thread_fence(std::memory_order_acquire);
    return (activeIdx == 0)
           ? &slot.coeffSetA
           : &slot.coeffSetB;
}

// 書き込み側用：非アクティブバッファの予約（CAS）
static inline bool reserveInactiveCoeffSet(AdaptiveCoeffBankSlot& slot) noexcept
{
    bool expected = false;
    // acquire: writeLock acquire CAS で直前のロック解放を観測し、CAS成功側で排他制御を確立。
    return convo::compareExchangeAtomic(slot.writeLock,
                                        expected,
                                        true,
                                        std::memory_order_acquire,
                                        std::memory_order_acquire);
}

// 書き込み側用：予約した非アクティブセットへのポインタ取得
static inline CoeffSet* getReservedInactiveCoeffSet(AdaptiveCoeffBankSlot& slot) noexcept
{
    // acquire: reserveInactiveCoeffSet CAS success の acquire と HB し、ロック下での確定インデックスを取得。
    int active = consumeAtomic(slot.activeIndex, std::memory_order_acquire);
    return (active == 0) ? &slot.coeffSetB : &slot.coeffSetA;
}

// DSP ポインタのセンチネル値（全ビット 1）をチェックし、無効値なら nullptr を返す
template <typename T>
static inline T* sanitizeRawPtr(T* ptr) noexcept
{
    constexpr uintptr_t kInvalidAllOnes = ~static_cast<uintptr_t>(0);
    return (reinterpret_cast<uintptr_t>(ptr) == kInvalidAllOnes) ? nullptr : ptr;
}

inline bool enqueueDeferredDeleteNonRt(void* ptr, void (*deleter)(void*)) noexcept
{
    if (ptr == nullptr || deleter == nullptr)
        return true;

    const uint64_t epoch = m_epochDomain.publish();
    if (m_epochDomain.enqueueRetire(ptr, deleter, epoch))
        return true;

    m_epochDomain.reclaimRetired();
    if (m_epochDomain.enqueueRetire(ptr, deleter, epoch))
        return true;

    std::lock_guard<std::mutex> lock(deferredDeleteFallbackMutex);
    deferredDeleteFallbackQueue.push_back(DeferredDeleteFallbackEntry{ ptr, deleter, epoch });
    return true;
}

inline convo::EpochDomain& epochDomain() noexcept
{
    return m_epochDomain;
}

inline void retireDSP(DSPCore* dsp) noexcept
{
    if (dsp == nullptr)
        return;

    convo::fetchAddAtomic(g_runtimeRetireCount,
                         static_cast<std::uint64_t>(1),
                         std::memory_order_acq_rel);
    if (enqueueDeferredDeleteNonRt(dsp, [](void* p)
    {
        auto* core = static_cast<DSPCore*>(p);
        core->~DSPCore();
        convo::aligned_free(core);
    }))
        return;
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
            publishAtomic(slot.writeLock, false, std::memory_order_release);
    }

    bool acquire() noexcept
    {
        bool expected = false;
        // acquire: writeLock acquire CAS で直前のロック解放を観測し、CAS成功側で排他制御を確立。
        acquired = convo::compareExchangeAtomic(slot.writeLock,
                            expected,
                            true,
                            std::memory_order_acquire,
                            std::memory_order_acquire);
        return acquired;
    }

    // commit() を呼ぶことで、デストラクタでのロック解放をスキップ
    void commit() noexcept
    {
        if (!acquired || committed)
            return;

        // acquire: acquire() の CAS success と HB し、ロック下での確定インデックスを取得。
        int oldActive = consumeAtomic(slot.activeIndex, std::memory_order_acquire);
        // release: 新インデックスをgetActiveCoeffSet acquire に公開し、バッファスイッチを原子化。
        publishAtomic(slot.activeIndex, 1 - oldActive, std::memory_order_release);
        // acq_rel: 世代を原子的に increment し、getActiveCoeffSet の generation 更新を release で公開。
        convo::fetchAddAtomic(slot.generation,
                             1u,
                             std::memory_order_acq_rel);
        // release: ロック解放を次の acquire() CAS failure に公開し、排他制御を終了。
        publishAtomic(slot.writeLock, false, std::memory_order_release);
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

    // スナップショット基盤（Phase 2）
    // ==================================================================
    convo::EpochDomain m_epochDomain;
    // DSP_THREAD_STATE: AudioEngine process系で使うaudio-thread専用RCU reader。
    convo::RCUReader audioThreadRcuReader { m_epochDomain };
    // ENGINE_CONTROL: Audio thread での deletion queue overflow 退避スロット。
    std::atomic<DSPCore*> audioThreadRetireOverflowPtr { nullptr };
    std::atomic<uint64_t> audioThreadRetireOverflowEpoch { 0 };
    std::atomic<uint64_t> audioThreadRetireEnqueueDropped { 0 };
    convo::SnapshotCoordinator m_coordinator;
    GenerationManager m_generationManager;

    // ==================================================================
    // Phase 3: コマンドバッファ + ワーカースレッド
    // ==================================================================
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
    std::array<convo::EQParameters, 2> latestEqParamsForFallback{};
    std::array<uint64_t, 2> latestEqHashForFallback{ 0, 0 };
    std::atomic<int> latestEqFallbackReadIndex{ 0 };

    // Audio Thread -> Message Thread 反映確認用 (RT安全: atomic store/fetch_add のみ)
    std::atomic<uint64_t> debugLastCreatedEqHash{ 0 };
    std::atomic<uint64_t> lastEnqueuedSnapshotDebounceKey_{ 0 };
    std::atomic<bool> hasLastEnqueuedSnapshotDebounceKey_{ false };
    uint64_t debugLastReportedCreatedEqHash{ std::numeric_limits<uint64_t>::max() }; // timerCallback 専用
    int debugLastReportedDspReady{ -1 }; // timerCallback 専用
    // Non-RT diagnostics: most recently rejected rebuild generation in commitNewDSP.
    std::atomic<uint64_t> lastRejectedGenerationNonRt{ 0 };
    int debugLastReportedTransitionActive{ -1 }; // timerCallback 専用
    int debugLastReportedTransitionPolicy{ -1 }; // timerCallback 専用
    uint64_t debugLastReportedTransitionCurrentPtr{ std::numeric_limits<uint64_t>::max() }; // timerCallback 専用
    uint64_t debugLastReportedTransitionNextPtr{ std::numeric_limits<uint64_t>::max() }; // timerCallback 専用
    double debugLastReportedTransitionFadeSec{ -1.0 }; // timerCallback 専用
    int debugLastReportedTransitionLatencyDeltaSamples{ std::numeric_limits<int>::max() }; // timerCallback 専用
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
    uint64_t debugLastReportedRuntimeSnapshotRevision{ std::numeric_limits<uint64_t>::max() }; // timerCallback 専用
    uint64_t debugLastReportedRuntimePublishCurrentUuid{ std::numeric_limits<uint64_t>::max() }; // timerCallback 専用
    uint64_t debugLastReportedRuntimePublishFadingUuid{ std::numeric_limits<uint64_t>::max() }; // timerCallback 専用
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
    // acquire: processLearningCommands publishAtomic release と HB し、learningCommandRead/Write 最新値を取得。
    const uint32_t currentWrite = consumeAtomic(learningCommandWrite, std::memory_order_acquire);
    const uint32_t currentRead = consumeAtomic(learningCommandRead, std::memory_order_acquire);
    const uint32_t next = (currentWrite + 1u) & learningCommandBufferMask;
    if (next == currentRead)
    {
        jassertfalse;
        return false;
    }

    learningCommandBuffer[currentWrite] = cmd;
    // release: バッファ重の書込みを dequeueLearningCommand acquire fence に公開。
    std::atomic_thread_fence(std::memory_order_release);
    // release: キュー埋まりを dequeueLearningCommand consumeAtomic acquire に公開。
    publishAtomic(learningCommandWrite, next, std::memory_order_release);
    return true;
}

inline bool AudioEngine::dequeueLearningCommand(LearningCommand& cmd) noexcept
{
    // acquire: enqueueLearningCommand publishAtomic release と HB し、learningCommandRead/Write 最新値を取得。
    const uint32_t currentRead = consumeAtomic(learningCommandRead, std::memory_order_acquire);
    const uint32_t currentWrite = consumeAtomic(learningCommandWrite, std::memory_order_acquire);
    if (currentRead == currentWrite)
        return false;

    std::atomic_thread_fence(std::memory_order_acquire);
    // acquire: fence でバッファ重の設定を後続のインストラクション下鉅に収所。
    cmd = learningCommandBuffer[currentRead];
    // release: キュー出しを enqueueLearningCommand consumeAtomic acquire に公開。
    publishAtomic(learningCommandRead, (currentRead + 1u) & learningCommandBufferMask, std::memory_order_release);
    return true;
}

inline bool AudioEngine::enqueueLearnerDispatch(const LearnerDispatchAction& action) noexcept
{
    // acquire: processDeferredLearningActions publishAtomic release と HB し、learnerDispatchRead/Write 最新値を取得。
    const uint32_t currentWrite = consumeAtomic(learnerDispatchWrite, std::memory_order_acquire);
    const uint32_t currentRead = consumeAtomic(learnerDispatchRead, std::memory_order_acquire);
    const uint32_t next = (currentWrite + 1u) & learnerDispatchBufferMask;
    if (next == currentRead)
    {
        // release: overflow フラグを processDeferredLearningActions consume acquire に公開。
        publishAtomic(lastFailedAction, action, std::memory_order_release);
        publishAtomic(learnerDispatchOverflow, true, std::memory_order_release);
        return false;
    }

    learnerDispatchBuffer[currentWrite] = action;
    // release: バッファ重の書込みを dequeueLearnerDispatch acquire fence に公開。
    std::atomic_thread_fence(std::memory_order_release);
    // release: キュー埋まりを dequeueLearnerDispatch consumeAtomic acquire に公開。
    publishAtomic(learnerDispatchWrite, next, std::memory_order_release);
    // release: overflow クリアを processDeferredLearningActions consume acquire に公開。
    publishAtomic(learnerDispatchOverflow, false, std::memory_order_release);
    return true;
}

inline bool AudioEngine::dequeueLearnerDispatch(LearnerDispatchAction& action) noexcept
{
    // acquire: enqueueLearnerDispatch publishAtomic release と HB し、learnerDispatchRead/Write 最新値を取得。
    const uint32_t currentRead = consumeAtomic(learnerDispatchRead, std::memory_order_acquire);
    const uint32_t currentWrite = consumeAtomic(learnerDispatchWrite, std::memory_order_acquire);
    if (currentRead == currentWrite)
        return false;

    std::atomic_thread_fence(std::memory_order_acquire);
    // acquire: fence でバッファ重の設定を後続のインストラクション下鉅に収所。
    action = learnerDispatchBuffer[currentRead];
    // release: キュー出しを enqueueLearnerDispatch consumeAtomic acquire に公開。
    publishAtomic(learnerDispatchRead, (currentRead + 1u) & learnerDispatchBufferMask, std::memory_order_release);
    return true;
}


