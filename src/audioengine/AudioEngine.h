
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
#include "ISRLifecycle.h"
#include "ISRRTExecution.h"
#include "ISRDSPHandle.h"
#include "ISRClosure.h"
#include "ISRPayloadTier.h"
#include "ISRHB.h"
#include "ISRRetire.h"
#include "ISRShutdown.h"
#include "ISRRuntimePublicationCoordinator.h"
#include "ISRDSPQuarantine.h"
#include "ISRClosureGraphWalker.h"
#include "ISRDebugRuntime.h"
#include "ISRRetireRuntimeEx.h"
#include "ISRBarrierOptimizer.h"
#include "ISREvidenceExporter.h"

class NoiseShaperLearner;
class AudioEngine;

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

struct RuntimeState : convo::isr::SealedObject<RuntimeState>
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
        const convo::EQParameters* snapshotEqParams = nullptr;
        uint64_t snapshotEqCoeffHash = 0;
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
            const convo::EQParameters* eqParams;
            const EQCoeffCache* eqCache;
            uint64_t eqCoeffHash;
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

        int maxSamplesPerBlock = 0;               // ホスト指定の入力ブロック上限（DSPCore::prepare() で設定）

        // ─────────────────────────────────────────────────────────────
        // 【Issue 3 修正】内部処理用最大バッファサイズ
        // 理由: Oversampling有効時（最大8x）、processSamplesUp() 後の
        //      ブロックサイズが SAFE_MAX_BLOCK_SIZE × 8 まで拡大しうるため。
        //      固定で ×8 確保することで RCU 再構築時の resize を回避する。
        // ─────────────────────────────────────────────────────────────
        int maxInternalBlockSize = 0;             // OS考慮後の最大サイズ（SAFE_MAX_BLOCK_SIZE × 8）
        static constexpr int FADE_IN_SAMPLES = 2048; // 42ms @ 48kHz
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
        static std::atomic<std::uint64_t> runtimeUuidCounterStorage_;
        static std::atomic<std::uint64_t>& runtimeUuidCounter() noexcept;
        [[nodiscard]] static std::uint64_t reserveNextRuntimeUuid() noexcept;
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
    [[nodiscard]] uint64_t publishRcuEpoch() noexcept;
    void enterRcuReader(int readerIndex) noexcept;
    void exitRcuReader(int readerIndex) noexcept;
    [[nodiscard]] uint64_t publishRetireEpoch() noexcept;
    [[nodiscard]] uint64_t currentRetireEpoch() const noexcept;
    uint64_t advanceRetireEpoch() noexcept;
    [[nodiscard]] bool enqueueRetireEpochBounded(void* ptr, void (*deleter)(void*), uint64_t epoch) noexcept;
    [[nodiscard]] uint32_t activeEpochObserverCount() const noexcept;

    void processBlockDouble (juce::AudioBuffer<double>& buffer);
    void changeListenerCallback(juce::ChangeBroadcaster* source) override;
    void convolverParamsChanged(ConvolverProcessor* processor) override;
    void timerCallback() override;

    //----------------------------------------------------------
    // 外部インターフェース (Message Thread)
    //----------------------------------------------------------
    ConvolverProcessor& getConvolverProcessor() { return uiConvolverProcessor; }
    [[nodiscard]] const ConvolverProcessor& getConvolverProcessor() const { return uiConvolverProcessor; }
    EQEditProcessor& getEQProcessor() { return uiEqEditor; }
    ThreadAffinityManager& getAffinityManager() noexcept { return affinityManager; }
    [[nodiscard]] const ThreadAffinityManager& getAffinityManager() const noexcept { return affinityManager; }

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

    [[nodiscard]] EQBandParams getEQBandParams(int band) const { return uiEqEditor.getBandParams(band); }
    [[nodiscard]] EQBandType getEQBandType(int band) const { return uiEqEditor.getBandType(band); }
    [[nodiscard]] EQChannelMode getEQBandChannelMode(int band) const { return uiEqEditor.getBandChannelMode(band); }
    [[nodiscard]] float getEQTotalGain() const { return uiEqEditor.getTotalGain(); }
    [[nodiscard]] bool isEQAGCEnabled() const { return uiEqEditor.getAGCEnabled(); }
    [[nodiscard]] float getEQNonlinearSaturation() const noexcept { return uiEqEditor.getNonlinearSaturation(); }
    [[nodiscard]] EQProcessor::FilterStructure getEQFilterStructure() const noexcept { return uiEqEditor.getFilterStructure(); }
    [[nodiscard]] const void* getEQStateSnapshot() const { return uiEqEditor.getEQStateSnapshot(); }
    void resetEQToDefaults() { uiEqEditor.reset(); }

    // acquire: prepareToPlay/releaseResources の release と HB し、有効なサンプルレートを取得。
    [[nodiscard]] double getSampleRate() const { return consumeAtomic(currentSampleRate, std::memory_order_acquire); }
    [[nodiscard]] double getProcessingSampleRate() const;
    struct LatencyBreakdown
    {
        int oversamplingLatencyBaseRateSamples = 0;
        int convolverAlgorithmLatencyBaseRateSamples = 0;
        int convolverIRPeakLatencyBaseRateSamples = 0;
        int convolverTotalLatencyBaseRateSamples = 0;
        int totalLatencyBaseRateSamples = 0;
    };

    [[nodiscard]] LatencyBreakdown getCurrentLatencyBreakdown() const;
    [[nodiscard]] int getCurrentLatencySamples() const;
    [[nodiscard]] int getTotalLatencySamples() const;  // PDC 用エイリアス (getCurrentLatencySamples と同値)

    // 【Fix Bug #8】gainToDecibels (std::log10 / libm) を Audio Thread から排除。
    // Audio Thread は linear gain を inputLevelLinear / outputLevelLinear に格納し、
    // getter (UI Thread) で dB 変換する。
    // acquire: Audio Thread の release publishAtomic (inputLevelLinear/outputLevelLinear) と HB
    //          し、最新のレベル値を UI Thread から安全に取得する。
    [[nodiscard]] float getInputLevel() const
    {
        const float linear = consumeAtomic(inputLevelLinear, std::memory_order_acquire);
        return (linear > LEVEL_METER_MIN_MAG)
               ? juce::Decibels::gainToDecibels(linear)
               : LEVEL_METER_MIN_DB;
    }
    [[nodiscard]] float getOutputLevel() const
    {
        const float linear = consumeAtomic(outputLevelLinear, std::memory_order_acquire);
        return (linear > LEVEL_METER_MIN_MAG)
               ? juce::Decibels::gainToDecibels(linear)
               : LEVEL_METER_MIN_DB;
    }




    [[nodiscard]] int getFifoNumReady() const { return analyzerFifo.getAvailableSamples(); }
    void readFromFifo(float* dest, int numSamples);
    void skipFifo(int numSamples);

    void calcEQResponseCurve(float* outMagnitudesL, float* outMagnitudesR, const std::complex<double>* zArray, int numPoints, double sampleRate);

    // パラメータ設定 (Thread-safe)
    void setEqBypassRequested (bool shouldBypass);
    void setConvolverBypassRequested (bool shouldBypass);
    // 以下 4 getter は acquire: 対応 setter の release publishAtomic/compareExchange と HB し、
    // Message Thread から bypass 状態の最新値を安全に観測する。
    [[nodiscard]] bool isEqBypassRequested() const noexcept { return consumeAtomic(eqBypassRequested, std::memory_order_acquire); }
    [[nodiscard]] bool isConvolverBypassRequested() const noexcept { return consumeAtomic(convBypassRequested, std::memory_order_acquire); }
    [[nodiscard]] bool isEQBypassed() const noexcept { return consumeAtomic(eqBypassActive, std::memory_order_acquire); }
    [[nodiscard]] bool isConvolverBypassed() const noexcept { return consumeAtomic(convBypassActive, std::memory_order_acquire); }

    void setConvolverPhaseMode(ConvolverProcessor::PhaseMode mode);
    [[nodiscard]] ConvolverProcessor::PhaseMode getConvolverPhaseMode() const;

    void requestEqPreset (int presetIndex);
    void requestEqPresetFromText(const juce::File& file);
    void requestConvolverPreset (const juce::File& irFile);

    void requestLoadState (const juce::ValueTree& state);
    [[nodiscard]] juce::ValueTree getCurrentState() const;
    void beginBulkParameterRestore() noexcept;
    void endBulkParameterRestore(bool requestRebuildNow = true) noexcept;

    void setProcessingOrder(ProcessingOrder order);
    // acquire: setProcessingOrder の release と HB し、最新の ProcessingOrder を観測。
    [[nodiscard]] ProcessingOrder getProcessingOrder() const { return consumeAtomic(currentProcessingOrder, std::memory_order_acquire); }

    // release: UI スレッドからの設定を Audio Thread が acquire で観測できるよう公開。
    // acquire: 対応 setter の release と HB し、最新値を取得。
    void setAnalyzerSource(AnalyzerSource source) { publishAtomic(currentAnalyzerSource, source, std::memory_order_release); }
    [[nodiscard]] AnalyzerSource getAnalyzerSource() const { return consumeAtomic(currentAnalyzerSource, std::memory_order_acquire); }
    void setAnalyzerEnabled(bool enabled) noexcept { publishAtomic(analyzerEnabled, enabled, std::memory_order_release); }
    [[nodiscard]] bool isAnalyzerEnabled() const noexcept { return consumeAtomic(analyzerEnabled, std::memory_order_acquire); }

    void setInputHeadroomDb(float db);
    [[nodiscard]] float getInputHeadroomDb() const;

    void setOutputMakeupDb(float db);
    [[nodiscard]] float getOutputMakeupDb() const;

    void setConvolverInputTrimDb(float db);
    [[nodiscard]] float getConvolverInputTrimDb() const;

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
    void setConvolverTailMode(ConvolverProcessor::TailMode mode) noexcept;
    void setConvolverTailStartSec(float sec) noexcept;
    void setConvolverTailStrength(float strength) noexcept;
    void setConvolverTailL1L2Multiplier(int multiplier) noexcept;

    // Convolver State Tree & Cache Settings
    [[nodiscard]] juce::ValueTree getConvolverStateTree() const;
    void setConvolverStateTree(const juce::ValueTree& state);
    [[nodiscard]] int getConvolverTargetUpgradeFFTSize() const;
    void setConvolverTargetUpgradeFFTSize(int fftSize);
    [[nodiscard]] bool isConvolverProgressiveUpgradeEnabled() const;
    void setConvolverEnableProgressiveUpgrade(bool enabled);
    [[nodiscard]] int getConvolverMaxCacheEntries() const;
    void setConvolverMaxCacheEntries(int maxEntries);
    void clearConvolverCache();

    void setDitherBitDepth(int bitDepth);
    [[nodiscard]] int getDitherBitDepth() const;

    void setNoiseShaperType(NoiseShaperType type);
    [[nodiscard]] NoiseShaperType getNoiseShaperType() const;
    void requestSnapshotForNoiseShaper();
    void requestRebuild(convo::RebuildKind kind) noexcept;
    void requestStructuredRebuildIntent(convo::RebuildKind kind) noexcept
    {
        submitRebuildIntent(kind,
                            RebuildTelemetryReason::RequestRebuildKindEntry,
                            RebuildTelemetryClass::Structural,
                            RebuildTelemetryPolicy::Replaceable);
    }
    void setFixedNoiseLogIntervalMs(int intervalMs) noexcept;
    [[nodiscard]] int getFixedNoiseLogIntervalMs() const noexcept;
    void setFixedNoiseWindowSamples(int windowSamples) noexcept;
    [[nodiscard]] int getFixedNoiseWindowSamples() const noexcept;

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
    [[nodiscard]] int getIRFadeSamples() const noexcept { return consumeAtomic(m_irFadeSamples, std::memory_order_acquire); }
    [[nodiscard]] int getEQFadeSamples() const noexcept { return consumeAtomic(m_eqFadeSamples, std::memory_order_acquire); }
    [[nodiscard]] bool isFading() const noexcept { return m_coordinator.isFading(); }
    // release: IR 変更フラグを Audio Thread の acquire で観測できるよう公開。
    void setIRChangeFlag() noexcept { publishAtomic(m_pendingIRChange, true, std::memory_order_release); }

    // acquire: Audio Thread の release (debugLastCreatedEqHash publish) と HB し、診断ハッシュを取得。
    [[nodiscard]] uint64_t getLastCreatedEqHashForDebug() const noexcept { return consumeAtomic(rtAuxMutable_.debugLastCreatedEqHash, std::memory_order_acquire); }

    void setSoftClipEnabled(bool enabled);
    [[nodiscard]] bool isSoftClipEnabled() const;

    void setSaturationAmount(float amount);
    [[nodiscard]] float getSaturationAmount() const;

    [[nodiscard]] bool isShutdownInProgress() const noexcept
    {
        // acquire: lifecycleState の setShutdownPhase/releaseResources release 側と HB し、
        //          Releasing/Destroyed 遷移を各スレッドから安全に観測する。
        const auto state = consumeAtomic(lifecycleState, std::memory_order_acquire);
        return state == EngineLifecycleState::Releasing
            || state == EngineLifecycleState::Destroyed;
    }

    [[nodiscard]] bool acceptsRuntimePublication() const noexcept;
    [[nodiscard]] bool isFullyDrained() noexcept;
    [[nodiscard]] bool waitForDrain(int timeoutMs = 2000, int pollIntervalMs = 2) noexcept;

    void drainDeferredRetireQueues(bool allowDuringShutdown) noexcept;

    void setOversamplingFactor(int factor);
    [[nodiscard]] int getOversamplingFactor() const;

    void setOversamplingType(OversamplingType type);
    [[nodiscard]] OversamplingType getOversamplingType() const;

    // ────────────────────────────────────────────────────────────────
    // 出力周波数フィルター設定 (Thread-safe)
    //
    // convHCMode / convLCMode: ① コンボルバー最終段の場合に使用
    // eqLPFMode              : ② EQ最終段の場合に使用
    // ────────────────────────────────────────────────────────────────
    void setConvHCFilterMode(convo::HCMode mode) noexcept;
    [[nodiscard]] convo::HCMode getConvHCFilterMode() const noexcept;

    void setConvLCFilterMode(convo::LCMode mode) noexcept;
    [[nodiscard]] convo::LCMode getConvLCFilterMode() const noexcept;

    void setEqLPFFilterMode(convo::HCMode mode) noexcept;
    [[nodiscard]] convo::HCMode getEqLPFFilterMode() const noexcept;

    // --- Adaptiveノイズシェイパー学習サポート ---
    void startNoiseShaperLearning(convo::NoiseShaperLearningMode mode, bool resume = false);
    void stopNoiseShaperLearning();
    void setNoiseShaperLearningMode(convo::NoiseShaperLearningMode mode);
    // acquire: setNoiseShaperLearningMode の release と HB し、最新の LearningMode を取得。
    [[nodiscard]] convo::NoiseShaperLearningMode getNoiseShaperLearningMode() const { return consumeAtomic(pendingLearningMode, std::memory_order_acquire); }
    [[nodiscard]] bool isNoiseShaperLearning() const;
    [[nodiscard]] const convo::NoiseShaperLearnerProgress& getNoiseShaperLearningProgress() const;
    [[nodiscard]] int copyNoiseShaperLearningHistory(double* outScores, int maxPoints) const noexcept;
    // 学習ワーカーが記録したエラーメッセージを返す（UI 表示用）。エラーなしは nullptr。
    [[nodiscard]] const char* getNoiseShaperLearningError() const noexcept;
    [[nodiscard]] static int getAdaptiveSampleRateBankCount() noexcept;

    struct RuntimeLifecycleDiagnostics
    {
        std::uint64_t publishCount = 0;
        std::uint64_t retireCount = 0;
        std::uint64_t reclaimCount = 0;
    };

    struct RuntimeBackpressureTelemetry
    {
        std::uint64_t retireQueueDepth = 0;
        std::uint64_t fallbackQueueDepth = 0;
        std::uint64_t quarantineResident = 0;
        std::uint64_t publicationBacklog = 0;
        std::uint64_t rebuildBacklog = 0;
        std::uint64_t saturationEnterCount = 0;
        std::uint64_t saturationExitCount = 0;
        std::uint64_t publicationRejectCount = 0;
        std::uint64_t rebuildCollapseCount = 0;
        double reclaimLatency = 0.0;
    };

    enum class ResidencyAuthority : uint8_t
    {
        PublicationCoordinator = 0,
        DeferredDeleteFallback,
        EpochRetire,
        ShutdownDrain
    };

    [[nodiscard]] RuntimeLifecycleDiagnostics getRuntimeLifecycleDiagnostics() const noexcept
    {
        return {
            consumeAtomic(rtAuxMutable_.runtimePublishCount),
            consumeAtomic(rtAuxMutable_.runtimeRetireCount),
            consumeAtomic(rtAuxMutable_.runtimeReclaimCount)
        };
    }

    [[nodiscard]] RuntimeBackpressureTelemetry getRuntimeBackpressureTelemetry() const noexcept
    {
        return {
            consumeAtomic(retireQueueDepth_, std::memory_order_acquire),
            consumeAtomic(fallbackQueueDepth_, std::memory_order_acquire),
            consumeAtomic(quarantineResident_, std::memory_order_acquire),
            consumeAtomic(publicationBacklog_, std::memory_order_acquire),
            consumeAtomic(rebuildBacklog_, std::memory_order_acquire),
            consumeAtomic(saturationEnterCount_, std::memory_order_acquire),
            consumeAtomic(saturationExitCount_, std::memory_order_acquire),
            consumeAtomic(publicationRejectCount_, std::memory_order_acquire),
            consumeAtomic(rebuildCollapseCount_, std::memory_order_acquire),
            consumeAtomic(reclaimLatency_, std::memory_order_acquire)
        };
    }

    struct CliProcessingTelemetrySnapshot
    {
        bool enabled = false;
        std::uint64_t callbackCount = 0;
        double lastProcessTimeUs = 0.0;
        double avgProcessTimeUs = 0.0;
        double maxProcessTimeUs = 0.0;
        int lastBlockSamples = 0;
        double sampleRateHz = 0.0;
    };

    struct RTLocalState
    {
        std::atomic<uint64_t> audioCallbackEpochCounter { 0 };
        std::atomic<uint64_t> audioSampleCursorCounter { 0 };
        std::atomic<uint32_t> audioCallbackActiveCount { 0 };
        std::atomic<uint64_t> audioThreadRetireEnqueueDropped { 0 };
        std::atomic<uint64_t> audioThreadRetireOverflowEpoch { 0 };
    };

    struct RTAuxMutable
    {
        uint64_t debugLastReportedCreatedEqHash { std::numeric_limits<uint64_t>::max() };
        int debugLastReportedDspReady { -1 };
        int debugLastReportedTransitionActive { -1 };
        int debugLastReportedTransitionPolicy { -1 };
        uint64_t debugLastReportedTransitionCurrentPtr { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedTransitionNextPtr { std::numeric_limits<uint64_t>::max() };
        double debugLastReportedTransitionFadeSec { -1.0 };
        int debugLastReportedTransitionLatencyDeltaSamples { std::numeric_limits<int>::max() };
        uint64_t debugLastReportedRuntimeSnapshotRevision { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedRuntimePublishCurrentUuid { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedRuntimePublishFadingUuid { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedRuntimePublishTransitionCurrentUuid { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedRuntimePublishTransitionNextUuid { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedRuntimePublishCount { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedRuntimeRetireCount { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedRuntimeReclaimCount { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedRebuildRequestCount { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedRebuildQueuedCount { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedRebuildBlockedPendingDuplicateCount { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedRebuildBlockedRecentDuplicateCount { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedRebuildRuntimeQueueFullCount { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedRebuildDrainedCommandCount { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedRebuildMatchedRuntimeCommandCount { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedRebuildTaskSnapshotFallbackCount { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedConvolverRebuildRequestCount { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedConvolverRebuildDeferredAfterLoadCount { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedConvolverRebuildScheduledCount { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedConvolverRebuildTriggeredCount { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedEqCacheSnapshotCreateMissCount { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedEqCacheRuntimeLookupMissCount { std::numeric_limits<uint64_t>::max() };
        int debugLastReportedShutdownPhase { -1 };
        bool debugEqCacheLookupMissLatched { false };
        uint64_t debugEqCacheLookupMissLatchedHash { 0 };
        uint32 fixedNoiseLastLogMs { 0 };
        int64_t lastQueuedTaskTicks { 0 };
        std::atomic<std::uint64_t> rebuildTelemetryNextIntentId { 1 };
        std::atomic<std::int64_t> lastEvidenceEmitHighResTicks { 0 };
        std::atomic<std::uint64_t> runtimePublishCount { 0 };
        std::atomic<std::uint64_t> runtimeRetireCount { 0 };
        std::atomic<std::uint64_t> runtimeReclaimCount { 0 };
        std::atomic<uint64_t> lastRejectedGenerationNonRt { 0 };
        std::atomic<std::uint64_t> debugRebuildDispatchRequestCount { 0 };
        std::atomic<std::uint64_t> debugRebuildDispatchQueuedCount { 0 };
        std::atomic<std::uint64_t> debugRebuildDispatchBlockedPendingDuplicateCount { 0 };
        std::atomic<std::uint64_t> debugRebuildDispatchBlockedRecentDuplicateCount { 0 };
        std::atomic<std::uint64_t> debugRebuildDispatchRuntimeQueueFullCount { 0 };
        std::atomic<std::uint64_t> debugRebuildDispatchDrainedCommandCount { 0 };
        std::atomic<std::uint64_t> debugRebuildDispatchMatchedRuntimeCommandCount { 0 };
        std::atomic<std::uint64_t> debugRebuildDispatchTaskSnapshotFallbackCount { 0 };
        std::atomic<std::uint64_t> eqCacheSnapshotCreateMissCountNonRt { 0 };
        std::atomic<std::uint64_t> eqCacheRuntimeLookupMissCountNonRt { 0 };
        std::atomic<uint64_t> debugLastCreatedEqHash { 0 };
        std::atomic<uint64_t> lastEnqueuedSnapshotDebounceKey { 0 };
        std::atomic<bool> hasLastEnqueuedSnapshotDebounceKey { false };
        std::atomic<bool> cliProcessingTelemetryEnabled { false };
        std::atomic<std::uint64_t> cliTelemetryCallbackCount { 0 };
        std::atomic<double> cliTelemetryAccumulatedUs { 0.0 };
        std::atomic<double> cliTelemetryMaxUs { 0.0 };
        std::atomic<double> cliTelemetryLastUs { 0.0 };
        std::atomic<int> cliTelemetryLastBlockSamples { 0 };
    };

    void setCliProcessingTelemetryEnabled(bool enabled) noexcept
    {
        publishAtomic(rtAuxMutable_.cliProcessingTelemetryEnabled, enabled, std::memory_order_release);
        if (!enabled)
        {
            exchangeAtomic(rtAuxMutable_.cliTelemetryCallbackCount, static_cast<std::uint64_t>(0), std::memory_order_acq_rel);
            exchangeAtomic(rtAuxMutable_.cliTelemetryAccumulatedUs, 0.0, std::memory_order_acq_rel);
            exchangeAtomic(rtAuxMutable_.cliTelemetryMaxUs, 0.0, std::memory_order_acq_rel);
            exchangeAtomic(rtAuxMutable_.cliTelemetryLastUs, 0.0, std::memory_order_acq_rel);
            exchangeAtomic(rtAuxMutable_.cliTelemetryLastBlockSamples, 0, std::memory_order_acq_rel);
        }
    }

    [[nodiscard]] bool isCliProcessingTelemetryEnabled() const noexcept
    {
        return consumeAtomic(rtAuxMutable_.cliProcessingTelemetryEnabled, std::memory_order_acquire);
    }

    [[nodiscard]] CliProcessingTelemetrySnapshot consumeCliProcessingTelemetrySnapshot() noexcept
    {
        CliProcessingTelemetrySnapshot snapshot {};
        snapshot.enabled = isCliProcessingTelemetryEnabled();
        snapshot.callbackCount = exchangeAtomic(rtAuxMutable_.cliTelemetryCallbackCount, static_cast<std::uint64_t>(0), std::memory_order_acq_rel);
        const double accumulatedUs = exchangeAtomic(rtAuxMutable_.cliTelemetryAccumulatedUs, 0.0, std::memory_order_acq_rel);
        snapshot.maxProcessTimeUs = exchangeAtomic(rtAuxMutable_.cliTelemetryMaxUs, 0.0, std::memory_order_acq_rel);
        snapshot.lastProcessTimeUs = consumeAtomic(rtAuxMutable_.cliTelemetryLastUs, std::memory_order_acquire);
        snapshot.lastBlockSamples = consumeAtomic(rtAuxMutable_.cliTelemetryLastBlockSamples, std::memory_order_acquire);
        snapshot.sampleRateHz = consumeAtomic(currentSampleRate, std::memory_order_acquire);
        if (snapshot.callbackCount > 0)
            snapshot.avgProcessTimeUs = accumulatedUs / static_cast<double>(snapshot.callbackCount);
        return snapshot;
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
    struct EqCacheMissDiagnostics
    {
        std::uint64_t snapshotCreateMissCount = 0;
        std::uint64_t runtimeLookupMissCount = 0;
    };

    [[nodiscard]] RebuildDispatchDiagnostics getRebuildDispatchDiagnostics() const noexcept
    {
        return {
            consumeAtomic(rtAuxMutable_.debugRebuildDispatchRequestCount),
            consumeAtomic(rtAuxMutable_.debugRebuildDispatchQueuedCount),
            consumeAtomic(rtAuxMutable_.debugRebuildDispatchBlockedPendingDuplicateCount),
            consumeAtomic(rtAuxMutable_.debugRebuildDispatchBlockedRecentDuplicateCount),
            consumeAtomic(rtAuxMutable_.debugRebuildDispatchRuntimeQueueFullCount),
            consumeAtomic(rtAuxMutable_.debugRebuildDispatchDrainedCommandCount),
            consumeAtomic(rtAuxMutable_.debugRebuildDispatchMatchedRuntimeCommandCount),
            consumeAtomic(rtAuxMutable_.debugRebuildDispatchTaskSnapshotFallbackCount)
        };
    }
    [[nodiscard]] EqCacheMissDiagnostics getEqCacheMissDiagnostics() const noexcept
    {
        return {
            consumeAtomic(rtAuxMutable_.eqCacheSnapshotCreateMissCountNonRt, std::memory_order_acquire),
            consumeAtomic(rtAuxMutable_.eqCacheRuntimeLookupMissCountNonRt, std::memory_order_acquire)
        };
    }

    EqCacheMissDiagnostics consumeEqCacheMissDiagnostics() noexcept
    {
        EqCacheMissDiagnostics snapshot {};
        snapshot.snapshotCreateMissCount = exchangeAtomic(rtAuxMutable_.eqCacheSnapshotCreateMissCountNonRt,
                                                          static_cast<std::uint64_t>(0),
                                                          std::memory_order_acq_rel);
        snapshot.runtimeLookupMissCount = exchangeAtomic(rtAuxMutable_.eqCacheRuntimeLookupMissCountNonRt,
                                                         static_cast<std::uint64_t>(0),
                                                         std::memory_order_acq_rel);
        return snapshot;
    }

    // --- NoiseShaperLearner Settings ---
    [[nodiscard]] convo::NoiseShaperLearnerSettings getNoiseShaperLearnerSettings() const;
    void setNoiseShaperLearnerSettings(const convo::NoiseShaperLearnerSettings& settings);

    [[nodiscard]] static double getAdaptiveSampleRateBankHz(int bankIndex) noexcept;
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
    [[nodiscard]] static int getAdaptiveCoeffBankIndex(double sampleRate, int bitDepth, convo::NoiseShaperLearningMode mode) noexcept;

    [[nodiscard]] bool getAdaptiveNoiseShaperState(int bankIndex, convo::NoiseShaperLearnerState& outState) const noexcept;
    void setAdaptiveNoiseShaperState(int bankIndex, const convo::NoiseShaperLearnerState& inState) noexcept;

private:
    static constexpr int kAudioEpochReaderIndex = 0;
    static constexpr int kControlEpochReaderIndex = 1;
    static constexpr int kCommitProducerEpochReaderIndex = 2;
    static constexpr int kCommitConsumerEpochReaderIndex = 3;

    void recordAudioCallbackProcessingStats(int numSamples, double processTimeUs) noexcept
    {
        if (!consumeAtomic(rtAuxMutable_.cliProcessingTelemetryEnabled, std::memory_order_relaxed))
            return;

        convo::fetchAddAtomic(rtAuxMutable_.cliTelemetryCallbackCount, static_cast<std::uint64_t>(1), std::memory_order_relaxed);
        publishAtomic(rtAuxMutable_.cliTelemetryLastBlockSamples, numSamples, std::memory_order_relaxed);
        publishAtomic(rtAuxMutable_.cliTelemetryLastUs, processTimeUs, std::memory_order_relaxed);

        double expectedAccum = consumeAtomic(rtAuxMutable_.cliTelemetryAccumulatedUs, std::memory_order_relaxed);
        while (!convo::compareExchangeAtomic(rtAuxMutable_.cliTelemetryAccumulatedUs,
                             expectedAccum,
                             expectedAccum + processTimeUs,
                             std::memory_order_relaxed,
                             std::memory_order_relaxed))
        {
        }

        double expectedMax = consumeAtomic(rtAuxMutable_.cliTelemetryMaxUs, std::memory_order_relaxed);
         while (processTimeUs > expectedMax
             && !convo::compareExchangeAtomic(rtAuxMutable_.cliTelemetryMaxUs,
                                  expectedMax,
                                  processTimeUs,
                                  std::memory_order_relaxed,
                                  std::memory_order_relaxed))
        {
        }
    }

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
        [[nodiscard]] bool containsNonRt(uint64_t hash) noexcept;
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
                        entry.second->release(owner->m_epochDomain);
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
        [[nodiscard]] bool tryEnqueueDeferredMap(CacheMap* map) noexcept;

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
    // active runtime DSP slot: 現行 DSP の非所有スロット。
    //            実際の解放は retireDSP() → deferred delete / retire queue で行う。
    convo::NonOwningPtr<DSPCore> activeRuntimeDSPSlot { nullptr };
    // fading runtime DSP slot: フェード中 DSP の非所有スロット。
    //               寿命は publish/retire の順序に従い、active runtime slot と独立して非所有で管理する。
    convo::NonOwningPtr<DSPCore> fadingRuntimeDSPSlot { nullptr };

    inline DSPCore* exchangeFadingRuntimeDSP(DSPCore* value) noexcept
    {


        DSPCore* previous = fadingRuntimeDSPSlot.get();
        fadingRuntimeDSPSlot.operator=(value);
        return previous;
    }

    [[nodiscard]] inline DSPCore* getActiveRuntimeDSP() const noexcept
    {
        return activeRuntimeDSPSlot.get();
    }

    inline void setActiveRuntimeDSP(DSPCore* value) noexcept
    {
        activeRuntimeDSPSlot = value;
    }

    [[nodiscard]] inline bool hasActiveRuntimeDSP() const noexcept
    {
        return getActiveRuntimeDSP() != nullptr;
    }

    inline DSPCore* releaseActiveRuntimeDSP() noexcept
    {
        DSPCore* activeRaw = getActiveRuntimeDSP();
        setActiveRuntimeDSP(nullptr);
        return activeRaw;
    }

    struct RuntimePublishView
    {
        RuntimePublishView(convo::EpochDomain& domain,
                           int readerIndex,
                           const convo::RuntimeGraph* graphIn,
                           const convo::TransitionState& transitionIn) noexcept
            : observed(domain, readerIndex), graph(graphIn)
            , transition(transitionIn)
        {
        }

        RuntimePublishView(const RuntimePublishView&) = delete;
        RuntimePublishView& operator=(const RuntimePublishView&) = delete;
        RuntimePublishView(RuntimePublishView&&) noexcept = default;
        RuntimePublishView& operator=(RuntimePublishView&&) noexcept = default;

        convo::ObservedRuntime observed;
        const convo::RuntimeGraph* graph = nullptr;
        convo::TransitionState transition {};
    };

    struct RuntimeReadView
    {
        RuntimeReadView(RuntimePublishView&& runtimePublishIn,
                             convo::ObservedRuntime&& observedSnapshotIn) noexcept
            : runtimePublish(std::move(runtimePublishIn))
            , observedSnapshot(std::move(observedSnapshotIn))
            , graph(runtimePublish.graph)
            , snapshot(observedSnapshot.get())
        {
        }

        RuntimeReadView(const RuntimeReadView&) = delete;
        RuntimeReadView& operator=(const RuntimeReadView&) = delete;
        RuntimeReadView(RuntimeReadView&&) noexcept = default;
        RuntimeReadView& operator=(RuntimeReadView&&) noexcept = default;

        RuntimePublishView runtimePublish;
        convo::ObservedRuntime observedSnapshot;
        const convo::RuntimeGraph* graph = nullptr;
        const convo::GlobalSnapshot* snapshot = nullptr;
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
    std::atomic<int64_t> deferredFinalizeFirstSeenTicks_ { 0 };
    std::atomic<bool> pendingChangeNotification { false };
    // 同一IR構造に対する Structural rebuild の多重発火を抑止する。
    // 値は「直近で rebuild を要求した UI 側 Convolver 構造ハッシュ」。
    std::atomic<uint64_t> lastIssuedConvolverStructuralHash_{ 0 };
    // active runtime slot 実体を直接読まずに判定するための、commit済み Convolver 構造スナップショット。
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
    alignas(64) std::atomic<bool> adaptiveCaptureActiveRt { false };
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

    enum class CommitReaderSlot : int;

    //----------------------------------------------------------
    // プライベートヘルパー (Message Thread のみ)
    //----------------------------------------------------------
    void applyDefaultsForCurrentMode();

    //----------------------------------------------------------
    // ヘルパー関数
    //----------------------------------------------------------
    // Note: This function performs memory allocation (including MKL) and other blocking operations
    // such as IR resampling. It MUST only be called from the message thread.
    // prepareToPlay() routes to the message thread before invoking this helper.
    void requestRebuild(double sampleRate, int samplesPerBlock, bool forceMustExecute = false);
    void commitNewDSP(DSPCore* newDSP, int generation);
    void prepareCommit(DSPCore* newDSP, int generation);
    void executeCommit();
    // acquire: commitNewDSP/requestRebuild の rebuildGeneration 更新 release と HB し、
    //          リビルド世代が古いか否かを各スレッドから安全に判定。
    [[nodiscard]] bool isRebuildObsolete(int generation) const { return generation != consumeAtomic(rebuildGeneration, std::memory_order_acquire); }
    bool enqueueLearningCommand(const LearningCommand& cmd) noexcept;
    [[nodiscard]] bool dequeueLearningCommand(LearningCommand& cmd) noexcept;
    bool enqueueLearnerDispatch(const LearnerDispatchAction& action) noexcept;
    [[nodiscard]] bool dequeueLearnerDispatch(LearnerDispatchAction& action) noexcept;
    void processLearningCommands() noexcept;
    void processDeferredLearningActions();
    void resetLearningControlState() noexcept;
    bool enqueueSnapshotCommand() noexcept;
    void appendPublicationIntentForCommitProducer(DSPCore* newDSP, int generation) noexcept;
    void appendPublicationIntentForCommitConsumer(DSPCore* newDSP, int generation) noexcept;
    void appendPublicationIntentForCommitSlot(DSPCore* newDSP, int generation, CommitReaderSlot readerSlot) noexcept;
    void drainPublicationLogForShutdown() noexcept;
    [[nodiscard]] bool hasPendingPublicationIntents() noexcept;
    [[nodiscard]] bool hasPublicationLogPending() noexcept;
    void processWithSnapshot(const juce::AudioSourceChannelInfo& bufferToFill,
                             const convo::GlobalSnapshot* snap,
                             bool isFadingTarget,
                             const convo::RuntimeGraph* runtimeGraphHint = nullptr);
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
        convo::RuntimeBuildSnapshot runtimeBuildSnapshot {};
        int generation = 0;
    };
    RebuildTask pendingTask;
    RebuildTask lastQueuedTaskSignature;

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
    [[nodiscard]] const AdaptiveCoeffBankSlot& getAdaptiveCoeffBankForIndex(int bankIndex) const noexcept;
    void selectAdaptiveCoeffBankForCurrentSettings() noexcept;
    void publishCoeffsToBank(int bankIndex, const double* coeffs);

    static constexpr uint32_t rebuildReasonMask(RebuildReason reason) noexcept
    {
        return static_cast<uint32_t>(reason);
    }

    [[nodiscard]] bool hasRebuildReason(RebuildReason reason) const noexcept
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

    [[nodiscard]] uint64_t nextRebuildTelemetryIntentId() noexcept
    {
        return convo::fetchAddAtomic(rtAuxMutable_.rebuildTelemetryNextIntentId, static_cast<uint64_t>(1), std::memory_order_acq_rel);
    }

    enum class RebuildTelemetryEvent : uint8_t
    {
        Requested,
        Merged,
        Suppressed,
        Deferred,
        ForcedDispatch,
        Dispatched
    };

    enum class RebuildTelemetryReason : uint8_t
    {
        ConvolverParamsChanged,
        MixedPhaseIntermediate,
        HashDedup,
        PreparedIRApplyWindow,
        SnapshotEnqueueFailed,
        SnapshotEnqueued,
        RequestRebuildKindEntry,
        UiEqEditorChangeListener,
        PrepareToPlayNonMt,
        RebuildThreadWarmupRetry,
        ShutdownInProgress,
        KindFiltered,
        DelegateRequestRebuildSrBs,
        MissingSrBs,
        NonMtTriggerAsync,
        NonMtAlreadyPending,
        AsyncBridgeConsume,
        AsyncBridgeDelegateSrBs,
        AsyncBridgeMissingSrBs,
        RequestRebuildSrBs,
        DeferredStructuralWindow,
        TaskQueued,
        RecentDuplicate,
        PendingDuplicate,
        DeferredStructuralDue,
        DeferredStructuralRebuildRequested,
        DeferredFinalizeReady,
        DeferredFinalizeRebuildRequested,
        EnqueueSnapshotCommand,
        SnapshotIntentDebounced,
        SnapshotCommandBufferFull,
        SnapshotCommandQueued,
        SnapshotCommandBufferFullNonMt,
        SnapshotCommandQueuedNonMt,
        SameAsPendingWouldMerge
    };

    enum class RebuildTelemetryClass : uint8_t
    {
        NA,
        Structural,
        FinalizeAware,
        Snapshot
    };

    enum class RebuildTelemetryPolicy : uint8_t
    {
        NA,
        Replaceable,
        MustExecute
    };
    enum class RebuildTelemetryDecision : uint8_t
    {
        Accepted,
        Suppressed,
        Deferred,
        Dispatched,
        Merged,
        Dropped,
        Released
    };

    static const char* toTelemetryEventString(RebuildTelemetryEvent value) noexcept
    {
        switch (value)
        {
            case RebuildTelemetryEvent::Requested: return "REBUILD_REQUESTED";
            case RebuildTelemetryEvent::Merged: return "REBUILD_MERGED";
            case RebuildTelemetryEvent::Suppressed: return "REBUILD_SUPPRESSED";
            case RebuildTelemetryEvent::Deferred: return "REBUILD_DEFERRED";
            case RebuildTelemetryEvent::ForcedDispatch: return "REBUILD_FORCED_DISPATCH";
            case RebuildTelemetryEvent::Dispatched: return "REBUILD_DISPATCHED";
        }
        return "REBUILD_UNKNOWN";
    }

    static const char* toTelemetryReasonString(RebuildTelemetryReason value) noexcept
    {
        switch (value)
        {
            case RebuildTelemetryReason::ConvolverParamsChanged: return "convolver_params_changed";
            case RebuildTelemetryReason::MixedPhaseIntermediate: return "mixed_phase_intermediate";
            case RebuildTelemetryReason::HashDedup: return "hash_dedup";
            case RebuildTelemetryReason::PreparedIRApplyWindow: return "prepared_ir_apply_window";
            case RebuildTelemetryReason::SnapshotEnqueueFailed: return "snapshot_enqueue_failed";
            case RebuildTelemetryReason::SnapshotEnqueued: return "snapshot_enqueued";
            case RebuildTelemetryReason::RequestRebuildKindEntry: return "requestRebuild_kind_entry";
            case RebuildTelemetryReason::UiEqEditorChangeListener: return "ui_eq_editor_change_listener";
            case RebuildTelemetryReason::PrepareToPlayNonMt: return "prepare_to_play_non_mt";
            case RebuildTelemetryReason::RebuildThreadWarmupRetry: return "rebuild_thread_warmup_retry";
            case RebuildTelemetryReason::ShutdownInProgress: return "shutdown_in_progress";
            case RebuildTelemetryReason::KindFiltered: return "kind_filtered";
            case RebuildTelemetryReason::DelegateRequestRebuildSrBs: return "delegate_requestRebuild_sr_bs";
            case RebuildTelemetryReason::MissingSrBs: return "missing_sr_bs";
            case RebuildTelemetryReason::NonMtTriggerAsync: return "non_mt_trigger_async";
            case RebuildTelemetryReason::NonMtAlreadyPending: return "non_mt_already_pending";
            case RebuildTelemetryReason::AsyncBridgeConsume: return "async_bridge_consume";
            case RebuildTelemetryReason::AsyncBridgeDelegateSrBs: return "async_bridge_delegate_sr_bs";
            case RebuildTelemetryReason::AsyncBridgeMissingSrBs: return "async_bridge_missing_sr_bs";
            case RebuildTelemetryReason::RequestRebuildSrBs: return "requestRebuild_sr_bs";
            case RebuildTelemetryReason::DeferredStructuralWindow: return "deferred_structural_window";
            case RebuildTelemetryReason::TaskQueued: return "task_queued";
            case RebuildTelemetryReason::RecentDuplicate: return "recent_duplicate";
            case RebuildTelemetryReason::PendingDuplicate: return "pending_duplicate";
            case RebuildTelemetryReason::DeferredStructuralDue: return "deferred_structural_due";
            case RebuildTelemetryReason::DeferredStructuralRebuildRequested: return "deferred_structural_rebuild_requested";
            case RebuildTelemetryReason::DeferredFinalizeReady: return "deferred_finalize_ready";
            case RebuildTelemetryReason::DeferredFinalizeRebuildRequested: return "deferred_finalize_rebuild_requested";
            case RebuildTelemetryReason::EnqueueSnapshotCommand: return "enqueue_snapshot_command";
            case RebuildTelemetryReason::SnapshotIntentDebounced: return "snapshot_intent_debounced";
            case RebuildTelemetryReason::SnapshotCommandBufferFull: return "snapshot_command_buffer_full";
            case RebuildTelemetryReason::SnapshotCommandQueued: return "snapshot_command_queued";
            case RebuildTelemetryReason::SnapshotCommandBufferFullNonMt: return "snapshot_command_buffer_full_non_mt";
            case RebuildTelemetryReason::SnapshotCommandQueuedNonMt: return "snapshot_command_queued_non_mt";
            case RebuildTelemetryReason::SameAsPendingWouldMerge: return "same_as_pending_would_merge";
        }
        return "unknown_reason";
    }

    static const char* toTelemetryClassString(RebuildTelemetryClass value) noexcept
    {
        switch (value)
        {
            case RebuildTelemetryClass::NA: return "N/A";
            case RebuildTelemetryClass::Structural: return "Structural";
            case RebuildTelemetryClass::FinalizeAware: return "FinalizeAware";
            case RebuildTelemetryClass::Snapshot: return "Snapshot";
        }
        return "N/A";
    }

    static const char* toTelemetryPolicyString(RebuildTelemetryPolicy value) noexcept
    {
        switch (value)
        {
            case RebuildTelemetryPolicy::NA: return "N/A";
            case RebuildTelemetryPolicy::Replaceable: return "Replaceable";
            case RebuildTelemetryPolicy::MustExecute: return "MustExecute";
        }
        return "N/A";
    }

    static const char* toTelemetryDecisionString(RebuildTelemetryDecision value) noexcept
    {
        switch (value)
        {
            case RebuildTelemetryDecision::Accepted: return "accepted";
            case RebuildTelemetryDecision::Suppressed: return "suppressed";
            case RebuildTelemetryDecision::Deferred: return "deferred";
            case RebuildTelemetryDecision::Dispatched: return "dispatched";
            case RebuildTelemetryDecision::Merged: return "merged";
            case RebuildTelemetryDecision::Dropped: return "dropped";
            case RebuildTelemetryDecision::Released: return "released";
        }
        return "unknown";
    }

    void emitRebuildTelemetry(RebuildTelemetryEvent eventName,
                              uint64_t intentId,
                              RebuildTelemetryReason reason,
                              RebuildTelemetryDecision decision,
                              uint64_t hash = 0,
                              uint64_t fingerprint = 0,
                              RebuildTelemetryClass rebuildClass = RebuildTelemetryClass::NA,
                              RebuildTelemetryPolicy collapsePolicy = RebuildTelemetryPolicy::NA,
                              const char* finalizeState = "N/A",
                              double latencyMs = -1.0) const noexcept
    {
        juce::String log = "[REBUILD_TELEMETRY] event=";
        log += juce::String(toTelemetryEventString(eventName));
        log += " intentId=" + juce::String(static_cast<juce::int64>(intentId));
        log += " reason=" + juce::String(toTelemetryReasonString(reason));
        log += " class=" + juce::String(toTelemetryClassString(rebuildClass));
        log += " policy=" + juce::String(toTelemetryPolicyString(collapsePolicy));
        log += " hash=0x" + juce::String::toHexString(static_cast<juce::int64>(hash));
        log += " fingerprint=0x" + juce::String::toHexString(static_cast<juce::int64>(fingerprint));
        log += " finalizeState=" + juce::String(finalizeState != nullptr ? finalizeState : "N/A");
        log += " decision=" + juce::String(toTelemetryDecisionString(decision));
        if (latencyMs >= 0.0)
            log += " latencyMs=" + juce::String(latencyMs, 3);

        DBG(log);
        juce::Logger::writeToLog(log);
    }

    void submitRebuildIntent(convo::RebuildKind kind,
                             RebuildTelemetryReason reason,
                             RebuildTelemetryClass rebuildClass = RebuildTelemetryClass::NA,
                             RebuildTelemetryPolicy collapsePolicy = RebuildTelemetryPolicy::NA) noexcept;

    struct RebuildAdmissionIntentState
    {
        bool valid = false;
        convo::RebuildKind kind = convo::RebuildKind::None;
        RebuildTelemetryClass rebuildClass = RebuildTelemetryClass::NA;
        RebuildTelemetryPolicy collapsePolicy = RebuildTelemetryPolicy::NA;
        std::uint32_t fingerprintVersion = 1u;
        uint64_t structuralHash = 0;
        uint64_t fingerprint = 0;
        bool deferCategory = false;
        int64_t lastIntentTicks = 0;
    };



    void debugAssertNotAudioThread() const;
    void debugAssertAudioThread() const;

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

        const auto runtimeReadView = readControlRuntimeView();
        DSPCore* fading = resolveFadingRuntimeDSPFromRuntimeWorldOnly(getRuntimeGraph(runtimeReadView));

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

    [[nodiscard]] inline CrossfadePreparedSnapshot consumeCrossfadePreparedSnapshot() const noexcept
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

    inline void publishLatencyDelayAtomics(int oldDelay,
                                           int newDelay) noexcept
    {
        convo::publishAtomic(latencyDelayOld, oldDelay, std::memory_order_release); // release: audio thread の latency read と HB
        convo::publishAtomic(latencyDelayNew, newDelay, std::memory_order_release); // release: audio thread の latency read と HB
    }

    inline void resetLatencyDelayRtState() noexcept
    {
    }

    enum class CommitReaderSlot : int
    {
        Producer = kCommitProducerEpochReaderIndex,
        Consumer = kCommitConsumerEpochReaderIndex
    };

    [[nodiscard]] static constexpr int toCommitReaderIndex(CommitReaderSlot slot) noexcept
    {
        switch (slot)
        {
            case CommitReaderSlot::Producer:
                return kCommitProducerEpochReaderIndex;
            case CommitReaderSlot::Consumer:
                return kCommitConsumerEpochReaderIndex;
        }

        jassertfalse;
        return kCommitConsumerEpochReaderIndex;
    }

    [[nodiscard]] inline RuntimePublishView makeRuntimePublishView(int readerIndex,
                                                                   bool assertAudioThread) noexcept
    {
        if (assertAudioThread)
            debugAssertAudioThread();
        else
            debugAssertNotAudioThread();

        const auto* world = RuntimePublicationCoordinator::observePublishedWorld(runtimeStore);
        return RuntimePublishView {
            m_epochDomain,
            readerIndex,
            world != nullptr ? &world->graph : nullptr,
            world != nullptr ? world->engine.transition : convo::TransitionState{}
        };
    }

    [[nodiscard]] inline RuntimeReadView makeRuntimeReadView(int readerIndex,
                                                             bool assertAudioThread) noexcept
    {
        return RuntimeReadView {
            makeRuntimePublishView(readerIndex, assertAudioThread),
            m_coordinator.observeCurrentRuntime(readerIndex)
        };
    }

    [[nodiscard]] inline RuntimeReadView readAudioRuntimeView() noexcept
    {
        return makeRuntimeReadView(kAudioEpochReaderIndex, true);
    }

    [[nodiscard]] inline RuntimeReadView readControlRuntimeView() noexcept
    {
        return makeRuntimeReadView(kControlEpochReaderIndex, false);
    }

    [[nodiscard]] static inline const convo::GlobalSnapshot* getRuntimeSnapshot(const RuntimeReadView& runtimeReadView) noexcept
    {
        return runtimeReadView.snapshot;
    }

    [[nodiscard]] static inline const convo::RuntimeGraph* getRuntimeGraph(const RuntimeReadView& runtimeReadView) noexcept
    {
        return runtimeReadView.runtimePublish.graph;
    }

    inline convo::RuntimeGraph makeRuntimeGraphState(const convo::EngineRuntime& state) noexcept
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

        const auto runtimeReadView = readControlRuntimeView();
        const auto* snapshot = getRuntimeSnapshot(runtimeReadView);
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

        return graph;
    }

    template <typename T>
    static inline void publishAtomicPtr(std::atomic<T*>& dst, T* value) noexcept
    {
        // release: pointer を consumeAtomicPtr acquire に公開。
        convo::publishAtomic(dst, value, std::memory_order_release);
    }

    template <typename T>
    [[nodiscard]] static inline T* consumeAtomicPtr(const std::atomic<T*>& src) noexcept
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

    [[nodiscard]] inline DSPCore* resolveFadingRuntimeDSPFromRuntimeWorldOnly(const convo::RuntimeGraph* runtimeGraph) const noexcept
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

    [[nodiscard]] inline DSPCore* resolveActiveRuntimeDSPFromRuntimeWorldOnly(const convo::RuntimeGraph* runtimeGraph) const noexcept
    {
        if (runtimeGraph == nullptr)
            return nullptr;

        auto* graphActive = static_cast<DSPCore*>(runtimeGraph->activeNode);
        const auto graphRuntimeUuid = runtimeGraph->runtimeUuid;
        if (graphActive != nullptr
            && graphRuntimeUuid != 0)
            return graphActive;

        return nullptr;
    }


    [[nodiscard]] inline std::uint64_t reserveNextRuntimeGraphGeneration() noexcept
    {
        // acq_rel: fetch-add で generation counter を advance し、他スレッドへ新 generation を公開。
        //          acquire 側で旧 generation 参照により stale request 検出を回避。
        const auto nextGraphGeneration = convo::fetchAddAtomic(runtimeGraphRevision,
                                                               static_cast<std::uint64_t>(1),
                                                               std::memory_order_acq_rel) + 1;
        // acq_rel: publish count を increment し、runtime world 公開イベント数を追跡。
        convo::fetchAddAtomic(rtAuxMutable_.runtimePublishCount,
                              static_cast<std::uint64_t>(1),
                              std::memory_order_acq_rel);
        return nextGraphGeneration;
    }

    static std::atomic<std::uint64_t> runtimeVersionCounterStorage_;
    static std::atomic<std::uint64_t>& runtimeVersionCounter() noexcept;
    [[nodiscard]] static std::uint64_t reserveNextRuntimeVersion() noexcept;

    //=== RuntimePublicationCoordinator NonRT helper API ===//
    // AudioEngine 内部の publish/retire helper（NonRT 専用）。

    [[nodiscard]] convo::aligned_unique_ptr<RuntimePublishWorld>
    buildRuntimePublishWorld(DSPCore* current,
                             DSPCore* next,
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
        worldOwner->assertMutable();
        worldOwner->generation = nextGraphGeneration;
        worldOwner->engine = engineState;
        worldOwner->graph = graphState;
        // runtimeVersion: monotonically increasing version number (magna_carta.md Section 2)
        worldOwner->runtimeVersion = reserveNextRuntimeVersion();
        // transitionId: unique per crossfade event
        worldOwner->transitionId = nextGraphGeneration + (active ? 0x1000000000000000ULL : 0);
        // Publish world must be frozen before it can be exposed to any reader.
        worldOwner->freeze();

        return worldOwner;
    }

    [[nodiscard]] bool runPublicationPrecheckNonRt(const RuntimePublishWorld& world) noexcept;
    void onRuntimePublishedNonRt(const RuntimePublishWorld& world) noexcept;
    void onRuntimeRetiredNonRt(const RuntimePublishWorld* world) noexcept;
    void emitEvidenceTickNonRt(bool force) noexcept;



    //=== End RuntimePublicationCoordinator NonRT helper API ===//

    class RuntimePublicationBridge final
    {
    public:
        explicit RuntimePublicationBridge(AudioEngine& engine) noexcept
            : engine_(&engine)
        {
        }

        [[nodiscard]] convo::aligned_unique_ptr<RuntimePublishWorld>
        buildRuntimePublishWorld(DSPCore* current,
                                 DSPCore* next,
                                 convo::TransitionPolicy policy,
                                 double fadeTimeSec,
                                 bool active) noexcept
        {
            return engine_->buildRuntimePublishWorld(current, next, policy, fadeTimeSec, active);
        }

        [[nodiscard]] bool validatePublicationNonRt(const RuntimePublishWorld& world) noexcept
        {
            return engine_->runPublicationPrecheckNonRt(world);
        }

        void didPublishRuntimeNonRt(const RuntimePublishWorld& world) noexcept
        {
            engine_->onRuntimePublishedNonRt(world);
        }

        void willRetireRuntimeNonRt(const RuntimePublishWorld* world) noexcept
        {
            if (world == nullptr)
                return;

            if (engine_->isShutdownInProgress())
                return;

            engine_->onRuntimeRetiredNonRt(world);
        }

        void retireRuntimePublishWorldNonRt(RuntimePublishWorld* world, bool resetRevision) noexcept
        {
            if (world == nullptr)
                return;

            engine_->enqueueDeferredDeleteNonRt(world, [](void* p)
            {
                auto* ptr = static_cast<RuntimePublishWorld*>(p);
                ptr->~RuntimePublishWorld();
                convo::aligned_free(ptr);
            });
            if (resetRevision)
            {
                convo::publishAtomic(engine_->runtimeGraphRevision,
                                     static_cast<std::uint64_t>(0),
                                     std::memory_order_release);
            }
        }

    private:
        AudioEngine* engine_ = nullptr;
    };

    using RuntimePublicationCoordinator = convo::RuntimePublicationCoordinator<RuntimePublishWorld,
                                                                               DSPCore*,
                                                                               RuntimePublicationBridge>;

    static_assert(!std::is_copy_constructible_v<RuntimePublicationCoordinator>,
                  "RuntimePublicationCoordinator must remain move-only");
    static_assert(!std::is_copy_assignable_v<RuntimePublicationCoordinator>,
                  "RuntimePublicationCoordinator must remain move-only");
    static_assert(std::is_move_constructible_v<RuntimePublicationCoordinator>,
                  "RuntimePublicationCoordinator must remain move-constructible");

    using RuntimePublishStore = RuntimePublicationCoordinator::Store;

    RuntimePublishStore runtimeStore;

    [[nodiscard]] inline RuntimePublicationCoordinator makeRuntimePublicationCoordinator() noexcept
    {
        using RuntimePublicationCoordinatorFactory = RuntimePublicationCoordinator;
        return RuntimePublicationCoordinatorFactory::create(RuntimePublicationBridge { *this }, runtimeStore);
    }

    [[nodiscard]] inline bool precheckRuntimePublication(const convo::isr::PayloadClosureDescriptor& closure,
                                                         const convo::isr::TieredPayloadDescriptor& descriptor) noexcept
    {
        auto& runtimePublicationCoordinator = runtimePublicationBridge_;
        return runtimePublicationCoordinator.precheckPublish(closure, descriptor);
    }

    inline void commitRuntimePublication(const RuntimePublishWorld& world) noexcept
    {
        auto& runtimePublicationCoordinator = runtimePublicationBridge_;
        runtimePublicationCoordinator.commit(convo::isr::PublishAuthority::Granted,
                                             convo::isr::RuntimeBoundary::NonRTWorld,
                                             &world,
                                             world.runtimeVersion);
    }

    inline void retireRuntimePublication(const RuntimePublishWorld* world) noexcept
    {
        auto& runtimePublicationCoordinator = runtimePublicationBridge_;
        runtimePublicationCoordinator.retire(convo::isr::RetireAuthority::Granted,
                                             convo::isr::RuntimeBoundary::NonRTWorld,
                                             world);
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

        DSPCore* current = getActiveRuntimeDSP();
        const auto* publishedWorld = RuntimePublicationCoordinator::observePublishedWorld(runtimeStore);
        const auto* runtimeGraph = (publishedWorld != nullptr) ? &publishedWorld->graph : nullptr;
        auto* fading = resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeGraph);
        const auto revision = (runtimeGraph != nullptr) ? runtimeGraph->generation : 0;
        const auto publishedCurrentUuid = (runtimeGraph != nullptr) ? runtimeGraph->runtimeUuid : 0;
        const auto publishedFadingUuid = (runtimeGraph != nullptr) ? runtimeGraph->fadingRuntimeUuid : 0;

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
        snapshot.adaptiveCaptureEnabled = consumeAtomic(adaptiveCaptureActiveRt, std::memory_order_acquire);
        if (snap != nullptr)
        {
            snapshot.snapshotEqParams = &snap->eqParams;
            snapshot.snapshotEqCoeffHash = snap->eqCoeffHash;
        }
        return snapshot;
    }

    inline DSPCore::ProcessingState buildAudioThreadProcessingState(DSPCore* dsp,
                                                                     const EngineParameterSnapshot& snapshot) noexcept
    {
        const convo::EQParameters* eqParams = snapshot.snapshotEqParams;
        uint64_t eqCoeffHash = snapshot.snapshotEqCoeffHash;

        const EQCoeffCache* eqCache = (eqParams != nullptr && eqCoeffHash != 0)
            ? eqCacheManager.get(eqCoeffHash)
            : nullptr;
        // ISR厳密化: hash から cache を解決できない場合は EQ を fail-close で bypass する。
        // 非snapshot系フォールバックへ降りるより、公開済み不変状態を維持することを優先する。
        const bool eqBypassedFailClosed = snapshot.eqBypassed || (eqParams == nullptr) || (eqCache == nullptr);

        return DSPCore::ProcessingState {
            .eqBypassed = eqBypassedFailClosed,
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
            .adaptiveCaptureQueue = snapshot.adaptiveCaptureEnabled ? &audioCaptureQueue : nullptr,
            .eqParams = eqParams,
            .eqCache = eqCache,
            .eqCoeffHash = eqCoeffHash
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

    inline void armCrossfadeIfPending(bool hasFading,
                                      bool& useDryAsOld,
                                      const CrossfadePreparedSnapshot& prepared) noexcept
    {
        const bool hasPendingCrossfade = prepared.pending;

        if (!hasPendingCrossfade)
        {
            dspCrossfadeArmed_RT = false;
            dspCrossfadeStartDelayBlocks_RT = 0;
            return;
        }

        const bool firstLoadDryPending = prepared.firstIrDryCrossfadePending;

        if ((hasFading || firstLoadDryPending)
            && !dspCrossfadeArmed_RT)
        {
            dspCrossfadeArmed_RT = true;
            dspCrossfadeStartDelayBlocks_RT = prepared.startDelayBlocks;

            // C1-2: activate のみ Audio Thread で実行 (reset/setCurrentAndTargetValue は Message Thread 側)
            dspCrossfadeGain.setTargetValue(1.0);

            if (firstLoadDryPending)
            {
                useDryAsOld = true;
                dspCrossfadeDryScaleGain.setTargetValue(prepared.dryScaleTarget);
            }
        }
    }

    template <typename ProcessFn>
    inline bool processCrossfadeDelayGateIfPending(DSPCore* fading,
                                                   bool useDryAsOld,
                                                   const CrossfadePreparedSnapshot& prepared,
                                                   ProcessFn processFn) noexcept
    {
        if (fading != nullptr
            && !useDryAsOld
            && prepared.pending
            && dspCrossfadeStartDelayBlocks_RT > 0)
        {
            --dspCrossfadeStartDelayBlocks_RT;
            processFn();
            return true;
        }

        return false;
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
                                             bool runtimeResetPending) noexcept
    {
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

    template <typename SampleType, typename MixFn>
    inline void runLatencyAlignedCrossfadeMixLoop(SampleType* dstL,
                                                  SampleType* dstR,
                                                  const SampleType* oldL,
                                                  const SampleType* oldR,
                                                  int numSamples,
                                                  int delayOld,
                                                  int delayNew,
                                                  bool runtimeResetPending,
                                                  MixFn mixFn) noexcept
    {
        const int bufferSize = latencyBufSize;
        int writePos = latencyWritePos;

        resetLatencyBuffersIfPending(bufferSize, writePos, runtimeResetPending);

        for (int i = 0; i < numSamples; ++i)
        {
            latencyBufOldL[writePos] = (oldL != nullptr) ? static_cast<double>(oldL[i]) : 0.0;
            latencyBufOldR[writePos] = (oldR != nullptr) ? static_cast<double>(oldR[i]) : 0.0;
            latencyBufNewL[writePos] = (dstL != nullptr) ? static_cast<double>(dstL[i]) : 0.0;
            latencyBufNewR[writePos] = (dstR != nullptr) ? static_cast<double>(dstR[i]) : 0.0;

            auto wrapIdx = [](int idx, int sz) { while (idx < 0) idx += sz; while (idx >= sz) idx -= sz; return idx; };
            const int readOld = wrapIdx(writePos - delayOld, bufferSize);
            const int readNew = wrapIdx(writePos - delayNew, bufferSize);

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
[[nodiscard]] static inline const CoeffSet* getActiveCoeffSet(const AdaptiveCoeffBankSlot& slot) noexcept
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
[[nodiscard]] static inline CoeffSet* getReservedInactiveCoeffSet(AdaptiveCoeffBankSlot& slot) noexcept
{
    // acquire: reserveInactiveCoeffSet CAS success の acquire と HB し、ロック下での確定インデックスを取得。
    int active = consumeAtomic(slot.activeIndex, std::memory_order_acquire);
    return (active == 0) ? &slot.coeffSetB : &slot.coeffSetA;
}

inline bool enqueueDeferredDeleteNonRt(void* ptr, void (*deleter)(void*)) noexcept
{
    if (ptr == nullptr || deleter == nullptr)
        return true;

    const uint64_t epoch = publishRetireEpoch();
    if (enqueueRetireEpochBounded(ptr, deleter, epoch))
        return true;

    m_epochDomain.reclaimRetired();
    if (enqueueRetireEpochBounded(ptr, deleter, epoch))
        return true;

    {
        std::lock_guard<std::mutex> lock(deferredDeleteFallbackMutex);
        deferredDeleteFallbackQueue.push_back(DeferredDeleteFallbackEntry{ ptr, deleter, epoch });
        if (deferredDeleteFallbackQueue.size() >= 1024)
        {
            juce::Logger::writeToLog("[DIAG] deferredDeleteFallbackQueue backlog="
                + juce::String(static_cast<juce::int64>(deferredDeleteFallbackQueue.size())));
        }
    }

    // Fallback queue remains bounded only if we periodically retry enqueue.
    // Kick one best-effort drain here so entries do not accumulate indefinitely
    // when timer cadence is low.
    drainDeferredRetireQueues(false);
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

    // 退役の唯一の入口。
    // ここでは「公開済みハンドルの解放」と「実体の deferred delete 予約」をまとめて行い、
    // active runtime slot / fading runtime slot / publicationLog など複数の非所有スロットからの回収責務を集約する。
    if (!retireDSPHandleForRuntime(dsp))
        return;

    convo::fetchAddAtomic(rtAuxMutable_.runtimeRetireCount,
                         static_cast<std::uint64_t>(1),
                         std::memory_order_acq_rel);
    if (enqueueDeferredDeleteNonRt(dsp, &AudioEngine::destroyDSPCoreNode))
        return;
}

inline convo::isr::DSPHandle registerDSPHandleForRuntime(DSPCore* dsp) noexcept
{
    if (dsp == nullptr)
        return convo::isr::DSPHandle::null();

    std::lock_guard<std::mutex> lock(runtimeDSPHandleMapMutex_);

    auto it = runtimeDSPHandleMap_.find(dsp);
    if (it != runtimeDSPHandleMap_.end())
        return it->second;

    const auto handle = dspHandleRuntime_.create(dsp);
    runtimeDSPHandleMap_.emplace(dsp, handle);
    return handle;
}

inline bool retireDSPHandleForRuntime(DSPCore* dsp) noexcept
{
    if (dsp == nullptr)
        return false;

    std::lock_guard<std::mutex> lock(runtimeDSPHandleMapMutex_);
    const auto it = runtimeDSPHandleMap_.find(dsp);
    if (it == runtimeDSPHandleMap_.end())
        return false;

    const auto handle = it->second;
    runtimeDSPHandleMap_.erase(it);
    if (!handle.isNull())
    {
        dspHandleRuntime_.retire(handle);
        dspHandleRuntime_.reclaim(handle);
    }

    return true;
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

    [[nodiscard]] bool isAcquired() const noexcept { return acquired; }
    [[nodiscard]] bool isCommitted() const noexcept { return committed; }

    CoeffSetWriteLockGuard(const CoeffSetWriteLockGuard&) = delete;
    CoeffSetWriteLockGuard& operator=(const CoeffSetWriteLockGuard&) = delete;

    private:
        AdaptiveCoeffBankSlot& slot;
        bool acquired;
        bool committed;
    };

    // リリースキューに溜まったエントリを解放可能なものから処理する
    void processDeferredReleases();
    static void destroyDSPCoreNode(void* p) noexcept;

    // スナップショット基盤（Phase 2）
    // ==================================================================
    convo::EpochDomain m_epochDomain;
    // DSP_THREAD_STATE: AudioEngine process系で使うaudio-thread専用RCU reader。
    convo::RCUReader audioThreadRcuReader { m_epochDomain };
    // ENGINE_CONTROL: Audio thread での deletion queue overflow 退避スロット。
    std::atomic<DSPCore*> audioThreadRetireOverflowPtr { nullptr };
    RTLocalState rtLocalState_ {};
    RTAuxMutable rtAuxMutable_ {};
    convo::SnapshotCoordinator m_coordinator;
    GenerationManager m_generationManager;

    // ==================================================================
    // Phase 3: コマンドバッファ + ワーカースレッド
    // ==================================================================
    convo::CommandBuffer m_commandBuffer;
    convo::WorkerThread m_workerThread;
    std::mutex rebuildAdmissionIntentMutex_;
    RebuildAdmissionIntentState rebuildAdmissionPendingIntent_ {};

    std::atomic<float> m_currentInputHeadroomDb { -6.0f };
    std::atomic<float> m_currentOutputMakeupDb { 12.0f };
    std::atomic<float> m_currentConvInputTrimDb { 0.0f };
    std::atomic<bool> m_currentEqBypass { false };
    std::atomic<bool> m_currentConvBypass { false };
    std::atomic<convo::ProcessingOrder> m_currentProcessingOrder { convo::ProcessingOrder::ConvolverThenEQ };
    std::atomic<bool> m_currentSoftClipEnabled { true };
    std::atomic<float> m_currentSaturationAmount { 0.1f };

    std::atomic<std::uint64_t> retireQueueDepth_ { 0 };
    std::atomic<std::uint64_t> fallbackQueueDepth_ { 0 };
    std::atomic<std::uint64_t> quarantineResident_ { 0 };
    std::atomic<std::uint64_t> publicationBacklog_ { 0 };
    std::atomic<std::uint64_t> rebuildBacklog_ { 0 };
    std::atomic<std::uint64_t> saturationEnterCount_ { 0 };
    std::atomic<std::uint64_t> saturationExitCount_ { 0 };
    std::atomic<std::uint64_t> publicationRejectCount_ { 0 };
    std::atomic<std::uint64_t> rebuildCollapseCount_ { 0 };
    std::atomic<double> reclaimLatency_ { 0.0 };
    std::atomic<int> retireHighWatermark_ { 3072 };
    std::atomic<int> retireLowWatermark_ { 1024 };
    std::atomic<bool> retireSaturationActive_ { false };
    std::atomic<bool> retireSaturationRecoveryPending_ { false };
    std::atomic<std::uint64_t> retireSaturationRecoveryBaselinePublishCount_ { 0 };
    std::atomic<std::int64_t> emergencyReclaimWindowStartTicks_ { 0 };
    std::atomic<std::int64_t> emergencyReclaimLastBoostTicks_ { 0 };
    std::atomic<int> emergencyReclaimBoostCount_ { 0 };
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
    // Audio Thread -> Message Thread 反映確認用 (RT安全: atomic store/fetch_add のみ)
    // Non-RT diagnostics: most recently rejected rebuild generation in commitNewDSP.
    juce::AudioBuffer<float> m_fadeFloatBuffer;
    juce::AudioBuffer<double> m_fadeDoubleBuffer;

    // ==================================================================
    // ISR Phase 0: Lifecycle Isolation Runtime（P0-A0）
    // ==================================================================

    convo::isr::LifecycleIsolationRuntime lifecycleRuntime_;
    convo::isr::LifecycleBarrierRuntime lifecycleBarrierRuntime_{ lifecycleRuntime_ };

    // ===================== ISR Phase 1-9: Core Runtimes =====================
    convo::isr::RuntimePublicationCoordinator runtimePublicationBridge_;
    convo::isr::DSPQuarantineManager dspQuarantineManager_;
    convo::isr::ClosureGraphWalker closureGraphWalker_;
    convo::isr::DebugRuntime debugRuntime_;
    convo::isr::RetireRuntime retireRuntime_;
    convo::isr::RetireRuntimeEx retireRuntimeEx_;
    convo::isr::ShutdownRuntime shutdownRuntime_;
    convo::isr::BarrierOptimizer barrierOptimizer_;
    convo::isr::EvidenceExporter evidenceExporter_;
    convo::isr::BudgetManager budgetManager_;
    convo::isr::FailureHandler failureHandler_;
    convo::isr::IntrospectionConsole introspectionConsole_;

    // ==================================================================

    JUCE_DECLARE_WEAK_REFERENCEABLE(AudioEngine)
        // ==================================================================
        // ISR Phase 1: RT Execution Frame Separation（P0-A1）
        // ==================================================================
        convo::isr::RTCapabilityFirewall rtCapabilityFirewall_;
        convo::isr::RTAllocatorFirewall rtAllocatorFirewall_;
        convo::isr::RTTraceRelay rtTraceRelay_;

        // Current fade accumulator (RCU-managed, updated during crossfade)
        convo::isr::FadeAccumulator currentFade_{ 0.0, 0.0, false };

        // ==================================================================
        // ISR Phase 2: DSPHandle Runtime（P0-A2）
        // ==================================================================
        convo::isr::DSPHandleRuntime dspHandleRuntime_;
        convo::isr::CrossfadeAuthorityRuntime crossfadeAuthorityRuntime_;
        std::mutex runtimeDSPHandleMapMutex_;
        std::unordered_map<DSPCore*, convo::isr::DSPHandle> runtimeDSPHandleMap_;
        std::atomic<convo::isr::CrossfadeId> activeCrossfadeId_{ 0u };

        // ==================================================================

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

[[nodiscard]] inline bool AudioEngine::dequeueLearningCommand(LearningCommand& cmd) noexcept
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

[[nodiscard]] inline bool AudioEngine::dequeueLearnerDispatch(LearnerDispatchAction& action) noexcept
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


