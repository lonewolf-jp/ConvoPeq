
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
    uint64_t sessionId = 0;
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
#include "NoiseShaperLearner.h"
#include "RefCountedDeferred.h"
#include "GenerationManager.h"
#include "core/Types.h"
#include "core/SnapshotCoordinator.h"
#include "core/ReaderEpoch.h"
#include "core/CommandBuffer.h"
#include "core/ThreadAffinityManager.h"
#include "core/WorkerThread.h"
#include "core/RebuildTypes.h"

// デバッグビルド時のみログを出力するマクロ
#if defined(JUCE_DEBUG) && !defined(NDEBUG)
    #define DBG_LOG(msg) juce::Logger::writeToLog(msg)
#else
    #define DBG_LOG(msg) ((void)0)
#endif

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

    enum class AnalyzerSource
    {
        Input,
        Output
    };

    using OversamplingType = convo::OversamplingType;
    using NoiseShaperType = convo::NoiseShaperType;

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

    void setIRFadeSamples(int samples) noexcept { m_irFadeSamples.store(samples, std::memory_order_relaxed); }
    void setEQFadeSamples(int samples) noexcept { m_eqFadeSamples.store(samples, std::memory_order_relaxed); }
    int getIRFadeSamples() const noexcept { return m_irFadeSamples.load(std::memory_order_relaxed); }
    int getEQFadeSamples() const noexcept { return m_eqFadeSamples.load(std::memory_order_relaxed); }
    bool isFading() const noexcept { return m_coordinator.isFading(); }
    const convo::SnapshotCoordinator& getSnapshotCoordinator() const noexcept { return m_coordinator; }
    void setIRChangeFlag() noexcept { m_pendingIRChange.store(true, std::memory_order_release); }

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
    NoiseShaperLearner::LearningMode getNoiseShaperLearningMode() const { return pendingLearningMode.load(std::memory_order_acquire); }
    bool isNoiseShaperLearning() const;
    const NoiseShaperLearner::Progress& getNoiseShaperLearningProgress() const;
    int copyNoiseShaperLearningHistory(double* outScores, int maxPoints) const noexcept;
    // 学習ワーカーが記録したエラーメッセージを返す（UI 表示用）。エラーなしは nullptr。
    const char* getNoiseShaperLearningError() const noexcept;
    static int getAdaptiveSampleRateBankCount() noexcept;

    // --- NoiseShaperLearner Settings ---
    NoiseShaperLearner::Settings getNoiseShaperLearnerSettings() const;
    void setNoiseShaperLearnerSettings(const NoiseShaperLearner::Settings& settings);

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
                        entry.second->addRef();

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

    //----------------------------------------------------------
     // DSPコア (Audio Threadで実行される処理のコンテナ)
    //----------------------------------------------------------
    struct DSPCore : public RefCountedDeferred<DSPCore>
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
        // 【最適化】出力 / 入力 DC 除去を UltraHighRateDCBlocker (1次IIR, ブロックモード) に統一。
        // 旧 DCBlocker (4次 Butterworth, サンプル単位) は 1 サンプルあたり ~20 演算を要したが、
        // 1次 IIR は ~4 演算で済みかつ process(data, N) ブロック呼び出しによりメモリアクセスも効率化。
        // DC 除去の目的 (3Hz 以下のカット) には 1 次で十分。
        convo::UltraHighRateDCBlocker dcBlockerL, dcBlockerR;
        convo::UltraHighRateDCBlocker inputDCBlockerL, inputDCBlockerR;
        convo::UltraHighRateDCBlocker osDCBlockerL, osDCBlockerR; // Oversampling後のDC除去用
        ::convo::PsychoacousticDither dither;
        ::convo::FixedNoiseShaper fixedNoiseShaper;
        ::convo::Fixed15TapNoiseShaper fixed15TapNoiseShaper;
        LatticeNoiseShaper adaptiveNoiseShaper;
        // 出力周波数フィルター (① ハイカット/ローカット / ② ローパス/ハイパス)
        convo::OutputFilter outputFilter;

        CustomInputOversampler oversampling;
        size_t oversamplingFactor = 1;
        int ditherBitDepth = 0; // DSPCore内でディザリング判定に使用
        NoiseShaperType noiseShaperType = NoiseShaperType::Psychoacoustic;
        uint32_t activeAdaptiveCoeffGeneration = 0;
        int activeAdaptiveCoeffBankIndex = -1;
        uint64_t currentCaptureSessionId = 0;
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
        AudioEngine* ownerEngine = nullptr;

        // B2: processDouble 用のバイパスフェード状態
        convo::ScopedAlignedPtr<double> dryBypassBufferDoubleL;
        convo::ScopedAlignedPtr<double> dryBypassBufferDoubleR;
        int dryBypassCapacityDouble = 0;
        convo::LinearRamp bypassFadeGainDouble;
        bool bypassedDouble = false;

        // パススルーDSPの固定レイテンシ付与用ディレイライン
        convo::ScopedAlignedPtr<double> fixedLatencyBufferL;
        convo::ScopedAlignedPtr<double> fixedLatencyBufferR;
        int fixedLatencyBufferSize = 0;
        int fixedLatencyWritePos = 0;
        int fixedLatencySamples = 0;

        // インターサンプルピーク近似: 前ブロック末尾のクリップ済み出力 (L/R)
        // softClipBlockAVX2() へブロック間状態を渡すために保持する。
        double softClipPrevSample[2] = {0.0, 0.0};

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

    std::atomic<bool> dspCrossfadePending { false };
    std::atomic<bool> firstIrDryCrossfadePending { false }; // 初回IRロード時に dry を旧信号として使用
    std::atomic<bool> firstIrDryCrossfadeDone { false };    // アプリ起動後の初回1回のみ有効
    std::atomic<bool> dspCrossfadeUseDryAsOld { false };    // Audio Thread 実行中フラグ
    convo::LinearRamp dspCrossfadeGain;
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
    std::atomic<uint32_t> pendingRebuildMask_{ 0 };
    std::atomic<int64_t> lastIRContentRebuildTicks_{ 0 };
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
        NoiseShaperLearner::LearningMode mode = NoiseShaperLearner::LearningMode::Short;
        uint64_t irGeneration = 0;
    };

    struct LearnerDispatchAction
    {
        enum class Type : uint8_t { Start, Stop };

        Type type = Type::Stop;
        bool resume = false;
        NoiseShaperLearner::LearningMode mode = NoiseShaperLearner::LearningMode::Short;
    };

    enum class LearningRuntimeState : uint8_t { Idle, WaitingForDSP, Running };

    static constexpr uint32_t learningCommandBufferSize = 128;
    static constexpr uint32_t learningCommandBufferMask = learningCommandBufferSize - 1;
    static constexpr uint32_t learnerDispatchBufferSize = 32;
    static constexpr uint32_t learnerDispatchBufferMask = learnerDispatchBufferSize - 1;

    #pragma warning(push)
    #pragma warning(disable: 4324)
    alignas(64) std::atomic<NoiseShaperLearner::LearningMode> pendingLearningMode { NoiseShaperLearner::LearningMode::Short };
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
    NoiseShaperLearner::LearningMode requestedLearningMode = NoiseShaperLearner::LearningMode::Short;
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
    void processRebuildRequestsInternal();
    void processRebuildRequestsFallback();

    static void onSnapshotRequired(void* userData, uint64_t generation);
    void createSnapshotFromCurrentState(uint64_t generation);
    void initWorkerThread();
    void shutdownWorkerThread();

    // Worker thread for rebuilds
    void rebuildThreadLoop();
    void stopRebuildThread();
    std::thread rebuildThread;
    std::mutex rebuildMutex;
    std::condition_variable rebuildCV;
    std::atomic<bool> rebuildThreadShouldExit { false };
    std::atomic<bool> rebuildThreadIsRunning { false };
    std::atomic<bool> shutdownInProgress { false };
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
        std::atomic<int> activeIndex { 0 };   // 0 = A, 1 = B
        std::atomic<uint32_t> generation { 1u };
        NoiseShaperLearner::State state {};
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

    class ConvolverStateReaderGuard
    {
    public:
        explicit ConvolverStateReaderGuard(ConvolverProcessor& conv) noexcept;
        ~ConvolverStateReaderGuard() noexcept;

        ConvolverStateReaderGuard(const ConvolverStateReaderGuard&) = delete;
        ConvolverStateReaderGuard& operator=(const ConvolverStateReaderGuard&) = delete;

    private:
        ConvolverProcessor& m_convolver;
    };

    void debugAssertNotAudioThread() const;

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

    // ==================================================================
    // RCU 基盤（段階 2+3：参照追跡＋Grace Period による安全なリリース遅延）
    // ==================================================================

public:
    void enterReader() noexcept;
    void exitReader() noexcept;
    uint64_t getMinReaderEpoch() const noexcept;
    uint64_t advanceGlobalEpoch() noexcept;

private:
    // Audio Thread 用：スレッドローカルなスロット番号を取得するヘルパー
    size_t getOrAllocateSlot() noexcept;

    // リリースキューに溜まったエントリを解放可能なものから処理する
    void processDeferredReleases();

    // ==================================================================
    // RCU 基盤
    // ==================================================================

    // マルチリーダー epoch 追跡配列
    static constexpr size_t MAX_READERS = 8;
    #pragma warning(push)
    #pragma warning(disable:4324)
    alignas(64) std::array<std::atomic<uint64_t>, MAX_READERS> readerEpochs{};
    #pragma warning(pop)
    std::atomic<size_t> nextReaderSlot{0};
    static thread_local size_t tls_readerSlot;

    // スレッド登録 API（Message/Timer スレッド用）
    size_t registerReader();
    void unregisterReader(size_t slot);
    void updateReaderEpoch(size_t slot, uint64_t epoch);

    // epoch カウンタ
    std::atomic<uint64_t> globalEpoch{1};
    std::atomic<uint64_t> audioThreadEpoch{0};
    uint64_t lastReclaimedEpoch{0};


    // epoch 比較ヘルパー（ラップアラウンド対応）
    static bool isOlder(uint64_t a, uint64_t b) noexcept
    {
        return (a - b) > (1ULL << 63);
    }

    // ==================================================================
    // スナップショット基盤（Phase 2）
    // ==================================================================
    convo::SnapshotCoordinator m_coordinator;
    GenerationManager m_generationManager;

    // ==================================================================
    // Phase 3: コマンドバッファ + ワーカースレッド
    // ==================================================================
    convo::CommandBuffer m_commandBuffer;
    convo::WorkerThread m_workerThread;

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
