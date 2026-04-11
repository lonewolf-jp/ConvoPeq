//============================================================================
#pragma once
// ConvolverProcessor.h  ── v0.3 (JUCE 8.0.12対応 / MKL NUC 統合)
//
// FFTベースコンボリューションプロセッサー
//
// ■ 信号処理フロー:
//   Input → [Convolver <-> EQ] → Output
//
// ■ 用途:
//   - ルームリバーブ（ホール、スタジオ等の空間シミュレーション）
//   - スピーカーキャビネット/マイク特性の畳み込み
//   - 位相補正/周波数特性補正
//
// ■ スレッド安全設計:
//   - loadImpulseResponse(): Message Thread で実行。バックグラウンドスレッドで読み込みを行い、完了後に atomic に差し替えます (RCU)。ロード中も音切れなく古いIRで処理を継続します。
//   - process(): Audio Thread で実行。MKLNonUniformConvolver を使用してパーティション分割畳み込みを行います。
//   - パラメータ: std::atomic でスレッドセーフ。Audio Thread内でのメモリ確保やIR再ロードは行いません。
//
// ■ エンジン:
//   Intel MKL Non-Uniform Partitioned Convolution (NUC) エンジンを使用します。
//============================================================================

#include <JuceHeader.h>
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>
#include <array>
#include <functional>
#include <deque>
#include <map>
#include <cmath>
#include "AlignedAllocation.h"
#include "MKLNonUniformConvolver.h"
#include "AllpassDesigner.h"

// ── Phase 0: Epoch-based RCU 基盤ヘッダー ──
#include "GenerationManager.h"
#include "ConvolverState.h"
#include "SafeStateSwapper.h"
#include "DeferredFreeThread.h"
#include "PreparedIRState.h"
#include "DeferredDeletionQueue.h"
#include "ConvolverRuntime.h"
#include "core/ReaderEpoch.h"

class AudioEngine;
class IRConverter;
class CacheManager;
class ProgressiveUpgradeThread;

class ConvolverProcessor : public juce::ChangeBroadcaster,
                           private juce::Timer
{
public:
    struct IRLoadPreview
    {
        bool success = false;
        juce::String errorMessage;
        float autoDetectedLengthSec = 1.0f;
        int autoDetectedLengthSamples = 0;
        float recommendedMaxSec = 3.0f;
        float hardMaxSec = 0.0f;
        bool exceedsRecommended = false;
        bool exceedsHardLimit = false;
    };

    enum class PhaseMode : int
    {
        AsIs = 0,
        Mixed = 1,
        Minimum = 2
    };

    // リサンプリング時の位相モード（r8brain CDSPResampler24IR 用）
    // Linear : 科学的に正確な線形位相（測定・ミキシング用途）
    // Minimum: アタック保持・自然な聴感（音楽用途）
    enum class ResamplingPhaseMode {
        Linear,   // デフォルト: 位相的に正確
        Minimum   // Minimum phase: トランジェント保持
    };

    void setResamplingPhaseMode(ResamplingPhaseMode mode);
    ResamplingPhaseMode getResamplingPhaseMode() const;

    class Listener
    {
    public:
        virtual ~Listener() = default;
        virtual void convolverParamsChanged(ConvolverProcessor* processor) = 0;
    };

    void addListener(Listener* listener) { listeners.add(listener); }
    void removeListener(Listener* listener) { listeners.remove(listener); }

    // 波形表示の解像度
    static constexpr int WAVEFORM_POINTS = 512;

    // IR処理定数
    static constexpr int MIN_PARTITION_SIZE = 256;
    static constexpr int PARTITION_SIZE_MULTIPLIER = 2;
    static constexpr float IR_SILENCE_THRESHOLD = 1.0e-6f;
    static constexpr float MIX_MIN = 0.0f;
    static constexpr float MIX_MAX = 1.0f;
    static constexpr float SMOOTHING_TIME_MIN_SEC = 0.01f;   // 10ms
    static constexpr float SMOOTHING_TIME_MAX_SEC = 0.5f;    // 500ms
    static constexpr float SMOOTHING_TIME_DEFAULT_SEC = 0.1f; // 100ms
    static constexpr float IR_LENGTH_MIN_SEC = 0.5f;
    static constexpr float IR_LENGTH_MAX_SEC = 3.0f;
    static constexpr float IR_LENGTH_DEFAULT_SEC = 1.0f;
    static constexpr float MIXED_F1_MIN_HZ = 100.0f;
    static constexpr float MIXED_F1_MAX_HZ = 400.0f;
    static constexpr float MIXED_F1_DEFAULT_HZ = 200.0f;
    static constexpr float MIXED_F2_MIN_HZ = 700.0f;
    static constexpr float MIXED_F2_MAX_HZ = 1300.0f;
    static constexpr float MIXED_F2_DEFAULT_HZ = 1000.0f;
    static constexpr float MIXED_TAU_MIN = 4.0f;
    static constexpr float MIXED_TAU_MAX = 256.0f;
    static constexpr float MIXED_TAU_DEFAULT = 32.0f;
    static constexpr float TAIL_ROLLOFF_START_MIN_HZ = 20.0f;
    static constexpr float TAIL_ROLLOFF_START_MAX_HZ = 20000.0f;
    static constexpr float TAIL_AIR_ROLLOFF_START_DEFAULT_HZ = 3500.0f;
    static constexpr float TAIL_LAYER_ROLLOFF_START_DEFAULT_HZ = 2000.0f;
    static constexpr float TAIL_ROLLOFF_START_DEFAULT_HZ = TAIL_AIR_ROLLOFF_START_DEFAULT_HZ;
    static constexpr float TAIL_ROLLOFF_STRENGTH_MIN = 0.0f;
    static constexpr float TAIL_ROLLOFF_STRENGTH_MAX = 2.0f;
    static constexpr float TAIL_AIR_ROLLOFF_STRENGTH_DEFAULT = 0.3f;
    static constexpr float TAIL_LAYER_ROLLOFF_STRENGTH_DEFAULT = 0.5f;
    static constexpr float TAIL_ROLLOFF_STRENGTH_DEFAULT = TAIL_AIR_ROLLOFF_STRENGTH_DEFAULT;
    static constexpr float TAIL_PARTITION_STRENGTH_MIN = 0.0f;
    static constexpr float TAIL_PARTITION_STRENGTH_MAX = 2.0f;
    static constexpr float TAIL_PARTITION_STRENGTH_DEFAULT = 1.0f;
    static constexpr int REBUILD_DEBOUNCE_MIN_MS = 50;
    static constexpr int REBUILD_DEBOUNCE_MAX_MS = 3000;
    static constexpr int REBUILD_DEBOUNCE_DEFAULT_MS = 400;

    // DelayLine用定数 (Audio Threadでのメモリ確保防止)
    // IRの最大長(kMaxIRCap)と最大ブロックサイズをカバーする値を設定
    // 3s @ 192kHz = 576000 samples. Next power of 2 is 1048576.
    static constexpr int MAX_IR_LATENCY = 2097152; // 2^21 (3.0s @ 384kHz = ~1.15M samples をカバー)
    // 最適なFFTパフォーマンスのために、この値は2の累乗である必要があります。
    // また、MIN_PARTITION_SIZE以上である必要があります。
    // MAX_PARTITION_SIZEもまた、maxBlockSize * oversamplingFactor以上である必要があります。
    static constexpr int MAX_BLOCK_SIZE = 524288;  // 65536 * 8 (Safe for 8x oversampling of max input block)
    static constexpr int MAX_TOTAL_DELAY = MAX_IR_LATENCY + MAX_BLOCK_SIZE;
    // リングバッファのラップアラウンドをビットマスクで高速化するため、2の累乗サイズを使用
    static constexpr int DELAY_BUFFER_SIZE = 4194304; // 2^22 (approx 4M samples > MAX_TOTAL_DELAY)
    static constexpr int DELAY_BUFFER_MASK = DELAY_BUFFER_SIZE - 1;
    static_assert((DELAY_BUFFER_SIZE & (DELAY_BUFFER_SIZE - 1)) == 0,
                  "DELAY_BUFFER_SIZE must be a power of 2 for bitmask optimization");
    static constexpr double CONVOLUTION_HEADROOM_GAIN = 1.0; // 0.0 dB (Unity Gain - Headroom is baked into IR)

    ConvolverProcessor();
    ~ConvolverProcessor();

    void setRcuProvider(AudioEngine* engine) noexcept { rcuProvider = engine; }

    // RCU リーダー (Audio Thread のみ)
    void enterReader();
    void exitReader();

    //----------------------------------------------------------
    // 準備（Audio Thread開始前）
    //----------------------------------------------------------
    void prepareToPlay(double sampleRate, int samplesPerBlock);
    void releaseResources();

    //----------------------------------------------------------
    // インパルス応答読み込み（Message Thread）
    //
    // 対応形式: WAV, AIFF, FLAC
    // @return true=読み込み開始成功（非同期）, false=開始失敗
    //----------------------------------------------------------
    bool loadImpulseResponse(const juce::File& irFile, bool optimizeForRealTime = false);
    void loadIR(const juce::File& irFile);
    void applyPreparedIRState(std::unique_ptr<PreparedIRState> prepared);
    void stopUpgradeThread();
    void startProgressiveUpgrade(const juce::File& file,
                                 double sampleRate,
                                 int currentFFTSize,
                                 uint64_t generation,
                                 uint64_t baseKey);

    void setTargetUpgradeFFTSize(int fftSize);
    int getTargetUpgradeFFTSize() const;
    void setEnableProgressiveUpgrade(bool enable);
    bool isProgressiveUpgradeEnabled() const;
    void setMaxCacheEntries(size_t maxEntries);
    size_t getMaxCacheEntries() const;
    void clearCache();
    bool isCacheEntrySafeToDelete(uint64_t cacheKey, int fftSize) const;

    // メイン処理（Audio Thread）
    //
    //----------------------------------------------------------
    void process(juce::dsp::AudioBlock<double>& block);

    //----------------------------------------------------------
    // バイパス制御
    //----------------------------------------------------------
    void setBypass(bool shouldBypass);
    bool isBypassed() const { return bypassed.load(); }
    const ConvolverState* getConvolverState() const { return rcuSwapper.getState(); }

    //----------------------------------------------------------
    // Dry/Wet Mix (0.0 = Dry only, 1.0 = Wet only)
    //----------------------------------------------------------
    void setMix(float mixAmount);
    float getMix() const;

    //----------------------------------------------------------
    // IR Phase Mode
    //----------------------------------------------------------
    void setPhaseMode(PhaseMode mode);
    PhaseMode getPhaseMode() const { return static_cast<PhaseMode>(phaseMode.load(std::memory_order_acquire)); }

    // 後方互換API
    void setUseMinPhase(bool useMinPhase);
    bool getUseMinPhase() const { return getPhaseMode() == PhaseMode::Minimum; }

    //------------------------------------------------------------------
    // NUC 出力周波数フィルターモード設定
    //
    // フィルターは SetImpulse() 内で irFreqDomain に焼き込まれる。
    // モード変更時は内部で rebuildAllIRs() を呼んで NUC を再構築する。
    // Message Thread からのみ呼ぶこと。
    //------------------------------------------------------------------
    void setNUCFilterModes(convo::HCMode hcMode, convo::LCMode lcMode);

    //------------------------------------------------------------------
    // NUC テール処理パラメータ設定
    // SetImpulse() 時の irFreqDomain 焼き込みに使用する。
    // 変更時は Debounced rebuild を要求する。
    //------------------------------------------------------------------
    void setTailProcessingMode(int mode);
    int getTailProcessingMode() const noexcept { return tailProcessingMode.load(std::memory_order_acquire); }
    void setTailRolloffStartHz(float hz);
    float getTailRolloffStartHz() const noexcept { return tailRolloffStartHz.load(std::memory_order_acquire); }
    void setTailRolloffStrength(float strength);
    float getTailRolloffStrength() const noexcept { return tailRolloffStrength.load(std::memory_order_acquire); }
    void setPartitionTailStrength(float strength);
    float getPartitionTailStrength() const noexcept { return partitionTailStrength.load(std::memory_order_acquire); }

    //----------------------------------------------------------
    // Experimental Direct Head Flag
    // 段階導入用の機能フラグ。変更時はIRを再構築する。
    //----------------------------------------------------------
    void setExperimentalDirectHeadEnabled(bool enabled);
    bool getExperimentalDirectHeadEnabled() const { return experimentalDirectHeadEnabled.load(std::memory_order_acquire); }

    //----------------------------------------------------------
    // Smoothing Time
    //----------------------------------------------------------
    void setSmoothingTime(float timeSec);
    float getSmoothingTime() const;

    //----------------------------------------------------------
    // Mixed Phase Parameters (f1/f2/tau)
    //----------------------------------------------------------
    void setMixedTransitionStartHz(float hz);
    float getMixedTransitionStartHz() const;
    void setMixedTransitionEndHz(float hz);
    float getMixedTransitionEndHz() const;
    void setMixedPreRingTau(float tau);
    float getMixedPreRingTau() const;

    //----------------------------------------------------------
    // Rebuild Debounce Time (Message/Worker burst control)
    //----------------------------------------------------------
    void setRebuildDebounceMs(int ms);
    int getRebuildDebounceMs() const;

    //----------------------------------------------------------
    // IR Length
    //----------------------------------------------------------
    void setTargetIRLength(float timeSec);
    float getTargetIRLength() const;
    void applyAutoDetectedIRLength(float timeSec);
    void setIRLengthManualOverride(bool isManual);
    bool hasManualIRLengthOverride() const { return irLengthManualOverride.load(std::memory_order_acquire); }
    float getAutoDetectedIRLength() const { return autoDetectedIRLengthSec.load(std::memory_order_acquire); }
    static float getMaximumAllowedIRLengthSecForSampleRate(double sampleRate);
    float getMaximumAllowedIRLengthSec(double sampleRate = 0.0) const;
    static IRLoadPreview analyzeImpulseResponseFile(const juce::File& irFile, double processingSampleRate);

    //----------------------------------------------------------
    // Thread Affinity & Optimization
    //----------------------------------------------------------
    void setThreadAffinityCallback(std::function<void(void*)> callback) { onSetThreadAffinity = callback; }

    //----------------------------------------------------------
    // 状態リセット
    //----------------------------------------------------------
    void reset();

    //----------------------------------------------------------
    // 状態取得
    //----------------------------------------------------------
    bool isIRLoaded() const { return rcuSwapper.getState() != nullptr; }
    bool isLoadingIR() const { return isLoading.load(std::memory_order_acquire); }
    juce::String getIRName() const { return irName; }
    int getIRLength() const { return irLength.load(std::memory_order_acquire); }
    juce::String getLastError() const { return lastError; }
    float getLoadProgress() const { return loadProgress.load(); }
    int getMixedPhaseState() const noexcept { return mixedPhaseState.load(std::memory_order_acquire); }
    void setMixedPhaseState(int state) noexcept { mixedPhaseState.store(juce::jlimit(0, 2, state), std::memory_order_release); }
    void setLoadingProgress(float p);
    int getCurrentBufferSize() const { return currentBufferSize.load(std::memory_order_acquire); }
    struct LatencyBreakdown
    {
        int algorithmLatencySamples = 0;
        int irPeakLatencySamples = 0;
        int totalLatencySamples = 0;
        bool directHeadActive = false;
    };

    struct LatencySnapshot
    {
        int32_t algorithmLatencySamples = 0;
        int32_t irPeakLatencySamples = 0;
        int32_t totalLatencySamples = 0;
        bool hasParallelDryPath = false;
    };

    LatencyBreakdown getLatencyBreakdown() const;
    int getLatencySamples() const;
    int getTotalLatencySamples() const;

    //----------------------------------------------------------
    // 波形表示用データ取得
    //----------------------------------------------------------
    std::vector<float> getIRWaveform() const;

    //----------------------------------------------------------
    // 周波数特性表示用データ取得
    //----------------------------------------------------------
    std::vector<float> getIRMagnitudeSpectrum() const;
    double getIRSpectrumSampleRate() const;

    //----------------------------------------------------------
    // State Management
    //----------------------------------------------------------
    juce::ValueTree getState() const;
    void setState (const juce::ValueTree& state);

    //----------------------------------------------------------
    // リビルド (サンプルレート変更時など)
    //----------------------------------------------------------
    void rebuildAllIRs();
    void rebuildAllIRsSynchronous(std::function<bool()> shouldCancel = nullptr);
    void invalidatePendingLoads();

    // 他のインスタンスから状態を同期 (AudioEngine用)
    void syncStateFrom(const ConvolverProcessor& other);
    void syncParametersFrom(const ConvolverProcessor& other);
    void shareConvolutionEngineFrom(const ConvolverProcessor& other);
    void refreshLatency();

    // 【NUCエンジン構築専用エントリポイント】
    // LoaderThreadからIRデータを受け取り、メッセージスレッド上で
    // SetImpulse / DftiCommitDescriptor / mkl_malloc を安全に実行する
    void finalizeNUCEngineOnMessageThread(convo::ScopedAlignedPtr<double> irL, // This is called from LoaderThread
                                          convo::ScopedAlignedPtr<double> irR,
                                          int length,
                                          double sr,
                                          int peakDelay,
                                          int maxFFTSize,
                                          int knownBlockSize,
                                          int firstPartition,
                                          int preferredCallSize,
                                          bool isRebuild,
                                          const juce::File& irFile,
                                          double scaleFactor, // This is for newConv->init
                                          std::shared_ptr<juce::AudioBuffer<double>> loadedIR,
                                          std::shared_ptr<juce::AudioBuffer<double>> displayIR);

    // 可視化データ生成の制御 (DSP用インスタンスでは無効化してメモリを節約)
    void setVisualizationEnabled(bool enabled) { visualizationEnabled = enabled; }
    bool isVisualizationEnabled() const { return visualizationEnabled; }

    // ガベージコレクション (Message Threadから定期的に呼ぶ)
    void cleanup();
    void forceCleanup();

    // ── Phase 0: Epoch-based RCU 状態更新 ──
    // Message Thread から呼ぶ。新しい ConvolverState を atomic にスワップし、
    // 旧状態を DeferredFreeThread に委ねる。
    // GenerationManager で陳腐化チェックを行い、古いタスク結果は破棄する。
    //
    // @param newState  新しい状態（所有権を移譲）。世代チェックに失敗した場合は即削除。
    void updateConvolverState(ConvolverState* newState);
    void updateConvolverState(std::unique_ptr<ConvolverState> newState);

    bool isConvolverGenerationCurrent(uint64_t generation) const
    {
        return convolverStateGeneration.isCurrentGeneration(generation);
    }

private:
    void timerCallback() override;
    void postCoalescedChangeNotification();
    void updateLatencyCache() noexcept;
    void requestHostDisplayUpdate();
    void debugCheckAtomicLockFree() const;
    void requestDebouncedRebuild();
    // リングバッファオーバーフロー通知コールバック (Audio Thread 安全; 生関数ポインタ)
    // MKLNonUniformConvolver::ringWrite() からオーバーフロー時に呼び出される。
    static void overflowCallbackThunk(void* userData) noexcept;

    struct StereoConvolver;
    class LoaderThread;

    void commitNewConvolver(StereoConvolver* newConv,
                            std::shared_ptr<juce::AudioBuffer<double>> loadedIR,
                            double loadedSR, int targetLength, bool isRebuild,
                            const juce::File& file, double scaleFactor,
                            std::shared_ptr<juce::AudioBuffer<double>> displayIR);

    void switchEngineOnMessageThread(StereoConvolver* newEngine) noexcept;

    void applyNewState(StereoConvolver* newConv, std::shared_ptr<juce::AudioBuffer<double>> loadedIR, double loadedSR, int targetLength, bool isRebuild, const juce::File& file, double scaleFactor, std::shared_ptr<juce::AudioBuffer<double>> displayIR);
    void handleLoadError(const juce::String& error);
    void createWaveformSnapshot (const juce::AudioBuffer<double>& irBuffer);
    void createFrequencyResponseSnapshot (const juce::AudioBuffer<double>& irBuffer, double sampleRate);
    int computeTargetIRLength(double sampleRate, int originalLength) const;

    // Stereo processing wrapper
    struct StereoConvolver
    {
        double* irData[2] = { nullptr, nullptr };

        std::array<convo::MKLNonUniformConvolver*, 2> nucConvolvers { nullptr, nullptr };
        int irDataLength = 0;

        int latency = 0;
        int irLatency = 0; // IR由来の遅延 (ピーク位置)
        int callQuantumSamples = 0;    // Audio Thread でのNUC呼び出し量子
        int prewarmedMaxSamples = 0;   // Message Thread でプリウォーム済みの最大呼び出し長

        // Clone用に初期化パラメータを保存
        double storedSampleRate = 0.0;
        int storedMaxFFTSize = 0;
        int storedKnownBlockSize = 0;
        int storedFirstPartition = 0;
        double storedScale = 1.0;
        bool storedDirectHeadEnabled = false;

        StereoConvolver() = default;

        static void destroyNUCConvolver(convo::MKLNonUniformConvolver*& ptr) noexcept
        {
            if (ptr != nullptr)
            {
                ptr->~MKLNonUniformConvolver();
                convo::aligned_free(ptr);
                ptr = nullptr;
            }
        }

        ~StereoConvolver() {
            destroyNUCConvolver(nucConvolvers[0]);
            destroyNUCConvolver(nucConvolvers[1]);
            if (irData[0]) { convo::aligned_free(irData[0]); irData[0] = nullptr; }
            if (irData[1]) { convo::aligned_free(irData[1]); irData[1] = nullptr; }
        }

        // RCU ベースのライフタイム管理 (参照カウントは使用しない)
        // 解放は retireStereoConvolver() を使用すること

        // コピーコンストラクタは禁止 (NUCエンジンは複製コストが高く、状態を持つため)
        StereoConvolver(const StereoConvolver& other) = delete;

        // 代入演算子は禁止 (使用しないため)
        StereoConvolver& operator=(const StereoConvolver&) = delete;

        bool init(double* irL, double* irR, int length, double sr, int peakDelay, int maxFFTSize, int knownBlockSize, int firstPartition, int preferredCallSize, double scale = 1.0,
              bool enableDirectHead = false,
              const convo::FilterSpec* filterSpec = nullptr,
              ConvolverProcessor* ownerProcessor = nullptr)
        {
            // Safety: Free existing data if init is called multiple times (Leak prevention)
            if (irData[0]) { convo::aligned_free(irData[0]); irData[0] = nullptr; }
            if (irData[1]) { convo::aligned_free(irData[1]); irData[1] = nullptr; }

            // Ownership transfer
            irData[0] = irL;
            irData[1] = irR;
            irDataLength = length;
            this->irLatency = peakDelay;
            callQuantumSamples = juce::jmax(1, preferredCallSize);
            prewarmedMaxSamples = callQuantumSamples;
            storedSampleRate = sr;
            storedMaxFFTSize = maxFFTSize;
            storedKnownBlockSize = knownBlockSize;
            storedFirstPartition = firstPartition;
            storedScale = scale;
            storedDirectHeadEnabled = enableDirectHead;

            try
            {
                // ── MKL Non-Uniform Partitioned Convolution (NUC) ──
                // new完全禁止 → aligned_malloc + placement new (規約準拠)
                void* rn0 = convo::aligned_malloc(sizeof(convo::MKLNonUniformConvolver), 64);
                new (rn0) convo::MKLNonUniformConvolver();
                destroyNUCConvolver(nucConvolvers[0]);
                nucConvolvers[0] = static_cast<convo::MKLNonUniformConvolver*>(rn0);

                void* rn1 = convo::aligned_malloc(sizeof(convo::MKLNonUniformConvolver), 64);
                new (rn1) convo::MKLNonUniformConvolver();
                destroyNUCConvolver(nucConvolvers[1]);
                nucConvolvers[1] = static_cast<convo::MKLNonUniformConvolver*>(rn1);

                if (nucConvolvers[0]->SetImpulse(irData[0], irDataLength, knownBlockSize, scale, enableDirectHead, filterSpec) &&
                    nucConvolvers[1]->SetImpulse(irData[1], irDataLength, knownBlockSize, scale, enableDirectHead, filterSpec))
                {
                    latency   = nucConvolvers[0]->getLatency();
                    DBG("Convolver: NUC Engine Active. Latency: " << latency << " samples");
                    if (ownerProcessor != nullptr)
                    {
                        nucConvolvers[0]->setOverflowCallback(ConvolverProcessor::overflowCallbackThunk, ownerProcessor);
                        nucConvolvers[1]->setOverflowCallback(ConvolverProcessor::overflowCallbackThunk, ownerProcessor);
                    }
                    return true;
                }
            }
            catch (const std::bad_alloc&)
            {
                // Fall through to cleanup on memory allocation failure
            }

            // NUC セットアップ失敗 or メモリ確保失敗
            destroyNUCConvolver(nucConvolvers[0]);
            destroyNUCConvolver(nucConvolvers[1]);
            return false;
        }

        // Deep Copyを作成する。
        // 失敗時 (MKLメモリ確保失敗等) は nullptr を返す。呼び出し元で必ずチェックすること。
        StereoConvolver* clone() const
        {
            StereoConvolver* newConv = nullptr;
            try
            {
                void* mem = convo::aligned_malloc(sizeof(StereoConvolver), 64);
                new (mem) StereoConvolver();
                newConv = static_cast<StereoConvolver*>(mem);

                if (irDataLength > 0 && irData[0] && irData[1])
                {
                    convo::ScopedAlignedPtr<double> l(static_cast<double*>(convo::aligned_malloc(irDataLength * sizeof(double), 64)));
                    convo::ScopedAlignedPtr<double> r(static_cast<double*>(convo::aligned_malloc(irDataLength * sizeof(double), 64)));

                    std::memcpy(l.get(), irData[0], irDataLength * sizeof(double));
                    std::memcpy(r.get(), irData[1], irDataLength * sizeof(double));

                    if (!newConv->init(l.release(), r.release(), irDataLength, storedSampleRate, irLatency, storedMaxFFTSize, storedKnownBlockSize, storedFirstPartition, callQuantumSamples, storedScale, storedDirectHeadEnabled))
                    {
                        newConv->~StereoConvolver();
                        convo::aligned_free(newConv);
                        return nullptr;
                    }
                }
                return newConv;
            }
            catch (const std::bad_alloc&)
            {
                if (newConv != nullptr)
                {
                    newConv->~StereoConvolver();
                    convo::aligned_free(newConv);
                }
                return nullptr;
            }
        }

        bool areNUCDescriptorsCommitted() const noexcept
        {
            for (const auto& conv : nucConvolvers)
            {
                if (!conv || !conv->areFftDescriptorsCommitted())
                    return false;
            }
            return true;
        }

        void reset();
        void process(int channel, const double* in, double* out, int numSamples);
    };

    std::atomic<StereoConvolver*> m_activeEngine { nullptr }; // Raw pointer for Audio Thread (Lock-free)
    std::atomic<bool> isLoading { false };
    std::atomic<bool> isRebuilding { false };
    std::unique_ptr<LoaderThread> activeLoader;
    std::deque<std::unique_ptr<LoaderThread>> loaderTrashBin;
    std::atomic<float> loadProgress { 0.0f };
    std::atomic<int> mixedPhaseState { 0 }; // 0=WaitingIR, 1=Optimizing, 2=Completed
    juce::String lastError;

    juce::dsp::ProcessSpec currentSpec = { 48000.0, 512, 2 };

    juce::ListenerList<Listener> listeners;

    //----------------------------------------------------------
    // レイテンシー補正用ディレイ
    //----------------------------------------------------------
    // juce::dsp::DelayLine<double> delayLine; // Replaced with custom AVX2 ring buffer
    convo::ScopedAlignedPtr<double> delayBuffer[2]; // L/R separate buffers
    int delayBufferCapacity = 0;
    int delayWritePos = 0;
    juce::SmoothedValue<double> latencySmoother;
    // [Issue 2 fix] latencySmoother のスレッドセーフティ向上のためのペンディングフラグ。
    // refreshLatency() (Message/Rebuild Thread) は直接 SmoothedValue を触らず、
    // このフラグと値を使用して Audio Thread に更新を委譲する。
    std::atomic<bool> latencyResetPending { false };
    std::atomic<double> pendingLatencyValue { 0.0 };
    // IR切り替え時の遅延ジャンプをクロスフェードに統合するためのフラグ。
    // refreshLatency() がセットし、process() がクロスフェード開始前に消費する。
    std::atomic<bool> latencyChangeRequested { false };
    // リサンプリング位相モード（r8brain CDSPResampler24IR に渡す）
    // Message Thread からのみ更新、LoaderThread が読む（memory_order_acquire）
    std::atomic<ResamplingPhaseMode> currentResamplingPhaseMode { ResamplingPhaseMode::Linear };
    // RCU経路でもUIがレイテンシー表示できるように、直近の内訳を保持する。
    std::atomic<int> uiAlgorithmLatencySamples { 0 };
    std::atomic<int> uiIrPeakLatencySamples { 0 };
    std::atomic<int> uiTotalLatencySamples { 0 };
    std::atomic<bool> uiDirectHeadActive { false };

    std::atomic<LatencySnapshot> cachedLatency {};
    std::atomic<bool> latencyChangePending { false };
    int lastReportedLatency = -1;

    // ドップラー効果対策: クロスフェード用
    juce::SmoothedValue<double> crossfadeGain;
    double oldDelay = 0.0;

    //----------------------------------------------------------
    // パラメータ（atomic）
    //----------------------------------------------------------
    // 【False Sharing 防止】頻繁な UI 更新変数を独立キャッシュラインへ配置
    #pragma warning(push)
    #pragma warning(disable: 4324)
    alignas(64) std::atomic<bool> bypassed{false};
    alignas(64) std::atomic<float> mixTarget{1.0f}; // UI からのターゲット値 (0.0-1.0)
    alignas(64) std::atomic<bool> experimentalDirectHeadEnabled{false};
    #pragma warning(pop)

    juce::SmoothedValue<double> mixSmoother; // オーディオスレッドでの平滑化用



    #pragma warning(push)
    #pragma warning(disable: 4324)
    alignas(64) std::atomic<int> phaseMode{static_cast<int>(PhaseMode::Mixed)};
    #pragma warning(pop)

    std::atomic<std::uint64_t> rebuildDebounceToken { 0 };
    std::atomic<bool> changeNotificationPending { false };
    std::atomic<bool> rebuildPendingAfterLoad { false };
    // リングバッファオーバーフローフラグ:
    //   Audio Thread から store(true)、process() 先頭で exchange(false) して rebuildPendingAfterLoad に転送。
    std::atomic<bool> overflowRequested { false };

    #pragma warning(push)
    #pragma warning(disable: 4324)
    alignas(64) std::atomic<int> rebuildDebounceMs { REBUILD_DEBOUNCE_DEFAULT_MS };
    #pragma warning(pop)

    // NUC 出力周波数フィルターモード (Message Thread で更新, finalizeNUC で読む)
    // int として保存し、使用時に enum へキャスト。
    std::atomic<int> nucHCMode { static_cast<int>(convo::HCMode::Natural) };
    std::atomic<int> nucLCMode { static_cast<int>(convo::LCMode::Natural) };
    #pragma warning(push)
    #pragma warning(disable: 4324)
    alignas(64) std::atomic<int> tailProcessingMode { 0 };
    alignas(64) std::atomic<float> tailRolloffStartHz { TAIL_ROLLOFF_START_DEFAULT_HZ };
    alignas(64) std::atomic<float> tailRolloffStrength { TAIL_ROLLOFF_STRENGTH_DEFAULT };
    alignas(64) std::atomic<float> partitionTailStrength { TAIL_PARTITION_STRENGTH_DEFAULT };
    #pragma warning(pop)
    std::atomic<float> targetIRLengthSec{IR_LENGTH_DEFAULT_SEC};
    std::atomic<float> autoDetectedIRLengthSec{IR_LENGTH_DEFAULT_SEC};
    std::atomic<bool> irLengthManualOverride{false};
    std::atomic<float> smoothingTimeSec{SMOOTHING_TIME_DEFAULT_SEC};
    std::atomic<float> mixedTransitionStartHz{MIXED_F1_DEFAULT_HZ};
    std::atomic<float> mixedTransitionEndHz{MIXED_F2_DEFAULT_HZ};
    std::atomic<float> mixedPreRingTau{MIXED_TAU_DEFAULT};

    // 【案 B】Smoothing Time 変更フラグ（Audio Thread 委譲用）
    std::atomic<bool> smoothingTimeChangePending { false };
    std::atomic<bool> mixSmootherResetPending { false };

    //----------------------------------------------------------
    // IR情報
    //----------------------------------------------------------
    juce::String irName;
    std::atomic<int> irLength { 0 };  // rebuildThread(read) と Message Thread(write) 間のデータレース防止
    // --- Visualization Data (accessed by worker and message threads) ---
    std::vector<float> irWaveform;
    std::vector<float> irMagnitudeSpectrum;
    double irSpectrumSampleRate = 0.0;
    mutable juce::CriticalSection visualizationDataLock;

    juce::File currentIrFile;
    juce::CriticalSection irFileLock;
    std::atomic<bool> currentIrOptimized { false };
    struct IRState {
        std::shared_ptr<juce::AudioBuffer<double>> irOwner;
        const juce::AudioBuffer<double>* ir = nullptr;
        double sampleRate = 0.0;
        uint64_t generation = 0;
    };
    std::atomic<IRState*> currentIRState { nullptr };
    AudioEngine* rcuProvider = nullptr;

    const IRState* acquireIRState() const noexcept;
    void releaseIRState(const IRState* state) const noexcept;
    void updateIRState(std::shared_ptr<juce::AudioBuffer<double>> newIR, double newSR);

    // MKL/AVX-512用に64byteアライメントを保証するアロケータを使用
    std::atomic<double> currentIRScale { 1.0 }; // IRのスケールファクター (Auto Makeup + Safety Margin)
    convo::ScopedAlignedPtr<float> cachedFFTBuffer; // FFT計算用キャッシュ (Message Thread)
    int cachedFFTBufferCapacity = 0;
    std::atomic<double> currentSampleRate { 0.0 };

    DFTI_DESCRIPTOR_HANDLE fftHandle = nullptr;
    int fftHandleSize = 0;

    //----------------------------------------------------------
    // Dry信号バッファ（Mix用）
    //----------------------------------------------------------
    juce::AudioBuffer<double> dryBuffer;
    convo::ScopedAlignedPtr<double> dryBufferStorage[2]; // Aligned storage for dryBuffer
    int dryBufferCapacity = 0;
    juce::AudioBuffer<double> smoothingBuffer; // スムーシングゲイン計算用 (Audio Threadでのメモリ確保回避)
    convo::ScopedAlignedPtr<double> smoothingBufferStorage[2]; // Aligned storage for smoothingBuffer
    int smoothingBufferCapacity = 0;

    // クロスフェード用バッファ
    juce::AudioBuffer<double> oldDryBuffer;
    convo::ScopedAlignedPtr<double> oldDryBufferStorage[2];
    int oldDryBufferCapacity = 0;

    // Wet信号用一時バッファ (StereoConvolver::process用)
    convo::ScopedAlignedPtr<double> wetBufferStorage[2];
    int wetBufferCapacity = 0;

    //----------------------------------------------------------
    // 準備完了フラグ
    //----------------------------------------------------------
    std::atomic<bool> isPrepared { false };
    bool visualizationEnabled = true; // Default true (for UI instance)
    std::atomic<int> currentBufferSize { 0 }; // prepareToPlay (Message Thread) で書き込み、Rebuild Worker Thread から読まれるためアトミック

    //----------------------------------------------------------
    // Phase 3: IR Cache
    //----------------------------------------------------------
    struct IRCacheKey {
        uint64_t fileHash;
        double sampleRate;
        PhaseMode phaseMode;
        float f1, f2, tau;
        int targetLength;

        bool operator<(const IRCacheKey& other) const {
            if (fileHash != other.fileHash) return fileHash < other.fileHash;
            if (sampleRate != other.sampleRate) return sampleRate < other.sampleRate;
            if (phaseMode != other.phaseMode) return phaseMode < other.phaseMode;
            if (std::abs(f1 - other.f1) > 1.0e-6f) return f1 < other.f1;
            if (std::abs(f2 - other.f2) > 1.0e-6f) return f2 < other.f2;
            if (std::abs(tau - other.tau) > 1.0e-6f) return tau < other.tau;
            return targetLength < other.targetLength;
        }
    };


        class ReaderGuard
        {
        public:
            explicit ReaderGuard(ConvolverProcessor& proc) noexcept : processor(proc)
            {
                processor.enterReader();
            }

            ~ReaderGuard() noexcept
            {
                processor.exitReader();
            }

            ReaderGuard(const ReaderGuard&) = delete;
            ReaderGuard& operator=(const ReaderGuard&) = delete;

        private:
            ConvolverProcessor& processor;
        };
    struct CacheEntry {
        std::shared_ptr<juce::AudioBuffer<double>> ir;
        std::vector<convo::SecondOrderAllpass> allpassSections;
        uint32_t lastUsedTime;
    };

    std::map<IRCacheKey, CacheEntry> irCache;
    juce::CriticalSection cacheMutex;
    static constexpr size_t MAX_CACHE_ENTRIES = 8;
    void evictOldestCacheEntry();

    static juce::AudioBuffer<double> convertToMixedPhase(ConvolverProcessor* owner,
                                                         uint64_t fileHash,
                                                         const juce::AudioBuffer<double>& linearIR,
                                                         const juce::AudioBuffer<double>& minimumIR,
                                                         double sampleRate,
                                                         double transitionLoHz,
                                                         double transitionHiHz,
                                                         double tau,
                                                         const std::function<bool()>& shouldExit,
                                                         bool* wasCancelled,
                                                         std::function<void(float)> progressCallback = nullptr);

    static juce::AudioBuffer<double> convertToMixedPhaseAllpass(ConvolverProcessor* owner,
                                                                 uint64_t fileHash,
                                                                 const juce::AudioBuffer<double>& linearIR,
                                                                 const juce::AudioBuffer<double>& minimumIR,
                                                                 double sampleRate,
                                                                 double transitionLoHz,
                                                                 double transitionHiHz,
                                                                 double tau,
                                                                 const std::function<bool()>& shouldExit,
                                                                 bool* wasCancelled,
                                                                 std::function<void(float)> progressCallback = nullptr);

    static juce::AudioBuffer<double> convertToMixedPhaseFallback(const juce::AudioBuffer<double>& linearIR,
                                                                 const juce::AudioBuffer<double>& minimumIR,
                                                                 double sampleRate,
                                                                 double transitionLoHz,
                                                                 double transitionHiHz,
                                                                 double tau,
                                                                 const std::function<bool()>& shouldExit,
                                                                 bool* wasCancelled);

    std::function<void(void*)> onSetThreadAffinity;
    std::atomic<bool> audioThreadAffinitySet{ false };

    std::unique_ptr<IRConverter> irConverter;
    std::unique_ptr<CacheManager> cacheManager;
    std::unique_ptr<ProgressiveUpgradeThread> upgradeThread;
    ConvolverRuntime runtime;
    std::atomic<bool> writerActive { false };
    std::atomic<int> targetUpgradeFFTSize { 4096 };
    std::atomic<bool> enableProgressiveUpgrade { true };
    std::atomic<size_t> maxCacheEntries { 10 };
    std::atomic<uint64_t> activeCacheKey { 0 };
    std::atomic<int> activeCacheFFTSize { 0 };

    // RCU 管理 (StereoConvolver 用)
    std::atomic<bool> firstProcessCall { true };

    static void retireStereoConvolver(StereoConvolver* conv, uint64_t retireEpoch);

    // ── Phase 0: Epoch-based RCU メンバー ──
    // SafeStateSwapper: IR 状態の lock-free swap と retired キュー管理
    SafeStateSwapper rcuSwapper;
    // DeferredFreeThread: 旧 ConvolverState を Audio Thread 外で安全に解放する専用スレッド
    // prepareToPlay() で生成、releaseResources() で停止・破棄する。
    std::unique_ptr<DeferredFreeThread> deferredFreeThread;
    // GenerationManager: IR ロードタスクの世代管理（陳腐化チェック用）
    GenerationManager convolverStateGeneration;

    JUCE_DECLARE_WEAK_REFERENCEABLE(ConvolverProcessor)
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ConvolverProcessor)

};
