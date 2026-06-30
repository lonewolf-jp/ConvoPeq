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
#include <optional>
#include "AlignedAllocation.h"
#include "MKLNonUniformConvolver.h"
#include "AllpassDesigner.h"
#include "IRConverter.h"
#include "InputBitDepthTransform.h"
#include "UltraHighRateDCBlocker.h"
#include "convolver/ConvolverProcessor.Internal.h"
#include "core/RCUReader.h"

// ── Phase 0: Epoch-based RCU 基盤ヘッダー ──
#include "GenerationManager.h"
#include "ConvolverState.h"
#include "DeferredFreeThread.h"
#include "core/ConvolverRuntimeCompatTypes.h"
#include "DeferredDeletionQueue.h"
#include "core/EpochDomain.h"
#include "DspNumericPolicy.h"
#include "DftiHandle.h"

class AudioEngine;
namespace convo::isr { class RuntimePublicationCoordinator; }
class CacheManager;
class ProgressiveUpgradeThread;

#pragma warning(push) // C4324 suppression scope begin: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
#pragma warning(disable : 4324) // Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
class ConvolverProcessor : public juce::ChangeBroadcaster,
                           private juce::Timer
{
public:
    // BuildSnapshot: 同期/シリアライズ/ハッシュ用の輸送スナップショット。
    // authoritative source-of-truth は pendingOverride store。
    struct BuildSnapshot
    {
        // ---- Structural hash 対象（rebuild / crossfade 要否に影響）----
        float mix = 1.0f;
        bool bypassed = false;
        int phaseMode = static_cast<int>(PhaseMode::AsIs);
        int resamplingPhaseMode = static_cast<int>(ResamplingPhaseMode::Linear);
        float smoothingTimeSec = SMOOTHING_TIME_DEFAULT_SEC;
        float targetIRLengthSec = IR_LENGTH_DEFAULT_SEC;
        float autoDetectedIRLengthSec = IR_LENGTH_DEFAULT_SEC;
        bool irLengthManualOverride = false;
        float mixedTransitionStartHz = MIXED_F1_DEFAULT_HZ;
        float mixedTransitionEndHz = MIXED_F2_DEFAULT_HZ;
        int rebuildDebounceMs = REBUILD_DEBOUNCE_DEFAULT_MS;
        bool experimentalDirectHeadEnabled = false;
        int tailMode = static_cast<int>(TailMode::LayerTailContouring);
        float tailStartSec = TAIL_START_DEFAULT_SEC;
        float tailStrength = TAIL_STRENGTH_DEFAULT;
        int tailL1L2Multiplier = TAIL_L1L2_MULT_DEFAULT;
        int targetUpgradeFFTSize = 0;
        bool enableProgressiveUpgrade = false;
        int maxCacheEntries = 0;

        // ---- Structural hash 非対象（同期/表示/復元用メタデータ）----
        juce::File irFile;
        juce::String irName;
        int irLength = 0;
        double currentIRScale = 1.0;

        // ---- Structural hash 対象（NUC フィルターモード）----
        int nucHCMode = static_cast<int>(convo::HCMode::Natural);
        int nucLCMode = static_cast<int>(convo::LCMode::Natural);

        // capture 時点のスナップショット整合確認用（比較/診断向け）
        std::uint64_t fingerprint = 0;
    };

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

    enum class TailMode : int
    {
        AirAbsorption = 0,
        LayerTailContouring = 1,
        Bypass = 2
    };

    void setResamplingPhaseMode(ResamplingPhaseMode mode);
    [[nodiscard]] ResamplingPhaseMode getResamplingPhaseMode() const;

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
    static constexpr int REBUILD_DEBOUNCE_MIN_MS = 10;
    static constexpr int REBUILD_DEBOUNCE_MAX_MS = 3000;
    static constexpr int REBUILD_DEBOUNCE_DEFAULT_MS = 20;
    static constexpr float TAIL_START_MIN_SEC = 0.01f;
    static constexpr float TAIL_START_MAX_SEC = 0.80f;
    static constexpr float TAIL_START_DEFAULT_SEC = 0.085f;
    static constexpr float TAIL_STRENGTH_MIN = 0.0f;
    static constexpr float TAIL_STRENGTH_MAX = 2.0f;
    static constexpr float TAIL_STRENGTH_DEFAULT = 1.0f;
    static constexpr int TAIL_L1L2_MULT_MIN = 2;
    static constexpr int TAIL_L1L2_MULT_MAX = 16;
    static constexpr int TAIL_L1L2_MULT_DEFAULT = 8;
    static constexpr float TAIL_EXTENDED_START_MIN_SEC = 0.12f;
    static constexpr float TAIL_EXTENDED_STRENGTH_MIN = 1.25f;
    static constexpr int TAIL_EXTENDED_L1L2_MULT_MIN = 12;

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

    void setRcuProvider(AudioEngine& engine) noexcept { rcuProvider = engine; }
    void setRetireCoordinator(convo::isr::RuntimePublicationCoordinator* coordinator) noexcept
    {
        rcuSwapper.setRetireCoordinator(coordinator);
    }

    // RCU リーダー (Audio Thread のみ)


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
    void applyComputedIR(std::unique_ptr<ConvolverIRPayload> payload);
    // acquire: applyComputedIR の release と HB し、IR 適用時刻を取得。
    [[nodiscard]] int64_t getLastPreparedIRApplyTicks() const noexcept { return convo::consumeAtomic(lastPreparedIRApplyTicks, std::memory_order_acquire); } // acquire: applyComputedIR の release と HB
    void stopUpgradeThread();
    void startProgressiveUpgrade(const juce::File& file,
                                 double sampleRate,
                                 int currentFFTSize,
                                 uint64_t generation,
                                 uint64_t baseKey);

    void setTargetUpgradeFFTSize(int fftSize);
    [[nodiscard]] int getTargetUpgradeFFTSize() const;
    void setEnableProgressiveUpgrade(bool enable);
    [[nodiscard]] bool isProgressiveUpgradeEnabled() const;
    void setMaxCacheEntries(size_t maxEntries);
    [[nodiscard]] size_t getMaxCacheEntries() const;
    void clearCache();
    [[nodiscard]] bool isCacheEntrySafeToDelete(uint64_t cacheKey, int fftSize) const;

    // メイン処理（Audio Thread）
    //
    //----------------------------------------------------------
    void process(juce::dsp::AudioBlock<double>& block);

    //----------------------------------------------------------
    // バイパス制御
    //----------------------------------------------------------
    void setBypass(bool shouldBypass);
    [[nodiscard]] bool isBypassed() { const juce::ScopedLock lock(pendingOverrideLock); return pendingOverride.bypassed; }
    // acquire: LoaderThread/executeCommit の publishNewConvolverState release と HB し、有効な state を取得。
    [[nodiscard]] const convo::ConvolverState* getConvolverState() const { return convo::consumeAtomic(convolverState, std::memory_order_acquire); } // acquire: publishNewConvolverState の release と HB
    void enterStateReader(int /*readerIndex*/) const noexcept {}
    void exitStateReader(int /*readerIndex*/) const noexcept {}

    void enterGlobalReader(int /*readerIndex*/) const noexcept;
    void exitGlobalReader(int /*readerIndex*/) const noexcept;

    //----------------------------------------------------------
    // Dry/Wet Mix (0.0 = Dry only, 1.0 = Wet only)
    //----------------------------------------------------------
    void setMix(float mixAmount);
    [[nodiscard]] float getMix() const;

    //----------------------------------------------------------
    // IR Phase Mode
    //----------------------------------------------------------
    void setPhaseMode(PhaseMode mode);
    [[nodiscard]] PhaseMode getPhaseMode() const;

    // 後方互換API
    void setUseMinPhase(bool useMinPhase);
    [[nodiscard]] bool getUseMinPhase() const { return getPhaseMode() == PhaseMode::Minimum; }

    //------------------------------------------------------------------
    // NUC 出力周波数フィルターモード設定
    //
    // フィルターは SetImpulse() 内で irFreqDomain に焼き込まれる。
    // モード変更時は内部で rebuildAllIRs() を呼んで NUC を再構築する。
    // Message Thread からのみ呼ぶこと。
    //------------------------------------------------------------------
    void setNUCFilterModes(convo::HCMode hcMode, convo::LCMode lcMode);

    //------------------------------------------------------------------
    // NUC テール処理パラメータ
    // Tail Mode / Start / Strength / L1-L2 Mult は pendingOverride に保持し、
    // rebuild 時に FilterSpec 経由で NUC 構成へ反映する。
    //------------------------------------------------------------------

    //----------------------------------------------------------
    // Experimental Direct Head Flag
    // 段階導入用の機能フラグ。変更時はIRを再構築する。
    //----------------------------------------------------------
    void setExperimentalDirectHeadEnabled(bool enabled);
    [[nodiscard]] bool getExperimentalDirectHeadEnabled() const;

    //----------------------------------------------------------
    // Smoothing Time
    //----------------------------------------------------------
    void setSmoothingTime(float timeSec);
    [[nodiscard]] float getSmoothingTime() const;

    //----------------------------------------------------------
    // RT-safe setters (Audio Thread dispatch 専用)
    // listeners.call() / 直接rebuild要求 を呼ばない。
    // （現行経路では通常使用しない。互換用途として保持）
    //----------------------------------------------------------
    // setMixRT / setSmoothingTimeRT: H3 修正により廃止 (shadow atomic 除去)

    //----------------------------------------------------------
    // Mixed Phase Parameters (f1/f2)
    //----------------------------------------------------------
    void setMixedTransitionStartHz(float hz);
    [[nodiscard]] float getMixedTransitionStartHz() const;
    void setMixedTransitionEndHz(float hz);
    [[nodiscard]] float getMixedTransitionEndHz() const;

    //----------------------------------------------------------
    // Rebuild Debounce Time (Message/Worker burst control)
    //----------------------------------------------------------
    void setRebuildDebounceMs(int ms);
    [[nodiscard]] int getRebuildDebounceMs() const;

    //----------------------------------------------------------
    // Tail Parameters
    //----------------------------------------------------------
    void setTailMode(TailMode mode);
    [[nodiscard]] TailMode getTailMode() const;
    void setTailStartSec(float sec);
    [[nodiscard]] float getTailStartSec() const;
    void setTailStrength(float strength);
    [[nodiscard]] float getTailStrength() const;
    void setTailL1L2Multiplier(int multiplier);
    [[nodiscard]] int getTailL1L2Multiplier() const;

    //----------------------------------------------------------
    // IR Length
    //----------------------------------------------------------
    void setTargetIRLength(float timeSec);
    [[nodiscard]] float getTargetIRLength() const;
    void applyAutoDetectedIRLength(float timeSec);
    void setIRLengthManualOverride(bool isManual);
    [[nodiscard]] bool hasManualIRLengthOverride() const;
    [[nodiscard]] float getAutoDetectedIRLength() const;
    [[nodiscard]] static float getMaximumAllowedIRLengthSecForSampleRate(double sampleRate);
    [[nodiscard]] float getMaximumAllowedIRLengthSec(double sampleRate = 0.0) const;
    [[nodiscard]] static IRLoadPreview analyzeImpulseResponseFile(const juce::File& irFile, double processingSampleRate);

    //----------------------------------------------------------
    // 状態リセット
    //----------------------------------------------------------
    void reset();

    //----------------------------------------------------------
    // 状態取得
    //----------------------------------------------------------
    [[nodiscard]] bool isIRLoaded() const
    {
        // acquire: LoaderThread/executeCommit の release と HB し、state/engine/metadata を取得。
        const bool hasPublishedState = (convo::consumeAtomic(convolverState, std::memory_order_acquire) != nullptr);
        const bool hasActiveEngine = (loadActiveEngine(std::memory_order_acquire) != nullptr);
        const bool hasIRMetadata = (convo::consumeAtomic(currentIRState, std::memory_order_acquire) != nullptr)
            || (convo::consumeAtomic(irLength, std::memory_order_acquire) > 0);
        return hasPublishedState || (hasActiveEngine && hasIRMetadata);
    }
    // acquire: LoaderThread の release と HB し、loading flag を観測。
    // acquire: LoaderThread の release と HB し、ロード中フラグを観測。
    [[nodiscard]] bool isLoadingIR() const { return convo::consumeAtomic(isLoading, std::memory_order_acquire); } // acquire: LoaderThread の release と HB
    // acquire: LoaderThread の release と HB し、最終化フラグを観測。
    [[nodiscard]] bool isIRFinalized() const noexcept { return convo::consumeAtomic(irFinalized, std::memory_order_acquire); } // acquire: LoaderThread の release と HB
    [[nodiscard]] juce::String getIRName() const { return irName; }
    // acquire: LoaderThread/applyComputedIR の release と HB し、IR長を取得。
    [[nodiscard]] int getIRLength() const { return convo::consumeAtomic(irLength, std::memory_order_acquire); } // acquire: apply/load 側 release と HB
    [[nodiscard]] juce::String getLastError() const { return lastError; }
    // acquire: setLoadingProgress の release と HB し、ロード進捗値を取得。
    [[nodiscard]] float getLoadProgress() const { return convo::consumeAtomic(loadProgress, std::memory_order_acquire); } // acquire: setLoadingProgress の release と HB
    // acquire: setMixedPhaseState の release と HB し、混合フェーズ状態を取得。
    [[nodiscard]] int getMixedPhaseState() const noexcept { return convo::consumeAtomic(mixedPhaseState, std::memory_order_acquire); } // acquire: setMixedPhaseState の release と HB
    // release: getMixedPhaseState の acquire と HB し、混合フェーズ状態を公開。
    void setMixedPhaseState(int state) noexcept { convo::publishAtomic(mixedPhaseState, juce::jlimit(0, 2, state), std::memory_order_release); } // release: getMixedPhaseState の acquire と HB
    void setLoadingProgress(float p);
    // acquire: setCurrentBufferSize の release と HB し、有効なバッファサイズを取得。
    [[nodiscard]] int getCurrentBufferSize() const { return convo::consumeAtomic(currentBufferSize, std::memory_order_acquire); } // acquire: setCurrentBufferSize の release と HB
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

    [[nodiscard]] LatencyBreakdown getLatencyBreakdown() const;
    [[nodiscard]] int getLatencySamples() const;
    [[nodiscard]] int getTotalLatencySamples() const;

    struct RebuildAutomationDiagnostics
    {
        std::uint64_t requestCount = 0;
        std::uint64_t deferredAfterLoadCount = 0;
        std::uint64_t scheduledCount = 0;
        std::uint64_t triggeredCount = 0;
    };

    [[nodiscard]] RebuildAutomationDiagnostics getRebuildAutomationDiagnostics() const noexcept
    {
        // acquire: rebuild automation の各カウント更新（release）と HB し、診断値を取得。
        return {
            convo::consumeAtomic(debugDebouncedRebuildRequestCount, std::memory_order_acquire),
            convo::consumeAtomic(debugDebouncedRebuildDeferredAfterLoadCount, std::memory_order_acquire),
            convo::consumeAtomic(debugDebouncedRebuildScheduledCount, std::memory_order_acquire),
            convo::consumeAtomic(debugDebouncedRebuildTriggeredCount, std::memory_order_acquire)
        };
    }

    //----------------------------------------------------------
    // 波形表示用データ取得
    //----------------------------------------------------------
    [[nodiscard]] std::vector<float> getIRWaveform();

    //----------------------------------------------------------
    // 周波数特性表示用データ取得
    //----------------------------------------------------------
    [[nodiscard]] std::vector<float> getIRMagnitudeSpectrum();
    [[nodiscard]] double getIRSpectrumSampleRate();

    //----------------------------------------------------------
    // State Management
    //----------------------------------------------------------
    [[nodiscard]] juce::ValueTree getState() const;
    void setState (const juce::ValueTree& state);
    [[nodiscard]] BuildSnapshot captureBuildSnapshot() const;
    void applyBuildSnapshot(const BuildSnapshot& snapshot);

    //----------------------------------------------------------
    // リビルド (サンプルレート変更時など)
    //----------------------------------------------------------
    enum class IncrementalRebuildSliceResult
    {
        InProgress,
        Completed,
        Failed
    };

    void rebuildAllIRs();
    void rebuildAllIRsSynchronous(std::function<bool()> shouldCancel = nullptr);
    bool beginIncrementalRebuild(std::function<bool()> shouldCancel = nullptr);
    IncrementalRebuildSliceResult advanceIncrementalRebuild() noexcept;
    void resetIncrementalRebuild() noexcept;
    void setUseIncrementalRebuild(bool enable) noexcept;
    [[nodiscard]] bool isIncrementalRebuildEnabled() const noexcept;
    void invalidatePendingLoads();

    // 他のインスタンスから状態を同期 (AudioEngine用)
    void syncStateFrom(const ConvolverProcessor& other);

        // 構造的パラメータのハッシュ値を返す（クロスフェード要否判定用）
        [[nodiscard]] uint64_t getStructuralHash() const noexcept;

        // 構造変更検出用 getter（不足分のみ追加）
        [[nodiscard]] uint64_t getActiveCacheKey() const noexcept;
        [[nodiscard]] int getActiveCacheFFTSize() const noexcept;
        [[nodiscard]] int getNUCHCMode() const noexcept;
        [[nodiscard]] int getNUCLCMode() const noexcept;

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
                                          int knownBlockSize,
                                          int preferredCallSize,
                                          bool isRebuild,
                                          const juce::File& irFile,
                                          const BuildSnapshot& buildSnapshot,
                                          double scaleFactor, // This is for newConv->init
                                          std::unique_ptr<juce::AudioBuffer<double>> loadedIR,
                                          std::unique_ptr<juce::AudioBuffer<double>> displayIR);

    // 可視化データ生成の制御 (DSP用インスタンスでは無効化してメモリを節約)
    void setVisualizationEnabled(bool enabled) { visualizationEnabled = enabled; }
    [[nodiscard]] bool isVisualizationEnabled() const { return visualizationEnabled; }

    // ガベージコレクション (Message Threadから定期的に呼ぶ)
    void cleanup();
    void forceCleanup();

    // 診断用：すべての NUC オブジェクトのガードチェックを一括実行（デバッグビルドのみ）
    #ifdef NUC_DEBUG_GUARDS
    void debugCheckNucGuards() const noexcept
    {
        auto* engine = loadActiveEngine(std::memory_order_acquire);
        if (engine)
        {
            for (int i = 0; i < 2; ++i)
                if (engine->nucConvolvers[i])
                    engine->nucConvolvers[i]->checkGuards();
        }
    }
    #endif


    // ── Phase 0: Epoch-based RCU 状態更新 ──
    // Message Thread から呼ぶ。新しい ConvolverState を atomic にスワップし、
    // 旧状態を DeferredFreeThread に委ねる。
    // GenerationManager で陳腐化チェックを行い、古いタスク結果は破棄する。
    //
    // @param newState  新しい状態（所有権を移譲）。世代チェックに失敗した場合は即削除。
    void updateConvolverState(convo::ConvolverState* newState);
    void updateConvolverState(std::unique_ptr<convo::ConvolverState> newState);

    [[nodiscard]] bool isConvolverGenerationCurrent(uint64_t generation) const
    {
        return convolverStateGeneration.isCurrentGeneration(generation);
    }

private:
    static std::atomic<int> latencyClampCounterStorage_;
    static std::atomic<int>& latencyClampCounter() noexcept;

    struct StereoConvolver;
#include "convolver/ConvolverProcessor.LoaderThreadInline.h"

#include "audioengine/AtomicAccess.h"

    struct IncrementalRebuildJob
    {
        enum class Stage
        {
            Idle,
            Prepared,
            Building,
            FinalizingPrepare,
            FinalizingApply,
            Done
        };

        Stage stage { Stage::Idle };
        std::unique_ptr<juce::AudioBuffer<double>> preparedIR;
        double preparedSampleRate = 0.0;
        std::function<bool()> shouldCancel;
        // LoaderThread ステートマシン (incremental 経路専用)
        std::unique_ptr<LoaderThread> incrementalLoader;
        bool loaderInitialized = false;
        StereoConvolver* pendingConv = nullptr;
        juce::AudioBuffer<double> pendingLoadedIR;
        double pendingLoadedSR = 0.0;
        int pendingTargetLength = 0;
        juce::AudioBuffer<double> pendingDisplayIR;
        double pendingScaleFactor = 1.0;
        juce::File pendingFile;
        bool pendingIsRebuild = false;
        int finalizeApplyStep = 0;
        bool finalizeApplied = false;
        juce::String lastError;

        void reset() noexcept;
    };

    void timerCallback() override;
    void processBypassWithLatencyCompensation(juce::dsp::AudioBlock<double>& block,
                                              const StereoConvolver& conv) noexcept;
    void postCoalescedChangeNotification();
    void updateLatencyCache() noexcept;
    void requestHostDisplayUpdate();
    void debugCheckAtomicLockFree() const;
    bool runIncrementalBuildStep(IncrementalRebuildJob& job);
    bool runIncrementalFinalizeStep(IncrementalRebuildJob& job);

    void commitNewConvolver(StereoConvolver* newConv,
                            std::unique_ptr<juce::AudioBuffer<double>> loadedIR,
                            double loadedSR, int targetLength, bool isRebuild,
                            const juce::File& file, double scaleFactor,
                            std::unique_ptr<juce::AudioBuffer<double>> displayIR);

    void switchEngineOnMessageThread(StereoConvolver* newEngine) noexcept;

    void applyNewStateBindStep(std::unique_ptr<juce::AudioBuffer<double>> loadedIR,
                               double loadedSR,
                               bool isRebuild,
                               const juce::File& file,
                               double scaleFactor);
    void applyNewStateUpdateStep(std::unique_ptr<juce::AudioBuffer<double>> displayIR,
                                 double loadedSR);
    void applyNewStatePublishStep(StereoConvolver* newConv,
                                  int targetLength,
                                  double loadedSR);
    void applyNewStateNotifyStep();

    // H3保守性: BuildSnapshot と pendingOverride store の相互マッピングを一元化
    // 呼び出し側で pendingOverrideLock を保持している前提。
    // [更新ガイド]
    // - BuildSnapshot または pendingOverride store に項目を追加/削除したら、
    //   2関数を同時に更新すること。
    // - 併せて captureBuildSnapshot() の fingerprint 対象、
    //   getStructuralHash() の対象可否も必ず見直すこと。
    void copyPendingToSnapshotUnlocked(BuildSnapshot& snapshot) const noexcept;
    void copySnapshotToPendingUnlocked(const BuildSnapshot& snapshot) noexcept;

    void applyNewState(StereoConvolver* newConv, std::unique_ptr<juce::AudioBuffer<double>> loadedIR, double loadedSR, int targetLength, bool isRebuild, const juce::File& file, double scaleFactor, std::unique_ptr<juce::AudioBuffer<double>> displayIR);
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

        // Clone用に初期化パラメータを保存
        double storedSampleRate = 0.0;
        int storedKnownBlockSize = 0;
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

        // 二重 retire 防止フラグ
        std::atomic<bool> retired { false };

        // リソース解放を一括で行う静的関数（retire コールバック用）
        static void destroyStereoConvolver(void* p) noexcept
        {
            auto* sc = static_cast<StereoConvolver*>(p);
            if (!sc) return;
            destroyNUCConvolver(sc->nucConvolvers[0]);
            destroyNUCConvolver(sc->nucConvolvers[1]);
            if (sc->irData[0]) { convo::aligned_free(sc->irData[0]); sc->irData[0] = nullptr; }
            if (sc->irData[1]) { convo::aligned_free(sc->irData[1]); sc->irData[1] = nullptr; }
            sc->~StereoConvolver();
            convo::aligned_free(sc);
        }

        // 外部から安全に破棄するためのエントリポイント
        static void retireStereoConvolver(StereoConvolver* sc, AudioEngine* provider = nullptr) noexcept;

        // デストラクタは空（実際の解放は retire 経由）
        ~StereoConvolver() {
            // 直接 delete は禁止だが、retire 経由の正規破棄ではデストラクタ自体は呼ばれる。
            // ここでは「未解放リソースを抱えたまま破棄されていないか」のみを検証する。
            #if JUCE_DEBUG
            jassert(nucConvolvers[0] == nullptr && nucConvolvers[1] == nullptr);
            jassert(irData[0] == nullptr && irData[1] == nullptr);
            #endif
        }

        // コピーコンストラクタは禁止 (NUCエンジンは複製コストが高く、状態を持つため)
        StereoConvolver(const StereoConvolver& other) = delete;

        // 代入演算子は禁止 (使用しないため)
        StereoConvolver& operator=(const StereoConvolver&) = delete;

        bool init(double* irL, double* irR, int length, double sr, int peakDelay, int knownBlockSize, int preferredCallSize, double scale = 1.0,
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
            storedSampleRate = sr;
            storedKnownBlockSize = knownBlockSize;
            storedScale = scale;
            storedDirectHeadEnabled = enableDirectHead;

            try
            {
                auto nuc0 = convo::aligned_make_unique<convo::MKLNonUniformConvolver>();
                auto nuc1 = convo::aligned_make_unique<convo::MKLNonUniformConvolver>();

                if (nuc0->SetImpulse(irData[0], irDataLength, knownBlockSize, scale, enableDirectHead, filterSpec) &&
                    nuc1->SetImpulse(irData[1], irDataLength, knownBlockSize, scale, enableDirectHead, filterSpec))
                {
                    destroyNUCConvolver(nucConvolvers[0]);
                    destroyNUCConvolver(nucConvolvers[1]);
                    nucConvolvers[0] = nuc0.release();
                    nucConvolvers[1] = nuc1.release();

                    latency   = nucConvolvers[0]->getLatency();
                    DBG("Convolver: NUC Engine Active. Latency: " << latency << " samples");
                    if (ownerProcessor != nullptr)
                    {
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
            if (irData[0]) { convo::aligned_free(irData[0]); irData[0] = nullptr; }
            if (irData[1]) { convo::aligned_free(irData[1]); irData[1] = nullptr; }
            irDataLength = 0;
            latency = 0;
            this->irLatency = 0;
            return false;
        }

        // Deep Copyを作成する。
        // 失敗時 (MKLメモリ確保失敗等) は nullptr を返す。呼び出し元で必ずチェックすること。
        [[nodiscard]] StereoConvolver* clone() const
        {
            try
            {
                auto newConv = convo::aligned_make_unique<StereoConvolver>();

                if (irDataLength > 0 && irData[0] && irData[1])
                {
                    auto l = convo::makeAlignedArray<double>(static_cast<size_t>(irDataLength));
                    auto r = convo::makeAlignedArray<double>(static_cast<size_t>(irDataLength));

                    std::memcpy(l.get(), irData[0], irDataLength * sizeof(double));
                    std::memcpy(r.get(), irData[1], irDataLength * sizeof(double));

                    if (!newConv->init(l.release(), r.release(), irDataLength, storedSampleRate, irLatency, storedKnownBlockSize, callQuantumSamples, storedScale, storedDirectHeadEnabled))
                        return nullptr;
                }
                return newConv.release();
            }
            catch (const std::bad_alloc&)
            {
                return nullptr;
            }
        }

        [[nodiscard]] bool areNUCDescriptorsCommitted() const noexcept
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

    std::atomic<std::uintptr_t> m_activeEngineBits { 0 }; // uintptr_t-backed lock-free handle
    std::atomic<bool> isLoading { false };
    std::atomic<bool> isRebuilding { false };
    std::atomic<bool> irFinalized { false };
    std::atomic<bool> useIncrementalRebuild { false };
    std::unique_ptr<IncrementalRebuildJob> rebuildJob;
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
    convo::LinearRamp latencySmoother;
    // [Issue 2 fix] latencySmoother のスレッドセーフティ向上のためのペンディングフラグ。
    // refreshLatency() (Message/Rebuild Thread) は直接 SmoothedValue を触らず、
    // このフラグと値を使用して Audio Thread に更新を委譲する。
    // 世代カウンター方式: NonRT が fetch_add(1), RT が load+比較のみ (atomic write 禁止)
    std::atomic<uint64_t> latencyResetPendingGen { 0 };
    std::atomic<double> pendingLatencyValue { 0.0 };
    // IR切り替え時の遅延ジャンプをクロスフェードに統合するためのフラグ。
    // refreshLatency() がセットし、process() がクロスフェード開始前に消費する。
    std::atomic<uint64_t> latencyChangeRequestedGen { 0 };
    // リサンプリング位相モード（r8brain CDSPResampler24IR に渡す）
    // Message Thread からのみ更新、LoaderThread が読む（memory_order_acquire）
    // pendingOverride に統合済み
    // RCU経路でもUIがレイテンシー表示できるように、直近の内訳を保持する。
    std::atomic<int> uiAlgorithmLatencySamples { 0 };
    std::atomic<int> uiIrPeakLatencySamples { 0 };
    std::atomic<int> uiTotalLatencySamples { 0 };
    std::atomic<bool> uiDirectHeadActive { false };

    std::atomic<LatencySnapshot*> cachedLatency { new LatencySnapshot() };
    std::atomic<bool> latencyChangePending { false };
    int lastReportedLatency = -1;
    int lastReportedClampCount_ = 0;

    // ドップラー効果対策: クロスフェード用
    convo::LinearRamp crossfadeGain;
    double oldDelay = 0.0;

    //----------------------------------------------------------
    // パラメータ（atomic）
    //----------------------------------------------------------
    // 【False Sharing 防止】頻繁な UI 更新変数を独立キャッシュラインへ配置
    // H3: bypassed / mixTarget shadow atomics 廃止済み。pendingOverride が唯一の Source of Truth。

    convo::RuntimeStateSwapper rcuSwapper;
    convo::LinearRamp mixSmoother; // オーディオスレッドでの平滑化用

    struct RuntimeProcessSnapshot
    {
        bool bypassed { false };
        float mixTarget { 1.0f };
        float smoothingTimeSec { SMOOTHING_TIME_DEFAULT_SEC };
        double currentSampleRate { 0.0 };
    };

    // H3: UI/Message Thread 側の authoritative parameter store。
    // BuildSnapshot はシリアライズ/輸送用として維持し、
    // pending state はこの専用構造体で責務分離する。
    struct PendingOverrideStore
    {
        float mix = 1.0f;
        bool bypassed = false;
        int phaseMode = static_cast<int>(PhaseMode::AsIs);
        int resamplingPhaseMode = static_cast<int>(ResamplingPhaseMode::Linear);
        float smoothingTimeSec = SMOOTHING_TIME_DEFAULT_SEC;
        float targetIRLengthSec = IR_LENGTH_DEFAULT_SEC;
        float autoDetectedIRLengthSec = IR_LENGTH_DEFAULT_SEC;
        bool irLengthManualOverride = false;
        float mixedTransitionStartHz = MIXED_F1_DEFAULT_HZ;
        float mixedTransitionEndHz = MIXED_F2_DEFAULT_HZ;
        int rebuildDebounceMs = REBUILD_DEBOUNCE_DEFAULT_MS;
        bool experimentalDirectHeadEnabled = false;
        int tailMode = static_cast<int>(TailMode::LayerTailContouring);
        float tailStartSec = TAIL_START_DEFAULT_SEC;
        float tailStrength = TAIL_STRENGTH_DEFAULT;
        int tailL1L2Multiplier = TAIL_L1L2_MULT_DEFAULT;
        int targetUpgradeFFTSize = 0;
        bool enableProgressiveUpgrade = false;
        int maxCacheEntries = 0;
        int nucHCMode = static_cast<int>(convo::HCMode::Natural);
        int nucLCMode = static_cast<int>(convo::LCMode::Natural);
    };

    alignas(64) RuntimeProcessSnapshot runtimeProcessSnapshots[2] {};
    std::atomic<uint32_t> runtimeProcessSnapshotIndex { 0 };

    static StereoConvolver* fromEngineBits(std::uintptr_t bits) noexcept
    {
        return reinterpret_cast<StereoConvolver*>(bits);
    }

    static std::uintptr_t toEngineBits(StereoConvolver* ptr) noexcept
    {
        return static_cast<std::uintptr_t>(reinterpret_cast<std::uintptr_t>(ptr));
    }

    // acquire: publishActiveEngine/exchangeActiveEngine の release/acq_rel と HB し、アクティブエンジンを取得。
    [[nodiscard]] StereoConvolver* loadActiveEngine(std::memory_order order = std::memory_order_acquire) const noexcept // default acquire: publish/exchange 側 release/acq_rel と HB
    {
        return fromEngineBits(convo::consumeAtomic(m_activeEngineBits, static_cast<std::memory_order>(order)));
    }

    // acq_rel: 旧エンジンを acquire で取得、新エンジンを release で公開。loadActiveEngine と双方向 HB。
    StereoConvolver* exchangeActiveEngine(StereoConvolver* value,
                                          std::memory_order order = std::memory_order_acq_rel) noexcept // default acq_rel: 旧値取得(acquire)+新値公開(release)
    {
        return fromEngineBits(convo::exchangeAtomic(m_activeEngineBits,
                                                    toEngineBits(value),
                                                    static_cast<std::memory_order>(order)));
    }

    // release: loadActiveEngine の acquire と HB し、アクティブエンジンを公開。
    void publishActiveEngine(StereoConvolver* value,
                             std::memory_order order = std::memory_order_release) noexcept // default release: loadActiveEngine acquire 側へ公開
    {
        convo::publishAtomic(m_activeEngineBits,
                             toEngineBits(value),
                             static_cast<std::memory_order>(order));
    }

    // acquire: publishRuntimeProcessSnapshot の release と HB し、有効なsnapshot indexを取得。
    [[nodiscard]] RuntimeProcessSnapshot captureRuntimeProcessSnapshot() const noexcept
    {
        const uint32_t index = convo::consumeAtomic(runtimeProcessSnapshotIndex, std::memory_order_acquire) & 1u;
        return runtimeProcessSnapshots[index];
    }

    void publishRuntimeProcessSnapshot() noexcept
    {
        // 書き込み側自身の前回値読み取りのため HB 不要。relaxed で十分。
        const uint32_t current = convo::consumeAtomic(runtimeProcessSnapshotIndex, std::memory_order_relaxed) & 1u;
        const uint32_t next = current ^ 1u;

        // H3: shadow atomic を廃止し pendingOverride（唯一の Source of Truth）から読む
        {
            const juce::ScopedLock lock(pendingOverrideLock);
            runtimeProcessSnapshots[next].bypassed       = pendingOverride.bypassed;
            runtimeProcessSnapshots[next].mixTarget      = pendingOverride.mix;
            runtimeProcessSnapshots[next].smoothingTimeSec = pendingOverride.smoothingTimeSec;
        }
        // acquire: Message Thread の release と HB し、有効なsample rateを取得してsnapshot更新。
        runtimeProcessSnapshots[next].currentSampleRate = convo::consumeAtomic(currentSampleRate, std::memory_order_acquire);

        // release: captureRuntimeProcessSnapshot の acquire と HB し、新snapshot indexを公開。
        convo::publishAtomic(runtimeProcessSnapshotIndex, next, std::memory_order_release);
    }



    std::atomic<std::uint64_t> rebuildDebounceToken { 0 };
    std::atomic<std::uint64_t> debugDebouncedRebuildRequestCount { 0 };
    std::atomic<std::uint64_t> debugDebouncedRebuildDeferredAfterLoadCount { 0 };
    std::atomic<std::uint64_t> debugDebouncedRebuildScheduledCount { 0 };
    std::atomic<std::uint64_t> debugDebouncedRebuildTriggeredCount { 0 };
    std::atomic<bool> changeNotificationPending { false };
    std::atomic<bool> rebuildPendingAfterLoad { false };
    // リングバッファオーバーフローフラグ:
    //   Audio Thread から store(true)、process() 先頭で exchange(false) して rebuildPendingAfterLoad に転送。
    // overflowRequested は削除: RT 側で値を使用せず dead code だったため除去

    // H3: smoothingTimeSec shadow atomic 廃止済み。pendingOverride.smoothingTimeSec が唯一の Source of Truth。

    // 【案 B】Smoothing Time 変更フラグ（Audio Thread 委譲用）
    // 世代カウンター方式
    std::atomic<uint64_t> smoothingTimeChangePendingGen { 0 };
    std::atomic<uint64_t> mixSmootherResetPendingGen { 0 };

    // [H3] Pending parameter override for snapshot-based setter transition
    // Setters write parameters here instead of directly publishing atomics.
    // This pending override is captured by rebuild requests.
    PendingOverrideStore pendingOverride;
    juce::CriticalSection pendingOverrideLock;

    //----------------------------------------------------------
    // IR情報
    //----------------------------------------------------------
    juce::String irName;
    std::atomic<int> irLength { 0 };  // rebuildThread(read) と Message Thread(write) 間のデータレース防止
    // --- Visualization Data (accessed by worker and message threads) ---
    std::vector<float> irWaveform;
    std::vector<float> irMagnitudeSpectrum;
    double irSpectrumSampleRate = 0.0;
    juce::CriticalSection visualizationDataLock;

    juce::File currentIrFile;
    juce::CriticalSection irFileLock;
    std::atomic<bool> currentIrOptimized { false };
    std::atomic<int64_t> lastPreparedIRApplyTicks { 0 };
    struct IRState {
        std::unique_ptr<juce::AudioBuffer<double>> irOwner;
        const juce::AudioBuffer<double>* ir = nullptr;
        double sampleRate = 0.0;
        uint64_t generation = 0;
    };
    std::atomic<IRState*> currentIRState { nullptr };
    std::optional<std::reference_wrapper<AudioEngine>> rcuProvider;

    [[nodiscard]] AudioEngine* getRcuProvider() noexcept { return rcuProvider ? &rcuProvider->get() : nullptr; }
    [[nodiscard]] AudioEngine* getRcuProvider() const noexcept { return rcuProvider ? &rcuProvider->get() : nullptr; }

    [[nodiscard]] const IRState* acquireIRState() const noexcept;
    void releaseIRState(const IRState* state) const noexcept;
    void updateIRState(const juce::AudioBuffer<double>& newIR, double newSR);
    void updateIRState(const std::unique_ptr<juce::AudioBuffer<double>>& newIR, double newSR)
    {
        if (newIR)
            updateIRState(*newIR, newSR);
    }

    // MKL/AVX-512用に64byteアライメントを保証するアロケータを使用
public: // Added for AudioEngine access
    // Thread-safe IR state transfer from source convolver (copies the AudioBuffer)
    // Must be called before rebuildAllIRsSynchronous() on this instance.
    void transferIRStateFrom(const ConvolverProcessor& source) noexcept
    {
        const IRState* srcState = source.acquireIRState();
        if (srcState && srcState->ir && srcState->ir->getNumSamples() > 0 && srcState->sampleRate > 0.0)
        {
            const int channels = srcState->ir->getNumChannels();
            const int length   = srcState->ir->getNumSamples();
            updateIRState(*srcState->ir, srcState->sampleRate);
            juce::Logger::writeToLog("[CONV_IR] transferIRStateFrom: IR transferred ch="
                + juce::String(channels) + " len=" + juce::String(length)
                + " sr=" + juce::String(srcState->sampleRate, 1));
        }
        else
        {
            juce::Logger::writeToLog("[CONV_IR] transferIRStateFrom: no IR data to transfer");
        }
        source.releaseIRState(srcState);
    }

    [[nodiscard]] double getCurrentIRScale() const noexcept { return convo::consumeAtomic(currentIRScale, std::memory_order_acquire); } // acquire: apply/load 側 release と HB
    std::atomic<double> currentIRScale { 1.0 }; // IRのスケールファクター (Auto Makeup + Safety Margin)
    convo::ScopedAlignedPtr<float> cachedFFTBuffer; // FFT計算用キャッシュ (Message Thread)
    int cachedFFTBufferCapacity = 0;
    convo::ScopedAlignedPtr<float> cachedLinearMagsBuffer; // スムージング入力用キャッシュ
    convo::ScopedAlignedPtr<float> cachedSmoothedMagsBuffer; // スムージング出力用キャッシュ
    int cachedMagnitudeBufferCapacity = 0;
    std::atomic<double> currentSampleRate { 0.0 };
    // work60: 現在のcallbackSeq/Cpu（AudioBlockからdsp->process()直前に設定）
    std::atomic<uint64_t> currentCallbackSeq { 0 };
    std::atomic<uint32_t> currentCpu { UINT32_MAX };

    convo::ScopedDftiDescriptor fftHandle;
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

    // ★ M-05: レイテンシクロスフェード用ゲインランプバッファ (wetBuf[0]流用の解消)
    convo::ScopedAlignedPtr<double> delayFadeRampBuffer;
    int delayFadeRampCapacity = 0;

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
        float f1, f2;
        int targetLength;

        bool operator<(const IRCacheKey& other) const {
            if (fileHash != other.fileHash) return fileHash < other.fileHash;
            if (sampleRate != other.sampleRate) return sampleRate < other.sampleRate;
            if (phaseMode != other.phaseMode) return phaseMode < other.phaseMode;
            if (f1 != other.f1) return f1 < other.f1;
            if (f2 != other.f2) return f2 < other.f2;
            return targetLength < other.targetLength;
        }
    };



    struct CacheEntry {
        std::unique_ptr<juce::AudioBuffer<double>> ir;
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

    convo::aligned_unique_ptr<IRConverter> irConverter;
    convo::aligned_unique_ptr<CacheManager> cacheManager;
    std::unique_ptr<ProgressiveUpgradeThread> upgradeThread;
    std::atomic<bool> writerActive { false };
    std::atomic<uint64_t> activeCacheKey { 0 };
    std::atomic<int> activeCacheFFTSize { 0 };

    // RCU 管理 (StereoConvolver 用)
    // 世代カウンター (init=1: 初回 prepare 後の最初の process() でブロックをクリアするため)
    std::atomic<uint64_t> firstProcessCallGen { 1 };
    // RT-local 世代追跡 (非atomic, RT スレッドのみアクセス)
    uint64_t m_firstProcessCallGenSeen       { 0 };
    uint64_t m_latencyResetPendingGenSeen    { 0 };
    uint64_t m_latencyChangeRequestedGenSeen { 0 };
    uint64_t m_smoothingTimeChangePendingGenSeen { 0 };
    uint64_t m_mixSmootherResetPendingGenSeen { 0 };
    // [P1-15] 内部 epoch 管理用 (Convolver 独自ドメイン)
    convo::EpochDomain m_epochDomain;
    // DSP_THREAD_STATE: Audio Thread process() で使用するRCU reader。
    convo::RCUReader runtimeRcuReader { m_epochDomain };

    void retireStereoConvolver(StereoConvolver* conv, uint64_t retireEpoch);

    // ── Phase 0: Epoch-based RCU メンバー ──
    std::atomic<convo::ConvolverState*> convolverState { nullptr };
    // DeferredFreeThread: 旧 ConvolverState を Audio Thread 外で安全に解放する専用スレッド
    // prepareToPlay() で生成、releaseResources() で停止・破棄する。
    convo::aligned_unique_ptr<convo::DeferredFreeThread> deferredFreeThread;
    // GenerationManager: IR ロードタスクの世代管理（陳腐化チェック用）
    GenerationManager convolverStateGeneration;

    JUCE_DECLARE_WEAK_REFERENCEABLE(ConvolverProcessor)
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ConvolverProcessor)

};
#pragma warning(pop) // C4324 suppression scope end: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
