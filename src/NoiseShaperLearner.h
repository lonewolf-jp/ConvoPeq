#pragma once

#include <JuceHeader.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>

#include "AlignedAllocation.h"
#include "AudioSegmentBuffer.h"
#include "CmaEsOptimizer.h"
#include "LatticeNoiseShaper.h"
#include "MklFftEvaluator.h"
#include "NoiseShaperLearnerTypes.h"

#include "audioengine/AtomicAccess.h"
#include "core/RCUReader.h"

class AudioEngine;
struct AudioBlock;
template <typename T, size_t Capacity>
class LockFreeRingBuffer;

struct AudioSegment
{
    static constexpr int kLength = MklFftEvaluator::kFftLength;
    double left[kLength] = {};
    double right[kLength] = {};
    std::array<double, MklFftEvaluator::kSpectrumBins> maskingThresholds {};
};

enum class SpectralType {
    Broadband,
    Tonal,
    Transient
};

struct LeveledSegment
{
    AudioSegment segment;
    double targetRMS = 0.0;
    double appliedGain = 1.0;
    SpectralType type = SpectralType::Broadband;
};

class NoiseShaperLearner
{
public:
    using Status = convo::NoiseShaperLearnerStatus;
    using LearningMode = convo::NoiseShaperLearningMode;
    using Progress = convo::NoiseShaperLearnerProgress;
    using State = convo::NoiseShaperLearnerState;
    using Settings = convo::NoiseShaperLearnerSettings;

    static constexpr int kOrder = LatticeNoiseShaper::kOrder;

    struct LearnedState
    {
        std::array<double, kOrder> bestCoefficients;
        std::vector<double> cmaMean;
        double bestScore = 0.0;
        double sampleRate = 0.0;
        int bitDepth = 0;
        int currentPhase = 1;
        double elapsedPlaybackSeconds = 0.0;
    };

    // ============================================================================
    // 追加: 学習設定（Multi-start / 安全マージン / 極配置チェック）
    // ============================================================================
    void setSettings(const Settings& newSettings) noexcept
    {
        settings = newSettings;
    }

    Settings getSettings() const noexcept
    {
        Settings s;
        s.cmaesRestarts = convo::consumeAtomic(settings.cmaesRestarts, std::memory_order_acquire);
        s.coeffSafetyMargin = convo::consumeAtomic(settings.coeffSafetyMargin, std::memory_order_acquire);
        s.enableStabilityCheck = convo::consumeAtomic(settings.enableStabilityCheck, std::memory_order_acquire);
        return s;
    }

    static constexpr int kMaxHistoryPoints = 256;
    static constexpr int kMaxParallelEvaluators = 6;

    // Multi-level normalization constants
    static constexpr int kNumLevels = 4;
    static constexpr double kTargetLevelsDB[kNumLevels] = { -40.0, -30.0, -20.0, -10.0 };
    static constexpr double kMinRMS = 1.0e-5; // -100dB
    static constexpr double kPeakHeadroom = 0.95;
    static constexpr int kMaxSegmentsPerLevel = 4;
    static constexpr int kMaxTrainingSegments = kNumLevels * kMaxSegmentsPerLevel;

    std::array<double, kNumLevels> currentLevelWeights = { 0.4, 0.3, 0.2, 0.1 };

    NoiseShaperLearner(AudioEngine& engineRef,
                       LockFreeRingBuffer<AudioBlock, 4096>& captureQueueRef);
    ~NoiseShaperLearner();

    void startLearning(bool resume = false);
    void stopLearning();
    bool isRunning() const noexcept;
    void setLearningMode(LearningMode mode) noexcept;

    const Progress& getProgress() const noexcept;
    void getState(State& outState) const noexcept;
    void setState(const State& inState) noexcept;
    int copyBestScoreHistory(double* destination, int maxPoints) noexcept;
    void onCoeffBankChanged(int newBankIndex) noexcept;

    double (*candidatePopulationMatrix() noexcept)[CmaEsOptimizer::kDim]
    {
        return reinterpret_cast<double (*)[CmaEsOptimizer::kDim]>(candidatePopulation.get());
    }

    const double (*candidatePopulationMatrix() const noexcept)[CmaEsOptimizer::kDim]
    {
        return reinterpret_cast<const double (*)[CmaEsOptimizer::kDim]>(candidatePopulation.get());
    }

    double* candidateFitnessData() noexcept
    {
        return candidateFitness.get();
    }

    const double* candidateFitnessData() const noexcept
    {
        return candidateFitness.get();
    }

    bool saveLearnedState(const juce::File& file) const;
    bool loadLearnedState(const juce::File& file);

    // UI 表示用：学習ワーカーが記録したエラーメッセージを返す。
    // エラーがなければ nullptr を返す。
    const char* getErrorMessage() const noexcept
    {
        return convo::consumeAtomic(errorMessage, std::memory_order_acquire);
    }

    void setErrorMessage(const char* msg) noexcept
    {
        convo::publishAtomic(errorMessage, msg, std::memory_order_release);
    }

private:
    enum class WorkerState : uint8_t
    {
        Idle = 0,
        Starting,
        Running,
        Stopping
    };

    struct SessionSignature
    {
        int sampleRateHz = 0;
        int bitDepth = 0;
        int adaptiveCoeffBankIndex = 0;
        uint64_t sessionId = 0;
    };

    // ★ v8.3: NoiseShaper ドロップ理由の集計用 enum
    enum class DropReason {
        SampleRate,   // block.sampleRateHz と session.sampleRateHz の不一致
        Session,      // sessionId の不一致
        Reserved,     // 将来拡張用（旧 QueueFull）
        Disabled,     // NoiseShaper learner が無効状態
        Unknown,      // 上記以外
        Count         // 列挙子の個数（std::array サイズ指定用、常に最後）
    };

    struct DrainStats
    {
        int acceptedBlocks = 0;
        int droppedBySession = 0;
        int droppedBySampleRate = 0;
        int droppedByBank = 0;
        // ★ v8.3: DropReason ごとの詳細カウンタ（サマリと内訳の二重管理）
        std::array<int, static_cast<size_t>(DropReason::Count)> dropReasonCounts{};
    };

    struct EvaluationContext
    {
        MklFftEvaluator fftEvaluator;
        LatticeNoiseShaper shaper;
        double shapedLeft[AudioSegment::kLength] = {};
        double shapedRight[AudioSegment::kLength] = {};
        double errorLeft[AudioSegment::kLength] = {};
        double errorRight[AudioSegment::kLength] = {};
    };

    struct EvaluationWorkerSlot
    {
        EvaluationContext context;
        std::jthread thread;
    };

    void workerThreadMain(std::stop_token stopToken);
    void startEvaluationWorkers();
    void stopEvaluationWorkers() noexcept;
    void configureEvaluationContexts(double sampleRateHz) noexcept;
    void evaluationWorkerMain(int workerIndex, std::stop_token stopToken) noexcept;
    void runEvaluationJobsForWorker(int workerIndex,
                                    int numSegments,
                                    int evaluationBitDepth,
                                    const std::stop_token* stopToken = nullptr) noexcept;
    int evaluatePopulation(int numSegments,
                           int evaluationBitDepth,
                           int& bestCandidateIndex,
                           double& bestCandidateScore,
                           const std::stop_token& stopToken);
    SessionSignature captureSessionSignature() noexcept;
    void resetLearningSession(const SessionSignature& session, bool resume) noexcept;
    DrainStats drainCaptureQueue(const SessionSignature& session) noexcept;
    int buildTrainingSegments() noexcept;
    double evaluateCandidate(EvaluationContext& context,
                             const double* candidateCoefficients,
                             int numSegments,
                             int evaluationBitDepth) noexcept;
    double evaluateCandidateMapped(EvaluationContext& context,
                                   const double* mappedCoefficients,
                                   int numSegments,
                                   int evaluationBitDepth) noexcept;
    void precomputeMaskingThresholds(LeveledSegment& seg, double sampleRate) noexcept;
    void publishGenerationResult(const double* coeffs, double score, int evaluatedCandidates) noexcept;
    void appendHistoryPoint(double score) noexcept;

    int computePhase(LearningMode mode, double playbackSeconds) const noexcept;
    void applyPhaseParams(LearningMode mode, int phase) noexcept;
    void handleModeSwitch() noexcept;

    AudioEngine& engine;
    LockFreeRingBuffer<AudioBlock, 4096>& captureQueue;
    convo::RCUReader rcuReader;

    std::jthread workerThread;
    std::atomic<WorkerState> workerState { WorkerState::Idle };
    std::atomic<bool> stopRequested { false };
    std::atomic<bool> pendingResume { false };

    Progress progress;
    Settings settings;
    std::atomic<const char*> errorMessage { nullptr };

    double accumulatedPlaybackSeconds = 0.0;           // Worker thread専用（非atomic）
    double sessionSampleRate = 0.0;
    int sessionBitDepth = 0;
    uint64_t currentSessionId = 0;
    std::chrono::steady_clock::time_point lastGenerationStart;
    double generationIntervalSeconds = 0.0;
    std::chrono::steady_clock::time_point lastSaveTime;
    static constexpr auto kSaveInterval = std::chrono::seconds(5);
    std::atomic<bool> modeSwitchRequested {false};
    LearningMode pendingMode {LearningMode::Short};
    LearningMode activeMode {LearningMode::Short};
    int currentPhase = 1;

    std::array<State, 6> savedStates {};

    AudioSegmentBuffer segmentBuffer;
    CmaEsOptimizer optimizer;
    std::array<EvaluationWorkerSlot, kMaxParallelEvaluators> evaluationWorkers {};
    int activeEvaluationWorkerCount = 1;
    int activeAuxEvaluationWorkerCount = 0;
    std::mutex evaluationDispatchMutex;
    std::condition_variable evaluationDispatchCv;
    std::mutex intervalMutex_;
    std::condition_variable intervalCv_;
    int pendingEvaluationSegmentCount = 0;
    int pendingEvaluationBitDepth = 24;
    std::atomic<int> completedAuxEvaluationWorkers{0};
    std::atomic<uint32_t> evaluationDispatchSerial{0};
    bool evaluationWorkersShouldExit = false;
    std::atomic<int> nextEvaluationCandidateIndex { 0 };
    convo::ScopedAlignedPtr<double> candidatePopulation;
    convo::ScopedAlignedPtr<double> candidateFitness;

    // ★ B03: Generation 単位で共有する vdTanh 結果 (64バイトアライメント)
    convo::ScopedAlignedPtr<double> sharedMappedPopulation;

    std::array<LeveledSegment, kMaxSegmentsPerLevel> levelBuckets[kNumLevels] = {};
    int levelBucketCounts[kNumLevels] = {};

    std::array<std::atomic<double>, kOrder> bestCoefficients {};
    std::array<double, kMaxHistoryPoints> bestScoreHistory {};
    std::atomic<int> historyCount { 0 };
    // bestScoreHistory リングバッファの書き込み先インデックス（historyMutex 保護下で使用）
    int historyHead { 0 };
    std::mutex historyMutex;

    // callAsync ラムダが this の生存を安全に確認するための WeakReference サポート。
    // デストラクタで masterReference が破棄されると WeakReference::get() が nullptr を返すため、
    // 破棄済みオブジェクトへのアクセスを防げる。
    JUCE_DECLARE_WEAK_REFERENCEABLE(NoiseShaperLearner)
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(NoiseShaperLearner)
};
