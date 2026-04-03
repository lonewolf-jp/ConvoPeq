#pragma once

#include <JuceHeader.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>

#include "AudioSegmentBuffer.h"
#include "CmaEsOptimizer.h"
#include "LatticeNoiseShaper.h"
#include "MklFftEvaluator.h"

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
    enum class Status
    {
        Idle,
        WaitingForAudio,
        Running,
        Completed,
        Error
    };

    enum class LearningMode { Shortest, Short, Middle, Long, Ultra, Continuous };

    static constexpr int kOrder = LatticeNoiseShaper::kOrder;

    struct Progress
    {
        std::atomic<int> iteration { 0 };
        std::atomic<uint64_t> totalGenerations { 0 };
        std::atomic<int> processCount { 0 };
        std::atomic<int> segmentCount { 0 };
        std::atomic<double> bestScore { 0.0 };
        std::atomic<double> latestScore { 0.0 };
        std::atomic<Status> status { Status::Idle };
        std::atomic<double> elapsedPlaybackSeconds {0.0};  // UI表示用
        std::atomic<int>    currentPhase {1};
        std::atomic<int>    learningMode {0};
    };

    struct State
    {
        double mean[9] = {};
        double covarianceUpperTriangle[45] = {};
        double sigma = 0.12;
        double bestCoefficients[9] = {};
        double elapsedPlaybackSeconds = 0.0;
        int currentPhase = 1;
        int iteration = 0;
        double bestScore = 0.0;
        int processCount = 0;
        uint64_t totalGenerations = 0;
        int learningMode = 0;
    };

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
    struct Settings
    {
        std::atomic<int> cmaesRestarts { 5 };
        std::atomic<double> coeffSafetyMargin { 0.85 };
        std::atomic<bool> enableStabilityCheck { true };

        // デフォルトコンストラクタ
        Settings() = default;

        // コピーコンストラクタ（アトミック値をロード）
        Settings(const Settings& other)
            : cmaesRestarts(other.cmaesRestarts.load()),
              coeffSafetyMargin(other.coeffSafetyMargin.load()),
              enableStabilityCheck(other.enableStabilityCheck.load()) {}

        // コピー代入演算子（アトミックにストア）
        Settings& operator=(const Settings& other)
        {
            cmaesRestarts = other.cmaesRestarts.load();
            coeffSafetyMargin = other.coeffSafetyMargin.load();
            enableStabilityCheck = other.enableStabilityCheck.load();
            return *this;
        }
    };

    void setSettings(const Settings& newSettings) noexcept
    {
        settings = newSettings;
    }

    Settings getSettings() const noexcept
    {
        Settings s;
        s.cmaesRestarts = settings.cmaesRestarts.load();
        s.coeffSafetyMargin = settings.coeffSafetyMargin.load();
        s.enableStabilityCheck = settings.enableStabilityCheck.load();
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

    static constexpr std::array<double, kOrder> kDefaultCoeffs = {
        0.82, -0.68, 0.55, -0.43, 0.33, -0.25, 0.18, -0.12, 0.07
    };

    // tanh mapping scale to allow CMA-ES to explore a reasonable range
    // atanh(0.995) is approx 3.0, so a range of [-4, 4] in CMA-ES space is good.
    static constexpr double kCmaEsInitialSigma = 0.15;
    static constexpr double kCmaEsCoordinateScale = 1.0;

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
    int copyBestScoreHistory(double* destination, int maxPoints) const noexcept;
    void onCoeffBankChanged(int newBankIndex) noexcept;

    bool saveLearnedState(const juce::File& file) const;
    bool loadLearnedState(const juce::File& file);

    // UI 表示用：学習ワーカーが記録したエラーメッセージを返す。
    // エラーがなければ nullptr を返す。
    const char* getErrorMessage() const noexcept
    {
        return errorMessage.load(std::memory_order_acquire);
    }

private:
    struct SessionSignature
    {
        int sampleRateHz = 0;
        int bitDepth = 0;
        int adaptiveCoeffBankIndex = 0;
        uint64_t sessionId = 0;
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
        std::thread thread;
    };

    void workerThreadMain();
    void startEvaluationWorkers();
    void stopEvaluationWorkers() noexcept;
    void configureEvaluationContexts(double sampleRateHz) noexcept;
    void evaluationWorkerMain(int workerIndex) noexcept;
    void runEvaluationJobsForWorker(int workerIndex, int numSegments, int evaluationBitDepth) noexcept;
    int evaluatePopulation(int numSegments, int evaluationBitDepth, int& bestCandidateIndex, double& bestCandidateScore);
    SessionSignature captureSessionSignature() const noexcept;
    void resetLearningSession(const SessionSignature& session, bool resume) noexcept;
    void drainCaptureQueue(const SessionSignature& session) noexcept;
    int buildTrainingSegments() noexcept;
    double evaluateCandidate(EvaluationContext& context,
                             const double* candidateCoefficients,
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

    std::thread workerThread;
    std::atomic<bool> stopRequested { false };
    std::atomic<bool> pendingResume { false };
    std::atomic<bool> startRequested { false };

    // workerThread が完全に終了したことを示すフラグ。
    // workerThreadMain() の最後（stopEvaluationWorkers() 完了後）で true にセットされる。
    // startLearning() が join() でメッセージスレッドをブロックしないために使用する。
    std::atomic<bool> workerThreadFinished { true };

    // startLearning() の callAsync リトライが多重発行されないようにするためのフラグ。
    // compare_exchange_strong で排他制御する。
    std::atomic<bool> pendingRestart { false };

    Progress progress;
    Settings settings;
    std::atomic<const char*> errorMessage { nullptr };

    double accumulatedPlaybackSeconds = 0.0;           // Worker thread専用（非atomic）
    double sessionSampleRate = 0.0;
    int sessionBitDepth = 0;
    uint64_t currentSessionId = 0;
    std::chrono::steady_clock::time_point lastGenerationStart;
    double generationIntervalSeconds = 0.0;
    mutable std::chrono::steady_clock::time_point lastSaveTime;
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
    int pendingEvaluationSegmentCount = 0;
    int pendingEvaluationBitDepth = 24;
    std::atomic<int> completedAuxEvaluationWorkers{0};
    std::atomic<uint32_t> evaluationDispatchSerial{0};
    bool evaluationWorkersShouldExit = false;
    std::atomic<int> nextEvaluationCandidateIndex { 0 };
    double candidatePopulation[CmaEsOptimizer::kPopulation][CmaEsOptimizer::kDim] = {};
    double candidateFitness[CmaEsOptimizer::kPopulation] = {};

    std::array<LeveledSegment, kMaxSegmentsPerLevel> levelBuckets[kNumLevels] = {};
    int levelBucketCounts[kNumLevels] = {};

    std::array<std::atomic<double>, kOrder> bestCoefficients {};
    std::array<double, kMaxHistoryPoints> bestScoreHistory {};
    std::atomic<int> historyCount { 0 };
    // bestScoreHistory リングバッファの書き込み先インデックス（historyMutex 保護下で使用）
    int historyHead { 0 };
    mutable std::mutex historyMutex;

    static juce::ThreadPool saveThreadPool;  // 非同期保存用スレッドプール

    // callAsync ラムダが this の生存を安全に確認するための WeakReference サポート。
    // デストラクタで masterReference が破棄されると WeakReference::get() が nullptr を返すため、
    // 破棄済みオブジェクトへのアクセスを防げる。
    JUCE_DECLARE_WEAK_REFERENCEABLE(NoiseShaperLearner)
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(NoiseShaperLearner)
};
