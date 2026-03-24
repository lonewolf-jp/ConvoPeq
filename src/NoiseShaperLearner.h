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

    struct Progress
    {
        std::atomic<int> iteration { 0 };
        std::atomic<uint64_t> totalGenerations { 0 };
        std::atomic<int> processCount { 0 };
        std::atomic<int> segmentCount { 0 };
        std::atomic<float> bestScore { 0.0f };
        std::atomic<float> latestScore { 0.0f };
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
        float bestScore = 0.0f;
        int processCount = 0;
        uint64_t totalGenerations = 0;
    };

    static constexpr int kOrder = LatticeNoiseShaper::kOrder;
    static constexpr int kMaxTrainingSegments = 8;
    static constexpr int kMaxHistoryPoints = 256;
    static constexpr int kMaxParallelEvaluators = 6;

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
    int copyBestScoreHistory(float* destination, int maxPoints) const noexcept;
    void onCoeffBankChanged(int newBankIndex) noexcept;

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
    void publishGenerationResult(const double* coeffs, double score, int evaluatedCandidates) noexcept;
    void appendHistoryPoint(float score) noexcept;

    int computePhase(LearningMode mode, double playbackSeconds) const noexcept;
    void applyPhaseParams(LearningMode mode, int phase) noexcept;
    void handleModeSwitch() noexcept;

    AudioEngine& engine;
    LockFreeRingBuffer<AudioBlock, 4096>& captureQueue;

    std::thread workerThread;
    std::atomic<bool> stopRequested { false };
    std::atomic<bool> pendingResume { false };

    // workerThread が完全に終了したことを示すフラグ。
    // workerThreadMain() の最後（stopEvaluationWorkers() 完了後）で true にセットされる。
    // startLearning() が join() でメッセージスレッドをブロックしないために使用する。
    std::atomic<bool> workerThreadFinished { true };

    // startLearning() の callAsync リトライが多重発行されないようにするためのフラグ。
    // compare_exchange_strong で排他制御する。
    std::atomic<bool> pendingRestart { false };

    Progress progress;
    std::atomic<const char*> errorMessage { nullptr };

    double accumulatedPlaybackSeconds = 0.0;           // Worker thread専用（非atomic）
    std::chrono::steady_clock::time_point lastGenerationStart;
    double generationIntervalSeconds = 0.0;
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

    AudioSegment trainingSegments[kMaxTrainingSegments] = {};
    std::array<double, kOrder> bestCoefficients {};
    std::array<float, kMaxHistoryPoints> bestScoreHistory {};
    std::atomic<int> historyCount { 0 };
    // bestScoreHistory リングバッファの書き込み先インデックス（historyMutex 保護下で使用）
    int historyHead { 0 };
    mutable std::mutex historyMutex;

    // callAsync ラムダが this の生存を安全に確認するための WeakReference サポート。
    // デストラクタで masterReference が破棄されると WeakReference::get() が nullptr を返すため、
    // 破棄済みオブジェクトへのアクセスを防げる。
    JUCE_DECLARE_WEAK_REFERENCEABLE(NoiseShaperLearner)
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(NoiseShaperLearner)
};
