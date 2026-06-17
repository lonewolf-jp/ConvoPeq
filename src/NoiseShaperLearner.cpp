#include "NoiseShaperLearner.h"
#include "AudioEngine.h"
#include "core/ThreadAffinityManager.h"
#include <JuceHeader.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>
#include <mkl_vml.h>
#include <xmmintrin.h>  // _MM_SET_FLUSH_ZERO_MODE
#include <pmmintrin.h>  // _MM_SET_DENORMALS_ZERO_MODE

namespace
{
    constexpr double kOutputHeadroom = 0.8912509381337456;
    constexpr int kSegmentHop = AudioSegment::kLength / 2;
    constexpr int kRecentSampleRequest = AudioSegment::kLength + (kSegmentHop * (NoiseShaperLearner::kMaxTrainingSegments - 1));
    juce::ThreadPool g_saveThreadPool(1);

    uint64_t hashLearningSeed(uint64_t seed, uint64_t value) noexcept
    {
        seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
        return seed;
    }

    uint64_t makeDeterministicRestartSeed(int sampleRateHz,
                                          int bitDepth,
                                          int bankIndex,
                                          uint64_t sessionId,
                                          int restartIndex) noexcept
    {
        uint64_t seed = 0x4e4f495345534850ull;
        seed = hashLearningSeed(seed, static_cast<uint64_t>(sampleRateHz));
        seed = hashLearningSeed(seed, static_cast<uint64_t>(bitDepth));
        seed = hashLearningSeed(seed, static_cast<uint64_t>(bankIndex));
        seed = hashLearningSeed(seed, sessionId);
        seed = hashLearningSeed(seed, static_cast<uint64_t>(restartIndex));
        return seed;
    }
}

NoiseShaperLearner::NoiseShaperLearner(AudioEngine& engineRef,
                                       LockFreeRingBuffer<AudioBlock, 4096>& captureQueueRef)
    : engine(engineRef),
    // lastSaveTime の初期化（コンストラクタ本体で行う）
      captureQueue(captureQueueRef),
      rcuReader(engineRef.getRetireRouter())
{
    const size_t populationCount = static_cast<size_t>(CmaEsOptimizer::kPopulation * CmaEsOptimizer::kDim);
    const size_t fitnessCount = static_cast<size_t>(CmaEsOptimizer::kPopulation);

    auto populationBuffer = convo::makeAlignedArray<double>(populationCount);
    auto fitnessBuffer = convo::makeAlignedArray<double>(fitnessCount);
    jassert(populationBuffer.get() != nullptr && fitnessBuffer.get() != nullptr);
    if (populationBuffer.get() == nullptr || fitnessBuffer.get() == nullptr)
    {
        DBG_LOG("[NoiseShaperLearner] failed to allocate candidate buffers");
        convo::publishAtomic(errorMessage, "Failed to allocate candidate buffers", std::memory_order_release);
        convo::publishAtomic(progress.status, Status::Error, std::memory_order_release);
        return;
    }
    candidatePopulation = std::move(populationBuffer);
    candidateFitness = std::move(fitnessBuffer);

    for (auto& c : bestCoefficients)
        convo::publishAtomic(c, 0.0, std::memory_order_release);

    const unsigned int hardwareThreadCount = std::thread::hardware_concurrency();
    lastSaveTime = std::chrono::steady_clock::now();
    const int usableWorkerCount = hardwareThreadCount > 1
        ? static_cast<int>(hardwareThreadCount) - 1
        : 1;
    activeEvaluationWorkerCount = std::max(1, std::min(usableWorkerCount, kMaxParallelEvaluators));
    activeAuxEvaluationWorkerCount = std::max(0, activeEvaluationWorkerCount - 1);
}

NoiseShaperLearner::~NoiseShaperLearner()
{
    // Transition:
    // Running/Starting -> Stopping
    convo::publishAtomic(workerState, WorkerState::Stopping, std::memory_order_release);
    convo::publishAtomic(stopRequested, true, std::memory_order_release);
    {
        const std::scoped_lock<std::mutex> lock(evaluationDispatchMutex);
        evaluationWorkersShouldExit = true;
    }
    evaluationDispatchCv.notify_all();
    intervalCv_.notify_all();

    if (workerThread.joinable())
        workerThread.request_stop();

    // Transition:
    // Stopping -> Idle
    convo::publishAtomic(workerState, WorkerState::Idle, std::memory_order_release);
}

void NoiseShaperLearner::startLearning(bool resume)
{
    juce::Logger::writeToLog("[NoiseShaperLearner] startLearning enter resume=" + juce::String(static_cast<int>(resume)));

    if (candidatePopulationMatrix() == nullptr || candidateFitnessData() == nullptr)
    {
        juce::Logger::writeToLog("[NoiseShaperLearner] startLearning aborted: candidate buffers unavailable"
            + juce::String(" pop=") + juce::String(candidatePopulationMatrix() == nullptr ? "null" : "ok")
            + juce::String(" fit=") + juce::String(candidateFitnessData() == nullptr ? "null" : "ok"));
        convo::publishAtomic(errorMessage, "Candidate buffers unavailable", std::memory_order_release);
        convo::publishAtomic(progress.status, Status::Error, std::memory_order_release);
        return;
    }

    // 前回セッション残骸のクリーンアップ
    if (isRunning() || workerThread.joinable() || convo::consumeAtomic(workerState, std::memory_order_acquire) != WorkerState::Idle)
    {
        juce::Logger::writeToLog("[NoiseShaperLearner] startLearning: cleaning up previous session");
        stopLearning();
        if (workerThread.joinable())
            workerThread.join();
    }

    // 全フラグを明示リセット
    convo::publishAtomic(stopRequested, false, std::memory_order_release);
    convo::publishAtomic(errorMessage, nullptr, std::memory_order_release);

    WorkerState expectedState = WorkerState::Idle;
    // Transition:
    // Idle -> Starting
    if (!convo::compareExchangeAtomic(workerState,
                                      expectedState,
                                      WorkerState::Starting,
                                      std::memory_order_acq_rel,
                                      std::memory_order_acquire))
    {
        return;
    }

    convo::publishAtomic(pendingResume, resume, std::memory_order_release);

    convo::publishAtomic(progress.status, Status::WaitingForAudio, std::memory_order_release);
    convo::publishAtomic(progress.iteration, 0, std::memory_order_release);
    convo::publishAtomic(progress.segmentCount, 0, std::memory_order_release);
    convo::publishAtomic(progress.bestScore, 0.0, std::memory_order_release);
    convo::publishAtomic(progress.latestScore, 0.0, std::memory_order_release);
    convo::publishAtomic(progress.elapsedPlaybackSeconds, 0.0, std::memory_order_release);
    convo::publishAtomic(progress.currentPhase, 1, std::memory_order_release);
    accumulatedPlaybackSeconds = 0.0;
    currentPhase = 1;
    segmentBuffer.clear();
    convo::publishAtomic(historyCount, 0, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(historyMutex);
        bestScoreHistory.fill(0.0);
        historyHead = 0;
    }

    activeMode = pendingMode;
    convo::publishAtomic(progress.learningMode, static_cast<int>(activeMode), std::memory_order_release);
    lastGenerationStart = std::chrono::steady_clock::time_point{};
    applyPhaseParams(activeMode, currentPhase);

    try
    {
        juce::Logger::writeToLog("[NoiseShaperLearner] startLearning: spawning worker thread");
        workerThread = std::jthread([this](std::stop_token stopToken)
        {
            juce::Logger::writeToLog("[NoiseShaperLearner] worker thread started");
            workerThreadMain(stopToken);
            juce::Logger::writeToLog("[NoiseShaperLearner] worker thread exited");
        });
    }
    catch (const std::exception& e)
    {
        juce::Logger::writeToLog(juce::String("[NoiseShaperLearner] failed to start worker thread: ") + juce::String(e.what()));
        convo::publishAtomic(errorMessage, "Failed to start learning thread", std::memory_order_release);
        convo::publishAtomic(progress.status, Status::Error, std::memory_order_release);
        convo::publishAtomic(workerState, WorkerState::Idle, std::memory_order_release);
    }
    catch (...)
    {
        juce::Logger::writeToLog("[NoiseShaperLearner] failed to start worker thread: unknown error");
        convo::publishAtomic(errorMessage, "Unknown error starting worker thread", std::memory_order_release);
        convo::publishAtomic(progress.status, Status::Error, std::memory_order_release);
        convo::publishAtomic(workerState, WorkerState::Idle, std::memory_order_release);
    }
}

void NoiseShaperLearner::stopLearning()
{
    // Transition:
    // Running/Starting -> Stopping
    convo::publishAtomic(workerState, WorkerState::Stopping, std::memory_order_release);
    convo::publishAtomic(stopRequested, true, std::memory_order_release);
    intervalCv_.notify_all();
    stopEvaluationWorkers();
    if (workerThread.joinable())
        workerThread.request_stop();

    // Transition:
    // Stopping -> Idle
    convo::publishAtomic(workerState, WorkerState::Idle, std::memory_order_release);
    convo::publishAtomic(pendingResume, false, std::memory_order_release);
    convo::publishAtomic(progress.status, Status::Idle, std::memory_order_release);
    convo::publishAtomic(errorMessage, nullptr, std::memory_order_release);

    evaluationDispatchCv.notify_all();
}

void NoiseShaperLearner::setLearningMode(LearningMode mode) noexcept
{
    pendingMode = mode;
    convo::publishAtomic(modeSwitchRequested, true, std::memory_order_release);

    if (!isRunning())
    {
        activeMode = mode;
        convo::publishAtomic(progress.learningMode, static_cast<int>(activeMode), std::memory_order_release);
    }
}

int NoiseShaperLearner::computePhase(LearningMode mode, double playbackSeconds) const noexcept
{
    // Phase 1: 初期探索, Phase 2: 収束, Phase 3: 微調整
    switch (mode)
    {
        case LearningMode::Shortest:
            if (playbackSeconds < 5.0) return 1;
            if (playbackSeconds < 10.0) return 2;
            return 3;
        case LearningMode::Short:
            if (playbackSeconds < 10.0) return 1;
            if (playbackSeconds < 20.0) return 2;
            return 3;
        case LearningMode::Middle:
            if (playbackSeconds < 30.0) return 1;
            if (playbackSeconds < 60.0) return 2;
            return 3;
        case LearningMode::Long:
            if (playbackSeconds < 60.0) return 1;
            if (playbackSeconds < 120.0) return 2;
            return 3;
        case LearningMode::Ultra:
            if (playbackSeconds < 120.0) return 1;
            if (playbackSeconds < 240.0) return 2;
            return 3;
        case LearningMode::Continuous:
            if (playbackSeconds < 30.0) return 1;
            if (playbackSeconds < 60.0) return 2;
            return 3;
    }
    return 1;
}

void NoiseShaperLearner::applyPhaseParams(LearningMode mode, int phase) noexcept
{
    CmaEsOptimizer::Params optParams;

    // Phase params based on mode and phase
    switch (mode)
    {
        case LearningMode::Shortest:
            generationIntervalSeconds = (phase == 1) ? 0.25 : ((phase == 2) ? 0.5 : 1.0);
            optParams.covRetentionTarget = (phase == 1) ? 0.80 : ((phase == 2) ? 0.85 : 0.90);
            optParams.covRetentionStep = 0.02;
            break;
        case LearningMode::Short:
            generationIntervalSeconds = (phase == 1) ? 0.5 : ((phase == 2) ? 1.0 : 2.0);
            optParams.covRetentionTarget = (phase == 1) ? 0.85 : ((phase == 2) ? 0.90 : 0.95);
            optParams.covRetentionStep = 0.01;
            break;
        case LearningMode::Middle:
            generationIntervalSeconds = (phase == 1) ? 1.0 : ((phase == 2) ? 2.0 : 4.0);
            optParams.covRetentionTarget = (phase == 1) ? 0.90 : ((phase == 2) ? 0.95 : 0.98);
            optParams.covRetentionStep = 0.005;
            break;
        case LearningMode::Long:
            generationIntervalSeconds = (phase == 1) ? 2.0 : ((phase == 2) ? 4.0 : 8.0);
            optParams.covRetentionTarget = (phase == 1) ? 0.95 : ((phase == 2) ? 0.98 : 0.99);
            optParams.covRetentionStep = 0.002;
            break;
        case LearningMode::Ultra:
            generationIntervalSeconds = (phase == 1) ? 4.0 : ((phase == 2) ? 8.0 : 16.0);
            optParams.covRetentionTarget = (phase == 1) ? 0.98 : ((phase == 2) ? 0.99 : 0.995);
            optParams.covRetentionStep = 0.001;
            break;
        case LearningMode::Continuous:
            generationIntervalSeconds = (phase == 1) ? 1.0 : ((phase == 2) ? 2.0 : 4.0);
            optParams.covRetentionTarget = (phase == 1) ? 0.90 : ((phase == 2) ? 0.95 : 0.98);
            optParams.covRetentionStep = 0.005;
            break;
    }

    // Adjust level weights based on phase
    if (phase == 1)
    {
        // Phase 1: Focus on high-level signals for stability
        currentLevelWeights = { 0.1, 0.2, 0.3, 0.4 };
    }
    else if (phase == 2)
    {
        // Phase 2: Balanced evaluation
        currentLevelWeights = { 0.25, 0.25, 0.25, 0.25 };
    }
    else
    {
        // Phase 3: Focus on low-level signals for idle tone detection
        currentLevelWeights = { 0.5, 0.3, 0.1, 0.1 };
    }

    optimizer.setParams(optParams);
}

void NoiseShaperLearner::handleModeSwitch() noexcept
{
    if (!convo::consumeAtomic(modeSwitchRequested, std::memory_order_acquire))
        return;

    // Save current state to AudioEngine
    State currentState;
    getState(currentState);
    const double sr = engine.getSampleRate();
    const int bd = engine.getDitherBitDepth();
    int bankIndex = AudioEngine::getAdaptiveCoeffBankIndex(sr, bd, activeMode);
    engine.setAdaptiveNoiseShaperState(bankIndex, currentState);

    activeMode = pendingMode;
    convo::publishAtomic(progress.learningMode, static_cast<int>(activeMode), std::memory_order_release);

    // Load new state from AudioEngine
    bankIndex = AudioEngine::getAdaptiveCoeffBankIndex(sr, bd, activeMode);
    State newState;
    if (engine.getAdaptiveNoiseShaperState(bankIndex, newState) && newState.iteration > 0)
    {
        setState(newState);
    }
    else
    {
        // Initialize if no saved state
        double initialCoefficients[kOrder] = {};
        engine.getAdaptiveCoefficientsForSampleRateAndBitDepth(sr, bd, initialCoefficients, kOrder);
        optimizer.initFromParcor(initialCoefficients);

        for (int i = 0; i < kOrder; ++i)
            convo::publishAtomic(bestCoefficients[static_cast<size_t>(i)], initialCoefficients[i], std::memory_order_release);

        accumulatedPlaybackSeconds = 0.0;
        convo::publishAtomic(progress.iteration, 0, std::memory_order_release);
        convo::publishAtomic(progress.bestScore, 0.0, std::memory_order_release);
        convo::publishAtomic(progress.latestScore, 0.0, std::memory_order_release);
        convo::publishAtomic(progress.processCount, 0, std::memory_order_release);
        convo::publishAtomic(progress.totalGenerations, 0, std::memory_order_release);
    }

    // 現在の再生時間に基づいてフェーズを再計算し、パラメータを適用
    currentPhase = computePhase(activeMode, accumulatedPlaybackSeconds);
    convo::publishAtomic(progress.currentPhase, currentPhase, std::memory_order_release);
    applyPhaseParams(activeMode, currentPhase);

    convo::publishAtomic(modeSwitchRequested, false, std::memory_order_release);
}

bool NoiseShaperLearner::isRunning() const noexcept
{
    return convo::consumeAtomic(progress.status, std::memory_order_acquire) == Status::Running
        || convo::consumeAtomic(progress.status, std::memory_order_acquire) == Status::WaitingForAudio;
}

const NoiseShaperLearner::Progress& NoiseShaperLearner::getProgress() const noexcept
{
    return progress;
}

void NoiseShaperLearner::getState(State& outState) const noexcept
{
    optimizer.serializeTo(outState.mean, outState.covarianceUpperTriangle, outState.sigma);
    for (int i = 0; i < kOrder; ++i)
        outState.bestCoefficients[i] = convo::consumeAtomic(bestCoefficients[i], std::memory_order_acquire);
    outState.elapsedPlaybackSeconds = accumulatedPlaybackSeconds;
    outState.currentPhase = currentPhase;
    outState.iteration = convo::consumeAtomic(progress.iteration, std::memory_order_acquire);
    outState.bestScore = convo::consumeAtomic(progress.bestScore, std::memory_order_acquire);
    outState.processCount = convo::consumeAtomic(progress.processCount, std::memory_order_acquire);
    outState.totalGenerations = convo::consumeAtomic(progress.totalGenerations, std::memory_order_acquire);
}

void NoiseShaperLearner::setState(const State& inState) noexcept
{
    optimizer.deserializeFrom(inState.mean, inState.covarianceUpperTriangle, inState.sigma);
    for (int i = 0; i < kOrder; ++i)
        convo::publishAtomic(bestCoefficients[i], inState.bestCoefficients[i], std::memory_order_release);
    accumulatedPlaybackSeconds = inState.elapsedPlaybackSeconds;
    currentPhase = inState.currentPhase;
    convo::publishAtomic(progress.iteration, inState.iteration, std::memory_order_release);
    convo::publishAtomic(progress.bestScore, inState.bestScore, std::memory_order_release);
    convo::publishAtomic(progress.elapsedPlaybackSeconds, accumulatedPlaybackSeconds, std::memory_order_release);
    convo::publishAtomic(progress.currentPhase, currentPhase, std::memory_order_release);
    convo::publishAtomic(progress.processCount, inState.processCount, std::memory_order_release);
    convo::publishAtomic(progress.totalGenerations, inState.totalGenerations, std::memory_order_release);
}

int NoiseShaperLearner::copyBestScoreHistory(double* destination, int maxPoints) noexcept
{
    if (destination == nullptr || maxPoints <= 0)
        return 0;

    const std::scoped_lock<std::mutex> lock(historyMutex);
    const int count = convo::consumeAtomic(historyCount, std::memory_order_acquire);
    const int available = std::min(count, maxPoints);
    if (available <= 0)
        return 0;

    if (count < kMaxHistoryPoints)
    {
        for (int i = 0; i < available; ++i)
            destination[i] = bestScoreHistory[static_cast<size_t>(i)];
    }
    else
    {
        for (int i = 0; i < available; ++i)
        {
            const int idx = (historyHead + i) % kMaxHistoryPoints;
            destination[i] = bestScoreHistory[static_cast<size_t>(idx)];
        }
    }

    return available;
}

void NoiseShaperLearner::onCoeffBankChanged(int newBankIndex) noexcept
{
    juce::ignoreUnused(newBankIndex);
    // The worker thread will detect the change via captureSessionSignature() and reset the session.
}

void NoiseShaperLearner::startEvaluationWorkers()
{
    if (activeAuxEvaluationWorkerCount <= 0)
        return;

    // 既存ワーカーを完全終了
    {
        const std::scoped_lock<std::mutex> lock(evaluationDispatchMutex);
        evaluationWorkersShouldExit = true;
    }
    evaluationDispatchCv.notify_all();

    for (int workerIndex = 1; workerIndex < activeEvaluationWorkerCount; ++workerIndex)
    {
        auto& slot = evaluationWorkers[static_cast<size_t>(workerIndex)];
        if (slot.thread.joinable())
            slot.thread.join();
    }

    {
        const std::scoped_lock<std::mutex> lock(evaluationDispatchMutex);
        evaluationWorkersShouldExit = false;
        convo::publishAtomic(completedAuxEvaluationWorkers, 0, std::memory_order_release);
        convo::publishAtomic(evaluationDispatchSerial, 0, std::memory_order_release);
    }

    for (int workerIndex = 1; workerIndex < activeEvaluationWorkerCount; ++workerIndex)
    {
        auto& slot = evaluationWorkers[static_cast<size_t>(workerIndex)];
        try
        {
            slot.thread = std::jthread([this, workerIndex](std::stop_token stopToken)
            {
                evaluationWorkerMain(workerIndex, stopToken);
            });
        }
        catch (const std::exception&)
        {
            DBG_LOG("[NoiseShaperLearner] failed to start evaluation worker");
            convo::publishAtomic(errorMessage, "Failed to start evaluation worker", std::memory_order_release);
            convo::publishAtomic(progress.status, Status::Error, std::memory_order_release);
            break;
        }
        catch (...)
        {
            convo::publishAtomic(errorMessage, "Unknown error starting evaluation worker", std::memory_order_release);
            convo::publishAtomic(progress.status, Status::Error, std::memory_order_release);
            break;
        }
    }
}

void NoiseShaperLearner::stopEvaluationWorkers() noexcept
{
    if (activeAuxEvaluationWorkerCount <= 0)
        return;

    {
        const std::scoped_lock<std::mutex> lock(evaluationDispatchMutex);
        evaluationWorkersShouldExit = true;
    }
    evaluationDispatchCv.notify_all();

    for (int workerIndex = 1; workerIndex < activeEvaluationWorkerCount; ++workerIndex)
    {
        auto& slot = evaluationWorkers[static_cast<size_t>(workerIndex)];
        if (slot.thread.joinable())
            slot.thread.join();
    }

    {
        const std::scoped_lock<std::mutex> lock(evaluationDispatchMutex);
        evaluationWorkersShouldExit = false;
        completedAuxEvaluationWorkers = 0;
        pendingEvaluationSegmentCount = 0;
        pendingEvaluationBitDepth = 24;
    }
}

void NoiseShaperLearner::configureEvaluationContexts(double sampleRateHz) noexcept
{
    for (int workerIndex = 0; workerIndex < activeEvaluationWorkerCount; ++workerIndex)
        evaluationWorkers[static_cast<size_t>(workerIndex)].context.fftEvaluator.configureForSampleRate(sampleRateHz);
}

void NoiseShaperLearner::evaluationWorkerMain(int workerIndex, std::stop_token stopToken) noexcept
{
    engine.getAffinityManager().applyCurrentThreadPolicy(ThreadType::LearnerEval, workerIndex);

    // FTZ/DAZ をスレッド開始時に設定する（スレッドローカルフラグ）。
    // 評価ワーカーは MKL DFTI および LatticeNoiseShaper を使用するため、
    // デノーマル数による CPU 速度低下を防ぐために必須。
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    uint32_t observedDispatchSerial = 0;

    for (;;)
    {
        try
        {
            int segmentCount = 0;
            int evaluationBitDepth = 24;

            {
                std::unique_lock<std::mutex> lock(evaluationDispatchMutex);
                evaluationDispatchCv.wait(lock, [&]
                {
                    return evaluationWorkersShouldExit
                        || stopToken.stop_requested()
                        || convo::consumeAtomic(evaluationDispatchSerial, std::memory_order_acquire) != observedDispatchSerial;
                });

                if (evaluationWorkersShouldExit || stopToken.stop_requested())
                    return;

                ++observedDispatchSerial;
                segmentCount = pendingEvaluationSegmentCount;
                evaluationBitDepth = pendingEvaluationBitDepth;
            }

            runEvaluationJobsForWorker(workerIndex,
                                       segmentCount,
                                       evaluationBitDepth,
                                       &stopToken);

            {
                const std::scoped_lock<std::mutex> lock(evaluationDispatchMutex);
                convo::fetchAddAtomic(completedAuxEvaluationWorkers, 1, std::memory_order_release);
            }
            evaluationDispatchCv.notify_all();
        }
        catch (const std::exception&)
        {
            DBG_LOG("[NoiseShaperLearner] evaluation worker exception");
            convo::publishAtomic(errorMessage, "Evaluation worker exception", std::memory_order_release);
            convo::publishAtomic(progress.status, Status::Error, std::memory_order_release);
            {
                const std::scoped_lock<std::mutex> lock(evaluationDispatchMutex);
                ++completedAuxEvaluationWorkers;
            }
            evaluationDispatchCv.notify_one();
            return;
        }
        catch (...)
        {
            convo::publishAtomic(errorMessage, "Error in evaluation worker", std::memory_order_release);
            convo::publishAtomic(progress.status, Status::Error, std::memory_order_release);
            {
                const std::scoped_lock<std::mutex> lock(evaluationDispatchMutex);
                ++completedAuxEvaluationWorkers;
            }
            evaluationDispatchCv.notify_one();
            return;
        }
    }
}

void NoiseShaperLearner::runEvaluationJobsForWorker(int workerIndex,
                                                    int numSegments,
                                                    int evaluationBitDepth,
                                                    const std::stop_token* stopToken) noexcept
{
    if (numSegments <= 0)
        return;

    auto& context = evaluationWorkers[static_cast<size_t>(workerIndex)].context;
    alignas(64) double mappedPopulation[CmaEsOptimizer::kPopulation][CmaEsOptimizer::kDim] = {};

    {
        constexpr int totalCoeffs = CmaEsOptimizer::kPopulation * CmaEsOptimizer::kDim;
        alignas(64) double tanhBuffer[totalCoeffs] = {};
        const auto* population = candidatePopulationMatrix();
        vdTanh(totalCoeffs,
               reinterpret_cast<const double*>(population),
               tanhBuffer);

        const double safetyMargin = convo::consumeAtomic(settings.coeffSafetyMargin, std::memory_order_acquire);
        for (int p = 0; p < CmaEsOptimizer::kPopulation; ++p)
            for (int d = 0; d < CmaEsOptimizer::kDim; ++d)
                mappedPopulation[p][d] = LatticeNoiseShaper::clampCoeff(
                    tanhBuffer[p * CmaEsOptimizer::kDim + d],
                    safetyMargin);
    }

    while (!convo::consumeAtomic(stopRequested, std::memory_order_acquire)
        && (stopToken == nullptr || !stopToken->stop_requested()))
    {
        const int populationIndex = convo::fetchAddAtomic(nextEvaluationCandidateIndex, 1, std::memory_order_acq_rel);
        if (populationIndex >= CmaEsOptimizer::kPopulation)
            break;

        const double score = evaluateCandidateMapped(context,
                                                     mappedPopulation[populationIndex],
                                                     numSegments,
                                                     evaluationBitDepth);
        candidateFitnessData()[populationIndex] = score;
        convo::fetchAddAtomic(progress.processCount, 1, std::memory_order_release);
    }
}

int NoiseShaperLearner::evaluatePopulation(int numSegments,
                                           int evaluationBitDepth,
                                           int& bestCandidateIndex,
                                           double& bestCandidateScore,
                                           const std::stop_token& stopToken)
{
    if (convo::consumeAtomic(stopRequested, std::memory_order_acquire)
        || stopToken.stop_requested())
        return 0;

    convo::publishAtomic(nextEvaluationCandidateIndex, 0, std::memory_order_release);

    {
        const std::scoped_lock<std::mutex> lock(evaluationDispatchMutex);
        pendingEvaluationSegmentCount = numSegments;
        pendingEvaluationBitDepth = evaluationBitDepth;
        convo::publishAtomic(completedAuxEvaluationWorkers, 0, std::memory_order_release);
        convo::fetchAddAtomic(evaluationDispatchSerial, static_cast<uint32_t>(1), std::memory_order_release);
    }

    if (activeAuxEvaluationWorkerCount > 0)
        evaluationDispatchCv.notify_all();

    runEvaluationJobsForWorker(0, numSegments, evaluationBitDepth, &stopToken);

    if (activeAuxEvaluationWorkerCount > 0)
    {
        std::unique_lock<std::mutex> lock(evaluationDispatchMutex);
        evaluationDispatchCv.wait(lock, [this]
        {
            return convo::consumeAtomic(completedAuxEvaluationWorkers, std::memory_order_acquire) >= activeAuxEvaluationWorkerCount
                || convo::consumeAtomic(stopRequested, std::memory_order_acquire);
        });
    }

    const int evaluatedCandidates = std::min(convo::consumeAtomic(nextEvaluationCandidateIndex, std::memory_order_acquire),
                                             CmaEsOptimizer::kPopulation);

    bestCandidateIndex = 0;
    bestCandidateScore = std::numeric_limits<double>::max();

    // First pass: find top candidates
    std::vector<int> sortedIndices(evaluatedCandidates);
    std::iota(sortedIndices.begin(), sortedIndices.end(), 0);
    std::sort(sortedIndices.begin(), sortedIndices.end(), [this](int a, int b) {
        return candidateFitnessData()[a] < candidateFitnessData()[b];
    });

    // Elite Re-evaluation: Re-evaluate top 3 candidates to reduce noise/variance
    const int numElitesToReevaluate = std::min(3, evaluatedCandidates);
    for (int i = 0; i < numElitesToReevaluate; ++i)
    {
        if (convo::consumeAtomic(stopRequested, std::memory_order_acquire)
            || stopToken.stop_requested())
            break;

        const int idx = sortedIndices[i];
        // Use a different context or just re-run (if segments were randomized, this would be more effective)
        // For now, we just re-run to ensure stability.
        const auto* population = candidatePopulationMatrix();
        const double secondScore = evaluateCandidate(evaluationWorkers[0].context, population[idx], numSegments, evaluationBitDepth);
        candidateFitnessData()[idx] = (candidateFitnessData()[idx] + secondScore) * 0.5;
    }

    // Final pass to find the best
    for (int populationIndex = 0; populationIndex < evaluatedCandidates; ++populationIndex)
    {
        if (convo::consumeAtomic(stopRequested, std::memory_order_acquire)
            || stopToken.stop_requested())
            break;

        const double score = candidateFitnessData()[populationIndex];
        if (score < bestCandidateScore)
        {
            bestCandidateScore = score;
            bestCandidateIndex = populationIndex;
        }
    }

    return evaluatedCandidates;
}

void NoiseShaperLearner::workerThreadMain(std::stop_token stopToken)
{
    // Transition:
    // Starting -> Running
    convo::publishAtomic(workerState, WorkerState::Running, std::memory_order_release);
    engine.getAffinityManager().applyCurrentThreadPolicy(ThreadType::LearnerMain);

    // FTZ/DAZ をスレッド開始時に設定する。
    // これらのフラグはスレッドローカルであるため、スレッドごとに設定が必要。
    // 設定しない場合、LatticeNoiseShaper の状態変数や MKL DFTI の中間バッファで
    // デノーマル数が発生し、CPU が 100 倍以上の速度低下を起こす可能性がある。
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    try
    {
        convo::publishAtomic(progress.status, Status::WaitingForAudio, std::memory_order_release);

        try
        {
            startEvaluationWorkers();
        }
        catch (...)
        {
            // 後処理で一貫して終了させる
        }
        SessionSignature activeSession = captureSessionSignature();
        bool resume = convo::exchangeAtomic(pendingResume, false, std::memory_order_acquire);
        juce::Logger::writeToLog("[NoiseShaperLearner] worker: resume=" + juce::String(static_cast<int>(resume))
            + " sampleRate=" + juce::String(activeSession.sampleRateHz)
            + " bank=" + juce::String(activeSession.adaptiveCoeffBankIndex)
            + " sessionId=" + juce::String(static_cast<juce::int64>(activeSession.sessionId)));
        resetLearningSession(activeSession, resume);

        // ============================================================================
        // 【追加】Multi-start ロジック
        // 初回起動時（resumeでない場合）かつ設定が有効な場合、複数のシードで試行
        // ============================================================================
        if (!resume && convo::consumeAtomic(settings.cmaesRestarts, std::memory_order_acquire) > 1)
        {
            convo::publishAtomic(progress.status, Status::WaitingForAudio, std::memory_order_release);

            // 評価に必要なセグメントが溜まるまで待機
            while (segmentBuffer.getNumAvailableSamples() < kRecentSampleRequest
                   && !convo::consumeAtomic(stopRequested, std::memory_order_acquire)
                   && !stopToken.stop_requested())
            {
                drainCaptureQueue(activeSession);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            if (!convo::consumeAtomic(stopRequested, std::memory_order_acquire) && !stopToken.stop_requested())
            {
                const int segmentCount = buildTrainingSegments();
                if (segmentCount >= 2)
                {
                    double bestRestartScore = std::numeric_limits<double>::max();
                    State bestRestartState;
                    State initialState;
                    getState(bestRestartState);
                    initialState = bestRestartState;   // 初期状態を保存

                    const int evaluationBitDepth = engine.getDitherBitDepth() > 0 ? engine.getDitherBitDepth() : 24;

                    int restarts = convo::consumeAtomic(settings.cmaesRestarts, std::memory_order_acquire);
                    for (int restartIdx = 0; restartIdx < restarts; ++restartIdx)
                    {
                        if (convo::consumeAtomic(stopRequested) || stopToken.stop_requested()) break;

                        // 初期状態に戻してからシードを変える
                        setState(initialState);
                        optimizer.setSeed(makeDeterministicRestartSeed(activeSession.sampleRateHz,
                                                                       activeSession.bitDepth,
                                                                       activeSession.adaptiveCoeffBankIndex,
                                                                       activeSession.sessionId,
                                                                       restartIdx));

                        // 数世代だけ回して良し悪しを判断
                        for (int g = 0; g < 3; ++g)
                        {
                            optimizer.sample(candidatePopulationMatrix());
                            int dummyIdx = 0;
                            double currentBestScore = 0.0;
                            evaluatePopulation(segmentCount,
                                               evaluationBitDepth,
                                               dummyIdx,
                                               currentBestScore,
                                               stopToken);
                            optimizer.update(candidatePopulationMatrix(), candidateFitnessData());

                            if (currentBestScore < bestRestartScore)
                            {
                                bestRestartScore = currentBestScore;
                                getState(bestRestartState);
                            }
                        }
                    }
                    // 最も良かった状態に復元
                    setState(bestRestartState);
                }
            }
        }

        double parcor[CmaEsOptimizer::kDim] = {};
        double bestScore = std::numeric_limits<double>::max();
        int generation = 0;
        DrainStats cumulativeDrainStats {};
        auto lastWaitingDiagnosticsLogTime = std::chrono::steady_clock::now();

        juce::Logger::writeToLog("[NoiseShaperLearner] worker: entering main loop");
        // Safety: force-reset stopRequested before main loop, in case a stale
        // Stop dispatch (e.g. from IRChanged during Idle state) set it before
        // the thread was created.
        convo::publishAtomic(stopRequested, false, std::memory_order_release);
        int mainLoopIter = 0;
        try
        {
        for (;;)
        {
            const bool sr = convo::consumeAtomic(stopRequested, std::memory_order_acquire);
            const bool st = stopToken.stop_requested();
            if (mainLoopIter <= 10 || mainLoopIter % 20 == 0)
                juce::Logger::writeToLog("[NoiseShaperLearner] worker: loop iter=" + juce::String(mainLoopIter)
                    + " stopReq=" + juce::String(static_cast<int>(sr))
                    + " stopTok=" + juce::String(static_cast<int>(st)));
            if (sr || st)
            {
                juce::Logger::writeToLog("[NoiseShaperLearner] worker: EXIT at iter=" + juce::String(mainLoopIter)
                    + " stopReq=" + juce::String(static_cast<int>(sr))
                    + " stopTok=" + juce::String(static_cast<int>(st)));
                DBG("[NoiseShaperLearner] worker: EXIT confirm");
                break;
            }
            ++mainLoopIter;
            const auto thisGenerationStart = std::chrono::steady_clock::now();

            // インターバル待機（start-to-start、condition_variable 使用）
            if (generationIntervalSeconds > 0.0 && lastGenerationStart != std::chrono::steady_clock::time_point{}) {
                auto next = lastGenerationStart + std::chrono::duration<double>(generationIntervalSeconds);
                std::unique_lock<std::mutex> lock(intervalMutex_);
                intervalCv_.wait_until(lock, next, [this, &stopToken]() -> bool {
                    return convo::consumeAtomic(stopRequested, std::memory_order_acquire)
                        || stopToken.stop_requested();
                });
            }
            lastGenerationStart = thisGenerationStart;

            handleModeSwitch();

            // 実再生時間ベースフェーズ判定
            const int newPhase = computePhase(activeMode, accumulatedPlaybackSeconds);
            if (newPhase != currentPhase) {
                currentPhase = newPhase;
                convo::publishAtomic(progress.currentPhase, currentPhase, std::memory_order_release);
                applyPhaseParams(activeMode, newPhase);
            }

            const SessionSignature currentSession = captureSessionSignature();
            if (activeSession.sampleRateHz != currentSession.sampleRateHz
                || activeSession.adaptiveCoeffBankIndex != currentSession.adaptiveCoeffBankIndex)
            {
                activeSession = currentSession;
                resetLearningSession(activeSession, true);
                bestScore = std::numeric_limits<double>::max();
                generation = 0;
                continue;
            }
            // When only sessionId changes (e.g. DSP replaced by HardReset), update
            // the sessionId to 0 (accept any) without resetting accumulated data,
            // so blocks from both old and new DSP are compatible.
            if (activeSession.sessionId != currentSession.sessionId)
                activeSession.sessionId = 0;

            const DrainStats latestDrainStats = drainCaptureQueue(activeSession);
            cumulativeDrainStats.acceptedBlocks += latestDrainStats.acceptedBlocks;
            cumulativeDrainStats.droppedBySession += latestDrainStats.droppedBySession;
            cumulativeDrainStats.droppedBySampleRate += latestDrainStats.droppedBySampleRate;
            cumulativeDrainStats.droppedByBank += latestDrainStats.droppedByBank;

            const int segmentCount = buildTrainingSegments();
            convo::publishAtomic(progress.segmentCount, segmentCount, std::memory_order_release);

            if (segmentCount < 2)
            {
                convo::publishAtomic(progress.status, Status::WaitingForAudio, std::memory_order_release);

                const auto now = std::chrono::steady_clock::now();
                if (now - lastWaitingDiagnosticsLogTime >= std::chrono::seconds(1))
                {
                    juce::Logger::writeToLog("[NoiseShaperLearner] Waiting diagnostics: accepted="
                        + juce::String(cumulativeDrainStats.acceptedBlocks)
                        + " dropSession=" + juce::String(cumulativeDrainStats.droppedBySession)
                        + " dropSampleRate=" + juce::String(cumulativeDrainStats.droppedBySampleRate)
                        + " dropBank=" + juce::String(cumulativeDrainStats.droppedByBank)
                        + " bufferedSamples=" + juce::String(segmentBuffer.getNumAvailableSamples())
                        + " sessionId=" + juce::String(static_cast<juce::int64>(activeSession.sessionId))
                        + " sampleRateHz=" + juce::String(activeSession.sampleRateHz)
                        + " bankIndex=" + juce::String(activeSession.adaptiveCoeffBankIndex));
                    cumulativeDrainStats = {};
                    lastWaitingDiagnosticsLogTime = now;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }

            convo::publishAtomic(progress.status, Status::Running, std::memory_order_release);
            optimizer.sample(candidatePopulationMatrix());

            int bestCandidateIndex = 0;
            double bestCandidateScore = std::numeric_limits<double>::max();
            const int evaluationBitDepth = engine.getDitherBitDepth() > 0 ? engine.getDitherBitDepth() : 24;
            const int evaluatedCandidates = evaluatePopulation(segmentCount,
                                                               evaluationBitDepth,
                                                               bestCandidateIndex,
                                                               bestCandidateScore,
                                                               stopToken);

            if (convo::consumeAtomic(stopRequested, std::memory_order_acquire) || stopToken.stop_requested())
                break;

            if (evaluatedCandidates < 1)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }

            for (int fi = evaluatedCandidates; fi < CmaEsOptimizer::kPopulation; ++fi)
                candidateFitnessData()[fi] = std::numeric_limits<double>::max();

            if (evaluatedCandidates < CmaEsOptimizer::kElite)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }

            optimizer.update(candidatePopulationMatrix(), candidateFitnessData());

            CmaEsOptimizer::toParcor(candidatePopulationMatrix()[bestCandidateIndex], parcor);
            if (bestCandidateScore < bestScore && bestCandidateScore < std::numeric_limits<double>::max())
            {
                bestScore = bestCandidateScore;
                publishGenerationResult(parcor, bestScore, evaluatedCandidates);
            }

            if (bestCandidateScore < std::numeric_limits<double>::max())
            {
                convo::publishAtomic(progress.latestScore, bestCandidateScore, std::memory_order_release);
                appendHistoryPoint(bestCandidateScore);
            }

            convo::publishAtomic(progress.iteration, generation + 1, std::memory_order_release);
            convo::fetchAddAtomic(progress.totalGenerations, 1, std::memory_order_acq_rel);
            ++generation;

            // 終了条件の判定
            double targetSeconds = 0.0;
            switch (activeMode)
            {
                case LearningMode::Shortest:   targetSeconds = 10.0; break;
                case LearningMode::Short:      targetSeconds = 30.0; break;
                case LearningMode::Middle:     targetSeconds = 60.0; break;
                case LearningMode::Long:       targetSeconds = 120.0; break;
                case LearningMode::Ultra:      targetSeconds = 300.0; break;
                case LearningMode::Continuous: targetSeconds = std::numeric_limits<double>::max(); break;
            }

            if (accumulatedPlaybackSeconds >= targetSeconds)
            {
                convo::publishAtomic(progress.status, Status::Completed, std::memory_order_release);
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        } // end for(;;)
        juce::Logger::writeToLog("[NoiseShaperLearner] worker: AFTER MAIN LOOP (iter=" + juce::String(mainLoopIter) + ")");

        } // end inner try
        catch (const std::exception& e)
        {
            juce::Logger::writeToLog(juce::String("[NoiseShaperLearner] worker: INNER EXCEPTION: ") + juce::String(e.what()));
            convo::publishAtomic(errorMessage, "Inner worker exception", std::memory_order_release);
        }
        catch (...)
        {
            juce::Logger::writeToLog("[NoiseShaperLearner] worker: INNER UNKNOWN EXCEPTION");
            convo::publishAtomic(errorMessage, "Unknown inner exception", std::memory_order_release);
        }

        if (convo::consumeAtomic(progress.status, std::memory_order_acquire) != Status::Completed)
            convo::publishAtomic(progress.status, Status::Idle, std::memory_order_release);
    }
    catch (const std::exception& e)
    {
        juce::Logger::writeToLog(juce::String("[NoiseShaperLearner] worker: OUTER EXCEPTION: ") + juce::String(e.what()));
        convo::publishAtomic(errorMessage, "Worker thread exception", std::memory_order_release);
        convo::publishAtomic(progress.status, Status::Error, std::memory_order_release);
    }
    catch (...)
    {
        juce::Logger::writeToLog("[NoiseShaperLearner] worker: OUTER UNKNOWN EXCEPTION");
        convo::publishAtomic(errorMessage, "Error in worker thread", std::memory_order_release);
        convo::publishAtomic(progress.status, Status::Error, std::memory_order_release);
    }

    stopEvaluationWorkers();
    // Transition:
    // Running/Stopping -> Idle
    convo::publishAtomic(workerState, WorkerState::Idle, std::memory_order_release);
}

NoiseShaperLearner::SessionSignature NoiseShaperLearner::captureSessionSignature() noexcept
{
    SessionSignature session;
    session.sampleRateHz = static_cast<int>(convo::consumeAtomic(engine.currentSampleRate, std::memory_order_acquire) + 0.5);
    session.bitDepth = engine.getDitherBitDepth();
    session.adaptiveCoeffBankIndex = convo::consumeAtomic(engine.currentAdaptiveCoeffBankIndex, std::memory_order_acquire);
    const auto ctx = convo::makeWorkerReaderContext(rcuReader, 0);
    const auto runtimeReadHandle = engine.makeRuntimeReadHandle(ctx);
    auto* dsp = engine.resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle);
    if (dsp != nullptr)
        session.sessionId = dsp->currentCaptureSessionId;
    return session;
}

void NoiseShaperLearner::resetLearningSession(const SessionSignature& session, bool resume) noexcept
{
    segmentBuffer.clear();

    if (!resume)
    {
        accumulatedPlaybackSeconds = 0.0;
        convo::publishAtomic(historyCount, 0, std::memory_order_release);
        {
            const std::scoped_lock<std::mutex> lock(historyMutex);
            bestScoreHistory.fill(0.0);
            historyHead = 0;
        }
        convo::publishAtomic(progress.iteration, 0, std::memory_order_release);
        convo::publishAtomic(progress.segmentCount, 0, std::memory_order_release);
        convo::publishAtomic(progress.bestScore, 0.0, std::memory_order_release);
        convo::publishAtomic(progress.latestScore, 0.0, std::memory_order_release);
    }

    convo::publishAtomic(progress.status, Status::WaitingForAudio, std::memory_order_release);

    const double sessionSampleRateVal = session.sampleRateHz > 0
        ? static_cast<double>(session.sampleRateHz)
        : AudioEngine::getAdaptiveSampleRateBankHz(session.adaptiveCoeffBankIndex);
    const int sessionBitDepthVal = session.bitDepth > 0
        ? session.bitDepth
        : kAdaptiveBitDepthValues[0]; // Fallback to 16-bit if not set

    this->sessionSampleRate = sessionSampleRateVal;
    this->sessionBitDepth = sessionBitDepthVal;
    this->currentSessionId = session.sessionId;

    configureEvaluationContexts(sessionSampleRateVal);

    // Try to load persistent state first
    const auto appDataDir = juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory).getChildFile("ConvoPeq");
    const auto stateFile = appDataDir.getChildFile("learned_state.xml");
    bool loaded = loadLearnedState(stateFile);

    if (loaded)
    {
        // State loaded, but we might still want to resume from engine state if it's more recent
        State engineState;
        if (resume && engine.getAdaptiveNoiseShaperState(session.adaptiveCoeffBankIndex, engineState) && engineState.iteration > 0)
        {
            setState(engineState);
        }
    }
    else
    {
        State savedState;
        if (resume && engine.getAdaptiveNoiseShaperState(session.adaptiveCoeffBankIndex, savedState) && savedState.iteration > 0)
        {
            setState(savedState);
        }
        else
        {
            double initialCoefficients[kOrder] = {};
            engine.getAdaptiveCoefficientsForSampleRateAndBitDepth(this->sessionSampleRate, this->sessionBitDepth, initialCoefficients, kOrder);
            optimizer.initFromParcor(initialCoefficients);

            for (int i = 0; i < kOrder; ++i)
                convo::publishAtomic(bestCoefficients[static_cast<size_t>(i)], initialCoefficients[i], std::memory_order_release);
        }
    }
}

NoiseShaperLearner::DrainStats NoiseShaperLearner::drainCaptureQueue(const SessionSignature& session) noexcept
{
    DrainStats stats {};
    AudioBlock block {};
    while (captureQueue.pop(block))
    {
        if (block.numSamples <= 0)
            continue;

        const bool sessionIdCompatible = (session.sessionId == 0u)
            || (block.sessionId == 0u)
            || (block.sessionId == session.sessionId);
        if (!sessionIdCompatible)
        {
            ++stats.droppedBySession;
            continue;
        }

        const bool sampleRateCompatible = (session.sampleRateHz <= 0)
            || (block.sampleRateHz <= 0)
            || (block.sampleRateHz == session.sampleRateHz);

        if (!sampleRateCompatible)
        {
            ++stats.droppedBySampleRate;
            continue;
        }

        if (block.adaptiveCoeffBankIndex != session.adaptiveCoeffBankIndex)
        {
            ++stats.droppedByBank;
            continue;
        }

        segmentBuffer.pushBlock(block.L, block.R, block.numSamples);
        const int playbackSampleRateHz = (session.sampleRateHz > 0)
            ? session.sampleRateHz
            : ((block.sampleRateHz > 0) ? block.sampleRateHz : 1);
        accumulatedPlaybackSeconds += static_cast<double>(block.numSamples)
            / static_cast<double>(playbackSampleRateHz);
        ++stats.acceptedBlocks;
    }
    convo::publishAtomic(progress.elapsedPlaybackSeconds, accumulatedPlaybackSeconds, std::memory_order_release);
    return stats;
}

int NoiseShaperLearner::buildTrainingSegments() noexcept
{
    // Reset buckets
    for (int i = 0; i < kNumLevels; ++i)
        levelBucketCounts[i] = 0;

    double recentLeft[kRecentSampleRequest] = {};
    double recentRight[kRecentSampleRequest] = {};

    const int maxRequired = kRecentSampleRequest;
    const int copiedSamples = segmentBuffer.copyLatest(recentLeft, recentRight, maxRequired);

    if (copiedSamples < AudioSegment::kLength)
        return 0;

    int totalSegments = 0;
    const int usableSamples = copiedSamples;

    for (int start = 0;
         start + AudioSegment::kLength <= usableSamples
         && totalSegments < kMaxTrainingSegments;
         start += kSegmentHop)
    {
        double sumSquares = 0.0;
        double maxPeak = 0.0;
        for (int sample = 0; sample < AudioSegment::kLength; ++sample)
        {
            const double l = recentLeft[start + sample];
            const double r = recentRight[start + sample];
            sumSquares += 0.5 * (l * l + r * r);
            maxPeak = std::max(maxPeak, std::max(std::abs(l), std::abs(r)));
        }

        const double rms = std::sqrt(sumSquares / static_cast<double>(AudioSegment::kLength));
        if (rms < kMinRMS)
            continue;

        // Spectral classification
        const double crestFactor = (rms > 1e-9) ? (maxPeak / rms) : 1.0;
        SpectralType type = SpectralType::Broadband;
        if (crestFactor > 5.0) type = SpectralType::Transient;
        else if (crestFactor < 1.6) type = SpectralType::Tonal;

        // For each target level, if the bucket is not full, add a normalized version
        for (int i = 0; i < kNumLevels; ++i)
        {
            if (levelBucketCounts[i] < kMaxSegmentsPerLevel)
            {
                const double targetRMS = std::pow(10.0, kTargetLevelsDB[i] / 20.0);

                // Calculate gain to reach target RMS, but limit by peak headroom
                double gain = targetRMS / rms;
                if (maxPeak * gain > kPeakHeadroom)
                    gain = kPeakHeadroom / maxPeak;

                auto& leveled = levelBuckets[i][levelBucketCounts[i]];
                leveled.targetRMS = targetRMS;
                leveled.appliedGain = gain;
                leveled.type = type;

                for (int sample = 0; sample < AudioSegment::kLength; ++sample)
                {
                    leveled.segment.left[sample] = recentLeft[start + sample] * gain;
                    leveled.segment.right[sample] = recentRight[start + sample] * gain;
                }

                // Precompute masking thresholds for this segment
                precomputeMaskingThresholds(leveled, this->sessionSampleRate);

                levelBucketCounts[i]++;
                totalSegments++;
            }
        }
    }

    return totalSegments;
}

double NoiseShaperLearner::evaluateCandidate(EvaluationContext& context,
                                             const double* candidateCoefficients,
                                             int numSegments,
                                             int evaluationBitDepth) noexcept
{
    juce::ScopedNoDenormals noDenormals;

    std::array<double, kOrder> mappedCoeffs;
    for (int i = 0; i < kOrder; ++i)
    {
        const double k = std::tanh(candidateCoefficients[i]);
        mappedCoeffs[static_cast<size_t>(i)] = LatticeNoiseShaper::clampCoeff(k, convo::consumeAtomic(settings.coeffSafetyMargin, std::memory_order_acquire));
    }

    return evaluateCandidateMapped(context, mappedCoeffs.data(), numSegments, evaluationBitDepth);
}

double NoiseShaperLearner::evaluateCandidateMapped(EvaluationContext& context,
                                                   const double* mappedCoefficients,
                                                   int numSegments,
                                                   int evaluationBitDepth) noexcept
{
    juce::ScopedNoDenormals noDenormals;

    // 【追加】安定性チェック（有効な場合）
    if (convo::consumeAtomic(settings.enableStabilityCheck, std::memory_order_acquire))
    {
        if (!LatticeNoiseShaper::isStable(mappedCoefficients, kOrder))
            return 1e18; // 不安定な場合は巨大なペナルティを返す
    }

    context.shaper.prepare(evaluationBitDepth);
    context.shaper.setCoefficients(mappedCoefficients, kOrder);

    double totalWeightedScore = 0.0;
    double totalWeight = 0.0;

    for (int i = 0; i < kNumLevels; ++i)
    {
        const int count = levelBucketCounts[i];
        if (count == 0) continue;

        double levelScoreSum = 0.0;
        for (int j = 0; j < count; ++j)
        {
            if (convo::consumeAtomic(stopRequested, std::memory_order_acquire))
                break;

            const auto& leveled = levelBuckets[i][j];

            context.shaper.reset();
            juce::FloatVectorOperations::copy(context.shapedLeft, leveled.segment.left, AudioSegment::kLength);
            juce::FloatVectorOperations::copy(context.shapedRight, leveled.segment.right, AudioSegment::kLength);

            context.shaper.processStereoBlock(context.shapedLeft,
                                              context.shapedRight,
                                              AudioSegment::kLength,
                                              kOutputHeadroom);

            // Calculate error relative to headroom-scaled input
            // ローカル__restrictポインタでエイリアスなしを明示（コンパイラの自動ベクトル化促進）
            {   double* __restrict dstL = context.errorLeft;
                double* __restrict dstR = context.errorRight;
                const double* __restrict srcL = context.shapedLeft;
                const double* __restrict srcR = context.shapedRight;
                const double* __restrict refL = leveled.segment.left;
                const double* __restrict refR = leveled.segment.right;
                for (int k = 0; k < AudioSegment::kLength; ++k)
                {
                    dstL[k]  = srcL[k] - (refL[k] * kOutputHeadroom);
                    dstR[k]  = srcR[k] - (refR[k] * kOutputHeadroom);
                }
            }

            const auto result = context.fftEvaluator.evaluate(context.errorLeft, context.errorRight, &leveled.segment.maskingThresholds);

    // Hybrid score: blend time domain RMS and frequency domain composite score
            // Low level (-40/-30dBFS) -> more time domain weight (smaller alpha for freqScore)
            // High level (-20/-10dBFS) -> more freq domain weight (larger alpha for freqScore)
            double alpha = 0.5;
            if (kTargetLevelsDB[i] < -30.0) alpha = 0.3;
            else if (kTargetLevelsDB[i] > -15.0) alpha = 0.7;

            // Normalize scores to a similar scale for blending
            // result.timeDomainRms is sigma.
            // result.compositeScore is roughly N * sigma^2 * penalties.
            // We use a heuristic scaling to bring them into a comparable range (~0.0 to 1.0 for typical noise).
            const double timeScore = result.timeDomainRms * 1000.0;
            const double freqScore = std::sqrt(result.compositeScore / MklFftEvaluator::kFftLength) * 1000.0;

            levelScoreSum += (alpha * freqScore + (1.0 - alpha) * timeScore);
        }

        const double levelAverageScore = levelScoreSum / count;
        const double weight = currentLevelWeights[static_cast<size_t>(i)];
        totalWeightedScore += levelAverageScore * weight;
        totalWeight += weight;
    }

    if (totalWeight <= 0.0)
        return std::numeric_limits<double>::max();

    return totalWeightedScore / totalWeight;
}

void NoiseShaperLearner::precomputeMaskingThresholds(LeveledSegment& leveled, double sampleRate) noexcept
{
    auto& evaluator = evaluationWorkers[0].context.fftEvaluator;

    MklFftEvaluator::CcsComplex spectrumL[MklFftEvaluator::kSpectrumBins];  // ← 変更後
    MklFftEvaluator::CcsComplex spectrumR[MklFftEvaluator::kSpectrumBins];  // ← 変更後


    evaluator.computeFft(leveled.segment.left, leveled.segment.right, spectrumL, spectrumR);

    const double binWidth = (sampleRate * 0.5) / (MklFftEvaluator::kSpectrumBins - 1);

    for (int k = 0; k < MklFftEvaluator::kSpectrumBins; ++k)
    {
        const double magSqL = spectrumL[k].real * spectrumL[k].real + spectrumL[k].imag * spectrumL[k].imag;
        const double magSqR = spectrumR[k].real * spectrumR[k].real + spectrumR[k].imag * spectrumR[k].imag;
        const double avgMagSq = 0.5 * (magSqL + magSqR);
        const double freq = k * binWidth;

        leveled.segment.maskingThresholds[k] = evaluator.computeMaskingThreshold(avgMagSq, freq);
    }
}

bool NoiseShaperLearner::saveLearnedState(const juce::File& file) const
{
    juce::XmlElement xml("ConvoPeqLearnedState");
    xml.setAttribute("bestScore", convo::consumeAtomic(progress.bestScore, std::memory_order_acquire));
    xml.setAttribute("sampleRate", this->sessionSampleRate);
    xml.setAttribute("bitDepth", this->sessionBitDepth);
    xml.setAttribute("phase", convo::consumeAtomic(progress.currentPhase, std::memory_order_acquire));
    xml.setAttribute("elapsedPlaybackSeconds", convo::consumeAtomic(progress.elapsedPlaybackSeconds, std::memory_order_acquire));

    auto* coeffsXml = xml.createNewChildElement("BestCoefficients");
    for (int i = 0; i < kOrder; ++i)
        coeffsXml->setAttribute("c" + juce::String(i), convo::consumeAtomic(bestCoefficients[i], std::memory_order_acquire));

    auto* cmaXml = xml.createNewChildElement("CmaMean");
    State state;
    getState(state);
    for (int i = 0; i < kOrder; ++i)
        cmaXml->setAttribute("m" + juce::String(i), state.mean[i]);

    return xml.writeTo(file);
}

bool NoiseShaperLearner::loadLearnedState(const juce::File& file)
{
    auto xml = juce::XmlDocument::parse(file);
    if (xml == nullptr || !xml->hasTagName("ConvoPeqLearnedState"))
        return false;

    const double savedSampleRate = xml->getDoubleAttribute("sampleRate");
    const int savedBitDepth = xml->getIntAttribute("bitDepth");

    // Only load if sample rate and bit depth match
    if (std::abs(savedSampleRate - sessionSampleRate) > 0.1 || savedBitDepth != sessionBitDepth)
        return false;

    convo::publishAtomic(progress.bestScore, xml->getDoubleAttribute("bestScore"), std::memory_order_release);
    convo::publishAtomic(progress.currentPhase, xml->getIntAttribute("phase"), std::memory_order_release);
    convo::publishAtomic(progress.elapsedPlaybackSeconds, xml->getDoubleAttribute("elapsedPlaybackSeconds"), std::memory_order_release);

    if (auto* coeffsXml = xml->getChildByName("BestCoefficients"))
    {
        for (int i = 0; i < kOrder; ++i)
            convo::publishAtomic(bestCoefficients[i], coeffsXml->getDoubleAttribute("c" + juce::String(i)), std::memory_order_release);
    }

    if (auto* cmaXml = xml->getChildByName("CmaMean"))
    {
        double mean[kOrder] = {};
        for (int i = 0; i < kOrder; ++i)
            mean[i] = cmaXml->getDoubleAttribute("m" + juce::String(i));

        // We only restore the mean, keeping other CMA state as literature defaults or engine state
        // unless we want full CMA state persistence. For now, mean + bestCoeffs is a good warm start.
        optimizer.setMean(mean);
    }

    return true;
}

void NoiseShaperLearner::publishGenerationResult(const double* coeffs, double score, int evaluatedCandidates) noexcept
{
    // coeffs are in unconstrained space, map to reflection coefficients for the engine
    std::array<double, kOrder> mappedCoeffs;
    for (int i = 0; i < kOrder; ++i)
    {
        const double k = std::tanh(coeffs[i]);
        mappedCoeffs[static_cast<size_t>(i)] = k;
        convo::publishAtomic(bestCoefficients[static_cast<size_t>(i)], k, std::memory_order_release);
    }

    convo::publishAtomic(progress.bestScore, score, std::memory_order_release);

    // 非同期保存（間隔制限付き）
    const auto now = std::chrono::steady_clock::now();
    if (now - lastSaveTime >= kSaveInterval)
    {
        lastSaveTime = now;

        const auto appDataDir = juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory).getChildFile("ConvoPeq");
        if (!appDataDir.exists()) appDataDir.createDirectory();
        const auto stateFile = appDataDir.getChildFile("learned_state.xml");
        const auto filePath = stateFile.getFullPathName();

        // スナップショットを取得
        State snapshot;
        getState(snapshot);

        juce::WeakReference<NoiseShaperLearner> weakSelf(this);
        g_saveThreadPool.addJob([weakSelf, filePath]()
        {
            if (auto* self = weakSelf.get())
                if (!self->saveLearnedState(juce::File(filePath)))
                    convo::publishAtomic(self->errorMessage, "Failed to save learned state", std::memory_order_release);
        });
    }

    // Capture state on worker thread
    // 注: 以下の currentState はスナップショットとは別に取得（非同期保存用のスナップショットは上で取得済み）
    State currentState;
    getState(currentState);

    // Capture bank info
    const double sr = engine.getSampleRate();
    const int bd = engine.getDitherBitDepth();
    const auto currentMode = static_cast<LearningMode>(convo::consumeAtomic(progress.learningMode, std::memory_order_acquire));
    const int bankIndex = AudioEngine::getAdaptiveCoeffBankIndex(sr, bd, currentMode);

    juce::WeakReference<NoiseShaperLearner> weakSelf(this);
    juce::MessageManager::callAsync([weakSelf, mappedCoeffs, currentState, bankIndex]()
    {
        if (auto* self = weakSelf.get())
        {
            self->engine.submitRebuildIntent(convo::RebuildKind::Structural,
                                             AudioEngine::RebuildTelemetryReason::EnqueueSnapshotCommand,
                                             AudioEngine::RebuildTelemetryClass::Snapshot,
                                             AudioEngine::RebuildTelemetryPolicy::Replaceable);
            self->engine.storeLearnedCoeffs(mappedCoeffs.data());
            self->engine.setAdaptiveNoiseShaperState(bankIndex, currentState);
            self->engine.requestAdaptiveAutosave();
        }
    });

}

void NoiseShaperLearner::appendHistoryPoint(double score) noexcept
{
    const std::scoped_lock<std::mutex> lock(historyMutex);
    const int count = convo::consumeAtomic(historyCount, std::memory_order_acquire);
    if (count < kMaxHistoryPoints)
    {
        bestScoreHistory[static_cast<size_t>(historyHead)] = score;
        historyHead = (historyHead + 1) % kMaxHistoryPoints;
        convo::publishAtomic(historyCount, count + 1, std::memory_order_release);
    }
    else
    {
        bestScoreHistory[static_cast<size_t>(historyHead)] = score;
        historyHead = (historyHead + 1) % kMaxHistoryPoints;
        convo::publishAtomic(historyCount, kMaxHistoryPoints, std::memory_order_release);
    }
}
