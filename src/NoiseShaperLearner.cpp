#include "NoiseShaperLearner.h"
#include "AudioEngine.h"

#include <JuceHeader.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>
#include <xmmintrin.h>  // _MM_SET_FLUSH_ZERO_MODE
#include <pmmintrin.h>  // _MM_SET_DENORMALS_ZERO_MODE

namespace
{
    constexpr double kOutputHeadroom = 0.8912509381337456;
    constexpr int kSegmentHop = AudioSegment::kLength / 2;
    constexpr int kRecentSampleRequest = AudioSegment::kLength + (kSegmentHop * (NoiseShaperLearner::kMaxTrainingSegments - 1));
}

// 静的メンバの定義
juce::ThreadPool NoiseShaperLearner::saveThreadPool(1);

NoiseShaperLearner::NoiseShaperLearner(AudioEngine& engineRef,
                                       LockFreeRingBuffer<AudioBlock, 4096>& captureQueueRef)
    : engine(engineRef),
    // lastSaveTime の初期化（コンストラクタ本体で行う）
      captureQueue(captureQueueRef)
{
    for (auto& c : bestCoefficients)
        c.store(0.0, std::memory_order_relaxed);

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
    stopRequested.store(true, std::memory_order_release);
    {
        const std::scoped_lock<std::mutex> lock(evaluationDispatchMutex);
        evaluationWorkersShouldExit = true;
    }
    evaluationDispatchCv.notify_all();

    if (workerThread.joinable())
        workerThread.join();

    pendingRestart.store(false, std::memory_order_release);
}

void NoiseShaperLearner::startLearning(bool resume)
{
    // 前回セッション残骸のクリーンアップ
    if (isRunning() || workerThread.joinable() || !workerThreadFinished.load(std::memory_order_acquire))
    {
        stopLearning();
        if (workerThread.joinable())
            workerThread.join();
    }

    // 全フラグを明示リセット
    workerThreadFinished.store(true, std::memory_order_release);
    pendingRestart.store(false, std::memory_order_release);
    startRequested.store(false, std::memory_order_release);
    stopRequested.store(false, std::memory_order_release);
    errorMessage.store(nullptr, std::memory_order_release);

    bool expectedStartRequested = false;
    if (!startRequested.compare_exchange_strong(expectedStartRequested, true,
                                                std::memory_order_acq_rel,
                                                std::memory_order_relaxed))
    {
        return;
    }

    pendingResume.store(resume, std::memory_order_release);
    workerThreadFinished.store(false, std::memory_order_release);

    progress.status.store(Status::WaitingForAudio, std::memory_order_release);
    progress.iteration.store(0, std::memory_order_relaxed);
    progress.segmentCount.store(0, std::memory_order_relaxed);
    progress.bestScore.store(0.0, std::memory_order_relaxed);
    progress.latestScore.store(0.0, std::memory_order_relaxed);
    progress.elapsedPlaybackSeconds.store(0.0, std::memory_order_relaxed);
    progress.currentPhase.store(1, std::memory_order_relaxed);
    accumulatedPlaybackSeconds = 0.0;
    currentPhase = 1;
    segmentBuffer.clear();
    historyCount.store(0, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(historyMutex);
        bestScoreHistory.fill(0.0);
        historyHead = 0;
    }

    activeMode = pendingMode;
    progress.learningMode.store(static_cast<int>(activeMode), std::memory_order_relaxed);
    lastGenerationStart = std::chrono::steady_clock::time_point{};
    applyPhaseParams(activeMode, currentPhase);

    try
    {
        workerThread = std::thread(&NoiseShaperLearner::workerThreadMain, this);
    }
    catch (const std::exception& e)
    {
        DBG("[NoiseShaperLearner] failed to start worker thread: " << e.what());
        errorMessage.store("Failed to start learning thread", std::memory_order_release);
        progress.status.store(Status::Error, std::memory_order_release);
        startRequested.store(false, std::memory_order_release);
        workerThreadFinished.store(true, std::memory_order_release);
    }
    catch (...)
    {
        errorMessage.store("Unknown error starting worker thread", std::memory_order_release);
        progress.status.store(Status::Error, std::memory_order_release);
        startRequested.store(false, std::memory_order_release);
        workerThreadFinished.store(true, std::memory_order_release);
    }
}

void NoiseShaperLearner::stopLearning()
{
    stopRequested.store(true, std::memory_order_release);
    stopEvaluationWorkers();
    if (workerThread.joinable())
        workerThread.join();

    workerThreadFinished.store(true, std::memory_order_release);
    startRequested.store(false, std::memory_order_release);
    pendingRestart.store(false, std::memory_order_release);
    pendingResume.store(false, std::memory_order_release);
    progress.status.store(Status::Idle, std::memory_order_release);
    errorMessage.store(nullptr, std::memory_order_release);

    evaluationDispatchCv.notify_all();
}

void NoiseShaperLearner::setLearningMode(LearningMode mode) noexcept
{
    pendingMode = mode;
    modeSwitchRequested.store(true, std::memory_order_release);

    if (!isRunning())
    {
        activeMode = mode;
        progress.learningMode.store(static_cast<int>(activeMode), std::memory_order_relaxed);
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
    if (!modeSwitchRequested.load(std::memory_order_acquire))
        return;

    // Save current state to AudioEngine
    State currentState;
    getState(currentState);
    const double sr = engine.getSampleRate();
    const int bd = engine.getDitherBitDepth();
    int bankIndex = AudioEngine::getAdaptiveCoeffBankIndex(sr, bd, activeMode);
    engine.setAdaptiveNoiseShaperState(bankIndex, currentState);

    activeMode = pendingMode;
    progress.learningMode.store(static_cast<int>(activeMode), std::memory_order_relaxed);

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
            bestCoefficients[static_cast<size_t>(i)].store(initialCoefficients[i], std::memory_order_relaxed);

        accumulatedPlaybackSeconds = 0.0;
        progress.iteration.store(0, std::memory_order_relaxed);
        progress.bestScore.store(0.0, std::memory_order_relaxed);
        progress.latestScore.store(0.0, std::memory_order_relaxed);
        progress.processCount.store(0, std::memory_order_relaxed);
        progress.totalGenerations.store(0, std::memory_order_relaxed);
    }

    // 現在の再生時間に基づいてフェーズを再計算し、パラメータを適用
    currentPhase = computePhase(activeMode, accumulatedPlaybackSeconds);
    progress.currentPhase.store(currentPhase, std::memory_order_relaxed);
    applyPhaseParams(activeMode, currentPhase);

    modeSwitchRequested.store(false, std::memory_order_release);
}

bool NoiseShaperLearner::isRunning() const noexcept
{
    return progress.status.load(std::memory_order_acquire) == Status::Running
        || progress.status.load(std::memory_order_acquire) == Status::WaitingForAudio;
}

const NoiseShaperLearner::Progress& NoiseShaperLearner::getProgress() const noexcept
{
    return progress;
}

void NoiseShaperLearner::getState(State& outState) const noexcept
{
    optimizer.serializeTo(outState.mean, outState.covarianceUpperTriangle, outState.sigma);
    for (int i = 0; i < kOrder; ++i)
        outState.bestCoefficients[i] = bestCoefficients[i].load(std::memory_order_relaxed);
    outState.elapsedPlaybackSeconds = accumulatedPlaybackSeconds;
    outState.currentPhase = currentPhase;
    outState.iteration = progress.iteration.load(std::memory_order_relaxed);
    outState.bestScore = progress.bestScore.load(std::memory_order_relaxed);
    outState.processCount = progress.processCount.load(std::memory_order_relaxed);
    outState.totalGenerations = progress.totalGenerations.load(std::memory_order_relaxed);
}

void NoiseShaperLearner::setState(const State& inState) noexcept
{
    optimizer.deserializeFrom(inState.mean, inState.covarianceUpperTriangle, inState.sigma);
    for (int i = 0; i < kOrder; ++i)
        bestCoefficients[i].store(inState.bestCoefficients[i], std::memory_order_relaxed);
    accumulatedPlaybackSeconds = inState.elapsedPlaybackSeconds;
    currentPhase = inState.currentPhase;
    progress.iteration.store(inState.iteration, std::memory_order_relaxed);
    progress.bestScore.store(inState.bestScore, std::memory_order_relaxed);
    progress.elapsedPlaybackSeconds.store(accumulatedPlaybackSeconds, std::memory_order_relaxed);
    progress.currentPhase.store(currentPhase, std::memory_order_relaxed);
    progress.processCount.store(inState.processCount, std::memory_order_relaxed);
    progress.totalGenerations.store(inState.totalGenerations, std::memory_order_relaxed);
}

int NoiseShaperLearner::copyBestScoreHistory(double* destination, int maxPoints) const noexcept
{
    if (destination == nullptr || maxPoints <= 0)
        return 0;

    const std::scoped_lock<std::mutex> lock(historyMutex);
    const int count = historyCount.load(std::memory_order_acquire);
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
        completedAuxEvaluationWorkers.store(0, std::memory_order_seq_cst);
        evaluationDispatchSerial.store(0, std::memory_order_seq_cst);
    }

    for (int workerIndex = 1; workerIndex < activeEvaluationWorkerCount; ++workerIndex)
    {
        auto& slot = evaluationWorkers[static_cast<size_t>(workerIndex)];
        try
        {
            slot.thread = std::thread(&NoiseShaperLearner::evaluationWorkerMain, this, workerIndex);
        }
        catch (const std::exception& e)
        {
            DBG("[NoiseShaperLearner] failed to start evaluation worker: " << e.what());
            errorMessage.store("Failed to start evaluation worker", std::memory_order_release);
            progress.status.store(Status::Error, std::memory_order_release);
            break;
        }
        catch (...)
        {
            errorMessage.store("Unknown error starting evaluation worker", std::memory_order_release);
            progress.status.store(Status::Error, std::memory_order_release);
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

void NoiseShaperLearner::evaluationWorkerMain(int workerIndex) noexcept
{
    // FTZ/DAZ をスレッド開始時に設定する（スレッドローカルフラグ）。
    // 評価ワーカーは MKL DFTI および LatticeNoiseShaper を使用するため、
    // デノーマル数による CPU 速度低下を防ぐために必須。
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    engine.pinCurrentThreadToNonAudioCoresIfNeeded();

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
                    return evaluationWorkersShouldExit || evaluationDispatchSerial.load(std::memory_order_acquire) != observedDispatchSerial;
                });

                if (evaluationWorkersShouldExit)
                    return;

                ++observedDispatchSerial;
                segmentCount = pendingEvaluationSegmentCount;
                evaluationBitDepth = pendingEvaluationBitDepth;
            }

            runEvaluationJobsForWorker(workerIndex, segmentCount, evaluationBitDepth);

            {
                const std::scoped_lock<std::mutex> lock(evaluationDispatchMutex);
                completedAuxEvaluationWorkers.fetch_add(1, std::memory_order_release);
            }
            evaluationDispatchCv.notify_all();
        }
        catch (...)
        {
            errorMessage.store("Error in evaluation worker", std::memory_order_release);
            progress.status.store(Status::Error, std::memory_order_release);
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
                                                    int evaluationBitDepth) noexcept
{
    if (numSegments <= 0)
        return;

    auto& context = evaluationWorkers[static_cast<size_t>(workerIndex)].context;
    double parcor[CmaEsOptimizer::kDim] = {};

    while (!stopRequested.load(std::memory_order_acquire))
    {
        const int populationIndex = nextEvaluationCandidateIndex.fetch_add(1, std::memory_order_acq_rel);
        if (populationIndex >= CmaEsOptimizer::kPopulation)
            break;

        CmaEsOptimizer::toParcor(candidatePopulation[populationIndex], parcor);
        const double score = evaluateCandidate(context, parcor, numSegments, evaluationBitDepth);
        candidateFitness[populationIndex] = score;
        progress.processCount.fetch_add(1, std::memory_order_release);
    }
}

int NoiseShaperLearner::evaluatePopulation(int numSegments,
                                           int evaluationBitDepth,
                                           int& bestCandidateIndex,
                                           double& bestCandidateScore)
{
    if (stopRequested.load(std::memory_order_acquire))
        return 0;

    nextEvaluationCandidateIndex.store(0, std::memory_order_seq_cst);

    {
        const std::scoped_lock<std::mutex> lock(evaluationDispatchMutex);
        pendingEvaluationSegmentCount = numSegments;
        pendingEvaluationBitDepth = evaluationBitDepth;
        completedAuxEvaluationWorkers.store(0, std::memory_order_seq_cst);
        evaluationDispatchSerial.fetch_add(1, std::memory_order_release);
    }

    if (activeAuxEvaluationWorkerCount > 0)
        evaluationDispatchCv.notify_all();

    runEvaluationJobsForWorker(0, numSegments, evaluationBitDepth);

    if (activeAuxEvaluationWorkerCount > 0)
    {
        std::unique_lock<std::mutex> lock(evaluationDispatchMutex);
        evaluationDispatchCv.wait(lock, [this]
        {
            return completedAuxEvaluationWorkers.load(std::memory_order_acquire) >= activeAuxEvaluationWorkerCount
                || stopRequested.load(std::memory_order_acquire);
        });
    }

    const int evaluatedCandidates = std::min(nextEvaluationCandidateIndex.load(std::memory_order_seq_cst),
                                             CmaEsOptimizer::kPopulation);

    bestCandidateIndex = 0;
    bestCandidateScore = std::numeric_limits<double>::max();

    // First pass: find top candidates
    std::vector<int> sortedIndices(evaluatedCandidates);
    std::iota(sortedIndices.begin(), sortedIndices.end(), 0);
    std::sort(sortedIndices.begin(), sortedIndices.end(), [this](int a, int b) {
        return candidateFitness[a] < candidateFitness[b];
    });

    // Elite Re-evaluation: Re-evaluate top 3 candidates to reduce noise/variance
    const int numElitesToReevaluate = std::min(3, evaluatedCandidates);
    for (int i = 0; i < numElitesToReevaluate; ++i)
    {
        if (stopRequested.load(std::memory_order_acquire))
            break;

        const int idx = sortedIndices[i];
        // Use a different context or just re-run (if segments were randomized, this would be more effective)
        // For now, we just re-run to ensure stability.
        const double secondScore = evaluateCandidate(evaluationWorkers[0].context, candidatePopulation[idx], numSegments, evaluationBitDepth);
        candidateFitness[idx] = (candidateFitness[idx] + secondScore) * 0.5;
    }

    // Final pass to find the best
    for (int populationIndex = 0; populationIndex < evaluatedCandidates; ++populationIndex)
    {
        if (stopRequested.load(std::memory_order_acquire))
            break;

        const double score = candidateFitness[populationIndex];
        if (score < bestCandidateScore)
        {
            bestCandidateScore = score;
            bestCandidateIndex = populationIndex;
        }
    }

    return evaluatedCandidates;
}

void NoiseShaperLearner::workerThreadMain()
{
    // FTZ/DAZ をスレッド開始時に設定する。
    // これらのフラグはスレッドローカルであるため、スレッドごとに設定が必要。
    // 設定しない場合、LatticeNoiseShaper の状態変数や MKL DFTI の中間バッファで
    // デノーマル数が発生し、CPU が 100 倍以上の速度低下を起こす可能性がある。
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    try
    {
        progress.status.store(Status::WaitingForAudio, std::memory_order_release);

        try
        {
            startEvaluationWorkers();
        }
        catch (...)
        {
            // 後処理で一貫して終了させる
        }
        engine.pinCurrentThreadToNoiseLearnerCoreIfNeeded();

        SessionSignature activeSession = captureSessionSignature();
        bool resume = pendingResume.exchange(false, std::memory_order_acquire);
        resetLearningSession(activeSession, resume);

        // ============================================================================
        // 【追加】Multi-start ロジック
        // 初回起動時（resumeでない場合）かつ設定が有効な場合、複数のシードで試行
        // ============================================================================
        if (!resume && settings.cmaesRestarts.load() > 1)
        {
            progress.status.store(Status::WaitingForAudio, std::memory_order_release);

            // 評価に必要なセグメントが溜まるまで待機
            while (segmentBuffer.getNumAvailableSamples() < kRecentSampleRequest && !stopRequested.load())
            {
                drainCaptureQueue(activeSession);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            if (!stopRequested.load())
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

                    int restarts = settings.cmaesRestarts.load();
                    for (int restartIdx = 0; restartIdx < restarts; ++restartIdx)
                    {
                        if (stopRequested.load()) break;

                        // 初期状態に戻してからシードを変える
                        setState(initialState);
                        optimizer.setSeed(static_cast<uint64_t>(std::chrono::system_clock::now().time_since_epoch().count()) + restartIdx);

                        // 数世代だけ回して良し悪しを判断
                        for (int g = 0; g < 3; ++g)
                        {
                            optimizer.sample(candidatePopulation);
                            int dummyIdx = 0;
                            double currentBestScore = 0.0;
                            evaluatePopulation(segmentCount, evaluationBitDepth, dummyIdx, currentBestScore);
                            optimizer.update(candidatePopulation, candidateFitness);

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

        while (!stopRequested.load(std::memory_order_acquire))
        {
            const auto thisGenerationStart = std::chrono::steady_clock::now();

            // インターバル待機（start-to-start）
            if (generationIntervalSeconds > 0.0 && lastGenerationStart != std::chrono::steady_clock::time_point{}) {
                auto next = lastGenerationStart + std::chrono::duration<double>(generationIntervalSeconds);
                while (std::chrono::steady_clock::now() < next && !stopRequested.load(std::memory_order_acquire))
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            lastGenerationStart = thisGenerationStart;

            handleModeSwitch();

            // 実再生時間ベースフェーズ判定
            const int newPhase = computePhase(activeMode, accumulatedPlaybackSeconds);
            if (newPhase != currentPhase) {
                currentPhase = newPhase;
                progress.currentPhase.store(currentPhase, std::memory_order_relaxed);
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

            drainCaptureQueue(activeSession);

            const int segmentCount = buildTrainingSegments();
            progress.segmentCount.store(segmentCount, std::memory_order_relaxed);

            if (segmentCount < 2)
            {
                progress.status.store(Status::WaitingForAudio, std::memory_order_release);
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }

            progress.status.store(Status::Running, std::memory_order_release);
            optimizer.sample(candidatePopulation);

            int bestCandidateIndex = 0;
            double bestCandidateScore = std::numeric_limits<double>::max();
            const int evaluationBitDepth = engine.getDitherBitDepth() > 0 ? engine.getDitherBitDepth() : 24;
            const int evaluatedCandidates = evaluatePopulation(segmentCount,
                                                               evaluationBitDepth,
                                                               bestCandidateIndex,
                                                               bestCandidateScore);

            if (stopRequested.load(std::memory_order_acquire))
                break;

            if (evaluatedCandidates < 1)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }

            for (int fi = evaluatedCandidates; fi < CmaEsOptimizer::kPopulation; ++fi)
                candidateFitness[fi] = std::numeric_limits<double>::max();

            if (evaluatedCandidates < CmaEsOptimizer::kElite)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }

            optimizer.update(candidatePopulation, candidateFitness);

            CmaEsOptimizer::toParcor(candidatePopulation[bestCandidateIndex], parcor);
            if (bestCandidateScore < bestScore && bestCandidateScore < std::numeric_limits<double>::max())
            {
                bestScore = bestCandidateScore;
                publishGenerationResult(parcor, bestScore, evaluatedCandidates);
            }

            if (bestCandidateScore < std::numeric_limits<double>::max())
            {
                progress.latestScore.store(bestCandidateScore, std::memory_order_relaxed);
                appendHistoryPoint(bestCandidateScore);
            }

            progress.iteration.store(generation + 1, std::memory_order_relaxed);
            progress.totalGenerations.fetch_add(1, std::memory_order_relaxed);
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
                progress.status.store(Status::Completed, std::memory_order_release);
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }

        if (progress.status.load(std::memory_order_acquire) != Status::Completed)
            progress.status.store(Status::Idle, std::memory_order_release);
    }
    catch (...)
    {
        errorMessage.store("Error in worker thread", std::memory_order_release);
        progress.status.store(Status::Error, std::memory_order_release);
    }

    stopEvaluationWorkers();
    workerThreadFinished.store(true, std::memory_order_release);
    startRequested.store(false, std::memory_order_release);
}

NoiseShaperLearner::SessionSignature NoiseShaperLearner::captureSessionSignature() const noexcept
{
    SessionSignature session;
    session.sampleRateHz = static_cast<int>(engine.currentSampleRate.load(std::memory_order_acquire) + 0.5);
    session.bitDepth = engine.getDitherBitDepth();
    session.adaptiveCoeffBankIndex = engine.currentAdaptiveCoeffBankIndex.load(std::memory_order_acquire);
    if (auto* dsp = engine.currentDSP.load(std::memory_order_acquire))
        session.sessionId = dsp->currentCaptureSessionId;
    return session;
}

void NoiseShaperLearner::resetLearningSession(const SessionSignature& session, bool resume) noexcept
{
    segmentBuffer.clear();

    if (!resume)
    {
        accumulatedPlaybackSeconds = 0.0;
        historyCount.store(0, std::memory_order_release);
        {
            const std::scoped_lock<std::mutex> lock(historyMutex);
            bestScoreHistory.fill(0.0);
            historyHead = 0;
        }
        progress.iteration.store(0, std::memory_order_relaxed);
        progress.segmentCount.store(0, std::memory_order_relaxed);
        progress.bestScore.store(0.0, std::memory_order_relaxed);
        progress.latestScore.store(0.0, std::memory_order_relaxed);
    }

    progress.status.store(Status::WaitingForAudio, std::memory_order_release);

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
                bestCoefficients[static_cast<size_t>(i)].store(initialCoefficients[i], std::memory_order_relaxed);
        }
    }
}

void NoiseShaperLearner::drainCaptureQueue(const SessionSignature& session) noexcept
{
    AudioBlock block {};
    while (captureQueue.pop(block))
    {
        if (block.numSamples <= 0)
            continue;

        if (block.sessionId != session.sessionId)
            continue;

        if (block.sampleRateHz == session.sampleRateHz
            && block.adaptiveCoeffBankIndex == session.adaptiveCoeffBankIndex)
        {
            segmentBuffer.pushBlock(block.L, block.R, block.numSamples);
            accumulatedPlaybackSeconds += static_cast<double>(block.numSamples) / session.sampleRateHz;
        }
    }
    progress.elapsedPlaybackSeconds.store(accumulatedPlaybackSeconds, std::memory_order_relaxed);
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

    // Map unconstrained CMA-ES parameters to reflection coefficients using tanh
    std::array<double, kOrder> mappedCoeffs;
    for (int i = 0; i < kOrder; ++i)
    {
        const double k = std::tanh(candidateCoefficients[i]);
        // 【追加】安全マージンの適用
        mappedCoeffs[static_cast<size_t>(i)] = LatticeNoiseShaper::clampCoeff(k, settings.coeffSafetyMargin.load());
    }

    // 【追加】安定性チェック（有効な場合）
    if (settings.enableStabilityCheck.load())
    {
        if (!LatticeNoiseShaper::isStable(mappedCoeffs.data(), kOrder))
            return 1e18; // 不安定な場合は巨大なペナルティを返す
    }

    context.shaper.prepare(evaluationBitDepth);
    context.shaper.setCoefficients(mappedCoeffs.data(), kOrder);

    double totalWeightedScore = 0.0;
    double totalWeight = 0.0;

    for (int i = 0; i < kNumLevels; ++i)
    {
        const int count = levelBucketCounts[i];
        if (count == 0) continue;

        double levelScoreSum = 0.0;
        for (int j = 0; j < count; ++j)
        {
            if (stopRequested.load(std::memory_order_acquire))
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
            for (int k = 0; k < AudioSegment::kLength; ++k)
            {
                context.errorLeft[k] = context.shapedLeft[k] - (leveled.segment.left[k] * kOutputHeadroom);
                context.errorRight[k] = context.shapedRight[k] - (leveled.segment.right[k] * kOutputHeadroom);
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

    MKL_Complex16 spectrumL[MklFftEvaluator::kSpectrumBins];
    MKL_Complex16 spectrumR[MklFftEvaluator::kSpectrumBins];

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
    xml.setAttribute("bestScore", progress.bestScore.load());
    xml.setAttribute("sampleRate", this->sessionSampleRate);
    xml.setAttribute("bitDepth", this->sessionBitDepth);
    xml.setAttribute("phase", progress.currentPhase.load());
    xml.setAttribute("elapsedPlaybackSeconds", progress.elapsedPlaybackSeconds.load());

    auto* coeffsXml = xml.createNewChildElement("BestCoefficients");
    for (int i = 0; i < kOrder; ++i)
        coeffsXml->setAttribute("c" + juce::String(i), bestCoefficients[i].load());

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

    progress.bestScore.store(xml->getDoubleAttribute("bestScore"));
    progress.currentPhase.store(xml->getIntAttribute("phase"));
    progress.elapsedPlaybackSeconds.store(xml->getDoubleAttribute("elapsedPlaybackSeconds"));

    if (auto* coeffsXml = xml->getChildByName("BestCoefficients"))
    {
        for (int i = 0; i < kOrder; ++i)
            bestCoefficients[i].store(coeffsXml->getDoubleAttribute("c" + juce::String(i)));
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
        bestCoefficients[static_cast<size_t>(i)].store(k, std::memory_order_relaxed);
    }

    progress.bestScore.store(score, std::memory_order_relaxed);

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
        saveThreadPool.addJob([weakSelf, filePath, snapshot]() mutable
        {
            if (auto* self = weakSelf.get())
                if (!self->saveLearnedState(juce::File(filePath)))
                    self->errorMessage.store("Failed to save learned state", std::memory_order_release);
        });
    }

    // Capture state on worker thread
    // 注: 以下の currentState はスナップショットとは別に取得（非同期保存用のスナップショットは上で取得済み）
    State currentState;
    getState(currentState);

    // Capture bank info
    const double sr = engine.getSampleRate();
    const int bd = engine.getDitherBitDepth();
    const auto currentMode = static_cast<LearningMode>(progress.learningMode.load(std::memory_order_relaxed));
    const int bankIndex = AudioEngine::getAdaptiveCoeffBankIndex(sr, bd, currentMode);

    juce::WeakReference<NoiseShaperLearner> weakSelf(this);
    juce::MessageManager::callAsync([weakSelf, mappedCoeffs, currentState, bankIndex]() mutable
    {
        if (auto* self = weakSelf.get())
        {
            self->engine.publishCoeffs(mappedCoeffs.data());
            self->engine.setAdaptiveNoiseShaperState(bankIndex, currentState);
            self->engine.requestAdaptiveAutosave();
        }
    });

}

void NoiseShaperLearner::appendHistoryPoint(double score) noexcept
{
    const std::scoped_lock<std::mutex> lock(historyMutex);
    const int count = historyCount.load(std::memory_order_relaxed);
    if (count < kMaxHistoryPoints)
    {
        bestScoreHistory[static_cast<size_t>(historyHead)] = score;
        historyHead = (historyHead + 1) % kMaxHistoryPoints;
        historyCount.store(count + 1, std::memory_order_release);
    }
    else
    {
        bestScoreHistory[static_cast<size_t>(historyHead)] = score;
        historyHead = (historyHead + 1) % kMaxHistoryPoints;
        historyCount.store(kMaxHistoryPoints, std::memory_order_release);
    }
}
