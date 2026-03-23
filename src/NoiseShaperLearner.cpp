#include "NoiseShaperLearner.h"
#include "AudioEngine.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <xmmintrin.h>  // _MM_SET_FLUSH_ZERO_MODE
#include <pmmintrin.h>  // _MM_SET_DENORMALS_ZERO_MODE

namespace
{
    constexpr double kOutputHeadroom = 0.8912509381337456;
    constexpr int kSegmentHop = AudioSegment::kLength / 2;
    constexpr int kRecentSampleRequest = AudioSegment::kLength + (kSegmentHop * (NoiseShaperLearner::kMaxTrainingSegments - 1));
}

NoiseShaperLearner::NoiseShaperLearner(AudioEngine& engineRef,
                                       LockFreeRingBuffer<AudioBlock, 4096>& captureQueueRef)
    : engine(engineRef),
      captureQueue(captureQueueRef)
{
    bestCoefficients.fill(0.0);

    const unsigned int hardwareThreadCount = std::thread::hardware_concurrency();
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
    if (isRunning())
        return;

    stopRequested.store(true, std::memory_order_release);
    {
        const std::scoped_lock<std::mutex> lock(evaluationDispatchMutex);
        evaluationWorkersShouldExit = true;
    }
    evaluationDispatchCv.notify_all();

    if (workerThread.joinable() && !workerThreadFinished.load(std::memory_order_acquire))
    {
        bool expected = false;
        if (pendingRestart.compare_exchange_strong(expected, true,
                                                    std::memory_order_acq_rel,
                                                    std::memory_order_relaxed))
        {
            pendingResume.store(resume, std::memory_order_release);
            juce::WeakReference<NoiseShaperLearner> weakSelf(this);
            juce::MessageManager::callAsync([weakSelf, resume]() mutable
            {
                if (auto* self = weakSelf.get())
                {
                    self->pendingRestart.store(false, std::memory_order_release);
                    self->startLearning(resume);
                }
            });
        }
        return;
    }

    pendingRestart.store(false, std::memory_order_release);
    if (workerThread.joinable())
        workerThread.join();

    stopRequested.store(false, std::memory_order_release);
    pendingResume.store(resume, std::memory_order_release);
    workerThreadFinished.store(false, std::memory_order_release);

    if (!resume)
    {
        progress.iteration.store(0, std::memory_order_relaxed);
        progress.segmentCount.store(0, std::memory_order_relaxed);
        progress.bestScore.store(0.0f, std::memory_order_relaxed);
        progress.latestScore.store(0.0f, std::memory_order_relaxed);
        progress.elapsedPlaybackSeconds.store(0.0, std::memory_order_relaxed);
        progress.currentPhase.store(1, std::memory_order_relaxed);
        accumulatedPlaybackSeconds = 0.0;
        currentPhase = 1;
        historyCount.store(0, std::memory_order_release);
        {
            const std::scoped_lock<std::mutex> lock(historyMutex);
            bestScoreHistory.fill(0.0f);
            historyHead = 0;
        }
    }
    else
    {
        accumulatedPlaybackSeconds = progress.elapsedPlaybackSeconds.load(std::memory_order_relaxed);
        currentPhase = progress.currentPhase.load(std::memory_order_relaxed);
    }

    progress.status.store(Status::WaitingForAudio, std::memory_order_release);
    errorMessage.store(nullptr, std::memory_order_release);
    segmentBuffer.clear();

    activeMode = pendingMode;
    progress.learningMode.store(static_cast<int>(activeMode), std::memory_order_relaxed);
    lastGenerationStart = std::chrono::steady_clock::time_point{};
    applyPhaseParams(activeMode, currentPhase);

    workerThread = std::thread(&NoiseShaperLearner::workerThreadMain, this);
}

void NoiseShaperLearner::stopLearning()
{
    stopRequested.store(true, std::memory_order_release);
    evaluationDispatchCv.notify_all();

    if (progress.status.load(std::memory_order_acquire) != Status::Error)
        progress.status.store(Status::Idle, std::memory_order_release);
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
            bestCoefficients[static_cast<size_t>(i)] = initialCoefficients[i];
            
        accumulatedPlaybackSeconds = 0.0;
        progress.iteration.store(0, std::memory_order_relaxed);
        progress.bestScore.store(0.0f, std::memory_order_relaxed);
        progress.latestScore.store(0.0f, std::memory_order_relaxed);
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
        outState.bestCoefficients[i] = bestCoefficients[i];
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
        bestCoefficients[i] = inState.bestCoefficients[i];
    accumulatedPlaybackSeconds = inState.elapsedPlaybackSeconds;
    currentPhase = inState.currentPhase;
    progress.iteration.store(inState.iteration, std::memory_order_relaxed);
    progress.bestScore.store(inState.bestScore, std::memory_order_relaxed);
    progress.elapsedPlaybackSeconds.store(accumulatedPlaybackSeconds, std::memory_order_relaxed);
    progress.currentPhase.store(currentPhase, std::memory_order_relaxed);
    progress.processCount.store(inState.processCount, std::memory_order_relaxed);
    progress.totalGenerations.store(inState.totalGenerations, std::memory_order_relaxed);
}

int NoiseShaperLearner::copyBestScoreHistory(float* destination, int maxPoints) const noexcept
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

void NoiseShaperLearner::getLearnedCoefficients(double* outCoeffs, int maxCoefficients) const noexcept
{
    if (outCoeffs == nullptr || maxCoefficients <= 0)
        return;

    const int limit = std::min(kOrder, maxCoefficients);
    for (int i = 0; i < limit; ++i)
        outCoeffs[i] = bestCoefficients[static_cast<size_t>(i)];
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

    {
        const std::scoped_lock<std::mutex> lock(evaluationDispatchMutex);
        evaluationWorkersShouldExit = false;
        completedAuxEvaluationWorkers.store(0, std::memory_order_seq_cst);
        evaluationDispatchSerial.store(0, std::memory_order_seq_cst);
    }

    for (int workerIndex = 1; workerIndex < activeEvaluationWorkerCount; ++workerIndex)
    {
        auto& slot = evaluationWorkers[static_cast<size_t>(workerIndex)];
        if (slot.thread.joinable())
            slot.thread.join();

        slot.thread = std::thread(&NoiseShaperLearner::evaluationWorkerMain, this, workerIndex);
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

    for (int populationIndex = 0; populationIndex < evaluatedCandidates; ++populationIndex)
    {
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

        startEvaluationWorkers();
        engine.pinCurrentThreadToNoiseLearnerCoreIfNeeded();

        SessionSignature activeSession = captureSessionSignature();
        bool resume = pendingResume.exchange(false, std::memory_order_acquire);
        resetLearningSession(activeSession, resume);

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
                progress.latestScore.store(static_cast<float>(bestCandidateScore), std::memory_order_relaxed);
                appendHistoryPoint(static_cast<float>(bestCandidateScore));
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
}

NoiseShaperLearner::SessionSignature NoiseShaperLearner::captureSessionSignature() const noexcept
{
    SessionSignature session;
    session.sampleRateHz = static_cast<int>(engine.currentSampleRate.load(std::memory_order_acquire) + 0.5);
    session.bitDepth = engine.getDitherBitDepth();
    session.adaptiveCoeffBankIndex = engine.currentAdaptiveCoeffBankIndex.load(std::memory_order_acquire);
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
            bestScoreHistory.fill(0.0f);
            historyHead = 0;
        }
        progress.iteration.store(0, std::memory_order_relaxed);
        progress.segmentCount.store(0, std::memory_order_relaxed);
        progress.bestScore.store(0.0f, std::memory_order_relaxed);
        progress.latestScore.store(0.0f, std::memory_order_relaxed);
    }
    
    progress.status.store(Status::WaitingForAudio, std::memory_order_release);

    const double sessionSampleRate = session.sampleRateHz > 0
        ? static_cast<double>(session.sampleRateHz)
        : AudioEngine::getAdaptiveSampleRateBankHz(session.adaptiveCoeffBankIndex);
    const int sessionBitDepth = session.bitDepth > 0
        ? session.bitDepth
        : kAdaptiveBitDepthValues[0]; // Fallback to 16-bit if not set

    configureEvaluationContexts(sessionSampleRate);

    State savedState;
    if (resume && engine.getAdaptiveNoiseShaperState(session.adaptiveCoeffBankIndex, savedState) && savedState.iteration > 0)
    {
        setState(savedState);
    }
    else
    {
        double initialCoefficients[kOrder] = {};
        engine.getAdaptiveCoefficientsForSampleRateAndBitDepth(sessionSampleRate, sessionBitDepth, initialCoefficients, kOrder);
        optimizer.initFromParcor(initialCoefficients);

        for (int i = 0; i < kOrder; ++i)
            bestCoefficients[static_cast<size_t>(i)] = initialCoefficients[i];
    }
}

void NoiseShaperLearner::drainCaptureQueue(const SessionSignature& session) noexcept
{
    AudioBlock block {};
    while (captureQueue.pop(block))
    {
        if (block.numSamples > 0
            && block.sampleRateHz == session.sampleRateHz
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
    double recentLeft[kRecentSampleRequest] = {};
    double recentRight[kRecentSampleRequest] = {};

    const int maxRequired = kRecentSampleRequest;
    const int copiedSamples = segmentBuffer.copyLatest(recentLeft, recentRight, maxRequired);

    if (copiedSamples < AudioSegment::kLength / 2)
        return 0;

    int segmentCount = 0;
    const int usableSamples = copiedSamples;
    const double segmentSec = static_cast<double>(AudioSegment::kLength) / engine.currentSampleRate.load(std::memory_order_acquire);

    for (int start = 0;
         start + AudioSegment::kLength <= usableSamples
         && segmentCount < kMaxTrainingSegments;
         start += kSegmentHop)
    {
        double sumSquares = 0.0;
        for (int sample = 0; sample < AudioSegment::kLength; ++sample)
        {
            const double leftSample = recentLeft[start + sample];
            const double rightSample = recentRight[start + sample];
            sumSquares += 0.5 * ((leftSample * leftSample) + (rightSample * rightSample));
        }

        const double rms = std::sqrt(sumSquares / static_cast<double>(AudioSegment::kLength));
        if (rms < 1.0e-4)
            continue;

        // BUG FIX: Normalize to a safe RMS level (0.2) to prevent clipping in the quantizer.
        // Removed gain normalization to allow the optimizer to adapt to the actual signal level.
        auto& segment = trainingSegments[segmentCount];
        for (int sample = 0; sample < AudioSegment::kLength; ++sample)
        {
            segment.left[sample] = recentLeft[start + sample];
            segment.right[sample] = recentRight[start + sample];
        }

        ++segmentCount;
    }

    return segmentCount;
}

double NoiseShaperLearner::evaluateCandidate(EvaluationContext& context,
                                             const double* candidateCoefficients,
                                             int numSegments,
                                             int evaluationBitDepth) noexcept
{
    const int safeBitDepth = evaluationBitDepth > 0 ? evaluationBitDepth : 24;
    context.shaper.prepare(safeBitDepth);
    context.shaper.setCoefficients(candidateCoefficients, kOrder);

    double totalScore = 0.0;
    int processedSegments = 0;
    for (int segmentIndex = 0; segmentIndex < numSegments; ++segmentIndex)
    {
        if (stopRequested.load(std::memory_order_acquire))
            break;

        const auto& segment = trainingSegments[segmentIndex];
        context.shaper.reset();
        std::memcpy(context.shapedLeft, segment.left, sizeof(context.shapedLeft));
        std::memcpy(context.shapedRight, segment.right, sizeof(context.shapedRight));
        context.shaper.processStereoBlock(context.shapedLeft,
                                          context.shapedRight,
                                          AudioSegment::kLength,
                                          kOutputHeadroom);

        // BUG FIX: The error must be calculated relative to the headroom-scaled input.
        for (int sample = 0; sample < AudioSegment::kLength; ++sample)
        {
            context.errorLeft[sample] = context.shapedLeft[sample] - (segment.left[sample] * kOutputHeadroom);
            context.errorRight[sample] = context.shapedRight[sample] - (segment.right[sample] * kOutputHeadroom);
        }

        totalScore += context.fftEvaluator.evaluate(context.errorLeft, context.errorRight).compositeScore;
        ++processedSegments;
    }

    if (processedSegments <= 0)
        return std::numeric_limits<double>::max();

    return totalScore / static_cast<double>(processedSegments);
}

void NoiseShaperLearner::publishGenerationResult(const double* coeffs, double score, int evaluatedCandidates) noexcept
{
    for (int i = 0; i < kOrder; ++i)
        bestCoefficients[static_cast<size_t>(i)] = coeffs[i];

    progress.bestScore.store(static_cast<float>(score), std::memory_order_relaxed);
    engine.publishCoeffs(coeffs);

    // Save current state to AudioEngine so it can be persisted
    State currentState;
    getState(currentState);
    
    // Get the current bank index from the engine
    const double sr = engine.getSampleRate();
    const int bd = engine.getDitherBitDepth();
    const auto currentMode = static_cast<LearningMode>(engine.getNoiseShaperLearningProgress().learningMode.load(std::memory_order_relaxed));
    const int bankIndex = AudioEngine::getAdaptiveCoeffBankIndex(sr, bd, currentMode);
    
    engine.setAdaptiveNoiseShaperState(bankIndex, currentState);

    engine.requestAdaptiveAutosave();
}

void NoiseShaperLearner::appendHistoryPoint(float score) noexcept
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
