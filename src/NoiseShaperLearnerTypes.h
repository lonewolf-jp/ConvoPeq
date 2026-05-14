#pragma once

#include <array>
#include <atomic>
#include <cstdint>

#include "audioengine/AtomicAccess.h"

namespace convo {

enum class NoiseShaperLearnerStatus
{
    Idle,
    WaitingForAudio,
    Running,
    Completed,
    Error
};

enum class NoiseShaperLearningMode
{
    Shortest,
    Short,
    Middle,
    Long,
    Ultra,
    Continuous
};

struct NoiseShaperLearnerProgress
{
    std::atomic<int> iteration { 0 };
    std::atomic<std::uint64_t> totalGenerations { 0 };
    std::atomic<int> processCount { 0 };
    std::atomic<int> segmentCount { 0 };
    std::atomic<double> bestScore { 0.0 };
    std::atomic<double> latestScore { 0.0 };
    std::atomic<NoiseShaperLearnerStatus> status { NoiseShaperLearnerStatus::Idle };
    std::atomic<double> elapsedPlaybackSeconds { 0.0 };
    std::atomic<int> currentPhase { 1 };
    std::atomic<int> learningMode { 0 };
};

struct NoiseShaperLearnerState
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
    std::uint64_t totalGenerations = 0;
    int learningMode = 0;
};

struct NoiseShaperLearnerSettings
{
    std::atomic<int> cmaesRestarts { 5 };
    std::atomic<double> coeffSafetyMargin { 0.85 };
    std::atomic<bool> enableStabilityCheck { true };

    NoiseShaperLearnerSettings() = default;

    NoiseShaperLearnerSettings(const NoiseShaperLearnerSettings& other)
        : cmaesRestarts(convo::consumeAtomic(other.cmaesRestarts)),
          coeffSafetyMargin(convo::consumeAtomic(other.coeffSafetyMargin)),
          enableStabilityCheck(convo::consumeAtomic(other.enableStabilityCheck))
    {
    }

    NoiseShaperLearnerSettings& operator=(const NoiseShaperLearnerSettings& other)
    {
        cmaesRestarts = convo::consumeAtomic(other.cmaesRestarts);
        coeffSafetyMargin = convo::consumeAtomic(other.coeffSafetyMargin);
        enableStabilityCheck = convo::consumeAtomic(other.enableStabilityCheck);
        return *this;
    }
};

} // namespace convo
