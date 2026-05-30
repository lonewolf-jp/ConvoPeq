#include <JuceHeader.h>
#include "AudioEngine.h"

namespace
{
    static_assert(CustomInputOversampler::isLinearPhaseFIR
                  && CustomInputOversampler::isSymmetricUpDown,
                  "Oversampling latency formula assumes symmetric linear-phase FIR with identical up/down taps");

    inline double estimateOversamplingLatencySamplesImpl(int oversamplingFactor,
                                                         AudioEngine::OversamplingType oversamplingType,
                                                         double baseSampleRate) noexcept
    {
        if (oversamplingFactor <= 1 || baseSampleRate <= 0.0)
            return 0.0;

        const int numStages = (oversamplingFactor == 8) ? 3 : ((oversamplingFactor == 4) ? 2 : ((oversamplingFactor == 2) ? 1 : 0));
        if (numStages <= 0)
            return 0.0;

        const int* taps = nullptr;
        static constexpr int iirLikeTaps[3] = { 511, 127, 31 };
        static constexpr int linearPhaseTaps[3] = { 1023, 255, 63 };
        taps = (oversamplingType == AudioEngine::OversamplingType::LinearPhase) ? linearPhaseTaps : iirLikeTaps;

        double totalLatencyBaseSamples = 0.0;
        for (int stage = 0; stage < numStages; ++stage)
        {
            const double stageRate = baseSampleRate * static_cast<double>(1 << (stage + 1));
            const double groupDelaySamplesAtStageRate = static_cast<double>(taps[stage] - 1); // up + down
            const double delayBaseSamples = groupDelaySamplesAtStageRate * (baseSampleRate / stageRate);
            totalLatencyBaseSamples += delayBaseSamples;
        }

        return totalLatencyBaseSamples;
    }
}

[[nodiscard]] double AudioEngine::getProcessingSampleRate() const
{
    const double sr = convo::consumeAtomic(currentSampleRate, std::memory_order_acquire);
    if (sr <= 0.0) return 0.0;

    int factor = convo::consumeAtomic(manualOversamplingFactor, std::memory_order_acquire);
    int actualFactor = 1;

    if (factor > 0)
    {
        if (factor == 1 || factor == 2 || factor == 4 || factor == 8)
            actualFactor = factor;
    }
    else
    {
        if (sr <= 96000.0)       actualFactor = 8;
        else if (sr <= 192000.0) actualFactor = 4;
        else if (sr <= 384000.0) actualFactor = 2;
        else                     actualFactor = 1;
    }

    int maxFactor = 1;
    if (sr <= 96000.0)       maxFactor = 8;
    else if (sr <= 192000.0) maxFactor = 4;
    else if (sr <= 384000.0) maxFactor = 2;

    actualFactor = std::min(actualFactor, maxFactor);

    return sr * static_cast<double>(actualFactor);
}

[[nodiscard]] int AudioEngine::getCurrentLatencySamples() const
{
    return getCurrentLatencyBreakdown().totalLatencyBaseRateSamples;
}

[[nodiscard]] int AudioEngine::getTotalLatencySamples() const
{
    return getCurrentLatencyBreakdown().totalLatencyBaseRateSamples;
}

[[nodiscard]] AudioEngine::LatencyBreakdown AudioEngine::getCurrentLatencyBreakdown() const
{
    LatencyBreakdown breakdown;

    const auto* publishedWorld = RuntimePublicationCoordinator::observeWorldHandle(runtimeStore);
    const auto* runtimeGraph = (publishedWorld != nullptr) ? &publishedWorld->graph : nullptr;
    auto* dsp = (runtimeGraph != nullptr)
        ? static_cast<DSPCore*>(runtimeGraph->activeNode)
        : nullptr;
    if (dsp == nullptr)
        return breakdown;

    const int osFactor = static_cast<int>(dsp->oversamplingFactor);
    const int safeOsFactor = std::max(1, osFactor);

    const auto toBaseRateSamples = [safeOsFactor](int processingRateSamples) -> int
    {
        return juce::jmax(0,
            static_cast<int>(std::lround(static_cast<double>(processingRateSamples)
                                         / static_cast<double>(safeOsFactor))));
    };

    breakdown.oversamplingLatencyBaseRateSamples = juce::jmax(0,
        static_cast<int>(std::lround(estimateOversamplingLatencySamples(
            safeOsFactor,
            dsp->activeOversamplingType,
            convo::consumeAtomic(currentSampleRate, std::memory_order_acquire)))));

    if (!convo::consumeAtomic(convBypassActive, std::memory_order_acquire))
    {
        auto convBreakdown = dsp->convolverRt().getLatencyBreakdown();

        if (convBreakdown.algorithmLatencySamples == 0 &&
            convBreakdown.irPeakLatencySamples == 0 &&
            convBreakdown.totalLatencySamples == 0)
        {
            convBreakdown = uiConvolverProcessor.getLatencyBreakdown();
        }

        breakdown.convolverAlgorithmLatencyBaseRateSamples = toBaseRateSamples(convBreakdown.algorithmLatencySamples);
        breakdown.convolverIRPeakLatencyBaseRateSamples = toBaseRateSamples(convBreakdown.irPeakLatencySamples);
        breakdown.convolverTotalLatencyBaseRateSamples = toBaseRateSamples(convBreakdown.totalLatencySamples);
    }

    breakdown.totalLatencyBaseRateSamples = juce::jmax(0,
        breakdown.oversamplingLatencyBaseRateSamples
      + breakdown.convolverTotalLatencyBaseRateSamples);

    return breakdown;
}

double AudioEngine::estimateOversamplingLatencySamples(int oversamplingFactor,
                                                       OversamplingType oversamplingType,
                                                       double baseSampleRate) noexcept
{
    return estimateOversamplingLatencySamplesImpl(oversamplingFactor, oversamplingType, baseSampleRate);
}
