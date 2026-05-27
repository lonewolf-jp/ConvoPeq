#include <JuceHeader.h>
#include "AudioEngine.h"
#include "NoiseShaperLearner.h"

namespace
{
    inline double absDiffNoLibm(double a, double b) noexcept
    {
        return absNoLibm(a - b);
    }
}

void AudioEngine::processWithSnapshot(const juce::AudioSourceChannelInfo& bufferToFill,
                                      const convo::GlobalSnapshot* snap,
                                      bool isFadingTarget,
                                      const convo::RuntimeGraph* runtimeGraphHint)
{
    ASSERT_AUDIO_THREAD();

    if (snap == nullptr)
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    const auto* runtimeGraph = runtimeGraphHint;
    if (runtimeGraph == nullptr)
    {
        const auto runtimeReadView = readAudioRuntimeView();
        runtimeGraph = getRuntimeGraph(runtimeReadView);
    }
    DSPCore* dsp = isFadingTarget
        ? resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeGraph)
        : resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeGraph);
    if (dsp == nullptr && isFadingTarget)
        dsp = resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeGraph);
    if (dsp == nullptr)
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    const double engineSampleRate = (runtimeGraph != nullptr) ? runtimeGraph->sampleRate : 0.0;
    if (engineSampleRate <= 0.0
        || absDiffNoLibm(dsp->sampleRate, engineSampleRate) > 1e-6)
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    const EngineParameterSnapshot parameterSnapshot = captureAudioThreadParameterSnapshot(snap, isFadingTarget);

    DSPCore::ProcessingState procState = buildAudioThreadProcessingState(dsp, parameterSnapshot);

    auto* inMeter = isFadingTarget ? nullptr : &inputLevelLinear;
    auto* outMeter = isFadingTarget ? nullptr : &outputLevelLinear;
    dsp->process(bufferToFill, analyzerFifo, inMeter, outMeter, procState);
}
