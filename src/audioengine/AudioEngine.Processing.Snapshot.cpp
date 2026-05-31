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
                                      bool isFadingTarget)
{
    ASSERT_AUDIO_THREAD();

    if (snap == nullptr)
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    const auto runtimeReadView = readAudioRuntimeView();
    const auto& runtimePublishView = runtimeReadView.runtimePublish;
    const auto* runtimeWorld = runtimeReadView.runtimeWorld;
    if (runtimeWorld == nullptr)
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }
    DSPCore* dsp = isFadingTarget
        ? (runtimeWorld->topology.hasFadingRuntime
            ? static_cast<DSPCore*>(runtimePublishView.transition.next)
            : nullptr)
        : static_cast<DSPCore*>(runtimePublishView.transition.current);
    if (dsp == nullptr && isFadingTarget)
        dsp = static_cast<DSPCore*>(runtimePublishView.transition.current);
    if (dsp == nullptr)
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    const double engineSampleRate = runtimePublishView.sampleRateHz;
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
