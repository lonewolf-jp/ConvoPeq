#include <JuceHeader.h>
#include "AudioEngine.h"
#include "NoiseShaperLearner.h"

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_SNAPSHOT)
namespace
{
    inline double absDiffNoLibm(double a, double b) noexcept
    {
        return absNoLibm(a - b);
    }
}
#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_SNAPSHOT)

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

    const auto runtimePublishView = getRuntimePublishView();
    const auto* runtimeGraph = runtimePublishView.graph;
    DSPCore* dsp = isFadingTarget
        ? resolveFadingDSPFromRuntimeWorldOnly(runtimeGraph)
        : resolveActiveDSPFromRuntimeWorldOnly(runtimeGraph);
    if (dsp == nullptr && isFadingTarget)
        dsp = resolveActiveDSPFromRuntimeWorldOnly(runtimeGraph);
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

#endif
