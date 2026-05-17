#include <JuceHeader.h>
#include "AudioEngine.h"
#include "NoiseShaperLearner.h"

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_SNAPSHOT)

void AudioEngine::processWithSnapshot(const juce::AudioSourceChannelInfo& bufferToFill,
                                      const convo::GlobalSnapshot* snap,
                                      bool isFadingTarget)
{
    ASSERT_AUDIO_THREAD();

    if (snap == nullptr)
    {
        applySafeSilentFallback(bufferToFill);
        return;
    }

    const auto runtimePublishView = getRuntimePublishView();
    const auto* runtimeGraph = runtimePublishView.graph;
    DSPCore* dsp = isFadingTarget
        ? resolveFadingDSPFromRuntimePublish(runtimeGraph)
        : resolveCurrentDSPFromRuntimePublish(runtimeGraph);
    if (dsp == nullptr && isFadingTarget)
        dsp = resolveCurrentDSPFromRuntimePublish(runtimeGraph);
    if (dsp == nullptr)
    {
        applySafeSilentFallback(bufferToFill);
        return;
    }

    const EngineParameterSnapshot parameterSnapshot = captureAudioThreadParameterSnapshot(snap, isFadingTarget);

    DSPCore::ProcessingState procState = buildAudioThreadProcessingState(dsp, parameterSnapshot);

    std::atomic<float> fadingInputMeter { 0.0f };
    std::atomic<float> fadingOutputMeter { 0.0f };
    auto& inMeter = isFadingTarget ? fadingInputMeter : inputLevelLinear;
    auto& outMeter = isFadingTarget ? fadingOutputMeter : outputLevelLinear;
    dsp->process(bufferToFill, analyzerFifo, inMeter, outMeter, procState);
}

#endif
