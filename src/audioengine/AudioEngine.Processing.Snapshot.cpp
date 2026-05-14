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

    const auto* world = getRuntimePublishWorld();
    const auto* engineRuntime = getEngineRuntimeState(world);
    const auto* runtimeGraph = getRuntimeGraphState(world);
    DSPCore* dsp = isFadingTarget
        ? resolveFadingDSPFromRuntimePublish(runtimeGraph, engineRuntime)
        : resolveCurrentDSPFromRuntimePublish(runtimeGraph, engineRuntime);
    if (dsp == nullptr && isFadingTarget)
        dsp = resolveCurrentDSPFromRuntimePublish(runtimeGraph, engineRuntime);
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
    auto& executionState = isFadingTarget ? dspExecutionStateFading : dspExecutionStateCurrent;
    syncEqAgcTableViewFromRuntimeGraph(executionState, runtimeGraph);
    dsp->processV2(bufferToFill, analyzerFifo, inMeter, outMeter, runtimeGraph, executionState, procState);
}

#endif
