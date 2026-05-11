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

    const auto* runtimePublish = getRuntimePublishState();
    DSPCore* dsp = isFadingTarget
        ? resolveFadingDSPFromRuntimePublish(runtimePublish)
        : resolveCurrentDSPFromRuntimePublish(runtimePublish);
    if (dsp == nullptr && isFadingTarget)
        dsp = resolveCurrentDSPFromRuntimePublish(runtimePublish);
    if (dsp == nullptr)
    {
        applySafeSilentFallback(bufferToFill);
        return;
    }

    const uint64_t hash = snap->eqCoeffHash;
    if (hash != debugLastAppliedEqHash.load(std::memory_order_relaxed))
    {
        debugLastAppliedEqHash.store(hash, std::memory_order_relaxed);
        debugAppliedEqHashVersion.fetch_add(1u, std::memory_order_relaxed);
    }

    const EngineParameterSnapshot parameterSnapshot = captureAudioThreadParameterSnapshot(snap, isFadingTarget);

    if (!isFadingTarget)
    {
        eqBypassActive.store(parameterSnapshot.eqBypassed, std::memory_order_relaxed);
        convBypassActive.store(parameterSnapshot.convBypassed, std::memory_order_relaxed);
    }
    DSPCore::ProcessingState procState = buildAudioThreadProcessingState(dsp, parameterSnapshot);

    std::atomic<float> fadingInputMeter { 0.0f };
    std::atomic<float> fadingOutputMeter { 0.0f };
    auto& inMeter = isFadingTarget ? fadingInputMeter : inputLevelLinear;
    auto& outMeter = isFadingTarget ? fadingOutputMeter : outputLevelLinear;
    dsp->process(bufferToFill, analyzerFifo, inMeter, outMeter, procState);
}

#endif
