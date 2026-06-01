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

    const auto runtimeReadHandle = readAudioRuntimeHandle();
    const auto* runtimeWorld = getRuntimeWorldFromReadHandle(runtimeReadHandle);
    if (runtimeWorld == nullptr)
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }
    DSPCore* dsp = isFadingTarget
        ? resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle)
        : resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle);
    if (dsp == nullptr && isFadingTarget)
        dsp = resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle);
    if (dsp == nullptr)
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    const double engineSampleRate = getRuntimeSampleRateHzFromWorld(runtimeReadHandle, 0.0);
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
