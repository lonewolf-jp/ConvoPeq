#include <JuceHeader.h>
#include "AudioEngine.h"

void AudioEngine::DSPCore::processToBuffer(const juce::AudioSourceChannelInfo& source,
                                           juce::AudioBuffer<float>& destination,
                                           LockFreeAudioRingBuffer& analyzerFifo,
                                           std::atomic<float>* inputLevelLinear,
                                           std::atomic<float>* outputLevelLinear,
                                           const ProcessingState& state)
{
    const int numSamples = source.numSamples;
    const int numChannels = std::min(2, source.buffer != nullptr ? source.buffer->getNumChannels() : 0);

    if (source.buffer == nullptr || numSamples <= 0 || destination.getNumSamples() < numSamples)
    {
        destination.clear();
        return;
    }

    for (int ch = 0; ch < numChannels; ++ch)
    {
        const float* src = source.buffer->getReadPointer(ch, source.startSample);
        float* dst = destination.getWritePointer(ch, 0);
        juce::FloatVectorOperations::copy(dst, src, numSamples);
    }

    for (int ch = numChannels; ch < destination.getNumChannels(); ++ch)
        destination.clear(ch, 0, numSamples);

    juce::AudioSourceChannelInfo destinationInfo(&destination, 0, numSamples);
    process(destinationInfo, analyzerFifo, inputLevelLinear, outputLevelLinear, state);
}
