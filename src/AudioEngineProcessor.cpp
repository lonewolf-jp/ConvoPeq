//============================================================================
#include "AudioEngineProcessor.h"

AudioEngineProcessor::AudioEngineProcessor(AudioEngine& engineRef)
    : juce::AudioProcessor(BusesProperties()
                           .withInput("Input", juce::AudioChannelSet::stereo(), true)
                           .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      audioEngine(engineRef)
{
}

const juce::String AudioEngineProcessor::getName() const
{
    return "ConvoPeqEngine";
}

bool AudioEngineProcessor::acceptsMidi() const { return false; }
bool AudioEngineProcessor::producesMidi() const { return false; }
bool AudioEngineProcessor::isMidiEffect() const { return false; }
double AudioEngineProcessor::getTailLengthSeconds() const { return 0.0; }

int AudioEngineProcessor::getNumPrograms() { return 1; }
int AudioEngineProcessor::getCurrentProgram() { return 0; }
void AudioEngineProcessor::setCurrentProgram(int) {}
const juce::String AudioEngineProcessor::getProgramName(int) { return {}; }
void AudioEngineProcessor::changeProgramName(int, const juce::String&) {}

void AudioEngineProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    audioEngine.prepareToPlay(samplesPerBlock, sampleRate);
}

void AudioEngineProcessor::releaseResources()
{
    audioEngine.releaseResources();
}

bool AudioEngineProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    const auto inSet = layouts.getMainInputChannelSet();
    const auto outSet = layouts.getMainOutputChannelSet();

    if (outSet.isDisabled())
        return false;
    if (outSet.size() < 1 || outSet.size() > 2)
        return false;
    if (!inSet.isDisabled() && (inSet.size() < 1 || inSet.size() > 2))
        return false;

    return true;
}

void AudioEngineProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::AudioSourceChannelInfo info(&buffer, 0, buffer.getNumSamples());
    audioEngine.getNextAudioBlock(info);
}

void AudioEngineProcessor::processBlock(juce::AudioBuffer<double>& buffer, juce::MidiBuffer&)
{
    audioEngine.processBlockDouble(buffer);
}

bool AudioEngineProcessor::hasEditor() const { return false; }
juce::AudioProcessorEditor* AudioEngineProcessor::createEditor() { return nullptr; }

void AudioEngineProcessor::getStateInformation(juce::MemoryBlock&) {}
void AudioEngineProcessor::setStateInformation(const void*, int) {}

