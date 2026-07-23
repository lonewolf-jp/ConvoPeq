//============================================================================
#pragma once

#include <JuceHeader.h>
#include <atomic>
#include "AudioEngine.h"

class AudioEngineProcessor final : public juce::AudioProcessor
{
public:
    explicit AudioEngineProcessor(AudioEngine& engineRef);
    ~AudioEngineProcessor() override = default;

    const juce::String getName() const override;
    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    bool supportsDoublePrecisionProcessing() const override { return true; }

    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) override;
    void processBlock(juce::AudioBuffer<double>& buffer, juce::MidiBuffer& midiMessages) override;

    bool hasEditor() const override;
    juce::AudioProcessorEditor* createEditor() override;

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

private:
    AudioEngine& audioEngine;
    // C5: Runtime Publish 時に計算・更新するキャッシュされた tail length
    // 他スレッド（JUCE host）から getTailLengthSeconds() が呼ばれる可能性があるため atomic を使用
    // relaxed: 単なるキャッシュ値。他データとの公開順序を要求しないため十分
    std::atomic<double> cachedTailLength { 0.0 };
};

