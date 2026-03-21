#pragma once

#include <JuceHeader.h>

#include <array>

#include "AudioEngine.h"

class NoiseShaperLearningComponent : public juce::Component,
                                     private juce::Timer
{
public:
    explicit NoiseShaperLearningComponent(AudioEngine& engine);
    ~NoiseShaperLearningComponent() override = default;

    void paint(juce::Graphics& g) override;
    void resized() override;

private:
    class ProgressGraph : public juce::Component
    {
    public:
        void setHistory(const float* values, int count);
        void paint(juce::Graphics& g) override;

    private:
        std::array<float, NoiseShaperLearner::kMaxHistoryPoints> history {};
        int historySize = 0;
    };

    void timerCallback() override;
    void refreshFromEngine();
    static juce::String statusToText(NoiseShaperLearner::Status status);
    static juce::String formatScore(float score);

    AudioEngine& audioEngine;
    ProgressGraph progressGraph;

    juce::TextButton startButton { "Start learning" };
    juce::TextButton stopButton { "Stop learning" };
    juce::ComboBox modeComboBox;
    juce::Label modeLabel { "Mode", "Learning mode:" };
    juce::Label statusLabel;
    juce::Label orderLabel;
    juce::Label iterationLabel;
    juce::Label processCountLabel;
    juce::Label segmentCountLabel;
    juce::Label bestScoreLabel;
    juce::Label latestScoreLabel;
    juce::Label elapsedLabel;
    juce::Label phaseLabel;
    juce::Label messageLabel;

    std::array<float, NoiseShaperLearner::kMaxHistoryPoints> historyBuffer {};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(NoiseShaperLearningComponent)
};
