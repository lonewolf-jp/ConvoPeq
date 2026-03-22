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

    // recent scoreグラフ拡張のため推奨サイズをさらに拡大（グラフ2倍分）
    int getRecommendedHeight() const noexcept { return 920; } // 760→920（+160px）

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
    juce::TextButton resumeButton { "Resume learning" };
    juce::ComboBox modeComboBox;
    juce::Label modeLabel { "Mode", "Learning mode:" };
    juce::Label statusLabel;
    juce::Label orderLabel;
    juce::Label sampleRateAndBitDepthLabel;
    juce::Label iterationLabel;
    juce::Label processCountLabel;
    juce::Label segmentCountLabel;
    juce::Label bestScoreLabel;
    juce::Label latestScoreLabel;
    juce::Label elapsedLabel;
    juce::Label phaseLabel;
    juce::Label messageLabel;

    std::array<float, NoiseShaperLearner::kMaxHistoryPoints> historyBuffer {};

    class PeriodicSaver : public juce::Timer
    {
    public:
        PeriodicSaver(AudioEngine& engine) : audioEngine(engine) {}
        void timerCallback() override
        {
            if (audioEngine.isNoiseShaperLearning())
                audioEngine.requestAdaptiveAutosave();
        }
    private:
        AudioEngine& audioEngine;
    };
    PeriodicSaver periodicSaver;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(NoiseShaperLearningComponent)
};
