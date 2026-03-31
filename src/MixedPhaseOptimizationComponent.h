#pragma once

#include <JuceHeader.h>
#include "ConvolverProcessor.h"

namespace convo {

/**
    Impulse Response の最適化（Mixed Phase 変換など）の進捗を表示するウィンドウ
*/
class MixedPhaseOptimizationComponent : public juce::Component,
                                         private juce::ChangeListener,
                                         private juce::Timer
{
public:
    MixedPhaseOptimizationComponent(ConvolverProcessor& processor);
    ~MixedPhaseOptimizationComponent() override;

    void paint(juce::Graphics& g) override;
    void resized() override;

private:
    void changeListenerCallback(juce::ChangeBroadcaster* source) override;
    void timerCallback() override;

    void updateStatus();

    ConvolverProcessor& processor;

    double progressValue = 0.0;
    juce::ProgressBar progressBar;
    juce::Label statusLabel;
    juce::Label infoLabel;
    juce::TextButton closeButton;

    double lastProgress = -1.0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MixedPhaseOptimizationComponent)
};

/**
    MixedPhaseOptimizationComponent を保持するウィンドウ
*/
class MixedPhaseOptimizationWindow : public juce::DocumentWindow
{
public:
    MixedPhaseOptimizationWindow(const juce::String& name, ConvolverProcessor& processor)
        : DocumentWindow(name,
                         juce::Desktop::getInstance().getDefaultLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId),
                         allButtons)
    {
        setUsingNativeTitleBar(true);
        setContentOwned(new MixedPhaseOptimizationComponent(processor), true);
        setResizable(false, false);
        setAlwaysOnTop(true);
        centreWithSize(400, 200);
    }

    void closeButtonPressed() override
    {
        delete this;
    }

private:
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MixedPhaseOptimizationWindow)
};

} // namespace convo
