#include "MixedPhaseOptimizationComponent.h"

namespace convo {

MixedPhaseOptimizationComponent::MixedPhaseOptimizationComponent(ConvolverProcessor& p)
    : processor(p),
      progressBar(progressValue)
{
    addAndMakeVisible(progressBar);
    addAndMakeVisible(statusLabel);
    addAndMakeVisible(infoLabel);
    addAndMakeVisible(closeButton);

    statusLabel.setText("Impulse Response Optimization", juce::dontSendNotification);
    statusLabel.setFont(juce::Font(juce::FontOptions(18.0f, juce::Font::bold)));
    statusLabel.setJustificationType(juce::Justification::centred);

    infoLabel.setText("Calculating Mixed Phase Allpass...", juce::dontSendNotification);
    infoLabel.setJustificationType(juce::Justification::centred);

    closeButton.setButtonText("Close");
    closeButton.onClick = [this] {
        if (auto* window = findParentComponentOfClass<juce::DocumentWindow>())
            window->closeButtonPressed();
    };

    processor.addChangeListener(this);
    startTimer(100); // 100ms ごとに進捗をポーリング (ChangeBroadcaster 経由でも更新するが、念のため)
}

MixedPhaseOptimizationComponent::~MixedPhaseOptimizationComponent()
{
    processor.removeChangeListener(this);
}

void MixedPhaseOptimizationComponent::paint(juce::Graphics& g)
{
    g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
}

void MixedPhaseOptimizationComponent::resized()
{
    auto area = getLocalBounds().reduced(20);
    statusLabel.setBounds(area.removeFromTop(30));
    area.removeFromTop(10);
    infoLabel.setBounds(area.removeFromTop(20));
    area.removeFromTop(10);
    progressBar.setBounds(area.removeFromTop(30));
    area.removeFromBottom(10);
    closeButton.setBounds(area.removeFromBottom(30).withSizeKeepingCentre(100, 30));
}

void MixedPhaseOptimizationComponent::changeListenerCallback(juce::ChangeBroadcaster* source)
{
    if (source == &processor)
    {
        updateStatus();
    }
}

void MixedPhaseOptimizationComponent::timerCallback()
{
    updateStatus();
}

void MixedPhaseOptimizationComponent::updateStatus()
{
    progressValue = processor.getLoadProgress();
    if (progressValue != lastProgress)
    {
        lastProgress = progressValue;

        if (progressValue >= 1.0)
        {
            infoLabel.setText("Optimization Complete", juce::dontSendNotification);
            closeButton.setEnabled(true);
        }
        else if (progressValue < 0.0)
        {
            infoLabel.setText("Optimization Failed", juce::dontSendNotification);
            closeButton.setEnabled(true);
        }
        else
        {
            infoLabel.setText(juce::String::formatted("Optimizing... %.1f%%", progressValue * 100.0), juce::dontSendNotification);
            closeButton.setEnabled(false);
        }
    }
}

} // namespace convo
