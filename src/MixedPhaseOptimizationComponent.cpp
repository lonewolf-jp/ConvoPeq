#include "MixedPhaseOptimizationComponent.h"

namespace convo {

MixedPhaseOptimizationComponent::MixedPhaseOptimizationComponent(ConvolverProcessor& p)
    : processor(p)
{
    addAndMakeVisible(statusLabel);
    addAndMakeVisible(infoLabel);
    addAndMakeVisible(closeButton);

    statusLabel.setText("Impulse Response Optimization", juce::dontSendNotification);
    statusLabel.setFont(juce::Font(juce::FontOptions(18.0f, juce::Font::bold)));
    statusLabel.setJustificationType(juce::Justification::centred);

    infoLabel.setText("Waiting IR", juce::dontSendNotification);
    infoLabel.setJustificationType(juce::Justification::centred);

    closeButton.setButtonText("Close");
    closeButton.onClick = [this] {
        juce::Logger::writeToLog("[MixedPhaseUI] Close button clicked");
        if (auto* window = findParentComponentOfClass<juce::DocumentWindow>())
            window->closeButtonPressed();
    };

    processor.addChangeListener(this);
    startTimerHz(30);
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
    infoLabel.setBounds(area.removeFromTop(30));
    area.removeFromBottom(10);
    closeButton.setBounds(area.removeFromBottom(30).withSizeKeepingCentre(100, 30));
}

void MixedPhaseOptimizationComponent::changeListenerCallback(juce::ChangeBroadcaster* source)
{
    if (source == &processor)
    {
        // State polling is handled in timerCallback().
    }
}

void MixedPhaseOptimizationComponent::timerCallback()
{
    const int state = processor.getMixedPhaseState();

    static int lastState = -1;
    if (state != lastState)
    {
        juce::Logger::writeToLog("[MixedPhaseUI] timerCallback: state = " + juce::String(state));
        lastState = state;
    }

    switch (state)
    {
        case 1:
            infoLabel.setText("Optimizing... (CMA-ES)", juce::dontSendNotification);
            break;
        case 2:
            infoLabel.setText("Optimization Completed", juce::dontSendNotification);
            break;
        default:
            infoLabel.setText("Waiting IR", juce::dontSendNotification);
            break;
    }
}

} // namespace convo
