#pragma once

#include <JuceHeader.h>

#include "AudioEngine.h"

class ConvolverSettingsComponent : public juce::Component,
                                   private juce::Timer,
                                   private juce::ComboBox::Listener,
                                   private juce::Button::Listener,
                                   private juce::Slider::Listener
{
public:
    explicit ConvolverSettingsComponent(AudioEngine& engineRef);
    ~ConvolverSettingsComponent() override;

    void resized() override;

private:
    void timerCallback() override;
    void comboBoxChanged(juce::ComboBox* comboBoxThatHasChanged) override;
    void buttonClicked(juce::Button* button) override;
    void sliderValueChanged(juce::Slider* slider) override;

    void syncFromProcessor();

    AudioEngine& engine;

    juce::Label targetFftLabel;
    juce::ComboBox targetFftBox;

    juce::ToggleButton progressiveToggle { "Enable Progressive Upgrade" };

    juce::Label cacheEntriesLabel;
    juce::Slider cacheEntriesSlider;

    juce::TextButton clearCacheButton { "Clear Cache" };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ConvolverSettingsComponent)
};
