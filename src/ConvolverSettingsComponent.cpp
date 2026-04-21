#include "ConvolverSettingsComponent.h"

ConvolverSettingsComponent::ConvolverSettingsComponent(AudioEngine& engineRef)
    : engine(engineRef)
{
    setOpaque(true);

    targetFftLabel.setText("Target FFT", juce::dontSendNotification);
    targetFftLabel.setJustificationType(juce::Justification::centredRight);
    addAndMakeVisible(targetFftLabel);

    targetFftBox.addItem("512", 1);
    targetFftBox.addItem("1024", 2);
    targetFftBox.addItem("2048", 3);
    targetFftBox.addItem("4096", 4);
    targetFftBox.addListener(this);
    addAndMakeVisible(targetFftBox);

    progressiveToggle.addListener(this);
    addAndMakeVisible(progressiveToggle);

    cacheEntriesLabel.setText("Max Cache Entries", juce::dontSendNotification);
    cacheEntriesLabel.setJustificationType(juce::Justification::centredRight);
    addAndMakeVisible(cacheEntriesLabel);

    cacheEntriesSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    cacheEntriesSlider.setRange(1.0, 64.0, 1.0);
    cacheEntriesSlider.setNumDecimalPlacesToDisplay(0);
    cacheEntriesSlider.addListener(this);
    addAndMakeVisible(cacheEntriesSlider);

    clearCacheButton.addListener(this);
    addAndMakeVisible(clearCacheButton);

    setSize(420, 150);
    syncFromProcessor();
    startTimerHz(5);
}

ConvolverSettingsComponent::~ConvolverSettingsComponent()
{
    targetFftBox.removeListener(this);
    progressiveToggle.removeListener(this);
    cacheEntriesSlider.removeListener(this);
    clearCacheButton.removeListener(this);
}

void ConvolverSettingsComponent::paint(juce::Graphics& g)
{
    g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
}

void ConvolverSettingsComponent::resized()
{
    auto area = getLocalBounds().reduced(10);
    const int rowH = 28;
    const int labelW = 130;

    auto row1 = area.removeFromTop(rowH);
    targetFftLabel.setBounds(row1.removeFromLeft(labelW));
    row1.removeFromLeft(8);
    targetFftBox.setBounds(row1.removeFromLeft(120));

    area.removeFromTop(8);
    progressiveToggle.setBounds(area.removeFromTop(rowH));

    area.removeFromTop(8);
    auto row3 = area.removeFromTop(rowH);
    cacheEntriesLabel.setBounds(row3.removeFromLeft(labelW));
    row3.removeFromLeft(8);
    cacheEntriesSlider.setBounds(row3.removeFromLeft(200));

    area.removeFromTop(8);
    clearCacheButton.setBounds(area.removeFromTop(rowH).removeFromLeft(140));
}

void ConvolverSettingsComponent::timerCallback()
{
    syncFromProcessor();
}

void ConvolverSettingsComponent::syncFromProcessor()
{
    auto& convolver = engine.getConvolverProcessor();

    const int target = convolver.getTargetUpgradeFFTSize();
    int selectedId = 4;
    if (target <= 512) selectedId = 1;
    else if (target <= 1024) selectedId = 2;
    else if (target <= 2048) selectedId = 3;

    if (!targetFftBox.isPopupActive())
        targetFftBox.setSelectedId(selectedId, juce::dontSendNotification);

    progressiveToggle.setToggleState(convolver.isProgressiveUpgradeEnabled(), juce::dontSendNotification);

    if (!cacheEntriesSlider.isMouseButtonDown())
        cacheEntriesSlider.setValue(static_cast<double>(convolver.getMaxCacheEntries()), juce::dontSendNotification);
}

void ConvolverSettingsComponent::comboBoxChanged(juce::ComboBox* comboBoxThatHasChanged)
{
    if (comboBoxThatHasChanged != &targetFftBox)
        return;

    int fft = 4096;
    switch (targetFftBox.getSelectedId())
    {
        case 1: fft = 512; break;
        case 2: fft = 1024; break;
        case 3: fft = 2048; break;
        default: break;
    }

    engine.getConvolverProcessor().setTargetUpgradeFFTSize(fft);
}

void ConvolverSettingsComponent::buttonClicked(juce::Button* button)
{
    auto& convolver = engine.getConvolverProcessor();

    if (button == &clearCacheButton)
    {
        convolver.clearCache();
    }
    else if (button == &progressiveToggle)
    {
        convolver.setEnableProgressiveUpgrade(progressiveToggle.getToggleState());
    }
}

void ConvolverSettingsComponent::sliderValueChanged(juce::Slider* slider)
{
    if (slider == &cacheEntriesSlider)
        engine.getConvolverProcessor().setMaxCacheEntries(static_cast<size_t>(juce::roundToInt(cacheEntriesSlider.getValue())));
}
