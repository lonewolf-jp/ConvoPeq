//============================================================================
#pragma once
// MainWindow.h  ── v0.1 (JUCE 8.0.12対応)
//
// Main application window - JUCE standard implementation
// Follows AudioPluginHost and ZLEqualizer patterns
//============================================================================

#include <JuceHeader.h>

#include "AudioEngine.h"
#include "ConvolverControlPanel.h"
#include "EQControlPanel.h"
#include "SpectrumAnalyzerComponent.h"
#include "DeviceSettings.h"
#include "AsioBlacklist.h"

class MainWindow : public juce::DocumentWindow,
                   private juce::Timer,
                   private juce::ChangeListener
{
public:
    explicit MainWindow (const juce::String& name);
    ~MainWindow() override;

    void closeButtonPressed() override;

private:
    void resized() override;
    void timerCallback() override;
    void changeListenerCallback (juce::ChangeBroadcaster* source) override;

    void createUIComponents();
    void loadSettings();
    void toggleDeviceSelector();
    void savePreset();
    void loadPreset();
    void launchFileChooser(bool isSaving);
    void showAboutDialog();

    juce::AudioDeviceManager audioDeviceManager;
    juce::AudioSourcePlayer audioSourcePlayer;
    AudioEngine audioEngine;
    AsioBlacklist asioBlacklist;

    std::unique_ptr<ConvolverControlPanel> convolverPanel;
    std::unique_ptr<EQControlPanel> eqPanel;
    std::unique_ptr<SpectrumAnalyzerComponent> specAnalyzer;
    std::unique_ptr<DeviceSettings> deviceSettings;

    juce::TextButton showDeviceSelectorButton;
    juce::ToggleButton eqBypassButton;
    juce::ToggleButton convolverBypassButton;
    juce::TextButton orderButton;
    juce::TextButton saveButton;
    juce::TextButton loadButton;
    juce::TextButton aboutButton;
    juce::Label cpuUsageLabel;
    std::unique_ptr<juce::DocumentWindow> settingsWindow;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MainWindow)
};
