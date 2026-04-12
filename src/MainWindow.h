//============================================================================
#pragma once
// MainWindow.h ── v0.2 (JUCE 8.0.12対応)
//
// メインアプリケーションウィンドウ - JUCE標準実装
// AudioPluginHostおよびZLEqualizerのパターンに準拠
//============================================================================

#include <JuceHeader.h>

#include "AudioEngine.h"
#include "AudioEngineProcessor.h"
#include "ConvolverControlPanel.h"
#include "EQControlPanel.h"
#include "SpectrumAnalyzerComponent.h"
#include "DeviceSettings.h"
#include "AsioBlacklist.h"

class MainWindow : public juce::DocumentWindow,
                   private juce::Timer,
                   private juce::ChangeListener,
                   private juce::Label::Listener
{
public:
    explicit MainWindow (const juce::String& name);
    ~MainWindow() override;

    void closeButtonPressed() override;
    AudioEngine* getAudioEngine() noexcept { return &audioEngine; }
    const AudioEngine* getAudioEngine() const noexcept { return &audioEngine; }

private:
    struct DownwardComboLookAndFeel : public juce::LookAndFeel_V4
    {
        juce::PopupMenu::Options getOptionsForComboBoxPopupMenu (juce::ComboBox& box, juce::Label& label) override
        {
            return juce::LookAndFeel_V4::getOptionsForComboBoxPopupMenu (box, label)
                .withPreferredPopupDirection (juce::PopupMenu::Options::PopupDirection::downwards);
        }
    };

    void resized() override;
    void timerCallback() override;
    void changeListenerCallback (juce::ChangeBroadcaster* source) override;
    void labelTextChanged(juce::Label* label) override;
    void editorShown(juce::Label* label, juce::TextEditor& editor) override;
    void orderModeBoxChanged();

    void createUIComponents();
    void loadSettings();
    void toggleDeviceSelector();
    void savePreset();
    void loadPreset();
    void launchFileChooser(bool isSaving);
    void showAboutDialog();

    AsioBlacklist asioBlacklist;
    AudioEngine audioEngine;
    std::unique_ptr<AudioEngineProcessor> audioEngineProcessor;
    juce::AudioProcessorPlayer audioProcessorPlayer;
    juce::AudioDeviceManager audioDeviceManager;

    std::unique_ptr<ConvolverControlPanel> convolverPanel;
    std::unique_ptr<EQControlPanel> eqPanel;
    std::unique_ptr<SpectrumAnalyzerComponent> specAnalyzer;
    std::unique_ptr<DeviceSettings> deviceSettings;

    juce::TextButton showDeviceSelectorButton;
    DownwardComboLookAndFeel orderModeLookAndFeel;
    juce::ComboBox orderModeBox;
    juce::TextButton saveButton;
    juce::TextButton loadButton;
    juce::TextButton aboutButton;
    juce::ToggleButton softClipButton;
    juce::Label saturationValueLabel;
    juce::Label saturationLabel;
    juce::Label latencyLabel;
    juce::Label cpuUsageLabel;
    std::unique_ptr<juce::DocumentWindow> settingsWindow;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MainWindow)
};
