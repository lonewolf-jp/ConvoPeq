//============================================================================
#pragma once
// ConvolverControlPanel.h  ── v0.2 (JUCE 8.0.12対応)
//
// Convolverコントロールパネル
//
// UI elements:
//   - Load IR ボタン
//   - Dry/Wet Mix スライダー
//   - 位相選択 (Linear/Minimum)
//   - IR波形表示
//   - IR情報表示
//============================================================================

#include <JuceHeader.h>
#include "AudioEngine.h"
#include "AudioEngine.h" // AudioEngineへの参照を提供

class ConvolverControlPanel : public juce::Component,
                              private juce::Button::Listener,
                              private juce::Slider::Listener
{
public:
    explicit ConvolverControlPanel(AudioEngine& audioEngine);
    ~ConvolverControlPanel() override;

    void paint(juce::Graphics& g) override;
    void resized() override;

    //----------------------------------------------------------
    // UI更新
    //----------------------------------------------------------
    void updateIRInfo();

private:
    AudioEngine& engine;

    //----------------------------------------------------------
    // UIコンポーネント
    //----------------------------------------------------------
    juce::TextButton loadIRButton{"Load IR..."};
    juce::ComboBox phaseChoiceBox;

    juce::Slider mixSlider;
    juce::Label mixLabel;

    juce::Slider smoothingTimeSlider;
    juce::Label smoothingTimeLabel;

    juce::Slider irLengthSlider;
    juce::Label irLengthLabel;

    juce::Label irInfoLabel;

    //----------------------------------------------------------
    // イベントハンドラ
    //----------------------------------------------------------
    void buttonClicked(juce::Button* button) override;
    void sliderValueChanged(juce::Slider* slider) override;

    void updateWaveformPath();

    juce::Path waveformPath;
    juce::Rectangle<int> waveformArea;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ConvolverControlPanel)
};
