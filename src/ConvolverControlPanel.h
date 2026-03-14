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

class ConvolverControlPanel  : public juce::Component,
                              private juce::Button::Listener,
                              private juce::Slider::Listener,
                              private juce::Timer
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

    juce::Slider rebuildDebounceSlider;
    juce::Label rebuildDebounceLabel;

    juce::Label irInfoLabel;

    //----------------------------------------------------------
    // 出力周波数フィルター UI (① コンボルバー最終段の場合に使用)
    //----------------------------------------------------------
    // ハイカットフィルターモード ラベル + ボタン (Sharp / Natural / Soft)
    juce::Label  hcfLabel;
    juce::TextButton hcfSharpButton   { "Sharp"   };
    juce::TextButton hcfNaturalButton { "Natural" };
    juce::TextButton hcfSoftButton    { "Soft"    };

    // ローカットフィルターモード ラベル + ボタン (Natural / Soft)
    juce::Label  lcfLabel;
    juce::TextButton lcfNaturalButton { "Natural" };
    juce::TextButton lcfSoftButton    { "Soft"    };

    // ボタン状態更新ヘルパー
    void updateFilterModeButtons();

    //----------------------------------------------------------
    // イベントハンドラ
    //----------------------------------------------------------
    void buttonClicked(juce::Button* button) override;
    void sliderValueChanged(juce::Slider* slider) override;
    void mouseDown (const juce::MouseEvent& event) override;
    void timerCallback() override;

    void updateWaveformPath();
    void markConvolverParameterDirty();
    void applyPendingConvolverParameters();
    bool hasPendingConvolverParameters() const noexcept;

    juce::Path waveformPath;
    juce::Rectangle<int> waveformArea;

    static constexpr int PARAMETER_RECALC_DEBOUNCE_MS = 3000;
    double lastParameterChangeMs = 0.0;
    bool pendingMixDirty = false;
    bool pendingSmoothingDirty = false;
    bool pendingIrLengthDirty = false;
    float pendingMixValue = 1.0f;
    float pendingSmoothingTimeSec = ConvolverProcessor::SMOOTHING_TIME_DEFAULT_SEC;
    float pendingIrLengthSec = ConvolverProcessor::IR_LENGTH_DEFAULT_SEC;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ConvolverControlPanel)
};
