//============================================================================
// ConvolverControlPanel.cpp  ── v0.2 (JUCE 8.0.12対応)
//
// Convolverコントロールパネルの実装
//============================================================================
#include "ConvolverControlPanel.h"
#include "MixedPhaseOptimizationComponent.h"
#include "ConvolverSettingsComponent.h"
#include <cmath>

namespace
{
int phaseModeToComboId(ConvolverProcessor::PhaseMode mode)
{
    switch (mode)
    {
        case ConvolverProcessor::PhaseMode::AsIs:    return 1;
        case ConvolverProcessor::PhaseMode::Mixed:   return 2;
        case ConvolverProcessor::PhaseMode::Minimum: return 3;
        default:                                      return 1;
    }
}

ConvolverProcessor::PhaseMode comboIdToPhaseMode(int id)
{
    switch (id)
    {
        case 2:  return ConvolverProcessor::PhaseMode::Mixed;
        case 3:  return ConvolverProcessor::PhaseMode::Minimum;
        default: return ConvolverProcessor::PhaseMode::AsIs;
    }
}

class IRAdvancedSettingsComponent : public juce::Component,
                                    private juce::Slider::Listener,
                                    private juce::Timer,
                                    private juce::ComboBox::Listener
{
public:
    explicit IRAdvancedSettingsComponent(AudioEngine& audioEngine)
        : engine(audioEngine)
    {
        auto configureLabel = [](juce::Label& label, const juce::String& text)
        {
            label.setText(text, juce::dontSendNotification);
            label.setJustificationType(juce::Justification::centredRight);
            label.setColour(juce::Label::textColourId, juce::Colours::white);
        };

        configureLabel(irLengthLabel, "IR Length:");
        configureLabel(rebuildLabel, "Rebuild:");
        configureLabel(mixedF1Label, "Mix Start f:");
        configureLabel(mixedF2Label, "Mix End f:");
        configureLabel(mixedTauLabel, "Mix tau:");
        configureLabel(tailModeLabel, "Tail Mode:");
        configureLabel(tailRolloffStartLabel, "Tail Start:");
        configureLabel(tailRolloffStrengthLabel, "Tail Strength:");
        configureLabel(partitionTailLabel, "L1/L2 Mult:");

        irLengthSlider.setSliderStyle(juce::Slider::LinearHorizontal);
        irLengthSlider.setRange(ConvolverProcessor::IR_LENGTH_MIN_SEC, ConvolverProcessor::IR_LENGTH_MAX_SEC, 0.1);
        irLengthSlider.setSkewFactorFromMidPoint(1.5);
        irLengthSlider.setTextValueSuffix(" s");
        irLengthSlider.setNumDecimalPlacesToDisplay(1);
        irLengthSlider.addListener(this);

        rebuildSlider.setSliderStyle(juce::Slider::LinearHorizontal);
        rebuildSlider.setRange(ConvolverProcessor::REBUILD_DEBOUNCE_MIN_MS,
                               ConvolverProcessor::REBUILD_DEBOUNCE_MAX_MS, 10.0);
        rebuildSlider.setSkewFactorFromMidPoint(static_cast<double>(ConvolverProcessor::REBUILD_DEBOUNCE_DEFAULT_MS));
        rebuildSlider.setTextValueSuffix(" ms");
        rebuildSlider.setNumDecimalPlacesToDisplay(0);
        rebuildSlider.addListener(this);

        mixedF1Slider.setSliderStyle(juce::Slider::LinearHorizontal);
        mixedF1Slider.setRange(ConvolverProcessor::MIXED_F1_MIN_HZ,
                               ConvolverProcessor::MIXED_F1_MAX_HZ, 1.0);
        mixedF1Slider.setSkewFactorFromMidPoint(ConvolverProcessor::MIXED_F1_DEFAULT_HZ);
        mixedF1Slider.setTextValueSuffix(" Hz");
        mixedF1Slider.setNumDecimalPlacesToDisplay(0);
        mixedF1Slider.addListener(this);

        mixedF2Slider.setSliderStyle(juce::Slider::LinearHorizontal);
        mixedF2Slider.setRange(ConvolverProcessor::MIXED_F2_MIN_HZ,
                               ConvolverProcessor::MIXED_F2_MAX_HZ, 1.0);
        mixedF2Slider.setSkewFactorFromMidPoint(ConvolverProcessor::MIXED_F2_DEFAULT_HZ);
        mixedF2Slider.setTextValueSuffix(" Hz");
        mixedF2Slider.setNumDecimalPlacesToDisplay(0);
        mixedF2Slider.addListener(this);

        mixedTauSlider.setSliderStyle(juce::Slider::LinearHorizontal);
        mixedTauSlider.setRange(ConvolverProcessor::MIXED_TAU_MIN,
                                ConvolverProcessor::MIXED_TAU_MAX, 1.0);
        mixedTauSlider.setSkewFactorFromMidPoint(48.0);
        mixedTauSlider.setTextValueSuffix(" smp");
        mixedTauSlider.setNumDecimalPlacesToDisplay(0);
        mixedTauSlider.addListener(this);

        tailModeCombo.addItem("Air Absorption (All Layers)", 1);
        tailModeCombo.addItem("Layer Tail Contouring (L1/L2)", 2);
        tailModeCombo.addListener(this);
        tailModeCombo.setTooltip("Air Absorption: Applies to all layers. Layer Tail: applies to L1/L2 only.");

        tailRolloffStartSlider.setSliderStyle(juce::Slider::LinearHorizontal);
        tailRolloffStartSlider.setRange(ConvolverProcessor::TAIL_ROLLOFF_START_MIN_HZ,
                                        ConvolverProcessor::TAIL_ROLLOFF_START_MAX_HZ, 1.0);
        tailRolloffStartSlider.setSkewFactorFromMidPoint(ConvolverProcessor::TAIL_ROLLOFF_START_DEFAULT_HZ);
        tailRolloffStartSlider.setTextValueSuffix(" Hz");
        tailRolloffStartSlider.setNumDecimalPlacesToDisplay(0);
        tailRolloffStartSlider.addListener(this);

        tailRolloffStrengthSlider.setSliderStyle(juce::Slider::LinearHorizontal);
        tailRolloffStrengthSlider.setRange(ConvolverProcessor::TAIL_ROLLOFF_STRENGTH_MIN,
                                           ConvolverProcessor::TAIL_ROLLOFF_STRENGTH_MAX, 0.01);
        tailRolloffStrengthSlider.setNumDecimalPlacesToDisplay(2);
        tailRolloffStrengthSlider.addListener(this);

        partitionTailSlider.setSliderStyle(juce::Slider::LinearHorizontal);
        partitionTailSlider.setRange(ConvolverProcessor::TAIL_PARTITION_STRENGTH_MIN,
                                     ConvolverProcessor::TAIL_PARTITION_STRENGTH_MAX, 0.01);
        partitionTailSlider.setNumDecimalPlacesToDisplay(2);
        partitionTailSlider.addListener(this);

        addAndMakeVisible(irLengthLabel);
        addAndMakeVisible(irLengthSlider);
        addAndMakeVisible(rebuildLabel);
        addAndMakeVisible(rebuildSlider);
        addAndMakeVisible(mixedF1Label);
        addAndMakeVisible(mixedF1Slider);
        addAndMakeVisible(mixedF2Label);
        addAndMakeVisible(mixedF2Slider);
        addAndMakeVisible(mixedTauLabel);
        addAndMakeVisible(mixedTauSlider);
        addAndMakeVisible(tailModeLabel);
        addAndMakeVisible(tailModeCombo);
        addAndMakeVisible(tailRolloffStartLabel);
        addAndMakeVisible(tailRolloffStartSlider);
        addAndMakeVisible(tailRolloffStrengthLabel);
        addAndMakeVisible(tailRolloffStrengthSlider);
        addAndMakeVisible(partitionTailLabel);
        addAndMakeVisible(partitionTailSlider);

        setSize(560, 350);
        syncFromProcessor();
        startTimerHz(8);
    }

    ~IRAdvancedSettingsComponent() override
    {
        irLengthSlider.removeListener(this);
        rebuildSlider.removeListener(this);
        mixedF1Slider.removeListener(this);
        mixedF2Slider.removeListener(this);
        mixedTauSlider.removeListener(this);
        tailRolloffStartSlider.removeListener(this);
        tailRolloffStrengthSlider.removeListener(this);
        partitionTailSlider.removeListener(this);
        tailModeCombo.removeListener(this);
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced(10);
        constexpr int rowH = 32;
        constexpr int labelW = 90;
        constexpr int gap = 8;

        auto placeRow = [&](juce::Label& label, juce::Slider& slider)
        {
            auto row = area.removeFromTop(rowH);
            label.setBounds(row.removeFromLeft(labelW));
            row.removeFromLeft(gap);
            slider.setBounds(row);
            area.removeFromTop(4);
        };

        auto placeComboRow = [&](juce::Label& label, juce::ComboBox& combo)
        {
            auto row = area.removeFromTop(rowH);
            label.setBounds(row.removeFromLeft(labelW));
            row.removeFromLeft(gap);
            combo.setBounds(row);
            area.removeFromTop(4);
        };

        placeRow(irLengthLabel, irLengthSlider);
        placeRow(mixedF1Label, mixedF1Slider);
        placeRow(mixedF2Label, mixedF2Slider);
        placeRow(mixedTauLabel, mixedTauSlider);
        placeComboRow(tailModeLabel, tailModeCombo);
        placeRow(tailRolloffStartLabel, tailRolloffStartSlider);
        placeRow(tailRolloffStrengthLabel, tailRolloffStrengthSlider);
        placeRow(partitionTailLabel, partitionTailSlider);
        placeRow(rebuildLabel, rebuildSlider);
    }

private:
    AudioEngine& engine;
    juce::Label irLengthLabel;
    juce::Slider irLengthSlider;
    juce::Label rebuildLabel;
    juce::Slider rebuildSlider;
    juce::Label mixedF1Label;
    juce::Slider mixedF1Slider;
    juce::Label mixedF2Label;
    juce::Slider mixedF2Slider;
    juce::Label mixedTauLabel;
    juce::Slider mixedTauSlider;
    juce::Label tailModeLabel;
    juce::ComboBox tailModeCombo;
    juce::Label tailRolloffStartLabel;
    juce::Slider tailRolloffStartSlider;
    juce::Label tailRolloffStrengthLabel;
    juce::Slider tailRolloffStrengthSlider;
    juce::Label partitionTailLabel;
    juce::Slider partitionTailSlider;

    void timerCallback() override
    {
        syncFromProcessor();
    }

    void sliderValueChanged(juce::Slider* slider) override
    {
        auto& convolver = engine.getConvolverProcessor();
        if (slider == &irLengthSlider)
        {
            convolver.setIRLengthManualOverride(true);
            convolver.setTargetIRLength(static_cast<float>(irLengthSlider.getValue()));
        }
        else if (slider == &rebuildSlider)
        {
            convolver.setRebuildDebounceMs(static_cast<int>(rebuildSlider.getValue()));
        }
        else if (slider == &mixedF1Slider)
        {
            convolver.setMixedTransitionStartHz(static_cast<float>(mixedF1Slider.getValue()));
        }
        else if (slider == &mixedF2Slider)
        {
            convolver.setMixedTransitionEndHz(static_cast<float>(mixedF2Slider.getValue()));
        }
        else if (slider == &mixedTauSlider)
        {
            convolver.setMixedPreRingTau(static_cast<float>(mixedTauSlider.getValue()));
        }
        else if (slider == &tailRolloffStartSlider)
        {
            convolver.setTailRolloffStartHz(static_cast<float>(tailRolloffStartSlider.getValue()));
        }
        else if (slider == &tailRolloffStrengthSlider)
        {
            convolver.setTailRolloffStrength(static_cast<float>(tailRolloffStrengthSlider.getValue()));
        }
        else if (slider == &partitionTailSlider)
        {
            convolver.setPartitionTailStrength(static_cast<float>(partitionTailSlider.getValue()));
        }
    }

    void comboBoxChanged(juce::ComboBox* comboBoxThatHasChanged) override
    {
        if (comboBoxThatHasChanged != &tailModeCombo)
            return;

        auto& convolver = engine.getConvolverProcessor();
        const int mode = juce::jmax(0, tailModeCombo.getSelectedId() - 1);
        convolver.setTailProcessingMode(mode);

        if (mode == 0)
        {
            convolver.setTailRolloffStartHz(ConvolverProcessor::TAIL_AIR_ROLLOFF_START_DEFAULT_HZ);
            convolver.setTailRolloffStrength(ConvolverProcessor::TAIL_AIR_ROLLOFF_STRENGTH_DEFAULT);
        }
        else
        {
            convolver.setTailRolloffStartHz(ConvolverProcessor::TAIL_LAYER_ROLLOFF_START_DEFAULT_HZ);
            convolver.setTailRolloffStrength(ConvolverProcessor::TAIL_LAYER_ROLLOFF_STRENGTH_DEFAULT);
        }
        updateTailControlsVisibility();
    }

    void syncFromProcessor()
    {
        auto& convolver = engine.getConvolverProcessor();
        if (!irLengthSlider.isMouseButtonDown())
            irLengthSlider.setValue(convolver.getTargetIRLength(), juce::dontSendNotification);
        if (!rebuildSlider.isMouseButtonDown())
            rebuildSlider.setValue(static_cast<double>(convolver.getRebuildDebounceMs()), juce::dontSendNotification);
        if (!mixedF1Slider.isMouseButtonDown())
            mixedF1Slider.setValue(convolver.getMixedTransitionStartHz(), juce::dontSendNotification);
        if (!mixedF2Slider.isMouseButtonDown())
            mixedF2Slider.setValue(convolver.getMixedTransitionEndHz(), juce::dontSendNotification);
        if (!mixedTauSlider.isMouseButtonDown())
            mixedTauSlider.setValue(convolver.getMixedPreRingTau(), juce::dontSendNotification);
        if (!tailRolloffStartSlider.isMouseButtonDown())
            tailRolloffStartSlider.setValue(convolver.getTailRolloffStartHz(), juce::dontSendNotification);
        if (!tailRolloffStrengthSlider.isMouseButtonDown())
            tailRolloffStrengthSlider.setValue(convolver.getTailRolloffStrength(), juce::dontSendNotification);
        if (!partitionTailSlider.isMouseButtonDown())
            partitionTailSlider.setValue(convolver.getPartitionTailStrength(), juce::dontSendNotification);
        if (!tailModeCombo.isPopupActive())
            tailModeCombo.setSelectedId(convolver.getTailProcessingMode() + 1, juce::dontSendNotification);

        const bool mixedEnabled = engine.getConvolverPhaseMode() == ConvolverProcessor::PhaseMode::Mixed;
        mixedF1Slider.setEnabled(mixedEnabled);
        mixedF2Slider.setEnabled(mixedEnabled);
        mixedTauSlider.setEnabled(mixedEnabled);
        mixedF1Label.setEnabled(mixedEnabled);
        mixedF2Label.setEnabled(mixedEnabled);
        mixedTauLabel.setEnabled(mixedEnabled);
        updateTailControlsVisibility();
    }

    void updateTailControlsVisibility()
    {
        const bool showPartitionControls = (tailModeCombo.getSelectedId() == 2);
        partitionTailLabel.setVisible(showPartitionControls);
        partitionTailSlider.setVisible(showPartitionControls);
    }
};
}

//--------------------------------------------------------------
// コンストラクタ
//--------------------------------------------------------------
ConvolverControlPanel::ConvolverControlPanel(AudioEngine& audioEngine)
    : engine(audioEngine)
{
    // Load IRボタン
    loadIRButton.setColour(juce::TextButton::buttonColourId,
                          juce::Colours::steelblue.withAlpha(0.7f));
    loadIRButton.setColour(juce::TextButton::textColourOffId,
                          juce::Colours::white);
    loadIRButton.addListener(this);
    addAndMakeVisible(loadIRButton);

    irAdvancedButton.setColour(juce::TextButton::buttonColourId,
                               juce::Colours::darkslategrey.withAlpha(0.85f));
    irAdvancedButton.setColour(juce::TextButton::textColourOffId,
                               juce::Colours::white);
    irAdvancedButton.setTooltip("Open detailed IR settings");
    irAdvancedButton.addListener(this);
    addAndMakeVisible(irAdvancedButton);

    convolverSettingsButton.setColour(juce::TextButton::buttonColourId,
                                      juce::Colours::darkslategrey.withAlpha(0.85f));
    convolverSettingsButton.setColour(juce::TextButton::textColourOffId,
                                      juce::Colours::white);
    convolverSettingsButton.setTooltip("Open phase2 convolver cache/upgrade settings");
    convolverSettingsButton.addListener(this);
    addAndMakeVisible(convolverSettingsButton);

    optimizationProgressButton.setTooltip("Show Mixed Phase Optimization Progress");
    optimizationProgressButton.addListener(this);
    addAndMakeVisible(optimizationProgressButton);

    // Phase Choice ComboBox
    phaseChoiceBox.addItem("As-Is", 1);
    phaseChoiceBox.addItem("Mixed", 2);
    phaseChoiceBox.addItem("Minimum", 3);
    phaseChoiceBox.setSelectedId(phaseModeToComboId(engine.getConvolverPhaseMode()), juce::dontSendNotification);
    phaseChoiceBox.setTooltip("Select IR Phase Type");
    phaseChoiceBox.setJustificationType(juce::Justification::centred);
    phaseChoiceBox.onChange = [this] {
        engine.setConvolverPhaseMode(comboIdToPhaseMode(phaseChoiceBox.getSelectedId()));
        updateMixedPhaseControlsEnabled();
    };
    addAndMakeVisible(phaseChoiceBox);

    experimentalDirectHeadToggle.setButtonText("Exp Direct Head");
    experimentalDirectHeadToggle.setTooltip("Experimental zero-latency direct head path. Rebuilds the convolver when changed.");
    experimentalDirectHeadToggle.setColour(juce::ToggleButton::textColourId, juce::Colours::white);
    experimentalDirectHeadToggle.addListener(this);
    addAndMakeVisible(experimentalDirectHeadToggle);

    // Dry/Wet Mixスライダー
    mixSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    mixSlider.setRange(0.0, 1.0, 0.01);
    mixSlider.setValue(1.0, juce::dontSendNotification);  // デフォルト100%
    mixSlider.setTextValueSuffix(" Mix");
    mixSlider.setNumDecimalPlacesToDisplay(2);
    mixSlider.addListener(this);
    addAndMakeVisible(mixSlider);

    // Mixラベル
    mixLabel.setText("Dry/Wet:", juce::dontSendNotification);
    mixLabel.setJustificationType(juce::Justification::centredRight);
    mixLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(mixLabel);

    // Smoothing Time スライダー
    smoothingTimeSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    smoothingTimeSlider.setRange(ConvolverProcessor::SMOOTHING_TIME_MIN_SEC * 1000.0,
                                 ConvolverProcessor::SMOOTHING_TIME_MAX_SEC * 1000.0, 1.0);
    smoothingTimeSlider.setSkewFactorFromMidPoint(100.0); // 対数的な操作感
    smoothingTimeSlider.setTextValueSuffix(" ms");
    smoothingTimeSlider.setNumDecimalPlacesToDisplay(0);
    smoothingTimeSlider.addListener(this);
    addAndMakeVisible(smoothingTimeSlider);

    // Smoothing Time ラベル
    smoothingTimeLabel.setText("Smoothing:", juce::dontSendNotification);
    smoothingTimeLabel.setJustificationType(juce::Justification::centredRight);
    smoothingTimeLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(smoothingTimeLabel);

    // IR Length スライダー
    irLengthSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    irLengthSlider.setRange(ConvolverProcessor::IR_LENGTH_MIN_SEC,
                            ConvolverProcessor::IR_LENGTH_MAX_SEC, 0.1);
    irLengthSlider.setSkewFactorFromMidPoint(1.5); // やや対数的な操作感
    irLengthSlider.setTextValueSuffix(" s");
    irLengthSlider.setNumDecimalPlacesToDisplay(1);
    irLengthSlider.addListener(this);
    addAndMakeVisible(irLengthSlider);

    // IR Length ラベル
    irLengthLabel.setText("IR Length:", juce::dontSendNotification);
    irLengthLabel.setJustificationType(juce::Justification::centredRight);
    irLengthLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(irLengthLabel);

    // Rebuild Debounce スライダー
    rebuildDebounceSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    rebuildDebounceSlider.setRange(ConvolverProcessor::REBUILD_DEBOUNCE_MIN_MS,
                                   ConvolverProcessor::REBUILD_DEBOUNCE_MAX_MS, 10.0);
    rebuildDebounceSlider.setSkewFactorFromMidPoint(static_cast<double>(ConvolverProcessor::REBUILD_DEBOUNCE_DEFAULT_MS));
    rebuildDebounceSlider.setTextValueSuffix(" ms");
    rebuildDebounceSlider.setNumDecimalPlacesToDisplay(0);
    rebuildDebounceSlider.setValue(static_cast<double>(engine.getConvolverProcessor().getRebuildDebounceMs()), juce::dontSendNotification);
    rebuildDebounceSlider.addListener(this);
    addAndMakeVisible(rebuildDebounceSlider);

    // Rebuild Debounce ラベル
    rebuildDebounceLabel.setText("Rebuild:", juce::dontSendNotification);
    rebuildDebounceLabel.setJustificationType(juce::Justification::centredRight);
    rebuildDebounceLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    rebuildDebounceLabel.setTooltip("IR rebuild debounce time (ms)");
    addAndMakeVisible(rebuildDebounceLabel);

    // Mixed F1 スライダー
    mixedF1Slider.setSliderStyle(juce::Slider::LinearHorizontal);
    mixedF1Slider.setRange(ConvolverProcessor::MIXED_F1_MIN_HZ,
                           ConvolverProcessor::MIXED_F1_MAX_HZ, 1.0);
    mixedF1Slider.setSkewFactorFromMidPoint(ConvolverProcessor::MIXED_F1_DEFAULT_HZ);
    mixedF1Slider.setTextValueSuffix(" Hz");
    mixedF1Slider.setNumDecimalPlacesToDisplay(0);
    mixedF1Slider.setValue(engine.getConvolverProcessor().getMixedTransitionStartHz(), juce::dontSendNotification);
    mixedF1Slider.addListener(this);
    mixedF1Slider.setTooltip("Mixed phase transition start frequency (f1)");
    addAndMakeVisible(mixedF1Slider);

    mixedF1Label.setText("Mix Start f:", juce::dontSendNotification);
    mixedF1Label.setJustificationType(juce::Justification::centredRight);
    mixedF1Label.setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(mixedF1Label);

    // Mixed F2 スライダー
    mixedF2Slider.setSliderStyle(juce::Slider::LinearHorizontal);
    mixedF2Slider.setRange(ConvolverProcessor::MIXED_F2_MIN_HZ,
                           ConvolverProcessor::MIXED_F2_MAX_HZ, 1.0);
    mixedF2Slider.setSkewFactorFromMidPoint(ConvolverProcessor::MIXED_F2_DEFAULT_HZ);
    mixedF2Slider.setTextValueSuffix(" Hz");
    mixedF2Slider.setNumDecimalPlacesToDisplay(0);
    mixedF2Slider.setValue(engine.getConvolverProcessor().getMixedTransitionEndHz(), juce::dontSendNotification);
    mixedF2Slider.addListener(this);
    mixedF2Slider.setTooltip("Mixed phase transition end frequency (f2)");
    addAndMakeVisible(mixedF2Slider);

    mixedF2Label.setText("Mix End f:", juce::dontSendNotification);
    mixedF2Label.setJustificationType(juce::Justification::centredRight);
    mixedF2Label.setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(mixedF2Label);

    // Mixed Tau スライダー
    mixedTauSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    mixedTauSlider.setRange(ConvolverProcessor::MIXED_TAU_MIN,
                            ConvolverProcessor::MIXED_TAU_MAX, 1.0);
    mixedTauSlider.setSkewFactorFromMidPoint(48.0);
    mixedTauSlider.setTextValueSuffix(" smp");
    mixedTauSlider.setNumDecimalPlacesToDisplay(0);
    mixedTauSlider.setValue(engine.getConvolverProcessor().getMixedPreRingTau(), juce::dontSendNotification);
    mixedTauSlider.addListener(this);
    mixedTauSlider.setTooltip("Mixed phase pre-ringing attenuation tau");
    addAndMakeVisible(mixedTauSlider);

    mixedTauLabel.setText("Mix tau:", juce::dontSendNotification);
    mixedTauLabel.setJustificationType(juce::Justification::centredRight);
    mixedTauLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(mixedTauLabel);

    // 詳細設定は別ウインドウへ移動 (メインパネルでは非表示)
    irLengthSlider.setVisible(false);
    irLengthLabel.setVisible(false);
    rebuildDebounceSlider.setVisible(false);
    rebuildDebounceLabel.setVisible(false);
    mixedF1Slider.setVisible(false);
    mixedF1Label.setVisible(false);
    mixedF2Slider.setVisible(false);
    mixedF2Label.setVisible(false);
    mixedTauSlider.setVisible(false);
    mixedTauLabel.setVisible(false);

    // IR情報ラベル
    irInfoLabel.setText("No IR loaded", juce::dontSendNotification);
    irInfoLabel.setJustificationType(juce::Justification::centred);
    irInfoLabel.setColour(juce::Label::textColourId,
                         juce::Colours::orange.withAlpha(0.8f));
    irInfoLabel.setFont(juce::FontOptions(13.0f, juce::Font::bold));
    // マウスクリックイベントを受け取り、IR再リンク機能を実現する
    irInfoLabel.addMouseListener(this, false);
    addAndMakeVisible(irInfoLabel);

    //----------------------------------------------------------
    // 出力周波数フィルター UI ── ① コンボルバー最終段の場合に使用
    //----------------------------------------------------------

    // ── ハイカットフィルターラベル ──
    hcfLabel.setText("HCF:", juce::dontSendNotification);
    hcfLabel.setJustificationType(juce::Justification::centredRight);
    hcfLabel.setColour(juce::Label::textColourId, juce::Colours::lightblue);
    hcfLabel.setTooltip("High-Cut Filter (active when Convolver is the last stage)");
    addAndMakeVisible(hcfLabel);

    // ── ハイカット: Sharp / Natural / Soft ラジオグループ ──
    constexpr int HC_GROUP_ID = 5001;

    hcfSharpButton.setClickingTogglesState(true);
    hcfSharpButton.setRadioGroupId(HC_GROUP_ID);
    hcfSharpButton.setColour(juce::TextButton::buttonOnColourId,  juce::Colours::steelblue);
    hcfSharpButton.setColour(juce::TextButton::buttonColourId,    juce::Colours::darkgrey.withAlpha(0.7f));
    hcfSharpButton.setColour(juce::TextButton::textColourOffId,   juce::Colours::white);
    hcfSharpButton.setTooltip("Sharp: Butterworth 4th-order (steep cutoff, preserves signal near 18kHz)");
    hcfSharpButton.onClick = [this] {
        if (hcfSharpButton.getToggleState())
        {
            engine.setConvHCFilterMode(convo::HCMode::Sharp);
            updateFilterModeButtons();
        }
    };
    addAndMakeVisible(hcfSharpButton);

    hcfNaturalButton.setClickingTogglesState(true);
    hcfNaturalButton.setRadioGroupId(HC_GROUP_ID);
    hcfNaturalButton.setColour(juce::TextButton::buttonOnColourId,  juce::Colours::steelblue);
    hcfNaturalButton.setColour(juce::TextButton::buttonColourId,    juce::Colours::darkgrey.withAlpha(0.7f));
    hcfNaturalButton.setColour(juce::TextButton::textColourOffId,   juce::Colours::white);
    hcfNaturalButton.setTooltip("Natural: Linkwitz-Riley 4th-order (default, excellent phase response)");
    hcfNaturalButton.onClick = [this] {
        if (hcfNaturalButton.getToggleState())
        {
            engine.setConvHCFilterMode(convo::HCMode::Natural);
            updateFilterModeButtons();
        }
    };
    addAndMakeVisible(hcfNaturalButton);

    hcfSoftButton.setClickingTogglesState(true);
    hcfSoftButton.setRadioGroupId(HC_GROUP_ID);
    hcfSoftButton.setColour(juce::TextButton::buttonOnColourId,  juce::Colours::steelblue);
    hcfSoftButton.setColour(juce::TextButton::buttonColourId,    juce::Colours::darkgrey.withAlpha(0.7f));
    hcfSoftButton.setColour(juce::TextButton::textColourOffId,   juce::Colours::white);
    hcfSoftButton.setTooltip("Soft: 2nd-order Q=0.5 (gentle slope, no time-domain smearing)");
    hcfSoftButton.onClick = [this] {
        if (hcfSoftButton.getToggleState())
        {
            engine.setConvHCFilterMode(convo::HCMode::Soft);
            updateFilterModeButtons();
        }
    };
    addAndMakeVisible(hcfSoftButton);

    // ── ローカットフィルターラベル ──
    lcfLabel.setText("LCF:", juce::dontSendNotification);
    lcfLabel.setJustificationType(juce::Justification::centredRight);
    lcfLabel.setColour(juce::Label::textColourId, juce::Colours::lightcoral);
    lcfLabel.setTooltip("Low-Cut Filter (active when Convolver is the last stage)");
    addAndMakeVisible(lcfLabel);

    // ── ローカット: Natural / Soft ラジオグループ ──
    constexpr int LC_GROUP_ID = 5002;

    lcfNaturalButton.setClickingTogglesState(true);
    lcfNaturalButton.setRadioGroupId(LC_GROUP_ID);
    lcfNaturalButton.setColour(juce::TextButton::buttonOnColourId,  juce::Colours::indianred);
    lcfNaturalButton.setColour(juce::TextButton::buttonColourId,    juce::Colours::darkgrey.withAlpha(0.7f));
    lcfNaturalButton.setColour(juce::TextButton::textColourOffId,   juce::Colours::white);
    lcfNaturalButton.setTooltip("Natural: Butterworth 2nd-order HPF at 18Hz");
    lcfNaturalButton.onClick = [this] {
        if (lcfNaturalButton.getToggleState())
        {
            engine.setConvLCFilterMode(convo::LCMode::Natural);
            updateFilterModeButtons();
        }
    };
    addAndMakeVisible(lcfNaturalButton);

    lcfSoftButton.setClickingTogglesState(true);
    lcfSoftButton.setRadioGroupId(LC_GROUP_ID);
    lcfSoftButton.setColour(juce::TextButton::buttonOnColourId,  juce::Colours::indianred);
    lcfSoftButton.setColour(juce::TextButton::buttonColourId,    juce::Colours::darkgrey.withAlpha(0.7f));
    lcfSoftButton.setColour(juce::TextButton::textColourOffId,   juce::Colours::white);
    lcfSoftButton.setTooltip("Soft: 2nd-order HPF Q=0.5 at 15Hz (gentler, sub-sonic removal)");
    lcfSoftButton.onClick = [this] {
        if (lcfSoftButton.getToggleState())
        {
            engine.setConvLCFilterMode(convo::LCMode::Soft);
            updateFilterModeButtons();
        }
    };
    addAndMakeVisible(lcfSoftButton);

    //----------------------------------------------------------
    // Convolver Input Trim スライダー (EQ→Conv モード時のみ表示)
    //----------------------------------------------------------
    convTrimLabel.setText("Conv Trim:", juce::dontSendNotification);
    convTrimLabel.setJustificationType(juce::Justification::centredRight);
    convTrimLabel.setColour(juce::Label::textColourId, juce::Colours::lightyellow);
    convTrimLabel.setTooltip("Pre-Convolver input gain trim (active in EQ->Conv order)");
    addAndMakeVisible(convTrimLabel);

    convTrimSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    convTrimSlider.setRange(-12.0, 0.0, 0.1);
    convTrimSlider.setTextValueSuffix(" dB");
    convTrimSlider.setNumDecimalPlacesToDisplay(1);
    convTrimSlider.addListener(this);
    addAndMakeVisible(convTrimSlider);

    // 現在の設定を反映
    updateFilterModeButtons();
    updateTrimSlider();
    updateMixedPhaseControlsEnabled();

    // リスナー登録（起動時XMLロードでも通知を受ける）
    engine.getConvolverProcessor().addListener(this);
}

ConvolverControlPanel::~ConvolverControlPanel()
{
    stopTimer();
    engine.getConvolverProcessor().removeListener(this);  // メモリリーク防止
    if (optimizationProgressWindow != nullptr)
    {
        optimizationProgressWindow->closeButtonPressed();
        optimizationProgressWindow = nullptr;
    }
}

void ConvolverControlPanel::convolverParamsChanged(ConvolverProcessor* processor)
{
    if (processor != &engine.getConvolverProcessor()) return;

    // XMLプリセットロード含む全経路で確実に進捗ウィンドウを制御
    const float progress = processor->getLoadProgress();
    if (progress > 0.0f && progress < 1.0f)
    {
        showOptimizationProgressWindow();          // 最適化中ウィンドウ表示
    }
    else if (optimizationProgressWindow != nullptr)
    {
        optimizationProgressWindow->closeButtonPressed(); // 完了で自動非表示
        // closeButtonPressed() はウィンドウを削除するため、SafePointer は自動的に nullptr になるが、
        // 明示的に nullptr を代入しても安全
        optimizationProgressWindow = nullptr;
    }

    updateIRInfo();
    updateWaveformPath();   // ← XMLロード完了時に即時グラフ更新（20秒遅延解消）
    repaint();
}

//--------------------------------------------------------------
// paint
//--------------------------------------------------------------
void ConvolverControlPanel::paint(juce::Graphics& g)
{
    // 背景
    g.setColour(juce::Colours::darkslategrey.withAlpha(0.85f));
    g.fillRoundedRectangle(getLocalBounds().toFloat(), 6.0f);

    // 枠線
    g.setColour(juce::Colours::lightblue.withAlpha(0.4f));
    g.drawRoundedRectangle(getLocalBounds().toFloat(), 6.0f, 2.0f);

    // タイトル
    g.setColour(juce::Colours::white);
    g.setFont(juce::FontOptions(15.0f, juce::Font::bold));
    g.drawText("CONVOLVER",
               getLocalBounds().reduced(8, 0).withHeight(22),
               juce::Justification::centredLeft);

    // 波形描画エリア
    auto bounds = getLocalBounds().reduced(10);
    bounds.removeFromTop(22); // タイトル分
    bounds.removeFromTop(5);

    // 波形背景
    g.setColour(juce::Colours::black.withAlpha(0.3f));
    g.fillRect(waveformArea);
    g.setColour(juce::Colours::white.withAlpha(0.1f));
    g.drawRect(waveformArea);

    // 波形描画
    if (!waveformPath.isEmpty())
    {
        g.setColour(juce::Colours::lightgreen.withAlpha(0.5f));
        g.fillPath(waveformPath);

        // ── 時間軸目盛と数値の描画 ──
        if (engine.getConvolverProcessor().isIRLoaded())
        {
            const int irSamples = engine.getConvolverProcessor().getIRLength();
            const double sampleRate = static_cast<double>(engine.getSampleRate());

            if (irSamples > 0 && sampleRate > 0.0)
            {
                const double durationSec = static_cast<double>(irSamples) / sampleRate;
                const double width = static_cast<double>(waveformArea.getWidth());

                // グリッド間隔の決定 (ピクセル幅に応じて調整)
                // 最小間隔 50px
                double intervalSec = 0.001;
                while (intervalSec * (width / durationSec) < 50.0)
                {
                    intervalSec *= 2.0;
                    if (intervalSec * (width / durationSec) >= 50.0) break;
                    intervalSec *= 2.5; // 2 -> 5
                    if (intervalSec * (width / durationSec) >= 50.0) break;
                    intervalSec *= 2.0; // 5 -> 10
                }

                g.setColour(juce::Colours::white.withAlpha(0.5f));
                g.setFont(10.0f);

                for (double t = 0.0; t <= durationSec; t += intervalSec)
                {
                    if (t <= 0.0001) continue; // 0は描画しない

                    float x = static_cast<float>(waveformArea.getX() + (t / durationSec) * width);
                    if (x > waveformArea.getRight() - 2) break;

                    // 目盛
                    g.drawVerticalLine(static_cast<int>(x), (float)waveformArea.getBottom() - 5.0f, (float)waveformArea.getBottom());

                    // 数値
                    juce::String label;
                    if (intervalSec < 1.0)
                        label = juce::String(static_cast<int>(t * 1000.0 + 0.5)) + "ms";
                    else
                        label = juce::String(t, 1) + "s";

                    g.drawText(label, static_cast<int>(x) - 20, waveformArea.getBottom() - 18, 40, 12, juce::Justification::centredBottom);
                }
            }
        }
    }
}

//--------------------------------------------------------------
// resized
//--------------------------------------------------------------
void ConvolverControlPanel::resized()
{
    auto bounds = getLocalBounds().reduced(10);
    bounds.removeFromTop(22); // タイトル
    bounds.removeFromTop(5);

    // 波形エリア
    waveformArea = bounds.removeFromTop(38);
    bounds.removeFromTop(4);

    // IR情報ラベル
    irInfoLabel.setBounds(bounds.removeFromTop(16));
    bounds.removeFromTop(4);

    constexpr int leftGap = 8;
    constexpr int loadWidth = 90;
    constexpr int phaseWidth = 80;
    constexpr int advancedWidth = 90;
    constexpr int settingsWidth = 110;

    auto controlRow1 = bounds.removeFromTop(26);
    bounds.removeFromTop(4);
    auto controlRow2 = bounds.removeFromTop(26);
    bounds.removeFromTop(4);

    const int labelWidth = 78;
    const int inlineGap = 6;
    const int controlsStartX = loadWidth + leftGap + phaseWidth + leftGap + advancedWidth + leftGap + settingsWidth + leftGap;

    // --- 1行目 ---
    loadIRButton.setBounds(controlRow1.removeFromLeft(loadWidth));
    controlRow1.removeFromLeft(leftGap);

    // 位相選択
    phaseChoiceBox.setBounds(controlRow1.removeFromLeft(phaseWidth));
    controlRow1.removeFromLeft(leftGap);

    // 詳細設定ボタン
    irAdvancedButton.setBounds(controlRow1.removeFromLeft(advancedWidth));
    controlRow1.removeFromLeft(leftGap);

    convolverSettingsButton.setBounds(controlRow1.removeFromLeft(settingsWidth));
    controlRow1.removeFromLeft(leftGap);

    // Dry/Wetミックス
    mixLabel.setBounds(controlRow1.removeFromLeft(labelWidth));
    controlRow1.removeFromLeft(inlineGap);
    mixSlider.setBounds(controlRow1);

    // --- 2行目 ---
    auto row2Left = controlRow2.removeFromLeft(controlsStartX - leftGap);
    experimentalDirectHeadToggle.setBounds(row2Left.removeFromTop(row2Left.getHeight() / 2));
    optimizationProgressButton.setBounds(row2Left);

    // スムージング時間 (Mixスライダーの下に配置)
    auto smoothingRow = controlRow2;
    smoothingRow.removeFromLeft(leftGap);
    smoothingTimeLabel.setBounds(smoothingRow.removeFromLeft(labelWidth));
    smoothingRow.removeFromLeft(inlineGap);
    smoothingTimeSlider.setBounds(smoothingRow);

    bounds.removeFromTop(6);

    // --- 8行目: ハイカットフィルターモード ---
    auto hcfRow = bounds.removeFromTop(26);
    hcfLabel.setBounds(hcfRow.removeFromLeft(38).reduced(0, 3));
    hcfRow.removeFromLeft(4);
    hcfSharpButton.setBounds(hcfRow.removeFromLeft(52).reduced(2, 2));
    hcfNaturalButton.setBounds(hcfRow.removeFromLeft(60).reduced(2, 2));
    hcfSoftButton.setBounds(hcfRow.removeFromLeft(48).reduced(2, 2));

    // --- 9行目: ローカットフィルターモード ---
    auto lcfRow = bounds.removeFromTop(26);
    lcfLabel.setBounds(lcfRow.removeFromLeft(38).reduced(0, 3));
    lcfRow.removeFromLeft(4);
    lcfNaturalButton.setBounds(lcfRow.removeFromLeft(60).reduced(2, 2));
    lcfSoftButton.setBounds(lcfRow.removeFromLeft(48).reduced(2, 2));

    // --- 10行目: Convolver Input Trim (EQ→Conv モード時のみ表示) ---
    auto trimRow = bounds.removeFromTop(26);
    convTrimLabel.setBounds(trimRow.removeFromLeft(72).reduced(0, 3));
    trimRow.removeFromLeft(4);
    convTrimSlider.setBounds(trimRow);

    updateWaveformPath();
}

//--------------------------------------------------------------
// updateFilterModeButtons
// エンジンの現在設定にあわせてボタンのトグル状態を同期する
//--------------------------------------------------------------
void ConvolverControlPanel::updateFilterModeButtons()
{
    const auto hcMode = engine.getConvHCFilterMode();
    hcfSharpButton  .setToggleState(hcMode == convo::HCMode::Sharp,   juce::dontSendNotification);
    hcfNaturalButton.setToggleState(hcMode == convo::HCMode::Natural, juce::dontSendNotification);
    hcfSoftButton   .setToggleState(hcMode == convo::HCMode::Soft,    juce::dontSendNotification);

    const auto lcMode = engine.getConvLCFilterMode();
    lcfNaturalButton.setToggleState(lcMode == convo::LCMode::Natural, juce::dontSendNotification);
    lcfSoftButton   .setToggleState(lcMode == convo::LCMode::Soft,    juce::dontSendNotification);
}

void ConvolverControlPanel::updateTrimSlider()
{
    const bool eqBypassed = engine.getEQProcessor().isBypassed();
    const bool convBypassed = engine.getConvolverProcessor().isBypassed();
    const bool isEqThenConv = (engine.getProcessingOrder() == AudioEngine::ProcessingOrder::EQThenConvolver);
    const bool shouldShowTrim = !eqBypassed && !convBypassed && isEqThenConv;

    convTrimLabel.setVisible(shouldShowTrim);
    convTrimSlider.setVisible(shouldShowTrim);
    convTrimSlider.setEnabled(shouldShowTrim);
    convTrimSlider.setValue(engine.getConvolverInputTrimDb(), juce::dontSendNotification);
}

void ConvolverControlPanel::updateMixedPhaseControlsEnabled()
{
    const auto phaseMode = comboIdToPhaseMode(phaseChoiceBox.getSelectedId());
    const bool enabled = (phaseMode == ConvolverProcessor::PhaseMode::Mixed);

    mixedF1Slider.setEnabled(enabled);
    mixedF2Slider.setEnabled(enabled);
    mixedTauSlider.setEnabled(enabled);

    mixedF1Label.setEnabled(enabled);
    mixedF2Label.setEnabled(enabled);
    mixedTauLabel.setEnabled(enabled);

    if (enabled)
    {
        mixedF1Slider.setTooltip("Mixed phase transition start frequency (f1)");
        mixedF2Slider.setTooltip("Mixed phase transition end frequency (f2)");
        mixedTauSlider.setTooltip("Mixed phase pre-ringing attenuation tau");
        mixedF1Label.setTooltip({});
        mixedF2Label.setTooltip({});
        mixedTauLabel.setTooltip({});
    }
    else
    {
        mixedF1Slider.setTooltip("Mixedで有効");
        mixedF2Slider.setTooltip("Mixedで有効");
        mixedTauSlider.setTooltip("Mixedで有効");
        mixedF1Label.setTooltip("Mixedで有効");
        mixedF2Label.setTooltip("Mixedで有効");
        mixedTauLabel.setTooltip("Mixedで有効");
    }
}

//--------------------------------------------------------------
// buttonClicked
//--------------------------------------------------------------
void ConvolverControlPanel::buttonClicked(juce::Button* button)
{
    if (button == &loadIRButton)
    {
        // ファイル選択ダイアログ
        // JUCE v8.0.12 推奨パターン: ローカルなshared_ptrを使用し、ラムダでキャプチャする。
        //
        // これにより、ダイアログ表示中に ConvolverControlPanel が破棄されても安全に動作する。
        auto fileChooser = std::make_shared<juce::FileChooser>("Select Impulse Response (IR) File",
                                  juce::File::getSpecialLocation(
                                      juce::File::userDocumentsDirectory),
                                  "*.wav;*.aif;*.aiff;*.flac");

        const auto chooserFlags = juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles;

        juce::Component::SafePointer<ConvolverControlPanel> safeThis(this);

        fileChooser->launchAsync(chooserFlags, [safeThis, fileChooser](const juce::FileChooser& fc)
        {
            if (safeThis == nullptr)
                return;

            if (fc.getResults().isEmpty())
                return;

            safeThis->engine.getConvolverProcessor().loadIR(fc.getResult());
        });
    }
    else if (button == &irAdvancedButton)
    {
        showIRAdvancedWindow();
    }
    else if (button == &convolverSettingsButton)
    {
        showConvolverSettingsWindow();
    }
    else if (button == &optimizationProgressButton)
    {
        showOptimizationProgressWindow();
    }
    else if (button == &experimentalDirectHeadToggle)
    {
        engine.getConvolverProcessor().setExperimentalDirectHeadEnabled(experimentalDirectHeadToggle.getToggleState());
        updateIRInfo();
    }
}

void ConvolverControlPanel::showIRAdvancedWindow()
{
    if (irAdvancedWindow != nullptr)
    {
        irAdvancedWindow->toFront(true);
        return;
    }

    juce::DialogWindow::LaunchOptions options;
    options.dialogTitle = "IR Advanced Settings";
    options.dialogBackgroundColour = juce::Colours::darkslategrey.withAlpha(0.95f);
    options.escapeKeyTriggersCloseButton = true;
    options.useNativeTitleBar = true;
    options.resizable = false;
    options.componentToCentreAround = this;
    options.content.setOwned(new IRAdvancedSettingsComponent(engine));

    auto* window = options.launchAsync();
    irAdvancedWindow = window;
    if (window != nullptr)
        window->setAlwaysOnTop(false);
}

void ConvolverControlPanel::showConvolverSettingsWindow()
{
    if (convolverSettingsWindow != nullptr)
    {
        convolverSettingsWindow->toFront(true);
        return;
    }

    juce::DialogWindow::LaunchOptions options;
    options.dialogTitle = "Convolver Settings";
    options.dialogBackgroundColour = juce::Colours::darkslategrey.withAlpha(0.95f);
    options.escapeKeyTriggersCloseButton = true;
    options.useNativeTitleBar = true;
    options.resizable = false;
    options.componentToCentreAround = this;
    options.content.setOwned(new ConvolverSettingsComponent(engine));

    auto* window = options.launchAsync();
    convolverSettingsWindow = window;
    if (window != nullptr)
        window->setAlwaysOnTop(false);
}

//--------------------------------------------------------------
// showOptimizationProgressWindow
// MixedPhaseOptimizationWindow を非モーダルで表示（最適化状況ウィンドウ）
//--------------------------------------------------------------
 void ConvolverControlPanel::showOptimizationProgressWindow()
 {
    if (optimizationProgressWindow != nullptr)
    {
        optimizationProgressWindow->toFront(true);
        return;
    }

    auto* window = new convo::MixedPhaseOptimizationWindow("Optimization Progress", engine.getConvolverProcessor());
    optimizationProgressWindow = window;
 }

//--------------------------------------------------------------
// sliderValueChanged
//--------------------------------------------------------------
void ConvolverControlPanel::sliderValueChanged(juce::Slider* slider)
{
    if (slider == &mixSlider)
    {
        pendingMixValue = static_cast<float>(slider->getValue());
        pendingMixDirty = true;
        markConvolverParameterDirty();
    }
    else if (slider == &smoothingTimeSlider)
    {
        pendingSmoothingTimeSec = static_cast<float>(slider->getValue()) / 1000.0f;
        pendingSmoothingDirty = true;
        markConvolverParameterDirty();
    }
    else if (slider == &irLengthSlider)
    {
        engine.getConvolverProcessor().setIRLengthManualOverride(true);
        pendingIrLengthSec = static_cast<float>(slider->getValue());
        pendingIrLengthDirty = true;
        markConvolverParameterDirty();
    }
    else if (slider == &mixedF1Slider)
    {
        pendingMixedF1Hz = static_cast<float>(slider->getValue());
        pendingMixedF1Dirty = true;
        markConvolverParameterDirty();
    }
    else if (slider == &mixedF2Slider)
    {
        pendingMixedF2Hz = static_cast<float>(slider->getValue());
        pendingMixedF2Dirty = true;
        markConvolverParameterDirty();
    }
    else if (slider == &mixedTauSlider)
    {
        pendingMixedTau = static_cast<float>(slider->getValue());
        pendingMixedTauDirty = true;
        markConvolverParameterDirty();
    }
    else if (slider == &rebuildDebounceSlider)
    {
        engine.getConvolverProcessor().setRebuildDebounceMs(static_cast<int>(slider->getValue()));
    }
    else if (slider == &convTrimSlider)
    {
        engine.setConvolverInputTrimDb(static_cast<float>(slider->getValue()));
    }
}

void ConvolverControlPanel::timerCallback()
{
    if (!hasPendingConvolverParameters())
    {
        stopTimer();
        return;
    }

    const double nowMs = juce::Time::getMillisecondCounterHiRes();
    if ((nowMs - lastParameterChangeMs) < static_cast<double>(PARAMETER_RECALC_DEBOUNCE_MS))
        return;

    applyPendingConvolverParameters();
    stopTimer();
}

void ConvolverControlPanel::markConvolverParameterDirty()
{
    lastParameterChangeMs = juce::Time::getMillisecondCounterHiRes();
    if (!isTimerRunning())
        startTimer(100);
}

bool ConvolverControlPanel::hasPendingConvolverParameters() const noexcept
{
    return pendingMixDirty || pendingSmoothingDirty || pendingIrLengthDirty
        || pendingMixedF1Dirty || pendingMixedF2Dirty || pendingMixedTauDirty;
}

void ConvolverControlPanel::applyPendingConvolverParameters()
{
    auto& convolver = engine.getConvolverProcessor();

    if (pendingMixDirty)
        convolver.setMix(pendingMixValue);

    if (pendingSmoothingDirty)
        convolver.setSmoothingTime(pendingSmoothingTimeSec);

    if (pendingIrLengthDirty)
        convolver.setTargetIRLength(pendingIrLengthSec);

    if (pendingMixedF1Dirty)
        convolver.setMixedTransitionStartHz(pendingMixedF1Hz);

    if (pendingMixedF2Dirty)
        convolver.setMixedTransitionEndHz(pendingMixedF2Hz);

    if (pendingMixedTauDirty)
        convolver.setMixedPreRingTau(pendingMixedTau);

    pendingMixDirty = false;
    pendingSmoothingDirty = false;
    pendingIrLengthDirty = false;
    pendingMixedF1Dirty = false;
    pendingMixedF2Dirty = false;
    pendingMixedTauDirty = false;

    updateIRInfo();
}

void ConvolverControlPanel::mouseDown(const juce::MouseEvent& event)
{
    // IR情報ラベルがクリックされたかチェック
    if (event.originalComponent == &irInfoLabel)
    {
        auto& convolver = engine.getConvolverProcessor();
        // エラーメッセージが「見つからない」場合のみファイル選択ダイアログを開く
        if (convolver.getLastError().startsWith("IR not found"))
        {
            // 既存のロードボタンのクリックイベントを再利用することで、
            // ファイル選択ダイアログのロジックを重複させずに済む。
            loadIRButton.triggerClick();
        }
    }
}

//--------------------------------------------------------------
// updateIRInfo
//--------------------------------------------------------------
void ConvolverControlPanel::updateIRInfo()
{
    auto& convolver = engine.getConvolverProcessor();

    // UIコントロールをプロセッサの状態と同期
    mixSlider.setValue(pendingMixDirty ? pendingMixValue : convolver.getMix(), juce::dontSendNotification);
    phaseChoiceBox.setSelectedId(phaseModeToComboId(convolver.getPhaseMode()), juce::dontSendNotification);
    updateMixedPhaseControlsEnabled();
    experimentalDirectHeadToggle.setToggleState(convolver.getExperimentalDirectHeadEnabled(), juce::dontSendNotification);
    smoothingTimeSlider.setValue((pendingSmoothingDirty ? pendingSmoothingTimeSec : convolver.getSmoothingTime()) * 1000.0,
                                 juce::dontSendNotification);
    const double irLengthSliderMax = std::max(static_cast<double>(ConvolverProcessor::IR_LENGTH_MAX_SEC),
                                              static_cast<double>(pendingIrLengthDirty ? pendingIrLengthSec : convolver.getTargetIRLength()));
    irLengthSlider.setRange(ConvolverProcessor::IR_LENGTH_MIN_SEC, irLengthSliderMax, 0.1);
    irLengthSlider.setValue(pendingIrLengthDirty ? pendingIrLengthSec : convolver.getTargetIRLength(),
                            juce::dontSendNotification);
    mixedF1Slider.setValue(pendingMixedF1Dirty ? pendingMixedF1Hz : convolver.getMixedTransitionStartHz(),
                           juce::dontSendNotification);
    mixedF2Slider.setValue(pendingMixedF2Dirty ? pendingMixedF2Hz : convolver.getMixedTransitionEndHz(),
                           juce::dontSendNotification);
    mixedTauSlider.setValue(pendingMixedTauDirty ? pendingMixedTau : convolver.getMixedPreRingTau(),
                            juce::dontSendNotification);
    rebuildDebounceSlider.setValue(static_cast<double>(convolver.getRebuildDebounceMs()), juce::dontSendNotification);
    updateFilterModeButtons();
    updateTrimSlider();

    if (convolver.isIRLoaded())
    {
        juce::String info = convolver.getIRName();
        info += " (" + juce::String(convolver.getIRLength()) + " smp)";
        info += " IR Len: " + juce::String(convolver.getTargetIRLength(), 2) + "s";

        if (convolver.hasManualIRLengthOverride())
            info += " [Manual, Auto " + juce::String(convolver.getAutoDetectedIRLength(), 2) + "s]";
        else
            info += " [Auto]";

        // A = convolver algorithm latency, T = convolver total latency (base sample rate)
        const auto latency = engine.getCurrentLatencyBreakdown();
        const int algorithmLatencySamples = latency.convolverAlgorithmLatencyBaseRateSamples;
        const int totalLatencySamples = latency.convolverTotalLatencyBaseRateSamples;
        const double processingSampleRate = engine.getSampleRate();

        const auto toRoundedMsInt = [processingSampleRate](int samples) -> int
        {
            if (processingSampleRate <= 0.0)
                return 0;

            const double ms = (static_cast<double>(samples) * 1000.0) / processingSampleRate;
            const double msRounded3 = std::round(ms * 1000.0) / 1000.0;
            return juce::roundToInt(msRounded3);
        };

        const int algorithmMs = toRoundedMsInt(algorithmLatencySamples);
        const int totalMs = toRoundedMsInt(totalLatencySamples);

        info += " Lat A: " + juce::String(algorithmMs) + "ms / T: " + juce::String(totalMs) + "ms";

        irInfoLabel.setText(info, juce::dontSendNotification);
        irInfoLabel.setColour(juce::Label::textColourId,
                             juce::Colours::lightgreen);
    }
    else if (const float progress = convolver.getLoadProgress(); progress > 0.0f && progress < 1.0f)
    {
        // 最適化進行中（XMLプリセットロード時も即時反映）
        const int percent = juce::roundToInt(progress * 100.0f);
        irInfoLabel.setText("Optimization Progress... " + juce::String(percent) + "%", juce::dontSendNotification);
        irInfoLabel.setColour(juce::Label::textColourId, juce::Colours::orange.withAlpha(0.9f));
        updateWaveformPath();
        return;
    }
    else
    {
        // エラーがある場合は赤字で表示
        if (convolver.getLastError().isNotEmpty())
        {
            juce::String errorMessage = convolver.getLastError();
            if (errorMessage.startsWith("IR not found"))
            {
                irInfoLabel.setText(errorMessage + " (Click to locate...)", juce::dontSendNotification);
                irInfoLabel.setColour(juce::Label::textColourId, juce::Colours::orange);
                irInfoLabel.setTooltip("The IR file for this preset was not found. Click here to find it.");
            }
            else
            {
                irInfoLabel.setText(errorMessage, juce::dontSendNotification);
                irInfoLabel.setColour(juce::Label::textColourId, juce::Colours::red);
                irInfoLabel.setTooltip("");
            }
        }
        else
        {
            irInfoLabel.setText("No IR loaded", juce::dontSendNotification);
            irInfoLabel.setColour(juce::Label::textColourId,
                                 juce::Colours::orange.withAlpha(0.8f));
        }
    }

    updateWaveformPath();
}

//--------------------------------------------------------------
// updateWaveformPath
//--------------------------------------------------------------
void ConvolverControlPanel::updateWaveformPath()
{
    waveformPath.clear();
    const auto& waveform = engine.getConvolverProcessor().getIRWaveform();

    if (waveform.empty() || waveform.size() < 2 || waveformArea.isEmpty())
    {
        repaint();
        return;
    }

    const float w = static_cast<float>(waveformArea.getWidth());
    const float h = static_cast<float>(waveformArea.getHeight());
    const float x = static_cast<float>(waveformArea.getX());
    const float y = static_cast<float>(waveformArea.getBottom()); // 下端基準

    waveformPath.startNewSubPath(x, y);
    for (size_t i = 0; i < waveform.size(); ++i)
    {
        float val = waveform[i];
        float px = x + (static_cast<float>(i) / static_cast<float>(waveform.size() - 1)) * w;
        float py = y - val * h;
        waveformPath.lineTo(px, py);
    }
    waveformPath.lineTo(x + w, y);
    waveformPath.closeSubPath();

    repaint();
}
