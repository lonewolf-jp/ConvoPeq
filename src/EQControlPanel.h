//============================================================================
#pragma once
// EQControlPanel.h (JUCE 8.0.12対応)
//
// EQのUIコントロールパネル
//
// ■ 機能:
//   - 各バンドのゲイン/周波数/Q値の数値入力
//   - 各バンドの有効/無効 ToggleButton
//   - バンド名ラベル
//   - フィルタタイプ選択 (ComboBox)
//   - チャンネルモード選択 (ComboBox)
//   - トータルゲイン / AGC 設定
//   - プリセット選択
//   - Reset All ボタン（全バンドデフォルト復元）
//
// ■ スレッド安全:
//   - labelTextChanged(), comboBoxChanged(), buttonClicked() は UI Thread
//   - EQProcessor の setBandXxx() は atomic 書き込みなので安全
//============================================================================

#include <JuceHeader.h>
#include "AudioEngine.h"

class EQControlPanel : public juce::Component,
                       private juce::Label::Listener,
                       private juce::Button::Listener,
                       private juce::ComboBox::Listener
{
private:
    // ── バンド名（20バンド）──
    static constexpr const char* BAND_NAMES[EQProcessor::NUM_BANDS] = {
        "Rule 1", "Rule 2", "Rule 3", "Rule 4", "Rule 5",
        "Rule 6", "Rule 7", "Rule 8", "Rule 9", "Rule 10",
        "Rule 11", "Rule 12", "Rule 13", "Rule 14", "Rule 15",
        "Rule 16", "Rule 17", "Rule 18", "Rule 19", "Rule 20"
    };

    // ── UIコントロールとバンドインデックスのマッピング用 ──
    enum class ControlType { Gain, Freq, Q, Enable, Type, Channel };

    struct ControlID
    {
        const juce::Component* control = nullptr;
        int bandIndex = -1;
        ControlType type;
    };

    std::vector<ControlID> controlMap;

public:
    explicit EQControlPanel(AudioEngine& audioEngine);

    void paint  (juce::Graphics& g) override;
    void resized() override;

    // ── ラベル更新ヘルパー ──
    void updateBandValues(int band);
    void updateAllControls();

private:
    AudioEngine& engine;

    // ── コントロール配列 ──
    juce::Label        gainLabels[EQProcessor::NUM_BANDS];
    juce::Label        freqLabels[EQProcessor::NUM_BANDS];
    juce::Label        qLabels[EQProcessor::NUM_BANDS];
    juce::ToggleButton enableButtons[EQProcessor::NUM_BANDS];
    juce::Label        bandLabels[EQProcessor::NUM_BANDS];
    juce::ComboBox     typeBoxes[EQProcessor::NUM_BANDS];
    juce::ComboBox     channelBoxes[EQProcessor::NUM_BANDS];
    juce::Label        totalGainLabel;                           // "Total Gain:"
    juce::Label        totalGainValueLabel;                      // 数値入力
    juce::ToggleButton agcButton;                                // AGC Checkbox
    juce::TextButton   resetButton;                              // 全バンドリセット
    juce::ComboBox     presetSelector;                           // プリセット選択

    // ── 周波数範囲の定数（20バンド）──
    struct FreqRange { float minHz; float maxHz; };
    static constexpr FreqRange FREQ_RANGES[EQProcessor::NUM_BANDS] = {
        { 20.0f, 20000.0f },   // Band 0
        { 20.0f, 20000.0f },   // Band 1
        { 20.0f, 20000.0f },   // Band 2
        { 20.0f, 20000.0f },   // Band 3
        { 20.0f, 20000.0f },   // Band 4
        { 20.0f, 20000.0f },   // Band 5
        { 20.0f, 20000.0f },   // Band 6
        { 20.0f, 20000.0f },   // Band 7
        { 20.0f, 20000.0f },   // Band 8
        { 20.0f, 20000.0f },   // Band 9
        { 20.0f, 20000.0f },   // Band 10
        { 20.0f, 20000.0f },   // Band 11
        { 20.0f, 20000.0f },   // Band 12
        { 20.0f, 20000.0f },   // Band 13
        { 20.0f, 20000.0f },   // Band 14
        { 20.0f, 20000.0f },   // Band 15
        { 20.0f, 20000.0f },   // Band 16
        { 20.0f, 20000.0f },   // Band 17
        { 20.0f, 20000.0f },   // Band 18
        { 20.0f, 20000.0f }    // Band 19
    };

    // ── Q値の対数スケール範囲 ──
    static constexpr float Q_MIN = 0.1f;
    static constexpr float Q_MAX = 10.0f;

    // ── ゲイン範囲 ──
    static constexpr float MIN_BAND_GAIN = -12.0f;
    static constexpr float MAX_BAND_GAIN = 12.0f;
    static constexpr float MIN_TOTAL_GAIN = -24.0f;
    static constexpr float MAX_TOTAL_GAIN = 24.0f;

    // ── Listener コールバック ──
    void labelTextChanged(juce::Label* label) override;
    void buttonClicked(juce::Button* button) override;
    void comboBoxChanged(juce::ComboBox* comboBox) override;
    void editorShown(juce::Label* label, juce::TextEditor& editor) override;

    // ── コントロール検索ヘルパー ──
    const ControlID* findControlId(const juce::Component* control) const;
};
