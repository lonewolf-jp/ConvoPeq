//============================================================================
// EQControlPanel.cpp ── v0.2 (JUCE 8.0.12対応)
//
// 20バンドパラメトリックEQコントロールパネルの実装
//
//============================================================================
#include "EQControlPanel.h"

namespace
{
//--------------------------------------------------------------
// 周波数を表示用の文字列に変換
// 1000以上なら "Xk" で表示
//--------------------------------------------------------------
    juce::String formatFreq(float freq)
    {
        if (freq >= 1000.0f)
            return juce::String(freq / 1000.0f, 1) + " kHz";
        else
            return juce::String(static_cast<int>(freq)) + " Hz";
    }
}

//--------------------------------------------------------------
// コンストラクタ
//--------------------------------------------------------------
EQControlPanel::EQControlPanel(AudioEngine& audioEngine)
    : engine(audioEngine)
{
    controlMap.reserve(EQProcessor::NUM_BANDS * 6);

    for (int i = 0; i < EQProcessor::NUM_BANDS; ++i)
    {
        //----------------------------------------------------
        // ── ゲイン入力ラベル ──
        //----------------------------------------------------
        gainLabels[i].setJustificationType(juce::Justification::centred);
        gainLabels[i].setColour(juce::Label::textColourId, juce::Colours::lightgreen);
        gainLabels[i].setEditable(true);
        gainLabels[i].setTooltip("Click to edit Gain (" + juce::String(MIN_BAND_GAIN, 1)
                               + " to " + juce::String(MAX_BAND_GAIN, 1) + " dB)");
        gainLabels[i].addListener(this);
        addAndMakeVisible(gainLabels[i]);
        controlMap.push_back({&gainLabels[i], i, ControlType::Gain});

        //----------------------------------------------------
        // ── 周波数入力ラベル ──
        //----------------------------------------------------
        freqLabels[i].setJustificationType(juce::Justification::centred);
        freqLabels[i].setColour(juce::Label::textColourId, juce::Colours::cyan);
        freqLabels[i].setEditable(true);
        freqLabels[i].setTooltip("Click to edit Frequency");
        freqLabels[i].addListener(this);
        addAndMakeVisible(freqLabels[i]);
        controlMap.push_back({&freqLabels[i], i, ControlType::Freq});

        //----------------------------------------------------
        // ── Q値入力ラベル ──
        //----------------------------------------------------
        qLabels[i].setJustificationType(juce::Justification::centred);
        qLabels[i].setColour(juce::Label::textColourId, juce::Colours::orange);
        qLabels[i].setEditable(true);
        qLabels[i].setTooltip("Click to edit Q factor (" + juce::String(Q_MIN, 1)
                               + " to " + juce::String(Q_MAX, 1) + ")");
        qLabels[i].addListener(this);
        addAndMakeVisible(qLabels[i]);
        controlMap.push_back({&qLabels[i], i, ControlType::Q});

        //----------------------------------------------------
        // ── 有効/無効ボタン ──
        //----------------------------------------------------
        enableButtons[i].setButtonText("ON");
        enableButtons[i].setToggleState(true, juce::dontSendNotification);
        enableButtons[i].addListener(this);
        addAndMakeVisible(enableButtons[i]);
        controlMap.push_back({&enableButtons[i], i, ControlType::Enable});

        //----------------------------------------------------
        // ── バンド名ラベル ──
        //----------------------------------------------------
        bandLabels[i].setJustificationType(juce::Justification::centred);
        bandLabels[i].setColour(juce::Label::textColourId,
                                juce::Colours::white.withAlpha(0.85f));
        bandLabels[i].setFont(juce::FontOptions(14.0f, juce::Font::bold));
        bandLabels[i].setText(BAND_NAMES[i], juce::dontSendNotification);
        addAndMakeVisible(bandLabels[i]);

        //----------------------------------------------------
        // ── フィルタータイプ選択 ──
        //----------------------------------------------------
        typeBoxes[i].addItem("Low Shelf",  1);
        typeBoxes[i].addItem("Peaking",    2);
        typeBoxes[i].addItem("High Shelf", 3);
        typeBoxes[i].addItem("Low Pass",   4);
        typeBoxes[i].addItem("High Pass",  5);
        typeBoxes[i].setJustificationType(juce::Justification::centred);
        typeBoxes[i].setTooltip("Select Filter Type");
        typeBoxes[i].addListener(this);
        addAndMakeVisible(typeBoxes[i]);
        controlMap.push_back({&typeBoxes[i], i, ControlType::Type});

        //----------------------------------------------------
        // ── チャンネル選択 ──
        //----------------------------------------------------
        channelBoxes[i].addItem("Stereo", 1);
        channelBoxes[i].addItem("Left",   2);
        channelBoxes[i].addItem("Right",  3);
        channelBoxes[i].setJustificationType(juce::Justification::centred);
        channelBoxes[i].setTooltip("Select Channel Mode");
        channelBoxes[i].addListener(this);
        addAndMakeVisible(channelBoxes[i]);
        controlMap.push_back({&channelBoxes[i], i, ControlType::Channel});
    }

    //------------------------------------------------------
    // ── トータルゲイン・AGC ──
    //------------------------------------------------------
    totalGainLabel.setText("Total Gain:", juce::dontSendNotification);
    totalGainLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    totalGainLabel.setJustificationType(juce::Justification::centredRight);
    addAndMakeVisible(totalGainLabel);

    totalGainValueLabel.setText("0.0 dB", juce::dontSendNotification);
    totalGainValueLabel.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
    totalGainValueLabel.setJustificationType(juce::Justification::centredLeft);
    totalGainValueLabel.setEditable(true);
    totalGainValueLabel.setTooltip("Master Output Gain (" + juce::String(MIN_TOTAL_GAIN, 1)
                                   + " to " + juce::String(MAX_TOTAL_GAIN, 1) + " dB)");
    totalGainValueLabel.addListener(this);
    addAndMakeVisible(totalGainValueLabel);

    agcButton.setButtonText("AGC");
    agcButton.setColour(juce::ToggleButton::textColourId, juce::Colours::white);
    agcButton.setTooltip("Auto Gain Control (Match Output Level to Input Level)");
    agcButton.onClick = [this]
    {
        engine.getEQProcessor().setAGCEnabled(agcButton.getToggleState());
        totalGainValueLabel.setEnabled(!agcButton.getToggleState());
    };
    addAndMakeVisible(agcButton);

    //------------------------------------------------------
    // ── Reset All ボタン ──
    //   タイトル行の右側に配置。全バンドのゲイン・周波数・Q・有効を初期値に戻す。
    //------------------------------------------------------
    resetButton.setButtonText("Reset");
    resetButton.setColour(juce::TextButton::buttonColourId,
                          juce::Colours::darkslategrey.withAlpha(0.7f));
    resetButton.setColour(juce::TextButton::textColourOffId,
                          juce::Colours::white.withAlpha(0.8f));
    resetButton.addListener(this);
    addAndMakeVisible(resetButton);

    //------------------------------------------------------
    // ── Preset Selector ──
    //------------------------------------------------------
    presetSelector.addItem("Default", 1);
    presetSelector.addItem("Flat", 2);
    presetSelector.setSelectedId(1, juce::dontSendNotification);
    presetSelector.setTooltip("Select EQ Preset (Applied on next playback start)");
    presetSelector.addListener(this);
    addAndMakeVisible(presetSelector);

    // UI全体をプロセッサの現在の状態で初期化
    updateAllControls();
}

//--------------------------------------------------------------
// updateBandValues  ──  プロセッサの値からUIを更新
//--------------------------------------------------------------
void EQControlPanel::updateBandValues(int band)
{
    if (band < 0 || band >= EQProcessor::NUM_BANDS) return;

    auto params = engine.getEQProcessor().getBandParams(band);

    gainLabels[band].setText(juce::String(params.gain, 1) + " dB", juce::dontSendNotification);
    freqLabels[band].setText(formatFreq(params.frequency), juce::dontSendNotification);
    qLabels[band].setText("Q: " + juce::String(params.q, 2), juce::dontSendNotification);
}

void EQControlPanel::updateAllControls()
{
    for (int i = 0; i < EQProcessor::NUM_BANDS; ++i)
    {
        updateBandValues(i);

        const bool isEnabled = engine.getEQProcessor().getBandParams(i).enabled;
        enableButtons[i].setToggleState(isEnabled, juce::dontSendNotification);
        enableButtons[i].setButtonText(isEnabled ? "ON" : "OFF");

        const int typeId = static_cast<int>(engine.getEQProcessor().getBandType(i)) + 1;
        typeBoxes[i].setSelectedId(typeId, juce::dontSendNotification);

        const int channelId = static_cast<int>(engine.getEQProcessor().getBandChannelMode(i)) + 1;
        channelBoxes[i].setSelectedId(channelId, juce::dontSendNotification);
    }

    const float totalGain = engine.getEQProcessor().getTotalGain();
    totalGainValueLabel.setText(juce::String(totalGain, 1) + " dB", juce::dontSendNotification);

    const bool agcOn = engine.getEQProcessor().getAGCEnabled();
    agcButton.setToggleState(agcOn, juce::dontSendNotification);
    totalGainValueLabel.setEnabled(!agcOn);
}

//--------------------------------------------------------------
// labelTextChanged  ──  ラベル編集完了時のコールバック
//--------------------------------------------------------------
void EQControlPanel::labelTextChanged(juce::Label* label)
{
    if (const auto* id = findControlId(label))
    {
        const int i = id->bandIndex;
        switch (id->type)
        {
            case ControlType::Gain:
            {
                float val = label->getText().retainCharacters("-0123456789.").getFloatValue();
                val = juce::jlimit(MIN_BAND_GAIN, MAX_BAND_GAIN, val);
                engine.getEQProcessor().setBandGain(i, val);
                updateBandValues(i); // フォーマットを整える
                return;
            }
            case ControlType::Freq:
            {
                float val = label->getText().retainCharacters("0123456789.").getFloatValue();
                val = juce::jlimit(FREQ_RANGES[i].minHz, FREQ_RANGES[i].maxHz, val);
                engine.getEQProcessor().setBandFrequency(i, val);
                updateBandValues(i);
                return;
            }
            case ControlType::Q:
            {
                float val = label->getText().retainCharacters("-0123456789.").getFloatValue();
                val = juce::jlimit(Q_MIN, Q_MAX, val);
                engine.getEQProcessor().setBandQ(i, val);
                updateBandValues(i);
                return;
            }
            default: break;
        }
    }

    // ── トータルゲイン入力 ──
    if (label == &totalGainValueLabel)
    {
        float val = label->getText().retainCharacters("-0123456789.").getFloatValue();
        val = juce::jlimit(MIN_TOTAL_GAIN, MAX_TOTAL_GAIN, val);
        engine.getEQProcessor().setTotalGain(val);
        totalGainValueLabel.setText(juce::String(val, 1) + " dB", juce::dontSendNotification);
    }
}

//--------------------------------------------------------------
// editorShown  ──  編集開始時に呼ばれる
//   数値のみを表示するようにテキストを加工する
//--------------------------------------------------------------
void EQControlPanel::editorShown(juce::Label* label, juce::TextEditor& editor)
{
    if (const auto* id = findControlId(label))
    {
        const int i = id->bandIndex;
        switch (id->type)
        {
            case ControlType::Gain:
                editor.setText(label->getText().replace(" dB", ""));
                return;
            case ControlType::Q:
                editor.setText(label->getText().replace("Q: ", ""));
                return;
            case ControlType::Freq:
            {
                // 編集時はHz単位の数値のみを表示
                float freq = engine.getEQProcessor().getBandParams(i).frequency;
                juce::String text = juce::String(freq, 1);
                if (text.endsWith(".0"))
                    text = text.dropLastCharacters(2);
                editor.setText(text);
                return;
            }
            default:
                break;
        }
    }

    if (label == &totalGainValueLabel)
    {
        juce::String text = label->getText().replace(" dB", "");
        editor.setText(text);
        return;
    }
}

//--------------------------------------------------------------
// buttonClicked  ──  ボタン押下時のコールバック
//   - enableButtons[i] : バンドの有効/無効トグル
//   - resetButton      : 全バンドデフォルト復元
//--------------------------------------------------------------
void EQControlPanel::buttonClicked(juce::Button* button)
{
    if (const auto* id = findControlId(button))
    {
        if (id->type == ControlType::Enable)
        {
            const bool isEnabled = enableButtons[id->bandIndex].getToggleState();
            engine.getEQProcessor().setBandEnabled(id->bandIndex, isEnabled);
            enableButtons[id->bandIndex].setButtonText(isEnabled ? "ON" : "OFF");
            return;
        }
    }

    // ── Reset All ボタン ──
    if (button == &resetButton)
    {
        engine.getEQProcessor().resetToDefaults();
        return;
    }
}

//--------------------------------------------------------------
// comboBoxChanged  ──  フィルタータイプ変更
//--------------------------------------------------------------
void EQControlPanel::comboBoxChanged(juce::ComboBox* comboBox)
{
    if (const auto* id = findControlId(comboBox))
    {
        const int i = id->bandIndex;
        const int selectedId = comboBox->getSelectedId();
        if (selectedId <= 0) return;

        switch (id->type)
        {
            case ControlType::Type:
                // ID(1-based) -> Enum(0-based)
                engine.getEQProcessor().setBandType(i, static_cast<EQBandType>(selectedId - 1));
                return;

            case ControlType::Channel:
                engine.getEQProcessor().setBandChannelMode(i, static_cast<EQChannelMode>(selectedId - 1));
                return;

            default:
                break;
        }
    }

    if (comboBox == &presetSelector)
    {
        engine.requestEqPreset(presetSelector.getSelectedItemIndex());
    }
}

const EQControlPanel::ControlID* EQControlPanel::findControlId(const juce::Component* control) const
{
    auto it = std::find_if(controlMap.begin(), controlMap.end(),
                           [control](const ControlID& entry) { return entry.control == control; });

    if (it != controlMap.end())
        return &(*it);

    return nullptr;
}

//--------------------------------------------------------------
// paint  ──  パネルの背景とタイトルを描画
//--------------------------------------------------------------
void EQControlPanel::paint(juce::Graphics& g)
{
    // ── 背景 ──
    g.setColour(juce::Colours::darkgrey.withAlpha(0.75f));
    g.fillRoundedRectangle(getLocalBounds().toFloat(), 5.0f);

    // ── 枠線 ──
    g.setColour(juce::Colours::grey.withAlpha(0.4f));
    g.drawRoundedRectangle(getLocalBounds().toFloat(), 5.0f, 1.0f);

    // ── タイトル ──
    g.setColour(juce::Colours::white);
    g.setFont(juce::FontOptions(14.0f, juce::Font::bold));
    g.drawText("20-Band Parametric EQ",
               getLocalBounds().reduced(8, 0).withHeight(20),
               juce::Justification::centredLeft);
}

//--------------------------------------------------------------
// resized  ──  レイアウト計算
//
// レイアウト構成:
//   [タイトル行(22px)]
//     左: タイトル
//     右: [Total Gain] [AGC] [Preset] [Reset]
//   [各バンド列(幅均等)]
//     ├─ バンド名ラベル
//     ├─ フィルタタイプ選択 (ComboBox)
//     ├─ チャンネル選択 (ComboBox)
//     ├─ ゲイン/周波数/Q値 入力ラベル
//     └─ ON/OFF ボタン
//--------------------------------------------------------------
void EQControlPanel::resized()
{
    auto bounds = getLocalBounds();

    // ── タイトル行とResetボタン ──
    auto titleRow = bounds.removeFromTop(22);
    resetButton.setBounds(titleRow.removeFromRight(64).reduced(2, 2));
    presetSelector.setBounds(titleRow.removeFromRight(100).reduced(2, 2));

    // タイトルの右側に配置 (タイトル幅 約170px確保)
    auto controlsArea = titleRow.withTrimmedLeft(170);

    agcButton.setBounds(controlsArea.removeFromRight(50).reduced(2, 2));
    totalGainValueLabel.setBounds(controlsArea.removeFromRight(60).reduced(2, 2));
    totalGainLabel.setBounds(controlsArea.removeFromRight(70).reduced(2, 2));

    // ── 各バンド列 ──
    // 2段表示 (10バンド x 2行)
    const int numCols = 10;
    const int rowHeight = bounds.getHeight() / 2;

    auto topRow = bounds.removeFromTop(rowHeight);
    auto bottomRow = bounds; // 残り

    const int bandWidth = topRow.getWidth() / numCols;

    for (int i = 0; i < EQProcessor::NUM_BANDS; ++i)
    {
        juce::Rectangle<int> bandBounds;
        if (i < numCols)
            bandBounds = topRow.removeFromLeft(bandWidth);
        else
            bandBounds = bottomRow.removeFromLeft(bandWidth);

        // レイアウト調整 (スライダー廃止に伴う高さ再設定)
        bandLabels[i].setBounds   (bandBounds.removeFromTop   (20));
        typeBoxes[i].setBounds    (bandBounds.removeFromTop   (20).reduced(2, 0));
        channelBoxes[i].setBounds (bandBounds.removeFromTop   (20).reduced(2, 0));
        enableButtons[i].setBounds(bandBounds.removeFromBottom(20).reduced(2, 2));

        const int paramH = bandBounds.getHeight() / 3;

        gainLabels[i].setBounds   (bandBounds.removeFromTop(paramH).reduced(2, 0));
        freqLabels[i].setBounds   (bandBounds.removeFromTop(paramH).reduced(2, 0));
        qLabels[i].setBounds      (bandBounds.removeFromTop(paramH).reduced(2, 0));
    }
}
