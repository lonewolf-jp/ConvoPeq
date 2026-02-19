//============================================================================
// ConvolverControlPanel.cpp (JUCE 8.0.12対応)
//
// Convolverコントロールパネルの実装
//============================================================================
#include "ConvolverControlPanel.h"

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

    // Phase Choice ComboBox
    phaseChoiceBox.addItem("Linear Phase", 1);
    phaseChoiceBox.addItem("Minimum Phase", 2);
    phaseChoiceBox.setSelectedId(engine.getConvolverUseMinPhase() ? 2 : 1, juce::dontSendNotification);
    phaseChoiceBox.setTooltip("Select IR Phase Type");
    phaseChoiceBox.setJustificationType(juce::Justification::centred);
    phaseChoiceBox.onChange = [this] {
        engine.setConvolverUseMinPhase(phaseChoiceBox.getSelectedId() == 2);
    };
    addAndMakeVisible(phaseChoiceBox);

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

    // IR情報ラベル
    irInfoLabel.setText("No IR loaded", juce::dontSendNotification);
    irInfoLabel.setJustificationType(juce::Justification::centred);
    irInfoLabel.setColour(juce::Label::textColourId,
                         juce::Colours::orange.withAlpha(0.8f));
    irInfoLabel.setFont(juce::FontOptions(13.0f, juce::Font::bold));
    addAndMakeVisible(irInfoLabel);
}

ConvolverControlPanel::~ConvolverControlPanel()
{
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

    // タイトル行
    bounds.removeFromTop(22);
    bounds.removeFromTop(5);

    // IR情報
    waveformArea = bounds.removeFromTop(60);
    irInfoLabel.setBounds(waveformArea); // 波形エリアに重ねる
    bounds.removeFromTop(8);

    // 3つのコントロール行を定義
    auto controlRow1 = bounds.removeFromTop(28);
    auto controlRow2 = bounds.removeFromTop(28);
    auto controlRow3 = bounds.removeFromTop(28);

    // --- 1行目 ---
    loadIRButton.setBounds(controlRow1.removeFromLeft(90));
    controlRow1.removeFromLeft(10);

    // 位相選択
    phaseChoiceBox.setBounds(controlRow1.removeFromLeft(120));
    controlRow1.removeFromLeft(5);

    // Dry/Wetミックス
    mixLabel.setBounds(controlRow1.removeFromLeft(65));
    controlRow1.removeFromLeft(5);
    mixSlider.setBounds(controlRow1);

    // --- 2行目 ---
    // スムージング時間 (Mixスライダーの下に配置)
    auto smoothingRow = controlRow2;
    smoothingRow.removeFromLeft(phaseChoiceBox.getRight() + 5);
    smoothingTimeLabel.setBounds(smoothingRow.removeFromLeft(65));
    smoothingRow.removeFromLeft(5);
    smoothingTimeSlider.setBounds(smoothingRow);

    // --- 3行目 ---
    // IR長 (Smoothing Timeの下に配置)
    auto lengthRow = controlRow3;
    lengthRow.removeFromLeft(phaseChoiceBox.getRight() + 5);
    irLengthLabel.setBounds(lengthRow.removeFromLeft(65));
    lengthRow.removeFromLeft(5);
    irLengthSlider.setBounds(lengthRow);
    updateWaveformPath();
}

//--------------------------------------------------------------
// buttonClicked
//--------------------------------------------------------------
void ConvolverControlPanel::buttonClicked(juce::Button* button)
{
    if (button == &loadIRButton)
    {
        // ファイル選択ダイアログ
        // JUCE v8.0.12 Recommended Pattern: Use local shared_ptr and capture it in lambda
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

            juce::File irFile = fc.getResult();

            // 安全なリクエスト経由でロード
            safeThis->engine.requestConvolverPreset(irFile);
        });
    }
}

//--------------------------------------------------------------
// sliderValueChanged
//--------------------------------------------------------------
void ConvolverControlPanel::sliderValueChanged(juce::Slider* slider)
{
    if (slider == &mixSlider)
    {
        engine.getConvolverProcessor().setMix(
            static_cast<float>(slider->getValue())
        );
    }
    else if (slider == &smoothingTimeSlider)
    {
        engine.getConvolverProcessor().setSmoothingTime(
            static_cast<float>(slider->getValue()) / 1000.0f
        );
    }
    else if (slider == &irLengthSlider)
    {
        engine.getConvolverProcessor().setTargetIRLength(
            static_cast<float>(slider->getValue())
        );
    }
}

//--------------------------------------------------------------
// updateIRInfo
//--------------------------------------------------------------
void ConvolverControlPanel::updateIRInfo()
{
    auto& convolver = engine.getConvolverProcessor();

    // UIコントロールをプロセッサの状態と同期
    mixSlider.setValue(convolver.getMix(), juce::dontSendNotification);
    phaseChoiceBox.setSelectedId(convolver.getUseMinPhase() ? 2 : 1, juce::dontSendNotification);
    smoothingTimeSlider.setValue(convolver.getSmoothingTime() * 1000.0, juce::dontSendNotification);
    irLengthSlider.setValue(convolver.getTargetIRLength(), juce::dontSendNotification);

    if (convolver.isIRLoaded())
    {
        juce::String info = convolver.getIRName();
        info += " (" + juce::String(convolver.getIRLength()) + " samples)";

        irInfoLabel.setText(info, juce::dontSendNotification);
        irInfoLabel.setColour(juce::Label::textColourId,
                             juce::Colours::lightgreen);
    }
    else
    {
        irInfoLabel.setText("No IR loaded", juce::dontSendNotification);
        irInfoLabel.setColour(juce::Label::textColourId,
                             juce::Colours::orange.withAlpha(0.8f));
    }

    updateWaveformPath();
    repaint();
}

//--------------------------------------------------------------
// updateWaveformPath
//--------------------------------------------------------------
void ConvolverControlPanel::updateWaveformPath()
{
    waveformPath.clear();
    const auto& waveform = engine.getConvolverProcessor().getIRWaveform();

    if (waveform.empty() || waveform.size() < 2 || waveformArea.isEmpty())
        return;

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
}
