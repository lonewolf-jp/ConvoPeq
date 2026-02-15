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

    // Min Phase Toggle
    minPhaseButton.setColour(juce::ToggleButton::textColourId, juce::Colours::white);
    minPhaseButton.setToggleState(engine.getConvolverUseMinPhase(), juce::dontSendNotification);
    minPhaseButton.setTooltip("Convert IR to Minimum Phase (Reduces latency and pre-ringing)");
    minPhaseButton.onClick = [this] {
        engine.setConvolverUseMinPhase(minPhaseButton.getToggleState());
    };
    addAndMakeVisible(minPhaseButton);

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

    // IR情報ラベル
    irInfoLabel.setText("No IR loaded", juce::dontSendNotification);
    irInfoLabel.setJustificationType(juce::Justification::centred);
    irInfoLabel.setColour(juce::Label::textColourId,
                         juce::Colours::orange.withAlpha(0.8f));
    irInfoLabel.setFont(juce::FontOptions(13.0f, juce::Font::bold));
    addAndMakeVisible(irInfoLabel);

    engine.getConvolverProcessor().addChangeListener(this);
}

ConvolverControlPanel::~ConvolverControlPanel()
{
    engine.getConvolverProcessor().removeChangeListener(this);
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

    // コントロール行
    auto controlRow = bounds.removeFromTop(28);

    // Load IRボタン
    loadIRButton.setBounds(controlRow.removeFromLeft(90));
    controlRow.removeFromLeft(10);

    // Min Phase
    minPhaseButton.setBounds(controlRow.removeFromLeft(90));
    controlRow.removeFromLeft(5);

    // Dry/Wet Mix
    mixLabel.setBounds(controlRow.removeFromLeft(65));
    controlRow.removeFromLeft(5);
    mixSlider.setBounds(controlRow);

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
        fileChooser = std::make_unique<juce::FileChooser>("Select Impulse Response (IR) File",
                                  juce::File::getSpecialLocation(
                                      juce::File::userDocumentsDirectory),
                                  "*.wav;*.aif;*.aiff;*.flac");

        const auto chooserFlags = juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles;

        fileChooser->launchAsync(chooserFlags, [this](const juce::FileChooser& fc)
        {
            if (fc.getResults().isEmpty())
                return;

            juce::File irFile = fc.getResult();

            // 安全なリクエスト経由でロード
            engine.requestConvolverPreset(irFile);
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
}

void ConvolverControlPanel::changeListenerCallback(juce::ChangeBroadcaster*)
{
    updateIRInfo();
}

//--------------------------------------------------------------
// updateIRInfo
//--------------------------------------------------------------
void ConvolverControlPanel::updateIRInfo()
{
    auto& convolver = engine.getConvolverProcessor();

    // UIコントロールをプロセッサの状態と同期
    mixSlider.setValue(convolver.getMix(), juce::dontSendNotification);
    minPhaseButton.setToggleState(convolver.getUseMinPhase(), juce::dontSendNotification);

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
