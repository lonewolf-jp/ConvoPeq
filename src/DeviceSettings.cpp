//============================================================================
// DeviceSettings.cpp  ── v0.2 (JUCE 8.0.12対応)
//============================================================================
#include "DeviceSettings.h"
#include "NoiseShaperLearningComponent.h"

namespace
{
juce::AudioDeviceManager::AudioDeviceSetup makeRelaxedSetupFromXml(const juce::XmlElement& xml)
{
    juce::AudioDeviceManager::AudioDeviceSetup setup;

    if (xml.getStringAttribute("audioDeviceName").isNotEmpty())
    {
        setup.inputDeviceName = setup.outputDeviceName = xml.getStringAttribute("audioDeviceName");
    }
    else
    {
        setup.inputDeviceName  = xml.getStringAttribute("audioInputDeviceName");
        setup.outputDeviceName = xml.getStringAttribute("audioOutputDeviceName");
    }

    setup.sampleRate = 0.0;
    setup.bufferSize = 0;
    setup.useDefaultInputChannels = true;
    setup.useDefaultOutputChannels = true;
    setup.inputChannels.clear();
    setup.outputChannels.clear();
    return setup;
}

juce::String makeAdaptiveCoeffPropertyName(double sampleRate, int coeffIndex)
{
    return "adaptiveCoeff_" + juce::String(static_cast<int>(sampleRate + 0.5)) + "_" + juce::String(coeffIndex);
}
}

//==============================================================================
// BlacklistedASIODeviceType - ASIOドライバをラップしてブラックリストフィルタを適用するクラス
//
// ■ 目的:
// JUCEのASIOデバイス管理に介入し、特定のドライバ（不安定なものや不要なもの）をデバイスリストから除外します。
// これにより、シングルクライアントASIO（BRAVO-HD, ASIO4ALL等）に起因する排他制御の問題や、
// 特定ドライバの不安定性によるアプリケーションのクラッシュを未然に防ぎます。
//
// ■ 実装に関する注意:
// このクラスは、JUCEの内部実装（`AudioDeviceManager`が`OwnedArray`で`AudioIODeviceType`を管理していること）に依存しています。
// `const_cast`を用いて内部の読み取り専用配列を書き換えるというハックを行っているため、ASIOデバイスのブラックリスト適用を実現しています。
// 将来のJUCEバージョンで互換性が失われる可能性があります。
//==============================================================================
class BlacklistedASIODeviceType : public juce::AudioIODeviceType
{
public:
    BlacklistedASIODeviceType (std::unique_ptr<juce::AudioIODeviceType> original, const AsioBlacklist& bl)
        : AudioIODeviceType (original->getTypeName()),
          inner (std::move (original)),
          blacklist (bl)
    {
    }

    void scanForDevices() override
    {
        inner->scanForDevices();
    }

    juce::StringArray getDeviceNames (bool wantInputNames) const override
    {
        auto names = inner->getDeviceNames (wantInputNames);

        // ブラックリストにあるデバイスを除外
        for (int i = names.size(); --i >= 0;)
        {
            if (blacklist.isBlacklisted (names[i]))
                names.remove (i);
        }

        return names;
    }

    int getDefaultDeviceIndex (bool forInput) const override
    {
        auto innerNames = inner->getDeviceNames (forInput);
        int innerDefault = inner->getDefaultDeviceIndex (forInput);

        if (innerDefault >= 0 && innerDefault < innerNames.size())
        {
            // デフォルトデバイスがブラックリスト入りしていないか確認
            juce::String defaultName = innerNames[innerDefault];
            if (! blacklist.isBlacklisted (defaultName))
                return getDeviceNames (forInput).indexOf (defaultName);
        }

        // フォールバック: デフォルトが無効またはブラックリスト入りの場合、最初の有効なデバイスを返す
        auto filteredNames = getDeviceNames(forInput);
        if (!filteredNames.isEmpty())
            return 0;

        return -1;
    }

    int getIndexOfDevice (juce::AudioIODevice* device, bool asInput) const override
    {
        int innerIndex = inner->getIndexOfDevice (device, asInput);
        if (innerIndex >= 0)
        {
            auto innerNames = inner->getDeviceNames (asInput);
            if (innerIndex < innerNames.size())
                return getDeviceNames (asInput).indexOf (innerNames[innerIndex]);
        }
        return -1;
    }

    bool hasSeparateInputsAndOutputs() const override
    {
        return inner->hasSeparateInputsAndOutputs();
    }

    juce::AudioIODevice* createDevice (const juce::String& outputDeviceName,
                                       const juce::String& inputDeviceName) override
    {
        // 生成時にも念のためチェック
        if (blacklist.isBlacklisted (outputDeviceName) || blacklist.isBlacklisted (inputDeviceName))
            return nullptr;

        return inner->createDevice (outputDeviceName, inputDeviceName);
    }

private:
    std::unique_ptr<juce::AudioIODeviceType> inner;
    const AsioBlacklist& blacklist;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (BlacklistedASIODeviceType)
};

//==============================================================================
DeviceSettings::DeviceSettings (juce::AudioDeviceManager& adm, AudioEngine& engine)
    : audioDeviceManager (adm),
      audioEngine (engine)
    , filterTypeTabs (juce::TabbedButtonBar::TabsAtTop)
{
    selector.reset (new juce::AudioDeviceSelectorComponent (
        audioDeviceManager,
        1, 2,    // min/max input channels
        1, 2,    // min/max output channels
        true,    // show MIDI inputs
        true,    // show MIDI outputs
        true,    // stereo pairs
        false    // hide advanced options
    ));

    addAndMakeVisible (*selector);

    // Filter Type Tabs
    addAndMakeVisible(filterTypeTabs);
    filterTypeTabs.addTab("IIR (Low Latency)", juce::Colours::darkgrey, new juce::Component(), true);
    filterTypeTabs.addTab("Linear Phase (FIR)", juce::Colours::darkgrey, new juce::Component(), true);
    filterTypeTabs.setCurrentTabIndex(engine.getOversamplingType() == AudioEngine::OversamplingType::LinearPhase ? 1 : 0);
    // TabbedButtonBarの変更を監視
    filterTypeTabs.getTabbedButtonBar().addChangeListener(this);

    // Oversampling Controls
    addAndMakeVisible(oversamplingLabel);
    oversamplingLabel.setText("Oversampling:", juce::dontSendNotification);
    oversamplingLabel.setJustificationType(juce::Justification::centredLeft);

    addAndMakeVisible(oversamplingComboBox);
    oversamplingComboBox.addItem("Auto", 1);

    const double sr = audioEngine.getSampleRate();

    oversamplingComboBox.addItem("1x (None)", 2); // Always available
    oversamplingComboBox.addItem("2x", 3);         // Always available

    //Conditionally add 4x and 8x options based on sample rate
    if (sr <= 192000)
        oversamplingComboBox.addItem("4x", 4);

    if (sr <= 96000)
        oversamplingComboBox.addItem("8x", 5);

    oversamplingComboBox.onChange = [this] {
        // 【パッチ5】重複する setOversamplingFactor 呼び出しを除去
        // 旧コードは同一ラムダ内で setOversamplingFactor を2回呼んでいた。
        // 1回目: std::map ルックアップ方式、2回目: if-else チェーン方式。
        // どちらも同じ値を算出するため2回目は完全なデッドコードだった。
        // AudioEngine::setOversamplingFactor() の if-guard により実際には
        // 2度目の rebuild は防がれるが、コードの意図が不明確で誤読を招く。
        // 正しく整理された単一の変換テーブルに統合する。
        int selectedId = oversamplingComboBox.getSelectedId();
        int factor = 0; // default = Auto
        if      (selectedId == 2) factor = 1;
        else if (selectedId == 3) factor = 2;
        else if (selectedId == 4) factor = 4;
        else if (selectedId == 5) factor = 8;
        audioEngine.setOversamplingFactor(factor);
    };

    // Dither Bit Depth Controls
    addAndMakeVisible(bitDepthLabel);
    bitDepthLabel.setText("Dither Bit Depth:", juce::dontSendNotification);
    bitDepthLabel.setJustificationType(juce::Justification::centredLeft);

    addAndMakeVisible(bitDepthComboBox);
   bitDepthComboBox.onChange = [this] {
        int id = bitDepthComboBox.getSelectedId();
        if (id == 999)
            audioEngine.setDitherBitDepth(0);   // Off
        else if (id > 0)
        {    audioEngine.setDitherBitDepth(id);  // 16/24/32
        }
    };

    // Noise Shaper Type Controls
    addAndMakeVisible(noiseShaperLabel);
    noiseShaperLabel.setText("Noise Shaper:", juce::dontSendNotification);
    noiseShaperLabel.setJustificationType(juce::Justification::centredLeft);

    addAndMakeVisible(noiseShaperComboBox);
    noiseShaperComboBox.addItem("4th-order", 1);
    noiseShaperComboBox.addItem("9th-order", 2);
    noiseShaperComboBox.addItem("9th-order adaptive", 3);
    noiseShaperComboBox.onChange = [this] {
        const int id = noiseShaperComboBox.getSelectedId();
        if (id == 1)
            audioEngine.setNoiseShaperType(AudioEngine::NoiseShaperType::Fixed4Tap);
        else if (id == 2)
            audioEngine.setNoiseShaperType(AudioEngine::NoiseShaperType::Psychoacoustic);
        else if (id == 3)
            audioEngine.setNoiseShaperType(AudioEngine::NoiseShaperType::Adaptive9thOrder);

        updateNoiseShaperControls();
    };

    addAndMakeVisible(adaptiveLearningButton);
    adaptiveLearningButton.setTooltip("Open the adaptive 9th-order learning window");
    adaptiveLearningButton.onClick = [this] { showAdaptiveLearningWindow(); };

    addAndMakeVisible(fixedNoiseLogIntervalLabel);
    fixedNoiseLogIntervalLabel.setText("NS Log Interval:", juce::dontSendNotification);
    fixedNoiseLogIntervalLabel.setJustificationType(juce::Justification::centredLeft);

    addAndMakeVisible(fixedNoiseLogIntervalComboBox);
    fixedNoiseLogIntervalComboBox.addItem("500 ms", 500);
    fixedNoiseLogIntervalComboBox.addItem("1000 ms", 1000);
    fixedNoiseLogIntervalComboBox.addItem("2000 ms", 2000);
    fixedNoiseLogIntervalComboBox.addItem("5000 ms", 5000);
    fixedNoiseLogIntervalComboBox.onChange = [this] {
        const int intervalMs = fixedNoiseLogIntervalComboBox.getSelectedId();
        if (intervalMs > 0)
            audioEngine.setFixedNoiseLogIntervalMs(intervalMs);
    };

    addAndMakeVisible(fixedNoiseWindowLabel);
    fixedNoiseWindowLabel.setText("NS Window:", juce::dontSendNotification);
    fixedNoiseWindowLabel.setJustificationType(juce::Justification::centredLeft);

    addAndMakeVisible(fixedNoiseWindowComboBox);
    fixedNoiseWindowComboBox.addItem("2048", 2048);
    fixedNoiseWindowComboBox.addItem("4096", 4096);
    fixedNoiseWindowComboBox.addItem("8192", 8192);
    fixedNoiseWindowComboBox.addItem("16384", 16384);
    fixedNoiseWindowComboBox.addItem("32768", 32768);
    fixedNoiseWindowComboBox.onChange = [this] {
        const int samples = fixedNoiseWindowComboBox.getSelectedId();
        if (samples > 0)
            audioEngine.setFixedNoiseWindowSamples(samples);
    };

    // Input Headroom Controls
    addAndMakeVisible(inputHeadroomLabel);
    inputHeadroomLabel.setText("Input Headroom:", juce::dontSendNotification);
    inputHeadroomLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(inputHeadroomEditor);
    inputHeadroomEditor.setInputRestrictions(0, "-0123456789.");
    inputHeadroomEditor.setText(juce::String(audioEngine.getInputHeadroomDb(), 1));
    inputHeadroomEditor.setJustification(juce::Justification::right);
    inputHeadroomEditor.onTextChange = [this] {
        double val = inputHeadroomEditor.getText().getDoubleValue();
        if (val < -12.0) val = -12.0;
        if (val > 0.0) val = 0.0;
        audioEngine.setInputHeadroomDb(static_cast<float>(val));
    };


    // Output Makeup Controls
    addAndMakeVisible(outputMakeupLabel);
    outputMakeupLabel.setText("Output Makeup:", juce::dontSendNotification);
    outputMakeupLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(outputMakeupEditor);
    outputMakeupEditor.setInputRestrictions(0, "-0123456789.");
    outputMakeupEditor.setText(juce::String(audioEngine.getOutputMakeupDb(), 1));
    outputMakeupEditor.setJustification(juce::Justification::right);
    outputMakeupEditor.onTextChange = [this] {
        double val = outputMakeupEditor.getText().getDoubleValue();
        if (val < -6.0) val = -6.0;
        if (val > 12.0) val = 12.0;
        audioEngine.setOutputMakeupDb(static_cast<float>(val));
    };


    // デバイス変更を監視してビット深度リストを更新
    audioDeviceManager.addChangeListener(this);

    // 初期値設定
    const std::map<int, int> factorToId = {{0, 1}, {1, 2}, {2, 3}, {4, 4}, {8, 5}};
    int currentFactor = audioEngine.getOversamplingFactor();
    if (auto it = factorToId.find(currentFactor); it != factorToId.end())
    {
        oversamplingComboBox.setSelectedId(it->second, juce::dontSendNotification);
    }
    else {
        oversamplingComboBox.setSelectedId(1, juce::dontSendNotification); // Default to Auto
    }

    switch (audioEngine.getNoiseShaperType())
    {
        case AudioEngine::NoiseShaperType::Fixed4Tap:
            noiseShaperComboBox.setSelectedId(1, juce::dontSendNotification);
            break;
        case AudioEngine::NoiseShaperType::Adaptive9thOrder:
            noiseShaperComboBox.setSelectedId(3, juce::dontSendNotification);
            break;
        case AudioEngine::NoiseShaperType::Psychoacoustic:
        default:
            noiseShaperComboBox.setSelectedId(2, juce::dontSendNotification);
            break;
    }

    {
        const int intervalMs = audioEngine.getFixedNoiseLogIntervalMs();
        fixedNoiseLogIntervalComboBox.setSelectedId(intervalMs, juce::dontSendNotification);
        if (fixedNoiseLogIntervalComboBox.getSelectedId() == 0)
            fixedNoiseLogIntervalComboBox.setSelectedId(2000, juce::dontSendNotification);
    }

    {
        const int windowSamples = audioEngine.getFixedNoiseWindowSamples();
        fixedNoiseWindowComboBox.setSelectedId(windowSamples, juce::dontSendNotification);
        if (fixedNoiseWindowComboBox.getSelectedId() == 0)
            fixedNoiseWindowComboBox.setSelectedId(8192, juce::dontSendNotification);
    }

    updateBitDepthList();
    updateNoiseShaperControls();
    // loadSettings()後にUIの値を更新

    updateGainStagingDisplay();
    startTimerHz(5);
}

DeviceSettings::~DeviceSettings()
{
    stopTimer();
    audioDeviceManager.removeChangeListener(this);
    filterTypeTabs.getTabbedButtonBar().removeChangeListener(this);
}

void DeviceSettings::resized()
{
    auto bounds = getLocalBounds();
    // Adaptive learningボタンの下の余白を詰めるため、controlsAreaの高さを自動計算
    constexpr int rowHeight = 30;
    constexpr int numRows = 6; // Dither, Input, Output, Tabs, Over/Noise, Adaptive
    auto controlsArea = bounds.removeFromTop(rowHeight * numRows); // 必要な分だけ
    auto row1 = controlsArea.removeFromTop(rowHeight); // Dither Bit Depth
    auto row2 = controlsArea.removeFromTop(rowHeight); // Input Headroom
    auto row3 = controlsArea.removeFromTop(rowHeight); // Output Makeup
    auto row4 = controlsArea.removeFromTop(rowHeight); // FilterTypeTabs
    auto row5 = controlsArea.removeFromTop(rowHeight); // Oversampling/NoiseShaper
    auto row6 = controlsArea.removeFromTop(rowHeight); // Adaptive learning

    // 1行目: Dither Bit Depth
    bitDepthLabel.setBounds(row1.removeFromLeft(200).reduced(5));
    bitDepthComboBox.setBounds(row1.removeFromLeft(120).reduced(2));

    // 2行目: Input Headroom
    inputHeadroomLabel.setBounds(row2.removeFromLeft(200).reduced(5));
    inputHeadroomEditor.setBounds(row2.removeFromLeft(120).reduced(5));

    // 3行目: Output Makeup
    outputMakeupLabel.setBounds(row3.removeFromLeft(200).reduced(5));
    outputMakeupEditor.setBounds(row3.removeFromLeft(120).reduced(5));

    // 4行目: FilterTypeTabs
    filterTypeTabs.setBounds(row4.removeFromLeft(400).reduced(2));

    // 5行目: Oversampling/NoiseShaper
    oversamplingLabel.setBounds(row5.removeFromLeft(120).reduced(5));
    oversamplingComboBox.setBounds(row5.removeFromLeft(100).reduced(2));
    noiseShaperLabel.setBounds(row5.removeFromLeft(120).reduced(5));
    // NoiseShaperの位置・幅を記録
    auto nsComboX = row5.getX();
    auto nsComboY = row5.getY();
    auto nsComboW = 160;
    auto nsComboH = row5.getHeight();
    noiseShaperComboBox.setBounds(nsComboX, nsComboY, nsComboW, nsComboH - 2);

    // 6行目: Adaptive learningボタンをNoiseShaperの真下・同じ幅で配置
    adaptiveLearningButton.setBounds(nsComboX, nsComboY + nsComboH, nsComboW, nsComboH - 2);

    fixedNoiseLogIntervalLabel.setBounds(0, 0, 0, 0); // 非表示時のダミー配置
    fixedNoiseLogIntervalComboBox.setBounds(0, 0, 0, 0);
    fixedNoiseWindowLabel.setBounds(0, 0, 0, 0);
    fixedNoiseWindowComboBox.setBounds(0, 0, 0, 0);

    // Audio device selectorをAdaptive learningボタンの直下に詰めて配置
    if (selector != nullptr) {
        auto selectorBounds = bounds;
        selectorBounds.setY(nsComboY + nsComboH * 2); // Adaptive learningボタンの下端から開始
        selector->setBounds(selectorBounds);
    }
}

void DeviceSettings::changeListenerCallback (juce::ChangeBroadcaster* source)
{
    // ソースを判定して処理を分岐
    if (source == &filterTypeTabs.getTabbedButtonBar())
    {
        // タブの変更チェック
        auto type = (filterTypeTabs.getCurrentTabIndex() == 1) ? AudioEngine::OversamplingType::LinearPhase : AudioEngine::OversamplingType::IIR;
        if (type != audioEngine.getOversamplingType())
            audioEngine.setOversamplingType(type);
    }
    else if (source == &audioDeviceManager)
    {
        updateBitDepthList();
    }
}

void DeviceSettings::timerCallback()
{
    updateGainStagingDisplay();
}

void DeviceSettings::showAdaptiveLearningWindow()
{
    audioEngine.setNoiseShaperType(AudioEngine::NoiseShaperType::Adaptive9thOrder);
    noiseShaperComboBox.setSelectedId(3, juce::dontSendNotification);
    updateNoiseShaperControls();

    if (adaptiveLearningWindow != nullptr)
    {
        adaptiveLearningWindow->setVisible(true);
        adaptiveLearningWindow->toFront(true);
        return;
    }

    juce::DialogWindow::LaunchOptions options;
    options.content.setOwned(new NoiseShaperLearningComponent(audioEngine));
    options.dialogTitle = "Adaptive Noise Shaper Learning";
    options.dialogBackgroundColour = juce::Colour(0xff20252b);
    options.escapeKeyTriggersCloseButton = true;
    options.useNativeTitleBar = true;
    options.resizable = true;

    if (auto* window = options.launchAsync())
    {
        window->setResizeLimits(480, 280, 900, 700);
        window->centreWithSize(560, 360);
        adaptiveLearningWindow = window;
    }
}

void DeviceSettings::updateNoiseShaperControls()
{
    const bool showFixedControls = noiseShaperComboBox.getSelectedId() == 1;
    const bool showAdaptiveButton = noiseShaperComboBox.getSelectedId() == 3;

    fixedNoiseLogIntervalLabel.setVisible(showFixedControls);
    fixedNoiseLogIntervalComboBox.setVisible(showFixedControls);
    fixedNoiseWindowLabel.setVisible(showFixedControls);
    fixedNoiseWindowComboBox.setVisible(showFixedControls);
    adaptiveLearningButton.setVisible(showAdaptiveButton);
    adaptiveLearningButton.setEnabled(showAdaptiveButton);

    resized();
}

void DeviceSettings::updateGainStagingDisplay()
{
    const bool eqBypassed = audioEngine.isEqBypassRequested();
    const bool convBypassed = audioEngine.isConvolverBypassRequested();
    const auto order = audioEngine.getProcessingOrder();

    float inputMaxDb = 0.0f;
    float makeupMinDb = 6.0f;
    float makeupMaxDb = 12.0f;
    juce::String modeText;

    if (convBypassed && !eqBypassed)
    {
        modeText = "PEQ only";
        inputMaxDb = 0.0f;
        makeupMinDb = -6.0f;
        makeupMaxDb = 0.0f;
    }
    else if (!convBypassed && !eqBypassed && order == AudioEngine::ProcessingOrder::EQThenConvolver)
    {
        modeText = "PEQ -> Conv";
        inputMaxDb = 0.0f;
        makeupMinDb = 6.0f;
        makeupMaxDb = 10.0f;
    }
    else if (eqBypassed && !convBypassed)
    {
        modeText = "Conv only";
        inputMaxDb = -6.0f;
        makeupMinDb = 6.0f;
        makeupMaxDb = 12.0f;
    }
    else
    {
        modeText = "Conv -> PEQ";
        inputMaxDb = -6.0f;
        makeupMinDb = 6.0f;
        makeupMaxDb = 12.0f;
    }

    const juce::String inputText = "Input Headroom (" + juce::String(-12.0f, 1) + ".." + juce::String(inputMaxDb, 1) + " dB):";
    const juce::String makeupText = "Output Makeup (" + juce::String(makeupMinDb, 1) + ".." + juce::String(makeupMaxDb, 1) + " dB):";
    const juce::String signature = modeText + "|" + inputText + "|" + makeupText;

    if (signature != gainDisplaySignature)
    {
        gainDisplaySignature = signature;
        inputHeadroomLabel.setText(inputText, juce::dontSendNotification);
        outputMakeupLabel.setText(makeupText, juce::dontSendNotification);
        const juce::String modeTip = "Current mode: " + modeText;
        inputHeadroomLabel.setTooltip(modeTip);
        outputMakeupLabel.setTooltip(modeTip);
    }

    const double currentInput = static_cast<double>(audioEngine.getInputHeadroomDb());
    const double currentMakeup = static_cast<double>(audioEngine.getOutputMakeupDb());
    if (std::abs(inputHeadroomEditor.getText().getDoubleValue() - currentInput) > 1.0e-6)
        inputHeadroomEditor.setText(juce::String(currentInput, 1), juce::dontSendNotification);
    if (std::abs(outputMakeupEditor.getText().getDoubleValue() - currentMakeup) > 1.0e-6)
        outputMakeupEditor.setText(juce::String(currentMakeup, 1), juce::dontSendNotification);
}

void DeviceSettings::updateBitDepthList()
{
    juce::Array<int> supportedBitDepths;

    // 標準的なビット深度を常に表示（送出前量子化ターゲットとして利用可能）。
    supportedBitDepths.add(16);
    supportedBitDepths.add(24);
    supportedBitDepths.add(32);

    // 現在開いているデバイスがあれば、その実デバイスの現在bit depthを追加（重複除去）。
    // JUCE 8.0.12 では available bit depth 一覧APIがないため current 値のみ参照する。
    if (auto* device = audioDeviceManager.getCurrentAudioDevice())
    {
        int current = device->getCurrentBitDepth();
        if (current > 0 && !supportedBitDepths.contains(current))
            supportedBitDepths.add(current);
    }

    supportedBitDepths.sort();

    // UI更新
    bitDepthComboBox.clear();
    int maxBitDepth = 0;

    for (int depth : supportedBitDepths)
    {
        bitDepthComboBox.addItem(juce::String(depth) + " bit", depth); // ID = depth
        if (depth > maxBitDepth)
            maxBitDepth = depth;
    }

    // "Off" オプションを追加
    bitDepthComboBox.addSeparator();
    bitDepthComboBox.addItem("Off", 999);

    // 選択状態の決定
    // 1. 現在のエンジンの設定が有効ならそれを維持
    // 2. 未設定(0)または無効なら、最大ビット深度を選択 (デフォルト)
    int currentEngineDepth = audioEngine.getDitherBitDepth();

    if (currentEngineDepth == 0)
    {
        bitDepthComboBox.setSelectedId(999, juce::dontSendNotification);
    }
    else if (supportedBitDepths.contains(currentEngineDepth))
    {
        bitDepthComboBox.setSelectedId(currentEngineDepth, juce::dontSendNotification);
    }
    else
    {
        // デフォルトで最大ビット深度
        if (maxBitDepth > 0)
        {
            bitDepthComboBox.setSelectedId(maxBitDepth, juce::dontSendNotification);
            // エンジンも更新
            audioEngine.setDitherBitDepth(maxBitDepth);
        }
        else // フォールバック
        {
            bitDepthComboBox.setSelectedId(999, juce::dontSendNotification);
            audioEngine.setDitherBitDepth(0);
        }
    }
}

juce::File DeviceSettings::getSettingsFile()
{
    auto appDataDir = juce::File::getSpecialLocation (juce::File::userApplicationDataDirectory)
                          .getChildFile ("ConvoPeq");

    if (! appDataDir.exists())
        appDataDir.createDirectory();

    return appDataDir.getChildFile ("device_settings.xml");
}

void DeviceSettings::saveSettings (const juce::AudioDeviceManager& deviceManager, const AudioEngine& engine)
{
    if (auto xml = deviceManager.createStateXml())
    {
        // ビット深度設定を追加属性として保存
        xml->setAttribute("ditherBitDepth", engine.getDitherBitDepth());
        xml->setAttribute("noiseShaperType", (int)engine.getNoiseShaperType());
        xml->setAttribute("fixedNoiseLogIntervalMs", engine.getFixedNoiseLogIntervalMs());
        xml->setAttribute("fixedNoiseWindowSamples", engine.getFixedNoiseWindowSamples());
        // オーバーサンプリング設定を追加
        xml->setAttribute("oversamplingFactor", engine.getOversamplingFactor());
        // フィルタタイプ設定を追加
        xml->setAttribute("oversamplingType", (int)engine.getOversamplingType());
        // 入力ヘッドルーム設定を追加
        xml->setAttribute("outputMakeupDb", engine.getOutputMakeupDb());
        xml->setAttribute("inputHeadroomDb", engine.getInputHeadroomDb());
        for (int bankIndex = 0; bankIndex < AudioEngine::getAdaptiveSampleRateBankCount(); ++bankIndex)
        {
            const double bankSampleRate = AudioEngine::getAdaptiveSampleRateBankHz(bankIndex);
            double adaptiveCoefficients[kAdaptiveNoiseShaperOrder] = {};
            engine.getAdaptiveCoefficientsForSampleRate(bankSampleRate, adaptiveCoefficients, kAdaptiveNoiseShaperOrder);

            for (int coeffIndex = 0; coeffIndex < kAdaptiveNoiseShaperOrder; ++coeffIndex)
                xml->setAttribute(makeAdaptiveCoeffPropertyName(bankSampleRate, coeffIndex), adaptiveCoefficients[coeffIndex]);
        }

        xml->writeTo (getSettingsFile());
    }
}

//--------------------------------------------------------------
// loadSettings
// 設定ファイルからAudioDeviceManagerを復元する
// JUCE v8.0.12 完全対応版（MMCSSはJUCE内部で自動管理）
//--------------------------------------------------------------
void DeviceSettings::loadSettings (juce::AudioDeviceManager& deviceManager, AudioEngine& engine)
{
    // ASIOドライバの切り替え時に発生しうるフリーズを防ぐため、初期化前に一度デバイスを閉じる
    deviceManager.closeAudioDevice();

    const auto initialiseDefaultDevice = [&deviceManager]() -> juce::String
    {
        return deviceManager.initialise (2, 2, nullptr, true);
    };

    auto file = getSettingsFile();

    if (file.existsAsFile())
    {
        if (auto xml = juce::XmlDocument::parse (file))
        {
            // 保存された設定でデバイスを初期化する（入力2ch、出力2chを要求）
            juce::String error = deviceManager.initialise (2, 2, xml.get(), false);

            if (error.isNotEmpty())
            {
                juce::String recoveredWithRelaxedSetup;
                const auto savedDeviceType = xml->getStringAttribute("deviceType");

                if (savedDeviceType.isNotEmpty()
                    && savedDeviceType != deviceManager.getCurrentAudioDeviceType())
                {
                    deviceManager.setCurrentAudioDeviceType(savedDeviceType, false);
                }

                auto relaxedSetup = makeRelaxedSetupFromXml(*xml);
                recoveredWithRelaxedSetup = deviceManager.setAudioDeviceSetup(relaxedSetup, false);

                if (recoveredWithRelaxedSetup.isNotEmpty() || deviceManager.getCurrentAudioDevice() == nullptr)
                {
                    const juce::String fallbackError = initialiseDefaultDevice();

                    if (fallbackError.isNotEmpty() || deviceManager.getCurrentAudioDevice() == nullptr)
                    {
                        juce::NativeMessageBox::showAsync(
                            juce::MessageBoxOptions()
                                .withIconType(juce::MessageBoxIconType::WarningIcon)
                                .withTitle("Audio Device Settings")
                                .withMessage("Could not restore the saved audio device settings.\nError: " + error
                                             + "\n\nSafe retry on the saved device also failed.\nError: "
                                             + (recoveredWithRelaxedSetup.isNotEmpty() ? recoveredWithRelaxedSetup : "The saved device is unavailable.")
                                             + "\n\nCould not start the fallback device either.\nError: "
                                             + (fallbackError.isNotEmpty() ? fallbackError : "No audio device available."))
                                .withButton("OK"),
                            nullptr);
                    }
                    else
                    {
                        juce::Logger::writeToLog("Audio device restore failed, using default device instead: " + error);
                    }
                }
                else
                {
                    juce::Logger::writeToLog("Audio device restore needed relaxed retry and succeeded: " + error);
                }
            }

            // ビット深度設定の読み込み (デフォルト0 = 自動/最大)
            int bitDepth = xml->getIntAttribute("ditherBitDepth", 0);
            engine.setDitherBitDepth(bitDepth);

            // ノイズシェーパー種類の読み込み (デフォルト0 = Current)
            int shaperType = xml->getIntAttribute("noiseShaperType", 0);
            engine.setNoiseShaperType((AudioEngine::NoiseShaperType)shaperType);

            {
                bool hasBankedAdaptiveCoefficients = false;

                for (int bankIndex = 0; bankIndex < AudioEngine::getAdaptiveSampleRateBankCount(); ++bankIndex)
                {
                    const double bankSampleRate = AudioEngine::getAdaptiveSampleRateBankHz(bankIndex);
                    double adaptiveCoefficients[kAdaptiveNoiseShaperOrder] = {};
                    bool hasBankCoefficients = false;

                    engine.getAdaptiveCoefficientsForSampleRate(bankSampleRate, adaptiveCoefficients, kAdaptiveNoiseShaperOrder);
                    for (int coeffIndex = 0; coeffIndex < kAdaptiveNoiseShaperOrder; ++coeffIndex)
                    {
                        const auto attributeName = makeAdaptiveCoeffPropertyName(bankSampleRate, coeffIndex);
                        if (xml->hasAttribute(attributeName))
                        {
                            adaptiveCoefficients[coeffIndex] = xml->getDoubleAttribute(attributeName, adaptiveCoefficients[coeffIndex]);
                            hasBankCoefficients = true;
                            hasBankedAdaptiveCoefficients = true;
                        }
                    }

                    if (hasBankCoefficients)
                        engine.setAdaptiveCoefficientsForSampleRate(bankSampleRate, adaptiveCoefficients, kAdaptiveNoiseShaperOrder);
                }

                if (!hasBankedAdaptiveCoefficients)
                {
                    double legacyAdaptiveCoefficients[kAdaptiveNoiseShaperOrder] = {};
                    bool hasLegacyAdaptiveCoefficients = false;

                    for (int coeffIndex = 0; coeffIndex < kAdaptiveNoiseShaperOrder; ++coeffIndex)
                    {
                        const auto attributeName = "adaptiveCoeff" + juce::String(coeffIndex);
                        if (xml->hasAttribute(attributeName))
                        {
                            legacyAdaptiveCoefficients[coeffIndex] = xml->getDoubleAttribute(attributeName, legacyAdaptiveCoefficients[coeffIndex]);
                            hasLegacyAdaptiveCoefficients = true;
                        }
                    }

                    if (hasLegacyAdaptiveCoefficients)
                    {
                        for (int bankIndex = 0; bankIndex < AudioEngine::getAdaptiveSampleRateBankCount(); ++bankIndex)
                            engine.setAdaptiveCoefficientsForSampleRate(AudioEngine::getAdaptiveSampleRateBankHz(bankIndex),
                                                                       legacyAdaptiveCoefficients,
                                                                       kAdaptiveNoiseShaperOrder);
                    }
                }
            }

            // Fixed 4-tap 比較ログ設定
            int logIntervalMs = xml->getIntAttribute("fixedNoiseLogIntervalMs", 2000);
            engine.setFixedNoiseLogIntervalMs(logIntervalMs);
            int windowSamples = xml->getIntAttribute("fixedNoiseWindowSamples", 8192);
            engine.setFixedNoiseWindowSamples(windowSamples);

            // オーバーサンプリング設定の読み込み (デフォルト0 = 自動)
            int oversampling = xml->getIntAttribute("oversamplingFactor", 0);
            engine.setOversamplingFactor(oversampling);

            // 入力ヘッドルーム設定の読み込み (デフォルト -6.0dB)
            float headroom = (float)xml->getDoubleAttribute("inputHeadroomDb", -6.0);
            engine.setInputHeadroomDb(headroom);

            // Output Makeup設定の読み込み (デフォルト +15.0dB)
            float makeup = (float)xml->getDoubleAttribute("outputMakeupDb", 12.0); // [Fix] default 15→12 dB
            engine.setOutputMakeupDb(makeup);

            // フィルタタイプ設定の読み込み (デフォルト0 = IIR)
            int type = xml->getIntAttribute("oversamplingType", 0);
            engine.setOversamplingType((AudioEngine::OversamplingType)type);
            return;
        }
    }

    // 設定ファイルが存在しない、または読み込みに失敗した場合はデフォルト初期化
    // MMCSSはJUCE 8.0.12で内部自動管理されるため明示的設定は不要
    initialiseDefaultDevice();

    // デフォルトで最大サンプルレートに設定
    auto* currentDevice = deviceManager.getCurrentAudioDevice();
    if (currentDevice != nullptr)
    {
        auto availableRates = currentDevice->getAvailableSampleRates();
        if (!availableRates.isEmpty())
        {
            // 最大レートを安全に取得
            double maxRate = *std::max_element(availableRates.begin(), availableRates.end());

            auto setup = deviceManager.getAudioDeviceSetup();
            if (std::abs(setup.sampleRate - maxRate) > 1e-6 && maxRate > 0.0)
            {
                setup.sampleRate = maxRate;
                deviceManager.setAudioDeviceSetup(setup, true);
            }
        }
    }

    engine.setDitherBitDepth(0); // 自動設定へ
    engine.setNoiseShaperType(AudioEngine::NoiseShaperType::Psychoacoustic);
    engine.setFixedNoiseLogIntervalMs(2000);
    engine.setFixedNoiseWindowSamples(8192);
    engine.setOversamplingFactor(0); // 自動設定へ
    engine.setInputHeadroomDb(-6.0f); // デフォルト -6dB
    engine.setOutputMakeupDb(12.0f); // [Fix] default 15→12 dB (unity gain)
    engine.setOversamplingType(AudioEngine::OversamplingType::IIR); // デフォルトIIR
}
void DeviceSettings::applyAsioBlacklist (juce::AudioDeviceManager& deviceManager, const AsioBlacklist& blacklist)
{
    // 既存のASIOタイプを探す
    // Note: JUCEのAudioDeviceManagerは、登録済みのAudioIODeviceTypeを外部から安全に
    //       置き換えるAPIを提供していません。
    //       そのため、const_castを用いて内部のOwnedArrayに直接アクセスし、
    //       既存のASIOドライバインスタンスを自作のラッパークラスで置き換えるというハックを行っています。
    //       この方法はJUCEの将来のバージョンで互換性がなくなる可能性があります。

    auto& availableTypes = const_cast<juce::OwnedArray<juce::AudioIODeviceType>&>(deviceManager.getAvailableDeviceTypes());
    int asioIndex = -1;

    for (int i = 0; i < availableTypes.size(); ++i)
    {
        if (availableTypes[i] != nullptr && availableTypes[i]->getTypeName() == "ASIO")
        {
            asioIndex = i;
            break;
        }
    }

    if (asioIndex != -1)
    {
        // remove() の第2引数に false を指定し、OwnedArrayによる自動削除を防ぎます。
        // これにより、オブジェクトの所有権を unique_ptr に安全に移行できます。
        auto* rawPtr = availableTypes[asioIndex];
        availableTypes.remove(asioIndex, false);
        auto originalAsio = std::unique_ptr<juce::AudioIODeviceType>(rawPtr);

        // ラッパーで包んで再登録します。addAudioDeviceType() が所有権を引き継ぎます。
        deviceManager.addAudioDeviceType (std::make_unique<BlacklistedASIODeviceType> (std::move(originalAsio), blacklist));
    }
}
