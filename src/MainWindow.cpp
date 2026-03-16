//============================================================================
// MainWindow.cpp ── v0.2 (JUCE 8.0.12対応)
//
// メインウィンドウの実装
// UIコンポーネントの配置とオーディオデバイス管理を行う
//============================================================================
#include "MainWindow.h"
#include <cmath>

namespace
{
    juce::String formatSaturationValue(float value)
    {
        return juce::String(value, 2);
    }

    class SettingsWindow : public juce::DocumentWindow
    {
    public:

        SettingsWindow (const juce::String& name, juce::Colour backgroundColour, int buttons)
            : DocumentWindow (name, backgroundColour, buttons)
        {
            setUsingNativeTitleBar (true);
        }

        void closeButtonPressed() override
        {
            if (onClose)
                onClose();

            setVisible (false);
        }

        std::function<void()> onClose;
    };

    // バージョン情報を表示するコンポーネント
    class AboutComponent : public juce::Component
    {
    public:
        AboutComponent()
        {
            setSize (400, 200);
        }

        void paint (juce::Graphics& g) override
        {
            g.fillAll (juce::Colours::darkgrey);

            auto area = getLocalBounds().reduced(20);

            g.setColour (juce::Colours::white);
            g.setFont (juce::FontOptions (24.0f, juce::Font::bold));
            g.drawText (juce::String(ProjectInfo::projectName), area.removeFromTop (40), juce::Justification::centred);
            g.setFont (juce::FontOptions (16.0f));
            g.drawText ("Version " + juce::String(ProjectInfo::versionString), area.removeFromTop (30), juce::Justification::centred);
            g.setColour (juce::Colours::lightgrey);
            g.setFont (juce::FontOptions (14.0f));
            g.drawText (juce::String(ProjectInfo::companyName), area.removeFromTop (20), juce::Justification::centred);
            g.drawText ("Made with JUCE", area.removeFromBottom (20), juce::Justification::centredBottom);
        }
    };
}

//==============================================================================
MainWindow::MainWindow (const juce::String& name)
    : DocumentWindow (name,
                      juce::Desktop::getInstance().getDefaultLookAndFeel()
                          .findColour (juce::ResizableWindow::backgroundColourId),
                      DocumentWindow::allButtons)
{
    setUsingNativeTitleBar (true);
    setResizable (true, true);
    setResizeLimits (720, 760, 10000, 10000);
    setSize (960, 980);

    // ── ASIO Blacklist 初期化 ──
    auto exeDir = juce::File::getSpecialLocation (juce::File::currentExecutableFile).getParentDirectory();
    auto blacklistFile = exeDir.getChildFile ("asio_blacklist.txt");

    // デフォルトのブラックリストファイルを作成（存在しない場合）
    // シングルクライアントASIOや不安定なドライバをデフォルトで除外
    if (! blacklistFile.existsAsFile())
    {
        blacklistFile.replaceWithText ("# ASIO Driver Blacklist\n"
                                       "# Add partial driver names to exclude them from the list.\n"
                                       "BRAVO-HD\n"
                                       "ASIO4ALL\n");
    }

    asioBlacklist.loadFromFile (blacklistFile);
    DeviceSettings::applyAsioBlacklist (audioDeviceManager, asioBlacklist);

    // エンジンを先に初期化してデフォルトのサンプルレート(48kHz)を設定
    audioEngine.initialize();

    // 設定読み込み（ブラックリスト適用後に実行することで、除外されたデバイスの自動ロードを防ぐ）
    // この時点でrebuildが呼ばれても、有効なサンプルレートが設定されている
    loadSettings();

    audioEngineProcessor = std::make_unique<AudioEngineProcessor>(audioEngine);
    audioProcessorPlayer.setDoublePrecisionProcessing(true);
    audioProcessorPlayer.setProcessor(audioEngineProcessor.get());
    audioEngine.addChangeListener (this);
    audioDeviceManager.addAudioCallback (&audioProcessorPlayer);

    // UIコンポーネントの作成
    createUIComponents();

    startTimer (500); // CPU使用率の更新頻度を上げる (500ms)
    setVisible (true);
}

//--------------------------------------------------------------
// デストラクタ
//--------------------------------------------------------------
MainWindow::~MainWindow()
{
    orderModeBox.setLookAndFeel (nullptr);

    // 【パッチ4】audioEngine の ChangeListener を最初に解除する
    // 理由: audioEngine はメンバ変数であり、このデストラクタ本体が完了した後に
    //       メンバの逆順破棄が始まる。もし audioEngine が本体完了後~audioEngine()
    //       呼び出し前に sendChangeMessage() を発火した場合、すでに破棄済みの
    //       UIコンポーネント (specAnalyzer / eqPanel 等) にアクセスする
    //       Use-After-Free が発生する。最初に removeChangeListener することで
    //       このウィンドウへの通知を即座に遮断し、安全にシャットダウンできる。
    audioEngine.removeChangeListener (this);

    audioProcessorPlayer.setProcessor (nullptr);
    stopTimer();

    DeviceSettings::saveSettings (audioDeviceManager, audioEngine);

    // 破棄される前にコールバックとしてAudioEngineの登録を解除
    audioDeviceManager.removeAudioCallback (&audioProcessorPlayer);

    // アプリ終了時にASIOドライバを確実に閉じるための安全手順
    audioDeviceManager.closeAudioDevice();
    audioEngineProcessor.reset();

    settingsWindow.reset();
    deviceSettings.reset();
    specAnalyzer.reset();
    eqPanel.reset();
    convolverPanel.reset();
}

//--------------------------------------------------------------
// 閉じるボタン押下時
//--------------------------------------------------------------
void MainWindow::closeButtonPressed()
{
    juce::JUCEApplication::getInstance()->systemRequestedQuit();
}

//--------------------------------------------------------------
// 変更通知コールバック
//--------------------------------------------------------------
void MainWindow::changeListenerCallback (juce::ChangeBroadcaster* source)
{
    if (source == &audioEngine)
    {
        if (eqPanel != nullptr)
            eqPanel->updateAllControls();
        if (convolverPanel != nullptr)
            convolverPanel->updateIRInfo();

        // メインウィンドウ上のコントロールを更新 (プリセットロード時など)
        const bool eqBypassed = audioEngine.getEQProcessor().isBypassed();
        const bool convBypassed = audioEngine.getConvolverProcessor().isBypassed();
        int modeId = 3; // Conv->Peq
        if (!eqBypassed && convBypassed)
            modeId = 2; // Peq
        else if (eqBypassed && !convBypassed)
            modeId = 1; // Conv
        else if (!eqBypassed && !convBypassed
              && audioEngine.getProcessingOrder() == AudioEngine::ProcessingOrder::EQThenConvolver)
            modeId = 4; // Peq->Conv
        orderModeBox.setSelectedId(modeId, juce::dontSendNotification);

        // ソフトクリップとサチュレーション
        softClipButton.setToggleState(audioEngine.isSoftClipEnabled(), juce::dontSendNotification);
        saturationValueLabel.setText(formatSaturationValue(audioEngine.getSaturationAmount()),
                                     juce::dontSendNotification);
    }
}

void MainWindow::labelTextChanged(juce::Label* label)
{
    if (label != &saturationValueLabel)
        return;

    float value = label->getText().retainCharacters("0123456789.").getFloatValue();
    value = juce::jlimit(0.0f, 1.0f, value);
    audioEngine.setSaturationAmount(value);
    saturationValueLabel.setText(formatSaturationValue(audioEngine.getSaturationAmount()),
                                 juce::dontSendNotification);
}

void MainWindow::editorShown(juce::Label* label, juce::TextEditor& editor)
{
    if (label != &saturationValueLabel)
        return;

    editor.setInputRestrictions(5, "0123456789.");
    editor.setText(saturationValueLabel.getText(), false);
}

//--------------------------------------------------------------
// UIコンポーネント作成
//--------------------------------------------------------------
void MainWindow::createUIComponents()
{
    convolverPanel = std::make_unique<ConvolverControlPanel> (audioEngine);
    eqPanel        = std::make_unique<EQControlPanel> (audioEngine);
    specAnalyzer   = std::make_unique<SpectrumAnalyzerComponent> (audioEngine);

    addAndMakeVisible (convolverPanel.get());
    addAndMakeVisible (eqPanel.get());
    addAndMakeVisible (specAnalyzer.get());

    deviceSettings = std::make_unique<DeviceSettings> (audioDeviceManager, audioEngine);

    showDeviceSelectorButton.setButtonText ("Audio Settings");
    showDeviceSelectorButton.setColour (juce::TextButton::buttonColourId,
                                      juce::Colours::darkslategrey.withAlpha (0.8f));
    showDeviceSelectorButton.setColour (juce::TextButton::textColourOffId,
                                      juce::Colours::white);
    showDeviceSelectorButton.onClick = [this] { toggleDeviceSelector(); };
    juce::Component::addAndMakeVisible (showDeviceSelectorButton);

    // 処理モード選択
    orderModeBox.addItem("Conv", 1);
    orderModeBox.addItem("Peq", 2);
    orderModeBox.addItem("Conv->Peq", 3);
    orderModeBox.addItem("Peq->Conv", 4);
    orderModeBox.setJustificationType(juce::Justification::centred);
    orderModeBox.setTooltip("Processing mode");
    orderModeBox.setLookAndFeel (&orderModeLookAndFeel);
    orderModeBox.onChange = [this] { orderModeBoxChanged(); };
    juce::Component::addAndMakeVisible(orderModeBox);

    // 保存/読み込みボタン
    saveButton.setButtonText ("Save");
    saveButton.onClick = [this] { savePreset(); };
    juce::Component::addAndMakeVisible (saveButton);

    loadButton.setButtonText ("Load");
    loadButton.onClick = [this] { loadPreset(); };
    juce::Component::addAndMakeVisible (loadButton);

    // CPU使用率ラベル
    cpuUsageLabel.setText ("CPU: --%", juce::dontSendNotification);
    cpuUsageLabel.setJustificationType (juce::Justification::centredRight);
    cpuUsageLabel.setColour (juce::Label::textColourId, juce::Colours::white);
    juce::Component::addAndMakeVisible (cpuUsageLabel);

    latencyLabel.setText ("Lat: -- ms", juce::dontSendNotification);
    latencyLabel.setJustificationType (juce::Justification::centredRight);
    latencyLabel.setColour (juce::Label::textColourId, juce::Colours::white);
    juce::Component::addAndMakeVisible (latencyLabel);

    // Aboutボタン
    aboutButton.setButtonText ("?");
    aboutButton.setTooltip ("About this application");
    aboutButton.onClick = [this] { showAboutDialog(); };
    juce::Component::addAndMakeVisible (aboutButton);

    // ソフトクリップボタン
    softClipButton.setButtonText("Soft Clip");
    softClipButton.setToggleState(audioEngine.isSoftClipEnabled(), juce::dontSendNotification);
    softClipButton.setTooltip("Enable/Disable Output Soft Clipper");
    softClipButton.onClick = [this] {
        audioEngine.setSoftClipEnabled(softClipButton.getToggleState());
    };
    juce::Component::addAndMakeVisible(softClipButton);

    // サチュレーションスライダー
    saturationValueLabel.setText(formatSaturationValue(audioEngine.getSaturationAmount()), juce::dontSendNotification);
    saturationValueLabel.setEditable(true);
    saturationValueLabel.setJustificationType(juce::Justification::centred);
    saturationValueLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    saturationValueLabel.setColour(juce::Label::outlineColourId, juce::Colours::grey);
    saturationValueLabel.setTooltip("Saturation Amount (0.0 - 1.0)");
    saturationValueLabel.addListener(this);
    juce::Component::addAndMakeVisible(saturationValueLabel);

    saturationLabel.setText("Sat:", juce::dontSendNotification);
    saturationLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    saturationLabel.setJustificationType(juce::Justification::centredRight);
    juce::Component::addAndMakeVisible(saturationLabel);

    // 初期選択をエンジン状態に同期
    changeListenerCallback(&audioEngine);
}

//--------------------------------------------------------------
// 処理モードドロップダウン
//--------------------------------------------------------------
void MainWindow::orderModeBoxChanged()
{
    const int mode = orderModeBox.getSelectedId();
    if (mode == 1)
    {
        audioEngine.setConvolverBypassRequested(false);
        audioEngine.setEqBypassRequested(true);
    }
    else if (mode == 2)
    {
        audioEngine.setConvolverBypassRequested(true);
        audioEngine.setEqBypassRequested(false);
    }
    else if (mode == 3)
    {
        audioEngine.setConvolverBypassRequested(false);
        audioEngine.setEqBypassRequested(false);
        audioEngine.setProcessingOrder(AudioEngine::ProcessingOrder::ConvolverThenEQ);
    }
    else if (mode == 4)
    {
        audioEngine.setConvolverBypassRequested(false);
        audioEngine.setEqBypassRequested(false);
        audioEngine.setProcessingOrder(AudioEngine::ProcessingOrder::EQThenConvolver);
    }

    if (eqPanel != nullptr)
        eqPanel->updateAllControls();
    if (convolverPanel != nullptr)
        convolverPanel->updateIRInfo();
}

//--------------------------------------------------------------
// 設定読み込み
//--------------------------------------------------------------
void MainWindow::loadSettings()
{
    DeviceSettings::loadSettings (audioDeviceManager, audioEngine);
}

//--------------------------------------------------------------
// デバイス設定画面の表示切り替え
//--------------------------------------------------------------
void MainWindow::toggleDeviceSelector()
{
    if (settingsWindow == nullptr)
    {
        auto background = juce::Desktop::getInstance().getDefaultLookAndFeel()
                              .findColour (juce::ResizableWindow::backgroundColourId);

        auto newSettingsWindow = std::make_unique<SettingsWindow> ("Audio Settings", background, DocumentWindow::allButtons);
        newSettingsWindow->setResizable (true, false);
        newSettingsWindow->setResizeLimits (400, 400, 800, 1000); // 最大高さを拡張
        newSettingsWindow->setContentNonOwned (deviceSettings.get(), false);
        newSettingsWindow->centreWithSize (500, 624);

        newSettingsWindow->onClose = [this]
        {
            showDeviceSelectorButton.setButtonText ("Audio Settings");
        };

        settingsWindow = std::move (newSettingsWindow);
    }

    if (settingsWindow->isVisible())
    {
        settingsWindow->userTriedToCloseWindow();
    }
    else
    {
        settingsWindow->setVisible (true);
        settingsWindow->toFront (true);
        showDeviceSelectorButton.setButtonText ("Hide Settings");
    }
}

//--------------------------------------------------------------
// リサイズ
//--------------------------------------------------------------
void MainWindow::resized()
{
    auto bounds = getLocalBounds();

    auto buttonRow = bounds.removeFromTop (28);

    // 右側: About / Audio Settings
    aboutButton.setBounds (buttonRow.removeFromRight (30).reduced (2, 2));
    showDeviceSelectorButton.setBounds (buttonRow.removeFromRight (130).reduced (2, 2));

    // 状態表示
    cpuUsageLabel.setBounds (buttonRow.removeFromRight (95).reduced (2, 2));
    latencyLabel.setBounds (buttonRow.removeFromRight (170).reduced (2, 2));

    // クリップ制御 (左→右: Soft Clip, Sat, 数値入力)
    saturationValueLabel.setBounds(buttonRow.removeFromRight(58).reduced(2, 2));
    saturationLabel.setBounds(buttonRow.removeFromRight(42).reduced(2, 2));
    softClipButton.setBounds(buttonRow.removeFromRight(90).reduced(2, 2));

    // 左側: 保存/読込 + 処理モード
    orderModeBox.setBounds (buttonRow.removeFromRight (145).reduced (2, 2));
    loadButton.setBounds (buttonRow.removeFromRight (46).reduced (2, 2));
    saveButton.setBounds (buttonRow.removeFromRight (46).reduced (2, 2));

    if (convolverPanel)
        convolverPanel->setBounds (bounds.removeFromTop (280));

    const int eqH = static_cast<int> (bounds.getHeight() * 0.48f);
    if (eqPanel)
        eqPanel->setBounds (bounds.removeFromTop (eqH));

    if (specAnalyzer)
        specAnalyzer->setBounds (bounds);
}

//--------------------------------------------------------------
// タイマーコールバック
//--------------------------------------------------------------
void MainWindow::timerCallback()
{
    double cpu = audioDeviceManager.getCpuUsage() * 100.0;
    cpuUsageLabel.setText ("CPU: " + juce::String (cpu, 1) + "%", juce::dontSendNotification);

    const auto breakdown = audioEngine.getCurrentLatencyBreakdown();
    const int latencySamples = breakdown.totalLatencyBaseRateSamples;
    const double sr = audioEngine.getSampleRate();
    const int latencyMs = (sr > 0.0)
        ? juce::roundToInt((static_cast<double>(latencySamples) * 1000.0) / sr)
        : 0;
    latencyLabel.setText ("Lat: " + juce::String (latencyMs) + "ms (" + juce::String(latencySamples) + " smp)",
                          juce::dontSendNotification);

#if JUCE_DEBUG
    {
        static double lastLatencyLogMs = 0.0;
        const double nowMs = juce::Time::getMillisecondCounterHiRes();
        if (nowMs - lastLatencyLogMs >= 1000.0)
        {
            lastLatencyLogMs = nowMs;

            const int osFactor = juce::jmax(1, audioEngine.getOversamplingFactor());
            const auto toMsRounded3 = [sr](int samples) -> double
            {
                if (sr <= 0.0)
                    return 0.0;
                const double ms = (static_cast<double>(samples) * 1000.0) / sr;
                return std::round(ms * 1000.0) / 1000.0;
            };

            DBG("lat sr=" << juce::String(sr, 1)
                << " osFactor=" << osFactor
                << " os=" << juce::String(toMsRounded3(breakdown.oversamplingLatencyBaseRateSamples), 3)
                << " convA=" << juce::String(toMsRounded3(breakdown.convolverAlgorithmLatencyBaseRateSamples), 3)
                << " convPeak=" << juce::String(toMsRounded3(breakdown.convolverIRPeakLatencyBaseRateSamples), 3)
                << " total=" << latencyMs);
        }
    }
#endif
}

//--------------------------------------------------------------
// プリセット保存
//--------------------------------------------------------------
void MainWindow::savePreset()
{
    launchFileChooser(true);
}

//--------------------------------------------------------------
// プリセット読み込み
//--------------------------------------------------------------
void MainWindow::loadPreset()
{
    launchFileChooser(false);
}

//--------------------------------------------------------------
// ファイル選択ダイアログ
//--------------------------------------------------------------
void MainWindow::launchFileChooser(bool isSaving)
{
    const juce::String title = isSaving ? "Save Preset" : "Load Preset";
    const juce::String wildcards = isSaving ? "*.xml" : "*.xml;*.txt";
    const int chooserFlags = isSaving ? (juce::FileBrowserComponent::saveMode | juce::FileBrowserComponent::canSelectFiles)
                                      : (juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles);

    auto fileChooser = std::make_shared<juce::FileChooser>(title,
                                                           juce::File::getSpecialLocation(juce::File::userDocumentsDirectory),
                                                           wildcards);

    // 安全性と整合性のためにSafePointerを使用
    juce::Component::SafePointer<MainWindow> safeThis(this);

    fileChooser->launchAsync(chooserFlags, [safeThis, isSaving, fileChooser](const juce::FileChooser& fc)
    {
        if (safeThis == nullptr)
            return;

        auto file = fc.getResult();
        if (file == juce::File())
            return;

        if (isSaving)
        {
            auto state = safeThis->audioEngine.getCurrentState();
            if (auto xml = state.createXml())
            {
                xml->writeTo(file);
            }
        }
        else // 読み込み中
        {
            if (file.existsAsFile())
            {
                if (file.hasFileExtension(".xml"))
                {
                    if (auto xml = juce::XmlDocument::parse(file))
                    {
                        auto state = juce::ValueTree::fromXml(*xml);
                        if (state.isValid())
                            safeThis->audioEngine.requestLoadState(state);
                    }
                }
                else if (file.hasFileExtension(".txt"))
                {
                    safeThis->audioEngine.requestEqPresetFromText(file);
                }
            }
        }
    });
}

//--------------------------------------------------------------
// バージョン情報ダイアログ
//--------------------------------------------------------------
void MainWindow::showAboutDialog()
{
    juce::DialogWindow::LaunchOptions options;
    options.content.setOwned (new AboutComponent());
    options.dialogTitle = "About " + juce::String(ProjectInfo::projectName);
    options.dialogBackgroundColour = juce::Colours::darkgrey;
    options.escapeKeyTriggersCloseButton = true;
    options.useNativeTitleBar = true;
    options.resizable = false;
    options.launchAsync();
}
