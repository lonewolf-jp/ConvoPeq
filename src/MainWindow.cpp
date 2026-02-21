//============================================================================
// MainWindow.cpp ── v0.2 (JUCE 8.0.12対応)
//
// メインウィンドウの実装
// UIコンポーネントの配置とオーディオデバイス管理を行う
//============================================================================
#include "MainWindow.h"

namespace
{
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
    setResizeLimits (720, 700, 10000, 10000);
    setSize (960, 920);

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

    audioSourcePlayer.setSource (&audioEngine);
    audioEngine.addChangeListener (this);
    audioDeviceManager.addAudioCallback (&audioSourcePlayer);

    // Create UI Components
    createUIComponents();

    startTimer (500); // CPU使用率の更新頻度を上げる (500ms)
    setVisible (true);
}

//--------------------------------------------------------------
// デストラクタ
//--------------------------------------------------------------
MainWindow::~MainWindow()
{
    stopTimer();

    DeviceSettings::saveSettings (audioDeviceManager, audioEngine);

    audioEngine.removeChangeListener (this);
    audioDeviceManager.removeAudioCallback (&audioSourcePlayer);
    audioSourcePlayer.setSource (nullptr);

    // アプリ終了時にASIOドライバを確実に閉じるための安全手順
    audioDeviceManager.closeAudioDevice();

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
        // Processing Order
        if (audioEngine.getProcessingOrder() == AudioEngine::ProcessingOrder::ConvolverThenEQ)
            orderButton.setButtonText("Order: Conv -> EQ");
        else
            orderButton.setButtonText("Order: EQ -> Conv");

        // Bypass Buttons
        eqBypassButton.setToggleState(!audioEngine.getEQProcessor().isBypassed(), juce::dontSendNotification);
        eqBypassButton.setButtonText(audioEngine.getEQProcessor().isBypassed() ? "EQ Off" : "EQ On");

        convolverBypassButton.setToggleState(!audioEngine.getConvolverProcessor().isBypassed(), juce::dontSendNotification);
        convolverBypassButton.setButtonText(audioEngine.getConvolverProcessor().isBypassed() ? "Conv Off" : "Conv On");

        // Soft Clip & Saturation
        softClipButton.setToggleState(audioEngine.isSoftClipEnabled(), juce::dontSendNotification);
        saturationSlider.setValue(audioEngine.getSaturationAmount(), juce::dontSendNotification);
    }
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
    addAndMakeVisible (showDeviceSelectorButton);

    // EQ On/Off Button
    eqBypassButton.setButtonText ("EQ On");
    eqBypassButton.setToggleState (!audioEngine.getEQProcessor().isBypassed(), juce::dontSendNotification);
    eqBypassButton.onClick = [this] { eqBypassButtonClicked(); };
    addAndMakeVisible (eqBypassButton);

    // Convolver On/Off Button
    convolverBypassButton.setButtonText ("Conv On");
    convolverBypassButton.setToggleState (!audioEngine.getConvolverProcessor().isBypassed(), juce::dontSendNotification);
    convolverBypassButton.onClick = [this] { convolverBypassButtonClicked(); };
    addAndMakeVisible (convolverBypassButton);

    // Processing Order Button
    orderButton.setButtonText ("Order: Conv -> EQ");
    orderButton.onClick = [this] { orderButtonClicked(); };
    addAndMakeVisible (orderButton);

    // Save/Load Buttons
    saveButton.setButtonText ("Save");
    saveButton.onClick = [this] { savePreset(); };
    addAndMakeVisible (saveButton);

    loadButton.setButtonText ("Load");
    loadButton.onClick = [this] { loadPreset(); };
    addAndMakeVisible (loadButton);

    // CPU Usage Label
    cpuUsageLabel.setText ("CPU: --%", juce::dontSendNotification);
    cpuUsageLabel.setJustificationType (juce::Justification::centredRight);
    cpuUsageLabel.setColour (juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible (cpuUsageLabel);

    // About Button
    aboutButton.setButtonText ("?");
    aboutButton.setTooltip ("About this application");
    aboutButton.onClick = [this] { showAboutDialog(); };
    addAndMakeVisible (aboutButton);

    // Soft Clip Button
    softClipButton.setButtonText("Soft Clip");
    softClipButton.setToggleState(audioEngine.isSoftClipEnabled(), juce::dontSendNotification);
    softClipButton.setTooltip("Enable/Disable Output Soft Clipper");
    softClipButton.onClick = [this] {
        audioEngine.setSoftClipEnabled(softClipButton.getToggleState());
    };
    addAndMakeVisible(softClipButton);

    // Saturation Slider
    saturationSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    saturationSlider.setRange(0.0, 1.0, 0.01);
    saturationSlider.setValue(audioEngine.getSaturationAmount(), juce::dontSendNotification);
    saturationSlider.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
    saturationSlider.setTooltip("Saturation Amount (Threshold & Knee)");
    saturationSlider.onValueChange = [this] {
        audioEngine.setSaturationAmount(static_cast<float>(saturationSlider.getValue()));
    };
    addAndMakeVisible(saturationSlider);

    saturationLabel.setText("Sat:", juce::dontSendNotification);
    saturationLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    saturationLabel.setJustificationType(juce::Justification::centredRight);
    addAndMakeVisible(saturationLabel);
}

//--------------------------------------------------------------
// EQバイパスボタン
//--------------------------------------------------------------
void MainWindow::eqBypassButtonClicked()
{
    const bool isBypassed = !eqBypassButton.getToggleState();
    audioEngine.setEqBypassRequested(isBypassed);
    eqBypassButton.setButtonText(isBypassed ? "EQ Off" : "EQ On");
    // Also update the UI processor state for consistency
    audioEngine.getEQProcessor().setBypass(isBypassed);
}

//--------------------------------------------------------------
// Convolverバイパスボタン
//--------------------------------------------------------------
void MainWindow::convolverBypassButtonClicked()
{
    const bool isBypassed = !convolverBypassButton.getToggleState();
    audioEngine.setConvolverBypassRequested(isBypassed);
    convolverBypassButton.setButtonText(isBypassed ? "Conv Off" : "Conv On");
    // Also update the UI processor state for consistency
    audioEngine.getConvolverProcessor().setBypass(isBypassed);
}

//--------------------------------------------------------------
// 処理順序ボタン
//--------------------------------------------------------------
void MainWindow::orderButtonClicked()
{
    if (audioEngine.getProcessingOrder() == AudioEngine::ProcessingOrder::ConvolverThenEQ)
    {
        audioEngine.setProcessingOrder(AudioEngine::ProcessingOrder::EQThenConvolver);
        orderButton.setButtonText("Order: EQ -> Conv");
    }
    else
    {
        audioEngine.setProcessingOrder(AudioEngine::ProcessingOrder::ConvolverThenEQ);
        orderButton.setButtonText("Order: Conv -> EQ");
    }
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
    aboutButton.setBounds (buttonRow.removeFromRight (30).reduced (2, 2));
    showDeviceSelectorButton.setBounds (buttonRow.removeFromRight (140).reduced (2, 2));
    orderButton.setBounds (buttonRow.removeFromRight (140).reduced (2, 2));
    loadButton.setBounds (buttonRow.removeFromRight (50).reduced (2, 2));
    saveButton.setBounds (buttonRow.removeFromRight (50).reduced (2, 2));
    convolverBypassButton.setBounds (buttonRow.removeFromRight (80).reduced (2, 2));
    eqBypassButton.setBounds (buttonRow.removeFromRight (80).reduced (2, 2));

    saturationSlider.setBounds(buttonRow.removeFromRight(80).reduced(2, 2));
    saturationLabel.setBounds(buttonRow.removeFromRight(30).reduced(2, 2));
    softClipButton.setBounds(buttonRow.removeFromRight(70).reduced(2, 2));
    cpuUsageLabel.setBounds (buttonRow.removeFromRight (70).reduced (2, 2));

    if (convolverPanel)
        convolverPanel->setBounds (bounds.removeFromTop (220));

    const int eqH = static_cast<int> (bounds.getHeight() * 0.45f);
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

    // Use SafePointer for safety consistency
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
        else // isLoading
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
