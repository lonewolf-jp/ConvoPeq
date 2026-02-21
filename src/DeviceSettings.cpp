//============================================================================
// DeviceSettings.cpp  ── v0.2 (JUCE 8.0.12対応)
//============================================================================
#include "DeviceSettings.h"

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
// `const_cast`を用いて内部の読み取り専用配列を書き換えるという危険な操作を行っているため、
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
    oversamplingComboBox.addItem("1x (None)", 2);
    oversamplingComboBox.addItem("2x", 3);
    oversamplingComboBox.addItem("4x", 4);
    oversamplingComboBox.addItem("8x", 5);
    oversamplingComboBox.onChange = [this] {
        int id = oversamplingComboBox.getSelectedId();
        int factor = 0; // Auto
        if (id == 2) factor = 1;
        else if (id == 3) factor = 2;
        else if (id == 4) factor = 4;
        else if (id == 5) factor = 8;
        audioEngine.setOversamplingFactor(factor);
    };

    // Dither Bit Depth Controls
    addAndMakeVisible(bitDepthLabel);
    bitDepthLabel.setText("Dither Bit Depth:", juce::dontSendNotification);
    bitDepthLabel.setJustificationType(juce::Justification::centredLeft);

    addAndMakeVisible(bitDepthComboBox);
    bitDepthComboBox.onChange = [this] {
        int id = bitDepthComboBox.getSelectedId();
        if (id > 0)
        {
            // IDにはビット深度そのものを使用する (例: 16, 24, 32)
            audioEngine.setDitherBitDepth(id);
        }
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

    updateBitDepthList();
}

DeviceSettings::~DeviceSettings()
{
    audioDeviceManager.removeChangeListener(this);
    filterTypeTabs.getTabbedButtonBar().removeChangeListener(this);
}

void DeviceSettings::resized()
{
    auto bounds = getLocalBounds();
    auto controlsArea = bounds.removeFromTop(90);
    auto row1 = controlsArea.removeFromTop(30);
    auto row2 = controlsArea.removeFromTop(30);
    auto row3 = controlsArea.removeFromTop(30);

    bitDepthLabel.setBounds(row1.removeFromLeft(100).reduced(5));
    bitDepthComboBox.setBounds(row1.removeFromLeft(100).reduced(2));

    filterTypeTabs.setBounds(row2.reduced(2));

    oversamplingLabel.setBounds(row3.removeFromLeft(100).reduced(5));
    oversamplingComboBox.setBounds(row3.removeFromLeft(100).reduced(2));

    if (selector != nullptr)
        selector->setBounds(bounds);
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

void DeviceSettings::updateBitDepthList()
{
    // 現在のデバイスを取得
    juce::Array<int> supportedBitDepths; // device変数は未使用のため削除
    // AudioIODeviceにはビット深度を取得する標準的なAPIがないため、一般的な値をリストアップ

    // デバイスがビット深度を報告しない場合、またはデバイスがない場合のフォールバック
    if (supportedBitDepths.isEmpty())
    {
        supportedBitDepths.add(16);
        supportedBitDepths.add(24);
        supportedBitDepths.add(32);
    }

    // UI更新
    bitDepthComboBox.clear();
    int maxBitDepth = 0;

    for (int depth : supportedBitDepths)
    {
        bitDepthComboBox.addItem(juce::String(depth) + "-bit", depth); // ID = depth
        if (depth > maxBitDepth)
            maxBitDepth = depth;
    }

    // 選択状態の決定
    // 1. 現在のエンジンの設定が有効ならそれを維持
    // 2. 未設定(0)または無効なら、最大ビット深度を選択 (デフォルト)
    int currentEngineDepth = audioEngine.getDitherBitDepth();

    if (supportedBitDepths.contains(currentEngineDepth))
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
        // オーバーサンプリング設定を追加
        xml->setAttribute("oversamplingFactor", engine.getOversamplingFactor());
        // フィルタタイプ設定を追加
        xml->setAttribute("oversamplingType", (int)engine.getOversamplingType());
        xml->writeTo (getSettingsFile());
    }
}

void DeviceSettings::loadSettings (juce::AudioDeviceManager& deviceManager, AudioEngine& engine)
{
    // ASIOフリーズ防止のため、初期化前にデバイスを閉じる
    deviceManager.closeAudioDevice();

    auto file = getSettingsFile();

    if (file.existsAsFile())
    {
        if (auto xml = juce::XmlDocument::parse (file))
        {
            // 保存された設定でデバイスを初期化する。
            // 入力は2チャンネル、出力は2チャンネルを要求する。
            deviceManager.initialise (2, 2, xml.get(), true);

            // ビット深度設定の読み込み (デフォルト0 = 自動/最大)
            int bitDepth = xml->getIntAttribute("ditherBitDepth", 0);
            engine.setDitherBitDepth(bitDepth);

            // オーバーサンプリング設定の読み込み (デフォルト0 = 自動)
            int oversampling = xml->getIntAttribute("oversamplingFactor", 0);
            engine.setOversamplingFactor(oversampling);

            // フィルタタイプ設定の読み込み (デフォルト0 = IIR)
            int type = xml->getIntAttribute("oversamplingType", 0);
            engine.setOversamplingType((AudioEngine::OversamplingType)type);
            return;
        }
    }

    // 設定ファイルが存在しない、または読み込みに失敗した場合は、
    // デフォルトのデバイスで初期化する。
    deviceManager.initialiseWithDefaultDevices (2, 2);

    // デフォルトで最大サンプルレートに設定
    auto* currentDevice = deviceManager.getCurrentAudioDevice();
    if (currentDevice != nullptr)
    {
        auto availableRates = currentDevice->getAvailableSampleRates();
        if (!availableRates.isEmpty())
        {
            double maxRate = 0.0;
            for (auto rate : availableRates)
                if (rate > maxRate)
                    maxRate = rate;

            auto setup = deviceManager.getAudioDeviceSetup();
            if (std::abs(setup.sampleRate - maxRate) > 1.0 && maxRate > 0.0)
            {
                setup.sampleRate = maxRate;
                deviceManager.setAudioDeviceSetup(setup, true);
            }
        }
    }

    engine.setDitherBitDepth(0); // 自動設定へ
    engine.setOversamplingFactor(0); // 自動設定へ
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
