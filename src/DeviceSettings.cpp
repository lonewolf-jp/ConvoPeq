//============================================================================
// DeviceSettings.cpp  ── デバイス設定の永続化実装 (JUCE 8.0.12対応)
//============================================================================
#include "DeviceSettings.h"

//==============================================================================
// ASIOドライバをラップしてブラックリストフィルタを適用するクラス
//
// JUCEのASIOデバイス管理に介入し、特定のドライバ（不安定なものや不要なもの）を
// デバイスリストから除外するためのラッパークラス。シングルクライアントASIO（BRAVO-HD, ASIO4ALL等）の問題を回避します。
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

    // Dither Bit Depth Controls
    addAndMakeVisible(bitDepthLabel);
    bitDepthLabel.setText("Dither Bit Depth:", juce::dontSendNotification);
    bitDepthLabel.setJustificationType(juce::Justification::centredLeft);

    addAndMakeVisible(bitDepthComboBox);
    bitDepthComboBox.addItem("16-bit", 1);
    bitDepthComboBox.addItem("24-bit", 2);

    // 初期値設定
    int currentDepth = audioEngine.getDitherBitDepth();
    bitDepthComboBox.setSelectedId(currentDepth == 16 ? 1 : 2, juce::dontSendNotification);

    bitDepthComboBox.onChange = [this] {
        int id = bitDepthComboBox.getSelectedId();
        if (id == 1) audioEngine.setDitherBitDepth(16);
        else if (id == 2) audioEngine.setDitherBitDepth(24);
    };
}

void DeviceSettings::resized()
{
    auto bounds = getLocalBounds();
    auto topRow = bounds.removeFromTop(30);

    bitDepthLabel.setBounds(topRow.removeFromLeft(100).reduced(5));
    bitDepthComboBox.setBounds(topRow.removeFromLeft(100).reduced(2));

    if (selector != nullptr)
        selector->setBounds (bounds);
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

            // ビット深度設定の読み込み (デフォルト24bit)
            int bitDepth = xml->getIntAttribute("ditherBitDepth", 24);
            engine.setDitherBitDepth(bitDepth);
            return;
        }
    }

    // 設定ファイルが存在しない、または読み込みに失敗した場合は、
    // デフォルトのデバイスで初期化する。
    deviceManager.initialiseWithDefaultDevices (2, 2);
    engine.setDitherBitDepth(24);
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
    juce::AudioIODeviceType* originalAsio = nullptr;

    for (auto* type : availableTypes)
    {
        if (type->getTypeName() == "ASIO")
        {
            originalAsio = type;
            break;
        }
    }

    if (originalAsio != nullptr)
    {

        // 配列から削除するが、オブジェクトは削除しない (false指定)
        availableTypes.removeObject (originalAsio, false);

        // ラッパーで包んで再登録 (所有権を委譲)
        deviceManager.addAudioDeviceType (std::make_unique<BlacklistedASIODeviceType> (std::unique_ptr<juce::AudioIODeviceType>(originalAsio), blacklist));
    }
}
