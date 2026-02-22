//============================================================================
#pragma once
// DeviceSettings.h  ── v0.2 (JUCE 8.0.12対応)
//
// オーディオデバイス設定をXMLファイルに保存・読み込みする機能を提供。
// アプリケーション終了時に設定を保存し、次回起動時に復元する。
//
// ■ 保存される設定:
//   - デバイスタイプ (WASAPI/ASIO/DirectSound等)
//   - 入力デバイス名
//   - 出力デバイス名
//   - サンプルレート
//   - バッファサイズ
//   - 有効な入力チャンネル (Bitmask)
//   - 有効な出力チャンネル (Bitmask)
//
// ■ 保存先:
//   - Windows: %APPDATA%/ConvoPeq/device_settings.xml (ローミングプロファイル)
//   - (注意: 本アプリケーションはWindows専用です)
//
// ■ 安全性:
//   - 設定ファイルが壊れている場合はデフォルト設定を使用
//   - 保存されたデバイスが利用不可の場合は他のデバイスにフォールバック
//============================================================================

#include <JuceHeader.h>
#include "AsioBlacklist.h"
#include "AudioEngine.h"

class DeviceSettings : public juce::Component,
                       private juce::ChangeListener
{
public:
    DeviceSettings (juce::AudioDeviceManager& adm, AudioEngine& engine);
    ~DeviceSettings() override;

    void resized() override;

    static void saveSettings (const juce::AudioDeviceManager& deviceManager, const AudioEngine& engine);
    static void loadSettings (juce::AudioDeviceManager& deviceManager, AudioEngine& engine);

    static void applyAsioBlacklist (juce::AudioDeviceManager& deviceManager, const AsioBlacklist& blacklist);

private:
    void changeListenerCallback (juce::ChangeBroadcaster* source) override;
    void updateBitDepthList();

    juce::AudioDeviceManager& audioDeviceManager;
    AudioEngine& audioEngine;
    std::unique_ptr<juce::AudioDeviceSelectorComponent> selector;


    juce::ComboBox oversamplingComboBox;
    juce::Label oversamplingLabel;
    juce::TabbedComponent filterTypeTabs;

    juce::ComboBox bitDepthComboBox;
    juce::Label bitDepthLabel;

    static juce::File getSettingsFile();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DeviceSettings)
};
