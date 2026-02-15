//============================================================================
#pragma once
// DeviceSettings.h  ── デバイス設定の永続化 (JUCE 8.0.12対応)
//
// オーディオデバイス設定をXMLファイルに保存・読み込みする機能を提供。
// アプリケーション終了時に設定を保存し、次回起動時に復元する。
//
// ■ 保存される設定:
//   - デバイスタイプ（WASAPI/ASIO/DirectSound等）
//   - 入力デバイス名
//   - 出力デバイス名
//   - サンプルレート
//   - バッファサイズ
//   - 有効な入力チャンネル (Bitmask)
//   - 有効な出力チャンネル (Bitmask)
//
// ■ 保存先:
//   - Windows: %APPDATA%/ConvoPeq/device_settings.xml
//   - (Note: This application is Windows-only)
//
// ■ 安全性:
//   - 設定ファイルが壊れている場合はデフォルト設定を使用
//   - 保存されたデバイスが利用不可の場合は他のデバイスにフォールバック
//============================================================================

#include <JuceHeader.h>
#include "AsioBlacklist.h"

class DeviceSettings : public juce::Component
{
public:
    explicit DeviceSettings (juce::AudioDeviceManager& adm);
    ~DeviceSettings() override = default;

    void resized() override;

    static void saveSettings (const juce::AudioDeviceManager& deviceManager);
    static void loadSettings (juce::AudioDeviceManager& deviceManager);

    static void applyAsioBlacklist (juce::AudioDeviceManager& deviceManager, const AsioBlacklist& blacklist);

private:
    juce::AudioDeviceManager& audioDeviceManager;
    std::unique_ptr<juce::AudioDeviceSelectorComponent> selector;

    static juce::File getSettingsFile();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DeviceSettings)
};
