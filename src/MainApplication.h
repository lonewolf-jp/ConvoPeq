//============================================================================
// MainApplication.h  ── v0.1 (JUCE 8.0.12対応)
//
// アプリケーションエントリポイント宣言
// 責任: JUCE アプリライフサイクル（起動・終了）・MainWindow の所有
// スレッド: メインスレッド（UI）のみ
//============================================================================
#pragma once

#include <JuceHeader.h>

class MainWindow; // フォワード宣言

class MainApplication : public juce::JUCEApplication
{
public:
    const juce::String getApplicationName() override
    {
        return juce::String(ProjectInfo::projectName);
    }

    const juce::String getApplicationVersion() override
    {
        return juce::String(ProjectInfo::versionString);
    }

    // 同一アプリの複数起動を禁止
    bool moreThanOneInstanceAllowed() override { return false; }

    // ─── 起動処理 ───
    // MainWindow を生成する。AudioDeviceManager は MainWindow で管理される。
    void initialise(const juce::String& /*commandLine*/) override;

    // ─── 終了処理 ───
    // mainWindow.reset() で MainWindow デストラクタが呼ばれ、
    // オーディオコールバックも安全に停止される。
    void shutdown() override;

private:
    std::unique_ptr<MainWindow> mainWindow;
};
