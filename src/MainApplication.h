//============================================================================
// MainApplication.h  ── v0.5.3 (JUCE 8.0.12対応)
//
// アプリケーションエントリポイント宣言
// 責任: JUCE アプリライフサイクル（起動・終了）・MainWindow の所有
// スレッド: メインスレッド（UI）のみ
//============================================================================
#pragma once

#include <JuceHeader.h>

#include "SizeCappedFileLogger.h"

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
    void initialise(const juce::String& commandLine) override;

    // ─── 終了処理 ───
    // mainWindow.reset() で MainWindow デストラクタが呼ばれ、
    // オーディオコールバックも安全に停止される。
    void shutdown() override;

    // ★ review: --cli-log-file のロガーは MainWindow より長生きさせる。
    //   ~AudioEngine と shutdown の最終ログを CLI ファイルへ出すため、
    //   所有権を MainApplication へ移す。
    void adoptCliLogger (std::unique_ptr<SizeCappedFileLogger> logger);

private:
    std::unique_ptr<MainWindow> mainWindow;
    std::unique_ptr<SizeCappedFileLogger> fileLogger;
    std::unique_ptr<SizeCappedFileLogger> cliLogger;
    juce::Logger* loggerBeforeCli = nullptr;

    // ★ Live evidence: `--verify-runtime-evidence-live`（実 Runtime 実行に基づく証拠）
    //   既存の `--verify-runtime-evidence`（Exporter 単体呼び出し）とは独立したモード。
    bool liveRuntimeEvidenceRequested = false;
    std::uint64_t liveStartedAtNs = 0;
    bool emitLiveRuntimeEvidence();
};
