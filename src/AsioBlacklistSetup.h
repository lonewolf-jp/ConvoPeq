//============================================================================
#pragma once
// AsioBlacklistSetup.h ── ASIO ブラックリストファイルの用意と、失敗時の通知判断
//
//   MainWindow のコンストラクタに埋まっていた処理を、テストできる形に切り出した。
//   実行ファイル隣接を優先し、書けなければユーザー領域へフォールバックする。
//   どちらも用意できない、または読めない場合は、除外が効かないまま起動している
//   ことを利用者へ伝える（自動実行では伝えない）。
//============================================================================

#include <JuceHeader.h>

namespace convo::asio
{

// 既定のブラックリスト。シングルクライアント ASIO と不安定なドライバを除外する。
inline constexpr const char* kDefaultAsioBlacklistText =
    "# ASIO Driver Blacklist\n"
    "# Add partial driver names to exclude them from the list.\n"
    "BRAVO-HD\n"
    "ASIO4ALL\n";

enum class BlacklistFileStatus
{
    readyInExecutableDirectory, // 実行ファイル隣接のファイルを使えた
    readyInUserDirectory,       // 書込不可のためユーザー領域へ退避した
    unavailable                 // どちらにも用意できなかった
};

struct BlacklistFileResolution
{
    juce::File file;
    BlacklistFileStatus status = BlacklistFileStatus::unavailable;
};

// 既存ファイルはそのまま使う。無ければ既定の内容で作る。
//   親ディレクトリは作らない（呼出側の責務）。存在しないディレクトリの下や、
//   ファイルを親に指定したパスでは false になる。
inline bool ensureBlacklistFile (const juce::File& file)
{
    if (file.existsAsFile())
        return true;

    return file.replaceWithText (kDefaultAsioBlacklistText) && file.existsAsFile();
}

// 実行ファイル隣接を優先し、失敗したらユーザー領域へフォールバックする。
//   返る file は、失敗時には診断用に実行ファイル隣接のパスを指す。
inline BlacklistFileResolution resolveBlacklistFile (const juce::File& executableDirectory,
                                                     const juce::File& userApplicationDirectory)
{
    BlacklistFileResolution resolution;
    resolution.file = executableDirectory.getChildFile ("asio_blacklist.txt");

    if (ensureBlacklistFile (resolution.file))
    {
        resolution.status = BlacklistFileStatus::readyInExecutableDirectory;
        return resolution;
    }

    auto userFile = userApplicationDirectory
                        .getChildFile ("ConvoPeq")
                        .getChildFile ("asio_blacklist.txt");
    userFile.getParentDirectory().createDirectory();

    if (ensureBlacklistFile (userFile))
    {
        resolution.file = userFile;
        resolution.status = BlacklistFileStatus::readyInUserDirectory;
        return resolution;
    }

    resolution.status = BlacklistFileStatus::unavailable;
    return resolution;
}

// コマンドラインが自動実行か。GUI 起動では false。
//   --cli-* は自動実行、--verify-runtime-evidence 系は証跡モードである。
inline bool isAutomationInvocation (const juce::String& commandLine) noexcept
{
    return commandLine.contains ("--cli-")
        || commandLine.contains ("--verify-runtime-evidence");
}

// 読み込みに失敗したときに利用者へ知らせるか。
//   自動実行ではダイアログを出さない（CI とテストを止めるため）。
inline bool shouldShowBlacklistWarning (bool loadSucceeded,
                                        bool automationInvocation) noexcept
{
    return ! loadSucceeded && ! automationInvocation;
}

} // namespace convo::asio
