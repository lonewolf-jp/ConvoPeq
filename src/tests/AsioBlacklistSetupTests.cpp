//============================================================================
// AsioBlacklistSetupTests.cpp ── ASIO ブラックリストの用意と通知判断の回帰
//
//   B-17: 実行ファイル隣接に書けない場合のユーザー領域フォールバックと、
//         それも失敗したときの「無言で起動しない」判断を固定する。
//         ダイアログ自体は GUI の都合で自動化できないため、判断関数を検証する。
//============================================================================

#include <JuceHeader.h>

#include "AsioBlacklist.h"
#include "AsioBlacklistSetup.h"

#include <cstdio>

namespace {

int gFailures = 0;

void check (bool ok, const char* id, const juce::String& detail = {})
{
    std::printf ("[%s] %s%s\n",
                 ok ? "PASS" : "FAIL",
                 id,
                 detail.isNotEmpty() ? (" (" + detail + ")").toRawUTF8() : "");

    if (! ok)
        ++gFailures;
}

juce::File makeBase (const juce::String& name)
{
    auto dir = juce::File::getSpecialLocation (juce::File::tempDirectory)
                   .getChildFile ("convo_asioblacklist_" + name);
    dir.deleteRecursively();
    dir.createDirectory();
    return dir;
}

} // namespace

int main()
{
    // A-1: 実行ファイル隣接に無ければ既定の内容で作る
    {
        const auto base = makeBase ("a1");
        const auto exeDir = base.getChildFile ("exe");
        const auto userDir = base.getChildFile ("user");
        exeDir.createDirectory();

        const auto r = convo::asio::resolveBlacklistFile (exeDir, userDir);
        check (r.status == convo::asio::BlacklistFileStatus::readyInExecutableDirectory,
               "A-1 status is executable directory", juce::String (static_cast<int> (r.status)));
        check (r.file.existsAsFile(), "A-1 file exists", r.file.getFullPathName());
        check (r.file.loadFileAsString().contains ("ASIO4ALL"),
               "A-1 default content written");

        // A-2: 既存ファイルは上書きしない
        const auto custom = "# custom\nMyDriver\n";
        r.file.replaceWithText (custom);
        const auto r2 = convo::asio::resolveBlacklistFile (exeDir, userDir);
        check (r2.status == convo::asio::BlacklistFileStatus::readyInExecutableDirectory,
               "A-2 status is executable directory");
        // 読み出しは改行を CRLF に正規化するため、内容は行単位で確かめる。
        const auto preserved = r2.file.loadFileAsString();
        check (preserved.contains ("MyDriver") && ! preserved.contains ("ASIO4ALL"),
               "A-2 existing file is preserved", preserved.trim());


        AsioBlacklist blacklist;
        check (blacklist.loadFromFile (r2.file), "A-2 loadFromFile succeeds");
        check (blacklist.isBlacklisted ("myDriver v2"), "A-2 custom entry is active");

        base.deleteRecursively();
    }

    // A-3: 実行ファイル隣接に書けなければユーザー領域へ退避する
    {
        const auto base = makeBase ("a3");
        const auto missingExeDir = base.getChildFile ("no_such_exe_dir");
        const auto userDir = base.getChildFile ("user");

        const auto r = convo::asio::resolveBlacklistFile (missingExeDir, userDir);
        check (r.status == convo::asio::BlacklistFileStatus::readyInUserDirectory,
               "A-3 status is user directory", juce::String (static_cast<int> (r.status)));
        check (r.file.existsAsFile(), "A-3 file exists", r.file.getFullPathName());
        check (r.file.getParentDirectory() == userDir.getChildFile ("ConvoPeq"),
               "A-3 file is under the user directory", r.file.getFullPathName());
        check (! missingExeDir.getChildFile ("asio_blacklist.txt").existsAsFile(),
               "A-3 executable directory was not created");

        AsioBlacklist blacklist;
        check (blacklist.loadFromFile (r.file), "A-3 loadFromFile succeeds");

        base.deleteRecursively();
    }

    // A-4: どちらにも用意できなければ unavailable。パスは診断用に実行ファイル隣接を指す
    {
        const auto base = makeBase ("a4");
        const auto missingExeDir = base.getChildFile ("no_such_exe_dir");
        const auto blocker = base.getChildFile ("blocker");
        blocker.replaceWithText ("not a directory");

        const auto r = convo::asio::resolveBlacklistFile (missingExeDir, blocker);
        check (r.status == convo::asio::BlacklistFileStatus::unavailable,
               "A-4 status is unavailable", juce::String (static_cast<int> (r.status)));
        check (r.file == missingExeDir.getChildFile ("asio_blacklist.txt"),
               "A-4 diagnostic path is the executable directory", r.file.getFullPathName());

        AsioBlacklist blacklist;
        check (! blacklist.loadFromFile (r.file), "A-4 loadFromFile fails for a missing file");

        base.deleteRecursively();
    }

    // A-5: 通知の条件
    {
        check (convo::asio::shouldShowBlacklistWarning (false, false),
               "A-5 warn on failure in GUI mode");
        check (! convo::asio::shouldShowBlacklistWarning (false, true),
               "A-5 no warning in automation");
        check (! convo::asio::shouldShowBlacklistWarning (true, false),
               "A-5 no warning on success");
        check (! convo::asio::shouldShowBlacklistWarning (true, true),
               "A-5 no warning on success in automation");
    }

    // A-6: 自動実行の判定
    {
        check (! convo::asio::isAutomationInvocation ("ConvoPeq.exe"),
               "A-6 plain GUI start is not automation");
        check (convo::asio::isAutomationInvocation ("ConvoPeq.exe --cli-run --cli-exit-ms 12000"),
               "A-6 --cli-run is automation");
        check (convo::asio::isAutomationInvocation ("ConvoPeq.exe --cli-ir=C:/x.wav"),
               "A-6 --cli- value form is automation");
        check (convo::asio::isAutomationInvocation ("ConvoPeq.exe --verify-runtime-evidence"),
               "A-6 evidence mode is automation");
        check (convo::asio::isAutomationInvocation ("ConvoPeq.exe --verify-runtime-evidence-live"),
               "A-6 live evidence mode is automation");
    }

    std::printf ("[SUMMARY] failures=%d\n", gFailures);
    return gFailures == 0 ? 0 : 1;
}
