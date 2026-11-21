//==============================================================================
// SizeCappedFileLogger.h
//
// サイズ上限を持つファイルロガー。
//
//   juce::FileLogger の maxInitialFileSizeBytes はコンストラクタでしか効かないため、
//   長時間動くプロセスではログが上限を超えて伸び続ける（C-23 / review 指摘）。
//   このクラスは書き込みのたびにサイズを見て、上限に達したら 1 回だけローテートする
//   （<name>.1 へ退避）。ディスク上の合計は 2 × cap 未満に収まる。
//
//   起動時に既存ログが上限を超えている場合は、末尾 cap バイトだけを残して切り詰める
//   （退避へ丸ごと移すと、巨大な旧ログがそのまま残る）。
//   ローテートに失敗した場合は、その場で切り詰める。それも失敗したら書込みを捨てる。
//
//   書き込み方式は juce::FileLogger と同じ（メッセージごとに FileOutputStream を開く）。
//   ロガーは任意のスレッドから呼ばれるため、内部で直列化する。
//==============================================================================

#pragma once

#include <JuceHeader.h>

class SizeCappedFileLogger : public juce::Logger
{
public:
    SizeCappedFileLogger (const juce::File& file,
                          const juce::String& welcomeMessage,
                          juce::int64 maxBytes)
        : logFile (file),
          cap (juce::jmax (juce::int64 { 64 * 1024 }, maxBytes))
    {
        const juce::ScopedLock sl (lock);

        trimToCapLocked();

        if (! logFile.exists())
            logFile.create();

        writeLocked (juce::newLine
                     + "**********************************************************" + juce::newLine
                     + welcomeMessage + juce::newLine
                     + "Log started: " + juce::Time::getCurrentTime().toString (true, true));
    }

    ~SizeCappedFileLogger() override = default;

protected:
    void logMessage (const juce::String& message) override
    {
        const juce::ScopedLock sl (lock);
        const auto incoming = static_cast<juce::int64> (message.getNumBytesAsUTF8()) + 2;

        if (logFile.getSize() + incoming > cap)
        {
            // 退避も切り詰めもできなければ、上限を守るためにこの 1 行を捨てる。
            if (! rotateLocked())
                return;
        }

        writeLocked (message);
    }

private:
    juce::File rotatedFile() const
    {
        return logFile.getSiblingFile (logFile.getFileNameWithoutExtension()
                                       + ".1"
                                       + logFile.getFileExtension());
    }

    // 戻り値: 上限内に収まる状態になったか（退避または切り詰めに成功）。
    bool rotateLocked()
    {
        if (! logFile.existsAsFile())
            return logFile.create();

        auto previous = rotatedFile();
        previous.deleteFile();

        if (logFile.moveFileTo (previous))
            return true;

        // 退避できない（バックアップがロックされている等）。その場で空にして上限を守る。
        return logFile.replaceWithText ({});
    }

    // 起動時: 既存ログが上限を超えていたら、末尾 cap バイトだけを残す。
    void trimToCapLocked()
    {
        const auto size = logFile.getSize();
        if (size <= cap)
            return;

        juce::TemporaryFile temp (logFile);
        {
            juce::FileOutputStream out (temp.getFile());
            juce::FileInputStream in (logFile);

            if (! (out.openedOk() && in.openedOk()))
            {
                // 切り詰められない場合は退避を試す。それも駄目なら空にする。
                rotateLocked();
                return;
            }

            in.setPosition (size - cap);

            // 行の途中から始めない。
            for (;;)
            {
                const auto c = in.readByte();
                if (c == 0)
                    return;
                if (c == '\n' || c == '\r')
                {
                    out << c;
                    break;
                }
            }

            out.writeFromInputStream (in, -1);
        }

        if (! temp.overwriteTargetFileWithTemporary())
        {
            temp.getFile().deleteFile();
            rotateLocked();
        }
    }

    void writeLocked (const juce::String& message)
    {
        juce::FileOutputStream out (logFile, 256);
        if (out.openedOk())
            out << message << juce::newLine;
    }

    juce::File logFile;
    juce::CriticalSection lock;
    juce::int64 cap;
};
