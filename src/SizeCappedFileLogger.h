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

        // 既存ファイルが上限を超えていたら退避してから始める。
        if (logFile.existsAsFile() && logFile.getSize() > cap)
            rotateLocked();

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
            rotateLocked();

        writeLocked (message);
    }

private:
    juce::File rotatedFile() const
    {
        return logFile.getSiblingFile (logFile.getFileNameWithoutExtension()
                                       + ".1"
                                       + logFile.getFileExtension());
    }

    void rotateLocked()
    {
        auto previous = rotatedFile();
        previous.deleteFile();
        if (logFile.existsAsFile())
            logFile.moveFileTo (previous);
        logFile.create();
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
