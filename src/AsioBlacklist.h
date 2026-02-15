//============================================================================
#pragma once
// AsioBlacklist.h (JUCE 8.0.12対応)
//
// ASIOドライバのブラックリスト管理クラス
// シングルクライアントASIO（BRAVO-HD, ASIO4ALL等）や
// 動作が不安定なドライバを除外するために使用します。
//============================================================================

#include <JuceHeader.h>

class AsioBlacklist
{
public:
    void loadFromFile (const juce::File& file)
    {
        blacklist.clear();

        if (! file.existsAsFile())
            return;

        juce::StringArray lines;
        file.readLines (lines);

        for (auto& line : lines)
        {
            line = line.trim();

            if (line.isEmpty() || line.startsWithChar ('#'))
                continue;

            blacklist.add (line);
        }
    }

    bool isBlacklisted (const juce::String& deviceName) const
    {
        for (auto& b : blacklist)
            if (deviceName.containsIgnoreCase (b))
                return true;

        return false;
    }

private:
    juce::StringArray blacklist;
};