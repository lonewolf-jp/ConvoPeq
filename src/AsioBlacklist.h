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
    bool loadFromFile (const juce::File& file) noexcept
    {
        blacklist.clear();

        if (! file.existsAsFile())
            return false;

        juce::StringArray lines;
        file.readLines (lines);

        for (auto& line : lines)
        {
            line = line.trim();

            if (line.isEmpty() || line.startsWithChar ('#'))
                continue;

            blacklist.add (line);
        }
        return true;
    }

    bool isBlacklisted (const juce::String& deviceName) const
    {
        for (auto& b : blacklist)
        {
            // より正確な比較のために、文字列全体が一致するか確認 (More robust string comparison)
            if (deviceName.equalsIgnoreCase(b))
                return true;
        }

        return false;
    }

private:
    juce::StringArray blacklist;
};