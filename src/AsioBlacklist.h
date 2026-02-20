//============================================================================
#pragma once
// AsioBlacklist.h  ── v0.2 (JUCE 8.0.12対応)
//
// ASIOドライバのブラックリスト管理クラス
// シングルクライアントASIO（BRAVO-HD, ASIO4ALL等）や
// 動作が不安定なドライバを除外するために使用します。
//============================================================================

#include <JuceHeader.h>

class AsioBlacklist
{
public:
    bool loadFromFile(const juce::File& file) noexcept
    {
        blacklist.clear();

        if (! file.existsAsFile())
            return false;

        juce::StringArray lines;
        file.readLines(lines);

        for (const auto& line : lines)
        {
            juce::String trimmedLine = line.trim();

            if (trimmedLine.isEmpty() || trimmedLine.startsWithChar ('#'))
                continue;

            blacklist.add (trimmedLine);
        }
        return true;
    }

    bool isBlacklisted(const juce::String& deviceName) const noexcept
    {
        for (const auto& b : blacklist)
        {
            // バージョン番号や接尾辞（"v2"など）に対応するため、部分一致を使用
            if (deviceName.containsIgnoreCase(b))
                return true;
        }

        return false;
    }

private:
    juce::StringArray blacklist;
};