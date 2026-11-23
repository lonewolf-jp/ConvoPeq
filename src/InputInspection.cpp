#include "InputInspection.h"

#include <cstddef>
#include <cstring>
#include <limits>

namespace convo::safeinput
{
namespace
{

// String 内のタグ構造を走査して最大の入れ子深さを返す。
// 解釈できない構造（未終端のコメント/タグ等）は -1 を返す。
// 引用符内の '>' はタグ終端として扱わない。
int computeMaxElementDepth (const juce::String& text)
{
    // ★ B 修正案 A: 文字位置参照（O(n^2)）をやめ、UTF-8 バイト列をポインタで走査する。
    //   判定条件・分岐順序・戻り値は現行実装と同一。
    const char* d = text.toRawUTF8();
    const std::size_t numBytes = text.getNumBytesAsUTF8();

    // C4267: size_t -> int の変換前に範囲を検査する。8 MiB の上限（kMaxXmlFileBytes）では
    // 到達しないが、公開 API が limits を差し替え可能なため、契約として明示的に弾く。
    if (numBytes > static_cast<std::size_t> (std::numeric_limits<int>::max()))
        return -1;

    const int n = static_cast<int> (numBytes);
    int depth = 0;
    int maxDepth = 0;
    int i = 0;
    const auto at = [d] (int k) { return static_cast<unsigned char> (d[k]); };

    while (i < n)
    {
        if (at (i) != '<')
        {
            ++i;
            continue;
        }

        // コメント
        if (i + 3 < n && at (i + 1) == '!' && at (i + 2) == '-' && at (i + 3) == '-')
        {
            int commentEnd = -1;

            for (int k = i + 4; k + 2 < n; ++k)
                if (at (k) == '-' && at (k + 1) == '-' && at (k + 2) == '>') { commentEnd = k; break; }

            if (commentEnd < 0)
                return -1;

            i = commentEnd + 3;
            continue;
        }

        // CDATA（大文字小文字を区別しない）
        if (i + 8 < n)
        {
            static const char* kCdata = "<![CDATA[";
            bool isCdata = true;

            for (int k = 0; k < 9; ++k)
            {
                unsigned char c = at (i + k);
                const unsigned char t = static_cast<unsigned char> (kCdata[k]);

                if (c >= static_cast<unsigned char> ('a') && c <= static_cast<unsigned char> ('z'))
                    c = static_cast<unsigned char> (c - 'a' + 'A');

                if (c != t) { isCdata = false; break; }
            }

            if (isCdata)
            {
                int cdataEnd = -1;

                for (int k = i + 9; k + 2 < n; ++k)
                    if (at (k) == ']' && at (k + 1) == ']' && at (k + 2) == '>') { cdataEnd = k; break; }

                if (cdataEnd < 0)
                    return -1;

                i = cdataEnd + 3;
                continue;
            }
        }

        // DOCTYPE / 宣言 / 処理命令
        if (i + 1 < n && (at (i + 1) == '!' || at (i + 1) == '?'))
        {
            int declEnd = -1;

            for (int k = i + 1; k < n; ++k)
                if (at (k) == '>') { declEnd = k; break; }

            if (declEnd < 0)
                return -1;

            i = declEnd + 1;
            continue;
        }

        const bool isClosing = (i + 1 < n && at (i + 1) == '/');

        // タグ終端を引用符を考慮して探す
        unsigned char quote = 0;
        int tagEnd = -1;

        for (int j = i + 1; j < n; ++j)
        {
            const unsigned char c = at (j);

            if (quote != 0)
            {
                if (c == quote)
                    quote = 0;

                continue;
            }

            if (c == static_cast<unsigned char> ('"') || c == static_cast<unsigned char> ('\''))
            {
                quote = c;
                continue;
            }

            if (c == '>')
            {
                tagEnd = j;
                break;
            }
        }

        if (tagEnd < 0)
            return -1;

        bool selfClosing = false;

        if (! isClosing)
        {
            int k = tagEnd - 1;

            while (k > i && (at (k) == static_cast<unsigned char> (' ')
                             || at (k) == static_cast<unsigned char> ('\t')
                             || at (k) == static_cast<unsigned char> ('\n')
                             || at (k) == static_cast<unsigned char> ('\r')))
                --k;

            selfClosing = (at (k) == '/');
        }

        if (isClosing)
        {
            if (depth > 0)
                --depth;
        }
        else if (! selfClosing)
        {
            ++depth;

            if (depth > maxDepth)
                maxDepth = depth;
        }

        i = tagEnd + 1;
    }

    return maxDepth;
}


juce::String describe (const juce::File& file)
{
    return file.getFileName();
}

} // namespace

bool inspectRiffChunkBounds (const juce::File& file, juce::String* reason)
{
    const auto fail = [reason, &file] (const juce::String& why)
    {
        if (reason != nullptr)
            *reason = why + " (" + describe (file) + ")";

        return false;
    };

    if (! file.existsAsFile())
        return fail ("IR file not found");

    const juce::int64 fileSize = file.getSize();

    if (fileSize < 12)
        return true; // RIFF ではない（他フォーマットの可能性）

    juce::FileInputStream in (file);

    if (! in.openedOk())
        return fail ("cannot open file for inspection");

    char header[12] = {};

    if (in.read (header, 12) != 12)
        return true;

    const bool isRiff = std::memcmp (header, "RIFF", 4) == 0
                     || std::memcmp (header, "RF64", 4) == 0;

    if (! isRiff || std::memcmp (header + 8, "WAVE", 4) != 0)
        return true; // RIFF/WAVE 以外は対象外

    constexpr int kMaxChunks = 4096;
    juce::int64 pos = 12;

    for (int chunk = 0; chunk < kMaxChunks && pos + 8 <= fileSize; ++chunk)
    {
        if (! in.setPosition (pos))
            return true;

        juce::uint8 header8[8] = {};

        if (in.read (header8, 8) != 8)
            return true;

        const juce::uint32 declared = static_cast<juce::uint32> (header8[4])
                                    | (static_cast<juce::uint32> (header8[5]) << 8)
                                    | (static_cast<juce::uint32> (header8[6]) << 16)
                                    | (static_cast<juce::uint32> (header8[7]) << 24);

        const juce::int64 payloadStart = pos + 8;

        if (declared == 0xFFFFFFFFu)
        {
            // RF64: 実サイズは ds64 チャンク側。ここでは境界を断定しない。
            pos = payloadStart;
            continue;
        }

        const juce::int64 remaining = fileSize - payloadStart;

        if (static_cast<juce::int64> (declared) > remaining)
            return fail ("declared chunk size exceeds the file size");

        pos = payloadStart + static_cast<juce::int64> (declared)
            + ((declared & 1u) != 0u ? 1 : 0);
    }

    return true;
}

bool inspectXmlFile (const juce::File& file, juce::String* reason, const XmlLimits& limits)
{
    const auto fail = [reason, &file] (const juce::String& why)
    {
        if (reason != nullptr)
            *reason = why + " (" + describe (file) + ")";

        return false;
    };

    if (! file.existsAsFile())
        return fail ("XML file not found");

    const juce::int64 size = file.getSize();

    if (size <= 0)
        return fail ("XML file is empty");

    if (size > limits.maxBytes)
        return fail ("XML file is too large (" + juce::String (size)
                     + " bytes; limit is " + juce::String (limits.maxBytes) + " bytes)");

    const juce::String text = file.loadFileAsString();

    if (limits.rejectDoctype)
    {
        if (text.containsIgnoreCase ("<!DOCTYPE"))
            return fail ("XML with DOCTYPE is not accepted");

        if (text.containsIgnoreCase ("<!ENTITY"))
            return fail ("XML with ENTITY declarations is not accepted");
    }

    const int maxDepth = computeMaxElementDepth (text);

    if (maxDepth < 0)
        return fail ("XML structure could not be inspected");

    if (maxDepth > limits.maxDepth)
        return fail ("XML nesting is too deep (depth " + juce::String (maxDepth)
                     + "; limit is " + juce::String (limits.maxDepth) + ")");

    return true;
}

std::unique_ptr<juce::XmlElement> parseXmlFile (const juce::File& file,
                                                juce::String* reason,
                                                const XmlLimits& limits)
{
    if (! inspectXmlFile (file, reason, limits))
        return nullptr;

    auto xml = juce::XmlDocument::parse (file);

    if (xml == nullptr && reason != nullptr && reason->isEmpty())
        *reason = "XML could not be parsed (" + describe (file) + ")";

    return xml;
}

} // namespace convo::safeinput
