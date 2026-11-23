// InputInspectionTests.cpp — B-04 / B-05 / B-06 回帰
//
//   B-04: RIFF/WAVE の宣言チャンク長が実ファイル残量を超える入力を拒否する
//   B-05: XML のサイズ・要素深さ・DOCTYPE/ENTITY を解析前に拒否する
//   B-06: プリセット XML は B-05 と同じ入口を通る（MainWindow 側の呼出）
//
// JUCE 本体は改修しない。ここでは ConvoPeq 側の事前検査だけを確認する。

#include <JuceHeader.h>

#include "InputInspection.h"

#include <cstdio>
#include <cstring>
#include <vector>

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

juce::File tempFile (const juce::String& name)
{
    auto f = juce::File::getSpecialLocation (juce::File::tempDirectory)
                 .getChildFile ("convo_inputinspect_" + name);
    f.deleteFile();
    return f;
}

void appendBytes (std::vector<char>& out, const char* data, size_t n)
{
    out.insert (out.end(), data, data + n);
}

void appendU32 (std::vector<char>& out, juce::uint32 v)
{
    out.push_back (static_cast<char> (v & 0xFFu));
    out.push_back (static_cast<char> ((v >> 8) & 0xFFu));
    out.push_back (static_cast<char> ((v >> 16) & 0xFFu));
    out.push_back (static_cast<char> ((v >> 24) & 0xFFu));
}

void appendU16 (std::vector<char>& out, juce::uint16 v)
{
    out.push_back (static_cast<char> (v & 0xFFu));
    out.push_back (static_cast<char> ((v >> 8) & 0xFFu));
}

// 最小の RIFF/WAVE を作る。dataSize を実体と変えて「宣言長の偽装」を再現する。
juce::File makeWav (const juce::String& name, juce::uint32 dataSize, int payloadBytes)
{
    std::vector<char> bytes;
    appendBytes (bytes, "RIFF", 4);
    appendU32 (bytes, 36u);
    appendBytes (bytes, "WAVE", 4);

    appendBytes (bytes, "fmt ", 4);
    appendU32 (bytes, 16u);
    appendU16 (bytes, 1u);        // PCM
    appendU16 (bytes, 2u);        // channels
    appendU32 (bytes, 48000u);    // sample rate
    appendU32 (bytes, 192000u);   // byte rate
    appendU16 (bytes, 4u);        // block align
    appendU16 (bytes, 16u);       // bits

    appendBytes (bytes, "data", 4);
    appendU32 (bytes, dataSize);

    for (int i = 0; i < payloadBytes; ++i)
        bytes.push_back (0);

    const auto f = tempFile (name);
    f.replaceWithData (bytes.data(), bytes.size());
    return f;
}

juce::String makeXml (int depth, bool withDoctype = false)
{
    juce::String s;

    if (withDoctype)
        s << "<!DOCTYPE root [<!ELEMENT root ANY>]>\n";

    s << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";

    for (int i = 0; i < depth; ++i)
        s << "<n" << juce::String (i) << ">";

    s << "x";

    for (int i = depth - 1; i >= 0; --i)
        s << "</n" << juce::String (i) << ">";

    return s;
}

} // namespace

int main()
{
    const juce::ScopedJuceInitialiser_GUI juceInit;

    // ---- B-05: XML ----
    {
        const auto f = tempFile ("ok.xml");
        f.replaceWithText (makeXml (8));

        juce::String reason;
        auto xml = convo::safeinput::parseXmlFile (f, &reason);
        check (xml != nullptr, "X-1 normal XML is accepted", reason);

        const auto g = tempFile ("deep.xml");
        g.replaceWithText (makeXml (200));

        reason.clear();
        xml = convo::safeinput::parseXmlFile (g, &reason);
        check (xml == nullptr && reason.containsIgnoreCase ("too deep"),
               "X-2 deep nesting is rejected", reason);

        const auto d = tempFile ("doctype.xml");
        d.replaceWithText (makeXml (4, true));

        reason.clear();
        xml = convo::safeinput::parseXmlFile (d, &reason);
        check (xml == nullptr && reason.containsIgnoreCase ("DOCTYPE"),
               "X-3 DOCTYPE is rejected", reason);

        const auto e = tempFile ("entity.xml");
        e.replaceWithText ("<?xml version=\"1.0\"?><r><!ENTITY a \"b\">&a;</r>");

        reason.clear();
        xml = convo::safeinput::parseXmlFile (e, &reason);
        check (xml == nullptr && reason.containsIgnoreCase ("ENTITY"),
               "X-4 ENTITY declaration is rejected", reason);

        const auto m = tempFile ("broken.xml");
        m.replaceWithText ("<root><child></root>");

        reason.clear();
        xml = convo::safeinput::parseXmlFile (m, &reason);
        check (xml == nullptr, "X-5 malformed XML is rejected", reason);

        const auto big = tempFile ("big.xml");
        {
            juce::String huge = "<r>";
            huge += juce::String::repeatedString ("a", 9 * 1024 * 1024);
            huge += "</r>";
            big.replaceWithText (huge);
        }

        reason.clear();
        xml = convo::safeinput::parseXmlFile (big, &reason);
        check (xml == nullptr && reason.containsIgnoreCase ("too large"),
               "X-6 oversized XML is rejected", reason);

        // X-7: 保存側のガードが読込上限と同じ値・同じ単位（UTF-8 バイト数）であること。
        //   境界（上限ちょうど＝可、上限+1＝不可）を固定する。
        {
            const juce::int64 limit = convo::safeinput::kMaxXmlFileBytes;

            const auto atLimit = juce::String::repeatedString ("a", static_cast<int> (limit));
            check (convo::safeinput::isXmlTextWithinLimit (atLimit),
                   "X-7 text exactly at the read limit is writable", juce::String (limit));

            const auto overLimit = juce::String::repeatedString ("a", static_cast<int> (limit) + 1);
            check (! convo::safeinput::isXmlTextWithinLimit (overLimit),
                   "X-7 text one byte above the read limit is not writable", juce::String (limit + 1));
        }

        for (auto* p : { &f, &g, &d, &e, &m, &big })
            p->deleteFile();
    }

    // ---- B-04: RIFF チャンク境界 ----
    {
        const auto okFile = makeWav ("ok.wav", 64u, 64);

        juce::String reason;
        check (convo::safeinput::inspectRiffChunkBounds (okFile, &reason),
               "R-1 well-formed WAV is accepted", reason);
        okFile.deleteFile();

        const auto badFile = makeWav ("badsize.wav", 0x7FFFFFFFu, 64);

        reason.clear();
        check (! convo::safeinput::inspectRiffChunkBounds (badFile, &reason),
               "R-2 overstated chunk size is rejected", reason);
        badFile.deleteFile();

        const auto textFile = tempFile ("notes.txt");
        textFile.replaceWithText ("this is not a RIFF file");

        reason.clear();
        check (convo::safeinput::inspectRiffChunkBounds (textFile, &reason),
               "R-3 non-RIFF input is out of scope (accepted)", reason);
        textFile.deleteFile();

        // data の後ろに巨大な bext を宣言したチャンクを足す（JUCE のメタデータ確保を狙う形）
        const auto metaFile = makeWav ("bext.wav", 64u, 64);
        {
            juce::MemoryBlock mb;
            metaFile.loadFileAsData (mb);

            std::vector<char> bytes (static_cast<const char*> (mb.getData()),
                                     static_cast<const char*> (mb.getData()) + mb.getSize());
            appendBytes (bytes, "bext", 4);
            appendU32 (bytes, 0xFFFFFF00u);
            metaFile.replaceWithData (bytes.data(), bytes.size());
        }

        reason.clear();
        check (! convo::safeinput::inspectRiffChunkBounds (metaFile, &reason),
               "R-4 oversized metadata chunk is rejected", reason);
        metaFile.deleteFile();
    }

    std::printf ("[SUMMARY] failures=%d\n", gFailures);
    return gFailures == 0 ? 0 : 1;
}
