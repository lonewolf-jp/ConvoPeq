#pragma once

#include <JuceHeader.h>

#include <cstdint>
#include <memory>

// B-04 / B-05: 入力ファイルの事前検査。
//
//   JUCE 本体（vendored）は改修せず、ConvoPeq 側の入口で上限を課す。
//   - RIFF/WAVE: 宣言チャンク長が実ファイルの残量を超えないこと（B-04）
//   - XML: サイズ・要素の深さ・DOCTYPE/ENTITY の不在（B-05）
//
//   どちらも「アプリの admission より前に確保・再帰が走る」経路への対処で、
//   ファイル内容の正当性までは判定しない。判定は後段のパーサと各 admission が行う。

namespace convo::safeinput
{

// B-05 の既定上限。8 MiB はプリセット・セッション・学習状態の実測最大
// （数百 KiB 規模）に対して十分な余裕があり、深さ 64 は JUCE の
// readChildElements 再帰が安全に扱える範囲として置いている。
inline constexpr juce::int64 kMaxXmlFileBytes = 8 * 1024 * 1024;
inline constexpr int kMaxXmlDepth = 64;

struct XmlLimits
{
    juce::int64 maxBytes = kMaxXmlFileBytes;
    int maxDepth = kMaxXmlDepth;
    bool rejectDoctype = true;
};

// B-04: RIFF/WAVE のチャンク境界検査。
//   RIFF/WAVE 以外（他フォーマット・短すぎるファイル）は対象外として true を返す。
bool inspectRiffChunkBounds (const juce::File& file, juce::String* reason = nullptr);

// B-05: XML のサイズ・深さ・DOCTYPE/ENTITY を事前検査する。
bool inspectXmlFile (const juce::File& file, juce::String* reason = nullptr,
                     const XmlLimits& limits = {});

// B-05 / B-06: 事前検査を通ったものだけを JUCE のパーサへ渡す。
//   失敗時は nullptr を返し、reason に理由を入れる（既存の parse と同型）。
std::unique_ptr<juce::XmlElement> parseXmlFile (const juce::File& file,
                                                juce::String* reason = nullptr,
                                                const XmlLimits& limits = {});

// 保存側と読込側の上限を一致させるための判定。
//   読込（inspectXmlFile）はファイルサイズを maxBytes と比較するので、保存側も
//   UTF-8 のバイト数で同じ上限と比較する。文字数で比較すると、非 ASCII を含む
//   文書で保存は通るが次回起動時に読込拒否される組み合わせが生じ得る。
//   （JUCE の XML 出力は非 ASCII を &#NNNN; へエスケープするため実測では一致するが、
//     契約としてはバイト数で揃える。）
inline bool isXmlTextWithinLimit (const juce::String& text, const XmlLimits& limits = {})
{
    return static_cast<juce::int64> (text.getNumBytesAsUTF8()) <= limits.maxBytes;
}

} // namespace convo::safeinput
