#pragma once

//============================================================================
// IRLoadAdmission.h
//
// WORK102 (big 1-8) — IR load admission contract の単一情報源。
//
// 契約の正本は doc/work102/big18_failure_contract_arbitration_20260917.md（R2 凍結）:
//   FC-FORM-1  source loadedIR byte bound      numChannels × fileLength × 8 ≤ kMaxIRLoadBytes
//   FC-FORM-2  channel bound                   numChannels ≤ kMaxIRLoadChannels
//   FC-FORM-3  INT32 representation bound      fileLength ≤ kMaxFileLengthSamples
//   FC-FORM-4  degenerate input bound          numChannels ≥ 1 ∧ fileLength ≥ 1
//   FC-FORM-5  resample output bound           L_trim / fileSR ≤ kMaxIRResampleOutputSamples / sr
//   FC-FORM-6  hash allocation bound           computeIRHash は物理ファイルサイズ比例の確保を行わない
//
// 本ヘッダは「値の追加」を行わない。定数は R2 で凍結された 4 値のみ。
// 目的は (i) predicate の純関数化（I/O 非依存 → 境界を単体テスト可能にする）と
//          (ii) 診断文言の単一情報源化（FC-INV-8）。
//
// 適用順序（FC-INV-9。実装側で守る）:
//   FC-FORM-4 → FC-FORM-3 → FC-FORM-2 → FC-FORM-1 → [確保開始] → FC-FORM-6 → FC-FORM-5
//   FC-FORM-5 だけは loadedIR の trim 後にしか評価できない（L_trim が必要）。他の 4 つは
//   reader のメタデータのみで判定できるため、O(fileLength)/O(file bytes) の確保より前に置ける。
//============================================================================

#include <JuceHeader.h>

#include <cstdint>

namespace convo::irload
{

//----------------------------------------------------------
// 凍結された契約値（R2 §11.2。変更禁止）
//----------------------------------------------------------

// FC-FORM-1: loadedIR の byte 上限。IR file SR envelope 44.1–768 kHz（R2-D2）からの
// 導出値: 2ch × ceil(2^21 × 768000/44100) × 8 = 584,349,296 B → 以上の最小 2 冪 = 2^30。
inline constexpr int64_t kMaxIRLoadBytes = 1073741824; // 1 GiB

// FC-FORM-2: channel 上限。policy value（導出値ではない。R2-D3）。
// N ≤ 8 では SR-01B の「>2ch→先頭2」凍結挙動を維持し、N > 8 のみ新規拒否する。
inline constexpr unsigned kMaxIRLoadChannels = 8u;

// FC-FORM-3: fileLength の表現上限（既存 MAX_FILE_LENGTH と同値）。
// 製品上限ではなく AudioBuffer::setSize(int) への narrowing の型契約。
inline constexpr int64_t kMaxFileLengthSamples = 2147483647; // INT32_MAX

// FC-FORM-5: resample 出力長の上限。MAX_IR_LATENCY (2^21) + 1。
// DSP の targetLength 上限に整合し、r8b getMaxOutLen の ceil(...) + 1 のテール分を許容する。
// allocation bound ではなく output length (utility) bound である（FC-INV-3）。
inline constexpr int64_t kMaxIRResampleOutputSamples = 2097153; // 2^21 + 1

//----------------------------------------------------------
// 純述語（unsigned / int64 入力。I/O 非依存）
//----------------------------------------------------------

// FC-FORM-2。narrowing 前の unsigned 値で判定する（IG §4.4）。
// reader->numChannels は unsigned int であり、int へ narrowing すると
// INT_MAX 超の値が負に化けるため、型レベルでここに集約する。
[[nodiscard]] inline bool admitChannelCount(unsigned numChannels) noexcept
{
    return numChannels >= 1u && numChannels <= kMaxIRLoadChannels;
}

// FC-FORM-4（channel 側）。上記のうち 0 を弾く部分を独立に述べたもの。
[[nodiscard]] inline bool admitChannelCountNonZero(unsigned numChannels) noexcept
{
    return numChannels >= 1u;
}

// FC-FORM-4（length 側）。
[[nodiscard]] inline bool admitFileLengthNonZero(int64_t fileLength) noexcept
{
    return fileLength >= 1;
}

// FC-FORM-3。
[[nodiscard]] inline bool admitFileLengthRepresentable(int64_t fileLength) noexcept
{
    return fileLength <= kMaxFileLengthSamples;
}

// FC-FORM-1。除算形で実装し、未検証値の乗算を 1 回も行わない（IG §4.2）。
// 前提: numChannels ∈ [1, kMaxIRLoadChannels]、fileLength ≥ 1（呼出順序で保証）。
[[nodiscard]] inline bool admitByteBudget(unsigned numChannels, int64_t fileLength) noexcept
{
    if (!admitChannelCount(numChannels) || !admitFileLengthNonZero(fileLength))
        return false;

    const int64_t denom = static_cast<int64_t>(numChannels) * static_cast<int64_t>(sizeof(double));
    // denom ∈ [8, 64]（上記 predicate により自明に安全）
    return fileLength <= kMaxIRLoadBytes / denom;
}

// FC-FORM-5。L_trim（末尾無音トリム後の loadedIR 長）に対してのみ評価する。
// raw fileLength では評価しない（R2-D4。無音テール付きファイルの回帰を避けるため）。
// fileSR / sr のいずれかが 0 以下のときはリサンプル自体が実行されないため vacuous に通す。
[[nodiscard]] inline bool admitResampleOutput(int64_t trimmedLength,
                                              double fileSampleRate,
                                              double processingSampleRate) noexcept
{
    if (fileSampleRate <= 0.0 || processingSampleRate <= 0.0)
        return true; // resample 非実行（呼出側の同じガードと一致）

    if (trimmedLength <= 0)
        return false;

    const double required = static_cast<double>(trimmedLength) / fileSampleRate;
    const double allowed = static_cast<double>(kMaxIRResampleOutputSamples) / processingSampleRate;
    return required <= allowed;
}

//----------------------------------------------------------
// 診断（FC-INV-8 の単一情報源。dimension + actual + limit を開示する）
//----------------------------------------------------------

// 前提: numChannels ≤ kMaxIRLoadChannels ∧ fileLength ≤ kMaxFileLengthSamples。
//       （FC-FORM-2/3 の後でしか呼ばれないため、積は 2^38 未満で int64 に収まる）
[[nodiscard]] inline juce::String diagnosticByteLimit(unsigned numChannels, int64_t fileLength)
{
    jassert(numChannels >= 1u && numChannels <= kMaxIRLoadChannels);
    jassert(fileLength >= 0 && fileLength <= kMaxFileLengthSamples);

    const int64_t bytes = static_cast<int64_t>(numChannels)
                        * static_cast<int64_t>(fileLength)
                        * static_cast<int64_t>(sizeof(double));

    return "IR file is too large for the load memory limit ("
         + juce::String(static_cast<juce::int64>(numChannels)) + " channels x "
         + juce::String(static_cast<juce::int64>(fileLength)) + " samples x "
         + juce::String(static_cast<juce::int64>(sizeof(double))) + " bytes = "
         + juce::String(static_cast<juce::int64>(bytes)) + " bytes; limit is "
         + juce::String(static_cast<juce::int64>(kMaxIRLoadBytes)) + " bytes).";
}

[[nodiscard]] inline juce::String diagnosticChannelLimit(unsigned numChannels)
{
    if (numChannels == 0u)
        return "Invalid channel count in IR file.";

    return "IR file has too many channels ("
         + juce::String(static_cast<juce::int64>(numChannels)) + " channels; limit is "
         + juce::String(static_cast<juce::int64>(kMaxIRLoadChannels)) + ").";
}

[[nodiscard]] inline juce::String diagnosticLengthLimit(int64_t fileLength)
{
    if (fileLength < 1)
        return "Invalid IR file length (" + juce::String(static_cast<juce::int64>(fileLength)) + " samples).";

    return "IR file is too long ("
         + juce::String(static_cast<juce::int64>(fileLength)) + " samples; limit is "
         + juce::String(static_cast<juce::int64>(kMaxFileLengthSamples)) + " samples).";
}

// 前提: fileSampleRate > 0 ∧ processingSampleRate > 0。
[[nodiscard]] inline juce::String diagnosticResampleLimit(int64_t trimmedLength,
                                                          double fileSampleRate,
                                                          double processingSampleRate)
{
    const double requiredSec = static_cast<double>(trimmedLength) / fileSampleRate;
    const double allowedSec = static_cast<double>(kMaxIRResampleOutputSamples) / processingSampleRate;

    return "IR is longer than the DSP can use at this sample rate ("
         + juce::String(requiredSec, 3) + " s at "
         + juce::String(fileSampleRate, 1) + " Hz; limit is "
         + juce::String(allowedSec, 3) + " s at "
         + juce::String(processingSampleRate, 1) + " Hz = "
         + juce::String(static_cast<juce::int64>(kMaxIRResampleOutputSamples)) + " samples).";
}

} // namespace convo::irload
