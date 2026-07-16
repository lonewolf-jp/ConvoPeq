#pragma once

#include <cstdint>
#include <type_traits>
#include <cmath>

#pragma warning(push)
#pragma warning(disable : 4324) // C4324: キャッシュライン分離用alignasによる意図的なパディングを許容

namespace convo {

// ★ v14.0: ポータブル finite チェック — icx/clang-cl では std::isfinite(float) が
//   利用できない場合があるため double 経由で判定。
[[nodiscard]] inline bool isFiniteFloat(float val) noexcept
{
    return std::isfinite(static_cast<double>(val)) != 0;
}

struct BuildInput final {
    double sampleRate = 0.0;
    int blockSize = 0;
    int ditherBitDepth = 0;
    int oversamplingFactor = 0;
    int oversamplingType = 0;
    int noiseShaperType = 0;
    int processingOrder = 0;
    bool eqBypassed = false;
    bool convBypassed = false;
    bool softClipEnabled = false;
    double saturationAmount = 0.0;
    double inputHeadroomGain = 1.0;
    double outputMakeupGain = 1.0;
    double convolverInputTrimGain = 1.0;
    bool autoGainStagingEnabled = false;  // ★ v14.0: Auto Gain Staging フラグ
};

struct RuntimeBuildFingerprint
{
    std::uint32_t fingerprintVersion = 1;
    std::uint64_t irIdentityHash = 0;
    std::uint64_t convolutionConfigHash = 0;
    std::uint64_t dspParameterHash = 0;
    double sampleRate = 0.0;
    int blockSize = 0;
};

struct RuntimeBuildSnapshot
{
    int generation = 0;
    BuildInput buildInput {};
    std::uint64_t convolverFingerprint = 0;
    RuntimeBuildFingerprint rebuildFingerprint {};
    bool sealed = false;

    // [PR-2] DSP semantic projection snapshot values
    // These fields are populated from DSPCore when snapshot is created,
    // and consumed by RuntimeBuilder::buildRuntimePublishWorld() to
    // construct dspProjection without DSPCore direct reads.
    bool irLoaded = false;
    bool irFinalized = false;
    std::uint64_t structuralHash = 0;
    int oversamplingFactor = 1;
    double sampleRate = 48000.0;
    int baseLatencySamples = 0;
};

// ★ v14.0: BuildAnalysis — DSP 解析結果の封印。
//   RuntimeBuildSnapshot から分離された解析値。
//   sealed 契約: generation 一致 / finite 検証 / Builder 変更禁止
struct BuildAnalysis {
    int generation = 0;
    float eqMaxGainDb = 0.0f;
    float additionalAttenuationDb = 0.0f;
    bool sealed = false;
};

// ★ v14.0: sealBuildAnalysis — BuildAnalysis を封印し Builder 変更禁止を表明。
//   封印契約:
//   ① sealedSnapshot != nullptr
//   ② generation == snapshot->generation
//   ③ snapshot->sealed == true
//   ④ 全浮動小数点値が finite
//   ⑤ sealed = true を設定
//   戻り値: 封印後の BuildAnalysis（sealed=true）。契約違反時はデフォルト構築を返す。
[[nodiscard]] inline BuildAnalysis sealBuildAnalysis(
    BuildAnalysis analysis,
    const RuntimeBuildSnapshot* snapshot) noexcept
{
    if (snapshot == nullptr)
        return BuildAnalysis{};

    if (analysis.generation != snapshot->generation)
        return BuildAnalysis{};

    if (!snapshot->sealed)
        return BuildAnalysis{};

    if (!isFiniteFloat(analysis.eqMaxGainDb) || !isFiniteFloat(analysis.additionalAttenuationDb))
        return BuildAnalysis{};

    analysis.sealed = true;
    return analysis;
}

// ★ v14.0: verifyBuildAnalysisPair — BuildAnalysis と RuntimeBuildSnapshot の
//   ペアリング整合性を検証。Orchestrator 側の jassert として使用可能。
//   検証内容:
//   ① analysis.sealed == true
//   ② snapshot.sealed == true
//   ③ analysis.generation == snapshot.generation
//   ④ 全浮動小数点値が finite
[[nodiscard]] inline bool verifyBuildAnalysisPair(
    const BuildAnalysis& analysis,
    const RuntimeBuildSnapshot& snapshot) noexcept
{
    if (!analysis.sealed || !snapshot.sealed)
        return false;

    if (analysis.generation != snapshot.generation)
        return false;

    if (!isFiniteFloat(analysis.eqMaxGainDb) || !isFiniteFloat(analysis.additionalAttenuationDb))
        return false;

    return true;
}

static_assert(std::is_same_v<decltype(RuntimeBuildSnapshot{}.buildInput), BuildInput>,
              "RuntimeBuildSnapshot must use convo::BuildInput as the sole semantic input descriptor.");

[[nodiscard]] inline bool isRuntimeBuildSnapshotSealedAndCompatible(const RuntimeBuildSnapshot& snapshot,
                                                                    const RuntimeBuildSnapshot& other) noexcept
{
    if (!snapshot.sealed || !other.sealed)
        return false;

    if (snapshot.rebuildFingerprint.fingerprintVersion != other.rebuildFingerprint.fingerprintVersion)
        return false;

    return snapshot.buildInput.sampleRate == other.buildInput.sampleRate
        && snapshot.buildInput.blockSize == other.buildInput.blockSize
        && snapshot.buildInput.ditherBitDepth == other.buildInput.ditherBitDepth
        && snapshot.buildInput.oversamplingFactor == other.buildInput.oversamplingFactor
        && snapshot.buildInput.oversamplingType == other.buildInput.oversamplingType
        && snapshot.buildInput.noiseShaperType == other.buildInput.noiseShaperType
        && snapshot.buildInput.processingOrder == other.buildInput.processingOrder
        && snapshot.buildInput.eqBypassed == other.buildInput.eqBypassed
        && snapshot.buildInput.convBypassed == other.buildInput.convBypassed
        && snapshot.buildInput.softClipEnabled == other.buildInput.softClipEnabled
        && snapshot.buildInput.saturationAmount == other.buildInput.saturationAmount
        && snapshot.buildInput.inputHeadroomGain == other.buildInput.inputHeadroomGain
        && snapshot.buildInput.outputMakeupGain == other.buildInput.outputMakeupGain
        && snapshot.buildInput.convolverInputTrimGain == other.buildInput.convolverInputTrimGain
        && snapshot.buildInput.autoGainStagingEnabled == other.buildInput.autoGainStagingEnabled
        && snapshot.convolverFingerprint == other.convolverFingerprint
        && snapshot.rebuildFingerprint.irIdentityHash == other.rebuildFingerprint.irIdentityHash
        && snapshot.rebuildFingerprint.convolutionConfigHash == other.rebuildFingerprint.convolutionConfigHash
        && snapshot.rebuildFingerprint.dspParameterHash == other.rebuildFingerprint.dspParameterHash;
}

} // namespace convo

#pragma warning(pop)
