#pragma once

#include <cstdint>
#include <type_traits>
#include <cmath>
#include <algorithm>

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

//==============================================================================
// ★ v14.27: OversamplingResult — オーバーサンプリング倍率の解決結果。
//   OversamplingPolicy::resolve() が唯一の生成権限（Authority Singularization）。
//   - resolvedOsFactor: 解決済み倍率 {1,2,4,8}
//   - requestedOsFactor: BuildInput からの要求値（0=Auto）
//   - isAutoResolved: Auto(0) からの解決済みか
//   - supported: Capability（入力 SR が処理可能か）。supported==false は Publish スキップ条件。
//   isValid(): resolvedOsFactor の値域のみ検証（supported とは独立）
//==============================================================================
struct OversamplingResult {
    int resolvedOsFactor = 1;
    int requestedOsFactor = 0;
    bool isAutoResolved = false;
    bool supported = true;

    [[nodiscard]] bool isValid() const noexcept {
        switch (resolvedOsFactor) {
            case 1: case 2: case 4: case 8: return true;
            default: return false;
        }
    }
};

//==============================================================================
// ★ v14.13: BoundMethod — 安全側上界（upperBound）算出方式の識別子。
//==============================================================================
enum class BoundMethod : uint8_t {
    Unknown = 0,
    Legacy = 1,
    TriangleProduct = 2,          // Π(1+|Hi-1|) — 現在のアルゴリズム（v14.22以降）
    ProductMaxMagnitude = 3,      // Π max(1,|Hi|) — 将来の候補（未実装）
    ExactSampling = 4             // 適応サンプリング直接 — 将来の候補（未実装）
};

//==============================================================================
// ★ v14.39: EqGainAlgorithm — 解析アルゴリズム識別子。
//==============================================================================
enum class EqGainAlgorithm : uint8_t {
    Legacy = 0,
    TriangleProductV1 = 1
};

//==============================================================================
// ★ v14.41: SelectedEstimate — Builder collapse で採用された推定値。
//==============================================================================
enum class SelectedEstimate : uint8_t {
    Unknown = 0,
    Measured = 1,
    UpperBound = 2
};

//==============================================================================
// ★ v14.21: AnalysisVersionPolicy — 解析バージョン管理。
//   kCurrent が唯一の現在バージョン定義。
//   increment 条件: Planner 入力フィールドの追加/削除/変更、解析アルゴリズム変更。
//   診断専用フィールドの追加/変更は version 不変。
//==============================================================================
struct AnalysisVersionPolicy {
    static constexpr uint8_t kCurrent = 2;
    static constexpr uint8_t kLegacy = 1;

    // 現在 version で既知の全 version を返す（verifyDiagnostics 用）
    [[nodiscard]] static constexpr bool isKnown(uint8_t version) noexcept {
        return version == kCurrent || version == kLegacy;
    }

    // 特定 version における collapse tolerance を返す
    [[nodiscard]] static constexpr float collapseTolerance(uint8_t version) noexcept {
        return (version >= kCurrent) ? 0.1f : 0.2f;
    }
};

//==============================================================================
// ★ v14.37: BuildDiagnostics — BuildAnalysis から完全分離された診断情報。
//   ISR 思想: Analysis は Publish 対象（Runtime World に写像される）、
//   Diagnostics は Debug 対象（Runtime World とは別世界）。
//==============================================================================
struct BuildDiagnostics {
    uint8_t analysisVersion = AnalysisVersionPolicy::kCurrent;
    EqGainAlgorithm eqGainAlgorithm = EqGainAlgorithm::TriangleProductV1;
    BoundMethod boundMethod = BoundMethod::TriangleProduct;
    SelectedEstimate selectedEstimate = SelectedEstimate::Measured;
    float eqMeasuredGainDb = 0.0f;
    float eqMeasuredRawGainDb = 0.0f;      // ★ v14.47: 放物線補間前の measured 生値（dB）
    float eqUpperBoundGainDb = 0.0f;
    float eqMeasuredFreqHz = 0.0f;         // measured のピーク周波数
    float eqUpperBoundFreqHz = 0.0f;       // upperBound のピーク周波数
    float boundExcessDb = 0.0f;
    float totalMaxQ = 0.0f;               // 全有効バンド中の最大Q値。診断専用
};

//==============================================================================
// ★ v14.0: BuildAnalysis — DSP 解析結果の封印。
//   RuntimeBuildSnapshot から分離された解析値。
//   sealed 契約: generation 一致 / finite 検証 / Builder 変更禁止
//   ★ v14.37: eqMaxQ / irFreqPeakGainDb 追加。additionalAttenuationDb は互換性維持。
//==============================================================================
struct BuildAnalysis {
    int generation = 0;
    float eqMaxGainDb = 0.0f;
    float eqMaxQ = 0.0f;                  // ★ v14.35: ブースト対象バンド中の最大Q値。Planner 使用
    float irFreqPeakGainDb = 0.0f;        // ★ v14.2: IR 周波数ピークゲイン
    float additionalAttenuationDb = 0.0f; // 互換性維持、常に0
    bool sealed = false;
};

//==============================================================================
// ★ v14.0: sealBuildAnalysis — BuildAnalysis を封印し Builder 変更禁止を表明。
//   封印契約:
//   ① sealedSnapshot != nullptr
//   ② generation == snapshot->generation
//   ③ snapshot->sealed == true
//   ④ 全浮動小数点値が finite（eqMaxGainDb, eqMaxQ, irFreqPeakGainDb）
//   ⑤ sealed = true を設定
//   戻り値: 封印後の BuildAnalysis（sealed=true）。契約違反時はデフォルト構築を返す。
//   ★ v14.37: eqMaxQ, irFreqPeakGainDb を finite チェック対象に追加。
//   additionalAttenuationDb は互換性維持のためチェック対象から除外。
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

    if (!isFiniteFloat(analysis.eqMaxGainDb)
        || !isFiniteFloat(analysis.eqMaxQ)
        || !isFiniteFloat(analysis.irFreqPeakGainDb))
        return BuildAnalysis{};

    analysis.sealed = true;
    return analysis;
}

//==============================================================================
// ★ v14.37: verifyBuildBundle — BuildAnalysis + BuildDiagnostics + OversamplingResult +
//   RuntimeBuildSnapshot の整合性を一括検証（4-object validation）。
//   ISR Authority Singularization: Validator はこの一箇所のみ。
//   ★ v14.46: Facade パターン — 内部的に verifyAnalysis(), verifyDiagnostics(), verifySnapshot() を呼び出す。
//==============================================================================

// 内部ヘルパー: BuildAnalysis 単体検証
[[nodiscard]] inline bool verifyAnalysis(const BuildAnalysis& analysis) noexcept
{
    if (!analysis.sealed)
        return false;
    if (!isFiniteFloat(analysis.eqMaxGainDb)
        || !isFiniteFloat(analysis.eqMaxQ)
        || !isFiniteFloat(analysis.irFreqPeakGainDb))
        return false;
    return true;
}

// 内部ヘルパー: RuntimeBuildSnapshot 単体検証
[[nodiscard]] inline bool verifySnapshot(const RuntimeBuildSnapshot& snapshot) noexcept
{
    if (!snapshot.sealed)
        return false;
    return true;
}

[[nodiscard]] inline bool verifyBuildBundle(
    const BuildAnalysis& analysis,
    const BuildDiagnostics& diagnostics,
    const OversamplingResult& oversampling,
    const RuntimeBuildSnapshot& snapshot) noexcept
{
    if (!verifyAnalysis(analysis))
        return false;
    if (!verifySnapshot(snapshot))
        return false;
    if (analysis.generation != snapshot.generation)
        return false;
    if (!oversampling.isValid())
        return false;
    // ★ v14.12 注: AnalysisPart の analysisVersion 検証は RuntimePublicationOrchestrator 側で実施。
    //   RuntimeBuilder.h をインクルードすると循環依存が発生するため、
    //   verifyBuildBundle() は 4 引数で保持。呼出し側で diag.analysisVersion と
    //   spec.analysis.analysisVersion の整合性を確認済み。

    // Builder collapse 契約の検証
    const float tolerance = AnalysisVersionPolicy::collapseTolerance(diagnostics.analysisVersion);
    const float expectedCollapse = std::max(diagnostics.eqMeasuredGainDb, diagnostics.eqUpperBoundGainDb);
    if (std::abs(analysis.eqMaxGainDb - expectedCollapse) > tolerance)
        return false;

    // selectedEstimate と実際の比較結果の整合性を検証
    if ((diagnostics.selectedEstimate == SelectedEstimate::Measured
            && diagnostics.eqMeasuredGainDb < diagnostics.eqUpperBoundGainDb - 0.01f)
        || (diagnostics.selectedEstimate == SelectedEstimate::UpperBound
            && diagnostics.eqUpperBoundGainDb < diagnostics.eqMeasuredGainDb - 0.01f))
        return false;

    return true;
}

//==============================================================================
// ★ v14.38: verifyDiagnostics — BuildDiagnostics の整合性を検証（Publish 可否とは独立）。
//   ISR 思想: verifyBuildBundle() は Publish 可否のみ、verifyDiagnostics() は Debug 情報の正当性を担当。
//==============================================================================
[[nodiscard]] inline bool verifyDiagnostics(const BuildDiagnostics& diagnostics) noexcept
{
    // finite チェック（診断値の数値健全性）
    if (!isFiniteFloat(diagnostics.eqMeasuredGainDb)
        || !isFiniteFloat(diagnostics.eqMeasuredRawGainDb)
        || !isFiniteFloat(diagnostics.eqUpperBoundGainDb)
        || !isFiniteFloat(diagnostics.eqMeasuredFreqHz)
        || !isFiniteFloat(diagnostics.eqUpperBoundFreqHz)
        || !isFiniteFloat(diagnostics.boundExcessDb)
        || !isFiniteFloat(diagnostics.totalMaxQ))
        return false;
    // eqGainAlgorithm が既知の範囲内か
    if (diagnostics.eqGainAlgorithm != EqGainAlgorithm::TriangleProductV1
        && diagnostics.eqGainAlgorithm != EqGainAlgorithm::Legacy)
        return false;
    // BoundMethod と analysisVersion の整合性
    if (diagnostics.analysisVersion == AnalysisVersionPolicy::kCurrent
        && diagnostics.boundMethod != BoundMethod::TriangleProduct)
        return false;
    if (diagnostics.analysisVersion == AnalysisVersionPolicy::kLegacy
        && diagnostics.boundMethod != BoundMethod::Legacy)
        return false;
    if (diagnostics.boundExcessDb < 0.0f)
        return false;
    return true;
}

// コンパイル時検証
static_assert(std::is_same_v<decltype(RuntimeBuildSnapshot{}.buildInput), BuildInput>,
              "RuntimeBuildSnapshot must use convo::BuildInput as the sole semantic input descriptor.");

static_assert(std::is_trivially_copyable_v<OversamplingResult>,
    "OversamplingResult must be trivially copyable");
static_assert(std::is_trivially_copyable_v<BuildDiagnostics>,
    "BuildDiagnostics must be trivially copyable");

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
