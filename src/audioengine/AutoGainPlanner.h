#pragma once

#include "core/Types.h"
#include "RuntimeBuildTypes.h"

#pragma warning(push)
#pragma warning(disable : 4324) // C4324: キャッシュライン分離用alignasによる意図的なパディングを許容

//==============================================================================
/**
    AutoGainPlanner V2 — 自動ゲインステージングの純粋関数型プランナー（ISR設計）。

    Engine/DSP オブジェクトを一切参照しない。
    入力は PlannerInput DTO のみ。出力は dB 値。
    線形ゲイン変換（decibelsToGain）は RuntimeBuilder 側で行う。

    ISR 思想:
    - Planner は PlannerInput のみを受け取り、BuildAnalysis.Diagnostics を参照不可能
    - eqMaxGainDb は Builder により max(measured, upperBound) で安全側保証済み
    - 責務は「与えられた入力からマージンを計算し、4パターン分岐すること」のみ

    特徴:
    - 4パターン判定（PEQ only / Conv only / Conv→PEQ / PEQ→Conv）
    - クランプ（input: -18〜0dB, trim: -12〜0dB, makeup: 0〜12dB）
    - ネット 0dB 整合
    - 固定 Ceiling 廃止（kConvFirstInputCeiling 削除）
*/

// Margin constants (inline constexpr, C++20)
// ★ v14.3: マージン定数を再設計。固定 Ceiling は一切使用しない。
inline constexpr float kMarginEqFirst      = 1.5f;   // 3.0→1.5
inline constexpr float kMarginConvFirst    = 1.0f;   // 1.5→1.0
inline constexpr float kMarginInterStage   = 1.0f;   // 2.0→1.0
inline constexpr float kSafetyMarginBase   = 0.8f;
inline constexpr float kSafetyMarginCoeffQ = 0.12f;
inline constexpr float kSafetyMarginCoeffGain = 0.04f;
inline constexpr float kSafetyMarginMax    = 2.5f;   // 6.0→2.5
inline constexpr float kClampInputMin      = -18.0f; // -12→-18
inline constexpr float kClampInputMax      = 0.0f;
inline constexpr float kClampTrimMin       = -12.0f;
inline constexpr float kClampTrimMax       = 0.0f;
inline constexpr float kClampMakeupMin     = 0.0f;
inline constexpr float kClampMakeupMax     = 12.0f;
// kConvFirstInputCeiling は削除（固定Ceiling廃止）

//==============================================================================
// ★ v14.14: PlannerInput — Planner 専用 DTO。物理的に Diagnostics へアクセス不可能。
//   ISR 思想: DTO を介して Planner と Builder を完全分離。
//   Planner は解析アルゴリズムを知らない。
//==============================================================================
struct PlannerInput {
    float eqMaxGainDb = 0.0f;          // Builder collapse 後の安全側値
    float eqMaxQ = 0.0f;               // ブースト対象バンド中の最大Q値
    float irFreqPeakGainDb = 0.0f;     // IR 周波数ピークゲイン
};

//==============================================================================
// ★ v14.7: EmpiricalSafetyMarginPolicy — 経験的安全マージン（旧称 QSurge）。
//   Bound ではなく経験式。ISR 思想に基づき Policy として分離。
//   Builder/Planner/Test で共有。
//==============================================================================
struct EmpiricalSafetyMarginPolicy {
    static constexpr float kBase         = kSafetyMarginBase;
    static constexpr float kCoeffQ       = kSafetyMarginCoeffQ;
    static constexpr float kCoeffGain    = kSafetyMarginCoeffGain;
    static constexpr float kMax          = kSafetyMarginMax;
    static constexpr float kButterworthQ = 0.707f;
    static constexpr float kMinimumBoostForMargin = 0.5f;

    [[nodiscard]] static float evaluate(float eqGainDb, float maxQ) noexcept {
        if (eqGainDb <= kMinimumBoostForMargin) return 0.0f;
        const float qTerm = std::max(0.0f, (maxQ - kButterworthQ) * kCoeffQ);
        const float gTerm = eqGainDb * kCoeffGain;
        return std::min(kMax, std::max(0.0f, kBase + qTerm + gTerm));
    }
};

//==============================================================================
// ★ v14.36: PlanDiagnostics — プランナー診断情報
//==============================================================================
struct PlanDiagnostics {
    float qMargin   = 0.0f;
    float eqBoost   = 0.0f;
    float convBoost = 0.0f;
    bool  clamped   = false;
    bool inputClamped  = false;
    bool trimClamped   = false;
    bool makeupClamped = false;
    enum class CombinedEstimate : uint8_t { Sum = 0 };
    CombinedEstimate combinedMethod = CombinedEstimate::Sum;
};

struct AutoGainPlan {
    float inputHeadroomDb = 0.0f;      // dB 値（下限 -18dB）
    float outputMakeupDb = 0.0f;       // dB 値（0..12dB）
    float convolverInputTrimDb = 0.0f; // dB 値（-12..0dB）
};

class AutoGainPlanner {
public:
    [[nodiscard]] static AutoGainPlan plan(
        bool autoGainEnabled,
        convo::ProcessingOrder processingOrder,
        bool eqBypassed,
        bool convBypassed,
        const PlannerInput& input) noexcept;

    // ★ v14.36: diagnostics 出力付き版。PlanDiagnostics に qMargin/eqBoost/convBoost/clamp 情報を格納
    [[nodiscard]] static AutoGainPlan plan(
        bool autoGainEnabled,
        convo::ProcessingOrder processingOrder,
        bool eqBypassed,
        bool convBypassed,
        const PlannerInput& input,
        PlanDiagnostics* diagnostics) noexcept;

    // ★ v14.7: EmpiricalSafetyMarginPolicy に委譲（旧 estimateQSafetyMargin を置換）
    [[nodiscard]] static float estimateQSafetyMargin(
        float eqMaxGainDb, float maxQ) noexcept
    {
        return EmpiricalSafetyMarginPolicy::evaluate(eqMaxGainDb, maxQ);
    }
};

#pragma warning(pop)
