#pragma once

#include "core/Types.h"

#pragma warning(push)
#pragma warning(disable : 4324) // C4324: キャッシュライン分離用alignasによる意図的なパディングを許容

//==============================================================================
/**
    AutoGainPlanner — 自動ゲインステージングの純粋関数型プランナー。

    Engine/DSP オブジェクトを一切参照しない。
    入力は AnalysisPart の値のみ。出力は dB 値。
    線形ゲイン変換（decibelsToGain）は RuntimeBuilder 側で行う。

    特徴:
    - 4パターン判定（PEQ only / Conv only / Conv→PEQ / PEQ→Conv）
    - クランプ（input: -12〜0dB, trim: -12〜0dB, makeup: 0〜12dB）
    - ネット 0dB 整合
    - Q Surge Margin（経験則ヒューリスティック、Phase 8 要較正）
*/

// Margin constants (inline constexpr, C++20)
inline constexpr float kMarginEqFirst    = 3.0f;   // EQ第1段の入力マージン
inline constexpr float kMarginConvFirst  = 1.5f;   // Conv第1段の入力マージン
inline constexpr float kMarginInterStage = 2.0f;   // 第2段保護マージン
inline constexpr float kClampInputMin    = -12.0f; // input 下限
inline constexpr float kClampInputMax    = 0.0f;   // input 上限
inline constexpr float kClampTrimMin     = -12.0f; // trim 下限
inline constexpr float kClampTrimMax     = 0.0f;   // trim 上限
inline constexpr float kClampMakeupMin   = 0.0f;   // makeup 下限
inline constexpr float kClampMakeupMax   = 12.0f;  // makeup 上限
inline constexpr float kConvFirstInputCeiling = -6.0f; // Conv-first 上限（std::min で ceiling。入力 0dB → -6dB にクランプ）

struct AutoGainPlan {
    float inputHeadroomDb = 0.0f;      // dB 値（下限 -12dB）
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
        float eqMaxGainDb,
        float additionalAttenuationDb) noexcept;

    // Q Surge Margin — 経験則ヒューリスティック（Phase 8 要較正）
    //   Peaking フィルタ主対象。Shelf は過剰だが安全側。Notch/AllPass は不要。
    [[nodiscard]] static float estimateQSafetyMargin(
        float eqMaxGainDb, convo::ProcessingOrder processingOrder) noexcept;
};

#pragma warning(pop)
