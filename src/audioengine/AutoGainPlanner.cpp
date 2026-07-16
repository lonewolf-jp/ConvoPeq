#include "AutoGainPlanner.h"
#include <algorithm>
#include <juce_core/juce_core.h>

AutoGainPlan AutoGainPlanner::plan(
    bool autoGainEnabled,
    convo::ProcessingOrder processingOrder,
    bool eqBypassed,
    bool convBypassed,
    float eqMaxGainDb,
    float additionalAttenuationDb) noexcept
{
    AutoGainPlan result {};
    if (!autoGainEnabled)
        return result;  // 0.0/0.0/0.0（= 0dB/0dB/0dB → 線形 1.0/1.0/1.0）

    // ★ v14.0 Phase 8 Review: 両方バイパス時は透過（EQもConvも無効なら自動ゲインも無効）
    if (eqBypassed && convBypassed)
        return result;

    float inputDb = 0.0f, trimDb = 0.0f;

    if (!eqBypassed && convBypassed)
    {
        // PEQ only
        inputDb = -std::max(0.0f, eqMaxGainDb - kMarginEqFirst);
        inputDb -= estimateQSafetyMargin(eqMaxGainDb, processingOrder);
    }
    else if (eqBypassed && !convBypassed)
    {
        // Conv only
        inputDb = -std::max(0.0f, additionalAttenuationDb - kMarginConvFirst);
    }
    else if (processingOrder == convo::ProcessingOrder::ConvolverThenEQ)
    {
        // Conv→PEQ: trim 不適用, input 上限 -6dB
        inputDb = -(std::max(0.0f, additionalAttenuationDb - kMarginConvFirst)
                    + std::max(0.0f, eqMaxGainDb - kMarginInterStage));
        inputDb = std::min(inputDb, kConvFirstInputCeiling);
    }
    else
    {
        // PEQ→Conv: trim 適用（デフォルト: EQThenConvolver）
        inputDb = -std::max(0.0f, eqMaxGainDb - kMarginEqFirst);
        inputDb -= estimateQSafetyMargin(eqMaxGainDb, processingOrder);
        trimDb = -std::max(0.0f, additionalAttenuationDb - kMarginInterStage);
    }

    // クランプ
    result.inputHeadroomDb = juce::jlimit(kClampInputMin, kClampInputMax, inputDb);
    result.convolverInputTrimDb = juce::jlimit(kClampTrimMin, kClampTrimMax, trimDb);

    // ネット 0dB 整合（クランプ後の実効値で makeup 計算）
    const float makeupDb = -result.inputHeadroomDb - result.convolverInputTrimDb;
    result.outputMakeupDb = juce::jlimit(kClampMakeupMin, kClampMakeupMax, makeupDb);

    return result;
}

float AutoGainPlanner::estimateQSafetyMargin(
    float eqMaxGainDb, convo::ProcessingOrder /*processingOrder*/) noexcept
{
    // ★ v14.0 Phase 8 Review: eqMaxGainDb ≦ 0 なら QSurge = 0（不要な減衰防止）
    if (eqMaxGainDb <= 0.0f)
        return 0.0f;

    // 式: Qsurge = min(6.0, 1.5 + peakingSurge)
    // 係数 0.15, 6.0 は経験則ヒューリスティック（Phase 8 要較正）
    constexpr float kQSurgeBase = 1.5f;
    constexpr float kQSurgeCoeff = 0.15f;
    constexpr float kQSurgeMax = 6.0f;
    constexpr float kButterworthQ = 0.707f;

    float peakingSurge = eqMaxGainDb * kQSurgeCoeff * (20.0f / kButterworthQ);  // worst-case Q=20

    return std::min(kQSurgeMax, kQSurgeBase + peakingSurge);
}
