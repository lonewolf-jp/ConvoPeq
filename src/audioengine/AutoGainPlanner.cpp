#include "AutoGainPlanner.h"
#include <algorithm>
#include <juce_core/juce_core.h>

AutoGainPlan AutoGainPlanner::plan(
    bool autoGainEnabled,
    convo::ProcessingOrder processingOrder,
    bool eqBypassed,
    bool convBypassed,
    const PlannerInput& input) noexcept
{
    return plan(autoGainEnabled, processingOrder, eqBypassed, convBypassed, input, nullptr);
}

AutoGainPlan AutoGainPlanner::plan(
    bool autoGainEnabled,
    convo::ProcessingOrder processingOrder,
    bool eqBypassed,
    bool convBypassed,
    const PlannerInput& input,
    PlanDiagnostics* diagnostics) noexcept
{
    AutoGainPlan result {};
    PlanDiagnostics diagLocal {};

    if (diagnostics == nullptr)
        diagnostics = &diagLocal;

    if (!autoGainEnabled)
    {
        if (diagnostics) {
            diagnostics->eqBoost = 0.0f;
            diagnostics->convBoost = 0.0f;
            diagnostics->qMargin = 0.0f;
            diagnostics->clamped = false;
        }
        return result;  // 0.0/0.0/0.0（= 0dB/0dB/0dB → 線形 1.0/1.0/1.0）
    }

    if (eqBypassed && convBypassed)
    {
        if (diagnostics) {
            diagnostics->eqBoost = 0.0f;
            diagnostics->convBoost = 0.0f;
            diagnostics->qMargin = 0.0f;
            diagnostics->clamped = false;
        }
        return result;
    }

    // ★ v14.10: eqMaxGainDb は Builder 側で max(measured, upperBound) 済み
    const float eqBoost   = std::max(0.0f, input.eqMaxGainDb);
    const float convBoost = std::max(0.0f, input.irFreqPeakGainDb);

    float inputDb = 0.0f, trimDb = 0.0f;
    float qMargin = 0.0f;

    if (!eqBypassed && convBypassed)
    {
        // PEQ only
        qMargin = EmpiricalSafetyMarginPolicy::evaluate(input.eqMaxGainDb, input.eqMaxQ);
        inputDb = -std::max(0.0f, eqBoost - kMarginEqFirst) - qMargin;
    }
    else if (eqBypassed && !convBypassed)
    {
        // Conv only
        qMargin = 0.0f;
        inputDb = -std::max(0.0f, convBoost - kMarginConvFirst);
    }
    else if (processingOrder == convo::ProcessingOrder::ConvolverThenEQ)
    {
        // Conv→PEQ: 固定Ceiling廃止、マージンのみで保護
        qMargin = EmpiricalSafetyMarginPolicy::evaluate(input.eqMaxGainDb, input.eqMaxQ);
        inputDb = -(std::max(0.0f, convBoost - kMarginConvFirst)
                  + std::max(0.0f, eqBoost - kMarginInterStage)
                  + qMargin);
    }
    else
    {
        // PEQ→Conv: trim 適用（デフォルト: EQThenConvolver）
        qMargin = EmpiricalSafetyMarginPolicy::evaluate(input.eqMaxGainDb, input.eqMaxQ);
        inputDb = -std::max(0.0f, eqBoost - kMarginEqFirst) - qMargin;
        trimDb  = -std::max(0.0f, convBoost - kMarginInterStage);
    }

    // クランプ
    const float clampedInput = juce::jlimit(kClampInputMin, kClampInputMax, inputDb);
    const float clampedTrim  = juce::jlimit(kClampTrimMin, kClampTrimMax, trimDb);
    result.inputHeadroomDb = clampedInput;
    result.convolverInputTrimDb = clampedTrim;

    // ネット 0dB 整合（クランプ後の実効値で makeup 計算）
    const float rawMakeupDb = -clampedInput - clampedTrim;
    const float clampedMakeup = juce::jlimit(kClampMakeupMin, kClampMakeupMax, rawMakeupDb);
    result.outputMakeupDb = clampedMakeup;

    // ★ v14.36: PlanDiagnostics を設定
    if (diagnostics) {
        diagnostics->eqBoost   = eqBoost;
        diagnostics->convBoost = convBoost;
        diagnostics->qMargin   = qMargin;
        diagnostics->inputClamped  = (clampedInput != inputDb);
        diagnostics->trimClamped   = (clampedTrim != trimDb);
        diagnostics->makeupClamped = (clampedMakeup != rawMakeupDb);
        diagnostics->clamped = diagnostics->inputClamped
                            || diagnostics->trimClamped
                            || diagnostics->makeupClamped;
    }

    return result;
}
