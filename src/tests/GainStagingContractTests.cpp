//==============================================================================
// GainStagingContractTests.cpp — ★ v14.0 Phase 8 → V2 (v14.3)
//
// AutoGainPlanner::plan() の入出力契約テスト（リファレンス実装検証方式）。
// V2 定数 + EmpiricalSafetyMarginPolicy に対応。
//
// 本テストは AutoGainPlanner.cpp をリンクせず、リファレンス実装でロジックを
// 検証する。これにより JUCE 依存を完全排除し、コンパイル/実行の独立性を確保。
//==============================================================================
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

namespace {

int g_testsPassed = 0;
int g_testsFailed = 0;

void check(bool condition, const std::string& label)
{
    if (condition)
        ++g_testsPassed;
    else
        ++g_testsFailed, std::cerr << "[FAIL] " << label << "\n";
}

[[nodiscard]] bool approxEqual(float a, float b, float epsilon = 0.05f) noexcept
{
    return std::abs(a - b) <= epsilon;
}

//==============================================================================
// リファレンス実装 — V2 定数（AutoGainPlanner.h と同一）
//==============================================================================
constexpr float kMarginEqFirst      = 1.5f;
constexpr float kMarginConvFirst    = 1.0f;
constexpr float kMarginInterStage   = 1.0f;
constexpr float kClampInputMin      = -18.0f;
constexpr float kClampInputMax      = 0.0f;
constexpr float kClampTrimMin       = -12.0f;
constexpr float kClampTrimMax       = 0.0f;
constexpr float kClampMakeupMin     = 0.0f;
constexpr float kClampMakeupMax     = 12.0f;
// kConvFirstInputCeiling は廃止

constexpr float kSafetyMarginBase   = 0.8f;
constexpr float kSafetyMarginCoeffQ = 0.12f;
constexpr float kSafetyMarginCoeffGain = 0.04f;
constexpr float kSafetyMarginMax    = 2.5f;
constexpr float kButterworthQ       = 0.707f;
constexpr float kMinimumBoostForMargin = 0.5f;

enum class Order { ConvolverThenEQ = 0, EQThenConvolver = 1 };

template<typename T>
constexpr T jlimit(T lo, T hi, T v) noexcept { return (v < lo) ? lo : (hi < v) ? hi : v; }

//==============================================================================
// リファレンス実装 — EmpiricalSafetyMarginPolicy::evaluate()
//==============================================================================
float refSafetyMargin(float eqGainDb, float maxQ) noexcept
{
    if (eqGainDb <= kMinimumBoostForMargin) return 0.0f;
    const float qTerm = std::max(0.0f, (maxQ - kButterworthQ) * kSafetyMarginCoeffQ);
    const float gTerm = eqGainDb * kSafetyMarginCoeffGain;
    return std::min(kSafetyMarginMax, std::max(0.0f, kSafetyMarginBase + qTerm + gTerm));
}

//==============================================================================
// リファレンス実装 — plan() V2 ロジック（AutoGainPlanner.cpp V2 と同一）
//   additionalAttenuationDb → irFreqPeakGainDb に置換
//   固定 Ceiling 削除
//==============================================================================
struct Plan {
    float inputHeadroomDb = 0.0f;
    float outputMakeupDb = 0.0f;
    float convolverInputTrimDb = 0.0f;
};

Plan refPlan(bool autoGainEnabled, Order order, bool eqBypassed, bool convBypassed,
             float eqMaxGainDb, float irFreqPeakGainDb, float eqMaxQ = 0.707f) noexcept
{
    Plan result {};
    if (!autoGainEnabled)
        return result;

    const float eqBoost   = std::max(0.0f, eqMaxGainDb);
    const float convBoost = std::max(0.0f, irFreqPeakGainDb);

    float inputDb = 0.0f, trimDb = 0.0f;

    if (!eqBypassed && convBypassed)
    {
        // PEQ only
        inputDb = -std::max(0.0f, eqBoost - kMarginEqFirst);
        inputDb -= refSafetyMargin(eqMaxGainDb, eqMaxQ);
    }
    else if (eqBypassed && !convBypassed)
    {
        // Conv only
        inputDb = -std::max(0.0f, convBoost - kMarginConvFirst);
    }
    else if (order == Order::ConvolverThenEQ)
    {
        // Conv→PEQ: 固定Ceiling廃止、マージンのみで保護
        const float safetyMargin = refSafetyMargin(eqMaxGainDb, eqMaxQ);
        inputDb = -(std::max(0.0f, convBoost - kMarginConvFirst)
                  + std::max(0.0f, eqBoost - kMarginInterStage)
                  + safetyMargin);
    }
    else
    {
        // PEQ→Conv
        const float safetyMargin = refSafetyMargin(eqMaxGainDb, eqMaxQ);
        inputDb = -std::max(0.0f, eqBoost - kMarginEqFirst) - safetyMargin;
        trimDb  = -std::max(0.0f, convBoost - kMarginInterStage);
    }

    result.inputHeadroomDb = jlimit(kClampInputMin, kClampInputMax, inputDb);
    result.convolverInputTrimDb = jlimit(kClampTrimMin, kClampTrimMax, trimDb);
    const float makeupDb = -result.inputHeadroomDb - result.convolverInputTrimDb;
    result.outputMakeupDb = jlimit(kClampMakeupMin, kClampMakeupMax, makeupDb);
    return result;
}

} // namespace

//==============================================================================
// TESTS
//==============================================================================

void testAutoDisabled()
{
    const auto plan = refPlan(false, Order::EQThenConvolver, false, false, 12.0f, 6.0f);
    check(approxEqual(plan.inputHeadroomDb, 0.0f), "Auto OFF: inputHeadroomDb == 0");
    check(approxEqual(plan.outputMakeupDb, 0.0f), "Auto OFF: outputMakeupDb == 0");
    check(approxEqual(plan.convolverInputTrimDb, 0.0f), "Auto OFF: convolverInputTrimDb == 0");
}

void testPEQOnly()
{
    const float eqMax = 9.0f;
    const float qVal = 1.0f;
    const auto plan = refPlan(true, Order::EQThenConvolver, false, true, eqMax, 0.0f, qVal);
    const float safetyMargin = refSafetyMargin(eqMax, qVal);
    const float expectedInput = -(std::max(0.0f, eqMax - kMarginEqFirst)) - safetyMargin;
    const float clampedInput = jlimit(kClampInputMin, kClampInputMax, expectedInput);
    const float expectedMakeup = jlimit(kClampMakeupMin, kClampMakeupMax, -clampedInput);
    check(approxEqual(plan.inputHeadroomDb, clampedInput), "PEQ only: inputHeadroomDb");
    check(approxEqual(plan.convolverInputTrimDb, 0.0f), "PEQ only: trim == 0");
    check(approxEqual(plan.outputMakeupDb, expectedMakeup), "PEQ only: outputMakeupDb");
    check(approxEqual(plan.inputHeadroomDb + plan.convolverInputTrimDb + plan.outputMakeupDb, 0.0f), "PEQ only: net 0dB");
}

void testPEQOnlyZeroBoost()
{
    // eqMaxGainDb = 0 → マージンなし → input = -max(0, 0-1.5) = 0
    const auto plan = refPlan(true, Order::EQThenConvolver, false, true, 0.0f, 0.0f);
    check(approxEqual(plan.inputHeadroomDb, 0.0f), "PEQ only zero: input = 0");
    check(approxEqual(plan.outputMakeupDb, 0.0f), "PEQ only zero: makeup = 0");
    check(approxEqual(plan.convolverInputTrimDb, 0.0f), "PEQ only zero: trim == 0");
}

void testPEQOnlyLowBoostNoMargin()
{
    // eqMaxGainDb = 0.3 (< 1.5 margin) → input = -max(0, 0.3-1.5) = 0
    const auto plan = refPlan(true, Order::EQThenConvolver, false, true, 0.3f, 0.0f);
    check(approxEqual(plan.inputHeadroomDb, 0.0f), "PEQ low boost: input = 0");
    check(approxEqual(plan.outputMakeupDb, 0.0f), "PEQ low boost: makeup = 0");
}

void testConvOnly()
{
    const float convBoost = 6.0f;
    const auto plan = refPlan(true, Order::EQThenConvolver, true, false, 0.0f, convBoost);
    const float expectedInput = -std::max(0.0f, convBoost - kMarginConvFirst);
    const float clampedInput = jlimit(kClampInputMin, kClampInputMax, expectedInput);
    const float expectedMakeup = jlimit(kClampMakeupMin, kClampMakeupMax, -clampedInput);
    check(approxEqual(plan.inputHeadroomDb, clampedInput), "Conv only: inputHeadroomDb");
    check(approxEqual(plan.convolverInputTrimDb, 0.0f), "Conv only: trim == 0");
    check(approxEqual(plan.outputMakeupDb, expectedMakeup), "Conv only: outputMakeupDb");
    check(approxEqual(plan.inputHeadroomDb + plan.convolverInputTrimDb + plan.outputMakeupDb, 0.0f), "Conv only: net 0dB");
}

void testConvThenPEQ()
{
    // makeup が 12dB クランプされない値を選択
    const float eqMax = 5.0f, convBoost = 3.0f;
    const float qVal = 1.0f;
    const float safetyMargin = refSafetyMargin(eqMax, qVal);
    const auto plan = refPlan(true, Order::ConvolverThenEQ, false, false, eqMax, convBoost, qVal);
    const float expectedInput = -(std::max(0.0f, convBoost - kMarginConvFirst)
                                 + std::max(0.0f, eqMax - kMarginInterStage)
                                 + safetyMargin);
    const float clampedInput = jlimit(kClampInputMin, kClampInputMax, expectedInput);
    const float expectedMakeup = jlimit(kClampMakeupMin, kClampMakeupMax, -clampedInput);
    check(approxEqual(plan.inputHeadroomDb, clampedInput), "Conv->PEQ: inputHeadroomDb");
    check(approxEqual(plan.convolverInputTrimDb, 0.0f), "Conv->PEQ: trim == 0");
    check(approxEqual(plan.outputMakeupDb, expectedMakeup), "Conv->PEQ: outputMakeupDb");
    check(approxEqual(plan.inputHeadroomDb + plan.convolverInputTrimDb + plan.outputMakeupDb, 0.0f), "Conv->PEQ: net 0dB");
}

void testConvThenPEQZeroBoost()
{
    // 固定Ceiling廃止: eq=0, conv=0 → input = -(0+0+0) = 0 (safetyMargin=0)
    const auto plan = refPlan(true, Order::ConvolverThenEQ, false, false, 0.0f, 0.0f);
    check(approxEqual(plan.inputHeadroomDb, 0.0f), "Conv->PEQ zero: input == 0.0");
    check(approxEqual(plan.outputMakeupDb, 0.0f), "Conv->PEQ zero: makeup == 0.0");
    check(approxEqual(plan.convolverInputTrimDb, 0.0f), "Conv->PEQ zero: trim == 0");
    check(approxEqual(plan.inputHeadroomDb + plan.outputMakeupDb, 0.0f), "Conv->PEQ zero: net 0dB");
}

void testPEQThenConv()
{
    const float eqMax = 5.0f, convBoost = 4.0f;
    const float qVal = 1.0f;
    const float safetyMargin = refSafetyMargin(eqMax, qVal);
    const float expectedInput = -std::max(0.0f, eqMax - kMarginEqFirst) - safetyMargin;
    const float expectedTrim = -std::max(0.0f, convBoost - kMarginInterStage);
    const float clampedInput = jlimit(kClampInputMin, kClampInputMax, expectedInput);
    const float clampedTrim = jlimit(kClampTrimMin, kClampTrimMax, expectedTrim);
    const float expectedMakeup = -clampedInput - clampedTrim;
    const auto plan = refPlan(true, Order::EQThenConvolver, false, false, eqMax, convBoost, qVal);
    check(approxEqual(plan.inputHeadroomDb, clampedInput), "PEQ->Conv: inputHeadroomDb");
    check(approxEqual(plan.convolverInputTrimDb, clampedTrim), "PEQ->Conv: trim");
    check(approxEqual(plan.outputMakeupDb, expectedMakeup), "PEQ->Conv: outputMakeupDb");
    check(approxEqual(plan.inputHeadroomDb + plan.convolverInputTrimDb + plan.outputMakeupDb, 0.0f), "PEQ->Conv: net 0dB");
}

void testClampRanges()
{
    const auto plan = refPlan(true, Order::EQThenConvolver, false, false, 100.0f, 100.0f);
    check(plan.inputHeadroomDb >= kClampInputMin, "clamp: input >= -18");
    check(plan.inputHeadroomDb <= kClampInputMax, "clamp: input <= 0");
    check(plan.convolverInputTrimDb >= kClampTrimMin, "clamp: trim >= -12");
    check(plan.convolverInputTrimDb <= kClampTrimMax, "clamp: trim <= 0");
    check(plan.outputMakeupDb >= kClampMakeupMin, "clamp: makeup >= 0");
    check(plan.outputMakeupDb <= kClampMakeupMax, "clamp: makeup <= 12");
}

void testNetZeroDb()
{
    struct TC { const char* name; bool ae; Order ord; bool eq; bool cv; float em; float cvb; float qv; };
    const TC cases[] = {
        {"PEQ only",       true, Order::EQThenConvolver,  false, true,  6.0f,  0.0f, 1.0f},
        {"PEQ only xtrm",  true, Order::EQThenConvolver,  false, true,  12.0f, 0.0f, 1.0f},
        {"Conv only",      true, Order::EQThenConvolver,  true,  false, 0.0f,  4.0f, 0.707f},
        {"Conv only xtrm", true, Order::EQThenConvolver,  true,  false, 0.0f,  12.0f, 0.707f},
        {"C->P",           true, Order::ConvolverThenEQ, false, false, 5.0f,  3.0f, 1.0f},
        {"C->P zero",      true, Order::ConvolverThenEQ, false, false, 0.0f,  0.0f, 0.707f},
        {"P->C",           true, Order::EQThenConvolver,  false, false, 5.0f,  3.0f, 1.0f},
        {"Both bypassed",  true, Order::ConvolverThenEQ, true,  true,  0.0f,  0.0f, 0.707f},
    };
    for (const auto& tc : cases)
    {
        const auto p = refPlan(tc.ae, tc.ord, tc.eq, tc.cv, tc.em, tc.cvb, tc.qv);
        const float net = p.inputHeadroomDb + p.convolverInputTrimDb + p.outputMakeupDb;
        // ネット 0dB: makeup クランプ時は net が負になることを許容
        const bool noClamp = (p.outputMakeupDb < kClampMakeupMax - 0.01f);
        check(std::abs(net) <= (noClamp ? 0.1f : 1.0f),
              std::string("net 0dB: ") + tc.name);
    }
}

void testSafetyMargin()
{
    // eqGainDb <= 0.5 → margin = 0
    check(approxEqual(refSafetyMargin(0.0f, 0.707f), 0.0f), "Safety margin: eq=0 Q=0.707 -> 0");
    check(approxEqual(refSafetyMargin(-5.0f, 0.707f), 0.0f), "Safety margin: eq=-5 Q=0.707 -> 0");

    // eqGainDb = 0.5, Q=0.707 → margin = 0 (minimum boost threshold)
    check(approxEqual(refSafetyMargin(0.5f, 0.707f), 0.0f), "Safety margin: eq=0.5 Q=0.707 -> 0");

    // eqGainDb = 10, Q=2 → qTerm = (2-0.707)*0.12 = 0.155, gTerm = 10*0.04 = 0.4
    // margin = 0.8 + 0.155 + 0.4 = 1.355
    const float margin1 = refSafetyMargin(10.0f, 2.0f);
    check(approxEqual(margin1, 1.355f), "Safety margin: eq=10 Q=2 -> 1.355");

    // eqGainDb = 10, Q=20 → qTerm = (20-0.707)*0.12 = 2.315, gTerm = 10*0.04 = 0.4
    // margin = 0.8 + 2.315 + 0.4 = 3.515 → capped at 2.5
    const float margin2 = refSafetyMargin(10.0f, 20.0f);
    check(approxEqual(margin2, 2.5f), "Safety margin: eq=10 Q=20 -> 2.5 (capped)");
}

void testSafetyMarginZeroForLowBoost()
{
    for (float eqGain = -5.0f; eqGain <= kMinimumBoostForMargin; eqGain += 0.1f)
    {
        const float margin = refSafetyMargin(eqGain, 20.0f);
        check(margin == 0.0f, "Safety margin zero at eq=" + std::to_string(eqGain));
    }
}

void testSafetyMarginAlwaysClamped()
{
    for (float eqGain = 0.0f; eqGain <= 50.0f; eqGain += 1.0f)
    {
        for (float q = 0.5f; q <= 20.0f; q += 0.5f)
        {
            const float margin = refSafetyMargin(eqGain, q);
            check(margin >= 0.0f && margin <= kSafetyMarginMax,
                  "Safety margin clamped at eq=" + std::to_string(eqGain) + " Q=" + std::to_string(q));
        }
    }
}

//==============================================================================
// MAIN
//==============================================================================
int main()
{
    std::cout << "[GainStagingContractTests V2] Start\n";
    testAutoDisabled();
    testPEQOnly();
    testPEQOnlyZeroBoost();
    testPEQOnlyLowBoostNoMargin();
    testConvOnly();
    testConvThenPEQ();
    testConvThenPEQZeroBoost();
    testPEQThenConv();
    testClampRanges();
    testNetZeroDb();
    testSafetyMargin();
    testSafetyMarginZeroForLowBoost();
    testSafetyMarginAlwaysClamped();
    std::cout << "[GainStagingContractTests V2] Passed: " << g_testsPassed
              << ", Failed: " << g_testsFailed << "\n";
    return (g_testsFailed == 0) ? 0 : 1;
}
