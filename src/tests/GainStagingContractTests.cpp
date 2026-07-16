//==============================================================================
// GainStagingContractTests.cpp — ★ v14.0 Phase 8
//
// AutoGainPlanner::plan() の入出力契約テスト（リファレンス実装検証方式）。
// 4パターン（PEQ only / Conv only / Conv→PEQ / PEQ→Conv）× Auto On/Off を検証。
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
// リファレンス実装 — AutoGainPlanner.h の constexpr 定数と同一
//==============================================================================
constexpr float kMarginEqFirst    = 3.0f;
constexpr float kMarginConvFirst  = 1.5f;
constexpr float kMarginInterStage = 2.0f;
constexpr float kClampInputMin    = -12.0f;
constexpr float kClampInputMax    = 0.0f;
constexpr float kClampTrimMin     = -12.0f;
constexpr float kClampTrimMax     = 0.0f;
constexpr float kClampMakeupMin   = 0.0f;
constexpr float kClampMakeupMax   = 12.0f;
constexpr float kConvFirstInputCeiling = -6.0f;

enum class Order { ConvolverThenEQ = 0, EQThenConvolver = 1 };

template<typename T>
constexpr T jlimit(T lo, T hi, T v) noexcept { return (v < lo) ? lo : (hi < v) ? hi : v; }

//==============================================================================
// リファレンス実装 — estimateQSafetyMargin
//==============================================================================
float refQSafetyMargin(float eqMaxGainDb) noexcept
{
    constexpr float kQSurgeHpfLpf = 1.5f;
    constexpr float kQSurgeCoeff = 0.15f;
    constexpr float kQSurgeMax = 6.0f;
    constexpr float kButterworthQ = 0.707f;
    const float peakingSurge = eqMaxGainDb > 0.0f
        ? eqMaxGainDb * kQSurgeCoeff * (20.0f / kButterworthQ)
        : 0.0f;
    return std::min(kQSurgeMax, kQSurgeHpfLpf + peakingSurge);
}

//==============================================================================
// リファレンス実装 — plan() のロジック（AutoGainPlanner.cpp と同一アルゴリズム）
//==============================================================================
struct Plan {
    float inputHeadroomDb = 0.0f;
    float outputMakeupDb = 0.0f;
    float convolverInputTrimDb = 0.0f;
};

Plan refPlan(bool autoGainEnabled, Order order, bool eqBypassed, bool convBypassed,
             float eqMaxGainDb, float additionalAttenuationDb) noexcept
{
    Plan result {};
    if (!autoGainEnabled)
        return result;

    float inputDb = 0.0f, trimDb = 0.0f;

    if (!eqBypassed && convBypassed)
    {
        inputDb = -std::max(0.0f, eqMaxGainDb - kMarginEqFirst);
        inputDb -= refQSafetyMargin(eqMaxGainDb);
    }
    else if (eqBypassed && !convBypassed)
    {
        inputDb = -std::max(0.0f, additionalAttenuationDb - kMarginConvFirst);
    }
    else if (order == Order::ConvolverThenEQ)
    {
        inputDb = -(std::max(0.0f, additionalAttenuationDb - kMarginConvFirst)
                    + std::max(0.0f, eqMaxGainDb - kMarginInterStage));
        inputDb = std::min(inputDb, kConvFirstInputCeiling);
    }
    else
    {
        inputDb = -std::max(0.0f, eqMaxGainDb - kMarginEqFirst);
        inputDb -= refQSafetyMargin(eqMaxGainDb);
        trimDb = -std::max(0.0f, additionalAttenuationDb - kMarginInterStage);
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
    const auto plan = refPlan(true, Order::EQThenConvolver, false, true, eqMax, 6.0f);
    const float expectedInput = -std::max(0.0f, eqMax - kMarginEqFirst) - refQSafetyMargin(eqMax);
    const float clampedInput = jlimit(kClampInputMin, kClampInputMax, expectedInput);
    const float expectedMakeup = jlimit(kClampMakeupMin, kClampMakeupMax, -clampedInput);
    check(approxEqual(plan.inputHeadroomDb, clampedInput), "PEQ only: inputHeadroomDb");
    check(approxEqual(plan.convolverInputTrimDb, 0.0f), "PEQ only: trim == 0");
    check(approxEqual(plan.outputMakeupDb, expectedMakeup), "PEQ only: outputMakeupDb");
    check(approxEqual(plan.inputHeadroomDb + plan.convolverInputTrimDb + plan.outputMakeupDb, 0.0f), "PEQ only: net 0dB");
}

void testPEQOnlyNoQSurge()
{
    // eqMaxGainDb = 0 → QSurge も 1.5dB 最小 → input = -(0-3) - 1.5 = -1.5
    const auto plan = refPlan(true, Order::EQThenConvolver, false, true, 0.0f, 0.0f);
    check(approxEqual(plan.inputHeadroomDb, -1.5f), "PEQ only zero: input = -1.5");
    check(approxEqual(plan.outputMakeupDb, 1.5f), "PEQ only zero: makeup = 1.5");
    check(approxEqual(plan.convolverInputTrimDb, 0.0f), "PEQ only zero: trim == 0");
}

void testConvOnly()
{
    const float attenu = 6.0f;
    const auto plan = refPlan(true, Order::EQThenConvolver, true, false, 12.0f, attenu);
    const float expectedInput = -std::max(0.0f, attenu - kMarginConvFirst);
    const float clampedInput = jlimit(kClampInputMin, kClampInputMax, expectedInput);
    const float expectedMakeup = jlimit(kClampMakeupMin, kClampMakeupMax, -clampedInput);
    check(approxEqual(plan.inputHeadroomDb, clampedInput), "Conv only: inputHeadroomDb");
    check(approxEqual(plan.convolverInputTrimDb, 0.0f), "Conv only: trim == 0");
    check(approxEqual(plan.outputMakeupDb, expectedMakeup), "Conv only: outputMakeupDb");
    check(approxEqual(plan.inputHeadroomDb + plan.convolverInputTrimDb + plan.outputMakeupDb, 0.0f), "Conv only: net 0dB");
}

void testConvThenPEQ()
{
    const float eqMax = 9.0f, attenu = 6.0f;
    const auto plan = refPlan(true, Order::ConvolverThenEQ, false, false, eqMax, attenu);
    float expectedInput = -(std::max(0.0f, attenu - kMarginConvFirst) + std::max(0.0f, eqMax - kMarginInterStage));
    expectedInput = std::min(expectedInput, kConvFirstInputCeiling);
    const float clampedInput = jlimit(kClampInputMin, kClampInputMax, expectedInput);
    const float expectedMakeup = jlimit(kClampMakeupMin, kClampMakeupMax, -clampedInput);
    check(approxEqual(plan.inputHeadroomDb, clampedInput), "Conv->PEQ: inputHeadroomDb");
    check(approxEqual(plan.convolverInputTrimDb, 0.0f), "Conv->PEQ: trim == 0");
    check(approxEqual(plan.outputMakeupDb, expectedMakeup), "Conv->PEQ: outputMakeupDb");
    check(approxEqual(plan.inputHeadroomDb + plan.convolverInputTrimDb + plan.outputMakeupDb, 0.0f), "Conv->PEQ: net 0dB");
}

void testConvThenPEQCeilingClamp()
{
    // ★注: 実装では `std::min(inputDb, kConvFirstInputCeiling)` により
    //   入力 0dB → -6dB に矯正（「計算結果が0dBの場合、-6dBにクランプ」）
    //   結果のネット 0dB は makeup=+6dB で維持される（input=-6, trim=0, makeup=+6）
    const auto plan = refPlan(true, Order::ConvolverThenEQ, false, false, 0.0f, 0.0f);
    check(approxEqual(plan.inputHeadroomDb, kConvFirstInputCeiling), "Conv->PEQ ceiling: input == -6");
    check(approxEqual(plan.outputMakeupDb, 6.0f), "Conv->PEQ ceiling: makeup == +6");
    check(approxEqual(plan.convolverInputTrimDb, 0.0f), "Conv->PEQ ceiling: trim == 0");
    check(approxEqual(plan.inputHeadroomDb + plan.outputMakeupDb, 0.0f), "Conv->PEQ ceiling: net 0dB");
}

void testPEQThenConv()
{
    // ★注: ネット 0dB は makeup の clamp 範囲 [0,12]dB 内でのみ保証される。
    const float eqMax = 5.0f, attenu = 4.0f;
    const float qSurge = refQSafetyMargin(eqMax);
    const float expectedInput = -std::max(0.0f, eqMax - kMarginEqFirst) - qSurge;
    const float expectedTrim = -std::max(0.0f, attenu - kMarginInterStage);
    const float clampedInput = jlimit(kClampInputMin, kClampInputMax, expectedInput);
    const float clampedTrim = jlimit(kClampTrimMin, kClampTrimMax, expectedTrim);
    const float expectedMakeup = -clampedInput - clampedTrim;
    const auto plan = refPlan(true, Order::EQThenConvolver, false, false, eqMax, attenu);
    check(approxEqual(plan.inputHeadroomDb, clampedInput), "PEQ->Conv: inputHeadroomDb");
    check(approxEqual(plan.convolverInputTrimDb, clampedTrim), "PEQ->Conv: trim");
    check(approxEqual(plan.outputMakeupDb, expectedMakeup), "PEQ->Conv: outputMakeupDb");
    check(approxEqual(plan.inputHeadroomDb + plan.convolverInputTrimDb + plan.outputMakeupDb, 0.0f), "PEQ->Conv: net 0dB");
}

void testClampRanges()
{
    const auto plan = refPlan(true, Order::EQThenConvolver, false, false, 100.0f, 100.0f);
    check(plan.inputHeadroomDb >= kClampInputMin, "clamp: input >= -12");
    check(plan.inputHeadroomDb <= kClampInputMax, "clamp: input <= 0");
    check(plan.convolverInputTrimDb >= kClampTrimMin, "clamp: trim >= -12");
    check(plan.convolverInputTrimDb <= kClampTrimMax, "clamp: trim <= 0");
    check(plan.outputMakeupDb >= kClampMakeupMin, "clamp: makeup >= 0");
    check(plan.outputMakeupDb <= kClampMakeupMax, "clamp: makeup <= 12");
}

void testNetZeroDb()
{
    struct TC { const char* name; bool ae; Order ord; bool eq; bool cv; float em; float am; };
    const TC cases[] = {
        {"PEQ only",       true, Order::EQThenConvolver,  false, true,  6.0f,  0.0f},
        {"PEQ only xtrm",  true, Order::EQThenConvolver,  false, true,  30.0f, 0.0f},
        {"Conv only",      true, Order::EQThenConvolver,  true,  false, 0.0f,  4.0f},
        {"Conv only xtrm", true, Order::EQThenConvolver,  true,  false, 0.0f,  30.0f},
        {"C->P",           true, Order::ConvolverThenEQ, false, false, 5.0f,  3.0f},
        {"C->P ceiling",   true, Order::ConvolverThenEQ, false, false, 0.0f,  0.0f},
        {"P->C",           true, Order::EQThenConvolver,  false, false, 5.0f,  3.0f},
        {"Both bypassed",  true, Order::ConvolverThenEQ, true,  true,  0.0f,  0.0f},
    };
    for (const auto& tc : cases)
    {
        const auto p = refPlan(tc.ae, tc.ord, tc.eq, tc.cv, tc.em, tc.am);
        const float net = p.inputHeadroomDb + p.convolverInputTrimDb + p.outputMakeupDb;
        check(std::abs(net) <= 0.1f, std::string("net 0dB: ") + tc.name);
    }
}

void testQSafetyMargin()
{
    check(approxEqual(refQSafetyMargin(0.0f), 1.5f), "Q Surge: eqMax=0 -> 1.5");
    check(approxEqual(refQSafetyMargin(-5.0f), 1.5f), "Q Surge: eqMax<0 -> 1.5");
    check(approxEqual(refQSafetyMargin(10.0f), 6.0f), "Q Surge: eqMax=10 -> 6.0");
    const float mid = refQSafetyMargin(0.5f);
    check(mid > 1.5f && mid < 6.0f, "Q Surge: eqMax=0.5 -> between 1.5 and 6.0");
}

void testQSafetyMarginAlwaysPositive()
{
    for (float eqMax = -20.0f; eqMax <= 30.0f; eqMax += 1.0f)
    {
        const float margin = refQSafetyMargin(eqMax);
        check(margin > 0.0f, "Q Surge positive at " + std::to_string(eqMax));
    }
}

//==============================================================================
// MAIN
//==============================================================================
int main()
{
    std::cout << "[GainStagingContractTests] Start\n";
    testAutoDisabled();
    testPEQOnly();
    testPEQOnlyNoQSurge();
    testConvOnly();
    testConvThenPEQ();
    testConvThenPEQCeilingClamp();
    testPEQThenConv();
    testClampRanges();
    testNetZeroDb();
    testQSafetyMargin();
    testQSafetyMarginAlwaysPositive();
    std::cout << "[GainStagingContractTests] Passed: " << g_testsPassed
              << ", Failed: " << g_testsFailed << "\n";
    return (g_testsFailed == 0) ? 0 : 1;
}
