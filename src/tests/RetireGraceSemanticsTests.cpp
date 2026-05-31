#include <stdexcept>

#include "audioengine/ISRRetireRuntimeEx.h"

[[nodiscard]] bool testGracePeriodCompletionRules()
{
    if (!convo::isr::RetireRuntimeEx::isGracePeriodCompleted(100, 101, 1))
        return false;

    if (!convo::isr::RetireRuntimeEx::isGracePeriodCompleted(100, 100, 0))
        return false;

    if (convo::isr::RetireRuntimeEx::isGracePeriodCompleted(100, 100, 1))
        return false;

    return true;
}

[[nodiscard]] bool testRetirePendingToFreeRules()
{
    if (!convo::isr::RetireRuntimeEx::canTransitionRetirePendingToFree(true, true, true))
        return false;

    if (convo::isr::RetireRuntimeEx::canTransitionRetirePendingToFree(false, true, true))
        return false;

    if (convo::isr::RetireRuntimeEx::canTransitionRetirePendingToFree(true, false, true))
        return false;

    if (convo::isr::RetireRuntimeEx::canTransitionRetirePendingToFree(true, true, false))
        return false;

    return true;
}

[[nodiscard]] bool testRetireStarvationDualThresholdRules()
{
    if (!convo::isr::RetireRuntimeEx::hasExceededDeferralThresholds(101, 10.0, 100, 5000.0))
        return false;

    if (!convo::isr::RetireRuntimeEx::hasExceededDeferralThresholds(10, 5001.0, 100, 5000.0))
        return false;

    if (convo::isr::RetireRuntimeEx::hasExceededDeferralThresholds(100, 5000.0, 100, 5000.0))
        return false;

    return true;
}

[[nodiscard]] bool testRetireEscalationSafetyRules()
{
    if (!convo::isr::RetireRuntimeEx::canReclaimAfterEscalation(true, true, true))
        return false;

    if (convo::isr::RetireRuntimeEx::canReclaimAfterEscalation(false, true, true))
        return false;

    if (convo::isr::RetireRuntimeEx::canReclaimAfterEscalation(true, false, true))
        return false;

    if (convo::isr::RetireRuntimeEx::canReclaimAfterEscalation(true, true, false))
        return false;

    return true;
}

[[nodiscard]] bool testRetirePressureThresholdPolicyRules()
{
    constexpr int kRetirePressureMildPercent = 75;
    constexpr int kRetirePressureMediumPercent = 90;
    constexpr int kRetirePressureSeverePercent = 95;

    auto evaluateLevel = [](std::uint64_t retireDepth, int highWatermark) noexcept {
        const int safeHwm = (highWatermark > 0) ? highWatermark : 1;
        const std::uint64_t ratioPercent = (retireDepth * 100ull) / static_cast<std::uint64_t>(safeHwm);
        if (ratioPercent >= static_cast<std::uint64_t>(kRetirePressureSeverePercent))
            return 3;
        if (ratioPercent >= static_cast<std::uint64_t>(kRetirePressureMediumPercent))
            return 2;
        if (ratioPercent >= static_cast<std::uint64_t>(kRetirePressureMildPercent))
            return 1;
        return 0;
    };

    if (evaluateLevel(74, 100) != 0)
        return false;
    if (evaluateLevel(75, 100) != 1)
        return false;
    if (evaluateLevel(89, 100) != 1)
        return false;
    if (evaluateLevel(90, 100) != 2)
        return false;
    if (evaluateLevel(94, 100) != 2)
        return false;
    if (evaluateLevel(95, 100) != 3)
        return false;

    auto isProtectiveMode = [](int retirePressureLevel, std::uint64_t retireDepth, int highWatermark) noexcept {
        const bool severe = retirePressureLevel >= 3;
        const int safeHwm = (highWatermark > 0) ? highWatermark : 1;
        return severe && (retireDepth >= static_cast<std::uint64_t>(safeHwm));
    };

    if (isProtectiveMode(2, 150, 100))
        return false;
    if (!isProtectiveMode(3, 100, 100))
        return false;
    if (isProtectiveMode(3, 99, 100))
        return false;

    return true;
}

int main()
{
    if (!testGracePeriodCompletionRules())
        throw std::runtime_error("grace period completion rules failed");

    if (!testRetirePendingToFreeRules())
        throw std::runtime_error("retire pending to free rules failed");

    if (!testRetireStarvationDualThresholdRules())
        throw std::runtime_error("retire starvation dual threshold rules failed");

    if (!testRetireEscalationSafetyRules())
        throw std::runtime_error("retire escalation safety rules failed");

    if (!testRetirePressureThresholdPolicyRules())
        throw std::runtime_error("retire pressure threshold policy rules failed");

    return 0;
}
