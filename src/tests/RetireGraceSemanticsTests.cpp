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

int main()
{
    if (!testGracePeriodCompletionRules())
        throw std::runtime_error("grace period completion rules failed");

    if (!testRetirePendingToFreeRules())
        throw std::runtime_error("retire pending to free rules failed");

    return 0;
}
