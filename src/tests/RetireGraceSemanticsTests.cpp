#include <algorithm>
#include <stdexcept>
#include <vector>

#include "audioengine/ISRRetire.h"
#include "audioengine/ISRRetireRuntimeEx.h"
#include "audioengine/ISRAuthorityClass.h"
#include "audioengine/ISRRetireOverflowRing.h"

// ── ★ Phase5: 複合ソートキー (priority, retireEpoch, generation, dspSlot) 検証 ──

[[nodiscard]] bool testPrioritySortCompositeKey()
{
    using convo::isr::RetireIntent;
    using convo::isr::RetirePriority;

    // dequeuePendingRetireIntents と同じ comparator
    auto sorter = [](const RetireIntent& lhs, const RetireIntent& rhs) noexcept {
        if (lhs.priority != rhs.priority)
            return lhs.priority > rhs.priority;   // priority降順（Critical最先頭）
        if (lhs.retireEpoch != rhs.retireEpoch)
            return lhs.retireEpoch < rhs.retireEpoch;
        if (lhs.generation != rhs.generation)
            return lhs.generation < rhs.generation;
        return lhs.dspSlot < rhs.dspSlot;
    };

    // 1. Critical > Normal (priority降順)
    {
        const RetireIntent critical{1, 100, 1000, true, RetirePriority::Critical};
        const RetireIntent normal{2, 100, 1000, true, RetirePriority::Normal};
        if (!sorter(critical, normal) || sorter(normal, critical))
            return false;
    }

    // 2. 同priority内: 古いepochが先（FIFO）
    {
        const RetireIntent older{1, 100, 500, true, RetirePriority::Normal};
        const RetireIntent newer{2, 100, 1000, true, RetirePriority::Normal};
        if (!sorter(older, newer) || sorter(newer, older))
            return false;
    }

    // 3. 同priority+epoch: 低いgenerationが先
    {
        const RetireIntent early{1, 50, 1000, true, RetirePriority::Normal};
        const RetireIntent late{2, 100, 1000, true, RetirePriority::Normal};
        if (!sorter(early, late) || sorter(late, early))
            return false;
    }

    // 4. 同priority+epoch+generation: 低いdspSlotが先
    {
        const RetireIntent first{1, 100, 1000, true, RetirePriority::Normal};
        const RetireIntent second{2, 100, 1000, true, RetirePriority::Normal};
        if (!sorter(first, second) || sorter(second, first))
            return false;
    }

    // 5. 完全ソート: Critical > High > Normal > Low
    {
        std::vector<RetireIntent> intents = {
            {3, 100, 1000, true, RetirePriority::Low},
            {1, 100, 1000, true, RetirePriority::Critical},
            {4, 100, 1000, true, RetirePriority::High},
            {2, 100, 1000, true, RetirePriority::Normal},
        };
        std::stable_sort(intents.begin(), intents.end(), sorter);
        if (intents[0].dspSlot != 1 || intents[0].priority != RetirePriority::Critical)
            return false;
        if (intents[1].dspSlot != 4 || intents[1].priority != RetirePriority::High)
            return false;
        if (intents[2].dspSlot != 2 || intents[2].priority != RetirePriority::Normal)
            return false;
        if (intents[3].dspSlot != 3 || intents[3].priority != RetirePriority::Low)
            return false;
    }

    return true;
}

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

// ── ★ Phase1: OverflowRing 基本 FIFO 検証 ──

[[nodiscard]] bool testOverflowRingFifoOrder()
{
    using convo::isr::RetireOverflowEntry;
    using convo::isr::RetireOverflowRing;
    using convo::isr::RetireIntent;
    using convo::isr::RetirePriority;

    RetireOverflowRing ring;

    RetireOverflowEntry e1{{1, 100, 1000, true, RetirePriority::Normal}, 100, 0};
    RetireOverflowEntry e2{{2, 200, 2000, true, RetirePriority::Normal}, 200, 0};
    RetireOverflowEntry e3{{3, 300, 3000, true, RetirePriority::Normal}, 300, 0};

    if (!ring.tryPush(e1)) return false;
    if (!ring.tryPush(e2)) return false;
    if (!ring.tryPush(e3)) return false;
    if (ring.residentCount() != 3) return false;

    RetireOverflowEntry out;
    if (!ring.pop(out) || out.intent.dspSlot != 1) return false;
    if (!ring.pop(out) || out.intent.dspSlot != 2) return false;
    if (!ring.pop(out) || out.intent.dspSlot != 3) return false;
    if (ring.residentCount() != 0) return false;
    if (ring.pop(out)) return false;

    std::vector<RetireOverflowEntry> drained;
    (void)ring.tryPush(e1);
    (void)ring.tryPush(e2);
    ring.drainAll(drained);
    if (drained.size() != 2) return false;

    return true;
}

// ── ★ Phase5: 優先度ソート Critical最優先 + 異種priority混合 ──

[[nodiscard]] bool testPrioritySortCriticalFirst()
{
    using convo::isr::RetireIntent;
    using convo::isr::RetirePriority;

    auto sorter = [](const RetireIntent& lhs, const RetireIntent& rhs) noexcept {
        if (lhs.priority != rhs.priority) return lhs.priority > rhs.priority;
        if (lhs.retireEpoch != rhs.retireEpoch) return lhs.retireEpoch < rhs.retireEpoch;
        if (lhs.generation != rhs.generation) return lhs.generation < rhs.generation;
        return lhs.dspSlot < rhs.dspSlot;
    };

    // Critical vs High vs Normal vs Low (same epoch)
    {
        std::vector<RetireIntent> intents = {
            {1, 100, 1000, true, RetirePriority::Low},
            {2, 100, 1000, true, RetirePriority::High},
            {3, 100, 1000, true, RetirePriority::Critical},
            {4, 100, 1000, true, RetirePriority::Normal},
        };
        std::stable_sort(intents.begin(), intents.end(), sorter);
        if (intents[0].priority != RetirePriority::Critical) return false;
        if (intents[1].priority != RetirePriority::High) return false;
        if (intents[2].priority != RetirePriority::Normal) return false;
        if (intents[3].priority != RetirePriority::Low) return false;
    }

    // Cross-priority with mixed epochs
    {
        std::vector<RetireIntent> intents = {
            {1, 100, 1000, true, RetirePriority::High},
            {2, 100, 3000, true, RetirePriority::Critical},
        };
        std::stable_sort(intents.begin(), intents.end(), sorter);
        if (intents[0].priority != RetirePriority::Critical) return false;
    }

    return true;
}

// ── ★ Phase5: 既存 enqueueRetire 互換性（Normal 優先度として動作）──

[[nodiscard]] bool testRetirePriorityCompatibility()
{
    using convo::isr::RetireIntent;
    using convo::isr::RetirePriority;

    // デフォルト priority が Normal であることを確認
    {
        const RetireIntent intent{1, 100, 1000, true};
        if (intent.priority != RetirePriority::Normal)
            return false;
    }

    // 明示的に Normal を設定
    {
        const RetireIntent intent{1, 100, 1000, true, RetirePriority::Normal};
        if (intent.priority != RetirePriority::Normal)
            return false;
    }

    // ソートで Normal が正しい位置に入る
    {
        auto sorter = [](const RetireIntent& lhs, const RetireIntent& rhs) noexcept {
            if (lhs.priority != rhs.priority) return lhs.priority > rhs.priority;
            return lhs.retireEpoch < rhs.retireEpoch;
        };

        std::vector<RetireIntent> intents = {
            {1, 100, 3000, true},                       // デフォルト Normal
            {2, 100, 1000, true, RetirePriority::High},
            {3, 100, 2000, true, RetirePriority::Normal},
        };
        std::stable_sort(intents.begin(), intents.end(), sorter);
        // High > Normal(2) > Normal(1, default)
        if (intents[0].priority != RetirePriority::High) return false;
        if (intents[1].priority != RetirePriority::Normal) return false;
        if (intents[1].dspSlot != 3) return false;  // epoch 2000 が先
        if (intents[2].priority != RetirePriority::Normal) return false;
        if (intents[2].dspSlot != 1) return false;  // epoch 3000 が後
    }

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

    if (!testPrioritySortCompositeKey())
        throw std::runtime_error("priority sort composite key failed");

    if (!testOverflowRingFifoOrder())
        throw std::runtime_error("overflow ring FIFO order failed");

    if (!testPrioritySortCriticalFirst())
        throw std::runtime_error("priority sort critical first failed");

    if (!testRetirePriorityCompatibility())
        throw std::runtime_error("retire priority compatibility failed");

    return 0;
}
