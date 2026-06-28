// PriorityIntegrationTests.cpp
// Phase 3/5 連動テスト: priority escalation + quarantine coordination
//
// テスト内容:
//   1. escalateAllRetires(Critical): 全保留中Intent → Critical 昇格
//   2. dequeuePendingRetireIntents: High優先度が先にdequeueされる
//   3. 複合ソート: epoch昇順 + priority降順
//
// ビルド: カスタム main() + bool testXxx() パターン（既存テストと同一）

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <algorithm>

#include "audioengine/ISRRetire.h"
#include "audioengine/ISRAuthorityClass.h"

namespace {

using convo::isr::RetireIntent;
using convo::isr::RetirePriority;
using convo::isr::RetireRuntime;

// ── ★ Phase5: escalateAllRetires — Critical昇格検証 ──

[[nodiscard]] bool testEscalateAllRetiresToCritical()
{
    RetireRuntime runtime;

    // 3つのIntentを投入（すべて Normal 優先度）
    runtime.emitRetireIntent({1, 100, 1000, true, RetirePriority::Normal});
    runtime.emitRetireIntent({2, 200, 2000, true, RetirePriority::Normal});
    runtime.emitRetireIntent({3, 300, 3000, true, RetirePriority::Normal});

    // ★ escalateAllRetires(Critical): 全intent の優先度を Critical に底上げ
    runtime.escalateAllRetires(RetirePriority::Critical);

    // dequeue → 全件 Critical であることを確認
    auto intents = runtime.dequeuePendingRetireIntents();
    if (intents.empty()) return false;

    for (const auto& intent : intents)
    {
        if (intent.priority != RetirePriority::Critical)
            return false;
    }

    return true;
}

// ── ★ Phase5: escalateAllRetires — 部分昇格（Low→Normal, Highは維持） ──

[[nodiscard]] bool testEscalateAllRetiresPartial()
{
    RetireRuntime runtime;

    runtime.emitRetireIntent({1, 100, 1000, true, RetirePriority::Low});
    runtime.emitRetireIntent({2, 200, 2000, true, RetirePriority::Normal});
    runtime.emitRetireIntent({3, 300, 3000, true, RetirePriority::High});

    // ★ Normal 以上に昇格: Low→Normal, Normal→Normal(維持), High→High(維持)
    runtime.escalateAllRetires(RetirePriority::Normal);

    auto intents = runtime.dequeuePendingRetireIntents();
    // ソート結果: High(3) > Normal(1,2) — 同priority内はepoch昇順
    if (intents.size() != 3) return false;
    // 先頭は High 優先度
    if (intents[0].priority != RetirePriority::High) return false;
    if (intents[0].dspSlot != 3) return false;
    // 残りは Normal
    if (intents[1].priority != RetirePriority::Normal) return false;
    if (intents[2].priority != RetirePriority::Normal) return false;

    return true;
}

// ── ★ Phase3/5連動: quarantineReader → High優先度投入 ──
//   quarantineReader でスタックした Reader を隔離した後、
//   High優先度で retire intent を発行 → dequeue 順序を検証

[[nodiscard]] bool testQuarantineTriggersHighPriority()
{
    RetireRuntime runtime;

    // ★ Phase3: Reader が quarantine された状況を模擬
    //   → Coordinator が quarantine 成功後、High優先度で retire intent を発行
    //   通常の Normal intent と混在させ、High が先に dequeue されることを確認

    // 通常の retire intent (Normal)
    runtime.emitRetireIntent({1, 100, 1000, true, RetirePriority::Normal});
    // quarantine トリガーの retire intent (High)
    runtime.emitRetireIntent({2, 100, 1000, true, RetirePriority::High});
    // 別の通常 intent
    runtime.emitRetireIntent({3, 100, 1000, true, RetirePriority::Normal});

    auto intents = runtime.dequeuePendingRetireIntents();

    // ソート期待: High(2) > Normal(1,3)
    if (intents.size() != 3) return false;
    if (intents[0].priority != RetirePriority::High) return false;
    if (intents[0].dspSlot != 2) return false;
    if (intents[1].priority != RetirePriority::Normal) return false;
    if (intents[2].priority != RetirePriority::Normal) return false;

    return true;
}

// ── ★ Phase5: Shutdown昇格 — Critical優先度で全intent即時処理 ──
//   Shutdown時は escalateAllRetires(Critical) が呼ばれ、
//   全intent が Critical として dequeue される

[[nodiscard]] bool testShutdownEscalation()
{
    RetireRuntime runtime;

    // 様々な優先度の intent を投入
    runtime.emitRetireIntent({1, 100, 3000, true, RetirePriority::Low});
    runtime.emitRetireIntent({2, 200, 2000, true, RetirePriority::Normal});
    runtime.emitRetireIntent({3, 300, 1000, true, RetirePriority::High});

    // ★ Shutdown: 全intent を Critical に昇格
    runtime.escalateAllRetires(RetirePriority::Critical);

    auto intents = runtime.dequeuePendingRetireIntents();

    // 全件 Critical かつ epoch 昇順でソートされている
    if (intents.size() != 3) return false;
    for (const auto& intent : intents)
    {
        if (intent.priority != RetirePriority::Critical)
            return false;
    }

    // 同priority内は epoch 昇順
    if (intents[0].retireEpoch != 1000) return false;  // High(元) → epoch 1000
    if (intents[1].retireEpoch != 2000) return false;  // Normal(元) → epoch 2000
    if (intents[2].retireEpoch != 3000) return false;  // Low(元) → epoch 3000

    return true;
}

// ── ★ Phase5: 優先度別バックログ内訳 ──
//   各優先度のintent数を確認（dequeue 後の統計ではなく、
//   投入段階での優先度分布）

[[nodiscard]] bool testPriorityBacklogBreakdown()
{
    RetireRuntime runtime;

    // 各優先度1つずつ
    runtime.emitRetireIntent({1, 100, 1000, true, RetirePriority::Critical});
    runtime.emitRetireIntent({2, 200, 2000, true, RetirePriority::High});
    runtime.emitRetireIntent({3, 300, 3000, true, RetirePriority::Normal});
    runtime.emitRetireIntent({4, 400, 4000, true, RetirePriority::Low});

    auto intents = runtime.dequeuePendingRetireIntents();

    // 4件全件取得
    if (intents.size() != 4) return false;

    // ソート期待: Critical(1) > High(2) > Normal(3) > Low(4)
    if (intents[0].priority != RetirePriority::Critical) return false;
    if (intents[0].dspSlot != 1) return false;
    if (intents[1].priority != RetirePriority::High) return false;
    if (intents[1].dspSlot != 2) return false;
    if (intents[2].priority != RetirePriority::Normal) return false;
    if (intents[2].dspSlot != 3) return false;
    if (intents[3].priority != RetirePriority::Low) return false;
    if (intents[3].dspSlot != 4) return false;

    return true;
}

} // namespace

int main()
{
    try
    {
        if (!testEscalateAllRetiresToCritical())
            throw std::runtime_error("escalate all retires to critical failed");

        if (!testEscalateAllRetiresPartial())
            throw std::runtime_error("escalate all retires partial failed");

        if (!testQuarantineTriggersHighPriority())
            throw std::runtime_error("quarantine triggers high priority failed");

        if (!testShutdownEscalation())
            throw std::runtime_error("shutdown escalation failed");

        if (!testPriorityBacklogBreakdown())
            throw std::runtime_error("priority backlog breakdown failed");
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }

    return 0;
}
