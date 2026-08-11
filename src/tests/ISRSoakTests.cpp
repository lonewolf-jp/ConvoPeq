//==============================================================================
// ISRSoakTests.cpp
//
// Work91 — Soak Test: データ構造耐久（ヘッドレス）
//
// publish を含むテストはすべて AudioEngineHarness 側に配置する原則（work91
// soak-test-design.md §2.2）に基づき、本テストは AudioEngine に依存しない
// データ構造レベルの耐久試験のみを担う:
//
//   S2a: IntentQueue 飽和と明示拒否（backpressure, B3 #4）
//   OwnerChannel: enqueue / take の連続サイクル耐久 + 容量満杯拒否
//   PendingPublishRegistry: register / lookup / unregister stress
//
// ヘッドレス駆動: convo::isr::RuntimeIntentCoordinator / RuntimeWorldAuthority
// を AudioEngine 無しで直接使用（ISRSemanticValidationTests と同じ方式）。
//
// ビルド: ISRSemanticValidationTests と同じ include/link パターン。
//   add_executable(ISRSoakTests src/tests/ISRSoakTests.cpp ...)
//   add_test(NAME ISRSoakTests COMMAND ISRSoakTests)
//==============================================================================

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "audioengine/ISRClosure.h"
#include "audioengine/ISRPayloadTier.h"
#include "audioengine/ISRRuntimePublicationCoordinator.h"
#include "audioengine/ISRRuntimeWorldAuthority.h"
#include "audioengine/ISRDSPHandle.h"
#include "audioengine/ISRDSPQuarantine.h"
#include "AudioEngine.h"
#include "OwnerChannel.h"

using convo::isr::RuntimeIntentCoordinator;

namespace {

//------------------------------------------------------------------------------
// テスト補助
//------------------------------------------------------------------------------
int g_testCount = 0;
int g_failCount = 0;

void testPass(const char* name)
{
    std::printf("  [PASS] %s\n", name);
    ++g_testCount;
}

void testFail(const char* name, const char* detail)
{
    std::printf("  [FAIL] %s — %s\n", name, detail);
    ++g_testCount;
    ++g_failCount;
}

//------------------------------------------------------------------------------
// S2a: IntentQueue 飽和と明示拒否（backpressure, B3 #4）
//
//   publish intent を 4096 件（kIntentQueueCapacity）enqueue して満杯にし、
//   次の enqueue が false（明示拒否）を返すことを確認。drain しないため
//   キュー内容は保全される（enqueue 成功数が 4096 のまま不変）。
//   満杯 → 拒否のサイクルを 50 回繰り返す。
//
//   ★注: enqueuePublicationIntent は pendingIntentCount_ を更新しない
//   （submitObserve / submitQuarantine のみ更新）。そのため満杯判定は
//   enqueue 返り値（true/false）と成功数で検証する（既存の
//   testPublishIntentQueueFullBackpressure と同じ方式）。
//------------------------------------------------------------------------------
[[nodiscard]] bool testIntentQueueSaturation()
{
    constexpr std::size_t kCapacity = 4096;   // kIntentQueueCapacity (FUTURE-10 common queue)
    constexpr int kCycles = 50;

    for (int cycle = 0; cycle < kCycles; ++cycle)
    {
        // Coordinator は約 953KB（intentQueue_ 4096×144B 等の巨大メンバ）のため
        // 1MB スタックでオーバーフローする → ヒープ確保が必須。
        auto coordinator = std::make_unique<RuntimeIntentCoordinator>();

        RuntimeIntentCoordinator::Intent intent{};
        intent.type = RuntimeIntentCoordinator::IntentType::Publish;

        std::size_t accepted = 0;
        for (std::size_t i = 0; i < kCapacity + 1; ++i)
        {
            intent.sequenceId = static_cast<std::uint64_t>(cycle * 100000 + i + 1);
            if (coordinator->enqueuePublicationIntent(intent))
                ++accepted;
        }
        if (accepted != kCapacity)
        {
            char buf[128];
            std::snprintf(buf, sizeof(buf), "cycle %d: accepted %zu != capacity %zu",
                          cycle, accepted, kCapacity);
            testFail("S2a: enqueue-to-capacity", buf);
            return false;
        }
        if (coordinator->enqueuePublicationIntent(intent))
        {
            char buf[96];
            std::snprintf(buf, sizeof(buf), "cycle %d: overflow enqueue was NOT rejected", cycle);
            testFail("S2a: backpressure explicit rejection", buf);
            return false;
        }
        // 満杯後も enqueue は引き続き拒否される（キュー内容が保全されている）
        for (std::size_t j = 0; j < 16; ++j)
        {
            if (coordinator->enqueuePublicationIntent(intent))
            {
                char buf[96];
                std::snprintf(buf, sizeof(buf), "cycle %d: queue reopened unexpectedly", cycle);
                testFail("S2a: queue stays saturated", buf);
                return false;
            }
        }
    }

    testPass("S2a: IntentQueue saturation + explicit rejection (50 cycles)");
    return true;
}

//------------------------------------------------------------------------------
// X5: Publish Intent residency 専用 counter（dash §6.5 / INV-X5-1）
//
//   publicationIntentResidencyCount_ は enqueuePublicationIntent の reservation→push→rollback
//   で維持される:
//     - enqueue 成功ごとに +1（getPublicationIntentResidencyCount == accepted 数）
//     - queue full で enqueue 拒否時は rollback（counter 不変）
//     - Publish は pendingIntentCount_ を触らない（独立 counter）
//   注: pop 側（processIntent の Publish fetchSub）は AudioEngine 統合パスが必要なため
//   AudioEngineHarness の統合テストで検証する（dash §6.8 X5 テスト基盤）。
//------------------------------------------------------------------------------
[[nodiscard]] bool testPublicationIntentResidencyCounter()
{
    constexpr std::size_t kCapacity = 4096;   // kIntentQueueCapacity

    auto coordinator = std::make_unique<RuntimeIntentCoordinator>();

    RuntimeIntentCoordinator::Intent intent{};
    intent.type = RuntimeIntentCoordinator::IntentType::Publish;

    // enqueue 成功ごとに +1
    for (std::size_t i = 0; i < kCapacity; ++i)
    {
        intent.sequenceId = static_cast<std::uint64_t>(i + 1);
        if (!coordinator->enqueuePublicationIntent(intent))
        {
            testFail("X5: enqueue up to capacity", "unexpected rejection before full");
            return false;
        }
        const auto resid = coordinator->getPublicationIntentResidencyCount();
        if (resid != i + 1)
        {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                          "after %zu enqueues: residency %llu != %zu",
                          i + 1, static_cast<unsigned long long>(resid), i + 1);
            testFail("X5: residency increments on enqueue", buf);
            return false;
        }
    }

    // queue full: enqueue 拒否 → rollback（counter 不変）
    intent.sequenceId = static_cast<std::uint64_t>(kCapacity + 1);
    if (coordinator->enqueuePublicationIntent(intent))
    {
        testFail("X5: overflow enqueue rejected", "was accepted");
        return false;
    }
    if (coordinator->getPublicationIntentResidencyCount() != kCapacity)
    {
        testFail("X5: overflow rollback", "counter changed on rejected enqueue");
        return false;
    }

    // Publish は pendingIntentCount_ を触らない（独立 counter — INV-X5-1 / P2-1 §1.1.6 W2）
    if (coordinator->getPendingIntentCount() != 0)
    {
        testFail("X5: Publish does not touch pendingIntentCount",
                 "pendingIntentCount changed by Publish enqueue");
        return false;
    }

    testPass("X5: Publish intent residency counter (enqueue +1 / full rollback / independent)");
    return true;
}

//------------------------------------------------------------------------------
// X6: Quarantine Intent / Ring residency counter 遷移（dash §6.6 / INV-X6-4）
//
//   submitQuarantine の transport residency を検証する:
//     - primary（intentQueue_, 4096）成功: quarantineIntentResidencyCount_++
//     - fallback（quarantineFallbackQueue_, 1024）へ移動: intent -1 → ring +1
//     - 全段失敗: rollback（pending 相殺）+ quarantineFallbackDropCount++
//     - 期待: 5120 件成功後 intent=3072 / ring=1024 / pending=5120。
//             次の submit は drop（counter 不変）。
//   DSPHandleRuntime / DSPQuarantineManager は submitQuarantine が (void) するため
//   ダミー実体を渡す。pop 側（processIntent）の減算は AudioEngine 統合で検証する。
//------------------------------------------------------------------------------
[[nodiscard]] bool testQuarantineIntentRingResidency()
{
    constexpr std::size_t kIntentCap = 4096;    // kIntentQueueCapacity
    constexpr std::size_t kFallbackCap = 1024;  // kQuarantineFallbackCapacity

    auto coordinator = std::make_unique<RuntimeIntentCoordinator>();
    convo::isr::DSPHandleRuntime handleRuntime;
    convo::isr::DSPQuarantineManager quarantineManager;

    // primary + fallback を両方満杯にする
    for (std::size_t i = 0; i < kIntentCap + kFallbackCap; ++i)
    {
        convo::isr::DSPHandle handle{
            static_cast<std::uint32_t>((i % 255u) + 1u), static_cast<std::uint64_t>(1u) };
        coordinator->submitQuarantine(handle,
            convo::isr::QuarantineReason::GenerationMismatch,
            handleRuntime, quarantineManager, static_cast<std::uint64_t>(0u));
    }

    // 期待値: intent = 4096（primary 満杯 — 各 submit が +1、fallback 移動ごとに -1 のため
    //   primary 占有数は 4096 に維持）, ring = 1024（fallback 満杯）, pending = 5120（全 admitted）
    //   不変条件: intent + ring == pending（各 quarantine intent は primary か fallback の
    //   どちらか一方にのみ存在 — INV-X6-4 / §6.6 状態遷移表）。
    const auto intentRes = coordinator->getQuarantineIntentResidencyCount();
    const auto ringRes = coordinator->getQuarantineRingResidencyCount();
    const auto pending = coordinator->getPendingIntentCount();
    if (intentRes != kIntentCap || ringRes != kFallbackCap || pending != (kIntentCap + kFallbackCap)
        || intentRes + ringRes != pending)
    {
        char buf[192];
        std::snprintf(buf, sizeof(buf),
                      "intent=%llu ring=%llu pending=%llu (expect intent=%zu ring=%zu pending=%zu)",
                      static_cast<unsigned long long>(intentRes),
                      static_cast<unsigned long long>(ringRes),
                      static_cast<unsigned long long>(pending),
                      kIntentCap, kFallbackCap, kIntentCap + kFallbackCap);
        testFail("X6: quarantine intent/ring residency after saturation", buf);
        return false;
    }

    // 全段失敗: rollback + drop（counter 不変・underflow なし — 二重減算の回帰検出）
    {
        convo::isr::DSPHandle handle{ 255u, 1u };
        coordinator->submitQuarantine(handle,
            convo::isr::QuarantineReason::GenerationMismatch,
            handleRuntime, quarantineManager, static_cast<std::uint64_t>(0u));
    }
    if (coordinator->getQuarantineIntentResidencyCount() != kIntentCap
        || coordinator->getQuarantineRingResidencyCount() != kFallbackCap
        || coordinator->getPendingIntentCount() != (kIntentCap + kFallbackCap))
    {
        testFail("X6: overflow rollback (all-layers full)", "counter changed on dropped submit");
        return false;
    }
    if (coordinator->quarantineFallbackDropCount() != 1)
    {
        testFail("X6: drop counter on all-layers full", "quarantineFallbackDropCount != 1");
        return false;
    }

    testPass("X6: quarantine intent/ring residency transitions (primary/fallback/full-drop)");
    return true;
}

//------------------------------------------------------------------------------
// OwnerChannel 耐久: enqueue / take の連続サイクル
//
//   convo::isr::OwnerChannel<convo::aligned_unique_ptr<const RuntimeState>>
//   をヘッドレスで駆動。50,000 サイクルで:
//     - enqueue → take で所有権が一往復すること
//     - take 後のスロットが確実に空くこと（2回目の take で nullptr）
//     - 同一 key の再 enqueue が拒否されること（no-overwrite）
//------------------------------------------------------------------------------
[[nodiscard]] bool testOwnerChannelEndurance()
{
    using Owner = convo::aligned_unique_ptr<const RuntimeState>;
    using Channel = convo::isr::OwnerChannel<Owner>;

    constexpr int kCycles = 50000;
    Channel channel;

    for (int i = 1; i <= kCycles; ++i)
    {
        auto world = RuntimeState::createForBuilder(RuntimeState::BuilderToken{});
        if (!world)
        {
            testFail("OwnerChannel endurance", "createForBuilder returned null");
            return false;
        }

        convo::isr::OwnerChannelKey key{
            static_cast<std::uint64_t>(i),
            static_cast<std::uint32_t>(i % 1000),
            static_cast<std::uint64_t>(i / 1000)
        };

        if (!channel.enqueue(key, std::move(world)))
        {
            char buf[96];
            std::snprintf(buf, sizeof(buf), "cycle %d: enqueue failed", i);
            testFail("OwnerChannel endurance: enqueue", buf);
            return false;
        }

        auto taken = channel.take(key);
        if (!taken)
        {
            char buf[96];
            std::snprintf(buf, sizeof(buf), "cycle %d: take returned null", i);
            testFail("OwnerChannel endurance: take", buf);
            return false;
        }
        if (channel.take(key))
        {
            char buf[96];
            std::snprintf(buf, sizeof(buf), "cycle %d: 2nd take not empty", i);
            testFail("OwnerChannel endurance: single-transfer", buf);
            return false;
        }
        if (channel.size() != 0)
        {
            char buf[96];
            std::snprintf(buf, sizeof(buf), "cycle %d: size %zu != 0", i, channel.size());
            testFail("OwnerChannel endurance: slot drained", buf);
            return false;
        }
    }

    testPass("OwnerChannel endurance: enqueue/take 50,000 cycles");
    return true;
}

//------------------------------------------------------------------------------
// OwnerChannel 容量満杯拒否
//
//   kCapacity = 256 を超える量の異なる key を enqueue し、満杯時に false を
//   返す（明示拒否、オーバーフローなし）。enqueue 成功数 == 容量 を確認。
//------------------------------------------------------------------------------
[[nodiscard]] bool testOwnerChannelCapacityReject()
{
    using Owner = convo::aligned_unique_ptr<const RuntimeState>;
    using Channel = convo::isr::OwnerChannel<Owner>;

    constexpr std::size_t kCapacity = 256;
    constexpr std::size_t kAttempts = kCapacity + 32;
    Channel channel;

    std::size_t accepted = 0;
    std::vector<convo::isr::OwnerChannelKey> acceptedKeys;
    acceptedKeys.reserve(kCapacity);

    for (std::size_t i = 1; i <= kAttempts; ++i)
    {
        auto world = RuntimeState::createForBuilder(RuntimeState::BuilderToken{});
        convo::isr::OwnerChannelKey key{
            static_cast<std::uint64_t>(i * 2654435761ULL),  // 分散 key（衝突回避）
            static_cast<std::uint32_t>(i),
            0
        };
        if (channel.enqueue(key, std::move(world)))
        {
            ++accepted;
            acceptedKeys.push_back(key);
        }
    }
    if (accepted != kCapacity)
    {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "accepted %zu != capacity %zu", accepted, kCapacity);
        testFail("OwnerChannel capacity reject", buf);
        return false;
    }
    if (channel.size() != kCapacity)
    {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "size %zu != capacity %zu", channel.size(), kCapacity);
        testFail("OwnerChannel capacity reject: size", buf);
        return false;
    }

    // 全件 take して確実に回収できる（owner リークなし）
    for (const auto& key : acceptedKeys)
    {
        if (!channel.take(key))
        {
            testFail("OwnerChannel capacity reject: reclaim", "take failed for accepted key");
            return false;
        }
    }
    if (channel.size() != 0)
    {
        testFail("OwnerChannel capacity reject: drain", "size != 0 after full reclaim");
        return false;
    }

    testPass("OwnerChannel capacity reject + full reclaim (256 slots)");
    return true;
}

//------------------------------------------------------------------------------
// PendingPublishRegistry 耐久
//
//   registerPublish / lookup / unregister を 50,000 サイクル。登録済み seqId が
//   lookup で一意に解決され、unregister 後は lookup が nullptr になることを確認。
//   （レジストリは 64 スロットの ring buffer のため、unregister しないまま
//   大量登録すると旧エントリが上書きされる。ここでは逐次 unregister を前提。）
//------------------------------------------------------------------------------
[[nodiscard]] bool testPendingPublishRegistryEndurance()
{
    // Coordinator は ~953KB のためヒープ確保（スタック 1MB 対策）
    auto coordinator = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    convo::isr::RuntimeWorldAuthority authority(*coordinator);
    auto& registry = authority.registry();

    constexpr int kCycles = 50000;

    for (int i = 1; i <= kCycles; ++i)
    {
        auto world = RuntimeState::createForBuilder(RuntimeState::BuilderToken{});
        if (!world)
        {
            testFail("Registry endurance", "createForBuilder returned null");
            return false;
        }
        const auto seqId = static_cast<convo::isr::PublicationSequenceId>(i);
        const void* worldPtr = world.get();

        registry.registerPublish(seqId, worldPtr);
        if (registry.lookup(seqId) != worldPtr)
        {
            char buf[96];
            std::snprintf(buf, sizeof(buf), "cycle %d: lookup mismatch", i);
            testFail("Registry endurance: lookup", buf);
            return false;
        }
        registry.unregister(seqId);
        if (registry.lookup(seqId) != nullptr)
        {
            char buf[96];
            std::snprintf(buf, sizeof(buf), "cycle %d: lookup after unregister not null", i);
            testFail("Registry endurance: unregister", buf);
            return false;
        }
    }

    testPass("PendingPublishRegistry endurance: register/lookup/unregister 50,000 cycles");
    return true;
}

//------------------------------------------------------------------------------
// PendingPublishRegistry 上書き競合（レジストリ容量 64 を超える並行登録）
//
//   容量 64 に対し 128 エントリを unregister 無しで登録 → ring buffer のため
//   後続登録が旧エントリを上書きする。lookup は最新 64 エントリ内で一意に
//   解決される（先頭の古い 64 エントリは上書きされ lookup 不能）。実運用の
//   enqueue→commit gap は 64 を大幅に下回るため、この状況は設計上の限界試験。
//   クラッシュしないことのみを検証する。
//------------------------------------------------------------------------------
[[nodiscard]] bool testPendingPublishRegistryOverwriteStress()
{
    // Coordinator は ~953KB のためヒープ確保（スタック 1MB 対策）
    auto coordinator = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    convo::isr::RuntimeWorldAuthority authority(*coordinator);
    auto& registry = authority.registry();

    constexpr int kEntries = 128;   // > kPendingPublishCapacity (64)
    std::vector<const void*> worlds;
    worlds.reserve(kEntries);

    for (int i = 1; i <= kEntries; ++i)
    {
        auto world = RuntimeState::createForBuilder(RuntimeState::BuilderToken{});
        worlds.push_back(world.get());
        registry.registerPublish(static_cast<convo::isr::PublicationSequenceId>(i), world.get());
    }

    // 後半 64 エントリ（上書きされていない）は lookup 可能。
    for (int i = kEntries - 64 + 1; i <= kEntries; ++i)
    {
        if (registry.lookup(static_cast<convo::isr::PublicationSequenceId>(i)) != worlds[i - 1])
        {
            char buf[96];
            std::snprintf(buf, sizeof(buf), "seq %d: lookable entry unresolved", i);
            testFail("Registry overwrite stress: recent entry", buf);
            return false;
        }
    }

    testPass("PendingPublishRegistry overwrite stress (128 regs > cap 64, no crash)");
    return true;
}

} // namespace

int main()
{
    std::printf("ISRSoakTests: データ構造耐久（ヘッドレス）\n");
    try
    {
        if (!testIntentQueueSaturation())
            return 1;
        if (!testPublicationIntentResidencyCounter())
            return 1;
        if (!testQuarantineIntentRingResidency())
            return 1;
        if (!testOwnerChannelEndurance())
            return 1;
        if (!testOwnerChannelCapacityReject())
            return 1;
        if (!testPendingPublishRegistryEndurance())
            return 1;
        if (!testPendingPublishRegistryOverwriteStress())
            return 1;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "ISRSoakTests FAILED (exception): %s\n", e.what());
        return 1;
    }

    if (g_failCount != 0)
    {
        std::fprintf(stderr, "ISRSoakTests: %d failures out of %d checks\n", g_failCount, g_testCount);
        return 1;
    }
    std::printf("ISRSoakTests: all headless data-structure endurance tests PASS\n");
    return 0;
}
