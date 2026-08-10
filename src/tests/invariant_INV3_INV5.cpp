//==============================================================================
// invariant_INV3_INV5.cpp — work88 (P2-2) ISR 不変条件テスト
//
// 検証対象（REPAIR_PLAN2-dash.md §1.4 / P2-2）:
//   INV-3: requestReclaim の retire → epoch 安全確認 → reclaim 順序
//     INV-3-1: retire → epoch 安全（retireEpoch < minReaderEpoch）→ reclaim 完了（true）
//               → handle は Retired ではなくなる（Reclaimed 遷移）
//     INV-3-2: epoch 非安全（retireEpoch >= minReaderEpoch）→ false（遅延通知）
//               → retire は実行済み・reclaim 未実行（isRetired == true）
//               → epoch 安全化後の再試行で reclaim 完了（TOCTOU 修正: 呼出し元が
//                 handle を再試行リストへ戻す契約の裏付け）
//   INV-5: Recovery Intent の drop 禁止 / 状態遷移ゲート
//     INV-5-1: submitRecoveryRequest を 256 回（kRecoveryIntentQueueCapacity）で full
//               → 257 回目は drop カウンタ増 + pendingIntentCount 不変（INV-5 計上整合）
//               → popRecoveryRequest で reservation 消費（pop 成功数 == push 成功数）
//     INV-5-2: QuarantineIntentHandler が依存する QuarantineService::executeQuarantine の
//               stateChanged 判定（HANDLER-1 precondition）:
//               - 無効 handle → stateChanged == false（Recovery 非発行）
//               - 有効 handle → stateChanged == true（Recovery 発行条件を満たす）
//               ※ ハンドラ本体は AudioEngine 完全型が必要なため、本テストはハンドラが
//                 尊重する唯一の判定源（QuarantineService）の契約を検証する。
//     INV-5-3 (P2-4 監査補正 — Step B): requestShutdown()（AdmissionClosed）後の
//               submitRecoveryRequest は enqueue されず、ShutdownDiscard として記録
//               （recoveryShutdownDiscardCount+1）。pendingIntentCount は不変（reservation 前 gate）。
//     INV-5-4 (P2-4 監査補正 — Step C): discardRecoveryRequestsOnShutdown() が残留 Recovery を
//               明示 discard（ShutdownDiscard）。queue empty + pending 0 + discard カウント増。
//
// ビルド: CMakeLists.txt の ISRSemanticValidationTests と同一パターンで
//   invariant_INV3_INV5Tests を定義する（ISR*.cpp を同一ターゲットでコンパイル）。
//==============================================================================

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>

#include "audioengine/AtomicAccess.h"     // convo::publishAtomic / consumeAtomic
#include "core/IEpochProvider.h"          // TestEpochProvider の基底
#include "audioengine/ISRRetireRouter.h"  // requestReclaim の router 引数
#include "audioengine/ISRRuntimePublicationCoordinator.h"  // テスト対象 Coordinator
#include "audioengine/ISRDSPHandle.h"     // DSPHandleRuntime / DSPHandle
#include "audioengine/ISRDSPQuarantine.h" // DSPQuarantineManager / QuarantineReason

namespace convo::isr {
namespace {

//==============================================================================
// ★ P2-2: 制御可能な IEpochProvider スタブ
//   ISRRetireRouter::currentEpoch() / minReaderEpoch() は provider へ委譲されるため、
//   このスタブの値を切り替えて INV-3 の安全（retireEpoch < minReaderEpoch）/
//   非安全（retireEpoch >= minReaderEpoch）判定を deterministic に検証する。
//==============================================================================
class TestEpochProvider : public convo::IEpochProvider
{
public:
    void setCurrentEpoch(std::uint64_t v) noexcept
    {
        convo::publishAtomic(current_, v, std::memory_order_release);
    }
    void setMinReaderEpoch(std::uint64_t v) noexcept
    {
        convo::publishAtomic(minReader_, v, std::memory_order_release);
    }

    // ── IReaderEpochProvider ──
    int registerReaderThread() noexcept override { return 0; }
    bool reserveReaderThread(int) noexcept override { return true; }
    void enterReader(int) noexcept override {}
    void exitReader(int) noexcept override {}
    std::uint64_t currentEpoch() const noexcept override
    {
        return convo::consumeAtomic(current_, std::memory_order_acquire);
    }
    std::uint32_t activeReaderCount() const noexcept override { return 0; }
    int readerCapacity() const noexcept override { return 1; }
    std::uint64_t getMinReaderEpoch() const noexcept override
    {
        return convo::consumeAtomic(minReader_, std::memory_order_acquire);
    }

    // ── IPublicationProvider ──
    std::uint64_t publishEpoch() noexcept override
    {
        return convo::consumeAtomic(current_, std::memory_order_acquire);
    }

    // ── IRetireProvider ──
    bool enqueueRetire(void*, void (*)(void*), std::uint64_t) noexcept override { return true; }
    void tryReclaim() noexcept override {}
    std::uint32_t pendingRetireCount() const noexcept override { return 0; }
    void drainAll() noexcept override {}

private:
    std::atomic<std::uint64_t> current_{0};
    std::atomic<std::uint64_t> minReader_{0};
};

//==============================================================================
// INV-3-1: retire → epoch 安全 → reclaim の順序
//==============================================================================
[[nodiscard]] bool testInv3_1RetireEpochSafeReclaim()
{
    DSPHandleRuntime handleRuntime;
    TestEpochProvider provider;
    provider.setCurrentEpoch(5);      // retireEpoch = 5
    provider.setMinReaderEpoch(10);   // 5 < 10 → epoch 安全
    ISRRetireRouter router(provider);

    auto coordinatorStorage = std::make_unique<RuntimePublicationCoordinator>();
    auto& coordinator = *coordinatorStorage;

    // 有効な DSP handle を作成（Active 状態）
    int dspInstance = 0;
    const auto handle = handleRuntime.create(&dspInstance);
    if (handle.isNull())
        return false;

    // requestReclaim: retire → epoch 安全確認 → reclaim が完了し true が返る
    if (!coordinator.requestReclaim(handle, handleRuntime, router))
        return false;

    // reclaim 完了後: state は Retired ではなくなる（Reclaimed 遷移）
    if (handleRuntime.isRetired(handle))
        return false;

    // reclaim-in-flight は 0 にリセット済み
    if (coordinator.getReclaimInFlightCount() != 0)
        return false;

    return true;
}

//==============================================================================
// INV-3-2: epoch 非安全 → 遅延（false）→ epoch 安全化後の再試行で reclaim
//   TOCTOU 修正: 呼出し元は false 時に handle を再試行リストへ戻す
//   （slot リーク防止）。本テストはその契約（遅延通知 + 再試行成功）を検証する。
//==============================================================================
[[nodiscard]] bool testInv3_2ReclaimDeferredThenSucceeds()
{
    DSPHandleRuntime handleRuntime;
    TestEpochProvider provider;
    ISRRetireRouter router(provider);

    auto coordinatorStorage = std::make_unique<RuntimePublicationCoordinator>();
    auto& coordinator = *coordinatorStorage;

    int dspInstance = 0;
    const auto handle = handleRuntime.create(&dspInstance);
    if (handle.isNull())
        return false;

    // epoch 非安全: retireEpoch(10) >= minReaderEpoch(10) → reclaim を遅延
    provider.setCurrentEpoch(10);
    provider.setMinReaderEpoch(10);
    if (coordinator.requestReclaim(handle, handleRuntime, router))
        return false;                          // 遅延が通知されるべき（false）

    // retire は実行済み（Retired に遷移）だが reclaim は未実行
    if (!handleRuntime.isRetired(handle))
        return false;

    // 遅延中は reclaim-in-flight が 1（pending として管理）
    if (coordinator.getReclaimInFlightCount() != 1)
        return false;

    // 呼出し元は handle を再試行リストへ戻す（TOCTOU 修正）。epoch が安全になった後、
    // 再試行 → reclaim 完了（true）
    provider.setMinReaderEpoch(20);            // 10 < 20 → epoch 安全
    if (!coordinator.requestReclaim(handle, handleRuntime, router))
        return false;

    // reclaim 完了: in-flight は 0、handle は Retired ではなくなる
    if (coordinator.getReclaimInFlightCount() != 0)
        return false;
    if (handleRuntime.isRetired(handle))
        return false;

    return true;
}

//==============================================================================
// INV-5-1: submitRecoveryRequest full（256）→ 257 回目 drop + pendingIntentCount 不変
//==============================================================================
[[nodiscard]] bool testInv5_1RecoveryFullDrop()
{
    auto coordinatorStorage = std::make_unique<RuntimePublicationCoordinator>();
    auto& coordinator = *coordinatorStorage;

    const auto handle = DSPHandle::null();
    convo::RuntimeBuildSnapshot buildSource{};

    constexpr int kCapacity = 256;   // kRecoveryIntentQueueCapacity
    for (int i = 0; i < kCapacity; ++i)
        coordinator.submitRecoveryRequest(handle, buildSource);

    // full 直前: pendingIntentCount == 容量、drop なし
    if (coordinator.getPendingIntentCount() != static_cast<std::uint64_t>(kCapacity))
        return false;
    if (coordinator.recoveryIntentDropCount() != 0)
        return false;

    // 257 回目: full → drop カウンタ増 + pendingIntentCount 不変（INV-5: drop 計上整合）
    coordinator.submitRecoveryRequest(handle, buildSource);
    if (coordinator.recoveryIntentDropCount() != 1)
        return false;
    if (coordinator.getPendingIntentCount() != static_cast<std::uint64_t>(kCapacity))
        return false;

    // pop 成功で reservation を消費 → 1 件減る（pop 成功数 == push 成功数の不変条件）
    if (!coordinator.popRecoveryRequest().has_value())
        return false;
    if (coordinator.getPendingIntentCount() != static_cast<std::uint64_t>(kCapacity - 1))
        return false;

    return true;
}

//==============================================================================
// INV-5-2: QuarantineIntentHandler が依存する executeQuarantine の stateChanged 判定
//   （無効 handle → false = Recovery 非発行 / 有効 handle → true = Recovery 発行）
//==============================================================================
[[nodiscard]] bool testInv5_2QuarantineStateChangedGate()
{
    DSPHandleRuntime handleRuntime;
    DSPQuarantineManager quarantineManager;
    QuarantineService service;

    int dspInstance = 0;
    const auto validHandle = handleRuntime.create(&dspInstance);
    if (validHandle.isNull())
        return false;

    // 無効 handle（null）→ stateChanged == false（Recovery は発行されない）
    {
        const QuarantineService::QuarantineRequest nullReq{
            DSPHandle::null(), QuarantineReason::Unknown, 0
        };
        const auto nullResult = service.executeQuarantine(handleRuntime, quarantineManager, nullReq);
        if (nullResult.stateChanged)
            return false;   // 無効 handle で stateChanged にしてはならない
    }

    // 有効 handle → stateChanged == true（Recovery 発行条件を満たす）
    {
        const QuarantineService::QuarantineRequest validReq{
            validHandle, QuarantineReason::Unknown, 1
        };
        const auto validResult = service.executeQuarantine(handleRuntime, quarantineManager, validReq);
        if (!validResult.stateChanged)
            return false;
    }

    return true;
}

//==============================================================================
// INV-5-3: requestShutdown()（AdmissionClosed）後の submit → ShutdownDiscard
//   P2-4 監査補正 (Step B): Coordinator in-flight phase 中に shutdown が発生しても、
//   submit 側 gate が Recovery admission の最終 linearization point になることを検証。
//==============================================================================
[[nodiscard]] bool testInv5_3SubmitAfterShutdownDiscards()
{
    auto coordinatorStorage = std::make_unique<RuntimePublicationCoordinator>();
    auto& coordinator = *coordinatorStorage;

    // shutdown 前に 1 件 enqueue（正常系）
    convo::RuntimeBuildSnapshot buildSource{};
    coordinator.submitRecoveryRequest(DSPHandle::null(), buildSource);
    if (coordinator.getPendingIntentCount() != 1)
        return false;

    // shutdown 確定（AdmissionClosed — requestShutdown が state=ShuttingDown を確定）
    coordinator.requestShutdown();
    if (coordinator.getState()
        != RuntimePublicationCoordinator::CoordinatorState::ShuttingDown)
        return false;

    // shutdown 後の submit → enqueue されず ShutdownDiscard として記録
    coordinator.submitRecoveryRequest(DSPHandle::null(), buildSource);
    if (coordinator.recoveryShutdownDiscardCount() != 1)
        return false;
    if (coordinator.recoveryIntentDropCount() != 0)
        return false;                        // drop（queue full）とは区別
    if (coordinator.getPendingIntentCount() != 1)
        return false;                        // pending 不変（reservation 前 gate）

    // queue には shutdown 前の 1 件のみ → pop で 1 件消費 → pending 0
    if (!coordinator.popRecoveryRequest().has_value())
        return false;
    if (coordinator.getPendingIntentCount() != 0)
        return false;
    if (coordinator.popRecoveryRequest().has_value())
        return false;

    return true;
}

//==============================================================================
// INV-5-4: discardRecoveryRequestsOnShutdown() による残留の明示 discard
//   P2-4 監査補正 (Step C): Builder 停止後に残留する Recovery を ShutdownDiscard として
//   明示破棄する（silent loss 禁止）。popRecoveryRequest が fetchSub するため counter 整合。
//==============================================================================
[[nodiscard]] bool testInv5_4DiscardResidualOnShutdown()
{
    auto coordinatorStorage = std::make_unique<RuntimePublicationCoordinator>();
    auto& coordinator = *coordinatorStorage;

    // 残留 3 件を enqueue（shutdown 前に submit された想定）
    convo::RuntimeBuildSnapshot buildSource{};
    for (int i = 0; i < 3; ++i)
        coordinator.submitRecoveryRequest(DSPHandle::null(), buildSource);
    if (coordinator.getPendingIntentCount() != 3)
        return false;

    // Builder 停止相当 → 明示 discard
    coordinator.discardRecoveryRequestsOnShutdown();
    if (coordinator.recoveryShutdownDiscardCount() != 3)
        return false;
    if (coordinator.getPendingIntentCount() != 0)
        return false;
    if (coordinator.popRecoveryRequest().has_value())
        return false;                        // queue empty

    return true;
}

} // anonymous namespace
} // namespace convo::isr

//==============================================================================
// main
//==============================================================================
int main()
{
    try
    {
        if (!convo::isr::testInv3_1RetireEpochSafeReclaim())
            throw std::runtime_error("INV-3-1: retire → epoch safe → reclaim 順序違反");
        if (!convo::isr::testInv3_2ReclaimDeferredThenSucceeds())
            throw std::runtime_error("INV-3-2: epoch 非安全 → pending 再登録（TOCTOU）違反");
        if (!convo::isr::testInv5_1RecoveryFullDrop())
            throw std::runtime_error("INV-5-1: Recovery full → drop + pendingIntentCount 不変 違反");
        if (!convo::isr::testInv5_2QuarantineStateChangedGate())
            throw std::runtime_error("INV-5-2: Quarantine stateChanged ゲート違反");
        if (!convo::isr::testInv5_3SubmitAfterShutdownDiscards())
            throw std::runtime_error("INV-5-3: shutdown 後 submit → ShutdownDiscard 違反");
        if (!convo::isr::testInv5_4DiscardResidualOnShutdown())
            throw std::runtime_error("INV-5-4: 残留 Recovery の明示 discard 違反");
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }

    std::printf("invariant_INV3_INV5Tests: ALL TESTS PASSED\n");
    return 0;
}
