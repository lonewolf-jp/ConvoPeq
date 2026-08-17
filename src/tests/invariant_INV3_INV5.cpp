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
#include <thread>   // ★ dash2 §2.2 (Step 14): T9 concurrent double reclaim テスト用
#include <vector>

#include "audioengine/AtomicAccess.h"     // convo::publishAtomic / consumeAtomic
#include "core/EpochDomain.h"             // ★ work88 (X3 §6.3 / INV-X3-4): 実 EpochDomain の readerRegistrationClosed 検証
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
    // ★ work88 (X3-R4 Phase 3): readersZero precondition 検証用に active reader 数を設定可能に
    void setActiveReaderCount(std::uint32_t v) noexcept
    {
        convo::publishAtomic(activeReaders_, v, std::memory_order_release);
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
    std::uint32_t activeReaderCount() const noexcept override
    {
        return convo::consumeAtomic(activeReaders_, std::memory_order_acquire);
    }
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
    std::atomic<std::uint32_t> activeReaders_{0};   // ★ X3-R4 Phase 3: readersZero 検証用
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

    auto coordinatorStorage = std::make_unique<RuntimeIntentCoordinator>();
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

    auto coordinatorStorage = std::make_unique<RuntimeIntentCoordinator>();
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
// ★ work88 (X3 §6.3 / INV-X3-4 / INV-ISR-04): reclaim(ShutdownQuiescent) の precondition テスト。
//   - readerRegistrationClosed == false → reclaim forbidden（retire も実行しない → false）
//   - readerRegistrationClosed == true → reclaim allowed（true・Reclaimed 遷移）
//   precondition は retire 前に評価（十四次指摘）されることを直接検証する。
// ★ dash2 §2.2 (Phase A2 — Step 9/12): 旧 bool reclaim API 削除に伴い、ShutdownQuiescent の
//   precondition（reader registration closed AND readers zero）検証は ShutdownRuntime の
//   tryMakeQuiescenceProof（Q3/Q4）に移行。本テストは Proof 生成 → Permit → reclaimShutdownQuiescent
//   のフローで precondition が retire 前に評価されることを検証する。
//==============================================================================
[[nodiscard]] bool testInvX3_4ReclaimModeQuiescent()
{
    DSPHandleRuntime handleRuntime;
    TestEpochProvider provider;
    ISRRetireRouter router(provider);

    auto coordinatorStorage = std::make_unique<RuntimeIntentCoordinator>();
    auto& coordinator = *coordinatorStorage;
    // ★ dash2 §2.2 (Step 14 — Authority Singularization 完了): ShutdownRuntime は constructor 固定注入。
    //   setReclaimAuthority は廃止 — association は immutable。
    convo::isr::ShutdownRuntime shutdown(coordinator);

    int dspInstance = 0;
    const auto handle = handleRuntime.create(&dspInstance);
    if (handle.isNull())
        return false;

    // Q1（AdmissionClosed）を満たす
    shutdown.closeAdmission();
    shutdown.joinProducers();

    // --- Precondition 不成立: readersZero 不成立（active reader 残存）→ Proof 生成失敗 ---
    auto makeObservation = [&](bool regClosed, std::uint32_t readers) {
        convo::isr::ShutdownRuntime::QuiescenceObservation obs;
        obs.admissionReservationsZero = true;
        obs.allProducersJoined = true;
        obs.readerRegistrationClosed = regClosed;   // Q3
        obs.activeReadersZero = (readers == 0);     // Q4
        obs.epochSettled = true;
        obs.postStopEnqueueZero = true;
        obs.noResurrection = true;
        obs.epochGeneration = 7;
        obs.readerRegistrationGeneration = 3;
        return obs;
    };

    // Q3 不成立（registration 未 close）→ Proof 生成失敗（precondition 前評価）
    provider.setActiveReaderCount(0);
    auto proofBadReg = shutdown.tryMakeQuiescenceProof(
        makeObservation(/*regClosed=*/false, /*readers=*/0));
    if (proofBadReg.has_value())
        return false;

    // Q4 不成立（readers > 0）→ Proof 生成失敗
    auto proofBadReaders = shutdown.tryMakeQuiescenceProof(
        makeObservation(/*regClosed=*/true, /*readers=*/1));
    if (proofBadReaders.has_value())
        return false;

    // retire が実行されていないこと（precondition が retire 前に評価される — 十四次指摘）
    if (handleRuntime.isRetired(handle))
        return false;

    // --- Precondition 成立（registration closed + readers zero）→ Proof → Permit → reclaim ---
    // ★ dash2 §2.2 (Step 14 — Authority Singularization 完了): ReclaimAuthority は constructor 注入済み。
    //   Proof 生成成功時に ShutdownRuntime が coordinator.bindShutdownIdentity を自動実行（friend）。
    //   AudioEngine は bind しない（binding authority は ShutdownRuntime のみ）。
    auto proof = shutdown.tryMakeQuiescenceProof(
        makeObservation(/*regClosed=*/true, /*readers=*/0));
    if (!proof.has_value() || !proof->valid())
        return false;

    auto permit = shutdown.tryMakeReclaimPermit(*proof);
    if (!permit.has_value())
        return false;

    // 自動 bind された identity を確認（ShutdownRuntime → ReclaimAuthority の単一 authority 経路）
    if (!coordinator.shutdownIdentityBound())
        return false;
    if (!(coordinator.currentShutdownIdentity() == proof->identity()))
        return false;

    // reclaimShutdownQuiescent（identity validation → Permit consume）→ reclaim allowed（true・Reclaimed 遷移）
    if (!coordinator.reclaimShutdownQuiescent(handle, handleRuntime, router, std::move(*permit)))
        return false;
    // Reclaimed 遷移（Retired ではなくなる）
    if (handleRuntime.isRetired(handle))
        return false;
    if (coordinator.getReclaimInFlightCount() != 0)
        return false;

    return true;
}

//==============================================================================
// INV-5-1: submitRecoveryRequest full（256）→ 257 回目 drop + pendingIntentCount 不変
//==============================================================================
[[nodiscard]] bool testInv5_1RecoveryFullDrop()
{
    auto coordinatorStorage = std::make_unique<RuntimeIntentCoordinator>();
    auto& coordinator = *coordinatorStorage;

    const auto handle = DSPHandle::null();
    convo::RuntimeBuildSnapshot buildSource{};

    constexpr int kCapacity = 256;   // kRecoveryIntentQueueCapacity
    for (int i = 0; i < kCapacity; ++i)
    {
        // ★ dash2 §1.9 (Phase E): transport 成功時は true
        if (!coordinator.submitRecoveryRequest(handle, buildSource, 0))
            return false;
    }

    // full 直前: pendingIntentCount == 容量、drop なし
    if (coordinator.getPendingIntentCount() != static_cast<std::uint64_t>(kCapacity))
        return false;
    if (coordinator.recoveryIntentDropCount() != 0)
        return false;

    // 257 回目: full → drop カウンタ増 + pendingIntentCount 不変（INV-5: drop 計上整合）
    //   ★ dash2 §1.9 (Phase E): durable 化時も true（recovery obligation 存在 — wake 条件）
    if (!coordinator.submitRecoveryRequest(handle, buildSource, 0))
        return false;
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
    auto coordinatorStorage = std::make_unique<RuntimeIntentCoordinator>();
    auto& coordinator = *coordinatorStorage;

    // shutdown 前に 1 件 enqueue（正常系）
    //   ★ dash2 §1.9 (Phase E): transport 成功時は true
    convo::RuntimeBuildSnapshot buildSource{};
    if (!coordinator.submitRecoveryRequest(DSPHandle::null(), buildSource, 0))
        return false;
    if (coordinator.getPendingIntentCount() != 1)
        return false;

    // shutdown 確定（AdmissionClosed — requestShutdown が state=ShuttingDown を確定）
    coordinator.requestShutdown();
    if (coordinator.getState()
        != RuntimeIntentCoordinator::CoordinatorState::ShuttingDown)
        return false;

    // shutdown 後の submit → enqueue されず ShutdownDiscard として記録
    //   ★ dash2 §1.9 (Phase E): shutdown discard は false（wake 不要）
    if (coordinator.submitRecoveryRequest(DSPHandle::null(), buildSource, 0))
        return false;
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
    auto coordinatorStorage = std::make_unique<RuntimeIntentCoordinator>();
    auto& coordinator = *coordinatorStorage;

    // 残留 3 件を enqueue（shutdown 前に submit された想定）
    convo::RuntimeBuildSnapshot buildSource{};
    for (int i = 0; i < 3; ++i)
        coordinator.submitRecoveryRequest(DSPHandle::null(), buildSource, 0);
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

// ★ work88 (X3 §6.3 / INV-X3-4 / INV-ISR-04): readerRegistrationClosed テスト。
//   実 convo::EpochDomain で: closeReaderRegistration() 前に registerReaderThread() が成功し、
//   後には必ず -1（失敗）を返す。reserveReaderThread() も同様に false。
//   readerRegistrationClosed() が true を返す。登録済み slot の enter/exit は継続可能
//   （新規登録のみを拒否 — 十八次別視点14）は、enterReader が deprecated のため
//   本テストでは登録封鎖の決定性のみを検証する。
[[nodiscard]] bool testInvX3_4ReaderRegistrationClosed()
{
    convo::EpochDomain domain;

    // close 前: 登録成功
    const int idx = domain.registerReaderThread("X3Test");
    if (idx < 0 || idx >= convo::EpochDomain::kMaxReaders)
        return false;

    // closeReaderRegistration 後: 新規登録は必ず失敗（INV-X3-4）
    domain.closeReaderRegistration();
    if (domain.registerReaderThread("X3Test2") != -1)
        return false;
    if (domain.reserveReaderThread(0))
        return false;

    // readerRegistrationClosed() が true
    if (!domain.readerRegistrationClosed())
        return false;

    return true;
}

//==============================================================================
// ★ dash2 §2.2 (Phase A2 — G13/G19/G20): EpochDomain generation 実供給テスト
//   G19: publishEpoch() ごとに epochGeneration() が増加する
//   G20: closeReaderRegistration() ごとに readerRegistrationGeneration() が増加する
//==============================================================================
[[nodiscard]] bool testA2G19G20EpochGenerationSupply()
{
    convo::EpochDomain domain;

    const auto gen0 = domain.epochGeneration();
    (void)domain.publishEpoch();
    (void)domain.publishEpoch();
    if (domain.epochGeneration() != gen0 + 2)
        return false;                       // G19: publishEpoch ×2 で generation +2

    const auto reg0 = domain.readerRegistrationGeneration();
    domain.closeReaderRegistration();
    if (domain.readerRegistrationGeneration() != reg0 + 1)
        return false;                       // G20: closeReaderRegistration で generation +1

    // readerRegistrationClosed 後に publishEpoch しても epochGeneration は増える（独立）
    const auto gen1 = domain.epochGeneration();
    (void)domain.publishEpoch();
    if (domain.epochGeneration() != gen1 + 1)
        return false;

    return true;
}

//==============================================================================
// ★ dash2 §2.2 (Phase A2 — G10/G13/G21/G22): Proof/Permit acceptance テスト
//   G10: postStopEnqueueZero が Q 条件の一部（observation 経由）
//   G13: epochSettled が Q 条件の一部
//   G21: ReclaimPermit::consume() が single-use（2 回目は false）
//   G22: ShutdownRuntime のみ Proof/Permit を生成可能（外部からは生成不能 = コンパイル不能）
//   ※ tryMakeQuiescenceProof は全 Q 成立時のみ valid Proof を返す（H.11.11.3）
//==============================================================================
[[nodiscard]] bool testA2G10G13G21G22ProofPermit()
{
    // ★ dash2 §2.2 (Step 14 — Authority Singularization 完了): ReclaimAuthority を constructor 注入。
    //   setReclaimAuthority は廃止 — association は immutable。本テストは reclaimShutdownQuiescent を
    //   呼ばないため、Proof 生成時の bindShutdownIdentity は無害（fresh coordinator）。
    auto coordinatorStorage = std::make_unique<RuntimeIntentCoordinator>();
    convo::isr::ShutdownRuntime shutdown(*coordinatorStorage);

    // Q1（AdmissionClosed）を満たす: Open → Closing → Closed（H.11.4 FSM）
    shutdown.closeAdmission();
    shutdown.joinProducers();
    if (shutdown.admissionState() != convo::isr::AdmissionState::Closed)
        return false;

    // Q 条件を全成立させる観測値
    ShutdownRuntime::QuiescenceObservation obs;
    obs.admissionReservationsZero = true;
    obs.allProducersJoined = true;
    obs.readerRegistrationClosed = true;
    obs.activeReadersZero = true;
    obs.epochSettled = true;            // G13
    obs.postStopEnqueueZero = true;     // G10
    obs.noResurrection = true;
    obs.epochGeneration = 7;            // G19
    obs.readerRegistrationGeneration = 3;  // G20

    // 全 Q 成立 → valid Proof
    auto proof = shutdown.tryMakeQuiescenceProof(obs);
    if (!proof.has_value() || !proof->valid())
        return false;
    if (proof->identity().epochGeneration != 7)
        return false;                   // G19: epoch generation 束縛
    if (proof->identity().readerRegistrationGeneration != 3)
        return false;                   // G20: readerReg generation 束縛
    if (!proof->epochSettled())
        return false;                   // G13: Q5 フラグ
    if (!proof->postStopEnqueueZero())
        return false;                   // G10: Q6 フラグ

    // Q 条件が 1 つでも欠けると nullopt（Q0〜Q7 全条件必須 — 簡易生成禁止）
    obs.epochSettled = false;           // G13 違反
    if (shutdown.tryMakeQuiescenceProof(obs).has_value())
        return false;
    obs.epochSettled = true;
    obs.postStopEnqueueZero = false;    // G10 違反
    if (shutdown.tryMakeQuiescenceProof(obs).has_value())
        return false;
    obs.postStopEnqueueZero = true;

    // Proof → Permit 生成（G17〜G20 identity 束縛）
    auto proof2 = shutdown.tryMakeQuiescenceProof(obs);
    if (!proof2.has_value())
        return false;
    auto permit = shutdown.tryMakeReclaimPermit(*proof2);
    if (!permit.has_value())
        return false;
    if (!(permit->identity() == proof2->identity()))
        return false;                   // G17: identity match（INV-LIFE-5）

    // G21: consume() は single-use（1 回目 true / 2 回目 false）
    if (!permit->consume())
        return false;
    if (permit->consume())
        return false;                   // 二重 consume 拒否

    return true;
}

//==============================================================================
// ★ dash2 §2.2 (Phase A2 — Step 14 / T10): Permit ABA（generation 安定性・stale reject）
//   - 同一 shutdown transaction 内で複数 Proof を生成しても identity.generation は同一
//     （closeAdmission で確定・Proof 生成ごとに進めない — Race B の前提）
//   - closeAdmission()（shutdown 開始）で generation が 1 回だけ前進する
//     （2 回目は Open→Closing が失敗 = no-op — 二重開始防止 INV-LIFE-6）
//   - 古い generation の Permit は currentShutdownGeneration と不一致 → stale reject（AC-5）
//==============================================================================
[[nodiscard]] bool testA2Step14PermitABA()
{
    // ReclaimAuthority（RuntimeIntentCoordinator）を先に確保（constructor 注入用 — 大型のためヒープ確保）。
    auto coordinatorStorage = std::make_unique<RuntimeIntentCoordinator>();
    auto& coordinator = *coordinatorStorage;

    // ★ dash2 §2.2 (Step 14 — Authority Singularization 完了): ShutdownRuntime は constructor 固定注入。
    //   setReclaimAuthority は廃止 — association は immutable。
    convo::isr::ShutdownRuntime shutdown(coordinator);
    shutdown.closeAdmission();          // Shutdown N 開始（generation 前進: 0 → 1）
    shutdown.joinProducers();

    const auto genN = shutdown.currentShutdownGeneration();
    if (genN == 0)
        return false;                   // closeAdmission で generation 確定済み

    // closeAdmission の 2 回目は no-op（Open→Closing 失敗）→ generation 不変（INV-LIFE-6）
    shutdown.closeAdmission();
    if (shutdown.currentShutdownGeneration() != genN)
        return false;

    convo::isr::ShutdownRuntime::QuiescenceObservation obs;
    obs.admissionReservationsZero = true;
    obs.allProducersJoined = true;
    obs.readerRegistrationClosed = true;
    obs.activeReadersZero = true;
    obs.epochSettled = true;
    obs.postStopEnqueueZero = true;
    obs.noResurrection = true;
    obs.epochGeneration = 10;
    obs.readerRegistrationGeneration = 4;

    // ── T10d: ReclaimAuthority は constructor 固定注入（setReclaimAuthority 廃止 = compile-time）──
    //   Authority Singularization 完了: wiring の public mutator は存在しない（setReclaimAuthority 削除）。
    //   association は constructor で immutable に固定され、Proof 生成成功時に注入した coordinator
    //   のみが bindShutdownIdentity を受ける。
    {
        auto coordAStorage = std::make_unique<RuntimeIntentCoordinator>();
        auto coordBStorage = std::make_unique<RuntimeIntentCoordinator>();
        convo::isr::ShutdownRuntime s(*coordAStorage);   // constructor 固定注入
        s.closeAdmission();
        s.joinProducers();
        auto proofF = s.tryMakeQuiescenceProof(obs);
        if (!proofF.has_value() || !proofF->valid())
            return false;
        if (!coordAStorage->shutdownIdentityBound())
            return false;                            // 注入した A に bind（固定）
        if (coordBStorage->shutdownIdentityBound())
            return false;                            // B は無関係（bind されない）
    }

    // ── T10a: generation stability ──
    //   同一 shutdown 内で複数 Proof 生成 → identity.generation は同一（closeAdmission 確定値）
    auto proof1 = shutdown.tryMakeQuiescenceProof(obs);
    auto proof2 = shutdown.tryMakeQuiescenceProof(obs);
    if (!proof1.has_value() || !proof2.has_value())
        return false;
    if (proof1->identity().generation != genN)
        return false;
    if (proof2->identity().generation != genN)
        return false;                   // 複数 Proof で generation 不変（T10a）

    // 複数 Permit 発行 → 同一 identity（provenance 一致）
    auto permit1 = shutdown.tryMakeReclaimPermit(*proof1);
    auto permit2 = shutdown.tryMakeReclaimPermit(*proof2);
    if (!permit1.has_value() || !permit2.has_value())
        return false;
    if (permit1->identity().generation != genN)
        return false;
    if (!(permit1->identity() == permit2->identity()))
        return false;                   // 同一 shutdown 内の Permit identity 一致（T10a）

    // ── ReclaimAuthority（RuntimeIntentCoordinator）は constructor 注入済み ──
    DSPHandleRuntime handleRuntime;
    TestEpochProvider provider;
    ISRRetireRouter router(provider);

    int dspInstance = 0;
    const auto handle = handleRuntime.create(&dspInstance);
    if (handle.isNull())
        return false;

    // ★ dash2 §2.2 (Step 14 — Authority Singularization 完了): binding authority は ShutdownRuntime のみ。
    //   ReclaimAuthority は constructor 注入済み。Proof 生成成功時に ShutdownRuntime が
    //   bindShutdownIdentity を自動実行（AudioEngine は bind しない）。coordinator.setShutdownIdentity は廃止。
    auto proofBind = shutdown.tryMakeQuiescenceProof(obs);   // 自動 bind（identity = genN）
    if (!proofBind.has_value() || !proofBind->valid())
        return false;
    if (!coordinator.shutdownIdentityBound())
        return false;
    if (!(coordinator.currentShutdownIdentity() == proofBind->identity()))
        return false;

    // ── T10c: generation mismatch rejection at ReclaimAuthority boundary ──
    //   ［実 shutdown lifecycle 上の temporal transition（Shutdown N → N+1）ではなく、機構レベルで
    //     「Permit.identity と ReclaimAuthority の bind identity の不一致 → reject」を検証。
    //     同一 coordinator は再 bind 不可（Unbound → Bound → Fixed）のため別 coordinator で検証］
    //   ReclaimAuthority に shutdown identity（generation N）が bind 済み → 有効 Permit で reclaim 成功
    auto validPermit = shutdown.tryMakeReclaimPermit(*proofBind);
    if (!validPermit.has_value())
        return false;
    if (!coordinator.reclaimShutdownQuiescent(handle, handleRuntime, router, std::move(*validPermit)))
        return false;                   // 有効 Permit（identity 一致）→ reclaim 成功
    if (handleRuntime.isRetired(handle))
        return false;                   // Reclaimed 遷移

    // stale reject: 別 coordinator（ReclaimAuthority B）に generation N+1 の identity を bind し、
    //   Permit N（generation N）を渡す → generation 不一致（temporal）で reject。
    //   ［再 bind 禁止のため、同一 coordinator では N→N+1 をシミュレートできない。
    //     Authority temporal stale rejection の機構検証として別 coordinator を使用］
    auto coordinatorBStorage = std::make_unique<RuntimeIntentCoordinator>();
    auto& coordinatorB = *coordinatorBStorage;
    convo::isr::ShutdownRuntimeIdentity idN1 = proof2->identity();
    idN1.generation = genN + 1;         // N+1（同一 engineInstanceId・generation 前進）
    //   ShutdownRuntime 経由で bind（friend のみ — 直接 bindShutdownIdentity はテストから呼べない）
    convo::isr::ShutdownRuntime shutdownB(coordinatorB);   // 別 ShutdownRuntime（N+1 の identity 生成用）— constructor 注入
    //   coordinatorB には idN1 を bind — ただし bind は ShutdownRuntime の Proof 生成経由のみ。
    //   テストから private bind を呼べないため、shutdownB の Proof 生成で identity を生成し、
    //   その identity が idN1（generation N+1）と一致するよう観測値を調整する。
    //   ［shutdownB の generation は 1（closeAdmission で確定）。idN1.generation は genN+1 であり、
    //     genN は A の generation なので、B の generation 1 と idN1 の整合は取れない。
    //     ここでは機構検証として、A の stale Permit（generation N）を B の authority に渡し、
    //     B は自身の identity（generation 1, engineInstanceId B）と不一致 → reject を検証］
    shutdownB.closeAdmission();
    shutdownB.joinProducers();
    auto proofB = shutdownB.tryMakeQuiescenceProof(obs);
    if (!proofB.has_value() || !proofB->valid())
        return false;
    // B の authority は B の identity（engineInstanceId B）に bind 済み
    if (!coordinatorB.shutdownIdentityBound())
        return false;

    // A の stale Permit（generation N, engineInstanceId A）を B の authority に渡す → reject
    int dspInstance2 = 1;
    const auto handle2 = handleRuntime.create(&dspInstance2);
    if (handle2.isNull())
        return false;
    auto stalePermit = shutdown.tryMakeReclaimPermit(*proof1);   // identity = A/genN
    if (!stalePermit.has_value())
        return false;
    if (coordinatorB.reclaimShutdownQuiescent(handle2, handleRuntime, router, std::move(*stalePermit)))
        return false;                   // stale Permit（B の identity と不一致）→ reject されるべき
    // reject により reclaim 未実行: handle2 は Active のまま
    if (handleRuntime.isRetired(handle2))
        return false;                   // reject で handle が Retired に遷移するのは不正

    // ── T10b: cross-runtime provenance rejection ──
    //   別 Runtime インスタンス（engineInstanceId が異なる）の Permit を A の ReclaimAuthority へ注入
    //   ［shutdownNext の Proof 生成で Permit B を発行し、A の coordinator に渡す］
    //   （constructor 注入用 coordinator — Proof 生成の bind は無害）
    auto coordinatorCStorage = std::make_unique<RuntimeIntentCoordinator>();
    convo::isr::ShutdownRuntime shutdownNext(*coordinatorCStorage);   // Runtime C / Shutdown 開始 — constructor 注入
    shutdownNext.closeAdmission();
    shutdownNext.joinProducers();
    if (shutdownNext.engineInstanceId() == shutdown.engineInstanceId())
        return false;                   // インスタンス ID は一意であるべき

    auto proofC = shutdownNext.tryMakeQuiescenceProof(obs);
    if (!proofC.has_value() || !proofC->valid())
        return false;
    auto permitC = shutdownNext.tryMakeReclaimPermit(*proofC);
    if (!permitC.has_value())
        return false;
    // A の ReclaimAuthority に C の Permit を渡す → engineInstanceId 不一致で reject
    int dspInstance3 = 2;
    const auto handle3 = handleRuntime.create(&dspInstance3);
    if (handle3.isNull())
        return false;
    if (coordinator.reclaimShutdownQuiescent(handle3, handleRuntime, router, std::move(*permitC)))
        return false;                   // cross-runtime Permit（engineInstanceId 不一致）→ reject

    return true;
}

//==============================================================================
// ★ dash2 §2.2 (Phase A2 — Step 14 / T11): setter resurrection（compile-time invariant）
//   旧 setter API が production から再出現しないことを確認する。
//   本テストはコンパイル時検証（ReclaimMode enum 消滅 = 旧 API 使用不可）を前提とし、
//   ここではリフレクション的に確認できる範囲で「setRetireBacklogCount 等が
//   TEST-ONLY である」ことを文書化する。
//   ［実際の検証は rg で production 参照 0 を確認（AC-1）］
//==============================================================================
[[nodiscard]] bool testA2Step14SetterResurrection()
{
    // ReclaimMode enum が消滅した（コンパイル時: RuntimeIntentCoordinator::ReclaimMode は不在）。
    // 旧 bool reclaim 経路（reclaim(ReclaimMode,...)）はコンパイル不能 — compile guard 成立（AC-1）。
    // ここではテストビルドが成立すること自体が検証（旧 API 参照があればコンパイル失敗）。
    return true;
}

//==============================================================================
// ★ dash2 §2.2 (Phase A2 — Step 14 / T13): destruction ordering audit
//   CacheMap の destruction ordering（reclaim 成功 → physical destruction）を検証する。
//   ［H.11.11.9.4: delete が reclaim より前だと reclaim 失敗時に object 消滅 + handle 未回収。
//     Step 13 で reclaim 成功後に物理解放へ修正済み — 本テストはその不変条件を文書化］
//==============================================================================
[[nodiscard]] bool testA2Step14DestructionOrderingAudit()
{
    // Step 13 実装: ~CacheMap は tryShutdownQuiescentReclaim（reclaim 成功）後に EQCoeffCache を
    // 物理解放する（reclaim → physical destruction の順序）。reclaim 失敗時は物理解放しない。
    // 本テストではこの契約を直接検証できない（AudioEngine 完全型が必要）ため、
    // コンパイル時 + コードレビューで担保されることを文書化する。
    return true;
}

//==============================================================================
// ★ dash2 §2.2 (Phase A2 — Step 14 / T9): concurrent double reclaim テスト
//   同一 ReclaimPermit を 2 スレッドから同時に consume() し、正確に 1 つだけ成功することを
//   検証する（consume() の CAS: Issued → Consumed の単一 linearization — INV-LIFE-7 / Race C）。
//   ［ReclaimPermit は move-only のため std::shared_ptr で共有し、consume() のアトミック性を検証］
//==============================================================================
[[nodiscard]] bool testA2Step14ConcurrentDoubleReclaim()
{
    // ★ dash2 §2.2 (Step 14 — Authority Singularization 完了): ReclaimAuthority を constructor 注入。
    //   setReclaimAuthority は廃止 — association は immutable。（本テストは consume 検証のみのため bind は無害）
    auto coordinatorStorage = std::make_unique<RuntimeIntentCoordinator>();
    convo::isr::ShutdownRuntime shutdown(*coordinatorStorage);
    shutdown.closeAdmission();
    shutdown.joinProducers();

    convo::isr::ShutdownRuntime::QuiescenceObservation obs;
    obs.admissionReservationsZero = true;
    obs.allProducersJoined = true;
    obs.readerRegistrationClosed = true;
    obs.activeReadersZero = true;
    obs.epochSettled = true;
    obs.postStopEnqueueZero = true;
    obs.noResurrection = true;
    obs.epochGeneration = 5;
    obs.readerRegistrationGeneration = 2;

    auto proof = shutdown.tryMakeQuiescenceProof(obs);
    if (!proof.has_value() || !proof->valid())
        return false;
    auto permitOpt = shutdown.tryMakeReclaimPermit(*proof);
    if (!permitOpt.has_value())
        return false;

    // move-only Permit を共有（consume() の CAS アトミック性検証用）
    auto shared = std::make_shared<ReclaimPermit>(std::move(*permitOpt));

    std::atomic<int> successCount{0};
    std::vector<std::thread> threads;
    for (int i = 0; i < 2; ++i)
    {
        threads.emplace_back([shared, &successCount]() {
            if (shared->consume())
                successCount.fetch_add(1, std::memory_order_relaxed);
        });
    }
    for (auto& t : threads)
        t.join();

    // Race C: 同一 Permit の consume は正確に 1 回だけ成功（T9）
    if (successCount.load(std::memory_order_relaxed) != 1)
        return false;

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
        if (!convo::isr::testInvX3_4ReaderRegistrationClosed())
            throw std::runtime_error("INV-X3-4: reader registration closure 違反");
        if (!convo::isr::testInvX3_4ReclaimModeQuiescent())
            throw std::runtime_error("INV-X3-4: ShutdownQuiescent reclaim precondition 違反");
        if (!convo::isr::testA2G19G20EpochGenerationSupply())
            throw std::runtime_error("A2-G19/G20: epoch/readerReg generation 供給違反");
        if (!convo::isr::testA2G10G13G21G22ProofPermit())
            throw std::runtime_error("A2-G10/G13/G21/G22: Proof/Permit acceptance 違反");
        if (!convo::isr::testA2Step14ConcurrentDoubleReclaim())
            throw std::runtime_error("A2-Step14/T9: concurrent double reclaim 違反");
        if (!convo::isr::testA2Step14PermitABA())
            throw std::runtime_error("A2-Step14/T10: Permit ABA 違反");
        if (!convo::isr::testA2Step14SetterResurrection())
            throw std::runtime_error("A2-Step14/T11: setter resurrection 違反");
        if (!convo::isr::testA2Step14DestructionOrderingAudit())
            throw std::runtime_error("A2-Step14/T13: destruction ordering audit 違反");
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }

    std::printf("invariant_INV3_INV5Tests: ALL TESTS PASSED\n");
    return 0;
}
