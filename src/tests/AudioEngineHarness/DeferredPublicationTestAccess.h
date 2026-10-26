// DeferredPublicationTestAccess.h
// AudioEngineHarness 統合テスト専用の Friend Test Access。
//
// AudioEngine は private メンバ（runtimeOrchestrator_ / testFadingRuntimePresent_）を
// 持つため、テストから Orchestrator へ到達するには friend 宣言（AudioEngine.h:3515）
// が必要。本クラスは DeferredFlowIntegrationTests / DeferredPublishViewStateMachineTests
// の双方から共有する（グローバル名前空間・AudioEngine.h:122 の前方宣言に合わせる）。
//
// ★ Testing Principle（第13回レビュー反映・design-D4 修正）:
//   RebuildThread ownership（jassert ガード付き）を強制するオブジェクト
//   （DeferredPublishView / RuntimePublicationOrchestrator の deferred 系）は
//   AudioEngineHarness 上の Integration Test で検証する。
//   Standalone 単体テスト（main() + add_executable/add_test）は純粋 Policy
//   （PublicationAdmission::evaluateDeferred 等）にのみ予約する。

#pragma once

#include "audioengine/AudioEngine.h"
#include "audioengine/RuntimePublicationOrchestrator.h"
#include "audioengine/AtomicAccess.h"

class DeferredPublicationTestAccess final
{
public:
    static convo::isr::RuntimePublicationOrchestrator& orchestrator(AudioEngine& e) noexcept
    {
        return *e.runtimeOrchestrator_;
    }

    // Option-2 hook: DeferredFadingActive の「前提条件」（published world に
    // fading runtime が存在）を決定論的に作る。Production の Decision 判定
    // ロジックは一切変更しない（PublicationAdmission.cpp evaluate の
    // if (hasFading) → DeferredFadingActive 分岐はそのまま）。
    static void setFadingRuntimePresent(AudioEngine& e, bool on) noexcept
    {
        convo::publishAtomic(e.testFadingRuntimePresent_, on, std::memory_order_release);
    }

    // ★ STG-8: Coordinator 観測（liveLogicalRecoveryObligationCount /
    //   recoveryRetryRedriveCount）。AudioEngine の friend のため private 到達可能。
    static convo::isr::RuntimeIntentCoordinator& coordinator(AudioEngine& e) noexcept
    {
        return e.runtimePublicationBridge_;
    }

    // ★ STG-8-D3: retire-pressure throttle の test-only 設定。
    //   PublicationAdmission::evaluate の Pressure 分岐（:40-48）を決定論的に発火させる。
    static void setRetirePressureThrottle(AudioEngine& e, bool on) noexcept
    {
        convo::publishAtomic(e.retirePressurePublicationThrottleActive_, on, std::memory_order_release);
    }

    // ★ STG-8-D1: rebuild generation の test-only  bump。
    //   新規 build の submit を伴わないため、deferred slot の generation-stale Discard を
    //   決定論的に発火させる（evaluateDeferred :82-84）。
    static void bumpRebuildGeneration(AudioEngine& e) noexcept
    {
        e.rebuildRequestGeneration.fetch_add(1, std::memory_order_acq_rel);
    }

    // ===== STG-9-D1 / RC-1: reclaim accounting 観測（production 変更ゼロ）=====
    //   AudioEngine は既に friend のため private へ到達可能。テスト専用 accessor のみで、
    //   production ヘッダ・production ロジック・可視性は一切変更しない。

    // outstanding deferred reclaim identity 数（RC-1 INV-1 の左辺）。
    static std::size_t pendingReclaimCount(AudioEngine& e) noexcept
    {
        std::lock_guard<std::mutex> lock(e.pendingReclaimHandlesMutex_);
        return e.pendingReclaimHandles_.size();
    }

    // reclaimInFlightCount_（Coordinator の近似カウンタ。RC-1 INV-1 の右辺）。
    static std::uint64_t reclaimInFlightCount(AudioEngine& e) noexcept
    {
        return e.runtimePublicationBridge_.getReclaimInFlightCount();
    }

    // DSPHandleRuntime へのテスト専用参照。STG-9-D1 の terminal sink
    // （AudioEngine::quarantineSlot の Step 3 と同じ DSPHandleRuntime::quarantineSlot）に
    // 生産コードと同一の経路で到達するために使う。
    static convo::isr::DSPHandleRuntime& handleRuntime(AudioEngine& e) noexcept
    {
        return e.dspHandleRuntime_;
    }

    // ===== HD-1 (V-11 handle lifetime): production destroy path, no handle retire =====
    //   Production frees a DSPCore only after retiring its handle (DSPLifetimeManager) or
    //   after proving shutdown quiescence. This hook runs the same destroy function without
    //   retiring, which is the state the 2026-10-25 crash dump showed (slot Active, memory
    //   freed). Used only by V11EngineRetireProtectionTests (HD-1).
    static void destroyCoreNodeWithoutRetire(void* core) noexcept
    {
        AudioEngine::destroyDSPCoreNode(core);
    }

    // ===== F1' (pin) observation =====
    //   pinnedCount はスロットの pin 数を返す。F1-2 / F1-3 が
    //   「lease 保持中は 0 でない」「解除後は 0」を直接観測するために使う。
    static std::uint32_t pinnedCount(AudioEngine& e, std::uint32_t slot) noexcept
    {
        return e.dspHandleRuntime_.pinnedCount(slot);
    }

    //   pin 受付を閉じる（破棄側の第 1 段階をテストから単独実行するため）。
    static bool closePins(AudioEngine& e, const convo::isr::DSPHandle& h) noexcept
    {
        return e.dspHandleRuntime_.closePinsForDestroy(h.slot, h.generation);
    }

    //   期限付きで pin の消滅を待つ（破棄側の第 3 段階）。false = 待ちきれず **未解放**。
    static bool invalidateHandleBeforeDestroy(AudioEngine& e, void* core) noexcept
    {
        return e.invalidateHandleBeforeDestroy(core);
    }

    // F1'-b (deferred destroy) observation
    static std::size_t deferredDestroyResidentCount(AudioEngine& e) noexcept
    {
        return e.deferredDestroyResidentCount();
    }
    // F1'-b: cumulative deferral count (monotonic, not racy with the retry swap).
    static std::uint64_t deferredDestroyCount(AudioEngine& e) noexcept
    {
        return e.deferredDestroyCount();
    }
    // F1'-b production-path test: the D queue only destroys entries whose epoch is OLDER
    // than the min-reader epoch, so a retire issued at the current epoch is parked and no
    // deleter runs. The test must advance the publication epoch (exactly what the audio
    // thread does with EpochDomain::publishEpoch) to make the real drain attempt the destroy.
    static std::uint64_t publishEpochForTest(AudioEngine& e) noexcept
    {
        return e.m_retireRouter->publishEpoch();
    }
    static std::uint64_t currentEpochForTest(AudioEngine& e) noexcept
    {
        return e.m_retireRouter->currentEpoch();
    }
    static std::uint64_t minReaderEpochForTest(AudioEngine& e) noexcept
    {
        return e.m_retireRouter->minReaderEpoch();
    }
    static std::size_t pendingRetireCountForTest(AudioEngine& e) noexcept
    {
        return e.m_retireRouter->pendingRetireCount();
    }
    static std::uint32_t residentAtomicForTest(AudioEngine& e) noexcept
    {
        return e.m_retireRouter->residentCountAtomic();
    }
    // F-2 (2026-10-25): quarantine-slot destruction honours the pin contract.
    static bool destroyQuarantineSlotForTest(AudioEngine& e, std::uint32_t slot,
                                             std::uint64_t generation) noexcept
    {
        return e.dspHandleRuntime_.destroyQuarantineSlot(slot, generation);
    }
    static void quarantineSlotForTest(AudioEngine& e, std::uint32_t slot) noexcept
    {
        e.dspHandleRuntime_.quarantineSlot(slot);
    }
    static std::size_t retryDeferredQuarantineForTest(AudioEngine& e) noexcept
    {
        return e.dspHandleRuntime_.retryDeferredQuarantineDestroy();
    }
    static std::uint32_t deferredQuarantineCountForTest(AudioEngine& e) noexcept
    {
        return e.deferredQuarantineDestroyCount();
    }
    // G (2026-10-25): drive the deferred-reclaim queue deterministically.
    static void drainDeferredRetireQueuesForTest(AudioEngine& e) noexcept
    {
        e.drainDeferredRetireQueues(true);
    }

    // G (2026-10-26): retirement epoch of the oldest pending reclaim identity. The safety
    // predicate is isOlder(retireEpoch, minReaderEpoch); exposing it lets a test assert the
    // PROPERTY (reclaim only after min passed the retire epoch) instead of assuming that no
    // other reader exists in the process.
    static std::uint64_t pendingRetireEpochForTest(AudioEngine& e) noexcept
    {
        std::lock_guard<std::mutex> lock(e.pendingReclaimHandlesMutex_);
        return e.pendingReclaimHandles_.empty()
            ? 0 : e.pendingReclaimHandles_.front().retireSequence;
    }
    static std::size_t retryDeferredDestroys(AudioEngine& e) noexcept
    {
        return e.retryDeferredDestroys();
    }

    // F1'-b 監査対応: 通常 tick（早期 return を含む）をテストから叩く。
    static void tickReclaimForTest(AudioEngine& e) noexcept
    {
        e.tryReclaimResources();
    }
    static std::uint32_t deferredResidentAtomic(AudioEngine& e) noexcept
    {
        return e.deferredDestroyResidentAtomic();
    }
    static std::size_t terminalDeferredResidentCount(AudioEngine& e) noexcept
    {
        return e.m_retireRouter->terminalDeferredResidentCount();
    }
    static std::size_t terminalRetryableResidentCount(AudioEngine& e) noexcept
    {
        return e.m_retireRouter->terminalRetryableResidentCount();
    }
    // E (2026-10-25): Deferred ownership transfers to the callee.
    static std::uint64_t terminalOwnershipTransferredCount(AudioEngine& e) noexcept
    {
        return e.m_retireRouter->terminalOwnershipTransferredCount();
    }

    // F1'-b: TerminalReclaimAuthority の drain をテストから駆動する
    //   （drainTerminalReclaim は private のため、公開 tryReclaim 経由で呼ぶ）。
    static void drainTerminalForTest(AudioEngine& e) noexcept
    {
        e.tryReclaimResources();
    }

    // F1'-b: drainAll() をテストから駆動する（shutdown 相当の強制 drain）。
    static void drainAllTerminalForTest(AudioEngine& e) noexcept
    {
        e.m_retireRouter->drainAllTerminalForTest();
    }
    // 再試行可能な破棄を直接駆動する（D/Q/E を経由せず、T の契約だけを見る）。
    // ★ 監査 B-1 対応: ルータの terminalReclaimRetryable は「Deferred を格納しない」
    //   契約に変わった（所有権の一元化）。Terminal 単体の Deferred 保持を検証するには
    //   storeRetryable を直接叩く必要がある。
    static bool storeRetryableDirectForTest(AudioEngine& e, void* ptr,
                                            RetryableDeleter retryable,
                                            std::uint64_t epoch) noexcept
    {
        return e.m_retireRouter->storeRetryableDirectForTest(ptr, retryable, epoch);
    }

    // ルータ API 経由（所有権一元化の検証用）。
    static bool storeRetryableForTest(AudioEngine& e, void* ptr,
                                      RetryableDeleter retryable,
                                      std::uint64_t epoch) noexcept
    {
        return e.m_retireRouter->terminalReclaimRetryable(
            ptr, retryable, epoch, DeletionEntryType::Generic, "f1b-test");
    }

    // R8-4: control the shutdown-in-progress predicate so the clear path can be exercised
    // in the ShutdownDiscarded branch, and restore the original state afterwards.
    // EngineLifecycleState: 0 Unprepared, 1 Preparing, 2 Prepared, 3 Releasing, 4 Destroyed
    // (isShutdownInProgress() is true for 3 and 4).
    static int getLifecycleStateRaw(AudioEngine& e) noexcept
    {
        return static_cast<int>(convo::consumeAtomic(e.lifecycleState, std::memory_order_acquire));
    }

    static void setLifecycleStateRaw(AudioEngine& e, int v) noexcept
    {
        convo::publishAtomic(e.lifecycleState,
                             static_cast<AudioEngine::EngineLifecycleState>(v),
                             std::memory_order_release);
    }

    // pendingReclaimHandles_ 内に指定 handle と一致する entry があるか。
    // STG-9-D1 の test-induced identity の presence / terminal-drop を直接検証する。
    // （slot 世代再利用により stale entry が同居し得るため、global 件数だけでは
    //   test-induced identity の終端を特定できない。）
    static bool pendingContains(AudioEngine& e, const convo::isr::DSPHandle& h) noexcept
    {
        std::lock_guard<std::mutex> lock(e.pendingReclaimHandlesMutex_);
        for (const auto& id : e.pendingReclaimHandles_)
        {
            if (id.handle == h)
                return true;
        }
        return false;
    }

    // ===== STG-11-D3: affinity failure observation（production 変更ゼロ）=====
    //   AudioEngine の friend のため private 到達可能。record/report は
    //   production の private member であり、test からのみ friend 経由で呼ぶ。
    //   RT 側の record は OS 呼び出しを伴わないため、合成値で直接検証できる。

    // RT-side record と同一の記録を実行する（synthetic kind/mask/err）。
    //   OS failure injection を行わずに transport 両端を検証するため、
    //   production と同じ private 関数を合成値で駆動する。
    static void recordAffinityFailure(AudioEngine& e, std::uint32_t kind, std::uint64_t mask,
                                      std::uint32_t err) noexcept
    {
        e.recordAffinityFailure(kind, static_cast<DWORD_PTR>(mask), static_cast<DWORD>(err));
    }

    // NonRT diagnosis を実行する（timerCallback と同一の入口）。
    static void reportAffinityFailure(AudioEngine& e) noexcept
    {
        e.reportAffinityFailureIfRecorded();
    }

    static bool affinityFailureObserved(AudioEngine& e) noexcept
    {
        return convo::consumeAtomic(e.affinityFailureObserved_, std::memory_order_acquire);
    }

    static std::uint64_t affinityFailureCount(AudioEngine& e) noexcept
    {
        return convo::consumeAtomic(e.affinityFailureCount_, std::memory_order_acquire);
    }

    static std::uint64_t affinityFailureLastMask(AudioEngine& e) noexcept
    {
        return convo::consumeAtomic(e.affinityFailureLastMask_, std::memory_order_acquire);
    }

    static std::uint32_t affinityFailureLastError(AudioEngine& e) noexcept
    {
        return convo::consumeAtomic(e.affinityFailureLastError_, std::memory_order_acquire);
    }

    static std::uint32_t affinityFailureLastKind(AudioEngine& e) noexcept
    {
        return convo::consumeAtomic(e.affinityFailureLastKind_, std::memory_order_acquire);
    }

    static std::uint64_t affinityFailureReportedCount(AudioEngine& e) noexcept
    {
        return convo::consumeAtomic(e.affinityFailureReportedCount_, std::memory_order_acquire);
    }

    // ===== STG-11-D4: success observation（production 変更ゼロ）=====
    //   D3 と同一の friend 経路。record/report は production の private member。
    //   RT 側の record は OS 呼び出しを伴わないため、合成値で直接検証できる。

    // RT-side record と同一の記録を実行する（synthetic kind/a/b/c）。
    static void recordSuccessObserved(AudioEngine& e, std::uint32_t kind,
                                      std::uint64_t a, std::uint64_t b,
                                      std::uint64_t c) noexcept
    {
        e.recordSuccessObserved(kind, a, b, c);
    }

    // NonRT diagnosis を実行する（timerCallback と同一の入口）。
    static void reportSuccess(AudioEngine& e) noexcept
    {
        e.reportSuccessIfRecorded();
    }

    static bool successObserved(AudioEngine& e) noexcept
    {
        return convo::consumeAtomic(e.successObserved_, std::memory_order_acquire);
    }

    static std::uint64_t successCount(AudioEngine& e) noexcept
    {
        return convo::consumeAtomic(e.successCount_, std::memory_order_acquire);
    }

    static std::uint32_t successKind(AudioEngine& e) noexcept
    {
        return convo::consumeAtomic(e.successKind_, std::memory_order_acquire);
    }

    static std::uint64_t successA(AudioEngine& e) noexcept
    {
        return convo::consumeAtomic(e.successA_, std::memory_order_acquire);
    }

    static std::uint64_t successB(AudioEngine& e) noexcept
    {
        return convo::consumeAtomic(e.successB_, std::memory_order_acquire);
    }

    static std::uint64_t successC(AudioEngine& e) noexcept
    {
        return convo::consumeAtomic(e.successC_, std::memory_order_acquire);
    }

    static std::uint64_t successReportedCount(AudioEngine& e) noexcept
    {
        return convo::consumeAtomic(e.successReportedCount_, std::memory_order_acquire);
    }

    // ===== STG-11-D5: MMCSS event observation（production 変更ゼロ）=====
    //   D3/D4 と同一の friend 経路。record/report は production の private member。

    static void recordMmcssEvent(AudioEngine& e, std::uint32_t kind,
                                 std::uint64_t a, std::uint64_t b,
                                 std::uint64_t c, std::uint64_t d) noexcept
    {
        e.recordMmcssEventObserved(kind, a, b, c, d);
    }

    // NonRT diagnosis を実行する（timerCallback と同一の入口）。
    static void reportMmcssEvent(AudioEngine& e) noexcept
    {
        e.reportMmcssEventIfRecorded();
    }

    static bool mmcssObserved(AudioEngine& e) noexcept
    {
        return convo::consumeAtomic(e.mmcssObserved_, std::memory_order_acquire);
    }

    static std::uint64_t mmcssCount(AudioEngine& e) noexcept
    {
        return convo::consumeAtomic(e.mmcssCount_, std::memory_order_acquire);
    }

    static std::uint32_t mmcssKind(AudioEngine& e) noexcept
    {
        return convo::consumeAtomic(e.mmcssKind_, std::memory_order_acquire);
    }

    static std::uint64_t mmcssA(AudioEngine& e) noexcept
    {
        return convo::consumeAtomic(e.mmcssA_, std::memory_order_acquire);
    }

    static std::uint64_t mmcssB(AudioEngine& e) noexcept
    {
        return convo::consumeAtomic(e.mmcssB_, std::memory_order_acquire);
    }

    static std::uint64_t mmcssC(AudioEngine& e) noexcept
    {
        return convo::consumeAtomic(e.mmcssC_, std::memory_order_acquire);
    }

    static std::uint64_t mmcssD(AudioEngine& e) noexcept
    {
        return convo::consumeAtomic(e.mmcssD_, std::memory_order_acquire);
    }

    static std::uint64_t mmcssReportedCount(AudioEngine& e) noexcept
    {
        return convo::consumeAtomic(e.mmcssReportedCount_, std::memory_order_acquire);
    }

    // ===== R8 (P1): RebuildThread lifecycle + ownership, for the C1 ordering and
    //   Release-enforcement tests (production change: none).
    static bool rebuildThreadShouldExitOf(AudioEngine& e) noexcept
    {
        return convo::consumeAtomic(e.rebuildThreadShouldExit, std::memory_order_acquire);
    }

    static bool rebuildThreadStoppedOf(AudioEngine& e) noexcept
    {
        return convo::consumeAtomic(e.rebuildThreadStopped, std::memory_order_acquire);
    }

    static bool isRebuildThreadOf(AudioEngine& e) noexcept
    {
        return e.isRebuildThread();
    }

    static bool rebuildThreadIdIsEmpty(AudioEngine& e) noexcept
    {
        return e.rebuildThreadId() == std::thread::id{};
    }

    // ★ R8 (P1): the ownership token the worker publishes on entry and clears on exit.
    static std::uint64_t rebuildThreadTokenOf(AudioEngine& e) noexcept
    {
        return convo::consumeAtomic(e.rebuildThreadToken_, std::memory_order_acquire);
    }

    // ===== D8-2-B-1 T10-c: the PRODUCTION provider used by SnapshotCoordinator =====
    //   AudioEngine constructs SnapshotCoordinator with m_epochDomain
    //   (AudioEngine.CtorDtor.cpp:28), NOT with the retire router. That distinction decides
    //   T10's RT-boundary question: EpochDomain::enqueueRetire is the lock-free MPSC ring
    //   (DeferredDeletionQueue::enqueue), while ISRRetireRouter::enqueueRetire may escalate to
    //   tryReclaim() -> deleter -> free on queue pressure. T10-c must therefore be observed
    //   against EpochDomain.
    static convo::EpochDomain& epochDomain(AudioEngine& e) noexcept
    {
        return e.m_epochDomain;
    }

    // ===== D8-2-B-1 T9: DSP delete-count observability =====
    //   destroyDSPCoreNodeRetryable() is the ONLY place that destroys a DSPCore
    //   (`core->~DSPCore(); convo::aligned_free(core);`, AudioEngine.Threading.cpp:39-42),
    //   and it reports whether the object was actually destroyed (Destroyed) or retained
    //   because a pin was still held (Deferred). Exposing it here lets a test count REAL DSP
    //   destructions through the production funnel without adding a production counter:
    //   DSPLifetimeManager hardcodes &AudioEngine::destroyDSPCoreNode as the deleter, so a
    //   counting deleter cannot be injected from outside (T9 option (a) is not available as
    //   written; this accessor is the test-only equivalent).
    static DestroyOutcome destroyCoreNodeRetryable(void* core) noexcept
    {
        return AudioEngine::destroyDSPCoreNodeRetryable(core);
    }
};
