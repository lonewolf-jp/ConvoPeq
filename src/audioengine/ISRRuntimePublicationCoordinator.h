#pragma once
#include <atomic>
#include <cstring>  // ★ work88 (Phase 7): Intent default ctor の union ゼロ初期化 (std::memset)
#include <memory>
#include <cstdint>
#include <type_traits>
#include <array>       // ★ D105-R5-8: RecoveryAdmissionTable
#include <optional>  // ★ FUTURE-3: popRecoveryRequest() return type
#include "ISRClosure.h"
#include "ISRPayloadTier.h"
#include "ISRSealedObject.h"
#include "ISRRetire.h"
#include "ISRHB.h"
#include "ISRShutdown.h"
#include "ISRRuntimeSemanticSchema.h"
#include "ISRAuthorityClass.h"
#include "ISRRetireRouter.h"
#include "ISRRetireOverflowRing.h"     // ★ Phase5: RetireOverflowEntry
#include "../LockFreeRingBuffer.h"     // ★ Phase5: coordinatorDeferredRing_
#include "../MpscBoundedRing.h"        // ★ FUTURE-10 (work88): intentQueue_ の MPSC 化 (Vyukov bounded)
#include "ISRDSPHandle.h"              // ★ P0-5: QuarantineService needs full DSPHandle
#include "RuntimeBuildTypes.h"          // ★ FUTURE-3 (work88): RuntimeBuildSnapshot (RecoveryIntent::buildSource 値コピー)

// ★ P0-4A: DSPLifetimeManager は global scope（DSPLifetimeManager.h 参照）
//   processIntent の完全定義には DSPLifetimeManager.h の include が必要。
//   ただし .h での include は循環依存防止のため、global 前方宣言＋.cpp で include する。
class DSPLifetimeManager;
class AudioEngine;
// ★ dash2 §1.7 (Phase G CW-3b): commit() の monotonicity baseline 型（確立済み semantic type）。
//   RuntimeState は global scope で定義（AudioEngine.h:140 — convo::isr::SealedObject 継承）。
//   前方宣言のみ（循環 include 回避）。
struct RuntimeState;

namespace convo::isr {

// ★ P0-4C: 前方宣言（完全定義は ISRDSPQuarantine.h）
enum class QuarantineReason : int;
class DSPQuarantineManager;

// ★ P0-5: QuarantineService — State変更 + Audit を単一トランザクションとして実行
//   QSVC-1: State変更 + Audit を単一トランザクション。
//   QSVC-3: State + Audit の整合性を保証。
//   QSVC-5: 失敗時は State + Audit + Receipt の3状態をロールバック。
class QuarantineService {
public:
    struct QuarantineRequest {
        DSPHandle handle;
        QuarantineReason reason;
        uint64_t contextEpoch;
    };

     struct QuarantineResult {
         bool stateChanged{false};
         bool auditLogged{false};
         // ★ FUTURE-3/QSVC-5: rolledBack 削除。Audit 失敗→State 不変（Publish後は Immutable）。Rollback 禁止。
     };

    QuarantineResult executeQuarantine(
        DSPHandleRuntime& handleRuntime,
        DSPQuarantineManager& quarantineManager,
        const QuarantineRequest& request) noexcept;
};

enum class PublishAuthority : uint8_t { Granted = 1 };
enum class RetireAuthority : uint8_t { Granted = 1 };
enum class ShutdownAuthority : uint8_t { Granted = 1 };

enum class RuntimeBoundary : uint8_t {
    RTWorld,
    NonRTWorld
};

// ★ work88 (X1〜X6 §6.9 Phase 0 / 二十五次レビュー): INV-ISR-01〜07 — ISR 全体の最上位不変条件。
//   （コード契約として固定 — Phase 0 invariant freeze / X_IMPL_CHECKLIST #2）
//   INV-ISR-01: isFullyDrained == true は以下を意味する:
//     all producers stopped AND all producer joins completed AND all transport queues empty
//     AND all deferred state empty AND all reclaim-in-flight == 0 AND all reader inactive
//     AND reader registration closed
//   INV-ISR-02: pendingIntentCount_ は queue size ではなく transport residency + producer
//     reservation である（residency + reservation — 二重計上禁止）
//   INV-ISR-03: 異なる semantic state を一つの counter で表現しない。特に Intent / DSP
//     resident / Retire resident を混ぜない（§6.6 X6 の4層分離と整合）
//   INV-ISR-04: ShutdownQuiescent reclaim は readerRegistrationClosed なしでは絶対に許可しない
//     （§6.3 X3 と整合）
//   INV-ISR-05: completion watermark を publication committed と同一視しない（§6.2 X2 と整合）
//   INV-ISR-06: 退役・ownership の identity source は publish() の oldWorld / Lifetime であり、
//     RuntimeStore::current は published-world read の単一 source（旧 currentWorld_ は CW-3c で削除）
//   INV-ISR-07: RuntimeStore::current の RuntimeState::publication identity が publish transaction
//     全体で整合（bake は publishAndSwap 前に実行 — 単一 source・INV-X4-6 更新版）
class RuntimeIntentCoordinator {
public:
    enum class CoordinatorState : uint8_t {
        Bootstrapping = 0,
        Ready,
        Publishing,
        Transitioning,
        Pressure,
        ShuttingDown,
        Faulted
    };

    RuntimeIntentCoordinator();
    bool precheckPublish(const PayloadClosureDescriptor& closure,
                         const TieredPayloadDescriptor& descriptor) noexcept;
    const char* lastRejectReason() const noexcept;
    void commit(PublishAuthority, RuntimeBoundary boundary, const void* newWorld, std::uint64_t version);
    void commit(PublishAuthority,
                RuntimeBoundary boundary,
                const void* newWorld,
                std::uint64_t version,
                PublicationSequenceId sequenceId,
                PublicationEpoch epoch,
                std::uint64_t mappedGeneration,
                const RuntimeState* prevWorld);
    void retire(RetireAuthority, RuntimeBoundary boundary, const void* oldWorld);
    [[nodiscard]] RetireEnqueueResult enqueueRetire(RetireAuthority auth,
                                                      ISRRetireRouter& router,
                                                      void* ptr,
                                                      void (*deleter)(void*),
                                                      std::uint64_t epoch) noexcept;
    [[nodiscard]] std::uint64_t retireAuthorityCount() const noexcept;
    // ★ dash2 §1.7 (Phase G CW-3c): getCurrent/getVersion/currentPublicationEpoch/currentPublicationSequenceId
    //   は production caller ゼロのため削除。published-world read は RuntimeStore::current が単一 source。

    // ── dash2 §1.4 (REPAIR_PLAN2-dash2): semantic event accounting ──
    //   外部 setter（setRetireBacklogCount 等）の廃止に伴い、Coordinator は自身のカウンタを
    //   semantic event API で原子的に維持する（fetch_add/fetch_sub・underflow ガード付き）。
    //   呼び出し元（AudioEngine）は setter で絶対値を上書きせず、本イベントで増減のみを通知する。
    //   ⚠️ isFullyDrained の retire/fallback/deferred 判定は AudioEngine 側（Layer 1）が
    //   実測値（m_retireRouter->pendingRetireCount() 等）を直接判定する（dash2 §1.4 設計方針）。
    //   ［本 API は NonRT-thread からのみ呼び出すこと（AC-ISR-1）］
    void onRetireAccepted() noexcept;      // retire backlog +1（atomic fetch_add + pressure 更新）
    void onRetireConsumed() noexcept;      // retire backlog -1（underflow ガード付き fetch_sub）
    void onReclaimBegin() noexcept;        // reclaim in-flight +1
    void onReclaimEnd() noexcept;          // reclaim in-flight -1（underflow ガード付き）
    // ★ D101-32-D: onFallbackAccepted/Consumed / onDeferredRetireAccepted/Consumed は削除済み。
    //   対応 counter（fallbackBacklogCount_ / deferredRetireResidencyCount_）と共に vestigial 判定
    //   （D101-32-C §4/§5）。fallback/deferred の実測は Layer 1 実測 + queue emptiness が担当。

    // ⚠️ 旧 setter API 群 — dash2 §1.4 により production からの呼び出しは全廃。
    //   残存するのはテスト初期化リセット（P2 教訓: テストでのリセットは許可）のみ。
    //   production からの絶対値上書きは禁止（コンパイル時参照 = 0 を維持すること）。
    //   ★ D101-32-D: setFallbackBacklogCount / setReclaimInFlightCount /
    //     setDeferredRetireResidencyCount / setQuarantineResidentCount は削除済み
    //     （D101-32-C §7 削除境界）。setRetireBacklogCount は Pressure FSM / drain violation
    //     の決定論的テスト駆動のため KEEP（絶対値注入がテストの意味本体）。
    void setRetireBacklogCount(std::uint64_t count) noexcept;        // TEST-ONLY（KEEP — Pressure FSM 駆動）
    void setPublicationBacklogCount(std::uint64_t count) noexcept;   // TEST-ONLY（dead counter）
    void setPendingIntentCount(std::uint64_t count) noexcept;        // TEST-ONLY
    void escalateAllRetires(RetirePriority minPriority) noexcept;    // ★ Phase5: 全RetireIntent の優先度を底上げ
    void setOverflowMaxAgeUs(std::uint64_t maxAgeUs) noexcept;       // ★ Phase5: OverflowRing 滞留年限警告しきい値
    void setSwapPending(bool pending) noexcept;
    [[nodiscard]] bool isSwapPending() const noexcept;
    // ★ A-2.4: getter 群（DrainAudit 用）
    [[nodiscard]] std::uint64_t getPublicationBacklogCount() const noexcept;
    // ★ work88 (X5 §6.5): Publish Intent residency counter（INV-X5-1）。isFullyDrained / 診断用。
    [[nodiscard]] std::uint64_t getPublicationIntentResidencyCount() const noexcept;
    [[nodiscard]] std::uint64_t getPendingIntentCount() const noexcept;
    [[nodiscard]] std::uint64_t getRetireBacklogCount() const noexcept;
    // ★ D101-32-D: getFallbackBacklogCount / getDeferredRetireResidencyCount /
    //   getQuarantineResidentCount（Coordinator側）は削除済み（D101-32-C §7 — vestigial counter
    //   の getter も domain 一括で除去）。実在 quarantine DSP は DSPQuarantineManager::residentCount()、
    //   Q+EmergencyQ は EpochControl::getQuarantineResidentCount()（別クラス・別semantic）が authority。
    // ★ work88 (X6 §6.6): Quarantine transport residency counter（INV-X6-4）。診断 / isFullyDrained 用。
    [[nodiscard]] std::uint64_t getQuarantineIntentResidencyCount() const noexcept;
    [[nodiscard]] std::uint64_t getQuarantineRingResidencyCount() const noexcept;
    // ★ work88 (FUTURE-10): Quarantine fallback ring の drop 回数（静かに破棄しない証跡）。
    //   AudioEngine 側 HealthMonitor が監視し ISRHealthState::Critical 昇格を駆動する。
    [[nodiscard]] std::uint64_t quarantineFallbackDropCount() const noexcept
    {
        return convo::consumeAtomic(quarantineFallbackDropCount_, std::memory_order_acquire);
    }
    // ★ work88 (六次レビュー — INV-5): Recovery Intent push 失敗（queue full）時の drop 回数。
    //   AudioEngine 側 HealthMonitor が監視し ISRHealthState 昇格を駆動する（静かな破棄を禁止）。
    [[nodiscard]] std::uint64_t recoveryIntentDropCount() const noexcept
    {
        return convo::consumeAtomic(recoveryIntentDropCount_, std::memory_order_acquire);
    }
    // ★ work88 (P2-4 監査補正 — Step B/C): shutdown 時（AdmissionClosed）に Recovery を
    //   意図的に破棄した回数（ShutdownDiscard）。queue full による drop（recoveryIntentDropCount）
    //   とは区別する — dash §8.1 の X1 telemetry 分離方針（Recovery lost ≠ ShutdownDiscard）。
    //   正常 shutdown 動作のため Critical 昇格対象外（getter 公開で観測可能に留める）。
    [[nodiscard]] std::uint64_t recoveryShutdownDiscardCount() const noexcept
    {
        return convo::consumeAtomic(recoveryShutdownDiscardCount_, std::memory_order_acquire);
    }
    [[nodiscard]] std::uint64_t getReclaimInFlightCount() const noexcept;
    [[nodiscard]] std::uint64_t getOverflowMaxAgeUs() const noexcept;          // ★ Phase5
    [[nodiscard]] bool isFullyDrained() const noexcept;
    [[nodiscard]] CoordinatorState getState() const noexcept;
    void markTransitionStart() noexcept;
    void markTransitionCommitted() noexcept;
    void requestShutdown() noexcept;
    void markShutdownComplete() noexcept;

    // ── ★ P0-4C: ISR Intent 発行インターフェース ──
    //   OBSERVE-1: Timer → submitObserve → Coordinator が retirePublishedDSP を起動
    //   QSVC-2:    Coordinator は QuarantineService を介さず直接 quarantine を呼ばない
    //   DELETE-1:  reclaim() は Coordinator 専用。外部からの直接呼び出し禁止。

    /// Observe Intent: Timer から定期観測要求を発行する。
    /// Coordinator は Intent Queue に追加し、非同期に処理する。
    /// OBSERVE-1〜8 に従い、Timer はこのメソッドのみを呼び出す。
    /// handle: 観測対象の DSPHandle（processIntent が retire する DSP を識別するために使用）
    void submitObserve(const DSPHandle& handle, PublicationEpoch epoch) noexcept;

    /// Quarantine Intent: 指定された DSPHandle を quarantine する要求を発行する。
    /// QSVC-2: Coordinator は QuarantineService 経由で quarantine を実行する。
     void submitQuarantine(const DSPHandle& handle,
                               QuarantineReason reason,
                               DSPHandleRuntime& handleRuntime,
                               DSPQuarantineManager& quarantineManager,
                               uint64_t contextEpoch = 0) noexcept;

     // ── ★ FUTURE-3: Recovery Intent (transport-only payload) ──
     //   submitRecoveryRequest() は enqueue のみ。pop は Builder Loop (FUTURE-10 共通 Intent Queue へ移行)。
     //   Decision Authority を持たない: push/pop 以外の意味なし（復旧 World build は Builder 側）。
     struct RecoveryIntent {
         DSPHandle handle;            // recovery 対象（quarantined DSPHandle）
         PublicationEpoch epoch;      // emit 時の publicationEpoch（FIFO/epoch 検証用）
         uint64_t intentId;           // 診断・モニタリング用シーケンス番号
        uint64_t obligationId{0};     // ★ D105-R5-8: logical recovery obligation id (allocated by Coordinator)
         // ★ FUTURE-3 (work88): build spec を値コピーで内包（POD、trivially copyable）。
         //   quarantinedHandle 単独では resolve() 不能（ISRDSPHandle.cpp:69）なため、build 入力は
         //   値コピーした snapshot から引当する（epoch 逆引き不要 — lifetime を構造的に解決）。
         //   IR data は内包しない（四次実測: RuntimeBuildSnapshot に IR AudioBuffer は無い）。
         //   IR 実体は build 時に transferIRStateFrom(engine.getConvolverProcessor()) で現在値取得
         //   （Recovery semantic = quarantined 除外した現在のユーザー構成の再構築）。
         //   ConvolverProcessor::BuildSnapshot は juce::File/String を含み POD でないため内包しない
         //   （五次レビュー案 i — build 時に uiConvolverProcessor.captureBuildSnapshot() から取得）。
         convo::RuntimeBuildSnapshot buildSource;
     };
     static_assert(std::is_trivially_copyable_v<RecoveryIntent>,
         "RecoveryIntent must be trivially copyable for LockFreeRingBuffer");
     static_assert(std::is_standard_layout_v<RecoveryIntent>,
         "RecoveryIntent must be standard layout for LockFreeRingBuffer");
    static_assert(std::is_trivially_copyable_v<convo::RuntimeBuildSnapshot>,
        "FUTURE-3: RuntimeBuildSnapshot must be trivially copyable to embed in RecoveryIntent");

    // ── ★ D105-R5-8: Logical Recovery Obligation (identity + capacity accounting) ──
    //   Coordinator-owned logical layer (recoveryAdmissions_ table) enforcing liveLogicalRecoveryObligationCount ≤ 32
    //   (INV-X1-7 / INV-CAP-7). Distinct from the single-slot durable *transport* fallback
    //   (pendingRecoveryAdmission_) which is preserved unchanged.
    using LogicalRecoveryObligationId = std::uint64_t;

    // Semantic recovery target — derived from build fingerprint; basis for coalescing
    // (R5-rev1 Option 1: existing O preserved, new attempt coalesced — ΔL = 0).
    struct SemanticRecoveryTarget {
        std::uint64_t irIdentityHash = 0;
        std::uint64_t convolutionConfigHash = 0;
        std::uint64_t dspParameterHash = 0;
        bool operator==(const SemanticRecoveryTarget& o) const noexcept {
            return irIdentityHash == o.irIdentityHash
                && convolutionConfigHash == o.convolutionConfigHash
                && dspParameterHash == o.dspParameterHash;
        }
    };
    // ★ D105-R5-9 MUST-3: CoalesceIdentity = quarantinedHandle + SemanticRecoveryTarget.
    //   Equivalence requires BOTH handle and target to match, so two distinct DSP handles that
    //   happen to share the same build fingerprint are NOT coalesced (R8 §4 counterexample:
    //   (H1,T) != (H2,T) => 2 obligations).
    struct CoalesceIdentity {
        DSPHandle quarantinedHandle{};
        SemanticRecoveryTarget target{};
        bool operator==(const CoalesceIdentity& o) const noexcept {
            return quarantinedHandle == o.quarantinedHandle
                && target == o.target;
        }
    };

    enum class ObligationState : std::uint8_t {
        NoObligation = 0,
        Live,               // admitted, outstanding (transport queued / durable / building)
        ResolvedSuccess,
        ResolvedFailed,
        ResolvedStaleSuperseded,   // ★ D105-R5-9 MUST-4: stale-generation rejection (R7 §11)
        ResolvedRetry,
        ShutdownDiscarded
    };

    // ★ D105-R5-10: per-obligation delivery residency. Independent of ObligationState.
    //   None     = Live but NO delivery representation (deferred; re-drive target).
    //   Transport= intent is (or was) in recoveryIntentQueue_ (enqueued for the Builder).
    //   Durable  = held in the single durable slot (DurablePending / Building sub-states).
    //   Lets the re-drive rediscover ONLY Live obligations lacking a delivery representation
    //   (§5: do not re-send Transport/Durable obligations → no duplicate delivery). CoordinatorLoop-
    //   only field (written solely on the producer thread) — non-atomic by design.
    enum class ObligationDeliveryState : std::uint8_t {
        None = 0,
        Transport,
        Durable
    };

    // Resolution outcome for a logical recovery obligation. Retry keeps the obligation Live (ΔL=0,
    // durable re-armed); the rest are terminal (Live→terminal via the single Completion Authority).
    // NOTE: `Superseded` (R7 §11) is intentionally NOT implemented — it has no real supersession
    // transition in the current source, so it is left as a Phase-II item (do not fake a transition).
    enum class RecoveryOutcome : std::uint8_t {
        Published = 0,      // recovery publish succeeded (Route A trySubmitImpl / Route B onPublishCommitted)
        Failed,             // build/publish hard failure → terminal (no leak)
        StaleSuperseded,    // ★ D105-R5-9 MUST-4: RejectedStaleGeneration → terminal (−1)
        Retry,              // transient build failure → obligation stays Live (ΔL=0, rebuild retried)
        ShutdownDiscarded   // shutdown close (table-centric discard; L −1 via single authority)
    };
    using RecoveryResolution = RecoveryOutcome;   // resolution input to the Completion Authority

    // Lock-free, ISR-safe obligation record. id/state are atomic so the single completion
    // authority (resolveRecoveryObligation, callable from ISR) can transition Live→terminal
    // without a mutex. identity is immutable once admitted (written only by CoordinatorLoop).
    struct LogicalRecoveryObligation {
        std::atomic<LogicalRecoveryObligationId> id{0};
        CoalesceIdentity identity{};
        std::atomic<ObligationState> state{ObligationState::NoObligation};
        DSPHandle handle{};
        PublicationEpoch epoch{0};
        std::uint64_t intentId = 0;
        convo::RuntimeBuildSnapshot buildSource{};
        ObligationDeliveryState delivery{ObligationDeliveryState::None};   // ★ D105-R5-10
        // ★ D105-R18: per-obligation retry budget. Single writer (CoordinatorLoop via
        //   markTransientFailure) for transient failures; reset to 0 in resolve() on
        //   terminal disposition. Reaches kMaxObligationConsecutiveFailures → only
        //   sanctioned path to ResolvedFailed.
        std::atomic<std::uint8_t> consecutiveFailureCount{0};
    };
    static constexpr std::size_t kMaxLogicalRecoveryObligations = 32;  // INV-CAP-7
    // ★ D105-R18: obligation-level retry exhaustion bound. Inherited from the existing
    //   AudioEngine.RebuildDispatch.cpp:1015 kMaxRecoveryConsecutiveFailures (Builder-local
    //   spin-prevention). Same value (4) but at the obligation lifetime, surviving
    //   durable↔transport transitions. Exhaustion routes the obligation to ResolvedFailed
    //   (the only sanctioned Failed terminal per R17-4).
    static constexpr std::uint8_t kMaxObligationConsecutiveFailures = 4;

    // ── ★ D105-R5-8: Coordinator-owned Recovery Admission Table (logical obligation accounting) ──
    //   Enforces liveLogicalRecoveryObligationCount ≤ 32 (INV-X1-7 / INV-CAP-7). Distinct from the
    //   single-slot durable *transport* fallback (pendingRecoveryAdmission_). The table owns the only
    //   +1 (tryInsert, post-coalesce, L<Capacity) and the only −1 (resolve, Live→terminal CAS), so the
    //   invariants are structural, not by-convention.
    template <std::size_t Capacity>
    class RecoveryAdmissionTable {
    public:
        static constexpr std::size_t kCapacity = Capacity;

        // Find a Live obligation by coalesce key (for coalescing a new attempt onto an existing O).
        std::size_t findByKey(const CoalesceIdentity& key) const noexcept {
            for (std::size_t i = 0; i < kCapacity; ++i) {
                if (slots_[i].state.load(std::memory_order_acquire) == ObligationState::Live
                    && slots_[i].identity == key)
                    return i;
            }
            return kCapacity; // npos
        }

        // Single +1 site: allocate a new Live obligation (post-coalesce). Returns nullopt at capacity.
        std::optional<std::size_t> tryInsert(const CoalesceIdentity& key) noexcept {
            if (liveCount_.load(std::memory_order_acquire) >= kCapacity)
                return std::nullopt; // capacity exhausted → caller rejects (ΔL=0)
            for (std::size_t i = 0; i < kCapacity; ++i) {
                // a non-Live slot is reusable (fresh id==0, or terminal → reclaimed)
                if (slots_[i].state.load(std::memory_order_acquire) != ObligationState::Live) {
                    const LogicalRecoveryObligationId id = ++nextId_;
                    slots_[i].id.store(id, std::memory_order_relaxed);
                    slots_[i].identity = key;
                    slots_[i].state.store(ObligationState::Live, std::memory_order_release);
                    slots_[i].delivery = ObligationDeliveryState::None;   // ★ D105-R5-10: fresh slot has none
                    // ★ D105-R18: reset retry budget on slot reuse (slot is fresh — any prior terminal
                    //   left consecutiveFailureCount at its terminal value; reset for clean accounting).
                    slots_[i].consecutiveFailureCount.store(0, std::memory_order_release);
                    convo::fetchAddAtomic(liveCount_, std::uint64_t{1}, std::memory_order_release);
                    return i;
                }
            }
            return std::nullopt; // invariant guard (unreachable while L < Capacity)
        }

        // Single −1 authority. ★ D105-R5-9 MUST-3: id-based resolution — re-scans by id and CASes
        // state within the same atomic step (NO cached index from a prior findById). A reused slot is
        // always assigned a strictly-greater id (nextId_ is monotonic), so a reused slot can never match
        // a stale id → no ABA / no wrong-obligation termination. Idempotent: a prior terminal state (or a
        // lost CAS race) makes the inner CAS fail → returns false (no double −1, no L underflow).
        // ★ D105-R18: on successful Live→terminal CAS, reset consecutiveFailureCount to 0 (slot is
        //   freed; any prior counter value is meaningless post-terminal). Reset is a plain store
        //   *after* the CAS so it is observed only by readers after the terminal is visible.
        bool resolve(LogicalRecoveryObligationId id, ObligationState terminalState) noexcept {
            for (std::size_t i = 0; i < kCapacity; ++i) {
                if (slots_[i].id.load(std::memory_order_acquire) != id)
                    continue;
                ObligationState expected = ObligationState::Live;
                if (slots_[i].state.compare_exchange_strong(expected, terminalState, std::memory_order_acq_rel)) {
                    slots_[i].consecutiveFailureCount.store(0, std::memory_order_release);
                    convo::fetchSubAtomic(liveCount_, std::uint64_t{1}, std::memory_order_release);
                    return true;
                }
                return false; // already terminal (or lost race) → idempotent no-op
            }
            return false;     // unknown/mismatched id → no-op
        }

        std::uint64_t liveCount() const noexcept {
            return convo::consumeAtomic(liveCount_, std::memory_order_acquire);
        }
        const LogicalRecoveryObligation& slot(std::size_t i) const noexcept { return slots_[i]; }
        LogicalRecoveryObligation& slot(std::size_t i) noexcept { return slots_[i]; }
        // ★ D105-R18: read-only accessor for the retry counter (used by tests / telemetry).
        std::uint8_t consecutiveFailureCount(std::size_t i) const noexcept {
            return slots_[i].consecutiveFailureCount.load(std::memory_order_acquire);
        }

    private:
        std::array<LogicalRecoveryObligation, kCapacity> slots_{};
        std::atomic<std::uint64_t> liveCount_{0};
        LogicalRecoveryObligationId nextId_{1}; // single-writer (CoordinatorLoop)
    };

     /// Recovery Intent: Quarantined DSPHandle の復旧要求を発行する。
     /// FUTURE-3/QSVC-5: rollback 廃止。New RuntimeWorld の Immutable Publish で復旧。
     /// Coordinator は Request enqueue のみ。Admission 判定は行わない（純粋発行関数）。
     /// ★ FUTURE-3 (work88): buildSource は build 入力の metadata/fingerprint を値コピーで運ぶ
     ///   （Recovery semantic = quarantined 除外した現在の authoritative configuration の再構築）。
     /// ★ dash2 §1.9 (Phase E): 戻り値 — この呼び出しが recovery obligation を生成・維持した場合 true。
     ///   transport（push 成功）と durable（queue full → recoveryAdmissionPending_）の両方が true
     ///   （INV-X1-2: queue full ≠ Recovery lost）。shutdown gate による discard は false（wake 不要）。
     ///   submitRecoveryIntent（AudioEngine）は戻り値に基づいて RebuildThread を起床する（§1.9）。
      bool submitRecoveryRequest(const DSPHandle& quarantinedHandle,
                                 const convo::RuntimeBuildSnapshot& buildSource,
                                 PublicationEpoch epoch) noexcept;

      // ★ D105-R5-8: single Completion Authority. Callable from ISR (onPublishCommitted) and
      //   RebuildThread (trySubmitImpl failure). Idempotent: only the first Live→terminal
      //   transition counts; subsequent calls (same id) are no-ops.
       void resolveRecoveryObligation(std::uint64_t obligationId, RecoveryResolution outcome) noexcept;

       // ★ D105-R18: transient-failure adjudication authority. Called by the Orchestrator
       //   (trySubmitImpl build/publish failure sites) and the Builder (durable build/warmup
       //   failure sites) for a single obligation id. Atomically performs:
       //     1. delivery = None (P-B: re-eligibility for redrive, stranded Transport→None)
       //     2. consecutiveFailureCount.fetch_add(1)
       //     3. if counter reaches kMaxObligationConsecutiveFailures → resolve(id, Failed)
       //        (only sanctioned path to ResolvedFailed per R17-4)
       //   The counter persists across durable↔transport transitions and across the obligation
       //   lifetime. Idempotent on a non-Live slot (no-op). ΔL=0 unless exhaustion fires.
       //   This is the only public method that produces ResolvedFailed in production.
       void markTransientFailure(std::uint64_t obligationId) noexcept;

      // ★ D105-R5-9 MUST-2: re-arm a Retry obligation's durable delivery. Only re-arms when the single
      //   durable slot already holds THIS obligation (Building state); never overwrites a distinct live
      //   obligation (would drop that obligation's only delivery). Otherwise the retry is deferred
      //   (obligation stays Live; bounded by L<=32) — the documented single-durable-slot limitation.
       void rearmRecoveryRetry(std::uint64_t obligationId) noexcept;

        // ★ D105-R5-10: re-drive deferred Live obligations (delivery==None) back onto a transport/durable
        //   delivery when a delivery resource frees. Runs ONLY on the CoordinatorLoop (producer) thread —
        //   SPSC-safe for recoveryIntentQueue_ / pendingRecoveryAdmission_. ΔL=0: never inserts a new
        //   obligation or terminates one; only re-attaches a delivery representation to an EXISTING Live
        //   obligation (existing id + buildSource preserved), so no new LogicalRecoveryObligationId is issued.
        void redriveDeferredRecoveryObligations() noexcept;
        void redriveDeferredRecovery(std::uint64_t obligationId) noexcept;


      // ★ D105-R5-8: telemetry accessors for live obligation count + rejections.
       [[nodiscard]] std::uint64_t liveLogicalRecoveryObligationCount() const noexcept {
           return recoveryAdmissions_.liveCount();
       }
       [[nodiscard]] std::uint64_t recoveryCoalescedCount() const noexcept {
           return convo::consumeAtomic(recoveryCoalescedCount_, std::memory_order_acquire);
       }
      [[nodiscard]] std::uint64_t recoveryCapacityExhaustedCount() const noexcept {
          return convo::consumeAtomic(recoveryCapacityExhaustedCount_, std::memory_order_acquire);
      }
       [[nodiscard]] std::uint64_t recoveryObligationShutdownDiscardCount() const noexcept {
           return convo::consumeAtomic(recoveryObligationShutdownDiscardCount_, std::memory_order_acquire);
       }
        [[nodiscard]] std::uint64_t recoveryRetryDeferredCount() const noexcept {
            return convo::consumeAtomic(recoveryRetryDeferredCount_, std::memory_order_acquire);
        }
        [[nodiscard]] std::uint64_t recoveryRetryRedriveCount() const noexcept {        // ★ D105-R5-10
            return convo::consumeAtomic(recoveryRetryRedriveCount_, std::memory_order_acquire);
        }
        [[nodiscard]] std::uint64_t recoveryRetryRedriveFailureCount() const noexcept { // ★ D105-R5-10
            return convo::consumeAtomic(recoveryRetryRedriveFailureCount_, std::memory_order_acquire);
        }
        // ★ D105-R18: telemetry for retry-exhaustion path. Incremented only by
        //   markTransientFailure when the obligation's consecutiveFailureCount reaches
        //   kMaxObligationConsecutiveFailures. Distinguished from any historical Failed
        //   (which is now unreachable in production per R17-4).
        [[nodiscard]] std::uint64_t recoveryRetryExhaustedCount() const noexcept {
            return convo::consumeAtomic(recoveryRetryExhaustedCount_, std::memory_order_acquire);
        }
        // ★ D105-R18: read-only accessor for a slot's retry counter (used by tests).
        [[nodiscard]] std::uint8_t recoveryConsecutiveFailureCount(std::size_t i) const noexcept {
            return recoveryAdmissions_.consecutiveFailureCount(i);
        }


     /// Recovery Intent を Builder Loop へ引き渡す (1件 pop, transport-only)。
     /// FUTURE-10 共通 Intent Queue 化後は processIntent へ統合。
    [[nodiscard]] std::optional<RecoveryIntent> popRecoveryRequest() noexcept;

    // ★ work88 (P2-4 監査補正 — Step C): shutdown 時（Builder 停止後）に recoveryIntentQueue_
    //   の残留 Recovery を ShutdownDiscard として明示破棄する。popRecoveryRequest() が
    //   pendingIntentCount_ を fetchSub するため counter は整合し、P2-4 の queue-empty
    //   判定（isFullyDrained）を正しく成立させる（queue observation を維持したまま残留を解消）。
    //   呼び出し元: stopRebuildThread()（Builder join 後）。Producer（CoordinatorLoop）は
    //   shutdownCoordinatorLoop() で join 済みのため決定的。
    void discardRecoveryRequestsOnShutdown() noexcept;

    // ★ work88 (X1 §6.1 — lease 方式): durable Recovery admission を Builder が消費する。
    //   DurablePending → Building への state transition（destructive dequeue ではない）。
    //   INV-X1-1: take 後も Building 中は recoveryAdmissionPending_ が true を維持
    //   （build gap を isFullyDrained が検出）。build 失敗時は Building → DurablePending へ戻す。
    [[nodiscard]] std::optional<RecoveryIntent> takePendingRecoveryAdmission() noexcept;
    // ★ work88 (X1 §6.1): durable Recovery admission の有無（isFullyDrained 用）。
    [[nodiscard]] bool hasPendingRecoveryAdmission() const noexcept;
    // ★ work88 (X1 §6.1): durable admission を破棄（shutdown 時 — RecoveryAdmissionClosed）。
    //   recoveryShutdownDiscardCount_ を増やし、discard を観測可能にする（ShutdownDiscard — INV-5）。
    void discardPendingRecoveryAdmission() noexcept;
    // ★ work88 (X1 §6.1 — lease 方式): Builder が build 結果に応じて durable admission を settle する。
    //   retry=true（transient failure）: Building → DurablePending へ戻す（次サイクルで再 take）。
    //   retry=false（build success / Discarded）: state を NoAdmission にクリア + recoveryAdmissionPending_ = false。
    //   INV-X1-1（exactly one durable state）が lease 方式で常に成立する。
    void settlePendingRecoveryAdmission(bool retry) noexcept;

    enum class IntentType : std::uint8_t {
        Observe,
        Publish,
        Recovery,
        Quarantine
    };
    static constexpr size_t kIntentTypeCount = 4;

    struct ObservePayload  { DSPHandle handle; PublicationEpoch epoch; };
    // ★ A3 Step 5-3: Decision Snapshot — audio-thread-computed publish-completion transition
    //   data. POD (trivially copyable) so Intent stays LockFreeRingBuffer-transportable.
    //   Decoupled from CrossfadeAuthority::Decision (which would create a
    //   Coordinator<-CrossfadeAuthority<-AudioEngine<-WorldAuthority<-Coordinator include cycle);
    //   ISR publish-executor reconstructs the typed Decision from this at execution (Option A).
    struct PublishDecisionSnapshot {
        bool needsCrossfade;
        bool oldHasIR;
        bool newHasIR;
        double fadeTimeSec;
        DSPHandle newHandle;
        DSPHandle oldHandle;
    };
    static_assert(std::is_trivially_copyable_v<PublishDecisionSnapshot>,
        "PublishDecisionSnapshot must be trivially copyable for LockFreeRingBuffer transport");
    static_assert(std::is_standard_layout_v<PublishDecisionSnapshot>,
        "PublishDecisionSnapshot must be standard layout for LockFreeRingBuffer transport");

    struct PublishPayload  {
        DSPHandle handle;                       // (retained; 5-2 migrates newWorld from sealedSnapshot)
        const void* newWorld;                    // ★ A3 Step 5-1: sealed RuntimeBuildSnapshot world (fixed at enqueue; HANDLER-1 read-only)
        std::uint64_t version;                   // ★ A3 Step 5-1: publish version (fixed at enqueue)
        PublicationEpoch epoch;                  // ★ A3 Step 5-1: currentPublicationEpoch at enqueue (HANDLER-1: do not re-read)
        std::uint64_t mappedGeneration;          // ★ A3 Step 5-1: mapped generation (fixed at enqueue)
        RuntimeBoundary boundary;                // ★ A3 Step 5-1: publish boundary (fixed at enqueue)
        PublishDecisionSnapshot decision;        // ★ A3 Step 5-3: Decision Snapshot (HANDLER-1 read-only, fixed at enqueue)
        std::uint64_t recoveryObligationId{0};    // ★ D105-R5-8: logical recovery obligation id (carried to completion)
    };
    struct RecoveryPayload { DSPHandle quarantinedHandle; convo::RuntimeBuildSnapshot buildSource; };
    struct QuarantinePayload { DSPHandle handle; QuarantineReason reason; uint64_t contextEpoch; };

    struct Intent {
        // ★ work88 (FUTURE-10 / Phase 7): RecoveryPayload が RuntimeBuildSnapshot（NSDMI 付き）
        //   を含むため union のデフォルトコンストラクタは削除される。明示的なデフォルト
        //   コンストラクタで先頭 variant（observe）を値初期化する（全 variant は trivially
        //   copyable のため、payload は割当時に正しく初期化される）。
        Intent() noexcept
            : type(IntentType::Observe), payload(ObservePayload{}), sequenceId(0)
        {
        }
        IntentType type;
        union {
            ObservePayload    observe;
            PublishPayload    publish;
            RecoveryPayload   recovery;
            QuarantinePayload quarantine;
        } payload;
        std::uint64_t sequenceId;
    };
    static_assert(std::is_trivially_copyable_v<Intent>,
        "Intent must be trivially copyable for LockFreeRingBuffer (QUEUE-21)");
    static_assert(std::is_standard_layout_v<Intent>,
        "Intent must be standard layout for LockFreeRingBuffer (QUEUE-21)");

    // ── ★ B3: Publish Intent enqueue (single gen site). ──
    //   Sole route that pushes an IntentType::Publish onto intentQueue_ (RETRY: FUTURE-10
    //   common queue). Producer = Non-RT publish thread (commitRuntimePublication), consumer
    //   = ISR Coordinator Loop (processIntent → PublishExecutor::executePublish).
    //   The caller (AudioEngine) has ALREADY transferred the immutable world ownership into
    //   RuntimeWorldAuthority.ownerChannel_ (key = publication seq/epoch/mappedGen); this
    //   Intent carries only the transport payload (Pointer + build-time metadata + decision),
    //   so the ISR coordinator stays ignorant of RuntimeState (no circular include).
    //   Returns false (without touching the queue) if the queue is full — caller then
    //   reclaims the outstanding Owner via RuntimeWorldAuthority::ownerChannel().take(key).
    [[nodiscard]] bool enqueuePublicationIntent(const Intent& intent) noexcept
    {
        // ★ D101-33-C (D101-33-B A′ design): state_ == ShuttingDown gate は削除済み。
        //   Publication admission authority は ShutdownRuntime::packedState_
        //   （enqueueRuntimePublicationFireAndForget 冒頭の tryAdmit CAS — closeAdmission と
        //   同一単語で線形化、No-Resurrection Case C を構造的排除）に一本化された。
        //   本関数が呼ばれる時点で caller は admission token を保持しており、
        //   CoordinatorState::ShuttingDown は drain-mode signal のみを担う
        //   （authority separation — D101-32-F/D101-33-B 確定）。
        //   ［旧実装: state_ load による check-then-act gate は close と非同期で TOCTOU
        //     window を持つため linearization point たり得なかった（D101-33-A Case C GAP）］

        Intent prepared = intent;
        prepared.type = IntentType::Publish;
        // ★ work88 (X5 §6.5): Publish intent residency 専用 counter の reservation→push→rollback。
        //   全 3 enqueue 経路（通常 rebuild / Recovery publish / deferred 再 enqueue）がここに
        //   集約されるため、本 counter は単一箇所で reservation され二重計上されない（§6.5）。
        //   push 前に fetchAdd（reservation-before-push）→ push 成功で維持 → push 失敗（full）で
        //   fetchSub rollback。INV-X5-1: publicationIntentResidencyCount = Publish intent queue
        //   residency + producer reservation（並行中は >=、producer quiescence 後は ==）。
        convo::fetchAddAtomic(publicationIntentResidencyCount_, std::uint64_t{1}, std::memory_order_acq_rel);
        if (intentQueue_.push(prepared))
            return true;
        convo::fetchSubAtomic(publicationIntentResidencyCount_, std::uint64_t{1}, std::memory_order_acq_rel);
        return false;
    }

    /// Reclaim Request: 指定された DSPHandle の reclaim を要求する。
    /// DELETE-2〜7: Coordinator は epoch 安全確認後、reclaim を実行する。
    /// handleRuntime: DSPHandleRuntime 参照（reclaim 委譲用）
    /// router: ISRRetireRouter 参照（epoch 確認 + enqueueWithRetry 用）
    /// 戻り値: true = reclaim 完了（epoch 安全確認済み）。false = Reader がアクティブで
    ///   遅延（呼出し元は handle を再試行リストへ戻すこと — slot リーク防止）。
    /// ★ work88 (六次レビュー — TOCTOU 修正): 呼出し元（requestReclaimHandle /
    ///   drainDeferredRetireQueues）は epoch 事前チェック後に本メソッドを呼ぶが、
    ///   本メソッド内部でも epoch 再確認するため、事前チェックと内部チェックの間に
    ///   epoch が進むと false が返る。戻り値で遅延を通知し、呼出し元が再試行登録する。
    [[nodiscard]] bool requestReclaim(const DSPHandle& handle,
                                     class DSPHandleRuntime& handleRuntime,
                                     class ISRRetireRouter& router) noexcept;

    // ★ work88 (X3 §6.3 / R4): Reclaim Authority の一本化 — ReclaimMode。
    //   Reclaim Authority は一つ、Safety Precondition が二種類（R4 Phase 1）。
    //   - RuntimeEBR:      通常 runtime — retire → epoch 安全確認（retireEpoch < minReaderEpoch）
    //                      → 不安全なら pending（false 返却・呼出し元が再試行登録）
    // ── ★ dash2 §2.2 (Phase A2 — H.11.17.5 15-Step 7-9, 分離 API) ──
    //   Mode 分岐を消し、Capability 型で経路を区別する（H.11.11.5）。
    //   ⚠️ Step 9: 旧 bool reclaim API（reclaim(ReclaimMode, ..., bool)）は削除済み（AC-1:
    //   reclaim(..., bool) production 0 件）。残存コードはコンパイル不能（compile guard）。
    //   - reclaimNormal(): RuntimeEBR / 通常経路（production 接続 — requestReclaim を内部委譲）
    //   - reclaimShutdownQuiescent(): ShutdownQuiescent。ReclaimPermit 必須（consume で認可）。

    /// RuntimeEBR（通常 runtime）reclaim。requestReclaim と同じ（retire → epoch 安全確認）。
    [[nodiscard]] bool reclaimNormal(
        const DSPHandle& handle,
        class DSPHandleRuntime& handleRuntime,
        class ISRRetireRouter& router) noexcept;

    // ── ★ dash2 §2.2 (Phase A2 — Step 14 / Authority Singularization) ──
    //   ReclaimAuthority（本 Coordinator）が現在の shutdown transaction identity を保持する。
    //   ［不変条件: INV-LIFE-4/6 — ReclaimPermit は ShutdownRuntime のみ生成し、ReclaimAuthority は
    //     自身が管理する shutdown identity と一致する Permit のみ消費］
    //   ★ binding authority は ShutdownRuntime のみ（friend）。AudioEngine は identity を bind できない
    //   （setShutdownIdentity を公開しない — bindShutdownIdentity は private + friend）。
    [[nodiscard]] bool shutdownIdentityBound() const noexcept;
    [[nodiscard]] const ShutdownRuntimeIdentity& currentShutdownIdentity() const noexcept;

    /// ShutdownQuiescent reclaim。ReclaimPermit を consume して認可する（single-use）。
    ///   - Permit.identity は shutdown transaction に束縛（INV-LIFE-5/6）
    ///   - ★ authority validation: permit.identity() == currentShutdownIdentity() を本メソッド内部で
    ///     検証（provenance / freshness）。不一致（cross-runtime / stale）は reject。
    ///   - permit.consume() 成功時のみ reclaim 実行（二重 reclaim 構造的防止 — INV-LIFE-7 / T9）
    ///   - bool readerRegistrationClosed の代替（Permit が quiescence を証明）
    ///   ［caller は identity check を行わない — 本メソッドが単一の ReclaimAuthority 認可点］
    [[nodiscard]] bool reclaimShutdownQuiescent(
        const DSPHandle& handle,
        class DSPHandleRuntime& handleRuntime,
        class ISRRetireRouter& router,
        ReclaimPermit&& permit) noexcept;

    /// Observe Intent Queue から蓄積された Intent を処理する。
    /// P0-4A: Timer から submitObserve でキューイングされた Intent を
    /// Coordinator Loop（NonRT）で取り出して retirePublishedDSP を実行する。
    /// OBSERVE-3〜10 に従い、FIFO 順序で処理し、古い世代の Intent を破棄する。
    void processIntent(AudioEngine& engine,
                       DSPLifetimeManager& lifetimeMgr) noexcept;

    // ── ★ Phase 5: OverflowRing 統合管理 ──

    struct OverflowDrainResult {
        size_t reinjectedCount{0};
        size_t deferredCount{0};
        size_t droppedCount{0};
        uint64_t oldestOverflowAgeUs{0};
        size_t deferredRingOccupancy{0};
    };

    // ★ OverflowRing の定期 drain + 再注入
    //   unlimited=true: 予算無制限（Shutdown Drain用）
    //   retireRuntime.emitRetireIntent() で再注入
    [[nodiscard]] OverflowDrainResult drainOverflowRing(
        class RetireOverflowRing& overflowRing,
        class LifetimeState& retireRuntime,
        bool unlimited = false) noexcept;

    // ★ 滞留年限警告コールバック
    using AgeWarnCallback = void(*)(uint64_t maxAgeUs, uint64_t droppedCount);
    void setOverflowAgeWarnCallback(AgeWarnCallback cb) noexcept;

    // ★ DeferredRing 占有状態
    [[nodiscard]] size_t deferredRingOccupancy() const noexcept;

private:
    // ── ★ dash2 §2.2 (Phase A2 — Step 14 / Authority Singularization) ──
    //   shutdown identity の binding authority は ShutdownRuntime のみ（friend）。
    //   AudioEngine からは bind 不能（setShutdownIdentity 公開 API は廃止）。
    //   ［不変条件: Unbound → Bound(N) → 固定。任意の caller による再 bind は禁止（INV-LIFE-4/6）］
    friend class ShutdownRuntime;

    // ★ dash2 §2.2: ShutdownRuntime が shutdown transaction 確定時に bind する（friend のみ）。
    //   bind は一度だけ（Unbound → Bound）。既に Bound 済みなら無視（任意再 bind 禁止）。
    void bindShutdownIdentity(ShutdownRuntimeIdentity identity) noexcept;

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // ★ Phase5: 内部スケジューラ — 3 scheduler inner classes
    //   RuntimePublicationCoordinator（公開API）は各 scheduler へ委譲
    //   責務分離: God Object 防止 + 単一責任 + ユニットテスト容易性
    //   各 scheduler は coordinator_ 参照を保持し、親クラスのプライベートメンバにアクセス
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    // ★ FUTURE-8/QUEUE-16: Observe Deferred Ring 回御（Retire drain と分離）。
    // ★ dash2 §1.7 (Phase G CW-3c): currentEpoch は caller（processIntent → engine.currentPublicationEpoch()）が
    //   渡す（Coordinator は currentWorld_ を参照しない — R6/R7 と同一方針）。
    void drainObserveDeferred(DSPLifetimeManager& lifetimeMgr, PublicationEpoch currentEpoch) noexcept;

    // ── dash2 §1.4: retire backlog 変更時の pressure slope 検出 + 状態遷移 ──
    //   setRetireBacklogCount（TEST-ONLY）と onRetireAccepted（production semantic event）が
    //   共通で使用する。count は更新後の絶対値。slope = count - previous で Pressure 遷移を判定。
    void noteRetireBacklogChanged(std::uint64_t count) noexcept;

    class OverflowScheduler {
        RuntimeIntentCoordinator& coordinator_;
    public:
        explicit OverflowScheduler(RuntimeIntentCoordinator& coord) noexcept : coordinator_(coord) {}
        [[nodiscard]] OverflowDrainResult drainOverflowRing(
            class RetireOverflowRing& overflowRing,
            class LifetimeState& retireRuntime,
            bool unlimited) noexcept;
        [[nodiscard]] size_t deferredRingOccupancy() const noexcept;
    };

    class ShutdownScheduler {
        RuntimeIntentCoordinator& coordinator_;
    public:
        explicit ShutdownScheduler(RuntimeIntentCoordinator& coord) noexcept : coordinator_(coord) {}
        [[nodiscard]] bool isFullyDrained() const noexcept;
        void requestShutdown() noexcept;
        void markShutdownComplete() noexcept;
    };

    class PriorityScheduler {
        RuntimeIntentCoordinator& coordinator_;
    public:
        explicit PriorityScheduler(RuntimeIntentCoordinator& coord) noexcept : coordinator_(coord) {}
        void escalateAllRetires(RetirePriority minPriority) noexcept;
        void setOverflowAgeWarnCallback(AgeWarnCallback cb) noexcept;
    };

    // ★ Phase5: 内部スケジューラインスタンス
    OverflowScheduler overflowScheduler_;
    ShutdownScheduler shutdownScheduler_;
    PriorityScheduler priorityScheduler_;

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    enum class RejectCode : uint8_t {
        None = 0,
        InvalidClosure,
        InvalidPayloadTier
    };

    // ★ dash2 §1.7 (Phase G CW-3c): currentWorld_（metadata observation alias）を削除。
    //   published-world metadata は RuntimeStore::current（RuntimeWorldAuthority）が単一 source。

    std::atomic<RejectCode> lastRejectCode_;
    std::atomic<std::uint64_t> retireBacklogCount_;
    std::atomic<std::uint64_t> publicationBacklogCount_;
    // ★ work88 (X5 §6.5): Publish Intent residency 専用 counter（INV-X5-1）。
    //   publicationIntentResidencyCount_ = intentQueue_ 内の Publish Intent 数 + producer
    //   enqueue reservation（queue residency + producer-side reservation）。
    //   - 対象: IntentType::Publish（enqueuePublicationIntent の単一箇所で reservation）
    //   - 非対象: deferredPublicationCount_（Orchestrator の deferred state・単一スロット 0/1）
    //     と hasDeferredCommit（commit 未完了の logical state）。Queue residency / Deferred
    //     state / Commit completion を混ぜない（dash §6.5）。
    //   - 増分: enqueuePublicationIntent が push 前に fetchAdd（reservation-before-push）
    //   - 減分: processIntent の intentQueue_.pop で type==Publish の場合 fetchSub
    //     （Publish pop は pendingIntentCount_ を触らない — P2-1 §1.1.6 W2）
    std::atomic<std::uint64_t> publicationIntentResidencyCount_{0};
    // ★ work88 (P2-1 §1.1.1): pendingIntentCount_ は「Intent transport residency + producer
    //   enqueue reservation」を追跡する。
    //   - 対象: Observe / Quarantine / Recovery の各 Intent（transport 内に存在する数）
    //   - 非対象: Publish と RetireIntent（混入禁止 — P2-1 §1.1.5）。Publish は
    //     enqueuePublicationIntent が reservation を取らない。RetireIntent は
    //     retireBacklogCount_（setRetireBacklogCount）が担当する。
    //   ★ INV-ISR-02 / Phase 0 #1: This counter excludes Publish and RetireIntent.
    //   - 増分: producer 側 enqueue 成功時（reservation-before-push で push 前に fetchAdd）
    //   - 減分: consumer 側 pop 成功時（processIntent / drainObserveDeferred /
    //     popRecoveryRequest で fetchSub）
    //   - 絶対値上書き（setPendingIntentCount）は本カウンタに対して禁止。AudioEngine.Commit /
    //     Threading からの RetireIntent 混入を排除するため。
    std::atomic<std::uint64_t> pendingIntentCount_;
    // ★ D101-32-D: fallbackBacklogCount_ / deferredRetireResidencyCount_ /
    //   quarantineResidentCount_（Coordinator側）は削除済み（D101-32-C §4 — vestigial 判定）。
    //   - fallback 実測   = Layer 1 overflow ring resident + quarantineFallbackQueue_.sizeApprox()
    //   - deferred 実測   = observeDeferredRing_.size() + router 側実測
    //   - quarantine DSP  = DSPQuarantineManager::residentCount()（唯一の source of truth）
    //   reclaimInFlightCount_ は onReclaimBegin/End（production wired）が authority のため KEEP。
    std::atomic<std::uint64_t> reclaimInFlightCount_;
    // ★ work88 (X6 §6.6): Quarantine の transport residency を semantic 分離（INV-X6-4）。
    //   quarantineIntentResidencyCount_ = intentQueue_ 内の Quarantine Intent 数（primary transport）
    //   quarantineRingResidencyCount_   = quarantineFallbackQueue_ 内の Quarantine Intent 数（fallback/ring）
    //   実在 quarantine DSP 数は Coordinator 外 — DSPQuarantineManager::residentCount() が唯一の
    //   source of truth（AudioEngine::isFullyDrained で直接判定）。Coordinator 側 resident counter
    //   は D101-32-D で削除済み（X6 以降 writer ゼロのため）。
    std::atomic<std::uint64_t> quarantineIntentResidencyCount_{0};   // ★ X6 新設（Intent lane residency）
    std::atomic<std::uint64_t> quarantineRingResidencyCount_{0};     // ★ X6 新設（ring/fallback 残留）
    std::atomic<std::uint64_t> previousRetireBacklogCount_;
    std::atomic<std::uint32_t> pressureNormalizedWindows_;
    std::atomic<bool> swapPending_{false}; // [work87 P2-5]
    std::atomic<CoordinatorState> state_;
    std::atomic<std::uint64_t> retireAuthorityCount_;
    std::atomic<std::uint64_t> overflowMaxAgeUs_{500'000};  // ★ Phase5: 500ms デフォルト

    // ── ★ dash2 §2.2 (Phase A2 — Step 14 / Authority Singularization) ──
    //   ReclaimAuthority が管理する現在の shutdown transaction identity。
    //   ShutdownRuntime が Proof 生成時に bind し、reclaimShutdownQuiescent 内部で permit.identity
    //   と照合する（cross-runtime / stale Permit reject — AUTH-09/13 / AC-5）。
    //   ［NonRT のみ設定・照合。コンストラクタ初期化は std::mutex 不要（Single-writer:
    //     ShutdownRuntime が shutdown 開始時に一度だけ bind）］
    std::atomic<bool> shutdownIdentityBound_{false};
    ShutdownRuntimeIdentity currentShutdownIdentity_{};

    // ★ Phase5: Overflow Ring / Deferred 管理メンバ
    static constexpr size_t kCoordinatorDeferredRingCapacity = 1024;
    LockFreeRingBuffer<RetireOverflowEntry, kCoordinatorDeferredRingCapacity> coordinatorDeferredRing_;
    std::atomic<size_t> coordinatorDeferredCount_{0};
    static constexpr size_t kLastResortQueueCapacity = 4096;
    RetireOverflowEntry lastResortQueue_[kLastResortQueueCapacity];
    std::atomic<size_t> lastResortCount_{0};

    // ── ★ P0-4A: Observe Intent Queue (4層 Overflow) ──
    // Timer Thread (RT) → submitObserve → push → Coordinator Loop (NonRT) → processIntent → pop
    // SPSC: Producer = Timer Thread, Consumer = Coordinator Loop
    // LockFreeRingBuffer は FIFO を保証し、SPSC なので atomic オーバーヘッドなし
    struct ObserveIntent {
        DSPHandle handle;           // ★ 観測対象の DSPHandle（自己完結型 Intent）。ISR: Coordinator は handle のみで retire 対象を識別可能。
        PublicationEpoch epoch;     // emit 時の publicationEpoch（FIFO順序保証、世代逆転検出用）
        uint64_t intentId;          // 診断・モニタリング用シーケンス番号
    };
    static_assert(std::is_trivially_copyable_v<ObserveIntent>,
        "ObserveIntent must be trivially copyable for LockFreeRingBuffer");
    static_assert(std::is_standard_layout_v<ObserveIntent>,
        "ObserveIntent must be standard layout for LockFreeRingBuffer");

    // ★ work88 (FUTURE-10 / Phase 7): 旧 SPSC 専用リング（observeIntentQueue_/observeFallbackQueue_）は
    //   削除済み — submitObserve は共通 intentQueue_ (MpscBoundedRing) に push するようになったため、
    //   push も pop もされないデッドコードだった。overflow 退避先は observeDeferredRing_ のみ（後続）。

    std::atomic<uint64_t> nextObserveIntentId_{0};
    // ★ FUTURE-8/QUEUE-13: Overflow カウンタを種別別に分離（Observe / Retire）。
    std::atomic<uint64_t> observeOverflowCounter_{0};           // Observe: Layer1→3 溢れ診断
    std::atomic<uint64_t> observeFallbackOverflowCounter_{0};   // Observe: Fallback 溢れ診断

    // ── ★ FUTURE-8/QUEUE-15: Observe Intent 専用 Deferred Ring ──
    //   Retire 系 coordinatorDeferredRing_ と分離。ObserveIntent をそのまま格納（handle 保持）。
    static constexpr size_t kObserveDeferredRingCapacity = 1024;
    LockFreeRingBuffer<ObserveIntent, kObserveDeferredRingCapacity> observeDeferredRing_;

     // ── ★ FUTURE-3: Recovery Intent Queue (transport-only SPSC) ──
    //   ★ dash2 §1.1 (Phase F 検証, 2026-08-15): 単一 Producer 不変条件を実コードで確認済み。
    //     Producer = CoordinatorLoop のみ（submitRecoveryRequest ← submitRecoveryIntent ←
    //     QuarantineIntentHandler / RecoveryIntentHandler — 両 handler とも processIntent 経由で
    //     CoordinatorLoop スレッド上で実行。RecoveryIntentHandler は現状 dead code）。
    //     Consumer = Builder Loop のみ（popRecoveryRequest / takePendingRecoveryAdmission）。
    //   ⇒ MPSC 化は現時点で不要（LockFreeRingBuffer は SPSC 前提 — 複数 Producer 不可）。
    //   ［将来 Timer 等から直接 submitRecoveryRequest を呼ぶ経路を追加する場合のみ MPSC 化
    //     （MpscBoundedRing 置換 + pendingRecoveryAdmission_ の mutex 保護 — plan §1.1.1）］
    //   ★ F-0 事前監査（2026-08-19, evidence/phase-f-0-recovery-intent-queue-audit.md）:
    //     判定 = NO-GO（現時点では実装しない）。単一 Producer（CoordinatorLoop）は検証済み不変条件、
    //     reservation→push→rollback / pop fetchSub は既に実装済み、第2 producer の引き金未発生。
    //     条件付き GO: 第2 Non-RT producer（例: Timer 直接経路）の設計確定時に実施
    //     （型置換 + pendingRecoveryAdmission_ 保護 + 2-producer テスト — plan §1.1）。
    static constexpr size_t kRecoveryIntentQueueCapacity = 256;
    LockFreeRingBuffer<RecoveryIntent, kRecoveryIntentQueueCapacity> recoveryIntentQueue_;
    std::atomic<uint64_t> nextRecoveryIntentId_{0};
    // ★ work88 (六次レビュー — INV-5): Recovery Intent push 失敗（queue full）時の drop 記録。
    //   getter（recoveryIntentDropCount()）は public セクションに定義。
    std::atomic<uint64_t> recoveryIntentDropCount_{0};
    // ★ work88 (P2-4 監査補正 — Step B/C): shutdown 時（AdmissionClosed）に Recovery を
    //   意図的に破棄した回数（ShutdownDiscard）。drop（queue full）とは区別 — dash §8.1。
    std::atomic<uint64_t> recoveryShutdownDiscardCount_{0};

    // ── ★ work88 (X1 §6.1): Recovery Durable Admission（lease 方式）──
    //   queue full で Recovery が「失われる」ことを構造的に排除する durable admission state。
    //   INV-X1-1: accepted ⇒ exactly one durable state（DurablePending OR Building）exists
    //   INV-X1-2: queue full ≠ Recovery lost（durable admission が保持）
    //   INV-X1-4: durable state は World ownership を持たない（DSPHandle / epoch / intentId /
    //             RuntimeBuildSnapshot のみ — 非所有）
    //   INV-X1-5: 1 logical Recovery admission = at most 1 reservation（coalesce で増やさない）
    //   INV-X1-6: durable admission は queue residency と二重計上しない
    //   SPSC: Producer = CoordinatorLoop（submitRecoveryRequest / QuarantineIntentHandler 経由）
    //         Consumer = Builder Loop（takePendingRecoveryAdmission）— 競合なし
    //   ★ 二十六次レビュー（lease 方式・必須修正1）: take は destructive dequeue ではなく
    //     DurablePending → Building の state transition。build 失敗（transient）は
    //     Building → DurablePending へ戻す（retry を構造的保証）。obsolete は Discarded。
#pragma warning(push)
#pragma warning(disable : 4324)   // DSPHandle(alignas 16) による構造体パディング警告を抑制
    struct PendingRecoveryAdmission {
        enum class State : uint8_t {
            NoAdmission = 0,
            DurablePending,
            Building
        };
        State state = State::NoAdmission;
        bool pending = false;                 // durable state 有効（state != NoAdmission）
        uint64_t recoveryGeneration = 0;      // 入ってきた時点の rebuildRequestGeneration（coalesce 判定用）
        convo::RuntimeBuildSnapshot buildSource{};  // latest（coalesce で更新）
        bool reservationOwned = false;        // 1 admission = 1 reservation（INV-X1-5）
        DSPHandle handle{};                   // recovery 対象（quarantined DSPHandle）— 消費時 isNull 検証
        PublicationEpoch epoch{0};            // emit 時 publicationEpoch（FIFO/epoch 検証用）
        uint64_t intentId{0};                 // 診断・モニタリング用シーケンス番号
        uint64_t recoveryObligationId{0};     // ★ D105-R5-8: logical recovery obligation id
    };
    PendingRecoveryAdmission pendingRecoveryAdmission_;   // SPSC（plain 構造体 — atomic 不要）
    std::atomic<bool> recoveryAdmissionPending_{false};   // durable 有効フラグ（isFullyDrained が読む）
    static_assert(std::is_trivially_copyable_v<PendingRecoveryAdmission>,
        "PendingRecoveryAdmission must be trivially copyable");
#pragma warning(pop)

    // ── ★ D105-R5-8: Logical Recovery Obligation table (capacity-enforced, lock-free) ──
    RecoveryAdmissionTable<kMaxLogicalRecoveryObligations> recoveryAdmissions_;
    std::atomic<std::uint64_t> recoveryCoalescedCount_{0};                   // coalesce-before-capacity hits
    std::atomic<std::uint64_t> recoveryCapacityExhaustedCount_{0};          // L==32 rejects
    std::atomic<std::uint64_t> recoveryObligationShutdownDiscardCount_{0};   // shutdown discards
    std::atomic<std::uint64_t> recoveryRetryDeferredCount_{0};              // ★ D105-R5-9 MUST-2: durable-slot
                                                                              //   occupied by a different live obligation
                                                                              //   → retry delivery deferred (L unchanged)
    std::atomic<std::uint64_t> recoveryRetryRedriveCount_{0};               // ★ D105-R5-10: re-drive attempts
    std::atomic<std::uint64_t> recoveryRetryRedriveFailureCount_{0};         // ★ D105-R5-10: both resources busy
                                                                               //   → remains deferred (L unchanged)
    std::atomic<std::uint64_t> recoveryRetryExhaustedCount_{0};              // ★ D105-R18: retry-budget exhaustion
                                                                               //   (only sanctioned path to ResolvedFailed)


    // ── ★ FUTURE-10: 共通 Intent Queue（種別問わず単一 FIFO） ──
    //   ★ work88 (FUTURE-10 前提 0): LockFreeRingBuffer（SPSC）→ MpscBoundedRing（MPSC）に置換。
    //   intentQueue_ は既に複数 Producer（Builder/Rebuild スレッド・Timer・CoordinatorLoop
    //   deferred resubmit）から push される MPSC 実態だった（潜在競合）。Vyukov bounded で
    //   CAS 予約（reservation order = seqId order）→ payload 書込み → seq release を保証。
    static constexpr size_t kIntentQueueCapacity = 4096;
    MpscBoundedRing<Intent, kIntentQueueCapacity> intentQueue_;
    std::atomic<uint64_t> nextIntentId_{0};

    // ── ★ FUTURE-10 (work88): Quarantine 専用 fallback ring（三次レビュー policy 表）──
    //   Quarantine intent の drop は安全要件違反（bad DSP が使用可能なまま残る）。
    //   intentQueue_ full 時はここへ退避。それも full なら HealthEvent / Critical へ昇格
    //   （drop カウンタを増やしつつ決して静かに破棄しない）。
    static constexpr size_t kQuarantineFallbackCapacity = 1024;
    MpscBoundedRing<Intent, kQuarantineFallbackCapacity> quarantineFallbackQueue_;
    std::atomic<uint64_t> quarantineFallbackDropCount_{0};

    // ★ Phase5: 滞留年限警告コールバック
    AgeWarnCallback overflowAgeWarnCallback_{nullptr};

    static constexpr std::uint64_t kPressureSlopeThreshold = 8;
    static constexpr std::uint32_t kPressureNormalizeWindows = 3;

    // ★ P0-5: QuarantineService インスタンス
    QuarantineService quarantineService_;
};

class MultiStagePublisher {
public:
    explicit MultiStagePublisher(RuntimeBoundary boundary = RuntimeBoundary::NonRTWorld) : boundary_(boundary) {}
    void publishTier(PayloadTier tier, const void* payload);
    [[nodiscard]] bool wasRejected() const noexcept { return rejected_; }

private:
    RuntimeBoundary boundary_;
    bool rejected_ = false;
};

} // namespace convo::isr
