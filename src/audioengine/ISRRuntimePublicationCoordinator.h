#pragma once
#include <atomic>
#include <cstring>  // ★ work88 (Phase 7): Intent default ctor の union ゼロ初期化 (std::memset)
#include <memory>
#include <cstdint>
#include <type_traits>
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

class RuntimePublicationCoordinator {
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

    RuntimePublicationCoordinator();
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
                std::uint64_t mappedGeneration);
    void retire(RetireAuthority, RuntimeBoundary boundary, const void* oldWorld);
    [[nodiscard]] RetireEnqueueResult enqueueRetire(RetireAuthority auth,
                                                      ISRRetireRouter& router,
                                                      void* ptr,
                                                      void (*deleter)(void*),
                                                      std::uint64_t epoch) noexcept;
    [[nodiscard]] std::uint64_t retireAuthorityCount() const noexcept;
    const void* getCurrent() const noexcept;
    std::uint64_t getVersion() const noexcept;
    // ★ FUTURE-4: latest publicationEpoch derived from currentWorld_ (RuntimeState::publication.epoch)
    [[nodiscard]] PublicationEpoch currentPublicationEpoch() const noexcept;
    // ★ A-1: sequence derived from currentWorld_ (RuntimeState::publication.sequenceId) — read-only Authority access.
    [[nodiscard]] PublicationSequenceId currentPublicationSequenceId() const noexcept;
    void setRetireBacklogCount(std::uint64_t count) noexcept;
    void setPublicationBacklogCount(std::uint64_t count) noexcept;
    void setPendingIntentCount(std::uint64_t count) noexcept;
    void setFallbackBacklogCount(std::uint64_t count) noexcept;
    void setReclaimInFlightCount(std::uint64_t count) noexcept;
    void setDeferredRetireResidencyCount(std::uint64_t count) noexcept;
    void setQuarantineResidentCount(std::uint64_t count) noexcept;  // ★ Phase2
    void escalateAllRetires(RetirePriority minPriority) noexcept;    // ★ Phase5: 全RetireIntent の優先度を底上げ
    void setOverflowMaxAgeUs(std::uint64_t maxAgeUs) noexcept;       // ★ Phase5: OverflowRing 滞留年限警告しきい値
    void setSwapPending(bool pending) noexcept;
    [[nodiscard]] bool isSwapPending() const noexcept;
    // ★ A-2.4: getter 群（DrainAudit 用）
    [[nodiscard]] std::uint64_t getPublicationBacklogCount() const noexcept;
    [[nodiscard]] std::uint64_t getPendingIntentCount() const noexcept;
    [[nodiscard]] std::uint64_t getRetireBacklogCount() const noexcept;
    [[nodiscard]] std::uint64_t getFallbackBacklogCount() const noexcept;
    [[nodiscard]] std::uint64_t getDeferredRetireResidencyCount() const noexcept;
    [[nodiscard]] std::uint64_t getQuarantineResidentCount() const noexcept;  // ★ Phase2
    // ★ work88 (FUTURE-10): Quarantine fallback ring の drop 回数（静かに破棄しない証跡）。
    //   AudioEngine 側 HealthMonitor が監視し ISRHealthState::Critical 昇格を駆動する。
    [[nodiscard]] std::uint64_t quarantineFallbackDropCount() const noexcept
    {
        return convo::consumeAtomic(quarantineFallbackDropCount_, std::memory_order_acquire);
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
    void submitObserve(const DSPHandle& handle) noexcept;

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

     /// Recovery Intent: Quarantined DSPHandle の復旧要求を発行する。
     /// FUTURE-3/QSVC-5: rollback 廃止。New RuntimeWorld の Immutable Publish で復旧。
     /// Coordinator は Request enqueue のみ。Admission 判定は行わない（純粋発行関数）。
     /// ★ FUTURE-3 (work88): buildSource は build 入力の metadata/fingerprint を値コピーで運ぶ
     ///   （Recovery semantic = quarantined 除外した現在の authoritative configuration の再構築）。
     void submitRecoveryRequest(const DSPHandle& quarantinedHandle,
                                const convo::RuntimeBuildSnapshot& buildSource) noexcept;

     /// Recovery Intent を Builder Loop へ引き渡す (1件 pop, transport-only)。
     /// FUTURE-10 共通 Intent Queue 化後は processIntent へ統合。
    [[nodiscard]] std::optional<RecoveryIntent> popRecoveryRequest() noexcept;

    // ── ★ FUTURE-10: 共通 Intent 型（種別別 Queue → 単一 intentQueue_） ──
    //   QUEUE-21: tagged-union variant。std::variant は trivially copyable 非保証のため不可。
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
        Intent prepared = intent;
        prepared.type = IntentType::Publish;
        return intentQueue_.push(prepared);
    }

    /// Reclaim Request: 指定された DSPHandle の reclaim を要求する。
    /// DELETE-2〜7: Coordinator は epoch 安全確認後、reclaim を実行する。
    /// handleRuntime: DSPHandleRuntime 参照（reclaim 委譲用）
    /// router: ISRRetireRouter 参照（epoch 確認 + enqueueWithRetry 用）
    void requestReclaim(const DSPHandle& handle,
                        class DSPHandleRuntime& handleRuntime,
                        class ISRRetireRouter& router) noexcept;

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
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // ★ Phase5: 内部スケジューラ — 3 scheduler inner classes
    //   RuntimePublicationCoordinator（公開API）は各 scheduler へ委譲
    //   責務分離: God Object 防止 + 単一責任 + ユニットテスト容易性
    //   各 scheduler は coordinator_ 参照を保持し、親クラスのプライベートメンバにアクセス
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    // ★ FUTURE-8/QUEUE-16: Observe Deferred Ring 回御（Retire drain と分離）。
    void drainObserveDeferred(DSPLifetimeManager& lifetimeMgr) noexcept;

    class OverflowScheduler {
        RuntimePublicationCoordinator& coordinator_;
    public:
        explicit OverflowScheduler(RuntimePublicationCoordinator& coord) noexcept : coordinator_(coord) {}
        [[nodiscard]] OverflowDrainResult drainOverflowRing(
            class RetireOverflowRing& overflowRing,
            class LifetimeState& retireRuntime,
            bool unlimited) noexcept;
        [[nodiscard]] size_t deferredRingOccupancy() const noexcept;
    };

    class ShutdownScheduler {
        RuntimePublicationCoordinator& coordinator_;
    public:
        explicit ShutdownScheduler(RuntimePublicationCoordinator& coord) noexcept : coordinator_(coord) {}
        [[nodiscard]] bool isFullyDrained() const noexcept;
        void requestShutdown() noexcept;
        void markShutdownComplete() noexcept;
    };

    class PriorityScheduler {
        RuntimePublicationCoordinator& coordinator_;
    public:
        explicit PriorityScheduler(RuntimePublicationCoordinator& coord) noexcept : coordinator_(coord) {}
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

    // ★ FUTURE-4: persistentState_ removed — metadata (epoch/sequenceId/mappedRuntimeGeneration)
    //   is derived from currentWorld_ (RuntimeState::publication) at read time.

    std::atomic<const void*> currentWorld_;
    std::atomic<RejectCode> lastRejectCode_;
    std::atomic<std::uint64_t> retireBacklogCount_;
    std::atomic<std::uint64_t> publicationBacklogCount_;
    std::atomic<std::uint64_t> pendingIntentCount_;
    std::atomic<std::uint64_t> fallbackBacklogCount_;
    std::atomic<std::uint64_t> reclaimInFlightCount_;
    std::atomic<std::uint64_t> deferredRetireResidencyCount_;
    std::atomic<std::uint64_t> quarantineResidentCount_;    // ★ Phase2: Quarantine滞留カウント
    std::atomic<std::uint64_t> previousRetireBacklogCount_;
    std::atomic<std::uint32_t> pressureNormalizedWindows_;
    std::atomic<bool> swapPending_{false}; // [work87 P2-5]
    std::atomic<CoordinatorState> state_;
    std::atomic<std::uint64_t> retireAuthorityCount_;
    std::atomic<std::uint64_t> overflowMaxAgeUs_{500'000};  // ★ Phase5: 500ms デフォルト

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

    static constexpr size_t kObserveIntentQueueCapacity = 1024;
    LockFreeRingBuffer<ObserveIntent, kObserveIntentQueueCapacity> observeIntentQueue_;

    // ★ QUEUE-11: Layer 2 (Fallback) — Primary 溢れのセカンダリキュー
    static constexpr size_t kObserveFallbackCapacity = 2048;
    LockFreeRingBuffer<ObserveIntent, kObserveFallbackCapacity> observeFallbackQueue_;

    std::atomic<uint64_t> nextObserveIntentId_{0};
    // ★ FUTURE-8/QUEUE-13: Overflow カウンタを種別別に分離（Observe / Retire）。
    std::atomic<uint64_t> observeOverflowCounter_{0};           // Observe: Layer1→3 溢れ診断
    std::atomic<uint64_t> observeFallbackOverflowCounter_{0};   // Observe: Fallback 溢れ診断

    // ── ★ FUTURE-8/QUEUE-15: Observe Intent 専用 Deferred Ring ──
    //   Retire 系 coordinatorDeferredRing_ と分離。ObserveIntent をそのまま格納（handle 保持）。
    static constexpr size_t kObserveDeferredRingCapacity = 1024;
    LockFreeRingBuffer<ObserveIntent, kObserveDeferredRingCapacity> observeDeferredRing_;

     // ── ★ FUTURE-3: Recovery Intent Queue (transport-only SPSC) ──
    static constexpr size_t kRecoveryIntentQueueCapacity = 256;
    LockFreeRingBuffer<RecoveryIntent, kRecoveryIntentQueueCapacity> recoveryIntentQueue_;
    std::atomic<uint64_t> nextRecoveryIntentId_{0};

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
