#pragma once
#include <atomic>
#include <memory>
#include <cstdint>
#include <type_traits>
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

namespace convo::isr {

// ★ P0-4C: 前方宣言（完全定義は ISRDSPHandle.h / ISRDSPQuarantine.h）
struct DSPHandle;
enum class QuarantineReason : int;

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
    // ★ HW-1: 最新の publicationEpoch を取得（storeReceipt 用）
    [[nodiscard]] PublicationEpoch currentPublicationEpoch() const noexcept {
        return persistentState_.publicationEpoch;
    }
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
    [[nodiscard]] std::uint64_t getReclaimInFlightCount() const noexcept;
    [[nodiscard]] std::uint64_t getOverflowMaxAgeUs() const noexcept;          // ★ Phase5
    [[nodiscard]] bool isFullyDrained() const noexcept;
    [[nodiscard]] CoordinatorState getState() const noexcept;
    void markTransitionStart() noexcept;
    void markTransitionCommitted() noexcept;
    void requestShutdown() noexcept;
    void markShutdownComplete() noexcept;

    // ── ★ P0-4C: ISR Intent 発行インターフェース ──
    //   OBSERVE-1: Timer → emitObserveIntent → Coordinator が retirePublishedDSP を起動
    //   QSVC-2:    Coordinator は QuarantineService を介さず直接 quarantine を呼ばない
    //   DELETE-1:  reclaim() は Coordinator 専用。外部からの直接呼び出し禁止。

    /// Observe Intent: Timer から定期観測要求を発行する。
    /// Coordinator は Intent Queue に追加し、非同期に処理する。
    /// OBSERVE-1〜8 に従い、Timer はこのメソッドのみを呼び出す。
    void emitObserveIntent() noexcept;

    /// Quarantine Intent: 指定された DSPHandle を quarantine する要求を発行する。
    /// QSVC-2: Coordinator は QuarantineService 経由で quarantine を実行する。
    void emitQuarantineIntent(const DSPHandle& handle,
                              QuarantineReason reason,
                              uint64_t contextEpoch = 0) noexcept;

    /// Reclaim Request: 指定された DSPHandle の reclaim を要求する。
    /// DELETE-2〜7: Coordinator は epoch 安全確認後、reclaim を実行する。
    void requestReclaim(const DSPHandle& handle) noexcept;

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
        class RetireRuntime& retireRuntime,
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

    class OverflowScheduler {
        RuntimePublicationCoordinator& coordinator_;
    public:
        explicit OverflowScheduler(RuntimePublicationCoordinator& coord) noexcept : coordinator_(coord) {}
        [[nodiscard]] OverflowDrainResult drainOverflowRing(
            class RetireOverflowRing& overflowRing,
            class RetireRuntime& retireRuntime,
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

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 方式C（採用）: PersistentStateBlock (plain struct)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 全アクセスが MessageThread 閉域であるため、atomic ではなく plain struct で十分。
    struct PersistentStateBlock {
        std::uint64_t publicationSequenceId = 0;
        std::uint64_t publicationEpoch      = 0;
        std::uint64_t mappedRuntimeGeneration = 0;

        [[nodiscard]] static bool isMonotonic(
            const PersistentStateBlock& prev,
            std::uint64_t nextSeqId,
            std::uint64_t nextEpoch,
            std::uint64_t nextGen) noexcept
        {
            const bool hasPrevious = prev.publicationSequenceId != 0
                || prev.publicationEpoch != 0
                || prev.mappedRuntimeGeneration != 0;
            if (!hasPrevious)
                return true;
            return nextSeqId > prev.publicationSequenceId
                && nextEpoch > prev.publicationEpoch
                && nextGen > prev.mappedRuntimeGeneration;
        }
    };

    static_assert(std::is_standard_layout_v<PersistentStateBlock>,
        "PersistentStateBlock must be standard-layout");
    static_assert(std::is_trivially_copyable_v<PersistentStateBlock>,
        "PersistentStateBlock must remain trivially copyable");
    static_assert(sizeof(PersistentStateBlock) == sizeof(std::uint64_t) * 3,
        "PersistentStateBlock must be exactly 3 uint64_t without padding");

    // ★ 3個別 atomic に代わり、plain struct で3フィールドを論理一貫管理
    // IMPORTANT: persistentState_ is MessageThread-only.
    //   Any cross-thread access requires conversion to std::atomic<PersistentStateBlock>.
    PersistentStateBlock persistentState_{};

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

    // ★ Phase5: 滞留年限警告コールバック
    AgeWarnCallback overflowAgeWarnCallback_{nullptr};

    static constexpr std::uint64_t kPressureSlopeThreshold = 8;
    static constexpr std::uint32_t kPressureNormalizeWindows = 3;
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
