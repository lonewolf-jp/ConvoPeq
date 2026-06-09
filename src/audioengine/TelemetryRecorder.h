#pragma once

#include <cstdint>
#include <atomic>
#include <array>
#include "RuntimePublicationState.h"

namespace convo::isr {

// ★ PublishStage: 出版進捗段階
enum class PublishStage : uint8_t {
    Submitted,
    Built,
    Validated,
    Published,
    Retired,
    Reclaimed
};

// ★ FailureStage: 失敗発生段階
enum class FailureStage : uint8_t {
    None,
    Admission,
    Validation,
    Execution,
    Bridge,
    Shutdown
};

// ★ FailureReason: 失敗理由
enum class FailureReason : uint8_t {
    None,
    AdmissionRejected,
    ValidationFailed,
    PublishFailed,
    BridgeFailed,
    ShutdownRejected,
    StaleGeneration,
    QueuePressure,
    Count   // バケット数。常に最後
};

// ★ FailureRecord: 軽量失敗レコード (常時記録)
struct FailureRecord {
    uint64_t correlationIdShort{0};
    FailureStage stage{FailureStage::None};
    FailureReason reason{FailureReason::None};
    const char* origin{nullptr};
    uint64_t timestampUs{0};
};

// ★ FailureSnapshot: 詳細版 (閾値超過時のみ)
struct FailureSnapshot {
    uint64_t correlationIdShort{0};
    uint64_t publicationSequenceId{0};
    uint64_t generation{0};
    uint64_t worldId{0};
    FailureStage stage{FailureStage::None};
    FailureReason reason{FailureReason::None};
    const char* origin{nullptr};
    uint64_t threadId{0};
    uint64_t coordinatorState{0};
    uint64_t shutdownPhase{0};
    uint64_t publicationClass{0};
    uint64_t activeReaderCount{0};
    uint64_t minReaderEpoch{0};
    uint64_t currentEpoch{0};
    uint64_t timestampUs{0};
};

// ★ PublicationProgressRecord: 出版進捗レコード
struct PublicationProgressRecord {
    uint64_t correlationIdShort{0};
    uint64_t generation{0};
    uint64_t worldId{0};
    PublishStage stage{PublishStage::Submitted};
    uint64_t timestampUs{0};
};

// ★ PublicationClass: Stall 閾値分類
enum class PublicationClass : uint8_t {
    FastPath,      // 5s  — 通常publish
    HeavyBuild,    // 30s — 大規模IR再構築
    Shutdown       // 60s — shutdown publish
};

// ★ OrchestratorHealthSnapshot: 健全性スナップショット
struct OrchestratorHealthSnapshot {
    uint64_t submittedCount{0};
    uint64_t publishedCount{0};
    uint64_t retiredCount{0};
    uint64_t reclaimedCount{0};
    uint32_t executorQueueDepth{0};
    uint64_t lastProgressTimestampUs{0};
    OrchestratorForwardProgress::StuckStage stuckStage{OrchestratorForwardProgress::StuckStage::None};
    uint64_t timestampUs{0};
};

// ★ RetireTimelineRecord: 退役タイムラインレコード
struct RetireTimelineRecord {
    uint64_t publicationSequenceId{0};
    uint64_t generation{0};
    uint64_t worldId{0};
    uint64_t retireEpoch{0};
    uint64_t reclaimEpoch{0};
};

// ★ RetireStallSnapshot: Retire停滞スナップショット
struct RetireStallSnapshot {
    uint64_t publicationSequenceId{0};
    uint64_t generation{0};
    uint64_t retireEpoch{0};
    uint64_t reclaimEpoch{0};       // 0=未完了
    uint64_t activeReaderCount{0};
    uint64_t minReaderEpoch{0};
    uint64_t currentEpoch{0};
    uint64_t pendingRetireCount{0};
};

// ★ DeferredHealth: Deferred Publish 健全性
struct DeferredHealth {
    uint64_t deferredCount{0};
    uint64_t oldestDeferredAgeMs{0};
    uint64_t overwriteCount{0};
    DiscardReason lastDiscardReason{DiscardReason::None};
    uint64_t lastDiscardTimestampUs{0};
};

// ★ FailureSnapshotController: FailureReason 別独立バケット
class FailureSnapshotController {
public:
    static constexpr uint32_t kMaxSnapshotsPerMinute = 10;
    static constexpr uint64_t kBucketRefreshIntervalUs = 60'000'000;  // 60秒
    static constexpr uint64_t kMinSnapshotIntervalUs = 1'000'000;     // 1秒

    bool shouldTakeSnapshot(FailureReason reason, uint64_t failureCount, uint64_t nowUs) noexcept {
        auto& bucket = buckets_[static_cast<size_t>(reason)];
        const uint64_t bucketStart = bucket.minuteStartUs_.load(std::memory_order_acquire);
        if (nowUs - bucketStart > kBucketRefreshIntervalUs) {
            bucket.minuteStartUs_.store(nowUs, std::memory_order_release);
            bucket.snapshotCountThisMinute_.store(0, std::memory_order_release);
        }
        if (failureCount <= failureSnapshotThreshold_)
            return false;
        const uint64_t lastSnap = lastSnapshotTimestampUs_.load(std::memory_order_acquire);
        if (nowUs - lastSnap < kMinSnapshotIntervalUs)
            return false;
        return bucket.snapshotCountThisMinute_.fetch_add(1, std::memory_order_acq_rel)
            < kMaxSnapshotsPerMinute;
    }

    void setFailureSnapshotThreshold(uint64_t threshold) noexcept {
        failureSnapshotThreshold_ = threshold;
    }

    void recordSnapshotTaken(uint64_t nowUs) noexcept {
        lastSnapshotTimestampUs_.store(nowUs, std::memory_order_release);
    }

private:
    struct ReasonBucket {
        std::atomic<uint32_t> snapshotCountThisMinute_{0};
        std::atomic<uint64_t> minuteStartUs_{0};
    };
    std::array<ReasonBucket, static_cast<size_t>(FailureReason::Count)> buckets_{};
    std::atomic<uint64_t> lastSnapshotTimestampUs_{0};
    uint64_t failureSnapshotThreshold_{10};  // デフォルト閾値
};

// ★ RingBuffer (固定サイズ)
template<typename T, size_t N>
class FixedRingBuffer {
public:
    static_assert(N > 0 && N <= 65536, "RingBuffer size must be 1..65536");

    // PushResult: tryPush の結果。overwriteDetection 有効時のみ overwritten が正確。
    struct PushResult {
        bool success;        // 常に true (overwrite方式)
        bool overwritten;    // 今回のpushで既存エントリを上書きした場合 true
    };

    PushResult tryPush(const T& item) noexcept {
        const auto idx = writePos_.fetch_add(1, std::memory_order_acq_rel);
        const auto slot = idx % N;
        data_[slot] = item;
        const bool overwritten = (idx >= N);
        if (overwritten) {
            overwriteCount_.fetch_add(1, std::memory_order_release);
        }
        return PushResult{true, overwritten};
    }

    [[nodiscard]] size_t size() const noexcept {
        const auto wp = writePos_.load(std::memory_order_acquire);
        return wp < N ? wp : N;
    }

    [[nodiscard]] size_t capacity() const noexcept { return N; }

    [[nodiscard]] uint64_t overwriteCount() const noexcept {
        return overwriteCount_.load(std::memory_order_acquire);
    }

    // 最新 count 件を取得 (count <= N)
    [[nodiscard]] size_t readLatest(T* out, size_t count) const noexcept {
        const auto wp = writePos_.load(std::memory_order_acquire);
        const auto available = wp < N ? wp : N;
        const auto toRead = count < available ? count : available;
        const auto startIdx = wp - toRead;
        for (size_t i = 0; i < toRead; ++i) {
            out[i] = data_[(startIdx + i) % N];
        }
        return toRead;
    }

private:
    std::array<T, N> data_{};
    std::atomic<uint64_t> writePos_{0};
    std::atomic<uint64_t> overwriteCount_{0};
};

// ============================================================
// ★ TelemetryRecorder: 副産物専用。StateOwner から完全分離
// ============================================================
class RuntimePublicationStateOwner;

class TelemetryRecorder {
public:
    TelemetryRecorder() noexcept = default;

    // ── 進捗記録 ──
    void recordProgress(const CorrelationId& correlationId,
                        uint64_t generation,
                        uint64_t worldId,
                        PublishStage stage,
                        uint64_t timestampUs) noexcept;

    // ── 失敗記録 ──
    void recordFailure(FailureStage stage,
                       FailureReason reason,
                       const char* origin,
                       uint64_t correlationIdShort,
                       uint64_t timestampUs) noexcept;

    // ── 詳細スナップショット (条件付き) ──
    bool recordFailureSnapshot(const FailureSnapshot& snapshot,
                               uint64_t failureCount,
                               uint64_t nowUs) noexcept;

    // ── 健全性 ──
    void recordHealth(const OrchestratorHealthSnapshot& snapshot) noexcept;

    // ── DeferredHealth ──
    void recordDeferredHealth(const DeferredHealth& health) noexcept;

    // ── RetireTimeline ──
    void recordRetireTimeline(const RetireTimelineRecord& record) noexcept;

    // ── RetireStall ──
    void recordRetireStall(const RetireStallSnapshot& stall) noexcept;

    // ── CorrelationId 採番 ──
    [[nodiscard]] CorrelationId nextCorrelationId(uint64_t engineInstanceId) noexcept {
        const uint64_t counter = localCounter_.fetch_add(1, std::memory_order_acq_rel);
        return CorrelationId{engineInstanceId, counter};
    }

    // ── スナップショット取得 (Evidence出力用) ──
    struct TelemetrySnapshot {
        std::array<FailureRecord, 512> failureRecords{};
        size_t failureRecordCount{0};
        std::array<FailureSnapshot, 64> failureSnapshots{};
        size_t failureSnapshotCount{0};
        std::array<PublicationProgressRecord, 4096> progressRecords{};
        size_t progressRecordCount{0};
        uint64_t progressOverwriteCount{0};  // ★ リングオーバーライト累積数
        OrchestratorHealthSnapshot lastHealthSnapshot{};
        DeferredHealth lastDeferredHealth{};
        RetireStallSnapshot lastRetireStall{};
    };

    [[nodiscard]] TelemetrySnapshot captureSnapshot() const noexcept;

    // ── 状態通知: StateOwner へのドロップ通知 ──
    void setStateOwner(RuntimePublicationStateOwner* owner) noexcept {
        stateOwner_ = owner;
    }

private:
    FixedRingBuffer<FailureRecord, 512> failureRecords_;
    FixedRingBuffer<FailureSnapshot, 64> failureSnapshots_;
    FixedRingBuffer<PublicationProgressRecord, 4096> progressRecords_;
    FixedRingBuffer<RetireTimelineRecord, 4096> retireTimelines_;
    std::atomic<OrchestratorHealthSnapshot> lastHealthSnapshot_{};
    std::atomic<DeferredHealth> lastDeferredHealth_{};
    std::atomic<RetireStallSnapshot> lastRetireStall_{};
    std::atomic<uint64_t> localCounter_{1};  // 0は無効値
    FailureSnapshotController snapshotController_;
    RuntimePublicationStateOwner* stateOwner_{nullptr};
};

} // namespace convo::isr
