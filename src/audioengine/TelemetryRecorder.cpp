#include "TelemetryRecorder.h"
#include "RuntimePublicationState.h"
#include <chrono>

namespace convo::isr {

namespace {
    uint64_t nowUs() noexcept {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
    }
}

void TelemetryRecorder::recordProgress(
    const CorrelationId& correlationId,
    uint64_t generation,
    uint64_t worldId,
    PublishStage stage,
    uint64_t timestampUs) noexcept
{
    const PublicationProgressRecord record{
        .correlationIdShort = correlationId.shortValue(),
        .generation = generation,
        .worldId = worldId,
        .stage = stage,
        .timestampUs = timestampUs != 0 ? timestampUs : nowUs()
    };

    const auto pushResult = progressRecords_.tryPush(record);
    if (pushResult.overwritten) {
        // リングオーバーライト → ドロップ通知
        if (stateOwner_ != nullptr) {
            const auto dropTime = nowUs();
            stateOwner_->notifyProgressDrop(dropTime);
        }
    }
}

void TelemetryRecorder::recordFailure(
    FailureStage stage,
    FailureReason reason,
    const char* origin,
    uint64_t correlationIdShort,
    uint64_t timestampUs) noexcept
{
    const auto ts = timestampUs != 0 ? timestampUs : nowUs();
    const FailureRecord record{
        .correlationIdShort = correlationIdShort,
        .stage = stage,
        .reason = reason,
        .origin = origin,
        .timestampUs = ts
    };
    const auto pushResult = failureRecords_.tryPush(record);

    // ★ Snapshot 自動生成判定: FailureRecord リングバッファの現在件数を failureCount として渡す
    const uint64_t currentFailureCount = failureRecords_.size();
    if (pushResult.success && snapshotController_.shouldTakeSnapshot(reason, currentFailureCount, ts)) {
        FailureSnapshot snap{
            .correlationIdShort = correlationIdShort,
            .stage = stage,
            .reason = reason,
            .origin = origin,
            .timestampUs = ts
        };
        failureSnapshots_.tryPush(snap);
        snapshotController_.recordSnapshotTaken(ts);
    }
}

bool TelemetryRecorder::recordFailureSnapshot(
    const FailureSnapshot& snapshot,
    uint64_t failureCount,
    uint64_t nowUs) noexcept
{
    if (!snapshotController_.shouldTakeSnapshot(snapshot.reason, failureCount, nowUs))
        return false;

    failureSnapshots_.tryPush(snapshot);
    snapshotController_.recordSnapshotTaken(nowUs);
    return true;
}

void TelemetryRecorder::recordHealth(const OrchestratorHealthSnapshot& snapshot) noexcept {
    convo::publishAtomic(lastHealthSnapshot_, snapshot, std::memory_order_release);
}

void TelemetryRecorder::recordDeferredHealth(const DeferredHealth& health) noexcept {
    convo::publishAtomic(lastDeferredHealth_, health, std::memory_order_release);
}

void TelemetryRecorder::recordRetireTimeline(const RetireTimelineRecord& record) noexcept {
    retireTimelines_.tryPush(record);
}

void TelemetryRecorder::recordRetireStall(const RetireStallSnapshot& stall) noexcept {
    convo::publishAtomic(lastRetireStall_, stall, std::memory_order_release);
}

TelemetryRecorder::TelemetrySnapshot TelemetryRecorder::captureSnapshot() const noexcept {
    TelemetrySnapshot snap{};

    snap.failureRecordCount = failureRecords_.readLatest(snap.failureRecords.data(),
        snap.failureRecords.size());
    snap.failureSnapshotCount = failureSnapshots_.readLatest(snap.failureSnapshots.data(),
        snap.failureSnapshots.size());
    snap.progressRecordCount = progressRecords_.readLatest(snap.progressRecords.data(),
        snap.progressRecords.size());
    snap.progressOverwriteCount = progressRecords_.overwriteCount();
    snap.lastHealthSnapshot = convo::consumeAtomic(lastHealthSnapshot_, std::memory_order_acquire);
    snap.lastDeferredHealth = convo::consumeAtomic(lastDeferredHealth_, std::memory_order_acquire);
    snap.lastRetireStall = convo::consumeAtomic(lastRetireStall_, std::memory_order_acquire);

    return snap;
}

} // namespace convo::isr
