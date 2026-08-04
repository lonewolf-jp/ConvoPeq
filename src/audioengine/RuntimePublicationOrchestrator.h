#pragma once

#include <atomic>
#include <cstdint>
#include <optional>
#include "RuntimePublicationState.h"
#include "TelemetryRecorder.h"
#include "PublicationAdmission.h"
#include "PublicationExecutor.h"
#include "DSPTransition.h"
#include "DSPLifetimeManager.h"
#include "ISRRuntimeSemanticSchema.h"
#include "core/RCUReader.h"
#include "core/TimeUtils.h"

class AudioEngine;

namespace convo::isr {

// ★ C-2.1: DeferredGuard — stale discard 用のガード情報
struct DeferredGuard {
    uint64_t generation;
    PublicationSequenceId sequence;
};

// ★ C-2.1: DeferredPublishSlot — sequence 番号付きの deferred publish slot
struct DeferredPublishSlot {
    PublicationAdmission::PublishRequest request;
    DeferredGuard guard;
    DiscardReason lastDiscardReason{DiscardReason::None};
    uint64_t enqueueTimestampUs{0};
};

// RuntimePublicationOrchestrator: AudioEngine レベルの publish オーケストレーション。
// Coordinator::submitPublishRequest() の実装を提供する。
// AudioEngine に注入され、Admission → Executor → DSPTransition の順で委譲する。
//
// ★ activate (DSP スロット書き換え) は publish 成功後に行う。
// ★ submitPublishRequest → evaluate → Accepted → execute の順を厳守。
//
// ★ v19: StateOwner + TelemetryRecorder の両方を保持。
//   Orchestrator が stateOwner.onXxx() + telemetryRecorder.recordXxx() を呼ぶ。
class RuntimePublicationOrchestrator {
public:
    explicit RuntimePublicationOrchestrator(AudioEngine& engine, uint64_t engineInstanceId) noexcept;

    // [work37 Phase 6] Deferred Publish TTL — 30秒超過で破棄
    static constexpr uint64_t kDeferredPublishTTLUs = 30'000'000;  // 30秒

    // trySubmit: publish 要求を試行する。
    // Admission → Accepted の場合のみ Executor → DSPTransition まで実行。
    // Deferred/Rejected の場合は caller が適切に処理するよう決定値を返す。
    // Returns: Admission::Decision (Accepted: 全処理完了 / Deferred: 保留 / Rejected*: 却下)
    [[nodiscard]] PublicationAdmission::Decision trySubmit(const PublicationAdmission::PublishRequest& req) noexcept;

    // trySubmitImpl: trySubmit / submitPublishRequest の共通実装。
    //   常に同期 publish。"submitPublishRequest() = Admission → Accepted なら同期 Publish" で
    //   API 意味論を一本化する。fire-and-forget (waitForReceipt=false) 経路は Coordinator が
    //   Builder を兼ねることによる自己待ちを招いたため削除。deferred resubmit は
    //   Coordinator が RebuildThread へハンドオフし、RebuildThread が同期 submitPublishRequest を
    //   実行する（receipt は CoordinatorLoop の processIntent が配送するため自己待ちにならない）。
    [[nodiscard]] PublicationAdmission::Decision trySubmitImpl(
        const PublicationAdmission::PublishRequest& req) noexcept;

    // ★ (a) Completion layer — ISR post-commit notifier.
    //   Single seam for "publish committed"; the ISR PublishExecutor routes here
    //   (NOT via IntentHandlerContext), keeping intent handlers HANDLER-1 (pure).
    //   Audio-thread trySubmit retains its inline completion (unchanged).
    void onPublishCommitted(PublicationSequenceId seqId) noexcept;

    // ★ A3 Step 5-3: access to the stateless publish-completion facade (ADR-D2), owned by
    //   the orchestrator (audio-thread world-publish path). Bound into IntentHandlerContext
    //   by processIntent so the ISR publish execution tail can drive activate/crossfade/retire.
    [[nodiscard]] DSPTransition& transition() noexcept { return transition_; }

    // submitPublishRequest: publish 要求を処理する (deferred は自動 enqueue)。
    //   常に同期。deferred resubmit は RebuildThread が本 API を呼ぶことで実行される。
    void submitPublishRequest(const PublicationAdmission::PublishRequest& req) noexcept;

    // hasDeferredRequest / consumeDeferredRequest: 保留中の publish 要求確認と消費
    [[nodiscard]] bool hasDeferredRequest() const noexcept { return hasDeferred_.load(std::memory_order_acquire); }
    [[nodiscard]] std::optional<PublicationAdmission::PublishRequest> consumeDeferredRequest() noexcept
    {
        if (!hasDeferred_.load(std::memory_order_acquire))
            return std::nullopt;
        hasDeferred_.store(false, std::memory_order_release);
        return deferredSlot_ ? std::optional(deferredSlot_->request) : std::nullopt;
    }

    // ★ C-2.2: shutdown 時に deferred publish を強制消去
    void clearDeferredForShutdown() noexcept;

    // ★ A-2.5: DrainAudit 用 — deferred publish 最長滞留時間
    [[nodiscard]] uint64_t getMaxDeferredAgeMs() const noexcept;

    // ★ C-2.1: 監査用 — deferred overwrite 回数
    [[nodiscard]] std::uint64_t deferredOverwriteCount() const noexcept;

    // ── StateOwner アクセサ ──
    [[nodiscard]] RuntimePublicationStateOwner& stateOwner() noexcept { return stateOwner_; }
    [[nodiscard]] const RuntimePublicationStateOwner& stateOwner() const noexcept { return stateOwner_; }

    // ★ C-1: World 退役を Pipeline Ledger に通知
    void notifyWorldRetired(uint64_t worldId) noexcept {
        stateOwner_.onRetired(worldId);
    }

    // ── TelemetryRecorder アクセサ ──
    [[nodiscard]] TelemetryRecorder& telemetryRecorder() noexcept { return telemetryRecorder_; }
    [[nodiscard]] const TelemetryRecorder& telemetryRecorder() const noexcept { return telemetryRecorder_; }

    // ★ P1-B: Admission に HealthState 参照を設定
    void setAdmissionHealthStateRef(const std::atomic<ISRHealthState>* ref) noexcept {
        admission_.setHealthStateRef(ref);
    }

    // ── 健全性スナップショット ──
    void publishHealthSnapshot(uint64_t externalReclaimedCount) noexcept;

    // ── CorrelationId 採番 ──
    [[nodiscard]] CorrelationId nextCorrelationId() noexcept;

    // ★ P1-6: 出版停滞監視 — 進捗観測の更新（非const、timerCallback から呼ぶ）
    void updateProgressObservation() noexcept {
        PublicationSequenceId current = engine_.getLastCommittedPublicationSequence();
        PublicationSequenceId last = m_lastObservedSequence.load(std::memory_order_relaxed); // NOLINT(atomic-dot-call): relaxed counter
        if (current > last) {
            m_lastObservedSequence.store(current, std::memory_order_relaxed); // NOLINT(atomic-dot-call): relaxed counter
            m_lastProgressTimestampUs.store(getCurrentTimeUs(), std::memory_order_relaxed); // NOLINT(atomic-dot-call): relaxed timestamp
        }
    }

    // ★ P1-6: 出版停滞監視 — 停滞検出（const、read-only）
    [[nodiscard]] bool isPublicationStalled() const noexcept {
        uint64_t elapsed = getCurrentTimeUs()
            - convo::consumeAtomic(m_lastProgressTimestampUs, std::memory_order_acquire);
        return elapsed >= kPublicationStallThresholdUs;
    }

    // ★ P1-6: prepareToPlay での再初期化用
    void resetProgressObservation() noexcept {
        convo::publishAtomic(m_lastProgressTimestampUs, getCurrentTimeUs(), std::memory_order_release);
    }

    // ★ P1-6: RuntimeHealthMonitor からのアクセス用
    [[nodiscard]] uint64_t getPendingIntentCount() const noexcept {
        return engine_.getRetirePendingIntentCount();
    }

    // ★ P1-6: PublicationBacklog の公開（RuntimeHealthMonitor → Orchestrator → AudioEngine → bridge）
    [[nodiscard]] uint64_t getPublicationBacklogCount() const noexcept {
        return engine_.getPublicationBacklogCount();
    }

private:
    // ★ P1-6: 出版停滞監視用フィールド（30秒以上 sequence が進まない場合に stall 検出）
    static constexpr uint64_t kPublicationStallThresholdUs = 30'000'000;
    std::atomic<PublicationSequenceId> m_lastObservedSequence {0};
    std::atomic<uint64_t> m_lastProgressTimestampUs {0};
    // ★ C-2.1: std::optional<PublishRequest> → DeferredPublishSlot
    std::optional<DeferredPublishSlot> deferredSlot_;
    std::atomic<bool> hasDeferred_{false};

    // ★ C-2.1: 監査カウンタ
    std::atomic<uint64_t> deferredOverwriteCount_{0};
    std::atomic<uint64_t> maxDeferredAgeMs_{0};

    void enqueueDeferred(const PublicationAdmission::PublishRequest& req) noexcept;

    AudioEngine& engine_;

    // ★ v19: StateOwner + TelemetryRecorder (分離)
    RuntimePublicationStateOwner stateOwner_;
    TelemetryRecorder telemetryRecorder_;

    PublicationAdmission admission_;
    PublicationExecutor executor_;
    DSPTransition transition_;
    DSPLifetimeManager lifetime_;
    convo::RCUReader publicationReader;
};

} // namespace convo::isr
