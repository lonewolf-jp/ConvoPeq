#pragma once

#include <cstdint>
#include <chrono>

namespace convo::isr {

// ★ C-2.1: DiscardReason — deferred publish が破棄された理由
enum class DiscardReason : uint8_t {
    None,
    ShutdownDiscard,
    StaleDiscard,
    SupersededDiscard
};

// ★ PublicationLedger: 一次情報源。ProgressRecord は副産物。
struct PublicationLedger {
    uint64_t submittedCount{0};
    uint64_t builtCount{0};
    uint64_t validatedCount{0};
    uint64_t publishedCount{0};
    uint64_t retiredCount{0};
    uint64_t reclaimedCount{0};
    uint64_t rejectedCount{0};
    uint64_t validationFailedCount{0};
    uint64_t executorFailedCount{0};
    uint64_t droppedProgressRecordCount{0};     // リング満杯によるドロップ累積数
    uint64_t firstProgressDropUs{0};            // 初回ドロップ時刻
    uint64_t lastProgressDropUs{0};             // 最終ドロップ時刻
};

// ★ OrchestratorForwardProgress: 7段階カウンタ + Stage-gap detection
struct OrchestratorForwardProgress {
    uint64_t submittedCount{0};      // Orchestrator が受付
    uint64_t builtCount{0};          // Builder が構築完了
    uint64_t validatedCount{0};      // Validator 通過
    uint64_t executedCount{0};       // Executor 実行
    uint64_t publishedCount{0};      // Coordinator publish 成功
    uint64_t retiredCount{0};        // retire 完了
    uint64_t reclaimedCount{0};      // reclaim 完了 (GC完了)
    uint32_t executorQueueDepth{0};
    uint64_t lastProgressTimestampUs{0};

    // 停止位置の診断 (7段階)
    enum class StuckStage : uint8_t {
        None,
        Builder,
        Validator,
        Executor,
        Coordinator,
        Retire,
        Reclaim
    };

    [[nodiscard]] StuckStage detectStuckStage() const noexcept {
        if (submittedCount > builtCount)     return StuckStage::Builder;
        if (builtCount > validatedCount)     return StuckStage::Validator;
        if (validatedCount > executedCount)  return StuckStage::Executor;
        if (executedCount > publishedCount)  return StuckStage::Coordinator;
        if (publishedCount > retiredCount)   return StuckStage::Retire;
        if (retiredCount > reclaimedCount)   return StuckStage::Reclaim;
        return StuckStage::None;
    }
};

// ★ CorrelationId: 128bit 相当の相関ID。wrap不可。
struct CorrelationId {
    uint64_t engineInstanceId{0};   // 上位: Engine インスタンス識別子
    uint64_t localCounter{0};       // 下位: 単調増加カウンタ (wrap不可)

    // 出力時短縮 (Evidence等)
    [[nodiscard]] uint64_t shortValue() const noexcept {
        return localCounter;
    }

    [[nodiscard]] bool isValid() const noexcept {
        return engineInstanceId != 0 || localCounter != 0;
    }
};

// ★ RuntimePublicationState: StateOwner が所有する State + Ledger
struct RuntimePublicationState {
    PublicationLedger ledger;
    OrchestratorForwardProgress progress;
    CorrelationId lastCorrelationId{};
    uint64_t engineInstanceId{0};
    uint64_t shutdownGeneration{0};

private:
    friend class RuntimePublicationStateOwner;
    // StateOwner のみが書込可能
};

// ============================================================
// ★ RuntimePublicationStateOwner: State + Ledger のみ (軽量)
//    GodObject化防止。Telemetry/Progress/Failure は呼ばない。
// ============================================================
class RuntimePublicationStateOwner {
    friend class RuntimePublicationOrchestrator;  // 唯一の書込権限者
public:
    explicit RuntimePublicationStateOwner(uint64_t engineInstanceId) noexcept
        : state_{}
    {
        state_.engineInstanceId = engineInstanceId;
    }

    // ── onXxx(): State + Ledger 更新のみ。Telemetry/Progress/Failure は呼ばない ──
    void onSubmitted(uint64_t correlationIdShort) noexcept {
        state_.ledger.submittedCount++;
        state_.progress.submittedCount++;
        state_.progress.lastProgressTimestampUs = timestampUs();
    }

    void onBuilt(uint64_t correlationIdShort) noexcept {
        state_.ledger.builtCount++;
        state_.progress.builtCount++;
        state_.progress.lastProgressTimestampUs = timestampUs();
    }

    void onValidated(uint64_t correlationIdShort) noexcept {
        state_.ledger.validatedCount++;
        state_.progress.validatedCount++;
        state_.progress.lastProgressTimestampUs = timestampUs();
    }

    void onPublished(uint64_t correlationIdShort) noexcept {
        state_.ledger.publishedCount++;
        state_.progress.publishedCount++;
        state_.progress.lastProgressTimestampUs = timestampUs();
    }

    void onRetired(uint64_t correlationIdShort) noexcept {
        state_.ledger.retiredCount++;
        state_.progress.retiredCount++;
        state_.progress.lastProgressTimestampUs = timestampUs();
    }

    void onReclaimed(uint64_t correlationIdShort) noexcept {
        state_.ledger.reclaimedCount++;
        state_.progress.reclaimedCount++;
        state_.progress.lastProgressTimestampUs = timestampUs();
    }

    void onRejected(uint64_t correlationIdShort) noexcept {
        state_.ledger.rejectedCount++;
    }

    void onValidationFailed(uint64_t correlationIdShort) noexcept {
        state_.ledger.validationFailedCount++;
    }

    void onExecutorFailed(uint64_t correlationIdShort) noexcept {
        state_.ledger.executorFailedCount++;
    }

    // ── ドロップ通知 ──
    void notifyProgressDrop(uint64_t nowUs) noexcept {
        state_.ledger.droppedProgressRecordCount++;
        if (state_.ledger.firstProgressDropUs == 0)
            state_.ledger.firstProgressDropUs = nowUs;
        state_.ledger.lastProgressDropUs = nowUs;
    }

    // ── ExecutorQueueDepth ──
    void setExecutorQueueDepth(uint32_t depth) noexcept {
        state_.progress.executorQueueDepth = depth;
    }

    // ── ShutdownGeneration ──
    void setShutdownGeneration(uint64_t generation) noexcept {
        state_.shutdownGeneration = generation;
    }

    // ── CorrelationId ──
    void setLastCorrelationId(const CorrelationId& cid) noexcept {
        state_.lastCorrelationId = cid;
    }

    // ── 読み取り ──
    [[nodiscard]] const RuntimePublicationState& state() const noexcept { return state_; }

private:
    RuntimePublicationState state_;

    static uint64_t timestampUs() noexcept {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
    }
};

} // namespace convo::isr
