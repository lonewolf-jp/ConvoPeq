#pragma once

#include <cstdint>
#include "PublicationAdmission.h"
#include "PublicationExecutor.h"
#include "DSPTransition.h"
#include "DSPLifetimeManager.h"
#include "ISRRuntimeSemanticSchema.h"
#include "core/RCUReader.h"

class AudioEngine;

namespace convo::isr {

// ★ C-2.1: DiscardReason — deferred publish が破棄された理由
enum class DiscardReason : uint8_t {
    None,
    ShutdownDiscard,
    StaleDiscard,
    SupersededDiscard
};

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
class RuntimePublicationOrchestrator {
public:
    explicit RuntimePublicationOrchestrator(AudioEngine& engine) noexcept;

    // trySubmit: publish 要求を試行する。
    // Admission → Accepted の場合のみ Executor → DSPTransition まで実行。
    // Deferred/Rejected の場合は caller が適切に処理するよう決定値を返す。
    // Returns: Admission::Decision (Accepted: 全処理完了 / Deferred: 保留 / Rejected*: 却下)
    [[nodiscard]] PublicationAdmission::Decision trySubmit(const PublicationAdmission::PublishRequest& req) noexcept;

    // submitPublishRequest: publish 要求を処理する (deferred は自動 enqueue)。
    void submitPublishRequest(const PublicationAdmission::PublishRequest& req) noexcept;

    // notifyTransitionComplete: クロスフェード完了時の処理 (Timer から呼ばれる)
    // 完了後に deferred publish request があれば自動的に再試行する。
    void notifyTransitionComplete(AudioEngine::DSPCore* currentAfterFade) noexcept;

    // hasDeferredRequest / consumeDeferredRequest: 保留中の publish 要求確認と消費
    [[nodiscard]] bool hasDeferredRequest() const noexcept { return hasDeferred_; }
    [[nodiscard]] std::optional<PublicationAdmission::PublishRequest> consumeDeferredRequest() noexcept
    {
        if (!hasDeferred_)
            return std::nullopt;
        hasDeferred_ = false;
        return deferredSlot_ ? std::optional(deferredSlot_->request) : std::nullopt;
    }

    // ★ C-2.2: shutdown 時に deferred publish を強制消去
    void clearDeferredForShutdown() noexcept;

    // ★ A-2.5: DrainAudit 用 — deferred publish 最長滞留時間
    [[nodiscard]] uint64_t getMaxDeferredAgeMs() const noexcept;

    // ★ C-2.1: 監査用 — deferred overwrite 回数
    [[nodiscard]] std::uint64_t deferredOverwriteCount() const noexcept;

private:
    // ★ C-2.1: std::optional<PublishRequest> → DeferredPublishSlot
    std::optional<DeferredPublishSlot> deferredSlot_;
    bool hasDeferred_ = false;

    // ★ C-2.1: 監査カウンタ
    std::atomic<uint64_t> deferredOverwriteCount_{0};
    std::atomic<uint64_t> maxDeferredAgeMs_{0};

    void enqueueDeferred(const PublicationAdmission::PublishRequest& req) noexcept;

    AudioEngine& engine_;
    PublicationAdmission admission_;
    PublicationExecutor executor_;
    DSPTransition transition_;
    DSPLifetimeManager lifetime_;
    convo::RCUReader publicationReader;
};

} // namespace convo::isr
