#pragma once

#include "PublicationAdmission.h"
#include "PublicationExecutor.h"
#include "DSPTransition.h"
#include "DSPLifetimeManager.h"

class AudioEngine;

namespace convo::isr {

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
    void notifyTransitionComplete(AudioEngine::DSPCore* currentAfterFade) noexcept;

    // hasDeferredRequest / consumeDeferredRequest: 保留中の publish 要求確認と消費 (Timer から呼ばれる)
    [[nodiscard]] bool hasDeferredRequest() const noexcept { return admission_.hasDeferred(); }
    [[nodiscard]] std::optional<PublicationAdmission::PublishRequest> consumeDeferredRequest() noexcept { return admission_.consumeDeferred(); }

private:
    AudioEngine& engine_;
    PublicationAdmission admission_;
    PublicationExecutor executor_;
    DSPTransition transition_;
    DSPLifetimeManager lifetime_;
};

} // namespace convo::isr
