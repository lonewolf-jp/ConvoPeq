#pragma once

#include <optional>
#include "RuntimeBuildTypes.h"

class AudioEngine;  // forward declaration (circular dep avoid)

namespace convo::isr {

// PublicationAdmission: publish 可否判定を行う Admission コンポーネント。
// Coordinator::submitPublishRequest() から呼ばれる。
// ★ evaluate() は必須。バイパス禁止。
class PublicationAdmission {
public:
    struct PublishRequest {
        void* newDSP = nullptr;  // AudioEngine::DSPCore*
        int generation = 0;
        RuntimeBuildSnapshot sealedSnapshot;
    };

    enum class Decision {
        Accepted,
        RejectedStaleGeneration,
        RejectedNotFinalized,
        RejectedPressure,
        RejectedShutdown,
        DeferredFadingActive
    };

    explicit PublicationAdmission() noexcept = default;

    // evaluate: publish 可否を判定する（AudioEngine 参照が必要）。
    // Accepted 以外の場合は Coordinator が対応する。
    [[nodiscard]] Decision evaluate(const PublishRequest& req,
                                    AudioEngine& engine) const noexcept;

    // Deferred Queue: 常に最新1件のみ保持。
    void enqueueDeferred(const PublishRequest& req) noexcept
    {
        deferredRequest_ = req;
        hasDeferred_ = true;
    }

    [[nodiscard]] bool hasDeferred() const noexcept { return hasDeferred_; }
    [[nodiscard]] std::optional<PublishRequest> consumeDeferred() noexcept
    {
        if (!hasDeferred_)
            return std::nullopt;
        hasDeferred_ = false;
        return deferredRequest_;
    }

private:
    std::optional<PublishRequest> deferredRequest_;
    bool hasDeferred_ = false;
};

} // namespace convo::isr
