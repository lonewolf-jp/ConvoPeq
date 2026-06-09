#pragma once

#include "RuntimeBuildTypes.h"
#include "ISRDSPHandle.h"
#include "core/RuntimeReaderContext.h"

class AudioEngine;  // forward declaration (circular dep avoid)

namespace convo::isr {

// PublicationAdmission: publish 可否判定を行う Admission コンポーネント。
// Coordinator::submitPublishRequest() から呼ばれる。
// ★ evaluate() は必須。バイパス禁止。
class PublicationAdmission {
public:
    struct PublishRequest {
        DSPHandle newDSP;  // DSPHandle (Phase2: Execution Path Handle Normalization)
        int generation = 0;
        RuntimeBuildSnapshot sealedSnapshot;
    };

    // ★ P1-6: Pressure レベル (Adaptive Backpressure)
    enum class PressureLevel : uint8_t {
        Ready = 0,          // 通常運用
        Pressure,           // retirePressurePublicationThrottleActive_ 有効化
        RejectLowPriority,  // timer/crossfade publish を拒否
        RejectMostRequests  // bootstrap以外の全publish拒否
    };

    enum class Decision {
        Accepted,
        RejectedStaleGeneration,
        RejectedNotFinalized,
        RejectedPressure,
        RejectedShutdown,
        DeferredFadingActive,
        RejectedLowPriority   // ★ P1-6: 低優先度要求拒否
    };

    explicit PublicationAdmission() noexcept = default;

    // evaluate: publish 可否を判定する（AudioEngine 参照が必要）。
    // Accepted 以外の場合は Coordinator が対応する。
    [[nodiscard]] Decision evaluate(const PublishRequest& req,
                                    AudioEngine& engine,
                                    const convo::RuntimeReaderContext& ctx) const noexcept;

    // Deferred Queue は PublicationAdmission から RuntimePublicationOrchestrator へ移設済み (PR-7)。
    // Admission は publish 可否判定のみ責務とする。
};

} // namespace convo::isr
