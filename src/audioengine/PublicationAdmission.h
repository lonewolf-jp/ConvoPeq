#pragma once

#include "RuntimeBuildTypes.h"
#include "ISRDSPHandle.h"
#include "core/RuntimeReaderContext.h"
#include "RuntimeHealthMonitor.h"  // ★ P1-B: ISRHealthState
#include "RuntimePublicationState.h"   // ★ Phase-1: DiscardReason
#include "ISRRuntimeSemanticSchema.h"  // ★ Phase-1: PublicationSequenceId

class AudioEngine;  // forward declaration (circular dep avoid)

namespace convo::isr {

// PublicationAdmission: publish 可否判定を行う Admission コンポーネント。
// Coordinator::submitPublishRequest() から呼ばれる。
// ★ evaluate() は必須。バイパス禁止。
class PublicationAdmission {
public:
    struct PublishRequest {
        DSPHandle newDSP;
        int generation = 0;
        RuntimeBuildSnapshot sealedSnapshot;
        BuildAnalysis buildAnalysis {};           // ★ v14.0: Auto Gain 解析値
        OversamplingResult oversamplingResult {}; // ★ v14.38
        BuildDiagnostics buildDiagnostics {};     // ★ v14.37
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
        // ★ 15-P-6: publish-time の内部失敗（genuine shutdown と区別）。
        //   trySubmitImpl の executor_.publish() 失敗時に使用。admission-time の
        //   RejectedShutdown（isShutdownInProgress()）とは意味論が異なる。
        RejectedPublishFailure,
        DeferredFadingActive,
        RejectedLowPriority   // ★ P1-6: 低優先度要求拒否
    };

    explicit PublicationAdmission() noexcept = default;

    // ★ P1-B: HealthState 参照を設定（RuntimeHealthMonitor の getHealthStateRef() を渡す）
    void setHealthStateRef(const std::atomic<ISRHealthState>* ref) noexcept {
        m_healthStateRef = ref;
    }

    // evaluate: publish 可否を判定する（AudioEngine 参照が必要）。
    // Accepted 以外の場合は Coordinator が対応する。
    [[nodiscard]] Decision evaluate(const PublishRequest& req,
                                    AudioEngine& engine,
                                    const convo::RuntimeReaderContext& ctx) const noexcept;

    // ★ Phase-1: Deferred stale-discard Admission (design-D4 D-13 / ADR-C4)。
    //   追加方針: Admission は「判定のみ」を返す (ADR design principle)。Store 変更は
    //   DeferredPublishView が行う。これらは additive（既存呼び出しゼロ）のため
    //   コンパイル安全。実装は PublicationAdmission.cpp で行う (Phase-1 3a)。
    //   ※ 実装前に JUCE CMakeビルドで可視性を確認すること。
    enum class DeferredDecision {
        Ready,    // 有効 → view.consume() へ進む
        Discard,  // 破棄 → view.discard(reason) へ（理由は evaluateDeferred が返す）
        // ★ Ready / Discard の2値のみ（RetryLater は利用経路ゼロ・YAGNI のため見送り。
        //   将来 Queue 多段化等で必要になった時点で追加。→ Appendix B-2）
    };

    // 判定結果: 動作指令 + 破棄理由（Discard 時のみ discardReason が有効）。
    //   ★ 責務分離: この struct は「判定」のみ。Store 変更は View が行う
    //   （ADR principle: Admission = Decision only; View = Store mutation;
    //    ADR-C4 §Consequences）。Admission は Storage Authority を持たない。
    struct DeferredAdmissionResult {
        DeferredDecision decision{DeferredDecision::Discard};  // 明示的に設定される（初期値は Discard）
        DiscardReason discardReason{DiscardReason::None};  // decision==Discard のとき有効
    };

    // enqueue 時点の immutable metadata（DeferredPublishSlot に格納・Snapshot 化）。
    struct DeferredPublishMetadata {
        int generation{0};
        PublicationSequenceId sequence{0};
        uint64_t enqueueTimestampUs{0};
    };

    // evaluate 時点の Observation Snapshot（Engine を直参照せず POD のみ受渡）。
    //   ※ ttlUs は将来的 PolicyTTLUs（ADR-C4:85）への拡張口。現在は Orchestrator が
    //     kDeferredPublishTTLUs を詰める。Admission はこれを読んで TTL 判定する。
    struct DeferredAdmissionSnapshot {
        int currentGeneration{0};
        PublicationSequenceId lastSequence{0};
        bool shutdown{false};
        uint64_t nowUs{0};
        uint64_t ttlUs{0};
    };

    // evaluateDeferred: Deferred publish の stale-discard を判定する。
    //   Engine 参照を取らず（AudioEngine& 不要）、DeferredAdmissionSnapshot のみで判定。
    //   Decision のみ返し、Store は一切変更しない（view.consume/discard は caller が行う）。
    [[nodiscard]] DeferredAdmissionResult evaluateDeferred(const DeferredPublishMetadata& metadata,
                                                           const DeferredAdmissionSnapshot& ctx) const noexcept;

    // Deferred Queue は PublicationAdmission から RuntimePublicationOrchestrator へ移設済み (PR-7)。
    // Admission は publish 可否判定のみ責務とする。

private:
    // ★ P1-B: HealthMonitor の統合 HealthState 参照
    const std::atomic<ISRHealthState>* m_healthStateRef = nullptr;
};

} // namespace convo::isr
