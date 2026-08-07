#include "PublicationAdmission.h"
#include "AudioEngine.h"

namespace convo::isr {

PublicationAdmission::Decision PublicationAdmission::evaluate(
    const PublishRequest& req, AudioEngine& engine,
    const convo::RuntimeReaderContext& ctx) const noexcept
{
    // 1. Shutdown check
    if (engine.isShutdownInProgress())
        return Decision::RejectedShutdown;

    // 2. Generation staleness check
    const int currentGen = convo::consumeAtomic(
        engine.rebuildRequestGeneration, std::memory_order_acquire);
    if (req.generation != currentGen)
        return Decision::RejectedStaleGeneration;

    // 3. DSP finalized check (from sealedSnapshot, not DSPCore*)
    if (req.sealedSnapshot.irLoaded && !req.sealedSnapshot.irFinalized)
        return Decision::RejectedNotFinalized;

    // 4. HealthState check (Practical-9: Admission Circuit Breaker)
    if (m_healthStateRef) {
        auto health = convo::consumeAtomic(*m_healthStateRef, std::memory_order_acquire);
        if (health == ISRHealthState::Critical) {
            // Critical: 全 publish 拒否（フェイルクローズ）
            return Decision::RejectedPressure;
        }
        if (health == ISRHealthState::Degraded) {
            // Degraded: 低優先度 publish を拒否
            //   generation==0 は存在しない（初回は 1）ため、一律 RejectedLowPriority は不可。
            //   代わりに RejectedPressure を返し、Coordinator 側で間引き制御に委ねる。
            return Decision::RejectedPressure;
        }
    }

    // 5. Pressure / throttle check (P1-6: Adaptive Backpressure)
    const bool pressureActive = convo::consumeAtomic(
        engine.retirePressurePublicationThrottleActive_, std::memory_order_acquire);
    if (pressureActive) {
        // ★ P1-6: Pressure レベル段階制御
        // RejectLowPriority: timer/crossfade publish を拒否
        // RejectMostRequests: bootstrap以外の全publish拒否
        // 現状は一律 RejectedPressure で対応
        return Decision::RejectedPressure;
    }

    // 5. Fading active check → defer
    const bool hasFading =
#if defined(CONVOPEQ_UNIT_TESTS)
        engine.testFadingRuntimePresent() ||
#endif
        engine.hasFadingRuntimeInWorld(
            engine.makeRuntimeReadHandle(ctx));
    if (hasFading)
        return Decision::DeferredFadingActive;

    return Decision::Accepted;
}

// ★ Phase-1: evaluateDeferred — Deferred publish の stale-discard を判定する。
//   design-D4 A-4 / D-9 / ADR-C4:41,64。Admission は **Decision のみ返す**
//   （ADR-C4 §Consequences: Admission = Decision only; View = Store mutation）。
//   Engine 参照を取らず、DeferredAdmissionSnapshot の5値のみで判定する。
//   判定順序: Shutdown → TTL → Generation → Sequence（design-D4 判定順序の根拠）。
//   （shutdown の強制消去は Orchestrator::clearDeferredForShutdown の補完。）
PublicationAdmission::DeferredAdmissionResult
PublicationAdmission::evaluateDeferred(const DeferredPublishMetadata& m,
                                       const DeferredAdmissionSnapshot& ctx) const noexcept
{
    // 0) Shutdown — 終了中は一切 publish しない（ADR-C4:64 snapshot の shutdown フィールド利用）。
    if (ctx.shutdown)
        return {DeferredDecision::Discard, DiscardReason::ShutdownDiscard};

    // 1) TTL (stale) — 30s 超過で破棄。ageUs は evaluate 時点 (ctx.nowUs) にて算出。
    const uint64_t ageUs = ctx.nowUs - m.enqueueTimestampUs;
    if (ageUs > ctx.ttlUs)
        return {DeferredDecision::Discard, DiscardReason::StaleDiscard};  // ★ work37: Expired を別 enum 化可能

    // 2) Generation — rebuild 時代が違う → Snapshot 失効（int 比較で -Wsign-compare 回避済み）。
    if (m.generation != ctx.currentGeneration)
        return {DeferredDecision::Discard, DiscardReason::StaleDiscard};

    // 3) Sequence — Publish済みSnapshotより古い → Reject（ISR-WORLD-001: 履歴の後戻り禁止）。
    if (m.sequence < ctx.lastSequence)
        return {DeferredDecision::Discard, DiscardReason::StaleDiscard};

    return {DeferredDecision::Ready, DiscardReason::None};
}

} // namespace convo::isr
