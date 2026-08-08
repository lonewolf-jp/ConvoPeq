#include "ISRRuntimePublicationCoordinator.h"
#include "DSPLifetimeManager.h"
#include "AudioEngine.h"
#include "ISRIntentDispatcher.h"  // ★ A3 Step 1: DispatchTable + IntentHandler
#include "RuntimePublishExecutor.h"  // ★ A3 Step 5-2: sole commit() gateway
#include "RuntimePublicationOrchestrator.h"  // ★ A3 Step 5-3: engine.runtimeOrchestrator_.transition() (Completion-layer facade)

namespace convo::isr {

void RuntimePublicationCoordinator::processIntent(
    AudioEngine& engine,
    DSPLifetimeManager& lifetimeMgr) noexcept
{
    // ★ A3 Step 1: Observe routing now flows through the DispatchTable (pure routing).
    //     Queues (observeIntentQueue_/observeFallbackQueue_) are unchanged → behavior preserved.
    // ★ A3 Step 4: IntentHandlerContext now carries QuarantineService (QSVC-2 execution boundary).
    // ★ A3 Step 5-3: IntentHandlerContext also binds the orchestrator's stateless
    //   publish-completion facade (DSPTransition), so the ISR publish execution tail reaches
    //   DSP activate/crossfade/retire. Completion-notify is routed through the orchestrator
    //   (Completion layer), NOT through IntentHandlerContext (Handler stays HANDLER-1 / pure).
    IntentHandlerContext ctx{engine, lifetimeMgr, quarantineService_, engine.runtimeOrchestrator_->transition()};

    auto dispatchObserve = [&](const ObserveIntent& obs) noexcept {
        const auto currentEpoch = currentPublicationEpoch();  // DISPATCH-1: epoch-FIFO filter lives in the Dispatcher.
        if (obs.epoch < currentEpoch || obs.handle.isNull())
            return;
        Intent intent{};
        intent.type = IntentType::Observe;
        intent.payload.observe = ObservePayload{obs.handle, obs.epoch};
        kDispatchTable[static_cast<std::size_t>(IntentType::Observe)]->handle(intent, ctx);
    };

    ObserveIntent intent;
    while (observeIntentQueue_.pop(intent))
        dispatchObserve(intent);

    while (observeFallbackQueue_.pop(intent))
        dispatchObserve(intent);

    // ★ A3 Step 4: drain Quarantine/Publish/Recovery Intents from the common intentQueue_.
    //   Observe stays on its dedicated SPSC rings — DoD #4/#7 (single intentQueue + cross-type FIFO)
    //   is deferred to FUTURE-10's unified Overflow Policy migration.
    Intent commonIntent;
    // ★ work88 (FUTURE-10): Quarantine 専用 fallback ring の drain（drop 禁止の退避先）。
    //   quarantine は安全要件（bad DSP のアクセス禁止）のため、intentQueue_ より先に処理する。
    while (quarantineFallbackQueue_.pop(commonIntent))
        kDispatchTable[static_cast<std::size_t>(commonIntent.type)]->handle(commonIntent, ctx);
    // メイン intentQueue_（MpscBoundedRing — MPSC 化済み）
    while (intentQueue_.pop(commonIntent))
        kDispatchTable[static_cast<std::size_t>(commonIntent.type)]->handle(commonIntent, ctx);

    drainObserveDeferred(lifetimeMgr);  // ★ FUTURE-8/QUEUE-16: Observe Intent 専用 Deferred Ring 回収（Retire drain と分離）

    engine.markReceiptReclaimComplete();

    setPendingIntentCount(0);
}

// ★ FUTURE-8/QUEUE-16: Observe Intent 専用 Deferred Ring 回収（Retire drain と分離）。
void RuntimePublicationCoordinator::drainObserveDeferred(DSPLifetimeManager& lifetimeMgr) noexcept
{
    ObserveIntent deferred{};
    while (observeDeferredRing_.pop(deferred)) {
        const auto currentEpoch = currentPublicationEpoch();
        if (deferred.epoch < currentEpoch || deferred.handle.isNull())
            continue;
        lifetimeMgr.retireByHandle(deferred.handle);  // handle 保持 ── 正しい retire 対象を特定
    }
}

// ★ A3 Step 1: IntentHandler definitions routed by kDispatchTable.
void ObserveIntentHandler::handle(const Intent& intent, IntentHandlerContext& ctx) const noexcept
{
    // Behavior-preserving (A3 Step 1): identical retire target to the pre-A3 inline loop.
    ctx.lifetimeMgr.retireByHandle(intent.payload.observe.handle);
}

// ★ A3 Step 4: QuarantineIntentHandler — async Quarantine execution (QSVC-1: State + Audit single tx).
//   Sources DSPHandleRuntime / DSPQuarantineManager through HandlerContext.engine (Authority boundary).
void QuarantineIntentHandler::handle(const Intent& intent, IntentHandlerContext& ctx) const noexcept
{
    QuarantineService::QuarantineRequest request{
        intent.payload.quarantine.handle,
        intent.payload.quarantine.reason,
        intent.payload.quarantine.contextEpoch  // ContextEpoch-DQ-1: epoch-stamped for audit
    };
    // QSVC-2: executeQuarantine is the sole State+Audit mutation path;
    //   Coordinator (NonRT) drives it — never bypassed from the handler side.
    ctx.quarantine.executeQuarantine(
        ctx.engine.dspHandleRuntime(),
        ctx.engine.dspQuarantineManager(),
        request);
}

// ★ A3 Step 5-2: PublishIntentHandler — Execution only (HANDLER-1). Calls PublishExecutor
//   → RuntimeWorldAuthority::commit(). Reads PublishPayload fixed at enqueue (Step 5-1);
//   does NOT call currentPublicationEpoch()/RuntimeBuilder/Queue/Retry — no new commit() caller.
// ★ A3 Step 5-3: Completion-notify routed through orchestrator.onPublishCommitted (Completion layer).
void PublishIntentHandler::handle(const Intent& intent, IntentHandlerContext& ctx) const noexcept {
    PublishExecutor{}.executePublish(ctx.engine.worldAuthority(), intent, ctx);
}

} // namespace convo::isr
