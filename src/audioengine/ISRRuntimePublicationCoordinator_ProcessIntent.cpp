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

    // ★ work88 (FUTURE-10 / Phase 7): Observe 統合 — 専用 SPSC リング（observeIntentQueue_ /
    //   observeFallbackQueue_）の while-pop を廃止し、共通 intentQueue_ (MPSC) に一本化。
    //   Observe は Dispatcher（ObserveIntentHandler）経由で処理され、epoch-FIFO フィルタは
    //   Handler 側（Dispatcher 層）に維持。observeDeferredRing_ は overflow 専用として
    //   drainObserveDeferred が引き続き回収する。
    // ★ A3 Step 4: drain Quarantine/Publish/Recovery/Observe Intents from the common intentQueue_.
    Intent commonIntent;
    // ★ work88 (FUTURE-10): Quarantine 専用 fallback ring の drain（drop 禁止の退避先）。
    //   quarantine は安全要件（bad DSP のアクセス禁止）のため、intentQueue_ より先に処理する。
    while (quarantineFallbackQueue_.pop(commonIntent))
        kDispatchTable[static_cast<std::size_t>(commonIntent.type)]->handle(commonIntent, ctx);
    // メイン intentQueue_（MpscBoundedRing — MPSC 化済み。Observe/Publish/Quarantine/Recovery を
    //   cross-type FIFO で処理。Recovery は Handler が enqueue-only で Builder Work Queue へ転送）
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
    // ★ work88 (FUTURE-10 / Phase 7): dispatchObserve の epoch-FIFO フィルタを Dispatcher 層に維持。
    //   Observe 統合により共通 intentQueue_ から流れるため、ここで世代（epoch）逆転・無効 handle を
    //   除外する（DISPATCH-1: epoch-FIFO filter lives in the Dispatcher）。
    const auto currentEpoch = ctx.engine.currentPublicationEpoch();
    if (intent.payload.observe.epoch < currentEpoch || intent.payload.observe.handle.isNull())
        return;
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

// ★ work88 (FUTURE-10 / Phase 7): RecoveryIntentHandler — enqueue-only + Builder Work Queue 転送。
//   二次レビュー NO-GO（案 A の intentQueue_ 再 enqueue で無限循環）→ enqueue-only に変更。
//   Recovery は Dispatcher 経路（CoordinatorLoop pop）で処理せず、Builder の作業として
//   Builder Work Queue（recoveryIntentQueue_）へ転送する（Intent Queue と Builder Work Queue の分離）。
//   HANDLER-1: Decision/World 書換禁止 — submitRecoveryRequest（push のみ）を呼ぶ。
//   循環排除: pop 元（intentQueue_）とは異なるキュー（recoveryIntentQueue_）に書くため
//   Dispatcher のループに再流入しない。
void RecoveryIntentHandler::handle(const Intent& intent, IntentHandlerContext& ctx) const noexcept
{
    const auto& p = intent.payload.recovery;
    // buildSource（RuntimeBuildSnapshot 値コピー）を Builder Work Queue へ転送。
    //   build 時に convolver metadata は uiConvolverProcessor.captureBuildSnapshot() から取得（案 i）。
    ctx.engine.submitRecoveryIntent(p.quarantinedHandle, p.buildSource);
}

} // namespace convo::isr
