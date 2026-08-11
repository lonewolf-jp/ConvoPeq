#include "ISRRuntimePublicationCoordinator.h"
#include "DSPLifetimeManager.h"
#include "AudioEngine.h"
#include "ISRIntentDispatcher.h"  // ★ A3 Step 1: DispatchTable + IntentHandler
#include "RuntimePublishExecutor.h"  // ★ A3 Step 5-2: sole commit() gateway
#include "RuntimePublicationOrchestrator.h"  // ★ A3 Step 5-3: engine.runtimeOrchestrator_.transition() (Completion-layer facade)

namespace convo::isr {

void RuntimeIntentCoordinator::processIntent(
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
    // ★ work88 (P2-1 §1.1.2): pop 成功で reservation を消費（fetchSub）。
    //   quarantineFallbackQueue_ には Quarantine Intent のみ格納される（submitQuarantine が
    //   reservation-before-push で fetchAdd 済み）ため無条件 fetchSub。
    while (quarantineFallbackQueue_.pop(commonIntent)) {
        convo::fetchSubAtomic(pendingIntentCount_, std::uint64_t{1}, std::memory_order_release);
        // ★ work88 (X6 §6.6): fallback ring は Quarantine 専用（submitQuarantine のみが push）。
        //   pop 成功で quarantineRingResidencyCount_ を fetchSub（INV-X6-4）。
        convo::fetchSubAtomic(quarantineRingResidencyCount_, std::uint64_t{1}, std::memory_order_acq_rel);
        kDispatchTable[static_cast<std::size_t>(commonIntent.type)]->handle(commonIntent, ctx);
    }
    // メイン intentQueue_（MpscBoundedRing — MPSC 化済み。Observe/Publish/Quarantine/Recovery を
    //   cross-type FIFO で処理。Recovery は Handler が enqueue-only で Builder Work Queue へ転送）
    // ★ work88 (P2-1 §1.1.2): Publish を除く Intent（Observe/Quarantine）の pop 成功で
    //   reservation を消費（fetchSub）。Publish は pendingIntentCount_ に計上されない
    //   （enqueuePublicationIntent は reservation を取らない — P2-1 §1.1.1）ため fetchSub しない。
    while (intentQueue_.pop(commonIntent)) {
        // ★ work88 (X5 §6.5 / X6 §6.6): type 分岐で独立 counter を減算（INV-X5-1 / INV-X6-4）。
        //   - Publish     : publicationIntentResidencyCount_--（X5。pendingIntentCount_ は触らない）
        //   - Quarantine  : quarantineIntentResidencyCount_--（X6）+ pendingIntentCount_--（P2）
        //   - Observe/Recovery : pendingIntentCount_--（P2）
        //   減算は processIntent の while ループで一元管理（handler では行わない — HANDLER-1）。
        switch (commonIntent.type)
        {
        case IntentType::Publish:
            convo::fetchSubAtomic(publicationIntentResidencyCount_, std::uint64_t{1}, std::memory_order_acq_rel);
            break;
        case IntentType::Quarantine:
            convo::fetchSubAtomic(quarantineIntentResidencyCount_, std::uint64_t{1}, std::memory_order_acq_rel);
            convo::fetchSubAtomic(pendingIntentCount_, std::uint64_t{1}, std::memory_order_release);
            break;
        default:  // Observe / Recovery
            convo::fetchSubAtomic(pendingIntentCount_, std::uint64_t{1}, std::memory_order_release);
            break;
        }
        kDispatchTable[static_cast<std::size_t>(commonIntent.type)]->handle(commonIntent, ctx);
    }

    drainObserveDeferred(lifetimeMgr);  // ★ FUTURE-8/QUEUE-16: Observe Intent 専用 Deferred Ring 回収（Retire drain と分離）

    engine.markReceiptReclaimComplete();

    // ★ work88 (P2-1 §1.1.2): setPendingIntentCount(0) による絶対値リセットは廃止。
    //   pendingIntentCount_ は reservation ベースの正確な残数として維持される
    //   （push 成功時 fetchAdd / pop 成功時 fetchSub）。絶対値リセットは RetireIntent 混入
    //   （AudioEngine.Commit / Threading の setPendingIntentCount 上書き — §1.1.5）の温床だった。
}

// ★ FUTURE-8/QUEUE-16: Observe Intent 専用 Deferred Ring 回収（Retire drain と分離）。
void RuntimeIntentCoordinator::drainObserveDeferred(DSPLifetimeManager& lifetimeMgr) noexcept
{
    ObserveIntent deferred{};
    while (observeDeferredRing_.pop(deferred)) {
        // ★ work88 (P2-1 §1.1.2): pop 成功直後・skip 判定前に reservation を消費（fetchSub）。
        //   古い世代 / null handle で skip される場合も、enqueue 済み（reservation 済み）の
        //   Intent は pop で消費されたため fetchSub する（pop 成功数 == push 成功数の不変条件）。
        convo::fetchSubAtomic(pendingIntentCount_, std::uint64_t{1}, std::memory_order_release);
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
    const auto qResult = ctx.quarantine.executeQuarantine(
        ctx.engine.dspHandleRuntime(),
        ctx.engine.dspQuarantineManager(),
        request);

    // ★ work88 (六次レビュー — Recovery 発行経路の配線漏れ修正):
    //   quarantine 実行後に Recovery Intent を発行する（Recovery = quarantined 除外した
    //   現在の構成の再 build — INV-4 / RECOVERY-SEMANTIC-001。過去 World の rollback ではない）。
    //   buildSource は現在 publish された構成の snapshot（enqueuePublicationIntentForRuntimeCommit が
    //   保持する currentBuildSnapshot_）を引当 — quarantined DSP の過去 spec は不要（四次実測:608-610）。
    //   発行経路: submitRecoveryIntent → submitRecoveryRequest → recoveryIntentQueue_（Builder Work Queue）→
    //   Builder Loop が popRecoveryRequest で消費（RebuildDispatch.cpp:911）。
    //   注意: RecoveryIntentHandler（intentQueue_ 経由）は将来の拡張用に残すが、本経路が primary。
    //   ★ work88 (六次レビュー追記): quarantine が失敗した場合（stateChanged==false、例: 既に隔離済み
    //   または handle 無効）は Recovery を発行しない。失敗 quarantine に対する Recovery は
    //   quarantined されていない DSP の無意味な世界再構築を引き起こす（HANDLER-1: 判定は
    //   QuarantineService が唯一行う — ハンドラは結果を尊重する）。
    if (qResult.stateChanged && !request.handle.isNull())
    {
        const auto buildSource = ctx.engine.getCurrentBuildSnapshotForRecovery();
        ctx.engine.submitRecoveryIntent(request.handle, buildSource);
    }
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
//   ★ work88 (六次レビュー — ドキュメントと実装の乖離修正):
//     Recovery 発行の primary 経路は QuarantineIntentHandler が直接 submitRecoveryIntent を呼ぶ
//     （intentQueue_ を経由しない）。本 Handler は「Recovery Intent を intentQueue_ に push する」
//     将来の拡張経路に備えた転送ハンドラであり、現状は誰も intentQueue_ に Recovery Intent を
//     push しないため dead code（二重発行なし）。
void RecoveryIntentHandler::handle(const Intent& intent, IntentHandlerContext& ctx) const noexcept
{
    const auto& p = intent.payload.recovery;
    // buildSource（RuntimeBuildSnapshot 値コピー）を Builder Work Queue へ転送。
    //   build 時に convolver metadata は uiConvolverProcessor.captureBuildSnapshot() から取得（案 i）。
    ctx.engine.submitRecoveryIntent(p.quarantinedHandle, p.buildSource);
}

} // namespace convo::isr
