#pragma once

#include "ISRRuntimeWorldAuthority.h"   // ★ A3 Step 5-2: RuntimeWorldAuthority + PublishAuthority + commit() + registry()
#include "ISRIntentDispatcher.h"        // IntentHandlerContext
#include "DSPTransition.h"              // ctx.transition (publish-completion facade, ADR-D2)
#include "DSPLifetimeManager.h"         // ISR lifetime mgr
#include "ISRDSPHandle.h"               // DSPHandleRuntime::resolve -> ResolvedDSP (.instance / .valid)
#include "CrossfadeAuthority.h"         // reconstruct typed Decision from PublishDecisionSnapshot
#include "AudioEngine.h"                // AudioEngine::DSPCore*, dspHandleRuntime(), advanceRetireEpoch(), runtimeOrchestrator_
#include "RuntimePublicationOrchestrator.h"  // onPublishCommitted (Completion layer, Step 5-3)

namespace convo::isr {

// ★ A3 Step 5-3: sole Execution gateway to RuntimeWorldAuthority::commit() on the Publish path.
//   HANDLER-1 boundary: PublishIntentHandler calls this — never the Coordinator directly.
//   Reads ONLY the publish payload fixed at enqueue (Decision Snapshot + handles); never re-decides.
//   Execution tail (commit → unregister → onPublishCompleted → advanceRetireEpoch →
//   completion-notify) now lives here (ISR), moved off the audio-thread trySubmit.
struct PublishExecutor {
    void executePublish(RuntimeWorldAuthority& authority,
                        const Intent& intent,
                        IntentHandlerContext& ctx) const noexcept
    {
        const auto& p = intent.payload.publish;


        // ★ B3/A1: acquire SOLE ownership from OwnerChannel. The Non-RT producer already
        //   transferred the immutable world into ownerChannel_ keyed by {seq, epoch, mappedGen}.
        //   INVARIANT: RuntimeStateOwner is moved exactly once below (into
        //   RuntimeWorldAuthority::publish — sole physical store-swap, INV-X4-3).
        auto owner = authority.ownerChannel().take(
            OwnerChannelKey{ intent.sequenceId,
                             static_cast<std::uint32_t>(intent.payload.publish.epoch),
                             intent.payload.publish.mappedGeneration });
        const bool hasOwner = static_cast<bool>(owner);

        // ★ D3: resolve newWorld for the metadata commit — the async owner when present, else the
        //   PendingPublishRegistry / sealed snapshot fallback (preserving 5-2/5-3 behavior while
        //   producers are still on the legacy route).
        //   ★ X4-B: didPublish で *newWorld を deref するため typed（const RuntimeState*）に統一する
        //     （registry.lookup / intent payload は const void* を返す — static_cast で型復元）。
        const auto* newWorld = owner ? owner.get()
                                     : static_cast<const RuntimeState*>(authority.registry().lookup(intent.sequenceId));
        if (newWorld == nullptr)
            newWorld = static_cast<const RuntimeState*>(p.newWorld);

        // ★ work88 (X4-B §6.4 / X4-B-5): 一時生成 coordinator（makeRuntimePublishAuthority()）を廃止し、
        //   RuntimeWorldAuthority が sole physical publish gateway（INV-X4-2）。publish() 内部で
        //   commit metadata（ISR currentWorld_ 更新）+ publishAndSwap（physical store swap）を束ねる
        //   （commit-before-swap ordering — Test 7）。PublishExecutor 側の authority.commit() は
        //   publish() が内包するため削除（commit 二重化防止・二十二次レビュー必須修正1）。
        //   ★ seal はここで実行（RuntimeState 完全型 — AudioEngine.h include 済み）。
        //   ★ didPublish / willRetire / retire は publish() の外（Execution tail）で Bridge 経由で
        //     実行する（X4-B の publish() 責務限定 — validate/commit/swap/return oldWorld・二十次 §20）。
        RuntimeState* oldWorld = nullptr;
        if (hasOwner)
        {
            const_cast<RuntimeState*>(owner.get())->sealRecursively();   // PR-5: publish 前 immutable 化
            bool committed = false;
            oldWorld = authority.publish(std::move(owner),
                                         RuntimeWorldAuthority::PublishMetadata{
                                             p.boundary, p.version, intent.sequenceId,
                                             p.epoch, p.mappedGeneration },
                                         &committed);
            // ★ work88（監査軽微指摘2）: publish() は seqId==0 / commit Faulted のみ失敗
            //   （producer 保証 + FIFO で到達不能）。失敗時は world が publish() 内で破棄されるため
            //   *newWorld を deref しない（dangling deref 防止）。committed==true のみ bridge 実行。
            AudioEngine::RuntimePublicationBridge bridge{ ctx.engine, ctx.engine.runtimePublicationValidator_ };
            if (committed)
            {
                // ★ X4-B: 従来は core publishWorld が内部で実行していた didPublish/willRetire/retire。
                //   Bridge は残る（十八次別視点8調査）— X4-B 後も didPublish/willRetire/retire を
                //   PublishExecutor の Execution tail から呼ぶ。
                bridge.didPublishRuntimeNonRt(*newWorld);
                bridge.willRetireRuntimeNonRt(oldWorld);
                bridge.retireRuntimePublishWorldNonRt(oldWorld, false);
            }
        }
        // ★ D2: commit succeeded ⇒ drop the pending registry entry
        authority.registry().unregister(intent.sequenceId);

        // ★ A3 Step 5-3 Execution tail (moved from audio-thread trySubmit).
        //   Decision Snapshot (p.decision) is fixed at enqueue (HANDLER-1 read-only).
        //   DSPHandle -> DSPCore* resolved here from the stable handle table (lifetime
        //   guaranteed: retire-after-commit ordering; table holds the DSPCore until this ISR call).
        DSPLifetimeManager lifetimeMgr(ctx.engine);
        const auto newResolved = ctx.engine.dspHandleRuntime().resolve(p.decision.newHandle);
        const auto oldResolved = ctx.engine.dspHandleRuntime().resolve(p.decision.oldHandle);
        CrossfadeAuthority::Decision decision{
            p.decision.needsCrossfade,
            p.decision.oldHasIR,
            p.decision.newHasIR,
            p.decision.fadeTimeSec
        };
        ctx.transition.onPublishCompleted(
            newResolved.valid ? static_cast<AudioEngine::DSPCore*>(newResolved.instance) : nullptr,
            oldResolved.valid ? static_cast<AudioEngine::DSPCore*>(oldResolved.instance) : nullptr,
            p.decision.oldHandle,
            decision,
            lifetimeMgr);
        ctx.engine.advanceRetireEpoch();

        // ★ (a): Completion-notify — ISR post-commit. Routed through the orchestrator
        //   (Completion layer), NOT via IntentHandlerContext (Handler stays pure / HANDLER-1).
        ctx.engine.runtimeOrchestrator_->onPublishCommitted(intent.sequenceId);
    }
};

} // namespace convo::isr
