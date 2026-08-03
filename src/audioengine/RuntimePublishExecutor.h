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
        //   INVARIANT: RuntimeStateOwner is moved exactly once below (into the RuntimeStore
        //   publishWorld()); the ISR commit() below only reads the address — non-owning.
        auto owner = authority.ownerChannel().take(
            OwnerChannelKey{ intent.sequenceId,
                             static_cast<std::uint32_t>(intent.payload.publish.epoch),
                             intent.payload.publish.mappedGeneration });

        // ★ D3: resolve newWorld for the metadata commit — the async owner when present, else the
        //   PendingPublishRegistry / sealed snapshot fallback (preserving 5-2/5-3 behavior while
        //   producers are still on the legacy route).
        const auto* newWorld = owner ? owner.get()
                                     : authority.registry().lookup(intent.sequenceId);
        if (newWorld == nullptr)
            newWorld = p.newWorld;
        authority.commit(PublishAuthority::Granted,
                         p.boundary,
                         newWorld,
                         p.version,
                         intent.sequenceId,
                         p.epoch,
                         p.mappedGeneration);
        // ★ B3/A1: THE sole RuntimeStore store-swap (RuntimeStore::WriteAccess::publishAndSwap).
        //   Ownership moves exactly once here: owner → runtimeStore. The producer has switched
        //   to the async enqueue route this is live; otherwise owner is empty and this branch
        //   stays inert — guaranteeing two active store-swaps never coexist at runtime.
        if (owner)
        {
            auto coordinator = ctx.engine.makeRuntimePublicationCoordinator();
            (void)coordinator.publishWorld(std::move(owner));
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
            decision,
            lifetimeMgr);
        ctx.engine.advanceRetireEpoch();

        // ★ (a): Completion-notify — ISR post-commit. Routed through the orchestrator
        //   (Completion layer), NOT via IntentHandlerContext (Handler stays pure / HANDLER-1).
        ctx.engine.runtimeOrchestrator_->onPublishCommitted(intent.sequenceId);
    }
};

} // namespace convo::isr
