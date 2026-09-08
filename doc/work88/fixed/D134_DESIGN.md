# D134 — Deferred Fading Recovery Design

**Properties:** read-only / design-only / forensic. `production source変更：0`.
Purpose: close the two design gaps left open by D133-1 —
1. *why* `DeferredFadingActive` is never released, and
2. the *conditions* under which a deferred request returns to publish —
by partitioning the mechanism into **Crossfade / Deferred / Generation / Recovery / Defect** layers.

All findings are backed by line-level source + the measured trace `evidence/D133-1_6burst_diag.log`.
No speculation.

---

## TL;DR Design Contract (D135 handoff)

A deferred request that returned `DeferredFadingActive` can only safely be re-driven to publish
when **both** hold simultaneously — and these are NOT currently guaranteed together:

> **P0 contract (the single gate a deferred request needs):**
> `hasFading` was *observed true at evaluate time* AND
> the **same** rebuild thread re-checks admission **after a world commit that set
> `RuntimeWorld.topology.fadingRuntimeUuid = 0`**.

Today the re-check never happens with a zero-fading world because:
- the crossfade-timeout recovery's idle-publish (`publishIdleWorldOnly(...,HardReset)`) writes
  `fadingRuntimeUuid=0`, but
- it fires **after** (or concurrent with) the deferred re-drive of the same generation, and
- the re-drive re-enters `processDeferredAdmission` in the **same cycle** with `submitPublishRequest`
  → `enqueueDeferred` (overwrite) with **no retry budget, no wait for world commit**, so `hasFading`
  is re-read still-true → 28-iteration overwrite busy-loop (gen8) that only terminates when the
  memory-starved rebuild thread is pre-empted by the `[MEM_SNAP]` health tick, **not** by fading clearing.

The minimal D135 fix must (a) make timeout-recovery's idle-publish commit observable to the
deferred re-drive of the **same generation**, and (b) bound the re-drive. Details per section.

---

## D134-1 — Crossfade timeout recovery: complete causal chain (line-level)

### The 4-layer state model (every atom that exists)

| Layer | Atom (type) | File:decl | Writers | Readers / role |
|---|---|---|---|---|
| Crossfade | `crossfadeRuntime_` (`CrossfadeRuntime`) | AudioEngine.h | Timer.cpp:960 `complete()`, 961 `setStartDelayBlocks(0)`, 962 `setDryHoldSamples(0)`, `:920 start()` | Timer.cpp:928 `tryCompleteFade`, :1647 `getQueuedFadeTimeSec` |
| Crossfade | `crossfadeAuthorityRuntime_` (`CrossfadeAuthority`) | AudioEngine.h | Timer.cpp:964 `unregisterCrossfade` | Timer.cpp:944 `getActiveCrossfades` |
| Deferred | `fadingRuntimeDSPSlot` (`atomic<DSPCore*>`) | AudioEngine.h:2207 | AudioEngine.h:2211 `exchangeFadingRuntimeDSPSlot`, Timer.cpp:961-966 CAS `nullptr`, CtorDtor.cpp:156, ReleaseResources.cpp:156 | DSPTransition.h:170 (CAS read), AudioEngine.cpp:154 (consume+retire) |
| **World** | **`RuntimeWorld.topology.fadingRuntimeUuid` (uint64)** | RuntimeTransition.h:36 / ISRRuntimeSemanticSchema.h:209 | **RuntimeBuilder.cpp:220 (ONLY)** | **AudioEngine.h:3304 `hasFadingRuntimeInWorld` (ONLY consumer)** |
| Deferred | `deferredSlot_` (`optional<DeferredPublishSlot>`) | RuntimePublicationOrchestrator.h:255 | enqueueDeferred (install), finishView (reset) | peekDeferred (borrow) |

### The single authority for `hasFading`

**D134-2-A — authoritative state is `RuntimeWorld.topology.fadingRuntimeUuid`.**
`hasFadingRuntimeInWorld` (AudioEngine.h:3304):
```cpp
return (runtimeWorld != nullptr) && (runtimeWorld->topology.fadingRuntimeUuid != 0);
```
`fadingRuntimeDSPSlot`, `crossfadeRuntime_`, `crossfadeAuthorityRuntime_` are **NOT consulted** by admission. They are RT-side / timer-side control state; the World is the only admitted truth. (Verified: `grep -rn hasFadingRuntimeInWorld` → only PublicationAdmission.cpp:80 and PrepareToPlay.cpp:145 / Timer.cpp:471 read it.)

### The causal chain, step by step

```
gen4 publish (CoordExit gen4, ConvoPeq.log:1298)
  ↓  spec.fadingRuntimeUuid = 3  →  RuntimeWorld.topology.fadingRuntimeUuid = 3  (RuntimeBuilder:220)
  ↓  hasFadingRuntimeInWorld → true   @ AudioEngine.h:3307
crossfade start (BuilderExit gen5 L1301 fadingRuntimeUuid=0 but gen5 is the NEXT build; committed world still =3 from gen4; [XFADE] start @ Timer.cpp ~920)
  ↓  hasFading = true
audio callback starvation (mem 897MB→1043MB→1604MB; 4× rebuildAllIRsSynchronous)
  ↓  crossfade never ramps out on RT path → EVENT_CROSSFADE_TIMEOUT @ RuntimeHealthMonitor
crossfade timeout → [HEALTH] Crossfade timeout detected @ Timer.cpp:1658
  ↓ 1. terminalizeFadingDSP()  (Timer.cpp:1667) — CAS fadingRuntimeDSPSlot→null, retire DSP
  ↓ 2. crossfadeAuthorityRuntime_.unregisterCrossfade(record.id)  @ Timer.cpp:1675
  ↓ 3. crossfadeRuntime_.complete()  @ Timer.cpp:1685   (timer-side CrossfadeRuntime: pending_=false)
  ↓ 4. refreshCrossclosePreparedSnapshotFromAtomics()  @ Timer.cpp:1689
  ↓ 5. publishIdleWorldOnly(currentAfterFade, HardReset)  @ Timer.cpp:1690
        └─ RuntimeBuilder.buildRuntimePublishWorld(currentAfterFade, nullptr, HardReset, 0.0, false)
        └─ topology.fadingRuntimeUuid = (active && next!=nullptr) ? next->runtimeUuid : 0
        └─ active=false, next=nullptr  ⇒  **fadingRuntimeUuid = 0**  @ RuntimeBuilder.cpp:220
        └─ commitRuntimePublication(...)  @ AudioEngine.Transition.cpp:25  (the idle-publish commit)
  ↓  [HEALTH] Crossfade timeout recovery completed @ Timer.cpp:1694
```

### The contradiction, resolved (D134-1 answer)

> *"Why does `hasFadingRuntimeInWorld()` stay 1 even after `recovery completed`?"*

**It does NOT stay 1 because of the recovery logic. It stays 1 because of *timing + re-drive overwrite*.**

Measured timeline (ConvoPeq.log):
- L2228 — gen5 first `DeferredFadingActive` (hasFading=1), begins re-drive loop.
- L24433–L24490 — **gen8 re-drive loop: 28 iterations**, each
  `evaluate→DeferredFadingActive → enqueueDeferred(overwrite, hasPrevDeferred=0) → processDeferredAdmission→Ready → submitPublishRequest→enqueueDeferred`.
- **L24490** — gen8 last DeferredFadingActive re-loop.
- **L24497** — `BuilderExit gen=7 ... fadingRuntimeUuid=0 transitionActive=0`  ← this is the **idle-publish world build** from `publishIdleWorldOnly` (recovery step 5).
- **L24504** — `[HEALTH] Crossfade timeout detected, initiating recovery`
- **L24505** — `[HEALTH] Crossfade timeout recovery completed`

**The idle world (fadingRuntimeUuid=0) was *built* at L24497 but the *commit* (`commitRuntimePublication`) that makes it the live `RuntimeWorld` happens inside the recovery block at L24504–24505.** The gen8 busy-loop (L24433–24490) had ALREADY exhausted its 28 iterations and been pre-empted by the `[MEM_SNAP] PUBLISH gen=6` health tick (L24497... actually the MEM_SNAP at L24497-region interrupts the rebuild thread) **before** the recovery commit landed. After L24505 there are **0 `evaluate` calls** (the run proceeds to SHUTDOWN_BEGIN at L27368). So:

- The recovery's idle-publish *does* correctly set `fadingRuntimeUuid=0`, but
- it lands **after** the deferred re-drive of gens 5–8 already failed, and
- **no deferred request is re-evaluated against the zero-fading world**, so `hasFading` is observed as 1 (stale world) for the entire re-drive, and as 0 only in a world nobody re-checks.

**Line-level proof of the "recovery doesn't clear hasFading" illusion:**
1. `hasFadingRuntimeInWorld` reads `RuntimeWorld.topology.fadingRuntimeUuid` (AudioEngine.h:3304-3307).
2. `fadingRuntimeUuid` is written **only** at RuntimeBuilder.cpp:220 (`commitRuntimePublication` publishes the built world).
3. The recovery's `publishIdleWorldOnly(...,HardReset)` builds `fadingRuntimeUuid=0` (idle) → RuntimeBuilder.cpp:220 `active=false, next=nullptr` ⇒ 0. ✓ builds correct world.
4. `commitRuntimePublication` (AudioEngine.Transition.cpp:10-33 inside publishIdleWorldOnly) is the CAS that swaps the live world. The commit is **async** (PublicationExecutor.cpp:48) and only observable after it lands.
5. The gen8 re-drive re-reads `hasFading` via `evaluate` → PublicationAdmission.cpp:76-80 reading the **world as of that evaluate time** — which is still the gen4/gen6 world (fadingRuntimeUuid=3), committed before recovery. The re-drive never waits for / re-reads after the recovery commit.

> **D134-1 conclusion:** `timeout recovery completed` **does** clear `RuntimeWorld.fadingRuntimeUuid` (via the idle-publish build at RuntimeBuilder.cpp:220 with active=false,next=nullptr). The reason `hasFading=1` *appeared* to persist is that the recovery commit lands **too late** — after the deferred re-drive of gens 5–8 already starved on the pre-recovery world, and no re-drive re-checks post-recovery. The recovery logic is **correct but unordered** relative to the deferred re-drive.

---

## D134-2 — CrossfadeRuntime vs RuntimeWorld state matrix

### A. Authoritative state
`RuntimeWorld.topology.fadingRuntimeUuid` (AudioEngine.h:3304). RT-side atoms (`fadingRuntimeDSPSlot`, `crossfadeRuntime_`, `crossfadeAuthorityRuntime_`) are **non-authoritative for admission** — they are control/lifecycle state for the RT path.

### B. Completion state (what "crossfade done" means)
| Atom | "done" writes | "done" clears |
|---|---|---|
| `crossfadeRuntime_` | `start()` (Timer.cpp:38) | `complete()` (Timer.cpp:1685, :960) sets `pending_=false` |
| `crossfadeAuthorityRuntime_` | `registerCrossfade` | `unregisterCrossfade` (Timer.cpp:1675) |
| `fadingRuntimeDSPSlot` | `exchangeFadingRuntimeDSPSlot` / CAS install (AudioEngine.h:2211, DSPTransition.h:165) | CAS null (Timer.cpp:961-966 timeout; :1959-1962 terminalize) |
| `RuntimeWorld.fadingRuntimeUuid` | RuntimeBuilder.cpp:220 `(active&&next)?next->uuid:0` | same line, `active=false||next==null ⇒ 0` |

### C. Publication state (how publishIdleWorldOnly produces fadingRuntimeUuid=0)
`publishIdleWorldOnly(currentAfterFade, idlePolicy)` (AudioEngine.Transition.cpp:10):
- `publishIdleWorldOnly(..., HardReset)` ⇒ `buildRuntimePublishWorld(currentAfterFade, nullptr, HardReset, 0.0, false)`.
- RuntimeBuilder.cpp:220: `worldOwner->topology.fadingRuntimeUuid = (active && next != nullptr) ? next->runtimeUuid : 0`.
- Callers pass `active=false` (spec.execution.transitionActive) and `next=nullptr` ⇒ **`fadingRuntimeUuid = 0`** always for the idle publish, regardless of HardReset/SmoothOnly. (SmoothOnly only controls ramp, not the uuid bit.) The `HardReset` vs `SmoothOnly` distinction affects the gain ramp (`makeEngineRuntimeState`), NOT `fadingRuntimeUuid`.

### D. Timeout state (what timeout recovery clears)
`EVENT_CROSSFADE_TIMEOUT` handler (Timer.cpp:1657-1694) clears:
- `fadingRuntimeDSPSlot` → null (`terminalizeFadingDSP`, Timer.cpp:1667) — RT-side DSP handle.
- `crossfadeAuthorityRuntime_` registry entries (`unregisterCrossfade`, L1674-1677) — authority registry.
- `crossfadeRuntime_` pending flag (`complete()`, L1685) — timer-side fade state.
- **Then** `publishIdleWorldOnly` (L1690) → builds & commits a **new RuntimeWorld with `fadingRuntimeUuid=0`**. ✓

### Integrity condition (the design gap)
The four atoms are **independent**. There is **no invariant** that they are cleared as a single atomic batch. The completion contract is *de facto*:
> "timeout recovery is complete" (L1694 log) is emitted AFTER `publishIdleWorldOnly` returns, but `publishIdleWorldOnly` only *begins* `commitRuntimePublication` (async, PublicationExecutor.cpp:48) — it does **not** wait for the world to be observed live. So `recovery completed` fires **before** the zero-fading world is actually observable by the RT/admission path. This is the root asymmetry.

---

## D134-3 — Deferred slot state machine (complete)

### Reality vs D132 assumption
- **D132 assumption:** "deferred slot is single-slot, reset at `finishView()`."
- **D133 reality:** confirmed single-slot `deferredSlot_` (RuntimePublicationOrchestrator.h:255, `optional<DeferredPublishSlot>`) with `hasDeferred_` flag (L256). `enqueueDeferred` (L437) **overwrites in place** when `hasDeferred_=true` → `deferredOverwriteCount_++` (L440). This is `INV-DEFERRED-2` (latest-only). `finishView()` (L562-587) is the sole ownership-release: `deferredSlot_.reset()` + `hasDeferred_=false` + telemetry.

### Full state machine

```
[Live: hasDeferred_ = false, deferredSlot_ empty]

  rebuild request arrives & admission returns DeferredFadingActive
                    │
                    ▼
  enqueueDeferred(req)   ──►  [Deferred: hasDeferred_=true, slot holds req]
       (overwrite: deferredOverwriteCount_++, retire old DSP per INV-DEFERRED-2)
                    │
                    ▼
  processDeferredAdmission()   (RebuildThread only; jassert L631)
       if hasDeferred_==false: return                          (no-op)
       view = peekDeferred()   (borrow; hasDeferred_ NOT flipped)
       result = evaluateDeferred(metadata, snapshot)
                    │
      ┌─────────────┴─────────────┐
      ▼                           ▼
  Discard                       Ready
  (view->discard                 (view->consume
   +finishView)                  +finishView)  ► req moved out
      │                           │
      ▼                           ▼
  [Live: slot reset,         submitPublishRequest(req)
   hasDeferred_=false]            │
                               │ trySubmitImpl → evaluate(req)
                  ┌────────────┴──────────┬─────────────────────┬───────────┐
                  ▼                       ▼                     ▼           ▼
            RejectedStaleGen   DeferredFadingActive      RejectedNotFinal   Accepted→Published
                  │ (onReject)        │ (enqueueDeferred)      │ (markTransientFail)   │
                  ▼                   ▼                        ▼                     ▼
          count=1, telemetry        [Deferred]            markTransientFailure    [Live]
          resolveRecovery           (overwrite req        ΔL=0                    publish done
          -1, STOP)                 back, loop)              ▲
                                                  recoveryObligationId!=0 →
                                                  markTransientFailure(req.recoveryObligationId)
                                                  ──► ΔL++ / ΔL-- / ΔL=0 → ResolvedFailed / Retry / Recover

  Shutdown ──► clearDeferredForShutdown() ──► [Live] (forced discard)
```

### The retry-budget problem (D134-3 answer)

`submitPublishRequest` (RuntimePublicationOrchestrator.cpp:357-435): the ONLY branch that loops is
```cpp
case Decision::DeferredFadingActive:
    enqueueDeferred(req);   // ← OVERWRITES the slot with the SAME request, re-enters [Deferred]
    return;
```
There is **NO retry counter, NO budget, NO epoch check** on this path. The loop continues until:
- `evaluateDeferred` returns `Discard` (Shutdown / TTL 30s / `m.generation != ctx.currentGeneration` / sequence stale), OR
- the rebuild thread is pre-empted / the run shuts down (measured: health-tick pre-emption at L24497 MEM_SNAP, then SHUTDOWN_BEGIN L27368).

**Why the measured loops were bounded (19 for gens 5-7; 28 for gen8):**
- Not a retry cap. The `D129_TASK_WAKE ... wokeByRetryReady=1` fires each iteration (a retry-ready signal), but `hasFading=1` is re-read each time → re-defer.
- Termination = external pre-emption: the `[MEM_SNAP]` health sampler tick (publishes gen6's `[PUBLISH] seq=6`) at the memory peak (1604MB) interrupts the rebuild thread, and the run proceeds to shutdown. `evaluateDeferred`'s `RejectedStaleGeneration` did NOT fire for gen8 (deferredGen=8==currentGen=8 held throughout) — gen8's loop was killed by pre-emption, not by the staleness gate. (The 2 `RejectedStaleGeneration` in the gen5 ledger were from *gens 5/6* during their late re-drive, when `currentGen` had just advanced to 6.)

### Bounded re-drive policy (D134-3 required decision)

> *Deferred request re-evaluated 'Ready' then re-hit by DeferredFadingActive in the same coordinator cycle — how many re-evaluations allowed?*

**Design decision (for D135):** a bounded re-drive is mandatory. Candidate bound:
- **Retry budget N** in the `DeferredPublishSlot.guard` / `DeferredGuard` (src/audioengine/RuntimePublicationOrchestrator.h:30). The `DeferredGuard` already exists (L30) but is only a stale-discard guard — repurpose/augment it as a **per-slot re-drive counter**.
- Recommended **N = 2** (one deferral is acceptable; a second immediate re-deferral indicates the recovery world did not land → force `Discard` with `DiscardReason::StaleDiscard` + emit `[HEALTH] Deferred publish starved`**).
- Hard ceiling: even with a cap, **terminate on generation advance** (`m.generation != ctx.currentGeneration`) which already works — the problem is only the no-cap loop *within* a generation.

### Busy-loop trigger (D134-3 answer)
Triggered iff: `DeferredFadingActive` returned by `evaluate` **AND** `submitPublishRequest` re-enqueues the **same generation** with `hasFading` still observed true — i.e., the crossfade-timeout recovery's idle-world commit has not yet been observed by this generation's re-drive. The 28-gen loop proves there is no counter-guard; only external pre-emption stops it.

---

## D134-4 — Generation taxonomy (deferred freshness = safety valve, NOT cause)

D132-RA's "generation race is primary" is **refuted** (D133-1: `currentGen==req.generation` on every primary queue→enqueue→evaluate; 0 RejectedStaleGeneration on primary path). Generation is, correctly, only a **freshessafety valve** for deferred re-drive. These 5 are distinct:

| Symbol | What it counts | Defined at | Written by | Read by (admission/deferred) | Meaning for deferred |
|---|---|---|---|---|---|
| `req.generation` | the rebuild-request generation the deferred request was captured under | PublishRequest | RebuildDispatch.cpp:655 `++rebuildRequestGeneration` | evaluate (PubAdmission.cpp:15), evaluateDeferred (126) | "stale if != current" |
| `currentGeneration` (= `reloadRequestGeneration`) | in-flight rebuild-request generation | AudioEngine.h:1666 `currentBuildGeneration()` | CtorDtor.cpp:143, ReleaseResources.cpp:146, RebuildDispatch.cpp:655 | evaluate (15), buildDeferredAdmissionSnapshot (554) | the moving target for staleness check |
| `lastCommittedRebuildGeneration` | last generation that reached **commit** (publish) | AudioEngine.h:2554 | **NONE (0 writers — see D134-LCG)** | Parameters.cpp:374/445, RebuildDispatch.cpp:178, Timer.cpp:843 | `outstandingRebuild = queued > committed` ⇒ blocks `finalizeReady` |
| `world.generation` / `runtimeVersion` | the publication sequence of a *committed RuntimeWorld* | RuntimeBuilder.cpp:125-128, :121 | RuntimeBuilder (per build) | hasFading read-context (Topology) | epoch of the world `hasFading` is read from |
| `publicationSequence` / `metadata.sequence` | monotonic publish ordinal (PublishRequest/PublicationSequenceId) | RuntimeBuilder.cpp:118-119 (`reserveRuntimePublicationIdentity`) | RuntimeBuilder | evaluateDeferred L140 (`m.sequence < ctx.lastSequence`) | "history rollback" guard (ISR-WORLD-001) |

### Log-label disambiguation (required)
`gen=` in all D133 `[D133]` logs means **req.generation / currentGen == rebuildRequestGeneration** (the rebuild-request epoch), NOT world.generation. ConvoPeq.log already separates this:
- `ConvoPeq.log:1301 [CoordExit gen=4]` = req.gen (PublicationIdentity::generation).
- `ConvoPeq.log:1304 [BuilderExit gen=6 transitionActive=1 ...]` = BuilderEntry/gen6 world build (a *different* worldId/seq).
These coexist in the same trace without confusion because CoordExit/BuilderExit are distinct markers. The **fix (D134 design decision)**: the deferred re-drive path must log, at re-deferral, BOTH `req.generation` and the **committed world's `topology.fadingRuntimeUuid` value it read** (so we can prove "I deferred because world-fading=3, not world-fading=0"). ConvoPeq.md's existing `[XFADE] start`/`BuilderExit ... fadingRuntimeUuid=` already expose the world value; admission just must echo it.

### RejectedStaleGeneration is secondary (D134-4 answer)
In D133-1 the 2 `RejectedStaleGeneration` occurred in deferred **re-drive** (`evaluateDeferred` L126: `m.generation(5) != ctx.currentGeneration(6)` and `6!=7`), after a *secondary* rebuild advanced `reloadRequestGeneration` during the re-drive. On the primary queue→enqueue→evaluate path it is **never** returned (currentGen==req.generation held). So it is a *correct* freshness discard of an obsolete deferred payload, not the cause of the shortfall. It is a **symptom of the re-drive outrunning the generation**, which itself is D134-3's no-cap problem. Keep it; do not weaken it.

---

## D134-5 — Last committed rebuild generation: independent defect (D134-LCG)

`lastCommittedRebuildGeneration` (AudioEngine.h:2554) — **writers = 0** (D133-1 confirmed; `grep -rn lastCommittedRebuildGeneration` shows only the decl `=0` and 5 readers).

1. **What it means:** "the generation of the last `commitRuntimePublication` that completed (world observable)."
2. **Who should own the writer:** the **commit path** — `commitRuntimePublication` (PublicationExecutor.cpp:53 / AudioEngine.Transition.cpp) is the single publication-commit authority. It must `publishAtomic(lastCommittedRebuildGeneration, committedWorld.generation)` after the CAS-swap succeeds.
3. **When to write:** at `commitRuntimePublication` post-CAS-success (one place only — INV-ADMISSION-1 authority singularization).
4. **`finalizeReady`** (Timer.cpp:852-856): `(!irLoaded || irFinalized) && ... && !outstandingRebuild`. `outstandingRebuild = (queuedGeneration > committedGeneration)` (Timer.cpp:844, Parameters.cpp:375/446). With `committedGeneration` **stuck at 0**, `outstandingRebuild` is **always true** once any rebuild is queued → `finalizeReady` **never becomes true** → the timer-side auto-finalize rebuild (`Timer.cpp:858 if (finalizeReady || timedOut)`) can only fire on `timedOut`, never on clean finalize.
5. **Impact on this bug:** secondary. The finalize-aware auto-recovery rebuild (`Timer.cpp:858-...`) is suppressed, so the system cannot self-heal a stuck crossfade via a clean finalize-driven rebuild; it only recovers via the crossfade-timeout path. Fixing LCG is **NOT** the D134 fix; it removes a recovery-suppression defect and prevents *future* starvation from the same root. Track as a separate defect ticket.

This is carved out from D134 so the DeferredFadingActive fix (D135) is scoped purely to the fading/deferred/re-drive ordering.

---

## D134-6 — Candidate comparison (F1′ / F2′ / F3′ / F4)

Evaluated against: **ownership RT-safety** (no RT-path blocking / single writer), **capacity** (no extra unbounded memory), **semantics** (does not violate the authority-singularization invariants in design-D4 §134/§1488).

| Candidate | Mechanism | Ownership safety | Capacity | Semantics risk | Resolves D133-1 shortfall? |
|---|---|---|---|---|---|
| **F1′** — make timeout-recovery's idle-publish commit **observable** before re-drive | After `publishIdleWorldOnly` in recovery, have the rebuild thread re-check `hasFadingRuntimeInWorld` and, if a deferred req exists, force `processDeferredAdmission` **only after** the commit lands (sequence-observe on the committed world seq). | ✓ RT-safe (commit is Non-RT; re-check on rebuild thread) | O(1) | Must not re-enter inside `submitPublishRequest` (no re-entrancy). Requires a "world seq observed ≥ recovery seq" guard. | **YES** — directly closes P0. |
| **F2′** — bounded deferred buffer (multi-slot / freshness window) | Replace single `deferredSlot_` with an ordered ring (O_max≈4) or a TTL+fading-cleared gate; drop oldest. | ✓ single owner (rebuild thread) | O(4) extra requests (acceptable) | Changes INV-DEFERRED-2 (latest-only) semantics — may violate "one obligation per live DSP" if recovery path expects single-slot. High review. | **PARTIAL** — prevents loss but does NOT stop the busy-loop unless combined with F3′/retry-cap. |
| **F3′** — treat `DeferredFadingActive` as **retryable-with-budget** state | Add `retryCount` to `DeferredGuard`; cap re-drive at N=2; on 2nd immediate re-deferral → `Discard(StaleDiscard)` + `[HEALTH] Deferred publish starved`. | ✓ single owner | O(1) | Adds a discard path; must guarantee a *later* valid request can still publish (generation still unique). Low risk. | **YES** — stops the 28-iteration loop (P1). Does **not** by itself make the deferred req publish (needs F1′). |
| **F4** — suppress redundant `rebuildAllIRsSynchronous` on identical fingerprint | Skip IR rebuild when fingerprint matches (CtorDtor.cpp `/RebuildDispatch`). | ✓ | frees ~7MB×4 bursts | **Out of scope for the shortfall**: memory pressure is an amplifier of RT-starvation, not the admission mechanism. D133-1 shows gen5-8 vanish at `hasFading=1` independent of OOM. | **NO** — separate causal layer. P4 only. |

### Recommended D135 minimum patch (D134-7)
**Combine F1′ + F3′ (not F2′, not F4-as-primary).** Rationale: keeps single-slot ownership (INV-DEFERRED-2 intact), stops the busy-loop (F3′ cap), and lets the deferred req actually observe the cleared world (F1′ ordering). F4 is a separate performance ticket (memory pressure), tracked under D134-LCG's sibling "IR duplicate rebuild" item.

---

## D134-7 — Minimal patch specification (D135 handoff, file/function/condition)

| # | File | Function | Change | Priority |
|---|---|---|---|---|
| P0 | `src/audioengine/AudioEngine.Timer.cpp` | `EVENT_CROSSFADE_TIMEOUT` handler (L1657-1694) | After `publishIdleWorldOnly` returns, record the **expected committed worldSequence**; before returning, ensure a pending deferred req (if any) is **scheduled for re-drive only after** that sequence is observed live (hook into PublicationExecutor completion or a `worldSeqObserved ≥ recoverySeq` check in `processDeferredAdmission`). | P0 |
| P1 | `src/audioengine/RuntimePublicationOrchestrator.h` | `DeferredGuard` (L30) + `DeferredPublishSlot` | Add `uint8_t retryCount{0}` (or add field to slot). | P1 |
| P1 | `src/audioengine/RuntimePublicationOrchestrator.cpp` | `submitPublishRequest` `DeferredFadingActive` branch (L377-380) | On re-enqueue: if slot's `req.generation == deferredSlot_.metadata.generation` (same gen = same cycle), increment `retryCount`; if `retryCount ≥ 2` → instead `Discard(StaleDiscard)` + `diagLog("[HEALTH] Deferred publish starved gen=N retries=2")`; cap hard at N=2. | P1 |
| P1 | `src/audioengine/RuntimePublicationOrchestrator.cpp` | `peekDeferred`/`processDeferredAdmission` | Guard the re-drive: only re-enqueue if `hasFadingRuntimeInWorld(readHandle)` transitioned to false **OR** retry budget > 0; else `Discard`. (Co-locates the world-observe with the retry count.) | P1 |
| P1 | `src/audioengine/AudioEngine.h:3304` | `hasFadingRuntimeInWorld` | **No change** — confirm it remains the single authority (the fix makes the re-drive *wait for* its verdict, not change its definition). | — |
| P2 | `src/audioengine/AudioEngine.RebuildDispatch.cpp:655` / `:177-178` | rebuild generation | **No change** — `currentGeneration`/`lastCommittedRebuildGeneration` semantics unchanged; generation remains a freshness valve only. | P2 |
| P3 | (separate) `src/audioengine/AudioEngine.h:2554` + commit path (PublicationExecutor.cpp:53) | `lastCommittedRebuildGeneration` | Add writer → `publishAtomic(lastCommittedRebuildGeneration, identity.generation)` post-CAS. **Deferred to D134-LCG / defect ticket**, NOT in D135. | P3 |

### Log-label design (for post-fix verification)
At every deferred re-deferral, emit the world state read:
```
[D135] re-defer gen=N currentGen=N hasFading=1 worldFadingUuid=3 worldSeq=M retryCount=k
```
So D135's success is auditable: retryCount stops ≤2, and the publish attempt after recovery sees `hasFading=0 / worldFadingUuid=0`.

---

## D134 exit-condition checklist (all 10)

| # | Condition | Status | Evidence |
|---|---|---|---|
| 1 | `hasFading=true` unique data flow, line-level | DONE | AudioEngine.h:3304-3307 (`RuntimeWorld.fadingRuntimeUuid!=0` only); RuntimeBuilder.cpp:220 only writer |
| 2 | timeout recovery clears which state | DONE | Timer.cpp:1667-1690 table in §D134-2-D |
| 3 | path where hasFading stays true post-recovery | DONE | §D134-1: recovery commit lands after re-drive; no re-check post-commit |
| 4 | RuntimeWorld.fadingRuntimeUuid authority | DONE | §D134-2-A: single reader = hasFadingRuntimeInWorld; single writer = RuntimeBuilder:220 |
| 5 | Deferred slot state machine complete | DONE | §D134-3 full diagram + processDeferredAdmission L633-693 |
| 6 | busy-loop trigger condition | DONE | §D134-3: re-enqueue same-gen with hasFading still true; no retry cap (measured 28-iters gen8) |
| 7 | RejectedStaleGeneration = secondary | DONE | §D134-4: only in deferred re-drive (evaluateDeferred L126); 0 on primary path |
| 8 | lastCommittedRebuildGeneration independent defect | DONE | §D134-LCG: 0 writers; suppresses finalizeReady (Timer.cpp:852-856) |
| 9 | F1′/F2′/F3′/F4 comparison | DONE | §D134-6 matrix (ownership/capacity/semantics) |
| 10 | D135 minimal patch unit (file/function/condition) | DONE | §D134-7 table (F1′+F3′; LCG deferred) |
