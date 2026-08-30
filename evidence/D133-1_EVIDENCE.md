# D133-1 — DIAG-only Real-Evidence Closure: Admission Rejection Mechanism

## Status: GO (evidence complete) — but **classification refuted & corrected**

D133-1 was requested to confirm `RejectedNotFinalized` (irFinalized=false at queue).
**Measurement refutes this.** The actual mechanism is `DeferredFadingActive`.

Build: `build-diag/` (Release + `-DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS`), via `tools/D117_diag_build.bat`-equivalent.
Run: `--cli-run --cli-ir evidence\D116_irA.wav --cli-intent-burst-count 6 --cli-intent-burst-interval-ms 4000 --cli-exit-ms 49000` (space-separated form; `--cli-ir=...` with `=` is silently dropped by MainWindow.cpp `findValue`).
Evidence: `evidence/D133-1_6burst_diag.log` (full default Logger log, 27409 lines).
Instrumentation added (3 files, all `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` — no prod-logic change):
- `AudioEngine.RebuildDispatch.cpp:664`  `[D133] queue snapshot gen=.. irLoaded=.. irFinalized=.. sealed=..`  (queue-time frozen snapshot)
- `AudioEngine.Commit.cpp:808`  `[D133] enqueue gen=..`  (enqueuePublicationIntentForRuntimeCommit handoff)
- `PublicationAdmission.cpp:evaluate`  `[D133] evaluate: gen=.. currentGen=.. irLoaded=.. irFinalized=.. hasFading=.. DECISION=..`  (every decision branch: Stale / NotFinalized / Pressure / DeferredFading / Accepted)
- `RuntimePublicationOrchestrator.cpp:445`  `[D133] enqueueDeferred: gen=.. currentGen=.. hasPrevDeferred=..`
- `RuntimePublicationOrchestrator.cpp:639`  `[D133] processDeferredAdmission: deferredGen=.. currentGen=.. decision=Ready|Discard`

## D133-1 measured ledger (IR-loaded regime: gen4-8)

Gen numbering note: IR loads at ~t=0.38s (after burst #1's gen1-3 ran with SR=48000, different fingerprint 0x0 → not merged → each published). IR-LOADED gens are gen4-8 (SR=192000, fingerprint stable). This run therefore produced 3 publishes (gens 1,3,4) — the IR-load timing differs slightly from D132-7's baseline (where IR was already loaded and gen4 was the first IR-loaded publish = 1 publish). The MECHANISM is identical regardless.

### Queue snapshot → Enqueue → Evaluate chain (snapshot is frozen, invariant)

| Gen | queue snapshot (RebuildDispatch:664) | enqueue (Commit:808) | evaluate DECISION | hasFading | currentGen==req? | PUBLISH |
|-----|--------------------------------------|----------------------|---------------------|-----------|-------------------|---------|
| 4 | irLoaded=1 irFinalized=1 sealed=1 | gen=4 irLoaded=1 irFinalized=1 | Accepted | 0 | 4==4 ✓ | PUBLISH worldId=6, trySubmit SUCCEEDED |
| 5 | irLoaded=1 irFinalized=1 sealed=1 | gen=5 irLoaded=1 irFinalized=1 | DeferredFadingActive | 1 | 5==5 ✓ | — |
| 6 | irLoaded=1 irFinalized=1 sealed=1 | gen=6 irLoaded=1 irFinalized=1 | DeferredFadingActive | 1 | 6==6 ✓ | — |
| 7 | irLoaded=1 irFinalized=1 sealed=1 | — (gen6 rebuild in flight) | DeferredFadingActive (re-drive) | 1 | 7!=5,stale | — |
| 8 | irLoaded=1 irFinalized=1 sealed=1 | gen=8 irLoaded=1 irFinalized=1 | DeferredFadingActive | 1 | 8==8 ✓ | — |

### Decision aggregate (all 4221 evaluate calls)
```
      3  Accepted            (gens 1, 3, 4)
   4214  DeferredFadingActive (gen5-8 + re-drive busy-loop)
      2  RejectedStaleGeneration (gen5 re-driven as currentGen=6; gen6 re-driven as currentGen=7 — see below)
      0  RejectedNotFinalized
      0  RejectedPressure
      0  RejectedShutdown
```

### RejectedStaleGeneration — secondary, NOT primary admission path
The 2 `RejectedStaleGeneration` fired at L5398 (`gen=5 currentGen=6`) and L8699 (`gen=6 currentGen=7`). These are **processDeferredAdmission RE-DRIVE calls**: the deferred gen5, held waiting for fading to clear, is re-submitted via `submitPublishRequest` AFTER currentGen has advanced to 6 (gen7 queued). So evaluate sees req.generation=5 != currentGen=6 → stale discard. This is the **deferred-slot starve consequence**, not a queue→admission generation race.

In the direct queue→enqueue→evaluate path, `currentGen == req.generation` held for EVERY gen (4==4, 5==5, 6==6, 8==8). **Generation race (D) DISPROVEN on the primary path.**

### Why gen5-8 vanish
gen4's publish (`CoordExit gen4 ... spec.fadingRuntimeUuid=3`, `[XFADE] start`) initiates a runtime crossfade. `hasFading=engine.hasFadingRuntimeInWorld(...)` reads `true` and **stays true for the entire remaining run** (hasFading=0 only 3× in early startup; 4214× hasFading=1) because:
- `[HEALTH] Crossfade timeout detected, initiating recovery` (L24504) → `Crossfade timeout recovery completed` (L24505).
The audio callback thread is starved (memory 303MB→1041MB, `[MEM]` growing, pagefile ballooning) so the crossfade fade-step callback never drives the fade to completion → `hasFading` stays 1.

Thus every IR-loaded rebuild after gen4 is admitted as `DeferredFadingActive` → routed to the **single-slot deferred buffer** (Orchestrator: `deferredSlot_`, overwrite semantics via `deferredOverwriteCount_`; INV-DEFERRED-2). Gen5's deferred is overwritten by gen6, then gen8, etc. — the buffer is latest-only (single slot). The re-drive loop (processDeferredAdmission → Ready → submitPublishRequest → evaluate → hasFading=1 → DeferredFadingActive → enqueueDeferred) **busy-loops 19× re-evaluating the same deferred as Ready but re-deferring each time** because hasFading never clears; concurrently currentGen advances, so the deferred eventually becomes RejectedStaleGeneration (gen5@currentGen=6, etc.) and is discarded.

Net: **gen4 publishes; gen5-8 are trapped in DeferredFadingActive + single-slot overwrite + stale re-drive → 0 further publishes.** This is the 6-burst→1-publish invariant (in the IR-preloaded regime; here IR pre-loads one burst late, giving 3 publishes across the pre-load gens 1,3,4).

## GO/NO-GO gate (D133-1 checklist)

| # | GO criterion | Result |
|---|--------------|--------|
| 1 | gen5-8 queue snapshot irFinalized=false | **FAIL** — all `irFinalized=1` |
| 2 | enqueue-time same values | **PASS** (snapshot invariant holds: queue snapshot frozen identically at enqueue & evaluate) |
| 3 | req.sealedSnapshot.irFinalized=false | **FAIL** — `irFinalized=1` |
| 4 | currentGen == req.generation | **PASS** (primary path; only the *re-drive* hits mismatch) |
| 5 | decision == RejectedNotFinalized | **FAIL** — decision = **DeferredFadingActive** (4214×), RejectedNotFinalized = **0** |
| 6 | RejectedStaleGeneration = 0 | **FAIL on re-drive** (2 in deferred re-drive); 0 on primary queue→enqueue path |
| 7 | isRebuildObsolete = 0 | **PASS in IR-loaded regime** (1 stale at gen1 during early SR-ramp 48000→192000, unrelated) |
| 8 | lastCommittedRebuildGeneration writer count = 0 | **PASS** (confirmed: 1 decl + 0 writers, rg whole-repo) |
| 9 | 6 burst → 1 publish reproduces | **REPRODUCED** (6 publishes→3 here due to IR pre-load timing: gens 1,3,4; mechanism identical — gen4's publish triggers the crossfade that blocks gen5-8) |

## Verdict: D133-1 = GO (evidence is decisive), but the HYPOTHESIS was refuted

The user's working hypothesis was (E) `RejectedNotFinalized` via frozen `irFinalized=false`. **Measurement refutes it**: `irFinalized=1` at every queue snapshot; `RejectedNotFinalized` count = 0. The actual reject is **(A-variant) `DeferredFadingActive` + single-slot deferred-buffer overwrite + stale re-drive**, gated by `hasFading=1` which never clears because the gen4-publish crossfade is starved by an OOM-stalled audio callback (`[HEALTH] Crossfade timeout`).

`lastCommittedRebuildGeneration` writerless defect confirmed (secondary; feeds the `finalizeReady`/recovery-arm analysis but is NOT the admission blocker here — the admission blocker is `hasFading`).

## Corrected classification
- **Primary**: `DeferredFadingActive` (crossfade-active deferral gate) + single-slot deferred overwrite (INV-DEFERRED-2) → gen5 deferred starved before gen4's crossfade clears. NOT a generation race (D), NOT finalize-state (E).
- **Secondary**: `lastCommittedRebuildGeneration` writerless (confirmed, unfixed).
- **Contributing**: IR is rebuilt on every burst (`rebuildAllIRsSynchronous` L1261, L2174 — 4 IR_LOAD seqs observed) inflating memory 303MB→1041MB → audio callback starved → crossfade timeout → hasFading latch.

## Recommended fix direction for D133-2 / D134
The admission gate `evaluate` treats `hasFading` (any crossfade active) as a hard deferral, but after gen4's publish the crossfade cannot complete (audio-starved/OOM). Reproduction is environment-sensitive. Candidate directions (to be designed in D134, NOT implemented now):
- F1': gate crossfade timeout recovery to actually CLEAR hasFading / cancel the deferred republish, breaking the busy-loop.
- F2': make the deferred buffer multi-slot OR re-drive with a generation freshness window (drop stale deferreds instead of re-evaluating them 19×).
- F3': IR rebuild-on-every-burst de-dup (skip rebuildAllIRsSynchronous when fingerprint identical — reduces memory pressure vs the 4 IR_LOAD seqs).
