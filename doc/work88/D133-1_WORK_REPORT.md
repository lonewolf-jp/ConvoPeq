# D133 Work Report — Admission Re-audit (D132-RA correction)

## Summary
D132-RA classified the 6-burst→1-publish shortfall as a **generation race (D)**.
D133 re-audited via diag-only instrumentation (+ read-only source audit). **D132-RA is incorrect.**

## Correct root cause
**DeferredFadingActive (A-variant) + single-slot deferred buffer overwrite (INV-DEFERRED-2) + crossfade-starve latch.**

Sequence (measured, `evidence/D133-1_6burst_diag.log`):
1. First IR-loaded publish (gen4 / worldId=6) succeeds: `CoordExit gen4 spec.fadingRuntimeUuid=3` + `[XFADE] start` → initiates a runtime crossfade.
2. Every IR-loaded rebuild thereafter (gen5–8) snapshots at queue time into a world where `hasFading == true` (the gen4 crossfade is still active).
3. `PublicationAdmission::evaluate` L68–76: `if (hasFading) return Decision::DeferredFadingActive;` → gen5–8 all return DeferredFadingActive → routed to the single-slot `deferredSlot_` (Orchestrator), which is **latest-only overwrite** → gen5's deferred is starved by gen6/7/8 overwrites.
4. `hasFading` stays `1` for the entire remainder of the run (4214 evaluate calls) because the audio callback is starved (memory 303MB→1041MB; IR reloaded 4× by `rebuildAllIRsSynchronous`) → `[HEALTH] Crossfade timeout detected → recovery completed`, but the recovery does not clear the admission-side `hasFading` latch for the deferred republish, nor does it unblock the single deferred slot.
5. The deferred re-drive busy-loops (processDeferredAdmission reports `Ready` 19×, each re-deferred because `hasFading` still 1); as `rebuildRequestGeneration` advances during the re-drive, the held deferred (e.g. gen5) becomes `RejectedStaleGeneration` (deferredGen=5, currentGen=6) at evaluateDeferred → discarded.

## Measured decision ledger (all evaluate calls)
```
      3  Accepted            (gens 1, 3, 4)
   4214  DeferredFadingActive (gens 5–8 + re-drive loop)
      2  RejectedStaleGeneration (deferred re-drive only: gen5@cur=6, gen6@cur=7)
      0  RejectedNotFinalized
      0  RejectedStaleGeneration on primary queue→enqueue path
      0  RejectedPressure
      0  RejectedShutdown
```

## What this falsifies
- **E (RejectedNotFinalized)**: refuted — all IR-loaded gens queue/enqueue/evaluate snapshot show `irFinalized=1` (invariant). RejectedNotFinalized = 0.
- **D (generation race)**: refuted — `currentGen == req.generation` on every primary queue→enqueue→evaluate. The 2 RejectedStaleGeneration are deferred **re-drive** staleness, not queue→admission races. `isRebuildObsolete` in the rebuild thread = 0 in the IR-loaded regime (1 only during early SR-ramp 48000→192000).
- **C (stale overtrigger) / (A) intentional supersede**: no.

## Secondary defect (confirmed, NOT fixed)
`lastCommittedRebuildGeneration` (`AudioEngine.h:2554`) has **0 writers** (decl only; 5 readers). This keeps `outstandingRebuild` permanently true, which suppresses the timer-side finalize-aware auto-recovery rebuild (`finalizeReady = ... && !outstandingRebuild` → never true). This is secondary to the present mechanism but compounds the starvation — fixed only if crossfade-recovery or finalize-aware rebuild is to self-heal. Left untouched per D133 read-only constraint.

## Artifacts
- Instrumentation (diag-only, `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS`): `AudioEngine.RebuildDispatch.cpp:664`, `AudioEngine.Commit.cpp:808`, `PublicationAdmission.cpp evaluate`, `RuntimePublicationOrchestrator.cpp:445 (enqueueDeferred)`, `639 (processDeferredAdmission)`.
- Build: `build-diag/` (Release + `-DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS`), rebuilt 2026-08-29 19:03.
- Evidence: `evidence/D133-1_6burst_diag.log` (full 27,409-line Logger output), `evidence/D133-1_EVIDENCE.md`.
- Git diff verified: 4 files changed, all additions diag-gated; zero production-branch logic altered.

## Reproduction note (transparency)
This run yielded 3 publishes (gens 1, 3, 4) rather than D132-7's 1, because IR pre-loads one burst late here (gen1–3 ran at SR=48000 with a different fingerprint → not merged → each published). From gen5 onward (IR-loaded, SR=192000, identical fingerprint) the D132-7 invariant holds exactly: **1 publish (gen4/gen5), gens 5–8 vanished via DeferredFadingActive.** To reproduce D132-7's exact 6→1 count, add `--cli-sample-rate-hz 192000` so burst #1 is already SR-matched and merges with #2/#3 before IR load. Not needed for mechanism closure.

## Next
D133-2 causal re-audit → refine fix candidates (F1' gate crossfade-timeout recovery to clear hasFading; F2' multi-slot or TTL-bounded deferred buffer; F3' skip rebuildAllIRsSynchronous when fingerprint identical to cut memory growth). D134 design gate held until candidates reconciled against deferred-ownership / generation semantics.
