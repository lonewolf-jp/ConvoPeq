# D133 — Generation Semantics: Read-Only Forensic Analysis

## Executive summary
The 6-burst → 1 publish shortfall is **not** a generation "race" (D-classification in D132-RA). It is
**classification E: a valid publish discarded as stale-finalize-state** — specifically `RejectedNotFinalized`
from `PublicationAdmission::evaluate` (`irLoaded && !irFinalized`), combined with a secondary starvation
arm caused by a writerless `lastCommittedRebuildGeneration`.

## 1. Generation State Model (current code)

| Counter | Decl | Writers | Readers |
|---|---|---|---|
| `rebuildRequestGeneration` | AudioEngine.h:2553 | `++` at RebuildDispatch.cpp:655 (`++rebuildRequestGeneration`) | isRebuildObsolete L2594, Parameters L374/L445, Timer L842/843, evaluate L15-16 |
| `lastCommittedRebuildGeneration` | AudioEngine.h:2554 | **NONE (writerless; stays 0)** | Parameters L374/L445, RebuildDispatch L178, Timer L843 |
| task snapshot `generation` | RebuildDispatch L656 `task.generation=generation` (frozen = RRG at queue) | — | rebuildThreadLoop L842/1090-1094, enqueue L1302 |
| `req.generation` (PublicationRequest) | Commit.cpp:808 `req.generation=generation` (frozen = task.generation) | — | evaluate L15-18, worldId/PUBLISH emit |
| `currentGeneration` (evaluate) | evaluate L15-16 = `consumeAtomic(rebuildRequestGeneration)` | same as RRG | evaluate L17 |

Key asymmetry: **`currentGeneration` (evaluate) and `isRebuildObsolete` BOTH read `rebuildRequestGeneration`**
(Commit.cpp:808 freeze + PublicationAdmission.cpp:15-16 read are the SAME atomic). They are NOT two
different counters. D132-RA's "gen=4 → PUBLISH gen=6" conflation is a label mix: `gen=N` in `[PUBLISH]`
= `worldId` (world generation, `FrozenRuntimeWorld::generation`); `gen=N` in `[PUBLISH] seq/gen` and in
`trySubmit gen=N` = rebuild-request generation. Distinct counters. (see WorldGen below.)

## 2. 6-Burst Generation Ledger (evidence/D132-7_6burst_diag.log)

Trace: 6 CLI bursts every 4s; burst #1 absorbed by same-fingerprint merge (`sameAsPendingWouldMerge`,
identical structuralHash 0xd9d3e460d6606ccd, fingerprint 0x73d6a0fa1460cba5) → 5 tasks.

| Burst | RRG (queue) | task.generation | rebuildThreadLoop build | enqueue→submitPublishRequest | evaluate decision | PUBLISH(seq) |
|---|---|---|---|---|---|---|
| 1 | 4 | 4 | build 59.7ms+IR 270.2ms | yes | Accepted | [PUBLISH] seq=6 worldId=6 |
| 2 | 5 | 5 | build 69.2ms+IR 296.9ms | yes | RejectedNotFinalized | — |
| 3 | 6 | 6 | build 65.8ms+IR 279.2ms | yes | RejectedNotFinalized | — |
| 4 | 7 | 7 | build 71.8ms+IR 281.5ms | yes | RejectedNotFinalized | — |
| 5 | 8 | 8 | build 72.2ms+IR 284.6ms | yes | RejectedNotFinalized | — |
| 6 | (merged w/5) | — | — | — | — | — |

Log evidence (verbatim):
- `trySubmit SUCCEEDED gen=4` (x1) — only gen4.
- `[PUBLISH] seq=6 gen=6 worldId=6` — worldGen=6 (world generation, incremented per successful commit).
- `obsolete` log lines: **0** (rules out `isRebuildObsolete` ever firing → RRG was NOT advanced past task.gen during builds).
- `[XFADE] start`: **0** (rules out DeferredFadingActive path).
- `trySubmit ... FAILED`: **0** (rules out publish-time execution failure).
- Shutdown drain (L499): `pendingPub=0 pendingRetire=0 crossfade=0 ... deferred=0`.

## 3. Why gen4 published but gen5-8 did not

### 3a. Gen4 — Accepted
`enqueuePublicationIntentForRuntimeCommit` (Commit.cpp:782) freezes `req.generation=task.generation=4`,
then calls `runtimeOrchestrator_->submitPublishRequest(req)` (L815) → `trySubmitImpl` →
`admission_.evaluate(req,...)`:
- L15-18: `currentGen = consumeAtomic(rebuildRequestGeneration)=4`; `req.generation=4` → match, not stale.
- L21-22: snapshot's `irFinalized` was **true** at gen4's queue-time capture (engine IR finalized from
  prior load) → `irLoaded(1) && !irFinalized(true→!true=0)` = false → NOT rejected.
- L51-58: `hasFading=false` (no crossfade) → not deferred.
- `Accepted` → build+commit → `[PUBLISH] seq=6 worldId=6` → `trySubmit SUCCEEDED gen=4`. ✔

### 3b. Gen5-8 — RejectedNotFinalized (silent)
After gen4's publish, the convolver's IR load pipeline is entered (the publish→transition path triggers
an IR (re)load on the runtime convolver; and `rebuildAllIRsSynchronous` runs on the rebuild thread).
`ConvolverProcessor.LoadPipeline.cpp:47` resets `irFinalized=false` at the START of every IR load and
sets it `true` only on commit-at-L550/L803.

The queue-time snapshot captured at RebuildDispatch.cpp:657-663 reads:
`uiConvolverProcessor.isIRFinalized()` (L662). For gen5-8, this read is taken while the convolver is
mid-(re)load following gen4's transition → **`irFinalized=false`** is frozen into
`task.runtimeBuildSnapshot.irFinalized=false` (seal L146), propagated to `req.sealedSnapshot`
(Commit.cpp:808 L808), and reaches evaluate L21-22:

```
irLoaded = true  (CONV_STATUS irLoaded=1 for gen5-8)
irFinalized = false
=> (irLoaded && !irFinalized) = (1 && 1) = TRUE
=> return RejectedNotFinalized
```

`submitPublishRequest` L389-402 routes `RejectedNotFinalized` to: `stateOwner_.onRejected(0)`,
`telemetryRecorder_.recordFailure(...)` (telemetry sink, NOT the diag log — no line appears),
and (since `recoveryObligationId==0`) `markTransientFailure` is a guarded no-op. **→ gen5-8 are
silently discarded; no `[PUBLISH]`, no `trySubmit SUCCEEDED`, no diag line.** ✓ matches ledger.

### 3c. Secondary starvation arm (writerless `lastCommittedRebuildGeneration`)
Timer.cpp:820-900 (DeferredFinalizeAware branch) computes:
```
finalizeReady = (!irLoaded || irFinalized) && !irLoading && !structuralDeferred && !pendingIrChange && !outstandingRebuild;
```
`outstandingRebuild = queuedGeneration > committedGeneration`. With
`lastCommittedRebuildGeneration` **writerless (always 0)**, `outstandingRebuild` is true whenever any
generation has been queued — so `finalizeReady` is **never** true once steady state is reached.
The timer's auto-recovery `submitRebuildIntent` (L894) fires ONLY via the 2s `timedOut` branch
(L858), and even those timer-issued rebuilds re-snapshot `irFinalized=false` in the same load window
→ **rejected identically** → the publish cadence stays stuck at 1. This is the starvation mechanism:
no self-healing path can ever capture a finalized snapshot while the load pipeline is mid-flight.

## 4. Classification verdict
- **(A) intentional supersede** — No. Fingerprints are identical; nothing supercedes gen5-8 except the
  admission gate itself.
- **(B) valid publish discarded as stale-finalize-state** — YES. `RejectedNotFinalized` (irFinalized=false)
  discards gen5-8. This is the root cause.
- **(C) stale overtrigger** — No. No `obsolete`/stale logs; RRG never over-advanced during builds.
- **(D) semantic mixing of generation counters** — Partially relevant ONLY as a documentation defect
  (the word "gen=" denotes two different counters in different log sites); it is NOT the causal
  mechanism. D132-RA's "gen=4→PUBLISH gen=6 generation race" framing is incorrect.

**Conclusion: B (primary) + writerless-committed-generation starvation (secondary, E').**
D132-RA's "generation race" (D) is disproven line-by-line.

## 5. Residual empirical gap (needs confirmation, read-only constraint)
The single unconfirmed sub-hypothesis: that gen5-8's queue-time snapshot actually read `irFinalized=false`.
The logic requires the IR load pipeline to be mid-(re)load at gen5 queue (~t=4s), i.e. the post-gen4
publish transition must trigger a UI-convolver reload that has NOT committed by t=4s.

The D132-7 diag log captures `irFinalized` only in the warmup-FAILURE path (RebuildDispatch L1206),
which never fires (all builds succeed). The `irLoaded` field is printed in CONV_STATUS (L1257-1264) but
`irFinalized` is not.

**To close the gap (DIAG-ONLY, NO behavioral change):** add one line inside the
`#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` CONV_STATUS block (RebuildDispatch.cpp ~L1257) printing
`uiConvolverProcessor.isIRFinalized()`, rebuild with `-DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON`,
re-run D132-7. Expected: gen4=irFinalized=1, gen5-8=irFinalized=0.

## 6. Candidate fix directions (D133-4 to enumerate after user confirmation)
- F1: do NOT freeze a pre-finalized snapshot; instead, in `enqueuePublicationIntentForRuntimeCommit`,
  gate the publish intent on `sealedSnapshot.irFinalized` and, when false, defer via the
  `DeferredFinalizeAware` reason rather than emitting a req that is guaranteed to be rejected.
- F2: make `lastCommittedRebuildGeneration` write its real value at L1302 commit (post-build enqueue)
  so `finalizeReady`/`outstandingRebuild` compute correctly and the timer's auto-recovery can engage.
- F3: admission policy relaxation — treat a rebuild whose irLoaded==true but !irFinalized as
  DeferredFadingActive-like deferred (enqueueDeferred) instead of RejectedNotFinalized, so the
  latest deferred resubmit wins once finalized (bounded single-slot deferred discipline).

(M2 HardReset-independent: `lastCommittedRebuildGeneration` writerless + XFADE-not-logged-for-first
transition are flagged for separate M2 investigation.)
