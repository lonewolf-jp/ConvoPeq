# D135-1 — Deferred Fading Recovery Minimal Implementation

**Type:** production implementation  
**Scope:** 5 files, 311 ins / 39 del  
**Build:** `build-diag` Release + `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` — `ConvoPeq.exe` 47,865,344 @ 2026-08-29 21:24, `cmake --build --target ConvoPeq` EXIT 0  
**Gates:** A PASS, B-D PASS (40/40 ctest Release), E DIAG-gated defer/recovery logs ready for next 6-burst

## 1. RetryExhaustedDiscard (RuntimePublicationState.h:21)

Added `DiscardReason::RetryExhaustedDiscard`. `StaleDiscard` not reused — retry-exhausted payload is generation/TTL/sequence-fresh (D135-0 P3).

## 2. Orchestrator-level retry counter (RuntimePublicationOrchestrator.h:294-295, 184-185)

`deferredRetryGeneration_{0} : int`, `deferredRetryCount_{0} : uint8_t`, `kMaxDeferredRetries=2`, `lastRecoveryPublishSeq_{0}`, `deferredRecoveryRearmed_{false}`.  
Generation-keyed (D135-1 gate #2): `PublishRequest` carries only `generation`; new obligation = `++rebuildRequestGeneration` (RebuildDispatch.cpp:655). Sequence is freshness snapshot (`getLastCommittedPublicationSequence()` re-snapshotted each `enqueueDeferred`), not identity. Counter lives on Orchestrator because `enqueueDeferred` does full `DeferredPublishSlot` struct-replace (INV-DEFERRED-2) — slot-level counter would reset to 0 every re-enqueue.

Added `setRecoveryPublishSeq()` / `recoveryPublishSeq()` / `resetDeferredRetryBudget()` (generation-keyed reset, timer-thread calls reset on recovery).

## 3. enqueueDeferred retry accounting (RuntimePublicationOrchestrator.cpp:487-546)

Before `deferredSlot_ = DeferredPublishSlot{}`:

- `sameObligation = (req.generation == deferredRetryGeneration_)` → `++deferredRetryCount_` else `count=0, key=req.generation`.
- DIAG `[D135] re-defer (new|retry) gen= seq= currentGen= worldFadingUuid= worldSeq= hasFading= retryCount=` via `publicationReader`/`ObserveChannel::Publication` + `getRuntimeWorldFromReadHandle()->topology.fadingRuntimeUuid` (correct read path; `observePublishedWorld()` is `RuntimeState*` with no topology).
- If `count > kMaxDeferredRetries(2)` (actual code condition — this doc previously said `>=`; corrected per F5-2/F6-6, the working-tree source `RuntimePublicationOrchestrator.cpp` uses `>`): retire incoming `req.newDSP` via `resolveDSPHandle`+`retireDSPHandleForRuntime` (INV-DEFERRED-2/D131-G2, no `deferredSlot_.reset()` outside `finishView`), log `[HEALTH] Deferred publish starved gen= seq= retryCount=3 reason=RetryExhaustedDiscard`, return without `hasDeferred_=true` (loop ends).
- **★ F6 correction (supersedes the above for retention)**: `DeferredFadingActive` re-drive is *retention*, not a retry — `deferredRetryCount_` is NOT incremented on re-enqueue, and obligation identity is `(generation, recoveryObligationId)` (not generation alone; recovery reuses the current generation with a different payload). Consequently `RetryExhaustedDiscard` is dormant in current production (no Type-A retry path exists); the churn bound is now the coordinator wake watchdog + the 30s TTL measured from `deferredObligationCreatedAtUs` (preserved across re-drive). See `evidence/D135-8-9_GATE_C_F5_FINAL_BOUNDARY_AUDIT.md`.

`clearDeferredForShutdown` also resets `deferredRetryGeneration_/Count_/deferredRecoveryRearmed_/lastRecoveryPublishSeq_`.

## 4. PublicationAdmission DIAG (PublicationAdmission.cpp:76-102)

`evaluate` now holds `readHandle` to emit `worldFadingUuid` in both `DeferredFadingActive` and `Accepted` DIAG lines: `worldFadingUuid = getRuntimeWorldFromReadHandle(readHandle)->topology.fadingRuntimeUuid`. Enables log-only audit `worldFadingUuid=3 → Deferred → recovery → 0 → Accepted`.

## 5. Timeout recovery re-trigger (AudioEngine.Timer.cpp:1689-1732)

After `publishIdleWorldOnly(...,HardReset)` (sync `commitRuntimePublication` → 250ms `waitForPublishReceipt` → `onPublishCommitted` sets `m_lastObservedSequence`):

```
if (runtimeOrchestrator_ && hasDeferredRequest()) {
  recoverySeq = getLastCommittedPublicationSequence();
  setRecoveryPublishSeq(recoverySeq);
  resetDeferredRetryBudget(); // fresh budget for waiting obligation (Gate C/E)
  // DIAG [D135] recovery-redrive recoveryWorldSeq= observedWorldSeq= worldFadingUuid= hasDeferred= currentGen=
  { lock(rebuildMutex) publishRetryReady=true; } rebuildCV.notify_one();
}
```

Timer thread never calls `processDeferredAdmission()` inline (rebuild-thread `jassert`). Handoff via existing `publishRetryReady`/`rebuildCV` (RebuildDispatch.cpp:904). Sequence gate is existing API only (`m_lastObservedSequence`/`observePublishedWorld`), no new API.

## 6. Invariants preserved (Gate A)

- `hasFadingRuntimeInWorld` (AudioEngine.h:3304) unchanged, `RuntimeBuilder.cpp:220` unchanged, `commitRuntimePublication` wait/receipt unchanged, `PublicationExecutor`/`RuntimePublicationCoordinator` unchanged, `INV-DEFERRED-2` single-slot unchanged, `submitPublishRequest`→`processDeferredAdmission` direct call =0, Timer→`processDeferredAdmission` direct call =0, `lastCommittedRebuildGeneration` defect not touched (D134-LCG). `git diff --stat` for D135-1 scope = 5 production files.

## 7. Gates

- **A Static** PASS — 5-file diff, `RetryExhaustedDiscard` exists, retry fields not in slot/guard, no banned calls, `hasFading`/`RuntimeBuilder:220` untouched.
- **B Unit** PASS — `PublicationAdmissionTests` 9/9, `AudioEngineHarness` DeferredFlow/StateMachine all publish pipeline PASS.
- **C Recovery** design-verified — recovery commits `fadingRuntimeUuid=0` sync, records seq, resets budget, wakes rebuild thread; rebuild re-evaluates `hasFading==0` → `Accepted` (Gate C P0).
- **D Regression** PASS — `ctest -C Release` 40/40 PASS (including `RuntimeHealthMonitorTierTests`, `HeadlessAudioPathVerification` 5.73s, `AudioEngineHarness` 14.99s).
- **E 6-burst** — DIAG lines `[D135] re-defer`, `[D135] recovery-redrive`, `[HEALTH] starved` ready; P0 success = `worldFadingUuid=3 → Deferred → recovery commit → 0 → re-drive → Accepted` observable on next DIAG 6-burst. `lastCommittedRebuildGeneration` writerless remains separate defect.

## 8. Next

D135-2 DIAG 6-burst reproduction (`gen4 publish → fading=1 → gen5-8 deferred → timeout recovery → 0 → ≥1 latest deferred publishes`) to close Gate E empirically. Minimal patch is frozen until Gate E log is captured.
