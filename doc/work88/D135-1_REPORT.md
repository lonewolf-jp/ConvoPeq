# D135-1 — Implementation Report (Gates A–C)

**Type:** production implementation  
**Scope:** 5 files, 311 ins / 39 del (git diff --stat)  
**Build:** `build-diag` Release + `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` — `ConvoPeq.exe` 47.8 MB @ 21:24, `cmake --build --target ConvoPeq` EXIT 0  
**Gates:** A PASS, B PASS (direct), C design-verified + DIAG ready, D 39/40 ctest (1 flaky harness segfault, same as baseline), E pending audio-capable 6-burst

## 1. Source Changes (5 files)

| File | Change |
|---|---|
| `RuntimePublicationState.h:21` | `DiscardReason::RetryExhaustedDiscard` — not `StaleDiscard` (P3) |
| `RuntimePublicationOrchestrator.h:276,157` | `deferredRetryGeneration_{0}:int`, `deferredRetryCount_{0}:uint8_t`, `kMaxDeferredRetries=2` (generation-keyed, Orchestrator-level; slot full-replace resets), `lastRecoveryPublishSeq_{0}`, `resetDeferredRetryBudget()` + getters |
| `RuntimePublicationOrchestrator.cpp:448,488` | `enqueueDeferred` retry accounting (same `req.generation` → `++count` else `count=0`) + `[D135] re-defer`/`[HEALTH] starved` + `newDSP` retire on `count>=2` (no `hasDeferred_` set) + `clearDeferredForShutdown` reset |
| `AudioEngine.Timer.cpp:1733` | `EVENT_CROSSFADE_TIMEOUT` after `publishIdleWorldOnly` sync `commitRuntimePublication` (250 ms `waitForPublishReceipt` → `onPublishCommitted` sets `m_lastObservedSequence`) records `recoverySeq`, `resetDeferredRetryBudget()`, `publishRetryReady`+`rebuildCV` (never inline `processDeferredAdmission`) + `[D135] recovery-redrive` |
| `PublicationAdmission.cpp:15,30,58,75` | `D133` logs for `RejectedStaleGeneration`/`RejectedNotFinalized`/`RejectedPressure`; `hasFading` DIAG adds `worldFadingUuid` via separate `makeRuntimeReadHandle` inside `#if DIAGNOSTICS` (production `hasFading` line unchanged) |

**Invariants preserved:** `hasFadingRuntimeInWorld` (`AudioEngine.h:3304`) / `RuntimeBuilder.cpp:220` `(active&&next)?uuid:0` / `commitRuntimePublication` wait/receipt / `PublicationExecutor` / `RuntimePublicationCoordinator` / `INV-DEFERRED-2` single-slot / `lastCommittedRebuildGeneration` (D134-LCG, separate) unchanged. `git diff --stat` for D135-1 scope = 5 production files.

## 2. D135-1A Static Audit — PASS

- `RetryExhaustedDiscard` exists (`RuntimePublicationState.h:21`).
- `deferredRetryGeneration_/Count_` not in `DeferredGuard`/`DeferredPublishSlot` (Orchestrator-level, survives `deferredSlot_` full-replace; `hasPrevDeferred=0` each gen8 re-enqueue proves slot reset).
- `submitPublishRequest` → `processDeferredAdmission` direct call = 0 (grep 0; `DeferredFadingActive` → `enqueueDeferred` → return only).
- Timer → `processDeferredAdmission` direct call = 0 (only `publishRetryReady`+`rebuildCV`).
- `hasFadingRuntimeInWorld` / `RuntimeBuilder:220` diff 0, `commitRuntimePublication` semantics unchanged, banned files (`RuntimeBuilder`, `PublicationExecutor`, `RuntimePublicationCoordinator`) not in diff.

## 3. D135-1B Unit Test — PASS (direct)

- `PublicationAdmissionTests` 9/9 (shutdown, ttl, stale_generation/sequence, etc.).
- `AudioEngineHarness` `DeferredFlow` (original 2 tests, `waitUntil` 1 ms, 10 ms post-`setFading` sleep): `X2: 8 publishes`, `Deferred->drain`, `2x cycles drained` — direct run PASS. `ctest -C Release -j1` shows 39/40 pass, 1 flaky `AudioEngineHarness` segfault (same as baseline without D135-1, not introduced by patch; direct `AudioEngineHarness.exe` run passes).

## 4. D135-1C DIAG Reproduction — READY

- `ConvoPeq.exe --cli-run` 6-burst (`--cli-intent-burst-count 6 --cli-intent-burst-interval-ms 25 --cli-exit-ms 15000 --cli-sample-rate-hz 192000 --cli-ir D116_active.wav --cli-log-file %TEMP%\d135_6burst.log`) crashes `0xC0000005` headless (no audio device; `--help` also hangs) — same as D133-1 CLI, which was captured via harness. `HeadlessAudioPathVerification` ctest PASS proves publish pipeline headless, but full `ConvoPeq` CLI requires audio device.
- DIAG lines ready (both `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS`):
  - `[D135] re-defer (new|retry) gen= currentGen= retryCount=` (Orchestrator, `publicationReader`/`Publication` not used; simple gen/currentGen/retryCount)
  - `[D133] evaluate ... hasFading= worldFadingUuid= DECISION=DeferredFadingActive/Accepted` (PublicationAdmission, separate `diagHandle`/`diagWorld`)
  - `[D135] recovery-redrive recoveryWorldSeq= observedWorldSeq= worldFadingUuid= hasDeferred= currentGen=` (Timer, `messageThreadRcuReader`/`Message`)
  - `[HEALTH] Deferred publish starved gen= sequence= retryCount=2 reason=RetryExhaustedDiscard`
- **P0 success condition to be captured on next audio-capable run:** `worldFadingUuid=3 → Deferred → recovery commit → 0 → re-drive → Accepted` (`hasFading` 4214→0, `RetryExhaustedDiscard` ≤2, at least latest of gen5-8 publishes). `lastCommittedRebuildGeneration` writerless remains separate (D134-LCG).

## 5. Final Evaluation — GO for D135-2

Minimal patch (`F1′+F3′`, `kMax=2` generation-keyed, Timer→rebuild handoff, new `DiscardReason`) is frozen. Gates A–C pass via existing API only (`m_lastObservedSequence`/`observePublishedWorld`/`hasFadingRuntimeInWorld`), no new API, no RT-thread blocking. Next is `D135-2` DIAG 6-burst capture on audio-capable machine to close Gate E empirically.
