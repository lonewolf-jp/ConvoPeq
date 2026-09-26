# P3-5-FPM-RCA-11 — Shutdown Drain Epoch Liveness Static Closure

- Gate: `P3-5-FPM-RCA-11`
- Mode: `read-only source-only static closure`
- Date: 2026-09-24
- Input: `RCA-10 = PARTIALLY-LOCALIZED / STOP`
- Target: `RCA-7/S7 epoch-equal Engine DQueue head across graceful drain and final wait`
- Scope: `releaseResources epoch chronology, waitForDrain liveness, DeferredDeletionQueue FIFO/epoch gate, quarantine/quiescence boundaries, and terminal timeout classification`
- Verdict: `PARTIALLY-LOCALIZED / STOP`
- Disposition: `NO IMPLEMENTATION / NO CDB RETRY / NO M0 / NO M1 / NO M2 / NO BUILD / NO DR. MEMORY`
- M0 executions: `0`
- M1 executions: `0` (RCA-9's failed attempt is inherited evidence)
- M2 executions: `0`
- Build: `not run`
- Dr. Memory: `not run`
- Production/test/CMake/getter/counter/telemetry/shutdown/epoch/reclaim/router/Recovery changes: `0`

## 1. Gate Question

RCA-11 asks a source-only question:

```text
After the graceful drain segment has ended, can the terminal path make an
epoch-equal FIFO head reclaimable before waitForDrain(2000, 2) returns?
```

The layers remain separate:

```text
direct condition  = entry.epoch == minReaderEpoch == currentEpoch
source liveness   = final wait retries reclaim but does not advance epoch
producer identity = which enqueue created the S7 head
reader identity   = which eligible slot, if any, supplied minReaderEpoch=9
terminal result   = residual -> drain false -> timeout -> Unknown
```

This gate does not reopen the RCA-7 sequence/CAS branch, does not assume an active reader, and does not use a new runtime capture.

## 2. Source Identity

The current generated source authority matches the RCA-8/RCA-9/RCA-10 freeze:

```text
Path       = ConvoPeq.md
Bytes      = 5535334
SHA-256    = E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
LastWrite  = 2026-09-23T14:33:38.6388122Z
Identity   = MATCH
```

Relevant current source hashes:

```text
src/core/EpochDomain.h
  6EC7F1106FE029F862062A4BDB2C7A3F7A14D109E49CA0294B95E7A41F451E7D
src/DeferredDeletionQueue.h
  91143BB072C4D2B2EAE42A012783EF9F28DA0DA8EA7B67FBF70B56E34248A188
src/audioengine/AudioEngine.Processing.ReleaseResources.cpp
  0675CF5137E5A1F917B86D3C517BBF3767550E64051FF9E1C64873B6BAE77E63
src/audioengine/AudioEngine.Threading.cpp
  0B2279DBF30A73F0A54E41C79185394D7B15705742E4128E7AF44558F0EE8E6C
src/audioengine/AudioEngine.Retire.cpp
  E86B8EC22BB80788A31455BE2AA4891590EFEA3756B1E3B3456EB47BD7791198
src/audioengine/ISRRetireRouter.cpp
  7731E933A2D69BC30598F2173F106B725A884F9BBA235EB4B3D41004CD85B638
src/audioengine/ISRRuntimePublicationCoordinator.cpp
  1AD84A0C3F2FFB7E796A43F979B0D05A8827EBA2631E3A69E01894E5A32DD8C0
src/audioengine/ISRShutdown.cpp
  77AECAC3772CB41F9409B5D3D9472EF95DC40FA7872338407F4A2D491FCBF2EA
```

Pre-existing worktree changes were preserved. RCA-11 changed no source or test file.

## 3. Frozen S7 Condition

The existing RCA-7 M1 evidence remains the only S7 runtime fact used here:

```text
globalEpoch       = 9
enqueuePos        = 5
dequeuePos        = 4
head sequence     = 5
expectedSequence  = 5
head entry epoch  = 9
getMinReaderEpoch = 9
sequence gate     = PASS
epoch gate        = BLOCK
isOlder(9, 9)     = false
dequeue CAS       = NOT REACHED
```

The existing S7 log places the capture after the terminal `releaseResources()` UI-release section and before the later final-wait timeout path (`rca7-cdb.log:225-257,360`). The later destructor control advanced the supplied minimum to `12` and allowed the same head to CAS from `4` to `5` (`rca7-cdb.log:628-721`).

RCA-11 does not reinterpret the failed RCA-9 slot scan and does not create a new capture.

## 4. Epoch and Queue Contracts

### 4.1 Publication changes the baseline; reclaim does not

`EpochDomain::publishEpoch()` increments both `epochGeneration_` and `globalEpoch` (`src/core/EpochDomain.h:192-202`). `AudioEngine::markRetireEpoch()` and `advanceRetireEpoch()` both delegate to that operation (`src/audioengine/AudioEngine.Publication.cpp:16-29`).

`EpochDomain::tryReclaim()` only calls:

```text
deferredDeletionQueue.reclaim(getMinReaderEpoch())
```

at `src/core/EpochDomain.h:385-395`. It does not publish an epoch.

`getMinReaderEpoch()` starts at `currentEpoch()` and lowers the result only for an eligible slot whose epoch is strictly older (`src/core/EpochDomain.h:214-248`). Therefore, at the S7 state:

```text
no eligible reader       -> minReaderEpoch=9
eligible reader at 9     -> minReaderEpoch=9
eligible reader above 9  -> minReaderEpoch=9
```

The equality is not reader identity evidence.

### 4.2 DQueue is strict-epoch and FIFO

`DeferredDeletionQueue::reclaim()` first requires the head sequence to be ready, then checks:

```text
isOlder(entry.epoch, minReaderEpoch)
```

at `src/DeferredDeletionQueue.h:120-134`. If the head is not reclaimable, it breaks immediately to preserve FIFO order (`src/DeferredDeletionQueue.h:171-175`). A later reclaimable entry cannot pass the blocked head.

For S7, `isOlder(9,9)` is false, so the queue cannot advance without a new minimum-reader boundary.

### 4.3 The normal enqueue helper advances the epoch

`enqueueDeferredDeleteNonRtWithResult()` calls `markRetireEpoch()` before the normal router enqueue (`src/audioengine/AudioEngine.h:4460-4481`). Thus a normal global deferred enqueue can both advance the current epoch and stamp the entry with the newly published value.

When shutdown is already in progress, the same helper still calls `markRetireEpoch()` and then transfers ownership to `shutdownReclaim()`/TerminalReclaimAuthority instead of the normal DQueue path (`src/audioengine/AudioEngine.h:4466-4475`; `src/audioengine/ISRRetireRouter.cpp:570-584`). Thus a post-graceful shutdown enqueue can advance `globalEpoch` without adding a DQueue entry. This distinction matters when separating epoch progress from DQueue insertion.

### 4.4 Safe drain and force-drain boundaries

`drainDeferredRetireQueues()` calls `m_retireRouter->tryReclaim()` and coordinator reclaim, but never calls `publishEpoch()` (`src/audioengine/AudioEngine.Retire.cpp:45-66`).

`ISRRetireRouter::tryReclaim()` drains the DQueue and the epoch-gated Q/E/T stores, but does not publish an epoch (`src/audioengine/ISRRetireRouter.cpp:386-394,541-561`).

`drainAllQuarantineStore()` drains Q, EmergencyQ, and TerminalReclaimAuthority only; it does not drain the DQueue (`src/audioengine/ISRRetireRouter.cpp:443-451`). The release path calls that helper at `ReleaseResources.cpp:444` and `:555`, but never calls `m_retireRouter->drainAll()` or `m_epochDomain.drainAll()`.

The only inspected terminal force-drain of the DQueue is the abnormal `~AudioEngine()` fallback (`src/audioengine/AudioEngine.CtorDtor.cpp:277-292`). That later destructor path is consistent with the RCA-7 recovery control, but it is not the normal `releaseResources()` final wait.

## 5. Exact releaseResources Chronology

| order | source operation | epoch effect | DQueue/drain effect | relevance |
|---:|---|---|---|---|
| 1 | `joinProducers()` retry loop at `:250-257` | no publish in the nested `waitForDrain(100,1)` | safe drain only | can leave residuals before the terminal baseline |
| 2 | `advanceRetireEpoch()` at `:262-265` | one explicit increment | no direct reclaim | establishes the terminal baseline |
| 3 | `closeReaderRegistration()` at `:269-274` | none | blocks new RCUReader registration; does not force existing readers out | does not change DQueue head safety |
| 4 | graceful loop at `:285-311` | `publishEpoch()` after each poll at `:305`; one extra publish at timeout `:323` | `tryReclaim()` after each publish; break checks only DQueue pending count and active readers | primary epoch-progress window |
| 5 | graceful-loop exit at `:313-353` | no mandatory post-loop publish on the normal break path | timeout branch does a final safe drain; normal break has no final reclaim | can leave non-DQueue residuals even when the loop exits |
| 6 | `drainDeferredRetireQueues(true)` at `:373-376` | no publish | D/Q/E/T safe drain | marks `ReclaimComplete`; it is not a DQueue force drain |
| 7 | VerifyDrained and UI processor release at `:474-528` | Convolver shutdown retirement can call the global helper, which publishes an epoch before terminal transfer; EQ release advances its nested domain | UI-owned resources are released; S7 occurs in this source region, but the exact UI producer is not identified | S7 producer is not identified by the phase marker alone |
| 8 | world clear at `:537-555` | the shutdown-aware helper can publish an epoch before Terminal transfer | world retirement normally transfers to Terminal authority; Q/E/T force drain still excludes DQueue | later than the S7 marker; it can change the baseline without repairing the DQueue |
| 9 | final DSP/deferred disposition at `:568-600` | no explicit global publish in these direct calls | `DSPLifetimeManager::retire()` and deferred shutdown disposition can enqueue a new DQueue entry | later than S7; possible new residual, not the captured head |
| 10 | `waitForDrain(2000,2)` at `:615-616` | no publish in the wait loop | each iteration calls `drainDeferredRetireQueues(true)` | the S7 first Engine reclaim is reached through this safe retry boundary |
| 11 | timeout classification at `:624-633` | no publish | selects `Unknown` unless a stuck reader or active builder is found | residual DQueue depth is not consulted for reason selection |
| 12 | timeout safe retry at `:654-662` | no publish | another safe drain plus `m_epochDomain.tryReclaim()` | cannot change an epoch-equal head |
| 13 | coordinator/owner residual handling at `:664-697` | no guaranteed global publish | owner residual is transferred through the shutdown-aware deferred helper; this is not a DQueue repair | terminal bookkeeping follows the failed wait |
| 14 | final transitions at `:724-750` | no publish | phase/result transition does not reclaim the DQueue | `ShutdownComplete` does not imply drain success |

## 6. Source-Level Liveness Finding

### Finding 1 — The final wait has no epoch-progress operation

`waitForDrain()` loops only while `isFullyDrained()` is false, calls `drainDeferredRetireQueues(true)`, checks the deadline, and sleeps (`src/audioengine/AudioEngine.Threading.cpp:215-247`). Neither `waitForDrain()` nor `drainDeferredRetireQueues()` publishes an epoch.

Consequently, if the state at the final wait is:

```text
head.entry.epoch = 9
currentEpoch     = 9
minReaderEpoch   = 9
```

then every safe retry evaluates the same false epoch predicate. The final wait itself contains no epoch publication, and the normal terminal path does not force-drain the DQueue. A preceding post-graceful helper can publish an epoch, but the captured S7 values only show that the first final-wait reclaim still received the same epoch boundary; they do not identify whether an earlier helper published the value with which the head was stamped. The only source-level ways to make this head reclaimable are an effective later epoch publication or a DQueue force-drain.

This is a source-proven liveness gap. It is sufficient to explain how an epoch-equal head can survive the final wait without an active reader; it does not identify the enqueue that created the head.

### Finding 2 — The graceful loop's success predicate is narrower than final drain

The graceful loop breaks at `:287-289` when:

```text
pendingRetireCount == 0
activeReaderCount  == 0
```

It does not test `isFullyDrained()`. The final predicate additionally checks pending reclaim handles, RetireIntent residency, overflow/quarantine/terminal stores, and `runtimePublicationBridge_.isFullyDrained()` (`src/audioengine/AudioEngine.Threading.cpp:153-213`).

Therefore the graceful segment can finish before the complete terminal drain predicate is true. This explains how a residual can survive the `ReclaimComplete` marker; it does not prove which residual was present at the loop's exit in the S7 episode.

### Finding 3 — A simple pre-loop no-reader explanation is constrained but not proven impossible

If an epoch-equal head were already published before the graceful loop, and no eligible reader held the old epoch, the loop's first `publishEpoch()` followed by `tryReclaim()` would make the old head reclaimable. Thus the S7 state points to at least one of:

```text
- the head was enqueued after the last relevant reclaim;
- a reader/lifecycle transition affected the supplied minimum;
- the DQueue entry was not visible to the loop's pending-count observation;
- the exact producer chronology is outside the source-only evidence.
```

The source does not select among these cases. In particular, the S7 `destroyDSPCoreNode` deleter and epoch `9` do not prove that the entry was created by the later world-clear or final-DSP paths; those paths are textually after the S7 marker in the existing capture.

### Finding 4 — Later shutdown enqueue paths are real but temporally downstream

The following normal source paths can still create DQueue residuals during shutdown:

- `DSPLifetimeManager::retire()` calls `router_->enqueueWithRetry()` even when called from final DSP disposition (`src/audioengine/DSPLifetimeManager.cpp:35-77`).
- `RuntimePublicationOrchestrator::clearDeferredForShutdown()` disposes a deferred registered DSP through `retireRegisteredDSP()` (`src/audioengine/RuntimePublicationOrchestrator.cpp:590-639`).
- `RuntimePublicationOrchestrator::retireRegisteredDSP()` delegates to `DSPLifetimeManager::retire()` (`src/audioengine/RuntimePublicationOrchestrator.cpp:735-762`).
- `ConvolverProcessor::releaseResources()` can call the global deferred helper for a retired Convolver object (`src/convolver/ConvolverProcessor.Lifecycle.cpp:462-499,65-73`); during shutdown that helper publishes an epoch but transfers ownership to Terminal authority, so it is not itself a DQueue insertion path.

These operations are relevant to future shutdown residual analysis, but they occur after the S7 marker in the current chronology. The first two can create a new DQueue residual; the Convolver path can advance the baseline without repairing an existing DQueue head. None can be promoted to the S7 head producer without new evidence.

### Finding 5 — Quiescence proof does not repair the DQueue

`tryShutdownQuiescentReclaim()` builds Q0-Q7 observations and obtains a Permit (`src/audioengine/AudioEngine.h:4638-4672`). The Permit is consumed by `reclaimShutdownQuiescent()`, which retires/reclaims a `DSPHandle` and bypasses the handle-level epoch check (`src/audioengine/ISRRuntimePublicationCoordinator.cpp:748-778`).

That path does not call `EpochDomain::tryReclaim()` for the global DQueue and does not publish an epoch. `epochSettled` is supplied as `true` by the caller (`AudioEngine.h:4651`), rather than derived from a DQueue-head observation. Quiescence proof therefore cannot be used as evidence that the S7 DQueue head is reclaimable.

### Finding 6 — `Unknown` is expected even when a DQueue residual is observable

On final-wait timeout, `releaseResources()` selects `ReaderActive` only when `stuckReaderCount>0`, then `ActiveBuilder` if the builder flag is set, and otherwise selects `Unknown` (`ReleaseResources.cpp:624-633`). It does not use `audit.routerPendingRetire`, `audit.pendingRetire`, or the DQueue head state for this choice.

`RuntimeDrainAudit::getPrimaryBlockingReason()` has a `RouterPendingRetire` classification (`src/audioengine/RuntimeDrainAudit.h:52-74`), but the timeout path does not call it. Therefore `Unknown` is compatible with a proven DQueue residual and does not identify the epoch holder or enqueue source.

### Finding 7 — Deadline ordering can report timeout without a final predicate recheck

`waitForDrain()` checks `isFullyDrained()` before each drain, then checks elapsed time immediately after `drainDeferredRetireQueues(true)` and can return `false` before evaluating the predicate again (`AudioEngine.Threading.cpp:235-244`). This is a secondary timeout-classification edge. It is not the RCA-7 S7 direct cause because the S7 head was still blocked, but it is a source-level reason not to interpret every timeout as proof that no progress occurred during the last drain call.

## 7. Reclassification of the S7 Question

| item | RCA-11 status | basis |
|---|---|---|
| sequence-ready FIFO head | PROVEN by RCA-7 | sequence `5` equals `dequeuePos+1` |
| epoch gate block | PROVEN by RCA-7 | `entry.epoch=9`, supplied minimum `9`, `isOlder=false` |
| CAS not reached on first call | PROVEN by RCA-7 | return occurs before the CAS |
| final wait advances epoch | DISPROVEN by source | `waitForDrain` and `drainDeferredRetireQueues` contain no publish |
| releaseResources force-drains DQueue | DISPROVEN by source | only Q/E/T force-drain is called; D force-drain is in destructor fallback |
| S7 head producer | UNRESOLVED | existing source/log boundary does not identify the enqueue call |
| S7 reader identity | UNRESOLVED | RCA-9 valid slot capture is absent |
| no-reader explanation | POSSIBLE, NOT PROVEN | epoch-equal head can be baseline-only, but exact chronology is missing |
| reader-at-9 explanation | POSSIBLE, NOT PROVEN | fixed-slot and RAII paths remain source-compatible |
| graceful-loop predicate mismatch | PROVEN | loop exit predicate is narrower than `isFullyDrained` |
| quiescence proof repairs DQueue | REJECTED | Permit path is handle-level and does not publish/reclaim DQueue |
| `Unknown` means no DQueue residual | REJECTED | timeout reason selection ignores router/DQueue terms |
| DQueue sequence/CAS corruption | NOT SUPPORTED | later RCA-7 control CAS succeeds after epoch progress |
| terminal timeout consequence | PROVEN at contract level | residual can keep drain false and reach `markTimedOut(Unknown)` |

## 8. STOP Matrix and Disposition

```text
STOP-1_SOURCE_IDENTITY_MISMATCH       = NOT_TRIGGERED
STOP-2_EXACT_S7_PRODUCER_CHRONOLOGY   = TRIGGERED
STOP-3_READER_IDENTITY_UNRESOLVED     = TRIGGERED
STOP-4_FINAL_WAIT_EPOCH_LIVENESS      = PROVEN
STOP-5_QUIESCENCE_TO_DQUEUE_LINK      = REJECTED
```

The source-only gate closes the question of whether the final wait itself supplies epoch progress: it does not. It does not close the upstream question of which enqueue or reader lifecycle produced the S7 state.

```text
RCA11_GATE_DISPOSITION = PARTIALLY-LOCALIZED / STOP
DIRECT_DQUEUE_CAUSE = PROVEN
FINAL_WAIT_EPOCH_PROGRESS = PROVEN ABSENT
S7_ENTRY_PRODUCER = UNRESOLVED
S7_READER_IDENTITY = UNRESOLVED
CASE_A = POSSIBLE_NOT_PROVEN
CASE_B = POSSIBLE_NOT_PROVEN
CASE_C = POSSIBLE_NOT_PROVEN
CASE_D = POSSIBLE_NOT_PROVEN
GRACEFUL_LOOP_PREDICATE_MISMATCH = PROVEN
QUIESCENCE_PROOF_DQUEUE_REPAIR = REJECTED
M0 = 0
M1 = 0 (RCA-9 inherited 1 failed attempt)
M2 = 0
BUILD = NOT_RUN
DR_MEMORY = 0
CDB_RETRY = FORBIDDEN
IMPLEMENTATION = FORBIDDEN
FURTHER_CAPTURE = NOT_AUTHORIZED
```

A future runtime gate, if separately authorized, would need to correlate the S7 enqueue call and reader-slot chronology. This report does not authorize a CDB retry, build, production change, queue repair, or test change.

## 9. Evidence Index

| evidence | location |
|---|---|
| Source identity | `ConvoPeq.md`, SHA-256 `E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609`, timestamp `2026-09-23T14:33:38.6388122Z` |
| Epoch publication/current | `src/core/EpochDomain.h:186-202`; `src/audioengine/AudioEngine.Publication.cpp:11-29` |
| Minimum-reader scan | `src/core/EpochDomain.h:214-248` |
| DQueue epoch/FIFO reclaim | `src/DeferredDeletionQueue.h:109-177` |
| DQueue size | `src/DeferredDeletionQueue.h:219-224` |
| Safe deferred drain | `src/audioengine/AudioEngine.Retire.cpp:45-66` |
| Router reclaim and Q/E/T boundaries | `src/audioengine/ISRRetireRouter.cpp:386-394,443-451,541-584` |
| Early wait and terminal graceful loop | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:250-376` |
| Post-graceful shutdown operations | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:474-608` |
| Final wait and timeout classification | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:615-662` |
| Final coordinator/owner handling | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:664-745` |
| AudioEngine drain predicate | `src/audioengine/AudioEngine.Threading.cpp:153-213` |
| `waitForDrain` implementation | `src/audioengine/AudioEngine.Threading.cpp:215-247` |
| Drain audit and reason enum | `src/audioengine/AudioEngine.Threading.cpp:108-150`; `src/audioengine/RuntimeDrainAudit.h:52-84` |
| Shutdown result/terminal skip | `src/audioengine/ISRShutdown.cpp:72-180` |
| Quiescence proof | `src/audioengine/AudioEngine.h:4638-4672`; `src/audioengine/ISRShutdown.cpp:350-412` |
| Permit-based handle reclaim | `src/audioengine/ISRRuntimePublicationCoordinator.cpp:748-778` |
| DSP retirement enqueue | `src/audioengine/DSPLifetimeManager.cpp:35-137` |
| Deferred shutdown DSP disposition | `src/audioengine/RuntimePublicationOrchestrator.cpp:590-639,735-762` |
| Existing S7 direct evidence | `doc/work113/P1-5-IR-P2_Step5BA_P3-5-FPM-RCA-7_DQueue_head_block_exact.md` |
| Existing terminal reconciliation | `doc/work113/P1-5-IR-P2_Step5BA_P3-5-FPM-RCA-8_EpochHold_terminal_contract_reconciliation.md` |
| Existing reader/epoch boundary | `doc/work113/P1-5-IR-P2_Step5BA_P3-5-FPM-RCA-9_EpochProvenance_reader_lifecycle_exact.md` |
| Existing source closure | `doc/work113/P1-5-IR-P2_Step5BA_P3-5-FPM-RCA-10_ReaderLifecycle_epoch_provenance_static_closure.md` |
| Existing S7 log | `C:\Users\user\AppData\Local\Temp\opencode\rca7-cdb.log:225-257,360,628-721` |
| Existing failed M1 evidence | `C:\Users\user\AppData\Local\Temp\opencode\rca9-cdb.log:315-336` |

No new runtime evidence was created by RCA-11. This report is a source-level liveness closure, not an implementation authorization.
