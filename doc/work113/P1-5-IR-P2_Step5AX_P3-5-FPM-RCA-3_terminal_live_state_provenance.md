# P3-5-FPM-RCA-3 — M0 Terminal Live-State / D-Queue Provenance

- Gate: `P3-5-FPM-RCA-3`
- Mode: read-only M0 live debugger capture
- Date: 2026-09-24
- Primary verdict: `PARTIALLY-LOCALIZED`
- Gate disposition: `STOP / S7-BRIDGE-TEMPORAL-CAPTURE-INCOMPLETE`
- `S7-DIRECT-LAYER1`: `PARTIAL`
- `S7-S10-CONTINUITY`: `PROVEN`
- `S10-NEW-DENTRY`: `PROVEN`
- `D-QUEUE-2-PROVEN`: `NO`
- `SHUTDOWN-COMPLETE`: `NOT-PROVEN`
- Production/test/CMake/getter/counter/telemetry/shutdown/epoch/reclaim/router/Recovery changes: `0`
- Build: `not run`
- Dr. Memory: `not run`
- M1/M2 live runs: `0`

## 1. Scope / State Freeze

This gate observes the existing Release executable without changing source, tests, CMake, counters, telemetry, shutdown logic, epoch logic, reclaim logic, router logic, or Recovery logic.

The accepted live-debug authorization was used for the existing `--fpm-m0` executable. The evidence chain used for the verdict is the final uninterrupted CDB execution from S7 through S10. CDB setup/recovery executions occurred before that chain; they are not used as independent causal evidence.

Frozen values:

- production source changes: 0
- test vehicle changes: 0
- CMake changes: 0
- getter/counter additions: 0
- shutdown, epoch, reclaim, router, Recovery semantic changes: 0
- binary changes: 0
- M1/M2 runs: 0
- build: not run
- Dr. Memory: not run

The report does not propose a fix.

## 2. `ConvoPeq(3).md` Reconciliation

The user clarification is applied: `ConvoPeq(3).md` means `C:\VSC_Project\ConvoPeq\ConvoPeq.md`.

Observed authority:

- Path: `ConvoPeq.md`
- Size: `5,535,334` bytes
- Lines: `121,294`
- SHA-256: `E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609`

Relevant live-source authorities:

- `AudioEngine::isFullyDrained()`: `src/audioengine/AudioEngine.Threading.cpp:153-213`
- `AudioEngine::waitForDrain()`: `src/audioengine/AudioEngine.Threading.cpp:215-247`
- `releaseResources()` drain/finalize sequence: `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:615-745`
- `RuntimeIntentCoordinator::ShutdownScheduler::isFullyDrained()`: `src/audioengine/ISRRuntimePublicationCoordinator.cpp:511-562`
- `EpochDomain::pendingRetireCount()`: `src/core/EpochDomain.h:415-427`
- `DeferredDeletionQueue::sizeApprox()`: `src/DeferredDeletionQueue.h:219-224`
- `SnapshotCoordinator::finalizeShutdown()`: `src/core/SnapshotCoordinator.h:60-72`
- `SnapshotCoordinator::retireCurrentAndTarget()`: `src/core/SnapshotCoordinator.h:165-184`
- `ShutdownRuntime::markTimedOut()` declaration: `src/audioengine/ISRShutdown.h:216-218`
- `RuntimePublicationBridge::markShutdownComplete()`: `src/audioengine/ISRRuntimePublicationCoordinator.cpp:568-579`

## 3. Symbol / Address Recovery

The Release PDB contains type records but no code-debug stream, globals, or publics:

- `Has Debug Info: false`
- `Has Globals: false`
- `Has Publics: false`
- `Has Types: true`

Therefore source-line and function-name breakpoints were not usable. Runtime addresses were recovered from PE disassembly, `.pdata` unwind ranges, and CDB type information. The preferred-image addresses and the actual loaded image base were:

- preferred image base: `0x140000000`
- actual image base: `0x7ff7ae6b0000`
- PDB: `build/Release/AudioEngineHarness.pdb`
- PDB summary GUID: `{D0019BC1-920C-46AE-B4EA-46E2FCDA3722}`

Recovered address map:

| point | preferred address | purpose |
|---|---:|---|
| `releaseResources` entry | `0x141fcaef0` | live M0 shutdown entry |
| `waitForDrain(2000,2)` call setup | `0x141fcc50a` | confirms timeout/poll arguments |
| `waitForDrain` entry | `0x141fa7eb0` | function entry |
| S7 timeout return | `0x141fa80b3` | `xor al,al` before false epilogue |
| drain audit call | `0x141fcc535` | timeout audit boundary |
| `markTimedOut` call | `0x141fcc558` | reason in `edx` |
| after `markTimedOut` | `0x141fcc55d` | phase/reason readback |
| `finalizeShutdown` inline entry | `0x141fcc7ba` | S10 pre-state |
| `publishEpoch` call | `0x141fcc7cb` | epoch publication boundary |
| after `publishEpoch` | `0x141fcc7ce` | returned retire epoch/current epoch |
| current-slot enqueue call | `0x141fcc7f9` | `retireCurrentAndTarget` current |
| after current enqueue | `0x141fcc7fd` | enqueue result |
| target-slot enqueue call | `0x141fcc86d` | target enqueue boundary |
| after target enqueue | `0x141fcc872` | target enqueue result |
| before finalized flag | `0x141fcc8ca` | all retire work complete |
| after finalized flag | `0x141fcc8d2` | `m_shutdownFinalized=true` path |
| `markShutdownComplete` call | `0x141fccddc` | bridge completion boundary |
| after bridge completion call | `0x1fcce01` | bridge state readback |

The PDB member layout was stale by `0x40` for two late `AudioEngine` members. Runtime addresses were corrected from the executed code:

- actual `ShutdownRuntime` base: `this + 0x15462c0`
- actual `RuntimeIntentCoordinator`/bridge base: `this + 0x110a480`
- stale PDB-reported bases were not used for the final bridge/phase reads.

This offset correction is material to the verdict and is the reason the exact S7 bridge snapshot is marked incomplete below.

## 4. Runtime Object Topology

The live M0 object addresses captured at S7 were:

| object | address / derivation |
|---|---|
| `AudioEngine *this` | `0x00000288e5730080` |
| `m_epochDomain` | `this + 0x10b76c0 = 0x00000288e67e7740` |
| engine `DeferredDeletionQueue` | `m_epochDomain + 0x1440 = 0x00000288e67e8b80` |
| `m_retireRouter` storage | `this + 0x10ecca8` |
| `m_retireRouter` object | `0x00000288e3d14020` |
| `runtimeOrchestrator_` object | `0x00000288e3d242d0` |
| `m_coordinator` | `this + 0x10edac0 = 0x00000288e681db40` |
| actual bridge coordinator | `this + 0x110a480 = 0x00000288e683a500` |
| actual `ShutdownRuntime` | `this + 0x15462c0 = 0x00000288e6c76340` |
| `pendingReclaimHandles_` storage | `this + 0x10f1d88 = 0x00000288e6821e08` |

`m_retireRouter->pendingRetireCount()` delegates to the provider's D queue. Therefore the provider queue depth, not the router's separate `m_trackedPendingEntries_` field, is the Layer-1 `retireDepth` source.

## 5. S7 — `waitForDrain()` End State

The S7 breakpoint at `0x141fa80b3` was reached in the same CDB session that later reached S10.

Entry arguments were also observed:

- `rcx = this`
- `edx = 0x7d0` (`2000` ms)
- `r8d = 0x2` (`2` ms poll interval)

At S7:

- `drainedWithinBudget = false`
- `isFullyDrained() = false`
- the false-return branch was executing
- the CDB register state showed `rbx = this`

### 5.1 Layer-1 nine-predicate state

| predicate | observed S7 value | result | evidence |
|---|---:|---|---|
| `!hasDeferredCommit` | `hasDeferred_ = 0` | PASS | orchestrator `0x00000288e3d242d0 + 0x370` |
| `pendingReclaimEmpty` | vector begin/end/capacity all `0` | PASS | vector storage `0x00000288e6821e08` |
| `retireDepth == 0` | `5` | FAIL | provider D queue `enqueuePos=5`, `dequeuePos=0`; `EpochDomain::pendingRetireCount()` delegates to this queue |
| `lifetimeRetireIntentPending == 0` | `2` | FAIL | `enqueueTicket=2`, `dequeuePos=0`, `fallbackCount=0` |
| `ringResident == 0` | `0` | PASS | `LifetimeState::overflowRing_ = null` |
| `dspQuarantineResident == 0` | `0` | PASS | S7 audit/terminal diagnostic; flag-base temporal capture is listed as a limitation in Section 16 |
| `retireQuarantineResident == 0` | `0` | PASS | router Q and EmergencyQ resident fields were zero |
| `terminalReclaimResident == 0` | `0` | PASS | terminal resident atomic was zero and terminal vector was empty |
| `runtimePublicationBridge_.isFullyDrained()` | `false` | FAIL | exact S7 temporal snapshot incomplete; continuous S10 state has three false bridge predicates |

The direct Layer-1 result is therefore not a single `pendingReclaimHandles_` failure. At least the provider D depth (`5`) and LifetimeState pending-intent count (`2`) were already non-zero at S7. The bridge predicate was also false in the immediately following state.

### 5.2 `pendingReclaimHandles_`

The vector storage was read as:

- begin: `0`
- end: `0`
- capacity: `0`

Therefore:

```text
pendingReclaimHandles_ = PROVEN-EMPTY
```

This member did not cause the S7 timeout.

### 5.3 D queue depth and entries at S7

The engine `DeferredDeletionQueue` positions were:

- `enqueuePos = 5`
- `dequeuePos = 0`
- `sizeApprox = 5`
- published sequence values: `0x1000, 0x1001, 0x1002, 0x1003, 0x1004`

The five resident entries were:

| slot | ptr | deleter | epoch | type | publicationSequenceId | generation |
|---:|---|---|---:|---|---:|---:|
| 0 | `0x0000000000000000` | `0x0000000000000000` | 1 | `Generic (0)` | 0 | 0 |
| 1 | `0x0000000000000000` | `0x0000000000000000` | 2 | `Generic (0)` | 0 | 0 |
| 2 | `0x0000000000000000` | `0x0000000000000000` | 4 | `Generic (0)` | 0 | 0 |
| 3 | `0x0000000000000000` | `0x0000000000000000` | 5 | `Generic (0)` | 0 | 0 |
| 4 | `0x00000288f4d3e080` | `0x00007ff7b06579c0` | 9 | `Generic (0)` | 0 | 0 |

`DeletionEntry` has no `objectBytes` field. That requested value is therefore `UNAVAILABLE` for every entry.

The S7 D queue was not the two-entry aggregate assumed from the prior FPM report. The live queue contained five published positions, including four null-pointer entries and one concrete pointer.

### 5.4 Epoch and reader state

- `currentEpoch` at S7: `9`
- `minReaderEpoch`: `9`, derived from `getMinReaderEpoch()` with no active readers
- active readers: `0`
- stuck readers: `0`
- reader slot scan: no nonzero depth observed

The D entries at epoch `9` were not reclaimable solely by the epoch-advance rule while the live epoch remained `9` and no later epoch had yet been observed at S7.

## 6. RuntimePublicationBridge 14 Predicates

The final continuous-state bridge read used the execution-derived base `0x00000288e683a500`, not the stale PDB-reported base.

The following values were captured at the post-S10/pre-`markShutdownComplete` boundary. They are the best continuous-state bridge snapshot, but the exact temporal S7 read was not captured after correcting the base-address drift.

| # | `ShutdownScheduler::isFullyDrained()` predicate | value | result |
|---:|---|---:|---|
| 1 | `swapPending_` | 0 | PASS |
| 2 | `intentQueue_.sizeApprox()` | 2 (`enq=2`, `deq=0`) | FAIL |
| 3 | `observeDeferredRing_.size()` | 0 | PASS |
| 4 | `quarantineFallbackQueue_.sizeApprox()` | 0 | PASS |
| 5 | `recoveryIntentQueue_.size()` | 0 | PASS |
| 6 | `retireBacklogCount_` | 61 | FAIL |
| 7 | `publicationBacklogCount_` | 0 | PASS |
| 8 | `publicationIntentResidencyCount_` | 0 | PASS |
| 9 | `pendingIntentCount_` | 0 | PASS |
| 10 | `reclaimInFlightCount_` | 0 | PASS |
| 11 | `quarantineIntentResidencyCount_` | 2 | FAIL |
| 12 | `quarantineRingResidencyCount_` | 0 | PASS |
| 13 | `recoveryAdmissionPending_` | 0 | PASS |
| 14 | `liveLogicalRecoveryObligationCount()` | 0 | PASS |

The continuous bridge state was not drained. The bridge state byte at the `markShutdownComplete` boundary was `0` (`Bootstrapping`), not `ShuttingDown` (`5`). `markShutdownComplete()` therefore returned without changing the state, consistent with `ISRRuntimePublicationCoordinator.cpp:568-579`.

The exact S7 temporal values for these 14 fields remain `SOURCE-UNRESOLVED`. The three false values above are sufficient to explain why the immediately following bridge predicate was false, but they are not promoted to an exact S7 snapshot without a same-point capture.

## 7. S10 — `finalizeShutdown(true)` Before / After

The S10 pre-state was captured at `0x141fcc7ba` in the same execution as S7.

### 7.1 Before finalize

- `timedOut = true`
- `drainedWithinBudget = false`
- `m_shutdownFinalized = false`
- `current = 0x00000288ee1dc010`
- `target = 0`
- `currentEpoch = 9`
- D queue depth: `5`

The separate `AudioEngine::shutdownPhase` atomic was `5` (`DrainRetire`).

### 7.2 `publishEpoch()`

At `0x141fcc7cb` the provider virtual call returned:

- returned retire epoch (`rax`): `9`
- global epoch before call: `9`
- global epoch after call: `10`

The provider uses a fetch-add return value, so the captured return value `9` is the retire epoch stamped into the enqueue entry.

### 7.3 Current-slot enqueue

The current slot was exchanged to null and the first enqueue completed successfully:

- retired pointer: `0x00000288ee1dc010`
- deleter: `0x00007ff7b06473a0`
- retire epoch: `9`
- enqueue result: `true` (`al=1`)
- D queue depth after enqueue: `6`

The target slot was null, so no target entry was enqueued.

The new entry was:

| queue slot | ptr | deleter | epoch | type | publicationSequenceId | generation |
|---:|---|---|---:|---|---:|---:|
| 5 | `0x00000288ee1dc010` | `0x00007ff7b06473a0` | 9 | `Generic (0)` | 0 | 0 |

This is a direct provenance match for the current-slot branch of `SnapshotCoordinator::retireCurrentAndTarget()`.

### 7.4 After finalize

At the before/after finalized-flag boundary:

- `current = 0`
- `target = 0`
- `m_shutdownFinalized = 1`
- global epoch: `10`
- D queue depth: `6`
- `timedOut = true`
- provider `tryReclaim()` was skipped by the timed-out branch

The S10 new-entry provenance is therefore:

```text
S10-NEW-DENTRY = PROVEN
```

The five entries already present at S7 remain separate from this newly proven entry. Their writer/caller sites are not identified by the live capture.

## 8. Timeout, `Unknown`, and Phase State

At the `markTimedOut` call boundary:

- `edx = 9`
- `ShutdownBlockingReason(9) = Unknown`
- `shutdownPhase` (the `AudioEngine` atomic) was `5` (`DrainRetire`)
- the outer `ShutdownRuntime` last non-terminal phase became `7` (`VerifyDrained`)

After the call:

- `ShutdownRuntime::phase_ = 8` (`TimedOut`)
- `lastNonTerminalPhase_ = 7` (`VerifyDrained`)
- `blockingReason_ = 9` (`Unknown`)

The source chooses `ReaderActive` only for a nonzero stuck-reader count and `ActiveBuilder` only for a running builder. The live state had zero active/stuck readers and the rebuild thread was stopped, so the default `Unknown` reason is source-consistent and directly observed at the call boundary.

No `ShutdownComplete` transition was observed in the live breakpoints. The bridge state remained `Bootstrapping` at the completion call, and the terminal diagnostic later reported the coordinator in `Faulted` state after completion handling.

## 9. D-Queue Timeline and Provenance Verdict

| point | D depth | observed action |
|---|---:|---|
| S7 timeout return | 5 | five entries already resident |
| after System-1 drain boundary | 5 | D queue unchanged at the audit boundary |
| S10 before finalize | 5 | current snapshot still present, target null |
| after current-slot enqueue | 6 | one new `Generic` entry, epoch 9 |
| after finalized flag | 6 | current/target null, finalized true |
| later audit diagnostic | not used as a depth snapshot | log reported `routerPendingRetire=2`; asynchronous/intermediate timing is not equated with the breakpoint queue snapshots |

The router/provider distinction is essential:

- `m_retireRouter->pendingRetireCount()` delegates to the provider D queue.
- The router's separate `m_trackedPendingEntries_` field was not used as the Layer-1 predicate value.
- The S7 provider D depth was directly observed as `5`.
- The later diagnostic value `routerPendingRetire=2` is a different temporal/audit sample and does not change the S7 raw queue snapshot.

Verdict:

```text
D-QUEUE-2-PROVEN = NO
S10-NEW-DENTRY = YES
PREEXISTING-ENTRY-CALLER = UNRESOLVED
```

The live capture proves one new D entry and its exact metadata. It does not prove the writer/caller identity of the five entries already resident at S7.

## 10. `pendingReclaimHandles_` Audit

The S7 vector was empty:

- begin `0`
- end `0`
- capacity `0`

The vector mutex-protected predicate therefore passed. No `ReclaimIdentity` could be the S7 timeout cause in this execution.

The source lifecycle remains:

- insertion occurs on reclaim admission/TOCTOU failure paths;
- shutdown retries the vector through the reclaim path;
- no source or live evidence shows a non-empty S7 vector in this run.

## 11. Quarantine / TerminalReclaim Audit

The S7 state showed:

- DSP quarantine resident: `0` in the drain audit/terminal diagnostic
- router Q resident: `0`
- router EmergencyQ resident: `0`
- terminal reclaim resident: `0`
- active readers: `0`
- stuck readers: `0`

The bridge's `quarantineIntentResidencyCount_ = 2` is a transport/intent counter and is not counted as a DSP quarantine resident or as an engine D entry. The bridge's `retireBacklogCount_ = 61` is likewise a coordinator semantic counter, not the five-entry D queue depth.

No direct S7 flag-base capture was retained after the PDB offset correction. This is a provenance limitation, not a basis for converting the zero audit value into a stronger claim.

## 12. Source / Binary Cross-Check

The binary sequence matches the source contract:

1. `releaseResources` enters the recovered function.
2. `waitForDrain(2000,2)` is called with the recovered immediate values.
3. S7 reaches the false return after the bounded loop.
4. The timeout branch calls `markTimedOut` with `Unknown`.
5. The inlined `SnapshotCoordinator::finalizeShutdown(true)` calls `publishEpoch()` once.
6. The non-null current snapshot is exchanged to null and enqueued successfully.
7. The null target is not enqueued.
8. The finalized flag is set.
9. Bridge completion handling sees a non-drained coordinator and does not establish `ShutdownComplete` in the captured state.

No implementation conclusion is drawn from the line-only source breakpoints because the PDB has no line/code symbols.

## 13. M0/M1/M2 Differential Interpretation

- M0 live debugger chain: captured in this gate.
- M1: not run.
- M2: not run.
- Existing FPM evidence remains historical evidence only.

The prior FPM `routerPendingRetire=2` aggregate cannot be substituted for the live S7 provider D depth `5` or the S10 new-entry observation. Run-local queue state, asynchronous draining, and audit timing are separate observations.

## 14. T3a / T3b Terminal Contract Re-validation

- T3a remains inherited as true for the historical FPM runs; this debugger gate did not alter the FPM contract.
- T3b remains false/incomplete for the live chain: `drainedWithinBudget=false`, `isFullyDrained=false`, and no `ShutdownComplete` proof was captured.
- The outer timeout phase is directly proven as `TimedOut` with reason `Unknown`.

No contract repair or implementation change is authorized by this report.

## 15. Root-Cause Localization Verdict

Directly localized:

- S7 timeout return
- S7 provider D depth `5`
- S7 five-entry metadata, including one concrete pointer
- S7 `pendingReclaimHandles_` empty
- S7 no active/stuck readers
- S7 `markTimedOut(Unknown)` reason
- continuous S7-to-S10 execution
- one S10 current-snapshot D entry with exact pointer, deleter, epoch, and type
- current/target exchange and finalized flag transition
- timed-out reclaim skip

Not fully localized:

- exact S7 temporal values of all 14 bridge predicates after correcting the binary/PDB offset drift
- caller/site identity for the five pre-existing D entries
- object byte sizes, which are not represented by `DeletionEntry`
- a proven `ShutdownComplete` transition

## 16. Remaining Unresolved Items

1. A same-point S7 read of the execution-derived bridge base `this + 0x110a480` is still required to promote the 14-predicate table from continuous-state evidence to exact S7 evidence.
2. The five S7 D entries have exact slot metadata but no writer/caller identity.
3. `objectBytes` is unavailable because the live `DeletionEntry` type has no such member.
4. The relationship between the later audit log's `routerPendingRetire=2` and the breakpoint queue snapshots requires a separately instrumented audit-time capture.
5. The PDB's stale `+0x40` late-member offsets must not be reused for a future bridge/phase capture.
6. `ShutdownComplete` was not observed; the captured state is `TimedOut`/`Unknown`, with bridge state not entering the completion path.

## 17. STOP / Next Gate

Gate result:

```text
PRIMARY_VERDICT = PARTIALLY-LOCALIZED
GATE_DISPOSITION = STOP / S7-BRIDGE-TEMPORAL-CAPTURE-INCOMPLETE
D-QUEUE-2-PROVEN = NO
S7-DIRECT-LAYER1 = PARTIAL
S10-NEW-DENTRY = YES
S7-S10-CONTINUITY = YES
SHUTDOWN-COMPLETE = NOT-PROVEN
```

This gate stops without a fix proposal. A future gate must either capture the exact S7 bridge state at the execution-derived base in the same run or explicitly accept the partial-localization boundary.

## Appendix A. Evidence and Tool Ledger

- CDB session: `cdb-4593d9a6`
- CDB executable: `C:\Users\user\AppData\Local\Microsoft\WindowsApps\Microsoft.WinDbg_1.2606.22001.0_x64__8wekyb3d8bbwe\amd64\cdbX64.exe`
- remote target: `tcp:Port=5005,Server=127.0.0.1`
- target command: `C:\VSC_Project\ConvoPeq\build\Release\AudioEngineHarness.exe --fpm-m0`
- symbol path: `C:\VSC_Project\ConvoPeq\build\Release`
- actual image base: `0x00007ff7ae6b0000`
- no source edit, build, Dr. Memory, or M1/M2 execution
- type/layout inspection: CDB `dt`, `dps`, `dq`, `dd`, `db`
- address recovery: `llvm-objdump`, `llvm-readobj --unwind`, `llvm-pdbutil dump --summary`
- source cross-check: `AudioEngine.Threading.cpp`, `AudioEngine.Processing.ReleaseResources.cpp`, `SnapshotCoordinator.h`, `ISRRuntimePublicationCoordinator.cpp`, `ISRRetire.cpp`, `EpochDomain.h`, `DeferredDeletionQueue.h`

## Appendix B. Direct Numeric Snapshot

```text
S7:
  this=0x00000288e5730080
  timeoutMs=2000
  pollIntervalMs=2
  drainedWithinBudget=false
  pendingReclaimHandles={begin=0,end=0,capacity=0}
  DQueue.enqueuePos=5
  DQueue.dequeuePos=0
  DQueue.depth=5
  currentEpoch=9
  minReaderEpoch=9
  activeReaders=0
  stuckReaders=0
  lifetime.enqueueTicket=2
  lifetime.dequeuePos=0
  lifetime.fallbackCount=0
  router.pendingRetireCount=5 (provider D depth)
  router.trackedPendingEntries=0 (separate field; not predicate)
  orchestrator.hasDeferred=false
  terminalReclaimResident=0
  quarantineStores=0
  markTimedOut.reason=9 (Unknown)

S10:
  timedOut=true
  finalize.current=0x00000288ee1dc010
  finalize.target=0
  publishEpoch.returnedRetireEpoch=9
  currentEpoch.after=10
  current.enqueueResult=true
  DQueue.enqueuePos=6
  DQueue.depth=6
  newEntry={ptr=0x00000288ee1dc010,deleter=0x00007ff7b06473a0,epoch=9,type=Generic,sequence=0,generation=0}
  finalize.currentAfter=0
  finalize.targetAfter=0
  m_shutdownFinalized=1
  ShutdownRuntime.phase=8 (TimedOut)
  ShutdownRuntime.lastNonTerminalPhase=7 (VerifyDrained)
  ShutdownRuntime.blockingReason=9 (Unknown)
  bridge state at completion call=0 (Bootstrapping)
```
