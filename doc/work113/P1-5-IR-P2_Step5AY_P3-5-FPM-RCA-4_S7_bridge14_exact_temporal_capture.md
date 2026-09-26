# P3-5-FPM-RCA-4 — S7 Bridge-14 Exact Temporal Capture

- Gate: `P3-5-FPM-RCA-4`
- Mode: read-only M0 live debugger capture
- Date: 2026-09-24
- Narrow-gate verdict: `LOCALIZED`
- Gate disposition: `STOP / NO IMPLEMENTATION`
- `S7-BRIDGE-BASE`: `PROVEN`
- `S7-BRIDGE-14`: `PROVEN`
- `S7-BRIDGE-isFullyDrained`: `FALSE`
- `S7-LAYER1-FULL-REPLAY`: `PARTIAL / RCA-3-CARRY-FORWARD`
- `S10-PROVENANCE`: `CARRIED FROM RCA-3`
- `D-QUEUE-2-PROVEN`: `NO`
- Production/test/CMake/getter/counter/telemetry/shutdown/epoch/reclaim/router/Recovery changes: `0`
- Build: `not run`
- Dr. Memory: `not run`
- M1/M2: `0`

## 1. Scope / State Freeze

This gate performs one read-only M0 live-debug capture at the S7 false-return point. It does not implement a fix and does not attempt to force `ShutdownComplete`.

Frozen:

- production source changes: 0
- test source changes: 0
- CMake changes: 0
- getter/counter additions: 0
- telemetry changes: 0
- shutdown semantics changes: 0
- epoch/reclaim changes: 0
- router/Recovery changes: 0
- binary changes: 0
- M0 live executions for this gate: 1
- M1/M2 executions: 0
- build: not run
- Dr. Memory: not run

The old RCA-3 CDB server and its already-terminated debuggee were removed before this gate so that the M0 run and port were not duplicated. The RCA-4 capture itself used one new M0 execution and stopped at S7.

## 2. `ConvoPeq.md` Reconciliation

The user clarification remains in force: `ConvoPeq(3).md` means `C:\VSC_Project\ConvoPeq\ConvoPeq.md`.

Observed authority:

- Path: `ConvoPeq.md`
- Size: `5,535,334` bytes
- Lines: `121,294`
- SHA-256: `E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609`

The Bridge predicate contract is taken from:

- `src/audioengine/ISRRuntimePublicationCoordinator.cpp:511-562`
- `src/audioengine/ISRRuntimePublicationCoordinator.cpp:568-579`
- `src/audioengine/ISRRuntimePublicationCoordinator.h:943-1120`

The RCA-3 source and binary evidence is carried forward without re-investigating the already-proven D-entry, timeout-reason, or S10 enqueue facts.

## 3. RCA-3 Carry-forward Evidence

The following RCA-3 observations are reused as prior evidence:

- S7 `isFullyDrained() == false`
- S7 engine D queue depth: `5`
- S7 `pendingReclaimHandles_`: empty
- S7 `lifetimeRetireIntentPending`: `2`
- active/stuck readers: `0`
- `markTimedOut(Unknown)`: directly observed
- S10 `publishEpoch()` followed by one current-snapshot D entry: directly observed
- S7-to-S10 continuity: proven
- S10 entry metadata: `ptr=0x00000288ee1dc010`, `deleter=0x00007ff7b06473a0`, `epoch=9`, `type=Generic`, `publicationSequenceId=0`, `generation=0`

The present gate does not reinterpret the five existing S7 D entries as two entries and does not attribute them to a caller.

## 4. Runtime Address / Offset Authority

The PDB has no code-debug stream, so no function/line symbol was used for the Bridge address. The runtime address was derived from the S7 object and the execution-confirmed member base.

RCA-4 runtime values:

- `AudioEngine *this`: `0x000001ca36356080`
- actual image base: `0x00007ff7ae6b0000`
- S7 RIP: `0x00007ff7b06580b3`
- S7 instruction: `xor al,al`
- actual Bridge base: `this + 0x110a480 = 0x000001ca37460500`
- S7 `ShutdownRuntime` base: `this + 0x15462c0 = 0x000001ca3789c340`
- S7 epoch-domain base: `this + 0x10b76c0 = 0x000001ca3740d740`
- S7 D queue base: `epoch-domain + 0x1440 = 0x000001ca37422b80`

The old PDB-reported `+0x40` late-member offset was not used for the Bridge or ShutdownRuntime reads. All Bridge values below use the runtime base `0x000001ca37460500`.

The capture command file was:

`C:\Users\user\AppData\Local\Temp\opencode\rca4-cdb-capture.txt`

The CDB log was:

`C:\Users\user\AppData\Local\Temp\opencode\rca4-cdb.log`

CDB engine:

- `C:\VSC_Project\ConvoPeq\tmp\cdb.exe`
- version `10.0.29617.1000`

The MCP-WinDbg tool surface was not exposed in this runtime; the same CDB engine was driven directly with `-cf` in read-only mode. No source or binary was modified.

## 5. S7 Exact Layer-1 Snapshot

The S7 breakpoint was reached before the `xor al,al` false-return instruction. Registers included:

- `rbx = this = 0x000001ca36356080`
- `r12 = 1`
- `r13 = 0`
- `rax = 1` immediately before the return-value clear

The following Layer-1 values were read at the same stop:

| Layer-1 item | S7 value | interpretation |
|---|---:|---|
| `pendingReclaimHandles_.empty()` | `true` | vector begin/end/capacity all `0` |
| engine D queue `enqueuePos` | `5` | provider D depth component |
| engine D queue `dequeuePos` | `0` | provider D depth component |
| `retireDepth` | `5` | `ISRRetireRouter::pendingRetireCount()` delegates to provider D depth |
| `lifetime.enqueueTicket` | `2` | System-1 pending component |
| `lifetime.dequeuePos` | `0` | System-1 consumed component |
| `lifetime.fallbackCount` | `0` | fallback component |
| `lifetimeRetireIntentPending` | `2` | `2 - 0 + 0` |
| `currentEpoch` | `9` | `EpochDomain::globalEpoch` |
| DSP quarantine active flags | all `0` across 256 bytes | direct flag-array read at actual `this+0x1546000` |
| outer `ShutdownRuntime::phase_` | `7` | `VerifyDrained` |
| outer `ShutdownRuntime::blockingReason_` | `0` | `None` before `markTimedOut` |

RCA-3 carry-forward values remain:

- `hasDeferredCommit = false`
- `ringResident = 0`
- router Q/EmergencyQ resident = `0`
- terminal reclaim resident = `0`
- active/stuck readers = `0`

The directly re-read fields above are sufficient to show that S7 had both provider D depth `5` and LifetimeState pending-intent count `2` false conditions. The full nine-item replay is marked partial because the optional nested values were not all reissued in this narrowed script.

## 6. S7 Exact Bridge-14 Snapshot

All values in this section were read at the same S7 stop and the same runtime Bridge base `0x000001ca37460500`.

The Bridge `state_` byte at `base+0x65` was `0` (`Bootstrapping`).

| # | predicate | raw S7 value | result |
|---:|---|---:|---|
| 1 | `swapPending_` | `0` | PASS |
| 2 | `intentQueue_.sizeApprox()` | `2` (`enqueuePos=2`, `dequeuePos=0`) | FAIL |
| 3 | `observeDeferredRing_.size()` | `0` | PASS |
| 4 | `quarantineFallbackQueue_.sizeApprox()` | `0` | PASS |
| 5 | `recoveryIntentQueue_.size()` | `0` | PASS |
| 6 | `retireBacklogCount_` | `61` | FAIL |
| 7 | `publicationBacklogCount_` | `0` | PASS |
| 8 | `publicationIntentResidencyCount_` | `0` | PASS |
| 9 | `pendingIntentCount_` | `0` | PASS |
| 10 | `reclaimInFlightCount_` | `0` | PASS |
| 11 | `quarantineIntentResidencyCount_` | `2` | FAIL |
| 12 | `quarantineRingResidencyCount_` | `0` | PASS |
| 13 | `recoveryAdmissionPending_` | `0` | PASS |
| 14 | `liveLogicalRecoveryObligationCount()` | `0` | PASS |

The three false Bridge predicates are therefore exactly:

1. `intentQueue_.sizeApprox() == 2`
2. `retireBacklogCount_ == 61`
3. `quarantineIntentResidencyCount_ == 2`

No other Bridge predicate was false in the same-point capture.

## 7. Bridge `isFullyDrained()` Result

`RuntimeIntentCoordinator::ShutdownScheduler::isFullyDrained()` is an AND of the 14 conditions in `ISRRuntimePublicationCoordinator.cpp:511-562`.

Because predicates 2, 6, and 11 are non-zero/false, the exact S7 result is:

```text
runtimePublicationBridge_.isFullyDrained() = false
```

This is a same-point result, not a later S10 inference. The result is independent of the separate outer Layer-1 `AudioEngine::isFullyDrained()` result, which was also false at S7.

## 8. Layer-1 × Bridge Correlation

The S7 state separates the two layers:

```text
Layer-1:
  retireDepth = 5                         -> false
  lifetimeRetireIntentPending = 2         -> false
  pendingReclaimHandles_.empty() = true
  DSP quarantine flags = 0
  shutdown phase = VerifyDrained (7)

Bridge:
  intentQueue size = 2                    -> false
  retireBacklogCount = 61                 -> false
  quarantineIntentResidency = 2           -> false
  remaining 11 Bridge predicates          -> true

Both layers therefore independently prevent a drained result.
```

The `retireBacklogCount_ = 61` value is a coordinator semantic counter. It is not substituted for the engine D queue depth `5`, and it is not treated as a D entry count.

The `intentQueue size = 2` and `quarantineIntentResidency = 2` values are Bridge transport/intent state. They are not substituted for the D queue's five entries.

## 9. D-Queue Evidence Reuse

The S7 D queue depth was re-read as `5`, matching the RCA-3 carry-forward value. The five-entry metadata and their unresolved caller identities are not repeated as a new provenance claim.

The already-proven S10 operation remains:

- `publishEpoch()` returned retire epoch `9`
- current snapshot `0x00000288ee1dc010` was exchanged to null
- one `Generic` D entry was enqueued successfully
- D depth changed from `5` to `6`
- target was null and produced no second finalize entry

RCA-4 does not modify that conclusion.

## 10. S7 Root Blocking Predicate

The exact direct blocking set observed at S7 is:

```text
Layer-1:
  retireDepth = 5
  lifetimeRetireIntentPending = 2

Bridge:
  intentQueue_.sizeApprox() = 2
  retireBacklogCount_ = 61
  quarantineIntentResidencyCount_ = 2
```

`pendingReclaimHandles_` was empty and is excluded from the blocking set for this M0 capture.

The S7 outer `ShutdownRuntime` state was `VerifyDrained` with `blockingReason=None`; RCA-3 directly proved that the subsequent timeout path passed `Unknown` to `markTimedOut`. This gate does not reinterpret `Unknown` as a new root cause.

## 11. D-QUEUE-2-PROVEN Status

The exact Bridge result does not identify the writer/caller of the five pre-existing D entries.

Therefore:

```text
D-QUEUE-2-PROVEN = NO
S7-DQUEUE-DEPTH = 5
S10-NEW-CURRENT-SNAPSHOT-ENTRY = PROVEN
```

No `retireBacklogCount_ == 61` value is treated as a repair target or as proof of a D-queue writer.

## 12. RCA Localization Verdict

Narrow RCA-4 verdict:

```text
RCA-4-BRIDGE-VERDICT = LOCALIZED
S7-BRIDGE-BASE = PROVEN
S7-BRIDGE-14 = PROVEN
S7-BRIDGE-isFullyDrained = FALSE
S7-BRIDGE-FALSE-PREDICATES = (2, 6, 11)
D-QUEUE-2-PROVEN = NO
```

The requested objective is complete: the Bridge base was obtained from the live S7 `this` value, all 14 predicates were read at that same stop, and the false Bridge terms were identified.

The optional full nine-predicate Layer-1 replay is not claimed as a fresh all-fields capture. RCA-3 values are explicitly carried forward where noted, and the exact same-point fields re-read in RCA-4 are separated from carry-forward values in Section 5.

## 13. Remaining Unresolved Items

1. The caller/site identity of the five pre-existing D entries remains unresolved.
2. `D-QUEUE-2-PROVEN` remains `NO` by design.
3. The optional full nested Layer-1 replay for `hasDeferredCommit`, overflow-ring residency, router Q/EmergencyQ, and TerminalReclaim was not repeated in the narrowed RCA-4 script; those values are carried forward from RCA-3.
4. No `ShutdownComplete` transition was attempted or required by RCA-4.
5. The later audit-time `routerPendingRetire=2` log sample is not substituted for the exact S7 Bridge `intentQueue size=2` or D depth `5`.

## 14. STOP / Next Gate

RCA-4 stops at evidence localization. No fix is proposed for `retireBacklogCount_ == 61`, `intentQueue == 2`, `quarantineIntentResidency == 2`, or any D entry.

The resulting causal state is:

```text
S7
├─ Layer-1 false: retireDepth=5, lifetimePending=2
├─ Bridge-14 false: intentQueue=2, retireBacklog=61, quarantineIntent=2
└─ DQueue existing entries: depth=5
        |
        v
S10
└─ finalizeShutdown(true)
   └─ one current-snapshot D entry added
```

No implementation, build, Dr. Memory, M1, or M2 work is authorized by this gate.

## 15. Evidence Ledger

- Release executable: `C:\VSC_Project\ConvoPeq\build\Release\AudioEngineHarness.exe`
- Release PDB: `C:\VSC_Project\ConvoPeq\build\Release\AudioEngineHarness.pdb`
- Target argument: `--fpm-m0`
- CDB executable: `C:\VSC_Project\ConvoPeq\tmp\cdb.exe`
- CDB version: `10.0.29617.1000`
- S7 address: `AudioEngineHarness+0x1fa80b3`
- actual image base: `0x00007ff7ae6b0000`
- exact Bridge base expression: `this+0x110a480`
- exact Bridge base value: `0x000001ca37460500`
- CDB log: `C:\Users\user\AppData\Local\Temp\opencode\rca4-cdb.log`
- CDB capture script: `C:\Users\user\AppData\Local\Temp\opencode\rca4-cdb-capture.txt`
- Headroom doctor: proxy running at `http://127.0.0.1:8787`; client routing warnings unchanged
- RTK: `0.49.0`; WSL path verified
- context-mode: used for source/tool output reduction
- AiDex: project index active and session initialized
- Serena: initialization and relevant memory/tool workflow completed
- Source/build/FPM/M1/M2/Dr. Memory changes: none

## Appendix A. Raw S7 Register Snapshot

```text
S7 RIP       = 0x00007ff7b06580b3
instruction  = xor al,al
rbx         = 0x000001ca36356080
r12         = 1
r13         = 0
rax_before  = 1
this        = 0x000001ca36356080
bridge      = 0x000001ca37460500
```

## Appendix B. Raw S7 Bridge Memory

```text
bridge+0x20 retireBacklogCount_              = 61
bridge+0x28 publicationBacklogCount_         = 0
bridge+0x30 publicationIntentResidencyCount_ = 0
bridge+0x38 pendingIntentCount_              = 0
bridge+0x40 reclaimInFlightCount_             = 0
bridge+0x48 quarantineIntentResidencyCount_  = 2
bridge+0x50 quarantineRingResidencyCount_    = 0
bridge+0x64 swapPending_                     = 0
bridge+0x65 state_                           = 0 (Bootstrapping)
intentQueue enqueue/dequeue                  = 2/0
observeDeferredRing write/read               = 0/0
quarantineFallbackQueue enqueue/dequeue      = 0/0
recoveryIntentQueue write/read               = 0/0
recoveryAdmissionPending_                    = 0
liveLogicalRecoveryObligationCount_          = 0
```
