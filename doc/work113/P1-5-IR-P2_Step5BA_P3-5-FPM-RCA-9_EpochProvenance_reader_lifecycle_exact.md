# P3-5-FPM-RCA-9 — Epoch Provenance / Reader Lifecycle Exact Capture

- Gate: `P3-5-FPM-RCA-9`
- Mode: `read-only source reconciliation + one bounded M1 attempt`
- Date: 2026-09-24
- Input: `RCA-8 = PARTIALLY-LOCALIZED / STOP`
- Target: `RCA-7 first Engine DQueue reclaim: provenance of minReaderEpoch=9`
- Scope: `Case A/B/C/D separation, EpochDomain/reader lifecycle, shutdown ordering, P9-P22 coordinator predicates, and terminal consequence`
- Verdict: `PARTIALLY-LOCALIZED / STOP`
- Disposition: `NO IMPLEMENTATION / NO M2 / NO RETRY`
- M0 executions: `0`
- M1 executions: `1` (capture aborted before slot evidence)
- M2 executions: `0`
- Build: `not run`
- Dr. Memory: `not run`
- Production/test/CMake/getter/counter/telemetry/shutdown/epoch/reclaim/router/Recovery changes: `0`

## 1. Gate / Scope

RCA-9 asks one question only:

```text
At RCA-7's first Engine DQueue reclaim, which source/runtime lifecycle
established minReaderEpoch=9?
```

The investigation is not a DQueue repair task and does not assume that an active reader stopped shutdown. It separates:

```text
direct cause      = head epoch == minReaderEpoch at the DQueue epoch gate
upstream cause    = why minReaderEpoch was 9 at that observation
terminal result   = residual -> drain false -> timeout -> Unknown
```

A single M1 run was permitted only after source reconciliation remained inconclusive. The run was attempted once and is not repeatable under this gate.

## 2. Latest `ConvoPeq.md` Identity

The identity check matched the RCA-8 freeze:

```text
Path       = ConvoPeq.md
Bytes      = 5535334
SHA-256    = E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
LastWrite  = 2026-09-23T14:33:38.6388122Z
RCA-8 hash = E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
Identity   = MATCH
```

The generated document is the source authority for this gate. The source files below are the line-addressable authority for the contracts.

Relevant source hashes:

```text
src/core/EpochDomain.h
  6EC7F1106FE029F862062A4BDB2C7A3F7A14D109E49CA0294B95E7A41F451E7D
src/core/RCUReader.h
  E70CE078A3B402891E9431223C2C0AE9BEE42808D8B3FB119E2B593DD6BEC8F8
src/DeferredDeletionQueue.h
  91143BB072C4D2B2EAE42A012783EF9F28DA0DA8EA7B67FBF70B56E34248A188
src/audioengine/AudioEngine.Processing.ReleaseResources.cpp
  0675CF5137E5A1F917B86D3C517BBF3767550E64051FF9E1C64873B6BAE77E63
src/audioengine/DSPLifetimeManager.cpp
  7A0F6F406B69E9F2B21820FECA5BADBD2C819850B7A92BE3311525413F9E6982
src/audioengine/ISRRuntimePublicationCoordinator.cpp
  1AD84A0C3F2FFB7E796A43F979B0D05A8827EBA2631E3A69E01894E5A32DD8C0
src/audioengine/ISRRuntimePublicationCoordinator_ProcessIntent.cpp
  4390798F3BCD66B9C3730194E633D56487E113BF9EF03B558F2A9A9CA0B454EA
```

Binary identity used by the one M1 attempt:

```text
EXE SHA-256 = E5C7AFB9C4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75
PDB SHA-256 = A793738F12F9BA3E0C7A19981B35212659107A356C6904C2BF809933CE1F4391
```

## 3. RCA-7 Frozen Evidence

RCA-7 established the direct first-call condition:

```text
globalEpoch       = 9
dequeuePos        = 4
head sequence     = 5
expectedSequence  = 5
entry.epoch       = 9
getMinReaderEpoch = 9
sequence gate     = PASS
epoch gate        = BLOCK
isOlder(9, 9)     = false
dequeue CAS       = NOT REACHED
```

The later M1 control in RCA-7 observed `minReaderEpoch=12` and a successful CAS from `4` to `5`. That control proves the epoch gate and CAS path can pass after epoch progress; it does not identify the S7 reader state.

The RCA-7 reader-slot scan is not used as exact reader identity. RCA-9 preserves that boundary.

## 4. EpochDomain Source Provenance

### 4.1 Current epoch and publication

`EpochDomain` initializes `globalEpoch` to `1` and reader slots to inactive/zero-depth at `src/core/EpochDomain.h:26-36`.

`currentEpoch()` is an acquire load of `globalEpoch` at `src/core/EpochDomain.h:186-190`.

`publishEpoch()` increments `epochGeneration_` and then `globalEpoch` at `src/core/EpochDomain.h:192-202`. It changes the epoch baseline but does not itself reclaim a DQueue entry.

The static Release disassembly of the virtual `getMinReaderEpoch` target reached by the M1 pre-call was:

```text
target RVA = 0x1f9c880
0x1f9c886  mov rax,[rcx]
0x1f9c88c  call [rax+0x28]       ; currentEpoch provider call
0x1f9c88f  lea r8,[rbx+0x1400]
0x1f9c899  lea rdx,[rbx+0x20]    ; readers[0]
0x1f9c89d  lea r10,[rbx+0x1420]  ; readers[64] end
0x1f9c8b0  movzx ecx,[rdx+0x48] ; quarantine flags
0x1f9c8b9  mov eax,[rdx+0x8]    ; depth
0x1f9c8c0  mov rcx,[rdx]        ; slot epoch
0x1f9c8c9  sub rax,r9
0x1f9c8cf  shr rax,3fh
0x1f9c8d5  cmovne r9,rcx        ; lower min only when older
0x1f9c8d9  add rdx,0x50         ; ReaderSlot stride
```

The PDB static type query independently confirmed:

```text
ReaderSlot +0x00 epoch
ReaderSlot +0x08 depth
ReaderSlot +0x10 enterCount
ReaderSlot +0x18 residencyStartTimestampUs
ReaderSlot +0x20 ownerThreadId
ReaderSlot +0x28 ownerTag[32]
ReaderSlot +0x48 quarantineFlags
stride = 0x50
```

This is a static type/disassembly result. It did not launch the target and is not counted as M1.

### 4.2 Epoch passed to reader entry

`enterReader()` reads `currentEpoch()` and publishes it into the slot before incrementing depth at `src/core/EpochDomain.h:117-141`.

Therefore a reader epoch is an entry-time snapshot:

```text
reader enter at globalEpoch=E -> slot.epoch=E, depth becomes >0
```

It is not a later copy of a retirement entry's epoch.

### 4.3 Epoch passed to retirement enqueue

`DSPLifetimeManager::retire()` selects the supplied publication epoch when nonzero, otherwise `router_->currentEpoch()`, then enqueues through `ISRRetireRouter::enqueueWithRetry()` at `src/audioengine/DSPLifetimeManager.cpp:35-77`.

`retireByHandle()` also uses `router_->currentEpoch()` for its enqueue at `src/audioengine/DSPLifetimeManager.cpp:79-137`.

Thus the source has two distinct epoch paths:

```text
reader slot epoch = epoch sampled by enterReader
retirement entry epoch = epoch supplied by retire/enqueue path
```

The source does not prove that the S7 DQueue entry epoch was produced by the reader path or by a retirement enqueue path.

### 4.4 Epoch used by reclaim

`EpochDomain::tryReclaim()` passes the current `getMinReaderEpoch()` into `DeferredDeletionQueue::reclaim()` at `src/core/EpochDomain.h:385-395`.

`ISRRetireRouter::tryReclaim()` delegates to the provider and then drains the quarantine layers at `src/audioengine/ISRRetireRouter.cpp:386-394`.

The DQueue predicate is `isOlder(entry.epoch, minReaderEpoch)`, with FIFO head blocking at `src/DeferredDeletionQueue.h:120-175`.

## 5. Reader Lifecycle Provenance

### 5.1 Registration and reservation

`registerReaderThread()` and `reserveReaderThread()` acquire a slot by changing `epoch` from `kInactiveEpoch` to `kReservedEpoch`, publish depth zero, and record owner metadata at `src/core/EpochDomain.h:39-114`.

A reserved slot is not eligible for `getMinReaderEpoch()` until `enterReader()` publishes a real epoch and increments depth.

### 5.2 RAII enter/exit

`RCUReader::enter()` obtains a slot and calls the provider's `enterReader()` at `src/core/RCUReader.h:36-82`.

`RCUReader::exit()` calls the provider's `exitReader()` on the outer exit at `src/core/RCUReader.h:105-147`.

Nested enter increments the RAII nesting depth without replacing the active slot epoch. The provider's `enterReader()` also does not replace the epoch on nested provider entry at `src/core/EpochDomain.h:129-134`.

### 5.3 Final exit and quarantine

`EpochDomain::exitReader()` clears the slot epoch to `kInactiveEpoch` only on the final depth transition at `src/core/EpochDomain.h:143-184`.

`quarantineReader()` excludes depth-zero slots immediately and marks depth-positive slots pending until final exit at `src/core/EpochDomain.h:271-325`.

`getMinReaderEpoch()` skips quarantined slots and depth-zero slots at `src/core/EpochDomain.h:214-248`.

### 5.4 Case source separation

The source supports these mutually non-exclusive provenance cases:

| case | source condition | RCA-9 evidence |
|---|---|---|
| A | no eligible reader; `minEpoch` remains the `currentEpoch` base | not proven; slot scan absent |
| B | eligible reader with slot epoch equal to the base | not proven; slot scan absent |
| C | slot/lifecycle state changes between pre-call and return | not proven; return not reached |
| D | epoch publication or other non-reader lifecycle supplies the observed baseline | not proven; no global-epoch write chronology |

The static implementation proves how each case would behave, but not which case occurred at S7.

## 6. Shutdown Ordering

The current terminal pass has the following source control flow:

| order | operation | epoch relation | DQueue effect |
|---:|---|---|---|
| 1 | `requestShutdown()` / `closeAdmission()` | no epoch change | stops new admission/producers; no direct DQueue change |
| 2 | `joinProducers()` | no epoch change | producer completion is required before authoritative drain |
| 3 | `advanceRetireEpoch()` at `:263` | `globalEpoch` increments | makes older entries potentially reclaimable |
| 4 | graceful loop `publishEpoch()` + `tryReclaim()` at `:305-306` | epoch advances each poll | may decrement DQueue if head is sequence-ready and epoch-older |
| 5 | `ReclaimComplete` at `:376` | no additional epoch operation | marker for graceful segment, not proof that later enqueues are absent |
| 6 | world clear at `:537-543` | clear path may retire the old world | can enqueue a new EBR entry at current epoch |
| 7 | active/fading terminal disposition at `:499-517,568-588` | `DSPLifetimeManager` uses current epoch | can enqueue new DQueue entries at current epoch |
| 8 | final `waitForDrain(2000,2)` at `:615` | no `publishEpoch()` in the wait loop | only safe reclaim retries; epoch-equal head can remain |
| 9 | `markTimedOut()` / finalization / terminal marks at `:624-745` | terminal state/result fields updated | DQueue residual is not repaired by the phase transition |

The current source has `publishEpoch()` calls in `ReleaseResources.cpp` at lines `305` and `323`, but no later `publishEpoch()` before the final wait. `waitForDrain()` calls `drainDeferredRetireQueues(true)` at `src/audioengine/AudioEngine.Threading.cpp:235-244`, which invokes safe reclaim without advancing the epoch.

The final-wait behavior and the origin of the S7 entry are separate questions:

```text
Source proves that entries enqueued after ReclaimComplete can be held at current epoch
until a later epoch advance.

Source does not prove that the S7 slot-4 entry was enqueued by that later path.
```

RCA-7's S7 breakpoint occurs after the reconfigure-phase message and before the later terminal disposition in the M1 log (`rca9-cdb.log:257-260`). Therefore the S7 head cannot be attributed to the later world-clear/terminal-disposition route from this capture.

## 7. `minReaderEpoch=9` Exact Source

### 7.1 Source truth table

The static `getMinReaderEpoch` implementation initializes `minEpoch` with `currentEpoch()` and only replaces it when a scanned eligible slot is strictly older:

```text
currentEpoch = 9
no eligible slot       -> minEpoch = 9
eligible slot epoch=9  -> isOlder(9,9)=false -> minEpoch = 9
eligible slot epoch<9  -> minEpoch becomes <9
eligible slot epoch>9  -> minEpoch remains 9
```

Therefore the equality observed by RCA-7 is compatible with both A and B. It is not evidence of an active reader by itself.

### 7.2 M1 chronology attempt

The one permitted M1 run was:

```text
C:\VSC_Project\ConvoPeq\tmp\cdb.exe
  -logo C:\Users\user\AppData\Local\Temp\opencode\rca9-cdb.log
  -y C:\VSC_Project\ConvoPeq\build\Release
  -cf C:\Users\user\AppData\Local\Temp\opencode\rca9-cdb-capture.txt
  AudioEngineHarness.exe --fpm-m0
```

The M1 log proves the following partial facts:

```text
S7 globalEpoch                 = 9       (rca9-cdb.log:281-288)
S7 enqueuePos                  = 5
S7 dequeuePos                  = 4
Engine getMin call precondition reached
rbx                            = EpochDomain base + 0x10
getMin virtual target          = AudioEngineHarness+0x1f9c880
```

The log then stopped at the first slot-scan command:

```text
RCA9_ENGINE_SLOT_SCAN_PRE
Bad register error
```

at `rca9-cdb.log:317-332`.

No `RCA9_ENGINE_SLOT_SCAN` data, `RCA9_ENGINE_GETMIN_RETURN`, or `RCA9_ENGINE_RECLAIM_ENTRY` event was produced. The log ended with `RCA9_PROCESS_EXIT` and `quit` at `rca9-cdb.log:333-336`.

This is an instrumentation failure, not evidence that no reader existed. The M1 capture is incomplete and cannot select A, B, C, or D.

### 7.3 Capture boundary

```text
MIN_READER_EPOCH_9_SOURCE = UNKNOWN
MIN_EPOCH_SOURCE           = UNKNOWN
READER_IDENTITY           = UNRESOLVED
CASE_A                    = NOT PROVEN
CASE_B                    = NOT PROVEN
CASE_C                    = NOT PROVEN
CASE_D                    = NOT PROVEN
```

The CDB script and log are preserved as failed-capture evidence. The script is not corrected and the M1 target is not rerun.

## 8. S7 Reader Slot Evidence

### 8.1 Static layout evidence

The PDB type query established the required slot layout and stride:

```text
epoch            +0x00
depth            +0x08
enterCount       +0x10
ownerThreadId    +0x20
quarantineFlags  +0x48
stride            0x50
```

This proves where a valid capture would read each field. It does not provide runtime values.

### 8.2 Runtime evidence

The M1 log contains no valid runtime slot dump. Therefore:

```text
eligible reader count at S7       = UNRESOLVED
slot epoch at S7                  = UNRESOLVED
slot depth at S7                  = UNRESOLVED
slot quarantine state at S7       = UNRESOLVED
reader owner/thread at S7         = UNRESOLVED
pre/post slot transition          = UNRESOLVED
```

The absence of a valid slot dump must not be promoted to Case A.

## 9. RuntimeIntentCoordinator Predicate Mapping

The coordinator predicate is separate from the Engine DQueue predicate. The current source mapping is:

| ID | predicate | source writer(s) | zero transition / consumer | shutdown relevance | RCA-9 status |
|---|---|---|---|---|---|
| P9 | `swapPending_ == false` | `commit()` true/false; `setSwapPending()` | `commit()` clears at `:119`; `isFullyDrained()` reads at `:512`; `markTransitionCommitted()` reads at `:494` | any pending swap blocks coordinator drain; independent of DQueue | source mapped; runtime value not captured |
| P10 | `intentQueue_.sizeApprox() == 0` | `submitObserve()`, `enqueuePublicationIntent()`, `submitQuarantine()` | `processIntent()` pops at `ProcessIntent.cpp:47`; type-specific counters decrement | residual common Intent blocks coordinator drain | source mapped; runtime value not captured |
| P11 | `observeDeferredRing_.size() == 0` | `submitObserve()` fallback push | `drainObserveDeferred()` pops and decrements `pendingIntentCount_` at `ProcessIntent.cpp:91-99` | fallback Observe residual blocks coordinator drain | source mapped; runtime value not captured |
| P12 | `quarantineFallbackQueue_.sizeApprox() == 0` | `submitQuarantine()` fallback push | `processIntent()` pops first and decrements ring residency at `ProcessIntent.cpp:35-40` | quarantine fallback residual blocks drain | source mapped; runtime value not captured |
| P13 | `recoveryIntentQueue_.size() == 0` | `submitRecoveryRequest()`; redrive path | Builder `popRecoveryRequest()`; shutdown `discardRecoveryRequestsOnShutdown()` | shutdown discard is designed to empty it | source mapped; runtime value not captured |
| P14 | `retireBacklogCount_ == 0` | `onRetireAccepted()` through legacy `enqueueRetire()`; test setter | `onRetireConsumed()` has no production caller found; setter/test paths | comments state Layer 1 DQueue measurement is authoritative, not this counter | source mapped; likely vestigial in production, not proven zero at S7 |
| P15 | `publicationBacklogCount_ == 0` | only direct writer found is `setPublicationBacklogCount()` | `isFullyDrained()` reads at `:532` | if nonzero blocks coordinator drain; no DQueue identity | source mapped; runtime value not captured |
| P16 | `publicationIntentResidencyCount_ == 0` | `enqueuePublicationIntent()` reservation at header `:795`; push rollback at `:798` | `processIntent()` Publish pop decrement at `ProcessIntent.cpp:55-63` | pending Publish Intent blocks drain | source mapped; runtime value not captured |
| P17 | `pendingIntentCount_ == 0` | Observe/Recovery/Quarantine reservation-before-push; redrive reservation | common pop, deferred pop, and recovery pop decrement at `ProcessIntent.cpp:36,66,69,95` and coordinator `:1445` | transport/reservation residual blocks drain | source mapped; runtime value not captured |
| P18 | `reclaimInFlightCount_ == 0` | `onReclaimBegin()` at coordinator `:212` | `onReclaimEnd()` at `:224`; called by normal reclaim and AudioEngine drain wrappers | deferred reclaim residual can block; source calls it approximate | source mapped; runtime value not captured |
| P19 | `quarantineIntentResidencyCount_ == 0` | `submitQuarantine()` primary push at `:1504` | primary pop decrement at `ProcessIntent.cpp:64-66`; fallback move subtracts at coordinator `:1516` | primary quarantine transport residual blocks drain | source mapped; runtime value not captured |
| P20 | `quarantineRingResidencyCount_ == 0` | `submitQuarantine()` fallback push at `:1518` | fallback pop decrement at `ProcessIntent.cpp:35-40` | fallback quarantine residual blocks drain | source mapped; runtime value not captured |
| P21 | `!recoveryAdmissionPending_` | durable attach CAS sets true at coordinator `:1387-1390` | Builder settle false or shutdown discard clears at `:1324,1352` | durable Recovery building/pending blocks drain | source mapped; runtime value not captured |
| P22 | `liveLogicalRecoveryObligationCount() == 0` | `RecoveryAdmissionTable::tryInsert()` increments `liveCount_` at header `:461-499` | `resolve()` decrements on terminal CAS at header `:513-535`; shutdown discard resolves all Live obligations | any unresolved logical obligation blocks drain | source mapped; runtime value not captured |

The source mapping proves that P9-P22 are not one predicate and that the Engine DQueue is not itself a P9-P22 field. It does not identify which coordinator predicate was nonzero in the prior `Faulted` observation.

## 10. Engine DQueue -> Drain -> Unknown Chain

The direct chain remains:

```text
RCA-7:
  head sequence 5 = dequeuePos 4 + 1
  head epoch 9 == minReaderEpoch 9
  isOlder(9,9) == false
  epoch gate returns before CAS

Engine DQueue:
  sizeApprox = enqueuePos - dequeuePos
  residual remains while epoch gate is false

AudioEngine:
  isFullyDrained() includes retireDepth == 0
  nonzero DQueue residual makes that term false

waitForDrain:
  retries safe reclaim without advancing epoch
  can time out

ShutdownRuntime:
  markTimedOut(Unknown) stores the reason
  terminal-skip can later expose ShutdownComplete/completed=true

Coordinator:
  its separate P9-P22 predicate may also be false
  exact false subpredicate is not captured here
```

The direct DQueue condition and terminal consequence are fixed. The upstream reader/epoch provenance is not.

## 11. Direct Cause / Upstream Cause / Terminal Consequence

| layer | classification | statement |
|---|---|---|
| direct cause | PROVEN | S7 head is sequence-ready but fails `isOlder(9,9)` at the epoch gate; CAS is not reached |
| upstream cause | UNRESOLVED | source permits no eligible reader, eligible reader at epoch 9, a lifecycle transition, or another epoch baseline; M1 did not obtain slot chronology |
| terminal consequence | PROVEN at contract level | residual can keep AudioEngine drain false, leading to timeout and `Unknown`; `completed=true` is not a drain-success predicate |
| coordinator relation | UNRESOLVED | P9-P22 source domains are mapped, but no exact false term was captured; no same-cause claim is made |
| DQueue corruption | NOT SUPPORTED | sequence is ready and the later RCA-7 control CAS succeeds after epoch progress |

## 12. Proven / Unresolved Matrix

| item | status | reason |
|---|---|---|
| `ConvoPeq.md` identity | PROVEN / MATCH | bytes, timestamp, and SHA-256 match RCA-8 |
| `globalEpoch=9` at S7 | PROVEN | RCA-9 M1 S7 field read |
| `dequeuePos=4`, `enqueuePos=5` at S7 | PROVEN | RCA-9 M1 S7 field reads |
| S7 sequence-ready head | PROVEN by RCA-7 | sequence `5` equals `dequeuePos+1` |
| S7 epoch gate block | PROVEN by RCA-7 | `entry.epoch=9`, minReader `9`, `isOlder=false` |
| S7 CAS not reached | PROVEN by RCA-7 | return before CAS |
| source `getMinReaderEpoch` algorithm | PROVEN | source plus static disassembly |
| ReaderSlot offsets/stride | PROVEN statically | PDB type query; stride `0x50` |
| M1 Engine getMin pre-call reached | PROVEN | `rca9-cdb.log:317-328` |
| M1 slot values | NOT CAPTURED | CDB `.for` address expression failed at `:330` |
| M1 getMin return chronology | NOT CAPTURED | no return event after failure |
| M1 reclaim chronology | NOT CAPTURED | no reclaim entry after failure |
| Case A | NOT PROVEN | no valid slot scan |
| Case B | NOT PROVEN | no valid slot scan |
| Case C | NOT PROVEN | no pre/return comparison |
| Case D | NOT PROVEN | no epoch-write chronology |
| `MIN_READER_EPOCH_9_SOURCE` | UNRESOLVED | A/B/C/D not selected |
| `READER_IDENTITY` | UNRESOLVED | no slot evidence |
| P9-P22 source mapping | PROVEN | current source audit |
| exact coordinator false subpredicate | UNRESOLVED | no runtime predicate capture |
| same-cause vs independent coordinator residual | UNRESOLVED | P9-P22 are separate from DQueue |
| STOP-1 source identity mismatch | NOT TRIGGERED | identity matched |
| STOP-2 exact S7 chronology unavailable | TRIGGERED | M1 aborted before slot scan/return |
| STOP-3 lifecycle/shutdown connection unproven | TRIGGERED | no lifecycle chronology |
| STOP-4 coordinator false source multiple/unknown | TRIGGERED | no P9-P22 runtime values |

## 13. Disposition

```text
RCA9_GATE_DISPOSITION = PARTIALLY-LOCALIZED / STOP
STOP_1_SOURCE_IDENTITY = NOT_TRIGGERED
STOP_2_SLOT_CHRONOLOGY = TRIGGERED
STOP_3_LIFECYCLE_CONNECTION = TRIGGERED
STOP_4_COORDINATOR_SOURCE = TRIGGERED
MIN_READER_EPOCH_9_SOURCE = UNRESOLVED
READER_IDENTITY = UNRESOLVED
CASE_A = NOT_PROVEN
CASE_B = NOT_PROVEN
CASE_C = NOT_PROVEN
CASE_D = NOT_PROVEN
DIRECT_RECLAIM_CAUSE = PROVEN
UPSTREAM_CAUSE = UNRESOLVED
TERMINAL_CONSEQUENCE = PROVEN_AT_CONTRACT_LEVEL
M0 = 0
M1 = 1 OF 1 USED (incomplete capture)
M2 = 0
BUILD = NOT_RUN
DR_MEMORY = 0
IMPLEMENTATION = FORBIDDEN
FURTHER_CAPTURE = FORBIDDEN_UNDER_THIS_GATE
```

RCA-9 does not retry the failed M1, does not use M2, and does not advance to an implementation or a new lifecycle experiment. The failed CDB script and log remain preserved for audit.

## 14. Evidence Index

| evidence | location |
|---|---|
| Source identity | `ConvoPeq.md`, SHA-256 `E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609`, timestamp `2026-09-23T14:33:38.6388122Z` |
| Epoch initialization/registration | `src/core/EpochDomain.h:26-114` |
| Reader enter/exit | `src/core/EpochDomain.h:117-184` |
| Current/publish epoch | `src/core/EpochDomain.h:186-202` |
| `getMinReaderEpoch` | `src/core/EpochDomain.h:214-248` |
| Reader quarantine | `src/core/EpochDomain.h:271-325` |
| Try reclaim / pending count | `src/core/EpochDomain.h:385-433` |
| Reader close registration | `src/core/EpochDomain.h:626-639` |
| RCU enter/exit | `src/core/RCUReader.h:36-181` |
| DQueue reclaim/FIFO gate | `src/DeferredDeletionQueue.h:110-177` |
| DQueue size | `src/DeferredDeletionQueue.h:219-224` |
| Router epoch/reclaim delegation | `src/audioengine/ISRRetireRouter.cpp:161-223,386-394,586-590` |
| DSPLifetimeManager epoch enqueue | `src/audioengine/DSPLifetimeManager.cpp:35-77,79-137` |
| AudioEngine drain predicate | `src/audioengine/AudioEngine.Threading.cpp:153-213` |
| AudioEngine wait loop | `src/audioengine/AudioEngine.Threading.cpp:215-247` |
| Shutdown ordering | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:95-109,250-376,499-588,615-745` |
| Coordinator P9-P22 | `src/audioengine/ISRRuntimePublicationCoordinator.cpp:417-579,611-717,873-1020,1310-1471` |
| Coordinator queue consumers | `src/audioengine/ISRRuntimePublicationCoordinator_ProcessIntent.cpp:10-100` |
| Recovery table P21/P22 | `src/audioengine/ISRRuntimePublicationCoordinator.h:408-539,1079-1142` |
| Shutdown result semantics | `src/audioengine/ISRShutdown.cpp:72-180` |
| Runtime audit semantics | `src/audioengine/RuntimeDrainAudit.h:11-84` |
| Static ReaderSlot type query | CDB `dt convo::EpochDomain::ReaderSlot`, offsets epoch/depth/quarantine and stride `0x50` |
| Static getMin disassembly | `AudioEngineHarness+0x1f9c880-0x1f9c8ea` |
| RCA-7 report | `doc/work113/P1-5-IR-P2_Step5BA_P3-5-FPM-RCA-7_DQueue_head_block_exact.md` |
| RCA-8 report | `doc/work113/P1-5-IR-P2_Step5BA_P3-5-FPM-RCA-8_EpochHold_terminal_contract_reconciliation.md` |
| RCA-9 M1 script | `C:\Users\user\AppData\Local\Temp\opencode\rca9-cdb-capture.txt` |
| RCA-9 M1 log | `C:\Users\user\AppData\Local\Temp\opencode\rca9-cdb.log` |
| M1 S7 field evidence | `rca9-cdb.log:260-288` |
| M1 pre-call and capture failure | `rca9-cdb.log:315-336` |
