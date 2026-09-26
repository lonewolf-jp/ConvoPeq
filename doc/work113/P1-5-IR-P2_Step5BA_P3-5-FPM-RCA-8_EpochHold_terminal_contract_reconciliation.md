# P3-5-FPM-RCA-8 — Epoch Hold / Terminal Contract Reconciliation

- Gate: `P3-5-FPM-RCA-8`
- Mode: `read-only / source and existing-evidence contract reconciliation`
- Date: 2026-09-24
- Input: `RCA-7 = LOCALIZED / STOP`
- Target: `RCA-7 direct stop condition -> P3-5 terminal contract / FPM failure chain`
- Scope: `EpochDomain epoch and reader lifecycle contracts, AudioEngine drain predicate, ShutdownRuntime result semantics, RuntimeIntentCoordinator state, and Unknown timeout classification`
- Verdict: `PARTIALLY-LOCALIZED / STOP`
- Disposition: `NO IMPLEMENTATION / NO M2 CAPTURE`
- M0 executions: `1` (prior RCA-7)
- M1 executions: `1` (prior RCA-7)
- M2 executions: `0`
- Runtime capture: `0` in RCA-8
- Build: `not run`
- Dr. Memory: `not run`
- Production/test/CMake/getter/counter/telemetry/shutdown/epoch/reclaim/router/Recovery changes: `0`

## 1. Gate and Frozen Scope

RCA-7 proved the direct DQueue stop condition. RCA-8 does not reopen the DQueue branch and does not attempt to repair it. The purpose is to place that condition into the P3-5 terminal contract and to separate the direct reclaim stop from the upstream reason that a minimum-reader epoch of `9` was supplied.

This gate uses only:

- the current source tree;
- the current generated `ConvoPeq.md`;
- the RCA-7 M1 log/report;
- existing FPM/R32/R33/R34 reports and evidence.

No live execution is required to establish the source-level truth table or the terminal-state semantics. No M2 capture is performed. The unresolved reader/epoch holder is not converted into an assumed active reader.

## 2. Latest `ConvoPeq.md` Identity

The current generated source document is:

```text
ConvoPeq.md bytes = 5535334
ConvoPeq.md SHA-256 = E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
```

Relevant source identities used for this reconciliation:

```text
src/core/EpochDomain.h
  SHA-256 = 6EC7F1106FE029F862062A4BDB2C7A3F7A14D109E49CA0294B95E7A41F451E7D
src/DeferredDeletionQueue.h
  SHA-256 = 91143BB072C4D2B2EAE42A012783EF9F28DA0DA8EA7B67FBF70B56E34248A188
src/audioengine/ISRRetireRouter.cpp
  SHA-256 = 7731E933A2D69BC30598F2173F106B725A884F9BBA235EB4B3D41004CD85B638
src/audioengine/AudioEngine.Processing.ReleaseResources.cpp
  SHA-256 = 0675CF5137E5A1F917B86D3C517BBF3767550E64051FF9E1C64873B6BAE77E63
src/audioengine/ISRShutdown.cpp
  SHA-256 = 77AECAC3772CB41F9409B5D3D9472EF95DC40FA7872338407F4A2D491FCBF2EA
src/audioengine/RuntimeDrainAudit.h
  SHA-256 = 277B1952B93AB21F4323F8252BDCD2D160E67E2ADCB7E24466DF86D412827B7E
src/audioengine/ISRRuntimePublicationCoordinator.cpp
  SHA-256 = 1AD84A0C3F2FFB7E796A43F979B0D05A8827EBA2631E3A69E01894E5A32DD8C0
```

The generated document contains the current definitions at these locations:

```text
ConvoPeq.md:18579-18586  AudioEngine reader wrappers
ConvoPeq.md:21070+       AudioEngine::isFullyDrained
ConvoPeq.md:31966+       RuntimeIntentCoordinator::markShutdownComplete
ConvoPeq.md:35240+       ShutdownRuntime::markTimedOut
ConvoPeq.md:57133+       EpochDomain definition
```

The source files, not the generated concatenation alone, are the line-addressable authority for the contracts below.

## 3. RCA-7 Evidence Carried Forward

RCA-7 M1 established the following first Engine DQueue call:

```text
globalEpoch       = 9
dequeuePos        = 4
head sequence     = 5
expectedSequence  = dequeuePos + 1 = 5
entry.epoch       = 9
getMinReaderEpoch = 9
sequence gate     = PASS
epoch gate        = BLOCK
isOlder(9, 9)     = false
dequeue CAS       = NOT REACHED
```

The same M1 run later observed:

```text
getMinReaderEpoch = 12
entry.epoch       = 9
epoch gate        = PASS
dequeue CAS       = 4 -> 5, SUCCESS
```

The direct head condition is therefore fixed as:

```text
sequence-ready FIFO head
+ entry.epoch == supplied minReaderEpoch
=> reclaim returns before dequeue CAS
```

RCA-7 also established that the sequence path and CAS path are functional: a later minimum-reader value allowed the same queue head to advance from `4` to `5`. DQueue sequence corruption, dequeue corruption, and a primary CAS failure are not supported as the direct S7 stop cause.

The upstream identity of the `9` is not carried forward as an active-reader fact. The RCA-7 reader-slot scan is explicitly excluded from exact reader identity evidence.

## 4. `EpochDomain` Contract

### 4.1 Global epoch

`EpochDomain` initializes `globalEpoch` to `1` and initializes every reader slot to `kInactiveEpoch` with zero depth at `src/core/EpochDomain.h:26-36`.

`currentEpoch()` is an acquire load of `globalEpoch` at `src/core/EpochDomain.h:186-190`.

`publishEpoch()` increments both `epochGeneration_` and `globalEpoch` with release/acquire-relaxed ordering at `src/core/EpochDomain.h:192-202`.

Therefore:

```text
globalEpoch is the current publication/retirement epoch authority.
publishEpoch() advances it; it does not itself reclaim an entry.
```

### 4.2 Reader registration and reservation

`registerReaderThread()` and `reserveReaderThread()` acquire a slot by changing `epoch` from `kInactiveEpoch` to `kReservedEpoch`, publish zero depth, and record owner metadata at `src/core/EpochDomain.h:39-114`.

A reserved slot is not an active reader. `closeReaderRegistration()` only blocks new registration; already registered slots remain able to enter and exit at `src/core/EpochDomain.h:626-639`.

The shutdown path closes registration before its graceful drain at `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:269-275`. This prevents a new slot from appearing after a zero-reader observation; it does not retroactively identify which slot, if any, held the S7 epoch.

### 4.3 Reader enter and exit

`enterReader()` writes `currentEpoch()` into the slot before incrementing depth at `src/core/EpochDomain.h:117-141`. Nested enter does not replace the outer epoch.

`exitReader()` decrements depth and, on the final exit, clears the slot epoch to `kInactiveEpoch` at `src/core/EpochDomain.h:143-184`. Thus a reader that remains at depth greater than zero continues to expose its entered epoch to `getMinReaderEpoch()`.

`RCUReader::enter()` reserves or reuses a slot and calls the provider's `enterReader()` at `src/core/RCUReader.h:36-82`. `RCUReader::exit()` calls the provider's `exitReader()` on the outer exit at `src/core/RCUReader.h:105-147`. There is no separate production `reader release` primitive in this contract; the release path is the final `exitReader()` / `RCUReader::exit()` transition.

### 4.4 Quarantine

`quarantineReader()` immediately marks a depth-zero slot quarantined, or sets pending quarantine for a depth-positive slot at `src/core/EpochDomain.h:271-325`. A pending slot becomes quarantined on its final exit. `getMinReaderEpoch()` skips quarantined slots.

`unquarantineAllReaders()` clears quarantine flags at shutdown but is not evidence of the S7 slot state; no S7 quarantine identity was captured.

## 5. `getMinReaderEpoch()` Contract

`getMinReaderEpoch()` starts with `currentEpoch()` and scans all reader slots at `src/core/EpochDomain.h:214-248`:

1. skip a quarantined slot;
2. skip a slot with `depth == 0`;
3. skip `kInactiveEpoch` and `kReservedEpoch`;
4. lower the result only when `isOlder(slot.epoch, minEpoch)` is true;
5. return the resulting minimum.

The equality case is decisive:

| S7 state | `isOlder(slotEpoch, 9)` | returned minimum |
|---|---:|---:|
| no eligible reader, `currentEpoch=9` | not evaluated | `9` |
| eligible reader at epoch `9` | `false` | `9` |
| eligible reader at epoch `<9` | `true` | `<9` |
| eligible reader at epoch `>9` | `false` for this ordering | `9` |

Therefore:

```text
globalEpoch == minReaderEpoch == 9
```

does **not** prove that an active reader exists. It is compatible with both:

```text
A. no eligible reader, so minReaderEpoch starts and remains at currentEpoch=9
B. an eligible reader exists at epoch 9, but equality does not lower the result
```

The source contract cannot distinguish A from B without an exact reader-slot observation at the same instant. RCA-7 did not provide that observation, and the post-shutdown `activeReaders=0` audit cannot be retroactively applied to S7.

## 6. Reader Lifecycle and the Unresolved Epoch Holder

The current source provides the lifecycle mechanisms but not the S7 holder identity:

```text
registration/reservation -> slot reserved
RCUReader enter          -> slot epoch=currentEpoch, depth++
nested enter             -> depth++, epoch unchanged
outer RCUReader exit     -> depth--, final exit clears epoch
shutdown close           -> no new registration; existing enter/exit remains possible
quarantine               -> exclude from minReaderEpoch after depth/quiescence contract
```

The following remain unresolved for the RCA-7 value `9`:

- whether S7 had no eligible reader;
- whether an eligible S7 reader held epoch `9`;
- which thread or `RCUReader` instance, if any, held the slot;
- whether a reader lifecycle transition occurred between the S7 field read and the first reclaim call;
- whether the S7 epoch was produced by a pre-existing entry, a current-epoch terminal enqueue, or an earlier publication/retirement path.

The source-level `currentEpoch` / `publishEpoch` contract explains how a later value of `12` can make an entry at epoch `9` reclaimable. It does not identify the source of the earlier value `9`.

## 7. `globalEpoch` versus `minReaderEpoch`

The two values have different meanings:

```text
globalEpoch
  = EpochDomain::globalEpoch
  = current publication/retirement epoch

minReaderEpoch
  = result of getMinReaderEpoch()
  = currentEpoch lowered only by eligible active reader epochs
```

The direct reclaim predicate is:

```text
isOlder(entry.epoch, minReaderEpoch)
= static_cast<int64_t>(entry.epoch - minReaderEpoch) < 0
```

At RCA-7 S7:

```text
entry.epoch - minReaderEpoch = 9 - 9 = 0
```

so the entry is not older than the supplied grace boundary. This is an epoch-hold condition, not evidence of sequence corruption.

The later control transition is consistent with the same contract:

```text
9 - 12 < 0
=> entry epoch 9 is older than minReaderEpoch 12
=> epoch gate passes
```

## 8. P3-5 / FPM Terminal Condition Mapping

There are two drain authorities that must not be collapsed into one predicate.

### 8.1 AudioEngine drain predicate

`AudioEngine::isFullyDrained()` at `src/audioengine/AudioEngine.Threading.cpp:153-213` requires, among other conditions:

```text
pendingReclaimHandles_.empty()
retireDepth == 0
lifetimeRetireIntentPending == 0
ringResident == 0
dspQuarantineResident == 0
retireQuarantineResident == 0
terminalReclaimResident == 0
runtimePublicationBridge_.isFullyDrained()
```

`retireDepth` comes from `ISRRetireRouter::pendingRetireCount()` at `src/audioengine/AudioEngine.Threading.cpp:170-172`. The router delegates to `EpochDomain::pendingRetireCount()` at `src/audioengine/ISRRetireRouter.cpp:586-590`, which returns `DeferredDeletionQueue::sizeApprox()` at `src/core/EpochDomain.h:430-433`.

`DeferredDeletionQueue::sizeApprox()` is `enqueuePos - dequeuePos` at `src/DeferredDeletionQueue.h:219-224`. Therefore the RCA-7 S7 state (`enqueuePos=5`, `dequeuePos=4`) is a direct residual of `1` in the AudioEngine `retireDepth` term and is sufficient to make that term non-zero at the S7 instant.

This proves the DQueue residual-to-`isFullyDrained=false` implication for the AudioEngine predicate. It does not prove that every other AudioEngine predicate was zero at the same instant.

`waitForDrain()` loops while `!isFullyDrained()` and calls `drainDeferredRetireQueues(true)` at `src/audioengine/AudioEngine.Threading.cpp:215-247`.

### 8.2 RuntimeIntentCoordinator drain predicate

`RuntimeIntentCoordinator::ShutdownScheduler::isFullyDrained()` is a separate predicate at `src/audioengine/ISRRuntimePublicationCoordinator.cpp:511-562`. It checks coordinator transport queues, backlog/residency counters, reclaim-in-flight state, quarantine transport, recovery admission, and logical recovery obligations.

`markShutdownComplete()` calls that predicate and sets the coordinator state to `Bootstrapping` only when it is true; otherwise it sets the state to `Faulted` at `src/audioengine/ISRRuntimePublicationCoordinator.cpp:568-579`.

The RCA-7 M1 log contains the observed coordinator `Faulted` diagnostic after `markShutdownComplete`. That proves the coordinator-side predicate was false at that later terminal point, but it does not identify which P9-P22 subpredicate was nonzero and does not by itself prove that the Engine DQueue was the coordinator's sole cause.

### 8.3 ShutdownRuntime result semantics

`ShutdownRuntime::markTimedOut()` stores the supplied blocking reason, saves the last non-terminal phase, and writes `ShutdownPhase::TimedOut` at `src/audioengine/ISRShutdown.cpp:72-112`.

`transitionTo()` permits a forward transition when all skipped states are terminal at `src/audioengine/ISRShutdown.cpp:124-151`. Therefore `TimedOut -> ShutdownComplete` is allowed, and an allowed terminal skip does not increment `transitionViolations_`.

`collectResult()` computes:

```text
completed = (phase == ShutdownComplete)
finalPhase = current phase
blockingReason = independently stored reason
transitionViolations = independently stored counter
```

at `src/audioengine/ISRShutdown.cpp:160-180`.

Consequently, the previously observed combination is internally consistent:

```text
phase == ShutdownComplete
completed == true
blockingReason == Unknown
transitionViolations == 0
```

`completed=true` means the final phase is `ShutdownComplete`; it does not mean that the drain predicate was true. The source does not clear `blockingReason_` when transitioning to `ShutdownComplete`.

The release path selects `Unknown` when `waitForDrain(2000,2)` times out and neither a stuck reader nor an active builder supplies a more specific reason at `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:615-633`. It later calls safe `tryReclaim`, finalizes the snapshot coordinator, marks the coordinator complete/faulted, and transitions the ShutdownRuntime phase at `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:654-745`.

### 8.4 `Unknown` is not an epoch-holder identity

`RuntimeDrainAudit::getPrimaryBlockingReason()` returns `Unknown` when none of its enumerated audit terms matches at `src/audioengine/RuntimeDrainAudit.h:52-74`. `RuntimeDrainAudit::isAllZero()` is explicitly audit-only and is not the shutdown authority at `src/audioengine/RuntimeDrainAudit.h:76-84`.

Therefore:

```text
blockingReason == Unknown
```

means that the terminal result did not carry a more specific selected reason. It does not mean:

- no DQueue residual existed;
- no reader existed at S7;
- no epoch hold existed;
- the coordinator-side predicate was true.

### 8.5 Reclaimable source path during the drain loop

The graceful drain publishes the epoch and calls `tryReclaim()` on each poll at `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:279-311`. The terminal pass then reaches `ReclaimComplete` at line 376. After that point, the world clear and active/fading DSP terminal disposition can enqueue new EBR entries at `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:537-588`, before the final `waitForDrain(2000,2)` at line 615.

The current source has no `publishEpoch()` call after the graceful-drain `ReclaimComplete` path and before that final wait; the only `publishEpoch()` calls in this file are at lines 305 and 323. `waitForDrain()` itself only calls `drainDeferredRetireQueues(true)`, which invokes safe reclaim without advancing the epoch.

This creates a source-level route by which a newly enqueued current-epoch entry can remain unreclaimable during the final wait when `minReaderEpoch` stays equal to that current epoch. This route is consistent with the RCA-7 direct condition, but it is not proof that the particular S7 slot 4 was created by this later terminal disposition.

## 9. Failure-Chain Reconciliation

| link | status | basis / boundary |
|---|---|---|
| FPM terminal timeout occurred | PROVEN by prior terminal evidence | `blockingReason=Unknown` follows the timeout path |
| `phase == ShutdownComplete` | PROVEN by prior terminal evidence | terminal-skip transition is allowed |
| `completed == true` | PROVEN by prior terminal evidence | `collectResult.completed` is phase-only |
| `transitionViolations == 0` | PROVEN by prior terminal evidence | terminal-skip path does not increment violations |
| `blockingReason == Unknown` | PROVEN by prior terminal evidence | `markTimedOut(Unknown)` selection when no specific reason |
| Engine DQueue residual | PROVEN at RCA-7 S7 | `enqueuePos=5`, `dequeuePos=4` |
| S7 sequence gate pass | PROVEN by RCA-7 M1 | sequence `5` equals `dequeuePos+1` |
| S7 epoch gate block | PROVEN by RCA-7 M1 | `entry.epoch=9`, minReader `9`, `isOlder=false` |
| S7 CAS not reached | PROVEN by RCA-7 M1 | return occurs before dequeue CAS |
| later epoch progress permits CAS | PROVEN by RCA-7 M1 | minReader `12`, CAS `4 -> 5` |
| AudioEngine residual implies its drain predicate false | SOURCE-PROVEN | `retireDepth` maps to DQueue `sizeApprox()` |
| exact S7 `isFullyDrained` subpredicate set | PARTIAL | DQueue term is proven; other terms are not all same-instant observed |
| coordinator `isFullyDrained` false | OBSERVED, subpredicate unresolved | later coordinator `Faulted` state |
| `Unknown` identifies the epoch holder | REJECTED | Unknown is a terminal classification, not reader provenance |
| no eligible S7 reader | POSSIBLE, not proven | currentEpoch initializes minReader to `9` |
| eligible S7 reader at epoch `9` | POSSIBLE, not proven | equality does not lower minReader |
| reader identity / lifecycle that held `9` | UNRESOLVED | exact slot evidence absent |
| S7 slot 4 produced by terminal disposition | UNRESOLVED | later source route exists, but S7 predates that route |
| DQueue corruption as direct cause | NOT SUPPORTED | sequence ready and later CAS success |

The fixed chain is therefore:

```text
RCA-7 direct stop:
  Engine DQueue head
    -> sequence-ready
    -> epoch 9 == minReader 9
    -> epoch gate returns before CAS
    -> residual remains

P3-5 terminal consequence:
  residual retire depth / drain predicate remains false
    -> waitForDrain can time out
    -> markTimedOut(Unknown)
    -> later terminal-skip can still report ShutdownComplete/completed=true

Upstream cause:
  why minReaderEpoch was 9 at S7
    -> UNRESOLVED
```

## 10. Whether Live Capture Is Necessary

For this RCA-8 contract reconciliation, live capture is **not necessary**. The source and existing evidence are sufficient to establish:

- the two meanings of `globalEpoch` and `minReaderEpoch`;
- the no-eligible-reader versus epoch-9-reader ambiguity;
- the reader registration/enter/exit/quarantine lifecycle;
- the DQueue residual-to-AudioEngine-drain-predicate mapping;
- the separation between AudioEngine and RuntimeIntentCoordinator drain predicates;
- the `Unknown` / `ShutdownComplete` / `completed=true` terminal semantics;
- the direct RCA-7-to-terminal chain and its unresolved upstream boundary.

A future exact reader/epoch provenance investigation would require a separately authorized capture or additional observability. It is not started here, and it is not justified as an M2 execution within RCA-8.

The existing `tgrep` index is stale and was not used as authority. The `ccc` semantic query returned no result for the focused query; neither tool overrides the current source reads. Serena initialization timed out in this environment. The source, AiDex index, graphify graph, Semble result, and direct file reads were sufficient for the reconciliation.

## 11. Disposition

```text
RCA8_GATE_DISPOSITION = PARTIALLY-LOCALIZED / STOP
DIRECT_RECLAIM_STOP = PROVEN
FPM_TERMINAL_CHAIN = PROVEN UP TO UNKNOWN TERMINAL REASON
MIN_READER_9_HOLDER = UNRESOLVED
READER_IDENTITY = UNRESOLVED
EXACT_S7_DRAIN_SUBPREDICATES = PARTIAL
COORDINATOR_FALSE_SUBPREDICATE = UNRESOLVED
DQUEUE_CORRUPTION_CAUSE = NOT SUPPORTED
IMPLEMENTATION = FORBIDDEN
M0 = 1 OF 1 USED
M1 = 1 OF 1 USED
M2 = 0
RUNTIME_CAPTURE_IN_RCA8 = 0
DR_MEMORY = 0
RCA-9 = NOT STARTED
```

RCA-8 stops at this boundary. It does not authorize a production fix, counter/getter addition, queue repair, reader lifecycle change, M2 capture, or automatic transition to RCA-9.

## 12. Evidence Index

| evidence | location |
|---|---|
| Generated source identity | `ConvoPeq.md` SHA-256 `E5E742...EF3609` |
| Generated EpochDomain definition | `ConvoPeq.md:57133+` |
| Generated reader wrappers | `ConvoPeq.md:18579-18586` |
| Generated AudioEngine drain predicate | `ConvoPeq.md:21070+` |
| Generated coordinator terminal wrapper | `ConvoPeq.md:31966+` |
| Generated ShutdownRuntime timeout | `ConvoPeq.md:35240+` |
| Epoch initialization and reader registration | `src/core/EpochDomain.h:26-114` |
| Reader enter/exit lifecycle | `src/core/EpochDomain.h:117-184` |
| Epoch publication/current contract | `src/core/EpochDomain.h:186-202` |
| `getMinReaderEpoch()` implementation | `src/core/EpochDomain.h:214-248` |
| Reader quarantine lifecycle | `src/core/EpochDomain.h:271-325` |
| `tryReclaim()` and pending count | `src/core/EpochDomain.h:386-433` |
| RCU reader enter/exit and slot reuse | `src/core/RCUReader.h:36-181` |
| Router epoch delegation | `src/audioengine/ISRRetireRouter.cpp:161-223` |
| Router reclaim and pending count | `src/audioengine/ISRRetireRouter.cpp:386-394,586-590` |
| DQueue reclaim and FIFO gate | `src/DeferredDeletionQueue.h:110-177` |
| DQueue residual calculation | `src/DeferredDeletionQueue.h:219-224` |
| AudioEngine drain predicate | `src/audioengine/AudioEngine.Threading.cpp:153-213` |
| AudioEngine wait loop | `src/audioengine/AudioEngine.Threading.cpp:215-247` |
| ReleaseResources epoch/reclaim ordering | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:262-376` |
| Final wait, timeout, terminal completion | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:615-745` |
| ShutdownRuntime timeout and terminal skip | `src/audioengine/ISRShutdown.cpp:72-180` |
| ShutdownRuntime result/state fields | `src/audioengine/ISRShutdown.h:148-158,325-346` |
| Coordinator drain predicate and Faulted transition | `src/audioengine/ISRRuntimePublicationCoordinator.cpp:511-579` |
| Audit-only classification boundary | `src/audioengine/RuntimeDrainAudit.h:11-84` |
| Prior source reconciliation | `doc/work113/P1-5-IR-P2_Step5AO_P3-5-R33-A_shutdown_drain_retire_reclaim_cause_isolation.md` |
| Prior terminal semantics evidence | `doc/work113/P1-5-IR-P2_Step5AN_P3-5-R32_recovery_origin_terminal_observation_build_run_gate.md` |
| Prior FPM contract definition | `doc/work113/P1-5-IR-P2_Step5AR_P3-5-FPM-Prep1_full_pipeline_measurement_preparation.md` |
| RCA-7 direct branch report | `doc/work113/P1-5-IR-P2_Step5BA_P3-5-FPM-RCA-7_DQueue_head_block_exact.md` |
| RCA-7 M1 log | `C:\Users\user\AppData\Local\Temp\opencode\rca7-cdb.log` |
