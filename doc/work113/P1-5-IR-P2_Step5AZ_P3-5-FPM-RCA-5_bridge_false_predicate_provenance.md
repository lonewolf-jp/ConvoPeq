# P3-5-FPM-RCA-5 — Bridge False-Predicate Provenance

- Gate: `P3-5-FPM-RCA-5`
- Mode: read-only source audit plus one M0 live-debug capture
- Date: 2026-09-24
- Verdict: `PARTIALLY-LOCALIZED / STOP`
- Disposition: `NO IMPLEMENTATION`
- M0 executions: `1`
- M1/M2 executions: `0`
- Build: `not run`
- Dr. Memory: `not run`
- Production/test/CMake/getter/counter/telemetry/shutdown/epoch/reclaim/router/Recovery changes: `0`
- D-Queue follow-up: `carried forward only; not reopened`

## 1. Objective and Frozen Scope

RCA-5 traces the three directly evidenced non-zero Bridge terms at the S7 false-return point and correlates them with the current source writer/consumer graph. It does not repair a counter, alter a queue, rebuild, or attempt `ShutdownComplete`.

The one permitted execution was the existing Release target with `--fpm-m0`. The S7 breakpoint was `AudioEngineHarness+0x1fa80b3`, immediately before the `xor al,al` false-return instruction.

The RCA-3/RCA-4 D-queue facts remain carry-forward evidence. RCA-5 does not attribute any D-queue entry to a caller.

## 2. Important Correction to the RCA-4 Interpretation

RCA-4's raw memory reads remain useful, but its semantic labels for the fields after `+0x30` were not valid for the binary that produced the capture. The binary contains the newer `coordinatorTakeCount_` member, while the PDB type stream used for the earlier interpretation did not contain that member.

The current source declares the inserted member at `src/audioengine/ISRRuntimePublicationCoordinator.h:971-987`:

- `publicationIntentResidencyCount_` is at binary `+0x30`.
- `coordinatorTakeCount_` is at binary `+0x38`.
- `pendingIntentCount_` is at binary `+0x40`.
- `reclaimInFlightCount_` is at binary `+0x48`.
- `quarantineIntentResidencyCount_` is at binary `+0x50`.
- `quarantineRingResidencyCount_` is at binary `+0x58`.

The binary's own process-intent code confirms this map: a type-1 Publish pop decrements `+0x30` and increments `+0x38` when the recovery obligation is zero; a type-3 Quarantine pop decrements `+0x50` and `+0x40`; the fallback pop decrements `+0x40` and `+0x58`.

Therefore the RCA-4 value previously labeled `quarantineIntentResidencyCount_=2` is, in this binary, `coordinatorTakeCount_=2`. The actual primary quarantine-intent lane is zero. The directly evidenced quarantine-related non-zero term is `quarantineRingResidencyCount_=61` at `+0x58`.

The correct binary predicate map is:

| # | Predicate | Binary read | M0 raw value | RCA-5 result |
|---:|---|---:|---:|---|
| 1 | `swapPending_` | `B+0x6c` | not directly captured | not reclassified |
| 2 | `intentQueue_.sizeApprox()` | positions `B+0x35cc40/B+0x35cc80` | `2/0` | `FAIL` |
| 3 | `observeDeferredRing_.size()` | `B+0x44180/B+0x441c0` | `0/0` in prior same-stop capture | `PASS` carry-forward |
| 4 | `quarantineFallbackQueue_.sizeApprox()` | `B+0x415d00/B+0x415d40` | not directly reissued | not reclassified |
| 5 | `recoveryIntentQueue_.size()` | `B+0x72200/B+0x72240` | `0/0` in prior same-stop capture | `PASS` carry-forward |
| 6 | `retireBacklogCount_` | `B+0x20` | `61` | `FAIL` |
| 7 | `publicationBacklogCount_` | `B+0x28` | `0` | `PASS` |
| 8 | `publicationIntentResidencyCount_` | `B+0x30` | `0` | `PASS` |
| 9 | `pendingIntentCount_` | `B+0x40` | `0` | `PASS` |
| 10 | `reclaimInFlightCount_` | `B+0x48` | `0` | `PASS` |
| 11 | `quarantineIntentResidencyCount_` | `B+0x50` | `0` | `PASS` |
| 12 | `quarantineRingResidencyCount_` | `B+0x58` | `61` | `FAIL` |
| 13 | `recoveryAdmissionPending_` | `B+0x72590` | `0` in prior same-stop capture | `PASS` carry-forward |
| 14 | `liveLogicalRecoveryObligationCount()` | `B+0x78ba0` | `0` in prior same-stop capture | `PASS` carry-forward |

The RCA-5 script read the old `B+0x64` state bytes; those bytes are not the binary's `swapPending_` and `state_` fields. They are not used for the corrected predicate table.

The directly evidenced blocking terms are therefore:

```text
intentQueue_.sizeApprox() = 2
retireBacklogCount_ = 61
quarantineRingResidencyCount_ = 61
```

The correct `swapPending_` byte was not reissued in the one-shot RCA-5 script. RCA-5 does not claim a freshly re-read complete 14-value table beyond the direct and explicitly carried-forward values above.

## 3. M0 Capture

Target:

- Executable: `build/Release/AudioEngineHarness.exe`
- PDB: `build/Release/AudioEngineHarness.pdb`
- Argument: `--fpm-m0`
- S7 RIP: `0x00007ff7b06580b3`
- `this` (`rbx`): `0x00000266a8bd9080`
- Bridge base `this+0x110a480`: `0x00000266a9ce3500`

The capture script and log are:

- `C:\Users\user\AppData\Local\Temp\opencode\rca5-cdb-capture.txt`
- `C:\Users\user\AppData\Local\Temp\opencode\rca5-cdb.log`

The direct counter read was:

```text
B+0x20 = 0x3d = 61
B+0x28 = 0
B+0x30 = 0
B+0x38 = 2
B+0x40 = 0
B+0x48 = 0
B+0x50 = 0
B+0x58 = 0x3d = 61
B+0x60 = 0
B+0x68 = 0x500
```

`B+0x68` is not used as `retireAuthorityCount_`; the binary writer disassembly places `retireAuthorityCount_` at `+0x70`.

The common intent queue position read was:

```text
enqueuePos = 2
dequeuePos = 0
sizeApprox = 2
```

The queue sequence array and first three raw entry slots were also captured. The sequence array at `B+0x78c40` was:

```text
slot 0 = 0x1000
slot 1 = 0x1001
slot 2 = 0x0002
slot 3 = 0x0003
slot 4 = 0x0004
slot 5 = 0x0005
slot 6 = 0x0006
slot 7 = 0x0007
```

The first two raw entries were:

| slot | type byte | sequenceId at `+0x2d0` | payload words at `+0x10` | value at `+0x20` | value at `+0x28` |
|---:|---:|---:|---|---:|---:|
| 0 | `0x01` | `2` | `0xfe`, `0x1` | `0xa72d9d80` | `2` |
| 1 | `0x01` | `3` | `0xfd`, `0x1` | `0x00c765b1` | `3` |
| 2 | `0x00` | `0` | `0`, `0` | `0` | `0` |

Under the current `Intent` layout, type `1` is `Publish`, payload begins at `+0x10`, and `sequenceId` is at `+0x2d0`. These are raw slot contents; the sequence metadata prevents treating them as proven consumable entries.

## 4. Binary Predicate Disassembly

The S7 caller at image address `0x141fa8053` loads the Bridge with `this+0x110a480` and calls the wrapper at `0x141fb4670`. The wrapper adjusts to the scheduler subobject and enters the predicate body at `0x141fb4680`.

That body reads, in order:

1. `B+0x6c` for `swapPending_`.
2. `B+0x35cc40` and `B+0x35cc80` for the common intent queue positions.
3. `B+0x44180` and `B+0x441c0` for the observe deferred ring.
4. `B+0x415d00` and `B+0x415d40` for the quarantine fallback queue.
5. `B+0x72200` and `B+0x72240` for the recovery intent queue.
6. `B+0x20`, `+0x28`, `+0x30`, `+0x40`, `+0x48`, `+0x50`, and `+0x58` for the seven counter terms.
7. `B+0x72590` for recovery admission.
8. `B+0x78ba0` for live logical recovery obligations.

This is the authoritative binary predicate map for this capture. It is independent of the stale PDB member labels and does not use the old `+0x40` late-member workaround.

## 5. `intentQueue_ == 2` Provenance

### Source producer and consumer graph

The current common-queue producers are:

- `RuntimeIntentCoordinator::enqueuePublicationIntent` at `src/audioengine/ISRRuntimePublicationCoordinator.h:775-799`. It sets the prepared intent to `Publish`, reserves `publicationIntentResidencyCount_`, pushes to `intentQueue_`, and rolls back the reservation on push failure.
- `RuntimeIntentCoordinator::submitObserve` at `src/audioengine/ISRRuntimePublicationCoordinator.cpp:611-657`. It reserves `pendingIntentCount_`, pushes an `Observe` intent, and uses `observeDeferredRing_` on primary-queue overflow.
- `RuntimeIntentCoordinator::submitQuarantine` at `src/audioengine/ISRRuntimePublicationCoordinator.cpp:1474-1529`. It reserves both `pendingIntentCount_` and `quarantineIntentResidencyCount_`, then pushes a `Quarantine` intent or moves it to the fallback ring.

The consumer is `RuntimeIntentCoordinator::processIntent` at `src/audioengine/ISRRuntimePublicationCoordinator_ProcessIntent.cpp:10-84`. It drains the fallback ring first, then pops the common queue and applies the type-specific counter decrement before dispatching the handler. The caller chain is `CoordinatorLoop::run` at `src/audioengine/ISRCoordinatorLoop.cpp:31-48` to `AudioEngine::runCoordinatorPhase` at `src/audioengine/AudioEngine.Threading.cpp:293-299`.

### MPSC invariant check

`MpscBoundedRing::push` at `src/MpscBoundedRing.h:69-113` writes the entry and then publishes `sequences_[slot] = pos + 1`. `MpscBoundedRing::pop` at `src/MpscBoundedRing.h:117-131` consumes only when the sequence equals `dequeuePos + 1`.

For the captured `enqueuePos=2` and `dequeuePos=0`, a normally published pair at slots 0 and 1 would have sequence markers `1` and `2`. The captured markers are `0x1000` and `0x1001`. The position counters therefore report two resident slots while the sequence metadata does not describe two normally published entries.

The type/sequence bytes in the entries are shaped like old `Publish` payloads, but they cannot be promoted to live queue provenance without a matching published sequence marker. The most defensible classification is:

```text
intentQueue size metadata = 2
raw slot contents = two type-1/sequence-2,3-shaped values
normal consumable Publish entries = NOT PROVEN
```

This is a transport metadata/payload inconsistency. The current source graph has no legitimate operation that writes the captured sequence-marker pattern together with the captured position pair. The exact writer is therefore unresolved; no queue repair is proposed.

The counter values reinforce the mismatch:

```text
publicationIntentResidencyCount_ = 0
pendingIntentCount_ = 0
intentQueue_.sizeApprox() = 2
```

Two normal live `Publish` entries would require the publication residency counter to be nonzero. Two normal live `Observe`/`Quarantine`/`Recovery` entries would require pending residency to be nonzero. Neither source-consistent accounting is present.

## 6. `retireBacklogCount_ == 61` Provenance

### Current source graph

`RuntimeIntentCoordinator::enqueueRetire` is implemented at `src/audioengine/ISRRuntimePublicationCoordinator.cpp:148-171`:

- increments `retireAuthorityCount_`;
- delegates the actual router enqueue to `ISRRetireRouter::enqueueWithRetry`;
- on success calls `onRetireAccepted()`.

`onRetireAccepted()` at `:188-194` performs the atomic increment of `retireBacklogCount_`. `onRetireConsumed()` at `:196-205` is the paired decrement path. `setRetireBacklogCount()` at `:263-267` is explicitly test-only.

The source search found no production caller of `runtimePublicationBridge_.enqueueRetire`, and no call site for `onRetireConsumed()`. The visible `enqueueRetire` references in production are router/provider paths, not a textual call to this concrete Bridge method.

### Binary evidence

The binary contains the corresponding `enqueueRetire` body at image address `0x141fb4430`. Its successful path:

- increments the authority field at `+0x70`;
- calls the router operation;
- performs `xadd` on `B+0x20`;
- updates the pressure bookkeeping.

A direct binary call xref exists at image address `0x142009fc4`. That call site is not represented by a matching current-source production call to the concrete Bridge method. The binary therefore has a writer path that the current source call graph cannot attribute.

RCA-5 classification:

```text
binary writer class = PROVEN
current-source caller identity = NOT PROVEN
S7 value = 61
```

The value must not be rewritten as a D-queue count or treated as a repair target.

## 7. `quarantineRingResidencyCount_ == 61` Provenance

### Current source graph

The primary producer is `RuntimeIntentCoordinator::submitQuarantine` at `src/audioengine/ISRRuntimePublicationCoordinator.cpp:1474-1529`. On primary push failure it moves the intent to `quarantineFallbackQueue_` and increments `quarantineRingResidencyCount_` at `:1517-1519`.

The current production call sites are conditional Timer paths:

- `src/audioengine/AudioEngine.Timer.cpp:1973-1980`, reason `PublishViolation`;
- `src/audioengine/AudioEngine.Timer.cpp:2010-2017`, reason `ReceiptReset`.

The separate `AudioEngine::quarantineSlot` calls in `AudioEngine.Commit.cpp:622-643` use `RetireDeferralTimeout` and are not calls to `RuntimeIntentCoordinator::submitQuarantine`.

The fallback consumer is `src/audioengine/ISRRuntimePublicationCoordinator_ProcessIntent.cpp:35-40`. The binary fallback pop explicitly decrements `B+0x58` at image address `0x141fb7723`.

### M0 correlation

The captured primary quarantine counter at `B+0x50` is zero. The two raw common-queue slots have type `1`, not the current source `Quarantine` value `3`. No primary Quarantine intent is present in the directly readable common-queue state; the fallback positions were not directly reissued in this one-shot capture.

Nevertheless, the binary predicate reads `B+0x58` and the raw value is `61`. The current source graph therefore has no source-consistent path that explains this S7 value. A binary-only writer, stale counter state, or memory corruption remains possible; the exact writer cannot be selected from the available evidence.

RCA-5 classification:

```text
fallback consumer decrement = PROVEN
current-source producer call = not observed at S7
current-source writer identity = NOT PROVEN
S7 value = 61
```

## 8. Correlation and Verdict

The S7 false return is explained by three directly evidenced non-zero binary predicate terms:

```text
1. intentQueue_.sizeApprox() = 2
2. retireBacklogCount_ = 61
3. quarantineRingResidencyCount_ = 61
```

The first term has a direct queue-size observation but an unresolved sequence-marker/payload inconsistency. The second has a binary writer class but no matching current-source production caller. The third has a binary consumer and a conditional source producer, but no source-consistent S7 producer event.

These are Bridge transport/semantic-counter states. They are not the five engine D-queue entries and do not alter the RCA-3/RCA-4 D-queue carry-forward facts.

Narrow verdict:

```text
RCA-5-BINARY-PREDICATE-MAP = PROVEN
S7-INTENT-QUEUE-SIZE = 2
S7-RETIRE-BACKLOG = 61
S7-QUARANTINE-RING = 61
S7-RAW-QUEUE-ENTRIES = NOT NORMAL-LIVE-PROVEN
CURRENT-SOURCE-CALLER-ATTRIBUTION = INCOMPLETE
BINARY/SOURCE-FIELD-DRIFT = PROVEN
RCA-5-VERDICT = PARTIALLY-LOCALIZED / STOP
```

`PARTIALLY-LOCALIZED` is intentional. The binary predicate and writer/consumer classes are localized, but the current-source caller identity and the normal queue-publication explanation are not proven. The evidence does not justify a fix or a counter rewrite.

## 9. Constraints and Evidence Ledger

- M0: exactly one RCA-5 execution using the existing Release binary.
- Build: not run.
- M1/M2: not run.
- Dr. Memory: not run.
- No source, test, CMake, getter, counter, telemetry, shutdown, epoch, reclaim, router, or Recovery implementation changes.
- No D-queue repair or caller attribution was attempted.
- CDB/harness process check after M0: no remaining `cdb`, `windbg`, or `AudioEngineHarness` process.

Evidence files:

- `C:\Users\user\AppData\Local\Temp\opencode\rca5-cdb-capture.txt`
- `C:\Users\user\AppData\Local\Temp\opencode\rca5-cdb.log`
- `build/Release/AudioEngineHarness.exe`
- `build/Release/AudioEngineHarness.pdb`
- `src/audioengine/ISRRuntimePublicationCoordinator.cpp:148-206,263-267,511-562,611-657,1474-1529`
- `src/audioengine/ISRRuntimePublicationCoordinator_ProcessIntent.cpp:10-84,86-100`
- `src/audioengine/ISRRuntimePublicationCoordinator.h:702-799,971-1007`
- `src/audioengine/AudioEngine.Timer.cpp:1973-1980,2010-2017`
- `src/audioengine/AudioEngine.Threading.cpp:293-299`
- `src/audioengine/ISRCoordinatorLoop.cpp:31-48`
- `src/MpscBoundedRing.h:69-139`

## 10. STOP / Next Gate

RCA-5 stops after provenance classification. It does not select a repair, rebuild, or reopen D-queue analysis.

A later gate, if separately authorized, would need to reconcile the current source layout, the Release binary, and the stale PDB, then capture the correctly addressed `B+0x6c` state and the unissued fallback positions before making any causal claim beyond the present partial localization.
