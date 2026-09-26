# P3-5-FPM-RCA-6 — S7 DQueue Five-Entry Provenance

- Gate: `P3-5-FPM-RCA-6`
- Mode: `read-only / S7 DQueue provenance audit`
- Date: 2026-09-24
- Input: `RCA-4 = LOCALIZED / RCA-5 = PARTIALLY-LOCALIZED`
- Target: `Engine DQueue S7 depth = 5`
- Scope: `five pre-existing S7 DQueue entries only`
- Verdict: `PARTIALLY-LOCALIZED / STOP`
- Disposition: `NO IMPLEMENTATION`
- M0 executions: `1`
- M1/M2 executions: `0`
- Build: `not run`
- Dr. Memory: `not run`
- Production/test/CMake/getter/counter/telemetry/shutdown/epoch/reclaim/router/Recovery changes: `0`

## 1. Objective and Frozen Scope

RCA-6 attributes the five raw slots already resident in the Engine `DeferredDeletionQueue` at S7, identifies the concrete object/deleter where possible, and determines whether shutdown reclaim attempted to advance the queue.

The audit is read-only. It does not reopen the Bridge predicate investigation, modify production or test code, change counters, force a drain, advance an epoch, alter a router, or propose a fix. The single permitted live execution was the existing Release target with `--fpm-m0`.

The S7 breakpoint was `AudioEngineHarness+0x1fa80b3`, the false-return site used by the preceding RCA captures.

## 2. Correction to the Earlier DQueue Interpretation

The earlier `dequeuePos = 0` and `sizeApprox = 5` statements are not independently proven by the RCA-6 capture.

The current PDB type and the current Release disassembly place the fields separately:

```text
DeferredDeletionQueue + 0x34000 = enqueuePos
DeferredDeletionQueue + 0x34040 = dequeuePos
```

The RCA-6 M0 script read `dd` at `enqueuePos` with a length of two DWORDs. The second DWORD was padding after the 32-bit `enqueuePos`; it was not the separate 32-bit `dequeuePos` at `+0x34040`. Therefore:

- `enqueuePos = 5` is a valid first-DWORD observation.
- The initial `dequeuePos` value in this M0 is `UNREAD` and is not promoted from the earlier report.
- `sizeApprox = 5` is not a fresh RCA-6 observation; it remains only a carry-forward interpretation from RCA-3.
- The absence of a later dequeue write is still valid because the hardware watchpoint was installed at the actual `+0x34040` address.

The same script read `EpochDomain + 0x0`, which is the vtable area, instead of `globalEpoch` at `EpochDomain + 0x18`. The current global epoch is therefore also not a fresh RCA-6 field read. `epoch=9` on the concrete entry and the earlier S7 `currentEpoch=9` observation are kept separate.

## 3. M0 Target and Layout

Target:

- Executable: `build/Release/AudioEngineHarness.exe`
- PDB: `build/Release/AudioEngineHarness.pdb`
- Argument: `--fpm-m0`
- S7 RVA: `0x1fa80b3`
- S7 `this` (`rbx`): `0x000001763522b080`
- `EpochDomain`: `this + 0x10b76c0 = 0x00000176362e2740`
- `DeferredDeletionQueue`: `EpochDomain + 0x1440 = 0x00000176362e3b80`
- `enqueuePos` address: `0x0000017636316740`
- `dequeuePos` address: `0x0000017636316780`

Capture artifacts:

- Script: `C:\Users\user\AppData\Local\Temp\opencode\rca6-cdb-capture.txt`
- Log: `C:\Users\user\AppData\Local\Temp\opencode\rca6-cdb.log`

The runtime type dump confirmed:

```text
EpochDomain + 0x018 = globalEpoch
EpochDomain + 0x020 = readers[64]
EpochDomain + 0x1440 = deferredDeletionQueue
EpochDomain + 0x35540 = reclaimAttemptCount_
EpochDomain + 0x35548 = reclaimSuccessCount_
EpochDomain + 0x35580 = reclaimLocalCounter_

DeletionQueue + 0x00000 = ringBuffer
DeletionQueue + 0x30000 = sequences
DeletionQueue + 0x34000 = enqueuePos
DeletionQueue + 0x34040 = dequeuePos
DeletionQueue + 0x34080 = maxRetireAgeUs_
DeletionQueue + 0x340c0 = worldReclaimCount_
```

`CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=OFF` is present in `build/CMakeCache.txt`; the Release `DeletionEntry` has no `objectBytes` field.

## 4. Fresh S7 Entry Table

The five raw slots were captured at S7 in `rca6-cdb.log:330-334`.

| slot | ptr | deleter | epoch | type | publicationSequenceId | generation | objectBytes |
|---:|---|---|---:|---|---:|---:|---|
| 0 | `0x0000000000000000` | `0x0000000000000000` | 1 | `Generic (0)` | 0 | 0 | unavailable |
| 1 | `0x0000000000000000` | `0x0000000000000000` | 2 | `Generic (0)` | 0 | 0 | unavailable |
| 2 | `0x0000000000000000` | `0x0000000000000000` | 4 | `Generic (0)` | 0 | 0 | unavailable |
| 3 | `0x0000000000000000` | `0x0000000000000000` | 5 | `Generic (0)` | 0 | 0 | unavailable |
| 4 | `0x00000176448f0080` | `0x00007ff7b06579c0` | 9 | `Generic (0)` | 0 | 0 | unavailable |

The sequence array at S7 was:

```text
sequences[0] = 0x00001000
sequences[1] = 0x00001001
sequences[2] = 0x00001002
sequences[3] = 0x00001003
sequences[4] = 0x00001004
```

The first position DWORD was `enqueuePos = 5`. The actual `dequeuePos` DWORD was not issued by the M0 script.

The fresh reader scan emitted no nonzero-depth reader. The prior S7 `activeReaders=0` observation is consistent with this run. The fresh script did not read `globalEpoch` at the correct `+0x18` offset, so `currentEpoch=9` and `minReaderEpoch=9` remain carry-forward values, not new RCA-6 fields.

The reclaim counters at S7 were:

```text
reclaimAttemptCount_ = 0x6800
reclaimSuccessCount_ = 4
worldReclaimCount_ = 0
maxRetireAgeUs_ = 0
```

The values are from `rca6-cdb.log:306-312`. The third qword in that old dump was padding after `reclaimSuccessCount_`; it was not `reclaimLocalCounter_`.

## 5. Reclaim Observation

The source reclaim contract is:

- `EpochDomain::tryReclaim()` calls `deferredDeletionQueue.reclaim(getMinReaderEpoch())` at `src/core/EpochDomain.h:385-395`.
- `DeferredDeletionQueue::reclaim()` reads the FIFO head, requires the sequence to equal `dequeuePos + 1`, checks the entry epoch, and only then CAS-advances `dequeuePos` at `src/DeferredDeletionQueue.h:110-177`.
- If the head is not ready or is not reclaimable, the function exits without advancing the queue.

The current Release disassembly independently confirms the same full-ticket implementation:

- `0x141f9c550`: enqueue reads `enqueuePos` and compares the sequence.
- `0x141f9c57d`: CAS advances `enqueuePos` by one.
- `0x141f9c5cd-0x141f9c5d2`: increments the full ticket and publishes the sequence.
- `0x141f9cd00`: reclaim entry.
- `0x141f9cd0d`: reads `dequeuePos`.
- `0x141f9cd3c-0x141f9cd48`: requires `sequence == dequeuePos + 1`.
- `0x141f9cd59-0x141f9cd66`: checks the entry epoch against the supplied minimum reader epoch.
- `0x141f9cd72`: CAS-advances `dequeuePos`.

Four hardware watchpoints were installed after S7:

```text
dequeuePos       = this + 0x10eb700
enqueuePos       = this + 0x10eb6c0
reclaimLocal     = EpochDomain + 0x35580
reclaimSuccess   = EpochDomain + 0x35548
```

The addresses are visible in `rca6-cdb.log:343-346`.

Observed events after S7:

- `RCA6_WATCH_RECLAIM_LOCAL_WRITE`: 6 occurrences.
- `RCA6_WATCH_RECLAIM_SUCCESS_WRITE`: 6 occurrences.
- `RCA6_WATCH_DEQUEUE_WRITE`: 0 occurrences.
- `RCA6_WATCH_ENQUEUE_WRITE`: 0 occurrences.

The paired local/success writes prove that `tryReclaim` executed repeatedly. The missing dequeue writes prove that none of those calls advanced the DQueue head. The command-string dumps after a watchpoint used the then-current `rbx` and printed unreadable addresses; only the watchpoint event occurrence and the absolute watchpoint address are used as evidence.

The log explicitly records `drain timeout reached, performing safe tryReclaim (drainAll skipped)` at `rca6-cdb.log:350`. The subsequent audit records `routerPendingRetire=2` at `rca6-cdb.log:375`; that is the separate System 1/router metric and is not substituted for the Engine DQueue depth.

## 6. Writer and Deleter Provenance

### 6.1 Concrete slot 4

The slot-4 deleter address is:

```text
module base + 0x1fa79c0
preferred VA 0x141fa79c0
```

The current binary at that address calls the DSPCore destructor path and then the aligned free/operator-delete path. This matches:

- `AudioEngine::destroyDSPCoreNode(void*)` at `src/audioengine/AudioEngine.Threading.cpp:18-40`.
- The `DSPLifetimeManager::retire` enqueue at `src/audioengine/DSPLifetimeManager.cpp:40-61`.
- The `DSPLifetimeManager::retireByHandle` enqueue at `src/audioengine/DSPLifetimeManager.cpp:79-124`.

Both manager paths pass `&AudioEngine::destroyDSPCoreNode` to `ISRRetireRouter::enqueueWithRetry` with `DeletionEntryType::Generic`; the source does not pass a nonzero publication sequence or generation. This exactly matches slot 4's zero metadata.

The strongest pre-S7 production candidate is the orphan-current-DSP path in `src/audioengine/AudioEngine.RebuildDispatch.cpp:800-805`, which calls `DSPLifetimeManager::retire(currentToRelease)` before the rebuild task is committed. The alternative registered-DSP path is `RuntimePublicationOrchestrator::retireRegisteredDSP` at `src/audioengine/RuntimePublicationOrchestrator.cpp:745-761`.

Therefore:

```text
slot 4 object class: DSPCore                         PROVEN by deleter mapping
slot 4 writer family: DSPLifetimeManager -> Router   STRONGLY-LOCALIZED
slot 4 exact enqueue event and DSP instance:        PARTIAL
slot 4 publication identity:                        UNAVAILABLE; metadata is zero
```

The final-shutdown and destructor retire calls in `AudioEngine.Processing.ReleaseResources.cpp:568-587` and `AudioEngine.CtorDtor.cpp:172-211` occur after S7 and are not the preferred explanation for this pre-existing slot.

### 6.2 Null slots 0-3

Normal production entry points reject null ownership pairs:

- `AudioEngine::enqueueDeferredDeleteNonRtWithResult` returns success without enqueue when `ptr` or `deleter` is null at `src/audioengine/AudioEngine.h:4460-4464`.
- `ISRRetireRouter::enqueueRetire` does the same at `src/audioengine/ISRRetireRouter.cpp:239-246`.

No normal production writer was found that intentionally enqueues a null pointer/deleter pair. The source reclaim path clears `ptr`, `deleter`, and `type` but leaves `epoch`, `publicationSequenceId`, and `generation` at `src/DeferredDeletionQueue.h:143-161`.

The four null slots, their nonzero epochs, and the pre-S7 `reclaimSuccessCount_=4` are therefore strongly consistent with four already-cleared/tombstoned entries, not four fresh live retirements. This is `PROBABLE`, not exact per-slot causality, because the initial `dequeuePos` and per-slot sequence-to-dequeue relation were not read correctly in the M0 capture.

## 7. Queue-State Invariant Finding

The current binary does not mask the ticket counters. It uses the low 12 bits only to select the ring index and retains the full 32-bit ticket for sequence arithmetic and CAS.

A healthy queue therefore cannot simultaneously expose:

```text
enqueuePos = 5
sequences[0..4] = 0x1000..0x1004
```

as a normal initial/current state. The sequence values indicate a later generation, while the position field is still at the initial five-ticket range. This is a DQueue state invariant violation in the captured runtime, independent of the unresolved initial `dequeuePos` value.

The most conservative explanation is:

1. Four historical entries were cleared, leaving null tombstone fields.
2. One concrete DSPCore entry remains with epoch 9.
3. The queue's position/sequence generation is inconsistent, so `reclaim` sees a non-ready head or otherwise cannot CAS-advance it.
4. Repeated `tryReclaim` calls consequently produce no dequeue write.

The source of the invariant violation—memory corruption, an out-of-band reset/write, stale binary/PDB state, or an unmodeled producer path—is not localized by this read-only gate.

## 8. Why the Five Raw Slots Remain

The evidence supports the following boundary:

```text
Five raw slots observed at S7:                         PROVEN
Four null pointer/deleter pairs:                       PROVEN
One concrete DSPCore entry with destroyDSPCoreNode:    PROVEN
tryReclaim invoked repeatedly after S7:                PROVEN
No dequeue position write after S7:                    PROVEN
No forced drainAll in the observed path:               PROVEN
Exact initial dequeuePos:                              UNREAD
Exact head stop branch:                                UNRESOLVED
Exact enqueue event/instance for slot 4:               PARTIAL
```

It is not valid to state from this M0 alone that the head was blocked only because `epoch=9 == currentEpoch=9`. The fresh script did not read the current epoch or initial dequeue position. The source-level alternatives remain:

- sequence not equal to `dequeuePos + 1` (`diff != 0`), or
- a head that is not older than the supplied minimum reader epoch.

The observed sequence/ticket inconsistency makes the first branch the stronger current hypothesis, but the gate stops before promoting it to a single proven branch.

## 9. Disposition

```text
GATE_DISPOSITION = PARTIALLY-LOCALIZED / STOP
IMPLEMENTATION = FORBIDDEN
RCA-7 = NOT STARTED
M0 = 1 OF 1 USED
M1/M2/DrMemory = 0
```

No production fix, test change, counter change, queue repair, epoch advance, forced reclaim, or shutdown modification is authorized by this report.

A future capture, if separately authorized, must read `EpochDomain+0x18`, `enqueuePos+0x34000`, and `dequeuePos+0x34040` as separate fields and use persistent absolute watchpoint addresses. The corrected script is retained at `C:\Users\user\AppData\Local\Temp\opencode\rca6-cdb-capture.txt`; it was not rerun under this gate.

## 10. Evidence Index

- M0 script: `C:\Users\user\AppData\Local\Temp\opencode\rca6-cdb-capture.txt`
- M0 log: `C:\Users\user\AppData\Local\Temp\opencode\rca6-cdb.log`
- M0 S7 hit: `rca6-cdb.log:259-263`
- M0 layout/type: `rca6-cdb.log:268-305`
- M0 counters: `rca6-cdb.log:306-312`
- M0 sequence/position dump: `rca6-cdb.log:315-323`
- M0 five entries: `rca6-cdb.log:329-334`
- M0 watchpoint addresses: `rca6-cdb.log:335-346`
- M0 reclaim/no-dequeue evidence: `rca6-cdb.log:350-424`
- Source queue implementation: `src/DeferredDeletionQueue.h:58-177`
- Epoch reclaim wrapper: `src/core/EpochDomain.h:385-395`
- Router null guards and enqueue: `src/audioengine/ISRRetireRouter.cpp:239-336`
- AudioEngine enqueue wrapper: `src/audioengine/AudioEngine.h:4452-4496`
- DSPCore lifetime writer: `src/audioengine/DSPLifetimeManager.cpp:35-124`
- DSPCore deleter: `src/audioengine/AudioEngine.Threading.cpp:18-46`
- Pre-S7 orphan retire candidate: `src/audioengine/AudioEngine.RebuildDispatch.cpp:800-805`
- System 1/System 2 separation: `ConvoPeq.md:18300-18304`
- Current Release binary: `build/Release/AudioEngineHarness.exe`
- Current PDB: `build/Release/AudioEngineHarness.pdb`

The first launcher attempt used the wrong CDB log option and failed before debuggee initialization; it did not consume an M0 execution. The single actual M0 was the subsequent `-logo` run, which reached S7, ran to process termination, and returned exit code 0.
