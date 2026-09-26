# P3-5-FPM-RCA-7 — Engine DQueue Head Block: Exact Branch

- Gate: `P3-5-FPM-RCA-7`
- Mode: `read-only / M1 exact DQueue head branch capture`
- Date: 2026-09-24
- Input: `RCA-6 = PARTIALLY-LOCALIZED`
- Target: `Engine DQueue S7 head and first post-S7 reclaim`
- Scope: `exact globalEpoch, dequeuePos, head sequence, head epoch, minReaderEpoch, sequence gate, epoch gate, and CAS boundary`
- Verdict: `LOCALIZED / STOP`
- Disposition: `NO IMPLEMENTATION`
- M0 executions: `1`
- M1 executions: `1`
- M2 executions: `0`
- Build: `not run`
- Dr. Memory: `not run`
- Production/test/CMake/getter/counter/telemetry/shutdown/epoch/reclaim/router/Recovery changes: `0`

## 1. Objective and Frozen Scope

RCA-6 proved the five raw S7 slots and the slot-4 deleter, but could not promote the head stop to one exact reclaim branch because the earlier capture did not read the actual `dequeuePos` field or the current `globalEpoch` field.

RCA-7 uses one read-only M1 CDB execution of the existing Release target. It reads the current fields at S7, captures the first Engine `DeferredDeletionQueue::reclaim()` after S7, and records the exact branch outcome. It does not modify production or test code, change counters, force an epoch, repair the queue, or start RCA-8.

## 2. M1 Target and Binary Identity

Target:

- Executable: `build/Release/AudioEngineHarness.exe`
- PDB: `build/Release/AudioEngineHarness.pdb`
- Argument: `--fpm-m0`
- S7 RVA: `0x1fa80b3`
- S7 marker: `AudioEngineHarness+0x1fa80b3`
- M1 script: `C:\Users\user\AppData\Local\Temp\opencode\rca7-cdb-capture.txt`
- M1 log: `C:\Users\user\AppData\Local\Temp\opencode\rca7-cdb.log`

Binary identity:

```text
EXE SHA-256 = E5C7AFB9C4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75
PDB SHA-256 = A793738F12F9BA3E0C7A19981B35212659107A356C6904C2BF809933CE1F4391
```

The M1 command completed normally with process exit code `0`. The log ends at `RCA7_PROCESS_EXIT` followed by `quit` (`rca7-cdb.log:2693-2698`).

## 3. S7 Object and Queue Addresses

At the S7 breakpoint (`rca7-cdb.log:258-274`):

```text
S7 this / rbx                 = 0x00000186bbd92080
EpochDomain base               = 0x00000186bce49740
Engine DQueue base              = 0x00000186bce4ab80
globalEpoch address             = 0x00000186bce49758
enqueuePos address              = 0x00000186bce7eb80
dequeuePos address              = 0x00000186bce7ebc0
head sequence address           = 0x00000186bce7ab90
head entry address              = 0x00000186bce4ac40
reclaimAttemptCount address     = 0x00000186bce7ec80
reclaimSuccessCount address     = 0x00000186bce7ec88
reclaimLocalCounter address     = 0x00000186bce7ecc0
```

The first Engine reclaim is identified by the MSVC x64 `this` register `RCX=0x00000186bce4ab80` (`rca7-cdb.log:374-380`). The other DQueue instances captured by the global code breakpoints are not substituted for the Engine DQueue.

## 4. Fresh S7 Field State

The exact field reads are at `rca7-cdb.log:287-298`.

```text
globalEpoch       = 9
enqueuePos        = 5
dequeuePos        = 4
headIndex         = dequeuePos & 0xfff = 4
expectedSequence  = dequeuePos + 1 = 5
head sequence     = 5
head entry ptr    = 0x00000186cb383080
head deleter      = 0x00007ff7b06579c0
head entry epoch  = 9
head metadata     = type=0, publicationSequenceId=0, generation=0
```

The S7 head relation is explicitly printed at `rca7-cdb.log:299-302`:

```text
dequeuePos=0000000000000004
headIndex=0000000000000004
expectedSequence=0000000000000005
actualSequence=0000000500000005
head=00000186bce4ac40
```

The `actualSequence` value in that line is a 64-bit `poi()` print. The low DWORD is `0x00000005`; the high DWORD is the adjacent sequence slot. The exact 32-bit sequence is independently shown by the first reclaim read at `0x141f9cd3c` (`rca7-cdb.log:385-396`): the instruction reads `dword ptr [r15+0x30000]` and the captured value is `0x00000005`. Therefore the semantic sequence is exactly `5`, not the 64-bit printed pair.

The five S7 slots remain:

| slot | ptr | deleter | epoch | type | publicationSequenceId | generation |
|---:|---|---|---:|---:|---:|---:|
| 0 | `0x0000000000000000` | `0x0000000000000000` | 1 | `Generic (0)` | 0 | 0 |
| 1 | `0x0000000000000000` | `0x0000000000000000` | 2 | `Generic (0)` | 0 | 0 |
| 2 | `0x0000000000000000` | `0x0000000000000000` | 4 | `Generic (0)` | 0 | 0 |
| 3 | `0x0000000000000000` | `0x0000000000000000` | 5 | `Generic (0)` | 0 | 0 |
| 4 | `0x00000186cb383080` | `0x00007ff7b06579c0` | 9 | `Generic (0)` | 0 | 0 |

The table is from `rca7-cdb.log:312-319`.

The S7 counters are:

```text
reclaimAttemptCount_ = 0x0000000000006000
reclaimSuccessCount_ = 0x0000000000000004
worldReclaimCount_   = 0x0000000000000002
```

The first two values are the two qwords at `EpochDomain+0x35540` (`rca7-cdb.log:295-296`). The S7 reader scan emitted no nonzero-depth row, but that scan is not promoted as proof of reader identity because its per-slot offset probe was not independently validated against the current PDB layout.

## 5. Source and Binary Reclaim Contract

The source contract is:

- `EpochDomain::tryReclaim()` passes `getMinReaderEpoch()` to `deferredDeletionQueue.reclaim()` at `src/core/EpochDomain.h:385-395`.
- `getMinReaderEpoch()` starts at `currentEpoch()` and only lowers the result for an eligible active reader at `src/core/EpochDomain.h:214-247`.
- `reclaim()` reads the FIFO head, requires `sequence == dequeuePos + 1` at `src/DeferredDeletionQueue.h:115-127`, checks `isOlder(entry.epoch, minReaderEpoch)` at `src/DeferredDeletionQueue.h:129-134`, and only then CAS-advances `dequeuePos` at `src/DeferredDeletionQueue.h:136-142`.
- If the head is sequence-ready but not epoch-reclaimable, FIFO order requires an immediate break at `src/DeferredDeletionQueue.h:171-175`.
- `isOlder(a,b)` is the signed comparison `static_cast<int64_t>(a-b) < 0` at `src/core/EpochDomain.h:435-438`.

The Release disassembly independently shows the same branches:

```text
0x141f9cd00  reclaim entry
0x141f9cd3c  read dword sequence
0x141f9cd45  sequence comparison
0x141f9cd59  read entry epoch
0x141f9cd66  epoch gate
0x141f9cd72  lock cmpxchg [rdi+0x34040], r14d
0x141f9cdf9  CAS failure path
0x141f9ce04  reclaim return path
```

## 6. First Engine DQueue Reclaim: Exact Path

### 6.1 Minimum-reader value

The first Engine reclaim is preceded by `getMinReaderEpoch()` returning `rax=9` (`rca7-cdb.log:363-373`). At reclaim entry, the same value is in the second argument `rdx=9` and the Engine DQueue is in `rcx=0x00000186bce4ab80` (`rca7-cdb.log:374-384`):

```text
Engine DQueue this = 0x00000186bce4ab80
minReaderEpoch     = 9
globalEpoch         = 9
```

### 6.2 Sequence gate passes

At the first reclaim sequence read:

```text
dequeuePos / scanPos = 4
head sequence        = 5
expected sequence    = dequeuePos + 1 = 5
```

`rca7-cdb.log:385-396` shows the dword read at `0x141f9cd3c` returning `0x00000005` with `rbp=4`. The comparison at `0x141f9cd45` reaches the equality-success path; the first call proceeds to the epoch read at `0x141f9cd59` rather than returning from the sequence failure path.

The sequence branch is therefore:

```text
sequence_ready = true
head = dequeuePos + 1
```

This excludes Case A, a sequence mismatch, for this first Engine call.

### 6.3 Epoch gate blocks the head

The entry address is `0x00000186bce4ac40`. The epoch read at `0x141f9cd59` captures:

```text
entry.epoch = 9
minReaderEpoch = 9
```

At the epoch gate (`rca7-cdb.log:418-431`):

```text
epoch difference after the compiled comparison = 0
ZF = 1
je 0x141f9ce04 [br=1]
```

The next captured event is `RCA7_RECLAIM_RETURN`; there is no `RCA7_CAS_BEFORE` or `RCA7_CAS_SUCCESS` between the epoch gate and return (`rca7-cdb.log:421-443`).

The source predicate is false for equality:

```text
isOlder(9, 9) = false
```

The first Engine head therefore stops at the epoch gate, before the dequeue CAS. This excludes Case C (CAS failure) for the first call; there was no CAS attempt.

The exact first-call classification is:

```text
Case A: sequence mismatch             EXCLUDED
Case B: head epoch not reclaimable    PROVEN
Case C: dequeue CAS failure            EXCLUDED (CAS not reached)
Case D: unrelated/empty head           EXCLUDED
```

## 7. Repeated Calls and Recovery Control

The first three Engine DQueue calls captured after S7 (`rca7-cdb.log:374-617`) all show the same structure: sequence comparison, epoch gate, and return, with no dequeue write. The `reclaimSuccessCount_` watch remains `4` during these calls.

The shutdown log then records the force-epoch phase (`rca7-cdb.log:620-626`). A later Engine DQueue call captures:

```text
getMinReaderEpoch() = 12
entry.epoch         = 9
epoch gate           = pass
old dequeuePos       = 4
CAS desired value    = 5
CAS result           = success
observed dequeuePos  = 5
```

The exact CAS evidence is `rca7-cdb.log:628-721`:

```text
0x141f9cd66  je ... [br=0]
0x141f9cd72  lock cmpxchg dword ptr [rdi+34040h], r14d
             old = 0x00000004
             new = 0x00000005
0x141f9cd7b  CAS success
watchpoint  dequeuePos: 4 -> 5
```

This control transition demonstrates that the S7 head was not permanently un-CAS-able. Once the supplied minimum reader epoch became older than the entry epoch, the same FIFO head passed the epoch gate and advanced normally. The later queue activity and the coordinator fault diagnostic do not change the first-call classification.

The M1 process continued through shutdown and exited normally after the captures. The `[FAULT] coordinator in Faulted state...` diagnostic at `rca7-cdb.log:2534` is recorded as an observed shutdown symptom, not promoted here to the DQueue head root cause.

## 8. Root-Cause Boundary

RCA-7 localizes the exact S7 head blocking condition:

```text
S7 Engine DQueue head:
  dequeuePos = 4
  head sequence = 5 = dequeuePos + 1
  head epoch = 9
  actual minReaderEpoch = 9
  isOlder(9, 9) = false
  => epoch gate returns before dequeue CAS
```

What is proven:

- The S7 field addresses and values are fresh and identity-checked.
- The first Engine reclaim call is distinct from the EQ and other DQueue calls by `RCX`.
- The sequence gate passes.
- The head epoch equals the actual supplied minimum reader epoch.
- The epoch gate is taken and the function returns zero without attempting dequeue CAS.
- A later minimum-reader value of `12` permits the same head to CAS from `4` to `5`.

What remains outside this gate:

- Which reader, reader lifecycle event, or epoch publication history produced `minReaderEpoch=9` is not identified by M1.
- The S7 reader-slot scan is not accepted as exact reader identity evidence.
- No claim is made that the queue was permanently corrupt or that a production fix is required.
- No M2, Dr. Memory, build, source change, queue repair, forced reclaim, or RCA-8 work was performed.

## 9. Evidence Index

| evidence | location |
|---|---|
| M1 S7 registers and marker | `rca7-cdb.log:258-274` |
| S7 exact field reads | `rca7-cdb.log:276-298` |
| S7 head relation | `rca7-cdb.log:299-302` |
| S7 head and five slots | `rca7-cdb.log:303-319` |
| Breakpoint and watchpoint setup | `rca7-cdb.log:323-356` |
| First Engine reclaim entry | `rca7-cdb.log:363-384` |
| First sequence read | `rca7-cdb.log:385-408` |
| First epoch read and gate return | `rca7-cdb.log:409-443` |
| Repeated zero-return calls | `rca7-cdb.log:448-617` |
| Force-epoch recovery CAS | `rca7-cdb.log:620-721` |
| Process termination | `rca7-cdb.log:2693-2698` |
| Reclaim source contract | `src/DeferredDeletionQueue.h:110-177` |
| Minimum-reader source contract | `src/core/EpochDomain.h:214-247,385-395,435-438` |

## 10. Disposition

```text
GATE_DISPOSITION = LOCALIZED / STOP
FIRST_HEAD_BRANCH = EPOCH_GATE / EQUAL_HEAD_AND_MIN_READER
SEQUENCE_GATE = PASS
DEQUEUE_CAS = NOT REACHED IN FIRST CALL
LATER_CONTROL_CAS = SUCCESS, 4 -> 5
IMPLEMENTATION = FORBIDDEN
RCA-8 = NOT STARTED
M0 = 1 OF 1 USED
M1 = 1 OF 1 USED
M2 = 0
DR_MEMORY = 0
```

No production fix, test change, counter change, queue repair, epoch advance, forced reclaim, or shutdown modification is authorized by this report.
