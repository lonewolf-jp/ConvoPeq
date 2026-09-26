# P3-5-FPM-C3-T1-Capture-Redesign-2 — Script Preparation / Static Validation

## 1. Gate result

```text
Gate                         = P3-5-FPM-C3-T1-Capture-Redesign-2
Mode                         = debugger-script preparation + static validation
Script created               = YES, exactly 1
Script                       = doc/work113/P1-5-IR-P2_Step5BB_P3-5-FPM-C3-T1-Capture-Redesign-2.cdb
Script SHA-256               = FBE32851F795AF900E6C46D6E38F25138E01217C2B52C0BDDE1F015029CFC39E
Script bytes                 = 14,377
Breakpoint count             = 13
ReaderSlot literal windows   = 256 = 4 blocks x 64 slots
Static validation            = PASS, 27/27
CDB parser execution         = 0
Harness execution            = 0
CDB Retry-2 execution        = 0
M1/M2                        = 0
Dr.Memory                     = 0
Build                        = 0
Production source changes    = 0
Test source changes          = 0
CMake changes                = 0
Gate verdict                 = SCRIPT READY / STATIC VALIDATED
Runtime authorization        = NOT GRANTED
S7_READER                    = UNRESOLVED
Case A/B/C/D                 = NOT_PROVEN
minReaderEpoch               = NOT_CAPTURED
IMPLEMENTATION               = FORBIDDEN
```

This gate prepares and statically validates one CDB script. It does not load the script into CDB, start `AudioEngineHarness`, or authorize Retry-2.

## 2. Frozen identities

| artifact | SHA-256 | bytes | result |
|---|---|---:|---|
| `ConvoPeq.md` | `E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609` | 5,535,334 | PASS |
| `AudioEngineHarness.exe` | `E5C7AFB9C4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75` | 41,216,512 | PASS |
| `AudioEngineHarness.pdb` | `A793738F12F9BA3E0C7A19981B35212659107A356C6904C2BF809933CE1F4391` | 59,355,136 | PASS |
| Redesign-2 CDB script | `FBE32851F795AF900E6C46D6E38F25138E01217C2B52C0BDDE1F015029CFC39E` | 14,377 | PASS |

`ConvoPeq.md` remains the sole production source authority. The separate older attachment named `ConvoPeq(3).md` was not substituted or reinterpreted.

## 3. Breakpoint inventory

| # | RVA | purpose | static disposition |
|---:|---:|---|---|
| 1 | `0x1f9c550` | T0 enqueue entry | PASS |
| 2 | `0x1f9c57d` | T0 CAS pre | PASS |
| 3 | `0x1f9c59a` | T0 CAS post | PASS |
| 4 | `0x1f9c5da` | sequence publication / target arming | PASS |
| 5 | `0x1fa80b3` | S7 terminal anchor | PASS |
| 6 | `0x1f9cfb0` | EpochDomain tryReclaim candidate | PASS |
| 7 | `0x1f9cfe2` | getMinReaderEpoch pre-call + ReaderSlots | PASS |
| 8 | `0x1f9cfe5` | getMinReaderEpoch return in RAX | PASS |
| 9 | `0x1f9cd00` | DQueue reclaim entry | PASS |
| 10 | `0x1f9cd59` | target entry epoch gate / T1_SELECTION | PASS |
| 11 | `0x1f9cd72` | target dequeue CAS pre | PASS |
| 12 | `0x1f9ce04` | CAS success / epoch blocked / contradiction | PASS |
| 13 | `0x1fa80b2` | T2 waitForDrain return | PASS |

The expected RVA sequence is exact and contains no drift.

## 4. Pseudo-register role and writer validation

| register | role | validated writers |
|---|---|---|
| `$t0` | `targetArmed` | initialization + T0 arming only; exactly one `r $t0=1` |
| `$t1` | epochBase | initialization + T0 arming only |
| `$t2` | target DQueue | initialization + T0 arming only |
| `$t3` | target EpochDomain | initialization + T0 arming only |
| `$t4/$t5` | target ptr low/high | initialization + T0 arming only |
| `$t6/$t7` | target deleter low/high | initialization + T0 arming only |
| `$t8/$t9` | target epoch low/high | initialization + T0 arming only |
| `$t10` | target ticket | initialization + T0 arming only |
| `$t11` | candidate thread | initialization + reset + tryReclaim candidate capture |
| `$t12` | candidateActive | initialization + reset + candidate admission + return completion |
| `$t13` | invocation token | initialization + reset + monotonic candidate increment; zero-wrap repaired to 1 |
| `$t14/$t15` | same-invocation minReaderEpoch | initialization + reset + candidate reset + getMin return from RAX only |
| `$t16` | T1 selection uniqueness | initialization + reset + T1_SELECTION only |
| `$t17` | T2 captured | initialization + reset + T2 only |
| `$t18` | anomaly | initialization + reset + every C3_CR1_STOP path |
| `$t19` | S7 terminal-thread diagnostic | initialization + T0 reset + S7 thread capture only |

No custom pseudo-register, `$s7`, `$s7e`, `$s7q`, or `$s7rbx` is present. All user-defined pseudo-registers are within `$t0` through `$t19`.

`$t14/$t15` are never written after T1_SELECTION. They are populated from the live `RAX` at `0x1f9cfe5`, not from T0/S7/T2 or a hardcoded value.

## 5. T0 arming validation

The script does not arm T1 at T0 entry, CAS pre, or CAS post.

The only `r $t0=1` site is `0x1f9c5da`, and it requires:

```text
targetArmed == 0
entry epoch argument == 9
ticket+1 register == 5
ticket == 4
enqueuePos == 5
sequence[4] == 5
DQueue != 0
```

This is the first instruction after the current EXE sequence release store and closes the incomplete-entry arming race.

## 6. Candidate and same-invocation correlation

At `0x1f9cfb0`, the script requires:

```text
targetArmed == 1
T2 not captured
RCX == target EpochDomain
T1 not already selected
candidateActive == 0
```

It then captures:

```text
$t11 = $tid
$t12 = 1
$t13 = incremented nonzero invocation token
$t14 = 0
$t15 = 0
```

A second candidate while `candidateActive==1` is represented as a hard conflict and cannot overwrite the active thread/token.

At T1-B, T1-C, CAS, and reclaim return, the join is tri-state:

```text
active + same thread + same object -> join/check
active + same object + wrong thread -> C3_CR1_STOP
active + same thread + wrong object -> C3_CR1_STOP
active + different thread + different object -> ignore unrelated provider
inactive                          -> gc
```

This prevents a shared `0x1f9cd*` code address from either merging evidence or creating a false conflict for an unrelated domain.

## 7. getMinReaderEpoch capture

The pre-call breakpoint `0x1f9cfe2` requires the candidate thread, domain, and token, then captures:

```text
globalEpoch
ReaderSlot[0..63]
DQueue enqueue/dequeue/sequence state
```

The ReaderSlot block is 64 literal `L80` reads with stride `0x50`.

The return breakpoint `0x1f9cfe5` requires the same candidate thread/domain/token and writes:

```text
$t14 = RAX & 0xffffffff
$t15 = RAX >> 32
```

No T0/S7/T2 snapshot or hardcoded epoch is used to reconstruct minReaderEpoch.

## 8. T1-C target identity join

The epoch-gate breakpoint `0x1f9cd59` requires:

```text
candidateActive == 1
current thread == candidate thread
RDI == target DQueue
invocation token != 0
R13 low/high == captured $t14/$t15
RBP == target ticket
RSI == DQueue + 0xC0 + (ticket & 0xFFF) * 0x30
sequence == ticket + 1
entry ptr == captured target ptr
entry deleter == captured target deleter
entry epoch == captured target epoch
```

T1_SELECTION is permitted only when `$t16==0`. A second target selection is a `C3_CR1_STOP_MULTIPLE_TARGET_SELECTION` anomaly.

A same-ticket/expected-sequence head with a different ptr, deleter, or epoch is a `C3_CR1_STOP_TARGET_IDENTITY_CONFLICT` anomaly.

A lower scan position is retained as `C3_PREDECESSOR_HEAD`; it does not discard the candidate, because the same DQueue invocation may reclaim earlier FIFO entries and later reach the target.

The selected record prints:

```text
candidate thread/token
same-invocation minReaderEpoch
current entry
head sequence
scan position
dequeue position
expected sequence
```

## 9. CAS and return correlation

`0x1f9cd72` captures CAS pre only when T1 is selected and all target/thread/token/minReader conditions still hold.

At `0x1f9ce04`, the script classifies the selected return by exact dequeue position:

```text
dequeuePos == ticket + 1 -> C3_T1_CAS_SUCCEEDED
dequeuePos == ticket     -> C3_T1_EPOCH_BLOCKED
otherwise                -> C3_CR1_STOP_T1_RETURN_POSITION_CONTRADICTION
```

If T1 was not selected, the same return completes the provisional candidate as `C3_CANDIDATE_NO_TARGET` without assigning a Case A/B/C/D verdict.

After successful CAS, the entry may be cleared, so return attribution intentionally uses the selected T1 latch, thread, DQueue, and exact dequeue position rather than requiring cleared entry fields.

## 10. T2 terminal anchor

T2 does not require `candidateActive`, T1 success, or any `$t0` phase value.

It requires:

```text
targetArmed == 1
T2 not already captured
S7 anchor thread diagnostic exists
current thread == S7 thread
engine-derived domain == target EpochDomain
```

This binds the terminal anchor to the same S7 waitForDrain return rather than the first unrelated `waitForDrain()` call in the process.

The required final log order remains:

```text
T0 sequence publication < T1_SELECTION < T2
```

The script captures the order; static validation does not claim that the order has already been observed at runtime.

## 11. ReaderSlot validation

Four contractual blocks are present:

| block | literal reads | stride | role |
|---|---:|---:|---|
| T0 | 64 | `0x50` | auxiliary enqueue snapshot |
| S7 | 64 | `0x50` | auxiliary terminal snapshot |
| T1-B pre-call | 64 | `0x50` | contractual getMinReaderEpoch snapshot |
| T2 | 64 | `0x50` | final-state comparison |

Total: `256` literal `L80` reads.

There is no `.for`, `.while`, dynamic ReaderSlot index, or helper expansion at runtime.

## 12. Static syntax and forbidden constructs

| check | result |
|---|---|
| quote balance | PASS |
| brace balance | PASS |
| explicit MASM evaluator selection | PASS |
| C++ `&&` / `||` operators | 0 |
| MASM `and` joins | 7 |
| custom pseudo-register | 0 |
| `.for` / `.while` | 0 |
| expression assignment/side effect | 0 |
| assignment destination form | `r $tN = ...` only |
| reads use `@$tN` | PASS |
| breakpoint count | 13 |
| breakpoint RVA drift | 0 |
| one Redesign-2 CDB file | PASS |

CDB syntax was checked statically against Microsoft-documented command/MASM forms. Loading the script into CDB was intentionally not performed in this preparation gate; that remains part of the separately gated runtime-authorization review.

## 13. STOP path validation

The script contains the following hard-stop classes, each echoing `C3_CR1_STOP_*`, setting `$t18=1`, dumping registers, and quitting:

```text
NESTED_OR_CONCURRENT_CANDIDATE
T1B_THREAD_TOKEN_JOIN
TARGET_DOMAIN_MISMATCH
CANDIDATE_THREAD_MISMATCH
T1C_DQUEUE_ENTRY_JOIN
T1C_THREAD_TOKEN_R13_JOIN
MULTIPLE_TARGET_SELECTION
TARGET_IDENTITY_CONFLICT
T1_CAS_CORRELATION
T1_RETURN_POSITION_CONTRADICTION
T1_RETURN_THREAD_DOMAIN
```

`C3_CR1_STOP` marker occurrences: 37 due to duplicated terminal `.else` paths emitted by the deterministic guard builder. This is parser-tree duplication, not runtime invocation; each path writes `$t18=1` and quits.

## 14. Mechanical validation result

```text
scriptCount                       PASS
breakpointCount                   PASS
breakpointRvas                    PASS
quoteBalance                      PASS
braceBalance                      PASS
masmOnly                          PASS
pseudoRegisterRange               PASS
customPseudoZero                  PASS
loopZero                          PASS
expressionSideEffectZero          PASS
literalSlotBlocks                 PASS
literalSlotTotal                  PASS
targetArmedOnlyAtPublication      PASS
targetRingAddress                 PASS
candidateThreadToken              PASS
getMinPre64Slots                  PASS
minReaderOnlyFromRAX              PASS
targetGateJoin                    PASS
selectionUnique                   PASS
predecessorRetained               PASS
casReturnCorrelation              PASS
t2Independent                     PASS
stopLatch                         PASS
oldChronologyZero                 PASS
minReconstructionZero             PASS
t19DiagnosticOnly                 PASS

TOTAL = 27/27 PASS
```

## 15. Scope verification

| prohibited action | result |
|---|---|
| CDB Retry-2 | NOT PERFORMED |
| Harness execution | NOT PERFORMED |
| M1/M2 | NOT PERFORMED |
| Dr.Memory | NOT PERFORMED |
| build | NOT PERFORMED |
| production source edit | NOT PERFORMED |
| test source edit | NOT PERFORMED |
| CMake edit | NOT PERFORMED |
| instrumentation/telemetry addition | NOT PERFORMED |
| source/EXE/PDB identity drift | NOT TRIGGERED |
| new CDB script count other than 1 | NOT TRIGGERED |

The working tree already contained unrelated source/test modifications before this gate. This gate did not edit or overwrite them.

## 16. STOP-R2 matrix

| stop | result | reason |
|---|---|---|
| STOP-R2-1 pseudo-register shortage/collision | NOT TRIGGERED | exact `$t0..$t19` map and writer boundaries validated |
| STOP-R2-2 CDB parser ambiguity | NOT TRIGGERED statically | MASM form, quotes, braces, assignments, and operators pass static validation; CDB load is deferred by scope |
| STOP-R2-3 breakpoint RVA mismatch | NOT TRIGGERED | exact 13-RVA sequence matches frozen EXE |
| STOP-R2-4 RAX/R13 liveness contradiction | NOT TRIGGERED | return RAX and same-invocation R13 joins are explicit |
| STOP-R2-5 target identity join unavailable | NOT TRIGGERED | ptr/deleter/epoch/ticket/sequence/DQueue/domain all joined |
| STOP-R2-6 literal ReaderSlot conversion failure | NOT TRIGGERED | four blocks x 64 reads validated |
| STOP-R2-7 safe C3_CR1_STOP unavailable | NOT TRIGGERED | 11 stop classes validated |
| STOP-R2-8 source/EXE identity drift | NOT TRIGGERED | frozen hashes match |
| STOP-R2-9 unvalidated validator item | NOT TRIGGERED | 27/27 checks pass |
| STOP-R2-10 execution required | NOT TRIGGERED | no execution performed or needed to close preparation/static validation |

## 17. Final disposition

```text
P3-5-FPM-C3-T1-Capture-Redesign-2 = SCRIPT READY / STATIC VALIDATED
Script SHA-256                   = FBE32851F795AF900E6C46D6E38F25138E01217C2B52C0BDDE1F015029CFC39E
Harness execution                = 0
Retry-2                          = 0 / NOT AUTHORIZED
S7_READER                        = UNRESOLVED
Case A/B/C/D                     = NOT_PROVEN
IMPLEMENTATION                   = FORBIDDEN
```

The next permissible gate is `P3-5-FPM-C3-T1-Capture-Redesign-3` (Runtime Authorization Review). Passing this preparation gate does not authorize Retry-2.
