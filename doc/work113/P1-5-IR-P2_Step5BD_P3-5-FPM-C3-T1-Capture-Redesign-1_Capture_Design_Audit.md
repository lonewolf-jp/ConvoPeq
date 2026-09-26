# P3-5-FPM-C3-T1-Capture-Redesign-1 — Identity-Correlated T1 Capture Design Audit

## 1. Gate and verdict

```text
Gate                         = P3-5-FPM-C3-T1-Capture-Redesign-1
Mode                         = read-only debugger-script design audit
Purpose                      = design a valid first-target-reclaim capture without runtime execution
Production source changes    = 0
Test source changes          = 0
CMake changes                = 0
Build                        = 0
Harness execution            = 0
CDB Retry-2 execution        = 0
M1/M2                        = 0
Dr.Memory                    = 0
Revised CDB script generated = 0
T1 chronology state removed  = PASS
Target-entry correlation key = DEFINED
getMin same-invocation join  = DEFINED
DQueue epoch-gate join       = DEFINED
T2 terminal anchor           = DEFINED
Gate verdict                 = CLOSED / DESIGN VALIDATED
Retry-2 authorization        = NOT GRANTED
S7_READER                    = UNRESOLVED
Case A/B/C/D                 = NOT_PROVEN
IMPLEMENTATION               = FORBIDDEN
```

This gate establishes a debugger capture design only. `Case P` from the prior audit means the old debugger state gate was incompatible with the real chronology. It does **not** mean DQueue Case A, B, C, or D.

The designed capture replaces `state == 2 / state == 3` chronology gating with runtime identity correlation:

```text
target T0 identity
  + target DQueue/domain
  + same-thread invocation latch
  + first target-head epoch-gate observation
```

The design is sufficient to prepare a future script, but it does not create that script and does not authorize Retry-2.

## 2. Frozen inputs

| artifact | SHA-256 | bytes | result |
|---|---|---:|---|
| `ConvoPeq.md` | `E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609` | 5,535,334 | PASS |
| `AudioEngineHarness.exe` | `E5C7AFB9C4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75` | 41,216,512 | PASS |
| `AudioEngineHarness.pdb` | `A793738F12F9BA3E0C7A19981B35212659107A356C6904C2BF809933CE1F4391` | 59,355,136 | PASS |
| C3-RP frozen script | `AFCC0017DA95D36F23D497123547AEE0B0B8D7572F43231835A25AE532EF0F18` | 11,150 | PASS, unchanged |
| C3-Retry-1 CDB log | `1F5109C6113F718AD5E50704056D108B5184042BEFA93AB0EEA3967E1B98BBE2` | 178,875 | PASS |
| T1-Path-Audit-1 report | `E64A6408964495F9870D1443AD348A3929655E18F302A3C3F9A8B419DF87F2F0` | 26,286 | PASS |

CDB/Harness process residue at design-audit completion: zero.

`ConvoPeq.md` is the current production source authority. The user clarification that “`ConvoPeq(3).md`” refers to this root file is applied.

## 3. Frozen evidence retained

The redesign uses the following already-proven facts and does not reinterpret them:

```text
T0 ticket                  = 4, PROVEN
T0 published sequence      = 5, PROVEN
T0 target ptr              = 0x0000017655A01080, PROVEN for Retry-1
T0 target deleter          = 0x00007FF759A979C0, PROVEN for Retry-1
T0 target epoch            = 9, PROVEN
S7 dequeuePos              = 4, PROVEN
S7 enqueuePos              = 5, PROVEN
S7 head sequence           = 5, PROVEN
S7 head ptr/deleter/epoch  = same as T0, PROVEN
T0/S7 same entry           = PROVEN
T0/S7/T2 ReaderSlot bytes  = identical and inactive
T1 runtime capture         = NOT CAPTURED
minReaderEpoch at T1       = NOT CAPTURED / MUST NOT BE RECONSTRUCTED
```

A future script must derive all target values from a new T0 runtime observation. The pointer values above are prior-run evidence, not hardcoded identity for a future process.

## 4. Design objective

The capture must observe this temporal chain:

```text
T0 enqueue
  -> enqueuePos CAS 4 -> 5
  -> target entry fields written
  -> sequence release-publication 5
  -> target identity armed
  -> first subsequent Engine-domain tryReclaim candidate
  -> getMinReaderEpoch pre-snapshot
  -> getMinReaderEpoch return
  -> DQueue scans FIFO entries
  -> first epoch gate whose entry is the target entry
  -> target head/minReader comparison
  -> dequeue CAS or blocked-return
  -> T2 wait-return terminal anchor
```

The target is not merely “the first tryReclaim after T0.” It is the first same-DQueue reclaim invocation that actually reaches ticket 4’s epoch gate.

## 5. Why the old chronology gate must be removed

The old script used:

```text
S7 marker -> $t0 = 2
T1-A requires $t0 == 2
T1-B/T1-C require $t0 == 3
T2 marker -> $t0 = 4
```

The source and EXE chronology prove:

```text
post-T0 reclaim executes while $t0 == 1
S7 and T2 are adjacent with no production call between them
post-T2 reclaim executes while $t0 == 4
```

Therefore states 2/3 do not describe the desired production chain. They must not be translated to a new numeric state schedule.

The replacement uses boolean/latch values that represent identity and invocation correlation, not chronology phases:

```text
targetArmed
candidateActive
candidateThread
invocationToken
t1Selected
t2Captured
anomaly
```

## 6. Target identity definition

### 6.1 Primary key

The following values are captured from the new T0 enqueue and then held unchanged:

```text
K.process       = current CDB process
K.dqueue        = RCX at target DQueue enqueue
K.epochBase     = DQueue - 0x1440
K.domain        = epochBase + 0x10
K.ticket        = enqueue position before successful CAS
K.entry         = RDX
K.deleter       = R8
K.epoch         = R9
K.publicationId = entry.publicationSequenceId
K.generation    = entry.generation
K.expectedSeq   = ticket + 1
```

All 64-bit values are stored as low/high pseudo-register pairs where required for exact comparison.

### 6.2 Why pointer + deleter + epoch + sequence are required

Ticket and epoch alone are insufficient because queue positions and epochs are reused over time. The pointer/deleter pair provides payload identity; expected sequence and DQueue/domain provide queue-slot identity.

A later payload may reuse an address. The hard conflict rule is:

```text
same ticket/sequence
AND different ptr or deleter or epoch
    => anomaly / hard STOP
```

If a different entry is at the current head, it is not automatically the target. It is scanned as an earlier FIFO entry within the same invocation until the target is reached or the invocation ends.

## 7. Arming boundary

T1 monitoring is armed only after the target enqueue has published `sequence=5`.

The current EXE enqueue sequence is:

```text
0x1f9c57d  lock cmpxchg enqueuePos 4 -> 5
0x1f9c59a  entry fields begin to be written
0x1f9c5d2  mov sequence, 5
0x1f9c5da  first instruction after sequence publication
```

The existing `T0_WRITE_DONE` breakpoint at `0x1f9c5da` is the correct arming point. At that instruction, before execution:

```text
RBX  = ticket 4
R10  = ticket + 1 = 5
R11  = DQueue
RDX  = target ptr
R8   = target deleter
R9   = target epoch 9
sequence[4] = 5 is already visible in memory
```

The redesign must initialize the target key and set `targetArmed=1` at this boundary. Earlier T0 entry/CAS/write markers remain evidence but must not arm T1.

Arming after the sequence store closes the race in which another thread could observe an incomplete target entry.

## 8. Candidate invocation correlation

### 8.1 Why target identity alone is insufficient

Multiple threads may execute `EpochDomain::tryReclaim` on the same Engine domain. A global pseudo-register latch without a thread key could mix T1-A from one invocation with getMin or DQueue evidence from another.

The correlation key is therefore:

```text
(process, thread, target DQueue, target domain, target entry, invocation token)
```

`$tpid` or `$tid` is available in this live user-mode CDB session. `$tpid` identifies the process; `$tid` identifies the current thread. The future script must use a process key plus one thread key, not only a breakpoint-local value.

### 8.2 Candidate admission

At `EpochDomain::tryReclaim` entry (`0x1f9cfb0`), a candidate is admitted only if:

```text
targetArmed == 1
RCX == targetDomain
candidateActive == 0
T1 has not already been selected
T2 has not already closed the accepted chronology
```

Admission captures:

```text
candidateThread = current thread id
candidateToken  = monotonically incremented token
candidateDQueue = targetDomain + 0x1430
candidateHeadPtr/deleter/epoch/sequence/dequeuePos
```

If another thread already owns `candidateActive`, the new entry is not allowed to overwrite the latch. It is counted as a concurrent invocation. If the active invocation later reaches the target gate, a concurrent invocation observed before that gate is an ambiguity and must terminate the future capture as STOP-CR2. If the active invocation does not reach the target, the concurrent invocation remains diagnostic-only and monitoring continues.

A nested same-thread `tryReclaim` while `candidateActive==1` is structurally anomalous for this non-reentrant reclaim chain and is a hard ambiguity.

### 8.3 First relevant tryReclaim

“First relevant” is defined as:

> The first admitted same-domain invocation after target arming that reaches an epoch-gate instruction whose DQueue head is the exact target entry.

A tryReclaim whose entry head is earlier than ticket 4 is not immediately discarded. The DQueue implementation is FIFO and may reclaim earlier entries and then continue to ticket 4 in the same invocation. The candidate latch therefore remains active through getMin and the DQueue scan.

## 9. T1-A — EpochDomain tryReclaim entry

### 9.1 Capture point

```text
RVA = 0x1f9cfb0
function = EpochDomain::tryReclaim
entry object register = RCX
DQueue object = RCX + 0x1430
epochBase = RCX - 0x10
```

### 9.2 Required output

At the candidate entry, record:

```text
process id
thread id
invocation token
EpochDomain object
DQueue object
epochBase
globalEpoch
enqueuePos
dequeuePos
head sequence
head ptr
head deleter
head epoch
head type
head publicationSequenceId
head generation
target ptr/deleter/epoch/ticket/expected sequence
```

### 9.3 Head classification

| observed head | classification | action |
|---|---|---|
| exact target | `TARGET_AT_ENTRY` | keep candidate; T1 remains selected only after getMin + DQueue gate join |
| same ticket/sequence, different identity | `TARGET_IDENTITY_CONFLICT` | hard anomaly; print and terminate future capture |
| earlier valid FIFO entry | `PREDECESSOR_HEAD` | keep candidate; same invocation may later reach target |
| empty/invalid sequence | `EMPTY_OR_NOT_READY` | capture diagnostic; candidate completes without T1 selection |
| different DQueue/domain | `OTHER_DOMAIN` | not admitted |

The first T1-A output is therefore a **provisional candidate entry**. The final T1 record is selected only when the same invocation reaches the target epoch gate.

## 10. T1-B — getMinReaderEpoch pre/return

### 10.1 Pre-call point

```text
RVA = 0x1f9cfe2
instruction = call qword ptr [rax+0x40]
RBX = EpochDomain object
RCX = adjusted vtable object
```

The breakpoint fires before the call executes.

Required output:

```text
candidate token/thread/domain/DQueue identity
globalEpoch
ReaderSlot[0..63]
```

The 64-slot block remains exactly 64 literal windows of `0x80` bytes with `ReaderSlot stride=0x50`. The 64 slots must not be derived through a CDB loop.

The pre-call snapshot is immediately before the same invocation’s getMin call. No T0/S7/T2 snapshot is substituted.

### 10.2 Return point

```text
RVA = 0x1f9cfe5
instruction = mov rdx, rax
RAX = getMinReaderEpoch return value
```

The breakpoint fires after the call and before `RAX` is moved to `RDX`. The design captures `RAX` as two 32-bit halves:

```text
minReaderEpochLow  = RAX & 0xffffffff
minReaderEpochHigh = RAX >> 32
```

A second 64-slot snapshot is optional for drift diagnosis but cannot replace the pre-call snapshot. The pre-call snapshot is the contractual temporal observation.

### 10.3 Same-invocation proof

T1-B is accepted only if all are true:

```text
candidateActive == 1
current thread == candidateThread
RBX == targetDomain
current invocation token == candidateToken
```

A different thread, domain, or token is skipped. It must not update the selected minReader value.

## 11. T1-C — target DQueue epoch gate

### 11.1 Capture point

```text
RVA = 0x1f9cd59
instruction = mov rax, [rsi+0x10]
```

At this instruction, before `RAX` is loaded:

```text
RBP   = current scanPos / dequeue position
R13   = minReaderEpoch for the same DQueue invocation
R15   = address of sequence[scanPos & 0xfff]
RSI   = address of current ring entry
RDI   = DQueue object
```

Required values are read from their source locations, not inferred:

```text
dequeuePos/scanPos   = RBP & 0xffffffff
expectedSequence     = scanPos + 1
headSequence         = dword [R15]
headPtr              = qword [RSI+0x00]
headDeleter          = qword [RSI+0x08]
headEpoch            = qword [RSI+0x10]
minReaderEpoch       = R13
headType             = byte [RSI+0x18]
publicationSequenceId= qword [RSI+0x20] when diagnostics enabled
generation           = qword [RSI+0x28] when diagnostics enabled
```

The runtime predicate is:

```text
isOlder(headEpoch, minReaderEpoch)
    = (headEpoch - minReaderEpoch) interpreted as signed int64 < 0
```

CDB should print the two operands and may compute a signed-difference diagnostic, but the report must classify from the captured values rather than from a previous run.

### 11.2 Target selection

This point is the decisive T1 selection point. It is T1 only if:

```text
candidateActive == 1
current thread == candidateThread
RDI == targetDQueue
R13 low/high == candidate minReader
RSI head ptr/deleter/epoch == target ptr/deleter/epoch
headSequence == target expectedSequence
scanPos == target ticket
```

At selection, record:

```text
T1_SELECTION
T0 identity
candidate entry identity
minReaderEpoch from the same getMin invocation
head entry identity
head sequence/ticket
signed epoch difference
isOlder result
dequeue CAS pre state
```

The value in `R13` is not taken from a pseudo-register reconstructed from T0/S7. It is the compiler-carried result of the getMin call in the same function invocation.

### 11.3 CAS and return points

Retain:

```text
0x1f9cd66  epoch-gate branch
0x1f9cd72  dequeue CAS pre
0x1f9ce04  reclaim return / blocked-or-empty return
```

The CAS and return payloads must require the same target identity and invocation token. If target epoch is not older, the expected result is blocked return with dequeuePos unchanged. If it is older, the expected result is CAS advance and reclamation. A different result is a runtime contradiction, not a Case classification.

## 12. Nonmatching and conflict behavior

### 12.1 Normal nonmatching cases

| case | treatment |
|---|---|
| before target arming | ignore |
| different EpochDomain/provider | ignore as out of target scope |
| same domain, earlier head entry | keep one provisional candidate; continue same invocation |
| empty/not-ready head | complete candidate without selecting T1; continue monitoring |
| sequence gate fails before target | complete candidate without selecting T1; continue monitoring |

### 12.2 Hard conflicts

The following are not skipped because skipping could hide target confusion:

```text
same target ticket/sequence with different ptr
same target ptr with different deleter or epoch
target DQueue/domain mismatch
T1-B on a different thread or invocation token
T1-C on a different DQueue or token
two target epoch-gate observations attributed to one invocation
nested same-thread candidate while latch is active
target observed at entry, then absent without explicit CAS evidence
```

On a hard conflict, a future script must emit `C3_CR1_STOP`, print all correlation fields, and terminate without modifying the target or continuing to a later invocation.

## 13. T2 terminal anchor

### 13.1 Purpose

T2 remains a chronology anchor, not a T1 phase.

```text
RVA = 0x1fa80b2
function = AudioEngine::waitForDrain return
```

T2 is captured once after target arming, using only target identity plus the existing Engine-derived DQueue/domain layout. It does not require `candidateActive`, `$t0==2`, `$t0==3`, or T1 success.

Required T2 output remains:

```text
ReaderSlot[0..63]
globalEpoch
enqueuePos
dequeuePos
head sequence
head ptr/deleter/epoch/type/publicationSequenceId/generation
```

### 13.2 Ordering contract

The accepted chronology is valid only if log order is:

```text
T0 write-done / sequence publication
  < T1 target epoch-gate selection
  < T2 wait-return anchor
```

If T2 arrives before a target T1 selection, the future capture records T2 and marks the accepted T1 chronology unresolved. A later target reclaim may be captured diagnostically, but it cannot be substituted as “T1 before T2” without a new gate decision.

If T1 is never selected, T2 remains valid final-state evidence but the gate is STOP-CR1/STOP-CR2, not a successful T1 capture.

## 14. Pseudo-register allocation

The design uses only documented user-defined pseudo-registers `$t0` through `$t19`.

| register(s) | role |
|---|---|
| `$t0` | `targetArmed` boolean |
| `$t1` | target epochBase |
| `$t2` | target DQueue object |
| `$t3` | target EpochDomain object |
| `$t4`, `$t5` | target ptr low/high |
| `$t6`, `$t7` | target deleter low/high |
| `$t8`, `$t9` | target epoch low/high |
| `$t10` | target ticket; expected sequence is defined as `ticket + 1` |
| `$t11` | candidate thread id (`$tid`) |
| `$t12` | `candidateActive` boolean |
| `$t13` | candidate invocation token |
| `$t14`, `$t15` | same-invocation minReaderEpoch low/high |
| `$t16` | `t1Selected` boolean |
| `$t17` | `t2Captured` boolean |
| `$t18` | anomaly latch/counter |
| `$t19` | reserved diagnostic value; initialized to zero and may be updated only by an explicit diagnostic marker |

Process identity is fixed by the single CDB process session and is printed in every marker; no separate pseudo-register is consumed. `publicationSequenceId` and `generation` are read and printed from the target/head entry as correlation diagnostics, but they are not latch inputs. This keeps the mandatory same-invocation minReader pair within `$t0..$t19` without overlapping assignments.

A future validator must still ensure each listed role is assigned only by its declared boundary. The design does not use `$s7`, `$s7e`, `$s7q`, `$s7rbx`, or any custom pseudo-register name.

No phase values `2` or `3` are assigned to T1 chronology. Integer values in pseudo-registers represent pointers, halves, tokens, thread identity, and booleans only.

## 15. CDB syntax constraints for future preparation

Microsoft documentation confirms:

```text
$t0..$t19 are the 20 writable user-defined pseudo-registers
r $tN = value is the supported assignment form
MASM expressions may use @$tN
C++ expressions require @$tN
r command does not accept @ before the destination pseudo-register
$tid/$tpid are available in a live user-mode session
```

The future script must use MASM consistently:

```text
r $t4 = @rdx
.if (@$t4 == $targetPtrLow) { ... }
```

C++ side effects in `.if` expressions are forbidden. Assignments occur only through `r`.

The following are forbidden in the future script:

```text
custom pseudo-register names
expression assignment or side effect
.for / .while
unbounded CDB loops
complex runtime address expressions
non-literal ReaderSlot access
using a different provider/domain
reconstructing minReaderEpoch from T0/S7/T2
```

## 16. Alternative designs and rejection rationale

### Alternative A — keep S7 → T1 state gating

**Rejected.** The prior audit proved there is no production reclaim instruction between S7 and T2. This design reproduces the original zero-hit failure.

### Alternative B — allow T1 only before S7

**Rejected.** The first relevant reclaim can occur after T0 and before S7, but tying T1 to an artificial pre-S7 window depends on marker order rather than target identity and can miss a valid later first target reclaim.

### Alternative C — capture every `tryReclaim` without identity

**Rejected.** SnapshotCoordinator, EQProcessor, and other domains may use the same function RVAs. Multiple same-domain invocations can also occur. Unfiltered capture cannot prove same-invocation or same-entry attribution.

### Alternative D — use only pointer + epoch

**Rejected.** Pointers and epochs can be reused. Sequence/ticket/deleter and DQueue/domain are required to prevent a later entry from being confused with ticket 4.

### Alternative E — discard a reclaim when its entry head is not target

**Rejected.** The DQueue is FIFO. One invocation may reclaim predecessor entries and then reach ticket 4. The candidate must remain correlated through the same DQueue scan.

### Alternative F — use production instrumentation

**Rejected and forbidden.** The existing source/EXE exposes every required boundary without telemetry or getter changes.

### Alternative G — reuse a global boolean without thread/token correlation

**Rejected.** Concurrent invocations can overwrite or merge evidence. Process, thread, domain, and invocation token are part of the join contract.

## 17. Acceptance criteria for future script preparation

The redesign is complete when a future separately authorized script-preparation gate proves all of the following without running the Harness.

### Happy path

- Given target sequence 5 is published, when the same-domain tryReclaim entry is reached, then the candidate key contains the new runtime target identity and the same thread.
- Given a provisional candidate is active, when the same thread reaches getMin pre, then exactly 64 ReaderSlots and globalEpoch are captured.
- Given getMin returns on that same thread/token, when `0x1f9cfe5` is reached, then `RAX` is stored as the candidate minReaderEpoch.
- Given the same DQueue scan reaches ticket 4, when `0x1f9cd59` is reached, then target ptr/deleter/epoch/sequence and candidate minReader are proven from the same invocation.
- Given T1 was selected, when T2 return is later reached, then the final target-domain state is captured as the terminal anchor.

### Edge cases

- Given an earlier FIFO head, when a provisional candidate is active, then it remains correlated and does not select T1 until the target head is reached.
- Given an empty/not-ready head, when reclaim returns, then the candidate is abandoned without selecting T1 and monitoring continues.
- Given a different domain/provider, when tryReclaim is reached, then it is ignored.
- Given T2 arrives without prior T1 selection, when the log is closed, then the chronology is STOP/unresolved, not backfilled.

### Error states

- Given same ticket/sequence but different entry identity, when observed, then `C3_CR1_STOP` is emitted.
- Given a different thread/token reaches a selected T1-B or T1-C point, then evidence is not joined.
- Given a nested same-thread invocation occurs while the candidate is active, then `C3_CR1_STOP` is emitted.
- Given any CDB expression/register error occurs, then the future run terminates immediately and no in-place script repair/retry is allowed.

### Auditability

- Every marker includes process, thread, candidate token, domain, DQueue, and target identity.
- Every selected T1-C record names the pre-call snapshot and getMin return used in the join.
- T1 and T2 log line numbers establish temporal order.
- The script fingerprint records breakpoint count, allowed pseudo-registers, 64 literal slot blocks, forbidden constructs, and SHA-256.

## 18. STOP-condition matrix for this design gate

| stop | result | reason |
|---|---|---|
| STOP-CR1 target entry cannot be uniquely tracked | NOT TRIGGERED | pointer + deleter + epoch + sequence + ticket + DQueue/domain key is defined |
| STOP-CR2 T1 invocation cannot be joined to target | NOT TRIGGERED | process/thread/token/domain plus target epoch-gate selection is defined |
| STOP-CR3 getMin return cannot be joined to invocation | NOT TRIGGERED | pre/return RVAs and RAX/R13 liveness are proven from EXE CFG |
| STOP-CR4 ReaderSlot snapshot cannot align with minReader | NOT TRIGGERED | pre-call snapshot and same-thread/token return are a closed pair |
| STOP-CR5 capture points cannot be safely defined | NOT TRIGGERED | current EXE contains the required entry/getMin/gate/CAS/return/T2 instructions |
| STOP-CR6 production instrumentation required | NOT TRIGGERED | debugger-only design is sufficient |

No source change or runtime execution is required to resolve the design-level unknowns.

## 19. Prohibition verification

| prohibition | result |
|---|---|
| revised CDB script creation | NOT PERFORMED |
| CDB Retry-2 | NOT PERFORMED |
| Harness execution | NOT PERFORMED |
| production source edit | NOT PERFORMED |
| test source edit | NOT PERFORMED |
| CMake edit | NOT PERFORMED |
| getter/counter/telemetry addition | NOT PERFORMED |
| build | NOT PERFORMED |
| M1/M2 | NOT PERFORMED |
| Dr.Memory | NOT PERFORMED |
| minReaderEpoch reconstruction | EXPLICITLY REJECTED |
| Case A/B/C/D selection | NOT PERFORMED |
| implementation authorization | FORBIDDEN |

## 20. Final gate result

```text
P3-5-FPM-C3-T1-Capture-Redesign-1 = CLOSED / DESIGN VALIDATED
T1 chronology state dependency     = REMOVED IN DESIGN
Target correlation key            = DEFINED
First relevant reclaim            = DEFINED
getMin same-invocation join        = DEFINED
T1-C epoch-gate join               = DEFINED
T2 terminal anchor                 = DEFINED
Required capture points safe       = YES
Production instrumentation needed  = NO
Retry-2 script created             = NO
Retry-2 execution authorized       = NO
S7_READER                         = UNRESOLVED
Case A/B/C/D                      = NOT_PROVEN
IMPLEMENTATION                    = FORBIDDEN
```

The next permissible action is a separate debugger-script preparation/static-validation gate. Runtime authorization must remain separate and must not be inferred from this design closure.

## 21. Authoritative debugger references

- Microsoft Learn, Pseudo-Register Syntax: `https://learn.microsoft.com/en-us/windows-hardware/drivers/debuggercmds/pseudo-register-syntax`
- Microsoft Learn, Registers: `https://learn.microsoft.com/en-us/windows-hardware/drivers/debuggercmds/r--registers-`
- Microsoft Learn, Registers and pseudo-registers in C++ expressions: `https://learn.microsoft.com/en-us/windows-hardware/drivers/debuggercmds/c---numbers-and-operators#registers-and-pseudo-registers-in-c-expressions`
