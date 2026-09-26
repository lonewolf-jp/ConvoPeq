# P3-5-FPM-C3-T1-Path-Audit-1 — T1 Zero-Hit Source / Symbol / Disassembly Audit

## 1. Gate and verdict

```text
Gate                         = P3-5-FPM-C3-T1-Path-Audit-1
Mode                         = read-only source/static audit
Purpose                      = explain C3-Retry-1 T1 marker 0-hit without another runtime run
Production source changes    = 0
Test source changes          = 0
CMake changes                = 0
Build                        = 0
Runtime retry                = 0
M1/M2                        = 0
Dr.Memory                    = 0
T1 zero-hit classification   = Case P
Subcase                      = state-gated capture sequencing failure
Downstream disposition       = PARTIALLY-LOCALIZED / STOP
S7_READER                    = UNRESOLVED
Case A/B/C/D                 = NOT_PROVEN
IMPLEMENTATION               = FORBIDDEN
```

The T1 zero-hit is explained without a new execution: the frozen script enables T1 only in state `2` at `EpochDomain::tryReclaim` and state `3` after that entry. The M0 production reclaim calls relevant to this execution occur either **before S7** while state is `1`, or **after T2** while state has already been set to `4`. There is no production reclaim instruction between S7 and T2. Consequently, all T1 conditional command strings evaluate to their `gc` fallback and emit no T1 marker.

This is classified as **Case P — debugger capture-point/state-mapping failure**, specifically a state-machine sequencing defect in the CDB command conditions. It is not Case Q, R, or S.

No `minReaderEpoch` value is reconstructed or inferred. The existing C3-Retry-1 value remains forbidden for attribution.

## 2. Scope and hard boundaries

This audit performs only source, frozen-log, PDB-record, and EXE-disassembly analysis.

The following were not performed:

```text
CDB Retry-2                       NOT PERFORMED
M1 / M2                           NOT PERFORMED
production source modification   NOT PERFORMED
test instrumentation              NOT PERFORMED
getter/counter/telemetry add     NOT PERFORMED
shutdown/EpochDomain/reclaim edit NOT PERFORMED
router edit                       NOT PERFORMED
CMake edit                        NOT PERFORMED
build                             NOT PERFORMED
Dr.Memory                          NOT PERFORMED
debugger execution of Harness     NOT PERFORMED
```

The only durable artifact created by this gate is this report. The pre-existing working-tree modifications remain outside this gate and are not represented as audit-created changes.

## 3. Input identity and freshness

| artifact | SHA-256 / identity | bytes | UTC mtime | result |
|---|---|---:|---|---|
| `ConvoPeq.md` | `E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609` | 5,535,334 | 2026-09-23T14:33:38.6388122Z | PASS |
| `AudioEngineHarness.exe` | `E5C7AFB9C4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75` | 41,216,512 | 2026-09-23T16:00:47.6227315Z | PASS |
| `AudioEngineHarness.pdb` | `A793738F12F9BA3E0C7A19981B35212659107A356C6904C2BF809933CE1F4391` | 59,355,136 | 2026-09-22T12:50:18.5624865Z | PASS |
| frozen retry script | `AFCC0017DA95D36F23D497123547AEE0B0B8D7572F43231835A25AE532EF0F18` | 11,150 | 2026-09-25T11:51:52.3879283Z | PASS |
| C3-Retry-1 CDB log | `1F5109C6113F718AD5E50704056D108B5184042BEFA93AB0EEA3967E1B98BBE2` | 178,875 | 2026-09-25T12:13:32.5823901Z | PASS |
| prior Retry-1 report | `5C16F6709A803BD3A7321B79E0605AFF9292EF039CB72EEF68EF3B7169DABA52` | 13,479 | n/a | PASS |
| CDB binary | `5F54ABAFCA3AE5638BBF807D402FABB350A64575C1DFA9FBFC7F5732DF5BEE67` | 178,016 | n/a | PASS |

CDB version:

```text
cdb version 10.0.29617.1000
```

`ConvoPeq.md` is the current source authority for this audit. The user clarification that “`ConvoPeq(3).md`” means this root `ConvoPeq.md` removes the earlier filename ambiguity.

## 4. Frozen Retry-1 evidence

The following facts are frozen from the completed C3-Retry-1 capture and are not recomputed in this audit:

```text
T0 entry                      = PROVEN
T0 CAS enqueuePos 4 -> 5      = PROVEN
T0 write sequence 4 -> 5      = PROVEN
T0 entry epoch                = 9, PROVEN

S7 dequeuePos                 = 4, PROVEN
S7 enqueuePos                 = 5, PROVEN
S7 head sequence              = 5, PROVEN
S7 head epoch                 = 9, PROVEN

T0/S7 same entry              = PROVEN
T0/S7/T2 ReaderSlots          = complete and identical; all inactive

T1 marker set                 = 0
T2                            = CAPTURED
minReaderEpoch at T1          = NOT_CAPTURED / MUST NOT BE RECONSTRUCTED
```

The frozen log contains the following marker lines:

```text
log:292   C3_T0_ENTRY
log:828   C3_T0_CAS_PRE
log:841   C3_T0_CAS_POST
log:854   C3_T0_WRITE_DONE
log:872   C3_S7_MARKER
log:1405  C3_T2_WAIT_RETURN
log:1938  [DIAG] releaseResources: drain timeout reached, performing safe tryReclaim (drainAll skipped)
```

No `C3_T1_*` runtime marker is present.

Technical qualification: the script executes `bl` before `g` (log lines 98–112) but does not execute a post-run `bl`. Therefore the log independently proves **zero T1 marker output**, while a physical instruction-hit count is not separately preserved. This distinction strengthens rather than weakens the present diagnosis: the source and script state predicates statically explain why the command payloads suppress their markers.

## 5. Frozen script state contract

The retry script uses `$t0` as the capture state.

| transition | script condition/effect | resulting state |
|---|---|---:|
| initialization | `r $t0 = 0` | 0 |
| T0 enqueue entry | `@r9==9`, `enqueuePos==4`, then `r $t0=1` | 1 |
| S7 timeout marker | breakpoint `+0x1fa80b3`, unconditional after T0, then `r $t0=2` | 2 |
| T1 EpochDomain entry | breakpoint `+0x1f9cfb0`, requires `@$t0 == 2`, then sets `r $t0=3` | 3 only if captured |
| T1 getMin/DQueue | requires `@$t0 == 3` | 3 only |
| T2 wait return | breakpoint `+0x1fa80b2`, requires `2 <= $t0 < 4`, then `r $t0=4` | 4 |
| any later EpochDomain entry | T1 entry condition requires state `2` | rejected at state 4 |

The decisive commands are:

```text
T1 entry:  .if (@$t0 == 2) { .if (@rcx == @$t3) { ...markers... } .else { gc } }
T1 inner:  .if (@$t0 == 3) { ...markers... } .else { gc }
T2 return: .if (@$t0 >= 2) { .if (@$t0 < 4) { r $t0=4; ...T2 marker... } }
```

The pointer filter is not the cause of the miss:

```text
$t1 = epochBase
$t2 = dqueueBase = epochBase + 0x1440
$t3 = EpochDomain address = epochBase + 0x10
```

Those values agree with the source layout and the captured runtime identities.

## 6. Object provenance

### 6.1 Construction

`AudioEngine::AudioEngine()` constructs the router with the Engine-owned domain:

```cpp
// src/audioengine/AudioEngine.CtorDtor.cpp:26
, m_coordinator(m_epochDomain)

// src/audioengine/AudioEngine.CtorDtor.cpp:39
m_retireRouter = std::make_unique<convo::isr::ISRRetireRouter>(
    m_epochDomain, &worldRetirementReference_);
```

The member declarations are:

```text
src/audioengine/AudioEngine.h:5017  convo::EpochDomain m_epochDomain;
src/audioengine/AudioEngine.h:5022  std::unique_ptr<convo::isr::ISRRetireRouter> m_retireRouter;
```

### 6.2 Runtime layout cross-check

The captured object addresses are:

```text
epochBase       = 0x00000176447379740
EpochDomain     = 0x00000176447379750 = epochBase + 0x10
DQueue          = 0x0000017644737ab80 = epochBase + 0x1440
```

The source member order is:

```text
EpochDomain vptr
globalEpoch
ReaderSlot readers[64]
DeferredDeletionQueue deferredDeletionQueue
```

The captured offsets therefore confirm:

```text
$t3 = &m_epochDomain, not a pointer to a separate provider
$t2 = &m_epochDomain.deferredDeletionQueue
```

### 6.3 Provider dispatch

`ISRRetireRouter` delegates through `provider_`:

```cpp
// src/audioengine/ISRRetireRouter.cpp:386-394
void ISRRetireRouter::tryReclaim() noexcept
{
    assert(provider_ != nullptr);
    provider_->tryReclaim();
    drainQuarantineStore();
    drainEmergencyAndTerminal();
}
```

For this Engine, `provider_` was constructed from `m_epochDomain`. There is no alternate Engine provider, alternate Engine domain, or different DQueue in this dispatch path.

`SnapshotCoordinator` is also initialized with `m_epochDomain`, but it manages SnapshotCoordinator slots and is not the owner of the captured DQueue entry. Its calls are listed separately below to prevent queue conflation.

## 7. Production T1 caller inventory

### 7.1 Engine DQueue path

| caller | source | condition in M0 | object → provider → EpochDomain → DQueue | M0 relevance |
|---|---|---|---|---|
| `AudioEngine::tryReclaimResources` | `AudioEngine.Retire.cpp:35-43` | pending D count or router resident count is nonzero | `m_retireRouter` → `m_epochDomain` → Engine DQueue | production; recovery/Convolver release entry points |
| `AudioEngine::drainDeferredRetireQueues` | `AudioEngine.Retire.cpp:45-66` | shutdown drain always proceeds; non-shutdown path suppresses only when empty | same | **primary M0 shutdown path** |
| emergency boost inside `drainDeferredRetireQueues` | `AudioEngine.Retire.cpp:330-363` | saturated/protective mode, window and interval permit | same | conditional production path |
| `AudioEngine::runCoordinatorPhase` | `AudioEngine.Threading.cpp:362-383` | overflow ring reinjection > 0, then non-shutdown drain | same | stopped before terminal drain in M0 |
| `AudioEngine::releaseResources` graceful poll | `ReleaseResources.cpp:285-311` | every poll while pending/reader condition persists | same | runs before VerifyDrained/T0 region |
| `AudioEngine::releaseResources` graceful timeout | `ReleaseResources.cpp:313-339` | `waitedMs >= 5000` | same | conditional terminal path |
| `AudioEngine::releaseResources` unconditional drain | `ReleaseResources.cpp:373-376` | terminal pipeline reached | same | before final DSP retire/T0 |
| EmergencyDrain direct call | `ReleaseResources.cpp:383-404` | `isEmergencyDrainRequested()` | direct `m_epochDomain` → Engine DQueue | conditional; separate domain false |
| `AudioEngine::releaseResources` post-wait branch | `ReleaseResources.cpp:615-661` | `!drainedWithinBudget || !isFullyDrained()` | `drainDeferredRetireQueues(true)` then direct `m_epochDomain.tryReclaim()` | **log line 1938 proves this branch was entered after T2** |
| `AudioEngine::~AudioEngine` fallback poll | `CtorDtor.cpp:227-270` | pending D count or active reader count remains nonzero | same | runs after `releaseResources`; M0 log shows `routerPendingRetire=2` immediately before destructor |
| `ISRRetireRouter::enqueueRetire` QueuePressure recovery | `ISRRetireRouter.cpp:257-268` | initial enqueue fails and 500 ms cooldown permits | provider → Engine EpochDomain | not expected for successful ticket-4 enqueue |
| `ISRRetireRouter::enqueueWithRetry` | `ISRRetireRouter.cpp:321-336` | initial D enqueue fails; up to two retries | provider → Engine EpochDomain | not expected for successful ticket-4 enqueue |
| `ISRRetireRouter::tryReclaim` | `ISRRetireRouter.cpp:386-394` | every external router reclaim | provider → Engine EpochDomain | primary dispatch |

Direct `m_retireRouter->tryReclaim()` production occurrences are closed at:

```text
AudioEngine.Retire.cpp:42
AudioEngine.Retire.cpp:63
AudioEngine.Retire.cpp:357
AudioEngine.Threading.cpp:369
AudioEngine.CtorDtor.cpp:250
AudioEngine.Processing.ReleaseResources.cpp:306
AudioEngine.Processing.ReleaseResources.cpp:336
```

`AudioEngine::tryReclaimResources()` callers are production recovery and Convolver release paths:

```text
AudioEngine.Timer.cpp:1839  RecoveryAction::Recover
AudioEngine.Timer.cpp:1855  RecoveryAction::Restore
ConvolverProcessor.Lifecycle.cpp:226  ConvolverProcessor releaseResources
```

`ConvolverProcessor::releaseResources()` occurs in the terminal UI release sequence before the final DSP retire/T0. It is a valid Engine-domain reclaim caller, not a separate provider.

### 7.2 Internal provider dispatch inventory

`ISRRetireRouter.cpp` contains exactly three production `provider_->tryReclaim()` occurrences:

```text
line 263  QueuePressure forced reclaim
line 329  enqueueWithRetry retry reclaim
line 389  ISRRetireRouter::tryReclaim public method
```

All three reach the same Engine `m_epochDomain` for the captured M0 object.

### 7.3 getMinReaderEpoch inventory

| caller | source | target |
|---|---|---|
| `ISRRetireRouter::minReaderEpoch` | `ISRRetireRouter.cpp:219-223` | `provider_->getMinReaderEpoch()` |
| `AudioEngine::drainDeferredRetireQueues` | `AudioEngine.Retire.cpp:65` | router → Engine domain; result passed to `m_coordinator.reclaim` for SnapshotCoordinator |
| emergency boost | `AudioEngine.Retire.cpp:358` | same SnapshotCoordinator path |
| `EpochDomain::tryReclaim` | `EpochDomain.h:386-395` | local/virtual getMin result passed directly to Engine DQueue |
| `EpochDomain::collectDrainAudit` | `EpochDomain.h:465` | diagnostic read only |
| deprecated `EpochDomain::reclaimRetired` | `EpochDomain.h:555-559` | private compatibility implementation; not the active external path |
| `DeferredFreeThread` | `DeferredFreeThread.h:156` | provider passed by its own owner; not the Engine shutdown DQueue path |

The active Engine DQueue relationship is only:

```text
EpochDomain::tryReclaim
  -> getMinReaderEpoch()
  -> deferredDeletionQueue.reclaim(minReaderEpoch)
```

`m_coordinator.reclaim(m_retireRouter->getMinReaderEpoch())` is a subsequent SnapshotCoordinator action. It must not be counted as reclaim of the captured Engine DQueue head.

### 7.4 DQueue reclaim inventory

The only direct `deferredDeletionQueue.reclaim(...)` implementations in the active source are:

```text
EpochDomain.h:394  EpochDomain::tryReclaim active path
EpochDomain.h:558  private deprecated reclaimRetired compatibility path
```

No other production component directly calls the Engine `DeferredDeletionQueue` object.

### 7.5 Separate provider/domain exclusion

`EQProcessor` owns a different `EpochDomain` and calls it in its destructor:

```text
src/eqprocessor/EQProcessor.Core.cpp:158
src/eqprocessor/EQProcessor.Core.cpp:160
```

The M0 log shows `~EQProcessor` after the AudioEngine post-wait reclaim diagnostic. These calls are not the captured Engine DQueue and cannot explain the captured entry.

This rules out Case R for the target entry.

### 7.6 Test-stub separation

Only three test-side `getMinReaderEpoch` overrides were found:

```text
src/tests/D8_2_B_2_Tests.cpp:129
src/tests/invariant_INV3_INV5.cpp:88
src/tests/RetireGraceSemanticsTests.cpp:642
```

They are stubs and are excluded from the production inventory.

## 8. RVA provenance: source ↔ PDB ↔ EXE ↔ call target

### 8.1 PDB availability

The current PDB matches the frozen EXE identity. However, `llvm-pdbutil dump --symbols` and `dump --publics` do not expose private/local procedure names for:

```text
EpochDomain::tryReclaim
ISRRetireRouter::tryReclaim
getMinReaderEpoch
DeferredDeletionQueue::reclaim
```

`llvm-symbolizer` also returns `??:0:0` for the queried RVAs. Therefore a named-symbol three-way match is unavailable from this PDB. This is an evidence limitation, not a contradictory mapping.

The RVA mapping is instead closed by unique source structure, member offsets, generated instruction sequence, and direct call target.

### 8.2 `0x1f9cfb0` — EpochDomain tryReclaim

Source structure at `EpochDomain.h:385-396`:

```cpp
reclaimLocalCounter_.fetch_add(...)
if (localCount % 1024 == 0)
    reclaimAttemptCount_.fetch_add(...)
const auto n = deferredDeletionQueue.reclaim(getMinReaderEpoch());
reclaimSuccessCount_.fetch_add(n, ...);
```

EXE disassembly at `AudioEngineHarness+0x1f9cfb0`:

```asm
141f9cfb0  push  rbx
141f9cfb6  mov   rbx, rcx
141f9cfbf  xadd  dword ptr [rcx+0x35570], eax
141f9cfc8  test  eax, 0x3ff
141f9cfcf  add   qword ptr [rcx+0x35530], 0x400
141f9cfdb  lea   rcx, [rcx-0x10]
141f9cfdf  mov   rax, [rcx]
141f9cfe2  call  qword ptr [rax+0x40]
141f9cfe5  mov   rdx, rax
141f9cfe8  lea   rcx, [rbx+0x1430]
141f9cfef  call  0x141f9cd00
141f9cff7  add   qword ptr [rbx+0x35538], rax
141f9d003  ret
```

This is a unique match:

```text
+0x35570 counter update   = reclaimLocalCounter_
+0x35530 periodic update  = reclaimAttemptCount_
vtable +0x40 call          = getMinReaderEpoch()
call +0x1f9cd00            = DeferredDeletionQueue::reclaim()
+0x35538 result update     = reclaimSuccessCount_
```

The captured `$t3` (`epochBase+0x10`) is the `rcx` domain address used at the T1 entry breakpoint. The pointer condition is valid.

### 8.3 `0x1f9cfe2` / `0x1f9cfe5` — getMin call and return

These are not separate compiled function addresses. They bracket the virtual getMin call in the optimized `EpochDomain::tryReclaim` body:

```text
0x1f9cfe2 = call site
0x1f9cfe5 = first instruction after the call, with result in rax
```

This is an appropriate pre/return capture pair if the state predicate allows it. `getMinReaderEpoch` was not eliminated; its effect is required to produce the DQueue reclaim argument.

### 8.4 `0x1f9cd00` — DeferredDeletionQueue reclaim

The disassembly begins with the source-equivalent sequence and queue offsets:

```asm
141f9cd0d  movl  ebp, [rcx+0x34040]   ; dequeuePos
141f9cd16  movq  rsi, rdx             ; minReaderEpoch
141f9cd30  movl  ecx, ebp
141f9cd32  andl  ecx, 0xfff
141f9cd3c  movl  eax, [rdi+rcx*4+0x30000] ; sequence
...
141f9cd5d  movq  rax, [rsi+0x10]      ; entry.epoch
...
141f9cd72  cmpxchg [rdi+0x34040], ... ; dequeue CAS
```

This matches `DeferredDeletionQueue::reclaim` at `DeferredDeletionQueue.h:109-177`, including:

```text
dequeuePos offset      = 0x34040
enqueuePos offset      = 0x34000
sequence base          = 0x30000
DQueue base from Engine = epochBase + 0x1440
```

The other frozen T1 addresses:

```text
0x1f9cd59 = entry.epoch comparison/read boundary
0x1f9cd66 = epoch-gate branch
0x1f9cd72 = dequeue CAS pre
0x1f9ce04 = reclaim return/empty-or-blocked path
```

all lie in the same generated DQueue function and are structurally valid.

### 8.5 S7 and T2

The EXE tail around `waitForDrain()` is:

```asm
141fa8068  call  0x141fa8640
141fa806d  call  0x14204a230
...
141fa8083  vcomisd xmm1, xmm2
141fa8087  jae   0x141fa80b3
141fa8090  call  0x1420554e0
141fa8095  jmp   0x141fa7f28
...
141fa80b2  ret
141fa80b3  xor   al, al
141fa80b5  jmp   0x141fa809c
```

`0x1fa80b3` is the timeout-false branch. `0x1fa80b2` is the function return instruction reached after the epilogue.

There is no `call` instruction from S7 (`+0x1fa80b3`) to T2 (`+0x1fa80b2`). The only instructions are result write, jump, epilogue, and return. Therefore D is false: no production reclaim can execute strictly between S7 and T2.

## 9. M0 shutdown chronology

The runtime and source chronology resolve as follows:

```text
releaseResources enter                         log:259
terminal release accepted                      log:260
RUNNING -> STOP_ACCEPTING_WORK                 log:261
STOP_ACCEPTING_WORK -> STOP_AUDIO               log:262
UI Convolver/EQ release                        log:270-290
  Convolver release may call Engine tryReclaimResources
  (state is still 0, T1 marker filter rejects)
final DSP lifetime retire                       ReleaseResources.cpp:577/586
  -> T0 enqueue, epoch=9, ticket=4              log:292
waitForDrain(2000,2)                            ReleaseResources.cpp:615
  loop:
    drainDeferredRetireQueues(true)             Threading.cpp:237
      -> router -> m_epochDomain -> DQueue
      (state is 1; T1 requires state 2)
    elapsed >= timeout
      -> S7 timeout branch                      log:872
      -> epilogue
      -> T2 return breakpoint sets state 4      log:1405
return false
post-wait timeout branch entered               log:1938
  drainPendingRetireIntentsForShutdown
  drainDeferredRetireQueues(true)               ReleaseResources.cpp:660
  m_epochDomain.tryReclaim()                    ReleaseResources.cpp:661
  (state is 4; T1 requires state 2/3)
destructor fallback                             log:1941+
  conditional publishEpoch/tryReclaim          CtorDtor.cpp:249-250
  drainDeferredRetireQueues(true)               CtorDtor.cpp:270
  (state remains 4; T1 marker filter rejects)
```

The critical point is the state window:

```text
state=1: T0 observed; post-T0/pre-S7 reclaim exists, but T1 is disabled
state=2: S7 marker set; no production call before T2
state=4: T2 set; post-wait/destructor reclaim exists, but T1 is disabled
```

There is no state `2` or `3` interval containing a production EpochDomain reclaim. This fully explains the zero T1 marker output.

## 10. A/B/C/D determination

### A. Is EpochDomain::tryReclaim called during M0 shutdown?

**Yes, through conditional production paths, and the terminal timeout path was entered in this execution.**

Evidence:

```text
source: waitForDrain calls drainDeferredRetireQueues before testing timeout
source: post-wait branch calls drainDeferredRetireQueues + direct domain reclaim
runtime: log:1938 enters “drain timeout reached” branch
runtime: log:1939 reports routerPendingRetire=2
source: destructor fallback calls tryReclaim while pending count remains nonzero
```

The exact number of physical executions is not preserved because the final `bl` is absent, but the branch and call sites are statically closed and the process continued through them without a CDB error.

### B. Does current EXE RVA `0x1f9cfb0` correspond to the execution point?

**Yes.**

The counter offsets, getMin virtual call, DQueue direct call, and success-counter update uniquely match `EpochDomain::tryReclaim`. PDB names are unavailable, but no PDB mapping contradicts the EXE/source mapping.

### C. Can optimization, inlining, or tail-call explain all eight misses?

**No as the primary explanation.**

- The function body is present in the current EXE.
- getMin is represented by a real virtual call and return.
- DQueue reclaim is a real direct call.
- The T1 points are not bypassed due to inlining or tail-call transformation.
- The all-eight pattern is explained uniformly by the state predicate.

The only optimization-related fact is that getMin is not represented as a separately named out-of-line function; the chosen pre/return points correctly bracket its call. This does not suppress the target entry.

### D. Can reclaim occur strictly between S7 and T2?

**No.**

S7 and T2 are in the same function tail. EXE CFG contains no call between the timeout result write and return.

## 11. P/Q/R/S classification

| case | definition | verdict | reason |
|---|---|---|---|
| **P** | T1 breakpoint/capture mapping fails to expose an executed reclaim | **SELECTED** | T1 is state-gated to 2/3, while relevant execution is state 1 or 4; the RVA itself is correct |
| Q | M0 has no reclaim path after S7 | REJECTED | post-wait branch and destructor fallback contain direct Engine reclaim calls; log enters the post-wait timeout branch |
| R | router dispatches to another provider/domain | REJECTED | router provider is `AudioEngine::m_epochDomain`; EQProcessor has a separate domain but is not the captured queue |
| S | source call exists but optimization removes the chosen RVA | REJECTED | EXE contains the complete generated tryReclaim/getMin/DQueue sequence at the chosen RVAs |

Final classification:

```text
T1 zero-hit reason = Case P / state-gated capture sequencing failure
```

## 12. STOP-condition matrix

| stop condition | result | evidence |
|---|---|---|
| STOP-A latest `ConvoPeq.md` identity mismatch | NOT TRIGGERED | expected SHA-256 and size match |
| STOP-B production caller inventory not unique | NOT TRIGGERED | Engine/router, SnapshotCoordinator, EQProcessor, and test stubs are separated |
| STOP-C EpochDomain object identity not closed | NOT TRIGGERED | constructor, member declaration, runtime offset `+0x10`, DQueue offset `+0x1440` all agree |
| STOP-D RVA/source/PDB mapping mismatch | NOT TRIGGERED | PDB procedure names are absent, but source structure + EXE CFG are uniquely consistent; no contradictory PDB record |
| STOP-E optimization/control flow not statically decidable | NOT TRIGGERED | exact EXE CFG closes S7/T2 and T1 call sequence |
| STOP-F runtime capture required to classify this audit | NOT TRIGGERED | state predicates and production call graph classify the zero-marker reason statically |
| STOP-G source change required | NOT TRIGGERED | no source change is needed or performed for this audit result |

The reader-attribution result itself remains stopped because T1 runtime values were never captured.

## 13. Prohibitions check

| prohibition | verification | result |
|---|---|---|
| CDB Retry-2 | no Harness/CDB execution in this gate | PASS |
| M1/M2 | no process started | PASS |
| production source change | no production file write/edit | PASS |
| test source change | no test file write/edit | PASS |
| CMake change | no CMake file write/edit | PASS |
| getter/counter/telemetry addition | no production/test instrumentation added | PASS |
| shutdown change | no edit | PASS |
| EpochDomain change | no edit | PASS |
| reclaim/router change | no edit | PASS |
| build | no build command | PASS |
| Dr.Memory | not launched | PASS |
| `minReaderEpoch=9` reconstruction | explicitly rejected | PASS |

## 14. Attribution boundary after this audit

The audit determines why the existing T1 capture emitted no markers. It does **not** determine why the S7 head was not reclaimed.

The following remain unproven:

```text
minReaderEpoch at the relevant reclaim call
active/eligible reader state at the exact getMin call
reader lifecycle transition between T0/S7/T1/T2
Case A / B / C / D
S7_READER attribution
root cause of residual DQueue entry
```

The T0/S7/T2 inactive snapshots cannot replace a T1 `getMinReaderEpoch` observation. Existing RCA-7 values are not imported into Retry-1.

Current disposition remains:

```text
P3-5-FPM-C3-T1-Path-Audit-1 = CLOSED
T1 zero-hit mechanism        = CASE P / PARTIALLY LOCALIZED
S7_READER                    = UNRESOLVED
Downstream disposition       = PARTIALLY-LOCALIZED / STOP
IMPLEMENTATION               = FORBIDDEN
```

Any future debugger-script change or runtime retry requires a separate explicitly authorized gate. This report does not create a revised script and does not authorize Retry-2.
