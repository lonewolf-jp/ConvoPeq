# Redesign-6 Runtime Result Audit

## 1. Execution

```text
script   doc/work113/P1-5-IR-P2_Step5CX_…_Redesign-6_EntryGate-Diagnostic.cdb
SHA-256  8D64862031F24A5285EEAFC10DBCD00C4A94C4CFB635CD2E891D227AB5CC23A5   unchanged after the run
target   build/Release/AudioEngineHarness.exe
SHA-256  E5C7AFB9C4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75
args     --measurement=normal
cdb      tmp/cdb.exe 10.0.29617.1000, SHA 5F54ABAF…FBEE67
form     cdb.exe -cf <script> -logo <log> <exe> --measurement=normal

count    = 1        the single authorized measurement
exit code = 0
elapsed   = 10.9 s
log bytes = 56,806
log lines = 824
log       doc/work113/P1-5-IR-P2_Step5CY_…_Redesign-6_Single-Execution_CDB.log
```

`-logo` was used. `-o` was not. No re-run.

## 2. R1, script completion and normal termination

```text
SCRIPT_BEGIN  whole-line = 1
SCRIPT_END    whole-line = 1
ZwTerminateProcess       = 1
'quit:'                  = 1
breakpoints set (bl)    = 13
exit code                = 0

R1 VERDICT = PASS
```

The harness ran its full lifecycle, produced its own summary on the console, and terminated
normally. The debuggee was never left stopped.

```text
D6-10 runtime portion = PROVEN.  diagnostic -> separator -> guard -> gc held, 464 times.
D6-11                 = PROVEN.  no diagnostic failure stranded the debuggee.
continuation property = RUNTIME PROVEN
```

This is the result the whole Stage 1 existed to obtain, and it is the direct contrast with
Redesign-5, where the body aborted, the terminal `gc` was never reached, the harness was killed
while stopped, and elapsed time collapsed to 0.3 s.

## 3. R2 to R5, T0 site reachability

```text
R2  C3_DG_T0E_HIT  0x1f9c550 = 116
R3  C3_DG_T0P_HIT  0x1f9c57d = 116
R4  C3_DG_T0Q_HIT  0x1f9c59a = 116
R5  C3_DG_T0S_HIT  0x1f9c5da = 116
```

All four sites reached, equal counts, and the ordering is exact:

```text
total marker lines = 464 = 116 x 4
complete 4-marker cycles aligned from index 0 = 116
pattern             T0E -> T0P -> T0Q -> T0S, repeated without a break
```

**Every question left NOT DETERMINED by Redesign-5 is now answered.** All four sites execute.

The pattern also carries a structural fact that follows from the disassembly recorded at Step5CS:
`T0P` is the compare-and-swap that claims a queue slot, `T0Q` is its success edge, and `T0S` is the
store of the entry followed by the sequence publication. `T0P`, `T0Q` and `T0S` all firing 116 times
means the target's enqueue path performed **116 complete claim-and-publish cycles**. The queue is
working; the region is hot.

## 4. R9 and R10, the T0 entry condition

```text
C3_T0_ENTRY              = 0
C3_T0_CAS_PRE            = 0
C3_T0_CAS_POST           = 0
C3_T0_SEQUENCE_PUBLISHED = 0
```

The entry condition was true **zero times out of 116 complete enqueue cycles**.

## 5. R11 onward, not interpreted

```text
C3_T1A_CANDIDATE  0     C3_T1B_GETMIN_CALL_PRE  0     C3_T1B_GETMIN_RETURN  0
C3_T1C_DQUEUE_ENTRY 0    C3_T1_SELECTION         0     C3_T1_CAS_PRE         0
C3_T1_CAS_SUCCEEDED 0    C3_T1_EPOCH_BLOCKED     0
C3_TERMINAL_S7_ANCHOR 0  C3_T2_WAIT_RETURN       0
C3_PREDECESSOR_HEAD 0    C3_NON_TARGET_HEAD 0    C3_CANDIDATE_NO_TARGET 0
C3_CR1_STOP_*           0
C3_*_SLOTS_BEGIN        0
```

R9 and R10 did not fire, so per the owner's instruction **none of the above is interpreted, and none
of it is recorded as a failure.** These are the expected values of a T0 chain that never armed, since
every T1 gate is preconditioned on `$t0 == 1` or `$t12 == 1` and `$t0` never became 1.

```text
S7_READER      = NOT DETERMINED, not a failure
S7_READER_SLOT = NOT DETERMINED, not a failure
minReaderEpoch = NOT CAPTURED, not a failure
Case A / B / C / D = NOT PROVEN, not a failure
```

## 6. Error census

```text
Extra character error     = 0
Syntax error              = 0
Illegal / Invalid         = 0
Undefined                 = 0
^Error                    = 0
Unable to read memory     = 0
Cannot                    = 0
Evaluate expression:      = 0     expected, Stage 1 has no '?'

Unable to add extension DLL = 3    ntsdexts, uext, exts, benign and unchanged across all runs
WARNING                    = 1    target image checksum, unsigned, unchanged across all runs
unexpected breakpoint stop = 0
```

Zero errors of any kind. Compare Redesign-5, which carried one `Extra character error` and a
truncated 0.3 s run.

## 7. Stage 1 absences, by design

```text
R6 operand capture  = absent by design
R7 conjunct verdicts = absent by design
R8 R9CHANGE         = absent by design
```

Recorded as absent by design and **not** as defects, per the authorization.

## 8. What this run establishes

```text
PROVEN   the T0 enqueue region executes 116 complete claim-and-publish cycles
PROVEN   all four T0 sites are reachable and are reached, in strict lockstep
PROVEN   a single-command diagnostic can precede the guard without harming continuation
PROVEN   D6-10 runtime portion, and D6-11
PROVEN   the debuggee is not stranded by a diagnostic

PROVEN   the T0 entry condition was true 0 times in 116 cycles
```

### 8.1 What it does not establish, and the one inference that is now available

Not established:

```text
which conjunct of the T0 entry condition is false
what @r9, @r10, @rbx, enqueuePos or sequences[slot] actually held on those 116 hits
whether the entry condition is mis-specified or merely never coincided
```

The available inference, and its exact limit. The T0E guard is a conjunction of three conjuncts:
`@$t0 == 0`, `@r9 == 9`, and `enqueuePos == 4`. Across 116 cycles the conjunction held zero times.
A conjunct that pins a monotonically advancing counter, such as `enqueuePos == 4`, would be expected
to hold in a small number of cycles if the counter traverses the pinned value during the window. A
conjunct that is not a function of that counter would hold either always or never.

**That reasoning narrows the candidates but does not identify the conjunct**, because this run
measured no operand values. Stage 1 deliberately carried no operand capture, so the discrimination
is not available here, and it is not manufactured by inference. The identification is exactly the
next construct class, admitted per D14 behind its own authorization.

No constant is proposed. `@r9 == 9` and `enqueuePos == 4` remain unmodified and unjudged.

## 9. A log-mode characteristic, reproduced

```text
[DIAG] lines in the -logo file   = 42    debugger-channel output, captured
[normal] lines in the -logo file = 0     debuggee stdout, present on console, absent from file
```

The Step5CR audit reported this once. It is now reproduced on an independent run, so it is a
property of `-logo` rather than a defect of any one measurement: the log captures the debugger's
stream, not the debuggee's console handle. This does not affect any C3 marker, because markers are
`.echo` output from the debugger, and the 464 marker lines are all present.

## 10. Constraints honored

```text
.cdb modified during the run  = 0, SHA identical before and after
ConvoPeq.md / harness / cdb.exe / production source / test / CMake / build = untouched
$t0..$t19 / guards / topology / T1 correlation / ReaderSlot windows / A2 = untouched
guard changes = 0        new constants = 0        canary = not added
additional executions = 0        re-run = 0
src / CMakeLists.txt / build.bat = 12 pre-existing entries, unchanged
.cdb total = 11, no new artifact created this gate
residue = 0
```

## 11. Final state

```text
Redesign-6 Runtime Result Audit
= CLOSED / R1 PASS / continuation RUNTIME PROVEN

execution  count 1, exit 0, 10.9 s, log 56,806 bytes / 824 lines
R1   SCRIPT_BEGIN 1, SCRIPT_END 1, ZwTerminateProcess 1, quit 1  -> PASS
R2   C3_DG_T0E_HIT  = 116
R3   C3_DG_T0P_HIT  = 116
R4   C3_DG_T0Q_HIT  = 116
R5   C3_DG_T0S_HIT  = 116      116 complete T0E->T0P->T0Q->T0S cycles, 464 marker lines
R9   C3_T0_ENTRY   = 0
R10  C3_T0_SEQUENCE_PUBLISHED = 0
R11+ all zero, NOT INTERPRETED, not failures

errors  Extra character 0, Syntax 0, Unable-to-read-memory 0, unexpected stops 0
benign  extension DLL 3, image checksum WARNING 1, both unchanged across all runs

D6-10 runtime = PROVEN      D6-11 = PROVEN
A1 separator  = RUNTIME PROVEN
A2            = installed, accepted, execution count 0, globalEpoch value UNPROVEN
T0 gate cause = STILL UNRESOLVED.  Now bounded: 0 of 116 cycles satisfied the entry
               condition.  The specific failing conjunct is not measured.
S7_READER / S7_READER_SLOT / minReaderEpoch(T1) / Case A-D = UNRESOLVED, not failures
IMPLEMENTATION = FORBIDDEN
```

## 12. Next gate, not requested here

Per D14, the next construct class is admitted one at a time, behind its own authorization, and the
first candidate is the one this run most clearly motivates: capturing the operand values at the
reachability markers that are now proven to fire 116 times each. That is a design decision about
which single construct class to admit, and it is not taken here. No diagnostic is added, no guard is
touched, and no constant is changed on the strength of this run.
