# P3-5-FPM-C3-T1-Capture — Redesign-4 Single Execution / Result Audit R1–R10

## 1. Gate result

```text
Gate    = P3-5-FPM-C3-T1-Capture-Redesign-4-Single-Execution-Result-Audit
invocation failures = 1, corrected, not counted as a measurement
CDB executions      = 1  (the single authorized measurement)
harness executions  = 1
build               = 0
re-runs             = 0

R1  = PASS      CDB parsed and executed the script end to end
R2  = NOT MET   no T0 ticket evidence
R3  = NOT MET   no T1 invocation evidence
R4  = NOT MET   no globalEpoch / minReaderEpoch pair
R5  = NOT MET   no ReaderSlot eligibility evidence
R6  = NOT MET   S7_READER, S7_READER_SLOT undetermined
R7  = NOT MET   no DQueue entry / sequence / epoch evidence
R8  = NOT MET   epoch gate undetermined
R9  = NOT MET   no T1 CAS pre or post evidence
R10 = NOT MET   Case A / B / C / D undetermined

A1 separator fix = PROVEN at runtime
VERDICT = MEASUREMENT VALID, T1 EVIDENCE NOT OBTAINED
```

## 2. Execution record

```text
script   doc/work113/P1-5-IR-P2_Step5CP_P3-5-FPM-C3-T1-Capture-Redesign-4_Corrected.cdb
SHA-256  A935369D711535C9837DFD25761B7886F6CB49D4157534BA34446DBB15F1022C   unchanged
bytes    14,624
target   build/Release/AudioEngineHarness.exe
SHA-256  E5C7AFB9C4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75
args     --measurement=normal
cdb      tmp/cdb.exe 10.0.29617.1000, SHA 5F54ABAF…FBEE67

command  tmp\cdb.exe -cf <script> -logo <log> <exe> --measurement=normal
started  2026-09-26 13:03:52
ended    2026-09-26 13:04:00
elapsed  8.7 s
exit     0
log      doc/work113/P1-5-IR-P2_Step5CQ_…_Redesign-4_Single-Execution_CDB.log
         49,755 bytes, 361 lines
```

## 3. The invocation failure, classified

The first attempt failed before any measurement and was corrected under a separate authorization.

```text
category                          = debuggee launch failure, CDB startup parameter error
cause                             = -o given a file argument; -o is a boolean flag
                                    (cdb -?: "-o debugs all processes launched by debuggee")
                                    the log-file option is -logo <logfile>
effect                            = log path consumed as the debuggee image
                                    "Debuggee initialization failed, Win32 error 0n2"
harness launched                  = no
breakpoints reached               = 0
-cf script processed              = no, -cf runs at the first debugger prompt, never reached
memory read                       = 0
log produced                      = none
exit                              = -2147024894
```

This is distinct from the two other categories the owner asked to separate, and neither of those
occurred:

```text
CDB startup, script parse failure       did NOT occur, see R1
breakpoint reach, measurement failure    THIS is what occurred, see R6
```

## 4. R1, script parse and execute, PASS

```text
C3_T1_REDESIGN2_SCRIPT_BEGIN whole-line   = 1
C3_T1_REDESIGN2_SCRIPT_END   whole-line   = 1
breakpoints set and resolved              = 13 of 13, all 'e', all with addresses
'g' issued, debuggee ran to ZwTerminateProcess = yes
'quit:'                                   = 1
exit code                                 = 0
```

Error census over the whole log:

```text
'Syntax error'      0        'Illegal'    0        'Invalid'    0
'Extra character'   0        'Undefined'  0        'error'      0
'Cannot'            0
'Unable to'         4, all benign: 3 extension DLLs absent (ntsdexts, uext, exts)
                             1 "Unable to verify checksum" for the target exe, expected,
                               the image is not signed and carries no debug directory
'WARNING'           1, the same checksum line
```

### 4.1 A1 is now PROVEN at runtime

This is the substantive positive result of the run.

```text
inter-block transitions of the form  } ; .else / } ; .if   = 85
across 13 breakpoint bodies, 3 to 14 each
parse failures attributable to them  = 0
```

The B1 chain had characterised this separator rule from a probe, on this CDB build, across three
executions. Redesign-4 applied it to all thirteen bodies and executed without a single parse error.
The rule is confirmed on the real capture script, not only on the probe that discovered it. The
previous gate's prediction that all thirteen bodies would have aborted at their first block
boundary had `} .else` remained is contradicted by this result, which is the expected outcome of
having applied A1.

## 5. R2–R10, all gates closed

Marker census by whole trimmed line equality, which excludes the `bp` command echo and the `bl`
listing:

```text
distinct C3_* marker lines observed = 2
  C3_T1_REDESIGN2_SCRIPT_BEGIN   1
  C3_T1_REDESIGN2_SCRIPT_END     1
C3_CR1_STOP_* observed           = 0
```

Every substantive marker is absent:

```text
ABSENT (17)  C3_T0_ENTRY, C3_T0_CAS_PRE, C3_T0_CAS_POST, C3_T0_SEQUENCE_PUBLISHED,
             C3_TERMINAL_S7_ANCHOR, C3_T1A_CANDIDATE, C3_T1B_GETMIN_CALL_PRE,
             C3_T1B_GETMIN_RETURN, C3_T1C_DQUEUE_ENTRY, C3_T1_SELECTION, C3_T1_CAS_PRE,
             C3_T1_CAS_SUCCEEDED, C3_T1_EPOCH_BLOCKED, C3_T2_WAIT_RETURN,
             C3_PREDECESSOR_HEAD, C3_NON_TARGET_HEAD, C3_CANDIDATE_NO_TARGET
```

Not one gate opened, and not one stop condition fired. The absence of stop markers matters: the
script was designed to halt loudly on any correlation contradiction, and it halted on none. There
is no contradiction in the captured data, because there is no captured data.

### 5.1 The instrumented activity did occur on the harness side

```text
[PUBLISH]  seq=3 gen=3 worldId=3 publishDurationUs=17218 publishCallbackIdx=20
REBUILD_DISPATCHED  x3
[AFFINITY]  x3     [FAULT]  x1     [DIAG]  x42     [DSPCORE_PREPARE]  x62
```

Publishing, rebuild dispatch and teardown all ran. The measurement window was live. The capture
produced nothing because its entry conditions were never satisfied, not because the harness was
idle.

### 5.2 Why the log cannot say which condition failed

The entry gate at `0x1f9c550`, as executed, verbatim:

```text
bp AudioEngineHarness+0x1f9c550 ".if (@$t0 == 0) { .if (@r9 == 9) { .if (dd(@rcx+0x34000) == 4)
  { .echo C3_T0_ENTRY; r; dq @rsp L1; ln poi(@rsp); gc } ; .else { gc } } ; .else { gc } } ; .else { gc }"
```

Every guard in the script is a silent fall-through. On any mismatch the body executes a bare `gc`
and produces no output. The four static guards that gate entry are:

```text
C3_T0_ENTRY            $t0==0  AND @r9==9  AND dd(@rcx+0x34000)==4
C3_T0_CAS_PRE          @r9==9  AND @r10==4 AND dd(@r11+0x34000)==4  AND dd(@r11+0x30010)==4
C3_T0_CAS_POST         @r9==9  AND @r10==5 AND dd(@r11+0x34000)==5  AND dd(@r11+0x30010)==4
C3_T0_SEQUENCE_PUBLISHED $t0==0 AND @r9==9 AND @r10==5 AND @rbx==4
                        AND dd(@r11+0x34000)==5 AND dd(@r11+0x30010)==5 AND @r11!=0
```

Consequently this log **cannot distinguish** two different failures:

```text
(a) the instruction at the breakpoint was never executed
(b) it was executed and at least one guard was false
```

This is a defect in the script's diagnostic design, not in its arithmetic. It was not visible during
static validation because static validation cannot evaluate runtime register values, and the
omission is not one of the constructs V01–V20 test for. Recorded here so it is not rediscovered as
a mystery later.

`@r9 == 9`, `@r10 == 4`, `@r10 == 5` and `@rbx == 4` are hardcoded magic constants. Nothing in this
run provides evidence for or against their correctness, and this audit does not assert anything
about them. Establishing what those registers actually hold requires a measurement that this script
does not perform.

## 6. A separate finding, the log does not capture the debuggee's stdout

The console showed five measurement-summary lines that the `-logo` file does not contain.

```text
present on console, ABSENT from the log:
  [normal] baseline A=3 R=2 observedOutstandingMax=1 worldReclaimCount=2 ...
  [normal] afterPublish acquireObserved=11 referenceAcquire=11 referenceRelease=10
  [normal] state after End = 0
  [normal] O_w=1 T_w=2 E_w=1 windowId=1 sampleCount=23 ...
  [normal] evidence A_start=3 R_start=2 A_end=11 R_end=10 ...
```

The log does contain `[DIAG]` 42, `[DSPCORE_PREPARE]` 62, `[REBUILD_TELEMETRY]` 6,
`[AUTO_GAIN_PLAN]` 11, `[PUBLISH]` 1, `[FAULT]` 1, `[AFFINITY]` 3, so the file is not truncated at
the head. The missing lines are the debuggee's direct stdout writes, which bypass the debugger's
output channel; the retained lines arrive through it. `-logo` captures the debugger's stream, not
the debuggee's console handle.

This does **not** weaken the R2–R10 finding. C3 markers are produced by `.echo`, a debugger command,
and `.echo` capture is proven to work in this very log by `C3_T1_REDESIGN2_SCRIPT_BEGIN` and
`C3_T1_REDESIGN2_SCRIPT_END`. The absence of the 17 markers is therefore a real absence, not a log
artifact.

The consequence for future gates is that harness-side summary telemetry must be read from the
console stream or from a harness-side file, not from a `-logo` debugger log.

## 7. Frozen items, unchanged

```text
S7_READER          = UNRESOLVED   no execution evidence
S7_READER_SLOT     = UNRESOLVED   no execution evidence
minReaderEpoch(T1) = NOT CAPTURED  no execution evidence
Case A / B / C / D = NOT PROVEN   no execution evidence
T1 correlation C1-C13 = NOT EVALUATED, no data to evaluate
IMPLEMENTATION     = FORBIDDEN
```

None of these advanced. A valid measurement that yields no T1 evidence leaves all of them exactly
where they were. In particular `dd @$t1 L1` was installed at all seven T1 sites and never executed,
so A2 is validated as present and syntactically accepted by CDB, and is **not** validated as having
produced a `globalEpoch` value.

## 8. Constraints honored

```text
.cdb modified                  = 0, SHA unchanged
ConvoPeq.md modified           = 0, SHA unchanged
production source / test / CMake modified = 0
build invoked                  = 0
AudioEngineHarness modified    = 0, SHA unchanged
Dr.Memory used                 = 0
M1 / M2 performed              = 0
Retry-3 performed              = 0
breakpoint topology changed    = 0, 13 sites, V05 unchanged
$t0..$t19 allocation changed   = 0
dd @$t1 L1 re-edited           = 0
additional executions          = 0, one authorized measurement, one corrected invocation
```

## 9. Final state

```text
Redesign-4 single execution + Result Audit
= CLOSED / R1 PASS / R2-R10 NOT MET

log  doc/work113/P1-5-IR-P2_Step5CQ_P3-5-FPM-C3-T1-Capture_Redesign-4_Single-Execution_CDB.log
     49,755 bytes, 361 lines, exit 0

A1 separator rule  = PROVEN at runtime, 85 transitions, 0 parse failures, 13 of 13 bodies
A2 globalEpoch read = installed and accepted by CDB, never executed, value unproven
7-site 0xC0 fix    = installed and accepted by CDB, never reached, effect unproven

S7_READER / S7_READER_SLOT / minReaderEpoch(T1) / Case A-D = UNRESOLVED
IMPLEMENTATION = FORBIDDEN
```

## 10. Next gate, not requested here

The capture needs entry gates that either open on the values actually present or report why they did
not. That is a script change, and it needs its own design and authorization, on the same gate
discipline as every previous step. This audit requests nothing and proposes no edit.
