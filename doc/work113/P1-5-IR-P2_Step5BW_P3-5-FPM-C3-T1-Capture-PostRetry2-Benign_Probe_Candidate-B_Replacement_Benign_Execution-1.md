# P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Candidate-B-Replacement-Benign-Execution-1

## 1. Gate result

```text
Gate                    = P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Candidate-B-Replacement-Benign-Execution-1
Mode                    = single authorized benign execution + mechanical log capture
CDB execution           = 1
ping execution          = 1
Authorization           = CONSUMED
script modified         = 0
new .cdb created        = 0
verdict assigned here   = NONE (deferred to Result Audit-1)
AudioEngineHarness      = 0
Retry-3                 = 0
production C3 capture   = 0
Preparation-4           = 0 (still BLOCKED)
M1/M2/build/Dr.Memory   = 0
rerun                   = 0
```

This gate records the run and freezes the log. It deliberately assigns no B1 verdict; that is the next gate's job.

## 2. Pre-execution identity recheck

All eight items verified immediately before invocation, with no value refreshed in place:

| # | item | expected | observed | result |
|---|---|---|---|---|
| 1 | `ConvoPeq.md` | `E5E74200...EF3609` | identical | PASS |
| 2 | B1 script | `403C8E2B...BB8329` | identical | PASS |
| 3 | `cdb.exe` | `5F54ABAF...5BEE67` | identical | PASS |
| 4 | CDB version | `10.0.29617.1000` | `10.0.29617.1000` | PASS |
| 5 | `ping.exe` | `E4224D18...DD468` | identical | PASS |
| 6 | script bytes | `1481` | `1481` | PASS |
| 7 | process residue | `0` | `0` | PASS |
| 8 | reserved log | absent | absent | PASS |

No drift. The authorization was consumed by this single invocation.

## 3. Invocation

```text
C:\VSC_Project\ConvoPeq\tmp\cdb.exe
  -logo doc/work113/P1-5-IR-P2_Step5BV_P3-5-FPM-C3-T1-Capture-PostRetry2-Benign_Probe_Candidate-B1_Execution-1_CDB.log
  -cf   doc/work113/P1-5-IR-P2_Step5BU_P3-5-FPM-C3-T1-Capture-PostRetry2-Benign_Probe_Candidate-B1_Anchor_Probe.cdb
  C:\Windows\System32\ping.exe 127.0.0.1 -n 32
```

## 4. Frozen log

```text
Path    = doc/work113/P1-5-IR-P2_Step5BV_P3-5-FPM-C3-T1-Capture-PostRetry2-Benign_Probe_Candidate-B1_Execution-1_CDB.log
SHA-256 = 74E1C8185ADB2A599C4AB63164D3052E85068B4A455E3E7691C23B811BD79AF7
bytes   = 7,941
```

Post-execution state:

```text
B1 script     = 403C8E2B...BB8329 (unchanged)
ConvoPeq.md   = E5E74200...EF3609 (unchanged)
process residue = 0
```

## 5. Mechanical marker inventory

Recorded verbatim. No interpretation is applied in this section.

```text
C3B1_SCRIPT_BEGIN         = 1
C3B1_SENTINEL_SET         = 1
C3B1_ASSIGN_DONE          = 1
C3B1_READBACK_DONE        = 1
C3B1_BP_SET               = 1
C3B1_BREAKPOINT_HIT       = 1
C3B1_CANDIDATE_B1_DERIVED = 1
C3B1_CROSSCHECK_BEGIN     = 1
XCHK_T1_AGREE             = 1
XCHK_T2_AGREE             = 0
XCHK_T3_AGREE             = 0
XCHK_T4_AGREE             = 0
XCHK_T5_AGREE             = 0
XCHK_T*_DISAGREE          = 0 (all five)
C3B1_CROSSCHECK_END       = 0
C3B1_SENTINEL_CLEARED     = 0
C3B1_SENTINEL_RETAINED    = 0
C3B1_COUNTS               = 0
C3B1_ALREADY_HIT          = 0
C3B1_SCRIPT_END           = 1
quit:                     = 1

bl entries                = 1
```

## 6. Observed values, verbatim

```text
line  93  $t11=00007ff601b80000
line  94  $t1=00007ff601b8003c
line  95  $t2=00007ff601b84fd0
line  96  $t3=00007ff601b82600
line  97  $t4=00007ff601b85000
line  98  $t5=00007ff601b80040
line 102  ping+0x39d9:
```

Module load for this run:

```text
ModLoad: 00007ff6`01b80000 00007ff6`01b8c000   ping.exe
```

## 7. Error observed

One error line, at log line 101, immediately after `XCHK_T1_AGREE`:

```text
^ Extra character error in ' r $t16=1; r $t17=@$t17+1; .echo C3B1_BREAKPOINT_HIT; r $t11=@$bp0-0x39d9; ... '
```

Other error classes:

```text
Bad register error  = 0
Memory access error = 0
Syntax error        = 0
Couldn't resolve    = 0
```

The comment-origin error class that produced fifteen lines in the Preparation-3 execution did **not** reappear. Zero comment lines existed in the script, and zero were produced.

The `Extra character error` is recorded here as an observation only. Its cause, its relationship to the body, and its effect on `$t18` are explicitly left to Result Audit-1.

## 8. Sequence observed in the log

```text
breakpoint registered (1 entry, ping+0x39d9)
g
actual hit at ping+0x39d9
C3B1_BREAKPOINT_HIT
C3B1_CANDIDATE_B1_DERIVED
$t11, $t1..$t5 dumped
C3B1_CROSSCHECK_BEGIN
XCHK_T1_AGREE
Extra character error
ping+0x39d9:  (stop context)
.echo C3B1_SCRIPT_END
q
quit:
```

## 9. R3-CHECK raw observations

Provided for the Result Audit. The audit assigns the verdicts.

```text
R3-CHECK-A  bl showed exactly one registered breakpoint          observed: yes (1 entry)
R3-CHECK-B  C3B1_BREAKPOINT_HIT present                          observed: yes (1)
R3-CHECK-C  C3B1_CANDIDATE_B1_DERIVED present                    observed: yes (1)
R3-CHECK-D  C3B1_CROSSCHECK_BEGIN present                        observed: yes (1)
R3-CHECK-E  XCHK_T*_AGREE/DISAGREE counts and $t18 tally         observed: T1_AGREE=1, others 0,
                                                                    C3B1_COUNTS absent so $t18 not dumped
R3-CHECK-F  error taxonomy attributed to @$bp0 only              observed: Bad register = 0;
                                                                    one Extra character error present
R3-CHECK-G  process residue 0                                    observed: yes (0)
```

## 10. State carried forward, unchanged

```text
DIRECT_DQUEUE_CAUSE = epoch gate equality PROVEN
Candidate A         = PROVEN
B1 static           = PROVEN
B1 runtime          = NOT PROVEN  (verdict deferred to Result Audit-1)
S7_READER          = UNRESOLVED
S7_READER_SLOT     = UNRESOLVED
minReaderEpoch     = NOT CAPTURED
T1                 = NOT CAPTURED
Case A/B/C/D       = NOT PROVEN
IMPLEMENTATION     = FORBIDDEN
```

Even if B1 is later judged PROVEN, this authorization does not permit proceeding to T1 capture. `dwo`, `poi`, ReaderSlot access, `minReaderEpoch`, and all DQueue inspection remain unverified and unauthorized.

The RT-side observation boundary and the NonRT lifetime isolation line are unchanged. No ConvoPeq reader, epoch, or reclaim evidence was produced or altered.

## 11. Final state

```text
P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Candidate-B-Replacement-Benign-Execution-1
= CLOSED / EXECUTED ONCE / LOG FROZEN / NO VERDICT ASSIGNED

log          = 74E1C818...BD79AF7 (7,941 bytes)
script       = 403C8E2B...BB8329 (unchanged)
production source = E5E74200...EF3609 (unchanged)
process residue  = 0
comment-origin errors = 0
errors observed = 1 (Extra character error, line 101)
authorization  = CONSUMED / NOT REUSABLE
rerun          = FORBIDDEN

B1 verdict     = DEFERRED
Candidate A    = PROVEN (unaffected)
Preparation-4 / Retry-3 / production C3 / AudioEngineHarness / M1 / M2 / build / Dr.Memory
= FORBIDDEN
implementation = FORBIDDEN
```

## 12. Next gate

```text
Candidate-B Replacement-Benign-Execution-1-Result-Audit-1   NOT STARTED  <- next
```

That gate must classify, using R3-CHECK-A through R3-CHECK-G, into exactly one of:

```text
B1 = PROVEN
B1 = FAILED
B1 = INCONCLUSIVE
```

The execution result must not be used in place of that audit.
