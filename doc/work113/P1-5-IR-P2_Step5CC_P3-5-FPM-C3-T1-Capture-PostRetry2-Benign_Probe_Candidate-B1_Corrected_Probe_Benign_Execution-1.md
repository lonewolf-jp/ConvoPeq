# P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Candidate-B1-Corrected-Probe-Benign-Execution-1

## 1. Gate result

```text
Gate                    = P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Candidate-B1-Corrected-Probe-Benign-Execution-1
Mode                    = single authorized benign execution
CDB execution           = 1
ping execution          = 1
rerun                   = 0
second execution        = 0
script modification     = 0
new .cdb created        = 0
verdict rendered here   = NONE  (deferred to Execution-1 Result Audit-1 by instruction)
```

This gate performs the one authorized invocation and records what occurred. It renders no verdict.

## 2. Pre-execution identity recheck

All eight were recomputed immediately before launch. Any drift would have been a STOP.

| # | artifact | expected | observed | result |
|---|---|---|---|---|
| 1 | `tmp\cdb.exe` | `5F54ABAF…FBEE67` | identical, 178,016 bytes | PASS |
| 2 | corrected B1 script | `FFC5228C…AE739DE` | identical, 1,172 bytes | PASS |
| 3 | `C:\Windows\System32\ping.exe` | `E4224D18…4DD468` | identical, 45,056 bytes | PASS |
| 4 | `ConvoPeq.md` | `E5E74200…F3609` | identical | PASS |
| 5 | CDB version | `10.0.29617.1000` | `10.0.29617.1000` | PASS |
| 6 | process residue | `0` | `0` | PASS |
| 7 | reserved log | absent | absent | PASS |
| 8 | `ConvoPeq.md` size | `5,535,334` | `5,535,334` | PASS |

```text
identity failures = 0
pre-launch guard: reserved log absent = TRUE
```

No authorization value was refreshed. The authorization was consumed as issued.

## 3. Invocation

```text
cdb.exe -logo <reserved log> -cf <frozen script> C:\Windows\System32\ping.exe 127.0.0.1 -n 32
```

`-logo` and `-cf` reproduce the form evidenced by the previous Execution-1 log, which shows
`Opened log file ...`, script lines issued at the `0:000>` prompt, and the default `srv*` symbol path.

```text
CDB exit code    = 0
log written      = TRUE
log bytes        = 6,979
log SHA-256      = E2ED51F89863E7664AEA5C1EB56A65096E1F93BB4DC272EF9F21751971CC18C4
process residue  = 0
```

The authorization is consumed. This invocation is not repeatable.

## 4. Observed runtime sequence

Counted from output lines only. Command echoes in the `bp` line, the `bl` listing, and the error
message each repeat every marker string, so substring counts are misleading; the table below counts
lines whose entire trimmed content is the marker.

| # | marker | emitted | line |
|---|---|---|---|
| 1 | `C3B1C_SCRIPT_BEGIN` | 1 | 49 |
| 2 | `C3B1C_SENTINEL_SET` | 1 | 70 |
| 3 | `C3B1C_ASSIGN_DONE` | 1 | 77 |
| 4 | `C3B1C_READBACK_DONE` | 1 | 81 |
| 5 | `C3B1C_BP_SET` | 1 | 84 |
| 6 | `C3B1C_BREAKPOINT_HIT` | 1 | 91 |
| 7 | `C3B1C_B1_DERIVED` | 1 | 92 |
| 8 | `r $t11; r $t1..$t5` | 1 | 93–98 |
| 9 | `C3B1C_CROSSCHECK_BEGIN` | 1 | 99 |
| 10 | `XCHK_T1_AGREE` | 1 | 100 |
| 11 | `XCHK_T2_AGREE` | **0** | — |
| 12 | `XCHK_T3_AGREE` | **0** | — |
| 13 | `XCHK_T4_AGREE` | **0** | — |
| 14 | `XCHK_T5_AGREE` | **0** | — |
| 15 | `C3B1C_CROSSCHECK_END` | **0** | — |
| 16 | `C3B1C_SENTINEL_CLEARED` | **0** | — |
| 17 | `C3B1C_TALLY` | **0** | — |
| — | `C3B1C_SCRIPT_END` | 1 | 105 |
| — | `quit:` | 1 | 107 |

```text
markers emitted = 11 / 17
error           = "^ Extra character error in '<entire 696-char body>'"   line 101
stop context    = ping+0x39d9, call qword ptr [ping+0x5308] = KERNELBASE!Sleep
```

`$t18` was incremented once by the T1 agreement but was never read back, because the readback sits
after `C3B1C_TALLY`. Its final value is therefore **not observed**.

## 5. What did execute

```text
bp set at ping+0x39d9                confirmed by bl, 00007ff6`01b839d9
breakpoint reached                   yes, hit count 0001
@$bp0 accepted                       yes, produced a usable value
$t11 = 0x00007ff601b80000            image base, matches ModLoad of ping.exe
$t11 - 0x39d9                        = the address shown in bl
Candidate A readback                 $t1..$t5 = base + 0x3c / 0x4FD0 / 0x2600 / 0x5000 / 0x40
T1 comparison                        agreed
```

The B1 anchor-address derivation primitive behaved as designed at runtime for the second
consecutive run.

## 6. What did not execute

```text
T2..T5 comparisons        not executed
CROSSCHECK_END            not reached
SENTINEL_CLEARED          not reached
TALLY                     not reached
$t16 / $t17 / $t18 / $t19 readback   not reached
```

## 7. Structural observation

The body stopped at the same logical position as the previous execution, but the construct there
has changed. Comparing the two runs:

```text
previous body   first crosscheck introduced by  "; .if"     ran
                next command introduced by       "} .else "  rejected
this body       first crosscheck introduced by  "; .if"     ran
                next command introduced by       "} .if  "  rejected
```

Separator census of the frozen 696-char body:

```text
"; .if" transitions   = 2     ( T1 at 244, sentinel after CROSSCHECK_END )
"} .if" transitions   = 4     ( T2 304, T3 364, T4 424, T5 484 )
"}" + space + command = 6     ( 304, 364, 424, 484, 543, 645 )
```

Every block-structured command in this body that was introduced by `;` ran. Every one introduced by
a space after a block-closing `}` was rejected. This is consistent across both executions despite
the bodies differing in length, in the failing construct, and in the removal of `.else`, `q`, `gc`
and the outer `.if` wrapper.

## 8. Caret position is not a reliable localizer

Recorded because the previous gate relied on it.

```text
previous run   caret at column 399, mapped to the boundary before ".else {"
this run       caret at column 313, mapped to offset 297, inside "r $t18=@$t18+1"
```

The two carets do not identify a common character, while the behavioral boundary is identical. In
this run the error quotes the **entire** 696-character body, whereas the previous run quoted a
955-character region. The caret therefore reflects where the debugger chose to print within the
reported region, not a character-level fault position.

The reliable signal is behavioral: execution stops at the first block-closing `}` that is followed by
a space and another command.

## 9. Constraints honored

```text
script SHA unchanged        = FFC5228C2387BDC5264F4DB956D66E6DFFB46A97F97ADE2E59E0A7BA1AE739DE
.cdb files in doc/work113   = 6, unchanged
source / test / CMake       = untouched
build                       = not run
Dr.Memory                   = not run
AudioEngineHarness          = not run
production C3               = not run
T1 capture                  = not performed
ReaderSlot / EpochDomain    = not accessed
getMinReaderEpoch           = not called
DQueue / reclaim            = not touched
M1 / M2                     = not run
Retry-3 / Preparation-4     = not started
rerun                       = 0
```

## 10. Outcome

```text
B1 runtime status = NOT RENDERED IN THIS GATE
```

Per instruction, the result is not classified here. The observed evidence is recorded above and is
classified in `Corrected-Probe-Execution-1_Result_Audit-1`, which is a separate gate.

The authorization is now spent. Any further execution requires a new preparation, a new static
validation, and a new runtime authorization.

## 11. Final state

```text
P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Candidate-B1-Corrected-Probe-Benign-Execution-1
= CLOSED / ONE INVOCATION CONSUMED / VERDICT DEFERRED

identity recheck        = 8 / 8 PASS, 0 drift
CDB execution           = 1
ping execution          = 1
CDB exit code           = 0
log SHA-256             = E2ED51F89863E7664AEA5C1EB56A65096E1F93BB4DC272EF9F21751971CC18C4
process residue         = 0

markers emitted         = 11 / 17
breakpoint reached      = yes
B1_DERIVED reached      = yes
T1 agreement            = yes
T2..T5 agreements       = 0
CROSSCHECK_END          = not reached
SENTINEL_CLEARED        = not reached
TALLY                   = not reached
$t18                    = not observed
parser error            = Extra character error

rerun = FORBIDDEN and NOT performed
``` 
