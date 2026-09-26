# P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Execution-1

## 1. Verdict

```text
Gate                                  = P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Execution-1
Mode                                  = runtime benign probe / exactly one attempt
Authorization                         = CONSUMED
Invocation count                      = 1
Result                                = PROBE_RUNTIME_STOP
AudioEngineHarness                    = 0
Retry-3                               = 0
Production capture                    = 0
Probe rerun                           = 0
Source/test/CMake/build/Dr.Memory/M1/M2 = 0
```

The result is STOP, not a partial semantic PASS. Breakpoint registration and an actual target hit occurred, but Family A-G did not execute to a classified result.

## 2. Frozen identity and invocation

| artifact | SHA-256 | result |
|---|---|---|
| `C:\Windows\System32\ping.exe` | `E4224D18C3C96826E6F4240893FAC02C8A1FF335CE92614FC8744369594DD468` | PASS |
| `tmp\cdb.exe` | `5F54ABAFCA3AE5638BBF807D402FABB350A64575C1DFA9FBFC7F5732DF5BEE67` | PASS |
| frozen probe script | `5C12E1515CC2206430F34ADC7B8FE39D84A04BF4B33F2D7628970C395EC7699B` | PASS |
| `ConvoPeq.md` | `E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609` | unchanged identity |

CDB version:

```text
10.0.29617.1000
```

Exact invocation:

```text
C:\VSC_Project\ConvoPeq\tmp\cdb.exe
  -logo C:\VSC_Project\ConvoPeq\doc\work113\P1-5-IR-P2_Step5BL_P3-5-FPM-C3-T1-Capture-PostRetry2_Benign-Probe-Execution-1_CDB.log
  -cf C:\VSC_Project\ConvoPeq\doc\work113\P1-5-IR-P2_Step5BK_P3-5-FPM-C3-T1-Capture-PostRetry2_Benign_Probe_Runtime_Authorization-1.cdb
  C:\Windows\System32\ping.exe
  127.0.0.1
  -n
  32
```

Preflight passed immediately before this sole invocation:

```text
three required SHA-256 identities = PASS
CDB version                      = PASS
CDB/ping/AudioEngineHarness residue = 0
reserved log absent              = PASS
```

## 3. Preserved CDB log

```text
Path   = doc/work113/P1-5-IR-P2_Step5BL_P3-5-FPM-C3-T1-Capture-PostRetry2_Benign-Probe-Execution-1_CDB.log
SHA-256 = BCFE1398DB6896C3C1FBB381C65C43EBD48ECAC5ACAE3D75F213F595B1C5DD2B
Bytes  = 20,447
```

The authorization log path was created by this one attempt and is now frozen evidence.

## 4. Runtime chronology

```text
CDB initial break
  -> ping.exe loaded
  -> .reload /f ping.exe reported no symbol file found
  -> MASM evaluator selected
  -> $t0..$t19 initialized
  -> five target-image base assignments produced Syntax error
  -> three bp commands registered and retained
  -> g
  -> actual hit at ping+0x39D9
  -> breakpoint body failed at dwo(@$t1) with Memory access error
  -> C3_BENIGN_PROBE_SCRIPT_END
  -> final q
  -> quit:
  -> process residue 0
```

Breakpoint registration and retention:

```text
ping+0x1184 = registered
ping+0x387D = registered
ping+0x39D9 = registered
registered count = 3
actual hit = ping+0x39D9
```

## 5. Exact failure evidence

### 5.1 Target-base initialization failures

CDB reported these runtime syntax errors before reaching `g`:

```text
r $t1 = ping!+0x3c      -> Syntax error at 'ping!+0x3c'
r $t2 = ping!+0x4FD0    -> Syntax error at 'ping!+0x4FD0'
r $t3 = ping!+0x2600    -> Syntax error at 'ping!+0x2600'
r $t4 = ping!+0x5000    -> Syntax error at 'ping!+0x5000'
r $t5 = ping!+0x40      -> Syntax error at 'ping!+0x40'
```

Therefore `$t1` through `$t5` were not established as their intended image-relative bases. The frozen script was not changed and no corrective action was taken.

### 5.2 Actual breakpoint hit

The log contains:

```text
ping+0x39d9:
00007ff7`deea39d9 ff1529190000 call qword ptr [ping+0x5308]
ds:00007ff7`deea5308={KERNELBASE!Sleep}
```

This proves an actual benign-target hit, not merely registration.

### 5.3 Execution-time expression failure

The first Family A condition at the hit produced:

```text
Memory access error at ') == 0x100) { .echo PASS_A_POS; ... }'
```

The body did not emit `PASS_A_POS` or `FAIL_A_POS`. Because `$t1` had not been successfully assigned, the first `dwo(@$t1)` could not dereference its intended PE-header address.

No other expression family was reached. In particular, this run does not establish hit-time acceptance of:

```text
dwo
poi
64-bit low/high splitting
complex ring arithmetic
MASM and in a compound condition
dwo inside dd/dq address arguments
positive/negative branch execution
PASS_TERMINATION_Q
```

## 6. Marker classification

Standalone runtime marker counts:

```text
PASS_A_POS          = 0
PASS_A_NEG          = 0
PASS_B_POS          = 0
PASS_B_NEG          = 0
PASS_B_HW_POS       = 0
PASS_B_HW_NEG       = 0
PASS_C_POS          = 0
PASS_C_NEG          = 0
PASS_D_POS          = 0
PASS_D_NEG          = 0
PASS_E_POS          = 0
PASS_E_NEG          = 0
PASS_F_POS          = 0
PASS_F_NEG          = 0
PASS_G_POS          = 0
PASS_G_NEG          = 0
PASS_TERMINATION_Q  = 0
standalone FAIL_*   = 0
quit:               = 1
```

`quit:` proves the debugger process reached its final quit command. It does not override the required absence of parser/runtime errors and missing PASS markers, so the overall classification remains STOP.

## 7. Clean termination and residue

```text
CDB final q reached = yes
quit: emitted       = yes
CDB process residue  = 0
ping process residue = 0
AudioEngineHarness   = 0
```

The process boundary was clean. The semantic probe result was not PASS.

## 8. Runtime result audit

| requirement | result |
|---|---|
| three breakpoint registrations | PASS |
| breakpoint retained in `bl` | PASS |
| actual breakpoint hit | PASS |
| no Syntax error | FAIL |
| no Memory access error | FAIL |
| no Bad register | PASS (not observed) |
| no Couldn't resolve | PASS (not observed) |
| no Operand error | PASS (not observed) |
| Family A-G complete | FAIL |
| Family G dd/dq executed | NOT REACHED |
| PASS_TERMINATION_Q | FAIL |
| clean debugger termination | PASS |
| process residue 0 | PASS |
| overall | PROBE_RUNTIME_STOP |

## 9. Evidence boundary

This execution proves only:

```text
CDB breakpoint registration and actual benign-target hit
failure of the frozen ping!+RVA initialization form in this run
failure of the subsequent dwo expression due to invalid pseudo-register value
clean debugger termination and process cleanup
```

It does not change:

```text
DIRECT_DQUEUE_CAUSE = epoch gate equality PROVEN
S7_READER          = UNRESOLVED
S7_READER_SLOT     = UNRESOLVED
minReaderEpoch     = NOT_CAPTURED
Case A/B/C/D       = NOT_PROVEN
IMPLEMENTATION     = FORBIDDEN
```

No source, test, CMake, build, Dr.Memory, M1, M2, Harness, Retry-3, production capture, or reader-attribution work was performed.

## 10. Consumption and next boundary

```text
Benign probe authorization = CONSUMED
Probe execution attempts   = 1
Rerun authorization        = NONE
Next action                = read-only CDB Expression-Semantics Result Audit
Production authorization   = NOT GRANTED
```

The next audit may explain the two observed forms—`ping!+RVA` assignment failure and invalid-pseudo-register dereference—but it must not rewrite this script, rerun this probe, or authorize production capture.
