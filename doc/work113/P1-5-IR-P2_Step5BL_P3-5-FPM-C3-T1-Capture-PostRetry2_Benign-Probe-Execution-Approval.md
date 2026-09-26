# P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Execution-Approval

## 1. Approval result

```text
Gate                              = P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Execution-Approval
Mode                              = execution approval only
Preflight                         = PASS
Frozen script static validation   = PASS
Target/tool identity              = PASS
Process residue pre-approval      = 0
BENIGN PROBE EXECUTION            = AUTHORIZED / EXACTLY ONE FUTURE ATTEMPT
Execution in this approval gate  = 0
AudioEngineHarness                = FORBIDDEN
Retry-3                           = FORBIDDEN
Production capture                = FORBIDDEN
M1/M2                             = FORBIDDEN
Build / Dr.Memory                 = FORBIDDEN
IMPLEMENTATION                    = FORBIDDEN
```

This report grants one future benign probe execution. It does not execute the probe and does not grant any second attempt, production capture, or implementation work.

## 2. Frozen preflight identities

| artifact | SHA-256 | bytes | result |
|---|---|---:|---|
| `C:\Windows\System32\ping.exe` | `E4224D18C3C96826E6F4240893FAC02C8A1FF335CE92614FC8744369594DD468` | 45,056 | PASS |
| `tmp\cdb.exe` | `5F54ABAFCA3AE5638BBF807D402FABB350A64575C1DFA9FBFC7F5732DF5BEE67` | 178,016 | PASS |
| `ConvoPeq.md` | `E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609` | 5,535,334 | PASS |
| Authorization-1 script | `5C12E1515CC2206430F34ADC7B8FE39D84A04BF4B33F2D7628970C395EC7699B` | 7,233 | PASS |
| Authorization-1 report | `6E2F06EF7DC867C188E41687696454F4D5FB99B11DD36B7D9B7EF8FA684AB7B6` | 9,850 | PASS |

CDB version:

```text
10.0.29617.1000
```

## 3. Frozen execution object

```text
Script = doc/work113/P1-5-IR-P2_Step5BK_P3-5-FPM-C3-T1-Capture-PostRetry2_Benign_Probe_Runtime_Authorization-1.cdb
SHA-256 = 5C12E1515CC2206430F34ADC7B8FE39D84A04BF4B33F2D7628970C395EC7699B
breakpoints = ping+0x1184, ping+0x387D, ping+0x39D9
pseudo-register range = $t0..$t19
state ownership = $t16 state, $t17 hit count, $t18 pass count, $t19 fail count
```

Static validation rechecked:

```text
three target-local breakpoints = PASS
MASM operators                = PASS
quotes/braces                 = PASS
51 PASS marker sites          = PASS
60 FAIL marker sites          = PASS
Family A-G                    = present
Family C negative logic       = PASS
fail-closed q paths           = PASS
AudioEngineHarness reference  = 0
Retry-3 reference             = 0
```

## 4. Approved command

The only approved future command is:

```text
C:\VSC_Project\ConvoPeq\tmp\cdb.exe
  -logo C:\VSC_Project\ConvoPeq\doc\work113\P1-5-IR-P2_Step5BL_P3-5-FPM-C3-T1-Capture-PostRetry2_Benign-Probe-Execution-1_CDB.log
  -cf C:\VSC_Project\ConvoPeq\doc\work113\P1-5-IR-P2_Step5BK_P3-5-FPM-C3-T1-Capture-PostRetry2_Benign_Probe_Runtime_Authorization-1.cdb
  C:\Windows\System32\ping.exe
  127.0.0.1
  -n
  32
```

The log path is reserved for this one attempt. A different script, target, arguments, log path, debugger, or target identity is not covered by this approval.

## 5. One-attempt consumption rule

```text
Approved attempts         = 1
Maximum attempts         = 1
Execution in this gate   = 0
Remaining before next gate = 1
Remaining after any invocation = 0
Immediate rerun           = FORBIDDEN
Script edit               = FORBIDDEN
Second execution attempt  = FORBIDDEN
```

The first invocation consumes this approval regardless of its result. A runtime failure, parser failure, marker failure, or process crash does not create a second authorized attempt.

## 6. Required pre-execution checks

The next execution gate must run these checks immediately before the one invocation:

```text
ping SHA-256    = E4224D18C3C96826E6F4240893FAC02C8A1FF335CE92614FC8744369594DD468
CDB SHA-256     = 5F54ABAFCA3AE5638BBF807D402FABB350A64575C1DFA9FBFC7F5732DF5BEE67
script SHA-256 = 5C12E1515CC2206430F34ADC7B8FE39D84A04BF4B33F2D7628970C395EC7699B
CDB version    = 10.0.29617.1000
CDB/ping/AudioEngineHarness process residue = 0
log path does not already exist
```

Any mismatch is a STOP before invocation and does not consume the attempt.

## 7. Required runtime evidence

The execution gate must preserve the CDB log and determine these independently:

```text
registration of all three bp commands
actual target hit at one of the three RVAs
no Syntax error / Bad register / Couldn't resolve / Operand error
each standalone PASS marker for families A-G
branch selection correctness
dd/dq display output
PASS_TERMINATION_Q if all checks pass
clean q / quit: termination
CDB/ping process residue = 0 after termination
```

The following are not acceptable substitutes:

```text
bu registration without hit
bl listing without execution
command echo instead of standalone marker
CDB exit code without marker evidence
historical Retry-1/Retry-2 evidence for this ping run
```

## 8. Runtime result classification

A successful execution must classify exactly one of:

```text
PROBE_RUNTIME_PASS
  = all expected PASS markers + PASS_TERMINATION_Q + clean termination

PROBE_RUNTIME_STOP
  = any FAIL marker, parser/register/operand error, missing marker,
    no clean termination, or unresolved hit/registration state
```

A STOP does not authorize a second attempt. Runtime evidence may update only CDB syntax/command-execution knowledge. It must not update S7 reader attribution.

## 9. Scope and evidence boundary

This approval applies only to:

```text
benign ping probe
command-expression semantics
hit-time branch/marker/termination behavior
```

It does not apply to:

```text
AudioEngineHarness
Retry-3
ConvoPeq production capture
S7 reader attribution
Case A/B/C/D
source/test/CMake
build
Dr.Memory
M1/M2
implementation
```

## 10. Final authorization state

```text
P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Execution-Approval
= CLOSED / ONE BENIGN PROBE ATTEMPT AUTHORIZED

Approved target = C:\Windows\System32\ping.exe 127.0.0.1 -n 32
Approved script SHA-256 = 5C12E1515CC2206430F34ADC7B8FE39D84A04BF4B33F2D7628970C395EC7699B
Approved maximum attempts = 1
Execution performed in this gate = 0
AudioEngineHarness = FORBIDDEN
Retry-3 = FORBIDDEN
Production capture = FORBIDDEN
Implementation = FORBIDDEN
DIRECT_DQUEUE_CAUSE = epoch gate equality PROVEN
S7_READER = UNRESOLVED
S7_READER_SLOT = UNRESOLVED
minReaderEpoch = NOT_CAPTURED
Case A/B/C/D = NOT_PROVEN
```

The next gate is the one-time benign probe execution. Its result must be preserved and classified; it must not be followed by an implicit rerun or production capture.
