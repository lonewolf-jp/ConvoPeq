# P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Candidate-B-Replacement-Runtime-Authorization-1

## 1. Gate result

```text
Gate                     = P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Candidate-B-Replacement-Runtime-Authorization-1
Mode                     = authorization review only
CDB execution            = 0
ping execution           = 0
new .cdb created         = 0
script modified          = 0
AudioEngineHarness       = 0
Retry-3                  = 0
production C3 capture    = 0
Preparation-4            = 0 (still BLOCKED)
M1/M2/build/Dr.Memory    = 0
rerun                    = 0
VERDICT                  = ONE INVOCATION AUTHORIZED (CONDITIONAL ON PRE-EXECUTION IDENTITY RECHECK)
```

This gate authorizes a future single benign execution. It performs no execution itself.

## 2. Frozen inputs

| artifact | fixed value | observed | result |
|---|---|---|---|
| `ConvoPeq.md` | `E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609` | identical | PASS |
| B1 probe script | `403C8E2BBDAACC5C8010A664A8188C87A7167D28209F203D79CAA418BBEB8329` | identical | PASS |
| B1 probe bytes | `1,481` | `1,481` | PASS |
| `cdb.exe` | `5F54ABAFCA3AE5638BBF807D402FABB350A64575C1DFA9FBFC7F5732DF5BEE67` | identical | PASS |
| CDB version | `10.0.29617.1000` | `10.0.29617.1000` | PASS |
| `ping.exe` | `E4224D18C3C96826E6F4240893FAC02C8A1FF335CE92614FC8744369594DD468` | identical | PASS |
| process residue | `0` | `0` | PASS |
| reserved log | absent | absent | PASS |

```text
breakpoint = ping+0x39d9
B1 form    = @$bp0 - 0x39d9
```

Any single identity drift is a STOP before invocation. No value may be refreshed in place.

## 3. Script content re-confirmation

Independently re-checked at this gate, not inherited from the static validation:

| item | expected | observed | result |
|---|---|---|---|
| `@$pc` | 0 | 0 | PASS |
| `;` comment lines | 0 | 0 | PASS |
| `*` comment lines | 0 | 0 | PASS |
| `dwo` | 0 | 0 | PASS |
| `poi` | 0 | 0 | PASS |
| `dd` / `dq` | 0 | 0 | PASS |
| ReaderSlot access | 0 | 0 | PASS |
| epoch / reclaim capture | 0 | 0 | PASS |
| AudioEngineHarness / Retry-3 | 0 | 0 | PASS |
| `@$bp0` occurrences | exactly 1 | 1 | PASS |
| B1 form present | `@$bp0-0x39d9` | present | PASS |
| breakpoints | 1, at `ping+0x39d9` | 1 | PASS |
| Candidate A anchors | 5/5 | 5/5 | PASS |
| Candidate A FAIL path | absent | absent | PASS |
| `DISAGREE` branches | 5 | 5 | PASS |
| final command | `q` | `q` | PASS |
| hit / derived markers | present | present | PASS |
| unconditional readback | present | present | PASS |
| `$t11` and `$t1..$t5` dumped at hit | present | present | PASS |
| counters dumped before `q` | present | present | PASS |

```text
checks = 20 / 20 PASS
script edited at this gate = NO
```

## 4. Authorized scope

```text
Target    = C:\Windows\System32\ping.exe
Arguments = 127.0.0.1 -n 32
CDB       = tmp\cdb.exe (frozen binary and version)
Script    = B1 anchor probe (frozen SHA)
Log       = doc/work113/P1-5-IR-P2_Step5BV_P3-5-FPM-C3-T1-Capture-PostRetry2-Benign_Probe_Candidate-B1_Execution-1_CDB.log
```

```text
maximum CDB invocations  = 1
maximum ping invocations = 1
```

The authorization is consumed the moment CDB is launched, regardless of outcome. A consumed authorization is not reusable, and no retry of this probe is authorized by this document.

## 5. Runtime verdict contract

The sole objective is B1.

```text
B1 = PROVEN   requires all of:
                 breakpoint hit occurred
                 C3B1_BREAKPOINT_HIT emitted
                 C3B1_CANDIDATE_B1_DERIVED emitted
                 $bp0 yielded a non-zero, valid breakpoint address
                 $t11 = $bp0 - 0x39d9 equals the observed image base
                 $t12..$t15 and $t6 agree with Candidate A $t1..$t5

B1 = FAILED   any of:
                 Bad register error on @$bp0
                 $bp0 evaluates to zero or an implausible address
                 expression failure in the B1 arithmetic
                 B1 anchor mismatch against Candidate A

B1 = INCONCLUSIVE  if:
                 no breakpoint hit occurred
                 C3B1_BREAKPOINT_HIT absent
```

`B1 = FAILED` never demotes Candidate A. Candidate A remains PROVEN in every outcome, because it is already established independently against the observed image base and its FAIL path does not exist in the script.

A/B disagreement is recorded as evidence, not as a script failure. The script has no `FAIL_T*` branch by design.

## 6. R3 hazard and required audit checks

The script contains no `bd`, `bc`, or `bl -1` safety net. If `ping+0x39d9` failed to resolve, no breakpoint would be installed, `g` would run `ping` to completion, and the script would still reach its final `q`, producing a clean-looking log with zero evidence.

This is not a false PASS, because every PASS condition above requires marker evidence. It is recorded so the Result Audit performs these explicit checks:

```text
R3-CHECK-A  bl shows exactly one registered breakpoint
R3-CHECK-B  C3B1_BREAKPOINT_HIT present      (proves the body ran at a real hit)
R3-CHECK-C  C3B1_CANDIDATE_B1_DERIVED present (proves the @$bp0 line completed)
R3-CHECK-D  C3B1_CROSSCHECK_BEGIN present
R3-CHECK-E  XCHK_T*_AGREE / XCHK_T*_DISAGREE counts and the $t18 tally
R3-CHECK-F  error taxonomy attributed to @$bp0 only, not to anything else
R3-CHECK-G  process residue 0 after termination
```

R3-CHECK-B and R3-CHECK-C together are decisive: a breakpoint that never fired, or a body that aborted at `$bp0`, cannot emit both.

## 7. Not authorized by this document

```text
dwo / poi                              0
dd / dq                                0
ReaderSlot read                        0
minReaderEpoch capture                 0
T1 reclaim capture                     0
DQueue inspection                      0
epoch gate capture                     0
Retry-3                                FORBIDDEN
production C3                          FORBIDDEN
AudioEngineHarness                     FORBIDDEN
Preparation-4                          BLOCKED
M1 / M2                                FORBIDDEN
build                                  FORBIDDEN
Dr.Memory                              FORBIDDEN
source / test / CMake modification     FORBIDDEN
```

Even if B1 succeeds, this authorization does not permit proceeding to T1 capture. Anchor-construction validation and memory-expression validation remain separated.

## 8. Sequence

```text
Candidate-B Replacement Preparation-1            CLOSED
        |
        v
Candidate-B Replacement Static Validation-1      CLOSED
        |
        v
Runtime Authorization-1                          CLOSED  (this gate)
        |
        v
Benign Execution-1                               NOT STARTED  <- next
        |
        v
Result Audit-1                                   NOT STARTED
        |
        +-- B1 PROVEN -> next probe in the expression-family sequence
        |
        +-- B1 FAILED -> record; Candidate A remains PROVEN; decide next candidate
        |
        +-- B1 INCONCLUSIVE -> investigate breakpoint resolution; no verdict on B1
```

The result of Benign Execution-1 must not be used in place of Result Audit-1.

## 9. What remains unchanged

```text
DIRECT_DQUEUE_CAUSE = epoch gate equality PROVEN
Candidate A         = PROVEN
B1 static validity  = PROVEN
B1 runtime validity = NOT PROVEN
S7_READER          = UNRESOLVED
S7_READER_SLOT     = UNRESOLVED
minReaderEpoch     = NOT CAPTURED
T1 capture         = NOT CAPTURED
Case A/B/C/D       = NOT PROVEN
IMPLEMENTATION     = FORBIDDEN
```

The RT-side observation boundary and the NonRT lifetime isolation line are unchanged. No ConvoPeq reader, epoch, or reclaim evidence was produced or altered.

## 10. Final state

```text
P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Candidate-B-Replacement-Runtime-Authorization-1
= CLOSED / ONE INVOCATION AUTHORIZED / NOT EXECUTED IN THIS GATE

identities           = 8 / 8 PASS
script content      = 20 / 20 PASS
script modified     = NO
new .cdb            = 0
CDB execution       = 0
ping execution      = 0
maximum invocations = 1 (consumed on launch)
B1 runtime status   = NOT PROVEN
Candidate A         = PROVEN, unaffected by any B1 outcome
R3 hazard recorded  = yes, with R3-CHECK-A..G

Preparation-4 / Retry-3 / production C3 / AudioEngineHarness / M1 / M2 / build / Dr.Memory
= FORBIDDEN
rerun = FORBIDDEN
implementation = FORBIDDEN
```
