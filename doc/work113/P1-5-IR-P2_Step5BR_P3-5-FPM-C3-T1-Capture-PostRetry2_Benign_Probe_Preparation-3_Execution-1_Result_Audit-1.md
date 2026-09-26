# P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Preparation-3-Execution-1 / Result-Audit-1

## 1. Gate result

```text
Gate                       = P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Preparation-3-Execution-1
Audit                      = P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Preparation-3-Result-Audit-1
Authorization              = CONSUMED
Execution attempts         = 1
CDB execution              = 1
ping execution             = 1
AudioEngineHarness         = 0
Retry-3                    = 0
production capture         = 0
script modification        = 0
source/test/CMake/build/Dr.Memory/M1/M2 = 0
VERDICT                    = PREPARATION-3 FAILED / STOP

Candidate A anchors        = PROVEN (all five == image base + RVA, sentinel cleared)
Candidate B (@$pc)         = FAILED (Bad register error)
A==B comparison criteria   = NOT SATISFIED (never executed)
```

The run is classified FAILED. No repair and no rerun occur in this gate.

## 2. Preflight (all passed before invocation)

| item | expected | observed | result |
|---|---|---|---|
| `ConvoPeq.md` | `E5E74200...EF3609` | identical | PASS |
| `ping.exe` | `E4224D18...DD468` | identical | PASS |
| `cdb.exe` | `5F54ABAF...5BEE67` | identical | PASS |
| Preparation-3 script | `671F9D45...D63771` | identical, 2,646 bytes | PASS |
| CDB version | `10.0.29617.1000` | `10.0.29617.1000` | PASS |
| process residue | 0 | 0 | PASS |
| execution log | absent | absent | PASS |

## 3. Frozen execution log

```text
Path    = doc/work113/P1-5-IR-P2_Step5BP_P3-5-FPM-C3-T1-Capture-PostRetry2_Benign_Probe_Preparation-3_Execution-1_CDB.log
SHA-256 = 4E2BB488253B7FAE83579576DBAE9416777EC2BC6326D4113DDB278488E4BBB4
bytes   = 11,590
```

Script identity after execution: `671F9D45...D63771` (unchanged). `ConvoPeq.md` after execution: `E5E74200...EF3609` (unchanged).

## 4. R2 checks

```text
R2-CHECK-A  bl registered exactly one breakpoint (id 0)     PASS
R2-CHECK-B  C3BP3_BASE_DERIVED emitted                       FAIL (count = 0)
R2-CHECK-C  PASS_TERMINATION_Q once, FAIL_* zero            FAIL (Q = 0, FAIL_* = 0)
R2-CHECK-D  actual breakpoint hit                           PASS (ping+0x39d9, line 133)
```

`C3BP3_BASE_DERIVED` count is 0, which is the decisive evidence that the breakpoint body terminated at its first statement. The breakpoint did fire, but its body never reached the echo.

## 5. Candidate A — anchor establishment

Observed image base at this run:

```text
ModLoad: 00007ff7`1cd00000 00007ff7`1cd0c000   ping.exe
base    = 0x7FF71CD00000
```

Unconditional readback emitted by the script (this is the gap the earlier run lacked):

```text
$t1=00007ff71cd0003c $t2=00007ff71cd04fd0 $t3=00007ff71cd02600 $t4=00007ff71cd05000 $t5=00007ff71cd00040
```

Verified against `base + RVA`:

| register | observed | expected (base+RVA) | result |
|---|---|---|---|
| `$t1` | `0x7FF71CD0003C` | `0x7FF71CD0003C` (RVA `0x3C`) | MATCH |
| `$t2` | `0x7FF71CD04FD0` | `0x7FF71CD04FD0` (RVA `0x4FD0`) | MATCH |
| `$t3` | `0x7FF71CD02600` | `0x7FF71CD02600` (RVA `0x2600`) | MATCH |
| `$t4` | `0x7FF71CD05000` | `0x7FF71CD05000` (RVA `0x5000`) | MATCH |
| `$t5` | `0x7FF71CD00040` | `0x7FF71CD00040` (RVA `0x40`) | MATCH |

```text
sentinel cleared = YES ($t1 != 0xDEAD0001)
Candidate A      = PROVEN
```

This is the substantive positive result of the run: the `ping+offset` form is accepted as an `r` right-hand side, and every anchor resolves to the intended image-relative address.

## 6. Candidate B — observed failure

The breakpoint body failed at its first state assignment:

```text
Bad register error at '@$pc-0x39d9; r $t12=@$t11+0x3c; ...'
```

`@$pc` is not a valid register in this CDB build. Per Microsoft Learn the instruction pointer is `$ip`, `.`, or the architecture register (`@rip`); `$pc` is not in the documented pseudo-register list.

This is exactly the R1 risk recorded in the authorization review. It failed closed as predicted: no false PASS was produced, `$t11` was never established, and the T1–T5 comparisons were never evaluated.

```text
Candidate B = FAILED / NOT RUNTIME-PROVEN
```

## 7. Secondary observation: comment syntax

The log contains 5 `Couldn't resolve error` and several `pass count must be preceeded by whitespace` / `Syntax error` messages, all originating from `;` comment lines in the frozen script.

```text
Couldn't resolve error count = 5
Syntax error count           = 0 (as standalone leading-token errors)
```

These are non-fatal: the script continued past every one of them and all functional commands executed. This indicates `;` is not treated as a comment introducer in a CDB `-cf` command file. It is recorded here as a defect for the Failure Audit to consider; it did not cause the STOP.

## 8. Full error taxonomy for this run

```text
Bad register error  = 1   (fatal: @$pc, this is the STOP cause)
Memory access error = 0
Syntax error        = 0
Couldn't resolve    = 5   (non-fatal, comment lines)
Operand error       = 0
```

## 9. Marker summary

```text
C3BP3_SCRIPT_BEGIN        = 1
C3BP3_SENTINEL_SET        = 2 (echo + log duplication)
C3BP3_ASSIGN_DONE         = 2
C3BP3_READBACK_DONE       = 2
C3BP3_BP_SET              = 1
C3BP3_BASE_DERIVED        = 0   <-- decisive
PASS_T1 .. PASS_T5        = 0
PASS_SENTINEL_CLEARED     = 0
PASS_TERMINATION_Q        = 0
FAIL_*                    = 0
C3BP3_ALREADY_HIT         = 0
C3BP3_SCRIPT_END          = 1
quit:                     = 1
process residue           = 0
```

The `FAIL_*` count of 0 does not indicate success. The body aborted before reaching any conditional, so no failure branch could execute.

## 10. Success criteria evaluation

| criterion | result |
|---|---|
| T1 `$t1 == $t12` | NOT EVALUATED |
| T2 `$t2 == $t13` | NOT EVALUATED |
| T3 `$t3 == $t14` | NOT EVALUATED |
| T4 `$t4 == $t15` | NOT EVALUATED |
| T5 `$t5 == $t6` | NOT EVALUATED |
| sentinel cleared | SATISFIED (independent of body) |
| breakpoint registered | SATISFIED |
| actual hit | SATISFIED |
| clean q | SATISFIED |
| process residue 0 | SATISFIED |
| overall | FAILED |

The frozen criteria required the A==B comparisons. They were never evaluated, so the gate cannot be declared PROVEN.

## 11. Evidence boundary

Established by this run:

```text
r $tN = ping+offset  is accepted as an assignment RHS
all five anchors resolve to image base + RVA
unconditional pseudo-register readback works
breakpoint at ping+0x39d9 fires and stops
@$pc is rejected as a register in CDB 10.0.29617.1000
```

Not established:

```text
Candidate B base-derivation scheme   (unimplemented at runtime)
A==B agreement                       (never evaluated)
dwo / poi / & / >> / ring arithmetic / and / dd / dq
Family A-G logic
```

## 12. What remains unchanged

```text
DIRECT_DQUEUE_CAUSE = epoch gate equality PROVEN
S7_READER          = UNRESOLVED
S7_READER_SLOT     = UNRESOLVED
minReaderEpoch     = NOT_CAPTURED
Case A/B/C/D       = NOT_PROVEN
IMPLEMENTATION     = FORBIDDEN
```

The RT-side observation boundary and the NonRT lifetime isolation line are unchanged. This probe added no ConvoPeq reader, epoch, or reclaim evidence.

## 13. Final state

```text
Preparation-3-Execution-1      = CONSUMED / FAILED
Preparation-3 anchor establishment (gate-level) = NOT PROVEN
Candidate A anchor establishment               = PROVEN
Candidate B                                    = FAILED (@$pc Bad register)
log          = 4E2BB488...4BBB4 (11,590 bytes)
script       = 671F9D45...D63771 (unchanged)
production source = E5E74200...EF3609 (unchanged)
process residue   = 0

Preparation-4   = NOT DESIGNED / BLOCKED
Retry-3          = FORBIDDEN
production C3    = FORBIDDEN
AudioEngineHarness = FORBIDDEN
M1 / M2 / build / Dr.Memory = FORBIDDEN
repair or rerun  = NOT PERFORMED
next step        = Failure Audit (separate gate)
```
