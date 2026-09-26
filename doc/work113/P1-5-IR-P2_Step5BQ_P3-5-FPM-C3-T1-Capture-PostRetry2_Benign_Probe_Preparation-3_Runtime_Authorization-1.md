# P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Preparation-3-Runtime-Authorization-1

## 1. Gate result

```text
Gate                                  = P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Preparation-3-Runtime-Authorization-1
Mode                                  = read-only authorization review
CDB execution                         = 0
ping execution                        = 0
AudioEngineHarness                    = 0
Preparation-4                         = 0 (not designed)
Retry-3 naming                        = NOT USED
production capture                    = 0
script modification                   = 0
source/test/CMake/build/Dr.Memory/M1/M2 = 0
VERDICT                               = ONE ATTEMPT AUTHORIZED (CONDITIONAL ON FROZEN IDENTITY RECHECK)
```

This gate reviewed the frozen Preparation-3 script and did not execute anything. It does not itself constitute the execution; the one permitted run is a separate gate.

## 2. Frozen identity recheck

| artifact | expected SHA-256 | observed | bytes | result |
|---|---|---|---|---|
| `ConvoPeq.md` | `E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609` | identical | 5,535,334 | PASS |
| `C:\Windows\System32\ping.exe` | `E4224D18C3C96826E6F4240893FAC02C8A1FF335CE92614FC8744369594DD468` | identical | — | PASS |
| `tmp\cdb.exe` | `5F54ABAFCA3AE5638BBF807D402FABB350A64575C1DFA9FBFC7F5732DF5BEE67` | identical | — | PASS |
| Preparation-3 script | `671F9D453FF814DD2AE01217A6CAE7F69E0D033A7A0AA70BD42D21E797D63771` | identical | 2,646 | PASS |

```text
CDB version            = 10.0.29617.1000
process residue        = 0
reserved log path      = absent (not yet created)
prior script / log     = unchanged (5C12E151...7699B / BCFE1398...1C5DD2B)
```

All six required checks passed. The script was read only and not modified.

## 3. Candidate A review

```text
r $t1 = ping+0x3c
r $t2 = ping+0x4FD0
r $t3 = ping+0x2600
r $t4 = ping+0x5000
r $t5 = ping+0x40
```

Static findings:

```text
all five assignments present                          PASS
no 'ping!' form anywhere in any code line             PASS
no module-qualified RHS assignment remains            PASS
uses documented name[+|offset] form                   PASS
```

Preparation-3 completely eliminates the form that failed in the consumed run. The `!` misuse is absent, so the known failure primitive is not reproduced by construction.

## 4. Candidate B review

```text
r $t11=@$pc-0x39d9
r $t12=@$t11+0x3c
r $t13=@$t11+0x4FD0
r $t14=@$t11+0x2600
r $t15=@$t11+0x5000
r $t6=@$t11+0x40
```

Static findings:

```text
all six derivations present                          PASS
contiguous inside the single breakpoint body         PASS
block contains no module symbol reference             PASS
derived constant 0x39d9 equals the bp RVA             PASS
not hard-coded to a specific register name            PASS
```

Candidate B is genuinely independent of symbol resolution: it reconstructs every anchor from the instruction pointer at the hit.

```text
Candidate A = DESIGN CANDIDATE / STATICALLY VALIDATED
Candidate B = DESIGN CANDIDATE / STATICALLY VALIDATED / NOT RUNTIME-PROVEN
```

Neither is asserted correct in advance. Any mismatch is a FAIL, not an expected outcome.

## 5. `@pc` risk recorded before execution

Microsoft Learn, *MASM Numbers and Operators* and *Pseudo-Register Syntax*, document the instruction pointer as:

```text
$ip   "the instruction pointer register; x64: same as rip"
.     "current instruction pointer; same meaning as $ip; no @ sign"
@rip  architecture register
```

`$pc` does not appear in the documented pseudo-register list. The frozen script uses `@$pc`. Per instruction the script is not modified, so this is carried forward as a declared risk:

```text
RISK R1 = @$pc may be rejected or undefined in this CDB build
          If so, $t11 fails to derive, all T1..T5 comparisons fail, and
          the run reports FAIL_T1 rather than producing false evidence.
          Impact on validity = NONE (failure is fail-closed)
          Impact on schedule  = Candidate B is not confirmed; Preparation-4
                                may still be designed on Candidate A alone
```

Because the comparison is symmetric, an undefined `@$pc` cannot manufacture a PASS. This risk does not block authorization.

## 6. Success criteria (frozen, not extensible)

```text
T1 : Candidate A $t1 == Candidate B $t12
T2 : Candidate A $t2 == Candidate B $t13
T3 : Candidate A $t3 == Candidate B $t14
T4 : Candidate A $t4 == Candidate B $t15
T5 : Candidate A $t5 == Candidate B $t6
plus sentinel cleared, 5 anchor markers, PASS_TERMINATION_Q, clean q, residue 0
```

Explicitly out of scope and not added to the criteria:

```text
dwo   poi   dd   dq   dereference   ring arithmetic   MASM and
64-bit low/high split   Family A/B/C/D/E/F/G logic
```

The probe performs no memory read. If an anchor were wrong, the result is a clean FAIL, never a dereference error.

## 7. Failure conditions

Any one of the following is an immediate STOP, with no repair and no rerun in the same gate:

```text
ping+offset assignment failure
@pc / $t11 derivation failure
Candidate A/B mismatch
sentinel still present at hit
unexpected memory access error
unexpected CDB syntax error
no breakpoint hit
process residue != 0
```

## 8. Hard boundary for the single permitted execution

```text
maximum execution attempts = 1
target                    = C:\Windows\System32\ping.exe 127.0.0.1 -n 32
CDB                       = tmp\cdb.exe 10.0.29617.1000 (frozen SHA)
script                    = Preparation-3 frozen SHA 671F9D45...D63771
log                       = doc/work113/P1-5-IR-P2_Step5BP_P3-5-FPM-C3-T1-Capture-PostRetry2_Benign_Probe_Preparation-3_Execution-1_CDB.log
production source         = untouched
AudioEngineHarness        = 0
Retry-3                   = 0
production capture        = 0
M1 / M2                   = 0
build                     = 0
Dr.Memory                 = 0
```

This authorization covers the Preparation-3 benign anchor probe, one attempt only. It does not include Preparation-4, any expression family beyond anchor establishment, or the production C3 capture.

The prior authorization from the earlier execution remains consumed and is not reusable. This is a new, single-use authorization for a different, narrower artifact.

## 9. Residual risk R2: unresolved breakpoint yields an empty pass-like run

The script sets one breakpoint, dumps `bl`, then runs `g` and finally `q`. It contains no `bd`, `bc`, or `bl -1` safety net.

If `ping+0x39d9` fails to resolve, no breakpoint is installed, `ping` runs to completion, and the script still reaches its final `q`. The process boundary would look clean while no marker was ever produced.

This is not a silent false PASS, because the success criteria require six markers plus `PASS_TERMINATION_Q`; their absence fails the run. It is recorded so the result audit performs the explicit checks:

```text
R2-CHECK-A : 'bl' shows exactly one registered breakpoint
R2-CHECK-B : C3BP3_BASE_DERIVED is present (proves the body actually executed)
R2-CHECK-C : exactly one PASS_TERMINATION_Q and no FAIL_* marker
R2-CHECK-D : a hit was recorded, not merely a completed process
```

`R2-CHECK-B` is the decisive one: a breakpoint that never fired cannot emit `C3BP3_BASE_DERIVED`.

## 10. Command inventory of the frozen script

```text
 1 .echo C3BP3_SCRIPT_BEGIN
 2 .sympath+ C:\VSC_Project\ConvoPeq
 3 .expr /s masm
 4 r $t1 = 0xDEAD0001
 5 r $t2 = 0xDEAD0002
 6 r $t3 = 0xDEAD0003
 7 r $t4 = 0xDEAD0004
 8 r $t5 = 0xDEAD0005
 9 r $t16 = 0
10 r $t17 = 0
11 r $t18 = 0
12 r $t19 = 0
13 .echo C3BP3_SENTINEL_SET
14 r $t1 = ping+0x3c
15 r $t2 = ping+0x4FD0
16 r $t3 = ping+0x2600
17 r $t4 = ping+0x5000
18 r $t5 = ping+0x40
19 .echo C3BP3_ASSIGN_DONE
20 r $t1, $t2, $t3, $t4, $t5
21 .echo C3BP3_READBACK_DONE
22 bp ping+0x39d9 "<anchor body>"
23 .echo C3BP3_BP_SET
24 bl
25 g
26 .echo C3BP3_SCRIPT_END
27 q
```

27 commands. No `gc`, no `dd`/`dq`, no `dwo`/`poi`, no harness reference, no ConvoPeq target. The only ConvoPeq string is the symbol search path.

Static validation: FAILURES = 0.

## 11. What remains unchanged

```text
DIRECT_DQUEUE_CAUSE = epoch gate equality PROVEN
S7_READER          = UNRESOLVED
S7_READER_SLOT     = UNRESOLVED
minReaderEpoch     = NOT_CAPTURED
Case A/B/C/D       = NOT_PROVEN
IMPLEMENTATION     = FORBIDDEN
```

## 12. Final state

```text
P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Preparation-3-Runtime-Authorization-1
= CLOSED / ONE ATTEMPT AUTHORIZED / EXECUTION NOT PERFORMED IN THIS GATE

identities rechecked              = 6 / 6 PASS
script modified                   = NO
Candidate A                       = ping+offset, no '!' , STATICALLY VALIDATED
Candidate B                       = @pc-derived, no module symbol, NOT RUNTIME-PROVEN
success criteria                  = T1..T5 A==B only, not extensible
memory reads in probe             = 0
CDB execution in this gate        = 0
one-time benign execution         = AUTHORIZED, separate gate
Preparation-4                     = NOT DESIGNED
Retry-3                           = NOT USED
Production capture                = FORBIDDEN
AudioEngineHarness                = FORBIDDEN
Implementation                    = FORBIDDEN
```

Next permitted step, in order:

```text
one-time benign execution (exactly 1)
      |
      v
Preparation-3 Result Audit  -- PASS -> Preparation-4 design
                            \-- FAIL -> failure audit, no repair, no rerun
```
