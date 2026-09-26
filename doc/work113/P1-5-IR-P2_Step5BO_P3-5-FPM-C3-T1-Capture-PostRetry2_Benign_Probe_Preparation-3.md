# P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Preparation-3

## 1. Gate result

```text
Gate                                   = P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Preparation-3
Mode                                   = design + static validation only
CDB execution                          = 0
ping execution                         = 0
AudioEngineHarness                     = 0
Retry-3                                = 0
production capture                     = 0
existing script modification           = 0
source modification                    = 0
build                                  = 0
Dr.Memory                              = 0
prior probe authorization              = CONSUMED
VERDICT                                = DESIGN COMPLETE / STATICALLY VALIDATED / NOT EXECUTED
```

This gate produced a design artifact and a static validation. It did not start CDB, did not run `ping`, and does not authorize any execution.

## 2. Re-fixed identities

| artifact | SHA-256 | bytes | result |
|---|---|---|---|
| `ConvoPeq.md` (production source authority) | `E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609` | 5,535,334 | PASS |
| `C:\Windows\System32\ping.exe` | `E4224D18C3C96826E6F4240893FAC02C8A1FF335CE92614FC8744369594DD468` | — | PASS |
| `tmp\cdb.exe` | `5F54ABAFCA3AE5638BBF807D402FABB350A64575C1DFA9FBFC7F5732DF5BEE67` | — | PASS |

```text
CDB version = 10.0.29617.1000
consumed script = 5C12E1515CC2206430F34ADC7B8FE39D84A04BF4B33F2D7628970C395EC7699B (unchanged)
consumed log    = BCFE1398DB6896C3C1FBB381C65C43EBD48ECAC5ACAE3D75F213F595B1C5DD2B (unchanged)
process residue = 0
```

The prior authorization script and its log remain byte-identical. This gate adds a new file only.

## 3. Root cause now grounded in documentation

The previous audit correctly refused to name a cause. This gate grounds it.

Microsoft Learn, *Symbol Syntax and Symbol Matching*:

> A symbol name may be qualified by a module name. An exclamation mark (`!`) separates the module name from the symbol (for instance, **mymodule!main**).

Microsoft Learn, *Address and Address Range Syntax*:

> **name[ + | − ] offset** — A flat 32-bit or 64-bit address. *name* can be any symbol.

Therefore:

```text
ping!+0x3c     = module 'ping' + '!' + EMPTY symbol name + offset   -> malformed
ping+0x1184    = name 'ping' + offset                              -> documented form
```

This is consistent with the consumed log, where `bp ping+0x1184` resolved to a concrete address while `r $t1 = ping!+0x3c` returned `Syntax error at 'ping!+0x3c'`. The failure is a malformed module-qualified form, not a missing-symbol condition. The `.reload /f ping.exe` "no symbol file" line is a separate fact and is not required to explain the rejection.

Per Microsoft Learn, the 20 user-defined pseudo-registers `$t0`..`$t19` hold integers and default to zero, and `r` writes them. The `r` command's second argument follows the default expression syntax, which was MASM in the consumed run.

## 4. Anchor validation against the real image

`ping.exe` is PE32+, `e_lfanew=0x100`, `SizeOfImage=0xC000`, `SizeOfHeaders=0x1000`, section alignment 0x1000.

| register | RVA | region | in image | role |
|---|---|---|---|---|
| `$t1` | `0x3C` | DOS header | yes | `e_lfanew` slot |
| `$t2` | `0x4FD0` | `fothk` (0x4000–0x5000) | yes | Family G display base |
| `$t3` | `0x2600` | `.text` (0x1000–0x3670) | yes | Family G display base |
| `$t4` | `0x5000` | `.rdata` (VA 0x5000) | yes | `.rdata` base |
| `$t5` | `0x40` | DOS header | yes | arithmetic base |

Additional checks:

```text
dword at RVA 0x3C = 0x100 = e_lfanew  -> consistent with frozen Family A expectation 0x100
Family E computed RVA = 0x40 + 0xC0 + 4*0x30 = 0x1C0, inside headers, valid
RVA 0x30400 (discarded Family G target) is outside SizeOfImage and remains invalid
```

A correction to an earlier working note: an intermediate section-based check suggested `0x1C0` was out of image. That was wrong, because section tables do not cover the header region. `0x1C0` is valid. The only genuinely invalid RVA in the lineage is `0x30400`.

The frozen anchor set is therefore sound; only the way it was written was broken.

## 5. Designed probe

Artifact:

```text
doc/work113/P1-5-IR-P2_Step5BO_P3-5-FPM-C3-T1-Capture-PostRetry2_Benign_Probe_Preparation-3_Anchor_Only_Probe.cdb
SHA-256 = 671F9D453FF814DD2AE01217A6CAE7F69E0D033A7A0AA70BD42D21E797D63771
bytes   = 2,646
```

Sequence, matching the required nine steps:

```text
1  CDB start
2  ping load confirmed by breakpoint resolution at ping+0x39d9
3  .expr /s masm
4  sentinel assignment ($t1..$t5 = 0xDEAD0001..0xDEAD0005)
5  anchor assignment (Candidate A) and base derivation (Candidate B)
6  immediate unconditional readback: r $t1, $t2, $t3, $t4, $t5
7  value classification at breakpoint hit
8  PASS/FAIL markers per anchor
9  clean termination
```

### Candidate A — module name + offset, no exclamation mark

```text
r $t1 = ping+0x3c
r $t2 = ping+0x4FD0
r $t3 = ping+0x2600
r $t4 = ping+0x5000
r $t5 = ping+0x40
```

This is the documented `name[+|offset]` form, already proven acceptable as a `bp` address argument in the consumed run.

### Candidate B — base derived from `@pc` at hit time

```text
r $t11=@$pc-0x39d9
r $t12=@$t11+0x3c
r $t13=@$t11+0x4FD0
r $t14=@$t11+0x2600
r $t15=@$t11+0x5000
r $t6=@$t11+0x40
```

No module-qualified symbol appears in this path. It reconstructs every anchor from the instruction pointer at the already-proven hit.

### Cross-validation

The breakpoint body compares each Candidate A value against its Candidate B derivation:

```text
PASS_T1 : @$t1 == @$t12
PASS_T2 : @$t2 == @$t13
PASS_T3 : @$t3 == @$t14
PASS_T4 : @$t4 == @$t15
PASS_T5 : @$t5 == @$t6
PASS_SENTINEL_CLEARED : @$t1 != 0xDEAD0001
PASS_TERMINATION_Q : @$t18 == 6
```

Any anchor mismatch quits immediately with `FAIL_T*`. Because no memory is read, an incorrect anchor produces a clean FAIL rather than a dereference error.

### The gap this design closes

The consumed run left `$t1..$t5` post-failure values as `NOT ESTABLISHED BY LOG` because no dump was ever requested. This design makes the readback unconditional, and uses distinct sentinels so a rejected assignment is distinguishable from a coincidentally correct value.

## 6. Explicit exclusions

The design deliberately does not test any of the following in this gate:

```text
dwo   poi   dd   dq   dereference   ring arithmetic   MASM and
64-bit low/high split
Family A/B/C/D/E/F/G logic
termination contract beyond the anchor count
```

Anchor construction and memory-expression evaluation remain fully separated. `dwo` stays `NOT PROVEN`; it is scheduled for Preparation-4.

## 7. Static validation result

Automated structural validation over the artifact:

```text
no star-prefixed (invalid) comment lines          PASS
no dwo( call                                      PASS
no poi( call                                      PASS
no dd / dq display command                        PASS
no memory read inside breakpoint body             PASS
five distinct sentinels present                   PASS
unconditional readback line present               PASS
readback ordered after assignment                  PASS
Candidate A uses ping+offset with no '!'          PASS
no 'ping!' form anywhere in code                  PASS
Candidate B derives base from @pc                 PASS
all PASS/FAIL markers present                     PASS
termination threshold == 6                        PASS
exactly one breakpoint                            PASS
all five anchor RVAs inside image                 PASS
$DWORD at RVA 0x3C == 0x100                       PASS
script terminates with q                          PASS
FAILURES = 0
```

## 8. Success criteria and next gate

Preparation-3 success criterion, if and when separately authorized:

```text
$t1..$t5 anchor establishment = PROVEN
```

Sequenced continuation, unchanged from instruction:

```text
Preparation-3  anchor establishment      (this gate, design only)
      |
      v
static validation                        (complete)
      |
      v
RESULT AUDIT
      |
      v
separate runtime authorization
      |
      v
exactly one benign execution
      |
      v
result audit
      |
      v
Preparation-4  dwo only
      |
      v
Preparation-5  poi / & / >>
      |
      v
Preparation-6  ring arithmetic / and
      |
      v
Preparation-7  dd / dq display expressions
      |
      v
production C3 capture redesign
```

## 9. What remains unchanged

```text
DIRECT_DQUEUE_CAUSE = epoch gate equality PROVEN
S7_READER          = UNRESOLVED
S7_READER_SLOT     = UNRESOLVED
minReaderEpoch     = NOT_CAPTURED
Case A/B/C/D       = NOT_PROVEN
IMPLEMENTATION     = FORBIDDEN
```

No ConvoPeq reader, epoch, or reclaim evidence was produced or altered.

## 10. Final state

```text
P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Preparation-3
= CLOSED / DESIGN COMPLETE / STATICALLY VALIDATED / NOT EXECUTED

failure primitive root cause = malformed module-qualified form 'ping!+offset'
                             = '!' requires a symbol name; documented form is 'name+offset'
anchor set validated          = all five RVAs inside ping.exe image
Candidate A                   = ping+offset, no exclamation mark
Candidate B                   = base derived from @pc at hit time
readback                      = unconditional, with distinct sentinels
memory dereference in probe   = none
CDB execution                 = 0
runtime authorization granted = NO
prior authorization           = CONSUMED / NOT REUSABLE
Retry-3                       = FORBIDDEN
AudioEngineHarness            = FORBIDDEN
Production capture            = FORBIDDEN
Implementation                = FORBIDDEN
```
