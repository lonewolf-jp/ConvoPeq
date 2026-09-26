# P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Candidate-B1-Corrected-Probe-Runtime-Authorization-1

## 1. Gate result

```text
Gate                    = P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Candidate-B1-Corrected-Probe-Runtime-Authorization-1
Mode                    = authorization review only
CDB execution           = 0
ping execution          = 0
script modification     = 0
new .cdb created        = 0
VERDICT                 = ONE INVOCATION AUTHORIZED (CONDITIONAL ON PRE-EXECUTION IDENTITY RECHECK)
```

This gate authorizes a single future benign execution. It performs none itself.

## 2. Source authority

```text
ConvoPeq.md SHA-256 = E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
ConvoPeq.md bytes   = 5,535,334
```

`ConvoPeq.md` is the only production source authority. The superseded `ConvoPeq(3).md` was not used as a substitute.

## 3. Frozen inputs

All eight were recomputed at this gate, not inherited.

| artifact | fixed value | observed | result |
|---|---|---|---|
| corrected B1 script | `FFC5228C2387BDC5264F4DB956D66E6DFFB46A97F97ADE2E59E0A7BA1AE739DE` | identical | PASS |
| script size | `1,172` bytes | `1,172` | PASS |
| `cdb.exe` | `5F54ABAFCA3AE5638BBF807D402FABB350A64575C1DFA9FBFC7F5732DF5BEE67` | identical | PASS |
| CDB version | `10.0.29617.1000` | `10.0.29617.1000` | PASS |
| `ping.exe` | `E4224D18C3C96826E6F4240893FAC02C8A1FF335CE92614FC8744369594DD468` | identical | PASS |
| `ConvoPeq.md` | `E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609` | identical | PASS |
| process residue | `0` | `0` | PASS |
| reserved log | absent | absent | PASS |

Any single drift is a STOP before invocation. No value may be refreshed in place, and the authorization may not be reused.

## 4. Fault localization re-derived from the caret

The previous Execution-1 log was re-measured at this gate, because the correctness of the
correction depends on the fault being in the right place and this is a one-shot authorization.

```text
caret line in Execution-1 log            = line 101
caret column                             = 399
leading characters before caret          = 399, all whitespace
first quote column                       = 426
text quoted in the error                 = 955 chars, begins " r $t16=1;", ends "... r $t19; q "
old body length                          = 1011 chars
old inner content (after ".if (...) {")  = 993 chars
omitted tail                             = 38 chars = "} .else { .echo C3B1_ALREADY_HIT; gc }"
```

Mapping the caret into the old body places it at the boundary between the `T1` `.else` keyword and
the `{` that opens its block:

```text
body offsets   333..351  .if (@$t1 == @$t12)
               361..373  XCHK_T1_AGREE
               376..389  r $t18=@$t18+1
               391       }                      <- closes the T1 .if
               393..397  .else
               398       <-- caret
               400       {                      <- would open the .else block
```

The old body contains **seven** `} .else {` transitions, not one:

```text
T1..T5 agreement/disagreement pairs   = 5
sentinel cleared/retained pair        = 1
outer already-hit pair                = 1
                                        ---
                                        7
```

### 4.1 Correction to the Step5BY classification

Step5BY section 6, row D states that `.else { ... }` is "NOT PROCESSED - lies entirely inside the
omitted 39-char tail". That holds for the **outermost** `.else` only. The other six `.else`
constructs lie inside the region the debugger processed, and the caret lands in the T1 one. This
gate records the corrected localization:

```text
PROVEN     the old run executed T1's .if body, emitted XCHK_T1_AGREE, incremented $t18,
           and stopped at the } .else { transition that followed it
NOT PROVEN the internal CDB mechanism that rejects the transition
```

The mechanism is not asserted. A plausible reading is that after a completed `.if (...) { ... }`
the parser expects a new command, and `.else` is a continuation keyword rather than a standalone
one, so it is rejected as an extra character. This reading is consistent with the caret position
but is not established, and no claim is made on it.

Step5BY's off-by-one in the tail length (39 vs the measured 38) is immaterial; the tail content it
identified is correct.

### 4.2 Why the correction is expected to hold

The corrected body substitutes the failing transition class outright. The old body's seven
`} .else {` transitions become zero `.else` constructs and four `} .if (` transitions:

```text
old body       7 x "} .else {"     0 x "} .if ("
new body       0 x "} .else {"     4 x "} .if ("
```

`}` followed by `.if` is the ordinary command-boundary form, whereas `}` followed by `.else` is the
form observed to fail on this CDB build. The construct at which the previous run stopped cannot
recur, because it is absent from the body in every instance. This is the reason the authorization
is judged worth spending, and it is the reason the outcome must be classified in a separate Result
Audit rather than assumed here.

## 5. Structural repair re-confirmed at this gate

Independently re-checked rather than inherited:

```text
R1  no q in breakpoint body                    PASS
R2  no gc in breakpoint body                   PASS
R3  no .else in breakpoint body                PASS
R4  no outer .if wrapper in body               PASS
R5  braces balanced (6 / 6)                    PASS
R6  no resume command in body                  PASS
R7  no .else anywhere in the script            PASS
R8  no "] .if" or other stray character        PASS
```

```text
corrected body length = 696 chars
corrected body .else  = 0
corrected body gc     = 0
corrected body q      = 0
```

The three constructs that produced the previous failure are absent from the breakpoint body.

The script-level `g` and `q` remain outside the breakpoint body, as designed, and are permitted.

## 6. Authorized scope

```text
Target    = C:\Windows\System32\ping.exe
Arguments = 127.0.0.1 -n 32
CDB       = tmp\cdb.exe (frozen binary and version)
Script    = corrected B1 probe (frozen SHA)
Log       = doc/work113/P1-5-IR-P2_Step5CB_P3-5-FPM-C3-T1-Capture-PostRetry2-Benign_Probe_Candidate-B1_Corrected_Probe_Execution-1_CDB.log

CDB execution = exactly 1
ping execution = exactly 1
```

The authorization is consumed at the moment CDB is launched, regardless of outcome.

The log name is corrected here. The prior draft of this document reserved a path that dropped the
`Corrected_Probe` segment and so did not match the established naming of the Execution-1 log. Since
the log is the sole evidence artifact of a single non-repeatable execution, its path is fixed
unambiguously.

## 7. Expected runtime sequence

Taken from the frozen script, in script order:

```text
C3B1C_SCRIPT_BEGIN
C3B1C_SENTINEL_SET
C3B1C_ASSIGN_DONE
C3B1C_READBACK_DONE
C3B1C_BP_SET
  -- g, then at the hit --
C3B1C_BREAKPOINT_HIT
C3B1C_B1_DERIVED
r $t11, r $t1..$t5                     <- readback group 1, before the comparisons
C3B1C_CROSSCHECK_BEGIN
XCHK_T1_AGREE
XCHK_T2_AGREE
XCHK_T3_AGREE
XCHK_T4_AGREE
XCHK_T5_AGREE
C3B1C_CROSSCHECK_END
C3B1C_SENTINEL_CLEARED
C3B1C_TALLY
r $t16, r $t17, r $t18, r $t19         <- readback group 2, after the tally
  -- body ends with no resume command, so the debugger stays stopped --
C3B1C_SCRIPT_END
quit
```

Two corrections to the sequence as stated in the instruction for this gate:

```text
C3B1C_SENTINEL_CLEARED  is emitted between CROSSCHECK_END and TALLY, and was omitted
r $t11 / $t1..$t5        readback occurs BEFORE CROSSCHECK_BEGIN, not after TALLY
```

All seventeen markers are present in the frozen script and are therefore expected to be observable
at runtime.

The body contains no resume command, so the breakpoint stops rather than continues. The script's
`g` returns at the stop, `C3B1C_SCRIPT_END` is emitted, and `q` quits. This control flow is not an
assumption: the previous Execution-1 log shows `C3B1C_SCRIPT_END` and `quit:` at lines 104 and 105
after its breakpoint error, so the script-level tail does resume following a stopped body.

## 8. PASS conditions

```text
B1 = PROVEN  requires all of:
    C3B1C_BREAKPOINT_HIT  = 1
    C3B1C_B1_DERIVED       = 1
    XCHK_T1_AGREE          = 1
    XCHK_T2_AGREE          = 1
    XCHK_T3_AGREE          = 1
    XCHK_T4_AGREE          = 1
    XCHK_T5_AGREE          = 1
    C3B1C_CROSSCHECK_END   = 1
    C3B1C_TALLY            = 1
    $t18                   = 6   (see section 9)
    process residue        = 0

B1 = FAILED   the breakpoint is reached, the crosscheck completes, and one or more
              of the five comparisons does not agree. Recorded as the absence of the
              corresponding AGREE marker, since the body has no DISAGREE branch.

B1 = INCONCLUSIVE  any of:
              breakpoint not reached
              C3B1C_CROSSCHECK_END not reached
              C3B1C_TALLY not reached
              a CDB parser or runtime error occurs
```

`B1 = FAILED` never demotes Candidate A, which is already proven independently. Any outcome leaves
Candidate A = PROVEN.

Note that the body has no DISAGREE branch, so a disagreement is observable only as a missing AGREE
marker. The Result Audit must therefore confirm the crosscheck ran to completion before it may
call a disagreement, and must not read a missing marker as a mismatch while `CROSSCHECK_END` is
itself absent.

## 9. Correction to the stated `$t18` expectation

The instruction for this gate stated `t18 = 5`. The frozen script yields **6** when every agreement
occurs and the sentinel clears:

```text
XCHK_T1_AGREE .. XCHK_T5_AGREE   5 increments
C3B1C_SENTINEL_CLEARED            1 increment
                                   ---
total $t18                        6
```

The increment sites were counted directly in the frozen script:

```text
r $t18=@$t18+1 occurrences = 6
```

The sentinel check `.if (@$t1 != 0xDEAD0001)` must fire, because `$t1` was assigned `ping+0x3c`
during Candidate A setup and therefore cannot still hold the sentinel. The increment is real and is
not a duplicate of a crosscheck. The correct PASS condition is therefore:

```text
$t18 = 6 with all five AGREE markers and C3B1C_SENTINEL_CLEARED present
```

A `$t18` of 5 with all five agreements present would indicate the sentinel branch did not execute.
The Result Audit must read `$t18` together with the marker set rather than as a standalone number.

## 10. Expression semantics confirmed

The crosscheck depends on MASM-mode comparison of user pseudo-registers. Both were confirmed
against vendor documentation at this gate rather than assumed:

```text
MASM evaluator comparison operators include  ==  and  !=     confirmed
user-defined pseudo-registers @$t0 .. @$t19 are supported      confirmed
script sets .expr /s masm                                     confirmed in script
```

## 11. Forbidden under this authorization

```text
rerun                    FORBIDDEN
second execution         FORBIDDEN
AudioEngineHarness       FORBIDDEN
production C3            FORBIDDEN
T1 capture               FORBIDDEN
ReaderSlot access        FORBIDDEN
EpochDomain access       FORBIDDEN
getMinReaderEpoch        FORBIDDEN
DQueue / reclaim         FORBIDDEN
M1 / M2                  FORBIDDEN
Retry-3                  FORBIDDEN
Preparation-4            FORBIDDEN
build                    FORBIDDEN
Dr.Memory                FORBIDDEN
production / test / CMake modification  FORBIDDEN
source modification      FORBIDDEN
```

Even if B1 is proven, this authorization does not permit proceeding toward the production C3
capture. B1 is a validation of a CDB anchor-address derivation primitive only.

## 12. Post-execution sequence

```text
Corrected-B1 Static Validation-1     CLOSED / PASS
        |
        v
Corrected-B1 Runtime Authorization-1 CLOSED  (this gate)
        |
        v
Corrected-B1 Benign Execution-1      NOT STARTED  <- next
        |
        v
Result Audit-1                      NOT STARTED
        |
        +-- B1 PROVEN      -> next probe in the expression-family sequence
        |
        +-- B1 FAILED      -> record; Candidate A unchanged; decide next candidate
        |
        +-- B1 INCONCLUSIVE-> failure audit; no verdict on B1
```

The execution result must be classified in a separate Result Audit gate, not on the spot.

## 13. What remains unchanged

```text
DIRECT_DQUEUE_CAUSE = epoch gate equality PROVEN
Candidate A         = PROVEN
@$bp0 primitive     = PROVEN
$t11 derivation     = PROVEN
B1 full runtime     = INCONCLUSIVE

S7_READER          = UNRESOLVED
S7_READER_SLOT     = UNRESOLVED
minReaderEpoch     = NOT CAPTURED
T1                 = NOT CAPTURED
Case A/B/C/D       = NOT PROVEN
IMPLEMENTATION     = FORBIDDEN
```

The reclaim condition remains `retireEpoch < minReaderEpoch`, with reader and epoch observation kept
separate from reclaim. The RT-side observation boundary and the NonRT lifetime isolation line are
unchanged. No ConvoPeq reader, epoch, or reclaim evidence was produced or altered.

## 14. Final state

```text
P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Candidate-B1-Corrected-Probe-Runtime-Authorization-1
= CLOSED / ONE INVOCATION AUTHORIZED / NOT EXECUTED IN THIS GATE

inputs verified        = 8 / 8 PASS
structural repair      = 8 / 8 PASS (R1..R8)
fault re-localized     = caret at the T1 "} .else {" transition, body offset 398
Step5BY row D          = corrected, see section 4.1
expected markers       = 17 / 17 present in script
script modified        = NO
new .cdb               = 0
CDB execution          = 0
ping execution         = 0

maximum invocations    = 1 (consumed on launch)
B1 runtime status      = INCONCLUSIVE (unchanged)
Candidate A            = PROVEN (unaffected by any outcome)
$t18 PASS condition    = 6, corrected from the stated 5
sequence corrections   = 2, see section 7
log path               = corrected, see section 6

Preparation-4 / Retry-3 / production C3 / AudioEngineHarness / T1 capture / ReaderSlot / epoch / reclaim / M1 / M2 / build / Dr.Memory = FORBIDDEN
rerun = FORBIDDEN
implementation = FORBIDDEN
```
