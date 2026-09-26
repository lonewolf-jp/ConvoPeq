# P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Candidate-B1-Corrected-Probe-Design-Preparation-1

## 1. Gate result

```text
Gate                        = P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Candidate-B1-Corrected-Probe-Design-Preparation-1
Mode                        = read-only design / preparation
CDB execution               = 0
ping execution              = 0
rerun                       = 0
new .cdb created            = 1
static validation performed = PRE-CHECK ONLY (formal gate is separate)
runtime authorization       = NOT GRANTED
production C3               = 0
T1 capture                  = 0
ConvoPeq source modification= 0
B1                          = INCONCLUSIVE (unchanged)
Candidate A                 = PROVEN (unchanged)
```

This gate designs and creates the corrected probe. It executes nothing and grants no authorization.

## 2. Established facts carried forward, not re-verified

| item | state |
|---|---|
| Candidate A `ping+offset` | **PROVEN** |
| `@$bp0` | **PROVEN** |
| `$t11 = @$bp0 - 0x39d9` | **PROVEN** |
| A/B image-base agreement | **PROVEN** |
| T1 comparison | **PROVEN** |
| B1 full runtime | **INCONCLUSIVE** |
| `@$pc` | FAILED |
| ReaderSlot | NOT CAPTURED |
| `minReaderEpoch` | NOT CAPTURED |
| T1 reclaim capture | NOT CAPTURED |

## 3. Source authority and baseline

```text
ConvoPeq.md SHA-256 = E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
ConvoPeq.md bytes   = 5,535,334
process residue     = 0
```

`ConvoPeq(3).md` was not used as a substitute for the current source. No source-level claim is made in this document beyond the identity above. The architecture principle relied on is the separation of reader/epoch observation from reclaim, with the reclaim condition remaining `retireEpoch < minReaderEpoch`.

Five pre-existing `.cdb` files were verified unchanged before this gate created the sixth.

## 4. Correction applied

The fault origin isolated by the Failure Audit was the breakpoint command body's structure:

```text
previous body: .if (@$t16 == 0) { ... ; q } .else { .echo C3B1_ALREADY_HIT; gc }
fault region  : the trailing " } .else { .echo C3B1_ALREADY_HIT; gc }" (39 chars)
```

Two changes were made, and only these two:

```text
1. the resume command q was removed from the breakpoint command body
2. the outer .else branch and its gc were removed together with the
   already-hit guard that required them
```

The already-hit guard existed only to support a re-entry path. Removing it shrinks the probe to a single-hit structure and eliminates the entire trailing construct that the failure was attributed to.

Nothing else changed. `@$bp0`, `$t11`, the five B1 anchors, the five comparisons, and Candidate A are byte-identical in intent to the proven forms.

## 5. Created artifact

```text
Path    = doc/work113/P1-5-IR-P2_Step5BZ_P3-5-FPM-C3-T1-Capture-PostRetry2-Benign_Probe_Candidate-B1_Corrected_Probe.cdb
SHA-256 = FFC5228C2387BDC5264F4DB956D66E6DFFB46A97F97ADE2E59E0A7BA1AE739DE
bytes   = 1,172
lines   = 28
breakpoint body = 696 chars  (previous failed probe: 1011)
```

## 6. Script-level flow

```text
.echo C3B1C_SCRIPT_BEGIN
.sympath+ C:\VSC_Project\ConvoPeq
.expr /s masm
sentinel  $t1..$t5 = 0xDEAD0001..0xDEAD0005
counters  $t16..$t19 = 0
.echo C3B1C_SENTINEL_SET
Candidate A: r $t1 = ping+0x3c / 0x4FD0 / 0x2600 / 0x5000 / 0x40
.echo C3B1C_ASSIGN_DONE
unconditional readback: r $t1, $t2, $t3, $t4, $t5
.echo C3B1C_READBACK_DONE
bp ping+0x39d9 "<body>"
.echo C3B1C_BP_SET
bl
g
.echo C3B1C_SCRIPT_END
q
```

Termination is handled by the script-level `q` after `g`, outside the breakpoint body.

## 7. Breakpoint body, as 21 top-level command units

```text
U01 .echo C3B1C_BREAKPOINT_HIT
U02 r $t11=@$bp0-0x39d9
U03 r $t12=@$t11+0x3c
U04 r $t13=@$t11+0x4FD0
U05 r $t14=@$t11+0x2600
U06 r $t15=@$t11+0x5000
U07 r $t6=@$t11+0x40
U08 .echo C3B1C_B1_DERIVED
U09 r $t11
U10 r $t1
U11 r $t2
U12 r $t3
U13 r $t4
U14 r $t5
U15 .echo C3B1C_CROSSCHECK_BEGIN
U16 .if (@$t1 == @$t12) { .echo XCHK_T1_AGREE; r $t18=@$t18+1 }
    .if (@$t2 == @$t13) { .echo XCHK_T2_AGREE; r $t18=@$t18+1 }
    .if (@$t3 == @$t14) { .echo XCHK_T3_AGREE; r $t18=@$t18+1 }
    .if (@$t4 == @$t15) { .echo XCHK_T4_AGREE; r $t18=@$t18+1 }
    .if (@$t5 == @$t6)  { .echo XCHK_T5_AGREE; r $t18=@$t18+1 }
U17 .if (@$t1 != 0xDEAD0001) { .echo C3B1C_SENTINEL_CLEARED; r $t18=@$t18+1 }
    .echo C3B1C_CROSSCHECK_END
    .echo C3B1C_TALLY
U18 r $t16
U19 r $t17
U20 r $t18
U21 r $t19
```

Each crosscheck is a self-contained `.if { ... }` with no `.else`. Agreement increments `$t18`; a disagreement simply produces no marker, which the audit reads as absence.

## 8. Structural integrity pre-check

```text
braces in body            = 6 opening / 6 closing, balanced
.if constructs            = 6  (5 crosschecks + 1 sentinel)
each .if self-contained   = yes, closes before the next command
outer .if wrapper         = none
.else branches            = 0
resume commands in body   = 0 (no q, no gc)
top-level command units   = 21
```

One pre-check assertion was miscounted on first run: it expected five `.if` constructs and five closing braces, while the body correctly contains six of each (the sixth is the sentinel check). The script was not changed in response. Re-inspection confirmed the body is correct.

## 9. Design constraint pre-check

| # | constraint | pre-check |
|---|---|---|
| S1 | exact source identity | PASS (`ConvoPeq.md` = `E5E74200...EF3609`) |
| S2 | no production source modification | PASS |
| S3 | no `@$pc` | PASS |
| S4 | exactly one `@$bp0` | PASS |
| S5 | `@$bp0 - 0x39d9` preserved | PASS, verbatim |
| S6 | Candidate A unchanged | PASS, 5/5 |
| S7 | five B1-derived anchors | PASS, 5/5 |
| S8 | T1..T5 comparisons present | PASS, 5/5 |
| S9 | `CROSSCHECK_END` present | PASS |
| S10 | final tally present | PASS |
| S11 | **no `q` inside breakpoint body** | PASS |
| S12 | **no `.else` in breakpoint body** | PASS |
| S13 | no comment-line syntax | PASS |
| S14 | no `dwo` / `poi` / `dd` / `dq` | PASS |
| S15 | no ReaderSlot / epoch / reclaim access | PASS |

This is a pre-check, not the formal static validation. S11 and S12 are the core of the correction and both hold.

## 10. Deliberately excluded

```text
dwo, poi, dd, dq
ReaderSlot, EpochDomain, getMinReaderEpoch, DQueue, reclaim
AudioEngineHarness
T1 reclaim capture
```

The sole purpose is to demonstrate that the `@$bp0` B1 image-base derivation plus the T1..T5 five-point crosscheck can run to completion.

## 11. Expected runtime outcome, stated in advance

```text
all five XCHK_T*_AGREE and C3B1C_CROSSCHECK_END and C3B1C_TALLY appear
   -> B1 = PROVEN

crosscheck completes with some DISAGREE absent
   -> B1 = FAILED (mismatch), Candidate A unchanged

breakpoint does not fire
   -> B1 = INCONCLUSIVE
```

A disagreement is recorded by the absence of an AGREE marker. No DISAGREE branch exists, so the body never aborts on a mismatch and the run always reaches `CROSSCHECK_END`. This removes the fail-fast behavior that the previous probe used, and it is the intended trade: completion of the comparison set is now observable in every outcome.

## 12. Sequence

```text
Candidate-B1 Corrected-Probe Design-Preparation-1   CLOSED  (this gate)
        |
        v
Corrected-B1 Static Validation-1                    NOT STARTED  <- next
        |
        v
Corrected-B1 Runtime Authorization-1                 NOT STARTED
        |
        v
one benign execution                                 NOT STARTED
        |
        v
Result Audit-1                                      NOT STARTED
```

No CDB or ping execution before the static validation closes. No runtime authorization is issued by this gate.

## 13. What remains unchanged

```text
DIRECT_DQUEUE_CAUSE = epoch gate equality PROVEN
Candidate A         = PROVEN
@$bp0 primitive     = PROVEN
$t11 base derivation= PROVEN
B1 full runtime     = INCONCLUSIVE

S7_READER          = UNRESOLVED
S7_READER_SLOT     = UNRESOLVED
minReaderEpoch     = NOT CAPTURED
T1                 = NOT CAPTURED
Case A/B/C/D       = NOT PROVEN
IMPLEMENTATION     = FORBIDDEN
```

The RT-side observation boundary and the NonRT lifetime isolation line are unchanged. No ConvoPeq reader, epoch, or reclaim evidence was produced or altered.

## 14. Final state

```text
P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Candidate-B1-Corrected-Probe-Design-Preparation-1
= CLOSED / CORRECTED PROBE CREATED / NOT EXECUTED

new script  = FFC5228C...AE739DE (1,172 bytes, 28 lines)
body        = 696 chars, 21 top-level units, braces 6/6 balanced
correction  = q removed from breakpoint body
              outer .else and its gc removed with the already-hit guard
S1..S15 pre-check  = 15 / 15
retained unchanged = @$bp0, $t11, five B1 anchors, five comparisons, Candidate A

CDB execution / ping execution / rerun / production C3 / T1 capture = 0
runtime authorization = NOT GRANTED
Preparation-4 / Retry-3 / AudioEngineHarness / M1 / M2 / build / Dr.Memory = FORBIDDEN
```
