# P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Candidate-B-Replacement-Static-Validation-1

## 1. Gate result

```text
Gate                    = P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Candidate-B-Replacement-Static-Validation-1
Mode                    = script creation + static validation only
new .cdb created        = 1
CDB execution           = 0
ping execution          = 0
AudioEngineHarness      = 0
Retry-3                 = 0
production C3 capture   = 0
Preparation-4           = 0 (still BLOCKED)
M1/M2/build/Dr.Memory   = 0
ConvoPeq source change  = 0
test/CMake change       = 0
rerun                   = 0
runtime authorization   = NOT GRANTED (separate gate)
VERDICT                 = STATIC VALIDATION PASS / B1 STRUCTURALLY VALID / NOT RUNTIME-PROVEN
```

## 2. Source authority and baseline

```text
ConvoPeq.md  SHA-256 = E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
             bytes   = 5,535,334
```

`ConvoPeq(3).md` was not used as an alternative source. No new source-level claim is made in this document.

Pre-existing command files, all unchanged at this gate:

| file | SHA-256 |
|---|---|
| C3-RP retry (not executed) | `AFCC0017DA95D36F23D497123547AEE0B0B8D7572F43231835A25AE532EF0F18` |
| C3-T1 Capture Redesign-2 | `FBE32851F795AF900E6C46D6E38F25138E01217C2B52C0BDDE1F015029CFC39E` |
| consumed benign authorization | `5C12E1515CC2206430F34ADC7B8FE39D84A04BF4B33F2D7628970C395EC7699B` |
| Preparation-3 anchor probe (consumed) | `671F9D453FF814DD2AE01217A6CAE7F69E0D033A7A0AA70BD42D21E797D63771` |

```text
process residue = 0
```

## 3. Created artifact

```text
Path    = doc/work113/P1-5-IR-P2_Step5BU_P3-5-FPM-C3-T1-Capture-PostRetry2-Benign_Probe_Candidate-B1_Anchor_Probe.cdb
SHA-256 = 403C8E2BBDAACC5C8010A664A8188C87A7167D28209F203D79CAA418BBEB8329
bytes   = 1,481
lines   = 28
```

Exactly one new command file was created.

## 4. Structure

```text
.echo C3B1_SCRIPT_BEGIN
.sympath+ C:\VSC_Project\ConvoPeq
.expr /s masm
r $t1 = 0xDEAD0001 ... r $t5 = 0xDEAD0005
r $t16 = 0 / $t17 = 0 / $t18 = 0 / $t19 = 0
.echo C3B1_SENTINEL_SET
r $t1 = ping+0x3c
r $t2 = ping+0x4FD0
r $t3 = ping+0x2600
r $t4 = ping+0x5000
r $t5 = ping+0x40
.echo C3B1_ASSIGN_DONE
r $t1, $t2, $t3, $t4, $t5
.echo C3B1_READBACK_DONE
bp ping+0x39d9 "<B1 body>"
.echo C3B1_BP_SET
bl
g
.echo C3B1_SCRIPT_END
q
```

Breakpoint body:

```text
.if (@$t16 == 0) {
  r $t16=1; r $t17=@$t17+1;
  .echo C3B1_BREAKPOINT_HIT
  r $t11=@$bp0-0x39d9
  r $t7=@$t11+0x1c0
  r $t12=@$t11+0x3c
  r $t13=@$t11+0x4FD0
  r $t14=@$t11+0x2600
  r $t15=@$t11+0x5000
  r $t6=@$t11+0x40
  .echo C3B1_CANDIDATE_B1_DERIVED
  r $t11; r $t1; r $t2; r $t3; r $t4; r $t5
  .echo C3B1_CROSSCHECK_BEGIN
  XCHK_T1..T5 : AGREE increments $t18, DISAGREE echoes only
  .echo C3B1_CROSSCHECK_END
  sentinel cleared -> echo + $t18++
  .echo C3B1_COUNTS
  r $t16; r $t17; r $t18; r $t19
  q
} .else { .echo C3B1_ALREADY_HIT; gc }
```

## 5. Static validation results

| # | check | result |
|---|---|---|
| S1 | exactly one new `.cdb`, total 5 in tree | PASS |
| S2 | `@$pc` completely absent | PASS |
| S3 | no `;` comment lines | PASS |
| S4 | `@$bp0` appears exactly once | PASS |
| S5 | breakpoint ID 0 consistency | PASS |
| S6 | exactly one breakpoint, at `ping+0x39d9` | PASS |
| S7 | Candidate A `ping+offset` anchors retained (5/5) | PASS |
| S8 | B1 arithmetic contains no memory dereference | PASS |
| S9 | `dwo` count = 0 | PASS |
| S10 | `poi` count = 0 | PASS |
| S11 | `dd` / `dq` count = 0 | PASS |
| S12 | ReaderSlot access = 0 | PASS |
| S13 | epoch/reclaim capture = 0 | PASS |
| S14 | Harness reference = 0 | PASS |
| S15 | clean `q` termination | PASS |

Structural extras:

```text
X1  no >> or & operator in breakpoint body          PASS
X2  no MASM and operator                            PASS
X3  B1 derivation present: r $t11=@$bp0-0x39d9      PASS
X4  B1 anchors present 5/5                          PASS
X5  five non-fatal DISAGREE branches                PASS
X6  no FAIL-on-disagree, no Candidate-A fail path   PASS
X7  .echo phase markers present                     PASS
X8  unconditional readback present                  PASS
X9  MASM evaluator selected                         PASS
X10 hit-once re-entry guard defined                 PASS

FAILURES = 0
```

## 6. Validator artifacts corrected

Two initial checks reported failures that were defects in the checker, not the script.

```text
S3 false positive
    cause = every semicolon in the file lies inside the bp quoted command string,
            where ';' is the documented CDB command separator and is required.
            The file contains 0 lines beginning with ';' and 0 star-prefixed lines.
    fix   = the check was redefined as "no line begins with ; and no semicolon
            appears outside the bp command string"

S10 false positive
    cause = the case-insensitive substring "poi" matched inside the marker text
            C3B1_BREAKPOINT_HIT.
    fix   = the check was redefined as a whole-word token match plus a call-form
            match: bare-word poi = 0, poi( = 0
```

Both were re-run under corrected definitions and pass. The script was not modified in response to either.

## 7. S5 rationale

`$bp0` refers to breakpoint ID 0. In this file exactly one breakpoint is created, and CDB assigns IDs sequentially from 0, as documented for `bl` output. With a single breakpoint the ID is necessarily 0, so `@$bp0` unambiguously denotes it. This is a structural argument, not a runtime confirmation; the actual ID is verified in the next gate's log via `bl`.

## 8. Success criteria for this static gate

```text
Script structurally valid                     PASS
B1 candidate present                          PASS
Candidate A retained                          PASS
no comments                                   PASS
no memory dereference                         PASS
no production / Harness logic                 PASS
```

Static validation PASS means the script is well-formed and structurally faithful to the design. It does **not** mean `@$bp0` is runtime-usable.

## 9. Runtime interpretation contract

If executed under a separate authorization, the following dispositions are pre-agreed:

```text
B1 works and agrees with Candidate A  -> B1 PROVEN; anchor construction doubly confirmed
B1 works but disagrees                 -> B1 FAILED / DISAGREE; Candidate A remains PROVEN
B1 rejected (Bad register or zero)    -> B1 FAILED; Candidate A remains PROVEN
no breakpoint hit                     -> inconclusive; CANDIDATE_A_INCONCLUSIVE
```

In no outcome does an A/B disagreement demote Candidate A. The DISAGREE branches echo and continue by design; there is no FAIL path for Candidate A anywhere in the file.

## 10. What remains unchanged

```text
DIRECT_DQUEUE_CAUSE = epoch gate equality PROVEN
S7_READER          = UNRESOLVED
S7_READER_SLOT     = UNRESOLVED
minReaderEpoch     = NOT_CAPTURED
T1 capture         = NOT CAPTURED
Case A/B/C/D       = NOT PROVEN
IMPLEMENTATION     = FORBIDDEN
```

The RT-side observation boundary and the NonRT lifetime isolation line are unchanged. No ConvoPeq reader, epoch, or reclaim evidence was produced or altered.

## 11. Final state

```text
P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Candidate-B-Replacement-Static-Validation-1
= CLOSED / STATIC VALIDATION PASS / NOT EXECUTED

new script   = 403C8E2B...BB8329 (1,481 bytes, 28 lines)
S1..S15      = 15 / 15 PASS
extras X1-X10= 10 / 10 PASS
FAILURES     = 0
comments     = 0
deref        = 0
B1 runtime   = NOT PROVEN (structural validity only)

CDB execution = 0
ping execution = 0
runtime authorization = NOT GRANTED
Preparation-4 = BLOCKED / NOT STARTED
Retry-3 / production C3 / AudioEngineHarness / M1 / M2 / build / Dr.Memory = FORBIDDEN
rerun = 0 / implementation = FORBIDDEN
```

## 12. Next gate

```text
Candidate-B Replacement Preparation-1        CLOSED
        |
        v
Candidate-B Replacement Static Validation-1  CLOSED  (this gate)
        |
        v
Runtime Authorization-1                     NOT STARTED  <- next
        |
        v
Benign Execution-1                          NOT STARTED
        |
        v
Result Audit-1                              NOT STARTED
```
