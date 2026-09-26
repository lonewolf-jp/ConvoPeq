# P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Candidate-B1-Semicolon-Separator-Probe-Runtime-Authorization-1

## 1. Gate result

```text
Gate                    = P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Candidate-B1-Semicolon-Separator-Probe-Runtime-Authorization-1
Mode                    = authorization review only
CDB execution           = 0
ping execution          = 0
rerun                   = 0
script modification     = 0
new .cdb created        = 0
reserved log created    = 0
VERDICT                 = ONE INVOCATION AUTHORIZED
                          (CONDITIONAL ON PRE-EXECUTION IDENTITY RECHECK)
```

This gate authorizes a single future benign execution. It performs none itself and creates no log.

## 2. Source authority

```text
ConvoPeq.md SHA-256 = E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
ConvoPeq.md bytes   = 5,535,334
```

`ConvoPeq.md` is the only production source authority. The superseded `ConvoPeq(3).md` was not used
as a substitute, and no other authority was consulted in its place. The working tree contains exactly
one `ConvoPeq*.md` file, so no substitution was available even in principle:

```text
E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609  5,535,334 bytes  ConvoPeq.md
```

This probe does not reach ConvoPeq reader, epoch, or reclaim state, so no source-level change arises
and none is made or requested.

## 3. Identity recheck at this gate

All values were recomputed here rather than inherited from Static Validation.

| # | item | fixed value | observed | result |
|---|---|---|---|---|
| A1 | Semicolon Separator Probe SHA-256 | `EA87DE2E…0E2F55` | identical | PASS |
| A2 | Semicolon Separator Probe bytes | `1,184` | `1,184` | PASS |
| A3 | `cdb.exe` SHA-256 | `5F54ABAF…FBEE67` | identical | PASS |
| A4 | `cdb.exe` bytes | `178,016` | identical | PASS |
| A5 | CDB version | `10.0.29617.1000` | `10.0.29617.1000` | PASS |
| A6 | `ConvoPeq.md` SHA-256 | `E5E74200…F3609` | identical | PASS |
| A7 | `ConvoPeq.md` bytes | `5,535,334` | identical | PASS |
| A8 | `ping.exe` SHA-256 | `E4224D18…4DD468` | identical | PASS |
| A9 | process residue | `0` | `0` | PASS |
| A10 | reserved execution log | absent | absent | PASS |

```text
identity drift = 0
```

The version was compared as a number, not as a raw string. `FileVersion` on this binary reports
`10.0.29617.1000 (WinBuild.160101.0800)`, and a naive whole-string comparison against the bare
version number reports a spurious mismatch. The numeric part is exact, and
`FileMajorPart/FileMinorPart/FileBuildPart/FilePrivatePart` reads `10.0.29617.1000`.

Any single drift at the pre-execution recheck is a STOP. No value may be refreshed in place, and this
authorization may not be reused.

## 4. Authorized artifact

```text
Path    = doc/work113/P1-5-IR-P2_Step5CE_P3-5-FPM-C3-T1-Capture-PostRetry2-Benign_Probe_Candidate-B1_Semicolon_Separator_Probe.cdb
SHA-256 = EA87DE2E419BC73941FB220AF5BA642DB2009E52C5AEF7D07BF45DFC620E2F55
bytes   = 1,184
body    = 708 chars
```

Separator form carried by the authorized artifact, confirmed at this gate:

```text
adopted    "} ; .if"    x4
adopted    "} ; .echo"  x2
untested  "}; .if"     x0
untested  "}; .echo"   x0
```

`} ; .if` is the only candidate under this authorization. The `}; .if` spelling is unverified and
belongs to a possible future candidate. It must not be substituted here, including after a failure.

## 5. Authorized scope

```text
Target    = C:\Windows\System32\ping.exe
Arguments = 127.0.0.1 -n 32
CDB       = tmp\cdb.exe (frozen binary and version)
Script    = Semicolon Separator Probe (frozen SHA, section 4)
Log       = doc/work113/P1-5-IR-P2_Step5CE_P3-5-FPM-C3-T1-Capture-PostRetry2-Benign_Probe_Candidate-B1_Semicolon_Separator_Probe_Execution-1_CDB.log

CDB execution = exactly 1
ping execution = exactly 1
```

The log path is fixed here and reserved. **The log is not created at this gate.** It must not exist
before the authorized invocation, and its absence is confirmed as A10.

The authorization is consumed at the moment CDB is launched, regardless of outcome.

## 6. What this execution tests

One question only:

```text
Does this CDB build accept "} ; .if" and "} ; .echo" as command boundaries
inside a breakpoint command body?
```

The expected arrival order below is an **expectation, not a result**. Nothing in this document
asserts that any marker will be emitted.

```text
BREAKPOINT_HIT
B1_DERIVED
CROSSCHECK_BEGIN
XCHK_T1_AGREE
XCHK_T2_AGREE
XCHK_T3_AGREE
XCHK_T4_AGREE
XCHK_T5_AGREE
CROSSCHECK_END
C3B1C_SENTINEL_CLEARED
C3B1C_TALLY
final counter readback  ( r $t16; r $t17; r $t18; r $t19 )
C3B1C_SCRIPT_END
quit
```

The complete-success value of the tally counter is fixed:

```text
5 comparison increments  (T1..T5)
1 sentinel increment
                        ---
$t18                    = 6
```

Crosscheck completion may be judged only when all of the following hold together:

```text
XCHK_T1..T5_AGREE = 5 present
SENTINEL_CLEARED   = 1 present
$t18               = 6
```

Marker counting must distinguish emitted output from echoed command text. The `bp` line, the `bl`
listing and any error message all repeat every marker string, so a substring count over the log is
not a valid observation. Only lines whose entire trimmed content is the marker count as emitted.

## 7. Classification reserved for the Result Audit

No verdict is rendered at this gate and none may be inferred from this document.

```text
B1 = PROVEN        BREAKPOINT_HIT, B1_DERIVED, T1..T5_AGREE, CROSSCHECK_END,
                   SENTINEL_CLEARED, TALLY all reached, and $t18 = 6, residue = 0

B1 = FAILED        BREAKPOINT_HIT, B1_DERIVED and CROSSCHECK_END all reached,
                   and one or more comparisons is explicitly shown not to agree

B1 = INCONCLUSIVE  breakpoint not reached, or CROSSCHECK_END not reached, or
                   TALLY not reached, or a CDB parser or runtime error occurs
```

The rule from the previous audit is retained: a `T2..T5_AGREE` marker missing **before**
`CROSSCHECK_END` is reached is **not** a disagreement. The body has no DISAGREE branch, so a
disagreement is observable only as a missing marker after the crosscheck is known to have run to
completion. `B1 = INCONCLUSIVE` never demotes Candidate A, which is independently proven.

## 8. Forbidden under this authorization

```text
rerun                    FORBIDDEN
second execution         FORBIDDEN
substituting "}; .if"    FORBIDDEN
AudioEngineHarness       FORBIDDEN
production C3            FORBIDDEN
T1 capture               FORBIDDEN
ReaderSlot               FORBIDDEN
EpochDomain              FORBIDDEN
getMinReaderEpoch        FORBIDDEN
minReaderEpoch           FORBIDDEN
DQueue / reclaim         FORBIDDEN
M1 / M2                  FORBIDDEN
Retry-3                  FORBIDDEN
Preparation-4            FORBIDDEN
build                    FORBIDDEN
Dr.Memory                FORBIDDEN
production source modification  FORBIDDEN
test modification              FORBIDDEN
CMake modification             FORBIDDEN
```

Even if B1 is proven, this authorization does not permit proceeding toward the production C3
capture. It validates a CDB command-body construct only.

## 9. Not performed in this gate

```text
CDB execution / ping execution / rerun / script modification / new .cdb / reserved log = 0
process residue = 0
execution logs on disk = 4, unchanged; the reserved log is not among them
.ccdb inventory = 7, unchanged
frozen corrected probe (BZ) = FFC5228C2387BDC5264F4DB956D66E6DFFB46A97F97ADE2E59E0A7BA1AE739DE, unchanged
```

## 10. Stop point

```text
Gate 1  Preparation              CLOSED
Gate 2  Static Validation        CLOSED / PASS   (V01..V46)
Gate 3  Runtime Authorization    CLOSED          (this gate)
        ---- STOP ----
Gate 4  Benign Execution-1       NOT STARTED
        ---- STOP ----
Gate 5  Result Audit-1           NOT STARTED
```

This gate stops after issuing the authorization. It confers no evidence of runtime acceptance; that
is established only by a Benign Execution and its separate Result Audit, and only then.

## 11. Base state, unchanged

```text
DIRECT_DQUEUE_CAUSE = epoch gate equality PROVEN
Candidate A         = PROVEN
@$bp0 primitive     = PROVEN
$t11 derivation     = PROVEN
T1 agreement        = PROVEN
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

## 12. Final state

```text
P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Candidate-B1-Semicolon-Separator-Probe-Runtime-Authorization-1
= CLOSED / ONE INVOCATION AUTHORIZED / NOT EXECUTED IN THIS GATE

identity recheck          = 10 / 10 PASS, drift 0
adopted separator form    = "} ; .if" x4, "} ; .echo" x2
untested form present     = 0
authorized script         = EA87DE2E419BC73941FB220AF5BA642DB2009E52C5AEF7D07BF45DFC620E2F55
authorized script bytes   = 1,184
reserved log              = fixed, not created
expected markers          = 17
expected $t18 on success  = 6

maximum invocations       = 1 (consumed on launch)
B1 runtime status         = INCONCLUSIVE (unchanged)
Candidate A               = PROVEN (unaffected by any outcome)

CDB execution / ping execution / rerun / script modification / new .cdb = 0
runtime acceptance        = NOT ESTABLISHED by this gate
IMPLEMENTATION = FORBIDDEN
```
