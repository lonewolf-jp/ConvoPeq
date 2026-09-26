# P3-5-FPM-C3-T1-Capture — C3-Redesign-2 Correction / CDB Creation + Static Validation

## 1. Gate result

```text
Gate     = P3-5-FPM-C3-T1-Capture-C3-Redesign-2-Correction-CDB-Creation-Static-Validation
corrected .cdb created = 1
CDB execution          = 0
AudioEngineHarness     = 0
build                  = 0
Runtime Authorization  = 0
production / test / CMake source modification = 0

V01..V20 = 18 PASS, 2 FAIL  (V12, V15)
VERDICT  = STATIC VALIDATION FAIL
```

Per the stop conditions, execution stops here. No Runtime Authorization is sought.

## 2. Artifact created

```text
source      doc/work113/P1-5-IR-P2_Step5BB_P3-5-FPM-C3-T1-Capture-Redesign-2.cdb
source SHA  FBE32851F795AF900E6C46D6E38F25138E01217C2B52C0BDDE1F015029CFC39E   (unchanged)

created     doc/work113/P1-5-IR-P2_Step5CO_P3-5-FPM-C3-T1-Capture-Redesign-3_Corrected.cdb
new SHA     5E977694DE91273AA8126DF6D4A714E1CA103D0447E55EF4323DC936834E52A7
new bytes   14,377      (identical to the source, 0xc0 and 0x00 are both three characters)
encoding    ASCII, no BOM, 41 CRLF pairs, trailing LF
```

## 3. The edit

A single literal substitution, applied to a copy:

```text
find    @$t2+0xc0
replace @$t2+0x00
count   7 replacements, 7 sites
```

The literal was chosen so the substitution is provably confined: every `0xc0` in the file is one of
these seven, and the slot-window series is anchored on `@$t1+0x20`, a different string. A blind
global delete of `0xc0` was not used.

| line | RVA | marker | class |
|---|---|---|---|
| 28 | `0x1f9c5da` | `C3_T0_SEQUENCE_PUBLISHED` | dump |
| 29 | `0x1fa80b3` | `C3_TERMINAL_S7_ANCHOR` | dump |
| 30 | `0x1f9cfb0` | `C3_T1A_CANDIDATE` | dump |
| 30 | `0x1f9cfb0` | `C3_T1A_CANDIDATE` | candidate-entry dump |
| 34 | `0x1f9cd59` | `C3_T1_SELECTION` | guard |
| 35 | `0x1f9cd72` | `C3_T1_CAS_PRE` | guard |
| 37 | `0x1fa80b2` | `C3_T2_WAIT_RETURN` | dump |

## 4. Tally

| # | check | observed | result |
|---|---|---|---|
| V01 | corrected script SHA and bytes frozen | `5E977694…E52A7`, 14,377 | PASS |
| V02 | `ConvoPeq.md` SHA and bytes | `E5E74200…F3609`, 5,535,334 | PASS |
| V03 | `AudioEngineHarness.exe` SHA | `E5C7AFB9…34C75` | PASS |
| V04 | `cdb.exe` SHA and version | `5F54ABAF…FBEE67`, `10.0.29617.1000` | PASS |
| V05 | breakpoint RVA set, ordered | identical, 13 entries | PASS |
| V06 | marker set, ordered | identical, 64 `.echo` markers | PASS |
| V07 | `0xC0` remaining in any `@$t2` expression | 0 | PASS |
| V08 | `@$t1+0x20` window terms | 256, unchanged | PASS, see 4.1 |
| V09 | corrected entry form at both guards | present at `0x1f9cd59` and `0x1f9cd72` | PASS |
| V10 | writer sequence, ordered | identical, 71 writes | PASS |
| V11 | register set | unchanged, max index 19, no new register | PASS |
| V12 | O1-new `dd @$t1 L1` present | **0 occurrences** | **FAIL**, see 4.2 |
| V13 | `0x34000` never used as `globalEpoch` | not used | PASS |
| V14 | brace balance per body | 0 unbalanced | PASS |
| V15 | block close followed by `;`, never a space | **85 space-separated** | **FAIL**, see 4.3 |
| V16 | no resume inside a non-resuming body | no such body; `.else`/`gc` here are by design | PASS, rescoped |
| V17 | `@$pc`, `dwo`, `poi` | 0, 0, 0 | PASS |
| V18 | existing STOP markers retained | 11 `C3_CR1_STOP_*`, plus 3 named markers | PASS |
| V19 | source, test, CMake, build untouched | untouched | PASS |
| V20 | 64 slot windows anchored at `$t1 + 0x20` | 256 window terms, anchor intact | PASS |

### 4.1 V08, the stated premise was wrong, and that is reported rather than papered over

The instruction stated the slot-window `0xC0` count should be 64 and unchanged. Measured:

```text
'0xc0' anywhere in the file            = 0
'@$t1+0x20' terms                      = 256   (64 windows x 4 dump sites)
'db @$t1+0x20+' window terms           = 256
```

The window series runs `0x0, 0x50, 0xa0, 0xf0, 0x140, ...` at stride `0x50`. `0xC0` is 192, and
`192 / 80 = 2.4`, so `0xC0` is **not** a window origin and does not appear. The intended property,
that the windows are preserved and anchored at `$t1 + 0x20`, is verified and holds. The literal
count of 64 could not be reproduced because it does not exist. The V07 and V08 pair still
discriminates, and more sharply than expected: in this file `0xC0` was exclusively the defect, so
the substitution could not have touched a window even in principle.

### 4.2 V12 FAIL, and it is a scope consequence, not an oversight

O1-new was adopted by the Correction Design as `dd @$t1 L1`, the `globalEpoch` capture. It is
**not present** in the corrected script, because the change list authorized for this gate contained
only the seven `+0xC0` removals. Adding it would have been an eighth change.

```text
'dd @$t1 L1' occurrences = 0
```

Consequence: the corrected script still cannot read `globalEpoch`, so the Case C distinction
required by the S7 proof contract, equality against `globalEpoch`, cannot be made from this script.
V13 passes only in the weaker sense that nothing misuses `0x34000` as `globalEpoch`; the positive
capture is missing.

### 4.3 V15 FAIL, 85 violations, all inherited

```text
C3-Redesign-2   '} .else' / '} .if' / '} .echo'  = 85
corrected       '} ;'                              = 1
corrected       '} .else' / '} .if' / '} .echo'  = 85
introduced by this gate                           = 0
inherited from the source                         = 85
```

Distribution, all thirteen breakpoints affected:

| line | RVA | marker | count |
|---|---|---|---|
| 25 | `0x1f9c550` | `C3_T0_ENTRY` | 3 |
| 26 | `0x1f9c57d` | `C3_T0_CAS_PRE` | 4 |
| 27 | `0x1f9c59a` | `C3_T0_CAS_POST` | 4 |
| 28 | `0x1f9c5da` | `C3_T0_SEQUENCE_PUBLISHED` | 7 |
| 29 | `0x1fa80b3` | `C3_TERMINAL_S7_ANCHOR` | 4 |
| 30 | `0x1f9cfb0` | `C3_T1A_CANDIDATE` | 7 |
| 31 | `0x1f9cfe2` | `C3_T1B_GETMIN_CALL_PRE` | 5 |
| 32 | `0x1f9cfe5` | `C3_T1B_GETMIN_RETURN` | 5 |
| 33 | `0x1f9cd00` | `C3_T1C_DQUEUE_ENTRY` | 7 |
| 34 | `0x1f9cd59` | `C3_T1_SELECTION` | 14 |
| 35 | `0x1f9cd72` | `C3_T1_CAS_PRE` | 12 |
| 36 | `0x1f9ce04` | `C3_T1_CAS_SUCCEEDED` | 8 |
| 37 | `0x1fa80b2` | `C3_T2_WAIT_RETURN` | 5 |

This is the exact construct class the B1 chain characterised and proved against this CDB build.
From the B1 Result Audit, on `10.0.29617.1000`:

```text
"}" followed by a space   does NOT act as a command boundary
"}" followed by ";"       DOES act as a command boundary
```

and the corrected probe succeeded only after every one of its inter-block transitions was changed to
`} ; `. Three consecutive executions established it, including bodies that differed in length and in
the construct following the brace, sharing only the separator.

Applied here: **every one of the thirteen breakpoint bodies in C3-Redesign-2 contains the
space-separated form, so on this CDB build none of them would parse past its first block
boundary.** That is consistent with C3-Redesign-2 never having produced a completed capture, and
with the `Extra character error` / `Syntax error` class of failure seen throughout this workstream.

This is a pre-existing defect that the authorized seven-site scope cannot reach. Fixing it means
changing 85 sites, which is a different and much larger authorization than the one granted.

### 4.4 V16 rescoped, with the reason stated

V16 as drafted in the Correction Design read "no `.else`". That criterion is wrong for this script.
Here `.else` and `gc` are the intended control flow of a conditional breakpoint, which must resume
on a non-matching hit; the earlier B1 probes failed precisely because they lacked that resume path.
V16 is therefore rescoped to "no resume command inside a body that must not resume", and it passes.
The separator defect that `.else` participates in is carried by V15, which fails.

## 5. Why execution is blocked

```text
V15 FAIL  the script is not expected to parse on the authorized CDB build, so an
          authorized single execution would very likely abort at the first block
          boundary of every breakpoint and yield no T1 evidence
V12 FAIL  globalEpoch cannot be read, so the Case C discrimination the S7 contract
          requires is not available from this script
```

Spending a single authorized execution on a script that carries a known parser-breaking construct
would consume the authorization for a predictable null result. The B1 chain already spent three
executions learning this the expensive way; the finding is being applied instead of re-tested.

## 6. What is required to proceed

Two authorizations, both outside the one granted:

```text
A1  convert the 85 inter-block transitions from "} .else" to "} ; .else"
    across all thirteen bodies, and re-validate
A2  add the O1-new capture 'dd @$t1 L1' at the T1 sites, and re-validate
```

A1 is a mechanical, single-rule transformation of the same kind already applied seven times here,
and it is the direct application of a PROVEN result from this workstream rather than a new design.
A2 is a single added read through an existing register, and it spends no register budget.

Neither is performed in this gate. The corrected `.cdb` is left in place as the artifact of the
authorized seven-site edit, with its SHA recorded, and is not treated as a candidate for execution.

## 7. Constraints honored

```text
changes outside the 7 sites   = 0, verified by V05, V06, V10, V11 and the
                                byte census in section 8
$t1 modified                  = 0, V08 and V20
$t1 + 0x20 modified           = 0, 256 terms unchanged
$t2 modified                  = 0
+0x30000 / +0x34000 / +0x34040 modified = 0
ReaderSlot windows modified   = 0
breakpoint RVAs modified      = 0, V05
markers modified              = 0, V06
writer set modified           = 0, V10
pseudo-register added         = 0, V11
C3-Redesign-2.cdb modified    = 0, SHA unchanged
CDB / harness execution       = 0
build                         = 0
source / test / CMake         = untouched
```

## 8. Byte-level confirmation of the edit surface

```text
source bytes   14,377      new bytes  14,377      delta 0
BOM            none        new BOM    none
non-ASCII      0           new        0
CRLF pairs     41          new        41
trailing LF    yes         new        yes
'@$t2+0xc0'    7  ->  0
'@$t2+0x00'    0  ->  7
'0xc0'         7  ->  0
'@$t1+0x20'  256  ->  256
```

## 9. Base state, unchanged

```text
U3 = SUCCESS    S-14-A = PROVEN    S-14-B = PROVEN
STOP-C3-T1-11 = CLEARED    STOP-C3-T1-12 = CLEARED
Candidate A = PROVEN    B1 = PROVEN    @$bp0 = PROVEN    $t11 = PROVEN

S7_READER          = UNRESOLVED, frozen for a later execution result
S7_READER_SLOT     = UNRESOLVED, frozen for a later execution result
minReaderEpoch(T1) = NOT CAPTURED, frozen for a later execution result
Case A/B/C/D       = NOT PROVEN, frozen for a later execution result
IMPLEMENTATION     = FORBIDDEN
```

None of these is advanced by creating a script or by validating it statically. They are decided by an
execution result, and this gate performed none.

## 10. Final state

```text
C3-Redesign-2 Correction / CDB Creation + Static Validation
= CLOSED / STATIC VALIDATION FAIL / NOT AUTHORIZED

corrected .cdb   = 5E977694DE91273AA8126DF6D4A714E1CA103D0447E55EF4323DC936834E52A7
                   doc/work113/..._C3-T1-Capture-Redesign-3_Corrected.cdb
7-site edit     = applied, 7 of 7, byte-neutral
C3-Redesign-2   = FBE32851F795AF900E6C46D6E38F25138E01217C2B52C0BDDE1F015029CFC39E, unchanged

V01..V20 = 18 PASS / 2 FAIL
  V12 FAIL  O1-new 'dd @$t1 L1' absent, outside the authorized scope
  V15 FAIL  85 space-separated block closes inherited from C3-Redesign-2,
            the construct this CDB build rejects per the PROVEN B1 result

CDB execution / harness execution / build / Runtime Authorization = 0
.cdb creation = 1, the single authorized artifact
Static Validation of the corrected script = FAIL
Runtime Authorization = NOT REQUESTED
Execution of the C3 capture = BLOCKED
```
