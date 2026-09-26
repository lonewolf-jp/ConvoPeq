# P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Candidate-B1-Semicolon-Separator-Probe-Static-Validation-1

## 1. Gate result

```text
Gate                  = P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Candidate-B1-Semicolon-Separator-Probe-Static-Validation-1
Mode                  = read-only formal static validation
CDB execution         = 0
ping execution        = 0
runtime authorization = 0
rerun                 = 0
script modification   = 0
new .cdb              = 0
V01..V46              = PASS
VERDICT               = STATIC VALIDATION PASS
```

## 2. Validation target

```text
Path    = doc/work113/P1-5-IR-P2_Step5CE_P3-5-FPM-C3-T1-Capture-PostRetry2-Benign_Probe_Candidate-B1_Semicolon_Separator_Probe.cdb
SHA-256 = EA87DE2E419BC73941FB220AF5BA642DB2009E52C5AEF7D07BF45DFC620E2F55
bytes   = 1,184
body    = 708 chars
```

## 3. Identity

| # | check | expected | observed | result |
|---|---|---|---|---|
| V01 | artifact SHA-256 | `EA87DE2E…0E2F55` | identical | PASS |
| V02 | artifact bytes | `1,184` | `1,184` | PASS |
| V03 | `cdb.exe` SHA-256 | `5F54ABAF…FBEE67` | identical | PASS |
| V04 | `cdb.exe` bytes | `178,016` | identical | PASS |
| V05 | `ConvoPeq.md` SHA-256 | `E5E74200…F3609` | identical | PASS |
| V06 | `ConvoPeq.md` bytes | `5,535,334` | identical | PASS |

```text
cdb version = 10.0.29617.1000
process residue = 0
frozen corrected probe (BZ) = FFC5228C2387BDC5264F4DB956D66E6DFFB46A97F97ADE2E59E0A7BA1AE739DE, unchanged
identity failures = 0
```

No drift, so no STOP.

## 4. The six transitions, individually

Each transition was located by exact full-pattern match, required to occur exactly once, and required
to sit at a distinct ascending position. Each was additionally checked for the command it introduces
and for the semantics of the block that command opens.

| # | transition | required form | exact count | position | introduces | block semantics | result |
|---|---|---|---|---|---|---|---|
| V07 | T1 → T2 | `} ; .if` | 1 | 304 | `.if (@$t2 == @$t13)` | `XCHK_T2_AGREE` | PASS |
| V08 | T2 → T3 | `} ; .if` | 1 | 366 | `.if (@$t3 == @$t14)` | `XCHK_T3_AGREE` | PASS |
| V09 | T3 → T4 | `} ; .if` | 1 | 428 | `.if (@$t4 == @$t15)` | `XCHK_T4_AGREE` | PASS |
| V10 | T4 → T5 | `} ; .if` | 1 | 490 | `.if (@$t5 == @$t6)` | `XCHK_T5_AGREE` | PASS |
| V11 | T5 → CROSSCHECK_END | `} ; .echo` | 1 | 551 | `.echo C3B1C_CROSSCHECK_END` | n/a | PASS |
| V12 | sentinel → TALLY | `} ; .echo` | 1 | 655 | `.echo C3B1C_TALLY` | n/a | PASS |

```text
separator 6/6 PASS
positions distinct = TRUE, strictly ascending
```

## 5. No separator lacks its semicolon

| # | check | observed | result |
|---|---|---|---|
| V13 | `}` not followed by `;` | 0 | PASS |
| V13 | `} ; .if` occurrences | 4 | PASS |
| V13 | `} ; .echo` occurrences | 2 | PASS |
| V13 | `} .` in the previous, space-only form | 0 | PASS |
| V13 | `};` no-space form | 0 | PASS |

The adopted spelling is `} ;` as specified for this gate. The untested `};` spelling was **not**
introduced, as instructed.

## 6. Separator-only transformation, independently re-derived

Preparation-1 was not taken on trust. The transformation was re-derived from the artifact.

| # | check | observed | result |
|---|---|---|---|
| V14 | removals of `; ` following a `}` | 6 | PASS |
| V14 | stripped length | 696, equal to the frozen body | PASS |
| V14 | case-sensitive exact equality with the frozen body | `True` | PASS |
| V14 | ordinal comparison, invariant culture | `True` | PASS |
| V15 | lines differing from the frozen file | 22 only | PASS |
| V15 | line count | 28 / 28 | PASS |
| V16 | body starts with `.echo`, ends with `r $t19` | true / true | PASS |
| V16 | `;;` absent, `{ ;` and `; {` absent | 0 / 0 / 0 | PASS |

Removing exactly the six inserted separators reproduces the previously frozen and static-validated
body byte for byte. The candidate therefore changes nothing except the separators.

## 7. B1 main structure

| # | check | expected | observed | result |
|---|---|---|---|---|
| V17 | `$t11 = @$bp0 - 0x39d9` | 1 | 1 | PASS |
| V18 | B1 anchor `$t12 = @$t11+0x3c` | 1 | 1 | PASS |
| V18 | B1 anchor `$t13 = @$t11+0x4FD0` | 1 | 1 | PASS |
| V18 | B1 anchor `$t14 = @$t11+0x2600` | 1 | 1 | PASS |
| V18 | B1 anchor `$t15 = @$t11+0x5000` | 1 | 1 | PASS |
| V18 | B1 anchor `$t6 = @$t11+0x40` | 1 | 1 | PASS |
| V19 | comparison T1 … T5 | 5 | 5 | PASS |
| V20 | sentinel check | 1 | 1 | PASS |
| V21 | counter increment `r $t18=@$t18+1` | 6 | 6 | PASS |
| V22 | `CROSSCHECK_END` | 1 | 1 | PASS |
| V23 | `SENTINEL_CLEARED` | 1 | 1 | PASS |
| V24 | `TALLY` | 1 | 1 | PASS |
| V25 | final counter readback | 1 | 1 | PASS |

## 8. Candidate A / B independence, a PASS condition

| # | check | expected | observed | result |
|---|---|---|---|---|
| V26 | Candidate A anchors at script level, all `ping`-derived | 5 | 5 | PASS |
| V27 | A side `$t1..$t5` assigned from `$t11` | 0 | 0 | PASS |
| V27 | B side `$t12,$t13,$t14,$t15,$t6` assigned from `@$t11` | 5 | 5 | PASS |
| V28 | comparison count | 5 | 5 | PASS |
| V28 | every A operand drawn from `{t1..t5}` | all | all | PASS |
| V28 | every B operand drawn from `{t6,t12,t13,t14,t15}` | all | all | PASS |

```text
pairings:  t1/t12   t2/t13   t3/t14   t4/t15   t5/t6
```

The two sides remain independently derived. Candidate A is **not** re-expressed as `$t11 + RVA`;
had it been, each comparison would compare a value against itself and the independence under test
would have been destroyed. This is enforced as a PASS condition, not merely noted.

## 9. Marker and counter order on the script

Order was verified on an execution-order model, since the body text sits textually inside the `bp`
line but executes only after the resume command. The model is: script lines 1–21, the `bp` command,
lines 23–24 (`BP_SET`, `bl`), the resume command, then the body, then lines 26–27.

| step | marker | offset | | step | marker | offset |
|---|---|---|---|---|---|---|
| 1 | `C3B1C_SCRIPT_BEGIN` | 6 | | 10 | `XCHK_T2_AGREE` | 767 |
| 2 | `C3B1C_SENTINEL_SET` | 218 | | 11 | `XCHK_T3_AGREE` | 829 |
| 3 | `C3B1C_ASSIGN_DONE` | 339 | | 12 | `XCHK_T4_AGREE` | 891 |
| 4 | `C3B1C_READBACK_DONE` | 389 | | 13 | `XCHK_T5_AGREE` | 952 |
| 5 | `C3B1C_BP_SET` | 415 | | 14 | `C3B1C_CROSSCHECK_END` | 992 |
| 6 | `C3B1C_BREAKPOINT_HIT` | 437 | | 15 | `C3B1C_SENTINEL_CLEARED` | 1047 |
| 7 | `C3B1C_B1_DERIVED` | 586 | | 16 | `C3B1C_TALLY` | 1096 |
| 8 | `C3B1C_CROSSCHECK_BEGIN` | 653 | | 17 | `C3B1C_SCRIPT_END` | 1146 |
| 9 | `XCHK_T1_AGREE` | 705 | | | | |

| # | check | result |
|---|---|---|
| V29 | all 17 markers present | PASS |
| V30 | strictly ascending execution order | PASS |
| V31 | first readback group precedes `CROSSCHECK_BEGIN` | PASS |
| V31 | final readback follows `TALLY` | PASS |
| V31 | `SCRIPT_END` follows the whole body | PASS |

This is script-level order only. **No attainment is asserted.** Nothing here claims any marker will
be emitted.

## 10. Forbidden constructs, body scope

| # | construct | observed | result |
|---|---|---|---|
| V32 | `.else` | 0 | PASS |
| V33 | `gc` | 0 | PASS |
| V33 | `q` | 0 | PASS |
| V33 | `g`, resume inside body | 0 | PASS |
| V34 | `@$pc` | 0 | PASS |
| V35 | `dwo` | 0 | PASS |
| V36 | `poi` | 0 | PASS |
| V37 | `dd` | 0 | PASS |
| V38 | `dq` | 0 | PASS |
| V39 | the word `resume` | 0 | PASS |

## 11. Forbidden identifiers, whole-script scope

| # | identifier | observed | result |
|---|---|---|---|
| V40 | `ReaderSlot` | 0 | PASS |
| V40 | `EpochDomain` | 0 | PASS |
| V40 | `getMinReaderEpoch` | 0 | PASS |
| V40 | `minReaderEpoch` | 0 | PASS |
| V40 | `DQueue` | 0 | PASS |
| V40 | `reclaim` | 0 | PASS |
| V40 | `AudioEngineHarness` | 0 | PASS |
| V40 | `AudioEngineHarness.exe` | 0 | PASS |

## 12. Script-level resume and quit, evaluated separately

The two scopes are assessed independently, as required. Presence at script level is by design and
does not offset the body-scope results in section 10.

```text
script-level "g"  exact line present = 1   permitted, outside the body
script-level "q"  exact line present = 1   permitted, outside the body
script-level "bl" exact line present = 1   breakpoint listing, by design
"g" inside body                      = 0
"q" inside body                      = 0
```

## 13. Brace and block structure

| # | check | observed | result |
|---|---|---|---|
| V41 | brace balance | 6 open / 6 close | PASS |
| V41 | maximum depth | 1, so no nesting | PASS |
| V41 | minimum depth | 0, so no premature close | PASS |
| V41 | final depth | 0 | PASS |
| V42 | block count | 6 | PASS |
| V42 | each block opens on a well-formed `.if` head | 6 / 6 | PASS |
| V42 | spans strictly ascending, no interleaving | yes | PASS |
| V43 | each block body exactly `.echo <marker>; r $t18=@$t18+1` | 6 / 6 | PASS |
| V44 | each block head well formed | 6 / 6 | PASS |
| V45 | top-level units, all separators at depth 0 | 27 units, final depth 0 | PASS |
| V46 | every `}` followed by `;` as the next separator | 6 / 6 | PASS |
| V46 | `}` not followed by `;` | 0 | PASS |

Block table, derived independently of textual enumeration:

| block | role | open | close | span | inner body |
|---|---|---|---|---|---|
| 1 | T1 | 266 | 304 | 38 | `.echo XCHK_T1_AGREE; r $t18=@$t18+1` |
| 2 | T2 | 328 | 366 | 38 | `.echo XCHK_T2_AGREE; r $t18=@$t18+1` |
| 3 | T3 | 390 | 428 | 38 | `.echo XCHK_T3_AGREE; r $t18=@$t18+1` |
| 4 | T4 | 452 | 490 | 38 | `.echo XCHK_T4_AGREE; r $t18=@$t18+1` |
| 5 | T5 | 513 | 551 | 38 | `.echo XCHK_T5_AGREE; r $t18=@$t18+1` |
| 6 | sentinel | 608 | 655 | 47 | `.echo C3B1C_SENTINEL_CLEARED; r $t18=@$t18+1` |

No block's braces interleave with another's. Each `.if` body is a single self-contained pair, and
the `} ;` boundary between consecutive blocks sits outside every block's own extent.

A negative control was run against the separator pattern: the text `} .if `, the form that failed at
runtime twice, is correctly **rejected** by the `^\}\s*;\s*` test used in V46. The check therefore
discriminates and does not pass vacuously.

## 14. Harness integrity

Several checks initially reported FAIL for reasons in the checking harness rather than in the
artifact. Each was diagnosed, corrected, and re-run. They are recorded so that the PASS results are
not credited to a harness that was merely permissive.

| # | harness defect | correction |
|---|---|---|
| 1 | literal `$` used unescaped as a regex, so `$t11` matched nothing | escaped via `Regex.Escape` |
| 2 | `IndexOf` on a shared prefix always matched the first transition | matched separator and introduced command jointly |
| 3 | marker search mixed body and whole-file frames, breaking monotonicity | single execution-order stream |
| 4 | execution-order model placed `BP_SET` after the body | the `bp` line's inline text excluded from the prefix |
| 5 | expected block body built by interpolation, yielding `r =@+1` | single-quoted literal |
| 6 | fixed-length lookback truncated `.if` to `if` | head extracted from the previous `;` |
| 7 | separator regex required nothing between braces | corrected to `^\}\s*;\s*` |
| 8 | cast of a register name such as `t1` to `Int32` threw, making a set test pass vacuously | string membership instead |

The artifact was not modified in response to any of these. The final consolidated figures in
section 15 come from a single re-verification pass.

## 15. Consolidated figures

```text
artifact SHA-256          = EA87DE2E419BC73941FB220AF5BA642DB2009E52C5AEF7D07BF45DFC620E2F55
artifact bytes            = 1,184
body length               = 708
'}' total                 = 6
'} ; .if'                 = 4
'} ; .echo'               = 2
'}' not followed by ';'   = 0
'} .if' old form          = 0
'.else'                   = 0
'q' / 'gc' / 'g' in body  = 0 / 0 / 0
counter increments        = 6
markers                   = 17
```

```text
V01..V46 = PASS
```

No item failed, so no Static Validation Failure Audit is required.

## 16. What this gate does not establish

```text
STATIC VALIDATION PASS = the semicolon-separated body is a structurally valid candidate
CDB runtime acceptance = NOT ESTABLISHED
```

Static validation confirms structure only. Whether this CDB build parses `} ; .if` as a command
boundary is a runtime property and remains unproven. The residual design risk recorded in
Preparation-1 stands unchanged: `} ; .if` is the adopted form, `}; .if` is untested, and the latter
belongs to a future candidate, not to this one.

## 17. Constraints honored

```text
CDB execution / ping execution / runtime authorization / rerun / script modification / new .cdb = 0
process residue = 0
execution logs unchanged = 4
.cdb total = 7, only the Preparation-1 artifact present
frozen BZ corrected probe = FFC5228C…AE739DE, unchanged
ConvoPeq.md = E5E74200…F3609, unchanged
src, test, CMake, build.bat delta = 12, unchanged from before this gate
```

`AudioEngineHarness`, production C3, T1 capture, ReaderSlot, EpochDomain, `getMinReaderEpoch`,
DQueue, reclaim, M1, M2, Retry-3, Preparation-4, build and Dr.Memory were not performed.

## 18. Base state, unchanged

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

## 19. Final state

```text
P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Candidate-B1-Semicolon-Separator-Probe-Static-Validation-1
= CLOSED / STATIC VALIDATION PASS

identity                     = 6 / 6 PASS
separator transitions        = 6 / 6 PASS
separator-only transform     = PASS, ordinal exact
Candidate A anchors          = 5 / 5, all ping-derived
B1 anchors                   = 5 / 5, all @$t11-derived
A/B independence             = PASS
comparisons                  = 5 / 5, correctly paired
marker presence and order    = 17 / 17, strictly ascending
counter structure            = 6 increments, 1 readback group
brace balance                = 6 / 6, max depth 1
block self-containment       = 6 / 6, no nesting, no interleaving
forbidden constructs         = 0
harness defects found        = 8, corrected and re-run

CDB execution / ping execution / runtime authorization / rerun / script modification / new .cdb = 0
IMPLEMENTATION = FORBIDDEN
```
