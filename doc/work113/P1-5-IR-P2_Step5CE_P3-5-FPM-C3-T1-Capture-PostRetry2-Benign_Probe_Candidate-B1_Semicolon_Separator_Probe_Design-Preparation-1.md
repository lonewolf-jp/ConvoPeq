# P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Candidate-B1-Semicolon-Separator-Probe-Design-Preparation-1

## 1. Gate result

```text
Gate                    = P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Candidate-B1-Semicolon-Separator-Probe-Design-Preparation-1
Mode                    = syntax design and artifact preparation only
CDB execution           = 0
ping execution          = 0
runtime authorization   = 0
static validation       = 0   (deferred to its own gate)
rerun                   = 0
new .cdb created        = 1
source modification     = 0
VERDICT                 = DESIGN FROZEN / NOT VALIDATED / NOT AUTHORIZED
```

This gate fixes a syntax design. It runs nothing and asserts nothing about runtime behavior.

## 2. Source authority

```text
ConvoPeq.md SHA-256 = E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
ConvoPeq.md bytes   = 5,535,334
```

`ConvoPeq.md` is the only production source authority. The superseded `ConvoPeq(3).md` was not used
as a substitute. This probe does not reach ConvoPeq reader, epoch, or reclaim state, so no
source-level change arises and none was made.

## 3. Design intent

One change only: make the command separator explicit after every block-closing brace.

```text
before    ".if (T1) { ... } .if (T2) { ... } .if (T3) { ... }"
after     ".if (T1) { ... } ; .if (T2) { ... } ; .if (T3) { ... }"
```

The evidence base is the two consecutive executions already performed:

```text
run 1  old body, 1011 chars, 7 x "} .else {"   stopped after the first crosscheck
run 2  new body,  696 chars, 4 x "} .if  {"   stopped after the first crosscheck
```

In both runs the first crosscheck, introduced by `;`, executed; every block reached through a space
after `}` was rejected with `Extra character error`. Removing `.else`, `q`, `gc` and the outer `.if`
wrapper did not change the stopping point, so the separator, not the construct, is the variable
under test.

## 4. Prepared artifact

```text
Path    = doc/work113/P1-5-IR-P2_Step5CE_P3-5-FPM-C3-T1-Capture-PostRetry2-Benign_Probe_Candidate-B1_Semicolon_Separator_Probe.cdb
SHA-256 = EA87DE2E419BC73941FB220AF5BA642DB2009E52C5AEF7D07BF45DFC620E2F55
bytes   = 1,184
lines   = 27
encoding= ASCII, no BOM, LF only, trailing LF present
```

The artifact was produced by transforming the frozen corrected probe rather than by retyping it, so
that the change surface is mechanically bounded.

## 5. Change surface

```text
lines changed            = 1  (the "bp" line, line 22)
line 22 length           = 713 -> 725 chars, delta +12
breakpoint body length   = 696 -> 708 chars, delta +12
transitions rewritten    = 6
```

| # | transition | before | after |
|---|---|---|---|
| 1 | T1 → T2 | `} .if` | `} ; .if` |
| 2 | T2 → T3 | `} .if` | `} ; .if` |
| 3 | T3 → T4 | `} .if` | `} ; .if` |
| 4 | T4 → T5 | `} .if` | `} ; .if` |
| 5 | T5 → CROSSCHECK_END | `} .echo` | `} ; .echo` |
| 6 | sentinel → TALLY | `} .echo` | `} ; .echo` |

```text
"} ; .if"   transitions in new body = 4
"} ; .echo" transitions in new body = 2
"} ." without a semicolon           = 0
```

## 6. Proof that nothing else changed

The new body was stripped of the inserted separator and compared to the frozen body:

```text
remove every "; " that follows a "}"  ->  byte-identical to the frozen 696-char body
comparison                             =  case-sensitive exact string equality, TRUE
```

Since the transform is exactly reversible to the previously frozen and static-validated body, every
element of that body other than the six separators is preserved by construction. This is a stronger
guarantee than a re-read of the new file would give.

## 7. Preserved invariants

| invariant | evidence in new body |
|---|---|
| `@$bp0 - 0x39d9` derivation | `r $t11=@$bp0-0x39d9` x1 |
| B1 derived anchor set | `$t12/$t13/$t14/$t15/$t6` = `@$t11` + `0x3c/0x4FD0/0x2600/0x5000/0x40` |
| five comparisons | `.if (@$t1 == @$t12)` … `.if (@$t5 == @$t6)` x1 each |
| sentinel | `.if (@$t1 != 0xDEAD0001)` x1 |
| counter | `r $t18=@$t18+1` x6 |
| readback | `r $t11; r $t1..$t5` and `r $t16; r $t17; r $t18; r $t19` |
| markers | 17 total, 11 in the body |
| brace balance | 6 open / 6 close |

```text
Candidate A at script level = r $t1 = ping+0x3c
                              r $t2 = ping+0x4FD0
                              r $t3 = ping+0x2600
                              r $t4 = ping+0x5000
                              r $t5 = ping+0x40
script-level g and q         = present, outside the body, unchanged
```

## 8. Design decision on the Candidate A side of each comparison

The instruction for this gate listed the preserved anchors as `$t1 = $t11 + 0x3c` and so on. That
notation was **not** applied to the Candidate A assignments, and the reason is recorded here because
it is a semantic decision rather than a formatting one.

If Candidate A were also written as `$t11 + RVA`, each comparison would compare `$t11 + RVA` against
`$t12`, which is itself assigned `$t11 + RVA` a few statements earlier. The comparison would then be
self-referential, would hold for any RVA including a wrong one, and would prove nothing. The offsets
are preserved exactly; the derivation *source* is what differs, and that difference is the invariant
under test:

```text
Candidate A side   image base from the "ping" module symbol, resolved before the breakpoint
B1 side            image base from the breakpoint address "@$bp0" minus the known RVA
agreement          two independent derivations of the same base
```

Re-deriving Candidate A from `$t11` would delete the property being measured. The anchors, the five
comparisons, the sentinel, the counter and the readback are therefore all unchanged in meaning, as
required.

## 9. Residual design risk

One untested choice remains, and it is recorded so that a failure is not misdiagnosed.

```text
chosen form     "} ; .if"    (space, semicolon, space)
untested form   "}; .if"     (semicolon immediately after the brace)
```

The form proven to work in both prior runs is `X; .if`, where the `;` directly follows the previous
command. The design uses `} ; .if`, placing a space between `}` and `;`, because that is the form
specified for this gate. If a future execution again stops after the first crosscheck, the
difference between these two spellings is the first variable to examine, ahead of any other
hypothesis. No claim is made here that either form is accepted by this CDB build; only Static
Validation can certify the structure, and only execution can certify the behavior.

## 10. What this probe does not do

Its purpose is a single question, on a benign target:

```text
Can several .if blocks be chained and fully executed inside one CDB breakpoint command body?
```

A positive result establishes nothing about the production capture:

```text
S7_READER          = UNRESOLVED
S7_READER_SLOT     = UNRESOLVED
minReaderEpoch     = NOT CAPTURED
T1                 = NOT CAPTURED
Case A/B/C/D       = NOT PROVEN
IMPLEMENTATION     = FORBIDDEN
```

## 11. Not performed in this gate

```text
CDB execution / ping execution / runtime authorization / static validation / rerun = 0
AudioEngineHarness / production C3 / T1 capture / ReaderSlot / EpochDomain
getMinReaderEpoch / DQueue / reclaim / M1 / M2 / Retry-3 / Preparation-4
build / Dr.Memory / production, test, CMake, source modification = 0
```

The frozen corrected probe and the four other `.cdb` files are untouched. Only the new artifact named
in section 4 was created.

## 12. Next gates

```text
Semicolon-Separator Design/Preparation-1   CLOSED  (this gate)
        |
        v
Semicolon-Separator Static Validation-1    NOT STARTED   <- separate gate
        |
        v
Semicolon-Separator Runtime Authorization-1 NOT STARTED  <- separate gate, only if validation passes
        |
        v
Semicolon-Separator Benign Execution-1      NOT STARTED
        |
        v
Result Audit-1                             NOT STARTED
```

Static Validation is a separate gate and is not performed here. Runtime Authorization is a further
separate gate and is conditional on Static Validation passing. No execution is authorized by this
document.

## 13. Final state

```text
P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Candidate-B1-Semicolon-Separator-Probe-Design-Preparation-1
= CLOSED / DESIGN FROZEN / NOT VALIDATED / NOT AUTHORIZED

artifact                = EA87DE2E419BC73941FB220AF5BA642DB2009E52C5AEF7D07BF45DFC620E2F55
artifact bytes          = 1,184
lines changed vs frozen = 1
body length             = 708  (696 + 12)
separators rewritten    = 6  ( 4 x "} ; .if", 2 x "} ; .echo" )
reversible to frozen    = TRUE, case-sensitive exact
".else" / gc / q / @$pc / dwo / poi / dd / dq in body = 0
brace balance           = 6 / 6
markers                 = 17
expected $t18 on success = 6

CDB execution / ping execution / runtime authorization / static validation / rerun = 0
IMPLEMENTATION = FORBIDDEN
```
