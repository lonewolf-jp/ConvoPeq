# C3 Entry-Gate Diagnostic — Redesign-6 Implementation + Static Validation

## 1. Gate result

```text
Gate   = P3-5-FPM-C3-T1-Capture-Redesign-6-IMPLEMENTATION-STATIC-VALIDATION
.cdb created            = 1, Redesign-6
bodies rewritten        = 4 of 4
C1..C6                  = all satisfied, self-checked at apply time
mechanical checks       = 17 total, 15 PASS, 2 not-decidable-by-design
VERDICT = STATIC VALIDATION PASS

D6-1 .. D6-9            = PASS
D6-10 static portion    = PASS
D6-10 runtime portion   = UNPROVEN, deferred to the single authorized execution
D6-11                   = NOT STATISTICALLY PROVABLE, deferred to the same execution
D6-12 .. D6-14          = PASS

CDB execution = 0    harness execution = 0    build = 0
production source = 0    test = 0    CMake = 0
Runtime Authorization = 0, NOT GRANTED
```

## 2. Artifacts

```text
source (frozen)  doc/work113/P1-5-IR-P2_Step5CT_…_Redesign-5_EntryGate-Diagnostic.cdb
                 247B6A71752AC1986600A52A5D5F4EFC84DDACD1DC8A27203EDB78657C45B47F   unchanged
                 17,427 bytes

created          doc/work113/P1-5-IR-P2_Step5CX_…_Redesign-6_EntryGate-Diagnostic.cdb
                 8D64862031F24A5285EEAFC10DBCD00C4A94C4CFB635CD2E891D227AB5CC23A5
                 14,712 bytes, delta -2,715
                 ASCII, no BOM, 41 CRLF, 41 lines, trailing CRLF
```

## 3. What each T0 body now is

```text
bp AudioEngineHarness+0x<rva> ".echo C3_DG_<tag>_HIT ; <existing guard, byte-identical>"
```

| line | RVA | diagnostic | diagnostic bytes | guard bytes | guard is the frozen Redesign-4 guard |
|---|---|---|---|---|---|
| 25 | `0x1f9c550` | `.echo C3_DG_T0E_HIT` | 19 | 166 | yes |
| 26 | `0x1f9c57d` | `.echo C3_DG_T0P_HIT` | 19 | 229 | yes |
| 27 | `0x1f9c59a` | `.echo C3_DG_T0Q_HIT` | 19 | 230 | yes |
| 28 | `0x1f9c5da` | `.echo C3_DG_T0S_HIT` | 19 | 2,207 | yes |

Everything the authorization forbade is absent, and the absence is checked rather than asserted:

```text
?                            0 in any diagnostic
register write               0 in any diagnostic
memory read                  0 in any diagnostic
address formation            0 in any diagnostic
.if / .else verdict pairs    0 in any diagnostic
counter                      0 in any diagnostic
R9CHANGE                     0 in any diagnostic
operand capture              0 in any diagnostic
new constants                0
guard edits                  0, byte-identical to the frozen Redesign-4
```

Redesign-5's preambles were 540, 661, 661 and 929 bytes. They are now 19 bytes each, a reduction of
2,823 bytes of unverified composition, of which 108 bytes are the register writes that were never
observed to take effect.

## 4. C1 to C6, self-checked during the transform

| ID | constraint | result |
|---|---|---|
| C1 | the diagnostic is a single command, one separator | PASS, no `;` inside any diagnostic region |
| C2 | that command is `.echo <reachability marker>` | PASS, exact match |
| C3 | no pseudo-register written | PASS, regex `r $tN=` finds 0 in diagnostics |
| C4 | no memory read, no address formed | PASS, no `?` and no `dd` in diagnostics |
| C5 | no constant, no comparison, no inference | PASS, no `=`, `!`, `<`, `>` in diagnostics |
| C6 | exactly one ` ; ` then the byte-identical guard | PASS, separator unique and guard equal |

Whole-line exact reconstruction also held for all four:
`new_line == prefix + diagnostic + " ; " + frozen_guard + suffix`.

## 5. The empirical basis for the one construct used

Redesign-6 depends on `.echo` tolerating a following `;`. That was observed directly, at this
position, in this body, in the Step5CU log:

```text
Redesign-5 preamble began   .echo C3_DG_T0E_HIT ; ? @$t0 ; ...
the log then emitted, as separate lines
  C3_DG_T0E_HIT
  Evaluate expression: 0 = 00000000`00000000
```

`.echo` printed only its argument and the `;` began a new command. It is also the idiom the existing
guards already use, for example `.echo C3_T0_ENTRY; r; dq @rsp L1; ln poi(@rsp); gc`. This is the
only construct class in the previous preamble with directly observed terminal-semicolon tolerance
at this position, which is exactly the criterion D14 requires for admission.

## 6. Static validation D6-1 to D6-14

| ID | result | check | observed |
|---|---|---|---|
| D6-1 | PASS | the four T0 addresses unchanged, topology identical | 13 RVAs ordered identical |
| D6-2 | PASS | `$t0..$t19` roles unchanged | diagnostic register writes 0; guard writes 71 = 71; registers 0..19 |
| D6-3 | PASS | T1/T2/S7 preconditions unchanged | all 9 non-T0 bodies byte-identical |
| D6-4 | PASS | `@r9 == 9` unchanged | R4 = 4, R6 = 4 |
| D6-5 | PASS | enqueuePos `== 4` unchanged | `@r11+0x34000) == 4` 1 = 1, `@rcx+0x34000) == 4` 1 = 1 |
| D6-6 | PASS | `@rcx` / `@r11` site-specific base selection unchanged | vacuously: the diagnostic forms no address |
| D6-7 | PASS | ReaderSlot windows unchanged | 256 terms, 64 distinct offsets, base `0x0`, last `0x13b0` |
| D6-8 | PASS | A2 `dd @$t1 L1` unchanged | 7 = 7 |
| D6-9 | PASS | existing guard byte identity | `guard[j:] ==` frozen Redesign-4 guard, whole-line exact reconstruction, 4 of 4 |
| D6-10 static | PASS | `gc` count unchanged, diagnostic is 1 command, separator is 1 | `gc` 61 = 61, single-command true, single-separator true |
| D6-10 runtime | **UNPROVEN** | that the `gc` is actually reached | not statically decidable |
| D6-11 | **NOT PROVABLE** | diagnostic failure does not strand the debuggee | not statically decidable at all |
| D6-12 | PASS | no new constants or inference | every changed line is exactly prefix + diagnostic + `; ` + guard; `C3_DG_*` tokens are exactly the four |
| D6-13 | PASS | source, harness, build, CMake unchanged | see section 8 |
| D6-14 | PASS | execution count 0 | no debugger process launched |

Diff confirmation, as required:

```text
lines differing from the frozen Redesign-4 = [25, 26, 27, 28]   and no other
per line, the decomposition [diagnostic] ; [original guard] holds, with the guard taken
byte-for-byte from Redesign-4 rather than from the parent, so identity is transitive
```

### 6.1 Why two entries are reported FAIL and the verdict is still PASS

The two FAIL rows are D6-10 runtime and D6-11. They are recorded as not decidable rather than
quietly omitted, because the Design Gate stated that they are not statically decidable and it would
be dishonest to let a mechanical tally imply otherwise. The tally is 15 PASS, 2 not-decidable, and
the two not-decidable items are the explicit acceptance criteria of the next execution.

This is also the direct lesson of Redesign-5: the Step5CT static set was 32 of 32 PASS and the
preamble still destroyed the measurement. No static check in this set can see that failure, which is
why continuation is a runtime criterion and not a certified property.

## 7. Two tooling defects during this gate

Neither changed the artifact; both produced wrong intermediate results and were caught before the
artifact was accepted.

```text
DEFECT 1  the transitivity pre-check compared Redesign-4's guard against Redesign-5's whole
          quoted body, which of course includes the Redesign-5 preamble, so it aborted on
          line 25 with a false "guard differs". The check was redundant: the unique-split
          test already requires body[j+3:] == the frozen guard exactly. Removed.

DEFECT 2  C1 was specified as "exactly one ';' in the body", but the guard legitimately
          contains 61 ';' of its own, so C1 failed for all four lines. C1 was rescoped to
          the diagnostic region, which is the region the constraint is about. The
          constraint's intent, one command and one separator, holds.
```

Both are the recurring family: a check written to a convenient proxy for the property rather than
to the property. C1 counting semicolons across the whole body is the same error shape as the
earlier `$d` concatenation and the backtick-in-single-quotes defects.

## 8. Constraints honored

```text
.cdb edited beyond the 4 T0 bodies   = 0, DIFF shows [25,26,27,28] only
pseudo-registers written             = 0
$t0 .. $t19                          = untouched
@r9 == 9                             = untouched, 4 occurrences
enqueuePos == 4                      = untouched
CAS conditions                       = untouched
T1 correlation                       = untouched, 9 bodies byte-identical
ReaderSlot windows / @$t1+0x20       = untouched, 256 terms
A2 'dd @$t1 L1'                      = untouched, 7 occurrences
gc                                   = untouched, count 61
q                                    = untouched
breakpoint topology                  = untouched, 13
marker set                           = 64 pre-existing, untouched
new constants                        = 0
canary                               = not added
Redesign-5 / Redesign-4 / C3-Redesign-2 = unchanged, SHAs recorded
ConvoPeq.md / harness / cdb.exe      = unchanged, SHAs recorded
src / CMakeLists.txt / build.bat    = 12 pre-existing entries, unchanged
CDB / harness execution / build      = 0
```

## 9. Frozen items, unchanged

```text
S7_READER / S7_READER_SLOT / minReaderEpoch(T1) / Case A / B / C / D = UNRESOLVED
A1 separator = RUNTIME PROVEN
A2           = installed, CDB accepted, execution count 0, globalEpoch value UNPROVEN
T0 gate cause = UNRESOLVED OVER TIME.  Branch B is a sample-level observation only.
IMPLEMENTATION = FORBIDDEN
```

Stage 1 answers reachability and continuation only. It does not adjudicate the entry condition, and
the one operand sample from Redesign-5 is not treated as representative.

## 10. Final state

```text
Redesign-6 Implementation + Static Validation
= CLOSED / STATIC VALIDATION PASS

execution candidate
  doc/work113/P1-5-IR-P2_Step5CX_…_Redesign-6_EntryGate-Diagnostic.cdb
  8D64862031F24A5285EEAFC10DBCD00C4A94C4CFB635CD2E891D227AB5CC23A5
  14,712 bytes

CDB execution / harness execution / build = 0
Runtime Authorization = NOT GRANTED
```

## 11. Runtime Authorization requested

One execution of Redesign-6 against
`build/Release/AudioEngineHarness.exe --measurement=normal`, using the form established at Step5CQ:

```text
tmp\cdb.exe -cf <script> -logo <log> <exe> --measurement=normal
```

```text
R1   script completes AND ZwTerminateProcess is reached    <- primary, tests D6-10r and D6-11
R2   C3_DG_T0E_HIT
R3   C3_DG_T0P_HIT
R4   C3_DG_T0Q_HIT
R5   C3_DG_T0S_HIT
R9   C3_T0_ENTRY
R10  C3_T0_SEQUENCE_PUBLISHED
R11+ T1A / T1B onward, minReaderEpoch, ReaderSlot, CAS_PRE, CAS_POST
     interpreted ONLY if R9 or R10 actually fired
```

R6 operand capture, R7 conjunct verdicts and R8 R9CHANGE are **absent by design** in Stage 1. Their
absence must not be recorded as a defect. Per D14, the next construct class is admitted only after
this run proves continuation, one authorization at a time.
