# Redesign-9 — Option Selection, Implementation Authorization, Implementation, Static Validation

## 1. Option selected: C

```text
SELECTED  Option C, .if and .echo only, zero '?'
REJECTED  Option A, structurally eliminated
REJECTED  Option B, structurally eliminated
```

### 1.1 Why A and B are eliminated rather than merely riskier

The selection turned on a correction to my own Redesign-7 Result Audit, issued before the choice was
made rather than after it.

The Step5DB report originally attributed the Redesign-7 failure to a second `?`. That was wrong. The
payload was

```text
.echo HIT ; ? dd(@rcx+0x34000) ; .echo EQ1 ; ? @r9 ; .echo EQ2 ; <guard>
```

and the observation is

```text
.eecho C3_DG_T0E_HIT    emitted,        line 175
? dd(@rcx+0x34000)     COMPLETED,      line 176 emitted the evaluation
.eecho C3_DG_T0E_EQ1    NOT emitted,    absent from the entire log
```

The witness bracket is `HIT | [ ? dd ] | EQ1`. The bracketed `?` completed, so it cannot be the
failing command, and the closing witness is absent. The command that did not complete is
**`.echo C3_DG_T0E_EQ1`, the first command following a completed `?`**. Confirmation that `EQ1` was
never emitted rather than emitted in another form: the only whole-line `C3_DG_*` occurrence in the
log is `C3_DG_T0E_HIT`; the two lines that merely contain the strings `EQ1` and `EQ2` are the `bp`
command echo and the `bl` listing, which is exactly what the whole-line marker rule excludes.

The command that failed was an `.echo`, the construct with the strongest evidence in this file, not a
second `?`. **The constraint is therefore not "at most one `?` per body". It is:**

```text
NO COMMAND MAY FOLLOW A COMPLETED '?' IN A BREAKPOINT BODY.
```

The failure is a property of the **position after a `?`**, not of the `?` construct.

The consequence is decisive rather than incremental:

```text
'?' can only ever be the LAST command of a body.
A body whose last command is '?' has no guard, so it has no terminal gc, so the debuggee is
stranded.

Option A   '.echo ; ? ; guard'   the guard follows the '?'   ELIMINATED
Option B   the same shape in each of its two runs           ELIMINATED
Option C   no '?' at all                                     the only surviving option
```

This is stronger than "A carries an unproven position". A and B are excluded by the constraint, and
the Design Gate's three-way comparison is resolved by measurement rather than by preference.

### 1.2 What Option C costs, stated plainly

```text
C yields the two BOOLEANS needed for the four-way discrimination.
C does NOT yield the raw values.
C cannot report what @r9 is, only whether it equals 9.
C cannot report what enqueuePos is, only whether it equals 4, and therefore cannot bound
  enqueuePos over the window or say whether 4 was ever approached.
```

The stated Design Gate objective was the two values, and C does not meet it as stated. C is selected
anyway because it is the only option that can produce a valid measurement at all, and because the
discrimination of the failing conjunct is the question that actually gates the next stage. The loss
of the raw values is a real loss and is carried forward explicitly, not absorbed.

### 1.3 What is not assumed by this selection

```text
NOT assumed  enqueuePos equals 4 at some hit
NOT assumed  enqueuePos can never equal 4
              the only datum remains enqueuePos = 221 at the first hit of two runs
NOT assumed  @r9 differs from 9.  Redesign-5 read @r9 = 1 at one hit, a single sample of an
              opaque token, not a verdict
NOT assumed  the mechanism behind the after-`?` failure
              it is not known.  The constraint is a bound, not a cause.
NOT assumed  that the four T0 sites always fire equally often
              measured once, 116 of 116, and a code path exists that can break the equality
```

## 2. Implementation Authorization, scoped to Option C

```text
TARGET   Redesign-9
SCOPE    the C3 T0E diagnostic body at 0x1f9c550, line 25, and nothing else

ALLOWED
  insertion of an Option C payload ahead of the T0E guard
  the payload consists solely of '.echo' and '.if' / '.else' carrying '.echo'

FORBIDDEN, and none of these was done
  guard modification
  '@r9 == 9' modification
  'enqueuePos == 4' modification
  any change to the meaning of $t0 .. $t19
  breakpoint topology modification
  T1 correlation modification
  ReaderSlot window modification
  A2 'dd @$t1 L1' modification
  source, test, CMake, build modification
  runtime execution
```

## 3. Artifact

```text
created   doc/work113/P1-5-IR-P2_Step5DD_…_Redesign-9_Conjunct-Verdicts.cdb
SHA-256   D48665B8E8B4DF780B6B174640EA869FFF6E2FF0F951D1799C1589809F5DAD14
bytes     14,887      delta +175 from Redesign-6
encoding  ASCII, no BOM, 41 CRLF, 41 lines, trailing CRLF

parent    Redesign-6   8D64862031F24A5285EEAFC10DBCD00C4A94C4CFB635CD2E891D227AB5CC23A5   unchanged
frozen    Redesign-4   A935369D711535C9837DFD25761B7886F6CB49D4157534BA34446DBB15F1022C   unchanged
frozen    Redesign-7   5AA98A592DD5B8745317A439AEC5BBC8B564A207D1447BC1E819504015E4C5F9   unchanged
frozen    C3-Redesign-2 FBE32851F795AF900E6C46D6E38F25138E01217C2B52C0BDDE1F015029CFC39E unchanged
```

## 4. The produced line 25, in full

```text
bp AudioEngineHarness+0x1f9c550 ".echo C3_DG_T0E_HIT ; .if (@r9 == 9) { .echo C3_DG_T0E_R9_PASS } ;
  .else { .echo C3_DG_T0E_R9_FAIL } ; .if (dd(@rcx+0x34000) == 4) { .echo C3_DG_T0E_EQ_PASS } ;
  .else { .echo C3_DG_T0E_EQ_FAIL } ; .if (@$t0 == 0) { .if (@r9 == 9) { .if (dd(@rcx+0x34000) == 4) {
  .echo C3_T0_ENTRY; r; dq @rsp L1; ln poi(@rsp); gc } ; .else { gc } } ; .else { gc } } ; .else { gc }"
```

Decomposition, by the unique-split method:

```text
payload  194 B, 5 command tokens
   '.echo C3_DG_T0E_HIT'
   '.if (@r9 == 9) { .echo C3_DG_T0E_R9_PASS }'
   '.else { .echo C3_DG_T0E_R9_FAIL }'
   '.if (dd(@rcx+0x34000) == 4) { .echo C3_DG_T0E_EQ_PASS }'
   '.else { .echo C3_DG_T0E_EQ_FAIL }'
guard   166 B, byte-identical to the frozen Redesign-4 guard
```

The payload asks the guard's own two questions and reports the answers instead of acting on them. The
constant `9` and the constant `4` are not introduced by the diagnostic; both already exist in the
frozen guard, and the payload restates each predicate **verbatim, exactly once**, verified by S8-03b
and S8-04b.

## 5. Static Validation S8-01 to S8-19

| ID | result | check | observed |
|---|---|---|---|
| S8-01 | PASS | only line 25 differs from Redesign-6 | changed = [25] |
| S8-02 | PASS | guard byte-identical to the frozen Redesign-4 guard | 166 B, unique split, whole-line exact reconstruction |
| S8-03 | PASS | GUARD region: `@r9 == 9` unchanged | R4 = 4, R9 = 4 |
| S8-04 | PASS | GUARD region: `dd(@rcx+0x34000) == 4` unchanged | R4 = 1, R9 = 1 |
| S8-05 | PASS | GUARD region: `@$t0 == 0` unchanged | R4 = 2, R9 = 2 |
| S8-03b | PASS | payload restates `@r9 == 9` verbatim, once | guard = 1, payload = 1 |
| S8-04b | PASS | payload restates `dd(@rcx+0x34000) == 4` verbatim, once | guard = 1, payload = 1 |
| S8-06 | PASS | breakpoint topology unchanged | 13 RVAs ordered identical |
| S8-07 | PASS | `$t0..$t19` untouched | payload register writes 0, guard writes 71 = 71, registers 0..19 |
| S8-08 | PASS | **zero `?` commands in the file** | file-wide `?` commands = 0 |
| S8-09 | PASS | every payload command is `.echo`, `.if` or `.else` carrying `.echo` | offending = none |
| S8-10 | PASS | only new tokens are `.echo` and five `C3_DG_T0E_*` markers | marker set as designed |
| S8-11 | PASS | T0P / T0Q / T0S byte-identical to Redesign-6 | lines 26, 27, 28 |
| S8-12 | PASS | all other 40 lines byte-identical to Redesign-6 | compared |
| S8-13 | PASS | ReaderSlot windows | 256 terms, 64 distinct offsets |
| S8-14 | PASS | A2 `dd @$t1 L1` | 7 occurrences |
| S8-15 | PASS | `gc` count unchanged | 61 = 61 |
| S8-16 | PASS | `q` count unchanged | 1 = 1 |
| S8-17 | PASS | payload and guard separated: every payload memory operand and numeric literal already exists in the guard | operands `['@rcx+0x34000']`, new addresses none, new constants none |
| S8-18 | PASS | source, harness, build, CMake, `ConvoPeq.md` unchanged | seven SHA-256 comparisons, git delta 12 pre-existing |
| S8-19 | PASS | execution count 0 | no debugger process, no CDB log for this gate |

**19 of 19 PASS.**

### 5.1 One validation-tooling defect, recorded

The first run reported S8-03 and S8-04 as FAIL, with `@r9 == 9` at 4 versus 5 and
`dd(@rcx+0x34000) == 4` at 1 versus 2. Both were **scoping errors in the validator**, not defects in
the artifact: the counts were taken over the whole body, and the Option C payload deliberately
restates each guard predicate once, so each total rises by exactly one. The guard region was already
proved byte-identical by S8-02. The checks were rescoped to the guard region, and S8-03b and S8-04b
were added to assert the stronger property that the payload's predicate text is identical to the
guard's.

This is the third instance of that family: the C1 semicolon check at Step5CX, the S7-09 `dd` check at
Step5DA, and now this. The pattern is constant, a check written against a convenient proxy rather
than against the property, and the defence is the same each time, check the property's own region.

## 6. What this gate does not certify

```text
does NOT certify that the Option C payload completes and reaches the guard's gc.
```

By the construct inventory, every construct it uses is already proven inside a body in this file:
`.echo` 117 times at the T0E position and 464 across four positions, and `.if` / `.else` 464 times,
including the guard's own two- and three-level chain evaluated at this exact address. So Redesign-9
carries **no unproven construct**, which is a stronger position than any payload before it in this
lineage.

That is an evidence statement, not a guarantee. The run is still required.

## 7. Frozen items, unchanged

```text
S7_READER / S7_READER_SLOT / minReaderEpoch(T1) / Case A / B / C / D = UNRESOLVED
A1 separator  = RUNTIME PROVEN
A2            = installed, accepted, execution count 0, globalEpoch value UNPROVEN
T0 gate cause = UNRESOLVED
IMPLEMENTATION = FORBIDDEN
```

## 8. Final state

```text
Option selection + Implementation + Static Validation
= CLOSED / OPTION C SELECTED / STATIC VALIDATION PASS 19 of 19

execution candidate
  doc/work113/P1-5-IR-P2_Step5DD_…_Redesign-9_Conjunct-Verdicts.cdb
  D48665B8E8B4DF780B6B174640EA869FFF6E2FF0F951D1799C1589809F5DAD14
  14,887 bytes

CDB execution / harness execution / build = 0
RUNTIME AUTHORIZATION = NOT GRANTED
```

## 9. Runtime Authorization requested, on the terms already fixed

```text
tmp\cdb.exe -cf <Redesign-9> -logo <log> build\Release\AudioEngineHarness.exe --measurement=normal
```

Primary criterion, unchanged in form from the Redesign-7 authorization:

```text
R1  SCRIPT_BEGIN = 1, SCRIPT_END = 1, ZwTerminateProcess = 1, exit = 0, zero debugger errors
    Continuation must hold with the Option C payload in place.
```

Then, only if R1 passes:

```text
R2  C3_DG_T0E_R9_PASS / _FAIL    conjunct 2, '@r9 == 9'
R3  C3_DG_T0E_EQ_PASS / _FAIL    conjunct 3, 'enqueuePos == 4'
R4  the per-hit pairing of the two verdicts, by ordinal position in the marker stream
R5  the four-way discrimination, evaluated per hit and not inferred
R6  the T0P / T0Q / T0S marker counts and their strict interleaving, which is the
    correspondence evidence, and is checked rather than assumed
```

conjunct 1, `@$t0 == 0`, is not re-measured. It was established from the Redesign-6 run: the script
initialises `$t0` to 0, the sole writer of `$t0` is `C3_T0_SEQUENCE_PUBLISHED`, and that marker fired
zero times.

No constant is proposed and no guard is changed. The raw values remain unobtained and are carried
forward as an open item.
