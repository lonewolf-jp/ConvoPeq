# C3 Entry-Gate Diagnostic — Redesign-6 Design Gate (read-only)

## 1. Gate

```text
Gate     = P3-5-FPM-C3-T1-Capture-Redesign-6_EntryGate-Diagnostic-DESIGN
mode     = read-only design and static soundness review
.cdb edited        = 0
CDB launched       = 0
harness launched   = 0
build              = 0
production source  = untouched
concrete command text for Redesign-6 = NOT fixed here, by instruction
VERDICT = DESIGN COMPLETE / IMPLEMENTATION NOT AUTHORIZED
```

## 2. State this design starts from

```text
Redesign-5 execution = CLOSED
R1  PARTIAL, preamble aborted with a CDB Extra character error
R2  0x1f9c550 reached                       PROVEN, 1 hit
R3-R5  0x1f9c57d / 59a / 5da reachability   NOT DETERMINED
R6  one partial operand sample              $t0=0, r9=1, r10=rbx=0x000000b4`44efe510,
                                             rcx=0x0000025c`6d434580, r11=0x00007ff9`ae8be717,
                                             dd(rcx+0x34000)=221
R7  conjunct verdict markers                NOT EMITTED
R9  C3_T0_ENTRY                             0
R10 T1                                      not reached, $t0 never became 1

BRANCH = B at sample level only.
         T0 gate cause = UNRESOLVED OVER TIME.  Not a final cause.

A1 = RUNTIME PROVEN
A2 = installed, CDB accepted, execution count 0, globalEpoch value UNPROVEN
S7_READER / S7_READER_SLOT / minReaderEpoch(T1) / Case A-D = UNRESOLVED
IMPLEMENTATION = FORBIDDEN
```

## 3. Item A, register-write syntax, measured rather than assumed

The previous audit asserted that existing bodies use the compact form and the preamble the spaced
form. That assertion was made with an unreliable excision heuristic, and a re-measurement using the
**unique-split-point** method gave different intermediate numbers. The corrected figures:

```text
preamble region :  spaced 'r $tN = EXPR'  = 12     compact 'r $tN=EXPR' = 0
guard region    :  spaced                   =  0     compact              = 20
top-level lines :  spaced                   = 20    (one command per line)
```

The nine unchanged breakpoints report no split point, which is the correct result for a body that has
no preamble.

Only **two** distinct register-write tokens are unique to the preamble:

```text
r $t16 = @r9
r $t17 = @$t17
```

So the corrected statement is: the spaced form occurs only in the Redesign-5 preambles, never in a
guard body, and also at top level where each command owns its line. That is a **correlation**. It
is not offered as a cause, for the reason in section 5.

## 4. Item B, the preamble as executed

Command inventory of the T0E body, in source order, against the log:

| # | command | observed |
|---|---|---|
| 1 | `r $t17 = @$t17 + 1` | inferred executed, the `<= 8` branch was taken |
| 2 | `.if (@$t17 <= 8) {` | executed, branch taken |
| 3 | `r $t16 = @r9` | not separately observable |
| 4 | `.echo C3_DG_T0E_HIT` | **executed**, log line 175 |
| 5 | `? @$t0` | executed, line 176, value 0 |
| 6 | `? @r9` | executed, line 177, value 1 |
| 7 | `? @r10` | executed, line 178 |
| 8 | `? @rbx` | executed, line 179 |
| 9 | `? @rcx` | executed, line 180 |
| 10 | `? @r11` | executed, line 181 |
| 11 | `? dd(@rcx+0x34000)` | executed, line 182, value 221 |
| 12 | `? dd(@rcx+0x30000)` | **did not execute** |
| 13 | the three verdict `.if` / `.else` pairs | **did not execute** |

The execution boundary is between 11 and 12. Both are `?`. No construct class changes there. The
boundary is simply the eighth consecutive read command.

## 5. Item C, the parser boundary, from the documentation

Microsoft documents the relevant rule in the `z` command page:

```text
"In many debugger commands, the semicolon is used to separate unrelated commands."
"Such commands can be any debugger commands that permit a terminal semicolon."
```

And for `?`:

```text
? Expression
  Expression  Specifies the expression to evaluate.
  (a single expression; no terminal-semicolon capability is documented)
```

So the documented mechanism class for an `Extra character error` is: **a command that does not
permit a terminal semicolon absorbs the `;` and the text after it, and the debugger then reports
trailing text it cannot use.** Composition with `;` is a per-command capability, not a global
guarantee.

The honest limit of what this settles:

```text
SETTLED   `;` composition requires every participating command to permit a terminal semicolon,
          and that capability is documented per command, not globally
NOTED     the `?` page documents one Expression and does not document terminal-semicolon
          tolerance, while our run shows seven `?` commands each terminating correctly,
          so absence from the page is not evidence of absence in the parser
UNKNOWN   which command in the preamble lacks the capability
```

The error's own report cannot identify it. The caret sits at column 102 and the quoted span is
`r $t16 = @r9 ; .echo C3_DG_T0E_HIT`, which names commands 3 and 4, **and command 4 demonstrably
executed**. A report that names a command which ran cannot be used to locate the failure. This
audit therefore makes no claim about the mechanism, and specifically does not claim that the
spacing is responsible.

### 5.1 The real design gap, stated construct-agnostically

The gap is not a token. It is that the preamble was composed of **thirteen `;`-joined commands whose
individual terminal-semicolon tolerance was never verified**, while the design treated `;` as a
uniform separator. Section 3 shows the same assumption produced a body whose register writes are
the only ones in the file written in a form no guard uses. Both facts point at the composition
assumption, not at any one command.

## 6. Principle D13

```text
D13  Diagnostic observation must not be allowed to prevent the mandatory
     continuation path.
```

Stated precisely, because the loose form is not achievable:

In a CDB breakpoint body there is no exception handling. `.if` / `.else` selects on a value; it does
not trap a parse or command failure, and a parse failure occurs before any command executes. The
mandatory `gc` sits at the end of the existing guard. Therefore **any** diagnostic placed ahead of
the guard can, if it fails, prevent the `gc`, and the observed consequence is exactly what
Redesign-5 produced: the body aborts, the debuggee stays stopped, the script's trailing lines are
consumed, and the harness is killed while stopped.

So D13 cannot be discharged by in-body structure. It can only be discharged by **removing the
possibility of failure**, which means every command in the diagnostic must be individually verified
on this build, at this position, before it is admitted.

## 7. Principle D14, the enforceable form of D13

```text
D14  No diagnostic payload may be introduced until a strictly smaller diagnostic has been
     proven, by one authorized execution, to run to completion and still reach the guard's
     terminal gc.

     Proof is per construct class and per position. One authorization, one construct class.
```

D14 converts an unachievable absolute into an incremental discipline, and it makes the next
execution a **continuation-safety probe** rather than another attempt at full capture.

## 8. Redesign-6 content, derived from evidence

Applying D14 to the evidence, the diagnostic set whose terminal-semicolon tolerance has been
**directly observed at that position in that body** is:

```text
.echo <marker>          proven, C3_DG_T0E_HIT fired at log line 175
one following ' ; '     proven, the guard's own internal ' ; ' separators are what A1
                        validated at runtime across 85 transitions
```

Nothing else qualifies. `?` has run seven times but its eighth use is where the body stopped, so its
tolerance is **not** established. The spaced register writes have never been observed to take
effect, because the value of `$t16` and `$t17` was never read back. `.if` verdict pairs inside the
preamble never executed.

Therefore the Redesign-6 diagnostic, as a **constraint on the Implementation gate**, is:

```text
C1  the diagnostic is a single command per T0 body
C2  that command is '.echo <reachability marker>'
C3  it writes no pseudo-register
C4  it reads no memory and forms no address
C5  it introduces no constant, no comparison and no inference
C6  it is followed by exactly one ' ; ' and then the guard, byte-identical
```

The exact marker spellings are left to the Implementation gate, subject to C1 through C6.

### 8.1 What Redesign-6 deliberately gives up, and why that is correct

```text
gives up   operand value capture      ? is not tolerance-verified
gives up   per-conjunct verdicts      the .if/.else pairs never executed in a preamble
gives up   @r9 transition tracking    needs register writes
gives up   the $t16 / $t17 counters   needs register writes
```

Each is a real loss of diagnostic power. All four share one dependency, register writes or
repeated reads inside a body, and that dependency is exactly what has not been shown to survive.
Restoring them is Stage 2 and later, each behind its own authorization, per D14.

C3 additionally collapses the register-budget question entirely: with no pseudo-register written,
`$t0..$t19` are untouched by the diagnostic, so the Step5CS budget argument is not merely satisfied
but made vacuous.

## 9. Reachability is still fully answerable by Redesign-6

This is the point of doing the minimal probe first. With one `.echo` per site:

```text
C3_DG_T0E_HIT present or absent   answers reachability of 0x1f9c550
C3_DG_T0P_HIT                    answers reachability of 0x1f9c57d
C3_DG_T0Q_HIT                    answers reachability of 0x1f9c59a
C3_DG_T0S_HIT                    answers reachability of 0x1f9c5da
ZwTerminateProcess reached        proves the mandatory gc ran, which is D13 and D6-11
```

R2 through R5, which were NOT DETERMINED in Redesign-5 solely because the run died, are therefore
answerable by a single marker per site. Only R6 and R7 wait for a later stage.

## 10. Static validation checklist D6-1 to D6-14, with decidability

The owner identified D6-10 and D6-11 as the decisive difference from Redesign-5. Both are marked
with how far static checking can actually reach, because overstating them would repeat the fault
this workstream keeps correcting.

| ID | check | mechanically decidable | method |
|---|---|---|---|
| D6-1 | the four T0 breakpoint addresses unchanged | **yes** | 13 RVA tokens compared in order |
| D6-2 | `$t0..$t19` roles unchanged | **yes** | zero `$tN=` writes in any diagnostic; the 20 guard writes byte-identical |
| D6-3 | T1 preconditions unchanged | **yes** | the nine non-T0 bodies byte-identical |
| D6-4 | `@r9 == 9` unchanged | **yes** | count in the guard region identical |
| D6-5 | enqueuePos `== 4` unchanged | **yes** | count in the guard region identical |
| D6-6 | `@rcx` / `@r11` site-specific base selection unchanged | **yes, but vacuously** | the diagnostic forms no address at all under C4 |
| D6-7 | ReaderSlot windows unchanged | **yes** | 256 terms, 64 distinct offsets |
| D6-8 | A2 `dd @$t1 L1` unchanged | **yes** | 7 occurrences |
| D6-9 | existing guard byte identity | **yes** | unique-split-point reconstruction, the method already used and independently corroborated |
| D6-10 | terminal `gc` not lost to diagnostic failure | **partly** | static: guard `gc` count unchanged, and the diagnostic is one command followed by one `;`. **The runtime half is not statically decidable.** |
| D6-11 | diagnostic failure does not strand the debuggee | **no** | not statically decidable at all. Sole criterion: the authorized execution reaches `ZwTerminateProcess` |
| D6-12 | no new constants or inference introduced | **yes** | the only new tokens in the file are `.echo` and four marker names |
| D6-13 | source, harness, build, CMake unchanged | **yes** | SHA-256 of each |
| D6-14 | execution count 0 | **yes** | no debugger process launched during the gate |

Stated plainly: **D6-10 is half static and D6-11 is entirely runtime.** No static check can prove
that a body will reach its `gc`, because the failure that destroyed Redesign-5 was invisible to
every static check in the Step5CT set, all 32 of which passed. The redesign therefore defines
continuation as the *acceptance criterion of the next execution*, not as a property to be certified
in advance. Any report that claims D6-11 statically decidable is wrong.

## 11. Residual structural fragility, recorded not fixed

```text
the script ends  ... ; g ; .echo SCRIPT_END ; q
if any body aborts, CDB consumes the remaining lines and quits,
and the harness is terminated while stopped.  That is what happened.
```

Redesign-6 does not change this. The trailing `q` is frozen, the topology is frozen, and the
alternative, removing `q`, would change the artifact's exit behaviour and is out of scope. The
mitigation is therefore entirely on the diagnostic side: make the diagnostic non-fallible rather
than make the script tail forgiving. This remains a known residual risk, stated here so it is not
rediscovered as a surprise.

## 12. Rejected, recorded so they are not revisited

```text
REJECTED  conclude that the spaced assignment form caused the abort
          the owner's instruction, and independently the caret names a command that
          demonstrably ran.  Section 3 shows a correlation only.

REJECTED  conclude that `?` is the culprit
          seven `?` commands ran; the eighth did not; no construct boundary explains it.

REJECTED  keep the full Redesign-5 diagnostic and merely reorder it
          reordering does not reduce the number of unverified commands, and D14 requires
          strict reduction first.

REJECTED  move the diagnostic into the guard's success branch, next to the gc
          it would put the diagnostic on the continuation path itself, and it would break
          guard byte identity, which D6-9 requires.

REJECTED  add a fifteenth breakpoint or a canary to host the diagnostic
          topology is frozen at 13, and a canary was already declined at Step5CT.

REJECTED  wrap the diagnostic in .if to contain a failure
          `.if` selects on a value and cannot trap a parse or command failure.

REJECTED  strip the trailing `q` so an abort is survivable
          it changes artifact exit behaviour and is outside the authorized scope.

REJECTED  change `@r9 == 9` or enqueuePos `== 4`
          one sample cannot distinguish a mis-specified constant from a window not yet
          reached.  Forbidden and unproved either way.

REJECTED  treat Branch B as the final cause
          it is a sample-level observation.  The cause is unresolved over time.
```

## 13. The next execution, once Redesign-6 passes static validation

Runtime Authorization is requested separately, after Implementation and Static Validation.

```text
R1   script completes and the harness reaches ZwTerminateProcess   <- primary, tests D6-11
R2   C3_DG_T0E_HIT
R3   C3_DG_T0P_HIT
R4   C3_DG_T0Q_HIT
R5   C3_DG_T0S_HIT
R6   operand values                          expected absent in Stage 1, not a failure
R7   conjunct verdict markers                expected absent in Stage 1, not a failure
R8   R9CHANGE                                expected absent in Stage 1
R9   C3_T0_ENTRY
R10  C3_T0_SEQUENCE_PUBLISHED
R11  T1A / T1B onward                        interpreted ONLY if R9 or R10 fired
R12  minReaderEpoch                          conditional on R11
R13  ReaderSlot                              conditional on R11
R14  CAS_PRE                                 conditional on R11
R15  CAS_POST                                conditional on R11
```

R11 onward are conditional and will not be interpreted unconditionally. R6, R7 and R8 are expected
to be empty in Stage 1 by design, and their absence must not be read as a defect.

## 14. Branch discipline carried forward

```text
Branch B (sample level)     holds for the Redesign-5 sample only
T0 gate cause               UNRESOLVED OVER TIME
@r9 observed once as 1      not evidence that 9 is wrong
enqueuePos observed once as 221   not evidence that 4 is wrong, and 221 is far past the
                                  pinned window, which is itself unexplained
```

No constant, no guard and no inference is changed on the basis of one sample. The only thing this
design changes is the failure-isolation discipline.

## 15. What this design does not claim

```text
does not claim  the Extra character error mechanism is known
does not claim  which construct is at fault
does not claim  the Redesign-5 operand sample is representative
does not claim  the T0 entry condition is mis-specified
does not claim  the T0 entry condition is correctly specified
does not claim  the three later sites are reached or unreached
does not claim  anything about S7_READER, S7_READER_SLOT, minReaderEpoch(T1) or Case A-D
does not claim  A2 has produced a globalEpoch value
does not fix a concrete command string, by instruction
```

`IMPLEMENTATION` remains `FORBIDDEN`. The frozen items are untouched by a design.

## 16. Final state

```text
Redesign-6 Entry-Gate Diagnostic, read-only design
= CLOSED / DESIGN COMPLETE / NOTHING IMPLEMENTED / NO EXECUTION

.cdb edited   0      CDB launched  0      harness launched 0      build 0
production source, test, CMake, build, Harness   untouched
ConvoPeq.md  read-only, SHA unchanged

new this gate
  D13 stated, and shown to be unachievable as an absolute inside one body
  D14 stated, the enforceable incremental form
  the documented rule that `;` composition requires per-command terminal-semicolon tolerance
  corrected A/B measurement, using the unique-split method rather than a heuristic
  the execution boundary identified, 11 to 12, with no construct class change across it
  constraints C1..C6 fixed, and the diagnostic payload reduced to the tolerance-verified minimum
  D6-1..D6-14 mapped, with D6-10 half static and D6-11 entirely runtime, stated as such

unchanged
  @r9 == 9, enqueuePos == 4, all guards, topology, T1 correlation, ReaderSlot windows, A2
  Branch B sample-level only; T0 gate cause unresolved over time
  S7_READER / S7_READER_SLOT / minReaderEpoch(T1) / Case A-D = UNRESOLVED
  IMPLEMENTATION = FORBIDDEN
```
