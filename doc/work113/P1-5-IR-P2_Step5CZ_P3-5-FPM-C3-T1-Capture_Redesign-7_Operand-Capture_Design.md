# Redesign-7 Design Gate — Stage 2, Operand Capture (read-only)

## 1. Objective

Identify, by measurement, which conjunct of the T0E entry guard is false, at a site now proven to
execute.

```text
site            0x1f9c550, hit 116 times in the Redesign-6 run
guard           .if (@$t0 == 0) { .if (@r9 == 9) { .if (dd(@rcx+0x34000) == 4) { ... } } }
conjuncts       1  @$t0 == 0
                2  @r9 == 9
                3  enqueuePos, i.e. dd(@rcx+0x34000), == 4
observed        the conjunction held 0 times in 116 hits
```

The objective is discrimination, not capture for its own sake. Nothing about slots, ReaderSlot,
`minReaderEpoch` or CAS state is in scope.

### 1.1 One conjunct is already eliminated, by the run's own evidence

This is a deduction from the Redesign-6 result, not an assumption and not a narrowing heuristic.

```text
the script sets  r $t0 = 0   once, before g
exactly one breakpoint body writes $t0, and it writes 1:
    line 28, 0x1f9c5da, C3_T0_SEQUENCE_PUBLISHED
in the Redesign-6 run, C3_T0_SEQUENCE_PUBLISHED fired 0 times
```

Therefore `$t0` held `0` at every one of the 116 T0E hits, and **conjunct 1 held every time**. It is
eliminated by the previous run, not by argument about the target.

That leaves conjuncts 2 and 3 as the only candidates, and it is why the payload below observes two
values rather than three.

## 2. Single admitted construct class

```text
RETAINED, already admitted and runtime-proven at this position:
    .echo <marker>            464 emissions in the Redesign-6 run, 0 errors

NEWLY ADMITTED, exactly one class:
    ? <expression>            the evaluate-and-display command
```

Nothing else is admitted. In particular this design does **not** introduce, even in combination:

```text
r $tN = ...                  register write
r <reg>                      register display
dd / db / dq                 memory display
.printf                      formatted output
.if / .else / .elsif         conditional execution
z / j                        loop or conditional command
```

`.echo` appears in the payload, and that is not a mixing of classes: it is the Stage 1 class, already
proven, being reused as a witness. Its markers follow the naming convention the Design Gate fixed,
`C3_DG_T0E_*`.

## 3. Exact observation payload

Inserted after the opening quote of the `0x1f9c550` body, ahead of the guard, which is not touched:

```text
.echo C3_DG_T0E_HIT ; ? dd(@rcx+0x34000) ; .echo C3_DG_T0E_EQ1 ; ? @r9 ; .echo C3_DG_T0E_EQ2 ; <existing guard, byte-identical>
```

Resulting body, for the record:

```text
bp AudioEngineHarness+0x1f9c550 ".echo C3_DG_T0E_HIT ; ? dd(@rcx+0x34000) ; .echo C3_DG_T0E_EQ1 ;
  ? @r9 ; .echo C3_DG_T0E_EQ2 ; .if (@$t0 == 0) { .if (@r9 == 9) { .if (dd(@rcx+0x34000) == 4) {
  .echo C3_T0_ENTRY; r; dq @rsp L1; ln poi(@rsp); gc } ; .else { gc } } ; .else { gc } } ; .else { gc }"
```

Command inventory per hit, seven commands, of which two are new:

```text
1  .echo C3_DG_T0E_HIT      proven class   existing witness
2  ? dd(@rcx+0x34000)       NEW class      conjunct 3 value, the guard's own expression
3  .echo C3_DG_T0E_EQ1      proven class   witness that command 2 completed
4  ? @r9                    NEW class      conjunct 2 value
5  .echo C3_DG_T0E_EQ2      proven class   witness that command 4 completed
6  <guard>                                 byte-identical to the frozen guard
```

### 3.1 Why exactly two `?` and not one or three

```text
0   insufficient, conjunct 1 is already eliminated by section 1.1
1   can answer only one branch.  Observing enqueuePos alone: if it ever reads 4 then conjunct 3
    held at that hit and conjunct 2 is the failing one, which is a complete answer.  If it never
    reads 4, conjunct 3 is falsified as reachable and conjunct 2 remains unknown, which is not a
    complete answer.
2   gives both surviving conjunct values at every hit, so all three can be evaluated per hit by
    hand from the log, with no inference.  This is the minimum that guarantees the stated objective.
3   would observe the already-eliminated conjunct 1.  Redundant.
```

The failure-surface cost of the second `?` is real and is argued in section 5, not dismissed.

### 3.2 No new address, no new constant

The enqueuePos observation is `dd(@rcx+0x34000)`, which is **the guard's own address expression,
copied verbatim**. Consequences:

```text
no new address is formed
no new constant is introduced
no new base is chosen; @rcx is the base the guard already uses
the address selection is therefore correct by construction, not by a fresh judgement
```

`@rcx` as the queue base at `0x1f9c550` is already corroborated twice: the Step5CS section 4.2
disassembly correction, and the Redesign-5 runtime sample, in which `dd(@rcx+0x34000)` returned 221, a
plausible enqueue position, while `@r11` at the same instant was a caller value in a different
module. `@r9` is a bare register read and forms no address at all.

## 4. Why this construct class is admitted under D14

```text
D14   no diagnostic payload may be introduced until a strictly smaller diagnostic has been
      proven, by one authorized execution, to run to completion and still reach the guard's gc.
      Proof is per construct class and per position.
```

The class being admitted is `?`, at position `0x1f9c550`, in front of a guard whose `gc` has now
been observed to be reached 464 times. D14 is satisfied for the precondition: the strictly smaller
diagnostic, the single `.echo`, was proven in the Redesign-6 run, R1 PASS, D6-10 runtime PROVEN,
D6-11 PROVEN.

D14 is **not** satisfied for `?` itself, and this design does not pretend otherwise. `?` is admitted
as the next class precisely so that its tolerance can be measured, once, under its own authorization.
The empirical bound available is that seven consecutive `?` commands executed correctly in the
Redesign-5 run before the body stopped, so two instances sit well inside the region observed to work.
That is an empirical bound, not a guarantee, and the design is built so that a failure is
attributable rather than ambiguous.

## 5. Continuation safety argument

### 5.1 The honest statement

If either new `?` fails, the body aborts before the guard, the terminal `gc` is not reached, the
debuggee is stranded, CDB consumes the remaining script lines and quits, and the harness is killed
while stopped. That is the Redesign-5 failure mode, and it is the expected cost of admitting a
construct whose tolerance is unproven.

This cannot be engineered away while the guard is frozen. `.if`/`.else` cannot trap a failure, the
guard's `gc` is inside the guard, and no in-body structure exists that survives an aborted command.
D13 is not satisfiable as an absolute; it is satisfiable only as a bound plus attribution.

### 5.2 The bound

```text
new unproven commands per hit = 2
proven commands per hit       = 3 echoes, unchanged from Stage 1
total commands per hit        = 5, plus the byte-identical guard
comparison, Redesign-5        = 13 commands, of which 12 were new and unproven
comparison, Redesign-6        = 1 command, proven
```

The unproven surface is reduced from twelve constructs to two.

### 5.3 The attribution, which is the part that matters

Redesign-5's failure was not attributable: the debugger named `r $t16 = @r9 ; .echo C3_DG_T0E_HIT`,
and the `.echo` in that span had demonstrably executed, so the report could not locate the failure.
That ambiguity cost a full redesign cycle.

This payload removes that ambiguity using only the already-proven class. The `.echo` witnesses
bracket each new command:

```text
C3_DG_T0E_HIT present, C3_DG_T0E_EQ1 absent   the first  ?  failed
C3_DG_T0E_EQ1  present, C3_DG_T0E_EQ2 absent   the second ?  failed
both witnesses present                         both ?  completed
no HIT at all                                  the site was not reached this run
```

So whichever way the run goes, the outcome is attributable:

```text
both witnesses present  -> the ? class is tolerance-proven at this position, and the two operand
                           values are in hand for all 116 hits
first witness missing   -> ? is not tolerance-safe as the first body command; the witness design
                           worked, and the next class must be chosen accordingly
second witness missing  -> ? is safe once but not twice; the bound is between one and two
no HIT                  -> reachability itself changed, which would be a different and important
                           result
```

That is the property D7-4 asks for, and it is bought with the proven class rather than with a
structural guarantee that does not exist.

### 5.4 Log volume

```text
per hit   3 marker lines + 2 evaluation lines = 5
116 hits  580 lines, against 464 in the Redesign-6 run
```

No bound is placed on the number of hits, because bounding it would discard the very data the run
exists to collect. The 116-hit window is the entire measurement window and all of it is wanted.

## 6. Register and pseudo-register budget

```text
pseudo-registers written by the diagnostic = 0
pseudo-registers read by the diagnostic    = 0    ? @r9 is a machine register, not a pseudo-register
$t0 .. $t19                                = untouched
register set                               = $t0..$t19, unchanged, no register added
```

`?` is a read-only evaluation and `dd` is a read-only display. Neither writes a debugger
pseudo-register and neither perturbs the debuggee's architectural state. The Step5CS budget argument
is therefore not merely satisfied but made vacuous, exactly as in Stage 1.

## 7. Nothing beyond scope is added

```text
memory reads introduced     = 1, at the guard's own address expression
addresses introduced        = 0
constants introduced        = 0
comparisons introduced      = 0
inferences introduced       = 0
slot sequences              = not observed
ReaderSlot windows          = untouched, 256 terms
minReaderEpoch              = not observed
CAS state                   = not observed
T1 correlation              = untouched
A2 'dd @$t1 L1'             = untouched
```

## 8. Frozen guard proof

The guard is not edited, and the property is mechanically checkable rather than asserted.

```text
method    the unique index j such that body[j:j+3] == " ; " and body[j+3:] equals the frozen
          guard, with uniqueness required
reference the guard is taken from the frozen Redesign-4 body, not from the parent, so identity
          is transitive back to the artifact that has never had a preamble
claim     new_line == prefix + payload + " ; " + frozen_guard + suffix, exactly
sites     0x1f9c550 only
```

The same method proved guard identity at Step5CO, Step5CT and Step5CX, and it caught a real defect at
Step5CT when a weaker first-match heuristic was used. It is used again unchanged.

## 9. Scope: T0E only

```text
0x1f9c550  receives the payload
0x1f9c57d  unchanged, keeps .echo C3_DG_T0P_HIT only
0x1f9c59a  unchanged, keeps .echo C3_DG_T0Q_HIT only
0x1f9c5da  unchanged, keeps .echo C3_DG_T0S_HIT only
the nine non-T0 bodies  unchanged
```

Rationale, and it is a boundary argument rather than a convenience: the objective is to discriminate
the conjuncts of **one** guard, and all three surviving conjuncts are visible at `0x1f9c550`. The
other three sites have different conjunct sets and different site-specific bases, per the Step5CS
section 4.2 table, and admitting them now would multiply the unproven surface for no gain on the
stated question. They keep the Stage 1 markers, so their reachability remains observable at no risk.

## 10. Static acceptance criteria

| ID | criterion | decidable |
|---|---|---|
| S7-01 | only line 25 differs from the frozen Redesign-6 | yes |
| S7-02 | the guard is byte-identical to the frozen Redesign-4 guard, by unique-split reconstruction | yes |
| S7-03 | `@r9 == 9` count in the guard unchanged | yes |
| S7-04 | `dd(@rcx+0x34000) == 4` count in the guard unchanged | yes |
| S7-05 | `@$t0 == 0` count in the guard unchanged | yes |
| S7-06 | breakpoint topology unchanged, 13 RVAs ordered identical | yes |
| S7-07 | `$t0..$t19` untouched, zero register writes in the payload | yes |
| S7-08 | exactly two `?` commands in the whole file's payloads | yes |
| S7-09 | zero `r`, `dd`, `db`, `dq`, `.printf`, `.if`, `.else` introduced into any payload | yes |
| S7-10 | the only new tokens are `.echo` and the witness marker names, following the fixed convention | yes |
| S7-11 | the T0P, T0Q, T0S bodies are byte-identical to Redesign-6 | yes |
| S7-12 | the nine non-T0 bodies are byte-identical to Redesign-6 | yes |
| S7-13 | ReaderSlot windows still 256 terms, 64 distinct offsets | yes |
| S7-14 | A2 `dd @$t1 L1` still 7 occurrences | yes |
| S7-15 | `gc` count unchanged | yes |
| S7-16 | `q` count unchanged | yes |
| S7-17 | no new constant and no new address expression: the only memory operand is `@rcx+0x34000`, already present in the guard | yes |
| S7-18 | source, harness, build, CMake, `ConvoPeq.md` unchanged | yes |
| S7-19 | execution count 0 | yes |

## 11. Runtime acceptance criteria

```text
primary    R1  SCRIPT_BEGIN 1, SCRIPT_END 1, ZwTerminateProcess 1, exit 0,
               zero errors.  Continuation must still hold with the two new commands present.
               This is D6-10 and D6-11 re-proved under the enlarged payload.

R2         C3_DG_T0E_HIT  count, expected 116, not required to equal 116
R3         C3_DG_T0E_EQ1   count, must equal the EQ1-eligible hits
R4         C3_DG_T0E_EQ2   count
R5         the enqueuePos values, one per hit where EQ1 is present
R6         the @r9 values, one per hit where EQ2 is present
R7         attribution, per section 5.3, from which witnesses are present

discrimination, evaluated per hit from R5 and R6, not by inference:
   conjunct 2 held  <=>  the observed @r9 equals 9
   conjunct 3 held  <=>  the observed enqueuePos equals 4
   conjunct 1 held  <=>  already established by section 1.1, not re-measured

outcomes, all of which are acceptable and none of which is a defect:
   both conjuncts observed false at every hit   -> both conjuncts are unsatisfiable as pinned
   conjunct 3 true at some hit, conjunct 2 false -> conjunct 2 is the sole failing conjunct
   conjunct 2 true at some hit, conjunct 3 false -> conjunct 3 is the sole failing conjunct
   both true at some hit                        -> the guard should have opened, and the guard
                                                   body itself becomes the object of study
```

Stage 1 absences that remain expected: no slot sequence, no ReaderSlot observation, no
`minReaderEpoch`, no CAS observation, nothing from `0x1f9c57d`, `0x1f9c59a` or `0x1f9c5da` beyond
their reachability markers.

## 12. Rejected alternatives

```text
REJECTED  observe all three conjuncts including $t0
          conjunct 1 is eliminated by section 1.1.  A third `?` would spend unproven surface on
          a value already known.

REJECTED  observe only one of the two surviving conjuncts
          guarantees only one of the two answer branches.  Section 3.1.

REJECTED  use `r @r9` instead of `? @r9`
          `r` with a display argument is a different command class, unproven at this position, and
          admitting two new classes in one authorization is what D7-1 forbids.

REJECTED  use a compound expression such as `? (@r9 == 9) + (dd(@rcx+0x34000) == 4) * 2`
          a bitmask in one command is tempting, but it fuses a new expression form with a new
          command class, so a failure could not be attributed to either.  It also encodes an
          inference in the debugger, which section 7 excludes.

REJECTED  put the observation inside the guard's success branch
          the guard does not open, so nothing would be observed, and it would break guard byte
          identity.

REJECTED  move the observation after the guard
          the guard ends in `gc`, which transfers control; nothing after it is reliably executed.

REJECTED  add a fifth or sixth command for volume control
          Redesign-6 showed 464 emissions at no cost.  Volume is not a problem to be solved here.

REJECTED  extend to T0P, T0Q and T0S simultaneously
          different conjunct sets, different site-specific bases, and it multiplies the unproven
          surface for no gain on the stated question.  Section 9.

REJECTED  drop the `.echo` witnesses to keep the payload shorter
          that is precisely the Redesign-5 mistake, an unattributable failure.  The witnesses are
          the cheapest insurance available and they cost only proven-class commands.

REJECTED  conclude from the Redesign-5 sample that enqueuePos == 4 is unreachable
          one sample, and the counter observation is exactly what tests it.  Held as hypothesis.

REJECTED  change `@r9 == 9` or `dd(@rcx+0x34000) == 4`
          neither is proved correct or proved wrong.  Section 13.
```

## 13. What this design holds as hypothesis, not conclusion

The Redesign-6 audit offered a narrowing argument: a conjunct that pins a monotonically advancing
counter would be expected to hold in some cycles if the counter traverses the pinned value during the
window. Redesign-5's single sample read 221 at the first observed hit, which is already far past 4.

**This is retained strictly as a design hypothesis. It is not a determination.** Specifically, this
design does not conclude any of the following, and Stage 2 exists to measure them:

```text
that conjunct 3, enqueuePos == 4, is false
that conjunct 2, @r9 == 9, is false
that conjunct 1, $t0 == 0, is false        it is eliminated, positively, by section 1.1
that the pinned value 4 is wrong
that the pinned value 9 is wrong
```

The only conjunct-level statement this design makes is the positive one in section 1.1, and it rests
on the observed non-firing of the sole writer of `$t0`, not on any assumption about the target.

## 14. Gate state

```text
.cdb edited                = 0
CDB launched               = 0
harness launched           = 0
build                      = 0
production source / test / CMake = untouched
ConvoPeq.md                = read-only, SHA unchanged
Redesign-4 / 5 / 6         = unchanged, SHAs recorded

Design Gate                = CLOSED / DESIGN COMPLETE
Execution count            = 0
IMPLEMENTATION             = FORBIDDEN

unchanged and frozen
  @r9 == 9, enqueuePos == 4, all guards, breakpoint topology, T1 correlation,
  ReaderSlot windows, A2 'dd @$t1 L1', no canary
  S7_READER / S7_READER_SLOT / minReaderEpoch(T1) / Case A-D = UNRESOLVED
  T0 gate cause = UNRESOLVED, now bounded to 0 of 116 cycles
```

## 15. Next gate, not requested here

```text
Implementation Authorization
        |
        v
.cdb creation, payload inserted at 0x1f9c550 only
        |
        v
Static Validation S7-01 .. S7-19
        |
        v
Runtime Authorization, requested separately
        |
        v
one execution, one construct class
        |
        v
Result Audit
```

Nothing is requested here. No diagnostic is added, no guard is touched, and no constant is changed on
the strength of the Redesign-6 run.
