# Redesign-8 Design Gate — obtaining {enqueuePos, r9} with at most one `?` per body

## 1. Purpose and status

```text
Purpose  obtain the pair {enqueuePos, r9} for the T0 entry condition while respecting the
         measured constraint of at most one '?' per breakpoint body
Status   READ-ONLY DESIGN
CDB execution = 0     harness execution = 0     build = 0
source / test / CMake = untouched
IMPLEMENTATION = FORBIDDEN
```

No winner is selected here. The three options are compared on the same six axes, and the
implementable contract is stated explicitly for each, leaving the choice to the owner.

## 2. The two measured constraints this design starts from

```text
C-a   one '?' as the first body command at 0x1f9c550   = tolerance-proven
        Redesign-7, log line 176: 'Evaluate expression: 221 = 00000000`000000dd'
C-b   one '?' as a second body command                 = NOT tolerance-proven
        Redesign-7, the EQ1 witness was absent, the body aborted, the debuggee was stranded
```

A third measured fact bears on every option and is recorded because it is easy to misread:

```text
C-c   the debugger's own error report does not localise the failure.  Twice it named a command
      that had demonstrably executed.  The witness markers did localise it, both times.
      Attribution must therefore come from markers, not from the caret.
```

## 3. Evidence base, established read-only in this gate

### 3.1 `r9` is never written inside the T0 region

Disassembly of `0x1f9c4e0 .. 0x1f9c5f0`, 80 instructions, classified by AT&T operand position:

```text
instructions mentioning r9 in any width          = 1
  0x1f9c5c8   movq %r9, 0x10(%r11,%rcx,8)          destination is memory, so this is a READ
instructions whose DESTINATION operand is r9     = 0
```

An earlier pass at this check reported "no write" from an empty match set, because the pattern used
`$r9` while llvm-objdump emits AT&T `%r9`. That was a false pass and it is recorded here; the
corrected result is the one above, and it is now based on a non-empty match set.

Consequence, and it is the load-bearing result for Option A:

```text
@r9 at T0P, at T0Q and at T0S equals @r9 at T0E, within the same enqueue cycle,
as a property of the emitted instructions, not an assumption about the harness.
```

### 3.2 The four T0 sites interleave strictly, measured

From the Redesign-6 run, the only run in which the guards executed:

```text
total marker lines                             = 464
strict T0E -> T0P -> T0Q -> T0S from index 0   = 116 complete cycles
per-site counts  T0E = 116   T0P = 116   T0Q = 116   T0S = 116
```

### 3.3 Construct inventory, by evidence grade

| construct | grade | n | basis |
|---|---|---|---|
| `.echo` | **PROVEN** | 117 at the T0E position, 464 across four positions | Redesign-6 and Redesign-7 |
| `.if` / `.else`, nested | **PROVEN** | 464 evaluations | guards reached their terminal else branch |
| `gc` | **PROVEN** | 464 executions | the guard else branch ran to completion |
| `? expr`, first body command | **PROVEN** | 1 completion | Redesign-7 line 176 |
| `? expr`, second body command | **UNSAFE** | 1 failure | Redesign-7, attributed by the EQ1 witness |
| `r $tN = ...` | NOT PROVEN | 0 | no completion ever observed, value never read back |
| `dd addr L1` as a command | NOT PROVEN | 0 | guard forms never executed, no guard ever opened |
| `dq`, `db` | NOT PROVEN | 0 | guard forms never executed |
| `ln poi(@rsp)` | NOT PROVEN | 0 | guard form, never executed |
| `.printf` | NEVER USED | 0 | absent from every script in this lineage |

`dd` appears in a guard as a command and in the Redesign-7 payload as the *operand* of `?`. Those are
different things. The `?` form completed; the `dd` command form has never run.

### 3.4 Cross-run reproducibility

```text
enqueuePos at the first T0E hit
    Redesign-5 run = 221
    Redesign-7 run = 221
    agreement      = identical, two independent runs

T0E hit count
    Redesign-6 = 116   (the only run that completed)
    Redesign-7 =   1   (aborted at the first hit)
    completed-run sample size = 1
```

## 4. Option A, obtain the second value at a different T0 site

### Correctness

`@r9` is provably invariant across the four T0 sites within a cycle, by 3.1. Reading `@r9` at T0P
therefore yields the value that `@r9` held at T0E for that same cycle. The claim does not rest on
the harness's behaviour, on register-allocation luck, or on the four sites happening to be adjacent
in time; it rests on the absence of any write to `r9` in the emitted code between them.

### Same-hit correspondence

This is the part the owner required to be proved rather than assumed, so it is stated with its
exact status.

```text
WHAT IS PROVEN
    the k-th T0E marker and the k-th T0P marker belong to the same enqueue cycle,
    PROVIDED the two sites fire equally often and in strict order.

WHAT WAS MEASURED
    Redesign-6: 116 and 116, strict T0E->T0P->T0Q->T0S, zero deviations.

WHAT IS NOT PROVEN, AND MUST NOT BE ASSUMED
    that the counts will always be equal.  The emitted code contains an exit path at
    0x1f9c5e0, reachable from T0E by the branch at 0x1f9c58f, on which T0E leaves the function
    and T0P is never reached.  Equal counts are therefore a property of the observed run,
    not of the control flow.
```

The design consequence is that correspondence must be **verified in the log, not assumed**. The
counting is cheap and already proven: the Stage-1 markers at all four sites are `.echo`, the proven
class, and they cost nothing. A run in which the T0E count and the T0P count differ, or in which the
interleaving deviates, is a run in which the pairing is void and says so.

### Construct safety

```text
T0E body :  '.echo' + one '?' for enqueuePos        = 1 '?'
T0P body :  '.echo' + one '?' for @r9               = 1 '?'
every other body unchanged, 0 '?'
```

Both are at the proven position, the first body command, after a single proven `.echo`. This is the
exact shape that completed in Redesign-7. Nothing new is admitted; `?` is already admitted and
already bounded at one per body by measurement.

### Pseudo-register impact

```text
0.  No 'r' command, no pseudo-register write.  Neither '?' writes debugger state.
    $t0..$t19 keep their existing meanings, and T0_SEQUENCE_PUBLISHED still performs the only
    assignment of the T1 address algebra.
```

### Topology impact

```text
0.  The 13 breakpoint sites are unchanged.  No site is added, removed or moved.
    This is a payload change inside two existing bodies, not a topology change.
```

### Continuation impact

```text
Structurally identical to the Redesign-6 shape that was proven 464 times: a single proven '.echo'
followed by one '?' followed by the byte-identical guard ending in 'gc'.
The residual risk is the single '?' per body, which is measured-safe, not zero-risk.
```

## 5. Option B, two authorized runs, one value each

### Correctness

Each run would carry one `?`, so each run satisfies the one-per-body constraint trivially. The
values obtained would be, respectively, the enqueuePos sequence and the `r9` sequence.

### Same-hit correspondence

This is where the option stands or falls, and the honest answer is that **no pairing key is
established**.

```text
candidate key            status
enqueuePos value         circular.  To pair on it, Run-2 would have to observe enqueuePos as
                         well, which needs a second '?' in one body.  That is the very thing
                         measured unsafe.
hit ordinal              n=1.  Only one completed run exists, Redesign-6 at 116 hits.  Whether
                         the count and the per-hit state reproduce is unmeasured.
T0 marker sequence       the Stage-1 markers give a per-cycle ordinal, and that ordinal is
                         reproducible only if the run reproduces, which is the previous row.
publication sequence     the harness prints '[PUBLISH] seq=3 gen=3' on its own console stream.
                         It is not captured by -logo, reproduced twice, and it is a harness-side
                         value not a queue identity.
new identity             not introduced.  The owner forbade inventing one, and none is proposed.
```

The one encouraging datum is that enqueuePos at the first hit was 221 in two independent runs. That
is n=2 on a single sample, and it is not a correspondence key for the remaining 115 hits.

### Construct safety

Best possible: one `?` per body per run, the proven shape.

### Pseudo-register, topology, continuation impact

All zero, identically to Option A.

### Assessment

Option B is the only option that does not require a correspondence proof, because it does not
attempt one. Its cost is that it yields two unpaired sequences and leaves the pairing question
open, and it consumes two authorizations to obtain strictly less per authorization than Option A.

## 6. Option C, re-compose constructs that are already proven

### Correctness, and the observation that reframes the objective

The discrimination the owner specified needs only two **booleans** per hit, not two raw values:

```text
r9 = 9   and enqueuePos = 4    ->  all three conjuncts hold
r9 = 9   and enqueuePos != 4   ->  the enqueuePos conjunct fails
r9 != 9   and enqueuePos = 4   ->  the r9 conjunct fails
r9 != 9   and enqueuePos != 4  ->  both fail
```

Both booleans are decidable with `.if` and `.echo`, and by the inventory in 3.3 **both are proven
constructs inside a body**: `.if`/`.else` nested was evaluated 464 times and `.echo` emitted 465
times at T0E. The guard bodies are themselves two and three levels of exactly this shape.

So Option C can deliver the discrimination with **zero `?` and zero unproven constructs**:

```text
.if (@r9 == 9)            { .echo C3_DG_T0E_R9_PASS }    ; .else { .echo C3_DG_T0E_R9_FAIL }
.if (dd(@rcx+0x34000)==4) { .echo C3_DG_T0E_EQ_PASS }    ; .else { .echo C3_DG_T0E_EQ_FAIL }
```

### What Option C cannot deliver

```text
it cannot report what @r9 IS, only whether it equals 9
it cannot report what enqueuePos IS, only whether it equals 4
it therefore cannot bound enqueuePos over the window, and cannot say whether 4 was ever
approached
```

That is a real loss. The Redesign-7 Result Audit already records `enqueuePos = 221` at one hit; Option
C cannot extend that to a range, and it cannot produce the raw `r9` value that would let a later
gate reason about what the token actually is.

### Construct safety

Highest of the three. Every construct used is already proven in this file at this position.

### Pseudo-register, topology, continuation impact

All zero. And the continuation risk is the lowest of the three, because there is no `?` at all to
fail.

### Assessment

Option C is the only option whose construct set is entirely proven, and the only one that yields
booleans rather than values. If the objective is strictly *which conjunct is false*, it is
sufficient. If the objective includes *what the values are*, it is not.

## 7. Side-by-side

| axis | Option A, second value at T0P | Option B, two runs | Option C, proven-only verdicts |
|---|---|---|---|
| `?` per body | 1 at T0E, 1 at T0P | 1 at T0E, per run | **0** |
| obtains enqueuePos value | yes | yes, run 1 | **no, boolean only** |
| obtains r9 value | yes | yes, run 2 | **no, boolean only** |
| same-hit correspondence | provable, and checkable in the log | **not established** | same hit, trivially |
| basis of the r9 claim | 0 writes to r9 in the emitted code | n/a | n/a, no value claimed |
| unproven constructs used | 0 | 0 | **0** |
| pseudo-registers touched | 0 | 0 | 0 |
| breakpoint topology | unchanged, 13 | unchanged, 13 | unchanged, 13 |
| continuation risk | one `?` per body, measured-safe | one `?` per body, measured-safe | **none from `?`** |
| authorizations consumed | 1 | 2 | 1 |
| sufficient for the four-way discrimination | yes | only if pairing is later established | yes |

## 8. Mandatory decision table

| item | required judgement | status in this design |
|---|---|---|
| `?` per body | at most 1 | A: 1 at two sites. B: 1. C: 0. All satisfy |
| two values obtained | enqueuePos and r9 | A: yes, both raw. B: yes, unpaired. C: booleans only |
| same-hit correspondence | provable | A: proved from the code, verified in the log. B: **not established**. C: trivial, same hit |
| `$t0..$t19` | existing semantics unchanged | 0 writes in all three |
| guard | frozen Redesign-4, unchanged | unchanged in all three |
| T0P / T0Q / T0S | no unnecessary change | A: T0P gains one `?`. B and C: unchanged |
| T1 | untouched | untouched in all three |
| ReaderSlot | untouched | untouched in all three |
| A2 `dd @$t1 L1` | untouched | untouched in all three |
| breakpoint topology | 13 RVA maintained | maintained in all three |
| `gc` | existing continuation unbroken | guard retained verbatim, ending in `gc`, in all three |
| new constant | 0 | 0. The values 9 and 4 already exist in the frozen guard; the diagnostic introduces neither |
| new address | in principle 0 | 0. `@rcx+0x34000` is the guard's own expression; `@r9` forms no address |
| production source | 0 | 0 |
| build | 0 | 0 |
| runtime execution | 0 | 0 |

## 9. What is held as hypothesis and is not assumed

```text
NOT assumed  that enqueuePos equals 4 at some hit
NOT assumed  that enqueuePos can never equal 4
              the only datum is enqueuePos = 221 at the FIRST hit of two runs.
              One sample, taken at the first hit, bounding nothing about the window.
NOT assumed  that @r9 differs from 9
              zero r9 samples exist.  Redesign-5 read @r9 = 1 at one hit, which is a single
              sample of an opaque token and is not a verdict.
NOT assumed  that the four T0 sites always fire equally often
              measured once, 116 of 116, with a code path that can break the equality.
NOT assumed  that the '?' failure mechanism is known
              it is not.  C-b is a bound, not a cause.
```

## 10. Rejected, recorded so they are not revisited

```text
REJECTED  two '?' in one body
          measured unsafe, Redesign-7, attributed by the EQ1 witness.

REJECTED  a compound '?' expression carrying both values
          fuses a new expression form with a command already at its bound, so a failure could
          not be attributed to either.

REJECTED  a new pseudo-register to carry a value across sites
          writes a pseudo-register, which is not a proven construct, and $t0..$t19 semantics
          are frozen by the decision table.

REJECTED  a new identity to pair the two runs of Option B
          forbidden by the owner, and no candidate key is established anyway.

REJECTED  treating the caret or the quoted error fragment as an attribution channel
          shown unreliable twice.

REJECTED  reading the Redesign-7 value 221 as 'enqueuePos != 4'
          forbidden by that Result Audit, and one sample cannot support it.

REJECTED  changing @r9 == 9 or enqueuePos == 4
          neither is proved correct or proved wrong.  Both are measured next, not edited.
```

## 11. Gate state

```text
.cdb edited = 0     CDB execution = 0     harness execution = 0     build = 0
production source / test / CMake = untouched
ConvoPeq.md = read-only, SHA unchanged
Redesign-2 / 4 / 5 / 6 / 7 = unchanged, SHAs recorded

Design Gate   = CLOSED / DESIGN COMPLETE
Execution count = 0
IMPLEMENTATION = FORBIDDEN

frozen
  @r9 == 9, enqueuePos == 4, all guards, topology, T1 correlation, ReaderSlot windows,
  A2 'dd @$t1 L1', no canary
  S7_READER / S7_READER_SLOT / minReaderEpoch(T1) / Case A-D = UNRESOLVED
  T0 gate cause = UNRESOLVED
```

## 12. Next gate

```text
owner selects among the three implementable contracts stated above
        |
        v
Implementation Authorization, scoped to the selected option only
        |
        v
.cdb creation, guard from the frozen Redesign-4 by the unique-split method
        |
        v
Static Validation
        |
        v
Runtime Authorization, requested separately
```

Nothing is requested here. No option is recommended in this document, no script is edited, and no
constant is changed.
