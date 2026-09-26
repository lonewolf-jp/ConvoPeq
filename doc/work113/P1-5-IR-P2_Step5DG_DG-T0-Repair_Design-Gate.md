# DG-T0-Repair — Design Gate, read-only

```text
gate                = DG-T0-Repair-1
mode                = READ-ONLY DESIGN.  No .cdb created, no CDB executed, no source touched.
purpose             = adjudicate dwo vs poi, fix the repair scope, define the new frozen baseline
inputs              = Step5DF Result Audit (the dd() finding), src/DeferredDeletionQueue.h,
                      src/CMakeLists.txt, build/CMakeCache.txt, Microsoft Learn debugger docs,
                      the 13 archived .cdb artifacts and 16 archived CDB logs
execution this gate = 0
verdict             = ADJUDICATED, with one item that overturns the proposed scope
```

## 0. Two corrections to the instruction, before adjudication

### 0.1 `ConvoPeq(4).md` does not exist

The instruction states the verification referenced `ConvoPeq(4).md`. No such file exists, anywhere in
the tree.

```text
C:\VSC_Project\ConvoPeq\ConvoPeq.md      5,535,334 B   SHA-256 E5E74200F12784FDF37BE24864F5E4B1…
ConvoPeq(4).md                          NOT FOUND
```

A recursive search for `ConvoPeq*.md` returns 88 files, none of them `ConvoPeq(4).md`. This matches
the standing project rule that `ConvoPeq.md` is the sole source authority. All source citations in
this gate are against `ConvoPeq.md` and the live `src/` tree, and both were read.

The substantive claim in the instruction, that `enqueuePos` is `std::atomic<uint32_t> enqueuePos{0};`,
is **independently confirmed** in section 1. The claim survives; only the filename was wrong.

### 0.2 DG-T0-Repair-3 is not one open item, it is two, and one of them is already closed

The instruction records `dwo`'s `.if` syntax as unproven. The archived corpus contains an execution
of `dwo` inside a `.if` condition, and the **class of its error** settles the syntax question. See
section 3.

## 1. DG-T0-Repair-1 — `enqueuePos` actual type is `uint32_t`

**PASS.**

```text
src/DeferredDeletionQueue.h:267
    alignas(64) std::atomic<uint32_t> enqueuePos{0};
ConvoPeq.md:62184
    alignas(64) std::atomic<uint32_t> enqueuePos{0};
src/DeferredDeletionQueue.h:268
    alignas(64) std::atomic<uint32_t> dequeuePos{0};
ConvoPeq.md:62185
    alignas(64) std::atomic<uint32_t> dequeuePos{0};
```

Both the concatenated authority and the live source agree, and both neighbours are `uint32_t`.
Usage sites read it as `uint32_t` throughout: `DeferredDeletionQueue.h:78` `uint32_t pos =
convo::consumeAtomic(enqueuePos, …)`, and `:86` `convo::compareExchangeAtomic(enqueuePos, …)`.

### 1.1 The decisive extra fact the instruction did not carry: `alignas(64)`

This qualifier is not incidental. It fixes the width question that the whole `dwo` versus `poi`
decision turns on.

```text
enqueuePos occupies bytes [0x34000, 0x34004)      4 bytes
tail padding                                        60 bytes, up to the next 64-byte boundary
dequeuePos occupies bytes [0x34040, 0x34044)
```

Therefore, for the frozen guard's operand:

```text
dwo(@rcx+0x34000)   reads [0x34000, 0x34004)   = exactly enqueuePos.           width-exact
poi(@rcx+0x34000)   reads [0x34000, 0x34008)   = enqueuePos + 4 padding bytes. width-inexact
```

`poi` returns a 64-bit value whose upper 32 bits are **padding bytes**, which are indeterminate in
C++ and not written by any program logic. `poi(@rcx+0x34000) == 4` is therefore only true when
those padding bytes happen to be zero.

This is a stronger argument for `dwo` than width semantics alone. It is not "dwo reads the right
size". It is:

```text
dwo depends only on the declared object.
poi depends on the declared object AND on 4 bytes of padding that no code writes and no
language rule fixes.
```

Retry-1's `poi` conjunct did evaluate true, so the padding was zero on that occasion. That is a
lucky observation, not a guarantee, and it is the reason `poi` should not be inherited as a
baseline even though it is the only form with a successful execution on record.

### 1.2 Cross-check: the 4 bytes at `+0x34000` do hold the counter

Runtime, not declaration. Redesign-7 executed `? dd(@rcx+0x34000)` and the debugger printed:

```text
Evaluate expression: 221 = 00000000`000000dd
```

The low dword at `+0x34000` is `0x000000dd` = 221, and the dword above it is zero. So the counter
occupies exactly the 4 bytes `dwo` would read, and the padding above it happened to be zero on that
run, which is why `poi` also worked. Declaration and measurement agree.

## 2. DG-T0-Repair-2 — `dwo` is a 32-bit read and is semantically consistent

**PASS as a design intent. Not certified as executable.**

Microsoft documents the MASM unary operators for reading memory:

| operator | meaning |
|---|---|
| `by` | Low-order byte from the specified address |
| `wo` | Low-order word from the specified address |
| `dwo` | **Double-word from the specified address** |
| `qwo` | Quad-word from the specified address |
| `poi` | Pointer-sized data from the specified address, 32 or 64 bits per the target |

`dwo` is the exact-width operator for a `uint32_t` field. The predicate
`.if (dwo(@rcx+0x34000) == 4)` is therefore semantically aligned with
`std::atomic<uint32_t> enqueuePos`.

The documented caveat, stated so it is not lost: `std::atomic<uint32_t>` is not statically asserted
to be 4 bytes anywhere in this header. The assertions present are
`std::atomic<size_t>::is_always_lock_free` and `std::atomic<uint64_t>::is_always_lock_free`
(`DeferredDeletionQueue.h:45,47`). The 4-byte width is established instead by the declaration plus
the runtime read in section 1.2, which is sufficient for a 4-byte read at a known offset.

## 3. DG-T0-Repair-3 — `dwo` in a `.if` condition: syntax CLOSED, successful read OPEN

The instruction records this as uniformly unproven. The corpus splits it.

### 3.1 `dwo` in a `.if` condition is syntactically accepted — PROVEN

`Step5BK` set three breakpoints whose bodies contain `.if (dwo(@$t1) == 0x100)`. The `bl` listing
accepted all three, the process ran, and `ping+0x39d9` **did** fire. The body then produced:

```text
Memory access error at ') == 0x100) { .echo PASS_A_POS; … .if (dwo(@$t1) != 0x100) { …'
```

The distinction is the whole point:

```text
Syntax error        = parse-stage failure.  The token sequence is not a legal condition.
Memory access error = evaluation-stage failure.  The condition PARSED, and the read of the
                      address then failed.
```

`dwo(@$t1)` reached the evaluation stage, so `dwo` parsed as a legal operand of a `.if` condition.
Contrast the `dd` case, which died at the parse stage with the caret on the call parenthesis and
never reached evaluation at all.

The cause of the memory failure is identifiable and is a defect in that probe, not in `dwo`. The
preamble line `r $t1 = ping!+0x3c` itself failed with `Syntax error at 'ping!+0x3c'` (5 preamble
assignments failed the same way), so `$t1` was never initialised and held an unusable address.

```text
DG-T0-Repair-3a   'dwo' is a syntactically accepted .if condition operand        PROVEN
```

### 3.2 `dwo` returning a correct value from a valid address — UNPROVEN

```text
DG-T0-Repair-3b   'dwo' successfully reading a valid address inside .if        UNPROVEN, OPEN
```

The only execution of `dwo` in the corpus used an address that the preamble had failed to set. No
`PASS_A_POS` or `FAIL_A_POS` was ever emitted as a whole line, so no `dwo` comparison ever completed.
3b cannot be closed without an execution, and this gate performs none.

### 3.3 `dd` in a `.if` condition is a parse-stage failure — PROVEN

```text
.if (@r9 == 9)                executed, emitted a verdict      register operand parses
.if (dd(@rcx+0x34000) == 4)   Syntax error at '(' after 'dd'   display command does not parse
.if (dwo(@$t1) == 0x100)       Memory access error              operator parses, read failed
```

Three-way contrast, all measured. `dd` is the only one rejected at parse.

## 4. DG-T0-Repair-4 — `poi` is measured on this site

**PASS as a historical fact, and simultaneously the reason not to choose it.**

```text
Step5BB Retry-1, RVA 0x1F9C550, same 64-bit binary, module base 00007ff7`57af0000,
rip 00007ff759a8c550 = RVA 0x1F9C550, r9 = 0000000000000009

guard text, recovered from that run's own bp echo:
    .if (@$t5 == 0) { .if (@$t0 <= 1) { .if (@r9 == 9) { .if (poi(@rcx+0x34000) == 4) { .echo C3_T0_ENTRY; …

C3_T0_ENTRY emitted.  The gate opened.
```

`poi` is therefore the only form with a **successful** execution on this site. It is also the only
one that reads 4 bytes of indeterminate padding, per section 1.1. The two facts are in tension and
the tension is the decision.

## 5. DG-T0-Repair-5 — repair only `0x1f9c550`: **NOT SUFFICIENT**

The instruction recommends repairing `0x1f9c550` alone. Measured against the artifact, that does not
achieve the stated goal of a valid single execution, and the reason is structural.

The four T0 sites are consecutive points on one enqueue CAS bracket, and **every one of them carries
the malformed `dd(` as its innermost conjunct**:

| site | RVA | innermost conjunct chain | `dd(` in a `.if` |
|---|---|---|---|
| T0E | `0x1f9c550` | `@$t0 == 0` → `@r9 == 9` → `dd(@rcx+0x34000) == 4` | 1 |
| T0P | `0x1f9c57d` | `@r9 == 9` → `@r10 == 4` → `dd(@r11+0x34000) == 4` → `dd(@r11+0x30010) == 4` | 2 |
| T0Q | `0x1f9c59a` | `@r9 == 9` → `@r10 == 5` → `dd(@r11+0x34000) == 5` → `dd(@r11+0x30010) == 4` | 2 |
| T0S | `0x1f9c5da` | `@$t0 == 0` → `@r9 == 9` → `@r10 == 5` → `@rbx == 4` → 2 × `dd(` | 2 |

If T0E is repaired and its gate opens, the guard's terminal `gc` resumes the debuggee, which
proceeds to T0P. At T0P the outer conjunct is `@r9 == 9`, and `r9` is **invariant across all four T0
sites within a cycle**, proven from disassembly: no instruction in `0x1f9c4e0..0x1f9c5f0` has `r9`
as a destination operand. T0E's gate required `@r9 == 9`, so it holds at T0P too. The next conjunct
is `@r10 == 4`, the compare-exchange expected value. If that holds, T0P reaches its `dd(` and
aborts.

```text
repair T0E only
  -> C3_T0_ENTRY is emitted
  -> gc resumes the debuggee
  -> T0P is hit, @r9 == 9 holds, @r10 == 4 is the CAS expected value
  -> T0P reaches .if (dd(@r11+0x34000) == 4)
  -> Syntax error, body aborts, debuggee stranded
  -> ZwTerminateProcess not reached
  -> R1 FAILS AGAIN
```

So Option A does not fail loudly at T0E. It converts a silent no-op into a **stranding abort one
site later**, and R1 still fails. It also spends the single authorized execution to learn something
already derivable statically.

### 5.1 The minimum coherent repair set

```text
RECOMMENDED   A'   T0E + T0P + T0Q + T0S
             RVA   0x1f9c550, 0x1f9c57d, 0x1f9c59a, 0x1f9c5da
             why   these four are one consecutive enqueue CAS bracket.  Repairing fewer than all
                   four leaves a malformed innermost conjunct on a path that the repaired sites
                   themselves cause to be taken.  Repairing them together is what makes the T0
                   path free of the defect end to end.
```

This preserves the attribution goal the instruction was protecting. It does not mix in the T1 phase.

### 5.2 Residual risk that must be stated, not hidden

Two sites remain malformed after A' and belong to the T1 phase:

| site | RVA | defect |
|---|---|---|
| T1 reclaim return | `0x1f9ce04` | 2 × `dd(` inside `.if` conditions |
| T1 candidate | `0x1f9cfb0` | 2 × `dd(` inside display-command address expressions |

`0x1f9cfb0` is gated on `@$t0 == 1` and sets `r $t12=1`, which is the gate for `0x1f9ce04`. So if
T0 entry is reached and `0x1f9cfb0` is subsequently hit, `0x1f9ce04` becomes reachable and would
abort in turn.

```text
RESIDUAL RISK   whether 0x1f9cfb0 and 0x1f9ce04 are reached in the same run is NOT KNOWN.
                It is not determinable statically, because reachability depends on runtime
                control flow that has never been observed with the T0 gate open.
                A' therefore carries a stated risk of a second stranding abort, at a
                site belonging to a different phase.
MITIGATION      the T0-entry evidence produced by A' bounds it.  If C3_T0_ENTRY is emitted and
                0x1f9cfb0 is never hit, the residual risk did not materialise and is retired
                for this run.  That determination belongs to the Result Audit, not to this gate.
```

Choosing A over A' does not avoid this risk; it only guarantees an R1 failure before it can be
evaluated.

## 6. DG-T0-Repair-6 — `0x1f9cfb0` is a separate item

**PASS. Separation is correct, and the corpus shows what the repair should look like.**

The two occurrences at `0x1f9cfb0` are not `.if` conditions. They are a display command whose
**address argument** is computed from a value read inline:

```text
dd @$t2+0x30000+((dd(@$t2+0x34040)&0xfff)*4) L1
dq (@$t2+0x00+((dd(@$t2+0x34040)&0xfff)*0x30)) L6
```

So the grammar situation is genuinely different from the T0 sites. The documented rule that a
condition must be an expression rather than a command does not directly cover an address argument,
and the call is being used to obtain a value, which is the same underlying misuse.

The corpus already contains the intended shape. `Step5BK` used precisely this pattern with `dwo`:

```text
.if (dwo(@$t1) == 0x100) { dd @$t2+((dwo(@$t1)&0xfff)*4) L1; dq @$t3+((dwo(@$t1)&0xfff)*0x30) L6; … }
```

`dwo(@$t1)&0xfff` used as an index inside a `dd` address, which is the direct analogue of
`0x1f9cfb0`'s `dd(@$t2+0x34040)&0xfff`. So the probable intent of the original author is visible.

```text
SEPARATE ITEM   yes
CANDIDATE FORM  dd(@$t2+0x34040)  ->  dwo(@$t2+0x34040)
STATUS          NOT VERIFIED.  The analogue in Step5BK never completed, because $t1 was invalid.
                Whether dwo is accepted as a value-producing call inside a display-command
                ADDRESS is therefore still unproven, and is a separate question from 3b.
NOT IN SCOPE    for this gate's repair, per the instruction, and correctly so.
```

## 7. DG-T0-Repair-7 — a corrected guard as the new frozen baseline

**Judgment item. Recommendation: yes, and it must be an explicit supersession, not a silent edit.**

The instruction's premise is right and is confirmed by the Step5DF audit: the Redesign-4 guard
contains `dd(`, so it can no longer serve as a measurement baseline, because the baseline would
encode a construct that provably cannot execute.

Three properties of the replacement baseline must be fixed explicitly, because each one has already
been the subject of a defect in this lineage.

```text
BASELINE NAME          T0-Guard-Repair-Baseline-1
SUPERSEDES             the T0E guard of Redesign-4  (SHA A935369D…022C, 14,624 B)
STATUS OF OLD BASELINE  RETIRED as a measurement baseline.  Retained byte-identical on disk as
                       history.  It is not deleted and not edited.
```

### 7.1 The three T0E conjuncts, fixed

```text
conjunct 1   .if (@$t0 == 0)                  established TRUE, Redesign-6
conjunct 2   .if (@r9 == 9)                   operand-verdict, n=1 FALSE, n=1 TRUE on Retry-1
conjunct 3   .if (dwo(@rcx+0x34000) == 4)    width-exact, syntax proven, successful read OPEN
```

The operand base is **per site** and must be stated per site, because getting this wrong is a
recorded defect of this lineage. At `0x1f9c550` only `@rcx` is a proven queue base; `@r10`, `@r11`
and `@rbx` are still caller values at that point. At `0x1f9c57d`, `0x1f9c59a` and `0x1f9c5da` the
base is `@r11`.

### 7.2 Operand offsets, now source-confirmed rather than only empirically derived

The layout follows from the source, given the build configuration:

```text
kQueueSize                        = 4096
sizeof(DeletionEntry)              = 0x30   (CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS off)
ringBuffer   4096 x 0x30           = 0x30000
sequences    4096 x 4              = 0x04000
enqueuePos   alignas(64) uint32    = +0x34000
dequeuePos   alignas(64) uint32    = +0x34040
maxRetireAgeUs_                    = +0x34080
worldReclaimCount_                 = +0x340C0
referenceObserver_                 = +0x34100
```

This reproduces the previously proven offset algebra exactly, `sequences` at `+0x30000`,
`enqueuePos` at `+0x34000`, `dequeuePos` at `+0x34040`, which is an independent confirmation of
operand bases that had only been measured.

The one load-bearing precondition is the entry stride. `DeletionEntry` has a conditionally compiled
member:

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    size_t objectBytes{0};
#endif
```

so `sizeof` is `0x30` with diagnostics off and `0x38` with it on, which would move `sequences` to
`+0x38000`. The build that produced the authorized harness was checked, not assumed:

```text
CMakeLists.txt:129   option(CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS ... OFF)
build/CMakeCache.txt CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS:BOOL=OFF
harness SHA-256      E5C7AFB9C4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75   match
```

**The baseline is therefore valid only while that harness build is the target.** A rebuild with
diagnostics enabled would invalidate every offset in every guard. This dependency is not recorded
anywhere in the current lineage and should be attached to the baseline.

### 7.3 `+0x30010` is `sequences[4]`

```text
sequences is at +0x30000, elements are 4 bytes
+0x30010 = sequences + 0x10 = sequences[4]
```

So the T0 bracket is the position-4 enqueue window, and the four sites are consistent with one CAS:

```text
T0E   enqueuePos == 4   sequences[4] implied     entry
T0P   enqueuePos == 4   sequences[4] == 4        CAS pre
T0Q   enqueuePos == 5   sequences[4] == 4        CAS post, counter advanced, slot not yet published
T0S   enqueuePos == 5   sequences[4] == 5        write done, slot published
```

This is a static consistency observation about the frozen text. It is **not** a claim that any of
these conditions was ever observed to hold. None of them was, because none of them could be
evaluated.

## 8. Adjudication summary

| ID | item | verdict |
|---|---|---|
| DG-T0-Repair-1 | `enqueuePos` actual type is `uint32_t` | **PASS**, with the load-bearing `alignas(64)` recorded |
| DG-T0-Repair-2 | `dwo` is a 32-bit read, semantically consistent | **PASS** as design intent, not certified executable |
| DG-T0-Repair-3a | `dwo` is a syntactically accepted `.if` operand | **PASS, newly closed in this gate** |
| DG-T0-Repair-3b | `dwo` returns a correct value from a valid address | **OPEN**, unproven, no execution performed |
| DG-T0-Repair-4 | `poi` is measured on this site | **PASS** as fact, and the reason not to choose it |
| DG-T0-Repair-5 | repair `0x1f9c550` only | **NOT SUFFICIENT.** Minimum coherent set is A', the four T0 sites |
| DG-T0-Repair-6 | `0x1f9cfb0` is a separate problem | **PASS**, separation correct, repair form identified but unverified |
| DG-T0-Repair-7 | corrected guard becomes the new baseline | **Judgment.** Recommend yes, as an explicit supersession of the Redesign-4 T0E guard, with the build-configuration dependency attached |

### 8.1 Operator choice, adjudicated

```text
dwo   SELECTED as the first candidate, for two independent reasons.
      1. width-exact: reads [0x34000,0x34004) = exactly the declared uint32_t object
      2. dependency-free: unlike poi, its result does not depend on 60 bytes of tail padding
      Status: syntax proven (3a), successful read unproven (3b). 3b closes only by execution.

poi   REJECTED as a baseline, despite being the only form with a successful execution on record.
      Reason: its correctness depends on 4 bytes of padding that no code writes and no
      language rule fixes.  A baseline must not rest on an indeterminate value.
```

The instruction's recommendation of `dwo` is upheld, on a stronger ground than the one offered.

## 9. Not decided here, and deliberately so

```text
the exact byte layout of T0-Guard-Repair-Baseline-1     requires the implementation authorization
whether to use dwo or a proven-in-place fallback         requires 3b to be closed by one execution
the residual T1-phase risk                             belongs to the A' Result Audit
0x1f9cfb0 repair form                                  separate gate, per DG-T0-Repair-6
```

## 10. State

```text
CDB execution this gate   = 0
.cdb created this gate    = 0
.cdb artifacts             = 13, all SHA unchanged
source / test / CMake     = untouched, read only
build                     = not invoked
ConvoPeq.md               = read only, SHA E5E74200F12784FDF37BE24864F5E4B1… unchanged
harness                   = SHA E5C7AFB9C4EAA48C1ACB031689910F38… unchanged
IMPLEMENTATION            = FORBIDDEN
RUNTIME AUTHORIZATION     = NOT GRANTED
```

Forbidden items from the instruction, all still forbidden and none performed:

```text
Redesign-9 retry                       not performed, and not useful, the cause is fully determined
Redesign-10 runtime                    not performed
dwo-variant .cdb                       not created
poi-variant .cdb                       not created
source / test / CMake / build          untouched
T1 capture                             not started
ReaderSlot measurement                 not started
Case A / B / C / D                     not started
```

## 11. Next gate, on the terms already fixed

```text
DG-T0-Repair adjudicated
   |
   +-- owner rules on DG-T0-Repair-5:  A' (four T0 sites)  or  B (all six sites)
   |
   v
CDB Repair Implementation Authorization, scoped to the ruled set
   |
   v
.cdb creation, taking every guard from the retired baseline and replacing only the memory-read
operand, with the unique-split method
   |
   v
Static Validation
   |
   v
Runtime Authorization, requested separately
   |
   v
single execution, no retry
   |
   v
R1, then R2 to R6 only if R1 passes
   |
   v
T0 entry evidence, which also bounds the residual T1-phase risk
```

The single decision this gate cannot make for the owner is the scope: A' or B.
