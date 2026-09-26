# r9 Value Capture — Design Gate (read-only)

```text
gate                = r9 Value Capture Design Gate
mode                = READ-ONLY DESIGN.  No .cdb created.  No CDB executed.
purpose             = determine what @r9 means at 0x1f9c550, and design a method that
                      captures its RAW VALUE without re-admitting the '?' failure mode
inputs              = src/DeferredDeletionQueue.h, the harness disassembly, and the archived
                      logs of Step5BB Retry-1, Step5DE, Step5DK
execution this gate = 0
verdict             = ADJUDICATED.  r9 identified.  Capture method selected.  Awaiting owner
                      ruling on one optional extension, then a separate Implementation
                      Authorization.
```

## 0. What this gate explicitly does not do

```text
does NOT re-run or re-verify the T0 dd( -> dwo( repair.  That is proven three ways:
  parse PASS, evaluation PASS, comparison TRUE at 4 of 116 hits.  No reason to re-measure.
does NOT authorise or design any T1 repair.  0x1f9ce04 and 0x1f9cfb0 remain malformed and
  were not reached.  This run says nothing about whether they would abort.
does NOT create a .cdb, and does NOT execute anything.
does NOT generalise any single r9 observation into an invariant.
```

## 1. What `@r9` is: `epoch`, the 4th argument of `DeferredDeletionQueue::enqueue`

### 1.1 The function at `0x1f9c550` is the enqueue path, in full

```text
141f9c550  movq  %rbx, 0x8(%rsp)
141f9c555  movl  0x34000(%rcx), %r10d        ; r10d = enqueuePos
141f9c55c  movq  %rcx, %r11                  ; r11 = this
141f9c560  movl  %r10d, %ebx                 ; pos
141f9c563  andl  $0xfff, %ebx                ; idx = pos & 0xFFF
141f9c569  movl  0x30000(%r11,%rbx,4), %eax  ; eax = sequences[idx]
141f9c571  subl  %r10d, %eax                 ; eax = sequences[idx] - pos
141f9c574  jne   0x141f9c58d
141f9c576  leal  0x1(%r10), %ecx             ; expected = pos + 1
141f9c57a  movl  %r10d, %eax                 ; <-- T0P
141f9c57d  lock
141f9c57e  cmpxchgl %ecx, 0x34000(%r11)      ; CAS enqueuePos
141f9c586  je   0x141f9c59a                  ; <-- T0Q
141f9c588  movl  %eax, %r10d
141f9c58b  jmp  0x141f9c560
141f9c58d  testl %eax, %eax
141f9c58f  js   0x141f9c5e0                  ; full -> exit
141f9c591  movl  0x34000(%r11), %r10d
141f9c598  jmp  0x141f9c560
141f9c59a  movzbl 0x28(%rsp), %eax            ; stack arg 4  -> type
141f9c59f  leaq  (%rbx,%rbx,2), %rcx
141f9c5a3  addq  %rcx, %rcx                  ; rcx = idx * 6
141f9c5a6  movb  %al, 0x18(%r11,%rcx,8)       ; entry.type
141f9c5ab  movq  0x30(%rsp), %rax            ; stack arg 5  -> publicationSequenceId
141f9c5b0  movq  %rax, 0x20(%r11,%rcx,8)
141f9c5b5  movq  0x38(%rsp), %rax            ; stack arg 6  -> generation
141f9c5ba  movq  %rax, 0x28(%r11,%rcx,8)
141f9c5bf  movq  %rdx, (%r11,%rcx,8)         ; entry.ptr        <- arg 1
141f9c5c3  movq  %r8,  0x8(%r11,%rcx,8)      ; entry.deleter    <- arg 2
141f9c5c8  movq  %r9,  0x10(%r11,%rcx,8)     ; entry.epoch      <- arg 3   ONLY r9 use
141f9c5cd  incl  %r10d
141f9c5d0  movb  $0x1, %al                   ; return true
141f9c5d2  movl  %r10d, 0x30000(%r11,%rbx,4) ; sequences[idx] = pos + 1
141f9c5da  movq  0x8(%rsp), %rbx              ; <-- T0S
141f9c5df  retq
141f9c5e0  movq  0x8(%rsp), %rbx              ; exit path, can skip T0P/Q/S
141f9c5e5  xorb  %al, %al
141f9c5e7  retq
```

### 1.2 The argument mapping is proved by two independent agreements

```text
Windows x64:  rcx = this, rdx = arg1, r8 = arg2, r9 = arg3,
              0x28(%rsp) = arg4, 0x30(%rsp) = arg5, 0x38(%rsp) = arg6

source, DeferredDeletionQueue.h:
  bool enqueue(void* ptr, void (*deleter)(void*), uint64_t epoch,
               DeletionEntryType type, uint64_t publicationSequenceId, uint64_t generation)
```

| register / slot | argument | source field | stored to |
|---|---|---|---|
| `rdx` | arg1 | `ptr` | `entry + 0x00` |
| `r8` | arg2 | `deleter` | `entry + 0x08` |
| **`r9`** | **arg3** | **`epoch`** | **`entry + 0x10`** |
| `0x28(%rsp)` | arg4 | `type` | `entry + 0x18` |
| `0x30(%rsp)` | arg5 | `publicationSequenceId` | `entry + 0x20` |
| `0x38(%rsp)` | arg6 | `generation` | `entry + 0x28` |

Both the register order and the store offsets match the source declaration order exactly, and the
entry stride is `idx * 0x30`, which is `sizeof(DeletionEntry)`. The mapping is not inferred from a
single field; it is confirmed by all six arguments and all six store offsets.

```text
PROVEN   @r9 at 0x1f9c550 is the 'epoch' argument of DeferredDeletionQueue::enqueue
PROVEN   @r9 is the 4th register argument, i.e. the 3rd declared parameter
```

### 1.3 `r9` is never written anywhere in the function

```text
mentions of r9 in 0x1f9c550 .. 0x1f9c620   = 1
   141f9c5c8  movq %r9, 0x10(%r11,%rcx,8)     the store to entry.epoch
writes to r9 in 0x1f9c550 .. 0x1f9c5e7        = 0
```

So `@r9` at the T0E breakpoint is the caller's incoming value, and it is **invariant across all four
T0 sites** for a given call, not merely within a cycle. That is a slightly stronger statement than
the earlier record, which bounded it to `0x1f9c4e0..0x1f9c5f0`; the tighter bound follows from
there being exactly one mention in the whole function.

The preceding function at `0x1f9c420` is a **different** function, the dequeue and reclaim path
operating on `dequeuePos` at `+0x34040`. It also never mentions `r9`. That does not extend
invariance across the two functions, because they have different callers.

### 1.4 Therefore the frozen guard's second conjunct is a test on `epoch`

```text
.if (@r9 == 9)      means      epoch == 9
```

## 2. What the three observations are, kept separate

| run | r9 read | epoch at enqueue | note |
|---|---|---|---|
| Redesign-5 | `1` | 1 | one sample, `? dd(...)` in the body, run did not complete |
| Step5BB Retry-1 | `9` | 9 | one sample, gate opened, `--fpm-m0` |
| Step5DK | not 9 | not 9 | **116 of 116** hits, `--measurement=normal` |

```text
NOT generalised.  Three runs, three different circumstances, three different values.
epoch is a small incrementing world counter, not a constant.  Step5DK's own debuggee output
carries [PUBLISH] seq=3 gen=3 worldId=3 and worldReclaimCount=2, so epochs in this harness
are small integers that advance during the run.
```

The harness's own stdout in the Step5DK run is the direct evidence that `epoch` is a moving
quantity:

```text
[PUBLISH] seq=3 gen=3 worldId=3 publishDurationUs=13970
[normal] worldReclaimCount=2 ... A_start=3 R_start=2 A_end=11 R_end=10 ... worldReclaimCount_end=10
```

### 2.1 A hypothesis this raises, stated as a hypothesis and not as a conclusion

At Retry-1, the log line immediately after the entry was

```text
00000176`47379758  00000000`00000009
```

which is the output of the script's own `dq @$t1+0x18 L1`, and `$t1` is the impl `this`. So at the
one moment the gate opened, the live field at `impl + 0x18` was also `9`, the same value as `r9`.

```text
HYPOTHESIS   the constant 9 was authored by reading the live global epoch at capture time and
             freezing it, when the intended predicate was a COMPARISON against the live global
             epoch rather than against a literal.

SUPPORTING   at the only observation where the gate opened, r9 == 9 and impl+0x18 == 9
             simultaneously.

NOT PROVEN   it is not shown that 9 was authored that way, nor that the intended predicate was
             a comparison.  It is a hypothesis that the value-capture run can test or refute.
```

This matters for the design, because if the hypothesis is right then no value of `r9` will ever make
a literal `9` correct, and the correct fix is a different predicate, not a different constant. The
capture method should therefore be able to observe **both** sides of that comparison. That is
section 4.

## 3. Capture method, and the `?` constraint

### 3.1 The constraint, restated from measurement

```text
PROVEN, Redesign-7:  a '?' evaluates and prints its value, but NO COMMAND AFTER A COMPLETED '?'
                    EXECUTES.  The command that failed was '.echo C3_DG_T0E_EQ1', a proven-class
                    construct, not a second '?'.  The failure is a property of position after
                    a '?', not of the '?' construct.

CONSEQUENCE        '?' can only be the last command of a body.  A body whose last command is '?'
                    has no guard and therefore no terminal gc, so the debuggee is stranded.
                    '?' is unusable in any body that must continue.
```

So `? @r9` is excluded by construction, not by preference. It is the one command that would give
the value most directly, and it is exactly the one that cannot be used.

### 3.2 The construct that does work, and its proof

`r`, the bare register dump, was executed **inside a breakpoint body** at Step5BB Retry-1 and the
body continued past it:

```text
[292] C3_T0_ENTRY
[293] rax=0000000000000000 rbx=0000017644793940 rcx=000001764737ab80
[294] rdx=0000017655a01080 rsi=0000017655a01080 rdi=00007ff759a979c0
[295] rip=00007ff759a8c550 rsp=0000000fb80fecf8 rbp=0000000000000009
[296]  r8=00007ff759a979c0  r9=0000000000000009 r10=0000017644740000
[297] r11=0000000fb80fec10 r12=0000017647808340 r13=0000000000000000
[298] r14=0000000000000000 r15=0000000000000000
[299] iopl=0         nv up ei pl nz na po nc
[300] cs=0033  ss=002b  ds=002b  es=002b  fs=0053  gs=002b             efl=00000202
[303] $t0=0000000000000001        <- later commands in the same body
[307] $t4=0000000000000000
[308] 0000000f`b80fecf8  00007ff7`59a8c735        <- dq @rsp L1
[314] C3_T0_SLOTS_BEGIN                             <- the body went on to the slot dump
```

```text
PROVEN   'r' executes inside a breakpoint body.
PROVEN   commands AFTER 'r' execute.  Eight further commands ran, including a 'dq' and the
         start of a large 'db' block.
PROVEN   'r' yields r9's raw value: line 296 reads r9=0000000000000009.
PROVEN   'r' did not prevent continuation.  This is the exact property that '?' lacks, and the
         contrast between the two is the whole basis of this design.
```

### 3.3 Options considered

| option | mechanism | new construct | verdict |
|---|---|---|---|
| V1 | `r` in the T0E payload | `r`, class proven, **position new** | **SELECTED** |
| V2 | `r r9` to narrow the output | `r` with a register operand, unproven form | rejected, see 3.4 |
| V3 | `? @r9` | `?` | **EXCLUDED by the measured constraint**, 3.1 |
| V4 | `.printf "%x", @r9` | `.printf`, never used in this lineage | rejected, never used anywhere |
| V5 | `r $tN = @r9` then read `$tN` at a later site | `r $tN=` proven at payload position, but needs a guaranteed-later site | rejected, see 3.5 |
| V6 | `?? @r9` | `??`, unproven | rejected |
| V7 | an `.if` / `.else` decode tree over candidate epochs | none, all constructs proven at this position | viable fallback, see 3.6 |

### 3.4 Why V2 is rejected rather than preferred

`r r9` would cut the output from roughly 8 lines to 1 per hit, which is a real saving over 116 hits.
It is rejected because it is a *different form* of a proven command, and this lineage has already
been bitten twice by a construct being safe in one form and not another: `poi` worked where `dd` did
not, and `?` worked where the command after it did not. Adding an unproven narrowing in the same
body as the one that must not fail would trade a large cosmetic saving for a whole-run stranding
risk. If `r r9` were wanted it would need its own authorisation and its own single run, not a
free addition to this one.

### 3.5 Why V5 is rejected

Saving `r9` into a pseudo-register and reading it later needs a breakpoint that is *guaranteed* to
fire for the same hit. The only such candidate is the S7 terminal anchor at `0x1fa80b3`, and it is
gated on `@$t0 == 1`, which only the T0S entry branch sets. That branch requires conjunct 2, which
is exactly what is false. So the later read would never happen. The design would silently produce
no data.

### 3.6 V7, the zero-new-construct fallback

A nested `.if` / `.else` tree comparing `@r9` against successive literals and `.echo`-ing the match
would resolve the value using only constructs already proven at this exact position, at the cost of
roughly 4 tokens per candidate value. It is bounded: a value outside the enumerated set yields only
"not among these", never the value. It is recorded as the fallback if the owner declines to admit
any new construct instance.

## 4. The optional extension, and why it is worth its cost

Capturing `r9` alone answers "what is epoch at enqueue". It does **not** settle whether the constant
`9` is wrong, because a single number cannot distinguish "the right value, observed once" from "a
value that happens to have been observed once". The section 2.1 hypothesis is only testable if the
live global epoch is captured from the same hit.

```text
EXTENSION (optional, owner ruling required)   also read the live global epoch at the same hit.

At T0E, rcx is the DQueue base.  r11 is not yet valid: it is assigned by the third instruction
at 0x1f9c55c, which has not executed when the breakpoint at 0x1f9c550 fires.  So the operand
base must be rcx.

impl this   = rcx - 0x1440        (already proven in this lineage)
globalEpoch = impl this + 0x18    (Retry-1's own dq @$t1+0x18 L1 printed 9 at the open)
therefore   live global epoch address = rcx - 0x1428
```

```text
COST       this introduces an address that is in neither Baseline-1 nor the frozen guard.  The
           A' authorisation explicitly forbade new addresses, and this is a new authorisation,
           so the cost is a fresh decision rather than a precedent violation.

VALUE      it converts the run from "one number" into a comparison, which is what the four-way
           discrimination and the section 2.1 hypothesis both actually need.

VERDICT    RECOMMENDED, but separated from V1 so it can be accepted or declined independently.
```

A read of that address is a **live** value at the moment of the enqueue, which is the correct
semantics for the hypothesis. It is not a snapshot of the gate's own pseudo-register state, so it
introduces no circularity.

## 5. The design, as it would be authorised

```text
observation point        0x1f9c550, T0E, in the payload ahead of the frozen guard
exact CDB command        r
                         (plus, if the extension is accepted, one memory read of rcx-0x1428)
command grammar validity 'r' is a display-only WinDbg command; it assigns nothing.  Proven to
                         execute in a breakpoint body and to be followed by further commands,
                         Step5BB Retry-1, eight commands after it.
continuation safety      the body's terminal 'gc' is preserved.  The only construct added is
                         'r', whose class is proven mid-body.  Position, the T0E payload slot,
                         is new.  This is the single admitted risk, and it is the same shape of
                         risk that scope A' carried and that Step5DK then discharged.
value-output witness     r9 appears in the 'r' output as a whole register line, e.g.
                             r8=... r9=0000000000000009 r10=...
                         which is a distinct, greppable shape and cannot be confused with the
                         bp command echo or the bl listing, both of which the whole-line marker
                         rule already excludes elsewhere in this lineage.
boolean/value separation V1 yields a raw value.  It does not touch the guard's '.if (@r9 == 9)',
                         and the guard stays byte-identical, so the conjunct verdict and the
                         raw value are independent observations of the same hit.
new baseline             the product is NOT Baseline-1.  It supersedes it as Baseline-2, because
                         the T0E body changes.  Baseline-1 stays frozen and unmodified.
```

### 5.1 What the run would and would not settle

```text
WOULD settle   the distribution of 'epoch' at enqueue across the run, as raw values.
WOULD settle   whether epoch ever equals 9 in this configuration, and how often.
WOULD settle   the section 2.1 hypothesis, if the extension is accepted.
WOULD NOT       whether 9 is the correct expected value.  One configuration producing one set
               of values cannot establish what the predicate should be.  That needs intent,
               not more measurement.
WOULD NOT       anything about T1.  0x1f9ce04 and 0x1f9cfb0 stay malformed and unreached.
```

## 6. Bound on the deliverable

```text
observation point              specified, section 5
exact CDB command              specified, 'r'
command grammar validity       specified and evidenced, sections 3.1 and 3.2
continuation safety            specified, with the one admitted risk named
value-output witness           specified, section 5
boolean and value separated    specified, section 5
new baseline                   Baseline-2, Baseline-1 frozen
no source/test/CMake/build     none touched, this gate is read-only
no T1 repair                   none, section 0
single-run boundary            one execution, no retry, requested separately
```

## 7. State

```text
@r9 at 0x1f9c550        = the 'epoch' argument of DeferredDeletionQueue::enqueue   PROVEN
@r9 written in function  = never, 0 writes in 0x1f9c550..0x1f9c5e7                 PROVEN
'r' in a bp body         = executes, continues, yields r9                          PROVEN
'?' in a bp body         = executes, then nothing after it executes               PROVEN, excluded
capture method           = V1, bare 'r'
fallback                 = V7, zero-new-construct '.if' decode tree
optional extension       = live global epoch at rcx-0x1428                          owner ruling
T0 dd( -> dwo( repair    = not repeated, already proven
T1 repair                = NOT AUTHORIZED, not designed here
r9 raw value             = UNKNOWN until the capture run
CDB execution this gate  = 0
.cdb created this gate   = 0
IMPLEMENTATION           = FORBIDDEN
RUNTIME                  = NOT AUTHORIZED
```

## 8. The one decision this gate cannot make

```text
accept or decline the section 4 extension, the live global epoch read at rcx-0x1428

  accept    the run yields a comparison rather than a single number, and the section 2.1
            hypothesis becomes testable.  Cost: one new address, which needs its own
            authorisation.
  decline   the run yields r9's raw distribution only.  Simpler, one fewer unproven element,
            and the question "is 9 correct" stays open after the run.
```

Everything else in this gate is settled and needs no further decision.
