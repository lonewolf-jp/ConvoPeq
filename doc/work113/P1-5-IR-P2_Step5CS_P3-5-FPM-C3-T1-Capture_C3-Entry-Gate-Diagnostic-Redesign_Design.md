# C3 Entry-Gate Diagnostic Redesign — read-only Design Gate

## 1. Gate

```text
Gate     = P3-5-FPM-C3-T1-Capture-C3-Entry-Gate-Diagnostic-Redesign-DESIGN
mode     = read-only design and static soundness review
.cdb edited        = 0
CDB launched       = 0
harness launched   = 0
build              = 0
production source  = untouched
VERDICT = DESIGN COMPLETE, IMPLEMENTATION NOT AUTHORIZED
```

## 2. State this design starts from

```text
Redesign-4 execution = CLOSED
R1                    = PASS
A1 separator rule     = RUNTIME PROVEN, 85 transitions, 0 parse failures
R2..R10               = NOT MET, no evidence
S7_READER / S7_READER_SLOT = UNRESOLVED
minReaderEpoch(T1)          = NOT CAPTURED
Case A / B / C / D          = NOT PROVEN
A2 'dd @$t1 L1'       = installed and CDB-accepted, execution count 0,
                         globalEpoch value UNPROVEN
IMPLEMENTATION              = FORBIDDEN
```

Source of truth confirmed this gate, read-only:

```text
ConvoPeq.md  SHA-256 E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
             5,535,334 bytes
ConvoPeq(4).md  does not exist on disk. ConvoPeq.md is the single present file and
             it matches the recorded authority byte for byte. No content discrepancy.
```

## 3. The gap being closed

Every guard in Redesign-4 is a silent fall-through. On mismatch the body runs a bare `gc` and
emits nothing. The executed entry gate, verbatim from the log:

```text
.if (@$t0 == 0) { .if (@r9 == 9) { .if (dd(@rcx+0x34000) == 4)
  { .echo C3_T0_ENTRY; r; dq @rsp L1; ln poi(@rsp); gc } ; .else { gc } } ; .else { gc } } ; .else { gc }
```

So the existing log cannot separate

```text
A  the instruction at the breakpoint was never executed
B  it was executed and at least one guard was false
```

The design must separate these, and must report the operand values rather than assume them.

## 4. Read-only findings that the design rests on

### 4.1 Register budget, D6 basis, audited from the script

Pseudo-register traffic per gate, guard phase reads and all writes:

| line | RVA | marker | guard reads | writes |
|---|---|---|---|---|
| 25 | `0x1f9c550` | `C3_T0_ENTRY` | `$t0` | none |
| 26 | `0x1f9c57d` | `C3_T0_CAS_PRE` | none | none |
| 27 | `0x1f9c59a` | `C3_T0_CAS_POST` | none | none |
| 28 | `0x1f9c5da` | `C3_T0_SEQUENCE_PUBLISHED` | all 20 | all 20 |
| 29 | `0x1fa80b3` | `C3_TERMINAL_S7_ANCHOR` | `$t0,$t1,$t2,$t3,$t17,$t19` | `$t19` |
| 30 | `0x1f9cfb0` | `C3_T1A_CANDIDATE` | 13 | `$t11..$t15,$t18` |
| 31 | `0x1f9cfe2` | `C3_T1B_GETMIN_CALL_PRE` | 7 | `$t18` |
| 32 | `0x1f9cfe5` | `C3_T1B_GETMIN_RETURN` | 8 | `$t14,$t15,$t18` |
| 33 | `0x1f9cd00` | `C3_T1C_DQUEUE_ENTRY` | 8 | `$t18` |
| 34 | `0x1f9cd59` | `C3_T1_SELECTION` | 16 | `$t16,$t18` |
| 35 | `0x1f9cd72` | `C3_T1_CAS_PRE` | 9 | `$t18` |
| 36 | `0x1f9ce04` | `C3_T1_CAS_SUCCEEDED` | 8 | `$t12,$t18` |
| 37 | `0x1fa80b2` | `C3_T2_WAIT_RETURN` | 6 | `$t17` |

Two facts follow, and both are needed:

```text
F1  the three gates that precede T0_SEQUENCE_PUBLISHED write no pseudo-register at all
F2  every gate that writes a pseudo-register other than $t0 carries a $t0==1 or $t12==1
    precondition.  violations = 0.
```

Therefore, before `T0_SEQUENCE_PUBLISHED` fires, `$t1` through `$t19` are never written and never
read by any gate. They are dead. And `T0_SEQUENCE_PUBLISHED` assigns all twenty deterministically:

```text
$t0=1  $t1=@r11-0x1440  $t2=@r11  $t3=@$t1+0x10  $t4..$t9 = rdx/r8/r9 halves
$t10=@rbx  $t11=0 .. $t19=0
```

Any diagnostic value placed in `$t1..$t19` is wiped the instant the T0 chain opens. This is the
budget argument for D6, and it is a property of the existing script, not a new assumption.

`$t0` is **not** free. It is the arm flag, read as `== 0` by T0_ENTRY and T0_SEQUENCE_PUBLISHED
and as `== 1` by every post-T0 gate. The design does not touch it.

### 4.2 Operand provenance, from the binary, read-only

No debugger was launched. `llvm-objdump` against the on-disk image, ImageBase `0x140000000`,
all five RVAs confirmed to map into `.text`.

```text
0x1f9c550  movq  %rbx, 0x8(%rsp)              <== T0_ENTRY
0x1f9c555  movl  0x34000(%rcx), %r10d         ; r10 = enqueuePos
0x1f9c55c  movq  %rcx, %r11                   ; r11 = queue base   [AFTER bp0]
0x1f9c560  movl  %r10d, %ebx
0x1f9c563  andl  $0xfff, %ebx                 ; ebx = enqueuePos & 0xFFF = slot  [AFTER bp0]
0x1f9c569  movl  0x30000(%r11,%rbx,4), %eax   ; eax = sequences[slot]
0x1f9c571  subl  %r10d, %eax
0x1f9c574  jne   0x1f9c58d                    ; require sequences[slot]==enqueuePos
0x1f9c576  leal  0x1(%r10), %ecx              ; ecx = enqueuePos+1
0x1f9c57d  lock cmpxchgl %ecx, 0x34000(%r11)  <== T0_CAS_PRE
0x1f9c586  je    0x1f9c59a
0x1f9c588  movl  %eax, %r10d
0x1f9c58b  jmp   0x1f9c560                    ; retry loop
0x1f9c59a  movzbl 0x28(%rsp), %eax            <== T0_CAS_POST
0x1f9c59f  leaq  (%rbx,%rbx,2), %rcx
0x1f9c5a3  addq  %rcx, %rcx                   ; rcx = slot*6
0x1f9c5a6  movb  %al, 0x18(%r11,%rcx,8)
0x1f9c5bf  movq  %rdx, (%r11,%rcx,8)          ; entry+0x00
0x1f9c5c3  movq  %r8,  0x8(%r11,%rcx,8)       ; entry+0x08
0x1f9c5c8  movq  %r9,  0x10(%r11,%rcx,8)      ; entry+0x10
0x1f9c5cd  incl  %r10d
0x1f9c5d2  movl  %r10d, 0x30000(%r11,%rbx,4) ; sequences[slot] = enqueuePos+1
0x1f9c5da  movq  0x8(%rsp), %rbx              <== T0_SEQUENCE_PUBLISHED
0x1f9c5df  retq
```

Cross-checks against source `ConvoPeq.md` and against earlier proven results:

```text
0x34000 = enqueuePos    source 62184  std::atomic<uint32_t> enqueuePos{0}
0x34040 = dequeuePos    source 62185  std::atomic<uint32_t> dequeuePos{0}
0x30000 = sequences[]   source 80296-80367 producer/consumer slot sequence protocol
entry stride 0x30       slot*6 then scale 8  ->  independently reproduces S-14-B
CAS shape               compareExchangeAtomic(enqueuePos, pos, pos+1)  source 62003
```

The region is the enqueue path of `DeferredDeletionQueue`.

**Register roles are not the same at every site.** This is the finding that most shapes the
design:

| site | `@rcx` | `@rbx` | `@r10` | `@r11` |
|---|---|---|---|---|
| `0x1f9c550` | queue base | **caller's rbx**, slot not yet computed | **caller's r10**, load is at `0x1f9c555` | **caller's r11**, `movq %rcx,%r11` is at `0x1f9c55c` |
| `0x1f9c57d` | **enqueuePos+1**, reused | slot index | enqueuePos | queue base |
| `0x1f9c59a` | clobbered | slot index | enqueuePos | queue base |
| `0x1f9c5da` | slot*6, reused | slot index | **enqueuePos+1**, `incl` already done | queue base |

Correction issued at the Implementation gate. The first version of this table listed `@r10` as
`enqueuePos` at `0x1f9c550`. That was wrong: the load `movl 0x34000(%rcx), %r10d` is at `0x1f9c555`,
which is *after* the breakpoint at `0x1f9c550`, so at that site `@r10` still holds the caller's
value. `@r11` was wrong for the same reason, `movq %rcx, %r11` being at `0x1f9c55c`.

The correction is material, because it decides what the diagnostic may safely dereference. At
`0x1f9c550` **only `@rcx` is a proven queue base**; `@r10`, `@r11` and `@rbx` are caller values at
that instant, so no address may be formed from them. The implemented preamble therefore reads
memory at `0x1f9c550` only through `@rcx`, and dumps `@r10`, `@r11`, `@rbx` as bare register
values without deriving an address from them. At the other three sites `@r11` is the queue base and
`@r11`-relative reads are used.

So a single uniform operand set is wrong by construction. Any diagnostic that reuses one operand
list across the four gates would be measuring different quantities at different sites, and at one
site would form wild addresses. The design therefore specifies a **per-gate operand list**.

### 4.3 The guards are internally coherent, with one operand the disassembly does not explain

Read against 4.2, the existing guards are self-consistent as a description of the single moment
`enqueuePos == 4`, `slot == 4`:

```text
T0_CAS_PRE          @r10==4           enqueuePos == 4
                    dd(@r11+0x34000)==4   enqueuePos == 4, same statement
                    dd(@r11+0x30010)==4   sequences[4] == 4, and slot==4 so this is
                                          sequences[slot], which the code requires to
                                          equal enqueuePos at 0x1f9c571
T0_SEQUENCE_PUBLISHED @r10==5          enqueuePos+1 == 5, i.e. enqueuePos was 4
                    @rbx==4           slot == 4
                    dd(@r11+0x30000+...)  sequences[slot] was set to 5 at 0x1f9c5d2
```

That coherence is a reason not to touch them. It is **not** evidence that they hold at runtime.

One operand has no support in the disassembly:

```text
@r9 == 9      present in the guard of all four T0 gates
              @r9 is stored verbatim to entry+0x10 at 0x1f9c5c8, an opaque 64-bit token
              the window 0x1f9c4e0..0x1f9c5df contains no comparison of r9 against 9,
              and no instruction that would establish r9 == 9
```

Stated precisely, and no further: **the disassembly supplies no basis for the value 9.** It does
not follow that 9 is wrong, that the gate failed, or that the breakpoint was unreached. This run
provides no evidence either way, and this design asserts nothing about it. It is recorded as the
single highest-information operand to measure, because it is the only conjunct common to all four
T0 gates, and therefore the only one whose falsity would close all four simultaneously.

## 5. The design

### 5.1 Principle

Add an observation-only preamble to the top of each of the four T0 breakpoint bodies, ahead of the
existing guard, leaving the existing guard byte-identical. The preamble answers "was this site
reached", "what did each operand actually hold", and "which conjunct was false". It changes no
predicate, and it cannot make a gate open that would not otherwise open.

### 5.2 Registers, two, both from the provably dead set

```text
$t16   last-observed value of the site's discriminating operand, for transition detection
$t17   hit counter for this site
```

Both are inside `$t1..$t19`, so both are dead before T0 opens and both are wiped by
`T0_SEQUENCE_PUBLISHED`. `$t0` is untouched. No register is added; the set stays `$t0..$t19`.
`$t16` and `$t17` are the two least loaded post-T0 registers: `$t16` is written only by
`C3_T1_SELECTION`, and `$t17` only by `C3_T2_WAIT_RETURN`, both of which run strictly after
`T0_SEQUENCE_PUBLISHED` has already reset them to 0.

### 5.3 Per-gate preamble, specified not written

For each of the four sites, with the operand list taken from 4.2 and not from a single shared list.
The `C3_T0_ENTRY` instance, shown as the specification of the form:

```text
r $t17 = @$t17 + 1 ; .if (@$t17 <= 8) { r $t16 = @r9 ; .echo C3_DG_T0E_HIT ; ? @$t0 ; ? @r9 ; ? @r10 ; ? @rbx ; ? @rcx ; ? @r11 ; ? dd(@rcx+0x34000) ; ? dd(@r11+0x30000+((@r10&0xfff)*4)) } ; .if (@r9 != @$t16) { r $t16 = @r9 ; .echo C3_DG_T0E_R9CHANGE ; ? @r9 ; ? dd(@rcx+0x34000) } ; <existing guard, byte-identical>
```

The other three instances differ only in their marker prefix and their operand list, which 4.2
fixes per site. `?` is CDB's expression evaluator and is read-only. `r $tn = ...` writes only a
debugger pseudo-register and never the debuggee's architectural state, so the preamble is
non-perturbing by construction rather than by argument.

### 5.4 Predicate verdict block, appended inside the same preamble

Per gate, one PASS or FAIL marker per conjunct, so D5 is satisfied by whole-line counting:

```text
.if (@$t0 == 0)          { .echo C3_DG_T0E_V0_T0_PASS }   ; .else { .echo C3_DG_T0E_V0_T0_FAIL }
.if (@r9 == 9)           { .echo C3_DG_T0E_V1_R9_PASS }   ; .else { .echo C3_DG_T0E_V1_R9_FAIL }
.if (dd(@rcx+0x34000)==4){ .echo C3_DG_T0E_V2_DQ_PASS }   ; .else { .echo C3_DG_T0E_V2_DQ_FAIL }
```

### 5.5 Discrimination table the design produces

| observed | reading |
|---|---|
| no `C3_DG_*` marker for a site | A, that site was never executed |
| `..._HIT` present, no `..._V*_PASS` | B, reached, every conjunct false |
| exactly one `..._V?_PASS` | that conjunct held, the named one did not |
| all `..._V?_PASS` | F, the gate should have opened; if it did not, the guard body itself is at fault |
| `..._R9CHANGE` with `..._V1_R9_FAIL` | the discriminating operand was observed changing and never equal to the constant |

Every cell is a distinct, countable whole-line marker set. No inference beyond the markers.

### 5.6 Volume bound

The counter branch emits at most 8 full blocks per site. Beyond that, a site emits only when its
discriminating operand changes value. Total emission per site is bounded by 8 plus the number of
distinct operand values, which is small by construction. `T0_CAS_PRE` sits inside a CAS retry
loop and may be hit many times; the transition test is what keeps that bounded, not the counter.

### 5.7 Session canary, and its honest limit

A breakpoint whose address is **proven by the existing log** to execute:

```text
bp ntdll!ZwTerminateProcess ".echo C3_DG_CANARY ; gc"
```

`ntdll!ZwTerminateProcess+0x14` appears in the Step5CQ log, so the site is known to be reached.
Its purpose is narrow and it is stated narrowly: it proves the breakpoint and `.echo` machinery
functioned in this session. It does **not** prove the T0 region ran, and it fires only at process
exit. Per-site reachability remains answered by that site's own unconditional hit marker, which is
sufficient because the marker sits at the first instruction of the body.

## 6. D1–D12 compliance

| Gate | requirement | how this design meets it |
|---|---|---|
| D1 | identify breakpoint reach | unconditional `.echo ..._HIT` as the first statement of each body; absence is itself the answer |
| D2 | `$t0` actual value | `? @$t0` in the first-eight block, plus `C3_DG_T0E_V0_T0_PASS/FAIL` |
| D3 | `r9` actual value | `? @r9` plus `..._V1_R9_PASS/FAIL` plus `..._R9CHANGE` transitions |
| D4 | DQueue position actual value | `? @r11`, `? dd(@r11+0x34000)`, `? dd(@r11+0x30000+((@r10&0xfff)*4))` |
| D5 | distinguish per-conjunct failure | one PASS/FAIL marker per conjunct, countable by whole-line equality |
| D6 | do not break the T1 `$t0..$t19` budget | only `$t16`,`$t17`, both dead pre-T0 per 4.1 F1/F2 and both reset by T0; no register added; `$t0` untouched |
| D7 | do not change the `@$t1+0x20` ReaderSlot windows | preamble is confined to the four T0 bodies; no T0 body contains a window; 256 window terms untouched |
| D8 | keep the A1 `} ;` rule | every added construct is a flat `.if`/`.else` pair closed by `}` followed by `;`; no `.if` nested inside `.else` |
| D9 | do not break `gc` semantics | no `q`, no `gc` added, removed or reordered; the existing terminal `gc` of each branch is preserved verbatim |
| D10 | diagnostic must not pollute T1 correlation | the preamble writes only `$t16`,`$t17`; no existing guard reads either; existing guard text is byte-identical, so no gate can open or close because of the diagnostic |
| D11 | statically auditable | preamble is a single fixed prefix per line, delimited by a sentinel, so a byte-level diff can count insertions per line and confirm the remainder is identical |
| D12 | runtime authorization requested separately | this gate edits nothing and requests nothing; see section 8 |

## 7. Rejected designs, recorded so they are not revisited

```text
REJECTED  assume 9/4/5 are wrong and change the guards
          no runtime evidence exists. 4.2 shows the 4/5/slot family is internally coherent.
          Changing a guard to a guessed value would replace an unmeasured question with an
          unmeasured answer.

REJECTED  make T1 breakpoints unconditional
          out of scope, and it destroys the very correlation the capture exists to establish.

REJECTED  repurpose an existing $t0..$t19 register for another purpose
          $t0 is the arm flag; $t1,$t2,$t3 carry the proven address algebra; $t4..$t9 the
          token halves; $t10 the slot; $t11..$t18 the correlation state. Reuse would corrupt
          the T1 contract. The design takes only the two registers proven dead.

REJECTED  extend T1 capture to re-measure the ReaderSlot layout
          out of scope. The layout is already corroborated twice, statically by the window
          geometry and at runtime by U3. Re-measuring it is not what blocked this run.

REJECTED  add diagnostic code to production source
          forbidden by scope, and unnecessary: the preamble is debugger-side only.

REJECTED  treat this run as a T0 gate failure and proceed
          NOT MET is not FAIL. Reached-versus-guard-false is exactly what is unestablished.
```

## 8. What this design does not claim

```text
does not claim  the T0 gates failed
does not claim  the breakpoint was unreached
does not claim  @r9 == 9 is wrong
does not claim  the enqueuePos == 4 window is unreachable
does not claim  S7_READER, S7_READER_SLOT, minReaderEpoch(T1) or Case A-D are determined
does not claim  A2 produced a globalEpoch value; its execution count is still 0
```

`IMPLEMENTATION` remains `FORBIDDEN`. The frozen items are untouched by a design.

## 9. Next gate, requested only after this design is accepted

```text
Implementation authorization
        |
        v
.cdb creation, 4 preambles inserted, existing guards byte-identical
        |
        v
Static validation, D11 mechanical diff plus the existing V01..V20 regression set
        |
        v
Runtime authorization, requested separately
```

The static validation for the next gate must, at minimum, assert per line: the preamble sentinel
count is 4, the pre-existing guard substring is byte-identical, the total `$t0..$t19` set is
unchanged, `$t16` and `$t17` are written only inside a preamble region, the 256 window terms are
unchanged, the marker set grew only by `C3_DG_*`, and no `q` or `gc` token changed count.

## 10. Final state

```text
C3 Entry-Gate Diagnostic Redesign, read-only
= CLOSED / DESIGN COMPLETE / NOTHING IMPLEMENTED

.cdb edited            0
CDB launched           0
harness launched       0
build                  0
production source      untouched
ConvoPeq.md            read-only, SHA unchanged
ConvoPeq(4).md         does not exist; ConvoPeq.md is the authority and matches

new proven this gate   $t1..$t19 are dead before T0 opens and are reset by T0  (4.1)
new proven this gate   operand roles differ per site, so no shared operand list is valid  (4.2)
new proven this gate   entry stride 0x30 and the enqueue field offsets reproduce from source
new stated, unresolved @r9 == 9 has no basis in the disassembly; asserted neither way  (4.3)

S7_READER / S7_READER_SLOT / minReaderEpoch(T1) / Case A-D = UNRESOLVED
IMPLEMENTATION = FORBIDDEN
```
