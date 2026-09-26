# P3-5-FPM-C3-T1-Capture — U2 Read-only Static Provenance Audit

## 1. Gate result

```text
Gate     = P3-5-FPM-C3-T1-Capture-U2-Read-only-Static-Provenance-Audit
Mode     = read-only static analysis of build\Release\AudioEngineHarness.exe
.cdb created        = 0
CDB execution       = 0
ping execution      = 0
AudioEngineHarness  = 0
build               = 0
production / test / CMake source modification = 0

U2-P1  @r11 provenance                = PROVEN
U2-P2  getMinReaderEpoch impl         = NOT PROVEN
U2-P3  readers[] walk                 = NOT PROVEN
U2-P4  readers[] stride 0x50          = NOT PROVEN
U2-P5  readers[] absolute displacement= NOT PROVEN
U2-P6  quarantineFlags +0x48          = NOT PROVEN
U2-P7  depth +0x08                    = NOT PROVEN
U2-P8  epoch +0x00                    = NOT PROVEN
U2-P9  globalEpoch displacement       = NOT PROVEN
U2-P10 currentEpoch -> globalEpoch    = NOT PROVEN

VERDICT = U2 FAILED / U2-STOP-2, U2-STOP-3, U2-STOP-6 TRIGGERED
        = S-14-A STILL NOT PROVEN / NO TRANSITION TO U3
```

Per the instruction, this audit stops at its STOP conditions. It does not fall through to U3 and it
creates no script.

## 2. Frozen source authority and target identity

```text
ConvoPeq.md SHA-256 = E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
target  = build\Release\AudioEngineHarness.exe
target SHA-256 = E5C7AFB9C4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75
size    = 41,216,512 bytes
ImageBase = 0x140000000     SizeOfImage = 42,409,984
.text RVA 0x1000 .. 0x20E2CA1, file offset = RVA - 0xC00
```

All RVAs below are read from this exact file. No PDB was used as a witness; see section 7.

## 3. U2-P1 PROVEN: the identity of `@r11`

### 3.1 The writer

The T0 chain is not four functions. `int3` padding at RVA `0x1f9c54c`–`0x1f9c54f` marks a function
boundary, and a single function spans all four C3 breakpoints.

```text
141f9c550: movq  %rbx, 0x8(%rsp)        ; prologue
141f9c555: movl  0x34000(%rcx), %r10d   ; enqueuePos
141f9c55c: movq  %rcx, %r11             ; <<< THE ONLY WRITER OF r11 IN THIS PATH
```

`rcx` is the first integer argument on entry, that is `this`. Therefore at RVA `0x1f9c5da`,
`@r11 == this` of the function that begins at `0x1f9c550`. The value is not guessed; it is the
destination of a single `mov`.

### 3.2 The same function identifies the object

Everything the function touches is `DeferredDeletionQueue`-relative:

```text
141f9c569: movl  0x30000(%r11,%rbx,4), %eax   ; sequences[index]
141f9c57e: cmpxchgl %ecx, 0x34000(%r11)        ; enqueuePos CAS
141f9c5a6: movb   %al, 0x18(%r11,%rcx,8)      ; entry.type
141f9c5b0: movq   %rax, 0x20(%r11,%rcx,8)     ; entry.publicationSequenceId
141f9c5ba: movq   %rax, 0x28(%r11,%rcx,8)     ; entry.generation
141f9c5bf: movq   %rdx, (%r11,%rcx,8)         ; entry.ptr
141f9c5c3: movq   %r8,  0x8(%r11,%rcx,8)      ; entry.deleter
141f9c5c8: movq   %r9,  0x10(%r11,%rcx,8)     ; entry.epoch
141f9c5d2: movl   %r10d, 0x30000(%r11,%rbx,4) ; sequences[index] release
```

with the index scale from

```text
141f9c59f: leaq  (%rbx,%rbx,2), %rcx   ; index * 3
141f9c5a3: addq  %rcx, %rcx             ; index * 6
141f9c5a6: ...  (%r11,%rcx,8)          ; * 8  =>  index * 48
```

```text
U2-P1  @r11 = DeferredDeletionQueue object base     PROVEN
       @r11 is NOT the EpochDomain base
```

### 3.3 The EpochDomain base follows, and it is not 0x1440 away

`EpochDomain::tryReclaim` at RVA `0x1f9cfb0` keeps `this` in `rbx` and passes the queue member to
the reclaim routine:

```text
141f9cfb6: movq  %rcx, %rbx            ; rbx = EpochDomain
141f9cfe8: leaq  0x1430(%rbx), %rcx    ; &deferredDeletionQueue
141f9cfef: callq 0x141f9cd00           ; reclaim
```

```text
EpochDomain base  = @r11 - 0x1430      PROVEN from the target binary
```

## 4. Two corrective findings against C3-Redesign-2

These are machine-code proofs, not reinterpretations, and both concern the existing script.

### 4.1 `$t1 = @r11 - 0x1440` is wrong for this binary

```text
C3-Redesign-2   $t1 = @r11 - 0x1440
binary          EpochDomain base = @r11 - 0x1430
difference      0x10

=> $t1 resolves to EpochDomain - 0x10, not the EpochDomain base
=> $t1 + 0x20 resolves to EpochDomain + 0x10, which is NOT readers[]
```

The `0x1440` figure originates from the orphan PDB discussed in Preparation-1 section 3. The binary
says `0x1430`. This is the concrete reason `STOP-C3-T1-11` is triggered, and it invalidates the
`$t1 + 0x20` slot-base assumption independently of any PDB question.

### 4.2 The T1 selection guard's `+0xC0` is wrong for this binary

```text
binary          entry = this + index*0x30          => ringBuffer at this + 0x00
C3-Redesign-2   @rsi == ($t2 + 0xc0 + index*0x30)   => ringBuffer at $t2 + 0xC0
difference      0xC0
```

Confirmed at a second, independent site, `reclaim` itself:

```text
141f9cd4e: leaq  (%rcx,%rcx,2), %rsi
141f9cd52: shlq  $0x4, %rsi
141f9cd56: addq  %rdi, %rsi        ; rdi = this, so entry = this + index*0x30
```

The `+0xC0` term is spurious for this binary. Note this is a *selection-guard* defect: it would make
`C3_T1_SELECTION` unreachable, so it must be corrected before any T1 capture can succeed.

### 4.3 Consistent with the binary, retained

```text
$t2 + 0x30000 + (index*4)   sequences[index]   MATCHES
$t2 + 0x34000               enqueuePos        MATCHES
$t2 + 0x34040               dequeuePos        MATCHES
```

so `$t2` is indeed the queue base. A single `$t2` cannot satisfy `+0xC0` and `+0x34000`
simultaneously only because `+0xC0` is the wrong term; with `+0x00` all four agree.

## 5. DeletionEntry, re-confirmed at a second site

```text
+0x00  ptr                     from rdx
+0x08  deleter                 from r8
+0x10  epoch                   from r9
+0x18  type                    byte, from 0x28(%rsp)
+0x20  publicationSequenceId   from 0x30(%rsp)
+0x28  generation              from 0x38(%rsp)
stride 0x30, computed as index*3*2*8
```

This agrees with Preparation-1 section 4 and with the source declaration at
`ConvoPeq.md:61945-61955`. `objectBytes` is absent, so `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` is OFF
in the target binary. `S-14-B` remains PROVEN and is now supported by two disjoint code sites.

## 6. U2-P2 attempted and failed: resolving vtable slot `+0x40`

The dispatch is confirmed:

```text
141f9cfdb: addq  $-0x10, %rcx        ; virtual-base thunk adjustment
141f9cfdf: movq  (%rcx), %rax         ; vtable pointer, read at (this - 0x10)
141f9cfe2: callq *0x40(%rax)          ; slot 0x40
```

To resolve slot `0x40` the vtable address is required. The RTTI route was attempted in full:

```text
1  byte search for the mangled name ".?AVEpochDomain@convo@@"
     1 occurrence, at RVA 0x2715228
2  TypeDescriptor therefore at RVA 0x2715218
3  scan every 4-byte boundary for an RTTICompleteObjectLocator whose
     pTypeDescriptor field equals 0x2715218
     0 candidates
```

A `COL` is required to be 4-byte aligned and must satisfy `pSelf == own RVA`; the scan enforced both.
No `COL` references the type descriptor, so this build exposes no locatable RTTI anchor for
`EpochDomain`, and the vtable cannot be located from RTTI.

**U2-STOP-2 triggered.** The vtable address remains unknown, therefore the implementation behind
slot `0x40` remains unknown, therefore the reader walk cannot be entered from the dispatch site.

## 7. Why the content searches did not converge

Four discriminators were tried against `.text`. Three were non-selective at this binary's size, which
is itself a result about the viability of a search-based U2.

| discriminator | intended | sites | outcome |
|---|---|---|---|
| `test byte [reg+0x48], 1` | quarantineFlags test | 0 | encoding not used |
| `movzbl [reg+0x48], %e` | quarantineFlags load | 5 | all inspected, all are `quarantineReader` and unrelated helpers |
| `add $0x50, reg64` | reader-array stride | 470 | far too many to discriminate |
| `shrq $0x3f, reg64` | the `isOlder` sign idiom | 1648 | far too many to discriminate |

The `isOlder` idiom is present at the known epoch gate, RVA `0x1f9cd60`, so the search is correct;
it simply has no selectivity in a 34 MB optimised text section. Static content search is therefore
not a viable route to the reader walk in this binary without first pinning the vtable.

Two of these searches initially reported zero results because of defects in the scanning harness
rather than the binary: an incorrect ModRM mask on the `shr` opcode class, and a malformed array
append. Both were found, corrected, and re-run; the figures above are from the corrected scans.

**U2-STOP-6 triggered.** Many candidates exist and none can be identified as
`getMinReaderEpoch` without the vtable.

## 8. What U2 did not deliver

| item | status | reason |
|---|---|---|
| readers[] absolute displacement | not proven | walk not entered, section 6 and 7 |
| readers[] stride 0x50 in the walk | not proven | walk not located |
| quarantineFlags `+0x48` in the walk | not proven | walk not located |
| depth `+0x08` in the walk | not proven | walk not located |
| epoch `+0x00` in the walk | not proven | walk not located |
| globalEpoch displacement | not proven | depends on the walk entry point |
| `currentEpoch()` to `globalEpoch` correlation | not proven | depends on the above |

The `ReaderSlot` field offsets in Preparation-1 section 6 remain what they were: supported by source
declaration and by an unbound PDB, and **not** independently re-proven against this binary. The
`0x50` stride in particular is still unconfirmed for the target, even though the source arithmetic
and the orphan PDB agree on it.

## 9. Provenance discipline observed

```text
the orphan PDB was NOT used to establish any conclusion in this gate
all positive findings come from the target executable's own bytes
0x20 was NOT adopted for readers[], per the instruction
no value was carried over from the orphan PDB
```

Section 4.1 is the clearest illustration. The tempting move was to keep `$t1 + 0x20` because the PDB
agreed with it. The binary disagrees by `0x10`, so the expression is recorded as wrong instead.

## 10. Read-only confirmation

```text
CDB / ping / AudioEngineHarness execution = 0
build = 0
production / test / CMake source modification = 0
.cdb files in doc/work113 = 7, unchanged
C3-Redesign-2 .cdb = FBE32851F795AF900E6C46D6E38F25138E01217C2B52C0BDDE1F015029CFC39E, unchanged
src, test, CMake, build.bat delta = 12, unchanged
target exe SHA-256 unchanged
```

## 11. Final state

```text
P3-5-FPM-C3-T1-Capture-U2-Read-only-Static-Provenance-Audit
= CLOSED / FAILED AT STOP CONDITIONS / NO TRANSITION TO U3

U2-P1  @r11 = DeferredDeletionQueue base         PROVEN
       EpochDomain base = @r11 - 0x1430          PROVEN
       DeletionEntry stride 0x30 and layout       PROVEN, two sites
U2-P2  getMinReaderEpoch implementation          NOT PROVEN   RTTI has no locatable COL
U2-P3  readers[] walk                            NOT PROVEN   search not selective
U2-P4  readers[] stride 0x50                     NOT PROVEN
U2-P5  readers[] absolute displacement           NOT PROVEN
U2-P6  quarantineFlags +0x48                     NOT PROVEN
U2-P7  depth +0x08                               NOT PROVEN
U2-P8  epoch +0x00                               NOT PROVEN
U2-P9  globalEpoch displacement                  NOT PROVEN
U2-P10 currentEpoch -> globalEpoch               NOT PROVEN

U2-STOP-2  TRIGGERED
U2-STOP-3  TRIGGERED
U2-STOP-6  TRIGGERED

S-14-A readers[] absolute base    = NOT PROVEN
STOP-C3-T1-11                     = STILL TRIGGERED
S-14-B DeletionEntry stride        = PROVEN, STOP-C3-T1-12 CLEARED

C3-Redesign-2 corrections now required, machine-code proven
  $t1 = @r11 - 0x1440        WRONG, must be @r11 - 0x1430
  T1 guard + 0xC0 ring term  WRONG, must be + 0x00

U1 rebuild                    = NOT AUTHORIZED
U2 static audit               = CLOSED, FAILED
U3 runtime measurement        = NOT AUTHORIZED, NOT STARTED
.cdb creation                 = BLOCKED
Static Validation             = BLOCKED
Runtime Authorization         = BLOCKED
Execution                     = BLOCKED
T1 capture / minReaderEpoch   = NOT CAPTURED
S7_READER / S7_READER_SLOT    = UNRESOLVED
IMPLEMENTATION                = FORBIDDEN
```
