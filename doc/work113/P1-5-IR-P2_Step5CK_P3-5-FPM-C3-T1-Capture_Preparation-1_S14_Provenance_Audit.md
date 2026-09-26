# P3-5-FPM-C3-T1-Capture — Preparation-1: S-14 Provenance Audit

## 1. Gate result

```text
Gate     = P3-5-FPM-C3-T1-Capture-Preparation-1
Mode     = read-only provenance audit
.cdb created        = 0
CDB execution       = 0
ping execution      = 0
AudioEngineHarness  = 0
build               = 0
production / test / CMake source modification = 0

S-14-A  readers[] absolute base        = PROVEN     (closed)
S-14-B  DeletionEntry stride          = NOT PROVEN (OPEN)

VERDICT = PREPARATION-1 INCOMPLETE / .cdb CREATION BLOCKED
```

One of the two items closed. Because S-14-A remains open, this gate does not complete, and no
debugger script may be created.

## 2. Frozen source authority

```text
ConvoPeq.md SHA-256 = E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
ConvoPeq.md bytes   = 5,535,334
```

Exactly one `ConvoPeq*.md` exists in the working tree. The superseded `ConvoPeq(3).md` was not used
and is not present. Every source citation below refers to line numbers in this file.

## 3. Evidence base, and its binding status

Two artifacts exist for the target:

```text
build\Release\AudioEngineHarness.exe   41,216,512 bytes   2026-09-23 16:00:47Z
build\Release\AudioEngineHarness.pdb   59,355,136 bytes   2026-09-22 12:50:18Z
```

The PDB is **not bound** to the executable. Four independent checks agree:

| check | result |
|---|---|
| exe's debug directory entry types | `POGO (0xD)` only, **no CodeView** |
| byte search of the exe for an `RSDS` record | 0 records |
| `RSDS` + GUID `{D0019BC1-920C-46AE-B4EA-46E2FCDA3722}` + age 88 | no match |
| `build-Release.ninja` search for `/DEBUG` or `/PDB` | none present |
| `.ninja_log` outputs for this target | `Release/AudioEngineHarness.exe` only, no `.pdb` |
| timestamps | PDB is 27 h **older** than the exe |

The current Release link emits no PDB. The `.pdb` on disk is a leftover from an earlier
configuration. It is evidence about a **different binary** and cannot serve as provenance for the
target, no matter how plausible its contents look.

This is not a formality. Section 6 shows the PDB is not merely unbound but **demonstrably wrong**
about this exe by 0x10.

## 4. S-14-B: DeletionEntry stride — PROVEN

Closed from the target binary's own machine code, not from the stale PDB and not from the CMake
default. All addresses are RVAs; the image base is `0x140000000`.

### 4.1 The stride is computed in the binary

`DeferredDeletionQueue::reclaim`, RVA `0x1f9cd00`:

```text
141f9cd30: movl  %ebp, %ecx              ; head = dequeuePos
141f9cd32: andl  $0xfff, %ecx            ; kMask, kQueueSize 4096
141f9cd38: leaq  (%rdi,%rcx,4), %r15     ; &sequences[index]
141f9cd3c: movl  0x30000(%r15), %eax     ; sequences[index]
141f9cd4e: leaq  (%rcx,%rcx,2), %rsi     ; index * 3
141f9cd52: shlq  $0x4, %rsi              ; index * 48
141f9cd56: addq  %rdi, %rsi              ; entry = this + index * 48
```

```text
entry stride = 3 * 16 = 48 = 0x30     PROVEN from the target binary
```

### 4.2 The member layout is confirmed in the binary

Same function:

| access | offset | field per source `ConvoPeq.md:61945-61955` |
|---|---|---|
| `movq (%rsi), %rcx` | `0x00` | `ptr` |
| `movq 0x8(%rsi), %rax` | `0x08` | `deleter` |
| `movq 0x10(%rsi), %rax` | `0x10` | `epoch` |
| `movzbl 0x18(%rsi), %ebx` | `0x18`, 1 byte | `type`, and `DeletionEntryType` has underlying type `unsigned char` |

The epoch gate itself, RVA `0x1f9cd59`, is the source `isOlder` compiled exactly:

```text
movq  0x10(%rsi), %rax     ; a = entry.epoch
subq  %r13, %rax           ; a - b,  b = minReaderEpoch
shrq  $0x3f, %rax          ; sign bit
testb %al, %al
je    ...                  ; skip when not older
```

### 4.3 Consequence

```text
sizeof(DeletionEntry) = 48 = 0x30
objectBytes           = ABSENT
=> CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS is OFF in the target binary
```

Corroboration, consistent and independent: `build\CMakeCache.txt` has
`CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS:BOOL=OFF`, and the macro is absent from
`build\build-Release.ninja`. The binary is the authority; the configuration only agrees with it.

### 4.4 Also established in the same function, DQueue-relative

```text
ringBuffer        at this + 0x00000
sequences         at this + 0x30000
dequeuePos        at this + 0x34040     read at entry, lock cmpxchg at 0x1f9cd72
reclaimSuccessCount_  at this + 0x340C0
referenceObserver_    at this + 0x340C8
kMask             = 0xFFF
```

The CAS at RVA `0x1f9cd72` is `lock cmpxchgl %r14d, 0x34040(%rdi)`, which is the post-condition the
C3 script tests as `dd(@rdi+0x34040) == (@$t10+1)`. That guard is consistent with the binary.

## 5. S-14-A: readers[] absolute base — NOT PROVEN

### 5.1 What the stale PDB claims

Read from `build\Release\AudioEngineHarness.pdb`, type record `0xC5FA`, field list `0xC5F9`:

```text
globalEpoch            offset 24      (0x18)
readers                offset 32      (0x20)   type std::array<convo::EpochDomain::ReaderSlot,64>
deferredDeletionQueue  offset 5184    (0x1440)
epochGeneration_       offset 218504  (0x35588)
sizeof(EpochDomain)    218560         (0x355A0)
```

`sizeof(convo::EpochDomain::ReaderSlot)` is 80 = `0x50`, with members `epoch 0`, `depth 8`,
`enterCount 16`, `residencyStartTimestampUs 24`, `ownerThreadId 32`, `ownerTag 40`,
`quarantineFlags 72`. `64 * 0x50 = 0x1400`, so `readers` would span `0x20 .. 0x1420` and
`deferredDeletionQueue` would begin at `0x1440`, a `0x20` gap. That is internally coherent, and it
is what would make `$t1 + 0x20` the slot base.

### 5.2 The binary contradicts it by 0x10

`EpochDomain::tryReclaim`, RVA `0x1f9cfb0`:

```text
141f9cfb6: movq  %rcx, %rbx           ; rbx = this, the EpochDomain
141f9cfdb: addq  $-0x10, %rcx         ; virtual thunk adjustment
141f9cfdf: movq  (%rcx), %rax         ; vtable
141f9cfe2: callq *0x40(%rax)          ; getMinReaderEpoch, virtual
141f9cfe5: movq  %rax, %rdx           ; minReaderEpoch
141f9cfe8: leaq  0x1430(%rbx), %rcx   ; this + 0x1430
141f9cfef: callq 0x141f9cd00          ; reclaim
```

```text
PDB     says deferredDeletionQueue at domain + 0x1440
binary  says deferredDeletionQueue at domain + 0x1430
delta   = 0x10
```

The PDB does not describe this executable. Its `readers` offset of `0x20` therefore carries no
proven value for the target, even though it is very likely close to correct.

### 5.3 What the binary does establish, and what it does not

Established for the target binary, relative to `this` of `reclaim`, that is `domain + 0x1430`:

```text
domain + 0x1430 + 0x00000  ringBuffer
domain + 0x1430 + 0x30000  sequences
domain + 0x1430 + 0x34040  dequeuePos
domain + 0x35530           reclaimAttemptCount_
domain + 0x35570           reclaimLocalCounter_
```

Not established:

```text
the identity of @r11 at C3_T0_SEQUENCE_PUBLISHED, RVA 0x1f9c5da
the EpochDomain object base, as the script sees it
the readers[] absolute offset
the globalEpoch absolute offset
```

At RVA `0x1f9c5da` the code is an epilogue, `movq 0x8(%rsp), %rbx; ret`, so `@r11` there is
whatever the T0 path left in that register. Nothing in the recovered evidence ties it to the
`EpochDomain` object.

### 5.4 A second, independent inconsistency

The C3-Redesign-2 T1 guard is

```text
@rsi == (@$t2 + 0xc0 + ((@$t10 & 0xfff) * 0x30))
```

which places ring entries at `$t2 + 0xC0`. The binary places them at `this + index*0x30`, that is
`this + 0x00` for index 0. The two disagree, so `$t2` is **not** `reclaim`'s `this`, and its true
value is unresolved. The script's own arithmetic therefore cannot be reconciled with the binary
using the available evidence.

`$t2 + 0x30000 + (index*4)` and `$t2 + 0x34040` do match the binary if `$t2 = reclaim`'s `this`,
but `$t2 + 0xC0` and `$t2 + 0x34000` do not. A single `$t2` cannot satisfy all four.

### 5.5 Targeted search for the reader walk, negative

The reader array is walked with stride `0x50`, and the loop must test `quarantineFlags` at `+0x48`.
A byte scan of `.text` for `test byte [reg+0x48], 1` and for `movzbl [reg+0x48]` returned 0 and 5
sites respectively. The five were inspected; all belong to `quarantineReader` and to unrelated
helpers, none is the `getMinReaderEpoch` array walk. The stride `0x50` for `ReaderSlot` is
therefore still supported only by the stale PDB and by source arithmetic.

### 5.6 Determination

```text
S-14-A readers[] absolute base = NOT PROVEN

  the sole type-layout witness is an orphan PDB from a different build
  that witness is demonstrably wrong about this binary by 0x10
  the binary alone yields only DQueue-relative offsets
  the script's own $t2 arithmetic is self-inconsistent against the binary
  the reader walk was not located

  category = INFERRED, therefore unusable
```

## 6. Status table required by the instruction

| item | prior status | Preparation-1 verdict |
|---|---|---|
| ReaderSlot field offsets | PROVEN | **held**, source `:57681-57697`; binary witness still the orphan PDB |
| ReaderSlot stride `0x50` | PROVEN | **held** by source arithmetic and the orphan PDB; not re-proven for the target |
| reader count 64 | PROVEN | **held**, `kMaxReaders` `:57136`, PDB `std::array<...,64>` |
| readers[] absolute base | UNRESOLVED | **NOT PROVEN**, section 5 |
| DeletionEntry fields | PROVEN | **re-proven from the binary**, section 4.2 |
| diagnostics default | OFF | **re-proven OFF in the binary**; `objectBytes` absent, section 4.3 |
| DeletionEntry stride `0x30` | UNRESOLVED | **PROVEN**, section 4.1 |
| globalEpoch source semantics | PROVEN | **held**, `:57300-57304`, `:57328-57362`; absolute offset still unknown |
| getMinReaderEpoch inline | PROVEN | **held**; in the binary it is a virtual call `*0x40(vtable)`, which refines rather than contradicts it |
| B1 anchor | PROVEN | **held**, address sanity check only |
| `$t0..$t19` allocation | fixed | **unchanged**, no reassignment |

## 7. Correction to the Design Gate

Design-Preparation-1 section 10.1 offered, as option `O1`, to obtain `globalEpoch` from the existing
`dd @$t2+0x34000 L1` dump. **That option is invalid and is withdrawn.**

```text
$2 + 0x34000  is a DeferredDeletionQueue field, not globalEpoch
globalEpoch   is an EpochDomain member at an offset that is currently unproven
```

The register budget is full, so the design still needs a way to read `globalEpoch` without a new
register, but it may not use `+0x34000` for that purpose. This is recorded now so it is not carried
into Preparation-1 script work.

## 8. Unblock options for S-14-A

Each requires a separate authorization. None is performed here.

```text
U1  relink the Release target with /DEBUG and PDB output, then re-run this audit.
    The resulting PDB would carry an RSDS record with a GUID and age that bind it to
    the exe, making the type records admissible. Requires build authorization.

U2  resolve the identity of @r11 at RVA 0x1f9c5da by static analysis of the T0 path,
    and locate the getMinReaderEpoch reader walk to read the base displacement
    directly from the binary. Read-only, but the walk was not found by a targeted
    scan and may require full .text disassembly.

U3  treat the readers[] base as an unknown to be MEASURED at runtime under a separate
    authorization, for example by correlating a known reader transition against a
    candidate base. This is the only option that does not require a rebuild, and it
    is a runtime measurement, which this gate forbids.
```

`U1` is the cleanest. `U3` is the only one compatible with the current binary.

## 9. STOP conditions triggered

| id | status |
|---|---|
| STOP-C3-T1-11, readers[] base provenance not confirmed | **TRIGGERED** |
| STOP-C3-T1-12, DeletionEntry stride not confirmed | **CLEARED**, section 4 |
| STOP-C3-T1-10, parser or marker ambiguity | not reached, no script exists |

## 10. Read-only confirmation

```text
CDB / ping / AudioEngineHarness execution = 0
build = 0
production / test / CMake source modification = 0
.cdb files in doc/work113 = 7, unchanged
C3-Redesign-2 .cdb = FBE32851F795AF900E6C46D6E38F25138E01217C2B52C0BDDE1F015029CFC39E, unchanged
src, test, CMake, build.bat delta = 12, unchanged
```

Artifacts written during this audit, all outside the repository: a full type-stream text dump under
the session temp directory, used only to read type records. No repository file other than this
document was modified.

## 11. Final state

```text
P3-5-FPM-C3-T1-Capture-Preparation-1
= CLOSED / INCOMPLETE / .cdb CREATION BLOCKED

S-14-A  readers[] absolute base       = NOT PROVEN   STOP-C3-T1-11 TRIGGERED
S-14-B  DeletionEntry stride 0x30     = PROVEN       stride read as x3<<4 from the binary
DeletionEntry layout                  = PROVEN from the binary, epoch @0x10, type @0x18 byte
objectBytes                           = ABSENT, diagnostics OFF in the target
PDB                                   = ORPHAN, unbound, 0x10 wrong about this binary
deferredDeletionQueue                 = domain + 0x1430 from the binary, 0x1440 in the orphan PDB
@r11 identity at T0                   = UNRESOLVED
$t2 consistency against the binary    = FAILS, +0xC0 and +0x34000 do not reconcile

cdb created = 0    CDB execution = 0    AudioEngineHarness execution = 0    build = 0
Design-Preparation-1 option O1 = WITHDRAWN
Preparation-1                   = INCOMPLETE
.cdb creation                   = BLOCKED
Static Validation               = NOT STARTED
Runtime Authorization           = NOT STARTED
```
