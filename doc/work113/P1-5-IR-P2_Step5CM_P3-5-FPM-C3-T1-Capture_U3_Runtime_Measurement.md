# P3-5-FPM-C3-T1-Capture — U3 Runtime Measurement

## 1. Gate result

```text
Gate     = P3-5-FPM-C3-T1-Capture-U3-Runtime-Measurement
Mode     = runtime measurement on the real target
.cdb created        = 0     (no .cdb anywhere; commands passed via -cf from outside the repo)
build               = 0
production / test / CMake source modification = 0
CDB sessions        = 3 attempts, all AudioEngineHarness under tmp\cdb.exe
process residue     = 0 after every attempt

S-14-A  readers[] absolute base = PROVEN
S-14-B  DeletionEntry stride    = PROVEN
STOP-C3-T1-11                    = CLEARED

VERDICT = U3 SUCCESS
```

No U3 STOP condition was reached.

## 2. Frozen inputs

```text
ConvoPeq.md SHA-256     = E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
AudioEngineHarness.exe  = build\Release\AudioEngineHarness.exe
target SHA-256          = E5C7AFB9C4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75
target bytes            = 41,216,512
cdb                     = tmp\cdb.exe, 10.0.29617.1000
invocation              = AudioEngineHarness.exe --measurement=normal
```

`--measurement=normal` reaches `runWorldRetirementMeasurement` -> `testNormalMeasurement` ->
`stabilizeMeasurementBaseline` / `waitForWorldReclaimCount` ->
`driveWorldRetirementReclaimForMeasurement` -> `tryReclaimResources` -> `tryReclaim`. The reclaim
path is therefore exercised, and the log confirms it: the harness prepared the DSP chain and the
breakpoint was reached.

## 3. Evidence logs

| attempt | SHA-256 | bytes | content |
|---|---|---|---|
| 1 | `369057DEB3AB16426C514B043508DB256DE773A010CB4CF6A909C95419521215` | 9,620 | harness-side command syntax error, no data captured |
| 2 | `3E7F8973F3C04F0C94E4EFDBB287E47B4BDEA8CFF5CD5F801FB17524C2089B3E` | 60,066 | domain dump, vtable slot, minReaderEpoch |
| 3 | `ED3333BB80CC7F4AAC51CA360F0B04FDC27671B48D84F0456727738E890E0ADB` | 7,161 | `getMinReaderEpoch` entry disassembly |

Attempt 1 aborted with a CDB `Syntax error` inside the breakpoint body. That is a defect in the
measurement command, not a U3 STOP condition, and no data was captured. Attempts 2 and 3 used
minimally reduced bodies. No C3 script was created or modified in any attempt.

## 4. How `getMinReaderEpoch` was located, which U2 could not do

Attempt 2 stopped at the dispatch, RVA `0x1f9cfe2`:

```text
00007ff7`1c39cfe2  ff5040   call qword ptr [rax+40h]
                             ds:00007ff7`1c6529b8 = 00007ff71c39c880
```

Reading the vtable slot at runtime, which is what U2 had no way to do:

```text
vtable                              = 0x00007ff71c652978
poi(vtable + 0x40)                  = 0x00007ff71c39c880
RVA of the implementation          = 0x00007ff71c39c880 - 0x00007ff71a400000 = 0x1F9C880
```

```text
U2-P2  getMinReaderEpoch implementation = PROVEN, by runtime observation of the vtable slot
```

This is the item U2 stopped on. It did not require symbols; it required executing the dispatch.

## 5. The implementation body, and what it fixes

Attempt 3 broke at RVA `0x1f9c880` and disassembled the entry. `rcx` is this:

```text
1c39c882  sub   rsp,20h
1c39c886  mov   rax, qword ptr [rcx]          ; globalEpoch, this + 0x00
1c39c889  mov   rbx, rcx                      ; rbx = this
1c39c88f  lea   r8,  [rbx+1400h]
1c39c896  mov   r9,  rax                      ; r9 = initial minEpoch = globalEpoch
1c39c899  lea   rdx, [rbx+20h]                ; slot cursor  = this + 0x20
1c39c89d  lea   r10, [rbx+1420h]              ; slot end     = this + 0x1420
1c39c8a4  cmp   rbx, r8
1c39c8a7  je    1c39c8e5
1c39c8b0  movzx ecx, byte ptr [rdx+48h]       ; quarantineFlags  = slot + 0x48
1c39c8b4  test  cl, 1                         ; kQuarantinedFlag
1c39c8b7  jne   1c39c8d9                      ; quarantined, skip
1c39c8b9  mov   eax, dword ptr [rdx+8]        ; depth            = slot + 0x08
1c39c8bc  test  eax, eax
1c39c8be  je    1c39c8d9                      ; depth == 0, skip
1c39c8c0  mov   rcx, qword ptr [rdx]          ; epoch            = slot + 0x00
1c39c8c3  cmp   rcx, 0FFFFFFFFFFFFFFFDh       ; kReservedEpoch, and kInactiveEpoch folded in
1c39c8c7  ja    1c39c8d9                      ; sentinel, skip
```

Every element of the source predicate `ConvoPeq.md:57328-57362` is present and at the offset the
source declares:

```text
U2-P6  quarantineFlags = slot + 0x48   PROVEN
U2-P7  depth           = slot + 0x08   PROVEN
U2-P8  epoch           = slot + 0x00   PROVEN
U2-P9  globalEpoch     = this  + 0x00   PROVEN
U2-P10 minEpoch initialised from globalEpoch, exactly as the source does   PROVEN
```

The sentinel test is `cmp rcx, -3 ; ja`, which accepts `epoch <= 0xFFFFFFFFFFFFFFFD`, that is, it
excludes exactly `-1` and `-2`. That is the source's two tests
`epoch == kInactiveEpoch || epoch == kReservedEpoch` folded into one unsigned compare, so the
sentinel semantics are confirmed against the running binary and not merely against the source.

## 6. Stride and slot count, measured rather than assumed

The array bounds are literal in the binary: `this + 0x20` to `this + 0x1420`, a span of `0x1400`.
Independently, the Attempt-2 domain dump was scanned for the `kInactiveEpoch` sentinel
`0xFFFFFFFFFFFFFFFF`, and the periodicity was measured from the data:

```text
sentinel occurrences in the dump = 63
distinct inter-sentinel periods  = 0x50      (one single value)
slots accounted for              = 64        (63 inactive + 1 active)
```

```text
U2-P4  readers[] stride = 0x50   PROVEN, measured as a single distinct period
```

`0x50` was not assumed and not inherited from the orphan PDB. `0x1400 / 64 = 0x50` agrees with the
independently measured period and with the declared `kMaxReaders = 64`.

## 7. `readers[]` absolute base, and the pointer identity that was missing

### 7.1 Two distinct `this` pointers exist, and confusing them is the whole error

At the C3 dispatch site, RVA `0x1f9cfe2`, `rcx` is **not** the pointer the implementation uses:

```text
141f9cfb6  mov  rbx, rcx        ; rbx = EpochDomain as seen by tryReclaim
141f9cfdb  add  rcx, -10h       ; thunk adjustment
141f9cfdf  mov  rax, [rcx]      ; vtable
```

Attempt 2 confirms this at runtime. With `rbx = 0x28d60126750`:

```text
rbx = 0000028d60126750        (tryReclaim this)
rcx = 0000028d60126740        (implementation this)  = rbx - 0x10
```

So the implementation's `this` is the tryReclaim `this` minus `0x10`.

### 7.2 The two addresses, resolved

```text
globalEpoch = impl_this + 0x00
readers[]   = impl_this + 0x20
```

Substituting `impl_this = $r11 - 0x1440`, where `$r11` is the `DeferredDeletionQueue` base proven
in U2 section 3:

```text
globalEpoch = $r11 - 0x1440
readers[]   = $r11 - 0x1420
```

### 7.3 Correlation check against the measured return value

Attempt 2 captured the return of the call:

```text
U3B_POST  rax = 0000000000000001        => minReaderEpoch = 1
```

The dump places slot 0's epoch field, that is `readers[0] + 0x00 = $r11 - 0x1420`, at domain
offset `+0x10`, holding `0000000000000001`. So:

```text
readers[0].epoch = 1 = minReaderEpoch
```

and the adjacent fields of that same slot read `depth = 1` at `+0x08`, `ownerTag` beginning
`"unnam"` at `+0x28`, and `quarantineFlags = 0` at `+0x48`. The slot that produced the returned
minimum is therefore the same slot the implementation's own loop reads, and it satisfies the
eligibility predicate.

```text
S-14-A  readers[] absolute base = PROVEN
        readers[]   = $r11 - 0x1420
        globalEpoch = $r11 - 0x1440
        stride      = 0x50, count 64
STOP-C3-T1-11 = CLEARED
```

## 8. Correction to the U2 finding, stated plainly

U2 section 4.1 concluded that `$t1 = @r11 - 0x1440` was wrong because `$r11 - 0x1430` is the
tryReclaim `this`. That reasoning was sound but the conclusion drawn from it was wrong, because it
assumed `$t1` was intended to be the tryReclaim `this`. It is not.

```text
$t1 = @r11 - 0x1440  is  globalEpoch's address
$t1 + 0x20           is  readers[] base
```

Both are exactly what C3-Redesign-2 uses them for. **The `$t1 + 0x20` slot base in C3-Redesign-2 is
correct for this binary and requires no change.** U2's finding 4.1 is superseded. Had U2's claim been
acted on, a working expression would have been broken.

The `$t2 + 0xC0` finding in U2 section 4.2 is unaffected and stands: the binary computes
`entry = this + index*0x30`, so the `+0xC0` term is spurious.

## 9. Disposition of the two machine-code defects

| expression | status after U3 |
|---|---|
| `$t1 = @r11 - 0x1440` | **CORRECT**, it is `globalEpoch` |
| `$t1 + 0x20` as slot base | **CORRECT**, it is `readers[]` |
| `$t2 + 0x30000 + (index*4)` | correct, `sequences[index]` |
| `$t2 + 0x34000` | correct, `enqueuePos` |
| `$t2 + 0x34040` | correct, `dequeuePos` |
| T1 guard `$t2 + 0xC0 + index*0x30` | **WRONG**, must be `$t2 + 0x00 + index*0x30` |

Replacement for the withdrawn design option `O1`:

```text
O1 was   dd @$t2+0x34000 L1        (withdrawn, that address is enqueuePos)
O1-new   dd @$t1 L1                 ($t1 is globalEpoch; no new pseudo-register required)
```

The register budget problem in the design gate is therefore resolved without spending a register,
because `$t1` already holds the address that is needed.

## 10. U3 STOP conditions

| condition | status |
|---|---|
| `readers[]` base not uniquely identifiable | not reached, uniquely resolved |
| multiple candidates remaining | not reached, one value, three independent confirmations |
| stride not uniquely determined | not reached, one distinct period, plus literal array bounds |
| slot not provably the object `getMinReaderEpoch` reads | not reached, the loop's own displacement and the return value agree |
| `globalEpoch` and `minReaderEpoch` correspondence unprovable | not reached, initialiser proven at `this+0x00` |
| multiple readers, T1 uniqueness undecidable | not reached, one active slot observed |
| correlation to the same invocation or DQueue ticket | **out of scope for U3**, see below |

The last row is deliberate. U3 measured object layout, not the C3 T1 capture. Correlation of a
ReaderSlot to a specific `tryReclaim` invocation, a specific DQueue ticket, and a specific T1 epoch
gate remains the job of the C3 capture itself, and `S7_READER` remains to be determined there.

## 11. Constraints honored

```text
.cdb created                = 0, none anywhere
build                       = 0
production / test / CMake source modification = 0
C3 T1 capture script        = not created, not modified
EpochDomain + 0x20 adopted  = NO, it was explicitly forbidden and was not used
orphan PDB values used      = NO, 0x20 and 0x1440 were both tested against the binary and rejected
reader / epoch / reclaim state altered = NO, the run only read memory
process residue             = 0 after every attempt
```

The measurement was observational. No register, no memory location and no ConvoPeq state was
written by any breakpoint body; every body was a read followed by `gc` or `q`.

## 12. Next gate

```text
U3 Runtime Measurement              CLOSED / SUCCESS
S-14-A readers[] absolute base      PROVEN, STOP-C3-T1-11 CLEARED
S-14-B DeletionEntry stride         PROVEN, STOP-C3-T1-12 CLEARED
C3-Redesign-2 correction design     NOT STARTED
.cdb creation                       still BLOCKED until that design closes
Static Validation                   BLOCKED
Runtime Authorization               BLOCKED
Execution of the C3 capture         BLOCKED
```

The correction design must record, as its input, the one genuine defect found, the `+0xC0` term in
the T1 selection guard, and must carry forward the four items now measured so that no later gate
re-derives them. The withdrawn `O1` and its replacement `$t1` are part of that input.

```text
DIRECT_DQUEUE_CAUSE = epoch gate equality PROVEN
Candidate A         = PROVEN
B1                  = PROVEN
readers[] layout    = PROVEN, stride 0x50, count 64
globalEpoch address = PROVEN
S7_READER           = UNRESOLVED
S7_READER_SLOT      = UNRESOLVED
minReaderEpoch      = NOT CAPTURED for a T1 invocation
Case A/B/C/D        = NOT PROVEN
IMPLEMENTATION      = FORBIDDEN
```
