# P3-5-FPM-C3-T1-Capture — C3-Redesign-2 Correction Design

## 1. Authority

```text
ConvoPeq.md SHA-256 = E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
ConvoPeq.md bytes   = 5,535,334
```

Matches the SHA of the attached `ConvoPeq(4).md`. Exactly one `ConvoPeq*.md` exists in the working
tree; the superseded `ConvoPeq(3).md` was not used and is not present.

Design target, read-only, unmodified by this gate:

```text
doc/work113/P1-5-IR-P2_Step5BB_P3-5-FPM-C3-T1-Capture-Redesign-2.cdb
SHA-256 = FBE32851F795AF900E6C46D6E38F25138E01217C2B52C0BDDE1F015029CFC39E
lines   = 41
```

## 2. U3 provenance, frozen as input

Not re-derived, not re-searched, not re-estimated in this gate.

```text
$r11                                   = DeferredDeletionQueue base
EpochDomain / tryReclaim this          = $r11 - 0x1430
getMinReaderEpoch implementation this   = $r11 - 0x1440
globalEpoch                             = $t1 = $r11 - 0x1440
readers[]                               = $t1 + 0x20 = $r11 - 0x1420
ReaderSlot stride / count               = 0x50 / 64
ReaderSlot  epoch +0x00  depth +0x08  quarantineFlags +0x48
DQueue entry                            = $t2 + index * 0x30
DQueue  sequences[index] = $t2 + 0x30000 + index*4
        enqueuePos       = $t2 + 0x34000
        dequeuePos       = $t2 + 0x34040
DeletionEntry stride                   = 0x30
```

## 3. Supersession of U2 finding 4.1

```text
U2 section 4.1   "$t1 = @r11 - 0x1440 is wrong"          SUPERSEDED
U3 section 8     "$t1 = @r11 - 0x1440 is correct, it is globalEpoch's address"
                 "$t1 + 0x20 is correct, it is readers[] base"
```

```text
MODIFICATION OF $t1 IS FORBIDDEN
```

The U2 reasoning that `$r11 - 0x1430` is the tryReclaim `this` was correct. The error was assuming
`$t1` was intended to hold that pointer. `$t1` holds `globalEpoch`'s address instead, which is
precisely what C3-Redesign-2 uses it for. Acting on U2 would have broken a working expression.

U2 finding 4.2, the `+0xC0` term, is **not** superseded and is the subject of section 4.

## 4. The single real defect, and its complete site list

```text
CURRENT   @rsi == ($t2 + 0xc0 + (idx * 0x30))
REQUIRED  @rsi == ($t2 + 0x00 + (idx * 0x30))
```

The binary computes `entry = this + index*0x30` at RVA `0x1f9cd4e`, so the ring base is `$t2 + 0x00`
and the `+0xC0` term is spurious.

### 4.1 The instruction described one site; the defect occurs at seven

Enumerated mechanically over the whole script. Every `0xC0` occurrence in the file is listed, so
nothing is hidden:

| line | RVA | marker | expression | class |
|---|---|---|---|---|
| 28 | `0x1f9c5da` | `C3_T0_SEQUENCE_PUBLISHED` | `dq @$t2+0xc0 L6` | dump |
| 29 | `0x1fa80b3` | `C3_TERMINAL_S7_ANCHOR` | `dq @$t2+0xc0 L6` | dump |
| 30 | `0x1f9cfb0` | `C3_T1A_CANDIDATE` | `dq @$t2+0xc0 L6` | dump |
| 30 | `0x1f9cfb0` | `C3_T1A_CANDIDATE` | `dq (@$t2+0xc0+((dd(@$t2+0x34040)&0xfff)*0x30)) L6` | dump of the candidate entry |
| 34 | `0x1f9cd59` | `C3_T1_SELECTION` | `@rsi == (@$t2+0xc0+((@$t10&0xfff)*0x30))` | **guard** |
| 35 | `0x1f9cd72` | `C3_T1_CAS_PRE` | `@rsi == (@$t2+0xc0+((@$t10&0xfff)*0x30))` | **guard** |
| 37 | `0x1fa80b2` | `C3_T2_WAIT_RETURN` | `dq @$t2+0xc0 L6` | dump |

```text
guard sites = 2      dump sites = 5      total = 7
```

### 4.2 Consequence of correcting only the site named in the instruction

```text
line 34 corrected, line 35 left as is
  => C3_T1_SELECTION becomes reachable
  => C3_T1_CAS_PRE remains unreachable, its identical guard still never holds
  => the T1 record is truncated before the CAS evidence
```

Two guards carry the identical expression, so correcting the selection guard alone leaves the CAS-pre
guard broken. Machine code establishes this, not preference.

The five dump sites read `ringBuffer + 0xC0`, which is `entry[6]`, so they emit valid-looking but
misattributed memory: six quadwords of entry 6 rather than entry 0, and the candidate entry shifted
by six slots. They do not block reachability, but they corrupt evidence attribution, and an
evidence defect in a capture contract is not acceptable.

### 4.3 What is *not* changed

Only the `+0xC0` term is deleted, at the seven sites above. No other edit is required or
proposed:

```text
$t1 + 0x20                        unchanged, correct per U3
$t2 + 0x30000 + (index*4)         unchanged, correct
$t2 + 0x34000                      unchanged, correct
$t2 + 0x34040                      unchanged, correct
$t2 + (index*0x30)                the corrected form, introduced only by deleting +0xC0
$t0..$t19 assignments              untouched, see section 8
breakpoint RVAs                    untouched, see section 7
```

The 64 slot-dump windows `db @$t1+0x20+0x0 L80`, `+0x50`, `+0xa0`, `+0xf0`, `+0x140`, ... are
correct as written; `0xC0` is not among them, because the series advances by `0x50` and `0xC0` is
not a multiple of `0x50`. They must not be altered, and this gate records that they are not.

### 4.4 Scope statement

The correction instruction for this gate named the T1 selection guard. The defect is the same
single defect at seven sites, and the edit is the same deletion at each. This gate therefore treats
`STOP-C3-CORR-4` as characterised rather than cleared, and section 11 carries it as the one item
requiring the gate owner's confirmation before any `.cdb` is created. Narrowing to one site is
machine-code insufficient, and that is reported rather than quietly resolved.

## 5. O1-new, adopted

```text
O1, withdrawn   dd @$t2+0x34000 L1     (that address is enqueuePos)
O1-new, adopted  dd @$t1 L1             ($t1 is globalEpoch)
```

```text
register budget = UNCHANGED, no new pseudo-register
```

U3 strengthened `$t1` from an unexplained offset into a named quantity, so the globalEpoch capture
the design gate had to solve with a new register is served by an existing one. `STOP-C3-CORR-5` is
cleared: the capture is a single absolute read through `$t1`, unambiguous.

## 6. Register budget

```text
registers in use = $t0 .. $t19 = 20 of 20
registers added  = 0
registers freed  = 0
```

## 7. Breakpoint topology, existing to corrected

One-to-one. No breakpoint is added, removed, moved, or re-addressed.

| RVA | marker | before | after |
|---|---|---|---|
| `0x1f9c550` | `C3_T0_ENTRY` | unchanged | unchanged |
| `0x1f9c57d` | `C3_T0_CAS_PRE` | unchanged | unchanged |
| `0x1f9c59a` | `C3_T0_CAS_POST` | unchanged | unchanged |
| `0x1f9c5da` | `C3_T0_SEQUENCE_PUBLISHED` | dump `+0xC0` term | term removed |
| `0x1fa80b3` | `C3_TERMINAL_S7_ANCHOR` | dump `+0xC0` term | term removed |
| `0x1f9cfb0` | `C3_T1A_CANDIDATE` | 2 dump `+0xC0` terms | terms removed |
| `0x1f9cfe2` | `C3_T1B_GETMIN_CALL_PRE` | unchanged | unchanged |
| `0x1f9cfe5` | `C3_T1B_GETMIN_RETURN` | unchanged | unchanged |
| `0x1f9cd00` | `C3_T1C_DQUEUE_ENTRY` | unchanged | unchanged |
| `0x1f9cd59` | `C3_T1_SELECTION` | guard `+0xC0` term | term removed |
| `0x1f9cd72` | `C3_T1_CAS_PRE` | guard `+0xC0` term | term removed |
| `0x1f9ce04` | `C3_T1_CAS_SUCCEEDED` | unchanged | unchanged |
| `0x1fa80b2` | `C3_T2_WAIT_RETURN` | dump `+0xC0` term | term removed |

`STOP-C3-CORR-1` is cleared: the mapping is total and unambiguous, thirteen to thirteen.

## 8. `$t0..$t19` writer set

Extracted from the frozen script. A writer is an `r $tN=` token, not a comparison. An earlier
extraction that counted `==` as a write was a harness defect and was corrected before these figures
were taken.

| register | writer sites | role |
|---|---|---|
| `$t0` | `0x1f9c5da` | T0 sequence-published, chain armed |
| `$t1` | `0x1f9c5da` | **globalEpoch address** |
| `$t2` | `0x1f9c5da` | DQueue base |
| `$t3` | `0x1f9c5da` | target-domain comparison address |
| `$t4`,`$t5` | `0x1f9c5da` | `publicationSequenceId` low / high |
| `$t6`,`$t7` | `0x1f9c5da` | `generation` low / high |
| `$t8`,`$t9` | `0x1f9c5da` | target `epoch` low / high |
| `$t10` | `0x1f9c5da` | DQueue sequence, the ticket |
| `$t11` | `0x1f9c5da`, `0x1f9cfb0` | candidate thread token |
| `$t12` | `0x1f9c5da`, `0x1f9cfb0`, `0x1f9ce04` | candidate admitted flag |
| `$t13` | `0x1f9c5da`, `0x1f9cfb0` | candidate admitted counter |
| `$t14`,`$t15` | `0x1f9c5da`, `0x1f9cfb0`, `0x1f9cfe5` | same-invocation `minReaderEpoch` |
| `$t16` | `0x1f9c5da`, `0x1f9cd59` | T1 selection uniqueness |
| `$t17` | `0x1f9c5da`, `0x1fa80b2` | T2 captured |
| `$t18` | `0x1f9c5da` plus 7 stop sites | anomaly |
| `$t19` | `0x1f9c5da`, `0x1fa80b3` | S7 terminal-thread diagnostic |

Registers with more than one writer are not ambiguous: each write belongs to a distinct phase gated
by `$t0`, `$t12` or `$t13`, so no write can be reached out of order. The initialiser at
`0x1f9c5da` establishes the T0 ticket; later sites refine within the admitted-candidate phase.

**Writer-set invariance under the correction is provable by construction.** All seven edits lie
inside a `dq` read or an `@rsi == (...)` comparison. No `r $tN=` token lies within any edited
region, as the site table in section 4.1 shows. Deleting `+0xC0` therefore cannot alter the writer
set. `STOP-C3-CORR-2` is cleared.

## 9. T1 correlation contract

The purpose of the capture is a single correlated observation, not a second look at the reader
layout. U3 proved layout only.

```text
C1  one T0 ticket, ($t4,$t5,$t6,$t7,$t8,$t9,$t10), from 0x1f9c5da
C2  one admitted candidate at 0x1f9cfb0, with thread token $t11
C3  the same thread token and the same domain at 0x1f9cfe2
C4  minReaderEpoch from RAX at 0x1f9cfe5, split into $t14 / $t15
C5  the same thread token and the same DQueue at 0x1f9cd00
C6  the epoch gate at 0x1f9cd59 reached by that same invocation
C7  the gate's rsi equal to $t2 + index*0x30, the corrected entry form
C8  the entry's ptr, deleter, epoch, publicationSequenceId, generation equal to the C1 ticket
C9  the entry sequence read from @r15 equal to $t10, and @r15 equal to $t10 + 1
C10 the gate's minReaderEpoch operand equal to $t14 / $t15
C11 selection uniqueness: $t16 written exactly once, a second write is STOP
C12 CAS pre at 0x1f9cd72 on the same domain and entry
C13 CAS post at 0x1f9ce04 deciding SUCCEEDED or EPOCH_BLOCKED
```

The correlating keys are `@tid` for the invocation, `$rdi` for the DQueue, `$t10` for the ticket,
`@rsi` for the entry, and `$t14`/`$t15` for the epoch. Each is available at more than one site, so
each can be cross-checked rather than trusted once. `STOP-C3-CORR-7` is cleared.

## 10. S7 proof contract, unchanged in substance

U3 proved the layout. It did not identify a reader, and the following is unchanged.

For `S7_READER` and `S7_READER_SLOT`, the T1 record must bind, within one invocation:

```text
the same EpochDomain, read as @$t1 - 0x10
the same getMinReaderEpoch invocation, at 0x1f9cfe2 and its return at 0x1f9cfe5
the slot that actually passed the eligibility predicate
that slot's epoch, at slot + 0x00
the returned minReaderEpoch, $t14 / $t15
the T1 epoch gate at 0x1f9cd59
the same DQueue ticket, $t10
```

The eligibility predicate is the one U3 read out of the running binary, not the source restatement:

```text
eligible  =  (quarantineFlags & 0x01) == 0        slot + 0x48
         and depth != 0                            slot + 0x08
         and epoch <= 0xFFFFFFFFFFFFFFFD           slot + 0x00
```

That last form is `epoch != kInactiveEpoch && epoch != kReservedEpoch` folded into one unsigned
compare, which is how the binary actually tests it.

Binding rule, restated because it is the failure this whole chain has been guarding against:

```text
minReaderEpoch == globalEpoch  MUST NOT be read as "no reader exists"
minReaderEpoch == globalEpoch  MUST NOT be read as a reader identity
minReaderEpoch == globalEpoch  MUST NOT be turned into a slot index
```

`minReaderEpoch` is initialised from `globalEpoch` at `$t1` and only ever moves downward, so that
equality means no eligible slot lowered it. It is a statement about the scan, not about reader
existence. `O1-new` supplies `globalEpoch` so that the audit can make this distinction with
evidence instead of assumption.

`STOP-C3-CORR-6` is cleared: the binding is defined on quantities that U3 has proven exist at
named addresses.

## 11. STOP conditions

| id | condition | status |
|---|---|---|
| STOP-C3-CORR-1 | breakpoint topology mapping unknown | CLEARED, 13 to 13, section 7 |
| STOP-C3-CORR-2 | `$t0..$t19` writer-set invariance unconfirmable | CLEARED, section 8 |
| STOP-C3-CORR-3 | `$t1` or `$t2` semantics not unique | CLEARED, U3 section 7 |
| STOP-C3-CORR-4 | change needed beyond the `+0xC0` removal | **CHARACTERISED, NOT CLEARED**, section 4.4 |
| STOP-C3-CORR-5 | O1-new globalEpoch capture ambiguous | CLEARED, one absolute read via `$t1` |
| STOP-C3-CORR-6 | S7 to T1 invocation correlation undefinable | CLEARED, section 10 |
| STOP-C3-CORR-7 | same-ticket or same-domain correlation undefinable | CLEARED, section 9 |

`STOP-C3-CORR-4` is the one open item. The gate instruction named the T1 selection guard; the defect
is present at seven sites of the same kind, two of which are guards that would otherwise leave
`C3_T1_CAS_PRE` unreachable. No edit other than the `+0xC0` deletion is proposed, but the number of
sites exceeds the instruction, so the owner of this gate must confirm the seven-site scope before a
`.cdb` is created. This gate does not resolve that by itself.

## 12. Static Validation checklist, for the gate after this one

To be applied to the corrected script, not to the present one.

```text
V01  script SHA and byte count frozen and re-verified
V02  ConvoPeq.md SHA and byte count frozen and re-verified
V03  AudioEngineHarness.exe SHA frozen, and bound to the RVAs
V04  cdb.exe SHA and version frozen
V05  breakpoint RVA set byte-identical to section 7
V06  marker string set byte-identical to section 7
V07  0xC0 remaining in any @$t2 expression, count = 0
V08  0xC0 remaining in any @$t1 slot-window expression, count = 64 expected, unchanged
V09  corrected entry form present at 0x1f9cd59 and 0x1f9cd72
V10  writer set byte-identical to section 8
V11  no new pseudo-register, $t0..$t19 only
V12  O1-new 'dd @$t1 L1' present at the T1 sites
V13  no '@$t2+0x34000' used as globalEpoch
V14  braces balanced, every .if self-contained
V15  block-closing '}' followed by ';', never by a space
V16  no .else, and no resume command inside a body that must not resume
V17  no @$pc, dwo, poi, dd or dq used as a side effect outside declared reads
V18  every STOP marker of the existing script retained
V19  no source, test, CMake or build modification
V20  the 64 slot dumps present and address-correct, readers[] = $t1 + 0x20
```

V07 and V08 are the pair that distinguishes the two meanings of `0xC0` in this script, so that a
blind find-and-replace cannot pass.

## 13. Not performed

```text
.cdb created or edited   = 0
CDB execution            = 0
AudioEngineHarness       = 0
build                    = 0
production / test / CMake source modification = 0
```

The existing `C3-Redesign-2.cdb` was read and analysed only. Its SHA is unchanged at
`FBE32851F795AF900E6C46D6E38F25138E01217C2B52C0BDDE1F015029CFC39E`. The seven `.cdb` files in
`doc/work113` are untouched. No ConvoPeq reader, epoch, or reclaim state was read or altered.

## 14. Final state

```text
C3-Redesign-2 Correction Design = CLOSED / DESIGN FROZEN / NOT VALIDATED / NOT AUTHORIZED

U2 = CLOSED, FAILED          U3 = CLOSED, SUCCESS
S-14-A = PROVEN             S-14-B = PROVEN
STOP-C3-T1-11 = CLEARED     STOP-C3-T1-12 = CLEARED
U2 finding 4.1 = SUPERSEDED, $t1 modification FORBIDDEN

defect       = @$t2+0xC0 ring base, 7 sites, 2 guards and 5 dumps
correction   = delete the +0xC0 term, at those 7 sites, nothing else
topology     = 13 breakpoints, unchanged
writers      = $t0..$t19, invariant by construction
budget       = unchanged, 0 registers added
O1-new       = dd @$t1 L1, adopted

STOP-C3-CORR-4 = CHARACTERISED, awaiting owner confirmation of the 7-site scope
six others      = CLEARED

.cdb creation      = BLOCKED
Static Validation  = NOT STARTED
Runtime Authorization = NOT STARTED
Execution          = BLOCKED

S7_READER / S7_READER_SLOT = UNRESOLVED
minReaderEpoch for a T1 invocation = NOT CAPTURED
Case A/B/C/D = NOT PROVEN
IMPLEMENTATION = FORBIDDEN
```
