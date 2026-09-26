# r9 Value Capture — Implementation Authorization

```text
gate                = r9 Value Capture / Baseline-2
design gate         = Step5DL, ADJUDICATED
owner ruling        = Extension ADOPTED
mode                = AUTHORIZATION ONLY.  No .cdb created.  No CDB executed.
parent artifact     = P1-5-IR-P2_Step5DI_…_T0-Guard-Repair-Baseline-1.cdb
parent SHA-256      = 74E8C64CD346BA981A1ABD2E1A941B2EDB31E1932A7D586C7D683C1E6EBF9142
parent bytes        = 14,895
product             = T0-Guard-Repair-Baseline-2, to be created under this authorisation
status              = GRANTED, scoped.  Implementation and Static Validation not yet started.
```

## 1. A correction to the design gate, made before the authorisation was written

The design gate proposed reading the live global epoch at `rcx - 0x1428` and named that field
`globalEpoch`. **That name is not source-confirmed, and the source now contradicts part of the
recorded offset algebra.** It must be corrected here, because this document names the address.

### 1.1 What the source establishes

`src/core/EpochDomain.h`:

```text
19  class EpochDomain : public IEpochProvider      -> vptr at +0x00
...
567     struct ReaderSlot
569         std::atomic<uint64_t> epoch
570         std::atomic<uint32_t> depth
571         std::atomic<uint64_t> enterCount
572         std::atomic<uint64_t> residencyStartTimestampUs
574         std::atomic<uint64_t> ownerThreadId
575         char ownerTag[32]
582         std::atomic<uint8_t> quarantineFlags
583     };
585     std::atomic<uint64_t> globalEpoch;                    <- first data member
586     std::array<ReaderSlot, kMaxReaders> readers;
587     DeferredDeletionQueue deferredDeletionQueue;
```

```text
sizeof(ReaderSlot) = 8 + 4 + 4pad + 8 + 8 + 8 + 32 + 1 + 7pad = 80 = 0x50
kMaxReaders = 64,  so sizeof(readers) = 64 x 0x50 = 0x1400

EpochDomain internal layout, PROVEN from the declaration order above:
    +0x0000   vptr
    +0x0008   globalEpoch
    +0x0010   readers[64]        spans +0x0010 .. +0x1410
    +0x1410   padding to the 64-byte alignment of DeferredDeletionQueue
    +0x1440   deferredDeletionQueue
```

The `0x1410` padding is forced: `DeferredDeletionQueue` contains `alignas(64)` members, so its
alignment is 64 and it must start at a 64-aligned offset. `0x1440` is 64-aligned; `0x1410` is not.

### 1.2 The recorded algebra is internally inconsistent by `0x10`

The lineage carries three anchors:

```text
impl this   = $r11 - 0x1440        $r11 = queue address
EpochDomain = $r11 - 0x1430
readers[]   = $t1 + 0x20           $t1 = impl this
```

Combining the second and third with the source layout gives `readers = EpochDomain + 0x10`, which is
consistent. But combining the first with the source gives `queue = EpochDomain + 0x1440`, hence
`EpochDomain = queue - 0x1440`, which contradicts the second anchor by `0x10`.

The contradiction is visible directly in the recorded numbers:

```text
implied sizeof(readers) = 0x1440 - 0x20 = 0x1420 = 5152 bytes
5152 / 64 slots = 80.50 bytes per slot      NOT AN INTEGER
```

`sizeof(readers)` must be `64 x sizeof(ReaderSlot)`, an exact multiple of 64. `0x1420` is not.
Independently, `0x1430` is not a multiple of 64, which the queue's own alignment forbids.

### 1.3 Consequence: two candidate addresses, and the design gate asserted one of them

```text
if EpochDomain = queue - 0x1440   then  globalEpoch = queue - 0x1438
if EpochDomain = queue - 0x1430   then  globalEpoch = queue - 0x1428
```

The design gate asserted `queue - 0x1428` without this analysis. It is one of the two candidates and
it is **not established**. The source fixes the *relative* offset of `globalEpoch` inside
`EpochDomain` at `+0x08` with certainty; it does not fix where `EpochDomain` sits relative to the
queue, because the recorded anchor that would do so is self-contradictory.

## 2. The Extension, revised to identify the field instead of assuming it

Reading a single hardcoded candidate would make the run depend on an unresolved contradiction. The
authorised form reads a **window that contains the field under either candidate**, so the run
*determines* which offset holds it.

```text
AUTHORISED EXTENSION COMMAND     dq @rcx-0x1450 L11
```

```text
window      [rcx - 0x1450 , rcx - 0x1450 + 0x58)      11 quads, 88 bytes

offsets of interest within the window:
    candidate A, globalEpoch at queue-0x1438      -> window + 0x18
    candidate B, globalEpoch at queue-0x1428      -> window + 0x28
    candidate A, readers at queue-0x1430          -> window + 0x20
    candidate B, readers at queue-0x1420          -> window + 0x30
    candidate A, EpochDomain at queue-0x1440      -> window + 0x10
    candidate B, EpochDomain at queue-0x1430      -> window + 0x20
```

All six land inside an `0x58` window, so one command resolves the ambiguity and identifies the field
from data rather than from a contested derivation.

```text
WHY THIS IS BETTER THAN THE DESIGN GATE'S SINGLE ADDRESS
  it removes the dependence on the contradictory anchor entirely
  it yields the identification as a by-product, which the single-address version could not
  it uses one command of an already-proven form, so it adds no new construct class
  the identification criterion is strong: the slot whose value tracks r9 across hits, and
  which advances with the world epoch observed in the harness output
```

Cost, stated plainly: it adds a second new position instance alongside `r`, and roughly four log
lines per hit instead of one. Over 116 hits that is a few hundred lines, which is immaterial against
a 61,500 byte log.

## 3. The authorised payload

At `0x1f9c550` only, inserted ahead of the frozen guard, in this order:

```text
1   .echo C3_DG_T0E_HIT                        proven at this exact position
2   .if (@r9 == 9) { .echo C3_DG_T0E_R9_PASS } ; .else { .echo C3_DG_T0E_R9_FAIL }
                                                   proven at this exact position, Step5DK
3   .if (dwo(@rcx+0x34000) == 4) { .echo C3_DG_T0E_EQ_PASS } ; .else { .echo …EQ_FAIL }
                                                   proven at this exact position, Step5DK
4   r                                           V1, the value capture
5   .echo C3_DG_T0E_EP0                         extension witness, offset 0x00 of the window
6   dq @rcx-0x1450 L11                          the extension window
7   .echo C3_DG_T0E_WIN_END                     closing witness
8   <frozen Baseline-1 guard, byte-identical>
```

Ordering rationale: the three constructs already proven at this position come first, so that if a
later command fails, the boolean verdicts for at least the first hit are already in the log and the
failure is attributable. The two new instances are last before the guard, so the guard's terminal
`gc` is preserved.

The extension witnesses exist because the window is 11 quads of positional data with no inherent
labels; an opening and a closing witness make the block greppable and prevent a partial block from
being read as a complete one.

## 4. Authorised and forbidden

```text
AUTHORISATION = r9 Value Capture / Baseline-2

ALLOW
  creating one new .cdb
  adding bare 'r' at 0x1f9c550
  adding 'dq @rcx-0x1450 L11' and its two .echo witnesses at 0x1f9c550
  one CDB runtime execution

FORBID
  retry, of any kind
  any further change to the T0 repair.  dd( -> dwo( is proven at parse, evaluation and
    comparison and is not to be revisited
  any T1 repair.  0x1f9ce04 and 0x1f9cfb0 stay malformed
  source, test, CMake, build, harness changes
  Dr.Memory
  M1 / M2
  adding any breakpoint or any site
  using '?' in any form.  Excluded by measurement: no command after a completed '?' executes
  using 'r r9'.  A proven command's unproven narrowing form is not admitted
  adding the V7 '.if' decode tree
  changing the guard, its conjuncts, its constants, its operand bases, its nesting or its gc
  changing 0x1f9c57d, 0x1f9c59a, 0x1f9c5da, 0x1f9ce04 or 0x1f9cfb0 in any way
  changing $t0..$t19 meaning, the script preamble, or the topology
```

## 5. Provenance of the two admitted constructs

```text
'r'  PROVEN in a breakpoint body, Step5BB Retry-1.  Executed, printed 8 lines including
        r9=0000000000000009, and EIGHT further commands ran after it, ending at
        C3_T0_SLOTS_BEGIN.  Continuation is therefore proven, which is exactly the property
        '?' lacks.
      NOT proven at the T0E payload position.  Position is the admitted risk.

'dq <expr> L<n>'  PROVEN in a breakpoint body, Step5BB Retry-1.  dq @rsp L1, dq @rsp+0x30 L2
        and dq @$t1+0x18 L1 all executed, each followed by further commands.  The output at
        @$t1+0x18 was 00000000`00000009, the observation that motivated this extension.
      NOT proven at the T0E payload position, and the address is new.  Second admitted risk.
```

Both are display-only and assign nothing. Neither can alter debuggee state.

## 6. Baselines

```text
Baseline-1   74E8C64C…F9142   14,895 B   FROZEN, retained unmodified
Baseline-2   the product of this authorisation.  Supersedes Baseline-1 for measurement
             purposes because the T0E body changes.  Its guard must be byte-identical to
             Baseline-1's guard, and that is the primary static check.
```

## 7. Static Validation, to follow implementation

Read-only, no debugger. The required set will mirror S-T0-01 to S-T0-17 and add:

```text
  the guard at 0x1f9c550 is byte-identical to Baseline-1's guard
  exactly three payload regions are added and nothing else changes:
      bare 'r', and the extension block '.echo ; dq ; .echo'
  'r' occurs exactly once in the file, at 0x1f9c550
  'dq @rcx-0x1450 L11' occurs exactly once, at 0x1f9c550
  no '?' anywhere
  no 'r r9' anywhere
  no 'poi(' count change
  every other line byte-identical to Baseline-1
  topology unchanged, 13 RVAs in identical order
  reversibility: deleting the added payload regions reproduces Baseline-1 byte for byte
  size delta equals exactly the added payload bytes
```

## 8. Runtime, on the terms already fixed

```text
execution_count = exactly 1
Retry            = 0
```

```text
R1  SCRIPT_BEGIN=1  SCRIPT_END=1  ZwTerminateProcess=1  quit=1  exit=0  debugger errors=0
    FAIL -> stop, attribute, close, no R2 to R6, no retry.
R2  T0E hit count
R3  raw r9 distribution, from the 'r' output
R4  r9 output paired to the T0E hit it belongs to
R5  the three relations, per hit, if the extension is present:
        r9 == value at window+0x18
        r9 == value at window+0x28
        r9 == 9
    plus, as an identification result rather than an assumption:
        which window slot tracks r9, and whether that slot advances with the world epoch
R6  T0E -> T0P -> T0Q -> T0S ordering
```

### 8.1 What the run may not conclude

```text
NOT  that 9 is wrong.  The run measures relations, not intent.
NOT  that the identified slot IS globalEpoch, unless the identification criterion in R5 is
     met, namely the slot tracks r9 across hits and advances with the world epoch.
NOT  any raw enqueuePos.  This gate captures epoch, not enqueuePos.
NOT  anything about T1.  0x1f9ce04 and 0x1f9cfb0 remain malformed and were not reached.
NOT  that the captured r9 distribution is invariant.  One configuration, one run.
```

### 8.2 The order that follows, fixed in advance

```text
Capture Result
      |
      v
Intent / Semantic Audit          <- decides what the predicate was meant to be
      |
      v
predicate validity determined
      |
      v
Repair Design Gate
      |
      v
separate Implementation Authorization
```

The capture result does **not** authorise changing the T0 predicate. That is the whole point of the
separation.

## 9. State

```text
design gate            CLOSED / ADJUDICATED
owner ruling           Extension ADOPTED, revised form below
parent                 Baseline-1, 74E8C64C…F9142, 14,895 B
product                Baseline-2, NOT YET CREATED
CDB execution          0
.cdb created           0
source/test/CMake/build/harness   untouched
T1                     untouched, NOT AUTHORIZED
IMPLEMENTATION         authorised, scope as above
RUNTIME                NOT AUTHORIZED, requested separately after Static Validation
```

## 10. Next

Implement the payload in section 3, run the Static Validation set in section 7, then stop and request
Runtime Authorization. Runtime is a separate gate and is not covered by this document.
