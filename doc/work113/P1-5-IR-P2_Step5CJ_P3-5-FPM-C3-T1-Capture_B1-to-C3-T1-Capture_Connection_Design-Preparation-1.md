# P3-5-FPM-C3-T1-Capture — B1 to C3 T1 Capture Connection Design-Preparation-1

## 1. Purpose

This gate specifies how the proven B1 anchor-address primitive connects to the existing C3 T1
capture. It creates no script, runs nothing, and changes no source.

```text
CDB execution = 0     ping execution = 0
.cdb creation = 0     runtime authorization = 0
production / test / CMake source = 0     build = 0
ReaderSlot / EpochDomain / getMinReaderEpoch / DQueue runtime access = 0
```

The design is additive to C3-Redesign-2. That design's breakpoint topology, register roles,
correlation guards and stop markers are retained. What this gate fixes is **where the B1 primitive is
inserted**, and the capture contract for the values the C3 chain must correlate.

The design deliberately does **not** port the `ping` result. B1 was proven on a benign target; it
contributes a method, not an address.

## 2. Frozen source authority

```text
ConvoPeq.md SHA-256 = E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
ConvoPeq.md bytes   = 5,535,334
```

`ConvoPeq.md` is the only production source authority. The superseded `ConvoPeq(3).md` was not used
and is not present. Every structural claim in sections 4, 6 and 7 below cites a line number in this
file.

## 3. Proven B1 primitive

```text
B1 = PROVEN, Result Audit-1, log 2797F5F4...28193F

  image base = @$bpN - (breakpoint RVA)
```

Established properties, reusable as-is:

```text
@$bpN is valid inside a breakpoint command body on this CDB build
the subtraction recovers the module image base exactly
module base + RVA and @$bpN - RVA agree across five independent anchors
the two derivations use different inputs, so agreement is evidence, not tautology
```

Established CDB command-body fact, also reusable:

```text
"}" followed by a space   does NOT act as a command boundary
"}" followed by ";"       DOES act as a command boundary
```

This is a property of CDB 10.0.29617.1000 command-body parsing, established across three executions
whose bodies differed in length and in the construct following the brace. Every multi-block command
string in the C3 script must use `} ;` as its block separator.

## 4. Existing C3 breakpoint topology, retained unchanged

From `P1-5-IR-P2_Step5BB_..._C3-T1-Capture-Redesign-2.cdb`. These are preserved:

| RVA | marker | role |
|---|---|---|
| `0x1f9c550` | `C3_T0_ENTRY` | T0 target arming |
| `0x1f9c57d` | `C3_T0_CAS_PRE` | T0 CAS pre |
| `0x1f9c59a` | `C3_T0_CAS_POST` | T0 CAS post |
| `0x1f9c5da` | `C3_T0_SEQUENCE_PUBLISHED` | arms the candidate chain, captures `$t0..$t19` |
| `0x1fa80b3` | `C3_TERMINAL_S7_ANCHOR` | terminal S7 anchor |
| `0x1f9cfb0` | `C3_T1A_CANDIDATE` | first relevant `tryReclaim` |
| `0x1f9cfe2` | `C3_T1B_GETMIN_CALL_PRE` | ReaderSlot snapshot, pre-call |
| `0x1f9cfe5` | `C3_T1B_GETMIN_RETURN` | `minReaderEpoch` from RAX |
| `0x1f9cd00` | `C3_T1C_DQUEUE_ENTRY` | DQueue entry correlation |
| `0x1f9cd59` | `C3_T1_SELECTION` | **the target epoch gate** |
| `0x1f9cd72` | `C3_T1_CAS_PRE` | T1 CAS pre |
| `0x1f9ce04` | `C3_T1_CAS_SUCCEEDED` / `C3_T1_EPOCH_BLOCKED` | T1 CAS post |
| `0x1fa80b2` | `C3_T2_WAIT_RETURN` | T2 captured |

### 4.1 Source correspondence of the chain

The topology is not arbitrary; it maps onto the source as follows.

```text
EpochDomain::tryReclaim()                     ConvoPeq.md:57500
  deferredDeletionQueue.reclaim(getMinReaderEpoch())        :57508
    getMinReaderEpoch() is INLINED at the call site
DeferredDeletionQueue::reclaim()              :62027
  head = dequeuePos                            :62032
  entry = ringBuffer[scanPos & kMask]         :62046
  if (isOlder(entry.epoch, minReaderEpoch))   :62049   <- the epoch gate
```

Consequences the design must respect:

```text
the getMinReaderEpoch "pre" and "return" breakpoints are inside the INLINED body
  within EpochDomain::tryReclaim, not in a standalone function frame
the epoch gate at 0x1f9cd59 is inside DeferredDeletionQueue::reclaim
the gate compares entry.epoch against the minReaderEpoch value computed at :57508
```

## 5. B1 insertion point, and the anchor sanity check

### 5.1 Where B1 is used

B1 is used for exactly one purpose: recovering the `AudioEngineHarness` image base from a runtime
breakpoint address, so that the script can confirm its own symbol-derived addresses.

```text
at C3_T1B_GETMIN_CALL_PRE, the target breakpoint is known to have been hit
  $t11_style base recovery = @rbx - (RVA of 0x1f9cfe2)
```

This does not replace any existing address computation. C3-Redesign-2 addresses the module with the
symbol `AudioEngineHarness+RVA`; that stays. B1 supplies an independent second derivation for
comparison.

### 5.2 The A == B anchor sanity check

Per the B1 pattern, the design requires the following cross-check as a **script self-check**:

```text
A = AudioEngineHarness + RVA        (module symbol, resolved at script load)
B = @$bpN - RVA                     (runtime breakpoint address, minus known RVA)
require A == B for every breakpoint the script arms
```

This is a guard against symbol/base drift, and it is a direct transfer of the proven B1 pattern.

**Boundary, stated explicitly:** this check yields **no** reader, epoch, or reclaim evidence. It is
address arithmetic only. It may not be cited toward `S7_READER`, toward `minReaderEpoch`, or toward
any Case A/B/C/D classification. A failure of this check is a script-integrity failure, not a
runtime finding about the epoch gate.

### 5.3 What B1 must not be used for

```text
B1 must not be used to discover an RVA          RVA is a source/build fact
B1 must not be used to locate ReaderSlot        slot base is a layout fact
B1 must not be used to identify S7_READER       unrelated to addressing
```

## 6. ReaderSlot capture schema, taken from source

Every field below is read from `ConvoPeq.md`. No field is inferred, and no semantics are invented.

### 6.1 The struct as declared

`ConvoPeq.md:57681-57697`

```cpp
struct ReaderSlot
{
    std::atomic<uint64_t> epoch { kInactiveEpoch };                    // +0x00
    std::atomic<uint32_t> depth { 0 };                                 // +0x08
    std::atomic<uint64_t> enterCount { 0 };                            // +0x10
    std::atomic<uint64_t> residencyStartTimestampUs { 0 };             // +0x18
    std::atomic<uint64_t> ownerThreadId { 0 };                         // +0x20
    char ownerTag[32] {};                                               // +0x28
    static constexpr uint8_t kQuarantinedFlag      = 0x01;
    static constexpr uint8_t kPendingQuarantineFlag = 0x02;
    std::atomic<uint8_t> quarantineFlags{0};                           // +0x48
};
```

| field | offset | width | meaning per source |
|---|---|---|---|
| `epoch` | `+0x00` | 8 | the reader's published epoch |
| `depth` | `+0x08` | 4 | nesting depth; `0` means not inside a reader section |
| `enterCount` | `+0x10` | 8 | cumulative enter count, comment: " enter count only" |
| `residencyStartTimestampUs` | `+0x18` | 8 | steady-clock entry timestamp |
| `ownerThreadId` | `+0x20` | 8 | hash of `std::thread::id` |
| `ownerTag` | `+0x28` | 32 | owner label, e.g. `"AudioThread"`, source notes stale reads are tolerated |
| `quarantineFlags` | `+0x48` | 1 | bit0 quarantined, bit1 pending quarantine |

Derived layout facts:

```text
sizeof(ReaderSlot) = 0x49, aligned to 8  ->  stride = 0x50     matches Redesign-2
kMaxReaders = 64                                                  :57136
```

### 6.2 Sentinel and flag values

`ConvoPeq.md:57137-57138`, `:57694-57695`

```text
kInactiveEpoch      = 0xFFFFFFFFFFFFFFFF   (uint64 max)
kReservedEpoch      = 0xFFFFFFFFFFFFFFFE   (max - 1)
kQuarantinedFlag    = 0x01
kPendingQuarantineFlag = 0x02
```

### 6.3 Capture requirement per slot

For all 64 slots the design requires these fields be captured, at the offsets in 6.1:

```text
slot index
quarantineFlags                       raw byte, and decoded bit0 / bit1
depth                                 raw u32
epoch                                 raw u64
ownerThreadId                         raw u64
ownerTag                              32 bytes
enterCount                            raw u64
residencyStartTimestampUs             raw u64
```

Redesign-2 emits raw `db` byte windows per slot. That is retained as the lossless record, and the
field-addressed reads are **added** so that the eligibility test in section 7 can be evaluated
without manual hex decoding.

No other field is captured, and no field's meaning is extended beyond what the source states.

### 6.4 Base address of the slot array: an open provenance item

```text
relative layout within a slot   PROVEN   from :57681-57697
stride 0x50 and count 64        PROVEN   from :57700 and :57136
absolute base of readers[]      ASSUMED  Redesign-2 uses @$t1+0x20 with $t1 = @r11-0x1440,
                                         i.e. base = domain + (-0x1420)
```

The absolute base is **not** derivable from the source text and is not proven. Section 12 defines
`STOP-C3-T1-11` for it. The design does not invent a value; it requires provenance confirmation
before the capture is trusted.

## 7. minReaderEpoch correlation rule

This is the semantic core of the capture and it is taken directly from the source.

### 7.1 The algorithm as implemented

`ConvoPeq.md:57328-57362`

```cpp
uint64_t minEpoch = currentEpoch();                 // == globalEpoch, :57300-57304
for (const auto& slot : readers) {
    const uint8_t flags = slot.quarantineFlags;
    if (flags & kQuarantinedFlag) continue;         // quarantined excluded
    const uint32_t depth = slot.depth;
    if (depth == 0) continue;                       // inactive excluded
    const uint64_t epoch = slot.epoch;
    if (epoch == kInactiveEpoch || epoch == kReservedEpoch) continue;
    if (isOlder(epoch, minEpoch)) minEpoch = epoch; // strictly downward only
}
return minEpoch;
```

### 7.2 Eligibility predicate

A slot contributes to `minReaderEpoch` **only** if all four hold:

```text
E1  (quarantineFlags & 0x01) == 0
E2  depth != 0
E3  epoch != 0xFFFFFFFFFFFFFFFF
E4  epoch != 0xFFFFFFFFFFFFFFFE
```

### 7.3 The `active` divergence, which the design must not miss

`ReaderSlotDetail::active` is computed as `depth > 0` at `ConvoPeq.md:57569`. That is **not** the
same predicate as 7.2. A slot can have `depth > 0` and still be excluded, by quarantine or by a
sentinel epoch.

```text
ReaderSlotDetail::active  = (depth > 0)
getMinReaderEpoch eligible = E1 and E2 and E3 and E4
```

The capture and the correlation must use **7.2**. Using `active` would silently over-count eligible
readers and could produce a wrong `minReaderEpoch` reconstruction. This divergence is a design
finding, not a source defect.

### 7.4 The correlation rule

The Result Audit must perform this check, and it is a STOP condition when it fails:

```text
R1  capture globalEpoch at the same instant as the slot snapshot
R2  m = globalEpoch
R3  for i in 0..63:  if eligible(i) and isOlder(epoch_i, m):  m = epoch_i
R4  require m == the RAX value captured at C3_T1B_GETMIN_RETURN
```

`isOlder(a, b)` is `static_cast<int64_t>(a - b) < 0`, `ConvoPeq.md:62174-62177`. The audit must use
that definition, not a plain `<`.

If `R4` fails, the snapshot and the returned value are inconsistent, which is Case D.

## 8. S7 reader identity classification

Four cases, kept strictly distinct. The classification applies **only** when the capture has passed
sections 5, 6 and 7.

### Case A, reader identity candidate

```text
minReaderEpoch < globalEpoch
AND exactly one slot satisfies E1..E4 with epoch == minReaderEpoch
-> that slot is a reader identity CANDIDATE
```

A candidate is not yet a proof. It becomes `S7_READER = PROVEN` only under section 8.5.

### Case B, minimum not unique

```text
two or more slots satisfy E1..E4 with epoch == minReaderEpoch
-> the epoch value alone cannot identify the reader
-> S7_READER_SLOT = UNRESOLVED
```

This is `STOP-C3-T1-5`. The capture is still valid evidence; the identity is simply not determined.

### Case C, no discriminating evidence, and why it must not be over-read

```text
minReaderEpoch == globalEpoch
```

From 7.1, `minEpoch` is **initialised to `globalEpoch`** and only ever moves downward. Therefore:

```text
minReaderEpoch == globalEpoch
  <=>  no eligible slot lowered the value
```

That is a proof about the scan, not about the existence of a reader. The following remain
indistinguishable from this evidence alone:

```text
no reader was inside a reader section at that instant       (all depth == 0)
readers existed but all were quarantined                    (bit0 set)
readers existed but all held a sentinel epoch               (E3 / E4)
a reader existed exactly at globalEpoch                     (isOlder false, so not lowered)
the snapshot and the scan were separated in time
```

This matches the recorded incident condition

```text
head epoch = 9, globalEpoch = 9, minReaderEpoch = 9, epoch gate blocked
```

which is precisely the signature of "no eligible reader lowered the minimum".

**Design rule, binding:**

```text
minReaderEpoch == globalEpoch  MUST NOT be reported as S7_READER = none
minReaderEpoch == globalEpoch  MUST NOT be reported as a reader identity
minReaderEpoch == globalEpoch  MUST NOT be used to name a slot
```

It is recorded as `S7_READER = UNRESOLVED`, with the discriminating experiments named in 8.5, and
the run is stopped under `STOP-C3-T1-6`.

### Case D, capture inconsistency

```text
the R4 recomputation does not equal the returned RAX value
OR the slot snapshot is incomplete
OR slot fields change between snapshot and gate within the same invocation
-> capture is inconsistent
-> no S7 conclusion may be drawn
```

This is a defect in the capture, not a finding about the epoch gate.

### 8.5 What `S7_READER = PROVEN` additionally requires

Separate from T1 capture success, per the instruction that the two must not be conflated:

```text
G1  the capture passed sections 5, 6 and 7 with no STOP
G2  Case A holds, with a unique eligible slot
G3  that slot's ownerThreadId and ownerTag are non-default
G4  the same slot is observed as the minimum at a second, independent epoch gate
G5  the reclaim decision at the gate is consistent with that slot's epoch under isOlder
```

`G4` is what separates a candidate from a proof. A single observation is consistent with coincidence.

## 9. T1 uniqueness rule

T1 is defined by instruction identity and correlation, never by a value.

```text
T1 = the invocation that
       was admitted as the single candidate at C3_T1A_CANDIDATE,
       captured minReaderEpoch at C3_T1B_GETMIN_RETURN,
       and reached the target epoch gate at 0x1f9cd59
       on the same thread token, with the same target DQueue entry,
       with entry identity equal to the T0-published ticket.
```

Binding conditions, all of which must hold:

```text
U1  exactly one candidate invocation admitted          else STOP-C3-T1-1
U2  @$tid identical at T1A, T1B and T1
U3  @rdi == domain object at T1C and at the gate
U4  the DQueue entry address equals ring base + (index * entry stride)
U5  the entry sequence number equals the T0-published sequence
U6  entry ptr, deleter, epoch, publicationSequenceId, generation all equal the T0 ticket
U7  @$r13 (minReaderEpoch) equals the RAX value captured at T1B
U8  the gate is reached once; a second selection is STOP-C3-T1-2
```

Redesign-2 already encodes U1 through U8 as nested `.if` guards. They are retained verbatim.

`entry stride` is 0x30 in Redesign-2's guard `@rsi == (@$t2+0xc0+((@$t10&0xfff)*0x30))`. The
declared `DeletionEntry` is `ConvoPeq.md:61945-61955`:

```cpp
struct DeletionEntry {
    void* ptr;  void (*deleter)(void*);  uint64_t epoch;  DeletionEntryType type;
    uint64_t publicationSequenceId;  uint64_t generation;
    size_t objectBytes;   // only under CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
};
```

The stride is **build-configuration dependent** because of the conditional member. A 3-entry stride
of 0x30 implies the diagnostics field is absent in the captured build. This must be confirmed, not
assumed; see `STOP-C3-T1-12`.

## 10. Pseudo-register allocation

Redesign-2's allocation is retained exactly. `$t0..$t19` only; `$s7`, `$s7e`, `$s7q`, `$s7rbx` and
all other user pseudo-registers remain excluded.

| reg | role, unchanged |
|---|---|
| `$t0` | T0 sequence-published, chain armed |
| `$t1` | slot-region base, `@r11 - 0x1440` |
| `$t2` | EpochDomain object pointer |
| `$t3` | target-domain comparison address, `$t1 + 0x10` |
| `$t4`,`$t5` | `publicationSequenceId` low / high |
| `$t6`,`$t7` | `generation` low / high |
| `$t8`,`$t9` | target `epoch` low / high |
| `$t10` | DQueue sequence number |
| `$t11` | candidate thread token |
| `$t12` | candidate admitted flag |
| `$t13` | candidate admitted counter |
| `$t14`,`$t15` | same-invocation `minReaderEpoch` low / high |
| `$t16` | T1 selection uniqueness |
| `$t17` | T2 captured |
| `$t18` | anomaly |
| `$t19` | S7 terminal-thread diagnostic |

### 10.1 The allocation is full, and that is a binding constraint

```text
20 registers, 20 roles.  No register is free.
```

`globalEpoch` is required by section 7.4 rule `R1`, and it has no register. The design therefore
does **not** silently repurpose one. Two admissible options are carried forward for Preparation-1:

```text
O1  capture globalEpoch from the log, at the existing "dd @$t2+0x34000 L1" dump that
    Redesign-2 already emits at every marker.  Requires no register change.
O2  free a register by retiring a role that is provably complete before first use.
    Requires a proof that no later breakpoint writes it.
```

`O1` is preferred because it changes no established role. **Both options require the offset
provenance of section 6.4; `+0x34000` is not proven to be `globalEpoch`.** Whichever option is
taken, the choice and its justification belong in Preparation-1, and any register whose writer set
is not provably single-valued is `STOP-C3-T1-9`.

## 11. Marker and state machine

Markers are retained from Redesign-2. State transitions:

```text
S_IDLE
  -> S_T0_ARMED            on C3_T0_SEQUENCE_PUBLISHED, sets $t0 = 1, captures ticket
S_T0_ARMED
  -> S_CANDIDATE           on C3_T1A_CANDIDATE, sets $t11, $t12 = 1
  -> STOP-C3-T1-1          on a second admitted candidate
S_CANDIDATE
  -> S_GETMIN_PRE          on C3_T1B_GETMIN_CALL_PRE, snapshot of 64 slots
  -> STOP-C3-T1-3          on thread or domain token mismatch
S_GETMIN_PRE
  -> S_GETMIN_RET          on C3_T1B_GETMIN_RETURN, RAX into $t14 / $t15
  -> STOP-C3-T1-3          on thread or domain token mismatch
S_GETMIN_RET
  -> S_DQUEUE              on C3_T1C_DQUEUE_ENTRY, entry join
  -> STOP-C3-T1-8          on entry join failure
S_DQUEUE
  -> S_T1_SELECTED         on C3_T1_SELECTION, sets $t16 = 1
  -> STOP-C3-T1-2          on a second selection
  -> STOP-C3-T1-7          on a gate hit by a non-candidate invocation
S_T1_SELECTED
  -> S_T1_CAS_PRE          on C3_T1_CAS_PRE
  -> S_T1_SETTLED          on C3_T1_CAS_SUCCEEDED or C3_T1_EPOCH_BLOCKED
S_T1_SETTLED
  -> S_DONE                script tail, C3_T1_REDESIGN2_SCRIPT_END
```

Retained terminal markers, unchanged in text: `C3_CR1_STOP_NESTED_OR_CONCURRENT_CANDIDATE`,
`C3_CR1_STOP_T1B_THREAD_TOKEN_JOIN`, `C3_CR1_STOP_TARGET_DOMAIN_MISMATCH`,
`C3_CR1_STOP_CANDIDATE_THREAD_MISMATCH`, `C3_CR1_STOP_T1C_DQUEUE_ENTRY_JOIN`,
`C3_CR1_STOP_T1C_THREAD_TOKEN_R13_JOIN`, `C3_CR1_STOP_TARGET_IDENTITY_CONFLICT`,
`C3_CR1_STOP_MULTIPLE_TARGET_SELECTION`, `C3_CR1_STOP_T1_CAS_CORRELATION`,
`C3_CR1_STOP_T1_RETURN_POSITION_CONTRADICTION`, `C3_CR1_STOP_T1_RETURN_THREAD_DOMAIN`,
`C3_PREDECESSOR_HEAD`, `C3_NON_TARGET_HEAD`, `C3_CANDIDATE_NO_TARGET`.

## 12. STOP conditions

| id | condition |
|---|---|
| STOP-C3-T1-1 | multiple candidate invocations admitted |
| STOP-C3-T1-2 | same target sequence but a different target identity, or a second T1 selection |
| STOP-C3-T1-3 | `minReaderEpoch` cannot be correlated to the same invocation |
| STOP-C3-T1-4 | ReaderSlot snapshot incomplete, fewer than 64 slots, or a slot unreadable |
| STOP-C3-T1-5 | multiple equally eligible minimum readers, Case B |
| STOP-C3-T1-6 | `minReaderEpoch == globalEpoch` without discriminating evidence, Case C |
| STOP-C3-T1-7 | T1 gate reached by an invocation other than the admitted candidate |
| STOP-C3-T1-8 | target DQueue identity mismatch |
| STOP-C3-T1-9 | pseudo-register writer ambiguity, a register with more than one writer |
| STOP-C3-T1-10 | CDB parser ambiguity, or marker ambiguity between echoed and emitted text |
| STOP-C3-T1-11 | **added**: the absolute base of `readers[]` is not confirmed by provenance |
| STOP-C3-T1-12 | **added**: the `DeletionEntry` stride is not confirmed for this build configuration |

STOP-C3-T1-11 and -12 are additions of this gate. Both concern addresses inferred by Redesign-2
rather than read from source, and neither may be waved through.

### 12.1 STOP-C3-T1-6 is the one that caused the original ambiguity

It is restated as a rule rather than a label:

```text
head epoch = 9, globalEpoch = 9, minReaderEpoch = 9, epoch gate blocked
```

From section 7.1 this is the signature of "no eligible reader lowered the minimum". The gate blocks,
the reclaim is correctly conservative, and **no reader can be named**. Any future run producing this
pattern must stop with `S7_READER = UNRESOLVED` and must not convert it into a reader identity, into
`S7_READER = none`, or into a slot index.

## 13. Forbidden operations

```text
.cdb creation in this gate                        FORBIDDEN
CDB / ping execution                              FORBIDDEN
runtime authorization                             FORBIDDEN
production / test / CMake source modification     FORBIDDEN
build                                             FORBIDDEN
AudioEngineHarness execution                      FORBIDDEN
production C3 execution                           FORBIDDEN
T1 capture outside an authorized single run       FORBIDDEN
ReaderSlot / EpochDomain runtime access           FORBIDDEN
getMinReaderEpoch call                            FORBIDDEN
DQueue / reclaim runtime observation              FORBIDDEN
M1 / M2                                           FORBIDDEN
Retry-3, Preparation-4                            FORBIDDEN
Dr.Memory                                         FORBIDDEN
```

## 14. Static validation requirements

Preparation-1 must be validated on all of the following before any authorization is considered.

```text
S-01  every breakpoint RVA is byte-identical to the Redesign-2 table in section 4
S-02  every register role in section 10 is byte-identical to Redesign-2
S-03  no pseudo-register outside $t0..$t19 appears
S-04  every register has exactly one writer per phase, or the ambiguity is resolved and justified
S-05  no block-closing "}" is followed by a space instead of ";"
S-06  no ".else", "q", "gc", "g" or other resume command inside a body that must not resume
S-07  no "@$pc", "dwo", "poi", "dd"-as-side-effect outside declared reads
S-08  brace balance, and every .if block self-contained
S-09  all 64 slot field reads present, at the offsets in 6.1
S-10  kInactiveEpoch and kReservedEpoch literals match 6.2
S-11  the isOlder definition used in any recomputation matches :62174-62177
S-12  the A == B anchor check is present and is documented as non-evidence
S-13  every STOP id in section 12 appears with its marker
S-14  the offset provenance of 6.4 and the stride provenance of section 9 are resolved
S-15  marker strings are unique and distinguishable from echoed command text
S-16  the script contains no comment-only line that CDB would reject
```

S-14 is the gate that Redesign-2 did not have.

## 15. Authorization requirements

A future authorization must fix, and re-verify immediately before the run:

```text
script path and SHA-256, and byte count
AudioEngineHarness.exe path, SHA-256, and the build identity the RVAs belong to
cdb.exe SHA-256 and version
ConvoPeq.md SHA-256 and byte count
process residue = 0
reserved log absent
exactly one CDB execution
exactly one AudioEngineHarness execution
rerun forbidden
```

The build identity item is an addition. The RVAs in section 4 are valid only for the exact
`AudioEngineHarness` binary they were derived from, and the authorization must bind them.

## 16. Expected evidence

On a successful run the log must contain, as emitted output rather than echoed commands:

```text
C3_T0_ENTRY, C3_T0_CAS_PRE, C3_T0_CAS_POST, C3_T0_SEQUENCE_PUBLISHED
C3_T1A_CANDIDATE
C3_T1B_GETMIN_CALL_PRE, 64 slot field reads, C3_T1B_SLOTS_END
C3_T1B_GETMIN_RETURN with $t14 / $t15
C3_T1C_DQUEUE_ENTRY
C3_T1_SELECTION with the full ticket comparison
C3_T1_CAS_PRE
C3_T1_CAS_SUCCEEDED or C3_T1_EPOCH_BLOCKED
C3_T2_WAIT_RETURN
```

and the audit must be able to state, from that log alone:

```text
the thread token that was admitted
the globalEpoch at the snapshot instant
the 64 slot records with eligibility per E1..E4
the reconstructed minReaderEpoch, and its equality to the RAX value
the DQueue entry identity, and its equality to the T0 ticket
which Case, A through D, the run falls into
S7_READER and S7_READER_SLOT, or UNRESOLVED with the reason
```

Marker counting follows the rule proven necessary in the B1 chain: only a line whose entire trimmed
content is the marker counts as emitted. The `bp` line, the `bl` listing and any error message repeat
every marker string and are not evidence.

## 17. No-runtime, no-source-change declaration

```text
CDB execution = 0        ping execution = 0
.cdb created  = 0        runtime authorization = 0
production / test / CMake source = 0        build = 0
ReaderSlot / EpochDomain / getMinReaderEpoch / DQueue runtime access = 0
```

This gate produced exactly one artifact, this document. The existing C3-Redesign-2 script, the six
earlier `.cdb` files and the seven `.cdb` files in `doc/work113` are untouched, and no ConvoPeq
reader, epoch, or reclaim evidence was produced or altered.

## 18. Base state and next gate

```text
DIRECT_DQUEUE_CAUSE = epoch gate equality PROVEN
Candidate A         = PROVEN
@$bp0               = PROVEN
$t11                = PROVEN
T1 agreement        = PROVEN
B1                  = PROVEN
CDB block-separator fact = PROVEN

S7_READER          = UNRESOLVED
S7_READER_SLOT     = UNRESOLVED
minReaderEpoch     = NOT CAPTURED
C3 T1 capture      = NOT CAPTURED
Case A/B/C/D       = NOT PROVEN
IMPLEMENTATION     = FORBIDDEN
```

```text
Design-Preparation-1                CLOSED   (this gate)
Preparation-1, script created      NOT STARTED
Static Validation                   NOT STARTED
Runtime Authorization               NOT STARTED
Benign / production execution       NOT STARTED
Result Audit                        NOT STARTED
```

Preparation-1 is where a `.cdb` is first created. It must resolve S-14, the two provenance items this
gate added, before the script is considered complete.
