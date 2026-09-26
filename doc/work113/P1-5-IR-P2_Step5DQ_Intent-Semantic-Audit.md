# Intent / Semantic Audit — T0 guard conjunct 2

```text
gate                = Intent / Semantic Audit, IA-1 .. IA-5
mode                = READ-ONLY.  No CDB run, no retry, no .cdb edit, no source/test/CMake/build.
primary source      = ConvoPeq.md, 5,535,334 B, SHA-256 E5E74200F12784FDF37BE24864F5E4B1…
                      and the live src/ tree.  ConvoPeq(4).md does not exist in this repository
                      and was not treated as equivalent to anything.
inputs              = the Step5DP Capture Result, the archived gate records of doc/work113,
                      src/core/EpochDomain.h, src/audioengine/ISRRetireRouter.cpp,
                      src/audioengine/AudioEngine.Retire.cpp, src/audioengine/AudioEngine.h
execution this gate = 0
verdict             = conjunct 2 INVALID as a production-semantic assertion;
                      INTENT-AMBIGUOUS as to which role the author intended
```

## 1. Scope / Authorization Boundary

In scope: determine what `@r9 == 9` was intended to compare, from implementation, design records and
history.

```text
NOT in scope   proposing or applying any repair
NOT in scope   writing a replacement predicate, including any comparison against globalEpoch
NOT in scope   any further runtime measurement.  Runtime is closed and consumed.
NOT in scope   T1 repair, S7_READER, minReaderEpoch, Case A-D
```

The design gate, the implementation authorisation, the runtime request and the Result Audit each
reserved this determination. This gate is where it is made, and it is made without a repair.

## 2. Evidence Frozen

```text
PROVEN   r9 at 0x1f9c550 is the 3rd declared parameter of DeferredDeletionQueue::enqueue,
         stored by the callee to entry.epoch.  Disassembly plus source signature, all six
         arguments and all six store offsets agreeing.  Step5DL.
PROVEN   no instruction writes r9 in 0x1f9c550..0x1f9c5e7.  Its only appearance is the store
         to entry.epoch.  Step5DL.
PROVEN   [rcx - 0x1428] is the live global epoch.  Three independent anchors: a vtable pointer
         at window+0x20, a value that tracks r9 at window+0x28, and the kInactiveEpoch /
         kReservedEpoch value set at window+0x30.  Step5DP.
MEASURED enqueue epoch equals the live global epoch, or trails it by exactly one.  delta in
         {0, +1} over 116 of 116 hits.  Step5DP.
MEASURED enqueue epoch took 13 distinct values at 116 T0E hits and never equalled 9.  Step5DP.
MEASURED 0 of 116 in each of the two --measurement=normal runs, so 0 of 232.
```

## 3. Original Predicate Provenance

The lineage's own records contain the provenance, so it is cited rather than re-derived.

### 3.1 The literals are an observation table

`Step5BB-Retry-1`, lines 110 to 124, is a table of values **observed at the one hit where the gate
opened**:

```text
caller return address = 0x00007ff759a8c735
DQueue this ($t2)     = 0x000001764737ab80
epochBase ($t1)       = 0x0000017647379740
EpochDomain ($t3)     = 0x0000017647379750
entry ptr (rdx)       = 0x0000017647655a01080
entry deleter (r8)    = AudioEngineHarness+0x1f9a79c0
entry epoch (r9)      = 9
entry type            = 0
publicationSequenceId = 0
generation            = 0
globalEpoch           = 9
enqueuePos            = 4
sequence[4]           = 4
```

The guard's literals are exactly the values in that table: `9` for the epoch, `4` for `enqueuePos`,
`4` for `sequence[4]`.

### 3.2 Two predecessor gates already recorded that the constants have no basis

```text
Step5CR L185   '@r9 == 9', '@r10 == 4', '@r10 == 5' and '@rbx == 4' are hardcoded magic
               constants.  Nothing in this run provides evidence for or against their
               correctness, and this audit does not assert anything about them.

Step5CS L379   '@r9 == 9 has no basis in the disassembly; asserted neither way'
```

This audit does not re-derive those findings. It inherits them and adds what has become available
since: the epoch's meaning, and the live relation.

### 3.3 No constant 9 exists in the epoch machinery

```text
src/core/EpochDomain.h:23   kInactiveEpoch = numeric_limits<uint64_t>::max()
src/core/EpochDomain.h:24   kReservedEpoch = numeric_limits<uint64_t>::max() - 1
src/core/EpochDomain.h:26   EpochDomain() : globalEpoch(1)
                             globalEpoch advances only through publishEpoch()
```

There is no protocol constant 9 for the epoch. The epoch is initialised at 1 and advances.

```text
IA-1  the 9 has no design derivation.
      Its only traceable origin is the Retry-1 observation table, i.e. an observed runtime
      value transcribed into a condition.  It was a snapshot, not a specified constant.
```

## 4. Meaning of the Literal 9

```text
9 is the value of the live global epoch at the single hit where the gate opened, observed
under the run flag --fpm-m0.

The two subsequent --measurement=normal runs gave enqueue epochs drawn from
  { 1, 2, 5, 6, 16, 19, 22, 25, 28, 31, 34, 37, 48 }
and 9 was not among them at any of 232 T0E hits.
```

Two facts must be kept apart:

```text
the epoch is a moving counter and DOES pass through 9 at some point in a run
  the progression observed was 1 -> 2 -> 3 and on to 48

conjunct 2 is evaluated only at T0E hits, and 9 was not the epoch at any of them
```

So the finding is **not** "9 is unreachable". The finding is that 9 did not coincide with a T0E hit
in this configuration. Whether it could, is not determined and this audit does not claim it cannot.

## 5. Intended Semantic Quantity

This is the part that is determinable, and it comes from the production source rather than from the
capture programme.

### 5.1 The quantity is the current epoch at the enqueue point

`src/audioengine/ISRRetireRouter.cpp:282-288` and `291-296`:

```cpp
bool ISRRetireRouter::retireRT(void* ptr, void (*deleter)(void*)) noexcept
{
    return provider_->enqueueRetire(ptr, deleter, provider_->currentEpoch());
}

void ISRRetireRouter::retire(void* ptr, void (*deleter)(void*)) noexcept
{
    const auto result = enqueueWithRetry(ptr, deleter, provider_->currentEpoch(), ...);
}
```

The epoch is not passed in by the caller. It is read from the provider at the moment of enqueue.

### 5.2 The production use of that quantity is relational, and the source says so

`src/audioengine/AudioEngine.h:5302-5303`, the project's own comment:

```text
publish 時に markRetireEpoch() で epoch は進行済みだが、enqueue 時点の epoch は「現在 epoch」のため、
もう一度 publishEpoch() で進めてから tryReclaim しないと isOlder(entry.epoch, minReaderEpoch) が偽になる。
```

```text
the epoch at the enqueue point IS "the current epoch"
and the correct use of it is isOlder(entry.epoch, minReaderEpoch)
and the whole point of the comment is that the comparison is RELATIVE: the enqueue-point epoch
must be compared against the reader minimum as of AFTER the epoch has advanced
```

`src/core/EpochDomain.h:243`, the reclaim decision itself:

```cpp
if (isOlder(epoch, minEpoch))
    minEpoch = epoch;
```

with `isOlder(a, b) = static_cast<int64_t>(a - b) < 0` at `EpochDomain.h:257-260`. A **difference**
comparison, not an equality against a literal.

`src/audioengine/AudioEngine.Retire.cpp:111-113`, the same pattern in the reclaim pre-check:

```cpp
const auto retireEpoch = m_retireRouter->currentEpoch();
const auto minReaderEpoch = m_retireRouter->minReaderEpoch();
if (retireEpoch < minReaderEpoch)
```

```text
PROVEN   the semantic role of entry.epoch is "the global epoch as of retirement", and the
         production code consumes it exclusively through RELATIVE comparisons against the
         current reader minimum or the current epoch.
         No production site compares entry.epoch against a literal.
```

## 6. r9 versus the Intended Quantity

```text
conjunct 2 asserts   enqueueEpoch == 9
the source defines   enqueueEpoch is the current global epoch at retirement
the source consumes  it as  isOlder(enqueueEpoch, minReaderEpoch)   i.e. relative

referent    CORRECT.  conjunct 2 is about the right quantity.  r9 is that quantity.
form        ABSOLUTE where the codebase uses RELATIVE, everywhere, without exception.
measured    the relative relation holds: delta in {0, +1}, 116 of 116
            the absolute relation does not hold at the observation points: 0 of 232
```

So conjunct 2 is a predicate about the correct quantity, expressed in a form the codebase does not
use for that quantity, with a literal whose only origin is an observation made under a different run
flag.

## 7. Conjunct 1 / 2 / 3 Semantic Roles

| conjunct | text | what it denotes | form | role |
|---|---|---|---|---|
| 1 | `.if (@$t0 == 0)` | `$t0`, a **script-owned** pseudo-register | state, internal | capture sequencer. `Step5BC` titles the table "Frozen script state contract" and the whole table is `$t0` transitions. Correctly written: a live condition compared to a state the script itself owns. |
| 2 | `.if (@r9 == 9)` | `entry.epoch`, the **current global epoch** at enqueue | absolute literal | **the subject of this audit** |
| 3 | `.if (dwo(@rcx+0x34000) == 4)` | `enqueuePos`, a **moving counter** | absolute literal | same provenance as conjunct 2, from the same observation table |

```text
conjunct 1 is a state condition and is sound.
conjunct 2 and conjunct 3 are value conditions against literals transcribed from one
observation, and neither literal has a design derivation.

they differ in consequence, which the measurements separate cleanly:
  conjunct 3  enqueuePos == 4   held at 4 of 116 and 4 of 116.  It recurs.
  conjunct 2  r9 == 9           held at 0 of 116 and 0 of 116.  It did not recur.
```

The separation of "literal snapshot" from "live relationship" is therefore complete: conjunct 1 is
live, conjuncts 2 and 3 are snapshots, and only one of the two snapshots still discriminates.

## 8. Step5BB Retry-1 Evidence

```text
r9 = 9, globalEpoch = 9, enqueuePos = 4, sequence[4] = 4, gate opened, run flag --fpm-m0
```

Positioned, without being treated as correctness:

```text
r9 and globalEpoch were 9 SIMULTANEOUSLY at that hit.  The literal therefore coincided with
the live global epoch, which is exactly the coincidence that makes "9" a snapshot of the live
value rather than an independent specification.

at that hit conjunct 3 also coincided with the live enqueuePos, 4.

so all three literals in the guard were simultaneously true of the live quantities at the one
reference observation.  That is consistent with the guard having been written from that
observation, and inconsistent with the literals having been derived from the design.
```

## 9. Step5DK / Step5DP Evidence

```text
Step5DK  Baseline-1, --measurement=normal
  conjunct 2  FALSE at 116 of 116
  conjunct 3  TRUE at 4 of 116

Step5DP  Baseline-2, --measurement=normal
  conjunct 2  r9 == 9 TRUE 0 of 116.  r9 took 13 distinct values.
  conjunct 3  TRUE at 4 of 116
  live global epoch identified, and enqueueEpoch trails it by 0 or 1 at every hit
```

The two runs agree, and the Baseline-2 run additionally shows **why** conjunct 2 could not hold: the
quantity it tests tracks the live global epoch, and the live global epoch was not 9.

## 10. Predicate Validity

Ruled in two parts, because the two questions have different answers.

```text
VALID          the referent.  conjunct 2 tests entry.epoch, which is the correct quantity.

INVALID        as a production-semantic assertion.
               The source defines the quantity and consumes it exclusively through relative
               comparison.  '@r9 == 9' is not that form, the literal has no design basis, and
               the relation the source actually relies on was measured to hold at 116 of 116
               while the literal held at 0 of 232.

INTENT-AMBIGUOUS  as to which role the author intended, and this is NOT resolved here.
               Two readings are coherent and no record states which was meant:

  (A) capture trigger, pinned to a reference observation.
      Supported by: Step5BC frames the guard as a script state contract; conjunct 1 is a
      script-owned register; every literal matches the Retry-1 observation table.
      Under this reading the predicate is well formed.  It is also non-discriminating in
      --measurement=normal, so the capture cannot be triggered.

  (B) assertion of a production semantic.
      Under this reading the predicate is invalid, because the production semantic is the
      relative isOlder test, not an equality against a literal.

The ambiguity does not need to be resolved in order to act, because under BOTH readings the
predicate is currently non-discriminating in the configuration being measured.
```

```text
NOT concluded  what the author intended.  That is the residual ambiguity above.
NOT concluded  that the correct predicate is a comparison against globalEpoch.  That is a
               candidate repair and belongs to the Repair Design Gate.
NOT concluded  that 9 is the wrong number.  It is the right number for one observation.
```

## 11. Ambiguities / Unresolved Items

```text
A1  the author's intended role for the T0 guard, trigger or assertion.  Not recoverable
    from any record.  Only the author can settle it.

A2  whether the reference observation, taken under --fpm-m0, was intended to define the
    trigger for a --measurement=normal baseline.  If so the guard was never expected to
    discriminate in the baseline configuration, which would make the entire Redesign
    programme a search for a conjunction that could not hold.  Not determined.

A3  whether conjunct 3's literal 4 was intended as "the reference position" or as a
    protocol position.  It recurs at 4 of 116, so it discriminates, and the audit does not
    judge it.

A4  the legacy offset algebra remains internally inconsistent by 0x10.  Step5DP resolved it
    empirically in favour of rcx-0x1430 for EpochDomain, but the incorrect anchor has not
    been corrected in the records that carry it.
```

A2 deserves emphasis. If the guard's literals were authored from an `--fpm-m0` observation and then
used against an `--measurement=normal` baseline, then conjunct 2 was unsatisfiable from the moment
the baseline was defined, and the Redesign-2 through Redesign-9 sequence was searching for a
conjunction that could not hold. That is not established, and this audit does not assert it. It is
the single most consequential open item, because it would reframe the whole programme.

## 12. Conclusion

```text
r9 is the epoch captured at the enqueue point, which the production source itself calls the
current epoch, and which the production code consumes only through relative comparison against
the reader minimum.

the literal 9 is an observation, not a specification.  Its only traceable origin is the
Retry-1 observation table, taken under a different run flag from the baseline.

under --measurement=normal the guard's conjunct 2 held 0 times in 232 hits, while the
relation the source actually relies on held 116 times in 116.

conjunct 2 is therefore INVALID as a production-semantic assertion, and INTENT-AMBIGUOUS as to
the author's intended role.  Both readings agree that it no longer discriminates.
```

## 13. Explicitly Forbidden Actions, all observed

```text
CDB re-run                       not performed
Retry                            not performed
Baseline-2 edited                not performed
T1 repair                        not performed
source / test / CMake / build    not performed
Harness edited                   not performed
predicate modified               not performed
changed toward a globalEpoch comparison   not performed
S7_READER / Case A-D advanced    not performed
repair proposal written          not performed
```

```text
read-only inputs this gate touched
  ConvoPeq.md            read, SHA unchanged
  src/                   read only
  doc/work113 records    read only
  CDB execution          0
  residue processes      0
```

## 14. Next Gate

```text
Intent / Semantic Audit, this document
      |
      v
predicate validity, ruled in section 10
      |
      v
Repair Design Gate
      |
      v
separate Implementation Authorization
```

The Repair Design Gate inherits one decision it cannot make for itself:

```text
it must be told which role the T0 guard is meant to have, trigger or assertion,
because the two readings imply different repairs and the records do not distinguish them.

section 11 item A2 is the question that most changes the framing, and it should be answered
before any repair is designed.
```

No repair is proposed here, and none should be inferred from this document.
