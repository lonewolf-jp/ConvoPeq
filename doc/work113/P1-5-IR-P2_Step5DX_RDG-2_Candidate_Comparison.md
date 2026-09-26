# Step5DX — RDG-2 Candidate Comparison

```text
gate                = Repair Design Gate, stage 2 of 4.  CANDIDATE COMPARISON.
mode                = READ-ONLY.  No CDB run, no retry, no .cdb edit, no source / test / CMake /
                      build / harness change.  No repair proposed.
execution this gate = 0
authority           = ConvoPeq.md  SHA-256 E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
inputs              = Q3 = a, Q4 CLOSED, the frozen Baseline-2 guards read verbatim,
                      src/DeferredDeletionQueue.h enqueue, the C1-C7 criteria from RDG-1
Selected candidate  = NONE.  See section 8.
```

## 1. Operand semantics, established from source. This is the C5 foundation.

`src/DeferredDeletionQueue.h` enqueue, the authoritative text:

```text
L78   uint32_t pos = consumeAtomic(enqueuePos, acquire);
L80   auto& seq_atom = sequences[pos & kMask];
L81   uint32_t seq   = consumeAtomic(seq_atom, acquire);
L83   int32_t  diff  = (int32_t)(uint32_t)(seq - pos);
L85   if (diff == 0) {
L86       if (compareExchangeAtomic(enqueuePos, pos, pos + 1, acq_rel, acquire)) {
L91           auto& entry = ringBuffer[pos & kMask];
L92-97        entry.{ptr,deleter,epoch,type,publicationSequenceId,generation} = ...
L98           publishAtomic(seq_atom, pos + 1, release);
L99           return true;
```

Four consequences, all exact:

```text
slot index        = pos & kMask                    kMask = 4095, so pos 4 -> index 4
free-slot test    = diff == 0, that is seq == pos  so sequences[4] == 4 IS the free predicate
CAS               enqueuePos : pos -> pos + 1     so 4 -> 5
publication       sequences[index] <- pos + 1     so sequences[4] == 5 is the PUBLISHED value
entry write       L92-97 occurs BETWEEN the CAS and the publish, so there is a real window in
                  which enqueuePos == 5 while sequences[4] is still 4
```

Mapping each frozen conjunct onto that text:

| site | conjunct | source meaning |
|---|---|---|
| T0E | `dwo(@rcx+0x34000) == 4` | L78 `pos == 4`. the acquisition target |
| T0P | `dwo(@r11+0x34000) == 4`, `@r10 == 4`, `dwo(@r11+0x30010) == 4` | L83-85 `diff == 0`, slot free, CAS at L86 pending |
| T0Q | `dwo(@r11+0x34000) == 5`, `@r10 == 5`, `dwo(@r11+0x30010) == 4` | L86 CAS done, L92-97 entry written, **L98 not yet executed** |
| T0S | `dwo(@r11+0x34000) == 5`, `@r10 == 5`, `@rbx == 4`, `dwo(@r11+0x30010) == 5` | L98 published `pos+1 = 5`; `rbx` is the slot index `pos & kMask` |

`rbx` as the slot index is independently corroborated by T0S's own payload, which computes
`dd @$t2+0x30000+((@$t10&0xfff)*4) L1` with `r $t10=@rbx` — that is `sequences[rbx & 0xFFF]`.

```text
the bracket is not "positions 4 and 5".  it is a four-state machine of ONE acquisition:

  T0E   target position loaded
  T0P   slot verified FREE, CAS pending
  T0Q   CAS won, entry written, NOT YET PUBLISHED   <-- the only state in which the producer
                                                        owns the slot while a consumer still
                                                        computes the slot as empty
  T0S   publication complete

T0Q is the semantically load-bearing state, and it is the one the Producer/Reader temporal
correlation workstream exists to examine.  it has value beyond position counting.
```

## 2. Two corrections to my own Step5DT extraction

```text
CORRECTION 1   T0P and T0Q carry NO @$t0 guard.  Only T0E and T0S read @$t0.

              Step5DT already recorded script-state as absent for T0P and T0Q, so this is a
              confirmation rather than a change, but it becomes decisive in section 4 and was
              not previously followed through.

CORRECTION 2   T0P / T0Q / T0S use `dd @r11+0x34000 L1` — `dd` as a DISPLAY command, which is
              valid.  This is NOT the `dd(`-as-expression-function parse failure of Step5DF.
              That defect lives at 0x1f9ce04 and 0x1f9cfb0, not at the T0 sites.  The T0 phase
              is free of the defect that silenced Redesign 2 through 9.
```

## 3. A limitation I cannot close read-only

```text
@r10 is 4 at T0P and 5 at T0Q and T0S.

in the source, `pos` REMAINS 4 along the entire success path: it is used at L91
(ringBuffer[pos & kMask]) and at L98 (pos + 1).  the local is never reassigned on success.

therefore r10 is NOT the source local `pos`.  it must be a re-materialised read of enqueuePos
after the CAS wrote it, or some other derived value.  I cannot confirm which from the source
alone, and the disassembly between 0x1f9c555 and 0x1f9c59a is not in evidence.

CONSEQUENCE FOR THE COMPARISON
  every conjunct that matters semantically is MEMORY based and unambiguous:
      dwo(@r11+0x34000)   enqueuePos
      dwo(@r11+0x30010)   sequences[4]
      @rbx                the slot index
  @r10 is redundant corroboration.  no candidate in this document depends on @r10 alone, and
  any candidate that drops @r10 loses no semantic content.
```

## 4. C5, the centre of this gate. The `$t0` asymmetry makes the four markers differ in cardinality.

```text
FACT, from the verbatim guards

  T0E   reads  @$t0 == 0      writes $t0 : none
  T0P   reads  no $t0         writes $t0 : none
  T0Q   reads  no $t0         writes $t0 : none
  T0S   reads  @$t0 == 0      writes r $t0 = 1      <-- the only latch

CONSEQUENCE, if the epoch literal is removed so the phase can actually open

  let N = the number of queue instances that reach position 4.  Q4-2 bounds N to 0..4.

  marker                       expected count under a positional-only guard
  --------------------------   ----------------------------------------------
  C3_T0_ENTRY                  N        no $t0 write, fires once per instance
  C3_T0_CAS_PRE                N        no $t0 guard at all
  C3_T0_CAS_POST               N        no $t0 guard at all
  C3_T0_SEQUENCE_PUBLISHED     min(N,1)  the latch fires on the first bracket and disables
                                         T0E and T0S for every later instance
```

```text
THIS IS A SPECIFICATION GAP AND IT IS THE MAIN FINDING OF RDG-2

Q4-1, Q4-2 and Q4-4 were derived in terms of "the capture count", but no marker was ever
specified as the definition of a capture.  Under a positional-only phase the four markers do
NOT share one cardinality: three of them scale with N and one is capped at 1.

an acceptance test that counts "captures" without naming the marker is therefore ambiguous, and
would pass or fail depending on an unstated choice.  this is recorded for RDG-3, not resolved
here.

concretely, for N = 4 the derivable prediction is

  ENTRY 4   CAS_PRE 4   CAS_POST 4   SEQUENCE_PUBLISHED 1

  and for N = 1,  ENTRY 1  CAS_PRE 1  CAS_POST 1  SEQUENCE_PUBLISHED 1.
  these follow from the source and from Q4-2 alone.  no log was consulted.
```

## 5. The four candidates

```text
CANDIDATE A   positional predicate at T0E only
             gate:  .if (@$t0 == 0) { .if (dwo(@rcx+0x34000) == 4) { C3_T0_ENTRY } }
             the epoch literal is removed at T0E.  T0P / T0Q / T0S are left untouched and
             therefore remain blocked behind the same literal, so no bracket is produced.

CANDIDATE B   full positional bracket
             the epoch literal is removed at all four sites, leaving each site's positional
             conjuncts exactly as frozen:
               T0E  @$t0==0 ; dwo(@rcx+0x34000) == 4
               T0P           @r10==4 ; dwo(@r11+0x34000) == 4 ; dwo(@r11+0x30010) == 4
               T0Q           @r10==5 ; dwo(@r11+0x34000) == 5 ; dwo(@r11+0x30010) == 4
               T0S  @$t0==0 ; @r10==5 ; @rbx==4 ; dwo(@r11+0x34000) == 5 ;
                    dwo(@r11+0x30010) == 5 ; @r11 != 0 ; r $t0 = 1
             no operand is added, removed or changed.  only the literal leaves.

CANDIDATE C   positional bracket with the slot index generalised from a literal to the
             register-derived index
               dwo(@r11+0x30010)  ->  dwo(@r11+0x30000 + ((@rbx & 0xfff) * 4))
             the guard then states "this slot's own sequence" rather than "the sequence at a
             hard-coded address happens to be 4".  no new address is required: the sequences
             base +0x30000 is already proven and rbx is already read.  this is the form T0S's
             own payload already computes for display.

CANDIDATE D   retain the epoch literal
             any design that keeps `@r9 == 9` as a conjunct of the target.  evaluated here as
             instructed, and NOT dismissed as obvious.
```

## 6. C1 to C7

| | C1 identify target | C2 reproducible | C3 mode-independent | C4 operand inventory | C5 sequencer | C6 T1-independent | C7 FP/FN definable |
|---|---|---|---|---|---|---|---|
| **A** | **PASS** — `enqueuePos == 4` is the target's entry condition, 4/116 in two runs | **PASS** — 4/116 twice | **PASS** — a ring-position fact, not a value read from a run | **PASS** — `dwo(@rcx+0x34000)` is already in the frozen guard, no new address | **PASS with a cost** — nothing writes `$t0`, so ENTRY fires N times, which matches Q4-2 exactly. But T0P/Q/S keep `@r9 == 9` and never fire, so **no bracket and no latch** | **PASS** | **PARTIAL** — FN = 0, but a real FP class remains: see 7.A |
| **B** | **PASS** — same entry condition, and the downstream sites now become reachable and are *stronger* than the entry condition | **PARTIAL** — the T0E conjunct is 4/116 twice; the T0P/Q/S conjuncts are **UNMEASURED** (OPEN-3), so only the entry has reproducibility evidence | **PASS** — every conjunct is a ring-position or sequence fact | **PASS** — every operand is already in the frozen guards; deleting the literal needs no new operand | **PASS but CHANGES CARDINALITY** — see section 4. ENTRY/CAS_PRE/CAS_POST = N, SEQUENCE_PUBLISHED = min(N,1) | **PASS** | **PASS** — see 7.B |
| **C** | **PASS**, equal to B on the entry condition | **PARTIAL or WEAKER** — as B, and the generalised index is additionally unmeasured | **PASS** | **PASS** — `+0x30000` proven, `rbx` already read; no new address | **PASS**, same as B | **PASS** | **PASS but NOT BETTER THAN B** — see 7.C |
| **D** | **FAIL** — the conjunction held 0 of 232 across two runs, so the target is never captured | **FAIL** — 0/232 is a measured total, not a rate with variance | **FAIL** — the literal's only provenance is the Retry-1 `--fpm-m0` observation, and Q2 already ruled that it carries no authority | **PASS** — `@r9` is already available | **N/A** — the phase can never open, so no sequencer behaviour is exercised | **PASS** | **TOTAL FALSE NEGATIVE** — every real acquisition is missed |

## 7. C7, per candidate

```text
TARGET, fixed by Q3 = a
  a slot-4 acquisition of DeferredDeletionQueue, by any instance, from the free-slot test
  through the CAS to the publication

7.A  CANDIDATE A
  True Positive   a T0E hit with enqueuePos == 4
  False Negative  none.  every instance that reaches position 4 produces a T0E hit.
  False Positive  a call that OBSERVES position 4 but LOSES the CAS at L86.  the code then
                  falls to L104, reloads pos, and the acquisition is performed by the winner,
                  not by this call.  A counts it anyway.

  so A has a genuine FP class: observers are counted as acquirers.  the class is empty only if
  no two producers ever contend for position 4, which is a runtime property and is NOT
  established.  whether it is empty is UNMEASURED.

7.B  CANDIDATE B
  True Positive   a completed bracket: T0P shows diff == 0 with the CAS pending, T0Q shows the
                  CAS won with the entry written and unpublished, T0S shows the publication.
  False Negative  an acquisition whose T0S conjuncts do not all hold.  T0S's conjuncts ARE the
                  success conditions, so this requires a foreign interference between L98 and
                  the breakpoint.  no evidence of it, and not measurable without a run.
  False Positive  none identified.  T0Q and T0S can only be reached on the success path,
                  because the retry path at L104 returns to the top of the loop and re-reads
                  pos, so it never carries pos 4 into T0Q.

  B therefore ELIMINATES A's FP class, and does so structurally rather than statistically.

7.C  CANDIDATE C
  identical TP / FN / FP to B, because with rbx == 4 the generalised index reduces to the
  literal one:  (4 & 0xfff) * 4 + 0x30000 == 0x30010.

  so C buys NO additional precision over B, while adding an arithmetic expression to the guard
  and an unmeasured conjunct.  on C7 it is DOMINATED by B, not better.
```

## 8. Selected candidate

```text
Selected candidate = NONE

RDG-2 is a comparison gate.  It records the ordering that the evidence supports and it does
not adopt a design.  On C1 to C7 the ordering is

    B  >  A  >  C  >  D

  B is the only candidate that eliminates A's false-positive class structurally, and it does so
  without adding a single operand.  C is dominated by B.  D fails C1, C2 and C3 on measurement
  and grounds already adjudicated in Q2.

This ordering is NOT a selection.  Two things block selection here, and both are acceptance
questions rather than candidate questions:

  BLOCK-1  the marker that defines "a capture" is unspecified.  section 4 shows the four markers
          do not share a cardinality.  Q4-2's 0..4 is correct for ENTRY, CAS_PRE and CAS_POST
          and wrong for SEQUENCE_PUBLISHED, which the latch caps at 1.  until the Owner names
          the marker, "capture count" is ambiguous and no candidate can be accepted.

  BLOCK-2  B's removal of the literal at all four sites is only reachable if the T0P / T0Q /
          T0S conjuncts actually hold, and they have never been evaluated (OPEN-3).  choosing B
          means committing to a conjunction whose joint hit rate is UNMEASURED.  RDG-2 was
          instructed not to start runtime measurement, so this cannot be closed here.
```

## 9. What RDG-2 hands forward

```text
TO RDG-3, acceptance criteria must fix
  1  which marker defines a capture.  the four have different cardinalities (section 4).
  2  the FP criterion.  B's structural elimination is the only one established; A's is not.
  3  N explicitly, as a symbol, not a constant.  Q4-2 bounds it 0..4 and does not fix it.
  4  what "bracket complete" means if a bracket is interrupted, since the latch is one-shot.

TO RDG-4, a selected design must state
  1  whether the phase is single-shot (latch retained, so exactly one SEQUENCE_PUBLISHED) or
     multi-shot (latch removed, so N brackets).  this is a design choice, not a measurement.
  2  whether C's generalised index is adopted for clarity or left as the literal for
     traceability.  C adds no precision, so this is a readability judgement.

CARRIED, unchanged
  OPEN-2   Baseline-1 has no register dump.  not a blocker; a C2 limitation.  C2 for B rests on
           the T0E conjunct alone, which does have two-run evidence.
  OPEN-3   T0P / T0Q / T0S conjuncts UNMEASURED.  this is BLOCK-2 above and is now the binding
           limitation on candidate B.
  OPEN-1b  address-span / identity mapping.  does not return as a blocker; the comparison uses
           no address from the log.
  r10      re-derivation not established (section 3).  no candidate depends on it.
```

## 10. State

```text
Q1 CONFIRMED   Q2 CONFIRMED   Q3 CONFIRMED (a)   Q4 CLOSED   Q4-5 CONFIRMED

RDG-2  CLOSED as a comparison.  Selected candidate = NONE.
       ordering recorded:  B > A > C > D
       BLOCK-1 marker-vs-cardinality ambiguity, referred to RDG-3
       BLOCK-2 T0P/T0Q/T0S conjuncts unmeasured, referred as the binding limitation

RDG-3  NOT STARTED
RDG-4  NOT STARTED

IMPLEMENTATION FORBIDDEN      RUNTIME NOT AUTHORIZED      BUILD NOT AUTHORIZED
CDB execution 0    .cdb 15 unchanged    baselines 7/7 intact
ConvoPeq.md / harness / cdb.exe unchanged    git delta 12, all pre-existing
no predicate adopted    no candidate implemented    no repair proposed
```

The selected candidate is NOT to be treated as a production implementation before Owner approval.
