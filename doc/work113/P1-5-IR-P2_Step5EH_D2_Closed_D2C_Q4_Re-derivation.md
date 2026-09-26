# Step5EH — RDG-REOPEN-D2 CLOSED, and D2-C Q4 Re-Derivation. One further amendment is forced by the authority.

```text
gate                = D2 decisions recorded, then D2-C Q4 re-derivation
mode                = READ-ONLY.  No .cdb edit.  No runtime.  No build.  No source modification.
execution           = 0
authority           = ConvoPeq.md  5,535,334 B
                      SHA-256 E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
                      VERIFIED BY DIRECT READ this gate, body present, enqueue body located at
                      line 61993.  ConvoPeq(4).md does not exist and was not used.
NEW CONTRACT        not yet drafted.  this gate supplies its inputs.
```

## 1. D2 decisions, recorded

```text
D2-A  Q3 target  = T-1, slot-4 acquisition                    DETERMINED
D2-B  R3b        = B-2, full register dump, T0E shape          DETERMINED

capture identity, as the Owner stated it
    @rbx == 4   AND   dwo(@r11 + 0x30010) == 4

  this is AMENDED by section 2.  the Owner stated it before the sequence semantics had been
  re-read against the authority, and the amendment is forced by the source, not preferred.
```

## 2. FORCED AMENDMENT — the sequence conjunct must compare against the position, not against 4

From `ConvoPeq.md`, read directly, lines 61995 to 62002:

```cpp
  uint32_t pos = convo::consumeAtomic(enqueuePos, std::memory_order_acquire);
  while (true) {
      auto& seq_atom = sequences[pos & kMask];
      uint32_t seq = convo::consumeAtomic(seq_atom, std::memory_order_acquire);
      int32_t diff = static_cast<int32_t>(static_cast<uint32_t>(seq - pos));
      if (diff == 0) { ... }
```

```text
  the free-slot predicate is   sequences[pos & kMask] == pos
  it is NOT  sequences[slot] == 4.

      pos = 4      slot 4,  sequence 4      the literal  == 4  holds
      pos = 4100   slot 4,  sequence 4100   the literal  == 4  DOES NOT HOLD

  the same holds on the publish side, from the same authority:
      line 62015   publishAtomic(seq_atom, pos + 1)               enqueue side
      line 62078   publishAtomic(seq_atom, scanPos + kQueueSize)   dequeue side
      line 62127   publishAtomic(seq_atom, pos + kQueueSize)       dequeue side

  the sequence array therefore stores ABSOLUTE position values, cycling through the ring.
```

```text
CONSEQUENCE FOR T-1

  T-1 was chosen precisely because slot 4 RECURS every 4096 enqueues.  the literal
  dwo(@r11+0x30010) == 4 is correct only in the FIRST cycle.  it contradicts the semantics
  T-1 was selected to preserve.

  at T0Q, r10d == pos, per the Step5EE correction.  so @r10 IS the absolute position.

  the capture identity must therefore be

      @rbx == 4                          slot identity, producer-local
      AND dwo(@r11 + 0x30010) == @r10     free-slot state, owner-exclusive

  note the address is UNCHANGED.  with @rbx == 4 already a conjunct, (@rbx & 0xfff) * 4 == 0x10,
  so the address reduces to +0x30010 either way.  ONLY THE COMPARISON OPERAND CHANGES,
  from the literal 4 to @r10.

  and it degenerates correctly: when pos == 4, @r10 == 4 and the predicate reduces to the
  literal form.  one expression covers both the first cycle and every later cycle.

  the Owner's stated identity is therefore correct for the measured run and INCORRECT for the
  semantics T-1 selects.  the amendment is not a preference.
```

## 3. D2-C item 1 — Q4-1 re-derived for T-1

```text
  a slot-4 acquisition on instance i occurs at a position

      p = 4 + 4096k        k = 0, 1, 2, ...

  and requires, at that position,
      the free test to pass,   sequences[4] == p
      the CAS to succeed

  let E_i be the number of successful enqueues on instance i during the measurement window.
  the instance reaches position p only if enqueuePos advanced to p, which needs p successful
  enqueues on that queue.  the number of candidate positions reached is therefore

      K_i = max(0, floor((E_i - 4) / 4096) + 1)

  and

      captures_i  <=  K_i
      captures_i  =  K_i   iff the slot was free and uncontended at every occurrence

  Q4-1, RE-DERIVED
      expected captures per instance = K_i, an upper bound attained only if no occurrence was
      skipped.  it is NOT a constant and it is NOT 1.
```

## 4. D2-C item 2 — Q4-2 re-derived, aggregate

```text
  Q4-2, RE-DERIVED

      0  <=  captures  <=  SUM over instances i of  K_i
                              K_i = max(0, floor((E_i - 4) / 4096) + 1)

  the number of instances is at most 4, from the source-defined cardinality of Step5DW
  (3 declaration sites x ConvolverProcessor being instantiated twice inside AudioEngine).

  SPECIAL CASE, and this is where 0..4 comes from
      if every E_i <= 4100, that is no instance completes two full cycles, then K_i <= 1 for
      every i, and

          0  <=  captures  <=  (number of instances that reached slot 4)  <=  4

      the previously used 0..4 form is therefore VALID, but only as this special case.
      it is NOT the general specification, and Step5DV's presentation of it as the general bound
      was imprecise.
```

### 4.1 the bound is testable without new instrumentation

```text
  E_i is already observable.  T0E emits an UNCONDITIONAL register dump, which includes rcx, and
  T0E fires once per enqueue.  counting T0E hits per distinct rcx yields E_i directly.

  that was in fact the method that produced the 21 / 13 / 41 / 41 per-queue figures in Step5DP.

  is the resulting check self-validating?  NO, and the distinction matters:

      the prediction  captures <= SUM f(E_i)  uses a source-derived recurrence f, and E_i is an
      input measured independently of the capture.  the prediction can FAIL on both sides:
          captures < SUM f(E_i)   the predicate is missing acquisitions
          captures > SUM f(E_i)   the predicate is over-firing
      so it is falsifiable in both directions, which is the test the self-validating family
      cannot be.
```

## 5. D2-C item 3 — `sequences[4] == 4` at capture time, corrected

```text
  at T0Q the publish at 0x141f9c5d2 has NOT executed, so the slot's sequence still holds the
  pre-publication value, which is p, the position this call just claimed.

  for p = 4     that value is 4      and  dwo(@r11+0x30010) == 4   is right
  for p = 4100  that value is 4100   and  dwo(@r11+0x30010) == 4   is WRONG

  owner-exclusive: only the owner of slot 4 writes that word, and the owner is the thread that
  won the CAS, which is this thread.  a competing producer claiming slot 5 writes sequences[5].
  so the conjunct is contention-immune in both forms.

  the corrected form  dwo(@r11+0x30010) == @r10  is read at the same instant and carries the
  same owner-exclusivity.
```

## 6. D2-C item 4 — FN = 0, structural

```text
  T0Q is reachable ONLY through  je 0x141f9c586  on the cmpxchg success branch.  a CAS loser
  falls to 0x141f9c58b, jumps to 0x141f9c560 and re-enters the loop; it never reaches T0Q.

  therefore, at T0Q,
      @rbx == 4   entails   the CAS for the position whose slot index is 4 was WON

  and every CAS win falls through to T0Q, so a genuine slot-4 acquisition always reaches T0Q,
  and at that moment rbx == pos & 0xFFF == 4.

  FN = 0 therefore follows STRUCTURALLY, and not statistically.  no run is needed to establish
  it; a run can only add contrary evidence.

  the corollary is that a capture count BELOW K_i is not an FN but evidence that the
  free test or the CAS declined an occurrence, which the full T0Q dump can now distinguish.
```

## 7. D2-C item 5 — placement of the unconditional full T0Q dump

```text
  T0E's proven shape, which is the model:
      .echo <HIT> ; <diag if/else> ; <diag if/else> ; r ; .echo <EP> ; dq <window> ;
      .echo <WIN_END> ; <guard>

  T0Q becomes:
      .echo C3_DG_T0Q_HIT ; r ; .echo C3_DG_T0Q_EP0 ; <operand probes> ;
      .echo C3_DG_T0Q_WIN_END ; <guard>

  the existing  r  inside the acceptance branch must be REMOVED, or the hit is dumped twice.

  what the unconditional dump yields, from the same capture point
      r10   the absolute position, diagnostic only
      rbx   the slot index, a predicate conjunct
      r11   the queue base, the address operand
      rdx   entry.ptr          r8   entry.deleter
      rsp   the incoming argument area, so [rsp+0x28] is entry.type, which Step5DT proved
            readable at 0x141f9c59a, the T0Q instruction itself

  volume   the  r  command is one token in the script and emits about 7 lines per hit.
           at 65 T0Q hits that is roughly 455 added lines on a 2,603-line log, of the order of
           23 percent.  accepted by the Owner as diagnostic cost.
```

## 8. D2-C item 6 — ladder impact

```text
  site   guard .if before   after   ladder levels before   after   note
  T0Q    3                  2       3                    2       the r10 conjunct is REMOVED,
                                                                 not replaced.  @r10 was
                                                                 withdrawn as a position conjunct
                                                                 at RDG-REOPEN-D1.

  and the corrected sequence conjunct changes the remaining T0Q guard to
      .if (@rbx == 4) { .if (dwo(@r11+0x30010) == @r10) { .echo C3_T0_CAS_POST ; <payload> ; gc }
        .else { gc } } .else { gc }

  the  r  in the payload must go, per section 7.

  FLAGGED, and NOT decided here

      T0P's conjuncts are  @r10 == 4 , dwo(@r11+0x34000) == 4 , dwo(@r11+0x30010) == 4
      under T-1 semantics the last of these is subject to the identical defect as section 2:
      it is correct only in the first cycle.  C3_T0_CAS_PRE would therefore become a
      first-cycle-only diagnostic.

      T0P is NOT the capture marker, so this is a DIAGNOSTIC CONSISTENCY issue, not a
      correctness one.  generalising it is a SEPARATE sub-decision and is not taken here.
      r11 is already assigned by 0x1f9c55c, which precedes T0P at 0x1f9c57d, so the
      generalisation is expressible at T0P.
```

## 9. D2-C item 7 — T0S single-shot, unchanged

```text
  T0S is NOT edited.  its conjuncts, its  r $t0 = 1  write, and its $t1..$t19 initialisation
  stay byte-identical.

  consequence, stated because it is a real interaction and not a defect:
      T0S is single-shot at $t0, so it publishes at most once per run.
      under T-1, captures scale with K_i while publications are capped at 1.
      this is the Step5EA section 4 situation and it is already accepted: per the BC restatement,
      captures 2..N are complete captures that carry no handoff, and the handoff is single-shot
      by design.

      T0S's own  dwo(@r11+0x34000) == 5  conjunct is the Step5EB section 4 residual and remains
      out of scope.  it is live shared state in a NON-capture marker, so RDG-3-2's principle does
      not bind it.
```

## 10. D2-C closure checklist

```text
  Q3 target                DETERMINED      T-1, slot-4 acquisition
  R3b payload              DETERMINED      B-2, full unconditional register dump at T0Q
  Q4-1 re-derived          DETERMINED      K_i = max(0, floor((E_i-4)/4096)+1) per instance
  Q4-2 re-derived          DETERMINED      0 <= captures <= SUM K_i, and <= 4 in the
                                            no-second-cycle regime
  capture predicate        DETERMINED      @rbx == 4  AND  dwo(@r11+0x30010) == @r10
  FN criterion             DETERMINED      structural, 0, from the je on the cmpxchg success
  runtime observability    DETERMINED      B-2 supplies register evidence on failure
  scope                    DETERMINED      T0E / T0P / T0Q / T0S only
  T1 / S7                  DETERMINED      unchanged

  OPEN, carried
      T0P sequence conjunct generalisation.  a diagnostic-consistency sub-decision, see 8.
      T0S's live enqueuePos conjunct.  the accepted Step5EB section 4 residual.
      the two malformed T1 sites.  out of scope.
      the 65 versus 116 hit non-comparability.
```

## 11. State

```text
RDG-REOPEN-D2   CLOSED.  both decisions determined.
D2-C            CLOSED.  all seven items fixed, one forced amendment to the Owner's stated
                predicate, and one sub-decision carried open.
NEW CONTRACT    NOT YET DRAFTED.  its inputs are now complete.
cdb edit        NONE
runtime         NONE
build           NOT INVOKED
source          unchanged
Baseline-3      unchanged, FAILED DESIGN, retained as evidence
baselines 7/7 + Baseline-3 = 8/8 intact    .cdb 16    git delta 12, all pre-existing
residue 0
```

```text
next, in order
    New Capture Predicate Contract, drafted from section 10, with the section 2 amendment
    Contract Review, read-only audit, against this same ConvoPeq.md
    Implementation Authorization
    ... and so on per the Owner's fixed order.

no edit, no run, no build until the Implementation Authorization is issued.
```
