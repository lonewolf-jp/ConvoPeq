# Step5EG — RDG-REOPEN-D2. Q3 Target Granularity and T0Q Diagnostic Payload. Decision gate.

```text
gate                = RDG-REOPEN-D2, read-only Owner Decision Gate
mode                = READ-ONLY.  No .cdb edit.  No Baseline-3 modification.  No runtime.
                      No build.  No source modification.  No T1 or S7 repair.
                      No predicate implemented.  Candidate F NOT advanced.
execution           = 0
inputs              = Step5EF, Step5ED runtime log, src/DeferredDeletionQueue.h,
                      the disassembly of AudioEngineHarness.exe
NEW CONTRACT        NOT AUTHORIZED.  STOP.  see section 5.
```

## 0. A claim of mine is WITHDRAWN before any decision is asked

Step5EF section 4 stated that under the slot reading the capture rate at this run length would
be near zero. **That was wrong, and the log refutes it.**

```text
  the reasoning error    I generalised a production property and applied it to a measurement
                         window without checking.  a queue starts at enqueuePos = 0 and
                         increments by 1, so it passes through pos = 4 once during its first
                         4096 enqueues, and at that moment pos & 0xFFF == 4 as well.
                         for pos in 0..4095 the two selectors are IDENTICAL.

  the measured evidence, C3_T0_CAS_PRE accepted attempts, verbatim from the log

      rax=0000000000000004 rbx=0000000000000004 rcx=0000000000000005
      ...
      r8=00007ff77f259f10  r9=0000000000000001 r10=0000000000000004
      r11=0000015161ced580 ...

      rax=0000000000000004 rbx=0000000000000004 rcx=0000000000000005
      ...
      r8=00007ff77f259f10  r9=0000000000000001 r10=0000000000000004
      r11=0000015168824580 ...

      attempt 1   r10 = 4   rbx = 4   r11 = 0x15161ced580
      attempt 2   r10 = 4   rbx = 4   r11 = 0x15168824580

      2 attempts, 2 DISTINCT queue instances, rbx == 4 on both, r10 == 4 on both.

  C3_T0_SEQUENCE_PUBLISHED fired once, r10 = 5, rbx = 4.  that is post-publish proof that at
  least one slot-4 acquisition COMPLETED.

CONSEQUENCE

  Candidate F,  @rbx == 4 AND dwo(@r11+0x30010) == 4,  WOULD HAVE FIRED on the winning
  attempt.  the expected capture count in this run is 1 or 2, NOT zero.

  WITHDRAWN   Step5EF section 4, "capture is essentially unobservable at this run length"
  and with it the framing of T-1 as "a choice that accepts near-zero captures".

  T-1 and T-2 are NOT distinguished by measurement feasibility.  they are distinguished only
  by PRODUCTION semantics.
```

## 1. Owner decision A — Q3 target granularity. NOT DECIDED HERE.

```text
  what is now established, and common to all three options

    @rbx == 4 is the slot-identity conjunct.  DETERMINED at RDG-REOPEN-D1.
    T0Q is reachable only through the je on the cmpxchg success branch, so @rbx == 4 at T0Q
    entails "the CAS for slot index 4 was won".  FN = 0 follows structurally.
    2 queue instances reached slot 4 in this run, so N = 2 here, and Q4-2's 0..4 upper bound
    is consistent with it and with the Step5DW source cardinality.
```

### T-1 — keep Q3 = a, slot-4 acquisition, slot-index selector

```text
  capture identity would be   @rbx == 4  AND  dwo(@r11 + 0x30010) == 4
  production behaviour       slot 4 recurs every 4096 enqueues.  a REUSABLE selector.
  observation in this run    1 or 2 captures, per section 0.  directly observable.
  Q4-1 re-derivation        one acquisition per instance per 4096-enqueue cycle that reaches
                            slot 4.  in a run shorter than 4096 enqueues per instance, that
                            is at most 1 per contributing instance.
  Q4-2 re-derivation        0 <= captures <= 4 still holds, the bound coming from the
                            source-defined instance cardinality of Step5DW, not from any
                            observation.
  commits to                 slot semantics as the target.  a change of Q3 wording from
                            "position 4" to "slot 4" in the Q4 derivation, not a change of Q3.
```

### T-2 — re-open Q3, absolute position 4

```text
  capture identity would be   @r10 == 4  AND  dwo(@r11 + 0x30010) == 4
  production behaviour       position 4 occurs ONCE per 2^32 enqueues, that is once per queue
                             per process lifetime.  a ONE-SHOT, not a selector.
  observation in this run    identical to T-1.  indistinguishable, see section 0.
  Q4-1 re-derivation        one acquisition per instance per PROCESS LIFETIME.
  Q4-2 re-derivation        0 <= captures <= 4 still holds.
  commits to                 a target that cannot recur.  the selector would fire at most once
                             per queue in production, which changes what the capture is FOR.
  note                       this is a production-semantics decision, not a
                             measurement-feasibility one.  Step5EF framed it wrongly.
```

### T-3 — change the granularity to a sampling trigger

```text
  capture identity would be   the position-independent free-slot condition at T0Q, namely
                             dwo(@r11 + 0x30000 + ((@rbx & 0xfff) * 4)) == @r10
                             with @r10 == 0 handled as a non-selection at pos 0
  production behaviour       fires on EVERY successful acquisition.  not a selector at all.
  observation in this run    on the order of the T0Q hit count, 65, not 1 or 2.
  Q4-1 re-derivation        total successful acquisitions per run.  no longer bounded by
                             instance count.
  Q4-2 re-derivation        the 0..4 bound becomes INVALID and must be re-derived against the
                             run's enqueue volume.  this is the only option that voids Q4-2.
  commits to                 moving the discrimination problem downstream into the T1 and S7
                             analysis, which are out of scope and whose gate protocol is
                             single-shot at T0S.
  risk                      with T0S single-shot at $t0, a multi-capture phase means later
                             captures have no publication, per the Step5EA section 4 BC
                             restatement.  that interaction is unexamined under T-3.
```

## 2. Owner decision B — R3b T0Q diagnostic payload. NOT DECIDED HERE.

The requirement itself, R3a, is DETERMINED: a T0Q capture that fails must leave evidence.
The content is open.

```text
  B-1  MINIMAL
         r10, rbx, live enqueuePos, sequences[4], unconditionally at T0Q

       falsifies         the predicate is fully checkable, since all three conjuncts'
                          operands plus the contention observable are present
       volume            small.  roughly 4 values per hit.
       does not yield    rdx, r8, rsp, and therefore not entry.ptr, entry.deleter, or the
                          argument area.  entry.type at [rsp+0x28] stays unobserved, although
                          Step5DT proved it is readable there.

  B-2  FULL
         the T0E shape, a complete register dump moved outside the T0Q guard

       falsifies         everything B-1 does, and additionally leaves the entry's ptr, deleter
                          and incoming argument area on record
       volume            about 7 lines per hit.  at 65 T0Q hits that is roughly 460 added
                          lines on a 2,603-line log, so of the order of 23 percent growth.
       also yields       entry.type at [rsp+0x28], which Q3=c had wanted and which no current
                          gate uses.
```

## 3. Consequence for Q4, which the Owner's instruction already anticipated

```text
  Q4-1 and Q4-2 as previously derived are NOT reusable.  they were derived under the
  absolute-position reading with the 4/116 observation in view.  both readings must be
  re-derived from structure after decision A lands.

  what survives unchanged
      the 0..4 upper bound's ORIGIN, which is the source-defined instance cardinality of
      Step5DW, 3 declaration sites x instantiation multiplicity = 4, with 0 at the
      T0E/T0P/T0Q/T0S fault sites all carrying 0 so far.  the bound is independent of the
      target only for T-1 and T-2.  T-3 voids it.
```

## 4. D2 closure checklist, current state

```text
  Q3 target                UNDECIDED      T-1 / T-2 / T-3
  R3b payload              UNDECIDED      B-1 minimal / B-2 full
  Q4-1 re-derived          BLOCKED        on Q3 target
  Q4-2 re-derived          BLOCKED        on Q3 target
  capture predicate        BLOCKED        on Q3 target and R3b
  FN criterion             PARTIAL        @rbx == 4 gives FN = 0 structurally at T0Q, but it
                                         must be restated against the chosen target
  runtime observability    PARTIAL        R3a determined, content open
  scope                    DETERMINED     T0E / T0P / T0Q / T0S only
  T1 / S7                  DETERMINED     unchanged

  UNDECIDED OR BLOCKED ITEMS REMAIN.  therefore:

  NEW CONTRACT = NOT AUTHORIZED
  STOP
```

## 5. What is fixed, and what is not

```text
  FIXED, carried forward
      @rbx == 4 is the slot-identity conjunct, success-conditional, FN = 0 structural
      dwo(@r11+0x30010) == 4 is owner-exclusive and contention-immune
      @r10 is a DIAGNOSTIC observable, never a position conjunct
      live enqueuePos is a diagnostic observable, never a capture conjunct
      the unconditional T0Q payload is REQUIRED
      scope is the four T0 sites only; T1 and S7 unchanged
      2 queue instances reached slot 4 in this run, N = 2, consistent with the 0..4 bound

  NOT FIXED
      the target granularity
      the payload content
      Q4-1, Q4-2, the predicate, the FN statement, all pending on the two decisions

  INVALID / SUPERSEDED, unchanged
      Candidate E   @r10 == 5   unsatisfiable at T0Q
      Candidate E'  @r10 == 4   window-dependent
      Step5EB contract, superseded
      Step5EC Baseline-3, FAILED DESIGN, retained unchanged as evidence
      Step5EF section 4, withdrawn in section 0 above
```

## 6. After D2 closes, the order is fixed

```text
  RDG-REOPEN-D2
      ->  New Capture Predicate Contract
      ->  Contract Review, read-only audit, against the CURRENT ConvoPeq.md
      ->  Implementation Authorization
      ->  Implementation
      ->  Static and structural gates, I-criteria and diff audit
      ->  Runtime Authorization
      ->  ONE runtime
      ->  Runtime Gate
```

The Owner's section 3 requires that the contract stage re-verify items A to H against the
authoritative `ConvoPeq.md`, SHA-256
`E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609`, and that `ConvoPeq(4).md`
is not used as a substitute. That verification belongs to the contract stage, not to this gate,
and nothing in this document substitutes for it.

## 7. State

```text
RDG-REOPEN-D2   CLOSED as a gate, with BOTH decisions OPEN
NEW CONTRACT    NOT AUTHORIZED
cdb edit        NONE
runtime         NONE
build           NOT INVOKED
source          unchanged
Baseline-3      unchanged, FAILED DESIGN, retained as evidence
baselines 7/7 + Baseline-3 = 8/8 intact    .cdb 16    git delta 12, all pre-existing
residue 0
```
