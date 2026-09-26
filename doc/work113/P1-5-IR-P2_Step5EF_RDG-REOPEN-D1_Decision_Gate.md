# Step5EF — RDG-REOPEN-D1. Decision gate. The redundancy premise is FALSE, and a granularity problem surfaces.

```text
gate                = RDG-REOPEN-D1, read-only design decision
mode                = READ-ONLY.  No .cdb edit.  No Baseline-3 modification.  No runtime re-run.
                      No build.  No source modification.  No T1 or S7 repair.
execution           = 0
inputs              = Step5EE, and src/DeferredDeletionQueue.h read directly
NEW CONTRACT        NOT AUTHORIZED.  STOP.  see section 5.
```

## 1. R1 — `@r10 == 4` as the T0Q position conjunct. NOT RULED. The premise fails.

Step5EE and the Owner's §2 both assume `pos ∈ 0..4095`. **That is false**, and the source says so
directly.

```text
src/DeferredDeletionQueue.h:267
    std::atomic<uint32_t> enqueuePos{0};

  a uint32 monotonic counter, incremented by 1 per successful CAS at 0x141f9c57e, wrapping only
  at 2^32.  kQueueSize is the ARRAY size, not a bound on the counter.

src/DeferredDeletionQueue.h:210
    convo::publishAtomic(seq_atom, pos + kQueueSize, std::memory_order_release);

  the dequeue side publishes a sequence value 4096 AHEAD of the scan position.  that construction
  is only coherent when pos routinely exceeds kQueueSize.  this is direct source proof that
  pos > 4096 occurs in normal operation, not a hypothetical.
```

```text
CONSEQUENCE FOR R1

  @r10 == 4   means  pos == 4            the ABSOLUTE position
              occurs ONCE per 2^32 enqueues on a given queue
              correct ONLY while that queue's enqueuePos is below 4096

  it is therefore WINDOW-DEPENDENT.  it is a property of a fresh process with a short run, not a
  specification.  ruling it as the position conjunct would repeat the defect family this
  workstream has recorded repeatedly, namely an observation mistaken for a specification.

  R1  NOT RULED.  @r10 == 4 is WITHDRAWN as a candidate for the position conjunct.
```

## 2. R2 — `@rbx == 4`. The redundancy argument is withdrawn. It is the correct conjunct.

```text
  @rbx == 4   means  pos & 0xFFF == 4    the SLOT INDEX
              derived at 0x141f9c560  movl %r10d, %ebx
                          0x141f9c563  andl $0xfff, %ebx
              occurs every 4096 enqueues
              correct at any point in the ring's life

  the two are NOT equivalent.  they coincide only while enqueuePos < 4096.
  Step5EE section 5.B recorded the redundancy, and that recording is WITHDRAWN.
  the Owner also asserted the equivalence in section 2 of the re-open instruction.  both were
  wrong, for the same reason.
```

```text
  and @rbx == 4 carries a property the position conjunct does not:

      T0Q is reachable ONLY through  je 0x141f9c586  on the cmpxchg success branch.
      a CAS loser falls to 0x141f9c58b -> jmp 0x141f9c560 and never reaches T0Q.

  therefore  @rbx == 4 at T0Q  entails  "the CAS for the slot whose index is 4 was won".
  it is not merely a position or slot test, it is SUCCESS-CONDITIONAL BY CONSTRUCTION.

  FN = 0 follows structurally.  a genuine slot-4 acquisition cannot be missed, because the only
  way to reach T0Q is to have won.

  this is a strictly stronger property than anything the ruled Candidate E or E' had.
```

```text
  R2  @rbx == 4 is DETERMINED as the slot-identity conjunct.  it is producer-local, it is
      success-conditional, and it is correct outside the measurement window.
```

## 3. The candidate that follows, and why it is NOT adopted

```text
  Candidate F, derived, NOT ruled
      @rbx == 4  AND  dwo(@r11 + 0x30010) == 4

    @rbx == 4              slot identity, producer-local, success-conditional
    sequences[4] == 4      owner-exclusive free-slot state, contention-immune

    both conjuncts are WINDOW-INDEPENDENT.  that is the property E and E' both lacked.
```

## 4. THE PROBLEM THAT BLOCKS THE CONTRACT — target granularity versus measurement window

```text
  Q3 = a names the target  "slot-4 acquisition",  the SLOT, recurring every 4096 enqueues.

  observed run length      65 T0E hits
  observed acquisitions    2 CAS_PRE, and both had enqueuePos == 4 in absolute terms

  with @rbx == 4, the conjunct fires when  pos is congruent to 4 mod 4096.
  in a 65-enqueue run, a queue reaches that only if it happens to sit on a 4096 boundary.
  the expected capture count is therefore near zero, and most likely ZERO.

  with @r10 == 4, the conjunct fires when  pos is exactly 4,  which happens once early in each
  queue's life.  that is why the measurement caught 2.

  so:
      @r10 == 4   reliable INSIDE a fresh short run, wrong as a specification
      @rbx == 4   correct as a specification, essentially UNOBSERVABLE at this run length

  neither granularity satisfies both the Q3 = a target and the available measurement.
```

```text
  the target is therefore mis-specified for the evidence available.  three coherent resolutions,
  none adopted here:

  T-1   keep Q3 = a and accept that the capture is usually zero at this run length.
        the gate then measures the absence of acquisitions rather than acquisitions.

  T-2   change the Q3 = a target to the ABSOLUTE position 4, which is what the run length can
        observe.  this contradicts the "slot" framing of Q3 = a and would require Q3 to be
        re-opened, not merely the predicate.

  T-3   change the target granularity entirely, for example to "the next acquisition" with no
        position conjunct at all.  the selector then degenerates from a selector into a
        sampling trigger, and Q4-1 / Q4-2 would have to be re-derived.

  T-1, T-2 and T-3 are incompatible with each other.  this is a Q3-level decision, not a
  predicate-level one, and it is the Owner's.
```

## 5. R3 — T0Q unconditional diagnostic payload

```text
  the requirement itself is DETERMINED as sound, and it is not optional:

      a T0Q capture that FAILS leaves no evidence at the T0Q site, because the  r  dump sits
      inside the acceptance branch.  Step5EE section 1.1 showed the concrete cost: FN could only
      be bounded to {1, 2} instead of being a point value.

      T0E already has the required shape, with the register dump and the window outside the
      guard.  the pattern is proven in this harness.

  R3a  ADOPT the unconditional payload.       DETERMINED

  R3b  its CONTENT is not determined, and it is a further decision:

        minimal   the predicate operands only, r10 and rbx, plus live enqueuePos and
                  sequences[4].  the smallest thing that makes the predicate falsifiable.
        full      the T0E shape, a complete register dump, roughly 7 lines per hit.

        volume    at 65 T0Q hits a full dump adds on the order of 460 lines to a 2,603-line log.
                  minimal adds far less.
        headroom  the full dump also yields rdx, r8 and rsp, which are the entry's ptr, deleter
                  and argument area.  this workstream has twice wished for evidence it had not
                  captured.

        neither is decided here.  R3b is open.
```

## 6. Stop-condition assessment

```text
  R1   NOT RULED.  the premise  pos in 0..4095  is false, and the correct position/slot conjunct
                  had to be re-derived.
  R2   DETERMINED.  @rbx == 4 is the slot-identity conjunct, and it is success-conditional.
  R3a  DETERMINED.  the unconditional payload is required.
  R3b  NOT DECIDED.  minimal or full register dump.
  T-*  NOT DECIDED.  the target granularity, which is a Q3-level choice among three incompatible
                  options, and which changes what Q4-1 and Q4-2 mean.

  UNRESOLVED DESIGN POINTS REMAIN.  therefore, per the Owner's stop condition:

  NEW CONTRACT = NOT AUTHORIZED
  STOP
```

## 7. What is invalidated, superseded or recorded

```text
INVALID            Candidate E    @r10 == 5   unsatisfiable at T0Q
                  Candidate E'   @r10 == 4   window-dependent, wrong as a specification
WITHDRAWN         Step5EE section 5.B, the claim that @rbx == 4 is redundant with a position
                                conjunct.  the two name DIFFERENT quantities
                  Step5ED's FN = 2.  already withdrawn in Step5EE
                  the Owner's re-open instruction section 2 premise, for the same reason
DETERMINED        @rbx == 4 as the slot-identity conjunct
                  @r10 as a DIAGNOSTIC observable only, never as the position conjunct
                  the unconditional T0Q diagnostic payload as REQUIRED
SUPERSEDED        Step5EB contract, which specified @r10 == 5
                  Step5EC Baseline-3, FAILED DESIGN, retained unchanged as evidence
OPEN, NEW         target granularity, Q3-level, three incompatible options
OPEN, NEW         R3b payload content
UNCHANGED         Q1, Q2, Q3 = a as a target statement, Q4-5
                  the RDG-3-2 principle, which was never what failed
                  T1 malformed sites, out of scope
                  the 65 versus 116 hit non-comparability
```

## 8. Consequence for Q4, stated because it is easy to miss

```text
  Q4-1 and Q4-2 were derived under the ABSOLUTE-position reading.
      Q4-1  one acquisition per instance that reaches position 4
      Q4-2  0 <= captures <= 4, from the source-defined instance cardinality

  under the SLOT reading those derivations do not transfer.  the slot recurs every 4096
  enqueues, so the expected count at a 65-hit run length is near zero, and the 0..4 bound
  describes something else.

  if T-2 is chosen, Q4-1 and Q4-2 must be RE-DERIVED, and the 0..4 bound re-checked against the
  instance cardinality of Step5DW rather than reused.

  this is recorded so that neither Q4 nor the predicate is carried forward on a superseded
  derivation.
```

## 9. State

```text
RDG-REOPEN-D1     CLOSED as a gate, with R1 / R3b / T-* unresolved
R2               DETERMINED
R3a              DETERMINED
NEW CONTRACT     NOT AUTHORIZED
cdb edit         NONE
runtime          NONE
build            NOT INVOKED
Baseline-3       unchanged, FAILED DESIGN, retained as evidence
baselines 7/7 + Baseline-3 = 8/8 intact     .cdb 16    git delta 12, all pre-existing
residue 0
```

```text
the next gate is an Owner decision, and it is a Q3-level one, not a predicate-level one:

  T-1  keep Q3 = a, accept a near-zero capture rate at this run length
  T-2  re-open Q3 and retarget to the absolute position
  T-3  change the granularity to a sampling trigger and re-derive Q4

  and, independently, R3b  minimal or full register dump at T0Q

only after those is a new contract, a new Implementation Authorization and a new Runtime
Authorization conceivable.  nothing may be edited on the strength of this document.
```
