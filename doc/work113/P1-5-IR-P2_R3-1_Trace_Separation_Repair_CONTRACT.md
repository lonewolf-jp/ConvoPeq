# Self-Test / RR-Phase Trace Separation Repair Contract

Status: **DRAFTED — awaiting Contract Review. NOT an authorisation to implement.**
Owner instruction: the §4 and §5 clauses of the R3 disposition.
Subject of the repair: occurrence #20, structural layer.
CDB executions: **0**

---

```text
gate                = a CONTRACT.  NOT an authorisation to implement.
mode                = READ-ONLY drafting.  no checker change.  no RR re-evaluation.
                    no CDB.  no debuggee.  no image executed.  no R1 probe.
owner decision      = R3-1 REPAIR CONTRACT AUTHORED; IMPLEMENTATION NOT AUTHORISED
NOT authorised      = trace separation implementation, checker modification, RR re-evaluation,
                      CDB execution, debuggee launch, R1 probe execution, g / go / gc,
                      source modification, build, production runtime, .cdb modification,
                      Git operation, Contract Amendment, Amendment 1 modification,
                      ConvoPeq.md modification, Phase 2-B
purpose             = specify, in mechanically checkable terms, how the self-test phase and
                      the RR-evaluation phase are prevented from sharing trace storage
```

## 0. What this contract is, and what it is not

```text
IS
    a specification of phase-separated trace storage, production, consumption, identity,
    contamination detection and lifecycle, for the RB1/RB2 checker.

IS NOT
    an implementation.  this document changes no file.
    an authorisation.  implementing it requires a separate Owner authorisation.
    a change to the RR instrument model.  descriptor count, instrument count and both
    digests are fixed and are restated in section 10 as invariants.
```

## 1. The defect being repaired

```text
occurrence #20, structural layer
    the self-test phase and the RR-evaluation phase append to ONE container, EVAL_TRACE.
    they are distinguishable only by call site, not by storage.
```

R3 established, independently:

```text
self-test protected-model evaluations = 0
RR-phase  protected-model evaluations = 40
RR-1..RR-18                           = 18 / 18
```

Those numbers show **no contamination occurred in this run**. They do not show that
contamination is *detectable*. R3-6c showed why: a container poisoned with 999 fabricated
records left every verdict unchanged, so verdict invariance cannot serve as the argument for
separation. **Verdict independence and trace separation are different properties.**

This contract repairs the second. It does not rely on the first.

## 2. Terms

| term | meaning, fixed by this contract |
| --- | --- |
| **phase** | one of `SELF_TEST`, `RR_EVAL` |
| **producer** | the function that appends a record to a trace container |
| **consumer** | the function that reads a trace container |
| **container** | a distinct in-memory list owned by exactly one phase |
| **record** | a tuple appended by a producer, carrying `(phase_tag, model_id, rr, key)` |
| **lifecycle** | the `OPEN` / `CLOSED` state of a container, with validated transitions |
| **detection** | a raised `TraceContamination` signal, not a changed verdict |

**Every term above is used in exactly one sense throughout this contract.** A term used in a
second sense is a defect in this contract.

## 3. Clause A — independent storage

```text
A.1  The implementation shall declare two module-level containers, named exactly
       SelfTestEvalTrace  and  RRPhaseEvalTrace.

A.2  They shall be two distinct objects, created independently.  Neither shall be an alias,
     a view, a slice, a copy-on-write handle or a re-binding of the other.

A.3  No third container shall hold evaluation records.  There shall be no container whose
     identity is shared between the two phases, and no module-level mapping, attribute or
     closure cell from which either phase can reach the other's records.

A.4  Each container shall be tagged with its owning phase at creation:
       SelfTestEvalTrace.phase == "SELF_TEST"
       RRPhaseEvalTrace.phase   == "RR_EVAL"

A.5  Records shall carry the phase of their producer:
       record.phase == the owning phase of the container they were appended to.

observation that establishes A
    the two containers exist simultaneously, are unequal, and each carries its own tag.

falsifier for A
    any arrangement in which a record appended under SELF_TEST is reachable from
    RRPhaseEvalTrace, or in which the two names denote the same object.
```

## 4. Clause B — producer ownership

```text
B.1  There shall be exactly two producers, one per phase.  Each producer shall be
     hard-bound to its own container.

B.2  A producer shall NOT take a container argument.  A producer whose target can be chosen
     by its caller does not satisfy B, because the phase would then be a caller-supplied
     value rather than a structural property.

B.3  A producer shall refuse to append a record whose phase tag differs from the container's
     own phase tag, and shall raise TraceContamination on the attempt.

B.4  The self-test phase shall not be able to reach RRPhaseEvalTrace through any producer,
     and vice versa.  This shall be a property of the binding, not of call discipline.

observation that establishes B
    a static inspection of the two producer bodies shows each writes only to its own
    container, and neither accepts a container parameter.

falsifier for B
    a producer with a container parameter, or a producer reachable from the other phase.
```

## 5. Clause C — consumer ownership

```text
C.1  Each consumer shall name exactly one container.  A consumer that reads both containers
     is prohibited.

C.2  The self-test's protected-model evaluation count shall be a function of
     SelfTestEvalTrace only.

C.3  The RR-phase's protected-model evaluation count shall be a function of
     RRPhaseEvalTrace only.

C.4  There shall be no consumer whose result is reported without naming which phase's
     container it read.  A report line that omits the phase is a defect in the report.

observation that establishes C
    the two count expressions each reference exactly one container name, and no count
    expression references both.

falsifier for C
    a count that is a function of both containers, or a reported count whose phase is not
    stated at the point of report.
```

## 6. Clause D — object identity

```text
D.1  The implementation shall assert, at construction and again at every phase transition,
       SelfTestEvalTrace is not RRPhaseEvalTrace

D.2  A comparison of names, of lengths, of contents, or of digests does NOT satisfy D.
     Only an identity test on the container objects does.

D.3  The identity assertion shall be re-evaluated after each transition of section 8, not
     only once at start-up.

observation that establishes D
    an identity assertion is present at each required point and passes.

falsifier for D
    separation asserted by name, by length, by content, or by digest.
```

## 7. Clause E — cross-contamination detection

```text
E.1  The implementation shall detect contamination in BOTH directions.  Two tests, mandatory:

       E.1a  append a record whose phase tag is "RR_EVAL" to SelfTestEvalTrace
             -> TraceContamination shall be raised
       E.1b  append a record whose phase tag is "SELF_TEST" to RRPhaseEvalTrace
             -> TraceContamination shall be raised

E.2  Detection means a raised signal.  A test in which the contaminated run merely produces
     the same verdict, or the same count, or the same report text, does NOT satisfy E.
     Verdict invariance is a different property and is prohibited as the argument here.

E.3  The two tests shall be run against the repaired implementation, and the raised signal
     shall be observed.  A test that is written but never executed does not satisfy E.

E.4  The detection mechanism shall be the producer-side or consumer-side tag check of
     B.3 and C.1, or an equivalent tag check.  Detection by "the number looks wrong" is
     prohibited, because a balanced contamination would evade it.

observation that establishes E
    both injections, and a raised TraceContamination for each, recorded with the phase tags
    that were injected.

falsifier for E
    an injection that produces no raised signal, or a separation argument that rests on
    verdict invariance, count invariance or report-text invariance.
```

## 8. Clause F — lifecycle

```text
F.1  Each container has a lifecycle state, OPEN or CLOSED.

F.2  Transitions, in this order, and no other:

       F.2a  SelfTestEvalTrace  is created OPEN
       F.2b  the self-test phase runs
       F.2c  SelfTestEvalTrace  is closed, and the identity assertion of D.1 is re-run
       F.2d  RRPhaseEvalTrace    is created OPEN
       F.2e  the RR phase runs
       F.2f  RRPhaseEvalTrace    is closed, and the identity assertion of D.1 is re-run

F.3  An append to a CLOSED container shall raise.  A phase may not write after it has closed.

F.4  F.2c shall execute before F.2d.  The self-test container shall not exist alongside the
     RR container, so that a record cannot be written to a container whose phase has ended.

F.5  Reading a CLOSED container is permitted, for reporting.  Reading the other phase's
     container is prohibited at all times, per C.1.

observation that establishes F
    the transition order is enforced by the implementation, and an append after close raises.

falsifier for F
    a write to a closed container that succeeds, or an RR container created before the
    self-test container is closed.
```

## 9. Clause G — trace is not an input to any verdict

```text
G.1  An RR verdict shall be a function of instrument return values only.  No trace
     container, no trace length, no record, and no reported value shall be an input.

G.2  A self-test gate result shall be a function of the self-test phase's own observations
     only.

G.3  G shall be verified separately from clauses A to F.  Passing G does not establish A to
     F, and passing A to F does not establish G.  The two are independent properties and
     this contract requires both.

observation that establishes G
    the verdict expression is shown to read no trace name, and the self-test gate
    expression is shown to read only its own phase's observations.

falsifier for G
    any verdict or gate expression that reads a trace container.
```

## 10. Invariants this repair must not break

```text
descriptor count     18
descriptor digest    E6E758784020CE82F7408C387F1D5CDB0AAEF0A26D142E8F382E88B3EC63F442
instrument count     40
instrument digest    A5D93923D73B5E4226A146ECB66B3612E076D67B8868811F9A30DC2CA2345ED4
property types       C / D / E / F / G as primary;  A and B supporting only
primary assignment   unchanged
hand-written needles / regions / hashes   0 / 0 / 0

amended contract     23,533 B
                     3658725552EBCA3D5FF398CBE28C850491B3AFE8E7A988E00F19A096B5E05B53
Amendment 1          9,546 B
                     B163DF9600B4A3B245C76F2889B95985767D17E05F960D27F67B1308508FA8F2
```

A change to any line of this section is a change of scope and is prohibited by this contract.

## 11. Out of scope

```text
This contract does NOT specify, and this repair does NOT touch:

    the RR descriptor set          the property type assignment
    the primary instrument set     the instrument bodies
    the §18 real-model boundary    the guard legs
    the evidence recorder          the parser
    the amended contract           Amendment 1
    the .cdb corpus                the Release harness
    ConvoPeq.md                    Git
```

The §18 boundary is restated as an invariant and is **not** relaxed by this repair.
In particular, no real-model evaluation may occur during the self-test phase; that property
is a precondition of this contract and is restated as R3-1-P8 in the Contract Review.

## 12. Acceptance of the implementation

```text
The implementation of clauses A to G is NOT authorised by this contract.

Order, fixed:
    1. this contract is drafted                          (done)
    2. Contract Review of this contract                 (next, read-only)
    3. STOP, and the Owner rules on the Review
    4. only then a separate implementation authorisation
```

The Contract Review checklist is R3-1-P1 .. R3-1-P8, fixed by the Owner and reproduced in
the Review document. **This contract does not define its own acceptance criteria.**

## 13. Prohibitions carried into the implementation

```text
CDB launch                  0        source modification        0
debuggee launch             0        build                      0
g / go / gc                 0        production runtime         0
R1 probe execution          0        .cdb modification          0
Git operation               0        Contract Amendment         0
RR re-evaluation            0        Amendment 1                0
Phase 2-B                   0        ConvoPeq.md                0
R1 execution authorisation  0
```

The implementation authorised under section 12.4 is a change to the reviewer's own
tooling. It carries none of the above.
