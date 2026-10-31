# Contract Amendment Review — Read-Only Adjudication Record

```text
record type        read-only adjudication record
purpose            fix the final state of the Contract Amendment Review after the Owner's
                   adjudication of the seven contract-wording questions
authorised scope   record creation only
subjects modified  0
gate entered       none
next gate          none; this record stops here
```

This record fixes state and carries forward the Owner's rulings. It adjudicates nothing itself.
Where a judgement is recorded below, it is the Owner's, quoted in substance, with the agent's
supporting evidence named. No contract text was changed. No occurrence was renumbered.

---

## 1. The seven contract-wording questions, as adjudicated

The Post-Implementation Review raised seven points where the contract's wording and the
implementation's shape do not line up exactly. It counted **none** of them as a PASS or a FAIL
and referred them to the Owner. The Owner has adjudicated all seven as **PASS against the
current implementation**, on the readings set out below.

These are the Owner's readings of the existing contract. The contract text is unchanged and is
not to be rewritten to fit the implementation.

| # | clause | the question raised | Owner's ruling | the reading adopted |
| --- | --- | --- | --- | --- |
| 1 | A.1 | "declare two **module-level** containers"; the implementation binds both names as attributes of the module-level object `LIFE` | **PASS** | what A.1 requires in substance is that a phase-owned container have module-level lifetime. It does not additionally require the container to *be* a module-level variable. |
| 2 | §2 terms | "container \| a distinct in-memory **list**"; the container is a `Trace` / `SealedTrace` object whose `records` slot is the list | **PASS** | the contract is not read as a literal type specification. A wrapper that holds `records` internally satisfies container semantics, so the physical type need not be `list`. |
| 3 | A.5 | `record.phase`; a record is a 4-tuple and the phase is element 0 | **PASS** | what matters is that the phase is held as an identifiable constituent of the record and that A's ownership and identity requirements are met. Tuple index `0` losing its meaning under future maintenance is a separate concern; no implementation change is required now. |
| 4 | B.3 | "a **producer** shall refuse"; the refusal lives in the container's `append`, one frame below the producer | **PASS** | the R3-1 Contract Review itself admits an equivalent tag check at E.4, so a rejection statement inside the producer body is not a necessary condition. What is required is that the ownership boundary hold: a foreign producer reaching a foreign container or tag meets rejection. |
| 5 | C.1 with §2 | `identity_holds()` names both containers; §2 defines a consumer as "the function that reads a trace container" | **PASS** | `identity_holds` is not treated as an additional consumer for ownership purposes. C's ownership property is which container an actual consumer reads; the reviewer's reference to both containers in order to verify their identity is a different act and is kept distinct. |
| 6 | F.4 | "the self-test container shall **not exist** alongside the RR container"; a `SealedTrace` tombstone leaves the name in place | **PASS** | what is prohibited is reuse of, or write capability over, a closed or sealed trace. A tombstone or name persisting for identification is not prohibited. `name exists` is not `mutable trace exists`. |
| 7 | G.2 | "a self-test gate result"; the implementation's referent is the guard-leg group `S_E` / `S_F` / `_guard_ok` | **PASS** | the contract does not require a particular variable name. The required property is that the self-test functions as a gate and does not enable the review verdict before it holds. That `S_E` / `S_F` / `_guard_ok` carry the property is confirmed by the review's evidence. |

### 1.1 What these seven PASS rulings do not mean

Recorded explicitly so the rulings cannot later be read wider than the Owner made them.

```text
they DO mean       the current implementation is accepted under the current contract's semantics
they DO NOT mean   the contract wording is fully unambiguous
they DO NOT mean   a future implementation would be read the same way
they DO NOT mean   the contract is sufficient as an API or type-level specification
```

A documentation clarification remains available if it is ever needed. It is not performed now.

### 1.2 Consequence for the contract

```text
Contract Amendment on the strength of these seven items : NOT REQUIRED
R3-1 Repair Contract text                                : UNCHANGED, 28805836...
```

Rewriting the R3-1 Repair Contract's wording to match the implementation is prohibited.

---

## 2. R3-1 Post-Implementation Review, as received

```text
Reviewer self-audit SA-1..SA-4       CLEAN
Clause A                             PASS
Clause B                             PASS
Clause C                             PASS
Clause D                             PASS
Clause E                             PASS
Clause F                             PASS
Clause G                             PASS
Falsifiers                           non-vacuous
AXIS 1 implementation conformance   PASS
AXIS 2 provenance completeness      NOT PROVEN
```

### 2.1 The two axes, held separate

```text
AXIS 1   does the current ECCB4396 implementation satisfy R3-1 clauses A to G
         -> PASS

AXIS 2   can the change set from the Owner-authorised 968377BE baseline to ECCB4396
         be proven byte-exact
         -> NOT PROVEN
```

`AXIS 1 PASS` does not entail that the `968377BE → ECCB4396` exact diff is proven. The two are
independent, and neither may be substituted for the other.

In particular, none of the following may be used to promote provenance to PASS:

```text
model digest unchanged
clauses A to G PASS
the falsifiers being non-vacuous
the reviewer's own self-audit being clean
```

And `717E957D → ECCB4396` may not be reused as a baseline diff. Reconstruction of `968377BE`
remains prohibited.

---

## 3. Occurrences

| # | classification | status | note |
| --- | --- | --- | --- |
| #20 | genuine structural defect, trace separation | **OPEN / structural issue addressed** | existing value record retained; see §4 |
| #21 | instrumentation defect | **RECORDED / CORRECTED** | not re-escalated |
| #22 | instrumentation defect | **RECORDED / CORRECTED** | not re-escalated |
| #23 | provenance / artifact-retention process defect, independent of #1..#22 | **OPEN / UNRECOVERED** | see §5 |

`#1` through `#23` are unchanged by this record. No occurrence was added, removed, renumbered,
merged or reduced.

---

## 4. Occurrence #20

```text
self-test protected-model evaluations : 0
RR-phase  protected-model evaluations  : 40
```

The existing record is **retained and was not re-measured**. The R3-1 finding was the label, not
the value; the two are kept apart. Structural trace separation is addressed by R3-1 and was
evaluated in the Post-Implementation Review.

What makes this safe to record is that no additional RR execution occurred:

```text
RR execution performed in the Post-Implementation Review  0
tmp/rr_evaluate_real.py executed                          0   (730C3C0B..., never run)
RR-1..RR-18 re-evaluated                                   0
protected-model counts produced by the Review              0
```

The Post-Implementation Review therefore did not cause an RR-1..RR-18 re-evaluation, and the
existing RR result is neither extended nor relied upon by it.

---

## 5. Occurrence #23

```text
classification : provenance / artifact-retention process defect
relation       : independent of #1..#22
status         : OPEN / UNRECOVERED
exact baseline : NOT RECOVERED
exact diff     : NOT PROVEN
```

This is the most consequential unresolved item and is not softened by AXIS 1. A conforming
implementation is not evidence of what changed to reach it.

The process control introduced for the Post-Implementation Review did not recur: five
pre-change versions of the reviewer are retained byte-exact, none overwritten.

---

## 6. The nine wrong verdicts in the reviewer itself, retained as process evidence

The Post-Implementation Review found nine defects in its own instruments that had produced a
wrong verdict before repair, plus six that stopped the run, plus two sites its own self-audit
caught in its own source.

Owner ruling:

```text
the C.4 vacuous PASS   RETAINED as process evidence, not erased as "already fixed"
the classification    same family as #1..#22, separate from #23
occurrence numbering   HELD.  no new occurrence number is assigned to the nine items.
#1..#23                unchanged
```

The detail remains in
`doc/work113/P1-5-IR-P2_R3-1_POST-IMPLEMENTATION_REVIEW_RESULT.md` §6 and in the process-defect
record's appendix 9. Neither file was modified by this record.

---

## 7. Why the Contract Amendment Review is not closed

The seven wording questions are adjudicated, and the R3-1 conformance question is answered. That
is not the whole of this review. The following remain open:

```text
the existing review history is not closed out
#20   remains OPEN
#21   RECORDED / CORRECTED, not closed out
#22   RECORDED / CORRECTED, not closed out
#23   remains OPEN / UNRECOVERED
RB3   final classification is undecided
```

Therefore:

```text
Contract Amendment Review : NOT CLOSED / NOT TRUSTED
```

This is a state statement, not a recommendation about when it should close.

---

## 8. RB3

```text
RB3 : DEFERRED
```

RB3's final classification was not performed in this step and may not be inferred from the fact
that nine wrong verdicts occurred. The Post-Implementation Review was a conformance review of
R3-1 clauses A to G. RB3 is a separate step.

---

## 9. RR-1..RR-18

```text
existing result  18/18   HOLD
```

Retained as an existing result. It is not re-activated and not reused on the strength of this
Review.

---

## 10. State at the close of this record

```text
R3-1 Repair Contract                ACCEPTED
R3-1 Contract Review                PASS
R3-1 implementation                 DONE / ECCB4396...
R3-1 Post-Implementation Review     PASS  (AXIS 1)
A to G                              PASS
AXIS 2 provenance                   NOT PROVEN

occurrence #20                     OPEN / structural separation addressed
                                    existing value record retained
occurrence #21                     RECORDED / CORRECTED
occurrence #22                     RECORDED / CORRECTED
occurrence #23                     OPEN / UNRECOVERED

7 contract wording questions       OWNER ADJUDICATED / PASS
                                    no contract amendment
                                    contract text unchanged

RB3                                DEFERRED

Contract Amendment Review          NOT CLOSED / NOT TRUSTED

RR-1..RR-18                        18/18 existing result - HOLD, do not reuse

CR-3                               PENDING
R1 execution                       NOT AUTHORIZED
Phase 2-B                          NOT STARTED
```

---

## 11. Artefact identity at the time of this record

```text
28805836D44165ECD0E8C13772654214   13139 B  R3-1 Repair Contract
4BE53811E0D4FA7884ADE1895092F91E   21896 B  R3-1 Post-Implementation Review result
E09A6AE62F633025FF0083D982D9FD37   68555 B  process-defect record, occurrences #1..#23 plus appendices
3658725552EBCA3D5FF398CBE28C8504   23533 B  amended contract under review
B163DF9600B4A3B245C76F2889B95985    9546 B  Amendment 1
E9D1B5138A4A026492DE9D727E3DA939   11428 B  R3-1 Contract Review result
5FA0A4D5F5CA481DE7F605AABDEAAD30   10202 B  R3 independent review result
21ADA62169753C83893F4CB994502C0A    9340 B  RR full re-derivation result
ECCB4396210ED103C79B449EE9EE7EF2   62277 B  the reviewed implementation
BF538B69F09D0BE34FA7A631682549B3   48361 B  the reviewer
730C3C0B03461E7B15F38ED0EBDC0C94   11496 B  the RR driver, not executed
C587352201F0BE3D8F39D30396137A66 5550764 B  ConvoPeq.md, unchanged
```

---

## 12. Prohibitions observed while producing this record

```text
RR-1..RR-18 re-evaluation     0        CR-3                     0
R1 execution                  0        CDB                      0
debuggee                      0        g / go / gc              0
Phase 2-B                     0        RB3 finalization         0
Contract Amendment            0        Amendment 1              0
production source change      0        build                    0
Git operation                 0        ConvoPeq.md regeneration 0
ConvoPeq.md re-pin            0        968377BE reconstruction  0
```

No subject was modified. The only file created by this step is this record.

---

## 13. What remains for the Owner

```text
#23   OPEN / UNRECOVERED.  no further recovery is authorised and none is possible
      without reconstruction, which is prohibited.  the axis stays NOT PROVEN.

#20   OPEN with its structural layer addressed.  its final status is the Owner's to set.

RB3   DEFERRED.  classification is a separate step and may not be inferred from the
      nine wrong verdicts in the reviewer.

Contract Amendment Review   NOT CLOSED / NOT TRUSTED.  closure is the Owner's decision
      and is not proposed here.

Order of any next step, and its authorisation, is the Owner's.  This record stops.
```
