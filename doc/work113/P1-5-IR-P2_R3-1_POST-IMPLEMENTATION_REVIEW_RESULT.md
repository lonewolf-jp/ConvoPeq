# R3-1 Post-Implementation Review — Result

```text
review type            read-only Post-Implementation Review
authorised by          the Owner, after occurrence #23 was ruled CASE B
reviewed artefact      tmp/rz2c_instruments.py
contract reviewed      doc/work113/P1-5-IR-P2_R3-1_Trace_Separation_Repair_CONTRACT.md
reviewer               tmp/r3_1_post_impl_review.py
records                57 OK / 1 FAIL
the single FAIL        AXIS-2, which is occurrence #23 stated as a verdict, not a defect
```

This review verifies the **current** implementation against clauses A to G of the accepted
Repair Contract. It does **not** verify `968377BE → ECCB4396`. That baseline is unrecovered,
the exact diff is NOT PROVEN, and occurrence #23 is unchanged by anything recorded here.

---

## 1. Identity, pinned before anything else

```text
target path                    tmp/rz2c_instruments.py
target size                    62277 B
target sha256                  ECCB4396210ED103C79B449EE9EE7EF2BD4775206E618BAFABF6BB698BA4C195   MATCH the pin
contract sha256                28805836D44165ECD0E8C13772654214361E10F047D146C6A90DE7CA14A8B53B   MATCH
reviewer sha256                BF538B69F09D0BE34FA7A631682549B3F54FB0E74D339C6D589CEF01D9BB0F97
reviewer size                  48361 B
```

Re-verified at the end of the review, unchanged:

```text
END-1  the target is byte-identical to the pin it was reviewed under      OK
END-2  the reviewer is byte-identical to its pristine pre-run copy         OK
END-3  every archived pre-change reviewer is retained byte-exact           OK
END-4  the contract is byte-identical to its accepted digest               OK
```

### 1.1 The #23 process control, and it fired

The Owner's §7 required that a change to the reviewer itself retain the pre-change artefact
byte-exact. That control is implemented in the reviewer itself, and it fired four times.

```text
r3_1_post_impl_review.v1.PRISTINE.py    88444953B96029AC...   the v1 draft, archived before v2 existed
r3_1_post_impl_review.PRISTINE.1.py      9D1836F3777FB7BD...   v2 as first written
r3_1_post_impl_review.PRISTINE.2.py      41BE92B0B3951B42...   after the first round of instrument repairs
r3_1_post_impl_review.PRISTINE.3.py      59C302863CAFE95E...   after the second round
r3_1_post_impl_review.PRISTINE.py        BF538B69F09D0BE3...   the version that produced this result
```

No version was overwritten. This is precisely the failure that produced occurrence #23, and it
did not recur.

### 1.2 Model identity, recomputed rather than read

```text
descriptor   18   E6E758784020CE82F7408C387F1D5CDB0AAEF0A26D142E8F382E88B3EC63F442   OK
instrument   40   A5D93923D73B5E4226A146ECB66B3612E076D67B8868811F9A30DC2CA2345ED4   OK
```

Recomputed by the reviewer from the imported module. The implementation's own output during
import was 6509 characters, **captured and discarded**. No verdict in this review is taken from
it.

---

## 2. Independence

```text
static evidence      read out of the implementation's own source through ast
live evidence        observed by this reviewer constructing objects and driving them,
                     watching for a raised exception
the singleton LIFE   never used as evidence for any clause verdict
the checker's stdout captured and discarded
```

Importing the module executes the checker's built-in self-test suite as a side effect. That is
disclosed rather than hidden. It is **not** an RR evaluation: RR-1..RR-18 live in
`tmp/rr_evaluate_real.py`, which this review did not run. For the same reason the reviewer never
called `self_test_protected_count()` or `rr_protected_count()`, so occurrence #20's figures were
not re-measured.

---

## 3. Clauses A to G

| clause | property | verdict |
| --- | --- | --- |
| A | independent storage | **PASS** |
| B | producer ownership | **PASS** |
| C | consumer ownership | **PASS** |
| D | object identity | **PASS** |
| E | cross-contamination detection | **PASS** |
| F | lifecycle | **PASS** |
| G | trace is not an input to any verdict | **PASS** |

### 3.1 A — independent storage

| item | verdict | evidence |
| --- | --- | --- |
| A.1 two containers, exact names | OK | `SelfTestEvalTrace` and `RRPhaseEvalTrace` both present, on the module-level `LIFE` |
| A.2 distinct objects, no alias | OK | identity test holds; non-vacuity proved by A.2-falsifier |
| A.3a no module-level route between phases | OK | 0 violations over 80 module-level bindings; falsifier OK |
| A.4 phase tag at creation | OK | both classes assign `self.phase` in `__init__`; A.4x live: `SELF_TEST / RR_EVAL` |
| A.5 record carries its container's phase | OK | A.5x live: `self[0]=SELF_TEST rr[0]=RR_EVAL` |
| A.3c distinct record lists, owner holds no list | OK | `records` do not alias; no list among the owner's attributes |

A.3a is deliberately narrower than a naive "no shared global" scan. A global is a violation only
if it **can receive records** and is reached by more than one phase function, or if it binds a
container and is reachable from a phase function. The owner object `LIFE`, which every phase
legitimately reaches, holds no list and opens no route. The first version of this instrument did
not make that distinction and produced a false FAIL; see §6.

### 3.2 B — producer ownership

| item | verdict | evidence |
| --- | --- | --- |
| B.1 exactly two producers | OK | `self_test_producer`, `rr_producer` |
| B.2 no container argument | OK | `(mi, m, extra_protected)` / `(mi, m)` |
| B.3 refusal raises TraceContamination | OK | B.3x, B.3y |
| B.4 structural binding | OK | self-test producer names only `SelfTestEvalTrace`; RR producer only `RRPhaseEvalTrace` |

```text
B.3x  RR_EVAL offered to the self-test sink
      -> TraceContamination: record tagged RR_EVAL offered to the SELF_TEST container
         raised in rz2c_instruments.append
B.3y  SELF_TEST offered to the same sink -> accepted, so B.3x discriminates
```

### 3.3 C — consumer ownership

| item | verdict | evidence |
| --- | --- | --- |
| C.1 no consumer names both | OK | 4 record-reading functions; none names both |
| C.2 self-test count from SelfTestEvalTrace only | OK | `sum((1 for r in c.records if r[1] in _protected_ids()))` |
| C.3 RR count from RRPhaseEvalTrace only | OK | same expression, bound to the other container |
| C.4 every reported count names its phase | OK | 1 report statement carries a count; 0 without a phase name |

Negative controls, using the consumer's **own** expression unparsed from its AST:

```text
C.2x  self-test count 1 -> 1 while the RR count moved 0 -> 1
C.3x  the RR count rose only when the RR container was written
```

### 3.4 D — object identity

| item | verdict | evidence |
| --- | --- | --- |
| D.1 assertion at construction and every transition | OK | 5 call sites in `TraceLifecycle` |
| D.2 identity test, not a surrogate | OK | `is not self.RRPhaseEvalTrace`; no len / sha / digest / repr / `==` |
| D.3 re-evaluated after each transition | OK | counter sequence `[1, 2, 3, 4, 5]` across construction and four transitions |
| D.x non-vacuity | OK | binding both names to one object raises `AssertionError: clause D: the two containers are the same object` |

### 3.5 E — cross-contamination detection

| item | verdict | evidence |
| --- | --- | --- |
| E.1a RR_EVAL into SelfTestEvalTrace | OK | `TraceContamination`, raised in `append` |
| E.1a-closed the same into the closed container | OK | `TraceContamination`, the more specific of the two signals |
| E.1b SELF_TEST into RRPhaseEvalTrace | OK | `TraceContamination`, raised in `append` |
| E.2 detection is a raised signal, not invariance | OK | innermost frame is the container's `append`; the whole test is `if rec[0] != self.phase:` |
| E.3 executed and observed, not merely written | OK | 5 injections raised, 2 correctly tagged controls accepted, on a lifecycle the reviewer built |
| E.4 mechanism is a tag check | OK | a balanced contamination cannot evade a tag comparison |

No verdict or count invariance is used anywhere as the argument, per E.2.

### 3.6 F — lifecycle

| item | verdict | evidence |
| --- | --- | --- |
| F.1 OPEN / CLOSED states | OK | `Trace.__slots__=('phase','records','open')`, `SealedTrace.__slots__=('phase','count','digest','sealed')` |
| F.2 order and no other | OK | observed and declared both `INIT → SELF_TEST_OPEN → SELF_TEST_CLOSED → RR_OPEN → RR_CLOSED` |
| F.2-order every out-of-order transition refused | OK | close before open, `open_rr` before the self-test opens, repeated open, close before `open_rr` — all `AssertionError` |
| F.3 append to CLOSED raises | OK | `TraceClosed` in both phases |
| F.4 F.2c before F.2d | OK | `open_rr` refused while the self-test is OPEN; the slot is writable then and `SealedTrace` after |
| F.5 reading a CLOSED container permitted | OK | `count=2`, equal to the records written before the close, and a digest |
| F.5x a phase may still write to its own OPEN container | OK | no raise |

### 3.7 G — trace is not an input to any verdict

| item | verdict | evidence |
| --- | --- | --- |
| G.1 no verdict line reads a trace identifier | OK | 0 of 19 verdict lines |
| G.2a no self-test gate leg reads a trace identifier | OK | 0 of 4 gate legs |
| G.2b no RR observation can enter a gate value | OK | 0 legs call `rr_producer` unguarded |
| G.3 decided independently of A to F | OK | each clause is its own boolean group |
| G.x non-vacuity | OK | unmutated 0 leaks, mutated 1 leak |

G.2b is a real test, not a tautology. It resolves the parent of every leg assignment and requires
that any leg calling `rr_producer` sit inside a `RealModelForbidden` guard, so the RR producer can
appear only as the subject of an expected refusal.

---

## 4. Falsifiers — every structural instrument proved non-vacuous

A PASS is worthless if the instrument could not have failed. Each was checked by breaking the
property on a copy and requiring the instrument to report it.

| falsifier | result |
| --- | --- |
| A.2 identity | binding both names to one object raises `AssertionError` |
| A.3a shared sink | planting a module-level `EVAL_TRACE` written by **both** producers — the removed shape itself — makes A.3a report 1 violation; the real source reports 0 |
| E discrimination | both correctly tagged controls accepted, so the contamination raises are specific |
| G.1 | mutating `S5` to read `LIFE.RRPhaseEvalTrace` makes G.1 report 1 leak; the real source reports 0 |

---

## 5. Contract-wording items — raised, not resolved

Seven points where the contract's wording and the implementation's shape do not line up exactly.
**None is counted as a PASS or a FAIL.** A PASS on a clause above means the substance was
verified, not that the wording was ruled on. These rulings are the Owner's.

| # | clause | the question |
| --- | --- | --- |
| 1 | A.1 "declare two **module-level** containers" | the implementation binds both names as attributes of the module-level object `LIFE`, so the reachable names are `LIFE.SelfTestEvalTrace` / `LIFE.RRPhaseEvalTrace`. They are not module-level bindings in the literal sense. This review read A.1 as satisfied on substance: two containers, exact names, reachable at module scope. |
| 2 | §2 "container \| a distinct in-memory **list**" | the container is a `Trace` / `SealedTrace` object whose `records` slot is the list. Distinct and phase-owned, but not itself a list. |
| 3 | A.5 `record.phase` | a record is a 4-tuple and the phase is element 0. §2 already defines a record as a tuple carrying `(phase_tag, model_id, rr, key)`, so this reads as shorthand. The property holds either way. Recorded because §2 says a term used in a second sense is a defect in the contract. |
| 4 | B.3 "a **producer** shall refuse" | the refusal lives in the container's `append`, one frame below the producer. E.4 expressly permits "an equivalent tag check", so this review read B.3 as satisfied. The producer's only append path does go through the check. |
| 5 | C.1 with §2 "consumer \| the function that reads a trace container" | `identity_holds()` names both containers. It reads no record and returns a distinctness boolean; it is D.1's assertion exposed as a function, and C.1's own observation and falsifier are both about count expressions. This review excluded it from the consumer set. Under §2's broad wording it could be counted as a consumer. |
| 6 | F.4 "the self-test container shall **not exist** alongside the RR container" | after the close the name still resolves, to a `SealedTrace` tombstone of a different class. What ceases to exist is a **writable** self-test container, not the name. F.4's falsifier — an RR container created before the self-test container is closed — is satisfied and was tested. This item was already open before this review. |
| 7 | G.2 "a self-test gate result" | the implementation has one self-test gate: the guard-leg group `S_E` / `S_F` / `_guard_ok` over `GUARD_LEGS`. This review took that as G.2's referent. |

---

## 6. Defects found in this reviewer, and corrected

The Owner's §6 required confirmation that this review's own instruments do not carry the
occurrences #21 and #22 shapes. They did. **Nine of this reviewer's instruments produced a wrong
verdict before being repaired**, and the repairs are part of the record, not a silent fix.

The recurring shape is the established family: **the shape of the judging instrument does not
match the property it claims to test.**

### 6.1 Instruments that produced a wrong verdict

| instrument | what it did | why the shape was wrong |
| --- | --- | --- |
| C.4 report scan | 0 report lines found, 0 without a phase, **OK** | a line-based filter missed a `print` whose count call sat on the next line. It returned OK on an **empty set** — a vacuous PASS, the most dangerous class. Now AST-based over `print` calls, and it requires at least one report to exist. |
| A.3b alias scan | 5 false FAILs | flagged any module-level assignment *mentioning* a container name, including a tuple of name-strings, a label list, and a reporting alias. Bound nothing. Replaced by A.3a, which asks whether a global can receive records or binds a container. |
| A.3a, first form | 1 false FAIL on `LIFE` | treated "reached by 2+ phase functions" as the violation, ignoring that the owner object holds no list and opens no route. |
| A.4 phase-tag scan | 1 false FAIL | assumed the form `self.phase = phase`. `SealedTrace` uses a tuple unpack. Now an AST test for the attribute assignment. |
| C.1 consumer scan | 1 false FAIL on `identity_holds` | scanned every function for "names both containers" rather than scanning consumers, i.e. functions whose value derives from a container's records. Now restricted to `.records` readers; `identity_holds` is surfaced as wording item 5. |
| D.3 counter | 1 false FAIL | expected `[0,1,2,3,4,5]`; the counter increments inside the assertion, so the true sequence is `[1,2,3,4,5]`. The property held; the expectation was wrong. |
| F.2 order | 1 false FAIL | recorded the lifecycle state after two *refused* probes, producing a duplicated `INIT`. Now state is recorded only after a transition that actually occurred. |
| F.5 closed read | 1 false FAIL | evaluated the sealed-container probe **before** the close, so it tested a live container. |
| END-3 retention | 1 false FAIL | counted the current pristine copy as a "pre-change version", which it is not by construction. |

### 6.2 Defects that stopped the run rather than misreporting

| defect | effect |
| --- | --- |
| the function lookup did not descend into classes | `_assert_identity` is a method, so the D instrument found nothing and the run stopped |
| `identity_holds` arity assumed wrongly | C instrument would have misread the signature |
| the clause F probe order placed the out-of-order `open_rr` after the legal point | the probe succeeded and the next legal call then raised, stopping the run |
| the falsifier plant anchored on `def _emit(` | the planted source did not parse |
| the retention helper moved the pristine copy without recreating it | the run stopped on a missing file |
| G.2b looked at `_leg = {}` instead of the leg assignments | would have reported a false FAIL |

### 6.3 What the self-audit caught

The reviewer's SA-2 control flagged **two sites in the reviewer's own source**: a 28-character
literal tested with `in` against raw source text, which is occurrence #22's wrap-sensitive-needle
shape. Both were routed through the wrap-insensitive `has()` helper. The control was not
decorative; it fired on the instrument that carries it.

### 6.4 What this means for the numbering

These are instances of the #1..#22 family, not of #23. They have **not** been numbered, merged, or
folded into any existing occurrence. Whether they become a new occurrence, and with what number,
is the Owner's decision. The Owner's instruction was to confirm this review's instruments do not
carry the #21 and #22 shapes; the honest answer is that they carried nine instances, all of which
were found by the review's own controls or by an observed contradiction, and all of which are
recorded above.

---

## 7. Occurrence #20

The recorded figures are **retained and not re-measured**.

```text
self-test protected-model evaluations : 0
RR-phase  protected-model evaluations  : 40
recorded at                          : R3, by that review's own independent reconstruction
```

The R3-1 finding was the **label**, not the value. The two are held apart:

```text
value : 0 and 40, confirmed at R3, retained as RECORDED
label : one reported line named 40 as a self-test quantity.  that label was wrong.
```

Not re-measured here. Reading them again means running the checker over the protected models,
which is an RR run, and RR re-evaluation is not authorised. This review therefore never called
`self_test_protected_count()` or `rr_protected_count()`.

What **is** established here is the structural layer #20 left open:

```text
two phase-owned containers           A, A.3a..c, A.4x, A.5x
two producers bound to their own     B, B.2, B.4, B.3x, B.3y
two single-container consumers       C, C.1..C.4, C.2x, C.3x
an object-identity assertion         D, D.1, D.2, D.3, D.x
a raise on contamination, both ways E.1a, E.1b, E.1a-closed
no shared record sink in the source  A.3a, A.3c, A.3-falsifier
```

## 8. Occurrences #21 and #22

```text
#21  RECORDED / CORRECTED   not re-escalated
#22  RECORDED / CORRECTED   not re-escalated
```

Neither was re-raised against the current implementation. §6 above is a statement about **this
review's** instruments, not about the implementation's, and is kept separate from #21 and #22 for
that reason.

## 9. Occurrence #23

```text
classification : provenance / artifact-retention process defect
relation       : independent of #1..#22
status         : OPEN / UNRECOVERED
exact baseline : NOT RECOVERED
exact diff     : NOT PROVEN
```

Not hidden, and not softened by the conformance result. §1.1 records that the process control
introduced for this review did **not** recur.

---

## 10. Two axes

| axis | question | verdict |
| --- | --- | --- |
| **AXIS 1** Implementation conformance | does `ECCB4396` satisfy contract clauses A to G | **PASS** — A B C D E F G each satisfied, reviewer self-audit clean, all falsifiers non-vacuous |
| **AXIS 2** Provenance completeness | can the change set from `968377BE` be proven byte-exact | **NOT PROVEN** — baseline not recovered, no diff attempted |

These are independent. **AXIS 1 must not be read as bearing on AXIS 2.** A conforming
implementation is not evidence of what changed to reach it.

---

## 11. Prohibitions observed

```text
R1 execution                  0
CDB launched                  0
debuggee launched             0
g / go / gc                   0
production runtime execution  0
RR-1..RR-18 re-evaluation     0
CR-3                          0
Phase 2-B                     0
Contract Amendment            0
Amendment 1                   0
ConvoPeq.md regeneration      0
ConvoPeq.md re-pin            0
production source change      0
build                         0
Git operations                0
968377BE reconstruction       0
```

Verified unchanged after the review:

```text
amended contract      3658725552EBCA3D5FF398CBE28C850491B3AFE8E7A988E00F19A096B5E05B53   OK
Amendment 1           B163DF9600B4A3B245C76F2889B95985767D17E05F960D27F67B1308508FA8F2   OK
R3-1 contract review  E9D1B5138A4A026492DE9D727E3DA93930E7FF530A48F771D4CABF61AA35D9AD   OK
R3 independent review 5FA0A4D5F5CA481DE7F605AABDEAAD30500DFB46E1FC09A9D00214C1A7A8634F   OK
RR re-derivation      21ADA62169753C83893F4CB994502C0AB005531F205A3D594B745465B23547DA   OK
.cdb corpus           16 files, unmodified
git HEAD              e565019d, 0 git operations
cdb / harness         0 processes
ConvoPeq.md           C587352201F0BE3D..., unchanged
```

`tmp/rr_evaluate_real.py` (`730C3C0B…`) was **not executed**.

---

## 12. Result

```text
reviewer self-audit SA-1..SA-4    CLEAN
clause A  independent storage     PASS
clause B  producer ownership      PASS
clause C  consumer ownership      PASS
clause D  object identity         PASS
clause E  contamination detection PASS
clause F  lifecycle               PASS
clause G  no verdict dependence   PASS
falsifiers                         all non-vacuous
AXIS 1  implementation conformance PASS
AXIS 2  provenance completeness    NOT PROVEN   (unchanged; occurrence #23)

records  57 OK / 1 FAIL
the single FAIL is AXIS-2, which is occurrence #23 stated as a verdict rather than a defect
```

Seven contract-wording items are raised in §5 and await the Owner's ruling. Nine defects in this
reviewer's own instruments are recorded in §6 and await the Owner's ruling on numbering.

**The review stops here.** RB3, Contract Amendment Review CLOSED, CR-3, RR re-evaluation, R1
execution and Phase 2-B were not entered.
