# RDG-1.1 — Owner Decision Gate, decision state

```text
gate                = Owner Decision Gate Q3 / Q4
mode                = READ-ONLY / OWNER DECISION.  No measurement added.
execution this gate = 0
purpose             = record which owner decisions are settled and which are blocked, so that
                      RDG-2 is not started on an incomplete specification
```

## 1. Decision state

| item | subject | state |
|---|---|---|
| Q1 | T0 guard purpose | **CONFIRMED** — A, trigger / capture selector |
| Q2 | literal 9 as a baseline specification | **CONFIRMED** — not established by any record |
| Q4-5 | latch structure | **CONFIRMED** — T0E is not a one-shot latch, T0S is the latch |
| Q3 | semantic target | **CONFIRMED — (a) specific ring position**, recorded in Step5DV |
| Q4-1 | expected capture count | **DERIVED** — one acquisition per instance reaching position 4 |
| Q4-2 | acceptable range | **BLOCKED** — instance count unreconciled; Owner decision required |
| Q4-3 | zero-capture disposition | **DERIVED** — conditional on a measurable existence test |
| Q4-4 | multiple-capture disposition | **DERIVED** — count must equal the instance count |
| RDG-2 | candidate comparison | NOT STARTED, blocked on Q3 |
| RDG-3 | acceptance criteria | NOT STARTED |
| RDG-4 | selected design | NOT STARTED |

## 2. Q4-5, confirmed, and why it was determined rather than asked

```text
0x1f9c550  T0E   reads  @$t0 == 0        writes $t0 : none
0x1f9c5da  T0S   reads  @$t0 == 0        writes r $t0 = 1
no other site in the script writes $t0
```

```text
T0E's entry branch is NOT a one-shot latch.  It can fire repeatedly, bounded only by how many
T0E hits satisfy every conjunct.  The latch is at T0S.

consequence   multiple captures are structurally possible, and their count is a function of
              the selector's firing condition and of owner policy, not of the latch.

measured      the ceiling for enqueuePos == 4 was 4, in both runs.  There is therefore no
              evidence-based ground for a "must fire exactly once" constraint.
```

This is a control-flow fact, not a specification, which is why it was settled from the artifact
rather than posed.

## 3. Why Q4-1 to Q4-4 are blocked, in the owner's terms

The owner's reasoning, recorded because it is the correct order:

```text
Q3  semantic target
      -> whether relevant retirement episodes EXISTED in the run
        -> expected capture count          Q4-1
        -> acceptable range                Q4-2
        -> whether capture = 0 is a result or a failure   Q4-3
        -> whether capture > 1 is acceptable or a defect Q4-4
```

```text
the already-measured rates are OBSERVATION FREQUENCIES, not a specification:

  enqueuePos == 4        4 / 116
  epoch lag == 1        11 / 116
  r9 == globalEpoch    105 / 116

none of them may be adopted as the expected count before the target is fixed.
```

```text
Q4-3 in particular cannot be fixed in the abstract:

  the target existed and capture = 0   -> possible selector failure
  the target did not exist             -> zero capture is normal

without Q3 there is no way to tell those apart, so zero capture is neither PASS nor FAIL
in advance.  It is recorded as UNDEFINED, blocked on Q3.
```

## 4. What Q3 must supply, and what it must not

Q3 fixes **what the selector selects**. It does not choose a predicate.

```text
Q3 does NOT authorise   @r9 == globalEpoch
Q3 does NOT authorise   @r9 == globalEpoch - 1
Q3 does NOT authorise   dropping conjunct 2, or changing conjunct 3
Q3 does NOT authorise   the address rcx-0x1428
Q3 does NOT authorise   any runtime execution
```

Choosing semantic target **b** in particular would not adopt the predicate `@r9 ==
globalEpoch - 1`. It would only fix the target, leaving RDG-2 to compare how that target may be
observed with the operand inventory of RDG-1 section 3.

## 5. Source note

The owner has twice referred to `ConvoPeq(4).md` as the accessible current source. That file does
not exist in this repository; the sole authority is `ConvoPeq.md`, 5,535,334 B, SHA-256
`E5E74200F12784FDF37BE24864F5E4B1…`, and all findings in this lineage are cited against it.

The owner's substantive point is confirmed against that file: the safety condition on retirement is
relational, `epoch < minReaderEpoch` and `isOlder(entry.epoch, minReaderEpoch)`, and no absolute
epoch value, nor a capture-test specification, can be derived from it. The conclusion therefore
stands regardless of which filename is used.

## 6. State

```text
nothing executed, nothing edited, nothing proposed
CDB execution 0    .cdb 15 unchanged    baselines 7/7 intact
ConvoPeq.md / harness / cdb.exe unchanged    git delta 12, all pre-existing
IMPLEMENTATION FORBIDDEN    RUNTIME NOT AUTHORIZED
```

## 7. Q3 is UNDECIDED by design, and the work is parked there

The Owner has declined to rule on Q3, on the grounds that Q3 is an Owner ruling by design. That is
accepted, and the agent does not take the decision by default.

```text
Q3        UNDECIDED BY DESIGN
Q4-1      BLOCKED ON Q3
Q4-2      BLOCKED ON Q3
Q4-3      BLOCKED ON Q3
Q4-4      BLOCKED ON Q3
Q4-5      CONFIRMED
RDG-2     BLOCKED ON Q3
```

```text
RDG-2 is not started and will not be started.  No candidate predicate has been enumerated,
no repair has been proposed, and no runtime has been requested.
```

### 7.1 The ruling that unblocks it, in one line

```text
Q3 = a    a specific ring position
Q3 = b    a specific temporal relationship
Q3 = c    a specific entry class, with the type value stated
```

### 7.2 An alternative the Owner may prefer

If the Owner's intent is for the decision to be made rather than relayed, the Owner may delegate it
**together with a criterion**, and the agent will rule inside that criterion and record the
reasoning. For example, a criterion such as "choose the target under which conjunct 2 carries
production meaning rather than a bare literal" would make the ruling decidable from evidence
already in hand, with no further measurement.

Either route unblocks RDG-2. Absent one of them, the correct state is the one above.

## 9. Q3 became RULABLE in Step5DT. The state is now WAIT FOR OWNER Q3

The blocking reason recorded in §7.1 was that Q3 could not be ruled on the information then held.
That reason has been removed by read-only work, without any execution and without any predicate
being selected. Full evidence: `P1-5-IR-P2_Step5DT_Q3_Ruling-Brief.md`.

```text
Q3 = a   RULABLE   the target's predicate evidence already exists, 4/116 in two runs,
                   the only reproducibility evidence any candidate holds
Q3 = b   RULABLE   the quantity is measured at 11/116; the predicate is not yet chosen
Q3 = c   PARTIALLY RULABLE
                   enum, propagation chain and observability point now established;
                   the hit rate is unmeasured and NOT recoverable from any existing log
Q3 = d   CLOSED    inadmissible under constraint C6, by constraint and not by preference
```

```text
a / b are NOT symmetric in cost, and this is the substance of the ruling:

  Q3 = a   the phase's existing positional bracket already defines the target.  the epoch literal
           is phase-wide, 4 of 4 sites, and redundant against that bracket.  deleting it is a
           consequence of the target, not a second decision.  the gate's rate is bounded by the
           already-measured 4/116.

  Q3 = b   the target names a quantity the phase does not measure.  a conjunct must be defined at
           all four sites, a new address must be authorised, and the joint rate of the new
           conjunct with each site's positional conjuncts is unmeasured, because the gate has
           never opened.

  Q3 = c   both enum values do reach T0E, so the class is not degenerate, but its rate is unknown
           and the observability point db @rsp+0x28 is an ABI inference, not a measurement.
           The source disclaims type as a lifetime authority in three places.
```

Two corrections to earlier records were made in Step5DT and are recorded there, not here:

```text
the claim "the harness output reports worldReclaimCount reaching 10", made in RDG-1 §5.2, is
WITHDRAWN.  No log in doc/work113 reports worldReclaimCount.  The counter exists in source and is
printed by other harnesses, but not by the --measurement=normal run that produced the 116 T0E hits.

the operand "+0x18" in the same RDG-1 paragraph is the struct field offset and is not where type
is read at T0E, because at T0E the entry is still in registers and the slot holds the previous
occupant.
```

### 9.1 Source authority, confirmed

```text
ConvoPeq.md   5,535,334 B
              SHA-256 E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609

this is the sole source authority for the lineage.  ConvoPeq(4).md is not a substitute and does
not exist.  every finding in Step5DT is cited against ConvoPeq.md.
```

### 9.2 What is held back until Q3 arrives

```text
RDG-2 candidate comparison          not started
additional runtime measurement      none requested
predicate selection                 none made
candidate enumeration               none made
source / test / CMake / .cdb edit   none
build                               not invoked
implementation                      FORBIDDEN
runtime                             NOT AUTHORIZED
```

### 9.3 The one-line ruling that releases the hold

```text
Q3 = a
Q3 = b
Q3 = c, type = <Generic | World>   and state whether an unmeasured hit rate is accepted
```

On receipt, the next task is the read-only Q4-1 to Q4-4 acceptance-semantics gate. The observed
rates 4/116, 11/116, 0/232 and 105/116 are observation values and must not be adopted as
specification values; they become acceptance semantics only after the target is fixed.

## 8. State
