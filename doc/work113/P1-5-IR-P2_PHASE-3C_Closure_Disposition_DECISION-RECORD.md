# Phase 3-C — Closure Disposition Decision Record / read-only

```text
gate                = DECISION INSTRUMENT.  No option is selected here.
                      D1 / D2 / D3 are recorded as OPEN with their consequences fixed.
                      Selecting one is the Owner's act alone.
mode                = READ-ONLY.  no re-pin, no regeneration, no checkout, no build, no CDB,
                      no new execution, no record amendment.
precondition        = Closure Validation executed once under authorization.
                      Result  CV-1 PASS, CV-2 FAIL, CV-3 PASS, CV-4 PASS,
                              CV-5 G2-UNCHANGED, CV-6 PASS, CV-7 PASS
                      Phase 3-B NOT CLOSED
                      A1's premise INTACT, re-confirmed by digest in that same run
```

## 0. Why a decision is needed at all

```text
  the gate returned CV-2 = FAIL.

  the substantive conditions of CV-2 were all measured and all met in the single authorised
  run:
      len(extracted)          5,414,041     MATCH
      sha1(blob header + data) 5907a1c3…    MATCH
      CR bytes in LF blob     0             OK
      CRLF SHA-256            E5E74200…     MATCH
      CRLF size               5,535,334     contract said 5,533,334   MISMATCH

  a byte sequence with a given SHA-256 has a determined length.  the digests agree, so the
  lengths agree, so the size condition is satisfied and the MISMATCH is spurious.

  the defect is in the CONTRACT, written by the agent:
      CRLF_SIZE = 5,533,334      correct value 5,535,334
      one digit, at the fourth position, a 3 written where a 5 belongs

  the agent does not get to say "it substantively passed".  the contract said 5,533,334, the
  check compared against 5,533,334, and it reported FAIL.  the only question is which of the
  three dispositions the Owner adjudicates, and the agent's opinion on that is offered in
  section 5 as information, not as a recommendation.
```

## 1. D1 — contract defect, corrected, no re-execution

```text
  D1 IS TWO DISTINCT ACTS, and the Owner must perform the first one
      the first act is ADJUDICATION, not correction.
          the Owner adjudicates that the contract's expected value was wrong.
          it is NOT a post-hoc declaration that CV-2 substantively passed.  those are
          different statements, and only the Owner can make the first.

      the second act is the mechanical correction, 5,533,334 -> 5,535,334, after which
      closure is re-decided on the evidence already gathered.

  RE-EXECUTION      not required.  and the reason is evidentiary, not procedural:
                    the SHA-256 match was measured INSIDE the authorised run, before any
                    diagnostic existed.  it is dispositive of byte identity on its own.

  WHAT D1 CLOSES
      CV-2 becomes PASS, on evidence already in hand, with no second invocation.

  WHAT D1 COSTS
      a contract's stated PASS condition is amended after the fact.  that is a real weakening
      of the separation between rule and outcome, and it should be entered knowingly.

  D1 FIELDS
      D1-1  adjudication   the contract's expected value was wrong        :  __________
      D1-2  corrected value accepted, 5,535,334                            :  __________
      D1-3  re-decide closure on the existing evidence, no re-execution   :  __________
```

## 2. D2 — the contract is binding, FAIL stands, a new gate is cut

```text
  CV-2 = FAIL stands.  Phase 3-B = NOT CLOSED stands.

  the measurements from this run are NOT discarded.  they are retained as evidence, and the
  new gate uses the corrected expected value.

  MINIMUM CONDITION FOR THE NEW GATE, as the Owner stated
      CV-2 expected CRLF size = 5,535,334

  ONE ADDITION THIS INCIDENT EARNED, offered as a concrete improvement rather than a
  requirement, and recorded here so it is not lost
      the new gate should assert the invariant BEFORE it compares, and fail loudly if the
      invariant does not hold:

          pinned size  ==  LF blob size  +  line count
          5,535,334    ==  5,414,041     +  121,293

      this invariant was available throughout and was never asserted.  asserting it at
      contract-drafting time would have caught the transcription error before it could reach a
      gate.  a checker that derives its expected value from the invariant rather than from a
      hand-written literal cannot fail this way again.

  D2 FIELDS
      D2-1  new gate authorised                                   :  __________
      D2-2  corrected expected value 5,535,334 adopted            :  __________
      D2-3  the invariant assertion added as specified above       :  __________
      D2-4  one execution, no retry                               :  __________
```

## 3. D3 — INCONCLUSIVE

```text
  CV-2 = INCONCLUSIVE.  Phase 3-B = NOT CLOSED.

  the stated reason
      the PASS/FAIL criterion as originally defined cannot be applied, because its expected
      value was wrong.

  WHAT D3 DOES NOT DO
      it does not deny the factual relation.  this run obtained 5,535,334 B and
      SHA-256 E5E74200…  D3 does not assert those are wrong; it asserts the criterion, not the
      measurement.

  THE LIMIT, stated by the Owner and agreed
      D3 is the most conservative option AND it does not advance the closure decision.  it
      holds NOT CLOSED.  reaching closure still requires something equivalent to D1 or D2
      later.

  D3 FIELDS
      D3-1  CV-2 recorded as INCONCLUSIVE                        :  __________
      D3-2  Phase 3-B remains NOT CLOSED                          :  __________
```

## 4. Held in common to all three, and not re-opened by any of them

```text
  A1   KEEP G0/G1                     maintained
  B3   NOT YET DETERMINABLE           maintained.  none of D1/D2/D3 says anything about intent.
  C1   ACCEPT MAPPING AS IT STANDS    maintained.  CV-4 re-confirmed all five anchors resolve
                                      on the on-disk generation, recorded as
                                      5,550,764 B / C5873522…FE352 / 121,634 lines.

  G0/G1 authority     maintained
  G2                  non-authoritative, unchanged, untouched
  CR-3                PENDING
  14/14               NOT MET
  Step5EK-R1          unauthorised, Contract Review NOT STARTED
  Phase 2-B           NOT STARTED, NOT AUTHORIZED
  Build 0   CDB 0   Re-pin 0   Regeneration 0
```

## 5. The agent's position, offered as information only

```text
  the agent does not select.  it records that:

    D2 is the option that best matches the discipline this workstream has followed, because it
    preserves the separation between a gate's rule and the agent's judgement about the gate.
    The cost is one invocation, and the invocation would be against a corrected contract with
    the invariant assertion added, so it would be a better gate than the first one.

    D1 is defensible on the merits and the evidence for it is strong.  its cost is that a
    stated PASS condition is amended after the fact.

    D3 is the most conservative description of what happened and the only one that advances
    nothing.

  the agent notes one thing that is NOT a preference.  whichever option is chosen, the
  following remains true and is not in dispute:

      the CRLF form of G1 is byte-exact G0, by digest, as measured in the authorised run.
```

## 6. The record-amendment gate — REQUIRED UNDER ALL THREE OPTIONS, and it is NOT this decision

The Owner stated that the four documents must not be edited automatically under any of
D1/D2/D3, and that the treatment of the wrong figure must be settled first. That is correct,
and it is worth separating explicitly:

```text
  THE AMENDMENT IS ORTHOGONAL TO D1/D2/D3

      the wrong figure exists in four documents.  that is true whichever option is chosen.
      D1 closes the gate, D2 opens another, D3 changes nothing.  in all three cases the four
      documents still carry 5,533,334.

      therefore the amendment gate is required under ALL THREE and is not a consequence of
      any of them.  it should be cut as its own gate, and it is NOT authorised here.

  THE AMENDMENT MUST NOT BE AN IN-PLACE EDIT
      the Owner's own earlier format applies, and it is the right one:

          Historical State
          ----------------
          record creation time:
          <the figure as it stands in that document>

          Subsequent Proven Event
          -----------------------
          <the run, and the three independent determinations of the correct value>

          Correction
          ----------
          The historical statement was true as recorded at record creation time and is
          superseded on this one field.  No historical record body was rewritten.

  SCOPE OF THE AMENDMENT GATE, when cut
      it may ADD an amendment record.  it may NOT edit the four historical documents.
      editing them would destroy the fact that the error existed and was carried through four
      documents before a gate caught it, which is the most useful thing this incident
      produced.

  WHAT THE AMENDMENT GATE MAY NOT DO
      it may not alter any substantive conclusion.  the conclusions in those documents stand,
      as recorded in the Closure Validation RESULT section 4.  only the decimal figure is
      superseded.

  STATUS   NOT AUTHORISED.  defined here so it is not lost, not requested here.
```

## 7. State

```text
Phase 2-A   CLOSED / MATCH
Phase 3-A   CLOSED / PREPARED
Phase 3-B   adjudicated A1 / B3 / C1.  Closure Validation executed once.  CV-2 FAIL.
            NOT CLOSED.
Phase 3-C   INSTRUMENT CREATED.  D1 / D2 / D3 = OPEN, awaiting the Owner.

authority              ADJUDICATED = G0/G1 content.  premise INTACT, re-confirmed by digest.
CR-3                   PENDING
14/14                  NOT MET
Implementation Authorization   FORBIDDEN
Step5EK-R1             CONTRACT DRAFTED 19,268 B.  Contract Review NOT STARTED, NOT AUTHORIZED.
Phase 2-B              NOT STARTED, NOT AUTHORIZED

cdb 0   build 0   source 0   commits 0   checkouts 0   re-pins 0
ConvoPeq.md regeneration 0   existing record modifications 0   payload artifacts 0
closure-validation executions 1   retries 0   new gate executions 0

frozen 7 + Baseline-3 = 8/8 intact    .cdb corpus 16, unmodified
Release AudioEngineHarness.exe  E5C7AFB9...  UNCHANGED

HOLD.  the only outstanding action is the Owner's D1 / D2 / D3 determination.
       the record-amendment gate is separately required and separately unauthorised.
```

```text
held, untouched
    1  T0P sequence conjunct generalisation
    2  the sequence operand not being observable on a non-accepting T0Q hit
    3  the two malformed T1 sites
    4  the 65 versus 116 hit non-comparability
```
