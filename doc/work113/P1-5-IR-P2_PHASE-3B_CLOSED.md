# Phase 3-B — CLOSURE

```text
gate                = CLOSURE RECORD.  Phase 3-B is closed by explicit Owner re-decision.
mode                = READ-ONLY.  no re-execution, no re-pin, no regeneration, no checkout,
                      no build, no CDB, no record amendment, no source modification.
new measurements    = 0.  the closure rests entirely on the single authorised run already
                      executed, re-evaluated against the corrected expected value.

CLOSURE-1  Phase 3-B closure 条件の再評価                    PASS
CLOSURE-2  CV-1                                               PASS
CLOSURE-3  CV-2   訂正後期待値 5,535,334 B に対する再評価    PASS
CLOSURE-4  CV-3                                               PASS
CLOSURE-5  CV-4                                               PASS
CLOSURE-6  CV-5   G2-UNCHANGED                               PASS
CLOSURE-7  CV-6                                               PASS
CLOSURE-8  CV-7                                               PASS
CLOSURE-9  Phase 3-B                                          CLOSED
```

## 1. The basis, in the Owner's terms

```text
  D1-1  ACCEPT
        the CV-2 expected value 5,533,334 B was a MIS-RECORDING in the contract.
        that gate input was defective and therefore did not generate a valid verdict.

  D1-2  ACCEPT
        corrected expected value = 5,535,334 B

  D1-3  ACCEPT
        re-evaluate the existing evidence from the single authorised run against the
        corrected expected value.  no new measurement, no re-execution.
```

## 2. The logical chain, as stated

```text
  D1-1 ACCEPT
    ->  CV-2's original gate input was invalid
  D1-2 ACCEPT
    ->  correct expected value = 5,535,334 B
  D1-3 ACCEPT
    ->  re-evaluate the existing authorised-run evidence against it
  CV-1 PASS   CV-2 PASS   CV-3 PASS   CV-4 PASS
  CV-5 G2-UNCHANGED         CV-6 PASS   CV-7 PASS
    ->  every closure condition satisfied
    ->  PHASE 3-B = CLOSED
```

## 3. The wording that must be preserved, and why

This is the Owner's own distinction and it is recorded verbatim because it is the
difference between an honest closure and a convenient one.

```text
  THIS IS NOT
      "CV-2 の FAIL を override した"

  THIS IS
      "無効な gate input を D1 により訂正した上で、
       既存 evidence に対して closure criteria を再評価した"

  THE DISTINCTION IS SUBSTANTIVE
      a gate that produced a wrong verdict and had it overridden  is a different and much
      worse record than a gate whose input was invalid, was corrected by adjudication, and was
      then re-read against evidence that never changed.

  THE SEQUENCE THAT PRODUCED IT, stated once and without softening
      old contract    expected 5,533,334   measured 5,535,334   ->  FAIL
      D1-1            the expected value in the contract was wrong
                      the old verdict is not a valid gate verdict
      D1-2            corrected expected 5,535,334
      existing        measured 5,535,334                          ->  MATCH
      re-evaluation   CV-2 = PASS

      the measured value was NEVER altered.  only the expectation was, and only by
      adjudication.  that is the whole of the change.
```

## 4. The evidence, as re-evaluated — no re-measurement

```text
  CV-1  PASS   unchanged expectation
               oid 5907a1c3d053b651f6edc860e022409c89afe0f2  MATCH
               type blob  MATCH   size 5,414,041  MATCH

  CV-2  PASS   RE-EVALUATED against corrected expected 5,535,334 B
               len(extracted)          5,414,041                      MATCH
               sha1(blob header+data)  5907a1c3…                     MATCH
               CR bytes in LF blob     0                              OK
               CRLF SHA-256            E5E74200…EF3609                MATCH
               CRLF size               5,535,334   vs 5,535,334       MATCH
               all five sub-conditions satisfied

  CV-3  PASS   unchanged expectation
               HEAD e565019d…  MATCH   ahead 0   behind 0

  CV-4  PASS   unchanged expectation
               L61993 OK  L62000 OK  L62015 OK  L62078 OK  L62127 OK
               read from ConvoPeq.md, strict name, 5,550,764 B /
               C5873522…FE352 / 121,634 lines.  no numbered variant present, none substituted.

  CV-5  PASS   unchanged expectation
               G2-UNCHANGED.  size, SHA-256, line count, mtime and the line-3 generation
               timestamp, all five equal to the Phase 3-A baseline.

  CV-6  PASS   unchanged expectation
               E5C7AFB9C4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75  MATCH

  CV-7  PASS   unchanged expectation
               .cdb 16   frozen 7 + Baseline-3 8/8, each exactly once
               cdb processes 0   AudioEngineHarness processes 0
```

## 5. The precise form of this closure — recorded so it cannot later be overstated

```text
  WHAT THIS CLOSURE IS
      a re-evaluation, on evidence that was never re-measured, after an adjudicated
      correction of an invalid gate input.

  WHAT THIS CLOSURE IS NOT, stated plainly
      it is NOT a run in which CV-2 was compared against the corrected constant.  no such
      run exists.  the corrected comparison was performed arithmetically, by reading the
      recorded 5,535,334 against the corrected 5,535,334.

      that is sufficient here, and the Owner has adjudicated that it is sufficient, because
      the SHA-256 that fixes the byte identity was measured inside the authorised run itself.
      but the distinction between "checked in a run" and "re-read from a run" is recorded so
      that no later document describes this as a passing execution of the corrected gate.

  WHAT WOULD HAVE BEEN NEEDED INSTEAD
      a second execution under a corrected contract, which is the route D2 offered and which
      the Owner rejected.  the choice between the two was made knowingly.
```

## 6. What closure does not change

```text
  authority            ADJUDICATED = G0/G1 content.  premise INTACT.
                        byte-exact recovery proven.  re-pin 0.  pair of record unchanged.

  A1  KEEP G0/G1                       unchanged
  B3  NOT YET DETERMINABLE             unchanged.  closure says nothing about intent.
  C1  ACCEPT MAPPING AS IT STANDS      unchanged.  CV-4 re-confirmed all five anchors.

  G2   non-authoritative, untouched
       5,550,764 B / SHA-256 C587352201F0BE3D8F39D30396137A663E35A7412EAA749FBE591D661E8A7DFE
       121,634 lines

  CR-3                                PENDING
  14/14                               NOT MET
  Implementation Authorization        FORBIDDEN
  Step5EK-R1                          NOT AUTHORIZED, Contract Review NOT STARTED
  Phase 2-B                           NOT STARTED, NOT AUTHORIZED

  re-pin 0   regeneration 0   build 0   CDB 0   source modification 0
  commit 0   checkout 0      new measurements 0
```

## 7. What remains open, and is NOT closed by this

```text
  THE RECORD-AMENDMENT GATE IS STILL REQUIRED AND STILL UNAUTHORISED.

      the corrected figure 5,535,334 has not been carried into the four documents that still
      print 5,533,334.  that is true now that Phase 3-B is closed, exactly as it was true
      before.  the closure did not touch them and was not supposed to.

      documents still carrying the superseded figure
        INCIDENT-2 ..._Phase-2A_RESULT_MATCH.md
        PHASE-3A ..._Authority_Adjudication_Preparation.md
        PHASE-3B ..._Post-Adjudication_A1-B3-C1.md
        PHASE-3B ..._Closure-Validation_AUTHORIZATION-REQUEST.md

      their substantive conclusions all stand.  only the decimal figure is superseded, and
      only by an amendment record.  none of them may be edited in place.

  ALSO STILL OPEN, unchanged by this closure
      CR-3, the whole Step5EK line, and the four held items
        1  T0P sequence conjunct generalisation
        2  the sequence operand not being observable on a non-accepting T0Q hit
        3  the two malformed T1 sites
        4  the 65 versus 116 hit non-comparability
```

## 8. Records superseded by this one, on the closure field only

```text
  NO EXISTING RECORD IS EDITED.  each of the following stated NOT CLOSED, which was true when
  written and is superseded here on that single field.

    P1-5-IR-P2_PHASE-3B_Closure-Validation_RESULT.md
        "PHASE 3-B   NOT CLOSED"

    P1-5-IR-P2_PHASE-3C_Post-Adjudication_D1-ACCEPTED.md
        "PHASE 3-B NOT CLOSED — closure re-decision withheld"

  every other statement in both records stands, including the CV-2 FAIL as executed, the
  root cause, the propagation list, and the evidence table.
```

## 9. State

```text
Phase 2-A   CLOSED / MATCH
Phase 3-A   CLOSED / PREPARED
Phase 3-B   *** CLOSED ***   by Owner closure re-decision, basis D1-1 / D1-2 / D1-3
Phase 3-C   D1 SELECTED and executed as far as the closure path goes

authority              ADJUDICATED = G0/G1 content, committed e565019d, pushed,
                       byte-exact recovery proven, premise INTACT
CR-3                   PENDING
14/14                  NOT MET
Implementation Authorization   FORBIDDEN
Step5EK-R1             CONTRACT DRAFTED 19,268 B.  NOT AUTHORIZED, Contract Review NOT STARTED.
Phase 2-B              NOT STARTED, NOT AUTHORIZED
record-amendment gate  REQUIRED, NOT AUTHORIZED

closure-validation executions 1   retries 0   new measurements 0
cdb 0   build 0   source 0   commits 0   checkouts 0   re-pins 0
ConvoPeq.md regeneration 0   existing record modifications 0   payload artifacts 0

frozen 7 + Baseline-3 = 8/8 intact    .cdb corpus 16, unmodified
Release AudioEngineHarness.exe  E5C7AFB9...  UNCHANGED

HOLD.  Phase 3-B is closed.  Nothing else is authorised by that closure.
```

```text
held, untouched
    1  T0P sequence conjunct generalisation
    2  the sequence operand not being observable on a non-accepting T0Q hit
    3  the two malformed T1 sites
    4  the 65 versus 116 hit non-comparability
```
