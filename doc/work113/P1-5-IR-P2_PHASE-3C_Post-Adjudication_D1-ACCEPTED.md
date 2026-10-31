# Phase 3-C — Post-Adjudication Record / D1 selected

```text
gate                = POST-ADJUDICATION RECORD.  records the Owner's D1 / D2 / D3 determination.
mode                = READ-ONLY.  no re-execution, no re-pin, no regeneration, no checkout,
                      no build, no CDB, no record amendment.
new measurements    = 0.  the re-evaluation in section 4 is arithmetic on values already
                      recorded by the single authorised run.  nothing was re-measured.

determination        D1-1  ACCEPT
                     D1-2  ACCEPT
                     D1-3  ACCEPT
                     D2-1  REJECT    D2-2  REJECT    D2-3  REJECT    D2-4  REJECT
                     D3-1  REJECT    D3-2  REJECT

Phase 3-B           NOT CLOSED.  closure re-decision is a SEPARATE, EXPLICIT step and is
                    NOT performed by this record.
```

## 1. D1-1 — the adjudication, in the Owner's terms

```text
  THE OWNER ADJUDICATES

      the gate's DECISION INPUT was defective.  the expected value written into the
      authorization contract was mis-recorded.  the measurement was never in question.

  THIS IS EXPLICITLY NOT
      a ruling that the CV-2 FAIL is to be ignored
      a post-hoc declaration that CV-2 "substantively passed"

  THE DISTINCTION IS THE POINT, and the Owner drew it before this record existed
      "CV-2 の期待値そのものが誤記されていたため、ゲート判定入力が不正だった"

  a gate whose input is invalid has not produced a valid verdict.  that is a different
  statement from a gate that produced a wrong verdict, and D1-1 selects the first.
```

## 2. D1-2 — the corrected value, and the three determinations that fix it

```text
  CORRECTED EXPECTED CRLF SIZE      5,535,334 B
  value as written in the contract  5,533,334 B
  the difference                    one digit, fourth position, a 3 where a 5 belongs

  DETERMINATION 1   the authorised run itself printed  CRLF form size 5,535,334
  DETERMINATION 2   format(n, ',') == "5,535,334"
                    n == 5533334  ->  False
                    n == 5535334  ->  True
  DETERMINATION 3   the CRLF form's SHA-256 is E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609,
                    which is the pinned pair of record.  a byte sequence with a given SHA-256
                    has a DETERMINED LENGTH, so digest agreement forces length agreement.

  IDENTITY, exact
      5,414,041  (LF blob)  +  121,293 (lines)  =  5,535,334  =  the pinned size

  CONSEQUENCE FOR A1's PREMISE
      unchanged and re-confirmed.  the CRLF form of G1 is byte-exact G0.  this was true
      before the gate ran, was true during it, and is true now.  A1 stands.
```

## 3. The Owner's grounds, recorded

```text
  1  the execution itself measured the CRLF size as 5,535,334
       5,414,041 + 121,293 = 5,535,334
       therefore 5,533,334 is not a measurement result but a wrong expected value in the
       contract

  2  the CRLF SHA-256 agrees with the existing pair of record
       this is the already-fixed G0/G1 byte identity
       so the correction is a RECONCILIATION of the run's own measurement with existing
       identity evidence, not a conjecture requiring new evidence

  3  re-execution is not required
       the inconsistency is a transcription error in a constant, not a change in the target
       state.  CV-5 was G2-UNCHANGED.  CV-1, CV-3, CV-4, CV-6, CV-7 all PASS.
       there is no event that would justify re-acquiring the target state.
```

## 4. Re-evaluation of the existing CV-1..CV-7 evidence — NO RE-MEASUREMENT

Every value below is transcribed from the single authorised run. Nothing was re-measured.

```text
  CV-1  recorded  oid 5907a1c3d053b651f6edc860e022409c89afe0f2  MATCH
                 type blob  MATCH        size 5,414,041  MATCH
        expectation unchanged                        -> PASS   (unchanged)

  CV-2  recorded  len(extracted)          5,414,041              MATCH
                 sha1(blob header+data)  5907a1c3…             MATCH
                 CR bytes in LF blob     0                      OK
                 CRLF SHA-256            E5E74200…EF3609        MATCH
                 CRLF size               5,535,334   vs contract 5,533,334   MISMATCH

        expectation CORRECTED to 5,535,334
                 CRLF size               5,535,334   vs corrected 5,535,334  MATCH

        all five sub-conditions now satisfied                         -> PASS   (re-evaluated)

  CV-3  recorded  HEAD e565019d…  MATCH   ahead 0   behind 0
        expectation unchanged                        -> PASS   (unchanged)

  CV-4  recorded  L61993 OK  L62000 OK  L62015 OK  L62078 OK  L62127 OK
                 on ConvoPeq.md, strict name, 5,550,764 B /
                 C5873522…FE352 / 121,634 lines, no numbered variant present or substituted
        expectation unchanged                        -> PASS   (unchanged)

  CV-5  recorded  G2-UNCHANGED
                 size / SHA-256 / line count / mtime / line-3 generation timestamp, all five equal
        expectation unchanged                        -> G2-UNCHANGED   (unchanged)

  CV-6  recorded  E5C7AFB9C4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75  MATCH
        expectation unchanged                        -> PASS   (unchanged)

  CV-7  recorded  .cdb 16   frozen 7 + Baseline-3 8/8   cdb 0   harness 0
        expectation unchanged                        -> PASS   (unchanged)
```

```text
  RESULTING EVIDENCE STATE, prior to any closure decision

      CV-1 PASS    CV-2 PASS (re-evaluated)    CV-3 PASS    CV-4 PASS
      CV-5 G2-UNCHANGED                        CV-6 PASS    CV-7 PASS

  every check satisfies the contract's condition, with CV-2 read against the corrected
  expectation that D1-2 accepted.

  THIS IS AN EVIDENCE STATEMENT.  IT IS NOT A CLOSURE DECISION.
```

## 5. What this record does NOT do

```text
  it does NOT declare Phase 3-B CLOSED.

      the Owner's instruction was explicit on this point, and it is honoured verbatim:
      "D1-3 の「closure 再決」は D1-1/D1-2 の記録後に別途明示的に行うべき。
       つまり、いきなり Phase 3-B = CLOSED としてはいけません。"

      D1-3 being ACCEPTED selects the PATH.  the closure re-decision is a distinct explicit act
      and is withheld.

  it does not re-run anything.  new measurements 0.
  it does not re-pin.  the pair of record is unchanged and remains the authority description.
  it does not amend the four historical documents.  the record-amendment gate is still
      separately unauthorised.
  it does not re-open A, B or C.  A1, B3, C1 all stand as adjudicated.
  it does not substitute any file for ConvoPeq.md, consistent with the Owner's own note.
  it does not start Step5EK-R1, and does not change CR-3.
  it does not build, and does not run CDB.
```

## 6. Held, unchanged

```text
  A1  KEEP G0/G1                      premise INTACT, re-confirmed by digest
  B3  NOT YET DETERMINABLE            unchanged.  D1 says nothing about intent.
  C1  ACCEPT MAPPING AS IT STANDS     unchanged.  CV-4 re-confirmed all five anchors.
  G0/G1 authority                     maintained
  G2                                  non-authoritative, unchanged, untouched
                                      5,550,764 B / C5873522…FE352 / 121,634 lines
  CR-3                                PENDING
  14/14                               NOT MET
  Implementation Authorization        FORBIDDEN
  Step5EK-R1                          unauthorised, Contract Review NOT STARTED
  Phase 2-B                           NOT STARTED, NOT AUTHORIZED
  record-amendment gate               NOT AUTHORISED, separately required
  Build 0   CDB 0   Re-pin 0   Regeneration 0   existing record modifications 0
```

## 7. State

```text
Phase 2-A   CLOSED / MATCH
Phase 3-A   CLOSED / PREPARED
Phase 3-B   adjudicated A1 / B3 / C1.  Closure Validation executed once.
            Evidence state now satisfies every check, CV-2 read against the corrected
            expectation.  PHASE 3-B NOT CLOSED — closure re-decision withheld.
Phase 3-C   D1 SELECTED.  D1-1 ACCEPT, D1-2 ACCEPT, D1-3 ACCEPT.
            D2 REJECTED in full.  D3 REJECTED in full.

authority              ADJUDICATED = G0/G1 content.  premise INTACT.
CR-3                   PENDING
14/14                  NOT MET
Step5EK-R1             CONTRACT DRAFTED 19,268 B.  NOT AUTHORIZED.
Phase 2-B              NOT STARTED, NOT AUTHORIZED

closure-validation executions 1   retries 0   new measurements this step 0
cdb 0   build 0   source 0   commits 0   checkouts 0   re-pins 0
ConvoPeq.md regeneration 0   existing record modifications 0   payload artifacts 0

frozen 7 + Baseline-3 = 8/8 intact    .cdb corpus 16, unmodified
Release AudioEngineHarness.exe  E5C7AFB9...  UNCHANGED

HOLD.  the next action is the Owner's explicit closure re-decision.
```

```text
held, untouched
    1  T0P sequence conjunct generalisation
    2  the sequence operand not being observable on a non-accepting T0Q hit
    3  the two malformed T1 sites
    4  the 65 versus 116 hit non-comparability
```
