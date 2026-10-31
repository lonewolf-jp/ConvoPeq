# Amendment 1 — supersession of one decimal literal in four historical documents

```text
gate                = record-amendment gate, executed once under explicit Owner authorization.
mode                = superseding amendment only.  no historical record was modified.
scope               = ONE decimal literal, in FOUR documents, at FIVE occurrences.
precondition        = RA-1 PASS, RA-2 PASS, RA-3 PASS, in the single authorized execution
precondition        = RA-3's detector was rebuilt as a context-aware property and validated
                      against the five Owner-adjudicated Category-1 occurrences as a control,
                      all five of which it classified AFFIRMATIVE.  a phrase list with a
                      hole in it therefore cannot produce a pass.

NO HISTORICAL FILE WAS WRITTEN, MOVED, OR ALTERED BY THIS GATE.
```

## 1. Historical State — the five occurrences, as they stand

Each is an **affirmative assertion** of the CRLF representation's size. Each is quoted
verbatim, with its file, its line number, and the document's SHA-256 at amendment time, so
that a later reader can confirm the quoted line is still the line that stands and that the
document has not changed since.

```text
  A-1  P1-5-IR-P2_INCIDENT-2_Legacy-Authority_Recovery-Test_Phase-2A_RESULT_MATCH.md
       11,537 B   SHA-256 4609D5FC8D41B28734309D485CE529BAA5C2117F328FB2C2CC0420E8273EE9B5
       L102
            size            5,533,334 B      residual 0

  A-2  P1-5-IR-P2_PHASE-3A_Authority_Adjudication_Preparation.md
       13,933 B   SHA-256 BED5006C118ECFA409BFC4DB60326C39CA9B3CFC1F895CB9D49F008D71148B11
       L66
            5,414,041 + 121,293 = 5,533,334... and the measured CRLF variant is 5,533,334 B with

  A-3  P1-5-IR-P2_PHASE-3B_Closure-Validation_AUTHORIZATION-REQUEST.md
       13,065 B   SHA-256 0335F94F5B200CB811D887AFFC0E56A964273498C61ECDE62511743CF263EB31
       L43
            its CRLF form         5,533,334 B   SHA-256 E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609

  A-4  P1-5-IR-P2_PHASE-3B_Closure-Validation_AUTHORIZATION-REQUEST.md
       same document
       L91
            PASS        CRLF form size      == 5,533,334

  A-5  P1-5-IR-P2_PHASE-3B_Post-Adjudication_A1-B3-C1.md
        9,927 B   SHA-256 7621FEDCD1DCE12F2088C6E644EA02CF5349B504D2E1C8823DC8F28EB314F844
       L176
            5,533,334 B and SHA-256 E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
```

A-4 is the direct cause of the CV-2 FAIL: it was the contractual PASS condition, written
against a value that was never the measured one.

## 2. Subsequent Proven Event

```text
  the Closure Validation run, executed once under authorization, read-only

    LF blob        5,414,041 B
    line count       121,293
    CRLF form      5,535,334 B
    CRLF SHA-256   E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609

  the contract's CRLF_SIZE constant was written as 5,533,334.  the correct constant is
  5,535,334.  one digit, at the fourth position.

  three independent determinations
    the authorized run itself printed the CRLF form length as 5,535,334
    format(n, ',') == "5,535,334";  n == 5533334 -> False ;  n == 5535334 -> True
    the CRLF form's SHA-256 is the pinned pair of record, and a byte sequence with a given
    SHA-256 has a DETERMINED LENGTH, so digest agreement forces length agreement

  the arithmetic identity
    5,414,041  (LF blob)  +  121,293 (lines)  =  5,535,334

  the adjudication
    D1-1 ACCEPT  the expected value in the contract was mis-recorded, so that gate input was
                 defective and did not generate a valid verdict
    D1-2 ACCEPT  corrected expected value = 5,535,334 B
    D1-3 ACCEPT  re-evaluation of the existing evidence, no new measurement, no re-execution

  the closure
    Phase 3-B = CLOSED, by explicit Owner closure re-decision

  THE TELL WAS ALREADY VISIBLE AND WAS NOTICED LATE
    A-1 as printed states a size of 5,533,334 with "residual 0" against a pinned size of
    5,535,334.  those cannot both hold; the residual would be -2,000.  that internal
    inconsistency was present in the Phase 2-A output and went unnoticed until the Closure
    Validation gate compared against it.  it is quoted here unaltered, because the fact that
    it was visible and missed is part of the record.
```

## 3. Correction

```text
  each of A-1 through A-5 is SUPERSEDED on one decimal literal.

      superseded   5,533,334 B
      correct      5,535,334 B

  A-1  the CRLF representation's size is 5,535,334 B.  the "residual 0" annotation on that
       line was, in fact, correct at the time it was written — the residual really was zero
       against the pinned size — and remains correct.  only the stated size was wrong.
  A-2  the sum is 5,535,334, and the measured CRLF variant is 5,535,334 B.
  A-3  the CRLF form is 5,535,334 B.
  A-4  the contractual PASS condition is  CRLF form size == 5,535,334.
  A-5  5,535,334 B, with the SHA-256 as stated.

  NO HISTORICAL RECORD BODY WAS REWRITTEN.  NO FILE WAS MODIFIED.  This record supersedes;
  it does not edit.
```

## 4. Explicitly NOT amended — twenty-one references, four documents

```text
  THE DISCRIMINATING QUESTION
      is an occurrence an AFFIRMATIVE ASSERTION of the size, or a REFERENCE to the error?
      an affirmative states what the size IS and is factually wrong.
      a reference names 5,533,334 in order to identify it as wrong, and is CORRECT AS WRITTEN.

  the following 21 occurrences are REFERENCES and are left exactly as they stand.
  replacing any of them with 5,535,334 would falsify the incident record.

    P1-5-IR-P2_PHASE-3B_CLOSED.md                                9,993 B
      SHA-256 272EB036CEEB0E2FB42E913E17EDE484CDF5DD21AED33F271BF1CD81C95006E0    3 occurrences
    P1-5-IR-P2_PHASE-3B_Closure-Validation_RESULT.md            11,329 B
      SHA-256 2B9BFB595C15BC7C3A4690A4ECFD3BD110B3C3AA9268C1B3B4EB8EF0576F8C3B    8 occurrences
    P1-5-IR-P2_PHASE-3C_Closure_Disposition_DECISION-RECORD.md  10,767 B
      SHA-256 1A3BE4CAD29843A6ECC11615268AD61024AD16534F10D363060D310B8107015D    6 occurrences
    P1-5-IR-P2_PHASE-3C_Post-Adjudication_D1-ACCEPTED.md        9,149 B
      SHA-256 248743B846C8DDAFC0087817DA893E8D9C5742788FFD59BEC8AF5496C1D2D8F6    4 occurrences

  representative, verbatim, and correct as written
      "the CV-2 expected value 5,533,334 B was a MIS-RECORDING in the contract"
      "old contract    expected 5,533,334   measured 5,535,334   ->  FAIL"
      "CRLF_SIZE = 5,533,334                the cause of the CV-2 failure"
      "of 5,535,334.  a size of 5,533,334 cannot have a residual of 0 against 5,535,334"
      "n == 5533334  ->  False"

  ALL 21 were verified as references in context, in the single authorized execution, by a
  detector that was itself first required to classify all five adjudicated Category-1
  occurrences as AFFIRMATIVE.  that control is what makes the classification evidential
  rather than asserted.
```

## 5. Preserved, unchanged, by this amendment

```text
  the substantive conclusions of all four amended documents
  every evidence table in them
  every SHA-256 recorded in them
  the Phase 2-A verdict MATCH, and P/Q/R all PASS
  the G0 <-> G1 identity conclusion, and the byte-exact recovery finding
  the A1 premise-intact status

  authority   ADJUDICATED = G0/G1 content, committed e565019d, pushed
  A1          KEEP G0/G1
  B3          NOT YET DETERMINABLE
  C1          ACCEPT MAPPING AS IT STANDS
  Phase 3-B   CLOSED
  CR-3        PENDING
  14/14       NOT MET
  Implementation Authorization   FORBIDDEN
  Step5EK-R1 Contract Review    CLOSED, RR 18/18 PASS
  Step5EK-R1 execution          NOT AUTHORIZED
  Phase 2-B                     NOT STARTED, NOT AUTHORIZED

  the four held OPEN items
      1  T0P sequence conjunct generalisation
      2  the sequence operand not being observable on a non-accepting T0Q hit
      3  the two malformed T1 sites
      4  the 65 versus 116 hit non-comparability
```

## 6. Out of scope of this amendment

```text
  DF-1  the Step5EK-R1 Contract's section 15 authority status.  a documentation finding on
        the R1 contract, not a decimal literal in these four documents.
  DF-2  the R1 contract's G-7 precedence over G-5/G-6.  adopting it would be an R1 contract
        amendment, and is NOT bundled into a typo-scoped gate.
  DF-3  the R1 contract's AB-6 observable.  likewise.

  all three remain with the Owner as R1-contract items.  they are named here only to record
  that this amendment does not touch them, and this amendment proposes no corrected text for
  any of them.
```

## 7. State after this amendment

```text
  superseded   5,533,334 B  ->  5,535,334 B   in 4 documents at 5 occurrences
  untouched    21 references in 4 documents
  historical files modified   0
  amendment records created   1   (this one)

  authority  ADJUDICATED = G0/G1 content, premise INTACT
  CR-3  PENDING   14/14  NOT MET   Implementation  FORBIDDEN
  Step5EK-R1 execution  NOT AUTHORIZED
  Phase 2-B  NOT STARTED

  cdb 0   build 0   source 0   re-pin 0   ConvoPeq.md regeneration 0
  .cdb edits 0   git operations 0
  frozen 7 + Baseline-3 = 8/8 intact    .cdb corpus 16, unmodified
  Release AudioEngineHarness.exe  E5C7AFB9...  UNCHANGED
```

```text
held, untouched
    1  T0P sequence conjunct generalisation
    2  the sequence operand not being observable on a non-accepting T0Q hit
    3  the two malformed T1 sites
    4  the 65 versus 116 hit non-comparability
```
