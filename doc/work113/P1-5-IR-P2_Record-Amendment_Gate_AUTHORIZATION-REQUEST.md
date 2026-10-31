# Record-Amendment Gate — AUTHORIZATION REQUEST

```text
gate                = an authorization REQUEST.  NOT an authorization.  NOT an amendment.
mode                = NOT EXECUTED.  no historical record modified.  no amendment record created.
precondition        = Step5EK-R1 Contract Review CLOSED, RR 18/18 PASS, findings DF-1..DF-3
                      which are NOT execution blockers and NOT part of this gate.
purpose             = supersede ONE wrong decimal literal in FOUR historical documents, by a
                      NEW amendment record, with the historical documents left byte-intact.
status              = AWAITING OWNER EXPLICIT AUTHORIZATION
```

## 0. The correction, stated exactly

```text
  SUPERSEDED   5,533,334 B     the CRLF representation of blob 5907a1c3… as stated in four
                               historical documents
  CORRECT      5,535,334 B     established by three independent determinations and confirmed
                               today by the Closure Validation run:
                                 5,414,041 (LF blob) + 121,293 (lines) = 5,535,334
                                 SHA-256 of the CRLF form = E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
                                 a matching SHA-256 determines the length
```

## 1. The literal occurs in EIGHT documents, and only FOUR of them may be amended

This is the most important section of the request, because a blanket replacement of the
literal across all eight files would corrupt the incident record.

```text
  THE DISCRIMINATING QUESTION
      is this occurrence an AFFIRMATIVE ASSERTION of the CRLF size, or a REFERENCE to the
      error itself?

      an affirmative assertion states what the CRLF size IS, and is factually wrong.
      a reference names 5,533,334 as the erroneous value, and is CORRECT AS WRITTEN.
      replacing a reference with 5,535,334 would erase the record of the mistake.
```

### Category 1 — affirmative assertions.  4 documents, 5 occurrences.  IN SCOPE.

```text
  1  P1-5-IR-P2_INCIDENT-2_Legacy-Authority_Recovery-Test_Phase-2A_RESULT_MATCH.md
                                                                 11,537 B
     SHA-256 4609D5FC8D41B28734309D485CE529BA…
     L102      size            5,533,334 B      residual 0
               AFFIRMATIVE.  and internally inconsistent as printed: a size of 5,533,334
               cannot have residual 0 against a pinned size of 5,535,334; it would be -2,000.
               this inconsistency was visible in the Phase 2-A output and went unnoticed until
               the Closure Validation gate compared against it.

  2  P1-5-IR-P2_PHASE-3A_Authority_Adjudication_Preparation.md
                                                                 13,933 B
     SHA-256 BED5006C118ECFA409BFC4DB60326C39…
     L66       5,414,041 + 121,293 = 5,533,334... and the measured CRLF variant is 5,533,334 B with
               AFFIRMATIVE, twice in one line.  the sum is also wrong: the correct sum is
               5,535,334.

  3  P1-5-IR-P2_PHASE-3B_Closure-Validation_AUTHORIZATION-REQUEST.md
                                                                 13,065 B
     SHA-256 0335F94F5B200CB811D887AFFC0E56A9…
     L43       its CRLF form         5,533,334 B   SHA-256 E5E74200…
               AFFIRMATIVE.
     L91       PASS        CRLF form size      == 5,533,334
               AFFIRMATIVE.  this one is the contractual PASS condition itself, and it is the
               direct cause of the CV-2 FAIL.

  4  P1-5-IR-P2_PHASE-3B_Post-Adjudication_A1-B3-C1.md
                                                                  9,927 B
     SHA-256 7621FEDCD1DCE12F2088C6E644EA02CF…
     L176      5,533,334 B and SHA-256 E5E74200…
               AFFIRMATIVE.
```

### Category 2 — references to the error.  4 documents, 21 occurrences.  OUT OF SCOPE, FOREVER.

```text
  P1-5-IR-P2_PHASE-3B_CLOSED.md                                     3 occurrences
  P1-5-IR-P2_PHASE-3B_Closure-Validation_RESULT.md                  8 occurrences
  P1-5-IR-P2_PHASE-3C_Closure_Disposition_DECISION-RECORD.md       6 occurrences
  P1-5-IR-P2_PHASE-3C_Post-Adjudication_D1-ACCEPTED.md             4 occurrences

  EVERY occurrence in these four is a reference.  representative, verbatim:

      "the CV-2 expected value 5,533,334 B was a MIS-RECORDING in the contract"
      "old contract    expected 5,533,334   measured 5,535,334   ->  FAIL"
      "CRLF_SIZE = 5,533,334      the cause of the CV-2 failure"
      "a size of 5,533,334 cannot have a residual of 0 against 5,535,334"
      "n == 5533334  ->  False"

  these are CORRECT AS WRITTEN.  each one names the wrong value in order to identify it as
  wrong.  amending them would make the incident record assert that the contract said
  5,535,334, that n == 5533334 was True, and that the residual was not 0.  every one of those
  would be false, and the falsification would be invisible to anyone who had not read this
  request.

  THIS GATE DOES NOT TOUCH THEM, AND NO FUTURE GATE SHOULD.
```

## 2. Method — superseding, never editing

```text
  REQUIRED STRUCTURE, the Owner's own earlier format

      Historical State
      ----------------
      record: <file>
      line:   <line>
      as recorded: <the verbatim line, exactly as it stands today>
      status: unchanged, retained byte-for-byte

      Subsequent Proven Event
      -----------------------
      the Closure Validation run, executed once under authorization
      three independent determinations of the correct value
      D1-1 adjudication that the contract's expected value was mis-recorded
      D1-2 corrected expected value accepted
      D1-3 re-evaluation on existing evidence, no re-measurement
      Phase 3-B CLOSED

      Correction
      ----------
      the Historical State line above is superseded on this one decimal literal.
      the corrected value is 5,535,334 B.
      NO historical record body was rewritten.  NO file was modified.

  BYTE-PRECISION
      the amendment quotes each of the 5 target lines verbatim, with its file, line number and
      the document's SHA-256 at amendment time, so that a later reader can verify that the
      quoted line is still the line that stands, and that the document has not changed since.

  VERIFICATION THE AMENDMENT ITSELF MUST CARRY
      - the 5 quoted lines still match the 4 documents byte-for-byte at amendment time
      - the 4 documents' SHA-256 are unchanged before and after the amendment is written
      - a post-write re-scan confirms 5 occurrences in Category 1 and 21 in Category 2,
        with every Category 2 occurrence still naming 5,533,334 as the erroneous value
```

## 3. Scope

```text
  ALLOW
      creation of ONE new amendment record under doc/work113/
      read-only inspection of the 4 target documents and the 4 reference documents
      recording of the corrected value, the basis, and the SHA-256 of all 8 documents

  FORBID
      in-place editing of ANY historical record, including the 4 in-scope documents
      any change to the 4 Category-2 documents
      any change to a substantive conclusion, an evidence table, a verdict, or a SHA-256
      ConvoPeq.md regeneration
      authority re-pin
      source modification
      build
      CDB execution
      Step5EK-R1 execution
      unblocking CR-3
      any R1 contract amendment, and specifically DF-1, DF-2 and DF-3
      any change to the G0/G1 authority adjudication, to A1 / B3 / C1, or to Phase 3-B CLOSED
```

## 4. What must be preserved verbatim by the amendment

```text
  PRESERVE, and the amendment states that it preserves these
      the substantive conclusions of all four in-scope documents
      every evidence table
      every SHA-256 recorded in them
      the G0/G1 authority adjudication and its premise-intact status
      A1  KEEP G0/G1
      B3  NOT YET DETERMINABLE
      C1  ACCEPT MAPPING AS IT STANDS
      Phase 3-B = CLOSED
      CR-3 = PENDING
      Step5EK-R1 execution = NOT AUTHORIZED
      14/14 = NOT MET
      Implementation Authorization = FORBIDDEN
      the four held OPEN items
```

## 5. The DF findings are explicitly excluded

```text
  DF-1  the Step5EK-R1 Contract's section 15 authority status.  a documentation finding on
        the R1 contract, not a decimal literal in these four documents.
  DF-2  the G-7 precedence finding on the R1 contract.  a SEPARATE matter.  adopting it would
        be an R1 contract amendment and is NOT bundled here.
  DF-3  the AB-6 observable.  likewise.

  ALL THREE REMAIN with the Owner as R1-contract items.  bundling them into a decimal-literal
  amendment would smuggle a contract change through a gate scoped to a typo, which is the
  pattern this whole incident was spent preventing.

  the amendment therefore must not mention DF-1, DF-2 or DF-3 except to record that they are
  out of its scope.
```

## 6. If the Owner later wants a category-2 amendment, it would need a different gate

```text
  NOT PROPOSED HERE, and recorded so the question is not lost

      the Category-2 documents are correct as written.  there is nothing to amend in them.
      what could be wanted is a NOTE appended to the incident record explaining the two
      categories, so that a future reader scanning for the literal does not "fix" the 21
      references.  that is a different purpose from superseding a wrong assertion, and it
      would need its own authorization.

      it is not requested now.
```

## 7. State

```text
Step5EI      DRAFTED, unchanged
Step5EJ      CLOSED  13 PASS / 0 FAIL / 1 PENDING
Step5EK      CLOSED / INCONCLUSIVE
Phase 3-A    CLOSED / PREPARED
Phase 3-B    CLOSED
Phase 3-C    D1 SELECTED, executed
Step5EK-R1   CONTRACT REVIEW CLOSED.  RR 18/18 PASS.  DF-1..DF-3 open, R1 contract items,
             NOT execution blockers.
             R1 EXECUTION NOT AUTHORIZED.

record-amendment gate   AUTHORIZATION REQUESTED, NOT EXECUTED

authority   ADJUDICATED = G0/G1 content, premise INTACT
CR-3        PENDING
14/14       NOT MET
Step5EK-R1 execution   NOT AUTHORIZED
Phase 2-B              NOT STARTED, NOT AUTHORIZED

cdb 0   build 0   source 0   commits 0   checkouts 0   re-pins 0
ConvoPeq.md regeneration 0   historical record modifications 0   payload artifacts 0
git operations 0

Category 1 occurrences  5 in 4 documents   IN SCOPE, not yet amended
Category 2 occurrences 21 in 4 documents   OUT OF SCOPE, must remain untouched

frozen 7 + Baseline-3 = 8/8 intact    .cdb corpus 16, unmodified
Release AudioEngineHarness.exe  E5C7AFB9...  UNCHANGED

HOLD.  the only outstanding action is the Owner's explicit authorization of this gate.
```

```text
held, untouched
    1  T0P sequence conjunct generalisation
    2  the sequence operand not being observable on a non-accepting T0Q hit
    3  the two malformed T1 sites
    4  the 65 versus 116 hit non-comparability
```
