# Phase 3-B Closure Validation — AUTHORIZATION REQUEST / EXECUTION CONTRACT

```text
gate                = an authorization REQUEST and an execution contract.  NOT an authorization.
                      NOT a result.
mode                = NOT EXECUTED.  CV-1..CV-7 have not been run.
precondition        = A1 / B3 / C1 adjudicated and recorded in
                      P1-5-IR-P2_PHASE-3B_Post-Adjudication_A1-B3-C1.md
purpose             = confirm that the premises A1 / B3 / C1 rest on still hold at closure
                      time, and decide whether Phase 3-B may be CLOSED
status              = AWAITING OWNER EXPLICIT EXECUTION AUTHORIZATION
```

## 0. What this gate is, and what it is not

```text
  IS      a read-only confirmation that the adjudication's factual premises have not moved
          between adjudication and closure.

  IS NOT  a re-adjudication.  A / B / C are settled.  nothing here reopens them.
  IS NOT  a re-validation of G2.  A2 was not chosen; no G2 validation gate is required or run.
  IS NOT  a re-derivation of the five citations.  C1 accepted them as they stand.  CV-4 only
          confirms they still RESOLVE, which is a weaker and different check.
  IS NOT  an authorization for Step5EK-R1.  even a full PASS authorises only Phase 3-B CLOSED.
```

## 1. The baseline this gate compares against

Fixed here so the comparison is anchored to a record, not to recollection.

```text
  PHASE 3-A G2 OBSERVATIONS, as recorded

    size                 5,550,764 B
    SHA-256              C587352201F0BE3D8F39D30396137A663E35A7412EAA749FBE591D661E8A7DFE
    line count           121,634
    mtime                2026-09-27 00:37:26
    embedded generation  G2 line 3 reads  "> Generated: 2026-09-27 00:36:31"

  IDENTITY BASELINE, from Phase 2-A and INCIDENT-1

    e565019d:ConvoPeq.md  blob 5907a1c3d053b651f6edc860e022409c89afe0f2   5,414,041 B
    its CRLF form         5,533,334 B   SHA-256 E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
    HEAD                  e565019df6d263370b3cac2ed8e7a14d7c7f8db4
    Release harness       E5C7AFB9C4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75

  CITATION BASELINE, from Phase 3-A

    L61993  bool enqueue(void* ptr, void (*deleter)(void*), uint64_t epoch, DeletionEntryType type,
    L62000  int32_t diff = static_cast<int32_t>(static_cast<uint32_t>(seq - pos));
    L62015  convo::publishAtomic(seq_atom, pos + 1, std::memory_order_release);
    L62078  convo::publishAtomic(seq_atom, scanPos + kQueueSize, std::memory_order_release);
    L62127  convo::publishAtomic(seq_atom, pos + kQueueSize, std::memory_order_release);
```

## 2. CV-1 — the G0/G1 identity

```text
  PROCEDURE   read-only
              git rev-parse e565019d:ConvoPeq.md
              git cat-file -t   <oid>
              git cat-file -s   <oid>

  PASS        oid  == 5907a1c3d053b651f6edc860e022409c89afe0f2
              type == blob
              size == 5,414,041

  NOT PASS    PREMISE MOVEMENT.  the object A1 was decided against is no longer the object
              named by that commit.  report and do not close.
```

## 3. CV-2 — G1's CRLF representation against the pair of record

This is the check that tests the adjudication's central factual claim, so its extraction is
proved rather than assumed, using the method already validated in Phase 2-A.

```text
  PROCEDURE   read-only, in memory, no shell redirection, no payload file
              python  subprocess.run(['git','cat-file','blob',OID], capture_output=True).stdout

  EXTRACTION PROOF, both required, and a failure of either is NOT PASS
              size  len(extracted) == 5,414,041, agreeing with git cat-file -s
              sha1  sha1(b"blob 5414041\0" + extracted) == 5907a1c3d053b651f6edc860e022409c89afe0f2
                    this is git's own object hash, so reproducing the OID is cryptographic
                    proof that the bytes are the blob's exact content

  CRLF FORM   data.replace(b"\n", b"\r\n")
              CR bytes inside the LF blob must be 0; if not, the transform is not a pure
              EOL conversion and the result is NOT PASS

  PASS        CRLF form size      == 5,533,334
              CRLF form SHA-256   == E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
              BOTH must agree.  a size agreement alone is not a pass.

  NOT PASS    PREMISE MOVEMENT.  A1 asserts the authority of record and the committed content
              are the same bytes.  if this fails, that assertion is no longer true and A1
              must be revisited.  report and do not close.
```

## 4. CV-3 — HEAD and push state

```text
  PROCEDURE   read-only
              git rev-parse HEAD
              git rev-list --count origin/main..main
              git rev-list --count main..origin/main

  PASS        HEAD    == e565019df6d263370b3cac2ed8e7a14d7c7f8db4
              ahead   == 0
              behind  == 0

  NOT PASS    report and do not close.  note which of the three moved.  a non-zero ahead with
              an unchanged HEAD would mean a commit exists that CV-1 and CV-2 do not cover.
```

## 5. CV-4 — the five citation anchors still resolve

```text
  SCOPE LIMIT, so this check cannot be over-read
      C1 ACCEPTED the mapping.  this does not re-derive anything.
      it confirms only that each cited line still carries the cited content, in whatever
      generation is on disk at closure time.  the generation actually read IS RECORDED, so
      the confirmation is never ambiguous about what it confirmed.

  PROCEDURE   read-only, no redirection
              read the on-disk ConvoPeq.md, extract lines 61993, 62000, 62015, 62078, 62127,
              match each against the expected content of section 1

  PASS        all five resolve to the expected content
              AND the on-disk generation's size, SHA-256 and line count are recorded, so the
              reader knows which generation the confirmation applies to

  NOT PASS    report and do not close.  record which anchor failed and what was found instead.
```

## 6. CV-5 — G2 drift since Phase 3-A, THREE-VALUED, and the important one

```text
  SCOPE LIMIT
      this check does not judge G2's validity, and B3 remains NOT YET DETERMINABLE whatever
      this returns.  a DRIFT-DETECTED result says nothing about intent and nothing about
      correctness.

  PROCEDURE   read-only
              size, SHA-256, line count, mtime of the on-disk ConvoPeq.md
              line 3, the embedded generation timestamp
              all four compared against section 1

  THREE VALUES, not two

    G2-UNCHANGED         size, SHA-256, line count, mtime and generation timestamp all equal
                         to the Phase 3-A baseline
                         -> PASS

    G2-DRIFT-DETECTED    any of the five differs
                         -> NOT a FAIL.  reported as DRIFT-DETECTED.
                         -> AND THE GATE DOES NOT CLOSE.  the reason is specific:

                             C1 accepted the anchor mapping AS IT STANDS.  "as it stands" is a
                             statement about a specific generation.  if the generation has
                             moved, the thing C1 accepted is no longer the thing on disk, and
                             accepting it now would be accepting a different artifact under the
                             same name.

                         -> the required action is to report and re-open C1 for the Owner.
                            it is NOT the agent's call to extend C1 to the new generation,
                            and it is NOT a FAIL of the adjudication either.

    G2-UNREADABLE        the file cannot be read or hashed
                         -> NOT PASS.  report.  do not treat as unchanged.

  THE ANTI-ABSORPTION RULE, stated so it cannot be softened later
      DRIFT-DETECTED must be surfaced verbatim, with the new and old values side by side.
      it must not be written up as PASS, as PASS-with-note, or as an expected consequence of
      regeneration.  a moving artifact is a fact to report, not a condition to accommodate.
```

## 7. CV-6 — the measurement basis

```text
  PROCEDURE   read-only
              SHA-256 of build\Release\AudioEngineHarness.exe

  PASS        E5C7AFB9C4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75

  NOT PASS    report and do not close.  a different harness binary would invalidate every
              disassembly-derived address in the workstream, which is a far larger event than
              the authority question, and it would have to be adjudicated on its own.
```

## 8. CV-7 — corpus, frozen set, and process residue

```text
  PROCEDURE   read-only
              count of *.cdb under doc/work113
              SHA-256 of the 7 frozen baselines plus Baseline-3, each required to appear
              exactly once
              process list for cdb and AudioEngineHarness

  PASS        .cdb corpus == 16
              frozen 7 + Baseline-3 == 8/8 intact
              cdb processes == 0, AudioEngineHarness processes == 0

  NOT PASS    report and do not close.
```

## 9. Scope, fixed

```text
  PERMITTED
      read-only git object inspection      rev-parse, cat-file -t, cat-file -s, rev-list
      byte-preserving in-memory extraction  no shell redirection, no payload file
      SHA-1 recomputation as extraction proof
      SHA-256 and byte-size comparison
      reading the on-disk ConvoPeq.md
      one new result record under doc/work113/

  PROHIBITED
      source modification                     0
      ConvoPeq.md regeneration                0
      checkout   restore   reset   rebase    0
      commit     push      stash   gc         0
      build                                   0
      CDB execution                           0
      authority re-pin                        0
      modification of any existing record     0
      G2 validation                           0
      re-opening A, B, or C by the agent      0
      re-running this gate to obtain a better outcome   0

  NO RETRY
      DRIFT-DETECTED, NOT PASS and PREMISE MOVEMENT are all terminal for this gate.  a second
      run to obtain G2-UNCHANGED would be a re-run in the sense this workstream has already
      ruled out twice.  remediation of any of them is a NEW gate with its own authorization.
```

## 10. Outcome rule

```text
  CV-1  PASS/FAIL
  CV-2  PASS/FAIL
  CV-3  PASS/FAIL
  CV-4  PASS/FAIL
  CV-5  G2-UNCHANGED / G2-DRIFT-DETECTED / G2-UNREADABLE
  CV-6  PASS/FAIL
  CV-7  PASS/FAIL

  ALL PASS  AND  CV-5 == G2-UNCHANGED
      -> Phase 3-B CLOSED.
         and nothing beyond that.  Step5EK-R1 Contract Review remains a separate gate with
         its own authorization.

  ANY NOT PASS,  OR  CV-5 != G2-UNCHANGED
      -> Phase 3-B NOT CLOSED.  report, name the specific failure, stop.

  ON ANY PREMISE MOVEMENT, CV-1 or CV-2
      -> A1's factual basis has changed.  report first and foremost THAT, because it is the
         only outcome here that would require the adjudication itself to be revisited rather
         than merely re-confirmed.

  ONE INTERPRETATION, FLAGGED SO IT CAN BE CORRECTED
      "report and stop" is read as stopping at the CLOSURE DECISION.  CV-1..CV-7 are all
      read-only and none has side effects, so the remaining observations are still collected
      so the report is complete rather than partial.  if the Owner intends an immediate halt at
      the first failure, that is a one-line change and should be stated.
```

## 11. Artifacts

```text
  ONE new record, written only after execution
      doc/work113/P1-5-IR-P2_PHASE-3B_Closure-Validation_RESULT.md
      carries CV-1..CV-7 verbatim results, the CV-5 three-value outcome with old and new
      values side by side, the closure decision, and its own SHA-256.

  NOT created
      no payload file.  no .cdb.  no modified file of any kind.  no re-pin.
```

## 12. State

```text
Phase 2-A   CLOSED / MATCH
Phase 3-A   CLOSED / PREPARED
Phase 3-B   adjudicated A1 / B3 / C1.  CLOSURE VALIDATION NOT RUN.

authority    ADJUDICATED = G0/G1 content, committed e565019d, pushed
CR-3         PENDING
14/14        NOT MET
Implementation Authorization   FORBIDDEN
Step5EK-R1   CONTRACT DRAFTED 19,268 B.  Contract Review NOT STARTED, NOT AUTHORIZED.
Step5EK-R1 execution          NOT AUTHORIZED
Phase 2-B    NOT STARTED, NOT AUTHORIZED, not required under A1

cdb 0   build 0   source 0   commits 0   checkouts 0   re-pins 0
ConvoPeq.md regeneration 0   existing record modifications 0   payload artifacts 0

frozen 7 + Baseline-3 = 8/8 intact    .cdb corpus 16
Release AudioEngineHarness.exe  E5C7AFB9...  UNCHANGED

HOLD.  the only outstanding action is the Owner's explicit authorization to execute
       CV-1 through CV-7.
```

```text
held, untouched, and not to be folded into this gate
    1  T0P sequence conjunct generalisation
    2  the sequence operand not being observable on a non-accepting T0Q hit
    3  the two malformed T1 sites
    4  the 65 versus 116 hit non-comparability
```
