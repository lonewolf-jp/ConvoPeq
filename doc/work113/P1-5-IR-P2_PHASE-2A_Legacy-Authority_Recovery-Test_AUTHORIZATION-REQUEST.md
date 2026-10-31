# PHASE 2-A — Legacy Authority Recovery Test / AUTHORIZATION REQUEST

```text
gate                = an authorization REQUEST.  NOT an authorization.  NOT a result.
mode                = NOT EXECUTED.  cdb not launched.  no git write.  no extraction performed.
owner position      = "上記の案を承認します"  — the PLAN is approved.
                      "Phase 2-A の明示的な Owner authorization が出るまでは、実行しないでください"
                      — the EXECUTION is not authorized.
status              = AWAITING EXPLICIT PHASE 2-A AUTHORIZATION
precondition        = INCIDENT-1 provenance fixed.  e565019d pushed.  authority unadjudicated.
```

## 0. The three authorizations, kept separate again

```text
  approval of THIS REQUEST DESIGN   !=   authorization to EXECUTE Phase 2-A
  a Phase 2-A result of MATCH       !=   a decision to RE-ADOPT the old authority
  the fact that e565019d is PUSHED  !=   a decision to adopt C5873522... as the new authority

  three of those four are the Owner's.  this request asks only for the first.
```

## 1. The single question Phase 2-A answers

```text
  does a byte-exact candidate for the pinned legacy authority

      5,535,334 B   SHA-256 E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609

  exist among the two Git objects

      e565019d^:ConvoPeq.md   113d1570e9401ebb978d48845c4855b41cafa015   5,203,789 B
      e565019d :ConvoPeq.md   5907a1c3d053b651f6edc860e022409c89afe0f2   5,414,041 B

  nothing else.  in particular this request does NOT ask whether the legacy authority is
  recoverable from the whole 159-commit history.  that is Phase 2-B and needs its own
  authorization.
```

## 2. Why the test needs a CRLF variant, stated before it runs

INCIDENT-1 established the structural constraint. It is repeated here because it determines
whether the test is capable of succeeding at all.

```text
  .gitattributes   *.md   text eol=lf
  check-attr       ConvoPeq.md: text: set, eol: lf

  therefore every committed ConvoPeq.md blob is LF-only, and the pinned SHA-256 was measured
  on a WORKING-TREE file whose end-of-line convention was never recorded.

  the disjunction, and it must be tested on both branches

    branch 1  the pinned file was LF
             then the raw blob bytes are the candidate and a direct SHA-256 is the test

    branch 2  the pinned file was CRLF
             then NO LF-normalised blob can ever hash to the pinned value, because the
             pinned digest is over a different byte sequence.  the candidate is the blob
             with LF converted to CRLF, and its SHA-256 is the test.

  without branch 2 the test is STRUCTURALLY INCAPABLE of returning MATCH, and a NO MATCH
  would carry no information about the CRLF case at all.

  this is why branch 2 is inside Phase 2-A rather than deferred.  it is the same single
  question, not an additional one.
```

## 3. Method — byte-preserving, and self-proving

The earlier recovery attempt failed for a specific, identified reason: the blobs were read
through a PowerShell redirection, which alters the byte stream. This method removes shell
redirection from the data path entirely.

```text
  EXTRACTION, in memory, no shell redirection, no intermediate file

      python  subprocess.run(['git','cat-file','blob',OID], capture_output=True).stdout

      capture_output=True yields the raw byte stream.  no shell is involved in the transfer,
      so no encoding conversion and no newline translation can occur.

  NO ARTIFACT OF THE PAYLOAD IS WRITTEN
      the 5.2 MB and 5.4 MB payloads exist only in process memory.  only digests and sizes
      are recorded.  this is narrower than "extraction to a file" and is proposed because it
      removes any chance of a stray 5 MB file entering the repository.

  EXTRACTION IS PROVED, NOT ASSUMED.  this is the part that makes the three-valued result
  trustworthy, and it is a precondition rather than a hope.

      three independent cross-checks, all of which must hold

      P  git cat-file -s <OID>            size from the object header, computed by git
                                          must equal len(extracted)

      Q  sha1( b"blob " + str(len) + b"\0" + extracted )  == OID
                                          this is git's own blob hash algorithm.  if it
                                          reproduces the OID, the bytes are PROVABLY the
                                          blob's exact content, not merely bytes that a
                                          tool claims came from it

      R  re-extraction in a second, independent process, digest compared
                                          rules out a one-off transport artefact

  if any of P, Q, R fails, the result is INCONCLUSIVE.  it is not NO MATCH.
```

## 4. Result — three values, and what each one does NOT license

```text
  MATCH
      a candidate reproduces the pinned pair exactly.
      LICENSES    a statement that the legacy authority is byte-exact recoverable.
      DOES NOT LICENSE
                   re-adopting it.  recoverability and adoption are separate judgements, and
                   the second one additionally depends on Phase 3.  a MATCH against a blob
                   that is 110 hours older than the current source is evidence about
                   recoverability, not evidence about which generation should be authoritative.

  NO MATCH
      neither object, on either EOL branch, reproduces the pinned pair.
      MEANS        "not in these two generations".
      DOES NOT MEAN
                   "absent from the 159-commit history".  Phase 2-B exists precisely because
                   this negative is scoped.  stating it otherwise is the error INCIDENT-1
                   already made once and corrected.

  INCONCLUSIVE
      extraction or comparison did not hold.  P, Q or R failed, or the OID could not be read.
      THIS IS A REAL OUTCOME, not a failure to be hidden.  a test that cannot return
      INCONCLUSIVE will eventually return a confident wrong answer.
```

## 5. Secondary observations, recorded but NOT decisive

Included because they are nearly free once the bytes are in memory, and because excluding
them would be selecting evidence.

```text
  per candidate, recorded alongside the verdict

      size, SHA-256 of the raw LF bytes
      size, SHA-256 of the CRLF variant
      CRLF pair count of each variant
      the arithmetic residual against the pinned size, 5,535,334

  THE HYPOTHESIS FROM INCIDENT-1 IS TESTED HERE, NOT ASSUMED
      5,535,334 - 5,414,041 = 121,293,  and the current working tree has 121,634 CRLF pairs.
      if the e565019d blob's CRLF variant has 121,293 pairs, its size is exactly 5,535,334
      and the hypothesis is confirmed as a size coincidence.

  EXPLICITLY, a size match is NOT a MATCH
      matching size with non-matching SHA-256 is a NO MATCH.  size is a filter, never a
      verdict.  the verdict is the digest pair only.

  the 341-pair discrepancy itself remains unexplained and is not to be resolved by
  preferring whichever reading is convenient.
```

## 6. Authorized scope, if and when granted

```text
  PERMITTED
      read-only Git object inspection        rev-parse, cat-file, cat-file -s
      byte-preserving blob extraction        in memory, no shell redirection
      SHA-256 and byte-size comparison
      recomputation of the git blob SHA-1 as an extraction proof
      creation of ONE new recovery-evidence record under doc/work113/

  PROHIBITED
      checkout        restore        reset         rebase        amend
      commit          push           stash         gc            filter-branch
      ConvoPeq.md regeneration
      working-tree modification of any existing file
      authority re-pin, in either direction
      modification of any existing workstream record
      the Phase 2-B 159-blob survey
      Step5EK-R1 Contract Review
      Step5EK-R1 execution, and any CDB invocation
      build
      any assessment of whether e565019d was intended
```

## 7. Artifacts

```text
  ONE new record
      doc/work113/P1-5-IR-P2_INCIDENT-2_Legacy-Authority_Recovery-Test_Phase-2A.md
      carries the P / Q / R cross-check results, the per-candidate digests on both EOL
      branches, the three-valued verdict, and its own SHA-256.

  NOT created
      no extracted payload file.  no .cdb.  no modified file of any kind.
      residue 0 at completion, verifiable by git status and by process list.
```

## 8. What Phase 2-A cannot tell the Owner, stated plainly

```text
  it CANNOT establish which ConvoPeq.md generation is authoritative.  that is Phase 3.

  it CANNOT establish whether e565019d was an intended commit.  that is Phase 3.

  it CANNOT establish that the legacy authority is lost.  a NO MATCH is scoped to two
  generations.  only a Phase 2-B NO MATCH over the full history, with a proven extraction,
  would support that statement, and even then the CRLF branch would remain a separate
  question for any generation whose pinned EOL convention is unknown.

  it CANNOT make CR-3 anything other than PENDING.  CR-3 is a parser question and remains
  untouched by any of this.

  the release harness anchor E5C7AFB9C4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75
  is unaffected throughout, so the measurement basis is not in question in any branch.
```

## 9. State

```text
INCIDENT-1   provenance fixed
PHASE 2-A    AUTHORIZATION REQUESTED, NOT EXECUTED

Step5EK-R1   CONTRACT DRAFTED 19,268 B, retained unchanged
CR-3         PENDING        14/14  NOT MET        Implementation Authorization  FORBIDDEN

cdb 0   build 0   source 0   commits 0   checkouts 0   extractions 0
re-pins 0   record modifications 0   regeneration 0   residue 0

ConvoPeq.md  UNADJUDICATED.  neither the pinned pair nor C5873522... is adopted.

HOLD.  the only outstanding action is the Owner's explicit Phase 2-A authorization.
```

```text
held, untouched, and not to be folded into Phase 2 or Phase 3
    1  T0P sequence conjunct generalisation
    2  the sequence operand not being observable on a non-accepting T0Q hit
    3  the two malformed T1 sites
    4  the 65 versus 116 hit non-comparability
```
