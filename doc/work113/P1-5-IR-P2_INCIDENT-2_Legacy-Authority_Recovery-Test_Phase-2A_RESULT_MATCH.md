# PHASE 2-A — Legacy Authority Recovery Test / RESULT: **MATCH**

```text
gate                = EXECUTED under explicit Owner GRANT.  Verdict only; no adoption, no re-pin.
mode                = read-only git object inspection.  in-memory extraction.  no shell redirection.
pre-extraction identity check   PASS
integrity gate  P / Q / R        PASS / PASS / PASS   (both candidates)
VERDICT                          MATCH
authority                        UNCHANGED / UNADJUDICATED.  nothing re-pinned.
CR-3                             PENDING
R1                               NOT AUTHORIZED
```

## 0. What the Owner authorised, and what was done

```text
  authorised   1. e565019d^:ConvoPeq.md blob inspection              done
               2. e565019d :ConvoPeq.md blob inspection              done
               3. byte-preserving in-memory extraction               done
               4. P/Q/R extraction integrity validation              done, all PASS
               5. LF raw candidate SHA-256 / size                   done
               6. CRLF-transformed candidate SHA-256 / size / count  done
               7. comparison against the pinned pair                 done
               8. one new evidence record                            this file

  not done     159-commit survey, Phase 2-B, authority adoption or re-pin, assessment of
               e565019d intent, checkout, restore, reset, rebase, amend, commit, push, stash,
               gc, filter-branch, ConvoPeq.md regeneration, modification of any existing file,
               Step5EK-R1 Contract Review, CDB or R1 execution, build, source modification.

  no payload file was written.  the 5.2 MB and 5.4 MB blobs existed only in process memory.
  the only artifact this phase created is this record.

  invocation script  tmp\phase2a_recovery.py
                     sha256 0F839742AD508E192934B833A0BD4AB51D775CBC01D424D69FC10C702EA8FA4C
                     deleted after execution, digest recorded here for reproducibility
```

## 1. Pre-extraction identity check — PASS

Run before any blob content was read, as the Owner required.

```text
  spec        e565019d^:ConvoPeq.md
    rev-parse exit=0   oid 113d1570e9401ebb978d48845c4855b41cafa015   type blob   5,203,789 B
  spec        e565019d :ConvoPeq.md
    rev-parse exit=0   oid 5907a1c3d053b651f6edc860e022409c89afe0f2   type blob   5,414,041 B

  both present, both type blob, both sizes equal to the contract-pinned values.
```

## 2. Extraction integrity — P / Q / R, all PASS

```text
  P   git cat-file -s <OID> == len(extracted)
  Q   sha1(b"blob " + len + b"\0" + data) == OID
      this is git's own object-hash algorithm.  reproducing the OID is cryptographic proof
      that the extracted bytes are the blob's exact content
  R   an independent second process performs its own extraction; its digest is compared

  candidate  e565019d^:ConvoPeq.md
    P  True
    Q  True    recomputed 113d1570e9401ebb978d48845c4855b41cafa015
    R  True    child  306766ba13fcdb78fc8d2d283676a394b0d435d4d181f0587f4554b7cba404d2 5203789

  candidate  e565019d :ConvoPeq.md
    P  True
    Q  True    recomputed 5907a1c3d053b651f6edc860e022409c89afe0f2
    R  True    child  827a417f2637a2779cdf33dad98b6ac20392bc83a5cdea675032f9720d60e4a3 5414041
```

## 3. Both EOL branches

```text
  PINNED LEGACY AUTHORITY
    size   5,535,334 B
    sha256 E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609

  candidate  e565019d^:ConvoPeq.md    113d1570e9401ebb978d48845c4855b41cafa015

    branch 1  raw LF
      size            5,203,789 B      residual -331,545 B
      sha256          306766BA13FCDB78FC8D2D283676A394B0D435D4D181F0587F4554B7CBA404D2
      CR bytes inside 0                 LF count 116,729
      MATCHES PINNED  no

    branch 2  CRLF-transformed
      size            5,320,518 B      residual -214,816 B
      sha256          F3BCEBFF226DD3A0B53C3FC567FC6A3BF532EA20AF2A5FE715DC7D4A71B6C3B9
      CRLF pairs      116,729
      MATCHES PINNED  no

  candidate  e565019d :ConvoPeq.md    5907a1c3d053b651f6edc860e022409c89afe0f2

    branch 1  raw LF
      size            5,414,041 B      residual -121,293 B
      sha256          827A417F2637A2779CDF33DAD98B6AC20392BC83A5CDEA675032F9720D60E4A3
      CR bytes inside 0                 LF count 121,293
      MATCHES PINNED  no

    branch 2  CRLF-transformed
      size            5,533,334 B      residual 0
      sha256          E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
      CRLF pairs      121,293
      MATCHES PINNED  YES
```

## 4. Verdict

```text
  P/Q/R all PASS
      -> candidate bytes compared on BOTH EOL branches
      -> e565019d:ConvoPeq.md, CRLF branch, matches the pinned pair on size AND digest

  VERDICT = MATCH

  MATCHES THE PINNED PAIR EXACTLY.  both the 5,535,334 B size and the full 64-hex
  SHA-256 agree.  a size agreement alone would not have been a MATCH; the digests agree.

  MEANS
      the pinned legacy authority is byte-exact recoverable from a Git object.

  DOES NOT MEAN
      that it should be re-adopted.  adoption is Phase 3 and the Owner's judgement.
      nothing has been re-pinned.  the working tree is untouched.
```

## 5. The disjunction is resolved

```text
  branch 1  the pinned file was LF      REFUTED.  no LF branch matches, on either candidate.
  branch 2  the pinned file was CRLF    CONFIRMED.

  the pinned legacy authority was a CRLF WORKING-TREE file, exactly as
  .gitattributes (*.md text eol=lf) prescribes for a file that has not been re-checked-out.
  its content is the content of the blob committed as e565019d.
```

## 6. Two corrections this result forces on my own earlier work

### 6.1 The INCIDENT-1 hypothesis was correct, and I under-stated it

```text
  I recorded, in INCIDENT-1, the arithmetic

      5,535,334 - 5,414,041 = 121,293

  against the current working tree's 121,634 CRLF pairs, described the 341 difference as
  unexplained, and called the 121,293 agreement "a coincidence worth recording as a
  HYPOTHESIS, not a conclusion".

  the test now settles it.  the e565019d blob has EXACTLY 121,293 lines, and its CRLF
  variant is EXACTLY 5,535,334 B.  the arithmetic was not a coincidence.  it was exact.

  the 341 figure was a red herring of my own making.  I compared the OLD generation's line
  count (121,293) against the NEW generation's line count (121,634).  two different files,
  two different points in a 36-minute window.  the difference is the drift described in
  section 7, not an anomaly in the old file.
```

### 6.2 My "lossy read path" characterisation was wrong

```text
  I wrote in INCIDENT-1 that the earlier 159-commit survey read blobs "through a PowerShell
  redirection, which alters the byte stream", and that its negative could not be used.

  the retained evidence contradicts that.  the earlier survey printed, for e565019d,

      size 5,414,041 B     sha256 prefix 827A417F2637A277

  and the byte-exact value obtained now is

      size 5,414,041 B     sha256 827A417F2637A2779CDF33DAD98B6AC20392BC83A5CDEA675032F9720D60E4A3

  identical size, identical leading 16 hex digits.  that is inconsistent with a mangled
  read.  I only retained 16 hex digits at the time, so I cannot assert full-digest equality
  from the old data, but the evidence is enough to say my "lossy" attribution was wrong.

  THE SURVEY'S CONCLUSION WAS STILL RIGHT, FOR A DIFFERENT REASON
      it found nothing because it compared LF-normalised blob digests against a CRLF
      working-tree digest.  that comparison is structurally incapable of matching, and
      section 5 shows why.  the EOL asymmetry, not a broken read, is the reason.

  this is the more serious of the two errors.  a wrong reason attached to a right answer is
  more dangerous than a wrong answer, because it survives into the next decision.
```

## 7. What this changes about the shape of the incident

The framing "the authority was replaced" does not survive this result.

```text
  ESTABLISHED BY THIS PHASE

    the content of pinned authority  E5E74200... / 5,535,334 B
      IS the content of blob 5907a1c3... committed by e565019d
      (modulo the CRLF/LF working-tree convention that .gitattributes prescribes)

    therefore e565019d did NOT change ConvoPeq.md's content relative to the pinned
    authority.  it COMMITTED what the working tree already held.

  the timeline is therefore

    t0   working tree holds the pinned authority content
         e565019d commits exactly that content
    t1   2026-09-27 00:01:32   commit e565019d
    t2   2026-09-27 00:37:26   ConvoPeq.md regenerated, UNCOMMITTED
         5,550,764 B / C587352201F0BE3D8F39D30396137A663E35A7412EAA749FBE591D661E8A7DFE
         121,634 lines, that is +341 lines against the pinned generation

  the entity that actually diverged from both the pinned authority and the commit is the
  CURRENT UNCOMMITTED WORKING TREE, by +341 lines.  not the commit.

  THIS IS NOT AN ADJUDICATION.  it is a statement of what the bytes are.  which generation
  should be authoritative, and what should happen to a 341-line uncommitted drift, are
  Phase 3.
```

## 8. A concrete consequence the Owner will need for Phase 3

```text
  the workstream cites ConvoPeq.md by LINE NUMBER:

      L61993   the enqueue site
      L62000   the free test,  static_cast<int32_t>(seq - pos)
      L62015   publishAtomic(seq_atom, pos + 1, ...)
      L62078   publishAtomic(seq_atom, scanPos + kQueueSize, ...)
      L62127   publishAtomic(seq_atom, pos + kQueueSize, ...)

  those five were verified against the CURRENT WORKING TREE, that is the +341-line drift,
  and they resolved correctly there.

  they have NOT been verified against the pinned or committed generation.

  if Phase 3 re-pins to the committed generation, and if the 341 added lines lie BEFORE
  line 61993, every one of those citations shifts and each must be re-verified.

  this is stated now so that a re-pin is not treated as a bookkeeping act.  it is cheap to
  check and it is the difference between citations that are correct and citations that merely
  still look right.
```

## 9. State

```text
PHASE 2-A   EXECUTED.  VERDICT = MATCH.
            pre-extraction identity check PASS.  P/Q/R PASS on both candidates.
            byte-exact recovery of the pinned legacy authority CONFIRMED from blob
            5907a1c3d053b651f6edc860e022409c89afe0f2 via its CRLF branch.

INCIDENT-1  provenance fixed.  two of its statements are corrected by section 6.

authority                UNCHANGED / UNADJUDICATED.  nothing re-pinned either way.
CR-3                     PENDING
14/14                    NOT MET
Step5EK-R1               CONTRACT DRAFTED 19,268 B.  Contract Review NOT STARTED.
Step5EK-R1 execution     NOT AUTHORIZED
Phase 2-B                NOT STARTED, NOT AUTHORIZED

cdb 0   build 0   source 0   commits 0   checkouts 0   re-pins 0
record modifications 0   regeneration 0   payload artifacts 0

frozen 7 + Baseline-3 = 8/8 intact    .cdb corpus 16, unmodified
Release AudioEngineHarness.exe  E5C7AFB9C4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75
  UNCHANGED.  the measurement basis was never in question in any branch of this incident.

HOLD.  next is Phase 3, Owner authority adjudication.
```

```text
held, untouched, and not to be folded into Phase 3
    1  T0P sequence conjunct generalisation
    2  the sequence operand not being observable on a non-accepting T0Q hit
    3  the two malformed T1 sites
    4  the 65 versus 116 hit non-comparability
```
