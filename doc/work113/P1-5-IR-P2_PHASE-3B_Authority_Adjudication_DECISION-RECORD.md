# Phase 3-B — Authority Adjudication / read-only decision record

```text
gate                = ADJUDICATION INSTRUMENT.  A decision is NOT made here.
                      A / B / C are recorded as OPEN, with their consequences fixed.
                      Filling them in is the Owner's act alone.
mode                = READ-ONLY.  adjudication only.
precondition        = Phase 2-A MATCH (legacy authority recovery proven)
                      Phase 3-A prepared (G0/G1/G2 provenance, drift located, anchors mapped)

PROHIBITED IN THIS PHASE, until A / B / C are all filled
    checkout
    restore / reset / rebase
    ConvoPeq.md regeneration
    authority re-pin
    source modification
    build
    CDB execution
    commit / push

ALSO UNTOUCHED, and deliberately not mixed with this phase
    CR-3                     PENDING.  a parser question, unrelated to the authority.
    Step5EK-R1               Contract Review NOT STARTED, execution NOT AUTHORIZED.
    Phase 2-B                NOT STARTED, NOT AUTHORIZED.
```

## 0. A factual note on referencing the authority, raised as a precondition

The Owner asked that the latest authoritative `ConvoPeq.md` be confirmed directly
referenceable before its content is re-examined. Stating precisely what is and is not
directly available, so the precondition can be evaluated rather than assumed.

```text
  G2   CURRENT WORKING TREE   DIRECTLY FILE-REFERENCABLE, and already has been, byte-exactly,
                               throughout this incident
                                 5,550,764 B
                                 SHA-256 C587352201F0BE3D8F39D30396137A663E35A7412EAA749FBE591D661E8A7DFE
                                 121,634 lines
                                 line-level content read at every cited anchor
                                 read via Get-FileHash, Get-Content, ReadAllBytes; no
                                 redirection in any measurement path

  G0 / G1  COMMITTED GENERATION   NOT a file on disk in that form.  they exist as a git object
                                 and are obtained by in-memory extraction, which is PROVED:
                                   sha1(b"blob 5414041\0" + data) == 5907a1c3d053b651f6edc860e022409c89afe0f2
                                 and independently re-derived in a second process, and the
                                 header size agrees with the extracted length.

                                 no payload file is written in either case, by design.

  ∴ the precondition is SATISFIED for G2 and SATISFIED-BY-PROOF for G0/G1.
    it is not a blocker for Phase 3-B.  it would become a blocker only for any option that
    required materialising a generation as a file, and no option here does.
```

## 1. Decision A — authoritative generation

```text
  QUESTION   which generation is the authoritative source for this workstream?

  A1   KEEP G0/G1
       what it asserts   the authority is the content that is COMMITTED and PUSHED.
                         G0 and G1 are the same content, proved byte-exact in Phase 2-A.

       what it is grounded in
                         the authority of record for the entire workstream was G0, and
                         Phase 2-A proved G0 == G1.  so A1 is not a change of authority at
                         all; it is a confirmation that the pinned authority is the committed
                         state, plus the closure of its recoverability question.

       consequences
                         citation anchors   unaffected, per Phase 3-A section 7.
                                           G2 L61993..62127 map to G1 L61993..62127, content SAME.
                         working tree       G2 remains an uncommitted, unadjudicated artifact
                                           on disk.  A1 does not delete it, revert it, or
                                           touch it.
                         re-pin             NOT required.  the pair of record already describes
                                           this content, in its CRLF form.  a re-pin would
                                           only be a restatement, and per Phase 3-A section 6 it
                                           would additionally pin a generation timestamp.
                         G2                 becomes "not authority" by exclusion, not by
                                           invalidation.  its content is not judged.

  A2   SELECT G2 AS CANDIDATE  --  and this is explicitly NOT adoption
       the Owner has already fixed the split, and it is recorded here as part of the option:

           A2 selected
             -> G2 INDEPENDENT VALIDATION, as a SEPARATE gate
             -> only on validation PASS is re-pin / adoption reconsidered
             -> until then the authority remains UNDETERMINED, not "G2 pending paperwork"

       what validation would have to establish, at minimum
           - G2's content is internally coherent, not a partial or interleaved write
           - the drifted region does not alter the DeferredDeletionQueue / EpochDomain
             behaviour that the C3 gate measures
           - the deferred-deletion queue excerpt the workstream relies on is unchanged
             between G1 and G2 in the region of interest
         NOTE  the third item is already partially answered.  Phase 3-A measured that all drift
         hunks begin at G2 line 95431 or later, while the queue excerpt sits at 61900-62130.
         that is a localisation result, not a validation, and it does not stand in for one.

       what A2 does NOT do
           it does not adopt.  it does not re-pin.  it does not declare G2 valid.
           it does not permit any ConvoPeq.md content to be relied upon for new citations
           until validation has passed.

  A3   HOLD
       the authority stays undetermined and the workstream stops.
       a legitimate terminal state, and the only option that consumes nothing.

  DECISION A   :  ____________________     ( A1 / A2 / A3 )      date  __________
```

## 2. Decision B — disposition of the +341-line drift

```text
  QUESTION   how is the post-commit +341-line uncommitted generation to be handled, as a
             provenance and workflow matter?

  SCOPE LIMIT, fixed by the Owner and recorded here so it cannot drift
      B does NOT judge the technical validity of G2.
      B does NOT judge whether e565019d was benign.  e565019d committed G0/G1, which is not
      in question, and it is pushed.
      B concerns ONLY the regeneration that produced G2 at 00:37:26, 36 minutes after the
      commit.

  B1   INTENDED UNCOMMITTED WORK IN PROGRESS
       what it asserts   the +341 lines are deliberate, ongoing work that simply has not been
                         committed yet.
       what it is grounded in
                         the drift is internally coherent and self-identifying: P3-5-R32
                         baseline-settle and terminal-observation work, and P3-5-FPM-Impl-1
                         M0/M1/M2 measurement-vehicle declarations and dispatch arms, with
                         matching comments, calls, summary fields and dispatch entries.  no
                         truncation, interleaving or partial-write signature was observed.
       consequence
                         the workstream may proceed with G2 on disk while treating it as
                         uncommitted.  this is a statement about intent, not about
                         correctness, and it does not make G2 the authority.  that is A.

  B2   UNINTENDED GENERATION RESULT
       what it asserts   G2 should not have been produced in that state.
       consequence
                         the remediation question opens.  and the remediation is NOT a local
                         operation, because e565019d is pushed: local main == origin/main.
                         a corrective commit, a revert, or a history rewrite would propagate
                         to the shared remote.  that is a materially larger decision than
                         undoing it locally, and it is a separate authorization.
       NOT DONE HERE
         no checkout, restore, reset, revert, corrective commit, or regeneration is performed
         or proposed for execution by this record.

  B3   NOT YET DETERMINABLE
       what it asserts   the intent cannot be established from the available evidence.
       consequence
                         this is a legitimate answer, not a failure.  the evidence bearing on
                         intent is the author's own account, which is not something a
                         read-only inspection can supply.  B3 with A1 leaves the workstream
                         fully unblocked.

  DECISION B   :  ____________________     ( B1 / B2 / B3 )      date  __________
```

## 3. Decision C — disposition of the five citation anchors

```text
  QUESTION   what is done about the five line-number citations?

  ESTABLISHED IN PHASE 3-A, by direct measurement of the property and not by a proxy
      G2 L61993 -> G1 L61993    content SAME     enqueue site
      G2 L62000 -> G1 L62000    content SAME     free test, seq - pos
      G2 L62015 -> G1 L62015    content SAME     publishAtomic(pos + 1)
      G2 L62078 -> G1 L62078    content SAME     publishAtomic(scanPos + kQueueSize)
      G2 L62127 -> G1 L62127    content SAME     publishAtomic(pos + kQueueSize)

      cumulative net line offset at every anchor = 0
      every non-zero-net drift hunk begins at G2 line 95431 or later
      the queue excerpt sits at 61900-62130, the drift at 95431+ and 99279+, about 33,000
      lines apart

      ∴ the concern that the +341 lines shift the anchors is RESOLVED.

  C1   ACCEPT THE MAPPING AS IT STANDS
       grounded in the above.  available immediately if A1 is chosen, and sound in that case
       because the anchors are identical in the authority and in the working tree.

  C2   FORMALLY RE-VERIFY AGAINST THE ADOPTED GENERATION
       the correct choice if A2 is chosen.  the reason is specific: under A2 the authority
       becomes a generation that is not committed and is regenerated, and Phase 3-A
       established that the artifact embeds its own generation timestamp, so two
       regenerations of identical source differ in bytes.  a citation carried over from a
       previous reading of a moving artifact is an assumption, and under A2 it should be
       re-derived rather than inherited.

  C3   DEFER UNTIL A IS SETTLED
       the default if B or C ordering is preferred, or if A3 is chosen.  under A3 the question
       is moot, because there is no adopted generation to verify against.

  DECISION C   :  ____________________     ( C1 / C2 / C3 )      date  __________
```

## 4. Dependencies between the three decisions

```text
        A1  keep G0/G1        A2  select G2 candidate     A3  hold
        ───────────────       ────────────────────       ─────────
  B     independent          independent                independent
        of A.  B is about    of A.  B is about           of A.  B can still be
        intent of the drift  intent of the drift        recorded under A3
        and does not judge   and does not judge
        G2's validity        G2's validity

  C     C1 available.       C2 required.  G2 cannot     C3.  the question is
        the anchors are     not authority until its      moot, no adopted
        identical in the    independent validation       generation exists to
        authority and in    passes, and a citation       verify against
        the working tree    must not be inherited
                             across a moving artifact

  THE ORDER MATTERS IN ONE PLACE ONLY
      A2 without C2 would leave the workstream citing line numbers in an artifact it has not
      validated and cannot pin.  A2 therefore implies C2.
      A1 does not imply anything about C beyond making C1 sound.
```

## 5. What each outcome authorises next, and what it still does not

```text
  A1  authority = G0/G1 content, confirmed committed and pushed
      next        Phase 3-B closes.  the workstream may return to Step5EK-R1 Contract Review,
                  which is a SEPARATE gate and is NOT authorised by A1.
      still not   CR-3 stays PENDING.  14/14 stays NOT MET.  Implementation Authorization
                  stays FORBIDDEN.  G2 is not deleted, reverted or judged.

  A2  G2 selected as CANDIDATE ONLY
      next        a separate G2 independent-validation gate, defined before it runs.
      still not   no re-pin.  no adoption.  no reliance on G2 content for new citations.
                  CR-3 stays PENDING.  Step5EK-R1 untouched.

  A3  authority UNDETERMINED, workstream stops
      next        nothing.  this is an endpoint, not a pause with queued work.
```

## 6. What this record has not done

```text
  it has not filled in A, B or C
  it has not adopted, re-pinned, or ranked G0/G1 against G2
  it has not judged G2 valid or invalid, and has not judged the drift intended or unintended
  it has not validated any ConvoPeq.md content
  it has not run, scheduled or proposed any G2 validation gate
  it has not touched CR-3, Step5EK-R1, Phase 2-B, or the four held OPEN items
  it has not written to ConvoPeq.md, to any source file, or to git
  it has not modified any earlier record; the incident records that used the superseded
  "authority replaced" framing remain as written and are superseded by Phase 3-A
```

## 7. State

```text
Phase 2-A   CLOSED / MATCH
Phase 3-A   PREPARED
Phase 3-B   INSTRUMENT CREATED.  A / B / C = OPEN, awaiting the Owner.

authority              UNADJUDICATED.  not re-pinned.  working tree untouched.
CR-3                   PENDING
14/14                  NOT MET
Step5EK-R1             CONTRACT DRAFTED 19,268 B.  Contract Review NOT STARTED.
Step5EK-R1 execution   NOT AUTHORIZED
Phase 2-B              NOT STARTED, NOT AUTHORIZED

cdb 0   build 0   source 0   commits 0   checkouts 0   re-pins 0
ConvoPeq.md regeneration 0   record modifications 0   payload artifacts 0

frozen 7 + Baseline-3 = 8/8 intact    .cdb corpus 16, unmodified
Release AudioEngineHarness.exe  E5C7AFB9C4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75
  UNCHANGED.

HOLD.  the only outstanding action is the Owner's A / B / C determination.
```

```text
held, untouched, and not to be folded into Phase 3
    1  T0P sequence conjunct generalisation
    2  the sequence operand not being observable on a non-accepting T0Q hit
    3  the two malformed T1 sites
    4  the 65 versus 116 hit non-comparability
```
