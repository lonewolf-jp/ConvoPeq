# Phase 3-A — Authority Adjudication Preparation / read-only record

```text
gate                = PREPARATION for Phase 3.  NOT Phase 3 itself.  NOT an adjudication.
mode                = READ-ONLY.  no checkout, restore, reset, rebase, amend, commit, push,
                      no ConvoPeq.md regeneration, no re-pin, no build, no CDB.
purpose             = fix the provenance of the three generations, the meaning and the
                      non-meaning of the Phase 2-A MATCH, the location of the +341-line drift,
                      and the citation-anchor consequence, so that the Owner can adjudicate.
status              = PREPARED.  Phase 3 adjudication NOT STARTED.  awaiting the Owner.
```

## 0. A correction to this step's own first verdict, stated before anything else

While producing this record I first computed a position verdict and got it **wrong**. It is
recorded here first because everything below depends on it being right.

```text
  the defective test
      I asked "is any hunk located before G2 L61993?" and answered
          "DRIFT IS BEFORE THE ANCHORS -> every anchor shifts if G1/G0 is adopted"

  why that is wrong
      the only hunk before L61993 is the one at G2 L3, and its NET line delta is 0.  it is a
      one-line replacement, not an insertion.  a net-zero replacement moves no line number.
      presence of a hunk is not the property.  the property is the CUMULATIVE NET OFFSET at
      the anchor.

      this is a proxy substituted for the property, which is the defect family this workstream
      already names twice.  the third occurrence is recorded rather than quietly fixed.

  the property, measured directly
      every one of the 13 hunks that carry a non-zero net delta begins at G2 line 95431 or
      later.  all five anchors are at G2 lines 61993..62127.  therefore

          cumulative net offset at every anchor = 0

      and the independent anchor mapping agrees: G2 L61993 maps to G1 L61993, G2 L62000 to
      G1 L62000, and so on, with content SAME on all five.

  CORRECTED VERDICT
      the +341-line drift DOES NOT AFFECT ANY OF THE FIVE CITATION ANCHORS.
      they hold the same line numbers, and the same content, in G1/G0 and in G2.
```

## 1. Provenance of the three generations

```text
  G0   LEGACY PINNED GENERATION
       representation   CRLF working-tree bytes
       size             5,535,334 B
       SHA-256          E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
       lines            121,293
       status           the authority of record for the whole workstream until this incident

  G1   e565019d COMMITTED GENERATION
       git blob         5907a1c3d053b651f6edc860e022409c89afe0f2
       size             5,414,041 B   LF
       SHA-256          827A417F2637A2779CDF33DAD98B6AC20392BC83A5CDEA675032F9720D60E4A3
       lines            121,293
       extraction       proved, sha1(b"blob 5414041\0" + data) == 5907a1c3...
       status           PUSHED.  local main == origin/main == e565019d.

  G0 <-> G1   IDENTITY, PROVED
       G1's CRLF representation is byte-exact G0.
       5,414,041 + 121,293 = 5,533,334... and the measured CRLF variant is 5,533,334 B with
       SHA-256 E5E74200...EF3609, matching G0 on BOTH size and digest.
       Phase 2-A verdict MATCH, P/Q/R all PASS.

  G2   CURRENT UNCOMMITTED WORKING TREE
       size             5,550,764 B   CRLF
       SHA-256          C587352201F0BE3D8F39D30396137A663E35A7412EAA749FBE591D661E8A7DFE
       lines            121,634
       git status       M  (363 insertions, 22 deletions against HEAD)
       mtime            2026-09-27 00:37:26, which is 36 minutes AFTER the commit
       status           NOT adjudicated
```

## 2. What the Phase 2-A MATCH means, and what it does not

```text
  MEANS
      a byte-exact candidate for G0 exists as a Git object.
      it is recoverable without any checkout, restore or regeneration.
      its content is the content of G1.

  DOES NOT MEAN
      that G0/G1 should be adopted as authoritative.  adoption is Decision A and it is the
      Owner's.  nothing has been re-pinned.
      that G2 is wrong.  G2 is simply a different generation, and the drift turns out to be
      coherent work rather than corruption.
      that the artifact is stable.  see section 6.
```

## 3. `e565019d` did not replace the authority — the earlier framing does not survive

```text
  the original incident framing
      "the pinned authority was replaced by an external commit"

  why it does not hold
      G0 and G1 are the same content.  e565019d committed what the working tree already held.
      the commit did not alter ConvoPeq.md's content relative to the pinned authority.

  what actually diverged
      G2, the CURRENT UNCOMMITTED WORKING TREE, diverged from both, by +341 lines, at
      00:37:26, 36 minutes after the commit.  the commit is not the entity that moved.

  the correct one-line statement
      e565019d committed the pinned authority content.  a later, uncommitted regeneration
      created G2.

  the incident records that used the old framing are not rewritten.  they are superseded by
  this one, and the supersession is stated rather than performed.
```

## 4. The +341-line drift, located

```text
  total          G2 121,634 lines  -  G1 121,293 lines  =  +341
  hunks          13 non-equal hunks

  G2 L3          replace, net 0        the generation timestamp line, see section 6
  G2 L95431      +1
  G2 L95434      -1
  G2 L95473      -3
  G2 L95488      -1
  G2 L95492       0
  G2 L95502       0
  G2 L95507      -3
  G2 L95511      +3
  G2 L95515     +28
  G2 L95544    +306      the dominant hunk
  G2 L99279      +5
  G2 L99295      +6
                 ---
  cumulative   +341      reconciles exactly with the line-count difference

  EVERY non-zero-net hunk begins at G2 line 95431 or later.  none is near the anchors.
```

## 5. What the drift actually contains — it is coherent new work, not damage

```text
  the drifted region is inside the P1PolyphaseGainCharacterization.cpp excerpt and its
  declaration block.  the added material is identifiable by its own markers:

      P3-5-R32      baseline settle, excluding bootstrap/rebuild publish from control
                    "episode 内 waitForDrain/drain_pre/drain_post は撤去"
                    "h.stop() の後だけ terminal state を既存 public API で観測する"
                    terminal-phase and terminalOk observation
      P3-5-FPM-Impl-1  M0/M1/M2 measurement vehicle, declared as runFpmM0/M1/M2
                    with --fpm-m0 / --fpm-m1 / --fpm-m2 dispatch entries

  the additions are internally consistent: new comments, new observation calls, new summary
  fields, new dispatch arms.  there is no sign of truncation, interleaving or partial write.

  ONE OBSERVATION THAT IS RELEVANT AND IS NOT A CLAIM
      the drifted file is a TEST HARNESS measurement vehicle, and the drift belongs to the
      P3-5-R32 / P3-5-FPM workstream.  the C3 / T0 capture workstream measures
      DeferredDeletionQueue and EpochDomain behaviour, which is not in this drifted region.
      whether the drifted harness affects the C3 gate is NOT determined here and is not
      claimed either way.
```

## 6. A property of the artifact that bears on pinning, found while doing this

```text
  G2 line 3 reads

      > Generated: 2026-09-27 00:36:31

  ConvoPeq.md EMBEDS ITS OWN GENERATION TIMESTAMP.

  CONSEQUENCE
      two generations produced from identical source content will still differ in bytes.
      a content hash of this artifact is therefore a hash of source PLUS generation time,
      not of source alone.

      this is the structural reason a content pin on ConvoPeq.md is fragile, and it is a
      property of how the file is produced rather than of any incident.  it was not visible
      while the artifact was quiescent and became visible only when it was being regenerated.

      it is recorded here because it bears directly on Decision A.  whichever generation is
      adopted, pinning by content hash pins a timestamp as well as the source.
```

## 7. The five citation anchors — verified on G2, and now also shown identical on G1

```text
  the workstream cites ConvoPeq.md by line number.  those citations were verified against the
  CURRENT WORKING TREE, that is G2.  that is now stated together with the new fact that the
  same line numbers hold on G1 with the same content.

    anchor                     G2 line   G1 line   content
    enqueue site                 61993      61993   SAME
    free test, seq - pos         62000      62000   SAME
    publishAtomic(pos + 1)       62015      62015   SAME
    publishAtomic(scanPos+kQS)   62078      62078   SAME
    publishAtomic(pos + kQS)     62127      62127   SAME

    content, verbatim on both sides
      L61993  bool enqueue(void* ptr, void (*deleter)(void*), uint64_t epoch, DeletionEntryType type,
      L62000  int32_t diff = static_cast<int32_t>(static_cast<uint32_t>(seq - pos));
      L62015  convo::publishAtomic(seq_atom, pos + 1, std::memory_order_release);
      L62078  convo::publishAtomic(seq_atom, scanPos + kQueueSize, std::memory_order_release);
      L62127  convo::publishAtomic(seq_atom, pos + kQueueSize, std::memory_order_release);

  WHY THIS MATTERS MORE THAN IT LOOKS
      a re-pin to G0/G1 does NOT invalidate the citations.  that was the risk flagged at the
      end of Phase 2-A, and the direct measurement closes it.  had the drift been positioned
      before L61993, every citation would have needed re-derivation.  it is not.

      and the reason it is not is worth stating precisely: the deferred-deletion queue
      excerpt sits at lines ~61900-62130, while all the drift sits at lines 95431+ and
      99279+.  the two regions are ~33,000 lines apart.
```

## 8. Authority has NOT been re-pinned

```text
  pinned pair of record     5,535,334 B / E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
                            retained as a historical record.  not re-affirmed, not replaced.

  G2                        not adopted.  not nominated.  not evaluated for adoption.

  working tree              untouched by this phase.  still 5,550,764 B /
                            C587352201F0BE3D8F39D30396137A663E35A7412EAA749FBE591D661E8A7DFE, still M.

  no file was written other than this record.
  no file was modified.  no git write of any kind.
```

## 9. What the Owner must decide in Phase 3 — three separable decisions

```text
  DECISION A   authoritative generation

      A1   keep G0/G1 as authority
             G0 and G1 are the same content, and G1 is PUSHED.  choosing this means the
             authority is the committed state and the uncommitted G2 drift is not authority.
             note the citation consequence is NIL, per section 7.

      A2   adopt G2 as authority
             this requires INDEPENDENT validation of G2's content.  G2 has never been
             validated by this workstream.  choosing A2 without that validation would make the
             authority an unexamined artifact, and the workstream's own rule is that an
             authority is verified before it is relied upon.

      A3   leave the authority undetermined and HOLD
             a legitimate terminal state.

  DECISION B   the +341-line uncommitted drift

      B1   intended uncommitted work in progress
      B2   unintended generation result
      B3   not yet determinable

      this is NOT a question about whether e565019d was benign.  e565019d committed G0/G1 and
      is not in question.  B concerns only the post-commit regeneration.

  DECISION C   the five citation anchors

      C1   accept them as they stand, on the strength of section 7
      C2   re-verify formally against whichever generation is adopted
      C3   defer until A is settled

      this is separable from A only in sequencing, not in substance.  if A2 is chosen, C must
      be re-verified against G2 rather than carried over from G2's own current state, because
      a re-pin to a moving artifact is exactly the situation in which a citation should be
      re-derived rather than assumed.
```

## 10. What this record does not do

```text
  it does not adopt, recommend-by-default, or rank G0/G1 over G2
  it does not declare G2 valid or invalid
  it does not declare the drift intended or unintended
  it does not re-pin anything
  it does not validate G2's content
  it does not determine whether the drifted harness affects the C3 gate
  it does not touch CR-3, which remains PENDING and is a parser question
  it does not touch Step5EK-R1, whose Contract Review is still NOT STARTED
  it does not rewrite the earlier incident records
```

## 11. State

```text
Phase 2-A    CLOSED / MATCH.  legacy authority recovery PROVEN.
Phase 3-A    PREPARED.  this record.  adjudication NOT STARTED.

authority              UNADJUDICATED.  not re-pinned.  working tree untouched.
CR-3                   PENDING
14/14                  NOT MET
Step5EK-R1             CONTRACT DRAFTED 19,268 B.  Contract Review NOT STARTED.
Step5EK-R1 execution   NOT AUTHORIZED
Phase 2-B              NOT STARTED, NOT AUTHORIZED

cdb 0   build 0   source 0   commits 0   checkouts 0   re-pins 0
regeneration 0   record modifications 0   payload artifacts 0   residue 0

frozen 7 + Baseline-3 = 8/8 intact    .cdb corpus 16, unmodified
Release AudioEngineHarness.exe  E5C7AFB9C4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75
  UNCHANGED.  the measurement basis has never been in question in any branch of this incident.

HOLD.  next is Phase 3 adjudication by the Owner: Decisions A, B and C.
```

```text
held, untouched, and not to be folded into Phase 3
    1  T0P sequence conjunct generalisation
    2  the sequence operand not being observable on a non-accepting T0Q hit
    3  the two malformed T1 sites
    4  the 65 versus 116 hit non-comparability
```
