# INCIDENT-1 — External Commit Provenance Audit (`e565019d`)

```text
gate                = PHASE 1, read-only provenance fixation.  NOT Phase 2.  NOT a repair.
authorized scope    = "e565019d の provenance を read-only で固定する調査を承認する"
mode                = READ-ONLY git plumbing only.
cdb launches 0      build 0      source edit 0      ConvoPeq.md regeneration 0
checkout 0          restore 0    reset 0            rebase 0                  amend 0
commits 0           working-tree modification 0
                    the only file created by this step is this record, under doc/work113/
NOT done            no blob CONTENT was extracted.  no SHA-256 was computed over any
                    historical blob.  that is Phase 2 and it is not authorized.
```

## 1. Commit identity, verbatim from the object

```text
  commit        e565019df6d263370b3cac2ed8e7a14d7c7f8db4
  tree          49574e4f168bfc645e35947cd4f03456d7ffb705
  parent        1e9e63e34bed7adb9342ebc81259ded9689fc48a
  merge?        NO.  single parent.
  author        lonewolf-jp <lonewolf-jp@users.noreply.github.com>
  committer     lonewolf-jp <lonewolf-jp@users.noreply.github.com>
                author and committer are the same identity and the same instant
  timestamp     1790434892 +0900
                = 2026-09-26 15:01:32 UTC
                = 2026-09-27 00:01:32 +0900
  parent date   2026-09-22 09:59:31 +0900
  interval      110 hours, that is 4.6 days, since the parent commit
  message       "commit"      one word, no body
  gpgsig        absent.  the commit is not signed.
```

```text
  repository context that governs byte identity

    object format        sha1
    core.autocrlf        false
    core.eol             unset
    core.safecrlf        unset
    .gitattributes       present, and it contains

                          *.md   text eol=lf

    git check-attr text eol -- ConvoPeq.md

                          ConvoPeq.md: text: set
                          ConvoPeq.md: eol: lf

    CONSEQUENCE, and this is the load-bearing fact of this whole incident
                          ConvoPeq.md is a TEXT file with eol=lf.  git converts CRLF to LF
                          when the file is added to the object store.  therefore

                              every committed ConvoPeq.md blob is LF-only
                              the working-tree copy is CRLF
                              the two are different byte sequences BY DESIGN

                          this is why git emitted
                              "in the working copy of 'ConvoPeq.md', CRLF will be replaced
                               by LF the next time Git touches it"
```

## 2. Git object identity of ConvoPeq.md — the provenance anchor

The Owner's point is taken: the working-tree SHA-256 and the Git blob OID are different
kinds of identity, and the OID is the right anchor for provenance.

```text
  blob OID   e565019d^:ConvoPeq.md   113d1570e9401ebb978d48845c4855b41cafa015
  size                            from the object header, no redirection   5,203,789 B

  blob OID   e565019d :ConvoPeq.md   5907a1c3d053b651f6edc860e022409c89afe0f2
  size                            from the object header, no redirection   5,414,041 B

  the two differ.  the commit DID change ConvoPeq.md.

  working-tree file, three distinct identities, all obtained without writing

    raw bytes, SHA-256                    C587352201F0BE3D8F39D30396137A663E35A7412EAA749FBE591D661E8A7DFE
    raw bytes, size                       5,550,764 B
    raw CRLF bytes, blob OID              0f01736e054be76883ba5c2f7f31f9909a7c7431
    git-CLEANED CRLF->LF, blob OID        a82bf5dbc56601f976d66e46be11334403ef4d41

    does the cleaned working tree equal the committed blob?   NO

  git hash-object was used WITHOUT -w, so nothing was written to the object database.

  CRLF accounting, which closes the size gap arithmetically

    working-tree size                    5,550,764 B
    CRLF pairs in working tree             121,634
    LF-only equivalent size               5,429,130 B
    committed blob at e565019d            5,414,041 B
    real content delta after normalisation    +15,089 B

    so the working tree is NOT merely a CRLF variant of the commit.  after normalising
    line endings there are still 15,089 bytes that the commit does not contain.  the file
    was regenerated AFTER the commit was made.

    timeline that follows from the two timestamps
        commit        2026-09-27 00:01:32
        worktree mtime  2026-09-27 00:37:26     ~36 minutes later
```

## 3. Scope of the commit — 247 paths, not 12

My earlier incident record described this as a commit that "swept in 12 source files". That
was the src/ portion only, and it understated the commit by an order of magnitude.

```text
  total paths changed      247
      A  added             216
      M  modified           21
      D  deleted            10

  aggregate line movement  +164,986 / -95,798

  the 216 additions are almost entirely documentation
      doc        213
      scripts      2
      tools        1

  the 21 modifications, complete
      .gitignore
      .mcp.json
      AGENTS.md
      ARCHITECTURE.md
      ConvoPeq.md
      opencode.json
      doc/work113/implementation_report_20260922_c1_ref_fidelity.md
      doc/work113/remediation_plan_20260922_v3.2_revised.md
      doc/work113/residual_tasks_20260919.md
      src/audioengine/AudioEngine.RebuildDispatch.cpp
      src/audioengine/AudioEngine.h
      src/audioengine/ISRRuntimePublicationCoordinator.cpp
      src/audioengine/ISRRuntimePublicationCoordinator.h
      src/audioengine/ISRRuntimePublicationCoordinator_ProcessIntent.cpp
      src/audioengine/PublicationExecutor.cpp
      src/audioengine/PublicationExecutor.h
      src/audioengine/RuntimePublicationOrchestrator.cpp
      src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp
      src/tests/AudioEngineHarness/P1PolyphaseGainCharacterization.cpp
      src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp
      tools/__pycache__/apply-solidlsp-bash-ls-patch.cpython-314.pyc

  the 10 deletions, complete
      .browser4-cli/snapshot/snapshot-2026-07-11T11-28-08-685Z.yml
      .browser4-cli/snapshot/snapshot-2026-07-11T11-28-23-436Z.yml
      .browser4-cli/snapshot/snapshot-2026-07-11T11-28-32-314Z.yml
      .opencode/plugin/headroom-proxy.ts
      .opencode/plugin/rtk-wsl.ts
      Testing/Temporary/CTestCostData.txt
      scripts/fix_lint.py
      scripts/fix_tables.py
      scripts/fix_tables_v2.py
      src/tools/check_layout_offsets.py

  items that deserve the Owner's attention beyond ConvoPeq.md itself, recorded without
  inflation and without further investigation in this phase

      opencode.json   MODIFIED.  this is the project's MCP server configuration, which this
                      session's tooling depends on.  what changed in it was NOT examined.
      .mcp.json       MODIFIED.  same caveat.
      AGENTS.md       MODIFIED.  the file that documents the tool policy for this repository.
      a tracked .pyc  MODIFIED.  tools/__pycache__/...cpython-314.pyc is version-controlled,
                      which is itself unusual.
      three scripts   DELETED, scripts/fix_lint.py, fix_tables.py, fix_tables_v2.py.
                      these are text-maintenance scripts; their loss was not assessed here.
```

## 4. Push state — the blast radius is shared, not local

```text
  local  main      e565019df6d263370b3cac2ed8e7a14d7c7f8db4
  origin/main      e565019df6d263370b3cac2ed8e7a14d7c7f8db4
  local ahead of origin/main      0
  local behind origin/main       0
  branch containing e565019d     main (only)

  the commit is PUSHED.  local and origin/main are the same object.

  WHY THIS MATTERS FOR THE DECISION AHEAD
      any remediation is therefore not a local operation.  a corrective commit, a revert, or
      a history rewrite would propagate to the shared remote.  that is a materially larger
      decision than "undo it locally", and it is a further reason the standing instruction to
      avoid checkout / restore / reset is the right default rather than mere caution.
```

## 5. A correction to my own earlier work, which Phase 1 has now made structural

Earlier in this incident I reported that none of the 159 commits touching ConvoPeq.md
matched the pinned hash, and I flagged the read path as lossy so the negative could not be
used as evidence. That was half the diagnosis. The size comparison was structurally
incapable of matching, for a reason Phase 1 establishes:

```text
  the pinned authority pair is    5,535,334 B / E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609

  that pair was measured on the WORKING TREE.  git blobs are LF-normalised.  therefore

    IF the pinned file was CRLF
        its SHA-256 is over CRLF bytes, and no LF-normalised blob can ever hash to it.
        it was never a git object, and never will be.

    IF the pinned file was LF
        then a blob of exactly 5,535,334 B would be the candidate.  the two blobs adjacent to
        the event are 5,203,789 B and 5,414,041 B, neither of that size.  but the wider
        history contains blobs of 13,197,708 B and 18,323,986 B, so sizes vary widely and
        this is NOT a survey.  a metadata-only size survey could settle it.

  the honest conclusion available from Phase 1 alone
      the EOL convention of the pinned generation was never recorded, so its recoverability
      cannot be decided from the pair (size, SHA-256) alone.  and the CRLF branch of the
      disjunction is already closed: under CRLF it is structurally unrecoverable from git.

  a coincidence worth recording as a HYPOTHESIS, not a conclusion
      5,535,334 - 5,414,041 = 121,293, and the current working tree has 121,634 CRLF pairs.
      the two numbers are within 341 of each other.  if the pinned file had 121,293 CRLF
      pairs its LF size would be exactly the e565019d blob size.  this may be pure
      coincidence, the two files being at different points in a 4.6-day interval.  it is
      recorded because it is testable in Phase 2 and because leaving it out would be
      selecting evidence.

  WHAT THIS DOES NOT DO
      it does not restore anything, and it does not decide the authority.  it says the
      recovery question is now sharply posed rather than open-ended, and that one of its two
      branches is already closed on structural grounds.
```

## 6. What Phase 1 deliberately did NOT do

```text
  not done   no blob content extracted, by any route
  not done   no SHA-256 computed over any historical blob
  not done   no metadata-only size survey of all 159 ConvoPeq.md blobs.  that is cheap and
             needs no content extraction, but it exists to answer the Phase 2 question, so
             it is left for Phase 2 authorization rather than taken here
  not done   no checkout, restore, reset, rebase, amend, stash, or garbage collection
  not done   no ConvoPeq.md regeneration
  not done   no edit to any existing workstream record
  not done   no authority re-pin, in either direction
  not done   no assessment of whether e565019d was INTENDED.  that is Phase 3 and it is the
             Owner's judgement, not an inference from metadata
```

## 7. State

```text
INCIDENT-1   provenance FIXED.  the facts above are read-only observations.

  NOT DECIDED   whether e565019d was an intended commit
  NOT DECIDED   which ConvoPeq.md generation is authoritative
  NOT STARTED   Phase 2 recovery test          (needs authorization)
  NOT STARTED   Phase 3 authority adjudication (Owner's)
  NOT STARTED   Phase 4 record amendment       (blocked on Phase 3)
  NOT STARTED   Step5EK-R1 Contract Review     (held)

Step5EK-R1   CONTRACT DRAFTED, 19,268 B, retained unchanged
CR-3         PENDING
14/14        NOT MET
Implementation Authorization   FORBIDDEN

Release AudioEngineHarness.exe  E5C7AFB9C7F4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75
  UNCHANGED.  the measurement basis is unaffected by any of the above.

frozen 7 + Baseline-3 = 8/8 intact    .cdb corpus 16, unmodified
ConvoPeq.md  UNVERIFIED as an authority.  working-tree 5,550,764 B /
             C587352201F0BE3D8F39D30396137A663E35A7412EAA749FBE591D661E8A7DFE
             committed blob 5907a1c3d053b651f6edc860e022409c89afe0f2 / 5,414,041 B
             neither is adopted.  the pinned pair remains a historical record only.

cdb 0   build 0   source 0   commits 0   checkouts 0   regeneration 0   residue 0

HOLD.  Phase 3, or Phase 2 with authorization.  Both are the Owner's.
```
