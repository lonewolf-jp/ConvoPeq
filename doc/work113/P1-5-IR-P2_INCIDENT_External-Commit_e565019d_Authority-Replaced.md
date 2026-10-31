# INCIDENT — external commit `e565019d` and authority replacement

```text
gate                = an INCIDENT RECORD.  NOT a gate.  NOT an authorization.  NOT a repair.
detected            = during the mechanical verification that followed drafting the
                      Step5EK-R1 Pre-Execution Contract
mode                = READ-ONLY throughout.  no commit, no checkout, no restore, no regeneration,
                      no build, no source edit, no .cdb edit
cdb launches 0      debuggees 0     images executed 0     harness 0
status              = REPORTED.  no repair attempted.  Contract Review NOT started.
```

## 1. What was found

```text
  the pinned authority has been replaced, and is not a stable pin

    of record    5,535,334 B   SHA-256 E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
    now          5,550,764 B   SHA-256 C587352201F0BE3D8F39D30396137A663E35A7412EAA749FBE591D661E8A7DFE
    delta        +15,430 B

  and it moved more than once while this was being inspected

    mtime 2026-09-27 00:28:59   SHA-256 F1E923584C7EFA2003E57BBEFCEE879C4E6E6B5D6C025149CD68C668584FE352
    mtime 2026-09-27 00:37:26   SHA-256 C587352201F0BE3D8F39D30396137A663E35A7412EAA749FBE591D661E8A7DFE

    same byte length, different content.  four samples over 15 s after 00:37:26 were identical,
    so it is currently quiescent, but it is not a fixed artifact.

  it is CRLF, and git has already warned that it will be normalised

    121,634 CRLF pairs
    git: "in the working copy of 'ConvoPeq.md', CRLF will be replaced by LF the next time
          Git touches it"
    git status:  M ConvoPeq.md      git diff:  363 insertions, 22 deletions

    CONSEQUENCE   a hash that moves between generations, and that git has announced it will
                 rewrite on the next touch, cannot serve as a pinned authority even if it is
                 re-pinned today.  this is a property of the artifact's production, not of
                 this workstream.
```

## 2. The commit that did it

```text
  e565019d   message: "commit"        2026-09-27 00:01:32 +0900
  reflog     e565019d HEAD@{0}: commit: commit
  HEAD~1     1e9e63e3  2026-09-22 09:59:31

  note on the message.  the neighbouring commits all carry descriptive messages
  ("feat(work113): ...", "docs(work113): ...", "chore(evidence): ..."), and a long history of
  bare "commit" messages exists in this repository, so the message alone is not evidence of
  accident.  what IS notable is the SCOPE: HEAD~1 was 4.5 days earlier.

  what the commit swept in, under src/  (12 files, +1944 / -84)

      M  src/audioengine/AudioEngine.RebuildDispatch.cpp
      M  src/audioengine/AudioEngine.h                                        +104
      M  src/audioengine/ISRRuntimePublicationCoordinator.cpp
      M  src/audioengine/ISRRuntimePublicationCoordinator.h
      M  src/audioengine/ISRRuntimePublicationCoordinator_ProcessIntent.cpp
      M  src/audioengine/PublicationExecutor.cpp
      M  src/audioengine/PublicationExecutor.h
      M  src/audioengine/RuntimePublicationOrchestrator.cpp
      M  src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp                  +659
      M  src/tests/AudioEngineHarness/P1PolyphaseGainCharacterization.cpp     +1126
      M  src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp       +24
      D  src/tools/check_layout_offsets.py                                      -52

  also in the commit
      M  .gitignore   .mcp.json   AGENTS.md   ARCHITECTURE.md   ConvoPeq.md
      D  .opencode/plugin/headroom-proxy.ts
      D  .opencode/plugin/rtk-wsl.ts
      D  Testing/Temporary/CTestCostData.txt
      D  .browser4-cli/snapshot/*.yml
      A  many doc/work113/* artifacts, INCLUDING this workstream's own Step5E* records

  the growth of the authority is EXPLAINED, and that is not the same as AUTHORISED
      ConvoPeq.md concatenates build.bat, CMakeLists.txt and src/.  the commit added about
      1,944 source lines.  15,430 B over 1,944 lines is roughly 7.9 B per line, which is
      plausible for this content.  the replacement is therefore consistent with a
      regeneration that absorbed the newly committed sources.

      an explanation is not a re-pin.  which generation is authoritative is a decision.
```

## 3. What survives — and it is the part that carries the workstream

```text
  THE DISASSEMBLY BASIS IS INTACT.  this is the load-bearing good news.

    build\Release\AudioEngineHarness.exe
      now       SHA-256 E5C7AFB9C4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75
      anchor    SHA-256 E5C7AFB9C4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75
      MATCH

    therefore all of the following remain valid, because they are properties of that binary
      ImageBase 0x140000000
      T0E 0x141f9c550   T0P 0x141f9c57d   T0Q 0x141f9c59a   T0S 0x141f9c5da
      the register roles, r10 == pos at T0Q, rbx == slot index, r11 == base
      the T0Q fingerprint that the Step5EI contract's BEFORE line was matched against
      the je 0x141f9c586 CAS-success topology that CR-7 rests on

    consistency check, and it holds
      src mtimes   AudioEngine.h 2026-09-23 22:06:18
                   P1PolyphaseGainCharacterization.cpp 2026-09-24 00:58:40
      binary mtime 2026-09-24 01:00:47
      the Release binary was built AFTER those source edits, so it already reflects the
      current sources.  the sources and the binary are not in conflict.

  THE SPECIFIC SOURCE ANCHORS STILL RESOLVE
    re-read at the same line numbers against the new generation

      L61993   bool enqueue(void* ptr, void (*deleter)(void*), uint64_t epoch, DeletionEntryType type,
      L62000           int32_t diff = static_cast<int32_t>(static_cast<uint32_t>(seq - pos));
      L62015               convo::publishAtomic(seq_atom, pos + 1, std::memory_order_release);
      L62078               convo::publishAtomic(seq_atom, scanPos + kQueueSize, std::memory_order_release);
      L62127               convo::publishAtomic(seq_atom, pos + kQueueSize, std::memory_order_release);

      all five still carry exactly the content this workstream cites them for.  the enqueue
      site, the free test, and the three publications that establish sequences[] holding
      ABSOLUTE position values are all still where the workstream says they are.
```

## 4. What is now false in the existing record

```text
  every workstream record states, in its State block, some form of

      "nothing committed or pushed"   /   "all doc/work113/* artifacts uncommitted"

  that statement is NOW FALSE.  e565019d committed the 12 long-uncommitted source files and
  added this workstream's own Step5E* records to git.

  this is recorded rather than quietly corrected, because the correction is not the agent's
  to make silently: which generation of which artifacts should be the committed record is an
  Owner decision, and until it is made, the honest statement is that the earlier claim was
  true when written and is no longer true now.
```

## 5. Collateral, assessed for accuracy rather than inflated

```text
  src/tools/check_layout_offsets.py   DELETED by e565019d

      what it actually checked, from git history:  "Confirms the rebuilt executables reference
      a single, consistent worldAuthority_.coordinator_ slot offset (ctor store == publish-side
      loads) and that the old crash-era offsets are absent."  tag D162-2-I3-4-D.

      that is the COORDINATOR slot offset family, not this workstream's queue offsets
      (+0x30000 sequences base, +0x30010 sequences[4], +0x34000 enqueuePos).

      IMPACT ON CR-3 AND ON THE T0Q PREDICATE:  none.  a real loss of a regression check for
      a different subsystem, and worth the Owner's attention, but it does not bear on this gate.

  .opencode/plugin/rtk-wsl.ts, headroom-proxy.ts   DELETED

      the deleted paths are  .opencode/plugin/  singular.
      AGENTS.md documents   .opencode/plugins/rtk-wsl/index.ts   plural, and a directory.

      different paths.  this record does NOT assert that the plugin AGENTS.md documents was
      removed.  the check was not made, because it is not load-bearing for CR-3 and this
      incident is not the place to expand into it.
```

## 6. Impact on the gate sequence — stated narrowly

```text
  CR-3 is UNAFFECTED, and it is worth being precise about why.

      CR-3 asks whether the cdb MASM expression grammar accepts
      dwo(@r11+0x30010) == @r10.

      that question is about a PARSER.  it reads no source, no authority file, and no
      measurement.  ConvoPeq.md has no bearing on it whatsoever.

  the Step5EK-R1 Pre-Execution Contract is UNAFFECTED IN SUBSTANCE.
      its fixed items are the debuggee identity, SHAPE A, the cdb pin, the probe script, the
      outcome classes and the abort conditions.  none of those reads ConvoPeq.md.
      what IS stale is the contract's citation of the authority, which is a documentation
      defect, not a substantive one.

  what IS blocked
      any statement of the form "the authority is 5,535,334 B / E5E74200...".  that is now
      false, and the Owner's standing instruction was to maintain exactly that state.
```

## 7. On recoverability of the original generation — an honest negative

```text
  159 commits touch ConvoPeq.md.  none of the 159 matched the pinned hash.

  BUT that comparison is NOT usable as evidence.  the blobs were read through a PowerShell
  redirection, which alters the byte stream, so the lengths and hashes obtained are not the
  committed bytes.  the correct statement is

      "no byte-exact recovery attempt was completed, and the quick check produced no match
       through a lossy read path"

  NOT "the original authority is unrecoverable", and NOT "it is recoverable".

  a byte-exact check, if the Owner wants one, is cheap:  git cat-file blob <rev>:ConvoPeq.md
  written to disk by the git process itself rather than by a shell redirection.
```

## 8. Options for the Owner

```text
  OPTION A  re-pin the authority to the current generation
            5,550,764 B / C587352201F0BE3D8F39D30396137A663E35A7412EAA749FBE591D661E8A7DFE
            and record that all prior citations were made against E5E74200...
            CAVEAT  the artifact is CRLF and git has announced it will LF-normalise it, and it
                    moved twice during this inspection.  a re-pin today is a pin on a moving
                    target unless the generation is first made deterministic.

  OPTION B  adjudicate e565019d before pinning anything
            if the commit was unintended, re-pinning now would cement an unintended state.
            this is the reason OPTION A is not simply the obvious choice.

  OPTION C  decouple the gate from the authority and proceed
            CR-3 does not read ConvoPeq.md, so the R1 Contract Review could proceed while the
            authority question is settled separately.
            this is defensible on the merits, and it is the Owner's call, not the agent's.

  OPTION D  a byte-exact recoverability check of the original generation, then decide.

  NOT ATTEMPTED, deliberately
    no git checkout, no restore, no regeneration of ConvoPeq.md, no attempt to reconstruct
    the previous generation, and no edit of any record's State block.  every one of those
    would destroy evidence or pre-empt a decision that is not the agent's.
```

## 9. State

```text
Step5EI        DRAFTED, unchanged
Step5EJ        CLOSED  13 PASS / 0 FAIL / 1 PENDING
Step5EK        CLOSED / INCONCLUSIVE
Step5EK-R1     PRE-EXECUTION CONTRACT DRAFTED, 19,268 B
               Contract Review NOT STARTED.
               EXECUTION NOT AUTHORIZED.

CR-3   PENDING          14/14  NOT MET          Implementation Authorization  FORBIDDEN

frozen 7 + Baseline-3 = 8/8 intact    .cdb corpus 16, unmodified
Release AudioEngineHarness.exe  E5C7AFB9...E34C75  UNCHANGED  — the measurement basis holds

cdb launches 0   debuggees 0   images executed 0   harness 0   production runs 0
builds 0   .cdb edits 0   source edits 0   commits 0   checkouts 0

HOLD.  Contract Review not started.  The authority question is the Owner's.
```
