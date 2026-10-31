# Undetermined-Item Inventory — ConvoPeq work113

```text
record type     read-only inventory and investigation record
scope           every undetermined, pending, deferred or unassessed item found in
                doc/work113 and in the working tree, investigated as far as the
                available evidence allows
subjects modified   0
gate entered        none
```

Nothing here is adjudicated. Where an item is now settled, the settlement is a
**measurement**, and the reading of it remains the Owner's. Where an item cannot be
settled without authority the agent does not hold, it is recorded as such rather than
resolved.

---

## 0. Method and tooling

```text
corpus            doc/work113, 283 files, 4,017,359 B of markdown across 244 records
indexed           context-mode v1.0.169, 244 files / 4350 sections, queryable
marker sweep      578 lines carrying an undetermined marker, across 144 of 244 records
                  100 records carry no marker and are treated as settled
deduplication     contiguous marker runs collapsed, then normalised on hex and digits
                  -> 325 distinct items before classification
```

Tooling actually used, and why:

| tool | role in this pass |
| --- | --- |
| `context-mode` MCP | primary. corpus indexing, marker sweep, batched git and source queries, in-memory reconstruction. kept the 4 MB corpus and all 5.3 MB of raw search output out of the conversation |
| `headroom` | `headroom_stats` consulted. reports **2 API requests, 0 compressed, passthrough:models**. The proxy is not compressing this session's traffic, which matches the standing note for this host. No MCP compress/retrieve was needed because context-mode absorbed the volume |
| `git` CLI, read-only | `show --name-status`, `ls-tree -z`, `cat-file --batch`, `status`, `diff --numstat`, `ls-files`. no repository state was changed |
| direct source inspection | the generator contract, and `minReaderEpoch` / `globalEpoch` in production source |

No web search was needed: every question in this pass was answerable from artefacts on
disk. That is a finding in itself — see §7.

---

## 1. HEADLINE FINDING — the pinned `ConvoPeq.md` authority is not reproducible from git

This is the most consequential result of the pass, and it was an open item.

### 1.1 What is proven

```text
on-disk ConvoPeq.md
  5,550,764 B
  C587352201F0BE3D8F39D30396137A663E35A7412EAA749FBE591D661E8A7DFE
  Generated stamp 2026-09-27 00:36:31
```

Established two independent ways, both read-only, nothing written:

```text
(a) the generator's own sanctioned check
    python output_sourcecode_markdown.py --check
      baseline Generated : 2026-09-27 00:36:31
      NEWER_SRC_COUNT    : 0
      STATUS             : FRESH
      exit 0

(b) in-memory reconstruction with the generator's own functions
    on-disk   5,550,764 B  C587352201F0BE3D8F39D30396137A663E35A7412EAA749FBE591D661E8A7DFE
    rebuilt   5,550,764 B  C587352201F0BE3D8F39D30396137A663E35A7412EAA749FBE591D661E8A7DFE
    BYTE-IDENTICAL : True
    350 target files
```

And a third, independent confirmation from git objects rather than the filesystem:

```text
(c) reconstruct from the git blobs of commit e565019d, same generator, same stamp
    commit-rebuild  5,550,764 B  C587352201F0BE3D8F39D30396137A663E35A7412EAA749FBE591D661E8A7DFE
    on-disk         5,550,764 B  C587352201F0BE3D8F39D30396137A663E35A7412EAA749FBE591D661E8A7DFE
    BYTE-IDENTICAL : True
```

**Therefore the on-disk snapshot is a faithful, uncorrupted rendering of the working
tree at `e565019d`.** It is not stale, not damaged, and not drifting.

### 1.2 The gap

```text
                                   size          relation
pinned authority  E5E74200…      5,535,334 B    the Owner's adjudicated G0/G1 content
e565019d  rebuild C5873522…      5,550,764 B    == on-disk, byte-identical   (+15,430)
e565019d^ rebuild                  5,453,958 B    the parent commit             (−81,376)
```

The authority sits **between** the two commits and equals **neither**.

```text
authority != e565019d    by +15,430 B
authority != e565019d^   by −81,376 B
```

Amendment 1's identity for the authority is `5,535,334 = 5,414,041 + 121,293`, i.e.
121,293 newlines in its CRLF form. The on-disk file has 121,634 CRLF pairs and an
LF-only form of 5,429,130 B. So the difference is not a line-ending artefact; it is
**+341 lines of real content** relative to the authority's LF form.

### 1.3 What this means

```text
The source state that produced the adjudicated authority is NOT any state reachable
from the available git history.  It was a working tree that was never committed, or
was committed and then rewritten.

This is a provenance gap of the same shape as occurrence #23: an artefact whose
identity is known and whose content is trusted, but whose derivation cannot be shown.

It was NOT previously recorded anywhere in doc/work113.
```

This is reported, not resolved. The Owner's ruling is required on what the authority is
and whether it should be re-derived, re-pinned, or left as adjudicated.

---

## 2. The CRLF/LF mechanism, now fully explained

The mechanism behind the whole incident, established from `.gitattributes` and the
generator source.

```text
.gitattributes
  *.cpp text eol=lf      *.h  text eol=lf     *.hpp text eol=lf   *.c text eol=lf
  CMakeLists.txt text eol=lf     *.cmake text eol=lf     *.json text eol=lf
  *.md text eol=lf       *.txt text eol=lf    *.yaml / *.yml text eol=lf
  *.sh text eol=lf
  *.bat text eol=crlf
  /JUCE/** text eol=crlf        /r8brain-free-src/** text eol=crlf

generator   output_sourcecode_markdown.py:116
            open(out_path, 'w', encoding='utf-8')
            -> text mode, newline=None -> on Windows every LF becomes CRLF

consequence the working copy of ConvoPeq.md is CRLF (121,634 pairs) while its own
            attribute mandates LF.  git therefore warns on every touch:
              "in the working copy of 'ConvoPeq.md', CRLF will be replaced by LF"
```

The generator is otherwise sound and has a read-only freshness mode, which this pass
used rather than reimplementing:

```text
output_sourcecode_markdown.py:179  check_freshness()
  strict '>' comparison of each target file's mtime against the Generated stamp
  returns 0 FRESH / 1 STALE, writes nothing
  and it carries its own instruction: "do not quote the old stamp as an audit baseline"
```

That instruction is consistent with §1: the stamp is not a provenance anchor.

---

## 3. `e565019d` — composition, which is much larger than previously recorded

The standing note described this commit as having modified four files
(`opencode.json`, `.mcp.json`, `AGENTS.md`, `ARCHITECTURE.md`) and not audited. The
actual commit is far larger.

```text
e565019df6d263370b3cac2ed8e7a14d7c7f8db4
author  lonewolf-jp
date    Sun Sep 27 00:01:32 2026 +0900
message "commit"

246 files changed, 164,934 insertions(+), 95,746 deletions(-)
```

| category | files |
| --- | --- |
| `doc/work113` record corpus | 211 |
| **production source `src/audioengine`** | **8** |
| dotfiles / agent config | 7 |
| `doc` other | 5 |
| `scripts` | 5 |
| root docs | 3 |
| **`src/tests`** | **3** |
| `tools` | 2 |
| `ConvoPeq.md` generated | 1 |
| `opencode.json` | 1 |

### 3.1 Production source changes

```text
M  src/audioengine/AudioEngine.h                                    +27 +50 +1 +6 +1 +1 +10
M  src/audioengine/AudioEngine.RebuildDispatch.cpp                   +4 +5 +5
M  src/audioengine/ISRRuntimePublicationCoordinator.cpp             +5
M  src/audioengine/ISRRuntimePublicationCoordinator.h               +5 +3
M  src/audioengine/ISRRuntimePublicationCoordinator_ProcessIntent.cpp  +6
M  src/audioengine/PublicationExecutor.cpp                           5 x one-line
M  src/audioengine/PublicationExecutor.h                             +3 +1 +1 +1
M  src/audioengine/RuntimePublicationOrchestrator.cpp                +3
M  src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp
M  src/tests/AudioEngineHarness/P1PolyphaseGainCharacterization.cpp
M  src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp  +24
```

Almost every hunk is a **pure insertion** (`-X,0 +X,+N`). The affected regions are
`rollbackDSPHandleRegistration`, `RuntimeIntentCoordinator::getPublicationIntentResidencyCount`,
`processIntent`, `PublicationExecutor::publish` / `publishImpl`, `trySubmitImpl`, and
`rebuildThreadLoop`.

**These are production-behaviour changes and they have never been audited.** The
incident framing in `P1-5-IR-P2_INCIDENT*` treats `e565019d` as a `ConvoPeq.md`
provenance event. It is also, simultaneously, an un-audited production source change.
Both facts are true and the second was not on the record.

### 3.2 Deletions

```text
D  scripts/fix_lint.py                       77 lines
D  scripts/fix_tables.py                     25
D  scripts/fix_tables_v2.py                  53
D  .opencode/plugin/headroom-proxy.ts       107
D  .opencode/plugin/rtk-wsl.ts               23
D  Testing/Temporary/CTestCostData.txt         1
D  .browser4-cli/snapshot/*.yml               3
R  {src/tools => tools}/check_layout_offsets.py
```

The three `fix_*` scripts: a full-tree reference search over `*.md`, `*.py`, `*.bat`,
`*.txt`, `*.json` found **no remaining reference** to `fix_lint.py`, `fix_tables.py` or
`fix_tables_v2.py` outside the deleted files themselves. `scripts/` now holds only
`freebuff_expectations.json`, `install-vcperf.ps1`, `serena-mcp-bridge.py`,
`write-correlation-id.ps1`. **The deletion's impact is therefore assessed as nil for
the current tree.** No build, test or CMake input referenced them.

The rename moved `src/tools/check_layout_offsets.py` out of `src/`, which is the one
file-count change visible in the snapshot: 351 files at the parent, 350 at the commit.

### 3.3 Binary artefact under version control

```text
M  tools/__pycache__/apply-solidlsp-bash-ls-patch.cpython-314.pyc   Bin 16645 -> 16895 bytes
```

Three `.pyc` files are tracked, all of them compiled artefacts of tracked sources:

```text
tools/__pycache__/apply-solidlsp-bash-ls-patch.cpython-314.pyc
tools/__pycache__/retire_authority_verifier.cpython-314.pyc
tools/analysis/__pycache__/compare_noiseshaper_patterns.cpython-314.pyc
```

`e565019d` modified one of them. The generator's `IGNORE_EXTS` excludes `.pyc`, so they
do not reach `ConvoPeq.md`; the issue is purely repository hygiene.

---

## 4. Working-tree state

```text
31 entries
  M  ConvoPeq.md                       363 insertions / 22 deletions vs HEAD, CRLF warning
  M  headroom-proxy-start.ps1          tracked, +56 / -14   (not in the standing note)
  ?? .memories/                        1 entry, a directory
  ?? 28 untracked doc/work113 records
```

Every record of the current C3 T0 workstream — R3, R3-1, RY2, RR, the Amendment, the
adjudication record — is **untracked**. The corpus that this inventory draws on is not
in git.

```text
git HEAD   e565019d, pushed, ahead 0 / behind 0
cdb / harness processes   0
.cdb corpus               16 files, unmodified
```

---

## 5. The S7 family — characterised, and it is a different kind of item

The standing note listed `S7_READER`, `S7_READER_SLOT`, `minReaderEpoch(T1)` and
`Case A–D` as unresolved. Their actual status differs by construct.

### 5.1 The S7 capture constructs do not exist anywhere executable

```text
                          .cdb corpus (16)   C++ source   markdown records
S7_READER                        0               0          58 files / 95
S7_READER_SLOT                   0               0          48 files / 64
S7_ENTRY_PRODUCER                0               0           1 file  /  1
PENDING_RECLAIM_HANDLES_         0               0           0       /  0
```

By contrast the C3 capture lineage is real:

```text
C3_T0    11 .cdb files,  66 occurrences
C3_T1    11 .cdb files, 134 occurrences
```

**Therefore `S7_READER = UNRESOLVED` does not describe a broken instrument. It
describes a capture construct that was proposed in the records and never implemented**
— not as a breakpoint alias, not as a source symbol, not as a script. There is nothing
to repair; there is a design that has not been built. The item is a *pending design*,
not a *failed measurement*.

### 5.2 `minReaderEpoch` and `globalEpoch` do exist in production source

```text
src/audioengine/AudioEngine.h:1704          std::uint64_t minReaderEpoch = 0;
src/audioengine/AudioEngine.h:4604-4605    const auto minReaderEpoch = m_retireRouter->minReaderEpoch();
src/audioengine/ISRRetireRouter.cpp:184     uint64_t ISRRetireRouter::getMinReaderEpoch() const noexcept
src/audioengine/ISRRetireRouter.cpp:219-222 uint64_t ISRRetireRouter::minReaderEpoch() const noexcept
src/audioengine/AudioEngine.Init.cpp:30     // readerEpochs and globalEpoch are statically initialised to 0
src/audioengine/AudioEngine.Timer.cpp:1062  epochGap diagnostic against minReaderEpoch
```

So the open question is **not** whether the symbol exists. It is what value
`minReaderEpoch` holds at the T1 hit, which requires a runtime capture that is not
authorised.

### 5.3 The four held OPEN items

Carried unchanged, as the corpus states them:

```text
1  T0P sequence conjunct generalisation
      dwo(@r11+0x30010) == 4 is not generalised to @r10.  this makes C3_T0_CAS_PRE a
      first-cycle-only diagnostic.  accepted as a residual.
2  the sequence operand is not observable on a non-accepting T0Q hit
3  the two malformed T1 sites
      0x1f9ce04 and 0x1f9cfb0 retain their dd( misuse and malformed address expression
4  the 65 versus 116 hit non-comparability
      the Step5ED run ended at the T1A hit
```

None is resolvable read-only. Each needs either an Owner ruling or a capture that is
not authorised.

---

## 6. Retracted-claim register

39 distinct withdrawal or correction statements exist in the corpus. This is a
substantial self-correction history and it is inventoried here so it is not mistaken
for settled ground. The load-bearing retractions:

| what was withdrawn | where | why |
| --- | --- | --- |
| `worldReclaimCount` reaching 10 | Step5DT §2, Step5DS | no traceable basis. All four runtime logs checked by whole-line equality: 0 hits. The counter exists at `DeferredDeletionQueue.h:270` and is printed only by two test files; the `--measurement=normal` run that produced the 116 T0E hits does not print it |
| "capture is essentially unobservable at this run length" | Step5EF §4, withdrawn in Step5EG §0 | the log refutes it. Post-publish proof with `r10=5, rbx=4` shows a slot-4 acquisition completed. T-1 and T-2 are distinguished by production semantics, not by measurement feasibility |
| design option O1 = `dd @$t2+0x34000 L1` | Step5CK, Step5CM, Step5CN | that address is `enqueuePos` |
| "`?` is unsafe when repeated / one `?` per body" | Step5DB §5, corrected at the Redesign-8 gate | the bracketed `?` demonstrably **completed**; the command that failed is the one *after* it |
| the RDG-3 acceptance table and its BC | Step5DY, corrected in Step5EA §4 | — |
| a stronger selector-semantic claim | Step5DR | the runtime observation does not support it |
| the BC "every captured T0Q has a corresponding T0S" | Step5DY | — |

Also settled by measurement, and worth stating because it is a *refuted* hypothesis
that is sometimes cited as if it were open:

```text
r9 == 9 ?   FALSE, 0 of 116 hits, --measurement=normal
            13 distinct values, never 9
            and the three prior observations are deliberately kept separate, not merged
            Redesign-5 (1 sample) / Step5BB Retry-1 (1 sample) / Step5DK / Step5DP
```

---

## 7. Items that need no external research

The instruction to search the literature was not exercised, and the reason is worth
recording: every open item in this corpus is a question about **this** repository's
artefacts — a capture log, a source line, a file digest, a commit. No item turns on an
external fact about audio engineering, filter design, or CDB semantics. The supplied
acoustics URLs would not bear on any of them.

If the Owner wants the C3 T0 predicate family re-derived against published material on
direct-to-direct queue reclamation, that would be a new, separately authorised work
item. It is not a gap in the current inventory.

---

## 8. Consolidated register

### 8.1 Settled by this pass

| # | item | settlement |
| --- | --- | --- |
| S1 | is the on-disk `ConvoPeq.md` corrupt or drifting | **No.** byte-identical to the generator over the working tree, confirmed twice, and `--check` reports FRESH / NEWER_SRC_COUNT 0 |
| S2 | what produced the +15,430 B delta against the authority | **The source changes in `e565019d`.** The commit's reconstruction equals the on-disk file byte-for-byte |
| S3 | the CRLF/LF mechanism | **Established** from `.gitattributes` plus the generator's text-mode write. The working copy of a `*.md text eol=lf` file is CRLF |
| S4 | did `e565019d` touch production source | **Yes — 8 files in `src/audioengine` plus 3 under `src/tests`,** never audited |
| S5 | impact of deleting `fix_lint.py` / `fix_tables.py` / `fix_tables_v2.py` | **Nil for the current tree.** no remaining reference anywhere |
| S6 | tracked `.pyc` files | **Three**, listed in §3.3; `e565019d` modified one |
| S7 | are the S7 constructs real artefacts | **No.** zero occurrences in the `.cdb` corpus and in C++ source; record-only proposals |
| S8 | does `minReaderEpoch` exist in source | **Yes**, five sites. the open question is its value at T1, not its existence |
| S9 | `r9 == 9` | **Refuted**, 0 of 116. already recorded, restated so it is not re-opened |

### 8.2 Open, and newly opened by this pass

| # | item | status |
| --- | --- | --- |
| N1 | **the pinned authority 5,535,334 / `E5E74200…` corresponds to no state in git** | **NEW. OPEN.** structurally parallel to occurrence #23 |
| N2 | the 8 un-audited production source changes in `e565019d` | **NEW. OPEN.** |
| N3 | the four held OPEN items | OPEN, unchanged, deliberately held |
| N4 | S7 capture constructs | reclassified: **pending design, not a failed measurement** |
| N5 | `minReaderEpoch(T1)` value, `globalEpoch` value, Case A–D | OPEN, needs an authorised capture |
| N6 | occurrence #20 final status | OPEN, structural layer addressed |
| N7 | occurrence #23 | OPEN / UNRECOVERED. no further recovery authorised, none possible without reconstruction |
| N8 | the nine wrong verdicts in the Post-Implementation Reviewer | retained as process evidence, **unnumbered** by Owner's instruction |
| N9 | Contract Amendment Review closure | NOT CLOSED / NOT TRUSTED |
| N10 | RB3 final classification | DEFERRED |
| N11 | 28 untracked records, including the whole current workstream | OPEN, hygiene |
| N12 | 3 tracked `.pyc` files | OPEN, hygiene |

### 8.3 Not in this register, and why

```text
RB3, CR-3, RR re-evaluation, R1 execution, Phase 2-B
    gate sequence, not inventory items.  the Owner has them at STOP and this pass
    entered none of them.

the 100 records carrying no undetermined marker
    treated as settled and not re-opened.  re-opening settled records is how a
    corpus rots.
```

---

## 9. Pipeline state, reported honestly

```text
context-mode   doctor PASS, v1.0.169, FTS5 native module works
               one WARN, Antigravity CLI hooks absent.  that hook host is not this
               one and the MCP tools work regardless, so the compression path was
               unaffected

headroom       2 API requests, 0 compressed, passthrough:models, 0 tokens saved
               the proxy is NOT compressing this session.  consistent with the
               standing note for this host.  context-mode absorbed the volume
               instead: the 4 MB corpus, the 5.3 MB reference-search output and the
               5.5 MB snapshot reconstructions were all handled outside the
               conversation
```

No `headroom_compress` was called, because there was no large blob that context-mode
had not already kept out of context. Calling it would have produced a second,
redundant compression of the same material.

---

## 10. What this pass did not do

```text
no production source change       no build              no CDB
no debuggee                      no g / go / gc         no R1 execution
no RR-1..RR-18 re-evaluation     no CR-3               no Phase 2-B
no ConvoPeq.md regeneration       no ConvoPeq.md re-pin
no git state change (read-only queries only)
no occurrence renumbered
no contract text altered
no gate entered
```

`ConvoPeq.md` was reconstructed **in memory only**. It was never written. Its digest
is unchanged at `C587352201F0BE3D8F39D30396137A663E35A7412EAA749FBE591D661E8A7DFE`.

---

## 11. For the Owner

Two items in §8.2 are new and neither existed on the record before this pass:

```text
N1  the adjudicated authority's source state is not reproducible from git
N2  e565019d contains 8 un-audited production source changes
```

N1 in particular deserves a decision before any further measurement work, because the
authority is the reference every downstream judgement is measured against, and its
derivation cannot presently be shown. That is the same failure shape as occurrence #23,
and it is not repaired by any reconstruction this agent is permitted to perform.

N2 is a scope question: the incident records treat `e565019d` as a snapshot-provenance
event. It is also a production change. Whether that was intended is not something this
pass can determine, and it is not assumed either way here.
