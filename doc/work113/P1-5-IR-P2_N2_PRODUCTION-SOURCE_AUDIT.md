# N2 Audit — `e565019d` production-source change set

```text
record type      read-only audit
authorised       N2 read-only audit, production source, 8 files
N1               OPEN, untouched by this record, not resolved here
gate entered     none
subjects modified   0
```

**N1 and N2 are independent.** N2 is a provenance-and-coverage question about a commit.
N1 is a provenance question about the adjudicated authority. Completing N2 does not close
N1, and N1's gap is not a reason to characterise the N2 changes as invalid. Neither
judgement is made here.

**No evaluation.** This record classifies and describes. It does not say safe, dangerous,
correct or incorrect, and no such inference is drawn from the classification.

---

## 1. N2 status

```text
question      what production-source change does e565019d contain
answer        five separately recorded and separately build-gated P3-5 workstream items,
              captured by e565019d from a working tree that had been left uncommitted
coverage      DIRECT.  the change set is already declared, sized and attributed in
              doc/work113, and the declaration matches the measured diff exactly
verdict       the change set is DOCUMENTED, not undocumented
new OPEN      1 candidate, see section 8
```

The standing note described these as "un-audited production source changes". That
description was accurate about the *audit status* and inaccurate about the *record
status*: the changes were implemented under their own authorisations and build gates.
What was missing was a single place that said so. This record is that place.

---

## 2. The 8 files, exact diff identity

`e565019d^` → `e565019d`, read-only.

| file | parent blob | commit blob | + | − |
| --- | --- | --- | --- | --- |
| `src/audioengine/AudioEngine.h` | `48db614ef225` | `ffc1559424eb` | 100 | 4 |
| `src/audioengine/AudioEngine.RebuildDispatch.cpp` | `20711e5dc969` | `1610ff1b7e59` | 14 | 0 |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp` | `5688650e9cea` | `cf197566edde` | 5 | 0 |
| `src/audioengine/ISRRuntimePublicationCoordinator.h` | `07b4d6c75a52` | `46b22b5133b4` | 8 | 0 |
| `src/audioengine/ISRRuntimePublicationCoordinator_ProcessIntent.cpp` | `b40ed2add9a5` | `b3b641b20f66` | 6 | 0 |
| `src/audioengine/PublicationExecutor.cpp` | `dffdab0dd754` | `799576118487` | 10 | 5 |
| `src/audioengine/PublicationExecutor.h` | `18f59f6bf591` | `d12074ce1112` | 8 | 2 |
| `src/audioengine/RuntimePublicationOrchestrator.cpp` | `86c131c32edb` | `297a8facfb17` | 4 | 1 |
| **total** | | | **155** | **12** |

### 2.1 The declaration already on the record

```text
P1-5-IR-P2_Step5AL_P3-5-R30_recovery_origin_vehicle_implementation_build_gate.md
  "production diff = R27 のみ（8 files・+155/−12・R30 追加 0）"
```

Measured against the declaration:

```text
file count   8    == 8         MATCH
added      155    == 155       MATCH
deleted     12    == 12        MATCH
```

Three independent quantities agree exactly. The change set is self-declaring.

### 2.2 Where the work came from

```text
every P3-5 record cites   HEAD 1e9e63e34bed7adb9342ebc81259ded9689fc48a
1e9e63e3 is an ancestor of e565019d                       rc 0
commits touching src/audioengine in 1e9e63e3..e565019d    1  (e565019d itself)
```

So the sequence is: work implemented and build-gated against `1e9e63e3`, left uncommitted
in the working tree, then captured by `e565019d` together with the record corpus. There is
no intervening source commit and therefore no unaccounted source delta.

```text
recent history
  e565019d  2026-09-27  commit
  1e9e63e3  2026-09-22  docs(work113): record P1-1 polyphase gain characterization
  d6995b2b  2026-09-22  feat(work113): gate polyphase gain convention correction
  eeeba325  2026-09-22  docs(work113): record Phase 1 preimplementation audit
  de95f335  2026-09-22  test(work113): freeze Phase 0 evidence and condition-D eligibility
  88c6f000  2026-09-22  chore(evidence): untrack generated verification artifacts
```

---

## 3. N2-B — the change set decomposes into five recorded items

Every added line carries a `★` workstream marker. The markers are the key: this is not
anonymous code, it is five labelled increments.

### 3.1 P3-5-R16 — rebuild take counter

```text
files     AudioEngine.h, AudioEngine.RebuildDispatch.cpp
symbols   rebuildTakeCount_  (new atomic, AudioEngine.h public/private boundary)
          getRebuildTakeCount()  (new [[nodiscard]] reader)
          writer in AudioEngine::rebuildThreadLoop()
fact      increments only immediately after the ownership transfer completes.
          explicitly NOT incremented on queue success, wake, backlog clear,
          build, commit or publish.
          process-lifetime cumulative, no reset, operated differentially.
          the code states: "queued==taken の再同一視をしないこと"
record    P1-5-IR-P2_Step5X_P3-5-R16_take_counter_implementation_build_gate.md
```

### 3.2 P3-5-R19 — B2 build-result counter

```text
files     AudioEngine.h, AudioEngine.RebuildDispatch.cpp
symbols   rebuildBuildResultCount_, getRebuildBuildResultCount()
fact      increments only on reaching a usable runtime.  explicitly NOT incremented on
          validation, commit or publish (cites R18 §6 and §11).
          process-lifetime cumulative, no reset.
record    P1-5-IR-P2_Step5AA_P3-5-R19_buildresult_counter_implementation_build_gate.md
          which also records an isolation check:
            "R16照合 PASS … R16以外のproduction変更の混入なし"
```

### 3.3 P3-5-R22 — main-site commit-enqueue counter

```text
files     AudioEngine.h, AudioEngine.RebuildDispatch.cpp
symbols   rebuildCommitEnqueueCount_, getRebuildCommitEnqueueCount()
fact      increments only on main-site commit-enqueue.  explicitly EXCLUDES the two
          recovery enqueue sites, named in the comment as :1085 and :1165.
          not incremented on queue, take, build, commit or publish.
record    P1-5-IR-P2_Step5AD_P3-5-R22_commit_enqueue_counter_implementation_build_gate.md
          which records "R21 設計（main-site commit-enqueue・単一位置）をそのまま実装"
```

### 3.4 P3-5-R10 — read-only active-runtime snapshot

```text
files     AudioEngine.h
symbols   struct RuntimeActiveSnapshot  (new, 16 POD fields)
          getActiveRuntimeSnapshot()    (new [[nodiscard]] inline reader)
fact      a diagnostics value object.  the comment states it returns no ownership,
          no handle and no pointer; the RCU read completes inside the scope.
          on no-world it returns zero-initialised, and the caller detects
          invalidity by all-zero generation / worldId / publicationSequence.
          "新 validity state なし" — no new validity state was introduced.
          reads worldAuthority_.acquireReadToken() then consumeWorldHandle()
record    P1-5-IR-P2_Step5R_P3-5-R10_accessor_implementation_build_gate.md
          whose gates include
            G4  new POD = 1 / new public method = 1
            G5  exposure of RuntimeWorld* / & / RuntimeReadHandle /
                GlobalSnapshot* / shared_ptr / unique_ptr  = 0 occurrences
          predecessor P1-5-IR-P2_Step5Q_P3-5-R9_accessor_implementation_approval.md
```

### 3.5 P3-5-R27 (R26-A path A) — origin plumbing and Main-origin take counter

This is the largest item, 8 files, and the only one that alters an existing value's
provenance rather than adding a diagnostic.

```text
(a) counter
    files     ISRRuntimePublicationCoordinator.h / .cpp / _ProcessIntent.cpp,
              AudioEngine.h
    symbols   coordinatorTakeCount_  (new atomic, private)
              getCoordinatorTakeCount()  (declared and defined; the AudioEngine.h
              reader delegates to runtimePublicationBridge_)
    fact      the sole writer is in RuntimeIntentCoordinator::processIntent, and it
              increments only when commonIntent.payload.publish.recoveryObligationId == 0.
              the comment states the counter is a separate thing from the residency
              gauge X5 and is not derived from it, and that the writer is only that
              one place, citing HANDLER-1.
              Decision and World are not written; only read-only references.

    the code declares its own imprecision, verbatim:
      "Main bucket＝obligationId==0 全体であり、idle／bootstrap 等の non-recovery
       publish も計数する（R23 §10 の既知混在）。cmt（main-site rebuild のみ）との
       対応は R28 で相関させる。"
      i.e. the Main bucket is the whole obligationId==0 population, so non-recovery
      publishes such as idle and bootstrap are counted too.  This is a known mixing
      condition, self-declared, with correlation deferred to R28.

(b) origin plumbing
    files     RuntimePublicationOrchestrator.cpp, PublicationExecutor.h / .cpp,
              AudioEngine.h
    symbols   trySubmitImpl, PublicationExecutor::publish, publishImpl,
              rollbackDSPHandleRegistration  (two overloads),
              enqueueRuntimePublicationFireAndForget
    fact      a std::uint64_t recoveryObligationId parameter, defaulted to 0, is
              threaded from trySubmitImpl's req through publish / publishImpl into
              intent.payload.publish.recoveryObligationId.  No new struct field is
              introduced.  Because the parameter defaults to 0, existing callers
              (idle, bootstrap and similar) are unchanged.
              the comment states the origin is preserved across deferred resubmit
              because the original req is moved out by consume().

    the one behavioural change to an existing value, verbatim from the diff:
      -  intent.payload.publish.recoveryObligationId = 0;   // ★ D105-R5-8:
             Route B (non-recovery) ⇒ no obligation
      +  intent.payload.publish.recoveryObligationId = recoveryObligationId;

    and the comment immediately records that the old invariant is void:
      "旧コメント「Route B (non-recovery) ⇒ no obligation」は本変更により無効：
       Route B intent も obligation を搬送し、onPublishCommitted が async 解決し得る。
       sync 側（Orchestrator.cpp:321）は後続で no-op になる
       （CAS single-winner・D105-R5-9）。"

    i.e. the previously documented D105-R5-8 invariant is explicitly declared
    invalid by the change, and the replacement behaviour is named, including that
    onPublishCommitted may now resolve a Route-B intent asynchronously and that the
    synchronous Route-A completion at Orchestrator.cpp:321 becomes a no-op under
    CAS single-winner (D105-R5-9).

record    P1-5-IR-P2_Step5AI_P3-5-R27_publish_origin_plumbing_implementation_build_gate.md
          whose six-point pre-implementation verification includes the old state:
            "6. 旧: Route-B intent は id=0 のため async completion は recovery を
                 resolve しない  → 一致"
          so the pre-state was verified, and the invalidation is recorded rather
          than silent.
design    P1-5-IR-P2_Step5AH_P3-5-R26_publish_origin_disambiguation_design_audit.md
          P1-5-IR-P2_Step5AG_P3-5-R25_coordinator_take_counter_implementation_build_gate.md
follow-on P1-5-IR-P2_Step5AJ_P3-5-R28_origin_aware_publish_measurement_c1_c2_closure.md
          P1-5-IR-P2_Step5AK_P3-5-R29_recovery_origin_vehicle_feasibility_design_audit.md
          P1-5-IR-P2_Step5AL_P3-5-R30_recovery_origin_vehicle_implementation_build_gate.md
```

### 3.6 Contract-surface contact, classified without evaluation

For each of the eleven surfaces the Owner named. "touched" means the change reads or
writes state belonging to that surface. No judgement is attached.

| surface | R16 | R19 | R22 | R10 | R27 |
| --- | --- | --- | --- | --- | --- |
| publication | | | | read | **write** — `intent.payload.publish.recoveryObligationId` |
| rebuild | **write** | **write** | **write** | | |
| intent | | | | read | **write** — `processIntent` |
| retire | | | | | read — `onPublishCommitted` named |
| RT / non-RT boundary | | | | read — `acquireReadToken` | read — same |
| authority | | | | **read** — `worldAuthority_` | |
| ownership | **read** — post-transfer point | | | | **write** — origin classification |
| epoch | | | | read — `publicationSequence` | |
| queue | | | | | read — coordinator pop |
| shutdown | | | | | |
| invariants | | cites R18 §6/§11 | | G4 / G5 ownership gates | **invalidates D105-R5-8** |

The shutdown surface is not touched by any of the five items. That is a statement about
the diff, not a statement about shutdown correctness.

---

## 4. N2-C — existing-record coverage

```text
classification for all five items:   DIRECTLY COVERED
no item is "recorded only indirectly"
no item is absent from the record
```

Coverage measured by identifier frequency across the 245 work113 records:

| identifier | records | primary record |
| --- | --- | --- |
| `recoveryObligationId` | 49 hits | Step5AI (R27) |
| `publicationSequence` | 41 | Step5BD |
| `D105` | 38 | Step5AK (R29) |
| `runtimePublicationBridge_` | 27 | Step5AV |
| `coordinatorTakeCount_` | 21 | Step5AI |
| `processIntent` | 20 | Step5AW |
| `trySubmitImpl` | 19 | Step5AI |
| `rebuildTakeCount_` | 9 | Step5X (R16) |
| `onPublishCommitted` | 9 | Step5AK |
| `structuralHash` | 11 | Step5O |
| `getRebuildCommitEnqueueCount` | 6 | Step5AD (R22) |
| `getRebuildBuildResultCount` | 4 | Step5AA (R19) |
| `getActiveRuntimeSnapshot` | 6 | Step5R (R10) |
| `getCoordinatorTakeCount` | 6 | Step5AG (R25) |
| `rebuildBuildResultCount_` | 6 | Step5AA (R19) |
| `rebuildCommitEnqueueCount_` | 5 | Step5AD (R22) |

Changed files referenced by name: `ISRRuntimePublicationCoordinator` 17 records,
`PublicationExecutor` 11, `RuntimePublicationOrchestrator` 11, `AudioEngine.RebuildDispatch` 11.

One symbol has no dedicated record:

```text
rollbackDSPHandleRegistration   1 hit, and that hit is this audit's own predecessor
                                inventory, not a work113 record
```

It is nevertheless covered, transitively and correctly: the only thing the change does to
it is add a defaulted parameter and pass it to `enqueueRuntimePublicationFireAndForget`,
which is the R27 subject and appears in 7 records including Step5AI. Recorded here as an
observation, not as a gap.

---

## 5. N2-D — authority relationship

**The working-tree `ConvoPeq.md` was not used as source authority in this audit.** Every
statement above is derived from git objects at `e565019d^` and `e565019d`, from the
`★`-marked source text itself, and from the work113 records.

The two are kept separate, as required:

```text
source authority for this audit
    the git blobs at e565019d^ and e565019d, and nothing else

adjudicated authority, untouched
    G0/G1, 5,535,334 B, E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
    content identity   ACCEPTED / unchanged
    provenance         OPEN / NOT PROVEN   (N1)

current working-tree snapshot, NOT an authority
    5,550,764 B, C587352201F0BE3D8F39D30396137A663E35A7412EAA749FBE591D661E8A7DFE
    not promoted, not re-pinned, not substituted
```

### 5.1 One observation that bears on N1 and does not resolve it

While establishing coverage, the size values that work113 records attach to
`ConvoPeq.md` were counted. **Fifteen distinct sizes are cited**, rising monotonically
over the record chronology:

```text
  5,203,789   5,269,690   5,276,742   5,359,772   5,361,988
  5,414,041   5,431,515   5,464,930   5,485,382   5,485,818
  5,488,742   5,521,528   5,525,709   5,535,334   5,550,764
             ^ the LF form of the adjudicated authority
                             ^ cited 12 times, the most-cited value in the corpus
```

Two of these are the working-tree snapshot as the P3-5 records saw it:

```text
Step5AG / Step5AH / Step5AI   5,521,528 B   stamp 2026-09-23 21:32:56
Step5AJ / Step5AK             5,525,709 B   stamp 2026-09-23 22:41:29
Step5AL                       regenerated 2026-09-23 23:04:35, size not restated
current                       5,550,764 B   stamp 2026-09-27 00:36:31
```

This is consistent with N1 and adds to it: the snapshot was regenerated many times, its
identity changed each time, and the adjudicated authority is the fourteenth of fifteen
cited values with no record tying it to a specific source state.

**N1 remains OPEN. Nothing here resolves it, and no reconstruction, re-pin or promotion
was performed or is proposed.**

---

## 6. What the audit does not establish

```text
that the changes are correct          not assessed, out of scope by instruction
that the changes are safe              not assessed
that the build gates passed            not re-run; build and test are prohibited here
that the binaries correspond           not verified; no build, no execution
that D105-R5-9 behaves as the comment says    requires runtime, not authorised
that the R23 §10 known mixing is tolerable    not assessed
that the invalidated D105-R5-8 invariant has an acceptable replacement  not assessed
that the five items are independent of one another    not assessed
```

The classification in §3.6 says which contract surface each change touches. It does not
say whether touching that surface is acceptable.

---

## 7. Prohibitions observed

```text
production source modification   0     build                        0
test execution                   0     runtime execution            0
CDB                              0     debuggee                     0
g / go / gc                      0     R1                           0
RR-1..RR-18 re-evaluation         0     CR-3                         0
Phase 2-B                        0     Contract Amendment           0
Amendment 1 modification         0     ConvoPeq.md regeneration     0
ConvoPeq.md re-pin                0     authority replacement        0
git write operation              0     commit / checkout / reset / restore   0
reconstruction of 968377BE       0
```

Git was used only through read-only queries: `rev-parse`, `diff`, `numstat`, `show`,
`log`, `rev-list`, `merge-base`, `ls-tree`. No repository state was altered.

The only file created by this audit is this record.

---

## 8. New OPEN candidate

```text
candidate   N2-OPEN-1
title       the D105-R5-8 invalidation has no recorded acceptance
fact        the R27 change declares the invariant "Route B (non-recovery) ⇒ no
            obligation" void, and names the replacement: onPublishCommitted may
            resolve a Route-B intent asynchronously, and the synchronous Route-A
            completion at Orchestrator.cpp:321 becomes a no-op under CAS
            single-winner (D105-R5-9).
            the invalidation is recorded in the source comment and the pre-state was
            verified in Step5AI.  what is not present in the corpus is a record whose
            subject is the ACCEPTANCE of the replacement behaviour.
status      OPEN CANDIDATE.  not adjudicated here.  for the Owner.
```

Two smaller observations, recorded without being raised as items:

```text
the R27 counter declares its own imprecision (R23 §10 known mixing: the Main bucket
is the whole obligationId==0 population, so idle and bootstrap publishes are counted).
The declaration is in the source.  Correlation is deferred to R28 by its own text.

the three rebuild counters are documented as process-lifetime cumulative with no
reset and as differentially operated, with the explicit warning not to re-identify
queuedCount with taken.  Whether any consumer does so is not assessed here.
```

---

## 9. Result

```text
N2 status               the change set is five recorded, separately build-gated P3-5
                        items, captured by e565019d from an uncommitted working tree
exact diff identity     8 files, +155 / -12, blob identities in section 2
declaration match       exact on all three of file count, added and deleted
coverage                DIRECT for all five items
new OPEN candidate      1   (N2-OPEN-1, D105-R5-8 acceptance)
evaluation made         none
subjects modified       0
gate entered            none
```

### 9.1 The two axes, held apart

```text
N1   authority provenance            OPEN / NOT PROVEN    unchanged by this audit
N2   e565019d production-source set  DOCUMENTED          this record

N2 completing does not close N1.
N1's gap does not make the N2 changes invalid.
The two are independent problems and are recorded as such.
```

### 9.2 Returned to the Owner

```text
N1   OPEN, authority identity unchanged
N2   read-only audit complete
all other gates   STOP
```

The next judgement is the Owner's. This record does not propose a step, and no step was
taken from it.
