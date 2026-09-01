# D135-8/9 Gate G-4.2 — Identity Wiring / Generation Domain Separation（Work Report）

**Gate:** G-4.2. **STOP after this — no G-4.3 / coalesce / durable-table / admission-change.**
**Base:** HEAD `5f6f48c` + I4 D12/D18 (G-2 normative) + G-4.1 types.
**Scope (allowed):** RecoveryEpisodeId allocator+threading, dedicated RecoveryGeneration allocator, intentId separation, buildSource.generation preservation, SemanticRecoveryTarget value generation, buildInputHash real, domainCoverage wiring, LogicalRecoveryIdentity generation-prep.
**NOT touched:** coalesce, durable table, blind-overwrite, reservation, canSupersede, supersession, stalled/retry, publish wiring, reclaim/shutdown wiring, tests, CMake.
**Files changed:** `src/audioengine/ISRRuntimePublicationCoordinator.h` (75±) + `.cpp` (78±). Total 136+/17-. **Only these 2 files.**

## What was done

### .h
- `LogicalRecoveryObligation` += `RecoveryEpisodeId episodeId{0}` + `RecoveryGeneration recoveryGeneration{0}` (stable per obligation).
- `RecoveryAdmissionTable` += `nextEpisodeId_{1}` + `nextRecoveryGeneration_{1}` (single-writer monotonic; like `nextId_`). In `tryInsert`: `slots_[i].episodeId = ++nextEpisodeId_; slots_[i].recoveryGeneration = ++nextRecoveryGeneration_;` (allocate per NEW obligation; coalesce never calls tryInsert → episode/generation stable across coalesce).
- `RecoveryIntent` += `RecoveryGeneration recoveryGeneration{0}`.

### .cpp
- Added anonymous-namespace helpers (before `submitRecoveryRequest`):
  - `computeBuildInputHash(const convo::BuildInput&)` — canonical FNV-1a over BuildInput config values (deterministic; no padding). **Real value, not a placeholder/id/generation.**
  - `computeDomainCoverage(const convo::RuntimeBuildSnapshot&)` — necessary-condition metadata (non-default domain → bit set); **not** a supersession gate.
- `submitRecoveryRequest` cid construction: replaced 3-field `{ir,conv,dspParam}` with I4 D12.2 6-field `SemanticRecoveryTarget{ ir-identityHash, convolutionConfigHash, dspParameterHash, computeDomainCoverage(buildSource), buildSource.convolverFingerprint, computeBuildInputHash(buildSource.buildInput) }`.
- Threaded `intent.episodeId` / `intent.recoveryGeneration` from the obligation slot (after `intent.obligationId = oblId`), and into the durable fallback `pendingRecoveryAdmission_`. **`recoveryGeneration` is now `intent.recoveryGeneration` (dedicated ordinal), NOT `intent.intentId`** (both sites — submitRecoveryRequest + redrive — fixed; R1/R9 conflation removed).
- `redriveDeferredRecovery`: restores `intent.episodeId`/`intent.recoveryGeneration` from the obligation slot `s` (no re-allocation) + durable fallback.
- `takePendingRecoveryAdmission`: threads `intent.episodeId`/`intent.recoveryGeneration` from the durable slot.
- `buildSource.generation` (BuildGeneration) untouched — stays `RuntimeBuildSnapshot::generation`; **never assigned RecoveryGeneration** (two separate systems per D8.3/D18.5).

### Where episode starts
EpisodeId allocated at `tryInsert` (new obligation created). Coalesce reuses the slot's episodeId; redrive restores it ⇒ same obligation → same episodeId; distinct obligations → distinct episodeId (monotonic `nextEpisodeId_`). This is the G-4.2 "episode ≈ obligation" approximation: a single quarantine that would later produce multiple targets gets distinct episodeIds here (proper episode-sharing refined in later gates when coalesce/supersession land) — **no coalesce implemented in G-4.2**, as instructed. Quarantine lifecycle NOT re-designed (allocation lives in the coordinator table, not the quarantine handler).

## V1-V8 verification

| check | result | evidence |
|---|---|---|
| V1 same episode → same episodeId | ✅ structural | episodeId allocated once per obligation (tryInsert:419); coalesce/redrive restore from slot (no re-alloc) |
| V2 episode A≠B → distinct episodeId | ✅ | `nextEpisodeId_` monotonic (h:468), `++` per new obligation |
| V3 intentId ≠ RecoveryGeneration; dedicated counter | ✅ | grep `recoveryGeneration = intent.intentId` = 0; `nextRecoveryGeneration_{1}` h:469; intentId untouched |
| V4 RecoveryGeneration ≠ buildSource.generation | ✅ | grep write-to-buildSource.generation from recoveryGeneration = 0; buildSource.generation left as BuildGeneration |
| V5 target 6 fields real values | ✅ | cid = D12.2 6-field (ir/conv/dspParam/domainCoverage/convolverFingerprint/buildInputHash) w/ real `computeBuildInputHash` (FNV-1a) |
| V6 LogicalRecoveryIdentity = handle+episodeId+target (no generation) | ✅ | G-4.1 unchanged; operator== no recoveryGeneration |
| V7 behavioral freeze (admission semantics) | ✅ | decision-logic lines (tryInsert/resolve/fetchAdd/fetchSub/liveCount/delivery/State/return) — 0 removed in diff (only comment/added-field lines); capacity +1/-1, K=4 retry, terminal single-authority, delivery logic unchanged; coalesce-key refined to D18 D12.2 identity (intended, no supersession/reservation change) |
| V8 diff boundary | ✅ | only `ISRRuntimePublicationCoordinator.h` + `.cpp`; no tests/CMake/reclaim/shutdown/publish/AudioEngine.h touched |

## Notes
- No literal `\n` leak (0 in both files).
- `SemanticRecoveryTarget` field order kept original-3-first (G-4.1) so aggregate init `{a,b,c,d,e,f}` maps ir→dspParam then domainCoverage/convolverFingerprint/buildInputHash — matches construction.
- `markTransientFailure` / `resolveRecoveryObligation` / `settlePendingRecoveryAdmission` / `submitRecoveryRequest` decision logic — unchanged (V7).

## STOP
G-4.2 complete & verified. **No G-4.3 (coalesce), no durable-table, no canSupersede, no admission rewrite** — per instruction. Await the user's re-audit + next Gate instruction. This is a Work Report; source edits beyond the 2 files were not made.
