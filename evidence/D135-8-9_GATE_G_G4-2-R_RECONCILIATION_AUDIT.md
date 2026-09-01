# D135-8/9 Gate G-4.2-R — Reconciliation Audit (read-only)

**Status:** **CONDITIONAL** (Phase-I decision semantics maintained; RecoveryEpisodeId reintroduced into production storage, contradicting R23's "design-only / Phase-II deferred" — resolve before G-4.3).
**Date:** 2026-08-30
**Scope:** read-only. production source 0 changes. No G-4.3 / coalesce / durable-table / admission rewrite / canSupersede / supersession / reservation / reclaim/shutdown / tests.
**Base:** HEAD `5f6f48c` + G-4.1/G-4.2 implemented. Authorities: latest ConvoPeq.md, `I4_DESIGN_CONTRACT.md`, `evidence/D105-R23_OBLIGATION_TABLE_MODEL_CONVERGENCE.md`, G-4.1/G-4.2 reports.

## Primary question
Is G-4.2's EpisodeId / Generation / Target wiring semantically consistent with the **latest Phase-I obligation-table contract (D105-R23)**?

## Latest Phase-I contract (D105-R23, authoritative)
- **D18.7 (R23-5):** Phase-I production `CoalesceIdentity = { quarantinedHandle, SemanticRecoveryTarget }` — **RecoveryEpisodeId 成分なし・Phase-II deferred**.
- **R23 table (lines 16-17):** `RecoveryEpisodeId` (type) = **0 production hits** (design-only); `nextRecoveryEpisodeId_` (counter) = **0 production hits** (design-only).
- D13 / D13.1-2 (7-domain, epoch≠episode) = **Phase-II deferred**.
- D18.1 `{handle, episodeId, target}` **REWRITTEN** as D18.7 `{handle, target}`.
- Phase-I production identity/closure = the obligation-table model (`liveLogicalObligationCount == 0` ≙ all obligations gone), NO episode decomposition.

## R1-R6 audit

### R1 — Contract consistency → **CONDITIONAL**
Three of the four G-4.2 wiring elements are R23-consistent; the episodeId element is not:
| G-4.2 element | R23 phase-I stance | G-4.2 actual | consistent? |
|---|---|---|---|
| `RecoveryGeneration` allocator (`nextRecoveryGeneration_`) + `intent.recoveryGeneration` (≠ intentId / ≠ buildSource.generation) | D16 ordinal (Phase-I OK) | implemented, dedicated | ✅ |
| `SemanticRecoveryTarget` 6-field (ir/conv/dspParam/domainCoverage/convolverFingerprint/buildInputHash) + `computeBuildInputHash`/`computeDomainCoverage` | D12.2 (Phase-I OK) | implemented | ✅ |
| `RecoveryEpisodeId` (`episodeId` fields, `nextEpisodeId_` counter, threading) | **design-only / 0 production hits / Phase-II deferred** | **production hits** | ❌ **deviation** |

### R2 — No episode resurrection → **PASS (in decisions)**
`RecoveryEpisodeId` is **NOT** used in any Phase-I production **decision**:
- coalesce identity: `findByKey` compares `slots_[i].identity == key` where `identity` = `CoalesceIdentity = {quarantinedHandle, SemanticRecoveryTarget}` (h:309-315) — **episodeId NOT in the key**.
- capacity invariant: `liveCount_` +1/-1 (tryInsert/resolve) — episodeId not involved.
- closure: `resolveRecoveryObligation(id, ...)` single authority — id-based, not episode.
- supersession: inactive in Phase-I (containment equality-only).
- admission gate: shutdown gate + capacity — not episode.
→ episodeId is **stored/threaded only** (inert metadata), never a decision input.

### R3 — Generation separation → **PASS**
`intentId` (diagnostic event seq) / `RecoveryGeneration` (dedicated `nextRecoveryGeneration_` ordinal) / `BuildGeneration` (`RuntimeBuildSnapshot::generation`, untouched) — three distinct domains. grep `recoveryGeneration = intent.intentId` = 0; grep write-to-buildSource.generation-from-recoveryGeneration = 0. `nextRecoveryGeneration_` is a dedicated counter (h:469). ✅

### R4 — Target correctness → **PASS**
`SemanticRecoveryTarget` built in `submitRecoveryRequest` (cpp:~888): `{ ir-identityHash (rebuildFingerprint.irIdentityHash), convolutionConfigHash, dspParameterHash, computeDomainCoverage(buildSource), buildSource.convolverFingerprint, computeBuildInputHash(buildInput) }`. `buildInputHash` = FNV-1a over BuildInput fields (canonical, not placeholder/id/generation). `domainCoverage` = necessary-condition metadata (non-default domain → bit; not a supersession gate). ✅

### R5 — Behavioral freeze → **PASS**
`liveCount` / delivery / reservation / blind-overwrite / retry(K=4) / terminal-disposition semantics unchanged (git diff removed decision-logic lines = 0; only field/comment/threading additions). Coalesce-key refined to D12.2 6-field target (intended D18.7 alignment), no supersession/reservation change. ✅

### R6 — Identity contract (MOST IMPORTANT) → **PASS**
Phase-I `CoalesceIdentity = { handle, SemanticRecoveryTarget }` **maintained**: struct (h:309-315) has only `quarantinedHandle` + `target`, NO episodeId member; `findByKey`/`tryInsert` use it. The episode-including `LogicalRecoveryIdentity` (h:297-302) is **defined but UNUSED** in production decisions (no producer/consumer; coalesce uses `CoalesceIdentity`). ✅

## Verdict: CONDITIONAL

**Maintained:** R2 (no episode in decisions), R3 (generation separation), R4 (target correctness), R5 (behavioral freeze), R6 (CoalesceIdentity = {handle,target}). The Phase-I **decision** model matches D105-R23 D18.7.

**Deviation (must resolve before G-4.3):** G-4.2 gave `RecoveryEpisodeId` **production hits** (~10 sites: `LogicalRecoveryObligation::episodeId`, `RecoveryAdmissionTable::nextEpisodeId_`, `RecoveryIntent::episodeId`, `PendingRecoveryAdmission::episodeId`, allocation `++nextEpisodeId_` at h:419, threading `intent.episodeId = slot.episodeId` cpp:959/1143, durable-pending episodeId writes, `take` threading, plus the unused `LogicalRecoveryIdentity` type). R23 explicitly requires `RecoveryEpisodeId` = **0 production hits / design-only / Phase-II deferred**. This reintroduces the **old D18 episode abstraction into Phase-I production storage** — exactly the D18/R23 contract mix you warned against.

## Required resolution before G-4.3 (read-only audit → next implement window)
Remove the `RecoveryEpisodeId` production wiring so it returns to R23's 0-production-hit / design-only state:
1. Drop `episodeId` from `LogicalRecoveryObligation`, `RecoveryIntent`, `PendingRecoveryAdmission`.
2. Remove `nextEpisodeId_` counter + `slots_[i].episodeId = ++nextEpisodeId_` allocation (h:416-419).
3. Remove episodeId threading (`intent.episodeId = ...` cpp:959/1143, durable `episodeId = intent.episodeId`, `take` threading).
4. Either remove `LogicalRecoveryIdentity` (episode-including) or redefine it without `episodeId` (Phase-I identity = `CoalesceIdentity = {handle,target}`); it is currently unused.
> Keep (R23-consistent, do NOT revert): `RecoveryGeneration` dedicated ordinal + separation from intentId/buildSource.generation; `SemanticRecoveryTarget` 6-field + real `buildInputHash` + metadata-only `domainCoverage`.

After this removal, G-4.3 (Phase-I coalesce = `CoalesceIdentity {handle,target}` equality → existing Live obligation → COALESCE, ΔL=0 / reservation=0) can proceed against a purely R23-consistent model.

## STOP
G-4.2-R complete (read-only). **No G-4.3 / no code change in this audit.** Await instruction on the required episode-wiring removal (G-4.2-R follow-up) before G-4.3.
