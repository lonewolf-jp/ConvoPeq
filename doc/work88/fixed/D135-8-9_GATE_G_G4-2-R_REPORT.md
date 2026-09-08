# D135-8/9 Gate G-4.2-R — Work Report

**Gate:** G-4.2-R Reconciliation Audit (read-only). **CONDITIONAL.**
**Changes:** production source 0. No G-4.3 / coalesce / durable-table / admission rewrite / canSupersede / supersession / reservation / reclaim-shutdown / tests.
**Base:** HEAD `5f6f48c` + G-4.1/G-4.2. Authorities: D105-R23 (Phase-I convergence), I4_DESIGN_CONTRACT, REPAIR_PLAN2-dash2.

## Outcome
- **R6 maintained (PASS)** — Phase-I `CoalesceIdentity = {quarantinedHandle, SemanticRecoveryTarget}` (h:309-315, no episodeId; findByKey/tryInsert use it). `LogicalRecoveryIdentity` (episode-including) is defined but UNUSED in production decisions.
- **R2 PASS** — RecoveryEpisodeId NOT used in coalesce identity / capacity invariant / closure / supersession / admission gate (storage+threading only).
- **R3 PASS** — intentId / RecoveryGeneration (dedicated nextRecoveryGeneration_) / BuildGeneration three distinct domains.
- **R4 PASS** — SemanticRecoveryTarget 6-field real; buildInputHash = FNV-1a canonical; domainCoverage = metadata-only.
- **R5 PASS** — behavioral freeze (decision-logic 0 removed); only field/comment/threading additions.
- **R1 → CONDITIONAL** — RecoveryEpisodeId has **production hits (~10 sites)**, contradicting D105-R23's "0 production hits / design-only / Phase-II deferred". This is the old-D18 episode abstraction in Phase-I production storage.

## Primary deviation (resolve before G-4.3)
G-4.2 gave RecoveryEpisodeId production hits: `LogicalRecoveryObligation::episodeId`, `RecoveryAdmissionTable::nextEpisodeId_` (counter + `++nextEpisodeId_` h:419), `RecoveryIntent::episodeId`, `PendingRecoveryAdmission::episodeId`, threading (`intent.episodeId = slot.episodeId` cpp:959/1143, durable-pending write, `take` threading), plus unused `LogicalRecoveryIdentity`. D105-R23 explicitly requires 0 production hits / design-only.

## Required follow-up (next implement window, before G-4.3)
1. Remove `episodeId` from LogicalRecoveryObligation / RecoveryIntent / PendingRecoveryAdmission.
2. Remove `nextEpisodeId_` counter + `slots_[i].episodeId = ++nextEpisodeId_` allocation.
3. Remove all episodeId threading (submit/redrive/take/durable).
4. Remove or redefine `LogicalRecoveryIdentity` without episodeId (Phase-I identity = CoalesceIdentity {handle,target}).
> Keep (R23-consistent): RecoveryGeneration dedicated ordinal + intentId/buildSource.generation separation; SemanticRecoveryTarget 6-field + real buildInputHash + metadata-only domainCoverage.

## Full evidence
`evidence/D135-8-9_GATE_G_G4-2-R_RECONCILIATION_AUDIT.md`

## STOP
G-4.2-R done (read-only). No code change. Await instruction to remove the episode wiring (G-4.2-R follow-up) — then G-4.3 can proceed against a purely R23-consistent model.
