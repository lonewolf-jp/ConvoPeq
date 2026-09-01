# D135-8/9 Gate G-4.2-RF Audit — read-only re-audit

**Status:** **PASS** (R1-R8). **STOP.**
**Type:** read-only re-audit. **Production source changes: 0** (this audit itself changed nothing; the tracked diff shown is G-4.2-RF's implementation, unchanged by this audit).
**Base:** HEAD `5f6f48c` + G-4.1/G-4.2/G-4.2-RF. Authorities: D105-R23 (Phase-I convergence), I4_DESIGN_CONTRACT, latest ConvoPeq.md.

## R1-R8

| R | check | result | evidence |
|---|---|---|---|
| R1 | EpisodeId 完全除去 | ✅ | `grep RecoveryEpisodeId\|nextEpisodeId_\|nextRecoveryEpisodeId_\|episodeId` production code = **0**（唯一のヒットはコメント「★ D105-R23 (phase-I): RecoveryEpisodeId ... Phase-II deferred — design-only here.」= 許容）。episode allocator/storage/threading = 0 |
| R2 | Phase-I identity = {handle, target} | ✅ | `CoalesceIdentity { quarantinedHandle, SemanticRecoveryTarget }` (h:295-302), `operator==` = handle+target（episode 成分なし）。`findByKey` (`slots_[i].identity == key`) / `tryInsert` (`slots_[i].identity = key`) / submit-cid すべて episode 不在 |
| R3 | RecoveryGeneration separation | ✅ | `grep recoveryGeneration = intent.intentId` = 0; `grep buildSource.generation = recoveryGeneration` = 0; dedicated `nextRecoveryGeneration_` (h:452) + `slots_[i].recoveryGeneration = ++nextRecoveryGeneration_` (h:404); intentId untouched |
| R4 | SemanticRecoveryTarget 6-field | ✅ | `{ irIdentityHash, convolutionConfigHash, dspParameterHash, domainCoverage, convolverFingerprint, buildInputHash }` (h:273-290); `operator==` = 5 semantic 値; `buildInputHash` = FNV-1a (`computeBuildInputHash`, real — not intentId/generation); `domainCoverage` = metadata only (not consulted by findByKey/capacity/admission/supersession) |
| R5 | Snapshot drift distinctness (episode removal で誤統合なし) | ✅ | `SemanticRecoveryTarget::operator==` compares 5 semantic values; `findByKey` = `{handle,target}` equality. **same handle+same target → existing obligation (coalesce); same handle+different target (drift → different hash/fingerprint) → distinct CoalesceIdentity → distinct obligation; different handle → distinct** (D18.8 honored). No episode in the comparison ⇒ episode removal did NOT collapse distinct targets. |
| R6 | Capacity/closure = liveLogicalObligationCount ≤ 32 only | ✅ | `liveCount_` single +1/-1 (tryInsert/resolve), ≤ 32. No `E_max` / `O_max` / `E×O` / `EpisodeAdmissionState` / episode-level closure / episode CAS re-introduced. (Grep hits for "Closed" are unrelated: ShutdownPhase::RetireClosed, RecoveryAdmissionClosed, lastClosedSnapshot, readerRegistrationClosed.) |
| R7 | Behavioral freeze (tryInsert decision unchanged) | ✅ | `tryInsert` capacity gate (`liveCount_.load >= kCapacity → reject`) + non-Live slot reuse **unchanged**; only added `slots_[i].recoveryGeneration = ++nextRecoveryGeneration_;` (a value store, not a condition). resolve/liveCount_/delivery/K=4 retry/terminal/blind-overwrite/reservation semantics unchanged. |
| R8 | Diff boundary | ✅ | tracked = `ISRRuntimePublicationCoordinator.h`(57±) + `.cpp`(73±) only (G-4.2-RF). **This read-only audit made 0 changes.** No tests/CMake/reclaim/shutdown/publish/AudioEngine.h. |

## Verdict: PASS
D105-R23 Phase-I model **actually restored**:
- `CoalesceIdentity = { quarantinedHandle, SemanticRecoveryTarget }` (episode absent).
- `RecoveryEpisodeId` / episode counter / episode-level closure & capacity → Phase-II deferred (0 production refs).
- `RecoveryGeneration` = dedicated lineage ordinal (≠ intentId, ≠ BuildGeneration); `nextRecoveryGeneration_` preserved.
- `SemanticRecoveryTarget` 6-field (real buildInputHash, metadata-only domainCoverage) preserved.
- `liveLogicalObligationCount ≤ 32` = single Phase-I capacity invariant (no episode decomposition).

No episode production-resurrection, no identity contamination, no generation conflation, no behavioral change.

## STOP
G-4.2-RF Audit = **PASS (R1-R8)**. **G-4.3 not started.** Per the pipeline `G-4.2 → G-4.2-R CONDITIONAL → G-4.2-RF removal PASS → G-4.2-RF Audit PASS → G-4.3`. Await the G-4.3 Phase-I Coalesce Admission instruction. Nothing beyond read-only audits was performed; no coalesce/durable-table/blind-overwrite/canSupersede/supersession/stalled/retry-redesign/publish-wiring/reclaim-shutdown/tests were changed.
