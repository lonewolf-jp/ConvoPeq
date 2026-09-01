# D135-8/9 Gate G-2 — Work Report

**Gate:** G-2 Implementation Preflight / Contract Reconciliation (read-only)
**Result:** PASS (12/12 finish conditions). **G-1 design corrected to I4 D12/D18 normative contract.**
**Base:** HEAD `5f6f48c`(F6); primary ref ConvoPeq.md `Generated: 2026-08-30 19:19:29`; git clean. No source/test/build/CTest.

## Critical outcome (user's I4-D12 warning validated)
My G-1 supersession rule ("same target + generation differs → retry/coalesce"; "same gen + EQ change + IR unchanged → CanSupersede") was **inconsistent with I4 D12/D18**. G-2 resolves it:
- **Normative = I4 D12/D18.** `canSupersede(newer,older) = same handle + same RecoveryEpisodeId + newer RecoveryGeneration (isAfter) + isSemanticSuperset`.
- **Phase-I containment = all-semantic-values-equal** (D12.2). Any target difference ⇒ NOT supersede ⇒ **retain both**. ⇒ **Phase-I supersession structurally INACTIVE** (obsolescence collapse = COALESCE only).
- G-1 "EQ-change-only supersession" **rejected** — D12.1/T17 shows A={IR=B}, B={IR=C} same domain ⇒ B does NOT supersede A.

## G-1 corrections (from G-2.2/2.3)
1. `LogicalRecoveryIdentity` = {handle, **RecoveryEpisodeId**, SemanticRecoveryTarget} — **generation removed** (D18; used only in isAfter).
2. Add **RecoveryEpisodeId** (required; D13; assigned at quarantine-episode start via counter; NOT epoch; absent in code → Phase-II migration per D105-R25).
3. Extend `SemanticRecoveryTarget` to D12.2 6-field (domainCoverage/convolverFingerprint/buildInputHash + 3 hashes); drop sampleRate/channelCount idea.
4. `recoveryGeneration` = buildSource.generation (fix R1/R9 intentId-misuse at cpp:932/1092), NOT intentId.
5. **Suppress standalone `RecoveryProvenance` enum** (redundant with existing ObligationState + ObligationDeliveryState + consecutiveFailureCount).
6. Add dormant `ObligationState::ResolvedSuperseded` (unreachable in Phase I, like RetryExhaustedDiscard).

## G-2.4 capacity (corrected)
Reservation-first = `liveCount_` (≤32, single +1/-1). Each Live obligation has exactly 1 delivery placement (None/Transport/Durable; Building⊂Durable). Sum-of-placements = liveCount_ ≤ 32. My earlier "Transport XOR Durable" under-counted (omitted deferred-None which also holds a reservation). `pendingIntentCount_` (transport 256) = separate container (capacity ≠ cardinality); queue-shrink prohibited.

## G-2.5 reclaim-close
Must be **episode-scoped** (handle + RecoveryEpisodeId), NOT handle-only (a handle can hold multiple Live obligations across episodes/targets — the R3 case). Reclaim must carry the episode-id.

## G-2.6 blind overwrite
Only 1 blind-overwrite site: **cpp:930-938** (submitRecoveryRequest durable fallback; overwrites same-id/empty unconditionally, risk Building-clobber). Others guarded (cpp:1089 NoAdmission-only; 1138/1171 lease; 1157/1176 clear). G-4.5 rewrites cpp:930-938 (3-way decision + Building protection). Elimination proof = all-5-site classification.

## Preserved (untouched)
`markTransientFailure` (K=4 exhaustion→Failed, ΔL=0) · shutdown semantics (`submitRecoveryRequest` cpp:823-828 / `discardPendingRecoveryAdmission` cpp:1152) · resolveRecoveryObligation single authority (cpp:944-980) · D/E/F authority boundary (DSPTransition/CrossfadeAuthority/DSPLifetimeManager/ISRRetireRouter/EBR/deferred/publishRetryReady/deferredClearRequested_).

## Full evidence
`evidence/D135-8-9_GATE_G_G2_PREFLIGHT_CONTRACT_RECONCILIATION.md`

## Next
**G-2 PASS → G-4.1 authorization** (per user's G-2.7). Await explicit go; then G-4.1 (R1+R2 types) → G-4.2 (identity wiring) → G-4.3 (canSupersede unit test) → G-4.4 → G-4.5 (P0-7) → G-4.6 → G-4.7 (D105-R3) → G-4.8 (build+CTest). Each step stops for verification. NOT started here.
