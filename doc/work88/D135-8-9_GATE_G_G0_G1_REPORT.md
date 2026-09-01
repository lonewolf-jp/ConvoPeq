# D135-8/9 Gate G-0/G-1 — Work Report

**Gate:** G-0/G-1 (Recovery identity design fixation, read-only)
**Result:** G-0 PASS (baseline + counterexample re-confirm) / G-1 design FIXED. Implementation deferred.
**Base:** HEAD `5f6f48c` (F6); primary ref regenerated ConvoPeq.md (`Generated: 2026-08-30 19:19:29`); git clean.
**Scope:** read-only — **no source/test/build/CTest**. evidence only.

## G-0
- Baseline verified (5f6f48c / ConvoPeq.md 19:19:29 / tracked 0 changes). 7 symbols re-extracted at current line numbers (submitRecoveryRequest cpp:819-942 / recoveryIntentQueue_ h:887 (256) / pendingRecoveryAdmission_ h:927 (single) / recoveryAdmissions_ h:934 / resolveRecoveryObligation cpp:944-980 / currentBuildSnapshot_ AudioEngine.h:4898 / RecoveryIntent h:218-238).
- **I1 line drift:** I1(2026-08-15) cited cpp:905-915 / h:637 / h:655 / cpp:988; current = cpp:923-941 / h:887 / h:927 / cpp:1040. Semantics unchanged.
- **D105-R3 counterexample HOLDS:** S@A → reclaim (reclaimNormal/requestReclaim do NOT resolve recoveryAdmissions_) → S@B → tryInsert → O2 Live while O1 Live ⇒ **O_max ≥ 2**; blind durable overwrite at **cpp:923-941** (guard 923-924 protects distinct; the blind part = 929-941 unconditional same-id/empty overwrite, P0-7 NO-GO pattern present).

## G-1 design fix (resolves I1 §104-108 four open decisions)
- **LogicalRecoveryIdentity** = handle + `buildSource.generation` (build-generation, epoch-consistent — NOT intentId; fixes R1/R9) + SemanticRecoveryTarget{ir,conv,dspParam hash} (+ sampleRate/channelCount); provenance excluded from equality.
- **RecoveryProvenance** = enum {Quarantine, Transport, Durable, Retry, Superseded} — separate from identity (retry keeps identity stable, ΔL=0).
- **SupersessionDecision**/canSupersede — CanSupersede (same handle, EQ-config-change only, IR unchanged → supersede) / DifferentHandle / DifferentSemanticTarget (IR change → distinct obligation) / NotSameGenerationDomain / NotSuperset. Compatibility≠Supersession explicit.
- **Blind overwrite fix** — replace cpp:923-941 w/ 3-way: same-identity→coalesce(but NEVER clobber Building lease); different-identity→distinct(never overwrite; defer or bounded-table insert); CanSupersede→old-Superseded+new-admitted (explicit resolveRecoveryObligation(Superseded)).

## G-3/G-4/G-5
- E_max=256(same, no shrink) × O_max≤32 (post fix) ≤ 32 via single-representation invariant (Transport XOR Durable per Live obligation). Capacity(256/32/1)≠cardinality(liveCount_≤32).
- Implementation order G-4.1→G-4.8 (R1+R2 → R1 wiring → R3 → supersession → blind-overwrite fix → bounded durable table → reclaim-close → I-5 test+build). Changed scope: ISRRuntimePublicationCoordinator.{h,cpp} + AudioEngine.Retire.cpp + tests. Rollback point = D2 I-5 (build+CTest PASS, reset 5f6f48c).
- G-5 test matrix T1-T11 incl. D105-R3 regression (S@A→reclaim→S@B ⇒ O2 admitted, O1 terminal/Superseded).

## Authority boundary preserved
No change to DSPTransition / CrossfadeAuthority / DSPLifetimeManager / ISRRetireRouter / EBR / deferredSlot_ / processDeferredAdmission / publishRetryReady / deferredClearRequested_.

## Full evidence
`evidence/D135-8-9_GATE_G_G0_G1_DESIGN_FIXATION.md`

## Next
**G-2+ implementation window** (G-4.1→G-4.8) — requires explicit user go. NOT started here.
