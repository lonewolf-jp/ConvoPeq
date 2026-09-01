# D135-8/9 Gate G-4.3 Audit — Phase-I Coalesce / Obligation Linearization

**Status: CONDITIONAL** (linearization correct; A5 diagnostic-refresh residue)
**Type:** read-only audit. **Production source changes: 0.**
**STOP.**

## A1-A12

| A | check | result | evidence |
|---|---|---|---|
| A1 | Canonical CoalesceIdentity = {handle, SemanticRecoveryTarget} | ✅ | `CoalesceIdentity { quarantinedHandle, SemanticRecoveryTarget }` (h:295-302); `operator==` = handle+target. No EpisodeId/intentId/RecoveryGeneration/epoch in identity. `findByKey` (`slots_[i].identity == key`) + `submitRecoveryRequest` cid = `{quarantinedHandle, srt}` (6-field target). same{h,target}→candidate; same h+diff target → no; diff h → no. |
| A2 | Lookup ≠ authorization | ✅ | `findByKey(cid)` is a snapshot lookup only; **authorization = the CAS Live→Live re-validation** (cpp:927) which confirms Live atomically. Lookup alone does not authorize mutation (D29.2). |
| A3 | LIVE→LIVE CAS on same atomic as terminal | ✅ | Coalesce CAS `slot.state.compare_exchange_strong(expectLive, Live, acq_rel, acquire)` (cpp:927, `slot = recoveryAdmissions_.slot(existing)` → member `slots_[i].state`); Terminal CAS `slots_[i].state.compare_exchange_strong(expected, terminalState, acq_rel)` (h:428, `LogicalRecoveryObligation::state`, `std::atomic<ObligationState>`). **Same atomic object** on the same table. |
| A4 | Terminal race paths | ✅ | All LIVE→TERMINAL via `recoveryAdmissions_.resolve(id, terminal)` (h:428 CAS): `Published→ResolvedSuccess` (cpp:1031), `StaleSuperseded` (1032), `Failed` (1034, from `markTransientFailure` exhaustion cpp:1084), `ShutdownDiscarded` (1042). Supersession dormant (no ResolvedSuperseded produced in Phase-I). Coalesce CAS (cpp:927) == the SAME `state` → same race domain. |
| A5 | CAS-first mutation ordering | ⚠️ CONDITIONAL residue | Order correct (CAS cpp:927 → oblId read → intentId write → delivery check). **Post-CAS mutations = diagnostic `slot(existing).intentId = intent.intentId` only** (identity/state/liveCount_/delivery/buildSource/recoveryGeneration NOT written on coalesce). The intentId write is a monitoring counter (h:364) on a slot that is recycled on next tryInsert — **no lifecycle/identity/count/corruption**. Strictly, a concurrent `resolve → CAS Live→terminal` between the CAS-success and this write would land the intentId write after terminal (benign diagnostic). |
| A6 | buildSource backward-rewind (D29.5) | ✅ | Coalesce branch does **NOT** write `buildSource` (only intentId). `buildSource` is written only in the NEW path (tryInsert, cpp:958 `slot.buildSource = buildSource`). No stale-COALESCE-buildSource-rewind possible. RecoveryGeneration and RuntimeBuildSnapshot::generation remain distinct (G-4.2-RF). |
| A7 | Identity immutability on COALESCE | ✅ | Coalesce does not touch `identity`(CoalesceIdentity)/`quarantinedHandle`/`SemanticRecoveryTarget`/`id`/`RecoveryGeneration`. Reuses existing oblId (V8). |
| A8 | ΔL / capacity | ✅ | `tryInsert` liveCount_+1 (h:408, single authority, cap `>= kCapacity` → reject h:393); `resolve` liveCount_-1 (h:430). Coalesce does NOT call tryInsert → ΔL=0. `liveCount_` = single identical-capacity authority (h:450). capacity full + matching Live → findByKey→coalesce(no tryInsert); + no matching → tryInsert→reject. |
| A9 | Delivery duplication | ✅ | `wasDeferredBefore && slot(existing).delivery != None → return true` (redrive ordering, R10-3/C16) — no second Transport/Durable on coalesce. No durable-fallback redesign. |
| A10 | EpisodeId / generation contamination | ✅ | `RecoveryEpisodeId` prod ref = 0, `nextEpisodeId_` = 0; `recoveryGeneration = intent.intentId` = 0; `buildSource.generation = recoveryGeneration` = 0. |
| A11 | Supersession not introduced | ✅ | `canSupersede`/`isSemanticSuperset`/`isDomainSuperset`/`isSemanticTargetSuperset` — comments only; no production decision path. `ResolvedSuperseded` dormant (no transition). Equality-only coalesce. |
| A12 | Diff boundary | ✅ | `ISRRuntimePublicationCoordinator.cpp`(95+) + `.h`(57+); no tests/CMake/AudioEngine/reclaim/shutdown/publish/durable-fallback/reservation. |

## Focus findings (A3+A4+A5+A6)
- **A3 PASS**: coalesce CAS and terminal CAS are on the **same `slots_[i].state` atomic** — genuine D27.2 linearization.
- **A4 PASS**: every terminal transition (Success/Failed/Stale/Superseded→dormant/ShutdownDiscard) goes through the single `resolve()` CAS on that atomic; coalesce CAS races with it on the same object.
- **A6 PASS**: coalesce does not write buildSource → no stale-generation backward-rewind (D29.5).
- **A5 (the one residue)**: CAS precedes the (only) mutation, and that mutation is a **diagnostic `intentId` counter refresh** — it does NOT write identity/state/liveCount_/delivery/buildSource/RecoveryGeneration, so it cannot corrupt the obligation or double-count. Strictly, a concurrent terminalize between CAS-success and the intentId write would place the write after terminal; harmless (diagnostic field on a recycled slot).

## Verdict: CONDITIONAL

The D27.2 linearization required by the request is **correctly implemented** (A3 same-atomic CAS; A4 race with all terminal paths; A6 no buildSource rewind; CAS-first ordering). No correctness/lifecycle/count defect. **A5 has a single non-blocking residue**: the post-CAS `slot(existing).intentId = intent.intentId` diagnostic refresh is a write that, in a narrow window, could land after a concurrent Live→terminal (benign — a monitoring counter, not identity/state/count/delivery/buildSource/recoveryGeneration; the slot is recycled on the next tryInsert).

**Recommended follow-up (before PASS→tests):** remove the `intentId` refresh from the COALESCE branch (it is diagnostic-only; the obligation's original admission intentId suffices) so the coalesce path performs **no** post-CAS mutation — making it fully D27.2-clean. One-line removal; no other change. (Alternatively, explicitly classify the intentId write as an exempt telemetry update outside D27.2's lifecycle-mutation scope.)

**Not NO-GO:** the CAS is on the terminal CAS's state; no CAS-failure mutation; no terminal-after lifecycle mutation; no stale buildSource rewind; no distinct-target coalesce; coalesce never increments liveCount_; no EpisodeId; no supersession.

## STOP
G-4.3 Audit = CONDITIONAL (read-only, 0 changes). **No test addition / durable-fallback fix / G-4.4.** Await instruction on the one-line intentId-refresh removal (or its classification) → then re-audit → PASS → regression tests.
