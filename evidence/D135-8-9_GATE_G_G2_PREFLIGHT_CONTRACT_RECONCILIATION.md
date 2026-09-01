# D135-8/9 Gate G-2 — Implementation Preflight / Contract Reconciliation (read-only)

**Status:** PASS (with G-1 design corrections resolved to I4 D12/D18). **No implementation yet — G-4.1 authorization requires this PASS + explicit go.**
**Date:** 2026-08-30
**Scope:** read-only. `source 0 / test 0 / build 0 / CTest 0`; evidence only.
**Base:** HEAD `5f6f48c` (F6), git clean; primary ref regenerated `ConvoPeq.md` (`Generated: 2026-08-30 19:19:29`).
**Authorities:** `doc/work88/I4_DESIGN_CONTRACT.md` (D12/D13/D14/D18 normative) · `evidence/D105-R24_I4_POST_CLOSURE_VERIFICATION.md` · `evidence/D105-R25_PHASE_II_EPISODE_MIGRATION_AUDIT.md` · `evidence/D105-R3_RECOVERY_EPISODE_TARGET_MULTIPLICITY_REAUDIT.md`.

> **Preamble — the user's G-2 warning is correct and is the centerpiece.** My G-1 supersession rule ("same handle + same target + generation differs → retry/coalesce" and "same generation + EQ change + IR unchanged → CanSupersede") is **inconsistent with I4 D12/D18**, which the user says must be normative. G-2 confirms this and re-fixes the design. I4 D12.1 explicitly rejects domain-based supersession (A={IR=B}, B={IR=C} same domain {IR} ⇒ B does NOT supersede A), and D18 says Phase-I containment = all-values-equal ⇒ **Phase-I supersession is structurally INACTIVE**.

---

## G-2.1 — Current source re-extraction (complete)

| # | element | current location |
|---|---|---|
| 1 | `LogicalRecoveryObligation` | `h:310-324` (id/identity[CoalesceIdentity]/state/handle/epoch/intentId/buildSource/delivery/count) |
| 2 | `RecoveryAdmissionTable` | `h:338-412` (findByKey :344, tryInsert :354, resolve :383, liveCount :398, nextId_ :411) |
| 3 | `ObligationState` | `h:271-279` (NoObligation/Live/ResolvedSuccess/ResolvedFailed/ResolvedStaleSuperseded/ResolvedRetry/ShutdownDiscarded) |
| 4 | `ObligationDeliveryState` | `h:288-292` (None/Transport/Durable) |
| 5 | `PendingRecoveryAdmission` | `h:911-926` (State{NoAdmission/DurablePending/Building}, pending, recoveryGeneration, buildSource, reservationOwned, handle, epoch, intentId, recoveryObligationId) |
| 6 | `RecoveryIntent` | `h:218-238` (handle/epoch/intentId/obligationId/buildSource) |
| 7 | `submitRecoveryRequest` | `cpp:819-942` |
| 8 | `resolveRecoveryObligation` | `cpp:944-980` (single authority; Retry early-return; terminal map) |
| 9 | `settlePendingRecoveryAdmission` | `cpp:1167-1178` (retry→Building→DurablePending; false→clear) |
| 10 | `redriveDeferredRecoveryObligations` | `cpp:1042-1052` (+ `redriveDeferredRecovery` :1058-1114) |
| 11 | `markTransientFailure` | `cpp:998-1024` (Live→Live, delivery=None, count++, K=4→resolve(Failed); ΔL=0 except exhaustion) |
| 12 | `rearmRecoveryRetry` | `cpp:1028-1035` (only Building+same-id → settle(true)) |
| 13 | `takePendingRecoveryAdmission` | `cpp:1124-1140` (DurablePending→Building lease) |
| 14 | `recoveryAdmissions_` mutations | `tryInsert` h:354 (+1), `resolve` h:383 (−1), `markTransientFailure` cpp:1010/1013/1019, `redriveDeferredRecovery` cpp:1100/1107 |
| 15 | `pendingRecoveryAdmission_` r/w | cpp:930-938, 1089-1098, 1138, 1157, 1171-1176 |
| 16 | `recoveryIntentQueue_` producer/consumer | producer `submitRecoveryRequest` cpp:909 & `redriveDeferredRecovery` cpp:1106; consumer `popRecoveryRequest` cpp:1181-1185 |
| 17 | `recoveryObligationId` generation | `nextId_` h:411 (monotonic, ++ at h:360), per-slot id store h:361; `nextRecoveryIntentId_` h:888 (diagnostic) |
| 18 | `currentBuildSnapshot_`→`buildSource` | `AudioEngine.h:4898` set `Commit.cpp:800`; QuarantineIntentHandler `submitRecoveryIntent(handle, currentBuildSnapshot_)` `h:4895`; fingerprint→target `cpp:836-841` |
| 19 | reclaim handle lifecycle | `requestReclaim` `AudioEngine.Retire.cpp:117` → `reclaimNormal` `cpp:684` (epoch-safe; does NOT touch `recoveryAdmissions_`) |

**Confirmed:** `RecoveryEpisodeId` is **ABSENT from code** (grep = 0) — it is a D13 design concept, Phase-II migration per D105-R25. `ObligationState` does NOT have `ResolvedSuperseded`. SemanticRecoveryTarget (h:248-257) has only 3 hashes (missing domainCoverage/convolverFingerprint/buildInputHash from I4 D12.2).

---

## G-2.2 — Identity / Provenance / Delivery / State responsibility separation

### A. Identity (PASS + corrections)
- **`generation` from `buildSource.generation` availability:** YES — `RuntimeBuildSnapshot.generation` (`RuntimeBuildTypes.h:50`), set at RebuildDispatch.cpp:93/1037-1039 from `rebuildRequestGeneration`; epoch-consistent. **BUT D18 says `generation` must NOT be in the identity key** (CoalesceIdentity = {handle, RecoveryEpisodeId, SemanticRecoveryTarget}). → **correction: drop `generation` from identity.** Generation is used only in `canSupersede(..., isAfter)` (D16) as a *newer* comparison, not in equality.
- **`intentId` used as generation:** `pendingRecoveryAdmission_.recoveryGeneration = intent.intentId` at **cpp:932** and **cpp:1092** — the R1/R9 mismatch I1 flagged. **Correction: recoveryGeneration = build generation** (epoch-consistent), NOT intentId.
- **semantic target deterministic from buildSource:** current target = rebuildFingerprint.{irIdentityHash, convolutionConfigHash, dspParameterHash} (cpp:838-840). I4 D12.2 needs 6 fields (+ `domainCoverage`, `convolverFingerprint`, `buildInputHash`). All derivable from `RuntimeBuildSnapshot` (h:52 `convolverFingerprint`; rebuildFingerprint has the rest; domainCoverage from baseline diff). → **correction: extend SemanticRecoveryTarget to D12.2 6-field**. Deterministic. ✓
- **sampleRate/channelCount needed?** **NO** — D12.2 uses build hashes, not sampleRate/channelCount. Drop the G-1 suggestion to add them. (They're inside buildInputHash if needed.)

### B. Provenance / Delivery / State (PASS — suppress redundant enum)
Existing model already separates:
| responsibility | existing type | values |
|---|---|---|
| Delivery placement | `ObligationDeliveryState` (h:288) | None / Transport / Durable |
| Lifecycle state | `ObligationState` (h:271) | NoObligation / Live / Resolved{Success,Failed,StaleSuperseded,Retry} / ShutdownDiscarded |
| Retry budget | `consecutiveFailureCount` (h:323) → K=4 exhaustion → ResolvedFailed (cpp:1017-1019) |

A standalone `RecoveryProvenance` enum (my G-1 R2) would **overlap exactly** with `ObligationState`+`ObligationDeliveryState`+`consecutiveFailureCount`. **Verdict: do NOT introduce a new `RecoveryProvenance` enum.** Keep provenance as a *derived* view (lifecycle state + delivery + retry count). **G-1 correction: remove the redundant type.**

> **Net G-1 corrections from G-2.2:** (1) identity = {handle, RecoveryEpisodeId, SemanticRecoveryTarget} (NO generation); (2) extend SemanticRecoveryTarget to D12.2 6-field; (3) fix recoveryGeneration=buildSource.generation (not intentId); (4) suppress standalone RecoveryProvenance; (5) drop sampleRate/channelCount idea.

---

## G-2.3 — 最重要: I4 D12 vs D-135 G-1 supersession — diff table + normative decision

### Diff table
| aspect | **I4 D12/D18 (NORMATIVE)** | D135 G-1 (earlier, WRONG) | reconcile |
|---|---|---|---|
| `canSupersede(newer,older)` | same handle **+ same RecoveryEpisodeId + newer RecoveryGeneration (isAfter) + isSemanticSuperset(target)** | same handle + (target via EQ-change) | **adopt I4 D12.4** |
| identity includes generation? | **NO** (D18: {handle, episode, target}) | YES (generation in identity) | **remove generation** |
| containment `isSemanticTargetSuperset` | Phase I = **all-semantic-values-equal** (D12.2; conservative) | compositional (EQ-change ⇒ supersede) | **all-values-equal** |
| Phase-I supersession activity | **STRUCTURALLY INACTIVE** (differing target ⇒ NOT supersede ⇒ retain both; D18) | active (EQ-change supersedes) | **inactive** |
| `RecoveryEpisodeId` | **required** (lineage gate; D13, assigned at episode start, NOT epoch) | absent | **add (required)** |
| user counterexample T17 (A={IR=B,EQ=1}, B={IR=C,EQ=1}, same episode+gen) | B does **NOT** supersede A | B supersedes (my EQ rule) | **B NOT supersede** |
| lineage check uses epoch? | **NO** (D12.5/D13.2: epoch removed; use RecoveryEpisodeId) | — | **use RecoveryEpisodeId** |

### Normative decision: **I4 D12/D18 WINS.**
Reject the G-1 "EQ-change-only supersession". The falsifying argument (D12.1, mirrored by the user): "the same domain is being changed" ≠ "the newer logically contains the older". Different IR value (IR=B vs IR=C) proves neither containment nor supersession; only **all-values-equal** is decidable in Phase I.

### Corrected contract (normative, binds G-4)
```
canSupersede(newer, older) =
      sameHandle(newer, older)                → DifferentHandle
   &  sameEpisode(newer.episodeId, older.episodeId)   → DifferentEpisode (RecoveryEpisodeId)
   &  newer.RecoveryGeneration isAfter older.RecoveryGeneration   → NotNewer
   &  isSemanticSuperset(newer.target, older.target)
        = isDomainSuperset(.) && isSemanticTargetSuperset(.)       → NotSemanticSuperset
```
where `isSemanticTargetSuperset(n,o) = n.ir==o.ir && n.conv==o.conv && n.convFp==o.convFp && n.dsp==o.dsp && n.buildInputHash==o.buildInputHash` (D12.2). `isDomainSuperset = (o.domainCoverage & ~n.domainCoverage)==0`.

### Practical consequence for G-4.4
Since Phase I containment = full-equality ⇒ ANY target difference fails containment ⇒ **Phase I supersession NEVER fires**. Obsolete-duplicate absorption = **COALESCE** (identical {handle, episode, target}), not supersession. `ObligationState::ResolvedSuperseded` is added as a type but is **unreachable in Phase I** (analogous to dormant RetryExhaustedDiscard/D-6). Supersession (partial-order) = **Phase-II extension** (deferred). **This makes G-4.4 minimal:** it must NOT implement an active EQ-change supersession; it only adds the (dormant) type + a defensive "never destroy a non-superseded obligation" guard.

> **Result: no logical-obligation disappearance from a false supersession is possible** — the containment rule structurally blocks it. This directly resolves the user's I4-Phase-I-NO-GO root cause.

---

## G-2.4 — Capacity proof (reservation-first D14) — PASS with restatement

**Current conservation model (verified):**
- `liveCount_` (h:410, +1 @ tryInsert:368 post-coalesce, −1 @ resolve:390) = **number of Live obligations = reservation count**. This is the reservation-first budget.
- Each Live obligation has **exactly 1 delivery placement** `delivery ∈ {None, Transport, Durable}` (h:318). `Building` is a sub-state of the Durable placement (PendingRecoveryAdmission.State: DurablePending→Building, cpp:1138) — it does NOT create a second reservation.
- `reservationOwned` (h:921/934/1094, INV-X1-5) = 1 admission = 1 reservation, set on durable admission; reservation is invariant across placement changes (transport→durable→building does not touch liveCount_).

**Corrected statement (replaces the too-narrow "Transport XOR Durable"):**
```
liveLogicalObligationCount (= liveCount_)  ≤ 32                       (single +1/-1, INV-CAP)
sum of placements                        = transportCount + durableCount(=incl. Building) + deferredCount(delivery=None)
                                        = liveCount_                    (each Live obligation has exactly 1 delivery placement)
⇒ transportCount + durableCount + buildingCount(⊂durable) + stalledCount(none in current) ≤ liveCount_ ≤ 32
```
This satisfies D14: `transportCount + durableCount + buildingCount + stalledCount ≤ kMaxLogicalRecoveryObligations`. My earlier "Transport XOR Durable" omitted the deferred (None) placements, which DO consume reservations (a deferred obligation holds a reservation with no storage representation) — so it under-counted. **The correct bound is liveCount_ (reservation), not the delivery enum.**

**Not conflating capacity vs cardinality:** `pendingIntentCount_` (transport queue residency, cpp:908/1105) is a **separate 256-capacity container** (queue fill), NOT a reservation. A single obligation MAY push multiple transport intents (coalesced resubmission re-push, cpp:895-902), so pendingIntentCount_ can reach 256 while liveCount_ ≤ 32. D14.1's "single bounded resource is not single" → here: the **reservation** budget (liveCount_ ≤ 32) is the logical bound; the transport FIFO (256) is a container. **Capacity ≠ cardinality** (consistent with G-3/F-4). G-4.6 will enforce the single-representation invariant so an obligation's transport residency ≤ 1 (or keep pendingIntentCount_ as a queue-residency metric separate from the reservation). Conclusion: **reservation-first model holds at the obligation layer (liveCount_); queue-shrink prohibited; both bounds kept.**

---

## G-2.5 — reclaim-close correspondence — PASS (conditional: episode-scoped, NOT handle-only)

- reclaim = `requestReclaim` (Retire.cpp:117) → `reclaimNormal` (cpp:684). Carries **handle only** (no obligation-id / episode-id).
- A handle can hold **MULTIPLE Live obligations** (different targets / episodes — precisely the R3 multi-obligation case). → **naive "same handle → resolve" is WRONG** (would terminal obligations of other concurrent episodes).
- **Correct pairing (D13/D13.2):** quarantine episode begins at quarantine-admission; `RecoveryEpisodeId` assigned there (immutable in episode); reclaim-close resolves Live obligations **matching (handle, RecoveryEpisodeId)** — i.e., the episode being torn down. Since one handle → one active episode at a time, this is effectively 1:1 for the current episode, but MUST be epische-scoped.
- **Implementation note for G-4.7:** quota upstream must tag the quarantine with `RecoveryEpisodeId` and thread it through `submitRecoveryRequest`/`submitRecoveryIntent`; reclaim (`requestReclaim`) must receive the episode-id (or a handle→episode map) to know which obligations to close. **Do NOT do handle-only resolution.**

---

## G-2.6 — Blind overwrite elimination proof (P0-7) — PASS with 1 fix site

All `pendingRecoveryAdmission_` mutations enumerated:
| site | kind | allowed? |
|---|---|---|
| cpp:930-938 (`submitRecoveryRequest` durable fallback) | overwrite fields w/ new intent | **blind-overwrite candidate — G-4.5 FIX** (must gate: NoAdmission-insert only / same-identity-coalesce / NEVER Building-clobber / NEVER distinct-identity) |
| cpp:1090-1098 (`redriveDeferredRecovery` durable re-arm) | guarded `state==NoAdmission` (1089) → empty-insert | **allowed** (guarded) |
| cpp:1138 (take DurablePending→Building) | lease transition | allowed |
| cpp:1171-1172 (settle retry Building→DurablePending) | lease rollback | allowed |
| cpp:1157/1176 (discard/settle clear) | termination clear | allowed |

**The single blind-overwrite site is cpp:930-938.** Current guard 923-924 defers a *distinct-identity* obligation (delivery=None, cpp:925-928) — so distinct-clobber is avoided. The residual risk: when `recoveryObligationId == oblId` (same identity) **or** slot empty, it overwrites **unconditionally** — including potentially overwriting a `Building` durable lease (the producer overwrite at 930 could clobber a consumer `take`→Building, because `pendingRecoveryAdmission_` is a plain struct with no lock between the two SPSC threads). → **G-4.5 rewrite:** 3-way per G-2 (empty-insert / same-identity-coalesce-with-Building-protection / explicit-supersede-or-defer). After G-4.5: **no different-identity overwrite, no Building clobber, no Live-non-supersedable overwrite.** Elimination proof = enumerate all 5 sites + classify (done above).

---

## G-2.7 — Finish-condition checklist (all met → G-4.1 authorized)

| condition | status |
|---|---|
| 最新 ConvoPeq 再抽出 | ✅ §G-2.1 (all 19 elements, current lines) |
| Identity field 最終形確定 | ✅ `{handle, RecoveryEpisodeId, SemanticRecoveryTarget}` (I4 D18), no generation |
| Provenance/Delivery/State 責務分離 | ✅ suppress redundant RecoveryProvenance; existing model sufficient |
| **I4 D12 vs G-1 supersession 差分解消** | ✅ **adopt I4 D12/D18 as normative** (reject EQ-change); supersession structurally inactive in Phase I |
| **RecoveryEpisodeId 要否確定** | ✅ **REQUIRED** (lineage gate; D13; add; not epoch) |
| **capacity 32 reservation proof** | ✅ liveCount_ reservation-first; sum-of-placements ≤ liveCount_ ≤ 32 (corrected §G-2.4) |
| **reclaim→obligation close 対応関係** | ✅ episode-scoped (handle+RecoveryEpisodeId), NOT handle-only |
| blind overwrite = 0 遷移表確定 | ✅ single fix site cpp:930-938; rest guarded (§G-2.6) |
| Building lease 保護 | ✅ G-4.5 adds Building-clobber guard |
| transient failure semantics 維持 | ✅ `markTransientFailure` untouched (K=4 exhaustion → Failed; ΔL=0 else) — NO redesign |
| shutdown semantics 維持 | ✅ `submitRecoveryRequest` cpp:823-828 shutdown gate; `discardPendingRecoveryAdmission` cpp:1152 — untouched |
| D/E/F authority boundary 維持 | ✅ G confined to `ISRRuntimePublicationCoordinator.{h,cpp}` + `AudioEngine.Retire.cpp` + tests; DSPTransition/CrossfadeAuthority/DSPLifetimeManager/ISRRetireRouter/EBR/deferred/publishRetryReady/deferredClearRequested_ untouched |

**12/12 met → G-2 PASS.**

---

## G-2 verdict

**G-2 PASS.** The reconciliation removed two real defects in my G-1 design (identified by the user's I4-D12 warning):
1. **Supersession rule corrected to I4 D12/D18**: `canSupersede = same handle + same RecoveryEpisodeId + newer RecoveryGeneration + isSemanticSuperset`; Phase-I containment = all-values-equal ⇒ **supersession structurally inactive** (obsolescence collapse = coalesce only). The G-1 "EQ-change-only supersession" is **rejected** (it would re-introduce the false-supersession logical-obligation disappearance I4 was written to prevent).
2. **Identity corrected**: {handle, RecoveryEpisodeId, SemanticRecoveryTarget}; generation removed from identity (used only in isAfter); `recoveryGeneration` fixed to buildSource.generation (not intentId); SemanticRecoveryTarget extended to D12.2 6-field; standalone RecoveryProvenance suppressed.

Also confirmed: reservation-first capacity = `liveCount_` (≤32), sum-of-placements ≤ liveCount_; reclaim-close must be episode-scoped; blind overwrite = exactly 1 site (cpp:930-938) to fix; `markTransientFailure` / shutdown semantics / D/E/F authority boundary all preserved.

**Read-only — no source/test/build/CTest.** **G-4.1 authorization is now granted by this PASS** (per the user's "G-2 が PASS したら次に G-4.1"), pending the user's explicit go to start implementing. Implementation windows G-4.1…G-4.8 proceed only on that go; each step stops for verification.
