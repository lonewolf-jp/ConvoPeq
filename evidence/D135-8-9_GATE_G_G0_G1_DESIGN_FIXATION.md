# D135-8/9 Gate G — G-0 / G-1 Design Fixation (read-only, no implementation)

**Status:** DESIGN FIXED (G-0 PASS, G-1 design resolved). **Not yet implemented — G-2+ implementation window requires explicit go.**
**Date:** 2026-08-30
**Scope:** read-only. `production source 0 / test source 0 / build 0 / CTest 0`; documentation = evidence only.
**Base:** HEAD `5f6f48c` (F6), git clean. Primary reference = regenerated `ConvoPeq.md` (`Generated: 2026-08-30 19:19:29`). stale index excluded.
**Design authority:** `doc/work88/I1_DESIGN_REVIEW.md` (2026-08-15, R1-R17 コード突合: R1/R2/R3 未実装, blind overwrite cpp:905-915 = NO-GO pattern P0-7).
**Gate-G boundary (unchanged):** D/E/F の authority-boundary 維持のため、G は **Recovery logical obligation の identity/conservation 修正のみ**。`DSPTransition`/`CrossfadeAuthority`/`DSPLifetimeManager`/`ISRRetireRouter`/EBR/`deferredSlot_`/`processDeferredAdmission`/`publishRetryReady`/`deferredClearRequested_` は**変更対象外**。

---

# G-0 — Source re-baseline (read-only, verified)

## G-0.1 HEAD / ConvoPeg.md consistency
- `git log -1` = `5f6f48c 2026-08-30 19:30:11 commit`。`git status --short | grep -v '^??'` = 空 (tracked 0 変更)。= Gate F 基準 5f6f48c と一致。✓
- `ConvoPeq.md` mtime 19:19:37 `Generated: 2026-08-30 19:19:29` (regerated). 一次参照 OK。✓

## G-0.2 7-symbol re-extraction (current working tree line numbers)
| symbol | 現行 location |
|---|---|
| `submitRecoveryRequest` | decl `ISRRuntimePublicationCoordinator.h:423`; impl `cpp:819-942` |
| `recoveryIntentQueue_` | `h:887`; `kRecoveryIntentQueueCapacity = 256` (`h:886`); `LockFreeRingBuffer<RecoveryIntent,256>` |
| `pendingRecoveryAdmission_` | `h:927` (single slot, `PendingRecoveryAdmission`, plain struct) |
| `recoveryAdmissions_` | `h:934`; `RecoveryAdmissionTable<kMaxLogicalRecoveryObligations>` (`h:338-412`) |
| `resolveRecoveryObligation` | decl `h:430`; impl `cpp:944-980` |
| `currentBuildSnapshot_` | `AudioEngine.h:4898`; set `AudioEngine.Commit.cpp:800`; getter `AudioEngine.h:4459` |
| `RecoveryIntent` | `h:218-238` (handle/epoch/intentId/obligationId/buildSource; trivially copyable) |

**Line-number drift vs I1 (2026-08-15):** I1 cites `cpp:905-915` (blind overwrite) / `h:637`(queue) / `h:655`(PendingAdmission) / `cpp:988`(redrive scan). Current (post-F6 5f6f48c) = `cpp:923-941` (blind overwrite) / `h:887` / `h:927` / `cpp:1040` (redrive). **Semantics unchanged; cite current lines.**
(Side note: the `submitRecoveryIntent(handle, currentBuildSnapshot_)` wiring is `AudioEngine.h:4895`, QuarantineIntentHandler → submitRecoveryRequest.)

## G-0.3 D105-R3 counterexample — current-code re-confirmation (HOLDS)

Counterexample: **S quarantined @ snapshot A → reclaim → S re-quarantined @ snapshot B.**

```
(a) S@A: QuarantineIntentHandler → submitRecoveryIntent(S, currentBuildSnapshot_=A)
    → submitRecoveryRequest(S, A) cpp:819
    cid_A = (handle=S, target=rebuildFingerprint{A}) cpp:836-841
    findByKey(cid_A) cpp:866 → none → tryInsert(cid_A) cpp:880 → O1 (Live, liveCount=1)
    → push transport cpp:909 (or durable 923-941), delivery set
(b) S reclaim: reclaimNormal/requestReclaim (cpp:684 / AudioEngine.Retire.cpp:117)
    → only `reclaimInFlightCount_` (onReclaimBegin/End cpp:211-227) + epoch-safe reclaim
    → **does NOT touch recoveryAdmissions_ (no resolveRecoveryObligation call)** ⇒ O1 stays Live
(c) S@B: submitRecoveryRequest(S, B)
    cid_B = (handle=S, target=rebuildFingerprint{B}) — target B ≠ target A
    findByKey(cid_B) cpp:866 → O1.identity=(S,targetA) ≠ cid_B → none
    tryInsert(cid_B) cpp:880 → O2 (Live)  ⇒ **liveCount_ = 2 (O1 still Live)** → O_max ≥ 2 CONFIRMED
```
- **O_max ≥ 2 mechanism confirmed:** reclaim does not resolve/close the recovery obligation → a re-quarantined handle yields a second Live obligation without closing the first (capacity guard = 32 only).
- **Blind overwrite confirmed:** durable fallback `cpp:923-941`. Guard `cpp:923-924` (`pendingRecoveryAdmission_.recoveryObligationId != oblId`) defers DISTINCT obligations (delivery=None, cpp:925-928) — so distinct-obligation clobber is avoided. **The blind part is `cpp:929-941`:** when `recoveryObligationId == oblId` (same id, e.g. same coalesce key after a zombie survives) **or** slot empty → it **unconditionally overwrites** `pendingRecoveryAdmission_` with `intent.intentId`/`buildSource`/`epoch` — no *provenance*/in-flight-Building protection. I1/P0-7 NO-GO pattern still present. ✓ Held.

**G-0 verdict:** baseline matches; counterexample HOLDS; capacity-vs-cardinality separation confirmed (transport 256 h:886 ≠ logical cap 32 h:325). No source/test/build/CTest changed.

---

# G-1 — Recovery Identity design fixation (resolves I1 §104-108 open decisions)

## G-1.1 LogicalRecoveryIdentity (R1)
```
struct LogicalRecoveryIdentity {
    DSPHandle             handle;      // quarantined recovery target handle
    int                   generation;  // BUILD generation — epoch-consistent (NOT intentId; R1/R9 /* R9 */)
    SemanticRecoveryTarget target;     // semantic build identity: irIdentityHash + convolutionConfigHash + dspParameterHash (+ sampleRate/channelCount)
    // provenance = separate (R1: admission source は含めない)
    bool operator==(o) = handle== && generation== && target==;
};
```
- **Generation rule (resolves I1 §104-1):** `generation = buildSource.generation` (the build generation, set `AudioEngine.RebuildDispatch.cpp:1037-1039` `recoverySnapshot.generation = recoveryGeneration` from `rebuildRequestGeneration`; `RuntimeBuildTypes.h:50`). **NOT `intent.intentId`** (which current code uses as `recoveryGeneration=intentId` at cpp:932 — an identity-mismatch I1 flagged; fixed here). This makes generation epoch-consistent (build-generation domain), satisfying R9 (generation-domain mismatch ⇒ cannot coalesce).
- **Semantic build identity source (resolves I1 §104-1):** `target{ irIdentityHash = buildSource.rebuildFingerprint.irIdentityHash, convolutionConfigHash = rebuildFingerprint.convolutionConfigHash, dspParameterHash = rebuildFingerprint.dspParameterHash }` (cpp:838-840; set from `RuntimeBuildSnapshot.rebuildFingerprint`, RebuildDispatch.cpp:96-100). Optionally add `sampleRate`/`channelCount` from `buildInput` (`RuntimeBuildTypes.h:64`). This composes the identity deterministically from `buildSource` (`RuntimeBuildTypes.h:49-54`).
- **Equality excludes provenance** (only handle+generation+target), so a Transport re-drive and a Durable re-drive of the same logical recovery have **equal identity** → coalesce to 1 obligation.

## G-1.2 RecoveryProvenance (R2)
```
enum class RecoveryProvenance : uint8_t { Quarantine, Transport, Durable, Retry, Superseded };
```
- **Separate from identity:** per-obligation delivery/lifecycle marker (aligns with current `ObligationDeliveryState` h:281-292: None/Transport/Durable + retry/superseded). Equality of `LogicalRecoveryIdentity` never consults provenance ⇒ provenance can change (retry) while identity is stable (ΔL=0).

## G-1.3 SupersessionDecision + canSupersede (R3) — Compatibility ≠ Supersession
```
enum class SupersessionDecision : uint8_t {
    CanSupersede,            // same handle + strict generation increase + isSemanticSuperset → old Superseded, new admitted
    DifferentHandle,         // different handle → distinct obligation
    DifferentSemanticTarget, // different (ir/conv/dspParam) target → distinct obligation
    NotSameGenerationDomain, // generation domain mismatch (build generation not comparable) → cannot coalesce (R9)
    NotSuperset              // generation increased but NOT semantic superset → distinct obligation (bounded table)
};
[[nodiscard]] SupersessionDecision canSupersede(const LogicalRecoveryIdentity& newer,
                                                const LogicalRecoveryIdentity& older) noexcept;
```
- **isSemanticSuperset policy (resolves I1 §104-3, EQ vs IR change distinction):** definition —
  - `different.handle  → DifferentHandle`
  - else `different.target  → DifferentSemanticTarget`
  - else (same handle, same target) → if `newer.generation != older.generation` → treat as **retry/coalesce** (same logical obligation, refresh), NOT supersession (same target = compatible, ΔL=0)
  - `newer.generation <= older.generation` or generation domains differ → NotSameGenerationDomain
  - To make supersession distinct: `newer.generation == older.generation` but `newer.target != older.target` on **EQ-change-only** (convolutionConfigHash differs, irIdentityHash same) → **CanSupersede** (config refresh supersedes older target; IR unchanged). **IR change (irIdentityHash differs) → DifferentSemanticTarget** (a genuinely different target ⇒ distinct obligation, NOT supersession).
- **Explicit:** "generation 増加 ≠ semantic superset" — a larger generation with a genuinely different IR target is a **distinct** obligation (coalesce fails → bounded table), not silent overwrite.

## G-1.4 Mapping to current types (no new type churn — augment existing)
| current type | role under G-1 |
|---|---|
| `CoalesceIdentity` (h:262-269) | superseded by → `LogicalRecoveryIdentity` (adds `generation`; keeps handle+target). Keep `CoalesceIdentity` as an alias or fold in. |
| `SemanticRecoveryTarget` (h:248-257) | becomes the `target` component (optionally + sampleRate/channelCount). |
| `LogicalRecoveryObligation` (h:310-324) + `RecoveryAdmissionTable` (h:338-412) | `identity` field type → `LogicalRecoveryIdentity`; add `provenance` marker; add `supersession` transition. `findByKey`/`tryInsert` unchanged shape (equality now includes generation). |
| `ObligationState` (h:271-279) | add `ResolvedSuperseded` (Phase-II supersession terminal — G-4.4). |

---

# G-2 — Blind overwrite prohibition design (P0-7 fix)

## G-2.1 Replace submitRecoveryRequest durable fallback (cpp:923-941) with a 3-way identity decision
Current NO-GO: unconditional `pendingRecoveryAdmission_ = intent` when same-id or empty (cpp:929-941).
Proposed (binding, replaces 923-941):
```cpp
// after transport push fails (cpp:909-912)
const auto newIdent = makeLogicalRecoveryIdentity(quarantinedHandle, buildSource, epoch);
const auto* pending = &pendingRecoveryAdmission_;
// 1) empty durable slot → insert (new durable delivery)
if (pending.state == NoAdmission) { /* cpp:930-941 insert with newIdent */ return true; }
// 2) same logical identity → COALESCE (ΔL=0): refresh this obligation's durable info,
//    but NEVER clobber a Building lease — if state==Building → defer (delivery=None), return true.
//    (resolves P0-7: no in-flight clobber)
if (makeLogicalRecoveryIdentity(pending) == newIdent) {
    if (pending.state == Building) { slot.delivery=None; defer-count++; return true; }
    /* refresh durable metadata (latest-wins ONLY for the SAME logical identity, and not Building) */
    return true;
}
// 3) different identity → determine compatibility/supersession
switch (canSupersede(newIdent, makeLogicalRecoveryIdentity(pending))) {
  case CanSupersede:
     resolveRecoveryObligation(pending.recoveryObligationId, RecoveryOutcome::Superseded);  // old → Superseded (explicit)
     /* insert newIdent as durable */  return true;
  case DifferentHandle: case DifferentSemanticTarget: case NotSameGenerationDomain: case NotSuperset:
     // distinct obligation → do NOT overwrite. If bounded durable table empty slot → insert (I-3);
     // else → DEFER this obligation (delivery=None, stays Live, re-drive when slot frees).
     // **NEVER drop / NEVER overwrite a distinct Live obligation (I1 §1.2.3: "coalesce できないから捨てる禁止").**
     slot.delivery = None; recoveryRetryDeferredCount_++; return true;
}
```

## G-2.2 Compatibility vs Supersession separation enforced
- **Compatible (same identity) → coalesce / refresh** (ΔL=0), with Building-lease protection.
- **Compatible-but-distinct (different target/handle/generation) → distinct obligation** (bounded table; if full → defer, never drop).
- **Supersedes (CanSupersede) → old→Superseded + new admitted** (explicit `resolveRecoveryObligation(old, Superseded)`), never "overwrite-and-forget".

---

# G-3 — O_max / E_max / logical capacity redefinition

| quantity | current | note |
|---|---|---|
| transport capacity (`kRecoveryIntentQueueCapacity`) | **256** (h:886) | keep (DO NOT shrink to 32 — G-3 prohibition) |
| logical obligation capacity (`kMaxLogicalRecoveryObligations`) | **32** (h:325) | liveCount_ ≤ 32; single +1/-1 (h:368/390) |
| durable capacity | **1** (h:927 single `pendingRecoveryAdmission_`) | → bounded durable table (I-3) |
| E_max | 256 | transport residency |
| O_max | **≥2 (D105-R3, G-0.3)** | after G-1 identity + G-2 supersession + reclaim-close, O_max bounded to ≤32 |

**E_max × O_max ≤ 32 (no queue-shrink):** satisfied by the *single-representation* invariant — after G-2/G-4, each Live logical obligation holds **≤1 delivery representation (Transport XOR Durable)**. Thus concurrent transport residency is bounded by the number of Live obligations (≤ O_max ≤ 32), independent of the 256 queue depth. The 256 queue is FIFO storage, not an ownership-count ledger. **Capacity ≠ cardinality** (matches I4 `pendingReclaimHandles_` container-bound vs retired-world cardinality separation). O_max itself is the distinct-obligation count, bounded by 32.

---

# G-4 — Implementation order (fixed; to be executed in a later implementation window)

| step | change | file | invariant |
|---|---|---|---|
| G-4.1 R1+R2 | add `LogicalRecoveryIdentity` + `RecoveryProvenance` types (generation=buildSource.generation; target from rebuildFingerprint) | `ISRRuntimePublicationCoordinator.h` | no semantic change; existing flow unchanged |
| G-4.2 R1 wiring | switch `CoalesceIdentity`→`LogicalRecoveryIdentity` in `RecoveryAdmissionTable`: identity field, `findByKey`/`tryInsert` (`h:344/354`) | `h` | equality now includes generation; R9 satisfied |
| G-4.3 R3 | add `SupersessionDecision` enum + `canSupersede()` (+`isSemanticSuperset`) | `h` + `.cpp` (or inline) | no semantic change yet |
| G-4.4 supersession | add `ObligationState::ResolvedSuperseded`; `resolveRecoveryObligation` Superseded outcome (`cpp:944-980`); **old→Superseded, new→admitted**; never overwrite-and-forget | `.cpp` + `h` | R15 (non-superseded 削除しない); R5 exactly-one-per-identity |
| G-4.5 blind overwrite | replace `submitRecoveryRequest` durable fallback (`cpp:923-941`) w/ G-2.1 3-way decision (coalesce-lease-protect / distinct-defer / supersede) | `.cpp` | P0-7 解消; R14 (ΔL=0 on coalesce/retry) |
| G-4.6 bounded durable table (I-3) | replace single `pendingRecoveryAdmission_` (`h:927`) with `std::array<..., kMaxDurableRecoveryAdmissions>` lease table (base = `kMaxLogicalRecoveryObligations` 32); extend `take`/`settle` (`cpp:927,968`) | `.cpp` + `h` | INV-X1-5/6 (1 admission=1 reservation; durable≠double-count); CanSupersede==false held |
| G-4.7 reclaim-close | reclaim (`reclaimNormal` cpp:684 / `requestReclaim` AudioEngine.Retire.cpp:117) → for the reclaimed handle, resolve its matching Live recovery obligation → `Superseded`/terminal (prevents zombie O_max≥2 for D105-R3) | `.cpp` + `AudioEngine.Retire.cpp` | R15; O_max ≤ 32; D105-R3 counterexample closed |
| G-4.8 test + I-5 | invariant test (`logicalAdmissionCount==1` + D105-R3 regression) → build + CTest | `tests/…` | I-5 rollback point |

> G-4.8 (build/CTest) is the **D2 I-5 rollback point** — commit boundary: any step before I-5 can `git reset 5f6f48c` to restore. Each G-4.x step committed independently.

**Removed from G scope (unchanged):** `DSPTransition` / `CrossfadeAuthority` / `DSPLifetimeManager` / `ISRRetireRouter` / EBR / `deferredSlot_` / `processDeferredAdmission` / `publishRetryReady` / `deferredClearRequested_` — none touched. G is confined to `ISRRuntimePublicationCoordinator.{h,cpp}` (+ `AudioEngine.Retire.cpp` reclaim-close + tests).

---

# G-5 — Test-case matrix (to be written in implementation window)

| # | case | expected | invariant |
|---|---|---|---|
| T1 | 同一 identity 2回 submit | obligation = 1 (coalesce ΔL=0) | R5 |
| T2 | 異なる target 2回 submit | obligation = 2 | O_max/distinct |
| T3 | 同一 handle + snapshot A/B | 別 obligation (distinct target) | R8 / D105-R3 |
| T4 | coalesce + transport overflow | logical obligation 消失なし (durable fallback) | INV-X1-2 |
| T5 | durable overwrite (distinct live obligation) | 消失なし (defer, not overwrite) | P0-7 |
| T6 | supersession | old→Superseded, new→admitted (explicit transition) | R15 |
| T7 | resolve 二重実行 | liveCount_ 二重 decrement しない (idempotent) | liveCount ±1 |
| T8 | shutdown 中 submit | 新規 obligation 生成なし | shutdown gate cpp:823-828 |
| T9 | stale generation | reject (R9 / NotSameGenerationDomain) | R9 |
| T10 | double consume (of one obligation's delivery) | exactly one succeeds | R10-3 single-representation |
| T11 | **D105-R3 regression**: S@A → reclaim → S@B | obligation = 2 (O1,Superseded/terminal, O2 admitted) — no zombie O_max≥2 | **G-0.3 counterexample** |

---

# G-6 — Gate-G pass conditions mapping (post-implementation)

| Gate-G condition | source |
|---|---|
| G-1 LogicalRecoveryIdentity 存在 | G-1.1 |
| G-2 RecoveryProvenance 分離 | G-1.2 |
| G-3 Compatibility≠Supersession | G-1.3 |
| G-4 blind overwrite = 0 | G-2.1 |
| G-5 same-obligation coalesce authority側 | G-2.1 (case 2) |
| G-6 distinct target → distinct obligation | G-1.3 (DifferentSemanticTarget) |
| G-7 resolveRecoveryObligation 唯一 terminal | cpp:944-980 single authority (existing) |
| G-8 liveCount_ +1/-1 conservation | h:368/390 (existing; validate post I-3) |
| G-9 transport cap ≠ logical cap | G-3 (256 vs 32) |
| G-10 shutdown 中新規 obligation なし | cpp:823-828 (existing; keep) |
| G-11 stale generation rejection | G-1.1 R9 |
| G-12 double-consume rejection | R10-3 / T10 |
| G-13 D105-R3 regression PASS | G-4.7 + T11 |
| G-14 build PASS | I-5 |
| G-15 CTest PASS | I-5 |

---

## G-0/G-1 verdict

**Baseline (G-0) PASS; Design (G-1) FIXED.** D105-R3 counterexample re-confirmed as live (O_max≥2 zombie + cpp:929-941 blind overwrite). The design is now concretely defined: `LogicalRecoveryIdentity`(handle + buildSource.generation + semantic target, provenance excluded), `RecoveryProvenance`(separate enum), `SupersessionDecision`/`canSupersede`(Compatibility≠Supersession, EQ-change vs IR-change distinction), replacing the single-slot durable blind overwrite with a 3-way (coalesce-lease-protect / distinct-defer / supersede) decision, and bounding the recovery logical obligation layer to O_max≤32 without shrinking transport (256).

**Read-only: NO source/test/build/CTest performed.** Implementation (G-4.1→G-4.8) is **deferred to an explicit G-2+ implementation window** per the user's sequencing ("設計確定後にのみ実装 window G-2 へ進むこと"). Changed-scope = `ISRRuntimePublicationCoordinator.{h,cpp}` + `AudioEngine.Retire.cpp` + tests only; Gate-F authority boundary (DSPTransition/CrossfadeAuthority/DSPLifetimeManager/ISRRetireRouter/EBR/deferred) preserved.
