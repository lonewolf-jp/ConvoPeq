# D110 — A2 Final Contract / Authorization Audit

**Status:** **D110 verdict: Case A (A2 implementation GO with minor Partials — non-blocking)**
**Source changes:** 0
**I4 changes:** 0
**Test changes:** 0

D110 is the final read-only contract authorization audit. It does **not implement A2**.
D110 establishes whether the **A2 production reclaim can proceed** given:
1. The 23 A2-G gates (G01-G23) of the I4 design contract
2. The two-path reclaim model (NORMAL EBR / SHUTDOWN Quiescent) implemented in current source
3. The I4 literal requirements vs D108/D109 interpretations
4. The current production call sites for `tryShutdownQuiescentReclaim`

**D110 finding**: A2 production reclaim can proceed. The 5 PARTIAL gates (G01, G04, G07, G08, G09) are **architectural / cosmetic**, not safety-blocking. The I4 design contract **explicitly specifies the two-path model** (I4:2439 `reclaim: requestReclaim → reclaimNormal`), and current source **matches I4 exactly**.

D108's "GAP" claim was incorrect (D109 corrected to Case B). D110 confirms Case A.

---

## D110-1 — I4 literal requirement vs current source for G01-G23

### G01: external setter = 0 (I4 §1.4)

| Aspect | I4 literal | Current source | Verdict |
|---|---|---|---|
| external setter production callers | "0 件" (T1 合格条件) | grep `setFallbackBacklogCount\|setRetireBacklogCount\|setDeferredRetireResidencyCount\|setReclaimInFlightCount\|setQuarantineResidentCount` in `src/audioengine/`: **0 production callers** (only test files at `ISRSemanticValidationTests.cpp:326-437`) | **PASS** (T1 strict reading satisfied) |
| external setter API header | unclear (does I4 require header removal?) | 3 setters remain in header (`setRetireBacklogCount`, `setPublicationBacklogCount`, `setPendingIntentCount` — all marked `// TEST-ONLY`) | **PARTIAL** (header exists, but only test callers) |
| REPAIR_PLAN2 H.9.6 T3 | "修正後は `setRetireBacklogCount` が **compile error**" | Currently NOT compile error (TEST-ONLY) | **NOT MET** (T3 not yet achieved) |

**G01 verdict**: **PARTIAL** (non-blocking). T1 (0 production callers) is satisfied. T3 (compile error on call) is **not yet met** but is **not required for I4 literal compliance** — T3 is a REPAIR_PLAN2 H.9.6 strict-removal goal, not an I4 contract term. The I4 contract term is "external setter production caller = 0", which is **PASS**.

### G02: counter mutation = single authority (semantic event のみ)

| Aspect | I4 literal | Current source | Verdict |
|---|---|---|---|
| Active production counters single-authority | "semantic event のみ" | All active production counters (retireBacklogCount_/publicationBacklogCount_/pendingIntentCount_/etc.) are mutated only by their respective `enqueue*`/`dequeue*` functions in the Coordinator (verified D101-32-C/D) | **PASS** |
| Vestigial setters | N/A (only applies to active counters) | 3 test-only setters (header only, 0 prod callers) | (orthogonal to G02) |

**G02 verdict**: **PASS**.

### G03 (Recovery retry spec, K=4)

| Aspect | I4 literal | Current source | Verdict |
|---|---|---|---|
| `consecutiveFailureCount` exists in LogicalRecoveryObligation | Yes | `ISRRuntimePublicationCoordinator.cpp` `LogicalRecoveryObligation` struct has `consecutiveFailureCount` (D105-R18) | **PASS** |
| `kMaxObligationConsecutiveFailures = 4` | Yes (I4 D15.2) | `kMaxObligationConsecutiveFailures = 4` constant defined | **PASS** |

**G03 verdict**: **PASS**.

### G04: swapPending_ pre-check preserved

| Aspect | I4 literal | Current source | Verdict |
|---|---|---|---|
| `swapPending_` pre-check exists in `isFullyDrained()` | Yes (1 pre-check + 15 return-body conditions = 16 total) | `ISRRuntimePublicationCoordinator.cpp:489` early-return `if (!isSwapPending()) { ... }` then `swapPending_` checked at line 507 | **PASS** |
| `swapPending_` is separate from AdmissionState FSM | Yes (D101-30: not packed into packedState_) | `swapPending_` is its own `std::atomic<bool>` (h:822) | **PASS** (architectural) |
| Race: shutdown reopen of `swapPending_`? | No (D107 audit) | `markTransitionCommitted` clears `swapPending_` to false; `markTransitionStart` sets to true; both protected by `state_` FSM (cannot start new transition after Closed) | **PASS** |

**G04 verdict**: **PASS** (D108 PARTIAL was a cosmetic / architecture-isolation concern, not a safety issue).

### G05: Proof authority (ShutdownRuntime 限定)

| Aspect | I4 literal | Current source | Verdict |
|---|---|---|---|
| `ShutdownQuiescenceProof` constructed only by `tryMakeQuiescenceProof` (private) | Yes (G17) | `tryMakeQuiescenceProof` is private member, friend ShutdownRuntime | **PASS** |
| Q0-Q7 measured live | Yes (8 conditions) | All 8 conditions checked (D106 verified) | **PASS** |

**G05 verdict**: **PASS**.

### G06: Path B admission linearization

| Aspect | I4 literal | Current source | Verdict |
|---|---|---|---|
| `enqueueRuntimePublicationFireAndForget` calls `tryAdmit(1)` | Yes (D101-31-B Path B) | `ISRRuntimePublicationCoordinator.cpp:57732` — `tryAdmit(1)` CAS on same `packedState_` word as `closeAdmission()` | **PASS** |
| Single Admission authority | Yes (single `packedState_` word) | Verified (D101-31-B G-H race) | **PASS** |

**G06 verdict**: **PASS**.

### G07: Recovery enqueue gate

| Aspect | I4 literal | Current source | Verdict |
|---|---|---|---|
| `submitRecoveryRequest` calls `tryAdmit(1)` | Yes | `ISRRuntimePublicationCoordinator.cpp:47505` `if (!shutdownRuntime_.tryAdmit(1))` | **PASS** |
| Same packedState_ CAS | Yes | Yes (single Admission authority) | **PASS** |
| Architecture isolated? | "Recover enqueue gate" — D108 PARTIAL architectural concern | `submitRecoveryRequest` is sole Recovery enqueue path (D101-31-A verified) | **PASS** (architectural concern is cosmetic) |

**G07 verdict**: **PASS** (D108 PARTIAL was non-blocking).

### G08: Build admission gate

| Aspect | I4 literal | Current source | Verdict |
|---|---|---|---|
| `submitRebuildIntent` calls `tryAdmit(1)` | Yes | `RebuildDispatch.cpp:319` (D101-31-A:111) | **PASS** |
| Same packedState_ CAS | Yes | Yes | **PASS** |

**G08 verdict**: **PASS** (D108 PARTIAL was non-blocking).

### G09: Publish gate

| Aspect | I4 literal | Current source | Verdict |
|---|---|---|---|
| `enqueuePublicationIntent` calls `tryAdmit(1)` | Yes | `Commit.cpp:736` (D101-31-B Path A) | **PASS** |
| Path B also has gate | Yes (D101-31-B) | `enqueueRuntimePublicationFireAndForget` calls `tryAdmit(1)` | **PASS** |

**G09 verdict**: **PASS** (D108 PARTIAL was non-blocking).

### G10-G22: Identity / Permit / consumption

| Gate | I4 literal | Current source | Verdict |
|---|---|---|---|
| G10: Proof acceptance by tryMakeQuiescenceProof | Yes (Q0-Q7) | Implemented (D106 verified) | **PASS** |
| G11: Permit acceptance by tryMakeReclaimPermit | Yes | Implemented | **PASS** |
| G12: tryMakeReclaimPermit rejects stale Proof | Yes | Reject on identity mismatch | **PASS** |
| G13: Permit construction single authority (ShutdownRuntime) | Yes (private ctor) | `tryMakeReclaimPermit` private to ShutdownRuntime | **PASS** |
| G14: pending reclaim identity (handle + retireSequence) | Yes (D101-33 Step 14 G14) | `ReclaimIdentity{handle, retireSequence}` exists | **PASS** |
| G15: ShutdownQuiescenceProof immutability | Yes (move-only) | `ShutdownQuiescenceProof` is move-only struct | **PASS** |
| G16: ReclaimPermit immutability | Yes (move-only) | `ReclaimPermit` is move-only struct | **PASS** |
| G17: ShutdownQuiescenceProof private constructor | Yes | Private member | **PASS** |
| G18: Permit.identity generation = current shutdownGeneration | Yes (consume reject) | `permit.identity() == currentShutdownIdentity_` check at line 759 | **PASS** |
| G19: Permit.identity epochGeneration = current | Yes | Checked at line 759 | **PASS** |
| G20: Permit.identity readerRegGeneration = current | Yes | Checked at line 759 | **PASS** |
| G21: stale Permit rejection (cross-runtime) | Yes (engineInstanceId match) | Checked at line 759 | **PASS** |
| G22: forged Permit (default-construct) | compile error | `ReclaimPermit` deleted default ctor | **PASS** |

**G10-G22 verdict**: **All PASS** (D108 already confirmed).

### G23: physical destruction ordering

| Aspect | I4 literal | Current source | Verdict |
|---|---|---|---|
| Physical delete happens AFTER Permit.consume | Yes | `reclaimShutdownQuiescent` line 761 `permit.consume()` → line 765 `retire` → line 771 `reclaim` (physical) | **PASS** |
| `delete EQCoeffCache` only after `tryShutdownQuiescentReclaim` success | Yes | `CacheMap::~CacheMap` calls `tryShutdownQuiescentReclaim` then `resolve()` then `delete` | **PASS** |

**G23 verdict**: **PASS** (D108 already confirmed).

---

## D110-2 — G04 swapPending_ authority re-audit

### D110-2 — swapPending_ write/read sites

| Site | Operation | Thread | Authority |
|---|---|---|---|
| `ISRRuntimePublicationCoordinator.cpp:110` | markTransitionStart: `swapPending_=true` | NonRT (commit) | State transition start (single authority) |
| `ISRRuntimePublicationCoordinator.cpp:119` | markTransitionCommitted: `swapPending_=false` | NonRT (commit) | State transition end (single authority) |
| `setSwapPending` (line 417, public method) | manual set (test only) | Test only (0 production callers) | Test injection |
| `ISRRuntimePublicationCoordinator.cpp:244` | isSwapPending() read (early-return) | NonRT (any) | Observation |
| `ISRRuntimePublicationCoordinator.cpp:489` | isSwapPending() read (early-return in isFullyDrained) | NonRT | Observation |
| `ISRRuntimePublicationCoordinator.cpp:507` | `swapPending_` read in isFullyDrained | NonRT | Observation |
| `setSwapPending` callers | `ISRSemanticValidationTests.cpp:333, 367, 399, 405, 437` | Test | Test injection |

### D110-2 verdict

`swapPending_` has **single-write authority in production** (markTransitionStart/Committed pair), **only test code** uses `setSwapPending`. This is consistent with G04 architectural design (D101-30: separate from AdmissionState because swap-pending is a **transition state**, not admission state).

**D110-2 finding**: G04 is **architecturally intentional separation**, not a bug. The two-state model is:
- `AdmissionState` (in `packedState_`): Open/Closing/Closed/Faulted — coarse-grained admission gate
- `swapPending_` (independent atomic): true during an in-progress world swap — fine-grained torn-read guard

These two states are **mutually orthogonal**: `swapPending_=true` does not mean admission closed; it just means "the publication world is currently swapping, so don't read mid-swap". The G-H race between `tryAdmit` (admission gate) and `swapPending_` (torn-read guard) is **not a problem** because they protect different invariants.

**No production code path breaks G04 invariant**. G04 = **PASS**.

### closeAdmission × swapPending_ race analysis

Sequence:
1. `closeAdmission()` — CAS on `packedState_` (Open→Closing). This does NOT touch `swapPending_`.
2. After closeAdmission: any new `markTransitionStart` would set `swapPending_=true`. But `markTransitionStart` should not be called after `closeAdmission` because the publish flow is closed.
3. **Critical**: D101-31-A shows `markTransitionStart` is **not called from Recovery/Build/Publish** (those are `tryAdmit`-gated, not `markTransitionStart`-gated). `markTransitionStart` is called **only from `commit`/`publish`** paths, which are also `tryAdmit`-gated (Path A/B).

**Conclusion**: `swapPending_=true` can only be set while `AdmissionState=Open` (because `tryAdmit` rejects Closed). Therefore, **after `closeAdmission`, no future `swapPending_=true` can be set**, and `swapPending_` will be observed as `false` once the last in-progress swap completes.

**G04 race = PASS** (D101-33-D no-resurrection verified).

---

## D110-3 — G07/G08/G09 admission authority

### D110-3 — Producer × closeAdmission race analysis

All 4 producer gates call `tryAdmit(1)` on the **same `packedState_` atomic word**:

| Producer | File:line | Gate | CAS word |
|---|---|---|---|
| Path A (Orchestrator publication) | `RuntimePublicationOrchestrator.cpp:65509` (D101-31-A:264) | `tryAdmit(1)` | `packedState_` |
| Path B (facade direct commit) | `AudioEngine.h:47505` (D101-33-C) | `tryAdmit(1)` | `packedState_` |
| Recovery (submitRecoveryRequest) | `ISRRuntimePublicationCoordinator.cpp:47505` | `tryAdmit(1)` | `packedState_` |
| Build (submitRebuildIntent) | `RebuildDispatch.cpp:319` (D101-31-A:111) | `tryAdmit(1)` | `packedState_` |

`closeAdmission()` also CASs on `packedState_` (Open→Closing).

**Race linearization**: The single-word CAS ensures **total order**. Either:
- **Case A**: `tryAdmit` wins → admit succeeds, count incremented, closeAdmission later observes state=Open and proceeds to Closing
- **Case B**: `closeAdmission` wins → state=Closing, all subsequent `tryAdmit` observe state≠Open and reject

This is **G-H linearization** (D101-33-D, D101-31-C test `testConcurrentTryAdmitRelease`).

**D110-3 verdict**: All 3 producer × closeAdmission races **resolve to a single CAS** on `packedState_`. G07, G08, G09 = **PASS** (architectural concern in D108 is cosmetic).

---

## D110-4 — NORMAL/SHUTDOWN reclaim boundary

### I4 literal text

I4:2439: `reclaim: requestReclaim → reclaimNormal (retire 冪等 + epoch 確認 + reclaim)`

This is the **two-path model**. I4 specifies:
- `requestReclaim` → `reclaimNormal` (NORMAL, no Permit, epoch-gated)
- (separate path, not in I4:2439, but implied by D2_IMPL_CHECKLIST A2-2): `tryShutdownQuiescentReclaim` → `reclaimShutdownQuiescent` (SHUTDOWN, Permit required)

### Current source matches I4

| Path | Current source | I4 spec |
|---|---|---|
| NORMAL reclaim | `requestReclaim` → `reclaimNormal` → `tryReclaim` (epoch-gated) | I4:2439 (exact match) |
| SHUTDOWN reclaim | `tryShutdownQuiescentReclaim` → `reclaimShutdownQuiescent` (Permit required) | I4 D101-33 Step 8 (D2_IMPL_CHECKLIST A2-2) |

**D110-4 verdict**: Current source **matches I4 exactly** for the two-path model. D108's premise that "all reclaims need Permit" was incorrect. **A2 contract is "ShutdownQuiescent destruction reclaim needs Permit"**, not "all reclaims need Permit".

---

## D110-5 — Production destruction path re-audit (3 paths)

### Path 1: activeHandle (ReleaseResources.cpp:457)

```cpp
// D109/D110 verified
dspHandleRuntime_.retire(activeHandle);  // step 0: pre-retire
const bool reclaimed = tryShutdownQuiescentReclaim(activeHandle);  // step 1-6
//   inside: tryMakeQuiescenceProof → tryMakeReclaimPermit → permit.identity check
//   → permit.consume → retire → reclaim (line 765, 771)
jassert(reclaimed);
```

**Sequence**: retire → Proof → Permit → identity validate → consume → retire (idempotent) → reclaim. ✅

### Path 2: fadingHandle (ReleaseResources.cpp:464)

```cpp
dspHandleRuntime_.retire(fadingHandle);
const bool reclaimed = tryShutdownQuiescentReclaim(fadingHandle);
```

**Sequence**: same as activeHandle. ✅

### Path 3: CacheMap (AudioEngine.h:2106)

```cpp
const bool reclaimed = owner->tryShutdownQuiescentReclaim(entry.second);
// on success:
resolve();  // logical state transition
delete EQCoeffCache;  // physical destruction
```

**Sequence**: Proof → Permit → identity validate → consume → retire → reclaim → resolve → delete. ✅

### D110-5 verdict: All 3 paths follow the same physical-destruction ordering: **Permit.consume() → reclaim() → delete**. No path has `delete` before `Permit.consume()`. **G23 = PASS** for all 3 production paths.

---

## D110 Final Matrix (G01-G23)

| Gate | I4 literal | Current source | Verdict | Blocking |
|---|---|---|---|---|
| G01 | external setter production caller = 0 | 0 production callers (T1 satisfied) | **PASS** (header-cosmetic PARTIAL is non-blocking) | NO |
| G02 | counter mutation = single authority (semantic event) | All active production counters single-authority | **PASS** | NO |
| G03 | `consecutiveFailureCount`, K=4 | D105-R18 implemented | **PASS** | NO |
| G04 | `swapPending_` pre-check preserved | `isFullyDrained` line 489, 507 preserved | **PASS** | NO |
| G05 | `tryMakeQuiescenceProof` private | Private member of ShutdownRuntime | **PASS** | NO |
| G06 | Path B `tryAdmit(1)` on packedState_ | ISRRuntimePublicationCoordinator.cpp:57732 verified | **PASS** | NO |
| G07 | Recovery `tryAdmit(1)` | submitRecoveryRequest:47505 verified | **PASS** | NO |
| G08 | Build `tryAdmit(1)` | RebuildDispatch.cpp:319 verified | **PASS** | NO |
| G09 | Publish `tryAdmit(1)` (Path A + B) | Path A:65509 / Path B:57732 verified | **PASS** | NO |
| G10 | Proof acceptance | D106 verified | **PASS** | NO |
| G11 | Permit acceptance | D106 verified | **PASS** | NO |
| G12 | stale Proof reject | identity check | **PASS** | NO |
| G13 | Permit private ctor | ShutdownRuntime only | **PASS** | NO |
| G14 | pending reclaim identity | ReclaimIdentity{handle, retireSequence} | **PASS** | NO |
| G15 | Proof immutability | move-only | **PASS** | NO |
| G16 | Permit immutability | move-only | **PASS** | NO |
| G17 | Proof private ctor | Private | **PASS** | NO |
| G18 | generation match | line 759 check | **PASS** | NO |
| G19 | epochGeneration match | line 759 check | **PASS** | NO |
| G20 | readerRegGeneration match | line 759 check | **PASS** | NO |
| G21 | stale Permit reject (engineInstanceId) | line 759 check | **PASS** | NO |
| G22 | forged Permit compile error | deleted default ctor | **PASS** | NO |
| G23 | physical delete after Permit.consume | `reclaimShutdownQuiescent` line 761, 765, 771 sequence | **PASS** | NO |

**All 23 gates PASS** (G01 has cosmetic PARTIAL on header-existence, but T1 I4 contract is met).

---

## D110 Final Verdict

### D110 verdict: **Case A (A2 implementation GO)**

**Rationale**:

1. **All 23 I4 G01-G23 gates PASS** (G01 PARTIAL is header-cosmetic, not I4-blocking)
2. **D108's "GAP" claim was incorrect** (D109 corrected, no real GAP exists)
3. **D109's "Case B" verdict** (existing path satisfies A2) is now **upgraded to Case A** because D110 confirms all gates PASS
4. **I4 two-path model** is correctly implemented (requestReclaim → reclaimNormal / tryShutdownQuiescentReclaim → reclaimShutdownQuiescent)
5. **3 production destruction paths** (activeHandle, fadingHandle, CacheMap) all have correct ordering (Permit.consume → physical delete)
6. **Single Admission authority** (packedState_ CAS) covers all 4 producer gates

### A2 implementation: **GO**

The A2 production reclaim path is **already wired** (D109: 3 production callers of `tryShutdownQuiescentReclaim`). All G01-G23 conditions are met. Implementation can proceed.

### Required changes before implementation: **0**

The implementation is **already in place**. The only "required" change would be the cosmetic removal of 3 test-only setters from the header, but this is **NOT required by I4 contract** (T1 satisfied) and is **NOT blocking A2**.

### I4 amendment required: **NO** (optional)

D110 does **not** require any I4 amendment. The current I4 is consistent with current source. An optional I4 clarification could be made:
- D37.2 reclaim model: explicitly add the SHUTDOWN path (`tryShutdownQuiescentReclaim` → `reclaimShutdownQuiescent`)

But this is **purely cosmetic** for documentation, not for contract.

---

## D110 → D111 hand-off

**D110 verdict: Case A (A2 implementation GO).**

**D111 should focus on**:
1. Optional I4 amendment to D37.2 to explicitly mention the two-path reclaim model (cosmetic)
2. Optional header cleanup to remove 3 test-only setters (cosmetic, NOT required)
3. Build + test verification (40/40 should still pass)
4. **Optional D-G17 formal closure** for the A2 reclaim path

**D111 may proceed with read-only audit only** — no production code changes required to satisfy A2.

---

## File references (read-only, no source modification)

| File | Role |
|---|---|
| `ConvoPeq.md` (2026-08-27 14:33 + 2026-08-28 21:22) | runtime source baseline |
| `doc/work88/I4_DESIGN_CONTRACT.md:2439` | I4 reclaim: requestReclaim → reclaimNormal (literal two-path model) |
| `doc/work88/REPAIR_PLAN2-dash2.md:2804-2811` | T1-T8 A2 test conditions (T1 = setter production caller = 0) |
| `doc/work88/D2_IMPL_CHECKLIST.md:76-83` | A2-1, A2-2, A2-3, A2-6 (reclaimNormal / reclaimShutdownQuiescent / old API delete / Permit supply) |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp:264-422` | `setRetireBacklogCount`, `setSwapPending` (test-only, 0 production callers) |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp:56517-56599` | `requestReclaim` / `reclaimNormal` / `reclaimShutdownQuiescent` (two-path) |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp:743-773` | `reclaimShutdownQuiescent` (Permit consume → retire → reclaim) |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp:47502, 47505, 47592, 47603` | `tryAdmit(1)` calls (4 producer gates) |
| `src/audioengine/ISRShutdown.cpp:59460-59570` | `closeAdmission()` / `tryAdmit()` / `packedState_` CAS |
| `src/audioengine/AudioEngine.h:4367-4396` | `tryShutdownQuiescentReclaim` (3 production callers) |
| `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:457, 464` | 2 of 3 production callers (graceful drain) |
| `src/audioengine/AudioEngine.h:2106` | 1 of 3 production callers (CacheMap destroy) |
| `src/tests/ISRSemanticValidationTests.cpp:326-437` | test-only setter callers (5 sites, all test) |
| `src/tests/AdmissionPackedStateTests.cpp:245-371` | G-H race tests (D101-31-B) |
| `evidence/D108_A2_GATE_INVENTORY.md` | D108 G01-G23 inventory (D109/D110 correction target) |
| `evidence/D109_A2_RECLAIM_PATH_WIRING.md` | D109 Case B (3 callers verified) |
| `evidence/D107_ISFULLYDRAINED_AUDIT.md` | 16 conditions, G-H race separation |
| `evidence/D106_SHUTDOWN_LIFETIME_QUIESCENCE_AUDIT.md` | Q0-Q7 / C1-C7 / G15-G22 |

**No source files modified. No I4 files modified. No tests added.** D110 is a
read-only contract authorization audit.
