# D113 — Phase-I Closure Baseline / Residual Surface Audit

**Status:** **D113 verdict: PASS — Phase-I baseline frozen / no implementation authorized**
**Source changes:** 0
**I4 changes:** 0
**Test changes:** 0
**Type:** Read-only baseline verification

D113 establishes the **frozen Phase-I closure baseline** by re-verifying the current
production source against the D112 closure state. D113 is the **read-only
boundary fix** between Phase-I (closed by D111 A2 + D112 Recovery) and any
future Phase-II work. D113 does **not** authorize any source change; it fixes
the **terminology and boundary** so that future audits cannot retroactively
re-import Phase-II structures as "D112 gaps".

---

## D113-1 — Current source matches D112 baseline

### D113-1.1 — D112 baseline anchor

| Artifact | D112 baseline | D113 verification (current source) | Match? |
|---|---|---|---|
| `kMaxLogicalRecoveryObligations = 32` | `h:325` | `h:325` (unchanged) | ✅ |
| `RecoveryAdmissionTable<kMaxLogicalRecoveryObligations>` (single instantiation) | `h:934` | `h:934` (unchanged) | ✅ |
| `tryInsert` definition | `h:354-373` | `h:354-373` (unchanged) | ✅ |
| `tryInsert` production call site | `cpp:880` (1 site) | `cpp:880` (1 site, unchanged) | ✅ |
| `resolve` definition | `h:383-396` | `h:383-396` (unchanged) | ✅ |
| `resolve` production call sites | `cpp:977, 1019` (2 sites, R20 final) | `cpp:977, 1019` (2 sites, unchanged) | ✅ |
| `markTransientFailure` definition | `cpp:998-1024` | `cpp:998-1024` (unchanged) | ✅ |
| `markTransientFailure` production call sites | 5 sites (Orchestrator :191/258/311/401 + Builder :1039/1063) | 5 sites (R20 final, unchanged) | ✅ |
| `redriveDeferredRecoveryObligations` definition | `cpp:1042-1056` | `cpp:1042-1056` (unchanged) | ✅ |
| `redriveDeferredRecoveryObligations` triggers | `Threading.cpp:266` (CoordinatorLoop tick) + `cpp:847` (submit start) | 2 sites (R5-10 final, unchanged) | ✅ |
| `submitRecoveryRequest` linearization | `cpp:820-942` (coalesce → tryInsert → transport/durable) | `cpp:820-942` (unchanged) | ✅ |
| `isFullyDrained` includes `liveLogicalRecoveryObligationCount() == 0` (R13 hardening) | `cpp:556` | `cpp:556` (unchanged) | ✅ |
| Shutdown admission gate (`state_ == ShuttingDown`) | `cpp:824-828` | `cpp:824-828` (unchanged) | ✅ |
| `discardRecoveryRequestsOnShutdown` table sweep | `cpp:1203` | `cpp:1203` (unchanged) | ✅ |
| `discardPendingRecoveryAdmission` durable clear | `cpp:1152` | `cpp:1152` (unchanged) | ✅ |
| `kMaxObligationConsecutiveFailures = 4` | `h:331` | `h:331` (unchanged) | ✅ |
| `CoalesceIdentity = {handle, target}` | `h:262-269` | `h:262-269` (unchanged) | ✅ |
| `SemanticRecoveryTarget` (3-field) | `h:248-257` | `h:248-257` (unchanged) | ✅ |

**D113-1.1 verdict**: **All 17 D112 baseline anchors are byte-identical in current
source**. The D111/D112 closure state is structurally preserved.

### D113-1.2 — Live reservation / delivery / terminal sub-state (all unchanged)

| Sub-state | Anchor | Status |
|---|---|---|
| `liveCount_` reservation counter (single +1/-1) | h:368/h:390 | unchanged (D113-2 below for strict re-audit) |
| `delivery = None / Transport / Durable` (R5-10) | h:281-285, h:318 | unchanged |
| `state = Live / 5 terminals` (R5-8 / R18 / R21) | h:271-279 | unchanged |
| `consecutiveFailureCount` (R18) | h:323 | unchanged |
| `id` monotonic allocator (D105-R5-8) | h:360-361 | unchanged |

**D113-1 verdict**: **PASS**. Current ConvoPeq source (2026-08-28 21:22 baseline,
4,633,246 bytes) matches the D112 Phase-I baseline with **0 source drift**.

---

## D113-2 — Single +1 / -1 Authority (strict re-audit)

The D112 §D112-8.4 audit found single +1 at h:368 and single -1 at h:390. D113
re-runs this audit **with the same scope** (full `src/` production) to confirm
no new mutation sites have been introduced.

### D113-2.1 — `liveCount_` mutation sites (full src/ production)

| Site | Operation | Code |
|---|---|---|
| `ISRRuntimePublicationCoordinator.h:355` | `load ≥ kCapacity` (read-only guard) | unchanged |
| `ISRRuntimePublicationCoordinator.h:368` | `fetchAddAtomic(liveCount_, 1, release)` — **+1** | unchanged |
| `ISRRuntimePublicationCoordinator.h:390` | `fetchSubAtomic(liveCount_, 1, release)` — **-1** | unchanged |
| `ISRRuntimePublicationCoordinator.h:399` | `consumeAtomic(liveCount_, acquire)` (read-only accessor) | unchanged |
| `ISRRuntimePublicationCoordinator.h:410` | `liveCount_{0}` (declaration) | unchanged |

**Result**: `+1` mutation = 1 site (h:368). `-1` mutation = 1 site (h:390). **0
bypass paths**.

### D113-2.2 — `recoveryAdmissions_.tryInsert` call sites

| Site | Role |
|---|---|
| `ISRRuntimePublicationCoordinator.cpp:880` | `submitRecoveryRequest` (the only production call site) |

**Result**: 1 production call site. **0 bypass paths**.

### D113-2.3 — `recoveryAdmissions_.resolve` call sites

| Site | Role |
|---|---|
| `ISRRuntimePublicationCoordinator.cpp:977` | `resolveRecoveryObligation` (single obligation resolution API) |
| `ISRRuntimePublicationCoordinator.cpp:1019` | `markTransientFailure` exhaustion branch (the only sanctioned Failed terminal) |

**Result**: 2 production call sites (both internal to the single -1 authority).
**0 bypass paths**.

### D113-2.4 — `state` and `id` write sites (other than +1/-1)

| Site | Operation | Code |
|---|---|---|
| `h:360-361` | `++nextId_` + `slots_[i].id.store(id, relaxed)` | tryInsert only |
| `h:363` | `slots_[i].state.store(Live, release)` | tryInsert only (initial state) |
| `h:388` | `compare_exchange_strong(Live → terminal)` | resolve only (idempotent CAS) |
| `h:389` | `consecutiveFailureCount.store(0, release)` | resolve (post-CAS reset) |

**Result**: All `id` writes, `state` writes, and `consecutiveFailureCount` resets
are inside `tryInsert` or `resolve`. **0 bypass paths**.

### D113-2.5 — Idempotency (resolve CAS)

`resolve` (`h:383-396`):
- Re-scans table by `id` (no cached index).
- `compare_exchange_strong(expected=Live, terminal)` → on success, `liveCount_--`.
- Lost CAS (already terminal) → returns false (no double -1).
- Unknown id (stale / late) → no-op (returns false).
- Monotonic `++nextId_` (h:360) ensures reused slots get strictly-greater ids.

**Result**: Idempotent CAS-based -1, no double-decrement, no L underflow.

### D113-2.6 — Practical Stable ISR Bridge Runtime compatibility

The D113 single +1/-1 authority matches the Practical Stable ISR Bridge Runtime
principles:

| PSIBR principle | D113 verification |
|---|---|
| "RT は判断・所有・解放を行わない" (RT does not judge / own / release) | `recoveryAdmissions_` is a **CoordinatorLoop-owned** resource (NonRT); Audio Thread never touches `liveCount_` (no RT path to +1/-1) |
| "Publish / Crossfade / Retire の決定権を単一化" (Unify decision authority) | `RuntimePublicationOrchestrator` (RuntimePublicationState.h:101: "唯一の書込権限者") + `RuntimeWorldAuthority` (RuntimeWorldAuthority.h:86-104 INV-X4-2/3/5: "sole physical write path to RuntimeStore") + `RecoveryAdmissionTable` (single +1/-1 authority) — **3 single authorities, none bypassed** |
| "Overflow しても失われない" (No loss on overflow) | D105-R5-9 MUST-2 (no-clobber) + R5-10 (redrive for deferred obligations) + R5-8 (capacity-reject with telemetry, not silent drop) |
| "Shutdown → Drain → Reclaim → Verify" | (1) `requestShutdown` → state_=ShuttingDown (cpp:560); (2) submit gate (cpp:824-828); (3) drain (discardRecoveryRequestsOnShutdown cpp:1203, discardPendingRecoveryAdmission cpp:1152); (4) verify (isFullyDrained cpp:506, with R13 hardening cpp:556) |

**D113-2 verdict**: **PASS**. The single +1/-1 authority is **structurally
intact**. No new mutation paths exist. The Practical Stable ISR Bridge Runtime
principles are **code-proven compatible** with the current Phase-I implementation.

---

## D113-3 — Residual Authority / Ownership Bypass Audit (read-only)

D113-3 is the **reverse-direction audit**: scan the entire production source for
**any code that could mutate the recovery logical-obligation domain outside the
single +1/-1 authority**. The D105-R5-8 / R5-9 / R18 / R20 / R21 chain established
this as a structural invariant; D113 confirms it has not regressed.

### D113-3.1 — `slot()` mutation sites (delivery / state / id / counter)

The 13 `slot(...)` access sites in `ISRRuntimePublicationCoordinator.cpp` were
classified as follows (all unchanged from D112):

| Site | Operation | Field | ΔL? |
|---|---|---|---|
| cpp:845 | read | `delivery` (wasDeferredBefore snapshot) | 0 |
| cpp:869 | read | `id.load` (coalesce) | 0 |
| cpp:877 | read | `delivery` (no-double-delivery guard) | 0 |
| cpp:910 | **write** | `delivery = Transport` (transport push) | 0 |
| cpp:925 | **write** | `delivery = None` (durable-slot busy → defer) | 0 |
| cpp:940 | **write** | `delivery = Durable` (durable admission) | 0 |
| cpp:1003 | read | `id.load` (markTransientFailure lookup) | 0 |
| cpp:1005 | read | `state.load` (Live guard) | 0 |
| cpp:1010 | **write** | `delivery = None` (P-B repair) | 0 |
| cpp:1013 | **write** | `consecutiveFailureCount.fetch_add(1, acq_rel)` (counter +1) | 0 |
| cpp:1064 | read | `id.load` (redrive lookup) | 0 |
| cpp:1214 | read | `id.load` (discard sweep lookup) | 0 |

**Write operations** are all on `delivery` (ΔL=0) or `consecutiveFailureCount` (ΔL=0).
**No write to `state` or `id` outside the table's `tryInsert` / `resolve` methods**.
**0 bypass paths**.

### D113-3.2 — `state` Live → terminal transitions

The only `Live → terminal` transition is via `compare_exchange_strong` at `h:388`
inside `resolve`. No direct `state.store(ResolvedXxx)` exists outside the table.
**0 bypass paths**.

### D113-3.3 — `id` write sites (monotonic allocator)

`id.store` appears only at `h:361` (inside `tryInsert`, after `++nextId_` at h:360).
No external `id.store` exists. The `nextId_` field is `private` to the table
class (h:408-410). **0 bypass paths**.

### D113-3.4 — `findByKey` / `slot(i)` external accessors

`findByKey` (h:344-351) is `const` and returns an index. It is used in
`submitRecoveryRequest` (cpp:866) for coalesce lookup and in
`submitRecoveryRequest` (cpp:843) for the `wasDeferredBefore` snapshot. **Read-only**.

`subscribe-style` external accessors (`slot(i)`, `consecutiveFailureCount(i)`,
`liveCount()`) are all read-only (no internal mutation). Used by:
- `markTransientFailure` (internal; the only mutator outside the table)
- `redriveDeferredRecovery` (internal; delivery write via slot access)
- Test code (read-only)

**0 bypass paths**.

### D113-3.5 — Recovery outcome / terminal state production callers

| Outcome | Production caller | ΔL | I4-sanctioned? |
|---|---|---|---|
| `Published` | `resolveRecoveryObligation(id, Published)` (cpp:977) | -1 | ✅ (Success) |
| `StaleSuperseded` | `resolveRecoveryObligation(id, StaleSuperseded)` (cpp:977) | -1 | ✅ (Superseded) |
| `ShutdownDiscarded` | (1) `discardRecoveryRequestsOnShutdown` (cpp:1203); (2) `resolveRecoveryObligation(id, ShutdownDiscarded)` (cpp:977) | -1 | ✅ (ShutdownDiscard) |
| `ResolvedFailed` (RetryExhaustion) | `markTransientFailure` exhaustion (cpp:1019) | -1 | ✅ (RetryExhaustion, R21 I4 amendment) |
| `ResolvedRetry` (defined but unused) | none | n/a | n/a (enum value reserved) |
| `Failed` (via `RecoveryOutcome::Failed`) | 0 production caller (R20 §R20-6 verified) | n/a | n/a |

**0 unsanctioned production terminals**. R15-A (D105-R15 contract violation) remains
**structurally closed** by R18 + R20 + R21.

### D113-3.6 — `redriveDeferredRecoveryObligations` trigger sites

| Site | Thread | Role |
|---|---|---|
| `cpp:847` (submitRecoveryRequest start) | CoordinatorLoop | opportunistic re-drive (D105-R5-10 §2) |
| `Threading.cpp:266` (runCoordinatorPhase 1ms tick) | CoordinatorLoop | periodic redrive (D105-R5-10 §2) |

Both run on CoordinatorLoop (the sole writer thread). SPSC-safe. **0 bypass paths**.

### D113-3.7 — `discardRecoveryRequestsOnShutdown` sweep

| Site | Role |
|---|---|
| `RebuildDispatch.cpp:802` (Builder join sequence) | calls `discardRecoveryRequestsOnShutdown` |

`discardRecoveryRequestsOnShutdown` (cpp:1203) loops all 32 slots; for each non-zero
`id`, calls `resolveRecoveryObligation(id, ShutdownDiscarded)`. This is **terminal-only**
(no +1, no new id) and is the **shutdown-time -1 mass authority**. After this sweep,
`liveCount_ == 0` is structurally enforced (R12).

**0 bypass paths**.

### D113-3.8 — Bypass verdict

D113-3 found **0** new production authority / ownership bypass paths. The D112
single +1/-1 authority is preserved. The D105-R5-8 / R5-9 / R18 / R20 / R21
chain remains structurally complete.

**D113-3 verdict**: **PASS** (0 new authority / ownership bypass).

---

## D113-4 — Phase-I / Phase-II Boundary Freeze

The D113 brief mandates an **explicit freeze boundary** between Phase-I
(closed) and Phase-II (deferred). The freeze is **declarative** (no source
change) but is a **future-proofing** invariant: if a future audit attempts to
re-introduce any of the deferred items as a "D112 gap", the freeze makes
that introduction **out of policy** (must be explicitly overridden with a
Phase-II objective + design audit).

### D113-4.1 — Phase-I production (FROZEN = CLOSED)

The following structures are **closed in Phase-I** and must not be modified
unless D113 itself is re-audited:

| # | Structure | Location | Status |
|---|---|---|---|
| 1 | `RecoveryAdmissionTable<kMaxLogicalRecoveryObligations>` (32) | h:325, h:339-411, h:934 | **CLOSED** (single +1/-1 authority, ABA-safe, R5-8/R5-9) |
| 2 | `CoalesceIdentity = {DSPHandle, SemanticRecoveryTarget}` | h:262-269 | **CLOSED** (R5-9 MUST-3) |
| 3 | `SemanticRecoveryTarget` (3-field equality) | h:248-257 | **CLOSED** (R5-9; 5-field extension = D113-B) |
| 4 | `LogicalRecoveryObligation` struct | h:310-324 | **CLOSED** (R5-8 + R5-10 + R18) |
| 5 | `kMaxLogicalRecoveryObligations = 32` (single source) | h:325 | **CLOSED** (R24 §3 uniqueness audit) |
| 6 | `tryInsert` (single +1) | h:354-373 | **CLOSED** (R5-8) |
| 7 | `resolve` (single -1, id-based CAS) | h:383-396 | **CLOSED** (R5-9 MUST-3) |
| 8 | `markTransientFailure` (transient-failure adjudication) | h:432-442, cpp:998-1024 | **CLOSED** (R18) |
| 9 | `kMaxObligationConsecutiveFailures = 4` | h:331 | **CLOSED** (R18; K=4 inherited from Builder-local counter) |
| 10 | `redriveDeferredRecoveryObligations` (R5-10 redrive) | h:455, cpp:1042-1056 | **CLOSED** (R5-10; R10 audit) |
| 11 | `redriveDeferredRecovery` (per-obligation) | h:456, cpp:1058+ | **CLOSED** (R5-10) |
| 12 | `recoveryRetryExhaustedCount_` (exhaustion telemetry) | h:944, cpp:1018 | **CLOSED** (R18 / R21) |
| 13 | `submitRecoveryRequest` linearization (coalesce → tryInsert → transport/durable) | cpp:820-942 | **CLOSED** (R5-9 / R5-10) |
| 14 | `submitRecoveryRequest` shutdown gate | cpp:824-828 | **CLOSED** |
| 15 | `isFullyDrained` predicate (with R13 hardening) | cpp:506-557 | **CLOSED** (D105-R12 / R13) |
| 16 | `discardRecoveryRequestsOnShutdown` (table sweep) | cpp:1203 | **CLOSED** (R12 / R18) |
| 17 | `discardPendingRecoveryAdmission` (durable clear) | cpp:1152 | **CLOSED** |

### D113-4.2 — Phase-II deferred (FROZEN = DO NOT IMPLEMENT)

The following structures are **explicitly deferred to Phase-II** and must NOT be
introduced into Phase-I production:

| # | Structure | Reason for deferral | Reference |
|---|---|---|---|
| 1 | `RecoveryEpisodeId` (allocator + field) | R23 §D19.3 removed from Phase-I; O_max unbounded without snapshot freeze (R26) | D105-R25 / R26 |
| 2 | `RecoveryGeneration` | R23 deferred; Phase-I uses `DSPHandle.generation` instead | D105-R25 |
| 3 | 5-field `SemanticRecoveryTarget` (add `convolverFingerprint`, `buildInputHash`) | Requires `RuntimeBuildFingerprint` expansion; not Phase-I requirement (3 fields sufficient for coalesce) | D112-4.3 / D113-B |
| 4 | Snapshot freeze of `currentBuildSnapshot_` | R26 Policy P3; required only if `E_max × O_max ≤ 32` becomes Phase-II requirement | D105-R26 |
| 5 | `E_max × O_max ≤ 32` (re-introduction) | R22 NO-GO (structurally unprovable); R23 explicitly removed from Phase-I | D105-R22 / R23 |
| 6 | MPSC化 of `recoveryIntentQueue_` | R28 NO 2nd producer; PSIBR principle: "Authority を単一化" (don't add producers arbitrarily) | D105-R28 |
| 7 | Phase-II supersession (`canSupersede`) | I4 D12.4 deferred; Phase-I uses `RejectedStaleGeneration → StaleSuperseded` (D105-R5-9 MUST-4) | D105-R5-9 / R25 |
| 8 | Phase-II episode closure linearization (D20 / D23) | I4 §D20 / §D23 explicitly "Phase-I production contract としては無効" | R24 §2 |
| 9 | `RuntimeBuildFingerprint` extension | D113-B; not Phase-I; not blocking | — |

**D113-4.2 verdict**: All 9 Phase-II deferred items have **0 production hits** in
current source. The freeze boundary is **structurally intact**.

### D113-4.3 — Boundary completeness (D112 PASS vs N/A)

The D113 brief mandates **explicit separation** between D112 PASS items and
D112 N/A items. The "Phase-II absence is not a Phase-I defect" principle must
be preserved across future audits.

| Item | D113 status | Phase | D112 classification |
|---|---|---|---|
| `liveLogicalRecoveryObligationCount ≤ 32` | **PASS** | Phase-I | D112 D112-2.2 |
| `CoalesceIdentity` (handle + target) | **PASS** | Phase-I | D112 D112-1 / D112-3 |
| `SemanticRecoveryTarget` (3-field equality) | **PASS** | Phase-I | D112 D112-4 |
| Retry-preserving failure (markTransientFailure) | **PASS** | Phase-I | D112 D112-7 |
| Ownership conservation (single +1/-1) | **PASS** | Phase-I | D112 D112-8 |
| Capacity gate (tryInsert @ L=32) | **PASS** | Phase-I | D112 D112-2.2 |
| `isFullyDrained` includes logical obligation | **PASS** | Phase-I | D112 D113-1.1 |
| Shutdown admission gate (state_==ShuttingDown) | **PASS** | Phase-I | D112 D113-1.1 |
| `discardRecoveryRequestsOnShutdown` (table sweep) | **PASS** | Phase-I | D112 D113-1.1 |
| Redrive (delivery=None re-attachment) | **PASS** | Phase-I | D112 D112-6.2 |
| Practical Stable ISR Bridge Runtime compatibility | **PASS** | Phase-I | D113-2.6 |
| --- | --- | --- | --- |
| `E_max × O_max ≤ 32` (re-introduction) | **N/A** | **Phase-II** | D112 D112-9 (R23 deferred) |
| `RecoveryEpisodeId` | **N/A** | **Phase-II** | D112 D112-5 (R23 deferred) |
| `RecoveryGeneration` | **N/A** | **Phase-II** | D112 D112-5 (R23 deferred) |
| 5-field `SemanticRecoveryTarget` | **N/A** | **Phase-II** | D112 D112-4.3 (RuntimeBuildFingerprint gap) |
| Snapshot freeze | **N/A** | **Phase-II** | D112 D112-9 (R26 Policy P3) |
| Semantic supersession (`canSupersede`) | **N/A** | **Phase-II** | D112 D112-9 (I4 D12.4) |
| MPSC化 | **N/A** | **Phase-II** | D112 D112-9 (R28 DEFER) |
| Episode closure linearization (D20/D23) | **N/A** | **Phase-II** | D112 D113-4.2 (R24) |

**D113-4 verdict**: **PASS** (boundary frozen; PASS and N/A items are explicitly
separated by Phase).

---

## D113-5 — D112 Follow-up Candidates (4 items classified)

D113-A through D113-D reclassify the D112 follow-up candidates per the D113 brief.

### D113-A — `CacheMap` destructor header comment cleanup

| Aspect | Value |
|---|---|
| Source | `AudioEngine.h:2085` (class-header comment for `~CacheMap`) |
| Stale text | `//   Shutdown: resolve → delete → reclaim（全マップ同時破棄のため安全）。` |
| Actual code | `AudioEngine.h:2106-2111` follows `reclaim → resolve → delete` (D105-R5-9 MUST-2 / R5-10 Step 13 fix) |
| D113 classification | **cosmetic / non-blocking** (D111-DOC-001) |
| Severity | documentation only (zero runtime impact) |
| Action | D113 does **not** modify. Recorded as a non-blocking follow-up. May be cleaned up in any D113+ future audit. |
| Verdict | **ACCEPTED AS-IS** (no Phase-I invariant at risk) |

### D113-B — `RuntimeBuildFingerprint` 5-field extension

| Aspect | Value |
|---|---|
| Goal | Add `convolverFingerprint` + `buildInputHash` to `RuntimeBuildFingerprint` and `SemanticRecoveryTarget` (5-field total) |
| Current state | 3-field (irIdentityHash, convolutionConfigHash, dspParameterHash) — sufficient for Phase-I coalesce (D112 D112-4) |
| Why NOT now | (1) Phase-I coalesce works with 3 fields; (2) `RuntimeBuildFingerprint` is a system-wide struct; expansion requires audit of all consumers; (3) not blocking Phase-I closure |
| PSIBR concern | Practical Stable ISR Bridge Runtime preserves "Authority を単一化"; 5-field extension is a **semantic expansion** (not a structural closure) and would need its own I4 amendment + design audit |
| D113 verdict | **DEFER** (5-field extension is a Phase-II semantic expansion; not Phase-I scope) |

### D113-C — Episode layer (RecoveryEpisodeId / RecoveryGeneration)

| Aspect | Value |
|---|---|
| Goal | Add `RecoveryEpisodeId` + `RecoveryGeneration` per I4 D13 (R25 Option B) |
| Current state | 0 production hits (R23 explicitly removed) |
| Why NOT now | (1) R22 NO-GO + R23 Path A: `E_max × O_max ≤ 32` is structurally unprovable without snapshot freeze (R26 Policy P3); (2) Phase-I has no Phase-II objective that requires episode lineage; (3) adding EpisodeId alone does **not** restore the 32 product form (R26); (4) R25/R26 explicitly DEFER |
| PSIBR concern | Adding Episode layer without a concrete Phase-II objective re-imports the **exact problem R23 resolved** (E_max / O_max / E×O ambiguity). This would re-open the NO-GO that R23 closed. |
| D113 verdict | **COMPLETE FREEZE** (do not implement without a concrete Phase-II objective + design audit) |

### D113-D — MPSC化 of `recoveryIntentQueue_`

| Aspect | Value |
|---|---|
| Goal | Replace `LockFreeRingBuffer<RecoveryIntent, 256>` (SPSC) with `MpscBoundedRing<RecoveryIntent, 4096>` (MPSC) + mutex on `pendingRecoveryAdmission_` |
| Current state | SPSC maintained; 1 NonRT producer (CoordinatorLoop) + 1 NonRT consumer (RebuildThread) (R28 verified) |
| Why NOT now | (1) R28 Case A: **no 2nd producer exists**; (2) MPSC化 requires `pendingRecoveryAdmission_` mutex (REPAIR_PLAN2 §1.1.1) + new test surface; (3) no measurable benefit in current code; (4) PSIBR principle explicitly says "Authority を単一化" — adding producers arbitrarily is the **opposite** direction |
| PSIBR concern | MPSC化 is **incompatible** with the "Authority を単一化" principle unless there is a concrete reason to add a 2nd producer. The R28 audit found no such reason. |
| D113 verdict | **COMPLETE FREEZE** (do not implement without a concrete 2nd-producer requirement) |

### D113-5 summary

| ID | Description | Verdict | Action |
|---|---|---|---|
| D113-A | CacheMap dtor header comment stale | **ACCEPTED** (cosmetic) | non-blocking follow-up |
| D113-B | RuntimeBuildFingerprint 5-field extension | **DEFER** (Phase-II semantic expansion) | not Phase-I scope |
| D113-C | Episode layer (RecoveryEpisodeId/Generation) | **COMPLETE FREEZE** (R23 explicit deferral) | re-opening requires Phase-II objective + design audit |
| D113-D | MPSC化 of recoveryIntentQueue_ | **COMPLETE FREEZE** (R28 Case A, no 2nd producer) | re-opening requires 2nd producer + design audit |

---

## D113-6 — Final Verdict

### **D113 verdict: PASS — Phase-I baseline frozen / no implementation authorized**

### D113-6.1 — D113 Completion Conditions (D113 brief §5)

| Condition | Status | Evidence |
|---|---|---|
| **D113-1**: Current ConvoPeq matches D112 Phase-I baseline | **PASS** | §D113-1 (17 anchors byte-identical; liveCount_/tryInsert/resolve/markTransientFailure/redrive/isFullyDrained/shutdown gate all unchanged) |
| **D113-2**: Phase-I invariant vs Phase-II deferred boundary clear | **PASS** | §D113-4 (17 Phase-I items CLOSED, 9 Phase-II items DEFERRED, 0 leakage) |
| **D113-3**: D112 missed new production authority / ownership path = 0 | **PASS** | §D113-3 (13 slot access sites classified; 0 bypass; single +1/-1 unchanged; Practical Stable ISR Bridge Runtime compatible) |
| **D113-4**: Phase-I change reason = 0 | **PASS** | §D113-5 (D113-A cosmetic only; D113-B/C/D Phase-II deferred) |

### D113-6.2 — Final closure statement

> **The Phase-I implementation is now FROZEN. No source change is authorized
> without an explicit Phase-II objective + Phase-II design audit.**

This D113 verdict **replaces** any prior "implementation authorized" status for
Phase-I. Future audits must:

1. Re-verify the D113-1 baseline (no drift) on every read-only audit.
2. Not introduce any of the D113-4.2 Phase-II items without a Phase-II
   objective + I4 amendment + design audit (D113-C / D113-D rules).
3. Treat any new authority / ownership path as a **regression** (D113-3 invariant).
4. Apply D113-A only as a cosmetic cleanup; not as a Phase-I invariant change.

### D113-6.3 — Hand-off to future work

D113 is the **last Phase-I closure audit**. After D113:

- **No further Phase-I implementation work** is authorized.
- **Phase-II work** requires: (a) concrete Phase-II objective (e.g., new
  diagnostic feature, new failure mode); (b) Phase-II design audit (R25-style
  Option B + freeze + registry for Episode layer; or REPAIR_PLAN2 §1.1.1
  MPSC infrastructure for 2nd producer); (c) I4 amendment for the new
  production invariant (D105-R21 model).
- **Operational validation** (runtime QA, performance benchmarks) is
  authorized (read-only) but is not a "D-audit" — it is a separate workstream.

---

## D113 — File references (read-only, no source modification)

| File | Role |
|---|---|
| `ConvoPeq.md` (2026-08-28 21:22 baseline, 4,633,246 B) | production source baseline |
| `src/audioengine/ISRRuntimePublicationCoordinator.h` | Recovery obligation table, types, accessors |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp` | submitRecoveryRequest, resolveRecoveryObligation, markTransientFailure, redrive, discard, isFullyDrained |
| `src/audioengine/AudioEngine.h:2085, 2106-2111` | D113-A: stale comment vs. actual code |
| `src/audioengine/AudioEngine.Threading.cpp:266` | redrive trigger (CoordinatorLoop tick) |
| `src/audioengine/AudioEngine.RebuildDispatch.cpp:802, 1039, 1063` | discard + markTransientFailure (Builder) |
| `src/audioengine/RuntimePublicationOrchestrator.cpp:191, 258, 311, 401` | markTransientFailure (Orchestrator) |
| `src/audioengine/RuntimePublicationState.h:101` | RuntimePublicationOrchestrator (唯一の書込権限者) |
| `src/audioengine/RuntimeWorldAuthority.h:86-104` | RuntimeWorldAuthority (sole write path to RuntimeStore, INV-X4-2/3/5) |
| `src/tests/ISRSemanticValidationTests.cpp` | C1-C16, T-R13-1..4, C8, T-R18-1..12, T-R20-1..4, T-R21-1..3 (all PASS in D111 40/40) |
| `evidence/D111_A2_IMPLEMENTATION_CLOSURE_AUDIT.md` | A2 Phase-I CLOSED |
| `evidence/D112_RECOVERY_PHASE_I_IMPLEMENTATION_READINESS_AUDIT.md` | Phase-I Recovery CLOSED |
| `evidence/D105-R5-8` / `D105-R5-9` / `D105-R5-10` / `D105-R12` / `D105-R13` / `D105-R15` / `D105-R16` / `D105-R17` / `D105-R18` / `D105-R20` / `D105-R21` / `D105-R22` / `D105-R23` / `D105-R24` / `D105-R25` / `D105-R26` / `D105-R27` / `D105-R28` | full D105 chain (Phase-I implementation + Phase-II deferral) |
| `doc/work88/I4_DESIGN_CONTRACT.md` (R23 amended) | I4 contract; E_max / O_max / E×O removed from Phase-I (R23 §D22.3) |
| `doc/work88/D2_IMPL_CHECKLIST.md` | Phase-I implementation checklist (D105-R5-9 / R5-10 / R18 / R20 / R21 closed) |
| `doc/work88/REPAIR_PLAN2-dash2.md` | MPSC化 future-item reference (REPAIR_PLAN2 §1.1.1) |

**No source files modified. No I4 files modified. No tests added.** D113 is a
read-only closure baseline audit that fixes the Phase-I / Phase-II boundary
and authorizes **no** further implementation work on Phase-I.
