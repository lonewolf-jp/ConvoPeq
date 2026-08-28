# D105-R27 — Phase-II Objective Necessity & Episode-Requirement Audit

**Status:** **R27 verdict: Case A (No concrete Phase-II objective → Episode layer DEFER permanently)**
**Production source changes:** 0
**I4 changes:** 0
**Test changes:** 0
**R20 + R23 + R24 + R25 + R26 = stable final state; no further Runtime work authorized**

R27 establishes that **no concrete Phase-II requirement currently exists that cannot
be satisfied by the current Phase-I obligation-table model (R23 closure)**. Therefore
the Episode layer is **permanently DEFERRED to Phase-III+** unless an explicit Phase-II
requirement later emerges.

---

## R27-1 — Phase-II objective inventory

Each row evaluates whether the current Phase-I `RecoveryAdmissionTable<32>` can
satisfy the objective. Only objectives that produce a **NO** verdict justify Episode
implementation.

| Objective | Obligation table alone? | EpisodeId required? | Episode implementation justified? |
|---|---|---|---|
| `liveCount_ ≤ 32` (INV-CAP-1) | **YES** (tryInsert gate h:355-356) | NO | **NO** |
| Coalesce by (handle, target) | **YES** (R5-9 MUST-3) | NO | **NO** |
| `findByKey` matching | **YES** (h:344) | NO | **NO** |
| Snapshot drift identification (R3) | **YES** (different target → new obligation) | NO | **NO** |
| Per-obligation identity (`obligationId`) | **YES** (monotonic counter) | NO | **NO** |
| Per-obligation telemetry (`recoveryObligationShutdownDiscardCount_`, `recoveryRetryExhaustedCount_`) | **YES** (each counted per obligation) | NO | **NO** |
| Transient failure (Live→Live, ΔL=0, P-B) | **YES** (R18) | NO | **NO** |
| RetryExhaustion (counter==K) | **YES** (R18/R21, K=4) | NO | **NO** |
| Shutdown discard (PendingRecoveryAdmission + table) | **YES** (R11) | NO | **NO** |
| Admission capacity enforcement (32) | **YES** (tryInsert) | NO | **NO** |
| `RecoveryIntentQueue<256>` (transport) | **YES** (SPSC) | NO | **NO** |
| `PendingRecoveryAdmission` (1 slot durable) | **YES** (single-slot struct) | NO | **NO** |
| `quarantineActiveFlags_[256>` (Q_max) | **YES** (DSP-side) | NO | **NO** |
| isFullyDrained predicate | **YES** (R11/R12) | NO | **NO** |
| Per-episode unit telemetry | NOT YET (current is per-obligation) | YES (if required) | **Re-evaluate** (Case B trigger) |
| Per-episode diagnostic identifier | NOT YET (current is obligationId) | YES (if required) | **Re-evaluate** (Case B trigger) |
| Per-episode quota | **NO** (no current per-episode cap) | YES | Re-evaluate (Phase-III+) |
| `E_max × O_max ≤ 32` (decomposition) | **NO** (R23 removed from contract) | YES (if re-required) | **Only with snapshot freeze** (Case C trigger) |
| Per-episode lineage tracking (handle) | partial (handle exists) | YES (if explicit lineage required) | Re-evaluate |

**Phase-II objective inventory result**: **No concrete objective currently requires
Episode layer**. All current requirements are satisfied by the Phase-I obligation
table alone.

---

## R27-2 — Concrete use-case test

### A. Diagnostics

**Question**: Is there a product requirement to display "which quarantine/recovery
episode this recovery request belongs to"?

**R27 verdict**: No such display exists in the current `ConvoPeq.md` UI components
(`IRAdvancedSettingsComponent`, `ConvolverSettingsComponent`, etc.). The
`RecoveryIntentHandler` does not surface any episode identifier. There is **no
explicit product feature** that requires `RecoveryEpisodeId` to be displayed.

**Verdict**: Episode not needed for diagnostics.

### B. Metrics

**Question**: Are episode-level metrics (start count, end count, per-episode
obligation count, lifetime, retry count) required as product metrics?

**R27 verdict**: Current telemetry counters (R21 §D18.3, R5-8) are
per-**obligation** counters, not per-episode. The product spec (checked in
`doc/work88/` and `evidence/D105-R*`) does not request episode-level metrics.
The 40/40 test suite has no episode-level metric assertion.

**Verdict**: Episode not needed for metrics.

### C. Failure isolation

**Question**: Is there a requirement to "treat multiple obligations in the same
episode as a unit for failure handling"?

**R27 verdict**: Current `markTransientFailure` and `resolveRecoveryObligation`
operate per-**obligation** by `obligationId`. There is no product requirement
that demands episode-level grouping of failures. The R18/R20 model treats
each obligation independently; cross-obligation effects are limited to
`RecoveryIntentQueue` overflow → defer-to-None (R5-10).

**Verdict**: Episode not needed for failure isolation.

### D. Capacity

**Question**: Are `E_max` and `O_max` required to be controlled independently?

**R27 verdict**: The only capacity invariant in the current Phase-I contract is
`liveCount_ ≤ 32` (R23 §D22.2, R24 §3). R23 **removed** the `E_max × O_max ≤ 32`
claim from the contract because:
- `E_max` = 256 (quarantineActiveFlags_) is **separate** from `liveCount_`
- `O_max` ≥ 2 (R3 counterexample) is **unbounded** without freeze

There is **no external product requirement** that demands independent `E_max`
and `O_max` control. The current `liveCount_ ≤ 32` is the single capacity
invariant that protects the system.

**Verdict**: Episode not needed for capacity.

### E. Future feature

**Question**: Is there a concrete future feature specification that requires Episode
abstraction?

**R27 verdict**: Search of `doc/work88/` (REPAIR_PLAN2, I3, I4, D2, D8, etc.) and
the `ConvoPeq.md` UI components for any mention of "episode" in the future-feature
context:
- `REPAIR_PLAN2-dash2.md:1882-1883` — `PendingRecoveryAdmission` is described as
  episode-less, with `retry=true` transitioning `Building → DurablePending` (no episode)
- `REPAIR_PLAN2.md:3435` — discussion of MPSC for `recoveryIntentQueue_` mentions
  "different producer contexts" but **not** episode decomposition
- `D104_D18_Final_Contract_Alignment_Audit.md` — supersession discussion
  (which is **deferred** to Phase-II, not episode-required)
- No product spec mentions `RecoveryEpisodeId` in a concrete feature context

The future-feature landscape does not have a **concrete** Episode-using feature.

**Verdict**: No concrete future feature requires Episode.

### R27-2 conclusion

All five concrete use-cases evaluate to **"Episode not needed"**. The R27
"concrete use-case test" is **negative** for Episode implementation.

---

## R27-3 — Episode abstraction necessity proof

### Step-by-step for each candidate objective

```
Objective X
    ↓
Is it achievable with the obligation table alone?
    ↓
YES → Episode not needed
NO
    ↓
Is EpisodeId alone sufficient?
    ↓
NO → Additional design required
YES
    ↓
Is there a concrete Phase-II need that justifies implementation?
    ↓
NO → DEFER
YES
    ↓
PROCEED with design
```

### Application of the test

For each candidate Episode-required objective from R27-1:

| Objective | Achievable with table? | EpisodeId alone sufficient? | Phase-II need? | Verdict |
|---|---|---|---|---|
| Per-episode telemetry | NOT YET (current is per-obligation) | YES (could add `recoveryEpisode*Count_`) | **NO** (no product spec) | DEFER |
| Per-episode diagnostics | NOT YET | YES (could add `episodeId` field for debug logs) | **NO** | DEFER |
| Per-episode quota | NO | NO (need episode registry + admission gate) | **NO** (no product spec) | DEFER |
| `E_max × O_max ≤ 32` | NO | **NO** (R26 proved: requires snapshot freeze + registry) | **NO** (removed in R23, not re-required) | DEFER |
| Per-episode lineage | partial (handle exists) | partial (would need `episodeId` field) | **NO** | DEFER |

**All candidates evaluate to "DEFER"** because no concrete Phase-II need justifies
implementation cost.

### The "EpisodeId for diagnostics" trap

R27 explicitly rejects the circular reasoning: "EpisodeId would be useful for
debugging, therefore we should add it." R25/R26 already confirmed that
**EpisodeId is identity, not capacity**. Adding it for diagnostics would
require:

1. New field on `LogicalRecoveryObligation`
2. Extended `CoalesceIdentity` (3-tuple instead of 2-tuple)
3. Episode birth policy decision (P1/P2/P3)
4. Migration / rollback consideration (R26 confirmed rollback is **High** cost)

None of this is justified by a concrete Phase-II objective. The
"diagnostics would be nice" argument does not qualify as a Phase-II requirement.

---

## R27-4 — Capacity objective isolation (liveCount_ vs E×O)

### Two invariants must not be confused

| Invariant | Source | R27 status |
|---|---|---|
| `liveCount_ ≤ 32` | R23 §D22.2, R5-8, R25 §D22.3 | **ACTIVE** in Phase-I contract |
| `E_max × O_max ≤ 32` | D19.3 (Phase-II Future Design) | **REMOVED** in R23, deferred to Phase-II |

### Why these are separate

- `liveCount_ ≤ 32` is enforced by the `tryInsert` capacity gate (h:355-356)
  in the `RecoveryAdmissionTable`. It is **runtime-observable** (the table's
  `liveCount_` atomic counter) and **runtime-enforced** (capacity is template
  parameter `kCapacity = 32`).
- `E_max × O_max ≤ 32` would require:
  1. `E_max` to be a code-derived finite number (not derivable from runtime
     state without an episode registry)
  2. `O_max` to be a code-derived finite number (not derivable without
     snapshot freeze)
  3. Their product to be ≤ 32

Neither `E_max` nor `O_max` is currently code-derived finite (R26).

### What "external product requirement" means here

R23 §D19.3 / D22.2 originally proposed `E_max × O_max ≤ 32` as a design
hypothesis. R2/R3 audits proved it is unsatisfiable. R23 **removed it** from the
Phase-I contract. R25 deferred the episode decomposition to Phase-II. R27 asks:

> Is there a **new** external product spec that requires `E_max × O_max ≤ 32`?

The answer is **NO**:
- `doc/work88/` (REPAIR_PLAN2, I3, I4, etc.) does not contain a new spec
  requiring episode decomposition
- `ConvoPeq.md` UI components have no episode-related requirements
- The D105 audit chain (R20-R26) has not surfaced any product spec that needs
  this invariant

**R27 verdict**: No new product spec. `E_max × O_max ≤ 32` remains **removed**
from the contract. `liveCount_ ≤ 32` is the single capacity invariant.

---

## R27-5 — Snapshot semantics audit

### Question: does "same Episode" for snapshot A→B have product meaning?

The current snapshot semantics:

1. `currentBuildSnapshot_` is mutable (updated by RebuildThread on publish
   commit)
2. `getCurrentBuildSnapshotForRecovery()` returns a **value copy** of the
   current snapshot at quarantine time
3. Between two re-quarantines of the same handle S, the snapshot can change

**Concrete trace** (R3 §4):
```
T1: H quarantine → intent_A(S, snapshot_A) → obligation_A
T1+ε: H publish → currentBuildSnapshot_ = snapshot_B
T2: H re-quarantine → intent_B(S, snapshot_B) → obligation_B
```

Two distinct obligations (A and B) for the same handle H, with different
targets (snapshot_A ≠ snapshot_B).

### "Same Episode" for A and B — does it have product meaning?

**R27 verdict**: There is **no product spec** that requires A and B to be in the
"same Episode". The R5-9 coalesce by `(handle, target)` already gives them
**separate obligation identities**. The only thing A and B "share" is the handle H
itself.

If we were to call A and B "the same Episode" (R25 P1 default), the consequence
is:
- `O_max` of that episode = **2** (R3 counterexample)
- No additional semantics
- No new invariant proven

So the grouping "A and B are the same Episode" has **no functional or invariant
meaning**. It's a label without content.

**R27 verdict**: "Same Episode for snapshot A→B" has **no product meaning**.
The grouping is **decorative**, not functional.

---

## R27-6 — Option B demoted to candidate design

### B without freeze = identity/diagnostics candidate

```cpp
struct LogicalRecoveryObligation {
    ...
    uint64_t episodeId;  // NEW: identifier for diagnostic grouping
    ...
};

struct CoalesceIdentity {
    DSPHandle handle;
    SemanticRecoveryTarget target;
    uint64_t episodeId;  // NEW (3rd key)
    ...
};
```

**Status**: identity-only. Does **not** improve capacity proof. Rollback is
**High** cost (cannot drop `episodeId` field on live obligations without false
coalesce risk). R26 verdict: **CONDITIONAL GO (DEFER to Phase-III+)**.

### B with freeze = capacity-contract candidate

```cpp
struct LogicalRecoveryObligation {
    ...
    uint64_t episodeId;             // NEW
    RuntimeBuildSnapshot frozenAt;  // NEW: immutable per episode
    ...
};

RuntimeBuildSnapshot freezeSnapshot() {
    RuntimeBuildSnapshot s = currentBuildSnapshot_.load();
    return s;  // marked immutable via interface (no setter)
}

// On episode birth:
RuntimeBuildSnapshot frozen = freezeSnapshot();
uint64_t episodeId = nextRecoveryEpisodeId_.fetch_add(1);
// On episode close:
episodeId 0 = closed;  // episode live = 0
```

**Status**: capacity-contract candidate. Requires:
1. **Snapshot freeze**: `currentBuildSnapshot_` becomes immutable per episode
   (or per-handle per-episode). Breaks current `enqueuePublicationIntentForRuntimeCommit`
   semantics (publish needs to update snapshot).
2. **Episode registry**: separate counter + admission gate for episode count.
3. **Snapshot drift proof**: `O_max = 1` per episode when freeze is in effect.

**R27 verdict**: B with freeze would restore the `E × O ≤ 32` capacity proof. But
this requires **product-side snapshot freeze** (no user can update parameters
during an active episode), which is **a product design decision**, not a Runtime
implementation choice. No such product decision exists in current `doc/work88/`.

---

## R27-7 — Implementation trigger table

R27 establishes the following trigger table. If **any** of these triggers fires,
R28 should re-evaluate the Episode layer. Until then, R20 + R23 closure is the
final state.

| Trigger condition | Episode layer response |
|---|---|
| `liveCount_ ≤ 32` becomes insufficient (capacity violation observed) | Re-audit `RecoveryAdmissionTable` capacity; expand `kMaxLogicalRecoveryObligations` if needed |
| Per-obligation diagnostics become insufficient (debug log requires "this obligation's episode") | Consider **identity-only** `episodeId` as diagnostic label, **NOT** for capacity |
| `E_max × O_max ≤ 32` is added back to the I4 contract by an external product spec | **MUST** use B with freeze (Policy P3); re-audit `O_max` and `E_max` from scratch |
| Snapshot freeze is added to product spec (no parameter changes during episode) | **MUST** use B with freeze (Policy P3); re-audit `O_max = 1` provability |
| Product needs per-episode quota | Re-audit Episode registry + admission gate; consider C2 (per-episode bounded) |
| External API exposes `episodeId` (e.g., REST endpoint, logging format) | Consider identity-only `episodeId` as labeling field; **NO** capacity re-introduction |
| No concrete trigger fires | **Episode layer remains permanently DEFERRED** |

**Until any trigger fires, R27 maintains the status quo**: R20-finalized Runtime +
R23 closure + R25 deferred + R26 Option B conditionally sound (but not implemented).

---

## R27-8 — Closure chain confirmation

R27 confirms the D105 audit chain:

```
R20 (production runtime finalized)
  ↓
R23 (obligation-table model closure)
  ↓
R24 (post-closure verification: 23/23 GO conditions)
  ↓
R25 (Episode layer deferred: Option B recommended, capacity NOT re-proven)
  ↓
R26 (Option B internal contradiction: identity vs capacity separation confirmed)
  ↓
R27 (this audit: no concrete Phase-II objective requires Episode)
```

### Status at R27

| Audit | Status | Output |
|---|---|---|
| R20 | PASS (production runtime) | Runtime finalized |
| R23 | PASS | I4 closure |
| R24 | PASS | 23/23 GO conditions |
| R25 | CONDITIONAL GO (DEFER) | Episode layer deferred to Phase-III+ |
| R26 | CONDITIONAL GO (DEFER) | Option B identity-only soundness confirmed |
| R27 | **Case A: DEFER permanently** | No concrete Phase-II objective requires Episode |

---

## R27 GO-condition verification

| # | Condition | Status | Evidence |
|---|---|---|---|
| 1 | Phase-II objective enumerated | ✅ | R27-1 (12 candidates, all "obligation table can") |
| 2 | Episode necessity proved per objective | ✅ | R27-3 (all candidates evaluate to "DEFER") |
| 3 | EpisodeId alone ≠ capacity proof | ✅ | R26 + R27-3 (verified) |
| 4 | `liveCount_ ≤ 32` and `E×O≤32` separated | ✅ | R27-4 (clear distinction) |
| 5 | Snapshot drift not over-interpreted | ✅ | R27-5 (no product meaning for "same Episode") |
| 6 | No Episode semantics without product spec | ✅ | R27-2 (5 use cases all negative) |
| 7 | Option B without freeze NOT implemented | ✅ | R27-6 (demoted to candidate) |
| 8 | `E×O≤32` NOT restored to I4 | ✅ | R27-4 (removed by R23, not re-required) |
| 9 | Implementation trigger documented | ✅ | R27-7 (7 trigger conditions) |
| 10 | Phase-II objective absent → Episode DEFER permanently | ✅ | R27-2 / R27-3 (all negative) |

**R27: All 10 GO conditions satisfied. Case A verdict confirmed.**

---

## R27 final verdict: **Case A — No concrete Phase-II objective → Episode layer permanently DEFER**

### Decision

The current Phase-I obligation-table model (R20-finalized) is **sufficient** for
all current requirements. No concrete Phase-II objective exists that:

1. Cannot be satisfied by the Phase-I model
2. Has an external product spec requiring Episode abstraction
3. Justifies the implementation cost of adding `RecoveryEpisodeId` + extended
   `CoalesceIdentity` + episode birth policy + rollback complexity

**R27 verdict: Episode layer is DEFERRED to Phase-III+ pending a concrete trigger.**

### What R27 does NOT do

- R27 does **not** authorize any Runtime modification
- R27 does **not** authorize any I4 modification
- R27 does **not** recommend implementing Option B (with or without freeze)
- R27 does **not** introduce new invariants or new authority

### What R27 DOES establish

- R27 confirms R20 + R23 closure is the **final stable state**
- R27 documents the **trigger conditions** for any future re-evaluation
- R27 confirms that the **diagnostic-only Option B** (without freeze) has **no
  functional advantage** over the current obligation table for any product-level
  requirement
- R27 confirms that **`E_max × O_max ≤ 32` re-introduction** requires BOTH snapshot
  freeze AND episode registry + admission gate (R26 finding), which is a **product
  design decision** not a Runtime implementation choice

### R28 (if scheduled) trigger conditions

R28 should be initiated only when:
1. A new product spec introduces an explicit Episode-level requirement, OR
2. A capacity violation is observed in production that requires decomposition, OR
3. A new Phase-II architecture review is commissioned by the project owner

Until any of these triggers, the D105 Episode branch is **closed** and
**R20 + R23 + R24 + R25 + R26 + R27 = final state**.

### File references (read-only, no source modification)

| File | Role |
|---|---|
| `ConvoPeq.md` (2026-08-27) | runtime source baseline |
| `src/audioengine/AudioEngine.h:4843-4862` | `currentBuildSnapshot_` field + `getCurrentBuildSnapshotForRecovery` |
| `src/audioengine/AudioEngine.Commit.cpp:782-805` | `enqueuePublicationIntentForRuntimeCommit` writer (RebuildThread) |
| `src/audioengine/ISRRuntimePublicationCoordinator.h:325-396` | `kMaxLogicalRecoveryObligations`, `RecoveryAdmissionTable` |
| `src/audioengine/ISRRuntimePublicationCoordinator.h:262-267` | `CoalesceIdentity` (current = `{handle, target}`) |
| `src/audioengine/ISRRuntimePublicationCoordinator.h:355-356` | `tryInsert` capacity gate (32) |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp:820-942` | `submitRecoveryRequest` admission path |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp:955-1024` | `resolveRecoveryObligation` + `markTransientFailure` |
| `src/audioengine/ISRDSPQuarantine.h:40-72` | `DSPQuarantineManager::quarantineHandle` + `quarantineActiveFlags_[256]` |
| `evidence/D105-R20_REJECTEDNOTFINALIZED_CENTRALIZATION.md` | R20 runtime finalization |
| `evidence/D105-R21_I4_CONTRACT_AMENDMENT.md` | R21 RetryExhaustion / conservation / MARK-TRANSIENT-FAILURE |
| `evidence/D105-R22_FINAL_CONSISTENCY_AUDIT.md` | R22 NO-GO discovery |
| `evidence/D105-R23_OBLIGATION_TABLE_MODEL_CONVERGENCE.md` | R23 Path A convergence + I4 closure |
| `evidence/D105-R24_I4_POST_CLOSURE_VERIFICATION.md` | R24 final verification (23/23 GO) |
| `evidence/D105-R25_PHASE_II_EPISODE_MIGRATION_AUDIT.md` | R25 Option B recommendation (identity-only) |
| `evidence/D105-R26_OPTION_B_SOUNDNESS_AUDIT.md` | R26 internal contradiction (identity ≠ capacity) |
| `doc/work88/REPAIR_PLAN2*.md` | existing future-task list (MPSC queue, etc.) — separate from Episode audit |
| `doc/work88/I3_DESIGN_CONTRACT.md` / `I4_DESIGN_CONTRACT.md` | I4 contract (R23-amended post-closure) |

**No source files modified. No I4 files modified. No tests added.** R27 is a
requirements-level audit that establishes Case A (no concrete Phase-II objective
requires Episode implementation as of R27).

---

## R27 closure

The D105 audit chain (R15 → R20 → R23 → R24 → R25 → R26 → R27) is now **closed**
at the Episode branch:

- R20: production runtime finalized ✅
- R23: obligation-table model closed ✅
- R24: post-closure verified (23/23 GO) ✅
- R25: Episode layer deferred to Phase-III+ ✅
- R26: Option B internal contradiction proved (identity ≠ capacity) ✅
- **R27: No concrete Phase-II objective requires Episode implementation → Episode DEFERRED permanently** ✅

R28 should only be initiated by an explicit product-side trigger documented in
R27-7. Until then, R20 + R23 closure = final state, and the project should
focus on other branches (e.g., MPSC queue migration from REPAIR_PLAN2) that have
concrete product-level justification.
