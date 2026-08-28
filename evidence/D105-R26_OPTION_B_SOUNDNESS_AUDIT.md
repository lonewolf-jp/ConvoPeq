# D105-R26 — Option B Pre-Implementation Soundness & Migration-Boundary Audit

**Status:** **R26 verdict: CONDITIONAL GO (DEFER to Phase-III+)**
**Production source changes:** 0
**I4 changes:** 0
**Test changes:** 0

R26 establishes that **R25's recommended Option B (EpisodeId + obligation association, without freeze) is sound as an identity/diagnostic layer, but it does NOT restore the `E_max × O_max ≤ 32` capacity invariant** to a code-provable form. The recommended action is:

1. Defer Option B implementation to Phase-III+
2. If `E × O ≤ 32` ever becomes a hard requirement again, use **Option B with snapshot freeze** (which IS provable)
3. Until then, R20-finalized runtime + R23 closure + R25 deferred Episode layer = final stable state

---

## R26-1 — Option B internal contradiction (the central audit)

### Question: does `RecoveryEpisodeId` alone bound `O_max` / `E_max` / `E × O ≤ 32`?

R25 stated: "Without snapshot freeze: O_max is bounded by 'snapshot drift rate × episode
lifetime' (still unbounded in principle)." R26 evaluates whether this counts as a code
proof or a hand-wave.

**R26 verdict**: A "rate × lifetime" bound is **NOT a code-proven finite bound** for these reasons:
1. `currentBuildSnapshot_` is mutable at any time (NonRT Timer or Commit)
2. `episode lifetime` has no fixed upper bound (depends on admission rate, coalesce
   pattern, and shutdown timing)
3. The "rate" itself depends on user activity (UI parameter changes)
4. Therefore `O_max` is **unbounded** in the current code structure, even with `RecoveryEpisodeId`

**Answer to the 4 questions**:
1. `EpisodeId` exists alone → `O_max` finite? **NO** (snapshot drift can produce ≥ 2 obligations
   in one episode lifetime)
2. `EpisodeId` exists alone → `E_max` finite? **NO** (no episode registry; `E_max` is not a runtime
   concept)
3. `EpisodeId` exists alone → contradicts `liveCount_ ≤ 32`? **NO** (the table enforces 32
   regardless of episode layer)
4. `E × O ≤ 32` re-introduced by adding `EpisodeId` alone? **NO** (`O_max` and `E_max` are
   still unbounded without freeze + registry)

**R26 verdict**: `RecoveryEpisodeId` is an **identity**, not a **capacity bound**. R25's
claim that Option B is "recommended" is correct as a design choice, but Option B does
**not** improve the capacity proof compared to Phase-I closure.

---

## R26-2 — Episode birth policy re-verification

### Sequence: A → resolve → re-quarantine → B

Current code (Phase-I) flow:

```
H quarantine
  ↓ submitRecoveryIntent
  ↓ submitRecoveryRequest
  ↓ CoalesceIdentity = (H, target_A)
  ↓ findByKey(cid) — npos (empty table)
  ↓ tryInsert(cid) — L++
  ↓ obligation A: id=1, episodeId=N/A, delivery=Transport
  ↓ obligation A publish
  ↓ onPublishCommitted → resolveRecoveryObligation(id, Published) → L--
  ↓ discardRecoveryRequestsOnShutdown (when shutdown) — drain queue
  ↓ quarantineActiveFlags_[H] = false (reclaim)

H re-quarantine (later)
  ↓ submitRecoveryIntent
  ↓ submitRecoveryRequest
  ↓ CoalesceIdentity = (H, target_B) [target_B ≠ target_A → different cid]
  ↓ findByKey(cid) — npos
  ↓ tryInsert(cid) — L++
  ↓ obligation B: id=2, episodeId=N/A, delivery=Transport
  ↓ obligation B publish
  ↓ resolveRecoveryObligation(id, Published) → L--
```

**R26 verdict on Episode birth policy (Option B)**:

If Option B is implemented as R25 proposed ("EpisodeId born on first tryInsert for a handle"):

| handle | episode | target | Result (Option B) |
|---|---|---|---|
| same | same | same | coalesce → 1 obligation, 1 episodeId |
| same | same | different | new obligation, **same episodeId** (no new episode) |
| same | different | same | new obligation, **new episodeId** |
| same | different | different | new obligation, **new episodeId** |
| different | same | same | new obligation, **new episodeId** (handle is a new lineage) |
| different | same | different | new obligation, **new episodeId** |
| different | different | same | new obligation, **new episodeId** |
| different | different | different | new obligation, **new episodeId** |

**Critical question**: When `handle` is the same but `target` is different, is it the same episode?

**R26 verdict**: Same-episode (snapshot drift case) is **the critical case R3 warned about**.
Under Option B's "episodeId born on first tryInsert for a handle" policy, this case
keeps both obligations in the same episode. Therefore:
- `O_max` for this single episode is **≥ 2** (R3 counterexample still applies)
- The "episode" abstraction **does not prevent** the unbounded-multi-target-per-episode case

The alternative policy — "different target → new episodeId" — would resolve the R3
counterexample but would make `O_max = 1` (each episode has exactly 1 obligation).
This is essentially a **freeze** of the snapshot at episode start. Without this freeze,
Option B does not bound `O_max`.

**R26's recommended policy decision** (must be made before implementation):

- **Policy P1**: same handle + different target → same episodeId (R25 default)
  - Pros: simple, single episode per handle
  - Cons: `O_max ≥ 2`, same as current code
- **Policy P2**: same handle + different target → new episodeId
  - Pros: `O_max = 1` per episode, matches D18.2 spec
  - Cons: episode count grows over handle lifetime
- **Policy P3**: snapshot freeze at episode birth
  - Pros: `O_max = 1` provable, matches D18.2 exactly
  - Cons: needs `currentBuildSnapshot_` mutation protection

**R26 verdict**: Without policy decision, R25 Option B is **underspecified** on this
critical point. P1 is R25's default but is **equivalent to current Phase-I semantics** (no
new bound proven). P2 or P3 are needed to make Option B meaningful as a capacity
improvement.

---

## R26-3 — RecoveryEpisodeId allocator ownership audit

### Authority Singularization principle

R25 proposed:
```
struct LogicalRecoveryObligation {
    ...
    uint64_t episodeId;  // NEW field (allocated on tryInsert)
    ...
};
```

**Authority question**: who allocates `episodeId`?

| Option | Allocator | Owner | Conflict? |
|---|---|---|---|
| **OA1** | `RecoveryAdmissionTable::tryInsert` | Table (extends existing) | **NO CONFLICT** — single owner |
| **OA2** | `RuntimeIntentCoordinator::nextRecoveryEpisodeId_` atomic counter | Coordinator (separate from Table) | **POTENTIAL CONFLICT** — Table is single owner of obligations, Coordinator is single owner of episodes |
| **OA3** | `submitRecoveryRequest` (call site) | Caller (No owner) | **CONFLICT** — no single owner; episodeId is computed ad-hoc |

**R26 verdict**: **OA1 is correct**. The `RecoveryAdmissionTable` is the single owner of
both obligation identity AND episode identity. The `tryInsert` function would:
- Determine episodeId from existing slots with same handle (findEpisode helper)
- Allocate new episodeId (atomic counter) if no live slots for that handle
- Store episodeId in the slot

This preserves Authority Singularization (R25 §D29.7). The episodeId is **derived
from table state**, not from a separate registry.

OA2 would create two owners: Table owns obligation count, Coordinator owns episode
count. This is **rejected** — same as R25 §D29.7 violation.

OA3 is **rejected** — no single owner means the episodeId is undefined when multiple
threads call `submitRecoveryRequest` (race condition).

---

## R26-4 — findByKey extension coalesce semantic audit

### Truth table for Option B's extended findByKey

Current Phase-I semantics: `CoalesceIdentity = (handle, target)`. Two obligations
with the same `(handle, target)` coalesce; different target or different handle
produces new obligation.

Option B extension: `CoalesceIdentity = (handle, episodeId, target)`. Adding
`episodeId` to the key.

| handle | episode | target | Current Phase-I | Option B | Same result? |
|---|---|---|---|---|---|
| same | same | same | coalesce | coalesce | YES (same identity preserved) |
| same | same | different | new obligation | new obligation | YES |
| same | different | same | new obligation | new obligation | YES (since Phase-I has no episode) |
| same | different | different | new obligation | new obligation | YES |
| different | any | any | new obligation | new obligation | YES |

**R26 verdict**: Option B's findByKey extension **preserves all current Phase-I
semantics** when `episodeId` is consistent. The only behavioral change is when
`episodeId` would be **reused** across re-quarantines — which Option B does not
intend (different episodeId for re-quarantine under policy P2 or P3).

Under policy P1 (R25 default), same handle with different target keeps the same
episodeId → coalesce doesn't happen (different target) but the episode is "shared".
The semantics is **identical to current Phase-I** from the obligation table's
perspective.

**Conclusion**: Option B's findByKey extension is **semantically safe** under
all 3 policies (P1, P2, P3). The obligation table's behavior is unchanged for the
coalesce + capacity path.

---

## R26-5 — O_max finiteness (3-way classification)

R25 admitted that Option B without freeze leaves `O_max` bounded by "snapshot
drift rate × episode lifetime". R26 evaluates this as a 3-way choice.

### Choice A: O_max finite (code-proven)

**Required**:
- Snapshot freeze at episode birth (immutable `currentBuildSnapshot_`)
- AND episode lifetime bound (closure on some finite condition)
- AND code-proven (not rate argument)

**Status**: NOT achievable in current code without Phase-II implementation work.

### Choice B: O_max unbounded (acknowledged)

**Status**: matches current code structure. `O_max` is **unbounded** in Phase-I
runtime because snapshot drift can produce multiple distinct targets per handle
without episode closure. R5-8 and R7 already proved this as the obligation
identity is per-LogicalRecoveryObligation (not per-handle).

### Choice C: O_max not needed for Phase-II

**Status**: If `E × O ≤ 32` is removed from Phase-II contract, then `O_max` doesn't
need to be proven finite. R23 already removed this claim. R25 deferred it to
Phase-II but did not restore it.

### R26 recommendation: **Choice C (O_max not needed for Phase-II as currently scoped)**

If `E × O ≤ 32` is **not** a Phase-II contract requirement (which R23 decided), then
`O_max` finite-ness is irrelevant. Option B can be implemented **without** the
O_max finite proof — the "snapshot drift rate × lifetime" hand-wave is **not** a
contractual issue if `E × O ≤ 32` is not in the contract.

**However**, if `E × O ≤ 32` **is** a Phase-II contract requirement (e.g., to
bound memory or latency), then `O_max` finite MUST be proven. The only way to prove
it in current code is **Option B with snapshot freeze** (Policy P3).

**R26 final classification**: `O_max` is **unbounded** in the absence of explicit
snapshot freeze. R25's "rate × lifetime" framing is a hand-wave and should not be
relied upon for any capacity claim.

---

## R26-6 — E_max bound and registry capacity

### Episode registry design (Option B without registry — using table state)

R25 Option B uses the obligation table itself as the episode registry:
- `episodeId` is a field on each obligation slot
- "open episodes" = distinct `episodeId` values across Live obligations
- `closed episodes` = `episodeId` values whose all obligations are terminal

**E_max derivation**: in this model, `E_max` is **not bounded** by any code structure.
It is bounded by `liveCount_` indirectly (each open episode has at least 1 obligation,
so E_max ≤ liveCount_ ≤ 32). But:

`liveCount_ ≤ 32` ⇒ `E_max ≤ 32` (since each episode has ≥ 1 obligation)

This is **valid as a derived bound** but **not as a designed constant**. The 32 in
`E_max ≤ 32` is the same 32 as `liveCount_ ≤ 32` — not a new constant.

### Episode allocation failure semantics

If Option B uses `nextRecoveryEpisodeId_` (atomic counter):
- Counter overflow: `uint64_t` is large enough; no realistic risk
- Episode allocation never fails (it's an atomic increment)

If Option B derives episodeId from existing handle:
- `findEpisode(handle)` returns existing episodeId if Live obligations exist
- Returns 0 (or "no episode") if all obligations of handle are terminal
- New episodeId allocated on first tryInsert after all closed

**Edge case**: What if `tryInsert` succeeds (obligation Live) but episodeId
allocation fails? The two must be **atomic**:
- Option B + tryInsert: `tryInsert` allocates episodeId as part of slot allocation
  (single atomic, no failure)
- Option B + separate episodeId counter: requires CAS or rollback

**R26 verdict**: Option B with **OA1 (table-owned episodeId)** is the cleanest
because episodeId allocation is bundled with slot allocation. The episodeId
allocation **cannot fail** after `tryInsert` succeeds. The "episodeId allocation
failure after tryInsert success" scenario is **avoided by construction**.

---

## R26-7 — E × O ≤ 32 re-introduction conditions

R26 explicitly defines the conditions under which `E × O ≤ 32` can be re-introduced
as a Phase-II contract.

### Required for re-introduction

All **3** of the following must hold:

1. **`E_max` is code-derived finite bound** (not "registry capacity" hand-wave)
2. **`O_max` is code-derived finite bound** (not "snapshot drift rate" hand-wave)
3. **`E_max × O_max ≤ 32` is a provable product**

### What "code-derived finite" means

- `E_max` derived from runtime state: yes, if episode registry has hard cap
  (`registry.capacity ≤ K`) and admission gate enforces it
- `E_max` derived from "obligation count divided by obligations per episode": no,
  this is not deterministic
- `E_max` derived from "lifecycle-based closure": no, closure depends on snapshot drift
  and admission rate

### What "O_max finite" means

- `O_max` derived from snapshot freeze: yes, if `currentBuildSnapshot_` is
  immutable per episode
- `O_max` derived from "1 obligation per handle at a time": no, because
  obligations persist after terminal resolution (slot reuse)

### Concrete condition

`E × O ≤ 32` is **only re-introducible** when BOTH:
- `currentBuildSnapshot_` is **frozen at episode birth** (immutable per episode)
  — this guarantees `O_max = 1` per episode
- An episode registry exists with hard capacity `registry.capacity` and admission
  gate — this guarantees `E_max = registry.capacity`

The combination gives `E × O ≤ registry.capacity × 1 ≤ 32` ⇒ `registry.capacity ≤ 32`.

**Important**: This is **NOT** "EpisodeId alone provides `E × O ≤ 32`". Both freeze
AND registry are needed.

### R26's clarification

> "`RecoveryEpisodeId` を導入しても目的を満たさず、現 Phase-I runtime に対する変更リスクが利益を上回る" (R25 final verdict)

**Refined**: The statement is **correct only when** `E × O ≤ 32` is **not** a Phase-II
contract requirement. If `E × O ≤ 32` **is** a Phase-II requirement, EpisodeId
**alone** does not satisfy it — the freeze + registry combination does.

---

## R26-8 — Phase-I compatibility audit (32, 256, 1, 256)

| Phase-I invariant | R26 verification |
|---|---|
| `kMaxLogicalRecoveryObligations = 32` | unchanged (OA1 keeps Table as single owner, table template parameter unchanged) |
| `RecoveryAdmissionTable<32>` | unchanged (same template instantiation at h:325) |
| `liveCount_ ≤ 32` | unchanged (capacity gate h:355-356, single atomic counter) |
| `RecoveryIntentQueue<256>` | unchanged (lock-free SPSC ring) |
| `PendingRecoveryAdmission` (1 slot) | unchanged (single-slot struct, no EpisodeId field) |
| `quarantineActiveFlags_[256]` | unchanged (DSP-level, no EpisodeId) |
| `RecoveryOutcome` enum (4 values) | unchanged (no new outcome) |
| `markTransientFailure` (R18) | unchanged (ΔL=0, counter+1, P-B delivery=None) |
| `K=4` (kMaxObligationConsecutiveFailures) | unchanged |
| `CoalesceIdentity = {handle, target}` | **EXTENDED** to `{handle, episodeId, target}` (Option B only; Phase-I preserved) |
| `recoveryAdmissionPending_` (durable flag) | unchanged |
| `nextRecoveryIntentId_` (per-attempt id) | unchanged |
| `pendingRecoveryAdmission_.recoveryObligationId` | unchanged |

**R26 verdict**: **All Phase-I invariants preserved** under Option B. The only
runtime change is the `CoalesceIdentity` key (extended with `episodeId`) and the
slot field (new `episodeId` field). No capacity change, no race change, no
ownership change.

---

## R26-9 — Transient failure / RetryExhaustion / Shutdown compatibility

### Transient failure (Live → Live)

Current: `markTransientFailure` increments `consecutiveFailureCount`, sets
`delivery=None`, and (if counter ≥ K) resolves to `ResolvedFailed`.

Under Option B: `markTransientFailure` still operates on a single obligation
identified by `obligationId`. The `episodeId` field is unchanged. The episode
closure state is **not** affected by `RetryExhaustion` (an episode stays open
if other obligations of the same episode are still Live).

**R26 verdict**: `RetryExhaustion ≠ EpisodeClosure` is preserved. An obligation
reaching `RetryExhaustion` does not close the episode (it only closes the obligation
slot). The other obligations in the same episode continue.

### Shutdown

Current: `discardRecoveryRequestsOnShutdown` resolves all Live obligations to
`ShutdownDiscarded`. No episode-level state is needed.

Under Option B: same. The `recoveryObligationShutdownDiscardCount_` increments per
obligation. The episodeId is irrelevant to shutdown (every Live obligation becomes
`ShutdownDiscarded` regardless of episode).

**R26 verdict**: Shutdown is **unaffected** by Episode layer.

### Admission gate

Current: `state_==ShuttingDown` blocks new admissions.

Under Option B: same. The `episodeId` is determined **inside** the admission
function, after the shutdown check. No new gate is needed.

**R26 verdict**: All 3 R21/R24 invariants (transient failure Live→Live, RetryExhaustion ≠
EpisodeClosure, Shutdown state_==ShuttingDown gate) are **preserved unchanged**.

---

## R26-10 — Rollback boundary (Phase-II Option B → Phase-I)

### State migration needed for rollback

| Change | Migration needed? | Cost |
|---|---|---|
| `episodeId` field on `LogicalRecoveryObligation` | **YES** — every Live obligation's field must be dropped | High (table reset) |
| `CoalesceIdentity = (handle, episodeId, target)` | **YES** — reverts to `(handle, target)`; existing coalesce logic re-activates | High |
| `nextRecoveryEpisodeId_` counter | **NO** — can remain (unused after rollback) | None |
| `tryInsert(cid)` extended signature | **YES** — reverts to single-param | High |
| `resolveRecoveryObligation` extended for episode live count | **NO** — episode live count derived, no separate state | None |
| `isFullyDrained` extended for episode iteration | **NO** — iterates table directly | None |
| Snapshot freeze (Option B with freeze) | **VERY HIGH** — undoing freeze is impossible without rebuild | Very High |

**Critical check**: Can a **live** obligation with `episodeId` field lose that
field without identity loss?

- The `episodeId` is part of `CoalesceIdentity`. If removed:
  - `findByKey` falls back to `(handle, target)` matching
  - Current Phase-I semantics: different episodes with same `(handle, target)` would
    now coalesce incorrectly!
  - **Risk**: false coalesce → lost identity

**Conclusion**: Option B without freeze **cannot be safely rolled back** to Phase-I
without resetting the live obligation table. The `episodeId` field's effect on
coalesce is **not reversible** in-place.

R25's "Medium reversibility" claim for Option B is **overly optimistic** in the
context of a live system. R26 corrects this to **High reversibility** in practice
(deployment requires maintenance window).

---

## R26-11 — Final verdict

R26 verdict: **CONDITIONAL GO (DEFER to Phase-III+)**

R26 selects the **CONDITIONAL GO** category from the brief's three options.

| Verdict category | Selection |
|---|---|
| GO (Option B without freeze is sound) | **NO** — `O_max` is not finite, `E × O ≤ 32` is not provable |
| CONDITIONAL GO (Option B is sound as identity/diagnostic layer; capacity invariant NOT re-introduced) | **YES — selected** |
| NO-GO (EpisodeId introduction has more risk than benefit) | **NO** — but conditional on absence of `E × O ≤ 32` requirement |

### Conditions for "CONDITIONAL GO → GO" promotion

1. **EpisodeId introduction does NOT re-introduce `E × O ≤ 32` as a contract**
   (current R23 decision is preserved)
2. **No Phase-I invariant is modified** (R24 verification confirmed)
3. **Migration rollback is feasible** with a maintenance window
4. **Snapshot freeze is NOT used** (current `currentBuildSnapshot_` mutability is preserved)

### Conditions for "CONDITIONAL GO → NO-GO" demotion

1. **`E × O ≤ 32` becomes a hard Phase-II requirement** → must use Option B with freeze
   (Policy P3) instead
2. **Live obligation identity is critical** → cannot drop `episodeId` field on rollback
3. **Production snapshot drift rate is unacceptably high** → `O_max` becomes an
   operational concern even without contract requirement

### R26's recommended next steps (R27+)

R26 does **not** authorize Option B implementation in current R20-finalized runtime.
The recommendation is:

> R27 (if scheduled) should be an **admission logic design** that addresses
> **what concrete Phase-II objective requires Episode abstraction**, not an
> implementation of R25's Option B.

If no Phase-II objective can be articulated, R20 + R23 + R24 closure **remains the
final stable state** and no further Runtime work is needed.

If a Phase-II objective is articulated (e.g., new diagnostic feature, new failure
mode to handle), R27 should re-audit that specific objective against R26's
constraint set:

- Snapshot freeze is required → use Option B with Policy P3
- EpisodeId as identity only → use Option B with Policy P1 (current R25 default)
- `E × O ≤ 32` MUST hold → Option B with **both** P3 (freeze) and registry

### File references (read-only)

| File | Role |
|---|---|
| `ConvoPeq.md` (2026-08-27) | runtime source baseline |
| `src/audioengine/ISRRuntimePublicationCoordinator.h:310-330` | `CoalesceIdentity` struct (current = `{handle, target}`) |
| `src/audioengine/ISRRuntimePublicationCoordinator.h:325-396` | `LogicalRecoveryObligation` struct (no `episodeId` field) |
| `src/audioengine/ISRRuntimePublicationCoordinator.h:320-345` | `RecoveryAdmissionTable` class (table-resolved logic) |
| `src/audioengine/ISRRuntimePublicationCoordinator.h:886` | `kRecoveryIntentQueueCapacity = 256` |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp:820-942` | `submitRecoveryRequest` (coalesce + tryInsert + capacity gate) |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp:955-1024` | `resolveRecoveryObligation` + `markTransientFailure` |
| `src/audioengine/ISRDSPQuarantine.h:40-72` | `DSPQuarantineManager::quarantineHandle` + `quarantineActiveFlags_[]` |
| `src/audioengine/AudioEngine.RebuildDispatch.cpp:1017-1076` | Builder durable retry loop + `markTransientFailure` |
| `evidence/D105-R21_I4_CONTRACT_AMENDMENT.md` | R21 contract (RetryExhaustion, MARK-TRANSIENT-FAILURE) |
| `evidence/D105-R22_FINAL_CONSISTENCY_AUDIT.md` | R22 NO-GO discovery |
| `evidence/D105-R23_OBLIGATION_TABLE_MODEL_CONVERGENCE.md` | R23 Path A convergence |
| `evidence/D105-R24_I4_POST_CLOSURE_VERIFICATION.md` | R24 final verification |
| `evidence/D105-R25_PHASE_II_EPISODE_MIGRATION_AUDIT.md` | R25 Option B recommendation |

**No source files modified. No I4 files modified. No tests added.** R26 is a
read-only soundness audit that re-evaluates R25's recommendation against the
remaining internal contradictions.

R26 establishes that **R25's Option B recommendation is conditionally correct as an
identity layer** but **cannot be promoted to GO** for capacity-invocation reasons
in the current code structure. The final state of D105 audit chain is:

> **R20 (production runtime) → R23 (obligation-table model closure) → R24 (final
> verification) → R25 (Episode layer deferred) → R26 (Option B conditionally sound,
> no implementation)**

— **all read-only audits, no source change after R20**, and **no further Runtime
implementation authorized** without an explicit Phase-II objective that requires
Option B with **snapshot freeze + registry**.

R27 (if scheduled) must articulate the Phase-II objective before any Runtime change.
