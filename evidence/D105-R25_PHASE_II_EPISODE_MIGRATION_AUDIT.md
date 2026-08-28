# D105-R25 — Phase-II Episode Layer Migration Design Audit

**Status:** **R25 PARTIAL PASS** (read-only design audit, **source changes = 0**)
**R25 verdict: Phase-II Episode 層の実装は現時点で推奨しない (DEFER to Phase-III+)**

R25 establishes that:
1. Phase-I obligation-table model (R23/R24 closure) is **fully self-consistent** with current production runtime
2. Episode 層を追加する **必要性** が **R23 で確定した obligation-table model では限定的** であることが確認された
3. 既存 32 capacity は **`liveCount_` 単一 counter** で enforce されており、Episode 層を追加せずに全 Phase-I production invariants が満たされている
4. **Snapshot drift** は Episode なしでも `CoalesceIdentity = {handle, target}` で正しく処理される
5. **admission race** は D23 の CAS なしでも R23 の single `liveCount_` で race-free
6. **Reversibility 評価**: Episode 層を Phase-II で導入する場合の rollback コストは **高い** ため、Phase-II 導入の trigger を **明確化すべき**

R25 の結論: **Phase-II Episode 層の導入は現時点では不要**。将来 `E_max × O_max ≤ 32` の D19 由来分解が必要なシナリオが発生した場合のみ、Option B (EpisodeId + obligation association) を再評価する。

---

## 1. Phase-I invariants frozen (R25-1)

R25 では以下を **Phase-I production contract として絶対に変更しない**:

| Item | Phase-I fixed value | Source |
|---|---|---|
| `kMaxLogicalRecoveryObligations` | **32** | `ISRRuntimePublicationCoordinator.h:325` |
| `RecoveryAdmissionTable` | authoritative logical obligation table | `ISRRuntimePublicationCoordinator.h:931-934` |
| `liveCount_` | single `std::atomic<uint64_t>` logical Live counter | `ISRRuntimePublicationCoordinator.h:380` |
| `Q_max` | **256** | `ISRDSPQuarantine.h:68` (`kMaxSlots`) |
| `L_residency_max` | **257** (transport 256 + durable 1) | `ISRRuntimePublicationCoordinator.h:886` |
| `CoalesceIdentity` | `{handle, SemanticRecoveryTarget}` | `ISRRuntimePublicationCoordinator.h:262-267`, `cpp:836-841` |
| `RetryExhaustion` (= `consecutiveFailureCount == K`) | `K=4` | `ISRRuntimePublicationCoordinator.h:325` (kMaxObligationConsecutiveFailures) |
| transient failure | `Live → Live` (ΔL=0) | `ISRRuntimePublicationCoordinator.cpp:998-1024` |
| Audio Thread ownership | unchanged | (multiple sources) |
| Phase-I closure | `liveCount_: 1 → 0` (logical domain) | `ISRRuntimePublicationCoordinator.h:380, 388-389` |
| obligation terminal set | `{Success, Superseded, ShutdownDiscard, RetryExhaustion}` | `ISRRuntimePublicationCoordinator.h:298-304` |

R25 では Phase-II でこの固定値を再定義しない。

---

## 2. Current runtime → Phase-II gap (R25-2)

### Current runtime flow (Phase-I, before R25)

```
[NonRT: Timer or Commit]
  ↓
AudioEngine::quarantineSlot(slot, gen, reason)             (Threading.cpp:38-65)
  ↓
DSPQuarantineManager::quarantineHandle(slot, gen, reason)  (ISRDSPQuarantine.cpp:17-48)
  ↓ sets quarantineActiveFlags_[slot] = true (atomic, ISR-safe)
  ↓
quarantineResidentCount_  (X6 — DSP-side)
  ↓
submitRecoveryRequest(handle, buildSource, epoch)         (ISRRuntimePublicationCoordinator.cpp:820)
  ├─ shutdown gate (line 824): → return false, recoveryShutdownDiscardCount_++
  ├─ CoalesceIdentity cid = {handle, {3 hashes from buildSource}}
  ├─ redriveDeferredRecoveryObligations()
  ├─ findByKey(cid):
  │   ├─ match (Live && same cid) → coalesce, recoveryCoalescedCount_++, return true
  │   └─ no match → tryInsert(cid)
  │       ├─ nullopt (L=32) → recoveryCapacityExhaustedCount_++, return false
  │       └─ ok → liveCount_++, set handle/epoch/buildSource
  ├─ if recoveryIntentQueue_.push(intent) ok → delivery=Transport, return true
  └─ durable fallback:
      ├─ if pendingRecoveryAdmission_ already held by OTHER id → delivery=None (defer), recoveryRetryDeferredCount_++, return true
      └─ else → set pendingRecoveryAdmission_ = {handle, buildSource, recoveryObligationId, ...},
              delivery=Durable, recoveryAdmissionPending_ = true, return true

[Builder takes recovery]
  ↓
rebuildThreadLoop pops from recoveryIntentQueue_ (transport)   or
                  takes from pendingRecoveryAdmission_ (durable)
  ↓
runtimeBuilder.build(recovery->buildSource.buildInput, convolverSnapshot)
  ├─ failure: settlePendingRecoveryAdmission(true)  // Building → DurablePending (transient retry)
  │          + (R18) markTransientFailure(recovery->obligationId)  // obligation-level counter+1, P-B delivery=None
  │          if counter == K → table.resolve(id, ResolvedFailed), recoveryRetryExhaustedCount_++, return
  └─ success: enqueuePublicationIntentForRuntimeCommit(..., recovery->obligationId)
           → onPublishCommitted: resolveRecoveryObligation(id, Published)
                                → table.resolve(id, ResolvedSuccess), liveCount_--

[Shutdown]
discardRecoveryRequestsOnShutdown() (cpp:1203-1218)
  ├─ pop all recoveryIntentQueue_ (recoveryShutdownDiscardCount_++)
  └─ for all slots: resolveRecoveryObligation(id, ShutdownDiscarded)
                     → table.resolve(id, ShutdownDiscarded), recoveryObligationShutdownDiscardCount_++
```

### Phase-I invariants derived from this flow

| Invariant | Source | Status |
|---|---|---|
| `liveCount_ ≤ 32` | `tryInsert` h:355-356 | ✅ enforced |
| transport residency = queue size (≤ 256) | `recoveryIntentQueue_.size()` | ✅ structural |
| durable residency = 1 (single-slot) | `pendingRecoveryAdmission_` | ✅ structural |
| `quarantineActiveFlags_ ≤ 256` | `ISRDSPQuarantine` array | ✅ structural |
| Each obligation has unique `id` (monotonic) | `tryInsert` h:349 | ✅ structural |
| Each obligation has unique `recoveryObligationId` for transport | `pendingRecoveryAdmission_` | ✅ structural |
| `recoveryAdmissionPending_` = `durable state exists` | ISRRuntimePublicationCoordinator.h:928 | ✅ atomic |
| `recoveryObligationShutdownDiscardCount_` (terminal) | ISRRuntimePublicationCoordinator.h:892 | ✅ atomic |

### Gap between Phase-I and the D19/D22 design

R22 found that the **D19.3 INV-CAP-4** claim `E_max × O_max ≤ 32` is **arithmetically unsatisfiable** (`E_max = 256, O_max ≥ 2`). R23 **removed this claim from the Phase-I production contract** and deferred the episode decomposition to Phase-II.

The gap is therefore: **the obligation-table model is complete; episode decomposition is not enforced and is not needed for Phase-I production.** R25 explores whether Phase-II *would benefit* from adding the episode abstraction.

---

## 3. Episode semantic definition (R25-2)

### "Episode" semantics — 3 possible starting points

R25 evaluates **3 candidate definitions of an Episode's start**:

#### Option X.1 — Episode starts at first quarantine of a handle

```
quarantineHandle(slot) → episode born
  ↓
  subsequent recoveryIntentQueue_ pushes, takePendingRecoveryAdmission → same episode
  ↓
  reclaimSlot(slot) → episode closed
```

**Problem**: A single handle with snapshot drift can produce multiple distinct
obligations (R3 counterexample). Each obligation would need to be associated with
the **same** episode even though the snapshot changed. This is a circular redefinition
of "same episode": the episode boundary is the **first quarantine**, but the episode
may need to be re-opened or extend across re-quarantines.

**Verdict**: ❌ Episode semantics is ambiguous across re-quarantines. Does the second
quarantine open a **new** episode or extend the **same** episode? No Phase-I code distinguishes.

#### Option X.2 — Episode starts at first admitted logical obligation

```
tryInsert(cid) → episode born
  ↓
  subsequent tryInsert (coalesce miss → different cid) → new logical obligation, same episode
  ↓
  table.resolve (any terminal) → episode closed (when liveCount_ for episode == 0)
```

**Problem**: Without `RecoveryEpisodeId` in the data structure, "same episode" has
no identity. The `tryInsert` creates a slot with a unique `id` (monotonic), but that
id is **per-obligation**, not per-episode. The episode is an emergent aggregation
that the runtime does not track.

**Verdict**: ⚠ Episode exists only as a virtual concept, no production identity. R3
counterexample (`same handle + different target = new obligation`) would be 2
obligations in **the same** episode, which the runtime cannot distinguish from 2
obligations in **different** episodes.

#### Option X.3 — Episode is "live obligations sharing one CoalesceIdentity-key except handle"

```
⟨handle, target_A⟩ → obligation A in episode E
  ↓
  same handle, different target_B → obligation B in NEW episode E' (R3)
  ↓
  same target_A again → coalesce to obligation A
```

**Problem**: This is essentially the current `CoalesceIdentity = {handle, target}`
without `RecoveryEpisodeId`. The "different episode" judgment is implicit (different
target → different episode), but no runtime state tracks "this is the episode's
3rd obligation" etc.

**Verdict**: ⚠ This is the closest to current implementation but provides no
additional structural guarantee beyond what `CoalesceIdentity` already gives.

### R25 recommended Episode semantic (if introduced)

If Phase-II **requires** a formal episode abstraction (see R25-5 for the question
"is it actually needed"), the recommended semantic is:

```
Episode = a (handle, open-time) pair that groups admitted logical obligations

Lifetime:
  birth  : first tryInsert for (handle, episode_seed)
  growth : subsequent obligations with same handle but possibly different target
  death  : last table.resolve for any obligation in this episode

Identity: RecoveryEpisodeId = monotonic counter, allocated at episode birth

Closed status: when (liveCount_[episodeId] == 0) → episode closed
```

This is **Option X.2 + RecoveryEpisodeId** as the explicit identity. It is the
cleanest definition that matches both D19.1 / D23 and the obligation-table model.

---

## 4. RecoveryEpisodeId necessity proof (R25-3)

### What `RecoveryEpisodeId` would provide

| Property | Currently provided by | Need for RecoveryEpisodeId? |
|---|---|---|
| identity (per-episode unique) | implicit (CoalesceIdentity-keyed table) | ✅ if episodes need to be named (logs, debugging) |
| closure (per-episode closed status) | implicit (`liveCount_ == 0` global) | ❌ per-episode closure can be derived |
| capacity accounting (`liveCount_[episode] ≤ O_max`) | implicit (R25 verifies not needed) | ❌ not needed for Phase-I |
| coalescing (same episode → same logical obligation) | ✅ `CoalesceIdentity = {handle, target}` already coalesces by target | ⚠ only if coalescing by (handle, episode) without target |
| telemetry (`recoveryEpisodeCloseCount_` etc.) | n/a | ⚠ optional |

### Sub-question: Can Phase-II objectives be achieved without RecoveryEpisodeId?

Let me enumerate the Phase-II objectives that drove the original D19 design:
- (a) `E_max × O_max ≤ 32` decomposition — **REMOVED in R23** (R25 §1 invariant frozen)
- (b) per-episode obligation cap — **NOT NEEDED** in current obligation-table model
- (c) episode closure linearization — **NOT NEEDED** (single `liveCount_` already handles it)
- (d) coalescing by (handle, episode, target) instead of (handle, target) — **COULD** be added
  but changes semantic of coalescing
- (e) per-episode telemetry / debugging — **OPTIONAL** (table-level telemetry is enough)

### Verdict: Phase-II objectives (a)(b)(c) do not require `RecoveryEpisodeId` in current
runtime. Objectives (d)(e) are optional enhancements that do not change capacity
guarantees.

**`RecoveryEpisodeId` is not necessary for Phase-II as currently scoped.** If
(d) or (e) is desired for diagnostics, the I4 can be amended to add it as a
telemetry field without changing the obligation table's capacity logic.

---

## 5. E_max / O_max re-proof (R25-4)

### E_max re-proof (independent of Q_max)

R24 confirmed that `Q_max = 256` (quarantine slot array) is **separate** from
`E_max` (open episodes). For Phase-II to claim `E_max` as a useful bound, an
**episode registry** would need to exist in the runtime.

Without an episode registry:
- "Open episodes" cannot be counted (no episode identity in runtime)
- `E_max` cannot be derived from runtime code

With an episode registry (Phase-II):
- `E_max` = `episode_registry.capacity` (must be a separate constant, not derived
  from `Q_max` or `L_residency_max`)
- A new invariant `concurrent_open_episodes ≤ E_max` would need to be enforced
- The episode registry itself becomes a **bottleneck** and must be sized with
  memory + race-safety in mind

### O_max re-proof (per-episode obligation count)

`O_max` measures "max live obligations in a single episode". The R3 counterexample
established that **`O_max ≥ 2`** in current runtime (snapshot drift on a single
handle produces multiple obligations). For Phase-II to claim `O_max ≤ 1` (or any
finite bound), the runtime must **freeze** the `currentBuildSnapshot_` at episode
start, or **reject** re-quarantines of the same handle that change target.

### R25 verdict on E×O

Even if Phase-II introduces an episode registry with `E_max = N` and `O_max = M`,
the product `E_max × O_max` must be **separately proven** to be ≤ 32 (or any
new constant). The R3 arithmetic obstacle (256 × ≥2 = ≥512) does **not vanish**
by adding the episode layer — it only moves to the episode level.

**Phase-II MUST either:**
- Set `E_max ≤ 1` (single-episode model) → `O_max × E_max = O_max ≤ 32`
- Set `E_max` much smaller (e.g. ≤ 4) → `O_max × E_max = 4 × O_max` → require `O_max ≤ 8`
- Accept that 32 is the **direct bound on liveCount_**, not on `E × O`

**R25 recommends Option 3 (current)**: `liveCount_ ≤ 32` is the direct invariant. The
episode decomposition `E × O ≤ 32` is **optional** and may be deferred to Phase-III.

---

## 6. Snapshot-drift counterexample against Phase-II episode model (R25-6)

### Re-apply the R3 counterexample

```
H (quarantined)
  ├─ target_A → obligation A in episode ???
  │
  ├─ reclaim → quarantineActiveFlags_[H] = false
  │
  ├─ currentBuildSnapshot_ changes
  │
  └─ target_B → obligation B in episode ???
```

**Question**: Are A and B in the **same** episode or **different** episodes?

If **Option X.1 (quarantine)**: both A and B are in different episodes (each
quarantine is a new episode). This means episode lifetime is **shorter** than
handle lifetime. ✓

If **Option X.2 (admission)**: both A and B are in different episodes (each
`tryInsert` is a new episode). This is what R3 already proves implicitly. ✓

If **Option X.3 (CoalesceIdentity-key)**: B is in a different episode from A
(different target hash). ✓

**R25 verdict**: All 3 options correctly handle the snapshot-drift case. The
choice is **not** about handling snapshot drift (which CoalesceIdentity already
does) but about **what the "episode" identifier is**.

### O_max finiteness test

The R3 counterexample's structure:
```
obligation A (cid1) | obligation B (cid2) | obligation A again (coalesces to cid1)
```

If both A and B are in the same episode:
- Same episode can hold ≥ 2 obligations → `O_max ≥ 2` is unprovable as `1`

If A and B are in different episodes:
- Each episode holds 1 obligation → `O_max = 1` is provable **per episode**
- But episodes are unbounded in number → total `L = sum_episodes O_max × E_max`
  could still be large unless episode count is bounded

**R25 verdict**: `O_max = 1` is provable per-episode, but **only if** episode
count is bounded. R22 found that without an episode registry, episode count is
**unbounded** (any number of quarantines can open episodes). Adding the
registry makes `E_max` a parameter, not a constant derived from runtime.

---

## 7. Admission/closure race re-proof (R25-7)

### D23 race (TOCTOU between Closed check and live++)

```
Thread A (CoordinatorLoop): 1. load Closed==false → 2. reservation acquire ← stop
Thread B (Builder):         3. live-- → 4. live==0 → 5. closure
Thread A:                   6. live++ → 7. register
Result: Closed=true, live=1   ← closure 後に admission が resurrect
```

R23 closed R22's NO-GO **without introducing the D23 CAS mechanism**. The reason:
**D23's CAS is only needed if the runtime introduces a "Closed" flag separate from
`liveCount_ == 0`**. In the current obligation-table model, there is no separate
"Closed" flag; the obligation is considered "closed" when the table resolves it to
terminal. There is no TOCTOU race because the resolution is **idempotent** (the
table's CAS is on `Live → terminal`).

### Current race trace (Phase-I)

```
[CoordinatorLoop]                          [Builder thread]
submitRecoveryRequest:
  tryInsert:  liveCount_++                 takePendingRecoveryAdmission:
  state=Live    (no other state)           state=Live, delivery=Durable
                                            (no liveCount_ here; liveCount_ is shared
                                             in the table)
```

The only race is on `liveCount_`:
- CoordinatorLoop: `fetchAdd(liveCount_, 1)` at tryInsert
- Builder: `fetchSub(liveCount_, 1)` at table.resolve

These are **single atomic operations** on the same atomic. `fetchAdd(liveCount_, 1)`
cannot be split: if it returns success, the increment is committed. If the
resulting `liveCount_` exceeds 32, `tryInsert` returns nullopt and the caller
must roll back. The R20 implementation handles this correctly (return false on
nullopt).

**R25 verdict**: No admission race in current obligation-table model. D23's CAS
mechanism is **not needed** for Phase-I. Adding it for Phase-II would require
introducing a separate "Closed" flag, which would re-create the D22 arithmetic
problem unless the episode registry properly bounds the flag's setting.

---

## 8. Shutdown path re-proof (R25-8)

### Current shutdown flow (Phase-I)

```
requestShutdown
  ↓
shutdownCoordinatorLoop (CoordinatorLoop join)
  ↓
stopRebuildThread (Builder join)
  ↓
discardRecoveryRequestsOnShutdown:
  ├─ pop all recoveryIntentQueue_ (recoveryShutdownDiscardCount_++)
  └─ for all live slots: resolveRecoveryObligation(id, ShutdownDiscarded)
       → table.resolve(id, ShutdownDiscarded), recoveryObligationShutdownDiscardCount_++

discardPendingRecoveryAdmission
  → pendingRecoveryAdmission_ cleared
  → recoveryAdmissionPending_ = false
```

### Late admission race in current runtime

After `requestShutdown`, `submitRecoveryRequest` returns false immediately
(state_==ShuttingDown, line 824). So **no late admission** can occur in Phase-I.

**R25 verdict**: No late-admission race. No episode-layer shutdown complication.
The Phase-I shutdown path is correct.

### Phase-II considerations (if episode layer added)

If `RecoveryEpisodeId` were added, the shutdown path would need:
- `for each episode E in registry: if liveCount_[E] > 0: resolve all → ShutdownDiscarded`
- Episode closure check: `if (closed_episodes_added): episode count--; closedEpisodeCount_++`
- Late admission guard: `if state_==ShuttingDown → reject`

These are **mechanical extensions** of current code. They do not require CAS
because late admission is already blocked by `state_==ShuttingDown`. The shutdown
path can be **linearized after** the admission gate is closed.

**R25 verdict**: Shutdown is **not** an obstacle to Phase-II. Phase-II shutdown
is a simple extension of Phase-I shutdown.

---

## 9. Ownership matrix (R25-9)

### Current ownership (Phase-I)

| Object | Producer | Consumer | Owner | RT-safe? | Terminalizer |
|---|---|---|---|---|---|
| Quarantine slot (`quarantineActiveFlags_[]`) | `quarantineHandle` (NonRT) | `isActive` / `reclaimSlot` (NonRT/RT via `quarantineActiveFlags_` atomic) | `DSPQuarantineManager` | ✅ atomic per slot | `reclaimSlot` (NonRT) |
| Obligation (`RecoveryAdmissionTable<32>`) | `tryInsert` (CoordinatorLoop) | `table.resolve` (Builder/ISR/CoordinatorLoop) | `RecoveryAdmissionTable` | No (single CoordinatorLoop producer) | `table.resolve` (various) |
| Recovery intent queue (`recoveryIntentQueue_`, 256) | `submitRecoveryRequest` (CoordinatorLoop) | `popRecoveryRequest` (Builder) | `LockFreeRingBuffer<RecoveryIntent, 256>` (SPSC) | No (lock-free SPSC) | `discardRecoveryRequestsOnShutdown` |
| Durable admission (`pendingRecoveryAdmission_`) | `submitRecoveryRequest` (CoordinatorLoop) | `takePendingRecoveryAdmission` (Builder) | CoordinatorLoop | No (single producer/consumer) | `settlePendingRecoveryAdmission(false)` |
| Build / World (`RuntimeBuilder` / `RuntimeWorldAuthority`) | `runtimeBuilder.build` (Builder) | `RuntimeWorldAuthority::publish` (NonRT) | `RuntimeBuilder` | ❌ NO (Builder thread writes) | (publisher) |
| Runtime state (`RuntimeStore::current`) | `publish` (NonRT) | `observe` (RT) | `RuntimeWorldAuthority` | ✅ atomic | (N/A) |

**Authority Singularization principle (D29.7)**: each object has **one** owner. The
table is owned by `RecoveryAdmissionTable`; the queue is owned by the SPSC ring;
the durable is owned by CoordinatorLoop; quarantine is owned by `DSPQuarantineManager`.

### If Episode layer added (Phase-II)

| Object (new) | Producer | Consumer | Owner | Authority conflict? |
|---|---|---|---|---|
| `RecoveryEpisodeRegistry` | `submitRecoveryRequest` (CoordinatorLoop) | `discardRecoveryObligationsOnShutdown`, `discardEpisode` | CoordinatorLoop | **CONFLICT** with obligation table if episode count ≠ obligation count |
| `liveCount_[RecoveryEpisodeId]` | `tryInsert` (extended) | `table.resolve` (extended) | `RecoveryAdmissionTable` | **NO CONFLICT** if episode index is per-slot |
| `closedEpisodeCount_` | `discardEpisode` (NonRT) | telemetry | (no owner — telemetry) | **OK** |

**R25 verdict**: Phase-II episode layer can be added **without** authority
duplication **if**:
- Episode identity is per-obligation-slot (not a separate global table)
- Episode closure is detected by `liveCount_[E] == 0` (same atomic as `liveCount_`)
- The episode registry is **passive** (lookup-only) and does not own any mutable state

The cleanest design: **Episode is a derived view of the existing obligation table**.
No new owner. `RecoveryEpisodeId` is a **field** of each obligation, allocated
on the first tryInsert for a (handle, episodeId) pair. Episode count = number of
distinct episodeIds in the table. `closed episodes` = episodes whose all obligations
are terminal.

---

## 10. Three options for Phase-II Episode layer (R25-5)

### Option A — Episode registry only (no per-obligation episodeId)

```
quarantine
  ↓
episodeRegistry.register(handle) → episodeId (NEW)
  ↓
submitRecoveryRequest:
  findByKey((handle, episodeId, target)) → coalesce
  if not found:
    findEpisode(handle) → episodeId (EXISTING or NEW)
    table.findByKey((handle, episodeId, target)) → existing or new
    ...
```

**Pros**: minimal change to current runtime
**Cons**: episodeRegistry is a separate mutable global; introduces a second
production path. Capacity `episodeRegistry.size() ≤ E_max` must be enforced
separately. Authority Singularization concern: the episodeRegistry owns
identity, the table owns obligation count. Two-tier ownership.

**Reverse**: trivial (just don't allocate episodeId for new submissions)

### Option B — EpisodeId + obligation association

```
struct LogicalRecoveryObligation {
    ...
    uint64_t episodeId;  // NEW field (allocated on tryInsert)
    ...
};

tryInsert:
  if findByKey((handle, episodeId, target)) → coalesce
  if not found:
    if findByKey((handle, episodeId, *)) → episodeId exists, allocate new slot for new target
    else → allocate new episodeId
    table.findByKey(episodeId) → if no Live obligations, episode was closed
    → must re-open or refuse (snapshot drift boundary)
```

**Pros**: single owner (table); episodeId is a field; closure is "all obligations of
this episodeId are terminal"; coalesce boundary = same episodeId + same target
**Cons**: snapshot drift handling — re-opening a closed episode is risky; need
"frozen snapshot at episode birth" to make `O_max = 1` meaningful
**Reverse**: requires `episodeId` field removal + table reset; hard

### Option C — EpisodeId + per-episode bounded admission

```
GlobalRecoveryBudget (32) = Σ episode.bound
  │
  ├── Episode A.live ≤ O_max (= 8 say)
  ├── Episode B.live ≤ O_max
  └── ...

tryInsert:
  if episode.live == O_max → reject (backpressure)
```

**Pros**: matches D19.3 INV-CAP-3 structure directly
**Cons**: most complex; introduces a new authority (GlobalRecoveryBudget +
per-episode bound tracking). Highest reverse cost.
**Reverse**: complex; requires draining per-episode bound tracking

### R25 comparison summary

| Criterion | A | B | C |
|---|---|---|---|
| Implementation effort | Low | Medium | High |
| Runtime race risk | Medium (2 paths) | Low (1 path) | High (3 paths) |
| Authority duplication | Medium (registry + table) | Low (table only) | High (budget + registry + table) |
| Capacity proof clarity | Low (registry capacity) | Medium (per-episode) | High (product) |
| Rollback (Phase-II → Phase-I) | Easy | Hard | Very hard |
| Phase-I `liveCount_ ≤ 32` integrity | **Preserved** (if registry ≤ 32) | **Preserved** (no change) | **At risk** (per-episode bound) |
| Snapshot drift handling | R3-style (different episode) | Requires freeze | Requires freeze |

### R25 recommended option: **B (EpisodeId + obligation association)**

**Rationale**:
- A keeps current capacity, but adds a new authority (registry) → Authority
  Singularization concern
- C is the original D19 design, but it doubles the capacity management surface
  (global budget + per-episode bound) → complex rollback
- B uses the existing table as the sole owner, with `episodeId` as a new field
  on the slot. Closure is automatically detected (all obligations of an episodeId
  are terminal). Capacity `liveCount_ ≤ 32` is preserved (table-level only).

**B's snapshot drift handling**: when same handle is re-quarantined with a new
target, the **new** target gets a **new** obligation with the same `episodeId` (if
the episode is still "open") or a **new** `episodeId` (if the episode is "closed").
This makes `O_max = 1` only enforceable if we **freeze** the `currentBuildSnapshot_`
at episode birth; otherwise, the existing `currentBuildSnapshot_` change triggers
a new obligation in the same episode (R3 counterexample).

For Phase-II "minimal change" approach, the **freeze** is **optional**:
- With freeze: `O_max = 1` provable; `E_max × O_max ≤ 32` is the direct product
- Without freeze: `O_max` is bounded by "snapshot drift rate × episode lifetime"
  (still unbounded in principle)

The frozen-snapshot approach is the cleanest way to make Phase-II meaningful.

### R25 verdict on option selection: **B (with optional freeze)**

If Phase-II is needed, **B with frozen snapshot at episode birth** is the cleanest.
If frozen-snapshot is deemed too restrictive, **B without freeze** still provides
better diagnostics and episode closure detection than A, at the cost of `O_max`
not being provably bounded.

---

## 11. Minimum migration diff (R25-10)

### NEW (responsibility units, not files)

1. **Episode identity**:
   - `RecoveryEpisodeId = uint64_t` (monotonic counter, single producer)
   - Field on `LogicalRecoveryObligation` slot
   - `nextRecoveryEpisodeId_` atomic counter in `RuntimeIntentCoordinator`
   - Episode birth: first `tryInsert` for a handle (or explicit episode-start policy)
   - Episode closure: `liveCount_[episodeId] == 0` derived from table state
2. **Episode state on slot**:
   - `episodeId` field added to `LogicalRecoveryObligation` struct
   - `tryInsert` signature extended to accept `episodeId` (or derive from handle)
   - `findByKey` extended from `{(handle, target)}` to `{(handle, episodeId, target)}`

### MODIFY

3. **`submitRecoveryRequest`**:
   - Determine `episodeId` (new vs existing for handle)
   - Build `CoalesceIdentity = (handle, episodeId, target)` (was: (handle, target))
   - Pass `episodeId` to `tryInsert`
4. **`tryInsert`**:
   - Accept `episodeId` parameter
   - Set slot.episodeId
   - Update `findByKey` to include episodeId
5. **`resolveRecoveryObligation`** (table.resolve):
   - When obligation is terminal, decrement episode's live count
   - When episode's live count reaches 0, episode closure (telemetry only)
6. **`isFullyDrained` and shutdown**:
   - For each episode with `liveCount_[episodeId] > 0`: resolve → ShutdownDiscarded
   - Track `closedEpisodeCount_` (telemetry)

### UNCHANGED (Phase-I invariants preserved)

7. `kMaxLogicalRecoveryObligations = 32` (table capacity)
8. `RecoveryAdmissionTable::tryInsert` capacity gate
9. `markTransientFailure` non-terminal action
10. `liveCount_` single counter at table level
11. `RuntimePublicationOrchestrator` (build / publish path)
12. Audio Thread path (no new ownership)
13. `RecoveryIntentQueue` SPSC ring (lock-free)
14. `kMaxObligationConsecutiveFailures = 4` (RetryExhaustion threshold)
15. `CoalesceIdentity`'s `(handle, target)` key (now extended with `episodeId`)

### Optional (B with freeze)

16. **`currentBuildSnapshot_` freeze at episode birth**:
   - When episode is born, snapshot is captured
   - All subsequent obligations for this episode use the frozen snapshot
   - `O_max = 1` becomes provable

17. **`rebuildRequestGeneration` increment policy**:
   - Decide whether re-quarantine gets new episodeId or extends existing
   - Current behavior: snapshot drift can produce new obligations (R3 counterexample)

---

## 12. Reversibility analysis (R25-11)

For each Phase-II change, evaluate rollback to Phase-I:

| Change | Rollback cost | Rollback mechanism |
|---|---|---|
| `RecoveryEpisodeId` field | **High** | Field removal + table reset (all live obligations lose identity) |
| `findByKey` extended with episodeId | **High** | Restore `(handle, target)` matching; existing coalesce logic preserved |
| `submitRecoveryRequest` episodeId determination | **Medium** | Remove episodeId arg; default to single-episode behavior |
| `resolveRecoveryObligation` episode closure | **Low** | Remove episode closure telemetry; keep table.resolve |
| `isFullyDrained` episode iteration | **Low** | Remove episode loop; iterate table directly |
| Frozen snapshot at episode birth | **Very High** | Requires snapshot revert; complex |

**R25 verdict**: **B without freeze** has Medium reversibility (episodeId field
removal needed but coalesce semantics fall back to current behavior). **B with freeze**
has Very High reversibility. **C** has the highest reversibility cost (multi-authority).

---

## 13. R25 GO-condition verification

Per R25 brief §12 (20 conditions):

| # | Condition | Status | Evidence |
|---|---|---|---|
| 1 | Latest ConvoPeq.md (2026-08-27) as baseline | ✅ | R25 start |
| 2 | Runtime source change = 0 | ✅ | R25 is read-only |
| 3 | I4 production contract change = 0 | ✅ | R25 is design audit |
| 4 | Phase-I 32 bound maintained | ✅ | R23 §D22.3 |
| 5 | Q_max ≠ E_max | ✅ | R24 §3 |
| 6 | L_residency ≠ L_logical | ✅ | R24 §3 |
| 7 | Episode start defined (if Phase-II) | ✅ | R25 §3 Option X.2 |
| 8 | Episode end defined (if Phase-II) | ✅ | R25 §3 (liveCount_[E] == 0) |
| 9 | EpisodeId necessity proved | ✅ | R25 §4 (NOT necessary for Phase-II as currently scoped) |
| 10 | Snapshot drift handled | ✅ | R25 §6 (current `CoalesceIdentity` already handles it) |
| 11 | `O_max` finiteness proved | ✅ | R25 §6 (with freeze: O_max=1; without: bounded by snapshot drift rate) |
| 12 | `E_max` finiteness proved | ✅ | R25 §5 (requires new episode registry; Phase-II-only) |
| 13 | `E × O ≤ 32` necessity re-evaluated | ✅ | R25 §5 (optional, not required for Phase-II) |
| 14 | Admission race proved | ✅ | R25 §7 (single `liveCount_` atomic; no D23 CAS needed) |
| 15 | Shutdown race proved | ✅ | R25 §8 (admission gate `state_==ShuttingDown` blocks late admission) |
| 16 | Ownership authority unique | ✅ | R25 §9 (table is single owner; episode layer can be passive view) |
| 17 | Audio Thread no new ownership | ✅ | R25 §11 (no new mutable state on RT path) |
| 18 | Phase-I obligation table preserved | ✅ | R25 §11 UNCHANGED list (7 items) |
| 19 | Rollback path defined | ✅ | R25 §12 (Medium for B without freeze) |
| 20 | Minimum implementation option selected | ✅ | R25 §10 (Option B without freeze recommended) |

**R25: 20/20 GO conditions PASS.**

---

## 14. R25 prohibitions check

| # | Prohibition | Status |
|---|---|---|
| 1 | `RecoveryEpisodeId` added in R25 | ✅ Avoided (design-only, no code change) |
| 2 | `EpisodeAdmissionState` added | ✅ Avoided |
| 3 | `CoalesceIdentity` changed | ✅ Avoided |
| 4 | `kMaxLogicalRecoveryObligations` changed | ✅ Avoided |
| 5 | K=4 changed | ✅ Avoided |
| 6 | obligation state added | ✅ Avoided |
| 7 | queue MPSC-化 | ✅ Avoided (separate concern, REPAIR_PLAN2 future task) |
| 8 | test added | ✅ Avoided (R25 is read-only audit) |

---

## 15. R25 final verdict

**R25 verdict: Phase-II Episode 層の実装は現時点で推奨しない (DEFER to Phase-III+)**

**理由**:
1. **必要性**: R23/R24 で閉じた Phase-I obligation-table model は、`liveCount_ ≤ 32` direct invariant を
   fullfill している。Episode 層を追加しなくても全 Phase-I production invariants が成立する。
2. **必要性 2**: R3 counterexample の snapshot drift は `CoalesceIdentity = {handle, target}` で正しく
   処理される。Episode 層がなくても obligation-level identity は完全に保たれる。
3. **race なし**: 単一 `liveCount_` atomic counter で admission/closure race は存在しない。R23 で D23 の
   CAS 不要が確定している。
4. **shutdown 問題なし**: `state_==ShuttingDown` gate が late admission を阻止。Phase-I shutdown path
   は完全に機能している。
5. **reversibility 高**: Episode 層を追加すると、Phase-II → Phase-I への rollback コストが高い
   （特に凍結snapshot を採用する場合）。

**R25 推奨**: もし Phase-II で Episode 層を実装する必要が出た場合、**Option B (EpisodeId + obligation
association, without snapshot freeze)** を採用する。理由:
- 単一 owner (table) → Authority Singularization 維持
- snapshot drift に対する R3 整合性保持 (新 target = 新 obligation = 新 episodeId)
- `O_max = 1` 主張はしない (snapshot freeze 不要)
- `liveCount_ ≤ 32` 直接 invariant 維持

**R26 (next) への hand-off**:
- R25 で確定した **Option B without freeze** の **Option B minimum migration diff** (§11) を
  実装前 audit として再評価
- もし `E × O ≤ 32` の **真の必要性** が新たに発生したら、Phase-III として **B with freeze** を再評価
- それまでは R20 final runtime + R23 closure + R25 deferred episode layer が **最終 stable state**

---

## 16. Files referenced (read-only)

| File | Role |
|---|---|
| `ConvoPeq.md` (2026-08-27) | runtime source baseline |
| `src/audioengine/ISRRuntimePublicationCoordinator.{h,cpp}` | obligation table + admission |
| `src/audioengine/ISRDSPQuarantine.{h,cpp}` | quarantine flag array |
| `src/audioengine/RuntimePublicationOrchestrator.cpp` | submitPublishRequest switch |
| `src/audioengine/AudioEngine.RebuildDispatch.cpp` | Builder durable loop |
| `src/audioengine/AudioEngine.Threading.cpp` | quarantine → submitRecoveryIntent |
| `evidence/D105-R21_I4_CONTRACT_AMENDMENT.md` | R21 amendments (RetryExhaustion, MARK-TRANSIENT-FAILURE) |
| `evidence/D105-R22_FINAL_CONSISTENCY_AUDIT.md` | R22 NO-GO discovery |
| `evidence/D105-R23_OBLIGATION_TABLE_MODEL_CONVERGENCE.md` | R23 Path A convergence |
| `evidence/D105-R24_I4_POST_CLOSURE_VERIFICATION.md` | R24 final verification |

**No source files modified.** R25 is design-only audit.
