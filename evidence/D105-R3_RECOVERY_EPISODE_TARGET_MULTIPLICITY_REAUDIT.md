# D105-R3 — Recovery Episode Lifetime / Target Multiplicity / Capacity Re-audit

**Type**: Read-only audit / re-validation (0 source changes)
**Purpose**: Re-validate `O_max = 1` by tracing `currentBuildSnapshot_` time semantics. Separate capacity from ownership violation. Derive `Q_max` / `E_max` / `L_max` independently. Validate 1:1 quarantine→episode against actual code.
**Date**: 2026-08-28
**Status**: ✅ Complete

---

## 0. Executive Summary

| Metric | D105-R1/R2 Claim | D105-R3 Verdict | Rationale |
|---|---|---|---|
| `E_max` | 2 (R1, rejected) → 256 (R2) | **256** (re-confirmed) | `quarantineActiveFlags_[256]` is the structural bound. No `RecoveryEpisodeId` exists in production to bound "open episodes" — the concept is design-only (I4 D13). Every distinct `DSPHandle` quarantine produces an independent recovery intent. |
| `O_max` | 1 (R1, contradicted by 16) → 1 (R2) | **≥2 (unbounded in transport path)** | `RecoveryEpisodeId`, `CoalesceIdentity`, and `SemanticRecoveryTarget` are **ALL 0 production hits**. `submitRecoveryRequest` performs blind overwrite of `pendingRecoveryAdmission_` with **no identity comparison**. Same-handle re-quarantine after reclaim (with a changed `currentBuildSnapshot_`) produces a distinct target obligation. |
| `Q_max` | ≤ 256 (implied by E_max) | **256** (separated from E_max) | `quarantineActiveFlags_[256]` bounds simultaneously-active quarantined slots. This is a **capacity** bound, independent of episode tracking (which is absent). |
| `L_max` | derived from E_max × O_max | **257** (transport+ durable) | `recoveryIntentQueue_` (256) + `pendingRecoveryAdmission_` (1) = 257 transport-durable residency. Separate from `Q_max`. |
| `Q_max × O_max ≤ 32` | E_max × O_max = 256 × 1 = 256 | **NO-GO** (256 ≠ 32) | Same as R2. The 32 bound is not supported by current code structure. |
| Durable blind overwrite | ownership violation (unaddressed in R2) | **INV-X1-7 violation confirmed** | `submitRecoveryRequest` overwrites `pendingRecoveryAdmission_` without any `LogicalRecoveryIdentity` / `CoalesceIdentity` check. This is an **ownership conservation violation**, NOT a capacity bound. |

**Bottom line**: `O_max = 1` is **NOT validated**. The R2 proof's leap — "`currentBuildSnapshot_` is single-valued → same-target → coalesce → O_max = 1" — confuses **architectural possibility** with **implementation reality**. Coalesce infrastructure is **not implemented**. The blind overwrite in `submitRecoveryRequest` does NOT coalesce; it **discards** the old obligation's identity.

---

## 1. User Critique Context

D105-R2 claimed `O_max = 1` based on:

> "For a given handle (RecoveryEpisodeId), there is at most 1 distinct `SemanticRecoveryTarget` at any time (since `currentBuildSnapshot_` is a single value). All recovery intents for the same handle have the same target → they coalesce."

The user's critique (verbatim):

> "`currentBuildSnapshot_` が「その瞬間に1個」であることと、同一RecoveryEpisodeの生存期間中に異なるsnapshotから異なるtargetを順次生成できないことは別問題です."

**Translation**: "The fact that `currentBuildSnapshot_` is single-valued at any instant is NOT the same as proving that different snapshots cannot produce different targets sequentially within a single RecoveryEpisode's lifetime."

The R2 proof's logical leap was:
1. `currentBuildSnapshot_` is single-valued → ✅ TRUE (1 writer under mutex)
2. → All recovery intents for the same handle share the same target → ❌ FALSE (snapshot can change between quarantines)
3. → They coalesce → ❌ FALSE (CoalesceIdentity is NOT implemented)
4. → O_max = 1 → ❌ INVALID (relies on steps 2-3)

This proof is **circular**: it assumes the D18 coalesce design is implemented, which is what it's trying to prove.

---

## 2. Step 1 — O_max = 1 Re-validation: Audit `currentBuildSnapshot_` Writers

### 2.1 Writer Audit: `currentBuildSnapshot_`

**Single writer**: `AudioEngine::enqueuePublicationIntentForRuntimeCommit` (Commit.cpp:799).

```cpp
// AudioEngine.Commit.cpp:782-805
void AudioEngine::enqueuePublicationIntentForRuntimeCommit(
    DSPCore* newDSP, int generation,
    const convo::RuntimeBuildSnapshot& sealedSnapshot, ...) {
    {
        std::lock_guard<std::mutex> lock(currentBuildSnapshotMutex_);
        currentBuildSnapshot_ = sealedSnapshot;  // ← ONLY writer
    }
    ...
}
```

**Readers**:
- `getCurrentBuildSnapshotForRecovery()` (AudioEngine.h:4409) — returns **value copy**: `return currentBuildSnapshot_;`
- Internal comment references at AudioEngine.h:4844, ISRRuntimePublicationCoordinator_ProcessIntent.cpp:129

**Conclusion**: `currentBuildSnapshot_` has exactly 1 writer, guarded by `currentBuildSnapshotMutex_`. It is a `RuntimeBuildSnapshot` value type (trivially copyable).

### 2.2 Reader Audit: `getCurrentBuildSnapshotForRecovery()`

```cpp
// AudioEngine.h:4409 (approx)
convo::RuntimeBuildSnapshot AudioEngine::getCurrentBuildSnapshotForRecovery() const noexcept {
    std::lock_guard<std::mutex> lock(currentBuildSnapshotMutex_);
    return currentBuildSnapshot_;  // ← value copy, NOT reference
}
```

**Confirmed**: Returns a **value copy**. No dangling reference risk.

### 2.3 Writer Call Sites: `enqueuePublicationIntentForRuntimeCommit`

Procedurally, the writer is invoked from 3 locations in `RebuildDispatch.cpp` (all on RebuildThread, inside `rebuildThreadLoop`):

| Call Site | Purpose | Source |
|---|---|---|
| RebuildDispatch.cpp:999 | Normal rebuild publish | `task.runtimeBuildSnapshot` (user config change) |
| RebuildDispatch.cpp:1072 | Recovery publish | `recoverySnapshot` (rebuilt after quarantine) |
| RebuildDispatch.cpp:1230 | Deferred re-enqueue | `task.runtimeBuildSnapshot` |

**Critical**: The writer runs on **RebuildThread**, which is a **separate thread** from CoordinatorLoop. `submitRecoveryIntent` (called from `QuarantineIntentHandler::handle` on CoordinatorLoop) captures a snapshot value-copy at the moment of quarantine detection. Between two quarantines of the same slot S, `enqueuePublicationIntentForRuntimeCommit` (RebuildThread) can update `currentBuildSnapshot_` to a new value.

### 2.4 Temporal Gap Analysis

**Scenario**: Same slot S quarantined twice with different `buildSource` values.

| Step | Thread | Action | `currentBuildSnapshot_` | `buildSource` captured for S |
|---|---|---|---|---|
| T1 | CoordinatorLoop | Slot S quarantined (1st time) | snapshot_A | snapshot_A → `recoveryIntentQueue_` push |
| T2 | RebuildThread | `enqueuePublicationIntentForRuntimeCommit` with snapshot_B | **snapshot_B** | — |
| T3 | CoordinatorLoop | PR1 reclaim: S's grace completed (new world published) | snapshot_B | — |
| T4 | CoordinatorLoop | Slot S quarantined (2nd time, flag was cleared at T3) | snapshot_B | snapshot_B → `recoveryIntentQueue_` push |

**Result**: `recoveryIntentQueue_` now contains **two intents for the same slot S with different `buildSource` values** (snapshot_A and snapshot_B). If the Builder hasn't consumed intent_A yet (queue not full but Builder slow), both coexist.

**Even if intent_A is consumed**: The `pendingRecoveryAdmission_` blind overwrite means that if the queue is full when intent_B is pushed, intent_A's durable state is silently overwritten. The old obligation's target is lost — no `CoalesceIdentity` check prevents this.

### 2.5 O_max = 1 Verdict: **FAIL (not validated)**

The R2 proof's central claim — that `currentBuildSnapshot_` being single-valued implies O_max = 1 — is **invalid** because:

1. **No coalesce implementation**: `RecoveryEpisodeId`, `CoalesceIdentity`, `SemanticRecoveryTarget` — **0 production hits**. The `submitRecoveryRequest` function performs **blind overwrite** of `pendingRecoveryAdmission_` (ISRRuntimePublicationCoordinator.cpp:865-872) with no identity comparison. This is explicitly documented in `evidence/D103-SemanticSupersession-Implementation-Readiness-Audit.md:65`:
   > "ON PUSH FAILURE: rollback + `pendingRecoveryAdmission_` overwrite (single-slot, **no coalesce check, no supersession check, no identity comparison**)"

2. **Snapshot can change between quarantines**: `currentBuildSnapshot_` is updated asynchronously by RebuildThread. Between reclaim (PR1 clears the flag) and re-quarantine, the snapshot can change.

3. **No temporal stability proof**: The R2 proof asserted "at most 1 active SemanticRecoveryTarget per handle at any instant" — but this ignores that a handle can be quarantined → reclaimed → re-quarantined with a DIFFERENT snapshot. The "instant" uniqueness does not prevent sequential different targets.

**O_max is ≥ 2 (and unbounded in the transport path without coalesce).**

---

## 3. Step 2 — 1:1 Quarantine → Episode Enforcement

### 3.1 `RecoveryEpisodeId`: 0 Production Hits

Direct code search confirms:

```
grep_search "RecoveryEpisodeId|CoalesceIdentity|SemanticRecoveryTarget" in src/**/*.h → 0 matches
grep_search "RecoveryEpisodeId|CoalesceIdentity|SemanticRecoveryTarget" in src/**/*.cpp → 0 matches
```

**D103 audit confirms** (`evidence/D103-SemanticSupersession-Implementation-Readiness-Audit.md:8-12`):
- `RecoveryEpisodeId`: **0 hits, MISSING** (design-only: I4 D13.1)
- `CoalesceIdentity`: **0 hits, MISSING** (design-only: I4 D18.1)
- `SemanticRecoveryTarget`: **0 hits, MISSING** (design-only: I4 D12.2)

### 3.2 What the Code Actually Does

The `RecoveryIntent` struct (ISRRuntimePublicationCoordinator.h:215-228):

```cpp
struct RecoveryIntent {
    DSPHandle handle;                // quarantined handle
    PublicationEpoch epoch;          // emit-time epoch
    uint64_t intentId;               // diagnostic sequence number
    convo::RuntimeBuildSnapshot buildSource;  // value-copied snapshot
};
```

There is **no episode field**. The `handle` is a `DSPHandle {slot, generation}`. The `intentId` is a monotonic counter — not an episode ID.

The `PendingRecoveryAdmission` struct (ISRRuntimePublicationCoordinator.h:667-682):

```cpp
struct PendingRecoveryAdmission {
    enum class State : uint8_t { NoAdmission, DurablePending, Building };
    State state = State::NoAdmission;
    bool pending = false;
    uint64_t recoveryGeneration = 0;
    convo::RuntimeBuildSnapshot buildSource{};
    bool reservationOwned = false;
    DSPHandle handle{};
    PublicationEpoch epoch{0};
    uint64_t intentId{0};
};
```

**No episode field. No CoalesceIdentity. No semantic target.** The struct is a plain SPSC single-slot — no identity tracking.

### 3.3 What `submitRecoveryRequest` Actually Does

ISRRuntimePublicationCoordinator.cpp:840-877 (the `submitRecoveryRequest` function):

```cpp
// Simplified — the core logic:
RecoveryIntent intent{
    quarantinedHandle,
    epoch,
    nextRecoveryIntentId_.fetch_add(1, ...),
    buildSource  // ← value copy captured at this moment
};

// Reserve before push (INV-X1-5: 1 logical admission = 1 reservation)
convo::fetchAddAtomic(pendingIntentCount_, 1, ...);
if (recoveryIntentQueue_.push(intent)) {
    return true;   // transport recovery exists
}

// Queue full: blind overwrite — NO identity check
convo::fetchSubAtomic(pendingIntentCount_, 1, ...);
convo::fetchAddAtomic(recoveryIntentDropCount_, 1, ...);

pendingRecoveryAdmission_.state = PendingRecoveryAdmission::State::DurablePending;
pendingRecoveryAdmission_.pending = true;
pendingRecoveryAdmission_.recoveryGeneration = intent.intentId;
pendingRecoveryAdmission_.buildSource = buildSource;  // ← OVERWRITTEN unconditionally
pendingRecoveryAdmission_.handle = quarantinedHandle;  // ← OVERWRITTEN unconditionally
...
```

**Critical**: When the queue is full, the durable slot is **overwritten unconditionally**. There is no check: "is this the same handle/episode as the existing durable admission?" If the new intent has the same handle but a different target (snapshot_B vs snapshot_A), the old target is simply **lost**.

### 3.4 1:1 Quarantine → Episode Verdict: **NOT ENFORCED**

The design contract D18.1 Step 0 requires a `CoalesceIdentity = {handle, RecoveryEpisodeId, SemanticRecoveryTarget}` lookup before issuing a recovery intent. If an existing open episode matches, the obligation is coalesced — no new episode is created.

**In production code, this lookup does not exist.** Every `QuarantineIntentHandler::handle` that detects `stateChanged` calls `submitRecoveryIntent` → `submitRecoveryRequest` → `recoveryIntentQueue_.push()` **unconditionally**. There is no dedup, no coalesce, no identity matching.

**D18.1 Step 0 is NOT implemented** → the 1:1 quarantine→episode mapping is NOT enforced.

---

## 4. Step 3 — O_max Counterexample

### 4.1 Counterexample: Same Handle, Two Distinct Targets

**Prerequisites**:
- Slot S is quarantined, recovery intent_A is pushed to `recoveryIntentQueue_`
- The Builder is **slow** or the queue is near-full, so intent_A is NOT yet consumed
- A new world is published → `currentBuildSnapshot_` changes from snapshot_A to snapshot_B
- PR1 reclaim: `isGracePeriodCompleted` returns true for S (new world published → `maxObservedGeneration > worldGeneration`)
- `reclaimSlot(S, 0)` clears `quarantineActiveFlags_[S]`
- Slot S is re-quarantined (a new DSP at same slot, or same DSP re-failing)
- `QuarantineIntentHandler` captures `getCurrentBuildSnapshotForRecovery()` → returns snapshot_B
- `submitRecoveryIntent(S, snapshot_B)` → `submitRecoveryRequest` → `recoveryIntentQueue_.push(intent_B)`

**State at this point**: `recoveryIntentQueue_` contains both `intent_A` (handle=S, buildSource=snapshot_A) and `intent_B` (handle=S, buildSource=snapshot_B).

**Two obligations for the same handle with different targets exist simultaneously.**

### 4.2 Why D105-R2's O_max=1 Is Circular

The R2 proof argued:

> "Since `currentBuildSnapshot_` is a single mutable `RuntimeBuildSnapshot`... there is at most 1 active SemanticRecoveryTarget per handle at any instant."

This is true **at a single instant**. But the flaw is:

1. `currentBuildSnapshot_` is single-valued at **instant T1** → intent_A captures snapshot_A
2. `currentBuildSnapshot_` changes to snapshot_B at **instant T2** (RebuildThread publishes)
3. Slot S is reclaimed at T3, re-quarantined at T4 → intent_B captures snapshot_B
4. Both intents coexist in the queue

**"At any instant, there is 1 currentBuildSnapshot_"** ≠ **"All recovery intents for the same handle have the same target"**. The snapshot can change during an episode's lifetime. The R2 proof conflated instantaneous uniqueness with temporal stability — the user's exact critique.

Furthermore, even if both intents had the **same** target, there is **no coalesce** to merge them. The queue would contain two duplicate intents. The O_max=1 claim relied on an **unimplemented design feature** (coalesce by `CoalesceIdentity`).

### 4.3 Counterexample Verdict: **O_max ≥ 2 (concrete)**

The counterexample is constructible from current code. `O_max` is not bounded by 1 in production.

---

## 5. Step 4 — Q_max vs E_max Separation

### 5.1 Q_max: Maximum Concurrent Quarantined Slots

**Definition (D13/D19.1)**: Q = simultaneously-quarantined DSP slots (capacity).

**Code bound**: `DSPQuarantineManager::quarantineActiveFlags_[256]` (`ISRDSPQuarantine.h:44`). Each slot is an `std::atomic<bool>`. The `quarantineHandle` method has an `alreadyActive` guard — a slot cannot be re-quarantined while active.

```cpp
// ISRDSPQuarantine.cpp:23-28
bool alreadyActive = convo::consumeAtomic(
    quarantineActiveFlags_[slot], std::memory_order_acquire);
if (alreadyActive)
    return false;  // already quarantined — cannot re-quarantine
```

**Q_max = 256** (all slots can be quarantined simultaneously).

This is a **pure capacity bound**. It counts quarantined slots, not episodes. In the current code, since `RecoveryEpisodeId` does not exist, every quarantined slot IS an independent recovery trigger. So Q_max = E_max in the current (unimplemented-coalesce) reality.

### 5.2 E_max: Maximum Concurrent Open Recovery Episodes

**Definition (I4 D19.3)**: E = number of simultaneously open `RecoveryEpisodeId` instances.

**Code reality**: `RecoveryEpisodeId` does not exist in production. The `quarantineActiveFlags_[256]` array is the only structural bound on simultaneously-active quarantines. There is no episode counter, no episode allocator, no per-episode state tracking.

**E_max is undefined in production code**. Per D19.1, `RecoveryEpisodeId` should be allocated by a dedicated monotonic counter at episode creation. Since this is absent, the "E_max" concept does not apply to current code. The closest structural bound is Q_max = 256.

### 5.3 Separation Confirmed

| Concept | D18 Design | Production Code | Separation |
|---|---|---|---|
| Q_max (quarantined slots) | ≤ 256 (`quarantineActiveFlags_[256]`) | ≤ 256 (`quarantineActiveFlags_[256]`) | Capacity — implemented |
| E_max (open episodes) | Bounded by `RecoveryEpisodeId` counter | **Undefined — `RecoveryEpisodeId` absent** | Not separated; Q is the only bound |
| O_max (distinct targets per episode) | 1 (D18.2 equality + coalesce) | **Unbounded** (no coalesce, snapshot drift) | Not enforced |

**The D105-R2 attempt to conflate E_max with Q_max is invalid.** D18 design separates them, but only Q is implemented. E is undefined, and O is unbounded.

---

## 6. Step 5 — Independent L_max Derivation

### 6.1 L_max: Maximum Recovery Obligations in Transport + Durable

**Definition (D13)**: L = logical recovery obligations awaiting Builder consumption.

**Transport path**: `recoveryIntentQueue_` — `LockFreeRingBuffer<RecoveryIntent, 256>` (SPSC, `kRecoveryIntentQueueCapacity = 256`).

**Durable path**: `pendingRecoveryAdmission_` — single-slot struct (`PendingRecoveryAdmission`, `ISRRuntimePublicationCoordinator.h:667`).

**Flow**: `submitRecoveryRequest` → try `recoveryIntentQueue_.push(intent)` → if fails (full) → overwrite `pendingRecoveryAdmission_`.

**Bound**: At most 256 in transport + 1 in durable = **257**.

```
L_max = kRecoveryIntentQueueCapacity + 1 = 256 + 1 = 257
```

**But this is transport+durable residency, NOT logical obligation count.** Without coalesce, the 256 transport entries + 1 durable entry can all hold **distinct targets for the same handle** (as shown in the counterexample). The L_max bound does not constrain O_max because there is no episode grouping.

### 6.2 L_max vs Q_max Independence

- Q_max = 256: quarantined slots (each can have at most 1 recovery intent in flight due to `alreadyActive` guard)
- L_max = 257: transport + durable recovery intents (can coexist for different handles, and potentially the same handle across cycles)
- These are independent because the `alreadyActive` guard prevents re-quarantine, but the transport/durable buffers can hold intents for different handles simultaneously.

### 6.3 L_max Verdict: **257 (transport+durable, not logical obligations)**

No 32 bound exists. No code invariant limits L to ≤ 32.

---

## 7. Step 6 — Durable Overwrite as Ownership Violation (INV-X1-7)

### 7.1 The Durable Overwrite

ISRRuntimePublicationCoordinator.cpp:865-873:

```cpp
// Queue full → blind overwrite of pendingRecoveryAdmission_
pendingRecoveryAdmission_.state = PendingRecoveryAdmission::State::DurablePending;
pendingRecoveryAdmission_.pending = true;
pendingRecoveryAdmission_.recoveryGeneration = intent.intentId;
pendingRecoveryAdmission_.buildSource = buildSource;
pendingRecoveryAdmission_.reservationOwned = true;
pendingRecoveryAdmission_.handle = quarantinedHandle;
pendingRecoveryAdmission_.epoch = epoch;
pendingRecoveryAdmission_.intentId = intent.intentId;
convo::publishAtomic(recoveryAdmissionPending_, true, std::memory_order_release);
```

### 7.2 This Is NOT a Capacity Bound

The D105-R2 document treated the single-slot overwrite as contributing to the O_max=1 bound ("single-slot struct, blind overwrite on queue-full"). This is incorrect.

The blind overwrite is an **ownership conservation violation** (INV-X1-7): the old obligation's identity is silently lost without any `LogicalRecoveryIdentity` or `CoalesceIdentity` check. The I4 D18 contract (Section 12.7) requires that when a new admission arrives for the same `LogicalRecoveryIdentity`, it must either:
- **Coalesce** (merge into existing obligation — same target), OR
- **Supersede** (replace only if `canSupersede` — requires `SemanticRecoveryTarget` superset check)

The current code does **neither**. It unconditionally overwrites. This means:

1. If the old obligation had target A and new has target B (different snapshots): target A's obligation is **lost**.
2. If the old obligation had target A and new has target A (same snapshot): the overwrite is semantically safe but still **unverified** (no identity check).

### 7.3 This Is Not O_max

O_max measures "distinct objectives per episode." The blind overwrite affects at most ONE obligation (the single durable slot). It does not bound O_max because:

1. The transport queue (`recoveryIntentQueue_`) has 256 independent slots — no overwrite, no dedup
2. The `alreadyActive` guard prevents re-quarantine of the same slot — but **different slots** can be quarantined independently
3. The same slot re-quarantined after reclaim can produce a **different target** (snapshot drift), and this new intent goes into the transport queue, not necessarily overwriting the durable slot

**The overwrite is an INV-X1-7 violation** (ownership conservation failure), which is a **correctness** and **liveness** concern — NOT a capacity bound for D105.

---

## 8. Final Verdict Table

| Metric | Value | Justification |
|---|---|---|
| **Q_max** | 256 | `quarantineActiveFlags_[256]` — all slots can be quarantined simultaneously |
| **E_max** | Undefined (Q_max = 256 in practice) | `RecoveryEpisodeId` is design-only (0 production hits). No episode counter exists. The structural bound on quarantined slots is Q_max = 256. |
| **O_max** | ≥2 (unbounded in transport) | No `CoalesceIdentity` (0 hits). `submitRecoveryRequest` performs blind overwrite with no identity comparison. `currentBuildSnapshot_` can change between re-quarantines of the same handle (RebuildThread updates asynchronously). Counterexample constructed. |
| **L_max** | 257 | `recoveryIntentQueue_` (256) + `pendingRecoveryAdmission_` (1) |
| **E_max × O_max ≤ 32** | **NO-GO** | 256 × (≥2) = ≥512 ≠ 32. No code invariant bounds quarantine count to ≤32. |
| **32 bound** | **NO-GO** | No structural reason for 32. The bound does not exist in current code. |
| **INV-X1-7** | Violated | Blind overwrite of `pendingRecoveryAdmission_` without `CoalesceIdentity`/`LogicalRecoveryIdentity` check — ownership conservation failure, not a capacity bound. |
| **1:1 quarantine→episode** | Not enforced | `RecoveryEpisodeId` absent from production source. No episode tracking exists. |

---

## 9. Recommendations

### 9.1 Short-Term (Audit Only — D105-R3 is read-only)

1. **D105-R2 must be revised**: The O_max=1 claim is invalid without coalesce infrastructure. D105-R2's verdict table must change `O_max` from **1** to **≥2 (unbounded)**.
2. **E_max derivation**: Must explicitly state that `RecoveryEpisodeId` is absent from production code (0 hits confirmed). E_max cannot be bounded by episode infrastructure that does not exist.
3. **32 bound**: Remains NO-GO. The `E_max × O_max = 32` target cannot be met without implementing `RecoveryEpisodeId` + `CoalesceIdentity` + `SemanticRecoveryTarget`.

### 9.2 Implementation Prerequisites (for future work)

The D105 Phase I capacity bound of 32 requires ALL of the following to be implemented (per I4 D18.1, D13):

| Component | Status | Purpose |
|---|---|---|
| `RecoveryEpisodeId` allocator | ❌ Not implemented (0 hits) | Bound E_max = number of distinct episodes |
| `CoalesceIdentity = {handle, RecoveryEpisodeId, SemanticRecoveryTarget}` | ❌ Not implemented (0 hits) | Merge same-identity obligations → enforce O_max |
| `SemanticRecoveryTarget` (5-field equality) | ❌ Not implemented (0 hits) | Define target identity for coalesce |
| `canSupersede()` | ❌ Not implemented (0 hits, Phase II) | Handle target changes within same episode (Phase II) |
| Bounded durable table (N slots) | ❌ Single-slot struct | Current `pendingRecoveryAdmission_` is 1-slot blind overwrite |

**Without at least `RecoveryEpisodeId` + `CoalesceIdentity`, the 32 bound is structurally impossible in current code.**

---

## 10. Traceability

| Artifact | Link |
|---|---|
| D105-R1 evidence | `evidence/D105-R1_PHASE1_CAPACITY_LIVENESS_CLOSURE_RESOLUTION_PROOF.md` |
| D105-R2 evidence | `evidence/D105-R2_PHASE1_CAPACITY_BOUND_REPROOF.md` |
| D103 implementation audit | `evidence/D103-SemanticSupersession-Implementation-Readiness-Audit.md` |
| D104 alignment audit | `evidence/D104_D18_Final_Contract_Alignment_Audit.md` |
| I4 Design Contract | `doc/work88/I4_DESIGN_CONTRACT.md` |
| REPAIR_PLAN2 (dash) | `doc/work88/REPAIR_PLAN2-dash.md` |
| PendingRecoveryAdmission struct | `src/audioengine/ISRRuntimePublicationCoordinator.h:667-682` |
| submitRecoveryRequest impl | `src/audioengine/ISRRuntimePublicationCoordinator.cpp:840-877` |
| currentBuildSnapshot_ writer | `src/audioengine/AudioEngine.Commit.cpp:782-805` |
| getCurrentBuildSnapshotForRecovery | `src/audioengine/AudioEngine.h:4409` |
| quarantineHandle guard | `src/audioengine/ISRDSPQuarantine.cpp:20-28` |
| reclaimSlot | `src/audioengine/ISRDSPQuarantine.cpp:49-80` |
| PR1 reclaim loop | `src/audioengine/AudioEngine.Commit.cpp:633-670` |
| QuarantineIntentHandler | `src/audioengine/ISRRuntimePublicationCoordinator_ProcessIntent.cpp:108-140` |
| RecoveryIntent struct | `src/audioengine/ISRRuntimePublicationCoordinator.h:215-228` |
| kRecoveryIntentQueueCapacity | `src/audioengine/ISRRuntimePublicationCoordinator.h:642` |
