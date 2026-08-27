# D105-R2 — Phase I Capacity Bound Re-proof

**Type**: Read-only audit / structural re-proof (0 code changes)
**Scope**: Re-prove `E_max` from actual code state transition chains. Resolve `O_max` contradiction (1 vs 16). Derive `kMaxLogicalRecoveryObligations` from proven code structure.
**Date**: 2026-08-28
**Status**: 🔄 IN PROGRESS — Re-proving from code structure (not topology assumptions)

---

## 1. User Critique Summary

The user rejected D105-R1's `E_max = 2` proof for the following reasons:

1. **Circular logic**: D105-R1 claimed `E_max = 2` via "active + fading = 2 → quarantined ≤ 2." But the PR1 reclaim loop (`AudioEngine.Commit.cpp:644-670`) iterates **ALL 256 slots** with `MAX_DSP_SLOTS = 256`, and `DSPQuarantineManager::residentCount()` is bounded by `quarantineActiveFlags_[256]` — the structural upper bound is 256, not 2.

2. **O_max contradiction**: D105-R1 simultaneously claimed:
   - `O_max = 1` (Phase I equality → same-target coalesces, so per-episode bound = 1)
   - `O_max = 16` (design parameter)

   These are contradictory. If `O_max = 1` is correct (per D18.2 equality semantics), then `E_max × O_max = E_max × 1 = E_max`, and the 32 bound requires only `E_max ≤ 32` — which may be tractable but still must be proven.

3. **32 not derived**: `E_max × O_max ≤ 32` depends on two unproven values. If `O_max = 1` (correct Phase I), then the bound is `E_max ≤ 32`, but this must be proven from actual code.

---

## 2. Methodology: Trace Complete State Transition Chain

This re-proof traces the **actual code** state transition chain:

```
Quarantine detection → submitRecoveryIntent → submitRecoveryRequest → recoveryIntentQueue_ / pendingRecoveryAdmission_
  → Builder Loop → popRecoveryRequest / takePendingRecoveryAdmission → build → publish → settle
  → PR1 reclaim loop → DSPQuarantineManager::reclaimSlot
```

**Key principle**: The bound must come from **code-level invariants** in the state transition chain, NOT from architectural assumptions about "active + fading handles."

---

## 3. Code-Traced State Transition Chain

### 3.1 Quarantine Insertion (Entry Point)

**Source**: `AudioEngine.Commit.cpp:598-605` (inside `onRuntimeRetiredNonRt`)

```cpp
quarantineSlot(pendingSlot, generation, convo::isr::QuarantineReason::RetireDeferralTimeout);
```

**What `quarantineSlot` does** (`AudioEngine.Threading.cpp:38-65`):
1. `dspQuarantineManager_.quarantineHandle(slot, generation, reason)` — sets `quarantineActiveFlags_[slot] = true` (code-verified: `ISRDSPQuarantine.cpp:27`)
2. `quarantineHandle` returns `false` if already active (guard against double-quarantine, line 22-24)
3. If applied → `submitRecoveryIntent()` is called

**Critical invariant**: A slot can only be quarantined **once** — `quarantineHandle` has a guard (`alreadyActive` check at `ISRDSPQuarantine.cpp:22`). Once quarantined, the same slot cannot be re-quarantined until `reclaimSlot` clears the flag (`ISRDSPQuarantine.cpp:31`).

### 3.2 Recovery Intent Issuance

**Source**: `QuarantineIntentHandler::handle` (`ISRRuntimePublicationCoordinator_ProcessIntent.cpp:139-140`)

```cpp
if (qResult.stateChanged && !request.handle.isNull())
{
    const auto buildSource = ctx.engine.getCurrentBuildSnapshotForRecovery();
    ctx.engine.submitRecoveryIntent(request.handle, buildSource);
}
```

**What `submitRecoveryIntent` does** (`AudioEngine.h:4415-4455`):
1. Checks `hasAuthoritativePublishedRuntime()` — if false, absorbs silently
2. Calls `runtimePublicationBridge_.submitRecoveryRequest(quarantinedHandle, buildSource, epoch)`
3. On admission success → `shutdownRuntime_.tryAdmit(1)` → `recoveryPending = true` + `rebuildCV.notify_all()`
4. On admission failure → returns false, no wake

### 3.3 Recovery Admission / Transport / Durable

**Source**: `submitRecoveryRequest` (`ISRRuntimePublicationCoordinator.cpp:836-877`)

```cpp
convo::fetchAddAtomic(pendingIntentCount_, 1, ...);  // reservation
if (recoveryIntentQueue_.push(intent)) {
    return true;   // transport path
}
// queue full → durable path (blind overwrite)
convo::fetchSubAtomic(pendingIntentCount_, 1, ...);
pendingRecoveryAdmission_.state = DurablePending;
pendingRecoveryAdmission_.handle = quarantinedHandle;
convo::publishAtomic(recoveryAdmissionPending_, true, ...);
return true;
```

**Critical structural properties**:
- `recoveryIntentQueue_`: `LockFreeRingBuffer<RecoveryIntent, 256>` — capacity 256 (SPSC)
- `pendingRecoveryAdmission_`: **single-slot struct** (no table) — blind overwrite on queue-full
- `pendingIntentCount_`: tracks transport residency (`recoveryIntentQueue_` entries) only (durable path decrements on rollback)
- **Producer is CoordinatorLoop (NonRT)** — single-threaded, sequential (`ISRCoordinatorLoop.cpp:39`)

### 3.4 Builder Consumption

**Source**: `rebuildThreadLoop` (`AudioEngine.RebuildDispatch.cpp:898-963`)

```cpp
// Transport path (sequential, 1 at a time):
while (auto recovery = runtimePublicationBridge_.popRecoveryRequest()) {
    // build → publish → settle
}

// Durable path (sequential, 1 at a time):
while (auto recovery = runtimePublicationBridge_.takePendingRecoveryAdmission()) {
    // build → publish → settle
}
```

**Critical structural properties**:
- Builder is a **single thread** processing recovery sequentially
- Transport loop drains `recoveryIntentQueue_` fully, then durable loop
- `kMaxRecoveryConsecutiveFailures = 4` (`RebuildDispatch.cpp:974`) — bounded retry

### 3.5 PR1 Reclaim Loop (Exit from Quarantine)

**Source**: `AudioEngine.Commit.cpp:644-670`

```cpp
for (uint32_t qslot = 0; qslot < MAX_DSP_SLOTS; ++qslot) {
    if (!dspQuarantineManager_.isActive(qslot)) continue;
    if (worldAuthority_.lifetime().laneOf(qslot) != RetireLane::Quarantine) continue;
    const bool graceCompleted = worldAuthority_.lifetime().isGracePeriodCompleted(
        static_cast<uint64_t>(world->generation), maxObservedGeneration, callbackActiveCount);
    if (!graceCompleted) continue;
    worldAuthority_.lifetime().reclaim(qslot);
    dspHandleRuntime_.destroyQuarantineSlot(qslot, 0);
    dspQuarantineManager_.reclaimSlot(qslot, 0);
}
```

**Critical structural properties**:
- Iterates ALL 256 slots (`MAX_DSP_SLOTS = 256`)
- Only reclaims slots where `isActive(qslot) == true` (flag set by `quarantineHandle`)
- Only reclaims slots where `graceCompleted == true` (epoch-based grace period)
- `reclaimSlot` clears `quarantineActiveFlags_[slot]` back to `false`

---

## 4. E_max Re-proof from Code Structure

### 4.1 The Structural Upper Bound: 256

From the code:
- `DSPQuarantineManager::quarantineActiveFlags_[kMaxSlots]` where `kMaxSlots = 256` (`ISRDSPQuarantine.h:34-40`)
- `quarantineHandle()` sets `quarantineActiveFlags_[slot] = true` (`ISRDSPQuarantine.cpp:27`)
- `isActive()` reads `quarantineActiveFlags_[slot]` (`ISRDSPQuarantine.cpp:44-48`)
- PR1 reclaim loop scans all `MAX_DSP_SLOTS = 256` slots (`AudioEngine.Commit.cpp:644`)
- `reclaimSlot()` clears the flag (`ISRDSPQuarantine.cpp:31`)

**Structural upper bound**: `E_max ≤ 256` (all slots could theoretically be quarantined).

### 4.2 Why E_max = 2 Was NOT Proven (D105-R1's Error)

D105-R1 claimed E_max = 2 via:
> "active DSP handles at a time: AudioEngine maintains activeHandle (single) and fadingHandle (optional, crossfade)"

**This is architecturally wrong** because:

1. **`quarantineHandle` operates on `slot` index (0–255)**, not on the activeHandle/fadingHandle topology. The `slot` is the DSPHandle's index into the 256-slot registry — it is NOT the "active/fading" handle pair.

2. **Quarantined handles are retired handles from prior generations**, not active handles. The PR1 reclaim loop processes `pendingIntents` from `dequeuePendingRetireIntents()` — these are handles that were retired (crossfade-out complete) and are pending reclamation. Each retired world generates one retire intent.

3. **No code-level invariant limits quarantined slots to 2**. The `quarantineActiveFlags_[256]` array allows up to 256 simultaneously quarantined slots. The reclaim loop only *drains* them — it doesn't *bound* them.

**User's critique is correct**: E_max = 2 is not provable from code structure. The structural bound is E_max ≤ 256.

### 4.3 The Actual Code Bound: E_max ≤ 256

The only code-proven bound on E_max is the slot registry size:

```
E_max ≤ DSPQuarantineManager::kMaxSlots = 256
    (ISRDSPQuarantine.h:40)
```

This is NOT ≤ 2. The "active + fading = 2" argument is an architectural assumption about *live* handles, not about *quarantined* handles. Quarantined handles are *retired* handles that have not yet been reclaimed — and the code allows up to 256 of them.

### 4.4 Can E_max Be Tighter Than 256?

Let's examine whether the recovery pipeline creates a tighter bound:

**The Builder Loop processes recovery sequentially** (one `popRecoveryRequest()` or `takePendingRecoveryAdmission()` at a time). But this does NOT bound E_max — it bounds the *rate* of recovery, not the *number of quarantined slots*.

The recovery pipeline:
1. Quarantine → `submitRecoveryIntent` → `submitRecoveryRequest` → `recoveryIntentQueue_` (256) or `pendingRecoveryAdmission_` (1)
2. Builder drains `recoveryIntentQueue_` then `pendingRecoveryAdmission_` sequentially

**Key observation**: The `pendingRecoveryAdmission_` is a **single-slot blind overwrite** — when the transport queue is full, new recovery requests overwrite the durable slot. This means:

- Transport: up to 256 distinct recovery intents can be queued
- Durable: 1 slot (overwrites on collision)

But the **quarantine state** (`quarantineActiveFlags_`) is independent of the recovery transport. A slot can be quarantined even if its recovery intent was overwritten in the durable slot. The PR1 reclaim loop checks `isActive(qslot)` — which is set by `quarantineHandle`, not by the recovery queue.

**Therefore**: The quarantine state and the recovery transport are decoupled. Multiple slots can be quarantined simultaneously, each generating a recovery intent. The recovery intents may coalesce/overwrite in the transport+durable pipeline, but the **quarantine flags remain set** until PR1 reclaims them.

### 4.5 E_max = 256 (Code-Proven, NOT 2)

```
E_max = 256
```

This is the structural bound from `DSPQuarantineManager::kMaxSlots = 256`. No tighter bound exists in the current code. The PR1 reclaim loop drains quarantined slots but does not prevent more from being inserted.

---

## 5. O_max Resolution

### 5.1 The Contradiction in D105-R1

D105-R1 claimed both:
- `O_max = 1` (Section E.6/E.11: "Phase I equality containment → O_max = 1")
- `O_max = 16` (Section E.13.2: "O_max = 16 — design parameter")

The user correctly identified this as an internal contradiction.

### 5.2 D18.2 Resolution: O_max = 1

Per I4 D18.2 (verbatim from `I4_DESIGN_CONTRACT.md:298`):

```
Phase I:
    semantic containment == exact SemanticRecoveryTarget 全値等価（isSemanticTargetSuperset）
    domainCoverage は supersession 判定から【外す】
    partial semantic containment（IR+EQ ⊇ IR 等）== deferred（Phase II）
```

And from `I4_DESIGN_CONTRACT.md:293-296`:

```
Phase I では containment = 全値等価のため、CoalesceIdentity が異なる（target が
異なる）ケースでは containment が成立せず、**Phase I で SUPERSEDE は構造的に不活性（常に retain both）**。
```

**Key**: In Phase I (equality containment):
- Same target → coalesce (same episode, same obligation)
- Different target → NOT supersede (Phase I supercede is inactive), BUT the code currently has **NO supersede, NO coalesce, NO identity comparison** in `submitRecoveryRequest` — it blindly overwrites the single durable slot

**However**, looking at the actual code flow:

1. Each quarantined handle maps to exactly 1 `RecoveryEpisodeId` (D19.1: episode created per handle)
2. Each quarantined handle generates exactly 1 recovery intent (1:1 mapping: `quarantineHandle` → `submitRecoveryIntent`)
3. With Phase I equality containment, obligations with the same target within the same episode coalesce
4. The `buildSource` for recovery is `getCurrentBuildSnapshotForRecovery()` — a single mutable snapshot (`RuntimeBuildSnapshot`, mutex-guarded)

**Therefore**: For a given handle (RecoveryEpisodeId), there is at most 1 distinct `SemanticRecoveryTarget` at any time (since `currentBuildSnapshot_` is a single value). All recovery intents for the same handle have the same target → they coalesce.

```
O_max = 1
```

This is correct per D18.2: Phase I equality-based containment means that within a single episode (single handle), all recovery targets are identical (derived from the single `currentBuildSnapshot_`), so they coalesce into O_max = 1.

### 5.3 Resolving the O_max = 16 Claim

D105-R1's O_max = 16 claim was based on a hypothetical: "same handle, same episode, but different SemanticRecoveryTarget." But this is impossible in the current code because:

1. `getCurrentBuildSnapshotForRecovery()` returns a single `RuntimeBuildSnapshot` (`AudioEngine.h:4843-4848`)
2. This snapshot is set by `enqueuePublicationIntentForRuntimeCommit` (single writer: CoordinatorLoop)
3. At any instant, there is exactly 1 `currentBuildSnapshot_` per handle
4. All recovery intents for a handle use the same snapshot → same target → coalesce

The "16 config variants" scenario requires **multiple distinct snapshots for the same handle**, which the current architecture does not support — `currentBuildSnapshot_` is a single mutable field, not a per-handle table.

**Conclusion**: O_max = 16 is **not supported by code structure**. O_max = 1 is the correct Phase I bound.

---

## 6. kMaxLogicalRecoveryObligations Derivation

### 6.1 The Invariant

Per I4 D19.3 (`I4_DESIGN_CONTRACT.md:440-443`):

```
INV-CAP-1: liveLogicalObligationCount ≤ kMaxLogicalRecoveryObligations
INV-CAP-2: concurrentRecoveryEpisodeIdCount ≤ E_max
INV-CAP-3: liveObligationsPerEpisode ≤ O_max
INV-CAP-4: E_max × O_max ≤ kMaxLogicalRecoveryObligations
```

### 6.2 The Code Reality

From the code structure:

- `recoveryIntentQueue_` (transport): `LockFreeRingBuffer<RecoveryIntent, 256>` — capacity 256
- `pendingRecoveryAdmission_` (durable): single-slot struct — 1 entry
- Builder: single thread, sequential consumption
- `recoveryIntentQueue_` stores `RecoveryIntent` structs, each containing a `DSPHandle` (slot 0-255)
- `pendingIntentCount_` tracks transport residency (not distinct episodes)

**Critical code observation**: The `recoveryIntentQueue_` is a **ring buffer of capacity 256**. It does NOT track distinct handles — it can contain multiple intents for the same handle. The transport queue is a *queue of intents*, not a *set of open episodes*.

### 6.3 E_max × O_max = 256 × 1 = 256

With the proven values:
- `E_max = 256` (from `DSPQuarantineManager::kMaxSlots`)
- `O_max = 1` (from D18.2 equality + single `currentBuildSnapshot_` per handle)

```
E_max × O_max = 256 × 1 = 256
kMaxLogicalRecoveryObligations ≥ 256
```

### 6.4 The 32 Bound: NOT Provable from Current Code

**The 32 bound is NOT derivable from the current code structure.** Here is why:

1. `E_max = 256` is the structural bound (all 256 slots can be quarantined)
2. `O_max = 1` is correct per D18.2
3. `E_max × O_max = 256`, not 32

The 32 bound requires `E_max ≤ 32`, which means at most 32 slots can be simultaneously quarantined. **No code-level invariant enforces this.**

The only way to get `E_max ≤ 32` is through:
- **Backpressure**: the recovery pipeline must drain quarantines faster than they accumulate, bounding simultaneous quarantines to a rate-dependent count. But this is a *rate* argument, not a *capacity* argument.
- **Admission control**: a mechanism that refuses new quarantines when the count exceeds 32. But no such mechanism exists in the current code.

**The `recoveryIntentQueue_` (capacity 256) and `pendingRecoveryAdmission_` (single slot) are the transport pipeline, not the quarantine capacity bound.** The quarantine capacity is `quarantineActiveFlags_[256]` = 256.

### 6.5 The Single-Slot Durable Problem (Why E_max Cannot Be Bounded Below 256 by Recovery Alone)

The `pendingRecoveryAdmission_` is a **single-slot blind overwrite**:

```cpp
// ISRRuntimePublicationCoordinator.cpp:865-872
pendingRecoveryAdmission_.state = PendingRecoveryAdmission::State::DurablePending;
pendingRecoveryAdmission_.handle = quarantinedHandle;  // OVERWRITES previous
pendingRecoveryAdmission_.buildSource = buildSource;   // OVERWRITES previous
```

When the transport queue (256) is full and a new recovery request arrives for a **different handle**, the old durable admission is **silently overwritten**. The old handle's recovery obligation is **lost** — but the quarantine flag for that slot remains set (`quarantineActiveFlags_[old_slot]` is still `true`).

This means:
- The Builder will only process the **latest** durable admission
- The old quarantined slot will **never be reclaimed** via the recovery path
- It can only be reclaimed via the PR1 loop (which checks `graceCompleted`)
- But the PR1 loop reclaims based on **epoch grace period**, not on recovery completion

**This is the D103 finding confirmed**: the single-slot overwrite is a blind overwrite with no coalesce identity search, no supersession check, no identity comparison.

### 6.6 Conclusion: kMaxLogicalRecoveryObligations is NOT Bounded by 32

```
E_max = 256 (proven from DSPQuarantineManager::kMaxSlots = 256)
O_max = 1 (proven from D18.2 equality + single currentBuildSnapshot_ per handle)
E_max × O_max = 256

kMaxLogicalRecoveryObligations ≥ 256 (NOT ≤ 32)
```

**The 32 bound is NOT achievable from the current code structure.** The code allows up to 256 simultaneously quarantined DSP slots, each potentially creating a recovery episode. The single-slot durable overwrite is a **correctness bug** (lost obligations), not a capacity bound.

---

## 7. What Would Make E_max ≤ 32?

To derive `E_max ≤ 32` from code structure, one of the following must be true:

### 7.1 Option A: Active Handle Topology (NOT Code-Proven)

If only 2 handles can be "live" (active + fading) at any time, then at most 2 handles can be retired/quarantined per transition. But:

- The code does NOT enforce that only 2 handles are simultaneously quarantined
- `quarantineActiveFlags_[256]` allows 256 concurrent quarantines
- Retired handles accumulate across multiple transition cycles (multiple crossfade transitions can overlap in the intent queue)
- No code invariant prevents 3+ slots from being quarantined

**This option requires new code** (an admission gate limiting concurrent quarantines to ≤ 32).

### 7.2 Option B: Transport Queue Backpressure (NOT a Capacity Bound)

The `recoveryIntentQueue_` (256) and `pendingRecoveryAdmission_` (1) form the recovery transport. If the Builder drains these faster than new quarantines are created, then at steady state, the number of outstanding episodes is bounded by the transport capacity plus what's being built.

But:
- Transport capacity = 256 + 1 = 257 (not 32)
- The transport holds *intents*, not *episodes* — multiple intents for the same handle coalesce
- Without coalesce logic in the transport (which is absent — D103 NO-GO), intents do NOT coalesce
- The Builder processes one intent at a time, but the transport can hold 257 unprocessed intents

**This option does NOT bound E_max to 32.** It bounds the *in-transport* count to 257, but the *quarantined* count (what E_max measures) is unbounded by the transport.

### 7.3 Option C: Epoch-Based Grace Period (NOT a Capacity Bound)

The PR1 reclaim loop only reclaims quarantined slots where `graceCompleted == true`. This means:

- A slot is quarantined → recovery intent issued → Builder processes → publishes new world → new epoch starts
- The old slot's epoch is now "stale" (grace period)
- After grace period completes (newer generation observed), PR1 reclaims it

This creates a **rate** bound: the PR1 loop reclaims slots at the epoch advance rate. But:
- Multiple slots can be quarantined in the same `onRuntimeRetiredNonRt` call (multiple `pendingIntents`)
- The grace period is epoch-based, and multiple epochs can pass during a single `runCoordinatorPhase` cycle
- The PR1 loop runs every `runCoordinatorPhase` (1ms fallback), but quarantine insertion happens during `onRuntimeRetiredNonRt` (commit phase)
- There is no invariant limiting the *number* of slots that can be quarantined between two PR1 reclaims

**This option does NOT bound E_max to 32.** It bounds the *rate* of reclamation, not the *peak* quarantine count.

---

## 8. D105-R2 Verdict

### E (Capacity): ❌ NO-GO

**E_max cannot be proven ≤ 32 from the current code.** The structural bound is E_max ≤ 256.

| Claim | Status | Evidence |
|---|---|---|
| E_max = 2 (D105-R1) | ❌ REJECTED | Circular: conflates live handles (active+fading) with quarantined handles. Code allows 256. |
| E_max ≤ 32 | ❌ NOT PROVEN | No code invariant limits quarantined slots to ≤ 32. `quarantineActiveFlags_[256]` allows 256. |
| E_max = 256 (structural) | ✅ PROVEN | `DSPQuarantineManager::kMaxSlots = 256` (ISRDSPQuarantine.h:40) |
| E_max × O_max ≤ 32 | ❌ NOT PROVEN | E_max × O_max = 256 × 1 = 256 ≠ 32 |

### F (Backpressure Liveness): ⚠️ CONDITIONAL PASS

The backpressure liveness (G) is structurally sound **if** `kMaxLogicalRecoveryObligations` is set to a value ≥ 256 (or the recovery pipeline is fixed to not lose obligations). But with the 32 bound, backpressure is NOT guaranteed — the transport can fill to 257, and the durable single-slot overwrite loses obligations.

### G (Backpressure Liveness): ⚠️ CONDITIONAL

Same as F — the liveness argument depends on the capacity bound being correct. With E_max = 256, the liveness proof holds but the system can accumulate up to 256 quarantined slots, which may exceed memory or latency budgets.

### H (Episode Closure): ⚠️ CONDITIONAL

Episode closure depends on:
1. The PR1 reclaim loop draining quarantined slots after grace period
2. The Builder processing all recovery intents and settling admissions
3. `isFullyDrained()` returning true when all queues are empty and `!recoveryAdmissionPending_`

With E_max = 256, closure works but may take longer (up to 256 grace periods to drain). The single-slot durable overwrite bug (D103) means some obligations may be lost, potentially preventing closure for those slots — but the PR1 loop provides an alternative reclaim path.

---

## 9. O_max Resolution (Final)

```
O_max = 1
```

**Correct per D18.2**: Phase I equality-based semantic containment. Each handle's recovery target is derived from the single `currentBuildSnapshot_` (`AudioEngine.h:4843-4848`), so there is at most 1 distinct target per handle per episode. All recovery intents for the same handle have the same target and coalesce.

**O_max = 16 is NOT supported by code structure.** It was a design hypothesis based on "16 config variants per handle per episode," but the code does not support multiple distinct `buildSource` values per handle — `currentBuildSnapshot_` is a single mutable field.

---

## 10. Remaining Ambiguity (Per I4 D19.3)

I4 D19.3 (`I4_DESIGN_CONTRACT.md:432-453`) states:

> **E_max が bound できない場合の設計**: **episode 自体を backpressure 単位**にする — episode 内の live obligation が O_max 到達 → 同一 episode への新規異種 target admission は admission authority で park/backpressure（INV-X1-9 と同一機構・lost なし）。
> **決定待ち F**: E_max / O_max の具体値（同時 quarantine 対象数・episode 内異種 target 数の設計/実測上限から導出）。**32 を invariant として証明するには INV-CAP-2/3 の導出が必須**（arbitrary constant のままにしない）。

The current code does NOT have the coalesce/supersede infrastructure to enforce E_max × O_max ≤ 32. The `submitRecoveryRequest` function has **NO coalesce identity search, NO supersession check, NO identity comparison** (D103 confirmed). This means:

1. Multiple recovery intents for the same handle can exist in `recoveryIntentQueue_` (256 capacity)
2. The single-slot `pendingRecoveryAdmission_` blindly overwrites on collision
3. No mechanism bounds the number of simultaneously quarantined handles

---

## 11. Summary Table

| Parameter | D105-R1 Claim | D105-R2 (Re-proven) | Code Evidence |
|---|---|---|---|
| E_max | 2 | **256** | `DSPQuarantineManager::kMaxSlots = 256` (ISRDSPQuarantine.h:40) |
| O_max | 1 then 16 (contradiction) | **1** | D18.2 equality + single `currentBuildSnapshot_` per handle (AudioEngine.h:4843-4848) |
| E_max × O_max | 2 × 16 = 32 | **256 × 1 = 256** | Product of proven values |
| kMaxLogicalRecoveryObligations | 32 | **≥ 256** | Must accommodate 256 quarantined slots |
| 32 bound achievable? | Claimed GO | **NO-GO** | No code invariant limits quarantine count to ≤ 32 |

## ⚠️ D105-R3 Correction: O_max = 1 Is NOT Validated

> **See `evidence/D105-R3_RECOVERY_EPISODE_TARGET_MULTIPLICITY_REAUDIT.md` for the full re-audit.**

The O_max = 1 argument in this document (Section 5.2) contains a **logically dangerous leap** identified by the user:

> "`currentBuildSnapshot_` が「その瞬間に1個」であることと、同一RecoveryEpisodeの生存期間中に異なるsnapshotから異なるtargetを順次生成できないことは別問題です."

Translation: "The fact that `currentBuildSnapshot_` is single-valued at any instant is NOT the same as proving that different snapshots cannot produce different targets sequentially within a single RecoveryEpisode's lifetime."

**The flaw**: This document assumed that because `currentBuildSnapshot_` is single-valued, all recovery intents for the same handle coalesce into the same target. However:

1. **`RecoveryEpisodeId`, `CoalesceIdentity`, and `SemanticRecoveryTarget` are ALL 0 production hits** — the coalesce infrastructure is NOT implemented.
2. `submitRecoveryRequest` performs **blind overwrite** of `pendingRecoveryAdmission_` with NO identity comparison (ISRRuntimePublicationCoordinator.cpp:865-872).
3. `currentBuildSnapshot_` is updated asynchronously by RebuildThread (`enqueuePublicationIntentForRuntimeCommit`), and can change between re-quarantines of the same slot (after PR1 reclaim clears the flag).
4. A concrete counterexample exists: quarantine slot S → capture snapshot_A → publish new world → `currentBuildSnapshot_` updated to snapshot_B → PR1 reclaims S → S re-quarantined → capture snapshot_B → two distinct targets for the same handle in flight.

**D105-R3 verdict**: O_max is **NOT 1** — it is ≥2 (unbounded in the transport path without coalesce). The table below is **corrected**:

| Parameter | D105-R1 Claim | D105-R2 (this doc) | D105-R3 (corrected) | Code Evidence |
|---|---|---|---|---|
| E_max | 2 | 256 | **256** | `DSPQuarantineManager::kMaxSlots = 256` |
| O_max | 1 then 16 | **1** | **≥2 (unbounded, NO-GO)** | 0 production hits for `RecoveryEpisodeId`, `CoalesceIdentity`, `SemanticRecoveryTarget`; blind overwrite without identity check |
| E_max × O_max | 2 × 16 = 32 | 256 × 1 = 256 | **256 × (≥2) = ≥512** | Product of corrected values |

**Final Verdict (corrected per D105-R3)**: D105's original NO-GO stands. The 32 bound is NOT provable from current code structure. E_max = 256, O_max ≥ 2 (not 1), and E_max × O_max ≥ 512 (not 32).
