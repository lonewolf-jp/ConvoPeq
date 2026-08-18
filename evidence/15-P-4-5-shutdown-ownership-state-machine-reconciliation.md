# 15-P-4-5: Shutdown Ownership State Machine Reconciliation

## Executive Summary

**GAP-CROSS-3 remains OPEN.** The abnormal destructor path (`~AudioEngine()`) does NOT drain
`RetireOverflowRing` → `LifetimeState MPSC queue` → `EpochControl` (slot state machine).

**However, the ownership model has been reclassified**: `RetireIntent` tracks DSPHandleRuntime
**slot state**, NOT actual `DSPCore*` pointer ownership. Actual pointer ownership flows through
`DSPLifetimeManager::retire()` → `router_->enqueueWithRetry(ptr, deleter, epoch)` →
`DeferredDeletionQueue`, which IS drained by `drainDeferredRetireQueues(true)`.

The risk of NOT draining the MPSC queue:
1. `pendingIntentCount() > 0` → `isFullyDrained()` returns false → `waitForDrain()` times out
2. Timed-out path skips some drain operations (but `drainAllUnsafe` is still called)
3. Slot state not transitioned to `Reclaimed` (ephemeral — slots freed at AudioEngine destruction)

## A. Abnormal dtor invokes overflow drain — **FAIL**

`~AudioEngine()` does NOT call `drainOverflowRing()`. The drain section only calls:
```cpp
m_retireRouter->drainAll();   // D+Q+E+Terminal only
```
OverflowRing is not targeted.

**Evidence**: `AudioEngine.CtorDtor.cpp` lines 236-246.

## B. Overflow entries reach terminal disposition — **FAIL**

`drainOverflowRing()` → `emitRetireIntent()` pushes to `LifetimeState` MPSC queue (`slots_`).
`dequeuePendingRetireIntents()` / `dequeueOne()` is ONLY called in `AudioEngine.Commit.cpp:492`
(RT commit path). After `stopTimer()`, the commit path is stopped, so MPSC queue is never drained.

**However**: `RetireIntent` tracks slot state, not pointer ownership. The actual `DSPCore*`
pointer disposal is via `DeferredDeletionQueue` (drained by `drainAllUnsafe`).

**Evidence**: `ISRRetire.cpp:144`, `AudioEngine.Commit.cpp:492`.

## C. Destruction-order safety — **PASS**

All members alive during dtor body. Member destruction order (reverse declaration):
1. `dspQuarantineManager_` (line 4780) — first destroyed
2. `ShutdownRuntime` (line 4787)
3. `runtimePublicationBridge_` (line 4748)
4. `RuntimeWorldAuthority` (line 4752) — contains LifetimeState, EpochControl
5. `m_epochDomain` (line 4676) — contains DeferredDeletionQueue
6. `m_retireRouter` (line 4681)

A proposed fix adding `drainOverflowRing()` + `dequeueOne()` + `reclaim()` in the dtor body
(all non-allocating, noexcept) would be safe.

## D. Normal→dtor idempotence — **PASS** (with proper fix)

- `drainOverflowRing()` is idempotent (ring empty → no-op)
- `dequeueOne()` / `dequeueFallback()` are idempotent (empty queue → return false)
- `reclaim(slot)` on already-Reclaimed slot is a no-op (slot state machine handles idempotently)

Normal path (`releaseResources`) also needs the fix — currently does NOT drain MPSC queue.
If both paths call the same drain, double-call on same slot is safe (state-machine idempotent).

## E. No silent RetireIntent loss — **FAIL** (current state)

Without the fix:
- `emitRetireIntent()` → MPSC queue
- MPSC queue never drained in shutdown
- `LifetimeState` (trivially destructible) silently discards queued intents
- `pendingIntentCount()` stays > 0 → `waitForDrain()` times out

With the proposed fix:
- `dequeueOne()` + `reclaim()` drains all intents
- `pendingIntentCount()` reaches 0

## F. Allocation/exception safety — **PASS** (with proper fix)

The proposed fix uses only:
- `drainOverflowRing()` — `noexcept`, non-allocating
- `dequeueOne()` — `noexcept`, non-allocating (direct slot access)
- `dequeueFallback()` — `noexcept`, uses existing mutex, non-allocating
- `emitIntent()` + `enqueueRetire()` + `settleEpoch()` + `reclaim()` — `noexcept`

No `std::vector` allocation, no `noexcept(false)` functions. No `dequeuePendingRetireIntents()`
(which allocates via `std::vector`).

## G. Normal shutdown path analysis — **FAIL** (existing latent bug)

`releaseResources()`:
- Calls `emitRetireIntent()` (OverflowRing drain → MPSC queue)
- Calls `m_retireRouter->tryReclaim()` (drains D only)
- Does NOT call `dequeueOne()` / `dequeuePendingRetireIntents()`
- `isFullyDrained()` checks `pendingIntentCount() == 0` — can hang
- `waitForDrain(2000, 2)` times out → enters `else` branch (drains D only)

**Both normal AND abnormal shutdown paths fail to drain the MPSC queue.**

## H. Epoch safety — **PASS** (with proper fix)

In `~AudioEngine()`:
- All threads joined (CoordinatorLoop, WorkerThread, rebuildThread)
- `closeReaderRegistration()` called → no new readers
- `activeReaderCount() == 0` confirmed before `drainAll()`
- `reclaim(slot)` is slot-state-only (NO pointer deletion, NO DeferredDeletionQueue enqueue)
- Epoch safety guaranteed: no active readers → no grace period needed

## I. Ownership model reclassification — **IMPORTANT**

Two parallel ownership tracking systems:

### System 1: RetireIntent (slot state)
```
RetireOverflowRing (entry = RetireIntent{dspSlot, generation})
    → drainOverflowRing() / emitRetireIntent()
    → LifetimeState MPSC queue (slots_[256])
    → dequeuePendingRetireIntents() / dequeueOne()
    → emitIntent + enqueueRetire + settleEpoch + reclaim (EpochControl)
    → slot state: Reclaimed (enables slot reuse)
```

### System 2: DSPCore* (actual pointer)
```
DSPLifetimeManager::retire(DSPCore*)
    → retireDSPHandleForRuntime(DSPCore*)
    → requestReclaimHandle(handle)
    → router_->enqueueWithRetry(ptr, deleter, epoch)
    → DeferredDeletionQueue (EpochDomain)
    → drainAllUnsafe() / tryReclaim() → deleter(ptr)
```

**System 1 and System 2 are DECOUPLED**:
- System 1 tracks slot retirement state (for slot reuse)
- System 2 tracks actual pointer lifetime (for memory deallocation)
- `RetireIntent` does NOT carry the `DSPCore*` pointer — it carries `dspSlot` (index into registry)

**GAP-CROSS-3 reclassification**: The original FAIL was based on conflating these systems.
The real risk is NOT pointer ownership loss (System 2 is properly drained), but:
1. `pendingIntentCount() > 0` → `isFullyDrained()` hangs → `waitForDrain()` times out
2. Timed-out shutdown path may skip non-essential drain steps
3. Slot state not cleaned up (ephemeral, non-critical)

## J. Required fix (non-allocating, noexcept, epoch-safe)

Add to BOTH `releaseResources()` and `~AudioEngine()`, after existing drain steps:

```cpp
// Drain OverflowRing → LifetimeState MPSC queue → EpochControl slot state
{
    // 1. OverflowRing → MPSC queue (emitRetireIntent)
    if (worldAuthority_.lifetime().getOverflowRing()) {
        worldAuthority_.lifetime().getOverflowRing()->drainAll(
            worldAuthority_.lifetime());  // calls emitRetireIntent per entry
    }
    // 2. MPSC queue → slot state (dequeueOne + reclaim)
    //    noexcept, non-allocating, epoch-safe (readers exited)
    convo::isr::RetireIntent intent;
    while (worldAuthority_.lifetime().dequeueOne(intent)) {
        if (intent.dspSlot != UINT32_MAX) {
            worldAuthority_.lifetime().emitIntent(intent.dspSlot, intent.generation);
            worldAuthority_.lifetime().enqueueRetire(intent.dspSlot);
            worldAuthority_.lifetime().settleEpoch(intent.dspSlot);
            worldAuthority_.lifetime().reclaim(intent.dspSlot);
        }
    }
    // 3. Fallback queue → slot state
    while (worldAuthority_.lifetime().dequeueFallback(intent)) {
        if (intent.dspSlot != UINT32_MAX) {
            worldAuthority_.lifetime().emitIntent(intent.dspSlot, intent.generation);
            worldAuthority_.lifetime().enqueueRetire(intent.dspSlot);
            worldAuthority_.lifetime().settleEpoch(intent.dspSlot);
            worldAuthority_.lifetime().reclaim(intent.dspSlot);
        }
    }
}
```

**Note**: `RetireOverflowRing::drainAll()` takes `std::vector<RetireOverflowEntry>&` output, not a `LifetimeState&`. The actual drain path needs to use `overflowRing.pop(entry)` + `emitRetireIntent(entry.intent)` directly, matching the pattern in `AudioEngine.Processing.ReleaseResources.cpp:258-259` and `AudioEngine.Threading.cpp:273`.

## K. Proposed test plan

| Test | Description |
|------|-------------|
| P4-5-01 | OverflowRing entries → shutdown drain → slot state Reclaimed |
| P4-5-02 | LifetimeState MPSC queue entries → shutdown drain → slot state Reclaimed |
| P4-5-03 | LifetimeState fallback queue entries → shutdown drain → slot state Reclaimed |
| P4-5-04 | Normal commit processed intents → shutdown → no double reclaim (idempotent) |
| P4-5-05 | Overflow + MPSC + fallback mixed → all drained → pendingIntentCount == 0 |
| P4-5-06 | Destructor-only path → no allocation (no std::vector) |
| P4-5-07 | Repeated shutdown drain → idempotent (second call no-op) |
| P4-5-08 | Epoch reader active → drain waits (no premature deletion) |

## GAP-CROSS-3

```
GAP-CROSS-3: OPEN

Status: Fix NOT yet applied. Root cause confirmed:
  1. OverflowRing not drained in ~AudioEngine()
  2. LifetimeState MPSC queue not drained in any shutdown path
     (only in RT commit path — AudioEngine.Commit.cpp:492)
  3. pendingIntentCount() never reaches 0 → isFullyDrained() hangs →
     waitForDrain() times out → skips drain steps

Fix design: non-allocating, noexcept drain using dequeueOne()/dequeueFallback()
+ emitIntent()/enqueueRetire()/settleEpoch()/reclaim() — all noexcept, no allocation.

Epoch safety: Confirmed safe in destructor (all readers exited, closeReaderRegistration called).
Idempotence: Confirmed (drainOverflowRing, dequeue, reclaim are idempotent).
Ownership model: RetireIntent tracks slot state, DSPCore* tracked separately
  via DeferredDeletionQueue (properly drained by drainAllUnsafe).
```
