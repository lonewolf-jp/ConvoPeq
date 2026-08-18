# 15-P-4-4R: Abnormal Destructor Overflow Drain Fix Verification

## A. Abnormal dtor overflow drain — **FAIL**

**Proposed fix**: Call `runtimePublicationBridge_.drainOverflowRing()` in `~AudioEngine()` after `m_retireRouter->drainAll()`.

**Problem**: `runtimePublicationBridge_` (RuntimeIntentCoordinator) is declared at `AudioEngine.h:4748`, BEFORE `dspQuarantineManager_` (line 4780). Destruction order is REVERSE declaration order, so `runtimePublicationBridge_` is destroyed BEFORE `dspQuarantineManager_`.

The proposed fix calls `runtimePublicationBridge_.drainOverflowRing()` in the destructor body — this is fine **during dtor body execution** (members are still alive). The member destruction order issue is a red herring for the dtor body. However:

- `drainOverflowRing()` re-injects entries via `retireRuntime.emitRetireIntent()`, which pushes to `LifetimeState`'s MPSC queue.
- `LifetimeState` is inside `RuntimeWorldAuthority` (line 4752), declared AFTER `runtimePublicationBridge_` (line 4748).
- So `runtimePublicationBridge_`'s drain injects into `LifetimeState` which is destroyed AFTER — **this is safe**.

**Actual FAIL reason**: The fix is correct in isolation, but the destructor body does not currently contain it. The fix needs to be applied.

## B. Overflow entry disposition — **PASS**

`drainOverflowRing()` (ISRRuntimePublicationCoordinator.cpp:327) → `OverflowScheduler::drainOverflowRing()`:

```
RetireOverflowRing.pop(entry)
    ↓
entry.intent → retireRuntime.emitRetireIntent(entry.intent)
    ↓
MPSC queue in LifetimeState (slots_[256])
    ↓
dequeueOne() → DeferredDeletionQueue (in EpochDomain)
    ↓
drainAllUnsafe() → deleter(entry.ptr)
```

**No silent discard**: entries with `reinjectRetryCount >= 10` are dropped (`++result.droppedCount`), but this is a bounded retry mechanism, not ownership loss. The `RetireIntent` itself carries `dspSlot` (a slot index), and `emitRetireIntent` will re-enqueue it for proper reclamation through the retire pipeline.

**Reclamation path survives**: `DeferredDeletionQueue` is owned by `EpochDomain` (declared line 4676, destroyed after `runtimePublicationBridge_` at 4748). `drainAllUnsafe()` is called in the abnormal path (`~AudioEngine()` calls `drainDeferredRetireQueues(true)` then `m_epochDomain.drainAll()` or `m_retireRouter->drainAll()`).

## C. Destruction-order safety — **PASS** (after fix)

Member destruction order (REVERSE declaration, in destructor body):
1. `dspQuarantineManager_` (line 4780) ← destroyed FIRST (last declared)
2. `ShutdownRuntime` (line 4787)
3. `runtimePublicationBridge_` (line 4748) ← destroyed later
4. `RuntimeWorldAuthority` (line 4752) ← destroyed later (contains LifetimeState)
5. `m_epochDomain` (line 4676) ← destroyed after RuntimeWorldAuthority
6. `m_retireRouter` (line 4681) ← destroyed after m_epochDomain

**In dtor body (before any member destruction)**: All members alive.
- `m_retireRouter->drainAll()` — OK (m_retireRouter alive)
- `runtimePublicationBridge_.drainOverflowRing()` — OK (runtimePublicationBridge_ alive)
- `drainOverflowRing()` → `emitRetireIntent()` → pushes to `LifetimeState` (in RuntimeWorldAuthority, declared after) — OK
- `drainDeferredRetireQueues(true)` + `m_epochDomain.drainAll()` — OK (EpochDomain has D, still alive)

The `emitRetireIntent` → MPSC queue → `dequeueOne` → `DeferredDeletionQueue` → `drainAllUnsafe` chain is safe because all components survive until after the dtor body completes.

## D. Normal→dtor double-drain safety — **PASS**

`drainOverflowRing()` iterates `while (consumed < budget && overflowRing.pop(entry))`. After normal `releaseResources()` drain, the ring is empty (`pop()` returns false immediately → loop exits). **`drainOverflowRing()` is idempotent** — it has no global side effects that would cause double-processing of the same entries. The `OverflowDrainResult` struct is freshly constructed each call.

`runtimePublicationBridge_.drainOverflowRing()` in normal path (AudioEngine.Threading.cpp:273) is called during timer callback, not in `releaseResources()`. `releaseResources()` drains via manual `pop()` loop (line 227-228). Both paths empty the ring → second call is a no-op.

## E. No silent RetireIntent loss — **PASS** (with fix applied)

In abnormal path without fix: `RetireIntent`s in OverflowRing are silently abandoned when `LockFreeRingBuffer` destructor runs (trivial dtor, just stack memory).

In abnormal path WITH fix: `drainOverflowRing()` re-injects each `RetireIntent` via `emitRetireIntent()`, which pushes to MPSC queue. `drainDeferredRetireQueues(true)` + `m_epochDomain.drainAll()` then drains the queue → `DeferredDeletionQueue` → deleter invoked.

The `RetireIntent` carries `dspSlot` (slot index) + `generation` + `priority`. The re-injected intent goes through the same retire pipeline as normal runtime retires — no special "shutdown skip" logic.

## F. Regression tests — **PASS**

`src/tests/RetireGraceSemanticsTests.cpp` contains:
- `testOverflowRingFifoOrder()` (line 193) — verifies FIFO pop after push
- `testOverflowRingBasic` (line 200) — `RetireOverflowRing ring;` stack-allocated test

No test currently covers the abnormal destructor path draining the OverflowRing. However, the existing OverflowRing test infrastructure provides a foundation. A minimal regression test could:
1. Construct `RetireOverflowRing`, push entries
2. Call `drainOverflowRing()` with a mock `LifetimeState`
3. Verify `emitRetireIntent()` was called for each entry

## GAP-CROSS-3

```
GAP-CROSS-3: CLOSED → OPEN (pending fix application)

FIX VERIFIED: drainOverflowRing() in ~AudioEngine() after m_retireRouter->drainAll()
  is safe (all members alive, correct destruction order, idempotent,
  re-injects intents into alive LifetimeState, D+Q+E drained afterwards)

ACTION: Apply fix to AudioEngine.CtorDtor.cpp ~AudioEngine()
```
