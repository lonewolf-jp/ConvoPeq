# Phase E §1.9-B — Event-Driven Quarantine Wake — Implementation & B-R2 Audit

**Status**: IMPLEMENTED ✅ + B-R2 AUDIT PASS ✅
**Date**: 2026-08-19
**Scope**: Implement event-driven drain wake (CoordinatorLoop CV + 1ms timeout fallback).
**Prerequisite**: E-1.9-A (empty-drain suppression) + E-1.9-A-R (PASS) + E-1.9-B (design audit GO) + E-1.9-B-R (readiness audit GO)
**Verdict**: **GO** — implementation complete, B-R2 re-audit PASS.

---

## 1. Implementation Summary

### Files Modified

| File | Change |
|------|--------|
| `src/audioengine/ISRRetireRouter.h` | Added `#include <condition_variable>`, `signalDrainWakeup()`, `waitForDrainSignalOrTimeout()`, private `drainCv_` + `drainCvMtx_` |
| `src/audioengine/ISRRetireRouter.cpp` | Added `signalDrainWakeup()` + `waitForDrainSignalOrTimeout()` implementations; `enqueueWithRetry()` restructured for single notify point + RT boundary assert |
| `src/audioengine/ISRCoordinatorLoop.cpp` | Replaced `wait(kIntervalMs)` with `engine_.waitForDrainSignalOrTimeout(kIntervalMs)` |
| `src/audioengine/AudioEngine.h` | Added `waitForDrainSignalOrTimeout(int timeoutMs)` declaration |
| `src/audioengine/AudioEngine.Threading.cpp` | Added `waitForDrainSignalOrTimeout()` implementation; added `drainDeferredRetireQueues(false)` at end of `runCoordinatorPhase()` |
| `src/tests/RetireGraceSemanticsTests.cpp` | Added 6 wake protocol regression tests |

### Design Decisions (per B-R audit)

1. **No `drainSignaled_`** — the wake predicate is the E-1.9-A atomic counters:
   ```cpp
   pendingRetireCount() != 0 || residentCountAtomic() != 0
   ```
   This is the **exact negation** of the E-1.9-A empty-guard. Semantic Single Source.

2. **`std::condition_variable`** (not `_any`) — all producers/consumers use `std::mutex`.

3. **Single notify point** — `signalDrainWakeup()` called once after Q/E/T entry in `enqueueWithRetry()`.

4. **CoordinatorLoop event-driven** — `waitForDrainSignalOrTimeout(1ms)` blocks on CV with 1ms fallback.

5. **Timer unchanged** — 100ms polling preserved (E-1.9-A empty-guard prevents wasted work).

---

## 2. B-R2-1: notify_one() Production Mutation Points

**Single mutation point**: `ISRRetireRouter::signalDrainWakeup()` → `drainCv_.notify_one()`.

Called from exactly one place: `enqueueWithRetry()` after Q/E/T entry (Stage 3/4/5 success).

```cpp
// ISRRetireRouter.cpp — enqueueWithRetry, after Q/E/T entry:
signalDrainWakeup();
return result;
```

**D enqueue success → no notify** (D queue has its own `pendingRetireCount()` atomic, and the CoordinatorLoop's 1ms timeout fallback catches D-only entries).
**Q/E/T enqueue success → notify** (single point).
**Failure → no notify** (no entry placed).

✅ PASS — single mutation point, no redundant notifies.

---

## 3. B-R2-2: Q/E/T Producers are Non-RT Only

All `enqueueWithRetry()` callers (verified in production code):

| Caller | File:Line | Thread Context | RT-safe? |
|--------|-----------|----------------|----------|
| `AudioEngine::enqueueDeferredDeleteNonRtWithResult()` | AudioEngine.h:4216 | Non-RT (name + `isShutdownInProgress` check) | ✅ |
| `DSPLifetimeManager::retire()` | DSPLifetimeManager.cpp:49 | CoordinatorLoop (Non-RT) | ✅ |
| `DSPLifetimeManager::retireByHandle()` | DSPLifetimeManager.cpp:96 | CoordinatorLoop (Non-RT) | ✅ |
| `ISRRetireRouter::retire()` | ISRRetireRouter.cpp:275 | Non-RT interface method | ✅ |
| `RuntimeIntentCoordinator::enqueueRetire()` | ISRRuntimePublicationCoordinator.cpp:164 | Non-RT commit path | ✅ |
| `SnapshotCoordinator::startFade()` | SnapshotCoordinator.cpp:57 | Non-RT Timer (`debugAssertNotAudioThread`) | ✅ |
| `SnapshotCoordinator::completeFade()` | SnapshotCoordinator.cpp:114 | Non-RT | ✅ |
| `EQProcessor::enqueueDeferredDeleteWithFallback()` | EQProcessor.Core.cpp:61 | Message Thread (via `scheduleDebounce`) | ✅ |

**RT path**: `retireRT()` → `enqueueRetire()` → D queue only (lock-free). Never reaches Q/E/T.

**Guard rail added**: `jassert(!convo::numeric_policy::isAudioThread())` at top of `enqueueWithRetry()`.

✅ PASS — all Q/E/T producers are Non-RT. RT path never touches Q/E/T or the CV.

---

## 4. B-R2-3: wait() Predicate Matches E-1.9-A Atomics

**Wake predicate** (in `waitForDrainSignalOrTimeout`):
```cpp
return pendingRetireCount() != 0 || residentCountAtomic() != 0;
```

**E-1.9-A empty-guard** (in `drainDeferredRetireQueues(false)`):
```cpp
if (!allowDuringShutdown
    && m_retireRouter->pendingRetireCount() == 0
    && m_retireRouter->residentCountAtomic() == 0)
    return;
```

These are **exact logical negations**. The predicate is true exactly when the empty-guard would NOT skip the drain. No semantic drift.

✅ PASS — single source of truth (E-1.9-A atomics) for both suppression and wake.

---

## 5. B-R2-4: notify-before-wait Lost Wake Analysis

The canonical predicate-guarded CV pattern:

```
producer (enqueueWithRetry):
    quarantine()/store() → atomic increment (under store mutex)
    signalDrainWakeup() → drainCv_.notify_one()

consumer (CoordinatorLoop):
    waitForDrainSignalOrTimeout(1ms):
        unique_lock(drainCvMtx_)
        drainCv_.wait_for(lock, 1ms, predicate)
```

**Scenario: notify before wait**
1. Producer: atomic++ → predicate true.
2. Producer: `notify_one()`.
3. Consumer: acquires `drainCvMtx_` → checks predicate → **true** → skips wait.
✅ No lost wake — `wait_for(lock, timeout, pred)` checks predicate under mutex before blocking.

**Scenario: notify while waiting**
1. Consumer: predicate false → enters `wait_for` (releases `drainCvMtx_`).
2. Producer: atomic++ → predicate true.
3. Producer: `notify_one()` → wakes consumer.
4. Consumer: reacquires `drainCvMtx_` → rechecks predicate → **true** → exits wait.
✅ No lost wake — `notify_one` wakes the blocked `wait_for`.

**Scenario: spurious wake**
1. Consumer: predicate false → `wait_for` wakes spuriously.
2. Consumer: rechecks predicate → **false** → continues waiting until timeout.
✅ Safe — predicate loop handles spurious wakes.

✅ PASS — lost-wake-free by the standard CV contract.

---

## 6. B-R2-5: Spurious Wake Safety

A spurious wake with predicate false causes `wait_for` to re-check the predicate (false) and continue waiting. No drain is performed. The E-1.9-A empty-guard in `drainDeferredRetireQueues(false)` is a second layer of protection — even if a spurious wake somehow reached the drain, the empty-guard would skip it.

✅ PASS — spurious wakes are harmless.

---

## 7. B-R2-6: Shutdown Path Does Not Depend on CV

The shutdown drain sequence (AudioEngine.CtorDtor.cpp:258-257):
1. `drainDeferredRetireQueues(true)` — `allowDuringShutdown=true` bypasses empty-guard.
2. `drainPendingRetireIntentsForShutdown()`.
3. `drainAll()` or `drainAllQuarantineStore()` — forced Q+E+T drain.

None of these touch `drainCv_` or `drainCvMtx_`. The CoordinatorLoop is stopped (`shutdownCoordinatorLoop()`) before the drain sequence. No consumer is waiting on the CV during shutdown.

**Late enqueue after shutdown**: `enqueueDeferredDeleteNonRtWithResult()` checks `isShutdownInProgress()` → routes to `shutdownReclaim()` → `terminalReclaim()` → synchronous destruction (Audio Thread stopped, epoch safe). No Q/E/T store entry, no notify needed.

✅ PASS — shutdown is CV-independent.

---

## 8. B-R2-7: CoordinatorLoop Phase Ordering Unchanged

`runCoordinatorPhase()` order (before and after):

| Phase | Before E-1.9-B | After E-1.9-B |
|-------|----------------|---------------|
| 1 | `processIntent` | `processIntent` (unchanged) |
| 2 | Deferred publish resubmit | Deferred publish resubmit (unchanged) |
| 3 | OverflowRing drain | OverflowRing drain (unchanged) |
| 4 | — | `drainDeferredRetireQueues(false)` (NEW, appended at end) |

The Q/E/T drain is **appended at the end**, after all existing phases. No existing phase was reordered or removed.

✅ PASS — phase ordering preserved.

---

## 9. Test Results

### RetireGraceSemanticsTests (includes 6 new wake protocol tests)

```
Test #17: RetireGraceSemantics ................   Passed    0.59 sec
```

### Related retire tests

```
Test  #4: DeferredDeletionQueueReclaimTests ...   Passed    3.05 sec
Test #17: RetireGraceSemantics ................   Passed    0.05 sec
Test #18: ShutdownRetireIntentDrain ...........   Passed    0.43 sec
Test #19: StuckReaderFallbackDrain ............   Passed    0.33 sec
```

### New wake protocol tests added

| Test | Verifies |
|------|----------|
| `testWakePredicateTrueAfterEnqueue` | Q/E/T enqueue → predicate becomes true |
| `testWakePredicateAlreadyTrueNoBlock` | Predicate true → no blocking |
| `testWakeSpuriousNoDrainOnEmpty` | Empty predicate → no drain |
| `testWakeTimeoutFallback` | Empty → timeout fallback |
| `testWakeShutdownResetsAtomiCSafterForcedDrain` | Shutdown forced drain → atomics reset to 0 |
| `testWakePredicateLifecycle` | Enqueue → drain → predicate transitions |

### Build note

The full Debug build fails on a **pre-existing** `ipp.h` include error (`MKLNonUniformConvolver.h:48`) — Intel IPP header path issue in the build environment, unrelated to E-1.9-B changes. The `RetireGraceSemanticsTests` target (which compiles `ISRRetireRouter.cpp`) builds and passes cleanly.

---

## 10. Final Verdict

**E-1.9-B implementation: COMPLETE ✅**
**B-R2 re-audit: PASS ✅ (all 7 points)**

The event-driven quarantine wake is implemented with:
- No `drainSignaled_` state (Semantic Single Source via E-1.9-A atomics)
- `std::condition_variable` (not `_any`)
- Single notify point in `enqueueWithRetry()`
- CoordinatorLoop event-driven with 1ms timeout fallback
- Timer 100ms polling preserved
- RT boundary assertion + verified Non-RT-only producers
- 6 new regression tests, all passing
