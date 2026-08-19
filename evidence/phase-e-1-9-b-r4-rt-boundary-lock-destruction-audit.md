# Phase E §1.9-B-R4 — RT Boundary / Lock Contention / Destruction Ordering Final Audit

**Status**: AUDIT PASS ✅
**Date**: 2026-08-19
**Scope**: Final audit of RT boundary, `drainCvMtx_` lock contention, and shutdown/destruction lifetime for the E-1.9-B event-driven wake.
**Prerequisite**: E-1.9-A + E-1.9-A-R + E-1.9-B design + E-1.9-B implementation + E-1.9-B-R2 + E-1.9-B-R3 (lost-wake fix)
**Verdict**: **PASS** — all 6 gates satisfied. The `std::mutex` introduced in R3 does NOT reach the RT path, and the dual-consumer (Timer + CoordinatorLoop) concurrent drain is safe.

---

## R4-1. `drainCvMtx_` / `drainCv_` Complete Access Enumeration

### Production code

| Access | Location | Type |
|--------|----------|------|
| `std::lock_guard<std::mutex> lock(drainCvMtx_)` | `ISRRetireRouter.cpp:451` (`signalDrainWakeup`) | lock_guard |
| `drainCv_.notify_one()` | `ISRRetireRouter.cpp:452` (`signalDrainWakeup`) | notify_one |
| `std::unique_lock<std::mutex> lock(drainCvMtx_)` | `ISRRetireRouter.cpp:463` (`waitForDrainSignalOrTimeout`) | unique_lock |
| `drainCv_.wait_for(lock, timeout, predicate)` | `ISRRetireRouter.cpp:464` (`waitForDrainSignalOrTimeout`) | wait_for |
| `std::condition_variable drainCv_` | `ISRRetireRouter.h:375` | member declaration |
| `std::mutex drainCvMtx_` | `ISRRetireRouter.h:376` | member declaration |

### Test-only code

| Access | Location | Type |
|--------|----------|------|
| `testDrainCv()` → `drainCv_` | `ISRRetireRouter.h:330` | test-only accessor |
| `testDrainCvMutex()` → `drainCvMtx_` | `ISRRetireRouter.h:331` | test-only accessor |

### Absent operations (verified)

- **No `notify_all()`** — only `notify_one()`.
- **No destructor/move/copy** — `ISRRetireRouter` is non-copyable/non-movable (deleted copy/move ctors). `drainCv_`/`drainCvMtx_` are destroyed with the router.
- **No direct member access outside `ISRRetireRouter`** — production code accesses only via `signalDrainWakeup()` / `waitForDrainSignalOrTimeout()`.

✅ **PASS** — production access fully enumerated.

---

## R4-2. RT Call Graph Re-verification

### RT path (Audio Thread)

```
AudioEngine::processBlock()  [ASSERT_AUDIO_THREAD + ThreadRole::AudioRealtime]
  → DSP processing
  → m_coordinator.advanceFade(numSamples)  [counter decrement only]
  → diagnostic atomic reads (e.g., retireQueueDepth_)
```

**The RT path does NOT call** any of:
- `retireRT()` — defined but **NOT called from production** (only `RefCountedDeferred::releaseRT` which is also unused)
- `enqueueRetireEpochBounded()` — defined but **NOT called from production**
- `enqueueWithRetry()` — Non-RT only (verified R3-5)
- `quarantine()` / `emergencyQuarantine()` / `terminalReclaim()` — Non-RT only
- `signalDrainWakeup()` — Non-RT only (called from `enqueueWithRetry`)
- `waitForDrainSignalOrTimeout()` — Non-RT only (called from CoordinatorLoop)
- `drainCvMtx_` / `drainCv_` — Non-RT only

### Structural guarantee (Release build, asserts disabled)

The `ASSERT_AUDIO_THREAD()` / `ASSERT_NON_RT_THREAD()` macros use `jassert` which is **compiled out in Release builds**. Therefore the safety does NOT rely on assertions. The structural call graph is the guarantee:

1. `retireRT()` (the only RT-designated retire entry) has **zero production callers**.
2. All actual Q/E/T producers (`enqueueWithRetry`, `quarantine`, `emergencyQuarantine`, `terminalReclaim`) are reachable ONLY from Non-RT contexts (CoordinatorLoop, MessageThread Timer, commit path — verified R3-5).
3. `signalDrainWakeup()` is called ONLY from `enqueueWithRetry()` (Non-RT).
4. `waitForDrainSignalOrTimeout()` is called ONLY from `CoordinatorLoop::run()` (Non-RT juce::Thread).

**Even with all asserts disabled, the RT thread has no code path to the mutex or CV.**

✅ **PASS** — RT cannot reach `drainCvMtx_` structurally.

---

## R4-3. `signalDrainWakeup()` Mutex Hold Time

```cpp
void ISRRetireRouter::signalDrainWakeup() noexcept
{
    std::lock_guard<std::mutex> lock(drainCvMtx_);
    drainCv_.notify_one();
}
```

### Verification

| Concern | Result |
|---------|--------|
| Mutex held during anything other than `notify_one()` | ❌ No — critical section contains ONLY `notify_one()` |
| Destructor / callback / logging implicitly called | ❌ No — `notify_one()` is a non-blocking primitive |
| Recursive path (re-acquire `drainCvMtx_`) | ❌ No — `notify_one()` does not call back into the router; the wait predicate (`pendingRetireCount` / `residentCountAtomic`) is lock-free |
| `notify_one()` blocking behavior | Non-blocking — wakes a waiter but does not wait for it |

**The critical section is minimal and non-blocking.** The mutex is held only for the duration of the `notify_one()` call.

✅ **PASS** — critical section limited to `notify_one()`.

---

## R4-4. Predicate / State Mutation Atomic Ordering

### Memory orders (verified in code)

| Operation | Location | Memory Order |
|-----------|----------|--------------|
| `RetireQuarantineStore::quarantine()` → `residentAtomic_.fetch_add(1)` | RetireQuarantineStore.h:93 | `release` |
| `RetireQuarantineStore::drain()` → `residentAtomic_.fetch_sub(pendingCount)` | RetireQuarantineStore.h:133 | `release` |
| `RetireQuarantineStore::drainAllUnsafe()` → `residentAtomic_.store(0)` | RetireQuarantineStore.h:171 | `release` |
| `TerminalReclaimAuthority::store()` → `residentAtomic_.fetch_add(1)` | ISRRetireRouter.cpp:35 | `release` |
| `TerminalReclaimAuthority::drain()` → `residentAtomic_.fetch_sub(pending.size())` | ISRRetireRouter.cpp:65 | `release` |
| `TerminalReclaimAuthority::drainAll()` → `residentAtomic_.store(0)` | ISRRetireRouter.cpp:84 | `release` |
| `residentCountAtomic()` → `consumeAtomic(residentAtomic_, ...)` | RetireQuarantineStore.h:197 | `acquire` |

### Ordering analysis

**Producer side** (release):
```
residentAtomic_.fetch_add(1, release)   // predicate becomes true
    ↓
signalDrainWakeup() → lock(drainCvMtx_) → notify_one() → unlock
```

**Consumer side** (acquire):
```
lock(drainCvMtx_)
    ↓
predicate check: residentCountAtomic()  // acquire load → sees the increment
    ↓
wait_for (if false)
```

**Separation of concerns**:
- **Wake correctness ordering**: The `drainCvMtx_` mutex serializes `notify_one()` with the consumer's wait transition (R3 fix). This is the CV protocol synchronization.
- **Count visibility**: The atomic `release`/`acquire` pairing ensures the consumer's predicate load sees the producer's increment. This is the data visibility.

Both are present and correctly paired. The `release` on the producer's increment synchronizes-with the `acquire` on the consumer's predicate load, establishing the happens-before edge for the counter value.

✅ **PASS** — atomic ordering is correct; wake correctness and count visibility are properly separated.

---

## R4-5. Shutdown / Destructor Lifetime

### Shutdown sequence (`~AudioEngine`, AudioEngine.CtorDtor.cpp:96-275)

```
StopAcceptingWork → lifecycleState = Releasing
StopAudio → stopTimer()
StopWorkers → shutdownCoordinatorLoop()  [join CoordinatorLoop] + stopRebuildThread()
  ... (detach published runtime pointers, retire captured runtimes)
ForceEpochAdvance
DrainRetire → drainAll() / drainAllQuarantineStore()
Destroy → lifecycleState = Destroyed
```

### Verification of the 5 concerns

| # | Concern | Result |
|---|---------|--------|
| 1 | CoordinatorLoop in `waitForDrainSignalOrTimeout()` while `ISRRetireRouter` destroyed | ❌ No — `shutdownCoordinatorLoop()` joins the thread at `StopWorkers`, BEFORE the router is destroyed. Member destruction order: `coordinatorLoop_` (declared after) destroyed BEFORE `m_retireRouter`. |
| 2 | `drainCv_` destroyed then producer calls `signalDrainWakeup()` | ❌ No — all producers (`enqueueWithRetry`) are stopped before `Destroy` phase. `shutdownReclaim()` (the only late-enqueue path) does NOT call `signalDrainWakeup()`. |
| 3 | `shutdownReclaim()` touches CV | ❌ No — `shutdownReclaim()` → `terminalReclaim()` → synchronous destruction or `m_terminalReclaim.store()`. No CV access. |
| 4 | `drainAllQuarantineStore()` touches CV | ❌ No — calls `drainAllUnsafe()` on Q/E + `drainAll()` on T. No CV access. |
| 5 | `threadShouldExit()` vs CV wake deadlock | ❌ No — `waitForDrainSignalOrTimeout(1ms)` wakes every 1ms (timeout) to check `threadShouldExit()`. `stopThread(2000)` join completes within ~1ms. |

### CoordinatorLoop exit path
```cpp
void CoordinatorLoop::run() {
    while (!threadShouldExit()) {
        if (engine_.isShutdownInProgress()) break;
        engine_.runCoordinatorPhase();
        engine_.waitForDrainSignalOrTimeout(kIntervalMs);  // 1ms timeout
    }
}
```
`stopLoop()` → `signalThreadShouldExit()` + `stopThread(2000)`. The 1ms timeout guarantees the loop checks `threadShouldExit()` within 1ms. **No shutdown deadlock.**

✅ **PASS** — shutdown/destruction lifetime is safe.

---

## R4-6. Timer + CoordinatorLoop Dual Consumer (PRIORITY)

### The two consumers

| Consumer | Thread | Cadence | Drain call |
|----------|--------|---------|------------|
| CoordinatorLoop | Non-RT juce::Thread | event-driven + 1ms timeout | `runCoordinatorPhase()` → `drainDeferredRetireQueues(false)` |
| MessageThread Timer | Non-RT MessageThread | 100ms polling | `tryReclaimResources()` + `drainDeferredRetireQueues(false)` + `processDeferredReleases()` |

Both can call `drainDeferredRetireQueues(false)` concurrently.

### Concurrent drain safety analysis

| Component | Synchronization | Concurrent-safe? |
|-----------|----------------|------------------|
| **D queue** (`DeferredDeletionQueue::reclaim`) | Lock-free CAS on `dequeuePos` | ✅ — only one CAS succeeds per slot; the other retries with updated position. No double-free. |
| **Q store** (`RetireQuarantineStore::drain`) | `mtx_` (mutex) for extraction; atomic `residentAtomic_` for counter | ✅ — concurrent drains serialize on `mtx_`. Each entry extracted exactly once. |
| **E store** (`EmergencyQuarantineStore::drain`) | Same as Q | ✅ |
| **T store** (`TerminalReclaimAuthority::drain`) | `mtx_` for extraction; atomic `residentAtomic_` | ✅ |
| **`residentAtomic_` counter** | Atomic `fetch_add`/`fetch_sub`/`store` | ✅ — each entry counted exactly once (quarantine +1, drain -N). Invariant `residentAtomic_ == size_` maintained under concurrent drain. |
| **`onReclaimBegin`/`onReclaimEnd`** | Atomic counters with `old > 0` guard | ✅ — concurrent begin/end net to 0. No underflow. |
| **`pendingReclaimHandles_`** | `pendingReclaimHandlesMutex_` | ✅ — extraction and re-registration serialized. |

### Counter invariant proof (concurrent quarantine + drain)

```
Initial: size_ = 3, residentAtomic_ = 3
Thread A (drain): lock mtx_, extract 2 safe entries, size_ = 1, unlock
Thread P (quarantine): lock mtx_, add 1 entry, size_ = 2, residentAtomic_ = 4, unlock
Thread A: residentAtomic_.fetch_sub(2) → 4 - 2 = 2
Final: size_ = 2, residentAtomic_ = 2  ✅ CONSISTENT
```

The counter is consistent because:
- Each `quarantine()` adds exactly 1 to `size_` and +1 to the counter.
- Each `drain()` removes exactly N from `size_` (under mutex) and -N from the counter (after mutex).
- Total counter = total quarantines - total drained = total added - total removed = `size_`.

### Redundant work assessment

Both consumers may drain the same entries (redundant work), but:
- The E-1.9-A empty-guard prevents wasted work when empty.
- The store mutexes serialize the actual drain.
- No double-free, no lost entries, no counter drift.

**The dual consumer is safe with clear serialization at the store mutex level and lock-free CAS at the D queue level.**

✅ **PASS** — concurrent drain is safe.

---

## R4 Final Verdict

| Gate | Condition | Result |
|------|-----------|--------|
| R4-1 | `drainCvMtx_` production access fully enumerated | ✅ PASS |
| R4-2 | RT call graph cannot reach mutex/CV structurally | ✅ PASS |
| R4-3 | Signal-side mutex critical section limited to `notify_one()` | ✅ PASS |
| R4-4 | Predicate/state atomic ordering correct | ✅ PASS |
| R4-5 | Shutdown/destruction vs CV lifetime — no race/deadlock | ✅ PASS |
| R4-6 | Timer/Coordinator concurrent drain safe (serialized) | ✅ PASS |

**E-1.9-B-R4: PASS ✅**

The `std::mutex` introduced in R3 (lost-wake fix) does NOT reach the RT path — the structural call graph guarantees RT cannot touch `drainCvMtx_` even with asserts disabled. The dual-consumer (Timer + CoordinatorLoop) concurrent drain is safe: D queue uses lock-free CAS, Q/E/T stores serialize on their mutexes, and the atomic resident counter maintains its invariant under concurrent drain.

---

## Files Referenced

| File | Role |
|------|------|
| `src/audioengine/ISRRetireRouter.cpp` | `signalDrainWakeup()`, `waitForDrainSignalOrTimeout()`, `enqueueWithRetry()`, `shutdownReclaim()`, `drainAllQuarantineStore()`, `TerminalReclaimAuthority::drain()` |
| `src/audioengine/ISRRetireRouter.h` | `drainCv_`/`drainCvMtx_` members, test-only accessors |
| `src/audioengine/RetireQuarantineStore.h` | `quarantine()`, `drain()`, `drainAllUnsafe()`, `residentAtomic_` memory orders |
| `src/audioengine/AudioEngine.Retire.cpp` | `drainDeferredRetireQueues()`, `tryReclaimResources()` |
| `src/audioengine/AudioEngine.CtorDtor.cpp` | `~AudioEngine()` shutdown sequence |
| `src/audioengine/ISRCoordinatorLoop.cpp` | `CoordinatorLoop::run()`, `stopLoop()` |
| `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | `processBlock()` — RT path |
| `src/audioengine/AudioEngine.Commit.cpp` | `onRuntimeRetiredNonRt()` — Non-RT commit path |
| `src/audioengine/AudioEngine.Threading.cpp` | `quarantineSlot()` (ASSERT_NON_RT), `runCoordinatorPhase()` |
| `src/core/EpochDomain.h` | `tryReclaim()`, `pendingRetireCount()` |
| `src/DeferredDeletionQueue.h` | `reclaim()` (lock-free CAS), `enqueue()` |
| `src/core/SnapshotCoordinator.h` | `reclaim()` → `tryReclaim()` |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp` | `onReclaimBegin/End()` |
| `src/DspNumericPolicy.h` | `ASSERT_AUDIO_THREAD` / `ASSERT_NON_RT_THREAD` macros |
| `src/RefCountedDeferred.h` | `releaseRT()` → `retireRT()` (unused in production) |

---

**Next step**: With R4 PASS, proceed to **E-1.9-A/B commit-pre-audit (R5: diff/invariant/test/evidence consistency)** before committing the combined changes.
