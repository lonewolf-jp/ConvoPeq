# Phase E §1.9 — Quarantine Wake Optimization Audit

**Phase:** Phase E §1.9 (Quarantine Wake Optimization — Tier 2)
**Date:** 2026-08-19
**Status:** INVESTIGATE — code exploration and impact assessment (no changes made)
**Scope:** Read-only code audit. No production code changes in this turn.

---

## 1. Objective

Investigate the wake/notify/polling mechanisms surrounding
`QuarantineStore` / `drainAllQuarantineStore()` to determine whether
unnecessary wake signals, polling cycles, or timer wakes can be reduced
without violating ISR invariants.

Key questions:

1. Who are the **wake producers** (who calls `notify_one` / `notify_all` / `wake`)?
2. Who are the **wake consumers** (who waits on CV / event / timer / polling)?
3. What is the **current wait mechanism** (Event / CV / timer / polling)?
4. **RT safety** — does the audio thread touch any blocking primitive?
5. **Redundant wake** candidates — which wake signals are consumed by
   a path that would drain anyway via timer/polling?
6. **Lost-wake risk** — is there any scenario where a quarantine enqueue
   is not followed by a wake, causing indefinite delay?
7. **Shutdown interaction** — does `drainAllQuarantineStore()` (shutdown)
   interfere with the normal wake path?
8. **Authority impact** — does any optimization change the D→Q→E→T
   authority transfer chain established in 15-P-4→P-13?

---

## 2. Architecture Overview

### 2.1 Ownership Chain (invariant from 15-P-4→P-13, unchanged)

```text
Audio Thread (RT)
  → enqueueRetire() / enqueueDeferredDeleteNonRt()
  → DeferredDeletionQueue (D) — lock-free MPMC, capacity 4096
  → RetireQuarantineStore (Q) — mutex-guarded, capacity 512
  → EmergencyQuarantineStore (E) — mutex-guarded, capacity 512
  → TerminalReclaimAuthority (T) — growable, singleton per AudioEngine
```

**Key constraint:** The Audio Thread (RT) must never block on a mutex,
condition_variable, or Event. All wake/notify mechanisms are Non-RT.

### 2.2 Threads Involved

| Thread | Type | RT? | Role |
| --- | --- | --- | --- |
| Audio Thread (callback) | RT | Yes | Enqueue retires (lock-free), observe only |
| CoordinatorLoop | Non-RT (juce::Thread) | No | 1ms tick: processIntent + overflow drain |
| RebuildThread | Non-RT (juce::Thread) | No | Wait on `rebuildCV`, execute rebuild tasks |
| MessageThread / Timer | Non-RT (juce::Timer) | No | 100ms periodic observation + drain |
| DSPLifetimeManager | Non-RT (called from Coordinator/Timer) | No | Retire DSP handles, drain quarantine |

---

## 3. Wake Producers (who signals?)

### 3.1 `rebuildCV.notify_one()` / `notify_all()`

**Call sites:**

| File:Line | Context | Signal | Wake Target |
| --- | --- | --- | --- |
| `AudioEngine.h:4436` | `submitRecoveryIntent()` — non-absorbed recovery request | `rebuildCV.notify_all()` | RebuildThread |
| `AudioEngine.RebuildDispatch.cpp:724` | `requestRebuild()` — new task queued | `rebuildCV.notify_all()` | RebuildThread |
| `AudioEngine.Threading.cpp:266` | `runCoordinatorPhase()` — deferred publish retry | `rebuildCV.notify_one()` | RebuildThread |
| `AudioEngine.RebuildDispatch.cpp:779` | `stopRebuildThread()` — shutdown signal | `rebuildCV.notify_all()` | RebuildThread |

**Phase E annotation:** `submitRecoveryIntent` at AudioEngine.h:4406 is
explicitly marked `★ dash2 §1.9 (Phase E — quarantine wake optimization)`:
a `RecoveryAdmissionPolicy` that silently absorbs recovery requests when no
authoritative published runtime exists (`!hasAuthoritativePublishedRuntime()`),
incrementing `quarantineAbsorptionCount_` instead of waking the builder.

### 3.2 Other notify/signal mechanisms

- `AudioEngine.Commit.cpp:458` — `runtimeOrchestrator_->notifyWorldRetired(world->worldId)`
  — Non-RT orchestrator notification (not a CV/Event wake).
- `AudioEngine.Processing.ReleaseResources.cpp:235` — `std::this_thread::sleep_for(10ms)`
  — polling sleep in graceful drain loop (Non-RT, shutdown-only).
- `AudioEngine.RebuildDispatch.cpp:828` — `rebuildCV.wait(lock, predicate)`
  — RebuildThread CV wait (consumer side, see §4).

### 3.3 No CV/Event in Quarantine Drain

**Critical finding:** Neither `RetireQuarantineStore::quarantine()`,
`ISRRetireRouter::enqueueWithRetry()`, nor
`TerminalReclaimAuthority::store()` issues any `notify_one` / `notify_all`
or sets any event. The quarantine stores are **passive** — they do not
wake any consumer when entries are added.

---

## 4. Wake Consumers (who waits?)

### 4.1 RebuildThread — `rebuildCV.wait()`

```text
// AudioEngine.RebuildDispatch.cpp:826-831
rebuildCV.wait(lock, [this] {
    return hasPendingTask
        || publishRetryReady
        || recoveryPending      // ★ work88: Recovery Intent wake
        || convo::consumeAtomic(rebuildThreadShouldExit, std::memory_order_acquire);
});
```text

The RebuildThread waits on `rebuildCV` with a predicate checking 4 conditions:
`hasPendingTask`, `publishRetryReady`, `recoveryPending`, `rebuildThreadShouldExit`.
It is woken by `rebuildCV.notify_all()`/`notify_one()` from §3.1.

### 4.2 CoordinatorLoop — `juce::Thread::wait(1ms)`

```text
// ISRCoordinatorLoop.cpp:42-44
engine_.runCoordinatorPhase();
wait(kIntervalMs);  // kIntervalMs = 1
```

The CoordinatorLoop is a **1ms periodic timer** using `juce::Thread::wait()`
(a blocking sleep, not a spin). It does not use a condition variable — it
polls every 1ms. This is the primary driver of:

- `runtimePublicationBridge_.processIntent()` (Coordinator work)
- `runtimePublicationBridge_.drainOverflowRing()` (overflow drain)
- `m_retireRouter->tryReclaim()` (if drainOverflowRing reinjected > 0)

### 4.3 MessageThread / Timer — `timerCallback()` (100ms)

The MessageThread timer fires every ~100ms and calls `processDeferredReleases()`
→ `drainDeferredRetireQueues(false)` → `m_retireRouter->tryReclaim()` +
`m_coordinator.reclaim()`. This is the **periodic drain path** for Q/E/T.

**No wake signal triggers the timer** — it is purely time-based.

---

## 5. Current Wait Mechanism Summary

| Component | Wait Mechanism | Periodicity | Non-RT? | Blocking Primitive? |
| --- | --- | --- | --- | --- |
| RebuildThread | `rebuildCV.wait(predicate)` | Event-driven | Yes | yes (CV + mutex) |
| CoordinatorLoop | `juce::Thread::wait(1ms)` | Polling | Yes | yes (sleep) |
| MessageThread Timer | `juce::Timer` (100ms) | Polling | Yes | yes (timer queue) |
| Audio Thread | None | N/A | No (RT) | **No** |
| Quarantine stores | None (passive) | N/A | — | **No** |

---

## 6. RT Safety Verification

### 6.1 RT thread touchpoints

The Audio Thread interacts with the retire system via:

1. **`enqueueRetire`** (`ISRRetireRouter.cpp:270`) — calls
   `enqueueWithRetry()` which attempts lock-free enqueue to D, then
   falls back to Q/E/T via `quarantine()` (mutex-guarded).

   **⚠️ RT thread acquires mutex in quarantine path:**
   `RetireQuarantineStore::quarantine()` uses `std::lock_guard<std::mutex> lock(mtx_)`.
   This is also noted in the class documentation:
   *"スレッド安全: 全操作は NonRT（Timer / CoordinatorLoop / DSPLifetimeManager）から。
   mutable std::mutex で保護（RT パスからは参照されない"*

2. **`tryReclaimResources`** (`AudioEngine.Retire.cpp:35`) — calls
   `m_retireRouter->tryReclaim()` which drains Q/E/T (mutex-guarded).

   **However:** `tryReclaimResources` is documented as Non-RT and is called
   from:
   - `timerCallback()` (MessageThread, line 1683)
   - `runCoordinatorPhase()` (CoordinatorLoop, line 276 — after OverflowRing drain)
   - Shutdown path (ReleaseResources.cpp:238, 268, 333)

   **Not called from Audio Thread** — verified.

### 6.2 mutex usage on RT path

`enqueueWithRetry()` (ISRRetireRouter.cpp:277) is called from:

- `enqueueRetireEpochBounded()` (AudioEngine.h:4268) — called from RT context
- `retireDSPHandleForRuntime()` → `enqueueRetire()` (AudioEngine.Retire.cpp)

The `enqueueWithRetry` flow:

1. `enqueueRetire()` → `provider_->enqueue()` (lock-free DeferredDeletionQueue) ✅
2. If queue full → `provider_->tryReclaim()` (drains D) ✅
3. If still full → `m_retireQuarantine.quarantine()` (mutex!) ⚠️
4. If Q full → `m_emergencyQuarantine.quarantine()` (mutex!) ⚠️
5. If E full → `m_terminalReclaim.store()` (mutex, via TerminalReclaimAuthority) ⚠️

**⚠️ Finding: The RT thread can acquire mutexes in the quarantine/emergency/terminal
fallback path.** This is a pre-existing design decision documented in the code:
*"allocation-free: std::array + index 配置（noexcept 保証下で push_back は禁止）。"* — the mutex
is expected to be uncontended in normal operation (only one RT thread), and the
fallback to Q/E/T only occurs under high pressure.

### 6.3 condition_variable / Event on RT path

- `rebuildCV` — only used in RebuildDispatch.cpp (Non-RT thread). ✅
- `pendingReclaimHandlesMutex_` — used in `isFullyDrained()`
  (AudioEngine.Threading.cpp:149) and `drainPendingRetireIntentsForShutdown()`.
  `isFullyDrained()` is called from `waitForDrain()` which has
  `ASSERT_NON_RT_THREAD()`. ✅
- No `Event` or `wait_for`/`wait_until` on RT path. ✅

**Verdict:** RT thread does not touch CV/Event/timer. It may touch mutexes
in the retirement fallback path (Q/E/T), but this is documented and uncontended
in normal operation.

---

## 7. Redundancy Analysis — Wake Signals vs. Timer/Polling

### 7.1 Current wake signals for retirement/drain

| Wake signal | Trigger | Consumer | Also covered by timer? |
| --- | --- | --- | --- |
| `rebuildCV.notify_all()` (AudioEngine.h:4436) | `submitRecoveryIntent` (non-absorbed) | RebuildThread | No — RebuildThread blocks on CV, not timer |
| `rebuildCV.notify_one()` (Threading.cpp:266) | Deferred publish retry | RebuildThread | No — RebuildThread blocks on CV |
| `rebuildCV.notify_all()` (RebuildDispatch.cpp:724) | `requestRebuild` task queued | RebuildThread | No |

### 7.2 Quarantine drain wake signals

**Critical finding:** The quarantine drain path (`tryReclaim` →
`drainQuarantineStore` → `drainEmergencyAndTerminal`) is triggered by:

1. **CoordinatorLoop (1ms polling)** — `runCoordinatorPhase()` calls
   `drainOverflowRing()` → if reinjected, calls `tryReclaim()`
2. **Timer (100ms)** — `timerCallback()` calls `processDeferredReleases()`
   → `drainDeferredRetireQueues(false)` → `tryReclaim()`
3. **Shutdown path** — `drainAllQuarantineStore()` (forced drain)

**There is NO wake signal for quarantine drain.** The only "wake" mechanism
is the periodic CoordinatorLoop (1ms) and Timer (100ms).

### 7.3 Redundant wake candidates

| # | Candidate | Current behavior | Optimization opportunity |
| --- | --- | --- | --- |
| C1 | `rebuildCV.notify_all()` in `submitRecoveryIntent` (line 4436) | Fires even when recovery obligation is absent (`!hasAuthoritativePublishedRuntime()` → silent absorb) | ✅ Already optimized: P-4-11 annotation states "absorb — enqueue も wake もしない" |
| C2 | Timer-driven `processDeferredReleases()` (100ms) | Calls `drainDeferredRetireQueues` every 100ms unconditionally | ⚠️ Could be suppressed when `isFullyDrained()` or when no quarantine residents exist |
| C3 | CoordinatorLoop `tryReclaim()` (1ms) | Only called when `drainOverflowRing.reinjectedCount > 0` | ✅ Already conditional — not always called |
| C4 | `rebuildCV.notify_all()` on every `requestRebuild` | Fires even if RebuildThread is already awake processing a task | ⚠️ Could coalesce; but predicate-based `wait()` already handles spurious wakes efficiently |

### 7.4 Phase E annotation: `quarantineAbsorptionCount`

The code already has Phase E infrastructure:

- `AudioEngine.h:1560` — `quarantineAbsorptionCount` telemetry
- `AudioEngine.h:4406` — `RecoveryAdmissionPolicy` with silent absorb
- `AudioEngine.h:4414` — `quarantineAbsAbsorptionCount_.fetch_add(1)` (absorb, no wake)

This is the **existing Phase E optimization**: recovery requests are silently
absorbed (no wake) when no authoritative runtime exists.

---

## 8. Lost-Wake Risk Assessment

### 8.1 Quarantine enqueue does NOT wake any consumer

Since `RetireQuarantineStore::quarantine()` does not issue any wake signal,
the only mechanism for draining Q is:

- **CoordinatorLoop (1ms)** → periodic `tryReclaim()` (only if OverflowRing
  reinjected entries)
- **Timer (100ms)** → periodic `drainDeferredRetireQueues()`

**Lost-wake risk:** If entries are enqueued to Q/E/T and neither the
CoordinatorLoop nor the Timer fires for a period, entries remain quarantined
longer than necessary. However, this is **bounded** by the 100ms timer
period — worst case latency is ~100ms.

**For Q specifically:** `drainQuarantineStore()` is only called from
`tryReclaim()`, which is only called when there are overflow reinjections
(1ms path) or periodically from the timer (100ms path). If entries sit in Q
but OverflowRing is empty, the 1ms path won't drain them — only the 100ms
timer will.

**⚠️ INVESTIGATE candidate:** The Q drain should ideally be triggered by
quarantine enqueue (wake the timer/Coordinator to drain sooner), but currently
relies on the 100ms polling. This is not a lost-wake (entries are eventually
drained), but it is a **latency** issue.

### 8.2 EmergencyQ drain

Same mechanism as Q — drained via `tryReclaim()` →
`drainEmergencyAndTerminal()`. Same 100ms worst-case latency.

### 8.3 TerminalReclaimAuthority drain

`drainTerminalReclaim()` is called from `tryReclaim()` and shutdown paths.
Same wake mechanism (or lack thereof).

### 8.4 Shutdown drain

`drainAllQuarantineStore()` is called from:

1. `AudioEngine.CtorDtor.cpp:257` — abnormal shutdown destructor path
2. `AudioEngine.Processing.ReleaseResources.cpp:378` — normal `releaseResources()`
3. `AudioEngine.Processing.ReleaseResources.cpp:473` — post-clear final drain
4. `ISRRetireRouter.cpp:514` — `drainAll()` (delegated to `drainAllQuarantineStore`)
5. `releaseResources` graceful drain loop: `tryReclaim()` at lines 238, 268

**Shutdown drain does NOT interact with normal wake path** — it uses
`drainAllUnsafe()` (forced, epoch-agnostic) which does not touch any CV/Event.
It runs after all threads are joined. ✅

---

## 9. Shutdown Interaction

### 9.1 `drainAllQuarantineStore()` shutdown path

Called from `releaseResources()` at three points (lines 378, 473) and
`~AudioEngine()` destructor (line 257). Uses `drainAllUnsafe()` which:

- Acquires mutex per store (Q, E, T)
- Extracts all entries under lock
- Releases lock
- Calls deleter for each entry (no-lock scope)

**No CV/Event wake involved.** Shutdown drain is purely synchronous forced
drain. ✅

### 9.2 Interaction with normal wake path

The normal wake path (`rebuildCV.notify_all`) is for RebuildThread, not for
quarantine drain. The CoordinatorLoop (1ms) and Timer (100ms) are the only
drivers of quarantine drain in normal operation. Shutdown drain bypasses
both entirely.

**No interference.** ✅

### 9.3 `isShutdownInProgress()` gate

`drainDeferredRetireQueues(false)` checks `isShutdownInProgress()` and
returns early. `drainDeferredRetireQueues(true)` is used in shutdown.
No wake signal can fire during `ShutdownComplete` (terminal phase). ✅

---

## 10. Authority Impact Assessment

### 10.1 D→Q→E→T chain integrity

Any wake optimization must not alter the ownership transfer chain:
`enqueueRetire → D → Q → E → TerminalReclaimAuthority → deleter`.

| Optimization target | Impacts authority chain? |
| --- | --- |
| Suppress timer drain when Q+E+T are empty | No — no entries to drain |
| Wake Coordinator/timer on Q enqueue | No — just accelerates existing drain path |
| Coalesce `rebuildCV.notify_all()` | No — separate from quarantine chain |
| Event-driven Q drain (new CV) | **Yes** — adds new synchronization primitive |

### 10.2 `drainAllNonRt()` GAP-CROSS-1 fix (15-P-6)

The GAP-CROSS-1 fix at `ReleaseResources.cpp:542`
(`drainAllNonRt()` → `enqueueDeferredDeleteNonRt` → `shutdownReclaim` →
`TerminalReclaimAuthority`) must NOT be affected. This path runs
post-shutdown and uses `drainAllUnsafe()` (no wake). ✅

### 10.3 `isFullyDrained()` implications

`isFullyDrained()` directly checks `terminalReclaimResident` (Threading.cpp:155).
If we add a wake-on-quarantine-enqueue optimization, `isFullyDrained()` semantics
do NOT change — it still checks all resident counts. The wake would only
accelerate the drain that makes these counts reach zero. ✅

---

## 11. Optimization Opportunities

### Tier 1 — Low-risk, clearly beneficial

| # | Opportunity | Description | Risk |
| --- | --- | --- | --- |
| O-1 | **Suppress timer drain when empty** | In `timerCallback()`, skip `processDeferredReleases()` when `quarantineResidentCount() == 0 && emergencyQuarantineResidentCount() == 0 && terminalReclaimResidentCount() == 0 && pendingRetireCount() == 0`. This eliminates the 100ms periodic drain overhead when there's nothing to drain. Already partially implemented via pressure monitoring telemetry. | Low — zero entries means no-op drain |
| O-2 | **Event-driven Q drain** | Add a wake signal from `quarantine()` (Q) to the CoordinatorLoop or a Non-RT drain thread, so Q entries are drained as soon as epoch becomes safe instead of waiting up to 100ms. Must use a Non-RT-only CV (not on RT path). | Medium — adds CV primitive; must verify no RT blocking |

### Tier 2 — Medium risk, requires careful design

| # | Opportunity | Description | Risk |
| --- | --- | --- | --- |
| O-3 | **Coalesce rebuildCV notify** | The RebuildThread predicate already handles spurious wakes efficiently. `notify_one` vs `notify_all` could reduce over-signaling when multiple producers signal. | Low — minimal gain |
| O-4 | **Adaptive timer period** | Increase timer period (e.g., 200ms) when system is stable, decrease (e.g., 10ms) under pressure. Currently fixed at 100ms. | Medium — latency vs. CPU tradeoff |

### Tier 3 — High risk, requires redesign

| # | Opportunity | Description | Risk |
| --- | --- | --- | --- |
| O-5 | **Direct Q wake to Non-RT drain** | Add a dedicated Non-RT `QuarantineDrainThread` that waits on a CV signaled by `quarantine()`. This thread would call `tryReclaim()` immediately on enqueue, eliminating the 100ms worst-case latency. | High — new thread, new CV, new synchronization surface |

---

## 12. Wake Optimization Verdict

### Verdict: INVESTIGATE

### 12.1 GO — Safe to implement

- **O-1 (Suppress timer drain when empty):** The timer-driven
  `processDeferredReleases()` call at `AudioEngine.Timer.cpp:881` can be
  guarded by a fast check of `quarantineResidentCount()`,
  `emergencyQuarantineResidentCount()`, and
  `terminalReclaimResidentCount()` all being zero. This is a pure
  performance optimization with zero risk to ownership invariants.

  The telemetry infrastructure (`quarantineAbsorptionCount`, `backpressure`)
  already tracks these metrics.

- **C1 (already optimized):** The `submitRecoveryIntent` silent absorb
  (AudioEngine.h:4414) is already implemented — recovery requests that
  would be spurious wakes are absorbed without `notify_all()`.

### 12.2 INVESTIGATE — Requires design verification

- **O-2 (Event-driven Q drain):** Adding a wake signal from Q enqueue to
  a Non-RT drain path could reduce worst-case drain latency from 100ms to
  near-immediate. However:
  - The wake must use a **Non-RT-only CV** (not on the Audio Thread path).
  - The `RetireQuarantineStore::quarantine()` currently uses a mutex —
    adding a CV wake in the same lock scope is acceptable (Non-RT consumer).
  - Lost-wake risk: if the consumer polls before the enqueue completes the
    wake, entries remain until next poll — bounded, not lost.
  - **Requires:** verifying that the RT thread never calls `quarantine()`
    on the CV's waiters, and that shutdown drain (`drainAllUnsafe`) is
    unaffected.

### 12.3 NO-GO — Not recommended

- **O-5 (Dedicated drain thread):** Adding a new thread introduces context
  switch overhead and a new synchronization surface. The existing 1ms
  CoordinatorLoop already provides low-latency drain for the
  OverflowRing path. A dedicated thread would need independent scheduling
  guarantees that conflict with the current single-writer (Non-RT) design.

### 12.4 Shutdown safety

All proposed optimizations preserve the shutdown ordering invariant
(15-P-6 §4.3):

- `drainAllQuarantineStore()` (shutdown) uses `drainAllUnsafe()` (no CV/Event).
- Wake signals only affect Non-RT consumer threads (CoordinatorLoop, Timer,
  RebuildThread), none of which are running during shutdown.
- `drainAllNonRt()` (GAP-CROSS-1 fix) is post-wake-drain and unaffected.

---

## 13. Call-Site Graph

```text
Enqueue side (RT Thread):
  enqueueRetire() / enqueueDeferredDeleteNonRt()
    → ISRRetireRouter::enqueueWithRetry()  [ISRRetireRouter.cpp:277]
      → Stage 1: DeferredDeletionQueue::enqueue()  [lock-free]
      → Stage 2: tryReclaim() → drainEmergencyAndTerminal()  [epoch-gated drain]
      → Stage 3: RetireQuarantineStore::quarantine()  [mutex, Q]
      → Stage 4: EmergencyQuarantineStore::quarantine()  [mutex, E]
      → Stage 5: TerminalReclaimAuthority::store()  [mutex, T, growable]
      ⚠️ No wake signal issued at any stage

Drain side (Non-RT):
  CoordinatorLoop (1ms)
    → runCoordinatorPhase()
      → drainOverflowRing()
      → if reinjected: tryReclaim()
        → provider_->tryReclaim()  [D drain]
        → drainQuarantineStore()  [Q drain]
        → drainEmergencyAndTerminal()  [E + T drain]
    → [NO wake issued to self or others]

  Timer (100ms)
    → timerCallback()
      → processDeferredReleases()
        → drainDeferredRetireQueues(false)
          → tryReclaimResources()
            → tryReclaim()  [same as above]
      → [NO wake issued]

  Shutdown (ReleaseResources.cpp / CtorDtor.cpp)
    → drainAllQuarantineStore()
      → RetireQuarantineStore::drainAllUnsafe()  [Q forced]
      → EmergencyQuarantineStore::drainAllUnsafe()  [E forced]
      → TerminalReclaimAuthority::drainAll()  [T forced]
    → [NO wake — synchronous, all threads joined]
```

---

## 14. Recommendations for Next Turn

1. **Implement O-1 (suppress timer drain when empty)** — lowest risk,
   eliminates unnecessary 100ms drain overhead. Requires adding a
   `quarantineDrainNeeded()` fast-path check before
   `processDeferredReleases()` in `timerCallback()`.

2. **Prototype O-2 (event-driven Q drain)** — add a Non-RT-only
   `std::condition_variable` in `ISRRetireRouter` that
   `quarantine()` signals and a Non-RT drain path (CoordinatorLoop or
   a dedicated wake) waits on. Verify RT thread never touches this CV.

3. **Measure current wake frequency** — add a telemetry counter for
   `quarantine()` calls that result in Q/E/T storage (vs. immediate
   reclaim in D), to quantify how often the 100ms timer actually
   drains real quarantine entries.

4. **Verify `quarantineAbsorptionCount` effectiveness** — the existing
   Phase E absorb mechanism in `submitRecoveryIntent` should be audited
   for coverage: does it cover all spurious-recovery cases?
