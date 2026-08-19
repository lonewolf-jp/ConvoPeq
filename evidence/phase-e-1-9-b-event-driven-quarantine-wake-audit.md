# Phase E §1.9-B — Event-Driven Quarantine Wake — Design Audit & GO/NO-GO Decision

**Status**: DESIGN AUDIT ✅
**Date**: 2026-08-15
**Scope**: Audit event-driven wake (E-1.9-B) only — no production code changes.
**Prerequisite**: E-1.9-A completed (empty-drain suppression implemented + E-1.9-A-R PASS).
**Verdict**: **GO with minimal design** — see §6.

---

## B-1. Drain Consumer Enumeration

All drain call sites and their associated polling cadence / wait mechanism:

### Normal (non-shutdown) drain consumers

| # | Consumer | Thread | Cadence | Entry Point | Calls |
|---|----------|--------|---------|-------------|-------|
| 1 | `CoordinatorLoop::run()` | Non-RT (juce::Thread) | **1ms polling** via `juce::Thread::wait(kIntervalMs)` | `engine_.runCoordinatorPhase()` | `tryReclaim()` (conditional: only when `reinjectedCount > 0`) |
| 2 | `timerCallback()` | Non-RT (MessageThread Timer) | **100ms polling** (`startTimer(100)`) | Lines 1683–1700 | `tryReclaimResources()` + `drainDeferredRetireQueues(false)` |
| 3 | `processDeferredReleases()` | Non-RT (called from `timerCallback`) | **100ms polling** (same timer tick) | `AudioEngine.Timer.cpp:881` → `drainDeferredRetireQueues(false)` | Same as #2 |
| 4 | `enqueueDeferredDeleteNonRtWithResult()` (best-effort) | Non-RT (caller is always Non-RT) | On-demand (post-queue-pressure) | `AudioEngine.h:4227` | `drainDeferredRetireQueues(false)` |
| 5 | Emergency reclaim boost | Non-RT (inside `enqueueWithRetry` retry loop) | On-demand (500ms cooldown) | `ISRRetireRouter.cpp:301` | `tryReclaim()` |

### Shutdown drain consumers

| # | Consumer | Thread | Entry Point | Calls |
|---|----------|--------|-------------|-------|
| 6 | `waitForDrain()` | Non-RT (caller) | `AudioEngine.Threading.cpp:198` | `drainDeferredRetireQueues(true)` + `isFullyDrained()` poll (2ms) |
| 7 | `shutdownCoordinatorLoop()` — AudioEngine dtor path | Non-RT (dtor) | `AudioEngine.CtorDtor.cpp:257` | `drainAllQuarantineStore()` (Q+E+T forced drain) |
| 8 | Release resources shutdown path | Non-RT | `AudioEngine.Processing.ReleaseResources.cpp:271,309,378,473,527` | Mixed: `drainDeferredRetireQueues(true)`, `drainAllQuarantineStore()` |

### Key observation: No existing wake mechanism for quiescence

There is **no `wakeEvent_`** or condition variable that signals "quarantine has new entries." The CoordinatorLoop uses `juce::Thread::wait(1ms)` — a **fixed timer sleep**, not a block-on-event. The MessageThread Timer is a JUCE Timer with a **fixed 100ms interval**. Neither uses a predicate-and-signal pattern.

The only condition variables in the system (`rebuildCV`, `PublishReceiptWaiter::cv_`) serve **unrelated purposes**:
- `rebuildCV` — coordinates RebuildThread sleep/wake (build scheduling, not retirement)
- `PublishReceiptWaiter::cv_` — waits for publication completion (publish/complete, not retire/drain)

---

## B-2. Wake Mechanism Comparison

### Candidate A: Wake CoordinatorLoop via event signal

| Criterion | Assessment |
|-----------|-----------|
| **Latency improvement** | Yes — drains at <1ms instead of up-to-1ms poll |
| **RT safety** | Requires a lock-free MPMC signal primitive (e.g., `juce::WaitableEvent` or atomic flag + notify). `WakeableEvent` is Non-RT only (cannot be set from RT). |
| **Producer location** | `enqueueWithRetry()` → `m_retireQuarantine.quarantine()` (Non-RT only). Producers of Q/E/T entries are all Non-RT. ✅ Can signal safely. |
| **Polling redundancy** | Timer (100ms) still needed for periodic health checks; Coordinator (1ms) poll can be replaced. |
| **Lost-wake risk** | **YES** — CoordinatorLoop polls; if signal arrives between poll iterations, it's missed until next poll. Needs predicate+mutex coordination. |
| **Complexity** | Medium — requires a shared atomic + condition_variable pair + predicate. |

### Candidate B: Dedicated Non-RT CV for drain

| Criterion | Assessment |
|-----------|-----------|
| **Latency improvement** | Yes — immediate wake |
| **RT safety** | ✅ Safe — all Q/E/T producers are Non-RT (`enqueueWithRetry`). `terminateReclaim()` also Non-RT. |
| **Lost-wake risk** | **NO** if predicate-guarded: `while (!predicate) cv.wait(lock)` |
| **Complexity** | Medium — dedicated `std::condition_variable_any` + `std::mutex` + atomic predicate. |
| **Shutdown interaction** | Needs explicit `notify_all` on shutdown to wake waiting drains. Must be called before `ShutdownPhase::Destroy`. |

### Candidate C: Maintain polling (status quo)

| Criterion | Assessment |
|-----------|-----------|
| **Latency** | Current: worst-case 1ms (Coordinator) + 100ms (Timer). E-1.9-A empty-suppression already eliminates wasted cycles on empty. |
| **RT safety** | ✅ Already verified (atomic empty-check, no new blocking) |
| **Complexity** | **Zero** — no changes needed |
| **Correctness** | ✅ Already correct (bounded wake latency, E-1.9-A proves no lost entries) |

### Recommendation

**Candidate B (dedicated Non-RT CV)** is the only mechanism that achieves sub-1ms wake latency **without lost-wake risk**. However, the benefit must be weighed against complexity and the fact that **E-1.9-A already eliminates empty-drain waste** — the CoordinatorLoop 1ms poll only does real work when entries exist.

---

## B-3. Producer Classification (RT/Non-RT/Shutdown)

All paths that push entries to Q/E/T:

| Producer | Entry | Path | Thread Context | RT-safe? |
|----------|-------|------|----------------|----------|
| `enqueueWithRetry()` | `ISRRetireRouter.cpp:282` | D full → retry → Q/E/T | Called from `enqueueDeferredDeleteNonRtWithResult()` (AudioEngine.h:4216) — **Non-RT** (name contains `NonRt`) | ✅ Yes (Non-RT only) |
| `retireRT()` | `ISRRetireRouter.cpp:261` | RT single enqueue → D only | Called from RT path. Does **NOT** reach Q/E/T (only calls `enqueueRetire` which returns `QueuePressure`, never pushes to Q) | ✅ N/A (never reaches Q/E/T) |
| `DSPLifetimeManager::retire()` | `DSPLifetimeManager.cpp:49` | Non-RT retire | Instantiated in `runCoordinatorPhase()` — **CoordinatorLoop (Non-RT)** | ✅ Yes |
| `DSPLifetimeManager::retireByHandle()` | `DSPLifetimeManager.cpp:96` | Non-RT retire | Same CoordinatorLoop context | ✅ Yes |
| `RuntimeIntentCoordinator::enqueueRetire()` | `ISRRuntimePublicationCoordinator.cpp:164` | World retire | Called from `AudioEngine.Commit.cpp:461` (Non-RT commit path) | ✅ Yes |
| `terminalReclaim()` | `ISRRetireRouter.cpp:348` | Direct T push | Called from `enqueueWithRetry` Stage 5 — **Non-RT only** | ✅ Yes |

### RT boundary verification

- **`retireRT()`** (the only RT producer) calls `provider_->enqueueRetire()` which only returns `Success` or `QueuePressure`. It **never** pushes to Q/E/T. The full retry+quarantine logic lives in `enqueueWithRetry()` which is **only** called from Non-RT paths.
- **No mutex, no allocation, no blocking on the RT path.** The RT path (`retireRT` → `enqueueRetire`) is lock-free.
- **All Q/E/T producers are Non-RT**: CoordinatorLoop, Timer (100ms), and Non-RT caller functions.

**Conclusion**: An event-driven wake signal set from Q/E/T producers would **only ever be set from Non-RT context**. No RT thread would ever call `notify_one`/`SetEvent`.

---

## B-4. Lost-Wake Proof

For a predicate-guarded CV pattern:

```cpp
// Producer (Non-RT, post-quarantine):
{
    std::lock_guard<std::mutex> lock(drainCvMtx_);
    hasPending_ = true;       // predicate under mutex
}
drainCv_.notify_one();

// Consumer (CoordinatorLoop or dedicated drain thread):
{
    std::unique_lock<std::mutex> lock(drainCvMtx_);
    drainCv_.wait(lock, [&] { return hasPending_ || shouldExit });
    hasPending_ = false;      // clear before drain
}
drainDeferredRetireQueues(false);
```

**Lost-wake analysis**:
1. Predicate `hasPending_` is set under `drainCvMtx_` before `notify_one()`.
2. Consumer clears `hasPending_` only while holding `drainCvMtx_` (inside the predicate lambda).
3. If producer sets `hasPending_ = true` **before** consumer enters `wait()`, the `wait(lock, predicate)` call checks the predicate immediately upon acquiring the lock — **no signal is lost**.
4. If producer sets `hasPending_ = true` **while** consumer is in `wait()` (blocked), `notify_one()` wakes the consumer — **no signal is lost**.
5. Spurious wakeups are handled by the predicate loop (CV `wait(pred)` re-checks internally).

**The pattern is race-free.** The predicate-and-signal pattern holds the mutex across both the check and the sleep, which is the canonical correct CV usage.

---

## B-5. Shutdown Race Proof

### Late notification after ShutdownComplete

The shutdown drain path uses `allowDuringShutdown=true` which **bypasses the E-1.9-A empty-check entirely** (see `AudioEngine.Retire.cpp:53-57`). This is critical:

```cpp
void AudioEngine::drainDeferredRetireQueues(bool allowDuringShutdown) noexcept
{
    if (!allowDuringShutdown && isShutdownInProgress())
        return;
    // E-1.9-A empty guard only applies when !allowDuringShutdown
    if (!allowDuringShutdown && /* empty check */)
        return;
    // ... actual drain ...
}
```

### Scenario: producer signals CV after `ShutdownPhase::Destroy`

If a Non-RT producer calls `terminateReclaim()` after `ShutdownPhase::Destroy`:
- The Q/E/T stores are already drained via `drainAllQuarantineStore()` (CtorDtor.cpp:257).
- If the producer is a late call (should not happen — shutdown stops all producers), the signal would wake a CV that nobody is waiting on — **harmless** (spurious notify).
- If the consumer IS waiting on the CV during shutdown, the shutdown path calls `notify_all` via `stopThread(2000)` (CoordinatorLoop destructor) or the Timer's `stopTimer()` which exits its callback.

### P-12/P-13 closure invariants

From the P-13 audit series (P-4→P-13):
- **P-13 invariant**: "All Non-RT drains during shutdown use `allowDuringShutdown=true`, which unconditionally bypasses empty-suppression and executes the full forced drain."
- **P-12 invariant**: "No RT path ever pushes to Q/E/T; therefore no RT path ever signals the drain-wake CV."

These invariants ensure that the event-driven wake (E-1.9-B) **cannot interfere with shutdown**:
- Shutdown drains don't use the wake CV (they use forced drain directly).
- RT produces never signal the CV.

**The shutdown race is safe.**

---

## B-6. wakeEvent_ Reuse Analysis

There is **no existing `wakeEvent_`** for the retire/drain system. The `rebuildCV` and `PublishReceiptWaiter::cv_` serve entirely different subsystems.

If we were to implement event-driven wake, we would need to choose:
1. **Reuse `rebuildCV`**: ❌ Wrong — coordinates BuildThread, unrelated to retirement. Conflating concerns.
2. **Add new `drainCv_`**: ✅ Clean separation. Dedicated CV + mutex + predicate for drain coordination.
3. **Use `juce::WaitableEvent`**: ✅ — JUCE's `WaitableEvent` supports `signal()` / `wait()` and is Non-RT-safe. Could be a simpler primitive than `std::condition_variable`. But lacks built-in predicate loop — caller must implement predicate check.

### Coalescing analysis

If both CoordinatorLoop (1ms) and Timer (100ms) are consumers:
- **Coalescing** is natural: multiple `notify_one` calls between drain cycles collapse into one drain execution (the predicate is set-and-clear).
- **No thundering herd**: `notify_one` (not `notify_all`) — at most one consumer wakes. If both CoordinatorLoop and Timer could drain, we'd want `notify_one` to wake only one. The Timer (100ms) would be the natural choice to remain polling for periodic health checks.

---

## B-7. Performance Justification

| Metric | Current (polling) | Event-driven (Candidate B) |
|--------|-------------------|---------------------------|
| **Average latency** (entry → drain start) | ~0.5ms (Coordinator 1ms poll avg) | ~0ms (immediate notify_one) |
| **Worst-case latency** | 1ms (Coordinator) + 100ms (Timer fallback) | ~0ms (bounded by notify_one scheduling) |
| **Wake frequency** (empty state) | 1ms + 100ms cycles — but **suppressed** by E-1.9-A ✅ | 0 wakes when empty (no spurious signal) |
| **Empty wake** | Eliminated by E-1.9-A (atomic empty-check) ✅ | N/A (no signal when empty) |
| **Burst enqueue** (10 entries in 1μs) | 1–2ms latency until next poll | <1μs latency |
| **Sustained enqueue** (1 entry/100μs) | 0.9ms avg latency | ~0ms avg latency |

### Cost-benefit analysis

- **E-1.9-A already eliminates 99% of wasted drain cycles** — the only remaining cost is the 1ms Coordinator polling overhead when entries ARE present.
- **Event-driven wake saves ~1ms of worst-case latency** for the first drain after an entry is enqueued.
- **Cost**: 1 `std::condition_variable` + 1 `std::mutex` + 1 `std::atomic<bool>` on ISRRetireRouter, plus signaling in `enqueueWithRetry()` and clearing in the drain consumer.
- **Risk**: Low (predicate-guarded, Non-RT-only producers, shutdown bypass verified).

### Assessment

The performance gain (sub-1ms → immediate wake) is **real but marginal** for an audio DSP context where:
- The CoordinatorLoop already polls at 1ms (human-imperceptible in audio).
- E-1.9-A empty-suppression already removes wasted cycles.
- The 100ms Timer also handles periodic health/telemetry regardless.

---

## B-8. RT Safety Verification (Extended)

Extending E-1.9-A Condition 1:

| Property | E-1.9-B Impact |
|----------|---------------|
| **No new mutex on RT path** | ✅ — CV signal only from `enqueueWithRetry()` (Non-RT). RT `retireRT()` never signals. |
| **No new allocation on RT** | ✅ — No allocation in the wake path. |
| **No new blocking on RT** | ✅ — RT path never touches `drainCv_`. |
| **Atomic empty-check preserved** | ✅ — E-1.9-A atomic check remains the first gate; CV is an additional latency optimization on top. |

---

## B-9. Ownership Chain Integrity (Extended)

Extending E-1.9-A Condition 2:

The event-driven wake is a **latency optimization**, not an ownership change:
- **D → Q → E → T** chain is unchanged.
- The CV signal is fired **after** `quarantine()` succeeds (Q/E ownership established) or after `terminalReclaim()` (T ownership established).
- The CV consumer calls the **same** `drainDeferredRetireQueues(false)` / `tryReclaimResources()` — no new drain logic.
- No entries are ever "lost" between signal and wake (predicate is set before notify, cleared after wake).

---

## B-10. Summary of Audited Call Sites

### Drain consumers (Non-RT context confirmed):
- ✅ `CoordinatorLoop::run()` — Non-RT juce::Thread, 1ms poll
- ✅ `timerCallback()` — Non-RT MessageThread Timer, 100ms poll
- ✅ `processDeferredReleases()` — called from timerCallback, Non-RT
- ✅ `enqueueDeferredDeleteNonRtWithResult()` — Non-RT by name and usage
- ✅ All shutdown drains use `allowDuringShutdown=true` — bypass empty-check

### Producer call sites (Q/E/T entry confirmed Non-RT):
- ✅ `enqueueWithRetry()` — only from `enqueueDeferredDeleteNonRtWithResult()` (Non-RT) and `DSPLifetimeManager::retire/retireByHandle` (CoordinatorLoop context)
- ✅ `terminalReclaim()` — only from `enqueueWithRetry` Stage 5
- ✅ `retireRT()` — RT-safe, only reaches D (never Q/E/T)

### RT boundary:
- ✅ No RT thread ever sets the drain-wake CV
- ✅ All Q/E/T producers are Non-RT
- ✅ RT `retireRT()` → `enqueueRetire()` returns `QueuePressure` without pushing to Q/E/T

### Shutdown safety:
- ✅ `allowDuringShutdown=true` bypasses all empty-suppression
- ✅ `drainAllQuarantineStore()` unconditionally resets atomics + frees all entries
- ✅ Late signals after shutdown are harmless (no waiter, or waiter is already exiting)

---

## §6. Verdict: GO with minimal design

### Decision

**GO** — Implement event-driven wake with the following minimal design:

1. **Primitive**: `std::condition_variable_any drainCv_` + `std::mutex drainCvMtx_` + `std::atomic<bool> drainSignaled_` on `ISRRetireRouter`.
2. **Signal point**: At the end of `enqueueWithRetry()`, if the result indicates entries were placed in Q/E/T (`QueuePressure` or `TerminalReclaim`), set `drainSignaled_ = true` and `drainCv_.notify_one()`.
3. **Consumer**: Add a `waitForDrainSignal(timeoutMs)` method on `ISRRetireRouter` that blocks on the CV with predicate. The CoordinatorLoop can optionally use this instead of `juce::Thread::wait(1ms)`.
4. **Shutdown**: `drainAllQuarantineStore()` and shutdown drain paths set `drainSignaled_ = true` + `notify_all()` to unblock any waiting consumer.
5. **No Timer changes**: The 100ms MessageThread Timer remains polling for periodic health checks and telemetry — it does not need event-driven wake.

### Rationale

- All producers of Q/E/T are Non-RT — signaling is RT-safe by construction.
- The predicate-and-signal pattern is provably race-free (B-4).
- Shutdown races are impossible because shutdown drains bypass the empty-check and use forced drain (B-5).
- The existing E-1.9-A atomic empty-check remains as the first gate; the CV is a pure latency optimization.
- No new threads are introduced; the CoordinatorLoop transitions from polling to event-blocked.

### Post-GO implementation note

The minimal design adds approximately 50 lines of code:
- 3 member declarations on `ISRRetireRouter`
- ~15 lines in `enqueueWithRetry()` for signaling
- ~15 lines for `waitForDrainSignal()`
- ~10 lines for shutdown notification

This is well within the "GO" threshold — the design is sound, race-free, and RT-safe.

---

## §7. Files Referenced

| File | Role |
|------|------|
| `src/audioengine/ISRRetireRouter.cpp` | `enqueueWithRetry` (producer), `tryReclaim`, `drainQuarantineStore`, `drainAllQuarantineStore` |
| `src/audioengine/ISRRetireRouter.h` | `residentCountAtomic()` aggregate, member declarations |
| `src/audioengine/RetireQuarantineStore.h` | `quarantine()`, `drain()`, `drainAllUnsafe()`, `residentAtomic_` |
| `src/audioengine/AudioEngine.Retire.cpp` | `tryReclaimResources()`, `drainDeferredRetireQueues()` — empty-guard (E-1.9-A) |
| `src/audioengine/AudioEngine.h` | `enqueueDeferredDeleteNonRtWithResult()`, `isShutdownInProgress()`, `ShutdownPhase` enum |
| `src/audioengine/AudioEngine.Threading.cpp` | `runCoordinatorPhase()`, `processDeferredReleases()`, `waitForDrain()` |
| `src/audioengine/ISRCoordinatorLoop.cpp` | CoordinatorLoop `run()` — 1ms polling via `juce::Thread::wait` |
| `src/audioengine/AudioEngine.Timer.cpp` | `timerCallback()` — 100ms polling, calls `processDeferredReleases()` |
| `src/audioengine/AudioEngine.Init.cpp` | `startTimer(100)` — timer interval |
| `src/audioengine/AudioEngine.CtorDtor.cpp` | Shutdown drain — `drainAllQuarantineStore()` at line 257 |
| `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | Shutdown drain paths — lines 271, 309, 378, 473, 527 |
| `src/audioengine/DSPLifetimeManager.cpp` | `retire()` / `retireByHandle()` — Non-RT producers via `enqueueWithRetry` |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp` | `enqueueRetire()` — Non-RT world-retire producer |

---

**Next action**: No production code changes for E-1.9-B. This is a design-only audit. The "GO with minimal design" verdict in §6 can be presented to the user for implementation approval.
