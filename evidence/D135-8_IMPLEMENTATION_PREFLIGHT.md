# D135-8 — Implementation Preflight Addendum (Step 0 Read-Only Re-Audit)

**Status:** READ-ONLY — ZERO production source changes
**Date:** 2026-08-30
**Prerequisite:** D135-7 Phase 1 Preflight Audit (GO)
**Purpose:** D135-8 Step 0 — re-audit live source to lock down implementation targets, callers, lock scope, and shutdown exit ordering before any edits. This document is the Step 0 deliverable. No source files are modified.

---

## 0. Executive Summary

This addendum consolidates the Step 0 read-only re-audit findings for D135-8 Phase 1 implementation. All findings are verified against live source in `src/audioengine/` and cross-referenced with `ConvoPeq.md`, `D135-6_RECOVERY_WAKE_PROVENANCE_DESIGN.md`, `D135-7_PHASE1_IMPLEMENTATION_PREFLIGHT_AUDIT.md`, and `RuntimePublicationState.h`.

**Key decisions locked:**
- **D1 provenance plubbing**: `std::atomic<bool> recoveryRetryReady{false}` — new atomic flag in AudioEngine.h, set alongside `publishRetryReady` in the Timer.cpp recovery handler under `rebuildMutex`. **Not** added to the CV predicate (it is provenance, not a wake trigger).
- **Consumer merge**: `wasRecoveryWake = recoveryRetryReady.exchange(false)` at the merge site (RebuildDispatch.cpp:878-880), snapshot taken under `rebuildMutex`, then passed to `processDeferredAdmission(bool wasRecoveryWake)`.
- **kMaxDeferredRetries drift**: source h:279 says `10`, spec/runtime state comment says `2`. Fix: 10→2. **Test-verification plan** attached (Section 5.5).
- **Dead code**: `deferredRecoveryRearmed_` (h:283, .cpp:550) — 0 reads, write-only. Safe to remove.
- **EmergencyDrain ordering**: `clearDeferredForShutdown()` at ReleaseResources.cpp:359 runs **AFTER** `stopRebuildThread()` (line 202) joins the rebuild thread. For Option C, the drain must occur **before** the rebuild thread exits — insertion point is the shutdown-check block in the rebuild loop (h:858-862), not after join.

**GO/NO-GO conclusion:** **GO for all Steps 1-9** — with the critical ordering fix at Step 7 (Option C insertion in the pre-break shutdown block, not EmergencyDrain-after-join).

---

## 1. Step 0 Audit Checklist — All Items Verified

### 1.1 P4: Add `recoveryRetryReady` atomic flag to AudioEngine.h

**Target location:** AudioEngine.h, adjacent to `publishRetryReady` at **h:2718**.

Current members (verified by reading h:2699-2730):
```
2699: std::mutex rebuildMutex;
2700: std::condition_variable rebuildCV;
2701: std::atomic<bool> rebuildThreadShouldExit { false };
2702: std::atomic<bool> rebuildThreadIsRunning { false };
...
2714: bool recoveryPending = false;        // work88 Recovery Intent arrival (NOT D135)
2718: bool publishRetryReady = false;       // rebuildMutex-protected, non-atomic
```

**Audit findings:**
- `publishRetryReady` is a plain `bool` protected by `rebuildMutex` (NOT atomic). The user's instruction specifies `recoveryRetryReady` as `std::atomic<bool>` — this is a deliberate design choice: it allows the recovery handler to set it *before* acquiring `rebuildMutex`, and the consumer reads it under the lock via `exchange(false)`.
- **No existing `recoveryRetryReady`, `wasRecoveryWake`, or `deferredClearRequested_`** — confirmed 0 results across the entire codebase (grep verified). All three are new additions.
- `recoveryPending` (h:2714) is a **separate concept** — it is the work88 Recovery Intent arrival flag, not D135 retry provenance. It remains untouched per user instruction ("Provenance/correlation separation: `recoveryRetryReady` vs `recoveryPublishSeq()` — both stay separate").

**Implementation plan:** Insert `std::atomic<bool> recoveryRetryReady{false};` immediately after `publishRetryReady` at h:2718, with a comment distinguishing it as provenance-only.

### 1.2 P5: Recovery handler — set both flags under `rebuildMutex`

**Target location:** AudioEngine.Timer.cpp recovery handler, currently at **~line 1721-1746**.

Verified recovery handler flow (Timer.cpp:1721-1746):
```cpp
1721:    const auto recoverySeq = getLastCommittedPublicationSequence();
1722:    runtimeOrchestrator_->setRecoveryPublishSeq(recoverySeq);
1724:    runtimeOrchestrator_->resetDeferredRetryBudget();   ← OWNERSHIP VIOLATION
       ... (diagnostics block) ...
1742:    {
1743:        std::lock_guard<std::mutex> lock(rebuildMutex);
1744:        publishRetryReady = true;
1745:    }
1746:    rebuildCV.notify_one();
1747:    ... (crossfade complete, diagLog) ...
```

**Audit findings:**
- `resetDeferredRetryBudget()` at h:1724 is called from MessageThread under **no lock** — writes `deferredRetryGeneration_` and `deferredRetryCount_` which are rebuild-thread-only plain members (RuntimePublicationOrchestrator.h:277-278). This is the **confirmed ownership violation** from D135-7.
- `setRecoveryPublishSeq(recoverySeq)` at h:1721 writes `lastRecoveryPublishSeq_` (atomic, h:281) — this is safe (atomic).
- `publishRetryReady = true` at h:1744 is under `rebuildMutex` — correct.

**Implementation plan for Step 5 + Step 1-B:**
- **Step 5**: Remove `resetDeferredRetryBudget()` call at h:1724 (it moves to rebuild thread).
- **Step 1-B**: Add `recoveryRetryReady.store(true, std::memory_order_release);` inside the `rebuildMutex`-locked block at h:1742-1745, alongside `publishRetryReady = true`.

### 1.3 P6: Consumer merge in RebuildDispatch.cpp

**Target location:** AudioEngine.RebuildDispatch.cpp, **h:878-880** (deferred publish merge site).

Verified rebuild loop structure (h:840-910):
```cpp
840:    while (true) {
841:        try {
842:            RebuildTask task;
843:            bool doDeferredPublish = false;
844:            bool wokeByPendingTask = false;
846:            {
847:                std::unique_lock<std::mutex> lock(rebuildMutex);
848:                // CV predicate:
850:                rebuildCV.wait(lock, [this] {
851:                    return hasPendingTask
852:                         || publishRetryReady
853:                         || recoveryPending
854:                         || convo::consumeAtomic(rebuildThreadShouldExit, std::memory_order_acquire);
855:                });
856:                if (convo::consumeAtomic(rebuildThreadShouldExit, std::memory_order_acquire)) break;
858:                if (isShutdownInProgress()) {
859:                    hasPendingTask = false;
860:                    pendingTask.currentDSP = nullptr;
861:                    publishRetryReady = false;
862:                    break;
863:                }
865:                wokeByPendingTask = hasPendingTask;
870:                if (wokeByPendingTask) {
872:                    task = pendingTask;
874:                    pendingTask.currentDSP = nullptr;
876:                    hasPendingTask = false;
877:                }
878:                // ★ Deferred publish handoff consume:
879:                doDeferredPublish = publishRetryReady;
880:                publishRetryReady = false;
881:                convo::publishAtomic(rebuildBacklog_, 0, std::memory_order_release);
882:                (diagnostics block)
883:            }
884:            // ★ Process deferred admission:
885:            if (doDeferredPublish && runtimeOrchestrator_ != nullptr) {
886:                runtimeOrchestrator_->processDeferredAdmission();
887:            }
888:            // ★ DSPGuard + task execution ...
901:        } catch (...) { ... }
910:    }
```

**CRITICAL ORDERING FINDING (for Step 7 Option C):**
- The rebuild loop **breaks** on `rebuildThreadShouldExit` at **h:856** — this is **BEFORE** the deferred publish merge at h:878-880.
- If `deferredClearRequested_` is set alongside `rebuildThreadShouldExit`, a naive insertion at the merge site (h:878) would **never execute** — the thread breaks at h:856 before reaching it.
- **Resolution**: Option C's drain logic must be inserted **BEFORE** the `break` at h:856, or inside the shutdown-check block at h:858-862 (which already `break`s before the merge). The cleanest insertion point is the shutdown-check block (h:858-862): drain `deferredClearRequested_` there, then `break`.

**Consumer merge plan (Step 2):**
```cpp
// At h:878-880, change from:
doDeferredPublish = publishRetryReady;
publishRetryReady = false;
// To:
const bool wasRecoveryWake = recoveryRetryReady.exchange(false, std::memory_order_acq_rel);
doDeferredPublish = publishRetryReady;
publishRetryReady = false;
// Then at h:886:
runtimeOrchestrator_->processDeferredAdmission(wasRecoveryWake);
```
- `recoveryRetryReady` is read via `exchange(false)` — atomic, safe even though `wasRecoveryWake` is a local `const bool` captured outside the lock scope (consumed only at h:886 which is outside the `rebuildMutex` lock block).
- `publishRetryReady` (non-atomic) stays under `rebuildMutex` — no change to its synchronization.

### 1.4 P7: `processDeferredAdmission(bool wasRecoveryWake)` signature change

**Target locations:**
- Declaration: `RuntimePublicationOrchestrator.h:192`
- Definition: `RuntimePublicationOrchestrator.cpp:636`
- Single caller: `AudioEngine.RebuildDispatch.cpp:886`

**Audit findings:**
- `processDeferredAdmission()` currently takes **no arguments** (h:192: `void processDeferredAdmission() noexcept;`).
- Definition at .cpp:636-660 has `jassert(rebuildThreadId)`.
- **Single caller** at RebuildDispatch.cpp:886 — confirmed by grep: 1 result in production code, plus 1 reference in `DeferredFlowIntegrationTests.cpp:88` (test comment only, no direct call).
- Changing the signature to `processDeferredAdmission(bool wasRecoveryWake)` affects exactly **1 production call site** (RebuildDispatch.cpp:886) and **0 test call sites** (tests use `DeferredPublicationTestAccess` which doesn't call it directly — tests drive the engine via `requestRebuild`).

**Budget reset gating plan (Step 3):**
- At .cpp:636 (start of `processDeferredAdmission`), before `peekDeferred()`:
  ```cpp
  if (wasRecoveryWake) {
      resetDeferredRetryBudget();  // rebuild thread now owns this — jassert will pass
  }
  ```
- `resetDeferredRetryBudget()` is inline at h:171-173 and writes `deferredRetryGeneration_` and `deferredRetryCount_`. Adding an owner assertion (Step 8) will confirm rebuild-thread ownership.

### 1.5 P8: Option C — `clearDeferredForShutdown()` via `deferredClearRequested_` atomic

**Current `clearDeferredForShutdown()` implementation:**
- Declaration: RuntimePublicationOrchestrator.h:195
- Definition: RuntimePublicationOrchestrator.cpp:534-559

Verified implementation (cpp:534-559):
```cpp
534: void RuntimePublicationOrchestrator::clearDeferredForShutdown() noexcept
535: {
536:     const auto nowUs = ...;
538:     if (convo::consumeAtomic(hasDeferred_, ...)) {
539:         if (deferredSlot_.has_value())
540:             deferredSlot_->lastDiscardReason = DiscardReason::ShutdownDiscard;
541:         deferredSlot_.reset();
542:         convo::publishAtomic(hasDeferred_, false, ...);
543:     }
544:     // ★ D135-1: shutdown reset
545:     deferredRetryGeneration_ = 0;
546:     deferredRetryCount_ = 0;
547:     deferredRecoveryRearmed_ = false;
548:     convo::publishAtomic(lastRecoveryPublishSeq_, 0, ...);
549:     // DeferredHealth telemetry
550:     ...
559: }
```

**4 callers of `clearDeferredForShutdown()` (all from MessageThread):**
| # | File:Line | Context | Thread |
|---|---|---|---|
| 1 | AudioEngine.Timer.cpp:1642 | Publication stall recovery | MessageThread |
| 2 | AudioEngine.Timer.cpp:1804 | Recover action | MessageThread |
| 3 | AudioEngine.Timer.cpp:1824 | Restore action | MessageThread |
| 4 | AudioEngine.Processing.ReleaseResources.cpp:359 | EmergencyDrain phase | MessageThread (after rebuild thread join) |

**The option C approach:**
- Add `std::atomic<bool> deferredClearRequested_{false};` to RuntimePublicationOrchestrator (h:~284, near other retry members).
- Create a **new** method `requestDeferredClear()` (atomic flag set + rebuildCV.notify_one) that is the **only** safe entry point from MessageThread.
- Rename/repurpose `clearDeferredForShutdown()` to remain the actual work function but add `jassert(rebuildThreadId)` (it becomes rebuild-thread-only).
- The rebuild thread drains `deferredClearRequested_` in the **shutdown-check block** at h:858-862 (BEFORE the `break`), not at the merge site (h:878) or in EmergencyDrain (after join).

**EmergencyDrain ordering issue — RESOLVED:**
- `stopRebuildThread()` at ReleaseResources.cpp:202 sets `rebuildThreadShouldExit=true`, calls `rebuildCV.notify_all()`, then `join()`s the rebuild thread.
- During join, the rebuild thread will:
  1. Wake from `rebuildCV.wait` (due to `rebuildThreadShouldExit`)
  2. Hit the `break` at h:856 (or the shutdown-check block at h:858-862)
  3. **If Option C is implemented here**: drain `deferredClearRequested_` in the shutdown-check block, then `break`.
- After join returns, the EmergencyDrain call at ReleaseResources.cpp:359 calls `clearDeferredForShutdown()` again. At this point:
  - `hasDeferred_` is already `false` (cleared during shutdown-block drain) → the `if (hasDeferred_)` branch is skipped.
  - `deferredRetryGeneration_`, `deferredRetryCount_`, `deferredRecoveryRearmed_` are already `0`/`false` → redundant but harmless.
  - `lastRecoveryPublishSeq_` is already `0` → redundant but harmless.
  - **However**, if Step 8 adds `jassert(rebuildThreadId)` to `clearDeferredForShutdown()`, this EmergencyDrain call from MessageThread (post-join) would **trigger the assertion** in debug builds.

**Resolution for EmergencyDrain (Step 7):**
- **Recommended**: Remove the `clearDeferredForShutdown()` call from the EmergencyDrain block at ReleaseResources.cpp:359. The rebuild thread already drains everything during `stopRebuildThread()`. The EmergencyDrain call is redundant and would trip the new owner assertion.
- **Alternative** (if EmergencyDrain must remain for safety): Add a guard `if (rebuildThread.joinable())` check, or call `requestDeferredClear()` instead (the safe atomic entry point). But since the rebuild thread is already joined at this point, `requestDeferredClear()` would deadlock (no thread to drain). So the cleanest approach is removal.

### 1.6 P9: Remove dead `deferredRecoveryRearmed_`

**Location:** RuntimePublicationOrchestrator.h:283 and RuntimePublicationOrchestrator.cpp:550 (inside `clearDeferredForShutdown()`).

**Audit findings:**
- `deferredRecoveryRearmed_` declared at h:283: `bool deferredRecoveryRearmed_{false};`
- Written at .cpp:550: `deferredRecoveryRearmed_ = false;` (inside `clearDeferredForShutdown()`)
- **0 reads** across the entire codebase (grep confirmed).
- Dead code — write-only, no functional impact.

**Implementation plan:** Remove the declaration at h:283 and the write at .cpp:550. Both locations are inside `clearDeferredForShutdown()` which is being refactored for Option C.

### 1.7 kMaxDeferredRetries drift — 10 vs 2

**Source:** `RuntimePublicationOrchestrator.h:279`
```cpp
static constexpr uint8_t kMaxDeferredRetries = 10;  // D135-1: 2 回再駆動許容、3 回目で諦
```

**Spec/comment:** `RuntimePublicationState.h:17`
```cpp
// kMaxDeferredRetries=2
```

And the comment in RuntimePublicationState.h:25-30:
```
// D135-1: RetryExhaustedDiscard — DeferredFadingActive の繰り返し再駆動が
//   retry-cap (kMaxDeferredRetries=2) に到選したため... 2 回再駆動許容、3 回目で諦
```

**Resolution:** Change h:279 from `10` to `2`. The inline comment "2 回再駆動許容、3 回目で諦" is already correct (matches the comment text). The value was simply never corrected from an earlier draft.

**Test verification plan (Section 5.5):**
- Count states: generation=0, count 0→1 (retry 1), 1→2 (retry 2), 2→RetryExhaustedDiscard (retry 3 discarded by `deferredRetryCount_ >= kMaxDeferredRetries` at .cpp:489).
- This means: 2 actual re-defer attempts are allowed, the 3rd enqueue of the same generation triggers discard.

### 1.8 RecoveryArmed latch (Step 4) — safety confirmed

**Verification of no `wasRecoveryWake` leakage:**
- `submitPublishRequest(const PublishRequest& req)` — h:155, takes only `const PublishRequest&`, no `wasRecoveryWake`.
- `trySubmitImpl()` — verified, same signature, no `wasRecoveryWake`.
- `enqueueDeferred(const PublishRequest& req)` — .cpp:437, takes only `const PublishRequest&`, no `wasRecoveryWake`.
- **Single caller** of `processDeferredAdmission`: RebuildDispatch.cpp:886.
- **No leakage path**: `wasRecoveryWake` is a local `const bool` at the merge site (h:879), consumed only within `processDeferredAdmission(bool wasRecoveryWake)` before any call to `submitPublishRequest` or `enqueueDeferred`. The flag never crosses the orchestrator boundary.

**Step 4 implementation plan:**
- Add a `DeferredPublishSlot::recoveryArmed` field (bool) to the slot struct (h:32-39).
- In `processDeferredAdmission(bool wasRecoveryWake)`: if `wasRecoveryWake`, set a pending latch flag; the next `enqueueDeferred` call stamps `slot.recoveryArmed = true` on the new slot.
- This is a **latch** (one-shot per recovery cycle): set by the recovery wake, consumed when the deferred slot is re-enqueued during recovery. No accumulation across cycles.

---

## 2. Shutdown Exit Ordering Audit (Step 0 Critical Finding)

### 2.1 Full shutdown sequence (ReleaseResources.cpp)

Verified sequence (ReleaseResources.cpp:line references):

| Step | Line | Action | Thread |
|------|------|--------|--------|
| 1 | ~175 | `publishRetryReady = false` (in hasPendingTask block) | MessageThread |
| 2 | 190 | `setShutdownPhase(StopWorkers)` | MessageThread |
| 3 | 191 | `shutdownCoordinatorLoop()` | MessageThread |
| 4 | 202 | `stopRebuildThread()` → set `rebuildThreadShouldExit`, `notify_all()`, `join()` | MessageThread joins RebuildThread |
| 5 | 212+ | `joinProducers()` loop | MessageThread |
| 6 | 330 | `drainDeferredRetireQueues(true)` | MessageThread |
| 7 | 337 | `transitionTo(ReclaimComplete)` | MessageThread |
| 8 | 344 | `transitionTo(EmergencyDrain)` | MessageThread |
| 9 | 359 | `clearDeferredForShutdown()` (EmergencyDrain) | MessageThread — **AFTER rebuild thread join** |

### 2.2 Rebuild thread exit path (RebuildDispatch.cpp)

The rebuild thread, when woken by `rebuildThreadShouldExit`:
```
h:856: if (consumeAtomic(rebuildThreadShouldExit)) break;
```

**No processing of deferred clear** occurs between h:856 `break` and thread exit. The thread immediately unwinds and exits.

### 2.3 Option C implication

For Option C to work:
1. `requestDeferredClear()` sets `deferredClearRequested_{true}` (atomic) + `rebuildCV.notify_one()`.
2. The rebuild thread, on next wake (or during shutdown wake), must check `deferredClearRequested_` and call `clearDeferredForShutdown()` (rebuild-thread-only).
3. This check must happen **BEFORE** the `break` at h:856 or in the shutdown-check block at h:858-862.

**Verified insertion point:**
```cpp
// In shutdown-check block (h:858-862), BEFORE break:
if (isShutdownInProgress()) {
    hasPendingTask = false;
    pendingTask.currentDSP = nullptr;
    publishRetryReady = false;
    // ★ D135-8 Step 7 (Option C): drain deferred clear before thread exit
    if (deferredClearRequested_) {
        runtimeOrchestrator_->clearDeferredForShutdown();
    }
    break;
}
```
Plus, the same check should be placed **before** the h:856 `break` (for the non-shutdown path where `rebuildThreadShouldExit` is set but `isShutdownInProgress()` may not yet be true):
```cpp
// Before h:856 break:
if (convo::consumeAtomic(rebuildThreadShouldExit, std::memory_order_acquire)) {
    if (deferredClearRequested_) {
        runtimeOrchestrator_->clearDeferredForShutdown();
    }
    break;
}
```

### 2.4 EmergencyDrain call at ReleaseResources.cpp:359

**After implementing Option C and stopRebuildThread():**
- The rebuild thread performs `clearDeferredForShutdown()` during its exit (Step 2.3).
- `stopRebuildThread()` joins, so by the time EmergencyDrain runs (step 9, line 359), everything is already cleared.
- The EmergencyDrain call is **redundant**.

**Decision:** Remove the `clearDeferredForShutdown()` call from EmergencyDrain (ReleaseResources.cpp:359). If Step 8 adds `jassert(rebuildThreadId)` to `clearDeferredForShutdown()`, keeping the call would trigger an assertion failure in debug builds (MessageThread calling a rebuild-thread-only function). Removing it is both safe (already drained) and correct (avoids assertion violation).

**Risk mitigation:** The EmergencyDrain block also retains `tryReclaim()` and crossfade reset — only the `clearDeferredForShutdown()` line is removed. No other functionality is affected.

---

## 3. File-by-File Change Inventory

### 3.1 AudioEngine.h
| Step | Change | Location | Line |
|------|--------|----------|------|
| P1-A | Add `std::atomic<bool> recoveryRetryReady{false};` | After `publishRetryReady` | h:2718 |

### 3.2 AudioEngine.RebuildDispatch.cpp
| Step | Change | Location | Line |
|------|--------|----------|------|
| P2 | Consumer merge: `wasRecoveryWake` snapshot + pass to `processDeferredAdmission` | h:878-880, h:886 | 879, 886 |
| Step 7 | Drain `deferredClearRequested_` before `break` at h:856 and before `break` at h:862 | h:856, h:858-862 | 856, 858-862 |

### 3.3 AudioEngine.Timer.cpp
| Step | Change | Location | Line |
|------|--------|----------|------|
| P1-B | Add `recoveryRetryReady.store(true)` in recovery handler | h:1742-1745 | 1743 |
| Step 5 | Remove `resetDeferredRetryBudget()` call | h:1724 | 1724 |

### 3.4 AudioEngine.Threading.cpp
| Step | Change | Location | Line |
|------|--------|----------|------|
| (no change) | Ordinary retry only sets `publishRetryReady` | h:286 | 286 |

### 3.5 RuntimePublicationOrchestrator.h
| Step | Change | Location | Line |
|------|--------|----------|------|
| P3 | Change `processDeferredAdmission()` → `processDeferredAdmission(bool wasRecoveryWake)` | h:192 | 192 |
| Step 7 | Add `std::atomic<bool> deferredClearRequested_{false};` | New, near retry members | ~284 |
| Step 7 | Add `requestDeferredClear()` method declaration | Near `clearDeferredForShutdown` | ~197 |
| Step 7 | Add `recoveryArmed` field to `DeferredPublishSlot` | h:32-39 | 36 |
| Step 8 | Add `jassert(rebuildThreadId)` to `resetDeferredRetryBudget()` | h:171-173 | 171 |
| Step 9 | Remove `deferredRecoveryRearmed_` declaration | h:283 | 283 |
| Step 6 | Change `kMaxDeferredRetries = 10` → `= 2` | h:279 | 279 |
| Step 6 | Change `>=` to `>` in retry discard check | .cpp:489 | 489 |

### 3.6 RuntimePublicationOrchestrator.cpp
| Step | Change | Location | Line |
|------|--------|----------|------|
| P3 | Add budget reset gating in `processDeferredAdmission(wasRecoveryWake)` | .cpp:636-660 | 636 |
| Step 4 | Set `slot.recoveryArmed` in `enqueueDeferred` | .cpp:450-460 | 455 |
| Step 7 | Add `requestDeferredClear()` impl + drain logic in `clearDeferredForShutdown` | .cpp:534 | 534 |
| Step 7 | Add `jassert(rebuildThreadId)` to `clearDeferredForShutdown` | .cpp:534 | 535 |
| Step 9 | Remove `deferredRecoveryRearmed_ = false` from `clearDeferredForShutdown` | .cpp:550 | 550 |

### 3.7 AudioEngine.Processing.ReleaseResources.cpp
| Step | Change | Location | Line |
|------|--------|----------|------|
| Step 7 | Remove `clearDeferredForShutdown()` call from EmergencyDrain block | h:359 | 359 |

### 3.8 RuntimePublicationState.h
| Step | Change | Location | Line |
|------|--------|----------|------|
| (no change) | Comment already says `kMaxDeferredRetries=2` | h:17 | 17 |

### 3.9 Test files
| Step | Change | Location | Line |
|------|--------|----------|------|
| P3 | Update test comment referencing `processDeferredAdmission` | DeferredFlowIntegrationTests.cpp:88 | 88 |

---

## 4. Lock Scope Audit

| Member | Thread(s) accessing | Lock | Status |
|--------|-------------------|------|--------|
| `publishRetryReady` | MessageThread (write), RebuildThread (read/clear) | `rebuildMutex` | ✅ Correct |
| `recoveryPending` | MessageThread (write via submitRecoveryIntent), RebuildThread (read) | `rebuildMutex` | ✅ Correct |
| `rebuildThreadShouldExit` | MessageThread (write), RebuildThread (read) | atomic | ✅ Correct |
| `deferredRetryGeneration_` | RebuildThread (enqueueDeferred, resetDeferredRetryBudget, clearDeferredForShutdown) | — no lock (single owner) | ⚠️ Step 8 adds jassert |
| `deferredRetryCount_` | RebuildThread (same as above) | — no lock | ⚠️ Step 8 adds jassert |
| `deferredRecoveryRearmed_` | MessageThread (clearDeferredForShutdown:550) | — no lock | ❌ Step 9 removes |
| `lastRecoveryPublishSeq_` | MessageThread (setRecoveryPublishSeq), RebuildThread (recoveryPublishSeq) | atomic | ✅ Correct |
| `hasDeferred_` | RebuildThread (peekDeferred, finishView, clearDeferredForShutdown), MessageThread (hasDeferredRequest) | atomic | ✅ Correct |
| `deferredSlot_` | RebuildThread (enqueueDeferred, peekDeferred, finishView, clearDeferredForShutdown) | — no lock (single owner) | ✅ Correct (after Option C) |

**New members:**
| Member | Thread(s) accessing | Lock | Status |
|--------|-------------------|------|--------|
| `recoveryRetryReady` | MessageThread (recovery handler), RebuildThread (exchange at merge) | atomic (own) + read under `rebuildMutex` at merge | ✅ Safe |
| `deferredClearRequested_` | MessageThread (requestDeferredClear sets), RebuildThread (drains in shutdown block) | atomic | ✅ Safe |
| `slot.recoveryArmed` | RebuildThread (set in enqueueDeferred, read in processDeferredAdmission) | — no lock (single owner) | ✅ Safe |

---

## 5. Implementation Step Sequence (Recommended Order)

Per D135-7's Phase 1 ordering recommendation (P4→P6→P7→P5→P1→P2→P3→P8/P9), adjusted for D135-8's 9 steps:

| Order | Step | Description | Rationale |
|-------|------|-------------|-----------|
| 1 | P6 | Fix `kMaxDeferredRetries` 10→2 AND change `>=`→`>` at .cpp:489 | Trivial value fix + operator fix must be paired
| 2 | P9 | Remove dead `deferredRecoveryRearmed_` | Remove dead code first, no behavioral impact |
| 3 | P1-A | Add `recoveryRetryReady` atomic to AudioEngine.h | Foundation for provenance plumbing |
| 4 | P1-B | Set `recoveryRetryReady` in recovery handler (Timer.cpp) | Writer side — safe to add before consumer |
| 5 | P2 | Consumer merge in RebuildDispatch.cpp | Read side — `exchange(false)` consumes the flag |
| 6 | P3 | Change `processDeferredAdmission(wasRecoveryWake)` signature + budget reset gating | Core consumer logic |
| 7 | Step 8 | Add `jassert(rebuildThreadId)` to `resetDeferredRetryBudget()` | Validate ownership before Option C adds more |
| 8 | Step 4 | Add `recoveryArmed` latch to DeferredPublishSlot + enqueueDeferred | Latch mechanics, no ownership risk |
| 9 | Step 5 | Remove `resetDeferredRetryBudget()` from Timer.cpp recovery handler | Now handled in P3 at rebuild thread |
| 10 | Step 7 | Option C: `deferredClearRequested_` + `requestDeferredClear()` + drain in shutdown block | Most complex — do last after all owners validated |
| 11 | Step 7 | Remove EmergencyDrain `clearDeferredForShutdown()` call (ReleaseResources.cpp:359) | Must come after Step 8 jassert is in place |

**Critical ordering constraint:** Step 7 (Option C) must be implemented **after** Step 8 (owner assertion), because the jassert will validate that `clearDeferredForShutdown()` is only called from the rebuild thread. The EmergencyDrain removal (also Step 7) depends on this assertion working correctly.

### 5.1 Step 0 Completion Checklist

- [x] AudioEngine.h: `publishRetryReady`, `rebuildMutex`, `rebuildCV`, `rebuildThreadShouldExit`, `recoveryPending` verified at h:2699-2720
- [x] AudioEngine.h: No `recoveryRetryReady` / `wasRecoveryWake` / `deferredClearRequested_` exist — confirmed new additions
- [x] RebuildDispatch.cpp: CV predicate (h:850-854), merge site (h:878-880), processDeferredAdmission call (h:886) verified
- [x] RebuildDispatch.cpp: Break-before-merge ordering (h:856 vs h:878) — CRITICAL for Option C
- [x] AudioEngine.Timer.cpp: Recovery handler (h:1721-1746) — ownership violation at resetDeferredRetryBudget confirmed
- [x] AudioEngine.Timer.cpp: 3 clearDeferredForShutdown callers (1642, 1804, 1824) verified
- [x] AudioEngine.Threading.cpp: Ordinary retry producer (h:286) — clean, only sets publishRetryReady
- [x] RuntimePublicationOrchestrator.h: processDeferredAdmission decl (h:192), resetDeferredRetryBudget inline (h:171-173), clearDeferredForShutdown decl (h:195), retry members (h:277-283) verified
- [x] RuntimePublicationOrchestrator.h: DeferredPublishSlot struct (h:32-39) — no recoveryArmed field
- [x] RuntimePublicationOrchestrator.cpp: enqueueDeferred (437), clearDeferredForShutdown (534), processDeferredAdmission (636), RetryExhaustedDiscard (489/499) verified
- [x] RuntimePublicationOrchestrator.cpp: deferredRecoveryRearmed_ write-only (h:283, .cpp:550) — 0 reads confirmed
- [x] ReleaseResources.cpp: stopRebuildThread (line 202), publishRetryReady=false (line 175), EmergencyDrain clearDeferredForShutdown (line 359) verified
- [x] RuntimePublicationState.h: kMaxDeferredRetries=2 comment (h:17), RetryExhaustedDiscard enum (h:25) verified
- [x] DeferredFlowIntegrationTests.cpp: single reference to processDeferredAdmission at line 88 (test comment)
- [x] DeferredPublishViewStateMachineTests.cpp: no direct processDeferredAdmission calls (uses test access pattern)
- [x] Step 4 leakage check: submitPublishRequest, enqueueDeferred, trySubmitImpl — none accept wasRecoveryWake
- [x] Step 4 single caller: RebuildDispatch.cpp:886 — only production call site

### 5.2 Test Verification Plan (for post-implementation gates)

**Gate A — Static audit:** Re-verify all 4 ownership violations from Section 4 are resolved.
**Gate B — Compile (Debug+Release):** No warnings on the new atomic or signature change.
**Gate C — CTest existing tests:** DeferredFlowIntegrationTests + DeferredPublishViewStateMachineTests still pass.
**Gate D — Retry state-machine test:** After kMaxDeferredRetries fix (Step 6).

**Counting trace** (enqueueDeferred at .cpp:437, increment at .cpp:472, check at .cpp:489):

The check uses `>=` after increment. Trace with `kMaxDeferredRetries=2`:

| Event | sameObligation | Action | count after increment | Check `count >= 2` | Result |
|-------|---------------|--------|----------------------|---------------------|--------|
| Initial enqueue | false (gen mismatch) | reset gen, count=0 | 0 | 0 >= 2? No | Slot stored — initial deferral |
| Re-defer 1 | true | ++count | 1 | 1 >= 2? No | Slot stored — retry 1 |
| Re-defer 2 | true | ++count | 2 | 2 >= 2? **Yes** | Discarded — retry 2 thrown away! |

**With `>=` and kMax=2, only 1 retry is actually allowed** — the 2nd re-defer is discarded. This contradicts the comment "2 回再駆動許容、3 回目で諦" (2 redrives allowed, 3rd discarded).

To match the comment's intent (2 retries, discard on 3rd), the check must be **strict `>`**:

| Event | After increment | Check `count > 2` | Result |
|-------|-----------------|---------------------|--------|
| Initial enqueue | 0 | 0 > 2? No | Slot stored |
| Re-defer 1 | 1 | 1 > 2? No | Retry 1 |
| Re-defer 2 | 2 | 2 > 2? No | Retry 2 |
| Re-defer 3 | 3 | 3 > 2? Yes | Retry 3 discarded |

**Step 6 implementation requires BOTH changes:**
1. Change `kMaxDeferredRetries` from `10` to `2` (h:279) — value fix.
2. Change the check at .cpp:489 from `>=` to `>` — operator fix to match comment semantics.

Without the operator change, kMax=2 is a no-op regression (fewer retries than the current kMax=10 allows). The operator fix is essential.

**Gate E — Coalescing test:** With `wasRecoveryWake` gating, verify recovery wakes don't coalesce with ordinary retry wakes (wasRecoveryWake=true resets budget, wasRecoveryWake=false increments).
**Gate F — Shutdown ordering test:** Verify `clearDeferredForShutdown` drains before rebuild thread exit during `stopRebuildThread()` + `join()`. Verify EmergencyDrain path no longer calls it (removed at ReleaseResources.cpp:359).
**Gate G — D135-2 rerun:** The full D135-2 test suite should pass with the provenance plumbing in place.

---

## 6. D135-8 vs D135-7 Mapping

| D135-7 Item | D135-8 Step | Status |
|-------------|-------------|--------|
| D1 `recoveryRetryReady` (provenance flag) | P1-A, P1-B | Locked |
| D2 Consumer merge (`wasRecoveryWake` snapshot) | P2 | Locked |
| D3 `processDeferredAdmission(bool)` signature change | P3 | Locked |
| D4 `recoveryArmed` latch | Step 4 | Safe (leakage check passed) |
| D5 Remove `resetDeferredRetryBudget` from Timer.cpp | Step 5 | Locked |
| kMaxDeferredRetries 10→2 + `>=`→`>` operator fix | Step 6 | Locked |
| P8 `clearDeferredForShutdown` ownership repair (Option C) | Step 7 | Locked (EmergencyDrain removal confirmed) |
| Owner assertion on `resetDeferredRetryBudget` | Step 8 | Locked |
| Dead code removal `deferredRecoveryRearmed_` | Step 9 | Locked |

---

## 7. Risk Register

| Risk | Severity | Mitigation | Status |
|------|----------|------------|--------|
| kMax=2 semantic mismatch (`>=` allows only 1 retry, comment says 2) | High | Change `>=` to `>` at .cpp:489 alongside value fix to 2 at h:279 | **Resolved** — both changes locked
| EmergencyDrain removal leaves residual deferred state | Low | Drain happens in shutdown-block before rebuild thread exit; EmergencyDrain is post-join, state already clean | Resolved |
| `recoveryRetryReady` race with CV predicate | Low | Flag is provenance-only; `publishRetryReady` (under rebuildMutex) handles the actual wake. `exchange(false)` at merge is atomic-safe. | Resolved |
| Test comment at DeferredFlowIntegrationTests.cpp:88 needs updating | Trivial | Update comment to reflect `processDeferredAdmission(bool)` signature | Locked |
| `jassert(rebuildThreadId)` on resetDeferredRetryBudget trips during recovery handler (pre-Step 5) | High | Step 5 (remove from Timer.cpp) must accompany or precede Step 8 (add jassert) | Ordering locked |

---

## 8. Conclusion

Step 0 read-only re-audit is **complete**. All P4-P9 implementation targets verified against live source. The two critical ordering findings — (1) the rebuild loop `break` at h:856 precedes the merge at h:878 (Option C insertion must be pre-break), and (2) EmergencyDrain runs after rebuild thread join (call is redundant and must be removed if owner assertion is added) — are documented with concrete insertion points.

**kMax=2 semantic discrepancy RESOLVED:** The `>=` operator at .cpp:489 with kMax=2 allows only 1 retry, contradicting the comment "2 回再駆動許容、3 回目で諦". Step 6 must fix BOTH the value (10→2 at h:279) AND the operator (`>=`→`>` at .cpp:489). Trace with `>` and kMax=2: initial (count=0, 0>2=false) → retry 1 (count=1, 1>2=false) → retry 2 (count=2, 2>2=false) → retry 3 discarded (count=3, 3>2=true). This matches the comment exactly.

**Production source changes: 0.** Proceeding to D135-8 implementation (Steps 1-9) per recommended sequence in Section 5.
