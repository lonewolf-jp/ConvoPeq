# D135-8 — Step 4 (P1-B) Provenance Writer Audit (Read-Only Verification)

**Status:** READ-ONLY verification — no build / no test
**Date:** 2026-08-30
**Spec locked by:** `evidence/D135-8_IMPLEMENTATION_PREFLIGHT.md` §8 (P1-B: producer-side provenance stamp)
**Precondition:** Step 3 (P1-A) PASS — `recoveryRetryReady` declared `std::atomic<bool>` in `AudioEngine.h` (provenance only, not in CV predicate).
**Scope:** One surgical edit — single exact-string `Edit` in `AudioEngine.Timer.cpp`'s crossfade-timeout recovery handler.

---

## 0. Edit applied

**File:** `src/audioengine/AudioEngine.Timer.cpp` — recovery handler locked section (inside
`AudioEngine::onHealthEvent`, under `if (runtimeOrchestrator_ != nullptr && runtimeOrchestrator_->hasDeferredRequest())`).

| | Before | After |
|---|--------|-------|
| locked block | `publishRetryReady = true;` (only) | `convo::publishAtomic(recoveryRetryReady, true, std::memory_order_release);` **then** `publishRetryReady = true;` |

```cpp
// AFTER STEP 4
{
    std::lock_guard<std::mutex> lock(rebuildMutex);
    convo::publishAtomic(recoveryRetryReady, true, std::memory_order_release);
    publishRetryReady = true;
}
rebuildCV.notify_one();
```

The provenance stamp is written **under the same `rebuildMutex` guard and before** `publishRetryReady = true`, preserving the invariant: stamp first, then set the wake trigger, then notify.

## 1. Nine-point read-only audit

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | `recoveryRetryReady` has exactly **1 writer** | **PASS** | `AudioEngine.Timer.cpp:1748` — `convo::publishAtomic(recoveryRetryReady, true, std::memory_order_release)` |
| 2 | `recoveryRetryReady` has **0 readers** (reader is P2/P3, not yet) | **PASS** | No `.load()`/`.exchange()`/comparison of `recoveryRetryReady` anywhere in `src/` |
| 3 | Writer sits in the recovery handler (not ordinary-retry path) | **PASS** | Inside `onHealthEvent` crossfade-timeout recovery block (`Timer.cpp:1721`), distinct from `AudioEngine.Threading.cpp:286` ordinary-retry producer |
| 4 | Writer is under `rebuildMutex` (matches Step 3 declaration scope note) | **PASS** | `std::lock_guard<std::mutex> lock(rebuildMutex);` immediately precedes the stamp |
| 5 | Stamp precedes `publishRetryReady = true` (provenance-before-trigger invariant) | **PASS** | stamp at `:1748`, trigger at `:1749` |
| 6 | Writer uses project `convo::publishAtomic` wrapper (not raw `.store()`) | **PASS** | `convo::publishAtomic(recoveryRetryReady, true, std::memory_order_release)` |
| 7 | `recoveryRetryReady` is **NOT** added to `rebuildCV` wake predicate | **PASS** | `AudioEngine.RebuildDispatch.cpp:849-854` predicate is `hasPendingTask || publishRetryReady || recoveryPending || consumeAtomic(rebuildThreadShouldExit)` — `recoveryRetryReady` absent |
| 8 | Step 3 declaration intact (not overwritten) | **PASS** | `AudioEngine.h:2721` — `std::atomic<bool> recoveryRetryReady { false };` |
| 9 | `resetDeferredRetryBudget()` **NOT** removed (that is Step 5) | **PASS** | `AudioEngine.Timer.cpp:1724` still calls `runtimeOrchestrator_->resetDeferredRetryBudget()`; `.h:171` definition intact |

## 2. Changed-files surface (vs HEAD)

`git diff --stat HEAD` for the Step 3+4 scope:
```
src/audioengine/AudioEngine.Timer.cpp   | 32 ++++++++++++++++++++++++++++++++
src/audioengine/AudioEngine.h           | 22 +++++++++++++++++++++-
```
Only `AudioEngine.h` (Step 3 declaration) and `AudioEngine.Timer.cpp` (Step 4 writer) are touched
for Steps 3–4. (The 80-line baseline diff in `RuntimePublicationOrchestrator.*` vs HEAD is
pre-existing uncommitted D127-E/D132/D133 work, documented in Step 1 audit §3 — not introduced here.)

## 3. Verdict

**Step 4 (P1-B) — PASS.** Producer-side provenance stamp added; exactly 1 writer, 0 readers;
provenance-only (excluded from CV predicate); written under `rebuildMutex` before the wake trigger;
`resetDeferredRetryBudget()` deliberately left in place for Step 5.

**Step 4 PASS clears the Step 4→Step 5 gate.** Proceed to Step 5 (remove blind `resetDeferredRetryBudget()`
from the recovery handler) per the user's ordered sequence.
