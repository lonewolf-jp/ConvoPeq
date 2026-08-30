# D135-8 — Step 5 Reset Relocation Audit (Read-Only Verification)

**Status:** READ-ONLY verification — no build / no test
**Date:** 2026-08-30
**Spec locked by:** `evidence/D135-8_IMPLEMENTATION_PREFLIGHT.md` §8 (Edit #5a: remove blind producer-side reset)
**Precondition:** Step 4 (P1-B) PASS — `recoveryRetryReady` provenance stamp wired in the recovery handler.
**Scope:** One surgical edit — single exact-string `Edit` removing the `resetDeferredRetryBudget()` call from the recovery handler and replacing it with an explanatory comment.

---

## 0. Edit applied

**File:** `src/audioengine/AudioEngine.Timer.cpp` — recovery handler (`onHealthEvent` crossfade-timeout branch).

| | Before | After |
|---|--------|-------|
| reset call | `runtimeOrchestrator_->resetDeferredRetryBudget();` | *(removed)* + 5-line explanatory comment |

```cpp
// AFTER STEP 5
const auto recoverySeq = getLastCommittedPublicationSequence();
runtimeOrchestrator_->setRecoveryPublishSeq(recoverySeq);
// ★ D135-5/8: blind producer-side reset removed. Recovery-retry budget reset is now
//   gated on recoveryRetryReady provenance in processDeferredAdmission (P3).
//   The provenance stamp below (publishRetryReady) wakes the rebuild thread; the
//   atomic recoveryRetryReady flag lets the consumer distinguish recovery redrive
//   (reset budget) from ordinary retry (increment count).
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
...
{
    std::lock_guard<std::mutex> lock(rebuildMutex);
    convo::publishAtomic(recoveryRetryReady, true, std::memory_order_release);
    publishRetryReady = true;
}
rebuildCV.notify_one();
```

The recovery redrive request (stamp + wake trigger + `setRecoveryPublishSeq`) is preserved; only the
blind, producer-side budget reset is removed. The reset is relocated to a `recoveryRetryReady`-gated
site in `processDeferredAdmission` (Step P3 — future, not this step).

## 1. Seven-point read-only audit

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | `resetDeferredRetryBudget()` **CALL** removed from `AudioEngine.Timer.cpp` recovery handler | **PASS** | `grep -n resetDeferredRetryBudget src/audioengine/AudioEngine.Timer.cpp` → no matches |
| 2 | Function **definition** retained in `.h` (re-wired in P3, not deleted) | **PASS** | `RuntimePublicationOrchestrator.h:171` — `void resetDeferredRetryBudget() noexcept {` intact; now 0 callers (intended transient state) |
| 3 | Step 4 provenance writer intact (unaffected by Step 5) | **PASS** | `AudioEngine.Timer.cpp:1748` — `convo::publishAtomic(recoveryRetryReady, true, std::memory_order_release)`; `:1749` `publishRetryReady = true;` (shifted +4 from Step 4's `:1744` due to net +4 lines: −1 call +5 comment) |
| 4 | `setRecoveryPublishSeq(recoverySeq)` still called (recovery path metadata preserved) | **PASS** | `AudioEngine.Timer.cpp:1723` |
| 5 | `rebuildCV` wake predicate unchanged (recoveryRetryReady still absent) | **PASS** | `AudioEngine.RebuildDispatch.cpp:849-854` unchanged |
| 6 | Only `AudioEngine.Timer.cpp` changed by Step 5 | **PASS** | `git diff --stat HEAD -- src/audioengine/AudioEngine.Timer.cpp` → only this file appears |
| 7 | Recovery handler control flow intact (lock block + notify_one still reached) | **PASS** | `sed -n '1721,1752p'` shows intact `if { ... lock_guard ... publishRetryReady ... rebuildCV.notify_one(); }` |

## 2. Note on the transient 0-caller state

After Step 5, `resetDeferredRetryBudget()` has **0 callers** for the interval until Step P3
injects the recovery-gated reset into `processDeferredAdmission`. This is the intended intermediate
state per the user-ordered sequence: D135-5 establishes that the *producer-side blind reset* must be
removed *before* the *consumer-side provenance-gated reset* is installed, so that the ordinary-retry
producer can never observe a reset it did not earn. The definition is retained (not deleted) so P3 is
a pure addition.

## 3. Verdict

**Step 5 — PASS.** Blind producer-side `resetDeferredRetryBudget()` removed from the recovery handler;
recovery redrive request, provenance stamp (Step 4), and wake trigger all preserved; only
`AudioEngine.Timer.cpp` touched; CV predicate unchanged; function definition retained for P3.

---

## 4. D135-8 Steps 1–5 cumulative state

| Step | Target | Change | Status |
|------|--------|--------|--------|
| 0 | `evidence/D135-8_IMPLEMENTATION_PREFLIGHT.md` | Read-only preflight | DONE |
| 1 | `RuntimePublicationOrchestrator.h:279` / `.cpp:489` | `kMax` 10→2, `>=`→`>` | PASS |
| 2 | `RuntimePublicationOrchestrator.h:282-283` / `.cpp:550` | Remove dead `deferredRecoveryRearmed_` | PASS |
| 3 (P1-A) | `AudioEngine.h:2721` | Declare `std::atomic<bool> recoveryRetryReady {false}` | PASS |
| 4 (P1-B) | `AudioEngine.Timer.cpp:1748` | Producer provenance stamp (`convo::publishAtomic`) | PASS |
| 5 | `AudioEngine.Timer.cpp:1724` | Remove blind `resetDeferredRetryBudget()` call | PASS |

## 5. Next steps (not yet started)

- **BLOCKED P1**: `processDeferredAdmission() → processDeferredAdmission(bool)` wake-provenance signal (D135-5: blind `publishRetryReady` cannot distinguish producers without a signal). Requires Step 3/P1-A→P3 ordering.
- **P2**: consumer-side merge in `AudioEngine.RebuildDispatch.cpp` — read `recoveryRetryReady`, gate budget reset.
- **P3**: relocate `resetDeferredRetryBudget()` to consumer side (gated on provenance).
- **Gates A–G**: static audit, compile, CTest, retry state-machine test, coalescing test, shutdown ordering test, D135-2 rerun.
