# D135-8 — Step 3 (P1-A) Audit: `recoveryRetryReady` provenance flag declaration

**Status:** READ-ONLY verification — no build / no test
**Date:** 2026-08-30
**Prerequisite:** Steps 1–2 — PASS
**Scope:** Add provenance-only `std::atomic<bool> recoveryRetryReady` to `AudioEngine.h`.

---

## 0. Edit applied

`src/audioengine/AudioEngine.h` — inserted immediately after `publishRetryReady` (h:2718):

```cpp
    bool publishRetryReady = false;
    // D135-8: Recovery crossfade-timeout wake provenance.
    // This is provenance only; it is not part of the rebuildCV wake predicate.
    std::atomic<bool> recoveryRetryReady { false };
```

## 1. Twelve-point audit

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | `recoveryRetryReady` declared exactly once | **PASS** | 1 match in `AudioEngine.h`; 0 in every other `src/` + `tests/` file (`grep -rc` = 1 only at h:2721) |
| 2 | Type is `std::atomic<bool>` | **PASS** | h:2721 `std::atomic<bool> recoveryRetryReady { false };` |
| 3 | Initial value `false` | **PASS** | braced init `{ false }` at h:2721 |
| 4 | Placed after `publishRetryReady` | **PASS** | `publishRetryReady` h:2718 → `recoveryRetryReady` h:2721 (comment at 2719–2720) |
| 5 | Writer count = 0 | **PASS** | No `.store()`, `.exchange()`, or `=` assignment anywhere (`grep -rn "recoveryRetryReady\s*="` → 0, excluding decl init) |
| 6 | Reader count = 0 | **PASS** | No `.load()`/`exchange`/comparison anywhere |
| 7 | CV predicate excludes `recoveryRetryReady` | **PASS** | `rebuildCV.wait` predicate at `AudioEngine.RebuildDispatch.cpp:849–854`: `hasPendingTask \|\| publishRetryReady \|\| recoveryPending \|\| rebuildThreadShouldExit` — no `recoveryRetryReady`; also 0 references in `RebuildDispatch.cpp` entirely |
| 8 | `publishRetryReady` unchanged | **PASS** | h:2718 `bool publishRetryReady = false;` (plain bool, unchanged) |
| 9 | `recoveryPending` unchanged | **PASS** | h:2711 `bool recoveryPending = false;` (unchanged) |
| 10 | `lastRecoveryPublishSeq_` unchanged | **PASS** | `RuntimePublicationOrchestrator.h:281` decl; `.cpp:165/.cpp:168` getter/setter — untouched |
| 11 | Step 1 intact (`kMax=2`, `>`) | **PASS** | h:279 `= 2`; `.cpp:489` `> kMaxDeferredRetries` |
| 12 | Step 2 intact (`deferredRecoveryRearmed_` = 0 refs) | **PASS** | `grep -rn` in src/ + tests/ → 0 matches |

## 2. CV predicate (C7 — full body)

Source: `src/audioengine/AudioEngine.RebuildDispatch.cpp:849–855`

```cpp
rebuildCV.wait(lock, [this] {
    return hasPendingTask
        || publishRetryReady
        || recoveryPending                              // ★ recovery intent (existing)
        || convo::consumeAtomic(rebuildThreadShouldExit, std::memory_order_acquire);
});
```

`recoveryRetryReady` is a provenance flag **only** — it does not and must not participate in the wake predicate. The existing `publishRetryReady` (set by both the ordinary retry path and the recovery handler) remains the actual wake trigger; the recovery handler will stamp provenance onto it in Step 4 (P1-B).

## 3. Verdict

**Step 3 (P1-A) — PASS.** Declaration isolated to `AudioEngine.h`; zero writers/readers; CV predicate unmodified; Steps 1–2 invariants preserved. Proceed to Step 4 (P1-B): add `recoveryRetryReady` producer write in `AudioEngine.Timer.cpp` recovery handler (without removing `resetDeferredRetryBudget()` — that is Step 5).
