# D135-8 Step 7 — P3 Consumer Reset Audit (diff-only, no build/test)

**Status:** PASS · **Date:** 2026-08-30 · **Layer:** Implementation (P3 budget reset relocation)
**Files changed:** 2 · **`src/audioengine/RuntimePublicationOrchestrator.h`** (declaration + comment), **`src/audioengine/RuntimePublicationOrchestrator.cpp`** (definition + provenance-gated reset)
**Baseline:** HEAD `f39fcd3`. Audit is **read-only / diff-only** — no build, no CTest (per standing gate rule: compile/static gates run only after P1–P3 complete).

## What Step 7 does
Step 6 left an **intentional compile-block**: `RebuildDispatch.cpp:914` called `processDeferredAdmission(wasRecoveryWake)` while the signature still accepted `()`. Step 7 resolves exactly that block by (1) changing the signature to `processDeferredAdmission(bool wasRecoveryWake)` and (2) relocating the producer-side `resetDeferredRetryBudget()` (removed in Step 5) to the **consumer side**, gated on the `wasRecoveryWake` provenance flag, at the START of `processDeferredAdmission` — before `peekDeferred()` per D135-7 preflight.

Net effect vs Step-6 state: `resetDeferredRetryBudget()` goes from **0 callers → 1 caller**, and the compile-block is cleared.

## Exact edits (vs Step-6 on-disk state)

### Edit 1 — `RuntimePublicationOrchestrator.h` (declaration, was line 192 → 193)
```cpp
// ★ D135-8 Step 7 (P3): wake-provenance discriminator wired end-to-end (P1 blocked → resolved).
void processDeferredAdmission(bool wasRecoveryWake) noexcept;
```

### Edit 2 — `RuntimePublicationOrchestrator.cpp` (definition start, :635–654)
```cpp
void RuntimePublicationOrchestrator::processDeferredAdmission(bool wasRecoveryWake) noexcept
{
    jassert(std::this_thread::get_id() == engine_.rebuildThreadId());
    // ★ D135-8 Step 7 (P3): recovery-wake provenance gate (D135-5 blind-bool blocker resolved by Step 6).
    //   ... (wasRecoveryWake == true iff recovery crossfade-timeout path stamped recoveryRetryReady
    //        at AudioEngine.Timer.cpp:1748 under rebuildMutex, consumed read-and-clear at
    //        AudioEngine.RebuildDispatch.cpp:888) ...
    if (wasRecoveryWake)
        resetDeferredRetryBudget();          // ← relocation of the Step-5-removed producer reset
    if (!convo::consumeAtomic(hasDeferred_, std::memory_order_acquire))
        return;

    auto view = peekDeferred();              // ← reset is BEFORE peekDeferred() (D135-7)
```

## 10-criterion read-only audit

| # | Criterion | Result | Evidence (line) |
|---|-----------|--------|-----------------|
| 1 | Signature change matches the single call site (no stranded bare call) | PASS | decl `h:193`; def `cpp:635`; call `RebuildDispatch.cpp:914` passes `wasRecoveryWake` |
| 2 | Reset is gated on `wasRecoveryWake` (consumer cannot blindly reset) | PASS | `cpp:649` `if (wasRecoveryWake)` |
| 3 | Reset placed at START before `peekDeferred()` (D135-7 ordering) | PASS | reset `cpp:650`; `hasDeferred_` guard `cpp:651-652`; `peekDeferred` `cpp:654` |
| 4 | `resetDeferredRetryBudget()` definition body unchanged | PASS | `h:171-174` (resets `deferredRetryGeneration_` + `deferredRetryCount_` to 0) |
| 5 | Ordinary-retry increment path untouched (must still increment, not reset) | PASS | enqueue/re-defer path `cpp:470-478` (sameObligation `++deferredRetryCount_` at :474; new-gen resets at :476-477); exhaustion `cpp:489` `>` |
| 6 | Producer stamp intact (recovery handler unchanged by Step 7) | PASS | `Timer.cpp:1748` `convo::publishAtomic(recoveryRetryReady, true, release)` under `rebuildMutex` (:1747); whole block guarded by `hasDeferredRequest()` (:1721) |
| 7 | Consumer consume intact (provenance read-and-clear) | PASS | `RebuildDispatch.cpp:888` `convo::exchangeAtomic(recoveryRetryReady, false, acq_rel)` |
| 8 | `recoveryRetryReady` NOT added to `rebuildCV` wake predicate | PASS | predicate `RebuildDispatch.cpp:854-859` = `hasPendingTask \|\| publishRetryReady \|\| recoveryPending \|\| consumeAtomic(rebuildThreadShouldExit)` — provenance-only |
| 9 | Thread-safety: reset touches only RebuildThread-owned plain members, called only here | PASS | `resetDeferredRetryBudget` writes `deferredRetryGeneration_`/`deferredRetryCount_` (plain, RebuildThread single-owner per `//:470` comment); recovery handler (Timer.cpp) does NOT touch these fields; jassert `cpp:637` enforces `rebuildThreadId()` |
| 10 | `resetDeferredRetryBudget()` now has exactly 1 caller (was 0 after Step 5) | PASS | `rg` confirms sole caller `cpp:650`; def `h:171` |

## AtomicAccess.h API verification (carried from Step 6, re-confirmed)
Authoritative source `src/audioengine/AtomicAccess.h` (read-only):
- `consumeAtomic` (`h:60-63`): **2-arg load-only** template (`std::atomic_load_explicit`); no 3-arg/exchange overload exists.
- `exchangeAtomic` (`h:68-73`): **read-and-clear** primitive (`std::atomic_exchange_explicit`, default `acq_rel`); this is what the consumer uses (`RebuildDispatch.cpp:888`) and what pairs with the producer's release-store `publishAtomic` (`Timer.cpp:1748`).
- `resetDeferredRetryBudget()` itself uses **no atomics** (plain member writes `h:171-174`) — safe because it is RebuildThread-exclusive; no atomic primitive is required for it.
- Step 7 introduces **no new atomic operations**; it only re-homes an existing plain-member reset behind a `bool` parameter. The `bool wasRecoveryWake` is a loop-local in the RebuildThread (declared outer-scope `RebuildDispatch.cpp:848`, assigned under lock `:888`, read at `:914`) — single-writer/single-reader, no atomicity needed.

## Ordering & provenance chain (end-to-end, post-Step-7)
```
crossfade-timeout recovery (Timer.cpp:1721 hasDeferredRequest() guard)
  └─ h:1747 lock(rebuildMutex)
      ├─ h:1748 publishAtomic(recoveryRetryReady, true,  release)   ← provenance stamp
      └─ h:1749 publishRetryReady = true                            ← wake trigger
  └─ h:1751 rebuildCV.notify_one()
→ RebuildThread wakes (RebuildDispatch.cpp:854 predicate)
  └─ :884-890 consume block: wasRecoveryWake = exchangeAtomic(recoveryRetryReady,false,acq_rel)  ← read-and-clear
                    doDeferredPublish = publishRetryReady; publishRetryReady = false
  └─ :914 processDeferredAdmission(wasRecoveryWake)   ← compile-block resolved; flag threaded in
→ processDeferredAdmission (cpp:635)
  └─ :649 if (wasRecoveryWake) resetDeferredRetryBudget()   ← P3 relocation, reset BEFORE peek
  └─ :651 hasDeferred_ guard; :654 peekDeferred()
```
Recovery redrive ⇒ `wasRecoveryWake==true` ⇒ reset budget (fresh generation/count baseline). Ordinary retry ⇒ `wasRecoveryWake==false` ⇒ increment path at `cpp:474` untouched. The two producers are now **distinguishable** at the consumer (D135-5 blind-bool blocker fully resolved).

## Invariants preserved
- `recoveryRetryReady` remains **provenance only**; the `rebuildCV` predicate is byte-identical to Step 6.
- `publishRetryReady` remains the **sole wake trigger** (plain bool, `rebuildMutex`-guarded).
- `kMaxDeferredRetries = 2` (`h:279` via `RuntimeState.h:17`) and `>` operator (`cpp:489`) — "2 retries allowed, 3rd discarded" — unchanged since Step 1.
- Recovery handler `resetDeferredRetryBudget()` call site removed in Step 5 stays removed; reset now lives **only** at consumer `:650`.
- `lastRecoveryPublishSeq_` is a separate correlation stamp (`setRecoveryPublishSeq` `@ Timer.cpp:1723`, reader `h:167-169`); `resetDeferredRetryBudget` does **not** touch it — correct, it is not retry-exhaustion accounting.

## Compile-block resolution (Step 6 → Step 7)
| Site | Step-6 state | Step-7 state |
|------|--------------|--------------|
| `h:193` decl | `processDeferredAdmission() noexcept;` | `processDeferredAdmission(bool wasRecoveryWake) noexcept;` |
| `cpp:635` def | `processDeferredAdmission() noexcept` | `processDeferredAdmission(bool wasRecoveryWake) noexcept` |
| `RebuildDispatch.cpp:914` call | `processDeferredAdmission(wasRecoveryWake)` — **does not compile** (no matching sig) | `processDeferredAdmission(wasRecoveryWake)` — **compiles** ✓ |
| `resetDeferredRetryBudget` callers | 0 (definition retained, dead) | 1 (consumer `:650`), gated on recovery provenance |

## Diff stat (vs Step-6 on-disk state)
- `RuntimePublicationOrchestrator.h`: +1 line (provenance comment) — decl line shifted 192→193.
- `RuntimePublicationOrchestrator.cpp`: +13 lines (9 provenance comment + `if (wasRecoveryWake)` + `resetDeferredRetryBudget();` + signature param). Net surgical; no other production file altered.

## Next gates (A–G) — still gated
Gates A–G (static audit, compile, CTest, retry state-machine, coalescing, shutdown-ordering, D135-2 rerun) remain **not started** — they require a full P1+P2+P3 closure, which Step 7 now provides. Per the no-gate rule, **no build or test is executed in this step**; that handoff awaits the user's explicit gate signal.

**Step 7 result: PASS.** Steps 0–7 complete. Compile-block resolved; recovery-retry-budget reset now consumer-side and provenance-gated.
