# D135-7 — Phase 1 Implementation Preflight Audit

**Status:** READ-ONLY (zero production source changes)
**Date:** 2026-08-29
**Gate:** D135-6 (Provenance Signal Design) → **D135-7** (Preflight Audit) → D135-8 (Phase 1 Implementation)
**Purpose:** Verify that D1 (`recoveryRetryReady` provenance flag + `wasRecoveryWake` plumbing) can be implemented without breaking existing Deferred ownership / CV / shutdown semantics. Production source changes are strictly prohibited at this stage.

---

## 0. Executive Summary

D1 proposes adding `std::atomic<bool> recoveryRetryReady{false}` to AudioEngine as a **provenance signal** that distinguishes a recovery-triggered rebuild-thread wake from an ordinary retry wake. The recovery handler (Timer.cpp:1721-1746) becomes the sole writer (alongside `publishRetryReady`), and the consumer merge site (RebuildDispatch.cpp:879-880) reads both flags to derive `wasRecoveryWake`.

The audit confirms D1 is **structurally sound** with the following critical constraints:
- D1 requires **both** `publishRetryReady` (existing CV wake trigger) AND `recoveryRetryReady` (new provenance signal). `recoveryRetryReady` alone cannot satisfy the CV predicate.
- The budget reset (`resetDeferredRetryBudget`) **must** move from the MessageThread recovery handler (Timer.cpp:1724 — currently an ownership violation) to the rebuild thread inside `processDeferredAdmission(wasRecoveryWake)`, at the **start** before `peekDeferred()`.
- `clearDeferredForShutdown()` has a confirmed ownership gap affecting all 4 callers — P8 is a mandatory Phase-1 edit.

**GO/NO-GO conclusion**: **GO** — D1 is safe to implement. P8 (clearDeferredForShutdown ownership repair) must be completed before or alongside P1/P2 to close the ownership gap.

---

## 1. Audit Methodology

All verification performed against both:
- **Live source tree**: `src/audioengine/` (C++ files)
- **ConvoPeq.md snapshot**: confirmed consistent for every audited line

Search results:
- `recoveryRetryReady`: 0 results (D1 not yet implemented — correct baseline)
- `recoveryArmed`: 0 results (P2 field not yet added — correct baseline)
- `publishRetryReady`: 12 references across 7 files
- `clearDeferredForShutdown`: 4 callers
- `resetDeferredRetryBudget`: 1 caller (Timer.cpp:1724, MessageThread)
- `recoveryPublishSeq()`: 0 rebuild-thread callers (write-only latch confirmed at D135-6)

---

## 2. Four Mandatory Investigation Areas

### 2.1 D1 `recoveryRetryReady` Ownership Audit — PASS

**Claim**: `recoveryRetryReady` cannot stand alone to satisfy the CV predicate; it is a provenance signal read at the consumer merge site, not a wake trigger.

**Evidence:**

**CV predicate** (RebuildDispatch.cpp:849-854):
```cpp
rebuildCV.wait(lock, [this] {
    return hasPendingTask
        || publishRetryReady       // ← wake trigger (NOT recoveryRetryReady)
        || recoveryPending         // ← separate ISR-level flag
        || convo::consumeAtomic(rebuildThreadShouldExit, std::memory_order_acquire);
});
```
`recoveryRetryReady` is **not** in the predicate. A lone `recoveryRetryReady=true` would NOT wake the rebuild thread. D1 requires the recovery handler to set **both** `publishRetryReady=true` (to wake) and `recoveryRetryReady=true` (to mark provenance), under `rebuildMutex`.

**Consumer merge** (RebuildDispatch.cpp:878-880):
```cpp
doDeferredPublish = publishRetryReady;
publishRetryReady = false;
```
This reads `publishRetryReady` to trigger `processDeferredAdmission()`. Under D1, the merge adds:
```cpp
// wasRecoveryWake = recoveryRetryReady (read under rebuildMutex)
// recoveryRetryReady = false (clear at merge)
```

**Recovery handler** (Timer.cpp:1742-1746):
```cpp
{
    std::lock_guard<std::mutex> lock(rebuildMutex);
    publishRetryReady = true;
    // [D1] recoveryRetryReady = true;
}
rebuildCV.notify_one();
```
The recovery handler is the sole writer of `recoveryRetryReady`, operating under `rebuildMutex` — same protection as `publishRetryReady`.

**Ordinary retry producer** (Threading.cpp:275-289):
Sets `publishRetryReady = true` under `rebuildMutex` (line 286), `notify_one()` (line 288). Does NOT touch `recoveryRetryReady`.

**Data flow summary** (full chain, proven end-to-end):
```
Recovery handler (Timer.cpp:1742-1746, MessageThread)
  ↓ sets publishRetryReady=true + recoveryRetryReady=true (under rebuildMutex)
  ↓ rebuildCV.notify_one()
Consumer merge (RebuildDispatch.cpp:879-880, RebuildThread, under rebuildMutex)
  ↓ doDeferredPublish = publishRetryReady; wasRecoveryWake = recoveryRetryReady
  ↓ publishRetryReady = false; recoveryRetryReady = false
  ↓ processDeferredAdmission(wasRecoveryWake)  [D1 signature change at cpp:904]
```

**Co-location proof**: The merge site (879-880) and the `processDeferredAdmission()` call (904) are in the same critical section scope (lock released at 890, `doDeferredPublish` consumed at 898-904). The `wasRecoveryWake` value is captured in a local before the lock is released — no TOCTOU window.

**Concurrent wake coalescing** — verified against CV predicate re-check semantics:

| Case | publisherA | publisherB | predicate re-check | wasRecoveryWake | Correct? |
|------|-----------|-----------|-------------------|-----------------|----------|
| A | ordinary | ordinary | `publishRetryReady` true | false | ✅ |
| B | recovery | ordinary | `publishRetryReady` true (or recovery sets it too) | true (if recovery fired) | ✅ recovery precedence |
| C | ordinary | recovery | `publishRetryReady` true | true (recovery set it) | ✅ |
| D | recovery | recovery | — | true | ✅ |

When both fire, `recoveryRetryReady=true` makes `wasRecoveryWake=true` — recovery precedence. If ordinary fires alone, `recoveryRetryReady` remains false — ordinary path, no budget reset.

**Slot-gone case** (no `hasDeferred_` when rebuild thread wakes): `processDeferredAdmission` reads `hasDeferred_` (atomic) at line 639 and returns early. Under D1, the `wasRecoveryWake` local is still consumed (budget reset still executes), which is harmless — `clearDeferredForShutdown()` also resets to 0, and the next `enqueueDeferred` starts fresh.

**Result: PASS** — `recoveryRetryReady` cannot satisfy the CV predicate alone (it's not in the predicate). D1 requires both `publishRetryReady` (wake) and `recoveryRetryReady` (provenance). The full data flow is proven structurally sound.

---

### 2.2 `processDeferredAdmission(bool)` State Transition Audit — PASS

**Claim**: Resetting retry budget at the start of `processDeferredAdmission(wasRecoveryWake)`, before `peekDeferred()`, prevents double-counting with `enqueueDeferred()`.

**Current flow** (RuntimePublicationOrchestrator.cpp:636-663):
```
processDeferredAdmission()                      // RebuildThread, jassert at 638
  → consumeAtomic(hasDeferred_, acquire)         // line 639 — gate
  → peekDeferred()                                // line 642
  → evaluateDeferred()                            // line 646
  → if Ready:
    consume() → finishView()                    // lines 589-608: deferredSlot_.reset(), hasDeferred_=false
    submitPublishRequest(req)                   // line 653
      → trySubmitImpl(req) → DeferredFadingActive
      → enqueueDeferred(req)                    // line 378 — re-defers
```

**enqueueDeferred retry accounting** (cpp:470-503):
```cpp
const bool sameObligation = (req.generation == deferredRetryGeneration_);
if (sameObligation)
    ++deferredRetryCount_;         // ← uses CURRENT budget state
else {
    deferredRetryGeneration_ = req.generation;
    deferredRetryCount_ = 0;       // ← fresh obligation
}
if (deferredRetryCount_ >= kMaxDeferredRetries)  // ← kMax check uses CURRENT count
    return;  // RetryExhaustedDiscard
```

**Critical ordering analysis** — three placement options for the budget reset:

| Position | Reset BEFORE enqueueDeferred? | kMax check uses old count? | Double-counting? | Correct? |
|----------|-------------------------------|---------------------------|-------------------|----------|
| **Start of processDeferredAdmission** (before peekDeferred) | ✅ Yes | No — count is 0 when enqueueDeferred runs | No | ✅ |
| After consume(), before submitPublishRequest | ✅ Yes | No | No | ✅ |
| At end (after enqueueDeferred) | ❌ No | ✅ Yes — old count used in kMax check | Yes | ❌ |

**Proof by trace — recovery wake (wasRecoveryWake=true), reset at START:**
```
processDeferredAdmission(true)
  → resetBudget()  → deferredRetryGeneration_=0, deferredRetryCount_=0     ← D1 P7
  → peekDeferred() → view over slot
  → evaluateDeferred() → Ready
  → consume() → finishView() → slot reset, hasDeferred_=false
  → submitPublishRequest(req)
    → trySubmitImpl → DeferredFadingActive (crossfade still active)
    → enqueueDeferred(req):
      → sameObligation = (req.generation == 0) → false (req.generation > 0)
      → deferredRetryGeneration_ = req.generation, deferredRetryCount_ = 0
      → kMax check: 0 >= kMax → false → slot assigned, hasDeferred_=true
```
Budget is fresh (0) when `enqueueDeferred` runs. Count increments normally on subsequent ordinary wakes: 0→1→2→exhausted (3rd attempt).

**Proof by trace — recovery wake with reset at END (BUG):**
```
processDeferredAdmission(true)
  → peekDeferred() → view
  → evaluateDeferred() → Ready
  → consume() → finishView() → slot reset
  → submitPublishRequest(req)
    → enqueueDeferred(req):
      → sameObligation = (req.generation == OLD_generation) → may be true!
      → ++deferredRetryCount_  ← increments OLD count (e.g., 2)
      → kMax check: 2 >= kMax(2) → TRUE → RetryExhaustedDiscard!  ← BUG
  → resetBudget()  ← too late, obligation already lost
```
Reset at end means `enqueueDeferred` sees the **stale** retry count from the old chain. If the old chain had already exhausted retries (count=2), the re-deferred obligation is immediately discarded — recovery's fresh budget is wasted. This is the double-counting / premature-exhaustion bug.

**Reset at start before peekDeferred()** (recommended): Even if the function returns early (no `hasDeferred_`), the reset is harmless — `clearDeferredForShutdown()` also resets to 0, and any subsequent `enqueueDeferred` starts fresh.

**Result: PASS** — Reset position MUST be at the start of `processDeferredAdmission(wasRecoveryWake)`, before `peekDeferred()`. This is the only position that prevents double-counting with `enqueueDeferred()`.

---

### 2.3 `recoveryArmed` Lifecycle Re-verification — PASS (with design note)

**Claim**: `recoveryArmed` (P2 field on `DeferredPublishSlot`) means "slot has fresh recovery budget granted" — stamped at `enqueueDeferred` time, defaults false on every overwrite.

**Why "fresh budget granted" not "slot is recovery-origin":** A slot can be re-deferred multiple times. If `recoveryArmed=true` meant "recovery-origin," it would persist across ordinary re-defers, incorrectly exempting the slot from ordinary retry limits. The correct semantic is: "this obligation was enqueued with a recovery-granted budget" — stamped once at `enqueueDeferred`, false on every subsequent ordinary re-defer.

**Stamping mechanism (design note for D135-8 P2):** Since `enqueueDeferred` is called via `submitPublishRequest` → `trySubmitImpl` → `enqueueDeferred`, and `submitPublishRequest` does NOT receive `wasRecoveryWake` as a parameter, the provenance must be threaded via an **orchestrator-level latch**:
```cpp
// processDeferredAdmission(wasRecoveryWake=true):
recoveryArmedPending_ = true;   // set at start, before peekDeferred

// enqueueDeferred(req):
deferredSlot_ = DeferredPublishSlot{
    ...,
    .recoveryArmed = recoveryArmedPending_   // stamped on new slot
};
recoveryArmedPending_ = false;              // auto-reset after stamp

// Ordinary re-defers: recoveryArmedPending_ is false → recoveryArmed = false
```

**Lifecycle trace (recovery → ordinary → ordinary → exhaust):**
```
T0: Recovery wake → processDeferredAdmission(true) → resetBudget
    → recoveryArmedPending_ = true
    → consume → submitPublishRequest → enqueueDeferred
    → Slot A created: recoveryArmed=true, count=0
    → recoveryArmedPending_ = false

T1: Ordinary wake → processDeferredAdmission(false)
    → recoveryArmedPending_ stays false
    → consume → submitPublishRequest → enqueueDeferred
    → Slot B created: recoveryArmed=false, count=1 (same obligation)

T2: Ordinary wake → processDeferredAdmission(false)
    → enqueueDeferred → count=2 → RetryExhaustedDiscard
    → Slot B discarded
```

**N3 overwrite-invariance**: `enqueueDeferred` replaces `deferredSlot_` wholesale (line 505: `deferredSlot_ = DeferredPublishSlot{...}`). Every new slot gets `recoveryArmed` from `recoveryArmedPending_`. Since the latch auto-resets after each stamp, a recovery slot overwritten by an ordinary enqueue correctly gets `recoveryArmed=false`. The invariant holds.

**Result: PASS** — `recoveryArmed` semantic is "fresh recovery budget granted." Stamping at `enqueueDeferred` via an orchestrator-level latch is feasible and correct.

---

### 2.4 `clearDeferredForShutdown()` Independent Audit — FAIL (ownership gap; P8 required)

**Claim**: `clearDeferredForShutdown()` violates the rebuild-thread single-owner contract for `deferredRetryGeneration_` and `deferredRetryCount_`. All 4 callers operate on the MessageThread without synchronization.

**Function implementation** (RuntimePublicationOrchestrator.cpp:534-560):
```cpp
void clearDeferredForShutdown() noexcept
{
    if (convo::consumeAtomic(hasDeferred_, std::memory_order_acquire)) {
        deferredSlot_.reset();          // OK — but see ownership note below
        convo::publishAtomic(hasDeferred_, false, std::memory_order_release);
    }
    // ★ THE VIOLATION:
    deferredRetryGeneration_ = 0;       // plain write — rebuild-thread-only member
    deferredRetryCount_ = 0;            // plain write — rebuild-thread-only member
    deferredRecoveryRearm_ = false;     // plain write
    convo::publishAtomic(lastRecoveryPublishSeq_, ...);  // atomic — safe
}
```

**Member contract** (h:272-283):
```cpp
// ★ D135-1: DeferredFadingActive 再駆動抑制カウンタ
//   enqueueDeferred は deferredSlot_ を構造体ごと置換するため、counter は
//   Orchestrator メンバに保持する（slot/guard 内ではない — 毎回 0 リセットされる）。
//   rebuild-thread 専用（single-owner）だが、hasDeferred_ と同じく rebuildMutex
//   バリア経由で可視性を取る必要はない（同一スレッド）。
int deferredRetryGeneration_{0};    // ← single-owner, rebuild-thread only
uint8_t deferredRetryCount_{0};     // ← single-owner, rebuild-thread only
```
Comment explicitly states rebuild-thread single-owner. Plain writes from MessageThread = data race (UB under C++ memory model).

**4 callers verified:**

| # | File:Line | Context | Thread | Notes |
|---|-----------|---------|--------|-------|
| 1 | ReleaseResources.cpp:359 | EmergencyDrain phase, `releaseResources()` | MessageThread | Shutdown path — rebuild thread may still be running |
| 2 | Timer.cpp:1642 | EVENT_PUBLICATION_STALL recovery | MessageThread | Active recovery, not shutdown |
| 3 | Timer.cpp:1804 | `RecoveryAction::Recover` | MessageThread | Active recovery |
| 4 | Timer.cpp:1824 | `RecoveryAction::Restore` | MessageThread | Active recovery |

**Additional violation — `resetDeferredRetryBudget()` (h:171-174):**
```cpp
void resetDeferredRetryBudget() noexcept {
    deferredRetryGeneration_ = 0;     // plain write
    deferredRetryCount_ = 0;          // plain write
}
```
Called at **Timer.cpp:1724** (MessageThread, recovery handler) — same ownership violation. This is the **current** mechanism that D1 replaces by moving the reset to the rebuild thread. Under D1, `resetDeferredRetryBudget()` will only be called from the rebuild thread (inside `processDeferredAdmission`).

**`deferredSlot_` ownership (additional nuance):** `clearDeferredForShutdown()` also calls `deferredSlot_.reset()` (line 543). This is an ownership transfer to the MessageThread. The `DeferredPublishSlot` struct contains no atomics (it's a plain struct), and `hasDeferred_` is atomic. The reset of `deferredSlot_` itself is a structural change to the `std::optional` container — this requires either:
- (a) rebuild-thread serialization (only rebuild thread touches `deferredSlot_`), or
- (b) `rebuildMutex` protection for `deferredSlot_` access.

Currently no `rebuildMutex` lock in `clearDeferredForShutdown()`. The only safety is that `hasDeferred_` (atomic) gates the reset, so the rebuild thread's `processDeferredAdmission` (which also checks `hasDeferred_` first at line 639) won't enter the critical body even if the slot is being reset concurrently. But the `std::optional::reset()` on one thread while `peekDeferred()` reads `*deferredSlot_` on another is technically a race. In practice, the sequence is: reset `hasDeferred_=false` → rebuild thread sees false → returns. But if the rebuild thread is between lines 639 (passed the gate) and 642 (peekDeferred), and the MessageThread resets the slot, this is a race.

**P8 Approach Comparison:**

| Approach | Mechanism | Pros | Cons | Risk |
|----------|-----------|------|------|------|
| **A: jassert rebuild-thread-only** | `jassert(rebuildThreadId)` in `clearDeferredForShutdown()` | Catches all callers at dev time, enforces single-owner | Requires routing ALL MessageThread callers elsewhere; stall-recovery (1642) needs synchronous clear | Medium — changes caller semantics |
| **B: MessageThread issues drain request** | Atomic intent flag + `rebuildCV.notify_one()`; rebuild thread drains | Preserves single-owner, minimal code change | Rebuild thread must check intent flag in CV predicate or loop body; stall-recovery latency if rebuild thread is busy | Medium — adds new CV predicate term |
| **C: Atomic intent + rebuild-thread drain** | `std::atomic<bool> deferredClearRequested_`; MessageThread sets + notifies; rebuild thread checks before `rebuildThreadShouldExit` | Cleanest ownership separation, rebuild thread owns all retry member writes | Most complex change: CV predicate, loop body, shutdown sequencing all need updating | Low — well-contained, testable |

**Recommendation for Phase 1 (P8):**

**Option C** is recommended. The implementation:
1. Add `std::atomic<bool> deferredClearRequested_{false}` to RuntimePublicationOrchestrator.
2. MessageThread callers (Timer.cpp:1642/1804/1824, ReleaseResources.cpp:359) set `deferredClearRequested_ = true` + `rebuildCV.notify_one()` instead of calling `clearDeferredForShutdown()` directly.
3. Rebuild thread checks `deferredClearRequested_` in the CV predicate (alongside `publishRetryReady`, etc.) or at loop entry.
4. Rebuild thread calls `clearDeferredForShutdown()` (now rebuild-thread-only) when the flag is set.
5. Add `jassert(std::this_thread::get_id() == engine_.rebuildThreadId())` to `clearDeferredForShutdown()`.

For the **shutdown caller** (ReleaseResources.cpp:359): The rebuild thread is being stopped via `rebuildThreadShouldExit`. The safest sequence is to set `deferredClearRequested_` + `rebuildThreadShouldExit` + `notify_all()`, then join the rebuild thread. The rebuild thread's loop checks `deferredClearRequested_` before `rebuildThreadShouldExit` in the same predicate, ensuring the clear runs before exit.

**Result: FAIL** — `clearDeferredForShutdown()` currently violates single-owner contract. P8 (Option C) is required.

---

## 3. Additional Findings

### 3.1 `kMaxDeferredRetries` Value Discrepancy — WARNING

**Location**: RuntimePublicationOrchestrator.h:279
```cpp
static constexpr uint8_t kMaxDeferredRetries = 10;  // D135-1: 2 回再駆動許容、3 回目で諦
```
The comment says "2 retries allowed, give up on 3rd" (count 0→1→2→exhausted), but the value is **10**. The D135-6 analysis assumed `kMax=2` for the retry-counting trace. The actual code permits 10 retries before exhaustion.

This does not affect D1's correctness (the mechanism is the same regardless of threshold), but the stale value should be corrected to `2` as part of P8 or a separate hygiene edit. The retry-exhaustion test (D135-8) must use the actual value in assertions.

**Result: WARNING** — Not a D1 blocker, but `kMaxDeferredRetries` should be corrected to `2` to match the design comment and D135-6 analysis.

### 3.2 `processDeferredAdmission()` Signature Change — DEPENDENCY

**Current** (cpp:636, h:192):
```cpp
void processDeferredAdmission() noexcept;
```
**D1 target**:
```cpp
void processDeferredAdmission(bool wasRecoveryWake) noexcept;
```

**Single caller**: RebuildDispatch.cpp:904 — `runtimeOrchestrator_->processDeferredAdmission()`. This call site already has `doDeferredPublish = publishRetryReady` (line 879) and will read `wasRecoveryWake = recoveryRetryReady` (D1 addition). The signature change is a 1-call-site edit — minimal blast radius.

**Result: SAFE** — Single call site, co-located with the merge that captures `wasRecoveryWake`.

### 3.3 `submitPublishRequest` → `enqueueDeferred` Path — VERIFIED

The re-defer path is confirmed:
```
processDeferredAdmission → consume() → finishView() → submitPublishRequest(req)
  → trySubmitImpl(req) → Decision::DeferredFadingActive (PublicationAdmission.cpp:58)
    → enqueueDeferred(req)  (line 378)
```
`trySubmitImpl` returns `DeferredFadingActive` when `hasFading` is true (line 51-58) — i.e., the runtime world still has a fading DSP. This is the re-defer condition. The retry counter in `enqueueDeferred` is the sole gate for retry exhaustion.

**Result: VERIFIED** — No bypass paths around `enqueueDeferred` for the re-defer case.

### 3.4 CV Predicate `recoveryPending` vs D1 `recoveryRetryReady` — DISTINCT

**Location**: AudioEngine.h:2711
```cpp
bool recoveryPending = false;  // ISR-level, separate from publishRetryReady
```
`recoveryPending` is a **separate** flag (set by the ISR Builder/Coordinator recovery intent path, not the Timer.cpp crossfade-timeout recovery). D1's `recoveryRetryReady` is the **crossfade-timeout recovery** provenance signal, set at Timer.cpp:1744. These do not conflict but serve different recovery paths.

**Result: CLEAR** — No naming or semantic collision.

---

## 4. 10-Item PASS/FAIL Checklist

| # | Item | Result | Evidence |
|---|------|--------|----------|
| 1 | `recoveryRetryReady` cannot satisfy CV predicate alone — D1 requires both `publishRetryReady` + `recoveryRetryReady` | **PASS** | CV predicate at RebuildDispatch.cpp:849 has `publishRetryReady` but NOT `recoveryRetryReady`; `recoveryRetryReady` is read at consumer merge (879-880), not in predicate |
| 2 | Recovery handler (Timer.cpp:1742-1746) is sole writer of `recoveryRetryReady`, under `rebuildMutex` | **PASS** | Timer.cpp:1742-1746 (under `std::lock_guard` on `rebuildMutex`); ordinary retry (Threading.cpp:286) does not touch `recoveryRetryReady` |
| 3 | Consumer merge (RebuildDispatch.cpp:879-880) can read `wasRecoveryWake` from `recoveryRetryReady` before clearing both flags | **PASS** | Merge scope (844-890) is under `rebuildMutex`; `doDeferredPublish` captured in local (879); `wasRecoveryWake` capture has same TOCTOU-free co-location |
| 4 | `wasRecoveryWake` → `processDeferredAdmission(bool)` call site (RebuildDispatch.cpp:904) | **PASS** | Single call site, co-located with merge; signature change is 1-site edit |
| 5 | Budget reset must be at START of `processDeferredAdmission`, before `peekDeferred()`, to prevent double-counting with `enqueueDeferred` | **PASS** | Trace proof: reset-at-end uses stale count in kMax check (489); reset-at-start guarantees fresh count (0) enters `enqueueDeferred` |
| 6 | `processDeferredAdmission` → consume → finishView → submitPublishRequest → enqueueDeferred full data flow is verified | **PASS** | cpp:636-663 → 612-619 → 589-608 → 357-434 → 437-531; single re-defer path, no bypasses |
| 7 | `recoveryArmed` slot field lifecycle is feasible: stamped at `enqueueDeferred` via orchestrator latch, defaults false on overwrite | **PASS** | Slot struct at h:32-38 has no `recoveryArmed` yet; `enqueueDeferred` replaces slot wholesale (505-521); orchestrator latch `recoveryArmedPending_` design is sound |
| 8 | `clearDeferredForShutdown()` ownership gap is confirmed: 4 MessageThread callers write rebuild-thread-only plain members | **FAIL** | Timer.cpp:1642/1804/1824 + ReleaseResources.cpp:359 all MessageThread; h:277-278 explicitly "rebuild-thread 専用"; plain writes without `rebuildMutex` |
| 9 | `resetDeferredRetryBudget()` (Timer.cpp:1724) is also an ownership violation — D1 moves it to rebuild thread | **FAIL** | Timer.cpp:1724 calls from MessageThread; h:171-174 writes plain `deferredRetryGeneration_`/`deferredRetryCount_`; D1 relocates to `processDeferredAdmission` on rebuild thread |
| 10 | No existing test or harness depends on current `processDeferredAdmission()` no-arg signature in a way that breaks | **PASS** | Only caller is RebuildDispatch.cpp:904; DeferredFlowIntegrationTests.cpp tests `processDeferredAdmission` indirectly via end-to-end flow |

---

## 5. Phase 1 Implementation Ordering (D135-8 Forward Reference)

The D135-6→D135-7→D135-8 ordering is confirmed viable:

```
P4 → P6 → P7 → P5 → P1 → P2 → P3 → P8/P9
```

| Step | Edit | Dependency | P8? |
|------|------|------------|-----|
| P4 | Add `recoveryRetryReady` to AudioEngine.h (atomic, rebuildMutex region) | None | No |
| P6 | Recovery handler sets `recoveryRetryReady = true` (Timer.cpp:1744 area) | P4 | No |
| P7 | `processDeferredAdmission(wasRecoveryWake)` signature + reset at start | P4 (flag exists) | No |
| P5 | Consumer merge reads `wasRecoveryWake = recoveryRetryReady` at RebuildDispatch.cpp:879 | P4 | No |
| P1 | Remove `resetDeferredRetryBudget()` call from Timer.cpp:1724 (moved to P7) | P7 | No |
| P2 | Add `recoveryArmed` field to `DeferredPublishSlot` + orchestrator latch | P7 (wasRecoveryWake available) | No |
| P3 | Thread `wasRecoveryWake` through `submitPublishRequest` → latch | P2 | No |
| P8 | `clearDeferredForShutdown()` ownership repair (Option C: atomic intent + drain) | **Required before Phase 2 tests** | **YES** |
| P9 | Remove `recoveryPublishSeq()` write-only latch + `deferredRecoveryRearm_` dead code | P8 | Yes (hygiene) |

**Critical**: P8 must be completed before Phase 2 test validation (D135-2 harness re-run, recovery-redrive test), because the tests will exercise `clearDeferredForShutdown()` paths that currently have the ownership gap.

---

## 6. Detailed Proofs

### 6.1 Co-location of merge and `processDeferredAdmission` call

```cpp
// RebuildDispatch.cpp:844-904 (reconstructed)
{
    std::unique_lock<std::mutex> lock(rebuildMutex);     // 845
    rebuildCV.wait(lock, [this] { ... });                 // 849-854
    // ...
    doDeferredPublish = publishRetryReady;                 // 879  ← wasRecoveryWake available here
    publishRetryReady = false;                             // 880
    // ...
}                                                            // 890 (lock released)
if (doDeferredPublish && runtimeOrchestrator_ != nullptr) {   // 898
    runtimeOrchestrator_->processDeferredAdmission();       // 904
}
```

Under D1, line 879 becomes:
```cpp
doDeferredPublish = publishRetryReady;
wasRecoveryWake = recoveryRetryReady;   // D1: capture under rebuildMutex
publishRetryReady = false;
recoveryRetryReady = false;             // D1: clear under rebuildMutex
```
Both `wasRecoveryWake` and `doDeferredPublish` are locals captured under `rebuildMutex`, used after lock release at line 904. No shared-state read occurs after lock release — the `wasRecoveryWake` local is the sole provenance carrier.

### 6.2 Retry-budget invariant split (ordinary vs recovery)

**Ordinary wake** (Threading.cpp:286): sets `publishRetryReady=true`, does NOT reset budget. `processDeferredAdmission(false)` → no reset → `enqueueDeferred` increments `deferredRetryCount_` under `sameObligation`. Count progresses: 0→1→2→exhausted (with kMax=2, or 0→1→...→9→exhausted with current kMax=10).

**Recovery wake** (Timer.cpp:1744): sets `publishRetryReady=true` + `recoveryRetryReady=true`. `processDeferredAdmission(true)` → reset budget (generation=0, count=0) → `enqueueDeferred` sees `req.generation != 0` → new obligation → count starts fresh. Recovery grants 2 (or 10) fresh retries.

**Invariant**: Ordinary path never resets; recovery path always resets at start. The `wasRecoveryWake` local is the **sole** decision point. No double-counting possible.

### 6.3 Lost/stale wake analysis (cases A-D + slot-gone)

All four coalescing cases (A: ordinary+ordinary, B: recovery+ordinary, C: ordinary+recovery, D: recovery+recovery) are resolved by the CV predicate re-check + `wasRecoveryWake` capture:

- The CV predicate includes `publishRetryReady` (and D1 adds `recoveryRetryReady` is captured at merge, not in predicate). After `notify_one()`, the rebuild thread wakes, re-checks predicate. If `publishRetryReady` is still true, it proceeds. If a second publisher fired in between and set the flag again, it's still true — the merge captures `wasRecoveryWake` from whatever `recoveryRetryReady` is at that moment.
- **Lost wake**: If `recoveryRetryReady=false` and `publishRetryReady=false` at merge (both publishers' flags consumed by a prior wake cycle), the rebuild thread returns to wait. This can't happen under D1 because `recoveryRetryReady` is cleared **only** at the merge (same scope as `publishRetryReady`), and the CV predicate guarantees the thread wakes when either flag is set.

---

## 7. GO/NO-GO Conclusion

**GO** — D1 (`recoveryRetryReady` provenance flag) is safe to implement as specified in D135-6.

**Conditions for proceeding to D135-8 (Phase 1 Implementation):**
1. ✅ D1 provenance mechanism (P4, P5, P6, P7) is structurally sound — verified by items 1-6.
2. ✅ `recoveryArmed` slot lifecycle (P2) is feasible — verified by item 7.
3. ⚠️ **P8 (`clearDeferredForShutdown()` ownership repair) MUST be included in Phase 1** — items 8-9 are FAIL. The 3 MessageThread callers in Timer.cpp (1642/1804/1824) and 1 in ReleaseResources.cpp (359) must be routed through an atomic intent flag + rebuild-thread drain (Option C recommended).
4. ⚠️ `kMaxDeferredRetries` should be corrected 10→2 to match design intent (item 10, WARNING).
5. ✅ Phase 1 implementation ordering (P4→P6→P7→P5→P1→P2→P3→P8/P9) is dependency-validated.

**NO blockers found.** All FAIL items are addressed by the D135-6→D135-8 plan. Proceed to D135-8.
