# D101-9 Phase 9-B Step 3 — Candidate A Feasibility / Ownership-Backpressure Contract Audit

## 1. Purpose

Audit **Candidate A** (bounded Terminal + `store() == false` → caller retains ownership + backpressure/retry) as a viable design for satisfying `K_terminal < ∞`.

This step does **NOT** implement Candidate A. It determines whether Candidate A can be implemented without violating:
1. P-4 (`store()` ALWAYS succeeds)
2. I4 D15.2 ownership conservation (INV-X1-7)
3. I4 D14.3 (backpressure on capacity exhaustion, NOT terminal-failure/destruction)

**Scope**: Read-only code audit. No code changes.

---

## 2. Step 3-1 — P-4 vs I4 D14.3 Relationship

### 2.1 P-4 statement (current production)

From `ISRRetireRouter.h:74-75`:
> **★ P-4 (15-P-4): entries_ is GROWABLE (std::vector). This guarantees the authority ALWAYS accepts an entry — there is NO "store full" failure path.**

And from `ISRRetireRouter.cpp:267-269`:
```cpp
// ★ P-4: Growable store により常に true → ownership は必ず移転。
//   assert(false) 経路は存在しない（EBR 破綻による L > 0 は構造的に排除）。
result = RetireEnqueueResult::TerminalReclaim;  // Terminal owns ptr ✅
```

### 2.2 I4 D14.3 statement

> **budget 枯渇 → admission BLOCK (backpressure / upstream stall)**

D14.3 requires backpressure on exhaustion, NOT terminal failure / destruction.

### 2.3 The conflict

P-4 guarantees `store()` ALWAYS returns true (no failure path). Candidate A requires `store()` to return false when bounded Terminal is full. This means:

**P-4 must be revised** to allow `store() == false`.

### 2.4 The two options

#### A-1: P-4 maintained at Terminal layer (keep growable Terminal)

```
logical obligation admission
        ↓
reservation
        ↓
bounded upstream stores (D/Q/E)
        ↓
Terminal は unlimited / growable  → P-4 preserved
```

**Result**: Candidate A is NOT possible. Terminal cannot return `false`.

#### A-2: P-4 revised to match I4 D14.3 (bounded Terminal + backpressure)

```
logical obligation admission
        ↓
reservation
        ↓
bounded placement (D/Q/E/T)
        ↓
full
        ↓
stalled / backpressure
```

**Result**: P-4 is revised from "Terminal ALWAYS accepts" to "Terminal is bounded; caller retains on full."

### 2.5 Verdict — Step 3-1

| Option | P-4 preserved? | Candidate A possible? | I4 D14.3 alignment |
|---|---|---|---|
| A-1 (keep growable Terminal) | ✅ Yes | ❌ No | ✅ Backpressure not needed at Terminal |
| A-2 (bounded Terminal) | ❌ No (revised) | ✅ Yes | ✅ Backpressure required |

**Decision: A-2 is the only viable path for Candidate A.** P-4 must be revised from "Terminal authority ALWAYS accepts" to "Terminal is bounded; on full, caller retains ownership and backpressure is applied upstream."

**This is a P-4 invariant change.** P-4 is a `◆H` (high-priority) invariant. Revision requires:
- I4 D14.3/D15.2 contract review (already aligned with backpressure model)
- Documentation update in `ISRRetireRouter.h:74-75`
- All `RetireEnqueueResult` return values must be re-evaluated for ownership semantics

**A3-1 Gate: ✅ — P-4 vs I4 D14.3 relationship is now clearly defined.**
- P-4 must be revised (A-2) for Candidate A to be feasible
- The revision aligns with I4 D14.3's backpressure model
- No code change yet — this is a design decision

---

## 3. Step 3-2 — caller retains ownership as I4 "stalled" state

### 3.1 The I4 D15.2 ownership conservation equation

From `doc/work88/I4_DESIGN_CONTRACT.md` §D15.2:

```
ownership conservation:
    transportCount + durableCount + buildingCount + stalledCount
        + supersededCount + shutdownDiscardCount
        == admittedLogicalObligationCount
```

The `stalled` placement state exists in the conservation equation.

### 3.2 The question: does `caller retains ptr` = `stalled`?

Candidate A proposes:
```
Terminal full → caller retains ptr → backpressure/retry
```

I4 D14.2 (reservation-first model):
```
Transport admission
        ↓
Logical obligation reservation (single budget)
        ↓
placement: Transport / DurablePending / Building / Stalled
```

### 3.3 Proposal A: caller retains = `stalled` placement

If we model "Terminal full → caller retains" as a **`stalled` ownership placement**:

```
Terminal full
    ↓
ptr ∈ caller (stalled state)
    ↓
backpressure → upstream stall
    ↓
retry when space freed
    ↓
DurablePending / Terminal / Success
```

This would mean:
- `stalledCount` in the conservation equation tracks ptrs retained by callers
- `admittedLogicalObligationCount` would need to increment BEFORE the ptr enters D/Q/E/T (at admission, not at enqueue)
- The `stalled` state is a **reservation-backed placement** — the caller holds the obligation but cannot transfer it to durable storage

### 3.4 Proposal B: caller retains is pre-admission (outside conservation)

If "caller retains" is BEFORE admission (before `admittedLogicalObligationCount` increments), then:
- `admittedLogicalObligationCount` must increment at the `admit` gate, not at D/Q/E/T entry
- The ptr is in `transport` reservation but not yet placed
- This requires a **two-phase admission**: reserve → place

### 3.5 Production code check — where does admission happen?

Let me trace the production caller chain:

**Retirement (not recovery) admission path:**

```cpp
// DSPLifetimeManager.cpp:89
const auto result = router_->enqueueWithRetry(
    dsp, &AudioEngine::destroyDSPCoreNode, epoch, DeletionEntryType::Generic);
// caller (DSPLifetimeManager) ignores result:
juce::ignoreUnused(result);
```

```cpp
// AudioEngine.Cache.cpp:16,41
owner.enqueueDeferredDeleteNonRt(map, [](void* p) { delete static_cast<CacheMap*>(p); });
// caller ignores return (enqueueDeferredDeleteNonRt returns bool, ignored)
```

```cpp
// ConvolverProcessor.Lifecycle.cpp:57,70
provider->enqueueDeferredDeleteNonRt(oldState, deleter);
// caller ignores return
```

```cpp
// RuntimePublicationBridge (AudioEngine.h:3536)
void retirePublishedRuntimeWorldNonRt(RuntimePublishWorld* world, bool resetRevision) noexcept
{
    // PRECONDITION: W ∈ PublishedDomain
    engine_->enqueueDeferredDeleteNonRt(world, [](void* p) { ... });
    // caller ignores return
}
```

```cpp
// EQProcessor.Core.cpp:52-68
auto result = m_retireCoordinator->enqueueRetire(...);
result = stackRouter.enqueueWithRetry(...);
return result == Success || result == QueuePressure || result == TerminalReclaim;
// caller returns false only on Shutdown
```

**Key finding:** None of these callers are `recovery obligation` admission paths. They are **retirement** paths — disposing of already-published, already-retired objects. The `stalled`/reservation model in I4 D14.2/D15.2 applies to **recovery obligations** (quarantine recovery), not to deferred deletion retirement.

### 3.6 The recovery obligation admission path (for reference)

```cpp
// ISRRuntimePublicationCoordinator.cpp:855-940
bool RuntimeIntentCoordinator::submitRecoveryRequest(...) {
    // Gate: shutdown → discard (ShutdownDiscard)
    if (state_ == ShuttingDown) { recoveryShutdownDiscardCount_++; return false; }

    // Reservation-before-push
    convo::fetchAddAtomic(pendingIntentCount_, 1, ...);   // ← admission increment

    if (recoveryIntentQueue_.push(intent)) {
        return true;  // Transport placement
    }

    // Queue full → DurablePending
    convo::fetchSubAtomic(pendingIntentCount_, 1, ...);
    convo::fetchAddAtomic(recoveryIntentDropCount_, 1, ...);
    pendingRecoveryAdmission_.state = State::DurablePending;
    pendingRecoveryAdmission_.reservationOwned = true;
    convo::publishAtomic(recoveryAdmissionPending_, true, ...);
    return true;  // DurablePending placement
}
```

This path DOES have a formal reservation + placement model. But the **retirement path** (DeferredDeletionQueue/DSPHandle retirement) does NOT — it uses `enqueueWithRetry()` directly.

### 3.7 Verdict

**Proposal A (caller retains = `stalled`) is NOT directly applicable** to the retirement path. The `stalled` ownership state in I4 D15.2 applies to **recovery obligations**, not to retired DSPHandle/World retirement.

For Candidate A to model "Terminal full → caller retains" as a `stalled` placement, it would need to:
1. Define a new ownership state for the retirement pipeline (not in I4 D15.2's 6-state model)
2. Add this state to the conservation equation
3. Track it as a new counter

This would require **I4 D15.2 revision** — adding an `"inFlightRetirement"` state to the conservation equation.

**A3-2 Gate: ⚠️ CONDITIONAL**
- Proposal A (stalled) requires I4 D15.2 revision for the retirement path
- Proposal B (pre-admission) requires admission timing re-definition
- Neither is cleanly supported by current I4 D15.2 (which targets recovery obligations, not retirement)

**Recommendation:** Model `caller retains` as a **new I4 ownership state** — `retiredPendingTerminal` — that sits between `transport` (in-flight) and `durable` (placed). This requires I4 D15.2 revision but maintains conservation.

---

## 4. Step 3-3 — Full Production Caller Audit

### 4.1 Call chain overview

```
[Caller] → enqueueDeferredDeleteNonRtWithResult() → enqueueWithRetry() → [D → Q → E → Terminal]
    ↑              ↑                              ↑           ↑
  Non-RT      AudioEngine.h                  ISRRetireRouter.cpp   DeferredDeletionQueue.h
```

### 4.2 All production callers of `enqueueWithRetry()`

Each must be analyzed for: what happens on `store() == false` (Candidate A).

| # | Caller | File:Line | Ownership before | Return semantics | Ownership after failure (Candidate A) | Retry owner | Current behavior on failure |
|---|---|---|---|---|---|---|---|
| 1 | `ISRRetireRouter::retire()` | ISRRetireRouter.cpp:272-280 | caller | void | **UNDEFINED** — `enqueueWithRetry` return ignored | NONE | Logs "Future: RuntimeHealthMonitor" (no retry) |
| 2 | `ISRRetireRouter::enqueueRetire()` (enum) | ISRRetireRouter.cpp:218-251 | caller (D queue) | `RetireEnqueueResult` | **Downshift to Q/E/T via retry loop inside enqueueWithRetry** | enqueueWithRetry internal | tryReclaim (500ms cooldown), then overflow → QueuePressure |
| 3 | `ISRRetireRouter::enqueueRetire()` (bool) | ISRRetireRouter.cpp:254-257 | caller (D queue) | bool | **Same as #2** | enqueueWithRetry internal | Delegates to enum version |
| 4 | `RuntimeIntentCoordinator::enqueueRetire()` | ISRRuntimePublicationCoordinator.cpp:150-174 | caller (Coordinator) | `RetireEnqueueResult` | **UNDEFINED** — `enqueueWithRetry` return ignored after Success check | NONE | No retry; returns result to caller |
| 5 | `DSPLifetimeManager::retireDSPCoreNode()` | DSPLifetimeManager.cpp:89 | caller (DSP lifetime mgr) | void | **UNDEFINED** — `juce::ignoreUnused(result)` | NONE | No retry; ignores result |
| 6 | `DSPLifetimeManager::retireByHandle()` | DSPLifetimeManager.cpp:101-115 | caller (DSP lifetime mgr) | void | **UNDEFINED** — `juce::ignoreUnused(result)` | NONE | No retry; ignores result |
| 7 | `EQProcessor::enqueueDeferredDeleteWithFallback()` | EQProcessor.Core.cpp:45-68 | caller (EQ processor) | bool | **Drops ptr** — returns `false` on Shutdown only; on QueuePressure/TerminalReclaim returns `true` (assumes ownership transferred) | NONE | Returns false ONLY on Shutdown; on QueuePressure/TerminalReclaim → caller treats as Success |
| 8 | `AudioEngine.Cache.cpp:tryEnqueueDeferredMap()` | AudioEngine.Cache.cpp:13-17 | caller (EQ CacheManager) | void | **UNDEFINED** — `enqueueDeferredDeleteNonRt` return ignored | NONE | No retry; ignores result |
| 9 | `AudioEngine.Cache.cpp:storeNewMap()` | AudioEngine.Cache.cpp:39-42 | caller (EQ CacheManager) | void | **UNDEFINED** — `enqueueDeferredDeleteNonRt` return ignored | NONE | No retry; ignores result |
| 10 | `ConvolverProcessor.StereoConvolver::applyIR` | ConvolverProcessor.Lifecycle.cpp:57 | caller (convolver lifecycle) | void | **UNDEFINED** — `enqueueDeferredDeleteNonRt` return ignored | NONE | No retry; ignores result |
| 11 | `ConvolverProcessor::retireStereoConvolver()` | ConvolverProcessor.Lifecycle.cpp:70 | caller (convolver lifecycle) | void | **UNDEFINED** — `enqueueDeferredDeleteNonRt` return ignored | NONE | No retry; ignores result |
| 12 | `enqueueDeferredDeleteNonRtWithResult()` | AudioEngine.h:4208-4236 | caller (any) | `RetireEnqueueResult` | **Shutdown path: return Shutdown**; **Normal path: return result of enqueueWithRetry** | NONE (returns to caller) | Best-effort drain; returns result |
| 13 | `SnapshotCoordinator::resetFadeStateAndRetireTarget()` | SnapshotCoordinator.cpp:92-95 | caller (snapshot coordinator) | void | **Drops ptr** — `enqueueRetire` (bool) return ignored; if D full → ptr NOT retired | NONE | No retry; ignores result |
| 14 | `RuntimePublicationBridge::retirePublishedRuntimeWorldNonRt()` | AudioEngine.h:3536-3540 | caller (bridge) | void | **UNDEFINED** — `enqueueDeferredDeleteNonRt` return ignored | NONE | No retry; ignores result |

### 4.3 Ownership transfer analysis

**Critical finding:** Under Candidate A, if `store()` returns `false`, the ownership transfer contract breaks at **multiple call sites**:

1. **`enqueueWithRetry()` itself (line 345)**: `(void)tstored;` ignores the return. If `store()` returns false, `result` is still set to `TerminalReclaim` — **caller believes ownership transferred when it hasn't**. This is the `tstored` latent bug (see §4.4 below).

2. **`DSPLifetimeManager` (callers #5, #6)**: `juce::ignoreUnused(result)` — if `enqueueWithRetry` returns `TerminalPressure` (new enum needed), the ptr is **NOT transferred** and the caller has **no retry mechanism**. **Leak**.

3. **`EQProcessor` (caller #7)**: Returns `true` for `QueuePressure`/`TerminalReclaim`. Under Candidate A, if `store()` returns false and `result` is changed to a new `TerminalPressure` value, the caller would still return `true` — **believing ownership transferred**. **Leak**.

4. **`AudioEngine.Cache.cpp` (callers #8, #9)**: Return type is void — **ptr silently dropped** if `enqueueDeferredDeleteNonRt` returns false.

5. **`ConvolverProcessor` (callers #10, #11)**: Same — void return, **ptr silently dropped**.

6. **`RuntimePublicationBridge` (caller #14)**: `retirePublishedRuntimeWorldNonRt` calls `enqueueDeferredDeleteNonRt` (void return) — **ptr silently dropped**.

### 4.4 The `tstored` latent bug — formal verification

```cpp
// ISRRetireRouter.cpp:338-348
const bool tstored = terminalReclaim(ptr, deleter, epoch, type,
                                     "enqueueWithRetry:TerminalReclaim");
(void)tstored;  // ★ P-4: 常に true（growable store）
result = RetireEnqueueResult::TerminalReclaim;  // Terminal owns ptr ✅
```

**Current state (P-4 growable):** `tstored` is always `true` → `result = TerminalReclaim` is correct. No bug.

**Candidate A (bounded store):** If `store()` returns `false`:
- `tstored = false` (Terminal did NOT accept)
- `result = TerminalReclaim` (incorrectly says Terminal owns ptr)
- Caller believes ownership transferred → **ptr is leaked** (no one owns it)

**Classification: `latent bug` (not current bug).** This bug is **activated only if** `store()` is changed to return false. Under current P-4 (growable), it is dormant.

### 4.5 Verdict — Step 3-3

| Gate | Status | Finding |
|---|---|---|
| **A3-3**: All production callers ownership flow tracked | ✅ PASS | 14 callers identified and analyzed |
| **A3-3**: Ownership after failure is unambiguous | ❌ FAIL (Candidate A unviable as-is) | 7 callers would leak ptr on `store() == false` |
| **A3-3**: Retry owner identified per caller | ❌ FAIL | No caller has retry/backpressure logic |

**7 of 14 callers would leak ptr under Candidate A** without:
1. Fixing the `tstored` ignore bug
2. Adding ownership return semantics to void-return callers
3. Adding retry/backpressure to all 7 callers

---

## 5. Step 3-4 — `tstored` Latent Bug Formal Verification

### 5.1 Current code trace

```cpp
// ISRRetireRouter.cpp:282-360 (enqueueWithRetry, Stage 5)
const bool tstored = terminalReclaim(ptr, deleter, epoch, type,
                                     "enqueueWithRetry:TerminalReclaim");
(void)tstored;  // ★ P-4: 常に true（growable store）
result = RetireEnqueueResult::TerminalReclaim;  // Terminal owns ptr ✅
signalDrainWakeup();
return result;
```

```cpp
// ISRRetireRouter.cpp:490-511 (terminalReclaim)
bool ISRRetireRouter::terminalReclaim(void* ptr, void (*deleter)(void*), uint64_t epoch,
                                      DeletionEntryType type, const char* reason) noexcept
{
    const uint64_t minReader = minReaderEpoch();
    const bool epochSafe = ISRRetireRouter::isOlder(epoch, minReader);
    const bool isRt = convo::numeric_policy::isAudioThread();

    if (epochSafe && !isRt) {
        deleter(ptr);                    // Case A: synchronous destruction
        if (type == DeletionEntryType::World) m_terminalReclaim.recordWorldReclaim();
        return true;                    // ← always true (ptr destroyed, not stored)
    }
    // Case B/C/D: store to Terminal (currently always succeeds, growable)
    return m_terminalReclaim.store(ptr, deleter, epoch, type, reason);
    // store() returns true always (P-4 growable: push_back always succeeds)
}
```

```cpp
// ISRRetireRouter.cpp:27-53 (store)
bool TerminalReclaimAuthority::store(void* ptr, void (*deleter)(void*), uint64_t epoch,
                                     DeletionEntryType type, const char* reason) noexcept
{
    if (ptr == nullptr || deleter == nullptr)
        return true;  // no-op は成功扱い

    std::lock_guard<std::mutex> lock(mtx_);
    entries_.push_back(Entry{ptr, deleter, epoch, type, reason});
    residentAtomic_.fetch_add(1, std::memory_order_release);
    return true;  // ★ P-4: ALWAYS true — growable std::vector, no capacity check
}
```

### 5.2 Latent bug analysis

**Current (P-4 growable):**
- `store()` → `push_back()` → always succeeds → returns `true`
- `terminalReclaim()` returns `true` (either destroyed or stored)
- `tstored = true` → `result = TerminalReclaim` → caller believes ownership transferred → ✅ Correct

**Under Candidate A (bounded):**
- `store()` would need to check capacity: `if (entries_.size() >= K_terminal) return false;`
- If full: `store()` returns `false`
- `terminalReclaim()` returns `false` (Case B/C/D only — Case A always returns true since deleter runs)
- `tstored = false`
- `(void)tstored;` — **ignores the false**
- `result = TerminalReclaim` — **incorrectly says Terminal owns ptr**
- `signalDrainWakeup()` — signals drain, but Terminal is full and can't accept new entries
- Caller receives `TerminalReclaim` → believes ownership transferred → **ptr LEAKED**

### 5.3 The `RetireEnqueueResult` semantics issue

```cpp
// ISRAuthorityClass.h:28-34
enum class RetireEnqueueResult : std::uint8_t {
    Success,
    QueuePressure,
    QueueFull,
    Shutdown,
    TerminalReclaim  // ← means "Terminal owns ptr"
};
```

**Problem:** `TerminalReclaim` means "ownership transferred to Terminal." There is **no enum value for "Terminal full, caller retains"** under Candidate A. The enum would need a new value (e.g., `TerminalPressure`) to represent this case.

### 5.4 Verdict

| Criterion | Status |
|---|---|
| Current `tstored` is a live bug | ❌ No — only latent under Candidate A |
| `tstored` becomes bug under Candidate A | ✅ Yes — ownership state misreported |
| Fix requires `RetireEnqueueResult` revision | ✅ Yes — need new `TerminalPressure` state |
| Fix requires all 14 callers to handle new state | ✅ Yes — 7 would leak without updates |

**A3-5 Gate: ✅ — `tstored` classified as "latent bug activated by bounded Terminal"**

---

## 6. Step 3-5 — `RetireEnqueueResult` State Machine

### 6.1 Current enum

```cpp
// ISRAuthorityClass.h:28-34
enum class RetireEnqueueResult : std::uint8_t {
    Success,        // ptr successfully enqueued (D queue)
    QueuePressure,  // D queue full → escalated to Q/E/T (ptr transferred)
    QueueFull,      // D queue full (rare — retry loop handles)
    Shutdown,       // shutdown in progress → caller retains ownership
    TerminalReclaim // ptr transferred to TerminalReclaimAuthority
};
```

### 6.2 Current ownership semantics per return value

| Return value | Ownership transferred? | Caller may retry? | Caller must retain? | Caller must stop? |
|---|---|---|---|---|
| Success | ✅ D queue owns | N/A (already placed) | ❌ No | ❌ No |
| QueuePressure | ✅ Q/E/T owns (via retry) | N/A (already placed) | ❌ No | ❌ No |
| QueueFull | ❌ NOT transferred | ✅ Yes (retry in enqueueWithRetry) | ⚠️ Temporarily | ❌ No |
| Shutdown | ❌ Caller retains | ❌ No | ✅ Yes | ✅ Yes (shutdown) |
| TerminalReclaim | ✅ Terminal owns | N/A (Terminal absorbs) | ❌ No | ❌ No |

### 6.3 Candidate A requires new state

Under Candidate A, when `store()` returns `false`:
- Ownership is **NOT transferred** (Terminal is full)
- Caller **retains** ownership
- Caller **must retry** (backpressure)
- This is **NOT Shutdown** (system is still running)

**New enum value needed:**
```cpp
enum class RetireEnqueueResult : std::uint8_t {
    Success,
    QueuePressure,
    QueueFull,
    Shutdown,
    TerminalReclaim,        // ✅ Terminal owns ptr (current)
    TerminalPressure,       // ❌ Terminal full — caller retains (Candidate A)
};
```

### 6.4 Revised state machine under Candidate A

| Return value | Ownership transferred? | Caller may retry? | Caller must retain? | Caller must stop? |
|---|---|---|---|---|
| Success | ✅ D queue owns | N/A | ❌ No | ❌ No |
| QueuePressure | ✅ Q/E/T owns | N/A | ❌ No | ❌ No |
| QueueFull | ❌ NOT transferred | ✅ Yes (internal retry) | ⚠️ Temporarily | ❌ No |
| Shutdown | ❌ Caller retains | ❌ No | ✅ Yes | ✅ Yes (shutdown) |
| TerminalReclaim | ✅ Terminal owns | N/A | ❌ No | ❌ No |
| **TerminalPressure** | ❌ **Terminal full — NOT transferred** | ✅ **Yes (backpressure)** | ✅ **Yes** | ❌ No |

### 6.5 Impact on callers

Under Candidate A, `TerminalReclaim` and `TerminalPressure` must be **distinguished**:

| Caller | Must handle TerminalPressure? | Current behavior on TerminalReclaim | Risk under Candidate A |
|---|---|---|---|
| `enqueueWithRetry()` | ✅ CRITICAL | Ignores `tstored`, returns `TerminalReclaim` | Leaks if `TerminalPressure` misreported as `TerminalReclaim` |
| `ISRRetireRouter::retire()` | ✅ Required | Ignores result entirely | Leaks (no ownership tracking) |
| `DSPLifetimeManager` (#5, #6) | ✅ Required | `juce::ignoreUnused(result)` | Leaks |
| `EQProcessor` (#7) | ✅ Required | Returns true for TerminalReclaim | Leaks (treats as Success) |
| `AudioEngine.Cache.cpp` (#8, #9) | ✅ Required | void return | Leaks (ptr dropped) |
| `ConvolverProcessor` (#10, #11) | ✅ Required | void return | Leaks (ptr dropped) |
| `RuntimePublicationBridge` (#14) | ✅ Required | void return (via enqueueDeferredDeleteNonRt) | Leaks (ptr dropped) |
| `SnapshotCoordinator.cpp` (#13) | ✅ Required | bool return, ignores | Leaks (ptr dropped) |

### 6.6 Verdict

**A3-3 Gate: ✅ — RetireEnqueueResult ownership semantics can be made unambiguous**

A new `TerminalPressure` enum value is needed to distinguish "Terminal owns ptr" from "Terminal full, caller retains." However, **all 12 remaining callers** must be updated to handle this new state. This is a **high-impact change** affecting the entire retirement pipeline.

---

## 7. Step 3-6 — Retry Authority

### 7.1 Current retry architecture

`enqueueWithRetry()` (ISRRetireRouter.cpp:282-360) contains the ONLY retry logic:

```cpp
// Stage 1: D queue (enqueueRetire with 500ms cooldown tryReclaim)
auto result = enqueueRetire(ptr, deleter, epoch, type);
if (result == RetireEnqueueResult::Success) return result;

// Stage 2: Retry loop (kMaxRetry=2)
for (int attempt = 0; attempt < kMaxRetry; ++attempt) {
    provider_->tryReclaim();
    drainEmergencyAndTerminal();
    result = enqueueRetire(ptr, deleter, epoch, type);
    if (result == RetireEnqueueResult::Success) return result;
}

// Stage 3-5: Q → E → Terminal (no retry — always succeeds under P-4)
```

**No retry exists after Stage 5.** Under P-4 (growable Terminal), there's no need — `store()` always succeeds.

### 7.2 Candidate A retry options

#### Option A: Internal retry in ISRRetireRouter

```
ISRRetireRouter
    └─ retry internally (wait for drain, retry store)
```

**Problems:**
- No wakeup mechanism for internal retry (drain only happens on CoordinatorLoop 1ms poll)
- Busy-wait / blocking inside Non-RT thread → stall CoordinatorLoop
- No timeout / backpressure signaling to upper layers
- Violates I4 D14.3 (backpressure should propagate upstream, not stall internally)

#### Option B: Retry by Coordinator/owner

```
ISRRetireRouter
    └─ return Backpressure (TerminalPressure)
             ↓
Coordinator / owner
    └─ retains obligation
             ↓
wake when space freed
             ↓
retry
```

**Alignment:** This matches I4 D14.3 exactly — backpressure propagates to the admission authority (Coordinator), which holds the obligation and retries when capacity is available.

**Evidence from I4 D14.3:**
> CoordinatorLoop（単一 producer）が in-flight quarantine intent の義務を保持し、budget 解放（terminal disposition）後に再試行。

#### Option C: Stalled queue in ISRRetireRouter

```
ISRRetireRouter
    └─ Stalled queue (ptr parked, pending retry)
             ↓
Coordinator drains/retries stalled queue
```

This is a hybrid: the Router holds the ptr in a stalled queue, Coordinator periodically retries.

**Problem:** This adds a new internal state to ISRRetireRouter without I4 D15.2 modeling. The `stalled` placement in D15.2 is at the **recovery obligation** layer (RuntimeIntentCoordinator), not at the **pointer retirement** layer (ISRRetireRouter).

### 7.3 The architectural mismatch

**Critical finding:** The I4 D14.3 backpressure model applies to **recovery obligations** (DSPHandle quarantine recovery via `submitRecoveryRequest`). The **pointer retirement** path (`enqueueWithRetry` / `enqueueDeferredDeleteNonRt`) is a **different subsystem** — it retires already-published, already-recovered objects (old RuntimeWorlds, old EQState, old IRState).

The recovery obligation model has:
- Formal reservation (`pendingIntentCount_` fetchAdd)
- Placement states (Transport / DurablePending / Building)
- Backpressure (queue full → DurablePending)
- Wakeup (Builder loop consumes)
- Shutdown discard tracking

The retirement path has:
- No formal reservation
- No placement state tracking
- No backpressure (P-4: always succeeds)
- Best-effort drain (CoordinatorLoop 1ms poll)
- No ownership state on failure (only Shutdown)

### 7.4 Verdict

**A3-6 Gate: ⚠️ CONDITIONAL**

- **Correct retry authority = Option B (Coordinator/owner)** — aligns with I4 D14.3
- **BUT:** This requires retrofitting the entire retirement path with I4-style reservation + placement + backpressure — essentially porting the recovery obligation model to pointer retirement
- **Option A (internal retry)** is architecturally wrong (violates backpressure propagation)
- **Option C (stalled queue)** adds unmodeled state without I4 revision

**Recommendation:** If Candidate A is pursued, Option B is the only I4-compliant approach. This requires:
1. New ownership state in ISRRetireRouter for `TerminalPressure`
2. CoordinatorLoop owns the retry obligation (not ISRRetireRouter)
3. Ownership conservation equation revision (new `retiredPendingTerminal` state)
4. Wakeup signal: Terminal full → Coordinator wakes → retries when drain frees space

---

## 8. Step 3-7 — Wakeup Semantics for Terminal Full

### 8.1 Current wakeup (Terminal NOT full)

```cpp
// ISRRetireRouter.cpp:352-353
signalDrainWakeup();
return result;
```

`signalDrainWakeup()` (ISRRetireRouter.cpp:370-380):
```cpp
void ISRRetireRouter::signalDrainWakeup() noexcept {
    std::lock_guard<std::mutex> lock(drainCvMtx_);
    drainCv_.notify_one();  // Wake CoordinatorLoop
}
```

**Purpose:** Signal CoordinatorLoop that Q/E/T has new entries → wake from `waitForDrainSignalOrTimeout(1ms)`.

**What drains:** CoordinatorLoop → `drainDeferredRetireQueues(false)` → `tryReclaim()` → `drainQuarantineStore()` → epoch-gated drain.

### 8.2 Current wakeup flow on Q/E/T entry

```
ptr enters Q/E/T
    ↓
signalDrainWakeup()
    ↓
CoordinatorLoop wakes (drainCv_ ← notify_one)
    ↓
CoordinatorLoop::run() → engine_.runCoordinatorPhase()
    ↓
drainDeferredRetireQueues(false) → tryReclaim() → drain()
    ↓
Epoch-gated: if minReaderEpoch advances, deleter runs
    ↓
ptr destroyed (ownership gone)
```

### 8.3 The Terminal-full wakeup problem

When Terminal is full (Candidate A), the ptr is in `caller`. The wakeup chain becomes:

```
ptr ∈ caller (TerminalPressure)
    ↓
signalDrainWakeup()  ← but Terminal can't accept ptr; caller still owns it
    ↓
CoordinatorLoop wakes
    ↓
drain()  ← drains epoch-safe entries in Terminal (makes space)
    ↓
who retries the CALLER's ptr?
    ↑
    ← NO ONE currently
```

**The gap:** `signalDrainWakeup()` wakes the Coordinator to drain Terminal. But the **caller's ptr** (now in `caller` ownership) is NOT in Terminal. After drain frees space, **nothing retries the caller's ptr**.

### 8.4 Required wakeup redesign

For Candidate A, the wakeup must close the loop:

```
Terminal full
    ↓
caller retains ptr (TerminalPressure)
    ↓
signalDrainWakeup()  ← signal Coordinator
    ↓
CoordinatorLoop: drain Terminal (epoch-gated) → frees space
    ↓
CoordinatorLoop: signal retry  ← NEW: retry signal needed
    ↓
caller wakes (via retry mechanism)
    ↓
caller: retry enqueueWithRetry() → store() succeeds
    ↓
ptr ∈ Terminal → normal drain cycle
```

**Problem:** The current architecture has **no upward retry signal** from ISRRetireRouter → caller. The `signalDrainWakeup()` only signals downward (CoordinatorLoop → drain). There is no mechanism for "Terminal has space, retry your retained ptrs."

### 8.5 What the wakeup should be

From I4 D14.3:
> CoordinatorLoop（単一 producer）が in-flight quarantine intent の義務を保持し、budget 解放（terminal disposition）後に再試行。

The wakeup must:
1. Signal drain (existing: `signalDrainWakeup`)
2. After drain frees space, signal retry to obligation holder (Coordinator)
3. Coordinator retries the `TerminalPressure` obligation

### 8.6 Verdict

**A3-7 Gate: ⚠️ CONDITIONAL**

The wakeup data flow can be designed:
```
TerminalFull → caller retains → signalDrainWakeup → Coordinator drains → space freed → Coordinator retries obligation
```

BUT this requires:
1. A **retry obligation queue** at the Coordinator layer (not in ISRRetireRouter)
2. A **retry signal** after drain (currently only drain signal exists)
3. **Ownership state** to track which ptrs are in `TerminalPressure` state

This is **not a simple addition** — it requires architectural changes to the Coordinator's responsibility model.

---

## 9. Step 3-8 — K_terminal Relationship to Reservation Budget

### 9.1 The I4 reservation budget model

From I4 D14.2/D14.3:
```
kMaxLogicalRecoveryObligations = 32  (I4 D22.2)
```

Budget model:
```
reservation (from kMaxLogicalRecoveryObligations=32)
    ↓
placement: Transport / DurablePending / Building / Stalled
    ↓
sum ≤ 32 (conservation)
```

### 9.2 K_terminal in the reservation model

The reservation budget (32 logical obligations) is for **recovery obligations** (DSPHandle quarantine recovery). The **pointer retirement** path (DeferredDeletionQueue/DSPHandle retirement) uses entirely different capacity:

| Storage | Type | Capacity | Applies to |
|---|---|---|---|
| D (DeferredDeletionQueue) | lock-free ring | `kQueueSize = 4096` | Pointer retirement |
| Q (RetireQuarantineStore) | std::array | `kMaxQuarantinedEntries = 512` | Pointer retirement |
| E (EmergencyQuarantineStore) | std::array | `kMaxQuarantinedEntries = 512` | Pointer retirement |
| T (TerminalReclaimAuthority) | std::vector | UNBOUNDED (growable) | Pointer retirement |
| Recovery obligation budget | — | `kMaxLogicalRecoveryObligations = 32` | Recovery obligations |

**These are SEPARATE budgeting systems.** The `kMaxLogicalRecoveryObligations = 32` budget does NOT govern the D/Q/E/T terminal retirement chain. They serve different purposes:

- **Recovery obligations** (32 budget): DSPHandle quarantine recovery requests, each requiring rebuild + re-admit
- **Terminal retirement** (5120 bounded + growable Terminal): deferred pointer deletion, purely memory management

### 9.3 The K_terminal sizing problem remains

K_terminal = 2048 (Step 2 proposal) or 4096 (Step 3-8 recommendation) is **not derived from the reservation budget** — it's an independent capacity decision for the pointer retirement chain.

The relationship between K_terminal and `kMaxLogicalRecoveryObligations = 32` is:
- **No direct relationship** — they operate on different object types (recovery obligations vs retired pointers)
- **Indirect relationship**: Each recovery obligation (quarantined DSPHandle) eventually flows through the retirement path when the rebuilt DSP replaces it → 1 recovery obligation can generate 1 retired pointer

But the **budget conservation** (32 ≤ kMaxLogicalRecoveryObligations) is about **live recovery obligations**, while K_terminal is about **retired pointer absorption capacity**.

### 9.4 Where K_terminal should be sized

K_terminal should be sized for **worst-case retired-pointer backlog**, which depends on:

1. **Grace lifetime**: How long retired ptrs wait in D/Q/E/T before epoch-safe drain
2. **Publish rate**: How fast new ptrs are queued for retirement (bounded by T_build)
3. **Drain rate**: How fast the Coordinator drains (1ms poll + event-driven)

```
K_terminal ≥ N_retired_max × sizeof(Entry)
```

Where `N_retired_max = ceil(grace_lifetime_max / T_build_min)` (from Phase 9-C proof).

Since `N_retired_max` is measurement-gated (Phase 9-C), **K_terminal cannot be decided without telemetry**.

### 9.5 Verdict

**A3-8 Gate: ✅ — K_terminal determination method is defined**

K_terminal should NOT be decided as an arbitrary constant (2048 or 4096). Instead:

```
K_terminal = f(grace_lifetime_max, T_build_min, drain_cycle, sizeof(Entry))
          = ceil(grace_lifetime_max / T_build_min) × sizeof(Entry)
          + safety margin for burst
```

Where:
- `grace_lifetime_max` = bounded by audio callback duration + drain cycle + T_stall (5s)
- `T_build_min` = measured build cycle time (Builder serialization)
- `sizeof(Entry)` ≈ 40 bytes (ptr + deleter + epoch + type + reason)
- Safety margin = 2× for burst tolerance

**K_terminal remains UNDECIDED** — the method is defined but the value requires Phase I-T1 telemetry.

---

## 10. Full Production Caller Ownership Flow

### 10.1 Master table — all 14 production callers

| # | Caller | File:Line | Ownership before call | Entry path | Ownership on `store()==false` (Candidate A) | Current behavior | Risk under Candidate A |
|---|---|---|---|---|---|---|---|
| 1 | `ISRRetireRouter::retire()` | .cpp:272-280 | caller | `enqueueWithRetry` | ptr ∈ caller (UNTRACKED) | `jassert`/log only | Leak — no retry, no ownership state |
| 2 | `ISRRetireRouter::enqueueRetire()` (bool) | .cpp:254-257 | caller (D) | `enqueueRetire` → D ring | ptr ∈ caller (if D full + retry exhausts) | Delegates to enum version | Risk if D ring saturates |
| 3 | `RuntimeIntentCoordinator::enqueueRetire()` | ISRRuntimePublicationCoordinator.cpp:150-174 | caller (Coordinator) | `enqueueWithRetry` | ptr ∈ caller (UNTRACKED) | Returns result, caller ignores non-Success | Leak for non-Success results |
| 4 | `DSPLifetimeManager::retireDSPCoreNode()` | DSPLifetimeManager.cpp:89 | caller (DSP mgr) | `enqueueWithRetry` | ptr ∈ caller (UNTRACKED) | `juce::ignoreUnused(result)` | Leak — result ignored |
| 5 | `DSPLifetimeManager::retireByHandle()` | DSPLifetimeManager.cpp:101-115 | caller (DSP mgr) | `enqueueWithRetry` | ptr ∈ caller (UNTRACKED) | `juce::ignoreUnused(result)` | Leak — result ignored |
| 6 | `EQProcessor::enqueueDeferredDeleteWithFallback()` | EQProcessor.Core.cpp:45-68 | caller (EQ proc) | `enqueueRetire` / `enqueueWithRetry` | ptr ∈ caller (if D full + retry exhausts) | Returns false only on Shutdown; true otherwise | Leak — treats TerminalPressure as Success |
| 7 | `AudioEngine.Cache.tryEnqueueDeferredMap()` | AudioEngine.Cache.cpp:13-17 | caller (Cache mgr) | `enqueueDeferredDeleteNonRt` | ptr ∈ caller (UNTRACKED) | void return — no tracking | Leak — ptr dropped silently |
| 8 | `AudioEngine.Cache.storeNewMap()` | AudioEngine.Cache.cpp:39-42 | caller (Cache mgr) | `enqueueDeferredDeleteNonRt` | ptr ∈ caller (UNTRACKED) | void return — no tracking | Leak — ptr dropped silently |
| 9 | `ConvolverProcessor.applyIR()` | ConvolverProcessor.Lifecycle.cpp:57 | caller (convolver) | `enqueueDeferredDeleteNonRt` | ptr ∈ caller (UNTRACKED) | void return — no tracking | Leak — ptr dropped silently |
| 10 | `ConvolverProcessor.retireStereoConviler()` | ConvolverProcessor.Lifecycle.cpp:70 | caller (convolver) | `enqueueDeferredDeleteNonRt` | ptr ∈ caller (UNTRACKED) | void return — no tracking | Leak — ptr dropped silently |
| 11 | `RuntimePublicationBridge::retirePublishedRuntimeWorldNonRt()` | AudioEngine.h:3536-3540 | caller (bridge) | `enqueueDeferredDeleteNonRt` | ptr ∈ caller (UNTRACKED) | void return — no tracking | Leak — ptr dropped silently |
| 12 | `enqueueDeferredDeleteNonRtWithResult()` | AudioEngine.h:4208-4236 | caller (any) | `enqueueWithRetry` / `shutdownReclaim` | ptr ∈ caller (Shutdown state) | Returns result | Propagates correctly for Shutdown |
| 13 | `SnapshotCoordinator::resetFadeStateAndRetireTarget()` | SnapshotCoordinator.cpp:92-95 | caller (snapshot) | `enqueueRetire` (bool) | ptr ∈ caller (if D full) | bool return ignored | Leak — ptr dropped silently |
| 14 | `ISRRetireRouter::enqueueWithRetry()` internal | ISRRetireRouter.cpp:345-348 | internal (after D/Q/E full) | `terminalReclaim` | ptr ∈ caller (if store() fails) | `(void)tstored;` — ignores failure | **Latent bug** (§5.4) |

### 10.2 Summary

- **3 callers** (1, 4, 5) completely ignore `enqueueWithRetry` / `enqueueWithRetry` return values → **guaranteed leak** under Candidate A
- **4 callers** (7, 8, 9, 10, 11, 13) use void-return or bool-return APIs that **silently drop** ptrs → **guaranteed leak**
- **1 caller** (6) (`EQProcessor`) returns `true` for `TerminalReclaim` — would need to distinguish `TerminalPressure` → returns `false` on pressure
- **1 caller** (12) (`enqueueDeferredDeleteNonRtWithResult`) correctly propagates `Shutdown` but would need to propagate `TerminalPressure` too
- **1 caller** (3) returns result but caller (`DSPLifetimeManager::retire`) ignores non-Success results
- **1 caller** (2) is internal to ISRRetireRouter (the `enqueueRetire` bool wrapper)

**7 distinct callers across 7 files** would require code changes to handle `TerminalPressure` ownership return.

---

## 11. Step 3-8 Extended — K_terminal vs I4 Budget Separation

### 11.1 The two-tier budget model

The I4 contract establishes **two separate budgeting systems**:

**Tier 1: Recovery Obligations** (I4 D14.2/D22.2)
```
kMaxLogicalRecoveryObligations = 32
    ↓
reservation-first: admit ≤ 32 recovery obligations
    ↓
placement: Transport / DurablePending / Building / Stalled
    ↓
backpressure: admission BLOCK on exhaustion (INV-X1-2: queue full ≠ loss)
```

**Tier 2: Pointer Retirement** (ISR EBR pipeline)
```
D(4096) → Q(512) → E(512) → Terminal(growable)
    ↓
capacity-based escalation (no reservation)
    ↓
P-4: Terminal always accepts (no backpressure)
    ↓
drain: epoch-gated (minReaderEpoch advance)
```

### 11.2 Separation verification

| Property | Tier 1 (Recovery) | Tier 2 (Retirement) |
|---|---|---|
| Budget constant | `kMaxLogicalRecoveryObligations = 32` | D+Q+E = 5120, Terminal = growable |
| Admission control | Reservation-first (fetchAdd before push) | None (best-effort enqueue) |
| Backpressure | Yes (queue full → DurablePending) | No (P-4: always accepts) |
| Ownership states | Transport/DurablePending/Building/Stalled | D→Q→E→Terminal cascade |
| Drain trigger | Builder Loop (takePendingRecoveryAdmission) | CoordinatorLoop (1ms poll) |
| T_stall | Not applicable (recovery retry) | `maxRetireWallClockMs_ = 5000.0` |
| Conservation | I4 D15.2 equation (6 states) | Not modeled (P-4: no loss possible) |

**They are architecturally SEPARATE.** K_terminal does NOT participate in the `kMaxLogicalRecoveryObligations = 32` budget. It is part of the pointer retirement capacity chain.

### 11.3 Why this matters for Candidate A

Candidate A attempts to add **backpressure at the Terminal layer** (Tier 2). But I4 D14.3's backpressure model is at the **recovery obligation layer** (Tier 1). These are different layers:

- Tier 1 backpressure: Coordinator holds obligation, retries when space frees
- Candidate A Tier 2 backpressure: ISRRetireRouter returns `false`, caller retains ptr

**The mismatch:** I4 D14.3's backpressure is designed for **recovery obligations** (high-level semantic objects), not for **pointer retirement** (low-level memory management). Adding backpressure to pointer retirement would be a **new architectural pattern** not modeled in I4.

---

## 12. A3 Gate Summary

| Gate | Criterion | Status | Evidence |
|---|---|---|---|
| **A3-1** | P-4 vs I4 D14.3 relationship defined | ✅ PASS | §2: A-2 (revise P-4) is the only viable path |
| **A3-2** | Terminal full: ptr ownership unambiguous | ⚠️ CONDITIONAL | Proposed: model as new `stalled`-like state, requires I4 D15.2 revision |
| **A3-3** | RetireEnqueueResult ownership semantics unique | ✅ PASS | §6: Need new `TerminalPressure` enum value |
| **A3-4** | All production callers ownership flow tracked | ✅ PASS | §10: 14 callers enumerated in full table |
| **A3-5** | `tstored` classified (current vs latent) | ✅ PASS | §5: Latent bug — activates under Candidate A |
| **A3-6** | Retry authority singularized | ⚠️ CONDITIONAL | §7: Option B (Coordinator) is I4-compliant; requires architectural changes |
| **A3-7** | TerminalFull → wake → drain → retry causal chain proven | ⚠️ CONDITIONAL | §8: Requires new retry obligation queue + retry signal (not in current architecture) |
| **A3-8** | K_terminal determination method defined | ✅ PASS | §9: K_terminal = f(grace_lifetime_max, T_build_min, sizeof(Entry)) — measurement-gated |
| **A3-9** | Ownership conservation (INV-X1-7) not violated | ⚠️ CONDITIONAL | Requires adding `retiredPendingTerminal` state to I4 D15.2 conservation equation |
| **A3-10** | Candidate A GO/NO-GO decision possible | ⚠️ NO-GO (as-is) | 7 callers would leak; requires 14-caller retrofit + I4 D15.2 revision + architectural wakeup redesign |

---

## 13. Final Verdict — Candidate A Feasibility

### Overall: **NO-GO (as-is) — requires major architectural changes**

Candidate A is **conceptually correct** (bounded Terminal + backpressure aligns with I4 D14.3 direction), but **not implementable without major changes**:

### Required changes if Candidate A is pursued:

1. **P-4 revision** (Step 3-1): Change from "Terminal ALWAYS accepts" to "Terminal bounded; caller retains on full"
2. **RetireEnqueueResult revision** (Step 3-5): Add `TerminalPressure` enum value to distinguish from `TerminalReclaim`
3. **`tstored` bug fix** (Step 3-4): Must NOT be fixed in isolation — requires coordinated enum + caller updates
4. **I4 D15.2 conservation revision** (Step 3-2): Add `retiredPendingTerminal` state to ownership conservation equation
5. **All 14 callers must handle `TerminalPressure`** (Step 3-3):
   - 7 callers currently ignore result → must add ownership tracking + retry
   - 4 callers (Cache, Convolver, Bridge, Snapshot) use void-return APIs → must change return types
   - 1 caller (EQProcessor) must distinguish `TerminalPressure` from `TerminalReclaim`
6. **Retry authority = Coordinator** (Step 3-6): Option B — Coordinator must own retry obligations, not ISRRetireRouter
7. **Wakeup redesign** (Step 3-7): Terminal full → signal drain → drain frees space → signal retry → caller retries — requires new retry obligation queue + upward retry signal

### Estimated change scope:

- **Files to modify:** 14+ files (ISRRetireRouter.h/.cpp, 7 caller files, I4 D15.2 doc, AudioEngine.h)
- **API changes:** `RetireEnqueueResult` enum, 7 caller return types (void → bool/enum)
- **New state:** `retiredPendingTerminal` ownership state in conservation equation
- **New infrastructure:** Retry obligation queue, upward retry signal
- **Testing:** All existing retirement tests + new backpressure tests

### Recommendation

**Candidate A is NO-GO for immediate implementation.** The architectural gap between I4 D14.3's reservation-first backpressure model (Tier 1: recovery obligations) and the pointer retirement pipeline (Tier 2: D/Q/E/T) is fundamental. Bridging it requires retrofitting the entire retirement path with I4-style reservation + placement + backpressure — a **Phase I** effort, not a Phase 9-B incremental change.

**Candidate B (shutdown-only bounded)** remains the recommended path — it preserves all invariants (P-4, D14.3, D15.2) with ZERO code changes, leveraging the existing `drainAllQuarantineStore()` at shutdown.

---

## 14. No-code-change confirmation

**Zero code changes in this step.** All findings are derived from read-only code analysis.

Files READ but NOT modified:
- `src/audioengine/ISRRetireRouter.h`
- `src/audioengine/ISRRetireRouter.cpp`
- `src/audioengine/ISRRetire.h`
- `src/audioengine/ISRRetireRuntimeEx.h`
- `src/audioengine/ISRAuthorityClass.h`
- `src/audioengine/ISRRuntimePublicationCoordinator.h`
- `src/audioengine/ISRRuntimePublicationCoordinator.cpp`
- `src/audioengine/AudioEngine.h`
- `src/audioengine/AudioEngine.Cache.cpp`
- `src/audioengine/AudioEngine.Commit.cpp`
- `src/audioengine/AudioEngine.Retire.cpp`
- `src/audioengine/AudioEngine.Threading.cpp`
- `src/audioengine/AudioEngine.Timer.cpp`
- `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`
- `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp`
- `src/audioengine/AudioEngine.CtorDtor.cpp`
- `src/audioengine/AudioEngine.RebuildDispatch.cpp`
- `src/audioengine/DSPLifetimeManager.cpp`
- `src/eqprocessor/EQProcessor.Core.cpp`
- `src/convolver/ConvolverProcessor.Lifecycle.cpp`
- `src/core/SnapshotCoordinator.cpp`
- `src/core/SnapshotCoordinator.h`
- `src/core/DeferredDeletionQueue.h`
- `src/core/IEpochProvider.h`
- `src/core/IRetireRouter.h`
- `src/core/RuntimePublicationCoordinator.h`
- `doc/work88/I4_DESIGN_CONTRACT.md`
