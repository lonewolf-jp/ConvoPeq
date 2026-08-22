# D101-9 Phase 9-B Step 1 — Terminal Capacity / Overflow Path Census

## 1. Purpose

Establish the complete ownership chain, capacity model, and overflow semantics of `TerminalReclaimAuthority` **before** any code changes. This census answers:

- What are ALL entry paths into Terminal?
- What are ALL exit paths from Terminal?
- What happens when Terminal is "full" (under current growable design)?
- What is the shutdown drain path?
- What proof obligations (B1–B7) does the bounded conversion need to satisfy?

**No code changes in this step.**

---

## 2. Terminal Reclaim Authority — Architecture Overview

### 2.1 Ownership Chain (D → Q → E → T)

The ownership transfer chain for deferred deletion (P-4 invariant), all callers are **Non-RT**:

```
enqueueWithRetry()  (ISRRetireRouter.cpp:290-360)
       │
       ├── Stage 1: DeferredDeletionQueue (D)   — lock-free ring, fixed capacity
       │       enqueueRetire() → provider_->enqueueRetireTyped()
       │       Returns: Success / QueuePressure / QueueFull / Shutdown
       │
       ├── Stage 2: Retry loop (kMaxRetry=2)
       │       tryReclaim(); drainEmergencyAndTerminal(); re-enqueue to D
       │
       ├── Stage 3: RetireQuarantineStore (Q)     — std::array, capacity=512
       │       m_retireQuarantine.quarantine()
       │       Returns: true (stored) / false (full → escalate)
       │
       ├── Stage 4: EmergencyQuarantineStore (E)  — std::array, capacity=512
       │       m_emergencyQuarantine.quarantine()
       │       Returns: true (stored) / false (full → escalate)
       │
       └── Stage 5: TerminalReclaimAuthority (T) — std::vector, GROWABLE
                terminalReclaim() → m_terminalReclaim.store()
                Returns: **ALWAYS true** (growable, never rejects)
                ↓
                signalDrainWakeup()  (8R-1 wakeup primitive)
```

### 2.2 Key Design Constants

| Store | Type | Capacity | Overflow Behavior | File |
|---|---|---|---|---|
| D (DeferredDeletionQueue) | lock-free ring | `m_ringSize` (runtime config, typically 256-1024) | Returns QueuePressure → retry → escalate to Q | `DeferredDeletionQueue.h` |
| Q (RetireQuarantineStore) | `std::array` | `kMaxQuarantinedEntries = 512` | Returns `false` → caller escalates to E | `RetireQuarantineStore.h:79` |
| E (EmergencyQuarantineStore) | `std::array` | `kMaxQuarantinedEntries = 512` (same type, separate instance) | Returns `false` → caller escalates to T | `RetireQuarantineStore.h:79` (same class) |
| T (TerminalReclaimAuthority) | `std::vector<Entry>` | **UNBOUNDED (growable)** | **ALWAYS `true`** — P-4 invariant | `ISRRetireRouter.h:47` |

**Current effective chain capacity (bounded):** D + Q + E = `D_capacity + 512 + 512 = D_capacity + 1024`

**Current Terminal capacity:** ∞ (std::vector grows as needed)

---

## 3. Terminal Entry Paths — Complete Census

### 3.1 Primary Entry: `enqueueWithRetry()` → `terminalReclaim()` (Non-RT runtime)

**File:** `src/audioengine/ISRRetireRouter.cpp`
**Lines:** 336-349 (Stage 5 of `enqueueWithRetry`)

```cpp
const bool tstored = terminalReclaim(ptr, deleter, epoch, type,
                                     "enqueueWithRetry:TerminalReclaim");
(void)tstored;  // ★ P-4: always true (growable store)
result = RetireEnqueueResult::TerminalReclaim;
```

**Conditions to reach Terminal (all must fail):**
1. D enqueue → `QueuePressure` (D full)
2. Retry loop exhausted (2 attempts) — both D + tryReclaim + drainEmergencyAndTerminal
3. Q quarantine → `false` (Q full, `kMaxQuarantinedEntries=512`)
4. E emergencyQuarantine → `false` (E full, `kMaxQuarantinedEntries=512`)

**Entry via `terminalReclaim()`:**
- `src/audioengine/ISRRetireRouter.cpp:430-465`

**Inside `terminalReclaim()` (lines 430-465):**
```cpp
bool ISRRetireRouter::terminalReclaim(void* ptr, void (*deleter)(void*), uint64_t epoch,
                                      DeletionEntryType type, const char* reason) noexcept
{
    const uint64_t minReader = minReaderEpoch();
    const bool epochSafe = ISRRetireRouter::isOlder(epoch, minReader);  // epoch < minReader → safe
    const bool isRt = convo::numeric_policy::isAudioThread();

    if (epochSafe && !isRt)
    {
        deleter(ptr);                    // synchronous destruction (Non-RT)
        if (type == DeletionEntryType::World)
            m_terminalReclaim.recordWorldReclaim();
        return true;  // destroyed immediately, no storage
    }

    // epoch unsafe OR RT caller → store for later drain
    return m_terminalReclaim.store(ptr, deleter, epoch, type, reason);  // ALWAYS true
}
```

**Two Terminal entry sub-paths:**
1. **Synchronous destruction** (epochSafe && !isRt): deleter called immediately, no Terminal storage. Entry count: 0
2. **Stored for drain** (epoch unsafe OR RT): `m_terminalReclaim.store()` — pushes to `entries_` std::vector

### 3.2 Secondary Entry: `shutdownReclaim()` → `terminalReclaim()` (Non-RT shutdown)

**File:** `src/audioengine/ISRRetireRouter.cpp`
**Lines:** 540-560

**Called from:**
- `AudioEngine.h:4221` — `enqueueDeferredDeleteNonRtWithResult()` during shutdown:
  ```cpp
  if (isShutdownInProgress()) {
      const uint64_t epoch = markRetireEpoch();
      const bool transferred = m_retireRouter->shutdownReclaim(ptr, deleter, epoch, type);
      ...
  }
  ```
- `enqueueDeferredDeleteNonRtWithResult()` at `AudioEngine.h:4208`

**Condition:** `isShutdownInProgress()` is true (shutdown state machine activated)

**Inside `shutdownReclaim()` (lines 548-559):**
```cpp
bool ISRRetireRouter::shutdownReclaim(void* ptr, void (*deleter)(void*), uint64_t epoch,
                                      DeletionEntryType type) noexcept
{
    if (ptr == nullptr || deleter == nullptr)
        return true;
    return terminalReclaim(ptr, deleter, epoch, type, "shutdownReclaim");
}
```

Delegates directly to `terminalReclaim()`, which uses the same epoch-safe check. During shutdown, Audio Thread is stopped, so `isAudioThread()` returns false — but if the epoch is still unsafe (stuck reader), it still stores to Terminal.

### 3.3 Direct Entry: `TerminalReclaimAuthority::store()` (API)

**File:** `src/audioengine/ISRRetireRouter.cpp`
**Lines:** 27-53

Direct call to `store()` — used in tests (`StuckReaderFallbackDrainTests.cpp:121`) and conceptually the final authority.

---

## 4. Terminal Exit Paths — Complete Census

### 4.1 Primary Exit: `drain()` (epoch-gated, Non-RT periodic)

**File:** `src/audioengine/ISRRetireRouter.cpp`
**Lines:** 39-75

Called via:
- `ISRRetireRouter::drainEmergencyAndTerminal()` (line 496)
- `ISRRetireRouter::drainTerminalReclaim()` (line 490)
- `ISRRetireRouter::tryReclaim()` (line 282: `provider_->tryReclaim()` + drainQuarantineStore + drainEmergencyAndTerminal)

**drain() logic:**
- Iterates `entries_` under lock (`mtx_`)
- For each entry: `isOlderFn(e.epoch, minReaderEpoch)` == true → extract to pending list
- Compacts `entries_` vector (removes null entries)
- `entries_.resize(w)` — **shrinks** the vector
- Releases lock → executes deleters outside lock (reentrancy-safe)
- For `DeletionEntryType::World`: increments `reclaimCount_`, calls `referenceObserver_->onRelease()`
- Decrements `residentAtomic_` by pending count

**Note:** The vector can grow during `store()` (push_back) but only shrinks during `drain()`. There is NO compaction of "empty slots" — null entries are overwritten during the drain compaction loop. The vector's capacity grows geometrically (std::vector default) but never shrinks below the high-water mark of `size()`.

### 4.2 Shutdown Exit: `drainAll()` (unconditional, forced)

**File:** `src/audioengine/ISRRetireRouter.cpp`
**Lines:** 77-98

Called via:
- `ISRRetireRouter::drainAllQuarantineStore()` (line 408-411):
  ```cpp
  m_terminalReclaim.drainAll();
  ```
- This is called from `ISRRetireRouter::drainAll()` (line 520-524):
  ```cpp
  void ISRRetireRouter::drainAll() noexcept {
      provider_->drainAll();      // D drain
      drainAllQuarantineStore();   // Q + E + Terminal drainAll
  }
  ```

**drainAll() logic:**
- Swaps `entries_` with empty vector under lock
- Resets `residentAtomic_` to 0
- Executes ALL deleters (no epoch check)
- Same World-type accounting as drain()

**Shutdown ordering context** (`AudioEngine.Processing.ReleaseResources.cpp`):
```
Line 191: closeAdmission()       ← Phase 9-A (Q7)
Line 192: joinProducers()        ← Phase 9-A (Q1)
Line 198: advanceRetireEpoch()   ← Q5
Line 206: closeReaderRegistration() ← Q3
Lines 217: drain loop             ← Q4/Q6 (waitForDrain)
Line 540: drainAllQuarantineStore() ← only if activeReaderCount()==0 (line 482)
Line 564: drainAllNonRt residual
```

### 4.3 Direct Exit: `tryReclaim()` (conditional)

**File:** `src/audioengine/ISRRetireRouter.cpp`
**Lines:** 99-120

Called when `entries_` is non-empty, drains epoch-safe subset. Returns true if any reclaimed.

---

## 5. Terminal "Full" Semantics — Current (Growable) Behavior

### 5.1 `store()` always returns true

```cpp
// ISRRetireRouter.cpp:27-53
bool TerminalReclaimAuthority::store(...) noexcept {
    ...
    std::lock_guard<std::mutex> lock(mtx_);
    entries_.push_back(Entry{ptr, deleter, epoch, type, reason});
    residentAtomic_.fetch_add(1, std::memory_order_release);
    return true;  // ★ P-4: growable store — ALWAYS accepts
}
```

**Critical implications:**
- **No "store full" failure path exists.** P-4 invariant holds: `enqueueWithRetry()` never returns with ptr unowned.
- `std::vector::push_back` can throw `std::bad_alloc` (heap allocation failure). In `noexcept` context, this would call `std::terminate()`. This is the only failure path — OOM.
- No backpressure is propagated to callers. Callers see `RetireEnqueueResult::TerminalReclaim` and treat it as success (ownership transferred).

### 5.2 No overflow counter for Terminal

Unlike `RetireQuarantineStore` which has `overflowCount_` (line 166), `TerminalReclaimAuthority` has **no overflow counter** because `store()` never rejects. The `m_overflowCount_` on `ISRRetireRouter` (line 239) tracks Q/E full rejections, but Terminal acceptance is always true.

### 5.3 No health escalation from Terminal

`RetireQuarantineStore` returns `false` on full, which triggers caller escalation to the next stage. `TerminalReclaimAuthority` never returns false, so there is **no Terminal-full health escalation path**. The ownership chain is effectively unbounded at Terminal.

---

## 6. Shutdown Path — Complete Trace

### 6.1 Normal shutdown sequence (releaseResources)

**File:** `AudioEngine.Processing.ReleaseResources.cpp`

| Line | Step | Action | Q-condition |
|---|---|---|---|
| 189 | Q2 | `shutdownCoordinatorLoop()` join | allProducersJoined |
| 190 | Q2 | `stopRebuildThread()` join | Builder joined |
| **191** | **9-A** | `closeAdmission()` (Open→Closing) | Q7: !isAdmissionOpen()=true |
| **192** | **9-A** | `joinProducers()` (Closing→Closed) | Q1: admissionState()==Closed |
| 198 | Q5 | `advanceRetireEpoch()` | epochSettled |
| 206 | Q3 | `closeReaderRegistration()` | readerRegClosed |
| 217 | Q4/Q6 | `waitForDrain()` loop | activeReadersZero + postStopEnqueueZero |
| 482 | (conditional) | `if activeReaderCount()==0: drainAllQuarantineStore()` | Terminal drain |
| 564 | | `drainAllNonRt` residual | final drain |

### 6.2 Key shutdown invariant: `drainAllQuarantineStore()` conditional

```cpp
// AudioEngine.Processing.ReleaseResources.cpp:482
if (m_retireRouter->activeReaderCount() == 0)
    m_retireRouter->drainAllQuarantineStore();
```

**If `activeReaderCount() > 0` (stuck reader):** Terminal drain is **SKIPPED**. Entries remain in Terminal (growable store holds them). This is by design — UAF avoidance. These entries are then drained via the epoch-gated path in `waitForDrain` (which calls `drainDeferredRetireQueues` → `tryReclaim` → `drainEmergencyAndTerminal` → `drainTerminalReclaim`).

### 6.3 `drainPendingRetireIntentsForShutdown()` (line 525)

Clears residual `RetireIntent` slot-state entries (System 1) — separate from Terminal but part of the Q4/Q6 drain completion.

---

## 7. World Conservation Impact — `DeletionEntryType::World`

### 7.1 World entries in Terminal

When `DeletionEntryType::World` entries are stored in Terminal:

**On drain():**
```cpp
if (e.type == DeletionEntryType::World) {
    ++reclaimCount_;
    if (referenceObserver_ != nullptr)
        referenceObserver_->onRelease();
}
```

**On drainAll():** Same World accounting.

**On synchronous destruction in terminalReclaim():**
```cpp
if (type == DeletionEntryType::World)
    m_terminalReclaim.recordWorldReclaim();
```
`recordWorldReclaim()` (ISRRetireRouter.h:89-93):
```cpp
void recordWorldReclaim() noexcept {
    ++reclaimCount_;
    if (referenceObserver_ != nullptr)
        referenceObserver_->onRelease();
}
```

### 7.2 Conservation invariant for World entries

- World entry stored in Terminal → ownership transferred to Terminal ✅
- World entry drained (epoch-safe) → deleter called, `onRelease()` emitted, `reclaimCount_`++ ✅
- World entry synchronously destroyed (epoch-safe, Non-RT) → deleter called, `recordWorldReclaim()`, `onRelease()` emitted ✅
- World entry in drainAll() (shutdown) → deleter called, same accounting ✅
- **No double-reclaim path:** drain() nulls the entry (`e = Entry{}`) before resize; drainAll() swaps to empty vector; synchronous path never stores ✅
- **No World loss:** If Terminal grows unboundedly, all entries eventually drained via drainAll() at shutdown, or via epoch-gated drain() during normal operation ✅

### 7.3 Test verification

`StuckReaderFallbackDrainTests.cpp` (line 100+):
- `fillTerminalStore()` fills Terminal with 50 entries (future epoch → stored, not destroyed)
- `drainAllQuarantineStore()` clears all (Q + E + Terminal)
- Verifies `terminalReclaimResidentCount() == 0` after drainAll ✅
- Verifies `trackerT.invokeCount == tCount` (deleter called exactly once per entry) ✅
- Verifies `TestObject::aliveCount == 0` (no leaks) ✅

---

## 8. Current Overflow Path — What Happens Today

### 8.1 Normal operation (all stores filling)

```
1. enqueueWithRetry called (Non-RT)
2. Stage 1: D.enqueue → QueuePressure (D full, RT readers active)
3. Stage 2: retry loop (2 attempts):
   a. tryReclaim() → drains D epoch-safe entries
   b. drainEmergencyAndTerminal() → drains E + Terminal epoch-safe
   c. D.enqueue again → QueuePressure again
4. Stage 3: Q.quarantine → false (Q full: 512 entries)
5. Stage 4: E.emergencyQuarantine → false (E full: 512 entries)
6. Stage 5: terminalReclaim() → store() → ALWAYS true
   → ptr now owned by Terminal (std::vector)
   → signalDrainWakeup() called
   → returns RetireEnqueueResult::TerminalReclaim
7. Entry sits in Terminal until:
   a. drainTerminalReclaim() drains it (epoch becomes safe) ← normal path
   b. drainAll() destroys it unconditionally ← shutdown path
```

### 8.2 Overflow accounting today

| Store | Full detection | Overflow counter | Health escalation |
|---|---|---|---|
| D | QueuePressure return | None (retry only) | None |
| Q | `quarantine()` returns false | `overflowCount_` in `RetireQuarantineStore` | Caller escalates to E |
| E | `quarantine()` returns false | `overflowCount_` in `RetireQuarantineStore` | Caller escalates to Terminal |
| T | **N/A** (never full) | **N/A** (no overflow counter) | **None** (always accepts) |

### 8.3 Memory growth under sustained pressure

If both Q (512) and E (512) are persistently full, Terminal grows unbounded:
- Each `store()` → `entries_.push_back()` → potential heap allocation
- No backpressure to caller (returns TerminalReclaim = success)
- Memory grows until: (a) epoch becomes safe and drain() clears entries, or (b) shutdown drainAll()

**This is the exact gap that Phase 9-B addresses:** converting Terminal from growable to bounded.

---

## 9. Proof Obligations for Bounded Conversion (B1–B7)

### B1 — Capacity

```text
∀t: |Terminal(t)| ≤ K_terminal
```
**Current state:** NOT satisfied. `entries_` is `std::vector<Entry>` — unbounded.
**Required:** Fixed-capacity storage (e.g., `std::array<Entry, K_terminal_max>` or bounded ring).

### B2 — Ownership continuity

Terminal full must NOT cause ptr loss:
```text
ptr ∈ caller
∨ ptr ∈ D
∨ ptr ∈ Q
∨ ptr ∈ E
∨ ptr ∈ Terminal
∨ ptr is synchronously destroyed
```

**Current analysis:**
- Today: Always `ptr ∈ Terminal` (store() always succeeds). B2 trivially holds.
- Bounded conversion: When Terminal is full, must either:
  - (a) **Caller retains ownership** (store returns false, caller handles) — but all callers are Non-RT, so caller could retry later or escalate. This risks breaking P-4 invariant.
  - (b) **Synchronous destruction** (destroy ptr immediately) — BUT this risks UAF if epoch is unsafe (RT reader may still reference). NOT safe.
  - (c) **Block/spin** until space available — risks deadlock if no drain thread wakes (but CoordinatorLoop runs). This is the backpressure approach.
  - (d) **Evict oldest entry** (drop it) — breaks conservation (ptr lost). NOT acceptable.

**Recommended approach:** Option (a) with upstream backpressure propagation: Terminal full → `store()` returns `false` → `terminalReclaim()` returns `false` → `enqueueWithRetry()` caller retains ownership → escalate to `enqueueDeferredDeleteNonRtWithResult` which returns `TerminalReclaim` → caller (e.g., DSPLifetimeManager) gets `false` → backpressure signal to RuntimeHealthMonitor.

**Problem:** This breaks P-4 invariant ("TerminalReclaimAuthority ALWAYS accepts"). The review attachment explicitly warns: "simple std::array replacement may break P-4."

### B3 — No double ownership

**Current state:** Each `store()` call transfers ownership atomically under `mtx_`. drain() nulls entries before erasing. drainAll() swaps to empty. Synchronous destruction path doesn't store. No overlap. ✅

**Bounded conversion risk:** If using a ring buffer with overwrite semantics, old entries could be overwritten before drain → double-free or leak. Must maintain "extract-then-destroy" discipline.

### B4 — Normal drain (epoch-gated)

**Current state:** `drain()` iterates entries, checks `isOlder(entry.epoch, minReaderEpoch)`, extracts safe entries, executes deleters outside lock. ✅

**Bounded conversion:** Must preserve epoch-gated drain. Ring buffer must support "mark slot empty after extraction" without losing undrained entries.

### B5 — Shutdown drain

**Current state:** `drainAll()` swaps all entries to local vector, resets counters, destroys all unconditionally. Called from `drainAllQuarantineStore()` when `activeReaderCount() == 0`, or from `drainAll()` during full shutdown. ✅

**Bounded conversion:** Must preserve drainAll(). With fixed array, drainAll() iterates all slots.

### B6 — Wakeup (8R-1 invariant)

**Current state:** `store()` → `residentAtomic_.fetch_add(1)` → `signalDrainWakeup()` (if called from enqueueWithRetry Stage 5).

**Bounded conversion:** Must preserve `residentAtomic_++` on insertion. The wakeup predicate (`pendingRetireCount() != 0 || residentCountAtomic() != 0`) includes Terminal's atomic counter. Must not break.

**Key detail:** `signalDrainWakeup()` is called from `enqueueWithRetry()` at line 357 (after Q/E/T path), NOT from `TerminalReclaimAuthority::store()` itself. The signal is at the ISRRetireRouter level. This means the wakeup happens regardless of whether the entry went to Q, E, or T.

### B7 — Conservation (`DeletionEntryType::World`)

**Current state:** World entries in Terminal follow the same drain/destroy accounting as non-World entries. World-specific code path is only in the deleter execution (increment `reclaimCount_`, call `onRelease()`). ✅

**Bounded conversion:** Must preserve `DeletionEntryType::World` accounting exactly. Capacity reduction must NOT change World reclaim counting.

---

## 10. Test Inventory — Existing Terminal Tests

### 10.1 StuckReaderFallbackDrainTests.cpp

**File:** `src/tests/StuckReaderFallbackDrainTests.cpp`

| Test | What it verifies |
|---|---|
| `testStuckReaderFallbackDrainsAllStores` | Fills Q (512), E (512), Terminal (50) → `drainAllQuarantineStore()` clears all → resident counts = 0, deleters called exactly once, no leaks |
| `testTerminalResidentCountAfterDrain` | Terminal drain reduces resident count |
| `testTerminalSynchronousDestruction` | epoch-safe + Non-RT → immediate deletion, no Terminal storage |

**Key test helper:**
```cpp
// fillTerminalStore — uses future epoch so terminalReclaim stores (not destroys)
static int fillTerminalStore(ISRRetireRouter& router, int startId,
                             DeleterTracker& tracker, int n);
```
- Creates objects with future epoch (`currentEpoch() + 1000`)
- Calls `router.terminalReclaim()` which stores (not destroys) all n entries
- Verifies `terminalReclaimResidentCount() == n`
- After `drainAllQuarantineStore()`: verifies count == 0

### 10.2 RetireGraceSemanticsTests.cpp

**File:** `src/tests/RetireGraceSemanticsTests.cpp`

- Line 364-366: Tests `TerminalReclaimAuthority` lock-free counter consistency
- Line 425-427: `TerminalReclaimAuthority auth;` direct testing

### 10.3 Missing test — Terminal full (bounded conversion)

**No test exists for Terminal full behavior** because Terminal is currently growable. The new tests from the review (9B-5) would be:
- `testTerminalCapacityExactBound` — Terminal at capacity K_terminal, K_terminal+1 th insertion
- `testTerminalFullOwnershipRetention` — when full, ownership-safe fallback
- `testTerminalTransferNoDoubleOwnership` — no double-free
- `testTerminalDrainEpochGate` — epoch-gated drain preserved
- `testTerminalDrainAllShutdown` — shutdown drainAll preserved
- `testTerminalWorldOnReleaseExactlyOnce` — World type accounting
- `testTerminalWakeupAfterInsertion` — 8R-1 wakeup invariant
- `testTerminalFullThenRecovery` — full → drain → recovery

---

## 11. Design Constraints from Existing Architecture

### 11.1 Non-RT only

`TerminalReclaimAuthority` uses `std::mutex` (not lock-free). All access is from Non-RT threads:
- `enqueueWithRetry()` — Non-RT (has `jassert(!isAudioThread())` guard, line 292)
- `terminalReclaim()` — called from `enqueueWithRetry()` (Non-RT) and `shutdownReclaim()` (Non-RT)
- `tryReclaim()` / `drainEmergencyAndTerminal()` / `drainTerminalReclaim()` — called from CoordinatorLoop (Non-RT) or `tryReclaim()` (Non-RT)
- `drainAll()` / `drainAllQuarantineStore()` — shutdown path (Audio Thread stopped, Non-RT)

### 11.2 Mutex contention

`mtx_` protects `entries_`, `size_`, `overflowCount_`, `residentAtomic_` (increment only), `reclaimCount_`. `residentAtomic_` is read lock-free from RT path (`residentCountAtomic()`).

### 11.3 No reentrancy in store()

`store()` holds `mtx_` during `push_back`. If `push_back` triggers reallocation, the old buffer is freed — but since `store()` holds `mtx_`, no concurrent drain can access it. ✅

### 11.4 `store()` with growable vector — only failure mode is OOM

If `push_back` fails (bad_alloc, in noexcept context → terminate), the behavior is undefined/abort. This is the only failure mode of the current growable design.

---

## 12. Summary — What Bounded Conversion Must Preserve

| Aspect | Current (Growable) | Bounded Conversion Requirement |
|---|---|---|
| `store()` return | Always `true` | Must decide: `false` on full (break P-4) or spin/wait (deadlock risk) or signal (backpressure) |
| Capacity | ∞ (std::vector) | Fixed `K_terminal` (proposal: 4096) |
| Ownership on full | Always transferred | Must define fallback (caller retain + backpressure) |
| World accounting | Via drain()/drainAll() deleter paths | Must preserve exactly |
| Wakeup (8R-1) | residentAtomic_++ on store + signalDrainWakeup() | Must preserve |
| Synchronous destruction | epoch-safe + Non-RT → immediate delete | Must preserve |
| Shutdown drainAll | Unconditional destroy all | Must preserve |

### Key design decision for Phase 9-B:

The review proposes converting `std::vector<Entry>` to `std::array<Entry, K_terminal_max>` with `store()` returning `false` when full. This **breaks P-4** (Terminal no longer always accepts). The fallback must be:

1. `store()` returns `false` → `terminalReclaim()` returns `false`
2. `enqueueWithRetry()` caller receives `false` → **retains ownership** (does NOT call `signalDrainWakeup` for this ptr)
3. Caller must **retry** (re-attempt the chain: D → Q → E → T) or **escalate** to backpressure/health
4. Since all callers are Non-RT, retry is safe (no RT deadlock)
5. The retry loop in `enqueueWithRetry()` already exists (Stage 2, kMaxRetry=2) — but it doesn't currently account for Terminal being full

**Alternative (Candidate B — shutdown-only emergency path):** Keep Terminal growable for normal operation, but cap it during shutdown. This preserves P-4 for normal operation but still makes `K_terminal < ∞` a production-code bound (shutdown is bounded because all readers are joined).

**Recommended:** Evaluate both candidates. Candidate A (full bounded) is cleaner but requires backpressure plumbing. Candidate B (shutdown-bounded) is conservative but still satisfies `K_world < ∞` proof requirement.

---

## 13. Next Steps for Phase 9-B Implementation

1. **Chosen candidate**: **Candidate D** (bounded Terminal + emergency synchronous destruction on full) — validated by review. This breaks P-4 but is acceptable because Terminal-full implies EBR catastrophe (readers stuck), at which point synchronous destruction with HealthMonitor escalation is the fail-safe.
2. **Define `K_terminal` value**: `2048` (review proposed 1024; census proposed 4096; 2048 balances headroom vs. cache locality. Total K_world ≤ A_max + 5120 + 2048 = A_max + 7168)
3. **Define overflow semantics**: `store()` returns `false` when `size_ >= K_terminal` → caller (`terminalReclaim`) performs synchronous destruction + HealthMonitor escalation
4. **Implement test suite**:
   - `testTerminalCapacityExactBound`
   - `testTerminalFullOwnershipRetention`
   - `testTerminalTransferNoDoubleOwnership`
   - `testTerminalDrainEpochGate`
   - `testTerminalDrainAllShutdown`
   - `testTerminalWorldOnReleaseExactlyOnce`
   - `testTerminalWakeupAfterInsertion`
   - `testTerminalFullThenRecovery`
5. **Update ConvoPeq.md** with new capacity constant and overflow semantics
6. **Verify existing tests pass** (StuckReaderFallbackDrainTests, RetireGraceSemanticsTests)

---

## Appendices

## Appendix A: Review Validation (Pasted text #1)

**Review source:** "D101-9 Phase 9-B Step 1 — Latest-Source Terminal Capacity / Overflow Path Census"

### Claim-by-claim verification

| # | Review Claim | Verdict | Evidence |
|---|---|---|---|
| 1 | 3 ingress paths (A: cascade fallback, B: shutdown reclaim, C: epoch-unsafe direct) | ✅ VALID | ISRRetireRouter.cpp:290-360 (enqueueWithRetry Stage 5), 490-511 (terminalReclaim), 548-560 (shutdownReclaim) |
| 2 | 2 egress paths (X: normal epoch-gated drain, Y: shutdown drainAll) | ✅ VALID | ISRRetireRouter.cpp:39-75 (drain), 77-98 (drainAll) |
| 3 | D=4096, Q=512, E=512, total D+Q+E=5120 | ✅ VALID | DeferredDeletionQueue.h:262 (`kQueueSize=4096`), RetireQuarantineStore.h:79 (`kMaxQuarantinedEntries=512`) |
| 4 | store() always returns true (P-4 invariant) | ✅ VALID | ISRRetireRouter.cpp:53 (`return true;`), ISRRetireRouter.h:47 |
| 5 | No overflow counter for TerminalReclaimAuthority | ✅ VALID | TerminalReclaimAuthority class (ISRRetireRouter.h:40-72) has no `overflowCount_` member |
| 6 | No health escalation from Terminal full | ✅ VALID | No upstream backpressure path exists (store() never returns false) |
| 7 | Candidate D: bounded + emergency synchronous destruction | ✅ VALID ANALYSIS | P-4 break acknowledged; synchronous destruction acceptable when epoch-unsafe is already catastrophe state |
| 8 | recordWorldReclaim doesn't update residentAtomic_ | ✅ VALID | ISRRetireRouter.h:89-93: `recordWorldReclaim()` only increments `reclaimCount_` and calls `onRelease()`. No `residentAtomic_` change — correct because synchronous path never called `store()` |
| 9 | drainAllQuarantineStore conditional on activeReaderCount==0 | ✅ VALID | AudioEngine.Processing.ReleaseResources.cpp:482 |
| 10 | signalDrainWakeup called after Q/E/T entry (not from store()) | ✅ VALID | ISRRetireRouter.cpp:355-357: signalDrainWakeup() called in enqueueWithRetry after the Q/E/T path, not inside TerminalReclaimAuthority::store() |
| 11 | K_terminal=1024 proposal is sufficient | ⚠️ REVIEW NOTE | Review proposes 1024. Census proposes 4096. 1024 gives K_world ≤ A_max + 5120 + 1024 = A_max + 6144. Either is finite — choose based on high-water mark telemetry from tests. |
| 12 | Line refs in review (ISRRetireRouter.cpp:342-350, 408-418, etc.) | ⚠️ SLIGHTLY STALE | Phase 9-A code change (closeAdmission + joinProducers at lines 191-192) shifted lines by +5. Review refs are approximately correct but should be +5 in current source. |

### Review's Candidate D evaluation — validation

**Review claim:** "Terminal full = EBR Catastrophe (D+Q+E=5120 entries all epoch-unsafe)"

✅ **VALID.** If D(4096) + Q(512) + E(512) = 5120 entries are all epoch-unsafe (all readers stuck at old epoch), and Terminal accepts more, this means >5120 Worlds cannot be destroyed. This can only happen when:
- Reader is stuck (not advancing epoch) — already a defect state
- Sustained high retire rate exceeds EBR reclamation — abnormal operation

**Review claim:** "Caller retains ownership on Terminal full breaks P-4; synchronous destruction is preferred fallback"

✅ **VALID.** The review correctly identifies the trade-off:
- Caller-retains (Candidate A) → memory leak under sustained pressure (callers are Non-RT, may never retry)
- Synchronous destruction (Candidate D) → UAF risk, but system is already in catastrophe state (readers stuck)
- The review's recommendation for synchronous destruction with HealthMonitor escalation is sound: if readers are stuck, the system is already compromised, and preventing OOM crash takes priority over UAF

**Review claim:** "K_terminal = 1024 is sufficient"

⚠️ **NEEDS VALIDATION.** The census proposed 4096 (matching the review's D=4096 reference). The review's 1024 is conservative (total chain = 5120 + 1024 = 6144). Recommendation: use 2048 as a balance — provides 2x D as headroom, total K_world ≤ A_max + 7168, while keeping array size manageable for cache locality in drain().

### Test strategy validation

The review's test list (Section 9B-5) is **VALID and COMPLETE**:

| Test | Review specifies? | Census specifies? | Status |
|---|---|---|---|
| testTerminalCapacityExactBound | ✅ | ✅ | Both agree |
| testTerminalFullOwnershipRetention | ✅ | ✅ | Both agree — KEY TEST for Candidate D |
| testTerminalTransferNoDoubleOwnership | ✅ | ✅ | Both agree |
| testTerminalDrainEpochGate | ✅ | ✅ | Both agree |
| testTerminalWorldOnReleaseExactlyOnce | ✅ (partial — 1 test) | ✅ (expanded to 8) | Review subset OK |
| testTerminalWakeupAfterInsertion | ❌ | ✅ | Census adds this — critical for 8R-1 invariance |
| testTerminalFullThenRecovery | ❌ | ✅ | Census adds this — recovery path |

**Recommendation:** Use the census's expanded test list (8 tests) — it's a superset that covers additional edge cases the review's 5-test list misses (wakeup invariant, recovery after full).
