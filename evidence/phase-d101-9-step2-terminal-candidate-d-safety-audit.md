# D101-9 Phase 9-B Step 2 — Candidate D Safety / Contract Audit

## 1. Verdict

**Candidate D (bounded Terminal + emergency synchronous destruction on full) = REJECTED**

Candidate D violates **I4_DESIGN_CONTRACT.md D15.2 (INV-X1-7)** and **D14.3**, which explicitly state that ownership disappearance is **only permitted by: Success / Superseded / ShutdownDiscard**. Synchronous destruction on epoch-unsafe ptr constitutes an **observable, non-supersedable ownership loss** — a P-4 violation and a terminal failure classified as **non-observable** (debug assert only).

The review's hypothesis ("EBR catastrophe ⇒ synchronous destruction safe") is **INVALID** — Terminal-full does not logically imply all readers are stuck on Terminal-epoch entries. The epoch-unsafe condition is per-entry, not global. Candidate A (caller retains ownership + backpressure) or Candidate B (shutdown-only bounded) must be pursued instead.

---

## 2. Latest-source evidence

### 2.1 Key source files

| Component | File | Key Lines |
|---|---|---|
| TerminalReclaimAuthority class | `src/audioengine/ISRRetireRouter.h` | 40-72 (class def), 287-297 (AdmissionState) |
| store() impl | `src/audioengine/ISRRetireRouter.cpp` | 27-53 |
| drain() impl | `src/audioengine/ISRRetireRouter.cpp` | 39-75 |
| drainAll() impl | `src/audioengine/ISRRetireRouter.cpp` | 77-98 |
| terminalReclaim() impl | `src/audioengine/ISRRetireRouter.cpp` | 490-511 |
| shutdownReclaim() impl | `src/audioengine/ISRRetireRouter.cpp` | 548-559 |
| drainAllQuarantineStore() | `src/audioengine/ISRRetireRouter.cpp` | 400-407 |
| enqueueWithRetry() Stage 5 | `src/audioengine/ISRRetireRouter.cpp` | 336-360 |
| signalDrainWakeup() | `src/audioengine/ISRRetireRouter.cpp` | 370-380 |
| waitForDrainSignalOrTimeout() | `src/audioengine/ISRRetireRouter.cpp` | 383-395 |
| recordWorldReclaim() | `src/audioengine/ISRRetireRouter.h` | 89-93 |
| ISRRetireRouter members | `src/audioengine/ISRRetireRouter.h` | 352-380 |
| RetireQuarantineStore | `src/audioengine/RetireQuarantineStore.h` | 76-150 (kMaxQuarantinedEntries=512) |
| DeferredDeletionQueue | `src/DeferredDeletionQueue.h` | 262 (kQueueSize=4096) |
| ReleaseResources shutdown | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | 188-560 |
| isFullyDrained() | `src/audioengine/AudioEngine.Threading.cpp` | 114-180 |
| waitForDrain() | `src/audioengine/AudioEngine.Threading.cpp` | 182-200 |
| tryShutdownQuiescentReclaim | `src/audioengine/AudioEngine.h` | 4369-4390 |
| enqueueDeferredDeleteNonRtWithResult | `src/audioengine/AudioEngine.h` | 4201-4258 |
| closeAdmission() impl | `src/audioengine/ISRShutdown.cpp` | 415-422 |
| joinProducers() impl | `src/audioengine/ISRShutdown.cpp` | 436-441 |
| isAdmissionOpen() / admissionState() | `src/audioengine/ISRShutdown.cpp` | 444-455 |
| RuntimeHealthMonitor | `src/audioengine/RuntimeHealthMonitor.h/.cpp` | h:1-108, cpp:1-30 (tick), 775-860 (checkOverflowRate) |
| executeRecoveryAction | `src/audioengine/AudioEngine.Timer.cpp` | 1713-1780 |
| D14/D15 design contract | `doc/work88/I4_DESIGN_CONTRACT.md` | 130-205 |

### 2.2 Key design documents

| Document | Section | Key Content |
|---|---|---|
| I4_DESIGN_CONTRACT.md | D14.3 (p.155) | budget 枯渇 → backpressure (NOT terminal-failure/destruction) |
| I4_DESIGN_CONTRACT.md | D14.3 (p.158) | non-supersedable obligation の terminal-failure 消失は許さない |
| I4_DESIGN_CONTRACT.md | D15.2 (p.202) | INV-X1-7: "A logical obligation may disappear only by: Success / Superseded / ShutdownDiscard" |
| I3_DESIGN_CONTRACT.md | D11.4 (p.317) | I3 version still allows "explicitly observable terminal failure policy" — but I4 D15.2 removes this |
| Practical Stable ISR Bridge Runtime.md | §10 (p.536) | "Overflowしても失われない" — overflow must not lose data |
| I4_DESIGN_CONTRACT.md | D15.2 | ownership conservation equation: transportCount + durableCount + buildingCount + stalledCount + supersededCount + shutdownDiscardCount == admittedLogicalObligationCount |

---

## 3. Terminal full preconditions — exact code trace

### 3.1 Terminal full ⇔ D=full ∧ Q=full ∧ E=full ∧ Terminal=size=K

```text
Terminal full
    ⇓ (store() would need to return false — current impl ALWAYS returns true)
D full?        ← DeferredDeletionQueue::enqueueRetireTyped() returns false (ring full)
Q full?        ← RetireQuarantineStore::quarantine() returns false (size_ >= 512)
E full?        ← RetireQuarantineStore::quarantine() returns false (size_ >= 512)
Terminal full? ← entries_.size() >= K_terminal (only possible with bounded array)
```

### 3.2 Code trace for Terminal full conditions

**Stage 1 (D):** `enqueueRetire()` → `provider_->enqueueRetireTyped()`:
- Returns `false` when `DeferredDeletionQueue` ring buffer is full (4096 entries)
- `enqueueRetire()` returns `RetireEnqueueResult::QueuePressure`

**Stage 2 (retry):** `tryReclaim()` + `drainEmergencyAndTerminal()`:
- Attempts to free entries in D, E, Terminal via epoch-gated drain
- If drain fails to free enough space, re-enqueue to D still fails

**Stage 3 (Q):** `m_retireQuarantine.quarantine()`:
- Returns `false` when `size_ >= kMaxQuarantinedEntries (512)`
- See `RetireQuarantineStore.h:82-87`

**Stage 4 (E):** `m_emergencyQuarantine.quarantine()`:
- Same class, same `kMaxQuarantinedEntries=512`
- Returns `false` when full

**Stage 5 (T):** `terminalReclaim()` → `m_terminalReclaim.store()`:
- Currently: ALWAYS returns `true` (std::vector push_back)
- With bounded: returns `false` when `entries_.size() >= K_terminal`

### 3.3 activeReaderCount and minReaderEpoch at Terminal full

When Terminal is full:
- **`activeReaderCount()`**: Could be > 0 (stuck reader) OR == 0 (all readers advanced). Terminal-full does NOT uniquely determine reader state.
- **`minReaderEpoch()`**: Could be stale (low) if a reader is stuck at old epoch. But other readers may have advanced.

**Critical insight:** Terminal-full means D+Q+E are all full, which means 5120+ entries are epoch-unsafe. But epoch-unsafe means `epoch < minReaderEpoch` — this is a **per-entry** condition, not a global one. Some entries may have `epoch < minReaderEpoch` and could be drained, but new entries with the current epoch cannot be drained until the reader advances.

### 3.4 Does Terminal-full imply all readers are stuck on this ptr?

**NO.** The code provides no such guarantee:

- `activeReaderCount() == 0` is a separate condition checked in `isFullyDrained()` and `releaseResources`:
  ```cpp
  // AudioEngine.Processing.ReleaseResources.cpp:482
  if (m_retireRouter->activeReaderCount() == 0)
      m_retireRouter->drainAllQuarantineStore();
  ```
- But in normal operation, `enqueueWithRetry()` does NOT check `activeReaderCount()` before escalating to Terminal.
- A reader could be active (epoch advanced) but the ptr's epoch is in the future relative to `minReaderEpoch`.

**Example scenario:**
1. Reader A is at epoch 10 (stuck)
2. Reader B is at epoch 100 (normal)
3. `minReaderEpoch()` = 10
4. A ptr with epoch 50 is enqueued → `isOlder(50, 10) = false` → epoch UNSAFE → stored
5. Reader B continues advancing, but Reader A is stuck
6. Terminal fills up with epoch-unsafe entries (all epoch > 10 but < 100)
7. All these entries are epoch-unsafe relative to `minReaderEpoch=10`
8. But Reader A being stuck is the root cause — NOT Terminal being full

**Terminal-full does NOT imply all readers are stuck on all Terminal entries.** It implies some reader(s) are stuck at a low epoch, preventing any entry with `epoch >= minReaderEpoch` from being reclaimed.

**Conclusion for D2-1:** "Terminal full ⇒ active reader references ptr" is **NOT PROVABLE**. Terminal-full implies some reader is stuck, but not that every Terminal entry is referenced by a reader. Synchronous destruction of any epoch-unsafe entry risks UAF.

---

## 4. Epoch safety analysis — 4-case matrix

### 4.1 terminalReclaim() dispatch logic (ISRRetireRouter.cpp:490-511)

```cpp
bool ISRRetireRouter::terminalReclaim(void* ptr, void (*deleter)(void*), uint64_t epoch,
                                      DeletionEntryType type, const char* reason) noexcept
{
    const uint64_t minReader = minReaderEpoch();
    const bool epochSafe = ISRRetireRouter::isOlder(epoch, minReader);
    const bool isRt = convo::numeric_policy::isAudioThread();

    if (epochSafe && !isRt)
    {
        deleter(ptr);                          // synchronous destruction
        if (type == DeletionEntryType::World)
            m_terminalReclaim.recordWorldReclaim();
        return true;
    }
    return m_terminalReclaim.store(ptr, deleter, epoch, type, reason);
}
```

### 4.2 4-case matrix

| Case | epoch safe | RT caller | Current behavior | Candidate D full behavior |
|---|---|---|---|---|
| A | Yes | No | Synchronous destroy (deleter), recordWorldReclaim, return true | **Same** — no change needed (already synchronous) |
| B | Yes | Yes | `store()` — Terminal store | **Cannot occur** — `enqueueWithRetry` asserts `!isAudioThread()`, and `retireRT` only hits D (not T) |
| C | No | No | `store()` — Terminal retained | **Candidate D: synchronous destroy** ← **CRITICAL** |
| D | No | Yes | `store()` — Terminal retained | **Cannot occur** — RT callers never reach Terminal |

### 4.3 Case C — THE CRITICAL ANALYSIS

**Case C:** `epoch < minReaderEpoch` = false (NOT older) = **epoch unsafe** + Non-RT caller.

This means: at least one active reader has `readerEpoch >= epoch` (i.e., the reader entered a critical section at or after the ptr's retire epoch). The ptr **MAY still be referenced** by that reader.

**Current behavior:** Store in Terminal, wait for epoch to become safe (reader advances → minReaderEpoch increases → `isOlder(epoch, minReaderEpoch)` becomes true → drain()).

**Candidate D proposed:** Synchronous destruction (`deleter(ptr)`).

### 4.4 UAF proof for Case C synchronous destruction

`isOlder(epoch, minReaderEpoch) == false` means `static_cast<int64_t>(epoch - minReaderEpoch) >= 0`, i.e., `epoch >= minReaderEpoch`.

This means there exists at least one reader with `readerEpoch == minReaderEpoch`. If `epoch >= minReaderEpoch`, it's possible that:
- `epoch == minReaderEpoch`: the reader is AT the same epoch as the ptr being retired. The reader may have entered its critical section with a reference to the ptr.
- `epoch > minReaderEpoch`: the reader's epoch is older, meaning the reader started before the ptr was retired. The reader holds a reference to the ptr.

In BOTH cases, the ptr **may be referenced by an active reader**. Destroying it synchronously = **UAF**.

**Code evidence:**
- `EpochDomain::getMinReaderEpoch()` returns the minimum epoch across all active readers
- `isOlder(a, b) = static_cast<int64_t>(a - b) < 0` — if false, `a >= b`, meaning some reader has epoch `b` and the entry has epoch `a >= b`
- The reader with epoch `b` started its critical section at epoch `b` and may still be referencing objects from that era

**Conclusion for Case C:** Synchronous destruction is **NOT SAFE**. Candidate D is **INVALID** for Case C.

### 4.5 Case B and Case D — RT reachability

**Case B:** `epochSafe && isRt` — RT caller reaching Terminal.
- `enqueueWithRetry()` has `jassert(!isAudioThread())` at line 292
- `retireRT()` (ISRRetireRouter.h:211) → `enqueueRetire()` → only hits D (lock-free), never Q/E/T
- So Case B cannot reach Terminal via production code.

**Case D:** `!epochSafe && isRt` — RT caller reaching Terminal.
- Same: RT callers only use `retireRT()` → D. Never reach Terminal.
- So Case D cannot occur in production.

**Conclusion:** Only Cases A and C are reachable. Case A is already synchronous (no change needed). Case C is **epoch-unsafe** and **synchronous destruction is UAF**. **Candidate D FAILS for Case C.**

---

## 5. D2-1 Verdict: INVALID

> **Claim:** "Terminal full は「EBR catastrophe state」とみなせる。" / "active reader が ptr を参照している可能性がない"

**Verdict: INVALID**

Terminal-full does NOT imply:
1. All readers are stuck → epoch may advance after Terminal-full occurs
2. All Terminal entries are epoch-safe → entries have different epochs; some may be safe, some unsafe
3. Synchronous destruction is safe → Case C (epoch unsafe + Non-RT) is reachable and UAF-prone

**The logical chain "Terminal full ⇒ EBR catastrophe ⇒ synchronous destruction safe" contains a non-sequitur.** Even if some readers are stuck (causing D/Q/E to fill), the specific ptr in question may be epoch-unsafe due to those stuck readers. Synchronous destruction of an epoch-unsafe ptr = UAF.

**I4 D15.2 (INV-X1-7) is clear:** "A logical obligation may disappear only by: Success, explicit Superseded decision, ShutdownDiscard." Synchronous destruction does not fall into any of these categories. It is a **terminal failure** classified as **non-observable** (debug assert only), per D14.3.

**Candidate D = REJECTED.**

---

## 6. Candidate A — Bounded Terminal + Caller Retains Ownership

### 6.1 Approach

```text
Terminal full
    ↓
store() returns false
    ↓
terminalReclaim() returns false
    ↓
enqueueWithRetry() Stage 5: caller retains ownership
    ↓
Caller retries or escalates
```

### 6.2 P-4 compatibility analysis

**Problem:** P-4 states "TerminalReclaimAuthority ALWAYS accepts — ownership always transfers." If `store()` returns false, this invariant is broken.

**But:** The review's census already notes this: "This breaks P-4 invariant." The question is WHETHER breaking P-4 is acceptable.

**I4 D15.2 analysis:**
- D15.2's conservation equation: `transportCount + durableCount + buildingCount + stalledCount + supersededCount + shutdownDiscardCount == admittedLogicalObligationCount`
- If Terminal full and caller retains ownership, the ptr is in `caller` state — NOT in any storage. This means the ptr is **in transit** (not at rest).
- The conservation equation does NOT account for "in-transit" state. But the review's Candidate A adds retry/backpressure, meaning the ptr will eventually be placed in D/Q/E/T or destroyed at shutdown.

**Key insight:** The ptr is Non-RT-owned during the retry period. The caller (Non-RT) must eventually either:
- Retry the chain (D→Q→E→T) — ptr gets placed somewhere
- Fail all retries — ptr must be handled (escalation)

**Risk:** If the caller cannot retry (e.g., one-shot call from `enqueueDeferredDeleteNonRt`), the ptr could leak.

### 6.3 Caller analysis

All callers of `terminalReclaim()` / `shutdownReclaim()`:

1. **`enqueueWithRetry()` (ISRRetireRouter.cpp:344):**
   - Sets `result = RetireEnqueueResult::TerminalReclaim` if `store()` succeeds
   - If `store()` fails (Candidate A), `tstored = false` → result is NOT set to TerminalReclaim
   - Signal: `signalDrainWakeup()` is called BEFORE `return result` (line 355-357)
   - **Problem:** If `store()` fails, the ptr is still owned by the caller. The caller returns `result` which is... what? The code sets `result = RetireEnqueueResult::TerminalReclaim` only if `tstored` is true (it's cast to void). Actually looking more carefully:

```cpp
const bool tstored = terminalReclaim(ptr, deleter, epoch, type, "enqueueWithRetry:TerminalReclaim");
(void)tstored;  // ignored!
result = RetireEnqueueResult::TerminalReclaim;  // always set regardless of tstored
```

**CURRENT CODE BUG:** `tstored` is cast to void and ignored. `result` is ALWAYS set to `TerminalReclaim`, even if `store()` returned false. This means:
- If `store()` returns false (Candidate A), `enqueueWithRetry()` returns `TerminalReclaim` as success, but the ptr is still owned by the caller!
- The caller (`enqueueDeferredDeleteNonRtWithResult`) receives `TerminalReclaim` and treats it as success (ownership transferred).
- **The ptr is LEAKED** — caller thinks it's owned, Terminal doesn't have it.

This is a **critical finding**: The current code has a latent bug where `tstored` is ignored. This means Candidate A requires fixing this bug: if `terminalReclaim()` returns false, `enqueueWithRetry()` must return `Shutdown` or `QueuePressure` (something that tells the caller "ptr not stored, you still own it").

2. **`shutdownReclaim()` (ISRRetireRouter.cpp:548-559):**
   - Returns `terminalReclaim(...)` directly
   - If false, returns false → caller (`enqueueDeferredDeleteNonRtWithResult`) checks:
   ```cpp
   const bool transferred = m_retireRouter->shutdownReclaim(ptr, deleter, epoch, type);
   return transferred ? RetireEnqueueResult::Success : RetireEnqueueResult::Shutdown;
   ```
   - If `shutdownReclaim` returns false → `Shutdown` → caller treats as "ptr not stored"
   - But the comment says "shutdown 中は Audio Thread 停止済みのため epoch は安全 → 即時破棄される" — implying it expects true.
   - If false, caller gets `Shutdown` and `enqueueDeferredDeleteNonRt` returns `false` (not `RetireEnqueueResult::Shutdown` check).

### 6.4 Backpressure feasibility

Candidate A requires the caller to retry. But:
- `enqueueWithRetry()` has a retry loop (Stage 2, kMaxRetry=2) for D, but NOT for Terminal full
- If Terminal is full after K_maxRetry=2 attempts, the caller has no more retries
- The caller (`enqueueDeferredDeleteNonRtWithResult`) can escalate to `drainDeferredRetireQueues(false)` and retry again

**Feasibility: LOW** without significant code changes. The retry/backpressure plumbing does not exist end-to-end.

### 6.5 B1-B7 analysis for Candidate A

| Obligation | Candidate A analysis |
|---|---|
| B1 (finite capacity) | ✅ `std::array<Entry, K_terminal>` satisfies |
| B2 (ownership continuity) | ✅ Caller retains on false → ptr ∈ caller (explicit state) |
| B3 (no double ownership) | ✅ store() succeeds OR caller retains — never both |
| B4 (epoch safety) | ✅ No change to drain() — epoch-gated drain preserved |
| B5 (shutdown drain) | ✅ drainAll() unchanged |
| B6 (wakeup) | ⚠️ Need to decide: full → wakeup? Reviewer says yes (Coordinator should drain). But if caller retains, ptr is still in flight. |
| B7 (World conservation) | ✅ DeletionEntryType::World accounting unchanged |
| P-4 compat | ❌ BROKEN — store() can return false |
| OOM avoidance | ✅ Fixed capacity |
| UAF risk | ✅ No synchronous destruction |
| Backpressure complexity | ⚠️ HIGH — needs retry/backpressure plumbing |
| HealthMonitor compat | ✅ Can add EVENT_TERMINAL_OVERFLOW |

---

## 7. Candidate D — Bounded + Emergency Synchronous Destruction

### 7.1 Approach

```text
Terminal full
    ↓
store() returns false
    ↓
terminalReclaim() performs synchronous destruction
    ↓
HealthMonitor escalation
```

### 7.2 B1-B7 analysis for Candidate D

| Obligation | Candidate D analysis |
|---|---|
| B1 (finite capacity) | ✅ `std::array<Entry, K_terminal>` satisfies |
| B2 (ownership continuity) | ❌ BROKEN — ptr destroyed while epoch-unsafe = ownership lost (not in caller/D/Q/E/T) |
| B3 (no double ownership) | ✅ N/A (ptr destroyed, not copied) |
| B4 (epoch safety) | **❌ VIOLATED** — synchronous destruction of epoch-unsafe ptr = UAF |
| B5 (shutdown drain) | ✅ drainAll() unchanged |
| B6 (wakeup) | ✅ signalDrainWakeup() called |
| B7 (World conservation) | ✅ World accounting preserved (recordWorldReclaim still called) |
| P-4 compat | ❌ BROKEN — ownership discontinuity (ptr annihilated) |
| OOM avoidance | ✅ Fixed capacity |
| **UAF risk** | **❌ CRITICAL** — Case C (epoch-unsafe) is reachable |
| Backpressure complexity | ✅ None (synchronous) |
| HealthMonitor compat | ✅ Can escalate |

### 7.3 Verdict: REJECTED

Candidate D violates B2 (ownership continuity), B4 (epoch safety), and P-4. The "EBR catastrophe" justification is a non sequitur — Terminal-full does not imply all entries are epoch-safe to destroy.

---

## 8. Candidate B — Normal Growable / Shutdown Bounded

### 8.1 Approach

```text
Normal operation: std::vector (growable, P-4 preserved)
Shutdown: drainAllQuarantineStore() → all Terminal entries destroyed unconditionally
```

### 8.2 B1-B7 analysis for Candidate B

| Obligation | Candidate B analysis |
|---|---|
| B1 (finite capacity) | ⚠️ CONDITIONAL — only during shutdown. Normal operation remains unbounded. |
| B2 (ownership continuity) | ✅ P-4 preserved (store() always true in normal op) |
| B3 (no double ownership) | ✅ Preserved |
| B4 (epoch safety) | ✅ drain() epoch-gated preserved; drainAll() is shutdown-only (readers joined) |
| B5 (shutdown drain) | ✅ drainAll() already handles this |
| B6 (wakeup) | ✅ Preserved (no change to signalDrainWakeup) |
| B7 (World conservation) | ✅ Preserved |
| P-4 compat | ✅ Fully preserved |
| OOM avoidance | ⚠️ Normal operation still unbounded (growable vector) |
| UAF risk | ✅ No synchronous destruction of epoch-unsafe entries |
| Backpressure complexity | ✅ None |
| HealthMonitor compat | ✅ Can add normal-op overflow monitoring |

### 8.3 Verdict: CONDITIONAL PASS (for shutdown safety only)

Candidate B does NOT fully satisfy B1 (finite capacity in normal operation). But it provides a **bounded shutdown** — which is the critical safety property for `K_world < ∞` proof. The `drainAllQuarantineStore()` at line 482 already destroys all Terminal entries when `activeReaderCount() == 0`.

**However:** During normal operation, Terminal remains growable. If readers are persistently stuck, memory grows unbounded. This is an **operational concern** (health monitoring/backpressure needed) but NOT a **correctness/conservation violation** — P-4 and D15.2 are preserved.

---

## 9. B1-B7 Matrix Comparison

| 項目 | A (bounded + caller retains) | D (bounded + sync destroy) | B (growable normal / bounded shutdown) |
|---|---|---|---|
| **B1** finite capacity | ✅ Always bounded | ✅ Always bounded | ⚠️ Normal: unbounded / Shutdown: bounded |
| **B2** ownership continuity | ✅ Caller retains (`ptr ∈ caller`) | ❌ **BROKEN** — ptr annihilated | ✅ Preserved (P-4) |
| **B3** no double ownership | ✅ store→true OR caller retains | ✅ (ptr destroyed) | ✅ Preserved |
| **B4** epoch safety | ✅ No change to drain | ❌ **VIOLATED** — UAF on Case C | ✅ Preserved |
| **B5** shutdown drain | ✅ drainAll preserved | ✅ drainAll preserved | ✅ drainAll preserved |
| **B6** wakeup (8R-1) | ⚠️ Need design (full→wakeup?) | ✅ signalDrainWakeup in tryReclaim path | ✅ No change |
| **B7** World conservation | ✅ World accounting unchanged | ✅ World accounting preserved | ✅ Preserved |
| **P-4 compatibility** | ❌ Broken (store can return false) | ❌ Broken (ptr annihilated) | ✅ Fully preserved |
| **OOM avoidance** | ✅ Bounded | ✅ Bounded | ⚠️ Normal unbounded |
| **UAF risk** | ✅ None | ❌ **CRITICAL** — Case C reachable | ✅ None |
| **Backpressure complexity** | ⚠️ HIGH — needs full plumbing | ✅ None (immediate destroy) | ✅ None |
| **HealthMonitor compat** | ✅ Add overflow event | ✅ Escalate on destroy | ⚠️ Needs overload detection |

### Winner: Candidate B (Conditional Pass)

Candidate D is **REJECTED** (B2/B4/P-4 violations, UAF). Candidate A is **not viable without major backpressure plumbing**. Candidate B is the **only option that preserves all correctness invariants** while providing bounded shutdown safety.

---

## 10. P-4 Compatibility Analysis

### 10.1 P-4 invariant statement

From `ISRRetireRouter.h:15-25` (class doc):
> "★ P-4 (15-P-4): The pending list is GROWABLE (std::vector). This guarantees the authority ALWAYS accepts an entry — there is NO 'store full' failure path."

From `ISRRetireRouter.h:47`:
> "std::vector<Entry> entries_;  // ★ P-4: Growable store"

From `ISRRetireRouter.cpp:16`:
> "★ P-4 (15-P-4): entries_ is GROWABLE (std::vector). store() ALWAYS succeeds"

From `ISRRetireRouter.cpp:53`:
> "return true;  // ★ P-4: growable store — ALWAYS accepts"

### 10.2 I4 D14.3 / D15.2 violation

I4 D15.2 explicitly states INV-X1-7:
> "A logical obligation may disappear only by: Success, explicit Superseded decision, ShutdownDiscard. terminal failure is structurally unreachable (observable as debug assert + telemetry only)."

Candidate D's synchronous destruction of epoch-unsafe ptr = **terminal failure causing ownership disappearance** that is NOT Success/Superseded/ShutdownDiscard. This directly violates D15.2.

### 10.3 P-4 is a Design Inviolable — Not Negotiable

P-4 is documented as "★ P-4" (star-prefixed = architectural invariant). It appears in:
- Class-level doc comment (ISRRetireRouter.h:22-25)
- Member declaration (ISRRetireRouter.h:47)
- store() implementation (ISRRetireRouter.cpp:16, 53)
- enqueueWithRetry() (ISRRetireRouter.cpp:296-297)
- terminalReclaim() (ISRRetireRouter.cpp:505)
- shutdownReclaim() (ISRRetireRouter.cpp:548-559)

**Any design that breaks P-4 requires an explicit D14.3/D15.2 revision and user approval.** Candidate D has not undergone this process.

---

## 11. HealthMonitor Responsibility Analysis

### 11.1 HealthMonitor is Detection-Only (not Decision)

**Evidence:**
- `RuntimeHealthMonitor::tick()` (Timer.cpp:1193) — runs checks, emits HealthEvents
- `onHealthEvent()` (Timer.cpp:1565) — receives events, may trigger escalation
- `executeRecoveryAction()` (Timer.cpp:1714) — PolicyEngine's action executor (RecoveryAction::Throttle/Recover/Restore/Safe/Critical)
- `setActionCallback()` — PolicyEngine → AudioEngine action binding (HealthMonitor.h:193)

**Design pattern:** HealthMonitor → `emitOnTransition()` → HealthEvent → `onHealthEvent()` → PolicyEngine evaluates → `executeRecoveryAction()`. HealthMonitor DETECTS, PolicyEngine DECIDES.

### 11.2 Existing event for overflow

- `EVENT_OVERFLOW_RATE_WARNING (1012)` / `EVENT_OVERFLOW_RATE_CRITICAL (1013)` — tracks Q/E overflow rate via `m_overflowCountRef` → `ISRRetireRouter::m_overflowCount_` (line 239)
- `checkOverflowRate()` (RuntimeHealthMonitor.cpp:775-860) — monitors Q/E overflow rate per second
- **Does NOT monitor Terminal overflow** — no `terminalReclaimOverflowCount_` exists

### 11.3 HealthMonitor CANNOT perform synchronous destruction

- HealthMonitor detects via `tick()` (pull-based, Non-RT Timer thread)
- PolicyEngine decides recovery action (Throttle, Recover, Restore, Safe, Critical)
- **No RecoveryAction = "destroy ptr"** exists
- Recovery actions are: throttle admissions, drain/recover, safe mode, emergency drain
- Synchronous destruction of epoch-unsafe ptr is NOT a HealthMonitor/PolicyEngine action

### 4-1. Terminal-specific overflow event — current API

**Can existing API notify?** YES — `emitOnTransition()` + callback → `onHealthEvent()`. New event code `EVENT_TERMINAL_OVERFLOW (1014)` could be added.

### 4-2. New HealthMonitor API needed?

**Yes** for Candidate D: need API to (a) detect Terminal full, (b) trigger synchronous destruction. But since Candidate D is REJECTED, this is moot.

### 4-3. HealthMonitor detect or decide?

**Detect only.** `checkOverflowRate()` → `emitOnTransition()` → HealthEvent → `onHealthEvent()` → PolicyEngine. The PolicyEngine (not HealthMonitor) makes decisions. This confirms PASS-D: "HealthMonitor is Decision Authority になっていない" — ✅ HealthMonitor is NOT decision authority.

---

## 12. I4 / Practical Design Contract Audit

### 12.1 I4 D14.3 — budget 枯渇 = backpressure

> "budget 枯渇で reservation を取 得できない場合、**admission を BLOCK**（backpressure / upstream stall）。"

This confirms: Terminal full should cause **backpressure** (Candidate A), NOT synchronous destruction (Candidate D).

### 12.2 I4 D15.2 — ownership conservation

> `transportCount + durableCount + buildingCount + stalledCount + supersededCount + shutdownDiscardCount == admittedLogicalObligationCount`

Synchronous destruction = ownership disappears from equation → **VIOLATION**.

> "A logical obligation may disappear only by: Success, explicit Superseded decision, ShutdownDiscard."

Candidate D's destruction is none of these → **VIOLATION**.

### 12.3 I4 D15.2 — terminal failure

> "terminal failure は構造的非到達の観測可能化 に留め、disappearance 理由としては認めない"

Candidate D makes terminal failure OBSERVABLE (synchronous destruction) → **VIOLATION** of non-observability requirement.

### 12.4 Practical Stable ISR Bridge Runtime §10

> "Overflowしても失われない" (overflow should not lose data)

Candidate D destroys data on overflow → **VIOLATION**.

### 12.5 B2 ownership continuity verdict

I4 D14.3 + D15.2 establish: **ownership MUST NOT disappear except via Success/Superseded/ShutdownDiscard.** Candidate D's synchronous destruction violates B2. **Candidate D = REJECTED.**

---

## 13. Fixed-capacity storage options comparison

| Option | Description | drain() compaction | drainAll() | hole handling | lock scope | deleter outside mutex | resident counter | insertion | no alloc | no overwrite |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `std::array<Entry, 2048>` + occupied flag | ✓ (compact nulls) | ✓ (clear all) | ✓ (null after drain) | mutex | ✓ | ✓ (atomic) | O(n) scan | ✅ | ✅ |
| 2 | `std::array<Entry, 2048>` + free bitmap | ✓ (bitmap clear) | ✓ (clear all + bitmap) | ✓ (bitmap) | mutex | ✓ | ✓ (atomic) | O(1) find free | ✅ | ✅ |
| 3 | Fixed-capacity ring buffer | ⚠️ complex (holes) | ✓ (drain all) | ⚠️ overwrite risk | mutex | ✓ | ✓ (atomic) | O(1) | ✅ | ⚠️ must prevent overwrite |
| 4 | `std::array<Entry, 2048>` + size_ index | ✓ (linear compact) | ✓ (clear all) | ✓ (size controls) | mutex | ✓ | ✓ (atomic) | O(1) append | ✅ | ✅ |

**Recommendation: Option 4** (`std::array` + `size_` counter) — identical semantics to current `std::vector` with `size()`, just fixed capacity. `store()` checks `size_ >= K_terminal` → return false. `drain()` compacts like current code (overwrites nulls, reduces size_). `drainAll()` resets size_ to 0.

**Ring buffer (Option 3) is REJECTED** — overwrite semantics conflict with B3 (no double ownership) and is explicitly forbidden by the review.

---

## 14. K_terminal sizing

### 14.1 Current chain capacities

| Store | Capacity | Source |
|---|---|---|
| D (DeferredDeletionQueue) | 4096 | `DeferredDeletionQueue.h:262` (`kQueueSize = 4096`) |
| Q (RetireQuarantineStore) | 512 | `RetireQuarantineStore.h:79` (`kMaxQuarantinedEntries = 512`) |
| E (EmergencyQuarantineStore) | 512 | Same type, separate instance |
| **D+Q+E total** | **5120** | Sum |

### 14.2 K_terminal sizing analysis

**NOT 2048 (arbitrary).** Must be derived from:

1. **World outstanding upper bound**: From D2_IMPL_CHECKLIST (Design-27, 2026-08-15):
   - `LIVE distinct World identity ≤ 2` (CLOSED)
   - `retired World identity` = time-dependent, NOT statically bounded
   - **No `live in-flight world counter` exists** — `retiredWorldCount_` is monotonic (increment only)

2. **Maximum concurrent epoch-unsafe entries**: Bounded by `(activeReaderCount × retireRate × maxStuckDuration)` — not structurally enforceable.

3. **Empirical high-water mark**: No telemetry currently tracks Terminal peak occupancy. The existing tests use 50 entries (StuckReaderFallbackDrainTests:162).

### 14.3 K_terminal = 2048 rationale

Since `D+Q+E = 5120` already provides substantial buffering, and Terminal is the LAST stage (only reached when all else fails), K_terminal should be:
- **Sufficient** to absorb burst pressure during transient reader stalls
- **Bounded** to prevent OOM
- **Proportional** to D (which is the primary queue)

`K_terminal = 2048 = D/2`:
- Provides 2x D-capacity as overflow headroom
- Total chain capacity = 5120 + 2048 = 7168
- Fits in cache (2048 × sizeof(Entry) ≈ 2048 × 40 bytes = 80KB — exceeds L2 but acceptable for Non-RT)
- Matches D's order of magnitude (both are "large" buffers)

**Alternative:** If health monitoring shows peak Terminal occupancy is consistently < 100, K_terminal = 512 (same as Q/E) would suffice. But without empirical data, 2048 is the safe choice.

### 14.4 K_total calculation

```text
K_total = K_D + K_Q + K_E + K_T
        = 4096 + 512 + 512 + 2048
        = 7168
```

This means: `K_world ≤ A_max + 7168` (where A_max is the reservation budget from Phase 9-D).

---

## 15. Open proof obligations

| ID | Item | Status | Notes |
|---|---|---|---|
| D2-1 | Terminal full ⇒ EBR catastrophe ⇒ sync destroy safe | **FAILED** | Non-sequitur; Case C UAF proven |
| B1 | Finite capacity | OPEN | Needs bounded implementation (Candidate A or B) |
| B2 | Ownership continuity | OPEN | Candidate A: caller retains. Candidate D: VIOLATED |
| B4 | Epoch safety | OPEN | Candidate D: VIOLATED (Case C) |
| B6 | Wakeup on full | OPEN | Need to decide wakeup semantics for full path |
| P-4 | store() always accepts | OPEN | Candidate A/D both break P-4 — requires D14/D15 revision |
| K_terminal | Sizing rationale | OPEN | 2048 proposed; needs empirical validation |
| I4 D14.3 | Backpressure on budget exhaustion | OPEN | Only satisfied by Candidate A (backpressure) |
| I4 D15.2 | Ownership conservation | OPEN | Only satisfied by Candidate B (no destruction) |

---

## 16. Final recommendation

### PASS-A through PASS-G evaluation

| Criterion | Status | Notes |
|---|---|---|
| **PASS-A**: sync destruction epoch-safe proven | ❌ **FAIL** | Case C (epoch-unsafe + Non-RT) is reachable; UAF proven |
| **PASS-B**: ownership disposition unique on full | ❌ **FAIL** | Candidate D destroys ptr → ownership annihilated |
| **PASS-C**: B1-B7 proof strategy exists | ⚠️ **CONDITIONAL** | Only Candidate A has complete strategy; D rejected |
| **PASS-D**: HealthMonitor not decision authority | ✅ **PASS** | HealthMonitor = detect-only; PolicyEngine decides |
| **PASS-E**: 8R-1 wakeup maintainable | ✅ **PASS** | No change needed for wakeup (drain path unchanged) |
| **PASS-F**: fixed-capacity allocation-free | ✅ **PASS** | std::array = zero allocation |
| **PASS-G**: K_terminal=2048 rationale | ⚠️ **CONDITIONAL** | Provisional: 2048 = D/2, needs empirical validation |

### Overall verdict: **FAIL — Candidate D rejected**

Candidate D fails PASS-A (epoch safety), PASS-B (ownership continuity), and implicitly PASS-C (B1-B7 not satisfiable without violating B2/B4).

### Recommended path forward

1. **Implement Candidate A** (bounded Terminal + caller-retains ownership + backpressure)
   - Fix the `tstored` ignore bug in `enqueueWithRetry()` (line 345-347)
   - Add `store() == false` → return `RetireEnqueueResult::Shutdown` path
   - Add retry loop in `enqueueWithRetry()` that re-attempts D→Q→E→T on Terminal full
   - Add `EVENT_TERMINAL_OVERFLOW` to HealthMonitor
2. **If Candidate A backpressure is too complex**, fall back to **Candidate B** (shutdown-only bounded)
   - Keep Terminal growable for normal operation
   - Document that `K_world < ∞` is conditional on shutdown completion (existing `drainAllQuarantineStore` already handles this)
3. **Do NOT implement Candidate D** under any circumstances.

---

## 17. No-code-change confirmation

**Zero code changes in this step.** This is a safety audit only. The following files were READ but NOT modified:

- `src/audioengine/ISRRetireRouter.h`
- `src/audioengine/ISRRetireRouter.cpp`
- `src/audioengine/RetireQuarantineStore.h`
- `src/DeferredDeletionQueue.h`
- `src/audioengine/RuntimeHealthMonitor.h` / `.cpp`
- `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp`
- `src/audioengine/AudioEngine.Threading.cpp`
- `src/audioengine/AudioEngine.h`
- `src/audioengine/AudioEngine.Timer.cpp`
- `src/audioengine/ISRShutdown.cpp`
- `doc/work88/I4_DESIGN_CONTRACT.md`
- `doc/work88/I3_DESIGN_CONTRACT.md`
- `doc/Practical Stable ISR Bridge Runtime.md`
