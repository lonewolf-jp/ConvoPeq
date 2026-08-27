# D105 — Phase I Capacity / Liveness / Closure Proof Audit

**Type**: Read-only audit (no source/CMake/test/contract changes)
**Scope**: E/F (capacity derivation), G (backpressure liveness), H (episode closure linearization)
**Baseline**: `ConvoPeq.md` (generated from `src/` + `CMakeLists.txt`) + `doc/work88/I4_DESIGN_CONTRACT.md`
**Date**: 2026-08-27
**Verdict**: ⚠️ **NO-GO** — E, F, G, H all remain **UNPROVEN** against current code structure

---

## 0. Scope / Read-only Constraints

| Constraint | Status |
|---|---|
| production source = 0 changes | ✅ |
| test source = 0 changes | ✅ |
| CMake = 0 changes | ✅ |
| contract = 0 changes | ✅ |
| Phase I implementation = **禁止** (開始禁止) | ✅ |

---

## 1. Latest Source Baseline

### Source files traced (ConvoPeq.md line references):

| File (ConvoPeq.md) | Production Source | Key Lines |
|---|---|---|
| `ISRRuntimePublicationCoordinator.cpp` | `src/audioengine/ISRRuntimePublicationCoordinator.cpp` | `submitRecoveryRequest` (812-920), `takePendingRecoveryAdmission` (880-896), `settlePendingRecoveryAdmission` (900-935), `isFullyDrained` (506-526) |
| `ISRRuntimePublicationCoordinator.h` | `src/audioengine/ISRRuntimePublicationCoordinator.h` | `RecoveryIntent` (215-228), `PendingRecoveryAdmission` (654-681), `recoveryIntentQueue_` (637), counters (616-630) |
| `AudioEngine.h` | `src/audioengine/AudioEngine.h` | `submitRecoveryIntent` (4405-4455), `recoveryPending` (2671), `rebuildCV` (2668), `submitRecoveryRequest` (via bridge) |
| `AudioEngine.RebuildDispatch.cpp` | `src/audioengine/AudioEngine.RebuildDispatch.cpp` | `rebuildThreadLoop` recovery section (911-999) |
| `ISRShutdown.cpp/h` | `src/audioengine/ISRShutdown.cpp/h` | `tryAdmit/release/closeAdmission/outstanding` (426-540) |
| `ISRDSPHandle.h` | `src/audioengine/ISRDSPHandle.h` | `DSPHandle` struct (28-50), `MAX_DSP_SLOTS` (120) |
| `ISRDSPQuarantine.h/cpp` | `src/audioengine/ISRDSPQuarantine.{h,cpp}` | `quarantineHandle`, `residentCount`, `kMaxSlots` (34) |
| `RuntimeBuildTypes.h` | `src/audioengine/RuntimeBuildTypes.h` | `RuntimeBuildSnapshot` (37-54), `RuntimeBuildFingerprint` (38-46), `isRuntimeBuildSnapshotSealedAndCompatible` (301-337) |

### Current state snapshot (code-derived facts):

1. `recoveryIntentQueue_` is a **SPSC** `LockFreeRingBuffer<RecoveryIntent, 256>` (kRecoveryIntentQueueCapacity = 256, `.h:637`)
2. `PendingRecoveryAdmission` is a **single-slot struct** (no table) — `.h:682`
3. `DSPHandleRuntime::MAX_DSP_SLOTS = 256` — `.h:120` of ISRDSPHandle.h
4. `DSPQuarantineManager::kMaxSlots = 256` — ISRDSPQuarantine.h:40
5. `ShutdownRuntime::packedState_` reservation field: **24 bits** (`kReservationMask = 0x00FFFFFFu`) → max 16,777,215 — ISRShutdown.cpp:24
6. `kMaxRecoveryConsecutiveFailures = 4` — RebuildDispatch.cpp:974 (local constexpr)
7. `pendingIntentCount_` is a `std::atomic<uint64_t>` — counter, unbounded

---

## 2. E — kMaxLogicalRecoveryObligations Derivation

### Requirement (I4 D14.2)

> `kMaxLogicalRecoveryObligations` must be derived from code structure, not set arbitrarily.

### Current Code State

**No `kMaxLogicalRecoveryObligations` exists in production code.** The D104 report listed it as a candidate value of 32, but this was **unproven**.

### Upper Bound Analysis from Code Structure

#### 2.1 Transport Layer Bound

The only **bounded** transport container for Recovery Intent is:
- `recoveryIntentQueue_`: `LockFreeRingBuffer<RecoveryIntent, 256>` — capacity = 256

This is a **hard upper bound** on transport-resident recovery intents. Beyond 256, `submitRecoveryRequest` performs a rollback (fetchSub) and falls back to durable admission.

#### 2.2 Durable Layer Bound

**No bounded durable table exists.** Currently:
- `PendingRecoveryAdmission pendingRecoveryAdmission_` — **single slot** (`.h:682`)
- No `kMaxDurableRecoveryAdmissions` constant
- Single-slot overwrite: `submitRecoveryRequest` unconditionally overwrites `pendingRecoveryAdmission_` on queue-full (`.cpp:865-872`)

Per D18.1, this single-slot overwrite is a **NO-GO pattern** — it loses non-supersedable recovery obligations. The fix requires:
- Bounded durable table with `kMaxDurableRecoveryAdmissions` entries
- CoalesceIdentity search across table slots
- Stall when table full

#### 2.3 Quarantine Layer Bound

- `DSPQuarantineManager::kMaxSlots = 256` (ISRDSPQuarantine.h:40)
- `DSPHandleRuntime::MAX_DSP_SLOTS = 256` (ISRDSPHandle.h:120)
- DSP slots are the **source** of quarantine → recovery requests
- Maximum simultaneously quarantinable DSPs = 256 (all slots)

#### 2.4 Generation/Episode Bound

- `nextRecoveryIntentId_` is `std::atomic<uint64_t>` — unbounded (diagnostic sequence)
- **No `nextRecoveryGeneration_` counter exists** — `recoveryGeneration = intent.intentId` (CONFIRMED bug, `.cpp:867`)
- **No `RecoveryEpisodeId` counter exists** — 0 hits in production source
- RecoveryEpisodeId allocation timing: **NOT IMPLEMENTED** — no counter, no allocator

### Derivation Attempt

```
E_max = maximum concurrent RecoveryEpisodeId count
      = maximum concurrently quarantined DSP handles
      ≤ MAX_DSP_SLOTS = 256

O_max = maximum live non-coalesced obligations per episode
      = currently UNBOUNDED (single-slot overwrite = 1, but loses obligations)
      = Phase I target: requires bounded durable table
      ≤ kRecoveryIntentQueueCapacity = 256 (transport) + 1 (single durable slot)
      = 257 (current, but BROKEN — single-slot overwrite loses obligations)

kMaxLogicalRecoveryObligations (current) = undefined
kMaxLogicalRecoveryObligations (required) ≥ E_max × O_max
```

### Current Code Cannot Derive 32

| Source | Bound | Notes |
|---|---|---|
| `MAX_DSP_SLOTS` | 256 | Upper bound on quarantine source, not recovery obligations |
| `kRecoveryIntentQueueCapacity` | 256 | Transport capacity, not logical obligations |
| `ShutdownRuntime` reservation bits | 24-bit (16M) | Admission reservation, not recovery obligation bound |
| `PendingRecoveryAdmission` | 1 (single slot) | Not a table — currently broken |
| `nextRecoveryIntentId_` | unbounded (uint64) | Diagnostic, not bounded |

### Verdict: E = **FAIL**

The value `32` cannot be derived from current code structure. `MAX_DSP_SLOTS = 256` is the only hard upper bound in the system, but:
- It bounds quarantine source handles, not logical recovery obligations
- The current single-slot durable admission means `E_max × O_max` is **effectively unbounded** (single-slot overwrite loses obligations rather than bounding them)
- No `kMaxLogicalRecoveryObligations` constant exists
- No `RecoveryEpisodeId` counter or allocator exists to bound episodes

---

## 3. F — E_max / O_max Derivation

### Requirement (I4 D19.3)

> `E_max × O_max ≤ kMaxLogicalRecoveryObligations` must be proven from code structure.

### E_max: Maximum Concurrent RecoveryEpisodeId

**No RecoveryEpisodeId exists in production code.** The concept is design-only (I4_D13).

### Attempted Derivation from Code Structure

#### F.1 Quarantine Source Bound → E_max Upper Limit

`DSPQuarantineManager` quarantines DSP handles. Each quarantined DSP → one Recovery request.

- `DSPHandleRuntime::MAX_DSP_SLOTS = 256` (`.h:120`)
- All 256 slots could theoretically be quarantined simultaneously
- Each quarantine → `submitRecoveryIntent` → `submitRecoveryRequest`
- **E_max ≤ 256** (if each quarantined DSP has a distinct episode)

But with D18.1/D13, same-config quarantine should share an episode. Without `RecoveryEpisodeId` implementation, we cannot bound E_max structurally.

#### F.2 Episode Lifetime Bound

- No `RecoveryEpisodeId` counter exists
- No episode closure mechanism (`liveLogicalObligationCount == 0`) exists
- `currentBuildSnapshot_` is overwritten on each publish — no episode baseline tracking
- **Cannot derive** episode lifetime or concurrent episode count

#### F.3 Distinct Target per Episode Bound (O_max)

- No `SemanticRecoveryTarget` struct exists in production (only `RuntimeBuildSnapshot` fields scattered across `RuntimeBuildFingerprint` + `RuntimeBuildSnapshot`)
- No `ObligationDomains` derivation exists
- No comparison of "distinct semantic targets within an episode" exists
- Current code: single-slot overwrite means **only 1 obligation per episode** can exist (broken)
- **Cannot derive** O_max

### Attempted Numerical Bound

```
E_max:
  Upper bound from MAX_DSP_SLOTS = 256
  BUT: no episode sharing mechanism → each quarantine could spawn distinct episode
  IF episodes are per-config-lineage (D13):
    E_max ≤ number of distinct active config lineages
    = bounded by number of distinct sealed RuntimeBuildSnapshot
    ≤ 2 (at most 1 active + 1 pending, since RuntimeWorld is single-DSPCore: REPAIR_PLAN2.md:172)
    But this is an empirical argument, not code-derived.

O_max:
  Upper bound = distinct SemanticRecoveryTarget per episode
  = distinct ObligationDomains per episode
  ≤ 2^4 = 16 (4 domains: IR, Conv, EQ, Config — each present or absent)
  But this assumes ObligationDomains as a 4-bit field, which does NOT exist in production code.
  Without SemanticRecoveryTarget, O_max = 1 (single-slot, broken) or UNBOUNDED (overwrite)
```

### Verdict: F = **FAIL**

`E_max` and `O_max` cannot be derived from current code structure because:
1. `RecoveryEpisodeId` is not implemented — no episode identity or closure mechanism
2. `SemanticRecoveryTarget` / `ObligationDomains` are not implemented — no target identity comparison
3. `MAX_DSP_SLOTS = 256` provides an upper bound on quarantine sources but NOT on episode count
4. The single-slot overwrite pattern means the current code does NOT bound obligations — it loses them

---

## 4. Combined Capacity Proof

### E × F Combined

```
kMaxLogicalRecoveryObligations = E_max × O_max

Current code provides:
  E_max ≤ 256 (MAX_DSP_SLOTS, if no episode sharing)
  O_max = UNBOUNDED (single-slot overwrite loses obligations)

  → kMaxLogicalRecoveryObligations = 256 × UNBOUNDED = UNBOUNDED

Required by D18/D19:
  E_max × O_max ≤ 32

But 32 is NOT derivable from code — it is an arbitrary constant from the design doc.
The code has no constant, no counter, no structure that bounds this to 32.

Even if we accept E_max = 2 (single DSPCore world → at most 1 active + 1 pending):
  O_max = UNBOUNDED (no semantic target comparison)
  → Still cannot prove 2 × O_max ≤ 32 without implementing SemanticRecoveryTarget

The ONLY hard bounds in current code are:
  - kRecoveryIntentQueueCapacity = 256 (transport)
  - MAX_DSP_SLOTS = 256 (quarantine source)
  - ShutdownRuntime 24-bit reservation = 16,777,215
  - None of these bound logical recovery obligations
```

### Verdict: Combined E/F = **FAIL**

`kMaxLogicalRecoveryObligations = 32` cannot be proven from current code structure. The value is **arbitrary** (from D104 candidate). No code-level constant, counter, or structural bound supports 32. The current single-slot overwrite pattern does not bound obligations — it silently loses them, which is the opposite of a capacity proof.

---

## 5. G — Backpressure Liveness Proof

### Requirement (I4 D21.2)

> When budget is full, the system must make forward progress:
> - CoordinatorLoop stalls (parks obligation, does NOT discard)
> - Builder operates independently (does NOT wait on CoordinatorLoop)
> - Release happens → wake → retry succeeds
> - No circular wait

### G.1 Current Backpressure Mechanism — Code Trace

#### Step 1: Admission Path (submitRecoveryRequest)

```cpp
// ISRRuntimePublicationCoordinator.cpp:845-872
convo::fetchAddAtomic(pendingIntentCount_, std::uint64_t{1}, std::memory_order_release);  // reservation
if (recoveryIntentQueue_.push(intent)) {
    return true;   // ✅ transport: push success, reservation maintained
}

// queue full → rollback + durable fallback
convo::fetchSubAtomic(pendingIntentCount_, std::uint64_t{1}, std::memory_order_release);  // rollback reservation
convo::fetchAddAtomic(recoveryIntentDropCount_, ...);  // telemetry

pendingRecoveryAdmission_.state = PendingRecoveryAdmission::State::DurablePending;
pendingRecoveryAdmission_.reservationOwned = true;  // ✅ durable: 1 reservation maintained
```

**Key observation**: When transport (queue) is full, the code **rolls back** the transport reservation and **transfers** it to the single durable slot. No new reservation is acquired for the durable slot — the same logical obligation is retained.

#### Step 2: Admission Reservation (tryAdmit)

```cpp
// AudioEngine.h:4449
if (!shutdownRuntime_.tryAdmit(1))
    return;  // ❌ admission closed — no obligation created

{
    std::lock_guard<std::mutex> lock(rebuildMutex);
    recoveryPending = true;
}
rebuildCV.notify_all();  // ✅ wake Builder
```

**Key observation**: `tryAdmit(1)` is called AFTER `submitRecoveryRequest` succeeds. This is the **ShutdownRuntime admission reservation**, NOT a recovery-specific budget. It tracks "how many obligations are in-flight during shutdown" — `outstanding()` must reach 0 for `closeAdmission` → `joinProducers` to succeed.

#### Step 3: Wake Predicate (rebuildThreadLoop)

```cpp
// AudioEngine.RebuildDispatch.cpp:838-843
rebuildCV.wait(lock, [this] {
    return hasPendingTask
        || publishRetryReady
        || recoveryPending
        || convo::consumeAtomic(rebuildThreadShouldExit, std::memory_order_acquire);
});
```

**Key observation**: `recoveryPending` is checked **inside** `rebuildCV.wait()` predicate. It is set under `rebuildMutex` and read under the same mutex (line 841-843). This is a standard condition variable pattern — no lost wakeups.

#### Step 4: Build → Publish → Release (Builder)

```cpp
// AudioEngine.RebuildDispatch.cpp:996-999
enqueuePublicationIntentForRuntimeCommit(dspToCommit, recoveryGeneration, recoverySnapshot);
runtimePublicationBridge_.settlePendingRecoveryAdmission(false);  // ✅ release durable slot
recoveryConsecutiveFailures = 0;
```

```cpp
// ISRRuntimePublicationCoordinator.cpp:933-935
void RuntimeIntentCoordinator::settlePendingRecoveryAdmission(bool retry) noexcept {
    if (retry) { ... Building → DurablePending; return; ... }
    pendingRecoveryAdmission_ = PendingRecoveryAdmission{};  // ✅ clear
    convo::publishAtomic(recoveryAdmissionPending_, false, std::memory_order_release);
}
```

```cpp
// AudioEngine.h:4454
shutdownRuntime_.release(1);  // ✅ release ShutdownRuntime admission reservation
```

### G.2 Liveness Analysis

#### G.2.1 CoordinatorLoop Stalls (Does it block?)

**Current code**: `submitRecoveryRequest` does NOT have a stall mechanism. When both transport AND durable slot are full (single-slot occupied by a Building obligation), the code **blindly overwrites** (`pendingRecoveryAdmission_ = ...`):

```cpp
// ISRRuntimePublicationCoordinator.cpp:865-872
pendingRecoveryAdmission_.state = PendingRecoveryAdmission::State::DurablePending;
pendingRecoveryAdmission_.recoveryGeneration = intent.intentId;  // 🔴 overwrites existing
pendingRecoveryAdmission_.buildSource = buildSource;             // 🔴 overwrites existing
```

**This is NOT stalling** — it is **silent overwrite**. Per D18.1, this is a NO-GO. The correct behavior per D10 (Stalled) is: hold the obligation in a bounded stall set, do NOT discard, retry later.

**Liveness issue**: If the Builder is stuck (e.g., `kMaxRecoveryConsecutiveFailures` reached, or build always fails), the single durable slot stays in Building/DurablePending forever. New recovery requests **silently overwrite** — the original obligation is lost.

**G.2.1 Verdict**: ❌ **FAIL** — No stall mechanism exists. Single-slot overwrite silently loses obligations.

#### G.2.2 Builder Independence (Does Builder wait on CoordinatorLoop?)

**Current code**: The Builder loop (`rebuildThreadLoop`) has TWO independent consumption paths:

1. **Transport path** (line 39298): `while (auto recovery = runtimePublicationBridge_.popRecoveryRequest())`
   - Consumer = Builder Loop only
   - Producer = CoordinatorLoop (via `submitRecoveryRequest`)
   - SPSC — no lock contention

2. **Durable path** (line 39376): `while (auto recovery = runtimePublicationBridge_.takePendingRecoveryAdmission())`
   - Consumer = Builder Loop only
   - Producer = CoordinatorLoop (via `submitRecoveryRequest` durable fallback)
   - SPSC — single-slot, no lock

**Builder independence**: The Builder loop does NOT hold `rebuildMutex` during build/execute/publish. It only holds `rebuildMutex` during the initial `wait()` predicate check (line 38620-38636). After `recoveryPending = false` (line 39292), the lock is released and the Builder proceeds without any Coordinator dependency.

**Builder wake**: `recoveryPending = true` + `rebuildCV.notify_all()` in `submitRecoveryIntent` (`.cpp:903-907`). The CV wait predicate checks `recoveryPending` — no lost wakeup because CV notifies are checked against the predicate atomically under `rebuildMutex`.

**G.2.2 Verdict**: ✅ **PASS** — Builder is fully independent of CoordinatorLoop after wake. No circular dependency in the wake mechanism.

#### G.2.3 Release → Wake → Retry

**Current code flow**:

1. Builder completes build → `enqueuePublicationIntentForRuntimeCommit` (publish transport)
2. Builder: `settlePendingRecoveryAdmission(false)` → clears durable slot, sets `recoveryAdmissionPending_ = false`
3. Builder: `shutdownRuntime_.release(1)` — releases admission reservation
4. **Missing**: No `recoveryPending = true` + `notify_all()` after durable consumption completes
   - The durable consumption loop (line 39376) runs inline in `rebuildThreadLoop`
   - After the durable loop ends, control falls through to the next iteration of the outer `while(true)` loop
   - The outer loop's `rebuildCV.wait()` checks `recoveryPending` — but this is only set by `submitRecoveryIntent`
   - **If a NEW recovery arrives during/after durable consumption**, `submitRecoveryIntent` sets `recoveryPending = true` + `notify_all()` → Builder wakes → processes
   - **If no new recovery arrives**, Builder goes back to sleep — correct behavior (no obligation pending)

**G.2.3 Verdict**: ✅ **PASS** for current single-slot model — wake mechanism is correct. **BUT** this relies on the single-slot overwrite being acceptable (which D18.1 says is NO-GO). With a bounded table, the wake mechanism must wake on:
- New transport push (current: `recoveryPending = true` ✅)
- Durable admission (current: `recoveryPending = true` ✅)
- Stall retry (current: **no mechanism** ❌)

#### G.2.4 Bounded Retry

**Current code**: `kMaxRecoveryConsecutiveFailures = 4` (RebuildDispatch.cpp:974) — Builder breaks out of durable consumption loop after 4 consecutive failures. The durable obligation remains in `DurablePending` state (settle(true) was called). On next wake (new transport push or rebuild timer tick), the Builder retries.

**Retry trigger**: Currently, retry is triggered ONLY by:
1. `recoveryPending` set by new `submitRecoveryIntent`
2. `hasPendingTask` set by CoordinatorLoop
3. `rebuildThreadShouldExit`

**No dedicated retry timer exists**. If no new recovery/observe/publish arrives, a failed durable obligation **may not be retried** until the next external trigger.

**G.2.4 Verdict**: ⚠️ **CONDITIONAL** — Retry is bounded (kMaxRecoveryConsecutiveFailures=4) but **liveness depends on external wake triggers**. No dedicated retry timer for stalled/durable obligations.

#### G.2.5 Single-threaded Admission Authority

```cpp
// ISRRuntimePublicationCoordinator.cpp:806-808
//   Producer = CoordinatorLoop（本メソッドは submitRecoveryIntent ← QuarantineIntentHandler /
//   RecoveryIntentHandler〔dead code〕経由で CoordinatorLoop スレッド上でのみ呼ばれる）。
//   Consumer = Builder Loop（popRecoveryRequest / takePendingRecoveryAdmission）。
//   ⇒ SPSC 維持
```

**G.2.5 Verdict**: ✅ **PASS** — Single producer (CoordinatorLoop) confirmed. SPSC maintained.

### G.3 Liveness Summary

| Condition | Current Code | Verdict |
|---|---|---|
| CoordinatorLoop stalls (parks, no discard) | ❌ Overwrites instead of stalls | **FAIL** |
| Builder independent of CoordinatorLoop | ✅ SPSC, no lock after wake | **PASS** |
| Release → wake → retry | ✅ CV predicate + notify pattern | **PASS** (conditional) |
| Bounded retry | ✅ kMaxRecoveryConsecutiveFailures=4 | **PASS** (but liveness depends on external wake) |
| Single producer authority | ✅ CoordinatorLoop only | **PASS** |
| Stall retry trigger | ❌ No dedicated timer | **CONDITIONAL** |

### Verdict: G = **FAIL** (partially passing, but stall mechanism missing)

The current code **passes** on Builder independence, wake mechanism, and retry bounding. However:
1. **No stall mechanism** — single-slot overwrite silently loses obligations (D18.1 NO-GO)
2. **No dedicated retry timer** for durable/stall obligations — liveness depends on external wake triggers
3. With a bounded durable table (required fix), the wake/retry mechanism must be re-validated

---

## 6. H — Episode Closure Linearization Proof

### Requirement (I4 D20 / D17)

> `D10.5`: For each RecoveryEpisodeId E:
> 1. admission, coalesce, and terminal disposition are totally ordered
> 2. `liveLogicalObligationCount: 1 → 0` is the unique closure point
> 3. After closure, no create/coalesce/supersede for E
> 4. Admission already linearized before closure remains valid
> 5. Admission not yet linearized at closure is rejected
> 6. RecoveryEpisodeId is never reused

### H.1 Current State — NO Episode Closure Mechanism

**Critical finding**: `RecoveryEpisodeId` does not exist in production code. There is:
- **No episode concept**
- **No episode closure mechanism**
- **No `liveLogicalObligationCount` counter**
- **No `Closed` flag**
- **No episode ID allocation**

The closest construct is `pendingRecoveryAdmission_` — a single-slot state machine with:
- States: `NoAdmission / DurablePending / Building`
- **No Episode field**
- **No generation/episode binding**

### H.2 Case Analysis (Against Hypothetical Implementation)

Since `RecoveryEpisodeId` doesn't exist, we analyze what the **required implementation** would need, and whether the **existing infrastructure** can support it.

#### Case A: Admission vs. last Success

**Required**: `liveLogicalObligationCount` decrement on Success must linearize before new admission can observe `live == 0`.

**Current code**:
```cpp
// Builder: settle(false) → pendingRecoveryAdmission_ = {} (clear)
void settlePendingRecoveryAdmission(bool retry) noexcept {
    if (retry) { ... return; ... }  // retry path: keep durable
    pendingRecoveryAdmission_ = PendingRecoveryAdmission{};  // ✅ clear on success
    convo::publishAtomic(recoveryAdmissionPending_, false, std::release);
}
```

After success: `recoveryAdmissionPending_ = false` (release). Next `submitRecoveryRequest` checks `recoveryAdmissionPending_` indirectly via `pendingRecoveryAdmission_.state != NoAdmission`. Since the slot is cleared, next admission can proceed.

**Analysis**: ✅ **PASS** for single-slot model. The `acquire`/`release` ordering on `recoveryAdmissionPending_` provides a happens-before relationship between Builder clear and Coordinator next-admit check.

**BUT**: This is a **single-slot** model, not multi-obligation per episode. No `liveLogicalObligationCount` exists.

#### Case B: Admission vs. last Superseded

**Required**: Superseded disposition must decrement `liveLogicalObligationCount`.

**Current code**: **Superseded does not exist.** `supersededCount` = 0 production hits. Phase I (per D18.2) has NO supersede — identical targets coalesce. So in Phase I, this case is vacuous (no superseded dispositions).

**Analysis**: ✅ **PASS** (vacuously true for Phase I — no supersede path)

#### Case C: Admission vs. ShutdownDiscard

**Required**: ShutdownDiscard must be observable and terminal.

**Current code**:
```cpp
// ISRRuntimePublicationCoordinator.cpp:910-916
void discardPendingRecoveryAdmission() noexcept {
    if (state != NoAdmission) {
        convo::fetchAddAtomic(recoveryShutdownDiscardCount_, ...);
        pendingRecoveryAdmission_ = PendingRecoveryAdmission{};
        convo::publishAtomic(recoveryAdmissionPending_, false, std::release);
    }
}
```

**Shutdown ordering** (from ReleaseResources.cpp:37810-37840):
1. `requestShutdown()` → `state_ = ShuttingDown` (acquire/release)
2. **stopRebuildThread()** — Builder thread joined (`jassert(!rebuildThreadIsRunning)`)
3. **discardRecoveryRequestsOnShutdown()** → drains transport queue, counts ShutdownDiscard
4. **discardPendingRecoveryAdmission()** → clears durable slot, counts ShutdownDiscard
5. `waitForDrain()` → polls `isFullyDrained()`

**Analysis**: ✅ **PASS** — ShutdownDiscard is observable (`recoveryShutdownDiscardCount_`), and the Builder is joined BEFORE discard, so no race between Builder completion and shutdown discard. The ordering is: `closeAdmission → join Builder → discard → drain check`.

#### Case D: Admission vs. Builder completion

**Required**: New admission after Builder completion but before episode closure must be rejected (if episode is closed).

**Current code**: There is NO episode closure in the current code. `pendingRecoveryAdmission_` is cleared on success (settle(false)), and the next `submitRecoveryRequest` can immediately write a new obligation. **No "closed" state exists.**

**Analysis**: ❌ **FAIL** — No episode closure mechanism. Admission can proceed immediately after Builder completion, with no episode boundary check.

#### Case E: Coalesce vs. episode closure

**Required**: Coalesce must operate within an open episode.

**Current code**: Coalesce does not exist. `submitRecoveryRequest` blindly overwrites. **No CoalesceIdentity search, no episode check.**

**Analysis**: ❌ **FAIL** — No coalesce mechanism, no episode association.

### H.3 Atomic Ordering Analysis

The I4 D20 design proposes:
```cpp
// Hypothetical — does NOT exist in current code
std::atomic<uint64_t> liveLogicalObligationCount;
std::atomic<bool> Closed;
```

**D20 linearization claims**:
- `closeAdmission()` CAS on `packedState_` (ShutdownRuntime) provides the closure point
- `tryAdmit()` CAS on same word provides admission linearization
- `liveLogicalObligationCount == 0` (atomic) combined with `Closed` (atomic) provides total order

**Current code analysis of atomic ordering**:

| Atomic Op | Memory Order | File:Line |
|---|---|---|
| `recoveryAdmissionPending_` write (settle) | `publish=release` | ISRRuntimePublicationCoordinator.cpp:935 |
| `recoveryAdmissionPending_` read (hasPending) | `consume=acquire` | ISRRuntimePublicationCoordinator.cpp:898 |
| `pendingIntentCount_` fetchAdd (reservation) | `release` | ISRRuntimePublicationCoordinator.cpp:845 |
| `pendingIntentCount_` fetchSub (rollback/release) | `release` | ISRRuntimePublicationCoordinator.cpp:861, 947 |
| `state_` write (ShuttingDown) | `release` | ISRRuntimePublicationCoordinator.cpp:505 |
| `state_` read (shutdown gate) | `acquire` | ISRRuntimePublicationCoordinator.cpp:825 |
| `recoveryAdmissionPending_` atomic | `release` (write), `acquire` (read) | ✓ |
| `pendingRecoveryAdmission_` (non-atomic struct) | **N/A** — plain struct, SPSC | ✓ safe under SPSC |

**Key issue**: The `pendingRecoveryAdmission_` struct is **non-atomic** (plain `PendingRecoveryAdmission`). It is safe ONLY because of SPSC (Producer = CoordinatorLoop, Consumer = Builder Loop). This is documented (`.h:682`: "SPSC（plain 構造体 — atomic 不要）").

**For Episode Closure (D20)**: The proposed `liveLogicalObligationCount` + `Closed` would need to be atomic and participate in the same CAS word as `tryAdmit`/`closeAdmission`. Currently, `ShutdownRuntime` has `packedState_` with a 24-bit reservation counter, but `RecoveryEpisodeId` and `liveLogicalObligationCount` are NOT part of it.

### H.4 Race / Interleaving Matrix

| Case | Scenario | Linearization Point | Race Risk | Current Code |
|---|---|---|---|---|
| A | Admission vs Success | `settle(false)`: clear struct + release `recoveryAdmissionPending_` | ✅ Safe (SPSC + release/acquire) | ✅ PASS |
| B | Admission vs Superseded | N/A (Phase I no supersede) | N/A | ✅ PASS (vacuous) |
| C | Admission vs ShutdownDiscard | `discardPendingRecoveryAdmission`: clear + release count | ✅ Safe (Builder joined first) | ✅ PASS |
| D | Admission vs Builder completion → episode close | No closure mechanism exists | ❌ Unsafe | ❌ FAIL |
| E | Coalesce vs episode closure | No coalesce exists | ❌ Unsafe | ❌ FAIL |
| F | Concurrent admission (multi-producer) | Single producer (CoordinatorLoop) | ✅ Safe (SPSC) | ✅ PASS |
| G | Admission after close | No close concept | ❌ Unsafe | ❌ FAIL |

### Verdict: H = **FAIL**

Episode closure linearization **cannot be proven** because:
1. `RecoveryEpisodeId` does not exist in production code
2. `liveLogicalObligationCount` does not exist
3. `Closed` flag does not exist
4. No episode lifecycle — `pendingRecoveryAdmission_` is cleared immediately on success with no episode boundary
5. No coalesce mechanism for D20 Case E
6. While the SPSC atomic ordering (A, C) is correct, F/D/G have no mechanism to prevent admission after "implicit closure"

---

## 7. Race / Interleaving Matrix (Complete)

| # | Race / Interleaving | Participants | Current Code Behavior | Verdict |
|---|---|---|---|---|
| R1 | Admission vs Builder settle(success) | CoordinatorLoop writes durable slot; Builder reads/clears | SPSC (no race), release/acquire on `recoveryAdmissionPending_` | ✅ Safe |
| R2 | Admission vs Builder settle(retry) | CoordinatorLoop overwrites durable; Builder in Building→DurablePending | SPSC, Builder only clears on retry=true (re-enters DurablePending) | ✅ Safe |
| R3 | Admission vs ShutdownDiscard | CoordinatorLoop submits; shutdown discards | submitRecoveryRequest checks `state_ == ShuttingDown` BEFORE any write (`.cpp:825`) | ✅ Safe |
| R4 | Builder settlement vs shutdown discard | Builder settle(false) clears slot; shutdown discard clears slot | Builder joined BEFORE shutdown discard (ReleaseResources.cpp ordering) | ✅ Safe |
| R5 | Transport push vs durable overwrite | CoordinatorLoop push fails → overwrite durable slot | Same thread (CoordinatorLoop) — sequential, no race | ✅ Safe |
| R6 | `recoveryPending` flag set vs CV wait | CoordinatorLoop sets flag+notify; Builder reads predicate | `rebuildMutex` protects flag, CV predicate checks under lock | ✅ Safe |
| R7 | `recoveryAdmissionPending_` read in isFullyDrained | Builder clears (release); isFullyDrained reads (acquire) | Release/acquire ordering on atomic bool | ✅ Safe |
| R8 | Episode closure vs new episode | N/A — no episode concept | N/A | ❌ Not applicable |
| R9 | Stalled admission retry trigger | CoordinatorLoop stalls; Builder releases | No stall mechanism; no retry timer | ❌ Unsafe |
| R10 | Shutdown admission vs ongoing build | `state_` ShuttingDown gate; Builder may be mid-build | Shutdown gate BEFORE reservation; Builder joined before discard | ✅ Safe |

---

## 8. Phase I / Phase II Boundary Recheck

### D104 Appendix B Issue

D104's Appendix B (R8-R10) references `canSupersede()` — **which is Phase II**. The implementation checklist must clearly separate:

### Phase I Required (from D104 + D18)

| Component | Status in Production | Phase I Requirement |
|---|---|---|
| `CoalesceIdentity` = {handle, RecoveryEpisodeId, SemanticRecoveryTarget} | ❌ MISSING (all 3) | MUST implement |
| `SemanticRecoveryTarget` (5 fields) | ❌ MISSING (struct + buildInputHash) | MUST implement |
| `semantic equality` (isSemanticTargetSuperset as equality) | ❌ MISSING | MUST implement |
| `RecoveryEpisodeId` (counter + allocation) | ❌ MISSING | MUST implement (D13 canonical) |
| `RecoveryGeneration` (dedicated counter) | ❌ MISSING (confused with intentId) | MUST implement (D8.2) |
| `reservation` (pendingIntentCount_ + recoveryAdmissionPending_) | ✅ Partially (single-slot) | MUST extend to bounded table |
| `Stalled` state | ❌ MISSING | MUST implement (D10) |
| `ownership conservation` (live + terminal = admitted) | ❌ MISSING (no successCount) | MUST implement (D18.3) |
| `kMaxLogicalRecoveryObligations` | ❌ MISSING | MUST define + prove bound |

### Phase II (Deferred)

| Component | Why Deferred |
|---|---|
| `canSupersede()` | D18.1: Phase I identical targets coalesce (Step 0); supersede is for different targets (Phase II) |
| `partial-order containment` | D18.2: Phase I uses equality; compositional superset requires semantic partial-order model |
| `compositional semantic superset` | D12.3: explicitly deferred from Phase I (correctness > efficiency) |
| `RecoveryProvenance` | Diagnostic only — not needed for Phase I correctness |

### D104 Appendix B R8-R10 Audit

D104 Appendix B lists:
- **R8**: `same handle` check — ✅ This is COALESCE identity, Phase I
- **R9**: `same epoch domain = same activation epoch` — ❌ **WRONG** (D13 corrected: use `RecoveryEpisodeId`, NOT epoch)
- **R10**: `isSemanticSuperset` — ❌ **WRONG** (D18.2: Phase I uses `isSemanticTargetSuperset` as equality, not superset)

**Issue**: D104 Appendix B R8-R10 are described in supersession language. They should be relabeled as:
- R8 → part of `CoalesceIdentity` (Phase I)
- R9 → replaced by `RecoveryEpisodeId` check (Phase I)
- R10 → `isSemanticTargetSuperset` as **equality** (Phase I), NOT superset

---

## 9. Remaining Ambiguities

| # | Ambiguity | Impact | Resolution Needed |
|---|---|---|---|
| A1 | `currentBuildSnapshot_` is mutex-guarded (NonRT only) — but `submitRecoveryRequest` is called from CoordinatorLoop (also NonRT). No RT path. ✅ | Low | Confirmed safe |
| A2 | No dedicated retry timer for durable obligations — relies on `recoveryPending` wake | Medium | Backpressure liveness (G) depends on this |
| A3 | `recoveryIntentDropCount_` (overflow telemetry) is NOT included in ownership conservation equation — it tracks transport overflow, not obligation disposition | Low | Correct — it's diagnostic, not ownership |
| A4 | `kMaxRecoveryConsecutiveFailures = 4` is local constexpr in RebuildDispatch.cpp — not accessible to tests | Low | Test access needed |
| A5 | No `RecoveryEpisodeId` closure — `pendingRecoveryAdmission_` cleared immediately on success | High | Episode closure (H) cannot be proven |
| A6 | Single-slot durable — no bounded table | High | Capacity bound (E/F) cannot be proven |
| A7 | `intentId` is used as `recoveryGeneration` — semantic conflation | High | Must separate (D8.3) |
| A8 | No `SemanticRecoveryTarget` struct — fields scattered across `RuntimeBuildFingerprint` + `RuntimeBuildSnapshot.convolverFingerprint` | High | Must consolidate + add `buildInputHash` |
| A9 | No `isDomainSuperset` / `isSemanticTargetSuperset` — `isRuntimeBuildSnapshotSealedAndCompatible` is field-by-field equality (not superset) | Medium | Phase I uses equality, but predicate name/struct missing |
| A10 | `DSPHandleRuntime` registry has no concurrent quarantine tracking API — `DSPQuarantineManager` tracks independently with mutex | Low | Separate concerns, not a capacity issue |
| A11 | `submitRecoveryIntent` calls `shutdownRuntime_.tryAdmit(1)` AFTER `submitRecoveryRequest` — if tryAdmit fails, the obligation exists in transport/durable but is NOT counted by ShutdownRuntime | Medium | Potential shutdown hang if obligation outlives admission reservation |

---

## 10. Final Verdict

### E: PASS/FAIL

| ID | Verdict |
|---|---|
| **E** | ❌ **FAIL** — `kMaxLogicalRecoveryObligations` cannot be derived from current code. No constant, no counter, no structural bound. `MAX_DSP_SLOTS=256` bounds quarantine sources, not logical obligations. Value 32 is arbitrary. |

### F: PASS/FAIL

| ID | Verdict |
|---|---|
| **F** | ❌ **FAIL** — `E_max` and `O_max` cannot be derived. No `RecoveryEpisodeId`, no `SemanticRecoveryTarget`, no `ObligationDomains`. Single-slot overwrite loses obligations (does not bound them). |

### G: PASS/FAIL

| ID | Verdict |
|---|---|
| **G** | ❌ **FAIL** — Stall mechanism does NOT exist (single-slot overwrite). No dedicated retry timer for durable obligations. Builder independence and wake mechanism pass, but the core stall/retry-on-release path is broken. |

### H: PASS/FAIL

| ID | Verdict |
|---|---|
| **H** | ❌ **FAIL** — No episode closure mechanism. No `RecoveryEpisodeId`, no `liveLogicalObligationCount`, no `Closed` flag. Atomic ordering is correct for single-slot SPSC but cannot support multi-obligation episode lifecycle. |

---

## 11. Phase I Implementation Gate

### Current Status: **NO-GO**

```text
D105 Gate Conditions:
┌─────────────┬──────┬─────────────────────────────────────────────────┐
| Condition   | Pass | Detail                                        |
├─────────────┼──────┼─────────────────────────────────────────────────┤
| E (capacity) |  ❌  | kMax=32 not derivable from code; no constant    |
| F (E/O bounds)| ❌  | E_max/O_max not derivable; no episode structure |
| G (liveness) |  ❌  | No stall mechanism; overwrite loses obligations  |
| H (closure)  |  ❌  | No episode closure; single-slot clear-on-success |
└─────────────┴──────┴─────────────────────────────────────────────────┘
```

### Required Resolution Before Implementation GO

1. **Implement `RecoveryEpisodeId`** — counter + episode lifecycle (D13)
2. **Implement `SemanticRecoveryTarget`** — 5-field struct + `buildInputHash` (D12.2)
3. **Implement `RecoveryGeneration` counter** — separate from `intentId` (D8.2)
4. **Replace single-slot with bounded durable table** — `kMaxDurableRecoveryAdmissions` + coalesce search
5. **Implement `Stalled` state** — bounded stall set with FIFO retry (D10)
6. **Implement `liveLogicalObligationCount` + `Closed` flag** — atomic, CAS-bound to admission
7. **Implement `successCount` + `admittedLogicalObligationCount`** — production counters (D18.3)
8. **Add `buildInputHash`** to `SemanticRecoveryTarget` — canonical hash of `BuildInput` (D12.2)
9. **Define `kMaxLogicalRecoveryObligations`** — derive 32 (or alternative) from `E_max × O_max ≤ kMax`

> **D105 が PASS して初めて D106（FINAL IMPLEMENTATION CONTRACT）へ進みます。**
> **D105 = NO-GO なため、production source への実装は引き続き禁止です。**

---

*D105 created by code-tracing ConvoPeq.md against I4_DESIGN_CONTRACT.md D18-D22.*
*Key source files: `ISRRuntimePublicationCoordinator.{h,cpp}`, `AudioEngine.h`, `AudioEngine.RebuildDispatch.cpp`, `ISRShutdown.{h,cpp}`, `ISRDSPHandle.h`, `ISRDSPQuarantine.{h,cpp}`, `RuntimeBuildTypes.h`*
