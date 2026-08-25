# D101-31-A — Caller Inventory Audit (Pre-Implementation)

> **Phase**: D101-31-A (read-only audit — NO code changes).
> Purpose: Enumerate all call sites and read/write sites before D101-30 contract is implemented.
> Prerequisite: D101-30 GO verdict.
> Code changes: PROHIBITED.

---

## 1. `admissionState_` — Full Read/Write Inventory

### Definition

```text
File: src/audioengine/ISRShutdown.h:353
std::atomic<AdmissionState> admissionState_{AdmissionState::Open};
```

### All production references (5 sites)

| File:Line | Operation | Context |
|---|---|---|
| ISRShutdown.cpp:419 | **write** (CAS Open→Closing) | `closeAdmission()` |
| ISRShutdown.cpp:440 | **write** (CAS Closing→Closed) | `joinProducers()` |
| ISRShutdown.cpp:447 | **read** (load == Open?) | `isAdmissionOpen()` |
| ISRShutdown.cpp:452 | **read** (load, return enum) | `admissionState()` |
| AudioEngine.Processing.ReleaseResources.cpp:196 | **comment only** | "Thread joins above do NOT update admissionState_" |

### All test references (0 production write sites outside ShutdownRuntime)

No test files reference `admissionState_` directly.

### Key observation

`admissionState_` is exclusively managed by `ShutdownRuntime`. No external class reads or writes it. This supports D101-30 Step 1 — `ShutdownRuntime` is the sole authority.

---

## 2. 3-Path Transaction Call Sites

### 2A. Publication Path

**Entry point chain**: `commitRuntimePublication()` → `enqueueRuntimePublicationFireAndForget()` → `runtimePublicationBridge_.enqueuePublicationIntent(intent)`

**Admission policy gate**: `PublicationAdmission::evaluate()` in `RuntimePublicationOrchestrator::trySubmitImpl()` (RuntimePublicationOrchestrator.cpp:37)

| Component | File:Line | Description |
|---|---|---|
| Policy evaluation | RuntimePublicationOrchestrator.cpp:37 | `admission_.evaluate(req, engine_, pubCtx)` → returns `Decision::Accepted` |
| Decision dispatch | RuntimePublicationOrchestrator.cpp:327 | `submitPublishRequest()` → `trySubmitImpl()` → switch on decision |
| Enqueue | AudioEngine.h:4569 | `runtimePublicationBridge_.enqueuePublicationIntent(intent)` |
| Enqueue impl | ISRRuntimePublicationCoordinator.h:351 | `CoordinatorState::ShuttingDown` gate + `publicationIntentResidencyCount_` reservation |
| Consumer | ISRRuntimePublicationCoordinator_ProcessIntent.cpp:128 | `processIntent()` — pops IntentType::Publish from intentQueue_, decrements residency count |

**tryAdmit/release placement for Publication**:
- `tryAdmit(1)`: After `evaluate()` returns `Accepted`, before `enqueuePublicationIntent`
- `release(1)`: After `enqueuePublicationIntent` returns (success or failure)

**Important**: `evaluate()` is the **policy** gate (health, pressure, generation staleness). `tryAdmit()` is the **shutdown** gate (atomic count on packedState_). These are distinct:
- `evaluate()` → per-request policy decision (can be RejectedStaleGeneration, RejectedPressure, etc.)
- `tryAdmit()` → shutdown-level reservation (only fails when AdmissionState != Open)

**Callers of `enqueueRuntimePublicationFireAndForget` / `commitRuntimePublication`**:
- `n()` — AudioEngine.h:4587 (primary publication entry point)
- `AudioEngine.Timer.cpp:994` — Timer publish
- `AudioEngine.Processing.PrepareToPlay.cpp:155,277` — PrepareToPlay publish
- `AudioEngine.RebuildDispatch.cpp:989,1062,1259` — Recovery rebuild publish
- `AudioEngine.Transition.cpp:25` — Transition publish

### 2B. Recovery Path

**Entry point chain**: `QuarantineIntentHandler` → `submitRecoveryIntent()` → `runtimePublicationBridge_.submitRecoveryRequest()` → `recoveryIntentQueue_.push()`

| Component | File:Line | Description |
|---|---|---|
| Shutdown gate | ISRRuntimePublicationCoordinator.cpp:875 | `CoordinatorState::ShuttingDown` check (returns false if shutting down) |
| Reservation | ISRRuntimePublicationCoordinator.cpp:892 | `pendingIntentCount_` fetchAdd (transport residency, NOT admission) |
| Enqueue | ISRRuntimePublicationCoordinator.cpp:894 | `recoveryIntentQueue_.push(intent)` |
| Rollback | ISRRuntimePublicationCoordinator.cpp:909 | `pendingIntentCount_` fetchSub (queue full) |
| Durable fallback | ISRRuntimePublicationCoordinator.cpp:917-929 | `pendingRecoveryAdmission_` (recoveryAdmissionPending_ = true) |
| Consumer | ISRRuntimePublicationCoordinator.cpp:982 | `popRecoveryRequest()` — decrements pendingIntentCount_ |
| Durable consumer | ISRRuntimePublicationCoordinator.cpp:953 | `discardPendingRecoveryAdmission()` / `takePendingRecoveryAdmission()` |

**tryAdmit/release placement for Recovery**:
- `tryAdmit(1)`: After `CoordinatorState::ShuttingDown` check, before `pendingIntentCount_` fetchAdd
- `release(1)`: After `recoveryIntentQueue_.push()` success OR after durable fallback set

**Important distinction**: `pendingIntentCount_` is transport residency (tracks intents in queue + producer reservation). `tryAdmit()` would add a **separate** shutdown-level admission count. These are NOT the same.

**Callers of `submitRecoveryRequest`**:
- `AudioEngine.h:4429` — `submitRecoveryIntent()` (inline function)
- `AudioEngine.Processing.PrepareToPlay.cpp` — (indirectly via rebuild)
- Various test files: ISRSoakTests.cpp (5 sites), ISRSemanticValidationTests.cpp (4 sites)

### 2C. Build/Rebuild Path

**Entry point**: `submitRebuildIntent()` → mutex-protected `rebuildAdmissionPendingIntent_` set

| Component | File:Line | Description |
|---|---|---|
| Shutdown gate | AudioEngine.RebuildDispatch.cpp:243 | `isShutdownInProgress()` check |
| Mutex | AudioEngine.RebuildDispatch.cpp:195 | `rebuildAdmissionIntentMutex_` (std::mutex) |
| Intent storage | AudioEngine.h:4260 | `rebuildAdmissionPendingIntent_` (single slot) |
| Consumer | ISRCoordinatorLoop.cpp | RebuildThread reads + clears intent |
| Defer path | AudioEngine.RebuildDispatch.cpp | `enqueueDeferred()` (Orchestrator) |

**tryAdmit/release placement for Build**:
- `tryAdmit(1)`: After `isShutdownInProgress()` check, before mutex acquire
- `release(1)`: After `rebuildAdmissionPendingIntent_` set (immediately after enqueue into intent slot)

**Important**: The Build path uses a mutex (`rebuildAdmissionIntentMutex_`). The `tryAdmit()` CAS must happen **outside** the mutex to maintain RT-safety for the atomic operation. The mutex is only for the `rebuildAdmissionPendingIntent_` slot, not for admission state.

**Callers of `submitRebuildIntent`**:
- AudioEngine.h:1376,1392,1402 — inline calls
- AudioEngine.Init.cpp:94
- AudioEngine.Parameters.cpp:160,171,216,240,259,273,287,343 — various parameter changes
- AudioEngine.Processing.PrepareToPlay.cpp:291
- AudioEngine.StateIO.cpp:164
- AudioEngine.Timer.cpp:794
- AudioEngine.UIEvents.cpp
- AudioEngine.RebuildDispatch.cpp:989,1062,1259 (recovery rebuild)
- EQEditProcessor.cpp:39
- NoiseShaperLearner.cpp:1517

---

## 3. `closeAdmission()` and `joinProducers()` Callers

### `closeAdmission()` callers

| File:Line | Context |
|---|---|
| AudioEngine.Processing.ReleaseResources.cpp:195 | `shutdownRuntime_.closeAdmission()` in `releaseResources()` |
| ISRShutdown.cpp:415 | definition |

### `joinProducers()` callers

| File:Line | Context |
|---|---|
| AudioEngine.Processing.ReleaseResources.cpp:197 | `shutdownRuntime_.joinProducers()` in `releaseResources()` |
| ISRShutdown.cpp:433 | definition |

### `isAdmissionOpen()` callers

| File:Line | Context |
|---|---|
| AudioEngine.h:4375 | `obs.noResurrection = !shutdownRuntime_.isAdmissionOpen()` (Q7 check in Proof) |
| ISRShutdown.cpp:447 | definition |
| ISRRuntimePublicationCoordinator.h:361 | `CoordinatorState::ShuttingDown` gate comment (references admission closure concept) |

### `admissionState()` callers

| File:Line | Context |
|---|---|
| ISRShutdown.cpp:352 | `q1 = (admissionState() == AdmissionState::Closed)` (Q1 check in Proof) |
| ISRShutdown.cpp:452 | definition |

### `tryMakeQuiescenceProof` callers

| File:Line | Context |
|---|---|
| AudioEngine.h:4379 | `shutdownRuntime_.tryMakeQuiescenceProof(obs)` in `tryShutdownQuiescentReclaim()` |
| ISRShutdown.cpp:355 | definition |

---

## 4. Q0 Hardcode Location

**Single location**: `AudioEngine.h:4368`
```cpp
obs.admissionReservationsZero = true;  // Q0: admission reservations は producer join 後 0
```

**Replacement target**:
```cpp
obs.admissionReservationsZero = shutdownRuntime_.outstanding() == 0;
```

**Boundary requirement**: AudioEngine must NOT access `packedState_` directly. Must go through `ShutdownRuntime::outstanding()`.

---

## 5. Shutdown Timeout Mechanism

### Current shutdown sequence (AudioEngine.Processing.ReleaseResources.cpp:185-199)

```cpp
setShutdownPhase(ShutdownPhase::StopWorkers, "releaseResources");
shutdownCoordinatorLoop();   // joins CoordinatorLoop thread
stopRebuildThread();         // joins RebuildThread
shutdownRuntime_.closeAdmission();    // Open→Closing
shutdownRuntime_.joinProducers();     // Closing→Closed (blind CAS, no retry)
```

### Existing timeout patterns (ReleaseResources.cpp)

| Pattern | File:Line | Timeout | Purpose |
|---|---|---|---|
| Graceful drain loop | ReleaseResources.cpp:497-531 | 5000ms | `m_retireRouter->pendingRetireCount() == 0 && activeReaderCount() == 0` |
| `isFullyDrained()` loop | AudioEngine.Threading.cpp:196 | 5000ms (inherited) | Wait for all intent queues empty |

### No existing retry on `joinProducers()`

Currently `closeAdmission()` and `joinProducers()` are called once. No retry loop. Since `joinProducers()` will return `bool` after modification (D101-31 plans), the caller in ReleaseResources.cpp must add a retry loop similar to the graceful drain pattern.

### Proposed retry pattern (following existing 5000ms pattern)

```cpp
shutdownRuntime_.closeAdmission();
{
    constexpr int kJoinProducersMaxMs = 5000;
    constexpr int kJoinProducersPollMs = 10;
    int waitedMs = 0;
    while (!shutdownRuntime_.joinProducers()) {
        if (waitedMs >= kJoinProducersMaxMs) {
            diagLog("[WARN] joinProducers timeout — proceeding with outstanding reservations");
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(kJoinProducersPollMs));
        waitedMs += kJoinProducersPollMs;
    }
}
```

---

## 6. AtomicAccess Wrapper Availability

All atomic operations in the codebase go through `convo::` wrappers in `src/audioengine/AtomicAccess.h`:

| Wrapper | Underlying | Usage for packedState_ |
|---|---|---|
| `convo::consumeAtomic(val, order)` | `std::atomic_load_explicit` | Read packedState_ |
| `convo::publishAtomic(dst, val, order)` | `std::atomic_store_explicit` | Write packedState_ |
| `convo::compareExchangeAtomic(dst, expected, desired, ...)` | `std::atomic_compare_exchange_strong_explicit` | CAS for tryAdmit, closeAdmission, joinProducers |
| `convo::fetchAddAtomic(dst, val, order)` | `std::atomic_fetch_add_explicit` | N/A (count uses CAS sub-loop) |
| `convo::fetchSubAtomic(dst, val, order)` | `std::atomic_fetch_sub_explicit` | N/A (count uses CAS sub-loop) |

**No direct `std::atomic_*` calls** exist in the codebase — all go through wrappers. New `packedState_` code must also use wrappers.

---

## 7. Authority Ownership Summary

### Single-owner confirmation

| Concern | Owner class | Field | External access? |
|---|---|---|---|
| AdmissionState | `ShutdownRuntime` | `admissionState_` (→ `packedState_`) | No (only via methods) |
| ReservationCount | `ShutdownRuntime` | (NEW, in `packedState_`) | No (only via `outstanding()`) |
| CoordinatorState | `RuntimeIntentCoordinator` | `state_` | No (only via `getState()`) |
| publicationIntentResidencyCount_ | `RuntimeIntentCoordinator` | `publicationIntentResidencyCount_` | Read via `getPublicationIntentResidencyCount()` |
| pendingIntentCount_ | `RuntimeIntentCoordinator` | `pendingIntentCount_` | Read via `getPendingIntentCount()` |
| retireBacklogCount_ | `RuntimeIntentCoordinator` | `retireBacklogCount_` | Read via `getRetireBacklogCount()` |
| ShutdownPhase | `ShutdownRuntime` | `phase_` | Read via `getPhase()` / `isShutdownInProgress()` |
| EngineLifecycleState | `AudioEngine` | `lifecycleState` | Read via `isShutdownInProgress()` |

**No authority splits found** — each concern has exactly one owner class with private fields, accessed only through public methods.

---

## 8. Conflict Resolution: D101-28 vs Current Code

### Conflict 1: AdmissionReservationAuthority class

- **D101-28 proposal**: New `AdmissionReservationAuthority` class
- **Current code**: No such class exists. `AdmissionState` lives in `ShutdownRuntime`, `CoordinatorState` lives in `RuntimeIntentCoordinator`
- **Resolution**: Do NOT create new class. AdmissionReservation count goes into `packedState_` in `ShutdownRuntime`. This satisfies D101-30 authority singularization.

### Conflict 2: Q0 measurement

- **Current code**: `obs.admissionReservationsZero = true` (hardcoded)
- **D101-28/D101-30**: `obs.admissionReservationsZero = outstanding() == 0`
- **Resolution**: Fix in D101-31-implementation — change AudioEngine.h:4368

### Conflict 3: `joinProducers()` return type

- **Current code**: `void joinProducers()` — blind CAS
- **D101-28/D101-30**: `bool joinProducers()` — returns false if count > 0
- **Resolution**: Change signature + add retry loop caller

### Conflict 4: Retire on Audio Thread

- **D101-29 proposal**: Consider gating `enqueueRetire` with `tryAdmit` on Audio Thread
- **D101-30 verdict**: EXCLUDED — Retire uses RCU epoch, separate from AdmissionReservation
- **Resolution**: Do NOT add tryAdmit to `enqueueRetire`. Confirm Audio Thread retire path does not touch admission state (verified — `enqueueRetire` → `EpochControl` only, `onRuntimeRetiredNonRt` has `ASSERT_NON_RT_THREAD`).

---

## 9. Implementation Prerequisites Checklist

| # | Requirement | Current status | Block? |
|---|---|---|---|
| 1 | `admissionState_` → `packedState_` (32-bit) | Not started | YES |
| 2 | `tryAdmit(n)` / `release(n)` / `outstanding()` methods added | Not exists | YES |
| 3 | `joinProducers()` → `bool` with count==0 check | Needs modification | YES |
| 4 | `closeAdmission()` uses packedState_ CAS | Needs modification | YES |
| 5 | Q0 hardcode → `outstanding()` | Needs modification | YES |
| 6 | Publication: tryAdmit after evaluate, release after enqueue | Not wired | YES |
| 7 | Recovery: tryAdmit after ShuttingDown check, release after push | Not wired | YES |
| 8 | Build: tryAdmit after isShutdownInProgress check, release after intent set | Not wired | YES |
| 9 | joinProducers retry loop in ReleaseResources.cpp | Not exists | YES |
| 10 | Audio Thread: no tryAdmit CAS | Confirmed clean | No |
| 11 | No direct `std::atomic_*` — use convo:: wrappers | Verified | No |

---

## 10. Audit Verdict: PASS

**All call sites enumerated. 3-path transaction boundaries confirmed. Authority ownership verified. Conflict resolution documented.**

The inventory is complete and consistent with D101-30's locked contract:

- **Admission authority**: `ShutdownRuntime` (sole owner of `admissionState_` / will own `packedState_`)
- **Reservation scope**: Publication + Recovery + Build (NOT Retire)
- **Reservation lifetime**: `tryAdmit` → `enqueue` → `release` (immediate release after enqueue, not at consumer pop)
- **G-H race**: Both `tryAdmit` and `closeAdmission` CAS on same `packedState_` word
- **Q0 source**: `ShutdownRuntime::outstanding()` (not AudioEngine direct access)
- **joinProducers**: `bool` return, caller retries on NonRT (matching existing 5000ms drain pattern)
- **RT boundary**: Audio Thread never touches admission state — RCU epoch only

**Proceed to D101-31-B implementation.**

---

**Audit completed**: 2026-08-24
**Auditor**: D101-31-A (read-only)
**Files changed**: 0 (inventory only — evidence only)
