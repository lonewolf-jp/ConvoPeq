# D101-29 — Implementation Authorization Gate

> **Phase**: D101-29 (read-only audit — NO code changes).
> Scope: Verify whether D101-28's design contract can be implemented against current code without changes.
> Verdict at the foot of this file.

---

## 1. Baseline

- **ConvoPeq.md Generated**: 2026-08-24 (regenerated via `python output_sourcecode_markdown.py`)
- **git status**: ` M ConvoPeq.md` only — no source (`src/`) modifications.
- **git diff --check**: 0 whitespace errors.

### 1.1 D101-29 target files verified

All 12 files from D101-28 §1 exist:

```text
src/audioengine/ISRShutdown.h/cpp                    ✅
src/audioengine/ISRRuntimePublicationCoordinator.h   ✅
src/audioengine/ISRRuntimePublicationCoordinator.cpp ✅
src/audioengine/ISRRuntimePublicationCoordinator_ProcessIntent.cpp ✅
src/audioengine/RuntimePublicationOrchestrator.cpp   ✅
src/audioengine/AudioEngine.h                       ✅
src/audioengine/AudioEngine.Commit.cpp              ✅
src/audioengine/AudioEngine.Timer.cpp               ✅
src/audioengine/AudioEngine.RebuildDispatch.cpp     ✅
src/audioengine/AudioEngine.Retire.cpp              ✅
src/audioengine/AudioEngine.Processing.ReleaseResources.cpp ✅
src/core/EpochDomain.h                              ✅ (note: src/core/, not src/audioengine/)
```

---

## 2. 4 Pathways — Full Caller Enumeration

### A. Publication Pathway (Path B)

**Primary gate**: `PublicationAdmission::evaluate()` (RuntimePublicationOrchestrator.cpp:37)
- Returns `Decision` enum: `Accepted | RejectedShutdown | RejectedStaleGeneration | RejectedNotFinalized | RejectedPressure | DeferredFadingActive`
- `RejectedShutdown` = `engine.isShutdownInProgress()` check (PublicationAdmission.cpp:11)

**All production callers of `commitRuntimePublication` / `enqueuePublicationIntent` / `trySubmit`:**

| Caller | Thread | File:Line | Admission check |
|---|---|---|---|
| `Timer publish` | NonRT (Timer Thread) | AudioEngine.Timer.cpp:994 | `isShutdownInProgress()` → `commitRuntimePublication` |
| `PrepareToPlay publish` | NonRT | AudioEngine.Processing.PrepareToPlay.cpp:155,277 | `isShutdownInProgress()` → `commitRuntimePublication` |
| `RebuildDispatch` | NonRT (Coordinator Loop) | AudioEngine.RebuildDispatch.cpp:989 | `enqueuePublicationIntentForRuntimeCommit` → `trySubmitImpl` → `admission_.evaluate()` |
| `RuntimePublishExecutor::executePublish` | NonRT (Publisher Thread) | RuntimePublishExecutor.h:71 | `PublishExecutor` — no admission check (post-decision execution) |

**Key finding**: Audio Thread never calls `commitRuntimePublication`. It only reads the published world via RCU (`makeRuntimeReadHandle`). All 4 pathways are NonRT.

### B. Recovery Pathway (Path C)

| Function | Thread | File:Line |
|---|---|---|
| `submitRecoveryIntent` (caller) | NonRT (Coordinator Loop) | AudioEngine.RebuildDispatch.cpp:911 |
| `submitRecoveryRequest` (implement) | NonRT (Coordinator Loop) | ISRRuntimePublicationCoordinator.cpp:855 |
| `recoveryIntentQueue_.push` | NonRT (Coordinator Loop) | ISRRuntimePublicationCoordinator.cpp:894 |
| `takePendingRecoveryAdmission` (consumer) | NonRT (Builder Loop) | ISRRuntimePublicationCoordinator.cpp:927 |
| `popRecoveryRequest` (consumer) | NonRT (Builder Loop) | ISRRuntimePublicationCoordinator.cpp:982 |
| `settlePendingRecoveryAdmission` | NonRT (Builder Loop) | ISRRuntimePublicationCoordinator.cpp:968 |
| `discardPendingRecoveryAdmission` | NonRT (Builder Loop) | ISRRuntimePublicationCoordinator.cpp:953 |

**Admission gate**: `CoordinatorState::ShuttingDown` check (ISRRuntimePublicationCoordinator.h:361) inside `enqueuePublicationIntent`.
**Reservation tracking**: `pendingIntentCount_` — transport residency counter, NOT admission reservation.
**Existing `reservationOwned`**: single-slot boolean in `PendingRecoveryAdmission` — NOT a count.

**Key finding**: Recovery admission is SPSC (Producer=CoordinatorLoop, Consumer=Builder Loop). No Audio Thread involvement.

### C. Build / Rebuild Pathway (Path D)

| Function | Thread | File:Line |
|---|---|---|
| `submitRebuildIntent` | Various (Timer, Params) | AudioEngine.Timer.cpp, AudioEngine.Parameters.cpp |
| `rebuildAdmissionPendingIntent_` | NonRT | AudioEngine.h:4260 |
| `rebuildOutstanding` | NonRT | AudioEngine.h |
| `rebuildAdmissionIntentMutex_` | NonRT | AudioEngine.h |
| `RebuildThread` (consumer) | NonRT (Rebuild Thread) | ISRCoordinatorLoop.cpp |
| `deferred retry` | NonRT (Coordinator Loop) | AudioEngine.RebuildDispatch.cpp |

**Admission gate**: `isShutdownInProgress()` checks in RebuildDispatch.cpp:241,287,380,437,479,515,838.
**Mutex usage**: `rebuildAdmissionIntentMutex_` exists — critical constraint for D101-28 Option C.

### D. Retire Pathway (Path E)

**Entry point**: `enqueueRetire` → `ISRRetireRouter::enqueueRetire` → `EpochControl::enqueueRetire`

| Step | Thread | Function | File:Line |
|---|---|---|---|
| 1. Enqueue retire | Audio Thread | `worldAuthority_.lifetime().enqueueRetire(slot)` | AudioEngine.Commit.cpp:600 |
| 2. Emit intent | Audio Thread | `worldAuthority_.lifetime().emitRetireIntentRT(intent)` | AudioEngine.Commit.cpp:485 |
| 3. NonRT drain | NonRT (Coordinator Loop) | `runtimePublicationBridge_.retire()` | AudioEngine.Commit.cpp:457 |
| 4. Router enqueue | NonRT | `m_retireRouter->enqueueRetire()` | ISRRetireRouter.cpp:239 |
| 5. Consumer drain | NonRT (Builder/Drain) | `tryReclaim()` / `drainDeferredRetireQueues()` | ISRRetireRouter.cpp |

**Key finding**: Audio Thread touches `enqueueRetire` (Commit.cpp:600) and `emitRetireIntentRT` (Commit.cpp:485) via the RCU world authority. However, these operate on `EpochControl` / `EpochDomain` — NOT on `AdmissionState`. The `ASSERT_NON_RT_THREAD` is on `onRuntimeRetiredNonRt` (the NonRT callback at Commit.cpp:450), not on `enqueueRetire` itself. Therefore:
- RT path: `enqueueRetire` → RCU epoch (no admission reservation needed)
- NonRT path: `onRuntimeRetiredNonRt` → `closeAdmission()` → `joinProducers()`

---

## 3. Step 3 — Packed FSM Operations

**Question**: Can 6 operations (tryAdmit/release/closeAdmission/joinProducers/outstanding/isAdmissionOpen) be defined on a single 32-bit packed atomic FSM?

**Finding**: YES — with critical constraints (see Step 7B for authority overlap).

### 3.1 Current FSM structures

**AdmissionState** (ISRShutdown.h:165-169):
```cpp
enum class AdmissionState : uint8_t { Open = 0, Closing, Closed, Faulted };
```
Already atomic via `admissionState_` with `compareExchangeAtomic` CAS operations.

**ShutdownPhase** (ISRShutdown.h:115-129):
```cpp
enum class ShutdownPhase : uint8_t {
    Running = 0, AudioStopped, AudioQuiesced,
    ReadersClosed, EpochsDrained, ReclaimComplete,
    TimedOut, Failed, ShutdownComplete
};
```
Sequential phase transitions.

### 3.2 Packed FSM feasibility

A 32-bit word can pack all state:

| Field | Bits | Capacity | Source |
|---|---|---|---|
| AdmissionState | 2 | 4 states | ISRShutdown.h:165 |
| ShutdownPhase | 4 | 16 states (current uses 9) | ISRShutdown.h:115 |
| version | 2 | 4 ABA cycles | D101-28 new |
| reservationCount | 24 | max 16,777,215 | D101-28 new |

**Packed FSM layout (32-bit atomic `packedState_`):**

```text
Bits [0:1]   AdmissionState (Open=0, Closing=1, Closed=2, Faulted=3)
Bits [2:5]   ShutdownPhase (Running=0 ... ShutdownComplete=8)
Bits [6:7]   version (2-bit ABA counter, wraps 0→1→2→3→0)
Bits [8:31]  reservationCount (24 bits, max 16,777,215)
```

### 3.3 Operations analysis

| Operation | Current | Packed FSM impl | CAS-safe? |
|---|---|---|---|
| `tryAdmit(n)` | `evaluate()` returns Decision enum | CAS: if state==Open: count += n | yes |
| `release(n)` | No explicit release (Q0 hardcoded true) | CAS: count -= n | yes |
| `closeAdmission()` | `admissionState_` Open→Closing | CAS: state=Open → Closing, increment version | yes |
| `joinProducers()` | `admissionState_` Closing→Closed | CAS: state=Closing → Closed, count must be 0 | yes |
| `outstanding()` | None — Q0 hardcoded | Return `(packed >> 8) & 0xFFFFFF` | yes |
| `isAdmissionOpen()` | `consumeAtomic(admissionState_) == Open` | Return `(packed & 3) == 0` | yes |

**Verdict**: All 6 operations can be defined on a single 32-bit packed atomic with CAS loops. No new fields needed beyond `reservationCount` (24-bit) and `version` (2-bit ABA protection).

---

## 4. Step 4 — `joinProducers()` semantics when `count > 0`

**Current implementation** (ISRShutdown.cpp:433):
```cpp
void ShutdownRuntime::joinProducers() noexcept {
    AdmissionState expected = AdmissionState::Closing;
    compareExchangeAtomic(admissionState_, expected, AdmissionState::Closed, ...);
}
```

**Finding**: `joinProducers()` is a blind CAS — it unconditionally transitions `Closing→Closed` regardless of any count. It does NOT check `count == 0`.

**D101-28 §4.3 requirement**: `joinProducers()` must verify `reservationCount == 0` before transitioning to `Closed`.

| Behavior | Current | D101-28 Requirement | Match? |
|---|---|---|---|
| Check count == 0 | NO — blind CAS | YES — must verify | ❌ |
| Return false if count > 0 | N/A — void return | YES — return false | ❌ |
| Retry until count == 0 | NO | Optional (caller manages) | ❌ |

**Verdict**: FAIL — `joinProducers()` MUST be modified to add a count==0 precondition. The 3-line change:
```cpp
bool ShutdownRuntime::joinProducers() noexcept {
    uint32_t packed = consumeAtomic(packedState_, ...);
    if ((packed & 3) != 1) return false;  // Closing?
    if ((packed >> 8) & 0xFFFFFF) return false;  // reservations > 0
    uint32_t desired = (packed & ~3) | 2;  // Closing→Closed
    return compareExchangeAtomic(packedState_, packed, desired, ...);
}
```

---

## 5. Step 5 — Shutdown sequence ordering

**Current shutdown sequence** (AudioEngine.Processing.ReleaseResources.cpp:185-199):

| Step | Action | File:Line |
|---|---|---|
| 1 | `shutdownCoordinatorLoop()` | ReleaseResources.cpp:187 |
| 2 | `stopRebuildThread()` | ReleaseResources.cpp:189 |
| 3 | `closeAdmission()` → Open→Closing | ReleaseResources.cpp:192 |
| 4 | `joinProducers()` → Closing→Closed | ReleaseResources.cpp:194 |

**D101-28 §4.4 requirement**: producer join → closeAdmission → reservation==0 → Closed → Q0/Q1 → Proof

| Step | Current | D101-28 | Match? |
|---|---|---|---|
| 1. Stop coordinator loop | `shutdownCoordinatorLoop()` | yes | ✅ |
| 2. Stop rebuild thread | `stopRebuildThread()` | yes | ✅ |
| 3. Close admission | `closeAdmission()` (Open→Closing) | closeAdmission | ✅ |
| 4. Join producers | `joinProducers()` (Closing→Closed, blind CAS) | joinProducers + verify count==0 | ❌ (missing count check) |
| 5. Verify Q0 (reservations==0) | `admissionReservationsZero = true` (hardcoded) | Q0 from `outstanding() == 0` | ❌ (Q0 hardcoded, not measured) |
| 6. Generate Proof | `tryMakeQuiescenceProof()` | Proof after Q0-Q7 verified | ✅ (proof exists, Q0 is hardcoded) |

**Verdict**: FAIL — Two modifications needed:
1. `joinProducers()` must check `reservationCount == 0`
2. `QuiescenceObservation::admissionReservationsZero` must be set from `outstanding() == 0`, not hardcoded `true`

---

## 6. Step 6 — Reservation lifetime map

### 6A. Publication (Path B)

```text
Acquire LP:     admission_.evaluate() returns Accepted  [RuntimePublicationOrchestrator.cpp:37]
Queue:          build world → executePublish (NonRT)  [RuntimePublishExecutor.h:71]
Release LP:     bridge.didPublishRuntimeNonRt()       [AudioEngine.Commit.cpp:320]
Rollback:       decision != Accepted → return          [RuntimePublicationOrchestrator.cpp:43-46]
RT:             NO — Audio Thread only reads via RCU handle
```

**Current reservation count**: N/A — `evaluate()` returns Decision enum, no counter.

### 6B. Recovery (Path C)

```text
Acquire LP:     enqueuePublicationIntent → ShuttingDown check  [ISRRuntimePublicationCoordinator.cpp:855]
Queue:          recoveryIntentQueue_.push                       [cpp:894]
Release LP:     popRecoveryRequest (consumer drains)            [cpp:982]
Rollback:       discardPendingRecoveryAdmission cancels          [cpp:953]
RT:             NO — SPSC (Coordinator Loop → Builder Loop)
```

**Current reservation count**: `pendingIntentCount_` is transport residency, NOT admission.

### 6C. Build (Path D)

```text
Acquire LP:     submitRebuildIntent → rebuildPendingIntent_ set  [AudioEngine.h:4260]
Queue:          rebuildIntent_ mutex-protected                  [RebuildDispatch.cpp]
Release LP:     RebuildThread consumes, clears                [ISRCoordinatorLoop.cpp]
Rollback:       deferred retry path                          [RebuildDispatch.cpp]
RT:             NO — mutex-protected, NonRT only
```

**Current reservation count**: No count field — mutex + boolean flag only.

### 6D. Retire (Path E)

```text
Acquire LP:     [Audio Thread] enqueueRetire → EpochControl  [AudioEngine.Commit.cpp:600]
Queue:          ISRRetireRouter::enqueueRetire              [ISRRetireRouter.cpp:239]
Release LP:     tryReclaim() / drainDeferredRetireQueues     [ISRRetireRouter.cpp]
Rollback:       requestRollback() on semantic violation        [AudioEngine.Commit.cpp:370]
RT:             YES — enqueueRetire on Audio Thread          [Commit.cpp:600]
```

**Critical RT boundary question**: `enqueueRetire` on Audio Thread accesses `worldAuthority_.lifetime()` (RCU epoch domain). The D101-28 reservation counter would be in `AdmissionState` (ShutdownRuntime). If `enqueueRetire` must increment `reservationCount`, it would be a RT atomic increment — 24-bit sub-word CAS on packed word.

**Finding**: `enqueueRetire` does NOT touch `AdmissionState`. It touches `EpochControl` / `EpochDomain`. The `ASSERT_NON_RT_THREAD` is on `onRuntimeRetiredNonRt` (the NonRT callback), not on `enqueueRetire` itself. Therefore:
- RT path: `enqueueRetire` → RCU epoch (no admission reservation needed)
- NonRT path: `onRuntimeRetiredNonRt` → `closeAdmission()` → `joinProducers()`

**Verdict**: The retire pathway can acquire/release reservations on NonRT only. Audio Thread `enqueueRetire` does RCU epoch bookkeeping that is independent of the admission reservation counter.

---

## 7. Step 7 — RT boundary audit

### 7A. `tryAdmit()` on Audio Thread — feasibility

The Audio Thread calls (Commit.cpp):

| Function | File:Line | Can be gated? |
|---|---|---|
| `commitRuntimePublication` | — (not called on Audio Thread) | N/A |
| `enqueueRetire` | AudioEngine.Commit.cpp:600 | Via `isAdmissionOpen()` — atomic load only |
| `emitRetireIntentRT` | AudioEngine.Commit.cpp:485 | Via `isAdmissionOpen()` — atomic load only |

**Finding**: Audio Thread admission gate = single atomic load of packed word + bitmask check `(packed & 3) == 0`. This is lock-free, wait-free, no allocation — suitable for RT.

### 7B. Authority overlap — the critical risk

5 separate state systems control admission/shutdown:

| System | Type | Scope | File |
|---|---|---|---|
| AdmissionState (AdmissionState enum) | 4-state FSM | Shutdown admission | ISRShutdown.h:165 |
| ShutdownPhase (phase_) | 9-state sequential | Shutdown lifecycle | ISRShutdown.h:115 |
| CoordinatorState | 7-state enum | Publication coordinator | ISRRuntimePublicationCoordinator.h:335 |
| PublicationAdmission::Decision | 6-value enum | Per-request decision | RuntimePublicationOrchestrator.cpp:37 |
| EngineLifecycleState | 4-state enum | Engine lifecycle | AudioEngine.h (lifecycleState) |

**Finding**: `isShutdownInProgress()` (AudioEngine.h:1480) ORs `EngineLifecycleState` AND `ShutdownPhase`. `enqueuePublicationIntent` checks `CoordinatorState::ShuttingDown`. `commitRuntimePublication` checks `isShutdownInProgress()`. `trySubmit` checks `admission_.evaluate()` → `evaluate()` checks `isShutdownInProgress()`.

**Verdict**: Adding a reservation counter to `AdmissionState` creates overlap with `CoordinatorState::ShuttingDown` and `ShutdownPhase`. The packed FSM must be the single source of truth for admission. Currently:
- `closeAdmission()` (AdmissionState Open→Closing) runs at ReleaseResources.cpp:192
- `CoordinatorState::ShuttingDown` is set in `shutdownCoordinatorLoop()` at ReleaseResources.cpp:187

The ordering is correct (Coordinator stops first, then admission closes), but the authority is split between `ShutdownRuntime` (AdmissionState) and `RuntimePublicationCoordinator` (CoordinatorState).

---

## 8. Step 8 — Packed FSM implementation check

### 8.1 Current atomic fields in ShutdownRuntime (ISRShutdown.h)

```text
phase_                  // ShutdownPhase (4 bits used)
lastNonTerminalPhase_   // ShutdownPhase (not packed)
admissionState_         // AdmissionState (2 bits used, uint8_t)
shutdownGeneration_     // uint64_t
sh6PostStopEnqueueCount_ // uint64_t (sh6 telemetry)
```

### 8.2 Packed FSM layout (32-bit atomic packedState_)

```text
Bits [0:1]   AdmissionState (Open=0, Closing=1, Closed=2, Faulted=3)
Bits [2:5]   ShutdownPhase (Running=0 ... ShutdownComplete=8)
Bits [6:7]   version (2-bit ABA counter)
Bits [8:31]  reservationCount (24 bits, max 16,777,215)
```

### 8.3 Required code changes (3 files)

| File | Change | Lines affected |
|---|---|---|
| ISRShutdown.h | Replace `admissionState_` + `phase_` → single `packedState_` (32-bit); add `tryAdmit(n)`, `release(n)`, `outstanding()` | ~115-290 |
| ISRShutdown.cpp | Rewrite `closeAdmission()`, `joinProducers()`, `isAdmissionOpen()`, `isShutdownInProgress()` to use packed FSM; change `joinProducers()` return to `bool` | ~1480-455 |
| AudioEngine.Processing.ReleaseResources.cpp | Set `admissionReservationsZero` from `outstanding() == 0` instead of hardcoded `true` | ~192-194 |

### 8.4 RT-safe operations

```cpp
// Audio Thread — single 32-bit load (lock-free, wait-free)
bool isAdmissionOpen() const noexcept {
    return (consumeAtomic(packedState_, std::memory_order_acquire) & 3) == 0;
}

// NonRT — CAS loop for tryAdmit
bool tryAdmit(uint32_t n) noexcept {
    uint32_t expected = consumeAtomic(packedState_, std::memory_order_acquire);
    do {
        if ((expected & 3) != 0) return false;  // not Open
        uint32_t count = (expected >> 8) & 0xFFFFFF;
        if (count + n > 0xFFFFFF) return false;  // overflow
        uint32_t desired = (expected & ~0xFFFFFF00) | ((count + n) << 8);
    } while (!compareExchangeAtomic(packedState_, expected, desired,
        std::memory_order_acq_rel, std::memory_order_acquire));
    return true;
}

// NonRT — CAS loop for joinProducers with count check
bool joinProducers() noexcept {
    uint32_t expected = consumeAtomic(packedState_, std::memory_order_acquire);
    if ((expected & 3) != 1) return false;  // not Closing
    if ((expected >> 8) & 0xFFFFFF) return false;  // reservations > 0
    uint32_t desired = (expected & ~3) | 2;  // Closing→Closed
    return compareExchangeAtomic(packedState_, expected, desired,
        std::memory_order_acq_rel, std::memory_order_acquire);
}
```

---

## 9. Step 9 — INV-LIFE-13 verification

**INV-LIFE-13**: `closeAdmission()` must be irreversible (Closed→Open forbidden — resurrection prevention).

**Current code** (ISRShutdown.cpp:420):
```cpp
AdmissionState expected = AdmissionState::Open;
if (convo::compareExchangeAtomic(admissionState_, expected, AdmissionState::Closing, ...)) {
    // success only if was Open — increment shutdownGeneration_
}
```

**Finding**: `closeAdmission()` only transitions Open→Closing. Once Closing, it cannot go back (CAS expects `Open`). `joinProducers()` transitions Closing→Closed. No path from Closed→Open. `Faulted` is terminal.

**INV-LIFE-13**: SATISFIED — no code change needed. The 2-bit AdmissionState field preserves this: once bits [0:1] leave 0 (Open), they never return (Closing=1, Closed=2, Faulted=3).

---

## 10. Step 10 — Test feasibility

### 10.1 Unit tests for packed FSM

| Test | Feasibility | Existing infrastructure |
|---|---|---|
| `tryAdmit` succeeds when Open | yes | `admissionState_` CAS already tested |
| `tryAdmit` fails when Closed | yes | New test case needed |
| `release` decrements count | yes | New test |
| `closeAdmission` Open→Closing | yes | Existing pattern at ISRShutdownTests.cpp |
| `joinProducers` fails when count > 0 | yes | New test |
| `joinProducers` succeeds when count == 0 | yes | New test |
| `isAdmissionOpen` returns false after Closed | yes | New test |

### 10.2 Integration tests

| Test | Feasibility | Risk |
|---|---|---|
| Publish during shutdown blocks | yes | Medium — needs coordination mock |
| Retire on Audio Thread during shutdown | yes | Low — RCU epoch already handles this |
| Recovery enqueue during Closing | yes | Medium — SPSC queue test |
| Build submit during Closed | yes | Low — `isShutdownInProgress()` guard exists |

**Test infrastructure**: CTest with CMake presets (`build/` directory). Tests compile and run via `ctest --test-dir build -C Debug --output-on-failure`.

**Verdict**: All tests feasible — no new infrastructure needed. Existing `ASSERT_NON_RT_THREAD` assertions in test code can be extended to verify the reservation counter is not accessed on Audio Thread.

---

## 11. Step 11 — Gate table

### 11.1 Implementation gates (from D101-28 §5)

| Gate | Requirement | Current code | Status | Action needed |
|---|---|---|---|---|
| G1 | `tryAdmit(n)` — atomic, RT-safe | Not implemented | NEW | Add `packedState_` + `tryAdmit()` |
| G2 | `release(n)` — atomic decrement | Not implemented | NEW | Add `release()` |
| G3 | `closeAdmission()` — Open→Closing CAS | exists | ok | Modify to packed word |
| G4 | `joinProducers()` — Closing→Closed with count==0 | exists (blind CAS) | MODIFY | Add count check, return `bool` |
| G5 | `outstanding()` — return count | Not implemented | NEW | Add `outstanding()` |
| G6 | `isAdmissionOpen()` — atomic load | exists | ok | Modify to packed word |
| G7 | Q0 = `outstanding() == 0` (not hardcoded) | hardcoded `true` | MODIFY | Set from `outstanding()` |
| G8 | `EnqueueDuringShutdown` → return false (not throw) | returns Decision | ok | No change |
| G9 | `ASSERT_NON_RT_THREAD` on `release()` | N/A (new) | NEW | Add to `release()` |
| G10 | `joinProducers()` failure → retry | blind CAS (no failure) | MODIFY | Return false on count > 0 |

### 11.2 Call-site changes

| File:Line | Current | Required change |
|---|---|---|
| RuntimePublicationOrchestrator.cpp:37 | `admission_.evaluate()` returns Decision | Add `tryAdmit(1)` after Accepted |
| AudioEngine.Commit.cpp:600 | `enqueueRetire(slot)` | Add `if (!isAdmissionOpen())` early return |
| AudioEngine.Commit.cpp:485 | `emitRetireIntentRT` | Add `isAdmissionOpen()` check |
| AudioEngine.Processing.ReleaseResources.cpp:192 | `closeAdmission()` void | Change to return `bool`, check result |
| AudioEngine.Processing.ReleaseResources.cpp:194 | `joinProducers()` blind CAS | Check return, retry if count > 0 |
| AudioEngine.h:4368 | `obs.admissionReservationsZero = true` | Set `= (outstanding() == 0)` |

---

## 12. Step 12 — Final verdict

### CAN IMPLEMENT (with modifications)

The D101-28 design contract CAN be implemented against the current codebase. The foundational FSM (`AdmissionState` in ISRShutdown.h/cpp) already exists with the correct 4-state semantics (Open→Closing→Closed→irreversible). The CAS patterns are established. The Audio Thread boundary is clean — RT only touches RCU epoch, not admission state.

### 4 required modifications (all small, NonRT-only)

| # | File | Change | Complexity |
|---|---|---|---|
| 1 | ISRShutdown.h / .cpp | Replace `admissionState_` + `phase_` → single `packedState_` (32-bit). Add `tryAdmit(n)`, `release(n)`, `outstanding()`. | Small — 2 fields → 1 packed word |
| 2 | ISRShutdown.cpp:433 | `joinProducers()` — add count==0 check, return `bool` | Small — 3-line add |
| 3 | ReleaseResources.cpp:194 | Call `joinProducers()` in retry loop if returns false | Small — 5-line loop |
| 4 | AudioEngine.Commit.cpp:485,600 | Gate Audio Thread `enqueueRetire`/`emitRetireIntentRT` with `isAdmissionOpen()` | Small — 2 guard checks |

### Risk: authority overlap (Step 7B)

5 state systems control admission/shutdown. The packed FSM must become the single source of truth. `CoordinatorState::ShuttingDown` and `EngineLifecycleState` must defer to `AdmissionState::Closed`. This is an architectural decision (not a code change) — D101-28 §7 must clarify the authority hierarchy.

### Risk: `enqueueRetire` on Audio Thread

If the reservation counter is meant to track all retire intents (including RT `enqueueRetire`), then `enqueueRetire` at Commit.cpp:600 must call `tryAdmit(1)` on the Audio Thread. This requires:
- The packed atomic must be 32-bit (single-word CAS — already RT-safe)
- `enqueueRetire` must NOT acquire any mutex (it currently doesn't — RCU epoch is lock-free)

**Finding**: `enqueueRetire` → `EpochControl::enqueueRetire` is already lock-free RCU. Adding a 32-bit CAS increment is RT-safe.

### Verdict: GO — Proceed to implementation

**Conditions**:
1. Authority overlap resolved — `AdmissionState` (packed FSM) is the canonical admission gate; `CoordinatorState` and `EngineLifecycleState` defer to it
2. `enqueueRetire` on Audio Thread is a count increment (RT-safe 32-bit CAS), NOT a state transition
3. `joinProducers()` returns `bool` to allow caller retry when count > 0

---

**Audit completed**: 2026-08-24
**Auditor**: D101-29 (read-only)
**Files changed**: 0 (audit only — evidence only)
