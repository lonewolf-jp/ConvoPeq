# D101-30 — AdmissionReservation Final Implementation Contract Audit

> **Phase**: D101-30 (read-only)
> Purpose: Lock the final implementation contract for AdmissionReservation.
> Code changes: PROHIBITED
> References: Latest ConvoPeq.md (2026-08-24), D101-28, D101-29
> Verdict: GO / NO-GO

---

## Step 1 — Authority Reconfirmation

### Authority hierarchy (single source of truth for each concern)

```text
AdmissionState (Open→Closing→Closed)
        ↑ owns
AdmissionReservation count (tryAdmit/release/outstanding)
        ↑ observed by
Q0 (admissionReservationsZero)
        ↓ feeds
tryMakeQuiescenceProof() [ShutdownRuntime]
        ↓ binds
ReclaimPermit identity (generation + engineInstanceId)
        ↓ consumed by
reclaimShutdownQuiescent() [RuntimeIntentCoordinator]

CoordinatorState (ShuttingDown) — SEPARATE gate: stops new intent enqueue
        ↑ set by
requestShutdown() [ShutdownScheduler inside RuntimeIntentCoordinator]
        ↓ triggers
ShutdownSequence (ReleaseResources.cpp:185-199):
  1. shutdownCoordinatorLoop()  → CoordinatorState::ShuttingDown
  2. stopRebuildThread()
  3. closeAdmission()            → AdmissionState Open→Closing
  4. joinProducers()             → AdmissionState Closing→Closed (requires count==0)
```

### Final authority assignments (LOCKED)

| Concern | Owner | Secondary (if any) |
|---|---|---|
| AdmissionState FSM | `ShutdownRuntime` (ISRShutdown.h:289-308) | None |
| AdmissionReservation count | `ShutdownRuntime` (NEW — packed into same atomic as AdmissionState) | None — cannot be in `RuntimeIntentCoordinator` |
| CoordinatorState | `RuntimeIntentCoordinator` (ISRRuntimePublicationCoordinator.h:89-96) | ShutdownScheduler (inner class) |
| `isShutdownInProgress()` | `AudioEngine` delegates to `ShutdownRuntime::isShutdownInProgress()` | `EngineLifecycleState` (OR'd) |
| `tryMakeQuiescenceProof()` | `ShutdownRuntime` (ISRShutdown.cpp:355) | Reads `QuiescenceObservation` from AudioEngine |
| `bindShutdownIdentity()` | `ShutdownRuntime` calls `reclaimAuthority_.bindShutdownIdentity()` | `RuntimeIntentCoordinator` |

### D101-28 vs current code analysis

D101-28 proposes creating a new `AdmissionReservationAuthority` class. **Current code does NOT have this class.**

Current architecture:
- `AdmissionState` lives in `ShutdownRuntime` (single-owner)
- `CoordinatorState` lives in `RuntimeIntentCoordinator` (single-owner)
- These are **already singular** per their respective domains

**Decision**: Do NOT create a separate `AdmissionReservationAuthority` class. The reservation count must live **inside `ShutdownRuntime`** alongside `AdmissionState`. This is the only way to ensure the G-H linearization point (Step 5) works correctly — `tryAdmit` CAS and `closeAdmission` CAS must race on the same atomic word.

**D101-28's proposal to put AdmissionReservation in `RuntimeIntentCoordinator`** is **REJECTED by current code architecture** — it would split AdmissionState + count into different objects, breaking CAS atomicity.

---

## Step 2 — Reservation vs Transport Residency Separation

### Three distinct concepts (MUST NOT be conflated)

| Concept | Current code | Purpose | Atomic field |
|---|---|---|---|
| **AdmissionReservation** | NOT YET IMPLEMENTED (hardcoded `true` for Q0) | Track in-flight tryAdmit→enqueue→release window | NEW: packed into `packedState_` |
| `publicationIntentResidencyCount_` | ISRRuntimePublicationCoordinator.h:548 | Transport residency = Queue residency + producer reservation (for Publish intents only) | 64-bit atomic |
| `pendingIntentCount_` | ISRRuntimePublicationCoordinator.h:561 | Transport residency for Observe/Quarantine/Recovery intents | 64-bit atomic |

### Current code explicit contracts

**INV-ISR-02** (ISRRuntimePublicationCoordinator.h:78):
> `pendingIntentCount_` is transport residency + producer reservation, NOT queue size.

**Comment block at ISRRuntimePublicationCoordinator.h:539**:
> `publicationIntentResidencyCount_` = intentQueue_ 内の Publish Intent 数 + producer enqueue reservation

**Comment at ISRRuntimePublicationCoordinator.cpp:989**:
> `fetchSub` するため counter は整合し、isFullyDrained の queue-empty + counter==0 が正しく成立する

### Explicit separation confirmed

```text
AdmissionReservation   ≠  publicationIntentResidency
                        ≠  pendingIntentCount
                        ≠  retireBacklog
```

- `publicationIntentResidencyCount_` tracks Publish intent transport residency (INV-X5-1 compliant)
- `pendingIntentCount_` tracks Observe/Quarantine/Recovery transport residency (INV-ISR-02 compliant)
- `retireBacklogCount_` tracks retired objects pending epoch safety
- **None of these are admission reservations**

### Reservation lifetime definition (LOCKED)

**AdmissionReservation is a transient gate reservation**, NOT a transport residency:

```text
tryAdmit()          ← CAS: Open, count → count+1
      ↓
enqueue             ← enqueue to transport queue
      ↓
release()           ← CAS: count → count-1
```

**Critical**: The reservation is released **immediately after successful enqueue**, NOT at consumer pop time. This is the reservation-before-push protocol already used for `publicationIntentResidencyCount_` and `pendingIntentCount_`:

```cpp
// Existing pattern (ISRRuntimePublicationCoordinator.h:372):
convo::fetchAddAtomic(publicationIntentResidencyCount_, 1, ...);  // reservation-before-push
if (intentQueue_.push(prepared)) return true;
convo::fetchSubAtomic(publicationIntentResidencyCount_, 1, ...);  // rollback on push failure
```

**Consumer pop does NOT release admission reservation** — the producer already called `release()` after successful `enqueue`. This prevents admission reservation from morphing into transport residency.

---

## Step 3 — 4-Path Reservation Lifetime Table

| Path | tryAdmit | enqueue | release | RT? | Details |
|---|---|---|---|---|---|
| **Publication** | YES — `tryAdmit(1)` in `trySubmitImpl` after `admission_.evaluate()` returns Accepted | `enqueuePublicationIntent()` in Coordinator | `release(1)` immediately after push success | No | Gate: `PublicationAdmission::evaluate()` + `isShutdownInProgress()` |
| **Recovery** | YES — `tryAdmit(1)` in `submitRecoveryRequest` after `CoordinatorState::ShuttingDown` check | `recoveryIntentQueue_.push()` | `release(1)` immediately after push success | No | Gate: `CoordinatorState::ShuttingDown` check at ISRRuntimePublicationCoordinator.cpp:875 |
| **Build/Rebuild** | YES — `tryAdmit(1)` in `submitRebuildIntent` after `isShutdownInProgress()` check | `rebuildAdmissionPendingIntent_` set | `release(1)` after intent queued or on rollback | No | Gate: `isShutdownInProgress()` checks in RebuildDispatch.cpp |
| **Retire** | NO — Retire is OUTSIDE AdmissionReservation | N/A — `enqueueRetire` on Audio Thread via RCU | N/A | YES (RT) | Retire domain has its own quiescence (EpochControl). AdmissionReservation does NOT cover retire |

### Retire path — FINAL decision: EXCLUDED from AdmissionReservation (LOCKED)

**Rationale**:

1. `enqueueRetire` on Audio Thread (Commit.cpp:600) operates on `EpochControl` / `EpochDomain` — a completely separate quiescence protocol
2. `onRuntimeRetiredNonRt` (Commit.cpp:450) has `ASSERT_NON_RT_THREAD()` — the world retirement bookkeeping is NonRT
3. Adding a 32-bit CAS to the Audio Thread retire path would create a false dependency between epoch quiescence and admission state
4. Retire quiescence is already tracked by:
   - `retireBacklogCount_` (ISRRuntimePublicationCoordinator.h)
   - `reclaimInFlightCount_` (ISRRuntimePublicationCoordinator.h:557)
   - `pendingRetireGenerationCount_` (AudioEngine.h)
   - EpochDomain reader epochs (EpochDomain.h)

5. INV-ISR-06 (ISRRuntimePublicationCoordinator.h:14): retirement/ownership identity source is publish() oldWorld / Lifetime, NOT admission state
6. The `QuiescenceObservation` already tracks retire quiescence via Q4 (activeReadersZero) and Q5 (epochSettled)

**Q0 definition (LOCKED)**: `admissionReservationsZero` = outstanding AdmissionReservation count == 0 for Publication + Recovery + Build paths ONLY. Retire is tracked by Q4/Q5.

---

## Step 4 — Packed FSM Scope Decision

### REJECT D101-29's full-pack proposal

D101-29 proposed packing AdmissionState + ShutdownPhase + version + count into a single 32-bit word.

**Decision: NO — too broad scope.**

### LOCKED Packed FSM: AdmissionPackedState (32-bit)

```text
Bits [0:1]   AdmissionState     (Open=0, Closing=1, Closed=2, Faulted=3)
Bits [2:7]   version            (6-bit ABA counter, wraps 0→63)
Bits [8:31]  reservationCount   (24 bits, max 16,777,215)
```

**ShutdownPhase stays SEPARATE**: It is a lifecycle state (sequential phase transitions), NOT a linearized CAS race. Mixing it with admission linearization point would:
1. Force `transitionTo()` to acquire the packed atomic (currently lock-free `publishAtomic`)
2. Create false CAS contention between phase transitions and admission operations
3. Break the existing phase transition ordering (ReleaseResources.cpp:185-199)

### Justification: atomize only what needs linearization

```text
What needs CAS linearization (G-H race):
  → tryAdmit CAS  VS  closeAdmission CAS  on the SAME atomic word

What does NOT need CAS:
  → ShutdownPhase transitions (sequential, single-writer during shutdown)
  → CoordinatorState transitions (single-writer within CoordinatorLoop)
```

### Atomic layout

```cpp
// ISRShutdown.h — NEW field replacing admissionState_
std::atomic<uint32_t> packedState_{0};  // AdmissionPackedState (Step 4)
```

**No changes to `phase_`** — remains separate `std::atomic<ShutdownPhase>`.

---

## Step 5 — G-H Linearization Point

### LOCKED linearization point

```text
tryAdmit(count)
    │
    ├── load packedState_
    ├── check: state == Open
    ├── CAS(Open, count → Open, count + n)    ◄── LINEARIZATION POINT
    │         │
    │         ├── success → enqueue → release(count)
    │         └── fail (state != Open) → return false
    │
    ▼
closeAdmission()
    │
    ├── CAS(Open, count → Closing, count)      ◄── LINEARIZATION POINT
    │         │
    │         ├── success → producer join
    │         └── fail (already Closing/Closed/Faulted) → return false
```

### G-H race semantics (LOCKED)

| Race outcome | tryAdmit wins | closeAdmission wins |
|---|---|---|
| tryAdmit success | Admission accepted, count incremented | tryAdmit CAS fails (state != Open), returns false |
| enqueue after | Proceeds normally | enqueue sees `isAdmissionOpen() == false`, returns false (RejectedShutdown) |
| Proof generation | Q0 fails (count > 0), Proof returns nullopt | Q0 passes, Proof succeeds |

### Formal invariant

```text
Proof success
  ⇒ closeAdmission already linearized
    ⇒ packedState_ bits [0:1] == Closing or Closed
    ⇒ subsequent tryAdmit CAS fails (state != Open)
    ⇒ no new reservations can be admitted
```

This is a direct consequence of the single-word CAS: once `closeAdmission` succeeds (Open→Closing), the `tryAdmit` CAS will observe `state != Open` and fail.

---

## Step 6 — Q0 Source-of-Truth

### Current code (AUDIO ENGINE) (AudioEngine.h:4368)

```cpp
obs.admissionReservationsZero = true;  // HARDCODED — NO measurement
```

### Required final contract

```cpp
obs.admissionReservationsZero =
    shutdownRuntime_.outstanding() == 0;
```

### Boundary enforcement

**AudioEngine MUST NOT** directly access any internal reservation counter. The read-only observation boundary is:

```text
AudioEngine
   ┓
   ┗ observe
       ↓ (read-only method)
       ShutdownRuntime::outstanding()
           ↓ reads
           packedState_ (bits [8:31])
```

**Implementation**: `outstanding()` is a non-mutating accessor on `ShutdownRuntime`:

```cpp
std::uint32_t outstanding() const noexcept {
    return (convo::consumeAtomic(packedState_, std::memory_order_acquire) >> 8) & 0xFFFFFF;
}
```

AudioEngine calls `shutdownRuntime_.outstanding()` — does NOT read `packedState_` directly.

### Current violation

AudioEngine.h:4368 hardcodes `true` instead of querying. This must be fixed in implementation (D101-31).

---

## Step 7 — joinProducers() Semantics

### LOCKED contract

```text
joinProducers():
  1. CAS(Closing, count → Closed, count) if count == 0
  2. If count > 0: return false (DO NOT wait)
  3. If state != Closing: return false
  4. Return true only if transition succeeded
```

```cpp
bool ShutdownRuntime::joinProducers() noexcept {
    uint32_t expected = consumeAtomic(packedState_, std::memory_order_acquire);
    // Must be Closing
    if ((expected & 3) != 1) return false;
    // Must have zero reservations
    if ((expected >> 8) & 0xFFFFFF) return false;
    // Closing→Closed, preserve count and version
    uint32_t desired = (expected & ~3u) | 2u;
    return compareExchangeAtomic(packedState_, expected, desired,
        std::memory_order_acq_rel, std::memory_order_acquire);
}
```

### Caller manages retry (NOT Audio Thread)

**Shutdown sequence** (ReleaseResources.cpp:185-199, NonRT only):

```cpp
// Modified shutdown sequence:
shutdownCoordinatorLoop();
stopRebuildThread();
closeAdmission();                    // Open→Closing (idempotent)
// Retry loop: wait for all reservations to drain
while (true) {
    if (joinProducers()) break;      // Closing→Closed succeeds when count==0
    std::this_thread::yield();       // NonRT spin-wait — NOT on Audio Thread
}
```

**Audio Thread NEVER waits** — `isAdmissionOpen()` check on Audio Thread is a single atomic load, returns immediately.

### Current code violation

`joinProducers()` is currently `void` and performs a blind CAS (no count check). Must change to `bool` return with count==0 precondition.

---

## Step 8 — Proof Sealing (Formalization)

### T18 — tryAdmit || tryMakeQuiescenceProof racing

```text
Thread 1: tryAdmit(1)           Thread 2: tryMakeQuiescenceProof(obs)
  CAS(Open,0 → Open,1)              load packedState_.state
  succeeds                          load outstanding() = 1
  enqueue                             Q0 fails (count=1)
  release(1)                          Proof = nullopt ✓
  CAS(Open,1 → Open,0)
```

**Proof**: If `tryAdmit` succeeds before `tryMakeQuiescenceProof` reads `outstanding()`, Q0 observes count > 0 → Proof fails. If `tryAdmit` fails (state != Open), no new reservation created → Q0 may pass. **Linearization is consistent.**

### T19 — Proof success blocks all 4 paths

After Proof succeeds:
- packedState_ state is Closing or Closed
- `isAdmissionOpen()` returns false for all callers

| Path | Check | Behavior after Proof |
|---|---|---|
| Publication | `admission_.evaluate()` → `isShutdownInProgress()` | Returns RejectedShutdown |
| Recovery | `CoordinatorState::ShuttingDown` check | Already blocked (Coordinator stopped) |
| Build | `isShutdownInProgress()` in RebuildDispatch | Returns false |
| Retire | RCU epoch (not AdmissionReservation) | Epoch safety check fails → retire deferred |

### T20 — Enqueue failure rollback

```text
tryAdmit(1)           ✓
enqueue (queue full)  ✗ → release(1)
outstanding() == 0    ✓
```

**Implementation**: The `tryAdmit` + `release` pair wraps the enqueue call. If enqueue fails, `release(1)` is called before returning. This is the same pattern already used for `publicationIntentResidencyCount_` reservation-before-push.

---

## Gate A-J Verdict

| Gate | Condition | Status | Notes |
|---|---|---|---|
| **A** | Admission authority is singular | ✅ **PASS** | AdmissionState + count both in `ShutdownRuntime` |
| **B** | Reservation ≠ transport residency | ✅ **PASS** | 3 separate concepts confirmed (AdmissionReservation vs publicationIntentResidencyCount_ vs pendingIntentCount_) |
| **C** | 4-path lifetime confirmed | ✅ **PASS** | Table in Step 3 — Retire EXCLUDED |
| **D** | Retire RT boundary decided | ✅ **PASS** | Retire is OUTSIDE AdmissionReservation (RCU epoch handles RT) |
| **E** | Packed FSM scope decided | ✅ **PASS** | Only AdmissionState + version + count (32-bit); ShutdownPhase stays separate |
| **F** | G-H linearization = 1 point | ✅ **PASS** | Single `packedState_` CAS word for both tryAdmit and closeAdmission |
| **G** | Q0 source-of-truth decided | ✅ **PASS** | `outstanding()` method on ShutdownRuntime; AudioEngine cannot access packedState_ directly |
| **H** | joinProducers() semantics decided | ✅ **PASS** | bool return, count==0 check, caller retry (NonRT only), no Audio Thread wait |
| **I** | Proof sealing formalized | ✅ **PASS** | T18/T19/T20 all specified with race conditions |
| **J** | T18/T19/T20 feasible | ✅ **PASS** | All testable with existing CTest infrastructure |

---

## Conflicts identified between D101-28/D101-29 and current code

| # | D101-28/D101-29 proposal | Current code reality | Resolution |
|---|---|---|---|
| 1 | `AdmissionReservationAuthority` separate class in `RuntimeIntentCoordinator` | AdmissionState already in `ShutdownRuntime`; splitting across 2 objects breaks CAS atomicity | **REJECT** — keep in `ShutdownRuntime` |
| 2 | Packed FSM includes ShutdownPhase | ShutdownPhase uses sequential `publishAtomic`, not CAS | **REJECT** — keep ShutdownPhase separate |
| 3 | Q0 from `outstanding() == 0` | Q0 is hardcoded `true` at AudioEngine.h:4368 | **MUST FIX** in D101-31 |
| 4 | `joinProducers()` checks count==0 | `joinProducers()` is blind CAS (no count check) | **MUST FIX** in D101-31 |
| 5 | Retire may need RT CAS | Retire uses RCU epoch on Audio Thread | **CONFIRMED OUT OF SCOPE** — Retire excluded from AdmissionReservation |
| 6 | Recovery reservation in `RuntimeIntentCoordinator` | Recovery gate is `CoordinatorState::ShuttingDown` in `RuntimeIntentCoordinator` | AdmissionReservation wraps ALL 3 paths (Publication/Recovery/Build), but lives in `ShutdownRuntime` |

---

## Final Verdict: GO

**All 10 gates PASS.**

The D101-28 design intent can be implemented with the following authoritative contract:

1. **AdmissionPackedState**: single 32-bit atomic in `ShutdownRuntime` (AdmissionState 2b + version 6b + count 24b)
2. **AdmissionReservation covers 3 paths**: Publication, Recovery, Build — NOT Retire
3. **Retire boundary**: RCU epoch on Audio Thread, completely separate from AdmissionReservation
4. **G-H linearization**: single CAS word, `closeAdmission` wins race against `tryAdmit`
5. **Q0 source**: `outstanding()` method on `ShutdownRuntime`; AudioEngine calls method, does not access packed word
6. **joinProducers()**: returns `bool`, checks count==0, caller retries on NonRT
7. **ShutdownPhase**: stays separate — not packed into AdmissionPackedState

**Proceed to D101-31-Implementation.**

---

**Audit completed**: 2026-08-24
**Auditor**: D101-30 (read-only)
**Files changed**: 0 (audit only — evidence only)
