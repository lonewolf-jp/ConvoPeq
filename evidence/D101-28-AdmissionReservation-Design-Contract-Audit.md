# D101-28 — AdmissionReservation Design Contract / Race G-H Closure Specification

> **Phase**: D101-28 (read-only Design Contract Audit — NO code changes).
> **Scope**: Phase 1–7 of the admission reservation design contract for ConvoPeq.
> **Verdict**: See Gate section at end. This document is a design contract, not implementation.
> **Prerequisite**: D101-27 (NO-GO confirmed) — no admission reservation counter exists in current code.
> **Consequence of D101-27**: Q0 is hardcoded `true` at `AudioEngine.h:4368` — not measured.

---

## Phase 1 — Producer / Admission Full Inventory (4 Pathways)

**Code change prohibited.**

Each of the 4 admission pathways is traced from producer caller through shutdown gate to queue push, with thread context classification.

### Path A — Publication

```text
Producer: Timer Thread (RT) → commitRuntimePublication()
  ↓
entry: AudioEngine::commitRuntimePublication()
  ↓ (delegates through)
RuntimePublicationOrchestrator::trySubmit() → trySubmitImpl() → admission_.evaluate()
  ↓
shutdown gate: isShutdownInProgress() check (AudioEngine.h:4206) + PublicationAdmission::evaluate() (RuntimePublicationOrchestrator.cpp:288)
  ↓
reservation acquire candidate: NONE — no admission reservation counter exists
  ↓
queue push: enqueuePublicationIntent() → intentQueue_.push() (ISRRuntimePublicationCoordinator.h:361)
  ↓
failure rollback: publicationIntentResidencyCount_ fetchSub on push failure (h:373)
  ↓
consumer: CoordinatorLoop::runCoordinatorPhase() → processIntent() → PublishExecutor::executePublish()
  ↓
reservation release candidate: NONE — publicationIntentResidencyCount_ fetchSub on pop (ProcessIntent.cpp:73)
```

**Thread context:**
- **Producer/Caller**: Timer Thread (RT) at `AudioEngine.Timer.cpp:994` calls `commitRuntimePublication()`
- **Admission decision**: RuntimePublicationOrchestrator (Message Thread context — `submitPublishRequest` is called from `enqueuePublicationIntentForRuntimeCommit` which can be Timer or Message Thread)
- **Consumer**: CoordinatorWorker (NonRT) — processes intentQueue_ via processIntent
- **Key file**: `src/audioengine/AudioEngine.Timer.cpp:994`, `src/audioengine/AudioEngine.Commit.cpp:782`, `src/audioengine/RuntimePublicationOrchestrator.cpp:34`

### Path B — Recovery

```text
Producer: QuarantineIntentHandler (CoordinatorLoop / NonRT) → submitRecoveryIntent() → AudioEngine::submitQuarantine()
  ↓
entry: AudioEngine::submitRecoveryIntent() (AudioEngine.h:4428)
  ↓
shutdown gate: isShutdownInProgress() (AudioEngine.h:4200) + submitRecoveryRequest() state_ check (ISRRuntimePublicationCoordinator.cpp:563)
  ↓
reservation acquire candidate: NONE — no admission reservation; PendingRecoveryAdmission::reservationOwned is a single-slot boolean, not a count
  ↓
queue push: submitRecoveryRequest() → recoveryIntentQueue_.push() (ISRRuntimePublicationCoordinator.cpp:890)
  ↓
failure rollback: pendingIntentCount_ fetchSub on push failure (cpp:899) + durable admission fallback (cpp:910-920)
  ↓
consumer: Builder Loop (NonRT) → popRecoveryRequest() / takePendingRecoveryAdmission()
  ↓
reservation release candidate: NONE — pendingIntentCount_ fetchSub on pop (cpp:946)
```

**Thread context:**
- **Producer/Caller**: CoordinatorLoop (NonRT) — QuarantineIntentHandler runs inside processIntent
- **Admission gate**: `state_ == CoordinatorState::ShuttingDown` check in `submitRecoveryRequest` (h:563)
- **Consumer**: RebuildThread (NonRT) — takes from recoveryIntentQueue_ or takesPendingRecoveryAdmission
- **Key file**: `src/audioengine/ISRRuntimePublicationCoordinator.cpp:855`, `src/audioengine/AudioEngine.h:4428`

### Path C — Build / Rebuild

```text
Producer: Message Thread (UI/params) + Rebuild Thread + Deferred Retry
  ↓
entry: AudioEngine::submitRebuildIntent() (AudioEngine.RebuildDispatch.cpp:151)
  ↓
shutdown gate: isShutdownInProgress() (AudioEngine.RebuildDispatch.cpp:250)
  ↓
reservation acquire candidate: NONE — uses std::mutex (rebuildAdmissionIntentMutex_) + rebuildAdmissionPendingIntent_ struct
  ↓
enqueue: rebuildAdmissionPendingIntent_ struct write (AudioEngine.RebuildDispatch.cpp:215-231) → wake RebuildThread
  ↓
failure rollback: rebuildAdmissionPendingIntent_.valid = false when !rebuildOutstanding (cpp:213)
  ↓
consumer: RebuildThread (NonRT) → rebuild loop processes rebuildAdmissionPendingIntent_
  ↓
reservation release candidate: NONE — consumed by RebuildThread
```

**Thread context:**
- **Producers/Caller**: Message Thread (UI events, AudioEngine.Parameters.cpp:552), RebuildThread (deferred retry), AudioEngine.RebuildDispatch.cpp
- **Admission gate**: `isShutdownInProgress()` check (h:250) — phase-based, NOT AdmissionState FSM
- **Consumer**: RebuildThread (NonRT)
- **Key file**: `src/audioengine/AudioEngine.RebuildDispatch.cpp:151`
- **Note**: Uses `std::mutex` — NOT RT-safe. This is a NonRT path only.

### Path D — Retire

```text
Producer: Audio Thread (RT) → enqueueRetire() → enqueueDeferredDeleteNonRt()
  ↓
entry: AudioEngine::enqueueDeferredDeleteNonRt() (AudioEngine.h:4196)
  ↓
shutdown gate: isShutdownInProgress() (AudioEngine.h:4206) → shutdownReclaim() path
  ↓
reservation acquire candidate: NONE — retireBacklogCount_ is transport residency, not admission reservation
  ↓
queue push: ISRRetireRouter::enqueueRetire() → provider_->enqueueRetireTyped() (ISRRetireRouter.cpp:239)
  ↓
failure rollback: QueuePressure return → caller handles retry (enqueueWithRetry)
  ↓
consumer: CoordinatorLoop NonRT reclaim / tryReclaim()
  ↓
reservation release candidate: NONE — retireBacklogCount_ decremented on successful retire
```

**Thread context:**
- **Producer/Caller**: Audio Thread (RT) — `enqueueDeferredDeleteNonRt` called from RT DSP code
- **Admission gate**: `isShutdownInProgress()` (h:4206) — if true, redirects to `shutdownReclaim()` (safe path)
- **Consumer**: CoordinatorLoop NonRT / tryReclaim (NonRT)
- **Key file**: `src/audioengine/AudioEngine.Retire.cpp:47`, `src/audioengine/AudioEngine.h:4196`, `src/audioengine/ISRRetireRouter.cpp:239`
- **Note**: During shutdown, `enqueueRetire` → `enqueueDeferredDeleteNonRt` → `shutdownReclaim` — this is the Q7 no-resurrection path (obs.noResurrection = !isAdmissionOpen())

### Thread Classification Summary

| Producer | Thread | Path | RT-safe? |
|----------|--------|------|----------|
| Timer Thread | RT | A (Publication) | No (isShutdownInProgress check, but enqueue is NonRT-via-delegation) |
| Message Thread | NonRT | C (Build/Rebuild) | No (uses std::mutex) |
| CoordinatorLoop | NonRT | B (Recovery) | Yes (lock-free atomics) |
| RebuildThread | NonRT | C (Build/Rebuild retry) | Yes (lock-free ring) |
| Audio Thread | RT | D (Retire) | Partially (enqueueRetire is RT-safe via retireRT; enqueueWithRetry is NonRT-only) |
| QuarantineIntentHandler | NonRT (CoordinatorLoop) | B (Recovery) | Yes (runs in processIntent) |

**Important correction from D101-27**: D101-27 classified "Producer = Coordinator + Rebuild Thread." D101-28 corrects this:
- **Path A**: Production caller is Timer Thread (RT) → commitRuntimePublication
- **Path B**: Production caller is QuarantineIntentHandler within CoordinatorLoop (NonRT)
- **Path C**: Production callers include Message Thread (UI), RebuildThread (deferred retry)
- **Path D**: Production caller is Audio Thread (RT) via enqueueRetire

---

## Phase 2 — G-H Race Linearization Definition

The current code structure:
```text
check admission state       ← acquire-load (non-atomic vs enqueue)
    ↓
[context switch / race window]
    ↓
enqueue (push)              ← atomic queue operation
```

The required structure:
```text
atomically acquire reservation iff admission open
    ↓
enqueue
    ↓
release reservation
```

### Three Design Options Compared

#### Option A — AdmissionState + reservation count

```text
tryAdmit:
    atomically acquire reservation iff Open
        ↓
    enqueue
        ↓
    release reservation

closeAdmission:
    Open → Closing
        ↓
    reject new reservations
        ↓
    wait/drain outstanding reservations
        ↓
    Closed
```

**Pros**: Simple 2-atomic model (AdmissionState + reservationCount). Clear ordering.
**Cons**: Requires `closeAdmission` to drain outstanding reservations before transitioning to Closed — adds blocking to shutdown path. May require producer cooperation (wait-for-zero mechanism).

**Current code status**: `closeAdmission()` (ISRShutdown.cpp:415) performs CAS Open→Closing but does NOT drain reservations. `joinProducers()` (ISRShutdown.cpp:433) performs blind CAS Closing→Closed with no reservation check. **This option would require modifying both.**

#### Option B — Admission token / RAII reservation

```text
AdmissionToken token = tryAdmit();

token.valid()
    → enqueue
    → token.release()
```

Proof observes:
```text
reservationCount == 0
```

**Pros**: RAII guarantees exactly-one-release (INV-LIFE-16). Token carries linearization handle. Exception-safe.
**Cons**: Requires C++ stack discipline — harder to retrofit into existing C-style enqueue APIs. Token must be RT-safe (no destructor side-effects that could trigger allocation/mutex).

**Current code status**: No token mechanism exists. `publicationIntentResidencyCount_` and `pendingIntentCount_` are counters, not tokens.

#### Option C — Single atomic FSM + count (combined word)

```text
// Concept: pack AdmissionState + reservationCount into a single atomic<uint64_t>
// State: bits [63:60], Count: bits [59:0]

tryAdmit:
    loop:
        old = state_.load(acquire)
        if old.state != Open → reject
        new = old.state=Open, old.count+1
        if CAS(state_, old, new) → success, break
        // else retry

closeAdmission:
    loop:
        old = state_.load(acq_rel)
        if old.state == Open:
            new = old.state=Closing, old.count  // preserve count
            if CAS(state_, old, new) → transition started
        elif old.state == Closed:
            return
        // wait for count → 0, then transition to Closed
```

**Pros**: Single atomic operation for check+reserve. No TOCTOU. Count and state always consistent.
**Cons**: Packed word requires careful bit manipulation. Count field is 60-bit (sufficient). More complex to implement but most robust.

**Current code status**: `admissionState_` (AdmissionState, 1 byte) and `publicationIntentResidencyCount_` (uint64_t) are **separate atomics** — the TOCTOU is structurally present. They must be unified.

### Recommendation

**Option C (single atomic FSM + count)** is the most robust for the RT safety requirements:
- Single CAS eliminates TOCTOU between admission check and reservation
- No separate counter drift possible
- `closeAdmission` can atomically transition state and preserve count
- Proof can observe count==0 via single atomic load

Option A is simpler but introduces a drain step in `closeAdmission` that may not be RT-safe.
Option B is cleanest semantically but harder to retrofit.

---

## Phase 3 — Authority Ownership

### Candidates

| Candidate | Pros | Cons |
|-----------|------|------|
| **ShutdownRuntime** | Owns AdmissionState FSM, closeAdmission, joinProducers, tryMakeQuiescenceProof. Already the Proof/Permit authority. | Does not currently own any counter. |
| **RuntimeIntentCoordinator** | Owns `publicationIntentResidencyCount_`, `pendingIntentCount_`, all enqueue paths. Has `shutdownScheduler_` inner class. | Currently no `tryAdmit` or reservation API. |
| **AudioEngine** | Has `isShutdownInProgress()` check, `tryShutdownQuiescentReclaim`. Owner of EpochDomain/RetireRouter. | Too large — AudioEngine is a God Object. Violates single-responsibility. |
| **Independent AdmissionReservationAuthority** | Clean separation — dedicated class. RT-safe by design. | Requires new class, new wiring, new tests. |

### Decision

**AdmissionReservationAuthority = RuntimeIntentCoordinator (via ReservationScheduler inner class)**

Rationale:
1. `RuntimeIntentCoordinator` already owns all 4 enqueue paths (Path A-D)
2. Already has `shutdownScheduler_` — logical place for a `reservationScheduler_`
3. Already owns `publicationIntentResidencyCount_` and `pendingIntentCount_`
4. Already has friend relationship with `ShutdownRuntime` for identity binding
5. `AudioEngine` is explicitly excluded per the instruction: "AudioEngine に reservation authority を置く案は第一候補から外す"

**Authority responsibilities** (consolidated into RuntimeIntentCoordinator):

| Operation | Who | Current implementation | Required change |
|-----------|-----|----------------------|-----------------|
| Creates reservation | RuntimeIntentCoordinator | NO | Add `AdmissionReservationAuthority` inner class |
| Increments | RuntimeIntentCoordinator | `publicationIntentResidencyCount_` / `pendingIntentCount_` (path-specific) | Unify into single `admissionReservationCount_` |
| Releases | RuntimeIntentCoordinator | fetchSub on pop | RAII token release |
| Closes admission | ShutdownRuntime | `closeAdmission()` — state-only CAS | Must drain/invalidate reservations before Closed |
| Reads Q0 | AudioEngine (tryShutdownQuiescentReclaim) | Hardcoded `true` | Replace with `RuntimeIntentCoordinator::getOutstandingAdmissionReservations()` |
| Mutates | RuntimeIntentCoordinator only | Multiple counters | Single atomic FSM+count |

### Who calls what (post-implementation):

```text
Who creates reservation?  → RuntimeIntentCoordinator::tryAdmit()
Who increments?           → tryAdmit (CAS on packed state word)
Who releases?             → AdmissionToken::release() / RAII destructor
Who closes admission?     → ShutdownRuntime::closeAdmission() → signals RuntimeIntentCoordinator
Who reads Q0?             → AudioEngine::tryShutdownQuiescentReclaim → RuntimeIntentCoordinator::outstandingAdmissionReservations()
Who is allowed to mutate? → RuntimeIntentCoordinator only (single writer for state transitions)
```

---

## Phase 4 — `publicationIntentResidencyCount_` Handling

**Decision: NO reuse.** `publicationIntentResidencyCount_` cannot be repurposed as the admission reservation counter.

### Reasons (verbatim from instructions)

1. **Publication domain exclusive** — `publicationIntentResidencyCount_` is incremented only in `enqueuePublicationIntent()` (ISRRuntimePublicationCoordinator.h:372). It does NOT cover Recovery (Path B), Build/Rebuild (Path C), or Retire (Path D).

2. **Recovery/Build/Retire not represented** — Recovery uses `pendingIntentCount_` (separate counter). Build/Rebuild uses `rebuildAdmissionPendingIntent_` (struct, mutex-protected). Retire uses `retireBacklogCount_` and `pendingRetireCount_`.

3. **Semantic mismatch** — `publicationIntentResidencyCount_` = "Publish intent queue residency + producer reservation" (INV-X5-1). This is a **transport residency** counter, not an **admission reservation** counter. The code explicitly documents this separation:
   > "publicationIntentResidencyCount_ = intentQueue_ 内の Publish Intent 数 + producer enqueue reservation（並行中は >=、producer quiescence 後は ==）" — ISRRuntimePublicationCoordinator.h:540

4. **Not atomic with shutdown gate** — `publicationIntentResidencyCount_` is incremented AFTER the `CoordinatorState::ShuttingDown` check (h:361 vs h:372). The check and increment are not in a single atomic operation. This is the exact TOCTOU that Phase 2 addresses.

### Separate accounting requirement

Per REPAIR_PLAN2-dash2 design principle (G-B5):
```text
AdmissionReservation ≠ transport residency
```

The admission reservation must be:
- A **single counter** that ALL 4 paths increment before enqueue
- **Checked** by `closeAdmission()` before transitioning Open→Closing
- **Zero** before `joinProducers()` can transition Closing→Closed
- **RT-safe** (no malloc, no mutex, bounded)
- **Distinct** from `publicationIntentResidencyCount_`, `pendingIntentCount_`, `retireBacklogCount_`, `deferredRetireResidencyCount_`

---

## Phase 5 — Q0 Semantic Contract

### Current Q0 construction (AudioEngine.h:4368)

```cpp
obs.admissionReservationsZero = true;   // Q0: HARDCODED — NOT MEASURED
```

### Required Q0 measurement

```text
Q0 = AdmissionReservationAuthority::outstanding() == 0
```

### Temporal conflict analysis

#### Forward order: tryAdmit vs closeAdmission

```text
T1: producer (Path A/B/C/D) calls tryAdmit()
T2: shutdown thread calls closeAdmission() (Open → Closing)
T3: proof observer calls outstanding()
```

If T1 succeeds (reservation acquired) before T2 transitions to Closing:
- `outstanding() > 0` at T3 → Proof cannot yet satisfy Q0 → correctly blocked.

If T2 transitions to Closing before T1's tryAdmit:
- tryAdmit must see `Closing` and reject → `outstanding()` remains 0 → Proof satisfies Q0.

**Linearization point**: The single atomic CAS in `tryAdmit()` (or the transition in `closeAdmission()` that atomically sets state to Closing) determines the winner. The loser is guaranteed to observe the other's change.

#### Reverse order: closeAdmission vs tryAdmit

```text
T1: shutdown thread calls closeAdmission() (Open → Closing)
T2: proof observer calls outstanding()
T3: producer calls tryAdmit()
```

If the reservation count is 0 at T2 (no producers acquired yet), Q0 is satisfied. But then at T3, tryAdmit sees Closing and rejects. **Invariant holds**: no new reservations after Proof observation.

#### Invariant: Proof Sealing

```text
Proof success (Q0 satisfied)
⇒
No subsequent tryAdmit() can succeed
```

This is enforced because:
- `tryAdmit()` checks `AdmissionState` atomically
- If `closeAdmission()` has run (Open→Closing), state is Closing or Closed
- `tryAdmit()` in Closing/Closed state returns invalid token → no reservation acquired
- Even if `tryAdmit()` races with `closeAdmission()`, the CAS ensures one wins — if closeAdmission wins, tryAdmit sees Closing and rejects

### Q0 implementation design

```cpp
class AdmissionReservationAuthority {
    // Packed: AdmissionState (4 bits) + reservationCount (60 bits)
    std::atomic<uint64_t> state_;

    enum State : uint8_t { Open=0, Closing=1, Closed=2, Faulted=3 };

public:
    struct Token {
        bool valid_;
        AdmissionReservationAuthority* authority_;
        // RAII: release on destruction (RT-safe — just a fetchSub)
    };

    [[nodiscard]] Token tryAdmit() noexcept {
        uint64_t old = state_.load(acquire);
        do {
            State s = extractState(old);
            if (s != Open) return {false, this};  // reject
            uint64_t new = pack(Open, extractCount(old) + 1);
            if (CAS(state_, old, new, acq_rel, acquire)) {
                return {true, this};  // acquired
            }
            // retry: state may have changed
        } while (true);
    }

    void release(Token t) noexcept {
        if (t.valid_) {
            fetchSub(state_, pack(0, 1), acq_rel);  // atomic count decrement
        }
    }

    uint64_t outstanding() const noexcept {
        return extractCount(state_.load(acquire));
    }

    void closeAdmission() noexcept {
        // Open → Closing (preserve count)
        // Does NOT wait for count == 0 — that's joinProducers' job
    }

    void joinProducers() noexcept {
        // Closing → Closed (only if count == 0)
        // CAS: requires state==Closing AND count==0
    }
};
```

### Q0 vs Q1/Q2 relationship

| Q | Definition | Current | Post-D101-28 |
|---|---|---|---|
| Q0 | `outstandingAdmissionReservations() == 0` | Hardcoded `true` | Measured from `AdmissionReservationAuthority::outstanding()` |
| Q1 | `AdmissionState == Closed` | `admissionState() == AdmissionState::Closed` | `AdmissionState` inside `AdmissionReservationAuthority` |
| Q2 | `allProducersJoined` | Hardcoded `true` | Thread join confirmation (unchanged) |

**Q0 is NOT derivable from Q1 + Q2**:
- Q1 (Closed) means `closeAdmission` + `joinProducers` succeeded
- But `joinProducers` currently does NOT check reservation count
- Post-D101-28: `joinProducers` will CAS Closing→Closed only if `count == 0`
- **However**: Even with this fix, Q0 measures "zero outstanding reservations" which is conceptually distinct from "producers joined" — a reservation can be outstanding after producer threads have been joined if the enqueue was in-flight before join.

**Conclusion**: Q0 requires **independent measurement** from a dedicated counter. Q1+Q2 can ensure no new reservations, but only Q0 directly measures outstanding reservations.

---

## Phase 6 — Required Invariants

### INV-LIFE-13
```text
outstandingAdmissionReservations == 0
⇔
all successfully acquired reservations have been released
```
**Status**: Requires implementation of `admissionReservationCount` with atomic acquire/release tracking. Currently UNSATISFIABLE (no counter exists).

### INV-LIFE-14
```text
Closed admission
⇒
new AdmissionReservation cannot be acquired
```
**Status**: `closeAdmission()` transitions Open→Closing. `tryAdmit()` must check state and reject if not Open. Currently UNSATISFIABLE (no `tryAdmit` function exists — paths check `isShutdownInProgress()` which is phase-based, not state-based).

### INV-LIFE-15
```text
QuiescenceProof success
⇒
new AdmissionReservation cannot subsequently succeed
```
**Status**: Proof generation checks Q0 (`admissionReservationsZero`). Currently UNSATISFIABLE (Q0 hardcoded true — Proof can succeed even if reservations are outstanding).

### INV-LIFE-16
```text
every successful AdmissionReservation
⇒
exactly one release
```
**Status**: Requires RAII token pattern. Currently UNSATISFIABLE (no token mechanism; `publicationIntentResidencyCount_` and `pendingIntentCount_` use manual fetchAdd/fetchSub with potential for missed release on exception paths).

### INV-LIFE-17
```text
reservation accounting
≠
transport residency accounting
```
**Status**: CONFIRMED by current code structure. `publicationIntentResidencyCount_` (X5 §6.5), `pendingIntentCount_` (P2-1 §1.1.1), `retireBacklogCount_`, `deferredRetireResidencyCount_` are all documented as separate domains. The admission reservation counter must be a NEW counter, distinct from all existing ones.

### ISR Constraints (existing, must be maintained)

| Constraint | Description | Current compliance |
|---|---|---|
| ISR-LIFE-05 | no malloc | `enqueuePublicationIntent`, `submitRecoveryRequest` use lock-free rings — ✅ |
| ISR-LIFE-06 | no mutex wait | `submitObserve`, `submitRecoveryRequest` — ✅. `submitRebuildIntent` uses `std::mutex` (rebuildAdmissionIntentMutex_) — ⚠️ (NonRT only, acceptable) |
| ISR-LIFE-07 | bounded/nonblocking queue push | `MpscBoundedRing::push` returns false on full — ✅ |
| ISR-LIFE-08 | atomic / RT-safe mutation | All counters use `std::atomic` with explicit memory ordering — ✅ |

**Note**: `submitRebuildIntent` (Path C) uses `std::mutex` (`rebuildAdmissionIntentMutex_`) — this violates ISR-LIFE-06. However, Path C is Message Thread/NonRT only (UI params, rebuild dispatch), so this is acceptable. The admission reservation counter must NOT introduce mutex on RT paths.

---

## Phase 7 — Test Specification (Specification Only)

| Test | Content |
|------|---------|
| T14 | `tryAdmit` vs `closeAdmission` single conflict — verify one wins, loser rejects |
| T15 | `tryAdmit` vs `closeAdmission` 100k round stress — verify no reservation leak, count always consistent |
| T16 | After `closeAdmission()` transitions to Closed, all subsequent `tryAdmit()` calls return invalid token |
| T17 | Reservation acquire/release balance — N acquires, N releases, outstanding == 0 |
| T18 | `tryMakeQuiescenceProof()` vs `enqueuePublicationIntent()` race — proof succeeds ⟹ no new reservation can be acquired |
| T19 | After Proof success, `tryAdmit()` on all 4 paths returns invalid (no-resurrection) |
| T20 | Queue full during `tryAdmit` + `enqueue` — reservation must be rolled back (token release) |
| T21 | After all producers join, `outstandingAdmissionReservations()` must be 0 (verified by Proof) |
| T22 | Concurrent `release()` from multiple threads — no underflow (count never goes negative) |
| T23 | All 4 paths (Publication/Recovery/Build/Retire) share single `admissionReservationCount_` — verify accounting across paths |
| T24 | Audio Thread never calls `tryAdmit()` — verify Path D (Retire) admission gate is enforced at NonRT boundary |

**T18/T19 are critical**: These verify the core G-H race closure. T18 tests the race where `tryAdmit` and `tryMakeQuiescenceProof` execute concurrently. T19 tests the post-Proofing invariant that no new admission can succeed.

---

## Gates

| Gate | Status | Evidence |
|------|--------|----------|
| **Gate A — Producer completeness** | ✅ | Phase 1 inventories all 4 pathways with explicit thread classification |
| **Gate B — Authority singularization** | ✅ | Phase 3 designates RuntimeIntentCoordinator as sole authority (NOT AudioEngine) |
| **Gate C — Linearization** | ✅ | Phase 2 specifies Option C (single atomic FSM+count) with CAS-based linearization point |
| **Gate D — Q0 measurability** | ✅ | Phase 5 defines `outstanding()` measured from dedicated counter, replacing hardcoded `true` |
| **Gate E — Transport separation** | ✅ | Phase 4 formally excludes reuse of `publicationIntentResidencyCount_` / `pendingIntentCount_` / `retireBacklogCount_` |
| **Gate F — RT safety** | ✅ | Phase 3 + 6 specify atomic-only operations, no malloc/mutex on RT paths; ISR-LIFE-05-08 compliance |
| **Gate G — Proof sealing** | ✅ | Phase 5 defines invariant: Proof success ⟹ no subsequent admission success (T18/T19) |
| **Gate H — No implementation yet** | ✅ | This document is read-only; no `src/` changes made. D101-29 will handle implementation. |

---

## Open Questions / Unknowns

1. **Q0 observation source**: Who populates `obs.admissionReservationsZero` in the production Proof path? Currently hardcoded in `AudioEngine.h:4368`. Post-D101-28, this must call `RuntimeIntentCoordinator::outstandingAdmissionReservations()`.

2. **Path C mutex concern**: `submitRebuildIntent` (Path C) uses `std::mutex` (`rebuildAdmissionIntentMutex_`). The admission reservation acquire/release must NOT add mutex to this path — the existing mutex is for rebuild coalescing, separate from admission.

3. **Path D RT boundary**: Retire intent (Path D) is enqueued from Audio Thread (RT) via `enqueueRetire` → `enqueueDeferredDeleteNonRt`. If `tryAdmit` is added to this path, it must be RT-safe. The current gate (`isShutdownInProgress()`) is RT-safe (acquire-load). The admission reservation acquire must also be RT-safe.

4. **Shutdown sequence ordering**: Current shutdown sequence at `AudioEngine.Processing.ReleaseResources.cpp:198`:
   ```
   shutdownCoordinatorLoop();  ← join CoordinatorWorker
   stopRebuildThread();        ← join RebuildThread
   shutdownRuntime_.closeAdmission();      ← Open→Closing
   shutdownRuntime_.joinProducers();       ← Closing→Closed
   ```
   Post-D101-28: `joinProducers()` must NOT transition to Closed until `admissionReservationCount_ == 0`. This means:
   - All 4 enqueue paths must have completed (including retries from overflow rings)
   - All in-flight reservations must have been released
   - This adds a dependency between thread join and reservation drain

5. **`enqueuePublicationIntent` Path A dual gate**: Path A has TWO gate checks:
   - `RuntimePublicationOrchestrator::trySubmitImpl` checks `engine_.isShutdownInProgress()` (cpp:288)
   - `enqueuePublicationIntent` checks `CoordinatorState::ShuttingDown` (h:361)

   Post-D101-28, these may need to be unified with the AdmissionState FSM. The `trySubmitImpl` path also has a `PublicationAdmission::evaluate()` step that is separate from admission state.

---

## Relationship to D101-27

| Aspect | D101-27 Status | D101-28 Response |
|--------|----------------|----------------|
| Reservation counter exists | NO | Design the counter architecture (Phase 2/3) |
| tryAdmit() exists | NO | Design the API (Option A/B/C comparison) |
| Race G-H closed | NO | Define linearization point (Phase 2, Option C) |
| Q0 measured | NO (hardcoded `true`) | Define Q0 measurement contract (Phase 5) |
| Audio Thread boundary | No violation | Must maintain RT safety for new counter |
| Producer list | Coordinator + RebuildThread only | Expanded to all actual producers (Phase 1) |

D101-27 concluded **BLOCKED** (NO-GO for implementation). D101-28 provides the **design contract** that D101-29 will implement. D101-28 does NOT unlock implementation — it freezes the design.

---

## D101-28 Verdict

**DESIGN CONTRACT FROZEN — Not blocked, not approved for implementation.**

D101-28 establishes the complete design contract for AdmissionReservation:
- 4 pathways inventoried with thread context (Phase 1)
- Linearization strategy selected (Option C: single atomic FSM+count) (Phase 2)
- Single authority designated (RuntimeIntentCoordinator) (Phase 3)
- Transport residency separation confirmed (Phase 4)
- Q0 measurement contract defined (Phase 5)
- 5 core invariants + 4 ISR constraints formalized (Phase 6)
- 11 test specifications defined (Phase 7)

**Next step**: D101-29 (Implementation Authorization Gate) — must verify the design contract against production feasibility before any code changes.
