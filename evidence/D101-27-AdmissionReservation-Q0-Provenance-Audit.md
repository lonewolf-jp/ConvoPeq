# D101-27 — AdmissionReservation / Q0 Provenance Audit

> **Phase**: D101-27 (read-only audit — NO code changes).
> Scope: Investigate whether an AdmissionReservation mechanism (reservation counter for admission producers) exists in current code, and verify race-free linearization point for the closeAdmission vs tryAdmit race (G-H).
> All findings are source-traced via read/grep of the current tree.
> Verdict at the foot of this file.

---

## 1. Baseline

- **ConvoPeq.md Generated**: 2026-08-24 (regenerated via `python output_sourcecode_markdown.py`)
- **git status**: ` M ConvoPeq.md` only — no source (`src/`) modifications.
- **git diff --check**: 0 whitespace errors (Step 1 clean).

---

## 2. Existing Reservation Mechanism

### 2.1 Search result summary

Searched for: `AdmissionReservation`, `PendingReservation`, `admissionReservation`, `reservationCount`, `pendingReservation`, `admissionReservations`, `tryAdmit`, `releaseAdmission`, `joinProducers`, `closeAdmission`.

| Keyword | Match in production? | Location |
|---|---|---|
| `AdmissionReservation` | No (only in comments) | ISRShutdown.h:256 (comment), ISRLifetimeProof.h:89/109 (field) |
| `PendingReservation` | No | — |
| `admissionReservation` | No | — |
| `reservationCount` | No | — |
| `pendingReservation` | No | — |
| `admissionReservations` | Yes (Q0 field) | ISRShutdown.h:257, ISRShutdown.cpp:353, AudioEngine.h:4368 |
| `tryAdmit` | No | — |
| `releaseAdmission` | No | — |
| `joinProducers` | Yes | ISRShutdown.h:290, ISRShutdown.cpp:433 |
| `closeAdmission` | Yes | ISRShutdown.h:289, ISRShutdown.cpp:415, AudioEngine.Processing.ReleaseResources.cpp:198 |

### 2.2 Reservation counter existence

**NO dedicated admission reservation counter exists.**

The `ShutdownRuntime` class has the following atomic counters (ISRShutdown.h:325-353):

```cpp
std::atomic<uint32_t> sh1CallbackCount_{0};        // shutdown callback count (telemetry)
std::atomic<uint32_t> sh2ActiveCrossfade_{0};     // active crossfade count (telemetry)
std::atomic<uint32_t> sh3PendingRetire_{0};       // pending retire count (telemetry)
std::atomic<uint32_t> sh4ObserverCount_{0};       // active observer count (telemetry)
std::atomic<uint32_t> sh5LateCallbackCount_{0};   // late callback count (telemetry)
std::atomic<uint32_t> sh6PostStopEnqueueCount_{0}; // post-stop enqueue count (Q6)
std::atomic<AdmissionState> admissionState_{AdmissionState::Open}; // Q1/Q7 FSM
```

None of these is an admission reservation counter.

The `publicationIntentResidencyCount_` in `ISRRuntimePublicationCoordinator.h:548` tracks **publication intent queue residency + producer reservation** for the Publication path — but it is explicitly documented as a **separate semantic domain** (not a Lifetime Budget admission reservation). Per prior session context:

> `publicationIntentResidencyCount_` — Publish Intent residency — **NO** — ISR Publish Intent (not Lifetime Budget reservation)

The `PendingRecoveryAdmission::reservationOwned` field (ISRRuntimePublicationCoordinator.h:679) is a **single-slot boolean** for per-admission tracking, not a count.

### 2.3 Admission Gate Architecture (4 Pathways)

The production code uses **two layers** of admission gating — NOT a reservation counter:

1. **Primary gate**: `AudioEngine::isShutdownInProgress()` (AudioEngine.h:1480-1490)
   - Checks `lifecycleState` (Releasing/Destroyed) OR `shutdownRuntime_.isShutdownInProgress()`
   - `ShutdownRuntime::isShutdownInProgress()` (ISRShutdown.cpp:153): checks `phase_ != Running && !isTerminalPhase`
   - This is the **actual** gate used in all 4 pathways (Timer, Transition, Retire, RebuildDispatch)

2. **Secondary gate (defense-in-depth)**: `CoordinatorState::ShuttingDown` check
   - `enqueuePublicationIntent` (ISRRuntimePublicationCoordinator.h:361): `if (state_ == ShuttingDown) return false;`
   - `submitRecoveryRequest` (ISRRuntimePublicationCoordinator.cpp:563): same pattern

3. **AdmissionState FSM** (ISRShutdown.h:296-297): `closeAdmission()` / `joinProducers()` / `isAdmissionOpen()`
   - Only used for Q1/Q7 in Proof construction — NOT for enqueue gating
   - `isAdmissionOpen()` is used as `!isAdmissionOpen()` for Q7 NoResurrection check

**Key insight**: The `PublicationAdmission` module (PublicationAdmission.cpp/h) provides decision-level admission control (health state, pressure, generation staleness, fading) but does NOT track reservation counts. It returns `Decision` enums (Accepted/Rejected/Deprecated/Deferred).

### 2.4 Answer

**D. reservation は未実装 (reservation is NOT implemented)**

Q0 (`admissionReservationsZero`) is currently a hardcoded `true` literal at `AudioEngine.h:4368`:

```cpp
obs.admissionReservationsZero = true;   // Q0: admission reservations は producer join 後 0
```

There is no mechanism to **measure** outstanding admission reservations. The existing counters (`publicationIntentResidencyCount_`, `pendingIntentCount_`) are in separate semantic domains and are explicitly documented as NOT Lifetime Budget reservations.

---

## 3. tryAdmit Provenance

### 3.1 Admission gate mechanism

No `tryAdmit()` function exists. Admission gating happens at the **enqueue path** via two independent layers:

**Layer 1 (primary gate)**: `AudioEngine::isShutdownInProgress()` — checked at every producer enqueue site:
- `AudioEngine.Timer.cpp:406,734,770,817,820,1192` — Timer publish/retry paths
- `AudioEngine.Transition.cpp:15` — Runtime transition
- `AudioEngine.Commit.cpp:195,467` — Commit publish / post-stop enqueue
- `AudioEngine.RebuildDispatch.cpp:241,287,380,437,479,515,838` — Builder dispatch
- `AudioEngine.Retire.cpp:47` — Retire path

`isShutdownInProgress()` (AudioEngine.h:1480-1490):
```cpp
bool isShutdownInProgress() const noexcept {
    const auto state = consumeAtomic(lifecycleState, std::memory_order_acquire);
    const bool lifecycleShutdown = (state == EngineLifecycleState::Releasing
                                 || state == EngineLifecycleState::Destroyed);
    return lifecycleShutdown || shutdownRuntime_.isShutdownInProgress();
}
```

**Layer 2 (secondary, defense-in-depth)**: `CoordinatorState::ShuttingDown` check in `enqueuePublicationIntent` (ISRRuntimePublicationCoordinator.h:361) and `submitRecoveryRequest` (ISRRuntimePublicationCoordinator.cpp:563).

**AdmissionState FSM** (ISRShutdown.h:165-169, 289-297): `closeAdmission()` / `joinProducers()` / `isAdmissionOpen()` — only used for Q1/Q7 in Proof construction, NOT for enqueue gating.

### 3.2 Producer lifecycle

Producers are joined sequentially at `ReleaseResources.cpp:185-199`:

### 3.2 Producer lifecycle

Producers are joined sequentially at `ReleaseResources.cpp:185-199`:

```cpp
shutdownCoordinatorLoop();  // join Coordinator Worker
stopRebuildThread();        // join Builder Worker
// ... (then):
shutdownRuntime_.closeAdmission();  // Open→Closing
shutdownRuntime_.joinProducers();   // Closing→Closed
```

**Producer = Coordinator Loop + Rebuild Thread.** Both are joined via `jthread`/mutex before `closeAdmission()` runs.

### 3.3 RT boundary

- `closeAdmission()` uses `compareExchangeAtomic` with `std::memory_order_acq_rel` — RT-safe, no malloc.
- `joinProducers()` uses `compareExchangeAtomic` with `std::memory_order_acq_rel` — RT-safe, no malloc.
- Enqueue paths (`enqueuePublicationIntent`, `submitRecoveryRequest`) check `CoordinatorState::ShuttingDown` via acquire-load — RT-safe.
- **No reservation counter increment/decrement touches the Audio Thread.** All producer admission tracking is NonRT side.

### 3.4 Answer

tryAdmit does not exist as a named function. Admission producers are:
- **Producer 1**: Coordinator Loop (spawns Publication + Recovery intents into intentQueue_)
- **Producer 2**: Rebuild Thread (processes intentQueue_, spawns Build intents)

Both are joined (`shutdownCoordinatorLoop()` + `stopRebuildThread()`) before `closeAdmission()`.

---

## 4. Shutdown Provenance

### 4.1 closeAdmission()

`ISRShutdown.cpp:415-430`:
```cpp
void ShutdownRuntime::closeAdmission() noexcept {
    AdmissionState expected = AdmissionState::Open;
    if (convo::compareExchangeAtomic(admissionState_, expected, AdmissionState::Closing,
                                     std::memory_order_acq_rel, std::memory_order_acquire)) {
        (void)convo::fetchAddAtomic(shutdownGeneration_, ..., std::memory_order_acq_rel);
    }
}
```
- Transitions: Open→Closing (CAS)
- Advances `shutdownGeneration_` on successful transition
- No producer join needed — state machine only

### 4.2 joinProducers()

`ISRShutdown.cpp:433-441`:
```cpp
void ShutdownRuntime::joinProducers() noexcept {
    AdmissionState expected = AdmissionState::Closing;
    convo::compareExchangeAtomic(admissionState_, expected, AdmissionState::Closed,
                                 std::memory_order_acq_rel, std::memory_order_acquire);
}
```
- Transitions: Closing→Closed (CAS, requires state==Closing)
- This is a **blind CAS** — does NOT verify reservation count == 0
- It only verifies state was Closing

### 4.3 Q0 construction

`AudioEngine.h:4361-4396` (`tryShutdownQuiescentReclaim`):
```cpp
obs.admissionReservationsZero = true;   // Q0: HARDCODED TRUE
obs.allProducersJoined = true;          // Q2: HARDCODED TRUE
...
auto proof = shutdownRuntime_.tryMakeQuiescenceProof(obs);
```

`ISRShutdown.cpp:348-388` (`tryMakeQuiescenceProof`):
```cpp
const bool q0 = observation.admissionReservationsZero;  // read from observation
const bool q1 = (admissionState() == AdmissionState::Closed);  // measured
const bool q2 = observation.allProducersJoined;  // read from observation
// ...
if (!(q0 && q1 && q2 && q3 && q4 && q5 && q6 && q7))
    return std::nullopt;
```

### 4.4 Answer

- `closeAdmission`: Open→Closing, atomic CAS, increments generation
- `joinProducers`: Closing→Closed, blind CAS (no reservation check)
- **Q0 is constructed from a hardcoded literal `true`, NOT measured from any counter**
- Producer join ≠ reservation release — the two are **not linked** in current code

---

## 5. Race G-H

### 5.1 The race

The race involves the **producer enqueue path** vs the **shutdown admission closure**:

```text
Producer (NonRT side, e.g., Coordinator Loop, Rebuild Thread, Timer):
    1. isShutdownInProgress() / CoordinatorState::ShuttingDown  (acquire-load check)
    2. [RACE WINDOW — closeAdmission() can transition Open→Closing here]
    3. fetchAdd publicationIntentResidencyCount_  (reservation AFTER gate check)
    4. intentQueue_.push()

Shutdown Thread:
    closeAdmission()          → Open→Closing (CAS)
    joinProducers()           → Closing→Closed (CAS)
```

### 5.2 Non-Atomic Admission Gate Patterns

Four patterns found in production code — all are **check-then-act without atomic coordination**:

**Pattern A — isShutdownInProgress check (Timer/Transition/Commit paths):**
```cpp
// AudioEngine.Timer.cpp:406
if (isShutdownInProgress())     // acquire-load check
    return;                      // — RACE WINDOW: closeAdmission fires here —
enqueuePublicationIntent();      // push without reservation tracking
```

**Pattern B — CoordinatorState check (Publication path):**
```cpp
// ISRRuntimePublicationCoordinator.h:361
if (convo::consumeAtomic(state_, std::memory_order_acquire) == CoordinatorState::ShuttingDown)
    return false;                // — RACE WINDOW —
// publicationIntentResidencyCount_++  happens AFTER gate check
intentQueue_.push(prepared);     // push without atomic gate coordination
```

**Pattern C — PublicationAdmission (decision-based, not counter-based):**
```cpp
// PublicationAdmission.cpp:11
if (engine.isShutdownInProgress())   // check
    return Decision::RejectedShutdown; // — RACE WINDOW —
// No reservation acquire; decision returned
```

**Pattern D — Phase-based gates (Rebuild/Builder paths):**
```cpp
// AudioEngine.RebuildDispatch.cpp:241
if (isShutdownInProgress())   // check
    return;                    // — RACE WINDOW —
// submitRecoveryIntent / enqueue without reservation tracking
```

### 5.3 Linearization point

There is **NO** well-defined linearization point that atomically:
1. Checks admission state
2. Acquires a reservation
3. Enqueues work

The current structure is:
```
check state → [RACE WINDOW] → acquire residency → push
```

Instead of the required:
```
acquire reservation (atomic) → check state (atomic) → push → release reservation (atomic)
```

**Critical finding**: The `publicationIntentResidencyCount_` in ISRRuntimePublicationCoordinator.h:548 is incremented AFTER the `CoordinatorState::ShuttingDown` check (h:361 vs h:372). It is NOT used for admission gating — it's a residency/telemetry counter. It cannot serve as an admission reservation because:
1. It's incremented after the gate check (race window)
2. It's never checked by `closeAdmission()` or `joinProducers()`
3. It counts publication intents only — not all 4 admission pathways

### 5.4 Verdict

**Race G-H is NOT closed at the reservation level.**

The admission gate is **state-machine-based** (Open→Closing→Closed via AdmissionState FSM + isShutdownInProgress phase check). `closeAdmission()` and `joinProducers()` operate on the `AdmissionState` FSM, which is orthogonal to `isShutdownInProgress()` (which checks `ShutdownPhase` in ISRShutdown.cpp:153). This creates a **layer mismatch**:

- Producers check `isShutdownInProgress()` (phase-based) — `phase_ != Running && !isTerminalPhase`
- Shutdown closure uses `closeAdmission()`/`joinProducers()` (state-based) — `AdmissionState` FSM
- These are **two different atomic variables** with no coordination

The comment at ISRShutdown.h:256 acknowledges this:
```cpp
// Q0: OutstandingAdmissionReservations == 0（第九者必須修正2 — close vs enqueue race を閉じる）
```
"Ninth-party required fix 2 — closing the close vs enqueue race."

---

## 6. Q0 / Q1 / Q2 Relationship

### 6.1 Definitions

| Q | Meaning | Current source |
|---|---|---|
| Q0 | OutstandingAdmissionReservations == 0 | Hardcoded `true` (AudioEngine.h:4368) |
| Q1 | AdmissionState == Closed | `admissionState() == AdmissionState::Closed` (ISRShutdown.cpp:355) |
| Q2 | AllProducersJoined | Hardcoded `true` (AudioEngine.h:4369) |

### 6.2 Are they derivable?

**Q0 is NOT derivable from Q1 + Q2.**

- Q1 (Closed) means `closeAdmission()` + `joinProducers()` succeeded. But `joinProducers()` is a blind CAS — it does NOT check reservation count.
- Q2 (all producers joined) means threads stopped. But **producer threads can be joined while pending enqueue operations are still in flight** — the threads are stopped, but work they enqueued before stopping may still be in the queue.
- Q0 requires **zero outstanding reservations** — a distinct semantic from thread join status.

**Proof that Q0 is independent**: If Q1+Q2 implied Q0, the code would not need to hardcode Q0 as `true`. The fact that Q0 is hardcoded (rather than derived) proves the developers recognized the independence.

### 6.3 Answer

**Q0 is INDEPENDENT** of Q1/Q2. It is not a derived predicate.

---

## 7. INV-LIFE-13

### Definition

```
admissionReservations == 0
⇔
成功した AdmissionReservation が全て release 済み
```

### Can current implementation satisfy it?

**NO.**

There is no `admissionReservations` counter in the codebase. The concept is defined only in comments (ISRShutdown.h:256). Q0 is a hardcoded `true` literal.

### Answer

**INV-LIFE-13 cannot be satisfied** — no reservation accounting mechanism exists.

---

## 8. Verdict

### Step 9 Classification

```
reservation の実体なし     ← MATCH: no counter exists
or
tryAdmit/closeAdmission race が未解決  ← MATCH: Race G-H is NOT closed
or
Audio Thread boundary に侵入  ← NO VIOLATION: all admission operations are NonRT
```

**→ BLOCKED**

### Rationale

1. **No reservation counter exists** — the codebase has no `admissionReservations` counter, no `tryAdmit()`/`releaseAdmission()` functions, and no reservation acquire/release protocol. The `publicationIntentResidencyCount_` and `PendingRecoveryAdmission::reservationOwned` are in different semantic domains.

2. **Race G-H is unresolved** — there is no atomic check-reserve-enqueue linearization point. The admission gate is purely state-machine-based (Open→Closing→Closed), without reservation accounting. The enqueue paths perform a non-atomic `check → increment → push` sequence with a race window between check and increment.

3. **Q0 ≠ Q1+Q2** — producer join does not imply zero reservations. The code acknowledges this by hardcoding Q0 as `true` rather than deriving it.

4. **Audio Thread boundary is NOT violated** — all admission state mutations (`closeAdmission`, `joinProducers`) and enqueue paths operate on the NonRT side (Coordinator Loop, Rebuild Thread, Message Thread shutdown context). The Audio Thread never touches admission state.

### D101-26 note

D101-26's PASS-A classification was appropriate for that audit's scope (verifying production-wiring of Q0-Q7). However, Q0 was confirmed as a **literal `true`**, and D101-26 explicitly noted: "Low — set only inside `tryShutdownQuiescentReclaim` which is NonRT-shutdown context post-join; reservations are structurally impossible once producers joined." This is an **assumption**, not a **measured invariant**. D101-27 confirms that assumption is unwarranted — reservations are NOT structurally impossible; they are simply **not tracked**.

---

## 9. Implementation authorization

### PASS-A
```
既存 reservation counter が存在 → atomic/RT-safe → shutdown lifecycle と整合 → Q0 に接続可能
```
**NOT MET** — no reservation counter exists.

### PASS-B
```
reservation semantics は存在するが accounting authority が分散
```
**NOT MET** — reservation semantics do not exist as a coherent mechanism. The `publicationIntentResidencyCount_` is a residency counter for a different domain.

### BLOCKED
```
reservation の実体なし  ✓
tryAdmit/closeAdmission race が未解決  ✓
Audio Thread boundary に侵入  ✗ (no violation)
```

**→ NO-GO**

---

## 10. Next steps

Before implementing Q0 real measurement:

1. **Define the Reservation FSM** (Phase B2 of REPAIR_PLAN2-dash2):
   - Where does `Reservation` struct live?
   - Who owns the counter? (ShutdownRuntime or ISRRuntimePublicationCoordinator?)
   - What memory ordering guarantees are needed?
   - How does it compose with the 4 admission pathways (Publication/Recovery/Build/Retire)?

2. **Close Race G-H**:
   - The linearizable operation must be: `if (admissionOpen) { reservation++; } else { return false; }` as a single atomic operation.
   - Consider: `closeAdmission()` must not proceed until all in-flight reservations are drained (not just producer joined).

3. **Q0 measurement**:
   - Q0 should be measured from the actual reservation counter at Proof construction time.
   - The current literal `true` at `AudioEngine.h:4368` must be replaced with a measured value.

4. **RT boundary protection**:
   - The reservation counter must NOT be touched by the Audio Thread.
   - `tryAdmit()` must satisfy: ISR-LIFE-05 (no malloc), ISR-LIFE-06 (no mutex wait), ISR-LIFE-07 (bounded/nonblocking queue push), ISR-LIFE-08 (atomic mutation).

### What NOT to do next (D101-27 scope)

- Do NOT implement Q0 real measurement yet — design must be fixed first.
- Do NOT replace `true` with `false` or any other literal.
- Do NOT touch Q5 (EpochSettled) — Phase C follows Phase B2.
- Do NOT use `isFullyDrained()` as Q0 substitute — it is an aggregate drain predicate, not a reservation counter.

---

## D101-27 Reporting (Step 9 format)

### 1. Baseline
ConvoPeq.md: Regenerated 2026-08-24
git status: clean (ConvoPeq.md only)
git diff --check: 0

### 2. Existing Reservation Mechanism
Type: N/A
Owner: N/A
Counter: N/A (none exists)
Acquire: N/A
Release: N/A

### 3. tryAdmit Provenance
Caller: N/A (no tryAdmit exists)
Admission gate: AdmissionState FSM (ISRShutdown.h:165, ISRShutdown.cpp:415) + CoordinatorState::ShuttingDown (ISRRuntimePublicationCoordinator.h:361)
Reservation: None
Queue: intentQueue_ (MpscBoundedRing, ISRRuntimePublicationCoordinator.h:530)
RT boundary: All NonRT (Coordinator Loop, Rebuild Thread, Message Thread)

### 4. Shutdown Provenance
closeAdmission: ISRShutdown.cpp:415 — Open→Closing, CAS + gen increment
joinProducers: ISRShutdown.cpp:433 — Closing→Closed, blind CAS
reservation drain: N/A (no reservation to drain)
Q0 construction: AudioEngine.h:4368 — hardcoded `true` literal

### 5. Race G-H
close vs tryAdmit: CoordinatorState::ShuttingDown check → enqueue (RACE WINDOW before residency increment)
atomicity: No atomic check-reserve-enqueue pattern
linearization point: None defined
verdict: NOT closed — admission gate is state-machine-based, not reservation-accounting-based

### 6. Q0 / Q1 / Q2 Relationship
Q0 independent or derived: **INDEPENDENT**
Proof: joinProducers() is a blind CAS; Q0 hardcoded `true` proves independence recognized

### 7. INV-LIFE-13
Definition: admissionReservations == 0 ⇔ all reservations released
Can current implementation satisfy it?: **NO** — no counter exists

### 8. Verdict
**BLOCKED**

### 9. Implementation authorization
**NO-GO**
