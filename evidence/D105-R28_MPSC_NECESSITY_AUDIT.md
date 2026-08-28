# D105-R28 — Recovery Intent Queue MPSC Necessity & Producer-Boundary Audit

**Status:** **R28 verdict: Case A (No 2nd Producer → MPSC化 DEFER)**
**Production source changes:** 0
**I4 changes:** 0
**Test changes:** 0
**`recoveryIntentQueue_` は SPSC 維持; `pendingRecoveryAdmission_` は単一 NonRT Producer 専用維持**

R28 establishes that the `recoveryIntentQueue_` has **only one production caller
in current code**, and no concrete Phase-II/III requirement exists to add a second
producer. MPSC化 is therefore DEFERRED. R20 + R23 + R24 + R25 + R26 + R27 = stable
final state is preserved.

---

## R28-1 — Production caller complete enumeration

### Producer side: `submitRecoveryRequest` / `submitRecoveryIntent` callers

| # | Call site | File:line | Thread/context | Status |
|---|---|---|---|---|
| 1 | `QuarantineIntentHandler::handle` calls `ctx.engine.submitRecoveryIntent(...)` | `ISRRuntimePublicationCoordinator_ProcessIntent.cpp:140` | **CoordinatorLoop (NonRT)** | **Active (primary path)** |
| 2 | `QuarantineIntentHandler::handle` (overload, line 169) | `ISRRuntimePublicationCoordinator_ProcessIntent.cpp:169` | **CoordinatorLoop (NonRT)** | **Active (overload variant)** |
| 3 | `RecoveryIntentHandler::handle` (dead code) | `ISRRuntimePublicationCoordinator_ProcessIntent.cpp:164-170` | CoordinatorLoop (NonRT) | **DEAD CODE** (nobody pushes Recovery Intent to `intentQueue_`; ProcessIntent.cpp:131 note + REPAIR_PLAN2 §6.2 line 64) |
| 4 | `submitRecoveryIntent` is the wrapper | `AudioEngine.h:4415-4455` | Called by the two `QuarantineIntentHandler::handle` paths above (call sites 1 and 2) | Active (transitively) |
| 5 | `submitRecoveryRequest` is the bridge | `ISRRuntimePublicationCoordinator.cpp:819` | Called by `submitRecoveryIntent` via `runtimePublicationBridge_.submitRecoveryRequest(...)` (AudioEngine.h:4436) | Active |
| 6 | `pushRecoveryIntent` to queue | `ISRRuntimePublicationCoordinator.cpp:894` | **NonRT, CoordinatorLoop thread only** | Active (SPSC) |
| 7 | Durable fallback (queue full) | `ISRRuntimePublicationCoordinator.cpp:929-940` | **NonRT, CoordinatorLoop thread only** | Active (single slot, plain struct) |

**Single Producer conclusion**: The entire Recovery admission pipeline has **exactly
one NonRT producer (CoordinatorLoop)**. There is no second producer in production
code. The `RecoveryIntentHandler` is registered in the dispatch table (`kDispatchTable`)
but is **dead code** (verified in REPAIR_PLAN2-dash2 §6.2 line 64 and D101-12 §4
"D101-12 §4 D101-12" file 7.5).

### Consumer side: `popRecoveryRequest` / `takePendingRecoveryAdmission` callers

| # | Call site | File:line | Thread/context |
|---|---|---|---|
| 1 | `rebuildThreadLoop` calls `popRecoveryRequest()` | `AudioEngine.RebuildDispatch.cpp:939, 1005` | **RebuildThread (NonRT)** |
| 2 | `rebuildThreadLoop` calls `takePendingRecoveryAdmission()` | `AudioEngine.RebuildDispatch.cpp:1017, 1081` | **RebuildThread (NonRT)** |
| 3 | `discardRecoveryRequestsOnShutdown` | `ISRRuntimePublicationCoordinator.cpp:1181-1205` | **NonRT (shutdown path)** |

**Single Consumer conclusion**: Only RebuildThread consumes from `recoveryIntentQueue_`
and `pendingRecoveryAdmission_`. No RT consumer exists.

### Audio Thread caller audit

Per R28 brief question 4, **AC-ISR-1** requires the Audio Thread to **not** be an
MPSC producer. Verified:

| Path | Audio Thread involvement? |
|---|---|
| `submitRecoveryRequest` | **NO** (all calls from NonRT CoordinatorLoop) |
| `popRecoveryRequest` | **NO** (called from NonRT RebuildThread) |
| `takePendingRecoveryAdmission` | **NO** (called from NonRT RebuildThread) |
| `RecoveryIntentHandler::handle` (dead code) | The handler dispatches to `submitRecoveryIntent` on CoordinatorLoop, not Audio Thread |

**AC-ISR-1 verdict: VERIFIED** — Audio Thread is **not** an MPSC producer or consumer
in the current code.

### R28-1 conclusion

The Recovery pipeline is **SPSC (CoordinatorLoop producer / RebuildThread consumer)**
with **no second producer** in production. R28 has verified this via grep +
production code trace.

---

## R28-2 — MPSC必要性 3-way verdict

### Choice A: 2nd producer already exists in production code

**Status**: **NO** — Verified by R28-1 grep. There is no second producer.

### Choice B: 2nd NonRT producer's design spec exists

**Status**: **NO** — `doc/work88/REPAIR_PLAN2*.md` mentions MPSC化 as a future task
item:

> 4. **2-producer テスト**（plan §1.1）
> MPSC化が必要になったときに MpscBoundedRing 置換 + pendingRecoveryAdmission_ の mutex 保護

This is a **future possibility**, not a current design. The plan explicitly
describes it as a **contingency**, not a commitment.

### Choice C: 2nd producer absent AND no Phase-II/III spec

**Status**: **YES** — This is the current state. R27 (Episode) and R28 (MPSC) both
evaluate to "no concrete Phase-II requirement".

### R28 verdict: **Choice C (MPSC化 DEFER)**

MPSC implementation would require:
1. Replace `LockFreeRingBuffer<RecoveryIntent, 256>` with `MpscBoundedRing<RecoveryIntent, 4096>` (REPAIR_PLAN2 default)
2. Add `std::mutex` or atomic protection to `pendingRecoveryAdmission_` (currently plain struct, SPSC-safe)
3. Audit and re-prove all d102/d103 capacity, race, ownership invariants
4. New tests for 2-producer scenarios (REPAIR_PLAN2 §1.1 "MpscBoundedRing 置換 + pendingRecoveryAdmission_ の mutex 保護 — plan §1.1.1")

This work **does not produce a measurable benefit** in current code (no second
producer). The R28 verdict is therefore **DEFER**, matching the R27 Case A
verdict for the Episode layer.

### Trigger conditions for re-evaluation

| Trigger | MPSC化 response |
|---|---|
| 2nd NonRT producer (Timer, separate request path, etc.) is added to production code | **Re-evaluate immediately**; MPSC化 becomes required |
| Future feature requires non-CoordinatorLoop admission | **Re-evaluate**; design spec must precede implementation |
| No concrete trigger fires | **DEFER 維持** |

---

## R28-3 — MPSC化時のdurable path競合 (state-transition proof)

### Hypothetical scenario: 2 NonRT producers A, B

```
Producer A (CoordinatorLoop) ─┐
                                ├→ submitRecoveryRequest (concurrent)
Producer B (Timer thread) ─────┘
        │
        ├→ recoveryIntentQueue_ (MPSC ring)
        │
        └→ pendingRecoveryAdmission_ (plain struct, currently single-slot)
```

### Race trace 1: simultaneous transport push to full queue

```
T1: A.submitRecoveryRequest → pushRecoveryIntentQueue_ → success (slot 255)
T2: B.submitRecoveryRequest → pushRecoveryIntentQueue_ → queue full
    → rollback pendingIntentCount_ (1 → 0)
    → durable fallback → check pendingRecoveryAdmission_
```

**State at T2 rollback**:
- `pendingIntentCount_ == 0` (rolled back)
- `pendingRecoveryAdmission_` may already hold A's data (if A's earlier
  admission was in durable form)

**Conflict scenario**:
1. A's earlier admission was durable (queue was full at the time)
2. B's queue push fails
3. B's durable fallback **overwrites** A's durable data
4. A's `recoveryObligationId` is **lost**
5. When A's `takePendingRecoveryAdmission` is later called, it returns B's
   `recoveryObligationId`
6. A's `resolveRecoveryObligation` is a no-op (wrong id)
7. A's obligation becomes a **leak** (Live, never resolved)

**Conflict scenario verdict**: 2-producer race on `pendingRecoveryAdmission_`
**loses obligation identity** in the current code structure.

### Race trace 2: simultaneous takePendingRecoveryAdmission

```
T1: A's obligation durable → pendingRecoveryAdmission_ = {A's data, A's obligationId}
T2: A.takePendingRecoveryAdmission() reads pendingRecoveryAdmission_
T3: B.submitRecoveryRequest → durable fallback → overwrites pendingRecoveryAdmission_ = {B's data, B's obligationId}
T4: A.takePendingRecoveryAdmission() returns B's data (not A's!)
T5: A's Builder loop processes B's data with A's obligationId resolution expectation
T6: A.resolveRecoveryObligation(A's obligationId) → no-op (table.resolve requires matching id)
```

**Leak scenario**: A's data is overwritten; A's `recoveryAdmissionPending_` is reset
to `false` by B's overwrite; A's obligation is now live but no Builder will pick
it up (since A's obligation is no longer durable, and A's transport entry was
already consumed).

### Race trace 3: current SPSC guarantee

| Path | Current state |
|---|---|
| Single NonRT producer (CoordinatorLoop) | NO race |
| `pendingRecoveryAdmission_` access | Single-threaded (CoordinatorLoop only) |
| `pendingIntentCount_` | `fetchAdd` on Producer side, `fetchSub` on Consumer side |

The current SPSC architecture avoids all 3 races above **by construction** (one
producer thread, one consumer thread). MPSC化 would require:

1. **Mutex on `pendingRecoveryAdmission_`** (REPAIR_PLAN2 §1.1.1 explicitly states this)
2. **Atomic compare-and-swap** in the durable fallback (to detect if another producer
   has overwritten since the queue-full check)
3. **Retry loop** on durable allocation failure

### R28-3 verdict

MPSC化 is **non-trivial**: it requires `pendingRecoveryAdmission_` protection that
is **not present** in current code. The current plain-struct single-slot is
**SPSC-safe by construction**. MPSC化 would require new mutex/atomic infrastructure
and a new design for the durable allocation retry loop.

---

## R28-4 — MPSC化しても維持すべき不変条件 (R28 brief §R28-4)

### INV-R1-1 (R28 brief, R28-4)

```
pendingIntentCount_ = (成功したreservation総数) - (成功したpop/discard総数)
```

**Current implementation**:
- Reservation: `convo::fetchAddAtomic(pendingIntentCount_, 1, ...)` at
  `submitRecoveryRequest` line 908 (transport push success) and 915 (durable
  fallback success)
- Consumption: `convo::fetchSubAtomic(pendingIntentCount_, 1, ...)` at
  `popRecoveryRequest` line 1127
- Discard: `convo::fetchSubAtomic(pendingIntentCount_, 1, ...)` at
  `discardRecoveryRequestsOnShutdown` line 1197 (within `while (popRecoveryRequest())`)

**INV-R1-1 status**: VERIFIED for current SPSC. The counter is updated atomically
at exactly 3 sites (1×push, 1×pop, 1×discard). Each atomic operation is `fetchAdd`
or `fetchSub` on a single `std::atomic<uint64_t>` (`pendingIntentCount_`).

**MPSC impact**: If a 2nd producer calls `submitRecoveryRequest` concurrently, the
`fetchAdd` at line 908 and 915 is still atomic. The counter is still consistent
**at the integer level**. The race is at the **durable slot** level, not the
counter level.

### INV-R1-2 (R28 brief, R28-4)

```
pendingIntentCount_ == 0  ⇒  queue visible residency == 0 AND producer reservation == 0
```

**Current implementation** (`isFullyDrained` predicate):
```cpp
pendingIntentCount_ == 0
    (-- required by the drain predicate)
&&
recoveryAdmissionPending_ == false
    (-- PendingRecoveryAdmission is in NoAdmission state)
&&
recoveryIntentQueue_.size() == 0
    (-- transport queue is empty)
```

**Three independent drain conditions**, all required. If any is non-zero, `isFullyDrained`
returns false.

**MPSC impact**: All three conditions remain well-defined under MPSC, but the
**durable slot** (`pendingRecoveryAdmission_`) requires a mutex to be race-free.
Without the mutex, two producers could both pass the "is NoAdmission" check,
both write, and the second overwrites the first. The counter `pendingIntentCount_`
is still correct, but the durable slot is corrupted (see R28-3).

### AC-ISR-1 (R28 brief, R28-4)

```
Audio Thread は MPSC enqueue producer にならない
```

**Current implementation**:
- `submitRecoveryRequest` is called from `submitRecoveryIntent` (line 4436 in
  AudioEngine.h), which is called from `QuarantineIntentHandler::handle`
  (`ProcessIntent.cpp:140, 169`).
- `QuarantineIntentHandler::handle` runs on the **CoordinatorLoop (NonRT)**.
- The Audio Thread (ISR) does not call `submitRecoveryIntent` or
  `submitRecoveryRequest`.

**AC-ISR-1 status**: VERIFIED. The current architecture explicitly keeps Recovery
admission on NonRT. MPSC化 design must preserve this constraint (per REPAIR_PLAN2).

### R28-4 verdict

The 3 invariants (INV-R1-1, INV-R1-2, AC-ISR-1) are **preserved in current code** and
**MPSC-compatible** at the **counter level**. However, the **durable slot**
(`pendingRecoveryAdmission_`) is **NOT** MPSC-compatible without explicit mutex
protection. This is the **single physical design change** required for MPSC化.

---

## R28 final verdict: **Case A — MPSC化 DEFER**

### Decision

The current SPSC architecture is **structurally correct**:
- 1 NonRT producer (CoordinatorLoop)
- 1 NonRT consumer (RebuildThread)
- 0 Audio Thread involvement
- 3 atomic drain conditions (pendingIntentCount_, queue, durable flag)
- AC-ISR-1 preserved

**No 2nd producer exists**. MPSC化 is **not required**.

### Implementation trigger

| Trigger | MPSC化 response |
|---|---|
| 2nd NonRT producer added to production code (Timer, separate request path, future migration) | **Re-evaluate immediately** |
| `pendingRecoveryAdmission_` requires concurrent writes | **Re-evaluate**; mutex or atomic design needed |
| No concrete trigger fires | **MPSC化 DEFER 維持** |

### Phase-III+ reservation

If a future feature requires MPSC:
1. **Required change**: Add `std::mutex pendingRecoveryAdmissionMutex_` to
   `RuntimeIntentCoordinator` (REPAIR_PLAN2 §1.1.1 explicit requirement)
2. **Required change**: Change durable fallback to CAS retry loop
3. **Required change**: Update `isFullyDrained` to be mutex-aware
4. **Required test**: `MPSC 2-producer stress test` (REPAIR_PLAN2 §1.1 "MPSC化が必要になったときに")
5. **Required test**: `durable slot concurrent write rejection test`
6. **Required invariant re-proof**: `INV-R1-1`, `INV-R1-2`, `AC-ISR-1` must hold under MPSC

### R28 closure

R20 + R23 + R24 + R25 + R26 + R27 + **R28 (MPSC DEFER)** = stable final state.
The D105 Episode branch (R25-R27) and the MPSC branch (R28) are both **DEFERRED**
pending concrete Phase-II/III requirements. No Runtime source change, no I4 change,
no test change is required.

### File references (read-only, no source modification)

| File | Role |
|---|---|
| `ConvoPeq.md` (2026-08-27) | runtime source baseline |
| `src/audioengine/ISRRuntimePublicationCoordinator_ProcessIntent.cpp:140, 169` | `submitRecoveryIntent` callers (production) |
| `src/audioengine/ISRRuntimePublicationCoordinator_ProcessIntent.cpp:164-170` | `RecoveryIntentHandler` (dead code) |
| `src/audioengine/AudioEngine.h:4415-4455` | `submitRecoveryIntent` wrapper |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp:819-942` | `submitRecoveryRequest` (single NonRT producer) |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp:894, 929-940` | queue push + durable fallback |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp:1127, 1181-1205` | `popRecoveryRequest` + `discardRecoveryRequestsOnShutdown` |
| `src/audioengine/ISRRuntimePublicationCoordinator.h:422-426` | `submitRecoveryIntent` declaration |
| `src/audioengine/ISRRuntimePublicationCoordinator.h:496, 510, 520` | `popRecoveryRequest` / `takePendingRecoveryAdmission` / `settlePendingRecoveryAdmission` |
| `src/audioengine/ISRRuntimePublicationCoordinator.h:880, 904, 905` | SPSC contract comment + `recoveryIntentQueue_` field |
| `doc/work88/REPAIR_PLAN2-dash2.md:64-65, 877-880, 2175, 2179` | primary path, MPSC化future-item, Producer caller verification |
| `doc/work88/REPAIR_PLAN2-dash2.md:1153-1184` | `submitRecoveryIntent` wake optimization, recoveryPending flag |
| `doc/work88/REPAIR_PLAN2-dash2.md:2205-2217` | X1 takePendingRecoveryAdmission, lost-wakeup safety |
| `evidence/D101-12-Phase-D-2-Retry-Policy-Minimal-Contract-Audit.md:435` | SPSC source verification |
| `evidence/D105-R20_REJECTEDNOTFINALIZED_CENTRALIZATION.md` | R20 finalization (Recovery centralized) |
| `evidence/D105-R25_PHASE_II_EPISODE_MIGRATION_AUDIT.md` | R25 Option B recommendation (deferred) |
| `evidence/D105-R27_PHASE_II_OBJECTIVE_NECESSITY.md` | R27 Case A verdict (Episode DEFERRED) |

**No source files modified. No I4 files modified. No tests added.** R28 is a
read-only audit that establishes Case A (MPSC化 DEFER permanently) for the
Recovery intent queue.

---

## R28 final summary

| Aspect | R28 verdict |
|---|---|
| Producer enumeration (R28-1) | 1 production caller (`QuarantineIntentHandler::handle`); `RecoveryIntentHandler` is dead code |
| MPSC必要性 3-way verdict (R28-2) | **Choice C**: no 2nd producer, no Phase-II/III spec |
| Durable path競合 (R28-3) | MPSC化は `pendingRecoveryAdmission_` の mutex を必要とする (REPAIR_PLAN2 §1.1.1 通り) |
| INV-R1-1 / INV-R1-2 / AC-ISR-1 (R28-4) | counter level: MPSC-compatible; durable slot: mutex required |
| Final verdict | **Case A — MPSC化 DEFER 維持** |

R28 confirms that the D105 MPSC branch is **closed at "deferred" state**, matching
R27's Episode branch closure. The D105 audit chain (R15-R20-R23-R24-R25-R26-R27-R28)
is now **structurally complete at the final stable state** for both Episode layer
and Recovery MPSC queue. No further Runtime implementation work is authorized
without explicit Phase-II/III requirements.
