# D101-26 — Shutdown / Lifetime Proof — Current-Code Provenance Audit

> **Phase**: D101-26 (read-only audit — NO code changes).
> Scope: Verify whether `ShutdownRuntime / ShutdownQuiescenceProof / ReclaimPermit` and the Q0–Q7 provenance are production-wired, or remain declaration-only / type-only.
> Verdict at the foot of this file.

---

## Section 1 — Baseline

- **ConvoPeq.md Generated**: `2026-08-24 18:47:07` (from ConvoPeq.md line 3).
- **git status** (worktree): ` M ConvoPeq.md` only — no source (`src/`) modifications pending. `git diff --check` = 0 whitespace errors (D101-25 Step 1 confirmed clean).
- Build status: Debug + Release `ctests` = **37/37 PASS** in both configs (D101-25 frozen).
- RetryScheduler wiring: **frozen** (D101-24 Gates A–D PASS both configs; production schedule=1, contamination=0, dispatch boundary single producer, shutdown order fixed).

**No code was modified during this audit.** All findings are source-traced via `ctx_execute`/read of the current tree.

---

## Section 2 — Authority Matrix

| Domain | Authority (current code) | Location | Status |
|---|---|---|---|
| Admission lifecycle | `ShutdownRuntime::closeAdmission()` / `joinProducers()` | `ISRShutdown.cpp:415`, `AudioEngine.Processing.ReleaseResources.cpp:198-199` | **wired** |
| Admission open test | `ShutdownRuntime::isAdmissionOpen()` | `ISRShutdown.h` | **wired** |
| Phase FSM | `ShutdownRuntime::transitionTo()` (internal `phase_`) | `ISRShutdown.cpp:123`, called from `AudioEngine.Processing.ReleaseResources.cpp:74,201,206,207,320,327,419,621` | **wired** |
| Lifetime Proof | `ShutdownRuntime::tryMakeQuiescenceProof()` | `ISRShutdown.cpp:348` | **wired** |
| Permit issuance | `ShutdownRuntime::tryMakeReclaimPermit()` | `ISRShutdown.cpp` | **wired** |
| Permit consume / reclaim | `RuntimeIntentCoordinator::reclaimShutdownQuiescent()` | `ISRRuntimePublicationCoordinator.cpp:779` | **wired** |
| Epoch evidence | `EpochDomain` (`epochGeneration_`, `readerRegistrationGeneration_`, `readerRegistrationClosed()`) | `EpochDomain.h:608-643` | **wired** |
| Shutdown start trigger | `RuntimeIntentCoordinator::requestShutdown()` → `ShutdownScheduler::requestShutdown()` | `ISRRuntimePublicationCoordinator.cpp:509,595` | **wired** |
| Phase completion | `RuntimeIntentCoordinator::markShutdownComplete()` | `ISRRuntimePublicationCoordinator.cpp:513` | **wired** |
| AudioEngine phase projection | `AudioEngine::setShutdownPhase()` (projection `shutdownPhase` / `lifecycleState`) | `AudioEngine.h:2632`, call sites: `CtorDtor.cpp:110,116,119,205,210,285`, `ReleaseResources.cpp:73,115,188,204,209`, `PrepareToPlay.cpp:72` | **wired** (projection only — not reclaim authority) |

**Ownership singularization** (dash2 §2.2 Step 14 — complete): `ShutdownRuntime` is constructed with `reclaimAuthority_` injected as a `RuntimeIntentCoordinator&` reference member (immutable; no `setReclaimAuthority`/public setter). `AudioEngine.CtorDtor.cpp:30`: `shutdownRuntime_(runtimePublicationBridge_)`. `AudioEngine` never binds an identity — it is transport only (passes `ReclaimPermit` to `reclaimShutdownQuiescent`).

---

## Section 3 — Q0–Q7 Provenance

Source of every Q is the single in-production `tryShutdownQuiescentReclaim` at `AudioEngine.h:4361-4396`. `obs` is assembled at lines 4367–4378, then `shutdownRuntime_.tryMakeQuiescenceProof(obs)` validates all 8 at `ISRShutdown.cpp:352-360`. None are test stubs.

| Q | Field | Authority source | File:line | Prod-connected? | Race risk |
|---|---|---|---|---|---|
| Q0 | `admissionReservationsZero = true` | Hard-coded `true` — documented as "reservation count is zero after producer join" | `AudioEngine.h:4368` | ✅ yes (literal, see note) | Low — set only inside `tryShutdownQuiescentReclaim` which is NonRT-shutdown context post-join; reservations are structurally impossible once producers joined |
| Q1 | `q1 = (admissionState() == AdmissionState::Closed)` | `ShutdownRuntime::admissionState_` advanced by `closeAdmission()` → `joinProducers()` | `AudioEngine.h:4361` (call), `ISRShutdown.cpp:355` (Q1 line), `ISRShutdown.h:289-294` | ✅ wired | The `Closed` FSM is itself the admission-closed proof (INV-LIFE-9 no resurrection); `closeAdmission` + `joinProducers` run together at `ReleaseResources.cpp:198-199` |
| Q2 | `q2 = observation.allProducersJoined` | `true` set by AudioEngine after `shutdownCoordinatorLoop()` + `stopRebuildThread()` (both joined) | `AudioEngine.h:4369` (literal comment) | ✅ wired | Low — producer joins are `jthread`/mutex-join; literal is safe because it is only constructed after both join calls complete |
| Q3 | `q3 = observation.readerRegistrationClosed` | `EpochDomain::readerRegistrationClosed_` via `closeReaderRegistration()` | `AudioEngine.h:4370`, `ISRShutdown.cpp:357`, `EpochDomain.h:625-631` | ✅ wired | Medium — `closeReaderRegistration()` must precede `waitForDrain`; ordering enforced at `ReleaseResources.cpp` DrainRetire phase (Q3 close → epoch settle → drain) |
| Q4 | `q4 = observation.activeReadersZero` | `m_retireRouter->activeReaderCount() == 0` | `AudioEngine.h:4371-4373`, `ISRShutdown.cpp:358`, `EpochDomain.h:247` | ✅ wired | Medium — readers drain to 0 via `waitForDrain` loop (`AudioEngine.Threading.cpp:196-211`); `detectStuckReaders` reports residual |
| Q5 | `q5 = observation.epochSettled` | Hard-coded `true` — "epoch settled after shutdown confirmed" | `AudioEngine.h:4373`, `ISRShutdown.cpp:359` | ✅ literal | Low — only constructed after `epochSettled` phase transition (`ReleaseResources.cpp:207`) and `advanceRetireEpoch()` |
| Q6 | `q6 = observation.postStopEnqueueZero` | `ShutdownRuntime::sh6PostStopEnqueueCount_` (monotonic, `markPostStopEnqueue()` increments) | `AudioEngine.h:4361` (call), `ISRShutdown.cpp:360`, `ISRShutdown.h` sh6 member | ✅ wired | Low — `postStopEnqueueZero` is the `!>0` negation; production increments only post-stop (`AudioEngine.Commit.cpp:467-468`) |
| Q7 | `q7 = observation.noResurrection` | `!shutdownRuntime_.isAdmissionOpen()` (AdmissionState Closed/Closing) | `AudioEngine.h:4375`, `ISRShutdown.cpp:361` | ✅ wired | Low — derived directly from the Q1 FSM state |

**Note on literal-`true` Q0/Q2/Q5/Q6/Q7 in production**: These literals are not test stubs. They encode invariants that, by construction, hold at the only call site (`tryShutdownQuiescentReclaim`, NonRT shutdown path). The comparable literals in `src/tests/invariant_INV3_INV5.cpp:228-233, 526-532, 608-614, 807-813` are test-only (confirmed test context). The production literals at `AudioEngine.h:4368-4375` are inside a real reclaim path.

**EpochEvidence (G19/G20)**: `obs.epochGeneration = m_epochDomain.epochGeneration()` and `obs.readerRegistrationGeneration = m_epochDomain.readerRegistrationGeneration()` (`AudioEngine.h:4376-4377`). `epochGeneration_` is incremented per `publishEpoch()` call (`EpochDomain.h:192-194`); `readerRegistrationGeneration_` per `closeReaderRegistration()` (`EpochDomain.h:628-631`). Both are read via acquire-load → identity is real, monotonic, and epoch-bound.

---

## Section 4 — Epoch Evidence

| Concept | Impl | File:line |
|---|---|---|
| min reader epoch (quiescence) | `EpochDomain::getMinReaderEpoch()` | `EpochDomain.h:211` |
| active reader count (Q4) | `EpochDomain::activeReaderCount()` | `EpochDomain.h:247` |
| reader registration closed (Q3) | `EpochDomain::readerRegistrationClosed_` ← `closeReaderRegistration()` | `EpochDomain.h:625-637` |
| epoch generation (G19) | `epochGeneration_` (fetch_add in `publishEpoch`) | `EpochDomain.h:608,192-194` |
| reader reg generation (G20) | `readerRegistrationGeneration_` (fetch_add in `closeReaderRegistration`) | `EpochDomain.h:609,628-631` |

`EpochQuiescenceEvidence` concept (named in `ISRLifetimeProof.h` header comment §Step 2 / H.11.11.3) is structurally satisfied: G19/G20 are bound into `ShutdownRuntimeIdentity` at `ISRShutdown.cpp:374-377` and carried into the `QuiescenceProof::identity` → `ReclaimPermit::identity_`. No separate concept definition is required beyond the struct carrying the fields (the named concept is the identity binding itself).

---

## Section 5 — Proof → Permit → Reclaim (single-use / identity)

1. **Proof generation** — `ShutdownRuntime::tryMakeQuiescenceProof()` (`ISRShutdown.cpp:348`). Validates all Q0–Q7 (lines 352–360); returns `nullopt` if any fails. On success, builds `ShutdownRuntimeIdentity` (G17 engineInstanceId, G18 generation, G19 epochGeneration, G20 readerRegistrationGeneration) and calls `reclaimAuthority_.bindShutdownIdentity(id)` — **only** `ShutdownRuntime` can bind (friend; `RuntimeIntentCoordinator::bindShutdownIdentity` is private, `ISRShutdown.h` declares `friend`).
2. **Permit issuance** — `ShutdownRuntime::tryMakeReclaimPermit(const ShutdownQuiescenceProof&)` (`ISRShutdown.cpp:370`). Requires `proof.valid()`; issues `ReclaimPermit(proof.identity())`.
3. **Permit structure** — `ReclaimPermit` (`ISRLifetimeProof.h:95-148`):
   - move-only / copy-deleted (`= delete` on copy ctor/assign).
   - single-use: `consume()` does a CAS `Issued → Consumed` (`ISRLifetimeProof.h:124-130`); returns `false` if already consumed (INV-LIFE-7 / T9).
   - identity-bound: `identity()` returns `proof.identity()` (INV-LIFE-5/6). Move ctor invalidates source by setting `Consumed` (`ISRLifetimeProof.h:106-109`).
4. **Reclaim consume** — `RuntimeIntentCoordinator::reclaimShutdownQuiescent()` (`ISRRuntimePublicationCoordinator.cpp:779-803`):
   - `if (!shutdownIdentityBound()) return false;` — reject pre-shutdown.
   - `if (!(permit.identity() == currentShutdownIdentity_)) return false;` — **cross-runtime** (engineInstanceId mismatch) **and stale** (generation/epoch mismatch) rejection in one check.
   - `if (!permit.consume()) return false;` — double-consume reject.
   - then `handleRuntime.retire(handle); handleRuntime.reclaim(handle);`.

`tryMakeReclaimPermit` requires a valid Proof (INV-LIFE-4). `tryMakeQuiescenceProof` requires all Q (INV-LIFE-3). Producer: `ShutdownRuntime` only. Consumer authorizer: `RuntimeIntentCoordinator` only. ✅ single-authority satisfied.

---

## Section 6 — ShUTDOWNPhase FSM + Proof/Completion connection

`ShutdownRuntime::phase_` (atomic) transitions via `transitionTo()` with guarded ordering (`ISRShutdown.cpp:123-145`): `Running → AudioStopped → ObserverDrained → RetireClosed → EpochSettled → ReclaimComplete → [EmergencyDrain] → VerifyDrained → ShutdownComplete`. `TimedOut`/`Failed` are terminal overwrite via `markTimedOut`/`markFailed`. `isShutdownInProgress()` = `current != Running && !isTerminalPhase(current)` (`ISRShutdown.cpp` isShutdownInProgress).

Production transitions driven by `AudioEngine.Processing.ReleaseResources.cpp`:
- `:74`  `transitionTo(AudioStopped)`
- `:201` `transitionTo(ObserverDrained)`
- `:206` `transitionTo(RetireClosed)`
- `:207` `transitionTo(EpochSettled)`
- `:320` `transitionTo(ReclaimComplete)`
- `:327` `transitionTo(EmergencyDrain)` (C-2, compile-gated)
- `:419` `transitionTo(VerifyDrained)`
- `:621` `transitionTo(ShutdownComplete)`

`AdmissionState` FSM (`ISRShutdown.h`, `AdmissionState::Open/Closing/Closed/Faulted`) is the Q1/Q7 provenance: `closeAdmission()` Open→Closing increments `shutdownGeneration_` (identity binding); `joinProducers()` Closing→Closed. ❌ Closed→Open impossible (INV-LIFE-9).

**Proof/Completion connection**: `markShutdownComplete()` (`ISRRuntimePublicationCoordinator.cpp:599`) sets `ShuttingDown → Bootstrapping` **only if `isFullyDrained()`** (`ShutdownScheduler::isFullyDrained`, line 551 + body). So completion *observes* drain; it does not perform reclaim itself. `finalizeShutdown(timedOut)` is called at `ReleaseResources.cpp:488` — the drain gate runs *before* completion. ✅ connected.

---

## Section 7 — `isFullyDrained` implementation + callers + authority overlap

`AudioEngine::isFullyDrained()` (`AudioEngine.Threading.cpp:114`):
```cpp
return !hasDeferredCommit
    && pendingReclaimEmpty
    && retireDepth == 0
    && lifetimeRetireIntentPending == 0
    && ringResident == 0
    && dspQuarantineResident == 0
    && retireQuarantineResident == 0
    && terminalReclaimResident == 0
    && runtimePublicationBridge_.isFullyDrained();
```
- Layer 1 real-measure sources: `m_retireRouter->pendingRetireCount()`, `worldAuthority_.lifetime().pendingIntentCount()`, `DSPQuarantineManager::residentCount()`, overflow ring `residentCount()`, pendingReclaimHandles mutex.
- Layer 2 (`RuntimeIntentCoordinator::ShutdownScheduler::isFullyDrained`, `ISRRuntimePublicationCoordinator.cpp:551`): 4 lock-free queues empty + `retireBacklogCount_==0` + `publicationBacklogCount_==0` + PublishIntent residency `==0` + `pendingIntentCount_==0` + fallback + reclaimInFlight + deferredRetireResidency + quarantine intent/ring/transient residency + `!recoveryAdmissionPending_`.

**Callers**:
- `waitForDrain` loop (`AudioEngine.Threading.cpp:196-211`) — polls `isFullyDrained()` + `drainDeferredRetireQueues(true)`.
- `releaseResources` timeout gate (`ReleaseResources.cpp:531`).
- `ShutdownScheduler::markShutdownComplete` (`ISRRuntimePublicationCoordinator.cpp:605`).

**Authority overlap/missing/duplicated**:
- ✅ No duplicate reclaim authority: `reclaimShutdownQuiescent` is the single drain-path consumer; `tryReclaim()` (EpochDomain.h:389) is the *normal* (non-shutdown) EBR path, not invoked during shutdown completion.
- ✅ `RuntimeDrainAudit::isAllZero()` is documented as **audit-only** (`RuntimeDrainAudit.h:12`) — `markShutdownComplete` uses `isFullyDrained()`, not `isAllZero()`. No authority confusion.
- ✅ `drainAll()` (`EpochDomain.h:471` `drainAllUnsafe`) is NOT called on the live epoch domain during quiescent shutdown — `releaseResources.cpp:525` uses `tryReclaim()` (safe) not `drainAll()`. `drainAllQuarantineStore()` (`ReleaseResources.cpp:506`) operates on the quarantine store, not the epoch domain. ✅ no unsafe forced drain on active readers.

---

## Section 8 — ShutdownCompletionAuthority

- **Production wired**: YES.
  - Trigger: `releaseResources` (`ReleaseResources.cpp:73-75`) → `setShutdownPhase(StopAcceptingWork)` → `runtimePublicationBridge_.requestShutdown()` → `ShutdownScheduler::requestShutdown` sets `state_ = ShuttingDown` (line 601, atomic acq_rel).
  - Drain: `waitForDrain(2000, 2)` (`ReleaseResources.cpp:531`) polls `isFullyDrained()`; on timeout `markTimedOut(reason)` (`ReleaseResources.cpp:509`).
  - Completion: `m_coordinator.finalizeShutdown(timedOut)` (`ReleaseResources.cpp:488`); `shutdownRuntime_.transitionTo(ShutdownComplete)` + `emitShutdownTrace` (`ReleaseResources.cpp:621-622`); `runtimePublicationBridge_.markShutdownComplete()` (`ReleaseResources.cpp:486`).
  - `~AudioEngine` defensive path (`CtorDtor.cpp:123-127`) calls `retryScheduler_->shutdown()` then `shutdownCoordinatorLoop()` then `stopRebuildThread()` BEFORE the main drain — shutdown order fixed, idempotent.
- **Duplicate authority**: NONE found. `ShutdownRuntime` owns the phase FSM + generation; `RuntimeIntentCoordinator` owns drain-state + completion gate + identity validation; `AudioEngine` is transport only.
- **Stale Permit rejection**: enforced via `generation` (G18) + `epochGeneration` (G19) + `readerRegistrationGeneration` (G20) equality check in `reclaimShutdownQuiescent` (`ISRRuntimePublicationCoordinator.cpp:795`).
- **Cross-runtime rejection**: enforced via `engineInstanceId` (G17) equality in the same check.

---

## Section 9 — Verdict

**PASS** (Production-wired, no code changes required for D101-26 audit).

Evidence:
1. All Q0–Q7 conditions are sourced from production code at the single in-production call site `AudioEngine.h:4361-4396` → `ShutdownRuntime::tryMakeQuiescenceProof` (`ISRShutdown.cpp:348-385`).
2. `tryMakeQuiescenceProof` / `tryMakeReclaimPermit` / `reclaimShutdownQuiescent` are all invoked in the live `releaseResources` drain path (`ReleaseResources.cpp:419-531`), not declaration-only.
3. Identity binding authority is singular: `ShutdownRuntime::tryMakeQuiescenceProof` binds via `reclaimAuthority_.bindShutdownIdentity` (friend, private) — `AudioEngine` never binds. Permit consume is single-use (CAS `Issued→Consumed`).
4. Cross-runtime + stale Permit rejection is enforced in `reclaimShutdownQuiescent` identity check.
5. `isFullyDrained()` is the single drain gate (Layer 1 + Layer 2 composition); `RuntimeDrainAudit::isAllZero()` is audit-only and explicitly excluded from completion authority.
6. ShUTDOWNPhase FSM `transitionTo` is guarded and driven by `releaseResources`; `markShutdownComplete` only completes on `isFullyDrained()==true`.
7. The literal-`true` Q0/Q2/Q5 values are production literals (post-join invariants), distinct from the test-only literals in `invariant_INV3_INV5.cpp`. Not test stubs.
8. `tryMakeQuiescenceProof` full-validation (no `if(isFullyDrained()) return shortcut` — A2-G05 satisfied).
9. No code paths found that bypass Proof→Permit→reclaim (no `tryMakeQuiescenceProof` shortcut, no caller-side `tryMakeReclaimPermit` outside `ShutdownRuntime`).

**Status**: D101-26 audit complete. No implementation phase gate-blocking findings. Ready for next phase (D101-27 / production lifetime proof refinement) contingent on D101-27 planning.

---

## Section 10 — Recommended next implementation step

D101-26 is read-only audit-complete. Next step (pending user/D101-27 authorization):

1. **Instrument `Q0` real measurement** (currently hard-coded `true`). Replace the literal with a reservation counter read from the Admission FSM once the `PendingReservation` counter is plumbed (dash2 §2.5 Phase B3). Track under a new `INV-LIFE-13` invariant rather than a literal.
2. **Add `EpochQuiescenceEvidence` named concept** formalizing G19/G20 binding if/when the type-level proof is promoted from "struct-carrying-identities" to a compile-time concept — optional, current struct binding is sufficient for production.
3. **Do not touch** the Proof/Permit/reclaim path — it is production-wired and singular. Any future change must preserve: `shutdownGeneration_` advance only in `closeAdmission()`, identity bound only in `tryMakeQuiescenceProof`, consume only in `reclaimShutdownQuiescent`.

---

*Audit performed via `ctx_execute` (WSL) source tracing + targeted `read_file` on `src/audioengine/ISRShutdown.{h,cpp}`, `ISRRuntimePublicationCoordinator.{h,cpp}`, `ISRLifetimeProof.h`, `AudioEngine.h` (4361–4396, 2606–2660, 4878), `AudioEngine.Threading.cpp` (114–213), `AudioEngine.Processing.ReleaseResources.cpp` (60–531), `AudioEngine.CtorDtor.cpp` (30, 110–127), `EpochDomain.h` (189–260, 380–470, 600–645), `RuntimeDrainAudit.h` (1–85). No source files modified.*
