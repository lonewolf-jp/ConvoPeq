# D105-R12 — Shutdown Drain Predicate / Logical Obligation Zero Structural Re-proof

**Status:** CONDITIONAL PASS. **Source changes: 0.**
**Base document:** `ConvoPepeg.md` baseline 2026-08-27 23:32:40. Preceded by D105-R11 (PASS).
**Goal:** Prove that **`liveLogicalRecoveryObligationCount() == 0`** is *logically* guaranteed
post-discard, and determine whether `isFullyDrained()` *structurally* asserts it.

---

## R12-1 — `isFullyDrained()` predicate audit (complete)

`ShutdownScheduler::isFullyDrained()` (`ISRRuntimePublicationCoordinator.cpp:506-550`) ANDs:
```cpp
swapPending_ == false                                    // (ctrl)
intentQueue_.sizeApprox() == 0                         // publish/observe transport
observeDeferredRing_.size() == 0                       // observe overflow
quarantineFallbackQueue_.sizeApprox() == 0             // quarantine fallback
recoveryIntentQueue_.size() == 0                       // recovery transport (A)
retireBacklogCount_ == 0                               // retire (C)
publicationBacklogCount_ == 0                         // publish (C)
publicationIntentResidencyCount_ == 0                  // publish intent (C)
pendingIntentCount_ == 0                              // publish+recovery reservations (A/C)
reclaimInFlightCount_ == 0                             // reclaim (C)
quarantineIntentResidencyCount_ == 0    // quarantine transport (A)
quarantineRingResidencyCount_ == 0      // quarantine fallback ring (A)
!recoveryAdmissionPersistent_                        // recovery durable (A)
```

**`liveLogicalRecoveryObligationCount() == 0` is *NOT* among the listed checks.**

Classification (R12-3): `isFullyDrained()` = **A ∧ C** (transport/durable + retire/epoch/publish),
**not** A ∧ B ∧ C. Segment **B (logical obligation residency)** is not structurally asserted by the
predicate. Layer 1 `AudioEngine::isFullyDrained()` (Threading.cpp:160-174) only adds a further AND with
`runtimePublicationBridge_.isFullyDrained()` (Layer 2) — same gap.

## R12-2 — `liveCount_` zero-reachability: independent structural proof

The recovery table exposes **exactly one** `+1` and **exactly one** `−1`:

| Mutation | Site | File:Line |
|---|---|---|
| `+1` (`liveCount_++`) | `tryInsert` | `ISRRuntimePublicationCoordinator.h:343-355`, called only at `cpp:873` |
| `−1` (`liveCount_--`) | `RecoveryAdmissionTable::resolve` (CAS Live→terminal) | `h:366-378`; called via `resolveRecoveryObligation` `cpp:954` and `discardRecoveryRequestsOnShutdown` `cpp:1149` |

No other code writes `recoveryAdmissions_.liveCount_` / mutates obligation `state` Live↔terminal
(verified by grep of `recoveryAdmissions_.` across `src/`): only `findByKey` (read), `slot(i)` reads,
delivery-field writes (`delivery=`, not state/liveCount), `tryInsert` (+1), `resolve` (−1), and the
discard scan (`slot(i).id.load()`; `cpp:1147`). `nextId_` is monotonically increasing (`++nextId_`, h:389),
so a reused slot is assigned a **strictly greater** id — `resolve` re-scans by id and can never
terminate the wrong obligation (ABA-safe by construction; C9 test cpp:1056 proves this).

**State-transition chain from `requestShutdown`:**
1. `requestShutdown()` publishes `state_=ShuttingDown` (`cpp:552-554`).
2. `submitRecoveryRequest()` returns `false` at the `ShuttingDown` gate **before** any `tryInsert`
   (`cpp:817-821` → `tryInsert` only reachable at `cpp:873`, which is *after* the gate). ⇒ **no new `+1`.**
3. `redriveDeferredRecoveryObligations()` / `redriveDeferredRecovery()` re-attach delivery to an
   **existing** id (`intent.obligationId = obligationId;`, `cpp:1019`); they **never** call `tryInsert`
   and **never** issue a new id. ⇒ redrive is `ΔL=0` and cannot resurrect/create. (R5-10 design, unchanged.)
4. `discardRecoveryRequestsOnShutdown()` (`cpp:1136-1151`) loops **all** `kCapacity` slots; for every
   non-zero id it calls `resolveRecoveryObligation(oblId, ShutdownDiscarded)` → `resolve()` CAS
   `Live→ShutdownDiscarded` + `liveCount_--`. Because the `−1` is idempotent (lost CAS = no-op, no underflow),
   every Live slot — regardless of its `delivery` state (None/Transport/Durable) — is terminalized
   exactly once. ⇒ **`liveCount_` reaches 0.**

`liveCount_ == 0` is therefore *logically compelled*: `+1` is gated-off post-`requestShutdown`, and the
single `−1` authority is driven over every Live slot by the discard loop.

## R12-3 — `isFullyDrained()` semantics: 3-layer separation

- **A (transport/durable drain)** ∈ predicate: `recoveryIntentQueue_.size()==0`, `!recoveryAdmissionPersistent_`,
  `quarantine*/intentQueue/observeDeferredRing...` empty. ✓ asserted.
- **C (retire/epoch/publish/reclaim drain)** ∈ predicate: `retireBacklogCount_==0`,
  `publicationBacklogCount_==0`, `publicationIntentResidencyCount_==0`, `pendingIntentCount_==0`,
  `reclaimInFlightCount_==0`, `quarantineIntentResidency==0`, `quarantineRingResidency==0`. ✓ asserted.
- **B (logical obligation residency)** ∉ predicate: `liveLogicalRecoveryObligationCount()` is **not** queried.
  B is satisfied **only by the shutdown ordering** (discard terminalizes all Live slots, see R12-2 step 4),
  not by the predicate itself.

`AudioEngine::isFullyDrained()` (Threading.cpp:160-174) = Layer-1 counters ∧ `runtimePublicationBridge_.isFullyDrained()` (Layer 2). Neither adds B.

## R12-4 — Shutdown ordering: no Live generation at any stage

SHUTDOWN-ORDER (ReleaseResources.cpp:507-512):
```
requestShutdown(:75)  ──► state_=ShuttingDown         (AdmissionClosed; submitRecoveryRequest → false)
  ↓ shutdownCoordinatorLoop Join(:189)                (only thread running runCoordinatorPhase → redrive; STOPS)
  ↓ stopRebuildThread Join(:190)                        (Builder thread stopped)
  ↓ discardRecoveryRequestsOnShutdown(:802)             (all Live → ShutdownDiscarded, −1; no +1)
      └─ pop all recoveryIntentQueue_ intents (pendingIntentCount_ fetchSub, cpp:1124)
      └─ resolve every non-zero id slot (cpp:1146-1150)
  ↓ discardPendingRecoveryAdmission(:810)               (durable slot cleared, recoveryAdmissionPersistent_=false, cpp:1110; no obligation generated)
  ↓ waitForDrain(:430)/isFullyDrained(:514,196)         (read-only predicate; no mutation)
  ↓ m_coordinator.finalizeShutdown(:563)                (SnapshotCoordinator — epoch/quiescence layer; m_coordinator
                                                          is `convo::SnapshotCoordinator` (AudioEngine.h: m_coordinator;).
                                                          Does NOT touch recoveryAdmissions_ → no +1.)
  ↓ markShutdownComplete(:623)                          (calls isFullyDrained; → Live or Faulted)
  ↓ transitionTo(ShutdownComplete)
```

Stage-by-stage Live-generation check (R12-4 targets):

| Stage / Component | Can generate a recovery Live obligation? | Why |
|---|---|---|
| `requestShutdown` | No | only sets `ShuttingDown`; closes admission |
| `shutdownCoordinatorLoop` thread (runCoordinatorPhase) | No (after join) / No even if running | redrive re-attaches existing ids only (no `tryInsert`); `submitRecoveryRequest` gated at cpp:817 |
| `RebuildThread` (Builder) | No (after join); no even if running | Builder only *consumes* (`popRecoveryRequest`, `takePendingRecoveryAdmission`) + resolves; never `tryInsert` |
| `redriveDeferredRecoveryObligations` | No | `ΔL=0`; id-passthrough only (cpp:991-1019) |
| `publish completion` (`onPublishCommitted` → `resolveRecoveryObligation(Published)`, cpp:346/312) | No | `−1` (resolve), idempotent; never `+1`. Wins/ loses the CAS vs discard — both idempotent. |
| `RecoveryOutcome::Retry` / `rearmRecoveryRetry` (cpp:396-398) | No | stays Live (ΔL=0); re-arms durable slot only if it already holds THIS obligation; no new id, no `+1` |
| `discardRecoveryRequestsOnShutdown` | No | only `−1` (resolve ShutdownDiscarded) |
| `discardPendingRecoveryAdmission` | No | clears durable flag; no obligation mutation |
| `finalizeShutdown` (SnapshotCoordinator) | No | epoch layer; unrelated to `recoveryAdmissions_` |

**Conclusion of R12-4:** no stage after `requestShutdown` performs `+1` on the recovery table.
The two producer threads that historically drove recovery (`CoordinatorLoop` running
`runCoordinatorPhase`→`redrive`; `RebuildThread` consuming intents) are **joined before** the discard that
closes the Live slots, so no concurrent `+1`/`−1` can race the drain. `liveCount_==0` is therefore reached
and **stuck**: idempotent `−1` + gated `+1` ⇒ once 0, always 0 for the remainder of shutdown.

---

## R11 hardening recommendation #1 → elevated to R13 implementation candidate

Because B (logical-obligation residency) is **not** in `isFullyDrained()`, the guarantee that
`liveCount_==0` holds relies on *ordering* (CoordinatorLoop/Buider joined before discard) rather than on
the predicate. This is a sound but **incidental** guarantee: a future refactor that calls
`isFullyDrained()` before `discardRecoveryRequestsOnShutdown()` completes (e.g., a different teardown
order, or a `waitForDrain` timeout path that skips discard) would observe `isFullyDrained()==true` while
Live obligations still exist.

**R12 decision: CONDITIONAL PASS.** The invariant is *logically* true, but the **structural assertion**
(`liveLogicalRecoveryObligationCount() == 0`) is **absent from the predicate**. Per R12 criteria, this is
the CONDITIONAL-PASS branch; R11's hardening recommendation #1 is elevated to the **D105-R13 implementation
candidate** (`add liveLogicalRecoveryObligationCount()==0 to isFullyDrained()`), to be applied **only after**
this proof is recorded — not ad-hoc.

---

## Verdict

- **Logically:** `requestShutdown ⇒ no new admission ⇒ discardTerminalizes all Live ⇒ liveCount_==0` — **PROVEN**.
- **Structurally:** `isFullyDrained()` does **not** assert `liveLogicalRecoveryObligationCount()==0`
  (it is **A ∧ C**, not **A ∧ B ∧ C**) — **CONDITIONAL**.
- **R12 outcome: CONDITIONAL PASS → R13 hardening candidate promoted.**
- **D105-R5-10 regression:** None (redrive is ΔL=0 by construction; discard closes every Live slot incl. `None`).

### Pending structural gap (not a shutdown defect; tracked for R13/R15)
`RecoveryOutcome::Failed → ResolvedFailed (−1)` is reachable in **normal operation**
(`RuntimePublicationOrchestrator.cpp:189,255,303,386`; test C8 `ISRSemanticValidationTests.cpp:1034`),
which `liveCount_`→0 and is idempotent, but which **vanishes** an obligation via a terminal *not* in I4's
`{Success, Superseded, ShutdownDiscard}` set. This is a cross-cutting I4↔runtime contract tension
(pre-existing, R5-8/R5-9), unrelated to shutdown lifetime, logged as a deferred R15 item (no R5-10 change).
