# D105-R11 — Shutdown Lifetime / Drain Contract — Read-Only Re-audit

**Status:** PASS (shutdown lifetime). **Source changes: 0.**
**Scope:** Verify the pre-existing + D105-R5-10 recovery delivery residency survives shutdown with **no omission, no orphan, no double-free, no leak**. R5-10 changes are NOT modified.
**Base document:** `ConvoPepeg.md` baseline 2026-08-27 23:32:40 (latest). Design refs: `doc/work88` (I4 ownership contract, REPAIR_PLAN2, SHUTDOWN-7).
**Method:** static code trace (no edits). LSP/Serena are environmentally broken in this workspace → all traces by `rg`/`read`.

---

## 0. The four mandatory invariants

```
Shutdown begins  ⇒  no new logical recovery admission
Every already-admitted obligation  ⇒  exactly one terminal disposition
terminal disposition  ∈  { Success, Superseded, ShutdownDiscard }
ShutdownComplete  ⇒  recovery transport == empty ∧ durable recovery == empty ∧ logical obligation residency == 0 ∧ retire residency == 0 ∧ epoch quiescence proven
```

### I-1 — "Shutdown begins ⇒ no new logical recovery admission"  —  **PASS**

- `requestShutdown()` publishes `CoordinatorState::ShuttingDown` via `ShutdownScheduler::requestShutdown`
  (`ISRRuntimePublicationCoordinator.cpp:552-554`; `RuntimeIntentCoordinator::requestShutdown` cpp:464-466).
- `submitRecoveryRequest()` checks the gate **before** doing anything (including before the D105-R5-10
  opportunistic `redriveDeferredRecoveryObligations()`):
  `AudioEngine`-path gate `if (state_ == ShuttingDown) return false;`
  (`ISRRuntimePublicationCoordinator.cpp:817-821` — increment `recoveryShutdownDiscardCount_`, return false).
- `tryInsert` (the **only** `+1`, h:343) is reached **only** on the `submitRecoveryRequest` path *after* the
  shutdown gate, so it is unreachable post-`requestShutdown`. No other site inserts obligations.
- **D105-R5-10 regression check:** `redriveDeferredRecoveryObligations` / `redriveDeferredRecovery`
  re-attach delivery to an **existing** Live slot (`intent.obligationId = obligationId;`, cpp:1019) — they
  never call `tryInsert` and never issue a new `LogicalRecoveryObligationId`. So a "deferred" obligation
  being re-driven post-gate is **not** a "new admission". ✅

> Evidence: gate cpp:817-821; `tryInsert` only at h:343 / cpp:858; `redriveDeferredRecovery` cpp:991-1019.

### I-2 — "Every already-admitted obligation ⇒ exactly one terminal disposition"  —  **PASS**

Single `-1` authority: `RecoveryAdmissionTable::resolve` (h:366) — CAS `Live→terminalState` +
`liveCount_--`; idempotent (lost CAS / already-terminal → returns `false`, no second decrement).
All dispositions route through this single authority:

| Outcome | Where resolved | Terminal? | −1? |
|---|---|---|---|
| `Published` | `onPublishCommitted` / orchestrator.cpp:312 | yes → `ResolvedSuccess` | yes |
| `StaleSuperseded` | `submitPublishRequest` cpp:377 | yes → `ResolvedStaleSuperseded` | yes |
| `ShutdownDiscarded` | `discardRecoveryRequestsOnShutdown` cpp:1149 | yes → `ShutdownDiscarded` | yes (idempotent) |
| `Retry` | cpp:396 / `resolveRecoveryObligation` cpp:948 | **no** (stays Live, ΔL=0) | no |

- Shutdown disposes **all** Live slots (not just Transport/Durable): `discardRecoveryRequestsOnShutdown`
  loops `kCapacity` (cpp:1146-1150) and calls `resolveRecoveryObligation(oblId, ShutdownDiscarded)` for
  every non-zero id — this closes **both** the Live+None (deferred) and Live+Transport / Live+Durable
  obligations. So a **deferred (delivery==None) obligation is NEVER abandoned at shutdown**. ✅ (R11-6)

> Evidence: `resolve` h:366-378; `discardRecoveryRequestsOnShutdown` cpp:1136-1151; `resolveRecoveryObligation` cpp:941-959.

### I-3 — "terminal disposition ∈ { Success, Superseded, ShutdownDiscard }"  —  **PASS @ SHUTDOWN / OPEN @ NORMAL OP**

- **At shutdown:** only `ShutdownDiscarded` is emitted by `discardRecoveryRequestsOnShutdown` (cpp:1149). ✅
  `Success`/`Superseded` arise only from genuine publish completion / `StaleSuperseded` rejection — not from
  the shutdown drain itself.
- **Caveat (pre-existing, R5-8/R5-9, not introduced by R5-10, not a shutdown defect):** the normal
  operation path also admits `RecoveryOutcome::Failed` → `ObligationState::ResolvedFailed` (h:300-303),
  which performs the `-1` (vanishes the obligation). `Failed` is reachable in production at
  `RuntimePublicationOrchestrator.cpp:189,255,303,386` and is codified as an intended terminal by test
  `ISRSemanticValidationTests.cpp:1034` (C8: after `Failed`, `liveLogicalRecoveryObligationCount()==0`).
  `Failed` is **excluded** from I4's disappearance set `{Success, Superseded, ShutdownDiscard}`.
  → This is a **design-contract tension (I4 vs runtime)**, recorded as an **open item for reconciliation**,
  **outside R11's shutdown scope**. R11 does not modify runtime; the shutdown path itself stays within I4.
- Note `RecoveryOutcome::Superseded` is intentionally **not implemented** (h:296-297); the actual
  normal-operation terminals are Published/StaleSuperseded/Failed.

> Evidence: `RecoveryOutcome` enum h:298-304; `resolveRecoveryObligation` cpp:948-955; C8 test cpp:1034-1037.

### I-4 — "ShutdownComplete ⇒ recovery transport empty ∧ durable empty ∧ logical residency 0 ∧ retire 0 ∧ epoch quiescence"  —  **PASS**

`ShutdownScheduler::isFullyDrained()` (cpp:506-550) is the drain predicate used by
`markShutdownComplete` (cpp:556-567): it requires (all of):
```
swapPending_ == false
intentQueue_            .sizeApprox() == 0
observeDeferredRing_    .size()    == 0
quarantineFallbackQueue_.sizeApprox() == 0
recoveryIntentQueue_    .size()    == 0            ← recovery transport drained
retireBacklogCount_     == 0                        ← retire drained
publicationBacklogCount_== 0
publicationIntentResidencyCount_ == 0
pendingIntentCount_     == 0
reclaimInFlightCount_   == 0
quarantineIntentResidencyCount_ == 0
quarantineRingResidencyCount_ == 0
!recoveryAdmissionPersistent_                       ← recovery durable drained
```

- **Recovery transport drained:** `discardRecoveryRequestsOnShutdown` pops every leftover
  `recoveryIntentQueue_` intent via `popRecoveryRequest()` (cpp:1138), which does the matched
  `pendingIntentCount_` fetchSub (cpp:1124) — counters stay consistent, queue → 0. ✅
- **Recovery durable drained:** `discardPendingRecoveryAdmission()` clears the durable slot and sets
  `recoveryAdmissionPersistent_ = false` (cpp:1110). ✅
- **Logical obligation residency → 0:** the same discard loop terminalizes **all** Live slots → 0
  (`liveCount_`). The drain predicate does **not** explicitly assert `liveCount()==0`, but
  `discardRecoveryRequestsOnShutdown` closes every Live slot, and `tryInsert`+`submitRecoveryRequest`
  are unreachable post-`requestShutdown`, so no Live slot can be (re)created between discard and
  `markShutdownComplete`. ✅ (see "Hardening recommendations" — predicate could be tightened.)
- **Retire 0 / Epoch quiescence:** produced by the separate `retire`/`EpochControl` layer, drained by
  `waitForDrain`/`drainDeferredRetireQueues`/`drainAllQuarantineStore`/`tryReclaim` in
  `releaseResources` (ReleaseResources.cpp:504-561, 559) **before** `finalizeShutdown`/`markShutdownComplete`.
  Not R5-10's concern; R5-10 recovery does not touch retire counters. ✅

> Evidence: `isFullyDrained` cpp:506-550; drain call sites ReleaseResources.cpp:514-561; `discardRecoveryRequestsOnShutdown` cpp:1136-1151.

---

## 1. Shutdown admission gate

- `requestShutdown()` → `state_ = ShuttingDown` (cpp:552-554).
- `submitRecoveryRequest` returns `false` immediately when `state_ == ShuttingDown` (cpp:817-821),
  **before** the R5-10 redrive trigger (cpp:840) and before `tryInsert`. ⇒ no new logical admission. ✅
- `onPublishCommitted` / `trySubmitImpl` completion authority (`resolveRecoveryObligation`) keeps
  resolving existing ids (idempotent); never inserts. ✅

## 2. Recovery delivery drain

Ordering (SHUTDOWN-ORDER contract, `ReleaseResources.cpp:507-512`):
```
requestShutdown(:75)
  → shutdownCoordinatorLoop(:189, join)        // CoordinatorLoop = producer of recovery intents; STOPS here
  → stopRebuildThread(:190, join)              // Builder fully stopped
    → discardRecoveryRequestsOnShutdown()      // cpp:802
          • pop all recoveryIntentQueue_ intents (pendingIntentCount_ fetchSub on each pop)
          • resolve all Live slots → ShutdownDiscarded (−1 authority, idempotent)   cpp:1146-1150
    → discardPendingRecoveryAdmission()        // cpp:810  → recoveryAdmissionPersistent_=false
  → waitForDrain(:430) / isFullyDrained check
  → finalizeShutdown(:563)
  → markShutdownComplete(:623)                 // → isFullyDrained() ? Live : Faulted   cpp:556-567
  → transitionTo(ShutdownComplete)
```

- Because `shutdownCoordinatorLoop` is **joined** before `stopRebuildThread`→`discardRecoveryRequestsOnShutdown`,
  the CoordinatorLoop thread (the only thread running `runCoordinatorPhase`) has **stopped** before
  `discard` runs. ⇒ `redriveDeferredRecoveryObligations` (Threading.cpp:266) cannot push into the queue
  *after* the discard drains it. The drain is therefore stable when `isFullyDrained` is evaluated. ✅
- Every transport intent left in the queue is popped by discard; every obligation is terminalized. ✅

## 3. Logical obligation ownership (I4 ownership conservation)

- `+1` only: `tryInsert` (h:343, cpp:858) — guarded by `liveCount_ < kMaxLogicalRecoveryObligations(32)`
  and the `ShuttingDown` gate above. ✅
- `-1` only: `RecoveryAdmissionTable::resolve` (h:366) — idempotent CAS, no underflow. ✅
- Slot reuse: `nextId_` is monotonic (`++nextId_`, h:389), so a reused slot never collides with a stale id
  (ABA-safe) — `resolve` re-scans by id, no cached index (h:361-365). ✅
- `liveLogicalRecoveryObligationCount()` (h:426) is the `liveCount_` of the table; post-shutdown it is
  driven to 0 by the discard loop. ✅

## 4. Quarantine → Recovery → Shutdown conflict surface

Each delivery residency is accounted and drained independently:

- `None` (deferred, Live, no delivery) → `discardRecoveryRequestsOnShutdown` resolves to
  `ShutdownDiscarded` (cpp:1149). Not abandoned. ✅ (R11-6)
- `Transport` (intent in `recoveryIntentQueue_`) → drained by the `pop` loop (cpp:1138) + reservation
  `fetchSub` (cpp:1124); the obligation itself resolved ShutdownDiscarded. ✅
- `Durable` (single `PendingRecoveryAdmission` slot, `recoveryAdmissionPersistent_=true`) →
  `discardPendingRecoveryAdmission()` clears the slot + clears the flag (cpp:1110). ✅

No state leaves a recovery obligation un-accounted at shutdown. ✅

## 5. Retire / Epoch side — ordering to ShutdownComplete

- `releaseResources` (`AudioEngine.Processing.ReleaseResources.cpp:75-643`):
  `requestShutdown(:75)` ⇒ `closeAdmission` ⇒ `waitForDrain(2000,2)` (cpp:514) ⇒ on timeout
  `drainDeferredRetireQueues(true)` + `m_coordinator.tryReclaim()` (cpp:559-561) ⇒
  `m_coordinator.finalizeShutdown(timedOut)` (cpp:563) ⇒ `markShutdownComplete` (cpp:623).
- `waitForDrain` (Threading.cpp:196 `while (!isFullyDrained())`) polls `AudioEngine::isFullyDrained()`,
  which ANDs with `runtimePublicationBridge_.isFullyDrained()` (Threading.cpp:173) — i.e. the ISR layer's
  predicate above. The 2s drain budget and retry interval (2) match `REPAIR_PLAN2` SHUTDOWN-7. ✅
- Epoch quiescence (`advanceRetireEpoch`, `drainAllQuarantineStore`, active-reader==0 path) is handled by
  `RuntimeWorldAuthority`/`EpochControl`, drained in `releaseResources` (cpp:487-505). Recovery does not
  touch epoch/retire counters. ✅

## 6. D105-R5-10 regression (must not be undone by shutdown)

| Concern | Status | Evidence |
|---|---|---|
| `ObligationDeliveryState::None` deferred obligation not abandoned at shutdown | **PASS** | discard loop resolves every Live slot incl. `None` (cpp:1146-1150) |
| `redriveDeferredRecoveryObligations()` must NOT re-drive after shutdown closes | **PASS** | CoordinatorLoop joined before discard ⇒ redrive can't run post-discard; redrive is no-op on already-terminal slots (cpp:979,1005-1008) |
| shutdown discard vs redrive race (double delivery) | **PASS** | discard terminalizes via idempotent −1; redrive re-attaches delivery only (no new id, no +1); no shared counter is double-counted |

---

## R10 read-only audit (consolidated, re-verified during R11 trace)

- **R10-1** `ObligationDeliveryState` {None,Transport,Durable} semantics match usage (h:288-292; cpp:886/1010/1116/1126). ✅
- **R10-2** `redriveDeferredRecovery` issues **no** `tryInsert`/`resolve`/`liveCount_`++/-- (cpp:991-1055: only re-attaches existing id at cpp:1019; single −1 authority `resolve` untouched). ✅
- **R10-3** no-double-delivery: scan guard `delivery != None → continue` (cpp:981) + per-call `delivery != None → return` (cpp:1007) + C16 coalesce early-return (`wasDeferredBefore && delivery!=None`). ✅
- **R10-4** queue-full cascade verified by tests C11/C13/C14. ✅

Build & runtime confirmation (carried forward): Debug + Release ConvoPeg green; `ISRSemanticValidationRejects` (C1–C16) PASS; `ISRSoakTests` (256-stress) PASS.

---

## Hardening recommendations (read-only audit — NOT applied; require design sign-off)

1. **Tighten `isFullyDrained` predicate** (cpp:506): add an explicit
   `liveLogicalRecoveryObligationCount() == 0` check, so the drain predicate *asserts* logical residency
   is 0 rather than relying solely on "discard ran before markShutdownComplete". This would make I-4 fully
   structural and remove the latent ordering dependency (CoordinatorLoop-join-before-discard).
2. **Gate the `runCoordinatorPhase` redrive** (Threading.cpp:266) with `!isShutdownInProgress()`,
   mirroring the existing gate on the deferred-publish block at :276. Currently safe (CoordinatorLoop is
   joined before discard), but the gate makes the shutdown boundary explicit and removes the
   ordering-dependency. (Does **not** change R5-10 semantics — `submitRecoveryRequest` already gates its own
   redrive; only the periodic CoatTick redrive currently lacks the gate.)
3. Reconcile the `Failed → ResolvedFailed (−1)` normal-operation terminal with the **I4** disappearance
   set `{Success, Superseded, ShutdownDiscard}`. (I4: terminal failure is NOT a disappearance reason →
   `Failed` obligations should stay Live / be re-admitted, not vanish.) Pre-existing; out of R11 scope.

---

## Verdict

**D105-R11 PASS** — the shutdown lifetime / drain contract is satisfied for the recovery layer: no new
admission after `requestShutdown`, every admitted obligation reaches exactly one terminal disposition,
shutdown drains recovery transport + durable and closes every Live slot (including deferred `None`),
and `markShutdownComplete` only transitions to a non-Faulted state when `isFullyDrained` holds.
**D105-R5-10 regression: NONE.**
