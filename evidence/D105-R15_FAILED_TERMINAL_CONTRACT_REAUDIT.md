# D105-R15 — Failed Terminal Disposition / I4 Ownership Contract Re-audit

**Status:** AUDIT COMPLETE — **R15-A: CONTRACT VIOLATION CONFIRMED** (pending R16 design).
**Source changes: 0 (read-only).** Prerequisite: D105-R13 PASS (drain predicate now structurally asserts
`liveLogicalRecoveryObligationCount()==0`).

Purpose: determine whether `RecoveryOutcome::Failed` is a **legitimate logical-obligation extinction**
(allowed by I4-D14.3/D15.2) or a **contract violation** (I4 disappearance set = {Success, Superseded,
ShutdownDiscard} only). Do **not** change `Failed` semantics here — classify, then hand off to R16.

---

## R15-1 — Complete producer enumeration of `RecoveryOutcome::Failed` (production)

The `RecoveryOutcome::Failed` outcome is emitted at **exactly four production sites** (plus the discard
path which uses `ShutdownDiscarded`, and test C8) — all flowing through the **single** completion authority
`resolveRecoveryObligation` → `RecoveryAdmissionTable::resolve` (single idempotent `−1`).

| # | File:Line | Context (caller → decision/failure condition) | Returned Decision |
|---|---|---|---|
| 1 | `RuntimePublicationOrchestrator.cpp:189` | `trySubmitImpl`: world build #1 `worldBuilder.buildRuntimePublishWorld` returns null (`!worldOwner`, cpp:176/178) | `RejectedNotFinalized` |
| 2 | `RuntimePublicationOrchestrator.cpp:255` | `trySubmitImpl`: world build #2 (crossfade rebuild, cpp:244) returns null | `RejectedNotFinalized` |
| 3 | `RuntimePublicationOrchestrator.cpp:303` | `trySubmitImpl`: `executor_.publish(...) != Success` (cpp:276) | `RejectedPublishFailure` (or `RejectedShutdown` cpp:305 if `isShutdownInProgress()`) |
| 4 | `RuntimePublicationOrchestrator.cpp:386` | `submitPublishRequest` switch `RejectedNotFinalized` → `resolveIfRecovery(Failed)` | `RejectedNotFinalized` (redundant/idempotent route of #1/#2) |

- Discard path: `ISRRuntimePublicationCoordinator.cpp:1156` uses **`ShutdownDiscarded`** (I4-allowed), **not** `Failed`.
- Test only: `ISRSemanticValidationTests.cpp:1034` (C8).

### Full call chain (identical for all four; idempotent)

```
caller (trySubmitImpl / submitPublishRequest switch)
  → decision/failure condition              (build-null / publish!=Success / RejectedNotFinalized)
  → resolveRecoveryObligation(req.recoveryObligationId, RecoveryOutcome::Failed)
        (cpp:948) id==0 → early return (non-recovery publishes are no-ops); Retry → early return (ΔL=0)
  → RecoveryAdmissionTable::resolve(id, ResolvedFailed)      (h:366-378)  ← single −1 authority
        CAS Live→ResolvedFailed (cpp:371); liveCount_-- (cpp:372); idempotent (lost CAS ⇒ no-op, no double −1)
  → liveCount_ == 0  (observable via liveLogicalRecoveryObligationCount(), h:426-428)
  → return Decision   (189→RejectedNotFinalized, 255→RejectedNotFinalized, 303→RejectedPublishFailure/Shutdown, 381→386 route)
```

**Key invariants verified (R15-6):** after every `Failed`, the code does **not** call `tryInsert`/`submitRecoveryRequest`
in the same path — all four producers `return` the `Decision` immediately after resolving (189→190, 255→256,
303→306, 386→387). `nextId_` is strictly monotonic (h:389), so no slot is ever reused with a stale id (ABA-safe, C9).
`LiveCount==0` is therefore a genuine disappearance, not a transition.

---

## R15-2 — Classification of each `Failed` producer

| # | Producer | Branch meaning | Class |
|---|---|---|---|
| 1 | cpp:189 build-null #1 | world-build failed; obligation's publish could not be constructed | **A** (obligation disappears) |
| 2 | cpp:255 build-null #2 | crossfade rebuild failed; obligation disappears | **A** |
| 3 | cpp:303 publish!=Success | publish execution failed; obligation disappears | **A** |
| 4 | cpp:386 switch route | general guarantee: RejectedNotFinalized ⇒ `Failed` (idempotent); net −1 once | **A** |

All four are **A — the recovery obligation itself failed and disappears; no −1→retry and no −1→new obligation**.
None are B (no "operation failed but obligation stays Live" — that would be `Retry`), none are C (no new
obligation id emitted), none are D (not a supersede/shutdown mislabeled — those use
`StaleSuperseded`/`ShutdownDiscarded`).

### Rejection-reason → outcome mapping (R15-4)

| Rejection reason | Producer site | Outcome | −1 / ΔL | I4-allowed disappearance? |
|---|---|---|---|---|
| `RejectedStaleGeneration` | cpp:372 switch | `StaleSuperseded`→`ResolvedStaleSuperseded` | −1 | ✅ (Superseded) |
| `RejectedNotFinalized` | cpp:189/255/386 | `Failed`→`ResolvedFailed` | −1 | ❌ **NOT in {S,SS,SD}** |
| `RejectedPublishFailure` | cpp:303 | `Failed`→`ResolvedFailed` | −1 | ❌ **NOT in {S,SS,SD}** |
| `RejectedPressure` | cpp:388 switch | `Retry` (rearm via rearmRecoveryRetry, cpp:396-398) | 0 | ✅ (stays Live — not a disappearance) |
| `RejectedShutdown` | cpp:400 switch | `ShutdownDiscarded` | −1 | ✅ (ShutdownDiscard) |
| `Accepted` | cpp:312 | `Published`→`ResolvedSuccess` | −1 | ✅ (Success) |

`RejectedNotFinalized` (build failure) and `RejectedPublishFailure` (publish-execution failure) are
**distinct failure conditions** (different `FailureReason`/`FailureStage`: `trySubmit:build`,
`trySubmit:rebuild`, `trySubmit:publish`, `publishFailure`) but **share** the same ownership outcome
(terminal `Failed`). They are separable for telemetry; the split does **not** change the
recovery-obligation contract violation. `RejectedPressure → Retry` is ΔL=0 (R5-9 MUST-2), correctly NOT a disappearance.

---

## R15-3 — Ownership tracking around the publish failure (case: RejectedPublishFailure, cpp:303)

`FailureReason::PublishFailed` path (cpp:277-307). Ownership of each entity before/after:

| Entity | Before `executor_.publish` | After `Failed` resolve | Mechanism |
|---|---|---|---|
| `newDSP` | owned by `req` / `frozen` world (publish in flight) | **destroyed immediately** (`destroyRolledBackDSP(newDSPResolved)`, cpp:287) | publish never activated → no epoch-gated retire needed |
| `RuntimeWorld` (old) | current active world | **retained** (publish did not activate crossfade/commit) | untouched |
| `RuntimeWorld` (new/frozen) | wrapped `worldOwner` | destroyed with `newDSP` via `destroyRolledBackDSP` | cpp:287 |
| `DSPHandle` (old) | active handle | retained | no change |
| `DSPHandle` (new) | in-flight publish | destroyed with new world | cpp:287 |
| `PublicationRequest` | pending | discarded (decision returned to caller) | caller handles re-admit as a *new* request |
| `RecoveryObligation` | Live, assigned `recoveryObligationId` | **`ResolvedFailed`**, `liveCount_--` (cpp:372) | `resolve` CAS Live→ResolvedFailed |

**Critical decoupling:** the *logical recovery obligation* and *DSP/resource ownership* are independent.
`Failed` severs the obligation (terminal + `liveCount_--`) **and** the DSP is reclaimed (`destroyRolledBackDSP`),
independently and safely — so `Failed` is not a leak. But it **is** a disappearance not sanctioned by I4.

Build-failure paths (cpp:189/255) differ only in reclamation: they `lifetime_.retire(newDSPResolved)`
(cpp:181/247, epoch-deferred retire) rather than `destroyRolledBackDSP`, because the world never reached
the publish-frozen stage. Either way: obligation terminalized (ResolvedFailed, −1), no new obligation id,
old world retained.

---

## R15-5 — I4 contract comparison

I4 fixed contract (DISAPPEARANCE): a logical obligation's Live count decrements **only** when it vanishes for
a sanctioned reason ∈ `{ Success, Superseded, ShutdownDiscard }`. Current terminal→`ObligationState` mapping
in `resolveRecoveryObligation` (cpp:957-960):

| RecoveryOutcome | → ObligationState | I4 disappearance? |
|---|---|---|
| `Published` | `ResolvedSuccess` | ✅ `Success` |
| `StaleSuperseded` | `ResolvedStaleSuperseded` | ✅ `Superseded` |
| `ShutdownDiscarded` | `ShutdownDiscarded` | ✅ `ShutdownDiscard` |
| `Failed` | `ResolvedFailed` | ❌ **`Failed` ∉ {Success,Superseded,ShutdownDiscard}** |
| `Retry` | (none — stays `Live`) | n/a (ΔL=0, not a disappearance) |

`resolveRecoveryObligation` (cpp:960) falls through: anything not Published/StaleSuperseded/ShutdownDiscarded
lands in `ObligationState::ResolvedFailed` (h:275). So `Failed` is a **genuine disappearance** (Live→terminal,
`liveCount_--`, no replacement id) that I4-D14.3/D15.2 **does not permit**. The disappearance is not a naming
quibble: `ResolvedFailed` is distinct from `ResolvedSuccess/ResolvedStaleSuperseded/ShutdownDiscarded`, and
the counter decrement is real.

---

## R15-6 — "Transition to another obligation" hypothesis: RULED OUT

A transition (C) would mean `old obligation → Failed → new obligation (new id)` such that
`liveLogicalRecoveryObligationCount()` is conserved across distinct obligations. Evidence against:

1. `nextId_` is strictly monotonic (h:389); every `tryInsert` (cpp:873, h:343) issues a strictly-greater id.
   A reused slot never matches a stale id (no ABA — C9 proves this). So a "Failed→re-admit" would carry a
   **new id**, i.e. a *new* obligation, not a continuation of the old one.
2. The four `Failed` producers **immediately return** the `Decision` after resolving (no `tryInsert`/`submitRecoveryRequest`
   in the failure branch). The publish-failure path only calls `destroyRolledBackDSP` (cpp:287) — resource cleanup, not re-admission.
3. `redriveDeferredRecoveryObligations` (cpp:982) is ΔL=0 by construction (R5-10): it re-attaches delivery to an
   *existing Live* (delivery==None) obligation only. A `ResolvedFailed` slot is terminal; `resolve`'s CAS
   (`Live→terminal`) will not fire again (idempotent), and `redrive` never creates ids. So a Failed
   obligation is never resurrected, nor does redrive substitute a new obligation for it.

⇒ `Failed → ResolvedFailed → liveCount_-- → no successor obligation` is a true disappearance. **R15-A confirmed.**

---

## R15-7 — Test (C8) vs I4 contract vs Implementation consistency

`testRLOE_C8_shutdownRaceOnce` (ISRSemanticValidationTests.cpp:1028-1044):
- submits obligation (id 44, L=1);
- `resolveRecoveryObligation(id, Failed)` (cpp:1034) **then** `resolveRecoveryObligation(id, ShutdownDiscarded)` (cpp:1035, no-op);
- asserts `liveLogicalRecoveryObligationCount()==0` (cpp:1036) and
  `recoveryObligationShutdownDiscardCount()` **unchanged** (cpp:1037 — Failed must NOT bump the
  shutdown-discard counter, which only `ShutdownDiscarded` increments, cpp:962).

Three layers are **distinct** here:
- **Implementation behavior:** `Failed` terminalizes & decrements L; idempotent. ✅ internally consistent.
- **Test expectation (C8):** codifies exactly that behavior (`Failed → L=0`, idempotent, no shutdown-discard count). ✅ consistent with implementation.
- **I4 contract (D14.3/D15.2):** `Failed` is NOT a sanctioned disappearance. ❌ violated.

C8 is an **implementation-behavior test**; it does not (and cannot) justify `Failed` against I4. The test
*codifies the violation* rather than proving its validity. Hence C8 must be **re-evaluated in R16**
(alongside any outcome reclassification); in R15 it is only audited, not changed.

---

## R15 verdict

### **R15-A — CONTRACT VIOLATION CONFIRMED**

- `Failed` (produced at RuntimePublicationOrchestrator.cpp:189, 255, 303, and defensively at 386) ⇒
  `resolveRecoveryObligation(..., Failed)` ⇒ `ObligationState::ResolvedFailed` ⇒ exactly one idempotent
  `liveCount_--` (single `−1` authority, h:366-378), **no new obligation id**, old world/DSP reclaimed
  independently.
- This is a **genuine logical-obligigation extinction (classification A)**, but `Failed` ∉
  I4's disappearance set `{ Success, Superseded, ShutdownDiscard }` ⇒ **violates I4-D14.3/D15.2.**
- Non-recovery publishes are unaffected (`resolveRecoveryObligation` early-returns on `id==0`, cpp:950).
- R13 hardening (predicate asserts `liveLogicalRecoveryObligationCount()==0`) is unaffected and remains
  sound — `Failed` still decrements L correctly; the predicate merely checks the *result* of all terminals.

**No source change in R15** (read-only). Outcome handed to **R16** for design: decide whether
`RejectedNotFinalized` / `RejectedPublishFailure` should become `Retry`/deferred-rebuild (preserve the
Live obligation for redrive) vs. add `Failed`-as-extinction to I4, or introduce a sanctioned terminal.
R14 is deferred until R15/R16 resolve the `Failed` semantics.

### Pending decision (R16 input — not resolved, not implemented)
- Should build/publish failure **preserve** the obligation (`Retry`-style, ΔL=0, redriven by the existing
  `redriveDeferredRecoveryObligations`) — matching I4's "no unsanctioned disappearance"?
- Or is `Failed`-as-extinction the *intended* semantics (the obligation gave up after one attempt), in which
  case I4-D14.3/D15.2 must be amended to add `Failed`/extinction?
- R5-10's `redriveDeferredRecovery` only re-attaches `delivery==None` obligations — a `Failed` obligation
  is `ResolvedFailed` (terminal), so it is *not* currently redriven; if "retry on failure" is desired, the
  failure path must NOT terminalize (route to `Retry` + re-enqueue) instead.
