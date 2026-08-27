# D105-R7 — Recovery Obligation Identity / Resolution Protocol Audit

**Type**: Read-only structural/design audit / 0 source changes
**Purpose**: Converge `Coordinator-owned LogicalRecoveryObligation` lifecycle — unique identity, coalesce preservation, Builder→Coordinator resolution, publication success/failure, retry, shutdown, transport/durable delivery — into **one resolution protocol**. **No R5-7 proof yet.**
**Date**: 2026-08-27
**Status**: ✅ Complete (R7-GATE = **PASS** — all G1–G12 determined; R5-7 now unblocked)
**Prereq**: `evidence/D105-R5_RECOVERY_LOGICAL_OBLIGATION_IDENTITY_ADMISSION_MODEL_AUDIT.md` (R5-rev1), `evidence/D105-R6_RECOVERY_OBLIGATION_LIFECYCLE_COMPLETION_AUTHORITY_AUDIT.md`

---

## 0. R7-GATE Verdict

G1–G12 (§12) are **all determined from source**. The protocol below is implementable. ⇒ **R7-GATE = PASS**. R5-7 (inductive capacity proof) is now permitted. (Per the R7 brief, this audit itself does **not** write R5-7.)

---

## 1. `RecoveryAttemptId` ≠ `LogicalRecoveryObligationId` (G1/G2)

| Id | Meaning | Source today | Allocated by |
|---|---|---|---|
| **`RecoveryAttemptId`** | one `submitRecoveryRequest` call = one AdmissionAttempt | `RecoveryIntent.intentId` = `nextRecoveryIntentId_.fetch_add(1)` (cpp:821) | Coordinator, at admission |
| **`LogicalRecoveryObligationId`** | one live logical obligation, may absorb ≥1 attempts via coalesce | **does NOT exist** (src grep `ObligationId` = 0 hits) | **Coordinator only** (NEW `nextLogicalRecoveryObligationId_`) |

```
Attempt A → attemptId=101 ┐
Attempt B → attemptId=102 ├─ CoalesceIdentity equal ─▶ LogicalRecoveryObligation O17 (obligationId=17)
Attempt C → attemptId=103 ┘
```
Reusing `intentId` as the obligation id is **forbidden** (it would make A/B/C collide incorrectly and break coalesce accounting). Obligation id is allocated **once**, when a *new* live obligation is admitted (post-coalesce), not per attempt.

---

## 2. `LogicalRecoveryObligationId` Uniqueness & Plumbing (G1)

| Property | Decision | Rationale |
|---|---|---|
| Allocator | Coordinator only (`RecoveryAdmissionAuthority`) | single owner (R6 §7) |
| Monotonic | `uint64_t nextLogicalRecoveryObligationId_++` | 64-bit ⇒ no practical wraparound; if it wraps, table is empty-by-then (shutdown clears) so reuse is safe |
| Wraparound | ignored (64-bit) | same as above |
| Reuse after shutdown | allowed — table cleared on shutdown init | no live entry survives |
| Plumbed to Builder | copied into `RecoveryIntent.obligationId` (NEW field) and `PendingRecoveryAdmission.obligationId` (NEW field) | Builder needs it to emit resolution |
| Plumbed to publish | copied into `PublishRequest.recoveryObligationId` (NEW field) at `enqueuePublicationIntentForRuntimeCommit` | **required** — see §6 |
| Reverse lookup | `RecoveryAdmissionTable` keyed by `obligationId` | single authoritative map |

**Feasibility gap (design requirement, not yet code):** `PublishRequest` (PublicationAdmission.h:19-26) currently has **no identity field** — only `newDSP/generation/sealedSnapshot/buildAnalysis/oversamplingResult/buildDiagnostics`. To carry the obligation id to completion, **add `uint64_t recoveryObligationId{0};` to `PublishRequest`** and thread it through `submitPublishRequest` → executor intent → `onPublishCommitted`. Without this field, the protocol cannot resolve completion by obligation id.

---

## 3. Coalesce Semantics — Option 1 (G3/G4)

**Adopt Option 1 (Phase I):**
```
Attempt B (CoalesceIdentity == O1.C) arrives:
    O1: unchanged (state, buildSource, obligationId preserved)
    Attempt B: discarded (no new transport entry, no +1)
```
- **Identity preserved** (G3): `O17` stays `O17` across coalesce; only the *attempt* id differs.
- **buildSource ownership unique** (G4): `O1` owns its original `buildSource`; incoming attempt's `buildSource` is dropped. No semantic-target mutation ⇒ proof stays simple.
- Even if `C == C'` but `buildSource != buildSource'` (possible: same fingerprint identity, different POD fields), Option 1 eliminates the ambiguity by keeping the original.

---

## 4. `CoalesceIdentity` vs `buildSource` (G4)

`CoalesceIdentity C = { handle, SemanticRecoveryTarget }` (R5 §3), where `SemanticRecoveryTarget` = `buildSource.rebuildFingerprint` (irIdentityHash/convolutionConfigHash/dspParameterHash + convolverFingerprint + sampleRate). Coalesce matches on `C`; the obligation's `buildSource` is locked at creation and **never overwritten** by later attempts (Option 1). Thus identity and buildSource ownership are both single-sourced.

---

## 5. Completion Protocol — Formal (G6)

```
Builder (or Shutdown authority)
        │  RecoveryResolution { obligationId, outcome }
        ▼
Coordinator completion authority  (single −1 authority)
        │  lookup RecoveryAdmissionTable[obligationId]
        │  if live: transition + count--   else: no-op (idempotent)
        ▼
LogicalRecoveryObligation terminal state
```
`outcome ∈ { Published, Retry, Failed, StaleSuperseded, ShutdownDiscard }`.

- `ShutdownDiscard` originates from the **Shutdown authority** (§9), routed through the same `RecoveryResolution` channel (not a separate Builder path) to keep one authority.
- Builder emits `RecoveryResolution` for build/retry/publish outcomes; it **never** directly ± the logical count (R6 §7).

---

## 6. Publish "Enqueue" vs "Committed" — Separated (G5)

R6's loose "enqueue ⇒ PublicationCompleted" is **refined**:
- `enqueuePublicationIntentForRuntimeCommit` (RebuildDispatch.cpp:999/1072) → `submitPublishRequest` (Commit.cpp:813) is **admission/enqueue**, NOT commit. The orchestrator can **reject** here (see §8).
- Actual commit completion = **`onPublishCommitted(PublicationSequenceId seqId)`** (RuntimePublicationOrchestrator.h:146; impl cpp:329), called from `RuntimePublishExecutor.h:105` after `executePublish` store-swap. Commit success is derived via `PublishStageResultTraits::isCommitted` (AudioEngine.h:3594, used :4659).
- **`−1` (Completed) fires at `onPublishCommitted`, carrying `recoveryObligationId`** (NEW param threaded from `PublishRequest`). Enqueue-time must NOT decrement.

**Conclusion**: publish completion point is source-identified as `onPublishCommitted`; the `recoveryObligationId` carry field (§2) is the only missing link.

---

## 7. Recovered DSP Handle ≠ Obligation Identity (G12)

`registerDSPHandleForRuntime(newDSP)` (AudioEngine.h:4249-4261) calls `dspHandleRuntime_.create(dsp)` → a **new** handle for the recovered world, distinct from the quarantined `S`. Therefore:
- `PublishRequest.newDSP` (the recovered-world handle) **must not** be used to reverse-lookup the obligation.
- The obligation id is carried **explicitly** via `recoveryObligationId` (§2), independent of any DSP handle.

---

## 8. Failure Decomposition — 3-Way (G8)

| Class | Trigger | Outcome | Count |
|---|---|---|---:|
| **A. Retryable build failure** | `runtimeBuilder.build()` returns null **and** O holds a durable reservation | `Retry` (→ Durable) | 0 |
| **A. (terminal)** | build null, **no** durable reservation (transport-only) | `Failed` | −1 |
| **B. Terminal build failure** | build null after retry exhausted | `Failed` | −1 |
| **C. Publication failure** — `QueuePressure` | orchestrator rejects (cpp:371) | `Retry` | 0 |
| **C. Publication failure** — `StaleGeneration` | world already superseded (cpp:357) | `StaleSuperseded` | −1 |
| **C. Publication failure** — `ValidationFailed`/`PublishFailed` | (cpp:365/386) | `Failed` | −1 |
| **C. Publication failure** — `ShutdownRejected` | (cpp:377) | `ShutdownDiscard` | −1 |

`FailureReason` enum is already returned synchronously by the orchestrator, so classification is feasible at `submitPublishRequest` return. **Key rule (R7 §8)**: build-ok but publish-failed does **not** silently destroy the obligation — `StaleGeneration` maps to a terminal −1 (moot), others to retry or terminal −1 explicitly. No silent loss.

---

## 9. Shutdown Protocol — Logical-Table-Centric (G9/G11)

```
on shutdown:
    for each live O in RecoveryAdmissionTable:
        emit RecoveryResolution{ O.obligationId, ShutdownDiscard }   // −1 (exactly once)
    // delivery cleanup (side effects, NOT accounting):
    discardRecoveryRequestsOnShutdown()   // drains transport queue
    discardPendingRecoveryAdmission()     // clears durable slot
    builder.stop()                        // stops in-flight build
```
`delivery cleanup ≠ logical discard`: the channel drains **never** emit a −1; only the table iteration does. This prevents double-count (transport+durable for one obligation) and Building-leak (popped-but-building obligation not in any channel).

---

## 10. Terminal-State Idempotency — No Double Completion (G10)

Completion authority pseudo-code:
```
onRecoveryResolution(id, outcome):
    O = table.lookup(id)
    if O == null or O.state is terminal: return   // no-op / diagnostic
    if outcome == Retry:
        O.state = Durable; return                  // count 0
    O.state = terminal(outcome)                    // Completed/Failed/StaleSuperseded/Superseded/ShutdownDiscard
    liveLogicalRecoveryObligationCount -= 1
```
Under a Builder-failure vs Shutdown race, whichever resolution arrives first transitions O to terminal; the second is a no-op ⇒ **exactly one −1**.

---

## 11. Final State Machine (source-backed) (G5/G6/G7/G8/G9/G10)

| Current | Event (source) | Next | Count | Authority |
|---|---|---|---:|---|
| none | unique admission (post-coalesce, `submitRecoveryRequest`+new id) | Created | +1 | Coordinator |
| live | equal admission (`CoalesceIdentity` match) | same | 0 | Coordinator |
| Created | `recoveryIntentQueue_.push` | Transport | 0 | Coordinator |
| Transport | `popRecoveryRequest` (cpp:944) | Building | 0 | Builder |
| Building | build fail + durable reservation | Durable | 0 | Builder→Coord |
| Building | build fail, no durable | Failed | −1 | Builder→Coord |
| Durable | `takePendingRecoveryAdmission` (cpp:884) | Building | 0 | Builder |
| Building | publish committed (`onPublishCommitted`+id) | Completed | −1 | Coordinator |
| Building | publish `QueuePressure` | Durable | 0 | Coordinator |
| Building | publish `StaleGeneration` | StaleSuperseded | −1 | Coordinator |
| Building | publish `ValidationFailed`/`PublishFailed` | Failed | −1 | Coordinator |
| Building | publish `ShutdownRejected` | ShutdownDiscard | −1 | Coordinator |
| live | supersede (Phase II) | Superseded | −1 | Coordinator |
| live | shutdown (table iter) | ShutdownDiscard | −1 | Shutdown auth |

Each terminal state (Completed/Failed/StaleSuperseded/Superseded/ShutdownDiscard) is reached by **exactly one event class**, idempotent (§10).

---

## 12. R7 Gate Checklist

| # | Condition | Status |
|---|---|---|
| G1 | `LogicalRecoveryObligationId` definable | ✅ new Coordinator-allocated id |
| G2 | separated from `RecoveryAttemptId` | ✅ `intentId` ≠ `obligationId` |
| G3 | coalesce preserves O identity | ✅ Option 1 |
| G4 | buildSource ownership single | ✅ O owns original |
| G5 | publish completion point source-identified | ✅ `onPublishCommitted` (not enqueue) |
| G6 | Builder→Coordinator resolution protocol defined | ✅ `RecoveryResolution{id,outcome}` |
| G7 | retry = count-neutral | ✅ Durable transition, 0 |
| G8 | terminal failure = exactly one −1 | ✅ per §8/§11 |
| G9 | shutdown = exactly one −1 | ✅ table-iter, not per-channel |
| G10 | double resolution impossible/idempotent | ✅ §10 guard |
| G11 | delivery cleanup ≠ logical accounting | ✅ §9 |
| G12 | newDSP handle ≠ obligation identity | ✅ explicit id carry |

---

## 13. Required (Design) Additions — no code yet

1. `uint64_t nextLogicalRecoveryObligationId_` (Coordinator).
2. `uint64_t recoveryObligationId{0};` on `RecoveryIntent`, `PendingRecoveryAdmission`, `PublishRequest`.
3. `RecoveryAdmissionTable` keyed by `obligationId` (R5 §10 #1).
4. `RecoveryResolution{obligationId, outcome}` message + single completion authority.
5. `onPublishCommitted(seqId, recoveryObligationId)` new param; thread `recoveryObligationId` from `enqueuePublicationIntentForRuntimeCommit` → `PublishRequest`.
6. Per §8 outcome classification at `submitPublishRequest` return.

All are design specs; **no source modified in R7**.

---

## 14. Traceability

| Item | Location |
|---|---|
| `RecoveryIntent.intentId` (attempt id) | ISRRuntimePublicationCoordinator.h:220; cpp:821 |
| `PublishRequest` (no id field) | PublicationAdmission.h:19-26 |
| `onPublishCommitted(seqId)` completion seam | RuntimePublicationOrchestrator.h:146; cpp:329; RuntimePublishExecutor.h:105 |
| commit-success derivation | AudioEngine.h:3594 (`PublishStageResultTraits::isCommitted`), :4659 |
| `registerDSPHandleForRuntime` (new handle) | AudioEngine.h:4249-4261 |
| `FailureReason` values | RuntimePublicationOrchestrator.cpp:357/365/371/377/386 |
| `submitPublishRequest` (sync, void) | RuntimePublicationOrchestrator.h:155; cpp:340 |
| transport pop / durable take | cpp:944 / cpp:884 |
| existing `ObligationId` in src | **0 hits** (confirms G1 new id needed) |

### Tools used
- WSL `rg`/`sed`/`fdfind` (structural trace)
- context-mode MCP (`ctx_batch_execute` + `rtk`)
- **AiDex** (`aidex_query` — confirmed `ObligationId` 0 hits, `PublishRequest` identity absent)
- **semble** (`onPublishCommitted completion`, `PublishRequest identity field` → confirmed no carry field)
- **cocoindex** (`onPublishCommitted seam`, `FailureReason publish failed` → surfaced orchestrator + executor completion chain)
- **graphify** (knowledge graph — no `LogicalRecoveryObligationId`/`RecoveryResolution` node exists)
- **serena MCP** — attempted; language server timed out (corroborated by the above)

---

## 15. Conclusion

R7 converges the entire R5/R6 lifecycle into a single `LogicalRecoveryObligationId`-centric resolution protocol:
- **Two ids**: `RecoveryAttemptId` (`intentId`) ≠ `LogicalRecoveryObligationId` (new, Coordinator-allocated).
- **Coalesce = Option 1** (O preserved, attempt discarded) ⇒ identity + buildSource single-sourced.
- **Completion = `onPublishCommitted`**, not enqueue; requires a new `recoveryObligationId` carry field on `PublishRequest` (the one missing link).
- **Failures 3-way**: retry (0) vs terminal −1 vs `StaleGeneration`−1; no silent loss.
- **Shutdown = per-obligation table iteration**; channel drains are side effects.
- **Terminal idempotency** prevents double −1.

All G1–G12 are determined ⇒ **R7-GATE = PASS**. The model is now fully specified for the R5-7 inductive capacity proof. No source was modified.
