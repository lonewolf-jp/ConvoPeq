# D105-R6 — Recovery Logical Obligation Lifecycle / Completion Authority Audit

**Type**: Read-only structural audit / 0 source changes
**Purpose**: Map the R5 `LogicalRecoveryObligation` lifecycle onto the real Builder / Coordinator / Commit / Reclaim / Shutdown code; fix the **single authority for +1 / −1** and the exact **completion point**. This audit **stops at defining authorities** — no implementation, no proof.
**Date**: 2026-08-27
**Status**: ✅ Complete (with **R6-GATE = NO-GO** → R5 design must be revised one level before R5-7)
**Prereq**: `evidence/D105-R5_RECOVERY_LOGICAL_OBLIGATION_IDENTITY_ADMISSION_MODEL_AUDIT.md`

---

## 0. R6-GATE Verdict

> Per the R6 brief: *"if `reclaimSlot(handle)` is insufficient as completion authority, OR `intentId` is not preserved in the existing transport path, do NOT proceed to R5-7; revise the design one more level."*

**Both conditions are TRUE** (proven below):
- **`reclaimSlot(handle)` is NOT a valid completion authority** (§2).
- **`intentId` is preserved only up to the Builder; it is dropped before publish/commit** (§3).

⇒ **R6-GATE = NO-GO**. The R5 model's §6 completion proposal (`recovered world published AND reclaimSlot(handle) fired`) is **wrong** and must be revised (R5-rev1, see §7 / appended to D105-R5). R5-7 inductive proof is **blocked** until the revision lands.

---

## 1. Three Distinct Completion Events (R6 §1)

| Event | Definition | Code anchor |
|---|---|---|
| **BuildCompleted** | `runtimeBuilder.build()` returned non-null | AudioEngine.RebuildDispatch.cpp:1030 (`recoveryResult.runtime != nullptr`) |
| **PublicationCompleted** | `enqueuePublicationIntentForRuntimeCommit(...)` enqueued the publish request | RebuildDispatch.cpp:999 (transport) / :1072 (durable) |
| **RecoveryObligationCompleted** | logical obligation resolved → **−1** | **MISSING** — no code today; must be a new `onRecoveryBuildResolved(handle, intentId, status)` signal (§6) |

**Critical separations confirmed:**
- `BuildCompleted != PublicationCompleted`: build can succeed but publish enqueue can still be skipped/aborted (e.g., `dspToCommit == nullptr` guard, or thread-exit mid-loop).
- `PublicationCompleted != reclaimSlot()`: publish commits a world; `reclaimSlot` is an independent generation-gated flag release (§2) that may fire **before** any recovery builds, **without** any recovery, or for a **different** obligation.

⇒ RecoveryObligationCompleted must key on **PublicationCompleted of the recovered world for a specific obligation identity**, NOT on reclaim.

---

## 2. `reclaimSlot()` is NOT the Completion Authority (R6 §2)

`reclaimSlot` (ISRDSPQuarantine.cpp:49):
```
void DSPQuarantineManager::reclaimSlot(uint32_t slot, uint64_t generation) {
    // resolves an auditLog_ entry by (slot, generation); frees the quarantine FLAG.
}
```
Only caller — AudioEngine.Commit.cpp:661, inside the reclaim loop gated by:
```
graceCompleted = worldAuthority_.lifetime().isGracePeriodCompleted(
                    world->generation, maxObservedGeneration, callbackActiveCount);
```

| Check | Result | Consequence for −1 |
|---|---|---|
| Recovery-specific? | **NO** — flag release for any quarantine, recovery or not | would −1 a non-existent obligation |
| Applies to Path-B? | **YES** — Path-B (`quarantineSlot`, Threading.cpp:38-72) sets the same flag with **no recovery intent** | phantom −1 with no matching +1 |
| Distinguishes O1 vs O2 (same handle, different target)? | **NO** — keyed by `slot` + `generation` only | ambiguous which obligation completed |
| Needs generation/`intentId`? | generation only; `intentId` absent | cannot bind to a logical obligation |
| Fires for non-recovery reasons? | **YES** — any newer-world observation | −1 on unrelated event |

**Conclusion**: Tying `−1` to `reclaimSlot(handle)` is **invalid**. It would (a) fire early (before recovery builds), (b) fire for Path-B with no obligation, (c) mis-attribute O1/O2, (d) fire on unrelated generation advances.

---

## 3. `intentId` Effectiveness (R6 §3)

`RecoveryIntent` (ISRRuntimePublicationCoordinator.h:217) carries `uint64_t intentId` ("diagnostic/monitoring sequence number"). `PendingRecoveryAdmission` (h:680) also carries `intentId`.

| Question | Answer | Evidence |
|---|---|---|
| 1. Who generates? | `nextRecoveryIntentId_.fetch_add(1, relaxed)` — monotonic, unique per attempt | submitRecoveryRequest (cpp:821) |
| 2. Survives transport? | **YES** — `recoveryIntentQueue_.push(intent)` | cpp:850 |
| 3. Survives durable? | **YES** — `pendingRecoveryAdmission_.intentId = intent.intentId` | cpp:867 |
| 4. Reaches Builder? | **YES (struct only)** — `popRecoveryRequest()` / `takePendingRecoveryAdmission()` return `RecoveryIntent` with `intentId` | cpp:944; cpp:884 |
| 5. Reaches publish? | **NO** — Builder reads `recovery->handle` + `recovery->buildSource` only; passes `rebuildRequestGeneration` (global), **not** `intentId`, to publish | RebuildDispatch.cpp:1062-1072 |
| 6. Completion-side reverse lookup? | **NO** — `enqueuePublicationIntentForRuntimeCommit(DSPCore*, int generation, snapshot, ...)` (AudioEngine.h:2551) carries **no handle, no intentId**; commit (Commit.cpp:782) does `registerDSPHandleForRuntime(newDSP)` (fresh handle) | Commit.cpp:782-840 |
| 7. Maintained across retry? | **YES** — durable retry reuses same slot/intentId (`settle(true)`) | cpp:1034/1056 |
| 8. Coalesce mapping? | **OPEN** — see §6 | — |

**Conclusion**: `intentId` is generated and preserved through transport + durable to the Builder, but is **discarded at the Builder→publish boundary**. The R5 candidate `onRecoveryObligationResolved(handle, intentId)` is **not implementable as-is**; `intentId` must be plumbed through `enqueuePublicationIntentForRuntimeCommit` → commit → completion signal.

---

## 4. `handle`-only Completion Counterexample (R6 §4)

Constructible in real code:
```
t0: S quarantined, Target A  →  Obligation O1 (intentId=1), +1
t1: O1 popped, building (publish not yet done)
t2: S quarantined again, Target B (different buildSource) → Obligation O2 (intentId=2), +1
t3: reclaimSlot(S) fires (generation advance)           ← cannot tell O1 vs O2
```
- `(S, A)` and `(S, B)` are **distinct targets** ⇒ distinct obligations (R5 §2) — both legitimately +1.
- `reclaimSlot(S)` resolves slot S by generation alone ⇒ **cannot identify which obligation completed**.
- Also O2's `buildSource` differs ⇒ even a target fingerprint wouldn't suffice if we only keyed on `handle`.

**Conclusion**: `handle` alone (and `handle+generation`) is **insufficient** as completion identity. Need `(handle, intentId)` — or, post-coalesce, the **logical obligation id** (§6).

---

## 5. Retry Semantics — Count-Neutral? (R6 §5)

| Path | Today | Count-neutral? |
|---|---|---|
| Durable retry | build fail → `settlePendingRecoveryAdmission(true)` → `Building→DurablePending`, same slot, same `intentId`, **no new RecoveryIntent, no +1** | ✅ Yes |
| Transport retry | build fail → `continue` (**no retry, no durable fallback, no −1**) | ✅ count-neutral BUT **leaks the obligation** (see §8) |

**Conclusion**: No path does `retry → new RecoveryIntent → +1`. Retry is count-neutral by construction. The transport build-failure `continue` is count-neutral but **incorrect** (lost recovery + leaked obligation). R5-rev1 must add a **Failed/Discarded terminal** that does `−1` (or a transport retry), so the obligation does not leak.

---

## 6. Coalesce → Transport Residency (R6 §6)

Current `submitRecoveryRequest` always `push`es to `recoveryIntentQueue_` (256), and on full does a **blind durable overwrite** (INV-X1-7 defect, R4 §7). Under the R5 model:

```
AdmissionAttempt A (intentId=1, C=Cs)  →  O1 created (+1), transport entry A pushed
AdmissionAttempt B (intentId=2, C=Cs)  →  coalesce: O1 already has C=Cs
AdmissionAttempt C (intentId=3, C=Cs)  →  coalesce: same
```
Required design (count-neutral):
- Coalesce lookup by `CoalesceIdentity C = {handle, SemanticRecoveryTarget}` happens **before** push (Admission authority).
- If `O` with matching `C` is live: **drop the new attempt** — do NOT push a new transport entry, do NOT overwrite durable, do NOT +1. Keep `O`'s existing delivery state (Transport/Durable/Building).
- `O.intentId` (the surviving obligation's id) is what completion must carry; `A.intentId/B.intentId/C.intentId` are **consumed attempts**, not completion identities. So completion identity = **logical obligation id** (which equals the coalesced O's id), NOT the per-attempt `intentId`.

**Open decision (carried to R5-rev1)**: when coalescing, should the surviving `O` be re-armed (e.g., if it is in `DurablePending` and a newer attempt arrives, prefer the newer `buildSource`)? Recommended: re-arm only the `buildSource` if `O` is still pre-Building; never create a new entry.

---

## 7. Table / Delivery Ownership Boundary (R6 §7)

| Question | R6 answer | Code evidence |
|---|---|---|
| Who owns `LogicalRecoveryObligation`? | **Coordinator (Admission Authority)** | `submitRecoveryRequest` documented as "Admission/Notify authority は Coordinator" (cpp:813) |
| Who owns transport entry? | Coordinator delivery (queue) — but it is **not** the obligation | `recoveryIntentQueue_` (h:642) |
| Who owns durable slot? | Coordinator delivery — "World ownership を持たない" (h:656) | `pendingRecoveryAdmission_` (h:667) |
| Who issues completion? | **Coordinator completion authority**, driven by a Builder signal | today only `settle` for durable; transport has none |
| Who does `−1`? | **Single**: Coordinator completion authority | must be added |
| May Builder directly ±logical count? | **NO** — Builder only notifies status | recommendation; currently Builder calls `settle` (durable) but no logical count exists |

**Corrected authority chain (R5-rev1):**
```
Builder  ──(build status: success/failure, handle, intentId)──▶  Coordinator completion authority
Commit/Reclaim  ──(physical reclaim event, NOT a completion)──▶  ignored for −1
Coordinator completion authority  ──(state transition + −1)──▶  LogicalRecoveryObligation
```

---

## 8. Shutdown Re-audit (R6 §8)

| Delivery state at shutdown | Today | Problem |
|---|---|---|
| Transport queue | `discardRecoveryRequestsOnShutdown()` drains → +1 `recoveryShutdownDiscardCount_` per entry | per-entry, not per-obligation |
| Durable slot | `discardPendingRecoveryAdmission()` clears → +1 `recoveryShutdownDiscardCount_` | per-entry, not per-obligation |
| **Building** (popped, in Builder) | **NOT covered by either** | obligation never gets ShutdownDiscard → **leak** |

Problems:
1. **Double -1 risk**: one obligation with both a transport entry AND a durable slot (e.g., transport in queue + durable reserved after a prior full) would be counted twice.
2. **Building leak**: an obligation mid-build is in neither queue nor durable slot → never discarded.
3. **No logical table iteration**: shutdown must iterate the **`RecoveryAdmissionTable` once** and `−1` each live obligation (regardless of delivery state), with the channel drains as side effects only.

**Conclusion**: Shutdown discard must be **per-logical-obligation**, not per-delivery-entry. Current per-channel discards must be reconciled.

---

## 9. Accounting Table → Code Mapping (R6 §9)

| Event | State transition | Logical count | Authority | Current code location |
|---|---|---:|---|---|
| New unique admission | none → Created | **+1** | Admission | **MISSING** — add in `submitRecoveryRequest` post-coalesce |
| Coalesce | existing → existing | 0 | Admission | **MISSING** — add C-lookup before push |
| Queue enqueue | Created → Transport | 0 | Delivery | `recoveryIntentQueue_.push` (cpp:850) |
| Queue pop | Transport → Building | 0 | Builder | `popRecoveryRequest` (cpp:944) |
| Retry | Building → Durable | 0 | Builder/Admission | `settlePendingRecoveryAdmission(true)` (cpp:1034/1056) |
| Publish | Building → (PublicationCompleted) | 0 | Publication | `enqueuePublicationIntentForRuntimeCommit` (cpp:999/1072) |
| **RecoverObligationCompleted** | (PublicationCompleted) → Completed | **−1** | **Completion authority** | **MISSING** — add `onRecoveryBuildResolved(handle, intentId, status)`; **NOT** `reclaimSlot` |
| Build-failure terminal | Building → Failed | **−1** | Completion authority | **MISSING** — transport `continue` leaks today |
| Supersede | live → Superseded | **−1** | Admission | **MISSING** (Phase II) |
| Shutdown | live → ShutdownDiscard | **−1** | Shutdown authority | per-obligation iteration **MISSING**; today per-entry only |
| Reject | none | 0 | Admission | **MISSING** — add at table-full |

**Rows with NO code location today** (must be added in R5-8): New admission +1, RecoverObligationCompleted −1, Build-failure −1, Supersede −1, Shutdown per-obligation −1, Reject, Coalesce.

---

## 10. R6 Stop-Condition Answers (R6 §10)

| Required | Answer |
|---|---|
| Unique owner of `LogicalRecoveryObligation` | **Coordinator (Admission Authority)** |
| Unique `+1` authority | **Admission** (post-coalesce, in `submitRecoveryRequest`) |
| Unique `−1` authority | **Coordinator completion authority** (single), driven by Builder's `onRecoveryBuildResolved` |
| Exact `Completed` condition | **PublicationCompleted** of the recovered world for obligation `(handle, intentId)` — i.e., Builder build success **and** `enqueuePublicationIntentForRuntimeCommit` called. **NOT** `reclaimSlot`. |
| Is `intentId` necessary? | **YES** — it is the only existing per-attempt identity that survives to the Builder; must be plumbed to publish/commit to serve as completion identity (or introduce a dedicated `recoveryObligationId`). |
| Can `handle` alone be completion identity? | **NO** (§4 counterexample). Need `(handle, intentId)` / logical-obligation-id. |
| Is retry count-neutral? | **YES** (durable retry reuses slot/intentId; transport has no retry). |
| Is coalesce count-neutral? | **YES** if new attempts are dropped (no new entry, no +1). |
| No double-count on shutdown? | **NO today** — per-entry discard; must become per-obligation. |
| Transport/durable ↔ logical mapping | **1 logical obligation → 0..1 delivery entries**; coalesce keeps exactly one; completion keyed by obligation id, not delivery entry. |

---

## 11. Required R5 Design Revision (R5-rev1) — blocks R5-7

1. **Completion authority = dedicated Builder→Coordinator signal** `onRecoveryBuildResolved(handle, intentId, status)` emitted at the Builder's recovery build-success (PublicationCompleted) **and** build-failure point, for **both transport and durable** paths.
2. **Plumb `intentId` + `handle`** from `RecoveryIntent` through `enqueuePublicationIntentForRuntimeCommit` (add params) into the commit path so the signal can be emitted. (Currently dropped at RebuildDispatch.cpp:1062-1072.)
3. **Drop `reclaimSlot` from the completion definition** in R5 §6 — it is a physical reclaim event, not a recovery-completion event.
4. **Add Build-failure terminal** (`Failed` → −1) so transport build failures do not leak the obligation.
5. **Shutdown = per-obligation iteration** of the `RecoveryAdmissionTable` for `−1`; channel drains are side effects only.
6. **Completion identity = logical obligation id** (surviving id after coalesce), not per-attempt `intentId`.

These are **design** changes only (R5-rev1); no source modified in R6.

---

## 12. Traceability

| Item | Location |
|---|---|
| `RecoveryIntent.intentId` (monotonic gen) | ISRRuntimePublicationCoordinator.h:217,220; cpp:821 |
| intentId → transport | cpp:850 (`recoveryIntentQueue_.push(intent)`) |
| intentId → durable | cpp:867 (`pendingRecoveryAdmission_.intentId = ...`) |
| Builder reads handle/buildSource only | AudioEngine.RebuildDispatch.cpp:1062-1072 |
| Builder passes global generation (not intentId) to publish | RebuildDispatch.cpp:1064 (`rebuildRequestGeneration`) |
| `enqueuePublicationIntentForRuntimeCommit` signature (no handle/intentId) | AudioEngine.h:2551; Commit.cpp:782-840 |
| durable `settle` (no logical count, no identity) | cpp:925; calls at RebuildDispatch.cpp:1023/1034/1056/1074 |
| transport build-failure `continue` (leak) | RebuildDispatch.cpp:~1031 |
| `reclaimSlot` def (flag audit resolve by slot+generation) | ISRDSPQuarantine.cpp:49-75 |
| `reclaimSlot` only caller (generation-gated) | AudioEngine.Commit.cpp:661 (loop gated by `isGracePeriodCompleted`, :655) |
| Path-B quarantine (no recovery intent) | AudioEngine.Threading.cpp:38-72 |
| transport discard (per-entry) | `discardRecoveryRequestsOnShutdown` (cpp:961) |
| durable discard (per-entry) | `discardPendingRecoveryAdmission` (cpp:917) |

### Tools used
- WSL `rg`/`sed`/`fdfind` (structural trace)
- context-mode MCP (`ctx_batch_execute` + `rtk`)
- **AiDex** (`aidex_query` — confirmed intentId/completion-identity absence)
- **semble** (`reclaimSlot completion authority`, `intentId publish path` → only the durable-overwrite / settle sites returned)
- **cocoindex** (`enqueuePublicationIntentForRuntimeCommit signature`, `reclaimSlot generation gated` → surfaced Commit.cpp + ISRDSPQuarantine definitions)
- **graphify** (knowledge graph — no `onRecoveryBuildResolved` / completion-authority node exists)
- **serena MCP** — attempted; language server timed out (corroborated by the above)

---

## 13. Conclusion

R6 maps the R5 lifecycle onto real code and **falsifies the R5 §6 completion candidate**:
- `reclaimSlot(handle)` is generation-gated flag release, recovery-independent, Path-B-affected, O1/O2-ambiguous → **cannot be the −1 event**.
- `intentId` survives to the Builder but is **dropped before publish** → completion identity must be plumbed (design change, not yet code).
- Transport recoveries have **no completion feedback at all**; build failures **leak** the obligation.
- Shutdown discards **per delivery entry**, not per obligation → double-count / Building-leak.

**R6-GATE = NO-GO for R5-7.** The R5 design must be revised one level (R5-rev1, §11) to define a dedicated Builder→Coordinator completion signal carrying `(handle, intentId)`. After R5-rev1 lands, R5-7 (inductive proof) becomes possible. No source was modified.
