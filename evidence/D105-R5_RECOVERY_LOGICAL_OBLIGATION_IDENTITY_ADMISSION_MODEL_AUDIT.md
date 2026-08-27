# D105-R5 — Recovery Logical Obligation / Identity / Admission Model Audit

**Type**: Read-only design re-fix audit / 0 source changes
**Purpose**: Re-define the capacity invariant as a *design* (not a code-derived number). Establish the logical-obligation identity/state/accounting model so that `liveLogicalRecoveryObligationCount ≤ 32` becomes provable in a later step (R5-7/R5-8). This audit **stops at "what minimal design changes are needed"** — no implementation, no proof.
**Date**: 2026-08-27
**Status**: ✅ Complete (model fixed; **revised R5-rev1 after R6 — see §6/§10/§11**)

---

## 0. Anchoring Design Fact (from I4, confirmed via cocoindex)

`doc/work88/I4_DESIGN_CONTRACT.md` **D25 — INV-CAP-7** already fixes the philosophy:

> *"kMaxLogicalRecoveryObligations = 32 is a deliberate resource bound. The value 32 is **NOT derived from upstream admission maxima**. The system MUST directly enforce: `reservedLogicalObligations <= 32`. No claim is made that 32 is an upstream behavioral maximum."*

This **validates the R5 approach**: adopt `liveLogicalRecoveryObligationCount ≤ 32` as the **direct top invariant**, and treat `Q_max=256 / queue=256 / A_max` as *trigger/delivery* mechanisms, NOT the bound. The `E_max × O_max ≤ 32` product form (Candidate D) is explicitly **not** the intended proof shape — INV-CAP-7 says 32 is not an upstream behavioral maximum.

---

## 1. Capacity Model — Formal Definitions

Five concepts, strictly separated (per R5 §1):

| Symbol | Definition | Current code value | Nature |
|---|---|---|---|
| `Q_max` | simultaneously **quarantined** slots | **256** (`kMaxSlots`, ISRDSPQuarantine.h:68) | trigger source |
| `A_max` | simultaneously **outstanding recovery admissions (attempts)** | ≤ **257** emergent (256 transport + 1 durable) | delivery mechanism |
| `E_max` | simultaneously **open RecoveryEpisodes** | **undefined** (`RecoveryEpisodeId` 0 hits) | lineage (diagnostic) |
| `O_max` | distinct **targets per episode** | **unbounded** (no per-lineage code cap) | target multiplicity |
| `L_transport` | recovery intents resident in `recoveryIntentQueue_` | ≤ **256** (h:642) | transport residency |
| `L_durable` | durable single slot `pendingRecoveryAdmission_` | ≤ **1** (h:667) | durable residency |
| `L_building` | obligations in `Building` state | ≤ **1** (durable only; no transport-build counter) | build in-flight |
| `L_stalled` | obligations in retry-stall | **0 / N-A** (no recovery-stalled counter) | retry state |
| **`L_logical`** | **live logical recovery obligations** (`liveLogicalRecoveryObligationCount`) | **does not exist** | **THE invariant** |

**Top invariant (adopted):**
```
liveLogicalRecoveryObligationCount  =  #obligations in {Created, Transport, Durable, Building}
                                    ≤ 32
```
`Q_max`, `A_max`, `L_transport`, `L_durable` are **independent, larger** quantities that must be *mapped down* into the 32-slot logical table by coalesce/backpressure. They are NOT used to derive 32.

---

## 2. State Transition Graph (R5 §4)

Each `LogicalRecoveryObligation` carries a state. Current code has only *implicit* transport + single durable slot; the full graph below is the **target model**.

```
                AdmissionAttempt
                     │  (coalesce check: C matches live obligation → merge, no new obligation)
                     │  (table full & no coalesce → REJECT with telemetry, no obligation created)
                     ▼
   ┌───────────► Created ──────────► Transport ──────────► Durable ──────────► Building
   │                 │                    │                    │                    │
   │                 │                    │                    │                    │ (build fail, retry=true)
   │                 │                    │                    │                    └──► Durable (retry)
   │                 │                    │                    │
   │                 │                    │                    │ (build success / Discarded)
   │                 │                    │                    ▼
   │                 │                    │                 settle(false) ─────► Completed  (−1)
   │                 │                    │
   │                 │   popRecoveryRequest() consumed, built & published
   │                 │                    ▼
   │                 │                 Completed  (−1)   ← NEEDS NEW BUILDER→COORDINATOR SIGNAL (see §6)
   │                 │
   │                 ├── (newer equal/ containing target arrives) ──► Superseded  (−1)
   │                 │
   │                 └── (shutdown gate, AdmissionClosed) ─────────► ShutdownDiscard  (−1)
   │
   └─ (reclaim of obligation's handle after recovery world publishes) is the *trigger* for Completed −1
```

**All −1 (obligation leaves live set) paths — currently MISSING entirely:**
1. `Completed` — Builder publishes recovered world + `reclaimSlot(handle)` fires. **No coordinator callback exists today** (transport pop has no completion signal; durable `settle(false)` only clears the *durable slot*, not a logical count).
2. `Superseded` — requires `canSupersede()` (I4 D12.4). **0 production hits**; not implementable in Phase I.
3. `ShutdownDiscard` — `discardRecoveryRequestsOnShutdown()` (cpp:961, transport) + `discardPendingRecoveryAdmission()` (cpp:917, durable) exist; they increment `recoveryShutdownDiscardCount_` but **do not decrement any logical count** (because none exists).

**All +1 (obligation enters live set) paths — currently MISSING:**
- None. There is no `LogicalRecoveryObligation` entity; `pendingIntentCount_` (the only counter) is incremented on *transport reservation* (cpp:850/998), not on logical-obligation creation, and it mixes Observe/Quarantine/Recovery (h:550, INV-ISR-02).

---

## 3. Identity Model (R5 §2, §9) — KEY FINDING

### 3.1 `SemanticRecoveryTarget` seed ALREADY EXISTS in source
`RuntimeBuildSnapshot` (RuntimeBuildTypes.h:48) already carries `RuntimeBuildFingerprint` (h:38-53):
```
struct RuntimeBuildFingerprint {
    uint32_t fingerprintVersion;
    uint64_t irIdentityHash;        // IR domain
    uint64_t convolutionConfigHash; // Conv domain
    uint64_t dspParameterHash;      // EQ domain
    uint64_t sampleRate;
    uint64_t blockSize;
};
// RuntimeBuildSnapshot also has: convolverFingerprint, rebuildFingerprint{...}
```
These are populated at build time (AudioEngine.RebuildDispatch.cpp:95-100). **This is exactly the 5-field set I4 D12.2 asked for** (`irIdentityHash, convolutionConfigHash, convolverFingerprint, dspParameterHash, buildInputHash`). Equality is *already partially implemented* (RuntimeBuildTypes.h:315-336 compares `rebuildFingerprint` fields).

⇒ **`SemanticRecoveryTarget` can be constructed from `RecoveryIntent.buildSource.rebuildFingerprint` with NO new runtime capture.** Only the *struct + named semantics* are missing.

### 3.2 Proposed identity chain
```
RecoveryIntent { handle; buildSource; intentId; epoch }
        │  buildSource.rebuildFingerprint + convolverFingerprint
        ▼
SemanticRecoveryTarget T  = { irIdentityHash, convolutionConfigHash,
                               dspParameterHash, convolverFingerprint, sampleRate }
        │  CoalesceIdentity = (quarantinedHandle, T)
        ▼
CoalesceIdentity C = { handle (excluded DSP), T (resulting-world config) }
        │  one C  ⇔  one LogicalRecoveryObligation
        ▼
LogicalRecoveryObligation  (state machine, §2)
```
- **`handle == same` alone is NOT identity** (R5 §2): `(S, snap_A)` and `(S, snap_B)` differ in `T` ⇒ different `C` ⇒ different obligation. `(S, A)` and `(T, A)` differ in excluded DSP ⇒ different resulting world ⇒ different `C`. ✔ satisfies the R3/R4 result.
- **`RecoveryEpisodeId`** (optional/diagnostic under the direct-invariant form): allocate a monotonic id per *open lineage* of a quarantined `handle` (across quarantine→reclaim→re-quarantine cycles). Needed only if the `E_max × O_max` product form is retained for diagnostics; **not required** for `liveLogicalRecoveryObligationCount ≤ 32`.

### 3.3 Coalesce rule (Phase I, conservative — equality)
```
canCoalesce(existing C, incoming C') = (C == C')   // same excluded DSP AND same target fingerprint
```
If `true`, incoming `AdmissionAttempt` merges into the existing live obligation (no +1, no new transport entry required beyond a re-arm). Distinct `T` ⇒ no coalesce ⇒ new obligation (subject to table cap).

---

## 4. Admission Rule (R5 §7) — backpressure, no overwrite

```
on AdmissionAttempt(handle, buildSource):
    T  = deriveTarget(buildSource)
    C  = {handle, T}
    if exists live obligation O with O.C == C:
        coalesce(O, attempt)          // merge, NO +1, re-arm if needed
        return Coalesced
    if liveLogicalRecoveryObligationCount == 32:
        // Phase I: canSupersede NOT available (I4 D9/D12 NO-GO)
        record recoveryAdmissionRejectedCount_   // INV-5: observable, NOT silent
        return Rejected            // quarantine flag REMAINS; reclaim must not assume recovery done
    // accept
    create LogicalRecoveryObligation(C)   // +1
    liveLogicalRecoveryObligationCount += 1
    enqueue transport / durable as delivery
    return Accepted
```
**Blind overwrite (`pendingRecoveryAdmission_ = newRequest;`) is FORBIDDEN** — it is the INV-X1-7 P0 correctness defect (R4 §7) and is excluded from the capacity solution.

`Full` resolution choice (Phase I): **Reject-with-telemetry** (recommended). `Defer` (bounded pending buffer) is a future option requiring its own bounded resource and is explicitly a *separate* decision.

---

## 5. Ownership Rule (R5 §5, §E)

```
No LogicalRecoveryObligation may leave the live set except:
    Success / Completed   (Builder published + handle reclaimed)
    Superseded           (newer equal/ containing target — Phase II)
    ShutdownDiscard      (AdmissionClosed gate)
```
The current blind-overwrite path violates this (obligation vanishes silently on queue-full). Fixing it is the INV-X1-7 work, **separate from the capacity proof** but **required for the proof's −1 accounting to be sound**.

---

## 6. Missing −1 Event — Completion Authority (R5-rev1, corrected by R6)

**R6 falsified the original candidate** (`recovered world published AND reclaimSlot(handle) fired`). `reclaimSlot(handle)` is a generation-gated flag release (AudioEngine.Commit.cpp:661), recovery-independent, Path-B-affected, and O1/O2-ambiguous — it **cannot** be the −1 event (R6 §2). The real gap:

- A transport `RecoveryIntent` is `popRecoveryRequest()`-ed by the Builder and built/published via `enqueuePublicationIntentForRuntimeCommit` (RebuildDispatch.cpp:999/1072), but **no signal returns to the Coordinator** to mark the obligation `Completed`.
- The Builder reads only `recovery->handle` + `recovery->buildSource`; it passes the **global** `rebuildRequestGeneration` (not `intentId`) to publish. `enqueuePublicationIntentForRuntimeCommit` (AudioEngine.h:2551) carries **no handle, no intentId** → `intentId` is **dropped at the publish boundary** (R6 §3).
- Transport build failures `continue` with **no −1** → obligation leak (R6 §5/§8).

**Required (design, R5-rev1):**
1. **Dedicated Builder→Coordinator completion signal** (single −1 authority):
   ```
   onRecoveryBuildResolved(handle, intentId, status):   // status ∈ {Published, Failed}
       if live obligation O matches (handle, intentId):
           O.state = (status==Published) ? Completed : Failed
           liveLogicalRecoveryObligationCount -= 1
   ```
2. **Plumb `intentId` + `handle`** from `RecoveryIntent` through `enqueuePublicationIntentForRuntimeCommit` (add params) into the commit path, so the signal can be emitted at **PublicationCompleted** (build success AND publish enqueued).
3. **Completion point = PublicationCompleted** of the recovered world for obligation `(handle, intentId)` — **NOT** `reclaimSlot`. `reclaimSlot` remains a physical reclaim event, unrelated to obligation lifecycle.
4. **Build-failure terminal** (`Failed` → −1) so transport build failures do not leak the obligation.
5. **Completion identity = logical obligation id** (surviving id after coalesce), not per-attempt `intentId` (R6 §6).

This closes the +1/−1 pair with a single, identity-carrying authority.

---

## 7. `pendingIntentCount_` Explicitly Excluded (R5 §5)

`pendingIntentCount_` (h:560) is **not** the logical-obligation counter and is excluded from the proof:
- it is "transport residency + producer reservation" (INV-ISR-02, h:78),
- it tracks **Observe/Quarantine/Recovery** (h:550), not recovery obligations alone,
- the durable admission rolls back its reservation (cpp:861) so the durable is uncounted,
- it is used for `isFullyDrained` (shutdown quiescence), not capacity.

Similarly `recoveryIntentQueue_.size()` (256) and `pendingRecoveryAdmission_` (1) are *delivery* states, not the logical count. **All three are excluded** from `liveLogicalRecoveryObligationCount`.

---

## 8. Capacity Derivation Direction (R5 §6)

NOT `queue=256 ⇒ logical=32`. The direction is:
```
logical capacity = 32   (deliberate resource bound, INV-CAP-7)
        │
        ▼  RecoveryAdmissionTable capacity = 32
   L_logical ≤ 32   (enforced invariant)
        │  transport/durable/building are DELIVERY of ≤32 live obligations
        ▼
   L_transport ≤ 256 (ring is a superset delivery buffer; transient attempts
                      map to ≤32 coalesced obligations or are Rejected)
   L_durable   ≤ 1   (or folded into the 32-table)
   L_building  ≤ 1   (single Builder thread; bounded by logical)
   L_stalled   ≤ 32  (retry-stall is a sub-state of live obligations)
```
`A_max` (admission attempts, ≤257) is **decoupled from `Q_max`** (R5 §8): `1 quarantine → 0/1/N admissions` (R4 §1.3), but every admission attempt resolves to ≤1 logical obligation via coalesce, so `L_logical ≤ 32` holds regardless of `Q_max=256` or transport=256.

---

## 9. Proof Obligation (R5 §F, deferred to R5-7)

Target (inductive):
```
∀ reachable states:  liveLogicalRecoveryObligationCount ≤ 32
```
Basis: table starts empty (0). Step: every `+1` is guarded by `count < 32` (§4 admission rule); every live obligation has exactly one terminating −1 (Completed / Superseded / ShutdownDiscard, §2). Therefore count never exceeds 32. **This proof is only possible AFTER the model below is implemented** — which is why R5 stops at the model, not the proof.

---

## 10. Minimal Design Changes to Maintain 32 (R5 stop condition)

Enumerated, **not implemented**:

| # | Change | Purpose | Blocks proof if missing? |
|---|---|---|---|
| 1 | `RecoveryAdmissionTable` (capacity 32) **or** `liveLogicalRecoveryObligationCount` atomic + gate | the bounded resource (INV-CAP-7) | YES |
| 2 | `SemanticRecoveryTarget` struct (seed = `buildSource.rebuildFingerprint`; reuse existing equality h:315-336) | target identity | YES |
| 3 | `CoalesceIdentity = {handle, SemanticRecoveryTarget}` + `canCoalesce()` | coalesce key | YES |
| 4 | `LogicalRecoveryObligation` struct + state enum (Created/Transport/Durable/Building/Completed/Superseded/ShutdownDiscard) | accounting entity | YES |
| 5 | `+1` at obligation creation (post-coalesce), `−1` at Completed/Superseded/ShutdownDiscard | invariant accounting | YES |
| 6 | Replace blind overwrite with coalesce → (Phase II supersede) → **reject-with-telemetry** | backpressure + INV-X1-7 fix | YES |
| 7 | Builder→Coordinator completion signal `onRecoveryBuildResolved(handle, intentId, status)` (R5-rev1) + plumb `intentId`/`handle` through `enqueuePublicationIntentForRuntimeCommit` | the missing −1, identity-carrying (§6) | YES |
| 8 | (Optional) `RecoveryEpisodeId` allocator | lineage/diagnostic; not required for `≤32` | no |
| 9 | Keep `pendingIntentCount_` for `isFullyDrained` only | exclude from proof (§7) | n/a |
| 10 | **Build-failure terminal** (`Failed` → −1) so transport build failures don't leak the obligation | closes transport retry/failure gap (R6 §5/§8) | YES |
| 11 | **Shutdown = per-logical-obligation iteration** of `RecoveryAdmissionTable` for −1; channel drains are side effects only | prevents double-count / Building-leak (R6 §8) | YES |
| 12 | **Do NOT** change queue 256→32, **do NOT** add bare `kMaxLogicalRecoveryObligations=32` constant without #1–#7, **do NOT** use `reclaimSlot` as −1 | avoid "pushing proof target into code" / invalid authority | guardrail |

---

## 11. Open Design Decisions (flagged, not resolved here)

- **D-a** `CoalesceIdentity = (handle, T)` vs `(T only)`. Recommended `(handle, T)` — keeps the *excluded DSP* in identity so `(S,A)≠(T,A)`.
- **D-b** Table-full with no coalesce candidate in Phase I → **Reject** (recommended, telemetry) vs **Defer** (needs separate bounded buffer). 
- **D-c** Adopt direct `liveLogicalRecoveryObligationCount ≤ 32` (recommended, matches INV-CAP-7). Retain `E_max × O_max` only as derived/diagnostic metrics, **not** as the proof shape.
- **D-d** *(RESOLVED by R6 §1/§2)* Exact `Completed` trigger = **PublicationCompleted** of the recovered world for obligation `(handle, intentId)` — i.e., Builder build success **AND** `enqueuePublicationIntentForRuntimeCommit` called. **`reclaimSlot(handle)` is explicitly NOT the trigger** (generation-gated flag release; recovery-independent; Path-B-affected; O1/O2-ambiguous).
- **D-e** *(RESOLVED by R6 §3/§6)* Completion identity = **logical obligation id** (surviving id after coalesce), carried by plumbed `intentId` + `handle` through the publish path. `intentId` is generated and preserved to the Builder but **dropped at the publish boundary today** → must be plumbed (R5-rev1 §6). Builder must NOT directly ± the logical count.

---

## 12. Traceability (verified on source)

| Item | Location |
|---|---|
| `kMaxSlots=256`, `quarantineActiveFlags_[256]` | ISRDSPQuarantine.h:68,72 |
| `recoveryIntentQueue_` cap 256, `pendingRecoveryAdmission_` single slot | ISRRuntimePublicationCoordinator.h:642,667 |
| `RuntimeBuildFingerprint` (irIdentityHash/convolutionConfigHash/dspParameterHash) | RuntimeBuildTypes.h:38-53 |
| fingerprint populated at build | AudioEngine.RebuildDispatch.cpp:95-100 |
| existing fingerprint equality | RuntimeBuildTypes.h:315-336 |
| `submitRecoveryRequest` blind overwrite | ISRRuntimePublicationCoordinator.cpp:864-873 |
| `settlePendingRecoveryAdmission` (retry/complete) | cpp:926-936 |
| `discardPendingRecoveryAdmission` (ShutdownDiscard) | cpp:917-924 |
| `discardRecoveryRequestsOnShutdown` (transport discard) | cpp:961-967 |
| Builder consumes transport / durable | AudioEngine.RebuildDispatch.cpp:939, 1017 |
| Builder publishes recovered world | AudioEngine.RebuildDispatch.cpp:997-1000 |
| `pendingIntentCount_` (mixed, not logical) | ISRRuntimePublicationCoordinator.h:78,550,560 |
| reclaim gate (generation-based) | AudioEngine.Commit.cpp:655-663 |
| `RecoveryEpisodeId`/`CoalesceIdentity`/`SemanticRecoveryTarget` | **0 production hits** (rg/aidex/graphify) |
| `SupersededDiscard` | enum value only, RuntimePublicationState.h:14 (publication ledger, not wired to recovery) |
| INV-CAP-7 (32 = deliberate bound, not upstream-derived) | doc/work88/I4_DESIGN_CONTRACT.md D25 |

### Tools used
- WSL: `rg`/`sed`/`fdfind` (structural trace)
- context-mode MCP (`ctx_batch_execute` + `rtk` compression)
- **AiDex** (`aidex_query` — confirmed fingerprint/identity absence in production)
- **semble** (`RecoveryEpisodeId CoalesceIdentity ...` → only the blind-overwrite site returns, confirming no identity structures)
- **cocoindex** (`liveLogicalRecoveryObligationCount admission table 32` → surfaced **I4 D25 INV-CAP-7**, the design anchor)
- **graphify** (knowledge graph — no `RecoveryEpisodeId`/`CoalesceIdentity`/`SemanticRecoveryTarget` nodes)
- **serena MCP** — attempted; language server timed out (corroboration supplied by the above).

---

## 13. Conclusion

R5 fixes the capacity model so that a future proof is possible:
- **Invariant**: `liveLogicalRecoveryObligationCount ≤ 32` (direct, per INV-CAP-7 — not derived from Q/A/E/O).
- **Identity**: `SemanticRecoveryTarget` is *already captureable* from `buildSource.rebuildFingerprint`; `CoalesceIdentity = {handle, T}` keeps handle≠identity.
- **Accounting**: +1 at logical-obligation creation, −1 at Completed/Superseded/ShutdownDiscard — **all currently absent**; the missing −1 (Builder→Coordinator completion signal) is the pivotal gap.
- **Backpressure**: coalesce → (Phase II supersede) → reject-with-telemetry; **blind overwrite forbidden** (INV-X1-7).
- **Excluded**: `pendingIntentCount_`, queue size, durable slot — delivery states, not the logical count.

**This audit deliberately stops at the model + minimal-change list.** Implementation belongs to R5-8 (after R5-7 proof). No source was modified.
