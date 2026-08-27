# D105-R4 — Upstream Recovery Admission Source Bound Audit

**Type**: Read-only structural audit / 0 source changes
**Purpose**: Confirm whether an *implemented upstream invariant* exists that can force `E_max × O_max ≤ 32`. Trace the quarantine→recovery-admission path end-to-end and separate `Q_max / A_max / E_max / O_max / L_*` with the "logical obligation count vs buffer residency" distinction kept explicit.
**Date**: 2026-08-27
**Status**: ✅ Complete (read-only)

---

## 0. Executive Summary

| Metric | R3 claim | R4 verdict (re-verified on source) | Grounding |
|---|---|---|---|
| `Q_max` | 256 | **256** | `kMaxSlots = 256` (ISRDSPQuarantine.h:68); `quarantineActiveFlags_` is `array<atomic<bool>,256>` (h:72) |
| `A_max` | — | **≤ 257 emergent (no 1:1 with Q)** | 1 active quarantine window can emit **N** recovery intents (see §2) |
| `E_max` | undefined | **undefined** | `RecoveryEpisodeId` 0 production hits |
| `O_max` | ≥2 unbounded | **unbounded (no per-lineage code cap; ≥2 concrete)** | see §3 |
| `L_transport_max` | — | **256** | `kRecoveryIntentQueueCapacity = 256` (h:642) |
| `L_durable_max` | — | **1** | single `pendingRecoveryAdmission_` (h:667) |
| `L_building_max` | — | **1 (durable Building state only; no counter)** | no `buildingCount` exists |
| `L_stalled_max` | — | **0 / N-A (no recovery-stalled counter)** | `RuntimePolicyEngine::stalledCount`(max 3) is retry-stall, unrelated |
| `L_storage_max` | 257 | **257** | 256 transport + 1 durable |
| `kMaxLogicalRecoveryObligations` | impossible | **IMPOSSIBLE to derive** | the 4 D14 counters do not exist; `pendingIntentCount_` is not a logical-obligation count |
| `E_max × O_max ≤ 32` | NO-GO | **NO-GO** | E undefined, O unbounded → product undefined, cannot be ≤32 |
| `INV-X1-7` | FAIL | **FAIL** (independent P0 correctness defect) | unconditional durable overwrite, no identity check |
| 1 quarantine → 1 episode | not enforced | **impossible to prove** | no `RecoveryEpisodeId`; 1→N admission is constructible |
| same handle + changed snapshot coexistence | possible | **POSSIBLE** | generation-based reclaim enables re-quarantine w/ new `buildSource` |

**Judgment: C — finite *logical* bound cannot be proven from current code.** The only finite bound provable is the 257 buffer-residency cap (transport+durable), which (a) is not ≤32, (b) conflates residency with logical obligations (forbidden by INV-ISR-02/03), and (c) does not bound `O_max` per lineage. The D18/I4 upstream invariants (`RecoveryEpisodeId`, `CoalesceIdentity`, `SemanticRecoveryTarget`, 4-count accounting, bounded admission table, backpressure) are 0 production hits. **Phase I implementation stays NO-GO**; capacity invariant must be re-fixed as *design* before D105-R5.

---

## 1. Simultaneous Quarantine Source — Real Upper Bound

### 1.1 Structural bound
- `DSPQuarantineManager::quarantineHandle` (ISRDSPQuarantine.cpp:17-47) guards with `alreadyActive = consumeAtomic(quarantineActiveFlags_[slot])` (line 24-25); if already active it returns `false` (line 26-27). So a **slot cannot be re-quarantined while its flag is set** — but **different slots are fully independent**.
- `kMaxSlots = 256` (ISRDSPQuarantine.h:68); `quarantineActiveFlags_` is `std::array<std::atomic<bool>, kMaxSlots>` (h:72).
- Therefore `Q_max = 256`: all 256 slots *can* be simultaneously quarantined.

### 1.2 Two distinct quarantine producers (critical new finding vs R3)
There are **two** quarantine paths, and only one of them issues a recovery admission:

- **Path A (recovery-generating)** — `AudioEngine.Timer.cpp:1906 / 1944` → `runtimePublicationBridge_.submitQuarantine(...)` → `intentQueue_.push(QuarantineIntent)` (ISRRuntimePublicationCoordinator.cpp:969-1002) → **`QuarantineIntentHandler::handle`** (ProcessIntent.cpp:111) → `QuarantineService::executeQuarantine` (sets flag) + `if (stateChanged) ctx.engine.submitRecoveryIntent(...)` (ProcessIntent.cpp:137-140).
- **Path B (flag only, NO recovery intent)** — `AudioEngine::quarantineSlot` (AudioEngine.Threading.cpp:38-72) calls `dspQuarantineManager_.quarantineHandle(...)` **directly** (line 44) and **never calls `submitRecoveryIntent`**. Callers: `AudioEngine.Commit.cpp:605, 625` (`RetireDeferralTimeout`). This path sets the active flag and retires the DSP but **produces zero recovery admissions**.

⇒ **Not every quarantine produces a recovery admission.** A `Q_max=256` quarantine set could contain Path-B slots with no admission at all. So `A_max ≤ Q_max` is *not* guaranteed in either direction by structure.

### 1.3 Can a single slot generate multiple admissions while quarantined?
**Yes — via misuse of `stateChanged`.** `QuarantineService::executeQuarantine` (ISRRuntimePublicationCoordinator.cpp:772-796):
```cpp
if (!request.handle.isNull() && request.handle.slot > 0) {
    handleRuntime.quarantine(request.handle);
    result.stateChanged = true;        // ← hardcoded true, NOT derived from flag transition
}
const bool auditLogged = quarantineManager.quarantineHandle(...);  // alreadyActive→false sets this false
result.auditLogged = auditLogged;      // ← written, but NEVER READ anywhere (only struct field, h:53)
```
`auditLogged` (the real "was this a *new* quarantine?" signal from `alreadyActive`) is computed at cpp:787-791 but consumed nowhere. The handler gates recovery on `stateChanged` (ProcessIntent.cpp:137), which is **always true**. `submitQuarantine` (Path A enqueue, cpp:969) has **no already-active dedupe**. Hence: repeated `submitQuarantine` calls for an already-active slot each enqueue a `QuarantineIntent`; each is processed with `stateChanged==true`; each calls `submitRecoveryIntent` → a **distinct recovery intent** with the *current* `buildSource` (see §3).

⇒ **1 active-quarantine window → N recovery admissions is constructible.** The "256 slots ⇒ 256 concurrent recovery episodes" intuition is doubly wrong: (i) some quarantines (Path B) emit no admission; (ii) a single slot can emit many. The only thing that stops unbounded intake is the **fixed buffer size** (§3), not any semantic invariant.

### 1.4 Reclaim does NOT wait for recovery completion
`reclaimSlot` (ISRDSPQuarantine.cpp:49) clears `quarantineActiveFlags_[slot]`. Its only caller is the PR-style loop in `AudioEngine.Commit.cpp:655-663`:
```cpp
const bool graceCompleted = worldAuthority_.lifetime().isGracePeriodCompleted(
    world->generation, maxObservedGeneration, callbackActiveCount);
if (!graceCompleted) continue;
... dspQuarantineManager_.reclaimSlot(qslot, 0);   // flag cleared
```
Reclaim is gated on **world-generation observation (`maxObservedGeneration`)**, *not* on whether the recovery intent for that slot was consumed/built/published. So a slot can be re-quarantined (flag cleared) **while its old recovery obligation is still alive in the transport queue** (Builder slow). This is the R3 counterexample, now confirmed at the reclaim-gate level.

---

## 2. `Q_max → A_max` conversion (per the R4 spec)

| Step | Code fact | Consequence |
|---|---|---|
| quarantine event | `QuarantineIntentHandler::handle` calls `submitRecoveryIntent` **once** per processed `QuarantineIntent` | one event → at most one `submitRecoveryRequest` call |
| `submitRecoveryRequest` (cpp:812-875) | `recoveryIntentQueue_.push(intent)` (851) **OR** (on full) overwrite `pendingRecoveryAdmission_` (864-873) — **mutually exclusive** | one event → **at most 1 admission** (transport *or* durable) |
| BUT repeated `submitQuarantine` (Path A) of an already-active slot | `stateChanged` hardcoded true (cpp:784-786); handler ignores `auditLogged`; no enqueue dedupe | one *active window* → **N admissions** |
| Path B quarantine | `quarantineSlot` sets flag, no `submitRecoveryIntent` | one event → **0 admissions** |
| reclaim(generation-based) + re-quarantine | flag cleared while old intent still in queue | old + new admissions for same handle **coexist** |

**Conclusion**: `1 quarantine → 1 recovery admission` is **NOT proven**. The honest statement is:
- Clean-path single event → ≤1 admission.
- Real code → `1 → 0` (Path B) **and** `1 → N` (Path A repeated) are both reachable.
- `A_max` therefore has **no clean 1:1 relationship with `Q_max`**. The only global ceiling is the fixed buffers: `A_max ≤ 256 (transport) + 1 (durable) = 257`, an **emergent residency cap**, not a semantic admission bound.

---

## 3. `O_max` ("unbounded") — formal proof

### 3.1 Is there a code upper bound on distinct targets per lineage?
Search for any per-lineage / per-handle cap on recovery intents: **none exists.**
- `recoveryIntentQueue_` is a fixed `LockFreeRingBuffer<RecoveryIntent,256>` (SPSC). It holds *up to 256 independent intents* with **no identity grouping, no coalesce, no per-handle count**.
- `pendingRecoveryAdmission_` is a single slot, unconditionally overwritten.
- No `RecoveryEpisodeId`, `CoalesceIdentity`, `SemanticRecoveryTarget` (0 production hits — confirmed by graphify knowledge graph: no such nodes).

### 3.2 Why `O_max ≥ 2` (concrete, same handle)
Two independent mechanisms produce distinct targets for the *same* handle:

**(a) Repeated submitQuarantine while flag active.**
`currentBuildSnapshot_` has exactly one writer `enqueuePublicationIntentForRuntimeCommit` (AudioEngine.Commit.cpp:799, on **RebuildThread**), guarded by `currentBuildSnapshotMutex_`. Between two processed `QuarantineIntent`s for slot S, the writer can change the snapshot. Each `submitRecoveryIntent(S, getCurrentBuildSnapshotForRecovery())` (ProcessIntent.cpp:139) captures the **then-current** value copy. ⇒ intent_A(S, snap_A) and intent_B(S, snap_B) both enqueued → 2 distinct targets for S.

**(b) reclaim(generation-based) → re-quarantine cycle.**
S quarantined → intent_A(snap_A) in queue (Builder slow). New world published → `maxObservedGeneration` advances → `isGracePeriodCompleted` true → `reclaimSlot(S)` clears flag (Commit.cpp:655-663) **even though intent_A is unconsumed**. S re-quarantined → intent_B(snap_B) pushed. Queue now holds intent_A(S,snap_A) **and** intent_B(S,snap_B).

Both satisfy `currentBuildSnapshot_` being single-valued *at each instant* — they exploit **temporal drift between quarantines**, exactly the user's critique of R2.

### 3.3 `O_max` classification
- `O_max ≥ 2` — concrete counterexample (§3.2).
- `O_max ≤ 256`? — transiently a single handle could flood the 256 transport slots (mechanism (a)) + 1 durable = **257** same-handle intents. This is the **buffer residency** ceiling, *not* a logical bound.
- `O_max ≤ 257`? — only as emergent buffer capacity. There is **no code invariant** establishing even this; it is a side-effect of ring-buffer sizing.
- `O_max unbounded`? — **Yes, in the logical sense the design requires.** Nothing in production code bounds the number of distinct targets a single lineage can accumulate across cycles; only the fixed buffer caps *simultaneous* residency. Per R4's explicit instruction to keep "logical obligation count" distinct from "buffer residency", `O_max` is **unbounded**.

---

## 4. `L_max = 257` stricter decomposition (I4 D14 invariant)

I4 D14 invariant under audit:
```
transportCount + durableCount + buildingCount + stalledCount ≤ kMaxLogicalRecoveryObligations
```

| Sub-count | Production entity | R4 value | Note |
|---|---|---|---|
| `L_transport_max` | `recoveryIntentQueue_` | **256** | `kRecoveryIntentQueueCapacity=256` (h:642) |
| `L_durable_max` | `pendingRecoveryAdmission_` | **1** | single SPSC slot (h:667) |
| `L_building_max` | durable `Building` state only | **1** | **no `buildingCount` counter**; transport intents being built are untracked |
| `L_stalled_max` | — | **0 / N-A** | **no recovery-stalled counter**; `RuntimePolicyEngine::stalledCount`(max 3, RuntimePolicyEngine.cpp:190-207/HealthMonitor.cpp:89-94) is retry-stall, unrelated to recovery obligations |
| `L_storage_max` | 256 + 1 | **257** | in-memory residency only |

### 4.1 Is `pendingIntentCount_` the "logical obligation count"? — **NO**
`pendingIntentCount_` (ISRRuntimePublicationCoordinator.h:560) is explicitly documented (h:78, INV-ISR-02):
> *"pendingIntentCount_ は queue size ではなく transport residency + producer reservation である（residency + reservation — 二重計上禁止）"*
and (h:550): tracks **Observe / Quarantine / Recovery** intents — i.e. *all* intent types, **not** recovery obligations alone.

Further: in `submitRecoveryRequest` the durable path **rolls back** the reservation — `fetchAdd` at cpp:850, `fetchSub` at cpp:861 on queue-full, then writes `pendingRecoveryAdmission_` **without** re-adding. So the durable admission is **not counted** in `pendingIntentCount_` at all. `pendingIntentCount_` is used for `isFullyDrained` (shutdown quiescence), **not** for bounding recovery obligations.

**Therefore the left-hand side of the I4 D14 inequality cannot be computed from current code.** Of the four required counters, **zero exist**. `kMaxLogicalRecoveryObligations` is **not derivable**.

---

## 5. Durable overwrite = independent P0 correctness defect (separate from capacity)

`submitRecoveryRequest` (cpp:864-873) unconditionally overwrites `pendingRecoveryAdmission_` on queue-full:
```cpp
pendingRecoveryAdmission_.state = DurablePending;
pendingRecoveryAdmission_.buildSource = buildSource;      // unconditional
pendingRecoveryAdmission_.handle = quarantinedHandle;     // unconditional
...
```
No `CoalesceIdentity` / `LogicalRecoveryIdentity` / `SemanticRecoveryTarget` comparison. This is an **ownership-conservation violation (INV-X1-7, FAIL)** — classification per D105 scope:

| Bucket | Contents |
|---|---|
| D105 Capacity | `O_max / E_max / Q_max / L_*` (this audit) |
| D105 Correctness | `INV-X1-7` violation, ownership loss, non-supersedable obligation eviction |

This matches I4 D15.2 (ownership conservation). It must **not** be folded into the capacity argument; it is a separate P0 defect to be fixed when `CoalesceIdentity`/`canSupersede` are implemented.

---

## 6. Final Table (R4 deliverable)

| 項目 | R4 で証明する値 |
|---|---:|
| `Q_max` | 256 |
| `A_max` | ≤ 257 (emergent buffer residency; **no 1:1 with Q**, 1→N reachable) |
| `E_max` | undefined (`RecoveryEpisodeId` 0 hits) |
| `O_max` | unbounded (≥2 concrete; only emergent 257 residency cap) |
| `L_transport_max` | 256 |
| `L_durable_max` | 1 |
| `L_building_max` | 1 (durable Building state only; no counter) |
| `L_stalled_max` | 0 / N-A (no recovery-stalled counter) |
| `L_storage_max` | 257 |
| `kMaxLogicalRecoveryObligations` | **導出不可能** (4 D14 カウンタ不在; `pendingIntentCount_` は論理 obligation カウントでない) |
| `E_max × O_max ≤ 32` | **NO-GO** |
| `INV-X1-7` | **FAIL** (独立 P0 correctness defect) |
| 1 quarantine → 1 episode | **証明不可能** |
| same handle + changed snapshot | **coexistence 可能** |

---

## 7. Judgment — **C**

**C. 有限 bound 自体を現行コードから証明不能 → Phase I implementation 継続 NO-GO.**

Rationale:
1. The specific target `E_max × O_max ≤ 32` is unprovable: `E_max` is undefined (no `RecoveryEpisodeId`), `O_max` is unbounded (no per-lineage code cap).
2. A *different, smaller finite bound* (choice B) also cannot be honestly presented: the only finite figure provable is the **257 buffer-residency** cap, which the design itself forbids using as `kMaxLogicalRecoveryObligations` (INV-ISR-02/03: residency ≠ logical obligation; `pendingIntentCount_` mixes Observe/Quarantine/Recovery and excludes durable).
3. The upstream invariants the proof would rest on — `RecoveryEpisodeId`, `CoalesceIdentity`, `SemanticRecoveryTarget`, 4-count accounting, bounded admission table, backpressure — are **0 production hits** (confirmed: rg/grep 0, aidex 0, graphify graph has no such nodes, R3/D103 audits).

**Before D105-R5 (mathematical capacity proof) the capacity invariant must be re-fixed as design**: implement `RecoveryEpisodeId` + `CoalesceIdentity` + bounded admission table + backpressure, *then* derive `kMaxLogicalRecoveryObligations` from real code. Until then, `32` remains an unproven design goal, not a code-proven capacity invariant.

---

## 8. Traceability (verified on source)

| Artifact | Location |
|---|---|
| `kMaxSlots` / `quarantineActiveFlags_` | `src/audioengine/ISRDSPQuarantine.h:68,72` |
| `quarantineHandle` + `alreadyActive` guard | `src/audioengine/ISRDSPQuarantine.cpp:17-47` |
| `reclaimSlot` (flag clear) | `src/audioengine/ISRDSPQuarantine.cpp:49` |
| reclaim gate (generation-based, not recovery-gated) | `src/audioengine/AudioEngine.Commit.cpp:655-663` |
| Path A producer | `src/audioengine/AudioEngine.Timer.cpp:1906,1944` |
| Path B producer (flag only, no recovery) | `src/audioengine/AudioEngine.Threading.cpp:38-72` (caller `AudioEngine.Commit.cpp:605,625`) |
| `executeQuarantine` (stateChanged hardcoded) | `src/audioengine/ISRRuntimePublicationCoordinator.cpp:772-796` |
| handler gates on `stateChanged`, ignores `auditLogged` | `src/audioengine/ISRRuntimePublicationCoordinator_ProcessIntent.cpp:137` |
| `submitRecoveryRequest` (push / durable overwrite) | `src/audioengine/ISRRuntimePublicationCoordinator.cpp:812-875` |
| `RecoveryIntent` / `PendingRecoveryAdmission` | `src/audioengine/ISRRuntimePublicationCoordinator.h:217,667` |
| `kRecoveryIntentQueueCapacity=256` | `src/audioengine/ISRRuntimePublicationCoordinator.h:642` |
| `pendingIntentCount_` (residency+reservation, not obligation) | `src/audioengine/ISRRuntimePublicationCoordinator.h:78,550,560` |
| `currentBuildSnapshot_` single writer (RebuildThread) | `src/audioengine/AudioEngine.Commit.cpp:799` |
| `RecoveryEpisodeId`/`CoalesceIdentity`/`SemanticRecoveryTarget` | **0 production hits** (rg/grep/aidex/graphify) |

### Tools used for cross-validation
- WSL: `rg`/`sed`/`fdfind` (structural trace)
- context-mode MCP (`ctx_batch_execute` + `rtk` compression) — parallel extraction
- **AiDex** (`aidex_query` `pendingIntentCount_` → 42 hits, confirms sole counter, mixed Observe/Quarantine/Recovery usage)
- **semble** (`submitRecoveryRequest recovery intent` → confirms queue/durable path)
- **cocoindex** (`pendingRecoveryAdmission_ overwrite` → confirms blind overwrite evidence)
- **graphify** (knowledge graph: no `RecoveryEpisodeId`/`CoalesceIdentity`/`SemanticRecoveryTarget` nodes → 0 production hits corroborated)
- **serena MCP** — attempted; language server timed out (unresponsive) — corroboration supplied by the above.
