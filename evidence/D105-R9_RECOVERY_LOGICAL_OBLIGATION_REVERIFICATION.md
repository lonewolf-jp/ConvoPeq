# D105-R9 — Recovery Logical Obligation Enforcement: Read-only Re-verification

**Date:** 2026-08-27
**Type:** Read-only re-audit / verification (source changes = 0)
**Prereq:** D105-R5-9 implementation complete
**GATE VERDICT: FAIL** (source defect R8-#4 present) — runtime rows BLOCKED (pre-existing unrelated build break).

---

## 0. Verdict at a glance

| Gate row | Result |
|----------|--------|
| source structural checks | PASS (except R8-#4) |
| R8 defect matrix | R8-#4 = **FAIL**; all other R8 defects = PASS |
| Debug build | **BLOCKED** — `AudioEngine.Commit.cpp:812` undeclared `recoveryObligationId` (pre-existing, unrelated to R5-9) |
| Release build | BLOCKED (depends on Debug build) |
| CTest | BLOCKED (build infra) |
| C1–C10 | BLOCKED (build infra) |
| 256 concurrent stress | BLOCKED (build infra) |

**Gate rule applied:** "FAIL if any source lifecycle/accounting/identity defect remains." Defect R8-#4 (deferred-obligation re-drive) remains → **GATE = FAIL**. The build BLOCK is environmental (pre-existing break in a non-R5-9 file) and explicitly does **not** mask the source FAIL.

---

## 1. Logical accounting — PASS

`liveCount_` (the L≤32 counter) is touched exactly twice (verified via `rg`, no `liveCount_++/--` anywhere):

- **`+1`** — `RecoveryAdmissionTable::tryInsert` (`ISRRuntimePublicationCoordinator.h:339`). Guarded by capacity check at line 330 (`liveCount_.load() >= kCapacity`) **before** the increment.
- **`−1`** — `RecoveryAdmissionTable::resolve(id, terminal)` (`ISRRuntimePublicationCoordinator.h:357`). Guarded by a CAS (`Live → terminal`) so it runs exactly once per obligation.

- `L == 32` → no unique admission: `tryInsert` returns `nullopt` → `submitRecoveryRequest` returns false (reject), no transport. ✓
- coalesce precedes capacity check: `findByKey` hit → coalesce branch (reuses existing Live obligation, no `tryInsert`) → `ΔL = 0`. ✓

## 2. Coalesce identity — PASS

`CoalesceIdentity` is now (header):
```cpp
struct CoalesceIdentity {
    DSPHandle quarantinedHandle;
    SemanticRecoveryTarget target;   // {irIdentityHash, convolutionConfigHash, dspParameterHash}
};
```
- `(H1,T) + (H1,T)` → same `cid` → `findByKey` hit → coalesce. ✓
- `(H1,T) + (H2,T)` → `quarantinedHandle` differs → distinct `cid` → `tryInsert` new slot → **two** obligations. ✓ (verified by C4 design and code).

## 3. ABA / TOCTOU — PASS

`resolve(std::uint64_t id, ObligationState terminal)` (header):
```
id → re-scan slot by id → CAS only if (slot.obligationId == id && state == Live) → Live→terminal
```
No `findById(id) → cached index → resolve(index)` remains (only a comment mentions the old pattern). `nextId_` is monotonic, so a reused slot gets a new id → a late/duplicate resolve of an old id is a no-op. ✓

## 4. Publication rejection routing — PASS (MUST-1)

`RuntimePublicationOrchestrator::submitPublishRequest` rejection branches route through `resolveIfRecovery(...)` (guarded by `req.recoveryObligationId != 0` so normal publishes are unaffected):

| Rejection | Outcome | Site |
|-----------|---------|------|
| `RejectedStaleGeneration` | `StaleSuperseded` | `RuntimePublicationOrchestrator.cpp:377` |
| `RejectedNotFinalized` / validation | `Failed` | `:386` |
| `RejectedPressure` | `Retry` (ΔL=0) | `:396` |
| `RejectedShutdown` | `ShutdownDiscarded` (idempotent) | `:405` |

`RejectedPublishFailure` needs no second resolve (its obligation was already `Failed` inside `trySubmitImpl`). ✓

**RejectedPressure correctness:** `resolve(Retry)` early-returns (no `−1`, obligation stays Live). Crucially, the obligation's transport intent was **already pushed to `recoveryIntentQueue_`** at `submitRecoveryRequest` time, so it remains deliverable (the Builder will pop and build it). `rearmRecoveryRetry(id)` is additionally called but is a no-op unless the durable slot already holds *this* obligation in `Building` — correct (nothing to re-arm when admission-rejected before durable handoff). ✓

## 5. Single durable slot — PARTIAL (no-clobber PASS, re-drive FAIL)

- **No-clobber (MUST-2 first half) — PASS:** in `submitRecoveryRequest`, if the single durable slot is occupied by a *different* live obligation, the new one is **deferred** (`recoveryRetryDeferredCount_++`, return true) instead of overwriting. The distinct obligation B keeps its only delivery representation. ✓
- **Re-drive (R8-#4) — FAIL:** see §11.

## 6. Completion authority — PASS

All five outcomes (`Published`, `Failed`, `Retry`, `StaleSuperseded`, `ShutdownDiscarded`) flow through the single `resolveRecoveryObligation` → `table.resolve`. No direct `liveCount_--` exists anywhere. Races are idempotent: `Published→Published`, `Failed→ShutdownDiscarded`, `ShutdownDiscarded→Failed` each perform the CAS exactly once (subsequent calls find non-Live → no second `−1`, no L underflow). ✓

## 7. State machine (R7 §11 correspondence)

Implemented transitions: `Created(transport)` → `Durable(DurablePending/Building)` → `Completed(Published)` / `Failed` / `StaleSuperseded` / `ShutdownDiscarded`; plus `Retry` (keeps Live, non-terminal). `Superseded` is **Phase II / not implemented** and was **excluded** from the R8 §14 required corrections (only `StaleSuperseded` was mandated). Its absence does **not** block the Phase-I gate; this fact is recorded here per spec §5.

## 8. Forbidden changes — PASS (verified)

- `kMaxSlots = 256` (`ISRDSPQuarantine.h:68`, `ISRRetireRuntimeEx.h:98`) — these are **DSP quarantine/retire slots**, a different subsystem from the recovery logical obligations; unchanged. ✓
- `kRecoveryIntentQueueCapacity = 256` (`ISRRuntimePublicationCoordinator.h:810`) — unchanged. ✓
- `PendingRecoveryAdmission` = single slot — unchanged. ✓
- `pendingIntentCount_` semantics — unchanged (transport-residency counter; separate from `liveCount_`/`L`). ✓
- `reclaimSlot()` (`ISRDSPQuarantine.cpp:49`) — does **not** touch `liveCount_` (different subsystem). ✓
- `kMaxLogicalRecoveryObligations = 32` (`ISRRuntimePublicationCoordinator.h:306`) is **real and tied**: `RecoveryAdmissionTable<kMaxLogicalRecoveryObligations>` (line 858) + the `tryInsert` capacity guard enforce `L ≤ 32`. ✓

## 9. R8 failure matrix re-judged

| R8 defect | R9 judgment |
|-----------|-------------|
| Publication rejection leak | **PASS** (routing fixed) |
| QueuePressure → Retry unconverted | **PASS** (Retry routes, L unchanged) |
| durable distinct-obligation clobber | **PASS** (no longer overwrites) |
| deferred obligation re-drive | **FAIL** (no re-drive source path — §11) |
| handle omitted from coalesce key | **PASS** |
| resolve TOCTOU/ABA | **PASS** |
| StaleSuperseded absent | **PASS** |
| Superseded | Phase II / N/A (does not block Phase-I gate) |
| runtime stress (256) | BLOCKED (build infra) |
| build / CTest | BLOCKED (pre-existing compile error) |

## 10. Gate

**FAIL.** A source lifecycle defect (R8-#4) remains. Build/CTest/C1–C10/256-stress are BLOCKED by a pre-existing, unrelated compile error (below) and could not be executed; this does not change the source-level FAIL.

## 11. Re-drive chain — the decisive gap (R8-#4 = FAIL)

Per spec §11, a deferred obligation must be provably re-driven to delivery. Trace:

```
O2 deferred (submitRecoveryRequest, queue full + durable slot busy by DIFFERENT obligation B)
   ↓
what HOLDS O2?      → RecoveryAdmissionTable slot (state = Live).  [OK]
what REDISCOVERS?   → NOTHING automatic. recoveryAdmissions_ is read only at
                       submit / resolve / shutdown. No periodic re-scan exists.
what RE-TRANSPORTS? → Only if the SAME identity is re-submitted (coalesce → push to queue).
                       Re-submission occurs only on a NEW quarantine transition of that DSP
                       (QuarantineIntentHandler issues recovery only on stateChanged).
                       ⇒ NOT guaranteed.
reaches BUILD?      → Only via incidental re-submission. Otherwise O2 stays Live & undelivered
                       for the rest of runtime (cleared only at shutdown).
```

The Builder consumer (`AudioEngine.RebuildDispatch.cpp`) only consumes `popRecoveryRequest()` (queue) and `takePendingRecoveryAdmission()` (single durable slot). A deferred obligation has **neither** representation, so it is invisible to the Builder. `recoveryRetryDeferredCount_` is incremented but never read to trigger a re-drive.

Conclusion: R5-9's "non-destructive deferral" claim is **structurally incomplete** — O2 is not clobbered (good) but is also **not guaranteed to be re-driven**. Per spec §3 ("if defer-without-re-drive source path exists → R8 defect residual → FAIL"), this is **FAIL**.

### Required correction (next cycle — NOT applied; R9 is read-only)
Add a re-drive mechanism that routes only through the single Completion Authority, e.g.:
- maintain a deferred-obligation id list set when the durable slot is busy by a different obligation; **or**
- when the durable slot frees (`settlePendingRecoveryAdmission`) and/or the queue drains, re-scan `recoveryAdmissions_` for Live obligations lacking a transport/durable representation and re-enqueue them (re-push to `recoveryIntentQueue_` or re-admit to durable).
No new `−1` paths may be introduced.

## 12. Pre-existing build break (separate from R5-9; noted for infra)

`AudioEngine.Commit.cpp:812`:
```cpp
req.recoveryObligationId = recoveryObligationId;   // ★ D105-R5-8: carry obligation id to completion
```
`recoveryObligationId` is **undeclared** in `enqueuePublicationIntentForRuntimeCommit` (it is not a parameter or local). Compounding errors (C2597/C2352) on `currentBuildSnapshotMutex_`, `pendingLearningMode`, `registerDSPHandleForRuntime`, `enqueueLearningCommand` indicate an `AudioEngine.h`/`.cpp` desync in the working tree (uncommitted changes from prior sessions).

- This file is **not** among the D105-R5-9 edits (R5-9 touched only `ISRRuntimePublicationCoordinator.*`, `RuntimePublicationOrchestrator.cpp`, `PublicationAdmission.h` (comment), and `ISRSemanticValidationTests.cpp`).
- It blocks the entire `ConvoPeq` dependency build, hence CTest/C1–C10/256-stress cannot run.
- It is a genuine repo defect but **outside** the R5-9 recovery-enforcement scope and outside R9's read-only mandate; track and fix separately.
- Its existence does **not** affect the R8-#4 source verdict above (which is in `ISRRuntimePublicationCoordinator.cpp`).

---

## Conclusion

**D105-R9 = FAIL.** All R8 defects except #4 are closed by the D105-R5-9 source corrections (rejection routing, handle-bearing coalesce, ABA-safe id-based resolve, StaleSuperseded, no durable clobber). However, the deferred-obligation re-drive (R8-#4) has **no source path** and must be implemented before the gate can pass. Runtime verification (Debug/Release build, CTest, C1–C10, 256 stress) is **BLOCKED** by a pre-existing, unrelated compile error in `AudioEngine.Commit.cpp` and could not be executed in this session; this BLOCK does not mask the source-level FAIL.
