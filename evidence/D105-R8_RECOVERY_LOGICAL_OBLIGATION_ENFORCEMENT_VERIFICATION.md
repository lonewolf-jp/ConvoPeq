# D105-R8 — Recovery Logical Obligation Enforcement Verification

- **Date**: 2026-08-27
- **Subject**: Read-only re-audit of D105-R5-8 implementation (`RecoveryAdmissionTable<32>` + single completion authority).
- **Mode**: READ-ONLY. No source modified.
- **Tooling**: `rg` / direct file reads against disk. LSP is broken in this environment (include paths unresolved → all "unknown type / file not found" diagnostics are environmental, NOT used for judgement). No build/run possible.
- **Verification basis**: `evidence/D105-R5-8_RECOVERY_LOGICAL_OBLIGATION_ENFORCEMENT_IMPLEMENTATION_PLAN.md` §1–§9 + R7 §11 state model.

---

## 0. Verdict

| Gate | Result | Reason |
|---|---|---|
| Structural Gate | **FAIL** | Source defects in §7 (admission-rejected recovery obligations never resolved → leak), §4 (coalesce key omits handle), §10 (StaleSuperseded/Superseded unimplemented). Plus §2/§3 TOCTOU hardening recommended. |
| Runtime Gate | **BLOCKED** | Cannot compile/run (LSP broken, no build invoked). 256-concurrent stress not executed. |
| **Overall D105-R8-GATE** | **FAIL** (do not PASS) | Per spec instruction *"実装コードがこの表と異なる場合は PASS にせず"* — §7 mapping differs from the required model. |

The **accounting primitive** is sound: exactly one `+1` (`tryInsert`) and one `−1` (`resolve`), coalesce-before-capacity, idempotent CAS, structural `L ≤ 32`. The failures are in **completion-routing coverage** and **identity/enum completeness**, not in the core counter invariant.

---

## 1. Admission / +1 audit — `submitRecoveryRequest` → `tryInsert`

Files: `src/audioengine/ISRRuntimePublicationCoordinator.cpp:812-885`, `:323-338`.

- coalesce (`findByKey` :840) **before** capacity check (`tryInsert` :847) ✅
- `tryInsert` capacity guard (`liveCount_ >= kCapacity` :324) **before** `+1` (:333) ✅
- insert-failure / no-free-slot returns `nullopt` **without** `+1` (:337) ✅
- `RecoveryAttemptId` (`nextRecoveryIntentId_`, transport intent id) is **separate** from `LogicalRecoveryObligationId` (table `nextId_`). No confusion ✅
- `nextLogicalRecoveryObligationId_` removed; id ownership solely the table (single-writer CoordinatorLoop) ✅
- Double-insert: single producer (CoordinatorLoop) invariant documented at `:775/:805` → no concurrent `tryInsert` ✅ (concurrency caveat on `resolve` in §3)

---

## 2. Table internal invariant

File: `src/audioengine/ISRRuntimePublicationCoordinator.h:299-361`.

- Exactly one `+1` (:333) and one `−1` (:345). `liveCount_` touched only there + `load`(:324)/`consume`(:352)/member(:359) ✅
- `resolve` = `compare_exchange_strong(Live→terminal)` → idempotent; 2nd call sees non-Live → `false` (no double −1, no underflow) ✅
- `tryInsert` reuses only `state != Live` slots; id is monotonic `++nextId_` → reused slot gets **new, higher** id → **no ABA by id** for `findById` ✅
- **CAVEAT (see §3)**: `resolve(index)` is index-based and does **not re-validate the id** before CAS.

---

## 3. `resolve()` ordering — key audit

Actual code (`resolveRecoveryObligation` :890-907 → `table.resolve` :342-349):

```
resolveRecoveryObligation(id):
    if id==0 return
    if outcome==Retry return                 // ΔL=0, stays Live
    idx = findById(id)                       // :898  (separate from the CAS)
    if idx==npos return                      // idempotent no-op
    terminal = map(outcome)                  // :901-903
    table.resolve(idx, terminal):            // :904
        CAS(slot[idx].state, Live→terminal)
        on success: L--
```

Order is **`findById → CAS Live→terminal → decrement`** (first form). Consistent with `findByKey`/`findById`/concurrent-resolution/shutdown.

**TOCTOU / ABA hazard (MEDIUM)**: between `findById(id)→idx` (:898) and `resolve(idx)` (:904) there is no id re-check. `tryInsert` (CoordinatorLoop) and `resolve` (RebuildThread **and** ISR via `onPublishCommitted`) can run concurrently. If slot `idx` is terminated and then **reused** (new id) in that window, `resolve(idx)` will CAS the *new* live obligation to terminal — wrong obligation terminated (count stays balanced, but recovery correctness broken). Window is tiny but real under 256-concurrent. **Recommended fix**: validate `slot.id == expectedId` inside `resolve` (pack state+id in one atomic, or re-check id post-CAS). Not a hard `L` invariant break, but a correctness hazard under concurrency.

---

## 4. Coalesce identity — **DISCREPANCY**

Files: `ISRRuntimePublicationCoordinator.h:248-258`, derive at `:834-838`.

- `CoalesceIdentity == SemanticRecoveryTarget{irIdentityHash, convolutionConfigHash, dspParameterHash}` — **handle is NOT part of the key**.
- Spec §4 requires `CoalesceIdentity = quarantinedHandle + SemanticRecoveryTarget` with *"different handle + same target → different obligation"*. Current code **coalesces** different handles with same fingerprint. ✗
- R7 Option-1 overwrite check: coalesce branch (:841-845) refreshes only `intentId`; `slot.handle/buildSource/epoch` are **not** overwritten → existing obligation's `buildSource` preserved ✅ (this part correct).
- **Fix needed**: include `quarantinedHandle` in `CoalesceIdentity` and its `operator==`; derive `cid` from handle + fingerprint at `:834-838`.

---

## 5. Transport / Durable / Building identity preservation

- `intent.obligationId` set at admission (:861) and durable fallback (:882); `takePendingRecoveryAdmission` copies `recoveryObligationId` (:927) ✅
- No `obligationId` copy-omission on any path ✅
- **Residual (pre-existing, NOT introduced by R5-8)**: single-slot durable fallback overwrites `pendingRecoveryAdmission_` (:873 "既存 durable があれば最新で上書き"). If two *distinct* obligations both fall to durable (queue full), the earlier one's slot is clobbered → that obligation's Live slot never resolved until shutdown. `obligationId` is preserved *within* the surviving slot, but a *different* obligation can be lost. R5-8's "no blind logical-layer overwrite" is satisfied (logical table keeps both Live), but the physical durable slot can still drop a distinct obligation. Known limitation.

---

## 6. Publish chain — full trace

`RecoveryIntent.obligationId` (:861/:927) → `enqueuePublicationIntentForRuntimeCommit(...,obligationId)` (`AudioEngine.Commit.cpp:812` → `PublishRequest.recoveryObligationId` `PublicationAdmission.h:26`) → `submitPublishRequest` → `trySubmitImpl(req)` uses `req.recoveryObligationId` in `resolve` calls (:189/:255/:303/:312) → `onPublishCommitted(seqId, recoveryObligationId)` (`RuntimePublishExecutor.h:105` → `Orchestrator.cpp:346`). ✅ chain intact.

- A. `newDSP` is **not** used as identity (only `recoveryObligationId` flows) ✅
- B. `−1` at publish **completion**, not at `submitPublishRequest` enqueue. Route B (`onPublishCommitted`) = commit path ✅. **Nuance (LOW)**: Route A (:312) resolves at `executor_.publish` *success* (async-facade enqueue-complete), not at ISR commit — both idempotent so exactly-once holds, but strictly Route A pre-completes vs "commit".
- C. `recoveryObligationId == 0` → `resolveRecoveryObligation` early-returns (:892) → normal publish lifecycle untouched ✅

---

## 7. FailureReason → RecoveryOutcome — **FAIL (gate-blocking)**

Call sites of `resolveRecoveryObligation`: `RuntimePublicationOrchestrator.cpp:189/255/303/312/346` — cover build/rebuild/publish failure → `Failed`, publish success → `Published`, commit → `Published`.

`submitPublishRequest`'s rejected branches (:363-397: `RejectedStaleGeneration`, `RejectedNotFinalized`, `RejectedPressure`, `RejectedShutdown`) call **no** `resolveRecoveryObligation`. A recovery obligation **rejected at admission is never resolved** → its Live slot stays occupied (leak; consumes 1 of 32 until shutdown).

| FailureReason | Spec expects | Actual |
|---|---|---|
| QueuePressure | Retry (ΔL=0) | **no resolve** → obligation stays Live, recovery intent consumed & not re-driven (effectively stuck) |
| StaleGeneration | StaleSuperseded (−1) | **no resolve** → LEAK (never −1) |
| ValidationFailed | Failed (−1) | **no resolve** → LEAK (never −1) |
| PublishFailed | Failed (−1) | ✅ `:189/:255/:303` |
| ShutdownRejected | ShutdownDiscard (−1) | resolved later by table iteration at shutdown (:1004-1008) — OK, but not at rejection site |
| (commit) | Published (−1) | ✅ `:312/:346` |

Also: `RecoveryOutcome::Retry` is **never produced** by any caller (:896 only consumes it) → QueuePressure→Retry mapping unimplemented. Enum has **no `StaleSuperseded`/`Superseded`** (see §10). Per spec: do **not** PASS.

---

## 8. Retry audit

`resolveRecoveryObligation` returns early for `Retry` (:896-897) → no −1, obligation stays Live, `buildSource`/`coalesceKey` unchanged ✅. **But** Retry is never emitted (§7), so the QueuePressure→durable-rearm path is not actually wired. Single-slot-durable overwrite concern noted in §5.

---

## 9. Shutdown / race audit

`discardRecoveryRequestsOnShutdown` (:994-1009):
1. `popRecoveryRequest()` loop → `recoveryShutdownDiscardCount_++` — **does not touch L** ✅
2. table iteration → `resolveRecoveryObligation(ShutdownDiscarded)` → `table.resolve` (−1, idempotent CAS) ✅
`discardPendingRecoveryAdmission` (:943-951) → `recoveryShutdownDiscardCount_++` only, **no L** ✅

- Races A/B/C: first terminal CAS wins (−1), second sees non-Live → no-op ✅
- **Order note**: code drains transport *before* table resolution (spec suggested table-first). Logically safe (drains don't touch L; table is sole −1 authority) but sequence differs. LOW.

---

## 10. State machine reachability (R7 §11)

`ObligationState` (:260-267): `NoObligation, Live, ResolvedSuccess, ResolvedFailed, ResolvedRetry, ShutdownDiscarded`.

| R7 §11 state | Implemented? | Creation / transition / authority |
|---|---|---|
| Created | implicit (NoObligation→Live) | `tryInsert` |
| Transport | derived (queue push) | `submitRecoveryRequest` |
| Durable | derived (durable fallback) | `submitRecoveryRequest` |
| Building | derived (take/durable→Building) | `takePendingRecoveryAdmission` |
| Completed | ✅ `ResolvedSuccess` | `resolve(Published)` from publish-success/commit |
| Failed | ✅ `ResolvedFailed` | `resolve(Failed)` from build/publish failure |
| StaleSuperseded | ❌ **not implemented** (Phase II) | enum/state absent |
| Superseded | ❌ **not implemented** (Phase II) | absent |
| ShutdownDiscard | ✅ `ShutdownDiscarded` | `resolve(ShutdownDiscarded)` at shutdown |

`ResolvedRetry` state exists but unused (Retry never emitted). **Do not claim StaleSuperseded/Superseded implemented.**

---

## 11. 256-concurrent stress — **RUNTIME GATE = BLOCKED**

Cannot compile/run (LSP broken, no build). Structural expectation: unique obligations ≤ 32, 33rd rejected (`tryInsert`→`nullopt`→`recoveryCapacityExhaustedCount_++`), coalesced attempts ΔL=0. Telemetry `recoveryCoalescedCount_`/`recoveryCapacityExhaustedCount_` wired (:845/:850). **Not executed** → BLOCKED.

---

## 12. Build / test — BLOCKED (environment)

Debug/Release compile, CTest, recovery tests, soak: not runnable here. Source-defect vs environment separation: §7/§4/§10 defects are **source defects** (static, env-independent); the stress test is **environment-blocked**.

---

## 13. Final gate checklist

```
[+] +1 authority exactly one
[+] −1 authority exactly one
[+] coalesce before capacity check
[+] capacity guard before +1
[+] table occupancy == logical live count (no-race)
[+] obligationId preserved through every delivery path
[+] obligationId preserved through publication path
[+] delivery drain does not perform logical accounting
[+] reclaimSlot does not affect L
[+] queue capacity remains 256 (kRecoveryIntentQueueCapacity=256)
[+] kMaxSlots remains 256 (ISRDSPQuarantine.h:68)
[+] pendingIntentCount_ semantics unchanged
[+] shutdown catches Building obligations
[+] terminal resolution exactly-once (per-id; under-race caveat §3)
[~] no blind durable overwrite of LOGICAL layer  (physical single-slot drop still possible, §5)
[~] commit-not-enqueue completes obligation      (Route A pre-completes at facade-success, §6B)
[~] Retry is ΔL=0                                (true, but Retry never produced, §7/§8)
[~] all implemented R7 states reachable          (StaleSuperseded/Superseded = Phase II, §10)
[−] FailureReason mapping complete               (§7 — 3 admission rejections don't resolve; Retry unproduced)
[−] CoalesceIdentity = handle + target           (§4 — handle omitted)
[−] resolve re-validates id (no TOCTOU/ABA)      (§2/§3 — index-based, no id re-check)
[ BLOCKED ] runtime stress confirms L≤32         (§11 — env)
```

---

## 14. Required corrections (NOT applied — R8 is read-only)

1. **§7 (gate-blocking)**: in `submitPublishRequest` rejected branches, call `resolveRecoveryObligation(req.recoveryObligationId, …)` for `recoveryObligationId != 0`:
   - `StaleGeneration → StaleSuperseded` (add `StaleSuperseded` to `RecoveryOutcome` + `ObligationState`)
   - `ValidationFailed → Failed`
   - `QueuePressure → Retry` (and re-arm durable so recovery retried)
   - `ShutdownRejected → ShutdownDiscarded` (idempotent; already handled by shutdown table iteration)
2. **§4**: include `quarantinedHandle` in `CoalesceIdentity` (and `operator==`); derive `cid` from handle + fingerprint.
3. **§2/§3**: make `resolve` validate the slot id (pack state+id atomically, or re-check id post-CAS) to close the TOCTOU.
4. **§10**: explicitly document StaleSuperseded/Superseded as Phase-II (not claimed implemented).

---

## 15. Evidence index (verified locations)

- `src/audioengine/ISRRuntimePublicationCoordinator.h:240-361` — types, `RecoveryAdmissionTable<32>`, single +1/−1.
- `src/audioengine/ISRRuntimePublicationCoordinator.cpp:812-885` — `submitRecoveryRequest` (coalesce → tryInsert → carry obligationId).
- `src/audioengine/ISRRuntimePublicationCoordinator.cpp:890-907` — `resolveRecoveryObligation` (single completion authority).
- `src/audioengine/ISRRuntimePublicationCoordinator.cpp:916-969` — `takePendingRecoveryAdmission` / `settlePendingRecoveryAdmission` (durable lease, ΔL=0 on retry).
- `src/audioengine/ISRRuntimePublicationCoordinator.cpp:994-1009` — `discardRecoveryRequestsOnShutdown` (table-centric, no drain −1).
- `src/audioengine/RuntimePublicationOrchestrator.cpp:40-99` — `trySubmitImpl` admission `evaluate()` (rejection paths do NOT resolve).
- `src/audioengine/RuntimePublicationOrchestrator.cpp:175-347` — build/publish failure → `Failed`; success → `Published`; `onPublishCommitted` → `Published`.
- `src/audioengine/RuntimePublicationOrchestrator.cpp:349-399` — `submitPublishRequest` rejected branches (no `resolveRecoveryObligation`).
- `src/audioengine/AudioEngine.Commit.cpp:782-816` — `enqueuePublicationIntentForRuntimeCommit` carries `recoveryObligationId`.
- `src/audioengine/RuntimePublishExecutor.h:105` — `onPublishCommitted(intent.payload.publish.recoveryObligationId)`.
- `src/audioengine/ISRDSPQuarantine.h:68` — `kMaxSlots = 256` (unchanged).
- `src/audioengine/ISRRuntimePublicationCoordinator.h:787` — `kRecoveryIntentQueueCapacity = 256` (unchanged).
- `src/audioengine/AudioEngine.h:4415-4454` — `submitRecoveryIntent` (CoordinatorLoop context; mutex/cv present → not hard ISR).

---

*D105-R8 stops here. No source modified. Awaiting correction of §7/§4/§10 (and §3 hardening) before re-verification.*
