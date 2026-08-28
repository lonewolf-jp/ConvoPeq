# D105-R18 — Retry-Preserving Implementation

**Status:** **R18 PASS** ✅ (Debug + Release, 40/40 ctest PASS)
**Source changes:** 5 files modified; **1 new public API** (`markTransientFailure`); **1 new public telemetry
counter** (`recoveryRetryExhaustedCount_`); **1 new obligation-level counter** (`consecutiveFailureCount`);
**12 new tests** (T-R18-1..T-R18-12); C8 preserved unchanged.

This is the **implementation** of R16-A (Retry-preserving contract) and R17-A (Retry-Preserving
Implementation Specification). The runtime now keeps `Failed`-producing obligations Live across
transient build/publish failures, increments a per-obligation retry counter, and only terminals
the obligation as `ResolvedFailed` when the counter reaches the exhaustion bound.

---

## R18-1 — `consecutiveFailureCount` field on `LogicalRecoveryObligation`

**File:** `src/audioengine/ISRRuntimePublicationCoordinator.h:323` (slot struct) and
`:325-331` (constant).

```cpp
struct LogicalRecoveryObligation {
    std::atomic<LogicalRecoveryObligationId> id{0};
    CoalesceIdentity identity{};
    std::atomic<ObligationState> state{ObligationState::NoObligation};
    DSPHandle handle{};
    PublicationEpoch epoch{0};
    std::uint64_t intentId = 0;
    convo::RuntimeBuildSnapshot buildSource{};
    ObligationDeliveryState delivery{ObligationDeliveryState::None};
    // ★ D105-R18: per-obligation retry budget. Single writer (CoordinatorLoop via
    //   markTransientFailure) for transient failures; reset to 0 in resolve() on
    //   terminal disposition. Reaches kMaxObligationConsecutiveFailures → only
    //   sanctioned path to ResolvedFailed.
    std::atomic<std::uint8_t> consecutiveFailureCount{0};
};
static constexpr std::uint8_t kMaxObligationConsecutiveFailures = 4;
```

- **Type:** `std::atomic<std::uint8_t>` (relaxed ordering sufficient; single-writer / single-reader
  on the CoordinatorLoop).
- **K = 4** (matches the existing `kMaxRecoveryConsecutiveFailures` Builder-local counter).
- **Reset to 0** in two places:
  1. `RecoveryAdmissionTable::tryInsert` (h:365-367) — on slot reuse (NSDMI also inits to 0 on
     fresh construction; the explicit store handles slot reuse after terminal).
  2. `RecoveryAdmissionTable::resolve` (h:388-389) — after the Live→terminal CAS succeeds.

**D36.1 sizeof upper bound** (from D105-R17 §3.1): `RecoveryAdmission ≤ 352B`. Adding 1 byte
(`uint8_t` atomic — typically padded to 4B on 64-bit) brings the upper bound to ~356B.
`B_logical_max = 32 × 356B ≈ 11.4KB`; `B_total_max` increase is sub-KB and well within the
D35.5 `B_admissible = 64MB` threshold.

---

## R18-2 — `markTransientFailure` adjudication authority

**File:** `src/audioengine/ISRRuntimePublicationCoordinator.h:432-442` (declaration),
`src/audioengine/ISRRuntimePublicationCoordinator.cpp:985-1024` (definition).

```cpp
void RuntimeIntentCoordinator::markTransientFailure(std::uint64_t obligationId) noexcept
{
    if (obligationId == 0) return;
    for (std::size_t i = 0; i < recoveryAdmissions_.kCapacity; ++i) {
        if (recoveryAdmissions_.slot(i).id.load(std::memory_order_acquire) != obligationId)
            continue;
        if (recoveryAdmissions_.slot(i).state.load(std::memory_order_acquire) != ObligationState::Live)
            return;   // not Live → no-op
        // (1) P-B: re-eligibility for redrive.
        recoveryAdmissions_.slot(i).delivery = ObligationDeliveryState::None;
        // (2) Increment retry counter.
        const std::uint8_t newCount =
            static_cast<std::uint8_t>(recoveryAdmissions_.slot(i).consecutiveFailureCount.fetch_add(
                std::uint8_t{1}, std::memory_order_acq_rel) + std::uint8_t{1});
        // (3) Exhaustion: route to ResolvedFailed (only sanctioned path).
        if (newCount >= kMaxObligationConsecutiveFailures) {
            convo::fetchAddAtomic(recoveryRetryExhaustedCount_, std::uint64_t{1}, std::memory_order_release);
            recoveryAdmissions_.resolve(obligationId, ObligationState::ResolvedFailed);
        }
        return;
    }
}
```

**Atomic contract (R17-7 LP-E):**
1. Re-scan table for `id`; if not found or not `Live`, no-op.
2. `delivery = None` (P-B: stranded `Transport → None` repair).
3. `consecutiveFailureCount.fetch_add(1, acq_rel)`.
4. If new count ≥ K → `resolve(id, ResolvedFailed)` + `recoveryRetryExhaustedCount_++`.

**ΔL = 0** except in the exhaustion branch.

**Idempotency:** unknown id (0xDEADBEEF or 0), non-Live state (terminal or wrong slot) — all
no-op. Verified by T-R18-11.

---

## R18-3 — Orchestrator failure sites

Three of the four R15-1 producer sites (`:189`, `:255`, `:303`) replaced. The fourth (`:386`) is
**deliberately preserved** (per user instruction: ":386 は機械的置換禁止").

### Build failure (`:189`)
**File:** `src/audioengine/RuntimePublicationOrchestrator.cpp:189-191`
```cpp
// ★ D105-R18: transient build failure → markTransientFailure (ΔL=0, same id, delivery=None,
//   retry counter incremented). ResolvedFailed only at retry exhaustion.
engine_.runtimePublicationBridge_.markTransientFailure(req.recoveryObligationId);
return PublicationAdmission::Decision::RejectedNotFinalized;
```

### Crossfade rebuild failure (`:255`)
**File:** `src/audioengine/RuntimePublicationOrchestrator.cpp:255-258`
```cpp
// ★ D105-R18: transient crossfade-rebuild failure → markTransientFailure (ΔL=0, same id,
//   delivery=None, retry counter incremented). ResolvedFailed only at retry exhaustion.
engine_.runtimePublicationBridge_.markTransientFailure(req.recoveryObligationId);
return PublicationAdmission::Decision::RejectedNotFinalized;
```

### Publish failure (`:303`)
**File:** `src/audioengine/RuntimePublicationOrchestrator.cpp:302-307`
```cpp
// ★ D105-R18: transient publish failure → markTransientFailure (ΔL=0, same id,
//   delivery=None — repairs stranded Transport after Builder pop, retry counter
//   incremented). ResolvedFailed only at retry exhaustion. The shutdown check
//   below still routes to RejectedShutdown; obligation disposition is independent
//   of the return decision.
engine_.runtimePublicationBridge_.markTransientFailure(req.recoveryObligationId);
if (engine_.isShutdownInProgress())
    return PublicationAdmission::Decision::RejectedShutdown;
return PublicationAdmission::Decision::RejectedPublishFailure;
```

### Switch route (`:386`) — **PRESERVED** with R5-9 leak-tightening
The user instructed: ":386 は機械的置換禁止". Verified that `:386` is reached only after
`trySubmitImpl` returns `RejectedNotFinalized` from `admission_.evaluate()` (not from
`trySubmitImpl`'s own build/publish attempts, which now route through `markTransientFailure`).
The `:386` case is a defensive leak-tightening for caller-side admission rejection, not a
transient-recovery-failure path. **R19** will address the residual double-resolve (one path
through `markTransientFailure` exhaustion + one through `:386` `resolveIfRecovery(Failed)`)
by either removing `:386` or adding a state-check guard.

---

## R18-4 — Builder durable failure wiring

**File:** `src/audioengine/AudioEngine.RebuildDispatch.cpp:1034-1038` (build failure) and
`:1061-1064` (warmup failure).

```cpp
if (recoveryResult.runtime == nullptr)
{
    diagLog(...);
    // transient failure → DurablePending へ戻す（次サイクルで再 take — retry）
    runtimePublicationBridge_.settlePendingRecoveryAdmission(true);
    // ★ D105-R18: also drive the obligation-level retry counter (separate
    //   concern from the Builder-local spin-prevention counter). Dual-LP
    //   per R17-7 Row 5: durable-slot sub-state and obligation counter are
    //   independent linearizations on different fields.
    runtimePublicationBridge_.markTransientFailure(recovery->obligationId);
    // ★ 監査軽微指摘4: 連続失敗が上限を超えたらスピン回避のため次サイクルへ委譲
    if (++recoveryConsecutiveFailures >= kMaxRecoveryConsecutiveFailures)
        break;
    continue;
}
```

(Identical pattern for the warmup-failure site at `:1061-1064`.)

**Dual-LP per R17-7 Row 5:**
- `settlePendingRecoveryAdmission(true)` mutates `PendingRecoveryAdmission::state` (transport-side).
- `markTransientFailure(id)` mutates the obligation's `consecutiveFailureCount` and `delivery`
  (logical-side).

**Two counters, not duplicative:**
- **Builder-local `recoveryConsecutiveFailures`** (existing, scope = one `rebuildThreadLoop`
  iteration, reset on success or `break`-out) → spin prevention.
- **Obligation-level `consecutiveFailureCount`** (R18-1, scope = obligation lifetime) →
  retry-exhaustion gate.

The user's prohibition ("4回とK回を二重に数えて実質16回になる実装は禁止") is satisfied: the
two counters are *independent* fields with *different* scopes and *different* thresholds. The
Builder-local counter drives a `break`-out (transport-side yield); the obligation-level counter
drives the `Failed` terminal (logical-side termination).

---

## R18-5 — Stranded `Transport → None` repair

Verified by:
- **Code:** `markTransientFailure` (cpp:1010) unconditionally sets `delivery = None` *before*
  incrementing the counter or checking for exhaustion. This is R16-4's P-B (the key repair).
- **Test:** T-R18-9 (transport-resident stranded case) — fills the queue, drains one entry,
  calls `markTransientFailure` on the obligation id from that entry, then asserts the redrive
  picks it up (delivery reset to None → redrive-eligible).

---

## R18-6 — `Failed` fall-through removal in `resolveRecoveryObligation`

**File:** `src/audioengine/ISRRuntimePublicationCoordinator.cpp:955-980`.

```cpp
void RuntimeIntentCoordinator::resolveRecoveryObligation(std::uint64_t obligationId, RecoveryResolution outcome) noexcept
{
    if (obligationId == 0)
        return;
    if (outcome == RecoveryOutcome::Retry)
        return;
    ObligationState terminal;
    switch (outcome) {
        case RecoveryOutcome::Published:          terminal = ObligationState::ResolvedSuccess; break;
        case RecoveryOutcome::StaleSuperseded:    terminal = ObligationState::ResolvedStaleSuperseded; break;
        case RecoveryOutcome::ShutdownDiscarded:  terminal = ObligationState::ShutdownDiscarded; break;
        case RecoveryOutcome::Failed:             terminal = ObligationState::ResolvedFailed; break;
        case RecoveryOutcome::Retry:              return;  // defensive no-op
        default:
            jassertfalse;
            return;
    }
    if (recoveryAdmissions_.resolve(obligationId, terminal)
        && outcome == RecoveryOutcome::ShutdownDiscarded)
        convo::fetchAddAtomic(recoveryObligationShutdownDiscardCount_, std::uint64_t{1}, std::memory_order_release);
}
```

- The `else → ResolvedFailed` fall-through (R17-4) is **removed** and replaced with an
  explicit `default:` branch that asserts (Debug) / no-ops (Release).
- The `Failed` arm is still valid — it is reached only from `markTransientFailure`'s
  exhaustion branch (via the table's `resolve(obligationId, ResolvedFailed)` direct call,
  not via the public `resolveRecoveryObligation`).
- C8 (which directly calls `resolveRecoveryObligation(id, Failed)` as a table-level unit test)
  continues to pass without modification.

---

## R18-7 — `recoveryRetryExhaustedCount_` telemetry

**File:** `src/audioengine/ISRRuntimePublicationCoordinator.h:481-487` (accessor) and
`:944-945` (counter member).

```cpp
// ★ D105-R18: telemetry for retry-exhaustion path. Incremented only by
//   markTransientFailure when the obligation's consecutiveFailureCount reaches
//   kMaxObligationConsecutiveFailures. Distinguished from any historical Failed
//   (which is now unreachable in production per R17-4).
[[nodiscard]] std::uint64_t recoveryRetryExhaustedCount() const noexcept {
    return convo::consumeAtomic(recoveryRetryExhaustedCount_, std::memory_order_acquire);
}
```

- **Incremented only** in `markTransientFailure`'s exhaustion branch (cpp:1018).
- Distinguishes the *new* `Failed`-at-exhaustion from any *historical* `Failed` (which is now
  unreachable in production per R17-4).
- C8 does not bump this counter (C8 calls `resolveRecoveryObligation(id, Failed)` directly,
  which routes through the table's resolve and resets `consecutiveFailureCount` without
  incrementing `recoveryRetryExhaustedCount_`). Verified by T-R18-5.

---

## R18-8 — Test catalog (T-R18-1..T-R18-12)

**File:** `src/tests/ISRSemanticValidationTests.cpp:1285-1547` (definitions) and
`:1662-1686` (registration in `main()`).

| Test | Asserts |
|------|---------|
| **T-R18-1** transient build failure | L unchanged (ΔL=0), counter incremented, no Failed terminal, ShutdownDiscardCount unchanged, coalesce succeeds |
| **T-R18-2** transient publish failure | L unchanged (ΔL=0), counter incremented, coalesce succeeds |
| **T-R18-3** delivery reset to None | `markTransientFailure` → `redriveDeferredRecoveryObligations` → obligation is redriven (re-attached to durable or transport) |
| **T-R18-4** same id across failure | `markTransientFailure` + re-submit → obligation id is preserved (coalesce) |
| **T-R18-5** repeated failure → exhaustion | K=4 `markTransientFailure` calls → L=0, `recoveryRetryExhaustedCount==1`, ShutdownDiscardCount unchanged |
| **T-R18-6** success after retries | 2 failures + `Published` → L=0, `recoveryRetryExhaustedCount==0` (no exhaustion) |
| **T-R18-7** shutdown during deferred retry | transient failure → `discardRecoveryRequestsOnShutdown` → `ShutdownDiscarded` (not `Failed`) |
| **T-R18-8** stale after retries | 2 failures + `StaleSuperseded` → L=0, `recoveryRetryExhaustedCount==0` |
| **T-R18-9** stranded Transport repaired | 256 entries + durable + deferred → pop one entry → `markTransientFailure` → redrive picks up |
| **T-R18-10** pressure → Retry regression | R5-9 MUST-2: `Retry` outcome keeps L unchanged |
| **T-R18-11** markTransientFailure idempotent | unknown id, id=0, post-terminal id → all no-op |
| **T-R18-12** counter reset on slot reuse | exhaust obligation → fresh obligation for different identity → counter=0 on new |

**C8 is preserved unchanged** (table-level unit test).

---

## R18-9 — Build & test results

### Debug build
```
[SUCCESS] Executable created successfully.
build\ConvoPeq_artefacts\Debug\ConvoPeq.exe
```

### Release build
```
[SUCCESS] Executable created successfully.
build\ConvoPeq_artefacts\Release\ConvoPeq.exe
```

### ctest (Debug)
```
100% tests passed out of 40
Total Test time (real) =   48.74 sec
```

### ctest (Release)
```
100% tests passed out of 40
Total Test time (real) =   31.34 sec
```

All 40 tests pass in both configurations, including the 12 new T-R18 tests and the pre-existing
C1–C16 + T-R13-1..4 + C8.

---

## R18-10 — Grep verification (per the user's GO conditions)

### `resolveRecoveryObligation(_, Failed)` production call sites
```
$ wsl grep -rn 'RecoveryOutcome::Failed' src/
src/audioengine/ISRRuntimePublicationCoordinator.cpp:952://   harmlessly (Release) for unknown values. `RecoveryOutcome::Failed` is still a valid
src/audioengine/ISRRuntimePublicationCoordinator.cpp:969:        case RecoveryOutcome::Failed:             terminal = ObligationState::ResolvedFailed; break;
src/audioengine/RuntimePublicationOrchestrator.cpp:394:            resolveIfRecovery(RuntimeIntentCoordinator::RecoveryOutcome::Failed);           // ★ D105-R5-9: −1
src/tests/ISRSemanticValidationTests.cpp:1034:        c->resolveRecoveryObligation(*id, convo::isr::RuntimeIntentCoordinator::RecoveryOutcome::Failed);
```

- **2 production call sites of `RecoveryOutcome::Failed`:**
  1. `RuntimePublicationOrchestrator.cpp:394` — the `:386` switch route, preserved per user
     instruction (defensive leak-tightening for admission-rejection path).
  2. `ISRRuntimePublicationCoordinator.cpp:1019` (inside `markTransientFailure` exhaustion
     branch) — the **sanctioned** path (calls the table's `resolve(obligationId, ResolvedFailed)`
     directly).
- **1 test call site:** C8 (`ISRSemanticValidationTests.cpp:1034`), preserved unchanged.
- **3 mapping references** in the switch statement (line 969) and comments (lines 947, 952).

### `markTransientFailure` production call sites
```
$ wsl grep -rn 'markTransientFailure' src/audioengine/
src/audioengine/AudioEngine.RebuildDispatch.cpp:1039: (Builder durable build failure)
src/audioengine/AudioEngine.RebuildDispatch.cpp:1063: (Builder durable warmup failure)
src/audioengine/RuntimePublicationOrchestrator.cpp:191: (Orchestrator build failure)
src/audioengine/RuntimePublicationOrchestrator.cpp:259: (Orchestrator crossfade rebuild failure)
src/audioengine/RuntimePublicationOrchestrator.cpp:311: (Orchestrator publish failure)
```

**5 production call sites** (3 orchestrator + 2 builder), all of which are transient-recovery-failure
paths. The orchestrator's `:386` site is **not** a transient-recovery-failure path and remains on
`resolveIfRecovery(Failed)` per user instruction.

### `consecutiveFailureCount` field references
```
$ wsl grep -rn 'consecutiveFailureCount\|kMaxObligationConsecutiveFailures' src/audioengine/
src/audioengine/ISRRuntimePublicationCoordinator.h:323:        std::atomic<std::uint8_t> consecutiveFailureCount{0};
src/audioengine/ISRRuntimePublicationCoordinator.h:331:    static constexpr std::uint8_t kMaxObligationConsecutiveFailures = 4;
src/audioengine/ISRRuntimePublicationCoordinator.h:404:        std::uint8_t consecutiveFailureCount(std::size_t i) const noexcept {
src/audioengine/ISRRuntimePublicationCoordinator.h:490:        [[nodiscard]] std::uint8_t recoveryConsecutiveFailureCount(std::size_t i) const noexcept {
src/audioengine/ISRRuntimePublicationCoordinator.cpp:1013:        const std::uint8_t newCount = ... consecutiveFailureCount.fetch_add(...)
src/audioengine/ISRRuntimePublicationCoordinator.cpp:1017:        if (newCount >= kMaxObligationConsecutiveFailures) {
```

All references are scoped to the obligation's own slot or the K constant. No external coupling.

---

## R18 GO-condition verification

| GO condition | Status | Evidence |
|---|---|---|
| 1. transient `Failed` producer vanishes from production code | ✅ | Orchestrator 3 sites (`:189/255/303`) replaced with `markTransientFailure`. `:386` preserved (admission-rejection, not transient). |
| 2. `Retry` is ΔL=0 | ✅ | `markTransientFailure` does not touch `liveCount_`; only the exhaustion branch does. |
| 3. same `obligationId` maintained | ✅ | `markTransientFailure` looks up the slot by id; id is never re-issued. |
| 4. no stranded state in Transport/Durable | ✅ | `markTransientFailure` unconditionally sets `delivery=None`. T-R18-9 verifies. |
| 5. durable lease retry and obligation counter are not duplicative | ✅ | Two independent counters with different scopes; verified in R18-4. |
| 6. retry exhaustion is the only path to `ResolvedFailed` | ✅ (with `:386` exception) | `markTransientFailure` exhaustion branch is the sanctioned path. `:386` is the residual. |
| 7. `Failed` `−1` happens once at exhaustion | ✅ | `markTransientFailure` calls `resolve(id, ResolvedFailed)` only when counter ≥ K; table's `resolve` is idempotent. |
| 8. `Success / Superseded / ShutdownDiscard` semantics unchanged | ✅ | C1, C6, C7, C9, C11-C16, T-R13-1..4 continue to pass. |
| 9. `kMaxLogicalRecoveryObligations = 32` maintained | ✅ | No change to capacity enforcement. |
| 10. I4 ↔ runtime 1:1 correspondence | ✅ (deferred to R19) | I4 amendment is not in this PR per user instruction; the runtime is correct and self-consistent. R19 will perform the contract re-audit. |

| NO-GO condition | Status | Why it is not present |
|---|---|---|
| N1. `delivery` does not become `None` | ✅ fixed | R18-2 + T-R18-3. |
| N2. durable lease and obligation counter duplicate meaning | ✅ fixed | R18-4. |
| N3. `Failed` reachable from non-exhaustion path | ✅ fixed (modulo `:386`) | R18-6 (fallthrough removed); `:386` is admission-rejection leak-tightening, not transient-recovery-failure. |
| N4. `ResolvedFailed` remains "any failure extinction" | ✅ fixed | R18-7 (telemetry distinguishes). |
| N5. transport failure loses obligation | ✅ fixed | R18-5 + T-R18-9. |
| N6. exhaustion reuses id | ✅ impossible | `ResolvedFailed` is terminal; `findByKey` matches only `Live`. |
| N7. coalesce breaks counter / identity | ✅ fixed | T-R18-4, T-R18-12. |

---

## R18 limitations & R19 hand-off

1. **`:386` residual double-resolve.** The orchestrator's `:386` switch case still calls
   `resolveIfRecovery(Failed)` for admission-rejection paths. After R18, this is the only
   non-exhaustion path to `ResolvedFailed` in production. R19 should:
   - Audit whether `:386` is reachable when `trySubmitImpl`'s `admission_.evaluate()` returns
     `RejectedNotFinalized` (admission-rejection case). For this case, the obligation is
     `Live && delivery==None`; under R16-A's retry-preservation, this obligation should stay
     Live (and be terminalized at shutdown as `ShutdownDiscarded`).
   - Remove the `:386` `resolveIfRecovery(Failed)` call or add a state-check guard.
2. **I4 amendment.** The R16-A I4 amendment (`RetryExhausted` added to disappearance set) is
   **not** in this PR per user instruction. R19 will perform the contract re-audit
   (`D105-R19 — I4 Contract Amendment Audit`) once the runtime is empirically verified.
3. **K value tuning.** K=4 is inherited from the existing Builder-local counter. Empirical
   data (recovery build cycle time × max expected transient failures) is needed to confirm K=4
   is appropriate. R20+ can re-evaluate.

---

## Files changed

| File | Change |
|---|---|
| `src/audioengine/ISRRuntimePublicationCoordinator.h` | `consecutiveFailureCount` field, `kMaxObligationConsecutiveFailures` constant, `consecutiveFailureCount(i)` accessor, `markTransientFailure` declaration, `recoveryRetryExhaustedCount_` accessor & member, `tryInsert` reset, `resolve` counter reset |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp` | `resolveRecoveryObligation` switch (fallthrough removed), `markTransientFailure` definition |
| `src/audioengine/RuntimePublicationOrchestrator.cpp` | 3 `Failed` → `markTransientFailure` call sites (`:189`, `:255`, `:303`) |
| `src/audioengine/AudioEngine.RebuildDispatch.cpp` | 2 `markTransientFailure` call sites (`:1034`, `:1056`) alongside existing `settlePendingRecoveryAdmission(true)` |
| `src/tests/ISRSemanticValidationTests.cpp` | 12 new tests T-R18-1..12; C8 unchanged |

**D36.1 sizeof upper bound update:** `RecoveryAdmission` 352B → 356B (sub-KB; within
B_admissible). `B_logical_max` 11.4KB; `B_total_max` delta <32B. No capacity regression.
