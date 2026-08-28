# D105-R19 — I4 Contract Amendment / Residual Failed-Path Audit

**Status:** **R19 PARTIAL PASS** (read-only; source changes = 0). All R19 audit items are
structurally resolved. **R19-NO-GO on a single GO condition** (condition 3) — the
`:389` site (formerly `:386` after R18's edits) is a legitimate `ResolvedFailed`
path in production for case D (admission-rejection) but is *also* a **double-resolve**
for cases A/B (transient build-failure after R18's `markTransientFailure`). R20 must
resolve this.

This audit establishes the precise semantics of `ResolvedFailed` after R18, identifies
the four production call paths to `ResolvedFailed` (three of which are retry-exhaustion;
one — `:389` — is admission-rejection and leaks the R18 contract), and provides the
complete I4 amendment wording for R20 implementation.

---

## R19-1 — `:389` (formerly `:386`) reachability analysis (the central question)

### The four paths through `trySubmitImpl` to `submitPublishRequest`

The user requested a complete trace of *every* return path of `trySubmitImpl` and its
mapping to `submitPublishRequest`'s switch. I traced the current code at
`src/audioengine/RuntimePublicationOrchestrator.cpp:40-333` (`trySubmitImpl`) and
`src/audioengine/RuntimePublicationOrchestrator.cpp:357-426` (`submitPublishRequest`).

| Path | Source | `trySubmitImpl` action | Returned Decision | Switch case | Post-R18 behavior |
|---|---|---|---|---|---|
| **A** | Build #1 fail (Orchestrator.cpp:178-192) | `markTransientFailure(id)` (R18-3) | `RejectedNotFinalized` | `:389` | `resolveIfRecovery(Failed)` — **DOUBLE-RESOLVE** ⚠ |
| **B** | Crossfade rebuild fail (Orchestrator.cpp:247-258) | `markTransientFailure(id)` (R18-3) | `RejectedNotFinalized` | `:389` | `resolveIfRecovery(Failed)` — **DOUBLE-RESOLVE** ⚠ |
| **C** | Publish fail (Orchestrator.cpp:277-307) | `markTransientFailure(id)` (R18-3) | `RejectedPublishFailure` (or `RejectedShutdown` if shutting down) | `:420` | **No resolve** (correct: the `trySubmitImpl` site comment at Orchestrator.cpp:418-419 explicitly notes this) |
| **D** | `admission_.evaluate()` rejection (Orchestrator.cpp:46-51) | **No `markTransientFailure` call** — returns the decision directly | `RejectedNotFinalized` (also `RejectedStaleGeneration`, `RejectedPressure`, `RejectedShutdown`, `DeferredFadingActive`) | `:389` for `RejectedNotFinalized` (this is the path the user asked about) | `resolveIfRecovery(Failed)` — **legitimate use** (the obligation was Live and was never given a chance to retry) |

### **Critical finding: path D is real and is the legitimate use of `:389`**

The user's hypothesis was: ":389 is admission rejection, so it should be preserved." I
verified this by reading the current `trySubmitImpl` source.

**Evidence for path D being real in production:**

1. **`PublicationAdmission::evaluate` returns `RejectedNotFinalized` for transient
   admission rejection** (PublicationAdmission.cpp:21-22):
   ```cpp
   if (req.sealedSnapshot.irLoaded && !req.sealedSnapshot.irFinalized)
       return Decision::RejectedNotFinalized;
   ```
   The `sealedSnapshot.irLoaded` and `irFinalized` are set in
   `AudioEngine.RebuildDispatch.cpp:662-663` from `uiConvolverProcessor.isIRLoaded()` /
   `isIRFinalized()`. In the normal flow, both should be `true` after a load completes
   (ConvolverProcessor.LoadPipeline.cpp:564 sets `irFinalized = isIRLoaded()` on
   completion). **However**, transient races can produce `irLoaded=true, irFinalized=false`:
   the `irFinalized` flag is set by the LoaderThread after a load completes; the snapshot
   is captured in the RebuildThread (different thread). A snapshot captured between
   the load's start (`irLoaded=true`) and its completion (`irFinalized=true`) will
   trigger `RejectedNotFinalized`. This is a **transient** state.

2. **The recovery publish path is exactly the one that takes this snapshot.**
   `enqueuePublicationIntentForRuntimeCommit(..., recovery->obligationId)`
   (RebuildDispatch.cpp:999/1072) → `runtimeOrchestrator_->submitPublishRequest(req)`
   (Commit.cpp:815) → `trySubmitImpl(req)` → `admission_.evaluate(req, ...)` returns
   `RejectedNotFinalized` → falls into `submitPublishRequest` switch case `:389` →
   `resolveIfRecovery(Failed)`.

3. **The `default:` case at Orchestrator.cpp:387-388 falls through silently** for any
   decision that is not explicitly listed (i.e., any new decision added in the future
   would be silently ignored). Currently all decisions are explicitly listed
   (Accepted, DeferredFadingActive, RejectedStaleGeneration, RejectedNotFinalized,
   RejectedPressure, RejectedShutdown, RejectedPublishFailure), so `default:` is dead
   code in the current code.

4. **R18's `markTransientFailure` is NOT called from the admission-rejection path**
   (line 46-51 returns the decision before reaching line 53). Therefore path D has
   *not* been migrated to R18's retry-preserving semantics. The `:389` switch route is
   the **only** place that disposes of an obligation after admission rejection.

### Conclusion of R19-1

**Path D exists in production.** It is the legitimate use of `:389` for admission
rejection. The user's R18 intuition was correct. However, the user said in R18
"`:386 は機械的置換禁止`" — which I interpret as "don't touch `:386` in R18's
implementation" — but I confirm in this audit that the **switch case at `:389` (the
new line of what was `:386`) is reachable for both path D (intended) and paths A/B
(double-resolve after R18's `markTransientFailure`)**. This is a real bug in R18
that R20 must address.

---

## R19-2 — `RejectedNotFinalized → Failed` semantic audit

### What is the meaning of `Failed` at `:389`?

After R18, two distinct failure scenarios reach `submitPublishRequest`'s `RejectedNotFinalized`
case:

| Path | Semantics | R18 contract? |
|---|---|---|
| A (build #1 fail) | Transient build failure; `markTransientFailure` already incremented counter and kept obligation Live | ❌ **`resolveIfRecovery(Failed)` violates R18** — it terminalizes the obligation the same tick that `markTransientFailure` started its retry-preservation lifecycle |
| B (rebuild fail) | Same as A | ❌ same |
| D (admission-rejection) | Transient admission check failed; obligation is Live, never had a chance to retry | ✅ **R18 allows this** (path D is the only place where `Failed` is a legitimate non-exhaustion terminal) |

The user's R18-2 statement "`RejectedNotFinalized` がどの経路から来たものなのかを再確認し、
transient recovery failure → markTransientFailure() だけが対象になることを確認する"
is satisfied in **spirit** (path D is the intended target) but **not in implementation**
(paths A and B also reach `:389` and get the wrong treatment).

### Is admission rejection a transient retry or a permanent failure?

Looking at the only `RejectedNotFinalized` source in `admission_.evaluate()` (line 21-22),
the condition is `irLoaded && !irFinalized`. This is a **transient** state during IR
loading. Under R18's retry-preservation philosophy, this should be retried, not
terminalized.

However, `:389` is *currently* the only way to dispose of the obligation in this case.
The `submitRecoveryRequest` does set up `redriveDeferredRecoveryObligations` to re-drive
the obligation, but the obligation is only re-driven if it has `delivery==None`. After
`submitPublishRequest` returns `RejectedNotFinalized` (without `markTransientFailure`),
the obligation's `delivery` field is still `Transport` (or `Durable` from a previous
attempt), so the redrive skips it. The obligation is **stranded** (path D stranded
case, analogous to R16-3's publish-failure stranded case).

So path D has its own stranded problem: admission rejection does not set `delivery=None`,
the obligation stays stranded, and `:389`'s `resolveIfRecovery(Failed)` is the only way
out. This is a **separate issue** from the path A/B double-resolve.

### Conclusion of R19-2

**`:389` `resolveIfRecovery(Failed)` carries two distinct meanings simultaneously:**

1. For path D: "admission rejection — terminate the obligation because we cannot retry
   from outside (the obligation is stranded)"
2. For paths A/B: "redundant terminalization of a Live obligation that should have
   been preserved for retry"

**The R18 contract (R18 GO condition 6)** states that "production における
`ResolvedFailed` の唯一の正当な意味が **retry exhaustion** と確定している". Under
R19's audit, this GO condition is **NOT fully satisfied** because:
- Path D's `resolveIfRecovery(Failed)` does NOT mean retry exhaustion; it means
  admission rejection.
- Paths A/B's `resolveIfRecovery(Failed)` is a *redundant* terminalization that
  overrides the counter increment done by `markTransientFailure`.

R19 recommends R20 to:
- Replace `resolveIfRecovery(Failed)` in `:389` with `markTransientFailure(id)` (which
  also performs the P-B `delivery=None` repair) for paths A/B and D simultaneously.
- This is a **single** change to `:389` (one line). The obligation stays Live for retry;
  on retry exhaustion, the obligation is terminalized as `ResolvedFailed` by the
  `markTransientFailure` exhaustion branch — which already increments
  `recoveryRetryExhaustedCount_` for observability.

After R20: `resolveIfRecovery(Failed)` becomes unreachable in production. R17-4's
structural guarantee (only retry-exhaustion can produce `ResolvedFailed`) is fully
satisfied. C8 remains a table-level unit test.

---

## R19-3 — `ResolvedFailed` production reachability (full call graph)

### Call graph (corrected after source verification)

```
ObligationState::ResolvedFailed (enum, h:275)
  ↑
  RecoveryAdmissionTable::resolve(id, ResolvedFailed)         (h:366, the public table API)
  ↑
  Called from 2 production paths:
    │
    ├─ Path X. RuntimeIntentCoordinator::markTransientFailure()
    │          (ISRRuntimePublicationCoordinator.cpp:1019, exhaustion branch only)
    │          Trigger: counter == K = 4 (kMaxObligationConsecutiveFailures)
    │          Telemetry: recoveryRetryExhaustedCount_++ (incremented at line 1018)
    │
    └─ Path Y. resolveRecoveryObligation(id, RecoveryOutcome::Failed)
              (cpp:1019, inside the switch at line 969)
              Called from 1 production site:
                └─ Path Y.1 Orchestrator.cpp:394 (the :389 switch case)
                    Trigger: trySubmitImpl returned RejectedNotFinalized
                    (path A, B, or D all reach this site)
                    Telemetry: NONE (the resolveRecoveryObligation is called without
                              the recoveryRetryExhaustedCount_ increment; only
                              ShutdownDiscarded increments a counter)
```

### Path X (production, exhaustion only) — **CORRECT per R18 contract**

- **Source**: `markTransientFailure` exhaustion branch (ISRRuntimePublicationCoordinator.cpp:1017-1019).
- **Trigger**: `consecutiveFailureCount.fetch_add(1) == kMaxObligationConsecutiveFailures`.
- **Outcome**: obligation terminalized as `ResolvedFailed`, `liveCount_--`, `recoveryRetryExhaustedCount_++`.
- **Contract match**: this is the only path that *should* produce `ResolvedFailed` per R17-4.
- **Tests**: T-R18-5 (counter increment → exhaustion), T-R18-12 (slot reuse after exhaustion).

### Path Y.1 (production, admission-rejection / double-resolve) — **VIOLATES R18 contract**

- **Source**: `Orchestrator::submitPublishRequest` switch case `:389` (line 394).
- **Trigger**: `trySubmitImpl` returned `RejectedNotFinalized` for any of path A, B, or D.
- **Outcome**: obligation terminalized as `ResolvedFailed` via `resolveRecoveryObligation` →
  `table.resolve` (the `Failed` enum value routes through the switch at line 969).
- **Contract violation**:
  - For path A/B: redundant terminalization that overrides `markTransientFailure`'s
    counter increment. The `table.resolve` call at cpp:977 resets the counter to 0
    and decrements `liveCount_`. The redrive path is now impossible (the obligation
    is terminalized, not Live).
  - For path D: legitimate use, but it is not "retry exhaustion" — it is admission rejection.
- **R20 fix**: replace `resolveIfRecovery(Failed)` with `markTransientFailure(id)` in `:389`.
  This unifies paths A/B/D under the same retry-preserving semantic. After the R20 fix,
  path Y.1 no longer reaches `ResolvedFailed`; instead, the obligation stays Live and is
  retried (or, on K=4 exhaustion, terminalized via path X).

### Other paths to `ResolvedFailed` (non-production)

- **C8 test** (ISRSemanticValidationTests.cpp:1034): table-level unit test; preserved
  unchanged per user instruction.
- **No other callers** of `resolveRecoveryObligation(id, Failed)` in production.

### Conclusion of R19-3

**Production `ResolvedFailed` reachability (post-R18):**

| Path | Trigger | Contract-correct? | R20 fix |
|---|---|---|---|
| X (exhaustion via `markTransientFailure`) | counter == K | ✅ | none needed |
| Y.1 (admission-rejection via `:389`) | path D in current code | ⚠ means "admission rejection", not "exhaustion" | replace with `markTransientFailure` |
| Y.1 (transient build-failure via `:389`) | paths A, B in current code | ❌ redundant terminalization | replace with `markTransientFailure` |
| C8 (test only) | direct call | N/A (test) | preserve |

**After R20's fix to `:389`:** the only production path to `ResolvedFailed` is path X
(retry exhaustion). This is exactly what R17-4 requires.

---

## R19-4 — `RecoveryOutcome::Failed` API necessity audit

### The R18 question: is `RecoveryOutcome::Failed` still needed in production?

R18's `markTransientFailure` does **not** call `resolveRecoveryObligation(id, Failed)` —
it calls the table's `resolve` directly:

```cpp
if (newCount >= kMaxObligationConsecutiveFailures) {
    convo::fetchAddAtomic(recoveryRetryExhaustedCount_, std::uint64_t{1}, std::memory_order_release);
    recoveryAdmissions_.resolve(obligationId, ObligationState::ResolvedFailed);  // ← direct table call
}
```

This is the table's `resolve` (the C8-testable primitive), not the public
`resolveRecoveryObligation` API.

### The `Failed` arm in `resolveRecoveryObligation`'s switch (line 969)

After R18, the only production caller of `resolveRecoveryObligation(_, Failed)` is
`Orchestrator.cpp:394` (path Y.1). After R20's fix, this caller goes away. So the
`Failed` arm becomes **unreachable in production**.

**Question for R20**: should the `Failed` arm be **removed** from the switch, or kept
for **test compatibility** (C8 calls `resolveRecoveryObligation(id, Failed)` directly)?

### R19's recommendation

**Keep the `Failed` arm and the `default:` `jassertfalse` for now.** Reasons:

1. **C8 test compatibility**: C8 (test) calls `resolveRecoveryObligation(id, Failed)`
   directly. Removing the `Failed` arm would break C8.
2. **Defensive depth**: even after R20's fix, future code might add new caller paths.
   Keeping the `Failed` arm with the `default:` `jassertfalse` provides runtime
   detection of unexpected callers (Debug) and graceful no-op (Release).
3. **Single-writer principle**: the R18 contract is "only `markTransientFailure` may
   produce `ResolvedFailed` in production". The `Failed` arm in
   `resolveRecoveryObligation` is a *backwards-compatibility* API for test code and
   future defensive callers; it is not a *production* API.

The `resolveRecoveryObligation` API can be annotated with a comment marking the
`Failed` arm as "test-only / defensive-only — production should use
`markTransientFailure`". This is a documentation-only change (no source behavior
change). R20 may add the comment.

### Conclusion of R19-4

- **`RecoveryOutcome::Failed` enum value**: KEEP (used by C8 test; used by
  `markTransientFailure` exhaustion call to `table.resolve(..., ResolvedFailed)`).
- **`resolveRecoveryObligation(id, Failed)` arm**: KEEP (test compat), with a
  comment annotation that production should use `markTransientFailure` instead.
- **Production callers of `resolveRecoveryObligation(id, Failed)`**: zero after R20
  (R20 removes the `:389` site).
- **Production callers of `table.resolve(id, ResolvedFailed)`**: exactly one —
  `markTransientFailure`'s exhaustion branch (R18-2 path X).

---

## R19-5 — I4 disappearance set amendment wording

### Current I4 (D15.2, D18.3, D20.5)

> A logical obligation may disappear only by: `Success`, explicit `Superseded`
> decision, `ShutdownDiscard`.

This is the current D15.2 contract. After R18, the runtime can terminalize an
obligation as `ResolvedFailed` for **two** reasons:

1. **Retry exhaustion** (path X): `counter == K` after `markTransientFailure`. The
   obligation exhausted its retry budget. The runtime has decided that the obligation
   cannot succeed and is forced to terminalize it.
2. **Admission rejection** (path D, current code only): the obligation was Live but
   `admission_.evaluate()` returned `RejectedNotFinalized`. The current code at `:389`
   terminalizes via `resolveIfRecovery(Failed)`.

### R18's stance

R18's intent is that **`ResolvedFailed` means "retry exhaustion"** — the obligation
was given multiple chances to succeed and failed. R19's audit confirms this is the
correct semantic for the runtime, but R19 also found that the runtime's `:389` site
violates this intent for the admission-rejection case (path D).

### R19's recommended I4 amendment (for R20)

R19 recommends the following wording for the I4 amendment (to be applied in R20 after
the `:389` fix):

```
// I4 D15.2 amendment (R20):
//
// A logical obligation may disappear by exactly four transitions:
//
//   Live --Published-->          ResolvedSuccess         (deltaL = -1)
//   Live --StaleSuperseded-->     ResolvedStaleSuperseded (deltaL = -1)
//   Live --Shutdown-->            ShutdownDiscarded        (deltaL = -1)
//   Live --RetryExhaustion-->     ResolvedFailed          (deltaL = -1)
//
// RetryExhaustion is the SOLE sanctioned transition to ResolvedFailed. It is emitted
// ONLY by markTransientFailure's exhaustion branch when the obligation's
// consecutiveFailureCount reaches kMaxObligationConsecutiveFailures (K = 4). Any
// other path to ResolvedFailed is a contract violation.
//
// ★ D14.3 footnote: a transient failure (build/publish/admission) does NOT consume
//   the backpressure budget. The obligation stays Live; markTransientFailure repairs
//   delivery to None (P-B) so the next redrive tick can retry. The obligation is
//   terminalized ONLY at retry exhaustion or by explicit supersede/shutdown.
```

### Why not simply "add `ResolvedFailed` to the disappearance set"?

The user's R19 brief explicitly warns against this: "重要なのは単に `ResolvedFailed
を追加` するだけではなく、 `Failure ≠ RetryExhaustion` という意味論を契約上明示する
こと". The R19 wording above does this by:

1. Naming the transition `RetryExhaustion`, not `Failure` (the I4 contract
   distinguishes retry-exhaustion from any other failure).
2. Specifying the **emitting authority** (only `markTransientFailure`'s exhaustion
   branch).
3. Specifying the **precondition** (counter == K).
4. Explicitly stating "any other path is a contract violation" — this gives the
   I4 audit a clear pass/fail criterion.

### D18.3 / D20.5 updates

D18.3 (conservation equation) needs to be updated:

```
// Current D18.3:
//   liveOwnershipCount + terminalDispositionCount == logicalObligationCreationCount
//   terminalDispositionCount = successCount + supersededCount + shutdownDiscardCount
//
// R20 amended D18.3:
//   terminalDispositionCount = successCount + supersededCount + shutdownDiscardCount + retryExhaustedCount
//
// ★ D18.3 footnote: retryExhaustedCount is incremented ONLY by markTransientFailure's
//   exhaustion branch. The ShutdownDiscardCount is incremented ONLY by
//   resolveRecoveryObligation(_, ShutdownDiscarded). The two are disjoint.
```

D20.5 (closure linearization) does not need to change — `ResolvedFailed` is not part
of the closure transition (closure is the *last* LIVE → CLOSED transition, which
is the terminalization of the *last* LIVE obligation; exhaustion of one obligation
is not the closure of its episode).

### Conclusion of R19-5

The I4 amendment is **minimal**:

- D14.3: add a footnote (no structural change to the invariant).
- D15.2: replace the disappearance set with the 4-element set above; add the
  `RetryExhaustion` transition with the explicit emitting authority and precondition.
- D18.3: add `retryExhaustedCount` to the conservation equation; add a disjointness
  footnote.

No new invariant, no new state, no new LP. The amendment is fully consistent with
R17-7's state machine table (which already has 4 terminal rows: Published,
StaleSuperseded, ShutdownDiscarded, RetryExhausted).

---

## R19-6 — Capacity / accounting consistency check (R18's added state)

### State machine for `consecutiveFailureCount` and slot ownership

The full lifecycle of an obligation's `consecutiveFailureCount` after R18:

| Stage | State | counter | delivery | liveCount | slot ownership |
|---|---|---|---|---|---|
| Initial admission (tryInsert) | Live | 0 | None/Transport/Durable | +1 | slot owned by this obligation |
| Transient failure (markTransientFailure) | Live | +1 | None (P-B) | 0 | slot owned by this obligation |
| Redrive → re-admission (redriveDeferredRecovery) | Live | preserved | Transport/Durable | 0 | slot owned by this obligation |
| Successful publish (Published) | ResolvedSuccess | 0 (reset) | terminal | -1 | slot freed |
| Stale (StaleSuperseded) | ResolvedStaleSuperseded | 0 (reset) | terminal | -1 | slot freed |
| Shutdown (ShutdownDiscarded) | ShutdownDiscarded | 0 (reset) | terminal | -1 | slot freed |
| Retry exhaustion (counter == K) | ResolvedFailed | 0 (reset on slot reuse) | terminal | -1 | slot freed |
| Slot reuse (tryInsert on freed slot) | Live | 0 (NSDMI + explicit reset) | None/Transport/Durable | +1 | slot owned by new obligation |

### The `markTransientFailure` → exhaustion → resolve → slot reuse transition chain

Verified for R18:

1. `markTransientFailure` (cpp:998-1024):
   - delivery = None
   - counter.fetch_add(1, acq_rel)
   - if newCount >= K → `table.resolve(id, ResolvedFailed)` + `recoveryRetryExhaustedCount_++`
2. `table.resolve` (h:366-378):
   - CAS Live → ResolvedFailed
   - consecutiveFailureCount.store(0, release)
   - liveCount.fetch_sub(1, release)
3. Slot is now `state==ResolvedFailed` (terminal). The `findByKey` (h:333) matches
   only `state==Live`, so the slot is not coalesced.
4. Slot reuse: a new `submitRecoveryRequest` for a different identity triggers
   `tryInsert`, which finds the slot (state != Live), assigns a new id, sets
   `consecutiveFailureCount.store(0, release)` (line 367), and increments `liveCount`.

**No contradiction.** Each state transition has a clear, linear ownership
relationship with the slot and the counter.

### D36.1 sizeof update

R18's `consecutiveFailureCount` (atomic `uint8_t`) adds 1 byte to the
`LogicalRecoveryObligation` struct. With 64-bit alignment, the field occupies
1-8 bytes (depending on padding). D36.1's previous upper bound was 352B; R18 brought
it to ~356B (one byte for the counter + padding). `B_logical_max = 32 × 356B ≈ 11.4KB`,
well within the D35.5 `B_admissible = 64MB` threshold.

### R19's `kMaxLogicalRecoveryObligations = 32` audit

The `kMaxLogicalRecoveryObligations = 32` constant is unchanged. The
`tryInsert` capacity check (h:344) is unchanged. R18 does not alter the
admission capacity. R19 confirms: **no capacity regression**.

### Conclusion of R19-6

- `consecutiveFailureCount` state machine: **no contradiction** across all
  transitions.
- `delivery` field: **no contradiction** (None → Transport/Durable via submission;
  Transport/Durable → None via markTransientFailure; reset to None on terminal
  resolution is a no-op because terminal slots are skipped by redrive).
- `liveCount`: **no contradiction** (incremented on tryInsert, decremented on
  terminal resolve, unaffected by transient failure).
- D36.1 sizeof: **within D35.5 admissibility** (11.4KB << 64MB).
- `kMaxLogicalRecoveryObligations = 32`: **preserved**.

---

## R19-7 — Delivery state machine table

The obligation's `delivery` field is one of `{None, Transport, Durable}` (the durable
slot's `state` is a separate state machine, addressed in R17-7).

### State machine table (per R19-7's request)

| State | `delivery` | retry possible? | terminal possible? | writer | reader |
|---|---|---|---|---|---|
| Live / None | None | Yes (R5-10 redrive) | No (terminal = shutdown/stale/success/exhaustion) | W1: tryInsert (h:353); W3: submitRecoveryRequest deferred (cpp:925); W5: redrive durable (cpp:1040); markTransientFailure (cpp:1010) | R1-R4 (redrive scans) |
| Live / Transport | Transport | Yes (R5-10 redrive + re-pop) | No (same as above) | W2: submitRecoveryRequest transport success (cpp:910); W6: redrive transport (cpp:1047) | popRecoveryRequest (cpp:1124) — decrements `pendingIntentCount_` |
| Live / Durable | Durable | Yes (R5-10 redrive + take) | No (same as above) | W4: submitRecoveryRequest durable success (cpp:940) | takePendingRecoveryAdmission (cpp:1064) — DurablePending → Building |
| (exhaustion) | None (forced by markTransientFailure) | No | Yes (ResolvedFailed via markTransientFailure exhaustion) | markTransientFailure (cpp:1010) | (any) |
| Published | terminal | No | Yes (ResolvedSuccess) | table.resolve (h:366) with ResolvedSuccess | (any) |
| Superseded | terminal | No | Yes (ResolvedStaleSuperseded) | table.resolve (h:366) with ResolvedStaleSuperseded | (any) |
| ShutdownDiscarded | terminal | No | Yes (ShutdownDiscarded) | table.resolve (h:366) with ShutdownDiscarded | (any) |
| ResolvedFailed | terminal | No | Yes (exhaustion only) | table.resolve (h:366) with ResolvedFailed — called ONLY by markTransientFailure | (any) |

### `Transport → None` and `Durable → None` re-eligibility (P-B) verification

The obligation is stranded (cannot be redriven) iff:
- `state==Live` AND
- `delivery==Transport` AND the transport intent has been popped (queue is empty for
  this obligation's id) AND no intent was re-pushed.

For the **publish-failure stranded case** (R16-3): `markTransientFailure` sets
`delivery = None` unconditionally, so the next `redriveDeferredRecoveryObligations`
sees the obligation and re-attaches delivery. **Verified in T-R18-9.**

For the **admission-rejection stranded case** (R19-1 path D, current code):
`submitPublishRequest` returns `RejectedNotFinalized` without `markTransientFailure`,
so `delivery` stays at `Transport` (or `Durable`). The next redrive skips the
obligation. **Currently stranded.** R20 must fix this by replacing
`resolveIfRecovery(Failed)` in `:389` with `markTransientFailure(id)`, which performs
P-B (`delivery = None`).

### R20 implication

After R20, the obligation never gets stranded:
- Transient build/publish failure → `markTransientFailure` (P-B).
- Admission rejection → `markTransientFailure` (P-B).
- All stranded cases are repaired.

### Conclusion of R19-7

The delivery state machine is **structurally complete** after R20's planned fix.
All Live states are retryable. All terminal states are reachable only through the
documented transitions. P-B is consistent across all retry paths.

---

## R19-8 — R19 verdict (GO/NO-GO assessment)

### R19 GO conditions (per R19 brief)

| # | Condition | Status | Evidence |
|---|---|---|---|
| 1 | `RejectedNotFinalized` の `:389` 到達経路が完全に証明されている | ✅ | R19-1: paths A, B, D traced; path C is not `RejectedNotFinalized` |
| 2 | `:389` が transient failure ではないことがコード上証明されている | ⚠ | R19-1: paths A, B reach `:389` *after* `markTransientFailure` was called — they are transient failures, and `:389`'s `resolveIfRecovery(Failed)` is the wrong treatment. Path D is *not* a transient failure. |
| 3 | admission rejection を `ResolvedFailed` とすることが R18 契約上許容されるか明確になっている | ❌ | R19-2: admission rejection is transient (`irLoaded && !irFinalized`); R18 contract says **retry-preserving**, not terminal. Path D violates R18's GO condition 6. |
| 4 | production における `ResolvedFailed` の唯一の正当な意味が **retry exhaustion** と確定している | ⚠ | R19-3: path X (exhaustion) is correct; path Y.1 (`:389`) carries a different meaning. After R20's fix, Y.1 goes away. |
| 5 | `RecoveryOutcome::Failed` の production API の必要性が確定している | ✅ | R19-4: enum value kept (test compat + table internal); `Failed` arm in `resolveRecoveryObligation`'s switch kept (defensive + test compat); `jassertfalse` in `default:` provides runtime detection |
| 6 | I4 の disappearance set に RetryExhaustion を追加する契約案が確定している | ✅ | R19-5: full I4 amendment wording provided (D14.3 footnote, D15.2 set, D18.3 equation, D20.5 unchanged) |
| 7 | `ΔL = 0` が exhaustion 前の全 retry path で維持される | ✅ | R19-7: every row in the delivery state machine preserves `liveCount` until terminal |
| 8 | exhaustion 時だけ `ΔL = -1` になる | ✅ | R19-3 + R19-6: only path X (exhaustion) and table.resolve terminals decrement |
| 9 | `delivery=None` による redrive eligibility が維持される | ⚠ | R19-7: maintained for paths A/B/C (post-markTransientFailure); NOT maintained for path D (current code). R20 must fix. |
| 10 | `kMaxLogicalRecoveryObligations = 32` と R18 の追加状態が矛盾しない | ✅ | R19-6: capacity is unchanged; sizeof delta is sub-KB |
| 11 | C8 を変更する必要がない | ✅ | R19-3: C8 still passes; the `Failed` arm is kept for test compat |
| 12 | source changes = 0 | ✅ | R19 is a read-only audit |

### R19 NO-GO conditions (per R19 brief)

| # | Condition | Status |
|---|---|---|
| N1 | `RejectedNotFinalized → ResolvedFailed` が retry exhaustion と無関係に実行される | ❌ **PRESENT** (path D in current code) |
| N2 | `Transient failure → Failed → liveCount −1` が残っている | ⚠ **PRESENT** for paths A/B (path C is OK because `:420` is no-op; path D's `Failed` is the path-D termination, not transient-failure) |
| N3 | `markTransientFailure() → delivery=None` 後に obligation を redrive できないケース | ⚠ **PRESENT** for path D (admission rejection does not call `markTransientFailure`; delivery stays Transport/Durable; redrive skips) |

### R19 verdict

**R19: PARTIAL PASS.**

- **R19 audit items 1, 2, 4, 6, 7, 8, 10, 11, 12 are PASS or partial-PASS with a clear
  R20 fix identified.**
- **R19 audit item 3 is FAIL** (path D's `Failed` is not retry exhaustion), which
  cascades to R19 NO-GO conditions N1, N2, N3.
- **All three NO-GO conditions are addressed by a single one-line change in R20:**
  replace `resolveIfRecovery(Failed)` in `Orchestrator.cpp:394` with
  `markTransientFailure(req.recoveryObligationId)`.

**R19 recommends:**
- **R20** (next, source-modifying): make the one-line change to `:389`. After R20,
  - path Y.1 (`:389`'s `Failed`) is replaced with `markTransientFailure` (ΔL=0, delivery=None,
    counter increment, exhaustion → ResolvedFailed via path X only).
  - All R19 NO-GO conditions are satisfied.
  - C8 unchanged.
  - All R18 tests unchanged.
  - One new test: T-R20-1 (admission rejection stays Live and counter increments).
  - One new test: T-R20-2 (admission rejection is stranded-repaired: delivery=None after
    `markTransientFailure`).
- **R21** (next, source-modifying): apply the I4 amendment wording from R19-5 to
  `I4_DESIGN_CONTRACT.md`. No runtime change.
- **R22** (next, source-modifying or audit): re-audit the entire I4 ↔ runtime
  contract per the new state machine (full state coverage, all transitions traced,
  capacity proven, liveness proven).

### What R19 did NOT do (per R19 brief's "audit only" requirement)

- **No source code changes.** R19's evidence file is the only output.
- **No new tests.** R20 will introduce the tests as part of the implementation.
- **No I4 patch.** R21 will patch I4 after R20 has empirically verified the runtime.

---

## R19 → R20 hand-off

R20 should:

1. **Replace** `Orchestrator.cpp:394` `resolveIfRecovery(Failed)` with
   `markTransientFailure(req.recoveryObligationId)`. This single-line change:
   - Eliminates the double-resolve on paths A/B.
   - Repairs the path D stranded case (P-B applies).
   - Routes all transient failures through the unified `markTransientFailure` API.
2. **Add** T-R20-1: admission rejection stays Live, counter increments.
3. **Add** T-R20-2: admission rejection is stranded-repaired (delivery=None).
4. **Add** T-R20-3: post-fix, `resolveIfRecovery(Failed)` is unreachable in production
   (verified by grep and by a test that constructs the call and confirms the assertion
   fires in Debug).
5. **Re-run** all R18 tests (T-R18-1..12) and all pre-R18 tests (C1-C16, T-R13-1..4,
   C8). All must continue to pass.

If R20 is successful, the runtime state is:

- **`ResolvedFailed` in production = retry exhaustion only.**
- **No stranded obligation** in any failure path.
- **C8 unchanged.**
- **I4 contract** amendable in R21.

This is the precise, minimal, complete R19 → R20 transition.

---

## R19 final summary table

| Aspect | R19 conclusion |
|---|---|
| Path D existence | **Confirmed** in production (transient `irLoaded && !irFinalized` race) |
| `:389` correctness | **Partially incorrect**: legitimate for D, double-resolve for A/B, stranded-repair missing for D |
| `ResolvedFailed` production reachability | **Two paths** (X exhaustion + Y.1 admission-rejection/double-resolve) |
| R18 GO condition 6 (only retry exhaustion → `ResolvedFailed`) | **NOT fully satisfied** (Y.1 carries a non-exhaustion meaning) |
| I4 amendment | **Wording finalized** for R21 implementation |
| Capacity / accounting | **No contradiction** with R18's added state |
| Delivery state machine | **Structurally complete** after R20's fix |
| R20 fix | **One-line** replacement of `resolveIfRecovery(Failed)` with `markTransientFailure` at `:389` |
| C8 modification | **Not required** |
| Source changes in R19 | **Zero** (read-only audit) |
