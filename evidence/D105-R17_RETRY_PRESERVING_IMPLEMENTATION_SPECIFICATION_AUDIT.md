# D105-R17 — Retry-Preserving Implementation Specification Audit

**Status:** AUDIT COMPLETE — **R17-A GO** (read-only specification, source change = 0). R16-A design is
**fully specifiable** with the four state machines (`ObligationState`, `ObligationDeliveryState`,
`PendingRecoveryAdmission::State`, per-obligation `consecutiveFailureCount`) and the existing
authority surfaces (`trySubmitImpl`, `submitRecoveryRequest`, `resolveRecoveryObligation`,
`rearmRecoveryRetry`, `redriveDeferredRecoveryObligations`, `takePendingRecoveryAdmission`,
`settlePendingRecoveryAdmission`).

Purpose: turn the R16-A *intent* ("transient failures stay Live, retry-preserving, terminal only at
exhaustion") into a **complete and unambiguous** set of: (a) state machine transitions, (b) mutation
authorities, (c) linearization points, (d) per-event field updates, (e) wakeup mechanisms. The
implementation is deferred to R18; this audit establishes the contract R18 must satisfy.

---

## R17-1 — Retry state transitions reconciled with the existing durable lease retry

### Two already-existing retry primitives

The current runtime has **two structurally distinct retry mechanisms** that are currently
**semi-decoupled**. Both must be reconciled before R16-A can be implemented without duplicating
state.

#### A. The durable slot lease retry (Builder-side, **already implemented**)

`AudioEngine.RebuildDispatch.cpp:1015-1076`:

```text
takePendingRecoveryAdmission()   ←  lease: DurablePending → Building (non-destructive)
   ↓
runtimeBuilder.build(recovery->buildSource.buildInput, convolverSnapshot)   ←  builds
   ↓
   if runtime == nullptr:                        build failure
       settlePendingRecoveryAdmission(true)       ←  Building → DurablePending (LEASE RETRY)
       if ++recoveryConsecutiveFailures >= 4:    ←  K=4, local to this loop
           break
   elif validateWarmup != None:                  warmup failure
       destroyDSPCoreNode(dspGuard.ptr)
       settlePendingRecoveryAdmission(true)       ←  Building → DurablePending
       if ++recoveryConsecutiveFailures >= 4:    break
   else:                                         success
       enqueuePublicationIntentForRuntimeCommit(..., recovery->obligationId)
       settlePendingRecoveryAdmission(false)      ←  NoAdmission (slot cleared)
       recoveryConsecutiveFailures = 0
```

**Key properties**:
- **State machine**: `NoAdmission ⇄ DurablePending → Building → DurablePending ⇄ NoAdmission` (via settle).
- **Counter**: `recoveryConsecutiveFailures` is a **local `int` inside `rebuildThreadLoop`**, reset on
  success (`:1075`) or on `break`-out (counter is discarded with the stack frame).
- **Scope**: only **durable-resident** obligations (the `take` path).
- **Obligation-level state**: **untouched** — `RecoveryAdmissionTable::state` stays `Live`, `delivery`
  stays `Durable` for the whole loop. The obligation's `Live` count and identity are *completely
  unaware* of the lease retry — the table sees a single `Live` obligation the entire time, the
  durable slot's `Building → DurablePending` is a *transport* detail.

#### B. The `Retry` outcome in `resolveRecoveryObligation` (Coordinator-side, **a no-op stub**)

`ISRRuntimePublicationCoordinator.cpp:948-964`:

```cpp
void RuntimeIntentCoordinator::resolveRecoveryObligation(std::uint64_t obligationId, RecoveryResolution outcome) noexcept
{
    if (obligationId == 0) return;
    if (outcome == RecoveryOutcome::Retry)
        return;                                                          // ★  no-op (no obligation mutation)
    const ObligationState terminal = (outcome == Published)        ? ResolvedSuccess
                                   : (outcome == StaleSuperseded) ? ResolvedStaleSuperseded
                                   : (outcome == ShutdownDiscarded)? ShutdownDiscarded
                                   :                                  ResolvedFailed;  // ★  fallthrough
    if (recoveryAdmissions_.resolve(obligationId, terminal) && outcome == ShutdownDiscarded)
        convo::fetchAddAtomic(recoveryObligationShutdownDiscardCount_, 1, ...);
}
```

**Key properties**:
- `Retry` is currently a *complete no-op* — it doesn't touch `ObligationState`, doesn't touch
  `delivery`, doesn't touch any counter. The only effect of a `Retry` call is to potentially pass
  through to `rearmRecoveryRetry` (caller decides; cpp:396-398).
- `Failed` is **reachable via the fallthrough** (`else → ResolvedFailed`). R15-1 identified the four
  production call sites of this fallthrough. R17-4 must eliminate this fallthrough.
- **`ResolvedRetry` is declared in `ObligationState` (h:277) but never reached** — the comment in
  R5-8 §11 noted it as a placeholder. R17-4 must decide whether to *use* it (as a diagnostic
  state) or *remove* it.

### How R16-A reconciles the two

| Mechanism | Scope | What it does | What R16-A changes |
|---|---|---|---|
| **A: durable slot lease retry** | durable-resident obligation only | `Building → DurablePending` (transport-level re-attempt) | **No change.** R16-A leaves A intact. |
| **B: `Retry` outcome in `resolveRecoveryObligation`** | any Live obligation (transport/durable/deferred) | Currently a no-op; intended for ΔL=0 retry-preservation | R16-A promotes B from a no-op to an **active obligation-level state transition** (see R17-2 / R17-4). |

The two are **not duplicative** because they act on **different state machines**:
- A acts on `PendingRecoveryAdmission::State` (transport-side durable slot).
- B acts on `ObligationState` (the table's logical lifecycle).

A `Live && delivery==Durable` obligation that fails a build in the durable path benefits from **both**
mechanisms working in concert: A keeps the durable slot alive (next take), B keeps the obligation
alive (next re-arm or next take). The obligation's identity (table slot, id) is unchanged across
retries; the durable slot's `Building` sub-state cycles; the obligation's `delivery` stays `Durable`
throughout.

A `Live && delivery==Transport` obligation that fails a publish in the transport path (R16-3's
stranded case) benefits from **B alone** — A cannot help because there is no durable slot to lease.
R16-A's contract for B must therefore:
- reset `delivery` to `None` (so the obligation is redrive-eligible — **P-B** in R16-4); and
- drive the obligation's failure counter.

### R17-1 conclusion

The two retry mechanisms are **complementary**, not duplicative. R16-A's implementation must:
1. **Keep A unchanged** (the durable lease retry already works for the durable-resident case).
2. **Promote B from a no-op to a real transition** for build/publish failures. The promotion is the
   *only* new state machinery R16-A requires.
3. **Identify the obligation-level counter location** (R17-3).
4. **Eliminate the `Failed` fallthrough** in `resolveRecoveryObligation` so that B's transition is the
   *only* path that can produce `ResolvedFailed` at exhaustion (R17-4).

---

## R17-2 — `delivery` field authority / mutation enumeration

### Writer enumeration (every site that mutates `delivery`)

A `rg 'delivery\s*=\s*ObligationDeliveryState'` over `src/` returns **exactly 5 writer sites**, all
in `ISRRuntimePublicationCoordinator.cpp` (line numbers from current source):

| # | Site | Authoritative authority (per D29.8 LP-B) | What it does |
|---|---|---|---|
| **W1** | `ISRRuntimePublicationCoordinator.h:353` (inside `RecoveryAdmissionTable::tryInsert`) | **Admission authority** (CoordinatorLoop) | On fresh slot: `delivery = None`. (Note: this is a *new obligation* entry, not a transition.) |
| **W2** | `ISRRuntimePublicationCoordinator.cpp:910` (inside `submitRecoveryRequest` — transport push success) | **Admission authority** (CoordinatorLoop) | `delivery = Transport` after `recoveryIntentQueue_.push` succeeds. |
| **W3** | `ISRRuntimePublicationCoordinator.cpp:925` (inside `submitRecoveryRequest` — durable slot held by *different* obligation) | **Admission authority** (CoordinatorLoop) | `delivery = None` (deferred; obligation stays Live, delivery re-driven by `redrive`). |
| **W4** | `ISRRuntimePublicationCoordinator.cpp:940` (inside `submitRecoveryRequest` — durable admission) | **Admission authority** (CoordinatorLoop) | `delivery = Durable` after the durable slot is filled. |
| **W5** | `ISRRuntimePublicationCoordinator.cpp:1040` (inside `redriveDeferredRecovery` — durable re-attachment) | **Redrive authority** (CoordinatorLoop, R5-10) | `delivery = Durable` on successful re-attachment to the durable slot. |
| **W6** | `ISRRuntimePublicationCoordinator.cpp:1047` (inside `redriveDeferredRecovery` — transport re-attachment) | **Redrive authority** (CoordinatorLoop, R5-10) | `delivery = Transport` on successful re-attachment to the transport queue. |

**There are 6 writers (W1–W6), all on the CoordinatorLoop (single producer for `recoveryAdmissions_`).**
The Builder, the Orchestrator, and the ISR executor **never write `delivery` directly**. The
obligation's `delivery` field is therefore a **CoordinatorLoop-private field**; reads from
elsewhere (e.g. the redrive scan) are atomic-load reads.

### Reader enumeration

| # | Site | Reads `delivery` to do what |
|---|---|---|
| **R1** | `ISRRuntimePublicationCoordinator.cpp:845` (inside `submitRecoveryRequest`'s `wasDeferredBefore` lambda) | snapshot of pre-redrive `delivery==None` for C16 single-representation early-return |
| **R2** | `ISRRuntimePublicationCoordinator.cpp:877` (inside `submitRecoveryRequest` coalesce branch) | `if (wasDeferredBefore && delivery != None) return true;` — avoid double push after redrive |
| **R3** | `ISRRuntimePublicationCoordinator.cpp:988` (inside `redriveDeferredRecoveryObligations` scan) | `if (s.delivery != None) continue;` — only re-drive deferred obligations |
| **R4** | `ISRRuntimePublicationCoordinator.cpp:1014` (inside `redriveDeferredRecovery` per-id) | `if (s.delivery != None) return;` — idempotent |

All reads are on the CoordinatorLoop (single producer). Reads from the Builder (e.g. the durable
loop at RebuildDispatch.cpp:1017) do not consult `delivery`; they consult
`pendingRecoveryAdmission_.state` (the durable slot's own state machine).

### Authority separation per I4/D29.8

I4-D29.8 separates three authorities:
- **Admission authority** (CoordinatorLoop) — owns `tryInsert`, coalesce lookup, and **delivery
  assignment** (W1–W4).
- **Settlement authority** (Builder thread, ISR completion) — owns `Published` / `StaleSuperseded`
  terminal calls and the durable slot's `Building` sub-state.
- **Completion authority** (CoordinatorLoop or ISR, the `resolveRecoveryObligation` function) —
  owns the table's `Live → terminal` CAS.

**R16-A's P-B ("set `delivery = None` at Retry") is a *delivery* mutation, not a *terminal* mutation**.
Per the I4-D29.8 authority model, the mutation belongs to the **Admission authority** (the entity that
already writes `delivery`). Putting it inside `resolveRecoveryObligation` is *conceptually
mismatched* because that function is the *Completion* authority — but the function is called from
the CoordinatorLoop (in the build-failure paths of `trySubmitImpl` at Orchestrator.cpp:189/255/303),
and the CoordinatorLoop is *also* the Admission authority. **The cleanest design is therefore to
split the `Retry` outcome's two effects**:

```
resolveRecoveryObligation(id, Retry):
    1. NO table mutation (Live stays Live, no -1)   ← Completion authority: ΔL=0 contract
    2. (new) delivery = None                        ← Admission authority: re-eligibility for redrive
    3. (new) consecutiveFailureCount++              ← Adjudication authority: per-obligation counter
    4. (new) if consecutiveFailureCount == K → resolve(id, Failed)  ← terminal exhaustion
```

This split is **structurally clean**: each of (1)(2)(3)(4) is a single field mutation or a single
terminal call, and each is by a clearly-named authority. The implementation in R18 will wrap this
in a single helper (`markTransientFailure(id)`) that the Orchestrator's failure sites call instead of
calling `resolveRecoveryObligation(id, Failed)`.

### What `Building` means in this model

The user asked whether `Building` is a delivery state. **No** — `Building` is a
`PendingRecoveryAdmission::State` (the durable slot's sub-state), not an `ObligationDeliveryState`.
The obligation's `delivery` is `Durable` for the entire `DurablePending → Building → DurablePending`
cycle; the slot's `state` is what tracks "in the middle of a build attempt". A `Live && delivery==Durable`
obligation corresponds to `state ∈ {DurablePending, Building}` of the slot; the obligation does not
care which.

### `delivery` after a build failure in the durable path (R16-A case)

For a durable-resident obligation:
- `trySubmitImpl` is **not** the entry point. The Builder takes the durable slot, runs
  `runtimeBuilder.build`, and on failure calls `settlePendingRecoveryAdmission(true)`
  (RebuildDispatch.cpp:1034). This puts the slot back into `DurablePending`. The obligation's
  `delivery` was `Durable` before the failure and is `Durable` after — **no `delivery` mutation is
  needed** for the durable-resident case under R16-A.
- The obligation's `consecutiveFailureCount` is incremented by the Builder (or, under R16-A, by the
  failure site — see R17-3 for the location debate).
- The Builder breaks out of the loop at K=4 consecutive failures (`break` at cpp:1037/1058). The
  obligation stays in `DurablePending` and is taken on the next Builder wake.

**The Orchestrator's `Failed` site is never reached for durable-resident failures**, because the
Builder's path is `RuntimeBuilder.build(...) → settle(true) → break`, *not*
`executor_.publish(...) → resolveRecoveryObligation(Failed)`. The Orchestrator's failure sites are
specifically the `trySubmitImpl` path (which is exercised by transport-resident obligations and by
the *second* attempt via `submitPublishRequest`).

### `delivery` after a build failure in the transport path (R16-A case, the stranded concern)

For a transport-resident obligation:
- `trySubmitImpl` is the entry point. The Orchestrator sets up the spec, calls
  `worldBuilder.buildRuntimePublishWorld` (cpp:176/244). On `!worldOwner`, the Orchestrator
  currently calls `resolveRecoveryObligation(req.recoveryObligationId, Failed)` (cpp:189/255).
- Under R16-A, the Orchestrator instead calls `markTransientFailure(req.recoveryObligationId)`:
  1. `delivery = None` (Admission authority — stranding repaired).
  2. `consecutiveFailureCount++` (Adjudication authority).
  3. If counter reaches K → `resolveRecoveryObligation(id, Failed)` (Completion authority, terminal).
- The obligation's identity (id, buildSource, coalesce key) is unchanged; the obligation stays Live
  until K is reached.

### `delivery` after a publish failure (R16-A case)

Same as the transport build failure above. The Orchestrator's publish-failure branch at cpp:303
(currently `resolveRecoveryObligation(id, Failed)`) becomes `markTransientFailure(id)`. The
`destroyRolledBackDSP` call (cpp:287) is unchanged — it cleans up the new DSP that failed to
publish; it does not touch `delivery`.

### R17-2 conclusion

| Aspect | Decision |
|---|---|
| `delivery` writers | All 5 sites (W1–W5/W6) on CoordinatorLoop. **No new writers** are required for R16-A — the *P-B* mutation is a re-use of the existing `delivery = None` writer at W3/W5/W6, but invoked from a new caller. |
| `delivery` readers | All 4 sites (R1–R4) on CoordinatorLoop. **No new readers** are required. |
| P-B authority | Admission authority (CoordinatorLoop). Implemented via a new `markTransientFailure(id)` helper that wraps (1) `delivery = None`, (2) `consecutiveFailureCount++`, (3) optional `resolve(id, Failed)` at exhaustion. **Not** by mutating `resolveRecoveryObligation`'s `Retry` branch. |
| `Building` | Confirmed: durable slot sub-state, not a delivery state. No changes. |
| Stranded case | Repaired by P-B at the `markTransientFailure` site: the obligation's `delivery` is forced to `None` so the next `redriveDeferredRecoveryObligations` tick picks it up. |

---

## R17-3 — Failure counter: location, reset/increment rules, K value

### The two candidate counter locations

| Candidate | Where | Pros | Cons |
|---|---|---|---|
| **C1: Per-obligation counter on `LogicalRecoveryObligation`** | table slot | survives durable↔transport transitions, survives coalesce, identity-tracked, makes R17-7's state machine self-contained | requires extending `LogicalRecoveryObligation` struct; counter is a separate field from the existing `delivery` |
| **C2: Keep `kMaxRecoveryConsecutiveFailures` as a Builder-local counter** | `AudioEngine.RebuildDispatch.cpp:1015` (existing) | zero source change to `LogicalRecoveryObligation` | counter is **reset on the next Builder take**, so a redriven obligation that fails in the durable path and then in the transport path would not see a unified counter; counter is invisible to the transport-path failure sites in `trySubmitImpl` |

**The user explicitly asked which is correct.** The R17-3 analysis is that **C1 is the only
correct location** for R16-A. Reason:

- Under R16-A, the *same obligation* can fail in the durable path (Builder, `Building → DurablePending`)
  and then in the transport path (Orchestrator, `trySubmitImpl`) — these are *distinct* failure
  sites in *different* threads, and the counter must persist across them. C2's Builder-local counter
  cannot track transport-path failures.
- The C8 test's `Failed → ResolvedFailed → liveCount--` semantics remain valid as a *table-level
  test* (C8 exercises the table directly without going through any thread or counter). C1 is
  therefore backward-compatible.
- The counter is a sub-byte field (`uint8_t` is enough; K=4 fits in 3 bits), well within the
  D36.1 352B upper bound for `RecoveryAdmission`.

### The exact counter field and its atomicity

R16-A's spec:
- Field: `std::atomic<std::uint8_t> consecutiveFailureCount{0};` (relaxed ordering is sufficient —
  the only writer is the CoordinatorLoop on the `markTransientFailure` path; the only reader is
  the CoordinatorLoop on the exhaustion check).
- Init: 0 on `tryInsert` (W1) and on every terminal resolution that is *not* `Failed`
  (i.e. reset to 0 in the `Published`, `StaleSuperseded`, `ShutdownDiscarded` paths).
- Increment: by `markTransientFailure(id)` — single writer (CoordinatorLoop).
- Threshold check: by `markTransientFailure(id)` — single reader (CoordinatorLoop).
- K value: 4 (matching the existing `kMaxRecoveryConsecutiveFailures`).

The `kMaxRecoveryConsecutiveFailures` local constexpr in `AudioEngine.RebuildDispatch.cpp:1015` is
**retained as a parallel local counter** for the Builder's own spin-prevention (`:1036/:1058`); R18
must *not* remove it. The two counters serve different purposes:
- **Builder-local counter**: spin-prevention within a single `rebuildThreadLoop` iteration. Reset
  on success or on `break`-out.
- **Obligation-level counter**: retry budget for the obligation's lifetime. Reset only on terminal
  success (`Published`). Used to drive `Failed` exhaustion.

The two counters are **not duplicative** because the Builder-local counter can only see *durable*
failures (the `take` path), while the obligation-level counter sees *all* failures (durable, transport,
publish). The obligation-level counter is the one that gates `Failed` exhaustion; the
Builder-local counter is the one that gates `break`-out of the durable loop.

### Reset / increment / terminal table

| Outcome | counter | ΔL | terminal? | Reset reason |
|---|---|---|---|---|
| initial admission (`tryInsert`) | `0` (init) | +1 | No | slot is fresh |
| build failure (transport path) | `+1` | 0 | No | counter drives exhaustion |
| build failure (durable path) | `+1` | 0 | No | counter drives exhaustion |
| publish failure (transport path) | `+1` | 0 | No | counter drives exhaustion |
| publish failure (durable path) | `+1` | 0 | No | counter drives exhaustion |
| queue pressure (`RejectedPressure → Retry`) | `+0` (no change) | 0 | No | pressure is not a build/publish failure; counter untouched |
| successful publish (`Published`) | `0` (reset) | −1 | **Yes** | obligation completed; counter is meaningless post-terminal |
| stale (`StaleSuperseded`) | `0` (reset) | −1 | **Yes** | obligation completed (superseded by newer) |
| shutdown (`ShutdownDiscarded`) | `0` (reset, although irrelevant since slot is freed) | −1 | **Yes** | obligation closed by shutdown |
| `Retry` outcome at the `resolveRecoveryObligation` API (orchestrator-level) | `+1` (via `markTransientFailure`) | 0 | No | the *direct* `Retry` call increments; the obligation-level exhaustion check is part of `markTransientFailure` |
| retry exhaustion (counter == K) | `K` (frozen, then transition) | −1 | **Yes** | the only sanctioned path to `ResolvedFailed` |

**Reset is `0` on all terminal outcomes *except* `Failed`-exhaustion itself** — when exhaustion
fires, the slot is freed (`tryInsert` can pick it again later), and the next admission initializes a
fresh counter via the field's default member initializer (NSDMI `{0}`).

### The K value question

K=4 is inherited from the existing `kMaxRecoveryConsecutiveFailures = 4` constexpr
(AudioEngine.RebuildDispatch.cpp:1015). The value was chosen as "Builder spin prevention" — *not*
as an obligation-level retry budget. R16-A repurposes the same number as the obligation-level
exhaustion threshold.

**Does K=4 make sense as the obligation-level exhaustion threshold?** The obligation's retry budget
is the total number of build / publish failures (across both the transport and the durable path)
the obligation can sustain before being terminalized as `Failed`. K=4 means "5 attempts total
(1 initial + 4 retries)". At a build cycle of, say, 50-200ms (typical non-realtime build), 5
attempts take 250-1000ms. This is a reasonable upper bound for "transient" failures; beyond that,
the obligation is *not* transient and should be terminalized.

**Verdict**: K=4 is acceptable as the initial value, but R18 should *expose* it as a named
constant (e.g. `kMaxObligationConsecutiveFailures`) and place it next to the
`LogicalRecoveryObligation` struct so future tuning is easy. The value can be re-evaluated in R19+
based on empirical build-duration data.

### R17-3 conclusion

| Aspect | Decision |
|---|---|
| Counter location | **`LogicalRecoveryObligation::consecutiveFailureCount` (per-obligation, atomic, uint8_t).** |
| Builder-local counter | **Retained** (`kMaxRecoveryConsecutiveFailures` at RebuildDispatch.cpp:1015) for spin prevention; not duplicative with obligation-level counter. |
| K value | **4** (inherited, named `kMaxObligationConsecutiveFailures` for clarity). |
| Reset on | `Published`, `StaleSuperseded`, `ShutdownDiscarded`, and on slot reclaim (next `tryInsert` re-initializes the field). |
| Increment on | `markTransientFailure(id)` — the single new helper. |
| Coalesce behavior | Counter is on the slot, not on the identity; coalesce (which reuses the same slot) preserves the counter. **Identity is preserved across coalesce; counter is preserved across coalesce.** ✓ |
| Cross-resident path | Counter survives `Durable → None → Transport → Durable` transitions (redrive path). ✓ |
| `Retry` API call | `Retry` outcome at the `resolveRecoveryObligation` boundary becomes a *trigger* for `markTransientFailure` (see R17-7). The current "no-op" behavior is removed. |

---

## R17-4 — `Failed` producer restriction: only retry-exhaustion can produce `ResolvedFailed`

### Current fallthrough is the danger

`ISRRuntimePublicationCoordinator.cpp:957-960`:

```cpp
const ObligationState terminal = (outcome == Published)        ? ResolvedSuccess
                               : (outcome == StaleSuperseded) ? ResolvedStaleSuperseded
                               : (outcome == ShutdownDiscarded)? ShutdownDiscarded
                               :                                  ResolvedFailed;  // ★ fallthrough
```

The `else` branch maps any `RecoveryOutcome` value not in the explicit list (i.e. `Failed`, the
legacy `Superseded` which is not emitted, or any future enum value) to `ResolvedFailed`. This is
the *exact* mechanism that produces `ResolvedFailed` in R15-1's four production sites
(Orchestrator.cpp:189/255/303/386) and in the C8 test.

### R17-4's structural guarantee

R16-A + R17-4 require that **`RecoveryOutcome::Failed` is emitted by exactly one path**: the
`markTransientFailure(id)` helper's exhaustion check, which calls
`resolveRecoveryObligation(id, Failed)`. The four Orchestrator failure sites must call
`markTransientFailure(id)` *instead of* `resolveRecoveryObligation(id, Failed)`, and
`markTransientFailure` is the only place that can route to `Failed` (via the exhaustion check).

To make this **structurally enforced** (not just "by convention"), R18 will:
1. **Remove the `else → ResolvedFailed` fallthrough in `resolveRecoveryObligation`** and replace it
   with an assertion or a Debug-only `jassert` that the outcome is one of the four known values.
   Production builds that pass an unknown outcome will fall into a `std::abort` / `__debugbreak`
   rather than silently terminalizing as `Failed`.
2. **Keep `RecoveryOutcome::Failed` as a valid enum value** (it is still needed for the exhaustion
   path) but make it *only* reachable from the `markTransientFailure` helper. R18 will mark
   `resolveRecoveryObligation` callers' call sites with comments pointing to the new helper, and
   remove the four `Failed` direct call sites.
3. **Add a new telemetry counter `recoveryRetryExhaustedCount_`** (R16-8 already mentioned this)
   that the `markTransientFailure` exhaustion branch increments. The counter is the only way to
   observe the `Failed`-at-exhaustion path in production; `recoveryObligationShutdownDiscardCount_`
   remains the only counter for `ShutdownDiscarded`.

### How R17-4's structural guarantee is verified

- **Static check**: `rg 'resolveRecoveryObligation\(.*Failed\)' src/` should return exactly
  **two** call sites after R18 lands — one in `markTransientFailure` (the exhaustion path) and
  one in test C8 (the table-level unit test). Pre-R18 there are **five** call sites
  (Orchestrator.cpp:189/255/303/386 + C8) plus possibly the `else` fallthrough handling.
- **Compile-time check**: removing the `else` fallthrough in `resolveRecoveryObligation` and
  replacing it with `jassert` (Debug) or `std::abort` (Release) means any *future* code that
  passes `Failed` from a non-exhaustion path will fail the assertion. C8 still passes because it
  exercises the table directly via the `resolve(id, ResolvedFailed)` path, not through
  `resolveRecoveryObligation`.
- **Runtime check**: a new test T-R17-3 (see R17-7) verifies that the only way to observe
  `ResolvedFailed` in production is via the exhaustion path.

### R17-4 conclusion

| Aspect | Decision |
|---|---|
| Fallthrough in `resolveRecoveryObligation` | **Removed.** Replaced with `jassert` (Debug) / `__debugbreak` (Release) for unknown outcomes. |
| `Failed` producers in production | **Exactly one** — `markTransientFailure(id)`'s exhaustion branch. |
| `Failed` in tests | **C8** (table-level unit test) — preserved. |
| `ResolvedRetry` enum value | **Kept but unused** as a diagnostic state. R18 may opt to *use* it (`markTransientFailure` sets it on every transient failure) to distinguish "transient failure pending retry" from "Live waiting on a delivery resource". The R18 design may revisit this. |
| Telemetry | **New** `recoveryRetryExhaustedCount_` to distinguish exhaustion `Failed` from any historical `Failed`. |

---

## R17-5 — I4 amendment: where does `RetryExhausted` go?

### Current I4 ownership-conservation invariant

I4-D15.2 (lines 178-203 of the I4 design contract):

> ```
> ownership conservation:
>     transportCount + durableCount + buildingCount + stalledCount
>         + supersededCount + shutdownDiscardCount
>         == admittedLogicalObligationCount
>
> （terminal-failure は消失理由に含めない — D14.3。Debug assert のみ）
> ```

The wording "terminal-failure は消失理由に含めない" is the **exact clause** that R16-A must
amend. The amendment is a **single-line addition** that adds `retryExhaustedCount` to the
conservation equation.

### Proposed amendment text (for I4 sign-off, not in this audit's scope)

```
ownership conservation:
    transportCount + durableCount + buildingCount + stalledCount
        + supersededCount + shutdownDiscardCount
        + retryExhaustedCount
        == admittedLogicalObligationCount

（terminal-failure は消失理由に含めない — D14.3。D17 amendment: retryExhausted は
  `ResolvedFailed` で表され、obligation-level counter `kMaxObligationConsecutiveFailures=4` 到達
  時の唯一の sanctioned terminal として出現する。 transient failure による任意 Failed は禁止。）
```

### Where in I4 to insert

The amendment is structurally a **refinement of D15.2** and a **clarification of D14.3** (which
already says "terminal-failure は disappearance 理由に含めない — Debug assert のみ"). R17-5
recommends:

- **D14.3 (budget exhaustion = backpressure)**: add a footnote that *transient* build/publish
  failure does not consume the backpressure budget (it stays Live and is redriven). The
  retry-exhaustion case is the only path that can terminalize an obligation as a non-disappearance
  reason.
- **D15.2 (ownership conservation)**: add `retryExhaustedCount` to the equation. Keep the "terminal-
  failure は消失理由に含めない" clause for *non-exhaustion* `Failed` — the R17-4 structural
  guarantee makes this clause a *property* of the runtime rather than a *requirement* on it.
- **D29.8 (end-to-end state machine)**: update the "Settlement (Builder)" branch to add
  `MARK-TRANSIENT-FAILURE` as a non-terminal transition (analogous to the existing
  `TERMINAL` line). This is a *new* settlement action that does not decrement `L` and does not
  close the episode; it only mutates the obligation's `consecutiveFailureCount` and may force
  `delivery = None`.

### What is *not* required

- I4 does **not** need to define a new `ObligationState` enum value. `ResolvedFailed` already
  exists (h:275); R16-A only changes the *semantics* of "when `ResolvedFailed` is reached" to
  "only at retry-exhaustion".
- I4 does **not** need to redefine `kMaxRecoveryConsecutiveFailures`. R16-A reuses the value
  4 as `kMaxObligationConsecutiveFailures` (named separately for clarity; same value).
- I4 does **not** need to add a new LP. The exhaustion path uses the existing LP-D
  (TERMINAL transition from D30.6) with the precondition "consecutiveFailureCount == K".

### R17-5 conclusion

The I4 amendment is a **single-line addition to D15.2** plus a **clarification footnote in
D14.3** plus a **state-machine line in D29.8**. No new invariants, no new LPs, no new state
values. The amendment is the minimum surface area consistent with R16-A's design.

---

## R17-6 — C8 is a low-level table idempotency test, distinct from T-R16

### What C8 actually tests

`ISRSemanticValidationTests.cpp:1028-1044`:

```cpp
[[nodiscard]] bool testRLOE_C8_shutdownRaceOnce()
{
    auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    auto id = submitAndGetId(*c, DSPHandle::null(), 44);
    const std::uint64_t sdBefore = c->recoveryObligationShutdownDiscardCount();
    c->resolveRecoveryObligation(*id, RecoveryOutcome::Failed);                         // ★ direct table call
    c->resolveRecoveryObligation(*id, RecoveryOutcome::ShutdownDiscarded);             // ★ idempotent no-op
    if (c->liveLogicalRecoveryObligationCount() != 0) return false;                    // ★ L=0 assertion
    if (c->recoveryObligationShutdownDiscardCount() != sdBefore) return false;         // ★ sdCount unchanged
    // direct shutdown-discard path increments exactly once
    auto id2 = submitAndGetId(*c, DSPHandle::null(), 45);
    c->resolveRecoveryObligation(*id2, RecoveryOutcome::ShutdownDiscarded);
    return c->liveLogicalRecoveryObligationCount() == 0
        && c->recoveryObligationShutdownDiscardCount() == sdBefore + 1;
}
```

C8 tests **the `RecoveryAdmissionTable::resolve` function directly**:
- It calls `resolveRecoveryObligation(id, Failed)` and asserts that L goes to 0 and the
  shutdown-discard counter is *not* incremented.
- It calls `resolveRecoveryObligation(id, ShutdownDiscarded)` and asserts the table accepts the
  terminal (L goes to 0) and increments the shutdown-discard counter.
- The two `resolve` calls on the same id are *idempotent* (the second is a no-op).

### C8's two-layer relationship

| Layer | Test | What it asserts | R16-A's status |
|---|---|---|---|
| **Table layer** | C8 | `RecoveryAdmissionTable::resolve` is idempotent, correctly transitions `Live→terminal`, and that `Failed` does *not* increment `recoveryObligationShutdownDiscardCount`. | **Preserved unchanged.** C8 still passes under R16-A: the table still maps `Failed` to `ResolvedFailed` and still does not increment the shutdown-discard counter for `Failed` (only `ShutdownDiscarded` does, cpp:962). |
| **Production layer** | T-R16-1..7 (R16-8) | The runtime *path* that produces `ResolvedFailed` is exclusively the retry-exhaustion path; transient failures stay Live; the obligation is redriven. | **New.** T-R17-1..7 (R17-7) document and verify this layer. |

### Why C8 is *not* invalidated by R16-A

- R16-A changes the **production call sites** of `resolveRecoveryObligation(id, Failed)` — those
  are now `markTransientFailure(id)` calls that *may* reach `Failed` (at exhaustion) or may not
  (transient case).
- R16-A does **not** change the `RecoveryAdmissionTable::resolve` function. C8 directly exercises
  the table's terminal-CAS-and-decrement logic. The table-level behavior is identical.
- R16-A does **not** change `resolveRecoveryObligation`'s `Failed` arm. C8 still calls
  `resolveRecoveryObligation(id, Failed)` and still gets `ResolvedFailed` + `liveCount_--`.
- R16-A's structural guarantee (R17-4) is *in the production code*, not in the table. C8 does
  not assert anything about the production code path; it asserts table-level behavior. C8 is
  *compatible* with the R16-A design at the table layer.

### C8's continued relevance

- It is the **only** test that exercises the `Failed` terminal via the `resolveRecoveryObligation`
  public API. The table-level `Failed` path is otherwise unreachable in production (R17-4's
  structural guarantee). Removing C8 would leave the `Failed` arm of `resolveRecoveryObligation`
  *uncovered by tests*, which would be a regression. **C8 is kept.**
- A future R18 implementation should add a comment to C8 explaining that the test is *only* a
  table-level unit test, and that the production path is exercised by T-R17-3 (R17-7).

### R17-6 conclusion

| Aspect | Decision |
|---|---|
| C8 status | **Kept unchanged** (table-level unit test, table layer only). |
| R17-4's fallthrough removal | R18 may need to adjust C8 if the fallthrough removal makes `RecoveryOutcome::Failed` an unrecognized value at the API. R18 will *keep* the `Failed` enum value in `resolveRecoveryObligation`'s switch (it maps to `ResolvedFailed`) and *only* add a `default:` branch that asserts. C8 continues to pass. |
| T-R17-1..7 | **New tests** (R17-7) for the production semantic layer. |

---

## R17-7 — Final state machine table (the spec R18 must implement)

This is the artifact the user asked for. Each row is a single transition with the full
authority / linearization / reader / writer / wakeup specification.

### 7.1 — Field-level state machine

The obligation has three independent fields that participate in R16-A's design:
1. `state` ∈ `{NoObligation, Live, ResolvedSuccess, ResolvedFailed, ResolvedStaleSuperseded, ResolvedRetry, ShutdownDiscarded}`.
2. `delivery` ∈ `{None, Transport, Durable}`.
3. `consecutiveFailureCount` ∈ `{0..K}` (where K=4).

The durable slot has its own state machine:
4. `pendingRecoveryAdmission_.state` ∈ `{NoAdmission, DurablePending, Building}`.

R16-A's specification touches all four. The transitions below are listed by **trigger event**
(what the runtime is doing), not by the field being mutated.

### 7.2 — Per-event transition table

Notation:
- **Authority**: who is allowed to perform the mutation per I4-D29.8.
- **LP (linearization point)**: the exact point at which the mutation is observed (D30.6 LP-A/B/C/D).
- **Reader**: the next read that observes the new value (proves LP is in the right place).
- **Writer**: the code site that performs the mutation.
- **Wakeup**: the mechanism (if any) that resumes the obligation's next attempt.

---

#### Row 1 — Initial admission (transport)

| Field | Before | After |
|---|---|---|
| `state` | `NoObligation` (slot fresh) | `Live` |
| `delivery` | `None` (slot fresh) | `Transport` |
| `consecutiveFailureCount` | 0 (slot fresh) | 0 |
| `liveCount_` | N | N+1 |
| terminal? | No | No |

- **Authority**: Admission (CoordinatorLoop).
- **LP**: **LP-A** (D30.6 first admission). The `tryInsert` (h:343) and the
  `recoveryIntentQueue_.push` (cpp:909) are **not** in the same atomic step today (R18 will not
  change this — they are *separately* linearized but SPSC-safe). The table's `state.store(Live)`
  is the LP for `state`/`delivery`/`counter`; the queue's `push` is the LP for the transport
  residency.
- **Reader**: `redriveDeferredRecoveryObligations` (cpp:982) sees the new `Live && delivery==Transport`
  in subsequent scans; the Builder's `popRecoveryRequest` (cpp:1121) sees the new intent in the queue.
- **Writer**: `RecoveryAdmissionTable::tryInsert` (h:343) + `submitRecoveryRequest` transport branch
  (cpp:909-911).
- **Wakeup**: `submitRecoveryRequest` returns `true`; the caller (`submitRecoveryIntent` in
  AudioEngine) calls `RebuildThread.wake()` to ensure the Builder sees the new intent.

#### Row 2 — Initial admission (durable, when transport queue is full)

| Field | Before | After |
|---|---|---|
| `state` | `NoObligation` | `Live` |
| `delivery` | `None` | `Durable` |
| `consecutiveFailureCount` | 0 | 0 |
| `pendingRecoveryAdmission_.state` | `NoAdmission` | `DurablePending` |
| `liveCount_` | N | N+1 |

- **Authority**: Admission (CoordinatorLoop).
- **LP**: **LP-A** (table insertion). The `pendingRecoveryAdmission_` mutation is a *separate*
  field on the CoordinatorLoop's own state; it is SPSC-safe with the Builder.
- **Reader**: Builder's `takePendingRecoveryAdmission` (cpp:1064) sees the new `DurablePending`.
- **Writer**: `tryInsert` + `submitRecoveryRequest` durable branch (cpp:929-941).
- **Wakeup**: `recoveryAdmissionPending_` is set; Builder's existing wake mechanism picks it up
  on the next loop iteration.

#### Row 3 — Initial admission (deferred, when both transport and durable are busy)

| Field | Before | After |
|---|---|---|
| `state` | `NoObligation` | `Live` |
| `delivery` | `None` | `None` |
| `consecutiveFailureCount` | 0 | 0 |
| `liveCount_` | N | N+1 |

- **Authority**: Admission (CoordinatorLoop).
- **LP**: **LP-A** (table insertion only — no delivery mutation).
- **Reader**: `redriveDeferredRecoveryObligations` (cpp:982) sees the new `Live && delivery==None`
  and attempts re-attachment in the same `submitRecoveryRequest` call.
- **Writer**: `tryInsert` + `submitRecoveryRequest` deferred branch (cpp:923-928).
- **Wakeup**: same `submitRecoveryRequest` call invokes `redriveDeferredRecoveryObligations`
  immediately (cpp:847); the obligation is re-driven synchronously.

---

#### Row 4 — Build failure (transport-resident obligation)

| Field | Before | After |
|---|---|---|
| `state` | `Live` | `Live` |
| `delivery` | `Transport` | `None` |
| `consecutiveFailureCount` | C | C+1 |
| `liveCount_` | N | N (ΔL=0) |
| terminal? | No | No (unless C+1 == K) |

- **Authority**: Adjudication (CoordinatorLoop) — single new helper `markTransientFailure(id)`.
  This is *not* the Admission authority (no new id) and *not* the Completion authority (no
  terminal). It is a **new, dedicated adjudication authority** that I4-D29.8 does not yet name
  (R18 will add it to I4-D29.8 as a Settlement sub-authority).
- **LP**: **LP-E** (new) — the `markTransientFailure` helper's CAS on `consecutiveFailureCount`.
  This is *separate* from LP-A/B/C/D; it is the obligation-level state machine's *non-terminal
  transition*. R18 will document LP-E in I4-D30.6.
- **Reader**: `redriveDeferredRecoveryObligations` (cpp:982) sees `delivery==None` and re-attaches.
- **Writer**: New `markTransientFailure(id)` (R18 introduces). It performs:
  ```cpp
  void RuntimeIntentCoordinator::markTransientFailure(uint64_t id) {
      for (i : slots) {
          if (slots[i].id != id) continue;
          if (slots[i].state != Live) return;          // not Live → no-op
          slots[i].delivery = None;                    // P-B
          auto c = slots[i].consecutiveFailureCount.fetch_add(1, acq_rel) + 1;
          if (c >= kMaxObligationConsecutiveFailures) {
              // exhaustion: route to Failed terminal
              resolveRecoveryObligation(id, Failed);
          }
          return;
      }
  }
  ```
- **Wakeup**: `redriveDeferredRecoveryObligations` is invoked from
  `submitRecoveryRequest` (cpp:847) and from `runCoordinatorPhase` (R5-10 trigger). The obligation
  is re-driven on the next CoordinatorLoop tick (≤1ms).

#### Row 5 — Build failure (durable-resident obligation, Builder path)

| Field | Before | After |
|---|---|---|
| `state` | `Live` | `Live` |
| `delivery` | `Durable` | `Durable` |
| `consecutiveFailureCount` | C | C+1 |
| `pendingRecoveryAdmission_.state` | `Building` | `DurablePending` |
| `liveCount_` | N | N |
| terminal? | No | No (unless C+1 == K) |

- **Authority**: Adjudication (Builder thread) — calls the Coordinator's `markTransientFailure(id)`
  via a new entry point. The Builder *does not* mutate the obligation's `consecutiveFailureCount`
  directly (R18 forbids this — single-writer invariant).
- **LP**: **LP-F** (new) — the **durable slot's `Building → DurablePending` transition
  (via `settlePendingRecoveryAdmission(true)`) AND the obligation's `consecutiveFailureCount++`
  (via `markTransientFailure(id)`) are **two independent linearizations**. They are **not in
  the same atomic step** (the slot is plain-memory SPSC, the counter is atomic). The slot's
  `state.store(DurablePending)` is observed by the next `take`; the counter's `fetch_add` is
  observed by the next `markTransientFailure` call. R18 will document this dual-LP in I4-D30.6.
- **Reader**: Builder's next `takePendingRecoveryAdmission` (cpp:1064) sees `DurablePending` and
  retries; the next `markTransientFailure` call sees the incremented counter.
- **Writer**: `settlePendingRecoveryAdmission(true)` (cpp:1107-1114) for the slot, and
  `markTransientFailure(id)` for the counter.
- **Wakeup**: Builder's next loop iteration picks the durable slot up.

**Important**: for the **durable-resident** case, the existing `kMaxRecoveryConsecutiveFailures = 4`
Builder-local counter and the **obligation-level** counter are *both* incremented. R18 must ensure
that:
- the Builder-local counter still gates `break`-out of the durable loop (existing behavior);
- the obligation-level counter drives the *exhaustion* path (`resolveRecoveryObligation(id, Failed)`).
- The two counters are *not* the same value: the obligation-level counter persists across durable↔
  transport transitions, the Builder-local counter is reset every `rebuildThreadLoop` iteration.

The simplest correct implementation is: `markTransientFailure` is called by the Builder at the same
call sites as the existing `settlePendingRecoveryAdmission(true)` (`:1034`, `:1056`). R18 must
call *both* in lockstep.

#### Row 6 — Publish failure (transport-resident obligation, Orchestrator path)

| Field | Before | After |
|---|---|---|
| `state` | `Live` | `Live` |
| `delivery` | `Transport` | `None` |
| `consecutiveFailureCount` | C | C+1 |
| `liveCount_` | N | N |
| terminal? | No | No (unless C+1 == K) |

- **Authority**: Adjudication (Orchestrator thread, called from `trySubmitImpl`) — single writer
  `markTransientFailure(id)`.
- **LP**: **LP-E** (same as Row 4).
- **Reader**: `redriveDeferredRecoveryObligations` (cpp:982).
- **Writer**: New `markTransientFailure(id)` called from `trySubmitImpl` at Orchestrator.cpp:303
  (replacing the current `resolveRecoveryObligation(req.recoveryObligationId, Failed)`).
- **Wakeup**: same as Row 4.

#### Row 7 — Publish failure (durable-resident obligation, Orchestrator path)

This case is **structurally impossible** under the current code. A `delivery==Durable` obligation
has its slot in `DurablePending` or `Building`. The Orchestrator's `trySubmitImpl` is *not* invoked
for durable-resident obligations — the Builder's `take → build → publish → settle` path runs
*instead*. The Orchestrator's `executor_.publish` call (cpp:276) is the Builder's path's final
step; on failure the Builder calls `settlePendingRecoveryAdmission(true)` (analogous to Row 5)
and then `markTransientFailure(id)`. **Row 7 is a no-op case under R16-A**: the durable-resident
publish failure routes through Row 5, not Row 7.

#### Row 8 — Queue pressure (`RejectedPressure`)

| Field | Before | After |
|---|---|---|
| `state` | `Live` | `Live` |
| `delivery` | Transport/Durable/None | unchanged |
| `consecutiveFailureCount` | C | C (no change) |
| `liveCount_` | N | N |
| terminal? | No | No |

- **Authority**: Completion (CoordinatorLoop) — `resolveRecoveryObligation(id, Retry)` is a
  complete no-op (current behavior). The `rearmRecoveryRetry` (cpp:968-975) is a *separate* call
  from the Orchestrator at cpp:398; it only re-arms the durable slot if it already holds this
  obligation in `Building` state.
- **LP**: none — no obligation mutation.
- **Reader**: n/a.
- **Writer**: n/a.
- **Wakeup**: existing R5-10 `redriveDeferredRecoveryObligations` continues to find the obligation
  in the `Live && delivery==None` state and re-attach.

**R16-A change for Row 8**: **none**. Pressure is not a build/publish failure; the counter is
unchanged. R18 preserves the current `RejectedPressure → Retry` path as-is.

#### Row 9 — Successful publish (`Published`)

| Field | Before | After |
|---|---|---|
| `state` | `Live` | `ResolvedSuccess` |
| `delivery` | Transport/Durable/None | irrelevant (terminal) |
| `consecutiveFailureCount` | C | 0 (reset) |
| `liveCount_` | N | N−1 |
| terminal? | No | **Yes** |

- **Authority**: Completion (ISR for Route B at Orchestrator.cpp:346; Orchestrator thread for
  Route A at Orchestrator.cpp:312).
- **LP**: **LP-D** (D30.6 terminal transition).
- **Reader**: `liveLogicalRecoveryObligationCount()` (h:426) returns the new value.
- **Writer**: `RecoveryAdmissionTable::resolve` (h:366) with `ObligationState::ResolvedSuccess`.
  R18 will additionally reset `consecutiveFailureCount` to 0 in the same atomic step (single
  field extension — counter is on the slot, CAS'd together with `state`).
- **Wakeup**: none (terminal).

**R16-A change for Row 9**: counter reset to 0 on success. This is the only "reset" branch
required (other resets are on terminal outcomes that free the slot).

#### Row 10 — Stale (`StaleSuperseded`)

| Field | Before | After |
|---|---|---|
| `state` | `Live` | `ResolvedStaleSuperseded` |
| `delivery` | any | irrelevant (terminal) |
| `consecutiveFailureCount` | C | 0 (reset, although slot is freed) |
| `liveCount_` | N | N−1 |
| terminal? | No | **Yes** |

- **Authority**: Completion (CoordinatorLoop, from `submitPublishRequest` at Orchestrator.cpp:377).
- **LP**: **LP-D**.
- **Reader**: as Row 9.
- **Writer**: `RecoveryAdmissionTable::resolve` (h:366) with `ObligationState::ResolvedStaleSuperseded`.
- **Wakeup**: none (terminal).

**R16-A change for Row 10**: counter reset to 0 (defensive; slot is freed anyway).

#### Row 11 — Shutdown (`ShutdownDiscarded`)

| Field | Before | After |
|---|---|---|
| `state` | `Live` | `ShutdownDiscarded` |
| `delivery` | any | irrelevant (terminal) |
| `consecutiveFailureCount` | C | 0 (reset, slot is freed) |
| `liveCount_` | N | N−1 |
| terminal? | No | **Yes** |

- **Authority**: Completion (CoordinatorLoop, from `discardRecoveryRequestsOnShutdown` at
  ISRRuntimePublicationCoordinator.cpp:1149).
- **LP**: **LP-D** (and the table-scan is *the* last terminal disposition for the obligation,
  per D11 INV-X1-7).
- **Reader**: `liveLogicalRecoveryObligationCount()` (h:426) returns 0 post-shutdown; D105-R13's
  `isFullyDrained` predicate asserts this.
- **Writer**: `RecoveryAdmissionTable::resolve` (h:366) with `ObligationState::ShutdownDiscarded`.
- **Wakeup**: none (terminal).

**R16-A change for Row 11**: counter reset to 0 (defensive; slot is freed).

#### Row 12 — Retry exhaustion (counter reaches K)

| Field | Before | After |
|---|---|---|
| `state` | `Live` | `ResolvedFailed` |
| `delivery` | None | irrelevant (terminal) |
| `consecutiveFailureCount` | K | 0 (reset, slot is freed) |
| `liveCount_` | N | N−1 |
| terminal? | No | **Yes** (the only sanctioned `Failed` path) |

- **Authority**: Adjudication (CoordinatorLoop or Builder thread, via `markTransientFailure`).
- **LP**: **LP-D** (the terminal transition is the *same* LP as `Published`/`StaleSuperseded`/
  `ShutdownDiscarded`; the *trigger* is the counter, but the *terminal* is a regular
  `resolveRecoveryObligation(id, Failed)` call). The exhaustion path is:
  ```
  markTransientFailure(id):
      delivery = None
      counter++ == K
      → resolveRecoveryObligation(id, Failed)  // single -1
      → telemetry: recoveryRetryExhaustedCount_++
  ```
- **Reader**: `liveLogicalRecoveryObligationCount()` decrements.
- **Writer**: `markTransientFailure` calls `resolveRecoveryObligation(id, Failed)` which calls
  `RecoveryAdmissionTable::resolve` (h:366) with `ObligationState::ResolvedFailed`. The
  `recoveryRetryExhaustedCount_` counter is incremented.
- **Wakeup**: none (terminal).

**R16-A change for Row 12**: this is the *new* terminal. R17-4's structural guarantee is
satisfied because this is the *only* call site of `resolveRecoveryObligation(id, Failed)` in
production code (plus C8's table-level test).

---

### 7.3 — Coalesce behavior (cross-cutting)

| Aspect | Decision |
|---|---|
| Coalesce during transient failure | The same `Live && delivery==X` obligation is matched by `findByKey` (h:333). The new `submitRecoveryRequest` re-uses the same slot, **same id, same counter**. The counter is *not* reset on coalesce (the obligation is in retry; resetting the counter would be a regression). |
| Coalesce while deferred (`delivery==None`) | The C16 single-representation invariant (R5-10 §2) still holds: the early-return at cpp:877-878 prevents a double push. |
| Coalesce at exhaustion | **Impossible**: `ResolvedFailed` is terminal; `findByKey` matches only `Live` (h:335). A new admission for the same coalesce key creates a new obligation with a new id (ΔL=+1). |
| Coalesce at `Live && delivery==Durable`, durable slot held by *this* obligation | Submitting a new attempt for the same target *coalesces* onto the existing slot (ΔL=0), and `submitRecoveryRequest` may push a transport representation **or** leave the durable slot as-is (R18 will preserve the existing precedence: durable > transport, but if the durable slot already holds the obligation, no double-write occurs). |

---

### 7.4 — Transport-resident stranded case repair (P-B)

The R16-3 / R16-4 concern (a `Live && delivery==Transport` obligation whose transport intent was
popped by the Builder, but the publish failed) is repaired by Row 4 / Row 6:
- Before the repair, the obligation was stranded (`Live && delivery==Transport`, no intent in
  the queue, no redrive target).
- After the repair, the obligation is `Live && delivery==None` and the next
  `redriveDeferredRecoveryObligations` picks it up.

The repair is *mandatory*: without it, the stranded obligation leaks (L stays 1, no resolution
path). R18 must implement Row 4 / Row 6's `delivery = None` reset for the stranded case to be
repaired.

---

### 7.5 — Test design catalog (T-R17-1..7)

The R16-8 catalog is the production-layer counterpart to C8. R17 refines it with explicit
expectations on the counter and on the exhaustion path.

| ID | Precondition | Action | Assertion |
|---|---|---|---|
| **T-R17-1** | empty coordinator | submit one obligation; simulate a build failure in `trySubmitImpl` | obligation stays `Live`; `delivery==None`; `consecutiveFailureCount==1`; `liveCount==1` |
| **T-R17-2** | as T-R17-1 | one `redriveDeferredRecoveryObligations` call | obligation is now `Live && delivery==Transport` (or `Durable` if slot is free); `consecutiveFailureCount==1`; `liveCount==1` |
| **T-R17-3** | as T-R17-2 | Builder takes the intent, calls `markTransientFailure` (build failure) | obligation stays `Live`; `consecutiveFailureCount==2`; `liveCount==1` |
| **T-R17-4** | as T-R17-3 | repeat T-R17-2 + T-R17-3 two more times (counter reaches K=4) | on the K-th call, `markTransientFailure` routes to `resolveRecoveryObligation(id, Failed)`; obligation becomes `ResolvedFailed`; `liveCount==0`; **`recoveryRetryExhaustedCount==1`**; `recoveryObligationShutdownDiscardCount` *unchanged* (Failed is not ShutdownDiscarded, same as C8) |
| **T-R17-5** | counter == K-1 | successful publish | obligation becomes `ResolvedSuccess`; `consecutiveFailureCount` reset to 0; `liveCount==0`; **`recoveryRetryExhaustedCount==0`** (no exhaustion) |
| **T-R17-6** | T-R17-1 through T-R17-4 with shutdown interleaved | `requestShutdown` between two failures | the obligation is `ShutdownDiscarded` (not `Failed`); `recoveryObligationShutdownDiscardCount==1`; `recoveryRetryExhaustedCount==0`; `isFullyDrained==true` post-discard (D105-R13) |
| **T-R17-7** | T-R17-1 with coalesce | submit a second obligation with the same coalesce key while the first is `Live && delivery==None` after T-R17-1 | ΔL=0; the second submission *coalesces* onto the same slot; `consecutiveFailureCount` preserved (not reset) |
| **T-R17-8** | T-R17-1 with stale | T-R17-1 + `RejectedStaleGeneration` | the obligation becomes `ResolvedStaleSuperseded`; `consecutiveFailureCount` reset; `liveCount==0`; **`recoveryRetryExhaustedCount==0`** (stale is not exhaustion) |

All 8 tests use the public API only and follow the R5-10 / R16 test patterns.

---

## R17-8 — Final verdict (R17-A GO)

### R17-A: R16-A is **fully specifiable** as an implementation contract

The 8 GO criteria from the user's R17 brief, verified against R17-1..7:

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | transient `Failed` producer vanishes from production code | **✅ specifiable** | R17-4: the only `resolveRecoveryObligation(id, Failed)` call site in production is `markTransientFailure`'s exhaustion branch. The four Orchestrator sites become `markTransientFailure(id)` calls. |
| 2 | `Retry` is ΔL=0 | **✅ specifiable** | R17-7 Rows 4/5/6/8: `markTransientFailure` does not touch `liveCount_`; the table stays `Live`. |
| 3 | same `obligationId` maintained | **✅ specifiable** | `markTransientFailure` looks up the slot by id (not by index) and mutates the existing slot. The id is never re-issued. |
| 4 | no stranded state in Transport / Durable | **✅ specifiable** | R17-2 / R17-7 Rows 4/6: `delivery = None` reset on transient failure. |
| 5 | durable lease retry and obligation retry counter are not duplicative | **✅ specifiable** | R17-1 / R17-3: durable lease is on `PendingRecoveryAdmission::State` (transport-side); obligation counter is on `LogicalRecoveryObligation` (logical-side). The Builder-local `kMaxRecoveryConsecutiveFailures=4` is retained for spin prevention; the obligation-level `kMaxObligationConsecutiveFailures=4` is the new exhaustion gate. |
| 6 | retry exhaustion is the only path to `ResolvedFailed` | **✅ specifiable** | R17-4: fallthrough removed; only `markTransientFailure`'s exhaustion branch emits `Failed`. |
| 7 | `Failed` `−1` happens once at exhaustion | **✅ specifiable** | R17-7 Row 12: `markTransientFailure` calls `resolveRecoveryObligation(id, Failed)` *only* when counter reaches K; the table's `resolve` is idempotent (lost CAS = no-op). |
| 8 | `Success / Superseded / ShutdownDiscard` semantics unchanged | **✅ specifiable** | R17-7 Rows 9/10/11: these rows are *unchanged* by R16-A; only the counter is reset on success. |
| 9 | `kMaxLogicalRecoveryObligations = 32` maintained | **✅ specifiable** | R17-7 all rows: `liveCount_` is incremented only at `tryInsert` (Rows 1/2/3) and decremented only at terminal `resolve` (Rows 9/10/11/12). No row increases the count beyond 32 (the existing `tryInsert` capacity check at h:344 enforces the bound). |
| 10 | I4 amendment text and runtime state machine are 1:1 | **✅ specifiable** | R17-5: the I4 amendment is a single line in D15.2 plus a D14.3 footnote plus a D29.8 settlement sub-authority. The runtime state machine (R17-7) has 12 rows that map 1:1 to the I4 enumeration. |

### R17 NO-GO criteria, verified as not present

| # | NO-GO criterion | Status | Why it is not present |
|---|---|---|---|
| N1 | `delivery` does not become `None` | ✅ fixed | R17-2 / R17-7 Rows 4/6: `markTransientFailure` sets `delivery = None` atomically with `consecutiveFailureCount++`. |
| N2 | durable lease and obligation counter duplicate meaning | ✅ fixed | R17-1 / R17-3: separate state machines, separate counters, separate responsibilities. |
| N3 | `Failed` reachable from non-exhaustion path | ✅ fixed | R17-4: fallthrough removed, only `markTransientFailure` emits `Failed`. |
| N4 | `ResolvedFailed` remains "any failure extinction" | ✅ fixed | R17-4 / R17-5 / R17-7 Row 12: `ResolvedFailed` *is* retry-exhaustion; no other path. |
| N5 | transport failure loses obligation | ✅ fixed | R17-2 / R17-7 Row 6: `delivery = None` + counter++ + redrive recovers the stranded obligation. |
| N6 | exhaustion reuses id | ✅ impossible | R17-7 Row 12: `ResolvedFailed` is terminal; `findByKey` only matches `Live`; new admission gets a new id. |
| N7 | coalesce breaks counter / identity | ✅ fixed | R17-7 §7.3: coalesce is a no-op for counter and identity (same slot, same id, same counter). |

### R17-A: GO

R16-A is **ready for implementation as R18**. The 12-row state machine in R17-7 is the contract
R18 must implement. The single helper `markTransientFailure(id)` is the only new public API on
`RuntimeIntentCoordinator`; the only new field on `LogicalRecoveryObligation` is
`consecutiveFailureCount`; the only I4 amendment is a single line in D15.2.

R18's implementation steps (for traceability, not for execution in R17):
1. Add `std::atomic<std::uint8_t> consecutiveFailureCount{0};` to `LogicalRecoveryObligation` (h:310).
   Update the D36.1 352B upper-bound estimate (still holds; +1B → 353B).
2. Add `markTransientFailure(uint64_t id)` to `RuntimeIntentCoordinator` (R17-7 Rows 4/5/6).
3. Update `RecoveryAdmissionTable::resolve` (h:366) to also reset `consecutiveFailureCount` to 0
   on success (Row 9) and on terminal (Rows 10/11/12). The reset is *part of* the LP-D atomic
   transition.
4. Update the four Orchestrator failure sites (cpp:189/255/303/386) to call
   `markTransientFailure(req.recoveryObligationId)` instead of
   `resolveRecoveryObligation(req.recoveryObligationId, Failed)`. The cpp:386 site is a switch
   route that may need careful handling (only `RejectedNotFinalized` cases that *originated from
   the orchestrator* go to `markTransientFailure`; caller-side `RejectedNotFinalized` due to
   other reasons continues to use `Retry` for the pressure path).
5. Update the Builder's durable-failure sites (cpp:1034/1056) to also call
   `markTransientFailure(recovery->obligationId)` alongside the existing
   `settlePendingRecoveryAdmission(true)`. R17-7 Row 5.
6. Remove the `else → ResolvedFailed` fallthrough in `resolveRecoveryObligation` (cpp:960) and
   replace with a `default:` branch that asserts (R17-4). The `Failed` enum value remains valid
   (it is now reached only via `markTransientFailure`'s exhaustion branch).
7. Add the new telemetry counter `recoveryRetryExhaustedCount_` and increment it in the
   exhaustion branch.
8. Add tests T-R17-1..8 (R17-7 §7.5).
9. Update the I4 design contract: D14.3 footnote, D15.2 line, D29.8 settlement sub-authority
   (R17-5).
10. Update the D36.1 sizeof estimate (RecoveryAdmission now 353B instead of 352B; B_logical_max
    becomes 32 × 353B ≈ 11.3KB; B_total_max increase is 32B; well within the D35.5 64MB
    admissibility threshold).

### R17 deliverables summary

- **R17-1**: Two retry mechanisms (durable lease, `Retry` outcome) are complementary; B becomes
  active; A unchanged. (Spec done.)
- **R17-2**: 6 writers, 4 readers, all on CoordinatorLoop. P-B authority is Adjudication
  (`markTransientFailure`), not Completion. (Spec done.)
- **R17-3**: Counter on `LogicalRecoveryObligation` (per-obligation, atomic, uint8_t). K=4.
  Builder-local counter retained. Reset table defined. (Spec done.)
- **R17-4**: Fallthrough removed; `Failed` only via `markTransientFailure`'s exhaustion branch.
  C8 compatible. (Spec done.)
- **R17-5**: I4 amendment is a single line in D15.2 + D14.3 footnote + D29.8 settlement
  sub-authority. (Spec done.)
- **R17-6**: C8 kept (table-level). T-R17-1..8 added (production layer). (Spec done.)
- **R17-7**: 12-row state machine table with full authority / LP / reader / writer / wakeup. (Spec
  done.)
- **R17-8**: All 10 GO criteria specifiable. All 7 NO-GO criteria absent. **R17-A GO** for R18
  implementation. (Verdict done.)
