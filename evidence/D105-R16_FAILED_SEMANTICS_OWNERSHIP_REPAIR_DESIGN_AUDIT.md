# D105-R16 — Failed Semantics / Ownership Contract Repair Design Audit

**Status:** AUDIT COMPLETE — **R16-A: Retry-preserving contract (recommended, BLOCKED on user sign-off).**
**Source changes: 0 (read-only).** R15-A confirmed `RecoveryOutcome::Failed` is a true logical-obligation
extinction (`Live → ResolvedFailed`, single `−1` authority, no successor obligation) that is **not** in
I4's disappearance set `{Success, Superseded, ShutdownDiscard}`. R16 decides *what should happen*; the
implementation is deferred to R17.

Goal: pick the semantic that the runtime **should** implement for the four `Failed` producers
(`RuntimePublicationOrchestrator.cpp:189 / 255 / 303 / 386`), so that the I4 ownership contract and
the runtime code converge on a single, defensible, testable invariant.

---

## R16-1 — Failure taxonomy confirmation (R15 → R16 framing)

R15-1/R15-2 already established the four `Failed` producer sites and their classification. R16 reframes
them as **design semantics** (not just observation). The "did R16 change the existing decision?" check
is the table the user requested.

| Failure                  | Site (cpp)   | FailureStage / FailureReason                | Current Decision | R16 design question (R16-1, R16-2, R16-3) |
|--------------------------|--------------|---------------------------------------------|------------------|--------------------------------------------|
| build #1 failure         | Orchestrator:189 | `trySubmit:build`                            | `Failed → ResolvedFailed` (−1) | Is retry on the same `recoveryObligationId` possible? |
| crossfade rebuild failure| Orchestrator:255 | `trySubmit:rebuild`                          | `Failed → ResolvedFailed` (−1) | Same question; build inputs identical to #1 |
| publish execution failure| Orchestrator:303 | `trySubmit:publish` / `publishFailure`      | `Failed → ResolvedFailed` (−1) | Resource already destroyed (`destroyRolledBackDSP` at :287) — *different* retry question |
| redundant switch route   | Orchestrator:386 | `submitPublishRequest:notFinalized`         | `Failed → ResolvedFailed` (−1) | Idempotent re-route of #1/#2; same answer as build |
| stale generation         | Orchestrator:377 | `submitPublishRequest:stale`                | `StaleSuperseded → ResolvedStaleSuperseded` (−1) | **R16: do not change.** Out of scope. |
| queue pressure           | Orchestrator:396 | `submitPublishRequest:pressure`             | `Retry` (ΔL=0, `rearmRecoveryRetry`) | **R16: do not change.** This is the comparison reference. |
| shutdown                 | discard loop:1156 | `discardRecoveryRequestsOnShutdown`         | `ShutdownDiscarded` (−1) | **R16: do not change.** |

**R16-1 conclusion:** Three categories *are* open for design judgment: the **build-failure cluster** (#1,
#2, #4) and the **publish-failure cluster** (#3). The "compare to current" anchor is the `RejectedPressure
→ Retry` path (cpp:396-398) — that decision is **already** ΔL=0 + redrivable, so it is the de-facto
correct semantic for *transient* failures inside recovery. R16 must decide whether build / publish
hard failures are "transient like pressure" or "permanent like shutdown".

---

## R16-2 — Build-failure retry analysis

### What is preserved across a `build #1` failure?

Trace (Orchestrator.cpp:176-191):

```text
worldBuilder.buildRuntimePublishWorld(&req.sealedSnapshot, spec)   // 入力 = sealedSnapshot + spec
   ↓ nullptr
if (!worldOwner) {
    if (!req.newDSP.isNull())
        lifetime_.retire(newDSPResolved);                          // DSP は epoch-deferred retire
    ...
    engine_.runtimePublicationBridge_.resolveRecoveryObligation(
        req.recoveryObligationId, RecoveryOutcome::Failed);        // ← ★ これが「消滅」
    return RejectedNotFinalized;
}
```

What is **lost** at the moment of `resolveRecoveryObligation(Failed)`:

- The Live slot in `RecoveryAdmissionTable` becomes `ResolvedFailed` (terminal, idempotent).
- The handle/identity pair `{coalesceKey(quarantinedHandle, fingerprint)}` is no longer reachable from
  the table (`findByKey` matches only `state == Live`, h:335).
- `liveCount_` decremented; the slot becomes reusable (a subsequent `tryInsert` could pick it again
  with a **new** `nextId_`).
- The caller (`submitPublishRequest`) is told `RejectedNotFinalized`. It does NOT know whether this is
  the "shut down" flavor or the "build broken" flavor, and it has no handle to re-invoke retry.

What is **preserved**: the `req` itself (caller's `PublishRequest`) — but only as a transient local in
`submitPublishRequest`. The `runtimePublicationBridge_` has no path to re-trigger `trySubmitImpl` for
the same `obligationId`.

### Can we re-build using the same `obligationId`?

`RecoveryAdmissionTable::tryInsert` (h:343) is the *only* `+1` site. It always issues a **strictly
greater** `LogicalRecoveryObligationId` (h:349, `++nextId_` monotonic, ABA-safe per C9). A retry that
re-enters via `submitRecoveryRequest` would **coalesce** onto the existing Live slot (only if it is
still `Live`); after `Failed`, the slot is terminal, so a new `tryInsert` is the only path — and that
issues a **new** id (and thus a **new** logical obligation).

Therefore, **`Failed → re-build on same obligation` is structurally impossible under the current
contract** — `tryInsert` cannot resurrect a `ResolvedFailed` slot, and `findByKey` (coalesce) only
matches `Live`. Any "retry" path must therefore either:

- **(a)** enter via `submitRecoveryRequest` with a **new** id (ΔL=+1, new obligation — strictly more
  capacity cost, breaks I4 "no unsanctioned disappearance" only by absorbing it into a new obligation),
  or
- **(b)** keep the obligation Live (`Retry` / deferred), so `findByKey` matches and the next
  `submitRecoveryRequest` coalesces back onto the same id with the latest `buildSource` and a fresh
  delivery attempt.

### Is `buildSource` / generation / handle / epoch sufficient for a same-id retry?

Yes. `LogicalRecoveryObligation::buildSource` is a value-copied `RuntimeBuildSnapshot` (h:317;
R5-8 / I2 §187 / I3 §174 — value semantics). The recovery path uses it as the immutable input to
`runtimeBuilder.build` (RebuildDispatch.cpp:1027 — `runtimeBuilder.build(recovery->buildSource.buildInput, convolverSnapshot)`).
The `RecoveryIntent` itself is reconstructed on redrive from the slot (cpp:1019-1026 — exactly the same
fields). For a `Retry` outcome, the obligation stays Live, the next `redriveDeferredRecovery` reuses the
same id + same `buildSource` (the Builder will simply re-take and re-build; if a fresh `buildSource`
arrives via coalesce, that *replaces* the value but preserves the identity — D105-R5-8 / D105-R5-10
`buildSource` ownership rules).

### Retry boundedness

The durable-side retry already has a hard cap: `kMaxRecoveryConsecutiveFailures = 4`
(`AudioEngine.RebuildDispatch.cpp:1015`). After 4 consecutive transient build / warmup failures the loop
`break`s and the obligation stays in `DurablePending` (`:1073` "成功で連続失敗カウンタをリセット" —
`recoveryConsecutiveFailures` resets only on success). The **transport-side** (Recovery intent queue)
path has **no** `kMaxRecoveryConsecutiveFailures` bound — it relies on the durable slot eventually
admitting it. Combined: the per-obligation retry budget is effectively **bounded by 5 build
attempts** in the durable path (1 initial + ≤4 retries), and **unbounded in the deferred/transport
path** unless a `rebuildRequest` event or wake-up re-prioritises it.

For the design we must ask: where does the 5th-attempt-failure go? Currently the obligation is in
`DurablePending` and will be re-taken on the next Builder wake. If the next wake re-runs `trySubmitImpl`
and it still fails, today that goes `Failed → terminalized` (the problem R15 identified). Under a
retry-preserving contract, that **5th-and-onward** attempt must also stay `Retry` (or transition to
something else — R16-5 covers this).

### R16-2 conclusion

**Same-obligation retry for build failure is structurally compatible with the current code** if and
only if the failure outcome is reclassified to `Retry` (ΔL=0, delivery re-attached by the existing
`redriveDeferredRecoveryObligations`). The `buildSource` / identity chain is intact, the retry
boundedness is `kMaxRecoveryConsecutiveFailures = 4` (Builder-side) plus deferred redrive (Coordinator
side), and no new obligation is created (ΔL=0 ⇒ capacity preserved). The first design candidate
("失敗したら新しい recovery obligation を作り直す") is **explicitly rejected** at the R16 design
level: it would (i) violate I4-D15.2's "admission event count ≥ logical obligation count" intent for
no semantic gain, (ii) consume the 32-slot budget faster than the existing same-id retry, and (iii)
require non-trivial coalesce-on-ResolvedFailed machinery that the current table does not provide
(`findByKey` matches only `Live`).

---

## R16-3 — Publish-failure retry analysis (different from build)

Trace (Orchestrator.cpp:277-307):

```text
executor_.publish(engine_, std::move(frozen), req.newDSP, oldHandle)
   ↓ != Success
if (newDSPResolved != nullptr)
    lifetime_.destroyRolledBackDSP(newDSPResolved);               // ★ DSP 破棄
...
engine_.runtimePublicationBridge_.resolveRecoveryObligation(
    req.recoveryObligationId, RecoveryOutcome::Failed);          // ← ★ obligation 消滅
```

### Why publish-failure is NOT the same as build-failure

1. **Resource state diverges**:
   - Build failure → `newDSP` was created, never reached `executor_.publish`. `lifetime_.retire`
     (epoch-deferred) is the disposal path (cpp:181 / 247).
   - Publish failure → `newDSP` was wrapped into `FrozenRuntimeWorld`, handed to `executor_.publish`,
     and the publish call committed something (e.g. set a flag, advanced an internal state). The
     rollback path is `destroyRolledBackDSP` (cpp:287), which performs **immediate** destruction
     (no epoch grace, no retire, no reclamation).
2. **`commitRuntimePublication` (work70 Phase2) has already done partial work**: the
   `ScopeExit`-based rollback reclaims the handle but the inner state of the publish path may have
   advanced (the comment at :282-285 notes "commitRuntimePublication の ScopeExit が Handle を
   rollback 済み（Reclaimed）").
3. **Retry semantics are not "re-call `executor_.publish`"**: re-calling `executor_.publish` for the
   same frozen world would re-trigger the same failure (the world is destroyed). The only honest
   retry is "rebuild from the same `buildSource` and re-publish" — i.e. the same retry shape as a
   build failure.

### Implication for the design

A retry-preserving semantics for publish failure must, in practice, route to the **same path as a
build failure**: re-acquire a builder, re-build, re-publish. From the recovery-obligation perspective
the only thing that distinguishes build-fail from publish-fail is **when** the obligation last had a
Live delivery representation.

- If the obligation was **Durable** (held in the single `pendingRecoveryAdmission_` slot) when publish
  failed: the slot is still `DurablePending` (publish failure does not call `settlePendingRecoveryAdmission`
  — verified by grep: 0 hits in Orchestrator.cpp). The next Builder take can retry; this is exactly
  the existing `kMaxRecoveryConsecutiveFailures` loop. **A `Retry` outcome is structurally correct.**
- If the obligation was **Transport** (held in `recoveryIntentQueue_`) when publish failed: the intent
  has already been `popRecoveryRequest`-ed (RebuildDispatch.cpp takes a `RecoveryIntent` and consumes
  it; cpp:1131 `popRecoveryRequest` returns and `fetchSubAtomic(pendingIntentCount_, ...)`). The
  obligation's `delivery` field is now stale (it still reads `Transport` even though the intent is
  gone — there is no `delivery= None` write in the publish-failure path). The next `redrive` will skip
  it (cpp:988 `if (s.delivery != None) continue;`).
- If the obligation was **None (deferred)**: trivially retriable (no work was done).

### R16-3 conclusion

**Publish failure and build failure can share the *same* `Retry` outcome**, because both reduce to
"rebuild from `buildSource` and re-publish". However the **precondition** is different:

- For the durable-resident case the existing `kMaxRecoveryConsecutiveFailures = 4` loop already
  provides the bounded retry; only the obligation's `Live` state needs to be preserved.
- For the transport-resident case the **post-publish-failure delivery state is `Transport` but the
  intent is gone** — the obligation is now **stranded**: it is `Live` but unreachable by either the
  Builder (no intent in the queue) or the redrive (delivery != None). **The current code does not
  recover this case.** Under the existing semantics, this case *is* a real failure (the obligation
  will eventually hit shutdown-discard). Under a retry-preserving contract, the obligation must be
  *re-driven* — i.e. the `Failed` path must transition `delivery` to `None` so that the next redrive
  recovers it. (This is a state mutation the R17 implementation must perform; R16 only decides the
  *semantic* — it is the same "ΔL = 0, same id, new delivery attempt" chain as build-failure.)

Therefore, **the design intent for R16 is**: `Retry` semantics for both build and publish failures,
**with the explicit precondition** that the implementation also restores the obligation's
`delivery` field to `None` (i.e. re-eligible for `redriveDeferredRecovery`) at the moment of the
`Retry` decision. The user asked us **not** to conflate "Retry" with a simple label substitution; the
correct retry target is the **recovery episode as a whole** (rebuild → publish), not a single
publish attempt.

---

## R16-4 — R5-10 redrive chain: does `Failed → Live → redrive → same id` hold under current code?

Walk the chain the user asked for:

```text
Failed
  ↓
Live維持（ΔL=0, delivery=None を再設定）
  ↓
redriveDeferredRecoveryObligations()（cpp:982-991）スキャン → delivery==None のみ対象
  ↓
redriveDeferredRecovery(obligationId)（cpp:998-1054）:
    durable slot free → delivery = Durable
    else transport queue has space → delivery = Transport
    else both busy → delivery = None（stay deferred）
  ↓
Builder 消費（popRecoveryRequest or takePendingRecoveryAdmission）
  ↓
trySubmitImpl（Orchestrator.cpp:40-333）— recovery->buildSource で再 build → 再 publish
```

**Result of the trace**:

1. **The `trySubmitImpl` path uses `recovery->buildSource` (RebuildDispatch.cpp:996-999)**, not
   `req.sealedSnapshot`. The obligation's `buildSource` (set at `submitRecoveryRequest` time and
   updated on coalesce, h:317 / cpp:892) is what the Builder consumes. This is **already the same
   identity chain** as the original submission: the obligation, its `buildSource`, and its identity
   (coalesce key) survive the redrive.
2. **`redriveDeferredRecovery` does not issue a new id**: cpp:1026 `intent.obligationId = obligationId;`
   (existing id preserved). The re-attached intent carries the obligation's stored `buildSource`
   (cpp:1019-1025). The Builder's `takePendingRecoveryAdmission` (cpp:1064) reconstructs the intent
   with the same `obligationId`.
3. **Single `−1` authority is untouched**: `resolveRecoveryObligation` (cpp:948-964) and
   `RecoveryAdmissionTable::resolve` (h:366-378) are not called on the retry path; they only fire on
   a true terminal resolution.
4. **`Retry` already works in the existing code** for the pressure path: cpp:396-398 routes
   `RejectedPressure` to `RecoveryOutcome::Retry` and `rearmRecoveryRetry` only re-arms the durable
   slot when the slot already holds *this* obligation. `redriveDeferredRecoveryObligations` (the
   periodic Coordinator tick + `submitRecoveryRequest` start, R5-10 §2) finds a `Live && delivery==None`
   obligation and re-attaches delivery.

**R16-4 conclusion**: The `Failed → Live → redrive → same obligationId → recoveryIntentQueue_ /
pendingRecoveryAdmission_` chain **is fully supported by the current code**, with **two missing
prerequisites** that the R17 implementation must add (and that R16 *names* but does not change):

- **P-A**: The `Failed` site at Orchestrator.cpp:189 / 255 / 303 must route to `RecoveryOutcome::Retry`
  instead of `Failed` (semantic swap), and the `submitPublishRequest` switch route at cpp:386
  likewise. The obligation stays Live; `liveCount_` is unchanged.
- **P-B**: At the moment of routing to `Retry`, the obligation's `delivery` field must be forced to
  `None` so that `redriveDeferredRecovery` will pick it up. (For the build-failure case the obligation
  was `Durable` before the failure; for the publish-failure case it was `Transport`; for the deferred
  case it was already `None`.) The current `Retry` path through `resolveRecoveryObligation` (cpp:955-956)
  early-returns *without* touching `delivery`, which means a `Live && delivery==Durable` obligation
  stays `Live && delivery==Durable` after a `Retry` — and the redrive skips it. **P-B is the
  pre-condition the R16 design must call out explicitly.**

If P-B is not done, the design is unsound: the obligation will leak (Live, never terminalized, but
also never re-driven, so it cannot complete). The current `RejectedPressure` case avoids this only
because it *originated* on the same publish path that set `delivery=Transport` — and even there the
`rearmRecoveryRetry` mechanism (cpp:968-975) only re-arms the durable slot when the slot already holds
*this* obligation; the *transport* variant of a `Live && delivery==Transport` obligation is also
stranded until either (i) the Builder pops it (which it will, on the next drain) or (ii) the
Coordinator tick scans and finds `delivery==None` — but that requires P-B.

---

## R16-5 — "Retry forever" boundedness and terminal-at-limit

`Failed → Retry` introduced without bound would let a permanently-failing obligation occupy one of
the 32 slots indefinitely. The user is right to flag this. The relevant existing bound is
`kMaxRecoveryConsecutiveFailures = 4` (RebuildDispatch.cpp:1015) which **already governs the durable
rebuild loop**.

### What the existing bound covers, and what it does not

| Path                                | Per-obligation retry bound           | R16 implications |
|-------------------------------------|--------------------------------------|------------------|
| `RejectedPressure` (existing)       | rearm + redrive + kMaxRecoveryConsec=4 (transport-not-enforced) | Already ΔL=0, no contract issue |
| Build failure (proposed Retry)      | kMaxRecoveryConsecutiveFailures=4 *if durable-resident*; **no bound for transport-resident** | R17 must extend the bound to the transport path (counter on the slot, or redrive-failure counter) |
| Publish failure (proposed Retry)    | Same as build + stranded-recovery case (R16-3) | R17 must repair `delivery` to `None` *and* enforce the same bound |
| ShutdownDiscarded (existing)        | n/a — terminal | unchanged |
| StaleSuperseded (existing)          | n/a — terminal | unchanged |

### Terminal-at-limit candidates (the user listed A through E)

| Candidate | Description | Verdict |
|-----------|-------------|---------|
| **A. Retry continues** (re-arm forever) | Reject. Violates boundedness, leaks L slot, contradicts existing `kMaxRecoveryConsecutiveFailures=4` semantics. |
| **B. ShutdownDiscard** at limit | **Reuse existing path**. The `discardRecoveryRequestsOnShutdown` table-scan (cpp:1153-1157) is *not* the right tool here — it only runs at shutdown, not at retry exhaustion. A new "exhausted → terminal" transition would be needed. **Not recommended** because (i) `ShutdownDiscarded` carries shutdown-specific semantics (cpp:962 increments `recoveryObligationShutdownDiscardCount_` — only the ShutdownDiscarded outcome does, and it is used for shutdown observability), (ii) shutdown vs retry-exhaustion are different *reasons* and conflating them weakens telemetry. |
| **C. Superseded** at limit | **Reuse existing path**, but `StaleSuperseded` already has a precise meaning (the obligation is overtaken by a newer one with a supersedable target — D105-R5-9 MUST-4 / R7 §11). Retry-exhaustion is **not** "stale" in that sense; the obligation is not replaced, it just failed too many times. **Not recommended** — semantic mismatch. |
| **D. `Failed` as a new sanctioned terminal** | **Possibly.** This *is* the I4-amendment path: add `ResolvedFailed` (which the table already has, h:275) to the I4 disappearance set, with the explicit interpretation "retry budget exhausted, logical obligation truly extinct". This keeps the runtime as the source of truth (it already does this), and asks I4 to recognize the resulting `−1` as sanctioned. **Recommended as the *limit-reached* fallback under R16-A.** |
| **E. Another explicit terminal** | Possible, but adds a new `ObligationState` enum value. Not necessary: the table already has `ResolvedFailed` and the `FailureReason`/`FailureStage` taxonomy already records why. **Not recommended** — increase state surface for no semantic gain. |

### Recommended terminal-at-limit design (under R16-A)

```
build/publish failure
  → Retry
  → redrive (delivery=None re-set)
  → Builder take → trySubmitImpl
       ↓ build/publish again fails
  → Retry (liveCount unchanged)
  → repeat, until...

Consecutive-failure counter on the slot
  = number of consecutive Failed-equivalent (transient) outcomes without a Published/Superseded in between
  reset on success (parallel to kMaxRecoveryConsecutiveFailures=4's existing reset on :1075)

counter == K (e.g. K=4 to match existing bound)
  → RecoveryOutcome::Failed (terminal, −1, R16-D)
  → I4 disappearance set extended to include Failed-as-terminal
```

This is **a single, deterministic `−1`** that occurs at a well-defined retry-budget boundary, with the
obligation previously in `Live` state the whole time. The cardinality proof is unaffected
(`L_max = 32`; the obligation is Live for the full retry lifetime; one `−1` at exhaustion; one slot
released). The `kMaxRecoveryConsecutiveFailures=4` already covers the durable-resident path; the
counter must be moved to the *slot* (so it survives redrive) for the transport-resident path. The
existing `kMaxRecoveryConsecutiveFailures=4` is a *local* constexpr inside `RebuildDispatch.cpp:1015`
— it is not a property of the obligation, it is a property of the loop. The R16 design therefore
**explicitly requires** that this counter (or its equivalent) be promoted to a *per-obligation*
field in `LogicalRecoveryObligation` (e.g. `consecutiveFailureCount`), reset on `Published`, decremented
to `ResolvedFailed` when it hits the bound.

The R16-2 / R16-3 answers above also reveal a subtle requirement: a `Retry` outcome that **does not
have any pending delivery resource** (i.e. `delivery==None` because the transport was full and the
durable slot was held by a *different* obligation) is a perfectly valid Live state and must not
trigger the counter — the obligation is waiting on a *resource*, not failing. The counter must
count *attempt failures*, not *waiting ticks*. The existing `kMaxRecoveryConsecutiveFailures` is
already structured this way (it increments only on a build / warmup failure, not on a no-op
"queue empty" cycle), which validates the design.

### R16-5 conclusion

**Bounded retry: yes, by promoting `kMaxRecoveryConsecutiveFailures` from a local loop counter to
a per-obligation counter, with terminal `Failed` (R16-D = R16-A's exhaustion-terminal) at the limit.
** This decision is conditional on the I4 amendment: `Failed` becomes a *new* sanctioned terminal
*only* when triggered by retry-exhaustion (not by the *initial* failure — see R16-6 for the
distinction). I4's disappearance set becomes `{Success, Superseded, ShutdownDiscard, RetryExhausted}`
where `RetryExhausted` is a documented alias for `Failed` at the exhaustion boundary.

---

## R16-6 — I4 / D14.3 / D15.2 comparison table (R16 final artifact)

This is the table the user asked for. The `?` column is what R16 decides.

| Outcome           | Logical obligation   | −1 / ΔL | I4 (D14.3 / D15.2) status         | R16 decision (and what it means for runtime/I4) |
|-------------------|----------------------|--------:|-----------------------------------|--------------------------------------------------|
| `Success`         | extinct (terminal)   |  1      | allowed (`Success`)               | keep                                              |
| `Superseded`      | extinct (terminal)   |  1      | allowed (`Superseded`)            | keep                                              |
| `ShutdownDiscard` | extinct (terminal)   |  1      | allowed (`ShutdownDiscard`)        | keep                                              |
| `Retry`           | remains `Live`       |  0      | allowed (ΔL=0)                     | keep                                              |
| `Failed` (initial transient — build / publish / stranded) | remains `Live` (R16-A) | 0 | **R16-allowed** (ΔL=0; redriven) | **change**: `Failed` is no longer reached for transient failure; the four `Failed` producer sites route to `Retry` instead. R17 will make the source change. |
| `Failed` (retry exhausted — consecutive-failure counter hits K) | extinct (terminal) | 1 | **R16-allowed** (extended I4 disappearance) | **amend I4-D15.2** to include `Failed` as a sanctioned disappearance *with the precondition* that the obligation is at retry-exhaustion (i.e. the counter rule). The runtime emits a `FailureReason::RetryExhausted` (or equivalent) so the audit trail distinguishes this from any non-exhaustion `Failed` (which R16-A removes). |

### What this means for I4

I4-D15.2 currently says:

> A logical obligation may disappear only by: `Success`, explicit `Superseded` decision,
> `ShutdownDiscard`.

R16-A proposes the following amendment (precise wording to be agreed in R17):

> A logical obligation may disappear only by: `Success`, explicit `Superseded` decision,
> `ShutdownDiscard`, **or retry-exhaustion** (when the per-obligation consecutive-failure counter
> reaches the bound `K = kMaxRecoveryConsecutiveFailures` — the obligation's state at the moment
> of the bound-triggered `Failed` is `Live` and `delivery == None`).

The I4 wording is **not changed in R16**; only the *intent* is decided. R17 will propose the actual
contract patch.

### What this means for C8

C8 (`ISRSemanticValidationTests.cpp:1028-1044`) currently asserts:

```text
c->resolveRecoveryObligation(*id, Failed);
c->resolveRecoveryObligation(*id, ShutdownDiscarded);  // no-op
liveLogicalRecoveryObligationCount() == 0             // ← key expectation
```

Under R16-A this test is **still passing** for a different reason than the one it was originally
written for:

- It directly invokes `resolveRecoveryObligation(id, Failed)`, bypassing the orchestrator entirely.
- Under R16-A the **runtime** no longer calls `resolveRecoveryObligation(_, Failed)` from the four
  producer sites; the only call sites that would invoke it are the **retry-exhaustion path** (R16-5,
  not yet implemented in R16) and a possible **explicit cancel-by-caller** (also not implemented).
- C8 as written remains a valid **table-level unit test** of `RecoveryAdmissionTable::resolve`'s
  idempotency contract for the `ResolvedFailed` terminal. The test is *not invalidated* by R16-A.
- **However**, C8 is *misleading* under R16-A: it codifies an implementation behavior that is no
  longer the production path. R17 should **either** (a) re-purpose C8 to test the retry-exhaustion
  path explicitly (e.g. with a stub counter on the slot) or (b) keep C8 as a low-level table test
  and add a separate test for the runtime path. Per the user's instruction ("C8 は現状 `Failed → L=0`
  を期待しているため、R16 で設計判断が確定するまで変更禁止です"), R16 leaves C8 unchanged.

### R16-6 final verdict: R16-A (Retry-preserving contract), with the following dual decision

1. **Runtime semantic change**: `RejectedNotFinalized` (build failures) and `RejectedPublishFailure`
   are reclassified as `RecoveryOutcome::Retry` (ΔL=0, obligation stays Live) with `delivery` reset
   to `None` so `redriveDeferredRecovery` can recover the obligation. This is the **R16-A primary
   decision**.
2. **I4 amendment**: `Failed` becomes a *new* sanctioned terminal **only** at the retry-exhaustion
   boundary, with a documented precondition (counter rule). This is the **R16-A exhaustion decision**
   (which is the same as the **R16-D** candidate from R16-5 — keeping the same `ResolvedFailed` slot
   for telemetry, but only emitting it under one well-defined condition).

The other R16 candidates (R16-B full I4 amendment; R16-C split by failure stage) are **rejected**:

- **R16-B** ("Failed is intentional extinction, −1 is correct, I4 amended wholesale") is rejected
  because it would lose the `buildSource` / identity chain across one failure and re-create the
  obligation as a new id on the next attempt — a strict regression vs. R16-A on both capacity
  (one extra slot consumed per transient failure) and identity preservation (R5-8 / R5-9 / R5-10
  have all been carefully designed to keep the same `obligationId` across retry).
- **R16-C** ("build-fail → Retry, publish-fail → terminal") is rejected because the *retry target* is
  the same (rebuild → publish) for both — splitting them would mean two retry mechanisms for what is
  semantically one operation, and would re-introduce the stranded-recovery case from R16-3 in a
  different guise (a publish-fail obligation that *cannot* be retried is exactly the same logical
  obligation as a build-fail obligation that *can*).

---

## R16-7 — Capacity impact (sanity check on R16-A's ΔL=0 promise)

### What changes under R16-A

Under the current code, every transient build/publish failure consumes one of the 32 slots and
releases it (`ResolvedFailed`, ΔL=−1); a subsequent attempt creates a new slot (ΔL=+1). **Net: 0.**
The slot is occupied *only for the duration of the failed attempt*, which is bounded by the build /
publish attempt time itself.

Under R16-A, every transient failure *keeps* the slot occupied (Live, delivery=None) until either
(a) a successful retry consumes the slot (ΔL=−1 via `Published`) or (b) the retry counter exhausts
(ΔL=−1 via `RetryExhausted`-flavored `Failed`). **The slot occupancy increases from "one attempt" to
"up to K attempts".** This is a real cost. The bounds:

- Per-obligation `L_i_max_lifetime = K × T_build` (K=4 → ~4 build cycles). Each build cycle
  generates one `RuntimeWorld` and one `FrozenRuntimeWorld`; `world→handle` cardinality per obligation
  remains 1 (the obligation owns its `buildSource`, the world is owned by the publish path which has
  its own EBR-based reclamation per D46 / D47).
- `Σ_i L_i(t) ≤ 32` (unchanged — `RecoveryAdmissionTable::tryInsert` still gates at 32).
- The `reservedLogicalObligations ≤ 32` invariant (D26.2) is unaffected: a Live obligation holds
  exactly one reservation regardless of its `delivery` state.

### Does R16-A break any of the prior `B_total` bounds (D35 / D36 / D37)?

- **R_logical (32 × ~1KB ≈ 11–50KB)**: unchanged. R16-A does not add a per-obligation counter
  significant enough to move the upper bound. (A `uint8_t` or `uint16_t` counter is <2B per slot, well
  within the D36.1 352B upper bound for `RecoveryAdmission`.)
- **R_recoveryQueue (256 × 224B = 57KB)**: unchanged. R16-A does not push additional entries to the
  queue — the redrive path is the same.
- **R_episode / R_quarantineMeta / R_allocator**: unchanged. No new storage is introduced.
- **R_builder (1 × B_build)**: unchanged (D32 / INV-RES-1).
- **R_runtime (2 + N_retired ≤ 4610 per D47)**: unchanged. R16-A does not affect the retired-world
  cardinality (worlds are still retired via `retireRuntimePublishWorldNonRt` regardless of which
  obligation triggered the publish).

### Liveness / progress

R16-A's "obligation stays Live across transient failures" is exactly the design the user already
endorsed in D105-R5-9 MUST-2 for the `Retry` case (`RejectedPressure → Retry, ΔL=0, redriven`). R16-A
extends the same `Retry` semantic to build / publish failures, where it is *more* defensible (the
existing code already has a `kMaxRecoveryConsecutiveFailures = 4` retry budget for the durable
path; R16-A simply connects the obligation-level `Retry` outcome to that budget).

The one new liveness risk is the *transport-resident stranded case* identified in R16-3: a
publish-failure obligation with `delivery==Transport` and no live intent in the queue. R16-A's
exhaustion counter (per-obligation) protects against unbounded occupancy, but a stranded obligation
will not be re-driven until P-B is implemented. R16 does not change source; R17 must implement P-B
and the per-obligation counter as a single coordinated change.

### R16-7 conclusion

**R16-A preserves all prior capacity bounds (D32 / D35 / D36 / D37) and the
`kMaxLogicalRecoveryObligations = 32` invariant.** The new per-obligation consecutive-failure
counter is a sub-byte field on the existing slot (no measurable `B_total` delta). The stranded
transport-resident case is a *liveness* concern handled by P-B (R16-4) and the exhaustion
counter (R16-5), not a *capacity* concern.

---

## R16-8 — Test design catalog (design only, no implementation)

These are the tests R17 will add (or extend) to validate R16-A. They are documented here for
architectural review; no source change in R16.

### Required test surface (R16 / R17)

| ID            | Pre-R16 (current) assertion | Post-R16-A assertion (R17) |
|---------------|------------------------------|----------------------------|
| **T-R16-1** build failure → obligation remains Live | not tested (production goes `Failed → −1`) | after a single `trySubmitImpl` build failure, `liveLogicalRecoveryObligationCount()` is unchanged; `obligation.state == Live`; `obligation.delivery == None`; `redriveDeferredRecoveryObligations()` re-attaches delivery |
| **T-R16-2** publish failure → obligation remains Live | not tested (production goes `Failed → −1`) | after a single `trySubmitImpl` publish failure, `liveLogicalRecoveryObligationCount()` is unchanged; `obligation.state == Live`; `obligation.delivery == None` (post-P-B); `redriveDeferredRecoveryObligations()` re-attaches delivery |
| **T-R16-3** failure → redrive → same `obligationId` | not tested (production `Failed` re-creates with a new id on next attempt) | after build failure and redrive, the `RecoveryIntent` consumed by the Builder carries the same `obligationId` as the original submission; `liveCount` is unchanged end-to-end; `coalesce` on the second attempt matches the original (ΔL=0) |
| **T-R16-4** repeated failure → bounded behavior | not tested | K+1 consecutive `trySubmitImpl` failures (K = `kMaxRecoveryConsecutiveFailures`) on the *same* obligation (forcing a per-slot counter) result in: (a) `obligation.state == Live` and `delivery == None` for attempts 1..K, (b) `obligation.state == ResolvedFailed` and `liveCount` decremented at attempt K+1, (c) `recoveryObligationShutdownDiscardCount` *unchanged* (the `Failed` outcome at exhaustion is not a `ShutdownDiscarded` and must not bump the shutdown-discard counter — that invariant is already codified in C8 and preserved) |
| **T-R16-5** successful retry → exactly one `−1` | not tested | after K transient failures and one `Published`, `liveCount` is 0 and `recoveredObligationSuccessCount` is exactly 1; the per-obligation counter is reset to 0 |
| **T-R16-6** shutdown during failed/deferred recovery | not tested | an obligation in the `Live && delivery==None` state (post-failure, pre-redrive) is included in `discardRecoveryRequestsOnShutdown`'s table scan; on shutdown it is terminalized as `ShutdownDiscarded` (not `Failed`); `recoveryObligationShutdownDiscardCount` is incremented; `liveCount` reaches 0; `isFullyDrained` returns true (D105-R13 predicate) |
| **T-R16-7** stale recovery after retry | not tested | an obligation that becomes `stale` (e.g. a newer admission with a supersedable target) is terminalized as `StaleSuperseded` regardless of whether it was previously in a failure-retry state; `recoverObligationStaleCount` is exactly 1, no `Failed` is emitted (D105-R5-9 MUST-4 preserves the stale-terminal invariant) |

### Tests that must NOT be modified in R16 (per user instruction)

- **C8** (`ISRSemanticValidationTests.cpp:1028-1044`) — codifies the `resolveRecoveryObligation(id,
  Failed)` table-level contract. R16-A keeps C8 *passing* (the table-level `Failed → ResolvedFailed →
  liveCount--` behavior is preserved as a unit test of the table). The test is *no longer exercising
  the production path* under R16-A; that is acceptable and should be acknowledged in the test
  documentation when R17 is implemented.

### Relationship to R5-10 (regression)

R5-10 established:

- `redriveDeferredRecovery` (cpp:998) and `redriveDeferredRecoveryObligations` (cpp:982) operate
  only on `Live && delivery==None` obligations.
- `redriveDeferredRecovery` issues no `tryInsert` / `resolve` / `liveCount_` mutation.
- Tests C11–C16 (added by R5-10) cover the deferred-recovery happy path.

R16-A is **strictly additive** to R5-10:

- The same `redrive` mechanism is used; R16-A only changes *which obligations reach* the
  `Live && delivery==None` state (previously only `Retry`-path obligations; under R16-A also
  build/publish-failure obligations).
- C11–C16 continue to pass under R16-A — they exercise the same code paths.
- T-R16-1..7 are the new tests that document the R16-A behavior; they are not regression tests for
  R5-10.

---

## R16 final verdict (R16-A)

### **R16-A: Retry-preserving contract — adopted (read-only, no source change)**

```
Failed producer (build / publish)   [R15-A confirmed: classification A, true extinction]
        ↓
R16-A reclassification: not Failed   [R16-1, R16-2, R16-3]
        ↓
RecoveryOutcome::Retry (ΔL = 0)      [R16-2: same-id re-build via buildSource; counter bounded by kMaxRecoveryConsecutiveFailures]
        ↓
obligation stays Live                [R16-2, R16-7: capacity preserved]
        ↓
delivery = None (P-B, R17 implements) [R16-4: required precondition for redrive]
        ↓
redriveDeferredRecoveryObligations() [R16-4: same code path as Retry-pressure]
        ↓
recoveryIntentQueue_ or pendingRecoveryAdmission_ (ΔL = 0, same id)
        ↓
trySubmitImpl re-builds from buildSource and re-publishes
        ↓
On success:  Published  → ΔL = −1, terminal
On repeat failure: counter += 1
On counter == K (= kMaxRecoveryConsecutiveFailures):
    Failed (terminal) → ΔL = −1, terminal  [R16-5, R16-6]
        ↓
I4 amendment: {Success, Superseded, ShutdownDiscard, RetryExhausted (= Failed at exhaustion)}
```

### R16-A decision table

| Concern | R16-A position |
|---|---|
| I4 disappearance set | amended to include `RetryExhausted` (= `Failed` at exhaustion) |
| Runtime code | `Failed` producer sites change to `Retry` (4 sites, R17) |
| C8 | unchanged (table-level unit test still valid) |
| Capacity (R_logical / R_recoveryQueue / R_episode / R_builder / R_runtime) | unchanged; per-obligation counter is a sub-byte field on the existing slot |
| I4 ↔ runtime consistency | achieved (R15-A's "extinction not sanctioned" violation removed; R16-6's new "extinction at exhaustion" is now sanctioned) |
| Boundedness | `kMaxRecoveryConsecutiveFailures = 4` (existing) is promoted from a local loop counter to a per-obligation counter; reset on `Published` |
| Stranded-recovery (R16-3) | resolved by P-B: `Failed` site must set `delivery = None` (R17 implements) |
| Liveness | same as R5-10's `Retry`-pressure path: periodic redrive + submit-trigger redrive |
| Test design | T-R16-1..7 added in R17; C11–C16 unchanged; C8 unchanged |

### R16 candidates explicitly rejected

- **R16-B** (Failed is intentional extinction, I4 wholesale amendment): rejected — loses the same-id
  retry property that R5-8 / R5-9 / R5-10 / R16-2 all rely on; consumes extra capacity; not the
  semantics R5-9 MUST-2 already established for `Retry`.
- **R16-C** (build-fail → Retry, publish-fail → terminal): rejected — both failure stages have the
  same retry target (rebuild → publish) and should share the same retry mechanism; splitting them
  re-introduces the stranded-recovery case in a different form.
- **R16-5 B (ShutdownDiscard at exhaustion)**: rejected — `ShutdownDiscarded` has shutdown-specific
  telemetry semantics (cpp:962) and conflates reasons.
- **R16-5 C (Superseded at exhaustion)**: rejected — `StaleSuperseded` has a precise "newer obligation
  took over" meaning; retry-exhaustion is not that.
- **R16-5 E (new explicit terminal)**: rejected — `ResolvedFailed` (h:275) is already in the table;
  the new `FailureReason::RetryExhausted` (or a counter-condition on `Failed`) is sufficient.

### Source changes: 0 (R16 is a read-only design audit)

R17 will implement:
- P-A: the four `Failed` producer sites route to `Retry` instead (R16-2, R16-3, R16-6).
- P-B: at the moment of routing to `Retry`, the obligation's `delivery` is set to `None`
  (R16-4 — required for redrive to recover the obligation).
- A per-obligation `consecutiveFailureCount` field on `LogicalRecoveryObligation` (R16-5 —
  promoted from the existing `kMaxRecoveryConsecutiveFailures` local constexpr).
- I4 patch proposal: add `RetryExhausted` (= `Failed` at exhaustion) to the disappearance set,
  with the precondition documented in R16-6.
- Tests T-R16-1..7 (R16-8).

The user explicitly requested that R16 itself not change source — the source change is gated on
the user's sign-off of R16-A, and on the I4 patch proposal being adopted.

### Open items (R16 hands off to R17)

1. **P-B location and atomicity**: `delivery = None` must be set before the `Retry` outcome is
   committed, atomically with the `Retry` decision. The current `resolveRecoveryObligation` (cpp:948)
   early-returns on `Retry` (cpp:955-956) *without* touching `delivery`. R17 must design the
   `Retry` site to mutate `delivery` (or, equivalently, a new `markRetry` method on the table that
   combines the `delivery = None` reset with the `Retry` outcome).
2. **Counter location**: a per-obligation `consecutiveFailureCount` is a natural choice but must
   be reset on `Published` (matching the existing `:1075` reset rule), reset on `StaleSuperseded`,
   and *not* reset on `ShutdownDiscarded` (since the obligation is terminal anyway). R17 will
   specify the exact `ObligationState` transition that resets the counter.
3. **K value**: should K remain `4` (matching the existing `kMaxRecoveryConsecutiveFailures`) or
   be raised to give transient failures more headroom? The current value was a "spin-prevention"
   bound for the durable loop; under R16-A it becomes an exhaustion bound. R17 should re-evaluate.
4. **Telemetry**: the new "exhaustion → `Failed`" path emits a `Failed` that is semantically
   distinct from any current `Failed`. R17 should add either a separate counter
   (`recoveryRetryExhaustedCount`) or a `FailureReason::RetryExhausted` enum value to distinguish
   the two in the audit trail.
5. **I4 patch proposal**: the wording proposed in R16-6 needs user review and acceptance before
   R17 begins the contract patch.
