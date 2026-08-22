# Phase I-T1-D101-2.5O-3E — Execution Binding Contract Closure Judgment

**Status**: COMPLETE (audit-only, 0 code changes).

## 0. Scope & Prohibitions (per user instruction)

Audit only — zero of:
コード変更 0 / evaluate変更 0 / evaluateDeferred変更 0 / 新 token 0 / 新 permit 0 /
新 ID 0 / 新 deadline 0 / 新 timeout 0 / 新 field 0 / 新 binding API 0 / 設計案の採用 0.

3E inherits 3B, 3C, 3D as facts. 3E = **Execution Binding Contract Closure**: does the O→E
binding, as proven structurally in 3C and preserved across failure paths in 3D, constitute a
**formally closed execution contract**? No code changes proposed — 3E is the final judgment.

## 1. The Execution Binding Contract — Definition

### 1-1 What "binding" means in this audit

Per user instruction (3C-9: "correlationId は O identity として仮定してはいけません" +
"本当に既存コード上に O→E binding を証明できる semantic relation が存在しないのか"):

- Binding = a **semantic relation** between an O Accepted obligation and its E execution.
- The relation must be: (B1) existent, (B2) unique, (B3) preserved, (B4) terminally conserved.
- **A symbolic O identity token annotating E is NOT required** — the relation can be
  structural (object identity + verbatim triple projection), as established by 3C.

### 1-2 The structural relation (recap from 3C-3)

> The `frozen` RuntimeState object produced by the Builder on behalf of an Accepted obligation.
> Its `publication.{sequenceId, epoch, mappedRuntimeGeneration}` is stamped once (from the reservation),
> then **verbatim** (no arithmetic) projected into the Intent → OwnerChannelKey → PublishMetadata,
> survives `take` (single-transfer) and `publishAndSwap`, and is re-attached to Completion via
> `onPublishCommitted(seqId)`.

The `RuntimeState*` (newWorld pointer) is object-identical end-to-end:
`worldOwner.release() → frozen.releaseState() → OwnerChannel enqueue → take re-wrap → publish → registry.lookup / payload.newWorld`.

## 2. The Semantic Transaction State Machine — E Side Closure (ISRRuntimeSemanticSchema.h:535-566)

```cpp
// §3.19.5 Semantic Transaction state machine.
// Permitted transitions:
//   Building -> Validated -> Committed -> Published
//   Building | Validated | Committed  -> Rejected (terminal)
// Published and Rejected are terminal states.
enum class SemanticTransactionState : std::uint8_t {
    Building = 0, Validated, Committed, Published, Rejected
};
```

| From → To | Valid? | Meaning |
| --- | --- | --- |
| Building → Validated | ✅ | builder validation complete |
| Validated → Committed | ✅ | publishAndSwap (store) done |
| Committed → Published | ✅ | lifecycle retire/epoch advance |
| Building/Validated/Committed → Rejected | ✅ | failure — terminal |
| Published → * | ❌ | terminal, immutable |
| Rejected → * | ❌ | terminal, discarded |

**Key closure property**: the state machine is **deterministic per world object**. A given
`RuntimeState*` enters exactly one terminal state: `Published` or `Rejected`. There is no
`Building → Building` loop, no `Rejected → Published` reversal. The world's object identity
pins the state machine instance.

### 2-1 E-side terminal states vs. 3C binding evidence

| Terminal state | O→E binding evidence (3C) | 3D disposition |
| --- | --- | --- |
| Published | seqId == onPublishCommitted(seqId) == receipt watermark (3C-6) | Transferred (E owns) |
| Rejected | world destroyed via CallerDestroy (no E execution) | CallerDestroy (O reclaims) |

## 3. The Ownership Lifecycle — Closure Witness

`AudioEngine.h:4597-4600, 4614`:

```cpp
//   タイムアウトしても所有権は移譲済み（executePublish が後続で commit する）ため
//   Transferred 扱い — 呼び出し元は world/DSP を破砄してはならない。
//   ★ work88 (X2 §6.2): timeout ≠ publish failure。
//     所有権は enqueue 時点で Transferred（Allocated → Transferred → Committed → Completed）。
```

The ownership lifecycle: `Allocated → Transferred → Committed → Completed` (or
`Allocated → CallerDestroy → reclaimed`).

### 3-1 Lifecycle × Semantic state — the closure table

| Ownership lifecycle | Semantic state | Meaning | O disposition |
| --- | --- | --- | --- |
| Allocated | Building | world built, not yet enqueued | O holds unique_ptr |
| Transferred | Committed | enqueue + intent enqueued; `take(key)` pending or done | E holds (OwnerChannel) |
| Committed | Committed→Published | `authority.publish` → `publishAndSwap` | E owns (store) |
| Completed | Published | `onPublishCommitted(seqId)` → receipt watermark | E owns (lifecycle-managed) |
| — | Rejected | executor publish fail / enqueue fail | CallerDestroy (O reclaims) |
| Transferred→CallerDestroy | Rejected | enqueue or intent-fail (before take) | O reclaims |

**Closure witness**: the `RuntimeState*` pointer that is the O→E binding carrier (3C-3) is the
SAME pointer that drives the `SemanticTransactionState` machine and the ownership lifecycle.
The state machine is attached to the world (`semanticTransactionState_` field, AudioEngine.h:4748-4749:
`RuntimeState` carries `semanticTransactionState_`). So the binding carrier IS the state machine host.

## 4. The Registry Asymmetry — 3D-8 / 3E Closure Requirement

### 4-1 The gap (recap 3D-8)

`PendingPublishRegistry::lookup(seqId)` matches seqId **only**, while `OwnerChannelKey` is the
composite `{seqId, epoch, mappedGeneration}`. Under shutdown-drain-without-executePublish, the
registry entry can go stale (points to a drained/destroyed world).

### 4-2 Why this does NOT break the binding contract

The registry is **not** part of the binding relation — it is a **fallback**. The binding is proven
by:

1. **Primary path**: `OwnerChannelKey` (composite triple) — `take(key)` is the sole owner
   acquisition (3C-4, 3C-5). This is the binding channel.
2. **Registry**: `registry().lookup(seqId)` is reached **only when `take` returned null**
   (RuntimePublishExecutor.h:60-63: `newWorld = owner ? owner.get() : registry().lookup(intent.sequenceId)`).
   This is a recovery/fallback for the case where the OwnerChannel take did not yield the world
   (e.g., it was already drained or not yet enqueued). It is **not** the primary binding proof.

### 4-3 Stale entry containment (why closure holds)

- seqId is monotonic per session (3B-2, 3C-B2-B). A stale registry entry's seqId is never
  reused within the session.

- The stale entry points to a `void* newWorld` that was `destroyRolledBackDSP`'d — but `lookup`
  is **fallback-only** and only reached when `take` returned null (meaning the OwnerChannel path
  also failed). In the shutdown-drain-without-executePublish scenario, neither `take` nor
  `lookup` is called (shutdown tears down the ISR loop). The stale entry is **never dereferenced**.

- `drainAllNonRt` (OwnerChannel.h) reclaims the world via the channel, not the registry.

**Closure**: the registry asymmetry is a **latent hazard**, not an active binding violation.
It cannot cause a mis-binding (O obligation → wrong E execution) because it is fallback-only
and only reached on OwnerChannel miss (where there is no competing execution to mis-bind to).

## 5. correlationId — The Closing Argument (3C-9 + 3E)

Per 3C-9 and reconfirmed: correlationId is **O telemetry correlation only**, never an E binding
carrier. 3E closes the contract by confirming the **absence** of correlationId on E is not a
deficiency:

```text
correlationId = O telemetry correlation
              ≠ O obligation identity
              ≠ E binding identity
```

The O Accepted obligation has **no dedicated obligation identity token** on E. The binding is
carried **structurally** by:

1. The frozen `RuntimeState*` object (object identity preserved end-to-end).
2. The publication triple `{seqId, epoch, mappedGeneration}` (verbatim projection, no arithmetic).

This is **sufficient** for contract closure because:

- B1 Existence: the relation is singular (1 frozen world per Accepted → 1 seqId → 1 execution).
- B2 Uniqueness: seqId monotonic + take-once = no collision.
- B3 Preservation: object identity + verbatim triple = no loss/transformation.
- B4 Terminality (3D): every terminal state maps to a disposition (Transferred or CallerDestroy).
- Semantic state machine: `RuntimeState*` pins a deterministic state machine instance to
  `Published` or `Rejected`.

**A symbolic O identity token on E is NOT required for closure.** The structural relation is
proven and exhaustive.

## 6. The Execution Binding Contract — Formal Judgment Table

| Condition (instruction §3) | 3C finding | 3D finding | 3E closure |
| --- | --- | --- | --- |
| O identity exists | ABSENT (no first-class O token; correlationId is telemetry) | N/A | ACCEPTABLE — not required |
| E identity exists | PASS: `{seqId, epoch, mappedGeneration}` | N/A | CLOSED |
| O→E existence (1O ⇒ ∃E) | PASS (B1) | single publish call | CLOSED |
| O→E uniqueness (1O ⇔ 1E) | PASS (B2, CAUTION-mitigated) | take-once + monotonic seqId | CLOSED (with CAUTION) |
| target preservation | PASS (B3: object-identity + verbatim triple) | no leak/double-free | CLOSED |
| failure/shutdown disposition | N/A (out of scope) | PASS (B4: Transferred/CallerDestroy exhaustive) | CLOSED |
| timeout ≠ failure | PASS (3B-6, 3C-8) | Transferred preserved | CLOSED |
| correlationId ≠ binding | PASS (3B-7, 3C-7/9) | N/A | CLOSED |
| registry asymmetry (seqId-only lookup) | CAUTION (B2-D) | failure-path gap contained | CLOSED (fallback-only, no mis-binding) |

## 7. B1/B2/B3/B4 — Consolidated Formal Judgment

```text
B1 — Existence
    PASS           (3C: frozen world object carries verbatim triple into all E artifacts)
    コード証拠：RuntimePublicationOrchestrator.cpp:90-96 (Accepted→single publish→frozen);
      AudioEngine.h:4508-4510 (key from world publication); AudioEngine.h:4530-4537 (intent fill);
      RuntimePublishExecutor.h:50 (single take); OwnerChannel.h:50-76 (take-once CAS).

B2 — Uniqueness
    PASS (CAUTION: registry seqId-only asymmetry, mitigated)   (3C-B2-D)
    理由：1O→2E impossible (single publish + reject-on-key-dup); 2O→1E impossible (monotonic seqId);
      alias impossible (3-field key + monotonic epoch/mappedGeneration). Registry asymmetry is
      fallback-only + single-flight under single-CoordinatorLoop.
    コード証拠：OwnerChannel.h:39 (enqueue reject); OwnerChannel.h:70-72 (take CAS-drain);
      reserveRuntimePublicationIdentity (RuntimeBuilder.cpp:63) monotonic;
      RuntimePublishExecutor.h:60-63 (lookup fallback-only).

B3 — Preservation
    PASS           (3C: RuntimeState* object-identity end-to-end + verbatim triple)
    理由：world pointer identity (Orchestrator.cpp:88 releaseState → PublicationExecutor.cpp:23-24 →
      AudioEngine.h:4519 enqueue → RuntimePublishExecutor.h:60-63 take/registry.lookup/newWorld all
      same pointer); triple projected verbatim (AudioEngine.h:4508-4510, 4530-4537;
      RuntimePublishExecutor.h:50-52); timeout/FF preserve Transferred (AudioEngine.h:4547, 4562-4565).

B4 — Terminality (failure/shutdown ownership conservation)
    PASS           (3D: {Transferred, CallerDestroy} exhaustive, mutually exclusive; no leak/double-free)
    理由：3-failure branches (enqueue fail / intent-fail / executor-fail) each return CallerDestroy with
      a sole reclamation site (unique_ptr scope-destruct or destroyRolledBackDSP);
      timeout/FF return Transferred (ISR still commits). Shutdown drain via drainAllNonRt idempotent.
    コード証拠：OwnershipDisposition enum (AudioEngine.h:3586-3589);
      enqueue/intent-fail CallerDestroy (AudioEngine.h:4563, 4583);
      executor-fail CallerDestroy (RuntimePublicationOrchestrator.cpp:268-291);
      timeout Transferred (AudioEngine.h:4547, 4614-4617 work88/X2 §6.2);
      destroyRolledBackDSP (DSPLifetimeManager.cpp:119-123).
```

## 8. Final Determination — Execution Binding Contract Closure

```text
Execution Binding Contract
    CLOSED (PROVISIONAL)

```

### Rationale

The existing code **establishes and preserves** an O→E semantic relation that satisfies the four
binding conditions (Existence, Uniqueness, Preservation, Terminality) **in the current model**:

1. **B1 Existence — PASS**: each Accepted obligation produces exactly one frozen world whose
   publication triple is verbatim-projected into the E transfer key, Intent, registry (fallback),
   and completion seqId. The `RuntimeState*` object identity is preserved end-to-end.

2. **B2 Uniqueness — PASS (CAUTION)**: 1O→2E and 2O→1E are structurally impossible under the
   single-CoordinatorLoop + monotonic-seqId + take-once model. The registry seqId-only asymmetry
   (B2-D/3D-8) is a latent hazard but cannot cause mis-binding (fallback-only, never
   reached when the OwnerChannel primary path succeeds, seqId never reused in session).

3. **B3 Preservation — PASS**: object identity + verbatim triple preserved across every boundary
   to completion; timeout and fire-and-forget preserve Transferred.

4. **B4 Terminality — PASS**: the semantic state machine (`Building → Validated → Committed →
   Published` / `→ Rejected`, ISRRuntimeSemanticSchema.h:535-566) is deterministic per
   `RuntimeState*`, with Published/Rejected terminal and immutable. Combined with the ownership
   lifecycle (`Allocated → Transferred → Committed → Completed` / `→ CallerDestroy`,
   AudioEngine.h:4614), every Accepted obligation terminates at a disposition-conserving state.

**The contract is CLOSED** because:

- A symbolic O obligation identity token on E is **not required** — the structural relation
  (frozen world object + verbatim triple + object-identity preservation + deterministic semantic
  state machine) is proven and exhaustive.

- The absence of correlationId on E is **by design** (telemetry-only), not a binding gap.
- The registry asymmetry is contained (does not enable mis-binding under the current model).

### PROVISIONAL scope (not PROVEN)

The determination is **PROVISIONAL** (not PROVEN) for the following reasons, all inherited from
3C-9 and 3D-8:

1. **No first-class O obligation identity symbol annotates E** — the binding is structural, not
   symbolic. If a literal "O identifier present in E artifact" is required as a proof convention,
   3E cannot produce it without code change (forbidden). The structural relation is proven; a
   symbolic annotation is not.

2. **Registry seqId-only asymmetry (B2-D/3D-8)**: `PendingPublishRegistry::lookup(seqId)` lacks
   epoch/mappedGeneration. Safe today only because (a) fallback-only, (b) seqId monotonic+unique
   per session, (c) drainAllNonRt idempotent on shutdown. Under a future multi-producer or
   seqId-reuse model, this would require `Entry{seqId, epoch, mappedGeneration}` (a code change).

3. **Single-CoordinatorLoop dependence**: uniqueness (B2) and registry safety rely on the
   single-CoordinatorLoop model (one consumer thread, one ISR dispatch). A multi-loop or
   parallel-publish model would require re-verification of the take-once invariant and registry
   keying. 3E does **not** propose or verify this change.

**The binding contract holds in the current single-CoordinatorLoop, SPSC OwnerChannel,
monotonic-seqId, deterministic-semantic-state-machine model.**

## 9. Phase I-T1-D101-2.5O Audit Completion

| Sub-phase | Status | Evidence file |
| --- | --- | --- |
| 2.5O-3A | ✅ CENSUS COMPLETE | `phase-I-T1-D101-2.5O-3A-accepted-execution-dataflow-census.md` |
| 2.5O-3B | ✅ COMPLETE | `phase-I-T1-D101-2.5O-3B-identity-binding-sufficiency-audit.md` |
| 2.5O-3C | ✅ COMPLETE (0 lint) | `phase-I-T1-D101-2.5O-3C-O-E-Binding-Existence-Uniqueness-Preservation.md` |
| 2.5O-3D | ✅ COMPLETE (0 lint) | `phase-I-T1-D101-2.5O-3D-failure-shutdown-ownership-conservation.md` |
| 2.5O-3E | ✅ COMPLETE (0 lint) | `phase-I-T1-D101-2.5O-3E-Execution-Binding-Contract-Closure.md` |

**2.5O (O→E Binding) closure: COMPLETE.** All four conditions (Existence, Uniqueness, Preservation,
Terminality) judged and proven under the current model. The Execution Binding Contract is CLOSED
(PROVISIONAL). The audit is **not** blocked on any remaining phase — 3D/3E were the final
2.5O sub-phases per 3A line 202.
