# D101-8 Step 4 — P Derivation (P — Publication-Side Bounded Concurrent Intent Residency)

> **Step 4-A: P_max の意味を固定**
> **Status**: ✅ COMPLETE — P counter corrected to two-layer (P_queue + P_accounting_reservation). P_max=4096 invalidated.
> **Status**: ✅ Step 4-B COMPLETE — production code exploration complete.
> **Status**: ✅ Step 4-C COMPLETE — candidate classification table.
> **Status**: ✅ Step 4-D COMPLETE — linearization split into accounting + enqueue + consumer-pop.
> **Status**: ⚠️ Step 4-E CONDITIONAL — producer contexts enumerated (2: Message Thread + RebuildThread), R3 non-reentrancy audited (NO REENTRANCY), R1-R4 + R-PROD invariants defined → R_max ≤ 2 (conditional upper bound).
> **Status**: ⚠️ Step 4-F CONDITIONAL — P_max ≤ 4098 (conditional upper bound, NOT established invariant).

---

## Step 4-A — P Semantic Definition (Two-Layer: P_queue + P_accounting_reservation)

### D101-8 Chain Context

| Step | Symbol | Meaning (D101-8 framework) | Status |
| --- | --- | --- | --- |
| Step 2 (A2) | Reservation Token | Design-only (WorldRetirementReservation D48-D53); 0 production matches | CLOSED, design-only |
| Step 3 (A3) | A_max | Lifetime Budget reservation acquire count | Design-defined, production evidence MISSING |
| Step 4 (this) | P | **Publication-side bounded concurrent intent residency** | **Production evidence found** |

### P — Precise Two-Layer Definition

Per reviewer correction: `publicationIntentResidencyCount_` is **not** bounded by `kIntentQueueCapacity` alone, because the counter is defined as **queue residency + producer-side reservation**. The counter can exceed 4096 during the window between `fetchAdd` and `push`.

Therefore P must be decomposed into two layers:

> **P_queue** = `intentQueue_` 内に residency している IntentType::Publish の数
> **P_accounting_reservation** = `enqueuePublicationIntent()` に入って `fetchAdd` 済みだが `push` が完了する前の producer reservation 数
> **P = P_queue + P_accounting_reservation**
> **publicationIntentResidencyCount_** = P_queue + P_accounting_reservation (counter definition, ISRRuntimePublicationCoordinator.h:539-545)

### ⚠️ `P_max = 4096` is INVALIDATED — counter can exceed queue capacity

The reviewer's interleaving is valid:

1. Producer A calls `fetchAdd(1)` → counter = 1, `intentQueue_.push()` not yet called
2. Producer B..N fill `intentQueue_` to full (4096 entries) via successful `push()`
3. At steady state: `P_queue = 4096`, `P_accounting_reservation >= 1`, **counter = 4097+**

Since `MpscBoundedRing` supports **multi-producer** (CAS-based `enqueuePos` reservation — confirmed by `MpscBoundedRing.h:1-30` and the comment at `ISRRuntimePublicationCoordinator.h:613-614` stating "intentQueue_ は既に複数 Producer... MPSC 実態"), there is **no architectural cap** on concurrent producers calling `enqueuePublicationIntent`.

**Therefore**: `publicationIntentResidencyCount_ <= 4096` is **UNPROVABLE** from the current code structure. The invariant `0 <= P_queue <= 4096` holds, but `P_queue + P_accounting_reservation <= 4096` does not.

### P_queue_max — the provable bound

```text
P_queue = number of IntentType::Publish entries actually resident in intentQueue_
P_queue_max = kIntentQueueCapacity = 4096

Proof: MpscBoundedRing<Intent, kIntentQueueCapacity> is bounded.
  push() returns false when full (no drop — "drop はしない" per MpscBoundedRing.h:28-30).
  Consumer pops (processIntent) reduce P_queue.
  ∴ 0 <= P_queue <= 4096. QED.
```

### P_accounting_reservation — the unproven component

`P_accounting_reservation` = number of producers that have executed `fetchAdd` but not yet completed `push()` in the reservation-before-push window.

**R_max** = max concurrent `enqueuePublicationIntent()` invocations in the reservation window — this is the upper bound of P_accounting_reservation. Per reviewer correction, R_max is **NOT** equivalent to "producer thread count" — it is the count of concurrent invocations in the reservation accounting window.

- **MPSC ring** supports concurrent producer invocation (CAS on `enqueuePos_`), so R_max > 0 is possible
- **No `kMaxConcurrentPublishers`** or serialization mutex caps the concurrent invocation count
- Producer threads are finite (Message Thread, RebuildThread, CoordinatorLoop) but no **design invariant** fixes R_max to a numerical bound
- Per reviewer: "MPSC が意味するのは「複数 producer から concurrent に push 可能」であれば、アプリケーションから無制限数の producer invocation が同時に存在する」ではない" — R_max is finite in practice but **not provable as a production invariant**

Since R_max is unproven as a numerical invariant, `P_max = 4096 + R_max` remains a **symbolic bound**, not a numerical bound.

### ⚠️ Explicit Exclusion — NO GlobalRecoveryBudget (D26 Recovery domain)

Per user warning: `GlobalRecoveryBudget` (`reservedLogicalObligations <= 32`, I4_D26) is **NOT** used as P_max source. Verified:

| Location | `GlobalRecoveryBudget` | `reservedLogicalObligations <= 32` |
| --- | --- | --- |
| `src/` | **0 matches** | **0 matches** |
| `ConvoPeq.md` | **0 matches** | **0 matches** |
| `doc/work88/I4_DESIGN_CONTRACT.md` | Present (D26, lines 736, 745, 766, 776, 815-816, 1041-1045) | Present |
| `doc/work88/I4_DESIGN_CONTRACT.md` D26 verdict | Recovery domain design contract | Recovery domain only |

**GlobalRecoveryBudget is Recovery domain (D26), NOT publication domain.** P_max source is the publication intent queue (`intentQueue_`), not the recovery budget.

---

## Step 4-B — Production Code Evidence

### P_source: The Publication Intent Queue (`intentQueue_`)

```text
src/audioengine/ISRRuntimePublicationCoordinator.h:692:  static constexpr size_t kIntentQueueCapacity = 4096;
src/audioengine/ISRRuntimePublicationCoordinator.h:693:  MpscBoundedRing<Intent, kIntentQueueCapacity> intentQueue_;
```

- **Type**: `MpscBoundedRing<Intent, kIntentQueueCapacity>` — Vyukov bounded MPSC ring (multi-producer: Builder/Rebuild/Timer/CoordinatorLoop; single-consumer: CoordinatorLoop)
- **Capacity**: `kIntentQueueCapacity = 4096` — compile-time constant
- **Bounded**: `MpscBoundedRing::push()` returns `false` when full — **no drop** (per header invariant: "drop はしない"). Rejection/dropped status surfaces via `PublishStageResult::Rejected` / `ReservationExhausted::Capacity`.

### P_counter: `publicationIntentResidencyCount_`

```text
src/audioengine/ISRRuntimePublicationCoordinator.h:542:  std::atomic<std::uint64_t> publicationIntentResidencyCount_{0};
```

**Header comment (lines 539-547):**

```cpp
// ★ work88 (X5 §6.5): Publish Intent residency 専用 counter（INV-X5-1）。
//   publicationIntentResidencyCount_ = intentQueue_ 内の Publish Intent 数 + producer
//   enqueue reservation（queue residency + producer-side reservation）。
//   - 対象: IntentType::Publish（enqueuePublicationIntent の単一箇所で reservation）
//   - 非対象: deferredPublicationCount_（Orchestrator の deferred state・単一スロット 0/1）
//     と hasDeferredCommit（commit 未完了の logical state）。Queue residency / Deferred
//     state / Commit completion を混ぜない（dash §6.5）。
//   - 増分: enqueuePublicationIntent が push 前に fetchAdd（reservation-before-push）
//   - 減分: processIntent の intentQueue_.pop で type==Publish の場合 fetchSub
//     （Publish pop は pendingIntentCount_ を触らない — P2-1 §1.1.6 W2）
std::atomic<std::uint64_t> publicationIntentResidencyCount_{0};
```

### P_event: enqueue → processIntent lifecycle

**`enqueuePublicationIntent`** (the sole producer-side reservation point for Publish intents):

```text
src/audioengine/ISRRuntimePublicationCoordinator.h:342:  //   Sole route that pushes an IntentType::Publish onto intentQueue_ (RETRY: FUTURE-10
src/audioengine/ISRRuntimePublicationCoordinator.h:373:      if (intentQueue_.push(prepared))
```

**Lifecycle (+1 / −1):**

| Layer | Event | Location | Operation | Direction |
| --- | --- | --- | --- | --- |
| P_accounting_reservation | Producer enqueue reservation | `enqueuePublicationIntent` (h:373) | `publicationIntentResidencyCount_.fetch_add(1)` before `intentQueue_.push()` | **+1** |
| P_queue | Queue push success | `enqueuePublicationIntent` (h:373) | `intentQueue_.push(prepared)` succeeds → slot reserved | +1 (queue) |
| P_queue − P_accounting_reservation | Consumer pop consumption | `processIntent` (pop from `intentQueue_`) | `publicationIntentResidencyCount_.fetch_sub(1)` when `type == Publish` | **−1** |

**Reservation-before-push (acquire)** → **pop-after-reservation-release (release)** ordering is enforced by `MpscBoundedRing`'s Vyukov algorithm: `enqueuePos` CAS reserves the slot atomically — the intent is visible in the reservation sequence before it is readable by the consumer.

### Admission Control: `PublicationAdmission::Decision`

```text
src/audioengine/PublicationAdmission.h:43:  //   trySubmitImpl の executor_.publish() 失敗時に使用。admission-time の
```

`PublicationAdmission::evaluate()` returns `Decision`:

- `Accepted` — proceeds to publish
- `RejectedPressure` — admission-time backpressure (P1-6 Adaptive Backpressure)
- `RejectedShutdown`, `RejectedStaleGeneration`, `RejectedNotFinalized`, `RejectedPublishFailure`, `DeferredFadingActive`, `RejectedLowPriority`

**Key distinction**: `Decision::Rejected` is an **admission-time** rejection — it prevents a Publish Intent from being enqueued (e.g., `RejectedPressure` throttles; `RejectedShutdown` blocks new intents during shutdown). The bounded authority `P_max` is the **queue residency** counter (`publicationIntentResidencyCount_`) bounded by `kIntentQueueCapacity`.

**Shutdown gate confirmed**: `enqueuePublicationIntent()` (ISRRuntimePublicationCoordinator.h:358) checks `CoordinatorState::ShuttingDown` before any `fetch_add` — during shutdown, the function returns `false` immediately, preventing both P_accounting_reservation and P_queue increments. This bounds producer activity during the shutdown phase but does NOT cap concurrent producers during normal operation (R_max remains unbounded by this gate — it only blocks new reservations during shutdown, not concurrent reservations during normal operation).

### P_facade: `commitRuntimePublication` — async facade, NOT the linearization point

Per Step 2 review feedback, `commitRuntimePublication` must be correctly scoped as a **producer-side async facade**, not a state transition itself. Verified in production code:

```text
src/audioengine/AudioEngine.h:4491:  // ★ B4: commitRuntimePublication — Producer が唯一利用する publish の入口（async facade）。
```

**Facade pipeline** (`enqueueRuntimePublicationFireAndForget`, AudioEngine.h:4509):

1. **`registerPublish(seqId, ...)`** → PendingPublishRegistry (64-slot metadata, non-owning)
2. **`OwnerChannel::enqueue(key, world)`** → ownership transfer (RuntimeWorldAuthority owns the channel; producer=Non-RT enqueue, ISR= `take()`)
3. **`enqueuePublicationIntent(intent)`** → ISR Intent Queue enqueue — **THIS is where P_accounting_reservation accounting + P_queue enqueue occurs**

**Key boundary**: `commitRuntimePublication` returns `{ Success, Transferred }` via `enqueueRuntimePublicationFireAndForget` — ownership is already moved to `OwnerChannel` and the Intent is enqueued to `intentQueue_` before this facade returns. The facade is the **producer's last checkpoint**; the actual P_accounting_reservation accounting (+1) and P_queue enqueue occur **inside** `enqueuePublicationIntent`.

**`enqueuePublicationIntent` implementation** (ISRRuntimePublicationCoordinator.h:344-374) confirms single-site reservation with rollback:

```cpp
// ★ work88 (X5 §6.5): fetchAdd before push, fetchSub on push-failure rollback.
convo::fetchAddAtomic(publicationIntentResidencyCount_, std::uint64_t{1}, ...);
if (intentQueue_.push(prepared))
    return true;
convo::fetchSubAtomic(publicationIntentResidencyCount_, std::uint64_t{1}, ...);
return false;
```

**Reservation accounting is centralized** — any call to `enqueuePublicationIntent()` performs the sole P_accounting_reservation accounting, regardless of which higher-level publish path originated it. The `publicationIntentResidencyCount_` counter is updated at exactly this single API point:

> ✅ Any call to `enqueuePublicationIntent()` performs the sole P_accounting_reservation accounting here.

- **push-fail rollback**: If `intentQueue_.push()` returns `false` (queue full), the `+1` is immediately reversed with `fetchSub` — and the caller (`enqueueRuntimePublicationFireAndForget`) reclaims the Owner via `ownerChannel().take(key)`. Ownership is never orphaned.
- **INV-X5-1**: `publicationIntentResidencyCount_` = queue residency + producer reservation (≥ during concurrent operation; == at producer quiescence).

### Linearization Point — Step 4-D (Corrected: three-layer)

The P linearizability is split into three distinct linearization points, due to the two-layer counter:

1. **Accounting linearization (P_accounting_reservation += 1)**: `enqueuePublicationIntent` calls `publicationIntentResidencyCount_.fetch_add(1)` **before** `intentQueue_.push()`. At this point the counter includes P_accounting_reservation but the intent is NOT yet queue-resident. This is the **producer reservation accounting point** — not the P_queue linearization. (The reviewer correctly notes that `fetch_add` alone does NOT constitute P_queue linearization, since `push()` may still fail and trigger `fetchSub` rollback.)

2. **Queue enqueue linearization (P_queue += 1)**: `MpscBoundedRing::push()` succeeds — `enqueuePos_` CAS reserves the slot, payload is written, sequence is released. At this point the intent is **queue-resident** (visible to consumer). This is when `P_queue` increments.

3. **Consumer release (P_queue −= 1, P_accounting_reservation net 0)**: `processIntent` pops from `intentQueue_` (single-consumer slot dequeue), then `fetch_sub(1)` on `publicationIntentResidencyCount_`. This decrements both P_queue and P_accounting_reservation net (since the reservation is completed and released).

**Key correction**: The `fetch_add` in `enqueuePublicationIntent` is **NOT** the P_queue linearization — it is the P_accounting_reservation accounting point. The counter `publicationIntentResidencyCount_` at `fetch_add` time includes the producer reservation but the entry is not yet queue-resident. The reviewer's interleaving (Producer A increments counter, Producers B..N fill queue) is exactly this scenario.

**Invariant (proven)**: `0 <= P_queue <= kIntentQueueCapacity` (4096) — enforced by `MpscBoundedRing` bounded capacity.
**Invariant (unproven)**: `0 <= P_accounting_reservation <= R_max` — requires producer concurrency bound proof.
**Invariant (unprovable without R_max)**: `0 <= publicationIntentResidencyCount_ (= P_queue + P_accounting_reservation) <= 4096` — **INVALID** without R_max = 0 proof.

---

## Step 4-C — Candidate Classification Table

| Candidate | Semantic Domain | P_max Source? | Evidence (file:line) | Verdict |
| --- | --- | --- | --- | --- |
| ✅ `publicationIntentResidencyCount_` | **Publication** (intent transport) | **YES — primary P counter (two-layer: P_queue + P_accounting_reservation)** | ISRRuntimePublicationCoordinator.h:542 | **P counter — NOT bounded by 4096 alone** |
| ✅ `kIntentQueueCapacity = 4096` | **Publication** (intent transport) | **YES — primary P bound** | ISRRuntimePublicationCoordinator.h:692 | **P_max = 4096** |
| ✅ `intentQueue_` (`MpscBoundedRing`) | **Publication** (intent transport) | YES — bounded ring structure | ISRRuntimePublicationCoordinator.h:693 | Bounded ring, push returns false when full |
| ✅ `enqueuePublicationIntent` (+1 reservation) | **Publication** (producer reservation) | YES — P_accounting_reservation +1 site | ISRRuntimePublicationCoordinator.h:373 | fetchAdd before push; rollback on full |
| ✅ `processIntent` (−1, type==Publish) | **Publication** (consumer release) | YES — P_event (−1) | ISRRuntimePublicationCoordinator.h (comment:547) | Paired release |
| ✅ `PublicationAdmission::Decision` | **Publication** (admission control) | NO — prevents enqueue, does not bound residency | PublicationAdmission.h:31-47 | Admission-time rejection, not residency bound |
| ✅ `pendingIntentCount_` | **Observe/Quarantine/Recovery** (non-Publish intents) | NO — INV-ISR-02: excludes Publish | ISRRuntimePublicationCoordinator.h:558 | Different intent class; P excludes |
| ✅ `retireBacklogCount_` | **Retirement** | NO — separate from Publish | ISRRuntimePublicationCoordinator.h:534 | Retire domain, not Publish |
| ✅ `hasDeferred_` (deferredPublicationCount_) | **Deferred publish** (single slot 0/1) | NO — comment explicitly excludes | ISRRuntimePublicationCoordinator.h:544 | "non-target" per counter doc |
| ✅ `PublicationAdmission::PressureLevel` | **Backpressure** (adaptive) | NO — throttle, not capacity bound | PublicationAdmission.h:38 | Pressure level, not bounded resource |
| ❌ `OwnerChannel::kCapacity = 256` | **Owner ownership slots** | NO — different axis | OwnerChannel.h:547-561 | Owner transfer slots, not intent residency |
| ❌ `PendingPublishRegistry` (64-slot) | **Audio-thread metadata** | NO — non-owning metadata ring | RuntimeWorldAuthority.h:207+ | Metadata, not intent residency |
| ❌ `pendingIntentCount_` (observe/retire) | **Observe/Quarantine/Recovery** | NO — excludes Publish (INV-ISR-02) | ISRRuntimePublicationCoordinator.h:558 | Semantic separation |
| ❌ `GlobalRecoveryBudget` (D26, <=32) | **Recovery** | **NO — must NOT use** | I4_DESIGN_CONTRACT.md:736,745,766,776,815-816,1041-1045 | **Recovery domain — 0 matches in src/ — explicitly excluded** |
| ❌ `RuntimePolicyEngine::n` | **Recovery storm detection** | NO — recovery domain | RuntimePolicyEngine.h:68 (`struct n`) | Recovery counter, not publication |
| ❌ `WorldRetirementReservation` (D48-D53) | **Reservation/retirement** | NO — design-only, not implemented | I4_DESIGN_CONTRACT.md | Step 2 confirmed: 0 production matches |

### Domain Separation Summary

| Domain | Authority | Bounded Resource | P? |
| --- | --- | --- | --- |
| **Publication (P)** | `intentQueue_` / `publicationIntentResidencyCount_` | `kIntentQueueCapacity = 4096` (P_queue bound); `R_max` unproven (P_accounting_reservation) | ⚠️ P_queue_max = 4096 PROVEN; P_max = 4096 INVALIDATED |
| **Reservation (A)** | `WorldRetirementReservation` (design-only) | D48-D53 (32) — **not implemented** | ❌ A_max (design) |
| **Recovery** | `GlobalRecoveryBudget` (D26, design-only) / `RuntimePolicyEngine::n` | `<= 32` / `kMaxRecover` — **not in src/** | ❌ Excluded |
| **Retirement (R)** | `retireBacklogCount_` | Dynamic backlog | ❌ Separate axis |
| **Owner** | `OwnerChannel` | `kCapacity = 256` | ❌ Different axis |
| **Admission** | `PublicationAdmission` | — | ❌ Prevents enqueue, doesn't bound residency |

---

## P — Final Derivation (Conditional Symbolic Upper Bound)

> **P_max は数値でなく、条件付きの記号式** — `P_max ≤ 4096 + R_max` は R_max が production invariant として証明されるまで仮の表現。P_queue_max = 4096 のみが設計上の確定的境界である。
> **Status: P_max ≤ 4098 (conditional upper bound)** — R_max ≤ 2 is derived from code audit (2 producer execution contexts: Message Thread + RebuildThread, CoordinatorLoop = consumer only, no reentrancy). P_max = 4098 is NOT yet an established production invariant. `C_prod = 2` is a code-derived architectural fact, not a structurally-enforced invariant.

```text
P_queue = number of IntentType::Publish entries resident in intentQueue_
P_accounting_reservation = producers that fetchAdd'd but not yet push'd (external accounting reservation)
P = P_queue + P_accounting_reservation
publicationIntentResidencyCount_ = P_queue + P_accounting_reservation  (counter definition)

P_queue_max = kIntentQueueCapacity = 4096   ← PROVEN invariant
P_accounting_reservation_max ≤ R_max ≤ 2   ← CONDITIONAL (R1-R4 + R-PROD, code-audit-derived)

P_max ≤ P_queue_max + P_accounting_reservation_max
    ≤ 4096 + 2
    = 4098                                    ← CONDITIONAL UPPER BOUND
                                              (NOT YET an established invariant)
```

### What IS proven from production code

1. **`publicationIntentResidencyCount_`** is the production P counter (ISRRuntimePublicationCoordinator.h:542) — defined as P_queue + P_accounting_reservation per its own comment
2. **`kIntentQueueCapacity = 4096`** is the bounded queue capacity (h:692-693)
3. **`enqueuePublicationIntent`** is the sole P_accounting_reservation accounting site with reservation-before-push + rollback (h:344-374) — all production publish paths converge to this single API
4. **`processIntent`** (pop, type==Publish) is the sole −1 site (h:547)
5. **`MpscBoundedRing`** is multi-producer safe — explicitly designed for concurrent producer invocation (MpscBoundedRing.h:1-30)
6. **`0 <= P_queue <= 4096`** — proven by `MpscBoundedRing` bounded capacity

### What is NOT proven

- **`P_max = 4096`** — **INVALIDATED** by the two-layer counter definition. Counter can exceed 4096 via P_accounting_reservation during producer push-delay.
- **`P_max = 4098`** — **NOT YET ESTABLISHED** as a production invariant. R_max ≤ 2 is derived from current production caller audit (2 producer execution contexts), but R-PROD invariants (R-PROD-3: same-context reentrancy prohibition) are code-audit-based conclusions, not structurally-enforced architectural invariants.
- **R-PROD formal invariance** — R3 non-reentrancy is confirmed by code audit (2 producer contexts: Message Thread + RebuildThread, CoordinatorLoop = consumer) but not formalized as an architectural invariant (e.g., INV- tag annotation). Formal assertion ≠ proof — the producer-context ownership invariant must be formally established.
- **Note on `sizeApprox()`**: `MpscBoundedRing::sizeApprox()` (MpscBoundedRing.h:133-134) is **best-effort** and includes producer reservations (it computes `enqueuePos_ - dequeuePos_`). It is **NOT** a proof of `P_queue <= 4096`. The proof comes from the **physical slot capacity** of the ring (4096 fixed slots), not from the approximate size measurement.

### Step 4-E — R_max Investigation (reservation-window concurrency)

**Revised R_max definition** (per reviewer correction):

> **R_max** = max concurrent `enqueuePublicationIntent()` invocations in the reservation-before-push window (not "producer thread count")

```text
R =
    count(producer invocations that have executed
        fetch_add(publicationIntentResidencyCount_, 1)
        AND
        corresponding push/rollback has not yet completed)
```

**Literal caller of `enqueuePublicationIntent()`** (single site, confirmed by grep):

| Caller | File:Line |
| --- | --- |
| `AudioEngine::enqueueRuntimePublicationFireAndForget` | AudioEngine.h:4569 |

**Callers of `enqueueRuntimePublicationFireAndForget`**:

| Call site | Thread | Path to `enqueuePublicationIntent` |
| --- | --- | --- |
| `AudioEngine.Processing.PrepareToPlay.cpp:155,277` | Message Thread | `commitRuntimePublication` → `enqueueRuntimePublicationFireAndForget` → `enqueuePublicationIntent` |
| `AudioEngine.Processing.ReleaseResources.cpp:175` | Message Thread | `commitRuntimePublication` → `enqueueRuntimePublicationFireAndForget` → `enqueuePublicationIntent` |
| `AudioEngine.Timer.cpp:964` | Message Thread (JUCE Timer) | `commitRuntimePublication` → `enqueueRuntimePublicationFireAndForget` → `enqueuePublicationIntent` |
| `AudioEngine.Transition.cpp:25` (via `publishIdleWorldOnly`) | Message Thread / Transition | `commitRuntimePublication` → `enqueueRuntimePublicationFireAndForget` → `enqueuePublicationIntent` |
| `AudioEngine.CtorDtor.cpp:79` | Message Thread (startup) | `publishIdleWorldOnly` → `commitRuntimePublication` → `enqueueRuntimePublicationFireAndForget` → `enqueuePublicationIntent` |
| `PublicationExecutor.cpp:53` (deferred resubmit) | **RebuildThread** (NOT CoordinatorLoop*) | `enqueueRuntimePublicationFireAndForget` → `enqueuePublicationIntent` |
| `AudioEngine.RebuildDispatch.cpp:1239` | RebuildThread | `enqueuePublicationIntentForRuntimeCommit` → `submitPublishRequest` → `trySubmitImpl` → `executor_.publish()` → `enqueueRuntimePublicationFireAndForget` → `enqueuePublicationIntent` |
| `AudioEngine.RebuildDispatch.cpp:987` | RebuildThread (recovery) | Same path as RebuildDispatch:1239 |
| `RuntimePublicationOrchestrator.cpp:542` | RebuildThread (deferred) | `submitPublishRequest` → `trySubmitImpl` → same path |

> ***Reviewer Round 3 correction**: The comment at RuntimePublicationOrchestrator.cpp:250 and PublicationExecutor.cpp:51 says "CoordinatorLoop 上の deferred resubmit" but this is misleading. `processDeferredAdmission` executes on RebuildThread (confirmed by `jassert(thread == rebuildThreadId())` at RuntimePublicationOrchestrator.cpp:528). `trySubmitImpl` calls `executor_.publish()` → `enqueueRuntimePublicationFireAndForget`. The "CoordinatorLoop" comment refers to the conceptual origin of the deferred request (set by CoordinatorLoop via `publishRetryReady` flag), NOT the execution context of the publish invocation. CoordinatorLoop only signals RebuildThread via `rebuildCV.notify_one()` (AudioEngine.Threading.cpp:278) and never directly calls `enqueuePublicationIntent`.

**Note on RebuildThread**: The reviewer notes that RebuildDispatch's primary path goes through `Orchestrator synchronous publish`. Code evidence confirms `submitPublishRequest` → `trySubmitImpl` → `executor_.publish()` → `commitRuntimePublication` → `enqueueRuntimePublicationFireAndForget` → `enqueuePublicationIntent`. **All production publish paths converge to the single `enqueuePublicationIntent` site.** The reservation accounting is centralized at this single API, regardless of originating thread/path.

**Concurrency findings:**

- **No `kMaxConcurrentPublishers`** or equivalent constant found in `src/audioengine/`
- **No serialization mutex** on `enqueuePublicationIntent` / `enqueueRuntimePublicationFireAndForget` — the enqueue path is lock-free (atomics only)
- **`rebuildMutex`** (AudioEngine.h:2608) only guards `recoveryPending` flag, NOT publish enqueue
- **`MpscBoundedRing`** is explicitly **multi-producer safe** (CAS on `enqueuePos_`), per `MpscBoundedRing.h:1-30` and the comment at `ISRRuntimePublicationCoordinator.h:613-614` ("intentQueue_ は既に複数 Producer... MPSC 実態")
- **Receipt-tracking mutex** (`mutex_` at AudioEngine.h:3712) guards `complete()` notification, NOT enqueue serialization (INV-X2-4)

**R_max verdict: CONDITIONAL — R_max ≤ 2 (code-derived architectural fact, pending structural enforcement).**

**R1 — Caller enumeration (COMPLETE):**

| R1 check | Result |
| --- | --- |
| Literal caller of `enqueuePublicationIntent` | ✅ Single site: `enqueueRuntimePublicationFireAndForget` (AudioEngine.h:4569) |
| All publish paths converge | ✅ Yes — Message Thread and RebuildThread route through the single API (CoordinatorLoop excluded) |
| Production caller inventory | ✅ 9 call sites identified (PrepareToPlay x2, ReleaseResources, Timer, Transition, CtorDtor, PublicationExecutor deferred, RebuildDispatch x2, RuntimePublicationOrchestrator) |

**R2 — Execution-context cardinality (COMPLETE):**

| Context | Thread | Concurrent instance count | Producer? |
| --- | --- | --- | --- |
| Message Thread | JUCE MessageManager thread | 1 | ✅ YES |
| RebuildThread | Dedicated `juce::Thread` (`rebuildThreadId()`) | 1 | ✅ YES |
| CoordinatorLoop | Dedicated `juce::Thread` (`coordinatorLoop_`) | 1 | ❌ NO (consumer + signaler only) |

> ***Reviewer Round 3 correction**: CoordinatorLoop is NOT a producer context. Its `runCoordinatorPhase` only calls `processIntent` (consumer), `drainOverflowRing` (consumer drain), and `rebuildCV.notify_one()` (signal — does NOT call `enqueuePublicationIntent`). Production publish invocation only occurs on Message Thread and RebuildThread.

```text
C_max = 2  ← CODE-DERIVED FACT (2 producer execution contexts confirmed: Message Thread + RebuildThread)
```

**R3 — Same-context reentrancy audit (COMPLETE — NO REENTRANCY FOUND):**

> **Audit question**: Between `fetch_add(publicationIntentResidencyCount_, 1)` (h:373) and `push()`/`fetchSub` rollback (h:373-375) within `enqueuePublicationIntent()`, does any code path invoke a callback, synchronous dispatch, or nested publication that could re-enter `enqueuePublicationIntent` from the same execution context?

**Code evidence (enqueuePublicationIntent h:358-375):**

```text
fetchAdd → intentQueue_.push() → [push succeeded: return true]
                                      ↓
                                 [push failed: fetchSub rollback + return false]
```

The `fetch_add` → `push`/`rollback` sequence is **directly adjacent** — no callbacks, no dispatch, no locks that could trigger re-entry. `MpscBoundedRing::push()` is `noexcept` with only atomic CAS + trivial copy.

**Upward call chain audit (producer → enqueuePublicationIntent):**

| Function in call path | Reentrancy risk? | Evidence |
| --- | --- | --- |
| `enqueueRuntimePublicationFireAndForget` (h:4509-4578) | ❌ NO | Only calls: `registerDSPHandleForRuntime` (map insert), `makePublishDecisionSnapshot` (pure data), `ownerChannel().enqueue` (atomics), `enqueuePublicationIntent` (the target). ScopeExit guard only calls `rollbackDSPHandleRegistration`. |
| `commitRuntimePublication` (h:4599-4610) | ❌ NO | Calls `enqueueRuntimePublicationFireAndForget`, then `waitForPublishReceipt` (CV wait — AFTER enqueue completes). |
| `submitPublishRequest` → `trySubmitImpl` | ❌ NO | Calls `admission_.evaluate()` (pure data), `RuntimeBuilder` (build), `executor_.publish()`. No callbacks. |
| `processDeferredAdmission` | ❌ NO | `jassert(thread == rebuildThreadId())` — RebuildThread only. Calls `peekDeferred`/`consume`/`submitPublishRequest`. No callbacks. |
| `enqueuePublicationIntentForRuntimeCommit` (h:782) | ❌ NO | Calls `registerDSPHandleForRuntime`, `submitPublishRequest`, `enqueueLearningCommand` (ring buffer write). No callbacks. |

**R3 verdict: NO REENTRANCY in current code.** No code path between `fetch_add` and `push`/`rollback` in `enqueuePublicationIntent` can re-enter `enqueuePublicationIntent` from the same execution context.

> **Reviewer note**: R3 non-reentrancy is a **code audit result**, not a formal architectural invariant. A code comment/annotation asserting non-reentrancy doesn't constitute a proof — what's needed is the **producer-context ownership invariant** (R-PROD-1 through R-PROD-4 below).

**R4 — Overlap verification (CONDITIONAL):**

Since R3 holds (no reentrancy in current code), each execution context can contribute at most 1 in-flight reservation at any time. With 2 concurrent producer execution contexts (R2):

```text
R_max ≤ C_max = 2  ← CONDITIONAL (requires R-PROD invariants, see below)
```

**Can both overlap simultaneously?** Message Thread (PrepareToPlay/Timer/etc.) and RebuildThread (enqueuePublicationIntentForRuntimeCommit/processDeferredAdmission) can have concurrent accounting reservations, since they are independent threads with no mutual exclusion on the enqueue path. While empirical overlap is timing-dependent, the **architectural upper bound is R_max ≤ 2** (conditional, code-derived).

> **Important**: R_max ≤ 2 does NOT automatically mean R_max = 2. Equality requires proof that both producer contexts can actually overlap simultaneously in the reservation window (constructive overlap proof). The reviewer correctly notes: "2 context の列挙" と "2 concurrent invocations の上限" は別証明。

### R-PROD: Producer-Context Ownership Invariants (required for R_max ≤ 2)

The reviewer correctly states that formal assertion ≠ proof. R_max ≤ 2 requires the following **architectural invariants** to hold:

**R-PROD-1 — Production caller context closure:**

`enqueuePublicationIntent()` の production invocation は、次の2 execution contextsからのみ発生する。

```text
C1 = Message Thread
C2 = RebuildThread
```

**Production evidence**: All 9 production call sites (5 Message Thread, 4 RebuildThread) trace through one of these 2 contexts. CoordinatorLoop is NOT a producer — it only consumes via `processIntent` and signals RebuildThread via `rebuildCV.notify_one()`. Test-only calls (`ISRSemanticValidationTests.cpp`, `ISRSoakTests.cpp`) are excluded from the production invariant.

**R-PROD-2 — Single-threaded execution context:**

各 execution context は single-thread execution context である。

**Production evidence**: Message Thread (JUCE MessageManager, single-threaded by definition), RebuildThread (dedicated `juce::Thread`, `jassert(std::this_thread::get_id() == rebuildThreadId())` at RuntimePublicationOrchestrator.cpp:528).

**R-PROD-3 — Same-context reentrancy prohibition:**

各 context について、

```text
fetch_add
    →
push / rollback
```

の間に同一 context から再入する経路は存在しない。

**Production evidence**: R3 audit above — all functions in the call chain between `enqueuePublicationIntent` invocation and the `push()` call are `noexcept`, atomic-only, or pure data structure operations. No callbacks, no synchronous dispatches, no nested `commitRuntimePublication`/`enqueueRuntimePublicationFireAndForget` calls exist in the path.

**R-PROD-4 — Single invocation per context in reservation window:**

各 context は reservation window に同時に1 invocation しか持てない（R-PROD-3 により、同一 context が再入しないため、1つの invocation が `push`/`rollback` を完了するまで、他の invocation は同じ context から開始されない）。

**Production evidence**: R-PROD-3 + sequential execution within a single thread = at most 1 active reservation per context.

> **Note**: R-PROD-1 through R-PROD-4 are code-audit-based conclusions derived from the **current production source**. They are not formal architectural invariants enforced by the type system. C_prod = 2 is a **code-derived architectural fact** — it reflects the current caller graph, thread ownership, and non-reentrancy. Formal architectural invariance would require either static enforcement (e.g., producer API only accessible from 2 context-tagged entry points) or invariant annotations that are structurally maintained. Comments alone are not proof; what matters is that the production caller closure and context execution seriality are **maintained as architectural invariants** in future code changes.
>
> ***Reviewer Round 3 note**: C_prod = 3 → corrected to 2. CoordinatorLoop is NOT a producer context. P_max ≤ 4098 (not 4099).**

### Two Types of Reservations (important distinction)

```text
P_accounting_reservation
    │
    ├─ external accounting reservation
    │      fetch_add (publicationIntentResidencyCount_)
    │      → intentQueue_.push()
    │
    └─ MPSC internal slot reservation (inside MpscBoundedRing::push)
           push() 内 CAS (enqueuePos_)
           ↓
           payload write
           ↓
           sequence release
```

> **Naming note**: `P_accounting_reservation` refers to the **external accounting reservation** (fetch_add → push). It must not be confused with the MPSC internal slot reservation (CAS on `enqueuePos_` inside `MpscBoundedRing::push`). Using distinct terminology avoids confusion in review.

### 3-Level Proof Structure

Per Reviewer Round 3 guidance, the proof is structured in 3 levels:

**Level 1 — P_queue_max = 4096 (PROVEN):**

```text
P_queue = number of IntentType::Publish entries actually resident in intentQueue_
P_queue_max = kIntentQueueCapacity = 4096

Proof: MpscBoundedRing<Intent, kIntentQueueCapacity> is a bounded container.
  push() returns false when full (no drop).
  Consumer pops (processIntent) reduce P_queue.
  ∴ 0 <= P_queue <= 4096. QED.
```

#### Level 1 — P_queue_max = 4096 (PROVEN)

**Level 2 — P_max ≤ P_queue_max + C_prod (conditional upper bound):**

```text
P = P_queue + P_accounting_reservation

P_accounting_reservation = count(producers in fetch_add → push/rollback window)

R-PROD-4: Each of the C_prod producer contexts can have at most 1 accounting
reservation in the window simultaneously.

∴ P_accounting_reservation ≤ C_prod
∴ P_max ≤ P_queue_max + C_prod
    ≤ 4096 + C_prod
```

**P_max ≤ 4096 + C_prod — CONDITIONAL UPPER BOUND** (depends on C_prod closure, see Level 3)

**Level 3 — C_prod = 2 (code-derived architectural fact, not formal invariant):**

```text
C_prod = number of production producer execution contexts

R-PROD-1: Production invocation of enqueuePublicationIntent() occurs only from
  C1 = Message Thread, C2 = RebuildThread

∴ C_prod = 2  ← CODE-DERIVED FACT (current production source)
  NOT a formal architectural invariant (requires structural enforcement)

P_max ≤ 4096 + 2 = 4098  ← CONDITIONAL UPPER BOUND (NOT established invariant)
```

> **Reviewer Round 3**: C_prod = 2 (CoordinatorLoop excluded — it is a consumer, not a producer). `C_prod = 2` is confirmed by current production caller audit but is not an architectural invariant that "can never exceed 2" without structural enforcement. If a new producer context (e.g., a new worker thread) calls `enqueuePublicationIntent`, C_prod would increase. P_max ≤ 4098 is a conditional upper bound. Equality (P_max = 4098) requires constructive overlap proof (both Message Thread and RebuildThread simultaneously in reservation window).

**Conclusion:**

```text
3-Level Proof:
  Level 1: P_queue_max = 4096                        ← PROVEN
  Level 2: P_max ≤ 4096 + C_prod                      ← CONDITIONAL UPPER BOUND
  Level 3: C_prod = 2 (Message Thread + RebuildThread) ← CONDITIONAL

  P_max ≤ 4098                                        ← CONDITIONAL UPPER BOUND (code-derived: C_prod=2)
  P_max = 4098                                        NOT YET ESTABLISHED (requires constructive overlap proof)
```

**Important note:** R_max ≤ 2 is derived from ConvoPeq's **current** producer topology (2 producer execution contexts: Message Thread + RebuildThread, with CoordinatorLoop as consumer-only). `C_prod = 2` is a **code-derived architectural fact**, not a formal invariant that "can never exceed 2" — a future code change adding a new producer context would invalidate it. To elevate P_max ≤ 4098 to a proven invariant, the production caller closure and context execution seriality must be maintained as structurally-enforced architectural invariants (e.g., producer API only accessible from context-tagged entry points), not merely documented in comments.

### Constructive Overlap for P_max = 4098

The constructive scenario achieving `P_max = 4098` is straightforward under the current topology:

```text
1. Queue fills to capacity:
   P_queue = 4096

2. Message Thread enters enqueuePublicationIntent:
   fetch_add(publicationIntentResidencyCount_, +1)  → counter = 4097
   ─────────────── preempt (before push completes)

3. RebuildThread enters enqueuePublicationIntent:
   fetch_add(publicationIntentResidencyCount_, +1)  → counter = 4098
   ─────────────── preempt (before push completes)

State: P_queue = 4096, P_accounting_reservation = 2
       P = 4098

4. Both producers reach push():
   → queue full → push() returns false → fetchSub rollback
   → both reservations released, counter returns to 4096
```

This demonstrates that both producer contexts can simultaneously hold an accounting reservation in the `fetch_add → push/rollback` window, making `P_max = 4098` **constructive reachable** under the current topology. However, equality is **NOT** a production invariant — it depends on timing/scheduling of both producer contexts, which is not structurally guaranteed.

> **See**: `evidence/phase-d101-8-step4-g-rprod-verification.md` for full Step 4-G structural verification of R-PROD-1 through R-PROD-4.

**Separation of P and I**: P (publication intent residency) and I (uncompleted publication lifecycles) are distinct metrics. `I_max > 1` (multiple worlds can be pending receipt timeout in PendingPublishRegistry/OwnerChannel) does NOT imply `R_max > 2` — P only counts accounting reservations in the `enqueuePublicationIntent` fetch_add→push/rollback window, not the broader publish lifecycle.

**Status: Step 4-E COMPLETE (conditional) — producer contexts enumerated (2: Message Thread + RebuildThread), R3 non-reentrancy audited (NO REENTRANCY), R1-R4 + R-PROD invariants defined → R_max ≤ 2 (conditional upper bound).**

Per reviewer guidance: "MPSC が意味するのは「複数 producer から concurrent に push 可能」であれば、アプリケーションから無制限数の producer invocation が同時に存在する」ではない" — R_max is bounded by execution-context cardinality (2: Message Thread + RebuildThread, CoordinatorLoop = consumer only) with non-reentrancy, but ≤ 2 is the conditional bound, not an established invariant.

### Relationship to A_max (Step 3)

| Property | A_max (Step 3) | P (Step 4) |
| --- | --- | --- |
| Semantic domain | **Reservation** (Lifetime Budget) | **Publication** (Intent Transport) |
| Definition | Successful Lifetime Budget reservation acquire events | Concurrent Publication-Intent in-flight (P_queue + P_accounting_reservation) |
| Bounded resource | `WorldRetirementReservation` (D48-D43, design-only) | `kIntentQueueCapacity = 4096` (queue) + `R_max = 2` (producer concurrency, producer-topology-derived) |
| Counter | `ReservationToken` acquire count (design-only) | `publicationIntentResidencyCount_` (production, two-layer) |
| Production evidence | ❌ MISSING (0 matches in src/) | ✅ FOUND (counter + capacity + R_max=2) / ⚠️ P_max=4098 provisional (R3 formal assertion pending) |
| Authority | `AdmissionAuthority → GlobalRecoveryBudget` (D26, Recovery domain) | `ISRRuntimePublicationCoordinator` (publication domain) |

---

## D101-8 Chain Completion (Steps 0-4)

| Step | Symbol | Definition | Production Evidence | Status |
| --- | --- | --- | --- | --- |
| Step 2 | Reservation Token | WorldRetirementReservation acquire | 0 matches in src/ | ✅ CLOSED (design-only, Phase I NO-GO) |
| Step 3 | A_max | Lifetime Budget reservation acquire count | 0 matches in src/ | ✅ CLOSED (design-defined, production MISSING) |
| Step 4 | P | Publication intent residency (two-layer: P_queue + P_accounting_reservation) | ✅ Counter + queue capacity found | ✅ P_queue_max = 4096 PROVED; P_max = 4096 INVALIDATED; R_max ≤ 2 (conditional, code audit); P_max ≤ 4098 (conditional) |
| Step 4-E | R_max | Max concurrent producers in reservation window | 2 producer execution contexts (Message Thread + RebuildThread), CoordinatorLoop = consumer only, no reentrancy (code audit), R-PROD invariants conditional | ✅ R1-R4 + R-PROD-1~4 → R_max ≤ 2 (conditional) |
| Step 4-F | P_max | Final P bound (4096 + R_max) | — | ⚠️ P_max ≤ 4098 (conditional upper bound, NOT established invariant) |
| Step 4-G | R-PROD | Structural verification of R-PROD-1..4 | Full source audit | ✅ R-PROD-1 PASS, R-PROD-2 PASS, R-PROD-3 PASS (through push() internals), R-PROD-4 PASS — see `phase-d101-8-step4-g-rprod-verification.md` |

---

## References

| Evidence | File | Lines |
| --- | --- | --- |
| `kIntentQueueCapacity` | `src/audioengine/ISRRuntimePublicationCoordinator.h` | 692-693 |
| `intentQueue_` (MpscBoundedRing) | `src/audioengine/ISRRuntimePublicationCoordinator.h` | 693 |
| `publicationIntentResidencyCount_` | `src/audioengine/ISRRuntimePublicationCoordinator.h` | 542 |
| Counter doc (+1/−1 semantics) | `src/audioengine/ISRRuntimePublicationCoordinator.h` | 539-547 |
| `enqueuePublicationIntent` (sole +1 path) | `src/audioengine/ISRRuntimePublicationCoordinator.h` | 342, 344-374 |
| `enqueueRuntimePublicationFireAndForget` (facade caller) | `src/audioengine/AudioEngine.h` | 4509, 4560-4578 |
| `pendingIntentCount_` (excludes Publish) | `src/audioengine/ISRRuntimePublicationCoordinator.h` | 558-566 |
| `MpscBoundedRing` (Vyukov bounded, push returns false) | `src/MpscBoundedRing.h` | 1-30 |
| `PublicationAdmission::Decision` | `src/audioengine/PublicationAdmission.h` | 31-47 |
| `GlobalRecoveryBudget` (excluded — Recovery domain) | `doc/work88/I4_DESIGN_CONTRACT.md` | 736, 745, 766, 776, 815-816, 1041-1045 |
| `RuntimePolicyEngine::n` (excluded — Recovery storm) | `src/audioengine/RuntimePolicyEngine.h` | 68 |
| `OwnerChannel::kCapacity = 256` (excluded — Owner axis) | `src/audioengine/OwnerChannel.h` | 547-561 |
| `PendingPublishRegistry` (excluded — metadata) | `src/audioengine/RuntimeWorldAuthority.h` | 207+ |
| `OwnerChannel` (SPSC, kCapacity=256, non-RT) | `src/audioengine/OwnerChannel.h` | 1-60 |
| `rebuildMutex` (not for enqueue serialization) | `src/audioengine/AudioEngine.h` | 2608, 4438 |
| `commitRuntimePublication` (async facade) | `src/audioengine/AudioEngine.h` | 4491, 4509, 4587-4599 |
| Producer call sites (PrepareToPlay/Timer/ReleaseResources/Transition) | `src/audioengine/AudioEngine.*.cpp` | 155, 277, 964, 175 |
| RebuildThread producer (enqueuePublicationIntentForRuntimeCommit) | `src/audioengine/AudioEngine.Commit.cpp` | 780-813 |
| `PublicationExecutor::publish` (deferral→enqueue path) | `src/audioengine/PublicationExecutor.h` | 19-48 |
| `enqueuePublicationIntent` literal caller (single) | `src/audioengine/AudioEngine.h` | 4569 |
| `MpscBoundedRing` (multi-producer safe, no cap) | `src/MpscBoundedRing.h` | 1-30 |
| `MpscBoundedRing::sizeApprox()` (best-effort, NOT proof basis) | `src/MpscBoundedRing.h` | 133-145 |
| `enqueuePublicationIntent` shutdown gate | `src/audioengine/ISRRuntimePublicationCoordinator.h` | 358 |
| `processIntent` consumer −1 (Publish) | `src/audioengine/ISRRuntimePublicationCoordinator.h` | 547 |
| `enqueueRuntimePublicationFireAndForget` call chain (no callbacks) | `src/audioengine/AudioEngine.h` | 4509-4578 |
| `commitRuntimePublication` (sync facade, wait AFTER enqueue) | `src/audioengine/AudioEngine.h` | 4587-4610 |
| `enqueuePublicationIntentForRuntimeCommit` (RebuildThread producer) | `src/audioengine/AudioEngine.Commit.cpp` | 782-820 |
| `submitPublishRequest` → `trySubmitImpl` (no callbacks) | `src/audioengine/RuntimePublicationOrchestrator.cpp` | 327-340 |
| `processDeferredAdmission` (CoordinatorLoop deferred, jassert rebuildThreadId) | `src/audioengine/RuntimePublicationOrchestrator.cpp` | 525-560 |
| `CoordinatorLoop` thread ownership (dedicated juce::Thread) | `src/audioengine/AudioEngine.Threading.cpp` | 256-260 |
| `rebuildThreadId()` assertion | `src/audioengine/AudioEngine.h` | 2500 |
| `enqueueLearningCommand` (ring buffer, no callbacks) | `src/audioengine/AudioEngine.h` | 4971-5005 |
| `DSPTransition::onPublishCompleted` (handler chain, no reentrancy) | `src/audioengine/DSPTransition.h` | 46-100 |
| `RuntimePublicationOrchestrator::onPublishCommitted` (notifyPublishReceipt only) | `src/audioengine/RuntimePublicationOrchestrator.cpp` | 313-320 |

---

## Step 4 Summary — Two-Layer P Model (Reviewer-Corrected)

```text
                    Publication side
                          │
              ┌───────────┴───────────┐
              │                       │
          P_queue               P_accounting_reservation
              │                       │
       intentQueue_          fetchAdd → push/rollback
              │                       │
       max = 4096                max ≤ R_max ≤ 2 (conditional)
              │                       │
              └───────────┬───────────┘
                          │
                          P
                          │
              publicationIntentResidencyCount_
                          │
                          ▼
                  P_max ≤ 4096 + 2 = 4098  (conditional upper bound;
                                              NOT an established invariant)
```

### Proof Status Summary

| Proposition | Verdict |
| --- | --- |
| `P_queue >= 0` | ✅ PROVEN |
| `P_queue <= 4096` | ✅ PROVEN |
| `P_accounting_reservation >= 0` | ✅ PROVEN |
| `P_accounting_reservation <= R_max` | ✅ by definition |
| `C_prod = 2` (producer execution contexts) | ⚠️ CURRENT-CODE FACT (Message Thread, RebuildThread; CoordinatorLoop = consumer only — NOT structural invariant) |
| `R3: same-context reentrancy` | ✅ AUDITED — NO REENTRANCY (callbacks/dispatches between fetch_add and push/rollback confirmed absent) |
| `R-PROD-3: same-context reentrancy prohibition` | ✅ R3 AUDITED (code-audit-based, not structural invariant) |
| `R_max <= 2` | ⚠️ CONDITIONAL (R1-R4 + R-PROD, code-audit-derived) |
| `P <= 4096` | ❌ INVALIDATED |
| `P_max = 4096` | ❌ INVALIDATED |
| `P_max = 4096 + R_max` | ✅ symbolic |
| `P_max ≤ 4096 + 2 = 4098` | ⚠️ CONDITIONAL UPPER BOUND (current-code fact, not structural invariant) |
| `P_max = 4098` | ⚠️ Constructive overlap possible (queue full + two fetch_add before push); equality NOT proven as production invariant |
