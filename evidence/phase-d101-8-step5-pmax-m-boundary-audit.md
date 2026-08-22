# D101-8 Step 5 — P_max → M Boundary Audit

> **Status**: COMPLETE — P_max remains CONDITIONAL (`P_max ≤ 4098`); the audit confirms
> P_queue_max = 4096 PROVEN; the structural world-bearing relation `P_queue ≤ OwnerChannel(k=256)`
> and the publish/unpublished domain exclusion (D101 Tier-4 fix) are established.
> P_max は **M の構造的上界とならない** — M は observation-error/envelope bound (D101 = OPEN)。
> **Code changes: none** — audit only.
> Date: 2026-08-22
> Verification: all listed tool families executed (WSL rg/ast-grep/sed/awk/fdfind/fd/ag/fzf,
> serena, AiDex MCP, graphify, semble; ccc attempted — requires `ccc init`).

---

## 5A — P_max final status (formal frozen)

### Frozen from Step 4 (NOT re-elevated)

```text
P_queue_max = 4096                          PROVEN   (MpscBoundedRing physical slots, h:692-693)
P_accounting_reservation ≤ R_max            DEFINITIONAL  (definition of reservation window)
R_max ≤ 2                                   CONDITIONAL  (R-PROD-1..4 code-audit, see 5B)
C_prod = 2                                  CODE-DERIVED FACT  (Message Thread, RebuildThread)
P_max ≤ 4096 + 2 = 4098                     CONDITIONAL UPPER BOUND
P_max = 4096                                INVALIDATED  (two-layer counter, reviewer correction)
P_max = 4098                                REACHABLE but NOT architectural invariant
```

### Case determination (5E — preliminary)

```text
R_max ≤ 2 の production-code 証拠は存在する（producer execution-context enumeration + non-reentrancy）
だが, それらは **structure-enforced invariant** ではなく **code-audit-derived conditional** である
（R-PROD-1..4 は型システムで強制されていない）。

⇒ Step 5-E 分類: Case B (R_max ≤ 2 の formal proof に production-code evidence が不足)
```

```text
P_queue_max = 4096                     PROVEN
P_accounting_reservation ≤ R_max       definitionally true
R_max ≤ 2                              CONDITIONAL (R-PROD, code-audit-derived — see 5B)
P_max ≤ 4096 + R_max                   CONDITIONAL (symbolic)
P_max ≤ 4098                           CONDITIONAL UPPER BOUND (current-code fact, NOT structural)
```

> **警告**: `4098` は新しい設計定数ではない。`4096` は queue architectural capacity、`R_max (=2)` は transient accounting gap。`4098 = conditional upper bound` であり `architectural invariant` ではない。P_max を B_max/M_max の直接値として用いないこと。

---

## 5B — R_max structural proof (re-audit)

### Re-statement of R_max

> **R_max** = max concurrent `enqueuePublicationIntent()` invocations in the fetch_add → push/rollback window (NOT "producer thread count")

### R-PROD-1 — Production caller closure (structural strength: code-audit, not type-enforced)

**Single literal production caller** (AiDex verified, exact match):

| Call site | Production? | Evidence |
|---|---|---|
| `AudioEngine.h:4577` inside `enqueueRuntimePublicationFireAndForget` | ✅ YES | AiDex: `enqueuePublicationIntent` — 1 production call; 16 test call sites (tests excluded) |
| `PublicationExecutor.cpp:57` (deferred resubmit) | ✅ YES — routes through SAME `enqueueRuntimePublicationFireAndForget` | code: `engine.enqueueRuntimePublicationFireAndForget(...)` |

AiDex exact query `enqueuePublicationIntent` returned: **1 production caller** (AudioEngine.h:4577) + 16 test callers (ISRSemanticValidationTests.cpp, ISRSoakTests.cpp — excluded).

**All production publish paths** converge to the single API:
```text
RebuildThread: RebuildDispatch → submitPublishRequest → trySubmitImpl → build → executor_.publish → enqueueRuntimePublicationFireAndForget → enqueuePublicationIntent
Message Thread: PrepareToPlay/Timer/ReleaseResources/Transition/CtorDtor → commitRuntimePublication(*) → enqueueRuntimePublicationFireAndForget → enqueuePublicationIntent
```

### R-PROD-2 — Single-threaded execution context (structural strength: code-audit)

| Context | Thread | Count | Producer? |
|---|---|---|---|
| Message Thread | JUCE MessageManager | 1 | ✅ |
| RebuildThread | dedicated `juce::Thread` (`jassert(thread == rebuildThreadId())`) | 1 | ✅ |
| CoordinatorLoop | dedicated `juce::Thread` | 1 | ❌ (consumer only — processIntent + notifyRebuild) |

`CoordinatorLoop` is **NOT** a producer: `runCoordinatorPhase` only calls `processIntent` (consumer), `drainOverflowRing` (consumer), `rebuildCV.notify_one()` (signal — does NOT call enqueuePublicationIntent). Verified at audioengine/AudioEngine.Threading.cpp:256-260 + RuntimePublicationOrchestrator.cpp:525-560 (`jassert(thread == rebuildThreadId())` for processDeferredAdmission — runs on RebuildThread, NOT CoordinatorLoop).

> **Reviewer Round-3 correction confirmed**: the comment at RuntimePublicationOrchestrator.cpp:250 ("CoordinatorLoop 上の deferred resubmit") describes the *conceptual origin* (CoordinatorLoop set the `publishRetryReady` flag), NOT the execution context.

**Structural strength**: C_prod = 2 is **current-code topology fact**, not type-system-enforced invariant. No `static_assert` / interface constraint prevents a future 3rd producer thread from calling `enqueueRuntimePublicationFireAndForget`.

### R-PROD-3 — Same-context reentrancy (structural strength: code-audit)

**Audit**: Between `fetch_add(publicationIntentResidencyCount_, +1)` (h:369) and `push()`/`fetchSub` rollback (h:370-375) within `enqueuePublicationIntent()`:

Code evidence (ISRRuntimePublicationCoordinator.h:365-375, re-read for Step 5):
```text
convo::fetchAddAtomic(publicationIntentResidencyCount_, std::uint64_t{1}, std::memory_order_acq_rel);
if (intentQueue_.push(prepared))
    return true;
convo::fetchSubAtomic(publicationIntentResidencyCount_, std::uint64_t{1}, std::memory_order_acq_rel);
return false;
```

The `fetch_add → push/rollback` is **directly adjacent** — no callbacks, no dispatch, no locks that could trigger re-entry. `MpscBoundedRing::push()` is `noexcept` with only atomic CAS + trivial copy (verified at MpscBoundedRing.h:1-30).

**Upward call chain** (producer → enqueuePublicationIntent) — re-audited for Step 5-D:

| Function | Reentrancy risk? | Evidence |
|---|---|---|
| `enqueueRuntimePublicationFireAndForget` (AudioEngine.h:4509) | ❌ NO | registerDSPHandleForRuntime → makePublishDecisionSnapshot → ownerChannel().enqueue → enqueuePublicationIntent. ScopeExit guard only calls rollbackDSPHandleRegistration. |
| `commitRuntimePublication` (AudioEngine.h:~4605) | ❌ NO | calls fire-and-forget THEN waits receipt (after enqueue done). |
| `submitPublishRequest` → `trySubmitImpl` (h:324-330) | ❌ NO | admission_.evaluate (pure) → buildRuntimePublishWorld → executor_.publish. No callbacks. |
| `processDeferredAdmission` (h:525-560) | ❌ NO | `jassert(thread == rebuildThreadId())` — RebuildThread. peekDeferred/consume/submitPublishRequest. No callbacks. |
| `enqueuePublicationIntentForRuntimeCommit` (AudioEngine.h:782) | ❌ NO | registerDSPHandleForRuntime → submitPublishRequest → enqueueLearningCommand (ring buffer write). No callbacks. |
| `executePublish` (RuntimePublishExecutor.h:~45) | ❌ N/A | This is the **consumer** path (processIntent → PublishIntentHandler → executePublish). It calls ownerChannel.take, NOT enqueuePublicationIntent. Not a producer-context re-entry. |

**Verdict**: NO REENTRANCY in current code. But this is **code-audit-based**, not **structural invariant** — a comment/annotation asserting non-reentrancy ≠ proof; the producer-context ownership invariant must be formally established.

### R-PROD-4 — Single reservation per context (derived)

R-PROD-2 (single thread per context) + R-PROD-3 (no reentrancy) ⟹ each context can contribute at most 1 in-flight reservation at any time.

### R-PROD-3-in-depth: push() internals (no hidden reentrancy)

`MpscBoundedRing::push` (MpscBoundedRing.h, re-confirmed):
```text
enqueuePos_ CAS (atomic) → payload copy (trivially copyable Intent) → seq release (atomic)
```
`static_assert(std::is_trivially_copyable_v<Intent>)` (h:336) guarantees no constructor/destructor side effects during push. NO reentrancy path inside push.

### Final R_max verdict: CONDITIONAL (Case B)

```text
C_prod = 2                    code-derived topology fact (NOT structural)
R-PROD-1: caller closure      ✅ PASS (code-audit, not type-enforced)
R-PROD-2: single-threaded     ✅ PASS (jassert thread affinity, code-audit)
R-PROD-3: non-reentrancy      ✅ PASS (audited through push() internals, no callbacks)
R-PROD-4: single reservation  ✅ PASS (derived from R-PROD-2 + R-PROD-3)
R_max ≤ 2                     CONDITIONAL (requires R-PROD structural enforcement)
```

> `R_max ≤ 2` は「R-PROD コメントを追加したから証明される」のではなく,
> 現行 production caller closure + 各 producer context の single-thread execution +
> same-context non-reentrancy から導出される **application-level invariant**.
> これを architecture invariant として維持・検証可能な形に固定するまで
> (type-system enforcement / context-tagged entry points), `R_max ≤ 2` は conditional。

**Step 5-E classification: Case B.**
```text
P_queue_max = 4096             PROVEN
P_accounting_reservation ≤ R_max         DEFINITIONAL
R_max ≤ 2                               CODE-EVIDENCE INCOMPLETE (conditional)
P_max ≤ 4096 + R_max                    CONDITIONAL
P_max ≤ 4098                           CONDITIONAL UPPER BOUND
P_max = 4098                           NOT ESTABLISHED
```

---

## 5C — Publish Intent → World cardinality (full audit)

### C-1: The production enqueue pipeline (verified, AudioEngine.h:4523-4588)

Exact ordering at `enqueueRuntimePublicationFireAndForget`:

```text
1. registerDSPHandleForRuntime(regCtx.dsp)  → rollbackHandle (if dsp != null)
2. newWorld = world.get()  [null check]
3. seqId / epoch / mappedGen extracted
4. decision = makePublishDecisionSnapshot(newWorld, ...)
5. registry().registerPublish(seqId, newWorld)   ← non-owning metadata
6. IF ownerChannel().enqueue(key, std::move(world)) FAILED → unregister + return Failed/CallerDestroy
7. intent.payload.publish.* populated (NON-OWNING: const void* newWorld = world.get())
8. IF enqueuePublicationIntent(intent) FAILED → ownerChannel().take(key) reclaims + unregister + return Failed/CallerDestroy
9. rollbackHandle = null  ← ownership transfer confirmed
10. return Success/Transferred
```

**Critical invariant (Step 5 pivotal finding):**
> **World ownership transfer into OwnerChannel (step 6) ALWAYS precedes Intent enqueue (step 8).**
> The Intent payload (PublishPayload) holds a **non-owning** `const void* newWorld`
> (h:308 comment: "HANDLER-1 read-only"; trivially-copyable static_assert h:336).

### C-2: PublishPayload — non-owning (verified, ISRRuntimePublicationCoordinator.h:306-314)

```cpp
struct PublishPayload  {
    DSPHandle handle;                       // (retained; 5-2 migrates newWorld from sealedSnapshot)
    const void* newWorld;                    // ★ HANDLER-1 read-only (non-owning)
    std::uint64_t version;                   // fixed at enqueue
    PublicationEpoch epoch;                  // fixed at enqueue
    std::uint64_t mappedGeneration;          // fixed at enqueue
    RuntimeBoundary boundary;                // fixed at enqueue
    PublishDecisionSnapshot decision;        // HANDLER-1 read-only, fixed at enqueue
};
```
- `newWorld` is `const void*` — **non-owning, read-only**. No `unique_ptr`, no `shared_ptr`.
- Intent is `trivially_copyable` + `standard_layout` (h:336-339) — the static_assert proves it CANNOT hold an owning smart pointer (which would have a non-trivial deleter).
- Comment h:347-348: "Intent carries only the transport payload (Pointer + build-time metadata + decision), so the ISR coordinator stays ignorant of RuntimeState (no circular include)."

### C-3: Admission → build ordering (verified, RuntimePublicationOrchestrator.cpp:40-310)

`trySubmitImpl` exact structure:
```text
:40  auto decision = admission_.evaluate(req, engine_, pubCtx)   ← admission FIRST
:46  if (decision != Decision::Accepted) → return (NO world built)
...
:165 auto worldOwner = worldBuilder.buildRuntimePublishWorld(...)  ← build AFTER admission pass
:167-180 if (!worldOwner)  → retire newDSP ; return RejectedNotFinalized  ← build failure: 0 worlds
...
:232 worldOwner = worldBuilder.buildRuntimePublishWorld(...)  ← crossfade re-build (1st destroyed)
:234-243 if (!worldOwner) → retire + RejectedNotFinalized
...
:~300 auto result = executor_.publish(engine_, std::move(frozen), ...)  ← world moved to publish
```

**Two build sites** (:165 pre-crossfade, :232 post-crossfade rebuild):
- If crossfade needed (:232): `worldOwner = worldBuilder.buildRuntimePublishWorld(...)` **reassigns** — the 1st `worldOwner` (from :165) is destroyed via `aligned_unique_ptr` destructor (RAII). **Only 1 world survives per trySubmitImpl.**
- Build failure (both sites): `worldOwner` is `nullptr`/empty → `return RejectedNotFinalized` → executor_ never called → **0 worlds**.

### C-4: The 9 audit questions — answers with production evidence

| # | Question | Answer | Evidence |
|---|----------|--------|----------|
| C1 | 1 intent が必ず 1 world を生成するか？ | **NO** — 1 successful enqueue ⟹ 1 world in S2. Build failure/admission reject ⟹ 0 worlds. | trySubmitImpl :46 (admission gate before build); :167/`!worldOwner` (build fail → 0) |
| C2 | 生成しない経路はあるか？ | **YES** — Admission rejection (6 types) / Build failure (InvalidInput/ResourceUnavailable/InternalError/WarmupFailed) | PublicationAdmission.h:31-47; RuntimeBuilder.cpp:410-460 |
| C3 | Build failure 時 World が生成されるか？ | **NO** — `worldOwner` is nullptr → `return RejectedNotFinalized` (executor_.publish 未到達) | RuntimePublicationOrchestrator.cpp:167-180 |
| C4 | Stale discard 時 World が生成されるか？ | **NO** — staleness checked at **admission** (evaluate → RejectedStaleGeneration) BEFORE any build/world | RuntimePublicationOrchestrator.cpp:338-341 (stale → FailureReason::StaleGeneration) |
| C5 | Publish rejection 時 World が生成されるか？ | **YES, transient** — world built + placed in OwnerChannel(S2), then: (a) enqueue fails → take() reclaims+destroy; (b) publishAndSwap faults (monotonicity) → authority.publish destroys inside | AudioEngine.h:4559 (enqueue may fail→take reclaim); RuntimePublishExecutor.h:82-88 (publish failure → world destroyed in publish()) |
| C6 | 1 intent が複数 World を生成し得るか？ | **NO** — crossfade rebuild (:232) destroys 1st world via RAII reassign; deferred resubmit creates fresh intent+world (old intent already popped/consumed) | trySubmitImpl :165 vs :232 (reassignment) |
| C7 | 1 world が複数 Intent に対応し得るか？ | **NO** — 1:1 identity mapping. `seqId = world.publication.sequenceId`; intent.sequenceId = same seqId. OwnerChannel take(key) drains owner exactly once (single-transfer). | PublishPayload (h:306-314); OwnerChannel.h:92-111 (take drains once); AudioEngine.h:4559-4561 (key = seq/epoch/gen) |
| C8 | Accounting reservation が World lifetime を保持するか？ | **NO** — World ownership already transferred to OwnerChannel (step 6) BEFORE fetch_add at enqueuePublicationIntent. The reservation holds only a counter increment. | AudioEngine.h:4559 (ownerChannel.enqueue BEFORE h:4577 enqueuePublicationIntent); ISRRuntimePublicationCoordinator.h:369 (fetch_add is counter-only) |
| C9 | Queue residency と World residency の時間的 overlap？ | **YES** — intent queued [t_enqueue, t_pop] ⟹ world in OwnerChannel(S2) for entire [t_enqueue, t_pop]. Overlap = queue dwell time. After pop: S2→S3 (commit) OR S2→S7 (destroy). | processIntent: pop→PublishIntentHandler→executePublish→take; RuntimePublishExecutor.h:69-76 |

### C-5: Consumer consume path (verified, RuntimePublishExecutor.h:20-108 + ProcessIntent.cpp)

`processIntent` → `kDispatchTable[Publish]→PublishIntentHandler::handle` → `PublishExecutor::executePublish`:
```text
:23  auto owner = authority.ownerChannel().take(key)   ← SOLE ownership claim (single-transfer)
:27  const auto* newWorld = owner ? owner.get() : registry.lookup(seqId);   ← fallback to registry
:46  RuntimeState* oldWorld = authority.publish(std::move(owner), metadata, &committed)  ← publishAndSwap
     if committed:
       :74  bridge.didPublishRuntimeNonRt(*newWorld)
       :75  bridge.willRetireRuntimeNonRt(oldWorld)
       :76  bridge.retirePublishedRuntimeWorldNonRt(oldWorld, false)   ← DeletionEntryType::World → onRelease
:88  authority.registry().unregister(intent.sequenceId)   ← drop metadata fallback entry
    ...
:98  ctx.transition.onPublishCompleted(newResolved, oldResolved, ...)   ← activate/crossfade/retire tail
:~104 ctx.engine.runtimeOrchestrator_->onPublishCommitted(intent.sequenceId)   ← notifyPublishReceipt
```

**Single take, single commit, single publishAndSwap** — `INV-X4-3: RuntimeStateOwner is moved exactly once below (into RuntimeWorldAuthority::publish)`. Comment at h:20: "sole physical store-swap, INV-X4-3".

### C-6: Cardinality summary table

| Relationship | Cardinality | Production evidence |
|---|---|---|
| 1 successful intent enqueue ⟺ worlds in OwnerChannel | **1:1 (non-owning intent)** | PublishPayload.newWorld = `const void*` (non-owning); OwnerChannel single-transfer take |
| 1 world ↔ 1 intent | **1:1 identity** | seqId binding (intent.sequenceId = world.publication.sequenceId) |
| Build failure | **0 worlds** | trySubmitImpl `!worldOwner` → RejectedNotFinalized, no enqueue |
| Admission reject | **0 worlds** | evaluate() gates BEFORE build |
| Enqueue push-fail | **0 worlds retained** | take() reclaim + destroy, Intent rollback fetchSub |
| Crossfade rebuild | **≤1 world** | 2nd build destroys 1st via RAII reassign |

---

## 5D — P-state → B_max contribution (structural proof)

### The pivotal structural theorem

```text
THEOREM:  contribution_to_B_max(P-state) = 0  (P adds no worlds beyond OwnerChannel's bound)

∀ t:  P_queue(t) ≤ |worlds in OwnerChannel(t)| ≤ kCapacity = 256
```

### Proof

**Step 1 — Enqueue ordering establishes ownership-before-intent (h:4559-4588):**

```text
enqueueRuntimePublicationFireAndForget(world, regCtx, oldHandle):
    5. registry().registerPublish(seqId, newWorld)        ← non-owning metadata
    6. ownerChannel().enqueue(key, std::move(world))      ← OWNERSHIP TRANSFER (S0→S2)
       if FAIL: unregister + destroy world, return Failed  ← NO intent enqueued
    8. enqueuePublicationIntent(intent)                    ← intent queued
       if FAIL: ownerChannel().take(key) → reclaim+destroy ← world NOT leaked
    9. rollbackHandle = null; return Success/Transferred
```

**Invariant I-5D.1 (ownership precedence):**
> `intent ∈ intentQueue_.queue` ⟹ `∃ key. OwnerChannel.slots[key].owner ≠ nullptr`

Proof: enqueuePublicationIntent is reached ONLY after ownerChannel().enqueue succeeded (step 6). If step 6 fails, step 8 is never reached. If step 8 fails, ownerChannel().take reclaims (step 8 rollback). ∎

**Step 2 — Intent is non-owning (h:306-336):**

PublishPayload holds `const void* newWorld` (read-only pointer, HANDLER-1). Intent is `trivially_copyable` (h:336-339) ⇒ cannot hold an owning smart-pointer. ∴ the Intent itself holds **zero owned worlds**.

**Step 3 — OwnerChannel single-transfer (OwnerChannel.h:67-111):**

- `enqueue`: slot `s.key = key; publish(owner)` — single producer, ownership to slot. Rejects duplicate key (no overwrite).
- `take`: drains `s.owner` to nullptr exactly once (single-transfer). `consume→publish(nullptr)` pattern means re-drain is a no-op.

**Step 4 — Count the co-resident worlds:**

```text
worlds simultaneously alive in S2 = |{ slot ∈ OwnerChannel : slot.owner ≠ nullptr }|
                                  ≤ kCapacity = 256      (fixed array<Slot, 256>)
```

By I-5D.1, `P_queue(t) ≤ |worlds in OwnerChannel(t)|`. ∎

**Step 5 — Accounting reservation contributes nothing (fetch_add → push window):**

During `P_accounting_reservation` (fetch_add at h:369 before push):
- The world is **already in OwnerChannel** (step 6 completed before step 8 entry at AudioEngine.h:4559).
- The reservation holds only a counter increment (`publicationIntentResidencyCount_`), no owning pointer.
- ∴ `P_accounting_reservation` holds **0 worlds** beyond OwnerChannel.

### Tightening over Step 4

Step 4 established `P_queue ≤ 4096` (physical ring slots) — **correct as architectural queue capacity**. Step 5 establishes the **tighter world-bearing bound**:

```text
P_queue(t) ≤ min(kIntentQueueCapacity, OwnerChannel occupancy carrying those intents' worlds)
           ≤ min(4096, 256)
           = 256        (worlds co-resident with queued intents)
```

BUT: `kIntentQueueCapacity = 4096` remains the **architectural** bound on queued non-world-bearing slot-occupancy (an intent can be queued transiently without a world ONLY during the fetch_sub rollback window — sub-microsecond, and even then OwnerChannel.take already reclaimed). For the **world-budget** (K_world / B_max), the binding structural term is **OwnerChannel = 256**, which is **already accounted in K_transferred (S2)**.

### Contribution function

```text
∀ P-state:
    contribution_to_B_max(P-state) = 0   (additive worlds beyond S2 term)

f(P_max) = 0  (additive);  P_queue ≤ 256 is CONTAINED within K_transferred(S2)
```

**Corollary (5D):** P_max does NOT extend B_max^true additively. The worlds carried by queued intents are a **subset** (bounded by 256) of the S2 OwnerChannel budget already counted in K_world. `P_max` bounds **intent transport capacity**, not **world lifecycle budget**.

### K_transferred reconciliation (correction to Steps 6/7/8 summaries)

| Source | Earlier summary | **Step 5 correction (current code)** |
|---|---|---|
| OwnerChannel | "1 per channel / single Owner slot" | **kCapacity = 256** (OwnerChannel.h:41: `static constexpr std::size_t kCapacity = 256; // >> max in-flight publishes`) — owning, single-transfer, SPSC |
| PendingPublishRegistry | 64 (counted in K_transferred) | **64 non-owning metadata** (RuntimeWorldAuthority.h:34) — NOT owning; does not add world budget |
| RuntimeStore::current | 1 (S3) | 1 (S3 — published/world current) |

⇒ **K_transferred(S2) ≤ 256** (OwnerChannel owners; registry is non-owning metadata gap buffer, bounded separately at 64 but contributing 0 owned-world count).

### Publish-rejection / world-destroy paths (5C.5 — bounded destruction, no leak)

All failure paths return OwnershipDisposition and ensure world destruction:

| Failure point | World fate | DeletionEntryType | onRelease? |
|---|---|---|---|
| ownerChannel.enqueue fails (full) | caller destroys (unique_ptr, AudioEngine.h:4570 path) | — | no (never enqueued) |
| enqueuePublicationIntent fails (queue full) | ownerChannel.take reclaims → caller destroys (AudioEngine.h:4580) | — | no |
| publishAndSwap Faulted (monotonicity) | authority.publish() destroys internally (RuntimePublishExecutor.h:82-88) | — | no |
| build failure (buildRuntimePublishWorld null) | unique_ptr destroyed (RAII) | — | no |
| successful publish, old world retired | retirePublishedRuntimeWorldNonRt → enqueueDeferredDeleteNonRt | World | **YES** |
| rejected world (unpublished) retire | retireRejectedRuntimeWorldNonRt → enqueueDeferredDeleteNonRt | Generic | **no** |

### D → Q → E → Terminal ownership-transfer chain (no leak)

`enqueueDeferredDeleteNonRt` (AudioEngine.h:4201) delegates to `m_retireRouter->enqueueWithRetry` (ISRRetireRouter.cpp:282):
- D full → retry + tryReclaim → Q (RetireQuarantineStore kMax=512, mutex)
- Q full → E (EmergencyQuarantineStore kMax=512)
- E full → Terminal (TerminalReclaimAuthority, std::vector growable, `store() ALWAYS true`)

Comment ISRRetireRouter.cpp:25: **"invariant: enqueueWithRetry() never returns with ptr unowned."** Each stage `store()`s or `push_back`s — ownership always held by some authority. shutdown path: `shutdownReclaim` → same chain (P-4). **No unowned-world leak path in production code.**

---

## 5E — P_max ≠ K_world ≠ M (separation, with evidence)

```text
P_max     = Publish Intent transport residency bound  (intent slots + producer accounting gap)
K_world   = World lifecycle residency (S1..S6)        (Owned worlds across OwnerChannel + RuntimeStore + D/Q/E/T)
B_max^true = burst reservation obligation             (burst publish rate × stall window)
M         = Observation-error + growth envelope       (sup Δ_k^growth + E^obs, D101)
```

| Symbol | Domain | Bounded by | Production evidence |
|---|---|---|---|
| **P_max** (≤4098 conditional) | Intent transport | kIntentQueueCapacity(4096) + R_max(2) | ISRRuntimePublicationCoordinator.h:542,692; Step 4 |
| **K_world** | World lifetime | S2:OwnerChannel(256) + S3:current(1) + S4:D(4096) + S5:Q+E(1024) + S6:Terminal(growable) | OwnerChannel.h:41; DeferredDeletionQueue.kQueueSize; RetireQuarantineStore.kMaxQuarantinedEntries(512)×2 |
| **B_max^true** | Burst obligation | ≤ max publishes in T window (D63: structural) | I4_DESIGN_CONTRACT.md:4272-4273 |
| **M** | Observation error | sup_k Δ_k^growth + E^obs | I4_DESIGN_CONTRACT.md:6960-6975 |

### Why P_max ≠ M (explicit)

```text
P_max      = intent transport capacity   (structural, code-proven for P_queue_max)
M          = observation error bound      (D101 = OPEN, NOT proven finite)
```

**P_max is NOT an observation error term.** M's components (G = sampling gap, λ = retire rate, τ_b = burst duration, μ_burst, jitter_bound) are about the **sampler's ability to observe retirement bursts** — a measurement-domain property. P_max is a **resource-transport capacity** in the publish pipeline. They are orthogonal domains.

> **`M = max(E_w)` の使用を安全側上界としないこと** — Step 5 preserves D94/D95: E_w (= T_w − O_w, measured =1 in burst tests) is a *measured gap*, not a *structural bound*. M must be proven via the D101.3 10-step review (envelopes, not peaks).

---

## 5F — M candidate census (event / occupancy / observation-error classification)

From I4_DESIGN_CONTRACT.md D94/D101 (lines 6402-6669, 6948-7044):

| M candidate | Meaning | Type (event/occupancy/observation) | Production/design status | Bounded? |
|---|---|---|---|---|
| `P_max` (≤4098) | Publish Intent transport residency | **occupancy** (intent queue) | Step 4: P_queue_max=4096 PROVEN; P_max≤4098 CONDITIONAL | ❌ NOT M (occupancy ≠ observation error) |
| `R_max` (≤2) | Producer accounting gap (fetch_add→push) | **occupancy** (transient counter) | 5B: CONDITIONAL (code-audit, not structural) | ❌ NOT M |
| `G` (maxSamplingGapUs) | Sampler max observation interval | **observation** | D94/D101 sampler spec (design: 100ms cadence); burst test harness measures | ⚠️ design constant, sampler not watchdog-guaranteed |
| `λ_retire` | World retirement/release rate | **event rate** | D63: ≤ publish rate (INV-PUB-SER serialization); telemetry-measurable | ⚠️ measured, structural envelope OPEN |
| `τ_b` (burst duration) | Retirement burst duration | **event/temporal** | T335 burst test harness implemented; measured | ⚠️ measured, structural bound OPEN |
| `μ_burst` | Burst retirement rate | **event rate** | D101.2: `μ_burst · G` = Δ_k^growth envelope candidate | ⚠️ OPEN |
| jitter_bound | Scheduler jitter / sampling delay | **observation** | T336/T337 3-condition measurement (normal/burst/jitter) | ⚠️ measured, structural proof OPEN |
| `M` | `≥ sup_k Δ_k^growth + E^obs` | **observation + growth envelope** | D101 = OPEN → **NOT proven finite** → Phase I NO-GO | ❌ UNPROVEN |
| `O_w` | Sampled windowMax (B_max^observed, 100ms sampler) | **observation** | T1 telemetry, D94 CLOSED (measurement model) | ✅ measured (not a bound) |
| `T_w` | Reference maximum (true value bound) | **reference** | D94 CLOSED (reference observer) | — (reference, not production) |
| `E_w = T_w − O_w` | Sampling under-observation | **observation error** | D94/D100 measured (E_w=1 in burst) | ✅ measured (not a safe bound — max(E_w) banned) |

### D101 M-bound proof status (D101.3 review order, I4_DESIGN_CONTRACT.md:7044+)

| Step | Name | Content | Status |
|---|---|---|---|
| #1 | Reference completeness (4-tier) | A_ref/R_ref observer balance | ⚠️ Tier 1-3 ✅; Tier 4 (published-domain exclusion) — **RESOLVED in Step 5-H** |
| #2 | State equation | B = A - R | ✅ (referenceOutstanding = acquire − release, ISRWorldRetirementReference.h:119-120) |
| #3 | Sampler gap | G = maxSamplingGapUs | ⚠️ design 100ms; watchdog NOT proven |
| #4 | Acquire/increase envelope | event rate / τ_b / μ_burst | OPEN |
| #5 | Single burst bound | — | OPEN |
| #6 | Multiple acquire in interval | — | OPEN (generalize E_w=1) |
| #7 | Delayed release | reclaim latency amplifies peak | OPEN |
| #8 | Shutdown/quarantine/deferred | non-steady paths | OPEN |
| #9 | Finite M proof | sup Δ_k^growth + E^obs < ∞ | ❌ UNPROVEN (D101 = OPEN) |
| #10 | D102 gate | B_max^true ≤ O_w + M | ❌ blocked on #9 |

### O_w / T_w / E_w 3-series model (D94, I4_DESIGN_CONTRACT.md:6402-6455)

```text
B_true(t)  = A_true(t) - R_true(t)   — mathematical truth (unobservable)
B_ref(t)   = A_ref(t) - R_ref(t)     — reference observer (T_w source)
B_obs(t_k)                         — sampler tick observation (O_w source)

O_w = max_k B_obs(t_k)             — sampled windowMax (T1 telemetry, 100ms sampler)
T_w = sup_t B_ref(t)               — reference maximum (NOT true value — D100.5)
E_w = T_w - O_w                    — sampling under-observation (observed =1 in burst)

M ≥ sup_k Δ_k^growth + E^obs      — STRICT: M is envelope, NOT max(E_w)  (D101.2)
```

> ⚠️ **max(E_w) is NOT a valid M** (D94/D95). E_w=1 is a *measured* gap, not a *structural* bound.

---

## 5G — M structural-bound candidates (mapping to production)

The D101.3 review #4, #6, #7, #8 map M's growth envelope to production structures:

| Envelope term | Production structure | Evidence |
|---|---|---|
| **Δ_k^growth** (interval growth) | `publicationIntentResidencyCount_` window max + OwnerChannel occupancy (256) + D/Q backlog (5120) — but P is transport, worlds bounded by OwnerChannel(256) | 5D proof: P-worlds already in OwnerChannel |
| **μ_burst · G** (burst × sampling gap) | `kIntentQueueCapacity(4096)` × sampler cadence(100ms) — but **μ_burst is publish rate, not P** | D101.2: μ_burst · G is Δ_k^growth envelope candidate |
| **reclaim latency amplifies** (#7) | epoch-gated FIFO reclaim (DeferredDeletionQueue::reclaim isOlder(entry.epoch, minReaderEpoch)) | DeferredDeletionQueue.h:133-145; EpochDomain.h:211 getMinReaderEpoch |
| **retirement rate λ** (#4) | ≤ publish rate (INV-PUB-SER waitForReceipt serializes) | I4 D63.2: λ_retire ≤ publish rate |
| **burst duration τ_b** | D→Q→E→T chain backpressure; `enqueueWithRetry` retry+tryReclaim loop | ISRRetireRouter.cpp:282-345 |
| **jitter_bound** (#3/#7) | `maxSamplingGapUs`, `missedTick` (D94 telemetry) | T336/T337 |

**Key structural insight (5G):** The growth envelope Δ_k^growth is **NOT** bounded by P_max. P_max bounds *intent transport* (4098 slots). The *world growth* is bounded by the publish path: admission → build → OwnerChannel.store → publishAndSwap. Since OwnerChannel is SPSC single-transfer with take-once semantics, and publish is serialized (waitForReceipt, INV-PUB-SER), the **world growth rate ≤ publish rate**, and the **world burst occupancy ≤ OwnerChannel(256) + D(4096) + Q+E(1024)** = all already in K_world. P_max does not add a separate growth term.

---

## 5H — CODE-PROVEN / DESIGN-DEFINED / MISSING classification

### P-domain proof status (for B_max contribution)

| Proposition | Verdict | Evidence |
|---|---|---|
| `Intent.payload` is non-owning | ✅ CODE-PROVEN | PublishPayload: `const void* newWorld` (read-only); trivially_copyable static_assert (h:336-339) |
| World ownership transfers to OwnerChannel BEFORE intent enqueue | ✅ CODE-PROVEN | AudioEngine.h:4559 (enqueue) → :4577 (intent) ordering |
| Intent → World 1:1 identity (seqId) | ✅ CODE-PROVEN | PublishPayload binds to seqId; OwnerChannelKey = {seqId, epoch, mappedGen}; take drains once |
| P_queue ≤ OwnerChannel occupancy ≤ 256 | ✅ CODE-PROVEN | OwnerChannel.h:41 kCapacity=256; I-5D.1 ownership-before-intent |
| Build failure → 0 worlds | ✅ CODE-PROVEN | trySubmitImpl `:167` `!worldOwner` → no enqueue |
| Admission reject → 0 worlds | ✅ CODE-PROVEN | trySubmitImpl `:46` admission gate before build |
| Enqueue push-fail → 0 retained worlds | ✅ CODE-PROVEN | AudioEngine.h:4580 take+reclaim |
| Crossfade rebuild → ≤1 world | ✅ CODE-PROVEN | :165/:232 reassign destroys prior via RAII |
| **P adds 0 additive worlds to B_max** | ✅ CODE-PROVEN | 5D Theorem: P_queue ⊆ OwnerChannel(S2) bound |

### R_domain (reference observer) proof status — **D101 Tier-4 RESOLVED**

| Tier | Requirement | Status | Evidence |
|---|---|---|---|
| Tier 1 (Acquire completeness) | `onAcquire` called on **all** publish-success terminal paths | ✅ CLOSED | AiDex: onAcquire — **1 production call site** (AudioEngine.Commit.cpp:408); definition ISRWorldRetirementReference.h:29 |
| Tier 2 (Release completeness) | `onRelease` called on **all** world-terminal-deletion paths | ✅ CLOSED | DeletionEntryType::World gated in DeferredDeletionQueue.h:148,199 + RetireQuarantineStore.h:140,177 + ISRRetireRouter.cpp:68,89,504 |
| Tier 3 (Accounting conservation) | B_ref = A_ref − R_ref (no reset) | ✅ CLOSED | referenceOutstanding() = acquire − release; referenceMax window max (ISRWorldRetirementReference.h:119-120) |
| **Tier 4 (published-domain exclusion)** | **No hidden World lifetime path outside observer** | ✅ **RESOLVED (Step 5)** | `retirePublishedRuntimeWorldNonRt` → `DeletionEntryType::World` (✅ onRelease) vs `retireRejectedRuntimeWorldNonRt` → `DeletionEntryType::Generic` (❌ no onRelease). All 4 production call sites use the **correct** path. See 5H-1. |

#### 5H-1 — Published vs Rejected world retire-path separation (Tier 4 closure)

The D101 audit (I4_DESIGN_CONTRACT.md:6843) flagged **"Init.cpp:67 — rejectedWorld → retireRuntimePublishWorldNonRt → World → onRelease"** as a `counterexample: W ∉ PublishedDomain`.

**Current code has been refactored to a two-method separation** (AudioEngine.h:3536-3562):

```cpp
void retirePublishedRuntimeWorldNonRt(RuntimePublishWorld* world, bool resetRevision)
{
    // PRECONDITION: W ∈ PublishedDomain (must have passed publishAndSwap LP)
    if (world == nullptr) return;
    engine_->enqueueDeferredDeleteNonRt(world, [](void* p){ ... },
        DeletionEntryType::World);    // ← triggers referenceObserver_->onRelease()
    (void)resetRevision;
}

void retireRejectedRuntimeWorldNonRt(RuntimePublishWorld* world)
{
    // PRECONDITION: W ∉ PublishedDomain (rejected/unpublished)
    if (world == nullptr) return;
    engine_->enqueueDeferredDeleteNonRt(world, [](void* p){ ... },
        DeletionEntryType::Generic);  // ← does NOT trigger onRelease()
}
```

| Call site | Function | Type tag | PublishDomain? | onRelease? |
|---|---|---|---|---|
| AudioEngine.Init.cpp:88 | `retirePublishedRuntimeWorldNonRt(oldWorld, false)` | World | ✅ (oldWorld = prev current, published) | ✅ |
| AudioEngine.Init.cpp:67 | `retireRejectedRuntimeWorldNonRt(rejectedWorld)` | Generic | ❌ (bootstrap reject) | ❌ ✅ |
| RuntimePublishExecutor.h:76 | `retirePublishedRuntimeWorldNonRt(oldWorld, false)` | World | ✅ | ✅ |
| AudioEngine.CtorDtor.cpp:234 | `retirePublishedRuntimeWorldNonRt(clearedWorld, true)` | World | ✅ (was published) | ✅ |
| AudioEngine.Processing.ReleaseResources.cpp:460 | `retirePublishedRuntimeWorldNonRt(clearedWorld, true)` | World | ✅ | ✅ |
| AudioEngine.Processing.ReleaseResources.cpp:547+ | (comment: retire path reference) | — | — | — |

The release observer path (DeferredDeletionQueue::reclaim / drainAllUnsafe / RetireQuarantineStore::drain) gates `onRelease()` on `entryType == DeletionEntryType::World`:
```text
// DeferredDeletionQueue.h:148, 199; RetireQuarantineStore.h:140,177
if (entryType == DeletionEntryType::World) {
    ++reclaimCount_;
    if (referenceObserver_ != nullptr)
        referenceObserver_->onRelease();    ← ONLY for published-domain worlds
}
```

**Conclusion**: The Tier-4 "unpublished World triggering onRelease" counterexample is **resolved by code separation** in the current production source. Rejected/unpublished worlds use `DeletionEntryType::Generic` and never fire `onRelease()`. The reference observer B_ref = A − R is **balanced for published-domain worlds only**.

> Note: D101 Tier-4 was listed OPEN in I4_DESIGN_CONTRACT.md:6879 (2026-08-16 audit). The two-method refactor (`retireRejectedRuntimeWorldNonRt` with `Generic` type) is the structural fix that resolves it. The audit note text ("Init.cpp:67 — rejectedWorld → retireRuntimePublishWorldNonRt → World → onRelease") describes the **pre-refactor** single-function form `retireRuntimePublishWorldNonRt`; current code has the **split**.

#### 5H-2 — `onRuntimeRetiredNonRt` does NOT affect observer (verified)

`onRuntimeRetiredNonRt` (AudioEngine.Commit.cpp:446) — called via `bridge.willRetireRuntimePublishWorldNonRt` (AudioEngine.h:3533):
- Calls `worldLifecycleAudit_.onWorldRetired(...)`, `runtimeOrchestrator_->notifyWorldRetired(...)`, `debugRuntime_.recordHBEdge(...)`, `emitRetireIntentRT(intent)`, shutdown-marking.
- **Does NOT call onAcquire/onRelease** (no `worldRetirementReference_` access).
- This is purely **observability/awareness** — retirement *intent* notification, not world *deletion*.
- AiDex confirmed: `onAcquire` has **only** 2 production-relevant hits (Commit.cpp:408 call + h:29 definition); `onRelease` sites are only in the terminal-deleter gated path.

### M-bound proof status (D101 = OPEN)

| D101 # | Requirement | Verdict |
|---|---|---|
| #1-#3 (reference completeness, state eq, sampler gap) | Reference observer + O_w/T_w/E_w model | ⚠️ Tier 4 RESOLVED; #3 G watchdog NOT structurally enforced |
| #4-#8 (envelopes) | Δ_k^growth, μ_burst·G, delayed release, burst, shutdown | ❌ OPEN (burst test harness measures; no structural proof) |
| #9 (finite M proof) | sup Δ_k^growth + E^obs < ∞ | ❌ UNPROVEN (D101 = OPEN) |
| **M** | `≥ sup_k Δ_k^growth + E^obs` | **UNPROVEN finite — Phase I NO-GO** |

**M = max(E_w) = 1** (measured) is **explicitly forbidden** as M (D94/D95). M must come from #1-#9.

---

## 5I — D101 M-bound proof obligations (from I4_DESIGN_CONTRACT.md D101.3, 6948-7044)

```text
B_max^true ≤ O_w + M(G, λ, τ_b, …)       ← structural upper bound (D101 purpose)
M ≥ sup_k Δ_k^growth + E^obs             ← strict: growth + observation error separation
R_required > R_baseline + B_max(T_stall) + M      ← D63.4 (D102 gate)
```

**Open proof obligations (D101 = OPEN → Phase I NO-GO):**

```text
[ ] D101 #3: G (maxSamplingGapUs, jitter_bound) = bounded structural upper bound (not just 100ms design)
[ ] D101 #4: μ_burst, τ_b  — acquire/increase envelope (structural, not measured)
[ ] D101 #5: single burst bound (structural)
[ ] D101 #6: multiple acquire in one interval — generalize E_w=1 (structural)
[ ] D101 #7: delayed release — reclaim latency amplification (structural)
[ ] D101 #8: shutdown/quarantine/deferred deletion — inclusion in M envelope (structural)
[ ] D101 #9: finite M = sup_k Δ_k^growth + E^obs < ∞ (from #1-#8)
[→] D102 gate: B_max^true ≤ O_w + M — blocked on D101 #9
```

**P_max relationship to these obligations:** P_max contributes to D101 #4 (acquire/increase envelope) via **μ_burst · G** (Δ_k^growth candidate). But per 5D, P_max's world-bearing contribution is **contained** within OwnerChannel(256) — already in K_world. P_max bounds *intent slots*, λ (retire rate) bounds *retirement events*. They are separate terms in the M envelope, not interchangeable.

---

## 5J — Verdict

```text
5A. P_max final status
     → P_queue_max = 4096 PROVEN; P_max ≤ 4098 CONDITIONAL (Case B)
5B. R_max structural proof
     → R-PROD-1..4 PASS (code-audit); R_max ≤ 2 CONDITIONAL (NOT structurally enforced)
5C. Publish Intent → World cardinality
     → 1:1 ownership-before-intent (OwnerChannel holds world, intent is non-owning ptr)
     → Build-failure/admission-reject = 0 worlds; enqueue-fail = reclaim+destroy
     → NO multi-world-per-intent; NO world-multiplexing
5D. P-state → B_max contribution
     → contribution_to_B_max(P-state) = 0 (P_queue ≤ OwnerChannel(256) ⊆ K_transferred)
     → P_max is NOT additive to B_max/K_world/M
5E. P_max ≠ K_world ≠ M
     → P_max (intent transport) ≠ K_world (world lifetime) ≠ M (observation-error envelope)
5F. M candidate census
     → G, λ, τ_b, μ_burst, jitter_bound 分類 (design/measured/OPEN)
     → P_max, R_max は M candidate には該当しない (occupancy domain)
5G. M structural-bound candidates
     → P_max feeds μ_burst·G envelope candidate; worlds bounded by OwnerChannel(256) not P_max
5H. CODE-PROVEN / DESIGN-DEFINED / MISSING
     → P-domain world-bearing: ALL CODE-PROVEN (5D theorem holds)
     → D101 Tier-4 published-domain exclusion: RESOLVED (two-method retire separation)
     → D101 #4-#9 M envelope: OPEN / MISSING (structural proof absent)
5I. D101 M-bound proof obligations
     → 8 open items (#3-#9); Phase I NO-GO maintained
```

### Final verdict

| Question | Verdict |
|---|---|
| P_queue_max = 4096 from production code | ✅ PROVEN |
| R_max structural upper bound | ⚠️ CONDITIONAL (Case B — code-audit, not type-enforced) |
| P_max ≤ 4098 formal status | ⚠️ CONDITIONAL UPPER BOUND (not architectural invariant) |
| Publish Intent ∧ World cardinality | ✅ 1:1 (ownership-before-intent; non-owning payload) |
| P_max as B_world/K_world surrogate | ❌ PROVEN NOT (P ⊆ OwnerChannel, additive contribution = 0) |
| P_max as M direct value | ❌ PROVEN NOT (P is occupancy; M is observation-error envelope) |
| P → M contribution function | ✅ f(P-state) = 0 additive (worlds carried by P are within OwnerChannel(256) ⊂ K_world(S2)) |
| M = max(E_w) safety bound | ❌ PROVEN INCORRECT (D94/D95 — measured gap, not structural bound) |
| D101 M bound closure | ❌ OPEN (Phase I NO-GO) — 8 open proof obligations |

---

## 5K — Step 6 handoff

### What Step 5 established for Step 6 (K_world / H_max handoff)

Step 5 does NOT re-prove K_world — that was Step 6's job (already done in `phase-d101-8-step6-k-world-derivation.md`). Step 5's contribution to the chain:

| Step 5 output | Handoff to Step 6 / beyond |
|---|---|
| **P contributes 0 additive worlds** to B_max/K_world | K_world = Σ(S1..S6 capacity); P is transport-layer (OwnerChannel ⊆ S2), NOT a separate world term |
| **K_transferred(S2) corrected to 256** (not 65 as summaries claimed) | Step 6's K_transferred ≤ 65 must be rechecked against OwnerChannel.h:41 kCapacity=256 |
| **D101 Tier-4 published-domain exclusion RESOLVED** | Reference observer (O_w/T_w/E_w) balance holds → D101 #1-#3 foundation stable |
| **P_max = 4098 NOT a constant** | Do NOT inject `4098` as a design constant; use symbolic `P_max ≤ 4096 + R_max` |
| **M = max(E_w) forbidden** | M derived from D101 #4-#9 envelopes, NOT from measured E_w peaks |

### Step 5 → Step 6 (existing) closure

Step 5's key correction to Step 6: **K_transferred ≤ 256** (OwnerChannel owning slots), not 65. PendingPublishRegistry (64) is non-owning metadata. Re-running Step 6's K_world bound:

```text
K_world ≤ K_reserved(A_max) + K_transferred(256) + K_current(1) + K_retire(4096) + K_quarantine(1024) + K_terminal(∞)
```

K_world remains CONDITIONAL (K_reserved=A_max, K_terminal=growable — both assumptions).

### Remaining chain

```text
B_max^true ≤ O_w + M         ← D101 OPEN (M unproven)
R_required > R_baseline + B_max(T_stall) + M   ← D63.4, blocked on D101
R_required / R_cap            ← D102 (Phase I NO-GO)
```

**Step 5 is audit-only — no code changes.** The structural relation `P ⊆ OwnerChannel(S2)` is production-code-proven; the open items (#3-#9) remain design/measurement-phase work (Phase I NO-GO).

---

## References

| Evidence | File | Lines |
|---|---|---|
| Intent struct (non-owning, trivially_copyable) | `src/audioengine/ISRRuntimePublicationCoordinator.h` | 318-339 |
| PublishPayload (`const void* newWorld` HANDLER-1 read-only) | `src/audioengine/ISRRuntimePublicationCoordinator.h` | 306-314 |
| enqueuePublicationIntent (sole reserve site h:365-375) | `src/audioengine/ISRRuntimePublicationCoordinator.h` | 342-375 |
| processIntent (consumer pop + counter decay) | `src/audioengine/ISRRuntimePublicationCoordinator_ProcessIntent.cpp` | 10-76 |
| counter doc (INV-X5-1) | `src/audioengine/ISRRuntimePublicationCoordinator.h` | 539-547 |
| enqueueRuntimePublicationFireAndForget (ownerChannel before intent) | `src/audioengine/AudioEngine.h` | 4509-4590 |
| commitRuntimePublication (sync wrapper) | `src/audioengine/AudioEngine.h` | ~4605-4630 |
| OwnerChannel kCapacity=256 (SPSC single-transfer) | `src/audioengine/OwnerChannel.h` | 1-145 |
| trySubmitImpl (admission→build→publish) | `src/audioengine/RuntimePublicationOrchestrator.cpp` | 40-310 |
| buildRuntimePublishWorld (build sites :165, :232) | `src/audioengine/RuntimePublicationOrchestrator.cpp` | 161-232 |
| BuildResult / BuildError (failure classification) | `src/audioengine/RuntimeBuilder.cpp` | 410-460 |
| PublishExecutor::executePublish (consumer commit tail) | `src/audioengine/RuntimePublishExecutor.h` | 20-108 |
| PublicationExecutor::publishImpl (deferred resubmit) | `src/audioengine/PublicationExecutor.cpp` | 1-98 |
| retirePublished/Rejected (Domain separation) | `src/audioengine/AudioEngine.h` | 3536-3562 |
| enqueueDeferredDeleteNonRt (D→Q→E→T chain) | `src/audioengine/AudioEngine.h` | 4201-4240 |
| enqueueWithRetry (no-leak invariant) | `src/audioengine/ISRRetireRouter.cpp` | 25 (invariant), 282-345 |
| onAcquire (sole production site) | `src/audioengine/AudioEngine.Commit.cpp` | 408 |
| onRelease (World-gated) | `src/DeferredDeletionQueue.h`, `src/audioengine/RetireQuarantineStore.h`, `src/audioengine/ISRRetireRouter.cpp` | 148,199 / 140,177 / 68,89,504 |
| onRuntimeRetiredNonRt (NOT observer) | `src/audioengine/AudioEngine.Commit.cpp` | 446 |
| Reference observer (B_ref = A − R) | `src/audioengine/ISRWorldRetirementReference.h` | 29-120 |
| M definition strict (D101.2) | `doc/work88/I4_DESIGN_CONTRACT.md` | 6960-6984 |
| D101 review order (D101.3) | `doc/work88/I4_DESIGN_CONTRACT.md` | 7044+ |
| B_max^true ≤ O_w + M (D101 purpose) | `doc/work88/I4_DESIGN_CONTRACT.md` | 6948 |
| R_required = R_baseline + B_max(T_stall) + M | `doc/work88/I4_DESIGN_CONTRACT.md` | 4280-4316 (D63) |
| AiDex cross-verification | AiDex MCP query `onAcquire` | 2 matches (1 production call + 1 def) |
| AiDex cross-verification | AiDex MCP query `enqueuePublicationIntent` | 1 production caller (AudioEngine.h:4577) |
| Previous Step 5 (H_max/G_contract) | `evidence/phase-d101-8-step5-hmax-gcontract-derivation.md` | — (separate scope) |

---

## Task Completion Checklist (Step 5 closure conditions)

```text
[x] P_queue_max = 4096 が production code から証明済み  (MpscBoundedRing h:692-693)
[x] R_max の structural upper bound が証明試行  (5B: CONDITIONAL — Case B, code-audit not type-enforced)
[x] Publish Intent と World の cardinality relationship が確定  (5C: 1:1 ownership-before-intent)
[x] P_max が B_world/K_world の代理値ではないことを証明  (5D: P ⊆ OwnerChannel ⊆ K_transferred)
[x] P_max が M の直接値ではないことを証明  (5E: P=occupancy, M=observation-error)
[x] P_max が M に寄与する場合の関数関係を明示  (5G: P feeds μ_burst·G candidate; additive contribution = 0)
[x] M candidate を event / occupancy / observation error に分離  (5F table)
[x] G, λ, τ_b, μ_burst, jitter_bound の production/design status を分類  (5F)
[x] M = max(E_w) を安全偺上界として使用しないことを維持  (5F: D94/D95 — measured, not structural)
```

**Open (not closed by Step 5 — Phase I NO-GO maintained):**
```text
[ ] D101 #3-#9 M envelope structural proof (M finiteness unproven)
[ ] R_max structural enforcement (type-system / context-tagged API)
```
