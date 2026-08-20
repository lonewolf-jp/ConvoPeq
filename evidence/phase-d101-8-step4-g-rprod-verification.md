# D101-8 Step 4-G — R-PROD Structural Verification

> **Purpose**: Re-verify R-PROD-1 through R-PROD-4 against current production source
> **Basis**: `ConvoPeq.md` (2026-08-20 15:06:54 generation) + `graphify-out/` design docs
> **Goal**: Determine whether `P_max ≤ 4098` can be elevated from "code-derived fact" to "conditional bound"

---

## G1 — `enqueuePublicationIntent()` Production Caller Closure

### Methodology (G1)

Searched for all call sites of the complete production call chain:

- `enqueuePublicationIntent(`
- `enqueueRuntimePublicationFireAndForget(`
- `commitRuntimePublication(`
- `executor_.publish(`
- `submitPublishRequest(`

Filters applied:

- Excluded comments (`//`, `/* */`)
- Excluded test files (`tests/`, `*Test*`, `Mock*`)
- Excluded design docs (`ConvoPeq.md`, `*.h` comments)

Also checked for indirect invocation patterns:

- Lambda / callback
- `std::function`
- `std::thread` pools / workers
- `juce::MessageManager::callAsync`
- `juce::AsyncUpdater`
- `juce::Timer` (indirect Timer callback dispatch)
- Virtual/interface dispatch

### Results

#### Production caller sites (9 total)

| # | Call site | Thread | Path to enqueuePublicationIntent |
| --- | --- | --- | --- |
| 1 | `AudioEngine.Processing.PrepareToPlay.cpp:155` | Message Thread | `commitRuntimePublication` → `enqueueRuntimePublicationFireAndForget` → `enqueuePublicationIntent` |
| 2 | `AudioEngine.Processing.PrepareToPlay.cpp:277` | Message Thread | Same as #1 |
| 3 | `AudioEngine.Processing.ReleaseResources.cpp:175` | Message Thread (JUCE contract) | `commitRuntimePublication` → `enqueueRuntimePublicationFireAndForget` → `enqueuePublicationIntent` |
| 4 | `AudioEngine.Timer.cpp:964` | Message Thread (JUCE Timer) | `commitRuntimePublication` → `enqueueRuntimePublicationFireAndForget` → `enqueuePublicationIntent` |
| 5 | `AudioEngine.Transition.cpp:25` (via `publishIdleWorldOnly`) | Message Thread | `commitRuntimePublication` → `enqueueRuntimePublicationFireAndForget` → `enqueuePublicationIntent` |
| 6 | `AudioEngine.CtorDtor.cpp:79` | Message Thread (RestoreStep2 callback via healthMonitor.tick → timerCallback) | Same as #5 |
| 7 | `PublicationExecutor.cpp:53` (deferred resubmit) | **RebuildThread** (NOT CoordinatorLoop*) | `enqueueRuntimePublicationFireAndForget` → `enqueuePublicationIntent` |
| 8 | `AudioEngine.RebuildDispatch.cpp:987` | RebuildThread (inside `rebuildThreadLoop`) | `enqueuePublicationIntentForRuntimeCommit` → `submitPublishRequest` → `trySubmitImpl` → `executor_.publish()` → `enqueueRuntimePublicationFireAndForget` → `enqueuePublicationIntent` |
| 9 | `AudioEngine.RebuildDispatch.cpp:1060` | RebuildThread (inside `rebuildThreadLoop`) | Same as #8 |
| 10 | `AudioEngine.RebuildDispatch.cpp:1239` | RebuildThread (inside `rebuildThreadLoop`) | Same as #8 |
| 11 | `RuntimePublicationOrchestrator.cpp:542` | RebuildThread (`processDeferredAdmission`) | `submitPublishRequest` → `trySubmitImpl` → same path |

> *The comment at `RuntimePublicationOrchestrator.cpp:250` and `PublicationExecutor.cpp:51` says "CoordinatorLoop 上の deferred resubmit" but this is misleading. The comment refers to the **conceptual origin** of the deferred request (set by CoordinatorLoop via `publishRetryReady` flag), NOT the execution context. `processDeferredAdmission` runs on RebuildThread (confirmed by `jassert(std::this_thread::get_id() == engine_.rebuildThreadId())` at RuntimePublicationOrchestrator.cpp:527).

#### Indirect invocation patterns checked

| Pattern | Found? | Notes |
| --- | --- | --- |
| Lambda / `std::function` caller | ❌ No | No lambda wraps `enqueuePublicationIntent` directly |
| `std::thread` pool / worker | ❌ No | `rebuildThread` is a single `std::thread` instance |
| `juce::MessageManager::callAsync` | ⚠️ Seen, but no publish path | `callAsync` in Learning.cpp:423 and AudioEngineProcessor.cpp:136 only call `onCoeffBankChanged` / `requestLoadState` — neither directly calls `enqueuePublicationIntent`; `requestLoadState` triggers rebuild (RebuildThread publish) |
| `juce::AsyncUpdater` | ⚠️ Seen, but no publish path | `handleAsyncUpdate` (RebuildDispatch.cpp:376) triggers rebuild, not direct publish |
| Virtual/interface dispatch | ❌ No | `enqueuePublicationIntent` is `inline` non-virtual method |

### Caller closure verdict: ✅ R-PROD-1 PASS

```text
Production caller closure = { Message Thread, RebuildThread }
CoordinatorLoop = consumer + signaler only (never calls enqueuePublicationIntent)
```

All 9 production call sites trace through one of these 2 contexts. No indirect invocation patterns found that bypass this closure.

---

## G2 — Message Thread Execution Cardinality (C_msg = 1)

### Methodology (G2)

Verified that all Message Thread call sites execute on the single JUCE MessageManager thread, with no concurrent Message Thread execution.

### Evidence (G2)

**JUCE contract**: `prepareToPlay`, `releaseResources`, `timerCallback`, AsyncUpdater callbacks, and `handleAsyncUpdate` all execute on the JUCE MessageManager thread (single thread).

**Source evidence**:

- `AudioEngine.Processing.PrepareToPlay.cpp:219`: `// JUCE 契約上 prepareToPlay 実行中は Audio Thread callback が走らないため`
- `AudioEngine.Processing.PrepareToPlay.cpp:290`: `juce::MessageManager::getInstance()->isThisTheMessageThread()` check
- `AudioEngine.Timer.cpp:371`: `timerCallback()` — JUCE Timer callback runs on Message Thread
- `AudioEngine.CtorDtor.cpp:79`: RestoreStep2 callback → `m_healthMonitor.tick()` → called from `timerCallback()` (Message Thread)
- `AudioEngine.Transition.cpp:25`: `publishIdleWorldOnly()` — called from `requestTransition()` paths, all Message Thread

**Concurrency check**: JUCE `MessageManager::callAsync` queues lambdas on the same Message Thread. Even if multiple `callAsync` lambdas are queued, they execute sequentially on the single Message Thread. No concurrent Message Thread execution is possible.

### Cardinality verdict: ✅ C_msg = 1 PROVEN

```text
Message Thread = 1 single-threaded execution context
JUCE MessageManager contract + single MessageManager thread
```

---

## G3 — RebuildThread Execution Cardinality (C_rebuild = 1)

### Methodology (G3)

Verified that `rebuildThread` is a single `std::thread` instance and that all RebuildThread call sites execute on this single thread.

### Evidence (G3)

**Thread creation**:

- `AudioEngine.Init.cpp:33`: `rebuildThread = std::thread(&AudioEngine::rebuildThreadLoop, this);`
- `AudioEngine.Processing.PrepareToPlay.cpp:85`: `rebuildThread = std::thread(&AudioEngine::rebuildThreadLoop, this);` (only if not joinable)

**Thread lifecycle**:

- `AudioEngine.RebuildDispatch.cpp:771`: `stopRebuildThread()` — `rebuildThread.join()` ensures single instance
- Single `std::thread` member variable, no thread pool

**Thread context enforcement**:

- `RuntimePublicationOrchestrator.cpp:455`: `jassert(std::this_thread::get_id() == engine_.rebuildThreadId());`
- `RuntimePublicationOrchestrator.cpp:480`: Same jassert
- `RuntimePublicationOrchestrator.cpp:527`: Same jassert (on `processDeferredAdmission`)

**Call site containment**:

- All 3 `enqueuePublicationIntentForRuntimeCommit` calls (RebuildDispatch.cpp:987, 1060, 1239) are inside `rebuildThreadLoop()` (line 804)
- `processDeferredAdmission` (line 525) has `jassert(thread == rebuildThreadId)`
- `submitPublishRequest` is only called from `enqueuePublicationIntentForRuntimeCommit` (Commit.cpp:813) and `processDeferredAdmission` (Orchestrator.cpp:542) — both RebuildThread

### Cardinality verdict: ✅ C_rebuild = 1 PROVEN

```text
RebuildThread = 1 single-threaded execution context
Single std::thread instance + jassert thread ownership enforcement
```

---

## G4 — Reentrancy Audit Through `MpscBoundedRing::push()` Internals

### Methodology (G4)

Audited the complete call chain from `enqueuePublicationIntent()` through `MpscBoundedRing::push()`, checking for any callback, dispatch, or reentry path in the entire `fetch_add → push → rollback` sequence.

### Call chain

```text
enqueuePublicationIntent()  [ISRRuntimePublicationCoordinator.h:351]
  → fetch_add(publicationIntentResidencyCount_)     [external accounting reservation]
  → intentQueue_.push()                              [MpscBoundedRing::push]
      → consumeAtomic(enqueuePos_)                   [atomic load — no callback]
      → consumeAtomic(seq_atom)                      [atomic load — no callback]
      → compareExchangeAtomic(enqueuePos_, ...)      [atomic CAS — no callback]
      → entries_[pos & kMask] = item                 [trivial copy — T is trivially_copyable]
      → test hooks (CONVO_TESTING only: yield())     [testing only]
      → publishAtomic(seq_atom, ...)                 [atomic store — no callback]
  → [push succeeded: return true]
  → [push failed: fetchSub rollback + return false]
```

### Reentrancy check items

| Check | Result | Evidence |
| --- | --- | --- |
| Callback in `push()` body | ❌ NO | Only atomic ops + trivial copy |
| Virtual dispatch in `push()` | ❌ NO | `MpscBoundedRing` is a template, all methods non-virtual |
| Lock acquisition in `push()` | ❌ NO | Lock-free (CAS only) |
| CV wait in `push()` | ❌ NO | No condition_variable |
| Exception handler in `push()` | ❌ NO | `noexcept`, no try/catch |
| Nested `enqueuePublicationIntent` in `push()` | ❌ NO | `push()` is pure data structure |
| Callback in upstream `enqueueRuntimePublicationFireAndForget` | ❌ NO | `registerDSPHandleForRuntime` (lock released before push), `makePublishDecisionSnapshot` (pure data), `ownerChannel().enqueue` (atomic CAS), `registerPublish` (atomic fetch_add) — all complete before `enqueuePublicationIntent` |
| `waitForPublishReceipt` CV wait | ❌ NO | Occurs AFTER `enqueueRuntimePublicationFireAndForget` returns (post-push) |

### Reentrancy verdict: ✅ R-PROD-3 PASS

```text
R-PROD-3: NO REENTRY in the fetch_add → push/rollback window
  Includes MpscBoundedRing::push() internals (atomic CAS + trivial copy only)
```

---

## G5 — Single Invocation Per Context (R-PROD-4)

### Logic

```text
R-PROD-2: Each context is single-threaded → sequential execution within context
R-PROD-3: No reentrancy in reservation window
⇒ R-PROD-4: count(active enqueuePublicationIntent invocations in context C) <= 1
```

### Evidence (G5)

- Message Thread: JUCE MessageManager is single-threaded. Even `callAsync` lambdas execute sequentially. No recursive/nested call to `enqueuePublicationIntent` exists.
- RebuildThread: Single `std::thread` executing `rebuildThreadLoop()`. The loop processes one task at a time. `processDeferredAdmission` and `enqueuePublicationIntentForRuntimeCommit` are called sequentially within the loop body, never concurrently.

### Verdict: ✅ R-PROD-4 PASS

```text
R-PROD-4: Each producer context can have at most 1 accounting reservation
  in the fetch_add → push/rollback window at any time.
```

---

## G6 — Constructive Overlap for P = 4098

### Scenario

```text
 preconditions:
   P_queue = 4096  (queue full — consumer slower than producers)

 Step 1: Message Thread enters enqueuePublicationIntent
   fetch_add(publicationIntentResidencyCount_, +1)
   ──> counter = 4097
   ──> P_accounting_reservation = 1
   scheduler preempts (before push)

 Step 2: RebuildThread enters enqueuePublicationIntent
   fetch_add(publicationIntentResidencyCount_, +1)
   ──> counter = 4098
   ──> P_accounting_reservation = 2
   scheduler preempts (before push)

 Step 3: State snapshot
   P_queue = 4096
   P_accounting_reservation = 2
   P = 4098

 Step 4: Both producers resume
   push() → queue full → returns false
   fetchSub(counter, -1)  [both rollback]
   counter = 4096
   P_accounting_reservation = 0
```

### Feasibility

- Queue full: Achievable when CoordinatorLoop (consumer) is slower than producers (e.g., heavy publish processing, RT callback interference)
- Dual preemption: Both threads can be preempted between `fetch_add` and `push` (very brief window, but exists)
- The `MpscBoundedRing::push()` CAS reservation is separate from the external accounting reservation

### Verdict: ⚠️ P = 4098 REACHABLE (constructive scenario exists, but timing-dependent)

```text
P_max = 4098 is constructive reachable under current topology
  BUT: not guaranteed — depends on scheduling interleaving
```

---

## G7 — Final Verdict

| Proposition | Verdict | Notes |
| --- | --- | --- |
| `P = P_queue + P_accounting_reservation` | ✅ PROVEN | Counter semantics (h:539-545) |
| `P_queue ≤ 4096` | ✅ PROVEN | Physical slot capacity |
| `P ≤ 4096` | ❌ INVALIDATED | Counter can exceed queue capacity |
| `R_max = producer thread count` | ❌ MISCONCEPTION | R_max = concurrent invocations in reservation window, not thread count |
| `C_prod = 2` (current caller closure) | ✅ CODE-DERIVED FACT | G1-G3 verified |
| `C_prod = 2` (structural invariant) | ❌ NOT ESTABLISHED | No type-system enforcement; future code change could add producer |
| `R-PROD-1: caller closure` | ✅ PASS | {Message Thread, RebuildThread} |
| `R-PROD-2: single-threaded context` | ✅ PASS | C_msg=1, C_rebuild=1 |
| `R-PROD-3: no reentrancy` | ✅ PASS | Audited through push() internals |
| `R-PROD-4: single reservation per context` | ✅ PASS | Derived from R-PROD-2 + R-PROD-3 |
| `R_max ≤ 2` | ⚠️ CONDITIONAL | Under R-PROD-1..4 (code-audit-derived) |
| `P_max ≤ 4096 + 2 = 4098` | ⚠️ CONDITIONAL UPPER BOUND | Under current production topology |
| `P_max = 4098` | ❌ NOT ESTABLISHED | Equality requires structural invariant maintenance |

### Final conclusion

```text
P_queue_max = 4096                         PROVEN

R_max ≤ 2                                   CONDITIONAL (R-PROD-1..4 verified for current code)

P_max ≤ 4096 + 2 = 4098                     CONDITIONAL UPPER BOUND
  (current-code fact, not structural invariant)

P_max = 4098                                REACHABLE (constructive overlap demonstrated)
  but NOT a production invariant
```

> **`R_max ≤ 2` は「R-PROD コメントを追加したから証明される」のではなく、**
> **現行 production caller closure、各 producer context の single-thread execution、**
> **および same-context non-reentrancy から導出される application-level invariant である。**
> **これらを architecture invariant として維持・検証可能な形に固定するまで、**
> **P_max ≤ 4098 は conditional upper bound とする。**
