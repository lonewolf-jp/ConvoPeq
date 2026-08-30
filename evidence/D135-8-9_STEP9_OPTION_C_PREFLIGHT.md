# D135-8 Step 9 — Option C (atomic intent + rebuild-thread drain) Read-Only Implementation Preflight

**Scope:** Step 9 — Option C preflight ONLY. Read-only. **No production source edits. No build. No CTest.**
This document is the evidence base required before the Option C *implementation* Edit (Step 9 Edit) can be authored. It supersedes the D135-8 implementation plan for the `clearDeferredForShutdown()` → `requestDeferredClear()` migration.

**Status:** PREFLIGHT COMPLETE — see GO/NO-GO at §10.

---

## 1. Caller census of `clearDeferredForShutdown()` (point 1)

Four call sites total. Definition at `RuntimePublicationOrchestrator.cpp:534`.

| # | File:Line | Enclosing function | Thread | Shutdown phase? | Live runtime? |
|---|-----------|--------------------|--------|-----------------|---------------|
| C1 | `AudioEngine.Processing.ReleaseResources.cpp:359` | `AudioEngine::releaseResources()` (def `:34`, body `:36` = **`ASSERT_NON_RT_THREAD()`**) | non-RT (MessageThread / shutdown dispatcher) | **Yes — EmergencyDrain phase** (`shutdownRuntime_.transitionTo(EmergencyDrain)` `:349`; block entered only if `m_healthMonitor.isEmergencyDrainRequested()` `:350`) | No |
| C2 | `AudioEngine.Timer.cpp:1642` | `AudioEngine::onHealthEvent()` (def `:1586`), `EVENT_PUBLICATION_STALL` branch (`:1636-1647`) | non-RebuildThread (health-event dispatch) | No | **Yes** |
| C3 | `AudioEngine.Timer.cpp:1809` | `AudioEngine::executeRecoveryAction(Recover)` (`:1804-1810`) | non-RebuildThread (recovery-action callback, see §6) | No | **Yes** |
| C4 | `AudioEngine.Timer.cpp:1829` | `AudioEngine::executeRecoveryAction(Restore)` (`:1812-1833`), "DeferredPublicationFlush" comment `:1827` | non-RebuildThread (recovery-action callback) | No | **Yes** |

**Key fact:** Every caller is **non-RebuildThread**. The RebuildThread is the single owner of the deferred-retry plain members (see `RuntimePublicationOrchestrator.h:277-278` comment: "rebuild-thread 専用 (single-owner)"). All four callers therefore reach across the thread boundary into RebuildThread-owned state — the latent race Option C must eliminate.

The comment-only reference at `PublicationAdmission.cpp:68` is **not** a call.

---

## 2. `deferredClearRequested_` / `requestDeferredClear` existence check (point 2)

`rg -n 'deferredClearRequested|requestDeferredClear' src/ tests/ ConvoPeq.md` → **(none — confirmed absent).**

Both symbols are brand-new. `requestDeferredClear()` is the Option-C entry point to be synthesized.

---

## 3. `stopRebuildThread()` exit chain (point 3)

`AudioEngine::stopRebuildThread()` — `AudioEngine.RebuildDispatch.cpp:791-815` (full body):

1. `setShutdownPhase(ShutdownPhase::StopWorkers, ...)` — `:793`
2. `convo::publishAtomic(rebuildThreadShouldExit, true, std::memory_order_release)` — `:796` (predicate break trigger)
3. `rebuildCV.notify_all()` — `:799` (definitive wake of the sleeping RebuildThread)
4. `if (rebuildThread.joinable()) rebuildThread.join();` — `:801-802` (blocks here until RebuildThread returns)
5. `runtimePublicationBridge_.discardRecoveryRequestsOnShutdown()` — `:810`

The RebuildThread exits via the wake-predicate at the top of the loop body (`AudioEngine.RebuildDispatch.cpp:861`):

```cpp
rebuildCV.wait(lock, [this] {
    return hasPendingTask
        || publishRetryReady
        || recoveryPending
        || convo::consumeAtomic(rebuildThreadShouldExit, std::memory_order_acquire);   // :858
});
if (convo::consumeAtomic(rebuildThreadShouldExit, std::memory_order_acquire)) break;  // :861  → thread returns → join() unblocks
```

**Chain is sound:** flag(set/release) → notify_all → predicate-true-on-wake → `break` → thread fn returns → `join()` returns.

---

## 4. Ordering: `rebuildThreadShouldExit` vs `isShutdownInProgress()` in the RebuildThread loop (point 4)

`AudioEngine.RebuildDispatch.cpp` (authoritative direct read, :841-916):

```cpp
:861  if (convo::consumeAtomic(rebuildThreadShouldExit, std::memory_order_acquire)) break;   // exit-flag checked FIRST
:862  if (isShutdownInProgress())
:863  {
:864      hasPendingTask = false;
:865      pendingTask.currentDSP = nullptr;
:866      publishRetryReady = false;
:868      break;
:868  }
...
:900  }   // ← lock scope (unique_lock<std::mutex> lock(rebuildMutex) :850) releases here
:908  if (doDeferredPublish && runtimeOrchestrator_ != nullptr)
:914      runtimeOrchestrator_->processDeferredAdmission(wasRecoveryWake);
```

**Ordering is fixed:** `rebuildThreadShouldExit` break (`:861`) is evaluated **before** `isShutdownInProgress()` (`:862`), both **inside** `rebuildMutex`. The lock scope closes at `:900`; `processDeferredAdmission` is invoked **outside** the lock at `:914`.

**Implication for the drain:** any Option-C drain must be placed **after** the `:861` exit-flag break and **after** the `:862 isShutdownInProgress()` guard, so it is only reached on a *live, non-exiting* wake — exactly the safety property desired.

---

## 5. Can `RuntimePublicationOrchestrator` signal `engine_.rebuildCV`? (point 5)

**YES — via an existing friend grant.**

- `AudioEngine.h:3664` → `friend class convo::isr::RuntimePublicationOrchestrator;`
- `RuntimePublicationOrchestrator.h:290` → `AudioEngine& engine_;`
- `AudioEngine.h:2700` → `std::condition_variable rebuildCV;` (private member of `AudioEngine`)
- Therefore `engine_.rebuildCV.notify_one()` / `engine_.rebuildCV.notify_all()` compiles today through the friend relationship. Symmetrically `engine_.rebuildMutex` (`:2699`) and `engine_.rebuildThreadShouldExit` (`:2701`) are reachable.

**Caveat:** there is no existing *public* wake-helper that fits. `requestRebuild(RebuildKind)` (`AudioEngine.h:1439`) is public but semantically wrong (it carries a RebuildKind / generation bump). The Orchestrator's existing cross-thread signal path is the CoordinatorLoop's 1 ms tick setting `publishRetryReady` under `rebuildMutex` + `notify_one` (comment `AudioEngine.h:2714`) — but the Orchestrator itself never signals `rebuildCV` today.

**Implementation guidance:** `requestDeferredClear()` may elect either:
- (A) direct `engine_.rebuildCV.notify_one()` via the friend grant (minimal, consistent with how `AudioEngine.Timer.cpp:1751` already signals `rebuildCV.notify_one()` from a non-RebuildThread caller), **or**
- (B) a new thin public helper `AudioEngine::wakeRebuildThreadForDeferredClear()` that encapsulates the notify (preferred for API hygiene; no source change required in Step 9 to decide).

Both are compile-legal today. No new friend grants needed.

---

## 6. The `RecoveryAction` restore-path (and recover-path) Timer caller (point 6)

`executeRecoveryAction` is defined at `AudioEngine.Timer.cpp:1792` and is **not** the RebuildThread. Its dispatchers:
- `AudioEngine.CtorDtor.cpp:69` — a lambda `[this](convo::RecoveryAction action){ executeRecoveryAction(action); }` registered as a callback (the health-monitor/policy-engine recovery-action callback).
- Referenced (comment-only) by `RuntimeHealthMonitor.h:166` and `RuntimePolicyEngine.h:16`.

The two `clearDeferredForShutdown()` sites inside it:
- `recover` branch → `:1809` (labelled "能動的回復試行 — drain + reclaim + 滞留 publish 解除")
- `restore` branch → `:1829` (comment `:1827` "DeferredPublicationFlush", the restore-path the Step-9 spec singles out)

**Thread:** non-RebuildThread (driven by the health/policy callback on the MessageThread or health timer).
**Phase:** live runtime (not a shutdown phase).

**Option-C migration target:** both `:1809` and `:1829` become `runtimeOrchestrator_->requestDeferredClear();` (deferred to the Step-9 Edit, **not** this read-only preflight). Because `executeRecoveryAction` is non-RebuildThread, the current direct clear is a cross-thread write of RebuildThread-only plain members (see §9) — the primary correctness defect Option C fixes.

---

## 7. EmergencyDrain caller — removable? (point 7)

**NO — not removable in Option C.** `ReleaseResources.cpp:359` sits inside `releaseResources()` (`ASSERT_NON_RT_THREAD()` at `:36`), in the `EmergencyDrain` phase block (`:344-383`), which is reached **after** `stopRebuildThread()` has already run.

Call ordering in `releaseResources()`:
- `stopRebuildThread()` — `:202` (sets `rebuildThreadShouldExit`, `notify_all`, **joins** the RebuildThread).
- ... phase transitions ...
- `EmergencyDrain` block `:344` → `clearDeferredForShutdown()` at `:359`.

By the time `:359` executes, the RebuildThread is **already stopped and joined** (`:801-802`). A pure `requestDeferredClear()` (set flag + notify rebuildCV) would have **no RebuildThread alive to drain** → the deferred publish obligation would leak through shutdown.

**Design constraint for Option C:** `requestDeferredClear()` MUST degrade gracefully when the rebuild thread is stopped/already-exiting: if `rebuildThreadShouldExit` is true (or the thread is not joinable/running), fall back to a **synchronous in-place clear** (i.e., call `clearDeferredForShutdown()` inline on the caller). Otherwise set `deferredClearRequested_` + `notify_one`. This keeps EmergencyDrain correct without reverting Step 8's ownership assertion (see §8 and §9).

This is a **hard implementation constraint**, not a preflight blocker — flag it so the Step-9 Edit honors it.

---

## 8. Exact drain insertion point (point 8)

The drain must execute only on a **live, non-exiting** RebuildThread wake. The only such stable site is immediately **after the lock scope releases at `:900`** and **before** the existing `doDeferredPublish` block at `:908`:

```
:900  }                       // rebuildMutex unlocked
:901  <insert Option-C drain here — new sibling step>
:908  if (doDeferredPublish && runtimeOrchestrator_ != nullptr)
:914      runtimeOrchestrator_->processDeferredAdmission(wasRecoveryWake);
```

Rationale:
- `:861 break` guarantees the thread is not exiting (rebuildThreadShouldExit was false).
- `:862 isShutdownInProgress()` guard guarantees we are not in a shutdown break.
- Lock is released at `:900`, so the drain runs lock-free — but it must re-acquire `rebuildMutex` itself if it touches `deferredSlot_`/`hasDeferred_` (see §9). Reading the `deferredClearRequested_` flag is atomic; the slot/telemetry mutation reuses `clearDeferredForShutdown()`, which on the RebuildThread satisfies its (future) ownership jassert.

**Recommended call:** `if (runtimeOrchestrator_->drainDeferredClearIfRequested()) { /* drained this wake; skip nothing else */ }` inserted at `:901`, implemented as: `exchangeAtomic(deferredClearRequested_, false, acq_rel)` → if was true, call `clearDeferredForShutdown()` (now RebuildThread-only). Because `publishRetryReady` is cleared at `:890` on every wake, interleaving `doDeferredPublish` and a drain on the same wake is safe (drain operates on the slot, not on `publishRetryReady`).

---

## 9. Race / lost-wakeup analysis — `deferredClearRequested_` vs `rebuildThreadShouldExit` simultaneous occurrence (point 9) — **priority**

### 9a. The latent hazard today
`clearDeferredForShutdown()` (`RuntimePublicationOrchestrator.cpp:534-559`) performs, without holding `rebuildMutex`:
- `consumeAtomic(hasDeferred_)` + `deferredSlot_.reset()` + `publishAtomic(hasDeferred_, false)` — `:540-544`
- **plain** writes `deferredRetryGeneration_ = 0; deferredRetryCount_ = 0` — `:548-549` (RebuildThread-only plain members, h:277-278)
- `publishAtomic(lastRecoveryPublishSeq_, 0)` — `:550`
- telemetry record `deferredRetryGeneration_`/`deferredRetryCount_` reads at `:472-474` (the enqueue/re-defer site) — compare against a value written by a foreign thread → classic data race.

So today's 4 cross-thread callers race the RebuildThread's retry accounting (`:470-503`). Option C centralizes all mutations on the RebuildThread.

### 9b. `deferredClearRequested_` setter (producer) — proposed
```cpp
// RuntimePublicationOrchestrator.h/.cpp  — non-RebuildThread caller
void requestDeferredClear() noexcept {
    convo::publishAtomic(deferredClearRequested_, true, std::memory_order_release);
    engine_.rebuildCV.notify_one();          // friend-granted access (AudioEngine.h:3664)
}
```
If `rebuildThreadShouldExit` is already true → degrade to synchronous `clearDeferredForShutdown()` (EmergencyDrain case, §7).

### 9c. `deferredClearRequested_` consumer (RebuildThread) — proposed drain at `:901`
```cpp
// AudioEngine.RebuildDispatch.cpp, after lock release @ :900
if (runtimeOrchestrator_ != nullptr &&
    convo::exchangeAtomic(runtimeOrchestrator_->deferredClearRequested_, false, std::memory_order_acq_rel))
    runtimeOrchestrator_->clearDeferredForShutdown();   // now RebuildThread-only → safe
```
`clearDeferredForShutdown()` would gain `jassert(rebuildThreadId)` (Step-8 idiom) — but **only valid once all 4 foreign callers are converted**, otherwise the jassert fires. This is the edit-coupling.

### 9d. Simultaneous-occurrence ordering (priority)

| Scenario | Producer ordering | Consumer observes | Lost wake? | Verdict |
|---|---|---|---|---|
| Live runtime, no exit | set `deferredClearRequested_`(rel) → notify_one | wakes at `:854` predicate; `:861` false; `:862` false; reaches `:901`; `exchangeAtomic`→true → drains | **No** — flag persists until consumed; every non-exit wake reaches `:901` | ✅ Safe |
| `stopRebuildThread` racing ahead of setter | `rebuildThreadShouldExit`=true (`:796`), notify_all, join (`:801`) | `:861` true → `break` → never reaches `:901` | **No lost deferred-publish work** — this is shutdown; deferred is cleared by `:862 isShutdownInProgress()` branch and EmergencyDrain | ✅ Safe (drain redundant here) |
| Setter racing `stopRebuildThread`'s notify | setter set `deferredClearRequested_`+notify_one **after** thread already broke at `:861` | thread already returned; `join()` at `:801` completes | **No** — flag is harmless (no thread to read it); shutdown clears deferred via `:862`/`EmergencyDrain` | ✅ Safe |
| Setter set under `rebuildMutex` window | producer sets under lock + notify; consumer reads under lock at `:858`/predicate | HB established via lock release/acquire + notify | **No** | ✅ Safe |

**Memory ordering HB chain (recommended):** producer sets `deferredClearRequested_` **under `rebuildMutex`** (lock_guard, matching `publishRetryReady` write pattern at `AudioEngine.h:2714`) then `notify_one`; consumer's `rebuildCV.wait` returns holding `rebuildMutex`; predicate reads `hasPendingTask`/`publishRetryReady` etc. under lock; lock releases at `:900`; consumer does `exchangeAtomic(deferredClearRequested_, false, acq_rel)` — the `notify_one` that woke the wait establishes the happens-before to the post-wake, lock-free read. Safe.

**Lost-wakeup verdict: NONE.** `deferredClearRequested_` is a persistent atomic bool consumed at a single point (`:901`) reached by every wake that does not exit. `rebuildThreadShouldExit` is checked first (`:861`), so a draining wake can never be lost to an exit — and during shutdown the deferred publish is cleared by the existing `:862` branch + EmergencyDrain regardless.

**Residual coupling (implementation guard, not a blocker):** adding `jassert(rebuildThreadId)` to `clearDeferredForShutdown()` (to mirror Step-8's `resetDeferredRetryBudget()` idiom) is **only safe in the same Edit that converts all 4 foreign callers to `requestDeferredClear()`** and gives EmergencyDrain its degrade path. Split that ordering and a debug build trips the new assert from C1-C4.

---

## 10. Planned changed files (Option C — Step-9 **Edit**, not this read-only preflight)

Per-user constraint: Step 9 is read-only. The files below are *planned* for the subsequent implementation Edit and are listed only to make the GO decision actionable. The 3 `Timer.cpp` callers are **deferred** (conversion only; not deleted) per the user's Step-9 forbiddance on deleting them in this preflight.

| File | Planned change | Deletion? |
|------|----------------|-----------|
| `src/audioengine/RuntimePublicationOrchestrator.h` | add `std::atomic<bool> deferredClearRequested_{false}`; declare `void requestDeferredClear() noexcept;` + `bool drainDeferredClearIfRequested() noexcept;`; add `jassert(rebuildThreadId)` to `clearDeferredForShutdown()` (coupled with caller conversion) | No |
| `src/audioengine/RuntimePublicationOrchestrator.cpp` | impl `requestDeferredClear()` (degrade-to-sync when `rebuildThreadShouldExit`/thread stopped; else set flag + `engine_.rebuildCV.notify_one()`) and `drainDeferredClearIfRequested()` (exchange + `clearDeferredForShutdown()`) | No |
| `src/audioengine/AudioEngine.RebuildDispatch.cpp` | insert `drainDeferredClearIfRequested()` call at `:901` (post-lock, pre-`doDeferredPublish`) | No (insertion only) |
| `src/audioengine/AudioEngine.Timer.cpp` | **deferred** — `clearDeferredForShutdown()` → `requestDeferredClear()` at `:1642`, `:1809`, `:1829` (convert only; not delete in Step 9) | No |
| `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | EmergencyDrain `:359` keeps `clearDeferredForShutdown()` (thread stopped → synchronous path is correct) | No — intentionally retained |
| `AudioEngine.h` | optional: add small public `wakeRebuildThreadForDeferredClear()` helper (alternative to direct `engine_.rebuildCV` friend notify) | Optional |

**Files deliberately NOT changed in Step 9:** `AudioEngine.h` rebuild-thread exit logic (`:861-868` ordering stays — it is already correct); `stopRebuildThread` (`:791-815`); the RecoveryAction restore/recover branches (deferred).

---

## 10-criterion audit (PASS/FAIL)

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| A | All `clearDeferredForShutdown()` callers enumerated with thread + phase | ✅ PASS | §1 (4 callers; all non-RebuildThread) |
| B | `deferredClearRequested_` / `requestDeferredClear` confirmed absent | ✅ PASS | §2 (rg: none) |
| C | `stopRebuildThread()` exit chain traced end-to-end | ✅ PASS | §3 (`:796-802`; break at `:861`) |
| D | `rebuildThreadShouldExit` vs `isShutdownInProgress()` ordering fixed | ✅ PASS | §4 (exit-flag `:861` precedes shutdown `:862`, both under `rebuildMutex`) |
| E | `RuntimePublicationOrchestrator` can signal `engine_.rebuildCV` | ✅ PASS | §5 (friend `AudioEngine.h:3664`; `engine_` h:290; `rebuildCV` h:2700) |
| F | RecoveryAction restore-path Timer caller located + thread-verified | ✅ PASS | §6 (`:1829`; non-RebuildThread dispatcher `CtorDtor.cpp:69`) |
| G | EmergencyDrain caller ownership analysed (removable? no) | ✅ PASS | §7 (called post-`stopRebuildThread` join `@:202`; `:359`) |
| H | Exact drain insertion point identified + justified | ✅ PASS | §8 (`:901`, post-lock `:900`, pre-`doDeferredPublish` `:908`) |
| I | Race/lost-wakeup for simultaneous `deferredClearRequested_` + `rebuildThreadShouldExit` | ✅ PASS | §9 (no lost wake; degrade-path for shutdown) |
| J | Planned changed files enumerated + edit-coupling noted | ✅ PASS | §10 + §9d (jassert coupling) |

---

## 8-item report

1. **Caller census (read-only):** 4 non-RebuildThread callers — EmergencyDrain `:359`, stall-handler `:1642`, Recover `:1809`, Restore `:1829`. Each reaches RebuildThread-owned plain members (`deferredRetryGeneration_`/`deferredRetryCount_`) — latent data race today.

2. **New symbols clean slate:** `deferredClearRequested_` / `requestDeferredClear` do not exist in `src/`, `tests/`, or `ConvoPeq.md` — no collision.

3. **Exit chain verified:** `stopRebuildThread` (`:791`): release-store `rebuildThreadShouldExit` (`:796`) → `rebuildCV.notify_all` (`:799`) → `join` (`:801-802`). RebuildThread surfaces from `wait` at `:858`, breaks at `:861`.

4. **Loop ordering is exit-first:** `rebuildThreadShouldExit` break (`:861`) is evaluated **before** `isShutdownInProgress()` (`:862`), both inside `rebuildMutex`. This is the invariant the drain placement relies on.

5. **CV signal path available:** the Orchestrator already owns `engine_` (h:290) and is a friend of `AudioEngine` (h:3664), so `engine_.rebuildCV.notify_one()` is legal today; no new grants. (Optionally wrap in a public helper.)

6. **RecoveryAction restore-path caller:** `executeRecoveryAction::Restore` at `:1829` (and `::Recover` at `:1809`) — driven by the health-policy callback (`CtorDtor.cpp:69`), non-RebuildThread, live-runtime phase.

7. **EmergencyDrain is NOT convertible to pure deferred:** it runs at `:359` **after** the RebuildThread is joined at `:202`. `requestDeferredClear()` must degrade to a synchronous clear when `rebuildThreadShouldExit` is true.

8. **Drain site locked in:** insert `drainDeferredClearIfRequested()` at `:901` (after lock release `:900`, before `doDeferredPublish` `:908`). Lost-wakeup analysis: none — the flag is persistent and consumed at every non-exit wake; simultaneous `rebuildThreadShouldExit` occurrence lands only in shutdown, where the existing `:862` branch + EmergencyDrain already clear deferred.

---

## GO / NO-GO

**GO (with implementation constraints)** to proceed from Step-9 read-only preflight to the Option-C Step-9 Edit, provided the Edit enforces:

- **(C1)** `requestDeferredClear()` degrades to a synchronous `clearDeferredForShutdown()` when `rebuildThreadShouldExit` is true — this is **mandatory** because EmergencyDrain (`:359`) fires post-`stopRebuildThread`/`:202` join. Without it, deferred publish leaks through shutdown.
- **(C2)** the `jassert(std::this_thread::get_id() == engine_.rebuildThreadId())` on `clearDeferredForShutdown()` (mirroring Step-8's `resetDeferredRetryBudget()` idiom) is added **in the same Edit** that converts all 4 foreign callers to `requestDeferredClear()`. Splitting the assert from the caller conversion makes a debug build trip.
- **(C3)** the drain insertion at `:901` is placed after `:861`/`:862` breaks (exit-first ordering) and reads `deferredClearRequested_` via `exchangeAtomic(acq_rel)`; the setter sets under `rebuildMutex` (lock_guard) + `notify_one`, establishing HB.
- **(C4)** the 3 `Timer.cpp` callers (`:1642/:1809/:1829`) are **converted**, not deleted, per the user's Step-9 forbiddance on deleting them in this preflight.

No source was edited and no build/CTest was executed in producing this preflight. The next action is the Option-C Step-9 Edit implementing C1–C4, after explicit user gate.
