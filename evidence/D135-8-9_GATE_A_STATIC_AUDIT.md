# D135-8 Step 9 — Gate A (Static Audit)  
**Mode:** read-only static audit · **production source変更 0** during audit · **build / CTest 0**  
**Scope:** cumulative Steps 1–9 state on disk (HEAD baseline f39fcd3 + 5 Step-edit files).  
**Method:** grep-first (`serena.search_for_pattern`) for authoritative line numbers, then targeted `Read` of every referenced block. A stale cached `Read` of `RebuildDispatch.cpp:895-912` (showing the ISR-Builder comment at 902) was DISCARDED in favor of the grep result (Step-9 drain comment at 901, call at 905); grep line numbers are treated as authoritative throughout.

---

## §0 Executive verdict

**Gate A = PASS** (after applying the Step-9 correction documented in §2).

One defect in the original Step-9 Edit was caught by Gate A itself and has been corrected:
`RuntimePublicationOrchestrator.cpp:580` used a **raw** `deferredClearRequested_.store(...)` instead of the project-mandated `convo::publishAtomic(...)` wrapper. Corrected to
`convo::publishAtomic(deferredClearRequested_, true, std::memory_order_release);`.
Post-correction re-verification (§2) shows **zero** raw `.store/.load/.exchange/.compare_exchange` on the latch.

Gate criterion satisfied:
> *Steps 1–9の変更後、deferred publicationのretry accountingとdeferred-clear ownershipが、RebuildThreadを唯一の通常実行主体として成立している。*

Hold for **Gate B (compile)** — pending explicit user GO.

---

## §1 Findings reference (grep-verified)

| Symbol | Def | Writers | Readers | Context |
|---|---|---|---|---|
| `clearDeferredForShutdown` | Orch.cpp:533 | (none — it IS the writer) | caller sites below | assert-free primitive |
| `requestDeferredClear` | Orch.cpp:567 | Timer.cpp:1641 (C2), 1808 (C3), 1828 (C4) | — | 3 callers → latches |
| `drainDeferredClearIfRequested` | Orch.cpp:587 | sole caller RebuildDispatch.cpp:905 | — | RebuildThread-only (`jassert` :590) |
| `deferredClearRequested_` | Orch.h:295 | `convo::publishAtomic` Orch.cpp:579 | `convo::exchangeAtomic` Orch.cpp:590 | predicate-ineligible |
| `rebuildCV` predicate | RebuildDispatch.cpp:~854 | — | Orch.cpp:567/590 (NOT predicate) | `hasPendingTask || publishRetryReady || recoveryPending || rebuildThreadShouldExit` |

`clearDeferredForShutdown` active call sites (bodies only):
- `AudioEngine.Processing.ReleaseResources.cpp:358` — EmergencyDrain (after `stopRebuildThread()` join → post-join, stop-the-world) ✓
- `RuntimePublicationOrchestrator.cpp:573` — C1 synchronous fallback (only when `rebuildThreadShouldExit` is set; no concurrent RebuildThread) ✓
- `RuntimePublicationOrchestrator.cpp:592` — inside `drainDeferredClearIfRequested` (RebuildThread, `jassert` :590) ✓

Timer.cpp direct calls of `clearDeferredForShutdown` → **0** (all 3 converted to `requestDeferredClear`). `PublicationAdmission.cpp:67` is a **comment**, not a call.

---

## §2 The one finding (#7) + correction  ✅ RESOLVED

**FINDING:** `RuntimePublicationOrchestrator.cpp` writer of `deferredClearRequested_` was `deferredClearRequested_.store(true, std::memory_order_release);` — a **raw member-function** store.

**CONVENTION:** `AtomicAccess.h:51` declares `publishAtomic(dst, value, order=release)` as the publication primitive; every other deferred atomic write (`hasDeferred_` :544/:636, `lastRecoveryPublishSeq_` :550) uses `convo::publishAtomic`. Raw `.store/.load/.exchange` are forbidden in new code.

**FIX (applied):**
```cpp
// before  : deferredClearRequested_.store(true, std::memory_order_release);          // :580
// after   : convo::publishAtomic(deferredClearRequested_, true, std::memory_order_release);
```

**RE-VERIFY (grep, read-only):**
- `deferredClearRequested_.(store|load|exchange|compare_exchange)` across `src/` → **0 matches** (empty).
- Full `deferredClearRequested_` census: `Orch.h:295` decl → `Orch.cpp:579` `convo::publishAtomic` (sole writer) → `Orch.cpp:590` `convo::exchangeAtomic` (sole reader). Predicate-ineligible by design (h:293-296).

---

## §3 Gate A checklist (10 items) — all PASS

1. **clearDeferredForShutdown caller census.** 3 active body-site callers (EmergencyDrain :358, C1 fallback :573, drain :592); **0** in Timer.cpp; h:196 is the declaration only. → **PASS**
2. **Plain-member ownership (no non-RebuildThread mutation).** Writers of `deferredRetryGeneration_`/:`deferredRetryCount_` only at `enqueueDeferred`/retry-accounting cpp:473/475/476 (RebuildThread, called via `processDeferredAdmission:702 → submitPublishRequest → enqueueDeferred`) and `clearDeferredForShutdown` cpp:547/548/543 (safe contexts only). `deferredSlot_` writers confined to Orchestrator methods reached via the same call chain (Commit.cpp:821 `enqueuePublicationIntentForRuntimeCommit` is the rebuild-completion handler — RebuildThread, per `AudioEngine.Threading.cpp:274`) plus the safe `clearDeferredForShutdown` sites. `hasDeferred_` is atomic (`publishAtomic`/`consumeAtomic`). The old bug (Timer/RecoveryWorker → `clearDeferredForShutdown` mutating plain members) is **eliminated**. → **PASS**
3. **C1 fallback ordering.** `requestDeferredClear`: `consumeAtomic(rebuildThreadShouldExit)` :572 → if set, synchronous `clearDeferredForShutdown()` + return (:573-576); else `publishAtomic(latch)` + `notify_one` (:579-581). No lost clear; no notify into a stopping thread. → **PASS**
4. **Latch discipline.** Writer `convo::publishAtomic` :579 (release); reader `convo::exchangeAtomic` :590 (acq_rel); **NOT** in `rebuildCV` predicate (h:293-296). Persistent latch drained on every wake → no lost wake / starved clear. *(Originally FAIL via #7 raw store; now PASS after correction.)* → **PASS**
5. **Drain position in rebuild loop.** Comment :901-903; `if (runtimeOrchestrator_ != nullptr)` :904; `drainDeferredClearIfRequested()` :905 — placed **after** lock release (~:900) and **before** `if (doDeferredPublish)` :908. Drained on every wake → predicate-ineligible latch cannot be starved. → **PASS**
6. **Deviation re-judgment (§0 of the post-Edit audit).** The ownership `jassert` is placed on the RebuildThread-only consumer `drainDeferredClearIfRequested` (:590), **not** on `clearDeferredForShutdown` (:534, assert-free). Rationale valid: `clearDeferredForShutdown` is also reached synchronously from EmergencyDrain post-`join()` (ReleaseResources.cpp:358) and the C1 fallback — a hard `jassert(rebuildThreadId)` there would fire in the live shutdown path (spec #1 vs #5 conflict). Mirrors the existing `peekDeferred` (:601) and `resetDeferredRetryBudget` (h:171-175) ownership idiom. → **PASS (Deviation VALID)**
7. **Atomic-API convention.** All Step-9 atomic accesses use `convo::` wrappers: `consumeAtomic(rebuildThreadShouldExit)` :572, `publishAtomic(deferredClearRequested_, …)` :579 (post-fix), `exchangeAtomic(deferredClearRequested_, false, acq_rel)` :590. Zero raw `.store/.load/.exchange`. → **PASS**
8. **Reset-before-peek.** `processDeferredAdmission(:673)`: `jassert(rebuildThread)` :675; `if (wasRecoveryWake) resetDeferredRetryBudget();` :687-688 **before** `consumeAtomic(hasDeferred_)` :689 and `peekDeferred()` :692 and `evaluateDeferred`/resubmit :696-703. → **PASS**
9. **Steps 1-8 regression.** `recoveryRetryReady` (decl `AudioEngine.h:2721`; release-stamp `AudioEngine.Timer.cpp:1748` under `rebuildMutex`; consume `AudioEngine.RebuildDispatch.cpp:888` via `exchangeAtomic`); `wasRecoveryWake` parameter (`processDeferredAdmission:673`); `publishRetryReady` (`AudioEngine.h:2718`). No regressions. → **PASS**
10. **D135-2 retry-accounting semantics.** `enqueueDeferred` retry block cpp:470-503: `:472` identity key `req.generation == deferredRetryGeneration_`; `:473` same-obligation `++deferredRetryCount_` (ordinary retry increment); `:475-476` new obligation resets gen+count (recovery path — reset already zeroed at :688, so the new world lands in the `else`/new branch → count starts 0); `:489` `if (deferredRetryCount_ > kMaxDeferredRetries)` strict `>` exhaustion; `kMaxDeferredRetries = 2` (h:295). Semantics preserved. → **PASS**

---

## §4 Gate criterion

> *Steps 1–9の変更後、deferred publicationのretry accountingとdeferred-clear ownershipが、RebuildThreadを唯一の通常実行主体として成立している。*

- **Retry accounting:** every plain writer (`deferredRetryGeneration_`/:475/:547, `deferredRetryCount_`/:473/:475/:476/:548, `deferredSlot_`/:505/:630/:543) is reached only from RebuildThread-driven paths (`processDeferredAdmission` → `evaluateDeferred`/`enqueueDeferred`/`peekDeferred`/`finishView`) or from the single safe clear primitive. Recovery reset gates on `wasRecoveryWake` before peek; ordinary retry increments. kMax=2, `>` exhaustion. → RebuildThread is the sole normal-execution owner.
- **Deferred-clear ownership:** the 3 live Timer.cpp sites now latch (`requestDeferredClear`) instead of mutating plain members; the lone real clear mutator is `clearDeferredForShutdown`, reachable only from RebuildThread (`drainDeferredClearIfRequested` :905→:592, `jassert` :590) or post-join/stopped contexts (EmergencyDrain :358, C1 fallback :573). No concurrent RebuildThread vs non-RebuildThread plain-member writer collision remains.

**Criterion: SATISFIED.** Gate A = **PASS**.

---

## §5 Transition to Gate B

Gate A (static) complete. No further source edits required before compile — the only correction (#7) is applied and verified. Awaiting explicit user GO to proceed to **Gate B (compile, no source change)** → `Gate C (CTest) → Gate D (retry state-machine) → Gate E (coalescing) → Gate F (shutdown-ordering) → Gate G (D135-2 rerun)`.
