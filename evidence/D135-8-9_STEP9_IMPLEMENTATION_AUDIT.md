# D135-8 Step 9 — Option C (atomic intent + rebuild-thread drain) — Implementation Audit

**Step:** D135-8 Step 9 (Option C)
**Edit type:** source (single coherent Edit across 4 files; ReleaseResources.cpp intentionally NOT edited)
**Build / CTest:** NOT run in this step (per user gate; gates A–G deferred).
**Verdict:** ✅ IMPLEMENTED — all C1–C4 contract guards verified by read-only census.

## 0. Deviation (intentional, rationale-backed)

The Step-9 spec listed two literal requirements that are mutually unsatisfiable as written:

> (1) `clearDeferredForShutdown()` 冒頭に `jassert(rebuildThreadId)` を追加
> (5) EmergencyDrain (ReleaseResources.cpp:358) の `clearDeferredForShutdown()` は変更しない

`clearDeferredForShutdown()` is invoked from **non-RebuildThread** contexts after Step 9:
EmergencyDrain (ReleaseResources.cpp:358, post-`stopRebuildThread` join) and
`requestDeferredClear()`'s C1 synchronous fallback (cpp:573, non-RebuildThread when
`rebuildThreadShouldExit`). A hard `jassert(std::this_thread::get_id() == engine_.rebuildThreadId())`
on `clearDeferredForShutdown()` would therefore fire in the live shutdown path.

**Resolution chosen:** keep `clearDeferredForShutdown()` as the **assert-free synchronous primitive**
(honoring (5) literally), and place the ownership `jassert` on the **RebuildThread-only consumer**
`drainDeferredClearIfRequested()` (cpp:590). This mirrors the existing `peekDeferred()`
(cpp:601) and `resetDeferredRetryBudget()` (h:171-175) ownership-assert idiom and is structurally
safe (see §4). Verified that `peekDeferred` already uses the exact
`jassert(std::this_thread::get_id() == engine_.rebuildThreadId())` form — so the idiom is
codebase-consistent.

> Related deviation: `[[nodiscard]]` was **not** placed on `drainDeferredClearIfRequested()`
> (spec wrote it with `[[nodiscard]]`). Motive: the sole call site (RebuildDispatch.cpp:906)
> discards the bool return, matching the adjacent non-`[[nodiscard]]` helpers
> (`clearDeferredForShutdown`, `processDeferredAdmission`, `peekDeferred`-style). `-Werror`
> would otherwise turn the discard into a build break. The bool is retained for future
> test/observability use.

## 1. Edit set applied (4 files; 5 regions)

| # | File | Region | Change |
|---|------|--------|--------|
| E1 | RuntimePublicationOrchestrator.h | decl block (h:196-204) | Added `void requestDeferredClear() noexcept;` + `bool drainDeferredClearIfRequested() noexcept;` |
| E2 | RuntimePublicationOrchestrator.h | member block (h:291-296) | Added `std::atomic<bool> deferredClearRequested_{false};` |
| E3 | RuntimePublicationOrchestrator.cpp | after `clearDeferredForShutdown` (cpp:559) | Implemented `requestDeferredClear()` (C1 sync-fallback + release-store + `notify_one`) and `drainDeferredClearIfRequested()` (`jassert` + `exchangeAtomic` acq_rel + clear) |
| E4 | AudioEngine.RebuildDispatch.cpp | cpp:902-906 | Inserted drain call between lock-scope close (`}`@900) and `doDeferredPublish` block |
| E5 | AudioEngine.Timer.cpp | C2:1641, C3:1808, C4:1828 | `clearDeferredForShutdown()` → `requestDeferredClear()` (convert, NOT delete) |

**NOT edited (per spec):**
- `AudioEngine.Processing.ReleaseResources.cpp:358` (EmergencyDrain) — retains synchronous
  `clearDeferredForShutdown()`. RebuildThread already joined here → stop-the-world clear, safe.

## 2. Read-only post-Edit censuses

### (A) `clearDeferredForShutdown` caller census (whole `src/`)
- **External callers:** only `ReleaseResources.cpp:358` (EmergencyDrain). ✅ (was 4 live external; now 1 post-join)
- **`AudioEngine.Timer.cpp`:** 0 occurrences. ✅ (all 3 converted)
- **Internal callers (intentional):** `RuntimePublicationOrchestrator.cpp:573` (C1 sync-fallback), `:592` (drain on RebuildThread).
- Decl `h:196`; definition `cpp:533`. Doc-only ref `PublicationAdmission.cpp:67` (comment).

### (B) `requestDeferredClear` census
- Decl `h:203`; def `cpp:568`. External callers (the convert-not-delete sites): `Timer.cpp:1641`/`1808`/`1828` (C2/C3/C4). ✅ exactly 3.

### (C) `drainDeferredClearIfRequested` census
- Decl `h:204`; def `cpp:588`. Sole external caller: `AudioEngine.RebuildDispatch.cpp:906` (RebuildThread loop, post-lock, pre-`doDeferredPublish`). ✅ exactly 1, on RebuildThread.

### (D) `deferredClearRequested_` census (latch discipline)
- Member `h:296`. **Writer (×1):** `cpp:579` `store(true, release)`. **Reader (×1):** `cpp:590` `exchangeAtomic(..., false, acq_rel)`. ✅
- **Predicate-ineligibility:** present in `src` ONLY at `h:296`, `h:201` (comment), `cpp:577` (comment), `cpp:579` (store), `cpp:590` (exchange). **NOT** in the `rebuildCV.wait` predicate. ✅ (predicate unchanged; see §4).

### (E) Non-revival of `deferredRecoveryRearmed_`
- `deferredRecoveryRearmed_` across `src/`: **0 matches.** ✅ (dead code per D135-3, not revived).

### (F) Steps 1-8 retained (no regression)
- `recoveryRetryReady` (atomic provenance, h:2721), `publishRetryReady` (h:2718), `wasRecoveryWake`
  (outer-scope `:848`), `exchangeAtomic(recoveryRetryReady, false, acq_rel)` at RebuildDispatch.cpp:888,
  CV predicate (RebuildDispatch.cpp:854-858) all present and unmodified. ✅
- `resetDeferredRetryBudget()` (h:171-175) ownership `jassert` intact. ✅

### (G) `kMaxDeferredRetries`
- `h:289` = `2`. Matches spec/binary (no drift). ✅ (drift was resolved in Steps 1-8.)

## 3. C1–C4 contract conformance

- **C1 (EmergencyDrain synchronous degrade):** `requestDeferredClear()` checks
  `consumeAtomic(engine_.rebuildThreadShouldExit, acquire)`; if true → synchronous
  `clearDeferredForShutdown()` inline (cpp:572-575). EmergencyDrain itself is left
  untouched and already clears synchronously post-join. ✅
- **C2 (convert-not-delete × coupled assert + caller conversion):** the 3 Timer.cpp
  callers converted to `requestDeferredClear()`; no production caller deleted. The
  ownership assert is coupled to the (single) RebuildThread consumer `drainDeferredClearIfRequested()`
  rather than the shared sync primitive (rationale §0). ✅
- **C3 (drain position + `exchangeAtomic` acq_rel):** drain inserted at RebuildDispatch.cpp:906,
  immediately after lock-scope close `:900` and before `doDeferredPublish`, reached **only**
  after the exit/`:861` and shutdown `:862` breaks → always running on the RebuildThread.
  Consume via `convo::exchangeAtomic(deferredClearRequested_, false, acq_rel)` (cpp:590). ✅
- **C4 (3 Timer.cpp callers converted, not deleted; RebuildDispatch drain):** ✅.

## 4. Race / lost-wakeup analysis (read-only)

`deferredClearRequested_` is an **independent intent latch** decoupled from
`publishRetryReady`/`recoverRetryReady` (per spec: no provenance coupling):

- **No lost clear.** `notify_one` may wake a non-waiting RebuildThread (harmless). The
  latch is **persistent** and drained on **every** non-exit wake via `:906`. The RebuildThread
  only exits the loop via `:861` (`rebuildThreadShouldExit`) or `:862` (`isShutdownInProgress`),
  both of which occur before `:906` is reached — and at those points EmergencyDrain / C1
  have already cleared directly. Hence the latch is always consumed or rendered moot. ✅
- **Latency bound.** If the RebuildThread is busy (not in `wait`), the store persists and is
  drained on the next wake. During live runtime, the CoordinatorLoop 1 ms tick sets
  `publishRetryReady` + `notify_one`, so the drain is reached within ≤1 ms. ✅
- **Stop-the-world safety of `clearDeferredForShutdown`.** When ever called from a
  non-RebuildThread (EmergencyDrain post-join, or C1 fallback when `rebuildThreadShouldExit`),
  the RebuildThread is stopped/joined → no concurrent writer touches the plain members
  (`deferredRetryGeneration_`, `deferredRetryCount_`, `deferredSlot_`, `hasDeferred_`). ✅
- **Predicate integrity.** `deferredClearRequested_` is deliberately **not** in the
  `rebuildCV.wait` predicate — preserves the D135-8 Step 6/7 design (predicate stays
  `hasPendingTask || publishRetryReady || recoveryPending || rebuildThreadShouldExit`).
  No busy-spin introduced. ✅

## 5. Static API sanity (cross-ref)

- `convo::consumeAtomic` / `convo::exchangeAtomic` / `convo::publishAtomic` — all used per
  `AtomicAccess.h:68-73,142`; `exchangeAtomic` chosen (not the non-existent 3-arg `consumeAtomic`). ✔
- `engine_.rebuildThreadId()` — public accessor `AudioEngine.h:1663`. ✔
- `engine_.rebuildCV`, `engine_.rebuildThreadShouldExit` — private; access via
  `friend class convo::isr::RuntimePublicationOrchestrator;` (AudioEngine.h:3664). ✔
- `jassert` / `std::this_thread::get_id()` — available (used identically by `peekDeferred` cpp:601). ✔

## 6. Out of scope (deferred — Gates A–G)

This step is the **Edit only**. The following gates remain pending the user's explicit go:
Gates A–G = {static audit, compile, CTest, retry state-machine, coalescing, shutdown ordering, D135-2 rerun}.

## 7. File manifest (evidence)

- `evidence/D135-8-9_STEP9_OPTION_C_PREFLIGHT.md` (preflight — GO w/ C1–C4) — retained, unchanged.
- `evidence/D135-8-9_STEP9_IMPLEMENTATION_AUDIT.md` (this file) — post-Edit read-only audit.

_Generated read-only from direct Read + serena `search_for_pattern` censuses; no build executed._
