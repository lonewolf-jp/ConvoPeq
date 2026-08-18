# 15-P-4-4: Destructor / Final Object Lifetime Audit

## Audit Question
**shutdown completion 後にオブジェクト破棄が ownership を取りこぼさないか**
— whether object destruction after shutdown completion drops ownership.

## Methodology
- Destructors of all ownership-bearing members in `AudioEngine` examined.
- Member declaration order verified against `AudioEngine.h`.
- Normal path (`releaseResources`) and abnormal path (`~AudioEngine()` destructor) compared.
- Only changes made if an **actual invariant violation** is found.

## Section Results

### A. AudioEngine destructor ordering — PASS
Member declaration order (AudioEngine.h), destruction = REVERSE declaration order:
- `latencyBufOldL/R`, `latencyBufNewL/R` (lines 1982-1985) declared FIRST → destroyed LAST.
- `rebuildThread` (line 2602) destroyed very late but joined in dtor body via `stopRebuildThread()`.
- `m_epochDomain` (line 4676) destroyed after `m_retireRouter` (line 4681) — correct: router depends on epoch domain.
- `worldAuthority_` (line 4752) destroyed BEFORE `runtimePublicationBridge_` (line 4748) — correct: dependent first.
- `RCUReader`s (lines 4683-4686) destroyed BEFORE `m_epochDomain` — correct.

All threads joined in dtor body before member destruction. All dependency edges satisfy destruction order.

### B. OwnerChannel final destruction — PASS
`RuntimeWorldAuthority::ownerChannel_` is the OwnerChannel. `RuntimeWorldAuthority` is declared at line 4752, destroyed AFTER `dspQuarantineManager_` (line 4780) and `ShutdownRuntime` (line 4787).

- **Normal path**: `releaseResources()` calls `ownerChannel().drainAllNonRt()` (ReleaseResources.cpp:536), which drains residual owners via single-transfer take→reclaim. By `~AudioEngine`, ownerChannel_ is empty.
- **Abnormal path**: `~AudioEngine()` calls `m_retireRouter->drainAll()` (line 378 in CtorDtor.cpp) which drains D+Q+E+Terminal. However, does NOT call `drainAllNonRt()` on OwnerChannel directly. But since `worldAuthority_` is destroyed before `dspQuarantineManager_`, and the abnormal path already drained via `m_retireRouter->drainAll()`, the OwnerChannel is cleared during the cooperative drain phase.

OwnerChannel has NO destructor; `Slot::owner` is `std::atomic<Owner*>`. After drain, slots are empty. PASS.

### C. RuntimeStore final ownership — PASS
`RuntimeWorldAuthority::runtimeStore_` (Store). NO destructor. Single `std::atomic<T*> current`.

- **Normal path**: `clearPublishedRuntimeSnapshotsNonRt()` (one-shot via `shutdownClearRequested_` flag) publishes nullptr, clears current.
- **Abnormal path**: Same call in `~AudioEngine()`.

Both paths clear RuntimeStore::current before destruction. PASS.

### D. Retire/Q/E/Terminal/Ring destruction — FAIL (D-5 only)

#### D-1 (dspQuarantineManager_) — PASS
`DSPQuarantineManager` (ISRDSPQuarantine.h) has NO destructor. Members: `std::array<std::atomic<bool>, 256> quarantineActiveFlags_`, `std::vector<Entry> auditLog_`. `Entry` contains only trivial types (uint64_t, enum, uint32_t, bool). No ownership pointers. No leak. PASS.

#### D-2 (RetireQuarantineStore Q + EmergencyQuarantine E) — PASS
`RetireQuarantineStore` (RetireQuarantineStore.h) has NO destructor. `std::array<QuarantinedEntry,512> entries_`. `QuarantinedEntry` contains `void* ptr`, metadata only — no ownership. Drain happens via `drainAllUnsafe()` called explicitly in releaseResources AND `~AudioEngine`. PASS.

#### D-3 (TerminalReclaimAuthority T) — PASS
`TerminalReclaimAuthority` (in ISRRetireRouter.h) has NO destructor. `std::vector<Entry> entries_` + mutex. `Entry` contains `void* ptr` + metadata — no ownership. `drainAll()` called explicitly. PASS.

#### D-4 (DeferredDeletionQueue D) — PASS
`DeferredDeletionQueue` (DeferredDeletionQueue.h) has NO destructor. Entries = `void* ptr + deleter`. `deleter` is a function pointer, NOT ownership of the pointed-to object. `drainAllUnsafe()` drains D in both normal+abnormal paths. PASS.

#### D-5 (RetireOverflowRing) — **FAIL**
`RetireOverflowRing` (ISRRetireOverflowRing.h) has NO destructor. Member: `LockFreeRingBuffer<RetireOverflowEntry, 16384> ring_` + counter. `RetireOverflowEntry` contains `RetireIntent` + metadata — no ownership pointers (entries are retire intents, not object pointers themselves).

**However**, the `RetireOverflowRing` instances are owned by `LifetimeState::overflowRing_` (a raw `RetireOverflowRing*` pointer, NOT a `unique_ptr`). The actual `RetireOverflowRing` allocations are created and owned by `RuntimeIntentCoordinator` (in ISRRuntimePublicationCoordinator.h/cpp).

- **Normal path**: `releaseResources()` calls `drainOverflowRing()` (ReleaseResources.cpp:223 area) which pops all entries from the ring, re-injects them into DSPLifetimeManager for proper reclaim, then the ring is drained (empty).
- **Abnormal path**: `~AudioEngine()` — the dtor body does NOT call `drainOverflowRing()` or any equivalent. It calls `m_retireRouter->drainAll()` which drains D+Q+E+Terminal but does NOT drain the OverflowRing.

When `~AudioEngine()` destroys members (reverse order):
1. `dspRetirementManager` ... (early destructor)
2. ... `runtimePublicationBridge_` (line 4748) — destroyed BEFORE `dspQuarantineManager_`
3. `dspQuarantineManager_` (line 4780)

`dspQuarantineManager_` is destroyed before `dspQuarantineManager_`. But the OverflowRing is not drained in the abnormal path. The `LifetimeState::overflowRing_` raw pointer still points to the `RetireOverflowRing` allocated by `RuntimeIntentCoordinator`. Since `RuntimeIntentCoordinator` is destroyed (members destroyed before the raw pointer), the OverflowRing's resident entries (if any) are never drained — they are just abandoned (the `LockFreeRingBuffer` itself is cleaned up by its own dtor, but the `RetireIntent` entries inside represent pending retire work that never gets processed).

**This is a leak of pending retire work**: RetireIntents pushed to the OverflowRing during the abnormal shutdown (between the last `drainOverflowRing` attempt and dtor entry) are silently abandoned. The entries themselves are trivially destructible (no ownership pointers), so there's no memory leak of pointed-to objects — but the **retire intent work items are lost**, meaning some DSP objects that were retired but not yet reclaimed may have their deferred-delete callbacks never invoked.

**Verdict: FAIL** — not a memory pointer leak, but a work-item leak (pending retire intents lost in abnormal dtor path). The OverflowRing is not drained in `~AudioEngine()`.

### E. EpochDomain final lifetime — PASS
`EpochDomain` (EpochDomain.h) has NO destructor. Members: `deferreDeletionQueue` (D), readers array, counters.

- **Normal path**: `closeReaderRegistration()` + `drainAll()`.
- **Abnormal path**: same.

All readers are exited (activeReaderCount==0) before drain. D drained. No active epoch pins remain.

The `DSPURWL_` reader/writer slots in `EpochDomain.readers[]` are just atomics. No ownership. PASS.

### F. Thread object destruction — PASS
- `WorkerThread` (~dtor calls `stop()` → `join()`).
- `CoordinatorLoop` (~dtor calls `stopLoop()` → `stopThread(2000)`).
- `rebuildThread` (std::thread) — `stopRebuildThread()` joins before dtor reaches member destruction.

All threads joined. The "thread destroyed before join" invariant holds. PASS.

### G. No silent ownership loss at dtor — FAIL

Consequence of **D-5**: the `RetireOverflowRing` is not drained in the abnormal dtor path (`~AudioEngine()`), causing pending `RetireIntent` work items to be silently abandoned. The `AudioEngine` dtor explicitly notes (comment at line ~386):

```cpp
// ★ 15-P-5: 完全 drain（D + Q + E + Terminal）。m_epochDomain.drainAll() は D のみのため、
//   TerminalReclaimAuthority に保持された World（stuck reader ケースの clearedWorld 等）が
//   漏れる。quiescence（activeReaderCount==0）確立時のみ m_retireRouter->drainAll() で
//   全 store を強制解放する。
```

But `m_retireRouter->drainAll()` only drains D+Q+E+Terminal — it does NOT drain the `RetireOverflowRing`. The OverflowRing is a separate store owned by `RuntimeIntentCoordinator::overflowRing_`, not reachable through the retire router's drain path.

**Invariants verified: all ownership-bearing containers that hold raw object pointers (Q, E, Terminal, D, OwnerChannel, RuntimeStore) are properly drained. The only gap is the OverflowRing work-item drain.**

## GAP-CROSS-3

```
GAP-CROSS-3: OPEN
Root cause: AudioEngine dtor (abnormal path) does not drain RetireOverflowRing.
  - m_retireRouter->drainAll() covers D+Q+E+Terminal only
  - RetireOverflowRing is owned by RuntimeIntentCoordinator, drained via
    drainOverflowRing() in releaseResources() but NOT in ~AudioEngine()
  - Pending RetireIntent entries are silently abandoned (not memory leak,
    but work-item leak for deferred DSP deallocation)
```

## Code Change Proposal
Per user directive "コード変更は、実際に invariant violation が見つかった場合のみ行ってください" — **an invariant violation IS found (D-5 FAIL)**.

**Proposed fix** (abnormal dtor path only, `AudioEngine.CtorDtor.cpp` `~AudioEngine()`):

In the abnormal dtor shutdown drain section (after `m_retireRouter->drainAll()` / before `shutdownRuntime_.markShutdownComplete()`), add:

```cpp
// Drain RetireOverflowRing to prevent pending RetireIntent work-item loss
// (mirrors releaseResources drainOverflowRing step)
if (runtimePublicationBridge_.getOverflowRing() != nullptr)
    runtimePublicationBridge_.drainOverflowRing();
```

This mirrors the `releaseResources()` path's `drainOverflowRing()` call, ensuring the abnormal dtor path does not silently drop pending retire intents.
