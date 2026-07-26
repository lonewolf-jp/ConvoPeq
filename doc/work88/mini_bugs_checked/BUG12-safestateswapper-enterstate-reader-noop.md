# BUG 12 (Critical) — SafeStateSwapper `enterStateReader` / `exitStateReader` Are No-Ops

**Severity:** Critical  
**Category:** Dead code / use-after-free (masked)  
**File:** `src/ConvolverProcessor.h:268-269`  
**Components:** `SafeStateSwapper.h`, `DeferredFreeThread.h`

## The Bug

`enterStateReader(int)` and `exitStateReader(int)` in `ConvolverProcessor` are empty stubs:

```cpp
// ConvolverProcessor.h:268-269
void enterStateReader(int /*readerIndex*/) const noexcept {}
void exitStateReader(int /*readerIndex*/) const noexcept {}
```

These should delegate to `SafeStateSwapper::enterReader(index)` / `SafeStateSwapper::exitReader(index)`. Because they do not:

- `SafeStateSwapper::readerEpochs[]` stays all-zeros (`kIdleEpoch`)
- `getMinReaderEpoch()` never finds an active reader, always returns `globalEpoch`
- `tryReclaim()` sees `entryEpoch < minReaderEpoch` → **always true** for any retired entry
- **All retired `ConvolverState` objects are immediately reclaimable** — zero deferral

## Impact

Masked in practice because the RT audio thread reads `ConvolverState` through the global `EpochDomain`/`RCUReader` system, not through `SafeStateSwapper`. The callers that do invoke `enterStateReader`/`exitStateReader` are:

| Caller | File:Line | Reader Index |
|--------|-----------|--------------|
| `isCacheEntrySafeToDelete()` | `ConvolverProcessor.LoadPipeline.cpp:213-214` | 2 |
| `createSnapshotFromCurrentState()` | `AudioEngine.Snapshot.cpp:24-25` | 1 |

If either of these callers reads `ConvolverState` fields while the `DeferredFreeThread` (1ms sleep) runs a reclaim cycle, a concurrent `swap()` from the load pipeline could retire the state and have it freed immediately, causing **use-after-free**.

## Root Cause

Historical stubs left unwired when the `SafeStateSwapper` RCU was refactored or when `ConvolverProcessor` was extracted from `AudioEngine`.

## Fixed

```cpp
void enterStateReader(int readerIndex) const noexcept
{
    rcuSwapper.enterReader(readerIndex);
}

void exitStateReader(int readerIndex) const noexcept
{
    rcuSwapper.exitReader(readerIndex);
}
```

## Status

- [x] Confirmed by reading source
- [x] Impact assessed (masked but architecturally broken)
- [ ] Fix applied

## Discovery

Found during systematic SafeStateSwapper correctness audit triggered by plan.md Phase 2 verification.
