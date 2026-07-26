# BUG 13 (High) — SafeStateSwapper: Epoch Bump Before Pointer Swap Allows Premature Reclamation

**Severity:** High  
**Category:** Design flaw (use-after-free window)  
**File:** `src/SafeStateSwapper.h:106-109`  
**Currently masked by BUG 12 (enterStateReader no-op)**

## The Bug

The swap sequence bumps the global epoch **before** swapping the pointer:

```cpp
// SafeStateSwapper.h:106-109
const uint64_t epoch1 = convo::fetchAddAtomic(globalEpoch, 1, acq_rel);  // bump #1
/* newEpoch = */ convo::fetchAddAtomic(globalEpoch, 1, acq_rel);          // bump #2
ConvolverState* oldState = convo::exchangeAtomic(activeState, newState, acq_rel); // SWAP AFTER
```

Expected semantics: if all readers that entered *before the swap* have exited, their epoch < minReaderEpoch means "no one can still see the old pointer."

Actual semantics: a reader can enter *between* the epoch bump and the pointer swap, recording an epoch **greater** than the retired entry's epoch, yet still see the **old** pointer (because `activeState` hasn't been updated yet). The reclaimer then sees `entryEpoch < readerEpoch`, assumes the reader can't possibly see the old pointer, and frees it → **use-after-free**.

### Interleaving

| Time | Writer (swap) | Reader | Reclaimer |
|------|---------------|--------|-----------|
| t0 | `epoch1 = fetchAdd(1)` → globalEpoch = N+1 | | |
| t1 | | **enterReader** → records epoch N+1 | |
| t2 | `fetchAdd(1)` → globalEpoch = N+2 | | |
| t3 | `exchangeAtomic(activeState, newState)` | | |
| t4 | retire oldState with entryEpoch = N | | |
| t5 | | **reads activeState** → still gets **oldState** (swap at t3 not yet visible) | |
| t6 | | | `getMinReaderEpoch()` = N+1 |
| t7 | | | `tryReclaim`: entryEpoch(N) < minEpoch(N+1) → **reclaims oldState** |
| t8 | | Reader uses oldState → **UAF** | |

## Correct Fix

Swap first, then bump, and record the epoch after the swap:

```cpp
ConvolverState* oldState = convo::exchangeAtomic(activeState, newState, acq_rel);
const uint64_t epoch2 = convo::fetchAddAtomic(globalEpoch, 1, acq_rel);
if (oldState != nullptr)
    retireEntry(oldState, epoch2);  // entryEpoch = epoch2
```

Now: any reader that enters after the exchange sees `newState`. Any reader that entered before the exchange recorded epoch oldEpoch ≤ epoch2. The reclaimer checks `entryEpoch ≤ readerEpoch` — if any reader holds epoch = epoch2 (= entryEpoch), it entered *before* the swap and could still see oldState, so reclamation is deferred. When that reader exits, no one can see oldState, and it's safe to free.

## Discovery

Found during SafeStateSwapper correctness audit. The initial entry/exit stubs (BUG 12) mask this bug in practice — if enterStateReader were wired, this race would manifest.
