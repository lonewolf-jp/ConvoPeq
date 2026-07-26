# BUG 11 (Critical) — `activeRuntimeDSPSlot` / `fadingRuntimeDSPSlot` Non-Atomic Data Race

**Severity:** Critical  
**Category:** Data race (C++ UB)  
**File:** `src/audioengine/AudioEngine.h:1996-2030`  
**Also:** `src/audioengine/AtomicAccess.h:31` (`NonOwningPtr::get()`)

## The Bug

`activeRuntimeDSPSlot` and `fadingRuntimeDSPSlot` use `convo::NonOwningPtr<DSPCore>`, which wraps a plain `std::uintptr_t` with **`constexpr` (non-atomic) load/store**:

```cpp
// AtomicAccess.h:31-33
constexpr T* get() const noexcept                //  ← plain load
{
    return reinterpret_cast<T*>(bits);
}

// AtomicAccess.h:25-28
constexpr NonOwningPtr& operator=(T* ptr) noexcept //  ← plain store
{
    bits = static_cast<std::uintptr_t>(reinterpret_cast<std::uintptr_t>(ptr));
    return *this;
}
```

These fields are written by the **NonRT Message Thread** (`setActiveRuntimeDSP()`, `exchangeFadingRuntimeDSP()`) and read by the **RT Audio Thread** (`getActiveRuntimeDSP()` at `Latency.cpp:84`), with **zero synchronization**.

This is a textbook data race under the C++ memory model → **undefined behavior**. The compiler may:
- Tear the pointer write (reading a half-updated pointer)
- Cache the old value indefinitely in a register
- Hoist or sink the access across any code boundary

## Call Sites

### Writes (NonRT Message Thread):
| Site | Function | Called From |
|------|----------|-------------|
| `AudioEngine.h:2017` | `setActiveRuntimeDSP(value)` | `RuntimePublicationOrchestrator.cpp:65`, `DSPTransition.h:125` |
| `AudioEngine.h:2005-2006` | `exchangeFadingRuntimeDSP(oldDSP)` | `DSPTransition.h:92`, `AudioEngine.Timer.cpp:880,1002,1554` |

### Reads (RT Audio Thread):
| Site | Function | Caller |
|------|----------|--------|
| `Latency.cpp:84` | `getActiveRuntimeDSP()` | `getCurrentLatencyBreakdown()` — called from `MainWindow.cpp:1536` (UI timer, likely UI thread, not RT) |
| `AudioEngine.h:2012` | `getActiveRuntimeDSP()` | `getDiagnosticActiveUuidPair()` — called from Timer thread |

**Verification:** `getCurrentLatencyBreakdown()` is called from `MainWindow.cpp:1536` (UI thread / message thread), NOT the RT audio thread. The RT path reads the DSP state through published worlds (`RuntimePublicationCoordinator`) rather than these raw slots.

**Impact is therefore reduced** — reads and writes happen on different threads but no read occurs on the RT audio path. However:
1. All readers run concurrently with the NonRT writer (Timer thread, UI thread)
2. On ARM / non-x86, plain pointer stores are not guaranteed to be atomic
3. Compiler can still optimize the plain load to skip the read entirely

## Recommended Fix

Convert `NonOwningPtr<DSPCore>` to `std::atomic<DSPCore*>` with appropriate memory ordering:

```cpp
// AudioEngine.h
std::atomic<DSPCore*> activeRuntimeDSPSlot { nullptr };
std::atomic<DSPCore*> fadingRuntimeDSPSlot { nullptr };

inline DSPCore* exchangeFadingRuntimeDSP(DSPCore* value) noexcept
{
    DSPCore* previous = fadingRuntimeDSPSlot.exchange(value, std::memory_order_acq_rel);
    return previous;
}

inline DSPCore* getActiveRuntimeDSP() const noexcept
{
    return activeRuntimeDSPSlot.load(std::memory_order_acquire);
}

inline void setActiveRuntimeDSP(DSPCore* value) noexcept
{
    activeRuntimeDSPSlot.store(value, std::memory_order_release);
}
```

**Note:** `NonOwningPtr` cannot be trivially replaced with `std::atomic<T*>` if the sentinel pattern (`~0` for "retiring") at `DSPTransition.h:93-94` is used — but that sentinel check uses `reinterpret_cast<uintptr_t>(prevRaw) == ~static_cast<uintptr_t>(0)`, which works identically on an `std::atomic`-returned pointer.

## Discovery

Found during ABA-vulnerability audit of all `compareExchangeAtomic` pointer sites in the codebase. The ABA analysis was prompted by the existing BUG 10 methodology (lock-free queue correctness).

No existing protection — no atomic, no RCU read-side guard, no memory barrier around these accesses.

## Status

- [ ] Confirmed by reading source
- [ ] Verified RT vs NonRT threading
- [ ] Reproducer: N/A (UB under standard, manifest depends on compiler/cpu)
- [ ] Fix proposed above
