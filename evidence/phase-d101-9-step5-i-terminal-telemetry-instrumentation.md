# Phase D101-9 Step 5-I: Terminal Telemetry Instrumentation Contract

## Status: COMPLETE ✅ (Step 5-I + Step 5-II)

## Date
2026-08-15

## Scope
Implement Terminal Telemetry Instrumentation Contract for Candidate B (shutdown-only bounded Terminal).
**Explicitly NOT** selecting `K_terminal` value or modifying HealthMonitor thresholds.

## Changes

### ISRRetireRouter.h
1. **4 new telemetry members** added to `TerminalReclaimAuthority`:
   - `terminalPeakResident_` (`std::atomic<uint32_t>`) — peak Terminal resident count (lock-free observable)
   - `terminalStoreCount_` (`std::atomic<uint64_t>`) — cumulative store() entry count (Generic + World)
   - `terminalDrainAllCount_` (`std::atomic<uint64_t>`) — cumulative drainAll() invocations
   - `terminalDrainEntryCount_` (`std::atomic<uint64_t>`) — cumulative entries drained by drainAll()

2. **4 new API methods** (all using `convo::consumeAtomic`):
   - `terminalPeakResident()`
   - `terminalStoreCount()`
   - `terminalDrainAllCount()`
   - `terminalDrainEntryCount()`

3. **AtomicAccess wrapper fixes** in inline methods:
   - `recordWorldReclaim()`: `++reclaimCount_` → `convo::fetchAddAtomic(reclaimCount_, ...)`

### ISRRetireRouter.cpp
1. **`store()`**: Fixed `fetch_add` → `convo::fetchAddAtomic`, added `terminalStoreCount_` increment, added peak update (CAS loop on `residentAtomic_`).
2. **`drain()`**: Fixed `fetch_sub` → `convo::fetchSubAtomic`, fixed `++reclaimCount_` → `convo::fetchAddAtomic`.
3. **`drainAll()`**: Fixed `store(0)` → `convo::publishAtomic`, fixed `++reclaimCount_` → `convo::fetchAddAtomic`, added `terminalDrainAllCount_` increment at entry, added `terminalDrainEntryCount_` per drained entry.

### CMakeLists.txt
- Added `TerminalTelemetryContractTests` target with full build config (include dirs, compile features, MKL, /utf-8, /EHsc, IPO off).

### TerminalTelemetryContractTests.cpp
- 6 unit tests: T-5.1 (initial state), T-5.2 (3 stores), T-5.3 (drain), T-5.4 (peak monotonicity), T-5.5 (drainAll), T-5.6 (Generic/World separation).

## Build Verification
- ✅ CMake reconfigure succeeded
- ✅ Build: `TerminalTelemetryContractTests.exe` linked successfully (148/148)
- ✅ All 6 tests pass (exit code 0)
- ✅ `RetireGraceSemanticsTests` — no regression (exit code 0)
- ✅ `ShutdownRetireIntentDrainTests` — no regression (exit code 0)
- ✅ `StuckReaderFallbackDrainTests` — no regression (exit code 0)

## Step 5-II Completion: Telemetry Snapshot Integration
- Added 5 new fields to `RuntimeBackpressureTelemetry` struct in `AudioEngine.h`:
  - `terminalStoreCount` (uint64_t)
  - `terminalDrainAllCount` (uint64_t)
  - `terminalDrainEntryCount` (uint64_t)
  - `terminalPeakResident` (uint32_t)
  - `terminalResident` (size_t — current snapshot)
- Added 4 wrapper methods to `ISRRetireRouter` delegating to `m_terminalReclaim`
- `getRuntimeBackpressureTelemetry()` now populates all 5 terminal telemetry fields
- ConfoPeq library builds cleanly (ninja: "no work to do")
