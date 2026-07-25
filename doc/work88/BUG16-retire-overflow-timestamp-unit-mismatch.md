# BUG 16 (High) — Retire Overflow Timestamp: Nanoseconds Stored, Microseconds Read

**Severity:** High  
**Category:** Unit mismatch / dead code  
**Files:**
- `src/audioengine/ISRRetire.cpp:55-57` (producer — stores nanoseconds)
- `src/audioengine/ISRRuntimePublicationCoordinator.cpp:279-284` (consumer — reads as microseconds)
- `src/audioengine/ISRRetireOverflowRing.h:45` (field `overflowTimestampUs`)

## The Bug

`RetireOverflowEntry::overflowTimestampUs` (note `Us` = microseconds) at `ISRRetireOverflowRing.h:45` is populated at `ISRRetire.cpp:56` with the **raw count** of `steady_clock::now().time_since_epoch().count()`.

On MSVC, `steady_clock::duration` is `std::chrono::nanoseconds` (rep = `int64_t`), so `.count()` returns **nanoseconds** (~10^9–10^12).

But the consumer at `ISRRuntimePublicationCoordinator.cpp:279-284` computes `nowUs` via `duration_cast<microseconds>`, giving **microseconds** (~10^6–10^9):

```cpp
// Producer (ISRRetire.cpp:55-57): stores NANOSECONDS
RetireOverflowEntry entry{
    localIntent,
    static_cast<uint64_t>(std::chrono::steady_clock::now()
        .time_since_epoch().count()),  // ← nanoseconds!
    0};

// Consumer (ISRRuntimePublicationCoordinator.cpp:279-284): reads as MICROSECONDS
const uint64_t nowUs = static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
if (entry.overflowTimestampUs > 0 && nowUs > entry.overflowTimestampUs)
    // μs ~10^6 vs ns ~10^9 → ALWAYS FALSE for many years after boot
```

## Impact

The comparison `nowUs > entry.overflowTimestampUs` is **effectively always false** for the first ~292 years of uptime (until `steady_clock` nanoseconds reach ~10^15, which is when μs values also reach ~10^15).

The entire `overflowAgeWarnCallback_` codepath is **dead** — stale overflow entries are never detected for age-based remediation.

## Fix

Consistently use the same unit (microseconds) at both producer and consumer:

```cpp
// Producer (ISRRetire.cpp:56):
static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count())

// Or rename field to overflowTimestampNs and fix the consumer comparison.
```

## Status

- [x] Confirmed by reading source
- [x] Unit mismatch identified
- [x] Impact: overflow age detection is dead code
- [ ] Fix applied

## Discovery

Found during systematic audit of `std::chrono` usage across the codebase.
