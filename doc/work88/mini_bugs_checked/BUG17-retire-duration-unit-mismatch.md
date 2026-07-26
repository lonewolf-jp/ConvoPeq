# BUG 17 (High) — `overflowDurationMs` Is Actually Microseconds, Fires 1000× Too Early

**Severity:** High  
**Category:** Unit mismatch / false positive  
**File:** `src/audioengine/AudioEngine.Retire.cpp:134-137`

## The Bug

```cpp
const auto now = static_cast<uint64_t>(
    std::chrono::steady_clock::now().time_since_epoch().count());  // nanoseconds
const uint64_t overflowDurationMs = (now - overflowStart) / 1000;  // → microseconds (!)
chronicByDuration = (overflowDurationMs > 5000);  // comment says ">5秒" (>5 seconds)
```

- `now` and `overflowStart` are both **nanoseconds** (from `.count()`)
- Dividing by 1000 converts nanoseconds to **microseconds**, not milliseconds
- Threshold is 5000 μs = **5 ms**, but the comment and variable name say "5 seconds"

## Impact

False positive chronic overflow detection: the throttle engages after only **5 ms** of continuous overflow instead of the intended **5 seconds**. Under sustained load, this causes premature tripping of the overflow-mitigation path, unnecessarily degrading audio quality.

## Fix

```cpp
const uint64_t overflowDurationMs = (now - overflowStart) / 1'000'000;  // → milliseconds
chronicByDuration = (overflowDurationMs > 5000);  // 5000 ms = 5 seconds (matches comment)
```

Or equivalently, use `std::chrono::duration_cast` for type-safe conversion.

## Status

- [x] Confirmed by reading source
- [x] Unit mismatch identified
- [x] Impact: false positive chronic overflow detection (5 ms instead of 5 s)
- [ ] Fix applied
