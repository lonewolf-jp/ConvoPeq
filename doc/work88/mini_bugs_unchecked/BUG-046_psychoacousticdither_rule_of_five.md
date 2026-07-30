# BUG-046: PsychoacousticDither — Rule of Five violation (raw owning pointer `shaperStateBuffer`)

**Severity:** Low  
**Category:** Rule of Five / Resource leak  
**File:** `src/PsychoacousticDither.h`  
**Lines:** 55–587  

## Summary

`convo::PsychoacousticDither` owns a raw `double* shaperStateBuffer` (line 580) allocated via `convo::makeAlignedArray<double>(...).release()` in the constructor (line 140) and freed in the destructor (line 98–100). Copy operations are deleted (lines 102–103), but move operations are **not** declared. Because the destructor is user-declared (line 98), the compiler will not implicitly generate move operations (C++11 Rule of Five). Any code path that inadvertently moves or returns a `PsychoacousticDither` by value will silently fall back to copy (which is already deleted), resulting in a compile error. However, if a future maintainer adds an explicit `= default` or otherwise enables move, the implicitly-generated move would copy the raw pointer, causing a **double-free** crash.

## Details

```cpp
class PsychoacousticDither {
public:
    ~PsychoacousticDither() {                                       // user-declared → no implicit move
        if (shaperStateBuffer) convo::aligned_free(shaperStateBuffer);
    }

    PsychoacousticDither(const PsychoacousticDither &) = delete;    // copy deleted
    PsychoacousticDither & operator=(const PsychoacousticDither &) = delete;

    // ⚠ No move constructor / move assignment declared

private:
    double* shaperStateBuffer = nullptr;  // [A] raw owning pointer
    VSLStream rng[MAX_CHANNELS];          // [B] VSLStream has deleted copy, proper dtor
};
```

The `VSLStream` member (lines 69–96) is correctly handled (deleted copy, proper destructor), so move-emission would also be blocked by this member. The hazard is narrow: today the code won't compile if moved. But the asymmetry between `shaperStateBuffer` (raw owning pointer, manual dtor) and the rest of the class's modern C++ design is a maintenance trap.

## Impact

- **Low**: No runtime crash today (move attempts are blocked by `VSLStream`'s deleted copy).
- **High** if a future edit enables move (e.g., wrapping `VSLStream` in a movable RAII helper without also adding `PsychoacousticDither` move operations): double-free from `shaperStateBuffer` on move.

## Suggested Fix

Declare `= default` move operations:

```cpp
PsychoacousticDither(PsychoacousticDither&&) noexcept = default;
PsychoacousticDither& operator=(PsychoacousticDither&&) noexcept = default;
```

This requires `VSLStream` to also be movable (or store `shaperStateBuffer` via `std::unique_ptr<double, AlignedFreeDeleter>`).
