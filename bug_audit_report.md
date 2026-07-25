# ConvoPeq C++ Static Analysis Bug Audit Report
**Date:** 2026-07-26  
**Scope:** `src/` — all .cpp and .h files  
**Method:** Manual pattern-based static analysis (10 categories)

---

## Summary

| Category | Total Findings | HIGH | MEDIUM | LOW |
|----------|---------------|------|--------|-----|
| P1: memset/memcpy on non-trivially-copyable | 0 | 0 | 0 | 0 |
| P2: Copy assignment missing `return *this` | 0 | 0 | 0 | 0 |
| P3: Move ctor/assign missing `noexcept` | 1 | 0 | 0 | 1 |
| P4: `operator bool()` without `explicit` | 0 | 0 | 0 | 0 |
| P5: Self-assignment not handled | 0 | 0 | 0 | 0 |
| P6: Static initializer order fiasco | 0 | 0 | 0 | 0 |
| P7: Virtual destructor missing | 0 | 0 | 0 | 0 |
| P8: Return reference to temporary | 0 | 0 | 0 | 0 |
| P9: `std::move` on const reference | 0 | 0 | 0 | 0 |
| P10: Integer overflow in loop bounds | 4 | 0 | 0 | 4 |
| **Additional: Narrowing conversions** | 28 | 0 | 0 | 28 |
| **Total** | **33** | **0** | **0** | **33** |

**No HIGH or MEDIUM risk findings discovered.** The codebase is notably well-engineered with strong C++ hygiene:
- All `operator=` overloads return `*this`
- All `operator bool()` are `explicit`
- Self-assignment guards are present
- All polymorphic base classes have virtual destructors
- Static objects are safely scoped (function-local, thread-safe C++11 init)
- All memset/memcpy targets are statically guarded as trivially copyable

---

## Detailed Findings

### FINDING-01: Pattern 3 — Move constructor missing `noexcept`

**File:** `src/eqprocessor/EQProcessor.h:304`  
**Category:** P3 — Move constructors/assignments missing `noexcept`  
**Risk:** LOW  
**Confidence:** HIGH  

**Code:**
```cpp
struct EQState
{
    // ...
    // Explicitly define the move constructor  ← line 303
    EQState(EQState&& other)                   ← line 304 — NO `noexcept`
        : bands(std::move(other.bands)),
          bandTypes(std::move(other.bandTypes)),
          bandChannelModes(std::move(other.bandChannelModes)),
          totalGainDb(other.totalGainDb),
          agcEnabled(other.agcEnabled),
          nonlinearSaturation(other.nonlinearSaturation),
          filterStructure(other.filterStructure)
    {
    }
```

**Analysis:** `EQState` contains:
- `std::array<EQBandParams, 20>` (trivially copyable — 4 floats × 20)
- `std::array<EQBandType, 20>` (trivially copyable — enum × 20)  
- `std::array<EQChannelMode, 20>` (trivially copyable — enum × 20)
- `float`, `bool`, `float`, `int` (scalars)

Since all members are trivially copyable, the "move" degrades to a copy anyway. The missing `noexcept` prevents `std::vector` move optimization on reallocation, but `EQState` is never stored in a `std::vector` — it is always heap-allocated and atomically pointer-swapped. **No observable impact in this codebase.**

**Severity rationale:** LOW because (a) all members are trivially copyable, (b) EQState is never put in a vector, (c) copy constructor is also defined and identical in effect.

---

### FINDING-02: Pattern 10 — `int` loop variable with `.size()` (signed/unsigned mismatch)

**File:** `src/DeviceSettings.cpp:1301`  
**Category:** P10 — Integer overflow / signed-unsigned mismatch  
**Risk:** LOW  
**Confidence:** MEDIUM (theoretical; would need >2B elements to manifest)

**Code:**
```cpp
for (int i = 0; i < availableTypes.size(); ++i)
```

**Analysis:** `availableTypes` is a `juce::StringArray`. If its `.size()` exceeds `INT_MAX` (2,147,483,647), `i` would overflow before reaching the end. In practice, audio device lists never approach this size, so the risk is purely theoretical.

**Files with same pattern:**
- `src/eqprocessor/EQProcessor.Core.cpp:333` — `for (int i = 1; i < tokens.size(); ++i)`
- `src/eqprocessor/EQProcessor.Core.cpp:403` — `for (int i = 1; i < tokens.size(); ++i)`
- `src/MainWindow.cpp:352` — `for (int i = 0; i < tokens.size(); ++i)`

---

### FINDING-03: Additional — Narrowing `static_cast<int>(container.size())`

**Category:** Narrowing conversion (size_t → int)  
**Risk:** LOW  
**Confidence:** MEDIUM  
**Count:** 28 sites across codebase (partial list below)

**Example:** `src/CmaEsOptimizerDynamic.cpp:91`
```cpp
const int lambda = static_cast<int>(candidates.size());
```

**Analysis:** `std::vector::size()` returns `size_t` (64-bit on x64). A static_cast to `int` silently truncates values above 2^31-1. While no current container approaches this size, patterns like this prevent the code from being safely reused with much larger datasets.

**Representative sites:**
| File | Line | Expression |
|------|------|-----------|
| `CmaEsOptimizerDynamic.cpp` | 91 | `static_cast<int>(candidates.size())` |
| `CmaEsOptimizerDynamic.cpp` | 94 | `static_cast<int>(fitness.size())` |
| `Fixed15TapNoiseShaper.h` | 382 | `static_cast<int>(PRESET_SAMPLE_RATES.size()) - 1` |
| `FixedNoiseShaper.h` | 327 | `static_cast<int>(PRESET_SAMPLE_RATES.size()) - 1` |
| `ThreadAffinityManager.h` | 206 | `static_cast<int>(topo.cores.size())` |
| `MixedPhasePersistentCache.cpp` | 177 | `static_cast<int>(rho.size())` |
| `MixedPhasePersistentCache.cpp` | 311 | `static_cast<int>(timedFiles.size())` |
| `PeakEstimator.cpp` | 22 | `static_cast<int>(samples.size()) - 1` |
| `EQResponseSampler.cpp` | 221 | `static_cast<int>(result.samples.size())` |
| `EQAnalysisUnitTests.cpp` | 234 | `static_cast<int>(samples.size()) - 1` |
| `EQBoundExcessBenchmark.cpp` | 407 | `static_cast<int>(result.samples.size())` |
| (17 more) | | |

---

## Categories with Zero Findings

### Pattern 1: memset/memcpy on non-trivially-copyable types — **CLEAN**
All sites are guarded by `static_assert(std::is_trivially_copyable_v<T>, ...)` or operate on raw `double*` / POD structs. Notable verified sites:
- `AlignedAllocation.h:159` — guarded by `static_assert(std::is_trivially_copyable_v<T>)`
- `EQProcessor.Core.cpp:259` — `filterState` is `std::array<std::array<std::array<double,2>,20>,4>` (all trivially copyable)
- `MKLNonUniformConvolver.cpp` — all on raw `double*` arrays
- `LoudnessMeter.cpp:40-41` — `KWeightingState` (struct of `double[2]` arrays)

### Pattern 2: Copy assignment operators not returning `*this` — **CLEAN**
All 11 `operator=` implementations return `*this`.

### Pattern 4: `operator bool()` without `explicit` — **CLEAN**
All 3 overloads use `explicit operator bool()`:
- `AlignedAllocation.h:88`
- `AtomicAccess.h:36`
- `ObservedRuntime.h:57`

### Pattern 5: Self-assignment not handled — **CLEAN**
All relevant `operator=` implementations check `if (this != &other)`. Notable verified:
- `AlignedAllocation.h:67`
- `ConvolverState.h:94`
- `EQProcessor.h:317,332`
- `PreparedIRState.h:55`

### Pattern 6: Static initializer order fiasco — **CLEAN**
All static objects are:
- Function-local statics (C++11 thread-safe initialization on first use)
- `static constexpr` (compile-time constants)
- POD aggregates with zero-initialization

No file-scope or class-scope non-trivial static objects with cross-translation-unit dependencies.

### Pattern 7: Virtual destructor missing — **CLEAN**
All interface/base classes checked have virtual destructors:
- `IReaderEpochProvider` — `virtual ~IReaderEpochProvider() = default;`
- `IPublicationProvider` — `virtual ~IPublicationProvider() = default;`
- `IRetireProvider` — `virtual ~IRetireProvider() = default;`
- `IRetireRouter` — `virtual ~IRetireRouter() = default;`
- `IEpochProvider` — `~IEpochProvider() override = default;`
- `SealedObject` — `virtual ~SealedObject() = default;`
- `RefCountedDeferred` — protected non-virtual (CRTP — correct pattern)
- `juce::ChangeBroadcaster` — virtual (framework)
- `juce::AudioProcessor` / `juce::AudioSource` — virtual (framework)

### Pattern 8: Return reference to temporary — **CLEAN**
All getter functions return pointers or values, not references to temporary objects.

### Pattern 9: `std::move` on const reference — **CLEAN**
No instances detected. One `return std::move(stepResult)` at `ConvolverProcessor.LoaderThread.cpp:143` is on a member variable (not const), and is a legitimate move-out of a member `LoadResult`. Not a bug.

---

## Conclusion

The ConvoPeq codebase demonstrates **strong C++ hygiene** across all 10 audited patterns. The only issues found are:

1. **1 missing `noexcept`** on a move constructor where all members are trivially copyable — LOW risk, no observable impact.
2. **4 signed/unsigned mismatches** in loop conditions — LOW risk, theoretical only.
3. **28 narrowing conversions** (`size_t` → `int`) — LOW risk, but largest category by volume.

**Recommendation:** The narrowing conversions (`static_cast<int>(.size())`) are the most actionable category with the most instances. Consider migrating to `size_t` or `auto` loop variables over time, especially in reusable algorithms like CMA-ES optimizer (`CmaEsOptimizerDynamic.cpp`) where future parameter changes could increase dimensions.
